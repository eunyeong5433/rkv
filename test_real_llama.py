#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from dataclasses import dataclass
from typing import List, Tuple, Optional
import csv
import os

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.cache_utils import DynamicCache
from datasets import load_dataset

# ----------------------
# Utilities
# ----------------------

def get_device(dev: Optional[str] = None) -> torch.device:
    if dev is not None:
        return torch.device(dev)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def to_cpu_pinned(t: torch.Tensor) -> torch.Tensor:
    if t.numel() == 0:
        return t.detach().cpu()
    return t.detach().to("cpu", non_blocking=True).pin_memory().contiguous()


def fetch_to_gpu(t_cpu: torch.Tensor, device: torch.device) -> torch.Tensor:
    if t_cpu.numel() == 0:
        return t_cpu
    return t_cpu.to(device, non_blocking=True).contiguous()


# ----------------------
# KV slices per layer/seq
# ----------------------

@dataclass
class PerSeqKV:
    k_front: torch.Tensor
    v_front: torch.Tensor
    k_tail: torch.Tensor
    v_tail: torch.Tensor
    k_mid_cpu: torch.Tensor
    v_mid_cpu: torch.Tensor
    full_cached: bool

    def lens(self) -> Tuple[int, int, int]:
        F = 0 if self.full_cached else self.k_front.size(1)
        M = 0 if self.full_cached else (self.k_mid_cpu.size(1) if self.k_mid_cpu.numel() else 0)
        return F, M, self.k_tail.size(1)


@dataclass
class LayerKV:
    seqs: List[PerSeqKV]


class KVManager:
    """Maintain FRONT+TAIL on GPU and MID on CPU, per model (draft/target)."""

    def __init__(self,
                 past_kv: List[Tuple[torch.Tensor, torch.Tensor]],
                 front_len: int,
                 tail_cap: int,
                 device: torch.device,
                 offload_buffer: int = 16):
        self.device = device
        self.front_len = front_len
        self.tail_cap = tail_cap
        self.offload_buffer = offload_buffer
        self.layers: List[LayerKV] = []

        if not past_kv:
            return

        B = past_kv[0][0].size(0)
        cur_len = past_kv[0][0].size(-2)
        remaining = max(0, cur_len - front_len)
        tail_len = min(tail_cap, remaining)
        mid_len = max(0, remaining - tail_len)

        for (k, v) in past_kv:
            layer_seqs: List[PerSeqKV] = []
            for b in range(B):
                k_b = k[b]; v_b = v[b]
                k_front = k_b[:, :front_len, :].contiguous()
                v_front = v_b[:, :front_len, :].contiguous()
                if mid_len > 0:
                    k_mid_cpu = to_cpu_pinned(k_b[:, front_len:front_len+mid_len, :])
                    v_mid_cpu = to_cpu_pinned(v_b[:, front_len:front_len+mid_len, :])
                else:
                    k_mid_cpu = torch.empty(0, device="cpu")
                    v_mid_cpu = torch.empty(0, device="cpu")
                k_tail = k_b[:, front_len+mid_len:, :].to(self.device).contiguous()
                v_tail = v_b[:, front_len+mid_len:, :].to(self.device).contiguous()
                layer_seqs.append(
                    PerSeqKV(
                        k_front.to(self.device), v_front.to(self.device),
                        k_tail, v_tail, k_mid_cpu, v_mid_cpu,
                        full_cached=False
                    )
                )
            self.layers.append(LayerKV(layer_seqs))

    def force_full(self, seq_indices: List[int]):
        if not seq_indices:
            return
        for layer in self.layers:
            for b in seq_indices:
                s = layer.seqs[b]
                if s.full_cached:
                    continue
                if s.k_mid_cpu.numel():
                    k_mid = fetch_to_gpu(s.k_mid_cpu, self.device)
                    v_mid = fetch_to_gpu(s.v_mid_cpu, self.device)
                    s.k_tail = torch.cat([s.k_front, k_mid, s.k_tail], dim=1)
                    s.v_tail = torch.cat([s.v_front, v_mid, s.v_tail], dim=1)
                else:
                    s.k_tail = torch.cat([s.k_front, s.k_tail], dim=1)
                    s.v_tail = torch.cat([s.v_front, s.v_tail], dim=1)
                s.k_front = torch.empty_like(s.k_front[:, :0])
                s.v_front = torch.empty_like(s.v_front[:, :0])
                s.k_mid_cpu = torch.empty_like(s.k_mid_cpu[:0])
                s.v_mid_cpu = torch.empty_like(s.v_mid_cpu[:0])
                s.full_cached = True

    def append_tail_only(self, seq_idx: int, layer_idx: int,
                         k_seg: torch.Tensor, v_seg: torch.Tensor):
        if k_seg.numel() == 0:
            return
        s = self.layers[layer_idx].seqs[seq_idx]
        s.k_tail = torch.cat([s.k_tail, k_seg.contiguous()], dim=1)
        s.v_tail = torch.cat([s.v_tail, v_seg.contiguous()], dim=1)

    def enforce_cap_for_seq(self, seq_idx: int):
        for li, layer in enumerate(self.layers):
            s = layer.seqs[seq_idx]
            if s.full_cached:
                continue
            if s.k_tail.size(1) >= self.tail_cap + self.offload_buffer:
                overflow = s.k_tail.size(1) - self.tail_cap
                k_off = s.k_tail[:, :overflow, :].contiguous()
                v_off = s.v_tail[:, :overflow, :].contiguous()
                s.k_tail = s.k_tail[:, overflow:, :].contiguous()
                s.v_tail = s.v_tail[:, overflow:, :].contiguous()
                if s.k_mid_cpu.numel() == 0:
                    s.k_mid_cpu = to_cpu_pinned(k_off)
                    s.v_mid_cpu = to_cpu_pinned(v_off)
                else:
                    s.k_mid_cpu = torch.cat([s.k_mid_cpu, to_cpu_pinned(k_off)], dim=1).contiguous()
                    s.v_mid_cpu = torch.cat([s.v_mid_cpu, to_cpu_pinned(v_off)], dim=1).contiguous()

    def build_past_and_masks(self, seq_indices: List[int], new_len: int
                            ) -> Tuple[DynamicCache, torch.Tensor, torch.Tensor]:
        past = DynamicCache()
        K_layers: List[torch.Tensor] = []
        V_layers: List[torch.Tensor] = []
        lengths = self.total_seq_lengths(seq_indices)
        gpu_lengths = self.seq_lengths(seq_indices)

        if not seq_indices:
            past.key_cache, past.value_cache = [], []
            attn_mask = torch.zeros((0, new_len), device=self.device, dtype=torch.long)
            pos_ids = torch.zeros((0, new_len), device=self.device, dtype=torch.long)
            return past, attn_mask, pos_ids

        global_T_max = 0
        for li, layer in enumerate(self.layers):
            seq_tensors_k, seq_tensors_v, T_list = [], [], []
            for b in seq_indices:
                s = layer.seqs[b]
                k_cat = torch.cat([s.k_front, s.k_tail], dim=1) if not s.full_cached else s.k_tail
                v_cat = torch.cat([s.v_front, s.v_tail], dim=1) if not s.full_cached else s.v_tail
                seq_tensors_k.append(k_cat)
                seq_tensors_v.append(v_cat)
                T_list.append(k_cat.size(1))
            T_max = max(T_list) if T_list else 0

            def pad_right(t: torch.Tensor, T_max_: int) -> torch.Tensor:
                H, T, D = t.shape
                if T == T_max_: return t
                pad = torch.zeros((H, T_max_ - T, D), device=t.device, dtype=t.dtype)
                return torch.cat([t, pad], dim=1)

            K_batched = torch.stack([pad_right(t, T_max) for t in seq_tensors_k], dim=0) if T_list else torch.empty(0)
            V_batched = torch.stack([pad_right(t, T_max) for t in seq_tensors_v], dim=0) if T_list else torch.empty(0)
            K_layers.append(K_batched)
            V_layers.append(V_batched)
            if li == 0: global_T_max = T_max

        past.key_cache = K_layers
        past.value_cache = V_layers

        B_sub = len(seq_indices)
        attn_mask = torch.zeros((B_sub, global_T_max + new_len), device=self.device, dtype=torch.long)
        for i, L in enumerate(gpu_lengths):
            attn_mask[i, global_T_max - L:global_T_max] = 1
            attn_mask[i, global_T_max:] = 1

        base = torch.tensor(lengths, device=self.device).unsqueeze(1)
        step = torch.arange(new_len, device=self.device).unsqueeze(0)
        position_ids = base + step
        return past, attn_mask, position_ids

    def seq_lengths(self, idxs: List[int]) -> List[int]:
        lens = []
        for b in idxs:
            F, _, Tt = self.layers[0].seqs[b].lens()
            lens.append(F + Tt)
        return lens

    def total_seq_lengths(self, idxs: List[int]) -> List[int]:
        lens = []
        for b in idxs:
            F, M, Tt = self.layers[0].seqs[b].lens()
            lens.append(F + M + Tt)
        return lens

def slice_from_kvcache(kv_cache, row: int, start: int, end: int):
    segs = []
    for (k, v) in kv_cache:
        kseg = k[row:row+1, :, start:end, :].squeeze(0).contiguous()
        vseg = v[row:row+1, :, start:end, :].squeeze(0).contiguous()
        segs.append((kseg, vseg))
    return segs

# ----------------------
# Speculative Decoding (Option A)
# ----------------------

class SpecDecOptionA:
    def __init__(self,
                 draft_model_path: str,
                 target_model_path: str,
                 device: torch.device,
                 gamma: int = 10, front: int = 4, tail: int = 256,
                 temperature: float = 1.0, think_token: str = "</think>",
                 cot_force_full: Optional[int] = None, offload_buffer: int = 16,
                 batch_size: int = 4):
        self.device = device
        self.gamma = gamma
        self.front = front
        self.tail = tail
        self.temperature = temperature
        self.cot_force_full = cot_force_full
        self.offload_buffer = offload_buffer
        self.draft = AutoModelForCausalLM.from_pretrained(draft_model_path, torch_dtype=torch.float16).to(device).eval()
        self.target = AutoModelForCausalLM.from_pretrained(target_model_path, torch_dtype=torch.float16).to(device).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(draft_model_path, use_fast=False)
        tok_n = len(self.tokenizer)
        self.draft.resize_token_embeddings(tok_n)
        self.target.resize_token_embeddings(tok_n)
        self.EOS = self.tokenizer.eos_token_id
        self.THINK_IDS = self.tokenizer.encode(think_token, add_special_tokens=False)
        self.buffers = [""] * batch_size
        self.batch_size = batch_size

    @staticmethod
    def _find_subseq(seq: List[int], pat: List[int]) -> int:
        if not pat: return -1
        for i in range(len(seq) - len(pat) + 1):
            if seq[i:i+len(pat)] == pat: return i
        return -1

    def stream_tokens(self, seq_idx: int, token_ids: List[int]):
        if not token_ids: return
        s = self.tokenizer.decode(token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        if not s: return
        self.buffers[seq_idx] += s
        print(f"[{seq_idx}]: {self.buffers[seq_idx]}")

    @staticmethod
    def evaluate_accept(draft_tokens: torch.Tensor, target_logp: torch.Tensor,
                        draft_logp: torch.Tensor, greedy: bool) -> torch.Tensor:
        B, n = draft_tokens.size(0), draft_tokens.size(1)
        if greedy:
            gt = torch.argmax(target_logp.exp(), dim=-1)
            is_acc = (gt[:, :-1] == draft_tokens)
        else:
            batch = torch.arange(B, device=draft_tokens.device).view(-1, 1)
            pos = torch.arange(n, device=draft_tokens.device).view(1, -1)
            p_i = target_logp[batch, pos, draft_tokens]
            q_i = draft_logp[batch, pos, draft_tokens]
            r = torch.exp(p_i - q_i)
            u = torch.rand_like(r)
            is_acc = (u <= r)
        cumrej = (~is_acc).cumsum(dim=1)
        num_accepted = (cumrej < 1).sum(dim=1, keepdim=True)
        return num_accepted

    @staticmethod
    def resample_token(reject_pos: torch.Tensor, target_logp: torch.Tensor,
                       draft_logp: torch.Tensor, greedy: bool) -> torch.Tensor:
        B = draft_logp.size(0)
        V = target_logp.size(-1)
        idx = reject_pos.view(B, 1, 1).expand(B, 1, V)
        if greedy:
            probs = target_logp.gather(1, idx + 1).squeeze(1).exp()
            return torch.argmax(probs, dim=-1, keepdim=True)
        p = target_logp.gather(1, idx + 1).squeeze(1)
        q = draft_logp.gather(1, idx).squeeze(1)
        mask = p > q
        out = torch.full_like(p, float('-inf'))
        out[mask] = p[mask] + torch.log1p((-(q[mask] - p[mask]).exp()))
        dist = torch.softmax(out, dim=-1)
        return torch.multinomial(dist, num_samples=1)

    def draft_propose_gamma(self, km: KVManager, seq_indices: List[int], prev_tokens: torch.Tensor):
        Bsub = prev_tokens.size(0)
        base_lens = km.seq_lengths(seq_indices)
        past, attn_mask, pos_ids = km.build_past_and_masks(seq_indices=seq_indices, new_len=1)
        nxt = prev_tokens.to(self.device)
        toks, logps = [], []
        out = self.draft(input_ids=nxt, past_key_values=past,
                         attention_mask=attn_mask, position_ids=pos_ids, use_cache=True)
        logits_f = out.logits[:, -1].float()
        if self.temperature > 0:
            probs = torch.softmax(logits_f / self.temperature, dim=-1)
            nxt = torch.multinomial(probs, num_samples=1)
            logps.append(torch.log(probs + 1e-12))
        else:
            nxt = torch.argmax(logits_f, dim=-1, keepdim=True)
            logps.append(logits_f)
        toks.append(nxt)
        kv_cache = out.past_key_values
        attn_mask = torch.cat([attn_mask, torch.ones((Bsub, 1), device=attn_mask.device, dtype=attn_mask.dtype)], dim=1)
        for _ in range(self.gamma - 1):
            pos_ids += 1
            out = self.draft(input_ids=nxt, past_key_values=kv_cache,
                             attention_mask=attn_mask, position_ids=pos_ids, use_cache=True)
            logits_f = out.logits[:, -1].float()
            if self.temperature > 0:
                probs = torch.softmax(logits_f / self.temperature, dim=-1)
                nxt = torch.multinomial(probs, num_samples=1)
                logps.append(torch.log(probs + 1e-12))
            else:
                nxt = torch.argmax(logits_f, dim=-1, keepdim=True)
                logps.append(logits_f)
            toks.append(nxt)
            kv_cache = out.past_key_values
            attn_mask = torch.cat([attn_mask, torch.ones((Bsub, 1), device=attn_mask.device, dtype=attn_mask.dtype)], dim=1)
        draft_tokens = torch.cat(toks, dim=-1)
        draft_logp = torch.stack(logps, dim=1)
        return draft_logp, draft_tokens, kv_cache, base_lens

    def target_verify(self, km: KVManager, seq_indices: List[int], prev_tokens: torch.Tensor, draft_tokens: torch.Tensor):
        Bsub, n = draft_tokens.size(0), draft_tokens.size(1)
        base_lens = km.seq_lengths(seq_indices)
        inputs = torch.cat([prev_tokens.to(self.device), draft_tokens.to(self.device)], dim=1)
        past, attn_mask, pos_ids = km.build_past_and_masks(seq_indices=seq_indices, new_len=inputs.size(1))
        out = self.target(input_ids=inputs, past_key_values=past,
                          attention_mask=attn_mask, position_ids=pos_ids, use_cache=True)
        logits_f = out.logits[:, -(n+1):, :].float()
        target_logp = torch.log_softmax(logits_f, dim=-1)
        kv_cache = out.past_key_values
        return target_logp, kv_cache, base_lens

    @torch.inference_mode()
    def run(self, prompts: List[str], max_new_tokens: int = 512, writer: Optional[csv.DictWriter] = None, f_handle: Optional = None):
        enc = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
        input_ids = enc.input_ids.to(self.device)
        attn = enc.attention_mask.to(self.device)
        B = input_ids.size(0)
        
        out_d = self.draft(input_ids=input_ids, attention_mask=attn, use_cache=True)
        out_t = self.target(input_ids=input_ids, attention_mask=attn, use_cache=True)

        km_d = KVManager(out_d.past_key_values, self.front, self.tail, self.device, self.offload_buffer)
        km_t = KVManager(out_t.past_key_values, self.front, self.tail, self.device, self.offload_buffer)

        prev = input_ids[:, -1:]
        phase, finished, total_gen, cot_len = ["cot"] * B, [False] * B, [0] * B, [None] * B
        answer_texts = [""] * B  
        
        self.buffers = [""] * self.batch_size
        
        while not all(finished):
            idx_active = [i for i in range(B) if not finished[i]]
            if not idx_active: break

            idx_cot = [i for i in idx_active if phase[i] == "cot"]
            idx_ans = [i for i in idx_active if phase[i] == "ans"]

            if idx_ans: km_d.force_full(idx_ans)

            prev_all = prev[idx_active, :]
            q_logp_all, q_tokens_all, q_kvcache_all, base_lens_d = self.draft_propose_gamma(km_d, idx_active, prev_all)
            greedy = (self.temperature == 0)
            n_gamma = q_tokens_all.size(1)

            if idx_cot:
                cot_rows = [row for row, sidx in enumerate(idx_active) if sidx in idx_cot]
                prev_cot = prev[idx_cot, :]
                q_tokens_cot = q_tokens_all[cot_rows, :]
                q_logp_cot = q_logp_all[cot_rows, :]
                p_logp_cot, t_kvcache, base_lens_t = self.target_verify(km_t, idx_cot, prev_cot, q_tokens_cot)
                num_acc = self.evaluate_accept(q_tokens_cot, p_logp_cot, q_logp_cot, greedy)
                cotrow_to_cotj = {row: j for j, row in enumerate(cot_rows)}
            else:
                t_kvcache, num_acc, cotrow_to_cotj = None, None, {}

            for row, seq_idx in enumerate(idx_active):
                if finished[seq_idx]: continue
                
                is_cot = seq_idx in idx_cot
                toks = q_tokens_all[row, :].tolist()
                remain = max_new_tokens - total_gen[seq_idx]
                if remain <= 0:
                    finished[seq_idx] = True
                    continue
                eos_pos = toks.index(self.EOS) + 1 if self.EOS in toks else n_gamma
                upper = min(remain, eos_pos, n_gamma)

                if is_cot:
                    j = cotrow_to_cotj[row]
                    k_acc = int(num_acc[j, 0].item())
                    if k_acc < upper:
                        committed = toks[:k_acc]
                        # This is the original, correct way to call resample_token
                        r_tok = self.resample_token(num_acc[j:j+1, :],
                                                    p_logp_cot[j:j+1, ...],
                                                    q_logp_all[row:row+1, ...],
                                                    greedy)
                        committed_plus = committed + [r_tok.item()]
                        num_to_commit = k_acc + 1
                        prev_val = r_tok.item()
                    else:
                        committed = toks[:upper]
                        committed_plus = committed
                        num_to_commit = upper
                        prev_val = committed[-1] if upper > 0 else int(prev[seq_idx, -1].item())
                else:
                    committed = toks[:upper]
                    committed_plus = committed
                    num_to_commit = upper
                    prev_val = committed[-1] if upper > 0 else int(prev[seq_idx, -1].item())

                self.stream_tokens(seq_idx, committed_plus)

                if phase[seq_idx] == "ans":
                    answer_chunk = self.tokenizer.decode(committed_plus, skip_special_tokens=True)
                    answer_texts[seq_idx] += answer_chunk
                    
                if num_to_commit > 0:
                    for li, (k, v) in enumerate(slice_from_kvcache(q_kvcache_all, row, base_lens_d[row], base_lens_d[row] + num_to_commit)):
                        km_d.append_tail_only(seq_idx, li, k, v)
                    if is_cot and t_kvcache:
                        j = cotrow_to_cotj[row]
                        for li, (k, v) in enumerate(slice_from_kvcache(t_kvcache, j, base_lens_t[j], base_lens_t[j] + num_to_commit)):
                            km_t.append_tail_only(seq_idx, li, k, v)
                    km_d.enforce_cap_for_seq(seq_idx)
                    if is_cot: km_t.enforce_cap_for_seq(seq_idx)

                prev[seq_idx, 0] = prev_val
                
                total_gen[seq_idx] += len(committed_plus)
                pos = self._find_subseq(committed_plus, self.THINK_IDS)
                if phase[seq_idx] == "cot" and pos != -1:
                    cot_len[seq_idx] = total_gen[seq_idx] - (len(committed_plus) - pos)
                    phase[seq_idx] = "ans"
                    km_d.force_full([seq_idx]); km_t.force_full([seq_idx])

                if self.EOS in committed_plus or total_gen[seq_idx] >= max_new_tokens:
                    finished[seq_idx] = True
                    # --- CSV 저장 로직 ---
                    if writer and f_handle:
                        final_cot_len = cot_len[seq_idx] if cot_len[seq_idx] is not None else total_gen[seq_idx]
                        row_data = {
                            "prompt": prompts[seq_idx],
                            "cot_length": final_cot_len,
                            "total_length": total_gen[seq_idx],
                            "answer": answer_texts[seq_idx].strip()  # <-- 이 줄을 추가하세요.
                        }
                        writer.writerow(row_data)
                        f_handle.flush()
                        print(f"\n✅ Seq {seq_idx} results saved. Total: {total_gen[seq_idx]}")

# ---------------
# CLI / main
# ---------------

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draft_model", type=str, required=True)
    ap.add_argument("--target_model", type=str, required=True)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--gamma", type=int, default=10)
    ap.add_argument("--front", type=int, default=4)
    ap.add_argument("--tail", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--prompts_file", type=str, default=None, help="Path to a text file with prompts.")
    ap.add_argument("--think_token", type=str, default="</think>")
    ap.add_argument("--cot_force_full", type=int, default=None)
    ap.add_argument("--offload_buffer", type=int, default=16)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--output_csv", type=str, default="results.csv", help="Path for CSV output.")
    return ap.parse_args()

def get_prompts_from_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]

def get_math500_batches(batch_size, split="test"):
    math500 = load_dataset("HuggingFaceH4/MATH-500", split=split)
    problems = math500["problem"]
    for i in range(0, len(problems), batch_size):
        yield problems[i:i+batch_size]

def main():
    torch.manual_seed(42)
    args = parse_args()
    device = get_device(args.device)

    runner = SpecDecOptionA(
        draft_model_path=args.draft_model, target_model_path=args.target_model,
        device=device, gamma=args.gamma, front=args.front, tail=args.tail,
        temperature=args.temperature, think_token=args.think_token,
        cot_force_full=args.cot_force_full, offload_buffer=args.offload_buffer,
        batch_size=args.batch_size
    )

    RESULT_CSV = args.output_csv
    CSV_HEADER = ["prompt", "cot_length", "total_length", "answer"]
    existing_prompts = set()
    if os.path.exists(RESULT_CSV):
        try:
            with open(RESULT_CSV, 'r', newline="", encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if 'prompt' in row: existing_prompts.add(row['prompt'])
            print(f"Found {len(existing_prompts)} existing prompts. They will be skipped.")
        except Exception as e:
            print(f"Could not read existing CSV: {e}")

    with open(RESULT_CSV, "a", newline="", encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADER)
        if f.tell() == 0: writer.writeheader()

        prompt_batches = get_prompts_from_file(args.prompts_file) if args.prompts_file else get_math500_batches(args.batch_size)
        if args.prompts_file:
            all_prompts = get_prompts_from_file(args.prompts_file)
            prompt_batches = [all_prompts[i:i + args.batch_size] for i in range(0, len(all_prompts), args.batch_size)]
        else:
            prompt_batches = get_math500_batches(args.batch_size)
        
        for batch_prompts in prompt_batches:
            prompts_to_run = [p for p in batch_prompts if p not in existing_prompts]
            if not prompts_to_run:
                print("Skipping batch, all prompts already processed.")
                continue
            
            print(f"\n--- Running batch of {len(prompts_to_run)} prompts ---")
            runner.run(prompts_to_run,
                       max_new_tokens=args.max_new_tokens,
                       writer=writer,
                       f_handle=f)
    print("\nAll tasks complete.")

if __name__ == "__main__":
    main()