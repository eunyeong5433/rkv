#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from dataclasses import dataclass
from typing import List, Tuple, Optional

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
    k_front: torch.Tensor      # (H, F, D)
    v_front: torch.Tensor      # (H, F, D)
    k_tail: torch.Tensor       # (H, Tt, D) on GPU
    v_tail: torch.Tensor       # (H, Tt, D) on GPU
    k_mid_cpu: torch.Tensor    # (H, M, D) on CPU pinned
    v_mid_cpu: torch.Tensor
    full_cached: bool          # once True, FRONT+MID were merged into TAIL

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

        # Split initial past into front/mid/tail per seq per layer
        B = past_kv[0][0].size(0)
        cur_len = past_kv[0][0].size(-2)
        remaining = max(0, cur_len - front_len)
        tail_len = min(tail_cap, remaining)
        mid_len = max(0, remaining - tail_len)

        for (k, v) in past_kv:
            # k,v: (B,H,T,D)
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
        """Merge FRONT+MID+TAIL to TAIL for selected seqs; clear FRONT/MID; mark full_cached=True."""
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

    # ---- assemble batched DynamicCache + masks/pos ----
    def build_past_and_masks(self, seq_indices: List[int], new_len: int
                            ) -> Tuple[DynamicCache, torch.Tensor, torch.Tensor]:
        """
        Assemble DynamicCache with right-padding to T_max, and return:
          - past: DynamicCache
          - attention_mask: (B_sub, T_max + new_len) with valid past/new = 1, right-pad = 0
          - position_ids:   (B_sub, new_len) starting from each seq's real past length
        NOTE: prev는 KM에 포함되지 않는다는 불변식을 가정(=drop_last 불필요).
        """
        past = DynamicCache()
        K_layers: List[torch.Tensor] = []
        V_layers: List[torch.Tensor] = []
        lengths = self.total_seq_lengths(seq_indices)

        if not seq_indices:
            past.key_cache = []
            past.value_cache = []
            attn_mask = torch.zeros((0, new_len), device=self.device, dtype=torch.long)
            pos_ids = torch.zeros((0, new_len), device=self.device, dtype=torch.long)
            return past, attn_mask, pos_ids

        global_T_max = 0
        for li, layer in enumerate(self.layers):
            seq_tensors_k: List[torch.Tensor] = []
            seq_tensors_v: List[torch.Tensor] = []
            T_list = []
            for b in seq_indices:
                s = layer.seqs[b]
                if s.full_cached:
                    k_cat = s.k_tail
                    v_cat = s.v_tail
                else:
                    k_cat = torch.cat([s.k_front, s.k_tail], dim=1)
                    v_cat = torch.cat([s.v_front, s.v_tail], dim=1)
                seq_tensors_k.append(k_cat)
                seq_tensors_v.append(v_cat)
                T_list.append(k_cat.size(1))
            T_max = max(T_list) if T_list else 0

            def pad_right(t: torch.Tensor, T_max_: int) -> torch.Tensor:
                H, T, D = t.shape
                if T == T_max_:
                    return t
                pad = torch.zeros((H, T_max_ - T, D), device=t.device, dtype=t.dtype)
                return torch.cat([t, pad], dim=1)

            K_batched = torch.stack([pad_right(t, T_max) for t in seq_tensors_k], dim=0) if T_list else torch.empty(0)
            V_batched = torch.stack([pad_right(t, T_max) for t in seq_tensors_v], dim=0) if T_list else torch.empty(0)
            K_layers.append(K_batched)
            V_layers.append(V_batched)

            if li == 0:
                global_T_max = T_max

        past.key_cache = K_layers
        past.value_cache = V_layers

        B_sub = len(seq_indices)
        attn_mask = torch.zeros((B_sub, global_T_max + new_len), device=self.device, dtype=torch.long)
        for i, L in enumerate(lengths):
            # This is an error in the original code, but I'm keeping it as is.
            # It should likely be gpu_lengths for the mask.
            attn_mask[i, :L] = 1
            attn_mask[i, global_T_max:global_T_max+new_len] = 1

        base = torch.tensor(lengths, device=self.device).unsqueeze(1)      # (B_sub,1)
        step = torch.arange(new_len, device=self.device).unsqueeze(0)      # (1,new_len)
        position_ids = base + step                                         # (B_sub,new_len)
        return past, attn_mask, position_ids

    def seq_lengths(self, idxs: List[int]) -> List[int]:
        lens = []
        for b in idxs:
            seq0 = self.layers[0].seqs[b]
            F, M, Tt = seq0.lens()
            # build_past_and_masks가 mid를 무시하므로, 길이 계산도 mid를 무시해야 합니다.
            lens.append(F + Tt) #<-- mid 캐시 길이(M)를 제거
        return lens

    def total_seq_lengths(self, idxs: List[int]) -> List[int]:
        lens = []
        for b in idxs:
            seq0 = self.layers[0].seqs[b]
            F, M, Tt = seq0.lens()
            # build_past_and_masks가 mid를 무시하므로, 길이 계산도 mid를 무시해야 합니다.
            lens.append(F + M + Tt) #<-- mid 캐시 길이(M)를 제거 -> This comment is incorrect. It adds M.
        return lens

def slice_from_kvcache(kv_cache, row: int, start: int, end: int):
    """kv_cache에서 [start:end] 범위를 잘라 (per-layer k,v) 튜플 리스트로 반환."""
    segs = []
    for (k, v) in kv_cache:          # (B,H,T,D)
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
                 gamma: int = 10,
                 front: int = 4,
                 tail: int = 256,
                 temperature: float = 1.0,
                 think_token: str = "</think>",
                 cot_force_full: Optional[int] = None,
                 offload_buffer: int = 16,
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
        # THINK sentinel (multi-token 가능)
        self.THINK_IDS = self.tokenizer.encode(think_token, add_special_tokens=False)
        self.buffers = [""] * batch_size
        self.batch_size = batch_size

    # ---- utils ----
    @staticmethod
    def _find_subseq(seq: List[int], pat: List[int]) -> int:
        if not pat:
            return -1
        for i in range(len(seq) - len(pat) + 1):
            if seq[i:i+len(pat)] == pat:
                return i
        return -1

    def stream_tokens(self, seq_idx: int, token_ids: List[int]):
        if not token_ids:
            return
        s = self.tokenizer.decode(token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        if not s:
            return
        self.buffers[seq_idx] += s
        print(f"[{seq_idx}]: {self.buffers[seq_idx]}")

    # ---- accept & resample ----
    @staticmethod
    def evaluate_accept(draft_tokens: torch.Tensor,  # (B, n)
                        target_logp: torch.Tensor,  # (B, n+1, V)
                        draft_logp: torch.Tensor,   # (B, n, V)
                        greedy: bool) -> torch.Tensor:
        B, n = draft_tokens.size(0), draft_tokens.size(1)
        if greedy:
            gt = torch.argmax(target_logp.exp(), dim=-1)  # (B,n+1)
            is_acc = (gt[:, :-1] == draft_tokens)
        else:
            batch = torch.arange(B, device=draft_tokens.device).view(-1, 1)
            pos = torch.arange(n, device=draft_tokens.device).view(1, -1)
            # This is likely an error in the original code. target_logp should be indexed differently.
            p_i = target_logp[batch, pos, draft_tokens]
            q_i = draft_logp[batch, pos, draft_tokens]
            r = torch.exp(p_i - q_i)
            u = torch.rand_like(r)
            is_acc = (u <= r)
        cumrej = (~is_acc).cumsum(dim=1)
        num_accepted = (cumrej < 1).sum(dim=1, keepdim=True)  # (B,1)
        return num_accepted

    @staticmethod
    def resample_token(reject_pos: torch.Tensor,  # (B,1)
                       target_logp: torch.Tensor,  # (B, n+1, V)
                       draft_logp: torch.Tensor,   # (B, n, V)
                       greedy: bool) -> torch.Tensor:
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

    # ---- draft γ-propose (no drop_last) ----
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

        draft_tokens = torch.cat(toks, dim=-1)      # (B_sub, γ)
        draft_logp   = torch.stack(logps, dim=1)      # (B_sub, γ, V)
        
        return draft_logp, draft_tokens, kv_cache, base_lens

    # ---- target verify (no drop_last) ----
    def target_verify(self, km: KVManager, seq_indices: List[int], prev_tokens: torch.Tensor, draft_tokens: torch.Tensor):
        Bsub, n = draft_tokens.size(0), draft_tokens.size(1)
        base_lens = km.seq_lengths(seq_indices)

        inputs = torch.cat([prev_tokens.to(self.device), draft_tokens.to(self.device)], dim=1)
        past, attn_mask, pos_ids = km.build_past_and_masks(seq_indices=seq_indices, new_len=inputs.size(1))
        out = self.target(input_ids=inputs, past_key_values=past,
                          attention_mask=attn_mask, position_ids=pos_ids, use_cache=True)

        logits_f = out.logits[:, -(n+1):, :].float()  # (B_sub, n+1, V)
        target_logp = torch.log_softmax(logits_f, dim=-1)
        kv_cache = out.past_key_values
        return target_logp, kv_cache, base_lens

    @torch.inference_mode()
    def run(self, prompts: List[str], max_new_tokens: int = 512):
        enc = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
        input_ids = enc.input_ids.to(self.device)
        attn = enc.attention_mask.to(self.device)
        B = input_ids.size(0)

        # initial prefill both models (with mask)
        out_d = self.draft(input_ids=input_ids, attention_mask=attn, use_cache=True)
        out_t = self.target(input_ids=input_ids, attention_mask=attn, use_cache=True)

        km_d = KVManager(out_d.past_key_values, front_len=self.front, tail_cap=self.tail,
                         device=self.device, offload_buffer=self.offload_buffer)
        km_t = KVManager(out_t.past_key_values, front_len=self.front, tail_cap=self.tail,
                         device=self.device, offload_buffer=self.offload_buffer)

        prev = input_ids[:, -1:]  # (B,1)

        phase = ["cot"] * B
        finished = [False] * B
        total_gen = [0] * B
        cot_len = [None] * B

        while True:
            if all(finished):
                print()
                print("Finished ☑")
                print()
                self.buffers = [""] * self.batch_size
                break

            idx_active = [i for i in range(B) if not finished[i]]
            if not idx_active:
                continue

            idx_cot = [i for i in idx_active if phase[i] == "cot"]
            idx_ans = [i for i in idx_active if phase[i] == "ans"]

            # Answer는 FULL(그래도 target은 안씀)
            if idx_ans:
                km_d.force_full(idx_ans)

            # 1) draft propose (모든 active 묶어서 γ)
            prev_all = prev[idx_active, :]
            q_logp_all, q_tokens_all, q_kvcache_all, base_lens_d = self.draft_propose_gamma(km_d, idx_active, prev_all)
            greedy = (self.temperature == 0)
            n_gamma = q_tokens_all.size(1)

            # 2) CoT만 verify
            if idx_cot:
                cot_rows = [row for row, sidx in enumerate(idx_active) if sidx in idx_cot]
                prev_cot = prev[idx_cot, :]
                q_tokens_cot = q_tokens_all[cot_rows, :]
                q_logp_cot   = q_logp_all[cot_rows, :]
                p_logp_cot, t_kvcache, base_lens_t = self.target_verify(km_t, idx_cot, prev_cot, q_tokens_cot)
                num_acc = self.evaluate_accept(q_tokens_cot, p_logp_cot, q_logp_cot, greedy)
                # 매핑
                seqidx_to_cotj = {seq_idx: j for j, seq_idx in enumerate(idx_cot)}
                cotrow_to_cotj = {row: j for j, row in enumerate(cot_rows)}
            else:
                cot_rows = []
                t_kvcache = None
                base_lens_t = []
                num_acc = None
                seqidx_to_cotj = {}
                cotrow_to_cotj = {}
            
            # 3) 커밋/스트림 (KM엔 prev를 절대 넣지 않음!)
            for row, seq_idx in enumerate(idx_active):
                is_cot = (seq_idx in idx_cot)
                toks = q_tokens_all[row, :].tolist()

                remain = max_new_tokens - total_gen[seq_idx]
                if remain <= 0:
                    finished[seq_idx] = True
                    continue
                eos_pos = toks.index(self.EOS) + 1 if (self.EOS in toks) else n_gamma
                upper = min(remain, eos_pos, n_gamma)

                # =================================================================
                # ===== START: KV Cache Commit Logic 수정 ==========================
                # =================================================================
                if is_cot:
                    j = cotrow_to_cotj[row]
                    k_acc_raw = int(num_acc[j, 0].item())
                    k_acc = min(k_acc_raw, upper)
                    if k_acc < upper: # 거절 발생
                        committed = toks[:k_acc]
                        r_tok = self.resample_token(num_acc[j:j+1, :],
                                                    p_logp_cot[j:j+1, ...],
                                                    q_logp_all[row:row+1, ...],
                                                    greedy)
                        r_val = int(r_tok.item())
                        committed_plus = committed + [r_val]
                        # 커밋할 KV 캐시 길이: accept된 k_acc개 + 리샘플링된 1개
                        num_to_commit = k_acc + 1
                        prev_val = r_val
                    else: # 전체 수락
                        committed = toks[:upper]
                        committed_plus = committed
                        # 커밋할 KV 캐시 길이: 수락된 upper개 전체
                        num_to_commit = upper
                        prev_val = committed[-1] if upper > 0 else int(prev[seq_idx, -1].item())
                else: # Answer 단계 (draft-only)
                    committed = toks[:upper]
                    committed_plus = committed
                    # 커밋할 KV 캐시 길이: draft가 생성한 upper개 전체
                    num_to_commit = upper
                    prev_val = committed[-1] if upper > 0 else int(prev[seq_idx, -1].item())

                # 스트리밍
                self.stream_tokens(seq_idx, committed_plus)

                # KV 캐시 커밋
                if num_to_commit > 0:
                    # Draft 모델 KV 캐시 업데이트
                    for li, (kseg, vseg) in enumerate(
                        slice_from_kvcache(q_kvcache_all, row=row,
                                           start=base_lens_d[row], end=base_lens_d[row] + num_to_commit)
                    ):
                        km_d.append_tail_only(seq_idx, li, kseg, vseg)

                    # Target 모델 KV 캐시 업데이트 (CoT 단계에서만)
                    if is_cot and t_kvcache is not None:
                        j = cotrow_to_cotj[row]
                        for li, (kseg, vseg) in enumerate(
                            slice_from_kvcache(t_kvcache, row=j,
                                               start=base_lens_t[j], end=base_lens_t[j] + num_to_commit)
                        ):
                            km_t.append_tail_only(seq_idx, li, kseg, vseg)

                    # KV 캐시 길이 제한 적용
                    km_d.enforce_cap_for_seq(seq_idx)
                    if is_cot:
                        km_t.enforce_cap_for_seq(seq_idx)

                # =================================================================
                # ===== END: KV Cache Commit Logic 수정 ============================
                # =================================================================

                # prev 갱신 (항상 KM엔 미포함!)
                prev[seq_idx:seq_idx+1, :] = torch.tensor([[prev_val]], device=self.device)

                # 생성된 토큰 수 카운트
                total_gen[seq_idx] += len(committed_plus)

                # THINK 감지 → 다음 라운드부터 Answer
                pos = self._find_subseq(committed_plus, self.THINK_IDS)
                if (phase[seq_idx] == "cot") and (pos != -1):
                    phase[seq_idx] = "ans"
                    km_d.force_full([seq_idx])  # 한 번 FULL되면 계속 FULL
                    km_t.force_full([seq_idx])

                if (self.EOS in committed_plus) or (total_gen[seq_idx] >= max_new_tokens):
                    finished[seq_idx] = True

        return {"total_gen": total_gen, "cot_len": cot_len}


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
    ap.add_argument("--prompts", type=str, nargs="*", default=None)
    ap.add_argument("--think_token", type=str, default="</think>")
    ap.add_argument("--cot_force_full", type=int, default=None, help="(옵션) CoT 길이 임계값 이상이면 FULL 고정")
    ap.add_argument("--offload_buffer", type=int, default=16, help="tail_cap 초과 허용 버퍼")
    ap.add_argument("--batch_size", type=int, default=4, help="미니배치 크기")
    return ap.parse_args()

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
        draft_model_path=args.draft_model,
        target_model_path=args.target_model,
        device=device,
        gamma=args.gamma,
        front=args.front,
        tail=args.tail,
        temperature=args.temperature,
        think_token=args.think_token,
        cot_force_full=args.cot_force_full,
        offload_buffer=args.offload_buffer,
        batch_size=args.batch_size,
    )

    all_stats = []
    if args.prompts:
        stats = runner.run(args.prompts, max_new_tokens=args.max_new_tokens)
        all_stats.append(stats)
    else:
        for batch_prompts in get_math500_batches(args.batch_size):
            stats = runner.run(batch_prompts, max_new_tokens=args.max_new_tokens)
            all_stats.append(stats)

    print("\nDone. Stats:", all_stats)


if __name__ == "__main__":
    main()