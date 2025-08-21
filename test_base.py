import re
import torch
import warnings
import csv
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from transformers.cache_utils import DynamicCache
import time
from collections import Counter
def repeated_ngram_detected(token_ids, n=3, threshold=3):
    if len(token_ids) < n:
        return False
    ngrams = [tuple(token_ids[i:i+n]) for i in range(len(token_ids)-n+1)]
    counts = Counter(ngrams)
    return any(v >= threshold for v in counts.values())

def detect_final_answer_repetition(sentences, keyword="Final Answer", threshold=5):
    
    count = sum(1 for s in sentences if keyword in s)

    if (count >= threshold):
        print(count, sentences)
    return count >= threshold
# ---------------- Hyper‑parameters ---------------- #
# MODEL  = "/data01/huggingface_model_weights/r1-qwen-32b/"
MODEL = "../DeepSeek-R1-Distill-Llama-8B/"
DEVICE = torch.device("cuda")
FRONT  = 4       # tokens kept at the very beginning
TAIL   = 256     # rolling window on GPU
STEP   = 16      # how often to off‑load
NUM    = 300
RESULT_CSV = "test.csv"
# -------------------------------------------------- #
MAX_RECENT_SENTENCES = 10

warnings.filterwarnings("ignore")

tokenizer = AutoTokenizer.from_pretrained(MODEL, use_fast=False)
model     = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16).to(DEVICE).eval()

EOS      = tokenizer.eos_token_id
THINK_ID = tokenizer.convert_tokens_to_ids("</think>")

def compute_redundancy_score(key_cache):
    similarity_scores = []
    for k in key_cache:
        B, H, T, D = k.shape
        k = k.squeeze(0)  # (H, T, D)

        for h in range(H):
            k_h = k[h]  # (T, D)
            k_norm = k_h / (k_h.norm(dim=-1, keepdim=True) + 1e-8)
            sim_matrix = k_norm @ k_norm.T  # (T, T)
            sim_matrix.fill_diagonal_(0)

            avg_sim = sim_matrix.mean(dim=0)  # (T,)
            score = torch.softmax(avg_sim, dim=0)  # normalize
            similarity_scores.append(score)

    all_scores = torch.stack(similarity_scores)  # (H*num_layers, T)
    mean_redundancy = all_scores.mean().item()
    print(mean_redundancy)

    return mean_redundancy

def redundancy_trigger(gpu_tail, thr=0.30):
    sims = []
    for k_t, _ in gpu_tail:            # tail-only
        k = k_t.squeeze(0)             # (H, T, D)
        k = k / (k.norm(dim=-1, keepdim=True) + 1e-8)
        sims.append((k @ k.transpose(-1,-2)).mean().item())
    print(sum(sims)/len(sims))
    return (sum(sims)/len(sims)) > thr

def offload_to_cpu_async(t):
    return t.contiguous().to("cpu", non_blocking=True).pin_memory()

def fetch_to_gpu_async(t_cpu):
    return t_cpu.to(DEVICE, non_blocking=True).contiguous()

@torch.inference_mode()
def run(prompt: str, prune: bool, tag: str, full: bool, ans_full: bool, early_stop: bool):
    torch.manual_seed(42)

    full_cache_hold_count = 0
    FULL_HOLD_STEPS = 0
    gpu_mid = None  

    """
    * prune = False → vanilla full‑cache baseline
    * prune = True  → FRONT + TAIL on GPU, middle on CPU, full‑cache only
                     for the very first token of every new sentence
    """
    print(f"\n[{tag}] {prompt.strip()}")

    # -------- Prefill -------------------------------------------------- #
    in_ids  = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    out     = model(input_ids=in_ids.input_ids, use_cache=True)
    past_kv = out.past_key_values
    orig_len = in_ids.input_ids.size(1)
    front_len = orig_len
    
    # prefill
    if not prune:
        gpu_front = [(k.contiguous(), v.contiguous()) for k, v in past_kv]
        gpu_tail  = None
        cpu_mid   = None
        tail_len  = orig_len
    
    else:
        cur_len    = past_kv[0][0].size(-2)
        front_len  = orig_len

        remaining  = cur_len - front_len
        tail_len   = min(TAIL, remaining)
        mid_len    = max(0, remaining - tail_len)

        gpu_front, gpu_tail, cpu_mid = [], [], []
        for k, v in past_kv:
            gpu_front.append((k[:, :, :front_len, :].contiguous(),
                              v[:, :, :front_len, :].contiguous()))
            gpu_tail.append((k[:, :, front_len + mid_len:, :].contiguous(),
                             v[:, :, front_len + mid_len:, :].contiguous()))
            if mid_len > 0:
                cpu_mid.append((offload_to_cpu_async(k[:, :, front_len:front_len+mid_len, :]),
                                offload_to_cpu_async(v[:, :, front_len:front_len+mid_len, :])))
            else:
                cpu_mid.append((torch.empty(0), torch.empty(0)))

    # -------- Runtime State -------------------------------------------- #
    total_gen    = 0
    full_steps   = 0
    cot_len      = None
    cot_finished = False
    use_full     = False          # toggled for the *next* step
    buf_ids, buf_str = [], ""
    flag = 0
    recent_tokens = []
    MAX_TRACK = 16384
    recent_sentences = [] 
    # generation stage
    while True:
        # 1) build cache ------------------------------------------------- #
        past = DynamicCache()
        key_cache, value_cache = [], []

        if use_full: # 특정 threshold 이후부터 계속 full cache 유지하기 위함
            if full:
                full_cache_hold_count = 1
            else:
                full_cache_hold_count = 0
            if cot_finished and ans_full:
                full_cache_hold_count = 100000
            if early_stop and total_gen > 5000:
                full_cache_hold_count = 100000
            if prune:
                gpu_mid = []  
                for i in range(len(cpu_mid)):
                    k_c, v_c = cpu_mid[i]
                    if k_c.numel():
                        k_mid = fetch_to_gpu_async(k_c)
                        v_mid = fetch_to_gpu_async(v_c)
                        torch.cuda.synchronize()
                    else:
                        k_mid = torch.empty(0, device=DEVICE)
                        v_mid = torch.empty(0, device=DEVICE)
                    gpu_mid.append((k_mid, v_mid))
                    
        ## for all cases, add repetition handling logic
        if prune:
            for i in range(len(gpu_front)):
                k_f, v_f = gpu_front[i]
                k_t, v_t = gpu_tail[i]

                if full_cache_hold_count > 0 and gpu_mid is not None:
                    k_m, v_m = gpu_mid[i]
                    if k_m.numel():
                        k_comb = torch.cat([k_f, k_m, k_t], dim=2)
                        v_comb = torch.cat([v_f, v_m, v_t], dim=2)
                    else:
                        k_comb = torch.cat([k_f, k_t], dim=2)
                        v_comb = torch.cat([v_f, v_t], dim=2)
                else:
                    k_comb = torch.cat([k_f, k_t], dim=2)
                    v_comb = torch.cat([v_f, v_t], dim=2)

                key_cache.append(k_comb.contiguous())
                value_cache.append(v_comb.contiguous())
        else:
            key_cache = [k for k, _ in gpu_front]
            value_cache = [v for _, v in gpu_front]

                
        past.key_cache   = key_cache
        past.value_cache = value_cache
        
        
        
        # if use_full:
        #     print(total_gen)
        #     compute_redundancy_score(key_cache) # need full cache for redundancy compute

        if prune and use_full and total_gen > FRONT + TAIL:
            full_steps += 1
        # reset full flag after it has been used
        use_full = False

        # 2) prepare input ---------------------------------------------- #
        if total_gen == 0:
            inp = in_ids.input_ids[:, -1:]
        else:
            inp = next_tok

        cache_len = orig_len + total_gen
        # print(cache_len)
        pos_ids = torch.tensor([[cache_len]], device=DEVICE) ## something wrong...
        
        torch.cuda.synchronize()
        t1 = time.time()
        out = model(input_ids=inp, position_ids=pos_ids, cache_position = pos_ids,
                    past_key_values=past, use_cache=True)
        
        torch.cuda.synchronize()
        t2 = time.time()
        
        logits = out.logits[:, -1].float()
        logits[torch.isnan(logits) | torch.isinf(logits)] = -1e9
        probs  = torch.softmax(logits, dim=-1)
        next_tok = torch.argmax(probs, dim=-1, keepdim=True)   # sampling (greedy)

        tok_id = next_tok.item()
        recent_tokens.append(tok_id)
            
        if len(recent_tokens) > MAX_TRACK:
            recent_tokens.pop(0)

        if (len(buf_str)):
            if len(recent_sentences) > MAX_RECENT_SENTENCES:
                recent_sentences.pop(0)

            if detect_final_answer_repetition(recent_sentences, threshold=5):
                print(f"[{tag}] 🚨 Repeated Final Answer → force </think>")
                if not cot_finished:
                    next_tok = torch.tensor([[THINK_ID]], device=DEVICE)
                else:
                    next_tok = torch.tensor([[EOS]], device=DEVICE)
                
        if next_tok.item() == EOS or total_gen >= 16384:
            if cot_len is None:
                cot_len = total_gen
            if buf_str.strip():
                print(f"[{tag}] 📝 {buf_str.strip()}")
            print(f"[{tag}] ✅ end (CoT={cot_len}, total={total_gen})")
            if prune:
                ratio = full_steps / max(total_gen, 1)
                print(f"[{tag}] 📊 full-cache ratio: {full_steps}/{total_gen}={ratio:.2%}")
            return cot_len, total_gen, full_steps / max(total_gen, 1)

        # 4) '</think>' flag -------------------------------------------- #
        if next_tok.item() == THINK_ID and cot_len is None:
            cot_len = total_gen + 1
            cot_finished = True
            # recent_tokens = []
            recent_sentences = []

        # 5) sentence detection ----------------------------------------- #
        buf_ids.append(next_tok.item())
        buf_str = tokenizer.decode(buf_ids, skip_special_tokens=True)
        if re.search(r"[.?!}](\s+|$)", buf_str):       # sentence boundary
            print(f"[{tag}] 📝 {buf_str.strip()}")
            recent_sentences.append(buf_str.strip())
            # print(tail_len, total_gen)
            buf_ids = []
            buf_str = ""
            use_full = True     # the *next* step will use full cache


        # 6) append new kv ---------------------------------------------- #
        new_kv = [(k[:, :, -1:, :].contiguous(),
                   v[:, :, -1:, :].contiguous()) for k, v in out.past_key_values]
        
        if not prune:
            for i in range(len(gpu_front)):
                k_old, v_old = gpu_front[i]
                k_new, v_new = new_kv[i]
                gpu_front[i] = (torch.cat([k_old.contiguous(), k_new.contiguous()], dim=2),
                                torch.cat([v_old.contiguous(), v_new.contiguous()], dim=2))
            
            tail_len += 1

        else:
            # concat to tail kv
            for i in range(len(gpu_tail)):
                k_t, v_t = gpu_tail[i]
                k_new, v_new = new_kv[i]
                gpu_tail[i] = (torch.cat([k_t.contiguous(), k_new.contiguous()], dim=2),
                               torch.cat([v_t.contiguous(), v_new.contiguous()], dim=2))
            tail_len += 1

            # ---- STEP‑wise offload ------------------------------------ #
            if tail_len > TAIL and (tail_len - TAIL >= STEP) and not flag:
                for i in range(len(gpu_tail)):
                    k_t, v_t = gpu_tail[i]
                    k_off = k_t[:, :, :STEP, :].contiguous()
                    v_off = v_t[:, :, :STEP, :].contiguous()
                    k_t   = k_t[:, :, STEP:, :].contiguous()
                    v_t   = v_t[:, :, STEP:, :].contiguous()
                    gpu_tail[i] = (k_t, v_t)

                    k_off_cpu = offload_to_cpu_async(k_off)
                    v_off_cpu = offload_to_cpu_async(v_off)
                    torch.cuda.synchronize()

                    k_cpu, v_cpu = cpu_mid[i]
                    
                    if k_cpu.numel() == 0:
                        k_cpu, v_cpu = k_off_cpu, v_off_cpu
                    else:
                        k_cpu = torch.cat([k_cpu, k_off_cpu], dim=2).contiguous()
                        v_cpu = torch.cat([v_cpu, v_off_cpu], dim=2).contiguous()
                    cpu_mid[i] = (k_cpu, v_cpu)
                tail_len -= STEP
                mid_len += STEP

        total_gen += 1

        if full_cache_hold_count > 0:
            full_cache_hold_count -= 1
            if full_cache_hold_count == 0:
                gpu_mid = None 

        # print(tail_len)

# ── 결과 CSV 처리 및 실행 ─────────────────────
existing_rows = set()
if os.path.exists(RESULT_CSV):
    with open(RESULT_CSV, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            existing_rows.add(int(row["index"]))

CSV_HEADER = [
    "index",
    "cot_len_NP", "total_NP", "ratio_NP",
    "cot_len_PR", "total_PR", "ratio_PR",
    "cot_len_PR_ANS", "total_PR_ANS", "ratio_PR_ANS",
    "cot_len_PR_ANS_EARLY", "total_PR_ANS_EARLY", "ratio_PR_ANS_EARLY"
]

ds = load_dataset("HuggingFaceH4/MATH-500", "default", split=f"test[:{300}]")

with open(RESULT_CSV, "a", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=CSV_HEADER)
    if os.stat(RESULT_CSV).st_size == 0:
        writer.writeheader()

    for i, row in enumerate(ds, 1):
        if i in existing_rows:
            continue

        question = row["problem"]
        print(f"\n=== [{i:03d}] {question.strip()} ===")

        # try:
        # cot_np, tot_np, r_np = run(question, prune=False, tag=f"{i}-NP", full=False, ans_full=True, early_stop=True)
        # cot_pr, tot_pr, r_pr = run(question, prune=True, tag=f"{i}-PR", full=False, ans_full=False, early_stop=False)
        # cot_pra, tot_pra, r_pra = run(question, prune=True, tag=f"{i}-PRA", full=False, ans_full=True, early_stop=False)
        cot_prae, tot_prae, r_prae = run(question, prune=True, tag=f"{i}-PRAE", full=False, ans_full=True, early_stop=True)

        # writer.writerow({
        #     "index": i,
        #     "cot_len_NP": cot_np, "total_NP": tot_np, "ratio_NP": r_np,
        #     "cot_len_PR": cot_pr, "total_PR": tot_pr, "ratio_PR": r_pr,
        #     "cot_len_PR_ANS": cot_pra, "total_PR_ANS": tot_pra, "ratio_PR_ANS": r_pra,
        #     "cot_len_PR_ANS_EARLY": cot_prae, "total_PR_ANS_EARLY": tot_prae, "ratio_PR_ANS_EARLY": r_prae
        # })
        # f.flush()
        # print(f"✅ Saved index {i}")

        # except Exception as e:
        #     print(f"[❌ ERROR] Failed on index {i}: {e}")