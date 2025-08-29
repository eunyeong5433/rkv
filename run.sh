python test_real_llama.py \
  --draft_model ../DeepSeek-R1-Distill-Qwen-1.5B/ \
  --target_model ../DeepSeek-R1-Distill-Qwen-7B/ \
  --gamma 5 --front 4 --tail 256 \
  --temperature 0.0 --max_new_tokens 16384 \
  --batch_size 2
