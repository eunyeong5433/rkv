python test_real_llama.py \
  --draft_model ../DeepSeek-R1-Distill-Llama-8B/ \
  --target_model ../DeepSeek-R1-Distill-Llama-8B/ \
  --gamma 5 --front 4 --tail 256 \
  --temperature 1.0 --max_new_tokens 16384 \
  --batch_size 1
