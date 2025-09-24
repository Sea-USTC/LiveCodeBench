CUDA_LAUNCH_BLOCKING=1 VLLM_ATTENTION_BACKEND=FLASHINFER \
python -m lcb_runner.runner.main --model Qwen/Qwen3-30B-A3B-Thinking-2507-FP8 --scenario codegeneration \
--evaluate --release_version v6 --max_tokens 32768 --tensor_parallel_size 4 --use_cache --kv_cache_quantized --temperature 0.7 \
--expert_parallel --top_p 0.8