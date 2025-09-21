CUDA_LAUNCH_BLOCKING=1 \
python -m lcb_runner.runner.main --model Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8 --scenario codegeneration \
--evaluate --release_version v6 --max_tokens 4096 --tensor_parallel_size 4 --temperature 0.7 \
--expert_parallel --top_p 0.8