vllm serve ./Qwen3-4B-Instruct-2507 \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.95 \
    --max-model-len 8192 \
    --trust-remote-code

