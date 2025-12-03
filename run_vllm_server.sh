

# start vLLM server

python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-4B-Instruct-2507 \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size 1 \
    --dtype auto \
    --gpu-memory-utilization 0.85 \
    --max_model_len 65536 \
    --trust_remote_code

