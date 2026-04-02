#!/bin/bash
# Activate venv from D drive
source /mnt/d/vllm-project/venv/bin/activate

# Force model downloads to D drive (belt-and-suspenders alongside ~/.bashrc)
export HF_HOME="/mnt/d/vllm-project/hf-cache"
export HUGGINGFACE_HUB_CACHE="/mnt/d/vllm-project/hf-cache"
export TRANSFORMERS_CACHE="/mnt/d/vllm-project/hf-cache"

vllm serve "Qwen/Qwen2.5-1.5B-Instruct" \
    --gpu-memory-utilization 0.80 \
    --quantization bitsandbytes \
    --load-format bitsandbytes \
    --max-model-len 2048 \
    --host 0.0.0.0 \
    --port 8000 \
    --served-model-name "qwen2.5-1.5b"

    