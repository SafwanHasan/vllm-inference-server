# vLLM Inference Server

OpenAI-compatible LLM inference server using [vLLM](https://github.com/vllm-project/vllm), running **Qwen2.5-1.5B-Instruct** with 4-bit BitsAndBytes quantization on consumer hardware (NVIDIA GTX 1650 Super, 4GB VRAM).

## Highlights

- **OpenAI-compatible REST API** via vLLM's built-in server (`/v1/chat/completions`)
- **4-bit quantization** (BitsAndBytes NF4) — fits a 1.5B model within 4GB VRAM
- **Streaming inference** with time-to-first-token (TTFT) measurement
- **Concurrency benchmarking** — aggregate throughput under 1/2/4 parallel requests
- Benchmark chart auto-generated as `benchmark_results.png`

## Stack

- [vLLM](https://github.com/vllm-project/vllm) — PagedAttention, dynamic batching, OpenAI-compatible API
- [BitsAndBytes](https://github.com/TimDettmers/bitsandbytes) — 4-bit NF4 weight quantization
- [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) — instruction-tuned LLM
- Python 3.11, CUDA 12.6, WSL2 (Ubuntu 24.04)

## Setup
```bash
# Terminal 1 — start the server (downloads model on first run ~1.5GB)
bash /mnt/d/vllm-project/server/serve.sh

# Terminal 2 — run benchmarks
cd /mnt/d/vllm-project/server
python benchmark.py

# Or chat interactively
python client.py
```

## Key Concepts Demonstrated

- **PagedAttention**: vLLM's memory manager treats KV cache like OS virtual memory pages, eliminating waste from static pre-allocation
- **Dynamic batching**: Concurrent requests are batched automatically to maximize GPU utilization
- **Quantization tradeoffs**: 4-bit NF4 reduces VRAM by ~75% vs FP16 with <5% perplexity increase on standard benchmarks

## Benchmark Results

| Prompt Size | Avg Latency | TTFT |
|-------------|-------------|------|
| Short       | 0.76s       | 242ms |
| Medium      | 4.90s       | 260ms |
| Long        | 11.24s      | 264ms |

| Concurrency | Aggregate Throughput |
|-------------|----------------------|
| 1 request   | 22.8 tok/s |
| 2 requests  | 12.4 tok/s |
| 4 requests  | 24.7 tok/s |