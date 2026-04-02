# LLM Inference Optimization Server

Multi-backend LLM inference optimization system benchmarking **vLLM**, **TensorRT**, and **ONNX Runtime** on consumer hardware (NVIDIA GTX 1650 Super, 4GB VRAM). Demonstrates production inference techniques including PagedAttention, kernel fusion, and quantization.

## Highlights

- **OpenAI-compatible REST API** via vLLM's built-in server (`/v1/chat/completions`)
- **4-bit quantization** (BitsAndBytes NF4) — fits a 1.5B LLM within 4GB VRAM with <5% quality loss
- **TensorRT optimization pipeline** — ONNX export → FP32/FP16 engine build → 11.6x latency speedup over CPU baseline
- **Streaming inference** with time-to-first-token (TTFT) and concurrency benchmarking
- All benchmark charts auto-generated and saved to disk

## Stack

- [vLLM](https://github.com/vllm-project/vllm) — PagedAttention, dynamic batching, OpenAI-compatible API
- [TensorRT](https://developer.nvidia.com/tensorrt) — ONNX graph optimization, FP32/FP16 engine compilation
- [ONNX Runtime](https://onnxruntime.ai/) — GPU-accelerated inference with full graph optimization
- [BitsAndBytes](https://github.com/TimDettmers/bitsandbytes) — 4-bit NF4 weight quantization
- [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) — instruction-tuned LLM
- [BERT-base-uncased](https://huggingface.co/bert-base-uncased) — transformer model for TensorRT pipeline
- Python 3.13, CUDA 12.8, WSL2 (Ubuntu 24.04)

## Project Structure

```
├── serve.sh                         # vLLM OpenAI-compatible server
├── client.py                        # Interactive streaming chat client
├── benchmark.py                     # vLLM latency, TTFT, concurrency benchmarks
├── benchmark_tensorrt.py            # TensorRT optimization pipeline + benchmarks
├── benchmark_results.png            # vLLM benchmark chart
└── benchmark_tensorrt_results.png   # TensorRT benchmark chart
```

## Setup

```bash
# Terminal 1 — start the vLLM server (downloads model on first run ~1.5GB)
bash /mnt/d/vllm-project/server/serve.sh

# Terminal 2 — run vLLM benchmarks
cd /mnt/d/vllm-project/server
python benchmark.py

# Run TensorRT optimization pipeline (builds engines on first run, ~2 min)
python benchmark_tensorrt.py

# Or chat interactively with the vLLM server
python client.py
```
   
## Benchmark Results

### vLLM — Qwen2.5-1.5B-Instruct (4-bit BitsAndBytes, GTX 1650 Super)

| Prompt Size | Avg Latency | TTFT |
|-------------|-------------|------|
| Short | 0.85s | 240ms |
| Medium | 5.93s | 262ms |
| Long | 11.54s | 252ms |

| Concurrency | Aggregate Throughput | Avg Latency |
|-------------|----------------------|-------------|
| 1 request | 23.0 tok/s | 4.96s |
| 2 requests | 12.5 tok/s | 18.24s |
| 4 requests | 24.7 tok/s | 18.46s |

### TensorRT — BERT-base-uncased (seq_len=128, batch=1, GTX 1650 Super)

| Backend | Mean Latency | P95 Latency | Speedup |
|---------|-------------|-------------|---------|
| HuggingFace CPU baseline | 114.67ms | 123.70ms | 1.00x |
| ONNX Runtime (GPU) | 13.62ms | 15.71ms | 8.42x |
| TensorRT FP32 | 11.23ms | 13.63ms | 10.21x |
| TensorRT FP16 | 9.87ms | 11.56ms | 11.62x |

> **Note on FP16 vs FP32:** On Turing (sm75) architecture, FP16 shows measurable gains (10.21x → 11.62x)
> but not the theoretical 2x improvement seen on Ampere and later. At BERT scale the bottleneck
> shifts from compute to memory bandwidth, which FP16 helps with but does not fully eliminate.

## Key Concepts Demonstrated

- **PagedAttention**: vLLM's memory manager treats the KV cache like OS virtual memory pages, eliminating the 60-80% waste from static pre-allocation
- **Dynamic batching**: Concurrent requests are grouped automatically to maximize GPU utilization
- **TensorRT kernel fusion**: Adjacent operations (LayerNorm + attention + matmul) are fused into single GPU kernels, reducing memory round-trips and achieving 11.6x latency reduction
- **Quantization tradeoffs**: 4-bit NF4 reduces VRAM by ~75% vs FP16 with <5% perplexity increase; FP16 vs FP32 gains are architecture-dependent
- **ONNX as interchange format**: Model exported from PyTorch → validated with ONNX checker → parsed by TensorRT, demonstrating the full production optimization pipeline