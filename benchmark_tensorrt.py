"""
benchmark_tensorrt.py

Demonstrates TensorRT optimization pipeline:
  1. Load a HuggingFace BERT model
  2. Export to ONNX (single file, no external weights)
  3. Build TensorRT engines (FP32 and FP16)
  4. Benchmark latency: HuggingFace baseline vs ONNX vs TRT FP32 vs TRT FP16
  5. Save engines and results to D drive

Hardware: NVIDIA GTX 1650 Super (sm75, Turing)
"""

import time
import statistics
import numpy as np
import torch
import tensorrt as trt
import onnx
import onnxruntime as ort
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path
from onnx.external_data_helper import load_external_data_for_model
from transformers import AutoTokenizer, AutoModel

BASE_DIR  = Path("/mnt/d/vllm-project")
ENGINE_DIR = BASE_DIR / "trt-engines"
MODEL_DIR = BASE_DIR / "hf-cache"
SERVER_DIR = BASE_DIR / "server"
ONNX_PATH = ENGINE_DIR / "bert_base.onnx"
TRT_FP32  = ENGINE_DIR / "bert_base_fp32.engine"
TRT_FP16 = ENGINE_DIR / "bert_base_fp16.engine"

ENGINE_DIR.mkdir(parents=True, exist_ok=True)

# Config 
MODEL_NAME = "bert-base-uncased"
SEQ_LEN = 128
BATCH_SIZE = 1
N_RUNS = 50       # inference runs per benchmark
WARMUP_RUNS = 10       # warmup runs (discarded)
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


# 1. EXPORT TO ONNX

def export_onnx(model, tokenizer) -> None:
    print("\n── Step 1: Exporting to ONNX ──────────")
    if ONNX_PATH.exists():
        print(f"ONNX already exists at {ONNX_PATH}, skipping export.")
        return

    # Force FP32 — TensorRT cannot parse BF16 weights
    model = model.float()

    dummy_input = tokenizer(
        "This is a benchmark test sentence for TensorRT optimization.",
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=SEQ_LEN,
    )
    input_ids= dummy_input["input_ids"]
    attention_mask = dummy_input["attention_mask"]
    token_type_ids = dummy_input["token_type_ids"]

    model.eval()
    with torch.no_grad():
        torch.onnx.export(
            model,
            (input_ids, attention_mask, token_type_ids),
            str(ONNX_PATH),
            input_names=["input_ids", "attention_mask", "token_type_ids"],
            output_names=["last_hidden_state", "pooler_output"],
            dynamic_axes={
                "input_ids":{0: "batch_size"},
                "attention_mask": {0: "batch_size"},
                "token_type_ids": {0: "batch_size"},
            },
            opset_version=13,
            do_constant_folding=True,
        )

    model_proto = onnx.load(str(ONNX_PATH), load_external_data=False)
    load_external_data_for_model(model_proto, str(ENGINE_DIR))
    onnx.save(model_proto, str(ONNX_PATH))

    data_file = Path(str(ONNX_PATH) + ".data")
    if data_file.exists():
        data_file.unlink()

    onnx_model = onnx.load(str(ONNX_PATH))
    onnx.checker.check_model(onnx_model)
    print(f"  ONNX export validated (single file) → {ONNX_PATH}")

# 2. BUILD TENSORRT ENGINE

def build_trt_engine(fp16: bool = False) -> Path:
    out_path= TRT_FP16 if fp16 else TRT_FP32
    precision = "FP16" if fp16 else "FP32"
    print(f"\n── Step 2: Building TensorRT {precision} Engine ─────────────")

    if out_path.exists():
        print(f" Engine already exists at {out_path}, skipping build.")
        return out_path

    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, TRT_LOGGER)
    config = builder.create_builder_config()

    # 1GB workspace — stays within 4GB VRAM budget
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)

    if fp16:
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            print("  FP16 fast path available on this GPU ")
        else:
            print("  FP16 fast path not available, falling back to FP32")
    print(f"  Parsing ONNX model from {ONNX_PATH} ...")
    with open(str(ONNX_PATH), "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(f"  ONNX parse error: {parser.get_error(i)}")
            raise RuntimeError("Failed to parse ONNX model")

    profile = builder.create_optimization_profile()
    input_names = ["input_ids", "attention_mask", "token_type_ids"]
    for name in input_names:
        profile.set_shape(
            name,
            min=(1, SEQ_LEN),   
            opt=(1, SEQ_LEN),   
            max=(4, SEQ_LEN),  
        )
    config.add_optimization_profile(profile)

    print("  Building engine (1–3 minutes on first run) ...")
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("Failed to build TensorRT engine")

    with open(str(out_path), "wb") as f:
        f.write(serialized)

    print(f"   Engine saved → {out_path}")
    return out_path


# 3. LOAD TRT ENGINE FOR INFERENCE

def load_trt_engine(engine_path: Path):
    runtime = trt.Runtime(TRT_LOGGER)
    with open(str(engine_path), "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    context = engine.create_execution_context()
    return engine, context


def trt_inference(context, engine, input_ids, attention_mask, token_type_ids):
    """Run one TensorRT inference pass, return outputs as numpy arrays."""
    inputs_np = {
        "input_ids": input_ids.numpy().astype(np.int32),
        "attention_mask": attention_mask.numpy().astype(np.int32),
        "token_type_ids": token_type_ids.numpy().astype(np.int32),
    }

    for name, arr in inputs_np.items():
        context.set_input_shape(name, arr.shape)

    device_ptrs = []
    outputs = []

    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)

        if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
            d_mem = torch.tensor(inputs_np[name]).cuda().contiguous()
        else:
           
            shape = tuple(context.get_tensor_shape(name))
            d_mem = torch.zeros(shape, dtype=torch.float32).cuda()
            outputs.append((name, d_mem))

        context.set_tensor_address(name, d_mem.data_ptr())
        device_ptrs.append(d_mem)

    stream = torch.cuda.current_stream().cuda_stream
    context.execute_async_v3(stream)
    torch.cuda.synchronize()

    return {name: tensor.cpu().numpy() for name, tensor in outputs}

# 4. BENCHMARKS

def benchmark_hf(model, tokenizer) -> dict:
    print("\n── Benchmark: HuggingFace Baseline (CPU) ───────────────────")
    text = "Benchmarking HuggingFace BERT inference latency on CPU baseline."
    inputs = tokenizer(
        text, return_tensors="pt",
        padding="max_length", truncation=True, max_length=SEQ_LEN,
    )

    model.eval()
    for _ in range(WARMUP_RUNS):
        with torch.no_grad():
            model(**inputs)

    times = []
    for _ in range(N_RUNS):
        t0 = time.perf_counter()
        with torch.no_grad():
            model(**inputs)
        times.append((time.perf_counter() - t0) * 1000)

    result = {
        "mean_ms": statistics.mean(times),
        "stdev_ms": statistics.stdev(times),
        "min_ms": min(times),
        "p95_ms": sorted(times)[int(0.95 * len(times))],
    }
    print(f" mean={result['mean_ms']:.2f}ms  p95={result['p95_ms']:.2f}ms  "
          f"std={result['stdev_ms']:.2f}ms")
    return result


def benchmark_onnx(tokenizer) -> dict:
    print("\n── Benchmark: ONNX Runtime (GPU) ───────────────────────────")
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(
        str(ONNX_PATH),
        sess_options=sess_options,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    print(f"  Execution provider: {session.get_providers()[0]}")

    text = "Benchmarking ONNX Runtime inference latency with GPU acceleration."
    inputs = tokenizer(
        text, return_tensors="np",
        padding="max_length", truncation=True, max_length=SEQ_LEN,
    )
    feed = {
        "input_ids": inputs["input_ids"].astype(np.int64),
        "attention_mask": inputs["attention_mask"].astype(np.int64),
        "token_type_ids": inputs["token_type_ids"].astype(np.int64),
    }

    for _ in range(WARMUP_RUNS):
        session.run(None, feed)

    times = []
    for _ in range(N_RUNS):
        t0 = time.perf_counter()
        session.run(None, feed)
        times.append((time.perf_counter() - t0) * 1000)

    result = {
        "mean_ms": statistics.mean(times),
        "stdev_ms": statistics.stdev(times),
        "min_ms":min(times),
        "p95_ms": sorted(times)[int(0.95 * len(times))],
    }
    print(f"  mean={result['mean_ms']:.2f}ms  p95={result['p95_ms']:.2f}ms  "
          f"std={result['stdev_ms']:.2f}ms")
    return result


def benchmark_trt(engine_path: Path, tokenizer, label: str) -> dict:
    print(f"\n── Benchmark: TensorRT {label} ──────────────────────────────")
    engine, context = load_trt_engine(engine_path)

    text = "Benchmarking TensorRT optimized inference latency on GTX 1650 Super."
    inputs = tokenizer(
        text, return_tensors="pt",
        padding="max_length", truncation=True, max_length=SEQ_LEN,
    )
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    token_type_ids = inputs["token_type_ids"]

    for _ in range(WARMUP_RUNS):
        trt_inference(context, engine, input_ids, attention_mask, token_type_ids)

    times = []
    for _ in range(N_RUNS):
        t0 = time.perf_counter()
        trt_inference(context, engine, input_ids, attention_mask, token_type_ids)
        times.append((time.perf_counter() - t0) * 1000)

    result = {
        "mean_ms": statistics.mean(times),
        "stdev_ms": statistics.stdev(times),
        "min_ms":min(times),
        "p95_ms":sorted(times)[int(0.95 * len(times))],
    }
    print(f" mean={result['mean_ms']:.2f}ms  p95={result['p95_ms']:.2f}ms  "
          f"std={result['stdev_ms']:.2f}ms")
    return result

# 5. CHART + SUMMARY

def plot_results(results: dict) -> None:
    labels = list(results.keys())
    means= [results[l]["mean_ms"]  for l in labels]
    p95s   = [results[l]["p95_ms"]   for l in labels]
    stdevs = [results[l]["stdev_ms"] for l in labels]
    colors = ["#888780", "#55A868", "#4C72B0", "#C44E52"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        "TensorRT Optimization Pipeline — BERT-base Inference\n"
        "GTX 1650 Super (sm75) | seq_len=128 | batch=1",
        fontsize=12, fontweight="bold",
    )

    # Panel 1: mean latency with error bars + speedup labels
    bars     = axes[0].bar(labels, means, yerr=stdevs,
                           color=colors, capsize=6,
                           edgecolor="black", linewidth=0.5)
    baseline = means[0]
    axes[0].set_title("Mean Inference Latency (lower = better)")
    axes[0].set_ylabel("Milliseconds")
    axes[0].yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
    for bar, mean, std in zip(bars[1:], means[1:], stdevs[1:]):
        speedup = baseline / mean
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            mean + std + 0.3,
            f"{speedup:.1f}x",
            ha="center", va="bottom", fontsize=10, fontweight="bold",
        )

    # Panel 2: p95 latency
    axes[1].bar(labels, p95s, color=colors, edgecolor="black", linewidth=0.5)
    axes[1].set_title("P95 Latency (lower = better)")
    axes[1].set_ylabel("Milliseconds")
    axes[1].yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))

    plt.tight_layout()
    out_path = SERVER_DIR / "benchmark_tensorrt_results.png"
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    print(f"\n  Chart saved → {out_path}")


def print_summary(results: dict) -> None:
    baseline = results[list(results.keys())[0]]["mean_ms"]
    print("\n" + "=" * 60)
    print("TENSORRT BENCHMARK SUMMARY")
    print(f"Model: BERT-base-uncased | seq_len={SEQ_LEN} | {N_RUNS} runs")
    print("=" * 60)
    print(f"{'Backend':<22} {'Mean (ms)':<12} {'P95 (ms)':<12} {'Speedup'}")
    print("-" * 56)
    for label, r in results.items():
        speedup = f"{baseline / r['mean_ms']:.2f}x"
        print(f"{label:<22} {r['mean_ms']:<12.2f} {r['p95_ms']:<12.2f} {speedup}")
    print("=" * 60)


if __name__ == "__main__":
    print("Loading BERT model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    model.eval()
    print(f" Model loaded: {MODEL_NAME}")

    # Step 1: Export to ONNX (single file, weights inlined)
    export_onnx(model, tokenizer)

    # Step 2: Build TensorRT engines
    build_trt_engine(fp16=False)
    build_trt_engine(fp16=True)

    # Step 3: Run all benchmarks
    results = {}
    results["HuggingFace (CPU)"] = benchmark_hf(model, tokenizer)
    results["ONNX Runtime (GPU)"] = benchmark_onnx(tokenizer)
    results["TensorRT FP32"]      = benchmark_trt(TRT_FP32, tokenizer, "FP32")
    results["TensorRT FP16"]      = benchmark_trt(TRT_FP16, tokenizer, "FP16")

    # Step 4: Output
    print_summary(results)
    plot_results(results)

    print("\n All done! Files saved to D drive:")
    print(f"  Engines : {ENGINE_DIR}")
    print(f" Chart   : {SERVER_DIR / 'benchmark_tensorrt_results.png'}")