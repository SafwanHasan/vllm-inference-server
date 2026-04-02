"""
benchmark.py
Benchmarks the vLLM server: latency, TTFT, throughput, concurrency.
Saves chart to /mnt/d/vllm-project/server/benchmark_results.png

Run AFTER serve.sh is up and showing "Uvicorn running on http://0.0.0.0:8000"
"""
import time
import statistics
import concurrent.futures
import requests
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path

BASE_URL = "http://localhost:8000/v1"
MODEL = "qwen2.5-1.5b"
HEADERS = {"Content-Type": "application/json", "Authorization": "Bearer placeholder"}
OUTPUT_DIR = Path("/mnt/d/vllm-project/server")

PROMPTS = {
    "short":  "What is 2 + 2?",
    "medium": "Explain what a transformer neural network is in 3 sentences.",
    "long": "Write a detailed explanation of how Attention works in Transformer"
}


def non_streaming_request(prompt: str, max_tokens: int = 256) -> dict:
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": False,
    }
    t0 = time.perf_counter()
    resp = requests.post(f"{BASE_URL}/chat/completions", json=payload, headers=HEADERS)

    elapsed = time.perf_counter() - t0
    resp.raise_for_status()

    data = resp.json()
    completion_tokens = data["usage"]["completion_tokens"]

    return {
        "elapsed_s": elapsed,
        "completion_tokens": completion_tokens,
        "prompt_tokens": data["usage"]["prompt_tokens"],
        "tokens_per_sec": completion_tokens / elapsed,
        "text": data["choices"][0]["message"]["content"],
    }


def time_to_first_token(prompt: str, max_tokens: int = 128) -> float:
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
    }

    t0 = time.perf_counter()
    with requests.post(
        f"{BASE_URL}/chat/completions", json=payload, headers=HEADERS, stream=True
    ) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if line and line != b"data: [DONE]":
                return time.perf_counter() - t0

    return -1.0


def concurrent_benchmark(prompt: str, n_concurrent: int = 4, max_tokens: int = 128) -> dict:
    def single():
        return non_streaming_request(prompt, max_tokens)

    t0 = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_concurrent) as ex:
        futures = [ex.submit(single) for _ in range(n_concurrent)]
        results = [f.result() for f in concurrent.futures.as_completed(futures)]
    wall_time = time.perf_counter() - t0

    total_tokens = sum(r["completion_tokens"] for r in results)
    latencies = [r["elapsed_s"] for r in results]
    return {
        "wall_time_s":    wall_time,
        "total_tokens" :   total_tokens,
        "aggregate_tps" :  total_tokens / wall_time,
        "avg_latency_s" :  statistics.mean(latencies),
        "p95_latency_s":  sorted(latencies)[int(0.95 * len(latencies))],
    }


def run_latency_benchmark(n_runs: int = 5) -> dict:
    print("\n── Latency Benchmark ──────────────────────────────────────")
    results = {}
    for label, prompt in PROMPTS.items():
        times = []
        for i in range(n_runs):
            r = non_streaming_request(prompt)
            times.append(r["elapsed_s"])
            print(f"  [{label}] run {i+1}: {r['elapsed_s']:.2f}s | "
                  f"{r['completion_tokens']} tokens | {r['tokens_per_sec']:.1f} tok/s")
        results[label] = {
            "mean_s" :  statistics.mean(times),
            "stdev_s": statistics.stdev(times) if len(times) > 1 else 0,
            "min_s" :   min(times),
            "max_s":   max(times),
        }
        print(f"  [{label}] avg={results[label]['mean_s']:.2f}s "
              f"+/-{results[label]['stdev_s']:.2f}s\n")
    return results


def run_ttft_benchmark(n_runs: int = 5) -> dict:
    print("── Time to First Token ─────────────────────────────────────")
    results = {}
    for label, prompt in PROMPTS.items():
        times= [time_to_first_token(prompt) for _ in range(n_runs)]
        mean = statistics.mean(times)
        results[label] = mean
        print(f" [{label}] TTFT avg: {mean*1000:.0f}ms")
    return results


def run_concurrency_benchmark() -> dict:
    print("\n── Concurrency Benchmark ───────────────────────────────────")
    prompt  = PROMPTS["medium"]
    results = {}
    for n in [1, 2, 4]:
        r = concurrent_benchmark(prompt, n_concurrent=n)
        results[n] = r
        print(f"  [{n} concurrent] wall={r['wall_time_s']:.2f}s | "
              f"total_tokens={r['total_tokens']} | "
              f"agg_tps={r['aggregate_tps']:.1f} | "
              f"avg_lat={r['avg_latency_s']:.2f}s")
    return results


def plot_results(latency_results: dict, ttft_results: dict, concurrency_results: dict):
    labels = list(latency_results.keys())
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        "vLLM Inference Server — Benchmark Results\n"
        "GTX 1650 Super (4GB VRAM) | Qwen2.5-1.5B | 4-bit BitsAndBytes quant",
        fontsize=12, fontweight="bold"
    )

    # panel 1: avg latency
    means = [latency_results[l]["mean_s"] for l in labels]
    errs  = [latency_results[l]["stdev_s"] for l in labels]
    axes[0].bar(labels, means, yerr=errs,
                color=["#4C72B0", "#55A868", "#C44E52"],
                capsize=6, edgecolor="black", linewidth=0.5)
    axes[0].set_title("Avg Latency by Prompt Size")
    axes[0].set_ylabel("Seconds")
    axes[0].yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))

    # Panel 2: TTFT
    ttft_vals = [ttft_results[l] * 1000 for l in labels]
    axes[1].bar(labels, ttft_vals,
                color=["#4C72B0", "#55A868", "#C44E52"],
                edgecolor="black", linewidth=0.5)
    axes[1].set_title("Time to First Token (ms)")
    axes[1].set_ylabel("Milliseconds")

    # panel 3: throughput vs concurrency
    ns       = sorted(concurrency_results.keys())
    tps_vals = [concurrency_results[n]["aggregate_tps"] for n in ns]
    axes[2].plot([str(n) for n in ns], tps_vals,
                 marker="o", linewidth=2, color="#4C72B0", markersize=8)
    axes[2].set_title("Aggregate Throughput vs Concurrency")
    axes[2].set_xlabel("Concurrent Requests")
    axes[2].set_ylabel("Tokens / Second")
    axes[2].yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f"))

    plt.tight_layout()
    out_path = OUTPUT_DIR / "benchmark_results.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\n  Chart saved to {out_path}")


def print_summary(latency_results, ttft_results, concurrency_results):
    print("\n" + "="*55)
    print("BENCHMARK SUMMARY")
    print("="*55)
    print(f"{'Prompt':<10} {'Avg Lat (s)':<14} {'TTFT (ms)':<12}")
    print("-"*36)
    for label in latency_results:
        lat  = latency_results[label]["mean_s"]
        ttft = ttft_results[label] * 1000
        print(f"{label:<10} {lat:<14.2f} {ttft:<12.0f}")
    print()
    print(f"{'Concurrency':<14} {'Agg TPS':<12} {'Avg Lat (s)'}")
    print("-"*38)
    for n, r in sorted(concurrency_results.items()):
        print(f"{n:<14} {r['aggregate_tps']:<12.1f} {r['avg_latency_s']:.2f}")
    print("="*55)


if __name__ == "__main__":
    print("Starting vLLM benchmark, make sure serve.sh is running!\n")

    try:
        models = requests.get(f"{BASE_URL}/models", headers=HEADERS, timeout=5)
        models.raise_for_status()
        print(f" Server online. Model: {models.json()['data'][0]['id']}\n")
    except Exception as e:
        print(f" Server not reachable: {e}")
        print("Start it with: bash /mnt/d/vllm-project/server/serve.sh")
        exit(1)

    latency_results = run_latency_benchmark(n_runs=5)
    ttft_results= run_ttft_benchmark(n_runs=5)
    concurrency_results = run_concurrency_benchmark()

    print_summary(latency_results, ttft_results, concurrency_results)
    plot_results(latency_results, ttft_results, concurrency_results)