"""
scripts/run_benchmark.py
------------------------
Benchmark inference latency and throughput for trained models.
Compares all three architectures side by side.

Usage:
    python scripts/run_benchmark.py --checkpoint models/cnn1d_best.pt --model cnn1d
    python scripts/run_benchmark.py --all   # benchmark all saved checkpoints
"""

import argparse
import time
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import numpy as np
from src.models import build_model
from src.utils  import get_device


def benchmark_model(model, device, input_shape, n_warmup=100, n_runs=1000):
    """
    Measure single-sample inference latency (ms) and batch throughput (samples/s).

    Returns:
        dict with latency_ms and throughput_sps
    """
    model.eval()
    model.to(device)

    dummy = torch.randn(1, *input_shape).to(device)

    # Warmup — GPU needs a few passes to reach stable clock speed
    with torch.no_grad():
        for _ in range(n_warmup):
            model(dummy)

    if device.type == "cuda":
        torch.cuda.synchronize()

    # Single sample latency
    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            t0 = time.perf_counter()
            model(dummy)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)

    latency_ms = np.mean(times)
    latency_std = np.std(times)

    # Batch throughput
    batch = torch.randn(512, *input_shape).to(device)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(20):
            model(batch)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    throughput = (512 * 20) / elapsed

    return {
        "latency_ms":   latency_ms,
        "latency_std":  latency_std,
        "throughput":   throughput,
        "params":       sum(p.numel() for p in model.parameters() if p.requires_grad),
    }


def print_benchmark_table(results: dict):
    print("\n" + "="*70)
    print(f"  {'Model':<15} {'Params':>10} {'Latency (ms)':>15} {'Throughput':>18}")
    print(f"  {'':<15} {'':>10} {'mean ± std':>15} {'(samples/sec)':>18}")
    print("-"*70)
    for name, r in results.items():
        print(
            f"  {name:<15} "
            f"{r['params']:>10,} "
            f"{r['latency_ms']:>8.3f} ± {r['latency_std']:.3f} ms"
            f"{r['throughput']:>16,.0f}"
        )
    print("="*70)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model",      type=str, default=None,
                   choices=["cnn1d", "cnn2d", "transformer"])
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--all",        action="store_true",
                   help="Benchmark all three models")
    p.add_argument("--n_runs",     type=int, default=1000)
    args = p.parse_args()

    device = get_device()

    # Input shapes per model
    shapes = {
        "cnn1d":       (2, 128),
        "cnn2d":       (1, 17, 9),
        "transformer": (2, 128),
    }

    if args.all:
        models_to_run = ["cnn1d", "cnn2d", "transformer"]
    elif args.model:
        models_to_run = [args.model]
    else:
        models_to_run = ["cnn1d", "cnn2d", "transformer"]

    results = {}
    for name in models_to_run:
        print(f"\n[Benchmark] {name.upper()} ...")
        model = build_model(name)

        ckpt = args.checkpoint or f"models/{name}_best.pt"
        if os.path.exists(ckpt):
            model.load_state_dict(torch.load(ckpt, map_location=device))
            print(f"  Loaded checkpoint: {ckpt}")
        else:
            print(f"  No checkpoint found at {ckpt} — using random weights")

        r = benchmark_model(model, device, shapes[name], n_runs=args.n_runs)
        results[name.upper()] = r
        print(f"  Latency : {r['latency_ms']:.3f} ms ± {r['latency_std']:.3f} ms")
        print(f"  Throughput : {r['throughput']:,.0f} samples/sec")
        print(f"  Parameters : {r['params']:,}")

    print_benchmark_table(results)
    print("\n[Benchmark complete]")


if __name__ == "__main__":
    main()
