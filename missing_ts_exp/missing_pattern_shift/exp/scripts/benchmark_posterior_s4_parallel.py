#!/usr/bin/env python3
"""Short same-GPU contention check after the full single-process benchmarks."""
import argparse
import json
import os
import subprocess
import sys
import time

from protocol import EXP_ROOT, atomic_json_dump
from posterior_s4_common import SUITE

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", choices=("0", "5"), required=True)
parser.add_argument("--concurrency", type=int, choices=(2, 3), required=True)
parser.add_argument("--batch-size", type=int, default=64)
parser.add_argument("--workers", type=int, default=10)
args = parser.parse_args()
root = EXP_ROOT / "diagnostics" / SUITE
root.mkdir(parents=True, exist_ok=True)
processes = []
start = time.time()
variants = ["LI-S4", "PU-S4"] if args.concurrency == 2 else ["LI-S4", "STAT-S4", "PU-S4"]
for slot, variant in enumerate(variants):
    stem = f"parallel_g{args.gpu}_n{args.concurrency}_s{slot}"
    output = root / f"{stem}.json"
    stream = (root / f"{stem}.log").open("w")
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=args.gpu, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", MPLCONFIGDIR="/tmp/posterior_s4_mpl")
    command = [sys.executable, "-u", str(EXP_ROOT / "scripts/validate_posterior_s4.py"),
               "--mode", "benchmark", "--variants", variant, "--batch-size", str(args.batch_size),
               "--workers", str(args.workers), "--steps", "100", "--warmup", "20", "--repeats", "1",
               "--output", str(output)]
    processes.append((subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, env=env), stream, output))
records = []
for process, stream, output in processes:
    code = process.wait()
    stream.close()
    if code:
        raise RuntimeError(f"Parallel preflight failed: {output.with_suffix('.log')}")
    records.extend(json.loads(output.read_text())["benchmarks"])
result = {"status": "passed", "gpu": args.gpu, "concurrency": args.concurrency,
          "aggregate_device_samples_per_second": sum(r["samples_per_second"] for r in records if r["mode"] == "device"),
          "aggregate_end_to_end_samples_per_second": sum(r["samples_per_second"] for r in records if r["mode"] == "end_to_end"),
          "records": records, "wall_seconds_including_startup": time.time() - start,
          "note": "short contention check, sums overlapping per-process rates; not a precise makespan benchmark"}
path = root / f"parallel_gpu{args.gpu}_n{args.concurrency}.json"
atomic_json_dump(result, path)
print(json.dumps({k: v for k, v in result.items() if k != "records"}), flush=True)
