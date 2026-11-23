#!/usr/bin/env python3
"""Run an isolated shard of the r=50%, L=H=96, per-channel-balanced suite."""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import time
from typing import Any

from protocol import EXP_ROOT, atomic_json_dump


SUITE = "r50_b24_l96_h96_per_channel"
RUNNER = EXP_ROOT / "scripts" / "run_official_baseline.py"
PYTHON = {
    "S4M": "/media/data1/wangzk/miniconda3/envs/hd-tts/bin/python",
    "BiTGraph": "/media/data1/wangzk/miniconda3/envs/itransformer/bin/python",
    "ChannelTokenFormer": "/media/data1/wangzk/miniconda3/envs/itransformer/bin/python",
}


def completed(path: pathlib.Path) -> bool:
    try:
        return path.is_file() and json.loads(path.read_text(encoding="utf-8")).get("status") == "complete"
    except json.JSONDecodeError:
        return False


def invoke(command: list[str], log: pathlib.Path, env: dict[str, str]) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as stream:
        stream.write("\n$ " + " ".join(command) + "\n")
        stream.flush()
        subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True, env=env)


def command(job: dict[str, Any], test: dict[str, Any], checkpoint: pathlib.Path, result: pathlib.Path,
            progress: pathlib.Path) -> list[str]:
    if job["baseline"] == "S4M":
        batch_size, epochs, patience = 16, 20, 4
    elif job["baseline"] == "BiTGraph":
        batch_size, epochs, patience = 32, 200, 201
    else:
        batch_size, epochs, patience = 32, 10, 3
    invocation = [
        PYTHON[job["baseline"]], "-u", str(RUNNER), "--baseline", job["baseline"], "--dataset", job["dataset"],
        "--train-family", job["train_family"], "--test-family", test["test_family"],
        "--block-len", str(test["block_len"]), "--train-seed", str(job["train_seed"]),
        "--test-mask-seed", str(test["test_mask_seed"]), "--seq-len", str(job["seq_len"]),
        "--pred-len", str(job["pred_len"]), "--missing-rate", str(job["missing_rate"]),
        "--mask-balance", job["mask_balance"], "--epochs", str(epochs), "--patience", str(patience),
        "--batch-size", str(batch_size), "--checkpoint-path", str(checkpoint), "--result-path", str(result),
        "--progress-path", str(progress),
    ]
    if job["baseline"] == "S4M":
        # The isolated validation selection selected e=4 for ETTh1, Weather,
        # and Electricity.  Use the authors' executable SGD configuration,
        # which passed the Electricity finite-value replay; do not reuse the
        # earlier Adam-based diagnostic suite.
        invocation += ["--s4m-e-layers", "4", "--s4m-optimizer", "official_sgd"]
    return invocation


def run_job(job: dict[str, Any], gpu: str, status_path: pathlib.Path) -> None:
    directory = EXP_ROOT / "results" / SUITE / job["id"]
    checkpoint = EXP_ROOT / "checkpoints" / SUITE / f"{job['id']}.pt"
    training_result = directory / "training_complete.json"
    progress = directory / "progress.json"
    log = EXP_ROOT / "logs" / SUITE / f"{job['id']}.log"
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu
    training_test = {"test_family": "P", "block_len": job["train_block_len"], "test_mask_seed": 9101}
    if not completed(training_result):
        atomic_json_dump({"status": "training", "job": job, "gpu": gpu, "updated_at": time.time()}, status_path)
        invoke(command(job, training_test, checkpoint, training_result, progress), log, env)
    for test in job["evaluations"]:
        result = directory / f"test_{test['test_family']}_b{test['block_len']}_m{test['test_mask_seed']}.json"
        if completed(result):
            continue
        atomic_json_dump({"status": "evaluating", "job": job, "test": test, "gpu": gpu, "updated_at": time.time()}, status_path)
        invoke(command(job, test, checkpoint, result, progress) + ["--eval-only"], log, env)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=pathlib.Path, required=True)
    parser.add_argument("--gpu", choices=[str(index) for index in range(8)], required=True)
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--shards", type=int, required=True)
    parser.add_argument("--baseline", choices=list(PYTHON), required=True)
    parser.add_argument("--status-tag", type=str, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    selected = [job for job in manifest["jobs"] if job["baseline"] == args.baseline]
    jobs = [job for number, job in enumerate(selected) if number % args.shards == args.shard]
    status = EXP_ROOT / "logs" / SUITE / f"{args.status_tag}.json"
    failures: list[dict[str, Any]] = []
    for job in jobs:
        try:
            run_job(job, args.gpu, status)
        except subprocess.CalledProcessError as error:
            failures.append({"job": job["id"], "returncode": error.returncode})
            atomic_json_dump({"status": "failed", "job": job, "gpu": args.gpu, "failures": failures,
                              "updated_at": time.time()}, status)
    atomic_json_dump({"status": "complete", "gpu": args.gpu, "jobs_assigned": len(jobs), "failures": failures,
                      "updated_at": time.time()}, status)
    if failures:
        raise SystemExit(f"{len(failures)} failed job(s)")


if __name__ == "__main__":
    main()
