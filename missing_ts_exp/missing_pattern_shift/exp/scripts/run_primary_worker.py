#!/usr/bin/env python3
"""Run one deterministic shard of the primary job manifest on a single GPU."""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys
import time
from typing import Any

from protocol import EXP_ROOT, atomic_json_dump


RUNNER = EXP_ROOT / "scripts" / "run_official_baseline.py"
PYTHON = {
    "S4M": "/media/data1/wangzk/miniconda3/envs/hd-tts/bin/python",
    "BiTGraph": "/media/data1/wangzk/miniconda3/envs/itransformer/bin/python",
    "ChannelTokenFormer": "/media/data1/wangzk/miniconda3/envs/itransformer/bin/python",
}


def result_path(job: dict[str, Any], test: dict[str, Any]) -> pathlib.Path:
    return EXP_ROOT / "results" / "primary" / job["id"] / f"test_{test['test_family']}_b{test['block_len']}_m{test['test_mask_seed']}.json"


def completed(path: pathlib.Path) -> bool:
    if not path.is_file():
        return False
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("status") == "complete"
    except json.JSONDecodeError:
        return False


def invoke(command: list[str], log: pathlib.Path, env: dict[str, str]) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as stream:
        stream.write("\n$ " + " ".join(command) + "\n")
        stream.flush()
        subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True, env=env)


def common(job: dict[str, Any], test: dict[str, Any], checkpoint: pathlib.Path, output: pathlib.Path, progress: pathlib.Path) -> list[str]:
    if job["baseline"] == "S4M":
        batch_size, epochs, patience = 16, 20, 4
    elif job["baseline"] == "BiTGraph":
        # ICLR'24 reports batch 32; the authors' README trains for 200 epochs.
        # The source has no active early-stop branch, so patience > epochs keeps
        # our generic loop behaviorally equivalent.
        batch_size, epochs, patience = 32, 200, 201
    else:
        # CTF Appendix A.2 and run.py defaults.  The local batched-missing
        # extension was verified against the released implementation at B=1.
        batch_size, epochs, patience = 32, 10, 3
    return [
        PYTHON[job["baseline"]], "-u", str(RUNNER), "--baseline", job["baseline"], "--dataset", job["dataset"],
        "--train-family", job["train_family"], "--test-family", test["test_family"],
        "--block-len", str(test["block_len"]), "--train-seed", str(job["train_seed"]),
        "--test-mask-seed", str(test["test_mask_seed"]), "--pred-len", str(job["pred_len"]),
        "--epochs", str(epochs), "--patience", str(patience), "--batch-size", str(batch_size),
        "--checkpoint-path", str(checkpoint), "--result-path", str(output), "--progress-path", str(progress),
    ]


def run_job(job: dict[str, Any], gpu: str, status_path: pathlib.Path) -> None:
    directory = EXP_ROOT / "results" / "primary" / job["id"]
    checkpoint = EXP_ROOT / "checkpoints" / "primary" / f"{job['id']}.pt"
    training_result = directory / "training_complete.json"
    progress = directory / "progress.json"
    log = EXP_ROOT / "logs" / "primary" / f"{job['id']}.log"
    env = os.environ.copy(); env["CUDA_VISIBLE_DEVICES"] = gpu
    training_test = {"test_family": "P", "block_len": job["train_block_len"], "test_mask_seed": 9101}
    if not completed(training_result):
        state = {"status": "training", "job": job, "gpu": gpu, "updated_at": time.time()}
        atomic_json_dump(state, status_path)
        invoke(common(job, training_test, checkpoint, training_result, progress), log, env)
    for test in job["evaluations"]:
        output = result_path(job, test)
        if completed(output):
            continue
        state = {"status": "evaluating", "job": job, "test": test, "gpu": gpu, "updated_at": time.time()}
        atomic_json_dump(state, status_path)
        invoke(common(job, test, checkpoint, output, progress) + ["--eval-only"], log, env)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=pathlib.Path, required=True)
    parser.add_argument("--gpu", choices=[str(index) for index in range(8)], required=True,
                        help="physical GPU exposed through CUDA_VISIBLE_DEVICES")
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--shards", type=int, default=2)
    parser.add_argument("--baselines", nargs="+", choices=["S4M", "BiTGraph", "ChannelTokenFormer"],
                        help="optionally retain only these baselines; preserves manifest job ids")
    parser.add_argument("--limit", type=int, default=0, help="0 means all jobs in this shard")
    parser.add_argument("--status-tag", type=str, default=None,
                        help="status filename stem; required when multiple workers share one GPU")
    args = parser.parse_args()
    payload = json.loads(args.manifest.read_text(encoding="utf-8"))
    selected = [job for job in payload["jobs"] if not args.baselines or job["baseline"] in args.baselines]
    jobs = [job for number, job in enumerate(selected) if number % args.shards == args.shard]
    if args.limit:
        jobs = jobs[:args.limit]
    status_stem = args.status_tag or f"worker_gpu{args.gpu}"
    status_path = EXP_ROOT / "logs" / "primary" / f"{status_stem}.json"
    failures: list[dict[str, Any]] = []
    for number, job in enumerate(jobs, start=1):
        try:
            run_job(job, args.gpu, status_path)
        except subprocess.CalledProcessError as error:
            failures.append({"job": job["id"], "returncode": error.returncode})
            atomic_json_dump({"status": "failed", "job": job, "gpu": args.gpu, "failures": failures, "updated_at": time.time()}, status_path)
    atomic_json_dump({"status": "complete", "gpu": args.gpu, "jobs_assigned": len(jobs), "failures": failures, "updated_at": time.time()}, status_path)
    if failures:
        raise SystemExit(f"worker {args.gpu} finished with {len(failures)} failed job(s)")


if __name__ == "__main__":
    main()
