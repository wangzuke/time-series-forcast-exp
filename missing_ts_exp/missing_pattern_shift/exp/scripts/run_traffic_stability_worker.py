#!/usr/bin/env python3
"""Run a resumable, isolated shard of the Traffic/S4M stability experiment."""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import time
from typing import Any

from protocol import EXP_ROOT, atomic_json_dump


SUITE = "s4m_traffic_electricity_stability_1115"
OFFICIAL = EXP_ROOT / "scripts" / "run_official_baseline.py"
INTERPOLATED = EXP_ROOT / "scripts" / "run_interpolated_baseline.py"
PYTHON = {
    "S4M": "/media/data1/wangzk/miniconda3/envs/hd-tts/bin/python",
    "ChannelTokenFormer": "/media/data1/wangzk/miniconda3/envs/itransformer/bin/python",
    "PatchTST": "/media/data1/wangzk/miniconda3/envs/itransformer/bin/python",
    "DLinear": "/media/data1/wangzk/miniconda3/envs/itransformer/bin/python",
    "iTransformer": "/media/data1/wangzk/miniconda3/envs/itransformer/bin/python",
}
BASELINES = tuple(PYTHON)


def completed(path: pathlib.Path) -> bool:
    try:
        return path.is_file() and json.loads(path.read_text(encoding="utf-8")).get("status") == "complete"
    except (OSError, json.JSONDecodeError):
        return False


def invoke(command: list[str], log: pathlib.Path, env: dict[str, str]) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as stream:
        stream.write("\n$ " + " ".join(command) + "\n")
        stream.flush()
        subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True, env=env)


def command(job: dict[str, Any], test: dict[str, Any], checkpoint: pathlib.Path,
            result: pathlib.Path, progress: pathlib.Path) -> list[str]:
    baseline = job["baseline"]
    runner = OFFICIAL if baseline in {"S4M", "ChannelTokenFormer"} else INTERPOLATED
    args = [
        PYTHON[baseline], "-u", str(runner), "--baseline", baseline, "--dataset", job["dataset"],
        "--train-family", job["train_family"], "--test-family", test["test_family"],
        "--block-len", "24", "--train-seed", str(job["train_seed"]),
        "--test-mask-seed", str(test["test_mask_seed"]), "--seq-len", "96", "--pred-len", "96",
        "--missing-rate", "0.5", "--mask-balance", "per_channel",
        "--checkpoint-path", str(checkpoint), "--result-path", str(result),
        "--progress-path", str(progress),
    ]
    if baseline == "S4M":
        args += ["--s4m-e-layers", "4", "--s4m-optimizer", "official_sgd",
                 "--s4m-ablation", job.get("ablation", "none"),
                 "--batch-size", "16", "--epochs", "20", "--patience", "4"]
    elif baseline == "ChannelTokenFormer":
        args += ["--batch-size", "4", "--gradient-accumulation-steps", "8",
                 "--epochs", "10", "--patience", "3"]
    return args


def run_job(job: dict[str, Any], gpu: str, status_path: pathlib.Path) -> None:
    directory = EXP_ROOT / "results" / SUITE / job["id"]
    checkpoint = EXP_ROOT / "checkpoints" / SUITE / f"{job['id']}.pt"
    training_result = directory / "training_complete.json"
    progress = directory / "progress.json"
    log = EXP_ROOT / "logs" / SUITE / f"{job['id']}.log"
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu
    train_eval = {"test_family": "P", "block_len": 24, "test_mask_seed": 9101}
    if not completed(training_result):
        atomic_json_dump({"status": "training", "job": job, "gpu": gpu, "updated_at": time.time()}, status_path)
        invoke(command(job, train_eval, checkpoint, training_result, progress), log, env)
    for test in job["evaluations"]:
        output = directory / f"test_{test['test_family']}_b24_m{test['test_mask_seed']}.json"
        if completed(output):
            continue
        atomic_json_dump({"status": "evaluating", "job": job, "test": test,
                          "gpu": gpu, "updated_at": time.time()}, status_path)
        invoke(command(job, test, checkpoint, output, progress) + ["--eval-only"], log, env)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=pathlib.Path, required=True)
    parser.add_argument("--gpu", choices=[str(i) for i in range(8)], required=True)
    parser.add_argument("--baseline", choices=BASELINES, required=True)
    parser.add_argument("--phase", choices=["traffic", "electricity_reference", "electricity_ablation"], default=None)
    parser.add_argument("--job-ids", nargs="*", default=None)
    parser.add_argument("--status-tag", required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    jobs = [job for job in manifest["jobs"] if job["baseline"] == args.baseline]
    if args.phase:
        jobs = [job for job in jobs if job["phase"] == args.phase]
    if args.job_ids is not None:
        requested = set(args.job_ids)
        unknown = requested - {job["id"] for job in jobs}
        if unknown:
            raise ValueError(f"unknown jobs for this selection: {sorted(unknown)}")
        jobs = [job for job in jobs if job["id"] in requested]
    status_path = EXP_ROOT / "logs" / SUITE / f"{args.status_tag}.json"
    failures = []
    for job in jobs:
        try:
            run_job(job, args.gpu, status_path)
        except subprocess.CalledProcessError as error:
            failures.append({"job": job["id"], "returncode": error.returncode})
            atomic_json_dump({"status": "failed", "job": job, "gpu": args.gpu,
                              "failures": failures, "updated_at": time.time()}, status_path)
            # A failing shape/configuration often affects every run of one
            # model. Stop this model shard so its failure can be reviewed.
            break
    atomic_json_dump({"status": "complete" if not failures else "failed", "gpu": args.gpu,
                      "baseline": args.baseline, "phase": args.phase,
                      "jobs_assigned": len(jobs), "failures": failures, "updated_at": time.time()}, status_path)
    if failures:
        raise SystemExit(f"{len(failures)} failed job(s)")


if __name__ == "__main__":
    main()
