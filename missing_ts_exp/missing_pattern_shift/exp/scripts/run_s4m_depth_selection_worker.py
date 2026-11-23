#!/usr/bin/env python3
"""Run an isolated shard of S4M depth selection and validation sensitivity screens."""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import time

from protocol import EXP_ROOT, atomic_json_dump


SUITE = "s4m_depth_selection_r50_l96_h96_official_sgd"
PYTHON = "/media/data1/wangzk/miniconda3/envs/hd-tts/bin/python"
RUNNER = EXP_ROOT / "scripts" / "run_official_baseline.py"
DIAGNOSTIC = EXP_ROOT / "scripts" / "diagnose_s4m_mask_path.py"


def complete(path: pathlib.Path) -> bool:
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("status") == "complete"
    except (FileNotFoundError, json.JSONDecodeError):
        return False


def invoke(command: list[str], log: pathlib.Path, env: dict[str, str]) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as stream:
        stream.write("\n$ " + " ".join(command) + "\n")
        stream.flush()
        subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True, env=env)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=pathlib.Path, required=True)
    parser.add_argument("--gpu", required=True, choices=[str(number) for number in range(8)])
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--shards", type=int, required=True)
    parser.add_argument("--status-tag", required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    jobs = [job for number, job in enumerate(manifest["jobs"])
            if number % args.shards == args.shard]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.gpu
    status = EXP_ROOT / "logs" / SUITE / f"{args.status_tag}.json"
    failures: list[dict[str, str]] = []
    for job in jobs:
        directory = EXP_ROOT / "results" / SUITE / job["id"]
        checkpoint = EXP_ROOT / "checkpoints" / SUITE / f"{job['id']}.pt"
        result = directory / "selection_complete.json"
        diagnostic = EXP_ROOT / "diagnostics" / SUITE / f"{job['id']}.json"
        log = EXP_ROOT / "logs" / SUITE / f"{job['id']}.log"
        try:
            if not complete(result):
                atomic_json_dump({"status": "training", "job": job, "gpu": args.gpu,
                                  "updated_at": time.time()}, status)
                command = [PYTHON, "-u", str(RUNNER), "--baseline", "S4M", "--dataset", job["dataset"],
                           "--train-family", "P", "--test-family", "P", "--block-len", "24",
                           "--train-seed", "2024", "--test-mask-seed", "9101", "--seq-len", "96",
                           "--pred-len", "96", "--missing-rate", "0.5", "--mask-balance", "per_channel",
                           "--s4m-e-layers", str(job["e_layers"]), "--epochs", "20", "--patience", "4",
                           "--s4m-optimizer", "official_sgd",
                           "--batch-size", "16", "--checkpoint-path", str(checkpoint), "--result-path", str(result),
                           "--progress-path", str(directory / "progress.json"), "--skip-test"]
                invoke(command, log, env)
            if not diagnostic.is_file():
                atomic_json_dump({"status": "diagnosing", "job": job, "gpu": args.gpu,
                                  "updated_at": time.time()}, status)
                invoke([PYTHON, "-u", str(DIAGNOSTIC), "--checkpoint", str(checkpoint), "--output", str(diagnostic),
                        "--dataset", job["dataset"], "--split", "val", "--seq-len", "96", "--pred-len", "96",
                        "--missing-rate", "0.5", "--block-len", "24", "--mask-seed", "9101", "--batch-size", "16",
                        "--s4m-e-layers", str(job["e_layers"])], log, env)
        except subprocess.CalledProcessError as error:
            failures.append({"job": job["id"], "returncode": str(error.returncode)})
            atomic_json_dump({"status": "failed", "job": job, "gpu": args.gpu, "failures": failures,
                              "updated_at": time.time()}, status)
    atomic_json_dump({"status": "complete", "gpu": args.gpu, "jobs_assigned": len(jobs), "failures": failures,
                      "updated_at": time.time()}, status)
    if failures:
        raise SystemExit(f"{len(failures)} depth-selection jobs failed")


if __name__ == "__main__":
    main()
