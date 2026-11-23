#!/usr/bin/env python3
"""Write durable three-hour status snapshots for interpolation baselines."""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import time

from protocol import EXP_ROOT, atomic_json_dump


SUITE = "r50_b24_l96_h96_per_channel_linear_interpolation_nw10"


def load(path: pathlib.Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=pathlib.Path, required=True)
    parser.add_argument("--gpus", nargs="+", default=["1", "2", "5"])
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    root = EXP_ROOT / "results" / SUITE
    train_complete = train_failed = test_complete = test_failed = active = 0
    for job in manifest["jobs"]:
        directory = root / job["id"]
        records = [directory / "training_complete.json"] + [
            directory / f"test_{test['test_family']}_b{test['block_len']}_m{test['test_mask_seed']}.json"
            for test in job["evaluations"]
        ]
        for path in records:
            record = load(path)
            if path.name == "training_complete.json":
                train_complete += bool(record and record.get("status") == "complete")
                train_failed += bool(record and record.get("status") == "failed")
            else:
                test_complete += bool(record and record.get("status") == "complete")
                test_failed += bool(record and record.get("status") == "failed")
        progress = load(directory / "progress.json")
        active += bool(progress and progress.get("status") == "running")
    try:
        gpu_lines = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu", "--format=csv,noheader,nounits"], text=True,
        ).strip().splitlines()
        gpu = [line for line in gpu_lines if line.split(",", 1)[0].strip() in set(args.gpus)]
    except (OSError, subprocess.CalledProcessError):
        gpu = ["unavailable"]
    workers = {path.stem: load(path) for path in (EXP_ROOT / "logs" / SUITE).glob("worker_*.json")}
    snapshot = {
        "timestamp": time.time(), "suite": SUITE,
        "training_runs_complete": train_complete, "training_runs_total": len(manifest["jobs"]),
        "training_runs_failed": train_failed, "test_evaluations_complete": test_complete,
        "test_evaluations_total": sum(len(job["evaluations"]) for job in manifest["jobs"]),
        "test_evaluations_failed": test_failed, "running_progress_records": active,
        "workers": workers, "gpu": gpu,
    }
    logs = EXP_ROOT / "logs" / SUITE
    logs.mkdir(parents=True, exist_ok=True)
    atomic_json_dump(snapshot, logs / "three_hour_status.json")
    with (logs / "three_hour_status.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(snapshot, ensure_ascii=False) + "\n")
    print(json.dumps(snapshot, ensure_ascii=False))


if __name__ == "__main__":
    main()
