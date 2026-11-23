#!/usr/bin/env python3
"""Write one durable, human-readable hourly status snapshot for the workers."""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import time

from protocol import EXP_ROOT, atomic_json_dump


def load(path: pathlib.Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=pathlib.Path,
                        default=EXP_ROOT / "configs" / "primary_jobs_2seeds.json")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    expected_tests: set[pathlib.Path] = set()
    expected_training: set[pathlib.Path] = set()
    for job in manifest["jobs"]:
        directory = EXP_ROOT / "results" / "primary" / job["id"]
        expected_training.add(directory / "training_complete.json")
        for test in job["evaluations"]:
            expected_tests.add(directory / f"test_{test['test_family']}_b{test['block_len']}_m{test['test_mask_seed']}.json")
    results = [path for path in (EXP_ROOT / "results" / "primary").glob("**/*.json")
               if path in expected_tests or path in expected_training or path.name == "progress.json"]
    # Keep the legacy aggregate counters, but also separate durable model
    # training artefacts from held-out test evaluations.  A job directory
    # contains ``training_complete.json`` and ``progress.json`` in addition
    # to one file per test mask, so using a single count as "experiments
    # completed" is otherwise misleading during a long run.
    complete = failed = running = 0
    test_complete = test_failed = 0
    train_complete = train_failed = 0
    for path in results:
        record = load(path)
        if not record:
            continue
        status = record.get("status")
        complete += status == "complete"
        failed += status == "failed"
        running += status == "running"
        if path.name.startswith("test_"):
            test_complete += status == "complete"
            test_failed += status == "failed"
        elif path.name == "training_complete.json":
            train_complete += status == "complete"
            train_failed += status == "failed"
    workers = {path.stem: load(path) for path in (EXP_ROOT / "logs" / "primary").glob("worker_gpu*.json")}
    try:
        gpu = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu", "--format=csv,noheader,nounits"], text=True
        ).strip().splitlines()
        gpu = [line for line in gpu if line.split(",", 1)[0].strip() in {"6", "7"}]
    except (OSError, subprocess.CalledProcessError):
        gpu = ["unavailable"]
    snapshot = {
        "timestamp": time.time(),
        # Backward-compatible aggregate counters.
        "results_complete": complete,
        "results_failed": failed,
        "progress_records": running,
        # Unambiguous counters for progress reporting.
        "test_evaluations_complete": test_complete,
        "test_evaluations_failed": test_failed,
        "training_runs_complete": train_complete,
        "training_runs_failed": train_failed,
        "worker": workers,
        "gpu_6_7": gpu,
    }
    out = EXP_ROOT / "logs" / "primary" / "hourly_status.json"
    atomic_json_dump(snapshot, out)
    with (EXP_ROOT / "logs" / "primary" / "hourly_status.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(snapshot, ensure_ascii=False) + "\n")
    print(json.dumps(snapshot, ensure_ascii=False))


if __name__ == "__main__":
    main()
