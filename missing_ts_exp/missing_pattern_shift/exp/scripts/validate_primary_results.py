#!/usr/bin/env python3
"""Validate durable primary results against the registered job manifest."""
from __future__ import annotations

import json
import math
import pathlib
import argparse

from protocol import EXP_ROOT, atomic_json_dump


def expected_results(jobs: list[dict]) -> dict[pathlib.Path, dict]:
    expected: dict[pathlib.Path, dict] = {}
    for job in jobs:
        directory = EXP_ROOT / "results" / "primary" / job["id"]
        for test in job["evaluations"]:
            path = directory / f"test_{test['test_family']}_b{test['block_len']}_m{test['test_mask_seed']}.json"
            expected[path] = {**job, **test}
    return expected


def finite(value: object) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=pathlib.Path,
                        default=EXP_ROOT / "configs" / "primary_jobs_2seeds.json")
    parser.add_argument("--baseline", action="append", dest="baselines",
                        help="Only validate this baseline; repeat for multiple baselines.")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    jobs = manifest["jobs"]
    if args.baselines:
        requested = set(args.baselines)
        jobs = [job for job in jobs if job["baseline"] in requested]
    expected = expected_results(jobs)
    errors: list[dict[str, str]] = []
    completed: list[pathlib.Path] = []
    seen_keys: set[tuple] = set()
    for path, spec in expected.items():
        if not path.is_file():
            continue
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            errors.append({"path": str(path), "error": f"invalid JSON: {error}"})
            continue
        if record.get("status") != "complete":
            errors.append({"path": str(path), "error": f"unexpected status {record.get('status')!r}"})
            continue
        for key in ("baseline", "dataset", "pred_len", "train_family", "test_family", "block_len", "train_seed", "test_mask_seed"):
            if record.get(key) != spec.get(key):
                errors.append({"path": str(path), "error": f"{key}: result={record.get(key)!r}, manifest={spec.get(key)!r}"})
        test = record.get("test", {})
        if not all(finite(test.get(metric)) for metric in ("mae_raw", "mse_raw", "objective")):
            errors.append({"path": str(path), "error": "non-finite or missing test metric"})
        key = tuple(record.get(field) for field in ("baseline", "dataset", "pred_len", "train_family", "test_family", "block_len", "train_seed", "test_mask_seed"))
        if key in seen_keys:
            errors.append({"path": str(path), "error": "duplicate logical evaluation key"})
        seen_keys.add(key)
        # Exact count matching permits one-element discretization error only.
        for split, audit in record.get("mask_audit", {}).items():
            rate = audit.get("actual_rate_mean")
            if not finite(rate) or abs(float(rate) - 0.30) > 0.002:
                errors.append({"path": str(path), "error": f"{split} mask rate outside tolerance: {rate!r}"})
        completed.append(path)
    # Older 3--5 seed artifacts remain durable in the shared result directory,
    # but are explicitly excluded from the amended two-seed primary analysis.
    # They are not integrity failures of the new preregistered manifest.
    all_results = set((EXP_ROOT / "results" / "primary").glob("**/test_*.json"))
    excluded = sorted(str(path) for path in all_results - set(expected))
    report = {
        "planned_test_evaluations": len(expected),
        "completed_test_evaluations": len(completed),
        "missing_test_evaluations": len(expected) - len(completed),
        "integrity_errors": len(errors),
        "errors": errors,
        "excluded_archived_test_evaluations": len(excluded),
    }
    out = EXP_ROOT / "summary" / "primary_validation.json"
    atomic_json_dump(report, out)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(f"primary validation found {len(errors)} integrity errors")


if __name__ == "__main__":
    main()
