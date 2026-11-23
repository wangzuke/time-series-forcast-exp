#!/usr/bin/env python3
"""Validate interpolation-baseline artifacts against their registered 2x2 manifest."""
from __future__ import annotations

import argparse
import json
import math
import pathlib

from protocol import EXP_ROOT, atomic_json_dump


SUITE = "r50_b24_l96_h96_per_channel_linear_interpolation_nw10"


def finite(value: object) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def expected(manifest: dict) -> dict[pathlib.Path, dict]:
    paths: dict[pathlib.Path, dict] = {}
    for job in manifest["jobs"]:
        folder = EXP_ROOT / "results" / SUITE / job["id"]
        for test in job["evaluations"]:
            paths[folder / f"test_{test['test_family']}_b{test['block_len']}_m{test['test_mask_seed']}.json"] = {
                **job, **test,
            }
    return paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=pathlib.Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    planned = expected(manifest)
    errors: list[dict[str, str]] = []
    completed = 0
    seen: set[tuple] = set()
    for path, spec in planned.items():
        if not path.exists():
            continue
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            errors.append({"path": str(path), "error": f"invalid JSON: {error}"})
            continue
        if record.get("status") != "complete":
            errors.append({"path": str(path), "error": f"unexpected status: {record.get('status')!r}"})
            continue
        for key in ("baseline", "dataset", "seq_len", "pred_len", "train_family", "test_family",
                    "block_len", "train_seed", "test_mask_seed", "missing_rate", "mask_balance"):
            if record.get(key) != spec.get(key):
                errors.append({"path": str(path), "error": f"{key}: {record.get(key)!r} != {spec.get(key)!r}"})
        if record.get("data_loader_num_workers") != manifest.get("data_loader_num_workers"):
            errors.append({"path": str(path), "error": "DataLoader worker count differs from registered official default"})
        metrics = record.get("test", {})
        if not all(finite(metrics.get(key)) for key in ("objective", "mae_raw", "mse_raw")):
            errors.append({"path": str(path), "error": "missing or non-finite test metric"})
        preprocessing = record.get("preprocessing", {})
        if not (preprocessing.get("input_finite") and preprocessing.get("output_finite") and
                preprocessing.get("observed_values_unchanged") and
                abs(float(preprocessing.get("observed_fraction", float("nan"))) - 0.5) < 1e-9):
            errors.append({"path": str(path), "error": "linear-interpolation audit failed"})
        for split in ("train", "val", "test"):
            audit = record.get("mask_audit", {}).get(split, {})
            if (not finite(audit.get("actual_rate_mean")) or
                    abs(float(audit["actual_rate_mean"]) - 0.5) > 1e-9 or
                    audit.get("mask_balance") != "per_channel" or
                    abs(float(audit.get("missing_per_channel_mean", float("nan"))) - 0.5) > 1e-9):
                errors.append({"path": str(path), "error": f"{split} mask audit is not exact 50% per channel"})
        logical_key = tuple(record.get(key) for key in (
            "baseline", "dataset", "pred_len", "train_family", "test_family", "block_len",
            "train_seed", "test_mask_seed",
        ))
        if logical_key in seen:
            errors.append({"path": str(path), "error": "duplicate logical evaluation"})
        seen.add(logical_key)
        completed += 1
    report = {
        "suite": SUITE,
        "planned_test_evaluations": len(planned),
        "completed_test_evaluations": completed,
        "missing_test_evaluations": len(planned) - completed,
        "integrity_errors": len(errors),
        "errors": errors,
    }
    out = EXP_ROOT / "summary" / SUITE / "validation.json"
    atomic_json_dump(report, out)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(f"validation found {len(errors)} integrity error(s)")


if __name__ == "__main__":
    main()
