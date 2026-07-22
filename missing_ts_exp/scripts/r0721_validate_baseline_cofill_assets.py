#!/usr/bin/env python3
"""Validate the 0721 baseline/CoFILL shared data assets.

This is intentionally read-only. It checks that all experiments in the
baseline/CoFILL line can consume the same canonical HDF5 files, observed-mask
bundle, and split metadata prepared for 0721.
"""

from __future__ import annotations

import csv
import hashlib
import json
import pathlib
import sys
from collections import defaultdict

import numpy as np


ROOT = pathlib.Path("/data/wangzuke/time-series-forecast-exp")
MASK_DIR = ROOT / "dataset" / "0721_missing_masks"
RESULTS = ROOT / "missing_ts_exp" / "results" / "0721_cofill_pguts_forecasting"
NOTES = RESULTS / "notes"

EXPECTED_SHAPES = {
    "Metr": (34272, 207, 1),
    "PEMS": (52116, 325, 1),
}
EXPECTED_MASKS = {
    ("Metr", "point", "0.50"),
    ("Metr", "point", "0.70"),
    ("Metr", "block_t", "0.50"),
    ("Metr", "block_t", "0.70"),
    ("Metr", "block_t", "0.90"),
    ("Metr", "block_st", "0.50"),
    ("Metr", "block_st", "0.70"),
    ("Metr", "block_st", "0.90"),
    ("PEMS", "point", "0.50"),
    ("PEMS", "point", "0.70"),
    ("PEMS", "block_t", "0.50"),
    ("PEMS", "block_t", "0.70"),
    ("PEMS", "block_t", "0.90"),
    ("PEMS", "block_st", "0.50"),
    ("PEMS", "block_st", "0.70"),
    ("PEMS", "block_st", "0.90"),
}


def sha256_file(path: pathlib.Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def norm_rate(value: str) -> str:
    return f"{float(value):.2f}"


def fail(message: str, errors: list[str]) -> None:
    print(f"[FAIL] {message}")
    errors.append(message)


def main() -> int:
    NOTES.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []
    lines: list[str] = [
        "# 0721 baseline / CoFILL asset validation",
        "",
        f"- root: `{ROOT}`",
        f"- mask_dir: `{MASK_DIR}`",
        "",
    ]

    manifest_path = MASK_DIR / "manifest.csv"
    if not manifest_path.exists():
        fail(f"missing manifest: {manifest_path}", errors)
        return 1

    rows = list(csv.DictReader(manifest_path.open(newline="")))
    print(f"[validate] manifest rows={len(rows)}")
    lines.append(f"- manifest rows: {len(rows)}")
    if len(rows) != 16:
        fail(f"manifest should contain 16 rows, got {len(rows)}", errors)

    seen: set[tuple[str, str, str]] = set()
    per_dataset = defaultdict(int)
    for row in rows:
        dataset = row["dataset"]
        missing_type = row["missing_type"]
        rate = norm_rate(row["target_missing_rate"])
        key = (dataset, missing_type, rate)
        seen.add(key)
        per_dataset[dataset] += 1

        if row.get("mask_convention") != "1=observed,0=missing":
            fail(f"{key} mask_convention is not 1=observed,0=missing", errors)

        data_path = pathlib.Path(row["data_path"])
        if not data_path.exists():
            fail(f"{key} data_path missing: {data_path}", errors)

        mask_path = pathlib.Path(row["mask_path"])
        if not mask_path.exists():
            fail(f"{key} mask_path missing: {mask_path}", errors)
            continue

        mask = np.load(mask_path)
        if mask.shape != EXPECTED_SHAPES.get(dataset):
            fail(f"{key} shape {mask.shape} != {EXPECTED_SHAPES.get(dataset)}", errors)

        unique_values = set(np.unique(mask).tolist())
        if not unique_values.issubset({0, 1, False, True}):
            fail(f"{key} mask has non-binary values: {sorted(unique_values)[:10]}", errors)

        actual_missing = 1.0 - float(mask.mean())
        manifest_actual = float(row["actual_missing_rate"])
        target = float(row["target_missing_rate"])
        if abs(actual_missing - manifest_actual) > 1e-6:
            fail(
                f"{key} actual_missing {actual_missing:.6f} != manifest {manifest_actual:.6f}",
                errors,
            )
        if abs(actual_missing - target) > 0.005:
            fail(f"{key} actual_missing {actual_missing:.6f} deviates from target {target:.2f}", errors)

        digest = sha256_file(mask_path)
        if digest != row["mask_sha256"]:
            fail(f"{key} sha256 mismatch: {digest} != {row['mask_sha256']}", errors)

    missing = EXPECTED_MASKS - seen
    extra = seen - EXPECTED_MASKS
    if missing:
        fail(f"missing manifest keys: {sorted(missing)}", errors)
    if extra:
        fail(f"unexpected manifest keys: {sorted(extra)}", errors)

    npy_count = len(list(MASK_DIR.glob("mask_observed_*.npy")))
    print(f"[validate] mask npy files={npy_count}")
    lines.append(f"- mask npy files: {npy_count}")
    if npy_count != 16:
        fail(f"expected 16 mask .npy files, got {npy_count}", errors)

    for dataset in ("Metr", "PEMS"):
        for horizon in (12, 24):
            split_path = MASK_DIR / f"split_{dataset}_h{horizon}.json"
            if not split_path.exists():
                fail(f"missing split metadata: {split_path}", errors)
                continue
            split = json.loads(split_path.read_text())
            for k in ("window", "horizon", "stride", "total_windows", "train_len", "val_len", "test_len"):
                if k not in split:
                    fail(f"{split_path.name} missing key {k}", errors)
            if split.get("window") != 24 or split.get("horizon") != horizon or split.get("stride") != 1:
                fail(f"{split_path.name} has wrong window/horizon/stride: {split}", errors)
            total = int(split.get("total_windows", -1))
            train = int(split.get("train_len", -1))
            val = int(split.get("val_len", -1))
            test = int(split.get("test_len", -1))
            if train + val + test != total:
                fail(f"{split_path.name} split lengths do not sum to total", errors)
            print(f"[validate] {split_path.name}: total={total} train={train} val={val} test={test}")

    lines.extend(
        [
            f"- rows by dataset: `{dict(per_dataset)}`",
            f"- status: `{'failed' if errors else 'passed'}`",
            "",
        ]
    )
    if errors:
        lines.append("## Errors")
        lines.extend(f"- {e}" for e in errors)
    else:
        lines.append("All required 0721 baseline/CoFILL assets passed validation.")

    out = NOTES / "baseline_cofill_asset_validation.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[validate] wrote {out}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
