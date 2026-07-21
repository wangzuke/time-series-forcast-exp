#!/usr/bin/env python3
"""Collect 0720 fair-experiment metrics from raw logs."""

from __future__ import annotations

import csv
import datetime as dt
import pathlib
import re


ROOT = pathlib.Path("/data/wangzuke/time-series-forecast-exp")
RESULTS = ROOT / "missing_ts_exp" / "results" / "0720_fair_b512"
LOG_DIR = RESULTS / "raw_logs"
CSV_DIR = RESULTS / "csv"
FAIR_DIR = RESULTS / "fair_data"
OUT_CSV = CSV_DIR / "fair_metrics_summary.csv"

BITGRAPH_RE = re.compile(
    r"loss,RMSE,MAPE\s+"
    r"(?P<mae>[-+]?\d+(?:\.\d+)?)\s*&\s*"
    r"(?P<rmse>[-+]?\d+(?:\.\d+)?)\s*&\s*"
    r"(?P<mape>[-+]?\d+(?:\.\d+)?)"
)
HDTTS_RE = re.compile(
    r"^\s*(?P<name>test_(?:loss|mae|mre|mse|mae_unmasked|mre_unmasked|mse_unmasked))\s+"
    r"(?P<value>[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s*$",
    re.MULTILINE,
)
FAIR_META_RE = re.compile(r"^\[fair_h5\]\s+(?P<json>\{.*\})$", re.MULTILINE)


def read_manifest() -> dict[tuple[str, str, str], dict[str, str]]:
    manifest = FAIR_DIR / "manifest.csv"
    if not manifest.exists():
        return {}
    with manifest.open(newline="") as f:
        rows = list(csv.DictReader(f))
    return {
        (r["dataset"], r["missing_type"], f"{int(float(r['target_missing_rate']) * 100):02d}"): r
        for r in rows
    }


def infer_from_name(path: pathlib.Path) -> dict[str, str]:
    stem = path.stem
    parts = stem.split("_")
    out: dict[str, str] = {}
    # bitgraph_Metr_point_r30_h24_b512_e200
    # hdtts_amp_Metr_point_r30_h24_b512_e200
    if stem.startswith("bitgraph_") and len(parts) >= 4:
        out["model"] = "BiTGraph"
        out["dataset"] = parts[1]
        out["missing_type"] = parts[2]
    elif stem.startswith("hdtts_amp_") and len(parts) >= 5:
        out["model"] = "HD-TTS-AMP"
        out["dataset"] = parts[2]
        out["missing_type"] = parts[3]
    m = re.search(r"_r(?P<rate>\d{2})_", stem)
    if m:
        out["target_missing_rate"] = f"0.{m.group('rate')}"
        out["rate_tag"] = m.group("rate")
    m = re.search(r"_h(?P<horizon>\d+)_", stem)
    if m:
        out["horizon"] = m.group("horizon")
    m = re.search(r"_b(?P<batch>\d+)_", stem)
    if m:
        out["batch_size"] = m.group("batch")
    m = re.search(r"_e(?P<epochs>\d+)$", stem)
    if m:
        out["epochs"] = m.group("epochs")
    return out


def parse_log(path: pathlib.Path, manifest: dict[tuple[str, str, str], dict[str, str]]) -> dict[str, str]:
    text = path.read_text(errors="replace")
    row = {
        "log_file": str(path.relative_to(ROOT)),
        "model": "",
        "dataset": "",
        "missing_type": "",
        "target_missing_rate": "",
        "actual_missing_rate": "",
        "window": "24",
        "horizon": "24",
        "batch_size": "512",
        "epochs": "",
        "status": "unknown",
        "mae": "",
        "rmse": "",
        "mape": "",
        "test_loss": "",
        "test_mae": "",
        "test_mre": "",
        "test_mse": "",
        "test_mae_unmasked": "",
        "test_mre_unmasked": "",
        "test_mse_unmasked": "",
        "data_path": "",
        "mask_path": "",
        "mask_sha256": "",
        "seed": "2024",
        "log_mtime": dt.datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
    }
    inferred = infer_from_name(path)
    rate_tag = inferred.pop("rate_tag", "")
    row.update(inferred)

    key = (row.get("dataset", ""), row.get("missing_type", ""), rate_tag)
    if key in manifest:
        meta = manifest[key]
        row["actual_missing_rate"] = meta["actual_missing_rate"]
        row["data_path"] = meta["data_path"]
        row["mask_path"] = meta["mask_path"]
        row["mask_sha256"] = meta["mask_sha256"]

    if "Traceback (most recent call last)" in text or "Error executing job" in text:
        row["status"] = "failed"

    m = BITGRAPH_RE.search(text)
    if m:
        row["status"] = "finished"
        row["mae"] = m.group("mae")
        row["rmse"] = m.group("rmse")
        row["mape"] = m.group("mape")

    for metric in HDTTS_RE.finditer(text):
        row["status"] = "finished"
        row[metric.group("name")] = metric.group("value")

    return row


def main() -> int:
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    manifest = read_manifest()
    rows = [
        parse_log(path, manifest)
        for path in sorted(LOG_DIR.glob("*.log"))
        if path.name.startswith(("bitgraph_", "hdtts_amp_"))
    ]
    fieldnames = [
        "log_file", "model", "dataset", "missing_type", "target_missing_rate",
        "actual_missing_rate", "window", "horizon", "batch_size", "epochs",
        "status", "mae", "rmse", "mape", "test_loss", "test_mae",
        "test_mre", "test_mse", "test_mae_unmasked", "test_mre_unmasked",
        "test_mse_unmasked", "data_path", "mask_path", "mask_sha256", "seed",
        "log_mtime",
    ]
    with OUT_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {OUT_CSV} rows={len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
