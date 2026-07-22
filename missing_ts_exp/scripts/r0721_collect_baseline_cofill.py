#!/usr/bin/env python3
"""Collect 0721 baseline/CoFILL logs into unified CSV files."""

from __future__ import annotations

import csv
import datetime as dt
import json
import pathlib
import re
from typing import Any


ROOT = pathlib.Path("/data/wangzuke/time-series-forecast-exp")
RESULTS = ROOT / "missing_ts_exp" / "results" / "0721_cofill_pguts_forecasting"
LOG_DIR = RESULTS / "raw_logs" / "baseline_cofill"
CSV_DIR = RESULTS / "csv"
MASK_DIR = ROOT / "dataset" / "0721_missing_masks"
BASELINE_CSV = CSV_DIR / "baseline_results.csv"
COFILL_CSV = CSV_DIR / "cofill_results.csv"
MAIN_CSV = CSV_DIR / "main_results.csv"

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
EXTERNAL_MASK_RE = re.compile(
    r"^\[external_mask\]\s+path=(?P<path>\S+)\s+actual=(?P<actual>[-+]?\d+(?:\.\d+)?)\s+shape=(?P<shape>.+)$",
    re.MULTILINE,
)
COFILL_METRICS_RE = re.compile(r"^\[cofill_metrics\]\s+(?P<json>\{.*\})$", re.MULTILINE)

FIELDNAMES = [
    "run_id",
    "experiment_line",
    "model",
    "variant",
    "source_code",
    "source_commit",
    "dataset",
    "num_nodes",
    "time_steps",
    "mask_type",
    "target_missing_rate",
    "actual_missing_rate",
    "mask_sha256",
    "T_in",
    "T_out",
    "seed",
    "batch_size",
    "effective_batch_size",
    "MAE",
    "RMSE_or_MSE",
    "MAPE_or_MRE",
    "epoch_time_sec",
    "train_time_sec",
    "gpu_peak_mb",
    "checkpoint_path",
    "prediction_path",
    "log_path",
    "status",
    "notes",
]


def read_manifest() -> dict[tuple[str, str, str], dict[str, str]]:
    rows = list(csv.DictReader((MASK_DIR / "manifest.csv").open(newline="")))
    return {
        (r["dataset"], r["missing_type"], f"{int(round(float(r['target_missing_rate']) * 100)):02d}"): r
        for r in rows
    }


def git_commit(path: pathlib.Path) -> str:
    import subprocess

    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return ""


def empty_row(path: pathlib.Path) -> dict[str, str]:
    return {k: "" for k in FIELDNAMES} | {
        "experiment_line": "baseline_cofill",
        "variant": "fair_h5_masked",
        "T_in": "24",
        "batch_size": "512",
        "effective_batch_size": "512",
        "log_path": str(path.relative_to(ROOT)),
        "status": "unknown",
    }


def infer_from_name(path: pathlib.Path) -> dict[str, str]:
    stem = path.stem
    out: dict[str, str] = {"run_id": stem}

    pattern = re.compile(
        r"^(?P<prefix>bitgraph|hdtts_amp|cofill(?:_forecast)?)_"
        r"(?P<dataset>Metr|PEMS)_"
        r"(?P<mask_type>point|block_t|block_st)_"
        r"r(?P<rate>\d{2})_h(?P<horizon>\d+)_"
        r"b(?P<batch>\d+)(?:_e(?P<epochs>\d+))?_s(?P<seed>\d+)$"
    )
    m = pattern.match(stem)
    if not m:
        return out

    prefix = m.group("prefix")
    if prefix == "bitgraph":
        out["model"] = "BiTGraph"
        out["source_code"] = "external_repro/BiTGraph"
    elif prefix == "hdtts_amp":
        out["model"] = "HD-TTS-AMP"
        out["source_code"] = "external_repro/hdtts"
    elif prefix == "cofill":
        out["model"] = "CoFILL"
        out["variant"] = "original_imputation"
        out["source_code"] = "external_repro/CoFILL"
    else:
        out["model"] = "CoFILL-Forecaster"
        out["variant"] = "forecasting_adaptation"
        out["source_code"] = "external_repro/CoFILL"

    out["dataset"] = m.group("dataset")
    out["mask_type"] = m.group("mask_type")
    out["target_missing_rate"] = f"0.{m.group('rate')}"
    out["rate_tag"] = m.group("rate")
    out["T_out"] = m.group("horizon")
    out["batch_size"] = m.group("batch")
    out["effective_batch_size"] = m.group("batch")
    out["seed"] = m.group("seed")
    return out


def parse_hdtts_meta(text: str) -> dict[str, Any]:
    matches = list(FAIR_META_RE.finditer(text))
    if not matches:
        return {}
    try:
        return json.loads(matches[-1].group("json"))
    except json.JSONDecodeError:
        return {}


def parse_json_line(regex: re.Pattern[str], text: str) -> dict[str, Any]:
    matches = list(regex.finditer(text))
    if not matches:
        return {}
    try:
        return json.loads(matches[-1].group("json"))
    except json.JSONDecodeError:
        return {}


def parse_log(path: pathlib.Path, manifest: dict[tuple[str, str, str], dict[str, str]]) -> dict[str, str]:
    text = path.read_text(errors="replace")
    row = empty_row(path)
    inferred = infer_from_name(path)
    rate_tag = inferred.pop("rate_tag", "")
    row.update(inferred)

    key = (row.get("dataset", ""), row.get("mask_type", ""), rate_tag)
    if key in manifest:
        meta = manifest[key]
        row["actual_missing_rate"] = meta["actual_missing_rate"]
        row["mask_sha256"] = meta["mask_sha256"]
        row["num_nodes"] = meta["n_nodes"]
        row["time_steps"] = meta["n_timesteps"]

    if row.get("source_code"):
        row["source_commit"] = git_commit(ROOT / row["source_code"])

    if "Traceback (most recent call last)" in text or "Error executing job" in text or "[r0721 cofill] missing" in text:
        row["status"] = "failed"

    ext_mask = EXTERNAL_MASK_RE.search(text)
    if ext_mask and not row["actual_missing_rate"]:
        row["actual_missing_rate"] = ext_mask.group("actual")

    bitgraph = BITGRAPH_RE.search(text)
    if bitgraph:
        row["status"] = "finished"
        row["MAE"] = bitgraph.group("mae")
        row["RMSE_or_MSE"] = bitgraph.group("rmse")
        row["MAPE_or_MRE"] = bitgraph.group("mape")
        ckpt = RESULTS / "checkpoints" / "baseline_cofill" / row["run_id"] / "best.pth"
        row["checkpoint_path"] = str(ckpt.relative_to(ROOT)) if ckpt.exists() else ""

    for metric in HDTTS_RE.finditer(text):
        row["status"] = "finished"
        name = metric.group("name")
        value = metric.group("value")
        if name == "test_mae":
            row["MAE"] = value
        elif name in ("test_mse", "test_loss"):
            row["RMSE_or_MSE"] = value
        elif name == "test_mre":
            row["MAPE_or_MRE"] = value

    hdtts_meta = parse_hdtts_meta(text)
    if hdtts_meta:
        row["T_in"] = str(hdtts_meta.get("window", row["T_in"]))
        row["T_out"] = str(hdtts_meta.get("horizon", row["T_out"]))
        row["batch_size"] = str(hdtts_meta.get("batch_size", row["batch_size"]))
        row["effective_batch_size"] = row["batch_size"]
        row["actual_missing_rate"] = f"{float(hdtts_meta.get('actual_missing_rate', row['actual_missing_rate'] or 0)):.6f}"
        ckpt_dir = RESULTS / "checkpoints" / "baseline_cofill" / row["run_id"]
        candidates = sorted(ckpt_dir.glob("*.ckpt"))
        row["checkpoint_path"] = str(candidates[-1].relative_to(ROOT)) if candidates else ""

    if row["status"] == "unknown" and "skipped optional unmasked test" in text:
        row["status"] = "finished"

    cofill_metrics = parse_json_line(COFILL_METRICS_RE, text)
    if cofill_metrics:
        row["status"] = str(cofill_metrics.get("status", "finished"))
        row["MAE"] = str(cofill_metrics.get("MAE", cofill_metrics.get("mae", row["MAE"])))
        row["RMSE_or_MSE"] = str(
            cofill_metrics.get("RMSE_or_MSE", cofill_metrics.get("MSE", cofill_metrics.get("mse", row["RMSE_or_MSE"])))
        )
        row["MAPE_or_MRE"] = str(
            cofill_metrics.get("MAPE_or_MRE", cofill_metrics.get("MRE", cofill_metrics.get("mre", row["MAPE_or_MRE"])))
        )
        row["epoch_time_sec"] = str(cofill_metrics.get("epoch_time_sec", row["epoch_time_sec"]))
        row["train_time_sec"] = str(cofill_metrics.get("train_time_sec", row["train_time_sec"]))
        row["gpu_peak_mb"] = str(cofill_metrics.get("gpu_peak_mb", row["gpu_peak_mb"]))
        row["checkpoint_path"] = str(cofill_metrics.get("checkpoint_path", row["checkpoint_path"]))
        row["prediction_path"] = str(cofill_metrics.get("prediction_path", row["prediction_path"]))
        row["notes"] = str(cofill_metrics.get("notes", row["notes"]))

    if row["status"] == "unknown" and path.exists():
        row["status"] = "running_or_incomplete"

    row["notes"] = dt.datetime.fromtimestamp(path.stat().st_mtime).strftime("log_mtime=%Y-%m-%d %H:%M:%S")
    return row


def write_csv(path: pathlib.Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {path} rows={len(rows)}")


def main() -> int:
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    manifest = read_manifest()
    logs = sorted(LOG_DIR.glob("*.log"))
    rows = [parse_log(path, manifest) for path in logs]
    baseline_rows = [r for r in rows if r["model"] in {"BiTGraph", "HD-TTS-AMP"}]
    cofill_rows = [r for r in rows if r["model"].startswith("CoFILL")]

    write_csv(BASELINE_CSV, baseline_rows)
    write_csv(COFILL_CSV, cofill_rows)

    main_rows = baseline_rows + cofill_rows
    write_csv(MAIN_CSV, main_rows)

    status_counts: dict[str, int] = {}
    for row in main_rows:
        status_counts[row["status"]] = status_counts.get(row["status"], 0) + 1
    print("status_counts", status_counts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
