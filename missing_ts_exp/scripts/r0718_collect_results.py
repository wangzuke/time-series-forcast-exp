#!/usr/bin/env python3
"""Collect metrics from 0718 experiment logs into a CSV summary.

The parser is intentionally conservative: it records only metrics that are
explicitly present in logs and leaves unavailable fields blank.

Usage:
  python missing_ts_exp/scripts/r0718_collect_results.py
"""

from __future__ import annotations

import csv
import datetime as dt
import pathlib
import re
from dataclasses import dataclass, asdict


ROOT = pathlib.Path("/data/wangzuke/time-series-forecast-exp")
RESULTS = ROOT / "missing_ts_exp" / "results" / "0718_block_hmbg"
LOG_DIR = RESULTS / "raw_logs"
CSV_DIR = RESULTS / "csv"
OUT_CSV = CSV_DIR / "metrics_summary.csv"


BITGRAPH_RE = re.compile(
    r"loss,RMSE,MAPE\s+"
    r"(?P<mae>[-+]?\d+(?:\.\d+)?)\s*&\s*"
    r"(?P<rmse>[-+]?\d+(?:\.\d+)?)\s*&\s*"
    r"(?P<mape>[-+]?\d+(?:\.\d+)?)"
)

HDTTS_METRIC_RE = re.compile(
    r"^\s*(?P<name>test_(?:loss|mae|mre|mse|mae_unmasked|mre_unmasked|mse_unmasked))\s+"
    r"(?P<value>[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s*$",
    re.MULTILINE,
)

SEED_RE = re.compile(r"Seed set to (?P<seed>\d+)")
BITGRAPH_ACTUAL_RATE_RE = re.compile(
    r"\[temporal_block_mask\] target=[\d.]+ actual=(?P<actual>[\d.]+)"
)
EPOCH_RE = re.compile(r"^Epoch (?P<epoch>\d+):", re.MULTILINE)


@dataclass
class Row:
    log_file: str
    model: str = ""
    variant: str = ""
    dataset: str = ""
    missing_mode: str = ""
    missing_rate: str = ""
    horizon: str = "24"
    batch_size: str = ""
    epochs: str = ""
    status: str = ""
    mae: str = ""
    rmse: str = ""
    mape: str = ""
    test_loss: str = ""
    test_mae: str = ""
    test_mre: str = ""
    test_mse: str = ""
    test_mae_unmasked: str = ""
    test_mre_unmasked: str = ""
    test_mse_unmasked: str = ""
    n_params: str = ""
    seed: str = ""
    actual_missing_rate: str = ""
    last_epoch: str = ""
    log_mtime: str = ""


# HD-TTS / Block-HMBGNet dataset/mode config names embed the missing rate as
# a direct percentage (e.g. "block_t_30" = 30%), while BiTGraph filenames use
# a truncated-tenths ratio tag (e.g. "_r05" = 0.5). These are NOT the same
# convention, so they need two separate regexes rather than one shared one.
HDTTS_MODE_RE = re.compile(r"(?:block_t|block_st|point)_(?P<rate>\d{2,3})")
BITGRAPH_RATIO_RE = re.compile(r"_r(?P<rate>\d{2})(?:_|$)")

PARAMS_RE = re.compile(r"^\s*(?P<count>[\d.]+)\s*K?\s+Trainable params\s*$", re.MULTILINE)


def infer_metadata(path: pathlib.Path) -> dict[str, str]:
    stem = path.stem
    parts = stem.split("_")
    meta: dict[str, str] = {}

    if stem.startswith("bitgraph_") or stem.startswith("smoke_bitgraph_"):
        meta["model"] = "BiTGraph"
        if stem.startswith("smoke_bitgraph_") and len(parts) >= 3:
            meta["dataset"] = parts[2]
        elif len(parts) >= 2:
            meta["dataset"] = parts[1]
        if "_point_" in stem or stem.endswith("_point"):
            meta["missing_mode"] = "random_point"
        else:
            meta["missing_mode"] = "temporal_block"
        ratio_match = BITGRAPH_RATIO_RE.search(stem)
        if ratio_match:
            meta["missing_rate"] = f"0.{ratio_match.group('rate').lstrip('0') or '0'}"
    elif stem.startswith("hdtts_") or stem.startswith("smoke_hdtts_") or stem.startswith("block_hmbg_"):
        if stem.startswith("block_hmbg_"):
            meta["model"] = "Block-HMBGNet"
            # block_hmbg_<variant>_<dataset>_<mode...>_hHH_bBB_eEE
            m = re.match(r"block_hmbg_(?P<variant>[a-z_]+?)_(?P<dataset>la|bay)_"
                        r"(?P<mode>block_t|block_st|point)_(?P<rate>\d{2,3})", stem)
            if m:
                meta["variant"] = m.group("variant")
                meta["dataset"] = m.group("dataset")
                meta["missing_mode"] = m.group("mode")
        else:
            meta["model"] = "HD-TTS"
            if stem.startswith("smoke_hdtts_") and len(parts) >= 3:
                meta["dataset"] = parts[2]
            elif len(parts) >= 2:
                meta["dataset"] = parts[1]
            m = re.search(r"(?P<mode>block_t|block_st|point)_(?P<rate>\d{2,3})", stem)
            if m:
                meta["missing_mode"] = m.group("mode")

        mode_match = HDTTS_MODE_RE.search(stem)
        if mode_match:
            raw = mode_match.group("rate")
            meta["missing_rate"] = f"0.{raw}" if len(raw) == 2 else f"0.{raw.rstrip('0') or '0'}"

    batch_match = re.search(r"_b(?P<batch>\d+)(?:_|$)", stem)
    if batch_match:
        meta["batch_size"] = batch_match.group("batch")

    epoch_match = re.search(r"_e(?P<epochs>\d+)(?:_|$)", stem)
    if epoch_match:
        meta["epochs"] = epoch_match.group("epochs")

    return meta


def parse_log(path: pathlib.Path) -> Row:
    text = path.read_text(errors="replace")
    mtime = dt.datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
    row = Row(log_file=str(path.relative_to(ROOT)), log_mtime=mtime)
    for key, value in infer_metadata(path).items():
        setattr(row, key, value)

    if "Traceback (most recent call last)" in text or "Error executing job" in text:
        row.status = "failed"
    elif ("all batch_probe jobs finished" in text or "all full jobs finished" in text
          or "all jobs finished" in text):
        row.status = "finished"
    elif BITGRAPH_RE.search(text) or "test_mae_unmasked" in text:
        row.status = "finished"
    else:
        row.status = "unknown"

    bitgraph_match = BITGRAPH_RE.search(text)
    if bitgraph_match:
        row.mae = bitgraph_match.group("mae")
        row.rmse = bitgraph_match.group("rmse")
        row.mape = bitgraph_match.group("mape")

    for metric in HDTTS_METRIC_RE.finditer(text):
        setattr(row, metric.group("name"), metric.group("value"))

    params_match = PARAMS_RE.search(text)
    if params_match:
        row.n_params = params_match.group("count")

    seed_match = SEED_RE.search(text)
    if seed_match:
        row.seed = seed_match.group("seed")

    actual_rate_match = BITGRAPH_ACTUAL_RATE_RE.search(text)
    if actual_rate_match:
        row.actual_missing_rate = actual_rate_match.group("actual")

    epoch_matches = EPOCH_RE.findall(text)
    if epoch_matches:
        row.last_epoch = str(max(int(e) for e in epoch_matches))

    return row


def merge_split_bitgraph_logs(rows: list[Row]) -> list[Row]:
    """Some Phase 2 BiTGraph runs (blockt r05-r08) have main.py's training
    output and test_forecasting.py's test-metric output in two separate log
    files (``<name>.log`` + ``<name>_test.log``) instead of one chained log
    (the pattern r0718_run_baselines.sh's run_bitgraph() normally produces).
    Fold the _test.log row's metrics into its base row and drop the
    standalone _test.log row so each real run appears exactly once."""
    by_stem = {pathlib.Path(r.log_file).stem: r for r in rows}
    merged: list[Row] = []
    consumed: set[str] = set()
    for stem, row in by_stem.items():
        if stem.endswith("_test"):
            base_stem = stem[: -len("_test")]
            base = by_stem.get(base_stem)
            if base is not None:
                if row.status == "finished" and base.status != "finished":
                    base.mae, base.rmse, base.mape = row.mae, row.rmse, row.mape
                    base.status = "finished"
                consumed.add(stem)
    for stem, row in by_stem.items():
        if stem in consumed:
            continue
        merged.append(row)
    merged.sort(key=lambda r: r.log_file)
    return merged


def main() -> int:
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    rows = [parse_log(p) for p in sorted(LOG_DIR.glob("*.log"))]
    rows = merge_split_bitgraph_logs(rows)
    fieldnames = list(asdict(Row(log_file="")).keys())
    with OUT_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))
    print(f"wrote {OUT_CSV} rows={len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
