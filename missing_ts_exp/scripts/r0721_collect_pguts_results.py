#!/usr/bin/env python3
"""Aggregate 0721 P-GUTS / HD-PGUTS per-run JSON metrics into plan CSVs."""
from __future__ import annotations

import argparse
import csv
import json
import pathlib
from typing import Any


ROOT = pathlib.Path("/data/wangzuke/time-series-forecast-exp")
RESULTS_ROOT = ROOT / "missing_ts_exp" / "results" / "0721_cofill_pguts_forecasting"


CSV_FIELDS = [
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
    "pooling_factors",
    "graph_scale",
    "adaptive_fusion",
    "architecture_signature",
    "seed",
    "batch_size",
    "micro_batch_size",
    "grad_accum_steps",
    "MAE",
    "RMSE_or_MSE",
    "MAPE_or_MRE",
    "MRE",
    "epoch_time_sec",
    "train_time_sec",
    "gpu_peak_mb",
    "checkpoint_path",
    "log_path",
    "prediction_path",
    "scale_weights_path",
    "notes",
]


def compact(row: dict[str, Any]) -> dict[str, Any]:
    return {field: row.get(field, "") for field in CSV_FIELDS}


def write_csv(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(compact(row) for row in rows)
    print(f"wrote {path} rows={len(rows)}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_root", default=str(RESULTS_ROOT))
    args = parser.parse_args()
    results_root = pathlib.Path(args.results_root)
    metrics_dir = results_root / "metrics" / "pguts_hdpguts"
    rows = []
    for path in sorted(metrics_dir.glob("*.json")):
        with path.open() as f:
            rows.append(json.load(f))

    pguts = [row for row in rows if row.get("model") == "pgutsf"]
    hd = [row for row in rows if row.get("model") == "hd_pguts" and row.get("variant") == "full"]
    ablation = [row for row in rows if row.get("model") == "hd_pguts"]
    csv_dir = results_root / "csv"
    write_csv(csv_dir / "pguts_results.csv", pguts)
    write_csv(csv_dir / "hd_pguts_results.csv", hd)
    write_csv(csv_dir / "hd_pguts_ablation_results.csv", ablation)
    write_csv(csv_dir / "pguts_hdpguts_all_results.csv", rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
