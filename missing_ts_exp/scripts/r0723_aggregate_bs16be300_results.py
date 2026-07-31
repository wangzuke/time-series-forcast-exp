#!/usr/bin/env python
"""Aggregate the formal R0723 bs16/be300 P-GUTS and CG-P-GUTS results."""

from __future__ import annotations

import csv
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "external_repro/pguts"))
if "code" in sys.modules and not hasattr(sys.modules["code"], "__path__"):
    del sys.modules["code"]

import torch


RESULT_ROOT = ROOT / "missing_ts_exp/results/0723_official_pguts_coarse_imputation"
CSV_DIR = RESULT_ROOT / "csv"
LOG_ROOT = ROOT / "external_repro/pguts/log"


RUNS = [
    {
        "dataset": "METR-LA",
        "method": "P-GUTS [3,6]",
        "tag": "repro_la_pguts36_s1",
        "seed": 1,
        "logdir": LOG_ROOT / "la_block/PGUTS/20260728T153814_1_bs16be300_repro_la_pguts36_s1",
    },
    {
        "dataset": "METR-LA",
        "method": "P-GUTS [3,6]",
        "tag": "repro_la_pguts36_s2",
        "seed": 2,
        "logdir": LOG_ROOT / "la_block/PGUTS/20260728T153813_2_bs16be300_repro_la_pguts36_s2",
    },
    {
        "dataset": "METR-LA",
        "method": "CG-P-GUTS [3,6]",
        "tag": "cgdist_la_pguts36_s1",
        "seed": 1,
        "logdir": LOG_ROOT / "la_block/PGUTS/20260728T153814_1_bs16be300_cgdist_la_pguts36_s1",
    },
    {
        "dataset": "METR-LA",
        "method": "CG-P-GUTS [3,6]",
        "tag": "cgdist_la_pguts36_s2",
        "seed": 2,
        "logdir": LOG_ROOT / "la_block/PGUTS/20260728T153814_2_bs16be300_cgdist_la_pguts36_s2",
    },
    {
        "dataset": "PEMS-BAY",
        "method": "P-GUTS [3,6]",
        "tag": "repro_bay_pguts36_s1",
        "seed": 1,
        "logdir": LOG_ROOT / "bay_block/PGUTS/20260728T153813_1_bs16be300_repro_bay_pguts36_s1",
    },
    {
        "dataset": "PEMS-BAY",
        "method": "P-GUTS [3,6]",
        "tag": "repro_bay_pguts36_s2",
        "seed": 2,
        "logdir": LOG_ROOT / "bay_block/PGUTS/20260728T153812_2_bs16be300_repro_bay_pguts36_s2",
    },
    {
        "dataset": "PEMS-BAY",
        "method": "CG-P-GUTS [3,6]",
        "tag": "cgdist_bay_pguts36_s1",
        "seed": 1,
        "logdir": LOG_ROOT / "bay_block/PGUTS/20260728T153812_1_bs16be300_cgdist_bay_pguts36_s1",
    },
    {
        "dataset": "PEMS-BAY",
        "method": "CG-P-GUTS [3,6]",
        "tag": "cgdist_bay_pguts36_s2",
        "seed": 2,
        "logdir": LOG_ROOT / "bay_block/PGUTS/20260728T153812_2_bs16be300_cgdist_bay_pguts36_s2",
    },
]

PAPER_MAIN = {
    ("METR-LA", "P-GUTS [3,6]"): 1.92,
    ("PEMS-BAY", "P-GUTS [3,6]"): 0.93,
}

PAPER_TABLE3 = {
    ("METR-LA", 0.05): 2.53,
    ("METR-LA", 0.10): 3.07,
    ("METR-LA", 0.15): 3.80,
    ("PEMS-BAY", 0.05): 1.69,
    ("PEMS-BAY", 0.10): 2.20,
    ("PEMS-BAY", 0.15): 2.86,
}


def mean(xs):
    return sum(xs) / len(xs)


def pstdev(xs):
    mu = mean(xs)
    return math.sqrt(sum((x - mu) ** 2 for x in xs) / len(xs))


def load_ckpt_summary(logdir: Path):
    ckpts = sorted(logdir.glob("*.ckpt"))
    if not ckpts:
        return "", "", ""
    ckpt = ckpts[0]
    m = re.match(r"epoch=(\d+)-step=(\d+)\.ckpt", ckpt.name)
    epoch = int(m.group(1)) if m else ""
    step = int(m.group(2)) if m else ""
    data = torch.load(ckpt, map_location="cpu")
    best = data.get("callbacks", {})
    best_score = ""
    for cb in best.values():
        if isinstance(cb, dict) and "best_model_score" in cb:
            score = cb["best_model_score"]
            best_score = float(score.item() if hasattr(score, "item") else score)
            break
    return epoch, step, best_score


def load_output_metrics(path: Path):
    out = torch.load(path, map_location="cpu")
    y = out["y"].float()
    y_hat = out.get("y_hat_reset", out["y_hat"]).float()
    mask = out["mask"].bool()
    err = y_hat[mask] - y[mask]
    mae = err.abs().mean().item()
    rmse = torch.sqrt((err**2).mean()).item()
    return mae, rmse, int(mask.sum().item())


def write_csv(path: Path, rows, fieldnames):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def aggregate_main():
    rows = []
    for run in RUNS:
        mae, rmse, n_eval = load_output_metrics(run["logdir"] / "output.pt")
        epoch, step, best_val_mae = load_ckpt_summary(run["logdir"])
        rows.append(
            {
                "dataset": run["dataset"],
                "method": run["method"],
                "tag": run["tag"],
                "seed": run["seed"],
                "mae": mae,
                "rmse": rmse,
                "n_eval": n_eval,
                "best_epoch": epoch,
                "best_step": step,
                "best_val_mae": best_val_mae,
                "logdir": str(run["logdir"].relative_to(ROOT)),
            }
        )
    write_csv(
        CSV_DIR / "bs16be300_main_runs.csv",
        rows,
        [
            "dataset",
            "method",
            "tag",
            "seed",
            "mae",
            "rmse",
            "n_eval",
            "best_epoch",
            "best_step",
            "best_val_mae",
            "logdir",
        ],
    )

    summary = []
    for dataset in sorted({r["dataset"] for r in rows}):
        for method in ["P-GUTS [3,6]", "CG-P-GUTS [3,6]"]:
            group = [r for r in rows if r["dataset"] == dataset and r["method"] == method]
            maes = [float(r["mae"]) for r in group]
            rmses = [float(r["rmse"]) for r in group]
            vals = [float(r["best_val_mae"]) for r in group]
            summary.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "n_seeds": len(group),
                    "mae_mean": mean(maes),
                    "mae_std": pstdev(maes),
                    "rmse_mean": mean(rmses),
                    "rmse_std": pstdev(rmses),
                    "best_val_mae_mean": mean(vals),
                    "best_val_mae_std": pstdev(vals),
                }
            )
    write_csv(
        CSV_DIR / "bs16be300_main_summary.csv",
        summary,
        [
            "dataset",
            "method",
            "n_seeds",
            "mae_mean",
            "mae_std",
            "rmse_mean",
            "rmse_std",
            "best_val_mae_mean",
            "best_val_mae_std",
        ],
    )
    return rows, summary


def aggregate_table3():
    pattern = re.compile(
        r"bs16be300_table3_table3_(repro|cgdist)_(la|bay)_s(\d+)_pf(0p05|0p1|0p15)\.log$"
    )
    seed_re = re.compile(r"SEED\s+(\d+)\s+-\s+Test MAE:\s+([0-9.]+)")
    summary_re = re.compile(r"MAE over 5 runs:\s+([0-9.]+)±([0-9.]+)")
    pf_map = {"0p05": 0.05, "0p1": 0.10, "0p15": 0.15}
    dataset_map = {"la": "METR-LA", "bay": "PEMS-BAY"}
    method_map = {"repro": "P-GUTS [3,6]", "cgdist": "CG-P-GUTS [3,6]"}

    rows = []
    for path in sorted((RESULT_ROOT / "raw_logs").glob("bs16be300_table3_table3_*.log")):
        m = pattern.match(path.name)
        if not m:
            continue
        kind, ds, seed, pf_key = m.groups()
        text = path.read_text(errors="replace")
        mask_pairs = seed_re.findall(text)
        summaries = summary_re.findall(text)
        rows.append(
            {
                "dataset": dataset_map[ds],
                "method": method_map[kind],
                "train_seed": int(seed),
                "p_fault_eval": pf_map[pf_key],
                "mask_seed_maes": ";".join(f"{s}:{v}" for s, v in mask_pairs),
                "mae_mean_5_masks": float(summaries[-1][0]) if summaries else "",
                "mae_std_5_masks": float(summaries[-1][1]) if summaries else "",
                "log": str(path.relative_to(ROOT)),
            }
        )

    write_csv(
        CSV_DIR / "bs16be300_table3_runs.csv",
        rows,
        [
            "dataset",
            "method",
            "train_seed",
            "p_fault_eval",
            "mask_seed_maes",
            "mae_mean_5_masks",
            "mae_std_5_masks",
            "log",
        ],
    )

    summary = []
    for dataset in sorted({r["dataset"] for r in rows}):
        for method in ["P-GUTS [3,6]", "CG-P-GUTS [3,6]"]:
            for pf in [0.05, 0.10, 0.15]:
                group = [
                    r
                    for r in rows
                    if r["dataset"] == dataset
                    and r["method"] == method
                    and abs(float(r["p_fault_eval"]) - pf) < 1e-12
                ]
                vals = [float(r["mae_mean_5_masks"]) for r in group]
                summary.append(
                    {
                        "dataset": dataset,
                        "method": method,
                        "p_fault_eval": pf,
                        "n_train_seeds": len(group),
                        "mae_mean": mean(vals),
                        "mae_std_train_seed": pstdev(vals),
                    }
                )
    write_csv(
        CSV_DIR / "bs16be300_table3_summary.csv",
        summary,
        [
            "dataset",
            "method",
            "p_fault_eval",
            "n_train_seeds",
            "mae_mean",
            "mae_std_train_seed",
        ],
    )
    return rows, summary


def aggregate_paper_comparison(main_summary, table3_summary):
    rows = []
    for row in main_summary:
        if row["method"] == "P-GUTS [3,6]":
            paper = PAPER_MAIN[(row["dataset"], row["method"])]
            rows.append(
                {
                    "section": "Table2-main",
                    "dataset": row["dataset"],
                    "method": row["method"],
                    "p_fault_eval": "",
                    "paper_mae": paper,
                    "ours_mae": row["mae_mean"],
                    "delta_abs": row["mae_mean"] - paper,
                    "delta_pct": (row["mae_mean"] - paper) / paper * 100,
                }
            )
    for row in table3_summary:
        if row["method"] == "P-GUTS [3,6]":
            key = (row["dataset"], float(row["p_fault_eval"]))
            paper = PAPER_TABLE3[key]
            rows.append(
                {
                    "section": "Table3-robustness",
                    "dataset": row["dataset"],
                    "method": row["method"],
                    "p_fault_eval": row["p_fault_eval"],
                    "paper_mae": paper,
                    "ours_mae": row["mae_mean"],
                    "delta_abs": row["mae_mean"] - paper,
                    "delta_pct": (row["mae_mean"] - paper) / paper * 100,
                }
            )
    write_csv(
        CSV_DIR / "bs16be300_paper_comparison.csv",
        rows,
        [
            "section",
            "dataset",
            "method",
            "p_fault_eval",
            "paper_mae",
            "ours_mae",
            "delta_abs",
            "delta_pct",
        ],
    )


def main():
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    main_rows, main_summary = aggregate_main()
    table3_rows, table3_summary = aggregate_table3()
    aggregate_paper_comparison(main_summary, table3_summary)
    print(f"wrote main runs: {len(main_rows)}")
    print(f"wrote table3 runs: {len(table3_rows)}")
    print(f"csv dir: {CSV_DIR}")


if __name__ == "__main__":
    main()
