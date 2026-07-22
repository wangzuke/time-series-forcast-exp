#!/usr/bin/env python3
"""Build summary tables and figures for the 0721 P-GUTS / HD-PGUTS report."""
from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path("/data/wangzuke/time-series-forecast-exp")
RESULTS = ROOT / "missing_ts_exp" / "results" / "0721_cofill_pguts_forecasting"
CSV_DIR = RESULTS / "csv"
DIAG_DIR = RESULTS / "diagnostics" / "pguts_hdpguts"
FIG_DIR = ROOT / "missing_ts_exp" / "docs" / "figures" / "r0721"


plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 140,
})


def condition_label(row: pd.Series) -> str:
    dataset = "METR-LA" if row["dataset"] == "Metr" else "PEMS-BAY"
    mask = {"block_t": "Block-T", "block_st": "Block-ST", "point": "Point"}[row["mask_type"]]
    return f"{dataset}\n{mask} {int(round(float(row['target_missing_rate']) * 100))}%"


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pguts = pd.read_csv(CSV_DIR / "pguts_results.csv")
    hd = pd.read_csv(CSV_DIR / "hd_pguts_ablation_results.csv")
    all_rows = pd.read_csv(CSV_DIR / "pguts_hdpguts_all_results.csv")
    return pguts, hd, all_rows


def write_summary_tables(pguts: pd.DataFrame, hd: pd.DataFrame, all_rows: pd.DataFrame) -> None:
    seed1_h24 = pguts[(pguts["seed"] == 1) & (pguts["T_out"] == 24)].copy()
    seed1_h24.to_csv(CSV_DIR / "r0721_pguts_h24_seed1_rows.csv", index=False)

    pivot = seed1_h24.pivot_table(
        index=["dataset", "mask_type", "target_missing_rate"],
        columns="pooling_factors",
        values="MAE",
    ).reset_index()
    pivot["delta_36_vs_3_pct"] = (pivot["3,6"] - pivot["3"]) / pivot["3"] * 100
    pivot.to_csv(CSV_DIR / "r0721_pguts_h24_pooling_summary.csv", index=False)

    critical = pguts[
        (pguts["T_out"] == 24)
        & (pguts["mask_type"].isin(["block_t", "block_st"]))
        & (pguts["target_missing_rate"].isin([0.7, 0.9]))
    ].copy()
    critical_summary = (
        critical.groupby(["dataset", "mask_type", "target_missing_rate", "pooling_factors"])
        .agg(
            MAE_mean=("MAE", "mean"),
            MAE_std=("MAE", "std"),
            RMSE_mean=("RMSE_or_MSE", "mean"),
            train_time_sec_mean=("train_time_sec", "mean"),
            gpu_peak_mb_mean=("gpu_peak_mb", "mean"),
        )
        .reset_index()
    )
    critical_summary.to_csv(CSV_DIR / "r0721_pguts_critical_multiseed_summary.csv", index=False)

    hd_summary = (
        hd.groupby(["dataset", "mask_type", "target_missing_rate", "variant"])
        .agg(
            MAE_mean=("MAE", "mean"),
            MAE_std=("MAE", "std"),
            RMSE_mean=("RMSE_or_MSE", "mean"),
            train_time_sec_mean=("train_time_sec", "mean"),
            gpu_peak_mb_mean=("gpu_peak_mb", "mean"),
        )
        .reset_index()
    )
    hd_summary.to_csv(CSV_DIR / "r0721_hdpguts_ablation_multiseed_summary.csv", index=False)

    baseline = (
        critical[critical["pooling_factors"] == "3,6"]
        .groupby(["dataset", "mask_type", "target_missing_rate"])
        .agg(pguts36_MAE=("MAE", "mean"), pguts36_RMSE=("RMSE_or_MSE", "mean"))
        .reset_index()
    )
    delta = hd_summary.merge(baseline, on=["dataset", "mask_type", "target_missing_rate"])
    delta["delta_vs_pguts36_pct"] = (delta["MAE_mean"] - delta["pguts36_MAE"]) / delta["pguts36_MAE"] * 100
    delta.to_csv(CSV_DIR / "r0721_hdpguts_vs_pguts36_delta.csv", index=False)

    efficiency = (
        all_rows.groupby(["model", "variant"])
        .agg(
            runs=("run_id", "count"),
            epoch_time_sec_mean=("epoch_time_sec", "mean"),
            train_time_sec_mean=("train_time_sec", "mean"),
            gpu_peak_mb_mean=("gpu_peak_mb", "mean"),
            MAE_mean=("MAE", "mean"),
        )
        .reset_index()
    )
    efficiency.to_csv(CSV_DIR / "r0721_efficiency_summary.csv", index=False)


def read_scale_weights() -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    pattern = re.compile(
        r"pguts_hdpguts_(?P<dataset>Metr|PEMS)_(?P<mask_type>block_st|block_t)_"
        r"r(?P<rate>70|90)_h24_pf3-6_(?P<variant>full|no_graph_coarsening)_"
        r"s(?P<seed>\d+)_scale_weights.npy"
    )
    for path in sorted(DIAG_DIR.glob("*_scale_weights.npy")):
        match = pattern.match(path.name)
        if not match:
            continue
        meta = match.groupdict()
        values = np.load(path).mean(axis=0)
        labels = ["identity", "temporal3", "temporal6", "full_graph"]
        if meta["variant"] == "full":
            labels.append("coarse_graph")
        row: dict[str, float | str] = {
            "dataset": meta["dataset"],
            "mask_type": meta["mask_type"],
            "target_missing_rate": int(meta["rate"]) / 100,
            "variant": meta["variant"],
            "seed": int(meta["seed"]),
        }
        for label, value in zip(labels, values):
            row[label] = float(value)
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(CSV_DIR / "r0721_adaptive_scale_weights_summary.csv", index=False)
    return df


def plot_pooling_delta(pguts: pd.DataFrame) -> None:
    seed1_h24 = pguts[(pguts["seed"] == 1) & (pguts["T_out"] == 24)].copy()
    pivot = seed1_h24.pivot_table(
        index=["dataset", "mask_type", "target_missing_rate"],
        columns="pooling_factors",
        values="MAE",
    ).reset_index()
    pivot["delta"] = (pivot["3,6"] - pivot["3"]) / pivot["3"] * 100
    pivot["label"] = pivot.apply(condition_label, axis=1)
    pivot = pivot.sort_values("delta")

    fig, ax = plt.subplots(figsize=(12, 5))
    colors = ["#3d7ea6" if v < 0 else "#b55a5a" for v in pivot["delta"]]
    ax.bar(range(len(pivot)), pivot["delta"], color=colors)
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.set_ylabel("MAE change of [3,6] vs [3] (%)")
    ax.set_title("P-GUTS temporal pooling effect, T_out=24, seed=1")
    ax.set_xticks(range(len(pivot)))
    ax.set_xticklabels(pivot["label"], rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "pguts_pooling_delta_h24.png")
    plt.close(fig)


def plot_hd_delta(pguts: pd.DataFrame, hd: pd.DataFrame) -> None:
    critical = pguts[
        (pguts["T_out"] == 24)
        & (pguts["pooling_factors"] == "3,6")
        & (pguts["mask_type"].isin(["block_t", "block_st"]))
        & (pguts["target_missing_rate"].isin([0.7, 0.9]))
    ]
    baseline = (
        critical.groupby(["dataset", "mask_type", "target_missing_rate"])
        .agg(pguts36_MAE=("MAE", "mean"))
        .reset_index()
    )
    hd_summary = (
        hd.groupby(["dataset", "mask_type", "target_missing_rate", "variant"])
        .agg(MAE_mean=("MAE", "mean"))
        .reset_index()
        .merge(baseline, on=["dataset", "mask_type", "target_missing_rate"])
    )
    hd_summary["delta"] = (hd_summary["MAE_mean"] - hd_summary["pguts36_MAE"]) / hd_summary["pguts36_MAE"] * 100
    hd_summary["condition"] = hd_summary.apply(condition_label, axis=1)
    variants = ["no_graph_coarsening", "no_adaptive_fusion", "full"]
    conditions = list(dict.fromkeys(hd_summary["condition"]))
    matrix = np.array([
        [
            hd_summary[(hd_summary["condition"] == cond) & (hd_summary["variant"] == variant)]["delta"].iloc[0]
            for variant in variants
        ]
        for cond in conditions
    ])

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(matrix, cmap="RdBu_r", vmin=-4, vmax=4, aspect="auto")
    ax.set_xticks(range(len(variants)))
    ax.set_xticklabels(["w/o coarse\nadaptive", "coarse\nfixed", "full"], rotation=0)
    ax.set_yticks(range(len(conditions)))
    ax.set_yticklabels(conditions)
    ax.set_title("HD-PGUTS variants vs P-GUTS [3,6]\nnegative means lower MAE")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:+.1f}%", ha="center", va="center", color="#111111")
    fig.colorbar(im, ax=ax, label="MAE change (%)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "hdpguts_delta_vs_pguts36.png")
    plt.close(fig)


def plot_scale_weights(weights: pd.DataFrame) -> None:
    if weights.empty:
        return
    summary = (
        weights.groupby(["variant", "mask_type", "target_missing_rate"])
        .mean(numeric_only=True)
        .reset_index()
    )
    labels = ["identity", "temporal3", "temporal6", "full_graph", "coarse_graph"]
    variants = ["no_graph_coarsening", "full"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, variant in zip(axes, variants):
        sub = summary[summary["variant"] == variant].copy()
        sub["label"] = sub["mask_type"].map({"block_t": "Block-T", "block_st": "Block-ST"}) + " " + (
            sub["target_missing_rate"] * 100
        ).astype(int).astype(str) + "%"
        bottoms = np.zeros(len(sub))
        x = np.arange(len(sub))
        for label in labels:
            if label not in sub:
                continue
            vals = sub[label].fillna(0).to_numpy()
            ax.bar(x, vals, bottom=bottoms, label=label)
            bottoms += vals
        ax.set_title(variant)
        ax.set_xticks(x)
        ax.set_xticklabels(sub["label"], rotation=30, ha="right")
        ax.set_ylim(0, 1.0)
    axes[0].set_ylabel("mean adaptive fusion weight")
    axes[1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.tight_layout()
    fig.savefig(FIG_DIR / "adaptive_scale_weights.png")
    plt.close(fig)


def plot_efficiency(all_rows: pd.DataFrame) -> None:
    summary = (
        all_rows.groupby(["model", "variant"])
        .agg(MAE=("MAE", "mean"), train_time=("train_time_sec", "mean"), gpu=("gpu_peak_mb", "mean"))
        .reset_index()
    )
    summary["label"] = summary["variant"].replace({
        "pguts": "P-GUTS",
        "no_graph_coarsening": "w/o coarse",
        "no_adaptive_fusion": "w/o adaptive",
        "full": "full",
    })
    fig, ax = plt.subplots(figsize=(7, 5))
    sizes = np.maximum(summary["gpu"] / 100, 40)
    ax.scatter(summary["train_time"] / 60, summary["MAE"], s=sizes, alpha=0.75, color="#4c78a8")
    for _, row in summary.iterrows():
        ax.text(row["train_time"] / 60, row["MAE"], row["label"], ha="left", va="bottom")
    ax.set_xlabel("mean train time per run (min)")
    ax.set_ylabel("mean MAE over available runs")
    ax.set_title("0721 runtime and error summary")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "efficiency_tradeoff_pguts_hdpguts.png")
    plt.close(fig)


def main() -> int:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    pguts, hd, all_rows = load_data()
    write_summary_tables(pguts, hd, all_rows)
    weights = read_scale_weights()
    plot_pooling_delta(pguts)
    plot_hd_delta(pguts, hd)
    plot_scale_weights(weights)
    plot_efficiency(all_rows)
    print(f"wrote figures to {FIG_DIR}")
    for path in sorted(FIG_DIR.glob("*.png")):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
