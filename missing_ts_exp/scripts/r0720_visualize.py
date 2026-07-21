#!/usr/bin/env python3
"""Build analysis tables and figures for the 0720 fair baseline experiment."""

from __future__ import annotations

import pathlib
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = pathlib.Path("/data/wangzuke/time-series-forecast-exp")
RESULTS = ROOT / "missing_ts_exp" / "results" / "0720_fair_b512"
SUMMARY = RESULTS / "csv" / "fair_metrics_summary.csv"
OUT_CSV = RESULTS / "csv"
OUT_FIG = ROOT / "missing_ts_exp" / "docs" / "figures" / "r0720"


def load_formal() -> pd.DataFrame:
    df = pd.read_csv(SUMMARY)
    df = df[df["epochs"].astype(str) == "200"].copy()
    if len(df) != 24:
        raise RuntimeError(f"Expected 24 formal rows, got {len(df)}")

    df["rate"] = (df["target_missing_rate"].astype(float) * 100).round().astype(int)
    df["mae_primary"] = np.where(
        df["model"].eq("BiTGraph"),
        pd.to_numeric(df["mae"]),
        pd.to_numeric(df["test_mae"]),
    )
    df["mse_primary"] = np.where(
        df["model"].eq("BiTGraph"),
        pd.to_numeric(df["rmse"]) ** 2,
        pd.to_numeric(df["test_mse"]),
    )
    df["rmse_primary"] = np.where(
        df["model"].eq("BiTGraph"),
        pd.to_numeric(df["rmse"]),
        np.sqrt(pd.to_numeric(df["test_mse"])),
    )
    order = {"Metr": 0, "PEMS": 1}
    miss_order = {"point": 0, "block": 1}
    model_order = {"BiTGraph": 0, "HD-TTS-AMP": 1}
    df["_dataset_order"] = df["dataset"].map(order)
    df["_missing_order"] = df["missing_type"].map(miss_order)
    df["_model_order"] = df["model"].map(model_order)
    df = df.sort_values(["_dataset_order", "_missing_order", "rate", "_model_order"])
    return df


def build_pairwise(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["dataset", "missing_type", "rate"]
    for key, group in df.groupby(keys, sort=False):
        by_model = {r["model"]: r for _, r in group.iterrows()}
        bg = by_model["BiTGraph"]
        hd = by_model["HD-TTS-AMP"]
        rows.append({
            "dataset": key[0],
            "missing_type": key[1],
            "rate": key[2],
            "actual_missing_rate": bg["actual_missing_rate"],
            "bitgraph_mae": bg["mae_primary"],
            "hdtts_mae": hd["mae_primary"],
            "hdtts_minus_bitgraph_mae": hd["mae_primary"] - bg["mae_primary"],
            "hdtts_advantage_pct": (bg["mae_primary"] - hd["mae_primary"]) / bg["mae_primary"] * 100,
            "bitgraph_mse": bg["mse_primary"],
            "hdtts_mse": hd["mse_primary"],
            "mask_sha256": bg["mask_sha256"],
        })
    return pd.DataFrame(rows)


def write_tables(df: pd.DataFrame, pairwise: pd.DataFrame) -> None:
    OUT_CSV.mkdir(parents=True, exist_ok=True)
    keep = [
        "model", "dataset", "missing_type", "rate", "actual_missing_rate",
        "mae_primary", "mse_primary", "rmse_primary", "mask_sha256",
        "data_path", "mask_path", "status", "epochs", "batch_size",
    ]
    df[keep].to_csv(OUT_CSV / "formal_metrics_0720.csv", index=False)
    pairwise.to_csv(OUT_CSV / "pairwise_comparison_0720.csv", index=False)


def plot_mae_bars(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    panels = [("Metr", "point"), ("Metr", "block"),
              ("PEMS", "point"), ("PEMS", "block")]
    width = 0.36
    for ax, (dataset, missing) in zip(axes.ravel(), panels):
        sub = df[(df["dataset"] == dataset) & (df["missing_type"] == missing)]
        rates = [30, 50, 70]
        x = np.arange(len(rates))
        bg = [sub[(sub["model"] == "BiTGraph") & (sub["rate"] == r)]["mae_primary"].iloc[0]
              for r in rates]
        hd = [sub[(sub["model"] == "HD-TTS-AMP") & (sub["rate"] == r)]["mae_primary"].iloc[0]
              for r in rates]
        ax.bar(x - width / 2, bg, width, label="BiTGraph")
        ax.bar(x + width / 2, hd, width, label="HD-TTS-AMP")
        ax.set_title(f"{dataset} / {missing}")
        ax.set_xticks(x, [f"{r}%" for r in rates])
        ax.set_ylabel("MAE")
        ax.grid(axis="y", alpha=0.25)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2)
    fig.suptitle("0720 Fair Experiment: MAE by Condition", y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUT_FIG / "mae_by_condition.png", dpi=200)
    plt.close(fig)


def plot_advantage_heatmap(pairwise: pd.DataFrame) -> None:
    labels = []
    values = []
    for dataset in ["Metr", "PEMS"]:
        for missing in ["point", "block"]:
            row = []
            labels.append(f"{dataset}/{missing}")
            for rate in [30, 50, 70]:
                v = pairwise[
                    (pairwise["dataset"] == dataset)
                    & (pairwise["missing_type"] == missing)
                    & (pairwise["rate"] == rate)
                ]["hdtts_advantage_pct"].iloc[0]
                row.append(v)
            values.append(row)
    arr = np.array(values)
    vmax = max(abs(arr.min()), abs(arr.max()))
    fig, ax = plt.subplots(figsize=(8, 4.2))
    im = ax.imshow(arr, cmap="RdYlGn", vmin=-vmax, vmax=vmax)
    ax.set_xticks(np.arange(3), ["30%", "50%", "70%"])
    ax.set_yticks(np.arange(4), labels)
    ax.set_title("HD-TTS-AMP MAE Advantage over BiTGraph (%)")
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            ax.text(j, i, f"{arr[i, j]:.1f}%", ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax, label="Positive = HD-TTS lower MAE")
    fig.tight_layout()
    fig.savefig(OUT_FIG / "hdtts_advantage_heatmap.png", dpi=200)
    plt.close(fig)


def plot_degradation(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=False)
    for ax, dataset in zip(axes, ["Metr", "PEMS"]):
        for missing, linestyle in [("point", "-"), ("block", "--")]:
            for model, marker in [("BiTGraph", "o"), ("HD-TTS-AMP", "s")]:
                sub = df[
                    (df["dataset"] == dataset)
                    & (df["missing_type"] == missing)
                    & (df["model"] == model)
                ].sort_values("rate")
                base = sub[sub["rate"] == 30]["mae_primary"].iloc[0]
                rel = (sub["mae_primary"].to_numpy() / base - 1.0) * 100
                ax.plot(sub["rate"], rel, marker=marker, linestyle=linestyle,
                        label=f"{model}/{missing}")
        ax.set_title(dataset)
        ax.set_xlabel("Missing rate (%)")
        ax.set_ylabel("MAE degradation from 30% (%)")
        ax.set_xticks([30, 50, 70])
        ax.grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    fig.suptitle("Relative Degradation as Missing Rate Increases")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(OUT_FIG / "relative_degradation_30_to_70.png", dpi=200)
    plt.close(fig)


def main() -> int:
    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
    })
    OUT_FIG.mkdir(parents=True, exist_ok=True)
    df = load_formal()
    pairwise = build_pairwise(df)
    write_tables(df, pairwise)
    plot_mae_bars(df)
    plot_advantage_heatmap(pairwise)
    plot_degradation(df)
    print(f"wrote tables to {OUT_CSV}")
    print(f"wrote figures to {OUT_FIG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
