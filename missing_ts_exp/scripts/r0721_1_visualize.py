#!/usr/bin/env python3
"""Build derived CSVs and figures for the 0721_1 baseline/CoFILL report."""

from __future__ import annotations

import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = pathlib.Path("/data/wangzuke/time-series-forecast-exp")
RESULTS = ROOT / "missing_ts_exp" / "results" / "0721_cofill_pguts_forecasting"
CSV_DIR = RESULTS / "csv"
FIG_DIR = ROOT / "missing_ts_exp" / "docs" / "figures" / "r0721_1"

BASELINE_CSV = CSV_DIR / "baseline_results.csv"
FORMAL_CSV = CSV_DIR / "formal_baseline_results_0721_1.csv"
PAIRWISE_CSV = CSV_DIR / "pairwise_seed1_0721_1.csv"
MULTISEED_CSV = CSV_DIR / "multiseed_key_summary_0721_1.csv"
COMPLETION_CSV = CSV_DIR / "completion_summary_0721_1.csv"


plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
    "figure.dpi": 160,
})


def load_formal() -> pd.DataFrame:
    df = pd.read_csv(BASELINE_CSV)
    formal = df[df["run_id"].astype(str).str.contains("_e200_")].copy()
    formal["target_missing_rate"] = formal["target_missing_rate"].astype(float)
    formal["T_out"] = formal["T_out"].astype(int)
    formal["seed"] = formal["seed"].astype(int)
    formal["MAE"] = formal["MAE"].astype(float)
    formal.to_csv(FORMAL_CSV, index=False)
    return formal


def make_pairwise(formal: pd.DataFrame) -> pd.DataFrame:
    seed1 = formal[formal["seed"].eq(1)]
    pair = (
        seed1.pivot_table(
            index=["dataset", "mask_type", "target_missing_rate", "T_out"],
            columns="model",
            values="MAE",
            aggfunc="first",
        )
        .reset_index()
        .rename_axis(columns=None)
    )
    pair["hdtts_advantage_pct"] = (
        (pair["BiTGraph"] - pair["HD-TTS-AMP"]) / pair["BiTGraph"] * 100
    )
    pair["winner"] = np.where(pair["hdtts_advantage_pct"] > 0, "HD-TTS-AMP", "BiTGraph")
    pair.to_csv(PAIRWISE_CSV, index=False)
    return pair


def make_multiseed(formal: pd.DataFrame) -> pd.DataFrame:
    key = formal[
        formal["T_out"].eq(24)
        & formal["mask_type"].isin(["block_t", "block_st"])
        & formal["target_missing_rate"].isin([0.7, 0.9])
    ].copy()
    agg = (
        key.groupby(["model", "dataset", "mask_type", "target_missing_rate"], as_index=False)
        .agg(
            mae_mean=("MAE", "mean"),
            mae_std=("MAE", "std"),
            mae_min=("MAE", "min"),
            mae_max=("MAE", "max"),
            n_seeds=("seed", "nunique"),
        )
    )
    wide = (
        agg.pivot_table(
            index=["dataset", "mask_type", "target_missing_rate"],
            columns="model",
            values="mae_mean",
            aggfunc="first",
        )
        .reset_index()
        .rename_axis(columns=None)
    )
    wide["hdtts_advantage_pct"] = (
        (wide["BiTGraph"] - wide["HD-TTS-AMP"]) / wide["BiTGraph"] * 100
    )
    out = agg.merge(
        wide[["dataset", "mask_type", "target_missing_rate", "hdtts_advantage_pct"]],
        on=["dataset", "mask_type", "target_missing_rate"],
        how="left",
    )
    out.to_csv(MULTISEED_CSV, index=False)
    return out


def make_completion(formal: pd.DataFrame) -> pd.DataFrame:
    rows = [
        {
            "scope": "formal_baseline",
            "expected_rows": 96,
            "actual_rows": len(formal),
            "finished_rows": int(formal["status"].eq("finished").sum()),
            "notes": "64 seed=1 main matrix + 32 key extra seeds; smoke e2 rows excluded",
        },
        {
            "scope": "cofill",
            "expected_rows": 4,
            "actual_rows": len(pd.read_csv(CSV_DIR / "cofill_results.csv")),
            "finished_rows": 0,
            "notes": "no local CoFILL runner/log metrics found in this workspace",
        },
    ]
    out = pd.DataFrame(rows)
    out.to_csv(COMPLETION_CSV, index=False)
    return out


def plot_advantage_heatmap(pair: pd.DataFrame) -> None:
    data = pair.copy()
    data["row"] = data["dataset"] + " / h" + data["T_out"].astype(str)
    data["col"] = (
        data["mask_type"].replace({"point": "Point", "block_t": "Block-T", "block_st": "Block-ST"})
        + " "
        + (data["target_missing_rate"] * 100).round().astype(int).astype(str)
        + "%"
    )
    row_order = ["Metr / h12", "Metr / h24", "PEMS / h12", "PEMS / h24"]
    col_order = [
        "Point 50%", "Point 70%",
        "Block-T 50%", "Block-T 70%", "Block-T 90%",
        "Block-ST 50%", "Block-ST 70%", "Block-ST 90%",
    ]
    mat = data.pivot(index="row", columns="col", values="hdtts_advantage_pct").reindex(row_order)[col_order]

    fig, ax = plt.subplots(figsize=(12, 4.6))
    im = ax.imshow(mat.values, cmap="YlGnBu", vmin=0, vmax=max(45, float(np.nanmax(mat.values))))
    ax.set_xticks(range(len(col_order)), labels=col_order, rotation=35, ha="right")
    ax.set_yticks(range(len(row_order)), labels=row_order)
    ax.set_title("HD-TTS-AMP MAE advantage over BiTGraph, seed=1")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            value = mat.values[i, j]
            if np.isfinite(value):
                ax.text(j, i, f"{value:.1f}%", ha="center", va="center", color="black", fontsize=9)
    fig.colorbar(im, ax=ax, label="Positive = HD-TTS-AMP lower MAE")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "hdtts_advantage_heatmap_seed1.png")
    plt.close(fig)


def plot_mae_trends(pair: pd.DataFrame) -> None:
    masks = ["point", "block_t", "block_st"]
    datasets = ["Metr", "PEMS"]
    horizons = [12, 24]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=False)
    for ax, dataset in zip(axes[:, 0], datasets):
        for horizon, linestyle in zip(horizons, ["-", "--"]):
            sub = pair[(pair["dataset"].eq(dataset)) & (pair["T_out"].eq(horizon))]
            for mask in masks:
                cur = sub[sub["mask_type"].eq(mask)].sort_values("target_missing_rate")
                if cur.empty:
                    continue
                ax.plot(
                    cur["target_missing_rate"] * 100,
                    cur["BiTGraph"],
                    marker="o",
                    linestyle=linestyle,
                    label=f"BiTGraph {mask} h{horizon}",
                    alpha=0.85,
                )
        ax.set_title(f"{dataset}: BiTGraph MAE")
        ax.set_xlabel("Missing rate (%)")
        ax.set_ylabel("MAE")
        ax.grid(alpha=0.25)

    for ax, dataset in zip(axes[:, 1], datasets):
        for horizon, linestyle in zip(horizons, ["-", "--"]):
            sub = pair[(pair["dataset"].eq(dataset)) & (pair["T_out"].eq(horizon))]
            for mask in masks:
                cur = sub[sub["mask_type"].eq(mask)].sort_values("target_missing_rate")
                if cur.empty:
                    continue
                ax.plot(
                    cur["target_missing_rate"] * 100,
                    cur["HD-TTS-AMP"],
                    marker="s",
                    linestyle=linestyle,
                    label=f"HD-TTS {mask} h{horizon}",
                    alpha=0.85,
                )
        ax.set_title(f"{dataset}: HD-TTS-AMP MAE")
        ax.set_xlabel("Missing rate (%)")
        ax.set_ylabel("MAE")
        ax.grid(alpha=0.25)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles[:6], labels[:6], loc="lower center", ncol=3)
    fig.suptitle("MAE trends by missing rate, seed=1", y=0.98)
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    fig.savefig(FIG_DIR / "mae_trends_seed1.png")
    plt.close(fig)


def plot_multiseed_advantage(multiseed: pd.DataFrame) -> None:
    wide = (
        multiseed.pivot_table(
            index=["dataset", "mask_type", "target_missing_rate"],
            columns="model",
            values="mae_mean",
            aggfunc="first",
        )
        .reset_index()
        .rename_axis(columns=None)
    )
    wide["label"] = (
        wide["dataset"]
        + " "
        + wide["mask_type"].replace({"block_t": "Block-T", "block_st": "Block-ST"})
        + " "
        + (wide["target_missing_rate"] * 100).round().astype(int).astype(str)
        + "%"
    )
    wide["hdtts_advantage_pct"] = (wide["BiTGraph"] - wide["HD-TTS-AMP"]) / wide["BiTGraph"] * 100
    wide = wide.sort_values("hdtts_advantage_pct")

    fig, ax = plt.subplots(figsize=(10, 4.8))
    bars = ax.barh(wide["label"], wide["hdtts_advantage_pct"], color="#4C78A8")
    ax.set_xlabel("HD-TTS-AMP advantage over BiTGraph (%)")
    ax.set_title("Key block-missing conditions, T_out=24, mean over seeds 1/2/3")
    ax.grid(axis="x", alpha=0.25)
    for bar, value in zip(bars, wide["hdtts_advantage_pct"]):
        ax.text(value + 0.6, bar.get_y() + bar.get_height() / 2, f"{value:.1f}%", va="center")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "multiseed_key_advantage.png")
    plt.close(fig)


def main() -> int:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    formal = load_formal()
    pair = make_pairwise(formal)
    multiseed = make_multiseed(formal)
    completion = make_completion(formal)

    plot_advantage_heatmap(pair)
    plot_mae_trends(pair)
    plot_multiseed_advantage(multiseed)

    print(f"wrote {FORMAL_CSV} rows={len(formal)}")
    print(f"wrote {PAIRWISE_CSV} rows={len(pair)}")
    print(f"wrote {MULTISEED_CSV} rows={len(multiseed)}")
    print(f"wrote {COMPLETION_CSV} rows={len(completion)}")
    print(f"wrote figures to {FIG_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
