#!/usr/bin/env python3
"""Render reproducible primary missing-pattern-shift diagnostic figures.

The script consumes only the CSVs emitted by ``aggregate_primary.py``.  It is
safe to run while workers are active because result JSON files are atomically
written and the aggregation step can simply be re-run on the next hour.
"""
from __future__ import annotations

import os
import pathlib

from protocol import EXP_ROOT

# The interactive account's default config directory is not writable in some
# batch environments.  Set this before importing pyplot.
os.environ.setdefault("MPLCONFIGDIR", str(EXP_ROOT / "figures" / ".mplconfig"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


EFFECT_LABELS = {
    "p_to_b_penalty": "P→B − B→B (primary)",
    "block_intrinsic_difficulty": "P→B − P→P (block intrinsic)",
    "reverse_b_to_p_penalty": "B→P − P→P (reverse shift)",
}
EFFECT_COLORS = {
    "p_to_b_penalty": "#b2182b",
    "block_intrinsic_difficulty": "#2166ac",
    "reverse_b_to_p_penalty": "#4d9221",
}


def output_dir() -> pathlib.Path:
    path = EXP_ROOT / "figures" / "primary"
    path.mkdir(parents=True, exist_ok=True)
    return path


def preliminary_suffix(part: pd.DataFrame) -> str:
    return " (preliminary)" if (part["paired_repetitions"] < 25).any() else ""


def plot_penalty_heatmap(summary: pd.DataFrame, baseline: str, out: pathlib.Path) -> pathlib.Path | None:
    part = summary[(summary["baseline"] == baseline) & (summary["effect"] == "p_to_b_penalty")].copy()
    if part.empty:
        return None
    part["row"] = part.apply(lambda r: f"{r['dataset']}  H={int(r['horizon'])}", axis=1)
    rows = sorted(part["row"].unique())
    blocks = sorted(part["block_len"].unique())
    matrix = np.full((len(rows), len(blocks)), np.nan)
    labels: dict[tuple[int, int], str] = {}
    for _, item in part.iterrows():
        i, j = rows.index(item["row"]), blocks.index(item["block_len"])
        matrix[i, j] = item["delta_mae_mean"]
        labels[i, j] = f"{item['delta_mae_mean']:.3f}\nn={int(item['paired_repetitions'])}"
    limit = np.nanmax(np.abs(matrix)) if np.isfinite(matrix).any() else 1.0
    limit = max(limit, 1e-12)
    fig, ax = plt.subplots(figsize=(1.55 * len(blocks) + 2.8, 0.6 * len(rows) + 2.3), constrained_layout=True)
    image = ax.imshow(matrix, cmap="coolwarm", vmin=-limit, vmax=limit, aspect="auto")
    ax.set_xticks(range(len(blocks)), [str(int(value)) for value in blocks])
    ax.set_yticks(range(len(rows)), rows)
    ax.set_xlabel("test block length")
    ax.set_title(f"{baseline}: P→B penalty ΔMAE{preliminary_suffix(part)}")
    for (i, j), label in labels.items():
        color = "white" if abs(matrix[i, j]) > 0.55 * limit else "black"
        ax.text(j, i, label, ha="center", va="center", color=color, fontsize=8)
    fig.colorbar(image, ax=ax, label="ΔMAE (P→B − B→B)")
    path = out / f"p_to_b_penalty_heatmap_{baseline}.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def plot_effect_curves(summary: pd.DataFrame, baseline: str, out: pathlib.Path) -> pathlib.Path | None:
    part = summary[summary["baseline"] == baseline].copy()
    if part.empty:
        return None
    datasets = sorted(part["dataset"].unique())
    horizons = sorted(part["horizon"].unique())
    fig, axes = plt.subplots(len(datasets), len(horizons), figsize=(5.3 * len(horizons), 3.5 * len(datasets)), squeeze=False, constrained_layout=True)
    for row, dataset in enumerate(datasets):
        for col, horizon in enumerate(horizons):
            ax = axes[row, col]
            cell = part[(part["dataset"] == dataset) & (part["horizon"] == horizon)]
            if cell.empty:
                # Avoid visually implying a 0-valued result for conditions
                # whose jobs have not reached this dataset/horizon yet.
                ax.set_visible(False)
                continue
            for effect, label in EFFECT_LABELS.items():
                values = cell[cell["effect"] == effect].sort_values("block_len")
                if values.empty:
                    continue
                y = values["delta_mae_mean"].to_numpy()
                lower = y - values["delta_mae_ci_low"].to_numpy()
                upper = values["delta_mae_ci_high"].to_numpy() - y
                ax.errorbar(values["block_len"], y, yerr=np.vstack([lower, upper]), marker="o", capsize=3,
                            label=label, color=EFFECT_COLORS[effect])
                # All three effects use the same paired experimental keys at
                # a given block length, so annotate n only once to prevent
                # overlapping labels from concealing confidence intervals.
                if effect == "p_to_b_penalty":
                    for x, yy, n in zip(values["block_len"], y, values["paired_repetitions"]):
                        ax.annotate(f"n={int(n)}", (x, yy), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=7)
            ax.axhline(0, color="0.35", linewidth=0.8)
            ax.set_title(f"{dataset}, H={int(horizon)}{preliminary_suffix(cell)}")
            ax.set_xlabel("block length")
            ax.set_ylabel("paired ΔMAE")
            ax.grid(axis="y", alpha=0.25)
            if row == 0 and col == len(horizons) - 1:
                ax.legend(fontsize=7, loc="best")
    fig.suptitle(f"{baseline}: 2×2 missing-pattern effects", fontsize=13)
    path = out / f"two_by_two_effect_curves_{baseline}.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def main() -> None:
    summary_path = EXP_ROOT / "summary" / "primary_2x2_effect_summary.csv"
    if not summary_path.is_file():
        raise SystemExit(f"missing summary: {summary_path}; run aggregate_primary.py first")
    summary = pd.read_csv(summary_path)
    if summary.empty:
        raise SystemExit("no paired effects are available yet")
    out = output_dir()
    outputs = []
    for baseline in sorted(summary["baseline"].unique()):
        for path in (plot_penalty_heatmap(summary, baseline, out), plot_effect_curves(summary, baseline, out)):
            if path is not None:
                outputs.append(path)
    print("\n".join(str(path) for path in outputs))


if __name__ == "__main__":
    main()
