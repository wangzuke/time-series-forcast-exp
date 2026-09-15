#!/usr/bin/env python3
"""Render paper-ready reliability and regionalization diagnostics from saved runs."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("/media/data1/wangzk/time-series-forcast-exp")
DIAGNOSTICS = ROOT / "missing_ts_exp/results/r0910_evidence_closure/stage_b_diagnostics"
ASSIGNMENT = ROOT / "missing_ts_exp/resources/r0905_blockt/regions/METR-LA/physical_connected_max8_assignment.npy"
OUT = ROOT / "my_paper/icassp/我的论文/figures/reliability_region_diagnostics.pdf"


def main() -> None:
    records = []
    for path in sorted(DIAGNOSTICS.glob("*_diagnostics.json")):
        row = json.loads(path.read_text())
        rate = 50 if "r50" in path.name else 70
        records.append((rate, row["edge_q_gate_correlation_off_diagonal"]["pearson"]))

    if len(records) != 10:
        raise ValueError(f"Expected 10 diagnostic files, found {len(records)}")

    assignment = np.load(ASSIGNMENT)
    if assignment.shape != (207, 31):
        raise ValueError(f"Unexpected METR-LA assignment shape: {assignment.shape}")

    plt.rcParams.update({"font.family": "serif", "font.size": 7.2, "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(3.25, 1.55))
    colors = {50: "#167c80", 70: "#b4473c"}
    for rate in (50, 70):
        values = [corr for current_rate, corr in records if current_rate == rate]
        x = np.full(len(values), rate, dtype=float) + np.linspace(-1.8, 1.8, len(values))
        ax.scatter(x, values, s=27, color=colors[rate], edgecolor="white", linewidth=0.55,
                   label=f"Block-T {rate}%", zorder=3)
        ax.hlines(np.mean(values), rate - 3.0, rate + 3.0, color=colors[rate], linewidth=1.4, zorder=2)
    ax.axhline(0.0, color="#555555", linewidth=0.75, linestyle="--", zorder=1)
    ax.set_xlim(43, 77)
    ax.set_ylim(-0.78, 0.10)
    ax.set_xticks([50, 70])
    ax.set_xlabel("Block-T missing rate")
    ax.set_ylabel("Pearson corr. of $q_{ij}$ and $g_{ij}$")
    ax.legend(frameon=False, loc="lower left", fontsize=6.5, handletextpad=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("Edge-level reliability diagnostic", fontsize=7.5, fontweight="bold", pad=3)

    fig.tight_layout(pad=0.45)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight", pad_inches=0.05)
    print(OUT)


if __name__ == "__main__":
    main()
