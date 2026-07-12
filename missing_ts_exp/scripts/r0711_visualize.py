"""R0711 visualization and aggregation for reliability-gated grouped-query fusion.

The script compares the 0711 gated variants with the historical 0709/0710
baselines and writes figures plus CSV summaries used by docs/0711实验报告.md.
Chart text is kept in English because the runtime may not provide a CJK font.
"""
from __future__ import annotations

import csv
import json
import math
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.15,
})

OUT_DIR = "docs/figures/r0711"
os.makedirs(OUT_DIR, exist_ok=True)

SEEDS = [2024, 2025]
CONDITIONS = [
    ("Weather", "continuous_segment", 0.3),
    ("Weather", "continuous_segment", 0.5),
    ("Weather", "continuous_segment", 0.7),
    ("Weather", "random_point", 0.5),
    ("Traffic", "random_point", 0.3),
    ("Traffic", "random_point", 0.5),
    ("Traffic", "random_point", 0.7),
    ("Traffic", "continuous_segment", 0.7),
    ("Electricity", "random_point", 0.5),
    ("Electricity", "continuous_segment", 0.5),
]

VARIANTS = {
    "corr": "r0709_P1A__misstsm_gq4corr__{ds}__{mt}_{rate}__h96_s{seed}",
    "soft": "r0709_P1B__misstsm_gq4soft__{ds}__{mt}_{rate}__h96_s{seed}",
    "fuse": "r0710_P2__misstsm_gq4fuse__{ds}__{mt}_{rate}__h96_s{seed}",
    "sgate": "r0711_P1__misstsm_fuse_sgate__{ds}__{mt}_{rate}__h96_s{seed}",
    "ggate": "r0711_P1__misstsm_fuse_ggate__{ds}__{mt}_{rate}__h96_s{seed}",
    "mgate": "r0711_P1__misstsm_fuse_mgate__{ds}__{mt}_{rate}__h96_s{seed}",
}

PHASE0 = {
    "soft_p0": "r0711_P0diag__misstsm_gq4soft__{ds}__{mt}_{rate}__h96_s{seed}",
    "fuse_p0": "r0711_P0diag__misstsm_gq4fuse__{ds}__{mt}_{rate}__h96_s{seed}",
}

LABELS = {
    "corr": "corr(A)",
    "soft": "soft(B)",
    "fuse": "fuse(E)",
    "sgate": "scalar gate",
    "ggate": "group gate",
    "mgate": "mask gate",
}

COLORS = {
    "corr": "#eda100",
    "soft": "#4a3aa7",
    "fuse": "#e34948",
    "sgate": "#2a78d6",
    "ggate": "#1baf7a",
    "mgate": "#008300",
}


def rate_label(rate: float) -> int:
    return int(round(rate * 100))


def condition_label(cond: tuple[str, str, float]) -> str:
    ds, mt, rate = cond
    mt_short = "rp" if mt == "random_point" else "cs"
    return f"{ds[:4]}-{mt_short}{rate_label(rate)}"


def load_json(path: str) -> dict | None:
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def aggregate_variant(tmpl: str, cond: tuple[str, str, float]) -> dict:
    ds, mt, rate = cond
    mses, maes, times, params, mems = [], [], [], [], []
    for seed in SEEDS:
        tag = tmpl.format(ds=ds, mt=mt, rate=rate_label(rate), seed=seed)
        data = load_json(os.path.join("results", f"{tag}.json"))
        if not data:
            continue
        mses.append(float(data["test"]["mse"]))
        maes.append(float(data["test"]["mae"]))
        params.append(float(data["n_params"]))
        mems.append(float(data.get("peak_mem_mb", float("nan"))))
        hist = data.get("history", [])
        epoch_times = [float(h["train"]["time_sec"]) for h in hist if "train" in h and "time_sec" in h["train"]]
        if epoch_times:
            times.append(float(np.mean(epoch_times)))
    return {
        "mse": float(np.mean(mses)) if mses else float("nan"),
        "mae": float(np.mean(maes)) if maes else float("nan"),
        "std": float(np.std(mses)) if mses else float("nan"),
        "time": float(np.mean(times)) if times else float("nan"),
        "n_params": float(np.mean(params)) if params else float("nan"),
        "peak_mem_mb": float(np.mean(mems)) if mems else float("nan"),
        "n": len(mses),
    }


def load_all() -> dict:
    data = {}
    for variant, tmpl in VARIANTS.items():
        data[variant] = {cond: aggregate_variant(tmpl, cond) for cond in CONDITIONS}
    data["_phase0"] = {
        variant: {cond: aggregate_variant(tmpl, cond) for cond in CONDITIONS}
        for variant, tmpl in PHASE0.items()
    }
    return data


def write_summary_csv(data: dict):
    path = os.path.join(OUT_DIR, "r0711_summary.csv")
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "dataset", "missing_type", "missing_rate", "variant", "mse", "mae",
            "std", "time_sec_per_epoch", "n_params", "peak_mem_mb", "n",
            "improve_vs_fuse_pct", "improve_vs_best_corr_soft_pct",
        ])
        for cond in CONDITIONS:
            ds, mt, rate = cond
            fuse = data["fuse"][cond]["mse"]
            best_corr_soft = min(data["corr"][cond]["mse"], data["soft"][cond]["mse"])
            for variant in VARIANTS:
                row = data[variant][cond]
                mse = row["mse"]
                writer.writerow([
                    ds, mt, rate, variant, mse, row["mae"], row["std"], row["time"],
                    row["n_params"], row["peak_mem_mb"], row["n"],
                    (fuse - mse) / fuse * 100 if fuse and not math.isnan(mse) else float("nan"),
                    (best_corr_soft - mse) / best_corr_soft * 100 if best_corr_soft and not math.isnan(mse) else float("nan"),
                ])
    print("saved", path)


def add_bar_labels(ax, bars):
    for bar in bars:
        h = bar.get_height()
        if h and not math.isnan(h):
            ax.text(bar.get_x() + bar.get_width() / 2, h + max(0.002, h * 0.006),
                    f"{h:.3f}", ha="center", va="bottom", fontsize=6, rotation=45)


def fig1_gate_heatmap(data: dict):
    variants = ["sgate", "ggate", "mgate"]
    mat = np.zeros((len(variants), len(CONDITIONS)))
    for vi, variant in enumerate(variants):
        for ci, cond in enumerate(CONDITIONS):
            base = data["fuse"][cond]["mse"]
            new = data[variant][cond]["mse"]
            mat[vi, ci] = (base - new) / base * 100
    vmax = max(3.0, abs(np.nanmin(mat)), abs(np.nanmax(mat)))
    fig, ax = plt.subplots(figsize=(12, 3.8))
    im = ax.imshow(mat, cmap="RdYlGn", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(CONDITIONS)))
    ax.set_xticklabels([condition_label(c) for c in CONDITIONS], rotation=35, ha="right")
    ax.set_yticks(range(len(variants)))
    ax.set_yticklabels([LABELS[v] for v in variants])
    for vi in range(len(variants)):
        for ci in range(len(CONDITIONS)):
            ax.text(ci, vi, f"{mat[vi, ci]:+.1f}", ha="center", va="center", fontsize=8)
    ax.set_title("Gated variants vs no-gate fuse, MSE improvement %")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig1_gate_vs_fuse_heatmap.png"))
    plt.close(fig)


def fig2_best_baseline_heatmap(data: dict):
    variants = ["fuse", "sgate", "ggate", "mgate"]
    mat = np.zeros((len(variants), len(CONDITIONS)))
    for vi, variant in enumerate(variants):
        for ci, cond in enumerate(CONDITIONS):
            base = min(data["corr"][cond]["mse"], data["soft"][cond]["mse"])
            new = data[variant][cond]["mse"]
            mat[vi, ci] = (base - new) / base * 100
    vmax = max(3.0, abs(np.nanmin(mat)), abs(np.nanmax(mat)))
    fig, ax = plt.subplots(figsize=(12, 4.3))
    im = ax.imshow(mat, cmap="RdYlGn", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(CONDITIONS)))
    ax.set_xticklabels([condition_label(c) for c in CONDITIONS], rotation=35, ha="right")
    ax.set_yticks(range(len(variants)))
    ax.set_yticklabels([LABELS[v] for v in variants])
    for vi in range(len(variants)):
        for ci in range(len(CONDITIONS)):
            ax.text(ci, vi, f"{mat[vi, ci]:+.1f}", ha="center", va="center", fontsize=8)
    ax.set_title("Fusion variants vs best(corr, soft), MSE improvement %")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig2_vs_best_corr_soft_heatmap.png"))
    plt.close(fig)


def grouped_bars(data: dict, conds: list[tuple[str, str, float]], variants: list[str], path: str, title: str):
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(conds))
    w = 0.13
    center = (len(variants) - 1) / 2
    for i, variant in enumerate(variants):
        vals = [data[variant][cond]["mse"] for cond in conds]
        bars = ax.bar(x + (i - center) * w, vals, w, label=LABELS[variant], color=COLORS[variant])
        add_bar_labels(ax, bars)
    ax.set_xticks(x)
    ax.set_xticklabels([condition_label(c) for c in conds])
    ax.set_ylabel("Test MSE")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(ncol=3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, path))
    plt.close(fig)


def fig3_weather(data: dict):
    conds = [
        ("Weather", "continuous_segment", 0.3),
        ("Weather", "continuous_segment", 0.5),
        ("Weather", "continuous_segment", 0.7),
        ("Weather", "random_point", 0.5),
    ]
    grouped_bars(
        data, conds, ["corr", "soft", "fuse", "sgate", "ggate", "mgate"],
        "fig3_weather_gate_bars.png",
        "Weather: corr/soft/fuse and gated variants",
    )


def fig4_traffic(data: dict):
    conds = [
        ("Traffic", "random_point", 0.3),
        ("Traffic", "random_point", 0.5),
        ("Traffic", "random_point", 0.7),
        ("Traffic", "continuous_segment", 0.7),
    ]
    grouped_bars(
        data, conds, ["corr", "soft", "fuse", "sgate", "ggate", "mgate"],
        "fig4_traffic_gate_bars.png",
        "Traffic: corr/soft/fuse and gated variants",
    )


def fig5_efficiency(data: dict):
    variants = ["fuse", "sgate", "ggate", "mgate"]
    datasets = ["Weather", "Traffic", "Electricity"]
    time_mat = np.zeros((len(datasets), len(variants)))
    param_mat = np.zeros((len(datasets), len(variants)))
    for di, ds in enumerate(datasets):
        ds_conds = [c for c in CONDITIONS if c[0] == ds]
        for vi, variant in enumerate(variants):
            time_mat[di, vi] = np.nanmean([data[variant][c]["time"] for c in ds_conds])
            param_mat[di, vi] = np.nanmean([data[variant][c]["n_params"] for c in ds_conds]) / 1000.0
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, mat, title, ylabel in [
        (axes[0], time_mat, "Mean epoch time", "seconds"),
        (axes[1], param_mat, "Parameter count", "K parameters"),
    ]:
        x = np.arange(len(datasets))
        w = 0.18
        for vi, variant in enumerate(variants):
            ax.bar(x + (vi - 1.5) * w, mat[:, vi], w, label=LABELS[variant], color=COLORS[variant])
        ax.set_xticks(x)
        ax.set_xticklabels(datasets)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)
    axes[0].legend(ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig5_efficiency.png"))
    plt.close(fig)


def load_diagnostics() -> list[dict]:
    rows = []
    diag_dir = os.path.join("results", "diagnostics", "r0711")
    if not os.path.isdir(diag_dir):
        return rows
    for name in sorted(os.listdir(diag_dir)):
        if not name.endswith(".json"):
            continue
        data = load_json(os.path.join(diag_dir, name))
        if not data:
            continue
        cfg = data["config"]
        variant = cfg.get("misstsm_variant", "")
        if variant == "grouped_q4_soft":
            short = "soft"
        elif variant == "grouped_q4_fuse":
            short = "fuse"
        elif variant.endswith("_sgate"):
            short = "sgate"
        elif variant.endswith("_ggate"):
            short = "ggate"
        elif variant.endswith("_mgate"):
            short = "mgate"
        else:
            short = variant
        row = {
            "dataset": cfg["dataset"],
            "missing_type": cfg["missing_type"],
            "missing_rate": float(cfg["missing_rate"]),
            "seed": int(cfg["seed"]),
            "variant": short,
            "batches": data.get("batches_analyzed", 0),
        }
        row.update(data.get("scalars", {}))
        vectors = data.get("vectors", {})
        for key, value in vectors.items():
            if isinstance(value, list):
                row[key] = "|".join(f"{x:.4f}" for x in value)
        rows.append(row)
    return rows


def write_diag_csv(rows: list[dict]):
    if not rows:
        return
    keys = [
        "dataset", "missing_type", "missing_rate", "variant", "seed", "batches",
        "route_entropy", "route_effective_groups", "out_a_norm", "out_b_norm",
        "out_ab_cosine", "proj_a_weight_norm", "proj_b_weight_norm", "gate_mean",
        "gate_std", "gate_group_mean", "route_top1_counts",
    ]
    path = os.path.join(OUT_DIR, "r0711_diagnostics.csv")
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print("saved", path)


def fig6_diagnostics(rows: list[dict]):
    if not rows:
        return
    grouped = defaultdict(list)
    for row in rows:
        key = (row["dataset"], row["missing_type"], row["missing_rate"], row["variant"])
        grouped[key].append(row)

    diag_conditions = [
        ("Weather", "continuous_segment", 0.5),
        ("Weather", "continuous_segment", 0.7),
        ("Traffic", "random_point", 0.3),
        ("Traffic", "random_point", 0.5),
        ("Traffic", "continuous_segment", 0.7),
        ("Electricity", "random_point", 0.5),
    ]
    variants = ["soft", "fuse", "sgate", "ggate", "mgate"]
    entropy = np.full((len(variants), len(diag_conditions)), np.nan)
    gate_mean = np.full_like(entropy, np.nan)
    for vi, variant in enumerate(variants):
        for ci, cond in enumerate(diag_conditions):
            vals = grouped.get((*cond, variant), [])
            ent = [float(v["route_entropy"]) for v in vals if "route_entropy" in v]
            gate = [float(v["gate_mean"]) for v in vals if "gate_mean" in v]
            if ent:
                entropy[vi, ci] = np.mean(ent)
            if gate:
                gate_mean[vi, ci] = np.mean(gate)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for ax, mat, title in [
        (axes[0], entropy, "Route entropy (log(4)=1.386; lower is sharper)"),
        (axes[1], gate_mean, "Gate mean (higher = more predefined path A)"),
    ]:
        masked = np.ma.masked_invalid(mat)
        im = ax.imshow(masked, cmap="viridis", aspect="auto")
        ax.set_xticks(range(len(diag_conditions)))
        ax.set_xticklabels([condition_label(c) for c in diag_conditions], rotation=35, ha="right")
        ax.set_yticks(range(len(variants)))
        ax.set_yticklabels([LABELS.get(v, v) for v in variants])
        for vi in range(len(variants)):
            for ci in range(len(diag_conditions)):
                if not np.isnan(mat[vi, ci]):
                    ax.text(ci, vi, f"{mat[vi, ci]:.3f}", ha="center", va="center", fontsize=8, color="white")
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig6_diagnostics.png"))
    plt.close(fig)


def print_key_tables(data: dict, diag_rows: list[dict]):
    print("\nGate variants vs fuse (% improvement):")
    print("condition,sgate,ggate,mgate,best_gate")
    for cond in CONDITIONS:
        vals = {}
        for variant in ["sgate", "ggate", "mgate"]:
            base = data["fuse"][cond]["mse"]
            vals[variant] = (base - data[variant][cond]["mse"]) / base * 100
        best = min(["sgate", "ggate", "mgate"], key=lambda v: data[v][cond]["mse"])
        print(condition_label(cond), *(f"{vals[v]:+.2f}" for v in ["sgate", "ggate", "mgate"]), best, sep=",")

    print("\nMean MSE by dataset:")
    for ds in ["Weather", "Traffic", "Electricity"]:
        conds = [c for c in CONDITIONS if c[0] == ds]
        chunks = []
        for variant in VARIANTS:
            chunks.append(f"{variant}={np.nanmean([data[variant][c]['mse'] for c in conds]):.4f}")
        print(ds, " ".join(chunks))

    if diag_rows:
        print("\nDiagnostics aggregated:")
        grouped = defaultdict(list)
        for row in diag_rows:
            grouped[(row["dataset"], row["missing_type"], row["missing_rate"], row["variant"])].append(row)
        for key in sorted(grouped):
            rows = grouped[key]
            entropy = [float(r["route_entropy"]) for r in rows if "route_entropy" in r]
            gate = [float(r["gate_mean"]) for r in rows if "gate_mean" in r]
            cos = [float(r["out_ab_cosine"]) for r in rows if "out_ab_cosine" in r]
            print(key, "entropy", np.mean(entropy) if entropy else None,
                  "gate", np.mean(gate) if gate else None,
                  "cos", np.mean(cos) if cos else None)


def main():
    data = load_all()
    write_summary_csv(data)
    fig1_gate_heatmap(data)
    fig2_best_baseline_heatmap(data)
    fig3_weather(data)
    fig4_traffic(data)
    fig5_efficiency(data)
    diag_rows = load_diagnostics()
    write_diag_csv(diag_rows)
    fig6_diagnostics(diag_rows)
    print_key_tables(data, diag_rows)
    print("figures written to", OUT_DIR)


if __name__ == "__main__":
    main()
