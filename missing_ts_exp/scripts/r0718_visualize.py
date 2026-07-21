"""0718 experiment visualization: figures required by 实验计划0718.md section 8.3.

Reads the five CSVs built by r0718_build_csvs.py and writes PNGs to
missing_ts_exp/results/0718_block_hmbg/figures/ (primary), with a copy under
missing_ts_exp/docs/figures/r0718/ for report embedding, matching the 0715/
r4 convention (docs/figures/r0715/*.png as report-embedded copies of
results/0715_bitgraph_hdtts/figures/*.png).
"""
from __future__ import annotations

import csv
import os
import shutil
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

ROOT = "/data/wangzuke/time-series-forecast-exp"
CSV_DIR = os.path.join(ROOT, "missing_ts_exp/results/0718_block_hmbg/csv")
OUT_DIR = os.path.join(ROOT, "missing_ts_exp/results/0718_block_hmbg/figures")
DOCS_OUT_DIR = os.path.join(ROOT, "missing_ts_exp/docs/figures/r0718")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(DOCS_OUT_DIR, exist_ok=True)

# Unify BiTGraph's dataset names (Metr/PEMS) with HD-TTS/Block-HMBGNet's
# (la/bay) -- both pairs refer to the same underlying METR-LA / PEMS-BAY
# datasets under different framework conventions.
DATASET_LABEL = {"Metr": "METR-LA", "PEMS": "PEMS-BAY", "la": "METR-LA", "bay": "PEMS-BAY"}
PANEL_DATASETS = ["METR-LA", "PEMS-BAY"]

MODEL_COLORS = {
    "BiaTCGNet": "#4E79A7",
    "hd_tts_amp": "#59A14F",
    "Block-HMBGNet_main": "#E15759",
}
MODEL_MARKERS = {"BiaTCGNet": "o", "hd_tts_amp": "s", "Block-HMBGNet_main": "^"}
MODEL_LABELS = {
    "BiaTCGNet": "BiTGraph (BiaTCGNet)",
    "hd_tts_amp": "HD-TTS-AMP",
    "Block-HMBGNet_main": "Block-HMBGNet (fusion, main)",
}


def read_csv(name):
    path = os.path.join(CSV_DIR, name)
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def save(fig, fname):
    for out_dir in (OUT_DIR, DOCS_OUT_DIR):
        fig.savefig(os.path.join(out_dir, fname))
    plt.close(fig)
    print(f"  Saved {fname}")


def add_bar_labels(ax, bars, fmt="{:.3f}", fontsize=7, rotation=0):
    for bar in bars:
        h = bar.get_height()
        if h and not np.isnan(h):
            ax.text(bar.get_x() + bar.get_width() / 2, h, fmt.format(h),
                     ha="center", va="bottom", fontsize=fontsize, rotation=rotation)


def _mae_rows(baseline_rows, fusion_rows, missing_type):
    """Collect (model_key, dataset_label, rate, mae) tuples for one missing_type
    across BiTGraph + HD-TTS-AMP (from baseline_24to24.csv) and Block-HMBGNet
    main (from fusion_main.csv)."""
    points = []
    for r in baseline_rows:
        if r["missing_type"] != missing_type:
            continue
        model_key = "BiaTCGNet" if r["paper"] == "BiTGraph" else "hd_tts_amp"
        points.append((model_key, DATASET_LABEL[r["dataset"]],
                       float(r["target_missing_rate"]), float(r["mae"])))
    for r in fusion_rows:
        if r["missing_type"] != missing_type:
            continue
        points.append(("Block-HMBGNet_main", DATASET_LABEL[r["dataset"]],
                        float(r["target_missing_rate"]), float(r["mae"])))
    return points


def _plot_mae_vs_rate(points, title, fname):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)
    for ax, ds in zip(axes, PANEL_DATASETS):
        for model_key in ("BiaTCGNet", "hd_tts_amp", "Block-HMBGNet_main"):
            sub = sorted((rate, mae) for mk, d, rate, mae in points if mk == model_key and d == ds)
            if not sub:
                continue
            rates, maes = zip(*sub)
            ax.plot(rates, maes, color=MODEL_COLORS[model_key], marker=MODEL_MARKERS[model_key],
                    label=MODEL_LABELS[model_key], linewidth=1.5, markersize=6)
        ax.set_xlabel("Target missing rate")
        ax.set_ylabel("MAE")
        ax.set_title(ds)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(title, fontsize=13, y=1.02)
    fig.tight_layout()
    save(fig, fname)


def fig_block_t_mae(baseline_rows, fusion_rows):
    points = _mae_rows(baseline_rows, fusion_rows, "temporal_block") + \
        _mae_rows(baseline_rows, fusion_rows, "block_t")
    _plot_mae_vs_rate(points, "MAE vs Missing Rate — Temporal Block Missing (window=24, horizon=24)",
                       "block_t_mae_vs_missing_rate.png")


def fig_block_st_mae(baseline_rows, fusion_rows):
    # BiTGraph has no spatiotemporal-block mode -- only HD-TTS-AMP and
    # Block-HMBGNet appear on this figure; this is disclosed via the legend
    # (BiaTCGNet simply has no line) rather than silently reshaping the panel.
    points = _mae_rows(baseline_rows, fusion_rows, "block_st")
    _plot_mae_vs_rate(points, "MAE vs Missing Rate — Spatiotemporal Block Missing (window=24, horizon=24)\n"
                       "(BiTGraph has no block_st mode; only HD-TTS-AMP / Block-HMBGNet shown)",
                       "block_st_mae_vs_missing_rate.png")


def fig_relative_degradation(baseline_rows, fusion_rows):
    """Relative MAE degradation 30%->80% for block_t and block_st, per model
    per dataset: (mae_80 - mae_30) / mae_30 * 100."""
    all_points = defaultdict(dict)  # (model, dataset, mode) -> {rate: mae}
    for mode in ("temporal_block", "block_t", "block_st"):
        for mk, ds, rate, mae in _mae_rows(baseline_rows, fusion_rows, mode):
            canon_mode = "block_t" if mode == "temporal_block" else mode
            all_points[(mk, ds, canon_mode)][round(rate, 2)] = mae

    labels, degradations, colors = [], [], []
    for mode in ("block_t", "block_st"):
        for ds in PANEL_DATASETS:
            for model_key in ("BiaTCGNet", "hd_tts_amp", "Block-HMBGNet_main"):
                rates_map = all_points.get((model_key, ds, mode))
                if not rates_map or 0.3 not in rates_map or 0.8 not in rates_map:
                    continue
                pct = (rates_map[0.8] - rates_map[0.3]) / rates_map[0.3] * 100
                labels.append(f"{mode}\n{ds}\n{MODEL_LABELS[model_key].split(' ')[0]}")
                degradations.append(pct)
                colors.append(MODEL_COLORS[model_key])

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(labels))
    bars = ax.bar(x, degradations, color=colors)
    add_bar_labels(ax, bars, fmt="{:+.1f}%", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("Relative MAE degradation 30%→80% (%)")
    ax.set_title("Relative Degradation Under Block Missing (30%→80%)")
    ax.grid(axis="y", alpha=0.3)
    ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
    fig.tight_layout()
    save(fig, "relative_degradation_block.png")


ABLATION_ORDER = ["wo_gate", "wo_coarse", "wo_boundary", "graph_everywhere", "fine_only", "coarse_only"]
ABLATION_LABELS = {
    "wo_gate": "w/o gate", "wo_coarse": "w/o coarse graph", "wo_boundary": "w/o boundary graph",
    "graph_everywhere": "graph everywhere (static)", "fine_only": "fine-only (no gate/coarse)",
    "coarse_only": "coarse-only (no gate/boundary)",
}


def fig_fusion_ablation(ablation_rows):
    configs = sorted({(r["dataset"], r["missing_type"], r["target_missing_rate"]) for r in ablation_rows})
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.flatten()
    for i, (ds, mode, rate) in enumerate(configs):
        ax = axes[i]
        sub = {r["ablation"]: float(r["delta_pct_vs_full"]) for r in ablation_rows
               if (r["dataset"], r["missing_type"], r["target_missing_rate"]) == (ds, mode, rate)
               and r["delta_pct_vs_full"]}
        vals = [sub.get(v, np.nan) for v in ABLATION_ORDER]
        colors = ["#E15759" if v > 0 else "#59A14F" for v in vals]
        bars = ax.bar(range(len(ABLATION_ORDER)), vals, color=colors)
        add_bar_labels(ax, bars, fmt="{:+.1f}%", fontsize=7)
        ax.set_xticks(range(len(ABLATION_ORDER)))
        ax.set_xticklabels([ABLATION_LABELS[v] for v in ABLATION_ORDER], fontsize=7, rotation=30, ha="right")
        ax.set_ylabel("ΔMAE vs main (%)")
        ax.set_title(f"{DATASET_LABEL[ds]} / {mode} / {float(rate):.0%}")
        ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Block-HMBGNet Module Ablation — MAE Change vs Full Model\n"
                 "(positive = worse than full model; single seed per bar, see report caveats)",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    save(fig, "fusion_ablation.png")


def fig_batch_size_throughput():
    """HD-TTS batch-size probe throughput (from notes/batch_size_notes.md,
    isolated single-job measurement -- the only clean, non-co-located timing
    data available this round). Peak GPU memory was not sampled live during
    the probe (see notes/batch_size_notes.md), so this figure shows
    throughput only, not a memory/throughput dual-axis plot."""
    batches = [64, 128, 256, 512]
    samples_per_sec = [1190, 1277, 1802, 2084]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(batches, samples_per_sec, color="#59A14F", marker="o", linewidth=2, markersize=8)
    for b, s in zip(batches, samples_per_sec):
        ax.annotate(f"{s}", (b, s), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9)
    ax.set_xscale("log", base=2)
    ax.set_xticks(batches)
    ax.set_xticklabels([str(b) for b in batches])
    ax.set_xlabel("Batch size")
    ax.set_ylabel("Throughput (samples/sec, approx.)")
    ax.set_title("HD-TTS-AMP Throughput vs Batch Size\n"
                  "(5-epoch isolated probe, la/block_t_50; peak GPU memory not sampled)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    save(fig, "batch_size_throughput.png")


def fig_efficiency_tradeoff(baseline_rows, fusion_rows):
    """MAE vs n_params as an efficiency-tradeoff proxy. Wall-clock time and
    peak GPU memory could not be captured this round (see efficiency.csv
    notes / notes/implementation_notes.md), so this figure substitutes model
    size (n_params) on the x-axis rather than fabricating timing data."""
    points = []
    for r in baseline_rows:
        model_key = "BiaTCGNet" if r["paper"] == "BiTGraph" else "hd_tts_amp"
        if r["n_params"]:
            points.append((model_key, float(r["n_params"]), float(r["mae"])))
    for r in fusion_rows:
        if r["n_params"]:
            points.append(("Block-HMBGNet_main", float(r["n_params"]), float(r["mae"])))

    fig, ax = plt.subplots(figsize=(8, 6))
    for model_key in ("BiaTCGNet", "hd_tts_amp", "Block-HMBGNet_main"):
        sub = [(p, m) for mk, p, m in points if mk == model_key]
        if not sub:
            continue
        xs, ys = zip(*sub)
        ax.scatter(xs, ys, color=MODEL_COLORS[model_key], marker=MODEL_MARKERS[model_key],
                   label=MODEL_LABELS[model_key], alpha=0.7, s=40)
    ax.set_xscale("log")
    ax.set_xlabel("n_params (log scale)")
    ax.set_ylabel("MAE")
    ax.set_title("MAE vs Model Size\n"
                 "(x-axis substitutes for wall-clock time / GPU memory,\n"
                 "neither of which was instrumented this round)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    save(fig, "efficiency_tradeoff_0718.png")


def note_missing_gate_behavior_figure():
    """gate_behavior_boundary_vs_center.png is intentionally NOT generated:
    diagnostics.csv is empty (gate_entropy/coarse_gate_mean/boundary_graph_
    weight_mean were never logged by predictor.py's test_step(), see
    notes/diagnostics_gap.md) -- there is no data to plot. Document the
    skip explicitly instead of fabricating a placeholder figure."""
    print("  Skipped gate_behavior_boundary_vs_center.png -- diagnostics.csv "
          "is empty (see notes/diagnostics_gap.md); no gate/attention values "
          "were ever logged during Phase 4 runs.")


if __name__ == "__main__":
    baseline_rows = read_csv("baseline_24to24.csv")
    fusion_rows = read_csv("fusion_main.csv")
    ablation_rows = read_csv("ablation.csv")

    print("Generating 0718 visualizations...")

    print("\n[1/7] block_t MAE vs missing rate")
    fig_block_t_mae(baseline_rows, fusion_rows)

    print("\n[2/7] block_st MAE vs missing rate")
    fig_block_st_mae(baseline_rows, fusion_rows)

    print("\n[3/7] relative degradation 30%->80%")
    fig_relative_degradation(baseline_rows, fusion_rows)

    print("\n[4/7] fusion module ablation")
    fig_fusion_ablation(ablation_rows)

    print("\n[5/7] batch size throughput")
    fig_batch_size_throughput()

    print("\n[6/7] gate behavior boundary vs center (skipped, no data)")
    note_missing_gate_behavior_figure()

    print("\n[7/7] efficiency tradeoff (MAE vs n_params proxy)")
    fig_efficiency_tradeoff(baseline_rows, fusion_rows)

    n_out = len([f for f in os.listdir(OUT_DIR) if f.endswith(".png")])
    print(f"\nDone! {n_out} figures saved to {OUT_DIR}/ (copies in {DOCS_OUT_DIR}/)")
