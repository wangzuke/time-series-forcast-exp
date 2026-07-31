#!/usr/bin/env python
"""Create dependency-free SVG figures for the formal R0723 report."""

from __future__ import annotations

import csv
import html
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
CSV_DIR = ROOT / "missing_ts_exp/results/0723_official_pguts_coarse_imputation/csv"
FIG_DIR = ROOT / "missing_ts_exp/figures_r0723_pguts_cg"

PAPER_MAIN = {"METR-LA": 1.92, "PEMS-BAY": 0.93}
PAPER_TABLE3 = {
    ("METR-LA", 0.05): 2.53,
    ("METR-LA", 0.10): 3.07,
    ("METR-LA", 0.15): 3.80,
    ("PEMS-BAY", 0.05): 1.69,
    ("PEMS-BAY", 0.10): 2.20,
    ("PEMS-BAY", 0.15): 2.86,
}


def read_csv(path: Path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def as_float(row, key):
    return float(row[key])


def setup():
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def svg_text(x, y, text, size=13, fill="#111827", anchor="middle", weight="400"):
    return (
        f'<text x="{x}" y="{y}" text-anchor="{anchor}" '
        f'font-family="Arial, sans-serif" font-size="{size}" '
        f'font-weight="{weight}" fill="{fill}">{html.escape(str(text))}</text>'
    )


def svg_line(x1, y1, x2, y2, stroke="#d1d5db", width=1):
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{stroke}" stroke-width="{width}"/>'


def write_svg(path: Path, width: int, height: int, body):
    content = "\n".join(
        [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
            '<rect width="100%" height="100%" fill="white"/>',
            *body,
            "</svg>",
        ]
    )
    path.write_text(content)


def plot_main():
    rows = read_csv(CSV_DIR / "bs16be300_main_summary.csv")
    datasets = ["METR-LA", "PEMS-BAY"]
    methods = ["Paper P-GUTS", "Ours P-GUTS", "Ours CG-P-GUTS"]
    colors = ["#6b7280", "#2563eb", "#dc2626"]

    values = []
    for dataset in datasets:
        ours_pguts = next(
            as_float(r, "mae_mean")
            for r in rows
            if r["dataset"] == dataset and r["method"] == "P-GUTS [3,6]"
        )
        ours_cg = next(
            as_float(r, "mae_mean")
            for r in rows
            if r["dataset"] == dataset and r["method"] == "CG-P-GUTS [3,6]"
        )
        values.append([PAPER_MAIN[dataset], ours_pguts, ours_cg])

    width, height = 760, 420
    left, right, top, bottom = 70, 30, 60, 70
    chart_w = width - left - right
    chart_h = height - top - bottom
    vmax = max(max(v) for v in values) * 1.15
    body = [svg_text(width / 2, 28, "Main BLOCK Benchmark", 16, weight="700")]
    for tick in np.linspace(0, vmax, 6):
        y = top + chart_h - tick / vmax * chart_h
        body.append(svg_line(left, y, left + chart_w, y))
        body.append(svg_text(left - 10, y + 4, f"{tick:.1f}", 11, "#6b7280", "end"))
    body.append(svg_line(left, top, left, top + chart_h, "#111827", 1.2))
    body.append(svg_line(left, top + chart_h, left + chart_w, top + chart_h, "#111827", 1.2))

    group_w = chart_w / len(datasets)
    bar_w = 46
    offsets = [-bar_w - 7, 0, bar_w + 7]
    for d_i, dataset in enumerate(datasets):
        cx = left + group_w * d_i + group_w / 2
        for m_i, method in enumerate(methods):
            val = values[d_i][m_i]
            h = val / vmax * chart_h
            x = cx + offsets[m_i] - bar_w / 2
            y = top + chart_h - h
            body.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w}" height="{h:.1f}" fill="{colors[m_i]}"/>')
            body.append(svg_text(x + bar_w / 2, y - 6, f"{val:.2f}", 11, "#111827"))
        body.append(svg_text(cx, top + chart_h + 28, dataset, 13, weight="700"))
    for i, method in enumerate(methods):
        lx = left + i * 190
        ly = height - 22
        body.append(f'<rect x="{lx}" y="{ly - 11}" width="13" height="13" fill="{colors[i]}"/>')
        body.append(svg_text(lx + 20, ly, method, 12, anchor="start"))
    body.append(svg_text(18, top + chart_h / 2, "MAE", 12, "#111827", anchor="middle"))
    write_svg(FIG_DIR / "bs16be300_main_mae.svg", width, height, body)


def plot_table3():
    rows = read_csv(CSV_DIR / "bs16be300_table3_summary.csv")
    pfs = [0.05, 0.10, 0.15]
    width, height = 920, 380
    colors = {
        "Paper P-GUTS": "#6b7280",
        "Ours P-GUTS": "#2563eb",
        "Ours CG-P-GUTS": "#dc2626",
    }

    body = [svg_text(width / 2, 28, "Table 3 Robustness Inference", 16, weight="700")]
    panel_w, panel_h = 390, 250
    panels = [(70, 65, "METR-LA"), (510, 65, "PEMS-BAY")]
    for left, top, dataset in panels:
        paper = [PAPER_TABLE3[(dataset, pf)] for pf in pfs]
        pguts = [
            as_float(r, "mae_mean")
            for pf in pfs
            for r in rows
            if r["dataset"] == dataset
            and r["method"] == "P-GUTS [3,6]"
            and abs(as_float(r, "p_fault_eval") - pf) < 1e-12
        ]
        cg = [
            as_float(r, "mae_mean")
            for pf in pfs
            for r in rows
            if r["dataset"] == dataset
            and r["method"] == "CG-P-GUTS [3,6]"
            and abs(as_float(r, "p_fault_eval") - pf) < 1e-12
        ]
        series_map = [
            ("Paper P-GUTS", paper),
            ("Ours P-GUTS", pguts),
            ("Ours CG-P-GUTS", cg),
        ]
        vmax = max(max(v) for _, v in series_map) * 1.12
        body.append(svg_text(left + panel_w / 2, top - 15, dataset, 14, weight="700"))
        for tick in np.linspace(0, vmax, 5):
            y = top + panel_h - tick / vmax * panel_h
            body.append(svg_line(left, y, left + panel_w, y))
            body.append(svg_text(left - 8, y + 4, f"{tick:.1f}", 10, "#6b7280", "end"))
        body.append(svg_line(left, top, left, top + panel_h, "#111827", 1.2))
        body.append(svg_line(left, top + panel_h, left + panel_w, top + panel_h, "#111827", 1.2))
        x_pos = {0.05: left, 0.10: left + panel_w / 2, 0.15: left + panel_w}
        for pf in pfs:
            body.append(svg_text(x_pos[pf], top + panel_h + 24, f"{int(pf * 100)}", 11))
        body.append(svg_text(left + panel_w / 2, top + panel_h + 46, "failure probability (%)", 11))
        body.append(svg_text(left - 50, top + panel_h / 2, "MAE", 11))
        for label, series in series_map:
            points = []
            for pf, val in zip(pfs, series):
                x = x_pos[pf]
                y = top + panel_h - val / vmax * panel_h
                points.append((x, y, val))
            for (x1, y1, _), (x2, y2, _) in zip(points[:-1], points[1:]):
                body.append(svg_line(x1, y1, x2, y2, colors[label], 2.4))
            for x, y, val in points:
                body.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{colors[label]}"/>')
                body.append(svg_text(x, y - 8, f"{val:.2f}", 10, colors[label]))
    legend_x = 290
    legend_y = height - 24
    for i, label in enumerate(["Paper P-GUTS", "Ours P-GUTS", "Ours CG-P-GUTS"]):
        lx = legend_x + i * 150
        body.append(svg_line(lx, legend_y - 5, lx + 22, legend_y - 5, colors[label], 3))
        body.append(f'<circle cx="{lx + 11}" cy="{legend_y - 5}" r="4" fill="{colors[label]}"/>')
        body.append(svg_text(lx + 30, legend_y, label, 11, anchor="start"))
    write_svg(FIG_DIR / "bs16be300_table3_mae.svg", width, height, body)


def main():
    setup()
    plot_main()
    plot_table3()
    print(FIG_DIR / "bs16be300_main_mae.svg")
    print(FIG_DIR / "bs16be300_table3_mae.svg")


if __name__ == "__main__":
    main()
