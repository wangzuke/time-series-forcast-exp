#!/usr/bin/env python3
"""Merge audited fixed-configuration OCPE evidence; no training or inference."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SUITES = ("posterior_s4_r50_b24_l96_h96_1119", "posterior_s4_weather_r50_b24_l96_h96_1119")
OUTPUT = ROOT / "summary/ocpe_s4_three_datasets_1120"
DATASETS = ("Electricity", "Weather", "ETTh1")
CONDITIONS = (("P", "P"), ("P", "B"), ("B", "P"), ("B", "B"))


def read(path):
    return json.loads(path.read_text())


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    fields = ("cells", "seed_cells", "comparisons", "mismatch_costs", "uncertainty", "training", "mask_audits")
    merged = {name: [] for name in fields}
    audits, protocol_reference = [], None
    unique_jobs = set()
    common_keys = ("batch", "effective_batch", "workers", "threads", "epochs", "patience", "aux_weight",
                   "optimizer", "optimizer_betas", "gradient_clip_norm", "protocol")
    for suite in SUITES:
        analysis = read(ROOT / "summary" / suite / "analysis/analysis.json")
        assert not analysis["audit"]["errors"]
        assert analysis["audit"]["checkpoints_deserialized_on_cpu"]
        audits.append({"suite": suite, **analysis["audit"]})
        for name in fields:
            merged[name].extend({"suite": suite, **row} for row in analysis[name])
        manifest = read(ROOT / "configs" / f"{suite}.json")
        for job in manifest["jobs"]:
            assert job["id"] not in unique_jobs
            unique_jobs.add(job["id"])
            record = read(ROOT / "results" / suite / job["id"] / "protocol.json")
            signature = {k: record[k] for k in common_keys}
            signature["model"] = {k: v for k, v in record["model"].items() if k != "channels"}
            if protocol_reference is None:
                protocol_reference = signature
            assert signature == protocol_reference, job["id"]
    assert len(unique_jobs) == 36
    assert sum(a["evaluations"] for a in audits) == 216
    assert sum(a["checkpoint_files"] for a in audits) == 72
    cell_keys = [(r["dataset"], r["variant"], r["train"], r["test"]) for r in merged["cells"]]
    assert len(cell_keys) == len(set(cell_keys)) == 54
    cells = {key: row for key, row in zip(cell_keys, merged["cells"])}

    decision = []
    for data in DATASETS:
        changes = [r for r in merged["comparisons"] if r["dataset"] == data and r["baseline"] == "LI-S4" and r["test"] != "B-uniform"]
        matched = [r["mse_standardized_delta_percent"] for r in changes if r["train"] == r["test"]]
        shifted = [r["mse_standardized_delta_percent"] for r in changes if r["train"] != r["test"]]
        seed_consistent = all(r["mse_delta_seed2024_percent"] < 0 and r["mse_delta_seed2025_percent"] < 0
                              for r in changes if r["train"] != r["test"])
        decision.append(dict(dataset=data, matched_mean_delta_percent=float(np.mean(matched)),
                             mismatch_mean_delta_percent=float(np.mean(shifted)),
                             mismatch_improves_both_seeds=seed_consistent,
                             practical_screen_pass=bool(np.mean(shifted) <= -3 and np.mean(matched) <= 2 and seed_consistent)))
    merged["decision_summary"] = decision
    merged.update(audit={"training_jobs": 36, "evaluations": 216, "checkpoint_files": 72,
                         "all_suites_audited": True, "hyperparameters_equal_except_channel_count": True,
                         "errors": []}, source_audits=audits, common_protocol=protocol_reference,
                  method_alias={"PU-S4": "OCPE-S4"},
                  aggregation="average masks within checkpoint, then training seeds; no pooled raw-scale dataset ranking")
    for name in fields + ("decision_summary",):
        write_csv(OUTPUT / f"{name}.csv", merged[name])
    (OUTPUT / "analysis.json").write_text(json.dumps(merged, ensure_ascii=False, indent=2) + "\n")

    figures = ROOT / "figures/ocpe_s4_three_datasets_1120"
    figures.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "svg.fonttype": "none"})
    fig, axes = plt.subplots(3, 2, figsize=(11.5, 9.0), sharey=True, layout="constrained")
    for row, data in enumerate(DATASETS):
        for col, baseline in enumerate(("LI-S4", "STAT-S4")):
            ax = axes[row, col]
            selected = {(r["train"], r["test"]): r for r in merged["comparisons"]
                        if r["dataset"] == data and r["baseline"] == baseline}
            values = [selected[c]["mse_standardized_delta_percent"] for c in CONDITIONS]
            colors = ["#4C78A8" if a == b else "#E69F00" for a, b in CONDITIONS]
            ax.bar(np.arange(4), values, color=colors, alpha=.86, width=.65, zorder=3)
            for i, condition in enumerate(CONDITIONS):
                seed_values = [selected[condition][f"mse_delta_seed{seed}_percent"] for seed in (2024, 2025)]
                ax.scatter([i - .07, i + .07], seed_values, color="#222222", s=15, zorder=5)
                y = max(values[i], *seed_values) + 2.0 if values[i] >= 0 else min(values[i], *seed_values) - 2.0
                ax.text(i, y, f"{values[i]:+.1f}%", ha="center", va="bottom" if values[i] >= 0 else "top", fontsize=9)
            ax.axhline(0, color="#444444", lw=.9, zorder=4)
            ax.grid(axis="y", alpha=.2, zorder=0)
            ax.set_xticks(np.arange(4), [f"{a} -> {b}" for a, b in CONDITIONS])
            ax.set_ylim(-62, 49)
            ax.set_title(f"{data}: OCPE-S4 vs {baseline}", fontsize=11)
            if col == 0:
                ax.set_ylabel("Change in standardized MSE (%)")
    fig.suptitle("OCPE-S4: matched accuracy and missing-pattern mismatch", fontsize=14)
    fig.legend(handles=[Patch(color="#4C78A8", label="Matched"), Patch(color="#E69F00", label="Mismatched"),
                        plt.Line2D([], [], marker="o", linestyle="", color="#222222", markersize=4, label="Individual training seeds")],
               loc="outside lower center", ncol=3, frameon=False)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(figures / f"relative_mse.{ext}", dpi=180, bbox_inches="tight")
    plt.close(fig)

    lines = ["# OCPE-S4：三数据集机器汇总", "", "36 次训练、216 项测试、72 个检查点；两个套件均审计通过。", "",
             "PU-S4 为原始实验 ID，展示名为 OCPE-S4。主指标为标准化 MSE，按检查点内掩码、训练种子依次平均。", "",
             "| 数据集 | 版本 | P→P | P→B | B→P | B→B |", "|---|---|---:|---:|---:|---:|"]
    for data in DATASETS:
        for variant in ("LI-S4", "STAT-S4", "PU-S4"):
            scores = [cells[(data, variant, tr, te)]["mse_standardized"] for tr, te in CONDITIONS]
            lines.append("| " + " | ".join([data, "OCPE-S4" if variant == "PU-S4" else variant] + [f"{x:.6f}" for x in scores]) + " |")
    lines += ["", "完整分析报告：[OCPE-S4 三数据集结果与分析](../../../docs/ocpe/1120-实验结果-OCPE-S4-三数据集-r50-2x2矩阵.md)。", ""]
    (OUTPUT / "report.md").write_text("\n".join(lines))
    print(json.dumps({"audit": merged["audit"], "decisions": decision, "output": str(OUTPUT), "figures": str(figures)}))


if __name__ == "__main__":
    main()
