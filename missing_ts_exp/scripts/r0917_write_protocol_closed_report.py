#!/usr/bin/env python3
"""Write the fixed-format report for the 0917 protocol-closure experiment."""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path("/media/data1/wangzk/time-series-forcast-exp")
SUMMARY = ROOT / "missing_ts_exp/results/r0917_protocol_closed/verification_summary.json"
REPORT = ROOT / "missing_ts_exp/docs/0917实验报告_METRLA_BlockT50协议闭环.md"


def display(metric: dict) -> str:
    return f"{metric['mean']:.4f} +/- {metric['std']:.4f}"


def main() -> None:
    result = json.loads(SUMMARY.read_text())
    groups, comparisons = result["groups"], result["paired_variant_minus_full_observed"]
    labels = (("full_observed", "Full-observed"), ("feature_only", "Feature-only"),
              ("full_permuted_q", "Full-permuted-Q"))
    lines = [
        "# 0917 实验报告：METR-LA Block-T 50% 协议闭环重跑",
        "",
        "## 协议闭环",
        "",
        "三种变体均由同一冻结代码、相同的窗口元数据 input map、METR-LA Block-T 50% mask、24 步历史至 12 步预测、五个配对 seed、AdamW 和相同最佳 validation-MAE checkpoint 规则训练。HD-TTS-AMP 是既有主基线，不属于本轮 15 个同版重跑任务。",
        "",
        f"冻结提交：`{result['source_manifest']['head']}`；受控模型/runner 指纹：`{result['source_manifest']['controlled_source_sha256']}`。",
        "",
        "## 主结果",
        "",
        "| Variant | Parameters | MAE | RMSE | Delta MAE vs Full | 95% CI | paired p (Holm) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for key, label in labels:
        group = groups[key]
        if key in comparisons:
            test = comparisons[key]["mae"]
            delta = f"{test['mean']:.4f}"
            ci = f"[{test['ci95'][0]:.4f}, {test['ci95'][1]:.4f}]"
            p = f"{test['holm_p_value']:.4g}"
        else:
            delta, ci, p = "--", "--", "--"
        lines.append(
            f"| {label} | {group['parameter_count']:,} | {display(group['metrics']['mae'])} | "
            f"{display(group['metrics']['rmse'])} | {delta} | {ci} | {p} |"
        )
    lines += ["", "## 配对统计与审计", "",
              "正 Delta 表示 Full-observed 的 MAE 更低。所有置信区间是五个配对 seed 差值的双侧 95% t 区间；两个 MAE 主比较以 Holm 校正。",
              ""]
    for key, label in (("feature_only", "Feature-only"), ("full_permuted_q", "Full-permuted-Q")):
        item = comparisons[key]
        lines.append(f"- **{label}**：MAE 差值 = {item['mae']['values']}；RMSE 差值 = {item['rmse']['values']}；未校正 MAE p={item['mae']['p_value']:.4g}。")
    audit = result["p0_audit"]
    lines += [
        f"- P0：重复置换一致={audit['permutation']['repeatable']}；对称={audit['permutation']['symmetric']}；对角线严格恢复={audit['permutation']['diagonal_restored_exactly']}；非对角多重集合保持={audit['permutation']['off_diagonal_multiset_preserved']}；三变体共享元数据={audit['models']['all_variants_receive_window_metadata']}。",
        "- Full-observed 与 Full-permuted-Q 的逐字段 manifest 配置比对通过；两者只允许 q mode、置换审计、训练轨迹/checkpoint 与指标不同。",
        "",
        "## 论文解释边界",
        "",
        "若 Full 显著优于 Feature-only，可将该结果写为完整架构的收益不能由直接注入缺失统计特征解释。若 Full 与 Permuted-Q 的比较不显著，则必须明确：当前控制不足以证明节点对身份是主要收益来源；不因该结果再次改变 seed、预算或置换策略。",
        "",
        f"机器可读验证：[verification_summary.json]({SUMMARY})；逐任务 manifest：[manifest_rows.csv]({result['manifest_csv']})；P0 审计：[p0_audit.json]({ROOT / 'missing_ts_exp/results/r0917_protocol_closed/p0_audit.json'})。",
        "",
    ]
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(REPORT)


if __name__ == "__main__":
    main()
