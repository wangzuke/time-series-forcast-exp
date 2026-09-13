#!/usr/bin/env python3
"""Render the validated horizon decomposition report."""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path("/media/data1/wangzk/time-series-forcast-exp")
SUMMARY = ROOT / "missing_ts_exp/results/r0910_evidence_closure/stage_d_horizon_summary.json"
REPORT = ROOT / "missing_ts_exp/docs/0910实验报告_METRLA_BlockT预测跨度分解.md"


def fmt(value: dict[str, float]) -> str:
    return f"{value['mean']:.4f} +/- {value['std']:.4f}"


def main() -> None:
    if not SUMMARY.is_file():
        raise FileNotFoundError(f"Run r0910_verify_stage_d_horizon.py first: {SUMMARY}")
    result = json.loads(SUMMARY.read_text())
    if result.get("status") != "passed":
        raise ValueError("Refusing to write an unverified horizon report.")
    lines = ["# 0910 实验报告：METR-LA Block-T 预测跨度分解", "", "## 1. 协议", "",
             "该后验分析加载已验证的 HD-TTS-AMP 与 MR-MSGF checkpoint，不重训模型。在 Block-T 50%/70%、五个配对 mask seed 下，分别在第 3、6、12 个未来预测步计算原始有效目标的未遮蔽 MAE/RMSE。", "",
             "## 2. 结果", "", "| 缺失率 | horizon | 方法 | MAE | RMSE | 相对 MR-MSGF 的配对 MAE 差值 |", "|---:|---:|---|---:|---:|---:|"]
    for rate in result["rates"]:
        for horizon in result["horizons"]:
            for model, label in (("official_base", "HD-TTS-AMP"), ("mrmsgf", "MR-MSGF")):
                group = result["groups"][f"r{rate}_{model}_h{horizon}"]
                delta = "0.0000" if model == "mrmsgf" else fmt(group["paired_mae_minus_mrmsgf"])
                lines.append(f"| {rate}% | {horizon} | {label} | {fmt(group['mae'])} | {fmt(group['rmse'])} | {delta} |")
    lines += [
        "",
        "## 3. 推理效率", "", "| 方法 | 参数量 | 每测试 batch 推理秒数 |", "|---|---:|---:|",
        *[f"| {label} | {result['efficiency'][model]['n_model_parameters']} | {fmt(result['efficiency'][model]['seconds_per_batch'])} |"
          for model, label in (("official_base", "HD-TTS-AMP"), ("mrmsgf", "MR-MSGF"))],
        "",
        "## 4. 读法与边界", "",
        "正的配对差值表示 MR-MSGF MAE 更低。该分析分解同一主实验的预测跨度，不增加新的训练重复；它可说明效果是否集中于近端或远端预测步，但不替代跨数据集验证。推理时间在相同设备的逐个 checkpoint 后验运行中测得，主要用于报告量级，不替代严格的端到端吞吐基准。",
        "",
        "## 5. 产物", "", f"- 自动汇总：[stage_d_horizon_summary.json]({SUMMARY})。", "",
    ]
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(REPORT)


if __name__ == "__main__":
    main()
