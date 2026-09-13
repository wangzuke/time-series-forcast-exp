# 0908 实验报告：官方基线与 Block-T 受控验证

## 1. 结论摘要

本轮正式实验已完成并通过自动校验，共包含 12 次官方 HD-TTS 运行、12 次 HD-TTS-AMP 与 MR-MSGF 的受控 Block-T 对比，以及 9 次 BiTGraph 官方 Point-missing 运行。

在与 HD-TTS-AMP 完全配对的 Block-T 主比较中，MR-MSGF 在约 50% 和约 70% 的缺失率下均降低了 MAE。50% 缺失率下，MR-MSGF 同时降低 MAE 与 RMSE；70% 缺失率下，MAE 降低，但 RMSE 略高。因此目前的严谨表述是：MR-MSGF 对 Block-T 下的平均误差有稳定改进证据，尤其在 50% 缺失率下；高缺失率下的平方误差收益尚不稳定，不能宣称对所有指标均有提升。

## 2. 实验完成情况与可追溯性

| 组别 | 协议 | 完成次数 | 实现来源 |
|---|---|---:|---|
| A | METR-LA、HD-TTS 官方 Block-T | 4 模型 x 3 训练 seed = 12 | HD-TTS 作者代码，commit `46a8717db3802e1684e991ba0a7c0bfe43d22535` |
| B | METR-LA、BiTGraph 官方 Point missing | 3 缺失率 x 3 独立重复 = 9 | BiTGraph 作者代码，commit `4f1d05bcc20bb3f5084bd1facc995478f84a40ed` |
| C | METR-LA、固定 Block-T | 2 方法 x 2 缺失率 x 3 seed = 12 | HD-TTS 官方 LA 流程 + MR-MSGF 完整图模块 |

- 数据文件为 `dataset/METR-LA/metr-la.h5`，SHA-256 为 `64784b76d6fb8ec9bff4b6decafb354da2bb37840468fdccee5044e511277c05`。
- C 组中每个 HD-TTS-AMP/MR-MSGF 配对复用同一份预生成观测 mask；结果 JSON 已记录各 mask 的 SHA-256。
- 自动校验检查了 12 份 C 组 JSON 的有限指标与掩码记录、12 份 HD-TTS 官方日志，以及 9 份 BiTGraph 的 `best.pth` 和测试日志；全部通过。
- 两个官方仓库的受版本控制源码均无 diff。BiTGraph 的运行产物仅为未跟踪的 checkpoint、日志和数据软链接。

## 3. 主结果：固定 Block-T 受控比较

评测指标为原始有效观测目标上的 unmasked MAE/RMSE，报告 `mean +/- sample std`（3 个配对 seed）。缺失率为观测掩码的实际约值，目标场景是目标节点在时间轴上缺失而仍可利用空间邻居观测的 Block-T。

| Block-T 缺失率 | 方法 | MAE | RMSE |
|---:|---|---:|---:|
| 约 50% | HD-TTS-AMP（官方基线） | 3.3500 +/- 0.0027 | 6.8761 +/- 0.0289 |
| 约 50% | MR-MSGF | **3.3156 +/- 0.0027** | **6.8201 +/- 0.0263** |
| 约 70% | HD-TTS-AMP（官方基线） | 3.4696 +/- 0.0092 | **7.0824 +/- 0.0547** |
| 约 70% | MR-MSGF | **3.4445 +/- 0.0119** | 7.1087 +/- 0.0162 |

相对官方基线的均值改变量：

| Block-T 缺失率 | MAE 改变量 | RMSE 改变量 | 解读 |
|---:|---:|---:|---|
| 约 50% | -0.0343 (-1.03%) | -0.0560 (-0.81%) | 两个主指标均改善 |
| 约 70% | -0.0251 (-0.72%) | +0.0264 (+0.37%) | MAE 改善，RMSE 轻微退化 |

该表只支持 C 组内的结论：两方法的时间编码、官方 LA 物理距离图、协变量、数据划分、优化器、学习率调度、训练预算和评测目标一致；区别仅为 MR-MSGF 替换了 HD-TTS-AMP 的图关系与细到粗尺度传播部分。由于样本量为 3，未做显著性检验，后续应增加 mask seed 后再报告统计显著性。

## 4. 官方 HD-TTS 复现结果

这一组用作者运行器原生生成 Block-T 掩码，故只用于复现四种官方内部变体，不与第 3 节的固定掩码主表逐格比较。下表为官方日志中 unmasked MAE 的三次运行均值。

| 官方模型 | unmasked MAE |
|---|---:|
| HD-TTS-AMP | 3.2141 +/- 0.0090 |
| HD-TTS-IMP | 3.2224 +/- 0.0055 |
| TTS-AMP | 3.1843 +/- 0.0122 |
| TTS-IMP | 3.1646 +/- 0.0284 |

这些值的掩码由官方脚本在线生成，其缺失实例、评价细节与第 3 节不同；不能据此直接排出 MR-MSGF 与这些条目的名次。

## 5. 官方 BiTGraph Point-missing 复现

BiTGraph 作者公开代码只接收 `mask_ratio`，不支持传入外部 Block-T mask。因而保持其原始 Point-missing、`seq_len=24`、`pred_len=24`、`horizon=24` 和 `test_forecasting.py` 评价流程，结果单列如下。

| Point 缺失率 | MAE | RMSE |
|---:|---:|---:|
| 0.3 | 6.1400 +/- 0.1044 | 10.8700 +/- 0.2339 |
| 0.5 | 6.2567 +/- 0.0723 | 10.6833 +/- 0.0153 |
| 0.7 | 6.3833 +/- 0.0643 | 10.9467 +/- 0.1168 |

作者源码会解析 `--seed`，但未用其设置 NumPy 或 PyTorch 的随机状态。因此这三次是独立的官方重复，不是严格受控的 seed 实验。它们不与第 3 节的 Block-T/12 步主比较直接比较。

## 6. 产物与后续

- 正式计划：[0908实验计划_官方基线与MRMSGF受控验证.md](0908实验计划_官方基线与MRMSGF受控验证.md)
- 自动校验汇总：`missing_ts_exp/results/r0908_official_baselines/formal_verification_summary.json`
- 受控运行原始 JSON：`missing_ts_exp/results/r0908_official_baselines/controlled/`
- 官方 HD-TTS 日志与 checkpoint：`missing_ts_exp/results/r0908_official_baselines/official_hdtts/`
- 官方 BiTGraph 逐次原始日志与 checkpoint：`missing_ts_exp/results/r0908_official_baselines/official_bitgraph/`

建议的下一步不是扩展 Block-ST，而是先在同一 Block-T 设定上增加独立 mask seed，并完成 MR-MSGF 的逐模块消融（观测关系图、物理图融合、区域粗图、可靠性门控），以定位 70% 缺失率下 RMSE 退化的来源。
