# 0909 实验报告：统一 Block-T 主实验结果

## 结论

主实验采用 METR-LA、24 步历史、12 步预测、固定 Block-T 掩码及原始有效目标上的 unmasked 评测。每个设置使用三个配对 mask seed（2024/2025/2026），报告 mean +/- sample std。

| 缺失率 | 方法 | MAE | RMSE |
|---:|---|---:|---:|
| 约 50% | HD-TTS-AMP | 3.3500 +/- 0.0027 | 6.8761 +/- 0.0289 |
| 约 50% | BiTGraph | 3.9659 +/- 0.0191 | 8.1596 +/- 0.0206 |
| 约 50% | MR-MSGF | **3.3156 +/- 0.0027** | **6.8201 +/- 0.0263** |
| 约 70% | HD-TTS-AMP | 3.4696 +/- 0.0092 | **7.0824 +/- 0.0547** |
| 约 70% | BiTGraph | 4.1281 +/- 0.0064 | 8.4853 +/- 0.0217 |
| 约 70% | MR-MSGF | **3.4445 +/- 0.0119** | 7.1087 +/- 0.0162 |

MR-MSGF 在两种缺失率下均取得最低 MAE；约 50% 时 MAE/RMSE 均优于 HD-TTS-AMP，约 70% 时 RMSE 比 HD-TTS-AMP 高 0.0264（0.37%），故不能表述为全部指标均优。BiTGraph 在统一 Block-T 下显著落后于两个 HD-TTS 系列方法，说明其官方架构在该“时间块缺失、邻域仍可见”的场景中并不占优。

## 可比性与实现

- HD-TTS-AMP 与 MR-MSGF 复用 HD-TTS 官方 METR-LA 数据流程；BiTGraph 直接调用作者未修改的 `BiaTCGNet.Model`（commit `4f1d05bcc20bb3f5084bd1facc995478f84a40ed`），保留作者网络参数、Adam `1e-3`、200 epoch 与最佳验证 checkpoint 选择。
- BiTGraph 作者脚本原本只生成随机 Point missing，因而仅在外部适配了数据接口；统一了 Block-T mask、窗口、划分与评测目标。它在主表中的准确名称是“BiTGraph official architecture under unified Block-T protocol”。
- 三种方法在每个 seed 使用同一 mask SHA-256；METR-LA 数据 SHA-256 为 `64784b76d6fb8ec9bff4b6decafb354da2bb37840468fdccee5044e511277c05`。

## 结果使用边界

此前 BiTGraph 的官方 Point-missing/24 步复现不进入主表，因为其缺失机制和预测协议不同；它只保留为官方复现记录。`bitgraph_unified_blockt/` 的第一批结果存在适配器重复反标准化错误，已保留以供审计，但明确废弃；本报告和主表只使用修正后 `bitgraph_unified_blockt_v2/` 的结果。

## 原始产物

- 主实验汇总：`missing_ts_exp/results/r0908_official_baselines/formal_verification_summary.json`
- BiTGraph 统一协议汇总：`missing_ts_exp/results/r0908_official_baselines/unified_blockt_v2_verification_summary.json`
- BiTGraph 逐 seed JSON/checkpoint：`missing_ts_exp/results/r0908_official_baselines/bitgraph_unified_blockt_v2/`

下一步应在同一 Block-T 主协议上扩充独立 mask seed，并开展 MR-MSGF 的观测图、物理图融合、区域粗图与可靠性门控消融；不建议把 Block-ST 压力测试混入当前主结论。
