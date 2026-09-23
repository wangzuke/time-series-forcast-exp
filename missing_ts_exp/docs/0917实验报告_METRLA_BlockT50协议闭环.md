# 0917 实验报告：METR-LA Block-T 50% 协议闭环重跑

## 协议闭环

三种变体均由同一冻结代码、相同的窗口元数据 input map、METR-LA Block-T 50% mask、24 步历史至 12 步预测、五个配对 seed、AdamW 和相同最佳 validation-MAE checkpoint 规则训练。HD-TTS-AMP 是既有主基线，不属于本轮 15 个同版重跑任务。

冻结提交：`1476259e0a0e5b61a7a4431a3ae71651ad167a6b`；受控模型/runner 指纹：`91d471f72d4ca936258609cffcbf0d4cf2c17deef099c9cc184d4c16be1acf43`。

## 主结果

| Variant | Parameters | MAE | RMSE | Delta MAE vs Full | 95% CI | paired p (Holm) |
|---|---:|---:|---:|---:|---:|---:|
| Full-observed | 407,447 | 3.3294 +/- 0.0180 | 6.8157 +/- 0.0601 | -- | -- | -- |
| Feature-only | 378,415 | 3.3607 +/- 0.0139 | 6.9029 +/- 0.0256 | 0.0313 | [-0.0008, 0.0635] | 0.1076 |
| Full-permuted-Q | 407,447 | 3.3289 +/- 0.0117 | 6.8703 +/- 0.0429 | -0.0006 | [-0.0208, 0.0197] | 0.9427 |

## 配对统计与审计

正 Delta 表示 Full-observed 的 MAE 更低。所有置信区间是五个配对 seed 差值的双侧 95% t 区间；两个 MAE 主比较以 Holm 校正。

- **Feature-only**：MAE 差值 = [-0.003459930419921875, 0.05800747871398926, 0.050106048583984375, 0.039138078689575195, 0.012766599655151367]；RMSE 差值 = [0.056221010809256455, 0.11034152958057586, 0.11072800501017532, 0.13178621300050875, 0.026771236549940625]；未校正 MAE p=0.05382。
- **Full-permuted-Q**：MAE 差值 = [-0.0036172866821289062, 0.010810375213623047, 0.0006530284881591797, 0.01565265655517578, -0.0262908935546875]；RMSE 差值 = [0.06220784852659822, 0.0538977870766697, 0.1196249406747274, 0.07884108585687688, -0.04151056275083853]；未校正 MAE p=0.9427。
- P0：重复置换一致=True；对称=True；对角线严格恢复=True；非对角多重集合保持=True；三变体共享元数据=True。
- Full-observed 与 Full-permuted-Q 的逐字段 manifest 配置比对通过；两者只允许 q mode、置换审计、训练轨迹/checkpoint 与指标不同。

## 论文解释边界

若 Full 显著优于 Feature-only，可将该结果写为完整架构的收益不能由直接注入缺失统计特征解释。若 Full 与 Permuted-Q 的比较不显著，则必须明确：当前控制不足以证明节点对身份是主要收益来源；不因该结果再次改变 seed、预算或置换策略。

机器可读验证：[verification_summary.json](/media/data1/wangzk/time-series-forcast-exp/missing_ts_exp/results/r0917_protocol_closed/verification_summary.json)；逐任务 manifest：[manifest_rows.csv](/media/data1/wangzk/time-series-forcast-exp/missing_ts_exp/results/r0917_protocol_closed/manifest_rows.csv)；P0 审计：[p0_audit.json](/media/data1/wangzk/time-series-forcast-exp/missing_ts_exp/results/r0917_protocol_closed/p0_audit.json)。
