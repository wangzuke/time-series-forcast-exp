# 0910 实验报告：AQI36 自然缺失与 Point 验证

## 1. 目的与协议

本阶段在具有站点地理坐标的 AQI36 空气质量数据上检验跨数据集可迁移性。输入有效掩码始终保留数据固有的自然缺失；在其基础上分别使用不额外注入、额外 Point 25% 和额外 Point 50% 三种条件。物理图由站点经纬度距离构建，区域分配由该物理图固定生成。

自动校验通过 30 条结果：AQI36, 24-to-12, natural validity mask plus fixed Point masks, five seeds, raw valid-future MAE/RMSE。

## 2. 五 seed 结果

| 条件 | 方法 | MAE | RMSE | 相对 MR-MSGF 的配对 MAE 差值 |
|---|---|---:|---:|---:|
| 自然缺失 | HD-TTS-AMP | 30.4839 +/- 0.5849 | 49.3332 +/- 1.4069 | -0.2430 +/- 0.6237 |
| 自然缺失 | MR-MSGF | 30.7269 +/- 0.2392 | 49.8879 +/- 0.9287 | 0.0000 |
| 自然缺失 + Point 25% | HD-TTS-AMP | 31.2034 +/- 0.6894 | 50.3712 +/- 1.0346 | 0.0864 +/- 0.5851 |
| 自然缺失 + Point 25% | MR-MSGF | 31.1169 +/- 0.4161 | 50.3325 +/- 0.7815 | 0.0000 |
| 自然缺失 + Point 50% | HD-TTS-AMP | 31.7359 +/- 0.4956 | 50.4029 +/- 1.2035 | 0.0908 +/- 0.6179 |
| 自然缺失 + Point 50% | MR-MSGF | 31.6451 +/- 0.7093 | 50.5270 +/- 1.3859 | 0.0000 |

## 3. 读法与边界

正的配对 MAE 差值代表 MR-MSGF 较低。AQI36 的量纲、自然缺失率与交通数据不同，因此本表只检验同一 AQI36 条件内的相对比较，不与 METR-LA 的绝对 MAE/RMSE 横向比较。自然缺失条件的五个 seed 固定同一数据掩码，只改变随机初始化；Point 条件同时使用固定的五份额外掩码。

## 4. 产物与下一步

- 自动校验摘要：[stage_c_verification_summary.json](/media/data1/wangzk/time-series-forcast-exp/missing_ts_exp/results/r0910_evidence_closure/stage_c_verification_summary.json)。
- 在跨数据集结果稳定后，下一步补充 horizon 分解、配对置信区间与模型效率统计；GinAR 仅在其官方实现完成统一协议审计后才进入可比表。
