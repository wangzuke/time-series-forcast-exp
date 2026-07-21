# 实验计划0721-A：P-GUTS / HD-PGUTS 主线实验计划

> 适用机器：机器 A，8 × NVIDIA A800  
> 计划日期：2026-07-21  
> 对应总计划：[`实验计划0721_CoFILL_PGUTS_HD_TTS预测改造.md`](实验计划0721_CoFILL_PGUTS_HD_TTS预测改造.md)  
> 机器定位：承担 P-GUTS-Forecaster 与 HD-PGUTS-Forecaster 主创新线；不负责最终 baseline 全量补跑和 CoFILL 主实验。  
> batch size 要求：所有正式训练 `batch_size >= 512`；如单模型因代码限制无法直接设到 512，必须先改 dataloader / gradient accumulation，不得把正式结果记为 batch_size < 512。

---

## 一、本机实验目标

机器 A 只回答两件事：

1. **P-GUTS 能否从 imputation 改造成缺失历史条件下的 forecasting 模型？**
2. **HD-TTS 的时空多尺度思想接入 P-GUTS 后，是否能提升高缺失率 Block-T / Block-ST 预测？**

本机不承担 CoFILL 扩散预测的主要实验，也不承担 HD-TTS / BiTGraph 全量 baseline；这些由机器 B 负责。机器 A 的所有实验必须读取机器 B 生成并同步过来的统一数据 / mask bundle，保证后续可以和机器 B 的 baseline 公平合并。

---

## 二、统一公平协议

本机不得自行定义新的数据切分、缺失率或指标口径。所有实验必须遵守以下统一协议。

| 项目 | 统一要求 |
| --- | --- |
| 数据源 | `dataset/metr_la/metr_la.h5`、`dataset/pems_bay/pems_bay.h5` |
| 数据形状 | METR-LA: `(34272, 207, 1)`；PEMS-BAY: `(52116, 325, 1)` |
| 空间图 | 优先使用真实交通距离图；若 P-GUTS 原代码要求其他图格式，必须从同一距离图转换并记录转换脚本 |
| 历史窗口 | `T_in=24` |
| 预测窗口 | `T_out=12, 24`；主分析优先 `24→24` |
| 样本切分 | 先按 `window=24,horizon=T_out,stride=1` 滑窗，再按窗口样本顺序 `70% / 10% / 20%` 切分 |
| 缺失 mask | 读取统一 `missing_ts_exp/results/0721_cofill_pguts_forecasting/fair_data/mask_observed_*.npy` |
| mask 语义 | `1=observed, 0=missing` |
| 缺失类型 | Point、Block-T、Block-ST |
| 缺失率 | Point: 50%、70%；Block-T / Block-ST: 50%、70%、90% |
| seed | 主矩阵先 `seed=1`；关键块缺失补 `seed=2,3` |
| batch size | 正式训练不低于 512 |
| 指标 | MAE 为主，RMSE/MSE、MAPE/MRE 为辅；最终以统一 evaluator 复算结果为准 |

机器 A 启动前必须确认本地存在：

```text
missing_ts_exp/results/0721_cofill_pguts_forecasting/fair_data/manifest.csv
```

并抽查每个 `(dataset, missing_type, rate)` 的 `mask_sha256` 与机器 B 一致。

---

## 三、目录与结果格式

机器 A 输出统一写入：

```text
missing_ts_exp/results/0721_cofill_pguts_forecasting/
├── raw_logs/machine_a/
├── checkpoints/machine_a/
├── csv/machine_a_pguts_results.csv
├── csv/machine_a_hd_pguts_results.csv
├── csv/machine_a_ablation_results.csv
└── notes/machine_a_repro_notes.md
```

每条结果至少包含：

```text
run_id
machine_id
model
variant
source_code
source_commit
dataset
num_nodes
time_steps
mask_type
target_missing_rate
actual_missing_rate
mask_sha256
T_in
T_out
pooling_factors
graph_scale
adaptive_fusion
seed
batch_size
MAE
RMSE_or_MSE
MAPE_or_MRE
epoch_time_sec
train_time_sec
gpu_peak_mb
checkpoint_path
log_path
notes
```

`run_id` 建议格式：

```text
A_<model>_<dataset>_<mask_type>_r<rate>_h<Tout>_<variant>_s<seed>
```

例如：

```text
A_pgutsf_PEMS_blockst_r70_h24_pf3-6_s1
A_hdpguts_Metr_blockt_r90_h24_full_s2
```

---

## 四、执行总顺序

机器 A 的执行顺序按依赖关系分为 5 个 wave：

```text
Wave A0：P-GUTS 原始代码与环境跑通
Wave A1：P-GUTS-Forecaster smoke
Wave A2：P-GUTS-Forecaster 全矩阵
Wave A3：HD-PGUTS 主创新与消融
Wave A4：补种子、补 horizon=12、导出统一 evaluator 所需预测文件
```

如果某个 wave 的核心 smoke 未通过，不得直接进入下一 wave 的全矩阵。

---

## 五、Wave A0：P-GUTS 原始代码跑通

### 5.1 目标

1. 固定 P-GUTS 官方代码 commit。
2. 跑通 P-GUTS 原始 forecasting。
3. 跑通 P-GUTS 原始 imputation。
4. 找到数据 loader、mask generator、model forward、loss 和 metric 的实际代码位置。

### 5.2 输出

```text
notes/machine_a_repro_notes.md
raw_logs/machine_a/pguts_original_forecasting_*.log
raw_logs/machine_a/pguts_original_imputation_*.log
csv/machine_a_pguts_original.csv
```

### 5.3 并行安排

| GPU | 任务 |
| ---: | --- |
| A0 | P-GUTS 原始 PEMS-BAY forecasting |
| A1 | P-GUTS 原始 METR-LA imputation |
| A2 | P-GUTS 原始 PEMS-BAY imputation |
| A3 | P-GUTS 数据 / mask / model 接口 dry-run |
| A4 ～ A7 | 环境修复、导出配置、后续 smoke 预留 |

### 5.4 验收

1. `source_commit` 写入 notes。
2. 原始命令、配置文件、日志路径完整记录。
3. 若原始 imputation 或 forecasting 无法直接跑通，必须写明阻塞点，允许进入“最小重写训练入口”，但不能伪造原始复现指标。

---

## 六、Wave A1：P-GUTS-Forecaster smoke

### 6.1 改造目标

把 forecasting 构造成 future-value imputation：

```text
X_all = concat(X_hist_observed, zeros_future)
M_all = concat(M_hist, zeros_future_mask)
Y_hat_future = model(X_all, M_all, A)[:, T_in:T_in+T_out]
loss = MAE(Y_hat_future, Y_future)
```

首版允许只用 future loss，不强制加入历史缺失辅助 loss。辅助项建议作为后续 ablation：

```text
L = L_future + λ_hist * L_hist_missing + λ_layer * L_layer_future
λ_hist = 0.2
λ_layer = 0.1
```

### 6.2 smoke 矩阵

| 数据集 | 缺失类型 | 缺失率 | T_out | pooling |
| --- | --- | ---: | ---: | --- |
| METR-LA | Block-T | 70% | 24 | `[3]` |
| METR-LA | Block-ST | 70% | 24 | `[3]` |
| PEMS-BAY | Block-T | 70% | 24 | `[3]` |
| PEMS-BAY | Block-ST | 70% | 24 | `[3]` |
| METR-LA | Block-T | 70% | 24 | `[3,6]` |
| METR-LA | Block-ST | 70% | 24 | `[3,6]` |
| PEMS-BAY | Block-T | 70% | 24 | `[3,6]` |
| PEMS-BAY | Block-ST | 70% | 24 | `[3,6]` |

共 8 条，直接占满 8 卡。

### 6.3 验收

1. 全部能完成至少 2 ～ 5 个 epoch，无 NaN。
2. log 中打印 `data_path`、`mask_path`、`mask_sha256`、`actual_missing_rate`、`T_in/T_out`、`batch_size`。
3. `batch_size >= 512`。
4. 输出预测文件或至少输出可由统一 evaluator 复算的 `y_true/y_pred/mask`。

---

## 七、Wave A2：P-GUTS-Forecaster 全矩阵

### 7.1 seed=1 全矩阵

| 维度 | 取值 |
| --- | --- |
| 数据集 | METR-LA、PEMS-BAY |
| 缺失类型 / 缺失率 | Point: 50%、70%；Block-T: 50%、70%、90%；Block-ST: 50%、70%、90% |
| T_out | 12、24 |
| pooling | `[3]`、`[3,6]` |
| seed | 1 |

总量：

```text
2 datasets × 8 missing conditions × 2 horizons × 2 pooling = 64 runs
```

### 7.2 关键块缺失补种子

在 seed=1 全矩阵启动后，不必等全部完成；优先补以下关键组合的 `seed=2,3`：

| 数据集 | 缺失类型 | 缺失率 | T_out | pooling | seed |
| --- | --- | ---: | ---: | --- | --- |
| METR-LA、PEMS-BAY | Block-T | 70%、90% | 24 | `[3]`、`[3,6]` | 2、3 |
| METR-LA、PEMS-BAY | Block-ST | 70%、90% | 24 | `[3]`、`[3,6]` | 2、3 |

总量：

```text
2 datasets × 2 missing types × 2 rates × 1 horizon × 2 pooling × 2 extra seeds = 32 runs
```

### 7.3 并行调度

每轮最多 8 条并行。建议排序：

1. 先跑 `T_out=24`。
2. 先跑 Block-ST，再跑 Block-T，最后跑 Point。
3. PEMS-BAY 与 METR-LA 混排，避免某一类长任务堆积。
4. 每 8 条作为一个 batch，结束后自动汇总一次中间 CSV。

### 7.4 判断标准

P-GUTS-Forecaster 值得进入 HD-PGUTS 阶段的最低条件：

1. Block-T 或 Block-ST 70% 下，MAE 不劣于机器 B 的 HD-TTS-AMP baseline 超过 5%。
2. 90% 块缺失不出现系统性 NaN 或全 batch 无有效监督。
3. `[3,6]` 与 `[3]` 至少在部分块缺失场景表现出差异，能支撑自适应尺度融合实验。

---

## 八、Wave A3：HD-PGUTS 主创新与消融

### 8.1 模型变体

| 变体 | 目的 |
| --- | --- |
| P-GUTS `[3]` | 单时间尺度基线，复用 Wave A2 结果 |
| P-GUTS `[3,6]` | 多时间尺度基线，复用 Wave A2 结果 |
| HD-PGUTS w/o graph coarsening | 只有时间尺度，无空间粗化 |
| HD-PGUTS w/o adaptive fusion | 有时空尺度，但固定 concat/MLP 融合 |
| HD-PGUTS full | 时间尺度 + 空间尺度 + 自适应融合 |

### 8.2 主实验矩阵

HD-PGUTS 新增 3 个变体，不重复跑 P-GUTS 两个基线。

| 数据集 | 缺失类型 | 缺失率 | T_out | HD 变体 | seed |
| --- | --- | ---: | ---: | --- | --- |
| METR-LA、PEMS-BAY | Block-T | 70%、90% | 24 | 3 个 | 1、2、3 |
| METR-LA、PEMS-BAY | Block-ST | 70%、90% | 24 | 3 个 | 1、2、3 |

新增总量：

```text
2 datasets × 2 missing types × 2 rates × 3 variants × 3 seeds = 72 runs
```

### 8.3 自适应尺度诊断

HD-PGUTS full 必须额外保存：

```text
scale_weights
mask_statistics
branch_outputs_optional
```

至少输出以下聚合：

```text
dataset
mask_type
target_missing_rate
seed
temporal_scale_weight_mean
spatial_scale_weight_mean
coarse_scale_weight_mean
```

用于后续生成：

```text
figures/adaptive_scale_weights.png
```

### 8.4 成功标准

HD-PGUTS 值得作为论文主线的条件：

1. 在 Block-ST 70% 或 90% 下，HD-PGUTS full 稳定优于 P-GUTS `[3,6]`。
2. 在 PEMS-BAY Block-ST 上相比机器 B 的 HD-TTS-AMP baseline 有明确 MAE 优势，或在相近 MAE 下显著更快 / 更省显存。
3. adaptive fusion 权重能解释缺失模式：块缺失越严重，粗时间尺度或粗空间尺度权重越高。

---

## 九、Wave A4：补实验与导出

### 9.1 补 horizon=12

若 `24→24` 主结果有正向信号，补：

| 模型 | 数据集 | 缺失类型 | 缺失率 | T_out | seed |
| --- | --- | --- | ---: | ---: | --- |
| P-GUTS `[3,6]` | METR-LA、PEMS-BAY | Block-ST | 70%、90% | 12 | 1、2、3 |
| HD-PGUTS full | METR-LA、PEMS-BAY | Block-ST | 70%、90% | 12 | 1、2、3 |

### 9.2 统一 evaluator 预测文件

每条正式结果建议保存：

```text
predictions/<run_id>.npz
```

内部字段：

```text
y_true
y_pred
target_mask
history_mask
metadata_json
```

若存储压力太大，至少保存主分析组合：

```text
Block-T 70/90, Block-ST 70/90, T_out=24, seed=1/2/3
```

---

## 十、本机最终交付物

机器 A 完成后交付：

1. `notes/machine_a_repro_notes.md`
2. `csv/machine_a_pguts_results.csv`
3. `csv/machine_a_hd_pguts_results.csv`
4. `csv/machine_a_ablation_results.csv`
5. `csv/machine_a_efficiency_results.csv`
6. `predictions/` 下可复算预测文件
7. `raw_logs/machine_a/` 与 `checkpoints/machine_a/`

合并进总报告前，机器 A 结果必须满足：

1. 所有正式结果 `batch_size >= 512`。
2. 所有正式结果 `mask_sha256` 能在统一 manifest 中找到。
3. 所有正式结果使用统一窗口切分。
4. 所有 HD-PGUTS 消融只改变目标模块，不混入数据、mask、batch_size、horizon 的变化。
