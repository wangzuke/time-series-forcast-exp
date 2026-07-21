# 实验计划0721-B：公平基线 / CoFILL / 统一评估实验计划

> 适用机器：机器 B，8 × NVIDIA A800  
> 计划日期：2026-07-21  
> 对应总计划：[`实验计划0721_CoFILL_PGUTS_HD_TTS预测改造.md`](实验计划0721_CoFILL_PGUTS_HD_TTS预测改造.md)  
> 机器定位：负责统一数据 / mask / evaluator，补齐 HD-TTS 与 BiTGraph 公平 baseline，推进 CoFILL 原始复现与 CoFILL-Forecaster 小矩阵，并最终合并机器 A/B 结果。  
> batch size 要求：所有正式训练 `batch_size >= 512`；若模型显式不支持，应通过 dataloader 修复或 gradient accumulation 使有效 batch size 不低于 512，并在结果表中区分 `batch_size` 与 `effective_batch_size`。

---

## 一、本机实验目标

机器 B 的任务不是追求单一新模型，而是保证整个 0721 项目的**公平性、可比性和可汇总性**。本机必须完成四类工作：

1. **统一协议资产**：生成并发布 0721 的 canonical mask bundle、manifest、统一 split 和 evaluator。
2. **公平 baseline**：在 0721 缺失条件下补跑 HD-TTS-AMP 与 BiTGraph，作为机器 A 的 P-GUTS / HD-PGUTS 对照。
3. **CoFILL 扩展线**：跑通 CoFILL 原始 block imputation，并验证 CoFILL-Forecaster 是否值得继续。
4. **结果合并**：合并机器 A/B 的 CSV、检查 `mask_sha256`、统一指标，并为最终 `0721实验报告.md` 提供主表和图。

---

## 二、统一公平协议

机器 B 是统一协议的来源。机器 A 的全部实验必须复用机器 B 生成的 `fair_data/`。

| 项目 | 统一要求 |
| --- | --- |
| 数据源 | `dataset/metr_la/metr_la.h5`、`dataset/pems_bay/pems_bay.h5` |
| 数据形状 | METR-LA: `(34272, 207, 1)`；PEMS-BAY: `(52116, 325, 1)` |
| 空间图 | HD-TTS 使用真实距离图；BiTGraph 使用学习型自适应图；P-GUTS/CoFILL 若使用图，必须从同一传感器距离图或统一转换文件读取 |
| 历史窗口 | `T_in=24` |
| 预测窗口 | `T_out=12, 24`；主比较以 `24→24` 为主 |
| 样本切分 | 先按 `window=24,horizon=T_out,stride=1` 滑窗，再按窗口样本顺序 `70% / 10% / 20%` 切分 |
| 缺失 mask | 统一 `.npy` observed-mask，`1=observed, 0=missing` |
| 缺失类型 | Point、Block-T、Block-ST |
| 缺失率 | Point: 50%、70%；Block-T / Block-ST: 50%、70%、90% |
| batch size | 正式训练不低于 512 |
| 指标 | MAE 为主，RMSE/MSE、MAPE/MRE 为辅；最终统一 evaluator 复算 |
| 结果主键 | `(model, variant, dataset, mask_type, target_missing_rate, T_out, seed)` |

---

## 三、目录与结果格式

统一输出根目录：

```text
missing_ts_exp/results/0721_cofill_pguts_forecasting/
├── fair_data/
│   ├── manifest.csv
│   └── mask_observed_*.npy
├── raw_logs/machine_b/
├── checkpoints/machine_b/
├── predictions/
├── csv/
│   ├── machine_b_baseline_results.csv
│   ├── machine_b_cofill_results.csv
│   ├── machine_b_efficiency_results.csv
│   ├── main_results.csv
│   ├── ablation_results.csv
│   └── efficiency_results.csv
├── figures/
└── notes/
    ├── machine_b_repro_notes.md
    └── merge_notes.md
```

`main_results.csv` 必须兼容机器 A 的字段：

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
seed
batch_size
effective_batch_size
MAE
RMSE_or_MSE
MAPE_or_MRE
epoch_time_sec
train_time_sec
gpu_peak_mb
checkpoint_path
prediction_path
log_path
notes
```

---

## 四、执行总顺序

机器 B 分为 6 个 wave：

```text
Wave B0：统一 mask / split / evaluator
Wave B1：HD-TTS / BiTGraph 0721 公平 baseline
Wave B2：CoFILL 原始 imputation 复现
Wave B3：CoFILL-Forecaster 小矩阵
Wave B4：关键 baseline 补种子 / 补 horizon=12
Wave B5：合并机器 A/B 结果并生成总表和图
```

---

## 五、Wave B0：统一 mask / split / evaluator

### 5.1 统一 mask 生成

生成以下 mask：

| 数据集 | 缺失类型 | 缺失率 |
| --- | --- | --- |
| METR-LA、PEMS-BAY | Point | 50%、70% |
| METR-LA、PEMS-BAY | Block-T | 50%、70%、90% |
| METR-LA、PEMS-BAY | Block-ST | 50%、70%、90% |

共：

```text
2 datasets × (2 point + 3 block-t + 3 block-st) = 16 masks
```

输出：

```text
fair_data/manifest.csv
fair_data/mask_observed_Metr_point_r50_seed2024.npy
...
fair_data/mask_observed_PEMS_blockst_r90_seed2024.npy
```

### 5.2 mask 标定要求

| 缺失类型 | 标定要求 |
| --- | --- |
| Point | 精确到目标缺失率 |
| Block-T | 实际缺失率误差控制在 `±0.5pp` |
| Block-ST | 实际缺失率误差控制在 `±0.5pp`，并记录空间传播半径 / 邻域策略 |

manifest 字段至少包含：

```text
dataset
data_path
data_sha256
n_timesteps
n_nodes
n_channels
mask_type
target_missing_rate
actual_missing_rate
block_length
spatial_propagation
seed
stream_seed
mask_path
mask_sha256
mask_convention
```

### 5.3 统一 split

为每个 `(dataset, T_out)` 生成 split metadata：

```text
fair_data/split_Metr_h12.json
fair_data/split_Metr_h24.json
fair_data/split_PEMS_h12.json
fair_data/split_PEMS_h24.json
```

字段：

```text
total_windows
train_len
val_len
test_len
train_start
val_start
test_start
window
horizon
stride
```

机器 A/B 所有模型必须读取或复现这个 split。

### 5.4 统一 evaluator

新增统一 evaluator，输入各模型保存的：

```text
y_true
y_pred
target_mask
metadata_json
```

输出统一指标：

```text
MAE
RMSE
MSE
MAPE_or_MRE
num_eval_points
```

若某模型暂时不能保存预测文件，允许先解析日志，但最终主表应尽量使用统一 evaluator 复算。

---

## 六、Wave B1：HD-TTS / BiTGraph 0721 公平 baseline

### 6.1 目的

为机器 A 的 P-GUTS-Forecaster / HD-PGUTS 提供公平对照。0721 baseline 不应沿用 0720 的全部数字，因为 0721 新增了：

1. `T_out=12`
2. Block-ST
3. 90% 块缺失
4. Point 只保留 50%、70%

### 6.2 baseline 矩阵

| 模型 | 数据集 | 缺失条件 | T_out | seed |
| --- | --- | --- | --- | --- |
| HD-TTS-AMP | METR-LA、PEMS-BAY | 16 个统一 mask 条件 | 12、24 | 1 |
| BiTGraph | METR-LA、PEMS-BAY | 16 个统一 mask 条件 | 12、24 | 1 |

总量：

```text
2 models × 16 mask conditions × 2 horizons = 64 runs
```

### 6.3 关键补种子

优先对下面组合补 `seed=2,3`：

| 模型 | 数据集 | 缺失类型 | 缺失率 | T_out | seed |
| --- | --- | --- | ---: | ---: | --- |
| HD-TTS-AMP | METR-LA、PEMS-BAY | Block-T、Block-ST | 70%、90% | 24 | 2、3 |
| BiTGraph | METR-LA、PEMS-BAY | Block-T、Block-ST | 70%、90% | 24 | 2、3 |

补种子总量：

```text
2 models × 2 datasets × 2 mask types × 2 rates × 1 horizon × 2 extra seeds = 32 runs
```

### 6.4 并行调度

| GPU | 优先任务 |
| ---: | --- |
| B0 | HD-TTS baseline |
| B1 | HD-TTS baseline |
| B2 | HD-TTS baseline |
| B3 | HD-TTS baseline |
| B4 | BiTGraph baseline |
| B5 | BiTGraph baseline |
| B6 | BiTGraph baseline |
| B7 | BiTGraph baseline / 失败重跑 |

如果 HD-TTS 单任务耗时显著长于 BiTGraph，可动态把 B6/B7 转给 HD-TTS。

### 6.5 验收

1. 所有 baseline 正式结果 `batch_size >= 512`。
2. 日志包含 `data_path`、`mask_path`、`mask_sha256`、`T_in/T_out`、`batch_size`。
3. 同一 `(dataset, mask_type, rate, T_out)` 下，HD-TTS 与 BiTGraph 的 `mask_sha256` 一致。
4. 可被统一 evaluator 复算，或至少能被统一 collector 解析。

---

## 七、Wave B2：CoFILL 原始 imputation 复现

### 7.1 目的

确认 CoFILL 在交通 block missing imputation 上的原始能力和工程成本，为后续 CoFILL-Forecaster 提供依据。

### 7.2 原始复现矩阵

| 数据集 | 缺失类型 | 缺失率 | 任务 |
| --- | --- | ---: | --- |
| METR-LA | Block-T | 70% | imputation |
| METR-LA | Block-ST | 70% | imputation |
| PEMS-BAY | Block-T | 70% | imputation |
| PEMS-BAY | Block-ST | 70% | imputation |

先不跑 90%。只有当 70% 能稳定跑通且时间可接受时，再扩展：

```text
METR-LA / Block-ST / 90%
PEMS-BAY / Block-ST / 90%
```

### 7.3 并行调度

| GPU | 任务 |
| ---: | --- |
| B4 | CoFILL METR-LA Block-T 70% |
| B5 | CoFILL METR-LA Block-ST 70% |
| B6 | CoFILL PEMS-BAY Block-T 70% |
| B7 | CoFILL PEMS-BAY Block-ST 70% |

若 Wave B1 baseline 尚未完成，CoFILL 不得抢占所有 baseline GPU；baseline 优先。

### 7.4 验收

1. 记录 CoFILL commit、环境、完整命令。
2. 记录 diffusion steps、采样数、训练时间、推理时间、显存。
3. 若原仓库不能直接跑通，写清入口梳理和阻塞点，不伪造结果。

---

## 八、Wave B3：CoFILL-Forecaster 小矩阵

### 8.1 改造目标

将 CoFILL 从插补扩散改为条件概率预测：

```text
condition = Encoder(X_hist, M_hist, A)
target = Y_future
diffusion = denoise noisy Y_future conditioned on history
forecast = mean(samples)
uncertainty = std(samples)
```

训练时 condition encoder 只能看历史窗口，不能看到真实未来值。

### 8.2 首批小矩阵

| 数据集 | 缺失类型 | 缺失率 | T_out | seed |
| --- | --- | ---: | ---: | --- |
| METR-LA | Block-T | 70% | 12 | 1 |
| METR-LA | Block-ST | 70% | 12 | 1 |
| PEMS-BAY | Block-ST | 70% | 12 | 1 |

默认设置：

```text
num_samples = 5
diffusion_steps = 20 或 50
batch_size >= 512
```

如果实际显存允许，优先 `diffusion_steps=50`；否则保留 `20`，但必须在结果中标记。

### 8.3 扩展条件

只有当以下条件同时满足，才扩展 CoFILL-Forecaster：

1. MAE 接近或优于机器 A 的 HD-PGUTS / P-GUTS-Forecaster。
2. 推理时间不超过 HD-PGUTS 的 `5×`。
3. 不确定性输出能产生可解释信号，例如高缺失率下预测方差更高。

扩展组合：

```text
PEMS-BAY / Block-T / 70% / T_out=12
METR-LA / Block-ST / 90% / T_out=12
PEMS-BAY / Block-ST / 90% / T_out=12
```

### 8.4 定位

CoFILL-Forecaster 默认是扩展线，不阻塞 P-GUTS / HD-PGUTS 主线。若 CoFILL 成本高且 MAE 无优势，应在报告中定位为“概率预测探索”，不进入主方法竞争。

---

## 九、Wave B4：关键 baseline 补实验

根据机器 A 的阶段性结果，机器 B 负责补齐以下比较：

### 9.1 HD-TTS-AMP 多种子

若 HD-PGUTS full 在某些组合接近或优于 HD-TTS-AMP，需要补：

```text
HD-TTS-AMP
METR-LA / PEMS-BAY
Block-ST 70%、90%
T_out=24
seed=1,2,3
batch_size>=512
```

### 9.2 horizon=12 对照

若机器 A 在 `T_out=12` 下补跑 HD-PGUTS，本机同步补：

```text
HD-TTS-AMP
BiTGraph
METR-LA / PEMS-BAY
Block-ST 70%、90%
T_out=12
seed=1
```

### 9.3 效率代表配置

每个模型至少选两个代表配置记录效率：

```text
METR-LA / Block-ST / 70% / 24→24
PEMS-BAY / Block-ST / 70% / 24→24
```

记录：

```text
epoch_time_sec
train_time_sec
inference_time_sec
gpu_peak_mb
n_params
```

---

## 十、Wave B5：合并机器 A/B 结果

### 10.1 合并输入

机器 B 最终接收机器 A 的：

```text
csv/machine_a_pguts_results.csv
csv/machine_a_hd_pguts_results.csv
csv/machine_a_ablation_results.csv
csv/machine_a_efficiency_results.csv
```

合并成本机：

```text
csv/main_results.csv
csv/ablation_results.csv
csv/efficiency_results.csv
```

### 10.2 合并检查

每条结果必须通过：

1. `mask_sha256` 存在于 `fair_data/manifest.csv`。
2. `batch_size >= 512` 或 `effective_batch_size >= 512`。
3. `T_in=24`。
4. `T_out` 只允许 12 或 24。
5. 同一主键不应重复；重复时保留更完整、可统一 evaluator 复算的一条。
6. 所有主结果应有 `log_path`；checkpoint 可选，但主创新模型必须保留。

### 10.3 图表

机器 B 负责统一生成：

```text
figures/missing_rate_vs_mae_block_st.png
figures/model_comparison_block_t.png
figures/model_comparison_block_st.png
figures/adaptive_scale_weights.png
figures/efficiency_tradeoff.png
```

其中 `adaptive_scale_weights.png` 依赖机器 A 的 HD-PGUTS full 导出权重。

---

## 十一、本机最终交付物

机器 B 完成后交付：

1. `fair_data/manifest.csv` 与全部 `mask_observed_*.npy`
2. `fair_data/split_*.json`
3. 统一 evaluator 脚本与使用说明
4. `csv/machine_b_baseline_results.csv`
5. `csv/machine_b_cofill_results.csv`
6. `csv/main_results.csv`
7. `csv/ablation_results.csv`
8. `csv/efficiency_results.csv`
9. `notes/machine_b_repro_notes.md`
10. `notes/merge_notes.md`
11. 最终图表与 `0721实验报告.md` 的数据基础

---

## 十二、与机器 A 的交接规则

### 12.1 机器 B → 机器 A

机器 B 在 Wave B0 完成后同步：

```text
fair_data/
统一 evaluator
run_id 命名规范
CSV 字段规范
```

机器 A 不得在未同步 `fair_data/manifest.csv` 的情况下启动正式训练。

### 12.2 机器 A → 机器 B

机器 A 每完成一个 wave，同步：

```text
csv/machine_a_*.csv
notes/machine_a_repro_notes.md
关键 raw_logs
HD-PGUTS adaptive fusion 权重统计
```

### 12.3 最终公平判断

最终报告里只有满足以下条件的结果才能进入主表：

1. 使用 canonical HDF5。
2. 使用统一 mask，且 `mask_sha256` 可核验。
3. 使用统一窗口切分。
4. batch size 或 effective batch size 不低于 512。
5. 指标由统一 evaluator 复算，或明确标注为日志解析结果。
