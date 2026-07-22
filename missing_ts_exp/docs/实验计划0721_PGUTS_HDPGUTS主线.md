# 实验计划0721：P-GUTS / HD-PGUTS 主线实验计划

> 计划日期：2026-07-21  
> 对应总计划：[`实验计划0721_CoFILL_PGUTS_HD_TTS预测改造.md`](实验计划0721_CoFILL_PGUTS_HD_TTS预测改造.md)  
> 计划定位：承担 P-GUTS-Forecaster 与 HD-PGUTS-Forecaster 主创新线；不负责最终 baseline 全量补跑和 CoFILL 主实验。  
> batch size 要求：所有正式训练 `batch_size >= 512`；如单模型因代码限制无法直接设到 512，必须先改 dataloader / gradient accumulation，不得把正式结果记为 batch_size < 512。

---

## 一、本计划实验目标

本计划只回答两件事：

1. **P-GUTS 能否从 imputation 改造成缺失历史条件下的 forecasting 模型？**
2. **HD-TTS 的时空多尺度思想接入 P-GUTS 后，是否能提升高缺失率 Block-T / Block-ST 预测？**

本计划不承担 CoFILL 扩散预测的主要实验，也不承担 HD-TTS / BiTGraph 全量 baseline；这些由配套的《公平基线 / CoFILL / 统一评估实验计划》负责。0721 前置缺失数据准备已经完成，本计划的所有正式实验必须直接读取 `dataset/0721_missing_masks/` 下的统一 mask bundle，保证后续可以和公平 baseline 合并比较。

---

## 二、统一公平协议

本计划不得自行定义新的数据切分、缺失率或指标口径。所有实验必须遵守以下统一协议。

| 项目 | 统一要求 |
| --- | --- |
| 数据源 | `dataset/metr_la/metr_la.h5`、`dataset/pems_bay/pems_bay.h5` |
| 数据形状 | METR-LA: `(34272, 207, 1)`；PEMS-BAY: `(52116, 325, 1)` |
| 空间图 | 优先使用真实交通距离图；若 P-GUTS 原代码要求其他图格式，必须从同一距离图转换并记录转换脚本 |
| 历史窗口 | `T_in=24` |
| 预测窗口 | `T_out=12, 24`；主分析优先 `24→24` |
| 样本切分 | 先按 `window=24,horizon=T_out,stride=1` 滑窗，再按窗口样本顺序 `70% / 10% / 20%` 切分 |
| 缺失 mask | 读取统一 `dataset/0721_missing_masks/mask_observed_*.npy` |
| mask 语义 | `1=observed, 0=missing` |
| 缺失类型 | Point、Block-T、Block-ST |
| 缺失率 | Point: 50%、70%；Block-T / Block-ST: 50%、70%、90% |
| seed | 主矩阵先 `seed=1`；关键块缺失补 `seed=2,3` |
| batch size | 正式训练不低于 512 |
| 指标 | MAE 为主，RMSE/MSE、MAPE/MRE 为辅；最终以统一 evaluator 复算结果为准 |

启动正式训练前必须确认本地存在：

```text
dataset/0721_missing_masks/manifest.csv
```

并抽查每个 `(dataset, missing_type, rate)` 的 `mask_sha256` 能在该 manifest 中找到。该目录已包含 point 50/70、Block-T 50/70/90、Block-ST 50/70/90，以及 `T_out=12/24` 对应 split metadata。

---

## 三、目录与结果格式

本计划输出统一写入：

```text
missing_ts_exp/results/0721_cofill_pguts_forecasting/
├── raw_logs/pguts_hdpguts/
├── checkpoints/pguts_hdpguts/
├── csv/pguts_results.csv
├── csv/hd_pguts_results.csv
├── csv/hd_pguts_ablation_results.csv
└── notes/pguts_hdpguts_repro_notes.md
```

每条结果至少包含：

```text
run_id
experiment_line
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
pguts_<model>_<dataset>_<mask_type>_r<rate>_h<Tout>_<variant>_s<seed>
```

例如：

```text
pguts_pgutsf_PEMS_blockst_r70_h24_pf3-6_s1
pguts_hdpguts_Metr_blockt_r90_h24_full_s2
```

---

## 四、执行总顺序

本计划按依赖关系分为 5 个阶段：

```text
Phase 0：P-GUTS 原始代码与环境跑通
Phase 1：P-GUTS-Forecaster smoke
Phase 2：P-GUTS-Forecaster 全矩阵
Phase 3：HD-PGUTS 主创新与消融
Phase 4：补种子、补 horizon=12、导出统一 evaluator 所需预测文件
```

如果某个阶段的核心 smoke 未通过，不得直接进入下一阶段的全矩阵。

---

## 五、Phase 0：P-GUTS 原始代码跑通

### 5.1 目标

1. 固定 P-GUTS 官方代码 commit。
2. 跑通 P-GUTS 原始 forecasting。
3. 跑通 P-GUTS 原始 imputation。
4. 找到数据 loader、mask generator、model forward、loss 和 metric 的实际代码位置。

### 5.2 输出

```text
notes/pguts_hdpguts_repro_notes.md
raw_logs/pguts_hdpguts/pguts_original_forecasting_*.log
raw_logs/pguts_hdpguts/pguts_original_imputation_*.log
csv/pguts_original.csv
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

## 六、Phase 1：P-GUTS-Forecaster smoke

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

## 七、Phase 2：P-GUTS-Forecaster 全矩阵

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

1. Block-T 或 Block-ST 70% 下，MAE 不劣于配套公平基线中的 HD-TTS-AMP baseline 超过 5%。
2. 90% 块缺失不出现系统性 NaN 或全 batch 无有效监督。
3. `[3,6]` 与 `[3]` 至少在部分块缺失场景表现出差异，能支撑自适应尺度融合实验。

---

## 八、Phase 3：HD-PGUTS 主创新与消融

### 8.1 模型变体

| 变体 | 目的 |
| --- | --- |
| P-GUTS `[3]` | 单时间尺度基线，复用 Phase 2 结果 |
| P-GUTS `[3,6]` | 多时间尺度基线，复用 Phase 2 结果 |
| HD-PGUTS w/o graph coarsening | 多时间尺度 + 全分辨率图 + 自适应融合；不使用粗化图分支 |
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
2. 在 PEMS-BAY Block-ST 上相比配套公平基线中的 HD-TTS-AMP baseline 有明确 MAE 优势，或在相近 MAE 下显著更快 / 更省显存。
3. adaptive fusion 权重能解释缺失模式：块缺失越严重，粗时间尺度或粗空间尺度权重越高。

---

## 九、Phase 4：补实验与导出

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

## 十、本计划最终交付物

本计划完成后交付：

1. `notes/pguts_hdpguts_repro_notes.md`
2. `csv/pguts_results.csv`
3. `csv/hd_pguts_results.csv`
4. `csv/hd_pguts_ablation_results.csv`
5. `csv/pguts_hdpguts_efficiency_results.csv`
6. `predictions/` 下可复算预测文件
7. `raw_logs/pguts_hdpguts/` 与 `checkpoints/pguts_hdpguts/`

合并进总报告前，本计划结果必须满足：

1. 所有正式结果 `batch_size >= 512`。
2. 所有正式结果 `mask_sha256` 能在统一 manifest 中找到。
3. 所有正式结果使用统一窗口切分。
4. 所有 HD-PGUTS 消融只改变目标模块，不混入数据、mask、batch_size、horizon 的变化。

---

## 十一、已准备代码与启动方式

本计划对应的可执行代码与脚本已经准备在仓库内，采用独立的 0721 P-GUTS / HD-PGUTS 入口，直接读取统一 `dataset/0721_missing_masks/` bundle，不重新生成 mask。

### 11.1 代码入口

```text
missing_ts_exp/src/models/pguts_hdpguts.py
missing_ts_exp/src/training/run_pguts_hdpguts.py
missing_ts_exp/scripts/r0721_build_pguts_cmds.py
missing_ts_exp/scripts/r0721_run_pguts_hdpguts.sh
missing_ts_exp/scripts/r0721_collect_pguts_results.py
```

训练入口实现的核心协议：

```text
X_all = concat(X_hist_observed, zeros_future)
M_all = concat(M_hist, zeros_future_mask)
Y_hat_future = model(X_all, M_all)[:, :T_out]
loss = MAE(Y_hat_future, Y_future)
```

每个 run 会先写独立 JSON：

```text
missing_ts_exp/results/0721_cofill_pguts_forecasting/metrics/pguts_hdpguts/<run_id>.json
```

随后由聚合脚本生成计划要求的 CSV：

```text
missing_ts_exp/results/0721_cofill_pguts_forecasting/csv/pguts_results.csv
missing_ts_exp/results/0721_cofill_pguts_forecasting/csv/hd_pguts_results.csv
missing_ts_exp/results/0721_cofill_pguts_forecasting/csv/hd_pguts_ablation_results.csv
missing_ts_exp/results/0721_cofill_pguts_forecasting/csv/pguts_hdpguts_all_results.csv
```

为控制磁盘占用，自动生成的命令默认只为 smoke、关键块缺失主分析组合、所有 HD-PGUTS 组合和 Phase 4 组合保存完整 `predictions/<run_id>.npz`。Point 条件和非关键 horizon sweep 仍保存 metrics JSON、CSV 行、checkpoint 与日志；如需对某条额外保存预测文件，在单条命令中追加 `--save_predictions`。

### 11.2 环境要求

正式启动前，Python 环境必须至少包含：

```text
torch
pandas
tables 或 pytables
numpy
```

当前 base 环境已有 `torch` 和 `pandas`，但缺少 pandas 读取 `.h5` 所需的 `tables/pytables`。因此真实训练需要先选择或准备一个包含 PyTables 的 PyTorch 环境。若不想激活环境，可用 `R0721_PYTHON_CMD` 指定：

```bash
R0721_PYTHON_CMD='conda run -n <env_name> python' \
  bash missing_ts_exp/scripts/r0721_run_pguts_hdpguts.sh smoke
```

如直接在当前 shell 激活好环境，则不需要设置 `R0721_PYTHON_CMD`。

已准备专用环境文件和预检脚本：

```text
missing_ts_exp/env_r0721_pguts.yml
missing_ts_exp/scripts/r0721_check_pguts_env.py
```

建议优先修当前 base 环境，只补 PyTables：

```bash
conda install -n base -c conda-forge pytables
python missing_ts_exp/scripts/r0721_check_pguts_env.py --require_cuda
```

如果不想改 base，可以创建独立环境：

```bash
conda env create -f missing_ts_exp/env_r0721_pguts.yml
R0721_PYTHON_CMD='conda run -n pguts0721 python' \
  bash missing_ts_exp/scripts/r0721_run_pguts_hdpguts.sh prepare
```

`prepare` 阶段现在会自动调用环境预检，检查 `torch`、`pandas`、`tables`、CUDA、mask manifest，以及两份 `.h5` 能否被读取。若只想先生成命令文件而暂时跳过 HDF5 读取检查，可临时设置：

```bash
R0721_SKIP_H5_ENV_CHECK=1 bash missing_ts_exp/scripts/r0721_run_pguts_hdpguts.sh prepare
```

若环境尚未修复、只想重新生成命令文件，可跳过整个环境预检：

```bash
R0721_SKIP_ENV_CHECK=1 bash missing_ts_exp/scripts/r0721_run_pguts_hdpguts.sh prepare
```

### 11.3 生成命令矩阵

```bash
cd /data/wangzuke/time-series-forecast-exp
python missing_ts_exp/scripts/r0721_build_pguts_cmds.py
```

默认生成：

```text
missing_ts_exp/scripts/r0721_pguts_smoke_cmds.txt       8 runs
missing_ts_exp/scripts/r0721_pguts_phase2_cmds.txt      96 runs
missing_ts_exp/scripts/r0721_hdpguts_phase3_cmds.txt    72 runs
missing_ts_exp/scripts/r0721_pguts_phase4_cmds.txt      24 runs
```

其中 Phase 2 的 96 条包括：

```text
64 条 seed=1 全矩阵
32 条 Block-T / Block-ST 70% / 90% 的 seed=2,3 补种子
```

如需调整训练轮数：

```bash
R0721_SMOKE_EPOCHS=5 R0721_EPOCHS=100 \
  python missing_ts_exp/scripts/r0721_build_pguts_cmds.py
```

### 11.4 启动顺序

先做准备和依赖检查：

```bash
cd /data/wangzuke/time-series-forecast-exp
bash missing_ts_exp/scripts/r0721_run_pguts_hdpguts.sh prepare
```

启动 Phase 1 smoke：

```bash
R0721_GPUS='0 1 2 3 4 5 6 7' \
  bash missing_ts_exp/scripts/r0721_run_pguts_hdpguts.sh smoke
```

smoke 通过后再启动 P-GUTS-Forecaster 全矩阵：

```bash
R0721_GPUS='0 1 2 3 4 5 6 7' \
  bash missing_ts_exp/scripts/r0721_run_pguts_hdpguts.sh phase2
```

本机为 8 张 A800-SXM4-80GB。当前模型在 `batch_size=512` 下预计不会吃满单卡 80GB，因此正式矩阵建议采用“每卡多 run 并发”而不是随意改变 batch size。推荐策略：

```bash
R0721_GPUS='0 1 2 3 4 5 6 7' \
R0721_SLOTS_PER_GPU=2 \
  bash missing_ts_exp/scripts/r0721_run_pguts_hdpguts.sh phase2
```

观察 `nvidia-smi`：若每卡显存和 GPU util 仍明显偏低，可把 `R0721_SLOTS_PER_GPU` 提到 `3` 或 `4`。这样单条 run 仍保持 `batch_size=512` 的公平口径，同时全机最大并发从 8 条提升到 16 / 24 / 32 条。

如后续决定统一放大 batch size，也必须全矩阵一致设置并在 notes 中记录：

```bash
R0721_BATCH_SIZE=1024 \
R0721_GPUS='0 1 2 3 4 5 6 7' \
R0721_SLOTS_PER_GPU=1 \
  bash missing_ts_exp/scripts/r0721_run_pguts_hdpguts.sh phase2
```

默认建议优先增加 `R0721_SLOTS_PER_GPU`，保留 `R0721_BATCH_SIZE=512`。

Phase 2 出现正向信号后启动 HD-PGUTS 主创新与消融：

```bash
R0721_GPUS='0 1 2 3 4 5 6 7' \
R0721_SLOTS_PER_GPU=2 \
  bash missing_ts_exp/scripts/r0721_run_pguts_hdpguts.sh phase3
```

需要补 `horizon=12` 时：

```bash
R0721_GPUS='0 1 2 3 4 5 6 7' \
  bash missing_ts_exp/scripts/r0721_run_pguts_hdpguts.sh phase4
```

也可以执行任意命令文件：

```bash
bash missing_ts_exp/scripts/r0721_run_pguts_hdpguts.sh cmdfile \
  missing_ts_exp/scripts/r0721_pguts_phase2_cmds.txt
```

### 11.5 单条 run 示例

```bash
python -m missing_ts_exp.src.training.run_pguts_hdpguts \
  --dataset Metr \
  --mask_type block_st \
  --missing_rate 0.70 \
  --T_in 24 \
  --T_out 24 \
  --pooling_factors 3,6 \
  --model hd_pguts \
  --variant full \
  --seed 1 \
  --batch_size 512 \
  --epochs 100 \
  --patience 20 \
  --save_predictions
```

若显存不足，可使用 micro-batch 加梯度累积，但正式记录的有效 batch size 必须不低于 512：

```bash
python -m missing_ts_exp.src.training.run_pguts_hdpguts \
  --dataset PEMS \
  --mask_type block_st \
  --missing_rate 0.90 \
  --T_in 24 \
  --T_out 24 \
  --pooling_factors 3,6 \
  --model hd_pguts \
  --variant full \
  --seed 1 \
  --batch_size 128 \
  --grad_accum_steps 4 \
  --epochs 100 \
  --save_predictions
```

### 11.6 输出与聚合

并行 runner 会在所有任务结束后自动执行：

```bash
python missing_ts_exp/scripts/r0721_collect_pguts_results.py
```

如需手动重聚合：

```bash
python missing_ts_exp/scripts/r0721_collect_pguts_results.py \
  --results_root missing_ts_exp/results/0721_cofill_pguts_forecasting
```

日志、checkpoint、预测文件和诊断文件分别写入：

```text
missing_ts_exp/results/0721_cofill_pguts_forecasting/raw_logs/pguts_hdpguts/
missing_ts_exp/results/0721_cofill_pguts_forecasting/checkpoints/pguts_hdpguts/
missing_ts_exp/results/0721_cofill_pguts_forecasting/predictions/
missing_ts_exp/results/0721_cofill_pguts_forecasting/diagnostics/pguts_hdpguts/
```

HD-PGUTS `full` 会额外保存 adaptive fusion 的 scale weights：

```text
diagnostics/pguts_hdpguts/<run_id>_scale_weights.npy
```

### 11.7 no_graph_coarsening 修正与重跑

2026-07-22 检查发现，首版 `no_graph_coarsening` 与 `P-GUTS [3,6]` baseline 在架构上完全相同，导致 24 组 `seed × dataset × mask_type × rate` 的 MAE/RMSE 与 baseline 逐位相等。该批 `no_graph_coarsening` 结果不能作为有效消融。

修正后的变体定义为：

```text
P-GUTS [3,6]          = 无粗图分支 + fixed linear fusion
no_graph_coarsening   = 无粗图分支 + adaptive fusion
no_adaptive_fusion    = 有粗图分支 + fixed linear fusion
full                  = 有粗图分支 + adaptive fusion
```

只需重跑 `no_graph_coarsening` 24 条，不需要重跑 P-GUTS baseline、`no_adaptive_fusion` 或 `full`：

```bash
python missing_ts_exp/scripts/r0721_build_pguts_cmds.py
R0721_GPUS='0 1 2 3 4 5 6 7' \
R0721_SLOTS_PER_GPU=2 \
  bash missing_ts_exp/scripts/r0721_run_pguts_hdpguts.sh cmdfile \
  missing_ts_exp/scripts/r0721_hdpguts_no_graph_coarsening_rerun_cmds.txt
```

重跑会覆盖旧的 `no_graph_coarsening` metrics/checkpoint/prediction，并在新 metrics/CSV 中写入 `architecture_signature` 用于核查消融路径。
