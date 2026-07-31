# 0721_1：基线 / CoFILL 实验启动说明

> 对应计划：[`实验计划0721_公平基线_CoFILL_统一评估.md`](实验计划0721_公平基线_CoFILL_统一评估.md)  
> 当前任务线：公平 baseline、CoFILL 接入、统一结果汇总  
> 输出根目录：`missing_ts_exp/results/0721_cofill_pguts_forecasting/`

---

## 一、这条实验线要完成什么

本实验线负责三件事：

1. 核验 0721 已生成的统一缺失 mask bundle。
2. 用完全相同的 canonical HDF5、mask、窗口切分和 batch size，补跑 HD-TTS-AMP 与 BiTGraph baseline。
3. 跑通 CoFILL 原始 imputation 接入：官方 CoFILL 代码已克隆到 `external_repro/CoFILL`，本实验线新增 0721 canonical HDF5 / shared mask runner。

所有正式训练必须满足：

| 项目 | 要求 |
| --- | --- |
| 数据 | `dataset/metr_la/metr_la.h5`、`dataset/pems_bay/pems_bay.h5` |
| mask | `dataset/0721_missing_masks/mask_observed_*.npy` |
| mask 语义 | `1=observed, 0=missing` |
| 历史窗口 | `T_in=24` |
| 预测窗口 | baseline 跑 `T_out=12,24` |
| batch size | `512` |
| seed | baseline full 先跑 `seed=1`；关键块缺失补 `seed=2,3` |
| 指标 | MAE 主指标，MSE/RMSE、MRE/MAPE 辅助 |

---

## 二、已经准备好的脚本

| 脚本 | 作用 |
| --- | --- |
| `missing_ts_exp/scripts/r0721_validate_baseline_cofill_assets.py` | 校验 0721 mask bundle、sha256、shape、实际缺失率、split metadata |
| `missing_ts_exp/scripts/r0721_run_baseline_cofill.sh` | 统一启动入口：validate / smoke / baseline_full / baseline_key_seeds / cofill_imputation / collect / status |
| `missing_ts_exp/scripts/r0721_collect_baseline_cofill.py` | 从 raw logs 汇总 `baseline_results.csv`、`cofill_results.csv`、`main_results.csv` |
| `missing_ts_exp/scripts/r0721_prepare_cofill_data.py` | 为 CoFILL 准备 HDF5 链接、train mean/std、METR-LA 与 PEMS-BAY 距离矩阵 |
| `missing_ts_exp/scripts/r0721_cofill_runner.py` | 真实 CoFILL 0721 runner：读取 canonical HDF5 和统一 observed-mask，并输出 `[cofill_metrics]` |
| `missing_ts_exp/scripts/r0721_run_cofill_0721.sh` | CoFILL shell wrapper，默认使用 `hd-tts` conda 环境运行 CoFILL 适配入口 |

运行脚本已经加了可执行权限，也可以直接用 `bash` 调用。

---

## 三、启动前检查

进入 workspace：

```bash
cd /data/wangzuke/time-series-forecast-exp
```

先做资产校验：

```bash
bash missing_ts_exp/scripts/r0721_run_baseline_cofill.sh validate
```

预期输出：

```text
manifest rows=16
mask npy files=16
split_Metr_h12.json: ...
split_Metr_h24.json: ...
split_PEMS_h12.json: ...
split_PEMS_h24.json: ...
```

校验记录会写到：

```text
missing_ts_exp/results/0721_cofill_pguts_forecasting/notes/baseline_cofill_asset_validation.md
```

任务清单会写到：

```text
missing_ts_exp/results/0721_cofill_pguts_forecasting/csv/baseline_cofill_task_manifest.csv
```

---

## 四、先跑 smoke

正式实验前先跑 smoke：

```bash
bash missing_ts_exp/scripts/r0721_run_baseline_cofill.sh smoke
```

smoke 内容：

| 模型 | 数据集 | 缺失类型 | 缺失率 | T_out | epoch |
| --- | --- | --- | ---: | ---: | ---: |
| BiTGraph | METR-LA | Block-ST | 70% | 12 | 2 |
| HD-TTS-AMP | METR-LA | Block-ST | 70% | 12 | 2 |

smoke 通过后，检查：

```bash
bash missing_ts_exp/scripts/r0721_run_baseline_cofill.sh status
```

重点看：

1. `failed_jobs=0`
2. `csv/baseline_results.csv` 里有 2 行 smoke 结果
3. 两行的 `mask_sha256` 一致且能在 `dataset/0721_missing_masks/manifest.csv` 中找到
4. `batch_size=512`

---

## 五、正式 baseline 主矩阵

smoke 通过后启动主矩阵：

```bash
bash missing_ts_exp/scripts/r0721_run_baseline_cofill.sh baseline_full
```

主矩阵规模：

```text
2 models × 2 datasets × 8 mask conditions × 2 horizons × seed=1 = 64 runs
```

其中 8 个 mask condition 是：

```text
point:   50%, 70%
block_t: 50%, 70%, 90%
block_st:50%, 70%, 90%
```

默认使用 GPU：

```text
1 2 3 4 5 6 7
```

默认不使用 0 号卡，因为当前 0 号卡已被占用。为了更充分利用 A800 80G 显存，启动脚本默认每张可用卡开放 2 个并发 slot，也就是最多同时启动：

```text
7 GPUs × 2 slots = 14 runs
```

这不会改变单个 run 的正式协议，所有 baseline 训练仍保持 `batch_size=512`。

如果只想指定部分 GPU，例如只用 1-3 卡：

```bash
R0721_GPUS="1 2 3" bash missing_ts_exp/scripts/r0721_run_baseline_cofill.sh baseline_full
```

如果观察到显存仍明显空闲，可以把每卡并发从 2 提高到 3：

```bash
R0721_SLOTS_PER_GPU=3 bash missing_ts_exp/scripts/r0721_run_baseline_cofill.sh baseline_full
```

如果出现 OOM 或训练抖动，则降回每卡 1 个 slot：

```bash
R0721_SLOTS_PER_GPU=1 bash missing_ts_exp/scripts/r0721_run_baseline_cofill.sh baseline_full
```

---

## 六、关键块缺失补种子

主矩阵完成后，再启动关键补种子：

```bash
bash missing_ts_exp/scripts/r0721_run_baseline_cofill.sh baseline_key_seeds
```

补种子矩阵：

```text
2 models × 2 datasets × 2 mask types × 2 rates × T_out=24 × seed=2,3 = 32 runs
```

具体为：

```text
mask types: block_t, block_st
rates: 70%, 90%
horizon: 24
seeds: 2, 3
```

这一步用于判断高缺失率块缺失下的结论是否稳定。

---

## 七、CoFILL 原始 imputation 接入

当前 workspace 已接入 CoFILL 官方代码：

```text
external_repro/CoFILL/
```

当前 commit 为总体实验计划指定的：

```text
d461621b213df7d682034a1da99721f2ba65b1ab
```

已新增 0721 适配入口：

```text
missing_ts_exp/scripts/r0721_run_cofill_0721.sh
missing_ts_exp/scripts/r0721_cofill_runner.py
```

该入口会：

1. 调用 `r0721_prepare_cofill_data.py` 准备 CoFILL 兼容数据资产。
2. 读取 `dataset/metr_la/metr_la.h5` 或 `dataset/pems_bay/pems_bay.h5`。
3. 读取 `dataset/0721_missing_masks/mask_observed_*.npy`，语义为 `1=observed,0=missing`。
4. 保留 CoFILL 原模型结构，只改数据入口和 mask 入口。
5. 输出 `[cofill_metrics] {...}`，供 `r0721_collect_baseline_cofill.py` 汇总。

默认使用已有 `hd-tts` 环境运行 CoFILL 适配入口：

```bash
R0721_COFILL_ENV=hd-tts
```

已经完成 dry-run 验证：CoFILL 能加载官方模型、METR-LA canonical HDF5、0721 Block-T 70% mask、距离矩阵，并输出 `status=dry_run`。dry-run 不代表正式训练完成，只代表接入口可执行。

脚本会启动 4 个原始 imputation 任务：

| 数据集 | 缺失类型 | 缺失率 | seed |
| --- | --- | ---: | ---: |
| METR-LA | Block-T | 70% | 1 |
| METR-LA | Block-ST | 70% | 1 |
| PEMS-BAY | Block-T | 70% | 1 |
| PEMS-BAY | Block-ST | 70% | 1 |

正式启动：

```bash
bash missing_ts_exp/scripts/r0721_run_baseline_cofill.sh cofill_imputation
```

可调参数：

```bash
R0721_COFILL_EPOCHS=200
R0721_COFILL_NSAMPLE=5
R0721_COFILL_DIFFUSION_STEPS=50
R0721_COFILL_NUM_WORKERS=4
R0721_COFILL_ENV=hd-tts
```

### CoFILL 结果日志格式

为了让汇总脚本自动识别 CoFILL 指标，runner 训练/测试结束后会打印一行 JSON：

```text
[cofill_metrics] {"status":"finished","MAE":1.23,"MSE":2.34,"MRE":0.12,"train_time_sec":3600,"gpu_peak_mb":24000}
```

可选字段：

```text
epoch_time_sec
train_time_sec
gpu_peak_mb
checkpoint_path
prediction_path
notes
```

---

## 八、结果汇总

任意阶段结束后都可以手动汇总：

```bash
bash missing_ts_exp/scripts/r0721_run_baseline_cofill.sh collect
```

汇总文件：

```text
missing_ts_exp/results/0721_cofill_pguts_forecasting/csv/baseline_results.csv
missing_ts_exp/results/0721_cofill_pguts_forecasting/csv/cofill_results.csv
missing_ts_exp/results/0721_cofill_pguts_forecasting/csv/main_results.csv
```

实时查看状态：

```bash
bash missing_ts_exp/scripts/r0721_run_baseline_cofill.sh status
```

原始日志位置：

```text
missing_ts_exp/results/0721_cofill_pguts_forecasting/raw_logs/baseline_cofill/
```

checkpoint 位置：

```text
missing_ts_exp/results/0721_cofill_pguts_forecasting/checkpoints/baseline_cofill/
```

---

## 九、可调参数

正式实验默认：

```text
epochs=200
HD-TTS train_batches=300
HD-TTS patience=30
batch_size=512
R0721_GPUS="1 2 3 4 5 6 7"
R0721_SLOTS_PER_GPU=2
```

如需调试，可以临时覆盖：

```bash
R0721_EPOCHS=5 \
R0721_HDTTS_TRAIN_BATCHES=20 \
R0721_HDTTS_PATIENCE=3 \
bash missing_ts_exp/scripts/r0721_run_baseline_cofill.sh smoke
```

正式结果不要降低 batch size。若某个模型无法承受 `batch_size=512`，应改 dataloader 或使用 gradient accumulation，并在 CSV 中记录 `effective_batch_size>=512`。

---

## 十、推荐执行顺序

建议按下面顺序执行：

```bash
cd /data/wangzuke/time-series-forecast-exp

bash missing_ts_exp/scripts/r0721_run_baseline_cofill.sh validate
bash missing_ts_exp/scripts/r0721_run_baseline_cofill.sh smoke
bash missing_ts_exp/scripts/r0721_run_baseline_cofill.sh baseline_full
bash missing_ts_exp/scripts/r0721_run_baseline_cofill.sh baseline_key_seeds
bash missing_ts_exp/scripts/r0721_run_baseline_cofill.sh cofill_imputation
bash missing_ts_exp/scripts/r0721_run_baseline_cofill.sh collect
```

---

## 十一、验收标准

baseline 结果进入正式报告前必须满足：

1. `batch_size=512` 或 `effective_batch_size>=512`。
2. `T_in=24`。
3. `T_out` 为 12 或 24。
4. 同一 `(dataset, mask_type, rate)` 下，BiTGraph 与 HD-TTS-AMP 的 `mask_sha256` 完全一致。
5. `mask_sha256` 能在 `dataset/0721_missing_masks/manifest.csv` 中找到。
6. 日志能够被 `r0721_collect_baseline_cofill.py` 汇总。
7. CoFILL 若未跑通，只能记录为 blocked / failed，不得进入主结果表冒充有效指标。
