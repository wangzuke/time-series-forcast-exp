# 观测缺失条件下多变量时间序列预测实验框架

按《实验计划 2.0》搭建的统一实验工程。本目录的脚本可独立运行，与
`/media/data1/wangzk/iTransformer`、`/media/data1/wangzk/PatchTST`、
`/media/data1/wangzk/LTSF-Linear` 等已有仓库解耦。

## 1. 目录结构

```
missing_ts_exp/
├── external/                外部参考仓库（仅参考，不直接 import）
│   ├── SAITS/  CoIFNet/  CRIB/  MissTSM/
├── src/
│   ├── data/                数据集 + 缺失注入器 + 时间特征
│   ├── imputation/          简单填补 + SAITS 补值器
│   ├── models/              DLinear / PatchTST / iTransformer / MissTSM / CRIB / CoIFNet
│   ├── training/
│   │   ├── pipelines.py     统一方法管线
│   │   ├── run_forecast.py  单次实验入口
│   │   ├── build_cmds.py    生成命令清单
│   │   ├── aggregate.py     汇总到 CSV / Markdown
│   │   └── plot.py          作图
│   └── utils/constants.py   数据集元数据 + 默认超参
├── scripts/
│   ├── run_experiments.sh   GPU round-robin 调度器
│   ├── smoke_cmds.txt       烟测命令
│   ├── baseline_cmds.txt    无缺失上界命令
│   └── all_cmds.txt         完整实验命令
├── results/                 每次实验产出 JSON
├── results_aggregated/      汇总 CSV / Markdown
├── figures/                 出图
└── logs/                    调度运行日志
```

## 2. 环境

```bash
source /media/data1/wangzk/miniconda3/etc/profile.d/conda.sh
conda activate itransformer    # torch 2.0.0+cu117, numpy 1.23, pandas 1.5, einops 0.8
```

## 3. 单次实验

```bash
python -m src.training.run_forecast \
    --dataset ETTh1 --method coifnet --predictor iTransformer \
    --missing_type random_point --missing_rate 0.3 \
    --seq_len 96 --pred_len 96 --seed 2024 \
    --epochs 10 --batch_size 32 --lr 1e-3 \
    --out_dir results
```

参数：
- `--method`: `baseline | simple | saits | misstsm | crib | coifnet`
- `--predictor`: `DLinear | PatchTST | iTransformer`（在 simple/saits/misstsm 下表示 backbone）
- `--impute`: `linear | none`（仅 simple 方法用；精简方案下只保留线性插值）
- `--missing_type`: `none | random_point | continuous_segment | variable_channel | mixed`

输出 JSON 含：测试集 MSE/MAE、补值位置 MSE、训练/推理时间、参数量、峰值显存、训练历史。

## 4. 批量实验

```bash
# 生成完整命令清单（精简方案：约 1900 个实验）
python -m src.training.build_cmds --group all --out scripts/all_cmds.txt \
    --epochs 10 --batch_size 32

# 8 卡并行（每卡 1 个任务）调度
bash scripts/run_experiments.sh scripts/all_cmds.txt 0,1,2,3,4,5,6,7 1 logs
```

各组（精简方案：缺失率仅 10%、30%；简单填补仅线性插值）：
- `baseline`：8.1 节，三模型 × 6 数据集 × 2 预测长度 × 3 种子 = 108 个
- `simple`：8.2 节，主数据集 4 × 缺失 2 × 率 2 × 预测 2 × 种子 3 × 预测器 3 × 填补 1 = 288 个
- `saits`：8.3 节，主数据集 4 × 缺失 2 × 率 2 × 预测 2 × 种子 3 × 预测器 3 = 288 个
- `aware`：8.4 节，3 方法 × 4 数据集 × 缺失 2 × 率 2 × 预测 2 × 种子 3 = 288 个
- `extension`：10 节，2 ds × 2 mt × 率 2 × 输入 2 × 预测 4 × 种子 3 × 5 方法 = 480 个
- `ablation_*`：11 节，消融实验，合计约 430 个

## 5. 汇总与作图

```bash
python -m src.training.aggregate --results_dir results --out_dir results_aggregated
python -m src.training.plot --agg_dir results_aggregated --out_dir figures
```

产出：
- `results_aggregated/raw.csv`：单实验逐行
- `results_aggregated/main_results.csv`：按 (数据集, 缺失类型, 缺失率, 预测长度, 方法, 预测器, 填补) 跨 seed 聚合的 mean/std
- `figures/curve_*.png`：缺失率—MSE 曲线（每个数据集×缺失类型×预测长度一张）
- `figures/impute_vs_forecast.png`：补值 MSE—预测 MSE 散点图（对应 11.2 消融实验）

## 6. 烟测

```bash
python -m src.training.build_cmds --group smoke --out scripts/smoke_cmds.txt
bash scripts/run_experiments.sh scripts/smoke_cmds.txt 0 1 logs
```

覆盖 8 个方法 ×（ETTh1, 1 epoch），用于在每次改动后快速验证管线。

## 7. 方法实现细节

- **DLinear / PatchTST / iTransformer**：紧凑实现，与官方公开版本逻辑一致。
- **SAITS**：移植自 `external/SAITS/modeling/saits.py`，含 MIT 自监督训练；可单独预训练再接预测器（`--saits_pretrain_epochs N`）。
- **MissTSM**：核心 `MissTSMLayer` 用 `key_padding_mask` 让缺失位置不参与跨变量注意力；后接选定 backbone。
- **CRIB**：紧凑版本，保留输入掩码、信息瓶颈（隐表征 KL 对 N(0,I)）；不实现原文的 patching/双重 ELBO，效果作为对照基线。
- **CoIFNet**：紧凑版 TSMixer 风格 backbone，共享层同时输出 (impute, forecast)，损失=预测 MAE + α·补值重建 MAE。

如需 faithful 复现，可把 `src/models/<m>.py` 替换为外部仓库代码，再调整 `src/training/pipelines.py` 中的封装即可。

## 8. 进度跟踪

- 显卡空闲时执行 `bash scripts/run_experiments.sh scripts/all_cmds.txt 0,1,2,3,4,5,6,7 1 logs`
- 实时观察 `tail -f logs/run_*.log`
- 跑完后 `python -m src.training.aggregate && python -m src.training.plot`
