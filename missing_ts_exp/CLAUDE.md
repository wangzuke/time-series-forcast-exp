# CLAUDE.md — 观测缺失条件下多变量时间序列预测实验框架

本目录是"观测缺失条件下多变量时间序列预测"的统一实验工程，按 `实验计划2.0.md` 搭建。下面给未来读到此目录的 Claude 一个快速上手的工作指南。

---

## 1. 目录速览

```
missing_ts_exp/
├── 实验计划2.0.md           研究方案（数据集 / 缺失构造 / 方法 / 评价指标 / 实验设计）
├── README.md                使用说明
├── environment.yml          conda 环境
├── requirements.txt         pip 依赖
├── external/                参考仓库（SAITS、CoIFNet、CRIB、MissTSM）—— 仅参考，不直接 import
├── src/
│   ├── data/                数据集 + 缺失注入器 + 时间特征
│   ├── imputation/          简单填补 (simple.py) + SAITS 补值 (saits.py)
│   ├── models/              DLinear / PatchTST / iTransformer / MissTSM / CRIB / CoIFNet
│   ├── training/
│   │   ├── pipelines.py     统一方法管线（TwoStage / SAITS / MissTSM / CRIB / CoIFNet）
│   │   ├── run_forecast.py  单次实验入口
│   │   ├── build_cmds.py    生成命令清单
│   │   ├── aggregate.py     汇总到 CSV / Markdown
│   │   └── plot.py          作图
│   └── utils/constants.py   数据集元数据 + 默认超参（缺失率、种子、序列长度）
├── scripts/
│   ├── run_experiments.sh   GPU round-robin 调度器
│   ├── all_cmds.txt         完整命令（精简方案 ~2300 条）
│   ├── smoke_cmds.txt       8 方法 × ETTh1 烟测
│   ├── baseline_cmds.txt    无缺失上界命令
│   └── <group>_cmds.txt     各分组独立命令清单（按需重生成）
├── results/                 单次实验 JSON 产出
├── results_aggregated/      汇总 CSV / Markdown
├── figures/                 出图
└── logs/                    调度运行日志
```

---

## 2. 当前实验规模（精简方案）

| 组 | 数量 | 备注 |
|---|---:|---|
| baseline (8.1, 无缺失上界) | 108 | 6 ds × 2 H × 3 模型 × 3 seed |
| simple (8.2, 线性插值填补) | 288 | 4 ds × 2 mt × 2 r × 2 H × 3 模型 × 3 seed |
| saits (8.3, SAITS 补值) | 288 | 同上维度，不含填补维 |
| aware (8.4, 缺失感知) | 288 | 同上维度，含 misstsm/crib/coifnet 三方法 |
| extension (10, 扩展) | 960 | 2 ds × 2 mt × 2 r × 2 sl × 4 H × 3 seed × 5 方法 |
| ablation_mask (11.1) | 144 | 4 ds × 2 mt × 2 r × 3 输入形式 × 3 seed |
| ablation_err (11.2) | 24 | 补值-预测误差散点 |
| ablation_gen (11.3) | 48 | 缺失类型泛化 |
| ablation_pred (11.5) | 144 | 预测模型复杂度 |
| **合计** | **2292** | |

精简方案的关键决策（与最早 4932 条命令版本的差异）：
- **缺失率从 4 个减到 2 个**：`MISSING_RATES = [0.1, 0.3]`（去掉 0.5、0.7）
- **简单填补只保留 1 种**：`SIMPLE_IMPUTERS = ["linear"]`（去掉 mean、forward）
- **预测模型仍保留 3 个**：DLinear / PatchTST / iTransformer

这两组常量都集中在 `src/training/build_cmds.py` 顶部和 `src/utils/constants.py`。改动需要同时改：常量 → build_cmds 内部各处显式 rate/imputer 列表 → 实验计划文档 → README → 重新生成命令清单。

---

## 3. 常用工作流

### 单次实验（联调用）

```bash
source /media/data1/wangzk/miniconda3/etc/profile.d/conda.sh
conda activate itransformer
cd /media/data1/wangzk/missing_ts_exp

python -m src.training.run_forecast \
    --dataset ETTh1 --method coifnet --predictor iTransformer \
    --missing_type random_point --missing_rate 0.3 \
    --seq_len 96 --pred_len 96 --seed 2024 \
    --epochs 10 --batch_size 32 --lr 1e-3 \
    --out_dir results
```

输出 JSON 含：测试集 MSE/MAE、补值位置 MSE、训练/推理时间、参数量、峰值显存、训练历史。

### 重新生成命令清单（改完常量后必做）

```bash
# 单组
python -m src.training.build_cmds --group baseline --out scripts/baseline_cmds.txt --epochs 10 --batch_size 32

# 全部
python -m src.training.build_cmds --group all --out scripts/all_cmds.txt --epochs 10 --batch_size 32
```

`--group` 可选：`all | baseline | simple | saits | aware | extension | ablation_mask | ablation_err | ablation_gen | ablation_pred | smoke`。

### 批量调度（多卡 round-robin）

```bash
# 8 卡每卡 1 任务
bash scripts/run_experiments.sh scripts/all_cmds.txt 0,1,2,3,4,5,6,7 1 logs

# 单卡 A800
bash scripts/run_experiments.sh scripts/all_cmds.txt 0 1 logs
```

参数顺序：`CMD_FILE GPU_LIST PER_GPU LOG_DIR`。

### 烟测（管线改动后快速验证）

```bash
python -m src.training.build_cmds --group smoke --out scripts/smoke_cmds.txt
bash scripts/run_experiments.sh scripts/smoke_cmds.txt 0 1 logs
```

覆盖 8 个方法 × ETTh1 × 1 epoch。

### 汇总与作图

```bash
python -m src.training.aggregate --results_dir results --out_dir results_aggregated
python -m src.training.plot --agg_dir results_aggregated --out_dir figures
```

---

## 4. 方法实现要点

| 方法 | 文件 | 关键实现 |
|---|---|---|
| DLinear | `src/models/dlinear.py` | trend + seasonal 双分支线性投影 |
| PatchTST | `src/models/patchtst.py` | patch 切分 + RevIN + Transformer encoder |
| iTransformer | `src/models/itransformer.py` | 变量作为 token，attention 学变量依赖 |
| SAITS | `src/imputation/saits.py` | 两层 DMSA block，自带 MIT 自监督；`--saits_pretrain_epochs N` 控制预训练 |
| MissTSM | `src/models/misstsm.py` | `MissTSMLayer` 用 `key_padding_mask` 让缺失位置不参与跨变量 attention；含 all-missing 时间步保护（让至少一个位置参与，避免 NaN） |
| CRIB | `src/models/crib.py` | 简化版信息瓶颈（隐表征 KL 对 N(0, I)），不实现原文 patching + 双重 ELBO |
| CoIFNet | `src/models/coifnet.py` | TSMixer 风格 backbone，共享层同时输出 (impute, forecast)，损失 = 预测 MAE + α · 补值重建 MAE |

所有方法通过 `src/training/pipelines.py` 统一封装：`forward` 返回 `{forecast, impute, aux_loss}`。新增方法只需写一个 `*Pipeline` 即可，不要污染 `run_forecast.py`。

---

## 5. 数据与缺失注入

- 数据在 `/media/data1/wangzk/dataset/`（路径写死在 `src/utils/constants.py:6` 的 `DATA_ROOT`，迁移服务器时改这里）。
- 主数据集：ETTh1 / Weather / Electricity / Traffic；扩展：ETTm1 / ExchangeRate。
- 切分：ETT 用论文标准（12 / 4 / 4 月）；其余按 7:1:2 时间顺序切。**绝不打乱**。
- 标准化只用训练集统计量。
- 缺失注入器在 `src/data/missing.py`：4 种缺失方式 random_point / continuous_segment / variable_channel / mixed（混合 = 50% 随机点 + 30% 连续片段 + 20% 变量通道）。
- 训练阶段每 epoch 缺失模式会变化：`dataset.set_epoch_seed_offset(ep)`。验证集/测试集种子固定，跨 epoch 不变。

---

## 6. 实验设计约束（写代码 / 改设计时遵守）

1. **只在历史输入注入缺失**，预测目标保持完整。指标反映"历史信息不全条件下的预测能力"，不是标签缺失。
2. **mask 约定**：`mask = 1` 表示观测，`mask = 0` 表示缺失。所有 pipeline、SAITS、MissTSM 都按这个约定。
3. **跨 seed 报告**：主实验 3 seed，消融 1-3 seed 视情况。聚合用 `aggregate.py` 跨 seed 取 mean ± std。
4. **预测器与缺失方法是正交维度**：`simple / saits` 下 `--predictor` 表示 backbone；`misstsm / crib / coifnet` 下 `--predictor` 仅占位（实际是方法内部固定结构）。
5. **early-stopping**：默认 `--patience 3`。如果 epoch 数减小，patience 也要同步缩。

---

## 7. 常见坑

- **PatchTST 在 Traffic 上慢**：862 通道，patch 后 token 数 × batch 巨大。单卡跑此组合可能 5-10×ETTh1 同设置的耗时。如果调度卡死，先单独跑 Traffic 任务摸底。
- **MissTSM all-missing 时间步**：某个时间步所有变量都缺失会导致 attention 输出 NaN。`MissTSMLayer.forward` 已做保护（检测到 all-missing 时让一个位置参与），改它时小心。
- **SAITS 输入 NaN**：上游缺失填 0 前要先看 mask；不要让 NaN 直接进 SAITS。
- **`build_cmds.py` 中常量 vs 函数默认参数**：顶层 `MISSING_RATES` / `SIMPLE_IMPUTERS` 是默认；但 ablation 系列函数里有硬编码的 rate 列表 (`gen_ablation_mask`、`gen_ablation_err`、`gen_ablation_pred`)，改设置时要逐个对齐，不然会出现"主实验跑 10/30、消融却跑 30/50"的不一致。
- **`results/` 文件名靠 `--tag`**：tag 唯一，重复 tag 会覆盖旧结果。`build_cmds.py` 已经按 group/dataset/method/.../seed 生成唯一 tag，但手动调实验时注意。

---

## 8. 修改实验规模的标准流程

如果未来需要再调整（例如缩减 / 扩张实验维度），按这个顺序改：

1. `src/utils/constants.py`：改 `MISSING_RATES` / `SEEDS` / `MAIN_PRED_LENS` 等。
2. `src/training/build_cmds.py`：
   - 顶部常量（`MISSING_RATES`, `SIMPLE_IMPUTERS`, `PRED_MODELS`, `PRED_LENS_*`, `SEEDS`）。
   - 各 ablation 函数中**硬编码**的 rate / imputer 列表。
3. `实验计划2.0.md`：
   - 6.2 缺失率设置表
   - 8.2 简单填补方法表
   - 9 主实验 / 10 扩展实验 / 11.4 缺失率泛化 / 11.5 预测模型复杂度
   - 13.1 主结果表 / 13.2 鲁棒性表（表头/示例行）
   - 14 预期结果（提及具体缺失率的句子）
   - 16 配置摘要表
4. `README.md`：批量实验数量说明、`--impute` 选项。
5. 重生成命令清单：`python -m src.training.build_cmds --group all --out scripts/all_cmds.txt`。
6. Sanity check：`grep -c "missing_rate 0.5" scripts/all_cmds.txt` 等，确认旧设置已经清干净。

---

## 9. 与用户协作的注意事项

- 用户偏好**中文沟通**，代码 / 命令保持英文原样。
- 用户当前的显卡可能被占用：**未经允许不要起训练**。所有"是否要跑一下"的动作先问。
- 用户希望先看到**方案 / 估算**再动手大改。例如"实验数量太多"这类问题，先给削减方案 + 数量估算 + 时间估算的表，等用户拍板后再改代码。
- 单 A800 卡跑完精简方案的全部 2292 条命令大约 110-230 小时（4.5-10 天）。8 卡并行 ~15-30 小时。
- 不要在源码里写"为了某需求加的"这种注释，过两周就成了陈旧上下文。
