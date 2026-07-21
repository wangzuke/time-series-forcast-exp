# 实验计划 0721：基于 CoFILL / P-GUTS / HD-TTS 的高维高缺失率块缺失预测改造

> 计划日期：2026-07-21  
> 目标场景：高维多变量/时空图时间序列，在 50% 以上高缺失率，尤其时间块缺失与时空块缺失条件下进行预测。  
> 核心思路：把 CoFILL 与 P-GUTS 在插补任务中的块缺失恢复能力迁移到 forecasting，同时吸收 HD-TTS 的多尺度时空下采样与自适应尺度融合思想。

---

## 一、研究目标

当前课题不是单纯提高插补精度，而是要回答：

> 当历史观测窗口存在高比例点缺失或块缺失时，如何利用空间图结构和多尺度时空信息提升未来预测能力？

本计划围绕三条实验线展开：

| 实验线 | 定位 | 优先级 |
|---|---|---|
| P-GUTS-Forecaster | 将 P-GUTS 的 future-value imputation 思路直接改造成预测模型 | 最高 |
| HD-PGUTS-Forecaster | 将 HD-TTS 的时空层次下采样与自适应尺度融合融入 P-GUTS | 最高 |
| HD-CoFILL-Forecaster | 将 CoFILL 的条件扩散插补改为概率预测模型，并用 HD-TTS 多尺度条件增强 | 中等，作为扩展线 |

第一阶段只使用当前已经下载到本项目 `dataset/` 的数据，避免复现实验被额外下载阻塞。

---

## 二、需要获取的代码与获取方式

所有外部仓库统一放在项目根目录下的 `external_repro/`，不要直接混入 `missing_ts_exp/src`。只有在模型改造阶段，才将必要模块复制或封装到本项目代码中。

### 2.1 官方/原始代码

| 代码 | 用途 | 获取命令 | 当前可访问 HEAD |
|---|---|---|---|
| CoFILL | 条件扩散时空插补；后续改造成概率预测 | `git clone https://github.com/joyHJL/CoFILL.git` | `d461621b213df7d682034a1da99721f2ba65b1ab` |
| P-GUTS | 多尺度 product graph U-Net 插补；优先改造成预测 | `git clone https://github.com/willtryagain/pguts.git` | `f8a26162de2e8d775bfcbb9bc714746fb5f8db30` |
| HD-TTS | 多尺度时空图预测基线；提供 HD 融合模块参考 | `git clone https://github.com/marshka/hdtts.git` | `46a8717db3802e1684e991ba0a7c0bfe43d22535` |
| BiTGraph | 已复现预测基线，用作对照 | `git clone https://github.com/chenxiaodanhit/BiTGraph.git` | `4f1d05bcc20bb3f5084bd1facc995478f84a40ed` |

建议执行：

```bash
mkdir -p external_repro
cd external_repro

git clone https://github.com/joyHJL/CoFILL.git
git clone https://github.com/willtryagain/pguts.git
git clone https://github.com/marshka/hdtts.git
git clone https://github.com/chenxiaodanhit/BiTGraph.git

cd CoFILL && git rev-parse HEAD && cd ..
cd pguts && git rev-parse HEAD && cd ..
cd hdtts && git rev-parse HEAD && cd ..
cd BiTGraph && git rev-parse HEAD && cd ..
```

将实际 commit 写入：

```text
missing_ts_exp/results/0721_cofill_pguts_forecasting/repro_notes.md
```

### 2.2 代码状态说明

| 工作 | 代码状态 | 计划中的处理 |
|---|---|---|
| CoFILL | IJCAI 2025 论文明确给出官方 GitHub | 先跑通原始 imputation，再新建 forecast wrapper |
| P-GUTS | 论文附录给出 `willtryagain/pguts`，含 imputation 与 forecasting 命令 | 先复现 PEMS-BAY forecasting，再扩展高缺失块缺失预测 |
| HD-TTS | ICML 2024 官方代码，已有复现实验经验 | 不直接改其完整代码，优先抽取思想与配置：temporal downsampling、graph coarsening、adaptive decoder |
| BiTGraph | ICLR 2024 官方代码，已作为前序 baseline | 只作为结果对照，不作为本轮主要改造对象 |

注意：P-GUTS 是 KDD 2025 U/M Consortium 论文，不应在论文写作中等同于 KDD Research Track 主会论文；但其方法和代码对本课题很有参考价值。

---

## 三、数据集与本轮限制

### 3.1 首轮必用数据

只使用当前已经下载到 `dataset/` 的交通图数据：

| 数据集 | 本地路径 | 规模 | 用途 |
|---|---|---:|---|
| METR-LA | `dataset/metr_la/metr_la.h5` | `34272 × 207` | 高维交通图预测，块缺失主实验 |
| PEMS-BAY | `dataset/pems_bay/pems_bay.h5` | `52116 × 325` | 高维交通图预测，块缺失主实验 |

这两个数据与 HD-TTS、P-GUTS、CoFILL、BiTGraph 都有交集，适合做统一评估。

### 3.2 可选扩展数据

| 数据集 | 状态 | 使用条件 |
|---|---|---|
| AQI / AQI36 | 本地未标准化准备；P-GUTS/CoFILL 论文均使用 | 只有在官方代码能自动通过 `torch-spatiotemporal` 下载并记录缓存路径时再加入 |
| EngRAD | 本地未下载；HD-TTS 高维多通道数据 | 作为后续扩展，不进入首轮计划 |
| PV-US | 本地未下载；HD-TTS 大规模 1081 节点数据 | 作为扩展压力测试，不进入首轮计划 |

执行 agent 不要把未下载数据写入首轮主结果表。

---

## 四、统一任务定义

### 4.1 输入输出

给定历史窗口：

```text
X_hist ∈ R^{T_in × N}
M_hist ∈ {0,1}^{T_in × N}
A      ∈ R^{N × N}
```

预测未来窗口：

```text
Y_future ∈ R^{T_out × N}
```

首轮统一设置：

| 参数 | 值 |
|---|---|
| `T_in` | 24 |
| `T_out` | 12 和 24 两档 |
| 数据切分 | chronological `60% / 20% / 20%`，若官方代码已有固定切分，先记录并对齐 |
| 指标 | MAE, RMSE/MSE, MAPE 或 MRE |
| 种子 | `1, 2, 3`，资源不足时至少跑 `1` 并明确标注 |

### 4.2 缺失模式

本轮重点是 block missing，同时保留 point missing 作为 sanity check。

| 缺失类型 | 说明 | 缺失率 |
|---|---|---|
| Point | 随机点缺失，独立 mask `(t,n)` | `50%, 70%` |
| Block-T | 每个传感器出现连续时间段缺失 | `50%, 70%, 90%` 实际缺失率 |
| Block-ST | 连续时间段 + 邻域传感器一起缺失 | `50%, 70%, 90%` 实际缺失率 |

实际缺失率必须在训练前统计：

```text
actual_missing = 1 - M_hist.mean()
```

所有结果表同时记录目标缺失率和实际缺失率。块缺失生成函数必须支持通过二分搜索或网格搜索标定到目标实际缺失率，误差控制在 `±0.5pp`。

---

## 五、Phase 0：环境准备与原始代码跑通

### 5.1 建议目录

```text
external_repro/
├── CoFILL/
├── pguts/
├── hdtts/
└── BiTGraph/

missing_ts_exp/results/0721_cofill_pguts_forecasting/
├── raw_logs/
├── csv/
├── checkpoints/
├── figures/
└── repro_notes.md
```

### 5.2 P-GUTS 环境与原始命令

先按仓库 README 安装。如果 README 不完整，则创建独立环境：

```bash
conda create -n pguts python=3.10 -y
conda activate pguts
cd external_repro/pguts
pip install -r requirements.txt
```

若无 `requirements.txt`，执行 agent 需要根据 import error 逐项安装并写入 `repro_notes.md`。P-GUTS 论文附录给出的命令是：

```bash
python -m experiments.run_imputation \
  --config imputation/<dataset>/pguts.yaml \
  --dataset-name <dataset>
```

预测结果复现命令是：

```bash
python -m experiments.run_imputation \
  --config forecasting/pguts.yaml \
  --dataset-name bay_fc
```

P-GUTS 增加缺失率推理：

```bash
python -m experiments.run_inference \
  --config inference.yaml
```

需要先检查 `config/inference.yaml` 中：

```text
exp_name
dataset name
failure probability / mask setting
checkpoint path
```

### 5.3 CoFILL 环境与原始命令

```bash
conda create -n cofill python=3.10 -y
conda activate cofill
cd external_repro/CoFILL
pip install -r requirements.txt
```

如果仓库没有统一训练脚本说明，执行 agent 需要：

1. 阅读 `README.md`、`configs/`、`main*.py`、`train*.py`。
2. 找到 METR-LA / PEMS-BAY 的 imputation 入口。
3. 先跑论文原始 block missing imputation。
4. 记录完整命令、配置文件、输出指标。

CoFILL 论文中的交通缺失设定：

```text
Point: 25% random masking
Block: 5% random masking + continuous missing segments
Block segment length: 1 to 4 hours per sensor
Failure probability: 0.15%
Datasets: METR-LA, PEMS-BAY
```

### 5.4 HD-TTS 参考代码

HD-TTS 不一定需要直接 fork 改造，但必须克隆并阅读以下模块：

```text
external_repro/hdtts/
├── lib/
├── models/
├── experiments/
└── config/
```

重点找：

| 模块 | 需要抽取的思想 |
|---|---|
| temporal downsampling | 多时间尺度扩大感受野 |
| k-MIS graph pooling / graph coarsening | 多空间尺度补偿邻域块缺失 |
| adaptive decoder | 根据输入动态选择时空尺度 |
| missing mask generator | Block-T / Block-ST 标定方式 |

---

## 六、Phase 1：原始能力复现

### 6.1 P-GUTS 原始 imputation 与 forecasting

目标：

1. 跑通 P-GUTS 在 PEMS-BAY / METR-LA 的 block imputation。
2. 跑通 P-GUTS 论文 Table 5 的 PEMS-BAY 1-hour forecasting。
3. 确认 P-GUTS 原生是否已支持把未来窗口作为 missing target。

输出：

```text
csv/pguts_original_imputation.csv
csv/pguts_original_forecasting.csv
raw_logs/pguts_original_*.log
```

必须记录：

```text
dataset, nodes, T_in, T_out, pooling factors, mask type,
target missing rate, actual missing rate, MAE, MSE/RMSE, MRE/MAPE,
epoch time, total time, GPU memory
```

### 6.2 CoFILL 原始 block imputation

目标：

1. 跑通 CoFILL 在 METR-LA / PEMS-BAY 的 block imputation。
2. 复核 CoFILL 是否能在交通 block setting 下优于 PriSTI / CSDI / GRIN。
3. 记录 diffusion steps、采样时间、显存。

输出：

```text
csv/cofill_original_imputation.csv
raw_logs/cofill_original_*.log
```

若 CoFILL 原仓库没有提供完整可运行脚本，则本阶段目标改为：

```text
完成代码入口梳理 + 最小训练脚本定位 + 环境依赖修复记录
```

不要在没有跑通的情况下伪造复现结果。

### 6.3 HD-TTS / BiTGraph 对照

直接复用已有 0715 结果；若另一个 agent 需要重新跑，参考：

```text
missing_ts_exp/docs/实验计划0715.md
missing_ts_exp/docs/0715复现实验报告.md
```

---

## 七、Phase 2：P-GUTS-Forecaster 改造

### 7.1 改造目标

将预测改写为未来窗口插补：

```text
X_all = concat(X_hist_observed, zeros_future)
M_all = concat(M_hist, zeros_future_mask)
```

模型输出：

```text
Y_hat_all = P-GUTS(X_all, M_all, A)
Y_hat_future = Y_hat_all[:, T_in:T_in+T_out, :]
```

训练损失：

```text
L = MAE(Y_hat_future, Y_future)
  + λ_hist * MAE(Y_hat_hist_missing, X_hist_true on artificial missing positions)
  + λ_layer * layerwise_future_loss
```

建议初值：

```text
λ_hist = 0.2
λ_layer = 0.1
```

### 7.2 需要新增/修改的代码

在 P-GUTS 原仓库中先做最小侵入式改造：

```text
external_repro/pguts/
├── experiments/run_forecasting_missing.py      # 新增
├── config/forecasting/missing_forecast.yaml    # 新增
├── code/data/forecasting_window.py             # 新增或复用现有 data loader
├── code/masks/block_missing.py                 # 新增统一 mask 生成
└── code/metrics/forecasting.py                 # 新增 MAE/RMSE/MAPE
```

如果 P-GUTS 原仓库结构不同，以实际文件名为准，但必须保持这些逻辑边界：

| 模块 | 职责 |
|---|---|
| `forecasting_window.py` | 从连续时序构造 `X_hist, M_hist, Y_future` |
| `block_missing.py` | 生成 Point / Block-T / Block-ST mask，并统计实际缺失率 |
| `run_forecasting_missing.py` | 训练、验证、测试预测任务 |
| `forecasting.py` | 指标计算，统一输出 CSV |

### 7.3 实验矩阵

| 数据集 | 缺失类型 | 实际缺失率 | `T_in→T_out` | pooling factor |
|---|---|---:|---|---|
| METR-LA | Point | 50%, 70% | `24→12`, `24→24` | `[3]`, `[3,6]` |
| METR-LA | Block-T | 50%, 70%, 90% | `24→12`, `24→24` | `[3]`, `[3,6]` |
| METR-LA | Block-ST | 50%, 70%, 90% | `24→12`, `24→24` | `[3]`, `[3,6]` |
| PEMS-BAY | Point | 50%, 70% | `24→12`, `24→24` | `[3]`, `[3,6]` |
| PEMS-BAY | Block-T | 50%, 70%, 90% | `24→12`, `24→24` | `[3]`, `[3,6]` |
| PEMS-BAY | Block-ST | 50%, 70%, 90% | `24→12`, `24→24` | `[3]`, `[3,6]` |

先跑单 seed；若趋势明确，再补 `seed=1,2,3`。

### 7.4 判断标准

P-GUTS-Forecaster 值得继续的条件：

1. 在 Block-T 或 Block-ST `70%` 下，MAE 不劣于 HD-TTS 超过 `5%`。
2. 在 `90%` 块缺失下不出现训练崩溃或 NaN。
3. `[3,6]` 与 `[3]` 的差异能支持后续“自适应尺度选择”设计。

---

## 八、Phase 3：HD-PGUTS-Forecaster 主创新线

### 8.1 核心假设

P-GUTS 的多 temporal pooling 能处理时间块缺失，但它原始实验主要使用 temporal pooling；HD-TTS 的空间 graph coarsening 可以补上空间尺度建模。两者结合后，应更适合 Block-ST。

### 8.2 结构设计

将 P-GUTS 的 branch 从“只有不同时间池化因子”扩展为“时间尺度 × 空间尺度”：

```text
Branch 1: temporal factor 3  + graph scale A0
Branch 2: temporal factor 6  + graph scale A1
Branch 3: temporal factor 12 + graph scale A2
```

其中：

```text
A0 = original graph
A1 = 1-level coarsened graph from HD-TTS k-MIS pooling
A2 = 2-level coarsened graph from HD-TTS k-MIS pooling
```

再加入 adaptive scale fusion：

```text
α_{k,l} = softmax(MLP([mask statistics, node representation, scale representation]))
H_fused = Σ_{k,l} α_{k,l} H_{k,l}
Y_hat_future = ForecastHead(H_fused)
```

### 8.3 需要新增/修改的代码

建议不要直接大改 P-GUTS 原始 branch，而是新增 HD 分支，便于 ablation：

```text
external_repro/pguts/
├── code/layers/graph_coarsening.py       # 从 HD-TTS 迁移或复写 k-MIS pooling
├── code/layers/hd_pguts_branch.py        # 新增时空双尺度 branch
├── code/layers/adaptive_scale_fusion.py  # 新增动态尺度融合
├── code/models/hd_pguts_forecaster.py    # 新模型
└── config/forecasting/hd_pguts.yaml      # 新配置
```

如果从 HD-TTS 复制代码，必须在文件头部注明来源：

```text
Adapted from https://github.com/marshka/hdtts, commit 46a8717db3802e1684e991ba0a7c0bfe43d22535.
```

### 8.4 消融实验

| 模型 | 目的 |
---|---|
| P-GUTS-Forecaster `[3]` | 单尺度时间池化 |
| P-GUTS-Forecaster `[3,6]` | 多尺度时间池化 |
| HD-PGUTS w/o adaptive fusion | 固定 concat/MLP 融合 |
| HD-PGUTS w/o graph coarsening | 只有时间尺度，无空间尺度 |
| HD-PGUTS full | 时间尺度 + 空间尺度 + 自适应融合 |

重点看：

```text
Block-T 70/90%
Block-ST 70/90%
PEMS-BAY 高维 325 节点
```

### 8.5 成功标准

HD-PGUTS 值得作为论文主线的条件：

1. 在 Block-ST `70%` 或 `90%` 下稳定优于 P-GUTS-Forecaster。
2. 在 PEMS-BAY 上相比 HD-TTS 有明确优势，或在相近 MAE 下显著更快/更省显存。
3. adaptive fusion 权重能解释缺失模式：块缺失越长，粗时间尺度/粗空间尺度权重越高。

---

## 九、Phase 4：HD-CoFILL-Forecaster 扩展线

### 9.1 改造目标

将 CoFILL 从 imputation diffusion 改为 conditional probabilistic forecasting：

```text
condition = Encoder(X_hist, M_hist, A)
target    = Y_future
diffusion = denoise noisy Y_future conditioned on history
```

训练时只能让 condition encoder 看历史窗口，不能看到真实未来值。

### 9.2 最小版本 CoFILL-Forecaster

新增：

```text
external_repro/CoFILL/
├── forecasting/run_forecasting.py
├── forecasting/dataset.py
├── forecasting/masks.py
├── forecasting/metrics.py
└── forecasting/config_metr_la.yaml
```

保留 CoFILL 的：

| 原模块 | 保留方式 |
---|---|
| TCN + GCN temporal/spatial condition | 改为只编码历史窗口 |
| DCT frequency branch | 对历史窗口做 DCT，作为周期性条件 |
| cross-attention fusion | 保留 |
| noise estimation module | 输出未来窗口噪声 |
| reverse diffusion | 生成未来预测样本 |

预测输出：

```text
mean forecast = average(samples)
uncertainty  = std(samples)
```

首轮采样数：

```text
num_samples = 5
diffusion_steps = 20 or 50
```

### 9.3 HD-CoFILL 增强版

将 CoFILL 原 condition encoder 替换为：

```text
HD temporal encoder from HD-TTS
HD spatial graph encoder from HD-TTS
DCT frequency encoder from CoFILL
adaptive fusion
```

由于扩散模型成本高，本阶段只跑：

| 数据集 | 缺失类型 | 缺失率 | `T_in→T_out` |
|---|---|---:|---|
| METR-LA | Block-T | 70% | `24→12` |
| METR-LA | Block-ST | 70% | `24→12` |
| PEMS-BAY | Block-ST | 70% | `24→12` |

### 9.4 成功标准

HD-CoFILL 值得继续的条件：

1. MAE 接近或优于 HD-PGUTS，但能额外提供可信不确定性。
2. 多样本预测的 CRPS 或 pinball loss 有优势。
3. 推理时间不超过 HD-PGUTS 的 `5×`；否则仅作为分析模型，不作为主方法。

---

## 十、统一结果表

所有阶段统一输出：

```text
missing_ts_exp/results/0721_cofill_pguts_forecasting/csv/main_results.csv
```

字段：

```text
run_id
model
source_code
source_commit
dataset
num_nodes
time_steps
mask_type
target_missing_rate
actual_missing_rate
block_length
spatial_propagation
T_in
T_out
seed
MAE
RMSE_or_MSE
MAPE_or_MRE
CRPS_optional
epoch_time_sec
train_time_sec
gpu_peak_mb
notes
```

另外输出：

```text
csv/ablation_results.csv
csv/efficiency_results.csv
figures/missing_rate_vs_mae_block_st.png
figures/model_comparison_block_t.png
figures/model_comparison_block_st.png
figures/adaptive_scale_weights.png
```

---

## 十一、建议执行顺序

### Week 1：代码跑通与原始复现

1. 克隆并记录 CoFILL / P-GUTS / HD-TTS / BiTGraph commit。
2. 跑通 P-GUTS 原始 PEMS-BAY forecasting。
3. 跑通 P-GUTS METR-LA / PEMS-BAY block imputation。
4. 跑通 CoFILL METR-LA / PEMS-BAY block imputation。
5. 整理 `repro_notes.md`。

### Week 2：P-GUTS-Forecaster

1. 新增 forecasting window dataloader。
2. 新增 point / block-t / block-st mask generator。
3. 将 future window 作为 missing target。
4. 跑 METR-LA 单 seed 全矩阵。
5. 跑 PEMS-BAY 重点配置。

### Week 3：HD-PGUTS

1. 从 HD-TTS 迁移 graph coarsening。
2. 新增 HD branch 与 adaptive fusion。
3. 跑 Block-T / Block-ST 70%、90%。
4. 做消融。
5. 可视化 scale weights。

### Week 4：HD-CoFILL

1. 实现 CoFILL-Forecaster 最小版本。
2. 跑 METR-LA Block-ST 70%。
3. 若成本可控，加入 PEMS-BAY。
4. 对比 deterministic forecast 与 probabilistic forecast。

---

## 十二、风险与处理

| 风险 | 处理 |
---|---|
| P-GUTS 代码环境不完整 | 先按论文命令定位入口；缺依赖逐项记录；必要时只复用模型层重写训练入口 |
| CoFILL 扩散推理太慢 | 降低 diffusion steps，先用 DDIM/少步采样；只跑 24→12 |
| 90% block 缺失训练不稳定 | 先确认 mask 标定；加入 gradient clipping；跳过全缺失 batch |
| P-GUTS 多 pooling 在极高缺失下退化 | 引入 adaptive fusion；同时保留 `[3]` 单尺度强基线 |
| HD-TTS graph coarsening 在交通图上不稳定 | 对比 original graph / undirected graph / learned adjacency 三种方式 |
| 各代码库指标定义不一致 | 在本项目统一实现 metric，并保存 raw prediction 用于复算 |

---

## 十三、最终交付物

执行 agent 完成后必须交付：

1. `repro_notes.md`：代码版本、环境、数据路径、所有运行命令。
2. `main_results.csv`：统一实验结果。
3. `ablation_results.csv`：P-GUTS / HD-PGUTS 消融。
4. `efficiency_results.csv`：训练时间、显存、推理时间。
5. `0721实验报告.md`：结论性报告。
6. 可复现实验脚本或命令列表。

报告必须回答：

1. P-GUTS 是否能直接作为高块缺失预测模型？
2. HD-TTS 的时空下采样融入 P-GUTS 后是否提升 Block-ST？
3. CoFILL 的扩散式恢复是否能转化为预测收益？
4. 在 METR-LA / PEMS-BAY 上，哪条路线最值得发展成论文方法？
