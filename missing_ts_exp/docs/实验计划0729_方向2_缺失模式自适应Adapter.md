# 方向 2 课题：缺失模式自适应的模型无关预测 Adapter

## 1. 课题名称

**PMA: Pattern-adaptive Missingness Adapter for Imputation-free Multivariate Time Series Forecasting**

中文表述：**面向观测缺失的缺失模式自适应、免插补、模型无关多变量时间序列预测适配器**。

核心任务是在输入历史窗口存在缺失值时，不先恢复完整序列，而是在预测模型前加入一个轻量 Adapter，将 `(观测值, mask, 缺失模式统计, 通道相关先验)` 转换成对下游预测有用的规则输入表示。Adapter 可接 `iTransformer`、`PatchTST`、`DLinear` 等 backbone。

## 2. 文献调研结论

### 2.1 直接相关工作

- **MissTSM**：提出 model-agnostic、imputation-free 的 IMTS 处理层。其 Time-Feature Independent embedding 将每个 `(time, feature)` 标量作为 token，再用 Missing Feature-Aware Attention 聚合同一时间步的观测变量。论文强调 MissTSM 在高缺失率、弱周期数据上相比两阶段插补更稳，但也承认 MFAA 本身不学习非线性时间动态，且高维变量下逐值 token 会带来开销。
- **Merlin**：不是新 backbone，而是训练框架。通过完整数据 teacher 的离线蒸馏和多缺失率 view 的对比学习，使一个模型适应 unfixed missing rates。这说明“缺失率/缺失模式变化”本身应该被作为泛化目标，而非只在单一缺失率上训练。
- **CRIB**：系统质疑 `impute-then-predict`。其经验分析指出缺失值无真实监督时，插补可能破坏数据分布和变量相关性；方法上用信息瓶颈和一致性正则学习预测相关表征。该工作支持本课题的基本立场：不要以恢复真实缺失值为目标，而要学习预测有用表征。
- **CoIFNet**：将观测值、mask、时间戳嵌入一起输入，使用 Cross-Timestep Fusion 和 Cross-Variate Fusion，并联合插补与预测 loss。它表明 mask、时间信息、跨变量融合都应进入同一个预测导向框架。
- **ChannelTokenFormer**：面向依赖、异步采样、test-time block missing 三者同时存在的现实设置，用 channel token 和 mask-guided attention 避免插值失真。其未来工作指出超高维通道需要稀疏化/分组通道表示，正好对应 PMA 的 grouped routing。
- **T-PATCHGNN / GraFITi / HyperIMTS / Hi-Patch**：这些 IMTS 工作共同说明，不规则/异步缺失下，硬对齐成完整矩阵会带来长度膨胀或插值噪声；图/超图/patch 表示可以保留原始观测依赖。但它们通常是专用架构，不是轻量、可插拔 Adapter。

### 2.2 研究空白

现有 MissTSM 已解决“免插补 + 可插拔”的第一步，但有三个空白：

1. **缺失模式不自适应**：单 query MFAA 对 random point、continuous segment、variable channel、mixed missing 采用同样聚合策略。
2. **通道依赖缺少结构先验**：完全共享 query 对高维变量容易混杂；静态相关分组又会在缺失模式变化时失配。
3. **缺失率泛化缺少明确评估**：很多实验只在同一缺失机制和缺失率上 train/test，无法证明面对 unfixed missing rates 时的稳定性。

## 3. 明确课题假设

PMA 在 MissTSM 前置层基础上加入三类机制：

- **静态通道相关先验**：用训练集通道相关矩阵聚类，将强相关变量放入同组。
- **观测条件软路由**：学习通道到组的软分配，允许不同缺失率下绕开静态分组失配。
- **mask-gated 双路融合**：硬分组路径和软路由路径并行，门控由全局观测率、组内观测率、组是否全缺失决定，使 Adapter 对缺失模式自适应。

对应当前代码中的主实现为：

```bash
--method misstsm --misstsm_variant grouped_q4_fuseobs_mgate
```

其中：

- `grouped_q4_corr`：静态完整训练集相关分组。
- `grouped_q4_corrobs`：按指定缺失条件下的观测序列相关分组。
- `grouped_q4_soft`：学习软路由。
- `grouped_q4_fuseobs_mgate`：观测条件相关分组 + 软路由 + mask gate，是 PMA 主模型。

## 4. 科学问题与可检验假设

**RQ1：PMA 是否比 MissTSM 原始单 query Adapter 更鲁棒？**

H1：在相同 backbone 下，PMA 在 `continuous_segment`、`variable_channel`、`mixed` 和高缺失率下的 MSE/MAE 低于 `misstsm/full`。

**RQ2：缺失模式自适应是否比固定分组或纯软路由有效？**

H2：`grouped_q4_fuseobs_mgate` 优于 `grouped_q4_corr`、`grouped_q4_corrobs`、`grouped_q4_soft`。

**RQ3：PMA 是否保持 model-agnostic？**

H3：PMA 接 `iTransformer`、`PatchTST`、`DLinear` 时均能带来收益，主收益不是某个 backbone 的偶然调参。

**RQ4：PMA 是否能泛化到未见缺失率？**

H4：训练时混合 `{0.1, 0.3, 0.5}` 缺失率，测试 `0.7` 时，PMA 相比只在单一缺失率训练的 Adapter 退化更小。

## 5. 实验设计

### 5.1 数据集

现阶段使用本仓库已有公开长序列数据：

- 小通道：`ETTh1`、`ETTm1`、`ExchangeRate`
- 中通道：`Weather`
- 高通道：`Electricity`、`Traffic`

主实验优先：`ETTh1`、`Weather`、`Electricity`、`Traffic`。其中 `Electricity` 和 `Traffic` 用来检验高维通道下分组/路由的价值。

### 5.2 缺失机制

输入窗口注入缺失，预测窗口保持完整：

- `random_point`：MCAR 点缺失。
- `continuous_segment`：单变量连续片段缺失，模拟传感器临时中断。
- `variable_channel`：整段变量缺失，模拟变量级不可用。
- `mixed`：点缺失、片段缺失、变量缺失混合。

缺失率：

- 主实验：`0.1, 0.3, 0.5`
- 压力测试：`0.7`

### 5.3 方法与对照

上界：

- `baseline + no_missing`：完整输入上界。

两阶段基线：

- `linear + iTransformer`
- `SAITS + iTransformer`

缺失感知基线：

- `CRIB`
- `CoIFNet`
- `MissTSM/full`

PMA 消融：

- `grouped_q4_corr`
- `grouped_q4_corrobs`
- `grouped_q4_soft`
- `grouped_q4_fuse`
- `grouped_q4_fuseobs_mgate`（主模型）

backbone 泛化：

- `DLinear`
- `PatchTST`
- `iTransformer`

### 5.4 训练设置

- `seq_len = 96`
- `pred_len = 96, 336`
- seeds: `2024, 2025, 2026`
- optimizer: 现有 `AdamW`
- epochs: pilot `3`，main `20`，patience `5`
- 指标：MSE、MAE、参数量、峰值显存、测试耗时、跨 seed 标准差。

### 5.5 结果判定标准

主结论成立需满足：

1. PMA 在至少 3/4 主数据集、至少 2/3 缺失机制中优于 MissTSM/full。
2. 在 `continuous_segment` 或 `variable_channel` 的高缺失率下，PMA 相比 MissTSM/full 的平均 MSE 至少降低 3%。
3. PMA 的参数量和显存不超过 `MissTSM/full` 的 1.5 倍，或准确率收益显著抵消效率成本。
4. 消融显示 mask-gated fusion 对 block/channel missing 有正贡献。

## 6. 实验命令资产

新增脚本：

- `scripts/r0729_direction2_prepare.py`：检查数据集可读性，预计算完整相关分组和观测相关分组缓存。
- `scripts/r0729_direction2_build_cmds.py`：生成方向2的 `smoke/pilot/main/ablation/backbone/stress` 命令清单。

推荐执行顺序：

```bash
cd /data/wangzuke/time-series-forecast-exp/missing_ts_exp

# 1. 检查数据与预计算分组缓存
python scripts/r0729_direction2_prepare.py --datasets ETTh1 Weather Electricity Traffic \
  --missing_types random_point continuous_segment variable_channel mixed \
  --missing_rates 0.1 0.3 0.5 0.7

# 2. 生成烟测命令（ETTh1 小通道，用于验证代码链路）
python scripts/r0729_direction2_build_cmds.py --suite smoke \
  --out results/cmds/r0729_direction2_smoke_cmds.txt \
  --base_out results/0729_direction2_smoke

# 3. 运行烟测
bash scripts/run_experiments.sh results/cmds/r0729_direction2_smoke_cmds.txt 0 1 logs/r0729_direction2_smoke

# 4. pilot
python scripts/r0729_direction2_build_cmds.py --suite pilot \
  --out results/cmds/r0729_direction2_pilot_cmds.txt \
  --base_out results/0729_direction2_pilot

# 5. 主实验
python scripts/r0729_direction2_build_cmds.py --suite main \
  --out results/cmds/r0729_direction2_main_cmds.txt \
  --base_out results/0729_direction2_main
```

汇总：

```bash
python -m src.training.aggregate \
  --results_dir results/0729_direction2_main \
  --out_dir results_aggregated/0729_direction2_main
```

## 7. 预期论文贡献

1. 提出一种介于 MissTSM 与专用图/超图 IMTS 模型之间的轻量 Adapter：保持 model-agnostic，同时具备缺失模式自适应能力。
2. 证明缺失处理不应只输入 mask，而应显式利用缺失模式统计控制跨通道信息融合。
3. 在高维通道与 block/channel missing 下提供比单 query MissTSM 更稳的预测前置表征。

## 8. 风险与备选方案

- 若 `grouped_q4_fuseobs_mgate` 效果不稳定，优先比较 `grouped_q4_fuse` 和 `grouped_q4_soft`，判断问题来自 gate 还是观测相关分组。
- 若高维数据训练过慢，主实验保留 `Weather/Electricity`，将 `Traffic` 放入 stress。
- 若 SAITS 显存或时间不可控，主表保留 `linear+iTransformer` 与 `CoIFNet/CRIB/MissTSM`，SAITS 只作为小数据补充。
- 若需要真正验证 unfixed missing rates，应扩展 dataset 支持 train/test 缺失率分离或 multi-rate training loader；当前第一阶段先用多缺失率独立训练和跨条件评估形成证据。

## 9. 调研来源

- MissTSM: https://arxiv.org/abs/2502.15785
- Merlin: https://arxiv.org/abs/2506.12459
- CRIB: https://arxiv.org/abs/2509.23494
- CoIFNet: https://arxiv.org/abs/2506.13064
- ChannelTokenFormer: https://arxiv.org/abs/2506.08660
- HyperIMTS: https://arxiv.org/abs/2505.17431
- GraFITi: https://arxiv.org/abs/2305.12932
- T-PATCHGNN: https://icml.cc/virtual/2024/poster/33940
- Hi-Patch: https://proceedings.mlr.press/v267/luo25r.html
