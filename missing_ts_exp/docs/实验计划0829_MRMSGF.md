# 实验计划0829：MR-MSGF 实现与实验安排

## 1. 目标

本轮实验目标是在 `missing_ts_exp` 现有工程基础上，实现并验证面向观测值缺失场景的多尺度多图融合预测模型 MR-MSGF。

研究问题：

在高缺失率、中低变量维度的多变量时间序列预测任务中，同时利用真实空间关系、观测值驱动的变量时序关系，以及区域化变量关系，是否能比单尺度图建模或纯序列建模获得更稳定的预测效果？

核心假设：

1. 细粒度变量关系适合刻画局部变量依赖，但在高缺失率下容易受到观测不完整影响。
2. 区域级变量关系可以降低缺失噪声和变量级图估计误差，对连续缺失、通道缺失和混合缺失更有价值。
3. 物理图和时序相关图提供互补信息；物理图更稳定，观测相关图更贴合当前数据分布，但需要可靠性约束。
4. 多尺度信息不能简单相加，需要根据缺失率、观测重叠率、聚类质量和图置信度进行自适应融合。

## 2. 已有工程基础

### 2.1 可直接复用的部分

当前工程已经具备统一实验框架：

- 数据加载：`src/data/datasets.py`
- 缺失注入：`src/data/missing.py`
- 数据集配置：`src/utils/constants.py`
- 方法注册：`src/training/pipelines.py`
- 统一训练入口：`src/training/run_forecast.py`
- 命令生成：`src/training/build_cmds.py` 与 `scripts/r0729_direction2_build_cmds.py`
- 结果聚合：`src/training/aggregate.py`

已有方法包括：

- `baseline`
- `simple`
- `saits`
- `misstsm`
- `crib`
- `coifnet`

其中 `misstsm` 已经包含若干与本方向相关的组件：

- observed-only normalization
- mask-aware input
- grouped channel order
- observed correlation grouping
- mask gate
- grouped_q4_fuseobs_mgate 等变体

这些可以作为 MR-MSGF 的工程参考，但 MR-MSGF 应作为新方法独立注册，避免和已有 `misstsm_variant` 继续混杂。

### 2.2 近期实验给出的约束

来自 0715、0718、0720、0721、0723、0729 实验文档的主要结论：

1. HD-TTS 类多尺度结构在连续/块缺失下更稳，说明多尺度时序或区域信息确实可能缓解缺失导致的信息断裂。
2. BiTGraph 类细粒度图传播在块缺失下更容易退化，说明单纯依赖同一时刻变量间传播存在风险。
3. P-GUTS/HD-PGUTS 实验显示 coarse graph 分支有收益，但 adaptive fusion 不稳定，不能直接复用已有门控设计。
4. CoarseGraph 插补实验中，METR-LA 提升明显而 PEMS-BAY 退化，说明区域图不是普适收益，必须加入可靠性判断和数据集适配分析。
5. 0729 的 PMA 实验已经实现了缺失模式自适应和观测相关分组的若干工程接口，可作为 MR-MSGF 的早期原型参考。

因此，本轮实现应强调：

- 独立方法注册
- 可关闭每个模块
- 先构建最小可运行模型
- 逐步加入多图和区域图
- 必须输出诊断量，不能只看最终 MAE/MSE

## 3. 方法定义

暂定方法名：

MR-MSGF：Missing-Robust Multi-Scale Graph Fusion

输入：

- 历史观测值 `x_obs`
- 观测 mask `mask`
- 时间特征 `time_features`，若数据集可用
- 可选物理图 `A_phy`
- 基于观测值估计的时序相关图 `A_temp`
- 区域 assignment 矩阵 `S`

输出：

- 未来窗口预测值 `y_hat`

基本结构：

```text
x_obs, mask
  -> observed-value embedding
  -> temporal encoder
  -> fine graph branch
       - physical graph path
       - temporal correlation graph path
  -> region graph branch
       - node-to-region aggregation
       - region graph propagation
       - region-to-node projection
  -> missingness-aware fusion gate
  -> forecasting head
```

## 4. 工程实现计划

### 4.1 新增文件

建议新增：

- `src/models/mrmsgf.py`
- `src/data/relations.py`
- `scripts/r0829_mrmsgf_prepare.py`
- `scripts/r0829_mrmsgf_build_cmds.py`

建议扩展：

- `src/training/pipelines.py`
- `src/training/run_forecast.py`
- `src/utils/constants.py`，仅在确实需要新增数据集或图路径时修改

### 4.2 `src/data/relations.py`

负责构建和缓存关系信息：

1. 物理图 `A_phy`
   - 对 ETTh1、Weather、Electricity、ExchangeRate 等无明确物理图数据，先使用 identity graph 或 learned graph fallback。
   - 对 METR-LA、PEMS-BAY，如后续接入数据，则从距离矩阵或邻接矩阵构建。

2. 时序相关图 `A_temp`
   - 基于训练集历史序列估计变量相关性。
   - 缺失场景下只使用共同观测位置计算相关性。
   - 输出相关系数矩阵、观测重叠率矩阵、可靠性矩阵。

3. 区域 assignment `S`
   - 第一版沿用 `src/data/grouping.py` 的相关性聚类思路。
   - 支持固定簇数 `num_regions`。
   - 后续可扩展为物理图聚类、时序图聚类、混合图聚类。

4. 缓存机制
   - 缓存目录建议为 `results_cache/mrmsgf_relations`。
   - 缓存 key 包含 dataset、seq_len、missing_type、missing_rate、mask_seed、relation_type、num_regions。

### 4.3 `src/models/mrmsgf.py`

第一版模型需要保持简单，优先保证实验可控。

核心模块：

1. 输入编码
   - value embedding
   - mask embedding
   - optional time feature embedding
   - observed-value normalization，参考 `misstsm` 的 `_revin`

2. 时间编码器
   - 第一版使用轻量 Transformer/TCN/MLP temporal block。
   - 为减少不确定性，可优先采用现有 `misstsm` 中较稳定的时序编码风格。

3. fine graph branch
   - 支持 `A_phy`
   - 支持 `A_temp`
   - 支持二者加权融合
   - 图传播先用简单 GCN/graph smoothing，避免引入过重 GNN。

4. region graph branch
   - `H_region = S^T H_node`
   - `H_region = GCN(A_region, H_region)`
   - `H_node_region = S H_region`

5. fusion gate
   - 输入：节点表示、mask statistics、graph reliability、region id。
   - 输出：fine branch 与 region branch 的融合权重。
   - 初始版本建议让 region residual 权重接近 0，避免训练初期破坏主干预测。

6. prediction head
   - 输出维度与现有框架一致：`[batch, pred_len, num_vars]`。

### 4.4 pipeline 注册

在 `src/training/pipelines.py` 中新增 `MRMSGFPipeline`：

- 读取 dataset 信息和 missing 配置。
- 调用 `relations.py` 获取图和区域 assignment。
- 构建 `MRMSGFModel`。
- 保持与现有训练循环兼容。

在 `src/training/run_forecast.py` 中新增参数：

- `--method mrmsgf`
- `--mrmsgf_num_regions`
- `--mrmsgf_relation_type`
- `--mrmsgf_use_phy`
- `--mrmsgf_use_temp`
- `--mrmsgf_use_region`
- `--mrmsgf_fusion`
- `--mrmsgf_graph_topk`
- `--mrmsgf_relation_cache`

参数默认值应使模型能在无物理图数据集上直接运行。

## 5. 实验阶段安排

### 阶段 A：工程 smoke test

目的：

验证 MR-MSGF 能跑通数据加载、前向传播、反向传播、保存结果。

数据：

- ETTh1
- Weather

设置：

- `seq_len=96`
- `pred_len=96`
- `missing_type=random_point`
- `missing_rate=0.3`
- `seed=2024`
- `max_epochs=1`
- `max_train_batches` 如训练入口支持则启用，否则使用小 batch 和少 epoch。

对比方法：

- `simple`
- `misstsm`
- `mrmsgf`

通过标准：

- 不出现 NaN。
- 输出结果能被 aggregate 脚本读取。
- 参数量、MSE、MAE、训练时间基本记录完整。

### 阶段 B：无物理图版本验证

目的：

先验证仅依靠观测相关图和区域图时，MR-MSGF 是否有收益。

数据：

- ETTh1
- Weather
- Electricity

缺失类型：

- `random_point`
- `continuous_segment`
- `variable_channel`
- `mixed`

缺失率：

- `0.3`
- `0.5`
- `0.7`

预测长度：

- `96`
- `336`

种子：

- `2024`
- `2025`
- `2026`

对比方法：

- `simple`
- `saits`
- `misstsm`
- `crib`
- `coifnet`
- `mrmsgf`

判断重点：

- MR-MSGF 是否在高缺失率下优于 `misstsm`。
- 区域图是否主要在 `continuous_segment`、`variable_channel`、`mixed` 中发挥作用。
- 低缺失率下是否出现明显负迁移。

### 阶段 C：模块消融

目的：

证明收益来自多尺度多图机制，而不是参数量或训练随机性。

消融版本：

- `mrmsgf_no_temp`：去掉观测相关图。
- `mrmsgf_no_region`：去掉区域图。
- `mrmsgf_no_gate`：直接平均融合。
- `mrmsgf_identity_region`：区域 assignment 退化为 identity。
- `mrmsgf_static_gate`：固定 coarse residual 权重。
- `mrmsgf_no_reliability`：不使用观测重叠率和图可靠性。

分析方式：

- 每个设置至少 3 个 seed。
- 重点看 MAE，同时报告 MSE。
- 记录不同缺失率下各分支 gate 均值。
- 记录区域图在不同数据集上的收益和退化。

### 阶段 D：聚类质量与区域数敏感性

目的：

解释 0723 实验中 coarse graph 在不同数据集上表现不一致的问题。

变量：

- `num_regions = 2, 4, 8, 16`
- temporal-correlation clustering
- physical-graph clustering，如有物理图
- random clustering control

诊断指标：

- 簇内相关性均值
- 簇间相关性均值
- 图 sparsity
- observed overlap ratio
- gate 对 region branch 的平均权重
- 每个变量的预测误差变化

预期解释：

如果 MR-MSGF 有效，应当看到高质量聚类下 region branch 权重更高，随机聚类或低可靠性聚类下权重更低。

### 阶段 E：物理图数据集扩展

目的：

验证真实空间信息和时序相关信息是否互补。

候选数据：

- METR-LA
- PEMS-BAY

前提：

当前通用数据 loader 尚未正式支持 METR-LA/PEMS-BAY，需要先确认本地数据格式、邻接矩阵、划分方式和历史实验代码是否可复用。

对比图设置：

- `temp_only`
- `phy_only`
- `phy_temp_fusion`
- `phy_temp_region`
- `phy_temp_region_gate`

重点问题：

- 物理图是否能在高缺失率下提供更稳定关系。
- 时序相关图是否能修正物理距离图中动态相关性不足的问题。
- 区域图是否在传感器数据上稳定收益，还是仍然存在数据集依赖。

### 阶段 F：最终主实验

建议最终表格：

1. 主结果表
   - 数据集 × 缺失类型 × 缺失率 × 方法。

2. 高缺失率鲁棒性表
   - 重点报告 `0.5/0.7/0.9`。

3. 消融表
   - fine graph、region graph、gate、reliability。

4. 聚类敏感性表
   - 不同 `num_regions` 和聚类方式。

5. 诊断图
   - gate 权重随缺失率变化。
   - 图可靠性与性能提升关系。
   - 变量级误差变化热图。

## 6. 推荐命令脚本设计

### 6.1 准备脚本

`scripts/r0829_mrmsgf_prepare.py` 应完成：

- 检查数据集是否存在。
- 预计算 `A_temp`。
- 预计算 `S`。
- 输出关系缓存 manifest。
- 打印每个数据集的变量数、样本数、缺失配置、区域数。

### 6.2 命令生成脚本

`scripts/r0829_mrmsgf_build_cmds.py` 建议支持：

- `--suite smoke`
- `--suite pilot`
- `--suite main`
- `--suite ablation`
- `--suite region`
- `--suite physical`

示例：

```bash
python scripts/r0829_mrmsgf_build_cmds.py --suite smoke --out scripts/r0829_mrmsgf_smoke.txt
bash scripts/run_experiments.sh scripts/r0829_mrmsgf_smoke.txt 0 1 logs/r0829_mrmsgf_smoke
```

## 7. 评估指标

主指标：

- MSE
- MAE

辅助指标：

- missing-position imputation MSE，如训练框架已经记录
- 参数量
- 训练时间
- peak GPU memory
- 不同缺失率下的性能退化比例

诊断指标：

- fine branch gate mean
- region branch gate mean
- physical graph weight
- temporal graph weight
- graph reliability mean
- observed overlap ratio
- cluster quality score

## 8. 结果判定标准

支持假设的结果：

- 在 `0.5/0.7` 高缺失率下，MR-MSGF 相比 `misstsm`、`crib`、`coifnet` 有稳定 MAE 改善。
- 区域图在连续缺失、通道缺失、混合缺失中贡献更明显。
- reliability gate 能减少低质量区域图导致的退化。
- 物理图和时序相关图融合优于单独使用任一图。

不支持假设的结果：

- MR-MSGF 只在少数低缺失场景有效。
- region branch 在多数数据集上造成退化。
- gate 权重无法解释性能变化。
- random clustering 与 correlation clustering 表现接近。

若出现不支持结果，应转向更保守的论文表述：

- 不声称通用多尺度图融合。
- 改为研究“缺失条件下区域关系何时有用”。
- 将贡献重点放在可靠性诊断和失败模式分析上。

## 9. 风险与缓解

风险 1：模型过复杂，训练不稳定。

缓解：

- 第一版只使用简单 graph smoothing。
- region residual 初始权重接近 0。
- 每个模块可关闭。

风险 2：区域图在部分数据集上退化。

缓解：

- 加入 random clustering control。
- 加入 graph reliability gate。
- 报告聚类质量与性能关系。

风险 3：物理图数据集接入成本高。

缓解：

- 第一阶段只做无物理图版本。
- 物理图作为第二阶段扩展，不阻塞主模型实现。

风险 4：收益来自参数量而不是方法设计。

缓解：

- 设置参数量接近的 `misstsm` stronger baseline。
- 报告参数量。
- 做 `no_region`、`no_gate`、`random_region` 消融。

风险 5：缺失 mask 和图估计存在数据泄漏。

缓解：

- 所有图关系只从训练集估计。
- 不使用 test target 计算相关性。
- 缓存 key 明确记录 split、seed、missing config。

## 10. 近期执行顺序

第一步：

实现 `src/data/relations.py`，先支持 observed correlation graph 和 region assignment。

第二步：

实现 `src/models/mrmsgf.py` 的最小版本：

- observed input embedding
- temporal encoder
- temporal graph smoothing
- region graph smoothing
- residual fusion
- prediction head

第三步：

注册 `MRMSGFPipeline` 和 `--method mrmsgf`。

第四步：

写 `r0829_mrmsgf_prepare.py` 与 `r0829_mrmsgf_build_cmds.py`。

第五步：

跑 smoke test：

- ETTh1
- Weather
- random_point
- missing_rate 0.3
- pred_len 96
- seed 2024

第六步：

跑 pilot：

- ETTh1、Weather、Electricity
- random_point、continuous_segment、variable_channel、mixed
- missing_rate 0.3、0.5、0.7
- seed 2024、2025、2026

第七步：

根据 pilot 结果决定是否进入物理图数据集扩展。

## 11. 产出物

本轮应产出：

- MR-MSGF 模型代码
- relation cache 构建代码
- smoke/pilot/main/ablation 命令文件
- smoke 实验报告
- pilot 实验报告
- 消融实验报告
- 最终论文实验表格草稿

## 12. 当前优先级

最高优先级：

1. 保证 MR-MSGF 在现有通用数据集上跑通。
2. 保证每个模块可独立关闭。
3. 保证图和聚类只使用训练集信息。
4. 保证诊断量可保存。

暂不优先：

1. 复杂动态图学习。
2. 大规模交通数据正式复现实验。
3. 过重的 Transformer 主干替换。
4. 端到端可学习聚类。

第一版的目标不是追求结构最复杂，而是建立一个可解释、可消融、能验证研究假设的 MR-MSGF 实验框架。
