# DRR-MSGF 小范围机制验证实验规划

## 0. 目的、边界与结论标准

本计划服务于三个尚未被现有实验充分回答的问题：

1. 双关系细图（物理图和窗口观测图）的融合是否带来可重复的预测收益；
2. 区域成员归属应保持静态，还是应随窗口观测动态变化；
3. 为什么采用“先在节点边层融合，再由融合图诱导区域图”，而不是先分别粗化两图后再融合。

本计划**不**试图证明物理图、观测图或共同观测量是“真实因果关系”，也不把共同观测 `q_ij` 解释为校准后的可靠性概率。可验证的结论仅限于：在预先固定的数据、缺失机制、HD-TTS 主干和训练预算下，哪种信息组织方式产生更低的预测误差。

为避免重新落入“单一条件堆很多变体”的问题，实验按闸门推进：

- 所有核心阶段只使用 **METR-LA、24 -> 12、Block-T 50%**；不扫缺失率。
- 每个新机制先做 1 个预先指定 seed 的正式预算试运行；它只用于判断是否值得扩展，不能用于显著性结论。
- 只有试运行达到对应闸门，才启动确认；一旦启动，确认变体和 seed 不得再根据试运行结果增删。
- 确认阶段最多使用 3 个 seed：`2025, 2026, 2027`；试运行固定为 `2024`。
- 所有同一阶段的变体共享同一 mask、初始化 seed、训练预算、验证集 checkpoint 规则和评估器。

这是一份计划，不启动训练，也不修改现有模型代码。实施前应将本文件中的“拟新增”项目实现为可审计脚本，并通过 P0/P1 烟测。

---

## 1. 当前可复用资产与历史证据

### 1.1 当前正式运行链路

DRR-MSGF 的当前可运行实现不是从零开始。下列文件构成应继续复用的主链路：

| 作用 | 当前文件 | 本计划中的处理 |
|---|---|---|
| 官方 HD-TTS 叠加模型 | `missing_ts_exp/proj1_DRR_MSGF/src/models/hdtts_official_reliable.py` | 在此添加小范围可控开关，保留官方骨架 |
| 正式训练入口 | `missing_ts_exp/proj1_DRR_MSGF/src/training/run_mrmsgf_official_controlled.py` | 扩充 CLI、manifest 与审计字段 |
| Point / Block-T 掩码工具 | `missing_ts_exp/proj1_DRR_MSGF/src/data/hdtts_missing.py` | 核心阶段不修改，直接复用已有 Block-T mask |
| 物理区域构造 | `missing_ts_exp/proj1_DRR_MSGF/scripts/r0906_prepare_mrmsgf_regions.py` | 复用现有物理分区；新增功能/随机分区准备脚本 |
| 命令、验证、报告范式 | `missing_ts_exp/proj1_DRR_MSGF/scripts/r0910_*`、`r0917_*` | 复制结构，不复用旧结果或旧目录 |
| 已有共享资源 | `missing_ts_exp/proj1_DRR_MSGF/resources/r0905_blockt/` | 只读复用，记录 SHA256 |

当前模型已支持的关键开关为：

- `--mr_variant phy_only`：物理细图；
- `--mr_variant obs_only`：观测细图；
- `--mr_variant fine`：逐边融合细图、不启用新增区域支路；
- `--mr_variant coarse_no_gate`：区域支路但无区域注入门；
- `--mr_variant full`：完整区域支路；
- `--q_mode observed|permuted_pairwise`：真实/置换的成对共同观测矩阵。

现有 `r0917` 协议闭环已经证明：同版 Full 与 Permuted-Q 的实现路径可被 manifest 审计；但其结果不支持把“正确节点对上的 q 身份”作为主要机制结论。因此，本计划不再把 Permuted-Q 作为主要正向证据，只保留它作为论文用语的边界。

### 1.2 已有结果能回答什么，不能回答什么

在历史 `METR-LA / Block-T 50%` 消融中，物理细图、观测细图与融合细图的 MAE 分别约为 3.329、3.335、3.318；该现象与双图互补一致。但这些旧结果来自更早的组合式消融，不能隔离“固定比例、全局比例、逐边比例”之间的差异，也没有形成当前代码版本下的专门实验矩阵。

`Block-T 70%` 与 `AQI36` 的结果并不稳定，说明不能先验认定融合、区域化或 `q_ij` 会在所有数据和所有缺失条件下有效。因而本计划先追求机制清晰度，再考虑泛化范围。

---

## 2. 固定协议与数据处理

### 2.1 核心数据与任务

核心验证只用 METR-LA：

| 项目 | 固定设置 |
|---|---|
| 原始数据 | `dataset/METR-LA/metr-la.h5` |
| 物理图 | `dataset/METR-LA/adj_mx.pkl` 中的既有道路图 |
| 历史 / 预测长度 | 24 / 12 |
| 数据切分 | 复用 `run_mrmsgf_official_controlled.py::make_datamodule` 的时间顺序切分；不得重新划分 |
| 输入缺失 | Block-T 50% |
| 人工 mask | `resources/r0905_blockt/masks/METR-LA/block_t_r50_seed{seed}_observed.npy` |
| 预测标签 | 不额外遮蔽未来标签；在原始有效 future target 上评估 |
| 主指标 | unmasked future MAE |
| 辅指标 | unmasked future RMSE、按 horizon 3/6/12 的 MAE |
| 优化器与选择 | AdamW，沿用当前 `lr=1e-3`、`weight_decay=0.01`，validation MAE 最优 checkpoint |
| 训练预算 | 100 epochs、每 epoch 300 train batches、batch size 32、patience 20 |

不在本轮通过改变优化器、隐藏维度、区域大小、top-k 或训练轮数去追逐结果。`graph_topk=8`、区域最大大小 8、HD-TTS 的 64 维隐藏表示均固定。

### 2.2 数据处理原则

1. 原始数据的时间顺序切分与归一化只使用训练部分的统计量，完全沿用 runner。
2. Block-T mask 只作用于历史输入；真实 future label 不能因人工 mask 而被排除。
3. 同一 seed 的所有变体必须加载同一 `.npy` mask，且在 manifest 中保存 mask SHA256。
4. 物理图仅在模型中加入自连接并行归一化；用于物理区域构造时的对称化规则必须与既有实现保持一致并记录。
5. 功能静态区域只能由**训练时间段**的原始有效历史构造；禁止读取 validation/test 的值、未来标签或人工缺失后的测试窗口。
6. 动态软区域只可使用当前历史窗口的编码表示、mask 和观测图，不可读取未来 target。

### 2.3 当前核心阶段不需要的新数据

核心 P0--P3 不下载新数据、不生成新的缺失率、不引入 AQI36。AQI36 仅作为最后的外部边界验证候选，是否启动取决于 METR-LA 机制确认结果。

---

## 3. 项目目录与产物约定

本目录将只保存本项目的新计划、配置、命令、验证和报告；已有 `r0905`--`r0917` 目录保持只读历史证据。

```text
missing_ts_exp/proj1_DRR_MSGF/
├── docs/1115实验计划_DRR-MSGF双图融合与区域化验证.md  # 本文件
├── README.md                            # 实施状态、唯一入口与结果索引（拟新增）
├── configs/                             # 每一阶段冻结的 JSON 配置（拟新增）
├── scripts/                             # 命令构建、审计、验证、报告脚本（拟新增）
├── resources/
│   ├── manifests/                       # 指向/记录共享 data、mask、图、分区 SHA
│   └── regions/                         # 新产生的 functional/random assignment
├── results/
│   ├── p0_audit/
│   ├── p1_fusion/
│   ├── p2_order/
│   └── p3_regions/
├── logs/                                # 仅本项目训练日志
└── reports/                             # 自动生成的阶段报告与图
```

每个正式 JSON 除现有字段外，必须记录：`protocol_id`、模型代码 SHA、项目配置 SHA、数据/mask/region SHA、`fusion_mode`、`regional_order`、`region_mode`、训练预算、seed、checkpoint、未遮蔽 MAE/RMSE。不同阶段不得复用 tag 或覆盖结果。

---

## 4. 阶段 P0：冻结协议与实现审计

### 目的

避免将新变体的性能差异误归因于数据通路、区域文件、未记录的代码改动或参数数量差异。

### 需要做的工作

1. 将现有模型与 runner 的受控文件 SHA 写入项目配置。
2. 生成核心三类区域资源的 manifest：现有物理静态分区、训练集功能静态分区、随机大小匹配分区。
3. 为新增融合方式、区域顺序和动态软分区写 CPU 单 batch forward/backward 测试。
4. 对每个正式配置执行 `1 epoch / 2 train batches / 2 eval batches` GPU 烟测。

### 必须通过的检查

- 每行传播邻接的元素有限、行和约为 1；
- 物理-only 与观测-only 的图开关确实不调用另一图参与 `A_fine`；
- 固定 0.5 融合严格等于 `RowNorm(0.5 A_phy + 0.5 A_obs)`；
- 全局融合只使用一个可学习标量，逐边融合只使用当前 edge gate；
- 两种区域顺序使用相同的 `A_fine` 和同一组细粒度节点表示；
- 动态软分区的每个节点对所有区域的分配和为 1，且无 NaN/Inf；
- 功能静态分区资源只读取训练时间索引；
- 所有正式变体的数据、mask、窗口元数据和 checkpoint 规则一致。

P0 不产生论文数值。任一检查失败时，应先修正实现并重新运行 P0，而不是继续正式训练。

---

## 5. 阶段 P1：双图融合的最小机制实验

### 5.1 要回答的问题

在不使用新增区域支路的前提下，逐边融合是否优于单一关系图、固定混合或全局混合？

这比“Full 是否优于 HD-TTS”更直接回答双图融合本身是否值得做。

### 5.2 共同控制条件

所有 P1 变体使用 `mr_variant=fine`：保留同一 HD-TTS 主干、同一窗口观测图构造、相同 readout reliability bias、相同参数主体与训练预算；仅改变 `A_fine` 的来源。这样测到的是“细粒度传播关系的组织方式”，不是区域支路或额外 mask 特征。

### 5.3 变体矩阵

| ID | 拟定开关 | `A_fine` 定义 | 目的 |
|---|---|---|---|
| F-P | `fusion_mode=physical` | `A_phy` | 稳定空间先验下界 |
| F-O | `fusion_mode=observation` | `A_obs` | 窗口数据关系下界 |
| F-H | `fusion_mode=fixed_half` | `RowNorm(0.5 A_phy + 0.5 A_obs)` | 检验“简单相加”是否已足够 |
| F-G | `fusion_mode=global_scalar` | `RowNorm(g A_phy + (1-g) A_obs)`，单个可学习 (g) | 检验是否只需全局偏好 |
| F-E | `fusion_mode=edgewise` | 当前逐边 gate 的融合图 | 检验边级异质性是否有价值 |

注意：这里的 `physical`/`observation` 是**细图消息来源**，不表示模型完全不再计算 mask 统计特征；各变体仍接收相同输入，从而避免“谁多拿了缺失特征”的混杂。

### 5.4 执行规模

| 子阶段 | 任务数 | seed | 用途 |
|---|---:|---|---|
| P1-S smoke | 5 | 2024 | 每变体短烟测，不纳入统计 |
| P1-P pilot | 5 | 2024 | 单 seed、完整预算；判断是否值得确认 |
| P1-C confirm | 12 | 2025/2026/2027 | F-P、F-O、F-G、F-E 各 3 次；F-H 不进入确认 |

总计最多 22 次训练，其中 5 次为极短烟测。P1-P 只决定是否启动确认，不产生统计结论；若启动 P1-C，其四个变体和三个 seed 已预先固定，不得再增删。

### 5.5 P1 结果呈现

主表只呈现 P1-C：

| fine relation | MAE | RMSE | Delta MAE vs F-E | 95% CI | paired p (Holm) |
|---|---:|---:|---:|---:|---:|

配对比较预先限定为：F-E vs F-P、F-E vs F-O、F-E vs F-G。三项 MAE 检验采用 Holm 校正。F-H 的单 seed 结果只进入开发记录，用于确认固定混合没有实现错误，不作论文显著性主张。

辅助图只做一张：按 `q_ij` 分桶、物理边/非物理边分组展示 F-E 的 gate 分布。该图只能说明模型响应，不能作为“边正确性”的证据。

### 5.6 P1 决策闸门

- 若 F-E 在三个确认 seed 的均值不优于 F-P、F-O 与 F-G 中至少两个，或配对差的 CI 大幅跨 0，则停止把“逐边融合优越性”作为论文主张；后续区域实验仍可做，但只能定位为架构探索。
- 若 F-E 对至少两个单源/全局融合有一致正向配对差，进入 P2。

不以单 seed pilot 的好坏决定结论，也不增加 seed、修改超参数或调节 gate 后再重复 P1。

---

## 6. 阶段 P2：先融合还是先区域化

### 6.1 要回答的问题

论文当前的结构主张是：在节点边层得到融合图，再由它诱导区域图。

当前方式为：

\[
A^{\mathrm{coarse}}_{\mathrm{FTR}}
= \operatorname{RowNorm}(S^\top \operatorname{Mix}(A^{\mathrm{phy}},A^{\mathrm{obs}})S).
\]

需要对照的方式为：

\[
A^{\mathrm{coarse}}_{\mathrm{CTF}}
= \operatorname{RowNorm}\!\left(
\bar g\odot\operatorname{RowNorm}(S^\top A^{\mathrm{phy}}S)
+(1-\bar g)\odot\operatorname{RowNorm}(S^\top A^{\mathrm{obs}}S)
\right).
\]

二者在逐边 gate 与归一化存在时不等价。该实验检验的是融合发生的位置，而不是“有没有区域模块”。

### 6.2 公平实现约束

为使差异只来自区域图构造顺序，两个变体必须：

- 共用相同的窗口 `A_obs`、`A_phy`、逐边 `g_ij` 与细粒度 `A_fine`；
- 共用由 `A_fine` 得到的 `Z_fine`；
- 共用物理静态分区 `S_phy`、区域特征均值、coarse message 参数、区域注入门和读出；
- **只**替换 `_coarse_propagate` 中的 `A_coarse`。

对于 CTF，区域级 gate 使用同一 `g_ij` 在区域对上的归并值：

\[
\bar g_{kl}=
\frac{\sum_{i\in k,j\in l}g_{ij}}
{\lvert k\rvert\lvert l\rvert}.
\]

它不增加新的 gate 网络，也不重新搜索参数。这样比“另写一个区域 gate”更干净。

### 6.3 变体与规模

| ID | `regional_order` | 描述 |
|---|---|---|
| O-FTR | `fuse_then_coarsen` | 当前论文方式 |
| O-CTF | `coarsen_then_fuse` | 两图先粗化、后在区域边层混合 |

| 子阶段 | 任务数 | seed |
|---|---:|---|
| P2-S smoke | 2 | 2024 |
| P2-P pilot | 2 | 2024 |
| P2-C confirm | 6 | 2025/2026/2027 |

P2 只有在 P1-C 显示 F-E 值得保留时启动。

### 6.4 结果呈现与闸门

主表为两行，报告 MAE、RMSE、逐 seed Delta MAE、95% CI 和 paired p。另生成一张诊断图：同一批窗口下 `A_coarse_FTR` 与 `A_coarse_CTF` 的 Frobenius 差异分布，以及二者的区域边稀疏度；它用于证明比较对象确实不同，不用于声称哪张图更“真实”。

- 若 O-FTR 无一致优势，则论文不得把“先融合再区域化”写成已验证的必要机制，只描述为当前实现选择。
- 若 O-FTR 一致优于 O-CTF，则可将其写为“在该协议下，结果支持先在细粒度关系层处理双源互补，再进行区域聚合”。

---

## 7. 阶段 P3：静态与动态区域归属

### 7.1 先澄清本轮的比较对象

当前模型已经是半动态的：`S` 固定，但 `A_fine`、`A_coarse` 和区域注入门随窗口变化。P3 不把“静态 vs 动态”误写成二元对立，而是比较三种**成员归属**策略：

| ID | 区域成员归属 | 回答的问题 |
|---|---|---|
| R-P | 当前物理硬分区 `S_phy` | 物理稳定锚点是否足够 |
| R-F | 训练历史得到的功能硬分区 `S_func` | 收益来自物理性还是仅来自静态分组 |
| R-D | 物理锚定的窗口软分区 `S_dyn` | 动态调整归属是否值得估计噪声 |
| R-N | 不使用新增区域支路 | 区域支路整体是否有贡献 |

`R-N` 使用现有 `mr_variant=fine`；其余使用 `mr_variant=full`，并固定 P1/P2 中确认的细图融合方式与区域顺序。

本轮不实现“每窗口从零开始的无约束硬聚类”。那种方案会带来簇标签置换、不可微 topological clustering、额外超参数和缺失驱动的分组抖动，无法在小范围实验中得到公平结论。若 R-D 显示价值，才将完全动态硬分区列为后续研究。

### 7.2 三种分区的精确定义

**R-P：物理静态分区**

复用 `physical_connected_max8_assignment.npy`。每个节点唯一归属、区域最大节点数为 8；该文件和 SHA256 原样记录。

**R-F：功能静态分区**

拟新增 `prepare_region_assignments.py`，流程为：

1. 只读取训练段的原始有效 METR-LA 历史；
2. 按训练段统计量标准化，对每个节点对计算 pairwise-complete Pearson 相关；
3. 以非负相关作为功能 affinity，构造对称 top-8 功能图；
4. 使用与物理分区相同的贪心、唯一归属、最大大小 8 规则生成 `S_func`；
5. 保存训练时间索引、有效样本计数、affinity SHA、assignment SHA 与区域大小分布。

R-F 的目的不是宣称 Pearson 图是真实关系，而是提供“非物理但静态”的可重复对照。

**R-D：物理锚定的窗口软分区**

保持区域数 (K) 与物理分区一致。由窗口观测图计算节点到物理区域槽位的支持：

\[
c_{ik}=\frac{\sum_j A^{\mathrm{obs}}_{ij}S^{\mathrm{phy}}_{jk}}
{\sum_j S^{\mathrm{phy}}_{jk}+\epsilon}.
\]

再以固定、预注册的物理锚定系数和温度形成软归属：

\[
S^{\mathrm{dyn}}_{ik}=\operatorname{softmax}_k
\left(c_{ik}/\tau+\lambda_{\mathrm{phy}}S^{\mathrm{phy}}_{ik}\right).
\]

建议固定 `tau=1.0`、`lambda_phy=2.0`，不做搜索。物理项是有限偏置而非 hard mask，因此它允许边界节点随窗口向其他区域槽位分配一部分权重，但不需要重新编号区域。`S_dyn` 只依赖当前历史输入，且在 batch 内形状为 `(B, N, K)`。

### 7.3 P3 先行诊断，不训练

在训练 R-D 前，先加载 P2 选择的一个冻结 checkpoint，在 test window 上计算 `S_dyn`，输出：

- 平均分区熵与有效区域大小；
- 相邻窗口的 `||S_dyn(t)-S_dyn(t-1)||_F`；
- 与 `S_phy` 的平均偏离；
- 低/高节点观测率、长/短 Block-T 子集上的分区熵；
- 每个区域的总软质量是否塌缩到 0。

若出现大量区域塌缩、NaN，或低观测窗口的归属剧烈抖动，则不进入 R-D 正式训练，直接记录“当前窗口证据不足以稳定动态分区”的负结果。

### 7.4 执行规模

| 子阶段 | 任务数 | seed | 说明 |
|---|---:|---|---|
| P3-D diagnostic | 0 | 使用 P2 checkpoint | 不训练，仅结构诊断 |
| P3-S smoke | 4 | 2024 | R-P/R-F/R-D/R-N 的短烟测 |
| P3-P pilot | 4 | 2024 | 完整预算单 seed |
| P3-C confirm | 12 | 2025/2026/2027 | 四个变体各 3 次 |

只有 P3-D 通过、且 P2 已完成，才启动 P3-S/P3-P/P3-C。

### 7.5 结果呈现

预测表：

| regional assignment | MAE | RMSE | Delta MAE vs R-P | 95% CI | paired p (Holm) |
|---|---:|---:|---:|---:|---:|

结构诊断图只做两张：

1. R-D 的分区熵与节点观测率/最长缺失段的关系；
2. 各策略的区域大小分布，以及 R-D 的有效软区域质量分布。

解释规则：

- R-P 优于 R-D：支持“在短窗口连续缺失下，稳定物理归属比动态重分配更合适”。
- R-D 优于 R-P 且诊断稳定：支持“物理锚定的半动态归属值得进一步研究”。
- R-F 优于 R-P：说明区域收益可能来自功能同质性，而非物理连接；需收缩物理区域的叙事。
- R-N 与 R-P 无差异或更优：区域分支不能作为当前论文的主贡献。

---

## 8. 阶段 P4：范围检查，仅在核心机制成立后执行

P4 不是当前启动项。其目标不是追求大量表格，而是检验 P1--P3 的结论是否只依赖 Block-T 50% 这一单元格。

### P4-A：缺失几何的最小外推

若 P1--P3 的确认实验有明确结果，只额外选择一个缺失几何：

- `Point 50%`：检验当共同观测较丰富、没有连续单节点故障时，融合/区域策略是否仍合理；或
- `Block-ST 50%`：检验空间邻近节点同时故障时，物理区域支路是否遭遇其应有的失败边界。

二者不应同时启动。优先选择与 P1 结论最有区分力者：若观测图在 Block-T 下较弱，选 Point；若区域分支在 Block-T 下有益，选 Block-ST 压力测试。

规模：只保留最终两个竞争变体，`2 variants x 3 seeds = 6` 次正式训练。

### P4-B：AQI36 外部边界

只有 METR-LA 核心结论成立后才考虑 AQI36。已有 AQI36 原生缺失结果偏负，因此该阶段的目标是确认边界，而不是寻找正结果。

实施前需要将 runner 从硬编码 METR-LA 改为显式 `--dataset METR-LA|AQI36`，并在 manifest 中记录 AQI36 的 Haversine-Gaussian 物理图和站点坐标 SHA。首轮只做 1 个条件、1 个 seed 的烟测；通过后至多比较最终 Full 与 HD-TTS-AMP 的 3 个 seed。

---

## 9. 拟新增或修改的代码

### 9.1 模型：`src/models/hdtts_official_reliable.py`

| 改动 | 用于 | 约束 |
|---|---|---|
| `fusion_mode` | P1 | `physical`、`observation`、`fixed_half`、`global_scalar`、`edgewise`；现有 `phy_only/obs_only/fine` 兼容映射，避免两套逻辑漂移 |
| `global_fusion_logit` | P1 | 在所有 P1 变体均注册、仅在 F-G 消费的一个标量，保证参数量严格一致 |
| `regional_order` | P2 | 只改变 `A_coarse` 构造，不改变 `Z_fine`、分区或区域 GNN 参数 |
| batch 级 `region_assignment` 支持 | P3 | `_coarse_propagate` 同时接受 `(N,K)` 与 `(B,N,K)`；所有矩阵乘法显式检查形状 |
| `region_mode` | P3 | `physical_static`、`functional_static`、`physical_anchored_dynamic` |
| 诊断导出 | P1--P3 | 增加 gate source fraction、coarse adjacency 摘要、soft assignment entropy/mass/drift；不保存整套大张量到每个结果 JSON |

### 9.2 Runner：`src/training/run_mrmsgf_official_controlled.py`

新增 CLI 参数：

```text
--fusion_mode
--regional_order
--region_mode
--region_assignment_path
--physical_anchor_lambda
--dynamic_region_temperature
--protocol_config_path
```

runner 要扩展 `source_manifest()`：除模型和 runner 外，也纳入项目配置文件、区域资源文件与准备脚本的 SHA256。结果 JSON 必须包含上述参数，避免后续再发生“同名 Full 实际不是同一代码”的问题。

核心 P1--P3 无需修改 `src/data/hdtts_missing.py` 或下载数据；P4 的 Block-ST 或 AQI36 才可能需要扩展数据/掩码入口。

### 9.3 项目专用脚本：拟新增

| 脚本 | 职责 |
|---|---|
| `scripts/prepare_region_assignments.py` | 构造 R-F 与随机负对照分区，生成泄漏审计 manifest |
| `scripts/build_p1_fusion_cmds.py` | 生成 P1 smoke/pilot/confirm 命令，不直接训练 |
| `scripts/verify_p1_fusion.py` | 检查完整性、配对 mask、SHA、参数与配置一致性，计算统计 |
| `scripts/write_p1_report.py` | 从验证摘要生成报告/表格 |
| `scripts/build_p2_order_cmds.py`、`verify_p2_order.py`、`write_p2_report.py` | P2 对应链路 |
| `scripts/diagnose_dynamic_regions.py` | P3-D，不训练，输出分区稳定性诊断 |
| `scripts/build_p3_region_cmds.py`、`verify_p3_regions.py`、`write_p3_report.py` | P3 对应链路 |
| `scripts/run_project_smoke.sh` | 只运行本项目的短烟测命令 |

所有命令生成器必须读取冻结 JSON 配置，禁止将 rate、seed、结果路径、模型开关散落在 shell 脚本中。

---

## 10. 统计、报告与停止规则

### 10.1 统计单位

每个 `(mask seed, initialization seed)` 形成一组配对观测；同一 seed 的不同模型共享输入 mask。统计单位是 seed 级 MAE，不是测试窗口数。

对比较 `variant - reference` 报告：

- 三个原始配对差值；
- mean +/- sample std；
- 95% paired t interval；
- 预先列明比较族内的 Holm 校正 p 值；
- RMSE 作为辅助结果，除非事先声明，否则不单独据此宣称机制有效。

三 seed 的区间必然较宽，因此结果解释优先看方向一致性、效应大小与区间，而不是机械追求 `p < 0.05`。

### 10.2 不允许的做法

- 不因结果不理想再添加 seed、换 mask、调区域大小或改动 gate 后重复同一个假设检验；
- 不只报告对 Full 有利的指标或 horizon；
- 不将 `q_ij` 的 gate 相关性、选边率当作“关系正确”的证明；
- 不把 P1/P2/P3 的开发 seed 2024 与确认 seed 混合后再声称独立验证；
- 不复用 `r0910`/`r0917` 的数值填入新表，除非 manifest 显示代码、配置和资源完全一致。

### 10.3 每阶段停止条件

| 阶段 | 继续条件 | 停止后的论文处理 |
|---|---|---|
| P1 | edgewise 对单源/全局融合至少呈现一致正向趋势 | 不主张逐边融合优越；保留为实现选择 |
| P2 | P1 成立，且 FTR 对 CTF 呈现一致正向趋势 | 不主张“先融合后区域化”必要 |
| P3 | P2 完成且动态分区诊断不塌缩 | 区域成员保持物理静态，强调它是保守设计选择 |
| P4 | P1--P3 至少一项机制结论成立 | 不外推跨缺失几何/跨域能力 |

---

## 11. 实施顺序与预计工作量

### 11.1 推荐顺序

1. P0：实现开关、区域资源与烟测审计；
2. P1：优先验证双图融合本身；
3. P2：仅在 P1 值得保留时验证融合顺序；
4. P3：最后讨论区域成员动态性；
5. P4：只作为最终边界检查，不在核心结果不清楚时启动。

### 11.2 最大训练数量

| 阶段 | 烟测 | 单 seed pilot | 三 seed confirm | 正式训练上限 |
|---|---:|---:|---:|---:|
| P1 双图融合 | 5 | 5 | 12 | 17 |
| P2 融合顺序 | 2 | 2 | 6 | 8 |
| P3 区域归属 | 4 | 4 | 12 | 16 |
| 核心总计 | 11 | 11 | 30 | 41 |

这些阶段是串行闸门，不是需要立刻同时提交的 41 个任务。若 P1 失败，后续可直接停止，核心正式训练只有 17 次；若 P1、P2 都成立，才值得花资源推进 P3。

---

## 12. 最终可形成的论文证据

若 P1--P3 按计划成功完成，论文可分别获得：

| 研究问题 | 对应证据 | 允许的谨慎表述 |
|---|---|---|
| 双图是否值得融合 | P1 的单源、固定、全局与逐边融合比较 | “逐边双关系融合在该缺失协议下优于所比较的单源与全局融合替代。” |
| 先融合是否有必要 | P2 的 FTR vs CTF | “结果支持在区域化前于细粒度边层整合两类关系。” |
| 静态还是动态区域 | P3 的 P/F/D/N 对照及稳定性诊断 | “在当前短窗口连续缺失设定中，稳定物理归属/物理锚定动态归属的相对适用范围。” |

若任何阶段为负结果，它仍然有研究价值：它会明确 DRR-MSGF 应收缩哪一条机制主张，而不是用更多未经控制的实验掩盖边界。
