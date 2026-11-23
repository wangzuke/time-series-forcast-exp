# 1123 实验执行：DRR-MSGF 融合图联合分区

## 本轮运行

沿用 METR-LA Block-T 50% 和 seed 2024，比较单区容量上限 8、16、32 下的联合静态与联合动态划分，共 6 组。两张图先融合，由边权决定成员和实际区域数量。P1 Edgewise checkpoint 中的图生成部分在本轮冻结。

队列先构建训练期参考图及逐窗归属，再在 GPU 6 执行六组烟测和六组完整预算训练。用静态和动态两组的平均验证 MAE 锁定共同容量，只用选中的一对 checkpoint 评估原始有效未来标签。图来源对照属于计划中的下一阶段。

GPU 6 空闲时启动首项任务。新任务加入前检查剩余显存和利用率，最多同卡并行两项。固定单区容量上限并不限制区域数，模型按每个 batch 的实际最大区域数对齐批量形状。

## 链路及产物

- [实验计划](1123实验计划_DRR-MSGF融合图联合分区与区域规模验证.md) · [冻结配置](../configs/joint_regions.json)
- [硬分区算法](../src/data/joint_regions.py) · [分区缓存生成](../scripts/prepare_joint_regions.py) · [训练与测试](../scripts/train_joint_regions.py) · [调度与汇总](../scripts/run_joint_regions.py)
- [缓存目录](../resources/1123_joint_regions) · [运行日志](../logs/1123_joint_regions) · [结果目录](../results/1123_joint_regions)

训练前审计相同窗口的在线与离线细图，替换未观测输入值以核验图保持一致，并校验填充区域对节点传播的影响。训练预选阶段保留原受控 runner 的训练与验证标签口径，最终测试恢复原始有效未来标签，预测输入继续使用带有人工缺失的历史。
