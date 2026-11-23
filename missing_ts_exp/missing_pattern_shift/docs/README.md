# 文档导航

按研究路线组织文档。根目录保留共同实验框架、既有基线结果与实现核查；OCPE 路线集中在 `ocpe/`。

| 内容 | 入口 |
|---|---|
| OCPE 方法、三数据集结果和实验计划 | [OCPE 路线文档目录](ocpe/README.md) |
| OCPE 已完成的阶段材料与运行追溯 | [OCPE 归档说明](ocpe/archive/README.md) |
| S4M、CTF、BiTGraph、PatchTST、DLinear、iTransformer 的既有结果 | [统一基线报告](1113-实验结果-统一框架-r50-2x2矩阵.md) |
| Traffic 与高通道数探索等早期历史材料 | [原基线阶段归档](archive/README.md) |

## 另一条路线的调研与候选方案

- [跨领域缺失模式变化调研](1119-文献调研-跨领域缺失模式变化.md)及[配套 PDF](1119-文献调研-跨领域缺失模式变化.pdf)。
- [观测归一化残差校正候选方案](1119-研究路线-观测归一化残差校正.md)。
- 相关附件在 `literature_sources/1119_cross_domain_missingness_shift/`，其中包括[参考文献记录](literature_sources/1119_cross_domain_missingness_shift/references.json)与[调研检索记录](literature_sources/1119_cross_domain_missingness_shift/search_log.json)。

这些材料已移出 OCPE 目录，不视为 OCPE 专属文献库或已确认的后续路线。

其他模型配置、CUDA 吞吐、优化器和掩码修订文件，按各自时间和适用协议查阅。不同实验套件作数值比较时，需核对掩码版本和评价口径。

本次调整只整理文档及其附件位置，并修正文档入口；模型、原始实验结果、日志和检查点的存放位置保持不变。
