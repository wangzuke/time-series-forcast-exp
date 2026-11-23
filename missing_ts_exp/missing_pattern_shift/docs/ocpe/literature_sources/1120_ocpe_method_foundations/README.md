# OCPE 专属文献资料

整理日期：2026-11-20。范围：当前“观测条件后验编码 OCPE + 普通 S4”原型的方法依据、相近方法和缺失机制变化场景。

**这是本轮新建的定向调研资料包。已有统计基础得到核验，但本轮没有得到“OCPE 完整组合此前无人研究”的结论。** 不将本轮补充文献追溯成最初设计来源。

## 建议阅读顺序

1. [调研报告](1120-文献调研-OCPE方法依据与相近工作.md)：先看哪些属于既有工具，以及当前创新边界。
2. [逐篇阅读笔记](reading_notes.md)：每篇论文的问题、对应模块、差别和应读章节。
3. [文献表 CSV](references.csv)：按来源、年份、关系、核验深度筛选；[JSON](references.json) 是结构化主表，[BibTeX](references.bib) 用于引用。
4. [检索日志](search_log.json)、[引用核验](citation_verification.json)与[相似性核查](route_novelty_checks.json)：核查证据从哪里来、还缺什么。
5. [PDF 阅读版](1120-文献调研-OCPE方法依据与相近工作.pdf)及[文献—模块关系图](ocpe_evidence_map.png)。公式以 Markdown 原稿为准。

## 资料如何分层

| 用途 | 条目 | 引用时的边界 |
|---|---|---|
| 已在原计划中注明的基础或实现依据 | R01 PPCA；R05 CRU；R10 S4 | 原计划对 S4 引用的是官方仓库，本轮补全论文 |
| 本轮补充的理论解释 | R02 线性高斯模型综述 | 用于区分静态因子模型和动态滤波；不追溯为原计划来源 |
| 需要比较的相近方法 | R03 缺失因子模型；R04 RKN；R06 BTF/BTMF；R07 NeuMiss；R09 LIFTS | 方法相关并不等于做了 P↔B 训练测试错配 |
| 缺失机制变化与跨模式场景 | R08 NeuMISE；R11 VaryST | 前者是一般缺失机制变化；后者确有表12跨模式实验，协议与我们不同 |

主表含 11 篇研究文献。调研工具的来源单独记在[工具与复现说明](tooling_and_reproducibility.md)，不混入 OCPE 方法依据。

## 原文和复现

- `sources/`：一手页面快照、10 篇论文的 PDF 与提取文本、原始检索结果。R03 的 ECB 工作论文下载遇到 TLS 证书错误，本轮通过 web 工具核对相关部分，保留在线入口与失败记录。
- `source_manifest.json`、`source_fetch_log.json`：原文入口、文稿版本、下载结果与校验值；网页返回 200 也不一定是有效学术结果，DBLP 的验证页已标为失败。
- `collect_sources.py`：重新采集原文和 DOI 元数据；会访问网络，且可能遇到限流或访问控制。
- `build_review_artifacts.py`：离线重建 CSV、BibTeX、引用核验汇总、关系图和阅读 PDF；不读取实验数据。
- `scope.md`：本轮纳入范围与判据。

本目录独立于其他路线的跨领域调研。返回 [OCPE 主目录](../../README.md)。
