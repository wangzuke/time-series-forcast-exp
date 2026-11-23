# 工具、来源与复现说明

## 使用的调研技能

本资料包使用 [literature-review 技能](/home/wangzk/.codex/skills/literature-review/SKILL.md) 组织范围、来源、阅读深度和引用核验。它是文献整理工具来源，不属于 OCPE 方法来源。

技能来源：Timothy Kassis, Vinayak Agarwal, Yuhuan He, Darshil Patel, Aubrey M. Brueckner (2026). *Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents*. arXiv:2609.00065。本轮核对的入口最新版本为 v2；引用使用不带版本后缀的稳定入口。[来源](https://arxiv.org/abs/2609.00065)

## 环境实际情况

- 未安装 parallel-cli，使用 web.run 进行学术搜索及一手页面读取，并补充 arXiv/Crossref 查询。DBLP 的 API 返回验证页面，保留为访问失败，不计成功结果。
- 技能建议的 AI 图示脚本因缺少 OPENROUTER_API_KEY 无法运行；未要求补充凭证。关系图改用 ReportLab 绘制，内容与生成代码可复现，不冒充 AI 图示。
- 使用技能自带 CitationVerifier 检查四项出版商 DOI；arXiv 条目另通过官方元数据和文稿版本核对。DOI 解析结果与元数据匹配分别记录。
- 使用 requests 获取原文，pdftotext 提取可检索文本，ReportLab 生成图示和中文阅读 PDF。下载与原始文本只作本地阅读依据，不把全文复制进调研报告。
- PDF 阅读版不排印完整 LaTeX 公式，公式以 Markdown 原稿为准。

## 离线重建

在仓库根目录执行：

```bash
python3 missing_ts_exp/missing_pattern_shift/docs/ocpe/literature_sources/1120_ocpe_method_foundations/build_review_artifacts.py
```

会读取本目录现存资料并重建 CSV、BibTeX、关系图、核验汇总、检索汇总和报告 PDF。不会修改实验模型、训练脚本、检查点或实验结果。

如需重新请求网络来源，显式运行 collect_sources.py。其输出会随网站变化，访问控制和版本变化也应保留记录；不要把重跑后的日期混同于最初设计日期。
