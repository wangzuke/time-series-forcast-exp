# OCPE 逐篇阅读与对应关系

核验日期：2026-11-20。以下是相关部分的定向阅读笔记，不代表逐篇全文及全部代码审计。对于未确认的 P↔B 实验，不把“未确认”改写成绝对不存在。

## R01｜PPCA

**Probabilistic Principal Component Analysis**  
Michael E. Tipping; Christopher M. Bishop；1999；Journal of the Royal Statistical Society: Series B。

- 来源定位：原实验计划附录 B 已列。
- 本次读取：PDF 相关小节核验；非全文逐段精读。重点位置：§2.1 因子分析；§3.1 式(6)；§4.1 缺失数据。
- 对 OCPE 的对应：低秩高斯生成模型与潜变量条件分布；非全等复现。
- 场景证据：未核实 P↔B 训练测试错配。
- 需要保留的区别：FA 式异方差噪声区别于 PPCA 各向同性噪声。
- 年份与版本：1999 期刊卷年；2002 是期刊网页上线时间，不是方法提出年份。
- [一手入口](https://academic.oup.com/jrsssb/article/61/3/611/7083217)；[原文 PDF 在线入口](https://www.cs.columbia.edu/~blei/seminar/2020-representation/readings/TippingBishop1999.pdf)；[本地 PDF](sources/R01_paper.pdf)；[可搜索文本](sources/R01_paper.txt)。

## R02｜Linear Gaussian review

**A Unifying Review of Linear Gaussian Models**  
Sam T. Roweis; Zoubin Ghahramani；1999；Neural Computation。

- 来源定位：本轮补充纳入；不追溯为原计划引用。
- 本次读取：作者 PDF 的模型与相关小节核验。重点位置：§4 推断讨论；§5.1 静态数据模型与因子分析。
- 对 OCPE 的对应：区分静态因子模型与动态 Kalman 模型。
- 场景证据：未核实 P↔B 训练测试错配。
- 需要保留的区别：当前 OCPE 没有隐状态转移，不能称为动态 Kalman 前端。
- 年份与版本：1999 正式期刊论文。
- [一手入口](https://cs.nyu.edu/home/people/in_memoriam/roweis/papers/NC110201.pdf)；[原文 PDF 在线入口](https://cs.nyu.edu/home/people/in_memoriam/roweis/papers/NC110201.pdf)；[本地 PDF](sources/R02_paper.pdf)；[可搜索文本](sources/R02_paper.txt)。

## R03｜Arbitrary missingness factor model

**Maximum Likelihood Estimation of Factor Models on Datasets with Arbitrary Pattern of Missing Data**  
Marta Bańbura; Michele Modugno；2014；Journal of Applied Econometrics。

- 来源定位：此前讨论的相关依据；本轮区分文稿版本。
- 本次读取：web 工具读取工作论文相关部分；期刊全文未取得，本地 PDF 下载失败已记录。重点位置：ECB WP 1189 §2 模型与估计；附录 B 推导；期刊摘要及元数据。
- 对 OCPE 的对应：任意观测子集下因子模型估计的既有路线。
- 场景证据：任意缺失模式处理不等于已验证 P↔B 错配。
- 需要保留的区别：采用动态因子、EM/Kalman；当前 OCPE 为静态窗口前端加预测主干。
- 年份与版本：2010 工作论文；2012 在线发表；2014 正式卷期。
- [一手入口](https://onlinelibrary.wiley.com/doi/10.1002/jae.2306)；[原文 PDF 在线入口](https://www.ecb.europa.eu/pub/pdf/scpwps/ecbwp1189.pdf)；本地 PDF 下载失败，参见 source_fetch_log.json 与 sources/web_versions.json。

## R04｜RKN

**Recurrent Kalman Networks: Factorized Inference in High-Dimensional Deep Feature Spaces**  
Philipp Becker; Harit Pandya; Gregor Gebhardt; Cheng Zhao; C. James Taylor; Gerhard Neumann；2019；International Conference on Machine Learning (PMLR 97)。

- 来源定位：此前讨论提及；非原计划直接引用。
- 本次读取：PMLR 页面与方法相关 PDF 小节核验。重点位置：§2；§2.3–2.6 因子化协方差、推断与网络；§3 实验概况。
- 对 OCPE 的对应：显式状态不确定性、可微解析更新、端到端训练已有先例。
- 场景证据：已读部分未确认我们的 P↔B 错配协议。
- 需要保留的区别：动态状态、神经观测编码与因子化协方差；不能据此声称 OCPE 的统计更新新颖。
- 年份与版本：ICML 2019。
- [一手入口](https://proceedings.mlr.press/v97/becker19a.html)；[原文 PDF 在线入口](https://proceedings.mlr.press/v97/becker19a/becker19a.pdf)；[本地 PDF](sources/R04_paper.pdf)；[可搜索文本](sources/R04_paper.txt)。

## R05｜CRU

**Modeling Irregular Time Series with Continuous Recurrent Units**  
Mona Schirmer; Mazin Eltayeb; Stefan Lessmann; Maja Rudolph；2022；International Conference on Machine Learning (PMLR 162)。

- 来源定位：原实验计划附录 B 已列。
- 本次读取：PMLR 页面与方法相关 PDF 小节核验。重点位置：§3 方法；§3.2 连续离散 Kalman filter；§3.3–3.5 参数化与训练。
- 对 OCPE 的对应：概率状态推断与神经时序模型结合的设计先例。
- 场景证据：不规则采样实验不自动等同 P↔B 错配。
- 需要保留的区别：CRU 有线性 SDE、动态传播和噪声观测；OCPE 没有这一步。
- 年份与版本：ICML 2022。
- [一手入口](https://proceedings.mlr.press/v162/schirmer22a.html)；[原文 PDF 在线入口](https://proceedings.mlr.press/v162/schirmer22a/schirmer22a.pdf)；[本地 PDF](sources/R05_paper.pdf)；[可搜索文本](sources/R05_paper.txt)。

## R06｜BTF/BTMF

**Bayesian Temporal Factorization for Multidimensional Time Series Prediction**  
Xinyu Chen; Lijun Sun；2022；IEEE Transactions on Pattern Analysis and Machine Intelligence。

- 来源定位：此前讨论提及；非原计划直接引用。
- 本次读取：作者提供的 early-access PDF 方法与实验设置核验；正式卷期对照作者出版列表。重点位置：§4.1–4.3 BTMF；§4.2 式(14)；预测算法及§6 实验设置。
- 对 OCPE 的对应：低秩因子、观测条件推断、预测与不确定性已有结合。
- 场景证据：随机/块缺失实验不能单独证明训练测试跨模式错配。
- 需要保留的区别：BTF 含矩阵与张量版本；BTMF 是其中矩阵版本；时间先验为 VAR，推断为 Gibbs。
- 年份与版本：2019 arXiv；2021 early access（本地作者 PDF）；2022 正式卷期。Crossref 尚返回早期页码 1--1。
- [一手入口](https://arxiv.org/abs/1910.06366)；[原文 PDF 在线入口](https://lijunsun.github.io/files/papers/2021-TPAMI-BTMF.pdf)；[本地 PDF](sources/R06_paper.pdf)；[可搜索文本](sources/R06_paper.txt)。

## R07｜NeuMiss

**NeuMiss networks: differentiable programming for supervised learning with missing values.**  
Marine Le Morvan; Julie Josse; Thomas Moreau; Erwan Scornet; Gaël Varoquaux；2020；Advances in Neural Information Processing Systems 33。

- 来源定位：此前讨论提及；非原计划直接引用。
- 本次读取：NeurIPS 正式页面与相关 PDF 小节核验。重点位置：§2.2 最优预测器；§3.1–3.2 Neumann 近似与网络；§4 实验概况。
- 对 OCPE 的对应：按观测子集改变条件预测映射与跨模式共享参数已有理论和模型。
- 场景证据：不同缺失机制下实验不自动证明我们的时序 P↔B 错配。
- 需要保留的区别：原始特征空间的协方差子矩阵逆近似，与 OCPE 低秩潜空间解析求解不同。
- 年份与版本：NeurIPS 2020。
- [一手入口](https://proceedings.neurips.cc/paper/2020/hash/42ae1544956fbe6e09242e6cd752444c-Abstract.html)；[原文 PDF 在线入口](https://proceedings.neurips.cc/paper_files/paper/2020/file/42ae1544956fbe6e09242e6cd752444c-Paper.pdf)；[本地 PDF](sources/R07_paper.pdf)；[可搜索文本](sources/R07_paper.txt)。

## R08｜NeuMISE

**Robust prediction under missingness shifts**  
Patrick Rockenschaub; Zhicong Xian; Alireza Zamanian; Marta Piperno; Octavia-Andreea Ciora; Elisabeth Pachl; Narges Ahmidi；2024；arXiv preprint。

- 来源定位：此前讨论提及；本轮核验 §5.1 NeuMISE。
- 本次读取：arXiv v1 HTML/PDF 的相关章节核验；未据其他匿名投稿认定正式发表。重点位置：§3–4 问题与定理1；§5.1 NeuMISE；§7 与缺失场景附录。
- 对 OCPE 的对应：可忽略缺失变化下条件预测的理论边界及 NeuMISE 模型创新。
- 场景证据：明确研究 source/target 缺失机制变化；当前已核验实验是一般表格/合成设置，非我们的时序块协议。
- 需要保留的区别：缺失率、机制及结果依赖性有专门讨论；理论依赖数据生成与可忽略性等条件，不能直接搬成 OCPE 定理。
- 年份与版本：本资料引用 2024 arXiv v1；未确认对应正式卷期。
- [一手入口](https://arxiv.org/abs/2406.16484)；[原文 PDF 在线入口](https://arxiv.org/pdf/2406.16484v1)；[本地 PDF](sources/R08_paper.pdf)；[可搜索文本](sources/R08_paper.txt)。

## R09｜LIFTS

**Latent-driven Imputation and Forecasting for partially observed Time Series (LIFTS)**  
Sanyou Wu; Ying Wei; Kaizheng Wang；2026；arXiv preprint。

- 来源定位：此前讨论提及；本轮核验具体机制。
- 本次读取：arXiv v2 方法、线性特例和实现相关部分核验；未逐一审计所有证明。重点位置：§2 生成模型；§3 前向/后向推断；§4 学习；§5.3 线性特例；§6.1 实现。
- 对 OCPE 的对应：潜状态、观测掩码与部分观测时序预测的强相关先行框架。
- 场景证据：已读部分未确认 P↔B 源目标错配实验。
- 需要保留的区别：通用方法是非线性状态空间、可学习生成式推断与 energy-score 训练；不能概括成和 OCPE 一样的静态高斯闭式编码。
- 年份与版本：2026-10-02 v1；2026-10-26 v2；本次按预印本处理。
- [一手入口](https://arxiv.org/abs/2610.03916)；[原文 PDF 在线入口](https://arxiv.org/pdf/2610.03916v2)；[本地 PDF](sources/R09_paper.pdf)；[可搜索文本](sources/R09_paper.txt)。

## R10｜S4

**Efficiently Modeling Long Sequences with Structured State Spaces**  
Albert Gu; Karan Goel; Christopher Ré；2022；International Conference on Learning Representations。

- 来源定位：原计划引用 S4 官方实现；本轮补全论文条目。
- 本次读取：原论文方法相关部分与已使用官方实现来源核对。重点位置：§2–3 SSM 与 S4 参数化；官方实现说明。
- 对 OCPE 的对应：当前普通 S4 DPLR 主干的出处。
- 场景证据：本引用不构成缺失错配方法证据。
- 需要保留的区别：S4、S4D 与 S4M 不可混写；主干引用与 OCPE 概率前端依据分开。
- 年份与版本：2021 arXiv 首发；ICLR 2022；本地 PDF v3。
- [一手入口](https://arxiv.org/abs/2111.00396)；[原文 PDF 在线入口](https://arxiv.org/pdf/2111.00396v3)；[本地 PDF](sources/R10_paper.pdf)；[可搜索文本](sources/R10_paper.txt)。

## R11｜VaryST

**VaryST: Generalizable Spatio-Temporal Forecasting Beyond the Closed-World Setting**  
Shengnan Guo; Jinjun Cheng; Xiaowei Mao; Jianhao Zhang; Haochen Lv; Yang Jiang; YiHeng Huang; Yan Lin; Youfang Lin; Huaiyu Wan；2026；arXiv preprint。

- 来源定位：此前场景讨论提及；不是 OCPE 设计来源。
- 本次读取：arXiv v1 方法相关小节、缺失输入协议及表12核验。重点位置：§4.4 IGOP；§5.5；附录 B.4.3 与 C.3 表12。
- 对 OCPE 的对应：确有跨散点/时空块缺失的 source→target 实验，适合场景比较。
- 场景证据：附录 C.3 表12：30% 缺失 PEMS04/08，RM→CM 与 CM→RM。
- 需要保留的区别：2016步历史/12步预测、时空块、IGOP 和指标掩码等与当前 L96/H96 每通道 B24 不一致，不能直接横比数值。
- 年份与版本：2026-10-14 arXiv v1；本次按预印本处理。
- [一手入口](https://arxiv.org/abs/2610.19813)；[原文 PDF 在线入口](https://arxiv.org/pdf/2610.19813v1)；[本地 PDF](sources/R11_paper.pdf)；[可搜索文本](sources/R11_paper.txt)。

