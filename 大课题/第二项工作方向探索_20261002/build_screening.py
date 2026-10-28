"""Build the audit table from explicit per-paper judgments; never change the archived catalog."""
import csv, json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
OLD=ROOT.parent/"文献调研_20260926"
judgments = """P01|直接预测基线|A;C|掩码卷积与图聚合；需超出散点协议并控制骨干差异
P02|直接竞争/空间骨干|A;C|已含缺失条件多尺度融合；防止重复第一项工作
P03|任务边界/强缺失扩展|A;C|已知节点整列历史缺失并预测全部变量；与一般点块主任务分列
P04|长空洞机制/基线|A;C|原型记忆和双流S4已处理块缺失；不能简单加入记忆
P05|直接竞争/训练策略|A|完整观测教师与多缺失率对比；须登记额外完整数据
P06|直接竞争/联合目标|B|共享表征联合插补预测；简单联合损失不构成充分差异
P07|接口基线/局限证据|A;C|作者明确时间动力学和高维token成本局限；并承认简单插补的适用条件
P08|直接竞争/一致性目标|A;B|已有不完整输入一致性与信息瓶颈；取消完整教师本身不新
P09|整变量与不确定性支撑|B|整变量依赖恢复；任务和发表状态需独立限定
P10|直接竞争/异步块缺失|A;C|训练patch遮蔽和测试块缺失已有证据；适合作更强参照
P11|任务边界|扩展|输出仍可用变量的VSF；不能直接与全变量预测排名
P12|生成机制/任务边界|B|条件流可借鉴；原任务是可用变量子集预测
P13|记忆与图基线|A;C|多种块缺失和局部全局记忆已存在
P14|训练流程对照|B|插补预训练后预测；顺序学习不应误写为同时优化
P15|任务导向插补线索|B|在线图优化强调预测价值；摘要证据不足以细述协议
P16|归纳任务/概率支撑|B;扩展|训练未见节点且量化不确定性；不等于临时掉线
P17|非深度流程对照|B|张量补全结合图预测；掩码与维度未核清
P18|结构和插补对照|B;扩展|动态图和插补已有工作；与第一项重叠且全文证据不足
P19|统计结构基线|B;C|低秩动态模态分解；可检验深度复杂性必要性
P20|应用补充|C|长时预测嵌入模块；原协议信息尚不充分
P21|不完整训练目标线索|A;B|缺失感知预训练；只有摘要证据，不据此断言缺口
P22|近邻监测/暂缓主证据|A;C|动态注意力图与第一项可能重叠；实验字段待核
P23|概率建模核心支撑|B|联合输入目标分布与缺失边缘化已存在；低维风电限制
P24|外生与跨站点支撑|A;C|跨域和协变量信息重要；信息预算与输入组合待核
P25|概率联合目标线索|B|非参数分位数联合插补预测；原文全文仍需补核
P26|直接时空概率近邻|B|贝叶斯空气质量预测已覆盖多种缺失几何
P27|归纳扩展线索|扩展|跨位置泛化与可靠性；目标集合和数据规模待核
P28|应用边界|背景|尚不能清楚排除同时刻估计；不作为严格未来预测主证据
P29|生成预测核心对照|A;B|推理时缺失上下文；完整序列训练和采样成本必须登记
P30|联合分布核心支撑|B|支持缺失训练和联合预测；不能称概率联合建模未研究
P31|联合分布基线|B|TACTiS改进；预测插值与缺失上下文实验要分开
P32|连续时间机制|C|演化流减少ODE求解；块缺失鲁棒性需单独验证
P33|连续时间概率机制|B;C|状态演化和不确定性融合；可用于空洞期间的状态处理
P34|稀疏表示直接近邻|C|观测为二部图边、支持未来查询；不以稀疏表示本身主张创新
P35|片段图直接近邻|C|非规则片段和时间自适应变量图；注意USHCN站点作样本
P36|高阶观测表示近邻|C|观测超图；不能推定已覆盖高维路网共同掉线
P37|层次片段直接近邻|C|已有片段内/间多尺度；数据总数含分类任务
P38|递归表示直接近邻|C|原始时间戳与递归尺度；只沿用旧轮关键全文证据
P39|时频方法直接近邻|C|本轮官方确认ICML2026；非均匀频谱已有路线
P40|概率流机制|B|未来任意查询联合分布；任务适配与密度评分需统一
P41|分布一致性机制|B|查询子集边缘一致性；不是缺失不确定性全部问题的答案
P42|时间表示基础|C|主要插补/分类；若改成预测需声明任务改造
P43|稀疏概率表示基础|B|异方差插补不等于未来预测；可借鉴密度信息传递
P44|条件生成恢复模块|B|多样本插补；未来后缀实验不能替代缺失历史预测验证
P45|状态空间生成模块|B;C|扩散结合S4；重构和预测表分开，记录迭代采样成本
P46|强插补流程基线|B|图递归插补；仅允许使用预测起点以前的双向历史
P47|稀疏图恢复支撑|B;C|稀疏节点重构；不是节点未来预测的直接证据
P48|插补流程基线|B|固定后续预测器，避免骨干差异；本轮不重复全文核对
P49|时空生成恢复模块|B|图先验与条件扩散；按相同图预算比较
P50|泛化插补强对照|A;B|低秩偏置已做缺失配置泛化；需评价完整预测流程
P51|非均匀频谱支撑|B;C|Lomb-Scargle条件扩散主任务为插补
P52|时频生成支撑|B;C|时频扩散插补已有；下游预测收益另验
P53|多尺度结构参照|背景|学生论坛/稿件；与现有小论文结构近，不作主会证据
P54|表示恢复近邻|B;C|变量身份及混合注意力；预印本和插补任务边界
P55|任务导向核心证据|B|逐点重构价值和下游目标已有专门评估，不重复发现
P56|预训练扩展|A;C|跨领域插补基础模型；预算和语料边界待核
P57|图插补补充|B|动态图插补与已有空间工作重叠；不单列第二项主线
P58|跨域插补补充|A;B|空气质量迁移；未来预测仍需外接验证
P59|物理空间重构扩展|C|长期大气网格插补；不同于传感器未来预测
P60|回溯文献工具|背景|综述用于覆盖检查，不作为单个方法效果证据
P61|评测协议支撑|背景|插补基准，不能当下游预测SOTA表
P62|自然缺失评测支撑|A;B;C|只在可得未来真值评价；规模与单模型输入维度分开
P63|可控动力学诊断|C|已知生成过程有利机制测试，不能代替真实网络
P64|掩码时间间隔基础|A;C|标准输入特征早已存在，原任务主要分类
P65|经典插补基础|B|双向RNN历史恢复；不能使用预测起点后的值
P66|连续状态基础|B;C|空洞中演化与观测更新已有；建立基础对照
P67|连续表示基础|C|受控动力学及插值路径；原任务常为分类
P68|潜变量概率基础|B|GP-VAE恢复不确定性；未来分布能力需另验"""
items={}
for line in judgments.splitlines():
    pid,role,directions,reason=line.split("|",3)
    items[pid]=(role,directions,reason)

# Explicit records added after the archived review, selected for their effect on the decision.
new = [
("N01","StableMiss",2023,"IJCAI","官方会议记录核实","缺失输入预测/交通预测","A","直接竞争",
 "已有掩码分布迁移且有自然缺失交通实验；MCAR/MAR与完整联合分布不变假设",
 "https://www.ijcai.org/proceedings/2023/525","sources/StableMiss.txt","全文关键段落与官方页面","10.24963/ijcai.2023/525"),
("N02","INTER",2025,"IJCAI","官方会议记录核实","不完整序列预测/插补/异常检测","A;C","直接竞争",
 "缺失率相关patch丢弃与缺失感知Transformer；不能把patch丢弃当空白",
 "https://www.ijcai.org/proceedings/2025/734","sources/INTER.txt","全文方法段落与官方页面","10.24963/ijcai.2025/734"),
("N03","APN",2026,"AAAI","官方会议记录核实；预印本首发2025","不规则预测","C","直接竞争",
 "软时间窗与通道独立聚合；全历史摘要与未来查询解码需分清",
 "https://ojs.aaai.org/index.php/AAAI/article/view/39563","sources/APN.txt","全文方法/数据段落与官方页面",""),
("N04","TimeCHEAT",2025,"AAAI","官方会议记录核实","不规则预测/分类/插值","C","直接竞争",
 "局部通道相关、全局通道独立已组合；不能仅拼两类策略",
 "https://ojs.aaai.org/index.php/AAAI/article/view/34076","sources/TimeCHEAT.txt","全文方法段落与官方页面","10.1609/aaai.v39i18.34076"),
("N05","KRNO / Shifting Time",2025,"ICML","官方会议记录核实","连续时间和时空预测","C","机制与边界",
 "连续时间平移算子及可分核；PDE网格与传感器图不能直接等同",
 "https://proceedings.mlr.press/v267/dama25a.html","sources/KRNO.txt","全文方法/实验段落与官方页面",""),
("N06","VIMTS",2025,"ICML","官方会议记录核实","不规则预测/预训练","C","近邻与预训练扩展",
 "视觉MAE、跨变量恢复及粗到细预测已存在；预训练信息预算另列",
 "https://proceedings.mlr.press/v267/hu25y.html","","官方摘要与作者仓库；未重读全文",""),
("N07","Post-Training Adaptive Conformal Prediction for Incomplete Time Series",2026,"TMLR","官方期刊列表核实","不完整时序预测区间校准","B","直接竞争",
 "缺失模式条件覆盖已有工作；全文直接访问失败，不转述完整理论保证",
 "https://openreview.net/pdf?id=KMBU4wx79B","","官方期刊列表及索引原文摘要；PDF浏览器验证阻挡",""),
("N08","Under-Cali",2026,"arXiv；作者稿列KDD2026","作者稿含会议与DOI；出版方页面未成功打开","在线不规则预测","A;B","信息预算边界",
 "获得标签后更新轻量校准器；误差分数不等于完整预测分布",
 "https://arxiv.org/abs/2605.28603","sources/UnderCali.txt","HTML/PDF方法关键段落","10.1145/3770855.3817729"),
("N09","Transformer-based Diffusion models for Hydrological Time Series Probabilistic Imputation and Forecasting",2026,"arXiv","预印本","水文插补/概率预测","B;C","现实场景支撑",
 "自然缺失水文数据；长空洞重构表不等于未来预测表",
 "https://arxiv.org/abs/2607.21200","","HTML数据/方法/评测关键段落",""),
("N10","A Hybrid Attention Model Learning Unified Time-aware Patch Representation for Irregular Multivariate Time Series Forecasting",2026,"arXiv","预印本","不规则序列基础模型预测","C","最新近邻监测",
 "时间感知patch、大规模不规则预训练；尚未核对完整协议和代码",
 "https://arxiv.org/abs/2609.22836","","官方摘要；未精读全文",""),
("N11","AMRC",2025,"arXiv；NeurIPS论文检索中可检得题名","本轮不独立确认最终会议书目","一般时间序列预测","A;B;C","遮蔽训练旁证",
 "有意删去冗余前缀与被动数据缺失不同；不当作缺失预测直接基线",
 "https://arxiv.org/abs/2510.19980","","摘要与原文方法检索段落；非完整评审",""),
]
reread={"P02","P04","P05","P06","P07","P08","P10","P23","P26","P29","P30","P34","P50","P55"}
old=json.loads((OLD/"papers.json").read_text())
assert len(old)==68 and set(items)=={p["id"] for p in old}
rows=[]
for p in old:
    role,directions,reason=items[p["id"]]
    current="本轮全文关键段落复核" if p["id"] in reread else "上一轮笔记/结构化记录复审；不表示本轮重读全文"
    verified=""
    status=p["status"]+"（旧库快照）"
    if p["id"]=="P39":
        current="本轮官方会议页面核验；方法沿用已存全文与笔记"
        verified="https://proceedings.mlr.press/v306/qiu26t.html"
        status="ICML 2026 官方记录核实；修正旧库预印本标签"
    rows.append(dict(id=p["id"],name=p["name"],year=p["year"],task=p["task"],
        directions=directions,role=role,reason=reason,
        old_evidence=p["evidence_level"],current_reading=current,status=status,
        source_url=p["source_url"],new_verified_url=verified,
        local_text=str(Path("../文献调研_20260926/sources")/p.get("local_text","")) if p.get("local_text") else ""))
for pid,name,year,venue,status,task,directions,role,reason,url,local,reading,doi in new:
    rows.append(dict(id=pid,name=name,year=year,task=task,directions=directions,role=role,
      reason=reason,old_evidence="新增",current_reading=reading,status=f"{venue}；{status}",
      source_url=url,new_verified_url=url,local_text=local))
with (ROOT/"全库筛选与用途.csv").open("w",encoding="utf-8-sig",newline="") as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]))
    w.writeheader();w.writerows(rows)
(ROOT/"sources"/"补充文献记录.json").write_text(json.dumps([
 dict(id=x[0],name=x[1],year=x[2],venue=x[3],status=x[4],task=x[5],directions=x[6],role=x[7],reason=x[8],
      source_url=x[9],local_text=x[10],reading=x[11],doi=x[12]) for x in new],ensure_ascii=False,indent=2))
print(f"Audited {len(old)} archived + {len(new)} new = {len(rows)} records; no archived files changed.")

