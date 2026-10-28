"""Generate the reproducible reading package from manually audited papers.json."""
from __future__ import annotations
import csv, html, json, re, textwrap
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FIG = ROOT / 'figures'
FIG.mkdir(exist_ok=True)
PAPERS = json.loads((ROOT / 'papers.json').read_text(encoding='utf-8'))
DOI_REPORT = json.loads((ROOT / 'sources/citation_verification.json').read_text(encoding='utf-8'))
DOI_STATUS = {x['id']: x for x in DOI_REPORT}

def flat(v): return '；'.join(v) if isinstance(v, list) else (v or '')
def esc_bib(s):
    return (s or '').replace('&', r'\&').replace('%', r'\%').replace('_', r'\_')
def clean_slug(s): return re.sub('[^A-Za-z0-9]+', '_', s).strip('_')
def direct_forecast(p):
    # Keep imputation-centric studies in their own evidence group even when they
    # offer a suffix-generation experiment.
    return '预测' in p['task'] and '插补' not in p['task'] and p['task'] not in ['基准','综述']

def write_csv():
    fields = ['id','name','title','year','venue','status','task','missing_patterns','domains',
              'dimension_tags','datasets_dimensions','method','contribution','limitation','evidence_level',
              'evidence_locator','doi','authors','source_url','code_url','local_pdf','access_date']
    with (ROOT/'文献分类总表.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader()
        for p in PAPERS:
            w.writerow({k:flat(p.get(k,'')) for k in fields})

def write_notes():
    groups = [('直接缺失预测与变量/节点缺失', lambda p: direct_forecast(p)),
              ('不规则与概率预测', lambda p: '不规则' in p['task'] or '概率预测/插值统一' in p['task']),
              ('插补与下游评测支撑', lambda p: '插补' in p['task'] or p['task']=='评测方法'),
              ('基准、综述与历史基础', lambda p: not direct_forecast(p) and '不规则' not in p['task'] and '插补' not in p['task'] and p['task']!='评测方法')]
    seen=set(); out=['# 逐篇文献笔记（68篇）','',
         '分类代码：P=散点，B=块，V=整变量，I=不规则；N=自然缺失；U=原文未核实。维度按本报告口径：低1—10、中11—100、高>100；“待核实”明确保留信息缺口。',
         '', '每条的“证据定位”指向本调研包内保存的全文文本或官方页面核验点。']
    for heading,pred in groups:
        rows=[p for p in PAPERS if p['id'] not in seen and pred(p)]
        if not rows: continue
        out += ['',f'## {heading}','']
        for p in rows:
            seen.add(p['id'])
            cite=f"DOI: `{p['doi']}`" if p['doi'] else 'DOI：未录入/非DOI来源'
            link=f"[原始来源]({p['source_url']})"
            if p['code_url']: link += f"；[代码]({p['code_url']})"
            out += [f"### {p['id']}｜{p['name']}（{p['year']}，{p['venue']}，{p['status']}）",'',
              f"**题名**：{p['title']}  ",f"**任务**：{p['task']}  ",
              f"**缺失模式**：{flat(p['missing_patterns'])}  ",f"**场景**：{flat(p['domains'])}  ",
              f"**维度**：{flat(p['dimension_tags'])}；{p['datasets_dimensions']}  ",
              f"**方法**：{p['method']}  ",f"**可用结论**：{p['contribution']}  ",
              f"**比较边界**：{p['limitation']}  ",
              f"**证据**：{p['evidence_level']}级；{p['evidence_locator']}  ",
              f"**书目**：{cite}；{link}", '']
    (ROOT/'逐篇文献笔记.md').write_text('\n'.join(out),encoding='utf-8')

def write_bib():
    rows=['% Generated 2026-09-26 from manually audited papers.json.',
          '% DOIs were checked using the installed literature-review skill; unavailable metadata is not invented.','']
    for p in PAPERS:
        typ = 'misc' if p['status'] in ['预印本','学生论坛/稿件'] or p['venue']=='arXiv' else ('article' if any(s in p['venue'] for s in ['Journal','Transactions','Systems','Knowledge','Neurocomputing','Expert','Applied','Physica','Future','Geophysical','Reports','Transport','Forecasting']) else 'inproceedings')
        key=clean_slug(f"{p['name']}_{p['year']}")
        fields=[f"  title = {{{esc_bib(p['title'])}}}",f"  author = {{{esc_bib(p['authors'])}}}",f"  year = {{{p['year']}}}",f"  howpublished = {{{esc_bib(p['venue'])}}}",f"  url = {{{p['source_url']}}}"]
        if p['doi']: fields.append(f"  doi = {{{p['doi']}}}")
        if p.get('volume'): fields.append(f"  volume = {{{p['volume']}}}")
        if p.get('issue'): fields.append(f"  number = {{{p['issue']}}}")
        pages=p.get('page') or p.get('article_number')
        if pages: fields.append(f"  pages = {{{pages}}}")
        if p['status']!='已发表': fields.append(f"  note = {{{esc_bib(p['status'])}; accessed 2026-09-26}}")
        rows.append('@'+typ+'{'+key+',\n'+',\n'.join(fields)+'\n}\n')
    (ROOT/'references.bib').write_text('\n'.join(rows),encoding='utf-8')

def svg_text(x,y,s,size=14,weight='normal',fill='#17202a',anchor='start'):
    return f'<text x="{x}" y="{y}" font-family="Noto Sans CJK SC,Arial" font-size="{size}" font-weight="{weight}" fill="{fill}" text-anchor="{anchor}">{html.escape(s)}</text>'
def write_figures():
    # Taxonomy schematic. Deliberately a source-native SVG rather than an AI-created research figure.
    elems=['<svg xmlns="http://www.w3.org/2000/svg" width="1360" height="690" viewBox="0 0 1360 690">',
           '<rect width="1360" height="690" fill="#fbfcfe"/>',svg_text(680,42,'缺失观测下时间序列预测：任务与证据边界',27,'bold',anchor='middle')]
    boxes=[(50,95,270,110,'历史散点 / 块缺失','预测未来全部或指定变量','BiTGraph · HD-TTS · S4M'),
           (375,95,270,110,'整变量历史缺失','预测缺失变量未来','GinAR · IBN'),
           (700,95,270,110,'训练未见节点','归纳预测新位置','UIGNN'),
           (1025,95,270,110,'异步 / 不规则观测','指定未来查询点','GraFITi · t-PatchGNN')]
    for x,y,w,h,a,b,c in boxes:
        elems += [f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="14" fill="#eaf3ff" stroke="#3b82c4" stroke-width="2"/>',svg_text(x+w/2,y+33,a,17,'bold',anchor='middle'),svg_text(x+w/2,y+61,b,14,anchor='middle'),svg_text(x+w/2,y+87,c,12,fill='#47627b',anchor='middle')]
    elems += [svg_text(680,270,'不要混合的三个目标',21,'bold',anchor='middle')]
    boxes2=[(70,305,360,125,'插补','同一历史范围内恢复空洞','指标：MAE/RMSE/CRPS（遮蔽点）','#fff7df','#c89018'),
            (500,305,360,125,'可用子集预测','只预测仍可观测变量','FDW/TRF；不能与全变量输出排名','#edf9ee','#4c9a59'),
            (930,305,360,125,'缺失变量/节点预测','输出未观测变量或新节点未来','GinAR/IBN/UIGNN；需外部先验','#fff0f1','#c85562')]
    for x,y,w,h,a,b,c,bg,stroke in boxes2:
        elems += [f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="14" fill="{bg}" stroke="{stroke}" stroke-width="2"/>',svg_text(x+w/2,y+34,a,18,'bold',anchor='middle'),svg_text(x+w/2,y+64,b,14,anchor='middle'),svg_text(x+w/2,y+93,c,13,fill='#465361',anchor='middle')]
    elems += [svg_text(680,500,'共同的实验约束',21,'bold',anchor='middle')]
    checks=['切分后再拟合归一化、图与插补器','人工mask嵌套在原本可观测条目中','历史双向可用，绝不跨预测起点','报告实际缺失率、块长度与目标真值mask','概率预测同时看校准、尖锐度与联合事件']
    for i,s in enumerate(checks):
        x=150+(i%3)*400; y=550+(i//3)*60
        elems += [f'<circle cx="{x-24}" cy="{y-5}" r="10" fill="#38a169"/>',svg_text(x-24,y-1,'✓',14,'bold','#fff','middle'),svg_text(x,y,s,14)]
    elems.append('</svg>')
    (FIG/'任务边界与评测框架.svg').write_text('\n'.join(elems),encoding='utf-8')

    pats=Counter(v for p in PAPERS for v in p['missing_patterns'])
    dims=Counter(v for p in PAPERS for v in p['dimension_tags'])
    status=Counter(p['status'] for p in PAPERS)
    width,height=1160,600; colors=['#3b82c4','#e59b38','#8b5bb8','#3ea47a','#cf5b6b','#778899']
    pieces=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}"><rect width="100%" height="100%" fill="#fbfcfe"/>',svg_text(width/2,42,'文献覆盖概览：多标签计数，不代表可直接横向比较',25,'bold',anchor='middle')]
    charts=[('缺失模式（可多选）',pats,70),('维度标签（可多选）',dims,405),('发表状态（单选）',status,755)]
    for title,data,start in charts:
        pieces += [svg_text(start+145,100,title,17,'bold',anchor='middle'),f'<line x1="{start}" y1="470" x2="{start+290}" y2="470" stroke="#637282"/>']
        ordered=data.most_common(); maxv=max(data.values())
        step=280/max(len(ordered),1)
        for i,(label,val) in enumerate(ordered):
            x=start+20+i*step; h=300*val/maxv; color=colors[i%len(colors)]
            pieces += [f'<rect x="{x:.1f}" y="{470-h:.1f}" width="{step-18:.1f}" height="{h:.1f}" rx="4" fill="{color}"/>',svg_text(x+(step-18)/2,460-h,str(val),15,'bold',anchor='middle'),svg_text(x+(step-18)/2,495,label,13,anchor='middle')]
    pieces.append(svg_text(width/2,555,'68篇记录；43篇已核对全文关键位置（A级），25篇为官方摘要/网页等B级证据。',16,anchor='middle'))
    pieces.append('</svg>')
    (FIG/'文献覆盖概览.svg').write_text('\n'.join(pieces),encoding='utf-8')

def write_html():
    records=[]
    for p in PAPERS:
        q={k:flat(p.get(k,'')) for k in ['id','name','title','year','venue','status','task','missing_patterns','domains','dimension_tags','datasets_dimensions','method','contribution','limitation','evidence_level','doi','authors','source_url','code_url']}
        records.append(q)
    j=json.dumps(records,ensure_ascii=False).replace('</','<\\/')
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>缺失预测文献库</title>
<style>body{font-family:"Noto Sans CJK SC",Arial,sans-serif;margin:28px;color:#17202a;background:#f8fafc}h1{margin-bottom:4px}.meta{color:#5c6975}.tools{display:flex;gap:10px;flex-wrap:wrap;margin:22px 0}.tools input,.tools select{padding:8px;border:1px solid #bcc8d4;border-radius:6px;font-size:14px}table{border-collapse:collapse;width:100%;background:white;font-size:13px}th{position:sticky;top:0;background:#eaf3ff}th,td{padding:9px;border:1px solid #d8e0e8;vertical-align:top;text-align:left}a{color:#1769aa}.tag{display:inline-block;background:#eef3f8;border-radius:12px;padding:2px 7px;margin:1px}.A{background:#dff5e6}.B{background:#fff4d6}.note{max-width:330px}</style>
<body><h1>缺失观测时间序列预测：可筛选文献库</h1><p class="meta">68篇记录｜检索截至2026-09-26｜多标签分类；“待核实”是保留的信息缺口。</p>
<div class="tools"><input id="q" placeholder="搜索题名、方法、数据集、结论" size="34"><select id="pattern"><option value="">所有缺失模式</option></select><select id="domain"><option value="">所有场景</option></select><select id="dim"><option value="">所有维度</option></select><select id="task"><option value="">所有任务</option></select><select id="status"><option value="">所有发表状态</option></select></div><p id="count"></p><table><thead><tr><th>文献</th><th>任务</th><th>缺失模式</th><th>场景 / 维度</th><th>方法</th><th>数据集与边界</th><th>证据</th></tr></thead><tbody id="rows"></tbody></table>
<script>const data='''+j+''';const fields=['missing_patterns','domains','dimension_tags','task','status'];const ids=['pattern','domain','dim','task','status'];fields.forEach((f,i)=>{let s=document.getElementById(ids[i]);[...new Set(data.flatMap(x=>x[f].split('；')))].sort().forEach(v=>{if(v){let o=document.createElement('option');o.value=v;o.textContent=v;s.append(o)}})});function tags(s){return s.split('；').map(x=>'<span class="tag">'+x+'</span>').join('')}function run(){let q=document.getElementById('q').value.toLowerCase();let fs=ids.map(x=>document.getElementById(x).value);let out=data.filter(x=>{let all=Object.values(x).join(' ').toLowerCase();return(!q||all.includes(q))&&fs.every((v,i)=>!v||x[fields[i]].includes(v))});document.getElementById('count').textContent='显示 '+out.length+' / '+data.length+' 篇';document.getElementById('rows').innerHTML=out.map(x=>`<tr><td><b>${x.id}｜${x.name}</b><br>${x.title}<br><small>${x.year} · ${x.venue} · ${x.status}</small><br><a href="${x.source_url}">原始来源</a>${x.code_url?' · <a href="'+x.code_url+'">代码</a>':''}</td><td>${x.task}</td><td>${tags(x.missing_patterns)}</td><td>${tags(x.domains)}<br>${tags(x.dimension_tags)}</td><td>${x.method}</td><td class="note">${x.datasets_dimensions}<br><br><b>结论：</b>${x.contribution}<br><b>边界：</b>${x.limitation}</td><td><span class="tag ${x.evidence_level}">${x.evidence_level}级</span><br>${x.doi?'DOI: '+x.doi:'无DOI/未录入'}</td></tr>`).join('')}ids.concat(['q']).forEach(x=>document.getElementById(x).oninput=run);run();</script></body></html>'''
    (ROOT/'文献分类浏览器.html').write_text(page,encoding='utf-8')

def write_method_notes():
    good=sum(p['evidence_level']=='A' for p in PAPERS); pubs=sum(p['status']=='已发表' for p in PAPERS)
    doi_ok=sum(1 for x in DOI_REPORT if x.get('resolver_ok'))
    doi_fail=len(DOI_REPORT)-doi_ok
    text=f'''# 检索方法、筛选与核验说明

## 范围

- 主题：历史观测存在缺失时的数值时间序列预测；包括散点、连续块、整变量/节点缺失和不规则采样。
- 期间：2021—2025为主，补充截至2026-09-26的新增工作；2018—2020的五篇基础论文单列。
- 纳入：有明确预测、插补后预测、概率预测、下游评测、专门基准或可支持任务定义的方法论文。
- 排除：仅异常检测、没有时间预测或插补关联的缺失数据论文、没有可核验书目信息的二手转述。项目内ICDiffAD因此未纳入。

## 检索与筛选

本包保留了检索日志、公开全文下载清单、转换文本和失败下载记录。使用的检索词围绕 `time series forecasting missing values`、`variable missing`、`block missing`、`irregular multivariate time series forecasting`、`traffic/air quality/wind power missing forecasting`、`spatiotemporal imputation forecasting` 等组合，并回溯官方会议、出版方、arXiv、OpenReview与作者机构仓库。来源覆盖至少三类数据库/出版体系：官方会议论文集、期刊出版方、预印本/开放评审平台。

筛选分为标题摘要相关性、发表状态/主来源核验、全文关键段或表格提取。共纳入 **{len(PAPERS)}** 篇：{pubs}篇已发表、{sum(p['status']=='预印本' for p in PAPERS)}篇预印本、1篇学生论坛/稿件；其中 {good}篇A类全文证据，{len(PAPERS)-good}篇B类官方摘要或网页证据。多标签计数不可相加为论文总数。

## 引文核验

对有DOI的记录调用 literature-review skill 提供的 `verify_citations.py`，经 DOI Handle 后再尝试Crossref；详见 `sources/citation_verification.json`。结果：{doi_ok}条解析成功，{doi_fail}条在当次网络请求中超时/元数据未返回。失败不等于DOI无效，故在总表中仍保留原文DOI，并将完整元数据缺口显式保留。`sources/crossref_metadata.json`保存第二次Crossref获取的结构化元数据，`references.bib`只写入已核实或原文直接可得字段。

## 限制

这是范围调研，不是PRISMA穷尽系统综述或meta-analysis。未访问付费墙全文，未进行双人独立筛选，也未复现实验代码。不同论文的数据版本、切分、归一化、预测跨度、真实缺失率和目标集合不同，因此本包不提供跨论文的“总SOTA”排名。维度阈值是整理规则，不是作者原始分类。
'''
    (ROOT/'检索方法与核验说明.md').write_text(text,encoding='utf-8')

def write_readme():
    text='''# 缺失观测时间序列预测：文献调研包

本包围绕“历史输入存在缺失时，如何做时间序列预测”整理了 **68篇** 文献，重点覆盖2021—2025及截至2026-09-26的更新。每篇都按缺失模式、数据场景、数据维度和模型方法标注；另明确任务目标、发表状态、证据等级、数据集维度、比较边界和来源链接。

## 建议阅读顺序

1. [综述正文.md](综述正文.md)：主题综合、方法脉络、研究空白和实验协议。
2. [文献分类浏览器.html](文献分类浏览器.html)：离线筛选68篇记录。
3. [文献分类总表.csv](文献分类总表.csv)：可在Excel中继续标注与排序。
4. [逐篇文献笔记.md](逐篇文献笔记.md)：每篇的原始来源、贡献、局限和核验位置。
5. [缺失观测时间序列预测_文献调研.pdf](缺失观测时间序列预测_文献调研.pdf)：可分享的精简阅读版。

## 关键数字

- 58篇已发表论文，9篇预印本，1篇学生论坛/稿件。
- 43篇全文关键段落或表格已核验（A类），25篇为官方摘要/网页等B类证据。
- 63篇为2021—2026记录；5篇2018—2020基础工作单列。
- 缺失模式、领域和维度均为多标签；不可把各列计数相加为论文数。

## 使用边界

本包是可追溯的范围调研，而非穷尽性系统综述或跨论文SOTA榜单。数据版本、预测跨度、目标集合和mask协议常不同，不能直接拼接原文指标。请以论文原始来源为最终引用依据；全文/摘要证据边界与引文核验详情见[检索方法与核验说明.md](检索方法与核验说明.md)。
'''
    (ROOT/'README.md').write_text(text,encoding='utf-8')

def wrap_canvas_text(c, text, x, y, width, font='STSong-Light', size=11, leading=16):
    # CID CJK glyphs are approximately one em wide.  A conservative width avoids
    # clipped right margins in the generated PDF.
    c.setFont(font,size); chars=max(18,int(width/(size*1.10)))
    for para in text.split('\n'):
        # Chinese has no whitespace between words, so it needs character-level wrapping.
        for line in textwrap.wrap(para,width=chars,break_long_words=True,replace_whitespace=False) or ['']:
            if y<55: c.showPage(); c.setFont(font,size); y=790
            c.drawString(x,y,line); y-=leading
        y-=4
    return y
def write_pdf():
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.cidfonts import UnicodeCIDFont
    from reportlab.pdfgen.canvas import Canvas
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.colors import HexColor
    pdfmetrics.registerFont(UnicodeCIDFont('STSong-Light'))
    path=ROOT/'缺失观测时间序列预测_文献调研.pdf'; c=Canvas(str(path),pagesize=A4); W,H=A4
    c.setTitle('缺失观测时间序列预测：文献调研')
    c.setFillColor(HexColor('#102a43')); c.rect(0,0,W,H,fill=1,stroke=0); c.setFillColor(HexColor('#ffffff'))
    c.setFont('STSong-Light',28); c.drawCentredString(W/2,H-170,'缺失观测场景下的时间序列预测')
    c.setFont('STSong-Light',17); c.drawCentredString(W/2,H-205,'文献调研、分类框架与实验建议')
    c.setFont('STSong-Light',12); c.drawCentredString(W/2,H-260,'检索截至 2026 年 9 月 26 日')
    c.setFont('STSong-Light',11); c.drawCentredString(W/2,H-292,'68篇记录｜63篇近五年及2026更新｜43篇全文关键位置核验')
    c.setFillColor(HexColor('#b9e3f7')); c.setFont('STSong-Light',10); c.drawCentredString(W/2,105,'范围调研，不做跨协议性能排行；预印本、插补与预测任务明确区分。')
    c.showPage(); c.setFillColor(HexColor('#17202a'))
    sections=[
      ('1. 如何使用这份调研','先用可筛选总表选定任务、缺失模式、数据领域和维度；随后阅读逐篇笔记的“比较边界”。本报告将插补、可用子集预测、全变量预测、未见节点预测和不规则预测分开，避免不相同的输出目标被放进同一排行榜。'),
      ('2. 四条主要结论','（1）缺失几何与缺失机制是不同概念：散点/块/变量描述掩码，MCAR/MAR/MNAR描述生成过程。\n（2）整变量缺失、可用子集预测和未见节点预测有不同的信息预算。\n（3）插补误差低不是下游预测更好的充分条件；应评价完整流水线。\n（4）同名数据集可能使用不同预处理与维度，报告中保留论文原始D。'),
      ('3. 方法地图','固定网格直接预测：BiTGraph、HD-TTS、S4M、Merlin、CoIFNet。\n整变量/节点：FDW、TRF、GinAR、IBN、UIGNN。\n不规则预测：CRU、GraFITi、t-PatchGNN、HyperIMTS、ReIMTS。\n概率生成：TSDiff、TACTiS、ProFITi、MOSES。\n插补支撑：GRIN、CSDI、SAITS、PriSTI、ImputeFormer、LSCD。'),
      ('4. 推荐实验最小集','规则网格：ETT7、Weather21、METR-LA207，分别测试散点、块、变量缺失。\n不规则：一个气候数据和一个临床数据，保留原始时间戳。\n基线：持久性/季节持久性、历史内简单插补＋同一预测器、直接缺失预测器、完整历史Oracle。\n共同约束：切分后拟合统计量；掩码只加在原本观测点；不跨预测起点；报告实际缺失率和目标可观测性。'),
      ('5. 证据与限制','A类表示已核对保存全文的关键段落或表格；B类为官方摘要或网页证据。58篇正式发表、9篇预印本和1篇学生论坛稿件。只将正式来源已确认的论文标为“已发表”。本调研没有访问付费全文、进行双人筛选或复现实验代码，因此适合作为选题与实验设计的可追溯起点。')]
    y=790
    for head,body in sections:
        c.setFillColor(HexColor('#175a8e'));c.setFont('STSong-Light',17);c.drawString(48,y,head);y-=28
        c.setFillColor(HexColor('#17202a'));y=wrap_canvas_text(c,body,55,y,W-110,size=11,leading=17);y-=15
    c.showPage(); c.setFillColor(HexColor('#17202a'));c.setFont('STSong-Light',19);c.drawString(48,795,'核心文献索引（按任务）')
    y=760
    pick=['BiTGraph2024','HDTTS2024','GinAR2024','S4M2025','Merlin2025','FDW2022','TRF2024','UIGNN2023','GraFITi2024','tPatchGNN2024','ReIMTS2026','CSDI2021','GRIN2022','SAITS2023','TaskEval2024','WindFCS2024','TACTiS2022','ProFITi2025','MOSES2026','AirQualityBench2026']
    for key in pick:
        p=next(x for x in PAPERS if x['key']==key)
        text=f"{p['id']}  {p['name']}（{p['year']}，{p['venue']}）\n任务：{p['task']}；模式：{flat(p['missing_patterns'])}；维度：{flat(p['dimension_tags'])}\n{p['contribution']}\n边界：{p['limitation']}"
        if y<120: c.showPage();c.setFillColor(HexColor('#17202a'));c.setFont('STSong-Light',19);c.drawString(48,795,'核心文献索引（续）');y=760
        c.setFillColor(HexColor('#175a8e'));c.setFont('STSong-Light',11);c.drawString(48,y,text.split('\n')[0]); y-=15;c.setFillColor(HexColor('#17202a'));y=wrap_canvas_text(c,'\n'.join(text.split('\n')[1:]),56,y,W-112,size=9,leading=13);y-=9
    c.showPage(); c.setFillColor(HexColor('#17202a'));c.setFont('STSong-Light',19);c.drawString(48,795,'随附文件与可复现性')
    body='综述正文.md：完整主题综合与研究问题。\n文献分类总表.csv：Excel可打开的68篇多字段总表。\n文献分类浏览器.html：离线筛选任务、缺失模式、领域、维度和发表状态。\n逐篇文献笔记.md：每篇的四轴标注、贡献、边界、来源与证据定位。\nreferences.bib：可导入的书目信息；检索方法与核验说明.md：检索范围、筛选、引文核验与限制。\n\n引用核验：对具DOI的条目运行了 skill 自带核验器；网络超时或Crossref没有返回的条目不伪造元数据。公开下载论文的哈希与失败记录保存于sources/download_manifest.json及download_supplement_manifest.json。\n\n建议引用时以论文原始来源或DOI为准；本PDF是研究工作底稿，不取代原文。'
    wrap_canvas_text(c,body,55,755,W-110,size=11,leading=18)
    c.save()

if __name__=='__main__':
    write_csv(); write_notes(); write_bib(); write_figures(); write_html(); write_method_notes(); write_readme(); write_pdf()
    print('generated CSV, notes, BibTeX, SVG, HTML, method notes, PDF')
