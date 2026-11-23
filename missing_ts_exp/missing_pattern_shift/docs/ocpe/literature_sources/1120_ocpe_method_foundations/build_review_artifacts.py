#!/usr/bin/env python3
"""Offline bibliography, evidence summary, schematic and reading-PDF builder."""
from pathlib import Path
import csv
import hashlib
import html
import json
import re
import shutil
import subprocess
import unicodedata
import xml.etree.ElementTree as ET
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.pdfgen import canvas
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, HRFlowable

ROOT = Path(__file__).resolve().parent
DATE = '2026-11-20'
REPORT = ROOT / '1120-文献调研-OCPE方法依据与相近工作.md'
FONT = 'STSong-Light'
BLUE = colors.HexColor('#315C82')
INK = colors.HexColor('#203348')
pdfmetrics.registerFont(UnicodeCIDFont(FONT))

def save(name, obj):
    (ROOT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')

def norm(value):
    return re.sub(r'[^a-z0-9]', '', value.lower())

def metadata(path):
    if not path.exists():
        return {}
    result = {}
    for tag in re.findall(r'<meta\s+([^>]+)>',path.read_text(errors='replace'),re.I):
        attrs = {k.lower():html.unescape(v) for k,_,v in re.findall(r'([\w:-]+)\s*=\s*([\"\'])(.*?)\2',tag)}
        name = attrs.get('name','')
        if name.startswith('citation_'):
            result.setdefault(name,[]).append(attrs.get('content',''))
    return result

def bibliography(refs):
    with (ROOT/'references.csv').open('w',encoding='utf-8-sig',newline='') as f:
        fields = list(dict.fromkeys(k for r in refs for k in r))
        writer = csv.DictWriter(f,fieldnames=fields)
        writer.writeheader();writer.writerows(refs)
    entries = []
    for r in refs:
        fields = {'title':'{'+r['title']+'}', 'author':r['authors'].replace('; ',' and '), 'year':str(r['year']), 'url':r['url']}
        kind = r['entry_type']
        if kind=='article': fields['journal'] = r['venue']
        elif kind=='inproceedings': fields['booktitle'] = r['venue']
        else: fields['howpublished'] = 'arXiv preprint'
        for k in ['doi','volume','pages']:
            if r.get(k): fields[k] = r[k]
        if r.get('issue'): fields['number'] = r['issue']
        if kind=='misc':
            fields['eprint'] = r['url'].rsplit('/',1)[-1]
            fields['archivePrefix'] = 'arXiv'
        fields['note'] = 'Checked '+DATE+'. '+r['version_note']
        entries.append('@'+kind+'{'+r['id']+',\n'+',\n'.join('  '+k+' = {'+v+'}' for k,v in fields.items())+'\n}\n')
    (ROOT/'references.bib').write_text('\n'.join(entries))

def verification(refs):
    raw_path = ROOT/'doi_verification_raw.json'
    raw = {r['id']:r for r in json.loads(raw_path.read_text())} if raw_path.exists() else {}
    records = []
    for r in refs:
        page = metadata(ROOT/'sources'/f"{r['id']}_landing.html")
        cpath = ROOT/'sources'/f"{r['id']}_crossref.json"
        crossref = json.loads(cpath.read_text()).get('message',{}) if cpath.exists() else {}
        titles = page.get('citation_title',[]) or crossref.get('title',[])
        if r['id']=='R07':
            page_text=(ROOT/'sources/R07_landing.html').read_text()
            titles=[html.unescape(re.sub('<[^>]+>','',m)).strip() for m in re.findall(r'<h1[^>]*>(.*?)</h1>',page_text,re.S)]
        title_match = any(norm(title)==norm(r['title']) for title in titles) if titles else None
        record = {'id':r['id'],'title':r['title'],'doi':r.get('doi',''),'primary_url':r['url'],
                  'title_candidates':titles,'title_match_machine':title_match,
                  'primary_citation_authors':page.get('citation_author',[]),
                  'crossref_authors':crossref.get('author',[]),
                  'recorded_year':r['year'],'version_note':r['version_note'],
                  'read_depth':r['read_depth'],'locator':r['locator'],
                  'method_evidence_scope':'targeted sections; not a full text/code/replication audit',
                  'doi_handle_check':raw.get(r['id']),
                  'year_check':'version-aware manual check; arXiv submission, online and issue years may differ'}
        if r['id']=='R06':
            record['version_resolution']={'crossref':'2021 early access, page 1-1',
                'final_issue':'2022, 44(9),4659-4673','primary_author_list':'https://wangx1024.github.io/_pages/publication.html',
                'local_pdf':'author early-access PDF; not final pagination','evidence_log':'sources/web_versions.json'}
        if r['id']=='R03':record['fulltext_limitation']='Publisher landing returned 403; ECB PDF TLS failed. Relevant WP parts verified through web tool. Final journal full text not obtained.'
        records.append(record)
    save('citation_verification.json',{'date':DATE,'method':'Skill CitationVerifier for four publisher DOIs; primary metadata/title checks and version-aware manual verification. arXiv DOIs are not asserted to be Crossref records.',
        'summary':{'research_references':len(refs),'machine_title_matches':sum(r['title_match_machine'] is True for r in records),
                   'publisher_doi_checks':len(raw),'publisher_doi_handles_resolved':sum(bool(r.get('handle_resolved')) for r in raw.values()),
                   'local_primary_pdfs':sum(bool(r.get('local_pdf')) and (ROOT/r['local_pdf']).is_file() for r in refs)},
        'results':records,'limitations':['A resolved DOI alone is not method verification.','Downloaded full text does not imply complete close reading.','No claim of independent peer review for preprints.']})

def search_summary():
    ns = {'a':'http://www.w3.org/2005/Atom','o':'http://a9.com/-/spec/opensearch/1.1/'}
    feed = ET.parse(ROOT/'sources/arxiv_search.txt').getroot()
    candidates = [{'title':e.findtext('a:title',default='',namespaces=ns).strip(),'url':e.findtext('a:id',default='',namespaces=ns),
                   'status':'included as R08' if 'Robust prediction under missingness shifts' in e.findtext('a:title',default='',namespaces=ns) else 'candidate; not fully assessed in this focused packet'} for e in feed.findall('a:entry',ns)]
    crossref = json.loads((ROOT/'sources/crossref_search.json').read_text())['message']
    logs=[]
    for p in sorted((ROOT/'sources').glob('web_*.json')):
        d=json.loads(p.read_text())
        logs.append({'path':str(p.relative_to(ROOT)),'args':d.get('args'),'note':'Raw tool-returned text preserved; unreported total hits remain unknown.'})
    if logs:
        for l in logs:
            if l['path'].endswith('web_search_01.json'):
                l['args']={'search_query':[
                    {'q':'"Gaussian posterior" "missing" "time series" latent uncertainty forecasting'},
                    {'q':'"missingness shifts" NeuMISE posterior'},
                    {'q':'"VaryST" "missing" Appendix'}],
                    'open':['https://arxiv.org/html/2610.03916v2','https://arxiv.org/html/2406.16484v1']}
    save('search_log.json',{'date':DATE,'review_type':'focused OCPE foundation and related-work scoping review; not exhaustive systematic review',
        'scope_file':'scope.md','selection':'purposeful reading set of 11 papers from prior cited foundations, discussed related work and this-round additions; not a PRISMA-derived final set',
        'channels':[
            {'source':'web.run academic discovery and primary-page reading','status':'used','raw_logs':logs,'total_hits':'not reported; no fabricated count'},
            {'source':'arXiv API','query':'all:"missingness shift"','requested_max_results':10,'returned_entries':len(candidates),
             'reported_total':feed.findtext('o:totalResults',namespaces=ns),'raw_path':'sources/arxiv_search.txt','candidates':candidates},
            {'source':'Crossref bibliographic search','query':'missingness shift prediction','requested_rows':10,'returned_items':len(crossref['items']),
             'reported_total':crossref['total-results'],'warning':'Fuzzy broad search; total is not a relevance-screened count.',
             'raw_path':'sources/crossref_search.json','candidate_titles':[v.get('title',[''])[0] for v in crossref['items']]},
            {'source':'DBLP API','query':'missingness shift','status':'failed: HTML bot-check page despite HTTP 200','raw_path':'sources/dblp_search.html'},
            {'source':'PMLR/NeurIPS/publishers/author PDFs','status':'primary text and bibliographic verification; not counted as independent exhaustive database searches'}],
        'dependencies':{'parallel_cli':'not installed; web.run fallback','ai_schematic':'OPENROUTER_API_KEY absent; deterministic vector diagram fallback'},
        'exclusions':[{'category':'nonacademic search hits, news, product pages','reason':'not primary research evidence'},
                      {'category':'other-route cross-domain attachment files','reason':'outside OCPE packet provenance; not copied'},
                      {'category':'additional relevant database candidates','reason':'kept as pending; not mislabeled irrelevant'}],
        'limitations':['No exhaustive search','No dual-reviewer screening','Not all official code or supplementary proofs audited','No reproduction of prior paper results']})

def novelty(refs):
    save('route_novelty_checks.json',{'date':DATE,'purpose':'OCPE-specific component correspondence and claim boundaries; separate from observation-normalized residual correction',
        'status':'complete novelty not established','definition':'static window latent; exact observed-prefix Gaussian mean/covariance; mean+log marginal variance features; S4; P-B source-target evaluation',
        'components':[
            {'component':'linear Gaussian low-rank posterior','prior_refs':['R01','R02','R03','R06'],'finding':'established statistical mechanism; not a new theorem'},
            {'component':'probabilistic latent inference and neural time-series prediction','prior_refs':['R04','R05','R09'],'finding':'established broad direction; exact architectures differ'},
            {'component':'observation-set-conditioned prediction','prior_refs':['R07','R08'],'finding':'established theoretical and neural modeling direction'},
            {'component':'missingness shifts','prior_refs':['R08'],'finding':'explicitly studied; assumptions matter'},
            {'component':'point/block training-test transfer','prior_refs':['R11'],'finding':'explicit experiment in Appendix C.3 Table12; protocol differs'},
            {'component':'complete OCPE implementation combination','prior_refs':[r['id'] for r in refs],'finding':'not established as identical to a listed method; this is not proof of uniqueness'}],
        'per_paper':[{'id':r['id'],'history_role':r['history_role'],'locator':r['locator'],'relation':r['ocpe_relation'],'shift_evidence':r['shift_evidence'],'boundary':r['boundary']} for r in refs],
        'open_checks':['Additional conditional Gaussian/set-aggregation/GP latent encoders','Covariance inputs versus ordinary mask-conditioned encodings','Official implementations and training augmentation parity','Common source-target protocol comparisons'],
        'not_authorized_experiments':'These are evidence gaps, not a newly approved experiment schedule.'})

def figure():
    p=ROOT/'ocpe_evidence_map.pdf'
    c=canvas.Canvas(str(p),pagesize=(900,520));c.setTitle('OCPE literature evidence map')
    c.setFont(FONT,21);c.setFillColor(INK);c.drawString(32,480,'OCPE：文献依据与当前方法的对应关系')
    c.setFont(FONT,11);c.drawString(32,457,'实线表示模块关系；文献相关性不表示完整组合新颖性或跨模式鲁棒性已经证明。')
    def box(x,y,w,h,title,lines,color='#E8EFF6'):
        c.setFillColor(colors.HexColor(color));c.setStrokeColor(BLUE);c.roundRect(x,y,w,h,8,fill=1,stroke=1)
        c.setFillColor(INK);c.setFont(FONT,13);c.drawString(x+12,y+h-24,title)
        c.setFont(FONT,10)
        for i,line in enumerate(lines):c.drawString(x+12,y+h-45-17*i,line)
    def arrow(x1,y1,x2,y2):
        c.setStrokeColor(BLUE);c.line(x1,y1,x2,y2);c.line(x2,y2,x2-6,y2+4);c.line(x2,y2,x2-6,y2-4)
    box(32,290,230,130,'实际观测前缀', ['观测值 + 观测位置','共享窗口潜变量 z','没有动态状态转移'])
    box(300,290,270,130,'高斯后验编码', ['均值 m + 对数边际方差','R01 PPCA / R02 线性高斯综述','统计公式是既有工具'])
    box(608,290,260,130,'S4 预测器', ['R10 S4：普通 DPLR 主干','输出未来预测值','不自动获得概率鲁棒性保证'])
    arrow(262,355,300,355);arrow(570,355,608,355)
    box(32,140,403,112,'相近方法：需要比较机制', ['R03 缺失因子模型 / R06 BTF-BTMF','R04 RKN / R05 CRU / R09 LIFTS','R07 NeuMiss：按观测集合改变预测映射'],'#F0F4F1')
    box(465,140,403,112,'问题场景：分别核对训练与测试', ['R08 NeuMISE：缺失机制变化','R11 VaryST：附录 C.3 表12跨缺失模式','场景相关不等于协议、模型或数值可直接等同'],'#FFF1DF')
    c.setFont(FONT,12);c.setFillColor(INK);c.drawString(32,95,'当前可确认：计算有统计基础；已有相近方法；已有缺失变化与跨模式实验。')
    c.drawString(32,71,'当前仍待论证：完整组合的新颖性、方差特征的独立作用、跨模式收益的适用边界。')
    c.setFont(FONT,9);c.drawString(32,30,'2026-11-20 | 可复现矢量示意图；不表示文献引用网络或性能比较结果。');c.save()
    if shutil.which('pdftoppm'):
        subprocess.run(['pdftoppm','-singlefile','-scale-to','1800','-png',str(p),str(ROOT/'ocpe_evidence_map')],check=True,capture_output=True)

def inline(value):
    # CID Chinese font lacks some Latin combining marks; preserve names in MD/JSON.
    value=''.join(ch for ch in unicodedata.normalize('NFKD',value) if not unicodedata.combining(ch))
    value=value.replace('↔',' <-> ').replace('→',' -> ').replace('Σ','Sigma').replace('μ','mu')
    value=html.escape(value)
    value=re.sub(r'\[([^\]]+)\]\(([^)]+)\)',lambda m:'<link href="'+m.group(2)+'" color="#245B87">'+m.group(1)+'</link>',value)
    value=re.sub(r'\*\*(.+?)\*\*',r'<b>\1</b>',value)
    return value.replace('`','')

def reading_pdf(refs):
    body=ParagraphStyle('body',fontName=FONT,fontSize=9.5,leading=15,wordWrap='CJK',spaceAfter=7,textColor=INK)
    table_style=ParagraphStyle('cell',parent=body,fontSize=8,leading=12,spaceAfter=0)
    heading=ParagraphStyle('heading',parent=body,fontSize=14,leading=21,spaceBefore=12,spaceAfter=8,keepWithNext=True,textColor=BLUE)
    title=ParagraphStyle('title',parent=heading,fontSize=20,leading=28)
    story=[];lines=REPORT.read_text().splitlines();i=0
    while i<len(lines):
        line=lines[i].strip();i+=1
        if not line:continue
        if line==r'\[':
            while i<len(lines) and lines[i].strip()!=r'\]':i+=1
            i+=1;story.append(Paragraph('公式请参阅 Markdown 原稿（阅读 PDF 不排印 LaTeX）。',body));continue
        if line.startswith('!['):
            match=re.search(r'\]\(([^)]+)\)',line)
            if match and (ROOT/match.group(1)).exists():story.append(Image(str(ROOT/match.group(1)),width=487,height=281.4))
            continue
        if line.startswith('|'):
            rows=[];i-=1
            while i<len(lines) and lines[i].strip().startswith('|'):
                parts=[v.strip() for v in lines[i].strip().strip('|').split('|')];i+=1
                if not all(re.fullmatch(r'[-:\s]+',v) for v in parts):rows.append([Paragraph(inline(v),table_style) for v in parts])
            if rows:
                count=len(rows[0]);weights={3:[.20,.37,.43],4:[.17,.25,.29,.29]}.get(count,[1/count]*count)
                table=Table(rows,colWidths=[487*w for w in weights],repeatRows=1,hAlign='LEFT')
                table.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#E6EEF5')),('VALIGN',(0,0),(-1,-1),'TOP'),('GRID',(0,0),(-1,-1),.3,colors.HexColor('#D5DEE6')),('LEFTPADDING',(0,0),(-1,-1),5),('RIGHTPADDING',(0,0),(-1,-1),5),('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5)]))
                story.extend([table,Spacer(1,8)])
            continue
        if line.startswith('# '):story.append(Paragraph(inline(line[2:]),title))
        elif line.startswith('## '):story.append(Paragraph(inline(line[3:]),heading))
        elif line.startswith('### '):story.append(Paragraph(inline(line[4:]),heading))
        else:story.append(Paragraph(inline(line),body))
    story.append(Paragraph('参考文献',heading))
    for r in refs:story.append(Paragraph(inline(r['id']+' '+r['authors']+' ('+str(r['year'])+'). '+r['title']+' '+r['venue']+'. [一手入口]('+r['url']+')'),body))
    def page(c,doc):
        c.setFont(FONT,8);c.setFillColor(INK);c.drawString(42,24,'OCPE 定向文献调研 | 2026-11-20');c.drawRightString(553,24,str(doc.page))
    doc=SimpleDocTemplate(str(REPORT.with_suffix('.pdf')),pagesize=(595,842),rightMargin=54,leftMargin=54,topMargin=42,bottomMargin=42,title='OCPE 方法依据与相近工作',author='OCPE research notes')
    doc.build(story,onFirstPage=page,onLaterPages=page)

def main():
    refs=json.loads((ROOT/'references.json').read_text());bibliography(refs);verification(refs);search_summary();novelty(refs);figure();reading_pdf(refs)
    print('Built',len(refs),'reference entries, audit summaries, diagram and reading PDF.')

if __name__=='__main__':main()
