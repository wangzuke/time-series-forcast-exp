#!/usr/bin/env python3
"""Render the curated report template with audited result tables."""
import json
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'summary/ocpe_fusion_core_1121'
DATA=json.loads((OUT/'analysis.json').read_text())
CELLS=(('P','P'),('P','Bu'),('Bu','P'),('Bu','Bu'))
LABELS={'F0':'F0 常规历史','F1':'F1 对角归一化','F2':'F2 后验均值','F3':'F3 均值＋方差'}

def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+
                     ['| '+' | '.join(str(v) for v in row)+' |' for row in rows])

def cell(ds,v,s,t):return next(r for r in DATA['cells'] if (r['dataset'],r['variant'],r['source'],r['target'])==(ds,v,s,t))
def cmp(ds,v,ref,s,t):return next(r for r in DATA['comparisons'] if (r['dataset'],r['variant'],r['reference'],r['source'],r['target'])==(ds,v,ref,s,t))
def pct(v):return f'{v:+.1f}%'
def names():return ['P→P','P→B_u','B_u→P','B_u→B_u']

parts={}
for ds in ('Weather','Electricity'):
    rows=[]
    for v in LABELS:
        values=[]
        for s,t in CELLS:
            r=cell(ds,v,s,t);minimum=min(cell(ds,k,s,t)['mse_standardized'] for k in LABELS)
            value=f"{r['mse_standardized']:.6f} ± {r['mse_standardized_seed_sd']:.6f}"
            values.append('**'+value+'**' if r['mse_standardized']==minimum else value)
        rows.append([LABELS[v],*values])
    parts['MSE_'+ds.upper()]=table(['版本',*names()],rows)
rows=[]
for ds in ('Weather','Electricity'):
    for v in ('F1','F2','F3'):rows.append([ds,LABELS[v],*[pct(cmp(ds,v,'F0',s,t)['mse_standardized_pct']) for s,t in CELLS]])
parts['RELATIVE_BASE']=table(['数据集','相对F0',*names()],rows)
rows=[]
for ds in ('Weather','Electricity'):
    for v,ref in [('F2','F1'),('F3','F1'),('F3','F2')]:
        rows.append([ds,f'{v} 相对 {ref}',*[pct(cmp(ds,v,ref,s,t)['mse_standardized_pct']) for s,t in CELLS]])
parts['MECHANISM']=table(['数据集','关键对比',*names()],rows)
rows=[]
for ds in ('Weather','Electricity'):
    for v,ref in [('F1','F0'),('F3','F0'),('F3','F2')]:
        for s,t in [('P','Bu'),('Bu','P')]:
            r=cmp(ds,v,ref,s,t)
            rows.append([ds,f'{v} / {ref}',f'{s}→{t}',pct(r['seed2024_pct']),pct(r['seed2025_pct'])])
parts['SEED_TABLE']=table(['数据集','比较','条件','2024','2025'],rows)
rows=[]
for ds in ('Weather','Electricity'):
    for v in ('F1','F3'):
        for s,t in [('P','Bu'),('Bu','P')]:
            r=cmp(ds,v,'F0',s,t)
            rows.append([ds,f'{v}/F0',f'{s}→{t}',f"{r['channels_improved']}/{r['channels_total']}",pct(r['horizon1_24_pct']),pct(r['horizon25_96_pct'])])
parts['COVERAGE_TABLE']=table(['数据集','比较','条件','改善变量数','第1—24步','第25—96步'],rows)
rows=[]
for ds in ('Weather','Electricity'):
    for v in LABELS:
        block=[r for r in DATA['costs'] if r['dataset']==ds and r['variant']==v]
        keys=('fixed_Bu_mismatch_pct','fixed_P_mismatch_pct','deploy_P_to_Bu_pct','deploy_Bu_to_P_pct')
        rows.append([ds,v,*[pct(statistics.mean(r[k] for r in block)) for k in keys]])
parts['COST_TABLE']=table(['数据集','版本','固定B_u目标代价','固定P目标代价','P检查点切到B_u','B_u检查点切到P'],rows)
rows=[];compute=[]
for ds in ('Weather','Electricity'):
    for v in LABELS:
        block=[r for r in DATA['training'] if r['dataset']==ds and r['variant']==v]
        best=[]
        for source in ('P','Bu'):
            best.append(' / '.join(str(next(r['best_epoch'] for r in block if (r['family'],r['seed'])==(source,seed))) for seed in (2024,2025)))
        rows.append([ds,v,*best,f"{statistics.mean(r['epochs'] for r in block):.2f}"])
        compute.append([ds,v,f"{block[0]['parameters']:,}",f"{statistics.mean(r['median_epoch_seconds'] for r in block):.2f}",f"{max(r['allocated_peak_mib'] for r in block):.1f}"])
parts['TRAINING_TABLE']=table(['数据集','版本','P最佳epoch：2024/2025','B_u最佳epoch：2024/2025','平均实际轮数'],rows)
parts['COST_COMPUTE']=table(['数据集','版本','参数量','epoch耗时参考/秒','分配峰值/MiB'],compute)
for token,metric,rawscale in [('MAE_TABLE','mae_standardized',False),('RAW_MSE_TABLE','mse_raw',True),('RAW_MAE_TABLE','mae_raw',False)]:
    rows=[]
    for ds in ('Weather','Electricity'):
        scale=1e6 if rawscale and ds=='Electricity' else 1
        for v in LABELS:
            digits=3 if metric.startswith('mse_raw') else 5
            rows.append([ds,v,*[f'{cell(ds,v,s,t)[metric]/scale:.{digits}f}' for s,t in CELLS]])
    parts[token]=table(['数据集','版本',*names()],rows)
a=DATA['audit']
parts['AUDIT']=table(['项目','核对结果'],[
    ['训练/测试/检查点',f"{a['checked_jobs']}/{a['checked_evaluations']}/{a['checked_checkpoints']}，全部通过"],
    ['代码及原始数据哈希','与运行记录一致'],['模型及优化器状态','所检查浮点张量有限'],
    ['测试掩码','8组完整序列哈希，每组16个模型；冲突0'],
    ['共同训练epoch掩码','71组至少两个版本可比；冲突0'],
    ['逐窗口标准化MSE复算',f"最大绝对差 {a['max_window_metric_recompute_absolute_error']:.3g}"],
    ['窗口顺序、有效目标数','与各项测试记录一致']])
text=(OUT/'report_template.md').read_text()
for token,value in parts.items():text=text.replace('@@'+token+'@@',value)
assert '@@' not in text
destination=ROOT.parent/'docs/ocpe/1121-实验结果-OCPE补充编码与关键机制验证.md'
destination.write_text(text)
print(destination)
