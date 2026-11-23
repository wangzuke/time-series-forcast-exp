#!/usr/bin/env python3
"""Read-only CPU audit, paired aggregation and figures for the frozen fusion suite."""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import time
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
SUITE='ocpe_fusion_uniform_r50_l96_h96_1120'
OUT=ROOT/'summary/ocpe_fusion_core_1121'
FIG=ROOT/'figures/ocpe_fusion_core_1121'
METRICS=('mse_standardized','mae_standardized','mse_raw','mae_raw')
CELLS=(('P','P'),('P','Bu'),('Bu','P'),('Bu','Bu'))
LABELS={'F0':'常规历史','F1':'对角归一化','F2':'后验均值','F3':'均值＋方差'}

def read(path):return json.loads(path.read_text())
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()
def write_json(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
def write_csv(name,rows):
    if not rows:return
    with (OUT/name).open('w',encoding='utf-8-sig',newline='') as f:
        fields=list(dict.fromkeys(k for r in rows for k in r))
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)

def finite_tensors(value):
    if isinstance(value,torch.Tensor):return bool(torch.isfinite(value).all())
    if isinstance(value,dict):return all(finite_tensors(v) for v in value.values())
    if isinstance(value,(list,tuple)):return all(finite_tensors(v) for v in value)
    return True

def analyze():
    torch.set_num_threads(2);OUT.mkdir(parents=True,exist_ok=True);FIG.mkdir(parents=True,exist_ok=True)
    manifest=read(ROOT/'configs'/f'{SUITE}.json')
    current_sources={k:hashlib.sha256((ROOT/k).read_bytes()).hexdigest() for k in manifest['source']}
    assert current_sources==manifest['source'],'Current frozen source differs from executed manifest'
    groups,epoch_groups,test_groups={},{},{}
    checkpoints,rows,training,quality,mask_rows=[],[],[],{},[]
    max_error=0.;job_checks=[]
    def match(mapping,key,value,owner):
        mapping.setdefault(key,[]).append((value,owner))
        assert mapping[key][0][0]==value,(key,mapping[key][0][1],owner)
    for job in manifest['jobs']:
        p=ROOT/'results'/SUITE/job['id']
        protocol=read(p/'protocol.json');train=read(p/'training_complete.json');progress=read(p/'progress.json')
        identity={k:v for k,v in protocol.items() if k not in ('runtime','fingerprint')}
        fingerprint=digest(identity)
        assert fingerprint==protocol['fingerprint']==train['fingerprint']==progress['fingerprint']
        assert protocol['source']==current_sources
        assert train['status']==progress['status']=='complete'
        assert not (p/'failure.json').exists()
        data=protocol['data_quality'];quality[job['dataset']]=data
        assert hashlib.sha256(Path(data['raw_file']).read_bytes()).hexdigest()==data['data_sha256']
        key=(job['dataset'],job['family'],job['seed'])
        match(groups,key,{**{k:protocol[k] for k in ('train_mask_epoch1','train_mask_epoch2','validation_mask','validation_effective_mask')},
              **{k:data[k] for k in ('data_sha256','validity_sha256','mean_sha256','std_sha256')}},job['id'])
        history=train['history'];best=min(history,key=lambda x:x['val_mse'])
        for h in history:
            match(epoch_groups,(*key,h['epoch']),h['train']['effective_mask_sequence_sha256'],job['id'])
            assert all(math.isfinite(float(h['train'][k])) for k in ('mse','observation_nll','seconds'))
            assert math.isfinite(h['val_mse'])
        # Both inference and resume files are trusted artifacts produced in this workspace.
        for kind in ('best','last'):
            cp=ROOT/'checkpoints'/SUITE/f"{job['id']}.{kind}.pt"
            saved=torch.load(cp,map_location='cpu',weights_only=False)
            assert saved['fingerprint']==fingerprint and finite_tensors(saved['model'])
            if kind=='last':assert finite_tensors(saved['optimizer'])
            state=saved['model'];channels=protocol['model']['channels'];rank=protocol['model']['rank']
            assert tuple(state['input_projection.weight'].shape)==(64,2*channels)
            if job['variant']!='F0':
                assert tuple(state['observation.loadings'].shape)==(96,channels,rank)
                assert tuple(state['encoding_projection.weight'].shape)==(64,2*rank)
            if kind=='best':assert saved['epoch']==best['epoch']
            checkpoints.append({'job':job['id'],'kind':kind,'bytes':cp.stat().st_size,'finite':True})
            del saved,state
        times=[h['train']['seconds'] for h in history]
        training.append({**job,'epochs':len(history),'best_epoch':best['epoch'],'best_val_mse':best['val_mse'],
                         'training_seconds':train['training_seconds_this_session'],'median_epoch_seconds':statistics.median(times),
                         'median_samples_per_second':statistics.median(h['train']['samples']/h['train']['seconds'] for h in history),
                         'allocated_peak_mib':train['allocated_peak_mib'],'reserved_peak_mib':train['reserved_peak_mib'],
                         'parameters':protocol['parameter_counts']['total'],'stop_reason':train['stop_reason']})
        for family in ('P','Bu'):
            for mask_seed in (9101,9102):
                result=read(p/f'test_{family}_m{mask_seed}.json')
                assert result['status']=='complete' and result['fingerprint']==fingerprint
                assert all(math.isfinite(result[k]) for k in METRICS)
                assert result['best_epoch']==best['epoch']
                with np.load(p/f'test_{family}_m{mask_seed}.npz') as datafile:
                    index=datafile['index'];sse=datafile['window_sse'];count=datafile['window_valid_count']
                    assert np.array_equal(index,np.arange(result['n_windows']))
                    assert np.isfinite(sse).all() and (count>0).all()
                    assert int(count.astype(np.float64).sum())==result['n_target_elements']
                    recomputed=float(sse.astype(np.float64).sum()/count.astype(np.float64).sum())
                    error=abs(recomputed-result['mse_standardized']);max_error=max(max_error,error)
                    assert error<max(1e-6,result['mse_standardized']*1e-5)
                match(test_groups,(job['dataset'],family,mask_seed),result['effective_mask_sequence_sha256'],job['id'])
                rows.append({**job,'test_family':family,'mask_seed':mask_seed,**{k:result[k] for k in METRICS},
                             'mse_by_channel':result['mse_by_channel'],'mse_by_horizon':result['mse_by_horizon'],
                             'n_windows':result['n_windows'],'target_elements':result['n_target_elements'],
                             'evaluation_seconds':result['seconds']})
                a=result['mask_audit']
                assert a['artificial_missing_per_channel_min']==a['artificial_missing_per_channel_max']==48
                if family=='Bu':assert a['missing_run_max']==24
                mask_rows.append({'job':job['id'],'dataset':job['dataset'],'test_family':family,'mask_seed':mask_seed,
                                  'sha256':result['effective_mask_sequence_sha256'],'n_windows':result['n_windows'],
                                  'missing_run_max':a['missing_run_max']})
        job_checks.append({'job':job['id'],'status':'passed'})
    assert len(rows)==128 and len(checkpoints)==64 and len(job_checks)==32
    paired=[]
    for ds in ('Weather','Electricity'):
        for variant in LABELS:
            for seed in (2024,2025):
                for source,target in CELLS:
                    block=[r for r in rows if (r['dataset'],r['variant'],r['seed'],r['family'],r['test_family'])==(ds,variant,seed,source,target)]
                    assert len(block)==2
                    paired.append({'dataset':ds,'variant':variant,'seed':seed,'source':source,'target':target,
                                   **{k:statistics.mean(r[k] for r in block) for k in METRICS},
                                   'mse_by_channel':np.mean([r['mse_by_channel'] for r in block],0).tolist(),
                                   'mse_by_horizon':np.mean([r['mse_by_horizon'] for r in block],0).tolist()})
    def selected(ds,v,source,target):return [r for r in paired if (r['dataset'],r['variant'],r['source'],r['target'])==(ds,v,source,target)]
    cells=[]
    for ds in ('Weather','Electricity'):
        for variant in LABELS:
            for source,target in CELLS:
                block=selected(ds,variant,source,target)
                cell={'dataset':ds,'variant':variant,'source':source,'target':target}
                for k in METRICS:
                    cell[k]=statistics.mean(r[k] for r in block);cell[k+'_seed_sd']=statistics.stdev(r[k] for r in block)
                cells.append(cell)
    def cell(ds,v,s,t):return next(c for c in cells if (c['dataset'],c['variant'],c['source'],c['target'])==(ds,v,s,t))
    comparisons=[]
    pairs=[('F1','F0'),('F2','F0'),('F3','F0'),('F2','F1'),('F3','F1'),('F3','F2')]
    for ds in ('Weather','Electricity'):
        for variant,reference in pairs:
            for source,target in CELLS:
                a,b=cell(ds,variant,source,target),cell(ds,reference,source,target)
                paired_a,paired_b=selected(ds,variant,source,target),selected(ds,reference,source,target)
                signs=[]
                for seed in (2024,2025):
                    av=next(r for r in paired_a if r['seed']==seed);bv=next(r for r in paired_b if r['seed']==seed)
                    signs.append(100*(av['mse_standardized']/bv['mse_standardized']-1))
                ch_a=np.mean([r['mse_by_channel'] for r in paired_a],0);ch_b=np.mean([r['mse_by_channel'] for r in paired_b],0)
                h_a=np.mean([r['mse_by_horizon'] for r in paired_a],0);h_b=np.mean([r['mse_by_horizon'] for r in paired_b],0)
                comparisons.append({'dataset':ds,'variant':variant,'reference':reference,'source':source,'target':target,
                    **{k+'_pct':100*(a[k]/b[k]-1) for k in METRICS},'seed2024_pct':signs[0],'seed2025_pct':signs[1],
                    'channels_improved':int((ch_a<ch_b).sum()),'channels_total':len(ch_a),
                    'horizon1_24_pct':100*(h_a[:24].mean()/h_b[:24].mean()-1),
                    'horizon25_96_pct':100*(h_a[24:].mean()/h_b[24:].mean()-1)})
    costs=[]
    for ds in ('Weather','Electricity'):
        for variant in LABELS:
            for seed in (2024,2025):
                def value(s,t):return next(r['mse_standardized'] for r in paired if (r['dataset'],r['variant'],r['seed'],r['source'],r['target'])==(ds,variant,seed,s,t))
                costs.append({'dataset':ds,'variant':variant,'seed':seed,
                    'deploy_P_to_Bu_pct':100*(value('P','Bu')/value('P','P')-1),
                    'deploy_Bu_to_P_pct':100*(value('Bu','P')/value('Bu','Bu')-1),
                    'fixed_Bu_mismatch_pct':100*(value('P','Bu')/value('Bu','Bu')-1),
                    'fixed_P_mismatch_pct':100*(value('Bu','P')/value('P','P')-1)})
    audit={'status':'passed','checked_jobs':len(job_checks),'checked_evaluations':len(rows),'checked_checkpoints':len(checkpoints),
           'protocol_groups':len(groups),'full_test_mask_groups':len(test_groups),'test_records_per_group':sorted({len(v) for v in test_groups.values()}),
           'training_epoch_groups':len(epoch_groups),'training_epoch_groups_with_two_or_more_versions':sum(len(v)>1 for v in epoch_groups.values()),
           'training_epoch_groups_with_four_versions':sum(len(v)==4 for v in epoch_groups.values()),
           'mask_conflicts':0,'max_window_metric_recompute_absolute_error':max_error,
           'source_fingerprint_match':True,'raw_data_hash_match':True,'checkpoint_finite':True,
           'limitations':'CPU artifact audit; no new inference. NPZ recomputation checks standardized MSE, not raw MSE.'}
    result={'suite':SUITE,'audit':audit,'cells':cells,'paired_seed_cells':paired,'comparisons':comparisons,
            'costs':costs,'training':training,'data_quality':quality}
    write_json(OUT/'analysis.json',result);write_json(OUT/'audit.json',audit)
    write_csv('cells.csv',cells);write_csv('seed_cells.csv',[{k:v for k,v in r.items() if not isinstance(v,list)} for r in paired])
    write_csv('comparisons.csv',comparisons);write_csv('mismatch_costs.csv',costs);write_csv('training.csv',training)
    write_csv('mask_audits.csv',mask_rows);write_csv('checkpoints.csv',checkpoints)
    figure(result)
    return result

def figure(result):
    fig,axes=plt.subplots(2,2,figsize=(12,7.5),layout='constrained')
    colors={'F1':'#3575A6','F2':'#D18536','F3':'#3E8D73'};labels={'F1':'Diagonal','F2':'Posterior mean','F3':'Mean + variance'}
    x=np.arange(4);width=.23
    for i,ds in enumerate(('Weather','Electricity')):
        for j,reference in enumerate(('F0','F1')):
            ax=axes[i,j];versions=('F1','F2','F3') if reference=='F0' else ('F2','F3')
            for k,variant in enumerate(versions):
                rows=[next(r for r in result['comparisons'] if (r['dataset'],r['variant'],r['reference'],r['source'],r['target'])==(ds,variant,reference,s,t)) for s,t in CELLS]
                positions=x+(k-(len(versions)-1)/2)*width
                ax.bar(positions,[r['mse_standardized_pct'] for r in rows],width*.92,color=colors[variant],label=labels[variant],alpha=.85)
                for shift,seed in [(-.025,2024),(.025,2025)]:
                    ax.scatter(positions+shift,[r[f'seed{seed}_pct'] for r in rows],s=15,c='#1D2531',marker='o' if seed==2024 else 'x',zorder=3)
            ax.axhline(0,c='#333333',lw=.8);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
            ax.set_xticks(x,['P -> P','P -> Bu','Bu -> P','Bu -> Bu']);ax.set_ylabel('MSE change (%) / lower is better')
            ax.set_title(ds+' | relative to '+('history baseline' if reference=='F0' else 'diagonal encoder'))
            ax.legend(fontsize=8,loc='best')
    fig.suptitle('OCPE fusion: cell means and paired training-seed changes',fontsize=14)
    for ext in ('png','pdf','svg'):fig.savefig(FIG/f'relative_mse.{ext}',dpi=200)
    plt.close(fig)

if __name__=='__main__':
    r=analyze();print(json.dumps({'audit':r['audit'],'summary':str(OUT),'figures':str(FIG)},ensure_ascii=False))
