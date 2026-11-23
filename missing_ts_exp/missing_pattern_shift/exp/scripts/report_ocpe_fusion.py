#!/usr/bin/env python3
"""Live summary and cross-model mask consistency audit; no inference."""
import csv
import json
from pathlib import Path
import time
from protocol import EXP_ROOT, atomic_json_dump
from ocpe_fusion_common import SUITE, LABELS


def read(path):
    try:return json.loads(path.read_text())
    except (FileNotFoundError,json.JSONDecodeError):return {}


def summarize(suite=SUITE):
    manifest=read(EXP_ROOT/'configs'/f'{suite}.json')
    rows,protocol_groups,train_groups,test_groups,conflicts=[],{},{},{},[]
    def match(group,key,value,owner):
        if key in group and group[key][0]!=value:
            conflicts.append({'key':str(key),'first':group[key][1],'second':owner})
        else:group[key]=(value,owner)
    completed=[]
    for job in manifest.get('jobs',[]):
        directory=EXP_ROOT/'results'/suite/job['id']
        p=read(directory/'protocol.json')
        if p:
            signature={k:p[k] for k in ('train_mask_epoch1','train_mask_epoch2','validation_mask','validation_effective_mask')}
            signature.update({k:p['data_quality'][k] for k in ('data_sha256','validity_sha256','mean_sha256','std_sha256')})
            match(protocol_groups,(job['dataset'],job['family'],job['seed']),signature,job['id'])
        training=read(directory/'training_complete.json')
        for epoch in training.get('history',[]):
            match(train_groups,(job['dataset'],job['family'],job['seed'],epoch['epoch']),
                  epoch['train']['effective_mask_sequence_sha256'],job['id'])
        all_tests=True
        for family in ('P','Bu'):
            for seed in (9101,9102):
                r=read(directory/f'test_{family}_m{seed}.json')
                if r.get('status')!='complete':all_tests=False;continue
                match(test_groups,(job['dataset'],family,seed),r['effective_mask_sequence_sha256'],job['id'])
                rows.append({**job,'test_family':family,'test_mask_seed':seed,
                             **{k:r[k] for k in ('mse_standardized','mae_standardized','mse_raw','mae_raw','best_epoch')},
                             'mask_sha256':r['effective_mask_sequence_sha256']})
        if all_tests and read(directory/'progress.json').get('status')=='complete':completed.append(job['id'])
    root=EXP_ROOT/'summary'/suite;root.mkdir(parents=True,exist_ok=True)
    payload={'updated_at':time.time(),'completed_jobs':completed,'training_jobs':len(manifest.get('jobs',[])),
             'evaluations':len(rows),'rows':rows,'mask_consistency':{'status':'failed' if conflicts else 'passed_so_far',
             'protocol_groups':len(protocol_groups),'training_epoch_groups':len(train_groups),
             'full_test_mask_groups':len(test_groups),'conflicts':conflicts},
             'note':'Live partial results; two test masks are repeated evaluation, not independent training.'}
    atomic_json_dump(payload,root/'results.json')
    if rows:
        with (root/'cells.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    text=['# OCPE 融合实验运行汇总','',f"正式训练完成：{len(completed)}/{payload['training_jobs']}；测试完成：{len(rows)}/128。",'',
          f"掩码一致性：{payload['mask_consistency']['status']}。这里只比较已存在的产物，不代表未完成任务已通过。",'',
          '| 数据集 | 版本 | 训练来源 | 测试目标 | 完成项/4 | 标准化MSE均值 |','|---|---|---|---|---:|---:|']
    for dataset in ('Weather','Electricity'):
        for variant in LABELS:
            for source in ('P','Bu'):
                for target in ('P','Bu'):
                    values=[r['mse_standardized'] for r in rows if (r['dataset'],r['variant'],r['family'],r['test_family'])==(dataset,variant,source,target)]
                    if values:text.append(f"| {dataset} | {LABELS[variant]} | {source} | {target} | {len(values)}/4 | {sum(values)/len(values):.6f} |")
    temporary=root/'report.md.tmp';temporary.write_text('\n'.join(text)+'\n');temporary.replace(root/'report.md')
    return payload


if __name__=='__main__':
    result=summarize();print(json.dumps({k:result[k] for k in ('training_jobs','evaluations','mask_consistency')}))
