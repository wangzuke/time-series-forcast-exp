#!/usr/bin/env python3
"""Targeted formula, data, mask, gradient/CUDA checks and timing."""
from __future__ import annotations
import argparse
import copy
import json
import time
import os
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from protocol import EXP_ROOT, atomic_json_dump
from posterior_s4_common import ExperimentDataset
from posterior_s4 import GaussianObservation, PosteriorS4, s4_module
from ocpe_fusion_common import (SUITE, FusionDataset, FusionS4, ConditionalEncoder, ModelConfig, VARIANTS,
    clean_dataset, mask_fingerprint, mask_audit, source_fingerprints, seed_all, move_batch, memory_stats, runtime_metadata, make_optimizer)


def numerical(device):
    cfg=ModelConfig(channels=3,seq_len=6,rank=2)
    ref=GaussianObservation(cfg).double()
    enc=ConditionalEncoder(cfg).double();enc.load_state_dict(ref.state_dict())
    x=torch.randn(4,6,3,dtype=torch.float64)
    mask=(torch.rand_like(x)>.4).double();mask[0]=0;mask[1,:2]=0
    expected,nll,details=ref(x,mask,return_details=True)
    errors={}
    for mode in ('F1','F2','F3'):
        features,value=enc(x,mask,mode=mode)
        target=(details['information']/details['precision'].diagonal(dim1=-2,dim2=-1) if mode=='F1' else details['mean'])
        target=torch.cat((target,expected[:,:,2:] if mode=='F3' else torch.zeros_like(target)),-1)
        errors[mode]=max(float((features-target).abs().max()),float((value-nll).abs().max()))
        assert errors[mode]<1e-10
        corrupted=torch.where(mask.bool(),x,float('nan'))
        f,n=enc(corrupted,mask,mode=mode)
        assert torch.equal(f,features) and torch.equal(n,value)
    gpu=ConditionalEncoder(cfg).to(device);gpu.load_state_dict(enc.float().state_dict())
    expected,_=enc.float()(x.float(),mask.float(),mode='F3')
    actual,_=gpu(x.float().to(device),mask.float().to(device),mode='F3')
    errors['gpu_float32']=float((actual.cpu()-expected).abs().max())
    assert errors['gpu_float32']<1e-4
    return errors


def data_checks():
    checks=[]
    for dataset in ('Weather','Electricity','ETTh1'):
        quality=clean_dataset(dataset)[-1]
        if dataset=='Weather':
            assert quality['invalid_by_split']['train']['total']==81
            assert quality['invalid_by_split']['val']['total']==quality['invalid_by_split']['test']['total']==0
            at=quality['feature_names'].index('wv (m/s)')
            assert 1.6<quality['std'][at]<1.8
        for family,oldfamily in [('P','P'),('Bu','B-uniform')]:
            for seed in (9101,9102):
                new=FusionDataset(dataset,'test',family,seed)
                old=ExperimentDataset(dataset,'test',oldfamily,seed)
                for ix in (0,17,len(new)-1):
                    assert np.array_equal(new.mask_for(ix),old.mask_for(ix))
                suite='posterior_s4_weather_r50_b24_l96_h96_1119' if dataset=='Weather' else 'posterior_s4_r50_b24_l96_h96_1119'
                record=json.loads((EXP_ROOT/'results'/suite/f'{dataset}_PU-S4_P_s2024'/f'test_{oldfamily}_m{seed}.json').read_text())
                assert mask_fingerprint(new)==record['mask_fingerprint']
                checks.append({'dataset':dataset,'family':family,'seed':seed,'legacy_mask_fingerprint_match':True,
                               'fingerprint':mask_fingerprint(new)})
        for family in ('P','Bu'):
            ds=FusionDataset(dataset,'train',family,2024,True)
            assert mask_fingerprint(ds,1)!=mask_fingerprint(ds,2)
            if family=='P':
                old=ExperimentDataset(dataset,'train','P',2024,True)
                for epoch in (1,2):
                    ds.set_epoch(epoch);old.set_epoch(epoch)
                    for ix in (0,17,len(ds)-1):assert np.array_equal(ds.mask_for(ix),old.mask_for(ix))
        if dataset=='Weather':
            ds=FusionDataset(dataset,'train','P',2024,True);ds.set_epoch(1)
            invalid_time=np.flatnonzero(~ds.valid.all(1))[0]
            index=max(0,min(invalid_time-48,len(ds)-1))
            item=ds[index]
            assert torch.isfinite(item['x']).all() and torch.isfinite(item['y']).all()
            assert not np.any(ds.effective_mask(index)[~ds.valid[index:index+96]])
    return {'quality':[clean_dataset(d)[-1] for d in ('Weather','Electricity','ETTh1')],
            'legacy_compatibility':checks,'P_training_compatibility':True,'dynamic_training_Bu':True}


def gpu_checks(device):
    ds=FusionDataset('Weather','train','Bu',2024,True)
    batch=move_batch(torch.utils.data.default_collate([ds[i] for i in range(8)]),device)
    cfg=ModelConfig(channels=ds.n_features)
    baseline=FusionS4('F0',cfg,2024).to(device).eval()
    legacy=PosteriorS4('LI-S4',cfg,2024).to(device).eval()
    with torch.no_grad():
        reference=baseline(batch['x'],batch['mask'])[0]
        assert torch.equal(reference,legacy(batch['x'],batch['mask'])[0])
    records=[]
    for variant in VARIANTS:
        cuda_rng=torch.cuda.get_rng_state().clone()
        model=FusionS4(variant,cfg,2024).to(device).eval()
        assert torch.equal(cuda_rng,torch.cuda.get_rng_state())
        optimizer,_=make_optimizer(model)
        with torch.no_grad():
            pred,_=model(batch['x'],batch['mask'])
            assert torch.equal(pred,reference)
            changed=torch.where(batch['mask'].bool(),batch['x'],float('nan'))
            assert torch.equal(model(changed,batch['mask'])[0],reference)
        if variant!='F0':
            model.zero_grad(set_to_none=True)
            pred,nll=model(batch['x'],batch['mask'])
            ((pred-batch['y']).square().mean()+.1*nll.mean()).backward()
            assert model.encoding_projection.weight.grad.abs().sum()>0
            optimizer.step();optimizer.zero_grad(set_to_none=True)
            pred,_=model(batch['x'],batch['mask']);(pred-batch['y']).square().mean().backward()
            assert model.observation.loadings.grad.abs().sum()>0
            assert model.observation.raw_variance.grad.abs().sum()>0
        model.zero_grad(set_to_none=True)
        module=s4_module();original=module.cauchy_cuda;calls=[]
        def counted(*a,**kw):calls.append(1);return original(*a,**kw)
        module.cauchy_cuda=counted
        try:
            with torch.autograd.profiler.profile(use_cuda=True) as prof:
                model(batch['x'],batch['mask'])[0].square().mean().backward()
            events=sorted({ev.name for ev in prof.function_events if 'cauchy' in ev.name.lower()})
            assert calls and events
        finally:module.cauchy_cuda=original
        with torch.no_grad():
            pred=model(batch['x'],batch['mask'])[0]
            saved=copy.deepcopy(model.state_dict());model.load_state_dict(saved)
            assert torch.equal(pred,model(batch['x'],batch['mask'])[0])
        records.append({'variant':variant,'counts':model.parameter_counts(),'cuda_calls':len(calls),'cuda_events':events})
        del model,optimizer;torch.cuda.empty_cache()
    assert len({r['counts']['total'] for r in records[1:]})==1
    return records


def benchmark(args,device):
    ds=FusionDataset(args.dataset,'train','P',2024,True);ds.set_epoch(1)
    model=FusionS4('F3',ModelConfig(channels=ds.n_features),2024).to(device).train()
    optimizer,_=make_optimizer(model)
    loader=iter(DataLoader(ds,batch_size=64,num_workers=args.workers,pin_memory=True,prefetch_factor=2 if args.workers else None))
    def step():
        batch=move_batch(next(loader),device);optimizer.zero_grad(set_to_none=True)
        pred,nll=model(batch['x'],batch['mask'])
        valid=batch['target_valid'];err=torch.where(valid.bool(),pred-batch['y'],0)
        (err.square().sum()/valid.sum()+.1*nll.mean()).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1,error_if_nonfinite=True);optimizer.step()
    for _ in range(8):step()
    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
    if args.barrier:
        barrier=Path(args.barrier);barrier.mkdir(parents=True,exist_ok=True)
        atomic_json_dump({'pid':os.getpid(),'ready':True},barrier/f'ready_{os.getpid()}.json')
        limit=time.time()+150
        while not (barrier/'go').exists():
            if time.time()>limit:raise TimeoutError('Benchmark barrier timed out')
            time.sleep(.05)
    started=time.perf_counter();wall_start=time.time()
    for _ in range(args.steps):step()
    torch.cuda.synchronize();seconds=time.perf_counter()-started
    return {'dataset':args.dataset,'variant':'F3','batch_size':64,'workers':args.workers,'steps':args.steps,
            'seconds':seconds,'wall_start':wall_start,'wall_end':time.time(),
            'samples_per_second':64*args.steps/seconds,**memory_stats()}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--benchmark',action='store_true')
    parser.add_argument('--dataset',default='Electricity');parser.add_argument('--steps',type=int,default=40)
    parser.add_argument('--workers',type=int,default=6);parser.add_argument('--output',type=str)
    parser.add_argument('--barrier')
    args=parser.parse_args();torch.set_num_threads(2);seed_all(2024);device=torch.device('cuda:0')
    if args.benchmark:
        result={'status':'passed','benchmark':benchmark(args,device),'runtime':runtime_metadata()}
    else:
        result={'status':'passed','numerical':numerical(device),'data':data_checks(),'gpu':gpu_checks(device),
                'runtime':runtime_metadata(),'source':source_fingerprints()}
    path=args.output or EXP_ROOT/'diagnostics'/SUITE/'correctness.json'
    atomic_json_dump(result,path if isinstance(path,type(EXP_ROOT)) else type(EXP_ROOT)(path))
    print(json.dumps({'status':result['status'],'output':str(path),'benchmark':result.get('benchmark')}),flush=True)
