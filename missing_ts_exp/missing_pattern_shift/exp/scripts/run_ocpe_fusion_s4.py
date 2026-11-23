#!/usr/bin/env python3
"""One resumable OCPE-fusion run, with valid targets and four paired tests."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
import traceback
import numpy as np
import torch
from torch.utils.data import DataLoader

from protocol import EXP_ROOT, atomic_json_dump, atomic_torch_save
from ocpe_fusion_common import (SUITE, PROTOCOL, FusionDataset, FusionS4, ModelConfig, VARIANTS, LABELS,
    source_fingerprints, seed_all, digest, mask_fingerprint, mask_audit, move_batch, memory_stats, runtime_metadata, make_optimizer)
from run_posterior_s4 import rng_state, restore_rng


def sequence_hash(hasher, batch):
    # Canonical per-window encoding, independent of microbatch boundaries.
    masks = np.packbits(batch['mask'].numpy().astype(bool).reshape(len(batch['index']), -1), axis=1)
    for index, mask in zip(batch['index'].tolist(), masks):
        hasher.update(np.asarray(index, dtype='<i8').tobytes())
        hasher.update(mask.tobytes())


@torch.no_grad()
def evaluate(model, loader, device, output=None):
    model.eval()
    channels, horizon = model.cfg.channels, model.cfg.pred_len
    sums = torch.zeros(6, dtype=torch.float64, device=device)
    channel_sq = torch.zeros(channels, dtype=torch.float64, device=device)
    channel_count = channel_sq.clone()
    step_sq = torch.zeros(horizon, dtype=torch.float64, device=device)
    step_count = step_sq.clone()
    mean = torch.as_tensor(loader.dataset.mean.copy(), device=device)
    std = torch.as_tensor(loader.dataset.std.copy(), device=device)
    window_errors, window_counts, indices = [], [], []
    hasher = hashlib.sha256()
    started = time.perf_counter()
    for batch in loader:
        sequence_hash(hasher, batch)
        batch = move_batch(batch, device)
        pred, _ = model(batch['x'], batch['mask'])
        valid = batch['target_valid']
        err = torch.where(valid.bool(), pred - batch['y'], 0)
        rawerr = torch.where(valid.bool(), pred * std + mean - batch['y_raw'], 0)
        sq = err.square()
        sums += torch.stack((sq.sum(), err.abs().sum(), rawerr.square().sum(), rawerr.abs().sum(),
                             (batch['y'].square()*valid).sum(), valid.sum())).double()
        channel_sq += sq.sum((0, 1)).double()
        channel_count += valid.sum((0, 1)).double()
        step_sq += sq.sum((0, 2)).double()
        step_count += valid.sum((0, 2)).double()
        window_errors.append(sq.sum((1, 2)).cpu().numpy())
        window_counts.append(valid.sum((1, 2)).cpu().numpy())
        indices.append(batch['index'].cpu().numpy())
    torch.cuda.synchronize()
    sums = sums.cpu().numpy()
    if not np.isfinite(sums).all() or sums[-1] <= 0:
        raise FloatingPointError(f'Invalid evaluation totals: {sums}')
    values = sums[:5]/sums[-1]
    result = dict(zip(('mse_standardized','mae_standardized','mse_raw','mae_raw','train_mean_baseline_mse'), values.tolist()))
    result.update(n_windows=len(loader.dataset), n_target_elements=int(sums[-1]),
                  effective_mask_sequence_sha256=hasher.hexdigest(), seconds=time.perf_counter()-started,
                  mse_by_channel=(channel_sq/channel_count.clamp_min(1)).cpu().tolist(),
                  target_count_by_channel=channel_count.cpu().tolist(),
                  mse_by_horizon=(step_sq/step_count.clamp_min(1)).cpu().tolist())
    if output:
        errors, counts = np.concatenate(window_errors), np.concatenate(window_counts)
        with tempfile.NamedTemporaryFile(dir=output.parent, delete=False) as f:
            temporary = Path(f.name)
            try:
                np.savez_compressed(f, index=np.concatenate(indices), window_sse=errors, window_valid_count=counts,
                                    window_mse=errors/np.maximum(counts,1))
                f.flush();os.replace(temporary,output.with_suffix('.npz'))
            finally:
                temporary.unlink(missing_ok=True)
    return result


def target_counts(ds):
    total = np.r_[0,np.cumsum(ds.valid.sum(1,dtype=np.int64))]
    ix = np.arange(len(ds))
    return total[ix+192]-total[ix+96]


def train_epoch(model, loader, order, optimizer, args, device, progress, base):
    model.train()
    sums = torch.zeros(3,dtype=torch.float64,device=device)
    n_samples, clip_steps = 0, 0
    accumulation = args.effective_batch//args.batch_size
    valid_counts = target_counts(loader.dataset)
    hasher = hashlib.sha256()
    started, last_report = time.perf_counter(), 0
    for number,batch in enumerate(loader):
        sequence_hash(hasher,batch)
        if number%accumulation==0:
            optimizer.zero_grad(set_to_none=True)
            left = number*args.batch_size
            group = order[left:left+args.effective_batch]
            group_valid = int(valid_counts[group].sum())
            group_samples = len(group)
            if group_valid <= 0:
                raise ValueError('No valid targets in effective batch')
        batch=move_batch(batch,device)
        pred,nll=model(batch['x'],batch['mask'])
        error=torch.where(batch['target_valid'].bool(),pred-batch['y'],0)
        sse=error.square().sum()
        loss=sse/group_valid+args.aux_weight*nll.sum()/group_samples
        if not torch.isfinite(loss):raise FloatingPointError('Non-finite train loss')
        loss.backward()
        sums+=torch.stack((sse.detach(),batch['target_valid'].sum(),nll.detach().sum())).double()
        n_samples+=len(pred)
        if (number+1)%accumulation==0 or number+1==len(loader):
            grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.0,error_if_nonfinite=True)
            clip_steps+=int(grad.item()>1)
            optimizer.step()
        if time.perf_counter()-last_report>10:
            atomic_json_dump({**base,'batch':number+1,'total_batches':len(loader),
                              'updated_at':time.time(),**memory_stats()},progress)
            last_report=time.perf_counter()
    values=sums.cpu().tolist()
    return {'mse':values[0]/values[1],'observation_nll':values[2]/n_samples,
            'samples':n_samples,'valid_targets':int(values[1]),'seconds':time.perf_counter()-started,
            'clip_steps':clip_steps,'effective_mask_sequence_sha256':hasher.hexdigest()}


def main(args):
    torch.set_num_threads(args.threads)
    seed_all(args.seed)
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    device=torch.device('cuda:0');torch.cuda.set_device(device);torch.cuda.reset_peak_memory_stats()
    train=FusionDataset(args.dataset,'train',args.family,args.seed,True)
    val=FusionDataset(args.dataset,'val',args.family,args.seed+10000)
    if args.smoke_windows:
        train.n_samples=min(len(train),args.smoke_windows)
        val.n_samples=min(len(val),args.smoke_windows)
    model=FusionS4(args.variant,ModelConfig(channels=train.n_features),args.seed).to(device)
    runtime=runtime_metadata()
    if not runtime['has_cuda_extension']:raise RuntimeError('Official S4 CUDA extension required')
    optimizer,groups=make_optimizer(model)
    directory=EXP_ROOT/'results'/args.suite/args.job_id
    checkpoint_dir=EXP_ROOT/'checkpoints'/args.suite
    directory.mkdir(parents=True,exist_ok=True);checkpoint_dir.mkdir(parents=True,exist_ok=True)
    progress=directory/'progress.json'
    best_path=checkpoint_dir/f'{args.job_id}.best.pt'
    last_path=checkpoint_dir/f'{args.job_id}.last.pt'
    record={'suite':args.suite,'job_id':args.job_id,'variant':args.variant,'label':LABELS[args.variant],
            'dataset':args.dataset,'train_family':args.family,'train_seed':args.seed,'protocol':PROTOCOL,
            'source':source_fingerprints(),'model':model.cfg.to_dict(),'batch_size':args.batch_size,
            'effective_batch':args.effective_batch,'workers':args.workers,'threads':args.threads,
            'epochs':args.epochs,'patience':args.patience,'aux_weight':args.aux_weight,
            'smoke_windows':args.smoke_windows,'optimizer':'AdamW','optimizer_groups':groups,
            'gradient_clip_norm':1.0,'parameter_counts':model.parameter_counts(),'data_quality':train.quality,
            'train_mask_epoch1':mask_fingerprint(train,1),'train_mask_epoch2':mask_fingerprint(train,2),
            'validation_mask':mask_fingerprint(val),'validation_effective_mask':mask_fingerprint(val,effective=True),
            'train_mask_audit':mask_audit(train),'validation_mask_audit':mask_audit(val)}
    fingerprint=digest(record);record.update(fingerprint=fingerprint,runtime=runtime)
    path=directory/'protocol.json'
    if path.exists() and json.loads(path.read_text())['fingerprint']!=fingerprint:
        raise RuntimeError('Frozen job identity differs; use a new job id/suite')
    atomic_json_dump(record,path)
    options=dict(batch_size=args.batch_size,num_workers=args.workers,pin_memory=True,persistent_workers=False)
    if args.workers:options['prefetch_factor']=2
    def loader(ds,order=None,epoch=0):
        return DataLoader(ds,sampler=order,shuffle=False,
                          generator=torch.Generator().manual_seed(args.seed+500000+epoch),**options)
    best,stale,start_epoch,history=float('inf'),0,1,[]
    complete_path=directory/'training_complete.json'
    if not complete_path.exists():
        if last_path.exists():
            saved=torch.load(last_path,map_location=device,weights_only=False)
            if saved['fingerprint']!=fingerprint:raise RuntimeError('Resume fingerprint mismatch')
            model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer'])
            best,stale,start_epoch=saved['best'],saved['stale'],saved['epoch']+1
            history=saved['history'];restore_rng(saved['rng'])
        started=time.time()
        for epoch in range(start_epoch,args.epochs+1):
            if stale>=args.patience:break
            train.set_epoch(epoch)
            order=torch.randperm(len(train),generator=torch.Generator().manual_seed(args.seed+100000+epoch)).tolist()
            base={'status':'training','job_id':args.job_id,'epoch':epoch,'pid':os.getpid()}
            stats=train_epoch(model,loader(train,order,epoch),order,optimizer,args,device,progress,base)
            value=evaluate(model,loader(val),device)['mse_standardized']
            improved=value<best
            if improved:
                best,stale=value,0
                atomic_torch_save({'model':model.state_dict(),'epoch':epoch,'val_mse':value,
                                  'fingerprint':fingerprint,'config':model.cfg.to_dict(),'variant':args.variant},best_path)
            else:stale+=1
            history.append({'epoch':epoch,'train':stats,'val_mse':value,'improved':improved})
            atomic_torch_save({'model':model.state_dict(),'optimizer':optimizer.state_dict(),'epoch':epoch,
                              'best':best,'stale':stale,'history':history,'rng':rng_state(),'fingerprint':fingerprint},last_path)
            print(json.dumps(history[-1]),flush=True)
        atomic_json_dump({'status':'complete','job_id':args.job_id,'fingerprint':fingerprint,
                          'best_val_mse':best,'history':history,'training_seconds_this_session':time.time()-started,
                          'stop_reason':'early_stopping' if stale>=args.patience else 'max_epochs',**memory_stats()},complete_path)
    elif json.loads(complete_path.read_text())['fingerprint']!=fingerprint:
        raise RuntimeError('Training-complete identity mismatch')
    saved=torch.load(best_path,map_location=device,weights_only=False)
    if saved['fingerprint']!=fingerprint:raise RuntimeError('Best identity mismatch')
    model.load_state_dict(saved['model'])
    for family in ('P','Bu'):
        for seed in (9101,9102):
            output=directory/f'test_{family}_m{seed}.json'
            if output.exists():
                existing=json.loads(output.read_text())
                if existing.get('fingerprint')==fingerprint and existing.get('status')=='complete':continue
                raise RuntimeError('Existing evaluation incompatible')
            ds=FusionDataset(args.dataset,'test',family,seed)
            if args.smoke_windows:ds.n_samples=min(len(ds),args.smoke_windows)
            atomic_json_dump({'status':'evaluating','job_id':args.job_id,'family':family,'mask_seed':seed,'updated_at':time.time()},progress)
            result=evaluate(model,loader(ds),device,output)
            result.update(status='complete',fingerprint=fingerprint,job_id=args.job_id,test_family=family,test_mask_seed=seed,
                          mask_fingerprint=mask_fingerprint(ds),mask_audit=mask_audit(ds),best_epoch=saved['epoch'])
            atomic_json_dump(result,output)
            print(json.dumps({'test':family,'mask_seed':seed,'mse':result['mse_standardized']}),flush=True)
    atomic_json_dump({'status':'complete','job_id':args.job_id,'fingerprint':fingerprint,'best_epoch':saved['epoch'],
                      'updated_at':time.time(),**memory_stats()},progress)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--variant',choices=VARIANTS,required=True)
    parser.add_argument('--dataset',choices=('Weather','Electricity','ETTh1'),required=True)
    parser.add_argument('--family',choices=('P','Bu'),required=True)
    parser.add_argument('--seed',type=int,required=True)
    parser.add_argument('--job-id',required=True)
    parser.add_argument('--suite',default=SUITE)
    parser.add_argument('--batch-size',type=int,default=64)
    parser.add_argument('--effective-batch',type=int,default=64)
    parser.add_argument('--workers',type=int,default=6)
    parser.add_argument('--threads',type=int,default=2)
    parser.add_argument('--epochs',type=int,default=30)
    parser.add_argument('--patience',type=int,default=5)
    parser.add_argument('--aux-weight',type=float,default=.1)
    parser.add_argument('--smoke-windows',type=int,default=0)
    args=parser.parse_args()
    if args.effective_batch%args.batch_size:parser.error('microbatch must divide effective batch')
    try:
        lockdir=EXP_ROOT/'results'/args.suite/args.job_id
        lockdir.mkdir(parents=True,exist_ok=True)
        with (lockdir/'job.lock').open('w') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            main(args)
    except Exception:
        atomic_json_dump({'status':'failed','traceback':traceback.format_exc(),'timestamp':time.time()},
                         EXP_ROOT/'results'/args.suite/args.job_id/'failure.json')
        raise
