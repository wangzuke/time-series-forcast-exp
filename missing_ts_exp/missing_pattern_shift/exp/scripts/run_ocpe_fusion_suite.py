#!/usr/bin/env python3
"""Durable core-only scheduler. Idle GPUs, bounded concurrent slots, paired masks."""
from __future__ import annotations
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from protocol import EXP_ROOT, atomic_json_dump
from ocpe_fusion_common import SUITE, PROTOCOL, VARIANTS, source_fingerprints
from report_ocpe_fusion import read, summarize


def gpu_status(gpus,own_pids):
    data=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True,timeout=15)
    devices,uuids={},{}
    for line in data.strip().splitlines():
        index,uuid,used,free,util=map(str.strip,line.split(','));uuids[uuid]=index
        if index in gpus:devices[index]={'used_mib':int(used),'free_mib':int(free),'utilization':int(util),'foreign_pids':[]}
    processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True,timeout=15)
    for line in processes.strip().splitlines():
        if not line.strip():continue
        uuid,pid=map(str.strip,line.split(','));index=uuids.get(uuid)
        if index in devices and int(pid) not in own_pids:devices[index]['foreign_pids'].append(int(pid))
    return devices


def complete(job):
    directory=EXP_ROOT/'results'/SUITE/job['id']
    if read(directory/'progress.json').get('status')!='complete':return False
    return all(read(directory/f'test_{family}_m{seed}.json').get('status')=='complete' for family in ('P','Bu') for seed in (9101,9102))


def main(args):
    logroot=EXP_ROOT/'logs'/SUITE;logroot.mkdir(parents=True,exist_ok=True)
    jobs=[{'id':f'{dataset}_{variant}_{family}_s{seed}','dataset':dataset,'variant':variant,'family':family,'seed':seed}
          for seed in (2024,2025) for family in ('P','Bu') for variant in VARIANTS for dataset in ('Weather','Electricity')]
    manifest={'suite':SUITE,'protocol':PROTOCOL,'jobs':jobs,'total_training_jobs':32,'total_evaluations':128,
              'gpus':args.gpus,'slots_per_gpu':args.slots_per_gpu,'batch_size':64,'effective_batch':64,
              'workers':args.workers,'threads':2,'epochs':30,'patience':5,'aux_weight':.1,
              'source':source_fingerprints(),'test_families':['P','Bu'],'test_seeds':[9101,9102],
              'stage':'E2 core only; no automatic candidate selection or ETTh1/S4M launch'}
    manifest_path=EXP_ROOT/'configs'/f'{SUITE}.json'
    if manifest_path.exists() and read(manifest_path)!=manifest:raise RuntimeError('Frozen manifest differs')
    correctness=read(EXP_ROOT/'diagnostics'/SUITE/'correctness.json')
    if correctness.get('status')!='passed' or correctness.get('source')!=manifest['source']:
        raise RuntimeError('Missing matching correctness preflight')
    atomic_json_dump(manifest,manifest_path)
    if args.build_only:print(json.dumps({'jobs':len(jobs),'manifest':str(manifest_path)}));return
    with (logroot/'supervisor.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);lock.write(str(os.getpid()));lock.flush()
        old=read(logroot/'supervisor_status.json')
        for worker in old.get('active',[]):
            path=Path(f"/proc/{worker['pid']}/cmdline")
            if path.exists() and b'run_ocpe_fusion_s4.py' in path.read_bytes():
                raise RuntimeError('Old workers are still alive; do not duplicate them')
        running,failures={},[];last_report=0
        while True:
            for key,worker in list(running.items()):
                code=worker['process'].poll()
                if code is None:continue
                worker['log'].close()
                if code!=0 or not complete(worker['job']):failures.append({'job':worker['job']['id'],'returncode':code,'time':time.time()})
                print(json.dumps({'finished':worker['job']['id'],'returncode':code}),flush=True)
                del running[key]
            done=[j['id'] for j in jobs if complete(j)]
            active={w['job']['id'] for w in running.values()}
            pending=[j for j in jobs if j['id'] not in set(done)|active]
            devices=gpu_status(args.gpus,{w['process'].pid for w in running.values()})
            if not failures:
                for gpu in args.gpus:
                    device=devices[gpu]
                    if device['foreign_pids']:continue
                    reserved=0
                    for slot in range(args.slots_per_gpu):
                        if (gpu,slot) in running or not pending:continue
                        if device['free_mib']-reserved<args.reserve_mib+2000:break
                        job=pending.pop(0)
                        log=(logroot/f"{job['id']}.log").open('a')
                        cmd=[sys.executable,'-u',str(EXP_ROOT/'scripts/run_ocpe_fusion_s4.py'),
                             '--variant',job['variant'],'--dataset',job['dataset'],'--family',job['family'],
                             '--seed',str(job['seed']),'--job-id',job['id'],'--workers',str(args.workers)]
                        env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=gpu,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
                        process=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=env,cwd=str(EXP_ROOT.parents[2]))
                        running[gpu,slot]={'job':job,'process':process,'log':log,'gpu':gpu,'start':time.time()}
                        reserved+=args.reserve_mib
                        print(json.dumps({'started':job['id'],'gpu':gpu,'pid':process.pid}),flush=True)
            state={'status':'failed' if failures else 'complete' if len(done)==len(jobs) else 'running',
                   'pid':os.getpid(),'updated_at':time.time(),'jobs_total':len(jobs),'jobs_complete':done,'failures':failures,
                   'active':[{'id':w['job']['id'],'pid':w['process'].pid,'gpu':w['gpu'],'start':w['start']} for w in running.values()],
                   'devices':devices,'pending':len(pending)}
            atomic_json_dump(state,logroot/'supervisor_status.json')
            if time.time()-last_report>45:
                result=summarize()
                if result['mask_consistency']['conflicts']:
                    failures.append({'reason':'mask consistency conflict','detail':result['mask_consistency']['conflicts']})
                with (logroot/'monitor.jsonl').open('a') as f:f.write(json.dumps(state)+'\n')
                last_report=time.time()
            if not running and (failures or len(done)==len(jobs)):
                result=summarize()
                if result['mask_consistency']['conflicts']:failures.append({'reason':'final mask consistency conflict'})
                state.update(status='failed' if failures else 'complete',failures=failures)
                atomic_json_dump(state,logroot/'supervisor_status.json')
                break
            time.sleep(5)
        if failures:raise SystemExit('Failure recorded. New scheduling stopped; inspect supervisor_status.json')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--gpus',nargs='+',default=['0','1','7'])
    p.add_argument('--slots-per-gpu',type=int,default=3);p.add_argument('--workers',type=int,default=6)
    p.add_argument('--reserve-mib',type=int,default=5000);p.add_argument('--build-only',action='store_true')
    main(p.parse_args())
