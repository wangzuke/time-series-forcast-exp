#!/usr/bin/env python3
"""Measure synchronized concurrent end-to-end F3 training throughput."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from protocol import EXP_ROOT, atomic_json_dump
from ocpe_fusion_common import SUITE


def main(args):
    root=EXP_ROOT/'diagnostics'/SUITE/'parallel';root.mkdir(parents=True,exist_ok=True)
    rows=[]
    for slots in (1,2,3):
        barrier=root/f'barrier_{slots}_{os.getpid()}';barrier.mkdir()
        processes=[]
        for slot in range(slots):
            output=root/f'gpu{args.gpu}_n{slots}_{slot}.json'
            stream=(root/f'gpu{args.gpu}_n{slots}_{slot}.log').open('w')
            env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=args.gpu,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
            cmd=[sys.executable,'-u',str(EXP_ROOT/'scripts/validate_ocpe_fusion.py'),'--benchmark',
                 '--workers',str(args.workers),'--steps','50','--output',str(output),'--barrier',str(barrier)]
            processes.append((subprocess.Popen(cmd,stdout=stream,stderr=subprocess.STDOUT,env=env),stream,output))
        deadline=time.time()+150
        while len(list(barrier.glob('ready_*.json')))<slots:
            if any(p.poll() is not None for p,_,_ in processes) or time.time()>deadline:
                raise RuntimeError('Benchmark worker failed before synchronized start; inspect logs')
            time.sleep(.2)
        (barrier/'go').write_text('go')
        for process,stream,_ in processes:
            code=process.wait(timeout=180);stream.close()
            if code:raise RuntimeError('Benchmark worker failed')
        data=[json.loads(path.read_text())['benchmark'] for _,_,path in processes]
        wall=max(r['wall_end'] for r in data)-min(r['wall_start'] for r in data)
        row={'slots':slots,'gpu':args.gpu,'workers_per_job':args.workers,
             'aggregate_samples_per_second':slots*64*50/wall,'wall_seconds':wall,'records':data}
        rows.append(row);print(json.dumps(row),flush=True)
    maximum=max(r['aggregate_samples_per_second'] for r in rows)
    chosen=min(r['slots'] for r in rows if r['aggregate_samples_per_second']>=.95*maximum)
    atomic_json_dump({'status':'passed','rows':rows,'chosen_slots_per_gpu':chosen,
                      'policy':'smallest concurrency within 5% of measured maximum aggregate throughput'},root/'summary.json')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--gpu',default='1');p.add_argument('--workers',type=int,default=6)
    main(p.parse_args())
