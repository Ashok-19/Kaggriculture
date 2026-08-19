#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,statistics,subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
GROUPS={
    'cold':[128,1024,4096,16384,32768,65534],
    'hot':[16,32,64,128,256,512],
    'state':[32,64,128,256,512],
    'intent':[0,8,16,32,64,128],
    'depth':[16,24,32,48],
}

def parse(line:str):
    out={}
    for tok in line.strip().split():
        if '=' not in tok:continue
        k,v=tok.split('=',1)
        try:out[k]=float(v) if any(c in v for c in '.eE') else int(v)
        except ValueError:out[k]=v
    return out

def run(cmd):
    p=subprocess.run(cmd,cwd=ROOT,text=True,capture_output=True,check=False)
    if p.returncode!=0:raise RuntimeError(f"rc={p.returncode}: {' '.join(map(str,cmd))}\n{p.stderr[-4000:]}")
    return parse(p.stdout.splitlines()[-1])

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--bench',default='scratch/bonsai_kg/v4_benchmark');ap.add_argument('--policies',default='scratch/bonsai_kg/v4_bench_policies');ap.add_argument('--compare',default='scratch/bonsai_kg/v4_bkg3_comparison');ap.add_argument('--tiny',default='scratch/bonsai_kg/v4_bench_policies/tiny_equiv.bkgc');ap.add_argument('--cpu-iters',type=int,default=20000);ap.add_argument('--gpu-envs',type=int,default=4096);ap.add_argument('--gpu-iters',type=int,default=32);ap.add_argument('--repeats',type=int,default=3);ap.add_argument('--out',default='scratch/bonsai_kg/v4_benchmark_matrix.json');args=ap.parse_args()
    root=ROOT/args.policies;bench=str((ROOT/args.bench).resolve());matrix={}
    for group,vals in GROUPS.items():
        rows=[]
        for case in vals:
            samples=[]
            for _ in range(args.repeats):samples.append(run([bench,str(root/f'{group}_{case}.bkgc'),str(args.cpu_iters),str(args.gpu_envs),str(args.gpu_iters)]))
            row=dict(samples[len(samples)//2]);row['case']=case;row['gpu_ticks_s_median']=statistics.median(float(x['gpu_ticks_s']) for x in samples);row['cpu_ticks_s_median']=statistics.median(float(x['cpu_ticks_s']) for x in samples);rows.append(row)
        matrix[group]=rows
    comp=run([str((ROOT/args.compare).resolve()),str((ROOT/args.tiny).resolve()),str(max(100000,args.cpu_iters)),str(args.gpu_envs),str(max(64,args.gpu_iters)),'7'])
    cold=matrix['cold'];cold_ratio=float(cold[-1]['gpu_ticks_s_median'])/float(cold[0]['gpu_ticks_s_median'])
    summary={
        'cold_65534_vs_128_gpu_ratio':cold_ratio,
        'cold_runtime_bytes_constant':len({int(x['runtime_bytes_env']) for x in cold})==1,
        'comparison':comp,
        'max_regs':max(int(x['regs']) for rows in matrix.values() for x in rows),
        'max_local_bytes':max(int(x['local_bytes']) for rows in matrix.values() for x in rows),
        'min_occupancy_threads_fraction':min(float(x['occupancy_threads_fraction']) for rows in matrix.values() for x in rows),
    }
    result={'summary':summary,'matrix':matrix};Path(ROOT/args.out).write_text(json.dumps(result,indent=2,sort_keys=True)+'\n');print(json.dumps(summary,sort_keys=True))
    if not summary['cold_runtime_bytes_constant']:return 2
    if cold_ratio<0.80:return 3
    if float(comp['gpu_ratio'])<0.75 or float(comp['cpu_ratio'])<0.70:return 4
    return 0

if __name__=='__main__':raise SystemExit(main())
