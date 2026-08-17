from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]

def run(cmd):
    p=subprocess.run(cmd,cwd=ROOT,text=True,capture_output=True,check=False)
    if p.returncode!=0:raise RuntimeError(f"rc={p.returncode}: {' '.join(map(str,cmd))}\n{p.stderr[-4000:]}")
    return [x for x in p.stdout.splitlines() if x.startswith("T=")]

def main(args):
    total=0;m=[];results=[]
    for seed in args.seeds:
        cpp=run([str(ROOT/args.cpp),str(seed),str(args.steps)])
        cuda=run([str(ROOT/args.cuda),str(seed),str(args.steps)])
        off=run(["uv","run","--with","kaggle-environments==1.32.7","python",str(ROOT/"bonsai-kg/tests/official_feature_parity.py"),"--seed",str(seed),"--steps",str(args.steps)])
        total+=len(cpp)
        if cpp!=cuda:m.append({"seed":seed,"pair":"cpp_cuda","first":next((i for i,(a,b) in enumerate(zip(cpp,cuda)) if a!=b),min(len(cpp),len(cuda)))})
        if cpp!=off:m.append({"seed":seed,"pair":"cpp_official","first":next((i for i,(a,b) in enumerate(zip(cpp,off)) if a!=b),min(len(cpp),len(off))),"cpp":cpp[:3],"official":off[:3]})
        results.append({"seed":seed,"ticks":len(cpp)})
    print(json.dumps({"seeds":len(args.seeds),"ticks_per_engine":total,"pairwise_hash_comparisons":total*2,"mismatches":len(m),"results":results},sort_keys=True))
    for x in m[:10]:print("MISMATCH",json.dumps(x,sort_keys=True))
    return 1 if m else 0

if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--cpp",default="scratch/bonsai_kg/cpp_feature_parity");ap.add_argument("--cuda",default="scratch/bonsai_kg/cuda_feature_parity");ap.add_argument("--steps",type=int,default=80);ap.add_argument("--seeds",type=int,nargs="+",default=[12345,54321]);raise SystemExit(main(ap.parse_args()))
