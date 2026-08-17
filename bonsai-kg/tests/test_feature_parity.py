from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
SCENARIOS=[
    ("seat0_own",0,0),
    ("seat1_own",1,1),
    ("seat0_observes_active_seat1",1,0),
]

def run(cmd):
    p=subprocess.run(cmd,cwd=ROOT,text=True,capture_output=True,check=False)
    if p.returncode!=0:raise RuntimeError(f"rc={p.returncode}: {' '.join(map(str,cmd))}\n{p.stderr[-4000:]}")
    return [x for x in p.stdout.splitlines() if x.startswith("T=")]

def raw_rows(lines):
    out=[]
    for line in lines:
        marker=" R="
        if marker not in line:raise RuntimeError(f"missing raw feature vector: {line}")
        out.append([int(x) for x in line.split(marker,1)[1].split(',')])
    return out

def main(args):
    total=0;m=[];results=[]
    private_nontrivial=False;opp_lifecycle_nontrivial=False
    for seed in args.seeds:
        for name,active,observe in SCENARIOS:
            tail=[str(seed),str(args.steps),str(active),str(observe)]
            cpp=run([str(ROOT/args.cpp),*tail])
            cuda=run([str(ROOT/args.cuda),*tail])
            off=run(["uv","run","--with","kaggle-environments==1.32.7","python",str(ROOT/"bonsai-kg/tests/official_feature_parity.py"),"--seed",str(seed),"--steps",str(args.steps),"--active-player",str(active),"--observe-player",str(observe)])
            total+=len(cpp)
            if cpp!=cuda:m.append({"seed":seed,"scenario":name,"pair":"cpp_cuda","first":next((i for i,(a,b) in enumerate(zip(cpp,cuda)) if a!=b),min(len(cpp),len(cuda)))})
            if cpp!=off:m.append({"seed":seed,"scenario":name,"pair":"cpp_official","first":next((i for i,(a,b) in enumerate(zip(cpp,off)) if a!=b),min(len(cpp),len(off)))})
            rows=raw_rows(cpp)
            if name=="seat1_own":
                private_nontrivial |= any(any(v!=0 for v in row[33:38]) for row in rows)
            if name=="seat0_observes_active_seat1":
                # OPP plant/animal lifecycle segment: planted/dry/lifespan/fertilized + animal placed/unfed/care-bonus.
                opp_lifecycle_nontrivial |= any(
                    row[25] > 0 or row[26] > 0 or row[27] > 0 or row[29] > 0 or row[30] > 0 or row[31] > 0 or row[32] > 0
                    for row in rows
                )
            results.append({"seed":seed,"scenario":name,"active_player":active,"observe_player":observe,"ticks":len(cpp)})
    if not private_nontrivial:m.append({"assertion":"player1 own-private canaries never became nontrivial"})
    if not opp_lifecycle_nontrivial:m.append({"assertion":"passive seat0 never observed nontrivial opponent lifecycle state"})
    reads_per_tick=3793;raw_per_tick=44
    print(json.dumps({"seeds":len(args.seeds),"scenarios":len(SCENARIOS),"ticks_per_engine":total,"pairwise_exact_line_comparisons":total*2,
                      "aggregate_feature_reads_per_tick":reads_per_tick,"aggregate_feature_reads_per_engine":total*reads_per_tick,
                      "pairwise_aggregate_read_positions":total*reads_per_tick*2,"targeted_raw_values_per_tick":raw_per_tick,
                      "pairwise_targeted_raw_values":total*raw_per_tick*2,"player1_private_nontrivial":private_nontrivial,
                      "opponent_lifecycle_nontrivial":opp_lifecycle_nontrivial,"mismatches":len(m),"results":results},sort_keys=True))
    for x in m[:10]:print("MISMATCH",json.dumps(x,sort_keys=True))
    return 1 if m else 0

if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--cpp",default="scratch/bonsai_kg/cpp_feature_parity");ap.add_argument("--cuda",default="scratch/bonsai_kg/cuda_feature_parity");ap.add_argument("--steps",type=int,default=80);ap.add_argument("--seeds",type=int,nargs="+",default=[12345,54321]);raise SystemExit(main(ap.parse_args()))
