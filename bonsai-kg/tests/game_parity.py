from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]

def run(cmd):
    p=subprocess.run(cmd,cwd=ROOT,text=True,capture_output=True,check=False)
    if p.returncode!=0:raise RuntimeError(f"command failed rc={p.returncode}: {' '.join(map(str,cmd))}\n{p.stderr[-4000:]}")
    return p.stdout.splitlines(),p.stderr.strip()

def native_rewards(stderr):
    m=re.search(r"reward0=([-+0-9.eE]+) reward1=([-+0-9.eE]+)",stderr)
    if not m:raise RuntimeError(f"native final rewards missing: {stderr}")
    return [float(m.group(1)),float(m.group(2))]

def main(args):
    policy_json=str((ROOT/args.policy_json).resolve());policy_bin=str((ROOT/args.policy_bin).resolve());cpp=str((ROOT/args.cpp).resolve());cuda=str((ROOT/args.cuda).resolve())
    total_ticks=0;mismatches=[];results=[]
    for seed in args.seeds:
        for seat in (0,1):
            tail=[policy_bin,str(seed),str(args.steps),str(seat)]
            cpp_lines,cpp_err=run([cpp,*tail]);cuda_lines,cuda_err=run([cuda,*tail])
            off_lines,off_err=run(["uv","run","--with","kaggle-environments==1.32.7","python",str((ROOT/"bonsai-kg/tests/official_game_canary.py").resolve()),policy_json,"--seed",str(seed),"--steps",str(args.steps),"--seat",str(seat)])
            try:off_meta=json.loads(off_err.splitlines()[-1])
            except Exception as exc:raise RuntimeError(f"official metadata parse failed: {off_err}") from exc
            if cpp_lines!=cuda_lines:mismatches.append({"seed":seed,"seat":seat,"pair":"cpp_cuda","first":next((i for i,(a,b) in enumerate(zip(cpp_lines,cuda_lines)) if a!=b),min(len(cpp_lines),len(cuda_lines)))})
            if cpp_lines!=off_lines:mismatches.append({"seed":seed,"seat":seat,"pair":"cpp_official","first":next((i for i,(a,b) in enumerate(zip(cpp_lines,off_lines)) if a!=b),min(len(cpp_lines),len(off_lines)))})
            cr=native_rewards(cpp_err);gr=native_rewards(cuda_err);orw=[float(x) for x in off_meta["rewards"]]
            if cr!=gr or cr!=orw:mismatches.append({"seed":seed,"seat":seat,"pair":"rewards","cpp":cr,"cuda":gr,"official":orw})
            if off_meta.get("seed_exposed_to_agent"):mismatches.append({"seed":seed,"seat":seat,"pair":"legality","detail":"official seed exposed to agent"})
            total_ticks+=len(cpp_lines);results.append({"seed":seed,"seat":seat,"ticks":len(cpp_lines),"rewards":cr,"official_seed":off_meta.get("official_seed")})
    summary={"seeds":len(args.seeds),"seats":2,"episode_steps":args.steps,"tick_records_per_engine":total_ticks,"cross_engine_tick_line_comparisons":total_ticks*2,"mismatches":len(mismatches),"results":results}
    print(json.dumps(summary,sort_keys=True))
    for x in mismatches[:10]:print("MISMATCH",json.dumps(x,sort_keys=True))
    return 1 if mismatches else 0

if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--policy-json",default="bonsai-kg/examples/reactive_move.json");ap.add_argument("--policy-bin",default="scratch/bonsai_kg/reactive_move.bkgc");ap.add_argument("--cpp",default="scratch/bonsai_kg/cpp_game_runner");ap.add_argument("--cuda",default="scratch/bonsai_kg/cuda_game_runner");ap.add_argument("--steps",type=int,default=80);ap.add_argument("--seeds",type=int,nargs="+",default=[12345,54321,62000,62001]);raise SystemExit(main(ap.parse_args()))
