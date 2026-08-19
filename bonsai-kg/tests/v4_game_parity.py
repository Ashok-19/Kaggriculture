from __future__ import annotations
import argparse,json,re,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def run(cmd):
 p=subprocess.run(cmd,cwd=ROOT,text=True,capture_output=True); 
 if p.returncode:raise RuntimeError(f"rc={p.returncode}: {' '.join(map(str,cmd))}\n{p.stderr[-4000:]}")
 return p.stdout.splitlines(),p.stderr.strip()
def rewards(err):
 m=re.search(r'reward0=([-+0-9.eE]+) reward1=([-+0-9.eE]+)',err)
 if not m:raise RuntimeError(err)
 return [float(m.group(1)),float(m.group(2))]
def main(a):
 total=0;bad=[]
 for seed in a.seeds:
  for seat in (0,1):
   tail=[str((ROOT/a.policy_bin).resolve()),str(seed),str(a.steps),str(seat)];c,ce=run([str((ROOT/a.cpp).resolve()),*tail]);g,ge=run([str((ROOT/a.cuda).resolve()),*tail]);o,oe=run(['uv','run','--with','kaggle-environments==1.32.7','python',str((ROOT/'bonsai-kg/tests/v4_official_game_canary.py').resolve()),str((ROOT/a.policy_json).resolve()),'--seed',str(seed),'--steps',str(a.steps),'--seat',str(seat)]);meta=json.loads(oe.splitlines()[-1]);total+=len(c)
   if c!=g:bad.append((seed,seat,'cpp_cuda'))
   if c!=o:bad.append((seed,seat,'cpp_official'))
   if rewards(ce)!=rewards(ge) or rewards(ce)!=[float(x) for x in meta['rewards']]:bad.append((seed,seat,'rewards'))
   if meta['seed_exposed_to_agent']:bad.append((seed,seat,'seed_exposed'))
 print(json.dumps({'seeds':len(a.seeds),'seats':2,'ticks_per_engine':total,'pairwise_tick_comparisons':total*2,'mismatches':len(bad)},sort_keys=True));[print('MISMATCH',x) for x in bad[:10]];return 1 if bad else 0
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--policy-json',default='bonsai-kg/examples/reactive_move_v4.json');ap.add_argument('--policy-bin',default='scratch/bonsai_kg/reactive_move_v4.bkgc');ap.add_argument('--cpp',default='scratch/bonsai_kg/v4_cpp_game');ap.add_argument('--cuda',default='scratch/bonsai_kg/v4_cuda_game');ap.add_argument('--steps',type=int,default=40);ap.add_argument('--seeds',type=int,nargs='+',default=[12345,54321]);raise SystemExit(main(ap.parse_args()))
