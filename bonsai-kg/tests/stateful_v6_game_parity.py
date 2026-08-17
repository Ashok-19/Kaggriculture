from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import subprocess
import sys
import warnings
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
BKG=HERE.parent
INTEGRATION=BKG/"integrations/stateful_v6"
ORACLE_DIR=ROOT/"scratch/bonsai_cpu/stateful_core_v6"
sys.path.insert(0,str(BKG/"python"));sys.path.insert(0,str(INTEGRATION))
import bonsai_kg as bkg
from runtime import StatefulV6BKG3Agent
warnings.filterwarnings("ignore",category=ResourceWarning)

def official_make(*args, **kwargs):
    try:
        from kaggle_environments import make
    except ModuleNotFoundError as e:
        raise RuntimeError("official BKG3-v6 canary requires an importable kaggle_environments package") from e
    return make(*args, **kwargs)


def load_oracle():
    spec=importlib.util.spec_from_file_location("stateful_core_v6_oracle_game",ORACLE_DIR/"main.py");m=importlib.util.module_from_spec(spec);assert spec.loader is not None;spec.loader.exec_module(m);return m

def unit_tuple(a):
    a=list(a or ["PASS"]);op=bkg.UNIT_OPS[a[0]];arg=0;n=1
    if len(a)>=2:
        if isinstance(a[1],str):arg=bkg.ITEM_ID[a[1]]
        else:n=int(a[1])
    if len(a)>=3:n=int(a[2])
    return op,arg,n

def order_tuple(a):
    a=list(a);op=bkg.MARKET_OPS[a[0]]
    if a[0] in ("HIRE","BUY_LAND"):return op,0,1
    return op,bkg.ITEM_ID[a[1]],int(a[2])

def trace_line(g,seed,step,agent,action):
    bb=agent.vm.state.blackboard;units=[unit_tuple(action.get("farmer") or ["PASS"])]+[unit_tuple(x) for x in (action.get("hands",[]) or [])];market=[order_tuple(x) for x in (action.get("market",[]) or [])]
    us=';'.join(f"{op},{arg},{n}" for op,arg,n in units);ms=';'.join(f"{op},{item},{n}" for op,item,n in market)
    return f"G={g} SEED={seed} T={step} L={bb[0]} R={bb[1]} S1={bb[2]} S2={bb[3]} F={agent.vm.state.fault} U={len(units)}:{us} M={len(market)}:{ms}"

def pass_agent(raw,configuration=None):
    del raw,configuration
    return {"farmer":["PASS"],"market":[]}

def official_batch(seed0,count,seat,steps):
    oracle=load_oracle();new=StatefulV6BKG3Agent(INTEGRATION/"assets");lines=[];action_mismatch=[];state_mismatch=[];ticks=0
    for g in range(count):
        seed=seed0+g;new.reset()
        def candidate(raw,configuration=None):
            nonlocal ticks
            old_action=oracle.agent(copy.deepcopy(raw),configuration);new_action=new(copy.deepcopy(raw),configuration);step=int(raw.get("step",0) or 0)
            if old_action!=new_action and len(action_mismatch)<10:action_mismatch.append((g,seed,step,old_action,new_action))
            ns=new.state();old_state=(oracle._state["leaf"],oracle._state["route"],bool(oracle._state["stage2"]));new_state=(ns["leaf"],ns["route"],bool(ns["stage216_done"]))
            if old_state!=new_state and len(state_mismatch)<10:state_mismatch.append((g,seed,step,old_state,new_state))
            lines.append(trace_line(g,seed,step,new,new_action));ticks+=1;return old_action
        agents=[candidate,pass_agent] if seat==0 else [pass_agent,candidate]
        env=official_make("kaggriculture",configuration={"episodeSteps":steps,"seed":seed},debug=True);env.run(agents)
    return lines,action_mismatch,state_mismatch,ticks

def invoke(exe,seed0,count,seat,steps):
    asset=INTEGRATION/"assets";cp=subprocess.run([str(ROOT/exe),str(asset/"controller.bkgc"),str(asset/"routes.bkv6"),str(seed0),str(count),str(seat),str(steps)],cwd=ROOT,text=True,capture_output=True,check=False)
    if cp.returncode!=0:raise RuntimeError(f"{exe} failed rc={cp.returncode}: {cp.stderr[-6000:]}")
    return cp.stdout.splitlines(),cp.stderr.splitlines()

def main(args):
    total_ticks=0;old_new_actions=[];old_new_states=[];cross=[];per=[]
    for seat in (0,1):
        official,am,sm,ticks=official_batch(args.seed_start,args.games_per_seat,seat,args.steps);cpp,cpp_meta=invoke(args.cpp,args.seed_start,args.games_per_seat,seat,args.steps);cuda,cuda_meta=invoke(args.cuda,args.seed_start,args.games_per_seat,seat,args.steps);total_ticks+=ticks;old_new_actions.extend(am);old_new_states.extend(sm)
        for engine,lines in (("cpp",cpp),("cuda",cuda)):
            if len(lines)!=len(official):cross.append((seat,engine,"count",len(official),len(lines)))
            for i,(a,b) in enumerate(zip(official,lines)):
                if a!=b:
                    cross.append((seat,engine,i,a,b))
                    if len(cross)>=20:break
        per.append({"seat":seat,"games":args.games_per_seat,"ticks":ticks,"cpp_meta_tail":cpp_meta[-1] if cpp_meta else "","cuda_meta_tail":cuda_meta[-1] if cuda_meta else ""})
    identity=json.loads((INTEGRATION/"assets/identity.json").read_text())
    summary={"games":args.games_per_seat*2,"games_per_seat":args.games_per_seat,"both_seats":True,"episode_steps":args.steps,"ticks":total_ticks,"old_v6_vs_python_action_comparisons":total_ticks,"old_v6_vs_python_action_mismatches":len(old_new_actions),"old_v6_vs_python_state_comparisons":total_ticks,"old_v6_vs_python_state_mismatches":len(old_new_states),"python_vs_cpp_action_state_comparisons":total_ticks,"python_vs_cuda_action_state_comparisons":total_ticks,"cross_engine_mismatches":len(cross),"controller_semantic_hash":identity["controller_semantic_hash"],"route_pack_sha256":identity["route_pack"]["pack_sha256"],"per_seat":per}
    print(json.dumps(summary,sort_keys=True))
    for x in old_new_actions[:5]:print("OLD_NEW_ACTION_MISMATCH",repr(x))
    for x in old_new_states[:5]:print("OLD_NEW_STATE_MISMATCH",repr(x))
    for x in cross[:10]:print("CROSS_MISMATCH",repr(x))
    return 1 if old_new_actions or old_new_states or cross else 0

if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--seed-start",type=int,default=990000);ap.add_argument("--games-per-seat",type=int,default=4);ap.add_argument("--steps",type=int,default=720);ap.add_argument("--cpp",default="scratch/bonsai_kg_v6_translation/stateful_v6_game_cpp");ap.add_argument("--cuda",default="scratch/bonsai_kg_v6_translation/stateful_v6_game_cuda");raise SystemExit(main(ap.parse_args()))
