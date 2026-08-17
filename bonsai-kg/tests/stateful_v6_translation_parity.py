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

SHOP_NAMES=("BAKERY","BRUNCH_SPOT","FARMERS_MARKET","ICE_CREAM_SHOP","PET_CAFE","PIZZA_SHOP","SMOOTHIE_SHOP","YARN_STORE")
PRODUCTS=bkg.ITEMS[:9]


def load_oracle():
    spec=importlib.util.spec_from_file_location("stateful_core_v6_oracle_trace",ORACLE_DIR/"main.py");m=importlib.util.module_from_spec(spec);assert spec.loader is not None;spec.loader.exec_module(m);return m

def invoke(exe,seed0,count,seat,steps):
    asset=INTEGRATION/"assets";cp=subprocess.run([str(ROOT/exe),str(asset/"controller.bkgc"),str(asset/"routes.bkv6"),str(seed0),str(count),str(seat),str(steps)],cwd=ROOT,text=True,capture_output=True,check=False)
    if cp.returncode!=0:raise RuntimeError(f"{exe} failed rc={cp.returncode}: {cp.stderr[-6000:]}")
    return cp.stdout.splitlines(),cp.stderr.splitlines()

def parse_obs_line(line):
    parts=line.split();vals={k:v for k,v in (x.split('=',1) for x in parts)};g=int(vals['G']);seed=int(vals['SEED']);step=int(vals['T']);player=int(vals['P']);own_money=float(vals['OM']);opp_money=float(vals['XM']);hands=int(vals['H']);nshop,shopraw=vals['SH'].split(':',1);shops=[] if not shopraw else [int(x) for x in shopraw.split(',')];assert len(shops)==int(nshop);npr,prraw=vals['PR'].split(':',1);prices=[int(x) for x in prraw.split(',')];assert len(prices)==int(npr)==9
    tiles0=[[None for _ in range(10)] for _ in range(10)];tiles1=[[None for _ in range(10)] for _ in range(10)]
    farms=[{"money":opp_money,"tiles":tiles0,"farmer":[0,0],"hands":[],"unlocked_quadrants":["NW"]},{"money":opp_money,"tiles":tiles1,"farmer":[0,0],"hands":[],"unlocked_quadrants":["NW"]}]
    farms[player]["money"]=own_money;farms[1-player]["money"]=opp_money;farms[player]["hands"]=[[0,0] for _ in range(hands)]
    obs={"step":step,"day":step//24,"hour":step%24,"player":player,"farms":farms,"town":{"unlocked_shops":[SHOP_NAMES[x] for x in shops]},"market":{"prices":{PRODUCTS[i]:prices[i] for i in range(9)},"inventory":{}},"private":{"shed":{},"seeds":{},"inventories":[{} for _ in range(hands+1)]}}
    return g,seed,step,obs

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
    bb=agent.vm.state.blackboard;units=[unit_tuple(action.get("farmer") or ["PASS"])]+[unit_tuple(x) for x in (action.get("hands",[]) or [])];market=[order_tuple(x) for x in (action.get("market",[]) or [])];us=';'.join(f"{op},{arg},{n}" for op,arg,n in units);ms=';'.join(f"{op},{item},{n}" for op,item,n in market)
    return f"G={g} SEED={seed} T={step} L={bb[0]} R={bb[1]} S1={bb[2]} S2={bb[3]} F={agent.vm.state.fault} U={len(units)}:{us} M={len(market)}:{ms}"

def main(args):
    old_new_action=[];old_new_state=[];py_cpp=[];cpp_cuda=[];ticks=0;per=[]
    for seat in (0,1):
        obs_lines,_=invoke(args.obs_cpp,args.seed_start,args.games_per_seat,seat,args.steps);cpp_lines,cpp_meta=invoke(args.cpp,args.seed_start,args.games_per_seat,seat,args.steps);cuda_lines,cuda_meta=invoke(args.cuda,args.seed_start,args.games_per_seat,seat,args.steps)
        if len(obs_lines)!=len(cpp_lines):raise RuntimeError(f"observation/cpp count mismatch seat {seat}: {len(obs_lines)} != {len(cpp_lines)}")
        oracle=load_oracle();new=StatefulV6BKG3Agent(INTEGRATION/"assets");python_lines=[]
        for i,ol in enumerate(obs_lines):
            g,seed,step,o=parse_obs_line(ol);old=oracle.agent(copy.deepcopy(o));fresh=new(copy.deepcopy(o));
            if old!=fresh and len(old_new_action)<20:old_new_action.append((seat,i,g,seed,step,old,fresh))
            ns=new.state();os=(oracle._state['last_step'],oracle._state['leaf'],oracle._state['route'],bool(oracle._state['stage2']));ts=(new.last_step,ns['leaf'],ns['route'],bool(ns['stage216_done']))
            if os!=ts and len(old_new_state)<20:old_new_state.append((seat,i,g,seed,step,os,ts))
            python_lines.append(trace_line(g,seed,step,new,fresh))
        for i,(a,b) in enumerate(zip(python_lines,cpp_lines)):
            if a!=b and len(py_cpp)<20:py_cpp.append((seat,i,a,b))
        if len(python_lines)!=len(cpp_lines):py_cpp.append((seat,'count',len(python_lines),len(cpp_lines)))
        for i,(a,b) in enumerate(zip(cpp_lines,cuda_lines)):
            if a!=b and len(cpp_cuda)<20:cpp_cuda.append((seat,i,a,b))
        if len(cpp_lines)!=len(cuda_lines):cpp_cuda.append((seat,'count',len(cpp_lines),len(cuda_lines)))
        ticks+=len(obs_lines);per.append({"seat":seat,"games":args.games_per_seat,"ticks":len(obs_lines),"cpp_meta_tail":cpp_meta[-1] if cpp_meta else "","cuda_meta_tail":cuda_meta[-1] if cuda_meta else ""})
    identity=json.loads((INTEGRATION/"assets/identity.json").read_text());summary={"games":args.games_per_seat*2,"games_per_seat":args.games_per_seat,"both_seats":True,"episode_steps":args.steps,"legal_history_ticks":ticks,"old_v6_vs_python_action_comparisons":ticks,"old_v6_vs_python_action_mismatches":len(old_new_action),"old_v6_vs_python_state_comparisons":ticks,"old_v6_vs_python_state_mismatches":len(old_new_state),"python_vs_cpp_action_state_comparisons":ticks,"python_vs_cpp_mismatches":len(py_cpp),"cpp_vs_cuda_action_state_comparisons":ticks,"cpp_vs_cuda_mismatches":len(cpp_cuda),"controller_semantic_hash":identity['controller_semantic_hash'],"route_pack_sha256":identity['route_pack']['pack_sha256'],"per_seat":per};print(json.dumps(summary,sort_keys=True))
    for tag,arr in (("OLD_NEW_ACTION_MISMATCH",old_new_action),("OLD_NEW_STATE_MISMATCH",old_new_state),("PY_CPP_MISMATCH",py_cpp),("CPP_CUDA_MISMATCH",cpp_cuda)):
        for x in arr[:5]:print(tag,repr(x))
    return 1 if old_new_action or old_new_state or py_cpp or cpp_cuda else 0

if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument('--seed-start',type=int,default=990000);ap.add_argument('--games-per-seat',type=int,default=4);ap.add_argument('--steps',type=int,default=720);ap.add_argument('--obs-cpp',default='scratch/bonsai_kg_v6_translation/stateful_v6_observation_cpp');ap.add_argument('--cpp',default='scratch/bonsai_kg_v6_translation/stateful_v6_game_cpp');ap.add_argument('--cuda',default='scratch/bonsai_kg_v6_translation/stateful_v6_game_cuda');raise SystemExit(main(ap.parse_args()))
