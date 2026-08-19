#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
import subprocess
import sys
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(HERE.parent/"python"))
import bonsai_kg_v4 as b


def cond(r):
    feature=r.choice(["STEP","HOUR","OWN_MONEY","OPP_MONEY","OWN_UNITS","MARKET_INVENTORY","MARKET_PRICE"])
    if feature in ("MARKET_INVENTORY","MARKET_PRICE"):
        lhs={"feature":feature,"arg":{"const":r.randrange(0,9),"type":"ITEM"}}
    else: lhs={"feature":feature}
    return {"type":"Condition","lhs":lhs,"op":r.choice(["LT","LE","GE","GT","EQ","NE"]),"rhs":r.randrange(0,120)}


def leaf(r, allow_running=True):
    choices=["success","failure","condition","set","add","unit","market"]
    if allow_running: choices += ["wait","move"]
    k=r.choice(choices)
    if k=="success":return {"type":"Success"}
    if k=="failure":return {"type":"Failure"}
    if k=="condition":return cond(r)
    if k=="set":return {"type":"SetBlackboard","slot":"x","value":r.randrange(-3,8)}
    if k=="add":return {"type":"AddBlackboard","slot":"x","value":r.randrange(-2,4)}
    if k=="unit":return {"type":"EmitUnitIntent","unit":{"const":r.randrange(0,2),"type":"UNIT"},"action":r.choice(["NORTH","SOUTH","EAST","WEST","PASS"]),"authority":r.choice(list(b.AUTHORITY)),"priority":r.randrange(0,256)}
    if k=="market":return {"type":"EmitMarketIntent","action":r.choice(["HIRE","BUY_LAND"]),"authority":r.choice(list(b.AUTHORITY)),"priority":r.randrange(0,256)}
    if k=="wait":return {"type":"WaitTurns","turns":r.randrange(1,5)}
    return {"type":"MoveUnitTo","unit":{"const":r.randrange(0,2),"type":"UNIT"},"target":{"const":r.randrange(0,100),"type":"TILE"},"authority":r.choice(list(b.AUTHORITY)),"priority":r.randrange(0,256)}


def tree(r,depth=0,max_depth=4,allow_running=True):
    if depth>=max_depth or r.random()<0.48:return leaf(r,allow_running)
    kind=r.choice(["Sequence","Selector","ReactiveSequence","ReactiveSelector","If","ReactiveIf","SwitchEnum","Parallel","Invert","ForceSuccess","TurnTimeout","Retry","Repeat","KeepRunningUntilFailure"])
    if kind in ("Sequence","Selector","ReactiveSequence","ReactiveSelector"):
        return {"type":kind,"children":[tree(r,depth+1,max_depth,allow_running) for _ in range(r.randrange(1,4))]}
    if kind=="If":return {"type":"If","condition":tree(r,depth+1,max_depth,allow_running),"then":tree(r,depth+1,max_depth,allow_running),"else":tree(r,depth+1,max_depth,allow_running)}
    if kind=="ReactiveIf":return {"type":"ReactiveIf","condition":cond(r),"then":tree(r,depth+1,max_depth,allow_running),"else":tree(r,depth+1,max_depth,allow_running)}
    if kind=="SwitchEnum":return {"type":"SwitchEnum","slot":"mode","cases":{"A":tree(r,depth+1,max_depth,allow_running),"B":tree(r,depth+1,max_depth,allow_running),"C":tree(r,depth+1,max_depth,allow_running)},"default":leaf(r,allow_running)}
    if kind=="Parallel":
        n=r.randrange(2,5);return {"type":"Parallel","children":[tree(r,depth+1,max_depth,allow_running) for _ in range(n)],"success_threshold":r.randrange(1,n+1),"failure_threshold":r.randrange(1,n+1)}
    child=tree(r,depth+1,max_depth,allow_running)
    if kind in ("Invert","ForceSuccess","KeepRunningUntilFailure"):return {"type":kind,"child":child}
    if kind=="TurnTimeout":return {"type":kind,"turns":r.randrange(1,5),"child":child}
    return {"type":kind,"n":r.randrange(1,4),"child":child}


def maybe_lifecycle(r,node):
    if r.random()>0.18:return node
    node=dict(node);node["lifecycle"]={"while":{"lhs":{"feature":"STEP"},"op":"LT","rhs":r.randrange(2,10)},"on_halted":[{"op":"add","slot":"x","value":1}]};return node


def policy(r):
    t=maybe_lifecycle(r,tree(r))
    return {"schema_version":4,"enums":[{"name":"Mode","values":["A","B","C"]}],"blackboard":[{"name":"mode","type":"Mode","initial":r.choice(["A","B","C"])},{"name":"x","type":"I32","initial":r.randrange(-2,3)}],"tree":t}


def frame_values(r,step):
    d={(b.FEATURES["STEP"],0,0):step,(b.FEATURES["HOUR"],0,0):step%24,(b.FEATURES["DAY"],0,0):step//24,(b.FEATURES["OWN_UNITS"],0,0):2,(b.FEATURES["OPP_UNITS"],0,0):r.randrange(1,9),(b.FEATURES["OWN_MONEY"],0,0):r.randrange(0,800),(b.FEATURES["OPP_MONEY"],0,0):r.randrange(0,800)}
    for item in range(9):d[(b.FEATURES["MARKET_PRICE"],item,0)]=r.randrange(10,180);d[(b.FEATURES["MARKET_INVENTORY"],item,0)]=r.randrange(0,80)
    for u in range(2):d[(b.FEATURES["OWN_WORKER_X"],u,0)]=r.randrange(0,10);d[(b.FEATURES["OWN_WORKER_Y"],u,0)]=r.randrange(0,10)
    return d


def write_seq(path,frames):
    lines=[f"BKS2 2 {len(frames)}"]
    for f in frames:
        lines.append(str(len(f)))
        for (fid,a0,a1),v in sorted(f.items()):lines.append(f"{fid} {a0} {a1} {v}")
    path.write_text("\n".join(lines)+"\n")


def line(ci,ti,res):
    a=res.action;s=f"C={ci} T={ti} S={res.status} F={res.fault} U={a.n_units}:"+";".join(f"{op},{arg},{n}" for op,arg,n in a.units)
    s+=f" M={len(a.market)}:"+";".join(f"{op},{item},{n}" for op,item,n in a.market)
    s+=" B="+",".join(str(x) for x in res.blackboard)
    s+=" N="+",".join(f"{x}/{y}" for x,y in res.node_state)
    return s


def targeted():
    # New-v4 preemption and state-elision patterns are always represented even when
    # random generation happens not to sample them.
    base={"schema_version":4,"enums":[{"name":"Mode","values":["A","B","C"]}],"blackboard":[{"name":"mode","type":"Mode","initial":"A"},{"name":"x","type":"I32","initial":0}]}
    xs=[]
    p=dict(base);p["tree"]={"type":"SwitchEnum","slot":"mode","cases":{"A":{"type":"WaitTurns","turns":4},"B":{"type":"Success"},"C":{"type":"Failure"}},"default":{"type":"Failure"}};xs.append(p)
    p=dict(base);p["tree"]={"type":"ReactiveIf","condition":{"type":"Condition","lhs":{"feature":"STEP"},"op":"LT","rhs":2},"then":{"type":"WaitTurns","turns":8},"else":{"type":"Success"}};xs.append(p)
    p=dict(base);p["tree"]={"type":"Parallel","success_threshold":2,"failure_threshold":2,"children":[{"type":"EmitUnitIntent","unit":{"const":0,"type":"UNIT"},"action":"NORTH","authority":"TACTICAL","priority":255},{"type":"EmitUnitIntent","unit":{"const":0,"type":"UNIT"},"action":"SOUTH","authority":"INVARIANT","priority":0}]};xs.append(p)
    p=dict(base);p["tree"]={"type":"Parallel","success_threshold":1,"failure_threshold":2,"children":[{"type":"EmitUnitIntent","unit":{"const":0,"type":"UNIT"},"action":"WEST","authority":"RECOVERY","priority":101},{"type":"EmitMarketIntent","action":"HIRE","authority":"TACTICAL","priority":187},{"type":"EmitMarketIntent","action":"BUY_LAND","authority":"TACTICAL","priority":117},{"type":"Success"}]};xs.append(p)
    return xs


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--cases",type=int,default=128);ap.add_argument("--ticks",type=int,default=8);ap.add_argument("--seed",type=int,default=440004);ap.add_argument("--work",default="scratch/bonsai_kg/v4_diff");ap.add_argument("--cpp");ap.add_argument("--cuda");args=ap.parse_args();work=Path(args.work);work=work if work.is_absolute() else ROOT/work;work.mkdir(parents=True,exist_ok=True);r=random.Random(args.seed);policies=targeted();attempts=0
    while len(policies)<args.cases and attempts<args.cases*100:
        attempts+=1
        try:p=policy(r);b.compile_policy(p);policies.append(p)
        except b.PolicyError:pass
    if len(policies)<args.cases:raise RuntimeError("could not generate requested valid policies")
    manifest=[];expected=[];stats={"nodes":0,"states":0,"max_nodes":0,"max_state":0,"max_depth":0,"max_live":0,"max_visits":0}
    for ci,p in enumerate(policies):
        cp=b.compile_policy(p);stats["nodes"]+=len(cp.nodes);stats["states"]+=cp.state_count;stats["max_nodes"]=max(stats["max_nodes"],len(cp.nodes));stats["max_state"]=max(stats["max_state"],cp.state_count);stats["max_depth"]=max(stats["max_depth"],cp.max_depth);stats["max_live"]=max(stats["max_live"],cp.max_live_intents);stats["max_visits"]=max(stats["max_visits"],cp.analysis[0].max_visits)
        pf=work/f"p{ci:04d}.bkgc";sf=work/f"s{ci:04d}.bks";b.save_compiled(cp,pf);frames=[frame_values(r,t) for t in range(args.ticks)];write_seq(sf,frames);manifest.append((pf,sf));vm=b.VM(cp)
        for ti,f in enumerate(frames):expected.append(line(ci,ti,vm.tick(b.SyntheticObservation({(fid,a0):v for (fid,a0,a1),v in f.items()}))))
    mf=work/"manifest.txt";mf.write_text("".join(f"{a} {s}\n" for a,s in manifest));mismatches={}
    for name,exe in (("cpp",args.cpp),("cuda",args.cuda)):
        if not exe:continue
        got=subprocess.run([exe,str(mf)],check=True,capture_output=True,text=True).stdout.splitlines();bad=[]
        if len(got)!=len(expected):bad.append(("length",len(expected),len(got)))
        for i,(a,z) in enumerate(zip(expected,got)):
            if a!=z:bad.append((i,a,z))
            if len(bad)>=10:break
        mismatches[name]=bad
    print({"seed":args.seed,"cases":len(policies),"ticks":args.ticks,"records":len(expected),"attempts":attempts,**stats,"mismatches":{k:len(v) for k,v in mismatches.items()}})
    for eng,bad in mismatches.items():
        for x in bad[:3]:print("MISMATCH",eng,repr(x))
    return 1 if any(mismatches.values()) else 0

if __name__=="__main__":raise SystemExit(main())
