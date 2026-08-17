from __future__ import annotations

import argparse
import json
import random
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE.parent / "python"))
import bonsai_kg as bkg

BB = [
    {"name":"counter","type":"I32","initial":0},
    {"name":"flag","type":"BOOL","initial":0},
    {"name":"quota","type":"U16","initial":3},
]


def P(tree, bb=BB): return {"schema_version":1,"blackboard":bb,"tree":tree}

def cond(feature="STEP", op="GE", rhs=1, arg=None):
    lhs={"feature":feature}
    if arg is not None: lhs["arg"]=arg
    return {"type":"Condition","lhs":lhs,"op":op,"rhs":rhs}


def coverage_policies():
    wait={"type":"WaitTurns","turns":3}
    return [
        P({"type":"Sequence","children":[{"type":"Success"},wait]}),
        P({"type":"Selector","children":[{"type":"Failure"},wait]}),
        P({"type":"ReactiveSequence","children":[cond("STEP","LT",4),wait]}),
        P({"type":"ReactiveSelector","children":[cond("STEP","GE",4),wait]}),
        P({"type":"If","condition":cond("STEP","LT",2),"then":wait,"else":{"type":"Success"}}),
        P({"type":"Parallel","success_threshold":2,"failure_threshold":2,"children":[wait,{"type":"Success"},{"type":"Failure"}]}),
        P({"type":"Invert","child":{"type":"Failure"}}),
        P({"type":"ForceSuccess","child":{"type":"Failure"}}),
        P({"type":"ForceFailure","child":{"type":"Success"}}),
        P({"type":"TurnTimeout","turns":2,"child":{"type":"WaitTurns","turns":8}}),
        P({"type":"Retry","n":3,"child":cond("STEP","GE",2)}),
        P({"type":"Repeat","n":3,"child":{"type":"Success"}}),
        P({"type":"KeepRunningUntilFailure","child":cond("STEP","LT",2)}),
        P({"type":"Success"}),
        P({"type":"Failure"}),
        P(cond("MARKET_PRICE","GT",20,0)),
        P({"type":"SetBlackboard","slot":"counter","value":7}),
        P({"type":"AddBlackboard","slot":"counter","value":2}),
        P({"type":"EmitUnitIntent","unit":0,"action":"WEST","priority":7}),
        P({"type":"EmitMarketIntent","action":"HIRE","priority":8}),
        P({"type":"WaitTurns","turns":2}),
        P({"type":"MoveUnitTo","unit":0,"x":0,"y":0,"priority":9}),
        P({"type":"WaitTurns","turns":9,"lifecycle":{"failure_if":{"lhs":{"feature":"STEP"},"op":"EQ","rhs":8},"success_if":{"lhs":{"feature":"STEP"},"op":"EQ","rhs":9},"skip_if":{"lhs":{"feature":"STEP"},"op":"EQ","rhs":10},"while":{"lhs":{"feature":"STEP"},"op":"LT","rhs":4},"on_success":[{"op":"set","slot":"counter","value":1}],"on_failure":[{"op":"set","slot":"counter","value":2}],"on_halted":[{"op":"clear","slot":"counter"}],"post":[{"op":"add","slot":"counter","value":1}]}}),
        P({"type":"Sequence","children":[{"type":"EmitMarketIntent","action":"SELL","item":"WHEAT","quantity":i+1,"priority":i} for i in range(20)]}),
        P({"type":"Parallel","success_threshold":3,"failure_threshold":3,"children":[{"type":"EmitMarketIntent","action":"HIRE","priority":10},{"type":"EmitMarketIntent","action":"BUY_LAND","priority":30},{"type":"EmitMarketIntent","action":"HIRE","priority":20}]}),
    ]


FEATURE_CHOICES = [
    ("STEP",None),("DAY",None),("HOUR",None),("OWN_MONEY",None),("OPP_MONEY",None),
    ("OWN_UNITS",None),("OPP_UNITS",None),("OWN_LAND",None),("OPP_LAND",None),
    ("OWN_HIRES_TODAY",None),("MARKET_PRICE","product"),("MARKET_INVENTORY","product"),
    ("OWN_SHED","item"),("OWN_SEED","crop"),("SHOP_COUNT",None),("SHOP_SEEN","shop"),
    ("OWN_WORKER_X","unit"),("OWN_WORKER_Y","unit"),("OPP_WORKER_X","unit"),("OPP_WORKER_Y","unit"),
]

LEAF_TYPES=["Success","Failure","Condition","SetBlackboard","AddBlackboard","EmitUnitIntent","EmitMarketIntent","WaitTurns","MoveUnitTo"]
COMPOSITES=["Sequence","Selector","ReactiveSequence","ReactiveSelector","If","Parallel"]
DECORATORS=["Invert","ForceSuccess","ForceFailure","TurnTimeout","Retry","Repeat","KeepRunningUntilFailure"]


def random_pred(r):
    if r.random()<0.22:
        return {"lhs":{"blackboard":r.choice(["counter","flag","quota"])},"op":r.choice(list(bkg.CMP)),"rhs":r.randint(0,5)}
    name,kind=r.choice(FEATURE_CHOICES)
    arg=None
    if kind=="product":arg=r.randrange(9)
    elif kind=="item":arg=r.randrange(12)
    elif kind=="crop":arg=r.randrange(5)
    elif kind=="shop":arg=r.randrange(8)
    elif kind=="unit":arg=r.randrange(4)
    lhs={"feature":name}
    if arg is not None: lhs["arg"]=arg
    return {"lhs":lhs,"op":r.choice(list(bkg.CMP)),"rhs":r.randint(-5,12000)}


def maybe_lifecycle(r,node):
    if r.random()>=0.18:return node
    lc={}
    if r.random()<0.55: lc["while"]={"lhs":{"feature":"STEP"},"op":"LT","rhs":r.randint(2,13)}
    if r.random()<0.15: lc["failure_if"]={"lhs":{"feature":"STEP"},"op":"EQ","rhs":r.randint(0,11)}
    if r.random()<0.15: lc["success_if"]={"lhs":{"feature":"STEP"},"op":"EQ","rhs":r.randint(0,11)}
    if r.random()<0.15: lc["skip_if"]={"lhs":{"feature":"STEP"},"op":"EQ","rhs":r.randint(0,11)}
    if r.random()<0.5: lc["on_halted"]=[{"op":"clear","slot":"counter"}]
    if r.random()<0.25: lc["on_success"]=[{"op":"add","slot":"counter","value":1}]
    if r.random()<0.25: lc["on_failure"]=[{"op":"set","slot":"flag","value":1}]
    if r.random()<0.25: lc["post"]=[{"op":"add","slot":"counter","value":1}]
    if lc: node["lifecycle"]=lc
    return node


def gen_leaf(r):
    t=r.choice(LEAF_TYPES)
    if t in ("Success","Failure"):node={"type":t}
    elif t=="Condition":
        pr=random_pred(r);node={"type":"Condition",**pr}
    elif t=="SetBlackboard":
        slot=r.choice(["counter","flag","quota"]);value=r.randint(-5,20) if slot=="counter" else r.randint(0,1) if slot=="flag" else r.randint(0,20);node={"type":t,"slot":slot,"value":value}
    elif t=="AddBlackboard":node={"type":t,"slot":"counter","value":r.randint(-3,4)}
    elif t=="EmitUnitIntent":node={"type":t,"unit":0,"action":r.choice(["NORTH","SOUTH","EAST","WEST","WATER","HARVEST","DIG"]),"priority":r.randrange(256)}
    elif t=="EmitMarketIntent":
        action=r.choice(["HIRE","BUY_LAND","BUY_SEED","BUY_PRODUCT","BUY_ANIMAL","SELL"]);node={"type":t,"action":action,"priority":r.randrange(256)}
        if action=="BUY_SEED":node.update(item=r.choice(bkg.ITEMS[:5]),quantity=r.randint(1,4))
        elif action=="BUY_PRODUCT":node.update(item=r.choice(["WHEAT","FERTILIZER"]),quantity=r.randint(1,4))
        elif action=="BUY_ANIMAL":node.update(item=r.choice(["GOOSE","COW","SHEEP"]),quantity=r.randint(1,2))
        elif action=="SELL":node.update(item=r.choice(bkg.ITEMS[:9]),quantity=r.randint(1,5))
    elif t=="WaitTurns":node={"type":t,"turns":r.randint(1,6)}
    else:node={"type":t,"unit":0,"x":r.randrange(10),"y":r.randrange(10),"priority":r.randrange(256)}
    return maybe_lifecycle(r,node)


def gen_node(r,depth=0):
    if depth>=3 or r.random()<0.48:return gen_leaf(r)
    if r.random()<0.62:
        t=r.choice(COMPOSITES)
        if t=="If":node={"type":t,"condition":gen_node(r,depth+1),"then":gen_node(r,depth+1),"else":gen_node(r,depth+1)}
        else:
            n=r.randint(2,4);node={"type":t,"children":[gen_node(r,depth+1) for _ in range(n)]}
            if t=="Parallel":node.update(success_threshold=r.randint(1,n),failure_threshold=r.randint(1,n))
    else:
        t=r.choice(DECORATORS);node={"type":t,"child":gen_node(r,depth+1)}
        if t=="TurnTimeout":node["turns"]=r.randint(1,6)
        elif t in ("Retry","Repeat"):node["n"]=r.randint(1,5)
    return maybe_lifecycle(r,node)


def frame_values(r,t):
    vals={(bkg.FEATURES["STEP"],0):t,(bkg.FEATURES["DAY"],0):t//24,(bkg.FEATURES["HOUR"],0):t%24,
          (bkg.FEATURES["OWN_MONEY"],0):r.randint(0,20000),(bkg.FEATURES["OPP_MONEY"],0):r.randint(0,20000),
          (bkg.FEATURES["OWN_UNITS"],0):r.randint(1,4),(bkg.FEATURES["OPP_UNITS"],0):r.randint(1,4),
          (bkg.FEATURES["OWN_LAND"],0):r.randint(1,4),(bkg.FEATURES["OPP_LAND"],0):r.randint(1,4),
          (bkg.FEATURES["OWN_HIRES_TODAY"],0):r.randint(0,6),(bkg.FEATURES["OWN_SHED_TOTAL"],0):r.randint(0,100),
          (bkg.FEATURES["SHOP_COUNT"],0):r.randint(0,8)}
    for i in range(9): vals[(bkg.FEATURES["MARKET_PRICE"],i)]=r.randint(1,500);vals[(bkg.FEATURES["MARKET_INVENTORY"],i)]=r.randint(9000,11000)
    for i in range(12): vals[(bkg.FEATURES["OWN_SHED"],i)]=r.randint(0,30)
    for i in range(5): vals[(bkg.FEATURES["OWN_SEED"],i)]=r.randint(0,20)
    for i in range(8): vals[(bkg.FEATURES["SHOP_AT"],i)]=r.randint(-1,7);vals[(bkg.FEATURES["SHOP_SEEN"],i)]=r.randint(0,3)
    for u in range(4):
        vals[(bkg.FEATURES["OWN_WORKER_X"],u)]=r.randrange(10);vals[(bkg.FEATURES["OWN_WORKER_Y"],u)]=r.randrange(10)
        vals[(bkg.FEATURES["OPP_WORKER_X"],u)]=r.randrange(10);vals[(bkg.FEATURES["OPP_WORKER_Y"],u)]=r.randrange(10)
    return vals


def write_seq(path,frames):
    with open(path,"w",encoding="utf-8") as f:
        f.write(f"BKS1 1 {len(frames)}\n")
        for vals in frames:
            items=sorted(vals.items());f.write(f"{len(items)}\n")
            for (fid,arg),v in items:f.write(f"{fid} {arg} {v}\n")


def py_line(ci,ti,res):
    a=res.action
    units=';'.join(f"{op},{arg},{n}" for op,arg,n in a.units)
    market=';'.join(f"{op},{item},{n}" for op,item,n in a.market)
    bb=','.join(str(x) for x in res.blackboard)
    ns=','.join(f"{x}/{y}" for x,y in res.node_state)
    return f"C={ci} T={ti} S={res.status} F={res.fault} U={a.n_units}:{units} M={len(a.market)}:{market} B={bb} N={ns}"


def node_counts(obj,counter):
    if not isinstance(obj,dict):return
    if "type" in obj:counter[obj["type"]]+=1
    for k in ("children",):
        for c in obj.get(k,[]) or []:node_counts(c,counter)
    for k in ("child","condition","then","else"):
        if k in obj:node_counts(obj[k],counter)


def run(args):
    scratch=(ROOT/args.scratch).resolve();
    if scratch.exists():shutil.rmtree(scratch)
    scratch.mkdir(parents=True)
    rng=random.Random(args.seed);policies=coverage_policies();attempts=0
    while len(policies)<args.cases:
        attempts+=1
        candidate=P(gen_node(rng))
        try:
            cp=bkg.compile_policy(candidate)
            if len(cp.nodes)<=48 and cp.state_count<=32:policies.append(candidate)
        except bkg.PolicyError:pass
        if attempts>args.cases*100:raise RuntimeError("failed to generate enough valid random policies")
    policies=policies[:args.cases]
    manifest=[];expected=[];coverage=Counter();total_nodes=0;max_nodes=0;max_state=0;compiled_bytes=0
    for ci,p in enumerate(policies):
        cp=bkg.compile_policy(p);node_counts(p["tree"],coverage);total_nodes+=len(cp.nodes);max_nodes=max(max_nodes,len(cp.nodes));max_state=max(max_state,cp.state_count);compiled_bytes+=cp.compiled_bytes
        ppath=scratch/f"case_{ci:04d}.bkgc";spath=scratch/f"case_{ci:04d}.bks";bkg.save_compiled(cp,ppath)
        frames=[frame_values(rng,t) for t in range(args.ticks)];write_seq(spath,frames);manifest.append((ppath,spath))
        vm=bkg.VM(cp)
        for ti,vals in enumerate(frames):expected.append(py_line(ci,ti,vm.tick(bkg.SyntheticObservation(vals),10,False)))
    mpath=scratch/"manifest.txt"
    with open(mpath,"w",encoding="utf-8") as f:
        for p,s in manifest:f.write(f"{p.relative_to(ROOT)} {s.relative_to(ROOT)}\n")
    def invoke(exe):
        cp=subprocess.run([str(ROOT/exe),str(mpath.relative_to(ROOT))],cwd=ROOT,text=True,capture_output=True,check=False)
        if cp.returncode!=0:raise RuntimeError(f"{exe} failed rc={cp.returncode}: {cp.stderr[-4000:]}")
        return cp.stdout.splitlines()
    cpp=invoke(args.cpp);cuda=invoke(args.cuda)
    mismatches=[]
    for engine,lines in (("cpp",cpp),("cuda",cuda)):
        if len(lines)!=len(expected):mismatches.append((engine,"record_count",len(expected),len(lines)))
        for i,(a,b) in enumerate(zip(expected,lines)):
            if a!=b:
                mismatches.append((engine,i,a,b))
                if len(mismatches)>=10:break
    summary={"seed":args.seed,"policies":len(policies),"ticks_per_policy":args.ticks,"tick_records":len(expected),
             "pairwise_engine_comparisons":len(expected)*2,"mismatches":len(mismatches),"generator_attempts":attempts,
             "total_nodes":total_nodes,"max_nodes":max_nodes,"max_state_slots":max_state,"mean_compiled_bytes":compiled_bytes/len(policies),
             "node_type_coverage":dict(sorted(coverage.items()))}
    print(json.dumps(summary,sort_keys=True))
    for m in mismatches[:5]:print("MISMATCH",repr(m))
    return 1 if mismatches else 0


if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--cpp",default="scratch/bonsai_kg/cpp_differential_runner");ap.add_argument("--cuda",default="scratch/bonsai_kg/cuda_differential_runner");ap.add_argument("--scratch",default="scratch/bonsai_kg/differential");ap.add_argument("--cases",type=int,default=256);ap.add_argument("--ticks",type=int,default=12);ap.add_argument("--seed",type=int,default=20260817);args=ap.parse_args();raise SystemExit(run(args))
