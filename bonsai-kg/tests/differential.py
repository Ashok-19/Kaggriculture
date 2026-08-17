from __future__ import annotations

import argparse
import json
import random
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(HERE.parent/"python"))
import bonsai_kg as bkg

BB=[
 {"name":"counter","type":"I32","initial":0},
 {"name":"flag","type":"BOOL","initial":0},
 {"name":"quota","type":"U16","initial":3},
 {"name":"unit","type":"UNIT","initial":0},
 {"name":"target","type":"TILE","initial":45},
 {"name":"item","type":"ITEM","initial":0},
 {"name":"qty","type":"I32","initial":3},
]

def P(tree,bb=BB,subtrees=None):
 d={"schema_version":3,"blackboard":bb,"tree":tree}
 if subtrees is not None:d["subtrees"]=subtrees
 return d

def cond(feature="STEP",op="GE",rhs=1,arg=None,args=None):
 lhs={"feature":feature}
 if arg is not None:lhs["arg"]=arg
 if args is not None:lhs["args"]=args
 return {"type":"Condition","lhs":lhs,"op":op,"rhs":rhs}

def port_seed_policies():
 move=[{"name":"MoveAndWater","parameters":{"worker":{"type":"UNIT","mode":"in"},"target":{"type":"TILE","mode":"in"}},"tree":{"type":"Sequence","children":[
  {"type":"MoveUnitTo","unit":{"binding":"worker"},"target":{"binding":"target"}},
  {"type":"Condition","lhs":{"feature":"OWN_TILE_WATERED","arg":{"binding":"target"}},"op":"EQ","rhs":0},
  {"type":"EmitUnitIntent","unit":{"binding":"worker"},"action":"WATER"}]}}]
 producer=[
  {"name":"Produce","parameters":{"dst":{"type":"TILE","mode":"out"}},"tree":{"type":"SetBlackboard","slot":{"binding":"dst"},"value":45}},
  {"name":"Forward","parameters":{"out":{"type":"TILE","mode":"out"}},"tree":{"type":"Subtree","subtree":"Produce","bindings":{"dst":{"binding":"out"}}}},
 ]
 bump=[{"name":"Bump","parameters":{"x":{"type":"I32","mode":"inout"}},"tree":{"type":"Sequence","children":[
  {"type":"AddBlackboard","slot":{"binding":"x"},"value":1},
  {"type":"Success","lifecycle":{"on_success":[{"op":"add","slot":{"binding":"x"},"value":1}]}}]}}]
 return [
  P({"type":"Subtree","subtree":"MoveAndWater","bindings":{"worker":{"blackboard":"unit"},"target":{"blackboard":"target"}}},subtrees=move),
  P({"type":"Sequence","children":[{"type":"Subtree","subtree":"Forward","bindings":{"out":{"blackboard":"target"}}},{"type":"Condition","lhs":{"feature":"OWN_TILE_WATERED","arg":{"blackboard":"target"}},"op":"EQ","rhs":0}]},subtrees=producer),
  P({"type":"Subtree","subtree":"Bump","bindings":{"x":{"blackboard":"counter"}}},subtrees=bump),
 ]

def coverage_policies():
 wait={"type":"WaitTurns","turns":3}
 out=[
  P({"type":"Sequence","children":[{"type":"Success"},wait]}),P({"type":"Selector","children":[{"type":"Failure"},wait]}),
  P({"type":"ReactiveSequence","children":[cond("STEP","LT",4),wait]}),P({"type":"ReactiveSelector","children":[cond("STEP","GE",4),wait]}),
  # Reactive stale-halt freeze seeds: the remembered child is revisited and completes
  # normally on tick 1, versus genuine preemption before that old child is reached.
  P({"type":"ReactiveSequence","children":[{"type":"Retry","n":2,"child":{"type":"Sequence","children":[cond("STEP","GE",1),{"type":"EmitMarketIntent","action":"HIRE","priority":90}]}},{"type":"WaitTurns","turns":5}]}),
  P({"type":"ReactiveSequence","children":[{"type":"Retry","n":2,"child":{"type":"Sequence","children":[cond("STEP","GE",1),{"type":"EmitUnitIntent","unit":0,"action":"NORTH","priority":90}]}},{"type":"WaitTurns","turns":5}]}),
  P({"type":"ReactiveSequence","children":[{"type":"Retry","n":2,"child":{"type":"Sequence","children":[cond("STEP","GE",1),{"type":"EmitMarketIntent","action":"HIRE","priority":89}]}},{"type":"Success"}]}),
  P({"type":"ReactiveSequence","children":[{"type":"Retry","n":2,"lifecycle":{"on_halted":[{"op":"set","slot":"counter","value":7}]},"child":{"type":"Sequence","children":[cond("STEP","GE",1),{"type":"EmitMarketIntent","action":"HIRE","priority":88}]}},{"type":"WaitTurns","turns":5}]}),
  P({"type":"ReactiveSequence","children":[cond("STEP","LT",1),{"type":"Sequence","lifecycle":{"on_halted":[{"op":"set","slot":"counter","value":8}]},"children":[{"type":"EmitMarketIntent","action":"HIRE","priority":87},{"type":"WaitTurns","turns":5}]}]}),
  P({"type":"ReactiveSelector","children":[{"type":"Retry","n":2,"lifecycle":{"on_halted":[{"op":"set","slot":"counter","value":9}]},"child":{"type":"Failure"}},{"type":"EmitMarketIntent","action":"HIRE","priority":86}]}),
  P({"type":"If","condition":cond("STEP","LT",2),"then":wait,"else":{"type":"Success"}}),
  P({"type":"Parallel","success_threshold":2,"failure_threshold":2,"children":[wait,{"type":"Success"},{"type":"Failure"}]}),
  P({"type":"Invert","child":{"type":"Failure"}}),P({"type":"ForceSuccess","child":{"type":"Failure"}}),P({"type":"ForceFailure","child":{"type":"Success"}}),
  P({"type":"TurnTimeout","turns":2,"child":{"type":"WaitTurns","turns":8}}),P({"type":"Retry","n":3,"child":cond("STEP","GE",2)}),
  P({"type":"Repeat","n":3,"child":{"type":"Success"}}),P({"type":"KeepRunningUntilFailure","child":cond("STEP","LT",2)}),
  P({"type":"Success"}),P({"type":"Failure"}),P(cond("MARKET_PRICE","GT",20,"WHEAT")),
  P({"type":"SetBlackboard","slot":"counter","value":7}),P({"type":"AddBlackboard","slot":"counter","value":2}),
  P({"type":"EmitUnitIntent","unit":{"blackboard":"unit"},"action":"WEST","priority":7}),
  P({"type":"EmitUnitIntent","unit":{"blackboard":"unit"},"action":"PICKUP","item":{"blackboard":"item"},"quantity":{"blackboard":"qty"},"priority":17}),
  P({"type":"EmitMarketIntent","action":"SELL","item":{"blackboard":"item"},"quantity":{"blackboard":"qty"},"priority":8}),
  P({"type":"Sequence","children":[{"type":"SetBlackboard","slot":"qty","value":0},{"type":"EmitMarketIntent","action":"SELL","item":{"blackboard":"item"},"quantity":{"blackboard":"qty"},"priority":18}]}),
  P({"type":"Sequence","children":[{"type":"SetBlackboard","slot":"item","value":{"const":"GOOSE","type":"ITEM"}},{"type":"EmitMarketIntent","action":"SELL","item":{"blackboard":"item"},"quantity":1,"priority":19}]}),
  P({"type":"Sequence","children":[{"type":"SetBlackboard","slot":"unit","value":{"const":-1,"type":"UNIT"}},{"type":"EmitUnitIntent","unit":{"blackboard":"unit"},"action":"WATER","priority":20}]}),
  P({"type":"WaitTurns","turns":2}),P({"type":"MoveUnitTo","unit":{"blackboard":"unit"},"target":{"blackboard":"target"},"priority":9}),
  P({"type":"WaitTurns","turns":9,"lifecycle":{"failure_if":{"lhs":{"feature":"STEP"},"op":"EQ","rhs":8},"success_if":{"lhs":{"feature":"STEP"},"op":"EQ","rhs":9},"skip_if":{"lhs":{"feature":"STEP"},"op":"EQ","rhs":10},"while":{"lhs":{"feature":"STEP"},"op":"LT","rhs":4},"on_success":[{"op":"set","slot":"counter","value":1}],"on_failure":[{"op":"set","slot":"counter","value":2}],"on_halted":[{"op":"clear","slot":"counter"}],"post":[{"op":"add","slot":"counter","value":1}]}}),
  P({"type":"Sequence","children":[{"type":"EmitMarketIntent","action":"SELL","item":"WHEAT","quantity":i+1,"priority":i} for i in range(20)]}),
  P({"type":"Selector","children":[{"type":"Sequence","children":[{"type":"EmitMarketIntent","action":"HIRE","priority":200},{"type":"Failure"}]},{"type":"EmitMarketIntent","action":"BUY_LAND","priority":1}]}),
  P({"type":"Selector","children":[{"type":"Sequence","children":[{"type":"EmitUnitIntent","unit":0,"action":"NORTH","priority":200},{"type":"Failure"}]},{"type":"EmitUnitIntent","unit":0,"action":"SOUTH","priority":1}]}),
  # Exact same-tick cancellation seeds.
  P({"type":"Parallel","success_threshold":1,"failure_threshold":2,"children":[{"type":"Sequence","children":[{"type":"EmitMarketIntent","action":"HIRE","priority":99},{"type":"WaitTurns","turns":5}]},{"type":"Success"}]}),
  P({"type":"Parallel","success_threshold":1,"failure_threshold":2,"children":[{"type":"Sequence","children":[{"type":"EmitUnitIntent","unit":0,"action":"NORTH","priority":99},{"type":"WaitTurns","turns":5}]},{"type":"Success"}]}),
  P({"type":"Parallel","success_threshold":1,"failure_threshold":2,"children":[{"type":"Parallel","success_threshold":2,"failure_threshold":2,"children":[{"type":"Sequence","children":[{"type":"EmitMarketIntent","action":"HIRE","priority":90},{"type":"WaitTurns","turns":5}]},{"type":"WaitTurns","turns":5}]},{"type":"Success"}]}),
  P({"type":"Parallel","success_threshold":2,"failure_threshold":2,"children":[{"type":"Sequence","children":[{"type":"EmitMarketIntent","action":"HIRE","priority":90},{"type":"WaitTurns","turns":5}]},{"type":"WaitTurns","turns":5}]}),
  P({"type":"Parallel","success_threshold":1,"failure_threshold":2,"children":[{"type":"EmitMarketIntent","action":"HIRE","priority":80},{"type":"WaitTurns","turns":5}]}),
  P({"type":"Parallel","success_threshold":2,"failure_threshold":1,"children":[{"type":"Sequence","children":[{"type":"EmitMarketIntent","action":"HIRE","priority":79},{"type":"WaitTurns","turns":5}]},{"type":"Failure"}]}),
  P({"type":"Parallel","success_threshold":1,"failure_threshold":2,"children":[{"type":"Sequence","lifecycle":{"on_halted":[{"op":"set","slot":"counter","value":1}]},"children":[{"type":"EmitMarketIntent","action":"HIRE","priority":78},{"type":"WaitTurns","turns":5}]},{"type":"Success"}]}),
  P(cond("OWN_TILE_CONSECUTIVE_UNWATERED","GE",0,{"blackboard":"target"})),
  P({"type":"Condition","lhs":{"feature":"OWN_WORKER_TILE","arg":{"blackboard":"unit"}},"op":"GE","rhs":{"const":-1,"type":"TILE"}}),
  P({"type":"Condition","lhs":{"feature":"OWN_INVENTORY","args":[{"blackboard":"unit"},{"blackboard":"item"}]},"op":"GE","rhs":0}),
 ]
 out.extend(port_seed_policies())
 return out

I32_FEATURES=["STEP","DAY","HOUR","OWN_MONEY","OPP_MONEY","OWN_UNITS","OPP_UNITS","OWN_LAND","OPP_LAND","OWN_HIRES_TODAY","OPP_HIRES_TODAY","OWN_TILE_WATERED","OWN_TILE_YIELD","OWN_TILE_CONSECUTIVE_UNWATERED","MARKET_PRICE","MARKET_INVENTORY","OWN_SHED","OWN_SEED","OWN_INVENTORY"]
LEAF_TYPES=["Success","Failure","Condition","SetBlackboard","AddBlackboard","EmitUnitIntent","EmitMarketIntent","WaitTurns","MoveUnitTo"]
COMPOSITES=["Sequence","Selector","ReactiveSequence","ReactiveSelector","If","Parallel"]
DECORATORS=["Invert","ForceSuccess","ForceFailure","TurnTimeout","Retry","Repeat","KeepRunningUntilFailure"]

def dynamic_arg_for(r,name):
 if name.startswith("OWN_TILE_") or name.startswith("OPP_TILE_"):return {"blackboard":"target"} if r.random()<.65 else r.randrange(100)
 if name in ("OWN_WORKER_X","OWN_WORKER_Y","OPP_WORKER_X","OPP_WORKER_Y","OWN_WORKER_TILE","OPP_WORKER_TILE"):return {"blackboard":"unit"} if r.random()<.65 else r.randrange(4)
 if name in ("MARKET_PRICE","MARKET_INVENTORY","OWN_SHED","OWN_SEED"):return {"blackboard":"item"} if r.random()<.55 else r.randrange(5 if name=="OWN_SEED" else 9 if name.startswith("MARKET_") else 12)
 return None

def random_pred(r):
 if r.random()<.20:return {"lhs":{"blackboard":r.choice(["counter","flag","quota"])},"op":r.choice(list(bkg.CMP)),"rhs":r.randint(0,5)}
 name=r.choice(I32_FEATURES)
 lhs={"feature":name}
 if name=="OWN_INVENTORY":lhs["args"]=[{"blackboard":"unit"},{"blackboard":"item"}]
 else:
  arg=dynamic_arg_for(r,name)
  if arg is not None:lhs["arg"]=arg
 return {"lhs":lhs,"op":r.choice(list(bkg.CMP)),"rhs":r.randint(-5,12000)}

def random_i32_ref(r):
 x=r.random()
 if x<.42:return r.randint(-5,20)
 if x<.68:return {"blackboard":r.choice(["counter","flag","quota","qty"])}
 return {"feature":r.choice(["STEP","DAY","HOUR","OWN_MONEY","OPP_MONEY","OWN_HIRES_TODAY","OPP_HIRES_TODAY"])}

def random_unit_ref(r):return {"blackboard":"unit"} if r.random()<.65 else {"const":r.randrange(3),"type":"UNIT"}
def random_tile_ref(r):
 x=r.random()
 if x<.55:return {"blackboard":"target"}
 if x<.80:return {"const":r.randrange(100),"type":"TILE"}
 return {"feature":r.choice(["OWN_WORKER_TILE","OPP_WORKER_TILE"]),"arg":{"blackboard":"unit"}}
def random_item_ref(r):return {"blackboard":"item"} if r.random()<.60 else {"const":r.choice(bkg.ITEMS),"type":"ITEM"}
def random_qty_ref(r):return {"blackboard":"qty"} if r.random()<.60 else r.randint(1,5)

def maybe_lifecycle(r,node):
 if r.random()>=.20:return node
 lc={}
 if r.random()<.55:lc["while"]={"lhs":{"feature":"STEP"},"op":"LT","rhs":r.randint(2,13)}
 if r.random()<.12:lc["failure_if"]={"lhs":{"feature":"STEP"},"op":"EQ","rhs":r.randint(0,11)}
 if r.random()<.12:lc["success_if"]={"lhs":{"feature":"STEP"},"op":"EQ","rhs":r.randint(0,11)}
 if r.random()<.12:lc["skip_if"]={"lhs":{"feature":"STEP"},"op":"EQ","rhs":r.randint(0,11)}
 if r.random()<.45:lc["on_halted"]=[{"op":"clear","slot":"counter"}]
 if r.random()<.25:lc["on_success"]=[{"op":"add","slot":"counter","value":random_i32_ref(r)}]
 if r.random()<.20:lc["on_failure"]=[{"op":"set","slot":"flag","value":1}]
 if r.random()<.22:lc["post"]=[{"op":"add","slot":"counter","value":random_i32_ref(r)}]
 if lc:node["lifecycle"]=lc
 return node

def gen_leaf(r):
 t=r.choice(LEAF_TYPES)
 if t in ("Success","Failure"):node={"type":t}
 elif t=="Condition":node={"type":t,**random_pred(r)}
 elif t=="SetBlackboard":
  slot=r.choice(["counter","flag","quota"]);value=random_i32_ref(r) if slot=="counter" else r.randint(0,1) if slot=="flag" else r.randint(0,20);node={"type":t,"slot":slot,"value":value}
 elif t=="AddBlackboard":node={"type":t,"slot":"counter","value":random_i32_ref(r)}
 elif t=="EmitUnitIntent":
  action=r.choice(["NORTH","SOUTH","EAST","WEST","WATER","HARVEST","DIG","PICKUP","PLACE","PLANT"]);node={"type":t,"unit":random_unit_ref(r),"action":action,"priority":r.randrange(256)}
  if action in ("PICKUP","PLACE","PLANT"):node["item"]=random_item_ref(r)
  if action in ("PICKUP","PLACE"):node["quantity"]=random_qty_ref(r)
 elif t=="EmitMarketIntent":
  action=r.choice(["HIRE","BUY_LAND","BUY_SEED","BUY_PRODUCT","BUY_ANIMAL","SELL"]);node={"type":t,"action":action,"priority":r.randrange(256)}
  if action not in ("HIRE","BUY_LAND"):node["item"]=random_item_ref(r);node["quantity"]=random_qty_ref(r)
 elif t=="WaitTurns":node={"type":t,"turns":r.randint(1,6)}
 else:node={"type":t,"unit":random_unit_ref(r),"target":random_tile_ref(r),"priority":r.randrange(256)}
 return maybe_lifecycle(r,node)

def gen_node(r,depth=0,max_depth=5):
 if depth>=max_depth or r.random()<(0.40 if depth<3 else 0.58):return gen_leaf(r)
 if r.random()<.62:
  t=r.choice(COMPOSITES)
  if t=="If":node={"type":t,"condition":gen_node(r,depth+1,max_depth),"then":gen_node(r,depth+1,max_depth),"else":gen_node(r,depth+1,max_depth)}
  else:
   n=r.randint(2,4);node={"type":t,"children":[gen_node(r,depth+1,max_depth) for _ in range(n)]}
   if t=="Parallel":node.update(success_threshold=r.randint(1,n),failure_threshold=r.randint(1,n))
 else:
  t=r.choice(DECORATORS);node={"type":t,"child":gen_node(r,depth+1,max_depth)}
  if t=="TurnTimeout":node["turns"]=r.randint(1,6)
  elif t in ("Retry","Repeat"):node["n"]=r.randint(1,5)
 return maybe_lifecycle(r,node)

def frame_values(r,t):
 vals={(bkg.FEATURES["STEP"],0):t,(bkg.FEATURES["DAY"],0):t//24,(bkg.FEATURES["HOUR"],0):t%24,(bkg.FEATURES["OWN_MONEY"],0):r.randint(0,20000),(bkg.FEATURES["OPP_MONEY"],0):r.randint(0,20000),(bkg.FEATURES["OWN_UNITS"],0):r.randint(1,4),(bkg.FEATURES["OPP_UNITS"],0):r.randint(1,4),(bkg.FEATURES["OWN_LAND"],0):r.randint(1,4),(bkg.FEATURES["OPP_LAND"],0):r.randint(1,4),(bkg.FEATURES["OWN_HIRES_TODAY"],0):r.randint(0,6),(bkg.FEATURES["OPP_HIRES_TODAY"],0):r.randint(0,6),(bkg.FEATURES["OWN_SHED_TOTAL"],0):r.randint(0,100),(bkg.FEATURES["SHOP_COUNT"],0):r.randint(0,8)}
 for i in range(9):vals[(bkg.FEATURES["MARKET_PRICE"],i)]=r.randint(1,500);vals[(bkg.FEATURES["MARKET_INVENTORY"],i)]=r.randint(9000,11000)
 for i in range(12):vals[(bkg.FEATURES["OWN_SHED"],i)]=r.randint(0,30)
 for i in range(5):vals[(bkg.FEATURES["OWN_SEED"],i)]=r.randint(0,20)
 for i in range(8):vals[(bkg.FEATURES["SHOP_AT"],i)]=r.randint(-1,7);vals[(bkg.FEATURES["SHOP_SEEN"],i)]=r.randint(0,3)
 for u in range(4):
  ox,oy,px,py=r.randrange(10),r.randrange(10),r.randrange(10),r.randrange(10);vals[(bkg.FEATURES["OWN_WORKER_X"],u)]=ox;vals[(bkg.FEATURES["OWN_WORKER_Y"],u)]=oy;vals[(bkg.FEATURES["OWN_WORKER_TILE"],u)]=oy*10+ox;vals[(bkg.FEATURES["OPP_WORKER_X"],u)]=px;vals[(bkg.FEATURES["OPP_WORKER_Y"],u)]=py;vals[(bkg.FEATURES["OPP_WORKER_TILE"],u)]=py*10+px
  for item in range(12):vals[(bkg.FEATURES["OWN_INVENTORY"],u,item)]=r.randint(0,9)
 for tile in (0,45):
  vals[(bkg.FEATURES["OWN_TILE_WATERED"],tile)]=r.randint(0,1);vals[(bkg.FEATURES["OWN_TILE_YIELD"],tile)]=r.randint(0,6);vals[(bkg.FEATURES["OWN_TILE_CONSECUTIVE_UNWATERED"],tile)]=r.randint(0,4)
 return vals

def write_seq(path,frames):
 with open(path,"w",encoding="utf-8") as f:
  f.write(f"BKS2 2 {len(frames)}\n")
  for vals in frames:
   items=sorted(vals.items());f.write(f"{len(items)}\n")
   for key,v in items:
    if len(key)==2:fid,a0=key;a1=0
    else:fid,a0,a1=key
    f.write(f"{fid} {a0} {a1} {v}\n")

def py_line(ci,ti,res):
 a=res.action;units=';'.join(f"{op},{arg},{n}" for op,arg,n in a.units);market=';'.join(f"{op},{item},{n}" for op,item,n in a.market);bb=','.join(str(x) for x in res.blackboard);ns=','.join(f"{x}/{y}" for x,y in res.node_state);return f"C={ci} T={ti} S={res.status} F={res.fault} U={a.n_units}:{units} M={len(a.market)}:{market} B={bb} N={ns}"

def tree_depth(obj):
 if not isinstance(obj,dict):return 0
 kids=list(obj.get("children",[]) or [])
 for k in ("child","condition","then","else"):
  if k in obj:kids.append(obj[k])
 return 1+max((tree_depth(x) for x in kids),default=0)

def port_counts(policy,counter):
 for d in policy.get("subtrees",[]) or []:
  for spec in d.get("parameters",{}).values():
   mode=spec.get("mode","in") if isinstance(spec,dict) else "in";counter[mode]+=1

def run(args):
 scratch=(ROOT/args.scratch).resolve()
 if scratch.exists():shutil.rmtree(scratch)
 scratch.mkdir(parents=True)
 rng=random.Random(args.seed);policies=coverage_policies();attempts=0
 while len(policies)<args.cases:
  attempts+=1;candidate=P(gen_node(rng,max_depth=rng.choice([4,5,6,8])))
  try:
   cp=bkg.compile_policy(candidate)
   if len(cp.nodes)<=96 and cp.state_count<=48:policies.append(candidate)
  except bkg.PolicyError:pass
  if attempts>args.cases*150:raise RuntimeError("failed to generate enough valid random policies")
 policies=policies[:args.cases]
 manifest=[];expected=[];nodes=Counter();value_kinds=Counter();value_types=Counter();arg_kinds=Counter();arg_types=Counter();ports=Counter();dynamic_actions=Counter();total_nodes=max_nodes=max_state=max_depth=total_lifecycles=compiled_bytes=0
 for ci,p in enumerate(policies):
  cp=bkg.compile_policy(p);total_nodes+=len(cp.nodes);max_nodes=max(max_nodes,len(cp.nodes));max_state=max(max_state,cp.state_count);max_depth=max(max_depth,tree_depth(p["tree"]));total_lifecycles+=len(cp.lifecycles);compiled_bytes+=cp.compiled_bytes;port_counts(p,ports)
  for n in cp.nodes:nodes[bkg.NODE_NAMES[n.type]]+=1
  for v in cp.values:value_kinds[{0:"CONST",1:"BB",2:"LEGAL_FEATURE"}[v.kind]]+=1;value_types[bkg.BB_NAMES[v.type]]+=1
  for a in cp.feature_args:arg_kinds[{0:"CONST",1:"BB"}[a.kind]]+=1;arg_types[bkg.BB_NAMES[a.type]]+=1
  for n in cp.nodes:
   if n.type==bkg.NODE_TYPES["EmitUnitIntent"]:
    for label,ref in (("unit",n.p0),("item",n.p2),("quantity",n.p3)):
     if cp.values[ref].kind!=bkg.VALUE_KINDS["CONST"]:dynamic_actions[f"unit_{label}"]+=1
   if n.type==bkg.NODE_TYPES["EmitMarketIntent"]:
    for label,ref in (("item",n.p1),("quantity",n.p2)):
     if cp.values[ref].kind!=bkg.VALUE_KINDS["CONST"]:dynamic_actions[f"market_{label}"]+=1
  ppath=scratch/f"case_{ci:04d}.bkgc";spath=scratch/f"case_{ci:04d}.bks";bkg.save_compiled(cp,ppath);frames=[frame_values(rng,t) for t in range(args.ticks)]
  if p.get("subtrees") and any(d.get("name")=="MoveAndWater" for d in p["subtrees"]):
   frames[0][(bkg.FEATURES["OWN_UNITS"],0)]=1
   frames[0][(bkg.FEATURES["OWN_WORKER_X"],0)]=5;frames[0][(bkg.FEATURES["OWN_WORKER_Y"],0)]=4;frames[0][(bkg.FEATURES["OWN_WORKER_TILE"],0)]=45
   frames[0][(bkg.FEATURES["OWN_TILE_WATERED"],45)]=0
  write_seq(spath,frames);manifest.append((ppath,spath));vm=bkg.VM(cp)
  for ti,vals in enumerate(frames):expected.append(py_line(ci,ti,vm.tick(bkg.SyntheticObservation(vals),10,False)))
 mpath=scratch/"manifest.txt"
 with open(mpath,"w",encoding="utf-8") as f:
  for p,s in manifest:f.write(f"{p.relative_to(ROOT)} {s.relative_to(ROOT)}\n")
 def invoke(exe):
  cp=subprocess.run([str(ROOT/exe),str(mpath.relative_to(ROOT))],cwd=ROOT,text=True,capture_output=True,check=False)
  if cp.returncode!=0:raise RuntimeError(f"{exe} failed rc={cp.returncode}: {cp.stderr[-4000:]}")
  return cp.stdout.splitlines()
 cpp=invoke(args.cpp);cuda=invoke(args.cuda);mismatches=[]
 for engine,lines in (("cpp",cpp),("cuda",cuda)):
  if len(lines)!=len(expected):mismatches.append((engine,"record_count",len(expected),len(lines)))
  for i,(a,b) in enumerate(zip(expected,lines)):
   if a!=b:mismatches.append((engine,i,a,b));
   if len(mismatches)>=10:break
 summary={"seed":args.seed,"policies":len(policies),"ticks_per_policy":args.ticks,"tick_records":len(expected),"pairwise_engine_comparisons":len(expected)*2,"mismatches":len(mismatches),"generator_attempts":attempts,"total_nodes":total_nodes,"max_nodes":max_nodes,"max_state_slots":max_state,"max_tree_depth":max_depth,"total_lifecycles":total_lifecycles,"mean_compiled_bytes":compiled_bytes/len(policies),"value_kind_coverage":dict(sorted(value_kinds.items())),"value_type_coverage":dict(sorted(value_types.items())),"feature_arg_kind_coverage":dict(sorted(arg_kinds.items())),"feature_arg_type_coverage":dict(sorted(arg_types.items())),"port_mode_coverage":dict(sorted(ports.items())),"dynamic_action_coverage":dict(sorted(dynamic_actions.items())),"targeted_v3_seed_policies":len(coverage_policies()),"node_type_coverage":dict(sorted(nodes.items()))}
 print(json.dumps(summary,sort_keys=True))
 for m in mismatches[:5]:print("MISMATCH",repr(m))
 return 1 if mismatches else 0

if __name__=="__main__":
 ap=argparse.ArgumentParser();ap.add_argument("--cpp",default="scratch/bonsai_kg/cpp_differential_runner");ap.add_argument("--cuda",default="scratch/bonsai_kg/cuda_differential_runner");ap.add_argument("--scratch",default="scratch/bonsai_kg/differential");ap.add_argument("--cases",type=int,default=512);ap.add_argument("--ticks",type=int,default=16);ap.add_argument("--seed",type=int,default=20260817);raise SystemExit(run(ap.parse_args()))
