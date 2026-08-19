#!/usr/bin/env python3
from __future__ import annotations
import argparse, subprocess, sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(HERE.parent/"python"))
import bonsai_kg_v4 as b


def max_policy():
    outer=[f"O{i}" for i in range(254)];inner=[f"I{i}" for i in range(254)]
    def inner_switch(special=False):
        cases={v:{"type":"Success"} for v in inner}
        if special:cases[inner[-1]]={"type":"Sequence","children":[{"type":"Success"} for _ in range(251)]}
        return {"type":"SwitchEnum","slot":"inner","cases":cases,"default":{"type":"Success"}}
    ocases={v:inner_switch(v==outer[-1]) for v in outer}
    huge={"type":"SwitchEnum","slot":"outer","cases":ocases,"default":inner_switch(False)}
    return {"schema_version":4,"enums":[{"name":"Outer","values":outer},{"name":"Inner","values":inner}],"blackboard":[{"name":"outer","type":"Outer","initial":"O0"},{"name":"inner","type":"Inner","initial":"I0"}],"tree":{"type":"Sequence","children":[huge,{"type":"Success"}]}}


def metadata_policy():
    bb=[{"name":"x","type":"I32","initial":0}]+[{"name":f"b{i}","type":"I32","initial":0} for i in range(1,128)]
    children=[]
    for i in range(300):
        children.append({"type":"Success","lifecycle":{"failure_if":{"lhs":{"feature":"STEP"},"op":"LT","rhs":i},"on_success":[{"op":"set","slot":"x","value":i}]}})
    return {"schema_version":4,"blackboard":bb,"tree":{"type":"Sequence","children":children}}


def high_source_cancel_policy():
    running={"type":"Sequence","children":[*({"type":"Success"} for _ in range(300)),{"type":"MoveUnitTo","unit":{"const":0,"type":"UNIT"},"target":{"const":0,"type":"TILE"},"authority":"TACTICAL","priority":1}]}
    return {"schema_version":4,"tree":{"type":"Parallel","success_threshold":1,"failure_threshold":2,"children":[running,{"type":"Success"}]}}


def high_order_policy():
    branches=[{"type":"Sequence","children":[{"type":"EmitMarketIntent","action":"HIRE","priority":7},{"type":"Failure"}]} for _ in range(300)]
    branches.append({"type":"EmitMarketIntent","action":"BUY_LAND","priority":7})
    return {"schema_version":4,"tree":{"type":"Selector","children":branches}}


def write_seq(path,steps=2):
    lines=[f"BKS2 2 {steps}"]
    for step in range(steps):
        entries=[(b.FEATURES["STEP"],0,0,step),(b.FEATURES["OWN_UNITS"],0,0,1),(b.FEATURES["OWN_WORKER_X"],0,0,4),(b.FEATURES["OWN_WORKER_Y"],0,0,4)]
        lines.append(str(len(entries)));lines.extend(f"{a} {x} {y} {v}" for a,x,y,v in entries)
    path.write_text("\n".join(lines)+"\n")


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--work",default="scratch/bonsai_kg/v4_boundary");ap.add_argument("--validator");ap.add_argument("--cpp");ap.add_argument("--cuda");args=ap.parse_args();w=Path(args.work);w=w if w.is_absolute() else ROOT/w;w.mkdir(parents=True,exist_ok=True)
    cps={}
    for name,src in [("max",max_policy()),("metadata",metadata_policy()),("high_source",high_source_cancel_policy()),("high_order",high_order_policy())]:
        cp=b.compile_policy(src);cps[name]=cp;b.save_compiled(cp,w/f"{name}.bkgc")
    # Parallel is intentionally capped at 16 children, so high state ids use nested groups.
    groups=[]
    for _ in range(17):groups.append({"type":"Sequence","children":[{"type":"WaitTurns","turns":1} for _ in range(16)]})
    sp={"schema_version":4,"tree":{"type":"Sequence","children":groups}}
    cps["state"]=b.compile_policy(sp);b.save_compiled(cps["state"],w/"state.bkgc")
    assert len(cps["max"].nodes)==65534 and len(cps["max"].child_refs)==65533
    assert cps["max"].nodes[0].subtree_end==65534 and cps["max"].nodes[-1].subtree_end==65534
    assert cps["metadata"].bb[-1].name=="b127" and len(cps["metadata"].values)>255 and len(cps["metadata"].predicates)>255 and len(cps["metadata"].lifecycles)>255 and len(cps["metadata"].effects)>255
    assert cps["state"].state_count>255
    hs=cps["high_source"];move_ids=[i for i,n in enumerate(hs.nodes) if n.type==b.NODE_TYPES["MoveUnitTo"]];assert move_ids and move_ids[0]>255
    r=b.VM(hs).tick(b.SyntheticObservation({(b.FEATURES["STEP"],0):0,(b.FEATURES["OWN_UNITS"],0):1,(b.FEATURES["OWN_WORKER_X"],0):4,(b.FEATURES["OWN_WORKER_Y"],0):4}));assert r.action.units[0][0]==b.UNIT_OPS["PASS"]
    assert cps["high_order"].analysis[0].max_emissions>255 and cps["high_order"].max_live_intents==1
    if args.validator:
        for name in cps:subprocess.run([args.validator,str(w/f"{name}.bkgc")],check=True,capture_output=True,text=True)
    seq=w/"seq.bks";write_seq(seq,2);manifest=w/"manifest.txt";manifest.write_text("".join(f"{w/(name+'.bkgc')} {seq}\n" for name in ("max","high_source","high_order")))
    outputs={}
    for name,exe in (("cpp",args.cpp),("cuda",args.cuda)):
        if exe:outputs[name]=subprocess.run([exe,str(manifest)],check=True,capture_output=True,text=True).stdout
    if len(outputs)==2 and outputs["cpp"]!=outputs["cuda"]:raise RuntimeError("boundary C++/CUDA mismatch")
    print({"max_nodes":len(cps["max"].nodes),"max_last_node":len(cps["max"].nodes)-1,"root_subtree_end":cps["max"].nodes[0].subtree_end,"max_hot_bound":cps["max"].analysis[0].max_visits,"max_state":cps["state"].state_count,"bb":len(cps["metadata"].bb),"values":len(cps["metadata"].values),"predicates":len(cps["metadata"].predicates),"lifecycles":len(cps["metadata"].lifecycles),"effects":len(cps["metadata"].effects),"high_source_node":move_ids[0],"high_order_emissions":cps["high_order"].analysis[0].max_emissions,"native_engines":sorted(outputs)})
    return 0
if __name__=="__main__":raise SystemExit(main())
