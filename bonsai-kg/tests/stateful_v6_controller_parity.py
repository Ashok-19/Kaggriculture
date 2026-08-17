from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
BKG=HERE.parent
INTEGRATION=BKG/"integrations/stateful_v6"
sys.path.insert(0,str(BKG/"python"));sys.path.insert(0,str(HERE));sys.path.insert(0,str(INTEGRATION))
import bonsai_kg as bkg
from test_stateful_v6_translation import STEP144_CASES, obs
from build_assets import LEAF_ID

FEATURE_CALLS=[
    ("STEP",0),("OWN_UNITS",0),("OWN_MONEY",0),("OPP_MONEY",0),
    ("SHOP_AT",0),("SHOP_AT",1),("SHOP_AT",2),
    ("SHOP_SEEN",3),("SHOP_SEEN",4),("SHOP_SEEN",5),("SHOP_SEEN",6),("SHOP_SEEN",7),
    ("MARKET_PRICE",bkg.ITEM_ID["TOMATO"]),("MARKET_PRICE",bkg.ITEM_ID["MILK"]),
]

def frame(o):
    d=bkg.DictObservation(o);vals={}
    for name,arg in FEATURE_CALLS:vals[(bkg.FEATURES[name],arg,0)]=d.feature(bkg.FEATURES[name],arg,0)
    return vals

def write_seq(path,frames):
    with open(path,"w",encoding="utf-8") as f:
        f.write(f"BKS2 2 {len(frames)}\n")
        for vals in frames:
            items=sorted(vals.items());f.write(f"{len(items)}\n")
            for (fid,a0,a1),v in items:f.write(f"{fid} {a0} {a1} {v}\n")

def py_line(ci,ti,res):
    a=res.action;units=';'.join(f"{op},{arg},{n}" for op,arg,n in a.units);market=';'.join(f"{op},{item},{n}" for op,item,n in a.market);bb=','.join(str(x) for x in res.blackboard);ns=','.join(f"{x}/{y}" for x,y in res.node_state);return f"C={ci} T={ti} S={res.status} F={res.fault} U={a.n_units}:{units} M={len(a.market)}:{market} B={bb} N={ns}"

def stage144_obs(leaf):
    for name,route,kw in STEP144_CASES:
        if name==leaf:return obs(step=144,**kw)
    raise KeyError(leaf)

def stage216_cases():
    refined=[
        ("A1",dict(shops=["BAKERY","BRUNCH_SPOT"])),
        ("A1",dict(shops=["SMOOTHIE_SHOP","BRUNCH_SPOT"])),
        ("A1",dict(shops=["ICE_CREAM_SHOP","BRUNCH_SPOT"],own_money=715)),
        ("A1",dict(shops=["ICE_CREAM_SHOP","BRUNCH_SPOT"],own_money=716)),
        ("B",dict(shops=["BAKERY","BRUNCH_SPOT","SMOOTHIE_SHOP"])),
        ("B",dict(shops=["BAKERY","BRUNCH_SPOT","YARN_STORE"])),
        ("C1",dict(shops=["BAKERY","BRUNCH_SPOT","SMOOTHIE_SHOP"])),
        ("C1",dict(shops=["BAKERY","BRUNCH_SPOT","YARN_STORE"])),
        ("D2",dict(shops=["BAKERY","BRUNCH_SPOT","SMOOTHIE_SHOP"])),
        ("D2",dict(shops=["BAKERY","BRUNCH_SPOT","YARN_STORE"])),
        ("E",dict(shops=["BAKERY","BRUNCH_SPOT"])),
        ("E",dict(shops=["PIZZA_SHOP","BRUNCH_SPOT"])),
        ("E",dict(shops=["PET_CAFE","BRUNCH_SPOT","SMOOTHIE_SHOP"])),
        ("E",dict(shops=["PET_CAFE","BRUNCH_SPOT","YARN_STORE"])),
    ]
    unchanged=[x for x in LEAF_ID if x not in ("A1","B","C1","D2","E")]
    out=[]
    for leaf,kw in refined:out.append((leaf,obs(step=216,**kw)))
    for leaf in unchanged:out.append((leaf,obs(step=216,shops=["YARN_STORE","YARN_STORE","YARN_STORE"])))
    return out

def invoke(exe,manifest):
    cp=subprocess.run([str(ROOT/exe),str(manifest.relative_to(ROOT))],cwd=ROOT,text=True,capture_output=True,check=False)
    if cp.returncode!=0:raise RuntimeError(f"{exe} failed rc={cp.returncode}: {cp.stderr[-6000:]}")
    return cp.stdout.splitlines()

def main(args):
    scratch=(ROOT/args.scratch).resolve();shutil.rmtree(scratch,ignore_errors=True);scratch.mkdir(parents=True);policy=INTEGRATION/"assets/controller.bkgc";cases=[]
    for leaf,route,kw in STEP144_CASES:cases.append([frame(obs(step=144,**kw))])
    for leaf,o216 in stage216_cases():cases.append([frame(stage144_obs(leaf)),frame(o216)])
    manifest=scratch/"manifest.txt";expected=[]
    with open(manifest,"w",encoding="utf-8") as mf:
        for ci,frames in enumerate(cases):
            seq=scratch/f"case_{ci:03d}.bks";write_seq(seq,frames);mf.write(f"{policy.relative_to(ROOT)} {seq.relative_to(ROOT)}\n");vm=bkg.VM(bkg.load_compiled(policy))
            for ti,vals in enumerate(frames):expected.append(py_line(ci,ti,vm.tick(bkg.SyntheticObservation(vals),0,False)))
    cpp=invoke(args.cpp,manifest);cuda=invoke(args.cuda,manifest);m=[]
    for engine,lines in (("cpp",cpp),("cuda",cuda)):
        if len(lines)!=len(expected):m.append((engine,"count",len(expected),len(lines)))
        for i,(a,b) in enumerate(zip(expected,lines)):
            if a!=b:m.append((engine,i,a,b))
            if len(m)>=20:break
    print(json.dumps({"stage144_cases":len(STEP144_CASES),"stage216_cases":len(stage216_cases()),"case_sequences":len(cases),"tick_records":len(expected),"pairwise_engine_comparisons":len(expected)*2,"leaf_families":len(LEAF_ID),"mismatches":len(m)},sort_keys=True))
    for x in m[:10]:print("MISMATCH",repr(x))
    return 1 if m else 0

if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument('--cpp',default='scratch/bonsai_kg/cpp_differential_runner');ap.add_argument('--cuda',default='scratch/bonsai_kg/cuda_differential_runner');ap.add_argument('--scratch',default='scratch/bonsai_kg_v6_translation/controller_diff');raise SystemExit(main(ap.parse_args()))
