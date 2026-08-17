from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
BKG=HERE.parent
INTEGRATION=BKG/"integrations/stateful_v6"
ORACLE=ROOT/"scratch/bonsai_cpu/stateful_core_v6"
sys.path.insert(0,str(BKG/"python"));sys.path.insert(0,str(INTEGRATION))
import bonsai_kg as bkg
from build_assets import ROUTE_IDS
from runtime import RoutePack


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

def line(idx,rid,step,a):
    units=[unit_tuple(a.get("farmer") or ["PASS"])]+[unit_tuple(x) for x in (a.get("hands",[]) or [])]
    market=[order_tuple(x) for x in (a.get("market",[]) or [])]
    u=';'.join(f"{op},{arg},{n}" for op,arg,n in units);m=';'.join(f"{op},{item},{n}" for op,item,n in market)
    return f"I={idx} R={rid} S={step} U={len(units)}:{u} M={len(market)}:{m}"

def invoke(exe,pack):
    cp=subprocess.run([str(exe),str(pack)],cwd=ROOT,text=True,capture_output=True,check=False)
    if cp.returncode!=0:raise RuntimeError(f"{exe} failed rc={cp.returncode}: {cp.stderr[-4000:]}")
    return cp.stdout.splitlines(),cp.stderr.strip()

def main(args):
    asset=INTEGRATION/"assets";identity=json.loads((asset/"identity.json").read_text());pack=RoutePack(asset/"routes.bkv6",identity)
    expected=[];python_pack=[];idx=0
    for rid in ROUTE_IDS:
        src=json.loads((ORACLE/f"route{rid}.json").read_text())["actions"]
        for step,a in enumerate(src):
            expected.append(line(idx,rid,step,a))
            p=pack.action(rid,step,len(a.get("hands",[]) or []));python_pack.append(line(idx,rid,step,p));idx+=1
    cpp,cpp_meta=invoke(ROOT/args.cpp,asset/"routes.bkv6");cuda,cuda_meta=invoke(ROOT/args.cuda,asset/"routes.bkv6")
    mismatches=[]
    for engine,lines in (("python_pack",python_pack),("cpp",cpp),("cuda",cuda)):
        if len(lines)!=len(expected):mismatches.append((engine,"count",len(expected),len(lines)))
        for i,(a,b) in enumerate(zip(expected,lines)):
            if a!=b:mismatches.append((engine,i,a,b))
            if len(mismatches)>=20:break
    print(json.dumps({"routes":len(ROUTE_IDS),"action_records":len(expected),"source_vs_python_comparisons":len(expected),"source_vs_cpp_comparisons":len(expected),"source_vs_cuda_comparisons":len(expected),"total_pairwise_comparisons":len(expected)*3,"mismatches":len(mismatches),"cpp_meta":cpp_meta,"cuda_meta":cuda_meta,"route_pack_sha256":identity["route_pack"]["pack_sha256"]},sort_keys=True))
    for m in mismatches[:10]:print("MISMATCH",repr(m))
    return 1 if mismatches else 0

if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--cpp",default="scratch/bonsai_kg_v6_translation/stateful_v6_pack_cpp");ap.add_argument("--cuda",default="scratch/bonsai_kg_v6_translation/stateful_v6_pack_cuda");raise SystemExit(main(ap.parse_args()))
