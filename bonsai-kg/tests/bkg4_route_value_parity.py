from __future__ import annotations
import argparse, importlib.util, subprocess, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; V6=ROOT/'bonsai-kg/integrations/stateful_v6'; sys.path.insert(0,str(V6))
spec=importlib.util.spec_from_file_location('rvp_v6',V6/'runtime.py'); m=importlib.util.module_from_spec(spec); assert spec.loader; spec.loader.exec_module(m)

def py_line(pack,base,ov,step,n):
    route=ov or base; a=pack.action(route,step,max(0,n-1)); units=[a.get('farmer') or ['PASS']]+list(a.get('hands') or [])
    import bonsai_kg as bkg
    def ut(x):
        x=list(x or ['PASS']); op=bkg.UNIT_OPS[x[0]]; arg=0; num=1
        if len(x)>=2:
            if isinstance(x[1],str): arg=bkg.ITEM_ID[x[1]]
            else: num=int(x[1])
        if len(x)>=3:num=int(x[2])
        return op,arg,num
    def mt(x):
        op=bkg.MARKET_OPS[x[0]]
        if x[0] in ('HIRE','BUY_LAND'):return op,0,1
        return op,bkg.ITEM_ID[x[1]],int(x[2])
    u=';'.join(f'{x},{y},{z}' for x,y,z in map(ut,units)); market=[mt(x) for x in (a.get('market') or [])]; mm=';'.join(f'{x},{y},{z}' for x,y,z in market)
    return f'{route} U={len(units)}:{u} M={len(market)}:{mm}'

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--assets',type=Path,required=True);ap.add_argument('--cpp',type=Path,required=True);ap.add_argument('--cuda',type=Path,required=True);args=ap.parse_args(); assets=args.assets if args.assets.is_absolute() else ROOT/args.assets; pack=m.RoutePack(assets/'routes.bkv6',{'route_pack':__import__('json').loads((assets/'identity.json').read_text())['v6_route_pack']})
    cases=[(64,0,26,1),(64,211,26,1),(64,211,33,2),(64,211,36,5),(64,212,26,1),(64,212,42,1),(64,212,45,1),(64,213,264,5),(64,213,267,13),(64,213,287,13),(64,214,192,6),(64,214,197,12),(64,214,215,12),(64,22,264,5),(64,22,287,8),(64,46,264,6),(64,46,287,10),(29,0,500,13)]
    bad=[]
    for c in cases:
        want=py_line(pack,*c); got=[]
        for exe in (args.cpp,args.cuda):
            p=exe if exe.is_absolute() else ROOT/exe; r=subprocess.run([str(p),str(assets/'routes.bkv6'),*map(str,c)],capture_output=True,text=True,check=True);got.append(r.stdout.strip())
        if got[0]!=want or got[1]!=want: bad.append((c,want,*got))
    print({'cases':len(cases),'mismatches':len(bad),'first':bad[:1]}); return 1 if bad else 0
if __name__=='__main__': raise SystemExit(main())
