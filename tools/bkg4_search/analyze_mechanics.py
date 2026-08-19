from __future__ import annotations

import argparse
import json
import struct
from dataclasses import asdict
from pathlib import Path

from tools.bkg4_search.campaign import fitness
from tools.bkg4_search.mechanics_codec import HEADER, GENOME, MechanicGenome

ROOT = Path(__file__).resolve().parents[2]
RESULT_HEADER = struct.Struct("<8sIII")
RESULT = struct.Struct("<iiHH")


def load_genomes(path: Path):
    raw = path.read_bytes(); magic, version, count = HEADER.unpack_from(raw, 0)
    if magic != b"B4MX1\0\0\0" or version != 1: raise RuntimeError("bad mechanic genome file")
    out=[];off=HEADER.size
    for i in range(count):
        mode,mask,repl,flags,wcap,_ = GENOME.unpack_from(raw,off+i*GENOME.size)
        out.append(MechanicGenome(mode,mask,repl,flags,wcap))
    return out


def load_results(path: Path):
    raw=path.read_bytes();magic,ver,ng,nc=RESULT_HEADER.unpack_from(raw,0)
    if magic!=b"B4MR1\0\0\0" or ver!=1:raise RuntimeError("bad mechanic result file")
    out=[];off=RESULT_HEADER.size
    for g in range(ng):
        row=[]
        for c in range(nc):row.append(RESULT.unpack_from(raw,off+(g*nc+c)*RESULT.size))
        out.append(row)
    return out


def main()->int:
    ap=argparse.ArgumentParser();ap.add_argument('--genomes',type=Path,required=True);ap.add_argument('--results',type=Path,required=True);ap.add_argument('--cases',type=Path,required=True);ap.add_argument('--top',type=int,default=30);ap.add_argument('--out',type=Path);a=ap.parse_args()
    gp=(ROOT/a.genomes).resolve() if not a.genomes.is_absolute() else a.genomes;rp=(ROOT/a.results).resolve() if not a.results.is_absolute() else a.results;cp=(ROOT/a.cases).resolve() if not a.cases.is_absolute() else a.cases
    genomes=load_genomes(gp);rows=load_results(rp);cases=[json.loads(x) for x in cp.read_text().splitlines() if x.strip()]
    if len(rows)!=len(genomes) or any(len(x)!=len(cases) for x in rows):raise RuntimeError('dimension mismatch')
    baseline=[x[0]-x[1] for x in rows[0]];idx=list(range(len(cases)))
    scored=[]
    for gi,(g,row) in enumerate(zip(genomes,rows)):
        margins=[x[0]-x[1] for x in row];rew=sum(x[2] for x in row);badflags=sum(bool(x[3]) for x in row);f=fitness(margins,idx,cases,baseline)
        # One bounded mechanic rule; use observed rewrite count as a secondary hot-work proxy only.
        f=type(f)(f.catastrophic_regressions,f.protected_regressions,f.own_history_regressions,f.losing_families,f.losses,f.ties,f.wins,f.worst_family_milli,f.floor_margin,f.mean_margin,rew,0 if gi==0 else 1)
        changed=[i for i,m in enumerate(margins) if m!=baseline[i]]
        scored.append((f.lexicographic_key(),gi,g,f,margins,rew,badflags,changed))
    scored.sort(key=lambda x:x[0])
    top=[]
    for rank,(_,gi,g,f,margins,rew,badflags,changed) in enumerate(scored[:a.top],1):
        details=[]
        for i in changed[:20]:details.append({'case':i,'team':cases[i]['team'],'seat':cases[i]['seat'],'seed':cases[i]['seed'],'base':baseline[i],'margin':margins[i],'delta':margins[i]-baseline[i]})
        top.append({'rank':rank,'index':gi,'id':g.semantic_id(),'genome':g.canonical(),'fitness':asdict(f),'rewrites':rew,'bad_flags':badflags,'changed_cases':len(changed),'changes':details})
    payload={'schema':1,'cases':len(cases),'genomes':len(genomes),'baseline_fitness':asdict(fitness(baseline,idx,cases,baseline)),'top':top}
    if a.out:
        out=(ROOT/a.out).resolve() if not a.out.is_absolute() else a.out;out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(payload,ensure_ascii=False,sort_keys=True,indent=2)+'\n')
    print(json.dumps(payload,ensure_ascii=False,sort_keys=True))
    return 0

if __name__=='__main__':raise SystemExit(main())
