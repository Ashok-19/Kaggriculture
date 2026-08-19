from __future__ import annotations
import argparse,json
from pathlib import Path

def main()->int:
 ap=argparse.ArgumentParser();ap.add_argument('--recorded',type=Path,default=Path('scratch/bkg4_search/base0_discovery16_p72/cases.jsonl'));ap.add_argument('--out',type=Path,default=Path('scratch/bkg4_search/cuda_cases16.tsv'));ap.add_argument('--prefix-override',type=int);a=ap.parse_args();root=Path(__file__).resolve().parents[2];src=(root/a.recorded).resolve() if not a.recorded.is_absolute() else a.recorded;out=(root/a.out).resolve() if not a.out.is_absolute() else a.out;rows=[json.loads(x) for x in src.read_text().splitlines() if x.strip()];families={name:i for i,name in enumerate(sorted({r['team'] for r in rows}))};lines=[]
 for r in rows:
  pm=r.get('prefix_meta') or {};eligible=int(bool(r.get('search_eligible')) and r.get('external_route') is None);prefix=int(a.prefix_override if a.prefix_override is not None else r['prefix']);lines.append('\t'.join(map(str,[r['base0_tape'],r['tape'],r['seat'],r['seed'],prefix,families[r['team']],r.get('protected',0),r.get('own_history',0),eligible,int(bool(pm.get('high_labor'))),int(bool(pm.get('carrot_latched'))),int(round(r['base_margin']))])))
 out.parent.mkdir(parents=True,exist_ok=True);out.write_text('\n'.join(lines)+'\n');(out.with_suffix('.families.json')).write_text(json.dumps({'families':families,'cases':[{'case':i,'team':r['team'],'seat':r['seat'],'base_margin':r['base_margin'],'eligible':bool(int(lines[i].split('\t')[8])),'source':r['source']} for i,r in enumerate(rows)]},ensure_ascii=False,sort_keys=True,indent=2)+'\n');print(json.dumps({'cases':len(rows),'families':families,'eligible':sum(int(x.split('\t')[8]) for x in lines)},ensure_ascii=False,sort_keys=True));return 0
if __name__=='__main__':raise SystemExit(main())
