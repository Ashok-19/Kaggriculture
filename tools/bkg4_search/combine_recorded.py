from __future__ import annotations
import argparse,json
from pathlib import Path

def main()->int:
 ap=argparse.ArgumentParser();ap.add_argument('--input',type=Path,action='append',required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();rows=[]
 for p in a.input:rows.extend(json.loads(x) for x in p.read_text().splitlines() if x.strip())
 # Keep source case numbering local to the combined corpus; base0/tape paths are stable.
 for i,r in enumerate(rows):r['case']=i
 a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(''.join(json.dumps(r,ensure_ascii=False,sort_keys=True)+'\n' for r in rows),encoding='utf-8');print(json.dumps({'cases':len(rows),'families':len({r['team'] for r in rows}),'wins':sum(r['base_margin']>0 for r in rows),'losses':sum(r['base_margin']<0 for r in rows)},sort_keys=True));return 0
if __name__=='__main__':raise SystemExit(main())
