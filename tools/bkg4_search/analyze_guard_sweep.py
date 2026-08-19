from __future__ import annotations
import argparse,json
from pathlib import Path
from tools.bkg4_search.cuda_io import load_results

def main()->int:
 ap=argparse.ArgumentParser();ap.add_argument('--results',type=Path,required=True);ap.add_argument('--cases',type=Path,required=True);ap.add_argument('--thresholds',type=int,nargs='+',required=True);a=ap.parse_args();cases=[json.loads(x) for x in a.cases.read_text().splitlines() if x.strip()];base=[int(round(c['base_margin'])) for c in cases];r=load_results(a.results);bad=[]
 for i,(o,p,route,flags) in enumerate(r[0]):
  if o-p!=base[i] or flags:bad.append((i,o-p,base[i],flags))
 rows=[]
 for t,row in zip(a.thresholds,r[1:]):
  margins=[o-p for o,p,route,flags in row];routes=[route for o,p,route,flags in row];rows.append({'threshold':t,'active':sum(x!=0xffff for x in routes),'wins':sum(x>0 for x in margins),'losses':sum(x<0 for x in margins),'converted_losses':sum(b<0 and m>0 for b,m in zip(base,margins)),'lost_wins':sum(b>0 and m<=0 for b,m in zip(base,margins)),'protected_regressions':sum(m<b-3000 for b,m in zip(base,margins)),'mean_delta':sum(m-b for b,m in zip(base,margins))/len(base),'min_delta':min(m-b for b,m in zip(base,margins))})
 print(json.dumps({'baseline_exact':not bad,'bad':bad,'rows':rows},sort_keys=True));return 0 if not bad else 2
if __name__=='__main__':raise SystemExit(main())
