from __future__ import annotations
import argparse,json
from pathlib import Path
from tools.bkg4_search.cuda_io import load_results

def main()->int:
 ap=argparse.ArgumentParser();ap.add_argument('--results',type=Path,required=True);ap.add_argument('--cases',type=Path,required=True);ap.add_argument('--labels',nargs='+');a=ap.parse_args();cases=[json.loads(x) for x in a.cases.read_text().splitlines() if x.strip()];rows=load_results(a.results);labels=a.labels or [f'g{i}' for i in range(len(rows))]
 if len(labels)!=len(rows):raise RuntimeError('label count mismatch')
 baseline=[int(round(c['base_margin'])) for c in cases];bad=[];brow=[]
 for i,(own,opp,route,flags) in enumerate(rows[0]):
  m=own-opp;brow.append(m)
  if m!=baseline[i] or flags:bad.append((i,m,baseline[i],flags))
 out={'baseline_exact':not bad,'baseline_mismatches':bad,'genomes':{}}
 for label,row in zip(labels,rows):
  margins=[own-opp for own,opp,route,flags in row];changed=[]
  for i,(m,b) in enumerate(zip(margins,baseline)):
   if m!=b:changed.append({'case':i,'team':cases[i]['team'],'seat':cases[i]['seat'],'first_shop':cases[i].get('first_shop'),'base':b,'margin':m,'delta':m-b})
  by={}
  for i,c in enumerate(cases):
   k=c.get('first_shop','unknown');by.setdefault(k,[]).append(i)
  bysum={}
  for k,idx in by.items():
   bysum[k]={'cases':len(idx),'wins':sum(margins[i]>0 for i in idx),'losses':sum(margins[i]<0 for i in idx),'converted_losses':sum(baseline[i]<0 and margins[i]>0 for i in idx),'lost_wins':sum(baseline[i]>0 and margins[i]<=0 for i in idx),'mean_delta':sum(margins[i]-baseline[i] for i in idx)/len(idx)}
  out['genomes'][label]={'wins':sum(m>0 for m in margins),'ties':sum(m==0 for m in margins),'losses':sum(m<0 for m in margins),'converted_losses':sum(b<0 and m>0 for b,m in zip(baseline,margins)),'lost_wins':sum(b>0 and m<=0 for b,m in zip(baseline,margins)),'protected_regressions':sum(m<b-3000 for b,m in zip(baseline,margins)),'mean_delta':sum(m-b for b,m in zip(baseline,margins))/len(margins),'min_delta':min((m-b for b,m in zip(baseline,margins)),default=0),'by_first_shop':bysum,'changed':changed}
 print(json.dumps(out,ensure_ascii=False,sort_keys=True));return 0 if not bad else 2
if __name__=='__main__':raise SystemExit(main())
