from __future__ import annotations
import argparse,csv,json
from pathlib import Path

DEFAULT_TEAMS=(
 'Thomas Tschinkel','Galaxantic','Calmracer','Efe Can Celiksoy','Ueddy','SCLim2022080004',
 'Eddy Despradel','Utkarsh #2','Michael Timbs','One-For-All','mandgeee','HKmgikao',
)

def main()->int:
 ap=argparse.ArgumentParser();ap.add_argument('--bundle',type=Path,default=Path('scratch/kaggriculture_meta_search_bundle'));ap.add_argument('--out',type=Path,default=Path('scratch/bkg4_search/activation_validation24.jsonl'));a=ap.parse_args();root=Path(__file__).resolve().parents[2];bundle=(root/a.bundle).resolve();rows=list(csv.DictReader((bundle/'populations/dev_manifest.tsv').open(),delimiter='\t'));chosen=[]
 for team in DEFAULT_TEAMS:
  candidates=[r for r in rows if r['team']==team and r['shop_regime'].split('|',1)[0] in ('PET_CAFE','PIZZA_SHOP')]
  if not candidates:raise RuntimeError(f'no activation tape for {team}')
  r=candidates[0];path=(bundle/r['bundle_path']).resolve();first=r['shop_regime'].split('|',1)[0]
  for seat in (0,1):chosen.append({'tape':str(path),'team':team,'seed':int(r['source_seed']),'seat':seat,'protected':1,'own_history':0,'source':'activation_dev','first_shop':first})
 out=(root/a.out).resolve() if not a.out.is_absolute() else a.out;out.parent.mkdir(parents=True,exist_ok=True);out.write_text(''.join(json.dumps(x,ensure_ascii=False,sort_keys=True)+'\n' for x in chosen),encoding='utf-8');print(json.dumps({'cases':len(chosen),'seeds':len(chosen)//2,'families':len(DEFAULT_TEAMS),'pet':sum(x['first_shop']=='PET_CAFE' for x in chosen),'pizza':sum(x['first_shop']=='PIZZA_SHOP' for x in chosen)},sort_keys=True));return 0
if __name__=='__main__':raise SystemExit(main())
