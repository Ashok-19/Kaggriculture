from __future__ import annotations
import argparse,csv,json
from pathlib import Path

# These dev-manifest rows were not used in discovery or the 24-case activation development set.
SEALED_PET_ROWS=(0,19,58,60,73,75,124,146,154)
# Fresh-current discovery rows left untouched by the original 16-case search.
SEALED_FRESH_ROWS=(26,27,34,35,36,37,42,43)

def main()->int:
 ap=argparse.ArgumentParser();ap.add_argument('--bundle',type=Path,default=Path('scratch/kaggriculture_meta_search_bundle'));ap.add_argument('--fresh-cases',type=Path,default=Path('scratch/bkg4_search/discovery_cases.jsonl'));ap.add_argument('--out',type=Path,default=Path('scratch/bkg4_search/sealed_holdout26.jsonl'));a=ap.parse_args();root=Path(__file__).resolve().parents[2];bundle=(root/a.bundle).resolve();dev=list(csv.DictReader((bundle/'populations/dev_manifest.tsv').open(),delimiter='\t'));rows=[]
 for ix in SEALED_PET_ROWS:
  r=dev[ix]
  if r['shop_regime'].split('|',1)[0]!='PET_CAFE':raise RuntimeError(f'row {ix} no longer PET_CAFE')
  path=(bundle/r['bundle_path']).resolve()
  for seat in (0,1):rows.append({'tape':str(path),'team':r['team'],'seed':int(r['source_seed']),'seat':seat,'protected':1,'own_history':0,'source':'sealed_pet','first_shop':'PET_CAFE'})
 fresh=[json.loads(x) for x in (root/a.fresh_cases).read_text().splitlines() if x.strip()]
 for ix in SEALED_FRESH_ROWS:
  r=dict(fresh[ix]);r['protected']=1;r['own_history']=0;r['source']='sealed_fresh_current';rows.append(r)
 out=(root/a.out).resolve() if not a.out.is_absolute() else a.out;out.parent.mkdir(parents=True,exist_ok=True);out.write_text(''.join(json.dumps(x,ensure_ascii=False,sort_keys=True)+'\n' for x in rows),encoding='utf-8');print(json.dumps({'cases':len(rows),'unique_seeds':len({r['seed'] for r in rows}),'teams':sorted({r['team'] for r in rows}),'pet_cases':sum(r.get('first_shop')=='PET_CAFE' or r['source']=='sealed_pet' for r in rows)},ensure_ascii=False,sort_keys=True));return 0
if __name__=='__main__':raise SystemExit(main())
