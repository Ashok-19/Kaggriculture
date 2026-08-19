from __future__ import annotations
import argparse,csv,json
from pathlib import Path


def main()->int:
 ap=argparse.ArgumentParser();ap.add_argument('--bundle',type=Path,default=Path('scratch/kaggriculture_meta_search_bundle'));ap.add_argument('--fresh',type=Path,default=Path('scratch/bkg4_search/league_aug19/fresh_manifest.jsonl'));ap.add_argument('--out',type=Path,default=Path('scratch/bkg4_search/discovery_cases.jsonl'));a=ap.parse_args();root=Path(__file__).resolve().parents[2];bundle=(root/a.bundle).resolve();fresh=(root/a.fresh).resolve();out=(root/a.out).resolve();rows=list(csv.DictReader((bundle/'populations/dev_manifest.tsv').open(),delimiter='\t'));by_rel={r['bundle_path'].split('populations/',1)[-1]:r for r in rows};core=[]
 for rel in (bundle/'populations/stage1_core12.txt').read_text().splitlines():
  rel=rel.strip();
  if not rel:continue
  key=rel.split('populations/',1)[-1] if 'populations/' in rel else rel
  r=by_rel[key];path=(bundle/'populations'/key).resolve()
  for seat in (0,1):core.append({'tape':str(path),'team':r['team'],'seed':int(r['source_seed']),'seat':seat,'protected':1,'own_history':0,'source':'core12'})
 fresh_rows=[json.loads(x) for x in fresh.read_text().splitlines() if x.strip()];seen=set();current=[]
 for r in fresh_rows:
  team=r['team'];
  # One high-value source tape per (team,episode seat) is enough; duplicates across the
  # five pairwise replays still provide distinct seeds for Ryo/tetsuya.
  key=(team,r['episode_id'],r['seat'])
  if key in seen:continue
  seen.add(key)
  own=int(team=='ReCurSiON')
  for seat in (0,1):current.append({'tape':r['path'],'team':team,'seed':int(r['seed']),'seat':seat,'protected':1,'own_history':own,'source':'fresh_current'})
 all_rows=core+current;out.parent.mkdir(parents=True,exist_ok=True);out.write_text(''.join(json.dumps(r,ensure_ascii=False,sort_keys=True)+'\n' for r in all_rows),encoding='utf-8');print(json.dumps({'cases':len(all_rows),'core_cases':len(core),'fresh_cases':len(current),'families':sorted({r['team'] for r in all_rows}),'own_history_cases':sum(r['own_history'] for r in all_rows)},ensure_ascii=False,sort_keys=True));return 0
if __name__=='__main__':raise SystemExit(main())
