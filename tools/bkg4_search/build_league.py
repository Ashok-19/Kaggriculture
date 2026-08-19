from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replay_tapes(path: Path, out_dir: Path) -> List[Dict[str, Any]]:
    replay=json.loads(path.read_text(encoding="utf-8"));steps=replay.get("steps") or []
    if len(steps)<2 or any(len(x)!=2 for x in steps):raise ValueError(f"invalid replay {path}")
    info=replay.get("info") or {};teams=list(info.get("TeamNames") or ["seat0","seat1"]);seed=int(info.get("seed"))
    episode=int(info.get("EpisodeId") or 0);replay_hash=sha256(path);rows=[]
    # Kaggle replay step 0 stores initial PASS placeholders. actions[t] in a native
    # tape corresponds to replay.steps[t+1][seat].action, exactly as the historical
    # development bank was extracted.
    for seat in (0,1):
        actions=[dict(step[seat].get("action") or {"farmer":["PASS"],"hands":[],"market":[]}) for step in steps[1:]]
        reward=float(replay.get("rewards",[0,0])[seat]);opp_reward=float(replay.get("rewards",[0,0])[1-seat])
        obj={"source":{"episode_id":episode,"seat":seat,"submission_id":None,"stream":teams[seat],"team":teams[seat],"opponent":teams[1-seat],"seed":seed,"reward":reward,"opponent_reward":opp_reward,"replay_sha256":replay_hash,"bank_role":"fresh_current"},"actions":actions}
        out=out_dir/f"{teams[seat].replace('/','_').replace(' ','_')}_s{seat}_ep{episode}.json";out.write_text(json.dumps(obj,ensure_ascii=False,separators=(",",":"))+"\n",encoding="utf-8")
        rows.append({"team":teams[seat],"opponent":teams[1-seat],"seat":seat,"episode_id":episode,"seed":seed,"reward":reward,"opponent_reward":opp_reward,"path":str(out),"sha256":hashlib.sha256(out.read_bytes()).hexdigest(),"source_kind":"fresh_current"})
    return rows


def main() -> int:
    ap=argparse.ArgumentParser();ap.add_argument('--bundle',type=Path,default=Path('scratch/kaggriculture_meta_search_bundle'));ap.add_argument('--fresh-replays',type=Path,default=Path('scratch/live_top_aug19'));ap.add_argument('--out',type=Path,default=Path('scratch/bkg4_search/league_aug19'));args=ap.parse_args()
    root=Path(__file__).resolve().parents[2];bundle=(root/args.bundle).resolve() if not args.bundle.is_absolute() else args.bundle;fresh=(root/args.fresh_replays).resolve() if not args.fresh_replays.is_absolute() else args.fresh_replays;out=(root/args.out).resolve() if not args.out.is_absolute() else args.out;out.mkdir(parents=True,exist_ok=True);fresh_out=out/'fresh_tapes';fresh_out.mkdir(exist_ok=True)
    base=list(csv.DictReader((bundle/'populations/dev_manifest.tsv').open(encoding='utf-8'),delimiter='\t'))
    if len(base)!=168:raise RuntimeError(f"expected frozen 168 development tapes, got {len(base)}")
    fresh_rows=[]
    for p in sorted(fresh.glob('episode-*-replay.json')):fresh_rows.extend(replay_tapes(p,fresh_out))
    # Exact-deduplicate fresh team/episode/seat. Replays are intentionally few and high-value.
    seen=set();fr=[]
    for r in fresh_rows:
        k=(r['episode_id'],r['seat'])
        if k not in seen:seen.add(k);fr.append(r)
    families=defaultdict(int)
    for r in base:families[r['team']]+=1
    for r in fr:families[r['team']]+=1
    manifest={
        'schema':1,'frozen_dev_tapes':len(base),'frozen_dev_families':len({r['team'] for r in base}),
        'fresh_tapes':len(fr),'fresh_episodes':len({r['episode_id'] for r in fr}),'fresh_families':sorted({r['team'] for r in fr}),
        'all_families':dict(sorted(families.items())),
        'splits':{
            'search':'frozen dev stage1/stage2 plus selected fresh current tapes',
            'development':'remaining frozen dev tapes plus current-family tapes',
            'sealed':'existing validation/blind/family holdout artifacts are not copied or enumerated here',
        },
        'notes':[
            'Fresh current tapes are replay-faithful action programs used only as benchmark opponents; strategy never receives team identity.',
            'Historical ReCurSiON source packages remain authoritative for causal analysis; self replay tapes are benchmarking evidence, not identity routes.'
        ],
    }
    (out/'fresh_manifest.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False,sort_keys=True)+'\n' for r in fr),encoding='utf-8')
    (out/'league_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,sort_keys=True,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(manifest,ensure_ascii=False,sort_keys=True))
    return 0

if __name__=='__main__':raise SystemExit(main())
