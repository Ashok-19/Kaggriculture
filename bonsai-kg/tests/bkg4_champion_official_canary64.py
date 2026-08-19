from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
KAG_ENV = ROOT / "kaggle-environments-master"
CHAMPION = ROOT / "bonsai-kg" / "integrations" / "bkg4_champion"
sys.path.insert(0, str(KAG_ENV)); sys.path.insert(0, str(CHAMPION))

from kaggle_environments import make
from runtime import BKG4ChampionAgent


def tape_obj(path: Path, team: str | None = None, seed: int | None = None) -> dict:
    obj = json.loads(path.read_text(encoding="utf-8"))
    actions = obj.get("actions") if isinstance(obj, dict) else obj
    if not isinstance(actions, list) or not actions:
        raise ValueError(f"invalid tape {path}")
    source = obj.get("source", {}) if isinstance(obj, dict) else {}
    return {
        "path": path.resolve(), "actions": actions,
        "team": str(source.get("team") or team or path.stem),
        "seed": int(source.get("seed") if source.get("seed") is not None else seed),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def collect_opponents(n: int) -> list[dict]:
    out=[];seen=set()
    fresh_dirs=[ROOT/'scratch/bkg4_search/animal_sealed24/tapes',ROOT/'scratch/bkg4_search/fresh_promotion24/tapes',ROOT/'scratch/bkg4_search/league_aug19/fresh_tapes']
    for d in fresh_dirs:
        if not d.is_dir(): continue
        for p in sorted(d.glob('*.json')):
            try:x=tape_obj(p)
            except Exception:continue
            key=x['sha256']
            if key in seen:continue
            seen.add(key);out.append(x)
            if len(out)>=n:return out
    manifest=ROOT/'scratch/kaggriculture_meta_search_bundle/populations/dev_manifest.tsv'
    rows=list(csv.DictReader(manifest.open(encoding='utf-8'),delimiter='\t'))
    # First maximize family coverage, then fill deterministically with additional seeds.
    ordered=[];used_family=set()
    for r in rows:
        if r['team'] not in used_family:used_family.add(r['team']);ordered.append(r)
    ordered.extend(r for r in rows if r not in ordered)
    for r in ordered:
        p=ROOT/'scratch/kaggriculture_meta_search_bundle'/r['bundle_path']
        x=tape_obj(p,r['team'],int(r['source_seed']))
        if x['sha256'] in seen:continue
        seen.add(x['sha256']);out.append(x)
        if len(out)>=n:return out
    raise RuntimeError(f"only found {len(out)} unique opponent tapes")


def run_game(assets: Path, opponent: dict, seat: int) -> float:
    champion=BKG4ChampionAgent(assets);actions=opponent['actions']
    def tape_agent(obs, configuration=None):
        step=int(obs.get('step',0) or 0);return actions[min(step,len(actions)-1)]
    env=make('kaggriculture',configuration={'seed':opponent['seed'],'episodeSteps':720},debug=True)
    env.run([champion,tape_agent] if seat==0 else [tape_agent,champion])
    rewards=[float(s.reward) for s in env.steps[-1]]
    return rewards[seat]-rewards[1-seat]


def main()->int:
    ap=argparse.ArgumentParser();ap.add_argument('--games',type=int,default=64);ap.add_argument('--start',type=int,default=0);ap.add_argument('--count',type=int,default=0);ap.add_argument('--checkpoint',type=Path,default=Path('scratch/bkg4_search/official_canary64_checkpoint.jsonl'));a=ap.parse_args()
    if a.games<=0 or a.games%2:raise ValueError('--games must be a positive even number')
    assets=CHAMPION/'champions'/'feed_deferral_v1';identity=json.loads((assets/'identity.json').read_text())
    opponents=collect_opponents(a.games//2);tasks=[(oi,opp,seat) for oi,opp in enumerate(opponents) for seat in (0,1)]
    checkpoint=(ROOT/a.checkpoint).resolve() if not a.checkpoint.is_absolute() else a.checkpoint;checkpoint.parent.mkdir(parents=True,exist_ok=True)
    done={}
    if checkpoint.exists():
        for line in checkpoint.read_text(encoding='utf-8').splitlines():
            if line.strip():
                row=json.loads(line);done[int(row['game_index'])]=row
    start=max(0,a.start);end=min(a.games,start+(a.count if a.count>0 else a.games-start))
    for game_index in range(start,end):
        if game_index in done:continue
        oi,opp,seat=tasks[game_index];margin=run_game(assets,opp,seat)
        row={'game_index':game_index,'opponent_index':oi,'team':opp['team'],'seed':opp['seed'],'seat':seat,'margin':margin,'tape_sha256':opp['sha256']}
        with checkpoint.open('a',encoding='utf-8') as f:f.write(json.dumps(row,ensure_ascii=False,sort_keys=True)+'\n');f.flush()
        done[game_index]=row;print(json.dumps(row,ensure_ascii=False,sort_keys=True),flush=True)
    rows=[done[i] for i in range(a.games) if i in done];w=sum(x['margin']>0 for x in rows);l=sum(x['margin']<0 for x in rows);t=sum(x['margin']==0 for x in rows)
    report={'schema':1,'target_games':a.games,'completed_games':len(rows),'complete':len(rows)==a.games,'wins':w,'losses':l,'ties':t,'win_rate':w/len(rows) if rows else 0.0,'families':dict(Counter(x['team'] for x in rows)),'unique_opponent_tapes':len({x['opponent_index'] for x in rows}),'semantic_hash':identity['controller_semantic_hash'],'combined_identity':identity['combined_agent_identity'],'checkpoint':str(checkpoint),'rows':rows}
    print('FINAL '+json.dumps(report,ensure_ascii=False,sort_keys=True));return 0

if __name__=='__main__':raise SystemExit(main())
