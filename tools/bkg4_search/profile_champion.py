from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
KAG_ENV = ROOT / "kaggle-environments-master"
BKG_PY = ROOT / "bonsai-kg" / "python"
CHAMPION = ROOT / "bonsai-kg" / "integrations" / "bkg4_champion"
for p in (KAG_ENV, BKG_PY, CHAMPION): sys.path.insert(0, str(p))

from kaggle_environments import make
import bonsai_kg_v4 as bkg
from runtime import BKG4ChampionAgent


def pct(xs: list[int], q: float) -> int:
    ys = sorted(xs)
    return ys[min(len(ys)-1, max(0, math.ceil(q * len(ys)) - 1))] if ys else 0


def main() -> int:
    ap=argparse.ArgumentParser();ap.add_argument('--seed-start',type=int,default=1830000);ap.add_argument('--seeds',type=int,default=2);ap.add_argument('--assets',type=Path,default=CHAMPION/'champions'/'feed_deferral_v1');a=ap.parse_args()
    assets=a.assets if a.assets.is_absolute() else (ROOT/a.assets).resolve();cp=bkg.load_compiled(assets/'controller.bkgc');visits=[];features=[];depth=[];active_ticks=0;authorized=0;ticks=0
    for k in range(a.seeds):
        seed=a.seed_start+k
        for seat in (0,1):
            champion=BKG4ChampionAgent(assets);vm=bkg.VM(cp)
            def agent(raw,configuration=None):
                nonlocal active_ticks,authorized,ticks
                r=vm.tick(raw,max_market_orders=0,profile=True);p=r.profile;visits.append(p.visits);features.append(p.feature_evals);depth.append(p.max_depth);n=sum(bool(x) for x in r.blackboard[5:18]);active_ticks+=n>0;authorized+=n;ticks+=1;return champion(raw,configuration)
            env=make('kaggriculture',configuration={'seed':seed,'episodeSteps':720},debug=True);env.run([agent,'pass'] if seat==0 else ['pass',agent])
    report={'schema':1,'games':a.seeds*2,'ticks':ticks,'nodes':len(cp.nodes),'state_slots':cp.state_count,'bb_slots':len(cp.bb),'static_max_visits':cp.analysis[0].max_visits,'max_live_intents':cp.max_live_intents,'visits':{'p50':pct(visits,.50),'p95':pct(visits,.95),'p99':pct(visits,.99),'max':max(visits)},'feature_evals':{'p50':pct(features,.50),'p95':pct(features,.95),'p99':pct(features,.99),'max':max(features)},'depth':{'p50':pct(depth,.50),'p95':pct(depth,.95),'p99':pct(depth,.99),'max':max(depth)},'feed_authorization_active_ticks':active_ticks,'feed_authorization_tick_rate':active_ticks/ticks,'authorized_unit_events':authorized}
    print(json.dumps(report,sort_keys=True));return 0

if __name__=='__main__':raise SystemExit(main())
