from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
KAG_ENV = ROOT / "kaggle-environments-master"
CHAMPION = ROOT / "bonsai-kg" / "integrations" / "bkg4_champion"
sys.path.insert(0, str(KAG_ENV))
sys.path.insert(0, str(CHAMPION))

from kaggle_environments import make
from runtime import BKG4ChampionAgent

HISTORICAL = (
    ("f120_04ad004", ROOT / "scratch" / "authoritative_self_agents" / "f120_04ad004"),
    ("v32_d52526d", ROOT / "scratch" / "authoritative_self_agents" / "v32_d52526d"),
)


def _purge(root: Path) -> None:
    names = {p.stem for p in root.glob("*.py")}
    q8 = root / "q8_shadow"
    if q8.is_dir():
        names.update(p.stem for p in q8.glob("*.py"))
    names.update({"historical_entry", "runtime_gateway"})
    for name in names:
        sys.modules.pop(name, None)


def load_source_agent(root: Path):
    _purge(root)
    sys.path.insert(0, str(root))
    try:
        spec = importlib.util.spec_from_file_location("historical_entry", root / "main.py")
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot load historical source {root}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.agent
    finally:
        sys.path.remove(str(root))


def run_game(asset_dir: Path, hist_root: Path, seed: int, seat: int) -> float:
    champion = BKG4ChampionAgent(asset_dir)
    historical = load_source_agent(hist_root)
    env = make("kaggriculture", configuration={"seed": seed, "episodeSteps": 720}, debug=True)
    env.run([champion, historical] if seat == 0 else [historical, champion])
    rewards = [float(s.reward) for s in env.steps[-1]]
    return rewards[seat] - rewards[1 - seat]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed-start", type=int, default=1820000)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--candidate-assets", type=Path, default=CHAMPION / "champions" / "feed_deferral_v1")
    args = ap.parse_args()
    base_dir = CHAMPION / "assets"
    cand_dir = args.candidate_assets if args.candidate_assets.is_absolute() else (ROOT / args.candidate_assets).resolve()
    rows = []
    regressions = 0
    improvements = 0
    for hi, (name, root) in enumerate(HISTORICAL):
        for si in range(args.seeds):
            seed = args.seed_start + hi * 1000 + si
            for seat in (0, 1):
                base = run_game(base_dir, root, seed, seat)
                cand = run_game(cand_dir, root, seed, seat)
                bo = 1 if base > 0 else (0 if base == 0 else -1)
                co = 1 if cand > 0 else (0 if cand == 0 else -1)
                regressions += co < bo
                improvements += co > bo
                rows.append({"history": name, "seed": seed, "seat": seat, "base": base, "candidate": cand, "delta": cand - base, "outcome_regression": co < bo, "outcome_improvement": co > bo})
    report = {
        "schema": 1,
        "games_per_agent": len(rows),
        "historical_versions": [x[0] for x in HISTORICAL],
        "regressions": regressions,
        "improvements": improvements,
        "mean_delta": sum(x["delta"] for x in rows) / len(rows),
        "min_delta": min(x["delta"] for x in rows),
        "rows": rows,
    }
    print(json.dumps(report, sort_keys=True))
    return 1 if regressions else 0


if __name__ == "__main__":
    raise SystemExit(main())
