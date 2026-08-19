from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
KAG_ENV = ROOT / "kaggle-environments-master"
CHAMPION = ROOT / "bonsai-kg" / "integrations" / "bkg4_champion"
sys.path.insert(0, str(KAG_ENV))
sys.path.insert(0, str(CHAMPION))

from kaggle_environments import make
from mechanics_reconciler import apply_feed_deferral
from runtime import BKG4ChampionAgent


def expected_units(observation: dict, limit: int = 13) -> list[int]:
    player = int(observation.get("player", 0) or 0)
    farms = list(observation.get("farms") or [])
    farm = farms[player] if 0 <= player < len(farms) else {}
    positions = [farm.get("farmer") or [-1, -1], *list(farm.get("hands") or [])]
    out = []
    tiles = farm.get("tiles") or []
    for unit, pos in enumerate(positions[:limit]):
        if not isinstance(pos, (list, tuple)) or len(pos) < 2:
            continue
        x, y = int(pos[0]), int(pos[1])
        tile = tiles[y][x] if 0 <= y < len(tiles) and 0 <= x < len(tiles[y]) else None
        if not isinstance(tile, dict):
            continue
        if (
            tile.get("animal") == "COW"
            and int(tile.get("consecutive_unfed", 0) or 0) == 0
            and int(tile.get("pending_care_bonus", 0) or 0) == 0
        ):
            out.append(unit)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed-start", type=int, default=1800100)
    ap.add_argument("--seeds", type=int, default=2)
    args = ap.parse_args()
    base_dir = CHAMPION / "assets"
    cand_dir = CHAMPION / "champions" / "feed_deferral_v1"
    mismatches = []
    ticks = 0
    rewrites = 0
    for k in range(args.seeds):
        seed = args.seed_start + k
        for seat in (0, 1):
            base = BKG4ChampionAgent(base_dir)
            cand = BKG4ChampionAgent(cand_dir)
            def tested(observation, configuration=None):
                nonlocal ticks, rewrites
                obs = copy.deepcopy(dict(observation))
                raw = base(obs, configuration)
                expected = expected_units(obs)
                slots = list(range(len(expected)))  # placeholder slots are not used below
                direct = copy.deepcopy(raw)
                farmer = list(direct.get("farmer") or ["PASS"])
                hands = [list(x) for x in list(direct.get("hands") or [])]
                for unit in expected:
                    if unit == 0 and farmer and farmer[0] == "FEED":
                        farmer = ["PASS"]; rewrites += 1
                    elif unit > 0 and unit - 1 < len(hands) and hands[unit - 1] and hands[unit - 1][0] == "FEED":
                        hands[unit - 1] = ["PASS"]; rewrites += 1
                direct["farmer"] = farmer; direct["hands"] = hands
                got = cand(obs, configuration)
                authorized = cand.state()["feed_defer_units"]
                ticks += 1
                if authorized != expected or got != direct:
                    mismatches.append({"seed": seed, "seat": seat, "step": int(obs.get("step", 0)), "expected_units": expected, "tree_units": authorized, "direct": direct, "got": got})
                return got
            env = make("kaggriculture", configuration={"seed": seed, "episodeSteps": 720}, debug=True)
            env.run([tested, "pass"] if seat == 0 else ["pass", tested])
    report = {"games": args.seeds * 2, "ticks": ticks, "rewrites": rewrites, "mismatches": len(mismatches), "first_mismatch": mismatches[0] if mismatches else None}
    print(json.dumps(report, sort_keys=True))
    return 1 if mismatches else 0


if __name__ == "__main__":
    raise SystemExit(main())
