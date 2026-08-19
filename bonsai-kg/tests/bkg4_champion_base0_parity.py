from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
KAG_ENV = ROOT / "kaggle-environments-master"
CHAMPION = ROOT / "bonsai-kg" / "integrations" / "bkg4_champion"
OLD = ROOT / "scratch" / "bkg_v6_engine_exploit_mining_20260819" / "h4max_4branch_engine_max.py"
sys.path.insert(0, str(KAG_ENV))
sys.path.insert(0, str(CHAMPION))

from kaggle_environments import make
from runtime import BKG4ChampionAgent


def load_old():
    spec = importlib.util.spec_from_file_location("bkg4_base0_old_engine_max", OLD)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load old engine-max baseline")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed-start", type=int, default=1290000)
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--steps", type=int, default=720)
    args = ap.parse_args()
    mismatches = []
    reward_mismatches = []
    ticks = 0
    games = 0
    for si in range(args.seeds):
        seed = args.seed_start + si
        for seat in (0, 1):
            old = load_old()
            observations = []
            old_actions = []

            def recorder(observation, configuration=None):
                observations.append(copy.deepcopy(dict(observation)))
                action = old.agent(observation, configuration)
                old_actions.append(copy.deepcopy(action))
                return action

            env = make(
                "kaggriculture",
                configuration={"episodeSteps": args.steps, "seed": seed},
                debug=True,
            )
            env.run([recorder, "pass"] if seat == 0 else ["pass", recorder])
            final = env.steps[-1]
            old_rewards = [float(s.reward) for s in final]

            new = BKG4ChampionAgent()
            for t, (observation, expected) in enumerate(zip(observations, old_actions)):
                got = new(observation, env.configuration)
                ticks += 1
                if got != expected:
                    mismatches.append({
                        "seed": seed,
                        "seat": seat,
                        "step": t,
                        "expected": expected,
                        "got": got,
                    })
                    break

            # If every action matches, rewards are necessarily identical under the same
            # deterministic engine. Keep the explicit reward record for auditability.
            if not mismatches or mismatches[-1].get("seed") != seed or mismatches[-1].get("seat") != seat:
                reward_mismatches.extend([] if len(old_rewards) == 2 else [(seed, seat, old_rewards)])
            games += 1

    report = {
        "games": games,
        "ticks_compared": ticks,
        "action_mismatches": len(mismatches),
        "reward_record_errors": len(reward_mismatches),
        "first_mismatch": mismatches[0] if mismatches else None,
    }
    print(json.dumps(report, sort_keys=True))
    return 1 if mismatches or reward_mismatches else 0


if __name__ == "__main__":
    raise SystemExit(main())
