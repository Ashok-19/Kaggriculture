#!/usr/bin/env python3
"""Self-contained parity checks for the optimized C++ bridge fast paths."""
from __future__ import annotations

import copy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "agent_system" / "tools"))

import cpp_oracle  # noqa: E402


def _route(phase: int):
    actions = []
    move = ["NORTH", "EAST", "SOUTH", "WEST"]
    for step in range(719):
        hour = step % 24
        market = []
        if hour == 0:
            market = [
                ["HIRE"], ["HIRE"], ["HIRE"],
                ["BUY_SEED", "WHEAT", 6 + phase],
                ["BUY_PRODUCT", "WHEAT", 4 + phase],
            ]
            if step in (0, 72, 144):
                market.append(["BUY_LAND"])
        elif hour == 8:
            market = [["BUY_SEED", "CARROT", 2 + phase]]
        elif hour == 16:
            market = [["SELL", "WHEAT", 3 + phase]]
        farmer = [move[(step + phase) % len(move)]] if hour not in (3, 9, 15, 21) else ["PASS"]
        hands = []
        for i in range(8):
            if hour in (1, 2) and i < 2:
                hands.append(["PLANT", "WHEAT"])
            elif hour in (4, 5) and i < 2:
                hands.append(["WATER"])
            else:
                hands.append([move[(step + phase + i) % len(move)]])
        actions.append({"farmer": farmer, "hands": hands, "market": market})
    return actions


def _tape_agent(route, log=None):
    def agent(obs, configuration=None):
        step = min(718, max(0, int(obs.get("step", 0) or 0)))
        action = copy.deepcopy(route[step])
        seat = int(obs.get("player", 0) or 0)
        farms = obs.get("farms", []) or []
        farm = farms[seat] if seat < len(farms) else {}
        hands = list(farm.get("hands", []) or [])
        planned = list(action.get("hands") or [])
        planned.extend([["PASS"]] * max(0, len(hands) - len(planned)))
        action["hands"] = planned[:len(hands)]
        action.setdefault("farmer", ["PASS"])
        action.setdefault("market", [])
        if log is not None:
            log.append(copy.deepcopy(action))
        return action
    return agent


def main() -> None:
    route0 = _route(0)
    route1 = _route(1)
    seeds = [11, 22, 33, 60200]

    python_rows = []
    for seed in seeds:
        python_rows.append(cpp_oracle.run(_tape_agent(route0), _tape_agent(route1), seed)[0])
    native1 = cpp_oracle.run_tapes_batch(route0, route1, seeds, workers=1)
    native2 = cpp_oracle.run_tapes_batch(route0, route1, seeds, workers=2)
    if python_rows != native1 or native1 != native2:
        raise AssertionError(f"compiled tape mismatch: python={python_rows} native1={native1} native2={native2}")
    print(f"PASS compiled tapes: {len(seeds)}/{len(seeds)} seeds exact, workers 1 == 2")

    log_det = [[], []]
    reward_det = cpp_oracle.run(
        _tape_agent(route0, log_det[0]), _tape_agent(route1, log_det[1]), 60200,
        shared_public=False,
    )[0]
    log_shared = [[], []]
    reward_shared = cpp_oracle.run(
        _tape_agent(route0, log_shared[0]), _tape_agent(route1, log_shared[1]), 60200,
        shared_public=True,
    )[0]
    if reward_det != reward_shared or log_det != log_shared:
        raise AssertionError("shared-public observation path changed actions or rewards")
    print("PASS shared-public observations: 1438/1438 actions exact")

    full = cpp_oracle.run_tapes_batch(route0, route1, [60200], workers=1)[0]
    snap0 = cpp_oracle.snapshot_vs_tape(_tape_agent(route0), route1, 60200, 0, 120)
    branch0 = cpp_oracle.branch_tape_candidates(snap0, [route0, route0], route1, 0, workers=2)
    snap1 = cpp_oracle.snapshot_vs_tape(_tape_agent(route1), route0, 60200, 1, 120)
    branch1 = cpp_oracle.branch_tape_candidates(snap1, [route1, route1], route0, 1, workers=2)
    if any(row != full for row in branch0 + branch1):
        raise AssertionError(f"snapshot branch mismatch: full={full} branch0={branch0} branch1={branch1}")
    print("PASS snapshot branching: both seats and two-worker duplicate continuations exact")


if __name__ == "__main__":
    main()
