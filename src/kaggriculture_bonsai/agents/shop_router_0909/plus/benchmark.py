"""Disjoint BUILD / INTERNAL_DEV / FINAL_TARGET evaluation against frozen targets.

Panels never overlap: BUILD may drive search, INTERNAL_DEV only screens frozen
candidates, and FINAL_TARGET is untouched until the composition is frozen.
"""
from __future__ import annotations

import argparse
import contextlib
import importlib
import io
import json
import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from ..reference_port.extract import ROOT

EVIDENCE = ROOT / "workspace/shop_router_0909"
PANELS = {"build": range(2000, 2200), "dev": range(3000, 3100), "final": range(5000, 5064)}
TARGETS = ("new_notebook", "gold_v25_plus", "gold_v25")


def build_agent(name):
    """Frozen opponents and candidates, each constructed fresh per game."""
    if name == "new_notebook":
        from ..reference_port.extract import reference
        return reference()["agent"]
    if name == "parent":
        from ..reference_port.agent import ShopRouter0909Agent
        return ShopRouter0909Agent()
    if name == "gold_v25":
        from kaggriculture_bonsai.agents.gold_v25.agent import GoldV25Agent
        return GoldV25Agent()
    if name == "gold_v25_plus":
        from kaggriculture_bonsai.agents.gold_v25_plus.agent import agent
        return agent
    if name.startswith("candidate:"):
        from .agent import ShopRouter0909Plus
        return ShopRouter0909Plus(json.loads(name.split(":", 1)[1]))
    module, _, attribute = name.rpartition(".")
    return getattr(importlib.import_module(module), attribute)


def _one(job):
    """One official game. The latched shop pair is recorded as regime evidence."""
    subject, target, seed, seat = job
    pair = []
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        from kaggle_environments import make
        inner = build_agent(subject)

        def watched(obs, cfg=None):
            if int(obs["step"]) == 144 and not pair:
                pair.extend(list(obs["town"]["unlocked_shops"][:2]))
            return inner(obs, cfg)
        agents = [None, None]
        agents[seat] = watched
        agents[1 - seat] = build_agent(target)
        env = make("kaggriculture", configuration={"episodeSteps": 720, "seed": seed}, debug=True)
        env.run(agents)
    rewards = [s.reward for s in env.state]
    status = [s.status for s in env.state]
    if status != ["DONE", "DONE"]:
        raise AssertionError((subject, target, seed, seat, status))
    mine, theirs = rewards[seat], rewards[1 - seat]
    return {"target": target, "seed": seed, "seat": seat, "subject": mine, "opponent": theirs,
            "pair": pair, "result": "W" if mine > theirs else ("L" if mine < theirs else "T")}


def evaluate(subject, targets=TARGETS, seeds=(), seats=(0, 1), workers=None):
    jobs = [(subject, t, s, seat) for t in targets for s in seeds for seat in seats]
    workers = workers or max(1, min(len(jobs), (os.cpu_count() or 2)))
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for done, row in enumerate(pool.map(_one, jobs, chunksize=1), 1):
            rows.append(row)
            if done % 24 == 0 or done == len(jobs):
                print(f"  {done}/{len(jobs)} games", flush=True)
    summary = {}
    for target in targets:
        counts = Counter(r["result"] for r in rows if r["target"] == target)
        summary[target] = {"W": counts["W"], "T": counts["T"], "L": counts["L"]}
    failures = [{k: r[k] for k in ("target", "seed", "seat", "result", "pair", "subject", "opponent")}
                for r in rows if r["result"] != "W"]
    return {"subject": subject, "games": len(rows), "summary": summary,
            "failures": failures, "rows": rows}


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("panel", choices=tuple(PANELS))
    parser.add_argument("--subject", default="parent")
    parser.add_argument("--targets", default=",".join(TARGETS))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--out", default="")
    parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args()
    seeds = list(PANELS[args.panel])
    if args.limit:
        seeds = seeds[:args.limit]
    result = evaluate(args.subject, tuple(args.targets.split(",")), seeds, workers=args.workers or None)
    result["panel"] = args.panel
    result["seeds"] = [seeds[0], seeds[-1]]
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2))
    if args.out:
        path = EVIDENCE / args.out
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
