#!/usr/bin/env python3
"""Paired-seat exact-engine arena with a hard v27 domination gate."""
from __future__ import annotations

import argparse
import importlib.util
import inspect
import json
import multiprocessing as mp
from pathlib import Path

from official_oracle import run


def load_agent(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    fn = mod.agent
    sig = inspect.signature(fn)
    positional = [
        p for p in sig.parameters.values()
        if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    has_varargs = any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in sig.parameters.values())
    if has_varargs or len(positional) >= 2:
        return fn
    if len(positional) == 1:
        def adapted(obs, configuration=None):
            return fn(obs)
        return adapted
    raise RuntimeError(f"unsupported agent signature for {path}: {sig}")


def one_game(args):
    candidate_path, baseline_path, seed, candidate_seat = args
    candidate = load_agent(candidate_path, f"candidate_{seed}_{candidate_seat}")
    baseline = load_agent(baseline_path, f"baseline_{seed}_{candidate_seat}")
    if candidate_seat == 0:
        reward, _, _ = run(candidate, baseline, seed)
        margin = reward[0] - reward[1]
    else:
        reward, _, _ = run(baseline, candidate, seed)
        margin = reward[1] - reward[0]
    return {"seed": seed, "seat": candidate_seat, "margin": margin}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--seed-start", type=int, default=2000)
    ap.add_argument("--seeds", type=int, default=16)
    ap.add_argument("--workers", type=int, default=max(1, min(8, mp.cpu_count())))
    ap.add_argument("--require-100", action="store_true", help="fail on any tie or loss")
    ap.add_argument("--json", action="store_true")
    ns = ap.parse_args()
    jobs = [
        (ns.candidate, ns.baseline, seed, seat)
        for seed in range(ns.seed_start, ns.seed_start + ns.seeds)
        for seat in (0, 1)
    ]
    with mp.get_context("spawn").Pool(ns.workers) as pool:
        rows = pool.map(one_game, jobs)
    wins = sum(r["margin"] > 0 for r in rows)
    losses = sum(r["margin"] < 0 for r in rows)
    ties = len(rows) - wins - losses
    result = {
        "games": len(rows), "wins": wins, "losses": losses, "ties": ties,
        "win_rate": wins / len(rows),
        "mean_margin": sum(r["margin"] for r in rows) / len(rows),
        "min_margin": min(r["margin"] for r in rows),
        "rows": rows,
    }
    print(json.dumps(result, indent=2 if ns.json else None))
    if ns.require_100 and (losses or ties):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
