"""BUILD-only search over bounded terminal gates, on top of a fixed route table.

A gate is one step window and one bit mask of bounded repairs. It never uses
opponent identity: the controller reads STEP only, and the repairs read the
candidate's own farm, its own projected shed and public prices.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

from .benchmark import EVIDENCE, PANELS, TARGETS, _one

# Step 718 is always the donor liquidation, so gates close at 717.
WINDOWS = ((716, 717), (712, 717), (704, 717), (690, 717), (648, 717), (576, 717),
           (144, 647), (144, 717))
MODES = (1, 2, 4, 8, 16, 5, 6, 18, 20, 24, 26, 30)


def subject(spec):
    return "candidate:" + json.dumps(spec, sort_keys=True, separators=(",", ":"))


def specs(routes, windows=WINDOWS, modes=MODES, rule_routes=(), base=None):
    """Sweep one gate rule; `rule_routes` restricts it to named plan regimes.

    With a `base` rule the swept rule is tried ahead of it, so regimes outside
    `rule_routes` keep the already-screened behaviour unchanged.
    """
    out = []
    for start, end in windows:
        for mode in modes:
            rule = {"start": start, "end": end, "mode": mode}
            if rule_routes:
                rule["routes"] = list(rule_routes)
            gate = [rule, base] if base else rule
            out.append({"routes": dict(routes or {}), "endgame": gate})
    return out


def evaluate_specs(candidates, seeds, targets, seats=(0, 1), workers=None):
    jobs, index = [], []
    for i, spec in enumerate(candidates):
        name = subject(spec)
        for target in targets:
            for seed in seeds:
                for seat in seats:
                    jobs.append((name, target, seed, seat))
                    index.append(i)
    grouped = defaultdict(list)
    with ProcessPoolExecutor(max_workers=workers or os.cpu_count()) as pool:
        futures = {pool.submit(_one, job): i for job, i in zip(jobs, index)}
        for done, future in enumerate(as_completed(futures), 1):
            grouped[futures[future]].append(future.result())
            if done % 32 == 0 or done == len(jobs):
                print(f"  {done}/{len(jobs)} games", flush=True)
    results = []
    for i, spec in enumerate(candidates):
        per_target = {}
        for target in targets:
            counts = {"W": 0, "T": 0, "L": 0}
            for row in grouped[i]:
                if row["target"] == target:
                    counts[row["result"]] += 1
            per_target[target] = counts
        results.append({"spec": spec, "summary": per_target, "rows": grouped[i]})
    return results


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--seeds", type=int, default=12)
    parser.add_argument("--panel", default="build")
    parser.add_argument("--targets", default=",".join(TARGETS))
    parser.add_argument("--routes", default="{}")
    parser.add_argument("--windows", default="")
    parser.add_argument("--modes", default="")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--rule-routes", default="")
    parser.add_argument("--base", default="")
    parser.add_argument("--seed-list", default="")
    parser.add_argument("--out", default="search/gate_sweep.json")
    args = parser.parse_args()
    seeds = ([int(v) for v in args.seed_list.split(",")] if args.seed_list
             else list(PANELS[args.panel])[:args.seeds])
    targets = tuple(args.targets.split(","))
    routes = json.loads(args.routes)
    windows = tuple(tuple(int(v) for v in w.split("-")) for w in args.windows.split(",")) if args.windows else WINDOWS
    modes = tuple(int(m) for m in args.modes.split(",")) if args.modes else MODES
    rule_routes = [int(v) for v in args.rule_routes.split(",")] if args.rule_routes else ()
    base = json.loads(args.base) if args.base else None
    candidates = specs(routes, windows, modes, rule_routes, base)
    results = evaluate_specs(candidates, seeds, targets, workers=args.workers or None)
    table = [{"endgame": r["spec"]["endgame"],
              "summary": {t: (c["W"], c["T"], c["L"]) for t, c in r["summary"].items()}} for r in results]
    path = EVIDENCE / args.out
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"panel": args.panel, "seeds": seeds,
                                "targets": list(targets), "routes": routes,
                                "rule_routes": list(rule_routes), "base": base,
                                "results": results}, indent=2) + "\n")
    for row in sorted(table, key=lambda r: -sum(v[0] for v in r["summary"].values())):
        rule = row["endgame"][0] if isinstance(row["endgame"], list) else row["endgame"]
        print(rule, row["summary"], flush=True)


if __name__ == "__main__":
    main()
