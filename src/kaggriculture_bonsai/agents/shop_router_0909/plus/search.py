"""BUILD-only deterministic search over donor plans per shop-pair regime.

The donor's own two-shop table was tuned against a different cohort. Every donor
tape is already a complete legal schedule, so the cheapest real improvement is to
re-choose the plan per observable shop-pair regime. Search runs on BUILD seeds
only; screening and the final panel never feed back into it.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict

from kaggriculture_bonsai.runtime import bonsai_kg_v4 as bkg

from .agent import DONOR_ROUTES
from .benchmark import EVIDENCE, PANELS, _one

PAIRS = [(a, b) for a in bkg.SHOPS for b in bkg.SHOPS]


def forced(plan):
    """A candidate whose every shop-pair regime selects one donor plan."""
    return {"routes": {f"{a}|{b}": plan for a, b in PAIRS}}


def subject(spec):
    return "candidate:" + json.dumps(spec, sort_keys=True, separators=(",", ":"))


def sweep(seeds, seats, targets, plans, workers=None):
    """One row per (plan, target, seed, seat). BUILD evidence only."""
    import os
    from concurrent.futures import ProcessPoolExecutor
    jobs = [(subject(forced(plan)), target, seed, seat)
            for plan in plans for target in targets for seed in seeds for seat in seats]
    with ProcessPoolExecutor(max_workers=workers or os.cpu_count()) as pool:
        rows = list(pool.map(_one, jobs, chunksize=1))
    for row, job in zip(rows, jobs):
        row["plan"] = json.loads(job[0].split(":", 1)[1])["routes"]["BAKERY|BAKERY"]
    return rows


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--seeds", type=int, default=32)
    parser.add_argument("--targets", default="new_notebook,gold_v25_plus,gold_v25")
    parser.add_argument("--plans", default=",".join(str(p) for p in range(13) if p != 2))
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--out", default="search/plan_sweep.json")
    args = parser.parse_args()
    seeds = list(PANELS["build"])[:args.seeds]
    seats = (0, 1)
    targets = tuple(args.targets.split(","))
    plans = [int(p) for p in args.plans.split(",")]
    rows = sweep(seeds, seats, targets, plans, args.workers or None)
    by = defaultdict(lambda: defaultdict(lambda: {"W": 0, "T": 0, "L": 0}))
    for row in rows:
        by[tuple(row["pair"])][row["plan"]][row["result"]] += 1
    summary = {"|".join(pair) or "<none>": {str(plan): counts for plan, counts in sorted(plans_.items())}
               for pair, plans_ in sorted(by.items())}
    donor = {"|".join(k): v for k, v in DONOR_ROUTES.items()}
    payload = {"panel": "build", "seeds": [seeds[0], seeds[-1]], "targets": list(targets),
               "plans": plans, "donor_routes": donor, "by_regime": summary, "rows": rows}
    path = EVIDENCE / args.out
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"regimes": {k: {p: c for p, c in v.items()} for k, v in summary.items()}}, indent=1))


if __name__ == "__main__":
    main()
