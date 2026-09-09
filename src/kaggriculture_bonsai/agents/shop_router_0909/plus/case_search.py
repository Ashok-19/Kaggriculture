"""Evaluate candidate specs on explicit BUILD cases: known failures plus controls.

Failure-directed search stays honest only if every candidate is also measured on
cases the incumbent already wins, so each run carries its own control set.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

from .benchmark import EVIDENCE, _one


def subject(spec):
    return "candidate:" + json.dumps(spec, sort_keys=True, separators=(",", ":"))


def evaluate(specs, cases, workers=None):
    jobs, index = [], []
    for i, spec in enumerate(specs):
        for target, seed, seat in cases:
            jobs.append((subject(spec), target, seed, seat))
            index.append(i)
    rows = defaultdict(list)
    with ProcessPoolExecutor(max_workers=workers or os.cpu_count()) as pool:
        for done, (i, row) in enumerate(zip(index, pool.map(_one, jobs, chunksize=1)), 1):
            rows[i].append(row)
            if done % 24 == 0 or done == len(jobs):
                print(f"  {done}/{len(jobs)} games", flush=True)
    out = []
    for i, spec in enumerate(specs):
        counts = defaultdict(int)
        for row in rows[i]:
            counts[row["result"]] += 1
        out.append({"spec": spec, "W": counts["W"], "T": counts["T"], "L": counts["L"],
                    "losses": [(r["target"], r["seed"], r["seat"]) for r in rows[i] if r["result"] != "W"],
                    "rows": rows[i]})
    return out


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--specs", required=True, help="JSON list of candidate specs")
    parser.add_argument("--cases", required=True, help="JSON list of [target, seed, seat]")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    specs = json.loads(args.specs)
    cases = [tuple(c) for c in json.loads(args.cases)]
    results = evaluate(specs, cases, args.workers or None)
    path = EVIDENCE / args.out
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"cases": cases, "results": results}, indent=2) + "\n")
    for r in results:
        print(json.dumps(r["spec"], sort_keys=True), r["W"], r["T"], r["L"], r["losses"], flush=True)


if __name__ == "__main__":
    main()
