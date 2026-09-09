"""Map panel seeds to the shop-pair regime the parent latches at step 144.

The regime is fixed before any candidate diverges, so one short probe per seed
is enough. Used to build regime-balanced BUILD and INTERNAL_DEV case sets.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

from .agent import DONOR_ROUTES
from .benchmark import EVIDENCE, PANELS, build_agent


def _probe(seed):
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        from kaggle_environments import make
        seen, inner = [], build_agent("parent")

        def watch(obs, cfg=None):
            if int(obs["step"]) == 144 and not seen:
                seen.append(list(obs["town"]["unlocked_shops"][:2]))
            return inner(obs, cfg)
        env = make("kaggriculture", configuration={"episodeSteps": 146, "seed": seed}, debug=True)
        env.run([watch, build_agent("parent")])
    pair = tuple(seen[0]) if seen else ()
    return seed, list(pair), DONOR_ROUTES.get(pair, 0)


def scan(seeds, workers=None):
    with ProcessPoolExecutor(max_workers=workers or os.cpu_count()) as pool:
        return list(pool.map(_probe, seeds, chunksize=2))


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--seeds", default="2000-2400")
    parser.add_argument("--out", default="panels/regimes.json")
    args = parser.parse_args()
    lo, hi = (int(v) for v in args.seeds.split("-"))
    rows = scan(range(lo, hi))
    by = defaultdict(list)
    for seed, pair, plan in rows:
        by[plan].append(seed)
    payload = {"seeds": [lo, hi - 1], "by_plan": {str(k): v for k, v in sorted(by.items())},
               "rows": [{"seed": s, "pair": p, "plan": n} for s, p, n in rows]}
    path = EVIDENCE / args.out
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({k: len(v) for k, v in sorted(by.items())}, indent=1))
    for plan, seeds in sorted(by.items()):
        print(plan, seeds[:14])


if __name__ == "__main__":
    main()
