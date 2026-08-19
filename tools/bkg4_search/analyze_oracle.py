from __future__ import annotations

import argparse
import json
from pathlib import Path

from tools.bkg4_search.cuda_io import load_results
from tools.bkg4_search.make_genomes import ROUTES


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--cases", type=Path, required=True)
    args = ap.parse_args()
    rows = [json.loads(x) for x in args.cases.read_text(encoding="utf-8").splitlines() if x.strip()]
    result = load_results(args.results)
    if len(result) != 1 + len(ROUTES) or len(result[0]) != len(rows):
        raise RuntimeError("oracle result shape mismatch")
    baseline_bad = []
    details = []
    for ci, case in enumerate(rows):
        own, opp, route, flags = result[0][ci]
        margin = own - opp
        expected = int(round(case["base_margin"]))
        if margin != expected or flags:
            baseline_bad.append((ci, margin, expected, flags))
        candidates = []
        for gi, rid in enumerate(ROUTES, 1):
            co, cp, selected, cflags = result[gi][ci]
            candidates.append((co - cp, rid, selected, cflags))
        best = max(candidates, key=lambda z: (z[0], -z[1]))
        details.append({
            "case": ci,
            "team": case["team"],
            "seat": case["seat"],
            "eligible": bool(case.get("search_eligible")) and case.get("external_route") is None,
            "base_margin": expected,
            "best_margin": best[0],
            "best_route": best[1],
            "regret": best[0] - expected,
            "base_win": expected > 0,
            "oracle_win": best[0] > 0,
        })
    eligible = [x for x in details if x["eligible"]]
    losses = [x for x in eligible if x["base_margin"] < 0]
    summary = {
        "baseline_exact": not baseline_bad,
        "baseline_mismatches": baseline_bad[:8],
        "cases": len(rows),
        "eligible": len(eligible),
        "eligible_base_losses": len(losses),
        "oracle_converted_losses": sum(x["oracle_win"] for x in losses),
        "mean_regret": sum(x["regret"] for x in eligible) / max(1, len(eligible)),
        "median_regret": sorted(x["regret"] for x in eligible)[len(eligible)//2] if eligible else 0,
        "max_regret": max((x["regret"] for x in eligible), default=0),
        "details": details,
    }
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0 if not baseline_bad else 2


if __name__ == "__main__":
    raise SystemExit(main())
