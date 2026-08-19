from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from tools.bkg4_search.campaign import broad_guard_seed_layer, fitness, with_complexity
from tools.bkg4_search.cuda_io import load_results
from tools.bkg4_search.make_genomes import ROUTES
from tools.bkg4_search.module_codec import ConditionGene, ModuleGenome, RuleGene, save_genome

ROOT = Path(__file__).resolve().parents[2]
NONE = 0xFFFF


def _mask(row) -> int:
    m = 0
    for i, (_, _, route, flags) in enumerate(row):
        if flags:
            raise RuntimeError(f"CUDA result flags={flags}")
        if route != NONE:
            m |= 1 << i
    return m


def _condition_key(c: ConditionGene) -> Tuple:
    return (c.feature, c.cmp, c.index, c.value)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed-results", type=Path, required=True, help="directory containing chunked seed_grid .b4rs files")
    ap.add_argument("--oracle-results", type=Path, required=True)
    ap.add_argument("--cases", type=Path, required=True)
    ap.add_argument("--prefix", type=int, default=72)
    ap.add_argument("--min-seeds", type=int, default=2)
    ap.add_argument("--min-families", type=int, default=2)
    ap.add_argument("--top", type=int, default=40)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    out = (ROOT / args.out).resolve() if not args.out.is_absolute() else args.out
    out.mkdir(parents=True, exist_ok=True)
    cases_path = (ROOT / args.cases).resolve() if not args.cases.is_absolute() else args.cases
    cases = [json.loads(x) for x in cases_path.read_text().splitlines() if x.strip()]
    baseline = [int(round(c["base_margin"])) for c in cases]
    indices = list(range(len(cases)))
    base_f = with_complexity(fitness(baseline, indices, cases, baseline), ModuleGenome(args.prefix, ()))

    seeds = broad_guard_seed_layer(args.prefix)
    result_dir = (ROOT / args.seed_results).resolve() if not args.seed_results.is_absolute() else args.seed_results
    chunks = sorted(result_dir.glob("batch_*seed_grid*.b4rs"))
    rows = []
    for p in chunks:
        rows.extend(load_results(p))
    if len(rows) != len(seeds):
        raise RuntimeError(f"seed/result mismatch {len(rows)} != {len(seeds)}")

    # Truth masks are route-independent. Verify repeated observations agree and keep
    # one representative per semantic condition and per distinct finite-corpus mask.
    by_shop_cond: Dict[int, Dict[Tuple, Tuple[ConditionGene, int]]] = {4: {}, 5: {}}
    for g, row in zip(seeds, rows):
        if len(g.rules) != 1 or len(g.rules[0].conditions) != 2:
            continue
        first, guard = g.rules[0].conditions
        if first.feature != "SHOP_AT" or first.cmp != "EQ" or first.value not in (4, 5):
            continue
        k = _condition_key(guard)
        m = _mask(row)
        old = by_shop_cond[first.value].get(k)
        if old is not None and old[1] != m:
            raise RuntimeError(f"route-dependent truth mask for {k}")
        by_shop_cond[first.value][k] = (guard, m)

    oracle = load_results((ROOT / args.oracle_results).resolve() if not args.oracle_results.is_absolute() else args.oracle_results)
    if len(oracle) != 1 + len(ROUTES):
        raise RuntimeError("oracle result must be base + one row per frozen route")
    route_margin: Dict[int, List[int]] = {}
    for route, row in zip(ROUTES, oracle[1:]):
        route_margin[route] = [own - opp for own, opp, _, flags in row]
        if any(flags for _, _, _, flags in row):
            raise RuntimeError("oracle flags")

    candidates = []
    for shop in (4, 5):
        # Deduplicate conditions with identical truth masks to avoid combinatorial
        # redundancy. Keep deterministic lexical representative for each mask.
        reps: Dict[int, ConditionGene] = {}
        for c, m in sorted(by_shop_cond[shop].values(), key=lambda x: _condition_key(x[0])):
            if m and m not in reps:
                reps[m] = c
        entries = [(c, m) for m, c in reps.items()]
        for ai in range(len(entries)):
            ca, ma = entries[ai]
            for bi in range(ai + 1, len(entries)):
                cb, mb = entries[bi]
                active = ma & mb
                if not active:
                    continue
                active_idx = [i for i in indices if active >> i & 1]
                seeds_hit = {int(cases[i]["seed"]) for i in active_idx}
                fam_hit = {str(cases[i]["team"]) for i in active_idx}
                if len(seeds_hit) < args.min_seeds or len(fam_hit) < args.min_families:
                    continue
                for route in ROUTES:
                    margins = list(baseline)
                    rm = route_margin[route]
                    for i in active_idx:
                        margins[i] = rm[i]
                    g = ModuleGenome(args.prefix, (RuleGene((ConditionGene("SHOP_AT", "EQ", 0, shop), ca, cb), route),))
                    f = with_complexity(fitness(margins, indices, cases, baseline), g)
                    if f.catastrophic_regressions or f.protected_regressions or f.own_history_regressions:
                        continue
                    if f.lexicographic_key() >= base_f.lexicographic_key():
                        continue
                    candidates.append((f.lexicographic_key(), g, f, active_idx, margins))

    candidates.sort(key=lambda x: x[0])
    report = []
    for rank, (_, g, f, active_idx, margins) in enumerate(candidates[: args.top], 1):
        path = out / f"candidate_{rank:03d}.json"
        save_genome(g, path)
        report.append({
            "rank": rank,
            "genome_id": g.semantic_id(),
            "genome": g.canonical(),
            "fitness": asdict(f),
            "active_cases": active_idx,
            "active_seeds": sorted({int(cases[i]["seed"]) for i in active_idx}),
            "active_families": sorted({str(cases[i]["team"]) for i in active_idx}),
            "converted_losses": sum(baseline[i] <= 0 < margins[i] for i in active_idx),
            "lost_wins": sum(baseline[i] > 0 >= margins[i] for i in active_idx),
            "mean_delta_active": sum(margins[i] - baseline[i] for i in active_idx) / len(active_idx),
        })
    payload = {
        "schema": 1,
        "cases": len(cases),
        "seed_genomes": len(seeds),
        "unique_guard_masks": {str(shop): len({m for _, m in by_shop_cond[shop].values() if m}) for shop in (4, 5)},
        "min_seeds": args.min_seeds,
        "min_families": args.min_families,
        "viable_conjunctions": len(candidates),
        "top": report,
    }
    (out / "report.json").write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
