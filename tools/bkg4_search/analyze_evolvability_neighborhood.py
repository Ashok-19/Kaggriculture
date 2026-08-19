from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from tools.bkg4_search.analyze_mechanics import load_genomes, load_results
from tools.bkg4_search.campaign import fitness
from tools.bkg4_search.evolvability import EvolvabilityTree, direct_scalar

ROOT = Path(__file__).resolve().parents[2]


def qtile(xs: list[float], q: float) -> float:
    if not xs:
        return 0.0
    ys = sorted(xs)
    pos = q * (len(ys) - 1)
    lo = int(pos); hi = min(len(ys) - 1, lo + 1); frac = pos - lo
    return ys[lo] * (1.0 - frac) + ys[hi] * frac


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--genomes", type=Path, required=True)
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--cases", type=Path, required=True)
    ap.add_argument("--candidate-index", type=int, default=1)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    gp = (ROOT / args.genomes).resolve() if not args.genomes.is_absolute() else args.genomes
    rp = (ROOT / args.results).resolve() if not args.results.is_absolute() else args.results
    cp = (ROOT / args.cases).resolve() if not args.cases.is_absolute() else args.cases
    genomes = load_genomes(gp); rows = load_results(rp); cases = [json.loads(x) for x in cp.read_text().splitlines() if x.strip()]
    baseline = [a - b for a, b, _, _ in rows[0]]; idx = list(range(len(cases)))
    base_fit = fitness(baseline, idx, cases, baseline)
    ci = args.candidate_index
    candidate_margins = [a - b for a, b, _, _ in rows[ci]]
    candidate_fit = fitness(candidate_margins, idx, cases, baseline)
    tree = EvolvabilityTree(candidate_fit, base_fit)
    descendants = []
    safe = 0; same_outcome = 0; elite_equal = 0; unsafe = 0
    candidate_key = candidate_fit.lexicographic_key()
    for i, (g, row) in enumerate(zip(genomes, rows)):
        if i in (0, ci):
            continue
        margins = [a - b for a, b, _, _ in row]
        f = fitness(margins, idx, cases, baseline)
        v = direct_scalar(f, base_fit)
        promotion_safe = f.catastrophic_regressions == 0 and f.protected_regressions == 0 and f.own_history_regressions == 0
        if promotion_safe:
            safe += 1
        else:
            unsafe += 1
        if f.wins >= candidate_fit.wins and f.losses <= candidate_fit.losses and promotion_safe:
            same_outcome += 1
        if f.lexicographic_key() == candidate_key:
            elite_equal += 1
        if v > 0.0:
            tree.add_accepted(0, f)
            descendants.append(v)
        else:
            tree.add_rejected_visit(0)
    root = tree.nodes[0]
    payload = {
        "schema": 1,
        "neighbor_count": max(0, len(genomes) - 2),
        "promotion_safe_neighbors": safe,
        "unsafe_neighbors": unsafe,
        "safe_fraction": safe / max(1, len(genomes) - 2),
        "candidate_outcome_or_better_neighbors": same_outcome,
        "lexicographically_identical_elite_neighbors": elite_equal,
        "candidate_v": root.v,
        "lineage_Q": root.q,
        "sigma_v_minus_Q": root.brittleness,
        "accepted_descendant_samples": len(descendants),
        "rejected_visits": unsafe,
        "descendant_v_mean": statistics.mean(descendants) if descendants else 0.0,
        "descendant_v_median": statistics.median(descendants) if descendants else 0.0,
        "descendant_v_q10": qtile(descendants, 0.10),
        "descendant_v_q25": qtile(descendants, 0.25),
        "base_fitness": base_fit.__dict__,
        "candidate_fitness": candidate_fit.__dict__,
    }
    if args.out:
        out = (ROOT / args.out).resolve() if not args.out.is_absolute() else args.out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n")
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
