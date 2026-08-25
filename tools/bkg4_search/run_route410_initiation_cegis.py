from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.bkg4_search.cuda_io import load_fragment_results
from tools.bkg4_search.edit_search.goals import CONTRASTIVE_FEATURES
from tools.bkg4_search.information_state_io import load_r0, registry_manifest
from tools.bkg4_search.macro_initiation_cegis import MacroEffectCase, evaluate_gate, selected_mixed_exact_cells
from tools.bkg4_search.module_codec import ConditionGene
from tools.bkg4_search.regret_macro_search import load_grid

PANELS = {
    "core585": (
        ROOT / "scratch/p0_3_repeatability/route410_18_24_core585.b4fr",
        ROOT / "scratch/p0_3_repeatability/route410_18_24_core585_grid.tsv",
        ROOT / "scratch/p0_3_repeatability/core585_step120.b4is",
        ROOT / "scratch/p0_3_repeatability/core585.tsv",
    ),
    "validation40": (
        ROOT / "scratch/p0_3_repeatability/frozen_validation40_route410.b4fr",
        ROOT / "scratch/p0_3_repeatability/frozen_validation40_route410_grid.tsv",
        ROOT / "scratch/p0_3_repeatability/frozen_validation40_step120.b4is",
        ROOT / "scratch/top10_archive_20260824/balanced_panels/validation.tsv",
    ),
    "sealed40": (
        ROOT / "scratch/p0_3_repeatability/frozen_sealed40_route410.b4fr",
        ROOT / "scratch/p0_3_repeatability/frozen_sealed40_route410_grid.tsv",
        ROOT / "scratch/p0_3_repeatability/frozen_sealed40_step120.b4is",
        ROOT / "scratch/top10_archive_20260824/balanced_panels/sealed.tsv",
    ),
    "fresh949": (
        ROOT / "scratch/p0_3_repeatability/frozen_fresh949_route410.b4fr",
        ROOT / "scratch/p0_3_repeatability/frozen_fresh949_route410_grid.tsv",
        ROOT / "scratch/p0_3_repeatability/frozen_fresh949_step120.b4is",
        ROOT / "scratch/bkg4_search/fresh_holdout_post949_c/cases7.tsv",
    ),
}


def load_panel(name: str) -> list[MacroEffectCase]:
    result_path, grid_path, state_path, cases_path = PANELS[name]
    _, rows = load_fragment_results(result_path)
    grid = load_grid(grid_path)
    candidate = next(
        i for i, row in grid.items()
        if row["donor"] == 0 and row["start"] == 120 and row["duration"] == 18
    )
    capture = load_r0(state_path)
    registry = registry_manifest()
    case_rows = [
        line.split("\t")
        for line in cases_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]
    if len(rows[0]) != capture.cases or len(case_rows) != capture.cases:
        raise ValueError(f"shape mismatch for {name}")

    out: list[MacroEffectCase] = []
    for i, (base, macro) in enumerate(zip(rows[0], rows[candidate])):
        if base[3] or macro[3]:
            raise ValueError(f"CUDA fault in {name}:{i}")
        frame = capture.frame(i, 0)
        context = {}
        for ri, (feature, args) in enumerate(registry):
            if feature in CONTRASTIVE_FEATURES and len(args) <= 1:
                context[(feature, int(args[0]) if args else 0)] = int(frame[ri])
        signature = tuple(
            int(x)
            for x in frame[:capture.registry_width + capture.bb_width + capture.compat_width]
        )
        out.append(MacroEffectCase(
            case_id=f"{name}:{i}",
            baseline_margin=int(base[0]) - int(base[1]),
            macro_margin=int(macro[0]) - int(macro[1]),
            context=context,
            exact_signature=signature,
            support_group=Path(case_rows[i][0]).name,
        ))
    return out


def _score(score):
    return {
        "activated": score.activated,
        "positive": score.positive,
        "neutral": score.neutral,
        "harmful": score.harmful,
        "loss_to_win": score.loss_to_win,
        "win_to_loss": score.win_to_loss,
        "worst_delta": score.worst_delta,
        "aggregate_delta": score.aggregate_delta,
        "support_groups": score.support_groups,
        "complexity": score.complexity,
    }


def _conds(clause):
    return [
        {"feature": c.feature, "index": c.index, "cmp": c.cmp, "value": c.value}
        for c in clause
    ]


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Reproduce the final P0.3 route410 initiation-set CEGIS campaign."
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=ROOT / "scratch/p0_3_repeatability/route410_cegis_reproduced.json",
    )
    args = ap.parse_args()

    panels = {name: load_panel(name) for name in PANELS}
    core = panels["core585"]
    discovery = core + panels["validation40"]
    all_cases = discovery + panels["sealed40"] + panels["fresh949"]

    gate0 = (ConditionGene("MARKET_INVENTORY", "LE", 0, 9934),)
    gate1 = (
        ConditionGene("MARKET_INVENTORY", "LE", 0, 9937),
        ConditionGene("OPP_MONEY", "LE", 0, 120),
    )
    gate2 = gate1 + (ConditionGene("MARKET_INVENTORY", "GE", 6, 9990),)

    payload = {
        "schema": 2,
        "macro": "route410@120+18",
        "iterations": [
            {
                "iteration": 0,
                "gate": _conds(gate0),
                "core": _score(evaluate_gate((gate0,), core)),
                "reason": "WHEAT drawdown alone is safe on the core but has only three supporting tapes.",
            },
            {
                "iteration": 1,
                "gate": _conds(gate1),
                "core": _score(evaluate_gate((gate1,), core)),
                "validation40": _score(evaluate_gate((gate1,), panels["validation40"])),
                "reason": "Broadening to the next WHEAT boundary plus low opponent liquidity adds support but exposes a harmful both-seat validation counterexample.",
            },
            {
                "iteration": 2,
                "gate": _conds(gate2),
                "discovery_plus_counterexample": _score(evaluate_gate((gate2,), discovery)),
                "sealed40": _score(evaluate_gate((gate2,), panels["sealed40"])),
                "fresh949": _score(evaluate_gate((gate2,), panels["fresh949"])),
                "all_panels": _score(evaluate_gate((gate2,), all_cases)),
                "reason": "Well-supplied MILK excludes the validation harm and matches the cow-deferral/labor-expansion mechanism.",
            },
        ],
        "final_gate": _conds(gate2),
        "final_mixed_exact_cells": [
            list(x) for x in selected_mixed_exact_cells((gate2,), all_cases)
        ],
        "wheat_threshold_sweep": {},
        "opponent_money_threshold_sweep": {},
        "milk_threshold_sweep": {},
    }

    for threshold in range(9932, 9941):
        gate = (
            ConditionGene("MARKET_INVENTORY", "LE", 0, threshold),
            ConditionGene("OPP_MONEY", "LE", 0, 120),
            ConditionGene("MARKET_INVENTORY", "GE", 6, 9990),
        )
        payload["wheat_threshold_sweep"][str(threshold)] = _score(
            evaluate_gate((gate,), discovery)
        )

    for threshold in range(80, 151, 5):
        gate = (
            ConditionGene("MARKET_INVENTORY", "LE", 0, 9937),
            ConditionGene("OPP_MONEY", "LE", 0, threshold),
            ConditionGene("MARKET_INVENTORY", "GE", 6, 9990),
        )
        payload["opponent_money_threshold_sweep"][str(threshold)] = _score(
            evaluate_gate((gate,), discovery)
        )

    for threshold in range(9978, 9997):
        gate = (
            ConditionGene("MARKET_INVENTORY", "LE", 0, 9937),
            ConditionGene("OPP_MONEY", "LE", 0, 120),
            ConditionGene("MARKET_INVENTORY", "GE", 6, threshold),
        )
        payload["milk_threshold_sweep"][str(threshold)] = _score(
            evaluate_gate((gate,), discovery)
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
