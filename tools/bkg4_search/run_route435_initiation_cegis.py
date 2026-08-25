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

PANELS = {
    "screen61": (
        ROOT / "scratch/p0_2_macro_cegis/screen61_route435.b4fr",
        ROOT / "scratch/p0_2_macro_cegis/screen61_step156.b4is",
        ROOT / "scratch/bkg4_search/current_shifted_aug20/screen61_7.tsv",
    ),
    "historical336": (
        ROOT / "scratch/p0_2_macro_cegis/historical336_route435.b4fr",
        ROOT / "scratch/p0_2_macro_cegis/historical336_step156.b4is",
        ROOT / "scratch/bkg4_search/current_shifted_aug20/historical336_7.tsv",
    ),
    "current140": (
        ROOT / "scratch/p0_2_macro_cegis/current140_route435.b4fr",
        ROOT / "scratch/p0_2_macro_cegis/current140_step156.b4is",
        ROOT / "scratch/p0_2_macro_cegis/current140_7.tsv",
    ),
    "fresh24": (
        ROOT / "scratch/p0_2_macro_cegis/fresh24_route435.b4fr",
        ROOT / "scratch/p0_2_macro_cegis/fresh24_step156.b4is",
        ROOT / "scratch/bkg4_search/current_shifted_aug20/fresh24_7.tsv",
    ),
    "sealed24": (
        ROOT / "scratch/p0_2_macro_cegis/sealed24_route435.b4fr",
        ROOT / "scratch/p0_2_macro_cegis/sealed24_step156.b4is",
        ROOT / "scratch/bkg4_search/current_shifted_aug20/sealed24_7.tsv",
    ),
}


def load_panel(name: str) -> list[MacroEffectCase]:
    results_path, state_path, cases_path = PANELS[name]
    meta, rows = load_fragment_results(results_path)
    capture = load_r0(state_path)
    registry = registry_manifest()
    case_rows = [line.split("\t") for line in cases_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if meta["candidates"] < 2:
        raise ValueError(f"route435 result lacks macro candidate for {name}")
    if len(case_rows) != capture.cases or len(rows[0]) != capture.cases:
        raise ValueError(f"shape mismatch for {name}")
    out: list[MacroEffectCase] = []
    for i, (base, macro) in enumerate(zip(rows[0], rows[1])):
        if base[3] or macro[3]:
            raise ValueError(f"CUDA fault in {name}:{i}")
        frame = capture.frame(i, 0)
        context = {}
        for ri, (feature, args) in enumerate(registry):
            if feature in CONTRASTIVE_FEATURES and len(args) <= 1:
                context[(feature, int(args[0]) if args else 0)] = int(frame[ri])
        signature = tuple(int(x) for x in frame[:capture.registry_width + capture.bb_width + capture.compat_width])
        tape, seat, seed = case_rows[i][0], int(case_rows[i][1]), int(case_rows[i][2])
        out.append(MacroEffectCase(
            case_id=f"{name}:{i}",
            baseline_margin=int(base[0]) - int(base[1]),
            macro_margin=int(macro[0]) - int(macro[1]),
            context=context,
            exact_signature=signature,
            support_group=Path(tape).name,
        ))
    return out


def conds_json(clause):
    return [{"feature": c.feature, "index": c.index, "cmp": c.cmp, "value": c.value} for c in clause]


def score_json(score):
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


def panel_scores(gate, panels):
    return {name: score_json(evaluate_gate((gate,), cases)) for name, cases in panels.items()}


def main() -> int:
    ap = argparse.ArgumentParser(description="Reproduce the bounded route435 initiation-set CEGIS campaign.")
    ap.add_argument("--out", type=Path, default=ROOT / "scratch/p0_2_macro_cegis/route435_cegis_reproduced.json")
    args = ap.parse_args()
    panels = {name: load_panel(name) for name in PANELS}
    all_cases = [case for cases in panels.values() for case in cases]

    # CEGIS iteration 0: PIZZA demand alone. Broad and unsafe.
    gate0 = (ConditionGene("SHOP_AT", "EQ", 1, 5),)
    # Counterexamples show that historical/current five-worker states and one-land
    # opponent states are unsafe, so require visible opponent labor+land commitment.
    gate1 = gate0 + (
        ConditionGene("OPP_UNITS", "GE", 0, 7),
        ConditionGene("OPP_LAND", "GE", 0, 2),
    )
    # Remaining same-labor/land Ryo counterexamples differ causally in strawberry
    # supply: harmful states are at inventory 9972, while supported states are 9993.
    # A sweep found the entire 9973..9993 interval behaviorally identical. 9990 is
    # deliberately round and leaves an 18-unit buffer from the harmful boundary.
    gate2 = gate1 + (ConditionGene("MARKET_INVENTORY", "GE", 3, 9990),)

    iterations = []
    for number, gate, reason in (
        (0, gate0, "PIZZA demand alone admits many harmful activations."),
        (1, gate1, "Visible opponent labor+land expansion removes historical/current expansion-mismatch counterexamples, but strawberry-regime harms remain."),
        (2, gate2, "Strawberry near-glut excludes the remaining harmful same-expansion counterexamples while retaining supported route435 effects."),
    ):
        mixed = selected_mixed_exact_cells((gate,), all_cases)
        iterations.append({
            "iteration": number,
            "gate": conds_json(gate),
            "reason": reason,
            "panels": panel_scores(gate, panels),
            "mixed_exact_cells": [list(x) for x in mixed],
        })

    threshold_sweep = {}
    for threshold in range(9970, 9994):
        gate = gate1 + (ConditionGene("MARKET_INVENTORY", "GE", 3, threshold),)
        threshold_sweep[str(threshold)] = panel_scores(gate, panels)

    payload = {
        "schema": 2,
        "macro": "route435@156+48",
        "iterations": iterations,
        "final_gate": conds_json(gate2),
        "final_scores": panel_scores(gate2, panels),
        "threshold_sweep": threshold_sweep,
        "final_mixed_exact_cells": [list(x) for x in selected_mixed_exact_cells((gate2,), all_cases)],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
