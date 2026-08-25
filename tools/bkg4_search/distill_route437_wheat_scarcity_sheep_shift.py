from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.bkg4_search.bounded_macro import (
    BoundedRouteMacro,
    ExactRoutePreemption,
    validate_nonoverlapping_macro_windows,
)
from tools.bkg4_search.distill_route410_wheat_liquidity_expansion import macro as route410_macro
from tools.bkg4_search.distill_route435_macro_cegis import macro as route435_macro
from tools.bkg4_search.module_codec import ConditionGene

CHAMP = ROOT / "bonsai-kg/integrations/bkg4_champion"
V6 = ROOT / "bonsai-kg/integrations/stateful_v6"
PARENT = CHAMP / "modules/animal_feed_deferral_plus_day2_plus_hire4_plus_fert_wool_plus_w33_s172_plus_route435_plus_route410_wheat_liquidity_expansion_v1.json"
DONOR = ROOT / "scratch/p0_1_macro_search/donors/route_437.json"
ROUTE_OUT = CHAMP / "modules/routes/route437_wheat_scarcity_sheep_shift_v1.json"
MACRO_OUT = CHAMP / "modules/macros/route437_wheat_scarcity_sheep_shift_v1.json"
MODULE_OUT = CHAMP / "modules/animal_feed_deferral_plus_day2_plus_hire4_plus_fert_wool_plus_w33_s172_plus_route435_plus_route410_plus_route437_wheat_scarcity_sheep_shift_v1.json"

ROUTE_ID = 218
DONOR_ROUTE_ID = 437
BASE_ROUTE_ID = 64
START_STEP = 264
DURATION = 24
FRAGMENT_SHA256 = "175853bd7517328c7a72d84f13cb149ef376221602b1bd911783dad920e0010c"
ROUTE_OVERRIDE_SLOT = "day2_capital_specialist_v1_route_override"

GATE = (
    ConditionGene("MARKET_PRICE", "GE", 0, 38),
    ConditionGene("OPP_MONEY", "GE", 0, 15750),
    ConditionGene("SHOP_AT", "NE", 2, 1),  # BRUNCH_SPOT
)


def macro() -> BoundedRouteMacro:
    out = BoundedRouteMacro(
        name="route437_wheat_scarcity_sheep_shift_v1",
        route_id=ROUTE_ID,
        base_route_id=BASE_ROUTE_ID,
        start_step=START_STEP,
        duration=DURATION,
        gate_clauses=(GATE,),
        provenance={
            "source_kind": "source_backed_route_archive",
            "source_route_id": DONOR_ROUTE_ID,
            "fragment_sha256": FRAGMENT_SHA256,
            "discovery": "P0.4_regret_guided_cuda_composition_search",
        },
    )
    validate_nonoverlapping_macro_windows((route435_macro(), route410_macro(), out))
    return out


def preemption() -> ExactRoutePreemption:
    return ExactRoutePreemption(
        preemptor_name=macro().name,
        preemptor_route_id=ROUTE_ID,
        preempted_name="fert_wool_expansion_v1",
        preempted_route_id=213,
        preempted_active_slot="fert_wool_expansion_v1_active",
        start_step=START_STEP,
        end_step=START_STEP + DURATION - 1,
        evidence_kind="exact_cuda_route_authority_conflict",
    )


def _load_v6_runtime():
    spec = importlib.util.spec_from_file_location("stateful_v6_runtime_for_route437_macro", V6 / "runtime.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import stateful-v6 runtime")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _base_actions() -> list[dict]:
    runtime = _load_v6_runtime()
    identity = json.loads((V6 / "assets/identity.json").read_text(encoding="utf-8"))
    pack = runtime.RoutePack(V6 / "assets/routes.bkv6", identity)
    return [pack.action(BASE_ROUTE_ID, step, int(pack._record(BASE_ROUTE_ID, step)[0])) for step in range(719)]


def _donor_actions() -> list[dict]:
    actions = json.loads(DONOR.read_text(encoding="utf-8")).get("actions")
    if not isinstance(actions, list) or len(actions) != 719:
        raise ValueError("route437 donor must contain exactly 719 actions")
    return actions


def build_route() -> dict:
    out = macro()
    donor = _donor_actions()
    raw = json.dumps(
        donor[START_STEP:START_STEP + DURATION],
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    if hashlib.sha256(raw).hexdigest() != FRAGMENT_SHA256:
        raise ValueError("route437 source fragment no longer matches P0.4 evidence")
    return out.route_document(_base_actions(), donor)


def build_module() -> dict:
    parent = json.loads(PARENT.read_text(encoding="utf-8"))
    out = copy.deepcopy(parent)
    bounded = macro()
    precedence = preemption()
    out["name"] = "animal_feed_deferral_plus_day2_plus_hire4_plus_fert_wool_plus_w33_s172_plus_route435_plus_route410_plus_route437_wheat_scarcity_sheep_shift_v1"
    if any(row.get("name") == bounded.active_slot for row in out.get("blackboard", [])):
        raise ValueError("route437 macro BB already exists")
    out.setdefault("blackboard", []).append(bounded.blackboard_decl())
    if out["tree"].get("type") != "Sequence":
        raise ValueError("expected P0.3 parent module sequence")
    out["tree"]["children"].append(precedence.specialist_tree(bounded, route_override_slot=ROUTE_OVERRIDE_SLOT))
    out.setdefault("strategic_routes", []).append({
        "route_id": ROUTE_ID,
        "name": bounded.name,
        "source": str(ROUTE_OUT.relative_to(ROOT)),
    })
    out["manifest"] = {
        "module": out["name"],
        "parent": "route410_wheat_liquidity_expansion_v1",
        "category": "day11/bounded_macro_cegis_with_exact_preemption",
        "macro_schema": "bounded_route_macro_v1",
        "macro_source": str(MACRO_OUT.relative_to(ROOT)),
        "macro_id": bounded.semantic_id,
        "route_id": ROUTE_ID,
        "active_window": [bounded.start_step, bounded.end_step],
        "entry_gate": "STEP == 264 and MARKET_PRICE[WHEAT] >= 38 and OPP_MONEY >= 15750 and SHOP_AT[2] != BRUNCH_SPOT",
        "legal_features": ["STEP", "MARKET_PRICE[WHEAT]", "OPP_MONEY", "SHOP_AT[2]"],
        "causal_hypothesis": (
            "At the day-11 boundary, scarce high-priced WHEAT and a highly liquid public opponent identify a capital-rich race. "
            "Without a third BRUNCH_SPOT adding another wheat-and-strawberry demand sink, route437's sheep allocation outperforms "
            "the incumbent cow allocation while preserving the labor, care, watering and planting schedule."
        ),
        "route_authority_preemption": precedence.canonical(),
        "priority": (
            "Route218 and fert_wool route213 both own steps264..287. On route218 authorization, the later specialist explicitly clears "
            "fert_wool_expansion_v1_active before latching itself; both windows end at step287, so no restoration state is required. "
            "Routes217 and216 ended at steps137 and203 and are temporally disjoint."
        ),
        "shared_market_note": (
            "The gate reads public market state, but CUDA ablation shows that reducing route437's MELON sell order to the incumbent amount "
            "changes no supported outcome, while reverting the SHEEP allocation to COW makes five of eight supported records harmful. "
            "The gain is therefore private allocation and execution, not a shared-price gift."
        ),
        "identity_rule": "No team/submission/replay/episode/seed/tape identity is present in the runtime gate or composition rule.",
        "composition": [
            "animal_feed_deferral_v1",
            "day2_capital_specialist_v1",
            "hire4_capital_cycle_v1",
            "fert_wool_expansion_v1",
            "w33_s172_expansion_v1",
            "route435_pizza_glut_rebalance_v1",
            "route410_wheat_liquidity_expansion_v1",
            bounded.name,
        ],
    }
    return out


def main() -> int:
    bounded = macro()
    route = build_route()
    module = build_module()
    ROUTE_OUT.parent.mkdir(parents=True, exist_ok=True)
    MACRO_OUT.parent.mkdir(parents=True, exist_ok=True)
    ROUTE_OUT.write_text(json.dumps(route, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    MACRO_OUT.write_text(json.dumps(bounded.canonical(), ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    MODULE_OUT.write_text(json.dumps(module, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "macro": str(MACRO_OUT.relative_to(ROOT)),
        "module": str(MODULE_OUT.relative_to(ROOT)),
        "route": str(ROUTE_OUT.relative_to(ROOT)),
        "macro_id": bounded.semantic_id,
        "route_id": ROUTE_ID,
        "window": [bounded.start_step, bounded.end_step],
        "preemption": preemption().canonical(),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
