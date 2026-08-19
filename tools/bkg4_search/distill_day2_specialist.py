from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CHAMP = ROOT / "bonsai-kg/integrations/bkg4_champion"
V6 = ROOT / "bonsai-kg/integrations/stateful_v6"
FEED = CHAMP / "modules/animal_feed_deferral_v1.json"
DONOR = ROOT / "scratch/bkg_v6_engine_exploit_mining_20260819/top10_bank_20260819/tapes/Kobe_BRYANT_s0_ep94397434.json"
ROUTE_OUT = CHAMP / "modules/routes/day2_capital_specialist_v1.json"
MODULE_OUT = CHAMP / "modules/animal_feed_deferral_plus_day2_capital_specialist_v1.json"
ROUTE_ID = 211
SPECIAL_STEPS = (26, 27, 28, 29, 30, 31, 32, 33, 36)


def _load_v6_runtime():
    spec = importlib.util.spec_from_file_location("stateful_v6_runtime_for_distill", V6 / "runtime.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import stateful-v6 runtime")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def build_route() -> dict:
    m = _load_v6_runtime()
    identity = json.loads((V6 / "assets/identity.json").read_text(encoding="utf-8"))
    pack = m.RoutePack(V6 / "assets/routes.bkv6", identity)
    donor_obj = json.loads(DONOR.read_text(encoding="utf-8"))
    donor = donor_obj["actions"]
    actions = [pack.action(22, step, int(pack._record(22, step)[0])) for step in range(719)]
    for step in SPECIAL_STEPS:
        actions[step] = copy.deepcopy(donor[step])
    return {
        "schema": 1,
        "name": "day2_capital_specialist_v1",
        "route_id": ROUTE_ID,
        "base_route": 22,
        "entry_step": 26,
        "active_steps": list(SPECIAL_STEPS),
        "source_archetype": "stable replay-identical day-2 action vocabulary; exhaustive 2^16 CUDA mask search minimized it to mask 0x01ff; identity is not used by runtime",
        "actions": actions,
    }


def _shop_eq(index: int, shop_id: int) -> dict:
    return {"type": "Condition", "lhs": {"feature": "SHOP_AT", "arg": index}, "op": "EQ", "rhs": shop_id}


def recovery_tree() -> dict:
    bakery_smoothie = {
        "type": "Sequence",
        "category": "recovery/bakery_smoothie_v1",
        "children": [
            {"type": "Condition", "lhs": {"blackboard": "day2_capital_specialist_v1_active"}, "op": "EQ", "rhs": 1},
            {"type": "Condition", "lhs": {"feature": "STEP"}, "op": "GE", "rhs": 264},
            {"type": "Condition", "lhs": {"feature": "STEP"}, "op": "LE", "rhs": 287},
            _shop_eq(0, 0),
            _shop_eq(1, 6),
            {"type": "SetBlackboard", "slot": "day2_capital_specialist_v1_route_override", "value": 22},
        ],
    }
    brunch_weed = {
        "type": "Sequence",
        "category": "recovery/brunch_weed14_v1",
        "children": [
            {"type": "Condition", "lhs": {"blackboard": "day2_capital_specialist_v1_active"}, "op": "EQ", "rhs": 1},
            {"type": "Condition", "lhs": {"feature": "STEP"}, "op": "GE", "rhs": 264},
            {"type": "Condition", "lhs": {"feature": "STEP"}, "op": "LE", "rhs": 287},
            _shop_eq(0, 1),
            _shop_eq(1, 1),
            {"type": "Condition", "lhs": {"feature": "OWN_TILE_KIND", "arg": 41}, "op": "EQ", "rhs": 2},
            {"type": "SetBlackboard", "slot": "day2_capital_specialist_v1_route_override", "value": 46},
        ],
    }
    return {"type": "ForceSuccess", "child": {"type": "Selector", "children": [bakery_smoothie, brunch_weed]}}


def specialist_tree() -> dict:
    active_step = {
        "type": "Selector",
        "children": [
            {"type": "Condition", "lhs": {"feature": "STEP"}, "op": "EQ", "rhs": step}
            for step in SPECIAL_STEPS
        ],
    }
    return {
        "type": "Sequence",
        "category": "opening/day2_capital_specialist",
        "children": [
            {"type": "SetBlackboard", "slot": "day2_capital_specialist_v1_route_override", "value": 0},
            {
                "type": "If",
                "condition": {"type": "Condition", "lhs": {"feature": "STEP"}, "op": "EQ", "rhs": 26},
                "then": {
                    "type": "If",
                    "condition": {"type": "Condition", "lhs": {"feature": "OPP_MONEY"}, "op": "LE", "rhs": 139},
                    "then": {
                        "type": "If",
                        "condition": {"type": "Condition", "lhs": {"feature": "OPP_UNITS"}, "op": "EQ", "rhs": 2},
                        "then": {"type": "SetBlackboard", "slot": "day2_capital_specialist_v1_active", "value": 1},
                        "else": {"type": "Success"}
                    },
                    "else": {"type": "Success"},
                },
                "else": {"type": "Success"},
            },
            {
                "type": "ForceSuccess",
                "child": {
                    "type": "Sequence",
                    "children": [
                        {"type": "Condition", "lhs": {"blackboard": "day2_capital_specialist_v1_active"}, "op": "EQ", "rhs": 1},
                        active_step,
                        {"type": "SetBlackboard", "slot": "day2_capital_specialist_v1_route_override", "value": ROUTE_ID},
                    ],
                },
            },
            recovery_tree(),
        ],
    }


def build_module() -> dict:
    feed = json.loads(FEED.read_text(encoding="utf-8"))
    module = copy.deepcopy(feed)
    module["name"] = "animal_feed_deferral_plus_day2_capital_specialist_v1"
    module["blackboard"].append({
        "initial": 0,
        "name": "day2_capital_specialist_v1_active",
        "owner": "day2_capital_specialist_v1",
        "type": "BOOL",
    })
    module["blackboard"].append({
        "initial": 0,
        "name": "day2_capital_specialist_v1_route_override",
        "owner": "day2_capital_specialist_v1",
        "type": "I32",
    })
    module["tree"] = {
        "type": "Sequence",
        "category": "bkg4_verified_strategy_modules",
        "children": [feed["tree"], specialist_tree()],
    }
    module["strategic_routes"] = [{
        "route_id": ROUTE_ID,
        "name": "day2_capital_specialist_v1",
        "source": "bonsai-kg/integrations/bkg4_champion/modules/routes/day2_capital_specialist_v1.json",
    }]
    module["runtime_reconciler"] = {
        "kind": "feed_deferral_route_value_v1",
        "defer_slots": list(feed["runtime_reconciler"]["defer_slots"]),
        "max_units": int(feed["runtime_reconciler"]["max_units"]),
        "route_override_slot": "day2_capital_specialist_v1_route_override",
        "materialization_rule": "apply feed deferral; if the BKG4 per-tick route slot is nonzero, materialize that route ID for this tick only; never mutate the frozen controller route slot",
    }
    module["manifest"] = {
        "module": "animal_feed_deferral_plus_day2_capital_specialist_v1",
        "bkg4_vm_frozen_commit": "8381e027646ba6494cbe4dd06a810ea55ec83575",
        "category": "opening/day2_capital_specialist",
        "causal_hypothesis": "When the opponent visibly commits to an early day-2 hire while still in a low-cash opening regime, the incumbent day-2 animal/fertilizer traversal leaves exploitable tempo. The minimized alternate traversal at steps 26-33 and 36 improves private production/capital without opponent identity or shared-price inflation.",
        "legal_features": ["STEP", "OPP_MONEY", "OPP_UNITS", "SHOP_AT", "OWN_TILE_KIND"],
        "route_id": ROUTE_ID,
        "entry_step": 26,
        "active_window": [26, 36],
        "gate": "STEP == 26 and OPP_MONEY <= 139 and OPP_UNITS == 2",
        "search_campaign": "bkg4_hard_loss_specialist_day2_v1_20260820",
        "search_mask": "0x01ff",
        "search_mask_steps": [26, 27, 28, 29, 30, 31, 32, 33, 36],
        "recoveries": [
            {"gate": "specialist_active and 264 <= STEP <= 287 and SHOP_AT(0)==BAKERY and SHOP_AT(1)==SMOOTHIE_SHOP", "route": 22, "cuda_gain_on_current_ryo_pair_vs_parent": [11746, 11434]},
            {"gate": "specialist_active and 264 <= STEP <= 287 and SHOP_AT(0)==BRUNCH_SPOT and SHOP_AT(1)==BRUNCH_SPOT and OWN_TILE_KIND(41)==WEED", "route": 46, "cuda_gain_vs_parent": 2941}
        ],
        "composition": ["animal_feed_deferral_v1", "day2_capital_specialist_v1"],
    }
    return module


def main() -> int:
    ROUTE_OUT.parent.mkdir(parents=True, exist_ok=True)
    ROUTE_OUT.write_text(json.dumps(build_route(), ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    MODULE_OUT.write_text(json.dumps(build_module(), ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"route": str(ROUTE_OUT.relative_to(ROOT)), "module": str(MODULE_OUT.relative_to(ROOT)), "route_id": ROUTE_ID, "steps": list(SPECIAL_STEPS)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
