from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CHAMP = ROOT / "bonsai-kg/integrations/bkg4_champion"
V6 = ROOT / "bonsai-kg/integrations/stateful_v6"
PARENT = CHAMP / "modules/animal_feed_deferral_plus_day2_capital_specialist_v1.json"
DONOR = ROOT / "scratch/bkg4_search/current_shifted_aug20/tapes/Izzoudine_Mohamed_KANTA_s1_ep94674701.json"
ROUTE_OUT = CHAMP / "modules/routes/hire4_capital_cycle_v1.json"
MODULE_OUT = CHAMP / "modules/animal_feed_deferral_plus_day2_plus_hire4_capital_cycle_v1.json"
ROUTE_ID = 212
START_STEP = 26
END_STEP = 47


def _load_v6_runtime():
    spec = importlib.util.spec_from_file_location("stateful_v6_runtime_for_hire4", V6 / "runtime.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import stateful-v6 runtime")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def build_route() -> dict:
    m = _load_v6_runtime()
    identity = json.loads((V6 / "assets/identity.json").read_text(encoding="utf-8"))
    pack = m.RoutePack(V6 / "assets/routes.bkv6", identity)
    donor = json.loads(DONOR.read_text(encoding="utf-8"))["actions"]
    actions = [pack.action(22, step, int(pack._record(22, step)[0])) for step in range(719)]
    for step in range(START_STEP, END_STEP + 1):
        actions[step] = copy.deepcopy(donor[step])
    return {
        "schema": 1,
        "name": "hire4_capital_cycle_v1",
        "route_id": ROUTE_ID,
        "base_route": 22,
        "entry_step": START_STEP,
        "active_window": [START_STEP, END_STEP],
        "source_archetype": "six independent fresh tapes exposed the same day-2 animal/fertilizer/capital cycle; donor identity is offline provenance only and is never used by runtime",
        "actions": actions,
    }


def specialist_tree() -> dict:
    return {
        "type": "Sequence",
        "category": "opening/hire4_capital_cycle_v1",
        "children": [
            {
                "type": "If",
                "condition": {"type": "Condition", "lhs": {"feature": "STEP"}, "op": "EQ", "rhs": START_STEP},
                "then": {
                    "type": "ForceSuccess",
                    "child": {
                        "type": "Sequence",
                        "children": [
                            {"type": "Condition", "lhs": {"feature": "OPP_HIRES_TODAY"}, "op": "EQ", "rhs": 4},
                            {"type": "Condition", "lhs": {"feature": "OPP_MONEY"}, "op": "GE", "rhs": 30},
                            {"type": "SetBlackboard", "slot": "hire4_capital_cycle_v1_active", "value": 1},
                        ],
                    },
                },
                "else": {"type": "Success"},
            },
            {
                "type": "ForceSuccess",
                "child": {
                    "type": "Sequence",
                    "children": [
                        {"type": "Condition", "lhs": {"blackboard": "hire4_capital_cycle_v1_active"}, "op": "EQ", "rhs": 1},
                        {"type": "Condition", "lhs": {"feature": "STEP"}, "op": "GE", "rhs": START_STEP},
                        {"type": "Condition", "lhs": {"feature": "STEP"}, "op": "LE", "rhs": END_STEP},
                        {"type": "SetBlackboard", "slot": "day2_capital_specialist_v1_route_override", "value": ROUTE_ID},
                    ],
                },
            },
        ],
    }


def build_module() -> dict:
    parent = json.loads(PARENT.read_text(encoding="utf-8"))
    module = copy.deepcopy(parent)
    module["name"] = "animal_feed_deferral_plus_day2_plus_hire4_capital_cycle_v1"
    if any(x.get("name") == "hire4_capital_cycle_v1_active" for x in module["blackboard"]):
        raise ValueError("hire4 BB already present")
    module["blackboard"].append({
        "initial": 0,
        "name": "hire4_capital_cycle_v1_active",
        "owner": "hire4_capital_cycle_v1",
        "type": "BOOL",
    })
    if module["tree"].get("type") != "Sequence":
        raise ValueError("expected parent module sequence")
    module["tree"]["children"].append(specialist_tree())
    strategic = list(module.get("strategic_routes") or [])
    strategic.append({
        "route_id": ROUTE_ID,
        "name": "hire4_capital_cycle_v1",
        "source": "bonsai-kg/integrations/bkg4_champion/modules/routes/hire4_capital_cycle_v1.json",
    })
    module["strategic_routes"] = strategic
    module["manifest"] = {
        "module": module["name"],
        "bkg4_vm_frozen_commit": "8381e027646ba6494cbe4dd06a810ea55ec83575",
        "parent": "day2_capital_specialist_v1",
        "category": "opening/hire4_capital_cycle",
        "causal_hypothesis": "When the opponent spends into four day-2 hires but still retains meaningful cash, the incumbent animal/fertilizer traversal is too passive. A coherent care/collect/liquidate/rebuy cycle converts private fertilizer and animal tempo into additional terminal value.",
        "entry_gate": "STEP == 26 and OPP_HIRES_TODAY == 4 and OPP_MONEY >= 30",
        "legal_features": ["STEP", "OPP_HIRES_TODAY", "OPP_MONEY"],
        "route_id": ROUTE_ID,
        "active_window": [START_STEP, END_STEP],
        "shared_market_note": "The coupled SELL FERTILIZER then BUY PRODUCT WHEAT sequence is required by CUDA ablation; it finances private animal production. It is not promoted on price movement alone.",
        "search_campaign": "fragment_specialists_v1_20260820",
        "composition": ["animal_feed_deferral_v1", "day2_capital_specialist_v1", "hire4_capital_cycle_v1"],
    }
    return module


def main() -> int:
    ROUTE_OUT.parent.mkdir(parents=True, exist_ok=True)
    ROUTE_OUT.write_text(json.dumps(build_route(), ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    MODULE_OUT.write_text(json.dumps(build_module(), ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"module": str(MODULE_OUT.relative_to(ROOT)), "route": str(ROUTE_OUT.relative_to(ROOT)), "route_id": ROUTE_ID, "window": [START_STEP, END_STEP]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
