from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CHAMP = ROOT / "bonsai-kg/integrations/bkg4_champion"
V6 = ROOT / "bonsai-kg/integrations/stateful_v6"
PARENT = CHAMP / "modules/animal_feed_deferral_plus_day2_plus_hire4_capital_cycle_v1.json"
DONOR = ROOT / "scratch/bkg4_search/current_shifted_aug20/tapes/Xiaowenhao404_s1_ep94673224.json"
ROUTE_OUT = CHAMP / "modules/routes/fert_wool_expansion_v1.json"
MODULE_OUT = CHAMP / "modules/animal_feed_deferral_plus_day2_plus_hire4_plus_fert_wool_expansion_v1.json"
ROUTE_ID = 213
START_STEP = 264
END_STEP = 287
FERTILIZER_PRODUCT_ID = 8
WOOL_PRODUCT_ID = 7
FERTILIZER_MIN = 10105
WOOL_MAX = 9940


def _load_v6_runtime():
    spec = importlib.util.spec_from_file_location("stateful_v6_runtime_for_fert_wool", V6 / "runtime.py")
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
        "name": "fert_wool_expansion_v1",
        "route_id": ROUTE_ID,
        "base_route": 22,
        "entry_step": START_STEP,
        "active_window": [START_STEP, END_STEP],
        "source_archetype": "fresh Xiaowenhao action fragment used only as offline provenance; runtime routing uses legal market state only",
        "actions": actions,
    }


def specialist_tree() -> dict:
    return {
        "type": "Sequence",
        "category": "midgame/fert_wool_expansion_v1",
        "children": [
            {
                "type": "If",
                "condition": {"type": "Condition", "lhs": {"feature": "STEP"}, "op": "EQ", "rhs": START_STEP},
                "then": {
                    "type": "ForceSuccess",
                    "child": {
                        "type": "Sequence",
                        "children": [
                            {"type": "Condition", "lhs": {"feature": "MARKET_INVENTORY", "args": [FERTILIZER_PRODUCT_ID]}, "op": "GE", "rhs": FERTILIZER_MIN},
                            {"type": "Condition", "lhs": {"feature": "MARKET_INVENTORY", "args": [WOOL_PRODUCT_ID]}, "op": "LE", "rhs": WOOL_MAX},
                            {"type": "SetBlackboard", "slot": "fert_wool_expansion_v1_active", "value": 1},
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
                        {"type": "Condition", "lhs": {"blackboard": "fert_wool_expansion_v1_active"}, "op": "EQ", "rhs": 1},
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
    module["name"] = "animal_feed_deferral_plus_day2_plus_hire4_plus_fert_wool_expansion_v1"
    if any(x.get("name") == "fert_wool_expansion_v1_active" for x in module["blackboard"]):
        raise ValueError("fert/wool BB already present")
    module["blackboard"].append({
        "initial": 0,
        "name": "fert_wool_expansion_v1_active",
        "owner": "fert_wool_expansion_v1",
        "type": "BOOL",
    })
    if module["tree"].get("type") != "Sequence":
        raise ValueError("expected parent module sequence")
    module["tree"]["children"].append(specialist_tree())
    strategic = list(module.get("strategic_routes") or [])
    strategic.append({
        "route_id": ROUTE_ID,
        "name": "fert_wool_expansion_v1",
        "source": "bonsai-kg/integrations/bkg4_champion/modules/routes/fert_wool_expansion_v1.json",
    })
    module["strategic_routes"] = strategic
    module["manifest"] = {
        "module": module["name"],
        "bkg4_vm_frozen_commit": "8381e027646ba6494cbe4dd06a810ea55ec83575",
        "parent": "hire4_capital_cycle_v1",
        "category": "midgame/fert_wool_expansion",
        "causal_hypothesis": "A fertilizer-heavy and wool-scarce shared market at step 264 identifies states where the incumbent is under-expanded. A 24-turn private expansion lane converts owned capital into land, labor, sheep, care, planting and production.",
        "entry_gate": "STEP == 264 and MARKET_INVENTORY[FERTILIZER] >= 10105 and MARKET_INVENTORY[WOOL] <= 9940",
        "legal_features": ["STEP", "MARKET_INVENTORY[FERTILIZER]", "MARKET_INVENTORY[WOOL]"],
        "route_id": ROUTE_ID,
        "active_window": [START_STEP, END_STEP],
        "shared_market_note": "The market state is a public lane selector, not the payoff. CUDA ablation shows removing private HIRE/BUY_LAND expansion makes every supported activation worse; removing SELL financing also destroys most gains. The specialist wins through private expansion despite shared-market externalities.",
        "search_campaign": "fragment_specialists_v1_20260820",
        "composition": ["animal_feed_deferral_v1", "day2_capital_specialist_v1", "hire4_capital_cycle_v1", "fert_wool_expansion_v1"],
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
