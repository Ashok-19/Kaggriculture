from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CHAMP = ROOT / "bonsai-kg/integrations/bkg4_champion"
V6 = ROOT / "bonsai-kg/integrations/stateful_v6"
PARENT = CHAMP / "modules/animal_feed_deferral_plus_day2_plus_hire4_plus_fert_wool_plus_w33_s172_v1.json"
DONOR_PACK = ROOT / "scratch/public_targets/top10_parent_routes/routes_top10_parents.bkv6"
ROUTE_OUT = CHAMP / "modules/routes/bakery_pizza_brunch_conversion_v1.json"
MODULE_OUT = CHAMP / "modules/animal_feed_deferral_plus_day2_plus_hire4_plus_fert_wool_plus_w33_s172_plus_bakery_pizza_brunch_v1.json"
ROUTE_ID = 215
DONOR_ROUTE_ID = 420
BASE_ROUTE_ID = 22
START_STEP = 248
END_STEP = 311
BAKERY_SHOP_ID = 0
PIZZA_SHOP_ID = 5
BRUNCH_SHOP_ID = 1


def _load_v6_runtime():
    spec = importlib.util.spec_from_file_location("stateful_v6_runtime_for_bakery_conversion", V6 / "runtime.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import stateful-v6 runtime")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _route_pack_identity(pack_path: Path) -> dict:
    m = _load_v6_runtime()
    raw = pack_path.read_bytes()
    if len(raw) < m.HEADER.size:
        raise ValueError(f"route pack too short: {pack_path}")
    magic, version, route_count, _action_count, _payload_hash = m.HEADER.unpack_from(raw, 0)
    if magic != m.PACK_MAGIC or version != m.PACK_VERSION:
        raise ValueError(f"invalid route pack: {pack_path}")
    route_ids = [int(m.DESC.unpack_from(raw, m.HEADER.size + i * m.DESC.size)[0]) for i in range(route_count)]
    return {"route_pack": {"pack_sha256": hashlib.sha256(raw).hexdigest(), "route_ids": route_ids}}


def _load_route_actions(pack_path: Path, route_id: int, identity: dict | None = None) -> list[dict]:
    m = _load_v6_runtime()
    pack = m.RoutePack(pack_path, identity or _route_pack_identity(pack_path))
    return [pack.action(route_id, step, int(pack._record(route_id, step)[0])) for step in range(719)]


def _load_base_route_actions() -> list[dict]:
    identity = json.loads((V6 / "assets/identity.json").read_text(encoding="utf-8"))
    return _load_route_actions(V6 / "assets/routes.bkv6", BASE_ROUTE_ID, identity)


def _load_donor_route_actions() -> list[dict]:
    return _load_route_actions(DONOR_PACK, DONOR_ROUTE_ID)


def build_route() -> dict:
    actions = _load_base_route_actions()
    donor = _load_donor_route_actions()
    for step in range(START_STEP, END_STEP + 1):
        actions[step] = copy.deepcopy(donor[step])
    return {
        "schema": 1,
        "name": "bakery_pizza_brunch_conversion_v1",
        "route_id": ROUTE_ID,
        "base_route": BASE_ROUTE_ID,
        "entry_step": START_STEP,
        "active_window": [START_STEP, END_STEP],
        "source_archetype": (
            "route420 high-performing action fragment discovered offline from the current-top10 parent bank; "
            "runtime routing uses only public shop-order state"
        ),
        "actions": actions,
    }


def specialist_tree() -> dict:
    return {
        "type": "Sequence",
        "category": "midgame/bakery_pizza_brunch_conversion_v1",
        "children": [
            {
                "type": "If",
                "condition": {"type": "Condition", "lhs": {"feature": "STEP"}, "op": "EQ", "rhs": START_STEP},
                "then": {
                    "type": "ForceSuccess",
                    "child": {
                        "type": "Sequence",
                        "children": [
                            {"type": "Condition", "lhs": {"feature": "SHOP_AT", "args": [0]}, "op": "EQ", "rhs": BAKERY_SHOP_ID},
                            {"type": "Condition", "lhs": {"feature": "SHOP_AT", "args": [1]}, "op": "EQ", "rhs": PIZZA_SHOP_ID},
                            {"type": "Condition", "lhs": {"feature": "SHOP_AT", "args": [2]}, "op": "EQ", "rhs": BRUNCH_SHOP_ID},
                            {"type": "SetBlackboard", "slot": "bakery_pizza_brunch_conversion_v1_active", "value": 1},
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
                        {"type": "Condition", "lhs": {"blackboard": "bakery_pizza_brunch_conversion_v1_active"}, "op": "EQ", "rhs": 1},
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
    module["name"] = "animal_feed_deferral_plus_day2_plus_hire4_plus_fert_wool_plus_w33_s172_plus_bakery_pizza_brunch_v1"
    if any(x.get("name") == "bakery_pizza_brunch_conversion_v1_active" for x in module["blackboard"]):
        raise ValueError("bakery/pizza/brunch BB already present")
    module["blackboard"].append({
        "initial": 0,
        "name": "bakery_pizza_brunch_conversion_v1_active",
        "owner": "bakery_pizza_brunch_conversion_v1",
        "type": "BOOL",
    })
    if module["tree"].get("type") != "Sequence":
        raise ValueError("expected parent module sequence")
    module["tree"]["children"].append(specialist_tree())
    strategic = list(module.get("strategic_routes") or [])
    strategic.append({
        "route_id": ROUTE_ID,
        "name": "bakery_pizza_brunch_conversion_v1",
        "source": "bonsai-kg/integrations/bkg4_champion/modules/routes/bakery_pizza_brunch_conversion_v1.json",
    })
    module["strategic_routes"] = strategic
    module["manifest"] = {
        "module": module["name"],
        "bkg4_vm_frozen_commit": "8381e027646ba6494cbe4dd06a810ea55ec83575",
        "parent": "w33_s172_expansion_v1",
        "category": "midgame/bakery_pizza_brunch_conversion",
        "causal_hypothesis": (
            "The public first-three-shop sequence BAKERY -> PIZZA_SHOP -> BRUNCH_SPOT identifies a production regime "
            "where the incumbent reaches the midgame with capital but converts it poorly. A coherent 64-turn donor "
            "fragment at steps 248-311 improves conversion and then releases back to the incumbent Bonsai."
        ),
        "entry_gate": "STEP == 248 and SHOP_AT[0] == BAKERY and SHOP_AT[1] == PIZZA_SHOP and SHOP_AT[2] == BRUNCH_SPOT",
        "legal_features": ["STEP", "SHOP_AT[0]", "SHOP_AT[1]", "SHOP_AT[2]"],
        "route_id": ROUTE_ID,
        "active_window": [START_STEP, END_STEP],
        "shared_state_note": (
            "Shop order is public environment state and is used only as a lane selector. No opponent identity, replay id, "
            "episode id, seed, source-program identity, or action hash is available to runtime routing."
        ),
        "search_campaign": "public_conversion_specialists_20260824",
        "discovery_evidence": {
            "public_frozen_original10": "2 loss-to-win conversions; target margins +754/+1374",
            "current_top10_discovery76": "0 activations; 0 outcome regressions",
            "historical336": "2 activations; 0 outcome regressions; score deltas +410/-64",
            "screen61": "0 activations; 0 outcome regressions",
        },
        "composition": [
            "animal_feed_deferral_v1",
            "day2_capital_specialist_v1",
            "hire4_capital_cycle_v1",
            "fert_wool_expansion_v1",
            "w33_s172_expansion_v1",
            "bakery_pizza_brunch_conversion_v1",
        ],
    }
    return module


def main() -> int:
    ROUTE_OUT.parent.mkdir(parents=True, exist_ok=True)
    ROUTE_OUT.write_text(json.dumps(build_route(), ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    MODULE_OUT.write_text(json.dumps(build_module(), ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "module": str(MODULE_OUT.relative_to(ROOT)),
        "route": str(ROUTE_OUT.relative_to(ROOT)),
        "route_id": ROUTE_ID,
        "window": [START_STEP, END_STEP],
        "donor_route": DONOR_ROUTE_ID,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
