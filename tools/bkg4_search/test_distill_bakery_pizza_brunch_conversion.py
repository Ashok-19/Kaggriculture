from __future__ import annotations

import importlib.util
from pathlib import Path


def _mod():
    path = Path(__file__).with_name("distill_bakery_pizza_brunch_conversion.py")
    spec = importlib.util.spec_from_file_location("distill_bakery_pizza_brunch_conversion", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_bakery_pizza_brunch_specialist_is_public_state_routed_and_windowed() -> None:
    m = _mod()

    assert m.ROUTE_ID == 215
    assert (m.START_STEP, m.END_STEP) == (248, 311)

    route = m.build_route()
    assert route["route_id"] == 215
    assert route["base_route"] == 22
    assert route["active_window"] == [248, 311]
    assert len(route["actions"]) == 719

    donor = m._load_donor_route_actions()
    base = m._load_base_route_actions()
    assert route["actions"][247] == base[247]
    assert route["actions"][248:312] == donor[248:312]
    assert route["actions"][312] == base[312]

    tree = m.specialist_tree()
    text = repr(tree)
    assert "SHOP_AT" in text
    assert "BAKERY" not in text  # numeric public shop ids, no named-opponent logic
    assert "opponent" not in text.lower()
    assert "episode" not in text.lower()
    assert "seed" not in text.lower()

    gate = tree["children"][0]["then"]["child"]["children"]
    assert gate[:3] == [
        {"type": "Condition", "lhs": {"feature": "SHOP_AT", "args": [0]}, "op": "EQ", "rhs": 0},
        {"type": "Condition", "lhs": {"feature": "SHOP_AT", "args": [1]}, "op": "EQ", "rhs": 5},
        {"type": "Condition", "lhs": {"feature": "SHOP_AT", "args": [2]}, "op": "EQ", "rhs": 1},
    ]

    module = m.build_module()
    assert module["blackboard"][-1]["name"] == "bakery_pizza_brunch_conversion_v1_active"
    assert module["strategic_routes"][-1]["route_id"] == 215
    assert module["manifest"]["entry_gate"] == (
        "STEP == 248 and SHOP_AT[0] == BAKERY and SHOP_AT[1] == PIZZA_SHOP "
        "and SHOP_AT[2] == BRUNCH_SPOT"
    )
    assert module["manifest"]["legal_features"] == [
        "STEP",
        "SHOP_AT[0]",
        "SHOP_AT[1]",
        "SHOP_AT[2]",
    ]
