from __future__ import annotations

import copy
from typing import Any, Dict, Iterable, Sequence


def apply_feed_deferral(action: Dict[str, Any], blackboard: Sequence[int], defer_slots: Iterable[int]) -> Dict[str, Any]:
    """Materialize BKG4 feed authorization against an already selected route action.

    The BKG4 tree owns the causal decision.  This reconciler only has one authority:
    when an authorized unit's selected route command is FEED, replace that command with
    PASS.  It never creates a new strategic condition and never changes non-FEED work.
    """
    slots = tuple(int(x) for x in defer_slots)
    if not slots:
        return action
    farmer = list((action or {}).get("farmer") or ["PASS"])
    hands = [list(x) if isinstance(x, list) else ["PASS"] for x in list((action or {}).get("hands") or [])]
    changed = False
    for unit, slot in enumerate(slots):
        if slot < 0 or slot >= len(blackboard) or not int(blackboard[slot]):
            continue
        if unit == 0:
            if farmer and farmer[0] == "FEED":
                farmer = ["PASS"]; changed = True
        else:
            hi = unit - 1
            if hi < len(hands) and hands[hi] and hands[hi][0] == "FEED":
                hands[hi] = ["PASS"]; changed = True
    if not changed:
        return action
    out = copy.deepcopy(action)
    out["farmer"] = farmer
    out["hands"] = hands
    return out


_ITEM_NAMES = (
    "WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG",
    "MILK", "WOOL", "FERTILIZER", "GOOSE", "COW", "SHEEP",
)
_ITEM_QUANTITY_UNIT_OPS = {"PICKUP", "PLACE"}
_ITEM_ONLY_UNIT_OPS = {"PLANT"}


def _rewrite_unit_action(op: str, arg: int, quantity: int) -> list[Any]:
    name = str(op)
    if name in _ITEM_QUANTITY_UNIT_OPS or name in _ITEM_ONLY_UNIT_OPS:
        item = int(arg)
        if not 0 <= item < len(_ITEM_NAMES):
            raise ValueError(f"invariant unit rewrite item arg out of range: {item}")
        if name in _ITEM_QUANTITY_UNIT_OPS:
            return [name, _ITEM_NAMES[item], int(quantity)]
        return [name, _ITEM_NAMES[item]]
    return [name]


def apply_invariant_unit_rewrites(
    action: Dict[str, Any],
    blackboard: Sequence[int],
    rewrites: Iterable[Dict[str, Any]],
) -> Dict[str, Any]:
    """Apply BKG4-authorized generic per-unit action rewrites after route materialization.

    Strategic conditions stay in the compiled BKG4 tree.  Each metadata row names only
    a resolved authorization slot, a static unit index, and the exact match/replacement
    operation.  This function cannot activate a rewrite on its own.
    """
    rows = tuple(dict(x) for x in rewrites)
    if not rows:
        return action
    farmer = list((action or {}).get("farmer") or ["PASS"])
    hands = [list(x) if isinstance(x, list) else ["PASS"] for x in list((action or {}).get("hands") or [])]
    changed = False
    for row in rows:
        slot = int(row["slot"])
        unit = int(row["unit"])
        if slot < 0 or slot >= len(blackboard) or not int(blackboard[slot]):
            continue
        if unit < 0:
            raise ValueError("invariant unit rewrite unit index must be nonnegative")
        current = farmer if unit == 0 else (hands[unit - 1] if unit - 1 < len(hands) else None)
        if not current or str(current[0]) != str(row["match_op"]):
            continue
        replacement = _rewrite_unit_action(str(row["replacement_op"]), int(row.get("arg", 0)), int(row.get("quantity", 1)))
        if unit == 0:
            farmer = replacement
        else:
            hands[unit - 1] = replacement
        changed = True
    if not changed:
        return action
    out = copy.deepcopy(action)
    out["farmer"] = farmer
    out["hands"] = hands
    return out
