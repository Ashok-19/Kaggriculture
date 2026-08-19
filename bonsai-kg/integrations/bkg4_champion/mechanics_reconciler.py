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
