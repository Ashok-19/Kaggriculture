"""Compile-time Bonsai-KG v4 strategic patterns.

These helpers return ordinary BKG4 authoring dictionaries.  They add no runtime opcode,
callback, dynamic dispatch or hidden state.  Callers remain responsible for declaring
referenced blackboard slots with appropriate semantic owners/types.
"""
from __future__ import annotations

from typing import Any, Dict

Node = Dict[str, Any]


def ensure(goal: Node, achiever: Node) -> Node:
    """Reactive goal test: succeed immediately when the goal holds, otherwise achieve it."""
    return {"type": "ReactiveSelector", "children": [goal, achiever]}


def maintain(condition: Node, behavior: Node) -> Node:
    """Continuously guard behavior; preemption halts behavior if the condition fails."""
    return {"type": "ReactiveSequence", "children": [condition, behavior]}


def recovery_fallback(normal: Node, recovery: Node) -> Node:
    """Run normal behavior first and enter an explicit recovery path only on failure."""
    return {"type": "Selector", "children": [normal, recovery]}


def deadline_guard(turns: int, behavior: Node) -> Node:
    """Bound behavior by game steps using the finite TurnTimeout decorator."""
    return {"type": "TurnTimeout", "turns": int(turns), "child": behavior}


def transactional_intent_group(*children: Node) -> Node:
    """Sequence transaction: descendant intents roll back if the group fails/skips."""
    return {"type": "Sequence", "children": list(children)}


def run_once(latch_slot: str, behavior: Node) -> Node:
    """OneShot using a semantic BOOL blackboard latch; no RunOnce VM opcode required."""
    return {
        "type": "Selector",
        "children": [
            {"type": "Condition", "lhs": {"blackboard": latch_slot}, "op": "EQ", "rhs": 1},
            {"type": "Sequence", "children": [
                behavior,
                {"type": "SetBlackboard", "slot": latch_slot, "value": 1},
            ]},
        ],
    }


def hysteresis_band(mode_slot: str, low: Node, high: Node, low_value: str | int, high_value: str | int) -> Node:
    """Two-threshold mode update: enter high only at high guard, leave only at low guard."""
    return {
        "type": "ReactiveSelector",
        "children": [
            {"type": "Sequence", "children": [high, {"type": "SetBlackboard", "slot": mode_slot, "value": high_value}]},
            {"type": "Sequence", "children": [low, {"type": "SetBlackboard", "slot": mode_slot, "value": low_value}]},
            {"type": "Success"},
        ],
    }


def guarded_commitment(
    mode_slot: str,
    committed_value: str | int,
    entry_guard: Node,
    invalidation_guard: Node,
    minimum_hold_guard: Node,
    behavior: Node,
    fallback: Node,
) -> Node:
    """Finite commitment skeleton expressed with typed BB + reactive control.

    The caller supplies mechanically meaningful entry/invalidation/minimum-hold guards.
    This avoids a generic runtime strategy-object system while making commitment logic
    reusable and auditable.
    """
    active = {"type": "Condition", "lhs": {"blackboard": mode_slot}, "op": "EQ", "rhs": committed_value}
    keep = {
        "type": "ReactiveSequence",
        "children": [
            active,
            {"type": "ReactiveSelector", "children": [minimum_hold_guard, {"type": "Invert", "child": invalidation_guard}]},
            behavior,
        ],
    }
    enter = {
        "type": "Sequence",
        "children": [entry_guard, {"type": "SetBlackboard", "slot": mode_slot, "value": committed_value}, behavior],
    }
    return {"type": "ReactiveSelector", "children": [keep, enter, fallback]}
