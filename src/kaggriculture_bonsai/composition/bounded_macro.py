from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from kaggriculture_bonsai.composition.module_codec import ConditionGene

MAX_GATE_CLAUSES = 6
MAX_GATE_CONDITIONS = 4
_FORBIDDEN_PROVENANCE_TERMS = (
    "opponent", "team", "submission", "replay", "episode", "seed", "tape",
)


def _check_identity_free(value: Any, path: str = "provenance") -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            text = str(key).lower()
            if any(term in text for term in _FORBIDDEN_PROVENANCE_TERMS):
                raise ValueError(f"identity-like field forbidden in bounded macro {path}: {key}")
            _check_identity_free(nested, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for i, nested in enumerate(value):
            _check_identity_free(nested, f"{path}[{i}]")
    elif isinstance(value, str):
        text = value.lower()
        if any(term in text for term in _FORBIDDEN_PROVENANCE_TERMS):
            raise ValueError(f"identity-like value forbidden in bounded macro {path}: {value}")


def validate_identity_free(value: Any, path: str = "provenance") -> None:
    """Validate reusable synthesis/composition provenance with the macro policy."""
    _check_identity_free(value, path)


def _clause_node(clause: Sequence[ConditionGene]) -> dict[str, Any]:
    children = [condition.node() for condition in clause]
    if len(children) == 1:
        return children[0]
    return {"type": "Sequence", "children": children}


@dataclass(frozen=True)
class BoundedRouteMacro:
    """Portable bounded route fragment with a legal initiation gate.

    The macro is authorized exactly once at ``start_step``. Authorization latches a
    module-owned BOOL, and the existing BKG4 strategic route-override BB is written
    only for the inclusive [start_step, end_step] window. After the window the macro
    writes nothing, so the parent Bonsai policy resumes automatically.
    """

    name: str
    route_id: int
    base_route_id: int
    start_step: int
    duration: int
    gate_clauses: tuple[tuple[ConditionGene, ...], ...]
    provenance: Mapping[str, Any] = field(default_factory=dict)

    @property
    def end_step(self) -> int:
        return self.start_step + self.duration - 1

    @property
    def active_slot(self) -> str:
        return f"{self.name}_active"

    def validate(self) -> None:
        if not self.name or not self.name.replace("_", "").isalnum():
            raise ValueError("bounded macro name must be non-empty alphanumeric/underscore")
        if not 0 <= int(self.route_id) <= 65535:
            raise ValueError("route_id out of range")
        if not 0 <= int(self.base_route_id) <= 65535:
            raise ValueError("base_route_id out of range")
        if self.route_id == self.base_route_id:
            raise ValueError("macro route_id must differ from base route")
        if not 0 <= int(self.start_step) < 719:
            raise ValueError("start_step out of range")
        if int(self.duration) <= 0 or self.end_step >= 719:
            raise ValueError("bounded macro window must fit the 719-action route pack")
        if not 1 <= len(self.gate_clauses) <= MAX_GATE_CLAUSES:
            raise ValueError(f"gate requires 1..{MAX_GATE_CLAUSES} clauses")
        for clause in self.gate_clauses:
            if not 1 <= len(clause) <= MAX_GATE_CONDITIONS:
                raise ValueError(f"each gate clause requires 1..{MAX_GATE_CONDITIONS} conditions")
            if len(set(clause)) != len(clause):
                raise ValueError("duplicate condition in gate clause")
            for condition in clause:
                condition.validate()
        _check_identity_free(self.provenance)

    def canonical(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema": 1,
            "kind": "bounded_route_macro_v1",
            "name": self.name,
            "route_id": int(self.route_id),
            "base_route_id": int(self.base_route_id),
            "start_step": int(self.start_step),
            "duration": int(self.duration),
            "gate_clauses": [
                [dataclasses.asdict(condition) for condition in clause]
                for clause in self.gate_clauses
            ],
            "provenance": dict(self.provenance),
        }

    @property
    def semantic_id(self) -> str:
        raw = json.dumps(self.canonical(), sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(b"BKG4-BOUNDED-ROUTE-MACRO-V1\0" + raw).hexdigest()

    def gate_node(self) -> dict[str, Any]:
        self.validate()
        clauses = [_clause_node(clause) for clause in self.gate_clauses]
        if len(clauses) == 1:
            return clauses[0]
        return {"type": "Selector", "children": clauses}

    def specialist_tree(self, *, route_override_slot: str) -> dict[str, Any]:
        """Return ordinary BKG4 source nodes implementing the bounded macro."""
        self.validate()
        return {
            "type": "Sequence",
            "module": self.name,
            "category": "strategic_lane/bounded_macro",
            "children": [
                {
                    "type": "If",
                    "condition": {"type": "Condition", "lhs": {"feature": "STEP"}, "op": "EQ", "rhs": self.start_step},
                    "then": {
                        "type": "ForceSuccess",
                        "child": {
                            "type": "Sequence",
                            "children": [
                                self.gate_node(),
                                {"type": "SetBlackboard", "slot": self.active_slot, "value": 1},
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
                            {"type": "Condition", "lhs": {"blackboard": self.active_slot}, "op": "EQ", "rhs": 1},
                            {"type": "Condition", "lhs": {"feature": "STEP"}, "op": "GE", "rhs": self.start_step},
                            {"type": "Condition", "lhs": {"feature": "STEP"}, "op": "LE", "rhs": self.end_step},
                            {"type": "SetBlackboard", "slot": route_override_slot, "value": self.route_id},
                        ],
                    },
                },
            ],
        }

    def blackboard_decl(self) -> dict[str, Any]:
        self.validate()
        return {"name": self.active_slot, "type": "BOOL", "initial": 0, "owner": self.name}

    def route_document(self, base_actions: Sequence[Mapping[str, Any]], donor_actions: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        self.validate()
        if len(base_actions) != 719 or len(donor_actions) != 719:
            raise ValueError("bounded route macro currently requires 719-action source routes")
        actions = [json.loads(json.dumps(action)) for action in base_actions]
        for step in range(self.start_step, self.end_step + 1):
            actions[step] = json.loads(json.dumps(donor_actions[step]))
        return {
            "schema": 1,
            "name": self.name,
            "route_id": self.route_id,
            "base_route": self.base_route_id,
            "entry_step": self.start_step,
            "active_window": [self.start_step, self.end_step],
            "macro_id": self.semantic_id,
            "actions": actions,
        }


def bounded_macro_windows_overlap(left: BoundedRouteMacro, right: BoundedRouteMacro) -> bool:
    """Return whether two bounded route macros can own the strategic route slot together."""
    left.validate()
    right.validate()
    return max(left.start_step, right.start_step) <= min(left.end_step, right.end_step)


def validate_nonoverlapping_macro_windows(macros: Sequence[BoundedRouteMacro]) -> None:
    """Fail closed until overlapping synthesized-macro priority has exact evidence.

    P0.3 deliberately does not build the future automatic composer. Sequential,
    disjoint bounded macros are safe to append to the existing strategic lane; an
    overlapping pair requires a later explicit composition decision rather than
    silently inheriting append order.
    """
    items = tuple(macros)
    for i, left in enumerate(items):
        left.validate()
        for right in items[i + 1:]:
            if bounded_macro_windows_overlap(left, right):
                raise ValueError(
                    f"bounded macro windows overlap without explicit composition evidence: "
                    f"{left.name}[{left.start_step},{left.end_step}] vs "
                    f"{right.name}[{right.start_step},{right.end_step}]"
                )


@dataclass(frozen=True)
class ExactRoutePreemption:
    """One evidence-backed precedence rule for an existing strategic-route owner.

    This is deliberately not a general scheduler. When the preemptor authorizes,
    it clears one named incumbent latch before setting its own latch. Both route
    owners must have the same exact window, so no restoration state is required.
    """

    preemptor_name: str
    preemptor_route_id: int
    preempted_name: str
    preempted_route_id: int
    preempted_active_slot: str
    start_step: int
    end_step: int
    evidence_kind: str

    @property
    def resume_policy(self) -> str:
        return "both_windows_end_together"

    def validate(self, macro: BoundedRouteMacro) -> None:
        macro.validate()
        if self.preemptor_name != macro.name or int(self.preemptor_route_id) != int(macro.route_id):
            raise ValueError("exact route preemption preemptor does not match macro")
        if (int(self.start_step), int(self.end_step)) != (macro.start_step, macro.end_step):
            raise ValueError("exact route preemption window must match macro window")
        if not self.preempted_name or not self.preempted_active_slot:
            raise ValueError("exact route preemption requires a named incumbent and active slot")
        if not 0 <= int(self.preempted_route_id) <= 65535:
            raise ValueError("preempted route id out of range")
        if int(self.preempted_route_id) == int(self.preemptor_route_id):
            raise ValueError("preemptor and preempted route ids must differ")
        if self.preempted_active_slot == macro.active_slot:
            raise ValueError("preempted active slot must be owned by the incumbent")
        if not self.evidence_kind:
            raise ValueError("exact route preemption requires evidence kind")
        _check_identity_free(self.evidence_kind, "composition.evidence_kind")

    def canonical(self) -> dict[str, Any]:
        return {
            "kind": "exact_route_preemption_v1",
            "preemptor": {"name": self.preemptor_name, "route_id": int(self.preemptor_route_id)},
            "preempted": {
                "name": self.preempted_name,
                "route_id": int(self.preempted_route_id),
                "active_slot": self.preempted_active_slot,
            },
            "overlap_window": [int(self.start_step), int(self.end_step)],
            "resume_policy": self.resume_policy,
            "evidence_kind": self.evidence_kind,
        }

    def specialist_tree(self, macro: BoundedRouteMacro, *, route_override_slot: str) -> dict[str, Any]:
        self.validate(macro)
        tree = macro.specialist_tree(route_override_slot=route_override_slot)
        authorization = tree["children"][0]["then"]["child"]["children"]
        authorization.insert(-1, {
            "type": "SetBlackboard",
            "slot": self.preempted_active_slot,
            "value": 0,
        })
        return tree
