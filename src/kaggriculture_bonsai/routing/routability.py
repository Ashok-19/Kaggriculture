"""Bounded legal-routability diagnostics before hardened outer CEGIS."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from kaggriculture_bonsai.search.schema import DatasetRole, canonical_json, validate_runtime_identity_safe

ROUTABILITY_SCHEMA_VERSION = 1
_ALLOWED_ROLES = frozenset({DatasetRole.TARGET, DatasetRole.TRANSFER, DatasetRole.VALIDATION})
_ROUTABILITY_PROVENANCE_FEATURES = frozenset({
    "route", "route_id", "route_filename", "source", "source_file", "donor_path",
    "artifact_directory", "candidate_attempt", "attempt_number",
})


class RoutabilityClassification(str, Enum):
    ROUTABLE_CANDIDATE = "ROUTABLE_CANDIDATE"
    REPAIR_LATER_CHECKPOINT = "REPAIR_LATER_CHECKPOINT"
    REPAIR_COMMON_PREFIX = "REPAIR_COMMON_PREFIX"
    REPAIR_BRANCH_AT_REVEAL = "REPAIR_BRANCH_AT_REVEAL"
    REPAIR_ABORT_RECOVERY = "REPAIR_ABORT_RECOVERY"
    FUTURE_DEPENDENT = "FUTURE_DEPENDENT"
    GATE_LIMITATION_SUSPECTED = "GATE_LIMITATION_SUSPECTED"


_NEXT_ACTION = {
    RoutabilityClassification.ROUTABLE_CANDIDATE: "ENTER_HARDENED_CEGIS",
    RoutabilityClassification.REPAIR_LATER_CHECKPOINT: "RESAMPLE_CAUSAL_CHECKPOINT",
    RoutabilityClassification.REPAIR_COMMON_PREFIX: "RETAIN_COMMON_PREFIX",
    RoutabilityClassification.REPAIR_BRANCH_AT_REVEAL: "ADD_BOUNDED_LEGAL_BRANCH",
    RoutabilityClassification.REPAIR_ABORT_RECOVERY: "ADD_BOUNDED_ABORT_RECOVERY",
    RoutabilityClassification.FUTURE_DEPENDENT: "PERMANENT_REJECT",
    RoutabilityClassification.GATE_LIMITATION_SUSPECTED: "REVIEW_GATE_VOCABULARY",
}
_BASE_SCORE = {
    RoutabilityClassification.ROUTABLE_CANDIDATE: 100,
    RoutabilityClassification.REPAIR_LATER_CHECKPOINT: 55,
    RoutabilityClassification.REPAIR_COMMON_PREFIX: 55,
    RoutabilityClassification.REPAIR_BRANCH_AT_REVEAL: 65,
    RoutabilityClassification.REPAIR_ABORT_RECOVERY: 50,
    RoutabilityClassification.FUTURE_DEPENDENT: 0,
    RoutabilityClassification.GATE_LIMITATION_SUSPECTED: 25,
}


def _exact_step(value: Any, field: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


def _context(value: Sequence[tuple[str, int, int]], field: str) -> tuple[tuple[str, int, int], ...]:
    if not isinstance(value, (tuple, list)):
        raise TypeError(f"{field} must be a sequence")
    result = []
    for item in value:
        if not isinstance(item, (tuple, list)) or len(item) != 3:
            raise ValueError(f"{field} entries must be (feature, index, value)")
        feature, index, feature_value = item
        if type(feature) is not str or not feature:
            raise ValueError(f"{field} feature must be a non-empty string")
        if feature.casefold().replace("-", "_") in _ROUTABILITY_PROVENANCE_FEATURES:
            raise ValueError(f"provenance feature forbidden in {field}: {feature}")
        if type(index) is not int or type(feature_value) is not int:
            raise TypeError(f"{field} index/value must be integers")
        validate_runtime_identity_safe({feature: feature_value}, field)
        result.append((feature, index, feature_value))
    if len({(feature, index) for feature, index, _ in result}) != len(result):
        raise ValueError(f"{field} contains duplicate feature slots")
    return tuple(sorted(result))


@dataclass(frozen=True)
class BoundaryState:
    step: int
    exact_signature: tuple[int, ...]
    legal_context: tuple[tuple[str, int, int], ...] = ()
    gate_context: tuple[tuple[str, int, int], ...] = ()

    def __post_init__(self) -> None:
        _exact_step(self.step, "boundary step")
        if not isinstance(self.exact_signature, tuple) or not self.exact_signature:
            raise ValueError("exact legal signature must be a non-empty tuple")
        if any(type(value) is not int for value in self.exact_signature):
            raise TypeError("exact legal signature must contain integers")
        object.__setattr__(self, "legal_context", _context(self.legal_context, "legal context"))
        object.__setattr__(self, "gate_context", _context(self.gate_context, "gate context"))


@dataclass(frozen=True)
class RequiredContinuation:
    step: int
    continuation: str

    def __post_init__(self) -> None:
        _exact_step(self.step, "required continuation step")
        if type(self.continuation) is not str or not self.continuation:
            raise ValueError("required continuation must be a non-empty string")


@dataclass(frozen=True)
class TrajectoryEvidence:
    case_id: str
    role: DatasetRole
    states: tuple[BoundaryState, ...]
    requirements: tuple[RequiredContinuation, ...]
    effect_delta: int
    support_group: str
    family: str
    seat: int
    candidate_induced_drift: bool = False
    required_support: bool = False
    provenance: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if type(self.case_id) is not str or not self.case_id:
            raise ValueError("case_id must be a non-empty string")
        if not isinstance(self.role, DatasetRole):
            raise TypeError("role must be a DatasetRole")
        if not self.states or any(not isinstance(item, BoundaryState) for item in self.states):
            raise ValueError("trajectory requires boundary states")
        states = tuple(sorted(self.states, key=lambda item: item.step))
        if len({item.step for item in states}) != len(states):
            raise ValueError("trajectory boundary steps must be unique")
        object.__setattr__(self, "states", states)
        if not self.requirements or any(not isinstance(item, RequiredContinuation) for item in self.requirements):
            raise ValueError("trajectory requires continuation evidence")
        requirements = tuple(sorted(self.requirements, key=lambda item: item.step))
        if len({item.step for item in requirements}) != len(requirements):
            raise ValueError("required continuation steps must be unique")
        object.__setattr__(self, "requirements", requirements)
        if type(self.effect_delta) is not int:
            raise TypeError("effect_delta must be an integer")
        if type(self.support_group) is not str or type(self.family) is not str:
            raise TypeError("support_group and family must be strings")
        if self.seat not in (0, 1):
            raise ValueError("seat must be 0 or 1")
        if type(self.candidate_induced_drift) is not bool:
            raise TypeError("candidate_induced_drift must be bool")
        if type(self.required_support) is not bool:
            raise TypeError("required_support must be bool")
        if self.required_support and self.effect_delta <= 0:
            raise ValueError("required support must have positive candidate effect")
        if not isinstance(self.provenance, tuple) or any(type(item) is not str for item in self.provenance):
            raise TypeError("provenance must be a tuple of strings")
        object.__setattr__(self, "provenance", tuple(sorted(set(self.provenance))))


@dataclass(frozen=True)
class RoutabilityEvidence:
    candidate_id: str
    initiation_step: int
    trajectories: tuple[TrajectoryEvidence, ...]
    causal_checkpoint_steps: tuple[int, ...] = ()
    reveal_steps: tuple[int, ...] = ()
    schema_version: int = ROUTABILITY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != ROUTABILITY_SCHEMA_VERSION:
            raise ValueError(f"unsupported routability schema version: {self.schema_version!r}")
        if type(self.candidate_id) is not str or not self.candidate_id:
            raise ValueError("candidate_id must be a non-empty string")
        _exact_step(self.initiation_step, "initiation_step")
        if not self.trajectories or any(not isinstance(item, TrajectoryEvidence) for item in self.trajectories):
            raise ValueError("routability evidence requires trajectories")
        if len({item.case_id for item in self.trajectories}) != len(self.trajectories):
            raise ValueError("trajectory case IDs must be unique")
        forbidden = sorted({item.role.value for item in self.trajectories if item.role not in _ALLOWED_ROLES})
        if forbidden:
            raise ValueError(f"generation-time routability forbids roles: {forbidden}")
        for field in ("causal_checkpoint_steps", "reveal_steps"):
            values = getattr(self, field)
            if not isinstance(values, tuple):
                raise TypeError(f"{field} must be a tuple")
            for value in values:
                _exact_step(value, field)
            object.__setattr__(self, field, tuple(sorted(set(values))))
        object.__setattr__(self, "trajectories", tuple(sorted(self.trajectories, key=lambda item: item.case_id)))


@dataclass(frozen=True)
class RoutabilityResult:
    candidate_id: str
    classification: RoutabilityClassification
    evidence_summary: str
    conflicting_groups: tuple[tuple[str, ...], ...]
    support_groups: tuple[str, ...]
    first_required_action_divergence: int | None
    first_legal_divergence: int | None
    recommended_action: str
    diagnostics: Mapping[str, Any]
    data_roles_used: tuple[str, ...]
    score: int
    schema_version: int = ROUTABILITY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != ROUTABILITY_SCHEMA_VERSION:
            raise ValueError(f"unsupported routability result version: {self.schema_version!r}")
        if not isinstance(self.classification, RoutabilityClassification):
            raise TypeError("classification must be RoutabilityClassification")
        if self.recommended_action != _NEXT_ACTION[self.classification]:
            raise ValueError("classification/recommended action mismatch")
        if type(self.score) is not int or not 0 <= self.score <= 100:
            raise ValueError("routability score must be in [0,100]")
        object.__setattr__(self, "diagnostics", MappingProxyType(dict(self.diagnostics)))

    def to_document(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "candidate_id": self.candidate_id,
            "classification": self.classification.value,
            "evidence_summary": self.evidence_summary,
            "conflicting_groups": [list(item) for item in self.conflicting_groups],
            "support_groups": list(self.support_groups),
            "first_required_action_divergence": self.first_required_action_divergence,
            "first_legal_divergence": self.first_legal_divergence,
            "recommended_action": self.recommended_action,
            "diagnostics": dict(self.diagnostics),
            "data_roles_used": list(self.data_roles_used),
            "score": self.score,
        }

    def canonical_json(self) -> str:
        return canonical_json(self.to_document())

    @classmethod
    def from_document(cls, document: Mapping[str, Any]) -> "RoutabilityResult":
        if document.get("schema_version") != ROUTABILITY_SCHEMA_VERSION:
            raise ValueError(f"unsupported routability result version: {document.get('schema_version')!r}")
        expected = {
            "schema_version", "candidate_id", "classification", "evidence_summary", "conflicting_groups",
            "support_groups", "first_required_action_divergence", "first_legal_divergence",
            "recommended_action", "diagnostics", "data_roles_used", "score",
        }
        if set(document) != expected:
            raise ValueError("malformed routability result document")
        return cls(
            document["candidate_id"], RoutabilityClassification(document["classification"]),
            document["evidence_summary"], tuple(tuple(item) for item in document["conflicting_groups"]),
            tuple(document["support_groups"]), document["first_required_action_divergence"],
            document["first_legal_divergence"], document["recommended_action"],
            document["diagnostics"], tuple(document["data_roles_used"]), document["score"],
            schema_version=document["schema_version"],
        )


def _state_at(row: TrajectoryEvidence, step: int) -> BoundaryState | None:
    values = [item for item in row.states if item.step <= step]
    return values[-1] if values else None


def _required_divergence(left: TrajectoryEvidence, right: TrajectoryEvidence) -> int | None:
    a = {item.step: item.continuation for item in left.requirements}
    b = {item.step: item.continuation for item in right.requirements}
    for step in sorted(set(a) | set(b)):
        if a.get(step, "PARENT") != b.get(step, "PARENT"):
            return step
    return None


def _legal_divergence(left: TrajectoryEvidence, right: TrajectoryEvidence) -> int | None:
    a = {item.step: item.exact_signature for item in left.states}
    b = {item.step: item.exact_signature for item in right.states}
    for step in sorted(set(a) & set(b)):
        if a[step] != b[step]:
            return step
    return None


def _condition_candidates(rows: Sequence[TrajectoryEvidence], *, gate: bool) -> tuple[dict[str, Any], ...]:
    positives = {row.case_id for row in rows if row.effect_delta > 0}
    harmful = {row.case_id for row in rows if row.effect_delta < 0}
    slots: dict[tuple[str, int], dict[str, int]] = defaultdict(dict)
    for row in rows:
        boundary = row.states[0]
        context = boundary.gate_context if gate else boundary.legal_context
        for feature, index, value in context:
            slots[(feature, index)][row.case_id] = value
    candidates = []
    for (feature, index), values in sorted(slots.items()):
        for comparator in ("EQ", "NE", "LE", "GE"):
            for threshold in sorted(set(values.values())):
                def hit(value: int) -> bool:
                    return {
                        "EQ": value == threshold, "NE": value != threshold,
                        "LE": value <= threshold, "GE": value >= threshold,
                    }[comparator]
                selected = {case_id for case_id, value in values.items() if hit(value)}
                selected_positive = selected & positives
                selected_harmful = selected & harmful
                if not selected_positive or selected_harmful:
                    continue
                selected_rows = [row for row in rows if row.case_id in selected_positive]
                groups = sorted({row.support_group for row in selected_rows if row.support_group})
                seats = sorted({row.seat for row in selected_rows})
                all_groups = sorted({row.support_group for row in rows if row.support_group})
                train_groups = set(all_groups[::2])
                held_out_groups = set(all_groups[1::2])
                candidates.append({
                    "feature": feature, "index": index, "comparator": comparator, "value": threshold,
                    "positive": len(selected_positive), "harmful": 0,
                    "support_groups": groups, "seats": seats,
                    "train_support": sum(row.case_id in selected_positive and row.support_group in train_groups for row in rows),
                    "held_out_support": sum(row.case_id in selected_positive and row.support_group in held_out_groups for row in rows),
                })
    return tuple(sorted(candidates, key=lambda item: (-len(item["support_groups"]), -item["positive"], item["feature"], item["index"], item["comparator"], item["value"])))


def screen_routability(evidence: RoutabilityEvidence) -> RoutabilityResult:
    """Classify whether exact evidence merits CEGIS, repair, or permanent rejection."""
    if not isinstance(evidence, RoutabilityEvidence):
        raise TypeError("evidence must be RoutabilityEvidence")
    rows = evidence.trajectories
    conflicts = []
    repairable = []
    future = []
    ignored_nonrequired = []
    all_required_steps = []
    all_legal_steps = []
    for index, left in enumerate(rows):
        for right in rows[index + 1:]:
            required = _required_divergence(left, right)
            if required is None:
                continue
            all_required_steps.append(required)
            left_initial = _state_at(left, evidence.initiation_step)
            right_initial = _state_at(right, evidence.initiation_step)
            if left_initial is None or right_initial is None:
                raise ValueError("every trajectory must expose legal state at initiation")
            if left_initial.exact_signature != right_initial.exact_signature:
                continue
            legal = _legal_divergence(left, right)
            if legal is not None:
                all_legal_steps.append(legal)
            pair = tuple(sorted((left.case_id, right.case_id)))
            conflicts.append(pair)
            required_conflict = any(row.required_support and row.effect_delta > 0 for row in (left, right))
            if not required_conflict:
                ignored_nonrequired.append(pair)
                continue
            if legal is None or legal >= required:
                future.append((pair, required, legal))
            else:
                repairable.append((pair, required, legal, left.candidate_induced_drift or right.candidate_induced_drift))

    legal_predicates = _condition_candidates(rows, gate=False)
    gate_predicates = _condition_candidates(rows, gate=True)
    if future:
        classification = RoutabilityClassification.FUTURE_DEPENDENT
    elif repairable:
        legal_steps = {item[2] for item in repairable}
        if any(item[3] for item in repairable):
            classification = RoutabilityClassification.REPAIR_ABORT_RECOVERY
        elif legal_steps & set(evidence.reveal_steps):
            classification = RoutabilityClassification.REPAIR_BRANCH_AT_REVEAL
        elif legal_steps & set(evidence.causal_checkpoint_steps):
            classification = RoutabilityClassification.REPAIR_LATER_CHECKPOINT
        elif min(item[1] for item in repairable) > evidence.initiation_step:
            classification = RoutabilityClassification.REPAIR_COMMON_PREFIX
        else:
            classification = RoutabilityClassification.REPAIR_BRANCH_AT_REVEAL
    elif legal_predicates and not gate_predicates and any(_required_divergence(a, b) is not None for i, a in enumerate(rows) for b in rows[i + 1:]):
        classification = RoutabilityClassification.GATE_LIMITATION_SUSPECTED
    else:
        classification = RoutabilityClassification.ROUTABLE_CANDIDATE

    support_groups = tuple(sorted({row.support_group for row in rows if row.support_group and row.effect_delta > 0}))
    both_seat = 0
    for group in support_groups:
        if {row.seat for row in rows if row.support_group == group and row.effect_delta > 0} == {0, 1}:
            both_seat += 1
    required_step = min(all_required_steps) if all_required_steps else None
    if future:
        legal_step = min((item[2] for item in future if item[2] is not None), default=None)
    else:
        legal_step = min(all_legal_steps) if all_legal_steps else None
    diagnostics = {
        "exact_collision_count": len(conflicts),
        "future_conflict_count": len(future),
        "repairable_conflict_count": len(repairable),
        "ignored_nonrequired_conflicts": len(ignored_nonrequired),
        "both_seat_support_groups": both_seat,
        "legal_predicate": legal_predicates[0] if legal_predicates else None,
        "gate_predicate": gate_predicates[0] if gate_predicates else None,
        "acceptance_authority": False,
    }
    summary = (
        f"{len(rows)} trajectories; {len(conflicts)} exact initiation conflicts; "
        f"{len(future)} conflicts lack timely legal separation"
    )
    return RoutabilityResult(
        evidence.candidate_id,
        classification,
        summary,
        tuple(sorted(set(conflicts))),
        support_groups,
        required_step,
        legal_step,
        _NEXT_ACTION[classification],
        diagnostics,
        tuple(sorted({row.role.value for row in rows})),
        _BASE_SCORE[classification],
    )
