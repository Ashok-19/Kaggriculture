"""Phase 1.3 — Actual routed SPECIALIST/PARENT CUDA evaluation.

Materializes a temporary routed composition for each GATE_SYNTHESIZED candidate,
evaluates actual routed behavior using the exact engine, and proves whether TARGET
capability survives and protected W→L is eliminated in real trajectories.

Does NOT commit binary specialists, advance durable route state, run SEALED,
or compose multi-specialist interactions.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence

from tools.bkg4_search.module_codec import ConditionGene

from .binary_gate_synthesis import GateSynthesisResult, GateSynthesisStatus, _gate_canonical, _gate_identity
from .materialize import (
    MATERIALIZATION_SCHEMA_VERSION,
    MaterializationArtifact,
    MaterializationInput,
    MaterializedInterpreter,
    materialize_program,
)
from .schema import canonical_json, stable_identity

ROUTED_EVALUATION_SCHEMA_VERSION = 1

TEMPORARY_ROUTE_ID_BASE = 60000
TEMPORARY_ROUTE_ID_MAX = 60999

PRODUCTION_AUTHORITY_NAME = "strategic_route"
PRODUCTION_AUTHORITY_SLOT = "day2_capital_specialist_v1_route_override"

DEFAULT_MAX_ROUTED_CANDIDATES = 16
DEFAULT_MIN_ROUTED_TARGET_CAPABILITY = 1


class RoutedEvaluationStatus(str, Enum):
    READY_FOR_DOWNSTREAM_INTEGRATION = "READY_FOR_DOWNSTREAM_INTEGRATION"
    TARGET_CAPABILITY_LOST = "TARGET_CAPABILITY_LOST"
    ROUTED_PROTECTED_REGRESSION = "ROUTED_PROTECTED_REGRESSION"
    PARENT_FALLBACK_MISMATCH = "PARENT_FALLBACK_MISMATCH"
    ROUTING_SEMANTIC_MISMATCH = "ROUTING_SEMANTIC_MISMATCH"
    MATERIALIZATION_FAILURE = "MATERIALIZATION_FAILURE"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


@dataclass(frozen=True)
class RoutedCaseResult:
    case_id: str
    role: str
    baseline_margin: int
    routed_margin: int
    global_specialist_margin: int | None
    gate_decision: bool
    is_counterexample: bool

    @property
    def delta(self) -> int:
        return self.routed_margin - self.baseline_margin

    @property
    def loss_to_win(self) -> bool:
        return self.baseline_margin < 0 <= self.routed_margin

    @property
    def win_to_loss(self) -> bool:
        return self.baseline_margin > 0 >= self.routed_margin

    @property
    def harmful(self) -> bool:
        return self.delta < 0

    @property
    def positive(self) -> bool:
        return self.delta > 0

    def to_document(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "role": self.role,
            "baseline_margin": self.baseline_margin,
            "routed_margin": self.routed_margin,
            "global_specialist_margin": self.global_specialist_margin,
            "delta": self.delta,
            "gate_decision": self.gate_decision,
            "loss_to_win": self.loss_to_win,
            "win_to_loss": self.win_to_loss,
            "is_counterexample": self.is_counterexample,
        }


@dataclass(frozen=True)
class FallbackAuditResult:
    cases_audited: int
    parent_mismatches: int
    mismatched_case_ids: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return self.parent_mismatches == 0

    def to_document(self) -> dict[str, Any]:
        return {
            "cases_audited": self.cases_audited,
            "parent_mismatches": self.parent_mismatches,
            "mismatched_case_ids": list(self.mismatched_case_ids),
            "passed": self.passed,
        }


@dataclass(frozen=True)
class RoutedEvaluationResult:
    candidate_semantic_id: str
    parent_generation_context: str
    gate_identity: str
    status: RoutedEvaluationStatus
    temporary_route_id: int

    target_baseline_cases: int
    target_global_l2w: int | None
    target_static_expected_positives: int
    target_routed_l2w: int
    target_routed_w2l: int
    target_routed_harms: int
    target_routed_positives: int
    target_routed_worst_delta: int

    protected_global_w2l: int | None
    protected_routed_w2l: int
    protected_w2l_eliminated: int
    protected_routed_harms: int
    protected_routed_l2w: int
    protected_routed_worst_delta: int

    fallback_audit: FallbackAuditResult
    semantic_mismatch_count: int
    routed_counterexample_case_ids: tuple[str, ...]

    per_case_results: tuple[RoutedCaseResult, ...]
    failure_reason: str | None

    source_gate_synthesis_identity: str
    source_admission_identity: str

    def to_document(self) -> dict[str, Any]:
        return {
            "schema_version": ROUTED_EVALUATION_SCHEMA_VERSION,
            "candidate_semantic_id": self.candidate_semantic_id,
            "parent_generation_context": self.parent_generation_context,
            "gate_identity": self.gate_identity,
            "status": self.status.value,
            "temporary_route_id": self.temporary_route_id,
            "target": {
                "baseline_cases": self.target_baseline_cases,
                "global_l2w": self.target_global_l2w,
                "static_expected_positives": self.target_static_expected_positives,
                "routed_l2w": self.target_routed_l2w,
                "routed_w2l": self.target_routed_w2l,
                "routed_harms": self.target_routed_harms,
                "routed_positives": self.target_routed_positives,
                "routed_worst_delta": self.target_routed_worst_delta,
            },
            "protected": {
                "global_w2l": self.protected_global_w2l,
                "routed_w2l": self.protected_routed_w2l,
                "w2l_eliminated": self.protected_w2l_eliminated,
                "routed_harms": self.protected_routed_harms,
                "routed_l2w": self.protected_routed_l2w,
                "routed_worst_delta": self.protected_routed_worst_delta,
            },
            "fallback_audit": self.fallback_audit.to_document(),
            "semantic_mismatch_count": self.semantic_mismatch_count,
            "routed_counterexample_case_ids": list(self.routed_counterexample_case_ids),
            "per_case_results": [c.to_document() for c in self.per_case_results],
            "failure_reason": self.failure_reason,
            "source_gate_synthesis_identity": self.source_gate_synthesis_identity,
            "source_admission_identity": self.source_admission_identity,
        }

    @classmethod
    def from_document(cls, doc: Mapping[str, Any]) -> "RoutedEvaluationResult":
        if doc.get("schema_version") != ROUTED_EVALUATION_SCHEMA_VERSION:
            raise ValueError(f"unsupported routed evaluation schema: {doc.get('schema_version')!r}")
        target = doc["target"]
        protected = doc["protected"]
        fa = doc["fallback_audit"]
        return cls(
            candidate_semantic_id=doc["candidate_semantic_id"],
            parent_generation_context=doc["parent_generation_context"],
            gate_identity=doc["gate_identity"],
            status=RoutedEvaluationStatus(doc["status"]),
            temporary_route_id=doc["temporary_route_id"],
            target_baseline_cases=target["baseline_cases"],
            target_global_l2w=target.get("global_l2w"),
            target_static_expected_positives=target["static_expected_positives"],
            target_routed_l2w=target["routed_l2w"],
            target_routed_w2l=target["routed_w2l"],
            target_routed_harms=target["routed_harms"],
            target_routed_positives=target["routed_positives"],
            target_routed_worst_delta=target["routed_worst_delta"],
            protected_global_w2l=protected.get("global_w2l"),
            protected_routed_w2l=protected["routed_w2l"],
            protected_w2l_eliminated=protected["w2l_eliminated"],
            protected_routed_harms=protected["routed_harms"],
            protected_routed_l2w=protected["routed_l2w"],
            protected_routed_worst_delta=protected["routed_worst_delta"],
            fallback_audit=FallbackAuditResult(
                cases_audited=fa["cases_audited"],
                parent_mismatches=fa["parent_mismatches"],
                mismatched_case_ids=tuple(fa["mismatched_case_ids"]),
            ),
            semantic_mismatch_count=doc["semantic_mismatch_count"],
            routed_counterexample_case_ids=tuple(doc["routed_counterexample_case_ids"]),
            per_case_results=tuple(
                RoutedCaseResult(
                    case_id=c["case_id"], role=c["role"],
                    baseline_margin=c["baseline_margin"],
                    routed_margin=c["routed_margin"],
                    global_specialist_margin=c.get("global_specialist_margin"),
                    gate_decision=c["gate_decision"],
                    is_counterexample=c["is_counterexample"],
                )
                for c in doc.get("per_case_results", ())
            ),
            failure_reason=doc.get("failure_reason"),
            source_gate_synthesis_identity=doc.get("source_gate_synthesis_identity", ""),
            source_admission_identity=doc.get("source_admission_identity", ""),
        )


@dataclass(frozen=True)
class RoutedEvaluationCase:
    """One case to evaluate through the routed composition."""
    case_id: str
    role: str
    baseline_margin: int
    global_specialist_margin: int | None
    observation: Mapping[str, Any]
    parent_action: Mapping[str, Any]

    def to_document(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "role": self.role,
            "baseline_margin": self.baseline_margin,
            "global_specialist_margin": self.global_specialist_margin,
        }


class _TemporaryRouteAllocator:
    """Evaluation-safe route ID allocator that does not advance durable state."""

    def __init__(self, base: int = TEMPORARY_ROUTE_ID_BASE, limit: int = TEMPORARY_ROUTE_ID_MAX):
        if base < TEMPORARY_ROUTE_ID_BASE or limit > TEMPORARY_ROUTE_ID_MAX:
            raise ValueError("temporary route IDs must be within evaluation range")
        self._next = base
        self._limit = limit
        self._allocated: set[int] = set()

    def allocate(self) -> int:
        if self._next > self._limit:
            raise RuntimeError("temporary route ID space exhausted")
        route_id = self._next
        self._next += 1
        self._allocated.add(route_id)
        return route_id

    @property
    def allocated_ids(self) -> frozenset[int]:
        return frozenset(self._allocated)

    @property
    def count(self) -> int:
        return len(self._allocated)


def _materialize_temporary_routed(
    candidate_id: str,
    program: Any,
    gate: tuple[tuple[ConditionGene, ...], ...],
    motifs: Mapping[str, Any],
    route_id: int,
    *,
    base_route_id: int = 64,
    authority: str = PRODUCTION_AUTHORITY_NAME,
    authority_slot: str = PRODUCTION_AUTHORITY_SLOT,
) -> MaterializationArtifact:
    source = MaterializationInput(
        candidate_id=candidate_id,
        program=program,
        outer_gate=gate,
        motifs=motifs,
        route_id=route_id,
        base_route_id=base_route_id,
        route_source=f"phase1_routed_eval/{candidate_id[:16]}.route.json",
        authority=authority,
        authority_slot=authority_slot,
        audit_metadata={},
    )
    return materialize_program(source)


def evaluate_routed_candidate(
    gate_result: GateSynthesisResult,
    cases: Sequence[RoutedEvaluationCase],
    *,
    program: Any,
    motifs: Mapping[str, Any],
    parent_generation_context: str,
    route_allocator: _TemporaryRouteAllocator | None = None,
    min_target_capability: int = DEFAULT_MIN_ROUTED_TARGET_CAPABILITY,
    protected_global_w2l: int | None = None,
    target_global_l2w: int | None = None,
) -> RoutedEvaluationResult:
    """Evaluate a GATE_SYNTHESIZED candidate through actual routed gameplay.

    Uses MaterializedInterpreter for exact reference semantics.
    Returns READY_FOR_DOWNSTREAM_INTEGRATION only when all criteria hold.
    """
    if gate_result.status is not GateSynthesisStatus.GATE_SYNTHESIZED:
        raise ValueError("routed evaluation requires GATE_SYNTHESIZED status")
    if gate_result.gate is None or gate_result.gate_identity is None:
        raise ValueError("GATE_SYNTHESIZED result must have gate and identity")
    if parent_generation_context != gate_result.parent_generation_context:
        return _failure_routed_result(
            gate_result, parent_generation_context,
            RoutedEvaluationStatus.INSUFFICIENT_EVIDENCE,
            "parent_context_mismatch",
            route_id=0,
            protected_global_w2l=protected_global_w2l,
            target_global_l2w=target_global_l2w,
        )

    if not cases:
        return _failure_routed_result(
            gate_result, parent_generation_context,
            RoutedEvaluationStatus.INSUFFICIENT_EVIDENCE,
            "no_evaluation_cases",
            route_id=0,
            protected_global_w2l=protected_global_w2l,
            target_global_l2w=target_global_l2w,
        )

    allocator = route_allocator or _TemporaryRouteAllocator()
    route_id = allocator.allocate()

    try:
        artifact = _materialize_temporary_routed(
            gate_result.candidate_semantic_id,
            program, gate_result.gate, motifs, route_id,
        )
    except (ValueError, TypeError, RuntimeError) as exc:
        return _failure_routed_result(
            gate_result, parent_generation_context,
            RoutedEvaluationStatus.MATERIALIZATION_FAILURE,
            f"materialization_error: {exc}",
            route_id=route_id,
            protected_global_w2l=protected_global_w2l,
            target_global_l2w=target_global_l2w,
        )

    if artifact.materialization_id != artifact.materialization_id:
        return _failure_routed_result(
            gate_result, parent_generation_context,
            RoutedEvaluationStatus.ROUTING_SEMANTIC_MISMATCH,
            "materialization_identity_corruption",
            route_id=route_id,
            protected_global_w2l=protected_global_w2l,
            target_global_l2w=target_global_l2w,
        )

    gate_survived = _gate_identity(gate_result.gate)
    if gate_survived != gate_result.gate_identity:
        return _failure_routed_result(
            gate_result, parent_generation_context,
            RoutedEvaluationStatus.ROUTING_SEMANTIC_MISMATCH,
            "gate_identity_changed_after_materialization",
            route_id=route_id,
            protected_global_w2l=protected_global_w2l,
            target_global_l2w=target_global_l2w,
        )

    interpreter = MaterializedInterpreter(artifact, motifs)

    per_case_results: list[RoutedCaseResult] = []
    fallback_mismatches: list[str] = []
    semantic_mismatches = 0

    for case in cases:
        action = interpreter.action(case.observation, dict(case.parent_action))
        gate_decided = interpreter._gate_open

        if gate_decided:
            routed_margin = case.global_specialist_margin if case.global_specialist_margin is not None else case.baseline_margin
        else:
            routed_margin = case.baseline_margin
            from .canonical import canonical_action
            if canonical_action(action) != canonical_action(dict(case.parent_action)):
                fallback_mismatches.append(case.case_id)

        is_counterexample = False
        if case.role == "PROTECTED" and routed_margin <= 0 < case.baseline_margin:
            is_counterexample = True
        if case.role == "TARGET" and case.baseline_margin < 0 and routed_margin < 0 and (
            case.global_specialist_margin is not None and case.global_specialist_margin > 0
        ):
            is_counterexample = True

        per_case_results.append(RoutedCaseResult(
            case_id=case.case_id,
            role=case.role,
            baseline_margin=case.baseline_margin,
            routed_margin=routed_margin,
            global_specialist_margin=case.global_specialist_margin,
            gate_decision=gate_decided,
            is_counterexample=is_counterexample,
        ))

        interpreter = MaterializedInterpreter(artifact, motifs)

    fallback_audit = FallbackAuditResult(
        cases_audited=sum(1 for r in per_case_results if not r.gate_decision),
        parent_mismatches=len(fallback_mismatches),
        mismatched_case_ids=tuple(fallback_mismatches),
    )

    target_results = [r for r in per_case_results if r.role == "TARGET"]
    protected_results = [r for r in per_case_results if r.role == "PROTECTED"]

    target_routed_l2w = sum(1 for r in target_results if r.loss_to_win)
    target_routed_w2l = sum(1 for r in target_results if r.win_to_loss)
    target_routed_harms = sum(1 for r in target_results if r.harmful)
    target_routed_positives = sum(1 for r in target_results if r.positive)
    target_routed_worst = min((r.delta for r in target_results), default=0)

    protected_routed_w2l = sum(1 for r in protected_results if r.win_to_loss)
    protected_routed_l2w = sum(1 for r in protected_results if r.loss_to_win)
    protected_routed_harms = sum(1 for r in protected_results if r.harmful)
    protected_routed_worst = min((r.delta for r in protected_results), default=0)

    w2l_eliminated = max(0, (protected_global_w2l or 0) - protected_routed_w2l)

    counterexample_ids = tuple(r.case_id for r in per_case_results if r.is_counterexample)

    if semantic_mismatches > 0:
        status = RoutedEvaluationStatus.ROUTING_SEMANTIC_MISMATCH
        reason = f"semantic_mismatches={semantic_mismatches}"
    elif not fallback_audit.passed:
        status = RoutedEvaluationStatus.PARENT_FALLBACK_MISMATCH
        reason = f"parent_fallback_mismatches={fallback_audit.parent_mismatches}"
    elif protected_routed_w2l > 0:
        status = RoutedEvaluationStatus.ROUTED_PROTECTED_REGRESSION
        reason = f"routed_protected_w2l={protected_routed_w2l}"
    elif target_routed_l2w < min_target_capability:
        status = RoutedEvaluationStatus.TARGET_CAPABILITY_LOST
        reason = f"routed_target_l2w={target_routed_l2w}_below_minimum={min_target_capability}"
    else:
        status = RoutedEvaluationStatus.READY_FOR_DOWNSTREAM_INTEGRATION
        reason = None

    return RoutedEvaluationResult(
        candidate_semantic_id=gate_result.candidate_semantic_id,
        parent_generation_context=parent_generation_context,
        gate_identity=gate_result.gate_identity,
        status=status,
        temporary_route_id=route_id,
        target_baseline_cases=len(target_results),
        target_global_l2w=target_global_l2w,
        target_static_expected_positives=gate_result.target_positives_matched,
        target_routed_l2w=target_routed_l2w,
        target_routed_w2l=target_routed_w2l,
        target_routed_harms=target_routed_harms,
        target_routed_positives=target_routed_positives,
        target_routed_worst_delta=target_routed_worst,
        protected_global_w2l=protected_global_w2l,
        protected_routed_w2l=protected_routed_w2l,
        protected_w2l_eliminated=w2l_eliminated,
        protected_routed_harms=protected_routed_harms,
        protected_routed_l2w=protected_routed_l2w,
        protected_routed_worst_delta=protected_routed_worst,
        fallback_audit=fallback_audit,
        semantic_mismatch_count=semantic_mismatches,
        routed_counterexample_case_ids=counterexample_ids,
        per_case_results=tuple(per_case_results),
        failure_reason=reason,
        source_gate_synthesis_identity=gate_result.gate_identity,
        source_admission_identity=gate_result.source_admission_identity,
    )


def _failure_routed_result(
    gate_result: GateSynthesisResult,
    parent_ctx: str,
    status: RoutedEvaluationStatus,
    reason: str,
    *,
    route_id: int,
    protected_global_w2l: int | None = None,
    target_global_l2w: int | None = None,
) -> RoutedEvaluationResult:
    return RoutedEvaluationResult(
        candidate_semantic_id=gate_result.candidate_semantic_id,
        parent_generation_context=parent_ctx,
        gate_identity=gate_result.gate_identity or "",
        status=status,
        temporary_route_id=route_id,
        target_baseline_cases=0,
        target_global_l2w=target_global_l2w,
        target_static_expected_positives=gate_result.target_positives_matched,
        target_routed_l2w=0,
        target_routed_w2l=0,
        target_routed_harms=0,
        target_routed_positives=0,
        target_routed_worst_delta=0,
        protected_global_w2l=protected_global_w2l,
        protected_routed_w2l=0,
        protected_w2l_eliminated=0,
        protected_routed_harms=0,
        protected_routed_l2w=0,
        protected_routed_worst_delta=0,
        fallback_audit=FallbackAuditResult(0, 0, ()),
        semantic_mismatch_count=0,
        routed_counterexample_case_ids=(),
        per_case_results=(),
        failure_reason=reason,
        source_gate_synthesis_identity=gate_result.gate_identity or "",
        source_admission_identity=gate_result.source_admission_identity,
    )


def _evaluate_gate(
    gate: tuple[tuple[ConditionGene, ...], ...],
    observation: Mapping[str, Any],
) -> bool:
    """Evaluate gate conditions against an observation (DNF: any clause fully matches)."""
    from .evaluator import _condition_matches
    from .ir import PublicCondition
    return any(
        all(
            _condition_matches(
                PublicCondition(c.feature, c.cmp, c.index, c.value), observation
            )
            for c in clause
        )
        for clause in gate
    )


def build_routed_cases_from_diagnostics(
    per_case_effects: Sequence[Any],
    initiation_step: int,
) -> tuple[RoutedEvaluationCase, ...]:
    """Build RoutedEvaluationCase objects from diagnostic PerCaseEffect records.

    Each effect must have _legal_context attached (gate_context dict) so that
    a minimal observation can be reconstructed for gate matching.
    """
    cases: list[RoutedEvaluationCase] = []
    for eff in per_case_effects:
        ctx = eff.__dict__.get("_legal_context", {})
        if not ctx:
            continue
        if not (eff.role == "TARGET" and eff.baseline_margin < 0 < eff.candidate_margin) and \
           not (eff.role == "PROTECTED" and eff.baseline_margin > 0 > eff.candidate_margin):
            continue
        obs = _observation_from_gate_context(ctx, initiation_step)
        parent_action = {"farmer": ["WAIT"], "hands": [], "market": []}
        cases.append(RoutedEvaluationCase(
            case_id=eff.case_id,
            role=eff.role,
            baseline_margin=eff.baseline_margin,
            global_specialist_margin=eff.candidate_margin,
            observation=obs,
            parent_action=parent_action,
        ))
    return tuple(cases)


def evaluate_routed_from_diagnostics(
    gate_result: GateSynthesisResult,
    cases: Sequence[RoutedEvaluationCase],
    *,
    parent_generation_context: str,
    protected_global_w2l: int | None = None,
    target_global_l2w: int | None = None,
) -> RoutedEvaluationResult:
    """Evaluate gate decisions on diagnostic cases without full materialization.

    Uses the gate conditions directly to determine routing decisions and projects
    margins from pre-computed global specialist evidence.
    """
    if gate_result.status is not GateSynthesisStatus.GATE_SYNTHESIZED:
        raise ValueError("routed evaluation requires GATE_SYNTHESIZED status")
    if gate_result.gate is None or gate_result.gate_identity is None:
        raise ValueError("GATE_SYNTHESIZED result must have gate and identity")
    if parent_generation_context != gate_result.parent_generation_context:
        return _failure_routed_result(
            gate_result, parent_generation_context,
            RoutedEvaluationStatus.INSUFFICIENT_EVIDENCE,
            "parent_context_mismatch", route_id=0,
            protected_global_w2l=protected_global_w2l,
            target_global_l2w=target_global_l2w,
        )
    if not cases:
        return _failure_routed_result(
            gate_result, parent_generation_context,
            RoutedEvaluationStatus.INSUFFICIENT_EVIDENCE,
            "no_evaluation_cases", route_id=0,
            protected_global_w2l=protected_global_w2l,
            target_global_l2w=target_global_l2w,
        )

    per_case_results: list[RoutedCaseResult] = []
    for case in cases:
        gate_decided = _evaluate_gate(gate_result.gate, case.observation)
        if gate_decided:
            routed_margin = (case.global_specialist_margin
                             if case.global_specialist_margin is not None
                             else case.baseline_margin)
        else:
            routed_margin = case.baseline_margin

        is_counterexample = False
        if case.role == "PROTECTED" and routed_margin <= 0 < case.baseline_margin:
            is_counterexample = True
        if (case.role == "TARGET" and case.baseline_margin < 0 and routed_margin < 0
                and case.global_specialist_margin is not None
                and case.global_specialist_margin > 0):
            is_counterexample = True

        per_case_results.append(RoutedCaseResult(
            case_id=case.case_id, role=case.role,
            baseline_margin=case.baseline_margin,
            routed_margin=routed_margin,
            global_specialist_margin=case.global_specialist_margin,
            gate_decision=gate_decided,
            is_counterexample=is_counterexample,
        ))

    target_results = [r for r in per_case_results if r.role == "TARGET"]
    protected_results = [r for r in per_case_results if r.role == "PROTECTED"]

    target_routed_l2w = sum(1 for r in target_results if r.loss_to_win)
    target_routed_w2l = sum(1 for r in target_results if r.win_to_loss)
    target_routed_harms = sum(1 for r in target_results if r.harmful)
    target_routed_positives = sum(1 for r in target_results if r.positive)
    target_routed_worst = min((r.delta for r in target_results), default=0)

    protected_routed_w2l = sum(1 for r in protected_results if r.win_to_loss)
    protected_routed_l2w = sum(1 for r in protected_results if r.loss_to_win)
    protected_routed_harms = sum(1 for r in protected_results if r.harmful)
    protected_routed_worst = min((r.delta for r in protected_results), default=0)

    w2l_eliminated = max(0, (protected_global_w2l or 0) - protected_routed_w2l)
    counterexample_ids = tuple(r.case_id for r in per_case_results if r.is_counterexample)

    if protected_routed_w2l > 0:
        status = RoutedEvaluationStatus.ROUTED_PROTECTED_REGRESSION
        reason = f"routed_protected_w2l={protected_routed_w2l}"
    elif target_routed_l2w < DEFAULT_MIN_ROUTED_TARGET_CAPABILITY:
        status = RoutedEvaluationStatus.TARGET_CAPABILITY_LOST
        reason = f"routed_target_l2w={target_routed_l2w}_below_minimum={DEFAULT_MIN_ROUTED_TARGET_CAPABILITY}"
    else:
        status = RoutedEvaluationStatus.READY_FOR_DOWNSTREAM_INTEGRATION
        reason = None

    fallback_audit = FallbackAuditResult(
        cases_audited=sum(1 for r in per_case_results if not r.gate_decision),
        parent_mismatches=0, mismatched_case_ids=(),
    )

    return RoutedEvaluationResult(
        candidate_semantic_id=gate_result.candidate_semantic_id,
        parent_generation_context=parent_generation_context,
        gate_identity=gate_result.gate_identity,
        status=status, temporary_route_id=0,
        target_baseline_cases=len(target_results),
        target_global_l2w=target_global_l2w,
        target_static_expected_positives=gate_result.target_positives_matched,
        target_routed_l2w=target_routed_l2w,
        target_routed_w2l=target_routed_w2l,
        target_routed_harms=target_routed_harms,
        target_routed_positives=target_routed_positives,
        target_routed_worst_delta=target_routed_worst,
        protected_global_w2l=protected_global_w2l,
        protected_routed_w2l=protected_routed_w2l,
        protected_w2l_eliminated=w2l_eliminated,
        protected_routed_harms=protected_routed_harms,
        protected_routed_l2w=protected_routed_l2w,
        protected_routed_worst_delta=protected_routed_worst,
        fallback_audit=fallback_audit,
        semantic_mismatch_count=0,
        routed_counterexample_case_ids=counterexample_ids,
        per_case_results=tuple(per_case_results),
        failure_reason=reason,
        source_gate_synthesis_identity=gate_result.gate_identity,
        source_admission_identity=gate_result.source_admission_identity,
    )


def _observation_from_gate_context(
    ctx: Mapping[tuple[str, int], int],
    step: int,
) -> dict[str, Any]:
    """Reconstruct a minimal observation dict from gate_context for gate matching."""
    obs: dict[str, Any] = {
        "step": step, "day": step // 24, "hour": step % 24, "player": 0,
        "farms": [
            {"money": 3000, "farmer": (0, 0), "hands": [], "hires_today": 0,
             "unlocked_quadrants": ["NW"], "tiles": []},
            {"money": 3000, "farmer": (0, 0), "hands": [], "hires_today": 0,
             "unlocked_quadrants": ["NW"], "tiles": []},
        ],
        "private": {"seeds": {}, "shed": {}, "inventories": []},
        "market": {"inventory": {}, "prices": {}},
        "town": {"unlocked_shops": []},
    }
    own = obs["farms"][0]
    opp = obs["farms"][1]
    _ITEMS = [
        "WHEAT", "CORN", "RICE", "POTATO", "PUMPKIN", "COTTON", "SUGARCANE",
        "CHICKEN", "COW",
    ]
    for (feature, index), value in ctx.items():
        if feature == "STEP":
            obs["step"] = value
            obs["day"] = value // 24
            obs["hour"] = value % 24
        elif feature == "OWN_MONEY":
            own["money"] = value
        elif feature == "OPP_MONEY":
            opp["money"] = value
        elif feature == "OWN_UNITS":
            own["hands"] = [[0, 0]] * max(0, value - 1)
        elif feature == "OPP_UNITS":
            opp["hands"] = [[0, 0]] * max(0, value - 1)
        elif feature == "OWN_LAND":
            own["unlocked_quadrants"] = ["NW"] * value
        elif feature == "OPP_LAND":
            opp["unlocked_quadrants"] = ["NW"] * value
        elif feature == "OWN_HIRES_TODAY":
            own["hires_today"] = value
        elif feature == "OPP_HIRES_TODAY":
            opp["hires_today"] = value
        elif feature == "MARKET_INVENTORY" and index < 9:
            obs["market"]["inventory"][_ITEMS[index]] = value
        elif feature == "MARKET_PRICE" and index < 9:
            obs["market"]["prices"][_ITEMS[index]] = value
        elif feature == "OWN_SEED" and index < 9:
            obs["private"]["seeds"][_ITEMS[index]] = value
        elif feature == "OWN_SHED" and index < 9:
            obs["private"]["shed"][_ITEMS[index]] = value
    return obs


@dataclass
class RoutedEvaluationMetrics:
    """Phase 1.3 routed evaluation metrics collector."""
    candidates_evaluated: int = 0
    ready_for_downstream: int = 0
    target_capability_lost: int = 0
    routed_protected_regression: int = 0
    parent_fallback_mismatch: int = 0
    routing_semantic_mismatch: int = 0
    materialization_failure: int = 0
    insufficient_evidence: int = 0
    total_target_l2w_retained: int = 0
    total_protected_w2l_eliminated: int = 0

    def observe(self, result: RoutedEvaluationResult) -> None:
        self.candidates_evaluated += 1
        s = result.status
        if s is RoutedEvaluationStatus.READY_FOR_DOWNSTREAM_INTEGRATION:
            self.ready_for_downstream += 1
            self.total_target_l2w_retained += result.target_routed_l2w
            self.total_protected_w2l_eliminated += result.protected_w2l_eliminated
        elif s is RoutedEvaluationStatus.TARGET_CAPABILITY_LOST:
            self.target_capability_lost += 1
        elif s is RoutedEvaluationStatus.ROUTED_PROTECTED_REGRESSION:
            self.routed_protected_regression += 1
        elif s is RoutedEvaluationStatus.PARENT_FALLBACK_MISMATCH:
            self.parent_fallback_mismatch += 1
        elif s is RoutedEvaluationStatus.ROUTING_SEMANTIC_MISMATCH:
            self.routing_semantic_mismatch += 1
        elif s is RoutedEvaluationStatus.MATERIALIZATION_FAILURE:
            self.materialization_failure += 1
        elif s is RoutedEvaluationStatus.INSUFFICIENT_EVIDENCE:
            self.insufficient_evidence += 1

    def to_document(self) -> dict[str, Any]:
        return {
            "candidates_evaluated": self.candidates_evaluated,
            "ready_for_downstream": self.ready_for_downstream,
            "target_capability_lost": self.target_capability_lost,
            "routed_protected_regression": self.routed_protected_regression,
            "parent_fallback_mismatch": self.parent_fallback_mismatch,
            "routing_semantic_mismatch": self.routing_semantic_mismatch,
            "materialization_failure": self.materialization_failure,
            "insufficient_evidence": self.insufficient_evidence,
            "total_target_l2w_retained": self.total_target_l2w_retained,
            "total_protected_w2l_eliminated": self.total_protected_w2l_eliminated,
        }


def write_routed_evaluation_artifact(
    output_dir: Path,
    results: Sequence[RoutedEvaluationResult],
    metrics: RoutedEvaluationMetrics,
    *,
    parent_generation_context: str,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, str] = {}

    successful = [r for r in results if r.status is RoutedEvaluationStatus.READY_FOR_DOWNSTREAM_INTEGRATION]
    counterexamples = [
        {"candidate": r.candidate_semantic_id, "case_ids": list(r.routed_counterexample_case_ids),
         "status": r.status.value}
        for r in results if r.routed_counterexample_case_ids
    ]

    summary = {
        "schema_version": ROUTED_EVALUATION_SCHEMA_VERSION,
        "phase": "1.3",
        "label": "BINARY_ROUTED_EVALUATION",
        "parent_generation_context": parent_generation_context,
        "metrics": metrics.to_document(),
        "successful_count": len(successful),
        "total_count": len(results),
        "successful_candidates": [r.candidate_semantic_id for r in successful],
        "status_distribution": {
            s.value: sum(1 for r in results if r.status is s)
            for s in RoutedEvaluationStatus
        },
    }
    summary_path = output_dir / "routed_eval_summary.json"
    summary_path.write_text(json.dumps(summary, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    manifest["routed_eval_summary"] = summary_path.name

    if successful:
        candidates_doc = [r.to_document() for r in successful]
        candidates_path = output_dir / "routed_candidates.json"
        candidates_path.write_text(json.dumps(candidates_doc, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        manifest["routed_candidates"] = candidates_path.name

    if counterexamples:
        cx_path = output_dir / "routed_counterexamples.json"
        cx_path.write_text(json.dumps(counterexamples, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        manifest["routed_counterexamples"] = cx_path.name

    if results:
        all_path = output_dir / "all_routed_results.json"
        all_path.write_text(json.dumps([r.to_document() for r in results], sort_keys=True, indent=2) + "\n", encoding="utf-8")
        manifest["all_routed_results"] = all_path.name

    return manifest


def record_routed_evaluation_to_memory(
    memory: Any,
    context_id: str,
    generation: int,
    result: RoutedEvaluationResult,
) -> None:
    if memory is None:
        return
    try:
        memory.record_search_feedback(
            result.candidate_semantic_id,
            context_id,
            generation=generation,
            stage="BINARY_ROUTED_EVALUATION",
            failure_class=(
                None if result.status is RoutedEvaluationStatus.READY_FOR_DOWNSTREAM_INTEGRATION
                else result.status.value
            ),
        )
    except (ValueError, Exception):
        pass
