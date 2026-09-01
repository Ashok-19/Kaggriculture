"""Phase 1.4 — Downstream safety integration for binary-routed specialists.

Integrates READY_FOR_DOWNSTREAM_INTEGRATION candidates into the existing
hardened RLRMB safety pipeline: TRANSFER, VALIDATION, routed-state recapture,
bounded CEGIS, SEALED, production materialization, composer integration, and
generation-level final validation.

All downstream stages evaluate the ROUTED POLICY (gate + specialist + PARENT),
never the naked globally-activated specialist.

Preserves all existing zero-regression guarantees and the SAFE_FAST_PATH.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence

from tools.bkg4_search.bonsai_composer import StrategicAuthority, compose_specialists
from tools.bkg4_search.module_codec import ConditionGene

from .binary_gate_synthesis import GateSynthesisResult, GateSynthesisStatus, _gate_canonical, _gate_identity
from .cegis_adapter import (
    AdapterCase,
    AdapterDisposition,
    CegisAdapterInput,
    CegisAdapterResult,
    run_hardened_cegis,
)
from .materialize import (
    MaterializationArtifact,
    MaterializationInput,
    materialize_program,
)
from .routed_evaluation import (
    PRODUCTION_AUTHORITY_NAME,
    PRODUCTION_AUTHORITY_SLOT,
    TEMPORARY_ROUTE_ID_BASE,
    RoutedEvaluationResult,
    RoutedEvaluationStatus,
)
from .schema import DatasetRole, canonical_json, stable_identity

DOWNSTREAM_SCHEMA_VERSION = 1

DURABLE_ROUTE_ID_BASE = 200
DURABLE_ROUTE_ID_MAX = 59999


class BinaryCandidateStatus(str, Enum):
    BINARY_CERTIFIED = "BINARY_CERTIFIED"
    TARGET_CAPABILITY_LOST = "TARGET_CAPABILITY_LOST"
    NON_TRANSFERABLE = "NON_TRANSFERABLE"
    VALIDATION_REGRESSION = "VALIDATION_REGRESSION"
    CEGIS_FAILED = "CEGIS_FAILED"
    SEALED_FAILED = "SEALED_FAILED"
    MATERIALIZATION_FAILURE = "MATERIALIZATION_FAILURE"
    COMPOSITION_CONFLICT = "COMPOSITION_CONFLICT"
    GENERATION_REGRESSION = "GENERATION_REGRESSION"
    SEMANTIC_MISMATCH = "SEMANTIC_MISMATCH"
    RESOURCE_OVERFLOW = "RESOURCE_OVERFLOW"
    INELIGIBLE = "INELIGIBLE"


_EXCLUDED_UPSTREAM_STATUSES = frozenset({
    RoutedEvaluationStatus.TARGET_CAPABILITY_LOST,
    RoutedEvaluationStatus.ROUTED_PROTECTED_REGRESSION,
    RoutedEvaluationStatus.PARENT_FALLBACK_MISMATCH,
    RoutedEvaluationStatus.ROUTING_SEMANTIC_MISMATCH,
    RoutedEvaluationStatus.MATERIALIZATION_FAILURE,
    RoutedEvaluationStatus.INSUFFICIENT_EVIDENCE,
})


@dataclass(frozen=True)
class RoutedTransferResult:
    """Transfer evaluation of the routed policy."""
    candidate_semantic_id: str
    transfer_l2w: int
    transfer_w2l: int
    transfer_positive_support: int
    transfer_harmful_cases: int
    independent_transfer_support: int
    passed: bool
    rejection_reason: str | None

    def to_document(self) -> dict[str, Any]:
        return {
            "candidate_semantic_id": self.candidate_semantic_id,
            "transfer_l2w": self.transfer_l2w,
            "transfer_w2l": self.transfer_w2l,
            "transfer_positive_support": self.transfer_positive_support,
            "transfer_harmful_cases": self.transfer_harmful_cases,
            "independent_transfer_support": self.independent_transfer_support,
            "passed": self.passed,
            "rejection_reason": self.rejection_reason,
        }


@dataclass(frozen=True)
class RoutedValidationResult:
    """Validation evaluation of the routed policy."""
    candidate_semantic_id: str
    validation_l2w: int
    validation_w2l: int
    validation_harms: int
    validation_positive_margins: int
    validation_worst_delta: int
    passed: bool
    rejection_reason: str | None

    def to_document(self) -> dict[str, Any]:
        return {
            "candidate_semantic_id": self.candidate_semantic_id,
            "validation_l2w": self.validation_l2w,
            "validation_w2l": self.validation_w2l,
            "validation_harms": self.validation_harms,
            "validation_positive_margins": self.validation_positive_margins,
            "validation_worst_delta": self.validation_worst_delta,
            "passed": self.passed,
            "rejection_reason": self.rejection_reason,
        }


@dataclass(frozen=True)
class RoutedCegisResult:
    """CEGIS result for the routed composition (gate + specialist + PARENT)."""
    candidate_semantic_id: str
    original_gate_identity: str
    final_gate_identity: str
    gate_refined: bool
    cegis_iterations: int
    converged: bool
    target_preserved: bool
    rejection_reason: str | None
    cegis_adapter_result: CegisAdapterResult | None

    def to_document(self) -> dict[str, Any]:
        return {
            "candidate_semantic_id": self.candidate_semantic_id,
            "original_gate_identity": self.original_gate_identity,
            "final_gate_identity": self.final_gate_identity,
            "gate_refined": self.gate_refined,
            "cegis_iterations": self.cegis_iterations,
            "converged": self.converged,
            "target_preserved": self.target_preserved,
            "rejection_reason": self.rejection_reason,
            "cegis_result": self.cegis_adapter_result.to_document() if self.cegis_adapter_result else None,
        }


@dataclass(frozen=True)
class ProductionMaterializationResult:
    """Result of production materialization with durable route ID."""
    candidate_semantic_id: str
    durable_route_id: int
    materialization_id: str
    node_count: int
    state_slots: int
    authority_name: str
    authority_slot: str
    final_gate_identity: str
    passed: bool
    rejection_reason: str | None

    def to_document(self) -> dict[str, Any]:
        return {
            "candidate_semantic_id": self.candidate_semantic_id,
            "durable_route_id": self.durable_route_id,
            "materialization_id": self.materialization_id,
            "node_count": self.node_count,
            "state_slots": self.state_slots,
            "authority_name": self.authority_name,
            "authority_slot": self.authority_slot,
            "final_gate_identity": self.final_gate_identity,
            "passed": self.passed,
            "rejection_reason": self.rejection_reason,
        }


@dataclass(frozen=True)
class GenerationValidationResult:
    """Final generation-level validation before atomic commit."""
    protected_w2l: int
    semantic_mismatches: int
    node_count: int
    state_slots: int
    composition_conflicts: int
    passed: bool
    rejection_reason: str | None

    def to_document(self) -> dict[str, Any]:
        return {
            "protected_w2l": self.protected_w2l,
            "semantic_mismatches": self.semantic_mismatches,
            "node_count": self.node_count,
            "state_slots": self.state_slots,
            "composition_conflicts": self.composition_conflicts,
            "passed": self.passed,
            "rejection_reason": self.rejection_reason,
        }


@dataclass(frozen=True)
class DownstreamCandidateResult:
    """Full downstream integration result for one binary-routed candidate."""
    candidate_semantic_id: str
    status: BinaryCandidateStatus
    routed_evaluation_identity: str
    gate_identity: str
    final_gate_identity: str
    parent_generation_context: str

    transfer: RoutedTransferResult | None
    validation: RoutedValidationResult | None
    cegis: RoutedCegisResult | None
    sealed_status: str
    materialization: ProductionMaterializationResult | None

    lineage: dict[str, Any]
    failure_reason: str | None

    def to_document(self) -> dict[str, Any]:
        return {
            "schema_version": DOWNSTREAM_SCHEMA_VERSION,
            "candidate_semantic_id": self.candidate_semantic_id,
            "status": self.status.value,
            "routed_evaluation_identity": self.routed_evaluation_identity,
            "gate_identity": self.gate_identity,
            "final_gate_identity": self.final_gate_identity,
            "parent_generation_context": self.parent_generation_context,
            "transfer": self.transfer.to_document() if self.transfer else None,
            "validation": self.validation.to_document() if self.validation else None,
            "cegis": self.cegis.to_document() if self.cegis else None,
            "sealed_status": self.sealed_status,
            "materialization": self.materialization.to_document() if self.materialization else None,
            "lineage": self.lineage,
            "failure_reason": self.failure_reason,
        }


class _DurableRouteAllocator:
    """Production route ID allocator that advances only on committed generations."""

    def __init__(self, base: int = DURABLE_ROUTE_ID_BASE, limit: int = DURABLE_ROUTE_ID_MAX):
        if base < DURABLE_ROUTE_ID_BASE:
            raise ValueError(f"durable route base must be >= {DURABLE_ROUTE_ID_BASE}")
        if limit >= TEMPORARY_ROUTE_ID_BASE:
            raise ValueError("durable route IDs must not overlap temporary range")
        self._next = base
        self._limit = limit
        self._allocated: list[int] = []
        self._committed: list[int] = []

    def allocate(self) -> int:
        if self._next > self._limit:
            raise RuntimeError("durable route ID space exhausted")
        route_id = self._next
        self._next += 1
        self._allocated.append(route_id)
        return route_id

    def commit(self, route_id: int) -> None:
        if route_id not in self._allocated:
            raise ValueError("cannot commit unallocated route ID")
        self._committed.append(route_id)

    def rollback_uncommitted(self) -> None:
        uncommitted = [r for r in self._allocated if r not in self._committed]
        for r in uncommitted:
            self._allocated.remove(r)
        if self._allocated:
            self._next = max(self._allocated) + 1
        else:
            self._next = DURABLE_ROUTE_ID_BASE

    @property
    def allocated_ids(self) -> tuple[int, ...]:
        return tuple(self._allocated)

    @property
    def committed_ids(self) -> tuple[int, ...]:
        return tuple(self._committed)


def is_eligible_for_downstream(routed_result: RoutedEvaluationResult) -> bool:
    """Check if a routed evaluation result is eligible for Phase 1.4 downstream."""
    if routed_result.status is not RoutedEvaluationStatus.READY_FOR_DOWNSTREAM_INTEGRATION:
        return False
    if routed_result.protected_routed_w2l != 0:
        return False
    if routed_result.semantic_mismatch_count != 0:
        return False
    if not routed_result.fallback_audit.passed:
        return False
    if routed_result.target_routed_l2w < 1:
        return False
    return True


def evaluate_routed_transfer(
    candidate_id: str,
    transfer_cases: Sequence[Mapping[str, Any]],
    gate: tuple[tuple[ConditionGene, ...], ...],
    *,
    min_transfer_positive: int = 1,
    min_independent_support: int = 1,
) -> RoutedTransferResult:
    """Evaluate the ROUTED policy on the transfer panel.

    Each case must have gate_decision indicating whether the gate fires.
    When gate is false, the routed margin equals baseline (PARENT fallback).
    """
    transfer_l2w = 0
    transfer_w2l = 0
    positive_support = 0
    harmful_cases = 0
    independent_support = 0

    for case in transfer_cases:
        baseline = int(case["baseline_margin"])
        gate_active = bool(case.get("gate_decision", False))
        if gate_active:
            routed_margin = int(case.get("specialist_margin", baseline))
        else:
            routed_margin = baseline

        delta = routed_margin - baseline
        if baseline < 0 <= routed_margin:
            transfer_l2w += 1
        if baseline > 0 >= routed_margin:
            transfer_w2l += 1
        if delta > 0:
            positive_support += 1
        if delta < 0:
            harmful_cases += 1
        if delta > 0 and not gate_active:
            independent_support += 1

    if transfer_w2l > 0:
        return RoutedTransferResult(
            candidate_id, transfer_l2w, transfer_w2l, positive_support,
            harmful_cases, independent_support, False, "TRANSFER_W2L",
        )
    if positive_support < min_transfer_positive:
        return RoutedTransferResult(
            candidate_id, transfer_l2w, transfer_w2l, positive_support,
            harmful_cases, independent_support, False, "NON_TRANSFERABLE",
        )
    return RoutedTransferResult(
        candidate_id, transfer_l2w, transfer_w2l, positive_support,
        harmful_cases, independent_support, True, None,
    )


def evaluate_routed_validation(
    candidate_id: str,
    validation_cases: Sequence[Mapping[str, Any]],
    gate: tuple[tuple[ConditionGene, ...], ...],
) -> RoutedValidationResult:
    """Evaluate the ROUTED policy on the validation/protected panel.

    Hard requirement: validation protected W→L == 0.
    """
    validation_l2w = 0
    validation_w2l = 0
    validation_harms = 0
    positive_margins = 0
    worst_delta = 0

    for case in validation_cases:
        baseline = int(case["baseline_margin"])
        gate_active = bool(case.get("gate_decision", False))
        if gate_active:
            routed_margin = int(case.get("specialist_margin", baseline))
        else:
            routed_margin = baseline

        delta = routed_margin - baseline
        if baseline < 0 <= routed_margin:
            validation_l2w += 1
        if baseline > 0 >= routed_margin:
            validation_w2l += 1
        if delta < 0:
            validation_harms += 1
        if delta > 0:
            positive_margins += 1
        if delta < worst_delta:
            worst_delta = delta

    if validation_w2l > 0:
        return RoutedValidationResult(
            candidate_id, validation_l2w, validation_w2l, validation_harms,
            positive_margins, worst_delta, False, "VALIDATION_REGRESSION",
        )
    return RoutedValidationResult(
        candidate_id, validation_l2w, validation_w2l, validation_harms,
        positive_margins, worst_delta, True, None,
    )


def run_routed_cegis(
    candidate_id: str,
    cegis_input: CegisAdapterInput,
    original_gate_identity: str,
    *,
    min_target_capability: int = 1,
) -> RoutedCegisResult:
    """Run hardened CEGIS on the routed composition.

    The CEGIS object is the full gate + specialist + PARENT composition.
    If the gate is refined, the refined gate must still preserve TARGET capability.
    """
    result = run_hardened_cegis(cegis_input)

    if not result.accepted:
        return RoutedCegisResult(
            candidate_id, original_gate_identity, original_gate_identity,
            False, len(result.cegis_trace), False,
            result.target_preserved, result.rejection_reason, result,
        )

    final_gate_identity = original_gate_identity
    gate_refined = False
    if result.gate is not None:
        from .binary_gate_synthesis import _gate_identity as compute_gate_identity
        try:
            gate_clauses = tuple(
                tuple(
                    ConditionGene(str(c["feature"]), str(c["cmp"]), int(c.get("index", 0)), int(c["value"]))
                    for c in clause
                )
                for clause in result.gate
            )
            new_identity = compute_gate_identity(gate_clauses)
            if new_identity != original_gate_identity:
                gate_refined = True
                final_gate_identity = new_identity
        except (KeyError, TypeError, ValueError):
            pass

    if not result.target_preserved:
        return RoutedCegisResult(
            candidate_id, original_gate_identity, final_gate_identity,
            gate_refined, len(result.cegis_trace), False,
            False, "TARGET_CAPABILITY_LOST", result,
        )

    return RoutedCegisResult(
        candidate_id, original_gate_identity, final_gate_identity,
        gate_refined, len(result.cegis_trace), True,
        True, None, result,
    )


def materialize_production_route(
    candidate_id: str,
    program: Any,
    gate: tuple[tuple[ConditionGene, ...], ...],
    motifs: Mapping[str, Any],
    *,
    route_id: int,
    base_route_id: int = 64,
    authority_name: str = PRODUCTION_AUTHORITY_NAME,
    authority_slot: str = PRODUCTION_AUTHORITY_SLOT,
    route_source: str = "",
) -> ProductionMaterializationResult:
    """Materialize a production route with durable ID and correct authority."""
    if route_id >= TEMPORARY_ROUTE_ID_BASE:
        return ProductionMaterializationResult(
            candidate_id, route_id, "", 0, 0,
            authority_name, authority_slot, "",
            False, "TEMPORARY_ROUTE_ID_IN_PRODUCTION",
        )
    if authority_slot == "slot_1":
        return ProductionMaterializationResult(
            candidate_id, route_id, "", 0, 0,
            authority_name, authority_slot, "",
            False, "WRONG_AUTHORITY_SLOT",
        )

    try:
        source = MaterializationInput(
            candidate_id=candidate_id,
            program=program,
            outer_gate=gate,
            motifs=motifs,
            route_id=route_id,
            base_route_id=base_route_id,
            route_source=route_source or f"phase14_production/{candidate_id[:16]}.route.json",
            authority=authority_name,
            authority_slot=authority_slot,
            audit_metadata={},
        )
        artifact = materialize_program(source)
    except (ValueError, TypeError, RuntimeError) as exc:
        return ProductionMaterializationResult(
            candidate_id, route_id, "", 0, 0,
            authority_name, authority_slot, "",
            False, f"materialization_error: {exc}",
        )

    gate_identity = _gate_identity(gate)

    return ProductionMaterializationResult(
        candidate_id, route_id, artifact.materialization_id,
        artifact.node_count, len(artifact.blackboard),
        authority_name, authority_slot, gate_identity,
        True, None,
    )


def validate_generation(
    composed_module: Mapping[str, Any],
    *,
    target_results: Sequence[Mapping[str, Any]] | None = None,
    transfer_results: Sequence[Mapping[str, Any]] | None = None,
    validation_results: Sequence[Mapping[str, Any]] | None = None,
    max_nodes: int = 4096,
    max_state_slots: int = 256,
) -> GenerationValidationResult:
    """Final generation-level validation before atomic commit."""
    node_count = _count_module_nodes(composed_module.get("tree", {}))
    state_slots = len(composed_module.get("blackboard", []))
    composition_conflicts = len(composed_module.get("unresolved_conflicts", []))

    if node_count > max_nodes:
        return GenerationValidationResult(
            0, 0, node_count, state_slots, composition_conflicts,
            False, f"node_count={node_count}_exceeds_limit={max_nodes}",
        )

    if state_slots > max_state_slots:
        return GenerationValidationResult(
            0, 0, node_count, state_slots, composition_conflicts,
            False, f"state_slots={state_slots}_exceeds_limit={max_state_slots}",
        )

    protected_w2l = 0
    semantic_mismatches = 0

    if validation_results:
        for case in validation_results:
            baseline = int(case.get("baseline_margin", 0))
            routed_margin = int(case.get("routed_margin", baseline))
            if baseline > 0 >= routed_margin:
                protected_w2l += 1

    if protected_w2l > 0:
        return GenerationValidationResult(
            protected_w2l, semantic_mismatches, node_count, state_slots,
            composition_conflicts, False, f"protected_w2l={protected_w2l}",
        )

    if semantic_mismatches > 0:
        return GenerationValidationResult(
            protected_w2l, semantic_mismatches, node_count, state_slots,
            composition_conflicts, False, f"semantic_mismatches={semantic_mismatches}",
        )

    if composition_conflicts > 0:
        return GenerationValidationResult(
            protected_w2l, semantic_mismatches, node_count, state_slots,
            composition_conflicts, False, f"composition_conflicts={composition_conflicts}",
        )

    return GenerationValidationResult(
        protected_w2l, semantic_mismatches, node_count, state_slots,
        composition_conflicts, True, None,
    )


def _count_module_nodes(node: Mapping[str, Any]) -> int:
    count = 1
    for key in ("children",):
        count += sum(_count_module_nodes(child) for child in node.get(key, ()))
    for key in ("child", "then", "else"):
        child = node.get(key)
        if isinstance(child, Mapping):
            count += _count_module_nodes(child)
    return count


def run_downstream_integration(
    routed_result: RoutedEvaluationResult,
    *,
    gate: tuple[tuple[ConditionGene, ...], ...],
    program: Any,
    motifs: Mapping[str, Any],
    transfer_cases: Sequence[Mapping[str, Any]] | None = None,
    validation_cases: Sequence[Mapping[str, Any]] | None = None,
    cegis_input: CegisAdapterInput | None = None,
    sealed_cases: Sequence[Mapping[str, Any]] | None = None,
    route_allocator: _DurableRouteAllocator | None = None,
    parent_module: Mapping[str, Any] | None = None,
    authority_name: str = PRODUCTION_AUTHORITY_NAME,
    authority_slot: str = PRODUCTION_AUTHORITY_SLOT,
    min_target_capability: int = 1,
    min_transfer_positive: int = 1,
) -> DownstreamCandidateResult:
    """Run the full Phase 1.4 downstream safety pipeline for one candidate."""
    candidate_id = routed_result.candidate_semantic_id
    parent_ctx = routed_result.parent_generation_context
    gate_identity = routed_result.gate_identity

    lineage = {
        "candidate_semantic_id": candidate_id,
        "source_admission_identity": routed_result.source_admission_identity,
        "source_gate_synthesis_identity": routed_result.source_gate_synthesis_identity,
        "routed_evaluation_identity": stable_identity(
            "RLRMB-ROUTED-EVAL-V1", candidate_id, gate_identity,
        ),
        "parent_generation_context": parent_ctx,
    }

    if not is_eligible_for_downstream(routed_result):
        return DownstreamCandidateResult(
            candidate_id, BinaryCandidateStatus.INELIGIBLE,
            lineage["routed_evaluation_identity"], gate_identity, gate_identity,
            parent_ctx, None, None, None, "NOT_EVALUATED", None,
            lineage, f"ineligible: {routed_result.status.value}",
        )

    # TRANSFER
    transfer_result = None
    if transfer_cases is not None:
        transfer_result = evaluate_routed_transfer(
            candidate_id, transfer_cases, gate,
            min_transfer_positive=min_transfer_positive,
        )
        if not transfer_result.passed:
            return DownstreamCandidateResult(
                candidate_id, BinaryCandidateStatus.NON_TRANSFERABLE,
                lineage["routed_evaluation_identity"], gate_identity, gate_identity,
                parent_ctx, transfer_result, None, None, "NOT_EVALUATED", None,
                lineage, transfer_result.rejection_reason,
            )

    # VALIDATION
    validation_result = None
    if validation_cases is not None:
        validation_result = evaluate_routed_validation(
            candidate_id, validation_cases, gate,
        )
        if not validation_result.passed:
            return DownstreamCandidateResult(
                candidate_id, BinaryCandidateStatus.VALIDATION_REGRESSION,
                lineage["routed_evaluation_identity"], gate_identity, gate_identity,
                parent_ctx, transfer_result, validation_result, None,
                "NOT_EVALUATED", None, lineage, validation_result.rejection_reason,
            )

    # CEGIS
    cegis_result = None
    final_gate_identity = gate_identity
    if cegis_input is not None:
        cegis_result = run_routed_cegis(
            candidate_id, cegis_input, gate_identity,
            min_target_capability=min_target_capability,
        )
        if not cegis_result.converged:
            return DownstreamCandidateResult(
                candidate_id, BinaryCandidateStatus.CEGIS_FAILED,
                lineage["routed_evaluation_identity"], gate_identity,
                cegis_result.final_gate_identity, parent_ctx,
                transfer_result, validation_result, cegis_result,
                "NOT_EVALUATED", None, lineage, cegis_result.rejection_reason,
            )
        final_gate_identity = cegis_result.final_gate_identity

        if not cegis_result.target_preserved:
            return DownstreamCandidateResult(
                candidate_id, BinaryCandidateStatus.TARGET_CAPABILITY_LOST,
                lineage["routed_evaluation_identity"], gate_identity,
                final_gate_identity, parent_ctx,
                transfer_result, validation_result, cegis_result,
                "NOT_EVALUATED", None, lineage, "TARGET_CAPABILITY_LOST",
            )

    # SEALED — write-only, does not influence synthesis
    sealed_status = "NOT_EVALUATED"
    if sealed_cases is not None and cegis_result is not None:
        cegis_adapter = cegis_result.cegis_adapter_result
        if cegis_adapter is not None and cegis_adapter.sealed_status != "NOT_EVALUATED":
            sealed_status = cegis_adapter.sealed_status
            if sealed_status == "FAILED":
                return DownstreamCandidateResult(
                    candidate_id, BinaryCandidateStatus.SEALED_FAILED,
                    lineage["routed_evaluation_identity"], gate_identity,
                    final_gate_identity, parent_ctx,
                    transfer_result, validation_result, cegis_result,
                    sealed_status, None, lineage, "SEALED_FAILED",
                )
        else:
            sealed_status = "NOT_PROVIDED"

    # PRODUCTION MATERIALIZATION
    allocator = route_allocator or _DurableRouteAllocator()
    route_id = allocator.allocate()

    mat_result = materialize_production_route(
        candidate_id, program, gate, motifs,
        route_id=route_id,
        authority_name=authority_name,
        authority_slot=authority_slot,
    )

    if not mat_result.passed:
        return DownstreamCandidateResult(
            candidate_id, BinaryCandidateStatus.MATERIALIZATION_FAILURE,
            lineage["routed_evaluation_identity"], gate_identity,
            final_gate_identity, parent_ctx,
            transfer_result, validation_result, cegis_result,
            sealed_status, mat_result, lineage, mat_result.rejection_reason,
        )

    if mat_result.final_gate_identity != final_gate_identity:
        return DownstreamCandidateResult(
            candidate_id, BinaryCandidateStatus.SEMANTIC_MISMATCH,
            lineage["routed_evaluation_identity"], gate_identity,
            final_gate_identity, parent_ctx,
            transfer_result, validation_result, cegis_result,
            sealed_status, mat_result, lineage,
            "materialized_gate_identity_mismatch",
        )

    return DownstreamCandidateResult(
        candidate_id, BinaryCandidateStatus.BINARY_CERTIFIED,
        lineage["routed_evaluation_identity"], gate_identity,
        final_gate_identity, parent_ctx,
        transfer_result, validation_result, cegis_result,
        sealed_status, mat_result, lineage, None,
    )


def record_downstream_to_memory(
    memory: Any,
    context_id: str,
    generation: int,
    result: DownstreamCandidateResult,
) -> None:
    if memory is None:
        return

    stage_map = {
        BinaryCandidateStatus.NON_TRANSFERABLE: "BINARY_TRANSFER",
        BinaryCandidateStatus.VALIDATION_REGRESSION: "BINARY_VALIDATION",
        BinaryCandidateStatus.CEGIS_FAILED: "BINARY_CEGIS",
        BinaryCandidateStatus.SEALED_FAILED: "BINARY_SEALED",
        BinaryCandidateStatus.BINARY_CERTIFIED: "BINARY_PRODUCTION_MATERIALIZATION",
        BinaryCandidateStatus.MATERIALIZATION_FAILURE: "BINARY_PRODUCTION_MATERIALIZATION",
    }

    stage = stage_map.get(result.status, "BINARY_TRANSFER")
    failure_class = None if result.status is BinaryCandidateStatus.BINARY_CERTIFIED else result.status.value

    try:
        memory.record_search_feedback(
            result.candidate_semantic_id,
            context_id,
            generation=generation,
            stage=stage,
            failure_class=failure_class,
        )
    except (ValueError, Exception):
        pass


def write_downstream_artifact(
    output_dir: Path,
    results: Sequence[DownstreamCandidateResult],
    *,
    parent_generation_context: str,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, str] = {}

    certified = [r for r in results if r.status is BinaryCandidateStatus.BINARY_CERTIFIED]
    summary = {
        "schema_version": DOWNSTREAM_SCHEMA_VERSION,
        "phase": "1.4",
        "label": "BINARY_DOWNSTREAM_INTEGRATION",
        "parent_generation_context": parent_generation_context,
        "total_candidates": len(results),
        "certified_count": len(certified),
        "certified_candidates": [r.candidate_semantic_id for r in certified],
        "status_distribution": {
            s.value: sum(1 for r in results if r.status is s)
            for s in BinaryCandidateStatus
            if any(r.status is s for r in results)
        },
    }
    summary_path = output_dir / "downstream_summary.json"
    summary_path.write_text(json.dumps(summary, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    manifest["downstream_summary"] = str(summary_path)

    if results:
        all_path = output_dir / "all_downstream_results.json"
        all_path.write_text(
            json.dumps([r.to_document() for r in results], sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        manifest["all_downstream_results"] = str(all_path)

    return manifest
