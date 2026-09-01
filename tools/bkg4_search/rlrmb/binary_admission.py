"""Phase 1.1 — Binary-route candidate admission path.

Classifies evaluated imperfect specialists into admission classes and
persists admitted BINARY_ROUTE_CANDIDATE records for Phase 1.2 gate
synthesis.  Does not synthesize gates, route, or promote candidates.

The existing SAFE_FAST_PATH (zero protected W→L) is unchanged.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence

from .diagnostics import (
    BinaryOpportunityClass,
    DiagnosticRoutabilityClass,
    ImperfectCandidateRecord,
    PerCaseEffect,
    classify_binary_opportunity,
    classify_diagnostic_routability,
)
from .schema import canonical_json, stable_identity

ADMISSION_SCHEMA_VERSION = 1

DEFAULT_SHORTLIST_LIMIT = 8


class CandidateAdmissionClass(str, Enum):
    SAFE_FAST_PATH = "SAFE_FAST_PATH"
    BINARY_ROUTE_CANDIDATE = "BINARY_ROUTE_CANDIDATE"
    GATE_LIMITED = "GATE_LIMITED"
    DEFERRED_ROUTE_CANDIDATE = "DEFERRED_ROUTE_CANDIDATE"
    TRUE_FUTURE_DEPENDENT = "TRUE_FUTURE_DEPENDENT"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class AdmissionResult:
    candidate_semantic_id: str
    admission_class: CandidateAdmissionClass
    reason: str
    target_loss_to_win: int
    target_harm_count: int
    protected_win_to_loss: int
    protected_loss_to_win: int
    protected_harm_count: int
    protected_worst_delta: int
    binary_opportunity: str
    diagnostic_routability: str

    def to_document(self) -> dict[str, Any]:
        return {
            "schema_version": ADMISSION_SCHEMA_VERSION,
            "candidate_semantic_id": self.candidate_semantic_id,
            "admission_class": self.admission_class.value,
            "reason": self.reason,
            "target_loss_to_win": self.target_loss_to_win,
            "target_harm_count": self.target_harm_count,
            "protected_win_to_loss": self.protected_win_to_loss,
            "protected_loss_to_win": self.protected_loss_to_win,
            "protected_harm_count": self.protected_harm_count,
            "protected_worst_delta": self.protected_worst_delta,
            "binary_opportunity": self.binary_opportunity,
            "diagnostic_routability": self.diagnostic_routability,
        }


def classify_candidate_admission(record: ImperfectCandidateRecord) -> AdmissionResult:
    """Classify an evaluated imperfect candidate into an admission class.

    SAFE_FAST_PATH: protected W→L == 0 — uses existing production path.
    BINARY_ROUTE_CANDIDATE: useful target capability with protected W→L,
        and routing evidence suggests binary specialist/PARENT separation.
    GATE_LIMITED: legal state separates but gate grammar cannot express it.
    DEFERRED_ROUTE_CANDIDATE: deferred-decision routing may be feasible.
    TRUE_FUTURE_DEPENDENT: correct behavior diverges before legal state can
        distinguish — permanent reject from binary route.
    UNKNOWN: insufficient evidence — fail closed.
    """
    common = dict(
        candidate_semantic_id=record.candidate_semantic_id,
        target_loss_to_win=record.target_loss_to_win,
        target_harm_count=record.target_harm_count,
        protected_win_to_loss=record.protected_win_to_loss,
        protected_loss_to_win=record.protected_loss_to_win,
        protected_harm_count=record.protected_harm_count,
        protected_worst_delta=record.protected_worst_delta,
        binary_opportunity=record.binary_opportunity,
        diagnostic_routability=record.diagnostic_routability,
    )

    if record.target_loss_to_win <= 0:
        return AdmissionResult(
            admission_class=CandidateAdmissionClass.UNKNOWN,
            reason="no_target_capability",
            **common,
        )

    if record.protected_win_to_loss == 0:
        return AdmissionResult(
            admission_class=CandidateAdmissionClass.SAFE_FAST_PATH,
            reason="zero_protected_win_to_loss",
            **common,
        )

    opportunity = BinaryOpportunityClass(record.binary_opportunity)

    if opportunity is BinaryOpportunityClass.TRUE_FUTURE_DEPENDENT:
        return AdmissionResult(
            admission_class=CandidateAdmissionClass.TRUE_FUTURE_DEPENDENT,
            reason="future_dependent_routing",
            **common,
        )

    if opportunity is BinaryOpportunityClass.GATE_LIMITED:
        return AdmissionResult(
            admission_class=CandidateAdmissionClass.GATE_LIMITED,
            reason="gate_grammar_insufficient",
            **common,
        )

    if opportunity is BinaryOpportunityClass.DEFERRED_DECISION_PLAUSIBLE:
        return AdmissionResult(
            admission_class=CandidateAdmissionClass.DEFERRED_ROUTE_CANDIDATE,
            reason="deferred_decision_routing",
            **common,
        )

    if opportunity is BinaryOpportunityClass.BINARY_PARENT_PLAUSIBLE:
        return AdmissionResult(
            admission_class=CandidateAdmissionClass.BINARY_ROUTE_CANDIDATE,
            reason="binary_parent_plausible",
            **common,
        )

    return AdmissionResult(
        admission_class=CandidateAdmissionClass.UNKNOWN,
        reason="unknown_opportunity_class",
        **common,
    )


def _admission_sort_key(record: ImperfectCandidateRecord) -> tuple[Any, ...]:
    """Deterministic lexicographic ranking for admission shortlist.

    Prefers: more target L→W, fewer protected W→L, fewer target harms,
    better worst protected delta, more aggregate target margin, then
    structural novelty via lane/checkpoint diversity, finally semantic ID
    for determinism.
    """
    return (
        -record.target_loss_to_win,
        record.protected_win_to_loss,
        record.target_harm_count,
        -record.protected_worst_delta,
        -record.target_aggregate_margin,
        record.lane,
        record.initiation_step,
        record.candidate_semantic_id,
    )


def select_binary_admission_shortlist(
    records: Sequence[ImperfectCandidateRecord],
    *,
    limit: int = DEFAULT_SHORTLIST_LIMIT,
) -> tuple[tuple[ImperfectCandidateRecord, AdmissionResult], ...]:
    """Select a bounded, deterministic shortlist of binary-route candidates.

    Only BINARY_ROUTE_CANDIDATE admissions are included. Ranking uses a
    transparent lexicographic order rewarding target capability and
    penalizing protected harm, with lane/checkpoint diversity as a
    tiebreaker.
    """
    if type(limit) is not int or limit <= 0:
        raise ValueError("admission shortlist limit must be positive")

    classified: list[tuple[ImperfectCandidateRecord, AdmissionResult]] = []
    for record in records:
        result = classify_candidate_admission(record)
        if result.admission_class is CandidateAdmissionClass.BINARY_ROUTE_CANDIDATE:
            classified.append((record, result))

    classified.sort(key=lambda pair: _admission_sort_key(pair[0]))

    seen_effects: set[str] = set()
    seen_epochs: set[tuple[str, int]] = set()
    selected: list[tuple[ImperfectCandidateRecord, AdmissionResult]] = []

    while classified and len(selected) < limit:
        def key(pair: tuple[ImperfectCandidateRecord, AdmissionResult]) -> tuple[Any, ...]:
            rec = pair[0]
            epoch = (rec.lane, rec.initiation_step // 24)
            novelty = (
                (rec.effect_signature not in seen_effects)
                + (epoch not in seen_epochs)
            )
            return (-novelty, *_admission_sort_key(rec))

        best = min(classified, key=key)
        classified.remove(best)
        selected.append(best)
        seen_effects.add(best[0].effect_signature)
        seen_epochs.add((best[0].lane, best[0].initiation_step // 24))

    return tuple(selected)


@dataclass(frozen=True)
class BinaryAdmissionEvidence:
    """Durable evidence package for one admitted BINARY_ROUTE_CANDIDATE."""
    candidate_semantic_id: str
    structural_id: str
    family: str
    lane: str
    initiation_step: int
    checkpoint_metadata: dict[str, Any]
    admission_class: str
    admission_reason: str
    target_loss_to_win: int
    target_harm_count: int
    target_worst_delta: int
    target_aggregate_margin: int
    protected_win_to_loss: int
    protected_loss_to_win: int
    protected_harm_count: int
    protected_worst_delta: int
    protected_aggregate_margin: int
    effect_signature: str
    routability_classification: str
    diagnostic_routability: str
    binary_opportunity: str
    per_case_target_positive: tuple[dict[str, Any], ...]
    per_case_protected_break: tuple[dict[str, Any], ...]
    gate_evidence: dict[str, Any]
    effect_vector_doc: dict[str, Any] | None
    parent_generation_context: str

    def to_document(self) -> dict[str, Any]:
        return {
            "schema_version": ADMISSION_SCHEMA_VERSION,
            "candidate_semantic_id": self.candidate_semantic_id,
            "structural_id": self.structural_id,
            "family": self.family,
            "lane": self.lane,
            "initiation_step": self.initiation_step,
            "checkpoint_metadata": self.checkpoint_metadata,
            "admission_class": self.admission_class,
            "admission_reason": self.admission_reason,
            "target_loss_to_win": self.target_loss_to_win,
            "target_harm_count": self.target_harm_count,
            "target_worst_delta": self.target_worst_delta,
            "target_aggregate_margin": self.target_aggregate_margin,
            "protected_win_to_loss": self.protected_win_to_loss,
            "protected_loss_to_win": self.protected_loss_to_win,
            "protected_harm_count": self.protected_harm_count,
            "protected_worst_delta": self.protected_worst_delta,
            "protected_aggregate_margin": self.protected_aggregate_margin,
            "effect_signature": self.effect_signature,
            "routability_classification": self.routability_classification,
            "diagnostic_routability": self.diagnostic_routability,
            "binary_opportunity": self.binary_opportunity,
            "per_case_target_positive": list(self.per_case_target_positive),
            "per_case_protected_break": list(self.per_case_protected_break),
            "gate_evidence": self.gate_evidence,
            "effect_vector": self.effect_vector_doc,
            "parent_generation_context": self.parent_generation_context,
        }

    @classmethod
    def from_document(cls, doc: Mapping[str, Any]) -> "BinaryAdmissionEvidence":
        if doc.get("schema_version") != ADMISSION_SCHEMA_VERSION:
            raise ValueError(f"unsupported admission schema: {doc.get('schema_version')!r}")
        return cls(
            candidate_semantic_id=doc["candidate_semantic_id"],
            structural_id=doc["structural_id"],
            family=doc["family"],
            lane=doc["lane"],
            initiation_step=doc["initiation_step"],
            checkpoint_metadata=doc["checkpoint_metadata"],
            admission_class=doc["admission_class"],
            admission_reason=doc["admission_reason"],
            target_loss_to_win=doc["target_loss_to_win"],
            target_harm_count=doc["target_harm_count"],
            target_worst_delta=doc["target_worst_delta"],
            target_aggregate_margin=doc["target_aggregate_margin"],
            protected_win_to_loss=doc["protected_win_to_loss"],
            protected_loss_to_win=doc["protected_loss_to_win"],
            protected_harm_count=doc["protected_harm_count"],
            protected_worst_delta=doc["protected_worst_delta"],
            protected_aggregate_margin=doc["protected_aggregate_margin"],
            effect_signature=doc["effect_signature"],
            routability_classification=doc["routability_classification"],
            diagnostic_routability=doc["diagnostic_routability"],
            binary_opportunity=doc["binary_opportunity"],
            per_case_target_positive=tuple(doc["per_case_target_positive"]),
            per_case_protected_break=tuple(doc["per_case_protected_break"]),
            gate_evidence=doc["gate_evidence"],
            effect_vector_doc=doc.get("effect_vector"),
            parent_generation_context=doc["parent_generation_context"],
        )


def build_admission_evidence(
    record: ImperfectCandidateRecord,
    admission: AdmissionResult,
    *,
    parent_generation_context: str,
) -> BinaryAdmissionEvidence:
    """Build a durable evidence package from an admitted candidate record."""
    target_positive = tuple(
        e.to_document() for e in record.per_case_effects
        if e.role == "TARGET" and e.solves
    )
    protected_break = tuple(
        e.to_document() for e in record.per_case_effects
        if e.role == "PROTECTED" and e.breaks
    )
    return BinaryAdmissionEvidence(
        candidate_semantic_id=record.candidate_semantic_id,
        structural_id=record.structural_id,
        family=record.family,
        lane=record.lane,
        initiation_step=record.initiation_step,
        checkpoint_metadata=record.checkpoint_metadata,
        admission_class=admission.admission_class.value,
        admission_reason=admission.reason,
        target_loss_to_win=record.target_loss_to_win,
        target_harm_count=record.target_harm_count,
        target_worst_delta=record.target_worst_delta,
        target_aggregate_margin=record.target_aggregate_margin,
        protected_win_to_loss=record.protected_win_to_loss,
        protected_loss_to_win=record.protected_loss_to_win,
        protected_harm_count=record.protected_harm_count,
        protected_worst_delta=record.protected_worst_delta,
        protected_aggregate_margin=record.protected_aggregate_margin,
        effect_signature=record.effect_signature,
        routability_classification=record.routability_classification,
        diagnostic_routability=record.diagnostic_routability,
        binary_opportunity=record.binary_opportunity,
        per_case_target_positive=target_positive,
        per_case_protected_break=protected_break,
        gate_evidence=record.gate_evidence,
        effect_vector_doc=record.effect_vector_doc,
        parent_generation_context=parent_generation_context,
    )


@dataclass
class BinaryAdmissionMetrics:
    """Phase-1.1 admission metrics collector."""
    useful_imperfect_candidates_seen: int = 0
    binary_route_candidates_admitted: int = 0
    gate_limited_candidates: int = 0
    deferred_route_candidates: int = 0
    true_future_dependent_candidates: int = 0
    unknown_route_candidates: int = 0
    safe_fast_path_candidates: int = 0
    binary_admission_by_family: dict[str, int] | None = None
    binary_admission_by_lane: dict[str, int] | None = None

    def __post_init__(self) -> None:
        if self.binary_admission_by_family is None:
            self.binary_admission_by_family = {}
        if self.binary_admission_by_lane is None:
            self.binary_admission_by_lane = {}

    def observe(self, result: AdmissionResult, family: str, lane: str) -> None:
        self.useful_imperfect_candidates_seen += 1
        cls = result.admission_class
        if cls is CandidateAdmissionClass.SAFE_FAST_PATH:
            self.safe_fast_path_candidates += 1
        elif cls is CandidateAdmissionClass.BINARY_ROUTE_CANDIDATE:
            self.binary_route_candidates_admitted += 1
            self.binary_admission_by_family[family] = (
                self.binary_admission_by_family.get(family, 0) + 1
            )
            self.binary_admission_by_lane[lane] = (
                self.binary_admission_by_lane.get(lane, 0) + 1
            )
        elif cls is CandidateAdmissionClass.GATE_LIMITED:
            self.gate_limited_candidates += 1
        elif cls is CandidateAdmissionClass.DEFERRED_ROUTE_CANDIDATE:
            self.deferred_route_candidates += 1
        elif cls is CandidateAdmissionClass.TRUE_FUTURE_DEPENDENT:
            self.true_future_dependent_candidates += 1
        elif cls is CandidateAdmissionClass.UNKNOWN:
            self.unknown_route_candidates += 1

    def to_document(self) -> dict[str, Any]:
        return {
            "useful_imperfect_candidates_seen": self.useful_imperfect_candidates_seen,
            "binary_route_candidates_admitted": self.binary_route_candidates_admitted,
            "gate_limited_candidates": self.gate_limited_candidates,
            "deferred_route_candidates": self.deferred_route_candidates,
            "true_future_dependent_candidates": self.true_future_dependent_candidates,
            "unknown_route_candidates": self.unknown_route_candidates,
            "safe_fast_path_candidates": self.safe_fast_path_candidates,
            "binary_admission_by_family": dict(sorted(
                (self.binary_admission_by_family or {}).items()
            )),
            "binary_admission_by_lane": dict(sorted(
                (self.binary_admission_by_lane or {}).items()
            )),
        }


def classify_all_candidates(
    records: Sequence[ImperfectCandidateRecord],
) -> tuple[tuple[ImperfectCandidateRecord, AdmissionResult], ...]:
    """Classify every diagnostic record into an admission class."""
    return tuple(
        (record, classify_candidate_admission(record))
        for record in records
    )


def write_admission_artifact(
    output_dir: Path,
    shortlist: Sequence[tuple[ImperfectCandidateRecord, AdmissionResult]],
    all_classified: Sequence[tuple[ImperfectCandidateRecord, AdmissionResult]],
    metrics: BinaryAdmissionMetrics,
    *,
    parent_generation_context: str,
) -> dict[str, str]:
    """Persist Phase-1.1 admission artifacts and return path manifest."""
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, str] = {}

    # Admission summary
    summary = {
        "schema_version": ADMISSION_SCHEMA_VERSION,
        "phase": "1.1",
        "label": "BINARY_ROUTE_CANDIDATE_ADMISSION",
        "parent_generation_context": parent_generation_context,
        "metrics": metrics.to_document(),
        "shortlist_size": len(shortlist),
        "shortlist_candidate_ids": [
            pair[0].candidate_semantic_id for pair in shortlist
        ],
        "all_classifications": {
            cls.value: sum(
                1 for _, r in all_classified if r.admission_class is cls
            )
            for cls in CandidateAdmissionClass
        },
        "unresolved": [
            r.to_document() for _, r in all_classified
            if r.admission_class in {
                CandidateAdmissionClass.GATE_LIMITED,
                CandidateAdmissionClass.DEFERRED_ROUTE_CANDIDATE,
                CandidateAdmissionClass.TRUE_FUTURE_DEPENDENT,
                CandidateAdmissionClass.UNKNOWN,
            }
        ],
    }
    summary_path = output_dir / "admission_summary.json"
    summary_path.write_text(
        json.dumps(summary, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    manifest["admission_summary"] = summary_path.name

    # Per-candidate evidence for admitted shortlist
    if shortlist:
        evidence_list = []
        for record, admission in shortlist:
            evidence = build_admission_evidence(
                record, admission,
                parent_generation_context=parent_generation_context,
            )
            evidence_list.append(evidence.to_document())
        evidence_path = output_dir / "binary_route_candidates.json"
        evidence_path.write_text(
            json.dumps(evidence_list, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        manifest["binary_route_candidates"] = evidence_path.name

    return manifest


def record_admission_to_memory(
    memory: Any,
    context_id: str,
    generation: int,
    admission: AdmissionResult,
) -> None:
    """Record candidate admission class in SearchMemory search_feedback.

    Reuses the existing search_feedback table with a BINARY_ADMISSION stage.
    Only records for candidates that already have exact evaluation records.
    """
    if memory is None:
        return
    try:
        memory.record_search_feedback(
            admission.candidate_semantic_id,
            context_id,
            generation=generation,
            stage="BINARY_ADMISSION",
            failure_class=(
                None if admission.admission_class
                is CandidateAdmissionClass.BINARY_ROUTE_CANDIDATE
                else admission.admission_class.value
            ),
        )
    except (ValueError, Exception):
        pass
