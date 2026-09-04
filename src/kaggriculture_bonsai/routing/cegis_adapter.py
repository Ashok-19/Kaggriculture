"""Thin RLRMB adapter to the authoritative hardened P0.6 CEGIS."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping

from kaggriculture_bonsai.composition.bonsai_synthesizer import (
    HOLDOUT_FAILURE,
    MIXED_LEGAL_STATE,
    NON_TRANSFERABLE,
    TARGET_CAPABILITY_LOST,
    THRESHOLD_BRITTLE,
    assess_sealed_holdout,
    assess_transfer_gate_support,
    run_cegis as _run_cegis,
    threshold_robustness,
)
from kaggriculture_bonsai.composition.macro_initiation_cegis import MacroEffectCase

from .routability import RoutabilityClassification, RoutabilityResult
from kaggriculture_bonsai.search.schema import DatasetRole, canonical_json, validate_runtime_identity_safe

ADAPTER_SCHEMA_VERSION = 1
_ROLE_NAMES = {
    DatasetRole.TARGET: "target_capability",
    DatasetRole.TRANSFER: "transfer",
    DatasetRole.VALIDATION: "validation",
    DatasetRole.SEALED: "sealed_holdout",
}


class AdapterDisposition(str, Enum):
    ENTER_CEGIS = "ENTER_CEGIS"
    RETURN_TO_REPAIR = "RETURN_TO_REPAIR"
    PERMANENT_REJECT = "PERMANENT_REJECT"
    DIAGNOSTIC_ONLY = "DIAGNOSTIC_ONLY"


_DISPOSITIONS = {
    RoutabilityClassification.ROUTABLE_CANDIDATE: AdapterDisposition.ENTER_CEGIS,
    RoutabilityClassification.REPAIR_LATER_CHECKPOINT: AdapterDisposition.RETURN_TO_REPAIR,
    RoutabilityClassification.REPAIR_COMMON_PREFIX: AdapterDisposition.RETURN_TO_REPAIR,
    RoutabilityClassification.REPAIR_BRANCH_AT_REVEAL: AdapterDisposition.RETURN_TO_REPAIR,
    RoutabilityClassification.REPAIR_ABORT_RECOVERY: AdapterDisposition.RETURN_TO_REPAIR,
    RoutabilityClassification.FUTURE_DEPENDENT: AdapterDisposition.PERMANENT_REJECT,
    RoutabilityClassification.GATE_LIMITATION_SUSPECTED: AdapterDisposition.DIAGNOSTIC_ONLY,
}


def _context(value: tuple[tuple[str, int, int], ...]) -> tuple[tuple[str, int, int], ...]:
    if not isinstance(value, tuple) or not value:
        raise ValueError("legal_context must be a non-empty tuple")
    slots = []
    for item in value:
        if not isinstance(item, tuple) or len(item) != 3:
            raise ValueError("legal_context entries must be (feature, index, value)")
        feature, index, feature_value = item
        if type(feature) is not str or type(index) is not int or type(feature_value) is not int:
            raise TypeError("legal_context fields must be str/int/int")
        validate_runtime_identity_safe({feature: feature_value}, "legal_context")
        slots.append((feature, index, feature_value))
    if len({item[:2] for item in slots}) != len(slots):
        raise ValueError("legal_context contains duplicate feature slots")
    return tuple(sorted(slots))


@dataclass(frozen=True)
class AdapterCase:
    case_id: str
    role: DatasetRole
    baseline_margin: int
    margin_delta: int
    legal_context: tuple[tuple[str, int, int], ...]
    exact_signature: tuple[int, ...]
    support_group: str
    family: str
    seat: int
    required_target_positive: bool = False
    required_target_l2w: bool = False
    evidence_id: str = ""
    provenance: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if type(self.case_id) is not str or not self.case_id:
            raise ValueError("case_id must be non-empty")
        if not isinstance(self.role, DatasetRole):
            raise TypeError("role must be DatasetRole")
        if type(self.baseline_margin) is not int or type(self.margin_delta) is not int:
            raise TypeError("margins must be exact integers")
        object.__setattr__(self, "legal_context", _context(self.legal_context))
        if not isinstance(self.exact_signature, tuple) or not self.exact_signature:
            raise ValueError("exact_signature must be non-empty")
        if any(type(value) is not int for value in self.exact_signature):
            raise TypeError("exact_signature must contain integers")
        if type(self.support_group) is not str or type(self.family) is not str:
            raise TypeError("support_group and family must be strings")
        if self.seat not in (0, 1):
            raise ValueError("seat must be 0 or 1")
        if self.required_target_positive or self.required_target_l2w:
            if self.role is not DatasetRole.TARGET or self.margin_delta <= 0:
                raise ValueError("required support must be positive TARGET evidence")
        if self.required_target_l2w and not (self.baseline_margin < 0 < self.baseline_margin + self.margin_delta):
            raise ValueError("required target L->W row must be an exact L->W")
        if type(self.evidence_id) is not str or not self.evidence_id:
            raise ValueError("evidence_id must be non-empty")
        if not isinstance(self.provenance, tuple) or any(type(item) is not str for item in self.provenance):
            raise TypeError("provenance must be a tuple of strings")
        object.__setattr__(self, "provenance", tuple(sorted(set(self.provenance))))


def adapt_case(row: AdapterCase) -> MacroEffectCase:
    """Faithfully convert one exact RLRMB case into the existing CEGIS type."""
    if not isinstance(row, AdapterCase):
        raise TypeError("row must be AdapterCase")
    if row.role not in _ROLE_NAMES:
        raise ValueError(f"unsupported CEGIS adapter role: {row.role.value}")
    return MacroEffectCase(
        case_id=row.case_id,
        baseline_margin=row.baseline_margin,
        macro_margin=row.baseline_margin + row.margin_delta,
        context={(feature, index): value for feature, index, value in row.legal_context},
        exact_signature=row.exact_signature,
        support_group=row.support_group,
        dataset_role=_ROLE_NAMES[row.role],
        required_target_positive=row.required_target_positive,
        required_target_l2w=row.required_target_l2w,
    )


@dataclass(frozen=True)
class CegisAdapterInput:
    candidate_id: str
    program_semantic_id: str
    program_schema_version: int
    internal_decision_count: int
    cases: tuple[AdapterCase, ...]
    routability: RoutabilityResult
    min_transfer_positive: int = 1
    min_transfer_independent: int = 1
    max_conditions: int = 4
    max_iterations: int = 8
    threshold_radius: int = 2
    lineage: tuple[str, ...] = ()
    schema_version: int = ADAPTER_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != ADAPTER_SCHEMA_VERSION:
            raise ValueError(f"unsupported adapter schema version: {self.schema_version!r}")
        for field in ("candidate_id", "program_semantic_id"):
            if type(getattr(self, field)) is not str or not getattr(self, field):
                raise ValueError(f"{field} must be non-empty")
        if self.routability.candidate_id != self.candidate_id:
            raise ValueError("routability candidate ID mismatch")
        if self.program_schema_version not in (1, 2):
            raise ValueError("program_schema_version must be 1 or 2")
        if type(self.internal_decision_count) is not int or not 0 <= self.internal_decision_count <= 2:
            raise ValueError("internal_decision_count must be in [0,2]")
        if self.program_schema_version == 1 and self.internal_decision_count:
            raise ValueError("v1 programs cannot contain internal decisions")
        if not self.cases or any(not isinstance(row, AdapterCase) for row in self.cases):
            raise ValueError("adapter input requires exact cases")
        if len({row.case_id for row in self.cases}) != len(self.cases):
            raise ValueError("case IDs must be unique")
        target = [row for row in self.cases if row.role is DatasetRole.TARGET]
        transfer = [row for row in self.cases if row.role is DatasetRole.TRANSFER]
        validation = [row for row in self.cases if row.role is DatasetRole.VALIDATION]
        if not target or not transfer or not validation:
            raise ValueError("TARGET, TRANSFER and VALIDATION evidence are required")
        if not any(row.required_target_positive for row in target):
            raise ValueError("required target-positive evidence is missing")
        if any(row.baseline_margin < 0 < row.baseline_margin + row.margin_delta for row in target):
            if not any(row.required_target_l2w for row in target):
                raise ValueError("required target L->W evidence is missing")
        for field in ("min_transfer_positive", "min_transfer_independent", "max_conditions", "max_iterations", "threshold_radius"):
            if type(getattr(self, field)) is not int or getattr(self, field) <= 0:
                raise ValueError(f"{field} must be a positive integer")
        if not isinstance(self.lineage, tuple) or any(type(item) is not str for item in self.lineage):
            raise TypeError("lineage must be a tuple of strings")
        object.__setattr__(self, "cases", tuple(sorted(self.cases, key=lambda row: row.case_id)))
        object.__setattr__(self, "lineage", tuple(sorted(set(self.lineage))))


@dataclass(frozen=True)
class CegisAdapterResult:
    candidate_id: str
    program_semantic_id: str
    program_schema_version: int
    internal_decision_count: int
    disposition: AdapterDisposition
    cegis_invoked: bool
    accepted: bool
    rejection_reason: str | None
    cegis_stop_reason: str | None
    gate: Any
    target_preserved: bool
    transfer_support: Mapping[str, Any]
    validation_counterexamples: tuple[str, ...]
    threshold_robustness: tuple[Mapping[str, Any], ...]
    sealed_status: str
    evidence_ids: tuple[str, ...]
    cegis_trace: tuple[Mapping[str, Any], ...]
    diagnostics: Mapping[str, Any]
    schema_version: int = ADAPTER_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != ADAPTER_SCHEMA_VERSION:
            raise ValueError(f"unsupported adapter result version: {self.schema_version!r}")
        if not isinstance(self.disposition, AdapterDisposition):
            raise TypeError("disposition must be AdapterDisposition")
        object.__setattr__(self, "transfer_support", MappingProxyType(dict(self.transfer_support)))
        object.__setattr__(self, "threshold_robustness", tuple(MappingProxyType(dict(row)) for row in self.threshold_robustness))
        object.__setattr__(self, "cegis_trace", tuple(MappingProxyType(dict(row)) for row in self.cegis_trace))
        object.__setattr__(self, "diagnostics", MappingProxyType(dict(self.diagnostics)))

    def to_document(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "candidate_id": self.candidate_id,
            "program_semantic_id": self.program_semantic_id,
            "program_schema_version": self.program_schema_version,
            "internal_decision_count": self.internal_decision_count,
            "disposition": self.disposition.value,
            "cegis_invoked": self.cegis_invoked,
            "accepted": self.accepted,
            "rejection_reason": self.rejection_reason,
            "cegis_stop_reason": self.cegis_stop_reason,
            "gate": self.gate,
            "target_preserved": self.target_preserved,
            "transfer_support": dict(self.transfer_support),
            "validation_counterexamples": list(self.validation_counterexamples),
            "threshold_robustness": [dict(row) for row in self.threshold_robustness],
            "sealed_status": self.sealed_status,
            "evidence_ids": list(self.evidence_ids),
            "cegis_trace": [dict(row) for row in self.cegis_trace],
            "diagnostics": dict(self.diagnostics),
        }

    def canonical_json(self) -> str:
        return canonical_json(self.to_document())

    @classmethod
    def from_document(cls, document: Mapping[str, Any]) -> "CegisAdapterResult":
        if document.get("schema_version") != ADAPTER_SCHEMA_VERSION:
            raise ValueError(f"unsupported adapter result version: {document.get('schema_version')!r}")
        expected = set(cls.__dataclass_fields__)
        if set(document) != expected:
            raise ValueError("malformed adapter result document")
        values = dict(document)
        values["disposition"] = AdapterDisposition(values["disposition"])
        values["validation_counterexamples"] = tuple(values["validation_counterexamples"])
        values["threshold_robustness"] = tuple(values["threshold_robustness"])
        values["evidence_ids"] = tuple(values["evidence_ids"])
        values["cegis_trace"] = tuple(values["cegis_trace"])
        return cls(**values)


def _base_result(source: CegisAdapterInput, disposition: AdapterDisposition) -> dict[str, Any]:
    return {
        "candidate_id": source.candidate_id,
        "program_semantic_id": source.program_semantic_id,
        "program_schema_version": source.program_schema_version,
        "internal_decision_count": source.internal_decision_count,
        "disposition": disposition,
        "cegis_invoked": False,
        "accepted": False,
        "rejection_reason": None,
        "cegis_stop_reason": None,
        "gate": None,
        "target_preserved": False,
        "transfer_support": {},
        "validation_counterexamples": (),
        "threshold_robustness": (),
        "sealed_status": "NOT_EVALUATED",
        "evidence_ids": tuple(sorted(row.evidence_id for row in source.cases)),
        "cegis_trace": (),
        "diagnostics": {
            "cegis_scope": "OUTER_INITIATION_ONLY",
            "routability": source.routability.classification.value,
            "lineage": list(source.lineage),
        },
    }


def run_hardened_cegis(source: CegisAdapterInput) -> CegisAdapterResult:
    """Invoke unchanged hardened CEGIS only for Phase-9-routable candidates."""
    if not isinstance(source, CegisAdapterInput):
        raise TypeError("source must be CegisAdapterInput")
    disposition = _DISPOSITIONS[source.routability.classification]
    payload = _base_result(source, disposition)
    if disposition is not AdapterDisposition.ENTER_CEGIS:
        payload["rejection_reason"] = (
            "FUTURE_DEPENDENT" if disposition is AdapterDisposition.PERMANENT_REJECT
            else source.routability.classification.value
        )
        return CegisAdapterResult(**payload)

    converted = {row.case_id: adapt_case(row) for row in source.cases}
    target_transfer = [converted[row.case_id] for row in source.cases if row.role in {DatasetRole.TARGET, DatasetRole.TRANSFER}]
    validation = [converted[row.case_id] for row in source.cases if row.role is DatasetRole.VALIDATION]
    transfer = [converted[row.case_id] for row in source.cases if row.role is DatasetRole.TRANSFER]
    sealed = [converted[row.case_id] for row in source.cases if row.role is DatasetRole.SEALED]
    positives = [row for row in target_transfer if row.delta > 0]
    harmful = [row for row in target_transfer if row.delta < 0 or row.win_to_loss]
    cegis = _run_cegis(
        positives, harmful, validation,
        max_conditions=source.max_conditions,
        max_iterations=source.max_iterations,
    )
    payload["cegis_invoked"] = True
    payload["cegis_stop_reason"] = cegis.stop_reason
    payload["cegis_trace"] = tuple(row.canonical() for row in cegis.trace)
    payload["validation_counterexamples"] = tuple(sorted({
        case_id for row in cegis.trace for case_id in row.newly_added_harms
    }))
    if cegis.final_gate is None:
        payload["rejection_reason"] = cegis.stop_reason
        return CegisAdapterResult(**payload)

    payload["gate"] = cegis.canonical()["final_gate"]
    required_positive = any(row.required_target_positive for row in target_transfer)
    required_l2w = any(row.required_target_l2w for row in target_transfer)
    target_preserved = (
        (not required_positive or cegis.final_score.target_positive > 0)
        and (not required_l2w or cegis.final_score.target_loss_to_win > 0)
    )
    payload["target_preserved"] = target_preserved
    if not target_preserved:
        payload["rejection_reason"] = TARGET_CAPABILITY_LOST
        return CegisAdapterResult(**payload)

    transfer_report = assess_transfer_gate_support(
        cegis.final_gate, transfer,
        min_positive=source.min_transfer_positive,
        min_independent=source.min_transfer_independent,
    )
    payload["transfer_support"] = transfer_report.canonical()
    if not transfer_report.accepted:
        payload["rejection_reason"] = NON_TRANSFERABLE
        return CegisAdapterResult(**payload)

    threshold = threshold_robustness(
        cegis.final_gate, target_transfer + validation, radius=source.threshold_radius,
    )
    payload["threshold_robustness"] = tuple(threshold)
    selected = [row for row in threshold if row["offset"] == 0]
    neighbors = [row for row in threshold if abs(row["offset"]) == 1]
    unsafe_neighbors = [row for row in neighbors if not row["safe"]]
    if any(not row["safe"] for row in selected) or (neighbors and len(unsafe_neighbors) == len(neighbors)):
        payload["rejection_reason"] = THRESHOLD_BRITTLE
        return CegisAdapterResult(**payload)

    if sealed:
        holdout = assess_sealed_holdout(cegis.final_gate, sealed)
        payload["sealed_status"] = "PASSED" if holdout.accepted else "FAILED"
        if not holdout.accepted:
            payload["rejection_reason"] = HOLDOUT_FAILURE
            return CegisAdapterResult(**payload)
    else:
        payload["sealed_status"] = "NOT_PROVIDED"

    payload["accepted"] = True
    return CegisAdapterResult(**payload)
