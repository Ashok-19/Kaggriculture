"""Machine-readable RLRMB search failures and high-level next actions."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping

from .schema import RLRMB_SCHEMA_VERSION, canonical_json


class FailureClass(str, Enum):
    NO_HEADROOM = "NO_HEADROOM"
    BAD_CHECKPOINT = "BAD_CHECKPOINT"
    BAD_HORIZON = "BAD_HORIZON"
    OPEN_LOOP_FRAGILITY = "OPEN_LOOP_FRAGILITY"
    MIXED_INITIATION = "MIXED_INITIATION"
    NON_TRANSFERABLE = "NON_TRANSFERABLE"
    DONOR_LIMITATION = "DONOR_LIMITATION"
    REPRESENTATION_LIMITATION = "REPRESENTATION_LIMITATION"
    TARGET_CAPABILITY_LOST = "TARGET_CAPABILITY_LOST"
    GATE_LIMITATION = "GATE_LIMITATION"
    FUTURE_DEPENDENT = "FUTURE_DEPENDENT"
    SEARCH_BUDGET_EXHAUSTED = "SEARCH_BUDGET_EXHAUSTED"
    CUDA_FAULT = "CUDA_FAULT"
    SEMANTIC_MISMATCH = "SEMANTIC_MISMATCH"


class RepairAction(str, Enum):
    ABANDON_BASIN = "ABANDON_BASIN"
    RESAMPLE_CHECKPOINT = "RESAMPLE_CHECKPOINT"
    ADJUST_HORIZON = "ADJUST_HORIZON"
    REACTIVE_REPAIR = "REACTIVE_REPAIR"
    SEMANTIC_REPAIR = "SEMANTIC_REPAIR"
    EXTEND_REPRESENTATION = "EXTEND_REPRESENTATION"
    PRESERVE_TARGET = "PRESERVE_TARGET"
    REVIEW_GATE_LIMITATION = "REVIEW_GATE_LIMITATION"
    PERMANENT_REJECT = "PERMANENT_REJECT"
    SWITCH_BASIN_OR_OPERATOR = "SWITCH_BASIN_OR_OPERATOR"
    INFRASTRUCTURE_STOP = "INFRASTRUCTURE_STOP"


FAILURE_ACTIONS: Mapping[FailureClass, RepairAction] = MappingProxyType({
    FailureClass.NO_HEADROOM: RepairAction.ABANDON_BASIN,
    FailureClass.BAD_CHECKPOINT: RepairAction.RESAMPLE_CHECKPOINT,
    FailureClass.BAD_HORIZON: RepairAction.ADJUST_HORIZON,
    FailureClass.OPEN_LOOP_FRAGILITY: RepairAction.REACTIVE_REPAIR,
    FailureClass.MIXED_INITIATION: RepairAction.REACTIVE_REPAIR,
    FailureClass.NON_TRANSFERABLE: RepairAction.SEMANTIC_REPAIR,
    FailureClass.DONOR_LIMITATION: RepairAction.EXTEND_REPRESENTATION,
    FailureClass.REPRESENTATION_LIMITATION: RepairAction.EXTEND_REPRESENTATION,
    FailureClass.TARGET_CAPABILITY_LOST: RepairAction.PRESERVE_TARGET,
    FailureClass.GATE_LIMITATION: RepairAction.REVIEW_GATE_LIMITATION,
    FailureClass.FUTURE_DEPENDENT: RepairAction.PERMANENT_REJECT,
    FailureClass.SEARCH_BUDGET_EXHAUSTED: RepairAction.SWITCH_BASIN_OR_OPERATOR,
    FailureClass.CUDA_FAULT: RepairAction.INFRASTRUCTURE_STOP,
    FailureClass.SEMANTIC_MISMATCH: RepairAction.INFRASTRUCTURE_STOP,
})


@dataclass(frozen=True)
class SearchFailure:
    """Small immutable record for a material candidate rejection."""

    candidate_id: str
    failure_class: FailureClass
    stage: str
    evidence_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if type(self.candidate_id) is not str or not self.candidate_id.strip():
            raise ValueError("candidate_id must be a non-empty string")
        if not isinstance(self.failure_class, FailureClass):
            raise TypeError("failure_class must be a FailureClass")
        if type(self.stage) is not str or not self.stage.strip():
            raise ValueError("stage must be a non-empty string")
        if not isinstance(self.evidence_ids, tuple):
            raise TypeError("evidence_ids must be a tuple")
        if any(type(item) is not str or not item.strip() for item in self.evidence_ids):
            raise ValueError("evidence_ids must contain non-empty strings")
        object.__setattr__(self, "evidence_ids", tuple(sorted(set(self.evidence_ids))))

    @property
    def suggested_action(self) -> RepairAction:
        return FAILURE_ACTIONS[self.failure_class]

    def to_document(self) -> dict[str, Any]:
        return {
            "schema_version": RLRMB_SCHEMA_VERSION,
            "candidate_id": self.candidate_id,
            "failure_class": self.failure_class.value,
            "stage": self.stage,
            "evidence_ids": list(self.evidence_ids),
            "suggested_action": self.suggested_action.value,
        }

    def canonical_json(self) -> str:
        return canonical_json(self.to_document())
