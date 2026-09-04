"""Canonical ASPE SpecialistProgram v1 representation."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from kaggriculture_bonsai.composition.module_codec import ConditionGene
from kaggriculture_bonsai.search.checkpoints import Checkpoint, EventType, Lane
from kaggriculture_bonsai.search.ir import (
    ACTION_HORIZON,
    DecisionPoint,
    FallbackPolicy,
    MacroProgram,
    Phase,
    PublicCondition,
    REACTIVE_IR_SCHEMA_VERSION,
)
from kaggriculture_bonsai.search.schema import stable_identity

SPECIALIST_PROGRAM_SCHEMA = "kaggriculture.aspe.specialist-program"
SPECIALIST_PROGRAM_SCHEMA_VERSION = 1
MAX_INITIATION_CLAUSES = 6
MAX_INITIATION_CONDITIONS = 4
MAX_BODY_PHASES = 3
MAX_REACTIVE_DECISIONS = 2

ASPE_PUBLIC_FEATURES = frozenset({
    "STEP", "DAY", "HOUR", "SHOP_AT", "SHOP_SEEN",
    "OWN_MONEY", "OPP_MONEY", "OWN_UNITS", "OPP_UNITS",
    "OWN_LAND", "OPP_LAND", "OWN_HIRES_TODAY", "OPP_HIRES_TODAY",
    "MARKET_PRICE", "MARKET_INVENTORY", "OWN_SHED", "OWN_SEED",
})


def _condition_key(condition: PublicCondition) -> tuple[str, int, str, int]:
    return condition.feature, condition.index, condition.comparator, condition.value


def _validate_aspe_condition(condition: PublicCondition, location: str) -> None:
    if not isinstance(condition, PublicCondition):
        raise TypeError(f"{location} must contain PublicCondition values")
    if condition.feature not in ASPE_PUBLIC_FEATURES:
        raise ValueError(f"{location} uses non-strategic feature {condition.feature}")
    if condition.feature == "STEP" and condition.comparator == "EQ":
        raise ValueError(f"{location} must use a range/event condition instead of exact STEP equality")


def _canonical_clauses(
    value: Sequence[Sequence[PublicCondition]],
) -> tuple[tuple[PublicCondition, ...], ...]:
    if not isinstance(value, (list, tuple)) or not 1 <= len(value) <= MAX_INITIATION_CLAUSES:
        raise ValueError(f"initiation requires 1..{MAX_INITIATION_CLAUSES} clauses")
    normalized = []
    for clause in value:
        if not isinstance(clause, (list, tuple)) or not 1 <= len(clause) <= MAX_INITIATION_CONDITIONS:
            raise ValueError(
                f"each initiation clause requires 1..{MAX_INITIATION_CONDITIONS} conditions"
            )
        for condition in clause:
            _validate_aspe_condition(condition, "initiation")
        if len(set(clause)) != len(clause):
            raise ValueError("duplicate condition in initiation clause")
        normalized.append(tuple(sorted(clause, key=_condition_key)))
    if len(set(normalized)) != len(normalized):
        raise ValueError("duplicate initiation clause")
    return tuple(sorted(normalized, key=lambda clause: tuple(_condition_key(c) for c in clause)))


def _reject_unknown(document: Mapping[str, Any], allowed: set[str], kind: str) -> None:
    unknown = set(document) - allowed
    if unknown:
        raise ValueError(f"unsupported {kind} fields: {sorted(unknown)}")


@dataclass(frozen=True)
class SpecialistProgram:
    """A compact state-conditioned option with mandatory exact parent fallback.

    Initiation is DNF and is evaluated once at ``activation_step``. Each active
    phase overrides only its declared action families; all other families remain
    PARENT. Finishing, aborting, faulting, or missing initiation returns PARENT.
    """

    lane: Lane
    activation_step: int
    initiation_clauses: tuple[tuple[PublicCondition, ...], ...]
    phases: tuple[Phase, ...]
    max_total_steps: int
    decisions: tuple[DecisionPoint, ...] = ()
    recovery_phase: Phase | None = None
    parent_fallback: FallbackPolicy = FallbackPolicy.RETURN_TO_PARENT
    schema_version: int = SPECIALIST_PROGRAM_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SPECIALIST_PROGRAM_SCHEMA_VERSION:
            raise ValueError(f"unsupported SpecialistProgram schema version: {self.schema_version!r}")
        if not isinstance(self.lane, Lane):
            raise TypeError("lane must be a Lane")
        if type(self.activation_step) is not int or not 0 <= self.activation_step < ACTION_HORIZON:
            raise ValueError("activation_step must fit the playable action horizon")
        object.__setattr__(self, "initiation_clauses", _canonical_clauses(self.initiation_clauses))
        if not isinstance(self.phases, (list, tuple)) or not 1 <= len(self.phases) <= MAX_BODY_PHASES:
            raise ValueError(f"SpecialistProgram requires 1..{MAX_BODY_PHASES} body phases")
        if any(not isinstance(phase, Phase) for phase in self.phases):
            raise TypeError("phases must contain Phase values")
        object.__setattr__(self, "phases", tuple(self.phases))
        if type(self.max_total_steps) is not int or self.max_total_steps <= 0:
            raise ValueError("max_total_steps must be a positive integer")
        if self.max_total_steps > ACTION_HORIZON - self.activation_step:
            raise ValueError("specialist horizon exceeds remaining playable actions")
        if self.parent_fallback is not FallbackPolicy.RETURN_TO_PARENT:
            raise ValueError("SpecialistProgram requires RETURN_TO_PARENT fallback")
        if not isinstance(self.decisions, (list, tuple)) or len(self.decisions) > MAX_REACTIVE_DECISIONS:
            raise ValueError(f"SpecialistProgram permits at most {MAX_REACTIVE_DECISIONS} decisions")
        if any(not isinstance(decision, DecisionPoint) for decision in self.decisions):
            raise TypeError("decisions must contain DecisionPoint values")
        decisions = tuple(sorted(self.decisions, key=lambda decision: decision.after_phase))
        object.__setattr__(self, "decisions", decisions)
        for decision in decisions:
            _validate_aspe_condition(decision.predicate, "reactive decision")
        if self.recovery_phase is not None and not isinstance(self.recovery_phase, Phase):
            raise TypeError("recovery_phase must be a Phase")
        for phase in (*self.phases, *((self.recovery_phase,) if self.recovery_phase else ())):
            condition = phase.termination.condition
            if condition is not None:
                _validate_aspe_condition(condition, "termination")

        # The existing reactive IR owns all acyclic control-flow, phase-cap,
        # recovery, and fallback validation. Constructing it is the contract.
        self.to_macro_program()

    @property
    def outer_gate(self) -> tuple[tuple[ConditionGene, ...], ...]:
        return tuple(
            tuple(ConditionGene(c.feature, c.comparator, c.index, c.value) for c in clause)
            for clause in self.initiation_clauses
        )

    def to_macro_program(self) -> MacroProgram:
        checkpoint = Checkpoint(
            self.activation_step,
            self.lane,
            (EventType.ROUTE_OR_PHASE_CHANGE,),
            fallback=True,
        )
        return MacroProgram(
            self.lane,
            checkpoint,
            self.phases,
            self.max_total_steps,
            self.parent_fallback,
            schema_version=REACTIVE_IR_SCHEMA_VERSION,
            decisions=self.decisions,
            recovery_phase=self.recovery_phase,
        )

    def to_document(self) -> dict[str, Any]:
        return {
            "schema": SPECIALIST_PROGRAM_SCHEMA,
            "schema_version": self.schema_version,
            "lane": self.lane.value,
            "activation_step": self.activation_step,
            "initiation_clauses": [
                [condition.to_document() for condition in clause]
                for clause in self.initiation_clauses
            ],
            "phases": [phase.to_document() for phase in self.phases],
            "max_total_steps": self.max_total_steps,
            "decisions": [decision.to_document() for decision in self.decisions],
            "recovery_phase": None if self.recovery_phase is None else self.recovery_phase.to_document(),
            "parent_fallback": self.parent_fallback.value,
        }

    def canonical_json(self) -> str:
        return json.dumps(self.to_document(), sort_keys=True, separators=(",", ":"))

    @property
    def semantic_id(self) -> str:
        return stable_identity("ASPE-SPECIALIST-PROGRAM-V1", self.to_document())

    @classmethod
    def from_document(cls, document: Mapping[str, Any]) -> "SpecialistProgram":
        if not isinstance(document, Mapping):
            raise TypeError("SpecialistProgram document must be a mapping")
        _reject_unknown(
            document,
            {
                "schema", "schema_version", "lane", "activation_step",
                "initiation_clauses", "phases", "max_total_steps", "decisions",
                "recovery_phase", "parent_fallback",
            },
            "SpecialistProgram",
        )
        if document.get("schema") != SPECIALIST_PROGRAM_SCHEMA:
            raise ValueError("unsupported SpecialistProgram schema")
        clauses = document.get("initiation_clauses")
        phases = document.get("phases")
        decisions = document.get("decisions", ())
        if not isinstance(clauses, (list, tuple)):
            raise TypeError("initiation_clauses must be a sequence")
        if not isinstance(phases, (list, tuple)):
            raise TypeError("phases must be a sequence")
        if not isinstance(decisions, (list, tuple)):
            raise TypeError("decisions must be a sequence")
        recovery = document.get("recovery_phase")
        try:
            lane = Lane(document.get("lane"))
            fallback = FallbackPolicy(document.get("parent_fallback"))
        except ValueError as exc:
            raise ValueError("invalid SpecialistProgram lane or fallback") from exc
        return cls(
            lane=lane,
            activation_step=document.get("activation_step"),
            initiation_clauses=tuple(
                tuple(PublicCondition.from_document(condition) for condition in clause)
                for clause in clauses
            ),
            phases=tuple(Phase.from_document(phase) for phase in phases),
            max_total_steps=document.get("max_total_steps"),
            decisions=tuple(DecisionPoint.from_document(decision) for decision in decisions),
            recovery_phase=None if recovery is None else Phase.from_document(recovery),
            parent_fallback=fallback,
            schema_version=document.get("schema_version"),
        )

    @classmethod
    def from_json(cls, value: str | bytes) -> "SpecialistProgram":
        return cls.from_document(json.loads(value))
