"""Bounded, open-loop Option-Program intermediate representation v1."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import re
from typing import Any, Mapping, Sequence

from tools.bkg4_search.module_codec import ConditionGene

from .checkpoints import CHECKPOINT_SCHEMA_VERSION, Checkpoint, EventType, Lane
from .schema import FORBIDDEN_RUNTIME_IDENTITY_FIELDS, stable_identity, validate_runtime_identity_safe

IR_SCHEMA_VERSION = 1
ACTION_HORIZON = 719
MAX_PHASES = 4

_SEMANTIC_REF = re.compile(r"[0-9a-f]{64}")
_ENUM_NAME = re.compile(r"[A-Z][A-Z0-9_]*")
_PARAMETER_KEY = re.compile(r"[a-z][a-z0-9_]*")
# V1 completion must be observable without adding an item selector to the IR.
# Land and labor have exact public counters; item purchases are deferred until a
# later schema can name the purchased item without aggregate-inventory ambiguity.
_PURCHASE_OBJECTIVES = frozenset({"BUY_LAND", "HIRE"})


class ActionSource(str, Enum):
    PARENT = "PARENT"
    DONOR_BLOCK = "DONOR_BLOCK"
    SEMANTIC_SPLICE = "SEMANTIC_SPLICE"
    PARAMETERIZED_PRIMITIVE = "PARAMETERIZED_PRIMITIVE"


class ActionFamily(str, Enum):
    FARMER = "FARMER"
    HANDS = "HANDS"
    MARKET = "MARKET"


class TerminationType(str, Enum):
    FIXED_CAP = "FIXED_CAP"
    NEXT_EOD = "NEXT_EOD"
    NEXT_SHOP_REVEAL = "NEXT_SHOP_REVEAL"
    PURCHASE_COMPLETED = "PURCHASE_COMPLETED"
    TARGET_PUBLIC_STATE = "TARGET_PUBLIC_STATE"
    PARENT_COMPATIBLE = "PARENT_COMPATIBLE"
    MAX_STEPS = "MAX_STEPS"


class FallbackPolicy(str, Enum):
    RETURN_TO_PARENT = "RETURN_TO_PARENT"


def _exact_positive(value: Any, field: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _exact_int(value: Any, field: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{field} must be an integer")
    return value


def _enum(enum_type: type[Enum], value: Any, field: str) -> Any:
    if isinstance(value, enum_type):
        return value
    if type(value) is not str:
        raise TypeError(f"{field} must be {enum_type.__name__}")
    try:
        return enum_type(value)
    except ValueError as exc:
        raise ValueError(f"unknown {field}: {value!r}") from exc


def _reject_unknown(document: Mapping[str, Any], allowed: set[str], kind: str) -> None:
    unknown = set(document) - allowed
    if unknown:
        raise ValueError(f"unsupported {kind} fields: {sorted(unknown)}")


def _reject_identity_token(value: str, field: str) -> None:
    normalized = value.casefold().replace("-", "_")
    if normalized in FORBIDDEN_RUNTIME_IDENTITY_FIELDS:
        raise ValueError(f"identity/provenance token forbidden in {field}: {value}")


@dataclass(frozen=True)
class PublicCondition:
    """One deliberately bounded legal-public-state condition."""

    feature: str
    comparator: str
    index: int = 0
    value: int = 0

    def __post_init__(self) -> None:
        if type(self.feature) is not str or type(self.comparator) is not str:
            raise TypeError("condition feature and comparator must be strings")
        _exact_int(self.index, "condition index")
        _exact_int(self.value, "condition value")
        ConditionGene(self.feature, self.comparator, self.index, self.value).validate()

    def to_document(self) -> dict[str, Any]:
        return {
            "feature": self.feature,
            "comparator": self.comparator,
            "index": self.index,
            "value": self.value,
        }

    @classmethod
    def from_document(cls, document: Mapping[str, Any]) -> "PublicCondition":
        if not isinstance(document, Mapping):
            raise TypeError("condition document must be a mapping")
        _reject_unknown(document, {"feature", "comparator", "index", "value"}, "condition")
        return cls(
            document.get("feature"),
            document.get("comparator"),
            document.get("index", 0),
            document.get("value", 0),
        )


@dataclass(frozen=True)
class Termination:
    """A finite phase termination descriptor; ``max_steps`` is always enforced."""

    kind: TerminationType
    max_steps: int
    condition: PublicCondition | None = None
    objective: str | None = None
    target_count: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", _enum(TerminationType, self.kind, "termination type"))
        _exact_positive(self.max_steps, "termination max_steps")
        if self.kind is TerminationType.TARGET_PUBLIC_STATE:
            if not isinstance(self.condition, PublicCondition):
                raise ValueError("TARGET_PUBLIC_STATE requires one PublicCondition")
            if self.objective is not None or self.target_count is not None:
                raise ValueError("TARGET_PUBLIC_STATE does not accept purchase fields")
        elif self.kind is TerminationType.PURCHASE_COMPLETED:
            if self.objective not in _PURCHASE_OBJECTIVES:
                raise ValueError("PURCHASE_COMPLETED requires a supported purchase objective")
            _exact_positive(self.target_count, "purchase target_count")
            if self.condition is not None:
                raise ValueError("PURCHASE_COMPLETED does not accept a public condition")
        elif self.condition is not None or self.objective is not None or self.target_count is not None:
            raise ValueError(f"{self.kind.value} accepts only max_steps")

    def to_document(self) -> dict[str, Any]:
        result: dict[str, Any] = {"kind": self.kind.value, "max_steps": self.max_steps}
        if self.condition is not None:
            result["condition"] = self.condition.to_document()
        if self.objective is not None:
            result["objective"] = self.objective
            result["target_count"] = self.target_count
        return result

    @classmethod
    def from_document(cls, document: Mapping[str, Any]) -> "Termination":
        if not isinstance(document, Mapping):
            raise TypeError("termination document must be a mapping")
        _reject_unknown(document, {"kind", "max_steps", "condition", "objective", "target_count"}, "termination")
        condition_doc = document.get("condition")
        return cls(
            _enum(TerminationType, document.get("kind"), "termination type"),
            document.get("max_steps"),
            PublicCondition.from_document(condition_doc) if condition_doc is not None else None,
            document.get("objective"),
            document.get("target_count"),
        )


def _canonical_parameters(value: Any) -> tuple[tuple[str, int | bool | str], ...]:
    if value is None:
        return ()
    if isinstance(value, Mapping):
        pairs = value.items()
    elif isinstance(value, (list, tuple)):
        pairs = value
    else:
        raise TypeError("parameters must be a flat mapping")
    result: dict[str, int | bool | str] = {}
    for pair in pairs:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise TypeError("parameter entries must be key/value pairs")
        key, item = pair
        if type(key) is not str or not _PARAMETER_KEY.fullmatch(key):
            raise ValueError(f"invalid parameter key: {key!r}")
        if type(item) not in (int, bool, str):
            raise TypeError(f"unsupported parameter value for {key}")
        if type(item) is int and not -(1 << 31) <= item < (1 << 31):
            raise ValueError(f"integer parameter {key} must fit int32")
        if type(item) is str and not _ENUM_NAME.fullmatch(item):
            raise ValueError(f"string parameter {key} must be an enum-like token")
        if type(item) is str:
            _reject_identity_token(item, f"parameter {key}")
        if key in result:
            raise ValueError(f"duplicate parameter key: {key}")
        result[key] = item
    validate_runtime_identity_safe(result)
    return tuple(sorted(result.items()))


@dataclass(frozen=True)
class Phase:
    action_source: ActionSource
    action_families: tuple[ActionFamily, ...]
    termination: Termination
    motif_id: str | None = None
    primitive_name: str | None = None
    parameters: tuple[tuple[str, int | bool | str], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "action_source", _enum(ActionSource, self.action_source, "action source"))
        if not isinstance(self.action_families, (list, tuple)):
            raise TypeError("action_families must be a sequence")
        families = tuple(sorted({_enum(ActionFamily, item, "action family") for item in self.action_families}, key=lambda item: item.value))
        object.__setattr__(self, "action_families", families)
        if not isinstance(self.termination, Termination):
            raise TypeError("termination must be a Termination")
        object.__setattr__(self, "parameters", _canonical_parameters(self.parameters))

        if self.action_source is ActionSource.PARENT:
            if families or self.motif_id is not None or self.primitive_name is not None or self.parameters:
                raise ValueError("PARENT phases override no families and accept no source or parameters")
        else:
            if not families:
                raise ValueError("overriding phases require at least one action family")
        if self.action_source in (ActionSource.DONOR_BLOCK, ActionSource.SEMANTIC_SPLICE):
            if type(self.motif_id) is not str or not _SEMANTIC_REF.fullmatch(self.motif_id):
                raise ValueError(f"{self.action_source.value} requires a semantic motif ID")
            if self.primitive_name is not None:
                raise ValueError("motif-backed phases cannot name a primitive")
        elif self.action_source is ActionSource.PARAMETERIZED_PRIMITIVE:
            if self.motif_id is not None:
                raise ValueError("parameterized primitives cannot reference a motif")
            if type(self.primitive_name) is not str or not _ENUM_NAME.fullmatch(self.primitive_name):
                raise ValueError("parameterized primitives require an enum-like name")
            _reject_identity_token(self.primitive_name, "primitive_name")
        elif self.action_source is not ActionSource.PARENT:
            raise ValueError(f"unsupported action source: {self.action_source}")

    def family_resolution(self) -> dict[str, str]:
        selected = set(self.action_families)
        return {family.value: "PHASE" if family in selected else "PARENT" for family in ActionFamily}

    def to_document(self) -> dict[str, Any]:
        return {
            "action_source": self.action_source.value,
            "action_families": [family.value for family in self.action_families],
            "termination": self.termination.to_document(),
            "motif_id": self.motif_id,
            "primitive_name": self.primitive_name,
            "parameters": {key: value for key, value in self.parameters},
        }

    @classmethod
    def from_document(cls, document: Mapping[str, Any]) -> "Phase":
        if not isinstance(document, Mapping):
            raise TypeError("phase document must be a mapping")
        _reject_unknown(
            document,
            {"action_source", "action_families", "termination", "motif_id", "primitive_name", "parameters"},
            "phase",
        )
        return cls(
            _enum(ActionSource, document.get("action_source"), "action source"),
            tuple(document.get("action_families", ())),
            Termination.from_document(document.get("termination")),
            document.get("motif_id"),
            document.get("primitive_name"),
            document.get("parameters", {}),
        )


def _checkpoint_from_semantic_document(document: Mapping[str, Any]) -> Checkpoint:
    if not isinstance(document, Mapping):
        raise TypeError("initiation checkpoint must be a mapping")
    _reject_unknown(document, {"schema_version", "step", "lane", "reasons", "state_signature", "fallback"}, "checkpoint")
    return Checkpoint(
        step=document.get("step"),
        lane=_enum(Lane, document.get("lane"), "lane"),
        reasons=tuple(_enum(EventType, value, "event reason") for value in document.get("reasons", ())),
        state_signature=document.get("state_signature"),
        fallback=document.get("fallback", False),
        schema_version=document.get("schema_version", CHECKPOINT_SCHEMA_VERSION),
    )


@dataclass(frozen=True)
class MacroProgram:
    """A deterministic 1--4 phase program with unconditional parent fallback."""

    lane: Lane
    initiation: Checkpoint
    phases: tuple[Phase, ...]
    max_total_steps: int
    parent_fallback: FallbackPolicy = FallbackPolicy.RETURN_TO_PARENT
    provenance: tuple[str, ...] = ()
    schema_version: int = IR_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != IR_SCHEMA_VERSION:
            raise ValueError(f"unsupported IR schema version: {self.schema_version!r}")
        object.__setattr__(self, "lane", _enum(Lane, self.lane, "lane"))
        object.__setattr__(self, "parent_fallback", _enum(FallbackPolicy, self.parent_fallback, "parent fallback"))
        if not isinstance(self.initiation, Checkpoint):
            raise TypeError("initiation must be a Phase-2 Checkpoint")
        if self.initiation.lane is not self.lane:
            raise ValueError("program lane must match initiation checkpoint lane")
        if not isinstance(self.phases, (list, tuple)) or not 1 <= len(self.phases) <= MAX_PHASES:
            raise ValueError(f"program requires 1..{MAX_PHASES} phases")
        if any(not isinstance(phase, Phase) for phase in self.phases):
            raise TypeError("phases must contain Phase values")
        object.__setattr__(self, "phases", tuple(self.phases))
        _exact_positive(self.max_total_steps, "max_total_steps")
        remaining = ACTION_HORIZON - self.initiation.step
        if self.max_total_steps > remaining:
            raise ValueError("program horizon exceeds remaining playable action horizon")
        if any(phase.termination.max_steps > self.max_total_steps for phase in self.phases):
            raise ValueError("phase cap cannot exceed program max_total_steps")
        if not isinstance(self.provenance, tuple) or any(type(item) is not str for item in self.provenance):
            raise TypeError("provenance must be a tuple of strings")
        object.__setattr__(self, "provenance", tuple(sorted(set(self.provenance))))

    def to_document(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "lane": self.lane.value,
            "initiation": self.initiation.semantic_document(),
            "phases": [phase.to_document() for phase in self.phases],
            "max_total_steps": self.max_total_steps,
            "parent_fallback": self.parent_fallback.value,
        }

    def to_artifact_document(self) -> dict[str, Any]:
        return {**self.to_document(), "semantic_id": self.semantic_id, "provenance": list(self.provenance)}

    @property
    def semantic_id(self) -> str:
        return stable_identity("RLRMB-MACRO-PROGRAM-V1", self.to_document())

    @classmethod
    def from_document(cls, document: Mapping[str, Any]) -> "MacroProgram":
        if not isinstance(document, Mapping):
            raise TypeError("program document must be a mapping")
        _reject_unknown(
            document,
            {"schema_version", "lane", "initiation", "phases", "max_total_steps", "parent_fallback"},
            "program",
        )
        phases = document.get("phases")
        if not isinstance(phases, (list, tuple)):
            raise TypeError("program phases must be a sequence")
        return cls(
            _enum(Lane, document.get("lane"), "lane"),
            _checkpoint_from_semantic_document(document.get("initiation")),
            tuple(Phase.from_document(item) for item in phases),
            document.get("max_total_steps"),
            _enum(FallbackPolicy, document.get("parent_fallback"), "parent fallback"),
            schema_version=document.get("schema_version"),
        )
