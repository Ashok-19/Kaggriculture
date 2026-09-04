"""Semantic action-block motifs for Option-Program sources."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .canonical import canonical_action_sequence, load_route_actions
from .ir import ActionFamily
from .schema import stable_identity, validate_runtime_identity_safe

MOTIF_SCHEMA_VERSION = 1


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return ("mapping", tuple((key, _freeze(item)) for key, item in sorted(value.items())))
    if isinstance(value, (list, tuple)):
        return ("sequence", tuple(_freeze(item) for item in value))
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, tuple) and len(value) == 2:
        tag, payload = value
        if tag == "mapping":
            return {key: _thaw(item) for key, item in payload}
        if tag == "sequence":
            return [_thaw(item) for item in payload]
    return value


def _families(values: Sequence[ActionFamily]) -> tuple[ActionFamily, ...]:
    if not isinstance(values, (list, tuple)) or not values:
        raise ValueError("motif requires at least one supported action family")
    if any(not isinstance(value, ActionFamily) for value in values):
        raise TypeError("supported families must contain ActionFamily values")
    return tuple(sorted(set(values), key=lambda value: value.value))


@dataclass(frozen=True)
class Motif:
    """Normalized behavior identity plus separately retained offline provenance."""

    action_block: tuple[Any, ...]
    supported_families: tuple[ActionFamily, ...]
    source_start: int = 0
    provenance: tuple[str, ...] = ()
    schema_version: int = MOTIF_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != MOTIF_SCHEMA_VERSION:
            raise ValueError(f"unsupported motif schema version: {self.schema_version!r}")
        families = _families(self.supported_families)
        object.__setattr__(self, "supported_families", families)
        if type(self.source_start) is not int or self.source_start < 0:
            raise ValueError("source_start must be a non-negative integer")
        if not isinstance(self.action_block, (list, tuple)) or not self.action_block:
            raise ValueError("motif action block must be non-empty")
        validate_runtime_identity_safe(self.action_block, "motif.action_block")
        normalized = canonical_action_sequence(self.action_block)
        keys = {family.value.lower() for family in families}
        projected = [{key: action[key] for key in sorted(keys)} for action in normalized]
        object.__setattr__(self, "action_block", tuple(_freeze(action) for action in projected))
        if not isinstance(self.provenance, tuple) or any(type(item) is not str for item in self.provenance):
            raise TypeError("motif provenance must be a tuple of strings")
        object.__setattr__(self, "provenance", tuple(sorted(set(self.provenance))))

    @property
    def source_length(self) -> int:
        return len(self.action_block)

    def semantic_document(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "supported_families": [family.value for family in self.supported_families],
            "action_block": [_thaw(action) for action in self.action_block],
        }

    @property
    def semantic_id(self) -> str:
        return stable_identity("RLRMB-MOTIF-V1", self.semantic_document())

    def to_document(self) -> dict[str, Any]:
        return {
            **self.semantic_document(),
            "semantic_id": self.semantic_id,
            "source_start": self.source_start,
            "source_length": self.source_length,
            "provenance": list(self.provenance),
        }


def motif_from_actions(
    actions: Sequence[Any],
    supported_families: Sequence[ActionFamily],
    *,
    source_start: int = 0,
    provenance: tuple[str, ...] = (),
) -> Motif:
    return Motif(tuple(actions), tuple(supported_families), source_start, provenance)


def motif_from_route_window(
    path: str | Path,
    start: int,
    duration: int,
    supported_families: Sequence[ActionFamily],
    *,
    provenance: tuple[str, ...] = (),
) -> Motif:
    if type(start) is not int or type(duration) is not int or start < 0 or duration <= 0:
        raise ValueError("route motif window requires non-negative start and positive duration")
    actions = load_route_actions(path)
    if start + duration > len(actions):
        raise ValueError("route motif window is outside the action sequence")
    audit = tuple(provenance) + (f"source_path={Path(path)}",)
    return motif_from_actions(
        actions[start : start + duration],
        supported_families,
        source_start=start,
        provenance=audit,
    )
