"""Exact, panel-aware strategic effect canonicalization."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .schema import DatasetRole, stable_identity

EFFECT_SCHEMA_VERSION = 1


def _integer_tuple(values: tuple[int, ...], field: str) -> tuple[int, ...]:
    if not isinstance(values, tuple):
        raise TypeError(f"{field} must be a tuple")
    if any(type(value) is not int for value in values):
        raise TypeError(f"{field} must contain exact integers")
    return values


@dataclass(frozen=True)
class EffectBlock:
    """Exact ordered effects for one explicitly identified dataset panel."""

    role: DatasetRole
    panel_id: str
    baseline_margins: tuple[int, ...]
    margin_deltas: tuple[int, ...]
    action_changes: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.role, DatasetRole):
            raise TypeError("role must be a DatasetRole")
        if type(self.panel_id) is not str or not self.panel_id.strip():
            raise ValueError("panel_id must be a non-empty semantic panel identifier")
        baseline = _integer_tuple(self.baseline_margins, "baseline_margins")
        deltas = _integer_tuple(self.margin_deltas, "margin_deltas")
        if not baseline or len(baseline) != len(deltas):
            raise ValueError("baseline and delta arrays must have the same non-zero length")
        if self.action_changes is not None:
            changes = _integer_tuple(self.action_changes, "action_changes")
            if len(changes) != len(deltas):
                raise ValueError("action_changes length must match effect arrays")

    @property
    def candidate_margins(self) -> tuple[int, ...]:
        return tuple(baseline + delta for baseline, delta in zip(self.baseline_margins, self.margin_deltas))

    @property
    def sign_pattern(self) -> tuple[int, ...]:
        return tuple((delta > 0) - (delta < 0) for delta in self.margin_deltas)

    @property
    def loss_to_win(self) -> tuple[bool, ...]:
        return tuple(
            baseline < 0 < candidate
            for baseline, candidate in zip(self.baseline_margins, self.candidate_margins)
        )

    @property
    def win_to_loss(self) -> tuple[bool, ...]:
        return tuple(
            baseline > 0 > candidate
            for baseline, candidate in zip(self.baseline_margins, self.candidate_margins)
        )

    def to_document(self) -> dict[str, Any]:
        return {
            "role": self.role.value,
            "panel_id": self.panel_id,
            "baseline_margins": list(self.baseline_margins),
            "margin_deltas": list(self.margin_deltas),
            "sign_pattern": list(self.sign_pattern),
            "loss_to_win": list(self.loss_to_win),
            "win_to_loss": list(self.win_to_loss),
            "action_changes": None if self.action_changes is None else list(self.action_changes),
        }


@dataclass(frozen=True)
class EffectVector:
    """Versioned collection of role-separated exact effect blocks."""

    blocks: tuple[EffectBlock, ...]
    schema_version: int = EFFECT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != EFFECT_SCHEMA_VERSION:
            raise ValueError(f"unsupported effect schema version: {self.schema_version!r}")
        if not isinstance(self.blocks, tuple) or not self.blocks:
            raise ValueError("EffectVector requires at least one block")
        if any(not isinstance(block, EffectBlock) for block in self.blocks):
            raise TypeError("blocks must contain EffectBlock values")
        roles = [block.role for block in self.blocks]
        if len(set(roles)) != len(roles):
            raise ValueError("EffectVector may contain only one block per dataset role")
        object.__setattr__(self, "blocks", tuple(sorted(self.blocks, key=lambda block: block.role.value)))

    def to_document(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "blocks": [block.to_document() for block in self.blocks],
        }


def effect_signature(effect_vector: EffectVector) -> str:
    if not isinstance(effect_vector, EffectVector):
        raise TypeError("effect_signature requires an EffectVector")
    return stable_identity("RLRMB-EFFECT-V1", effect_vector.to_document())


def _historical_panel_id(role: DatasetRole, baseline_margins: tuple[int, ...]) -> str:
    """Identify the ordered Stage-A panel without case/opponent provenance."""
    ordered_slots = [f"slot-{index:04d}" for index in range(len(baseline_margins))]
    return stable_identity(
        "RLRMB-STAGE-A-PANEL-V1",
        role.value,
        ordered_slots,
        baseline_margins,
    )


def _block_from_stage_a_evidence(role: DatasetRole, evidence: Mapping[str, Any]) -> EffectBlock:
    baseline = tuple(evidence["baseline_margins"])
    deltas = tuple(evidence["deltas"])
    changes_value = evidence.get("action_changes")
    changes = None if changes_value is None else tuple(changes_value)
    return EffectBlock(role, _historical_panel_id(role, baseline), baseline, deltas, changes)


def effect_vectors_from_stage_a_run(
    run_path: str | Path,
    *,
    include_transfer: bool,
) -> tuple[EffectVector, ...]:
    """Adapt the frozen P0.7 format without making it a core runtime dependency."""
    document = json.loads(Path(run_path).read_text(encoding="utf-8"))
    attempts = document.get("candidate_attempts")
    if not isinstance(attempts, list):
        raise ValueError("Stage-A run is missing candidate_attempts")
    vectors = []
    for attempt in attempts:
        if not isinstance(attempt, Mapping) or not isinstance(attempt.get("candidate"), Mapping):
            raise ValueError("malformed Stage-A candidate attempt")
        blocks = [_block_from_stage_a_evidence(DatasetRole.TARGET, attempt["candidate"])]
        if include_transfer:
            transfer = attempt.get("transfer")
            if not isinstance(transfer, Mapping):
                raise ValueError("Stage-A attempt is missing transfer evidence")
            blocks.append(_block_from_stage_a_evidence(DatasetRole.TRANSFER, transfer))
        vectors.append(EffectVector(tuple(blocks)))
    return tuple(vectors)


def analyze_stage_a_effect_diversity(run_path: str | Path) -> dict[str, int]:
    target = effect_vectors_from_stage_a_run(run_path, include_transfer=False)
    combined = effect_vectors_from_stage_a_run(run_path, include_transfer=True)
    return {
        "attempts": len(target),
        "unique_target_effect_signatures": len({effect_signature(vector) for vector in target}),
        "unique_target_transfer_effect_signatures": len({effect_signature(vector) for vector in combined}),
    }
