"""Deterministic safety-first effect-vector quality-diversity archive."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping

from .canonical import (
    action_fragment_content_id,
    candidate_semantic_id,
    canonical_action_sequence,
    load_route_actions,
)
from .checkpoints import Checkpoint, EventType, Lane
from .effects import EffectBlock, EffectVector, effect_signature, effect_vectors_from_stage_a_run
from .schema import DatasetRole, canonical_json, stable_identity

ARCHIVE_SCHEMA_VERSION = 1
ARCHIVE_CELL_SCHEMA_VERSION = 1


class TemporalPhase(str, Enum):
    EARLY = "EARLY"
    MID = "MID"
    LATE = "LATE"
    ENDGAME = "ENDGAME"


class InsertResult(str, Enum):
    INSERTED_NEW_CELL = "INSERTED_NEW_CELL"
    REPLACED_ELITE = "REPLACED_ELITE"
    REJECTED_DUPLICATE = "REJECTED_DUPLICATE"
    REJECTED_INFERIOR = "REJECTED_INFERIOR"
    REJECTED_INVALID = "REJECTED_INVALID"


_EVENT_REGIMES = (
    ("LAND", frozenset({EventType.BUY_LAND})),
    ("LABOR", frozenset({EventType.HIRE})),
    ("ANIMAL", frozenset({EventType.ANIMAL_OR_STRUCTURE_TRANSITION})),
    ("MARKET", frozenset({EventType.MARKET_PRICE_OR_INVENTORY_SHIFT, EventType.SHOP_REVEAL})),
    ("PRODUCTION", frozenset({EventType.PRODUCTION_OR_PAYOFF_EVENT})),
    ("DIVERGENCE", frozenset({EventType.FIRST_EFFECT_DIVERGENCE, EventType.POLICY_DIVERGENCE})),
    ("REGRET", frozenset({EventType.REGRET_PEAK})),
    ("BOUNDARY", frozenset({EventType.DAY_BOUNDARY, EventType.EOD, EventType.ROUTE_OR_PHASE_CHANGE})),
)
_MARKET_REGIMES = frozenset({"SCARCE", "BALANCED", "GLUT", "UNKNOWN"})


def temporal_phase_for_step(step: int) -> TemporalPhase:
    """Map the 719-action season into four broad, non-grid temporal regimes."""
    if type(step) is not int or not 0 <= step <= 718:
        raise ValueError("step must be in the playable range 0..718")
    if step < 168:
        return TemporalPhase.EARLY
    if step < 360:
        return TemporalPhase.MID
    if step < 576:
        return TemporalPhase.LATE
    return TemporalPhase.ENDGAME


def _bucket(value: int | None, boundaries: tuple[int, ...], labels: tuple[str, ...]) -> str:
    if value is None:
        return "UNKNOWN"
    if type(value) is not int or value < 0:
        raise ValueError("economic state quantities must be non-negative integers")
    for boundary, label in zip(boundaries, labels):
        if value < boundary:
            return label
    return labels[-1]


def initiation_cluster(checkpoint: Checkpoint, economic_state: Mapping[str, Any] | None = None) -> str:
    """Coarsen only available public initiation state; missing dimensions stay unknown."""
    if not isinstance(checkpoint, Checkpoint):
        raise TypeError("checkpoint must be a Checkpoint")
    state = {} if economic_state is None else dict(economic_state)
    allowed = {"own_money", "unlocked_quadrants", "worker_count", "market_regime"}
    unknown = set(state) - allowed
    if unknown:
        raise ValueError(f"unsupported initiation state fields: {sorted(unknown)!r}")
    reasons = set(checkpoint.reasons)
    event_regime = next((name for name, events in _EVENT_REGIMES if reasons & events), "GENERAL")
    market = state.get("market_regime", "UNKNOWN")
    if type(market) is not str or market not in _MARKET_REGIMES:
        raise ValueError("market_regime must be SCARCE, BALANCED, GLUT, or UNKNOWN")
    document = {
        "version": 1,
        "event_regime": event_regime,
        "capital": _bucket(state.get("own_money"), (1000, 4000, 2**63), ("LOW", "MEDIUM", "HIGH")),
        "land": _bucket(state.get("unlocked_quadrants"), (2, 4, 2**63), ("STARTER", "EXPANDING", "DEVELOPED")),
        "labor": _bucket(state.get("worker_count"), (2, 7, 2**63), ("SOLO", "TEAM", "HIGH_LABOR")),
        "market": market,
    }
    return stable_identity("RLRMB-INITIATION-CLUSTER-V1", document)


def _delta_bucket(value: int) -> str:
    if value <= -1000:
        return "LARGE_NEGATIVE"
    if value < 0:
        return "NEGATIVE"
    if value == 0:
        return "ZERO"
    if value < 1000:
        return "POSITIVE"
    return "LARGE_POSITIVE"


def _action_change_regime(block: EffectBlock) -> str:
    if block.action_changes is None:
        return "UNKNOWN"
    maximum = max(block.action_changes)
    if maximum == 0:
        return "NONE"
    if maximum <= 12:
        return "LOCAL"
    if maximum <= 48:
        return "BOUNDED"
    return "EXTENSIVE"


def effect_cluster_document(effect: EffectVector) -> dict[str, Any]:
    """Describe ordered response shape without reducing it to an aggregate score."""
    if not isinstance(effect, EffectVector):
        raise TypeError("effect must be an EffectVector")
    blocks = []
    for block in effect.blocks:
        if block.role not in {DatasetRole.TARGET, DatasetRole.TRANSFER, DatasetRole.VALIDATION}:
            raise ValueError(f"dataset role {block.role.value} may not define archive clusters")
        blocks.append(
            {
                "role": block.role.value,
                "panel_id": block.panel_id,
                "sign_pattern": list(block.sign_pattern),
                "loss_to_win": list(block.loss_to_win),
                "win_to_loss": list(block.win_to_loss),
                "delta_buckets": [_delta_bucket(value) for value in block.margin_deltas],
                "worst_delta_bucket": _delta_bucket(min(block.margin_deltas)),
                "action_change_regime": _action_change_regime(block),
            }
        )
    if not any(block["role"] == DatasetRole.TARGET.value for block in blocks):
        raise ValueError("effect clustering requires a TARGET block")
    return {"version": 1, "blocks": blocks}


def effect_cluster(effect: EffectVector) -> str:
    return stable_identity("RLRMB-EFFECT-CLUSTER-V1", effect_cluster_document(effect))


@dataclass(frozen=True)
class ArchiveCellKey:
    lane: Lane
    initiation_cluster: str
    effect_cluster: str
    temporal_phase: TemporalPhase
    schema_version: int = ARCHIVE_CELL_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != ARCHIVE_CELL_SCHEMA_VERSION:
            raise ValueError(f"unsupported archive cell schema version: {self.schema_version!r}")
        if not isinstance(self.lane, Lane):
            raise TypeError("lane must use the Phase-2 Lane taxonomy")
        if not isinstance(self.temporal_phase, TemporalPhase):
            raise TypeError("temporal_phase must be a TemporalPhase")
        for name in ("initiation_cluster", "effect_cluster"):
            value = getattr(self, name)
            if type(value) is not str or not value:
                raise ValueError(f"{name} must be a non-empty semantic reference")

    @property
    def semantic_id(self) -> str:
        return stable_identity("RLRMB-ARCHIVE-CELL-V1", self.to_document())

    def to_document(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "lane": self.lane.value,
            "initiation_cluster": self.initiation_cluster,
            "effect_cluster": self.effect_cluster,
            "temporal_phase": self.temporal_phase.value,
        }

    @classmethod
    def from_document(cls, document: Mapping[str, Any]) -> "ArchiveCellKey":
        if not isinstance(document, Mapping):
            raise TypeError("archive cell document must be a mapping")
        try:
            return cls(
                lane=Lane(document["lane"]),
                initiation_cluster=document["initiation_cluster"],
                effect_cluster=document["effect_cluster"],
                temporal_phase=TemporalPhase(document["temporal_phase"]),
                schema_version=document["schema_version"],
            )
        except (KeyError, ValueError) as exc:
            raise ValueError("invalid archive cell document") from exc


def make_cell_key(
    checkpoint: Checkpoint,
    effect: EffectVector,
    economic_state: Mapping[str, Any] | None = None,
) -> ArchiveCellKey:
    return ArchiveCellKey(
        lane=checkpoint.lane,
        initiation_cluster=initiation_cluster(checkpoint, economic_state),
        effect_cluster=effect_cluster(effect),
        temporal_phase=temporal_phase_for_step(checkpoint.step),
    )


@dataclass(frozen=True)
class ArchiveEntry:
    candidate_id: str
    program_id: str | None
    effect_signature: str
    cell_key: ArchiveCellKey
    semantic_valid: bool
    target_win_to_loss: int
    target_harm_count: int
    target_loss_to_win: int
    required_target_positive_support: int
    independent_positive_support: int
    both_seat_support: int
    worst_delta: int
    routability: bool | None
    program_simplicity: int
    aggregate_margin: int
    loss_to_win_pattern: tuple[bool, ...] = ()
    provenance: tuple[str, ...] = ()
    evidence_ids: tuple[str, ...] = ()
    cross_regime_supported: bool = False
    seed_support: int = 0
    seat_support: int = 0
    member_support: int = 0
    lane_support: int = 0
    trajectory_support: int = 0

    def __post_init__(self) -> None:
        for field in ("candidate_id", "effect_signature"):
            if type(getattr(self, field)) is not str or not getattr(self, field):
                raise ValueError(f"{field} must be a non-empty semantic reference")
        if self.program_id is not None and (type(self.program_id) is not str or not self.program_id):
            raise ValueError("program_id must be null or a non-empty semantic reference")
        if not isinstance(self.cell_key, ArchiveCellKey):
            raise TypeError("cell_key must be an ArchiveCellKey")
        if type(self.semantic_valid) is not bool:
            raise TypeError("semantic_valid must be bool")
        for field in (
            "target_win_to_loss", "target_harm_count", "target_loss_to_win", "required_target_positive_support",
            "independent_positive_support", "both_seat_support", "program_simplicity",
            "seed_support", "seat_support", "member_support", "lane_support", "trajectory_support",
        ):
            value = getattr(self, field)
            if type(value) is not int or value < 0:
                raise ValueError(f"{field} must be a non-negative integer")
        if type(self.cross_regime_supported) is not bool:
            raise TypeError("cross_regime_supported must be bool")
        if type(self.worst_delta) is not int or type(self.aggregate_margin) is not int:
            raise TypeError("delta and aggregate values must be exact integers")
        if self.routability is not None and type(self.routability) is not bool:
            raise TypeError("routability must be true, false, or unknown")
        if not isinstance(self.loss_to_win_pattern, tuple) or any(type(v) is not bool for v in self.loss_to_win_pattern):
            raise TypeError("loss_to_win_pattern must be a tuple of bools")
        for field in ("provenance", "evidence_ids"):
            values = getattr(self, field)
            if not isinstance(values, tuple) or any(type(value) is not str for value in values):
                raise TypeError(f"{field} must be a tuple of strings")
            object.__setattr__(self, field, tuple(sorted(set(values))))

    @property
    def structural_id(self) -> str:
        return self.program_id or self.candidate_id

    def rank_key(self) -> tuple[Any, ...]:
        routability_rank = 0 if self.routability is True else 1 if self.routability is None else 2
        return (
            -int(self.semantic_valid),
            -int(self.cross_regime_supported),
            self.target_win_to_loss,
            self.target_harm_count,
            -self.seed_support,
            -self.seat_support,
            -self.member_support,
            -self.lane_support,
            -self.trajectory_support,
            -self.target_loss_to_win,
            -self.required_target_positive_support,
            -self.independent_positive_support,
            -self.both_seat_support,
            -self.worst_delta,
            routability_rank,
            self.program_simplicity,
            -self.aggregate_margin,
            self.candidate_id,
        )

    def semantic_document(self) -> dict[str, Any]:
        document = self.to_document()
        document.pop("provenance")
        document.pop("evidence_ids")
        return document

    def to_document(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "program_id": self.program_id,
            "effect_signature": self.effect_signature,
            "cell_key": self.cell_key.to_document(),
            "semantic_valid": self.semantic_valid,
            "target_win_to_loss": self.target_win_to_loss,
            "target_harm_count": self.target_harm_count,
            "target_loss_to_win": self.target_loss_to_win,
            "required_target_positive_support": self.required_target_positive_support,
            "independent_positive_support": self.independent_positive_support,
            "both_seat_support": self.both_seat_support,
            "worst_delta": self.worst_delta,
            "routability": self.routability,
            "program_simplicity": self.program_simplicity,
            "aggregate_margin": self.aggregate_margin,
            "loss_to_win_pattern": list(self.loss_to_win_pattern),
            "cross_regime_supported": self.cross_regime_supported,
            "seed_support": self.seed_support,
            "seat_support": self.seat_support,
            "member_support": self.member_support,
            "lane_support": self.lane_support,
            "trajectory_support": self.trajectory_support,
            "provenance": list(self.provenance),
            "evidence_ids": list(self.evidence_ids),
        }

    @classmethod
    def from_document(cls, document: Mapping[str, Any]) -> "ArchiveEntry":
        if not isinstance(document, Mapping):
            raise TypeError("archive entry document must be a mapping")
        try:
            return cls(
                candidate_id=document["candidate_id"], program_id=document["program_id"],
                effect_signature=document["effect_signature"],
                cell_key=ArchiveCellKey.from_document(document["cell_key"]),
                semantic_valid=document["semantic_valid"],
                target_win_to_loss=document["target_win_to_loss"],
                target_harm_count=document["target_harm_count"],
                target_loss_to_win=document["target_loss_to_win"],
                required_target_positive_support=document["required_target_positive_support"],
                independent_positive_support=document["independent_positive_support"],
                both_seat_support=document["both_seat_support"], worst_delta=document["worst_delta"],
                routability=document["routability"], program_simplicity=document["program_simplicity"],
                aggregate_margin=document["aggregate_margin"],
                loss_to_win_pattern=tuple(document.get("loss_to_win_pattern", ())),
                provenance=tuple(document.get("provenance", ())),
                evidence_ids=tuple(document.get("evidence_ids", ())),
                cross_regime_supported=document.get("cross_regime_supported", False),
                seed_support=document.get("seed_support", 0),
                seat_support=document.get("seat_support", 0),
                member_support=document.get("member_support", 0),
                lane_support=document.get("lane_support", 0),
                trajectory_support=document.get("trajectory_support", 0),
            )
        except KeyError as exc:
            raise ValueError(f"archive entry missing field: {exc.args[0]}") from exc


_RANK_REASONS = (
    "SEMANTIC_VALIDITY", "CROSS_REGIME_SUPPORT", "W_TO_L_SAFETY", "TARGET_HARM",
    "SEED_SUPPORT", "SEAT_SUPPORT", "MEMBER_SUPPORT", "LANE_SUPPORT", "TRAJECTORY_SUPPORT",
    "TARGET_L_TO_W", "TARGET_SUPPORT",
    "INDEPENDENT_SUPPORT", "BOTH_SEAT_SUPPORT", "WORST_DELTA", "ROUTABILITY",
    "SIMPLICITY", "AGGREGATE_MARGIN", "SEMANTIC_ID",
)


def compare_entries(first: ArchiveEntry, second: ArchiveEntry) -> int:
    """Return -1 when first is superior, 1 when second is superior, else 0."""
    return (first.rank_key() > second.rank_key()) - (first.rank_key() < second.rank_key())


def _replacement_reason(winner: ArchiveEntry, loser: ArchiveEntry) -> str:
    for reason, left, right in zip(_RANK_REASONS, winner.rank_key(), loser.rank_key()):
        if left != right:
            return reason
    return "EXACT_TIE"


class Archive:
    """One-elite-per-cell archive with deterministic incremental cell updates."""

    def __init__(self, entries: Iterable[ArchiveEntry] = ()) -> None:
        self._seen: dict[str, ArchiveEntry] = {}
        self._invalid_rejections = 0
        self._repeat_rejections = 0
        self._considered = 0
        self._elites: dict[str, ArchiveEntry] = {}
        self._semantic_cells: dict[tuple[str, str], str] = {}
        for entry in entries:
            self.insert(entry)

    @staticmethod
    def _fingerprint(entry: ArchiveEntry) -> str:
        return stable_identity("RLRMB-ARCHIVE-ENTRY-V1", entry.semantic_document())

    def _rebuild(self) -> None:
        """Canonical recovery path used by deserialization, not normal inserts."""
        elites: dict[str, ArchiveEntry] = {}
        semantic_cells: dict[tuple[str, str], str] = {}
        for entry in sorted(self._seen.values(), key=self._fingerprint):
            cell_id = entry.cell_key.semantic_id
            semantic_key = (entry.structural_id, entry.effect_signature)
            claimed = semantic_cells.get(semantic_key)
            if claimed is not None and claimed != cell_id:
                raise ValueError("identical structure/effect semantics cannot claim multiple archive cells")
            semantic_cells[semantic_key] = cell_id
            incumbent = elites.get(cell_id)
            if incumbent is None or compare_entries(entry, incumbent) < 0:
                elites[cell_id] = entry
        self._elites = elites
        self._semantic_cells = semantic_cells

    def insert(self, entry: ArchiveEntry) -> InsertResult:
        if not isinstance(entry, ArchiveEntry):
            raise TypeError("archive accepts ArchiveEntry values")
        self._considered += 1
        if not entry.semantic_valid:
            self._invalid_rejections += 1
            return InsertResult.REJECTED_INVALID
        cell_id = entry.cell_key.semantic_id
        semantic_key = (entry.structural_id, entry.effect_signature)
        claimed = self._semantic_cells.get(semantic_key)
        if claimed is not None and claimed != cell_id:
            raise ValueError("identical structure/effect semantics cannot claim multiple archive cells")
        fingerprint = self._fingerprint(entry)
        prior_same = self._seen.get(fingerprint)
        if prior_same is not None:
            merged = replace(
                prior_same,
                provenance=tuple(sorted(set(prior_same.provenance) | set(entry.provenance))),
                evidence_ids=tuple(sorted(set(prior_same.evidence_ids) | set(entry.evidence_ids))),
            )
            self._seen[fingerprint] = merged
            incumbent = self._elites.get(cell_id)
            if incumbent is not None and self._fingerprint(incumbent) == fingerprint:
                self._elites[cell_id] = merged
            self._repeat_rejections += 1
            return InsertResult.REJECTED_DUPLICATE
        incumbent = self._elites.get(cell_id)
        self._seen[fingerprint] = entry
        self._semantic_cells[semantic_key] = cell_id
        if incumbent is None:
            self._elites[cell_id] = entry
            return InsertResult.INSERTED_NEW_CELL
        if compare_entries(entry, incumbent) < 0:
            self._elites[cell_id] = entry
            return InsertResult.REPLACED_ELITE
        if incumbent.structural_id == entry.structural_id or incumbent.effect_signature == entry.effect_signature:
            return InsertResult.REJECTED_DUPLICATE
        return InsertResult.REJECTED_INFERIOR

    def lookup(self, cell: ArchiveCellKey) -> ArchiveEntry | None:
        return self._elites.get(cell.semantic_id)

    def entries(self) -> tuple[ArchiveEntry, ...]:
        return tuple(sorted(self._elites.values(), key=lambda item: (item.cell_key.semantic_id, item.candidate_id)))

    def _canonical_simulation(self) -> dict[str, Any]:
        elites: dict[str, ArchiveEntry] = {}
        inserted = replacements = duplicates = inferior = 0
        reasons: dict[str, int] = {}
        for entry in sorted(self._seen.values(), key=self._fingerprint):
            cell_id = entry.cell_key.semantic_id
            incumbent = elites.get(cell_id)
            if incumbent is None:
                elites[cell_id] = entry
                inserted += 1
            elif compare_entries(entry, incumbent) < 0:
                replacements += 1
                reason = _replacement_reason(entry, incumbent)
                reasons[reason] = reasons.get(reason, 0) + 1
                elites[cell_id] = entry
            elif incumbent.structural_id == entry.structural_id or incumbent.effect_signature == entry.effect_signature:
                duplicates += 1
            else:
                inferior += 1
        return {"inserted": inserted, "replacements": replacements,
                "duplicate_rejections": duplicates + self._repeat_rejections,
                "inferior_rejections": inferior, "replacement_reasons": dict(sorted(reasons.items()))}

    def metrics(self) -> dict[str, Any]:
        simulated = self._canonical_simulation()
        lane_counts = {lane.value: 0 for lane in Lane}
        for entry in self._elites.values():
            lane_counts[entry.cell_key.lane.value] += 1
        return {
            "candidates_considered": self._considered,
            **simulated,
            "invalid_rejections": self._invalid_rejections,
            "occupied_cells": len(self._elites),
            "novelty_rate": 0.0 if self._considered == 0 else len(self._elites) / self._considered,
            "unique_effect_signatures": len({entry.effect_signature for entry in self._seen.values()}),
            "unique_structural_ids": len({entry.structural_id for entry in self._seen.values()}),
            "occupancy_by_lane": lane_counts,
            "target_solving_cells": sum(entry.target_loss_to_win > 0 for entry in self._elites.values()),
            "target_l_to_w_candidates": sum(entry.target_loss_to_win > 0 for entry in self._seen.values()),
            "distinct_retained_l_to_w_patterns": len({
                entry.loss_to_win_pattern for entry in self._elites.values()
                if any(entry.loss_to_win_pattern)
            }),
        }

    def serialize(self) -> str:
        document = {
            "schema_version": ARCHIVE_SCHEMA_VERSION,
            "considered": self._considered,
            "invalid_rejections": self._invalid_rejections,
            "repeat_rejections": self._repeat_rejections,
            "entries": [entry.to_document() for entry in sorted(self._seen.values(), key=self._fingerprint)],
        }
        return canonical_json(document)

    @classmethod
    def deserialize(cls, payload: str) -> "Archive":
        if type(payload) is not str:
            raise TypeError("archive serialization must be JSON text")
        document = json.loads(payload)
        if type(document.get("schema_version")) is not int or document["schema_version"] != ARCHIVE_SCHEMA_VERSION:
            raise ValueError(f"unsupported archive schema version: {document.get('schema_version')!r}")
        archive = cls()
        entries = document.get("entries")
        if not isinstance(entries, list):
            raise ValueError("archive entries must be a list")
        archive._seen = {archive._fingerprint(entry): entry for entry in map(ArchiveEntry.from_document, entries)}
        archive._considered = document.get("considered")
        archive._invalid_rejections = document.get("invalid_rejections")
        archive._repeat_rejections = document.get("repeat_rejections")
        for field in ("_considered", "_invalid_rejections", "_repeat_rejections"):
            if type(getattr(archive, field)) is not int or getattr(archive, field) < 0:
                raise ValueError("archive counters must be non-negative integers")
        archive._rebuild()
        return archive


def _historical_fragment_lane(actions: list[Any], start: int, duration: int) -> tuple[Lane, EventType]:
    """Infer a broad lane from selected actions, never from route identity."""
    normalized = canonical_action_sequence(actions[start : start + duration])
    unit_ops: set[str] = set()
    market_ops: set[str] = set()
    for action in normalized:
        for unit in (action["farmer"], *action["hands"]):
            if unit:
                unit_ops.add(unit[0])
        market_ops.update(order[0] for order in action["market"] if order)
    if "BUY_LAND" in market_ops:
        return Lane.LAND_INVESTMENT, EventType.BUY_LAND
    if market_ops & {"BUY_ANIMAL"} or unit_ops & {"BUILD_COOP", "BUILD_PASTURE", "CARE", "FEED"}:
        return Lane.ANIMAL_TRANSITION, EventType.ANIMAL_OR_STRUCTURE_TRANSITION
    if "HIRE" in market_ops or unit_ops & {"HARVEST", "WATER", "FERTILIZE"}:
        return Lane.WORKER_PRODUCTION, EventType.PRODUCTION_OR_PAYOFF_EVENT
    if market_ops & {"BUY_SEED"} or unit_ops & {"PLANT", "DIG"}:
        return Lane.CROP_TRANSITION, EventType.ROUTE_OR_PHASE_CHANGE
    if market_ops & {"SELL", "BUY_PRODUCT"}:
        return Lane.MARKET_LIQUIDITY, EventType.MARKET_PRICE_OR_INVENTORY_SHIFT
    return Lane.GENERAL, EventType.REGRET_PEAK


def archive_from_stage_a_run(run_path: str | Path) -> tuple[Archive, dict[str, Any]]:
    """Read-only P0.7 adapter used to validate archive behavior on frozen evidence."""
    path = Path(run_path)
    document = json.loads(path.read_text(encoding="utf-8"))
    attempts = document.get("candidate_attempts")
    if not isinstance(attempts, list):
        raise ValueError("Stage-A run is missing candidate_attempts")
    vectors = effect_vectors_from_stage_a_run(path, include_transfer=True)
    if len(vectors) != len(attempts):
        raise ValueError("Stage-A attempt/effect count mismatch")
    archive = Archive()
    route_cache: dict[str, list[Any]] = {}
    structural_ids: set[str] = set()
    effect_ids: set[str] = set()
    for index, (attempt, effect) in enumerate(zip(attempts, vectors), start=1):
        candidate = attempt.get("candidate")
        transfer = attempt.get("transfer")
        if not isinstance(candidate, Mapping) or not isinstance(transfer, Mapping):
            raise ValueError("malformed Stage-A attempt")
        donor_path = candidate.get("donor_path")
        start = candidate.get("start")
        duration = candidate.get("duration")
        if type(donor_path) is not str or type(start) is not int or type(duration) is not int:
            raise ValueError("Stage-A candidate lacks donor window semantics")
        actions = route_cache.setdefault(donor_path, load_route_actions(donor_path))
        fragment_id = action_fragment_content_id(actions, start, duration)
        structural_id = candidate_semantic_id(
            {"start": start, "duration": duration, "fragment_content_id": fragment_id}
        )
        structural_ids.add(structural_id)
        signature = effect_signature(effect)
        effect_ids.add(signature)
        lane, reason = _historical_fragment_lane(actions, start, duration)
        checkpoint = Checkpoint(start, lane, (reason,))
        target_block = next(block for block in effect.blocks if block.role is DatasetRole.TARGET)
        archive.insert(
            ArchiveEntry(
                candidate_id=structural_id,
                program_id=None,
                effect_signature=signature,
                cell_key=make_cell_key(checkpoint, effect),
                semantic_valid=True,
                target_win_to_loss=candidate["win_to_loss"],
                target_harm_count=candidate["harmful"],
                target_loss_to_win=candidate["loss_to_win"],
                required_target_positive_support=candidate["positive"],
                independent_positive_support=transfer["independent_positive_support"],
                both_seat_support=min(
                    candidate["both_seat_positive_support"], transfer["both_seat_positive_support"]
                ),
                worst_delta=min(candidate["worst_delta"], transfer["worst_delta"]),
                routability=None,
                program_simplicity=duration,
                aggregate_margin=candidate["sum_delta"] + transfer["sum_delta"],
                loss_to_win_pattern=target_block.loss_to_win,
                provenance=(donor_path, f"attempt-{index:02d}"),
                evidence_ids=(f"stage-a-slot-{index:02d}",),
            )
        )
    metrics = archive.metrics()
    retained = archive.entries()
    report = {
        "attempts": len(attempts),
        "structurally_unique_candidates": len(structural_ids),
        "exact_effect_signatures": len(effect_ids),
        "exact_effect_duplicates": len(attempts) - len(effect_ids),
        **metrics,
        "lower_aggregate_distinct_elites": sum(
            entry.aggregate_margin < max(item.aggregate_margin for item in retained)
            for entry in retained
        ) if retained else 0,
    }
    return archive, report
