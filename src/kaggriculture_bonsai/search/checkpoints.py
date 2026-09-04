"""Deterministic regret/event-localized checkpoint selection."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable, Mapping, Sequence

from .schema import stable_identity

CHECKPOINT_SCHEMA_VERSION = 1
DEFAULT_MAX_STEP = 718
DEFAULT_FALLBACK_STEPS = (0, 72, 144, 216, 288, 432, 576)


class EventType(str, Enum):
    REGRET_PEAK = "REGRET_PEAK"
    POLICY_DIVERGENCE = "POLICY_DIVERGENCE"
    SHOP_REVEAL = "SHOP_REVEAL"
    DAY_BOUNDARY = "DAY_BOUNDARY"
    EOD = "EOD"
    BUY_LAND = "BUY_LAND"
    HIRE = "HIRE"
    ANIMAL_OR_STRUCTURE_TRANSITION = "ANIMAL_OR_STRUCTURE_TRANSITION"
    MARKET_PRICE_OR_INVENTORY_SHIFT = "MARKET_PRICE_OR_INVENTORY_SHIFT"
    PRODUCTION_OR_PAYOFF_EVENT = "PRODUCTION_OR_PAYOFF_EVENT"
    FIRST_EFFECT_DIVERGENCE = "FIRST_EFFECT_DIVERGENCE"
    ROUTE_OR_PHASE_CHANGE = "ROUTE_OR_PHASE_CHANGE"
    FALLBACK_GRID = "FALLBACK_GRID"


class Lane(str, Enum):
    MARKET_LIQUIDITY = "MARKET_LIQUIDITY"
    WORKER_PRODUCTION = "WORKER_PRODUCTION"
    LAND_INVESTMENT = "LAND_INVESTMENT"
    CROP_TRANSITION = "CROP_TRANSITION"
    ANIMAL_TRANSITION = "ANIMAL_TRANSITION"
    ENDGAME_LIQUIDATION = "ENDGAME_LIQUIDATION"
    MECHANIC_EXPLOIT = "MECHANIC_EXPLOIT"
    GENERAL = "GENERAL"


_LANE_REASON_PRIORITY = (
    (EventType.ANIMAL_OR_STRUCTURE_TRANSITION, Lane.ANIMAL_TRANSITION),
    (EventType.BUY_LAND, Lane.LAND_INVESTMENT),
    (EventType.HIRE, Lane.WORKER_PRODUCTION),
    (EventType.PRODUCTION_OR_PAYOFF_EVENT, Lane.WORKER_PRODUCTION),
    (EventType.MARKET_PRICE_OR_INVENTORY_SHIFT, Lane.MARKET_LIQUIDITY),
)
_CAUSAL_WEIGHTS = {
    EventType.FIRST_EFFECT_DIVERGENCE: 8,
    EventType.REGRET_PEAK: 7,
    EventType.POLICY_DIVERGENCE: 6,
    EventType.SHOP_REVEAL: 6,
    EventType.BUY_LAND: 5,
    EventType.HIRE: 5,
    EventType.ANIMAL_OR_STRUCTURE_TRANSITION: 5,
    EventType.PRODUCTION_OR_PAYOFF_EVENT: 4,
    EventType.MARKET_PRICE_OR_INVENTORY_SHIFT: 4,
    EventType.ROUTE_OR_PHASE_CHANGE: 4,
    EventType.DAY_BOUNDARY: 3,
    EventType.EOD: 3,
    EventType.FALLBACK_GRID: 0,
}
_NEARBY_EVENT_TYPES = frozenset(
    {
        EventType.SHOP_REVEAL,
        EventType.DAY_BOUNDARY,
        EventType.EOD,
        EventType.MARKET_PRICE_OR_INVENTORY_SHIFT,
        EventType.PRODUCTION_OR_PAYOFF_EVENT,
        EventType.ROUTE_OR_PHASE_CHANGE,
    }
)


def _exact_nonnegative(value: Any, field: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


def _sorted_reasons(reasons: Sequence[EventType]) -> tuple[EventType, ...]:
    if not isinstance(reasons, (list, tuple)) or not reasons:
        raise ValueError("at least one event reason is required")
    if any(not isinstance(reason, EventType) for reason in reasons):
        raise TypeError("reasons must contain EventType values")
    return tuple(sorted(set(reasons), key=lambda reason: reason.value))


@dataclass(frozen=True)
class EventEvidence:
    """Offline causal observation used to propose search checkpoints."""

    step: int
    reasons: tuple[EventType, ...]
    family_ref: str | None = None
    state_signature: str | None = None
    severity: int = 0
    regret: int = 0
    recurrence_support: int = 1
    both_seat_support: int = 0
    recoverable: bool = True
    lane_hint: Lane | None = None
    provenance: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _exact_nonnegative(self.step, "step")
        object.__setattr__(self, "reasons", _sorted_reasons(self.reasons))
        _exact_nonnegative(self.severity, "severity")
        _exact_nonnegative(self.regret, "regret")
        _exact_nonnegative(self.recurrence_support, "recurrence_support")
        _exact_nonnegative(self.both_seat_support, "both_seat_support")
        if type(self.recoverable) is not bool:
            raise TypeError("recoverable must be bool")
        if self.lane_hint is not None and not isinstance(self.lane_hint, Lane):
            raise TypeError("lane_hint must be a Lane")
        if not isinstance(self.provenance, tuple):
            raise TypeError("provenance must be a tuple")
        if any(type(item) is not str for item in self.provenance):
            raise TypeError("provenance references must be strings")
        object.__setattr__(self, "provenance", tuple(sorted(set(self.provenance))))


@dataclass(frozen=True)
class Checkpoint:
    """Immutable versioned search-control checkpoint; never a runtime predicate."""

    step: int
    lane: Lane
    reasons: tuple[EventType, ...]
    state_signature: str | None = None
    severity: int = 0
    regret: int = 0
    recurrence_support: int = 1
    both_seat_support: int = 0
    recoverable: bool = True
    event_distance: int = 0
    fallback: bool = False
    family_refs: tuple[str, ...] = ()
    provenance: tuple[str, ...] = ()
    schema_version: int = CHECKPOINT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != CHECKPOINT_SCHEMA_VERSION:
            raise ValueError(f"unsupported checkpoint schema version: {self.schema_version!r}")
        _exact_nonnegative(self.step, "step")
        if not isinstance(self.lane, Lane):
            raise TypeError("lane must be a Lane")
        object.__setattr__(self, "reasons", _sorted_reasons(self.reasons))
        for field in ("severity", "regret", "recurrence_support", "both_seat_support", "event_distance"):
            _exact_nonnegative(getattr(self, field), field)
        if type(self.recoverable) is not bool or type(self.fallback) is not bool:
            raise TypeError("recoverable and fallback must be bool")
        if any(type(item) is not str for item in (*self.family_refs, *self.provenance)):
            raise TypeError("family/provenance references must be strings")
        object.__setattr__(self, "family_refs", tuple(sorted(set(self.family_refs))))
        object.__setattr__(self, "provenance", tuple(sorted(set(self.provenance))))

    def semantic_document(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "step": self.step,
            "lane": self.lane.value,
            "reasons": [reason.value for reason in self.reasons],
            "state_signature": self.state_signature,
            "fallback": self.fallback,
        }

    @property
    def semantic_id(self) -> str:
        return stable_identity("RLRMB-CHECKPOINT-V1", self.semantic_document())

    def rank_key(self) -> tuple[Any, ...]:
        causal = not self.fallback
        causal_weight = max(_CAUSAL_WEIGHTS[reason] for reason in self.reasons)
        return (
            -int(causal),
            -self.recurrence_support,
            -self.both_seat_support,
            -int(self.recoverable),
            -self.regret,
            -self.severity,
            -causal_weight,
            self.event_distance,
            self.step,
            self.semantic_id,
        )

    def to_document(self) -> dict[str, Any]:
        return {
            **self.semantic_document(),
            "semantic_id": self.semantic_id,
            "severity": self.severity,
            "regret": self.regret,
            "recurrence_support": self.recurrence_support,
            "both_seat_support": self.both_seat_support,
            "recoverable": self.recoverable,
            "event_distance": self.event_distance,
            "family_refs": list(self.family_refs),
            "provenance": list(self.provenance),
        }


def assign_lane(
    reasons: Sequence[EventType],
    *,
    hint: Lane | None = None,
    step: int = 0,
    max_step: int = DEFAULT_MAX_STEP,
) -> Lane:
    if hint is not None:
        return hint
    reason_set = set(reasons)
    if step >= max_step - 72 and reason_set & {
        EventType.MARKET_PRICE_OR_INVENTORY_SHIFT,
        EventType.PRODUCTION_OR_PAYOFF_EVENT,
    }:
        return Lane.ENDGAME_LIQUIDATION
    for reason, lane in _LANE_REASON_PRIORITY:
        if reason in reason_set:
            return lane
    return Lane.GENERAL


def _action_ops(action: Any) -> tuple[set[str], set[str]]:
    unit_ops: set[str] = set()
    market_ops: set[str] = set()
    if not isinstance(action, Mapping):
        return unit_ops, market_ops
    farmer = action.get("farmer")
    if isinstance(farmer, list) and farmer and isinstance(farmer[0], str):
        unit_ops.add(farmer[0])
    hands = action.get("hands")
    if isinstance(hands, list):
        for hand in hands:
            if isinstance(hand, list) and hand and isinstance(hand[0], str):
                unit_ops.add(hand[0])
    market = action.get("market")
    if isinstance(market, list):
        for order in market:
            if isinstance(order, list) and order and isinstance(order[0], str):
                market_ops.add(order[0])
    return unit_ops, market_ops


def extract_trace_events(records: Iterable[Mapping[str, Any]], *, turns_per_day: int = 24) -> tuple[EventEvidence, ...]:
    """Extract only explicit public/action causal events from ordered trace records."""
    if type(turns_per_day) is not int or turns_per_day <= 0:
        raise ValueError("turns_per_day must be a positive integer")
    events = []
    for record in records:
        step = record.get("step")
        if type(step) is not int or step < 0:
            raise ValueError("trace record step must be a non-negative integer")
        reasons: set[EventType] = set()
        if step > 0 and step % turns_per_day == 0:
            reasons.add(EventType.DAY_BOUNDARY)
        flags = {
            "regret_peak": EventType.REGRET_PEAK,
            "policy_diverged": EventType.POLICY_DIVERGENCE,
            "shop_revealed": EventType.SHOP_REVEAL,
            "eod": EventType.EOD,
            "market_shift": EventType.MARKET_PRICE_OR_INVENTORY_SHIFT,
            "production_event": EventType.PRODUCTION_OR_PAYOFF_EVENT,
            "first_effect_divergence": EventType.FIRST_EFFECT_DIVERGENCE,
            "route_changed": EventType.ROUTE_OR_PHASE_CHANGE,
        }
        for field, reason in flags.items():
            if record.get(field) is True:
                reasons.add(reason)
        unit_ops, market_ops = _action_ops(record.get("action"))
        if "BUY_LAND" in market_ops:
            reasons.add(EventType.BUY_LAND)
        if "HIRE" in market_ops:
            reasons.add(EventType.HIRE)
        if market_ops & {"BUY_SEED", "BUY_PRODUCT", "BUY_ANIMAL", "SELL"}:
            reasons.add(EventType.MARKET_PRICE_OR_INVENTORY_SHIFT)
        if "BUY_ANIMAL" in market_ops or unit_ops & {"BUILD_COOP", "BUILD_PASTURE", "PLACE"}:
            reasons.add(EventType.ANIMAL_OR_STRUCTURE_TRANSITION)
        if not reasons:
            continue
        events.append(
            EventEvidence(
                step=step,
                reasons=tuple(reasons),
                family_ref=record.get("family_ref"),
                state_signature=record.get("state_signature"),
                severity=record.get("severity", 0),
                regret=record.get("regret", 0),
                recurrence_support=record.get("recurrence_support", 1),
                both_seat_support=record.get("both_seat_support", 0),
                recoverable=record.get("recoverable", True),
                lane_hint=record.get("lane_hint"),
                provenance=tuple(record.get("provenance", ())),
            )
        )
    return tuple(events)


def _proposal_offsets(reasons: Sequence[EventType]) -> tuple[int, ...]:
    return (-1, 0, 1) if set(reasons) & _NEARBY_EVENT_TYPES else (0,)


def _merge_checkpoints(checkpoints: Iterable[Checkpoint], *, max_step: int) -> list[Checkpoint]:
    groups: dict[tuple[int, str | None], list[Checkpoint]] = {}
    for checkpoint in checkpoints:
        if 0 <= checkpoint.step <= max_step:
            groups.setdefault((checkpoint.step, checkpoint.state_signature), []).append(checkpoint)
    merged = []
    for (step, state_signature), rows in groups.items():
        semantic_rows = [row for row in rows if not row.fallback] or rows
        reasons = tuple({reason for row in semantic_rows for reason in row.reasons})
        lane_source = min(
            rows,
            key=lambda row: (
                row.event_distance,
                -row.regret,
                -row.severity,
                row.lane is Lane.GENERAL,
                row.lane.value,
            ),
        )
        lane = assign_lane(
            reasons,
            hint=None if lane_source.lane is Lane.GENERAL else lane_source.lane,
            step=step,
            max_step=max_step,
        )
        merged.append(
            Checkpoint(
                step=step,
                lane=lane,
                reasons=reasons,
                state_signature=state_signature,
                severity=max(row.severity for row in rows),
                regret=max(row.regret for row in rows),
                recurrence_support=max(row.recurrence_support for row in rows),
                both_seat_support=max(row.both_seat_support for row in rows),
                recoverable=any(row.recoverable for row in rows),
                event_distance=min(row.event_distance for row in rows),
                fallback=all(row.fallback for row in rows),
                family_refs=tuple(ref for row in rows for ref in row.family_refs),
                provenance=tuple(item for row in rows for item in row.provenance),
            )
        )
    return merged


def select_checkpoints(
    evidence: Iterable[EventEvidence],
    *,
    max_step: int = DEFAULT_MAX_STEP,
    fallback_steps: Sequence[int] = DEFAULT_FALLBACK_STEPS,
    always_include_fallback: bool = False,
) -> tuple[Checkpoint, ...]:
    """Propose, merge and rank causal checkpoints with sparse fallback only."""
    _exact_nonnegative(max_step, "max_step")
    causal = []
    for event in evidence:
        if not isinstance(event, EventEvidence):
            raise TypeError("evidence must contain EventEvidence")
        if event.step > max_step:
            continue
        for offset in _proposal_offsets(event.reasons):
            step = event.step + offset
            if not 0 <= step <= max_step:
                continue
            causal.append(
                Checkpoint(
                    step=step,
                    lane=assign_lane(event.reasons, hint=event.lane_hint, step=step, max_step=max_step),
                    reasons=event.reasons,
                    state_signature=event.state_signature,
                    severity=event.severity,
                    regret=event.regret,
                    recurrence_support=event.recurrence_support,
                    both_seat_support=event.both_seat_support,
                    recoverable=event.recoverable,
                    event_distance=abs(offset),
                    family_refs=() if event.family_ref is None else (event.family_ref,),
                    provenance=event.provenance,
                )
            )
    include_fallback = always_include_fallback or not causal
    fallback = []
    if include_fallback:
        for step in fallback_steps:
            step = _exact_nonnegative(step, "fallback step")
            if step <= max_step:
                fallback.append(
                    Checkpoint(
                        step=step,
                        lane=Lane.GENERAL,
                        reasons=(EventType.FALLBACK_GRID,),
                        recurrence_support=0,
                        recoverable=False,
                        fallback=True,
                    )
                )
    merged = _merge_checkpoints([*causal, *fallback], max_step=max_step)
    return tuple(sorted(merged, key=Checkpoint.rank_key))
