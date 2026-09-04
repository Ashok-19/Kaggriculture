"""BUILD-only regime metadata and cross-regime candidate support."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .effects import EffectBlock


@dataclass(frozen=True)
class BuildRegime:
    """Independent coordinates for one BUILD trajectory."""

    seed: int
    seat: int
    member: str
    lane: str
    trajectory: str

    @property
    def key(self) -> tuple[Any, ...]:
        return (self.seed, self.seat, self.member, self.lane, self.trajectory)


@dataclass(frozen=True)
class BuildSupport:
    """Exact BUILD outcome support across independent regime axes."""

    loss_to_win: int
    win_to_loss: int
    positive: int
    harmful: int
    regime_support: int
    seed_support: int
    seat_support: int
    member_support: int
    lane_support: int
    trajectory_support: int
    both_seat_member_support: int

    @property
    def cross_regime_supported(self) -> bool:
        return (
            self.loss_to_win > 0
            and self.seed_support >= 2
            and self.seat_support >= 2
            and self.member_support >= 2
        )

    def to_document(self) -> dict[str, Any]:
        return {
            "loss_to_win": self.loss_to_win,
            "win_to_loss": self.win_to_loss,
            "positive": self.positive,
            "harmful": self.harmful,
            "regime_support": self.regime_support,
            "seed_support": self.seed_support,
            "seat_support": self.seat_support,
            "member_support": self.member_support,
            "lane_support": self.lane_support,
            "trajectory_support": self.trajectory_support,
            "both_seat_member_support": self.both_seat_member_support,
            "cross_regime_supported": self.cross_regime_supported,
        }


def load_build_regimes(panel: str | Path) -> tuple[BuildRegime, ...]:
    """Load environment-only regime provenance in exact panel row order."""
    rows: list[BuildRegime] = []
    for line in Path(panel).read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        fields = line.split("\t")
        if len(fields) < 3:
            raise ValueError("BUILD panel row requires tape, seat and seed")
        tape = Path(fields[0])
        try:
            source = json.loads(tape.read_text(encoding="utf-8")).get("source", {})
        except (OSError, ValueError, TypeError):
            source = {}
        member = str(source.get("opponent_policy") or tape.stem)
        lane = str(source.get("strategic_lane") or "GENERAL")
        trajectory = str(
            source.get("exploiter_program")
            or source.get("mutation_origin")
            or source.get("policy_kind")
            or member
        )
        rows.append(BuildRegime(int(fields[2]), int(fields[1]), member, lane, trajectory))
    if not rows:
        raise ValueError("BUILD panel is empty")
    if len({row.key for row in rows}) != len(rows):
        raise ValueError("BUILD panel contains duplicate regime trajectories")
    return tuple(rows)


def build_support(block: EffectBlock, regimes: Sequence[BuildRegime]) -> BuildSupport:
    """Measure L->W support without allowing one trajectory to count as breadth."""
    if len(block.margin_deltas) != len(regimes):
        raise ValueError("BUILD regime metadata must align with the effect block")
    supported = [
        regime for regime, flipped in zip(regimes, block.loss_to_win) if flipped
    ]
    seats_by_member: dict[str, set[int]] = {}
    for regime in supported:
        seats_by_member.setdefault(regime.member, set()).add(regime.seat)
    return BuildSupport(
        loss_to_win=sum(block.loss_to_win),
        win_to_loss=sum(block.win_to_loss),
        positive=sum(delta > 0 for delta in block.margin_deltas),
        harmful=sum(delta < 0 for delta in block.margin_deltas),
        regime_support=len({row.key for row in supported}),
        seed_support=len({row.seed for row in supported}),
        seat_support=len({row.seat for row in supported}),
        member_support=len({row.member for row in supported}),
        lane_support=len({row.lane for row in supported}),
        trajectory_support=len({row.trajectory for row in supported}),
        both_seat_member_support=sum(seats == {0, 1} for seats in seats_by_member.values()),
    )
