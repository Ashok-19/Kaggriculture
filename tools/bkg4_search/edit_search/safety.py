from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Sequence

from .fitness import PairEvaluation


class SafetyTier(str, Enum):
    ELITE_FLIP = "elite_flip"
    ELITE_MARGIN = "elite_margin"
    STRICT_SAFE_USEFUL = "strict_safe_useful"
    SAFE_INCREMENTAL = "safe_incremental"
    BOUNDARY = "boundary"
    HARD_NEGATIVE = "hard_negative"
    CONTROLLED_NOOP = "controlled_noop"


_TIER_RANK = {
    SafetyTier.ELITE_FLIP: 0,
    SafetyTier.ELITE_MARGIN: 1,
    SafetyTier.STRICT_SAFE_USEFUL: 2,
    SafetyTier.SAFE_INCREMENTAL: 3,
    SafetyTier.CONTROLLED_NOOP: 4,
    SafetyTier.BOUNDARY: 5,
    SafetyTier.HARD_NEGATIVE: 6,
}


@dataclass(frozen=True)
class ConstraintViolation:
    new_losses: int
    w2l: int
    severe_regressions: int
    worst_protected_loss: int
    protected_tail_damage: float

    @property
    def feasible(self) -> bool:
        return self.new_losses == 0 and self.w2l == 0 and self.severe_regressions == 0

    def key(self) -> tuple[float, ...]:
        return (
            int(self.new_losses),
            int(self.w2l),
            int(self.severe_regressions),
            int(self.worst_protected_loss),
            float(self.protected_tail_damage),
        )


@dataclass(frozen=True)
class SafetyAssessment:
    tier: SafetyTier
    violation: ConstraintViolation
    losses_after: int
    l2w: int
    fragile_wins_after: int
    worst_candidate_margin: int
    mean_margin_delta: float
    worst_protected_delta: int
    protected_tail_mean: float
    behavior_changing: bool
    exact_noop: bool
    changed_cases: int
    action_change_total: int
    activation_total: int

    @property
    def feasible(self) -> bool:
        return self.violation.feasible

    @property
    def elite(self) -> bool:
        return self.tier in (SafetyTier.ELITE_FLIP, SafetyTier.ELITE_MARGIN)

    @property
    def safe_useful(self) -> bool:
        return self.feasible and self.behavior_changing and (
            self.l2w > 0 or self.mean_margin_delta > 0.0
        )

    def production_key(self, *, program_length: int = 0, rule_delta: int = 0) -> tuple[float, ...]:
        if not self.feasible:
            raise ValueError("production key requires a feasible candidate")
        # Locality/complexity are deliberately late tie breakers.
        return (
            _TIER_RANK[self.tier],
            int(self.losses_after),
            -int(self.l2w),
            -int(self.worst_candidate_margin),
            -int(self.worst_protected_delta),
            -float(self.protected_tail_mean),
            -float(self.mean_margin_delta),
            int(self.changed_cases),
            int(self.action_change_total),
            int(self.activation_total),
            abs(int(rule_delta)),
            int(program_length),
        )

    def repair_key(self) -> tuple[float, ...]:
        if self.feasible:
            return (0.0,) + self.production_key()
        # Unsafe states compete on distance to feasibility BEFORE strategic reward/novelty.
        return (
            1.0,
            *self.violation.key(),
            int(self.losses_after),
            -int(self.l2w),
            -int(self.worst_candidate_margin),
            -float(self.mean_margin_delta),
            int(self.changed_cases),
        )


@dataclass(frozen=True)
class ProgramSafetyAssessment:
    source: SafetyAssessment
    root: SafetyAssessment

    @property
    def production_feasible(self) -> bool:
        return self.root.feasible

    @property
    def high_quality_supervision(self) -> bool:
        return self.root.feasible and self.source.feasible and self.root.safe_useful


@dataclass(frozen=True)
class SafetySpec:
    severe_delta: int = -3000
    fragile_margin: int = 5000
    protected_tail_fraction: float = 0.25
    elite_mean_margin: float = 250.0
    boundary_protected_loss: int = 3000

    @staticmethod
    def _tail_mean(values: Sequence[int], fraction: float) -> float:
        if not values:
            return 0.0
        ordered = sorted(int(x) for x in values)
        count = max(1, int(math.ceil(len(ordered) * max(0.0, min(1.0, float(fraction))))))
        return float(sum(ordered[:count]) / count)

    def assess(self, pair: PairEvaluation) -> SafetyAssessment:
        aggregate = pair.certified_aggregate()
        protected_deltas = [
            int(row["margin_delta"])
            for row in pair.case_rows
            if bool(row.get("protected", True)) and int(row["baseline_margin"]) > 0
        ]
        worst_protected_delta = min(protected_deltas) if protected_deltas else 0
        protected_tail_mean = self._tail_mean(protected_deltas, self.protected_tail_fraction)
        violation = ConstraintViolation(
            new_losses=int(aggregate["new_loss_count"]),
            w2l=int(aggregate["w2l_count"]),
            severe_regressions=int(aggregate["severe_regression_count"]),
            worst_protected_loss=max(0, -int(worst_protected_delta)),
            protected_tail_damage=max(0.0, -float(protected_tail_mean)),
        )
        behavior = bool(aggregate["behavior_changing"])
        noop = bool(aggregate["exact_noop"])
        l2w = int(aggregate["l2w_count"])
        mean_delta = float(aggregate["mean_margin_delta"])
        if noop:
            tier = SafetyTier.CONTROLLED_NOOP
        elif not violation.feasible:
            near = (
                violation.new_losses <= 1
                and violation.w2l <= 1
                and violation.severe_regressions <= 1
                and violation.worst_protected_loss <= int(self.boundary_protected_loss)
            )
            tier = SafetyTier.BOUNDARY if near else SafetyTier.HARD_NEGATIVE
        elif behavior and l2w > 0 and worst_protected_delta >= 0 and protected_tail_mean >= 0.0:
            tier = SafetyTier.ELITE_FLIP
        elif behavior and mean_delta >= float(self.elite_mean_margin) and worst_protected_delta >= 0 and protected_tail_mean >= 0.0:
            tier = SafetyTier.ELITE_MARGIN
        elif behavior and (l2w > 0 or mean_delta > 0.0):
            tier = SafetyTier.STRICT_SAFE_USEFUL
        else:
            tier = SafetyTier.SAFE_INCREMENTAL
        return SafetyAssessment(
            tier=tier,
            violation=violation,
            losses_after=int(aggregate["losses_after"]),
            l2w=l2w,
            fragile_wins_after=int(aggregate["fragile_wins_after"]),
            worst_candidate_margin=int(aggregate["worst_candidate_margin"]),
            mean_margin_delta=mean_delta,
            worst_protected_delta=int(worst_protected_delta),
            protected_tail_mean=float(protected_tail_mean),
            behavior_changing=behavior,
            exact_noop=noop,
            changed_cases=len(pair.changed_case_ids),
            action_change_total=int(aggregate["action_change_total"]),
            activation_total=int(aggregate["activation_total"]),
        )

    def assess_program(self, *, source_pair: PairEvaluation, root_pair: PairEvaluation) -> ProgramSafetyAssessment:
        if source_pair.child_semantic_id != root_pair.child_semantic_id or source_pair.panel_id != root_pair.panel_id:
            raise ValueError("source/root program assessments must refer to the same child and panel")
        return ProgramSafetyAssessment(source=self.assess(source_pair), root=self.assess(root_pair))
