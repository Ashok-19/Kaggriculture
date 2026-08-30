"""Phase 16.6 — Adaptive deterministic search allocation for RLRMB.

Reads evidence from Phase 16.5 SearchMemory to rank eligible search regions
(family × evaluation-context) by expected useful yield per CUDA game.

Scoring
-------
score = exploration_bonus(N_eff) + yield_score - penalty

    exploration_bonus = EXPLORE_W / sqrt(1 + N_eff)
    yield_score       = NOVELTY_W * (novel_structural / N)
                      + TRANSFER_W * (effect_evaluations / N)
    penalty           = FAILURE_W * (failure_count / N)
                      + FD_EXTRA if >50% of evaluations are FUTURE_DEPENDENT

N_eff = N_current + HISTORICAL_DECAY × N_historical.

Historical priors seed exploration; current exact evidence dominates.
SEALED data is NEVER queried.

Determinism
-----------
Given identical memory state and configuration, allocation is always identical.
Tie-breaking uses config_index ascending, so unseen regions are explored in
the order they appear in config["families"] — preserving the round-robin
bootstrap of earlier phases.  When memory is None, all regions score equally
and config order is preserved exactly.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence


_EXPLORE_W: float = 2.0
_GLOBAL_EXPLORE_W: float = 0.5
_NOVELTY_W: float = 3.0
_TRANSFER_W: float = 2.0
_FAILURE_W: float = 1.5
_FD_EXTRA_PENALTY: float = 1.0
_HISTORICAL_DECAY: float = 0.3
_LOW_YIELD_THRESHOLD_N: int = 5
_LOW_YIELD_RATE: float = 0.1


@dataclass
class RegionStats:
    """Aggregate evidence for one search region (family × evaluation-context)."""
    context_id: str
    family_id: str
    config_index: int = 0
    region_id: str = ""
    total_evaluations: int = 0
    valid_evaluations: int = 0
    effect_evaluations: int = 0
    non_transferable_count: int = 0
    future_dependent_count: int = 0
    other_failure_count: int = 0
    total_cuda_games: int = 0
    novel_structural: int = 0
    historical_evaluations: int = 0
    historical_failures: int = 0
    allocation_visits: int = 0
    last_selected_generation: int = -1
    global_allocation_visits: int = 0


class SearchAllocator:
    """Deterministic adaptive search allocator backed by Phase 16.5 SearchMemory.

    When ``memory`` is None, all regions score identically (exploration bonus only)
    and config order is preserved via tie-breaking — equivalent to round-robin.
    """

    def __init__(self, memory: Any | None = None) -> None:
        self._memory = memory
        self.decisions: int = 0
        self.regions_considered: int = 0
        self.exploration_picks: int = 0
        self.exploitation_picks: int = 0
        self.historical_prior_uses: int = 0
        self.low_yield_deprioritizations: int = 0
        self._budget_by_family: dict[str, int] = {}
        self._last_scores: dict[str, float] = {}

    # ── query ──────────────────────────────────────────────────────────────────

    def _query_stats(
        self,
        context_id: str,
        family_id: str,
        config_index: int,
        *,
        region_id: str | None = None,
    ) -> RegionStats:
        stable_region = str(region_id or context_id)
        stats = RegionStats(
            context_id=context_id,
            family_id=family_id,
            config_index=config_index,
            region_id=stable_region,
        )
        if self._memory is None:
            return stats
        row = self._memory.get_context_evaluation_stats(context_id)
        if row:
            stats.total_evaluations = row.get("total_evaluations", 0)
            stats.valid_evaluations = row.get("valid_evaluations", 0)
            stats.effect_evaluations = row.get("effect_evaluations", 0)
            stats.non_transferable_count = row.get("non_transferable_count", 0)
            stats.future_dependent_count = row.get("future_dependent_count", 0)
            stats.other_failure_count = row.get("other_failure_count", 0)
            stats.total_cuda_games = row.get("total_cuda_games", 0)
            stats.novel_structural = row.get("novel_structural", 0)
        if hasattr(self._memory, "get_region_historical_stats"):
            hist = self._memory.get_region_historical_stats(stable_region)
        else:
            hist = self._memory.get_context_historical_stats(context_id)
        if hist and hist.get("total", 0) > 0:
            stats.historical_evaluations = hist["total"]
            stats.historical_failures = hist.get("failures", 0)
            self.historical_prior_uses += 1
        if hasattr(self._memory, "get_region_allocation_stats"):
            visits = self._memory.get_region_allocation_stats(stable_region)
            stats.allocation_visits = int(visits.get("selections", 0))
            stats.last_selected_generation = int(visits.get("last_generation", -1))
        return stats

    # ── scoring ────────────────────────────────────────────────────────────────

    def score_region(self, stats: RegionStats) -> float:
        """Deterministic utility score for one search region. Higher is better."""
        n = stats.total_evaluations
        n_hist = stats.historical_evaluations
        n_eff = n + _HISTORICAL_DECAY * n_hist

        # Durable completed-region visits contribute to exploration pressure even
        # when a search produced no evaluation rows. A generation commit therefore
        # cannot reset every family to an identical unseen score and permanently
        # favor config item #1.
        exploration_n = n_eff + stats.allocation_visits
        exploration_bonus = _EXPLORE_W / math.sqrt(1.0 + exploration_n)
        # Global visitation pressure is zero at bootstrap, but grows for regions
        # that remain unselected while other eligible regions consume durable
        # visits.  This is deterministic UCB-style exploration: exploitation can
        # dominate for long stretches, but no still-eligible region is starved
        # permanently merely because its historical prior is poor.
        if stats.global_allocation_visits > 0:
            exploration_bonus += _GLOBAL_EXPLORE_W * math.sqrt(
                math.log1p(stats.global_allocation_visits)
                / (1.0 + stats.allocation_visits)
            )

        if n_eff == 0:
            return exploration_bonus

        if n > 0:
            novelty_rate = stats.novel_structural / n
            effect_rate = stats.effect_evaluations / n
            failure_count = (
                stats.non_transferable_count
                + stats.future_dependent_count
                + stats.other_failure_count
            )
            failure_rate = failure_count / n
            yield_score = _NOVELTY_W * novelty_rate + _TRANSFER_W * effect_rate
            penalty = _FAILURE_W * failure_rate
            if stats.future_dependent_count / n > 0.5:
                penalty += _FD_EXTRA_PENALTY
            if (n >= _LOW_YIELD_THRESHOLD_N
                    and novelty_rate < _LOW_YIELD_RATE
                    and effect_rate < _LOW_YIELD_RATE):
                self.low_yield_deprioritizations += 1
        else:
            # Historical priors only
            hist_fail_rate = (
                stats.historical_failures / n_hist if n_hist > 0 else 0.0
            )
            yield_score = _HISTORICAL_DECAY * (1.0 - hist_fail_rate)
            penalty = _HISTORICAL_DECAY * hist_fail_rate

        return exploration_bonus + yield_score - penalty

    # ── ranking ────────────────────────────────────────────────────────────────

    @staticmethod
    def _candidate_parts(candidate: tuple[Any, ...]) -> tuple[str, str, str, int]:
        """Normalize legacy 3-tuples and production 4-tuples deterministically."""
        if len(candidate) == 3:
            family_id, eval_ctx, config_idx = candidate
            return str(family_id), str(eval_ctx), str(eval_ctx), int(config_idx)
        if len(candidate) == 4:
            family_id, eval_ctx, region_id, config_idx = candidate
            return str(family_id), str(eval_ctx), str(region_id), int(config_idx)
        raise ValueError("allocator candidate must have 3 or 4 fields")

    def rank_families(
        self,
        candidates: Sequence[tuple[Any, ...]],
        completed_exact: "set[str]",
    ) -> list[tuple[str, str, int, float]]:
        """Rank eligible families by adaptive score.

        Args:
            candidates: (family_id, eval_context_id, config_index) for each eligible family.
            completed_exact: set of search_context_ids already executed (for reference).

        Returns:
            List of (family_id, eval_context_id, config_index, score), sorted by
            score descending then config_index ascending for deterministic tie-breaking.
        """
        queried: list[tuple[str, str, int, RegionStats]] = []
        for candidate in candidates:
            family_id, eval_ctx, region_id, config_idx = self._candidate_parts(candidate)
            self.regions_considered += 1
            stats = self._query_stats(
                eval_ctx, family_id, config_idx, region_id=region_id
            )
            queried.append((family_id, eval_ctx, config_idx, stats))
        global_visits = sum(stats.allocation_visits for _, _, _, stats in queried)
        scored: list[tuple[str, str, int, float]] = []
        for family_id, eval_ctx, config_idx, stats in queried:
            stats.global_allocation_visits = global_visits
            score = self.score_region(stats)
            scored.append((family_id, eval_ctx, config_idx, score))
        scored.sort(key=lambda x: (-x[3], x[2]))
        self._last_scores = {fid: sc for fid, _, _, sc in scored}
        return scored

    def select(
        self,
        candidates: Sequence[tuple[Any, ...]],
        completed_exact: "set[str]",
    ) -> "tuple[str, str, int, float] | None":
        """Select the best eligible family. Returns None if candidates is empty."""
        ranked = self.rank_families(candidates, completed_exact)
        if not ranked:
            return None
        fid, ectx, cidx, score = ranked[0]
        region_id = ectx
        for candidate in candidates:
            cfid, cectx, cregion, ccidx = self._candidate_parts(candidate)
            if (cfid, cectx, ccidx) == (fid, ectx, cidx):
                region_id = cregion
                break
        stats = self._query_stats(ectx, fid, cidx, region_id=region_id)
        if stats.total_evaluations == 0 and stats.historical_evaluations == 0:
            self.exploration_picks += 1
        else:
            self.exploitation_picks += 1
        self.decisions += 1
        return ranked[0]

    def record_budget(self, family_id: str, cuda_games: int) -> None:
        self._budget_by_family[family_id] = (
            self._budget_by_family.get(family_id, 0) + cuda_games
        )

    def metrics_document(self) -> dict[str, Any]:
        return {
            "allocator_decisions": self.decisions,
            "allocator_regions_considered": self.regions_considered,
            "allocator_exploration_picks": self.exploration_picks,
            "allocator_exploitation_picks": self.exploitation_picks,
            "allocator_historical_prior_uses": self.historical_prior_uses,
            "allocator_low_yield_deprioritizations": self.low_yield_deprioritizations,
            "allocator_budget_by_region": dict(sorted(self._budget_by_family.items())),
            "allocator_last_region_scores": {
                k: round(v, 6) for k, v in sorted(self._last_scores.items())
            },
        }
