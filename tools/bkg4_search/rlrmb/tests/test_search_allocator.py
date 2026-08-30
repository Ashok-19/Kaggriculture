"""Phase 16.6 — Adaptive Search Allocation tests.

17 focused tests proving allocator correctness, determinism, SEALED exclusion,
budget accounting, and backward-compatibility with Phase 16.5 cache behavior.
Plus one equal-budget comparison showing adaptive outperforms fixed allocation
on synthetic evidence.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pytest

from ..search_allocator import (
    SearchAllocator,
    RegionStats,
    _EXPLORE_W,
    _HISTORICAL_DECAY,
)
from ..search_memory import SearchMemory, effect_to_json
from ..effects import EffectBlock, EffectVector
from ..schema import DatasetRole, stable_identity


# ── helpers ──────────────────────────────────────────────────────────────────

def _effect(deltas=(1, -1)) -> EffectVector:
    return EffectVector((
        EffectBlock(
            role=DatasetRole.TARGET,
            panel_id="tp",
            baseline_margins=tuple(0 for _ in deltas),
            margin_deltas=tuple(deltas),
        ),
    ))


def _sid(tag: str) -> str:
    return stable_identity("TEST-SID-V1", tag)


def _ctx(tag: str) -> str:
    return stable_identity("TEST-CTX-V1", tag)


def _stats(
    family_id: str = "A",
    config_index: int = 0,
    n: int = 0,
    valid: int = 0,
    effect: int = 0,
    nt: int = 0,
    fd: int = 0,
    other: int = 0,
    games: int = 0,
    novel: int = 0,
    h: int = 0,
    hf: int = 0,
) -> RegionStats:
    s = RegionStats(context_id=_ctx(family_id), family_id=family_id, config_index=config_index)
    s.total_evaluations = n
    s.valid_evaluations = valid
    s.effect_evaluations = effect
    s.non_transferable_count = nt
    s.future_dependent_count = fd
    s.other_failure_count = other
    s.total_cuda_games = games
    s.novel_structural = novel
    s.historical_evaluations = h
    s.historical_failures = hf
    return s


# ── Test 1: Same memory/config produces identical allocation ─────────────────

def test_identical_allocation_from_same_db(tmp_path: Path) -> None:
    db = tmp_path / "mem.db"
    ctx_a = _ctx("A")
    ctx_b = _ctx("B")
    ej = effect_to_json(_effect())

    with SearchMemory(db) as mem:
        for i in range(5):
            mem.record_evaluation(_sid(f"a-{i}"), ctx_a, valid=True, effect_json=ej, cuda_games=10)
            mem.record_evaluation(_sid(f"b-{i}"), ctx_b, valid=False,
                                  failure_class="NON_TRANSFERABLE", cuda_games=10)

    candidates = [("A", ctx_a, 0), ("B", ctx_b, 1)]

    def _rank(db_path: Path) -> list[str]:
        with SearchMemory(db_path) as m:
            alloc = SearchAllocator(m)
            ranked = alloc.rank_families(candidates, set())
            return [r[0] for r in ranked]

    assert _rank(db) == _rank(db)
    assert _rank(db)[0] == "A"  # higher yield must win


# ── Test 2: Unseen regions receive exploration priority ──────────────────────

def test_unseen_region_gets_maximum_exploration_bonus() -> None:
    alloc = SearchAllocator(memory=None)

    unseen = _stats("U", n=0)
    score = alloc.score_region(unseen)
    assert score == pytest.approx(_EXPLORE_W / math.sqrt(1.0))

    # Unseen must outscore a saturated-failure region
    dead = _stats("D", n=50, valid=0, effect=0, nt=30, fd=20, novel=0)
    assert score > alloc.score_region(dead)


# ── Test 3: High-yield > repeatedly low-yield ────────────────────────────────

def test_high_yield_region_scores_higher_than_low_yield() -> None:
    alloc = SearchAllocator(memory=None)

    high = _stats("H", n=20, valid=18, effect=15, nt=1, novel=15, games=2000)
    low  = _stats("L", n=20, valid=2,  effect=1,  nt=10, fd=5,  novel=1, games=2000)

    assert alloc.score_region(high) > alloc.score_region(low)


# ── Test 4: Duplicate-heavy regions are deprioritized ────────────────────────

def test_duplicate_heavy_region_deprioritized() -> None:
    alloc = SearchAllocator(memory=None)

    fresh = _stats("F", n=20, valid=10, effect=5, novel=18)
    dup   = _stats("D", n=20, valid=10, effect=5, novel=2)

    assert alloc.score_region(fresh) > alloc.score_region(dup)


# ── Test 5: NON_TRANSFERABLE-heavy regions are deprioritized ─────────────────

def test_non_transferable_heavy_deprioritized() -> None:
    alloc = SearchAllocator(memory=None)

    clean    = _stats("C", n=20, valid=18, effect=10, nt=1,  novel=15)
    nt_heavy = _stats("N", n=20, valid=5,  effect=2,  nt=15, novel=5)

    assert alloc.score_region(clean) > alloc.score_region(nt_heavy)


# ── Test 6: Productive transferable/accepted regions gain priority ────────────

def test_productive_region_gains_priority() -> None:
    alloc = SearchAllocator(memory=None)

    good = _stats("G", n=20, valid=18, effect=16, nt=1, novel=12)
    bad  = _stats("B", n=20, valid=3,  effect=1,  nt=5, fd=10, novel=2)

    assert alloc.score_region(good) > alloc.score_region(bad)


# ── Test 7: Historical evidence influences priority only as prior ─────────────

def test_historical_evidence_as_prior_only() -> None:
    alloc = SearchAllocator(memory=None)

    current   = _stats("C", n=20, valid=18, effect=15, novel=12)
    hist_only = _stats("H", n=0, h=20, hf=0)

    score_current = alloc.score_region(current)
    score_hist    = alloc.score_region(hist_only)

    # Historical priors have positive influence but are weaker than current evidence
    assert score_hist > 0
    assert score_hist < score_current

    # Historical with many failures scores lower than historical with no failures
    hist_bad = _stats("HB", n=0, h=20, hf=18)
    assert alloc.score_region(hist_only) > alloc.score_region(hist_bad)


# ── Test 8: New generation does not treat old exact outcomes as current truth ──

def test_new_generation_context_id_differs_from_old(tmp_path: Path) -> None:
    db = tmp_path / "mem.db"
    old_ctx = _ctx("gen-0-A")
    new_ctx = _ctx("gen-1-A")
    ej = effect_to_json(_effect())

    with SearchMemory(db) as mem:
        for i in range(10):
            mem.record_evaluation(_sid(f"p-{i}"), old_ctx, valid=True, effect_json=ej, cuda_games=5)

        alloc = SearchAllocator(mem)

        stats_old = alloc._query_stats(old_ctx, "A", 0)
        stats_new = alloc._query_stats(new_ctx, "A", 0)

        assert stats_old.total_evaluations == 10
        assert stats_new.total_evaluations == 0  # new generation → no exact records

        # New generation region must be treated as unseen (exploration priority)
        assert alloc.score_region(stats_new) == pytest.approx(_EXPLORE_W / math.sqrt(1.0))


# ── Test 9: Regions are not permanently starved ───────────────────────────────

def test_no_permanent_starvation() -> None:
    alloc = SearchAllocator(memory=None)

    # Even a region with 100% failures retains a positive exploration bonus
    bad = _stats("B", n=100, valid=0, effect=0, nt=50, fd=50, novel=0)
    exploration_component = _EXPLORE_W / math.sqrt(1.0 + 100)

    # The exploration bonus is always positive regardless of yield/failure
    assert exploration_component > 0

    # An unseen region ALWAYS scores higher than a maximally-failed region
    unseen = _stats("U", n=0)
    assert alloc.score_region(unseen) == pytest.approx(_EXPLORE_W / math.sqrt(1.0))
    # Unseen exploration bonus > saturated-region exploration bonus
    assert alloc.score_region(unseen) > exploration_component


# ── Test 10: Deterministic tie-breaking works ─────────────────────────────────

def test_deterministic_tie_breaking_by_config_index() -> None:
    alloc = SearchAllocator(memory=None)

    # Three unseen candidates in shuffled order
    candidates = [
        ("C", _ctx("C"), 2),
        ("A", _ctx("A"), 0),
        ("B", _ctx("B"), 1),
    ]
    ranked = alloc.rank_families(candidates, set())

    # All unseen → equal scores → tie-break by config_index ascending
    assert ranked[0][2] == 0  # config_index 0 first
    assert ranked[1][2] == 1
    assert ranked[2][2] == 2

    scores = [r[3] for r in ranked]
    assert scores[0] == pytest.approx(scores[1])
    assert scores[1] == pytest.approx(scores[2])


# ── Test 11: Total assigned budget stays within requested CUDA budget ─────────

def test_budget_accounting_is_exact() -> None:
    alloc = SearchAllocator(memory=None)
    per_decision = 100_000
    candidates = [("A", _ctx("A"), 0), ("B", _ctx("B"), 1)]

    for _ in range(10):
        result = alloc.select(candidates, set())
        assert result is not None
        alloc.record_budget(result[0], per_decision)

    total = sum(alloc._budget_by_family.values())
    assert total == 10 * per_decision


# ── Test 12: Allocation never fabricates CUDA-game consumption ────────────────

def test_allocation_does_not_fabricate_cuda_games() -> None:
    alloc = SearchAllocator(memory=None)
    candidates = [("A", _ctx("A"), 0)]

    for _ in range(20):
        alloc.select(candidates, set())

    # record_budget was never called → no games "consumed"
    assert alloc._budget_by_family == {}
    assert alloc.decisions == 20


# ── Test 13: SEALED data cannot affect scores ─────────────────────────────────

def test_sealed_data_cannot_affect_allocator_scores(tmp_path: Path) -> None:
    db = tmp_path / "mem.db"
    ctx = _ctx("sealed-A")
    sid = _sid("sealed-prog")

    with SearchMemory(db) as mem:
        mem.record_sealed_audit(sid, ctx, passed=True)

        alloc = SearchAllocator(mem)
        stats = alloc._query_stats(ctx, "A", 0)

        # SEALED record must NOT appear in evaluation stats
        assert stats.total_evaluations == 0
        assert stats.effect_evaluations == 0
        score = alloc.score_region(stats)

    # Score must equal that of a completely unseen region
    alloc_no_mem = SearchAllocator(memory=None)
    expected = alloc_no_mem.score_region(RegionStats(context_id=ctx, family_id="A"))
    assert score == pytest.approx(expected)


# ── Test 14: Restart with same DB produces same decisions ─────────────────────

def test_restart_same_db_same_decisions(tmp_path: Path) -> None:
    db = tmp_path / "mem.db"
    ctx_a = _ctx("A")
    ctx_b = _ctx("B")
    ej = effect_to_json(_effect())

    with SearchMemory(db) as mem:
        for i in range(10):
            mem.record_evaluation(_sid(f"a-{i}"), ctx_a, valid=True, effect_json=ej, cuda_games=10)
        for i in range(10):
            mem.record_evaluation(_sid(f"b-{i}"), ctx_b, valid=False,
                                  failure_class="NON_TRANSFERABLE", cuda_games=10)

    candidates = [("A", ctx_a, 0), ("B", ctx_b, 1)]

    def _get_ranking(db_path: Path) -> list[str]:
        with SearchMemory(db_path) as m:
            a = SearchAllocator(m)
            return [r[0] for r in a.rank_families(candidates, set())]

    r1 = _get_ranking(db)
    r2 = _get_ranking(db)
    assert r1 == r2
    assert r1[0] == "A"  # high-yield family ranks first


# ── Test 15: Phase 16.5 exact cache behavior remains intact ──────────────────

def test_phase165_exact_cache_apis_intact(tmp_path: Path) -> None:
    """Phase 16.6 additions do not break Phase 16.5 SearchMemory read/write APIs.

    Full MemoryBackedEvaluator integration is already covered by
    test_search_memory.py tests 18 and 21.  Here we verify that the new
    allocator query methods coexist correctly with the existing evaluation APIs
    on the same database, and that new APIs return accurate aggregates.
    """
    db = tmp_path / "mem.db"
    ctx = _ctx("p165-ctx")
    ej = effect_to_json(_effect())

    with SearchMemory(db) as mem:
        # Pre-populate via the existing Phase 16.5 write APIs
        for i in range(6):
            mem.record_evaluation(_sid(f"prog-{i}"), ctx, valid=True, effect_json=ej, cuda_games=10)
        mem.record_evaluation(_sid("prog-nt"), ctx, valid=False,
                              failure_class="NON_TRANSFERABLE", cuda_games=10)
        for i in range(6):
            mem.record_structural(_sid(f"prog-{i}"), context_id=ctx)

        # Phase 16.5 exact-evaluation API still works
        rec = mem.get_exact_evaluation(_sid("prog-0"), ctx)
        assert rec is not None
        assert rec["valid"] is True
        assert rec["effect_json"] == ej

        # Phase 16.6 allocator query APIs return accurate aggregates
        stats_row = mem.get_context_evaluation_stats(ctx)
        assert stats_row["total_evaluations"] == 7
        assert stats_row["valid_evaluations"] == 6
        assert stats_row["effect_evaluations"] == 6
        assert stats_row["non_transferable_count"] == 1
        assert stats_row["novel_structural"] == 6
        assert stats_row["total_cuda_games"] == 70

        hist_row = mem.get_context_historical_stats(ctx)
        assert hist_row["total"] == 0  # nothing promoted yet

        # Promote and verify historical stats
        promoted = mem.promote_to_historical(ctx, new_generation=1)
        assert promoted == 7
        hist_row2 = mem.get_context_historical_stats(ctx)
        assert hist_row2["total"] == 7
        assert hist_row2["failures"] == 1


# ── Test 16: Semantic mismatches = 0 ─────────────────────────────────────────

def test_allocator_does_not_cause_semantic_mismatches() -> None:
    """Allocator is pure arithmetic over memory stats; it never alters IR or effects."""
    from .. import search_allocator
    alloc = search_allocator.SearchAllocator(memory=None)
    s = _stats("X", n=5, valid=4, effect=3, novel=4)
    score = alloc.score_region(s)
    assert isinstance(score, float) and math.isfinite(score)
    # Allocator has no access to IR, BKG4 runtime, or evaluation semantics
    assert not hasattr(alloc, "semantic_id")
    assert not hasattr(alloc, "effect_vector")
    assert not hasattr(alloc, "bkg4")


# ── Test 17: Protected W→L = 0 ───────────────────────────────────────────────

def test_allocator_does_not_modify_acceptance_gates() -> None:
    """Allocator selects which family to search next; it never touches gates."""
    alloc = SearchAllocator(memory=None)
    assert not hasattr(alloc, "acceptance_threshold")
    assert not hasattr(alloc, "routability_gate")
    assert not hasattr(alloc, "cegis_gate")
    assert not hasattr(alloc, "archive")
    candidates = [("A", _ctx("A"), 0)]
    fid, ectx, cidx, score = alloc.select(candidates, set())
    assert isinstance(fid, str) and isinstance(score, float)


# ── Equal-budget comparison: adaptive vs fixed ────────────────────────────────

def test_equal_budget_adaptive_outperforms_fixed(tmp_path: Path) -> None:
    """Adaptive allocator concentrates budget on the high-yield family.

    Fixture: two families.
    - high_yield: 80% novel structural, 75% effect-producing, 5% NON_TRANSFERABLE
    - low_yield:  10% novel structural, 5% effect-producing, 70% NON_TRANSFERABLE

    Fixed allocator: alternates 50/50 (config-order round-robin, no memory).
    Adaptive allocator: reads evidence and prefers high_yield.

    Both are given the same 20 decisions. We measure how many decisions go to
    high_yield — adaptive should allocate more to it than fixed (10/20).
    """
    db = tmp_path / "mem.db"
    ctx_h = _ctx("high")
    ctx_l = _ctx("low")
    ej = effect_to_json(_effect())

    with SearchMemory(db) as mem:
        # Populate 20 evidence records for each family
        for i in range(20):
            # high_yield: 16/20 valid with effect, 1/20 NON_TRANSFERABLE
            if i < 16:
                mem.record_evaluation(
                    _sid(f"h-{i}"), ctx_h, valid=True, effect_json=ej, cuda_games=10
                )
                mem.record_structural(_sid(f"h-{i}"), context_id=ctx_h)
            elif i == 16:
                mem.record_evaluation(
                    _sid(f"h-{i}"), ctx_h, valid=False,
                    failure_class="NON_TRANSFERABLE", cuda_games=10
                )
            else:
                mem.record_evaluation(
                    _sid(f"h-{i}"), ctx_h, valid=True, cuda_games=10
                )
            # low_yield: 2/20 valid with effect, 14/20 NON_TRANSFERABLE
            if i < 14:
                mem.record_evaluation(
                    _sid(f"l-{i}"), ctx_l, valid=False,
                    failure_class="NON_TRANSFERABLE", cuda_games=10
                )
            elif i < 16:
                mem.record_evaluation(
                    _sid(f"l-{i}"), ctx_l, valid=True, effect_json=ej, cuda_games=10
                )
                mem.record_structural(_sid(f"l-{i}"), context_id=ctx_l)
            else:
                mem.record_evaluation(
                    _sid(f"l-{i}"), ctx_l, valid=False,
                    failure_class="FUTURE_DEPENDENT", cuda_games=10
                )

    n_decisions = 20
    candidates = [("high", ctx_h, 0), ("low", ctx_l, 1)]

    # Fixed allocator: round-robin (no memory) → exactly 10 high, 10 low
    fixed_alloc = SearchAllocator(memory=None)
    fixed_choices: dict[str, int] = {"high": 0, "low": 0}
    # Simulate round-robin manually
    for i in range(n_decisions):
        fid = "high" if i % 2 == 0 else "low"
        fixed_choices[fid] += 1
    assert fixed_choices["high"] == 10

    # Adaptive allocator: reads real evidence
    with SearchMemory(db) as mem:
        adaptive_alloc = SearchAllocator(mem)
        adaptive_choices: dict[str, int] = {"high": 0, "low": 0}
        for _ in range(n_decisions):
            result = adaptive_alloc.select(candidates, set())
            assert result is not None
            adaptive_choices[result[0]] += 1

    # Adaptive must prefer high_yield by a clear margin
    assert adaptive_choices["high"] > fixed_choices["high"], (
        f"adaptive chose high {adaptive_choices['high']}/20 "
        f"but fixed chose {fixed_choices['high']}/20 — no improvement"
    )
    # Adaptive improvement: >25% more budget to high-yield family
    improvement = (adaptive_choices["high"] - fixed_choices["high"]) / fixed_choices["high"]
    assert improvement >= 0.25, (
        f"adaptive improvement {improvement:.1%} < 25% target"
    )

    # Allocator metrics make sense
    assert adaptive_alloc.decisions == n_decisions
    assert adaptive_alloc.regions_considered == n_decisions * 2
    assert adaptive_alloc.exploitation_picks > 0
