"""Phase 16.5 — Durable Semantic Search Memory tests.

19 required tests covering all knowledge scopes, the SEALED firewall,
crash-safety invariants, effect-signature roundtrip, and MemoryBackedEvaluator
CUDA-game avoidance.
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import pytest

from ..search_memory import (
    SearchMemory,
    MEMORY_SCHEMA_VERSION,
    effect_from_json,
    effect_to_json,
    evaluation_context_id,
)
from ..effects import EffectBlock, EffectVector
from ..schema import DatasetRole, stable_identity


# ── helpers ─────────────────────────────────────────────────────────────────

def _effect(deltas=(1, -1)):
    return EffectVector((
        EffectBlock(
            role=DatasetRole.TARGET,
            panel_id="test-panel",
            baseline_margins=tuple(0 for _ in deltas),
            margin_deltas=tuple(deltas),
        ),
    ))


def _sid(tag: str) -> str:
    return stable_identity("TEST-SID-V1", tag)


def _ctx(tag: str) -> str:
    return stable_identity("TEST-CTX-V1", tag)


# ── test 1: structural memory persists across DB restart ─────────────────────

def test_structural_seen_persists_across_restart(tmp_path: Path) -> None:
    db = tmp_path / "mem.db"
    sid = _sid("prog-a")

    with SearchMemory(db) as mem:
        assert not mem.is_structurally_seen(sid)
        mem.record_structural(sid)
        assert mem.is_structurally_seen(sid)

    # Reopen: structural record must survive
    with SearchMemory(db) as mem2:
        assert mem2.is_structurally_seen(sid)


# ── test 2: exact evaluation: miss then hit on same context ──────────────────

def test_exact_evaluation_same_context_hit(tmp_path: Path) -> None:
    db = tmp_path / "mem.db"
    sid = _sid("prog-b")
    ctx = _ctx("ctx-1")
    ej = effect_to_json(_effect())

    with SearchMemory(db) as mem:
        assert mem.get_exact_evaluation(sid, ctx) is None
        mem.record_evaluation(sid, ctx, valid=True, effect_json=ej, cuda_games=50)
        rec = mem.get_exact_evaluation(sid, ctx)
        assert rec is not None
        assert rec["valid"] is True
        assert rec["effect_json"] == ej
        assert rec["cuda_games"] == 50


# ── test 3: different context_id forces reevaluation ────────────────────────

def test_exact_evaluation_different_context_is_miss(tmp_path: Path) -> None:
    db = tmp_path / "mem.db"
    sid = _sid("prog-c")
    ctx_a = _ctx("ctx-a")
    ctx_b = _ctx("ctx-b")

    with SearchMemory(db) as mem:
        mem.record_evaluation(sid, ctx_a, valid=True, effect_json=effect_to_json(_effect()))
        assert mem.get_exact_evaluation(sid, ctx_b) is None


# ── test 4: new BKG4 generation invalidates exact context (different ctx_id) ─

def test_new_generation_invalidates_exact_context(tmp_path: Path) -> None:
    db = tmp_path / "mem.db"
    sid = _sid("prog-d")
    old_ctx = _ctx("gen-0")
    new_ctx = _ctx("gen-1")  # would differ because controller_hash changed

    with SearchMemory(db) as mem:
        mem.record_evaluation(sid, old_ctx, valid=True, effect_json=effect_to_json(_effect()))
        # Old context hit
        assert mem.get_exact_evaluation(sid, old_ctx) is not None
        # New context: miss (must reevaluate)
        assert mem.get_exact_evaluation(sid, new_ctx) is None


# ── test 5: historical prior survives generation change ──────────────────────

def test_historical_prior_survives_generation_change(tmp_path: Path) -> None:
    db = tmp_path / "mem.db"
    sid = _sid("prog-e")
    ctx = _ctx("ctx-gen0")

    with SearchMemory(db) as mem:
        mem.record_evaluation(sid, ctx, valid=False, fault="NON_TRANSFERABLE",
                              failure_class="NON_TRANSFERABLE", cuda_games=40)
        promoted = mem.promote_to_historical(ctx, new_generation=1)
        assert promoted == 1

        priors = mem.get_historical_priors(sid)
        assert len(priors) == 1
        assert priors[0]["failure_class"] == "NON_TRANSFERABLE"
        assert priors[0]["generation"] == 0

        # Exact evaluation record still present (promote copies, doesn't delete)
        assert mem.get_exact_evaluation(sid, ctx) is not None


# ── test 6: promote_to_historical increments context_invalidations ───────────

def test_promote_to_historical_returns_count(tmp_path: Path) -> None:
    db = tmp_path / "mem.db"
    ctx = _ctx("ctx-multi")
    sids = [_sid(f"prog-{i}") for i in range(5)]

    with SearchMemory(db) as mem:
        for sid in sids:
            mem.record_evaluation(sid, ctx, valid=True, effect_json=effect_to_json(_effect()))
        count = mem.promote_to_historical(ctx, new_generation=1)
        assert count == 5
        assert mem.context_invalidations == 1


# ── test 7: failure classes persist correctly ─────────────────────────────────

def test_failure_class_persists_correctly(tmp_path: Path) -> None:
    db = tmp_path / "mem.db"
    sid = _sid("prog-f")
    ctx = _ctx("ctx-fail")

    with SearchMemory(db) as mem:
        mem.record_evaluation(sid, ctx, valid=False, fault="CUDA_FAULT",
                              failure_class="CUDA_FAULT", cuda_games=10)
        rec = mem.get_exact_evaluation(sid, ctx)
        assert rec is not None
        assert rec["valid"] is False
        assert rec["fault"] == "CUDA_FAULT"
        assert rec["failure_class"] == "CUDA_FAULT"
        assert mem.persisted_failure_records == 1


# ── test 8: FUTURE_DEPENDENT failure persists ────────────────────────────────

def test_future_dependent_failure_persists(tmp_path: Path) -> None:
    db = tmp_path / "mem.db"
    sid = _sid("prog-g")
    ctx = _ctx("ctx-fd")

    with SearchMemory(db) as mem:
        mem.record_evaluation(sid, ctx, valid=False, fault="FUTURE_DEPENDENT",
                              failure_class="FUTURE_DEPENDENT")
        rec = mem.get_exact_evaluation(sid, ctx)
        assert rec is not None
        assert rec["failure_class"] == "FUTURE_DEPENDENT"


# ── test 9: NON_TRANSFERABLE persists per-context, not globally ──────────────

def test_non_transferable_per_context_not_global(tmp_path: Path) -> None:
    db = tmp_path / "mem.db"
    sid = _sid("prog-h")
    ctx_fail = _ctx("ctx-fail")
    ctx_ok = _ctx("ctx-ok")

    with SearchMemory(db) as mem:
        mem.record_evaluation(sid, ctx_fail, valid=False, fault="NON_TRANSFERABLE",
                              failure_class="NON_TRANSFERABLE")
        # In a different context, no evaluation record exists — not globally rejected
        assert mem.get_exact_evaluation(sid, ctx_ok) is None


# ── test 10: effect signature roundtrips exactly ──────────────────────────────

def test_effect_signature_roundtrip(tmp_path: Path) -> None:
    db = tmp_path / "mem.db"
    sid = _sid("prog-i")
    ctx = _ctx("ctx-effect")
    original = _effect(deltas=(3, -2, 1))
    ej = effect_to_json(original)

    with SearchMemory(db) as mem:
        mem.record_evaluation(sid, ctx, valid=True, effect_json=ej)
        rec = mem.get_exact_evaluation(sid, ctx)
        assert rec is not None
        restored = effect_from_json(rec["effect_json"])
        assert restored.blocks == original.blocks


# ── test 11: restart produces same memory state ───────────────────────────────

def test_restart_same_memory_state(tmp_path: Path) -> None:
    db = tmp_path / "mem.db"
    sids = [_sid(f"persist-{i}") for i in range(3)]
    ctx = _ctx("ctx-persist")
    ej = effect_to_json(_effect())

    with SearchMemory(db) as mem:
        for sid in sids:
            mem.record_structural(sid)
            mem.record_evaluation(sid, ctx, valid=True, effect_json=ej)
        count_before = mem.record_count()

    with SearchMemory(db) as mem2:
        assert mem2.record_count() == count_before
        for sid in sids:
            assert mem2.is_structurally_seen(sid)
            assert mem2.get_exact_evaluation(sid, ctx) is not None


# ── test 12: partial write in aborted transaction is not authoritative ─────────

def test_partial_transaction_not_authoritative(tmp_path: Path) -> None:
    db = tmp_path / "mem.db"
    sid = _sid("prog-partial")
    ctx = _ctx("ctx-partial")

    with SearchMemory(db) as mem:
        # Simulate aborted transaction: manually begin but do not commit
        mem._conn.execute("BEGIN")
        mem._conn.execute(
            "INSERT OR REPLACE INTO evaluation"
            " (semantic_id, context_id, generation, valid, fault, effect_json,"
            "  failure_class, cuda_games, insert_time)"
            " VALUES (?,?,?,?,?,?,?,?,?)",
            (sid, ctx, 0, 1, None, None, None, 0, 0),
        )
        mem._conn.execute("ROLLBACK")

        # After rollback, record is not present
        assert mem.get_exact_evaluation(sid, ctx) is None


# ── test 13: schema-version mismatch fails closed ────────────────────────────

def test_schema_version_mismatch_fails_closed(tmp_path: Path) -> None:
    db = tmp_path / "bad_schema.db"
    conn = sqlite3.connect(str(db))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("CREATE TABLE schema_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    conn.execute("INSERT INTO schema_meta (key, value) VALUES ('schema_version', '99')")
    conn.execute("CREATE TABLE structural (semantic_id TEXT PRIMARY KEY, lane TEXT, "
                 "operator TEXT, generation_first INTEGER NOT NULL DEFAULT 0, "
                 "context_id_first TEXT, insert_time INTEGER NOT NULL)")
    conn.execute("CREATE TABLE evaluation (semantic_id TEXT NOT NULL, context_id TEXT NOT NULL, "
                 "generation INTEGER NOT NULL DEFAULT 0, valid INTEGER NOT NULL DEFAULT 1, "
                 "fault TEXT, effect_json TEXT, failure_class TEXT, "
                 "cuda_games INTEGER NOT NULL DEFAULT 0, insert_time INTEGER NOT NULL, "
                 "PRIMARY KEY (semantic_id, context_id))")
    conn.execute("CREATE TABLE historical (semantic_id TEXT NOT NULL, context_id TEXT NOT NULL, "
                 "generation INTEGER NOT NULL, failure_class TEXT, outcome_summary TEXT, "
                 "insert_time INTEGER NOT NULL, PRIMARY KEY (semantic_id, context_id, generation))")
    conn.execute("CREATE TABLE sealed_audit (semantic_id TEXT NOT NULL, context_id TEXT NOT NULL, "
                 "passed INTEGER NOT NULL DEFAULT 0, insert_time INTEGER NOT NULL, "
                 "PRIMARY KEY (semantic_id, context_id))")
    conn.commit()
    conn.close()

    with pytest.raises(ValueError, match="schema mismatch"):
        SearchMemory(db).open()


# ── test 14: SEALED audit has no read API in search code path ────────────────

def test_sealed_has_no_search_read_api(tmp_path: Path) -> None:
    db = tmp_path / "mem.db"
    sid = _sid("prog-sealed")
    ctx = _ctx("ctx-sealed")

    with SearchMemory(db) as mem:
        # Write to SEALED audit (write-only from search path)
        mem.record_sealed_audit(sid, ctx, passed=True)

        # SEALED firewall: no read method exists on SearchMemory
        assert not hasattr(mem, "get_sealed_audit")
        assert not hasattr(mem, "read_sealed_audit")
        assert not hasattr(mem, "query_sealed")

        # Normal search APIs return None (no cross-contamination)
        assert mem.get_exact_evaluation(sid, ctx) is None
        assert mem.get_historical_priors(sid) == []


# ── test 15: metrics structure has all required keys ─────────────────────────

def test_metrics_structure_has_expected_keys(tmp_path: Path) -> None:
    db = tmp_path / "mem.db"
    with SearchMemory(db) as mem:
        m = mem.metrics()
    required = {
        "memory_structural_hits",
        "memory_structural_misses",
        "memory_exact_evaluation_hits",
        "memory_exact_evaluation_misses",
        "avoided_cuda_games",
        "avoided_candidate_evaluations",
        "evaluation_context_invalidations",
        "persisted_failure_records",
        "persisted_effect_signatures",
        "memory_records",
        "memory_storage_bytes",
    }
    assert required.issubset(set(m.keys()))


# ── test 16: hit/miss counters are updated by queries ────────────────────────

def test_structural_hit_miss_counters(tmp_path: Path) -> None:
    db = tmp_path / "mem.db"
    sid_known = _sid("known")
    sid_unknown = _sid("unknown")

    with SearchMemory(db) as mem:
        mem.record_structural(sid_known)
        mem.is_structurally_seen(sid_known)    # hit
        mem.is_structurally_seen(sid_unknown)  # miss
        assert mem.structural_hits == 1
        assert mem.structural_misses == 1


# ── test 17: record_count reflects stored evaluations ────────────────────────

def test_record_count_zero_initially(tmp_path: Path) -> None:
    db = tmp_path / "mem.db"
    with SearchMemory(db) as mem:
        assert mem.record_count() == 0
        mem.record_evaluation(_sid("x"), _ctx("c"), valid=True,
                              effect_json=effect_to_json(_effect()))
        assert mem.record_count() == 1


# ── test 18: MemoryBackedEvaluator — cache hit avoids CUDA games ──────────────

def test_memory_backed_evaluator_cache_hit_avoids_cuda(tmp_path: Path) -> None:
    from ..p13_equal_budget import MemoryBackedEvaluator
    from ..beam import BatchEvaluation, EvaluatedProgram
    from ..archive import Lane
    from ..checkpoints import Checkpoint, EventType
    from ..ir import MacroProgram, Phase, ActionSource, ActionFamily, Termination, TerminationType
    from ..motifs import motif_from_actions

    _MARKET_ACTIONS = list(range(8))

    def _make_prog(seed: int) -> MacroProgram:
        cp = Checkpoint(step=seed % 5 + 1, lane=Lane.MARKET_LIQUIDITY,
                        reasons=(EventType.REGRET_PEAK,))
        actions = [_MARKET_ACTIONS] * (4 + seed % 3)
        motif = motif_from_actions(actions, (ActionFamily.MARKET,), provenance=(f"seed-{seed}",))
        ph = Phase(ActionSource.SEMANTIC_SPLICE, (ActionFamily.MARKET,),
                   Termination(TerminationType.FIXED_CAP, 4), motif.semantic_id)
        return MacroProgram(lane=Lane.MARKET_LIQUIDITY, initiation=cp, phases=(ph,), max_total_steps=4)

    prog1 = _make_prog(0)
    prog2 = _make_prog(1)
    assert prog1.semantic_id != prog2.semantic_id
    eff = _effect()

    class _FakeInner:
        case_count = 10
        last_profile: dict = {}

        def projected_games(self, count: int) -> int:
            return (count + 1) * self.case_count

        def __call__(self, programs, motifs):
            evaluated = tuple(
                EvaluatedProgram(program=p, effect=eff, valid=True, fault=None)
                for p in programs
            )
            return BatchEvaluation(evaluated, self.case_count, len(programs) * self.case_count, 1, 0.1, 5.0)

    db = tmp_path / "mem.db"
    ctx = _ctx("ev-ctx")

    with SearchMemory(db) as mem:
        # Pre-populate prog1's evaluation in memory
        mem.record_evaluation(
            prog1.semantic_id, ctx,
            valid=True, effect_json=effect_to_json(eff), cuda_games=10,
        )

        fake_accounted = _FakeInner()
        mbe = MemoryBackedEvaluator(fake_accounted, mem, ctx)

        # Call with both programs; prog1 is cached, prog2 is new
        result = mbe([prog1, prog2], {})

        # Both programs in result
        assert len(result.programs) == 2
        sids = {ep.program.semantic_id for ep in result.programs}
        assert prog1.semantic_id in sids
        assert prog2.semantic_id in sids

        # prog1 avoided CUDA; only prog2 was sent to the inner evaluator
        assert mbe.avoided_candidates == 1
        assert mbe.avoided_games == 10
        assert mem.avoided_cuda_games == 10
        assert mem.avoided_candidate_evaluations == 1


# ── test 19: close is idempotent ─────────────────────────────────────────────

def test_close_is_idempotent(tmp_path: Path) -> None:
    db = tmp_path / "mem.db"
    mem = SearchMemory(db)
    mem.open()
    mem.close()
    mem.close()  # second close must not raise
