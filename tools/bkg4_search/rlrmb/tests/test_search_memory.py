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


# ── test 20 (regression): promote_to_historical uses evaluation_context_id ──────
# Defect 1: cloud_runner was passing search_context_identity (includes generation)
# to promote_to_historical, but evaluations are stored under evaluation_context_id
# (excludes generation). The two IDs never matched → 0 records promoted.

def test_promote_uses_evaluation_context_id_not_search_context(tmp_path: Path) -> None:
    """Using evaluation_context_id for promotion must find the stored records."""
    from ..search_memory import evaluation_context_id

    # Minimal fake config that evaluation_context_id accepts.
    # No identity.json → falls back to stable_identity of the dir string.
    fake_config = {
        "asset_dir": str(tmp_path),
        "families": [{"id": "test-family", "target_panel": "tp"}],
        "transfer_panel": "xfr",
        "validation_panel": "val",
        "search": {},
    }
    eval_ctx = evaluation_context_id(fake_config)

    # Simulate what search_context_identity would produce: different ID because
    # it additionally hashes in generation, donor_manifest, checkpoint_evidence, etc.
    wrong_ctx = stable_identity("RLRMB-P16-SEARCH-CONTEXT-V1", {
        "controller": stable_identity("RLRMB-P165-ASSET-FALLBACK-V1", str(tmp_path)),
        "generation": 0,
        "family_id": "test-family",
    })
    assert eval_ctx != wrong_ctx, "test precondition: eval ctx != search ctx"

    sids = [_sid(f"p20-prog-{i}") for i in range(3)]
    ej = effect_to_json(_effect())

    db = tmp_path / "mem.db"
    with SearchMemory(db) as mem:
        for sid in sids:
            mem.record_evaluation(sid, eval_ctx, valid=True, effect_json=ej, cuda_games=5)

        # Using the wrong context (old bug): promotes 0 records
        promoted_wrong = mem.promote_to_historical(wrong_ctx, new_generation=1)
        assert promoted_wrong == 0, "wrong ctx must find nothing"

        # Using the correct evaluation context: promotes all records
        promoted_right = mem.promote_to_historical(eval_ctx, new_generation=1)
        assert promoted_right == len(sids)

        # Historical records were created
        for sid in sids:
            priors = mem.get_historical_priors(sid)
            assert len(priors) == 1
            assert priors[0]["generation"] == 0

        # New-generation context (different controller hash) has no exact hit
        new_ctx = _ctx("new-gen-controller")
        assert mem.get_exact_evaluation(sids[0], new_ctx) is None


# ── test 21 (regression): corrupt effect_json raises, does not silently drop ────
# Defect 2: corrupt effect_json was caught silently; the program disappeared from
# the result because it ended up in neither new_by_sid nor cached_map.

def test_corrupt_effect_json_fails_closed(tmp_path: Path) -> None:
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

    prog = _make_prog(5)
    eff = _effect()

    class _FakeInner:
        case_count = 10
        last_profile: dict = {}
        calls: int = 0

        def projected_games(self, count: int) -> int:
            return (count + 1) * self.case_count

        def __call__(self, programs, motifs):
            self.calls += 1
            evaluated = tuple(
                EvaluatedProgram(program=p, effect=eff, valid=True, fault=None)
                for p in programs
            )
            return BatchEvaluation(evaluated, self.case_count,
                                   len(programs) * self.case_count, 1, 0.1, 5.0)

    db = tmp_path / "mem.db"
    ctx = _ctx("corrupt-ctx")

    with SearchMemory(db) as mem:
        # Store a record with corrupt effect_json (looks like a cache hit, but won't parse)
        mem._conn.execute(
            "INSERT INTO evaluation"
            " (semantic_id, context_id, generation, valid, fault,"
            "  effect_json, failure_class, cuda_games, insert_time)"
            " VALUES (?,?,?,?,?,?,?,?,?)",
            (prog.semantic_id, ctx, 0, 1, None, "CORRUPT{{NOT_JSON", None, 10, 0),
        )

        fake = _FakeInner()
        mbe = MemoryBackedEvaluator(fake, mem, ctx)

        # Must raise — fail closed, not silently drop
        with pytest.raises(RuntimeError, match="corrupt effect_json"):
            mbe([prog], {})

        # Inner evaluator was never called (corrupt prog went to cache path, not new_programs)
        assert fake.calls == 0
        # No false "avoided" bookkeeping was recorded
        assert mem.avoided_cuda_games == 0
        assert mem.avoided_candidate_evaluations == 0


# ── Phase 16.1–16.6 integration hardening regressions ───────────────────────

def test_historical_region_identity_survives_controller_change(tmp_path: Path) -> None:
    """Historical priors use a stable search-region identity, not exact eval context."""
    from .. import search_memory as sm

    assert hasattr(sm, "historical_region_id"), "stable historical region identity API is missing"
    region_id_fn = getattr(sm, "historical_region_id")

    old_assets = tmp_path / "assets-old"
    new_assets = tmp_path / "assets-new"
    old_assets.mkdir()
    new_assets.mkdir()
    (old_assets / "identity.json").write_text(
        json.dumps({"controller_semantic_hash": "controller-old"}), encoding="utf-8"
    )
    (new_assets / "identity.json").write_text(
        json.dumps({"controller_semantic_hash": "controller-new"}), encoding="utf-8"
    )
    family = {
        "id": "family-stable",
        "target_panel": "target.tsv",
        "checkpoint_evidence": "checkpoint.jsonl",
    }
    common = {
        "families": [family],
        "donor_manifest": "donors.txt",
        "transfer_panel": "transfer.tsv",
        "validation_panel": "validation.tsv",
        "search": {"beam_depth": 3, "motif_steps": 48},
    }
    old_cfg = {**common, "asset_dir": str(old_assets)}
    new_cfg = {**common, "asset_dir": str(new_assets)}

    old_eval = sm.evaluation_context_id(old_cfg)
    new_eval = sm.evaluation_context_id(new_cfg)
    old_region = region_id_fn(old_cfg)
    new_region = region_id_fn(new_cfg)
    assert old_eval != new_eval
    assert old_region == new_region

    sid = _sid("stable-region")
    db = tmp_path / "region.db"
    with SearchMemory(db) as mem:
        mem.record_evaluation(
            sid, old_eval, generation=0, valid=True,
            effect_json=effect_to_json(_effect()), cuda_games=5,
        )
        assert mem.get_exact_evaluation(sid, new_eval) is None
        assert hasattr(mem, "get_region_historical_stats")
        promoted = mem.promote_to_historical(
            old_eval, new_generation=1, region_id=old_region
        )
        assert promoted == 1
        before = mem.get_region_historical_stats(new_region)
        assert before == {"total": 1, "failures": 0}

        # SEALED audit rows are outside every allocator-visible region query.
        mem.record_sealed_audit(_sid("sealed-region"), new_eval, passed=False)
        assert mem.get_region_historical_stats(new_region) == before


def test_region_allocation_history_persists_restart(tmp_path: Path) -> None:
    """Completed region visits remain durable across allocator/process restart."""
    db = tmp_path / "allocation.db"
    region = _ctx("stable-region-allocation")
    with SearchMemory(db) as mem:
        assert hasattr(mem, "record_region_allocation")
        mem.record_region_allocation(region, generation=0, cuda_games=11)
        mem.record_region_allocation(region, generation=1, cuda_games=7)
        assert mem.get_region_allocation_stats(region) == {
            "selections": 2,
            "last_generation": 1,
            "cuda_games": 18,
        }

    with SearchMemory(db) as mem:
        assert mem.get_region_allocation_stats(region) == {
            "selections": 2,
            "last_generation": 1,
            "cuda_games": 18,
        }


def test_cached_invalid_exact_failure_avoids_cuda(tmp_path: Path) -> None:
    """A completed invalid same-context evaluation is an exact cache hit too."""
    from types import SimpleNamespace
    from ..p13_equal_budget import MemoryBackedEvaluator
    from ..beam import BatchEvaluation, EvaluatedProgram

    prog = SimpleNamespace(semantic_id=_sid("invalid-cache-hit"))

    class _FakeInner:
        case_count = 10
        last_profile: dict = {}

        def __init__(self) -> None:
            self.calls = 0

        def projected_games(self, count: int) -> int:
            return (count + 1) * self.case_count

        def __call__(self, programs, motifs):
            self.calls += 1
            return BatchEvaluation(
                tuple(EvaluatedProgram(p, None, True, None) for p in programs),
                self.case_count,
                len(programs) * self.case_count,
                1,
                0.1,
                0.0,
            )

    ctx = _ctx("invalid-cache-context")
    db = tmp_path / "invalid-cache.db"
    with SearchMemory(db) as mem:
        mem.record_evaluation(
            prog.semantic_id,
            ctx,
            generation=2,
            valid=False,
            fault="CUDA_FAULT",
            failure_class="CUDA_FAULT",
            effect_json=None,
            cuda_games=10,
        )
        fake = _FakeInner()
        result = MemoryBackedEvaluator(fake, mem, ctx, generation=2)([prog], {})

        assert fake.calls == 0
        assert result.games == 0
        assert len(result.programs) == 1
        assert result.programs[0].valid is False
        assert result.programs[0].fault == "CUDA_FAULT"
        assert mem.avoided_cuda_games == 10
        assert mem.avoided_candidate_evaluations == 1


def test_v1_historical_row_is_backfilled_when_region_becomes_known(tmp_path: Path) -> None:
    """A pre-hardening DB can attach existing history to its stable region on promotion."""
    db = tmp_path / "v1-memory.db"
    sid = _sid("v1-existing-history")
    ctx = _ctx("v1-old-context")
    region = _ctx("v1-stable-region")
    conn = sqlite3.connect(str(db))
    conn.execute("CREATE TABLE schema_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    conn.execute("INSERT INTO schema_meta VALUES ('schema_version', '1')")
    conn.execute("CREATE TABLE structural (semantic_id TEXT PRIMARY KEY, lane TEXT, operator TEXT, generation_first INTEGER NOT NULL DEFAULT 0, context_id_first TEXT, insert_time INTEGER NOT NULL)")
    conn.execute("CREATE TABLE evaluation (semantic_id TEXT NOT NULL, context_id TEXT NOT NULL, generation INTEGER NOT NULL DEFAULT 0, valid INTEGER NOT NULL DEFAULT 1, fault TEXT, effect_json TEXT, failure_class TEXT, cuda_games INTEGER NOT NULL DEFAULT 0, insert_time INTEGER NOT NULL, PRIMARY KEY (semantic_id, context_id))")
    conn.execute("CREATE TABLE historical (semantic_id TEXT NOT NULL, context_id TEXT NOT NULL, generation INTEGER NOT NULL, failure_class TEXT, outcome_summary TEXT, insert_time INTEGER NOT NULL, PRIMARY KEY (semantic_id, context_id, generation))")
    conn.execute("CREATE TABLE sealed_audit (semantic_id TEXT NOT NULL, context_id TEXT NOT NULL, passed INTEGER NOT NULL DEFAULT 0, insert_time INTEGER NOT NULL, PRIMARY KEY (semantic_id, context_id))")
    conn.execute(
        "INSERT INTO evaluation VALUES (?,?,?,?,?,?,?,?,?)",
        (sid, ctx, 0, 1, None, effect_to_json(_effect()), None, 5, 0),
    )
    conn.execute(
        "INSERT INTO historical VALUES (?,?,?,?,?,?)",
        (sid, ctx, 0, None, '{}', 0),
    )
    conn.commit()
    conn.close()

    with SearchMemory(db) as mem:
        assert mem.get_region_historical_stats(region) == {"total": 0, "failures": 0}
        assert mem.promote_to_historical(ctx, new_generation=1, region_id=region) == 1
        assert mem.get_region_historical_stats(region) == {"total": 1, "failures": 0}


def test_memory_batch_cuda_cost_is_not_multiplied_per_candidate(tmp_path: Path) -> None:
    """Context memory must sum to actual batch games, not batch_games × candidates."""
    from types import SimpleNamespace
    from ..p13_equal_budget import MemoryBackedEvaluator
    from ..beam import BatchEvaluation, EvaluatedProgram

    programs = [
        SimpleNamespace(semantic_id=_sid("batch-cost-a")),
        SimpleNamespace(semantic_id=_sid("batch-cost-b")),
    ]
    eff = _effect()

    class _FakeInner:
        case_count = 10
        last_profile: dict = {}

        def projected_games(self, count: int) -> int:
            return (count + 1) * self.case_count

        def __call__(self, rows, motifs):
            return BatchEvaluation(
                tuple(EvaluatedProgram(p, eff, True, None) for p in rows),
                self.case_count,
                30,
                1,
                0.1,
                0.0,
            )

    ctx = _ctx("batch-cost-context")
    with SearchMemory(tmp_path / "batch-cost.db") as mem:
        result = MemoryBackedEvaluator(_FakeInner(), mem, ctx)(programs, {})
        assert result.games == 30
        stats = mem.get_context_evaluation_stats(ctx)
        assert stats["total_cuda_games"] == 30


def test_safe_feedback_is_separate_from_exact_evaluation_truth(tmp_path: Path) -> None:
    """Pre-SEALED routability can guide allocation without mutating exact cache truth."""
    sid = _sid("future-dependent-row")
    ctx = _ctx("future-dependent-context")
    ej = effect_to_json(_effect())
    with SearchMemory(tmp_path / "future-dependent.db") as mem:
        mem.record_evaluation(
            sid, ctx, generation=4, valid=True, effect_json=ej, cuda_games=5
        )
        mem.record_search_feedback(
            sid,
            ctx,
            generation=4,
            stage="FULL_ROUTABILITY",
            failure_class="FUTURE_DEPENDENT",
        )

        exact = mem.get_exact_evaluation(sid, ctx)
        assert exact is not None
        assert exact["valid"] is True
        assert exact["fault"] is None
        assert exact["failure_class"] is None
        assert exact["effect_json"] == ej

        stats = mem.get_context_evaluation_stats(ctx)
        assert stats["future_dependent_count"] == 1
        assert stats["other_failure_count"] == 0
        assert stats["routability_observed_count"] == 1
        assert stats["routable_count"] == 0


def test_safe_feedback_persists_restart_and_promotes_effective_failure(tmp_path: Path) -> None:
    """Durable pre-SEALED failure evidence survives restart and becomes historical prior."""
    db = tmp_path / "feedback-restart.db"
    sid = _sid("feedback-restart")
    ctx = _ctx("feedback-restart-context")
    region = _ctx("feedback-stable-region")
    ej = effect_to_json(_effect())

    with SearchMemory(db) as mem:
        mem.record_evaluation(
            sid, ctx, generation=2, valid=True, effect_json=ej, cuda_games=7
        )
        mem.record_search_feedback(
            sid,
            ctx,
            generation=2,
            stage="TARGET_ROUTABILITY",
            failure_class="FUTURE_DEPENDENT",
        )

    with SearchMemory(db) as mem:
        stats = mem.get_context_evaluation_stats(ctx)
        assert stats["future_dependent_count"] == 1
        assert stats["routability_observed_count"] == 1
        assert mem.promote_to_historical(ctx, new_generation=3, region_id=region) == 1
        assert mem.get_region_historical_stats(region) == {"total": 1, "failures": 1}
        prior = mem.get_historical_priors(sid)[0]
        assert prior["failure_class"] == "FUTURE_DEPENDENT"


def test_search_feedback_stage_whitelist_and_sealed_firewall(tmp_path: Path) -> None:
    """Only safe routability stages are allocator-visible; SEALED/CEGIS stay isolated."""
    sid = _sid("feedback-firewall")
    ctx = _ctx("feedback-firewall-context")
    with SearchMemory(tmp_path / "feedback-firewall.db") as mem:
        mem.record_evaluation(
            sid, ctx, generation=1, valid=True, effect_json=effect_to_json(_effect())
        )
        mem.record_search_feedback(
            sid,
            ctx,
            generation=1,
            stage="TARGET_ROUTABILITY",
            failure_class="FUTURE_DEPENDENT",
        )
        # Later full-panel evidence is authoritative for allocator feedback and may
        # replace the target-only classification without touching exact truth.
        mem.record_search_feedback(
            sid,
            ctx,
            generation=1,
            stage="FULL_ROUTABILITY",
            failure_class=None,
        )
        stats_before = mem.get_context_evaluation_stats(ctx)
        assert stats_before["future_dependent_count"] == 0
        assert stats_before["routability_observed_count"] == 1
        assert stats_before["routable_count"] == 1

        for forbidden in ("SEALED", "CEGIS", "CERTIFICATION"):
            with pytest.raises(ValueError, match="feedback stage"):
                mem.record_search_feedback(
                    sid,
                    ctx,
                    generation=1,
                    stage=forbidden,
                    failure_class="FUTURE_DEPENDENT",
                )

        mem.record_sealed_audit(sid, ctx, passed=False)
        assert mem.get_context_evaluation_stats(ctx) == stats_before


def test_production_routability_feedback_helper_records_only_safe_signal(tmp_path: Path) -> None:
    """The production helper records routability and rejects any non-pre-SEALED stage."""
    from types import SimpleNamespace
    from ..p13_equal_budget import _record_routability_feedback
    from ..routability import RoutabilityClassification

    sid = _sid("production-feedback-helper")
    ctx = _ctx("production-feedback-helper-context")
    program = SimpleNamespace(semantic_id=sid)
    with SearchMemory(tmp_path / "production-feedback-helper.db") as mem:
        mem.record_evaluation(
            sid, ctx, generation=5, valid=True, effect_json=effect_to_json(_effect())
        )
        _record_routability_feedback(
            mem,
            ctx,
            5,
            program,
            SimpleNamespace(classification=RoutabilityClassification.ROUTABLE_CANDIDATE),
            "TARGET_ROUTABILITY",
        )
        stats = mem.get_context_evaluation_stats(ctx)
        assert stats["routability_observed_count"] == 1
        assert stats["routable_count"] == 1
        assert stats["future_dependent_count"] == 0

        _record_routability_feedback(
            mem,
            ctx,
            5,
            program,
            SimpleNamespace(classification=RoutabilityClassification.FUTURE_DEPENDENT),
            "FULL_ROUTABILITY",
        )
        stats = mem.get_context_evaluation_stats(ctx)
        assert stats["routability_observed_count"] == 1
        assert stats["routable_count"] == 0
        assert stats["future_dependent_count"] == 1

        with pytest.raises(ValueError, match="feedback stage"):
            _record_routability_feedback(
                mem,
                ctx,
                5,
                program,
                SimpleNamespace(classification=RoutabilityClassification.FUTURE_DEPENDENT),
                "CEGIS",
            )
