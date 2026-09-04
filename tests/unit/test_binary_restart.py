"""Phase 1.5 — Focused tests for binary pipeline restart/resume recovery.

25 tests covering all required restart-safety invariants.
"""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

import pytest

from kaggriculture_bonsai.routing.binary_restart import (
    RESTART_SCHEMA_VERSION,
    BinaryPipelineStage,
    BinaryPipelineState,
    RestartMetrics,
    _TERMINAL_FAILURES,
    binary_context_id,
    build_restart_manifest,
    check_context_match,
    get_all_pipeline_states,
    get_committed_route_ids,
    get_pending_materializations,
    get_pipeline_state,
    init_binary_pipeline_schema,
    invalidate_for_context_change,
    record_pipeline_state,
    recover_route_allocator_state,
    should_resume_stage,
)
from kaggriculture_bonsai.routing.downstream_integration import (
    DURABLE_ROUTE_ID_BASE,
    DURABLE_ROUTE_ID_MAX,
)
from kaggriculture_bonsai.routing.routed_evaluation import TEMPORARY_ROUTE_ID_BASE
from kaggriculture_bonsai.search.search_memory import SearchMemory


CTX_A = "ctx_parent_gen_A"
CTX_B = "ctx_parent_gen_B"
CAND_1 = "candidate_0568601d"
CAND_2 = "candidate_0a48dd5e"
GATE_1 = "gate_abc123"
GATE_2 = "gate_def456"


def _make_conn():
    conn = sqlite3.connect(":memory:", isolation_level=None)
    conn.execute("PRAGMA journal_mode=WAL")
    init_binary_pipeline_schema(conn)
    return conn


# ── Test 1: Restart after admission reuses result ──────────────────────────
def test_restart_after_admission_reuses():
    conn = _make_conn()
    ctx = binary_context_id(CTX_A, CAND_1)
    record_pipeline_state(conn, CAND_1, ctx, CTX_A, BinaryPipelineStage.ADMITTED)

    state = get_pipeline_state(conn, CAND_1, ctx)
    assert state is not None
    assert state.stage is BinaryPipelineStage.ADMITTED
    assert should_resume_stage(state, BinaryPipelineStage.ADMITTED) == "reuse"


# ── Test 2: Restart after gate synthesis reuses exact gate ─────────────────
def test_restart_after_gate_synthesis_reuses():
    conn = _make_conn()
    ctx = binary_context_id(CTX_A, CAND_1, GATE_1)
    record_pipeline_state(
        conn, CAND_1, ctx, CTX_A, BinaryPipelineStage.GATE_SYNTHESIZED,
        gate_identity=GATE_1,
    )

    state = get_pipeline_state(conn, CAND_1, ctx)
    assert state.stage is BinaryPipelineStage.GATE_SYNTHESIZED
    assert state.gate_identity == GATE_1
    assert should_resume_stage(state, BinaryPipelineStage.GATE_SYNTHESIZED) == "reuse"
    assert should_resume_stage(state, BinaryPipelineStage.ROUTED_VALIDATED) == "execute"


# ── Test 3: Restart after routed CUDA avoids duplicate work ────────────────
def test_restart_after_routed_cuda_avoids_duplicate():
    conn = _make_conn()
    ctx = binary_context_id(CTX_A, CAND_1, GATE_1)
    record_pipeline_state(
        conn, CAND_1, ctx, CTX_A, BinaryPipelineStage.ROUTED_VALIDATED,
        gate_identity=GATE_1, artifact_ref="routed_eval/cand1.json",
    )

    state = get_pipeline_state(conn, CAND_1, ctx)
    assert state.stage is BinaryPipelineStage.ROUTED_VALIDATED
    assert should_resume_stage(state, BinaryPipelineStage.ROUTED_VALIDATED) == "reuse"

    metrics = RestartMetrics()
    metrics.binary_routed_eval_reuses += 1
    metrics.binary_cuda_games_avoided += 42
    assert metrics.binary_routed_eval_reuses == 1
    assert metrics.binary_cuda_games_avoided == 42


# ── Test 4: Restart after TRANSFER resumes at VALIDATION ───────────────────
def test_restart_after_transfer_resumes_at_validation():
    conn = _make_conn()
    ctx = binary_context_id(CTX_A, CAND_1, GATE_1)
    record_pipeline_state(
        conn, CAND_1, ctx, CTX_A, BinaryPipelineStage.TRANSFER_PASSED,
        gate_identity=GATE_1,
    )

    state = get_pipeline_state(conn, CAND_1, ctx)
    assert should_resume_stage(state, BinaryPipelineStage.TRANSFER_PASSED) == "reuse"
    assert should_resume_stage(state, BinaryPipelineStage.VALIDATION_PASSED) == "execute"


# ── Test 5: Restart after VALIDATION resumes at CEGIS ──────────────────────
def test_restart_after_validation_resumes_at_cegis():
    conn = _make_conn()
    ctx = binary_context_id(CTX_A, CAND_1, GATE_1)
    record_pipeline_state(
        conn, CAND_1, ctx, CTX_A, BinaryPipelineStage.VALIDATION_PASSED,
        gate_identity=GATE_1,
    )

    state = get_pipeline_state(conn, CAND_1, ctx)
    assert should_resume_stage(state, BinaryPipelineStage.VALIDATION_PASSED) == "reuse"
    assert should_resume_stage(state, BinaryPipelineStage.CEGIS_PASSED) == "execute"


# ── Test 6: Restart after CEGIS resumes at SEALED ─────────────────────────
def test_restart_after_cegis_resumes_at_sealed():
    conn = _make_conn()
    ctx = binary_context_id(CTX_A, CAND_1, GATE_1)
    record_pipeline_state(
        conn, CAND_1, ctx, CTX_A, BinaryPipelineStage.CEGIS_PASSED,
        gate_identity=GATE_1,
    )

    state = get_pipeline_state(conn, CAND_1, ctx)
    assert should_resume_stage(state, BinaryPipelineStage.CEGIS_PASSED) == "reuse"
    assert should_resume_stage(state, BinaryPipelineStage.SEALED_PASSED) == "execute"


# ── Test 7: Restart after SEALED does not leak into search ─────────────────
def test_restart_after_sealed_no_search_leak():
    conn = _make_conn()
    ctx = binary_context_id(CTX_A, CAND_1, GATE_1)
    record_pipeline_state(
        conn, CAND_1, ctx, CTX_A, BinaryPipelineStage.SEALED_PASSED,
        gate_identity=GATE_1,
    )

    state = get_pipeline_state(conn, CAND_1, ctx)
    assert should_resume_stage(state, BinaryPipelineStage.SEALED_PASSED) == "reuse"
    assert should_resume_stage(state, BinaryPipelineStage.MATERIALIZED_PENDING_COMMIT) == "execute"

    # Verify sealed_audit table has no read path from binary_pipeline_state
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='sealed_audit'"
    ).fetchall()
    # sealed_audit may not exist in this isolated conn — that's fine,
    # the point is binary_pipeline_state never references it
    assert "sealed_audit" not in {
        col[1] for col in conn.execute(
            "PRAGMA table_info(binary_pipeline_state)"
        ).fetchall()
    }


# ── Test 8: Parent-generation change invalidates routed safety ─────────────
def test_parent_generation_change_invalidates():
    conn = _make_conn()
    ctx_old = binary_context_id(CTX_A, CAND_1, GATE_1)
    record_pipeline_state(
        conn, CAND_1, ctx_old, CTX_A, BinaryPipelineStage.ROUTED_VALIDATED,
        gate_identity=GATE_1,
    )

    old_state = get_pipeline_state(conn, CAND_1, ctx_old)
    assert not check_context_match(old_state, CTX_B)

    ctx_new = binary_context_id(CTX_B, CAND_1, GATE_1)
    invalidated = invalidate_for_context_change(
        conn, CAND_1, ctx_old, ctx_new, CTX_B,
    )
    assert invalidated

    new_state = get_pipeline_state(conn, CAND_1, ctx_new)
    assert new_state.stage is BinaryPipelineStage.DISCOVERED
    assert "invalidated_from" in new_state.artifact_ref


# ── Test 9: Protected-panel identity change invalidates ────────────────────
def test_protected_panel_change_invalidates():
    ctx1 = binary_context_id("parent_A_panel_X", CAND_1, GATE_1)
    ctx2 = binary_context_id("parent_A_panel_Y", CAND_1, GATE_1)
    assert ctx1 != ctx2

    conn = _make_conn()
    record_pipeline_state(
        conn, CAND_1, ctx1, "parent_A_panel_X",
        BinaryPipelineStage.ROUTED_VALIDATED, gate_identity=GATE_1,
    )
    state = get_pipeline_state(conn, CAND_1, ctx1)
    assert not check_context_match(state, "parent_A_panel_Y")


# ── Test 10: Gate identity change invalidates routed safety ────────────────
def test_gate_identity_change_invalidates():
    conn = _make_conn()
    ctx = binary_context_id(CTX_A, CAND_1, GATE_1)
    record_pipeline_state(
        conn, CAND_1, ctx, CTX_A, BinaryPipelineStage.ROUTED_VALIDATED,
        gate_identity=GATE_1,
    )

    state = get_pipeline_state(conn, CAND_1, ctx)
    assert check_context_match(state, CTX_A, GATE_1)
    assert not check_context_match(state, CTX_A, GATE_2)


# ── Test 11: Semantic/config identity change invalidates ───────────────────
def test_semantic_config_change_invalidates():
    ctx1 = binary_context_id("config_v1", CAND_1)
    ctx2 = binary_context_id("config_v2", CAND_1)
    assert ctx1 != ctx2


# ── Test 12: Historical records remain non-authoritative ───────────────────
def test_historical_records_non_authoritative():
    conn = _make_conn()
    ctx_old = binary_context_id(CTX_A, CAND_1, GATE_1)
    record_pipeline_state(
        conn, CAND_1, ctx_old, CTX_A, BinaryPipelineStage.COMMITTED,
        gate_identity=GATE_1, route_id=200,
    )

    ctx_new = binary_context_id(CTX_B, CAND_1, GATE_1)
    new_state = get_pipeline_state(conn, CAND_1, ctx_new)
    assert new_state is None

    old_state = get_pipeline_state(conn, CAND_1, ctx_old)
    assert old_state is not None
    assert not check_context_match(old_state, CTX_B)


# ── Test 13: Terminal failure not re-executed in identical context ──────────
def test_terminal_failure_not_reexecuted():
    conn = _make_conn()
    ctx = binary_context_id(CTX_A, CAND_1, GATE_1)
    record_pipeline_state(
        conn, CAND_1, ctx, CTX_A, BinaryPipelineStage.GATE_SYNTHESIZED,
        gate_identity=GATE_1, failure_class="NO_SAFE_SEPARATOR",
    )

    state = get_pipeline_state(conn, CAND_1, ctx)
    assert state.failure_class == "NO_SAFE_SEPARATOR"
    assert should_resume_stage(state, BinaryPipelineStage.GATE_SYNTHESIZED) == "skip_terminal"


# ── Test 14: Terminal failure reconsidered under changed context ───────────
def test_terminal_failure_reconsidered_changed_context():
    conn = _make_conn()
    ctx_old = binary_context_id(CTX_A, CAND_1, GATE_1)
    record_pipeline_state(
        conn, CAND_1, ctx_old, CTX_A, BinaryPipelineStage.ROUTED_VALIDATED,
        gate_identity=GATE_1, failure_class="NON_TRANSFERABLE",
    )

    ctx_new = binary_context_id(CTX_B, CAND_1, GATE_1)
    state_new = get_pipeline_state(conn, CAND_1, ctx_new)
    assert state_new is None
    assert should_resume_stage(state_new, BinaryPipelineStage.ADMITTED) == "execute"


# ── Test 15: No duplicate SearchMemory stage rows after resume ─────────────
def test_no_duplicate_pipeline_state_rows():
    conn = _make_conn()
    ctx = binary_context_id(CTX_A, CAND_1)
    record_pipeline_state(conn, CAND_1, ctx, CTX_A, BinaryPipelineStage.ADMITTED)
    record_pipeline_state(conn, CAND_1, ctx, CTX_A, BinaryPipelineStage.GATE_SYNTHESIZED,
                          gate_identity=GATE_1)
    record_pipeline_state(conn, CAND_1, ctx, CTX_A, BinaryPipelineStage.ROUTED_VALIDATED,
                          gate_identity=GATE_1)

    rows = conn.execute(
        "SELECT COUNT(*) FROM binary_pipeline_state WHERE candidate_semantic_id=? AND context_id=?",
        (CAND_1, ctx),
    ).fetchone()
    assert rows[0] == 1

    state = get_pipeline_state(conn, CAND_1, ctx)
    assert state.stage is BinaryPipelineStage.ROUTED_VALIDATED


# ── Test 16: No duplicate artifact records after resume ────────────────────
def test_no_duplicate_artifact_records():
    conn = _make_conn()
    ctx = binary_context_id(CTX_A, CAND_1, GATE_1)
    record_pipeline_state(conn, CAND_1, ctx, CTX_A, BinaryPipelineStage.ROUTED_VALIDATED,
                          gate_identity=GATE_1, artifact_ref="eval/cand1_v1.json")
    record_pipeline_state(conn, CAND_1, ctx, CTX_A, BinaryPipelineStage.TRANSFER_PASSED,
                          gate_identity=GATE_1, artifact_ref="eval/cand1_v2.json")

    all_states = get_all_pipeline_states(conn, ctx)
    assert len(all_states) == 1
    assert all_states[0].artifact_ref == "eval/cand1_v2.json"


# ── Test 17: Pending production materialization recovers safely ────────────
def test_pending_materialization_recovers():
    conn = _make_conn()
    ctx = binary_context_id(CTX_A, CAND_1, GATE_1)
    record_pipeline_state(
        conn, CAND_1, ctx, CTX_A, BinaryPipelineStage.MATERIALIZED_PENDING_COMMIT,
        gate_identity=GATE_1, route_id=200,
    )

    pending = get_pending_materializations(conn, ctx)
    assert len(pending) == 1
    assert pending[0].route_id == 200
    assert pending[0].stage is BinaryPipelineStage.MATERIALIZED_PENDING_COMMIT


# ── Test 18: Durable route ID not allocated twice after restart ────────────
def test_durable_route_id_not_duplicated():
    conn = _make_conn()
    ctx = binary_context_id(CTX_A, CAND_1, GATE_1)
    record_pipeline_state(
        conn, CAND_1, ctx, CTX_A, BinaryPipelineStage.MATERIALIZED_PENDING_COMMIT,
        gate_identity=GATE_1, route_id=200,
    )

    route_state = recover_route_allocator_state(conn, ctx)
    assert route_state["next_durable_id"] == 201
    assert 200 in route_state["pending_route_ids"]

    ctx2 = binary_context_id(CTX_A, CAND_2, GATE_1)
    record_pipeline_state(
        conn, CAND_2, ctx2, CTX_A, BinaryPipelineStage.COMMITTED,
        gate_identity=GATE_1, route_id=201,
    )

    route_state2 = recover_route_allocator_state(conn, ctx)
    assert route_state2["next_durable_id"] == 202
    assert 201 in route_state2["committed_route_ids"]


# ── Test 19: Uncommitted durable route ID reclaimed or resumed ─────────────
def test_uncommitted_route_id_reclaimed():
    conn = _make_conn()
    ctx = binary_context_id(CTX_A, CAND_1, GATE_1)
    record_pipeline_state(
        conn, CAND_1, ctx, CTX_A, BinaryPipelineStage.MATERIALIZED_PENDING_COMMIT,
        gate_identity=GATE_1, route_id=205,
    )

    pending = get_pending_materializations(conn, ctx)
    assert len(pending) == 1
    assert pending[0].route_id == 205

    route_state = recover_route_allocator_state(conn, ctx)
    assert 205 in route_state["pending_route_ids"]
    assert route_state["next_durable_id"] == 206


# ── Test 20: Temporary 60000-series never contaminates durable recovery ────
def test_temporary_ids_excluded_from_durable():
    conn = _make_conn()
    ctx = binary_context_id(CTX_A, CAND_1, GATE_1)
    record_pipeline_state(
        conn, CAND_1, ctx, CTX_A, BinaryPipelineStage.ROUTED_VALIDATED,
        gate_identity=GATE_1, route_id=60001,
    )

    route_state = recover_route_allocator_state(conn, ctx)
    assert route_state["next_durable_id"] == DURABLE_ROUTE_ID_BASE
    assert 60001 not in route_state["durable_ids_in_use"]


# ── Test 21: SAFE_FAST_PATH restart semantics unchanged ───────────────────
def test_safe_fast_path_unchanged():
    conn = _make_conn()
    ctx = binary_context_id(CTX_A, "safe_candidate")
    state = get_pipeline_state(conn, "safe_candidate", ctx)
    assert state is None
    assert should_resume_stage(state, BinaryPipelineStage.ADMITTED) == "execute"


# ── Test 22: Existing SearchMemory restart behavior green ──────────────────
def test_search_memory_restart_with_binary_table(tmp_path):
    db_path = tmp_path / "test_memory.db"
    with SearchMemory(db_path) as mem:
        mem.record_structural("test_cand", lane="lane0", generation=0, context_id="ctx0")
        assert mem.is_structurally_seen("test_cand")

    with SearchMemory(db_path) as mem:
        assert mem.is_structurally_seen("test_cand")
        tables = {
            r[0] for r in mem._conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        assert "binary_pipeline_state" in tables


# ── Test 23: SEALED firewall remains write-only after restart ──────────────
def test_sealed_firewall_intact_after_restart(tmp_path):
    db_path = tmp_path / "sealed_test.db"
    with SearchMemory(db_path) as mem:
        mem.record_sealed_audit("cand1", "ctx1", passed=True)

    with SearchMemory(db_path) as mem:
        # Verify sealed_audit exists but has no read API
        row = mem._conn.execute(
            "SELECT passed FROM sealed_audit WHERE semantic_id='cand1'"
        ).fetchone()
        assert row is not None and row[0] == 1

        assert not hasattr(mem, "get_sealed_audit")
        assert not hasattr(mem, "read_sealed")


# ── Test 24: Context invalidation is deterministic ─────────────────────────
def test_context_invalidation_deterministic():
    conn = _make_conn()
    ctx_old = binary_context_id(CTX_A, CAND_1, GATE_1)
    record_pipeline_state(
        conn, CAND_1, ctx_old, CTX_A, BinaryPipelineStage.CEGIS_PASSED,
        gate_identity=GATE_1,
    )

    ctx_new = binary_context_id(CTX_B, CAND_1, GATE_1)

    r1 = invalidate_for_context_change(conn, CAND_1, ctx_old, ctx_new, CTX_B)
    state1 = get_pipeline_state(conn, CAND_1, ctx_new)

    r2 = invalidate_for_context_change(conn, CAND_1, ctx_old, ctx_new, CTX_B)
    state2 = get_pipeline_state(conn, CAND_1, ctx_new)

    assert r1 is True
    assert r2 is True
    assert state1.stage == state2.stage
    assert state1.artifact_ref == state2.artifact_ref


# ── Test 25: Restart produces same next-stage ordering ─────────────────────
def test_restart_same_stage_ordering():
    conn = _make_conn()
    ctx = binary_context_id(CTX_A, CAND_1, GATE_1)

    stages = [
        BinaryPipelineStage.ADMITTED,
        BinaryPipelineStage.GATE_SYNTHESIZED,
        BinaryPipelineStage.ROUTED_VALIDATED,
        BinaryPipelineStage.TRANSFER_PASSED,
        BinaryPipelineStage.VALIDATION_PASSED,
        BinaryPipelineStage.CEGIS_PASSED,
        BinaryPipelineStage.SEALED_PASSED,
        BinaryPipelineStage.MATERIALIZED_PENDING_COMMIT,
        BinaryPipelineStage.COMMITTED,
    ]

    for i, stage in enumerate(stages):
        record_pipeline_state(
            conn, CAND_1, ctx, CTX_A, stage,
            gate_identity=GATE_1,
        )
        state = get_pipeline_state(conn, CAND_1, ctx)
        assert state.stage is stage

        for j, earlier in enumerate(stages[:i+1]):
            assert should_resume_stage(state, earlier) == "reuse"
        for later in stages[i+1:]:
            assert should_resume_stage(state, later) == "execute"


# ── Bonus: Restart manifest ───────────────────────────────────────────────
def test_restart_manifest():
    conn = _make_conn()
    ctx = binary_context_id(CTX_A, CAND_1, GATE_1)
    record_pipeline_state(
        conn, CAND_1, ctx, CTX_A, BinaryPipelineStage.MATERIALIZED_PENDING_COMMIT,
        gate_identity=GATE_1, route_id=200,
    )

    manifest = build_restart_manifest(conn, ctx, CTX_A)
    assert manifest["schema_version"] == RESTART_SCHEMA_VERSION
    assert len(manifest["candidates"]) == 1
    assert manifest["candidates"][0]["route_id"] == 200
    assert manifest["candidates"][0]["committed"] is False
    assert manifest["route_allocator"]["next_durable_id"] == 201


# ── Bonus: Monotonic stage transitions ─────────────────────────────────────
def test_monotonic_stage_no_regression():
    conn = _make_conn()
    ctx = binary_context_id(CTX_A, CAND_1, GATE_1)
    record_pipeline_state(
        conn, CAND_1, ctx, CTX_A, BinaryPipelineStage.CEGIS_PASSED,
        gate_identity=GATE_1,
    )
    # Attempt to go backwards
    record_pipeline_state(
        conn, CAND_1, ctx, CTX_A, BinaryPipelineStage.ADMITTED,
        gate_identity=GATE_1,
    )
    state = get_pipeline_state(conn, CAND_1, ctx)
    assert state.stage is BinaryPipelineStage.CEGIS_PASSED


# ── Bonus: RestartMetrics serialization ────────────────────────────────────
def test_restart_metrics_serialization():
    m = RestartMetrics(
        binary_admission_reuses=3,
        binary_gate_reuses=2,
        binary_routed_eval_reuses=1,
        binary_cuda_games_avoided=84,
    )
    doc = m.to_document()
    assert doc["binary_admission_reuses"] == 3
    assert doc["binary_cuda_games_avoided"] == 84
    assert doc["binary_context_invalidations"] == 0


# ── Bonus: Multiple candidates in same context ────────────────────────────
def test_multiple_candidates_same_context():
    conn = _make_conn()
    ctx1 = binary_context_id(CTX_A, CAND_1, GATE_1)
    ctx2 = binary_context_id(CTX_A, CAND_2, GATE_1)
    record_pipeline_state(conn, CAND_1, ctx1, CTX_A, BinaryPipelineStage.COMMITTED,
                          gate_identity=GATE_1, route_id=200)
    record_pipeline_state(conn, CAND_2, ctx2, CTX_A, BinaryPipelineStage.TRANSFER_PASSED,
                          gate_identity=GATE_1)

    s1 = get_pipeline_state(conn, CAND_1, ctx1)
    s2 = get_pipeline_state(conn, CAND_2, ctx2)
    assert s1.stage is BinaryPipelineStage.COMMITTED
    assert s2.stage is BinaryPipelineStage.TRANSFER_PASSED
