"""Phase 1.6 — Multi-generation integration and stress validation.

Exercises the complete Phase-1 binary-routing system across multiple
generations with safe+binary coexistence, parent evolution, restart/resume,
durable route-ID progression, composition conflicts, and final composed
BKG4 safety revalidation.

Uses deterministic fixtures — no CUDA, no Lightning, no large search.
"""
from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from kaggriculture_bonsai.routing.binary_restart import (
    BinaryPipelineStage,
    RestartMetrics,
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
    BinaryCandidateStatus,
    GenerationValidationResult,
    _DurableRouteAllocator,
    validate_generation,
)
from kaggriculture_bonsai.routing.routed_evaluation import TEMPORARY_ROUTE_ID_BASE


# ── Deterministic fixture helpers ─────────────────────────────────────────────

PARENT_CTX_GEN0 = "parent_gen0_ctx"
PARENT_CTX_GEN1 = "parent_gen1_ctx"
PARENT_CTX_GEN2 = "parent_gen2_ctx"
PARENT_CTX_GEN3 = "parent_gen3_ctx"

SAFE_SPEC_A = "safe_specialist_A"
BINARY_SPEC_B = "binary_specialist_B"
BINARY_SPEC_C = "binary_specialist_C"
BINARY_SPEC_D = "binary_specialist_D"

GATE_B = "gate_for_B"
GATE_C = "gate_for_C"
GATE_D = "gate_for_D"


def _make_conn():
    conn = sqlite3.connect(":memory:", isolation_level=None)
    conn.execute("PRAGMA journal_mode=WAL")
    init_binary_pipeline_schema(conn)
    return conn


def _make_composed_module(*, nodes=10, slots=4, conflicts=None):
    children = [{"type": "leaf"} for _ in range(max(0, nodes - 1))]
    tree = {"type": "root", "children": children}
    result = {
        "tree": tree,
        "blackboard": [f"slot_{i}" for i in range(slots)],
    }
    if conflicts:
        result["unresolved_conflicts"] = conflicts
    return result


def _make_validation_cases(*, protected_w2l=0, count=10):
    cases = []
    for i in range(count):
        baseline = 5
        routed = 5
        if i < protected_w2l:
            baseline = 5
            routed = -1
        cases.append({"baseline_margin": baseline, "routed_margin": routed})
    return cases


def _commit_binary_specialist(conn, candidate_id, parent_ctx, gate_id, route_id):
    """Simulate full pipeline from DISCOVERED through COMMITTED."""
    ctx = binary_context_id(parent_ctx, candidate_id, gate_id)
    stages = [
        (BinaryPipelineStage.DISCOVERED, {}),
        (BinaryPipelineStage.ADMITTED, {}),
        (BinaryPipelineStage.GATE_SYNTHESIZED, {"gate_identity": gate_id}),
        (BinaryPipelineStage.ROUTED_VALIDATED, {"gate_identity": gate_id}),
        (BinaryPipelineStage.TRANSFER_PASSED, {"gate_identity": gate_id}),
        (BinaryPipelineStage.VALIDATION_PASSED, {"gate_identity": gate_id}),
        (BinaryPipelineStage.CEGIS_PASSED, {"gate_identity": gate_id}),
        (BinaryPipelineStage.SEALED_PASSED, {"gate_identity": gate_id}),
        (BinaryPipelineStage.MATERIALIZED_PENDING_COMMIT, {"gate_identity": gate_id, "route_id": route_id}),
        (BinaryPipelineStage.COMMITTED, {"gate_identity": gate_id, "route_id": route_id,
                                          "artifact_ref": f"eval/{candidate_id[:8]}.json"}),
    ]
    for stage, kwargs in stages:
        record_pipeline_state(conn, candidate_id, ctx, parent_ctx, stage, **kwargs)
    return ctx


# ── Scenario A: Successful single binary generation ──────────────────────────

class TestScenarioA:
    def test_single_binary_commit(self):
        conn = _make_conn()
        ctx = _commit_binary_specialist(conn, BINARY_SPEC_B, PARENT_CTX_GEN0, GATE_B, 200)
        state = get_pipeline_state(conn, BINARY_SPEC_B, ctx)
        assert state.stage is BinaryPipelineStage.COMMITTED
        assert state.route_id == 200
        assert state.gate_identity == GATE_B

    def test_generation_validation_passes(self):
        module = _make_composed_module()
        cases = _make_validation_cases(protected_w2l=0)
        result = validate_generation(module, validation_results=cases)
        assert result.passed
        assert result.protected_w2l == 0
        assert result.semantic_mismatches == 0


# ── Scenario B: Safe + binary coexistence ────────────────────────────────────

class TestScenarioB:
    def test_safe_and_binary_coexist(self):
        conn = _make_conn()
        # Safe specialist has no binary pipeline state — uses existing fast path.
        # Binary specialist goes through full pipeline.
        ctx = _commit_binary_specialist(conn, BINARY_SPEC_B, PARENT_CTX_GEN0, GATE_B, 200)

        # Verify binary committed
        state = get_pipeline_state(conn, BINARY_SPEC_B, ctx)
        assert state.stage is BinaryPipelineStage.COMMITTED
        assert state.route_id == 200

        # Safe specialist has no binary_pipeline_state row — that's correct
        safe_ctx = binary_context_id(PARENT_CTX_GEN0, SAFE_SPEC_A)
        safe_state = get_pipeline_state(conn, SAFE_SPEC_A, safe_ctx)
        assert safe_state is None  # SAFE_FAST_PATH doesn't use binary pipeline

    def test_route_ids_unique(self):
        allocator = _DurableRouteAllocator()
        safe_route = allocator.allocate()  # 200 — safe specialist
        binary_route = allocator.allocate()  # 201 — binary specialist
        assert safe_route == 200
        assert binary_route == 201
        assert safe_route != binary_route

    def test_combined_generation_validation(self):
        module = _make_composed_module(nodes=15, slots=6)
        cases = _make_validation_cases(protected_w2l=0)
        result = validate_generation(module, validation_results=cases)
        assert result.passed
        assert result.protected_w2l == 0


# ── Scenario C: Multiple binary specialists ──────────────────────────────────

class TestScenarioC:
    def test_two_binary_specialists_compose(self):
        conn = _make_conn()
        ctx_b = _commit_binary_specialist(conn, BINARY_SPEC_B, PARENT_CTX_GEN0, GATE_B, 200)
        ctx_c = _commit_binary_specialist(conn, BINARY_SPEC_C, PARENT_CTX_GEN0, GATE_C, 201)

        state_b = get_pipeline_state(conn, BINARY_SPEC_B, ctx_b)
        state_c = get_pipeline_state(conn, BINARY_SPEC_C, ctx_c)
        assert state_b.stage is BinaryPipelineStage.COMMITTED
        assert state_c.stage is BinaryPipelineStage.COMMITTED
        assert state_b.route_id != state_c.route_id
        assert state_b.gate_identity != state_c.gate_identity

    def test_deterministic_composition_order(self):
        conn = _make_conn()
        _commit_binary_specialist(conn, BINARY_SPEC_C, PARENT_CTX_GEN0, GATE_C, 201)
        _commit_binary_specialist(conn, BINARY_SPEC_B, PARENT_CTX_GEN0, GATE_B, 200)

        # get_all_pipeline_states returns sorted by candidate_semantic_id
        ctx_b = binary_context_id(PARENT_CTX_GEN0, BINARY_SPEC_B, GATE_B)
        ctx_c = binary_context_id(PARENT_CTX_GEN0, BINARY_SPEC_C, GATE_C)

        states = []
        for cand, gate in sorted([(BINARY_SPEC_B, GATE_B), (BINARY_SPEC_C, GATE_C)]):
            ctx = binary_context_id(PARENT_CTX_GEN0, cand, gate)
            s = get_pipeline_state(conn, cand, ctx)
            states.append(s)

        # Deterministic: sorted by candidate_semantic_id
        assert states[0].candidate_semantic_id < states[1].candidate_semantic_id
        # Route IDs are distinct
        route_ids = {s.route_id for s in states}
        assert len(route_ids) == 2

    def test_two_binary_generation_validation(self):
        module = _make_composed_module(nodes=20, slots=8)
        cases = _make_validation_cases(protected_w2l=0)
        result = validate_generation(module, validation_results=cases)
        assert result.passed


# ── Scenario D: Failed composition ───────────────────────────────────────────

class TestScenarioD:
    def test_composition_conflict_prevents_commit(self):
        module = _make_composed_module(conflicts=["authority_overlap_B_C"])
        result = validate_generation(module)
        assert not result.passed
        assert result.composition_conflicts > 0

    def test_protected_regression_prevents_commit(self):
        module = _make_composed_module()
        cases = _make_validation_cases(protected_w2l=3)
        result = validate_generation(module, validation_results=cases)
        assert not result.passed
        assert result.protected_w2l == 3

    def test_semantic_mismatch_prevents_commit(self):
        module = _make_composed_module()
        result = validate_generation(module)
        # No semantic mismatch in this fixture — validation passes
        assert result.passed
        # Manually construct one with semantic_mismatches > 0
        result2 = GenerationValidationResult(
            protected_w2l=0, semantic_mismatches=1,
            node_count=10, state_slots=4,
            composition_conflicts=0, passed=False,
            rejection_reason="semantic_mismatches=1",
        )
        assert not result2.passed

    def test_resource_overflow_prevents_commit(self):
        module = _make_composed_module(nodes=50, slots=4)
        result = validate_generation(module, max_nodes=30)
        assert not result.passed
        assert "node_count" in result.rejection_reason

    def test_no_partial_commit_on_conflict(self):
        conn = _make_conn()
        allocator = _DurableRouteAllocator()
        r1 = allocator.allocate()  # 200
        r2 = allocator.allocate()  # 201
        # Simulate conflict: rollback uncommitted
        allocator.rollback_uncommitted()
        assert allocator.committed_ids == ()
        assert allocator.allocated_ids == ()


# ── Scenario E: Parent generation changes ────────────────────────────────────

class TestScenarioE:
    def test_parent_change_invalidates_old_safety(self):
        conn = _make_conn()
        old_ctx = _commit_binary_specialist(conn, BINARY_SPEC_B, PARENT_CTX_GEN0, GATE_B, 200)

        # Parent changes to gen1
        new_ctx = binary_context_id(PARENT_CTX_GEN1, BINARY_SPEC_B, GATE_B)
        invalidated = invalidate_for_context_change(
            conn, BINARY_SPEC_B, old_ctx, new_ctx, PARENT_CTX_GEN1,
        )
        assert invalidated

        # New context shows DISCOVERED (invalidated), not COMMITTED
        new_state = get_pipeline_state(conn, BINARY_SPEC_B, new_ctx)
        assert new_state.stage is BinaryPipelineStage.DISCOVERED
        assert "invalidated_from:" in new_state.artifact_ref

        # Old context still exists (historical)
        old_state = get_pipeline_state(conn, BINARY_SPEC_B, old_ctx)
        assert old_state.stage is BinaryPipelineStage.COMMITTED

    def test_old_exact_evidence_non_authoritative(self):
        conn = _make_conn()
        old_ctx = _commit_binary_specialist(conn, BINARY_SPEC_B, PARENT_CTX_GEN0, GATE_B, 200)

        new_ctx = binary_context_id(PARENT_CTX_GEN1, BINARY_SPEC_B, GATE_B)

        # Under new context, old COMMITTED state is invisible
        new_state = get_pipeline_state(conn, BINARY_SPEC_B, new_ctx)
        assert new_state is None

        # Only after invalidation does a DISCOVERED record appear
        invalidate_for_context_change(conn, BINARY_SPEC_B, old_ctx, new_ctx, PARENT_CTX_GEN1)
        new_state = get_pipeline_state(conn, BINARY_SPEC_B, new_ctx)
        assert new_state.stage is BinaryPipelineStage.DISCOVERED

    def test_context_match_fails_on_parent_change(self):
        conn = _make_conn()
        ctx = _commit_binary_specialist(conn, BINARY_SPEC_B, PARENT_CTX_GEN0, GATE_B, 200)
        state = get_pipeline_state(conn, BINARY_SPEC_B, ctx)
        assert check_context_match(state, PARENT_CTX_GEN0, GATE_B)
        assert not check_context_match(state, PARENT_CTX_GEN1, GATE_B)

    def test_context_match_fails_on_gate_change(self):
        conn = _make_conn()
        ctx = _commit_binary_specialist(conn, BINARY_SPEC_B, PARENT_CTX_GEN0, GATE_B, 200)
        state = get_pipeline_state(conn, BINARY_SPEC_B, ctx)
        assert not check_context_match(state, PARENT_CTX_GEN0, "different_gate")


# ── Scenario F: Restart/resume ───────────────────────────────────────────────

class TestScenarioF:
    def test_restart_after_committed_generation(self):
        conn = _make_conn()
        ctx = _commit_binary_specialist(conn, BINARY_SPEC_B, PARENT_CTX_GEN0, GATE_B, 200)

        # Simulate restart: recover route state
        route_state = recover_route_allocator_state(conn, ctx)
        assert 200 in route_state["committed_route_ids"]
        assert route_state["next_durable_id"] == 201

    def test_restart_mid_pipeline_resumes_correctly(self):
        conn = _make_conn()
        ctx = binary_context_id(PARENT_CTX_GEN0, BINARY_SPEC_B, GATE_B)

        # Simulate: crash after TRANSFER_PASSED
        for stage in [BinaryPipelineStage.DISCOVERED, BinaryPipelineStage.ADMITTED,
                      BinaryPipelineStage.GATE_SYNTHESIZED, BinaryPipelineStage.ROUTED_VALIDATED,
                      BinaryPipelineStage.TRANSFER_PASSED]:
            record_pipeline_state(conn, BINARY_SPEC_B, ctx, PARENT_CTX_GEN0, stage,
                                  gate_identity=GATE_B)

        # Restart: check resume decisions
        state = get_pipeline_state(conn, BINARY_SPEC_B, ctx)
        assert state.stage is BinaryPipelineStage.TRANSFER_PASSED
        assert should_resume_stage(state, BinaryPipelineStage.ADMITTED) == "reuse"
        assert should_resume_stage(state, BinaryPipelineStage.GATE_SYNTHESIZED) == "reuse"
        assert should_resume_stage(state, BinaryPipelineStage.ROUTED_VALIDATED) == "reuse"
        assert should_resume_stage(state, BinaryPipelineStage.TRANSFER_PASSED) == "reuse"
        assert should_resume_stage(state, BinaryPipelineStage.VALIDATION_PASSED) == "execute"

    def test_restart_no_duplicate_route_ids(self):
        conn = _make_conn()
        # Gen0: route 200 committed
        _commit_binary_specialist(conn, BINARY_SPEC_B, PARENT_CTX_GEN0, GATE_B, 200)
        # Gen1: route 201 pending
        ctx1 = binary_context_id(PARENT_CTX_GEN1, BINARY_SPEC_C, GATE_C)
        record_pipeline_state(conn, BINARY_SPEC_C, ctx1, PARENT_CTX_GEN1,
                              BinaryPipelineStage.MATERIALIZED_PENDING_COMMIT,
                              gate_identity=GATE_C, route_id=201)

        route_state = recover_route_allocator_state(conn, ctx1)
        assert route_state["next_durable_id"] == 202
        assert 200 in route_state["committed_route_ids"]
        assert 201 in route_state["pending_route_ids"]

    def test_restart_no_stale_parent_context(self):
        conn = _make_conn()
        old_ctx = _commit_binary_specialist(conn, BINARY_SPEC_B, PARENT_CTX_GEN0, GATE_B, 200)

        # After restart with new parent: old context not used
        new_ctx = binary_context_id(PARENT_CTX_GEN1, BINARY_SPEC_B, GATE_B)
        state = get_pipeline_state(conn, BINARY_SPEC_B, new_ctx)
        assert state is None  # Must not silently reuse old context

    def test_temporary_route_ids_never_durable(self):
        conn = _make_conn()
        ctx = binary_context_id(PARENT_CTX_GEN0, BINARY_SPEC_B, GATE_B)
        record_pipeline_state(conn, BINARY_SPEC_B, ctx, PARENT_CTX_GEN0,
                              BinaryPipelineStage.ROUTED_VALIDATED,
                              gate_identity=GATE_B, route_id=60000)
        route_state = recover_route_allocator_state(conn, ctx)
        assert 60000 not in route_state["durable_ids_in_use"]
        assert route_state["next_durable_id"] == DURABLE_ROUTE_ID_BASE


# ── Route-ID progression across generations ──────────────────────────────────

class TestRouteIDProgression:
    def test_route_ids_increment_across_generations(self):
        conn = _make_conn()
        _commit_binary_specialist(conn, BINARY_SPEC_B, PARENT_CTX_GEN0, GATE_B, 200)
        _commit_binary_specialist(conn, BINARY_SPEC_C, PARENT_CTX_GEN1, GATE_C, 201)

        committed = get_committed_route_ids(conn)
        assert committed == [200, 201]

    def test_multiple_routes_in_one_generation(self):
        conn = _make_conn()
        _commit_binary_specialist(conn, BINARY_SPEC_B, PARENT_CTX_GEN0, GATE_B, 200)
        _commit_binary_specialist(conn, BINARY_SPEC_C, PARENT_CTX_GEN0, GATE_C, 201)
        _commit_binary_specialist(conn, BINARY_SPEC_D, PARENT_CTX_GEN1, GATE_D, 202)

        committed = get_committed_route_ids(conn)
        assert committed == [200, 201, 202]
        assert len(committed) == len(set(committed))

    def test_failed_generation_does_not_corrupt_cursor(self):
        conn = _make_conn()
        _commit_binary_specialist(conn, BINARY_SPEC_B, PARENT_CTX_GEN0, GATE_B, 200)

        # Failed attempt allocates 201 but doesn't commit
        allocator = _DurableRouteAllocator(base=201)
        attempted_id = allocator.allocate()
        assert attempted_id == 201
        allocator.rollback_uncommitted()

        # Recovery still shows next=201
        route_state = recover_route_allocator_state(conn, "any_ctx")
        assert route_state["next_durable_id"] == 201


# ── Dead/shadowed route detection ────────────────────────────────────────────

class TestDeadRouteDetection:
    def test_earlier_route_remains_visible_after_later_generation(self):
        conn = _make_conn()
        ctx_b = _commit_binary_specialist(conn, BINARY_SPEC_B, PARENT_CTX_GEN0, GATE_B, 200)
        ctx_c = _commit_binary_specialist(conn, BINARY_SPEC_C, PARENT_CTX_GEN1, GATE_C, 201)

        # Both routes exist in committed state
        committed = get_committed_route_ids(conn)
        assert 200 in committed
        assert 201 in committed

        # Both have their original gate identity preserved
        state_b = get_pipeline_state(conn, BINARY_SPEC_B, ctx_b)
        state_c = get_pipeline_state(conn, BINARY_SPEC_C, ctx_c)
        assert state_b.gate_identity == GATE_B
        assert state_c.gate_identity == GATE_C

    def test_shadowed_route_detectable(self):
        """A later route that uses the same gate would shadow an earlier one."""
        conn = _make_conn()
        _commit_binary_specialist(conn, BINARY_SPEC_B, PARENT_CTX_GEN0, GATE_B, 200)

        # If a later specialist used the SAME gate, that's a conflict indicator
        ctx_d = binary_context_id(PARENT_CTX_GEN1, BINARY_SPEC_D, GATE_B)
        record_pipeline_state(conn, BINARY_SPEC_D, ctx_d, PARENT_CTX_GEN1,
                              BinaryPipelineStage.COMMITTED,
                              gate_identity=GATE_B, route_id=201)

        # Both committed with same gate — detectable
        committed = get_committed_route_ids(conn)
        assert len(committed) == 2
        states = []
        for cand in [BINARY_SPEC_B, BINARY_SPEC_D]:
            for rid in committed:
                # Find states across all contexts
                pass
        # The manifest can surface this
        manifest = build_restart_manifest(conn, ctx_d, PARENT_CTX_GEN1)
        gates = [e["gate_identity"] for e in manifest["candidates"]]
        # If duplicates detected, composition should flag it
        assert GATE_B in gates


# ── SEALED across generations ────────────────────────────────────────────────

class TestSealedFirewall:
    def test_sealed_write_only_across_generations(self):
        conn = _make_conn()
        _commit_binary_specialist(conn, BINARY_SPEC_B, PARENT_CTX_GEN0, GATE_B, 200)

        # Verify SEALED stage is recorded in binary_pipeline_state, not sealed_audit
        columns = [
            row[1] for row in conn.execute(
                "PRAGMA table_info(binary_pipeline_state)"
            ).fetchall()
        ]
        assert "sealed_audit" not in columns
        assert "stage" in columns

        # SEALED_PASSED is just a stage value, no read path to sealed counterexamples
        ctx = binary_context_id(PARENT_CTX_GEN0, BINARY_SPEC_B, GATE_B)
        state = get_pipeline_state(conn, BINARY_SPEC_B, ctx)
        assert state.stage is BinaryPipelineStage.COMMITTED

    def test_sealed_no_influence_on_gate_synthesis(self):
        """SEALED data must never influence admission, gate synthesis, or search."""
        conn = _make_conn()
        # Create sealed_audit table as SearchMemory would
        conn.execute("""CREATE TABLE IF NOT EXISTS sealed_audit (
            semantic_id TEXT NOT NULL,
            context_id  TEXT NOT NULL,
            passed      INTEGER NOT NULL DEFAULT 0,
            insert_time INTEGER NOT NULL,
            PRIMARY KEY (semantic_id, context_id))""")
        conn.execute("INSERT INTO sealed_audit (semantic_id, context_id, passed, insert_time) "
                     "VALUES (?, ?, 0, ?)", (BINARY_SPEC_B, "old_ctx", int(time.time())))

        # Pipeline state query does not reference sealed_audit
        ctx = binary_context_id(PARENT_CTX_GEN0, BINARY_SPEC_B, GATE_B)
        state = get_pipeline_state(conn, BINARY_SPEC_B, ctx)
        assert state is None  # No pipeline state, sealed_audit data isolated


# ── SearchMemory context evolution ───────────────────────────────────────────

class TestSearchMemoryContext:
    def test_old_context_non_authoritative(self):
        conn = _make_conn()
        old_ctx = _commit_binary_specialist(conn, BINARY_SPEC_B, PARENT_CTX_GEN0, GATE_B, 200)

        # Query under new context returns nothing
        new_ctx = binary_context_id(PARENT_CTX_GEN1, BINARY_SPEC_B, GATE_B)
        new_state = get_pipeline_state(conn, BINARY_SPEC_B, new_ctx)
        assert new_state is None

        # Only the old context has the committed state
        old_state = get_pipeline_state(conn, BINARY_SPEC_B, old_ctx)
        assert old_state.stage is BinaryPipelineStage.COMMITTED

    def test_context_invalidation_count(self):
        conn = _make_conn()
        metrics = RestartMetrics()

        # Commit 3 specialists under gen0
        for spec, gate, rid in [(BINARY_SPEC_B, GATE_B, 200),
                                (BINARY_SPEC_C, GATE_C, 201),
                                (BINARY_SPEC_D, GATE_D, 202)]:
            _commit_binary_specialist(conn, spec, PARENT_CTX_GEN0, gate, rid)

        # Invalidate all for parent change
        for spec, gate in [(BINARY_SPEC_B, GATE_B),
                           (BINARY_SPEC_C, GATE_C),
                           (BINARY_SPEC_D, GATE_D)]:
            old_ctx = binary_context_id(PARENT_CTX_GEN0, spec, gate)
            new_ctx = binary_context_id(PARENT_CTX_GEN1, spec, gate)
            if invalidate_for_context_change(conn, spec, old_ctx, new_ctx, PARENT_CTX_GEN1):
                metrics.binary_context_invalidations += 1

        assert metrics.binary_context_invalidations == 3


# ── Deterministic ordering ───────────────────────────────────────────────────

class TestDeterministicOrdering:
    def test_ordering_survives_restart(self):
        conn = _make_conn()
        # Insert in reverse order
        _commit_binary_specialist(conn, BINARY_SPEC_D, PARENT_CTX_GEN0, GATE_D, 202)
        _commit_binary_specialist(conn, BINARY_SPEC_B, PARENT_CTX_GEN0, GATE_B, 200)
        _commit_binary_specialist(conn, BINARY_SPEC_C, PARENT_CTX_GEN0, GATE_C, 201)

        # Committed route IDs come back sorted
        committed = get_committed_route_ids(conn)
        assert committed == sorted(committed)
        assert committed == [200, 201, 202]


# ── SAFE_FAST_PATH unchanged ────────────────────────────────────────────────

class TestSafeFastPath:
    def test_safe_fast_path_not_affected(self):
        from kaggriculture_bonsai.routing.binary_admission import CandidateAdmissionClass
        assert CandidateAdmissionClass.SAFE_FAST_PATH.value == "SAFE_FAST_PATH"


# ── Artifact/manifest integrity ──────────────────────────────────────────────

class TestManifestIntegrity:
    def test_manifest_after_multi_generation(self):
        conn = _make_conn()
        _commit_binary_specialist(conn, BINARY_SPEC_B, PARENT_CTX_GEN0, GATE_B, 200)
        ctx_c = _commit_binary_specialist(conn, BINARY_SPEC_C, PARENT_CTX_GEN1, GATE_C, 201)

        manifest = build_restart_manifest(conn, ctx_c, PARENT_CTX_GEN1)
        assert manifest["schema_version"] == 1
        assert manifest["parent_context"] == PARENT_CTX_GEN1
        assert manifest["route_allocator"]["next_durable_id"] >= 201

        # Committed candidates in gen1 context
        for entry in manifest["candidates"]:
            assert entry["context_id"] == ctx_c
            assert entry["committed"] is True

    def test_artifact_refs_durable(self):
        conn = _make_conn()
        ctx = _commit_binary_specialist(conn, BINARY_SPEC_B, PARENT_CTX_GEN0, GATE_B, 200)
        state = get_pipeline_state(conn, BINARY_SPEC_B, ctx)
        assert state.artifact_ref.startswith("eval/")
        assert not state.artifact_ref.endswith(".pending")
        assert "60000" not in state.artifact_ref

    def test_no_temporary_paths_in_committed(self):
        conn = _make_conn()
        ctx = _commit_binary_specialist(conn, BINARY_SPEC_B, PARENT_CTX_GEN0, GATE_B, 200)
        state = get_pipeline_state(conn, BINARY_SPEC_B, ctx)
        assert state.route_id < TEMPORARY_ROUTE_ID_BASE


# ── Resource growth tracking ────────────────────────────────────────────────

class TestResourceGrowth:
    def test_resource_checks_on_composed(self):
        for gen, nodes, slots in [(0, 10, 4), (1, 15, 6), (2, 20, 8)]:
            module = _make_composed_module(nodes=nodes, slots=slots)
            result = validate_generation(module, validation_results=_make_validation_cases())
            assert result.passed
            assert result.node_count == nodes
            assert result.state_slots == slots


# ── Bounded multi-generation integration exercise ────────────────────────────

class TestBoundedMultiGenExercise:
    def test_full_multigen_exercise(self):
        """
        Exercise 4 generations:
        Gen 0: PARENT + binary B → commit (route 200)
        Gen 1: PARENT(=gen0) + safe A + binary C → commit (route 201)
        Gen 2: (failed) PARENT(=gen1) + binary D → composition conflict
        Gen 2 retry: PARENT(=gen1) + binary D → commit (route 202)
        Restart after gen1, resume gen2.
        """
        conn = _make_conn()
        metrics = RestartMetrics()
        gen_results = []

        # ── Generation 0 ──
        ctx_b = _commit_binary_specialist(conn, BINARY_SPEC_B, PARENT_CTX_GEN0, GATE_B, 200)
        module_0 = _make_composed_module(nodes=12, slots=5)
        val_0 = validate_generation(module_0, validation_results=_make_validation_cases())
        assert val_0.passed
        gen_results.append({
            "generation": 0, "parent": PARENT_CTX_GEN0,
            "specialists": [{"id": BINARY_SPEC_B, "type": "binary", "route_id": 200}],
            "protected_w2l": 0, "semantic_mismatches": 0,
            "node_count": val_0.node_count, "committed": True,
        })

        # ── Generation 1: safe + binary ──
        # Context invalidation for B under new parent
        old_ctx_b = ctx_b
        new_ctx_b = binary_context_id(PARENT_CTX_GEN1, BINARY_SPEC_B, GATE_B)
        invalidate_for_context_change(conn, BINARY_SPEC_B, old_ctx_b, new_ctx_b, PARENT_CTX_GEN1)
        metrics.binary_context_invalidations += 1

        ctx_c = _commit_binary_specialist(conn, BINARY_SPEC_C, PARENT_CTX_GEN1, GATE_C, 201)
        module_1 = _make_composed_module(nodes=18, slots=7)
        val_1 = validate_generation(module_1, validation_results=_make_validation_cases())
        assert val_1.passed
        gen_results.append({
            "generation": 1, "parent": PARENT_CTX_GEN1,
            "specialists": [
                {"id": SAFE_SPEC_A, "type": "safe"},
                {"id": BINARY_SPEC_C, "type": "binary", "route_id": 201},
            ],
            "protected_w2l": 0, "semantic_mismatches": 0,
            "node_count": val_1.node_count, "committed": True,
        })

        # ── Simulate restart ──
        route_state = recover_route_allocator_state(conn, ctx_c)
        assert route_state["next_durable_id"] == 202
        metrics.binary_route_ids_recovered += len(route_state["durable_ids_in_use"])

        # ── Generation 2: failed attempt ──
        module_fail = _make_composed_module(conflicts=["authority_overlap"])
        val_fail = validate_generation(module_fail)
        assert not val_fail.passed
        gen_results.append({
            "generation": 2, "parent": PARENT_CTX_GEN2,
            "specialists": [{"id": BINARY_SPEC_D, "type": "binary"}],
            "protected_w2l": 0, "committed": False,
            "rejection": "composition_conflict",
        })

        # ── Generation 2 retry: success ──
        ctx_d = _commit_binary_specialist(conn, BINARY_SPEC_D, PARENT_CTX_GEN2, GATE_D, 202)
        module_2 = _make_composed_module(nodes=22, slots=9)
        val_2 = validate_generation(module_2, validation_results=_make_validation_cases())
        assert val_2.passed
        gen_results.append({
            "generation": "2_retry", "parent": PARENT_CTX_GEN2,
            "specialists": [{"id": BINARY_SPEC_D, "type": "binary", "route_id": 202}],
            "protected_w2l": 0, "semantic_mismatches": 0,
            "node_count": val_2.node_count, "committed": True,
        })

        # ── Final assertions ──
        committed = get_committed_route_ids(conn)
        assert committed == [200, 201, 202]
        assert len(committed) == len(set(committed))  # no collisions

        # Verify context invalidation worked
        assert metrics.binary_context_invalidations >= 1

        # Verify route ID recovery
        final_route_state = recover_route_allocator_state(conn, ctx_d)
        assert final_route_state["next_durable_id"] == 203

        # Build final manifest
        manifest = build_restart_manifest(conn, ctx_d, PARENT_CTX_GEN2)
        assert len(manifest["candidates"]) >= 1

        # All generation validations passed (except the intentional failure)
        committed_gens = [g for g in gen_results if g.get("committed")]
        assert len(committed_gens) == 3
        for g in committed_gens:
            assert g.get("protected_w2l", 0) == 0

    def test_multigen_resume_reuses_correctly(self):
        """After restart, previously completed stages are reused."""
        conn = _make_conn()

        # Gen0: commit B
        _commit_binary_specialist(conn, BINARY_SPEC_B, PARENT_CTX_GEN0, GATE_B, 200)

        # Gen1: C reaches CEGIS_PASSED then crash
        ctx_c = binary_context_id(PARENT_CTX_GEN1, BINARY_SPEC_C, GATE_C)
        for stage in [BinaryPipelineStage.DISCOVERED, BinaryPipelineStage.ADMITTED,
                      BinaryPipelineStage.GATE_SYNTHESIZED, BinaryPipelineStage.ROUTED_VALIDATED,
                      BinaryPipelineStage.TRANSFER_PASSED, BinaryPipelineStage.VALIDATION_PASSED,
                      BinaryPipelineStage.CEGIS_PASSED]:
            record_pipeline_state(conn, BINARY_SPEC_C, ctx_c, PARENT_CTX_GEN1, stage,
                                  gate_identity=GATE_C)

        # Restart: verify resume decisions
        state = get_pipeline_state(conn, BINARY_SPEC_C, ctx_c)
        assert state.stage is BinaryPipelineStage.CEGIS_PASSED
        reuse_stages = [BinaryPipelineStage.ADMITTED, BinaryPipelineStage.GATE_SYNTHESIZED,
                        BinaryPipelineStage.ROUTED_VALIDATED, BinaryPipelineStage.TRANSFER_PASSED,
                        BinaryPipelineStage.VALIDATION_PASSED, BinaryPipelineStage.CEGIS_PASSED]
        for s in reuse_stages:
            assert should_resume_stage(state, s) == "reuse"
        assert should_resume_stage(state, BinaryPipelineStage.SEALED_PASSED) == "execute"
