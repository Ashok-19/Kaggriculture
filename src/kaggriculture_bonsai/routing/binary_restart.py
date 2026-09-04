"""Phase 1.5 — Durable binary-routing restart/resume recovery.

Makes the Phase-1 binary-routing pipeline restart-safe: admitted candidates,
synthesized gates, routed CUDA results, downstream certifications, failures,
and route-ID state survive interruption without repeating expensive work or
reusing stale safety evidence across incompatible parent generations.

Core invariant: exact-context evidence is reusable only when the evaluation
context identity matches exactly.  A changed parent/panel/config invalidates
all safety authority while preserving structural/historical records.

Extends SearchMemory with a ``binary_pipeline_state`` table.  Does NOT
create a second persistence layer.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence

from kaggriculture_bonsai.search.schema import canonical_json, stable_identity

RESTART_SCHEMA_VERSION = 1


class BinaryPipelineStage(str, Enum):
    DISCOVERED = "DISCOVERED"
    ADMITTED = "ADMITTED"
    GATE_SYNTHESIZED = "GATE_SYNTHESIZED"
    ROUTED_VALIDATED = "ROUTED_VALIDATED"
    TRANSFER_PASSED = "TRANSFER_PASSED"
    VALIDATION_PASSED = "VALIDATION_PASSED"
    CEGIS_PASSED = "CEGIS_PASSED"
    SEALED_PASSED = "SEALED_PASSED"
    MATERIALIZED_PENDING_COMMIT = "MATERIALIZED_PENDING_COMMIT"
    COMMITTED = "COMMITTED"


_TERMINAL_FAILURES = frozenset({
    "TARGET_CAPABILITY_LOST",
    "NO_SAFE_SEPARATOR",
    "GATE_LIMITED",
    "TRUE_FUTURE_DEPENDENT",
    "ROUTED_PROTECTED_REGRESSION",
    "NON_TRANSFERABLE",
    "VALIDATION_REGRESSION",
    "CEGIS_FAILED",
    "SEALED_FAILED",
    "MATERIALIZATION_FAILURE",
    "INSUFFICIENT_EVIDENCE",
    "COMPOSITION_CONFLICT",
    "GENERATION_REGRESSION",
    "SEMANTIC_MISMATCH",
    "RESOURCE_OVERFLOW",
    "INELIGIBLE",
    "PARENT_FALLBACK_MISMATCH",
    "ROUTING_SEMANTIC_MISMATCH",
})

_STAGE_ORDER = {s: i for i, s in enumerate(BinaryPipelineStage)}


def binary_context_id(
    parent_context: str,
    candidate_id: str,
    gate_identity: str = "",
) -> str:
    return stable_identity(
        "RLRMB-BINARY-CTX-V1", parent_context, candidate_id, gate_identity,
    )


_PARENT_CHANGE_SURVIVES = frozenset({
    "structural_id",
    "family",
    "lane",
    "effect_signature",
})

_PARENT_CHANGE_INVALIDATES = frozenset({
    "admission_safety",
    "gate_safety",
    "routed_cuda_safety",
    "transfer_safety",
    "validation_safety",
    "cegis_certification",
    "sealed_certification",
    "production_materialization",
})


@dataclass(frozen=True)
class BinaryPipelineState:
    candidate_semantic_id: str
    context_id: str
    parent_context: str
    stage: BinaryPipelineStage
    gate_identity: str
    route_id: int | None
    failure_class: str | None
    artifact_ref: str
    update_epoch: int

    def to_document(self) -> dict[str, Any]:
        return {
            "schema_version": RESTART_SCHEMA_VERSION,
            "candidate_semantic_id": self.candidate_semantic_id,
            "context_id": self.context_id,
            "parent_context": self.parent_context,
            "stage": self.stage.value,
            "gate_identity": self.gate_identity,
            "route_id": self.route_id,
            "failure_class": self.failure_class,
            "artifact_ref": self.artifact_ref,
            "update_epoch": self.update_epoch,
        }


@dataclass
class RestartMetrics:
    binary_admission_reuses: int = 0
    binary_gate_reuses: int = 0
    binary_routed_eval_reuses: int = 0
    binary_transfer_reuses: int = 0
    binary_validation_reuses: int = 0
    binary_cegis_reuses: int = 0
    binary_context_invalidations: int = 0
    binary_cuda_games_avoided: int = 0
    binary_pending_materializations_recovered: int = 0
    binary_route_ids_recovered: int = 0
    binary_terminal_failure_skips: int = 0

    def to_document(self) -> dict[str, Any]:
        return {
            "binary_admission_reuses": self.binary_admission_reuses,
            "binary_gate_reuses": self.binary_gate_reuses,
            "binary_routed_eval_reuses": self.binary_routed_eval_reuses,
            "binary_transfer_reuses": self.binary_transfer_reuses,
            "binary_validation_reuses": self.binary_validation_reuses,
            "binary_cegis_reuses": self.binary_cegis_reuses,
            "binary_context_invalidations": self.binary_context_invalidations,
            "binary_cuda_games_avoided": self.binary_cuda_games_avoided,
            "binary_pending_materializations_recovered": self.binary_pending_materializations_recovered,
            "binary_route_ids_recovered": self.binary_route_ids_recovered,
            "binary_terminal_failure_skips": self.binary_terminal_failure_skips,
        }


def init_binary_pipeline_schema(conn) -> None:
    conn.execute("""CREATE TABLE IF NOT EXISTS binary_pipeline_state (
        candidate_semantic_id TEXT NOT NULL,
        context_id            TEXT NOT NULL,
        parent_context        TEXT NOT NULL,
        stage                 TEXT NOT NULL,
        gate_identity         TEXT NOT NULL DEFAULT '',
        route_id              INTEGER,
        failure_class         TEXT,
        artifact_ref          TEXT NOT NULL DEFAULT '',
        update_epoch          INTEGER NOT NULL,
        PRIMARY KEY (candidate_semantic_id, context_id))""")
    conn.execute("""CREATE INDEX IF NOT EXISTS bps_by_context
        ON binary_pipeline_state (context_id)""")
    conn.execute("""CREATE INDEX IF NOT EXISTS bps_by_parent
        ON binary_pipeline_state (parent_context)""")


def get_pipeline_state(
    conn,
    candidate_id: str,
    context_id: str,
) -> BinaryPipelineState | None:
    row = conn.execute(
        "SELECT candidate_semantic_id, context_id, parent_context,"
        " stage, gate_identity, route_id, failure_class, artifact_ref, update_epoch"
        " FROM binary_pipeline_state"
        " WHERE candidate_semantic_id=? AND context_id=?",
        (candidate_id, context_id),
    ).fetchone()
    if row is None:
        return None
    return BinaryPipelineState(
        candidate_semantic_id=row[0],
        context_id=row[1],
        parent_context=row[2],
        stage=BinaryPipelineStage(row[3]),
        gate_identity=row[4],
        route_id=row[5],
        failure_class=row[6],
        artifact_ref=row[7],
        update_epoch=row[8],
    )


def record_pipeline_state(
    conn,
    candidate_id: str,
    context_id: str,
    parent_context: str,
    stage: BinaryPipelineStage,
    *,
    gate_identity: str = "",
    route_id: int | None = None,
    failure_class: str | None = None,
    artifact_ref: str = "",
) -> None:
    now = int(time.time())
    existing = get_pipeline_state(conn, candidate_id, context_id)
    if existing is not None and existing.failure_class is None:
        existing_order = _STAGE_ORDER.get(existing.stage, -1)
        new_order = _STAGE_ORDER.get(stage, -1)
        if new_order < existing_order and failure_class is None:
            return

    conn.execute("BEGIN")
    try:
        conn.execute(
            "INSERT INTO binary_pipeline_state"
            " (candidate_semantic_id, context_id, parent_context,"
            "  stage, gate_identity, route_id, failure_class, artifact_ref, update_epoch)"
            " VALUES (?,?,?,?,?,?,?,?,?)"
            " ON CONFLICT(candidate_semantic_id, context_id) DO UPDATE SET"
            " parent_context=excluded.parent_context,"
            " stage=excluded.stage,"
            " gate_identity=excluded.gate_identity,"
            " route_id=COALESCE(excluded.route_id, binary_pipeline_state.route_id),"
            " failure_class=excluded.failure_class,"
            " artifact_ref=excluded.artifact_ref,"
            " update_epoch=excluded.update_epoch",
            (candidate_id, context_id, parent_context, stage.value,
             gate_identity, route_id, failure_class, artifact_ref, now),
        )
        conn.execute("COMMIT")
    except Exception:
        try:
            conn.execute("ROLLBACK")
        except Exception:
            pass
        raise


def get_all_pipeline_states(
    conn,
    context_id: str,
) -> list[BinaryPipelineState]:
    rows = conn.execute(
        "SELECT candidate_semantic_id, context_id, parent_context,"
        " stage, gate_identity, route_id, failure_class, artifact_ref, update_epoch"
        " FROM binary_pipeline_state WHERE context_id=?"
        " ORDER BY candidate_semantic_id",
        (context_id,),
    ).fetchall()
    return [
        BinaryPipelineState(
            candidate_semantic_id=r[0], context_id=r[1], parent_context=r[2],
            stage=BinaryPipelineStage(r[3]), gate_identity=r[4], route_id=r[5],
            failure_class=r[6], artifact_ref=r[7], update_epoch=r[8],
        )
        for r in rows
    ]


def get_committed_route_ids(conn) -> list[int]:
    rows = conn.execute(
        "SELECT DISTINCT route_id FROM binary_pipeline_state"
        " WHERE route_id IS NOT NULL AND stage=?",
        (BinaryPipelineStage.COMMITTED.value,),
    ).fetchall()
    return sorted(int(r[0]) for r in rows)


def get_pending_materializations(conn, context_id: str) -> list[BinaryPipelineState]:
    rows = conn.execute(
        "SELECT candidate_semantic_id, context_id, parent_context,"
        " stage, gate_identity, route_id, failure_class, artifact_ref, update_epoch"
        " FROM binary_pipeline_state"
        " WHERE context_id=? AND stage=?",
        (context_id, BinaryPipelineStage.MATERIALIZED_PENDING_COMMIT.value),
    ).fetchall()
    return [
        BinaryPipelineState(
            candidate_semantic_id=r[0], context_id=r[1], parent_context=r[2],
            stage=BinaryPipelineStage(r[3]), gate_identity=r[4], route_id=r[5],
            failure_class=r[6], artifact_ref=r[7], update_epoch=r[8],
        )
        for r in rows
    ]


def recover_route_allocator_state(conn, context_id: str) -> dict[str, Any]:
    from .downstream_integration import DURABLE_ROUTE_ID_BASE, DURABLE_ROUTE_ID_MAX
    from .routed_evaluation import TEMPORARY_ROUTE_ID_BASE

    committed = get_committed_route_ids(conn)
    pending = get_pending_materializations(conn, context_id)
    pending_ids = [s.route_id for s in pending if s.route_id is not None]

    all_durable = sorted(set(
        rid for rid in committed + pending_ids
        if DURABLE_ROUTE_ID_BASE <= rid < TEMPORARY_ROUTE_ID_BASE
    ))

    next_id = DURABLE_ROUTE_ID_BASE
    if all_durable:
        next_id = max(all_durable) + 1

    return {
        "committed_route_ids": committed,
        "pending_route_ids": pending_ids,
        "next_durable_id": next_id,
        "durable_ids_in_use": all_durable,
    }


def should_resume_stage(
    state: BinaryPipelineState | None,
    target_stage: BinaryPipelineStage,
) -> str:
    if state is None:
        return "execute"
    if state.failure_class is not None:
        if state.failure_class in _TERMINAL_FAILURES:
            return "skip_terminal"
        return "execute"
    current_order = _STAGE_ORDER.get(state.stage, -1)
    target_order = _STAGE_ORDER.get(target_stage, -1)
    if current_order >= target_order:
        return "reuse"
    return "execute"


def invalidate_for_context_change(
    conn,
    candidate_id: str,
    old_context_id: str,
    new_context_id: str,
    new_parent_context: str,
) -> bool:
    old = get_pipeline_state(conn, candidate_id, old_context_id)
    if old is None:
        return False
    record_pipeline_state(
        conn, candidate_id, new_context_id, new_parent_context,
        BinaryPipelineStage.DISCOVERED,
        gate_identity="",
        failure_class=None,
        artifact_ref=f"invalidated_from:{old_context_id}",
    )
    return True


def check_context_match(
    state: BinaryPipelineState,
    current_parent_context: str,
    current_gate_identity: str = "",
) -> bool:
    if state.parent_context != current_parent_context:
        return False
    if current_gate_identity and state.gate_identity:
        if state.gate_identity != current_gate_identity:
            return False
    return True


def build_restart_manifest(
    conn,
    context_id: str,
    parent_context: str,
) -> dict[str, Any]:
    states = get_all_pipeline_states(conn, context_id)
    route_state = recover_route_allocator_state(conn, context_id)
    entries = []
    for s in states:
        entries.append({
            "candidate_id": s.candidate_semantic_id,
            "context_id": s.context_id,
            "stage": s.stage.value,
            "gate_identity": s.gate_identity,
            "route_id": s.route_id,
            "failure_class": s.failure_class,
            "committed": s.stage is BinaryPipelineStage.COMMITTED,
        })
    return {
        "schema_version": RESTART_SCHEMA_VERSION,
        "parent_context": parent_context,
        "context_id": context_id,
        "candidates": entries,
        "route_allocator": route_state,
    }
