"""Phase 16.5 — Durable semantic search memory for RLRMB cloud runs.

Three knowledge scopes
----------------------
Structural  : candidate existence, reusable globally across all contexts.
Evaluation  : exact outcomes keyed by evaluation-context identity.  A new
               BKG4 generation changes the controller hash, which changes the
               context ID, which invalidates exact records automatically.
Historical  : old-generation outcomes retained as priors, never authoritative.

SEALED firewall
---------------
Certification outcomes are stored in ``sealed_audit`` — a write-only table.
The search code path has NO read API for this table.  Structural isolation is
the firewall: ``get_exact_evaluation`` and ``get_historical_priors`` only read
from ``structural``, ``evaluation``, and ``historical`` tables.

Crash safety
------------
All multi-row mutations go through explicit ``BEGIN``/``COMMIT`` transactions
so that partial records are never authoritative on process restart.  WAL mode
lets readers proceed concurrently with writes.
"""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any

MEMORY_SCHEMA_VERSION = 1

_EVAL_CTX_KEYS = (
    "checkpoint_limit", "donor_limit", "motif_steps", "beam_depth",
    "max_candidates_per_expansion", "stagnation_rounds", "max_phases",
    "archive_elites_per_cell",
)


def evaluation_context_id(config: dict[str, Any]) -> str:
    """Deterministic identity for one (controller, family, panels, search-config) tuple.

    Excludes generation counter so that exact evaluation records survive
    same-controller retry attempts.  A new BKG4 generation changes the
    controller semantic hash, which changes this ID, invalidating records.
    """
    from .schema import stable_identity
    asset_dir = str(config.get("asset_dir", ""))
    identity_path = Path(asset_dir) / "identity.json"
    if identity_path.exists():
        doc = json.loads(identity_path.read_text(encoding="utf-8"))
        controller_hash = str(doc.get("controller_semantic_hash", ""))
    else:
        controller_hash = stable_identity("RLRMB-P165-ASSET-FALLBACK-V1", asset_dir)
    family = config["families"][0]
    search = config.get("search", {})
    context = {
        "controller": controller_hash,
        "family_id": family.get("id"),
        "target_panel": family.get("target_panel"),
        "transfer_panel": config.get("transfer_panel"),
        "validation_panel": config.get("validation_panel"),
        "search": {k: search.get(k) for k in _EVAL_CTX_KEYS},
    }
    return stable_identity("RLRMB-P165-EVAL-CTX-V1", context)


def effect_to_json(effect_vector: Any) -> str:
    """Serialize EffectVector to compact JSON for storage."""
    return json.dumps(
        effect_vector.to_document(), sort_keys=True, separators=(",", ":")
    )


def effect_from_json(effect_json: str) -> Any:
    """Reconstruct EffectVector from stored compact JSON."""
    from .effects import EffectBlock, EffectVector
    from .schema import DatasetRole
    doc = json.loads(effect_json)
    blocks = []
    for b in doc["blocks"]:
        ac = b.get("action_changes")
        blocks.append(EffectBlock(
            role=DatasetRole(b["role"]),
            panel_id=b["panel_id"],
            baseline_margins=tuple(b["baseline_margins"]),
            margin_deltas=tuple(b["margin_deltas"]),
            action_changes=None if ac is None else tuple(ac),
        ))
    return EffectVector(tuple(blocks))


class SearchMemory:
    """SQLite-backed durable semantic search memory (Phase 16.5).

    Must be used as a context manager or explicitly opened/closed.
    All writes are transactional; partial records never appear on restart.
    """

    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self._conn: sqlite3.Connection | None = None
        # Hit/miss counters
        self.structural_hits = 0
        self.structural_misses = 0
        self.exact_hits = 0
        self.exact_misses = 0
        # Avoided-work counters (incremented by MemoryBackedEvaluator callers)
        self.avoided_cuda_games = 0
        self.avoided_candidate_evaluations = 0
        self.context_invalidations = 0
        self.persisted_failure_records = 0
        self.persisted_effect_signatures = 0

    def open(self) -> "SearchMemory":
        self._conn = sqlite3.connect(str(self.db_path), isolation_level=None)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._init_schema()
        return self

    def _init_schema(self) -> None:
        conn = self._conn
        conn.execute("BEGIN")
        try:
            conn.execute("""CREATE TABLE IF NOT EXISTS schema_meta (
                key   TEXT NOT NULL PRIMARY KEY,
                value TEXT NOT NULL)""")
            conn.execute("""CREATE TABLE IF NOT EXISTS structural (
                semantic_id     TEXT NOT NULL PRIMARY KEY,
                lane            TEXT,
                operator        TEXT,
                generation_first INTEGER NOT NULL DEFAULT 0,
                context_id_first TEXT,
                insert_time     INTEGER NOT NULL)""")
            conn.execute("""CREATE TABLE IF NOT EXISTS evaluation (
                semantic_id  TEXT NOT NULL,
                context_id   TEXT NOT NULL,
                generation   INTEGER NOT NULL DEFAULT 0,
                valid        INTEGER NOT NULL DEFAULT 1,
                fault        TEXT,
                effect_json  TEXT,
                failure_class TEXT,
                cuda_games   INTEGER NOT NULL DEFAULT 0,
                insert_time  INTEGER NOT NULL,
                PRIMARY KEY (semantic_id, context_id))""")
            conn.execute("""CREATE INDEX IF NOT EXISTS evaluation_by_context
                ON evaluation (context_id)""")
            conn.execute("""CREATE TABLE IF NOT EXISTS historical (
                semantic_id   TEXT NOT NULL,
                context_id    TEXT NOT NULL,
                generation    INTEGER NOT NULL,
                failure_class TEXT,
                outcome_summary TEXT,
                insert_time   INTEGER NOT NULL,
                PRIMARY KEY (semantic_id, context_id, generation))""")
            # SEALED firewall: this table has NO read API in the search code path.
            conn.execute("""CREATE TABLE IF NOT EXISTS sealed_audit (
                semantic_id TEXT NOT NULL,
                context_id  TEXT NOT NULL,
                passed      INTEGER NOT NULL DEFAULT 0,
                insert_time INTEGER NOT NULL,
                PRIMARY KEY (semantic_id, context_id))""")
            row = conn.execute(
                "SELECT value FROM schema_meta WHERE key='schema_version'"
            ).fetchone()
            if row is None:
                conn.execute(
                    "INSERT INTO schema_meta (key, value) VALUES ('schema_version', ?)",
                    (str(MEMORY_SCHEMA_VERSION),),
                )
            elif int(row[0]) != MEMORY_SCHEMA_VERSION:
                conn.execute("ROLLBACK")
                raise ValueError(
                    f"SearchMemory schema mismatch: expected {MEMORY_SCHEMA_VERSION},"
                    f" got {row[0]}"
                )
            conn.execute("COMMIT")
        except Exception:
            try:
                conn.execute("ROLLBACK")
            except Exception:
                pass
            raise

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> "SearchMemory":
        return self.open()

    def __exit__(self, *args: Any) -> None:
        self.close()

    # ── structural scope ────────────────────────────────────────────────────────

    def is_structurally_seen(self, semantic_id: str) -> bool:
        row = self._conn.execute(
            "SELECT 1 FROM structural WHERE semantic_id=?", (semantic_id,)
        ).fetchone()
        if row:
            self.structural_hits += 1
            return True
        self.structural_misses += 1
        return False

    def record_structural(
        self,
        semantic_id: str,
        *,
        lane: str = "",
        operator: str = "",
        generation: int = 0,
        context_id: str = "",
    ) -> None:
        self._conn.execute(
            "INSERT OR IGNORE INTO structural"
            " (semantic_id, lane, operator, generation_first, context_id_first, insert_time)"
            " VALUES (?,?,?,?,?,?)",
            (semantic_id, lane, operator, generation, context_id, int(time.time())),
        )

    # ── exact evaluation scope ─────────────────────────────────────────────────

    def get_exact_evaluation(
        self, semantic_id: str, context_id: str
    ) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT valid, fault, effect_json, failure_class, cuda_games, generation"
            " FROM evaluation WHERE semantic_id=? AND context_id=?",
            (semantic_id, context_id),
        ).fetchone()
        if row is None:
            self.exact_misses += 1
            return None
        self.exact_hits += 1
        valid, fault, effect_json, failure_class, cuda_games, generation = row
        return {
            "valid": bool(valid),
            "fault": fault,
            "effect_json": effect_json,
            "failure_class": failure_class,
            "cuda_games": int(cuda_games),
            "generation": int(generation),
        }

    def record_evaluation(
        self,
        semantic_id: str,
        context_id: str,
        *,
        generation: int = 0,
        valid: bool = True,
        fault: str | None = None,
        effect_json: str | None = None,
        failure_class: str | None = None,
        cuda_games: int = 0,
    ) -> None:
        now = int(time.time())
        self._conn.execute("BEGIN")
        try:
            self._conn.execute(
                "INSERT OR REPLACE INTO evaluation"
                " (semantic_id, context_id, generation, valid, fault, effect_json,"
                "  failure_class, cuda_games, insert_time)"
                " VALUES (?,?,?,?,?,?,?,?,?)",
                (semantic_id, context_id, generation, int(valid), fault, effect_json,
                 failure_class, cuda_games, now),
            )
            self._conn.execute("COMMIT")
        except Exception:
            try:
                self._conn.execute("ROLLBACK")
            except Exception:
                pass
            raise
        if fault or failure_class:
            self.persisted_failure_records += 1
        if effect_json:
            self.persisted_effect_signatures += 1

    # ── historical scope ───────────────────────────────────────────────────────

    def promote_to_historical(self, context_id: str, new_generation: int) -> int:
        """Copy evaluation records for ``context_id`` into the historical table.

        Returns the number of records promoted.  Call this when the BKG4
        generation advances so old exact records become non-authoritative priors.
        """
        rows = self._conn.execute(
            "SELECT semantic_id, context_id, generation, failure_class, effect_json"
            " FROM evaluation WHERE context_id=?",
            (context_id,),
        ).fetchall()
        if not rows:
            return 0
        now = int(time.time())
        self._conn.execute("BEGIN")
        try:
            for semantic_id, ctx, gen, failure_class, effect_json in rows:
                summary = json.dumps(
                    {
                        "failure_class": failure_class,
                        "had_effect": effect_json is not None,
                        "promoted_to_generation": new_generation,
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                )
                self._conn.execute(
                    "INSERT OR IGNORE INTO historical"
                    " (semantic_id, context_id, generation, failure_class,"
                    "  outcome_summary, insert_time)"
                    " VALUES (?,?,?,?,?,?)",
                    (semantic_id, ctx, gen, failure_class, summary, now),
                )
            self._conn.execute("COMMIT")
        except Exception:
            try:
                self._conn.execute("ROLLBACK")
            except Exception:
                pass
            raise
        self.context_invalidations += 1
        return len(rows)

    def get_historical_priors(self, semantic_id: str) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT context_id, generation, failure_class, outcome_summary"
            " FROM historical WHERE semantic_id=? ORDER BY generation DESC",
            (semantic_id,),
        ).fetchall()
        return [
            {
                "context_id": r[0],
                "generation": int(r[1]),
                "failure_class": r[2],
                "outcome_summary": r[3],
            }
            for r in rows
        ]

    # ── SEALED firewall — write-only audit, no search-path read ───────────────

    def record_sealed_audit(
        self, semantic_id: str, context_id: str, *, passed: bool
    ) -> None:
        """Record a SEALED certification outcome in the audit table.

        This is intentionally write-only from the search code path.
        No read API exists here so SEALED outcomes cannot influence generation
        or search queries.
        """
        self._conn.execute(
            "INSERT OR REPLACE INTO sealed_audit"
            " (semantic_id, context_id, passed, insert_time)"
            " VALUES (?,?,?,?)",
            (semantic_id, context_id, int(passed), int(time.time())),
        )

    # ── allocator query API (Phase 16.6) — never reads sealed_audit ───────────

    def get_context_evaluation_stats(self, context_id: str) -> dict[str, Any]:
        """Aggregate evaluation statistics for one evaluation-context.

        Used by SearchAllocator for scoring.  Reads only the ``evaluation``
        and ``structural`` tables.  ``sealed_audit`` is intentionally excluded.
        Returns an empty dict when no records exist.
        """
        row = self._conn.execute(
            "SELECT COUNT(*),"
            "  SUM(valid),"
            "  SUM(CASE WHEN effect_json IS NOT NULL AND valid=1 THEN 1 ELSE 0 END),"
            "  SUM(CASE WHEN failure_class='NON_TRANSFERABLE' THEN 1 ELSE 0 END),"
            "  SUM(CASE WHEN failure_class='FUTURE_DEPENDENT' THEN 1 ELSE 0 END),"
            "  SUM(CASE WHEN failure_class IS NOT NULL"
            "       AND failure_class NOT IN ('NON_TRANSFERABLE','FUTURE_DEPENDENT')"
            "       THEN 1 ELSE 0 END),"
            "  SUM(cuda_games)"
            " FROM evaluation WHERE context_id=?",
            (context_id,),
        ).fetchone()
        total = int(row[0] or 0) if row else 0
        if total == 0:
            return {}
        nov = self._conn.execute(
            "SELECT COUNT(*) FROM structural WHERE context_id_first=?",
            (context_id,),
        ).fetchone()
        return {
            "total_evaluations": total,
            "valid_evaluations": int(row[1] or 0),
            "effect_evaluations": int(row[2] or 0),
            "non_transferable_count": int(row[3] or 0),
            "future_dependent_count": int(row[4] or 0),
            "other_failure_count": int(row[5] or 0),
            "total_cuda_games": int(row[6] or 0),
            "novel_structural": int(nov[0] or 0) if nov else 0,
        }

    def get_context_historical_stats(self, context_id: str) -> dict[str, Any]:
        """Aggregate historical prior statistics for one evaluation-context.

        Used by SearchAllocator for prior-based scoring.  Never reads
        ``sealed_audit``.  Returns ``{"total": 0, "failures": 0}`` when empty.
        """
        row = self._conn.execute(
            "SELECT COUNT(*),"
            "  SUM(CASE WHEN failure_class IS NOT NULL THEN 1 ELSE 0 END)"
            " FROM historical WHERE context_id=?",
            (context_id,),
        ).fetchone()
        if row is None or row[0] == 0:
            return {"total": 0, "failures": 0}
        return {"total": int(row[0] or 0), "failures": int(row[1] or 0)}

    # ── aggregate metrics ──────────────────────────────────────────────────────

    def record_count(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) FROM evaluation").fetchone()
        return int(row[0]) if row else 0

    def metrics(self) -> dict[str, Any]:
        try:
            size = self.db_path.stat().st_size
        except OSError:
            size = 0
        return {
            "memory_structural_hits": self.structural_hits,
            "memory_structural_misses": self.structural_misses,
            "memory_exact_evaluation_hits": self.exact_hits,
            "memory_exact_evaluation_misses": self.exact_misses,
            "avoided_cuda_games": self.avoided_cuda_games,
            "avoided_candidate_evaluations": self.avoided_candidate_evaluations,
            "evaluation_context_invalidations": self.context_invalidations,
            "persisted_failure_records": self.persisted_failure_records,
            "persisted_effect_signatures": self.persisted_effect_signatures,
            "memory_records": self.record_count(),
            "memory_storage_bytes": size,
        }


_ZERO_METRICS: dict[str, Any] = {
    "memory_structural_hits": 0,
    "memory_structural_misses": 0,
    "memory_exact_evaluation_hits": 0,
    "memory_exact_evaluation_misses": 0,
    "avoided_cuda_games": 0,
    "avoided_candidate_evaluations": 0,
    "evaluation_context_invalidations": 0,
    "persisted_failure_records": 0,
    "persisted_effect_signatures": 0,
    "memory_records": 0,
    "memory_storage_bytes": 0,
}
