"""Phase 16.3 — Persistent host resources for one cloud run.

A ``RunResources`` object holds process pools, caches and metrics that are
expensive to create and semantically safe to reuse across multiple attempts
within a single cloud run.  It is NOT authoritative resume state; caches
rebuild safely from scratch on process restart.

Design invariants
-----------------
* Pools are entered exactly once at cloud-run start and shut down at the end.
* ``LegalStateCaptureCache`` reuse is safe because its key binds ``asset_dir``
  (which changes on every BKG4 generation commit) alongside panel path and
  probe steps; cross-generation reuse is therefore correctly impossible.
* Stable capture files are copied to a cache-owned directory so entries
  survive the attempt-level directory renames that ``cloud_runner`` performs.
* In-memory event/checkpoint/donor-manifest caches are keyed on semantic
  inputs only; a different donor manifest or limit produces a miss.
* All metrics count real work avoided, never fictional games or outcomes.
"""
from __future__ import annotations

import hashlib
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from .host_pipeline import DeterministicProcessPool
from .pipeline import LegalStateCaptureCache


def _donor_paths_key(manifest_path: str, limit: int) -> tuple[str, int]:
    return (manifest_path, limit)


def _events_cache_key(family_id: str, donor_paths: Sequence[Path]) -> tuple[str, ...]:
    return (family_id, *sorted(str(p) for p in donor_paths))


def _checkpoint_cache_key(events_key: tuple, checkpoint_limit: int) -> tuple:
    return (*events_key, checkpoint_limit)


@dataclass
class RunResources:
    """Persistent process pools and invariant caches for one cloud run.

    Must be used as a context manager.  Pools are entered at ``__enter__``
    and shut down at ``__exit__``.  The stable cache directory (if any) is
    cleaned up at ``__exit__``.
    """

    producer_pool: DeterministicProcessPool
    candidate_pool: DeterministicProcessPool
    capture_cache: LegalStateCaptureCache = field(default_factory=LegalStateCaptureCache)

    # Invariant caches (populated lazily during the run)
    _donor_paths_cache: dict = field(default_factory=dict, init=False, repr=False)
    _generic_events_cache: dict = field(default_factory=dict, init=False, repr=False)
    _checkpoint_cache: dict = field(default_factory=dict, init=False, repr=False)

    # Lifecycle counts (each pool is created exactly once per cloud run)
    producer_pool_creations: int = field(default=1, init=False)
    candidate_pool_creations: int = field(default=1, init=False)
    process_pool_reuses: int = field(default=0, init=False)

    # Invariant cache metrics
    donor_manifest_loads: int = field(default=0, init=False)
    donor_manifest_cache_hits: int = field(default=0, init=False)
    generic_events_cache_hits: int = field(default=0, init=False)
    generic_events_cache_misses: int = field(default=0, init=False)
    checkpoint_cache_hits: int = field(default=0, init=False)
    checkpoint_cache_misses: int = field(default=0, init=False)

    _entered: bool = field(default=False, init=False, repr=False)
    # Phase 16.4: optional persistent CUDA evaluator (one GPU owner at a time).
    _evaluator: Any = field(default=None, init=False, repr=False)
    _evaluator_key: tuple[Any, ...] | None = field(default=None, init=False, repr=False)
    _retired_evaluator_metrics: dict[str, float] = field(default_factory=dict, init=False, repr=False)
    # Phase 16.5: optional durable semantic search memory
    _search_memory: Any = field(default=None, init=False, repr=False)
    _search_generation: int = field(default=0, init=False, repr=False)

    @classmethod
    def from_host_execution(
        cls,
        host_execution: Mapping[str, Any],
        stable_cache_dir: Path | None = None,
    ) -> "RunResources":
        """Create resources from a ``host_execution`` config block.

        ``stable_cache_dir`` is a directory the ``LegalStateCaptureCache``
        may use to keep capture files alive across attempt-level renames.
        When ``None`` (default) the cache behaves identically to the pre-16.3
        per-attempt cache.
        """
        producer_workers = int(host_execution.get("producer_workers", 1))
        producer_queue_depth = int(host_execution.get("queue_depth", 2))
        candidate_workers = int(host_execution.get("candidate_workers", 1))
        candidate_queue_depth = int(host_execution.get("candidate_queue_depth", 4))
        start_method = str(host_execution.get("start_method", "spawn"))
        producer_pool = DeterministicProcessPool(
            workers=producer_workers,
            queue_depth=producer_queue_depth,
            stage="producer",
            start_method=start_method,
        )
        candidate_pool = DeterministicProcessPool(
            workers=candidate_workers,
            queue_depth=candidate_queue_depth,
            stage="candidate_construction",
            start_method=start_method,
        )
        capture_cache = LegalStateCaptureCache(stable_dir=stable_cache_dir)
        return cls(
            producer_pool=producer_pool,
            candidate_pool=candidate_pool,
            capture_cache=capture_cache,
        )

    # --- context manager ---

    def __enter__(self) -> "RunResources":
        self.producer_pool.__enter__()
        try:
            self.candidate_pool.__enter__()
        except Exception:
            self.producer_pool.__exit__(None, None, None)
            raise
        self._entered = True
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self._entered = False
        if self._evaluator is not None:
            try:
                self._evaluator.stop()
            except Exception:
                pass
            self._evaluator = None
            self._evaluator_key = None
        if self._search_memory is not None:
            try:
                self._search_memory.close()
            except Exception:
                pass
            self._search_memory = None
        try:
            self.producer_pool.__exit__(exc_type, exc, tb)
        finally:
            self.candidate_pool.__exit__(exc_type, exc, tb)
        if self.capture_cache.stable_dir is not None:
            try:
                shutil.rmtree(self.capture_cache.stable_dir, ignore_errors=True)
            except Exception:
                pass

    # --- search memory (Phase 16.5) ---

    def open_search_memory(self, db_path: "Path") -> None:
        """Open (or reopen from disk) the durable semantic search memory.

        Must be called after the context manager is entered.  The memory is
        closed automatically by ``__exit__``.
        """
        from .search_memory import SearchMemory
        mem = SearchMemory(db_path)
        mem.open()
        self._search_memory = mem

    @property
    def search_memory(self) -> Any:
        return self._search_memory

    def set_search_generation(self, generation: int) -> None:
        if type(generation) is not int or generation < 0:
            raise ValueError("search generation must be a non-negative integer")
        self._search_generation = generation

    @property
    def search_generation(self) -> int:
        return self._search_generation

    # --- persistent evaluator (Phase 16.4) ---

    def get_or_create_evaluator(
        self,
        binary: "Path",
        asset_dir: "Path",
        *,
        feed_slot: int = 5,
        feed_count: int = 13,
        strategic_route_slot: int = 19,
    ) -> Any:
        """Return the shared persistent CUDA evaluator for one exact asset scope.

        Calls with the same binary/assets/ABI reuse the existing process.  A new
        committed BKG4 asset directory is a semantic boundary: the old process
        is stopped before a replacement is created, preserving exactly one GPU
        owner while guaranteeing the subprocess reloads controller/routes.
        """
        key = (
            str(Path(binary).resolve()),
            str(Path(asset_dir).resolve()),
            int(feed_slot),
            int(feed_count),
            int(strategic_route_slot),
        )
        if self._evaluator is not None and self._evaluator_key != key:
            try:
                doc = self._evaluator.metrics_document()
                if isinstance(doc, Mapping):
                    for name, value in doc.items():
                        if isinstance(value, (int, float)):
                            self._retired_evaluator_metrics[name] = (
                                self._retired_evaluator_metrics.get(name, 0.0) + float(value)
                            )
            finally:
                self._evaluator.stop()
                self._evaluator = None
                self._evaluator_key = None
        if self._evaluator is None:
            from .beam import PersistentCudaEvaluator
            ev = PersistentCudaEvaluator(
                binary=binary,
                asset_dir=asset_dir,
                feed_slot=feed_slot,
                feed_count=feed_count,
                strategic_route_slot=strategic_route_slot,
            )
            ev.start()
            self._evaluator = ev
            self._evaluator_key = key
        return self._evaluator

    # --- pool reuse recording ---

    def record_pool_reuse(self) -> None:
        self.process_pool_reuses += 1

    # --- donor manifest cache ---

    def get_donor_paths(self, manifest_path: str, limit: int) -> Sequence[Path] | None:
        key = _donor_paths_key(manifest_path, limit)
        if key in self._donor_paths_cache:
            self.donor_manifest_cache_hits += 1
            return self._donor_paths_cache[key]
        return None

    def put_donor_paths(self, manifest_path: str, limit: int, paths: Sequence[Path]) -> None:
        self._donor_paths_cache[_donor_paths_key(manifest_path, limit)] = tuple(paths)
        self.donor_manifest_loads += 1

    # --- generic events cache ---

    def get_generic_events(self, family_id: str, donor_paths: Sequence[Path]) -> Any | None:
        key = _events_cache_key(family_id, donor_paths)
        if key in self._generic_events_cache:
            self.generic_events_cache_hits += 1
            return self._generic_events_cache[key]
        self.generic_events_cache_misses += 1
        return None

    def put_generic_events(
        self, family_id: str, donor_paths: Sequence[Path], events: Any
    ) -> None:
        self._generic_events_cache[_events_cache_key(family_id, donor_paths)] = events

    # --- checkpoint cache ---

    def get_checkpoints(
        self, family_id: str, donor_paths: Sequence[Path], checkpoint_limit: int
    ) -> Any | None:
        key = _checkpoint_cache_key(
            _events_cache_key(family_id, donor_paths), checkpoint_limit
        )
        if key in self._checkpoint_cache:
            self.checkpoint_cache_hits += 1
            return self._checkpoint_cache[key]
        self.checkpoint_cache_misses += 1
        return None

    def put_checkpoints(
        self,
        family_id: str,
        donor_paths: Sequence[Path],
        checkpoint_limit: int,
        checkpoints: Any,
    ) -> None:
        key = _checkpoint_cache_key(
            _events_cache_key(family_id, donor_paths), checkpoint_limit
        )
        self._checkpoint_cache[key] = checkpoints

    # --- aggregate metrics ---

    def metrics_document(self) -> dict[str, Any]:
        doc: dict[str, Any] = {
            "producer_pool_creations": self.producer_pool_creations,
            "candidate_pool_creations": self.candidate_pool_creations,
            "process_pool_reuses": self.process_pool_reuses,
            "capture_launches": self.capture_cache.launches,
            "capture_reuses": self.capture_cache.reuses,
            "avoided_capture_launches": self.capture_cache.reuses,
            "capture_cache_hits": self.capture_cache.reuses,
            "capture_cache_misses": self.capture_cache.launches,
            "capture_host_seconds": self.capture_cache.seconds,
            "donor_manifest_loads": self.donor_manifest_loads,
            "donor_manifest_cache_hits": self.donor_manifest_cache_hits,
            "generic_events_cache_hits": self.generic_events_cache_hits,
            "generic_events_cache_misses": self.generic_events_cache_misses,
            "checkpoint_cache_hits": self.checkpoint_cache_hits,
            "checkpoint_cache_misses": self.checkpoint_cache_misses,
        }
        evaluator_metrics: dict[str, float] = {
            "evaluator_process_starts": 0.0,
            "evaluator_requests": 0.0,
            "persistent_evaluator_reuses": 0.0,
            "total_serialization_seconds": 0.0,
            "total_result_parsing_seconds": 0.0,
            "total_evaluator_host_seconds": 0.0,
        }
        for name, value in self._retired_evaluator_metrics.items():
            if name in evaluator_metrics:
                evaluator_metrics[name] += float(value)
        if self._evaluator is not None:
            current = self._evaluator.metrics_document()
            if isinstance(current, Mapping):
                for name, value in current.items():
                    if name in evaluator_metrics and isinstance(value, (int, float)):
                        evaluator_metrics[name] += float(value)
        doc["p164_evaluator"] = {
            "evaluator_process_starts": int(evaluator_metrics["evaluator_process_starts"]),
            "evaluator_requests": int(evaluator_metrics["evaluator_requests"]),
            "persistent_evaluator_reuses": int(evaluator_metrics["persistent_evaluator_reuses"]),
            "total_serialization_seconds": evaluator_metrics["total_serialization_seconds"],
            "total_result_parsing_seconds": evaluator_metrics["total_result_parsing_seconds"],
            "total_evaluator_host_seconds": evaluator_metrics["total_evaluator_host_seconds"],
        }
        if self._search_memory is not None:
            doc["p165_memory"] = self._search_memory.metrics()
        else:
            from .search_memory import _ZERO_METRICS
            doc["p165_memory"] = dict(_ZERO_METRICS)
        return doc
