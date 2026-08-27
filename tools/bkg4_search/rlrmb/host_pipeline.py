"""Small deterministic bounded multiprocessing primitive for RLRMB host work."""
from __future__ import annotations

import os
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass
from multiprocessing import get_context
from typing import Any, Callable, Iterable


class HostWorkerError(RuntimeError):
    """Fail-closed wrapper for a worker failure tied to one input sequence."""


def _run_indexed(function: Callable[[Any], Any], sequence: int, item: Any) -> tuple[int, Any, float, int]:
    started = time.perf_counter()
    result = function(item)
    return sequence, result, time.perf_counter() - started, os.getpid()


@dataclass(frozen=True)
class HostStageMetrics:
    stage: str
    tasks: int
    wall_seconds: float
    worker_busy_seconds: float
    coordinator_wait_seconds: float
    max_inflight: int
    reorder_buffer_high_water: int
    worker_pids: tuple[int, ...]

    def to_document(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "tasks": self.tasks,
            "wall_seconds": self.wall_seconds,
            "worker_busy_seconds": self.worker_busy_seconds,
            "coordinator_wait_seconds": self.coordinator_wait_seconds,
            "max_inflight": self.max_inflight,
            "reorder_buffer_high_water": self.reorder_buffer_high_water,
            "worker_pids": list(self.worker_pids),
        }


class DeterministicProcessPool:
    """Bounded process workers with canonical input-order delivery.

    Worker completion order is never observable through the returned result.
    At most ``workers + queue_depth`` application tasks are submitted at once.
    ``workers=1`` is an in-process serial fallback using the identical contract.
    """

    def __init__(self, *, workers: int, queue_depth: int, stage: str,
                 start_method: str = "spawn") -> None:
        if type(workers) is not int or workers <= 0:
            raise ValueError("workers must be a positive integer")
        if type(queue_depth) is not int or queue_depth <= 0:
            raise ValueError("queue_depth must be a positive integer")
        if type(stage) is not str or not stage:
            raise ValueError("stage must be non-empty")
        self.workers = workers
        self.queue_depth = queue_depth
        self.stage = stage
        self.start_method = start_method
        self._executor: ProcessPoolExecutor | None = None

    def __enter__(self) -> "DeterministicProcessPool":
        if self.workers > 1:
            try:
                context = get_context(self.start_method)
            except ValueError as exc:
                raise ValueError(f"unsupported multiprocessing start method: {self.start_method}") from exc
            self._executor = ProcessPoolExecutor(max_workers=self.workers, mp_context=context)
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=True)
            self._executor = None

    def map_ordered(self, function: Callable[[Any], Any], items: Iterable[Any]) -> tuple[tuple[Any, ...], HostStageMetrics]:
        values = tuple(items)
        wall_started = time.perf_counter()
        if not values:
            return (), HostStageMetrics(self.stage, 0, 0.0, 0.0, 0.0, 0, 0, ())
        if self.workers == 1:
            busy = 0.0
            results = []
            for sequence, item in enumerate(values):
                started = time.perf_counter()
                try:
                    results.append(function(item))
                except Exception as exc:
                    raise HostWorkerError(f"{self.stage} worker failure sequence={sequence}: {exc}") from exc
                busy += time.perf_counter() - started
            return tuple(results), HostStageMetrics(
                self.stage, len(values), time.perf_counter() - wall_started, busy, 0.0, 1, 0, ()
            )
        if self._executor is None:
            raise RuntimeError("process pool must be used as a context manager")

        limit = self.workers + self.queue_depth
        next_submit = 0
        next_emit = 0
        pending: dict[Any, int] = {}
        completed: dict[int, Any] = {}
        ordered: list[Any] = []
        worker_busy = 0.0
        coordinator_wait = 0.0
        maximum_inflight = 0
        reorder_high_water = 0
        worker_pids: set[int] = set()

        def submit_until_full() -> None:
            nonlocal next_submit, maximum_inflight
            while next_submit < len(values) and len(pending) < limit:
                sequence = next_submit
                future = self._executor.submit(_run_indexed, function, sequence, values[sequence])
                pending[future] = sequence
                next_submit += 1
                maximum_inflight = max(maximum_inflight, len(pending))

        submit_until_full()
        while pending:
            waited = time.perf_counter()
            done, _ = wait(tuple(pending), return_when=FIRST_COMPLETED)
            coordinator_wait += time.perf_counter() - waited
            for future in done:
                sequence = pending.pop(future)
                try:
                    returned_sequence, result, busy, pid = future.result()
                except Exception as exc:
                    for remainder in pending:
                        remainder.cancel()
                    raise HostWorkerError(f"{self.stage} worker failure sequence={sequence}: {exc}") from exc
                if returned_sequence != sequence:
                    for remainder in pending:
                        remainder.cancel()
                    raise HostWorkerError(
                        f"{self.stage} worker association mismatch sequence={sequence} returned={returned_sequence}"
                    )
                completed[sequence] = result
                worker_busy += busy
                worker_pids.add(pid)
            reorder_high_water = max(reorder_high_water, len(completed))
            while next_emit in completed:
                ordered.append(completed.pop(next_emit))
                next_emit += 1
            submit_until_full()

        if next_emit != len(values) or completed:
            raise HostWorkerError(f"{self.stage} result cardinality mismatch")
        return tuple(ordered), HostStageMetrics(
            self.stage, len(values), time.perf_counter() - wall_started, worker_busy, coordinator_wait,
            maximum_inflight, reorder_high_water, tuple(sorted(worker_pids)),
        )
