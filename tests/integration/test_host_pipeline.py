"""Deterministic bounded host multiprocessing contracts for Phase 16."""
from __future__ import annotations

import importlib
import importlib.util
import time
import unittest


MODULE = "kaggriculture_bonsai.search.host_pipeline"


def _delayed_identity(item: tuple[int, float]) -> int:
    value, delay = item
    time.sleep(delay)
    return value


def _explode_on_three(value: int) -> int:
    if value == 3:
        raise RuntimeError("synthetic worker failure")
    return value


class HostPipelineTests(unittest.TestCase):
    def module(self):
        spec = importlib.util.find_spec(MODULE)
        self.assertIsNotNone(spec, "Phase-16 host pipeline module is missing")
        return importlib.import_module(MODULE)

    def test_worker_count_independence_preserves_exact_order(self):
        module = self.module()
        items = tuple((value, 0.002 * (8 - value)) for value in range(8))
        expected = tuple(range(8))
        for workers in (1, 2, 4):
            with self.subTest(workers=workers):
                with module.DeterministicProcessPool(
                    workers=workers, queue_depth=2, stage="producer", start_method="spawn"
                ) as pool:
                    result, metrics = pool.map_ordered(_delayed_identity, items)
                self.assertEqual(result, expected)
                self.assertEqual(metrics.tasks, len(items))
                self.assertGreater(metrics.wall_seconds, 0.0)

    def test_out_of_order_completion_is_reordered_before_return(self):
        module = self.module()
        items = ((0, 0.06), (1, 0.04), (2, 0.02), (3, 0.0))
        with module.DeterministicProcessPool(
            workers=4, queue_depth=1, stage="producer", start_method="spawn"
        ) as pool:
            result, metrics = pool.map_ordered(_delayed_identity, items)
        self.assertEqual(result, (0, 1, 2, 3))
        self.assertGreater(metrics.reorder_buffer_high_water, 0)

    def test_inflight_work_is_bounded_by_workers_plus_queue_depth(self):
        module = self.module()
        with module.DeterministicProcessPool(
            workers=2, queue_depth=3, stage="producer", start_method="spawn"
        ) as pool:
            result, metrics = pool.map_ordered(_delayed_identity, tuple((i, 0.01) for i in range(20)))
        self.assertEqual(result, tuple(range(20)))
        self.assertLessEqual(metrics.max_inflight, 5)
        self.assertGreater(metrics.coordinator_wait_seconds, 0.0)

    def test_worker_failure_fails_closed_without_partial_result(self):
        module = self.module()
        with module.DeterministicProcessPool(
            workers=2, queue_depth=1, stage="consumer", start_method="spawn"
        ) as pool:
            with self.assertRaisesRegex(module.HostWorkerError, "sequence=3"):
                pool.map_ordered(_explode_on_three, tuple(range(8)))

    def test_serial_fallback_uses_same_ordered_contract(self):
        module = self.module()
        with module.DeterministicProcessPool(
            workers=1, queue_depth=2, stage="producer", start_method="spawn"
        ) as pool:
            result, metrics = pool.map_ordered(_delayed_identity, ((2, 0.0), (1, 0.0), (0, 0.0)))
        self.assertEqual(result, (2, 1, 0))
        self.assertEqual(metrics.max_inflight, 1)
        self.assertEqual(metrics.worker_pids, ())

    def test_parallel_checkpoint_motif_preparation_matches_current_semantics(self):
        module = self.module()
        campaign = importlib.import_module("kaggriculture_bonsai.search.p13_equal_budget")
        beam_tests = importlib.import_module("tests.unit.test_beam")
        prepare = getattr(campaign, "prepare_checkpoint_motif_batch")
        root = importlib.import_module("pathlib").Path.cwd()
        manifest = root / "configs/donors.txt"
        donors = tuple(importlib.import_module("pathlib").Path(line.strip()) for line in manifest.read_text().splitlines() if line.strip())[:4]
        checkpoints = (beam_tests.cp(0), beam_tests.cp(1))
        expected_by_step = {
            point.step: campaign._motifs_at(point.step, donors, 8)
            for point in checkpoints
        }
        with module.DeterministicProcessPool(
            workers=2, queue_depth=1, stage="producer", start_method="spawn"
        ) as pool:
            prepared = prepare(checkpoints, donors, 8, pool)
        self.assertEqual(tuple(prepared.checkpoint_motifs), tuple(point.step for point in checkpoints))
        for step, expected in expected_by_step.items():
            actual = prepared.checkpoint_motifs[step]
            self.assertEqual(tuple(actual), tuple(expected))
            self.assertEqual(
                [actual[key].semantic_document() for key in actual],
                [expected[key].semantic_document() for key in expected],
            )
        expected_all = {}
        for point in checkpoints:
            expected_all.update(expected_by_step[point.step])
        self.assertEqual(tuple(prepared.motifs), tuple(expected_all))
        self.assertEqual(prepared.metrics.tasks, len(donors))

    def test_checkpoint_batch_prefetch_overlaps_next_producer_batch(self):
        campaign = importlib.import_module("kaggriculture_bonsai.search.p13_equal_budget")
        prefetch = getattr(campaign, "prefetch_checkpoint_motif_batches")
        from unittest.mock import patch

        calls = []
        def fake_prepare(checkpoints, donor_paths, length, pool):
            time.sleep(0.08)
            value = tuple(point.step for point in checkpoints)
            calls.append(value)
            return value

        beam_tests = importlib.import_module("tests.unit.test_beam")
        batches = ((beam_tests.cp(0),), (beam_tests.cp(1),))
        with patch.object(campaign, "prepare_checkpoint_motif_batch", side_effect=fake_prepare):
            stream = prefetch(batches, (), 8, object())
            first_batch, first_value, first_wait = next(stream)
            time.sleep(0.10)
            started = time.perf_counter()
            second_batch, second_value, second_wait = next(stream)
            elapsed = time.perf_counter() - started
            stream.close()
        self.assertEqual(tuple(point.step for point in first_batch), (0,))
        self.assertEqual(tuple(point.step for point in second_batch), (1,))
        self.assertEqual((first_value, second_value), ((0,), (1,)))
        self.assertEqual(calls, [(0,), (1,)])
        self.assertGreater(first_wait, 0.05)
        self.assertLess(elapsed, 0.04)
        self.assertLess(second_wait, 0.04)


if __name__ == "__main__":
    unittest.main()
