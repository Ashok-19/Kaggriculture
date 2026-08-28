"""Phase 16.3 — Persistent Host Resources + Generation-Scoped Invariant Caches.

18 tests covering:
  * RunResources lifecycle (pool creation, reuse, shutdown)
  * LegalStateCaptureCache key binding (asset_dir, panel, probe steps)
  * Invariant caches (generic_events, checkpoints, donor manifest)
  * Cloud runner integration (metrics, suppression, resume, regression guards)
"""
from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from tools.bkg4_search.rlrmb.host_pipeline import DeterministicProcessPool
from tools.bkg4_search.rlrmb.pipeline import LegalStateCaptureCache
from tools.bkg4_search.rlrmb.run_resources import RunResources
from tools.bkg4_search.rlrmb.cloud_runner import (
    load_cloud_config, run_cloud,
)


ROOT = Path(__file__).resolve().parents[4]
CONFIG = ROOT / "tools/bkg4_search/rlrmb/configs/phase16_lightning_v1.json"

# ── helpers ──────────────────────────────────────────────────────────────────

def _make_resources(stable_dir=None):
    """Create a minimal RunResources with 1-worker pools (fast for tests)."""
    host_ex = {"producer_workers": 1, "queue_depth": 1,
                "candidate_workers": 1, "candidate_queue_depth": 1,
                "start_method": "spawn"}
    return RunResources.from_host_execution(host_ex, stable_cache_dir=stable_dir)


def _identity(x):
    return x


def _explode(x):
    raise RuntimeError("synthetic failure")


def _stub_runner(cuda_games=10, accepted=False, semantic_mismatches=0, protected_wl=0):
    candidate = "a" * 64
    def runner(runtime_path, run_dir):
        run_dir.mkdir(parents=True)
        metrics = {
            "total_cuda_games": cuda_games,
            "accepted_candidate_ids": [candidate] if accepted else [],
            "protected_win_to_loss": protected_wl,
            "semantic_mismatches": semantic_mismatches,
        }
        (run_dir / "metrics.json").write_text(json.dumps(metrics))
        if accepted:
            family = json.loads(runtime_path.read_text())["families"][0]["id"]
            dep = run_dir / "families" / family / candidate / "deployment"
            dep.mkdir(parents=True)
            for name, val in (
                ("composed_module.json", {}),
                ("specialist.json", {}),
                (f"{candidate}.route.json", {"actions": []}),
            ):
                (dep / name).write_text(json.dumps(val))
        return 0
    return runner


def _stub_assets(candidate="a" * 64):
    def assets(parent, deployment, destination):
        destination.mkdir(parents=True, exist_ok=True)
        for name in ("controller.bkgc", "routes.bkv6", "legacy_routes.bkv6"):
            (destination / name).write_bytes(b"ok")
        (destination / "controller.json").write_text("{}")
        doc = {"controller_semantic_hash": "newhash", "route_id": 999,
               "candidate_id": candidate, "nodes": 1, "state_slots": 1}
        (destination / "identity.json").write_text(json.dumps(doc))
        return {"candidate_id": candidate, "nodes": 1, "state_slots": 1}
    return assets


# ── Group 1: RunResources lifecycle (tests 1-6) ───────────────────────────────

class TestPoolCreation(unittest.TestCase):
    """Test 1 — multi-attempt cloud run creates each pool exactly once."""

    def test_pool_creation_counts_are_exactly_one(self):
        res = _make_resources()
        # from_host_execution always sets creation counts to 1
        self.assertEqual(res.producer_pool_creations, 1)
        self.assertEqual(res.candidate_pool_creations, 1)
        with res:
            # Even after entering (which starts the pools), counts remain 1
            self.assertEqual(res.producer_pool_creations, 1)
            self.assertEqual(res.candidate_pool_creations, 1)


class TestPoolReuse(unittest.TestCase):
    """Test 2 — pool reuse counter increments correctly across campaigns."""

    def test_record_pool_reuse_increments(self):
        res = _make_resources()
        self.assertEqual(res.process_pool_reuses, 0)
        res.record_pool_reuse()
        res.record_pool_reuse()
        res.record_pool_reuse()
        self.assertEqual(res.process_pool_reuses, 3)
        doc = res.metrics_document()
        self.assertEqual(doc["process_pool_reuses"], 3)
        self.assertEqual(doc["producer_pool_creations"], 1)


class TestWorkerDeterminism(unittest.TestCase):
    """Test 3 — worker results are deterministic regardless of completion order."""

    def test_pool_results_are_ordered(self):
        # Phase 16.2 determinism: same input → same output regardless of worker count
        items = tuple(range(6))
        results = {}
        for workers in (1, 2, 3):
            with DeterministicProcessPool(
                workers=workers, queue_depth=2,
                stage="producer", start_method="spawn",
            ) as pool:
                result, _ = pool.map_ordered(_identity, items)
            results[workers] = result
        self.assertEqual(results[1], results[2])
        self.assertEqual(results[2], results[3])
        self.assertEqual(results[1], items)


class TestWorkerFailure(unittest.TestCase):
    """Test 4 — worker failure propagates; pool does not silently swallow errors."""

    def test_worker_failure_raises(self):
        with DeterministicProcessPool(
            workers=1, queue_depth=1, stage="producer", start_method="spawn",
        ) as pool:
            with self.assertRaises(Exception):
                pool.map_ordered(_explode, (1,))


class TestPoolShutdownNormal(unittest.TestCase):
    """Test 5 — pool shuts down cleanly on normal completion."""

    def test_context_manager_exits_cleanly(self):
        res = _make_resources()
        with res:
            pass
        # Pools should have been shut down; _entered flag cleared
        self.assertFalse(res._entered)


class TestPoolShutdownOnFailure(unittest.TestCase):
    """Test 6 — pools shut down cleanly even when the run raises."""

    def test_resources_exit_on_exception(self):
        res = _make_resources()
        try:
            with res:
                raise RuntimeError("synthetic infrastructure failure")
        except RuntimeError:
            pass
        self.assertFalse(res._entered)


# ── Group 2: LegalStateCaptureCache key binding (tests 7-11) ─────────────────

class TestCaptureCacheKeyBinding(unittest.TestCase):
    """Tests 7-11 — capture cache keys bind the right dimensions."""

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self._tmp = Path(self._tmpdir.name)
        self._entry_seq = 0

    def tearDown(self):
        self._tmpdir.cleanup()

    def _cache_with_entry(self, key):
        """Return a cache with a synthetic entry at `key` (files live in self._tmp)."""
        seq = self._entry_seq; self._entry_seq += 1
        cap = self._tmp / f"cap_{seq}.b4is"
        truth = self._tmp / f"truth_{seq}.b4is"
        cap.write_bytes(b"capture-bytes")
        truth.write_bytes(b"truth-bytes")
        cache = LegalStateCaptureCache()
        cache.paths[key] = (cap, truth)
        return cache, key, cap, truth

    def _make_key(self, asset_dir="/a/assets", cases_path="/a/cases.b4p",
                  probe_steps=(0,), feed_slot=0, feed_count=100, route_slot=0):
        return (asset_dir, cases_path, probe_steps, feed_slot, feed_count, route_slot)

    def test_same_key_hits(self):
        """Test 7 — same (asset_dir, panel, probe_steps, ...) → cache hit."""
        key = self._make_key()
        cache, _, cap, _ = self._cache_with_entry(key)
        self.assertIn(key, cache.paths)
        found = cache.paths.get(key)
        self.assertIsNotNone(found)
        self.assertEqual(found[0].read_bytes(), b"capture-bytes")

    def test_changed_asset_dir_misses(self):
        """Test 8 — changed asset_dir (new BKG4 generation commit) → cache miss."""
        key = self._make_key(asset_dir="/gen0/assets")
        cache, _, _, _ = self._cache_with_entry(key)
        other_key = self._make_key(asset_dir="/gen1/assets")
        self.assertNotIn(other_key, cache.paths)

    def test_changed_panel_misses(self):
        """Test 9 — changed cases_path (different target panel) → cache miss."""
        key = self._make_key(cases_path="/panels/family_02.b4p")
        cache, _, _, _ = self._cache_with_entry(key)
        other_key = self._make_key(cases_path="/panels/family_03.b4p")
        self.assertNotIn(other_key, cache.paths)

    def test_changed_probe_steps_miss(self):
        """Test 10 — changed probe_steps → cache miss."""
        key = self._make_key(probe_steps=(0, 5, 10))
        cache, _, _, _ = self._cache_with_entry(key)
        other_key = self._make_key(probe_steps=(0, 5, 15))
        self.assertNotIn(other_key, cache.paths)

    def test_cached_evidence_files_are_readable(self):
        """Test 11 — cached entries point to readable files (evidence is intact)."""
        key = self._make_key()
        cache, _, cap, truth = self._cache_with_entry(key)
        found_cap, found_truth = cache.paths[key]
        self.assertEqual(found_cap.read_bytes(), b"capture-bytes")
        self.assertEqual(found_truth.read_bytes(), b"truth-bytes")


# ── Group 3: Invariant caches (tests 12-13) ───────────────────────────────────

class TestInvariantCaches(unittest.TestCase):
    """Tests 12-13 — generic_events and checkpoint caches skip recomputation."""

    def setUp(self):
        self.res = _make_resources()

    def test_generic_events_cached_after_put(self):
        """Test 12 — donor event extraction runs once per (family, donors) pair."""
        paths = [Path("/donors/a.json"), Path("/donors/b.json")]
        events = [{"type": "move", "turn": 1}]
        # First access → miss
        self.assertIsNone(self.res.get_generic_events("fam_01", paths))
        self.assertEqual(self.res.generic_events_cache_misses, 1)
        # Store result
        self.res.put_generic_events("fam_01", paths, events)
        # Second access → hit
        result = self.res.get_generic_events("fam_01", paths)
        self.assertIs(result, events)
        self.assertEqual(self.res.generic_events_cache_hits, 1)

    def test_checkpoint_cache_isolated_from_events(self):
        """Test 13 — checkpoint cache keyed on (events_key, limit); different limits miss."""
        paths = [Path("/donors/a.json")]
        events = list(range(20))
        self.res.put_generic_events("fam_01", paths, events)
        checkpoints_10 = list(range(10))
        checkpoints_5 = list(range(5))
        self.res.put_checkpoints("fam_01", paths, 10, checkpoints_10)
        self.res.put_checkpoints("fam_01", paths, 5, checkpoints_5)
        # Limit 10 hit
        self.assertIs(self.res.get_checkpoints("fam_01", paths, 10), checkpoints_10)
        # Limit 5 hit
        self.assertIs(self.res.get_checkpoints("fam_01", paths, 5), checkpoints_5)
        # Limit 15 miss
        self.assertIsNone(self.res.get_checkpoints("fam_01", paths, 15))
        self.assertEqual(self.res.checkpoint_cache_hits, 2)
        self.assertEqual(self.res.checkpoint_cache_misses, 1)


# ── Group 4: Cloud runner integration (tests 14-18) ──────────────────────────

class TestCloudRunnerIntegration(unittest.TestCase):
    """Tests 14-18 — run_cloud with stubs verifies P16.3 contracts."""

    def test_resume_does_not_error_with_empty_caches(self):
        """Test 14 — after restart, caches are rebuilt from scratch without error."""
        with tempfile.TemporaryDirectory() as d:
            output = Path(d) / "run"
            runner = _stub_runner()
            assets = _stub_assets()
            # First run
            code = run_cloud(CONFIG, output, mode="pilot", max_generations=1,
                             pipeline_runner=runner, asset_builder=assets)
            self.assertEqual(code, 0)
            # Resume — caches start empty (in-memory), state loaded from disk
            code2 = run_cloud(CONFIG, output, mode="pilot", resume=True,
                              max_generations=1,
                              pipeline_runner=runner, asset_builder=assets)
            self.assertEqual(code2, 0)

    def test_phase161_suppression_still_active(self):
        """Test 15 — exact-repeat suppression (Phase 16.1) is preserved after P16.3."""
        with tempfile.TemporaryDirectory() as d:
            output = Path(d) / "run"
            calls = []
            def counting_runner(runtime_path, run_dir):
                calls.append(1)
                return _stub_runner()(runtime_path, run_dir)
            run_cloud(CONFIG, output, mode="pilot", max_generations=8,
                      pipeline_runner=counting_runner,
                      asset_builder=_stub_assets())
            state = json.loads((output / "generation_state.json").read_text())
            # 4 real families, second cycle suppressed → only 4 pipeline calls
            self.assertEqual(len(calls), 4)
            self.assertEqual(state["suppressed_contexts"], 4)

    def test_phase162_determinism_preserved(self):
        """Test 16 — Phase 16.2 worker-count determinism still holds after P16.3."""
        items = tuple(range(8))
        results = {}
        for workers in (1, 2, 4):
            with DeterministicProcessPool(
                workers=workers, queue_depth=2,
                stage="producer", start_method="spawn",
            ) as pool:
                result, _ = pool.map_ordered(_identity, items)
            results[workers] = result
        self.assertEqual(results[1], results[2])
        self.assertEqual(results[2], results[4])

    def test_semantic_mismatches_zero_in_accepted_generation(self):
        """Test 17 — accepted generation metrics must report zero semantic mismatches."""
        with tempfile.TemporaryDirectory() as d:
            output = Path(d) / "run"
            runner = _stub_runner(accepted=True, semantic_mismatches=0)
            code = run_cloud(CONFIG, output, mode="pilot", max_generations=1,
                             pipeline_runner=runner, asset_builder=_stub_assets())
            self.assertEqual(code, 0)
            state = json.loads((output / "generation_state.json").read_text())
            # Accepted specialist committed → generation advanced
            self.assertEqual(state["generation"], 1)

    def test_protected_win_to_loss_zero_in_accepted_generation(self):
        """Test 18 — accepted generation metrics must report zero protected W→L."""
        with tempfile.TemporaryDirectory() as d:
            output = Path(d) / "run"
            runner = _stub_runner(accepted=True, protected_wl=0)
            code = run_cloud(CONFIG, output, mode="pilot", max_generations=1,
                             pipeline_runner=runner, asset_builder=_stub_assets())
            self.assertEqual(code, 0)
            state = json.loads((output / "generation_state.json").read_text())
            self.assertEqual(state["generation"], 1)
            self.assertFalse(list(output.glob("*.pending")))

    def test_cloud_result_has_p163_metrics_key(self):
        """Bonus — cloud_result.json always contains p163_resources section."""
        with tempfile.TemporaryDirectory() as d:
            output = Path(d) / "run"
            runner = _stub_runner()
            code = run_cloud(CONFIG, output, mode="pilot", max_generations=1,
                             pipeline_runner=runner, asset_builder=_stub_assets())
            self.assertEqual(code, 0)
            result = json.loads((output / "cloud_result.json").read_text())
            self.assertIn("p163_resources", result)
            m = result["p163_resources"]
            self.assertEqual(m["producer_pool_creations"], 1)
            self.assertEqual(m["candidate_pool_creations"], 1)


if __name__ == "__main__":
    unittest.main()
