from __future__ import annotations

import json
import struct
import tempfile
import unittest
from pathlib import Path

import pandas as pd


class EditCorpusPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            from tools.edit_corpus import pipeline
        except ImportError as exc:
            raise AssertionError(f"edit corpus pipeline is not implemented yet: {exc}") from exc
        cls.p = pipeline

    def test_stream_replay_shard_writes_native_tapes_and_candidate_cases(self):
        replay = {
            "info": {"EpisodeId": 101, "TeamNames": ["A", "B"], "seed": 77},
            "rewards": [1, 2],
            "steps": [
                [{"action": {"farmer": ["PASS"]}}, {"action": {"farmer": ["PASS"]}}],
                [{"action": {"farmer": ["NORTH"], "hands": [], "market": []}}, {"action": {"farmer": ["SOUTH"], "hands": [], "market": []}}],
            ],
        }
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            shard = root / "replays_2026-08-20.parquet"
            pd.DataFrame([{"episode_id": 101, "replay_json": json.dumps(replay)}]).to_parquet(shard, index=False)
            tapes, cases, tape_paths = self.p.stream_replay_shard_to_cases(
                shard,
                date="2026-08-20",
                out_dir=root / "tapes",
                daily_rank_by_team={"A": 1, "B": 2},
            )
            self.assertEqual(len(tapes), 2)
            self.assertEqual(len(cases), 4)
            self.assertEqual(len(tape_paths), 2)
            for path in tape_paths.values():
                obj = json.loads(Path(path).read_text())
                self.assertIn("actions", obj)
                self.assertIn("source", obj)
            self.assertEqual({c["seat"] for c in cases}, {0, 1})

    def test_build_edit_plan_is_deterministic_broad_and_parent_explicit(self):
        context = [
            {"feature": "OWN_MONEY", "index": 0, "value": 2500},
            {"feature": "MARKET_PRICE", "index": 3, "value": 120},
            {"feature": "OPP_UNITS", "index": 0, "value": 8},
            {"feature": "OWN_TILE_KIND", "index": 44, "value": 5},
        ]
        a = self.p.build_edit_plan("base-345", context, seed=123, seed_parent_count=4, candidates_per_parent=32)
        b = self.p.build_edit_plan("base-345", list(reversed(context)), seed=123, seed_parent_count=4, candidates_per_parent=32)
        self.assertEqual(a, b)
        self.assertGreaterEqual(len(a["parents"]), 2)
        self.assertGreaterEqual(len(a["edits"]), 40)
        ops = {x["operation"] for x in a["edits"]}
        self.assertGreaterEqual(len(ops), 8)
        parent_ids = {x["parent_id"] for x in a["parents"]}
        self.assertTrue(all(x["parent_id"] in parent_ids for x in a["edits"]))
        self.assertTrue(all(x["child_parent_id"] in parent_ids for x in a["edits"]))
        self.assertEqual(len({x["edit_id"] for x in a["edits"]}), len(a["edits"]))

    def test_build_label_rows_uses_parent_relative_trace_and_aggregates(self):
        from tools.edit_corpus.cuda_results import EditCudaResult
        cases = [
            {"case_id": "c0", "seat": 0, "baseline_margin": 5, "model_context_json": '{"seat":0}'},
            {"case_id": "c1", "seat": 1, "baseline_margin": -5, "model_context_json": '{"seat":1}'},
        ]
        edits = [{
            "edit_id": "e",
            "parent_id": "p",
            "child_parent_id": "q",
            "parent_overlay_semantic_id": "g0",
            "child_overlay_semantic_id": "g1",
            "operation": "CHANGE_WINDOW",
        }]
        results = {
            "g0": [EditCudaResult(10, 5, 0, 0, 0, 111), EditCudaResult(5, 10, 0, 0, 0, 222)],
            "g1": [EditCudaResult(10, 5, 3, 4, 0, 111), EditCudaResult(20, 10, 2, 3, 0, 333)],
        }
        label_rows, target_rows = self.p.build_label_rows(edits, cases, results)
        self.assertEqual(len(label_rows), 2)
        self.assertEqual(len(target_rows), 1)
        first = next(x for x in label_rows if x["case_id"] == "c0")
        second = next(x for x in label_rows if x["case_id"] == "c1")
        self.assertTrue(first["action_trace_equal"])
        self.assertFalse(first["behavior_changing"])
        self.assertTrue(second["l2w"])
        self.assertTrue(second["behavior_changing"])
        self.assertEqual(target_rows[0]["l2w_count"], 1)

    def test_r0_public_context_never_reads_hidden_truth(self):
        from tools.bkg4_search.information_state_io import HEADER, registry_manifest
        manifest = registry_manifest()
        registry = len(manifest)
        width = registry + 2
        probes = [72, 144]
        frames = list(range(width * len(probes)))
        raw = HEADER.pack(b"B4ISR01\0", 1, 1, len(probes), width, registry, 0, 0, 2)
        raw += struct.pack("<2i", *probes)
        raw += struct.pack(f"<{len(frames)}i", *frames)
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "r0.bin"
            path.write_bytes(raw)
            contexts, observed = self.p.load_public_context(path, ["case-x"])
            self.assertEqual(len(contexts), 1)
            self.assertEqual(contexts[0]["case_id"], "case-x")
            self.assertEqual(contexts[0]["context_probe_steps"], probes)
            self.assertEqual(len(contexts[0]["public_context_values"]), len(frames))
            self.assertTrue(observed)
            self.assertFalse(any("opp_total_item" in json.dumps(x) for x in observed))


if __name__ == "__main__":
    unittest.main()
