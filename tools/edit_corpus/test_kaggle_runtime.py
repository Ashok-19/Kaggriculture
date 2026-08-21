from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd


class KaggleRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            from tools.edit_corpus import kaggle_runtime as r
        except ImportError as exc:
            raise AssertionError(f"Kaggle runtime is not implemented yet: {exc}") from exc
        cls.r = r

    def test_source_rows_from_archive_uses_completed_replay_shards_only(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "replays_2026-08-19.parquet").write_bytes(b"a")
            (root / "replays_2026-08-20.parquet").write_bytes(b"b")
            (root / "processed_days.json").write_text(json.dumps({
                "2026-08-19": {"status": "complete", "replay_shard": "replays_2026-08-19.parquet", "replay_shard_sha256": "sha-a"},
                "2026-08-20": {"status": "partial", "replay_shard": "replays_2026-08-20.parquet", "replay_shard_sha256": "sha-b"},
            }))
            rows = self.r.source_rows_from_archive(root)
            self.assertEqual(rows, [{"date": "2026-08-19", "replay_shard": "replays_2026-08-19.parquet", "replay_shard_sha256": "sha-a"}])

    def test_commit_day_is_idempotent_rejects_source_mutation_and_preserves_prior_rows(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self.r.initialize_empty_corpus(root)
            first_cases = [{"case_id": "c1", "date": "2026-08-19", "model_context_json": '{"seat":0}', "prov_seed": 1}]
            first_edits = [{"edit_id": "e1", "date": "2026-08-19", "parent_id": "p", "child_id": "q"}]
            first_labels = [{"edit_id": "e1", "case_id": "c1", "date": "2026-08-19", "baseline_margin": -1, "candidate_margin": 1}]
            self.assertTrue(self.r.commit_day(root, date="2026-08-19", source_sha="sha-a", cases=first_cases, edits=first_edits, labels=first_labels))
            self.assertFalse(self.r.commit_day(root, date="2026-08-19", source_sha="sha-a", cases=first_cases, edits=first_edits, labels=first_labels))
            with self.assertRaisesRegex(RuntimeError, "changed after processing"):
                self.r.commit_day(root, date="2026-08-19", source_sha="other", cases=first_cases, edits=first_edits, labels=first_labels)

            second_cases = [{"case_id": "c2", "date": "2026-08-20", "model_context_json": '{"seat":1}', "prov_seed": 2}]
            second_edits = [{"edit_id": "e2", "date": "2026-08-20", "parent_id": "p", "child_id": "r"}]
            second_labels = [{"edit_id": "e2", "case_id": "c2", "date": "2026-08-20", "baseline_margin": 5, "candidate_margin": 6}]
            self.assertTrue(self.r.commit_day(root, date="2026-08-20", source_sha="sha-b", cases=second_cases, edits=second_edits, labels=second_labels))
            counts = self.r.consolidate_corpus(root)
            self.assertEqual(counts, {"cases": 2, "edits": 2, "labels": 2})
            self.assertEqual(set(pd.read_parquet(root / "cases.parquet")["case_id"]), {"c1", "c2"})
            self.assertEqual(set(pd.read_parquet(root / "edits.parquet")["edit_id"]), {"e1", "e2"})

    def test_commit_day_rejects_model_feature_identity_leakage(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self.r.initialize_empty_corpus(root)
            cases = [{"case_id": "c1", "date": "2026-08-19", "model_context_json": '{"seat":0,"seed":123}', "prov_seed": 123}]
            with self.assertRaisesRegex(ValueError, "forbidden model feature"):
                self.r.commit_day(root, date="2026-08-19", source_sha="sha", cases=cases, edits=[], labels=[])

    def test_base_and_overlay_parent_tables_are_deduplicated_by_identity(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self.r.initialize_empty_corpus(root)
            base = {"base_id": "b1", "controller_structure_json": '{"root":1}', "combined_agent_identity": "x"}
            parent = {"parent_id": "p1", "base_id": "b1", "overlay_semantic_id": "o1", "overlay_genome_json": '{"rules":[]}'}
            self.r.upsert_structures(root, [base, dict(base)], [parent, dict(parent)])
            self.assertEqual(len(pd.read_parquet(root / "base_structures.parquet")), 1)
            self.assertEqual(len(pd.read_parquet(root / "parents.parquet")), 1)

    def test_consolidation_handles_parquet_list_columns_from_numpy(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self.r.initialize_empty_corpus(root)
            cases = [{
                "case_id": "c1",
                "date": "2026-08-19",
                "model_context_json": '{"seat":0}',
                "context_probe_steps": [72, 144],
                "public_context_values": [1, 2, 3],
            }]
            edits = [{"edit_id": "e1", "parent_id": "p", "child_id": "q"}]
            labels = [{"edit_id": "e1", "case_id": "c1", "baseline_margin": 1, "candidate_margin": 2}]
            self.r.commit_day(root, date="2026-08-19", source_sha="sha", cases=cases, edits=edits, labels=labels)
            counts = self.r.consolidate_corpus(root)
            self.assertEqual(counts["cases"], 1)
            got = pd.read_parquet(root / "cases.parquet").iloc[0]
            self.assertEqual(list(got["context_probe_steps"]), [72, 144])
            self.assertEqual(list(got["public_context_values"]), [1, 2, 3])

    def test_consolidation_recomputes_global_edit_targets_from_all_case_labels(self):
        from tools.edit_corpus.corpus import derive_case_label
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self.r.initialize_empty_corpus(root)
            cases1 = [{"case_id": "c1", "date": "2026-08-19", "model_context_json": '{"seat":0}'}]
            edits = [{"edit_id": "e1", "parent_id": "p", "child_parent_id": "q", "operation": "CHANGE_WINDOW"}]
            l1 = derive_case_label(-10, 20, 1, 1); l1.update({"edit_id": "e1", "case_id": "c1", "date": "2026-08-19"})
            self.r.commit_day(root, date="2026-08-19", source_sha="a", cases=cases1, edits=edits, labels=[l1])
            cases2 = [{"case_id": "c2", "date": "2026-08-20", "model_context_json": '{"seat":1}'}]
            l2 = derive_case_label(100, -1, 1, 1); l2.update({"edit_id": "e1", "case_id": "c2", "date": "2026-08-20"})
            self.r.commit_day(root, date="2026-08-20", source_sha="b", cases=cases2, edits=edits, labels=[l2])
            self.r.consolidate_corpus(root)
            targets = pd.read_parquet(root / "edit_targets.parquet")
            self.assertEqual(len(targets), 1)
            row = targets.iloc[0]
            self.assertEqual(int(row["l2w_count"]), 1)
            self.assertEqual(int(row["w2l_count"]), 1)
            self.assertFalse(bool(row["strict_safe"]))

    def test_manifest_counts_use_global_targets_and_report_edit_dedup(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self.r.initialize_empty_corpus(root)
            for date, case_id, source_sha, delta in [
                ("2026-08-19", "c1", "a", 2),
                ("2026-08-20", "c2", "b", -3),
            ]:
                cases = [{"case_id": case_id, "date": date, "model_context_json": '{"seat":0}'}]
                # Same semantic edit definition on both dates should consolidate once.
                edits = [{"edit_id": "e1", "parent_id": "p", "child_id": "q", "operation": "CHANGE_WINDOW"}]
                labels = [{
                    "edit_id": "e1", "case_id": case_id, "baseline_margin": 10,
                    "candidate_margin": 10 + delta, "margin_delta": delta,
                    "l2w": False, "w2l": False, "protected": True,
                    "action_changes": 1, "activations": 1, "behavior_changing": True,
                }]
                self.r.commit_day(root, date=date, source_sha=source_sha, cases=cases, edits=edits, labels=labels)
            self.r.consolidate_corpus(root)
            m = self.r.write_corpus_manifest(root, replay_archive_version="r", engine_identity="e", generator_version="g")
            self.assertEqual(m["counts"]["edit_rows"], 1)
            self.assertEqual(m["counts"]["edit_target_rows"], 1)
            self.assertEqual(m["counts"]["dedup_counts"]["semantic_edit_rows_removed"], 1)
            self.assertEqual(m["counts"]["label_counts"]["behavior_changing"], 1)

    def test_manifest_validation_detects_tampered_consolidated_file(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self.r.initialize_empty_corpus(root)
            self.r.commit_day(root, date="2026-08-19", source_sha="sha", cases=[], edits=[], labels=[])
            self.r.consolidate_corpus(root)
            manifest = self.r.write_corpus_manifest(root, replay_archive_version="v1", engine_identity="cuda1", generator_version="g1")
            self.assertEqual(self.r.validate_corpus(root)["manifest_sha256"], manifest["manifest_sha256"])
            (root / "cases.parquet").write_bytes((root / "cases.parquet").read_bytes() + b"tamper")
            with self.assertRaisesRegex(RuntimeError, "(size|hash) mismatch"):
                self.r.validate_corpus(root)


if __name__ == "__main__":
    unittest.main()
