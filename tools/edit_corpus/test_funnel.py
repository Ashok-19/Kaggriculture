from __future__ import annotations

import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest import mock

from tools.bkg4_search.module_codec import ModuleGenome
from tools.edit_corpus.cuda_results import EditCudaResult


class EditCorpusFunnelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from tools.edit_corpus import corpus, notebook_runner, pipeline
        cls.c = corpus
        cls.n = notebook_runner
        cls.p = pipeline

    def test_screening_panel_is_deterministic_subset_of_final_panel(self):
        final_cases = []
        for i in range(24):
            final_cases.append({
                "case_id": f"c{i:02d}",
                "seat": i % 2,
                "baseline_margin": (-100 if i < 8 else 100 if i < 16 else 10000) + i,
            })
        a = self.c.select_screening_cases(final_cases, max_cases=8)
        b = self.c.select_screening_cases(list(reversed(final_cases)), max_cases=8)
        self.assertEqual([x["case_id"] for x in a], [x["case_id"] for x in b])
        self.assertEqual(len(a), 8)
        self.assertTrue({x["case_id"] for x in a}.issubset({x["case_id"] for x in final_cases}))

    def test_screening_selector_retains_promising_negatives_and_diversity_deterministically(self):
        edits = []
        screening = []
        for i in range(24):
            parent = f"p{i % 4}"
            op = f"OP{i % 6}"
            edit_id = f"e{i:02d}"
            edits.append({
                "edit_id": edit_id,
                "parent_overlay_semantic_id": parent,
                "child_overlay_semantic_id": f"g{i}",
                "operation": op,
                "target_kind": f"kind{i % 3}",
                "target_node": i % 5,
            })
            if i < 8:
                mean_delta, strict_safe, w2l, severe = 1000.0 - i, True, 0, 0
            elif i < 16:
                mean_delta, strict_safe, w2l, severe = -5.0 * (i - 7), False, 1 if i % 2 else 0, 0
            else:
                mean_delta, strict_safe, w2l, severe = -5000.0 - i, False, 0, 1
            screening.append({
                "edit_id": edit_id,
                "screening_case_count": 128,
                "screening_losses_before": 2,
                "screening_losses_after": 2 + w2l,
                "screening_l2w": 0,
                "screening_w2l": w2l,
                "screening_severe_regressions": severe,
                "screening_fragile_wins_after": 1,
                "screening_mean_delta": mean_delta,
                "screening_min_delta": int(mean_delta),
                "screening_worst_candidate_margin": 100 + int(mean_delta),
                "screening_worst_protected_win_delta": int(mean_delta),
                "screening_strict_safe": strict_safe,
                "screening_exact_noop": False,
                "screening_behavior_changing": True,
                "screening_action_changes": 10 + i,
                "screening_activations": 20 + i,
            })
        kept_a, annotated_a = self.p.select_screened_candidates(edits, screening, target_count=12)
        kept_b, annotated_b = self.p.select_screened_candidates(list(reversed(edits)), list(reversed(screening)), target_count=12)
        ids_a = [x["edit_id"] for x in kept_a]
        ids_b = [x["edit_id"] for x in kept_b]
        self.assertEqual(ids_a, ids_b)
        self.assertEqual(annotated_a, annotated_b)
        selected_meta = [x for x in annotated_a if x["screening_retained"]]
        self.assertEqual(len(selected_meta), 12)
        buckets = {x["screening_bucket"] for x in selected_meta}
        self.assertIn("exploitation", buckets)
        self.assertIn("negative", buckets)
        self.assertIn("diversity", buckets)
        self.assertTrue(any(x["screening_w2l"] or x["screening_severe_regressions"] for x in selected_meta))
        parent_counts = Counter(next(e["parent_overlay_semantic_id"] for e in edits if e["edit_id"] == x["edit_id"]) for x in selected_meta)
        op_counts = Counter(next(e["operation"] for e in edits if e["edit_id"] == x["edit_id"]) for x in selected_meta)
        self.assertLess(max(parent_counts.values()), len(selected_meta))
        self.assertLess(max(op_counts.values()), len(selected_meta))
        positive_ids = {x["edit_id"] for x in selected_meta if x["screening_bucket"] == "exploitation"}
        self.assertTrue(positive_ids)
        self.assertTrue(all(next(s for s in screening if s["edit_id"] == eid)["screening_strict_safe"] for eid in positive_ids))
        for row in annotated_a:
            for key in row:
                if key not in {"edit_id", "date"}:
                    self.assertTrue(key.startswith("screening_"), key)

    def test_process_one_day_screens_all_candidates_but_full_labels_only_retained(self):
        genomes = [ModuleGenome(prefix_step=216 + i, rules=()) for i in range(4)]
        ids = [g.semantic_id() for g in genomes]
        parents = [{
            "parent_id": f"p{i}",
            "base_id": "base",
            "overlay_semantic_id": ids[i],
            "overlay_genome_json": json.dumps(genomes[i].canonical(), sort_keys=True, separators=(",", ":")),
            "overlay_prefix_step": genomes[i].prefix_step,
            "overlay_rule_count": 0,
        } for i in range(4)]
        edits = [{
            "edit_id": f"e{i}",
            "base_id": "base",
            "parent_id": "p0",
            "child_parent_id": f"p{i}",
            "parent_overlay_semantic_id": ids[0],
            "child_overlay_semantic_id": ids[i],
            "operation": "CHANGE_WINDOW",
            "target_kind": "window",
            "target_node": 0,
        } for i in range(1, 4)]
        plan = {"parents": parents, "edits": edits}
        all_cases = [{
            "case_id": f"c{i}", "seat": i % 2, "prov_seed": 7 + i,
            "prov_date": "2026-08-22", "prov_daily_rank": i + 1,
            "tape_semantic_id": f"t{i}", "runtime_tape_path": f"tape{i}.b4tp",
        } for i in range(4)]

        baseline = [EditCudaResult(100 + i, 100, 0, 0, 0, 10 + i) for i in range(4)]
        screen_results = {
            ids[0]: [EditCudaResult(100, 100, 0, 0, 0, 1), EditCudaResult(100, 100, 0, 0, 0, 1)],
            ids[1]: [EditCudaResult(120, 100, 2, 2, 0, 2), EditCudaResult(120, 100, 2, 2, 0, 2)],
            ids[2]: [EditCudaResult(90, 100, 3, 3, 0, 3), EditCudaResult(90, 100, 3, 3, 0, 3)],
            ids[3]: [EditCudaResult(100, 100, 0, 0, 0, 1), EditCudaResult(100, 100, 0, 0, 0, 1)],
        }
        captured: dict[str, object] = {}
        calls: list[tuple[int, int]] = []

        def fake_eval(_binary, _assets, cases_tsv, eval_genomes, _tmp, **_kwargs):
            calls.append((len(eval_genomes), sum(1 for _ in Path(cases_tsv).read_text().splitlines())))
            if len(calls) == 1:
                return {eval_genomes[0].semantic_id(): baseline}
            if len(calls) == 2:
                return {g.semantic_id(): screen_results[g.semantic_id()] for g in eval_genomes}
            out = {}
            for g in eval_genomes:
                # Deliberately different from screening so final labels can be proven authoritative.
                out[g.semantic_id()] = [EditCudaResult(300 + j, 100, 7, 8, 0, 1000 + j) for j in range(4)]
            return out

        def fake_commit(_root, **kwargs):
            captured.update(kwargs)
            return True

        contexts = [{
            "case_id": f"c{i}", "model_context_json": '{"context_schema":"x"}',
            "context_probe_steps": [72], "public_context_width": 1,
            "public_context_registry_width": 1, "public_context_values": [1],
        } for i in range(4)]
        with tempfile.TemporaryDirectory() as td, \
             mock.patch.object(self.n, "_parent_assets", return_value={
                 "base_id": "base", "route_ids": (22,), "controller_json": Path(td) / "controller.json",
                 "controller": Path("controller"), "routes": Path("routes"), "legacy_routes": Path("legacy"),
                 "strategic_route_slot": 19, "controller_semantic_hash": "h", "controller_nodes": 1, "controller_bb_slots": 1,
             }), \
             mock.patch.object(self.n, "_daily_rank_map", return_value={}), \
             mock.patch.object(self.n, "_daily_episode_metadata", return_value={}), \
             mock.patch.object(self.n, "stream_replay_shard_to_cases", return_value=([], all_cases, {})), \
             mock.patch.object(self.n, "_capture_context", return_value=(contexts, [])), \
             mock.patch.object(self.n, "build_edit_plan", return_value=plan), \
             mock.patch.object(self.n, "_evaluate_genomes", side_effect=fake_eval), \
             mock.patch.object(self.n, "upsert_structures"), \
             mock.patch.object(self.n, "commit_day", side_effect=fake_commit), \
             mock.patch.object(self.n, "_base_structure_row", return_value={"base_id": "base"}):
            (Path(td) / "controller.json").write_text("{}")
            result = self.n.process_one_day(
                date_row={"date": "2026-08-22", "replay_shard": "x.parquet", "replay_shard_sha256": "sha"},
                archive_root=Path(td), stage_root=Path(td) / "stage", support_root=Path(td),
                support_manifest={"bkg4_abi": 4, "feature_registry": 1}, binaries={"edit_eval": Path("eval"), "capture": Path("capture")},
                tmp_root=Path(td) / "tmp", max_replays=None, max_cases=4, screen_cases=2,
                seed_parent_count=0, candidates_per_parent=16, eval_chunk_genomes=16,
                replay_converter_workers=1, cuda_devices=(0, 1), screen_target_cuda_games=64,
                final_retained_edits=2, eval_gpu_memory_fraction=0.7, probes=(72,),
            )
        self.assertEqual(calls[0], (1, 4))
        self.assertEqual(calls[1], (4, 2))
        self.assertLess(calls[2][0], calls[1][0])
        self.assertEqual(calls[2][1], 4)
        self.assertEqual(result["screened_edits"], 3)
        self.assertEqual(result["retained_edits"], 2)
        labels = captured["labels"]
        self.assertEqual(len(labels), 8)
        self.assertTrue(all(int(row["candidate_margin"]) >= 200 for row in labels))
        self.assertIn("screening", captured)
        self.assertEqual(len(captured["screening"]), 3)


if __name__ == "__main__":
    unittest.main()
