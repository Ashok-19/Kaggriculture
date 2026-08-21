from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path


class CorpusCoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            from tools.edit_corpus import corpus
        except ImportError as exc:
            raise AssertionError(f"corpus module is not implemented yet: {exc}") from exc
        cls.c = corpus

    def test_replay_to_native_tapes_uses_steps_after_initial_placeholder(self):
        replay = {
            "info": {"EpisodeId": 42, "TeamNames": ["A", "B"], "seed": 1234},
            "rewards": [10, 20],
            "steps": [
                [{"action": {"farmer": ["PASS"]}}, {"action": {"farmer": ["PASS"]}}],
                [{"action": {"farmer": ["NORTH"], "hands": [], "market": []}}, {"action": {"farmer": ["SOUTH"], "hands": [], "market": []}}],
                [{"action": None}, {"action": {"farmer": ["EAST"], "hands": [], "market": []}}],
            ],
        }
        rows = self.c.replay_to_native_tapes(replay, {"date": "2026-08-20", "daily_rank_by_team": {"A": 1, "B": 2}})
        self.assertEqual(len(rows), 2)
        a, b = rows
        self.assertEqual(a["source_seat"], 0)
        self.assertEqual(a["actions"][0]["farmer"], ["NORTH"])
        self.assertEqual(a["actions"][1], {"farmer": ["PASS"], "hands": [], "market": []})
        self.assertEqual(b["actions"][1]["farmer"], ["EAST"])
        self.assertEqual(a["prov_episode_id"], 42)
        self.assertEqual(a["prov_team_name"], "A")
        self.assertEqual(a["prov_daily_rank"], 1)

    def test_build_case_candidates_creates_both_candidate_seats_without_feature_identity_leakage(self):
        tape = {
            "source_seat": 0,
            "actions": [{"farmer": ["PASS"], "hands": [], "market": []}],
            "prov_episode_id": 42,
            "prov_team_name": "A",
            "prov_opponent_name": "B",
            "prov_seed": 1234,
            "prov_date": "2026-08-20",
            "prov_daily_rank": 1,
            "tape_semantic_id": "tape-a",
        }
        cases = self.c.build_case_candidates([tape])
        self.assertEqual([x["seat"] for x in cases], [0, 1])
        self.assertEqual(cases[0]["prov_seed"], 1234)
        for case in cases:
            model = json.loads(case["model_context_json"])
            self.assertNotIn("seed", model)
            self.assertNotIn("team", json.dumps(model).lower())
            self.assertNotIn("episode", json.dumps(model).lower())
            self.assertNotIn("replay", json.dumps(model).lower())

    def test_case_identity_separates_same_tape_across_meta_dates(self):
        base = {
            "source_seat": 0, "actions": [{"farmer": ["PASS"], "hands": [], "market": []}],
            "prov_episode_id": 1, "prov_team_name": "A", "prov_opponent_name": "B",
            "prov_seed": 7, "prov_daily_rank": 1, "tape_semantic_id": "same",
        }
        a = self.c.build_case_candidates([{**base, "prov_date": "2026-08-19"}])
        b = self.c.build_case_candidates([{**base, "prov_date": "2026-08-20"}])
        self.assertNotEqual(a[0]["case_id"], b[0]["case_id"])
        self.assertNotIn("date", json.loads(a[0]["model_context_json"]))

    def test_representative_case_selection_is_deterministic_balanced_and_hard_first(self):
        rows = []
        margins = [-1000, -500, 0, 100, 3000, 20000, -2000, 50, 500, 10000]
        for i, margin in enumerate(margins):
            rows.append({
                "case_id": f"c{i}",
                "seat": i % 2,
                "baseline_margin": margin,
                "prov_daily_rank": 1 + (i % 5),
                "prov_date": "2026-08-20",
            })
        a = self.c.select_representative_cases(rows, max_cases=6)
        b = self.c.select_representative_cases(list(reversed(rows)), max_cases=6)
        self.assertEqual([x["case_id"] for x in a], [x["case_id"] for x in b])
        self.assertEqual({x["seat"] for x in a}, {0, 1})
        self.assertTrue(any(x["baseline_margin"] < 0 for x in a))
        self.assertTrue(any(0 <= x["baseline_margin"] <= 5000 for x in a))

    def test_canonical_edit_identity_changes_with_parent_action_or_child(self):
        action = {"operation": "CHANGE_WINDOW", "target_node": 0, "prefix_step": 72}
        x = self.c.canonical_edit_identity("p1", action, "c1")
        self.assertEqual(x, self.c.canonical_edit_identity("p1", dict(action), "c1"))
        self.assertNotEqual(x, self.c.canonical_edit_identity("p2", action, "c1"))
        self.assertNotEqual(x, self.c.canonical_edit_identity("p1", {**action, "prefix_step": 73}, "c1"))
        self.assertNotEqual(x, self.c.canonical_edit_identity("p1", action, "c2"))

    def test_exact_label_derivation_preserves_lexicographic_components(self):
        baseline = [-100, 1000, 10000, 100]
        candidate = [200, -50, 6000, 100]
        effects = [3, 2, 1, 0]
        activations = [4, 3, 2, 0]
        rows = [self.c.derive_case_label(b, c, ch, ac) for b, c, ch, ac in zip(baseline, candidate, effects, activations)]
        agg = self.c.derive_edit_labels(rows, fragile_margin=500, severe_delta=-3000)
        self.assertEqual(agg["l2w_count"], 1)
        self.assertEqual(agg["w2l_count"], 1)
        self.assertEqual(agg["severe_regression_count"], 1)
        self.assertEqual(agg["losses_before"], 1)
        self.assertEqual(agg["losses_after"], 1)
        self.assertEqual(agg["fragile_wins_before"], 1)
        self.assertEqual(agg["fragile_wins_after"], 2)
        self.assertFalse(agg["strict_safe"])
        self.assertFalse(agg["exact_noop"])
        self.assertEqual(agg["action_change_total"], 6)
        self.assertEqual(agg["activation_total"], 9)
        self.assertEqual(agg["lex_losses_after"], 1)
        self.assertEqual(agg["lex_new_losses"], 1)

    def test_parent_relative_trace_identity_controls_exact_noop(self):
        rows = [
            self.c.derive_case_label(100, 100, 5, 2, parent_action_trace_hash=77, candidate_action_trace_hash=77),
            self.c.derive_case_label(-100, -100, 5, 2, parent_action_trace_hash=88, candidate_action_trace_hash=88),
        ]
        agg = self.c.derive_edit_labels(rows)
        self.assertTrue(agg["exact_noop"])
        self.assertFalse(agg["behavior_changing"])
        changed = [self.c.derive_case_label(100, 100, 5, 2, parent_action_trace_hash=77, candidate_action_trace_hash=78)]
        self.assertTrue(self.c.derive_edit_labels(changed)["behavior_changing"])

    def test_exact_noop_requires_equal_margins_and_zero_action_changes(self):
        rows = [self.c.derive_case_label(100, 100, 0, 2), self.c.derive_case_label(-100, -100, 0, 1)]
        agg = self.c.derive_edit_labels(rows)
        self.assertTrue(agg["exact_noop"])
        self.assertFalse(agg["behavior_changing"])
        self.assertEqual(agg["mean_margin_delta"], 0.0)

    def test_pending_replay_shards_is_idempotent_and_rejects_source_mutation(self):
        source = [
            {"date": "2026-08-19", "replay_shard": "replays_2026-08-19.parquet", "replay_shard_sha256": "a"},
            {"date": "2026-08-20", "replay_shard": "replays_2026-08-20.parquet", "replay_shard_sha256": "b"},
        ]
        processed = {"2026-08-19": {"status": "complete", "source_replay_shard_sha256": "a"}}
        self.assertEqual([x["date"] for x in self.c.pending_replay_shards(source, processed)], ["2026-08-20"])
        changed = {"2026-08-19": {"status": "complete", "source_replay_shard_sha256": "WRONG"}}
        with self.assertRaisesRegex(RuntimeError, "changed after processing"):
            self.c.pending_replay_shards(source, changed)

    def test_model_payload_leakage_audit_rejects_private_identity_keys(self):
        self.c.assert_model_payload_safe({"seat": 0, "baseline_margin": -10, "public_context": {"day": 3}})
        for key in ("team_name", "opponent", "seed", "episode_id", "replay_id"):
            with self.subTest(key=key):
                with self.assertRaisesRegex(ValueError, "forbidden model feature"):
                    self.c.assert_model_payload_safe({key: "x"})


if __name__ == "__main__":
    unittest.main()
