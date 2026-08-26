from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tools.bkg4_search.rlrmb.canonical import (
    action_fragment_content_id,
    action_sequence_semantic_id,
    candidate_semantic_id,
    canonical_action,
    canonical_candidate,
    donor_fragment_diversity,
)


class CandidateCanonicalizationTests(unittest.TestCase):
    def test_mapping_insertion_order_is_irrelevant(self) -> None:
        first = {"start": 24, "duration": 12, "parameters": {"b": 2, "a": 1}}
        second = {"parameters": {"a": 1, "b": 2}, "duration": 12, "start": 24}
        self.assertEqual(canonical_candidate(first), canonical_candidate(second))
        self.assertEqual(candidate_semantic_id(first), candidate_semantic_id(second))

    def test_market_order_is_semantic(self) -> None:
        first = {"market": [["HIRE"], ["BUY_LAND"]]}
        second = {"market": [["BUY_LAND"], ["HIRE"]]}
        self.assertNotEqual(action_sequence_semantic_id([first]), action_sequence_semantic_id([second]))

    def test_hands_order_is_semantic(self) -> None:
        first = {"hands": [["NORTH"], ["SOUTH"]]}
        second = {"hands": [["SOUTH"], ["NORTH"]]}
        self.assertNotEqual(action_sequence_semantic_id([first]), action_sequence_semantic_id([second]))

    def test_confirmed_empty_and_noop_forms_collapse(self) -> None:
        expected = {"farmer": ["PASS"], "hands": [], "market": []}
        self.assertEqual(canonical_action({}), expected)
        self.assertEqual(canonical_action({"farmer": None, "hands": None, "market": None}), expected)
        self.assertEqual(canonical_action({"farmer": ["NOT_AN_OP"]}), expected)
        self.assertEqual(canonical_action({"farmer": ["PICKUP", "WHEAT"]})["farmer"], ["PICKUP", "WHEAT", 1])
        self.assertEqual(canonical_action({"market": [["NOT_AN_ORDER"]]})["market"], [[]])

    def test_duplicate_fragments_ignore_donor_source(self) -> None:
        fragment = [{"farmer": ["NORTH"]}, {"market": [["HIRE"]]}]
        left = [{}, *fragment, {}]
        right = [{"farmer": ["PASS"]}, *fragment, {"hands": [], "market": []}]
        self.assertEqual(
            action_fragment_content_id(left, 1, 2),
            action_fragment_content_id(right, 1, 2),
        )

    def test_material_fragment_change_changes_identity(self) -> None:
        left = [{"farmer": ["NORTH"]}, {"market": [["HIRE"]]}]
        right = [{"farmer": ["SOUTH"]}, {"market": [["HIRE"]]}]
        self.assertNotEqual(
            action_fragment_content_id(left, 0, 2),
            action_fragment_content_id(right, 0, 2),
        )

    def test_temporal_candidate_position_is_semantic(self) -> None:
        content_id = action_sequence_semantic_id([{"farmer": ["NORTH"]}])
        at_24 = {"start": 24, "duration": 1, "fragment_content_id": content_id}
        at_48 = {"start": 48, "duration": 1, "fragment_content_id": content_id}
        self.assertNotEqual(candidate_semantic_id(at_24), candidate_semantic_id(at_48))

    def test_source_aliases_do_not_change_candidate_identity(self) -> None:
        semantic = {"start": 24, "duration": 12, "fragment_content_id": "fragment-1"}
        left = {**semantic, "donor_path": "/tmp/route_10.json", "donor_route_id": 10, "candidate_id": 1}
        right = {**semantic, "donor_path": "/other/route_99.json", "donor_route_id": 99, "candidate_id": 88}
        self.assertEqual(candidate_semantic_id(left), candidate_semantic_id(right))

    def test_identity_leakage_rejects(self) -> None:
        with self.assertRaisesRegex(ValueError, "replay_id"):
            candidate_semantic_id({"start": 24, "replay_id": "offline-case"})

    def test_noncanonical_numeric_forms_reject(self) -> None:
        with self.assertRaises(TypeError):
            canonical_candidate({"start": 24.0})
        with self.assertRaises(TypeError):
            action_fragment_content_id([{}], 0.0, 1)

    def test_repeated_canonicalization_is_deterministic(self) -> None:
        payload = {"actions": [{"market": [["SELL", "WHEAT", 2]]}], "start": 12}
        first = canonical_candidate(payload)
        self.assertEqual(first, canonical_candidate(payload))
        self.assertEqual(candidate_semantic_id(payload), candidate_semantic_id(payload))

    def test_donor_diversity_analyzer_reads_actions_without_source_identity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = []
            for name, middle in (("route_1.json", "NORTH"), ("route_2.json", "NORTH"), ("route_3.json", "SOUTH")):
                path = root / name
                path.write_text(json.dumps({"actions": [{}, {"farmer": [middle]}, {}]}), encoding="utf-8")
                paths.append(path)
            result = donor_fragment_diversity(paths, [(1, 1)])
            self.assertEqual(result[(1, 1)], 2)


if __name__ == "__main__":
    unittest.main()
