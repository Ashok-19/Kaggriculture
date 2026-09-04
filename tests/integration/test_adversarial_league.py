from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from kaggriculture_bonsai.league.adversarial_league import (
    ALL_LANES,
    LeagueManager,
    LeagueMember,
    make_bounded_exploiter_candidates,
    generate_league_build_cases,
)
from kaggriculture_bonsai.search.p13_equal_budget import _generic_events
from kaggriculture_bonsai.search.checkpoints import Lane


class AdversarialLeagueTests(unittest.TestCase):
    def test_build_rejects_single_seed_before_generation(self):
        with tempfile.TemporaryDirectory() as directory, self.assertRaisesRegex(
            ValueError, "at least two independent seeds"
        ):
            generate_league_build_cases(Path(directory), seeds=(17,), exploiters=())
    def test_manager_persists_complete_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "league.json"
            manager = LeagueManager(path)
            manager.add(LeagueMember(
                "exploit_1", "bkg4_current", "program.rlrb", 2,
                "cuda_exploiter", "CROP_TRANSITION",
                mutation_origin="route-window", exploiter_origin="rlrmb_eval",
            ))
            restored = LeagueManager(path)
            self.assertEqual(restored.members, manager.members)
            document = json.loads(path.read_text())
            self.assertEqual(document["members"][0]["parent"], "bkg4_current")

    def test_freeze_previous_champion_keeps_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            manager = LeagueManager(Path(directory) / "league.json")
            manager.freeze_previous_champion(
                policy_id="bkg4_old", checkpoint="champions/old", generation=3,
            )
            self.assertEqual(manager.members[0].member_type, "frozen_checkpoint")

    def test_candidate_budget_is_lane_balanced(self):
        root = Path.cwd()
        routes = [
            root / "assets/baseline/route_64.json",
            root / "assets/baseline/route_410.json",
        ]
        programs, motifs = make_bounded_exploiter_candidates(routes)
        self.assertEqual(len(programs), len(ALL_LANES) * 2)
        self.assertEqual({program.lane for program in programs}, set(ALL_LANES))
        self.assertTrue(motifs)

    def test_pipeline_checkpoint_budget_cycles_all_league_lanes(self):
        route = Path.cwd() / "assets/baseline/route_64.json"
        events = _generic_events([route], "league", tuple(Lane))
        self.assertEqual({event.lane_hint for event in events}, set(Lane))


if __name__ == "__main__":
    unittest.main()
