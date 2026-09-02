from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tools.bkg4_search.rlrmb.adversarial_league import (
    ALL_LANES,
    LeagueManager,
    LeagueMember,
    make_bounded_exploiter_candidates,
)
from tools.bkg4_search.rlrmb.p13_equal_budget import _generic_events
from tools.bkg4_search.rlrmb.checkpoints import Lane


class AdversarialLeagueTests(unittest.TestCase):
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
            root / "scratch/p0_1_macro_search/donors/route_64.json",
            root / "scratch/p0_1_macro_search/donors/route_410.json",
        ]
        if not all(path.exists() for path in routes):
            self.skipTest("internal donor routes unavailable")
        programs, motifs = make_bounded_exploiter_candidates(routes)
        self.assertEqual(len(programs), len(ALL_LANES) * 2)
        self.assertEqual({program.lane for program in programs}, set(ALL_LANES))
        self.assertTrue(motifs)

    def test_pipeline_checkpoint_budget_cycles_all_league_lanes(self):
        route = Path.cwd() / "scratch/p0_1_macro_search/donors/route_64.json"
        if not route.exists():
            self.skipTest("internal donor route unavailable")
        events = _generic_events([route], "league", tuple(Lane))
        self.assertEqual({event.lane_hint for event in events}, set(Lane))


if __name__ == "__main__":
    unittest.main()
