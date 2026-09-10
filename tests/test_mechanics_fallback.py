"""Overlay fidelity, candidate freezing and mechanics-boundary contracts."""
import unittest

from kaggriculture_bonsai.search import mechanics_fallback as M


class Prices(dict):
    def get(self, key, default=0):  # market prices are name-keyed strings
        return dict.get(self, key, default)


class View:
    """Minimal FarmView stand-in for the two bounded mechanics."""

    def __init__(self, shed, inventories, positions, prices):
        self.tiles = [[{} for _ in range(10)] for _ in range(10)]
        self.positions = positions
        self.inventories = inventories
        self.shed = shed
        self.prices = Prices(prices)

    def inventory(self, worker):
        return self.inventories[worker] if worker < len(self.inventories) else {}

    def beside_shed(self, position):
        return position[0] in (4, 5) and position[1] in (4, 5)


def tape(entry=None):
    blank = {"farmer": ["PASS"], "hands": [], "market": []}
    return [dict(entry or blank) for _ in range(719)]


class CandidateSet(unittest.TestCase):
    def test_exactly_thirty_two_frozen_candidates_split_sixteen_sixteen(self):
        items = M.candidate_set()
        self.assertEqual(len(items), 32)
        self.assertEqual(sum(c["family"] == "capacity" for c in items), 16)
        self.assertEqual(sum(c["family"] == "feed" for c in items), 16)

    def test_candidates_are_unique_and_never_the_reference(self):
        overlays = [tuple(c["overlay"]) for c in M.candidate_set()]
        self.assertEqual(len(set(overlays)), len(overlays))
        self.assertNotIn(M.REFERENCE, overlays)

    def test_candidate_set_is_deterministic(self):
        self.assertEqual(M.candidate_set(), M.candidate_set())


class OverlayContracts(unittest.TestCase):
    def test_zero_edit_overlay_returns_the_action_unchanged(self):
        agent = M.OverlayPlus(None)
        view = View({"MILK": 90}, [{"MILK": 30}], [(4, 4)], {"MILK": 5})
        action = {"farmer": ["DROP"], "hands": [], "market": []}
        self.assertEqual(agent._apply(dict(action), view, tape(), 300), action)

    def test_partial_deposit_keeps_the_cargo_a_full_shed_would_destroy(self):
        agent = M.OverlayPlus((1, 1, 0, 0, 0, 0, 0))
        # 5 free slots, 30 carried: DROP would deposit 5 and destroy 25.
        view = View({"WHEAT": 95}, [{"CARROT": 20, "MILK": 10}], [(4, 4)],
                    {"CARROT": 3, "MILK": 9, "WHEAT": 1})
        out = agent._apply({"farmer": ["DROP"], "hands": [], "market": []}, view, tape(), 300)
        self.assertEqual(out["farmer"], ["PLACE", "MILK", 5])  # highest value first

    def test_partial_deposit_passes_rather_than_discard_when_the_shed_is_full(self):
        agent = M.OverlayPlus((1, 1, 0, 0, 0, 0, 0))
        view = View({"WHEAT": 100}, [{"CARROT": 4}], [(4, 4)], {"CARROT": 3, "WHEAT": 1})
        out = agent._apply({"farmer": ["DROP"], "hands": [], "market": []}, view, tape(), 300)
        self.assertEqual(out["farmer"], ["PASS"])

    def test_deposit_rule_ignores_losses_below_the_activation_threshold(self):
        agent = M.OverlayPlus((1, 3, 0, 0, 0, 0, 0))
        view = View({"WHEAT": 99}, [{"CARROT": 2}], [(4, 4)], {"CARROT": 3, "WHEAT": 1})
        out = agent._apply({"farmer": ["DROP"], "hands": [], "market": []}, view, tape(), 300)
        self.assertEqual(out["farmer"], ["DROP"])

    def test_preemptive_sale_only_fires_when_a_deposit_is_actually_scheduled(self):
        agent = M.OverlayPlus((0, 0, 20, 4, 0, 0, 0))
        view = View({"WHEAT": 90, "CARROT": 8}, [{}], [(0, 0)], {"CARROT": 3, "WHEAT": 1})
        quiet = agent._apply({"farmer": ["PASS"], "hands": [], "market": []}, view, tape(), 300)
        self.assertEqual(quiet["market"], [])
        soon = tape({"farmer": ["DROP"], "hands": [], "market": []})
        loud = agent._apply({"farmer": ["PASS"], "hands": [], "market": []}, view, soon, 300)
        self.assertEqual(loud["market"], [["SELL", "CARROT", 8]])  # never wheat

    def test_feed_clamp_retains_forecast_demand_plus_the_reserve(self):
        agent = M.OverlayPlus((0, 0, 0, 0, 1, 2, 24))
        view = View({"WHEAT": 20}, [{}], [(4, 4)], {"WHEAT": 1})
        plan = tape()
        for step in range(300, 306):
            plan[step] = {"farmer": ["FEED"], "hands": [], "market": []}
        out = agent._apply({"farmer": ["PASS"], "hands": [],
                            "market": [["SELL", "WHEAT", 20]]}, view, plan, 300)
        self.assertEqual(out["market"], [["SELL", "WHEAT", 12]])  # 20 - (6 + 2)

    def test_feed_topup_picks_up_wheat_only_for_a_lane_that_will_feed(self):
        agent = M.OverlayPlus((0, 0, 0, 0, 3, 2, 24))
        view = View({"WHEAT": 9}, [{}, {}], [(4, 4), (4, 5)], {"WHEAT": 1})
        plan = tape()
        plan[305] = {"farmer": ["PASS"], "hands": [["FEED"]], "market": []}
        out = agent._apply({"farmer": ["PASS"], "hands": [["PASS"]], "market": []}, view, plan, 300)
        self.assertEqual(out["farmer"], ["PASS"])
        self.assertEqual(out["hands"], [["PICKUP", "WHEAT", 1]])

    def test_terminal_turn_is_never_overlaid(self):
        agent = M.OverlayPlus((1, 1, 20, 4, 1, 2, 24))
        view = View({"WHEAT": 100}, [{"CARROT": 9}], [(4, 4)], {"CARROT": 3, "WHEAT": 1})
        action = {"farmer": ["DROP"], "hands": [], "market": []}
        self.assertEqual(agent._apply(dict(action), view, tape(), 718), action)


if __name__ == "__main__":
    unittest.main()
