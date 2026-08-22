from __future__ import annotations

import importlib.util
import unittest

from tools.bkg4_search.edit_search.fitness import PairEvaluation, StateEvaluation


def ev(sid: str, margins: tuple[int, ...], traces: tuple[int, ...] | None = None) -> StateEvaluation:
    n = len(margins)
    return StateEvaluation(
        sid, "mixed", tuple(f"c{i}" for i in range(n)), margins,
        tuple(1 for _ in range(n)), tuple(1 for _ in range(n)),
        traces or tuple(100 + i for i in range(n)),
    )


class SafetySpecificationTests(unittest.TestCase):
    def _api(self):
        module = importlib.util.find_spec("tools.bkg4_search.edit_search.safety")
        self.assertIsNotNone(module, "central safety specification module must exist")
        if module is None:
            return None
        from tools.bkg4_search.edit_search.safety import SafetySpec, SafetyTier
        return SafetySpec, SafetyTier

    def test_hard_feasibility_and_elite_flip_are_distinct_from_merely_safe(self):
        api = self._api()
        if api is None: return
        SafetySpec, SafetyTier = api
        root = ev("root", (-20, 100, 500, 8000))
        elite = ev("elite", (10, 120, 700, 9000), (201, 202, 203, 204))
        weak = ev("weak", (-10, 101, 450, 7990), (301, 302, 303, 304))
        protected = {"c0": False, "c1": True, "c2": True, "c3": True}
        elite_a = SafetySpec().assess(PairEvaluation.from_states(root, elite, protected=protected))
        weak_a = SafetySpec().assess(PairEvaluation.from_states(root, weak, protected=protected))
        self.assertTrue(elite_a.feasible)
        self.assertEqual(elite_a.tier, SafetyTier.ELITE_FLIP)
        self.assertGreaterEqual(elite_a.worst_protected_delta, 0)
        self.assertGreaterEqual(elite_a.protected_tail_mean, 0)
        self.assertTrue(weak_a.feasible)
        self.assertNotIn(weak_a.tier, {SafetyTier.ELITE_FLIP, SafetyTier.ELITE_MARGIN})

    def test_infeasible_repair_key_prioritizes_distance_to_feasibility_before_reward(self):
        api = self._api()
        if api is None: return
        SafetySpec, _ = api
        root = ev("root", (-100, 1000, 1000))
        near = ev("near", (50, -1, 1100), (201, 202, 203))
        destructive = ev("bad", (100000, -9000, 50000), (301, 302, 303))
        protected = {"c0": False, "c1": True, "c2": True}
        near_a = SafetySpec().assess(PairEvaluation.from_states(root, near, protected=protected))
        bad_a = SafetySpec().assess(PairEvaluation.from_states(root, destructive, protected=protected))
        self.assertFalse(near_a.feasible); self.assertFalse(bad_a.feasible)
        self.assertLess(near_a.repair_key(), bad_a.repair_key())
        self.assertLess(near_a.violation.worst_protected_loss, bad_a.violation.worst_protected_loss)

    def test_program_assessment_tracks_source_and_root_feasibility_separately(self):
        api = self._api()
        if api is None: return
        SafetySpec, _ = api
        root = ev("root", (100, -100))
        unsafe_parent = ev("parent", (-10, -80), (201, 202))
        child = ev("child", (-1, 10), (301, 302))
        protected = {"c0": True, "c1": False}
        source_pair = PairEvaluation.from_states(unsafe_parent, child, protected=protected)
        root_pair = PairEvaluation.from_states(root, child, protected=protected)
        assessment = SafetySpec().assess_program(source_pair=source_pair, root_pair=root_pair)
        self.assertTrue(assessment.source.feasible)
        self.assertFalse(assessment.root.feasible)
        self.assertFalse(assessment.production_feasible)


if __name__ == "__main__":
    unittest.main(verbosity=2)
