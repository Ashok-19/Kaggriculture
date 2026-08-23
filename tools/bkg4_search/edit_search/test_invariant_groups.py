from __future__ import annotations

import unittest

from tools.bkg4_search.edit_search.panels import CaseRecord
from tools.bkg4_search.invariant_groups import (
    GroupOutcome,
    RobustGroupFitness,
    grouped_cases,
    group_outcomes_from_case_rows,
    production_qualification,
    rotating_tape_folds,
    solve_cross_tape_threshold,
)


def case(case_id: str, tape: str, seat: int, role: str = "discovery") -> CaseRecord:
    return CaseRecord(
        case_id=case_id,
        tape_path=tape,
        seat=seat,
        seed=100 + seat,
        family="family",
        protected=False,
        own_history=False,
        role=role,
        baseline_margin=-10,
    )


class InvariantGroupTests(unittest.TestCase):
    def test_grouping_keeps_both_seats_of_one_tape_together(self):
        rows = [case("a0", "tape-a.json", 0), case("a1", "tape-a.json", 1), case("b0", "tape-b.json", 0)]
        groups = grouped_cases(rows)
        self.assertEqual(set(groups), {"tape-a.json", "tape-b.json"})
        self.assertEqual({x.seat for x in groups["tape-a.json"]}, {0, 1})

    def test_rotating_folds_are_tape_disjoint_and_cover_every_group(self):
        rows = [case(f"{t}{s}", f"tape-{t}.json", s) for t in "abcdef" for s in (0, 1)]
        folds = rotating_tape_folds(rows, n_folds=3, seed=71)
        held = []
        for fold in folds:
            self.assertFalse(set(fold.train_tapes) & set(fold.validation_tapes))
            held.extend(fold.validation_tapes)
        self.assertEqual(set(held), {f"tape-{t}.json" for t in "abcdef"})

    def test_discovery_folds_reject_certification_cases(self):
        rows = [case("a", "tape-a", 0), case("c", "tape-c", 0, role="certification")]
        with self.assertRaisesRegex(ValueError, "certification"):
            rotating_tape_folds(rows, n_folds=2, seed=1, discovery_only=True)

    def test_production_requires_two_useful_independent_groups(self):
        one = [
            GroupOutcome("a", new_loss=False, severe_regression=False, behavior_changed=True, useful=True, l2w=1, margin_delta=50),
            GroupOutcome("b", new_loss=False, severe_regression=False, behavior_changed=False, useful=False, l2w=0, margin_delta=0),
        ]
        q = production_qualification(one, min_useful_groups=2)
        self.assertFalse(q.production)
        two = one + [GroupOutcome("c", False, False, True, True, 0, 20)]
        self.assertTrue(production_qualification(two, min_useful_groups=2).production)

    def test_worst_group_safety_dominates_large_single_tape_gain(self):
        safe = RobustGroupFitness.from_outcomes([
            GroupOutcome("a", False, False, True, True, 0, 10),
            GroupOutcome("b", False, False, True, True, 0, 8),
        ])
        brittle = RobustGroupFitness.from_outcomes([
            GroupOutcome("a", False, False, True, True, 1, 10000),
            GroupOutcome("b", True, True, True, False, 0, -5000),
        ])
        self.assertLess(safe.key(), brittle.key())

    def test_constant_solver_prefers_broad_cross_tape_interval(self):
        # Beneficial values lie high on all three tapes; protected negatives lie low.
        samples = {
            "a": [(80, True, False), (20, False, True)],
            "b": [(90, True, False), (30, False, True)],
            "c": [(100, True, False), (40, False, True)],
        }
        solved = solve_cross_tape_threshold(samples, comparator="GE")
        self.assertEqual(solved.support_groups, 3)
        self.assertEqual(solved.harmed_protected, 0)
        self.assertGreaterEqual(solved.interval_hi - solved.interval_lo, 10)
        self.assertTrue(solved.interval_lo <= solved.threshold <= solved.interval_hi)
        self.assertGreater(solved.boundary_margin, 0)

    def test_constant_solver_runtime_depends_on_boundaries_not_integer_span(self):
        samples = {
            "low": [(-1_000_000_000, False, True)],
            "high": [(1_000_000_000, True, False)],
        }
        solved = solve_cross_tape_threshold(samples, comparator="GE")
        self.assertEqual(solved.harmed_protected, 0)
        self.assertEqual(solved.support_groups, 1)
        self.assertLess(solved.interval_lo, solved.interval_hi)
        self.assertTrue(solved.interval_lo <= solved.threshold <= solved.interval_hi)

    def test_activation_without_action_or_terminal_change_is_not_behavior(self):
        outcomes = group_outcomes_from_case_rows(({
            "case_id": "a", "baseline_margin": 10, "candidate_margin": 10,
            "margin_delta": 0, "action_changed": False, "activation_count": 20,
        },), {"a": "tape-a"})
        self.assertFalse(outcomes[0].behavior_changed)
        self.assertFalse(outcomes[0].useful)


if __name__ == "__main__":
    unittest.main()
