from __future__ import annotations

import unittest

from tools.bkg4_search.invariant_groups import GroupOutcome
from tools.bkg4_search.edit_search.fitness import StateEvaluation
from tools.bkg4_search.edit_search.panels import CaseRecord
from tools.bkg4_search.invariant_module import AbstractAction, GuardAtom, InvariantRule, InvariantStrategicModule, LiftedGuard, ModuleSketch
from tools.bkg4_search.invariant_search import (
    CEGIS_FAILURES,
    CapabilityLadder,
    InvariantSearchArchive,
    InvariantSearchBackend,
    SearchGoal,
    classify_counterexample,
    default_search_goals,
    large_neighborhood,
    recombine_compatible,
    semantic_descriptor,
)


class UnsafeExactEvaluator:
    def evaluate(self, modules, panel, *, work_dir, panel_id):
        case_ids = tuple(x.case_id for x in panel)
        out = {}
        for index, module in enumerate(modules):
            baseline = index == 0
            out[module.semantic_id()] = StateEvaluation(
                module.semantic_id(), panel_id, case_ids,
                tuple(10 if baseline else -5 for _ in case_ids),
                tuple(0 if baseline else 1 for _ in case_ids),
                tuple(0 if baseline else 1 for _ in case_ids),
                tuple(index + 1 for _ in case_ids),
            )
        return out


def rule(effect: str, action: AbstractAction, *, domain: str = "GLOBAL", feature: str = "OWN_MONEY") -> InvariantRule:
    return InvariantRule(ModuleSketch(effect.lower(), "INSTANT", effect), domain, LiftedGuard((GuardAtom(feature, "GE", 1),)), action)


class InvariantSearchTests(unittest.TestCase):
    def test_default_goals_are_mechanism_oriented_and_identity_free(self):
        goals = default_search_goals((29, 64, 70, 211))
        self.assertGreaterEqual(len(goals), 6)
        text = repr(goals).lower()
        self.assertNotIn("opponent", text)
        self.assertNotIn("tape", text)
        self.assertTrue(any(g.effect_class == "CREATE_FINANCING_RUNWAY" for g in goals))
        self.assertTrue(any(g.effect_class == "RECOVER_RESOURCE_INVARIANT" for g in goals))

    def test_semantic_descriptor_ignores_constants(self):
        a = InvariantStrategicModule((rule("CREATE_FINANCING_RUNWAY", AbstractAction("MARKET_INTENT", market_op=6, item=6), feature="OWN_SHED"),))
        b = InvariantStrategicModule((InvariantRule(a.rules[0].sketch, "GLOBAL", LiftedGuard((GuardAtom("OWN_SHED", "GE", 99, arg=6),)), a.rules[0].action),))
        self.assertEqual(semantic_descriptor(a), semantic_descriptor(b))

    def test_recombination_rejects_conflicting_route_rules(self):
        a = InvariantStrategicModule((rule("ROUTE_MODE", AbstractAction("ROUTE_OVERRIDE", route_id=29)),))
        b = InvariantStrategicModule((rule("ACCELERATE_EXPANSION", AbstractAction("ROUTE_OVERRIDE", route_id=64)),))
        self.assertIsNone(recombine_compatible(a, b))

    def test_large_neighborhood_changes_subprogram_not_identity(self):
        goals = default_search_goals((29, 64, 70))
        parent = InvariantStrategicModule((rule("ROUTE_MODE", AbstractAction("ROUTE_OVERRIDE", route_id=29)),))
        child = large_neighborhood(parent, goals[0], seed=7, route_ids=(29, 64, 70))
        self.assertNotEqual(parent.semantic_id(), child.semantic_id())
        self.assertLessEqual(len(child.rules), 3)

    def test_cegis_failure_classifier_is_closed(self):
        self.assertEqual(classify_counterexample({"new_loss": True, "activation_count": 9}), "guard_too_broad")
        self.assertEqual(classify_counterexample({"action_conflict": True}), "action_conflict")
        self.assertIn(classify_counterexample({"margin_delta": -1, "activation_count": 0}), CEGIS_FAILURES)

    def test_archive_uses_semantic_qd_and_safety_first(self):
        m = InvariantStrategicModule((rule("CREATE_FINANCING_RUNWAY", AbstractAction("MARKET_INTENT", market_op=6, item=6)),))
        archive = InvariantSearchArchive()
        safe = [GroupOutcome("a", False, False, True, True, 0, 5.0), GroupOutcome("b", False, False, True, True, 0, 2.0)]
        unsafe = [GroupOutcome("a", True, False, True, True, 0, 100.0), GroupOutcome("b", False, False, True, True, 0, 100.0)]
        self.assertTrue(archive.consider(m, safe))
        self.assertFalse(archive.consider(m, unsafe))
        self.assertEqual(len(archive.elites), 1)

    def test_capability_ladder_escalates_on_saturation(self):
        ladder = CapabilityLadder(patience=2)
        self.assertFalse(ladder.observe(attempted=10, unique=10, useful=0).escalated)
        event = ladder.observe(attempted=10, unique=2, useful=0)
        self.assertTrue(event.escalated)
        self.assertGreater(event.level, 0)

    def test_useful_but_unqualified_candidate_does_not_hide_saturation(self):
        ladder = CapabilityLadder(patience=1)
        event = ladder.observe(
            attempted=10, unique=10, useful=1, production_qualified=0, archive_insertions=0,
        )
        self.assertTrue(event.escalated)

    def test_grouped_cegis_repairs_feed_the_next_real_proposal_batch(self):
        panel = (
            CaseRecord("a", "tape-a", 0, 1, "x", role="discovery"),
            CaseRecord("b", "tape-b", 0, 2, "x", role="discovery"),
        )
        backend = InvariantSearchBackend(route_ids=(29, 70), evaluator=UnsafeExactEvaluator())
        backend.ladder.level = backend.ladder.LEVELS.index("grouped_cegis")
        original = InvariantStrategicModule((rule(
            "CREATE_FINANCING_RUNWAY",
            AbstractAction("MARKET_INTENT", market_op=6, item=6),
            feature="OWN_SHED",
        ),))
        backend.evaluate_batch((original,), panel, work_dir="unused", panel_id="discovery")
        proposed = backend.propose(n=1, seed=7)
        self.assertEqual(len(proposed), 1)
        self.assertNotEqual(proposed[0].semantic_id(), original.semantic_id())
        self.assertIn("OWN_MONEY", {x.feature for x in proposed[0].rules[0].guard.atoms})

    def test_typed_proposer_fills_bounded_budget_across_uneven_goal_families(self):
        backend = InvariantSearchBackend(route_ids=(29, 64, 70, 211), evaluator=UnsafeExactEvaluator())
        proposed = backend.propose(n=64, seed=19)
        self.assertEqual(len(proposed), 64)
        self.assertGreaterEqual(len({x.rules[0].sketch.effect_class for x in proposed}), 6)

    def test_plant_rescue_does_not_replace_profitable_harvests(self):
        backend = InvariantSearchBackend(route_ids=(29, 64, 70, 211), evaluator=UnsafeExactEvaluator())
        proposed = backend.propose(n=96, seed=23)
        plant_rescues = [
            module for module in proposed
            if any(atom.feature == "OWN_TILE_CONSECUTIVE_UNWATERED" for atom in module.rules[0].guard.atoms)
        ]
        self.assertTrue(plant_rescues)
        self.assertNotIn(10, {module.rules[0].action.match_op for module in plant_rescues})


if __name__ == "__main__":
    unittest.main()
