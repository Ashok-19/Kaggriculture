from __future__ import annotations

import unittest
from dataclasses import replace

from tools.bkg4_search.invariant_groups import GroupOutcome
from tools.bkg4_search.edit_search.fitness import StateEvaluation
from tools.bkg4_search.edit_search.panels import CaseRecord
from tools.bkg4_search.information_state_io import registry_manifest
from tools.bkg4_search.invariant_module import AbstractAction, GuardAtom, InvariantRule, InvariantStrategicModule, LiftedGuard, ModuleSketch
from tools.bkg4_search.module_codec import ConditionGene, ModuleGenome, RuleGene
from tools.bkg4_search.invariant_search import (
    CEGIS_FAILURES,
    CapabilityLadder,
    InvariantSearchArchive,
    InvariantSearchBackend,
    SearchGoal,
    classify_counterexample,
    cegis_repair,
    derive_threshold_evidence,
    emit_skeletons,
    guard_evidence_key,
    grouped_counterexample_rows,
    internal_crossfit_split,
    default_search_goals,
    large_neighborhood,
    recombine_compatible,
    semantic_descriptor,
    structural_skeleton,
    translate_route_genome,
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


class MixedThresholdEvaluator:
    def evaluate(self, modules, panel, *, work_dir, panel_id):
        case_ids = tuple(x.case_id for x in panel)
        out = {}
        for index, module in enumerate(modules):
            baseline = index == 0
            margins = tuple(
                (0 if case_id == "a" else 100) if baseline else (10 if case_id == "a" else 90)
                for case_id in case_ids
            )
            out[module.semantic_id()] = StateEvaluation(
                module.semantic_id(), panel_id, case_ids, margins,
                tuple(0 if baseline else 1 for _ in case_ids),
                tuple(0 if baseline else 1 for _ in case_ids),
                tuple(index + 1 for _ in case_ids),
            )
        return out


class CrossFitEvaluator:
    def __init__(self, *, unsafe_selection: bool = False):
        self.unsafe_selection = bool(unsafe_selection)

    def evaluate(self, modules, panel, *, work_dir, panel_id):
        case_ids = tuple(x.case_id for x in panel)
        selection = "/selection" in panel_id
        out = {}
        for index, module in enumerate(modules):
            baseline = index == 0
            margins = tuple(
                0 if baseline else (-10 if selection and self.unsafe_selection else 10)
                for _ in case_ids
            )
            out[module.semantic_id()] = StateEvaluation(
                module.semantic_id(), panel_id, case_ids, margins,
                tuple(0 if baseline else 1 for _ in case_ids),
                tuple(0 if baseline else 1 for _ in case_ids),
                tuple(index + 1 for _ in case_ids),
            )
        return out


def context_case(case_id: str, tape: str, *, own_money: int, protected: bool = True) -> CaseRecord:
    registry = registry_manifest()
    values = [0] * len(registry)
    values[registry.index(("OWN_MONEY", ()))] = int(own_money)
    return CaseRecord(
        case_id, tape, 0, 1, "x", protected=protected, role="discovery",
        context_probe_steps=(48,), public_context_width=len(registry),
        public_context_registry_width=len(registry), public_context_values=tuple(values),
    )


def rule(effect: str, action: AbstractAction, *, domain: str = "GLOBAL", feature: str = "OWN_MONEY") -> InvariantRule:
    effect_domains = {
        "CREATE_FINANCING_RUNWAY": "market",
        "ROUTE_MODE": "route",
        "ACCELERATE_EXPANSION": "expansion",
    }
    return InvariantRule(ModuleSketch(effect_domains[effect], "INSTANT", effect), domain, LiftedGuard((GuardAtom(feature, "GE", 1),)), action)


class InvariantSearchTests(unittest.TestCase):
    def test_internal_crossfit_split_is_tape_disjoint_and_deterministic(self):
        panel = tuple(
            CaseRecord(f"{t}-{seat}", f"tape-{t}", seat, t * 10 + seat, "x", role="discovery")
            for t in range(8) for seat in (0, 1)
        )
        first = internal_crossfit_split(panel, selection_fraction=0.25, seed=91)
        second = internal_crossfit_split(tuple(reversed(panel)), selection_fraction=0.25, seed=91)
        self.assertEqual({x.case_id for x in first.search}, {x.case_id for x in second.search})
        self.assertEqual({x.case_id for x in first.selection}, {x.case_id for x in second.selection})
        self.assertFalse(set(first.search_tapes) & set(first.selection_tapes))
        self.assertEqual(len(first.selection_tapes), 2)

    def test_crossfit_freezes_top_k_and_selection_cannot_feed_search(self):
        panel = tuple(
            CaseRecord(f"c{t}", f"tape-{t}", 0, t, "x", role="discovery")
            for t in range(8)
        )
        good = InvariantSearchBackend(route_ids=(29, 70), evaluator=CrossFitEvaluator())
        bad = InvariantSearchBackend(route_ids=(29, 70), evaluator=CrossFitEvaluator(unsafe_selection=True))
        good_result = good.search_crossfit(
            panel, rounds=1, candidates_per_round=16, seed=73,
            work_dir="unused-good", top_k=4, selection_fraction=0.25,
        )
        bad_result = bad.search_crossfit(
            panel, rounds=1, candidates_per_round=16, seed=73,
            work_dir="unused-bad", top_k=4, selection_fraction=0.25,
        )
        self.assertEqual(good_result.frozen_semantic_ids, bad_result.frozen_semantic_ids)
        self.assertEqual(len(good_result.frozen_semantic_ids), 4)
        self.assertTrue(good_result.transferable_elites)
        self.assertFalse(bad_result.transferable_elites)
        self.assertEqual(good.threshold_evidence(), bad.threshold_evidence())
        self.assertEqual(
            [x.semantic_id() for x in good.propose(n=16, seed=999)],
            [x.semantic_id() for x in bad.propose(n=16, seed=999)],
        )

    def test_crossfit_rejects_backend_with_preexisting_search_evidence(self):
        panel = tuple(
            CaseRecord(f"c{t}", f"tape-{t}", 0, t, "x", role="discovery")
            for t in range(8)
        )
        backend = InvariantSearchBackend(route_ids=(29, 70), evaluator=CrossFitEvaluator())
        backend.evaluate_batch(
            backend.propose(n=4, seed=1), panel[:4],
            work_dir="unused-preload", panel_id="discovery/preload",
        )
        with self.assertRaisesRegex(ValueError, "pristine"):
            backend.search_crossfit(
                panel, rounds=1, candidates_per_round=4, seed=73,
                work_dir="unused", top_k=2, selection_fraction=0.25,
            )
    def test_default_goals_are_mechanism_oriented_and_identity_free(self):
        goals = default_search_goals((29, 64, 70, 211))
        self.assertGreaterEqual(len(goals), 6)
        text = repr(goals).lower()
        self.assertNotIn("opponent", text)
        self.assertNotIn("tape", text)
        self.assertTrue(any(g.effect_class == "CREATE_FINANCING_RUNWAY" for g in goals))
        self.assertTrue(any(g.effect_class == "RECOVER_RESOURCE_INVARIANT" for g in goals))
        self.assertTrue(any(g.goal_id == "adaptive_route_regime" for g in goals))
        self.assertTrue(any(g.goal_id == "price_floor_financing_mechanic" and g.effect_class == "ENGINE_MECHANIC" for g in goals))

    def test_semantic_descriptor_ignores_constants(self):
        a = InvariantStrategicModule((rule("CREATE_FINANCING_RUNWAY", AbstractAction("MARKET_INTENT", market_op=6, item=6), feature="OWN_SHED"),))
        b = InvariantStrategicModule((InvariantRule(a.rules[0].sketch, "GLOBAL", LiftedGuard((GuardAtom("OWN_SHED", "GE", 99, arg=6),)), a.rules[0].action),))
        self.assertEqual(semantic_descriptor(a), semantic_descriptor(b))

    def test_structural_skeleton_ignores_thresholds_but_preserves_mechanism(self):
        base = InvariantStrategicModule((rule(
            "CREATE_FINANCING_RUNWAY", AbstractAction("MARKET_INTENT", market_op=6, item=6),
            feature="OWN_SHED",
        ),))
        shifted = InvariantStrategicModule((replace(
            base.rules[0], guard=LiftedGuard((GuardAtom("OWN_SHED", "GE", 99),)),
        ),))
        different_action = InvariantStrategicModule((replace(
            base.rules[0], action=AbstractAction("MARKET_INTENT", market_op=6, item=7),
        ),))
        self.assertEqual(structural_skeleton(base), structural_skeleton(shifted))
        self.assertNotEqual(structural_skeleton(base), structural_skeleton(different_action))

    def test_portable_route_genome_translation_preserves_first_match_priority(self):
        genome = ModuleGenome(prefix_step=120, rules=(
            RuleGene((ConditionGene("OWN_MONEY", "GE", value=1000),), 29),
            RuleGene((ConditionGene("OWN_LAND", "LE", value=2),), 70),
        ))
        translated = translate_route_genome(genome, route_ids=(29, 70))
        self.assertEqual([x.action.route_id for x in translated.rules], [70, 29])
        self.assertTrue(all(any(a.feature == "STEP" and a.cmp == "GE" and a.value == 120 for a in x.guard.atoms) for x in translated.rules))
        translated.validate()

    def test_portable_route_translation_rejects_concrete_identity_guards(self):
        genome = ModuleGenome(prefix_step=120, rules=(
            RuleGene((ConditionGene("OWN_WORKER_X", "GE", index=3, value=1),), 29),
        ))
        with self.assertRaisesRegex(ValueError, "identity"):
            translate_route_genome(genome, route_ids=(29, 70))

    def test_recombination_rejects_conflicting_route_rules(self):
        a = InvariantStrategicModule((rule("ROUTE_MODE", AbstractAction("ROUTE_OVERRIDE", route_id=29)),))
        b = InvariantStrategicModule((rule("ACCELERATE_EXPANSION", AbstractAction("ROUTE_OVERRIDE", route_id=64)),))
        self.assertIsNone(recombine_compatible(a, b))

    def test_recombination_accepts_authority_priority_compatible_market_pair(self):
        a = InvariantStrategicModule((rule(
            "CREATE_FINANCING_RUNWAY", AbstractAction("MARKET_INTENT", market_op=6, item=4, quantity=1, authority=5, priority=220),
        ),))
        b = InvariantStrategicModule((rule(
            "CREATE_FINANCING_RUNWAY", AbstractAction("MARKET_INTENT", market_op=6, item=5, quantity=1, authority=5, priority=200),
        ),))
        child = recombine_compatible(a, b)
        self.assertIsNotNone(child)
        self.assertEqual(len(child.rules), 2)

    def test_recombination_rejects_overlapping_conflicting_unit_rewrites(self):
        def rewrite(unit_op):
            return InvariantStrategicModule((InvariantRule(
                ModuleSketch("recovery", "INSTANT", "RECOVER_RESOURCE_INVARIANT"), "EACH_OWN_UNIT",
                LiftedGuard((GuardAtom("OWN_TILE_HAS_ANIMAL", "EQ", 1),)),
                AbstractAction("UNIT_REWRITE", match_op=0, unit_op=unit_op),
            ),))
        self.assertIsNone(recombine_compatible(rewrite(15), rewrite(17)))

    def test_recombination_accepts_overlapping_nonconflicting_domains(self):
        route_module = InvariantStrategicModule((rule(
            "ROUTE_MODE", AbstractAction("ROUTE_OVERRIDE", route_id=29), feature="OWN_MONEY",
        ),))
        market_module = InvariantStrategicModule((rule(
            "CREATE_FINANCING_RUNWAY", AbstractAction("MARKET_INTENT", market_op=6, item=5, quantity=1, authority=5, priority=200),
            feature="OWN_MONEY",
        ),))
        child = recombine_compatible(route_module, market_module)
        self.assertIsNotNone(child)
        self.assertEqual(len(child.rules), 2)

    def test_recombination_rejects_rule_count_overflow(self):
        first = InvariantStrategicModule((
            rule("ROUTE_MODE", AbstractAction("ROUTE_OVERRIDE", route_id=29)),
            rule("CREATE_FINANCING_RUNWAY", AbstractAction("MARKET_INTENT", market_op=6, item=4, quantity=1, authority=5, priority=220)),
        ))
        second = InvariantStrategicModule((
            rule("ACCELERATE_EXPANSION", AbstractAction("MARKET_INTENT", market_op=1, quantity=1, authority=3, priority=180)),
            InvariantRule(ModuleSketch("recovery", "INSTANT", "RECOVER_RESOURCE_INVARIANT"), "EACH_OWN_UNIT",
                LiftedGuard((GuardAtom("OWN_TILE_CONSECUTIVE_UNFED", "GE", 1),)),
                AbstractAction("UNIT_REWRITE", match_op=0, unit_op=15)),
        ))
        self.assertIsNone(recombine_compatible(first, second))

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

    def test_grouped_counterexamples_weight_each_tape_once(self):
        rows = (
            {"case_id": "a0", "margin_delta": -9, "new_loss": True, "activation_count": 1},
            {"case_id": "a1", "margin_delta": -2, "new_loss": False, "activation_count": 1},
            {"case_id": "b0", "margin_delta": -3, "new_loss": False, "activation_count": 1},
        )
        grouped = grouped_counterexample_rows(rows, {"a0": "tape-a", "a1": "tape-a", "b0": "tape-b"})
        self.assertEqual([x["case_id"] for x in grouped], ["a0", "b0"])

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

    def test_public_boundary_cegis_repair_suppresses_weaker_fallback_repairs(self):
        class BoundaryEvaluator:
            def evaluate(self, modules, panel, *, work_dir, panel_id):
                case_ids = tuple(x.case_id for x in panel)
                out = {}
                for index, module in enumerate(modules):
                    baseline = index == 0
                    margins = (100, 100, 100) if baseline else (110, 120, -10)
                    out[module.semantic_id()] = StateEvaluation(
                        module.semantic_id(), panel_id, case_ids, margins,
                        tuple(0 if baseline else 1 for _ in case_ids),
                        tuple(0 if baseline else 1 for _ in case_ids),
                        tuple(index + 1 for _ in case_ids),
                    )
                return out

        panel = (
            context_case("a", "tape-a", own_money=100),
            context_case("b", "tape-b", own_money=90),
            context_case("c", "tape-c", own_money=10),
        )
        backend = InvariantSearchBackend(route_ids=(29, 70), evaluator=BoundaryEvaluator())
        backend.ladder.level = backend.ladder.LEVELS.index("grouped_cegis")
        original = InvariantStrategicModule((rule(
            "CREATE_FINANCING_RUNWAY",
            AbstractAction("MARKET_INTENT", market_op=6, item=6),
            feature="OWN_SHED",
        ),))
        backend.evaluate_batch((original,), panel, work_dir="unused", panel_id="boundary")
        repairs = backend.consume_cegis_repairs()
        self.assertEqual(len(repairs), 1)
        parent_sid, repaired = repairs[0]
        self.assertEqual(parent_sid, original.semantic_id())
        self.assertIn("OWN_MONEY", {atom.feature for atom in repaired.rules[0].guard.atoms})

    def test_grouped_cegis_repair_is_idempotent_when_default_guard_already_exists(self):
        panel = (
            CaseRecord("a", "tape-a", 0, 1, "x", role="discovery"),
            CaseRecord("b", "tape-b", 0, 2, "x", role="discovery"),
        )
        backend = InvariantSearchBackend(route_ids=(29, 70), evaluator=UnsafeExactEvaluator())
        backend.ladder.level = backend.ladder.LEVELS.index("grouped_cegis")
        original = InvariantStrategicModule((rule(
            "CREATE_FINANCING_RUNWAY",
            AbstractAction("MARKET_INTENT", market_op=6, item=6),
            feature="OWN_MONEY",
        ),))
        backend.evaluate_batch((original,), panel, work_dir="unused", panel_id="discovery")
        proposed = backend.propose(n=1, seed=11)
        self.assertEqual(len(proposed), 1)
        self.assertNotEqual(proposed[0].semantic_id(), original.semantic_id())
        atoms = proposed[0].rules[0].guard.atoms
        self.assertEqual(len(atoms), len(set(atoms)))
        proposed[0].validate()

    def test_cegis_goal_resolution_uses_structure_not_shared_effect_alone(self):
        backend = InvariantSearchBackend(route_ids=(29, 70), evaluator=UnsafeExactEvaluator())
        plant = InvariantStrategicModule((InvariantRule(
            ModuleSketch("recovery", "INSTANT", "RECOVER_RESOURCE_INVARIANT"),
            "EACH_OWN_UNIT",
            LiftedGuard((GuardAtom("OWN_TILE_CONSECUTIVE_UNWATERED", "GE", 1),)),
            AbstractAction("UNIT_REWRITE", match_op=11, unit_op=9),
        ),))
        self.assertEqual(backend._goal_for_module(plant).goal_id, "unwatered_plant_rescue")

    def test_cegis_guard_repair_respects_total_guard_capacity(self):
        atoms = tuple(GuardAtom(feature, "GE", 1) for feature in (
            "OWN_MONEY", "OWN_UNITS", "OWN_LAND", "OWN_SHED_TOTAL", "OWN_HIRES_TODAY",
        ))
        module = InvariantStrategicModule((InvariantRule(
            ModuleSketch("route", "INSTANT", "ROUTE_MODE"), "GLOBAL",
            LiftedGuard(atoms), AbstractAction("ROUTE_OVERRIDE", route_id=70),
        ),))
        repaired = cegis_repair(module, "guard_too_broad", route_ids=(29, 70))
        self.assertEqual(repaired.semantic_id(), module.semantic_id())

    def test_public_group_outcomes_drive_feature_specific_threshold_repair(self):
        panel = (
            context_case("a", "tape-a", own_money=100),
            context_case("b", "tape-b", own_money=20),
        )
        module = InvariantStrategicModule((InvariantRule(
            ModuleSketch("route", "INSTANT", "ROUTE_MODE"), "GLOBAL",
            LiftedGuard((GuardAtom("OWN_MONEY", "GE", 50),)),
            AbstractAction("ROUTE_OVERRIDE", route_id=70),
        ),))
        rows = (
            {"case_id": "a", "baseline_margin": 0, "candidate_margin": 10, "margin_delta": 10, "action_changed": True},
            {"case_id": "b", "baseline_margin": 100, "candidate_margin": 90, "margin_delta": -10, "action_changed": True},
        )
        evidence = derive_threshold_evidence(module, rows, panel)
        key = guard_evidence_key(module.rules[0].guard.atoms[0])
        self.assertEqual(set(evidence[key]), {"tape-a", "tape-b"})
        repaired = cegis_repair(
            module, "unstable_constant", route_ids=(29, 70), threshold_samples=evidence,
        )
        threshold = repaired.rules[0].guard.atoms[0].value
        self.assertGreater(threshold, 20)
        self.assertLessEqual(threshold, 100)

    def test_backend_automatically_uses_exact_public_threshold_evidence(self):
        panel = (
            context_case("a", "tape-a", own_money=100),
            context_case("b", "tape-b", own_money=20),
        )
        module = InvariantStrategicModule((InvariantRule(
            ModuleSketch("route", "INSTANT", "ROUTE_MODE"), "GLOBAL",
            LiftedGuard((GuardAtom("OWN_MONEY", "GE", 50),)),
            AbstractAction("ROUTE_OVERRIDE", route_id=70),
        ),))
        backend = InvariantSearchBackend(route_ids=(29, 70), evaluator=MixedThresholdEvaluator())
        backend.evaluate_batch((module,), panel, work_dir="unused", panel_id="threshold/train")
        self.assertIn("OWN_MONEY:0", backend.threshold_evidence()["adaptive_route_regime"])
        backend.ladder.level = backend.ladder.LEVELS.index("boundary_solver")
        proposed = backend.propose(n=32, seed=31)
        solved = [
            atom.value for candidate in proposed for rule in candidate.rules for atom in rule.guard.atoms
            if atom.feature == "OWN_MONEY" and atom.value not in {500, 1000, 2000}
        ]
        self.assertTrue(solved)

    def test_search_backend_rejects_non_discovery_evidence(self):
        panel = (
            CaseRecord("cert", "tape-cert", 0, 1, "x", role="certification"),
        )
        module = InvariantStrategicModule((rule(
            "CREATE_FINANCING_RUNWAY",
            AbstractAction("MARKET_INTENT", market_op=6, item=6),
            feature="OWN_SHED",
        ),))
        backend = InvariantSearchBackend(route_ids=(29, 70), evaluator=UnsafeExactEvaluator())
        with self.assertRaisesRegex(ValueError, "discovery"):
            backend.evaluate_batch((module,), panel, work_dir="unused", panel_id="certification")

    def test_typed_proposer_fills_bounded_budget_across_uneven_goal_families(self):
        backend = InvariantSearchBackend(route_ids=(29, 64, 70, 211), evaluator=UnsafeExactEvaluator())
        proposed = backend.propose(n=64, seed=19)
        self.assertEqual(len(proposed), 64)
        self.assertGreaterEqual(len({x.rules[0].sketch.effect_class for x in proposed}), 6)

    def test_systematic_proposer_materially_broadens_structural_mechanisms(self):
        backend = InvariantSearchBackend(route_ids=(29, 64, 70, 211), evaluator=UnsafeExactEvaluator())
        backend.ladder.level = backend.ladder.LEVELS.index("systematic_structural")
        proposed = backend.propose(n=128, seed=41)
        self.assertGreaterEqual(len({structural_skeleton(x) for x in proposed}), 20)
        self.assertTrue(any(len(x.rules) == 2 for x in proposed))
        rewrites = {
            (x.rules[0].action.match_op, x.rules[0].action.unit_op)
            for x in proposed if x.rules and x.rules[0].action.kind == "UNIT_REWRITE"
        }
        self.assertIn((0, 15), rewrites)  # PASS -> FEED rescue
        self.assertIn((0, 9), rewrites)   # PASS -> WATER rescue

    def test_plant_rescue_does_not_replace_profitable_harvests(self):
        backend = InvariantSearchBackend(route_ids=(29, 64, 70, 211), evaluator=UnsafeExactEvaluator())
        proposed = backend.propose(n=96, seed=23)
        plant_rescues = [
            module for module in proposed
            if any(atom.feature == "OWN_TILE_CONSECUTIVE_UNWATERED" for atom in module.rules[0].guard.atoms)
        ]
        self.assertTrue(plant_rescues)
        self.assertNotIn(10, {module.rules[0].action.match_op for module in plant_rescues})

    def test_adaptive_route_regimes_cover_economy_and_lifecycle_without_identity(self):
        goal = next(x for x in default_search_goals((29, 64, 70, 211)) if x.goal_id == "adaptive_route_regime")
        candidates = emit_skeletons(goal, n=512, seed=83, route_ids=(29, 64, 70, 211))
        feature_sets = [{atom.feature for atom in m.rules[0].guard.atoms} for m in candidates]
        self.assertTrue(any({"OWN_MONEY", "OWN_UNITS"} <= features for features in feature_sets))
        self.assertTrue(any({"OWN_MONEY", "OWN_LAND"} <= features for features in feature_sets))
        self.assertTrue(any(features == {"DAY"} and len(m.rules[0].guard.atoms) == 2 for m, features in zip(candidates, feature_sets)))
        self.assertNotIn("OPP_", repr(candidates))
        self.assertNotIn("STEP", repr(candidates))

    def test_low_cash_day1_route_anchors_are_always_in_bounded_proposals(self):
        routes = (29, 64, 70, 211)
        backend = InvariantSearchBackend(route_ids=routes, evaluator=UnsafeExactEvaluator())
        proposed = backend.propose(n=32, seed=101)
        anchored = {
            m.rules[0].action.route_id for m in proposed
            if len(m.rules) == 1
            and {(a.feature, a.cmp, a.value) for a in m.rules[0].guard.atoms}
                == {("DAY", "EQ", 1), ("OWN_MONEY", "GE", 20)}
        }
        self.assertEqual(anchored, set(routes))

    def test_portable_wool_and_day1_route_anchor_is_prioritized_when_routes_exist(self):
        backend = InvariantSearchBackend(
            route_ids=(29, 64, 70, 211, 212), evaluator=UnsafeExactEvaluator(),
        )
        proposed = backend.propose(n=32, seed=101)
        composites = [module for module in proposed if len(module.rules) == 2]
        self.assertTrue(composites)
        best = composites[0]
        self.assertEqual(best.rules[0].action.route_id, 70)
        self.assertEqual(
            {(atom.feature, atom.cmp, atom.value, atom.arg) for atom in best.rules[0].guard.atoms},
            {("MARKET_PRICE", "LE", 192, 7)},
        )
        self.assertIn(best.rules[1].action.route_id, {211, 212})
        self.assertEqual(
            {(atom.feature, atom.cmp, atom.value) for atom in best.rules[1].guard.atoms},
            {("DAY", "EQ", 1), ("OWN_MONEY", "GE", 20)},
        )

    def test_expansion_hire_has_self_disabling_daily_cadence(self):
        goal = next(x for x in default_search_goals((29, 64, 70, 211)) if x.goal_id == "accelerate_expansion")
        candidates = emit_skeletons(goal, n=512, seed=97, route_ids=(29, 64, 70, 211))
        daily = [m for m in candidates if any(a.feature == "OWN_HIRES_TODAY" for a in m.rules[0].guard.atoms)]
        self.assertTrue(daily)
        self.assertTrue(all(m.rules[0].action.market_op == 1 for m in daily))
        self.assertTrue(all(any(a.feature == "DAY" for a in m.rules[0].guard.atoms) for m in daily))


if __name__ == "__main__":
    unittest.main()
