from __future__ import annotations

import unittest

from tools.bkg4_search.edit_search.fitness import StateEvaluation
from tools.bkg4_search.edit_search.panels import CaseRecord
from tools.bkg4_search.invariant_groups import GroupOutcome
from tools.bkg4_search.invariant_module import (
    AbstractAction,
    GuardAtom,
    InvariantRule,
    InvariantStrategicModule,
    LiftedGuard,
    ModuleSketch,
)
from tools.bkg4_search.invariant_specialists import (
    SPECIALIST_TIERS,
    SpecialistArchive,
    SpecialistLineage,
    analyze_specialist,
    calibrate_specialist_thresholds,
    composition_metadata,
    diversity_report,
    select_screen_quota,
    specialist_boundary_guards,
    strategic_coverage_matrix,
)
from tools.bkg4_search.information_state_io import registry_manifest
from tools.bkg4_search.invariant_search import (
    CapabilityLadder,
    InvariantSearchBackend,
    default_search_goals,
    local_structural_neighborhood,
)


def _route(feature: str, value: int, route: int = 70) -> InvariantStrategicModule:
    return InvariantStrategicModule((InvariantRule(
        ModuleSketch("route", "INSTANT", "ROUTE_MODE"),
        "GLOBAL",
        LiftedGuard((GuardAtom(feature, "GE", value),)),
        AbstractAction("ROUTE_OVERRIDE", route_id=route),
    ),))


def _market(quantity: int = 1) -> InvariantStrategicModule:
    return InvariantStrategicModule((InvariantRule(
        ModuleSketch("market", "INSTANT", "CREATE_FINANCING_RUNWAY"), "GLOBAL",
        LiftedGuard((GuardAtom("OWN_SHED", "GE", quantity, arg=4),)),
        AbstractAction("MARKET_INTENT", market_op=6, item=4, quantity=quantity, authority=5, priority=220),
    ),))


class InvariantSpecialistTests(unittest.TestCase):
    def test_tier_thresholds_are_calibrated_from_repeat_positive_cohorts(self):
        values = [100.0, 200.0, 400.0, 800.0, 1600.0]
        thresholds = calibrate_specialist_thresholds(values)
        self.assertEqual(thresholds.strong_mean_delta, 200.0)
        self.assertEqual(thresholds.elite_mean_delta, 800.0)
        self.assertGreater(thresholds.elite_mean_delta, thresholds.strong_mean_delta)

    def test_tier_threshold_calibration_has_conservative_sparse_fallback(self):
        thresholds = calibrate_specialist_thresholds([50.0])
        self.assertEqual(thresholds.strong_mean_delta, 250.0)
        self.assertEqual(thresholds.elite_mean_delta, 1000.0)

    def test_mixed_risk_specialist_is_retained_as_elite_for_coherent_cohort(self):
        outcomes = (
            GroupOutcome("a0", False, False, True, True, 1, 4200),
            GroupOutcome("a1", False, False, True, True, 0, 2800),
            GroupOutcome("b0", True, True, True, False, 0, -6000),
        )
        evidence = analyze_specialist(
            _route("OWN_MONEY", 500), outcomes,
            cohort_by_group={"a0": "low_cash", "a1": "low_cash", "b0": "late_market"},
            activation_count=30, action_change_count=12,
        )
        self.assertEqual(evidence.tier, "ELITE_SPECIALIST")
        self.assertEqual(evidence.target_cohorts, ("low_cash",))
        self.assertEqual(evidence.harmed_groups, ("b0",))
        self.assertFalse(evidence.production_safe)

        archive = SpecialistArchive()
        self.assertTrue(archive.consider(evidence))
        self.assertIn(evidence.module.semantic_id(), archive.research)
        self.assertTrue(archive.specialist_elites())

    def test_single_replay_gain_is_not_strong_specialist(self):
        evidence = analyze_specialist(
            _route("OWN_MONEY", 500),
            (GroupOutcome("only", False, False, True, True, 1, 20000),),
            cohort_by_group={"only": "cash_crisis"},
            activation_count=100, action_change_count=50,
        )
        self.assertEqual(evidence.tier, "BOUNDARY_SPECIALIST")

    def test_inert_and_hard_negative_classes_are_explicit(self):
        inert = analyze_specialist(
            _route("OWN_MONEY", 500),
            (GroupOutcome("a", False, False, False, False, 0, 0),),
            cohort_by_group={"a": "broad"}, activation_count=0, action_change_count=0,
        )
        hard = analyze_specialist(
            _route("OWN_MONEY", 500),
            (GroupOutcome("a", True, True, True, False, 0, -9000),),
            cohort_by_group={"a": "broad"}, activation_count=20, action_change_count=20,
        )
        self.assertEqual(inert.tier, "INERT")
        self.assertEqual(hard.tier, "HARD_NEGATIVE")

    def test_diversity_report_ignores_threshold_jitter_and_reports_concentration(self):
        modules = (_route("OWN_MONEY", 500), _route("OWN_MONEY", 1000), _route("DAY", 2))
        report = diversity_report(modules, activation_profiles=("sparse", "broad", "mixed"))
        self.assertEqual(report.programs, 3)
        self.assertEqual(report.semantic_ids, 3)
        self.assertEqual(report.structural_skeletons, 2)
        self.assertAlmostEqual(report.top1_skeleton_share, 2 / 3)
        self.assertEqual(report.activation_profiles, 3)

    def test_composition_metadata_detects_route_conflict_and_separate_domain_compatibility(self):
        route = _route("OWN_MONEY", 500)
        other_route = _route("DAY", 2, route=212)
        rewrite = InvariantStrategicModule((InvariantRule(
            ModuleSketch("recovery", "INSTANT", "RECOVER_RESOURCE_INVARIANT"),
            "EACH_OWN_UNIT",
            LiftedGuard((GuardAtom("OWN_TILE_CONSECUTIVE_UNFED", "GE", 1),)),
            AbstractAction("UNIT_REWRITE", match_op=17, unit_op=15),
        ),))
        route_meta = composition_metadata(route)
        rewrite_meta = composition_metadata(rewrite)
        self.assertIn("route", route_meta.action_domains)
        self.assertIn("unit", rewrite_meta.action_domains)
        self.assertFalse(route_meta.compatible_with(composition_metadata(other_route)))
        self.assertTrue(route_meta.compatible_with(rewrite_meta))

    def test_quota_selection_preserves_specialist_and_diversity_classes(self):
        modules = tuple(_route("OWN_MONEY" if i < 4 else "DAY", i + 1, 70 + i) for i in range(8))
        evidences = []
        tiers = (
            "ELITE_SPECIALIST", "STRONG_SPECIALIST", "BOUNDARY_SPECIALIST", "HARD_NEGATIVE",
            "INERT", "GLOBAL_SAFE", "ELITE_SPECIALIST", "STRONG_SPECIALIST",
        )
        for module, tier in zip(modules, tiers):
            outcomes = (GroupOutcome(module.semantic_id(), False, False, tier != "INERT", tier not in {"HARD_NEGATIVE", "INERT"}, 0, 100),)
            evidence = analyze_specialist(
                module, outcomes, cohort_by_group={module.semantic_id(): tier.lower()},
                activation_count=1, action_change_count=int(tier != "INERT"), force_tier=tier,
            )
            evidences.append(evidence)
        selected = select_screen_quota(evidences, limit=6, seed=11)
        selected_tiers = {x.tier for x in selected}
        self.assertIn("ELITE_SPECIALIST", selected_tiers)
        self.assertIn("BOUNDARY_SPECIALIST", selected_tiers)
        self.assertIn("HARD_NEGATIVE", selected_tiers)
        self.assertGreaterEqual(len({repr(x.structural_skeleton) for x in selected}), 2)

    def test_quota_selection_keeps_training_contrast_at_scale(self):
        evidences = []
        for tier_index, tier in enumerate((
            "GLOBAL_SAFE", "ELITE_SPECIALIST", "STRONG_SPECIALIST",
            "BOUNDARY_SPECIALIST", "HARD_NEGATIVE", "INERT",
        )):
            for index in range(30):
                module = _route("OWN_MONEY", tier_index * 100 + index + 1, 1000 + tier_index * 100 + index)
                useful = tier not in {"HARD_NEGATIVE", "INERT"}
                outcomes = (GroupOutcome(module.semantic_id(), tier == "BOUNDARY_SPECIALIST", tier == "BOUNDARY_SPECIALIST", tier != "INERT", useful, 0, 100 if useful else -100),)
                evidences.append(analyze_specialist(
                    module, outcomes, cohort_by_group={module.semantic_id(): tier.lower()},
                    activation_count=1, action_change_count=int(tier != "INERT"), force_tier=tier,
                ))
        selected = select_screen_quota(evidences, limit=60, seed=31)
        counts = {tier: sum(row.tier == tier for row in selected) for tier in SPECIALIST_TIERS}
        self.assertGreaterEqual(counts["HARD_NEGATIVE"], 10)
        self.assertGreaterEqual(counts["INERT"], 8)
        self.assertGreaterEqual(counts["ELITE_SPECIALIST"], 12)

    def test_lineage_preserves_recursive_parent_and_source_relative_effect(self):
        root = _route("OWN_MONEY", 500)
        child = _route("OWN_MONEY", 1000)
        lineage = SpecialistLineage.root(root, parent_controller="w33_s172_expansion_v1")
        refined = lineage.child(child, operator="narrow_guard", source_margin_delta=1200)
        self.assertEqual(refined.depth, 1)
        self.assertEqual(refined.parent_semantic_id, root.semantic_id())
        self.assertEqual(refined.source_margin_delta, 1200)
        self.assertEqual(refined.root_controller, "w33_s172_expansion_v1")

    def test_coverage_matrix_exposes_major_domains_without_claiming_unobserved_support(self):
        rows = strategic_coverage_matrix(route_ids=(70, 212))
        domains = {row.domain for row in rows}
        self.assertTrue({"market", "workforce", "land", "crops", "animals", "inventory", "routing", "engine_mechanics"} <= domains)
        self.assertTrue(all(row.representable and row.cuda_executable and row.distillable for row in rows))
        self.assertTrue(any(not row.meaningful_observed for row in rows))

    def test_backend_retains_mixed_risk_candidate_in_specialist_archive(self):
        class MixedEvaluator:
            def evaluate(self, modules, panel, *, work_dir, panel_id):
                result = {}
                for index, module in enumerate(modules):
                    margins = (100, 100, 100) if index == 0 else (4300, 2900, -6000)
                    result[module.semantic_id()] = StateEvaluation(
                        module.semantic_id(), panel_id, tuple(row.case_id for row in panel),
                        margins, (0, 0, 0) if index == 0 else (1, 1, 1),
                        (0, 0, 0) if index == 0 else (2, 2, 2), (1, 1, 1),
                    )
                return result

        panel = (
            CaseRecord("a0", "ta0", 0, 1, "low_cash", role="discovery"),
            CaseRecord("a1", "ta1", 0, 2, "low_cash", role="discovery"),
            CaseRecord("b0", "tb0", 0, 3, "late_market", role="discovery"),
        )
        backend = InvariantSearchBackend(route_ids=(70, 212), evaluator=MixedEvaluator())
        module = _route("OWN_MONEY", 500)
        backend.evaluate_batch((module,), panel, work_dir="unused", panel_id="specialist")
        evidence = backend.specialist_archive.research[module.semantic_id()]
        self.assertEqual(evidence.tier, "ELITE_SPECIALIST")
        self.assertFalse(evidence.production_safe)

    def test_targeted_proposal_mode_limits_goal_families_without_identity(self):
        backend = InvariantSearchBackend(route_ids=(70, 212), evaluator=object())
        modules = backend.propose(
            n=32, seed=17,
            target_goal_ids=("weed_recovery", "endgame_liquidation"),
        )
        self.assertEqual(len(modules), 32)
        effects = {rule.sketch.effect_class for module in modules for rule in module.rules}
        self.assertTrue(effects <= {"RECOVER_RESOURCE_INVARIANT", "CREATE_FINANCING_RUNWAY", "ENGINE_MECHANIC"})
        self.assertTrue(any(rule.action.kind == "UNIT_REWRITE" for module in modules for rule in module.rules))
        self.assertTrue(any(rule.action.kind == "MARKET_INTENT" for module in modules for rule in module.rules))

    def test_proposal_profile_exposes_operator_costs(self):
        backend = InvariantSearchBackend(route_ids=(70, 212), evaluator=object())
        backend.ladder.level = 5
        backend.propose(n=64, seed=19)
        expected = {
            "goal_construction", "composition", "boundary_solver", "local_neighborhood",
            "recombination", "mcts", "cegis", "canonical_dedupe",
        }
        self.assertEqual(set(backend.last_proposal_profile), expected)
        self.assertTrue(all(value >= 0 for value in backend.last_proposal_profile.values()))

    def test_local_neighborhood_changes_real_structure_not_only_threshold(self):
        parent = _route("OWN_MONEY", 500)
        children = local_structural_neighborhood(parent, route_ids=(70, 212), budget=24, seed=9)
        self.assertGreaterEqual(len(children), 8)
        parent_shape = repr(parent.rules[0].guard.atoms)
        self.assertTrue(any(repr(child.rules[0].guard.atoms) != parent_shape for child in children))
        self.assertTrue(any(child.rules[0].action.route_id != 70 for child in children if child.rules[0].action.kind == "ROUTE_OVERRIDE"))
        first_features = {
            child.rules[0].guard.atoms[0].feature
            for child in children if child.rules[0].guard.atoms
        }
        self.assertGreaterEqual(len(first_features), 2)

    def test_local_neighborhood_explores_action_payload_not_only_guards(self):
        children = local_structural_neighborhood(_market(2), route_ids=(70, 212), budget=64, seed=13)
        quantities = {
            child.rules[0].action.quantity for child in children
            if child.rules[0].action.kind == "MARKET_INTENT"
        }
        self.assertGreaterEqual(len(quantities), 2)

    def test_capability_ladder_reaches_engine_and_composition_levels(self):
        self.assertEqual(CapabilityLadder.LEVELS[-3:], (
            "engine_mechanic_specialist", "specialist_composition", "new_structural_skeleton",
        ))

    def test_goal_catalog_has_broad_phase_and_mechanic_coverage(self):
        goals = default_search_goals((70, 212))
        goal_ids = {goal.goal_id for goal in goals}
        self.assertGreaterEqual(len(goals), 24)
        self.assertTrue({
            "opening_route_regime", "endgame_liquidation", "weed_recovery",
            "fertilizer_replenishment", "animal_acquisition", "crop_input_procurement",
            "market_order_cap_protection", "animal_collection_priority",
        } <= goal_ids)

    def test_boundary_learning_finds_public_regime_guard_between_help_and_harm(self):
        manifest = tuple(registry_manifest())
        pos = manifest.index(("OWN_MONEY", ()))

        def case(case_id, tape, money):
            values = [0] * len(manifest); values[pos] = money
            return CaseRecord(
                case_id, tape, 0, len(case_id), "cash", role="discovery",
                context_probe_steps=(120,), public_context_width=len(manifest),
                public_context_registry_width=len(manifest), public_context_values=tuple(values),
            )

        panel = (case("p0", "tp0", 1000), case("p1", "tp1", 1200), case("n0", "tn0", 100), case("n1", "tn1", 200))
        outcomes = (
            GroupOutcome("tp0", False, False, True, True, 0, 2000),
            GroupOutcome("tp1", False, False, True, True, 0, 1500),
            GroupOutcome("tn0", True, True, True, False, 0, -3000),
            GroupOutcome("tn1", True, False, True, False, 0, -1000),
        )
        guards = specialist_boundary_guards(outcomes, panel, limit=4)
        self.assertTrue(guards)
        money_guard = next(row for row in guards if row.atom.feature == "OWN_MONEY")
        self.assertEqual(money_guard.atom.cmp, "GE")
        self.assertGreaterEqual(money_guard.useful_groups, 2)
        self.assertEqual(money_guard.harmed_groups, 0)

    def test_grouped_cegis_queues_public_applicability_boundary_for_mixed_specialist(self):
        manifest = tuple(registry_manifest()); pos = manifest.index(("OWN_MONEY", ()))
        def case(case_id, tape, money):
            values = [0] * len(manifest); values[pos] = money
            return CaseRecord(
                case_id, tape, 0, len(case_id), "cash", role="discovery",
                context_probe_steps=(120,), public_context_width=len(manifest),
                public_context_registry_width=len(manifest), public_context_values=tuple(values),
            )
        panel = (case("p0", "tp0", 1000), case("p1", "tp1", 1200), case("n0", "tn0", 100), case("n1", "tn1", 200))
        class MixedEvaluator:
            def evaluate(self, modules, panel, *, work_dir, panel_id):
                out = {}
                for index, module in enumerate(modules):
                    margins = (100, 100, 100, 100) if index == 0 else (2100, 1600, -3000, -1000)
                    out[module.semantic_id()] = StateEvaluation(
                        module.semantic_id(), panel_id, tuple(row.case_id for row in panel), margins,
                        (0, 0, 0, 0) if index == 0 else (1, 1, 1, 1),
                        (0, 0, 0, 0) if index == 0 else (2, 2, 2, 2), (1, 1, 1, 1),
                    )
                return out
        backend = InvariantSearchBackend(route_ids=(70, 212), evaluator=MixedEvaluator())
        backend.ladder.level = backend.ladder.LEVELS.index("grouped_cegis")
        module = _route("DAY", 0)
        backend.evaluate_batch((module,), panel, work_dir="unused", panel_id="boundary")
        proposed = backend.propose(n=1, seed=8)
        features = {atom.feature for atom in proposed[0].rules[0].guard.atoms}
        self.assertIn("OWN_MONEY", features)


if __name__ == "__main__":
    unittest.main()
