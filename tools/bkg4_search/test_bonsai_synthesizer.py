from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from tools.bkg4_search.bonsai_composer import StrategicAuthority
from tools.bkg4_search.bounded_macro import BoundedRouteMacro
from tools.bkg4_search.macro_initiation_cegis import MacroEffectCase
from tools.bkg4_search.module_codec import ConditionGene
import tools.bkg4_search.bonsai_synthesizer as synth
import tools.bkg4_search.macro_initiation_cegis as cegis_mod
from tools.bkg4_search.bonsai_synthesizer import (
    ACCEPTED_RESEARCH_CANDIDATE,
    CPU_CUDA_MISMATCH,
    MIXED_LEGAL_STATE,
    NON_TRANSFERABLE,
    PROTECTED_REGRESSION,
    SEARCH_BUDGET_EXHAUSTED,
    CandidateEvidence,
    CaseRecord,
    SynthesisBudget,
    SynthesisConfig,
    SyntheticBackend,
    allocate_route_id,
    assess_certification,
    build_capture_request,
    build_search_grid,
    build_specialist_declaration,
    build_synthesized_macro,
    build_target_cases,
    compare_parity_rows,
    detect_mixed_legal_states,
    extend_composition_declaration,
    make_specialist_name,
    plan_falsification_panels,
    rank_candidates,
    rank_loss_frontier,
    run_cegis,
    synthesize_next_specialist,
    threshold_robustness,
    validate_identity_free_gate,
)


def case(i: int, *, family: int, seat: int, seed: int, protected: int = 1, own: int = 0) -> CaseRecord:
    return CaseRecord(
        case_id=f"c{i}",
        path=f"/tmp/tape_{seed}.json",
        seat=seat,
        seed=seed,
        family=family,
        protected=protected,
        own_history=own,
        role=0,
    )


def ctx(**values: int):
    return {(k, 0): v for k, v in values.items()}


class BonsaiSynthesizerPureTests(unittest.TestCase):
    def test_01_frontier_ranking_is_deterministic_and_severity_aware(self):
        cases = [case(0, family=2, seat=0, seed=10), case(1, family=2, seat=1, seed=10), case(2, family=1, seat=0, seed=20)]
        margins = [-20, -10, -25]
        a = rank_loss_frontier(cases, margins, max_families=8)
        b = rank_loss_frontier(tuple(reversed(cases)), tuple(reversed(margins)), max_families=8)
        self.assertEqual([(x.family, x.priority_score) for x in a], [(x.family, x.priority_score) for x in b])
        self.assertEqual(a[0].family, 2)
        self.assertEqual(a[0].both_seat_trajectories, 1)

    def test_02_target_family_contains_only_ranked_losses_and_records_why(self):
        cases = [case(0, family=4, seat=0, seed=1), case(1, family=4, seat=1, seed=1), case(2, family=4, seat=0, seed=2)]
        frontier = rank_loss_frontier(cases, [-9, -8, 2], max_families=3)[0]
        selected = build_target_cases(frontier, cases, [-9, -8, 2], max_cases=8)
        self.assertEqual([x.case_id for x in selected.cases], ["c0", "c1"])
        self.assertIn("severity", selected.grouping_features)
        self.assertEqual(selected.both_seat_support, 1)

    def test_03_search_grid_is_cross_product_bounded_and_has_baseline(self):
        grid = build_search_grid(["route_22.json", "route_64.json"], [96, 120], [12, 24], max_candidates=9)
        self.assertEqual(grid[0].candidate_id, 0)
        self.assertEqual(len(grid), 9)
        with self.assertRaisesRegex(ValueError, "budget"):
            build_search_grid(["a", "b"], [1, 2], [3, 4], max_candidates=8)

    def test_04_candidate_ranking_uses_cross_support_before_aggregate(self):
        cases = [case(0, family=1, seat=0, seed=1), case(1, family=1, seat=1, seed=1), case(2, family=2, seat=0, seed=2)]
        broad = CandidateEvidence(1, 10, "r10", 120, 24, (-5, -4, -3), (6, 5, 4), (3, 3, 3))
        spike = CandidateEvidence(2, 11, "r11", 120, 24, (-5, -4, -3), (20, -1, -1), (3, 3, 3))
        ranked = rank_candidates([spike, broad], cases)
        self.assertEqual(ranked[0].candidate_id, 1)
        self.assertGreaterEqual(ranked[0].independent_positive_support, 2)

    def test_05_failure_reason_is_propagated_machine_readably(self):
        backend = SyntheticBackend(candidate_outcomes=[{"failure": NON_TRANSFERABLE}])
        result = synthesize_next_specialist(SynthesisConfig.synthetic(), backend=backend)
        self.assertEqual(result.verdict, "REJECTED")
        self.assertEqual(result.terminal_reason, SEARCH_BUDGET_EXHAUSTED)
        self.assertEqual(result.rejections[0]["reason"], NON_TRANSFERABLE)

    def test_06_transfer_rejects_one_case_miracle(self):
        c = CandidateEvidence(1, 10, "r10", 100, 12, (-5, 10), (20, 0), (1, 1))
        rows = [case(0, family=1, seat=0, seed=1), case(1, family=2, seat=0, seed=2)]
        ranked = rank_candidates([c], rows)[0]
        self.assertFalse(ranked.transferable(min_positive=2, min_independent=2))

    def test_07_mixed_identical_legal_state_is_terminally_detected(self):
        rows = [
            MacroEffectCase("p", -5, 5, ctx(OPP_UNITS=4), (1, 2, 3), "a"),
            MacroEffectCase("h", 10, 1, ctx(OPP_UNITS=4), (1, 2, 3), "b"),
        ]
        self.assertEqual(detect_mixed_legal_states(rows), (("h", "p"),))

    def test_08_capture_request_uses_candidate_start_dynamically(self):
        cfg = SynthesisConfig.synthetic()
        req = build_capture_request(cfg, Path("cases.tsv"), 173, Path("out"))
        self.assertEqual(req.probes, (173,))
        self.assertIn("bkg4_information_state_capture", req.argv[0])
        self.assertNotIn("156", " ".join(req.argv))

    def test_09_cegis_adds_harmful_counterexample_and_refines_gate(self):
        discovery = [MacroEffectCase("p", -5, 5, ctx(OPP_UNITS=8, OPP_LAND=2), support_group="a")]
        counter = [MacroEffectCase("h", 10, 2, ctx(OPP_UNITS=8, OPP_LAND=1), support_group="b")]
        result = run_cegis(discovery, counter, (), max_conditions=4, max_iterations=4)
        self.assertIsNotNone(result.final_gate)
        self.assertEqual(result.final_score.harmful, 0)
        self.assertGreaterEqual(len(result.trace), 1)

    def test_10_config_rejects_manual_final_gate_injection(self):
        with self.assertRaisesRegex(ValueError, "manual.*gate"):
            SynthesisConfig.from_dict({**SynthesisConfig.synthetic().canonical(), "final_gate": []})

    def test_11_threshold_robustness_reports_safe_and_unsafe_neighbors(self):
        gate = ((ConditionGene("OPP_MONEY", "GE", 0, 100),),)
        rows = [
            MacroEffectCase("p", -2, 5, ctx(OPP_MONEY=100), support_group="a"),
            MacroEffectCase("h", 10, 1, ctx(OPP_MONEY=99), support_group="b"),
        ]
        report = threshold_robustness(gate, rows, radius=2)
        self.assertTrue(report[0]["safe"])
        self.assertTrue(any(not x["safe"] for x in report))

    def test_12_bounded_macro_is_created_from_selected_evidence(self):
        cand = CandidateEvidence(1, 410, "/x/route_410.json", 120, 18, (-1, -1), (2, 3), (4, 4))
        gate = ((ConditionGene("OPP_MONEY", "LE", 0, 120),),)
        macro = build_synthesized_macro(cand, gate, route_id=219, base_route_id=64, fragment_hash="f" * 64)
        self.assertIsInstance(macro, BoundedRouteMacro)
        self.assertEqual((macro.route_id, macro.start_step, macro.duration), (219, 120, 18))
        self.assertEqual(macro.provenance["source_route_id"], 410)

    def test_13_route_id_allocation_uses_next_free_strategic_id(self):
        self.assertEqual(allocate_route_id({213, 216, 217, 218, 220}), 219)
        self.assertEqual(allocate_route_id(set(range(216, 221))), 221)

    def test_14_specialist_declaration_is_generated_from_macro(self):
        macro = BoundedRouteMacro("synthesized_route10_s100_x", 219, 64, 100, 12, ((ConditionGene("OWN_MONEY", "GE", 0, 1),),), {"source_kind": "autonomous_synthesis", "source_route_id": 10, "fragment_sha256": "a" * 64})
        decl = build_specialist_declaration(macro, "route.json", "macro.json")
        self.assertEqual(decl.route_id, 219)
        self.assertEqual(decl.semantic_id, macro.semantic_id)
        self.assertEqual(decl.authorities, ("strategic_route_override",))

    def test_15_composition_extension_adds_declaration_without_manual_order(self):
        base = {"schema": 1, "kind": "bounded_bonsai_composition_v1", "output_name": "base", "specialists": [], "preemptions": [], "strategic_authorities": [{"name": "strategic_route_override", "blackboard_slot": "route_override", "blackboard_type": "I32"}]}
        macro = BoundedRouteMacro("synthesized_route10_s100_x", 219, 64, 100, 12, ((ConditionGene("OWN_MONEY", "GE", 0, 1),),), {"source_kind": "autonomous_synthesis", "source_route_id": 10, "fragment_sha256": "a" * 64})
        decl = build_specialist_declaration(macro, "route.json", "macro.json")
        out = extend_composition_declaration(base, decl)
        self.assertEqual(out["specialists"][-1]["route_id"], 219)
        self.assertNotIn("composition_order", out)

    def test_16_unresolved_composition_conflict_rejects_then_backend_advances(self):
        backend = SyntheticBackend(candidate_outcomes=[{"failure": "COMPOSITION_CONFLICT"}, {"accepted": True}])
        result = synthesize_next_specialist(SynthesisConfig.synthetic(), backend=backend)
        self.assertEqual(result.verdict, ACCEPTED_RESEARCH_CANDIDATE)
        self.assertEqual(result.rejections[0]["reason"], "COMPOSITION_CONFLICT")
        self.assertEqual(backend.candidates_attempted, 2)

    def test_17_falsification_panel_plan_is_declarative_and_sealed_last(self):
        cfg = SynthesisConfig.synthetic()
        plan = plan_falsification_panels(cfg)
        self.assertEqual([x.role for x in plan][-1], "sealed")
        self.assertTrue(any(x.role == "self_history" for x in plan))

    def test_18_protected_regression_is_rejected_even_without_win_to_loss(self):
        cases = [case(0, family=1, seat=0, seed=1, protected=1)]
        report = assess_certification(cases, [10], [9])
        self.assertFalse(report.accepted)
        self.assertEqual(report.failure_reason, PROTECTED_REGRESSION)

    def test_19_cpu_cuda_parity_failure_is_machine_reason(self):
        report = compare_parity_rows(["a", "b"], ["a", "c"])
        self.assertFalse(report.accepted)
        self.assertEqual(report.failure_reason, CPU_CUDA_MISMATCH)
        self.assertEqual(report.mismatches, 1)

    def test_20_budget_exhaustion_is_terminal_and_bounded(self):
        budget = SynthesisBudget(max_target_families=1, max_candidate_routes=1, max_coarse_candidates=2, max_refinement_candidates=2, max_cuda_games=10, max_cegis_iterations=2, max_conditions=4, max_certification_cases=5)
        cfg = replace(SynthesisConfig.synthetic(), budget=budget)
        backend = SyntheticBackend(candidate_outcomes=[{"failure": NON_TRANSFERABLE}, {"accepted": True}])
        result = synthesize_next_specialist(cfg, backend=backend)
        self.assertEqual(result.verdict, "REJECTED")
        self.assertEqual(result.terminal_reason, SEARCH_BUDGET_EXHAUSTED)
        self.assertEqual(backend.candidates_attempted, 1)

    def test_21_retry_next_candidate_after_gate_failure(self):
        backend = SyntheticBackend(candidate_outcomes=[{"failure": "NO_SAFE_GATE"}, {"accepted": True}])
        result = synthesize_next_specialist(SynthesisConfig.synthetic(), backend=backend)
        self.assertEqual(result.verdict, ACCEPTED_RESEARCH_CANDIDATE)
        self.assertEqual([x["reason"] for x in result.rejections], ["NO_SAFE_GATE"])

    def test_22_same_inputs_produce_same_run_identity_and_result(self):
        cfg = SynthesisConfig.synthetic()
        a = synthesize_next_specialist(cfg, backend=SyntheticBackend(candidate_outcomes=[{"accepted": True}]))
        b = synthesize_next_specialist(cfg, backend=SyntheticBackend(candidate_outcomes=[{"accepted": True}]))
        self.assertEqual(a.run_id, b.run_id)
        self.assertEqual(a.canonical(), b.canonical())

    def test_23_identity_like_runtime_feature_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "identity"):
            validate_identity_free_gate(((ConditionGene("OPPONENT_TEAM", "EQ", 0, 1),),))

    def test_24_legal_own_seed_runtime_feature_is_not_mistaken_for_offline_seed_identity(self):
        gate = ((ConditionGene("OWN_SEED", "GE", 0, 1),),)
        validate_identity_free_gate(gate)
        discovery = [
            MacroEffectCase("p0", -5, 5, ctx(OWN_SEED=3), support_group="a"),
            MacroEffectCase("p1", -4, 6, ctx(OWN_SEED=3), support_group="b"),
        ]
        result = run_cegis(discovery, (), (), max_conditions=4, max_iterations=4)
        self.assertIsNotNone(result.final_gate)
        self.assertEqual(result.final_gate, ((ConditionGene("OWN_SEED", "EQ", 0, 3),),))

    def test_24_end_to_end_synthetic_backend_accepts_without_manual_strategy_fields(self):
        cfg = SynthesisConfig.synthetic()
        backend = SyntheticBackend(candidate_outcomes=[{"accepted": True}])
        result = synthesize_next_specialist(cfg, backend=backend)
        self.assertEqual(result.verdict, ACCEPTED_RESEARCH_CANDIDATE)
        self.assertTrue(result.autonomy_proof)
        config_text = json.dumps(cfg.canonical(), sort_keys=True).lower()
        for forbidden in ("final_gate", "selected_route", "selected_start", "selected_duration"):
            self.assertNotIn(forbidden, config_text)

    def test_25_real_cuda_command_path_is_constructed_without_cpu_game_runner(self):
        cfg = SynthesisConfig.synthetic()
        request = build_capture_request(cfg, Path("cases.tsv"), 120, Path("out"))
        text = " ".join(request.argv)
        self.assertIn("bkg4_information_state_capture", text)
        self.assertNotIn("official", text.lower())
        self.assertNotIn("cpp-engine", text.lower())

    def test_26_capture_context_cardinality_matches_cases(self):
        class FakeCapture:
            cases = 3
            registry_width = 2
            bb_width = 2
            compat_width = 0
            def frame(self, case_index, probe_index):
                return (case_index, 10 + case_index, 0, 64)
        original = synth.registry_manifest
        try:
            synth.registry_manifest = lambda: [("STEP", ()), ("OPP_MONEY", ())]
            contexts, signatures, base_routes = synth._capture_contexts(FakeCapture())
        finally:
            synth.registry_manifest = original
        self.assertEqual((len(contexts), len(signatures), len(base_routes)), (3, 3, 3))
        self.assertEqual(base_routes, [64, 64, 64])

    def test_28_cegis_singleton_l2w_proposals_are_explicitly_bounded(self):
        rows = [
            MacroEffectCase(f"p{i}", -10, 10 + i, ctx(OPP_MONEY=100 + i), support_group=f"g{i}")
            for i in range(10)
        ] + [MacroEffectCase("h", 10, 1, ctx(OPP_MONEY=50), support_group="harm")]
        calls = []
        original = cegis_mod.synthesize_contrastive_conditions
        try:
            def fake(goal, contexts, **kwargs):
                calls.append(tuple(goal.target_case_ids))
                return []
            cegis_mod.synthesize_contrastive_conditions = fake
            cegis_mod.enumerate_safe_conjunctions(
                rows,
                max_conditions=4,
                limit=8,
                feature_budget=24,
                singleton_l2w_budget=3,
            )
        finally:
            cegis_mod.synthesize_contrastive_conditions = original
        self.assertEqual(len(calls), 5)
        self.assertEqual(sum(len(ids) == 1 for ids in calls), 3)
        self.assertEqual(set(calls[0]), {f"p{i}" for i in range(10)})
        self.assertEqual(set(calls[1]), {f"p{i}" for i in range(10)})

    def test_29_terminal_rejection_uses_required_verdict_and_preserves_attempt_evidence(self):
        cfg = SynthesisConfig.synthetic()
        backend = SyntheticBackend(candidate_outcomes=[{"failure": NON_TRANSFERABLE}])
        result = synthesize_next_specialist(cfg, backend=backend)
        self.assertEqual(result.verdict, "REJECTED")
        self.assertEqual(result.terminal_reason, SEARCH_BUDGET_EXHAUSTED)
        self.assertEqual(len(result.candidate_attempts), 1)
        self.assertEqual(result.candidate_attempts[0]["rejection"]["reason"], NON_TRANSFERABLE)


if __name__ == "__main__":
    unittest.main(verbosity=2)
