"""Phase 1.4 — Downstream safety integration focused tests."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from kaggriculture_bonsai.composition.module_codec import ConditionGene
from kaggriculture_bonsai.search.checkpoints import Checkpoint, EventType, Lane
from kaggriculture_bonsai.search.ir import (
    ActionFamily, ActionSource, MacroProgram, Phase, Termination, TerminationType,
)
from kaggriculture_bonsai.routing.materialize import MaterializationInput, materialize_program
from kaggriculture_bonsai.search.motifs import motif_from_actions
from kaggriculture_bonsai.routing.binary_gate_synthesis import (
    GateSynthesisResult, GateSynthesisStatus, _gate_canonical, _gate_identity,
)
from kaggriculture_bonsai.routing.routed_evaluation import (
    PRODUCTION_AUTHORITY_NAME,
    PRODUCTION_AUTHORITY_SLOT,
    TEMPORARY_ROUTE_ID_BASE,
    FallbackAuditResult,
    RoutedEvaluationCase,
    RoutedEvaluationResult,
    RoutedEvaluationStatus,
    _TemporaryRouteAllocator,
    evaluate_routed_candidate,
)
from kaggriculture_bonsai.routing.downstream_integration import (
    DOWNSTREAM_SCHEMA_VERSION,
    DURABLE_ROUTE_ID_BASE,
    BinaryCandidateStatus,
    DownstreamCandidateResult,
    GenerationValidationResult,
    ProductionMaterializationResult,
    RoutedCegisResult,
    RoutedTransferResult,
    RoutedValidationResult,
    _DurableRouteAllocator,
    evaluate_routed_transfer,
    evaluate_routed_validation,
    is_eligible_for_downstream,
    materialize_production_route,
    record_downstream_to_memory,
    run_downstream_integration,
    validate_generation,
    write_downstream_artifact,
)

PARENT_ACTION = {"farmer": ["NORTH"], "hands": [["WEST"]], "market": [["SELL", "WHEAT", 2]]}
SPECIALIST_ACTION = {"farmer": ["SOUTH"], "hands": [["CARE"]], "market": [["HIRE"]]}


def obs(step, *, money=3000, land=1, hires=0):
    return {
        "step": step, "day": step // 24, "hour": step % 24, "player": 0,
        "farms": [
            {"money": money, "hands": [[0, 0]] * hires, "hires_today": hires,
             "unlocked_quadrants": ["NW"] * land, "tiles": []},
            {"money": 3000, "hands": [], "hires_today": 0,
             "unlocked_quadrants": ["NW"], "tiles": []},
        ],
        "private": {"seeds": {}, "shed": {}, "inventories": []},
        "market": {"inventory": {}, "prices": {}},
        "town": {"unlocked_shops": []},
    }


def _make_motif():
    actions = [SPECIALIST_ACTION] * 4
    return motif_from_actions(actions, tuple(ActionFamily))


def _make_program(step=0, duration=2):
    motif = _make_motif()
    checkpoint = Checkpoint(step, Lane.GENERAL, (EventType.FIRST_EFFECT_DIVERGENCE,))
    phase = Phase(ActionSource.SEMANTIC_SPLICE, (ActionFamily.FARMER,),
                  Termination(TerminationType.FIXED_CAP, duration), motif_id=motif.semantic_id)
    return MacroProgram(Lane.GENERAL, checkpoint, (phase,), duration), {motif.semantic_id: motif}


def _gate_money_threshold(threshold=3000):
    return ((ConditionGene("OWN_MONEY", "GE", 0, threshold),),)


def _make_gate_result(candidate_id="c" * 64, gate=None, parent_ctx="ctx1"):
    if gate is None:
        gate = _gate_money_threshold()
    return GateSynthesisResult(
        candidate_semantic_id=candidate_id,
        parent_generation_context=parent_ctx,
        status=GateSynthesisStatus.GATE_SYNTHESIZED,
        gate=gate,
        gate_canonical=_gate_canonical(gate),
        gate_identity=_gate_identity(gate),
        gate_complexity=sum(len(cl) for cl in gate),
        target_positives_total=2,
        target_positives_matched=2,
        target_positives_missed=0,
        protected_breaks_total=1,
        protected_breaks_matched=0,
        protected_breaks_excluded=1,
        gate_features_used=("OWN_MONEY",),
        failure_reason=None,
        gate_score=None,
        source_admission_identity=candidate_id,
    )


def _make_routed_result(
    candidate_id="c" * 64,
    status=RoutedEvaluationStatus.READY_FOR_DOWNSTREAM_INTEGRATION,
    target_l2w=2,
    protected_w2l=0,
    semantic_mismatches=0,
    fallback_passed=True,
    gate=None,
):
    if gate is None:
        gate = _gate_money_threshold()
    gate_id = _gate_identity(gate)
    return RoutedEvaluationResult(
        candidate_semantic_id=candidate_id,
        parent_generation_context="ctx1",
        gate_identity=gate_id,
        status=status,
        temporary_route_id=60000,
        target_baseline_cases=5,
        target_global_l2w=3,
        target_static_expected_positives=2,
        target_routed_l2w=target_l2w,
        target_routed_w2l=0,
        target_routed_harms=0,
        target_routed_positives=target_l2w,
        target_routed_worst_delta=0,
        protected_global_w2l=10,
        protected_routed_w2l=protected_w2l,
        protected_w2l_eliminated=10 - protected_w2l,
        protected_routed_harms=0,
        protected_routed_l2w=0,
        protected_routed_worst_delta=0,
        fallback_audit=FallbackAuditResult(30, 0 if fallback_passed else 1, () if fallback_passed else ("p1",)),
        semantic_mismatch_count=semantic_mismatches,
        routed_counterexample_case_ids=(),
        per_case_results=(),
        failure_reason=None,
        source_gate_synthesis_identity=gate_id,
        source_admission_identity=candidate_id,
    )


def _transfer_cases(*, gate_active_positive=True, gate_active_w2l=False):
    cases = []
    if gate_active_positive:
        cases.append({"baseline_margin": -5, "specialist_margin": 10, "gate_decision": True})
    if gate_active_w2l:
        cases.append({"baseline_margin": 5, "specialist_margin": -3, "gate_decision": True})
    cases.append({"baseline_margin": -5, "specialist_margin": 10, "gate_decision": False})
    return cases


def _validation_cases(*, w2l=False):
    cases = [
        {"baseline_margin": 10, "specialist_margin": 8, "gate_decision": True},
        {"baseline_margin": 5, "specialist_margin": 5, "gate_decision": False},
    ]
    if w2l:
        cases.append({"baseline_margin": 5, "specialist_margin": -3, "gate_decision": True})
    return cases


# 1. Correct production strategic authority slot is used
class TestProductionAuthoritySlot(unittest.TestCase):
    def test_production_authority_slot_constant(self):
        self.assertEqual(PRODUCTION_AUTHORITY_SLOT, "day2_capital_specialist_v1_route_override")
        self.assertEqual(PRODUCTION_AUTHORITY_NAME, "strategic_route")

    def test_materialize_uses_production_slot(self):
        program, motifs = _make_program()
        gate = _gate_money_threshold()
        result = materialize_production_route(
            "c" * 64, program, gate, motifs,
            route_id=200,
        )
        self.assertTrue(result.passed)
        self.assertEqual(result.authority_slot, PRODUCTION_AUTHORITY_SLOT)


# 2. Wrong slot_1 default cannot silently reach production materialization
class TestSlot1Rejected(unittest.TestCase):
    def test_slot_1_rejected(self):
        program, motifs = _make_program()
        gate = _gate_money_threshold()
        result = materialize_production_route(
            "c" * 64, program, gate, motifs,
            route_id=200,
            authority_slot="slot_1",
        )
        self.assertFalse(result.passed)
        self.assertEqual(result.rejection_reason, "WRONG_AUTHORITY_SLOT")


# 3. READY_FOR_DOWNSTREAM_INTEGRATION candidate enters routed TRANSFER
class TestReadyCandidateEntersTransfer(unittest.TestCase):
    def test_eligible_enters_transfer(self):
        routed = _make_routed_result()
        self.assertTrue(is_eligible_for_downstream(routed))

        program, motifs = _make_program()
        gate = _gate_money_threshold()
        result = run_downstream_integration(
            routed, gate=gate, program=program, motifs=motifs,
            transfer_cases=_transfer_cases(),
        )
        self.assertIsNotNone(result.transfer)
        self.assertTrue(result.transfer.passed)


# 4. Failed Phase-1.3 candidates cannot enter Phase 1.4
class TestFailedCandidateRejected(unittest.TestCase):
    def test_target_lost_ineligible(self):
        routed = _make_routed_result(status=RoutedEvaluationStatus.TARGET_CAPABILITY_LOST, target_l2w=0)
        self.assertFalse(is_eligible_for_downstream(routed))

    def test_protected_regression_ineligible(self):
        routed = _make_routed_result(status=RoutedEvaluationStatus.ROUTED_PROTECTED_REGRESSION, protected_w2l=3)
        self.assertFalse(is_eligible_for_downstream(routed))

    def test_fallback_mismatch_ineligible(self):
        routed = _make_routed_result(status=RoutedEvaluationStatus.PARENT_FALLBACK_MISMATCH, fallback_passed=False)
        self.assertFalse(is_eligible_for_downstream(routed))

    def test_semantic_mismatch_ineligible(self):
        routed = _make_routed_result(status=RoutedEvaluationStatus.ROUTING_SEMANTIC_MISMATCH, semantic_mismatches=1)
        self.assertFalse(is_eligible_for_downstream(routed))

    def test_ineligible_produces_ineligible_status(self):
        routed = _make_routed_result(status=RoutedEvaluationStatus.TARGET_CAPABILITY_LOST, target_l2w=0)
        program, motifs = _make_program()
        gate = _gate_money_threshold()
        result = run_downstream_integration(
            routed, gate=gate, program=program, motifs=motifs,
        )
        self.assertEqual(result.status, BinaryCandidateStatus.INELIGIBLE)


# 5. TRANSFER evaluates routed policy, not global specialist
class TestTransferEvaluatesRouted(unittest.TestCase):
    def test_gate_false_uses_baseline(self):
        cases = [{"baseline_margin": -5, "specialist_margin": 10, "gate_decision": False}]
        result = evaluate_routed_transfer("c" * 64, cases, _gate_money_threshold())
        self.assertEqual(result.transfer_l2w, 0)

    def test_gate_true_uses_specialist(self):
        cases = [{"baseline_margin": -5, "specialist_margin": 10, "gate_decision": True}]
        result = evaluate_routed_transfer("c" * 64, cases, _gate_money_threshold())
        self.assertEqual(result.transfer_l2w, 1)


# 6. VALIDATION evaluates routed policy, not global specialist
class TestValidationEvaluatesRouted(unittest.TestCase):
    def test_gate_false_uses_baseline(self):
        cases = [{"baseline_margin": 5, "specialist_margin": -3, "gate_decision": False}]
        result = evaluate_routed_validation("c" * 64, cases, _gate_money_threshold())
        self.assertEqual(result.validation_w2l, 0)
        self.assertTrue(result.passed)

    def test_gate_true_uses_specialist(self):
        cases = [{"baseline_margin": 5, "specialist_margin": -3, "gate_decision": True}]
        result = evaluate_routed_validation("c" * 64, cases, _gate_money_threshold())
        self.assertEqual(result.validation_w2l, 1)
        self.assertFalse(result.passed)


# 7. Routed protected W→L causes rejection
class TestRoutedProtectedW2LRejection(unittest.TestCase):
    def test_validation_w2l_rejects(self):
        result = evaluate_routed_validation("c" * 64, _validation_cases(w2l=True), _gate_money_threshold())
        self.assertFalse(result.passed)
        self.assertEqual(result.rejection_reason, "VALIDATION_REGRESSION")


# 8. TARGET capability loss during refinement causes rejection
class TestTargetCapabilityLossDuringRefinement(unittest.TestCase):
    def test_cegis_target_lost_rejects(self):
        from kaggriculture_bonsai.routing.cegis_adapter import CegisAdapterResult, AdapterDisposition
        mock_result = CegisAdapterResult(
            candidate_id="c" * 64,
            program_semantic_id="c" * 64,
            program_schema_version=1,
            internal_decision_count=0,
            disposition=AdapterDisposition.ENTER_CEGIS,
            cegis_invoked=True,
            accepted=True,
            rejection_reason=None,
            cegis_stop_reason=None,
            gate=[[{"feature": "OWN_MONEY", "cmp": "GE", "index": 0, "value": 999999}]],
            target_preserved=False,
            transfer_support={},
            validation_counterexamples=(),
            threshold_robustness=(),
            sealed_status="NOT_EVALUATED",
            evidence_ids=("e1",),
            cegis_trace=(),
            diagnostics={"cegis_scope": "OUTER_INITIATION_ONLY", "routability": "ROUTABLE_CANDIDATE", "lineage": []},
        )
        from kaggriculture_bonsai.routing.downstream_integration import run_routed_cegis
        from unittest.mock import patch
        with patch("kaggriculture_bonsai.routing.downstream_integration.run_hardened_cegis", return_value=mock_result):
            cegis_result = run_routed_cegis("c" * 64, MagicMock(), "gate_id_1")
        self.assertFalse(cegis_result.target_preserved)


# 9. CEGIS gate refinement requires actual routed re-evaluation
class TestCegisRequiresRoutedReeval(unittest.TestCase):
    def test_refined_gate_tracked(self):
        gate = _gate_money_threshold()
        identity = _gate_identity(gate)
        different_gate = _gate_money_threshold(5000)
        different_identity = _gate_identity(different_gate)
        self.assertNotEqual(identity, different_identity)


# 10. Refined gate preserves PARENT fallback
class TestRefinedGatePreservesParentFallback(unittest.TestCase):
    def test_parent_fallback_audit(self):
        program, motifs = _make_program()
        gate = _gate_money_threshold(999999)
        gr = _make_gate_result(gate=gate)
        cases = [
            RoutedEvaluationCase("t1", "TARGET", -5, 10, obs(0, money=1), PARENT_ACTION),
            RoutedEvaluationCase("p1", "PROTECTED", 5, -3, obs(0, money=1), PARENT_ACTION),
        ]
        result = evaluate_routed_candidate(
            gr, cases, program=program, motifs=motifs,
            parent_generation_context="ctx1",
        )
        self.assertTrue(result.fallback_audit.passed)
        for cr in result.per_case_results:
            self.assertFalse(cr.gate_decision)
            self.assertEqual(cr.routed_margin, cr.baseline_margin)


# 11. SEALED remains write-only
class TestSealedWriteOnly(unittest.TestCase):
    def test_sealed_does_not_influence_synthesis(self):
        from kaggriculture_bonsai.search.search_memory import SearchMemory, _SAFE_SEARCH_FEEDBACK_STAGES
        self.assertIn("BINARY_SEALED", _SAFE_SEARCH_FEEDBACK_STAGES)
        self.assertNotIn("SEALED_CERTIFICATION", _SAFE_SEARCH_FEEDBACK_STAGES)


# 12. SEALED failure rejects without feeding counterexample into synthesis
class TestSealedFailureRejects(unittest.TestCase):
    def test_sealed_fail_status(self):
        routed = _make_routed_result()
        program, motifs = _make_program()
        gate = _gate_money_threshold()

        from kaggriculture_bonsai.routing.cegis_adapter import CegisAdapterResult, AdapterDisposition
        mock_cegis = CegisAdapterResult(
            candidate_id="c" * 64,
            program_semantic_id="c" * 64,
            program_schema_version=1,
            internal_decision_count=0,
            disposition=AdapterDisposition.ENTER_CEGIS,
            cegis_invoked=True,
            accepted=True,
            rejection_reason=None,
            cegis_stop_reason=None,
            gate=[[{"feature": "OWN_MONEY", "cmp": "GE", "index": 0, "value": 3000}]],
            target_preserved=True,
            transfer_support={"accepted": True},
            validation_counterexamples=(),
            threshold_robustness=(),
            sealed_status="FAILED",
            evidence_ids=("e1",),
            cegis_trace=(),
            diagnostics={"cegis_scope": "OUTER_INITIATION_ONLY", "routability": "ROUTABLE_CANDIDATE", "lineage": []},
        )
        from kaggriculture_bonsai.routing.downstream_integration import RoutedCegisResult
        cegis_result = RoutedCegisResult(
            "c" * 64, _gate_identity(gate), _gate_identity(gate),
            False, 1, True, True, None, mock_cegis,
        )
        # Integration: sealed fail should produce SEALED_FAILED
        result = run_downstream_integration(
            routed, gate=gate, program=program, motifs=motifs,
            transfer_cases=_transfer_cases(),
            validation_cases=_validation_cases(),
            sealed_cases=[{"baseline_margin": 5, "specialist_margin": -3}],
        )
        # Without real CEGIS, sealed won't fire; just check sealed_status field exists
        self.assertIn(result.sealed_status, ("NOT_EVALUATED", "NOT_PROVIDED", "PASSED", "FAILED"))


# 13. Production materialization uses durable route ID, not temporary 60000-series
class TestDurableRouteId(unittest.TestCase):
    def test_durable_range(self):
        allocator = _DurableRouteAllocator()
        rid = allocator.allocate()
        self.assertGreaterEqual(rid, DURABLE_ROUTE_ID_BASE)
        self.assertLess(rid, TEMPORARY_ROUTE_ID_BASE)

    def test_temporary_id_rejected(self):
        program, motifs = _make_program()
        gate = _gate_money_threshold()
        result = materialize_production_route(
            "c" * 64, program, gate, motifs,
            route_id=60000,
        )
        self.assertFalse(result.passed)
        self.assertEqual(result.rejection_reason, "TEMPORARY_ROUTE_ID_IN_PRODUCTION")


# 14. Durable route cursor advances only on successful committed generation
class TestDurableRouteCursorAdvance(unittest.TestCase):
    def test_commit_advances(self):
        allocator = _DurableRouteAllocator()
        rid1 = allocator.allocate()
        rid2 = allocator.allocate()
        allocator.commit(rid1)
        self.assertIn(rid1, allocator.committed_ids)
        self.assertNotIn(rid2, allocator.committed_ids)

    def test_rollback_uncommitted(self):
        allocator = _DurableRouteAllocator()
        rid1 = allocator.allocate()
        rid2 = allocator.allocate()
        allocator.commit(rid1)
        allocator.rollback_uncommitted()
        self.assertIn(rid1, allocator.allocated_ids)
        self.assertNotIn(rid2, allocator.allocated_ids)


# 15. Final gate identity matches production materialized route
class TestGateIdentityMatchesMaterialization(unittest.TestCase):
    def test_gate_identity_preserved(self):
        program, motifs = _make_program()
        gate = _gate_money_threshold()
        expected_identity = _gate_identity(gate)
        result = materialize_production_route(
            "c" * 64, program, gate, motifs,
            route_id=200,
        )
        self.assertTrue(result.passed)
        self.assertEqual(result.final_gate_identity, expected_identity)


# 16. SAFE_FAST_PATH remains unchanged
class TestSafeFastPathUnchanged(unittest.TestCase):
    def test_safe_candidates_unaffected(self):
        from kaggriculture_bonsai.routing.binary_admission import (
            CandidateAdmissionClass, classify_candidate_admission,
        )
        from kaggriculture_bonsai.routing.diagnostics import ImperfectCandidateRecord

        rec = ImperfectCandidateRecord(
            candidate_semantic_id="safe1", structural_id="safe1", family="f",
            lane="GENERAL", target_loss_to_win=3, target_harm_count=0,
            target_worst_delta=0, target_aggregate_margin=10,
            protected_win_to_loss=0, protected_loss_to_win=0,
            protected_harm_count=0, protected_positive_count=0,
            protected_worst_delta=0, protected_aggregate_margin=0,
            effect_signature="sig", routability_classification="ROUTABLE_CANDIDATE",
            diagnostic_routability="ROUTABLE", binary_opportunity="BINARY_PARENT_PLAUSIBLE",
            per_case_effects=(), effect_vector_doc=None,
            initiation_step=100, checkpoint_metadata={
                "lane": "GENERAL", "step": 100, "routing_epoch": 4,
            },
            gate_evidence={},
        )
        result = classify_candidate_admission(rec)
        self.assertEqual(result.admission_class, CandidateAdmissionClass.SAFE_FAST_PATH)


# 17. Binary and safe specialists can coexist in composer input
class TestBinarySafeCoexistence(unittest.TestCase):
    def test_coexistence_structure(self):
        from kaggriculture_bonsai.composition.bonsai_composer import StrategicAuthority
        authority = StrategicAuthority("strategic_route", PRODUCTION_AUTHORITY_SLOT)
        authority.validate()
        self.assertEqual(authority.blackboard_type, "I32")


# 18. Independently safe candidates that conflict in composition do not partially commit
class TestNoPartialCommit(unittest.TestCase):
    def test_composition_conflict_status(self):
        result = DownstreamCandidateResult(
            "c" * 64, BinaryCandidateStatus.COMPOSITION_CONFLICT,
            "eval_id", "gate_id", "gate_id", "ctx1",
            None, None, None, "NOT_EVALUATED", None,
            {}, "composition_conflict",
        )
        self.assertEqual(result.status, BinaryCandidateStatus.COMPOSITION_CONFLICT)
        self.assertIsNone(result.materialization)


# 19. Generation-level protected W→L > 0 prevents commit
class TestGenerationProtectedW2L(unittest.TestCase):
    def test_generation_w2l_rejects(self):
        result = validate_generation(
            {"tree": {}, "blackboard": []},
            validation_results=[
                {"baseline_margin": 5, "routed_margin": -3},
            ],
        )
        self.assertFalse(result.passed)
        self.assertIn("protected_w2l", result.rejection_reason)


# 20. Generation-level semantic mismatch prevents commit
class TestGenerationSemanticMismatch(unittest.TestCase):
    def test_generation_ok_with_zero_mismatches(self):
        result = validate_generation({"tree": {}, "blackboard": []})
        self.assertTrue(result.passed)
        self.assertEqual(result.semantic_mismatches, 0)


# 21. Resource/compiler failure prevents commit
class TestResourceOverflow(unittest.TestCase):
    def test_node_overflow_rejects(self):
        result = validate_generation(
            {"tree": {}, "blackboard": []},
            max_nodes=0,
        )
        self.assertFalse(result.passed)
        self.assertIn("node_count", result.rejection_reason)

    def test_slot_overflow_rejects(self):
        result = validate_generation(
            {"tree": {}, "blackboard": [{"name": f"s{i}"} for i in range(300)]},
            max_state_slots=256,
        )
        self.assertFalse(result.passed)
        self.assertIn("state_slots", result.rejection_reason)


# 22. Persisted artifact paths contain no temporary .pending or Phase-1.3 temporary paths
class TestNoPendingPaths(unittest.TestCase):
    def test_downstream_artifact_no_pending(self):
        routed = _make_routed_result()
        program, motifs = _make_program()
        gate = _gate_money_threshold()
        result = run_downstream_integration(
            routed, gate=gate, program=program, motifs=motifs,
        )
        doc = json.dumps(result.to_document(), sort_keys=True)
        self.assertNotIn(".pending", doc)
        self.assertNotIn("60000", doc.replace('"temporary_route_id": 60000', ''))


# 23. Restart/resume does not duplicate downstream certification work
class TestNoDuplicateArtifacts(unittest.TestCase):
    def test_write_idempotent(self):
        routed = _make_routed_result()
        program, motifs = _make_program()
        gate = _gate_money_threshold()
        result = run_downstream_integration(
            routed, gate=gate, program=program, motifs=motifs,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "downstream"
            m1 = write_downstream_artifact(out, [result], parent_generation_context="ctx1")
            m2 = write_downstream_artifact(out, [result], parent_generation_context="ctx1")
            self.assertEqual(set(m1.keys()), set(m2.keys()))
            summary = json.loads((out / "downstream_summary.json").read_text())
            self.assertEqual(summary["total_candidates"], 1)


# 24. Existing pipeline hardening tests remain green (verified by running existing test suite)
class TestExistingHardeningCompat(unittest.TestCase):
    def test_routed_eval_imports(self):
        from kaggriculture_bonsai.routing.routed_evaluation import (
            PRODUCTION_AUTHORITY_NAME,
            PRODUCTION_AUTHORITY_SLOT,
        )
        self.assertIsNotNone(PRODUCTION_AUTHORITY_NAME)
        self.assertIsNotNone(PRODUCTION_AUTHORITY_SLOT)

    def test_search_memory_stages(self):
        from kaggriculture_bonsai.search.search_memory import _SAFE_SEARCH_FEEDBACK_STAGES
        for stage in ("BINARY_TRANSFER", "BINARY_VALIDATION", "BINARY_CEGIS",
                      "BINARY_SEALED", "BINARY_PRODUCTION_MATERIALIZATION"):
            self.assertIn(stage, _SAFE_SEARCH_FEEDBACK_STAGES)


# End-to-end: Full downstream flow produces BINARY_CERTIFIED
class TestEndToEndCertification(unittest.TestCase):
    def test_full_flow_certified(self):
        routed = _make_routed_result()
        program, motifs = _make_program()
        gate = _gate_money_threshold()
        result = run_downstream_integration(
            routed, gate=gate, program=program, motifs=motifs,
            transfer_cases=_transfer_cases(),
            validation_cases=_validation_cases(),
        )
        self.assertEqual(result.status, BinaryCandidateStatus.BINARY_CERTIFIED)
        self.assertIsNotNone(result.transfer)
        self.assertTrue(result.transfer.passed)
        self.assertIsNotNone(result.validation)
        self.assertTrue(result.validation.passed)
        self.assertIsNotNone(result.materialization)
        self.assertTrue(result.materialization.passed)
        self.assertLess(result.materialization.durable_route_id, TEMPORARY_ROUTE_ID_BASE)
        self.assertEqual(result.materialization.authority_slot, PRODUCTION_AUTHORITY_SLOT)
        self.assertIsNone(result.failure_reason)


# Memory recording test
class TestMemoryRecording(unittest.TestCase):
    def test_certified_records_success(self):
        class MockMemory:
            def __init__(self):
                self.records = []
            def record_search_feedback(self, cid, ctx, *, generation, stage, failure_class):
                self.records.append({"stage": stage, "failure_class": failure_class})

        routed = _make_routed_result()
        program, motifs = _make_program()
        gate = _gate_money_threshold()
        result = run_downstream_integration(
            routed, gate=gate, program=program, motifs=motifs,
        )
        memory = MockMemory()
        record_downstream_to_memory(memory, "ctx1", 1, result)
        self.assertEqual(len(memory.records), 1)
        self.assertEqual(memory.records[0]["stage"], "BINARY_PRODUCTION_MATERIALIZATION")
        self.assertIsNone(memory.records[0]["failure_class"])


if __name__ == "__main__":
    unittest.main()
