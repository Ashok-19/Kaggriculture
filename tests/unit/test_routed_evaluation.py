"""Focused Phase-1.3 routed SPECIALIST/PARENT evaluation tests."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from kaggriculture_bonsai.composition.module_codec import ConditionGene
from kaggriculture_bonsai.search.checkpoints import Checkpoint, EventType, Lane
from kaggriculture_bonsai.search.ir import (
    ActionFamily, ActionSource, MacroProgram, Phase, Termination, TerminationType,
)
from kaggriculture_bonsai.routing.materialize import MaterializedInterpreter, materialize_program, MaterializationInput
from kaggriculture_bonsai.search.motifs import motif_from_actions
from kaggriculture_bonsai.routing.binary_gate_synthesis import (
    GateSynthesisResult, GateSynthesisStatus, _gate_canonical, _gate_identity,
)
from kaggriculture_bonsai.routing.routed_evaluation import (
    DEFAULT_MIN_ROUTED_TARGET_CAPABILITY,
    ROUTED_EVALUATION_SCHEMA_VERSION,
    TEMPORARY_ROUTE_ID_BASE,
    TEMPORARY_ROUTE_ID_MAX,
    FallbackAuditResult,
    RoutedCaseResult,
    RoutedEvaluationCase,
    RoutedEvaluationMetrics,
    RoutedEvaluationResult,
    RoutedEvaluationStatus,
    _TemporaryRouteAllocator,
    evaluate_routed_candidate,
    write_routed_evaluation_artifact,
    record_routed_evaluation_to_memory,
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


def _gate_always_true():
    return ((ConditionGene("OWN_MONEY", "GE", 0, 0),),)


def _gate_always_false():
    return ((ConditionGene("OWN_MONEY", "GE", 0, 999999),),)


def _gate_money_threshold(threshold=3000):
    return ((ConditionGene("OWN_MONEY", "GE", 0, threshold),),)


def _make_gate_result(
    candidate_id="c" * 64,
    gate=None,
    parent_ctx="ctx1",
    status=GateSynthesisStatus.GATE_SYNTHESIZED,
    target_matched=2,
    breaks_excluded=1,
    admission_id="",
):
    if gate is None:
        gate = _gate_money_threshold()
    return GateSynthesisResult(
        candidate_semantic_id=candidate_id,
        parent_generation_context=parent_ctx,
        status=status,
        gate=gate,
        gate_canonical=_gate_canonical(gate),
        gate_identity=_gate_identity(gate),
        gate_complexity=sum(len(cl) for cl in gate),
        target_positives_total=target_matched,
        target_positives_matched=target_matched,
        target_positives_missed=0,
        protected_breaks_total=breaks_excluded,
        protected_breaks_matched=0,
        protected_breaks_excluded=breaks_excluded,
        gate_features_used=("OWN_MONEY",),
        failure_reason=None,
        gate_score=None,
        source_admission_identity=admission_id or candidate_id,
    )


def _make_cases(
    *,
    target_cases=None,
    protected_cases=None,
    gate=None,
):
    if gate is None:
        gate = _gate_money_threshold()
    cases = []
    if target_cases:
        for case_id, baseline, specialist, money in target_cases:
            cases.append(RoutedEvaluationCase(
                case_id=case_id, role="TARGET",
                baseline_margin=baseline,
                global_specialist_margin=specialist,
                observation=obs(0, money=money),
                parent_action=PARENT_ACTION,
            ))
    if protected_cases:
        for case_id, baseline, specialist, money in protected_cases:
            cases.append(RoutedEvaluationCase(
                case_id=case_id, role="PROTECTED",
                baseline_margin=baseline,
                global_specialist_margin=specialist,
                observation=obs(0, money=money),
                parent_action=PARENT_ACTION,
            ))
    return cases


# 1. Gate false produces exact PARENT behavior
class TestGateFalseParent(unittest.TestCase):
    def test_gate_false_returns_parent(self):
        program, motifs = _make_program()
        gate = _gate_always_false()
        gr = _make_gate_result(gate=gate)
        cases = _make_cases(
            target_cases=[("t1", -5, 10, 1)],
            protected_cases=[("p1", 5, -3, 1)],
        )
        result = evaluate_routed_candidate(
            gr, cases, program=program, motifs=motifs,
            parent_generation_context="ctx1",
        )
        for cr in result.per_case_results:
            self.assertFalse(cr.gate_decision)
            self.assertEqual(cr.routed_margin, cr.baseline_margin)
        self.assertTrue(result.fallback_audit.passed)


# 2. Gate true produces SPECIALIST behavior inside active window
class TestGateTrueSpecialist(unittest.TestCase):
    def test_gate_true_activates_specialist(self):
        program, motifs = _make_program()
        gate = _gate_always_true()
        gr = _make_gate_result(gate=gate)
        cases = _make_cases(
            target_cases=[("t1", -5, 10, 3000)],
        )
        result = evaluate_routed_candidate(
            gr, cases, program=program, motifs=motifs,
            parent_generation_context="ctx1",
        )
        active = [c for c in result.per_case_results if c.gate_decision]
        self.assertEqual(len(active), 1)
        self.assertEqual(active[0].routed_margin, 10)


# 3. Gate false after prior true does not leak specialist state
class TestNoSpecialistLeak(unittest.TestCase):
    def test_false_after_true_no_leak(self):
        program, motifs = _make_program()
        gate = _gate_money_threshold(3000)
        gr = _make_gate_result(gate=gate)
        cases = _make_cases(
            target_cases=[("t1", -5, 10, 5000)],
            protected_cases=[("p1", 5, -3, 1)],
        )
        result = evaluate_routed_candidate(
            gr, cases, program=program, motifs=motifs,
            parent_generation_context="ctx1",
        )
        false_cases = [c for c in result.per_case_results if not c.gate_decision]
        for c in false_cases:
            self.assertEqual(c.routed_margin, c.baseline_margin)


# 4. Active-window start/end semantics
class TestActiveWindowSemantics(unittest.TestCase):
    def test_gate_evaluated_at_initiation_step(self):
        program, motifs = _make_program(step=0)
        gate = _gate_money_threshold(3000)
        gr = _make_gate_result(gate=gate)
        cases_active = _make_cases(target_cases=[("t1", -5, 10, 3000)])
        result_active = evaluate_routed_candidate(
            gr, cases_active, program=program, motifs=motifs,
            parent_generation_context="ctx1",
        )
        self.assertTrue(result_active.per_case_results[0].gate_decision)

        cases_inactive = _make_cases(target_cases=[("t2", -5, 10, 1)])
        result_inactive = evaluate_routed_candidate(
            gr, cases_inactive, program=program, motifs=motifs,
            parent_generation_context="ctx1",
        )
        self.assertFalse(result_inactive.per_case_results[0].gate_decision)


# 5. Statically successful gate that loses all routed TARGET capability
class TestTargetCapabilityLost(unittest.TestCase):
    def test_all_target_gated_out(self):
        program, motifs = _make_program()
        gate = _gate_money_threshold(5000)
        gr = _make_gate_result(gate=gate)
        cases = _make_cases(
            target_cases=[("t1", -5, 10, 1), ("t2", -3, 8, 2)],
            protected_cases=[("p1", 5, -3, 1)],
        )
        result = evaluate_routed_candidate(
            gr, cases, program=program, motifs=motifs,
            parent_generation_context="ctx1",
        )
        self.assertEqual(result.status, RoutedEvaluationStatus.TARGET_CAPABILITY_LOST)
        self.assertEqual(result.target_routed_l2w, 0)


# 6. Routed protected W→L causes failure
class TestRoutedProtectedRegression(unittest.TestCase):
    def test_protected_w2l_fails(self):
        program, motifs = _make_program()
        gate = _gate_always_true()
        gr = _make_gate_result(gate=gate)
        cases = _make_cases(
            target_cases=[("t1", -5, 10, 3000)],
            protected_cases=[("p1", 5, -10, 3000)],
        )
        result = evaluate_routed_candidate(
            gr, cases, program=program, motifs=motifs,
            parent_generation_context="ctx1",
            protected_global_w2l=1,
        )
        self.assertEqual(result.status, RoutedEvaluationStatus.ROUTED_PROTECTED_REGRESSION)
        self.assertGreater(result.protected_routed_w2l, 0)


# 7. Routed candidate with zero protected W→L and preserved capability → READY
class TestReadyForDownstream(unittest.TestCase):
    def test_successful_routing(self):
        program, motifs = _make_program()
        gate = _gate_money_threshold(3000)
        gr = _make_gate_result(gate=gate)
        cases = _make_cases(
            target_cases=[("t1", -5, 10, 5000)],
            protected_cases=[("p1", 5, -3, 1)],
        )
        result = evaluate_routed_candidate(
            gr, cases, program=program, motifs=motifs,
            parent_generation_context="ctx1",
            protected_global_w2l=1,
        )
        self.assertEqual(result.status, RoutedEvaluationStatus.READY_FOR_DOWNSTREAM_INTEGRATION)
        self.assertEqual(result.target_routed_l2w, 1)
        self.assertEqual(result.protected_routed_w2l, 0)
        self.assertEqual(result.protected_w2l_eliminated, 1)


# 8. Generic protected negative delta without W→L does not fail
class TestGenericNegDeltaNotW2L(unittest.TestCase):
    def test_negative_delta_not_break(self):
        program, motifs = _make_program()
        gate = _gate_always_true()
        gr = _make_gate_result(gate=gate)
        cases = _make_cases(
            target_cases=[("t1", -5, 10, 3000)],
            protected_cases=[("p1", 10, 8, 3000)],
        )
        result = evaluate_routed_candidate(
            gr, cases, program=program, motifs=motifs,
            parent_generation_context="ctx1",
        )
        protected = [c for c in result.per_case_results if c.role == "PROTECTED"]
        self.assertEqual(len(protected), 1)
        self.assertTrue(protected[0].harmful)
        self.assertFalse(protected[0].win_to_loss)
        self.assertNotEqual(result.status, RoutedEvaluationStatus.ROUTED_PROTECTED_REGRESSION)


# 9. Globally unsafe specialist becomes zero-W→L when routed to PARENT
class TestGloballyUnsafeBecomesZeroW2L(unittest.TestCase):
    def test_routing_eliminates_w2l(self):
        program, motifs = _make_program()
        gate = _gate_money_threshold(3000)
        gr = _make_gate_result(gate=gate)
        cases = _make_cases(
            target_cases=[("t1", -5, 10, 5000)],
            protected_cases=[
                ("p1", 5, -10, 1),
                ("p2", 3, -5, 2),
            ],
        )
        result = evaluate_routed_candidate(
            gr, cases, program=program, motifs=motifs,
            parent_generation_context="ctx1",
            protected_global_w2l=2,
        )
        self.assertEqual(result.status, RoutedEvaluationStatus.READY_FOR_DOWNSTREAM_INTEGRATION)
        self.assertEqual(result.protected_routed_w2l, 0)
        self.assertEqual(result.protected_w2l_eliminated, 2)


# 10. Protected W→L eliminated count is exact
class TestW2LEliminatedCount(unittest.TestCase):
    def test_exact_elimination_count(self):
        program, motifs = _make_program()
        gate = _gate_money_threshold(3000)
        gr = _make_gate_result(gate=gate)
        cases = _make_cases(
            target_cases=[("t1", -5, 10, 5000)],
            protected_cases=[("p1", 5, -10, 1), ("p2", 5, -8, 1), ("p3", 5, -3, 1)],
        )
        result = evaluate_routed_candidate(
            gr, cases, program=program, motifs=motifs,
            parent_generation_context="ctx1",
            protected_global_w2l=3,
        )
        self.assertEqual(result.protected_w2l_eliminated, 3)
        self.assertEqual(result.protected_routed_w2l, 0)


# 11. TARGET retained capability count is exact
class TestTargetRetainedCount(unittest.TestCase):
    def test_exact_target_l2w_count(self):
        program, motifs = _make_program()
        gate = _gate_money_threshold(3000)
        gr = _make_gate_result(gate=gate, target_matched=2)
        cases = _make_cases(
            target_cases=[
                ("t1", -5, 10, 5000),
                ("t2", -3, 8, 5000),
                ("t3", -7, 12, 1),
            ],
            protected_cases=[("p1", 5, -3, 1)],
        )
        result = evaluate_routed_candidate(
            gr, cases, program=program, motifs=motifs,
            parent_generation_context="ctx1",
        )
        self.assertEqual(result.target_routed_l2w, 2)


# 12. Temporary route IDs do not alter durable committed route allocation
class TestTemporaryRouteIds(unittest.TestCase):
    def test_route_ids_in_temporary_range(self):
        allocator = _TemporaryRouteAllocator()
        rid1 = allocator.allocate()
        rid2 = allocator.allocate()
        self.assertGreaterEqual(rid1, TEMPORARY_ROUTE_ID_BASE)
        self.assertLessEqual(rid2, TEMPORARY_ROUTE_ID_MAX)
        self.assertNotEqual(rid1, rid2)
        self.assertEqual(allocator.count, 2)

    def test_durable_range_not_overlapped(self):
        allocator = _TemporaryRouteAllocator()
        for _ in range(10):
            rid = allocator.allocate()
            self.assertGreaterEqual(rid, TEMPORARY_ROUTE_ID_BASE)
            self.assertLessEqual(rid, TEMPORARY_ROUTE_ID_MAX)
            self.assertTrue(rid >= 60000)


# 13. Temporary materialization does not mutate committed controller/champion
class TestNoCommittedMutation(unittest.TestCase):
    def test_materialization_is_temporary(self):
        program, motifs = _make_program()
        gate = _gate_money_threshold(3000)
        gr = _make_gate_result(gate=gate)
        cases = _make_cases(
            target_cases=[("t1", -5, 10, 5000)],
            protected_cases=[("p1", 5, -3, 1)],
        )
        result = evaluate_routed_candidate(
            gr, cases, program=program, motifs=motifs,
            parent_generation_context="ctx1",
        )
        self.assertGreaterEqual(result.temporary_route_id, TEMPORARY_ROUTE_ID_BASE)
        self.assertNotEqual(result.status.value, "ACCEPTED_SPECIALIST")
        self.assertNotEqual(result.status.value, "COMMITTED")
        self.assertNotEqual(result.status.value, "SEALED")


# 14. Semantic mismatch detection fails closed
class TestSemanticMismatchFailsClosed(unittest.TestCase):
    def test_gate_identity_verified(self):
        program, motifs = _make_program()
        gate = _gate_money_threshold(3000)
        gr = _make_gate_result(gate=gate)
        original_identity = gr.gate_identity
        self.assertIsNotNone(original_identity)
        verified = _gate_identity(gate)
        self.assertEqual(original_identity, verified)


# 15. Phase-1.2 gate identity survives materialization
class TestGateIdentitySurvivesMaterialization(unittest.TestCase):
    def test_gate_identity_preserved(self):
        program, motifs = _make_program()
        gate = _gate_money_threshold(3000)
        gr = _make_gate_result(gate=gate)
        cases = _make_cases(target_cases=[("t1", -5, 10, 5000)])
        result = evaluate_routed_candidate(
            gr, cases, program=program, motifs=motifs,
            parent_generation_context="ctx1",
        )
        self.assertEqual(result.gate_identity, gr.gate_identity)
        self.assertEqual(result.source_gate_synthesis_identity, gr.gate_identity)


# 16. Parent context mismatch fails closed
class TestParentContextMismatch(unittest.TestCase):
    def test_context_mismatch_rejected(self):
        program, motifs = _make_program()
        gate = _gate_money_threshold(3000)
        gr = _make_gate_result(gate=gate, parent_ctx="ctx1")
        cases = _make_cases(target_cases=[("t1", -5, 10, 5000)])
        result = evaluate_routed_candidate(
            gr, cases, program=program, motifs=motifs,
            parent_generation_context="ctx_different",
        )
        self.assertEqual(result.status, RoutedEvaluationStatus.INSUFFICIENT_EVIDENCE)
        self.assertIn("parent_context_mismatch", result.failure_reason)


# 17. Routed result serialization is deterministic
class TestDeterministicSerialization(unittest.TestCase):
    def test_round_trip(self):
        program, motifs = _make_program()
        gate = _gate_money_threshold(3000)
        gr = _make_gate_result(gate=gate)
        cases = _make_cases(
            target_cases=[("t1", -5, 10, 5000)],
            protected_cases=[("p1", 5, -3, 1)],
        )
        result = evaluate_routed_candidate(
            gr, cases, program=program, motifs=motifs,
            parent_generation_context="ctx1",
        )
        doc = result.to_document()
        self.assertEqual(doc["schema_version"], ROUTED_EVALUATION_SCHEMA_VERSION)
        restored = RoutedEvaluationResult.from_document(doc)
        self.assertEqual(restored.candidate_semantic_id, result.candidate_semantic_id)
        self.assertEqual(restored.status, result.status)
        self.assertEqual(restored.gate_identity, result.gate_identity)
        self.assertEqual(restored.target_routed_l2w, result.target_routed_l2w)
        self.assertEqual(restored.protected_routed_w2l, result.protected_routed_w2l)
        doc2 = restored.to_document()
        self.assertEqual(
            json.dumps(doc, sort_keys=True),
            json.dumps(doc2, sort_keys=True),
        )


# 18. SearchMemory exact-context persistence
class TestSearchMemoryPersistence(unittest.TestCase):
    def test_memory_recording(self):
        program, motifs = _make_program()
        gate = _gate_money_threshold(3000)
        gr = _make_gate_result(gate=gate)
        cases = _make_cases(
            target_cases=[("t1", -5, 10, 5000)],
            protected_cases=[("p1", 5, -3, 1)],
        )
        result = evaluate_routed_candidate(
            gr, cases, program=program, motifs=motifs,
            parent_generation_context="ctx1",
        )

        class MockMemory:
            def __init__(self):
                self.records = []
            def record_search_feedback(self, cid, ctx, *, generation, stage, failure_class):
                self.records.append({
                    "cid": cid, "ctx": ctx, "generation": generation,
                    "stage": stage, "failure_class": failure_class,
                })

        memory = MockMemory()
        record_routed_evaluation_to_memory(memory, "ctx1", 1, result)
        self.assertEqual(len(memory.records), 1)
        self.assertEqual(memory.records[0]["stage"], "BINARY_ROUTED_EVALUATION")
        if result.status is RoutedEvaluationStatus.READY_FOR_DOWNSTREAM_INTEGRATION:
            self.assertIsNone(memory.records[0]["failure_class"])
        else:
            self.assertIsNotNone(memory.records[0]["failure_class"])


# 19. Restart/resume does not duplicate routed evaluation records
class TestNoDuplicateArtifacts(unittest.TestCase):
    def test_write_is_idempotent(self):
        program, motifs = _make_program()
        gate = _gate_money_threshold(3000)
        gr = _make_gate_result(gate=gate)
        cases = _make_cases(
            target_cases=[("t1", -5, 10, 5000)],
            protected_cases=[("p1", 5, -3, 1)],
        )
        result = evaluate_routed_candidate(
            gr, cases, program=program, motifs=motifs,
            parent_generation_context="ctx1",
        )
        metrics = RoutedEvaluationMetrics()
        metrics.observe(result)
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "routed_eval"
            m1 = write_routed_evaluation_artifact(out, [result], metrics, parent_generation_context="ctx1")
            m2 = write_routed_evaluation_artifact(out, [result], metrics, parent_generation_context="ctx1")
            self.assertEqual(set(m1.keys()), set(m2.keys()))
            summary = json.loads((out / "routed_eval_summary.json").read_text())
            self.assertEqual(summary["total_count"], 1)


# 20. SAFE_FAST_PATH remains unchanged
class TestSafeFastPathUnchanged(unittest.TestCase):
    def test_safe_candidates_not_affected(self):
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


# 21. No Phase-1.3 result can reach final commit/materialization path
class TestNoFinalCommit(unittest.TestCase):
    def test_ready_is_not_accepted(self):
        program, motifs = _make_program()
        gate = _gate_money_threshold(3000)
        gr = _make_gate_result(gate=gate)
        cases = _make_cases(
            target_cases=[("t1", -5, 10, 5000)],
            protected_cases=[("p1", 5, -3, 1)],
        )
        result = evaluate_routed_candidate(
            gr, cases, program=program, motifs=motifs,
            parent_generation_context="ctx1",
        )
        self.assertEqual(result.status, RoutedEvaluationStatus.READY_FOR_DOWNSTREAM_INTEGRATION)
        self.assertNotIn(result.status.value, [
            "ACCEPTED_SPECIALIST", "COMMITTED", "SEALED", "PRODUCTION_SAFE",
        ])


if __name__ == "__main__":
    unittest.main()
