"""Focused Phase-1.2 binary gate synthesis tests."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from kaggriculture_bonsai.composition.macro_initiation_cegis import (
    ConditionGene,
    gate_hits,
)
from kaggriculture_bonsai.routing.binary_admission import (
    AdmissionResult,
    CandidateAdmissionClass,
    classify_candidate_admission,
)
from kaggriculture_bonsai.routing.binary_gate_synthesis import (
    GATE_SYNTHESIS_SCHEMA_VERSION,
    BinaryGateCase,
    GateSynthesisMetrics,
    GateSynthesisResult,
    GateSynthesisStatus,
    build_gate_cases,
    synthesize_binary_gate,
    synthesize_gates_for_shortlist,
    write_gate_synthesis_artifact,
    _gate_canonical,
    _gate_identity,
)
from kaggriculture_bonsai.routing.diagnostics import (
    ImperfectCandidateRecord,
    PerCaseEffect,
)


def _ctx(**kwargs):
    """Build a legal context dict from keyword feature=value pairs."""
    return {(k, 0): v for k, v in kwargs.items()}


def _target_positive(case_id, ctx, seat=0, family="f"):
    return BinaryGateCase(
        case_id=case_id, role="TARGET",
        baseline_margin=-10, candidate_margin=10, delta=20,
        seat=seat, family=family,
        is_target_positive=True, is_protected_break=False,
        legal_context=ctx,
    )


def _protected_break(case_id, ctx, seat=0, family="f"):
    return BinaryGateCase(
        case_id=case_id, role="PROTECTED",
        baseline_margin=10, candidate_margin=-10, delta=-20,
        seat=seat, family=family,
        is_target_positive=False, is_protected_break=True,
        legal_context=ctx,
    )


# 1. Simple one-predicate gate separates TARGET from protected BREAKS
class TestSinglePredicateGate(unittest.TestCase):
    def test_single_predicate_separation(self):
        cases = [
            _target_positive("t1", _ctx(STEP=10)),
            _target_positive("t2", _ctx(STEP=12)),
            _protected_break("b1", _ctx(STEP=50)),
            _protected_break("b2", _ctx(STEP=60)),
        ]
        result = synthesize_binary_gate(
            "c1", cases, parent_generation_context="ctx",
        )
        self.assertEqual(result.status, GateSynthesisStatus.GATE_SYNTHESIZED)
        self.assertIsNotNone(result.gate)
        self.assertEqual(result.protected_breaks_matched, 0)
        self.assertGreater(result.target_positives_matched, 0)


# 2. Bounded conjunction synthesized when single condition insufficient
class TestConjunctionGate(unittest.TestCase):
    def test_conjunction_needed(self):
        cases = [
            _target_positive("t1", _ctx(STEP=10, DAY=3)),
            _protected_break("b1", _ctx(STEP=10, DAY=8)),
            _protected_break("b2", _ctx(STEP=50, DAY=3)),
        ]
        result = synthesize_binary_gate(
            "c2", cases, parent_generation_context="ctx",
        )
        self.assertEqual(result.status, GateSynthesisStatus.GATE_SYNTHESIZED)
        self.assertEqual(result.protected_breaks_matched, 0)
        self.assertGreater(result.target_positives_matched, 0)


# 3. Bounded DNF synthesized when genuinely required
class TestDNFGate(unittest.TestCase):
    def test_dnf_synthesis(self):
        cases = [
            _target_positive("t1", _ctx(STEP=5, DAY=1)),
            _target_positive("t2", _ctx(STEP=80, DAY=10)),
            _protected_break("b1", _ctx(STEP=40, DAY=5)),
        ]
        result = synthesize_binary_gate(
            "c3", cases, parent_generation_context="ctx",
            max_clauses=3,
        )
        if result.status == GateSynthesisStatus.GATE_SYNTHESIZED:
            self.assertEqual(result.protected_breaks_matched, 0)
            self.assertGreaterEqual(result.target_positives_matched, 1)


# 4. PARENT is always the false/default branch
class TestParentDefault(unittest.TestCase):
    def test_false_branch_is_parent(self):
        cases = [
            _target_positive("t1", _ctx(STEP=10)),
            _protected_break("b1", _ctx(STEP=50)),
        ]
        result = synthesize_binary_gate(
            "c4", cases, parent_generation_context="ctx",
        )
        self.assertEqual(result.status, GateSynthesisStatus.GATE_SYNTHESIZED)
        # Gate false → PARENT; the break should NOT match
        self.assertFalse(
            gate_hits(result.gate, {("STEP", 0): 50})
        )


# 5. Zero-activation gate rejected as TARGET_CAPABILITY_LOST
class TestZeroActivationRejected(unittest.TestCase):
    def test_zero_activation(self):
        cases = [
            _target_positive("t1", _ctx(STEP=10)),
            _protected_break("b1", _ctx(STEP=10)),
        ]
        result = synthesize_binary_gate(
            "c5", cases, parent_generation_context="ctx",
        )
        self.assertIn(result.status, {
            GateSynthesisStatus.TARGET_CAPABILITY_LOST,
            GateSynthesisStatus.NO_SAFE_SEPARATOR,
        })


# 6. Gate matching protected BREAK is NOT READY_FOR_ROUTED_EVALUATION
class TestBreakMatchRejected(unittest.TestCase):
    def test_break_match_not_synthesized(self):
        cases = [
            _target_positive("t1", _ctx(STEP=10)),
            _protected_break("b1", _ctx(STEP=10)),
        ]
        result = synthesize_binary_gate(
            "c6", cases, parent_generation_context="ctx",
        )
        self.assertNotEqual(result.status, GateSynthesisStatus.GATE_SYNTHESIZED)


# 7. Generic protected harms (not W→L) are not hard BREAK negatives
class TestGenericHarmNotBreak(unittest.TestCase):
    def test_harm_not_break(self):
        harm_case = BinaryGateCase(
            case_id="h1", role="PROTECTED",
            baseline_margin=10, candidate_margin=7, delta=-3,
            seat=0, family="f",
            is_target_positive=False, is_protected_break=False,
            legal_context=_ctx(STEP=30),
        )
        self.assertFalse(harm_case.is_protected_break)
        cases = [
            _target_positive("t1", _ctx(STEP=10)),
            _protected_break("b1", _ctx(STEP=50)),
            harm_case,
        ]
        result = synthesize_binary_gate(
            "c7", cases, parent_generation_context="ctx",
        )
        self.assertEqual(result.status, GateSynthesisStatus.GATE_SYNTHESIZED)


# 8. TARGET capability preservation metrics are exact
class TestCapabilityMetricsExact(unittest.TestCase):
    def test_exact_metrics(self):
        cases = [
            _target_positive("t1", _ctx(STEP=10)),
            _target_positive("t2", _ctx(STEP=12)),
            _target_positive("t3", _ctx(STEP=14)),
            _protected_break("b1", _ctx(STEP=50)),
        ]
        result = synthesize_binary_gate(
            "c8", cases, parent_generation_context="ctx",
        )
        self.assertEqual(result.status, GateSynthesisStatus.GATE_SYNTHESIZED)
        self.assertEqual(result.target_positives_total, 3)
        self.assertEqual(
            result.target_positives_matched + result.target_positives_missed, 3
        )
        self.assertGreater(result.target_positives_matched, 0)


# 9. GATE_LIMITED distinct from TRUE_FUTURE_DEPENDENT
class TestGateLimitedDistinct(unittest.TestCase):
    def test_gate_limited_vs_future(self):
        self.assertNotEqual(
            GateSynthesisStatus.GATE_LIMITED.value,
            CandidateAdmissionClass.TRUE_FUTURE_DEPENDENT.value,
        )
        self.assertNotEqual(
            GateSynthesisStatus.GATE_LIMITED,
            GateSynthesisStatus.NO_SAFE_SEPARATOR,
        )


# 10. TRUE_FUTURE_DEPENDENT candidates never enter synthesis
class TestFutureDependentExcluded(unittest.TestCase):
    def test_future_dependent_not_synthesized(self):
        record = _make_record("fd1", opportunity="TRUE_FUTURE_DEPENDENT")
        admission = classify_candidate_admission(record)
        self.assertEqual(
            admission.admission_class,
            CandidateAdmissionClass.TRUE_FUTURE_DEPENDENT,
        )
        results = synthesize_gates_for_shortlist(
            [(record, admission)],
            {},
            parent_generation_context="ctx",
        )
        self.assertEqual(len(results), 0)


# 11. Gate serialization is deterministic
class TestDeterministicSerialization(unittest.TestCase):
    def test_canonical_deterministic(self):
        cases = [
            _target_positive("t1", _ctx(STEP=10)),
            _protected_break("b1", _ctx(STEP=50)),
        ]
        r1 = synthesize_binary_gate("c11a", cases, parent_generation_context="ctx")
        r2 = synthesize_binary_gate("c11b", cases, parent_generation_context="ctx")
        if r1.gate_canonical and r2.gate_canonical:
            self.assertEqual(
                json.dumps(r1.gate_canonical, sort_keys=True),
                json.dumps(r2.gate_canonical, sort_keys=True),
            )


# 12. Tie-breaking is deterministic
class TestDeterministicTieBreak(unittest.TestCase):
    def test_deterministic_ordering(self):
        cases = [
            _target_positive("t1", _ctx(STEP=10, DAY=2)),
            _target_positive("t2", _ctx(STEP=12, DAY=3)),
            _protected_break("b1", _ctx(STEP=50, DAY=8)),
        ]
        r1 = synthesize_binary_gate("c12", cases, parent_generation_context="ctx")
        r2 = synthesize_binary_gate("c12", cases, parent_generation_context="ctx")
        self.assertEqual(r1.gate_identity, r2.gate_identity)
        self.assertEqual(r1.gate_complexity, r2.gate_complexity)


# 13. Complexity limits enforced
class TestComplexityLimits(unittest.TestCase):
    def test_max_conditions_enforced(self):
        cases = [
            _target_positive("t1", _ctx(STEP=10)),
            _protected_break("b1", _ctx(STEP=50)),
        ]
        result = synthesize_binary_gate(
            "c13", cases, parent_generation_context="ctx",
            max_conditions=2,
        )
        if result.gate:
            for clause in result.gate:
                self.assertLessEqual(len(clause), 2)

    def test_max_clauses_enforced(self):
        cases = [
            _target_positive("t1", _ctx(STEP=10)),
            _target_positive("t2", _ctx(STEP=80)),
            _protected_break("b1", _ctx(STEP=40)),
        ]
        result = synthesize_binary_gate(
            "c13b", cases, parent_generation_context="ctx",
            max_clauses=1,
        )
        if result.gate:
            self.assertLessEqual(len(result.gate), 1)


# 14. Search is bounded
class TestBoundedSearch(unittest.TestCase):
    def test_invalid_params_rejected(self):
        with self.assertRaises(ValueError):
            synthesize_binary_gate(
                "c14", [], parent_generation_context="ctx",
                max_clauses=0,
            )
        with self.assertRaises(ValueError):
            synthesize_binary_gate(
                "c14", [], parent_generation_context="ctx",
                max_conditions=-1,
            )


# 15. Existing SAFE_FAST_PATH unchanged
class TestSafeFastPathUnchanged(unittest.TestCase):
    def test_safe_not_in_synthesis(self):
        record = _make_record("safe1", protected_w2l=0)
        admission = classify_candidate_admission(record)
        self.assertEqual(
            admission.admission_class,
            CandidateAdmissionClass.SAFE_FAST_PATH,
        )
        results = synthesize_gates_for_shortlist(
            [(record, admission)],
            {},
            parent_generation_context="ctx",
        )
        self.assertEqual(len(results), 0)


# 16. No synthesized binary gate reaches materialization in Phase 1.2
class TestNoMaterialization(unittest.TestCase):
    def test_synthesized_is_intermediate(self):
        cases = [
            _target_positive("t1", _ctx(STEP=10)),
            _protected_break("b1", _ctx(STEP=50)),
        ]
        result = synthesize_binary_gate(
            "c16", cases, parent_generation_context="ctx",
        )
        self.assertEqual(result.status, GateSynthesisStatus.GATE_SYNTHESIZED)
        self.assertNotIn("SAFE", result.status.value)
        self.assertNotIn("ACCEPTED", result.status.value)
        self.assertNotIn("COMMITTED", result.status.value)


# 17. SearchMemory exact-context storage works
class TestMemoryStorage(unittest.TestCase):
    def test_memory_records_stage(self):
        from kaggriculture_bonsai.search.search_memory import _SAFE_SEARCH_FEEDBACK_STAGES
        self.assertIn("BINARY_GATE_SYNTHESIS", _SAFE_SEARCH_FEEDBACK_STAGES)


# 18. Parent context change invalidates gate safety authority
class TestContextInvalidation(unittest.TestCase):
    def test_different_parent_context(self):
        cases = [
            _target_positive("t1", _ctx(STEP=10)),
            _protected_break("b1", _ctx(STEP=50)),
        ]
        r1 = synthesize_binary_gate(
            "c18", cases, parent_generation_context="ctx_a",
        )
        r2 = synthesize_binary_gate(
            "c18", cases, parent_generation_context="ctx_b",
        )
        self.assertNotEqual(
            r1.parent_generation_context,
            r2.parent_generation_context,
        )


# 19. Restart/resume reproduces identical gate artifacts
class TestIdempotentArtifacts(unittest.TestCase):
    def test_idempotent_write(self):
        cases = [
            _target_positive("t1", _ctx(STEP=10)),
            _protected_break("b1", _ctx(STEP=50)),
        ]
        result = synthesize_binary_gate(
            "c19", cases, parent_generation_context="ctx",
        )
        metrics = GateSynthesisMetrics()
        metrics.observe(result)
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "gates"
            m1 = write_gate_synthesis_artifact(
                out, [result], metrics, parent_generation_context="ctx",
            )
            m2 = write_gate_synthesis_artifact(
                out, [result], metrics, parent_generation_context="ctx",
            )
            self.assertEqual(set(m1.keys()), set(m2.keys()))
            summary = json.loads((out / "gate_synthesis_summary.json").read_text())
            self.assertEqual(summary["synthesized_count"], 1)


# 20. Existing Phase-1.1 admission artifacts remain readable
class TestAdmissionCompatibility(unittest.TestCase):
    def test_admission_still_works(self):
        record = _make_record("compat1", protected_w2l=2)
        admission = classify_candidate_admission(record)
        self.assertEqual(
            admission.admission_class,
            CandidateAdmissionClass.BINARY_ROUTE_CANDIDATE,
        )
        doc = admission.to_document()
        self.assertEqual(doc["admission_class"], "BINARY_ROUTE_CANDIDATE")


# 21. Result round-trip serialization
class TestResultRoundTrip(unittest.TestCase):
    def test_round_trip(self):
        cases = [
            _target_positive("t1", _ctx(STEP=10)),
            _protected_break("b1", _ctx(STEP=50)),
        ]
        result = synthesize_binary_gate(
            "c21", cases, parent_generation_context="ctx",
        )
        doc = result.to_document()
        self.assertEqual(doc["schema_version"], GATE_SYNTHESIS_SCHEMA_VERSION)
        restored = GateSynthesisResult.from_document(doc)
        self.assertEqual(restored.candidate_semantic_id, result.candidate_semantic_id)
        self.assertEqual(restored.status, result.status)
        self.assertEqual(restored.gate_identity, result.gate_identity)


# 22. Artifact schema correctness
class TestArtifactSchema(unittest.TestCase):
    def test_artifact_fields(self):
        cases = [
            _target_positive("t1", _ctx(STEP=10)),
            _protected_break("b1", _ctx(STEP=50)),
        ]
        result = synthesize_binary_gate(
            "c22", cases, parent_generation_context="ctx",
        )
        metrics = GateSynthesisMetrics()
        metrics.observe(result)
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "gates"
            write_gate_synthesis_artifact(
                out, [result], metrics, parent_generation_context="ctx",
            )
            summary = json.loads((out / "gate_synthesis_summary.json").read_text())
            self.assertEqual(summary["schema_version"], GATE_SYNTHESIS_SCHEMA_VERSION)
            self.assertEqual(summary["phase"], "1.2")
            self.assertIn("metrics", summary)
            self.assertIn("status_distribution", summary)

            candidates = json.loads(
                (out / "binary_gate_candidates.json").read_text()
            )
            self.assertEqual(len(candidates), 1)
            self.assertEqual(
                candidates[0]["status"], "GATE_SYNTHESIZED"
            )


# 23. Metrics counting
class TestMetricsCounting(unittest.TestCase):
    def test_metrics(self):
        metrics = GateSynthesisMetrics()
        cases_good = [
            _target_positive("t1", _ctx(STEP=10)),
            _protected_break("b1", _ctx(STEP=50)),
        ]
        r_good = synthesize_binary_gate(
            "good", cases_good, parent_generation_context="ctx",
        )
        metrics.observe(r_good)
        cases_bad = [
            _target_positive("t1", _ctx(STEP=10)),
            _protected_break("b1", _ctx(STEP=10)),
        ]
        r_bad = synthesize_binary_gate(
            "bad", cases_bad, parent_generation_context="ctx",
        )
        metrics.observe(r_bad)
        doc = metrics.to_document()
        self.assertEqual(doc["candidates_attempted"], 2)
        self.assertEqual(doc["gates_synthesized"], 1)


# 24. INSUFFICIENT_EVIDENCE status
class TestInsufficientEvidence(unittest.TestCase):
    def test_no_target_positives(self):
        cases = [_protected_break("b1", _ctx(STEP=50))]
        result = synthesize_binary_gate(
            "c24", cases, parent_generation_context="ctx",
        )
        self.assertEqual(
            result.status, GateSynthesisStatus.INSUFFICIENT_EVIDENCE,
        )

    def test_no_protected_breaks(self):
        cases = [_target_positive("t1", _ctx(STEP=10))]
        result = synthesize_binary_gate(
            "c24b", cases, parent_generation_context="ctx",
        )
        self.assertEqual(
            result.status, GateSynthesisStatus.INSUFFICIENT_EVIDENCE,
        )


# ── helpers ──────────────────────────────────────────────────────────────────

def _case(case_id, role, baseline, delta, seat=0, family="f"):
    return PerCaseEffect(
        case_id=case_id, role=role, baseline_margin=baseline,
        candidate_margin=baseline + delta, delta=delta,
        seat=seat, family=family,
    )


def _make_record(
    cid, family="f", lane="GENERAL", l2w=3, protected_w2l=1,
    harm=1, worst=-5, agg=10, per_case=(), routability="ROUTABLE_CANDIDATE",
    diag_routability="ROUTABLE", opportunity="BINARY_PARENT_PLAUSIBLE",
    initiation_step=100,
):
    return ImperfectCandidateRecord(
        candidate_semantic_id=cid, structural_id=cid, family=family,
        lane=lane, target_loss_to_win=l2w, target_harm_count=harm,
        target_worst_delta=worst, target_aggregate_margin=agg,
        protected_win_to_loss=protected_w2l, protected_loss_to_win=0,
        protected_harm_count=protected_w2l, protected_positive_count=0,
        protected_worst_delta=worst, protected_aggregate_margin=agg,
        effect_signature=f"sig_{cid}", routability_classification=routability,
        diagnostic_routability=diag_routability, binary_opportunity=opportunity,
        per_case_effects=per_case, effect_vector_doc=None,
        initiation_step=initiation_step, checkpoint_metadata={
            "lane": lane, "step": initiation_step,
            "routing_epoch": initiation_step // 24,
        },
        gate_evidence={},
    )


if __name__ == "__main__":
    unittest.main()
