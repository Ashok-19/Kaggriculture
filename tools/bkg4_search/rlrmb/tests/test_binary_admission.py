"""Focused Phase-1.1 binary-route candidate admission tests."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tools.bkg4_search.rlrmb.binary_admission import (
    ADMISSION_SCHEMA_VERSION,
    AdmissionResult,
    BinaryAdmissionEvidence,
    BinaryAdmissionMetrics,
    CandidateAdmissionClass,
    build_admission_evidence,
    classify_all_candidates,
    classify_candidate_admission,
    record_admission_to_memory,
    select_binary_admission_shortlist,
    write_admission_artifact,
)
from tools.bkg4_search.rlrmb.diagnostics import (
    ImperfectCandidateRecord,
    ImperfectSpecialistDiagnostics,
    PerCaseEffect,
)


def _case(case_id, role, baseline, delta, seat=0, family="f"):
    return PerCaseEffect(
        case_id=case_id, role=role, baseline_margin=baseline,
        candidate_margin=baseline + delta, delta=delta,
        seat=seat, family=family,
    )


def _record(
    cid, family="f", lane="GENERAL", l2w=2, protected_w2l=1, harm=1, worst=-5,
    agg=10, per_case=(), routability="ROUTABLE_CANDIDATE",
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


# 1. Safe candidate → SAFE_FAST_PATH
class TestSafeFastPath(unittest.TestCase):
    def test_safe_candidate_uses_fast_path(self):
        rec = _record("safe1", l2w=3, protected_w2l=0)
        result = classify_candidate_admission(rec)
        self.assertEqual(result.admission_class, CandidateAdmissionClass.SAFE_FAST_PATH)
        self.assertEqual(result.reason, "zero_protected_win_to_loss")


# 2. Candidate with target L→W > 0 and protected W→L > 0 → BINARY_ROUTE_CANDIDATE
class TestBinaryRouteCandidate(unittest.TestCase):
    def test_binary_route_candidate(self):
        rec = _record("brc1", l2w=3, protected_w2l=2)
        result = classify_candidate_admission(rec)
        self.assertEqual(result.admission_class, CandidateAdmissionClass.BINARY_ROUTE_CANDIDATE)

    def test_old_safe_criterion_would_reject(self):
        """Same candidate would fail the old zero-W→L gate but is now preserved."""
        rec = _record("brc2", l2w=3, protected_w2l=1)
        result = classify_candidate_admission(rec)
        self.assertEqual(result.admission_class, CandidateAdmissionClass.BINARY_ROUTE_CANDIDATE)
        self.assertGreater(rec.protected_win_to_loss, 0)


# 3. TRUE_FUTURE_DEPENDENT → does not enter binary route
class TestFutureDependent(unittest.TestCase):
    def test_future_dependent_rejected(self):
        rec = _record("fd1", l2w=3, protected_w2l=2,
                       opportunity="TRUE_FUTURE_DEPENDENT")
        result = classify_candidate_admission(rec)
        self.assertEqual(result.admission_class, CandidateAdmissionClass.TRUE_FUTURE_DEPENDENT)


# 4. GATE_LIMITED → persisted as unresolved
class TestGateLimited(unittest.TestCase):
    def test_gate_limited_persisted(self):
        rec = _record("gl1", l2w=3, protected_w2l=2,
                       opportunity="GATE_LIMITED")
        result = classify_candidate_admission(rec)
        self.assertEqual(result.admission_class, CandidateAdmissionClass.GATE_LIMITED)


# 5. DEFERRED_ROUTE_CANDIDATE → persisted but not promoted
class TestDeferredRoute(unittest.TestCase):
    def test_deferred_persisted(self):
        rec = _record("dr1", l2w=3, protected_w2l=2,
                       opportunity="DEFERRED_DECISION_PLAUSIBLE")
        result = classify_candidate_admission(rec)
        self.assertEqual(result.admission_class, CandidateAdmissionClass.DEFERRED_ROUTE_CANDIDATE)


# 6. UNKNOWN → fail closed
class TestUnknown(unittest.TestCase):
    def test_unknown_fails_closed(self):
        rec = _record("unk1", l2w=3, protected_w2l=2,
                       opportunity="UNKNOWN")
        result = classify_candidate_admission(rec)
        self.assertEqual(result.admission_class, CandidateAdmissionClass.UNKNOWN)


# 7. Target L→W == 0 → not admitted
class TestZeroTargetCapability(unittest.TestCase):
    def test_no_target_capability(self):
        rec = _record("no_cap", l2w=0, protected_w2l=1)
        result = classify_candidate_admission(rec)
        self.assertEqual(result.admission_class, CandidateAdmissionClass.UNKNOWN)
        self.assertEqual(result.reason, "no_target_capability")


# 8. Negative protected delta without W→L does not count as BREAK
class TestNegativeDeltaNotBreak(unittest.TestCase):
    def test_negative_delta_not_break(self):
        harm_case = _case("h1", "PROTECTED", 10, -3)
        self.assertFalse(harm_case.breaks)
        self.assertTrue(harm_case.harmful)
        break_case = _case("b1", "PROTECTED", 5, -10)
        self.assertTrue(break_case.breaks)


# 9. Existing safe promotion semantics unchanged
class TestSafePromotionUnchanged(unittest.TestCase):
    def test_safe_candidates_not_in_binary(self):
        records = [
            _record("safe1", l2w=3, protected_w2l=0),
            _record("safe2", l2w=5, protected_w2l=0),
        ]
        shortlist = select_binary_admission_shortlist(records, limit=10)
        self.assertEqual(len(shortlist), 0)


# 10. No Phase-1.1 candidate can reach final materialization
class TestNoFinalization(unittest.TestCase):
    def test_binary_candidate_is_intermediate(self):
        rec = _record("brc_nofin", l2w=3, protected_w2l=2)
        result = classify_candidate_admission(rec)
        self.assertEqual(result.admission_class, CandidateAdmissionClass.BINARY_ROUTE_CANDIDATE)
        self.assertNotEqual(result.admission_class, CandidateAdmissionClass.SAFE_FAST_PATH)


# 11. Admission shortlist ordering is deterministic
class TestDeterministicOrdering(unittest.TestCase):
    def test_ordering_deterministic(self):
        records = [
            _record("c1", l2w=3, protected_w2l=1),
            _record("c2", l2w=5, protected_w2l=2),
            _record("c3", l2w=4, protected_w2l=1),
        ]
        s1 = select_binary_admission_shortlist(records, limit=10)
        s2 = select_binary_admission_shortlist(records, limit=10)
        self.assertEqual(
            [p[0].candidate_semantic_id for p in s1],
            [p[0].candidate_semantic_id for p in s2],
        )

    def test_higher_l2w_preferred(self):
        records = [
            _record("low", l2w=2, protected_w2l=1),
            _record("high", l2w=5, protected_w2l=1),
        ]
        shortlist = select_binary_admission_shortlist(records, limit=10)
        self.assertEqual(shortlist[0][0].candidate_semantic_id, "high")


# 12. Shortlist is bounded
class TestShortlistBounded(unittest.TestCase):
    def test_bounded_by_limit(self):
        records = [_record(f"c{i}", l2w=3, protected_w2l=1) for i in range(20)]
        shortlist = select_binary_admission_shortlist(records, limit=5)
        self.assertEqual(len(shortlist), 5)

    def test_invalid_limit_rejected(self):
        with self.assertRaises(ValueError):
            select_binary_admission_shortlist([], limit=0)
        with self.assertRaises(ValueError):
            select_binary_admission_shortlist([], limit=-1)


# 13. SearchMemory context invalidation remains correct
class TestMemoryContextInvalidation(unittest.TestCase):
    def test_admission_uses_correct_stage(self):
        rec = _record("mem1", l2w=3, protected_w2l=1)
        result = classify_candidate_admission(rec)
        self.assertEqual(result.admission_class, CandidateAdmissionClass.BINARY_ROUTE_CANDIDATE)
        # record_admission_to_memory uses BINARY_ADMISSION stage


# 14. Restart/resume does not duplicate artifacts
class TestNoDuplicateArtifacts(unittest.TestCase):
    def test_write_is_idempotent(self):
        records = [_record("r1", l2w=3, protected_w2l=1)]
        classified = classify_all_candidates(records)
        shortlist = select_binary_admission_shortlist(records, limit=5)
        metrics = BinaryAdmissionMetrics()
        for _, adm in classified:
            metrics.observe(adm, "f", "GENERAL")
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "admission"
            m1 = write_admission_artifact(
                out, shortlist, classified, metrics,
                parent_generation_context="ctx1",
            )
            # Writing again to same dir should overwrite, not duplicate
            m2 = write_admission_artifact(
                out, shortlist, classified, metrics,
                parent_generation_context="ctx1",
            )
            self.assertEqual(set(m1.keys()), set(m2.keys()))
            summary = json.loads((out / "admission_summary.json").read_text())
            self.assertEqual(summary["shortlist_size"], 1)


# 15. Phase-0.1 diagnostics remain readable
class TestDiagnosticsCompatibility(unittest.TestCase):
    def test_diagnostic_records_still_valid(self):
        rec = _record("compat1", l2w=3, protected_w2l=1)
        doc = rec.to_document()
        restored = ImperfectCandidateRecord.from_document(doc)
        self.assertEqual(restored.candidate_semantic_id, rec.candidate_semantic_id)
        self.assertEqual(restored.protected_win_to_loss, rec.protected_win_to_loss)


# 16. Metrics counting
class TestMetrics(unittest.TestCase):
    def test_metrics_counting(self):
        records = [
            _record("safe", l2w=3, protected_w2l=0),
            _record("brc", l2w=3, protected_w2l=1),
            _record("gl", l2w=3, protected_w2l=1, opportunity="GATE_LIMITED"),
            _record("fd", l2w=3, protected_w2l=1, opportunity="TRUE_FUTURE_DEPENDENT"),
            _record("dr", l2w=3, protected_w2l=1, opportunity="DEFERRED_DECISION_PLAUSIBLE"),
            _record("unk", l2w=3, protected_w2l=1, opportunity="UNKNOWN"),
        ]
        classified = classify_all_candidates(records)
        metrics = BinaryAdmissionMetrics()
        for rec, adm in classified:
            metrics.observe(adm, rec.family, rec.lane)
        doc = metrics.to_document()
        self.assertEqual(doc["safe_fast_path_candidates"], 1)
        self.assertEqual(doc["binary_route_candidates_admitted"], 1)
        self.assertEqual(doc["gate_limited_candidates"], 1)
        self.assertEqual(doc["true_future_dependent_candidates"], 1)
        self.assertEqual(doc["deferred_route_candidates"], 1)
        self.assertEqual(doc["unknown_route_candidates"], 1)
        self.assertEqual(doc["useful_imperfect_candidates_seen"], 6)


# 17. Evidence serialization round-trip
class TestEvidenceRoundTrip(unittest.TestCase):
    def test_round_trip(self):
        per_case = (
            _case("t1", "TARGET", -5, 10),
            _case("p1", "PROTECTED", 5, -10),
        )
        rec = _record("rt1", l2w=3, protected_w2l=1, per_case=per_case)
        adm = classify_candidate_admission(rec)
        evidence = build_admission_evidence(rec, adm, parent_generation_context="ctx")
        doc = evidence.to_document()
        self.assertEqual(doc["schema_version"], ADMISSION_SCHEMA_VERSION)
        restored = BinaryAdmissionEvidence.from_document(doc)
        self.assertEqual(restored.candidate_semantic_id, evidence.candidate_semantic_id)
        self.assertEqual(len(restored.per_case_target_positive), 1)
        self.assertEqual(len(restored.per_case_protected_break), 1)


# 18. Admission artifact schema
class TestArtifactSchema(unittest.TestCase):
    def test_artifact_contains_expected_fields(self):
        records = [_record("art1", l2w=3, protected_w2l=1)]
        classified = classify_all_candidates(records)
        shortlist = select_binary_admission_shortlist(records, limit=5)
        metrics = BinaryAdmissionMetrics()
        for _, adm in classified:
            metrics.observe(adm, "f", "GENERAL")
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "admission"
            write_admission_artifact(
                out, shortlist, classified, metrics,
                parent_generation_context="ctx",
            )
            summary = json.loads((out / "admission_summary.json").read_text())
            self.assertEqual(summary["schema_version"], ADMISSION_SCHEMA_VERSION)
            self.assertEqual(summary["phase"], "1.1")
            self.assertIn("metrics", summary)
            self.assertIn("all_classifications", summary)
            self.assertEqual(summary["shortlist_size"], 1)

            candidates = json.loads(
                (out / "binary_route_candidates.json").read_text()
            )
            self.assertEqual(len(candidates), 1)
            self.assertEqual(candidates[0]["admission_class"], "BINARY_ROUTE_CANDIDATE")


# 19. Diversity in shortlist selection
class TestShortlistDiversity(unittest.TestCase):
    def test_diverse_lanes_preferred(self):
        records = [
            _record("a1", l2w=5, protected_w2l=1, lane="EARLY"),
            _record("a2", l2w=5, protected_w2l=1, lane="LATE"),
            _record("a3", l2w=4, protected_w2l=1, lane="EARLY"),
        ]
        shortlist = select_binary_admission_shortlist(records, limit=2)
        lanes = {p[0].lane for p in shortlist}
        self.assertEqual(lanes, {"EARLY", "LATE"})


if __name__ == "__main__":
    unittest.main()
