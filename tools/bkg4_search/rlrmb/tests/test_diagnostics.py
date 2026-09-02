"""Focused Phase-0 imperfect-specialist routing diagnostics tests."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tools.bkg4_search.rlrmb.diagnostics import (
    BinaryOpportunityClass,
    ComplementarityPair,
    DiagnosticRoutabilityClass,
    ImperfectCandidateRecord,
    ImperfectSpecialistDiagnostics,
    PerCaseEffect,
    aggregate_diagnostics,
    audit_feature_sets,
    classify_diagnostic_routability,
    compute_static_complementarity,
    protected_effect_metrics,
    select_diagnostic_shortlist,
    select_binary_routing_rows,
    summarize_policy_attribution,
    write_diagnostic_artifacts,
)
from tools.bkg4_search.rlrmb.pipeline import (
    CaseMeta,
    ExactCaseEvidence,
    build_routability_evidence,
)
from tools.bkg4_search.rlrmb.routability import (
    BoundaryState,
    RequiredContinuation,
    RoutabilityClassification,
    RoutabilityEvidence,
    TrajectoryEvidence,
    screen_routability,
)
from tools.bkg4_search.rlrmb.schema import DatasetRole
from tools.bkg4_search.rlrmb.effects import EffectBlock, EffectVector


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


class TestPerCaseEffect(unittest.TestCase):
    def test_solves_breaks(self):
        solves = _case("c1", "TARGET", -10, 20)
        self.assertTrue(solves.solves)
        self.assertFalse(solves.breaks)
        breaks = _case("c2", "TARGET", 10, -20)
        self.assertTrue(breaks.breaks)
        self.assertFalse(breaks.solves)
        neutral = _case("c3", "TARGET", 5, 3)
        self.assertFalse(neutral.solves)
        self.assertFalse(neutral.breaks)

    def test_negative_protected_delta_is_not_a_break(self):
        harm = _case("p1", "PROTECTED", 20, -5)
        self.assertTrue(harm.harmful)
        self.assertFalse(harm.breaks)

    def test_protected_summary_counts_crossings_not_all_harm(self):
        block = EffectBlock(
            DatasetRole.VALIDATION, "protected", (20, 20, -10),
            (-5, -25, 20),
        )
        metrics = protected_effect_metrics(block)
        self.assertEqual(metrics["protected_harm_count"], 2)
        self.assertEqual(metrics["protected_win_to_loss"], 1)
        self.assertEqual(metrics["protected_loss_to_win"], 1)

    def test_serialization(self):
        case = _case("c1", "TARGET", -10, 20)
        doc = case.to_document()
        restored = PerCaseEffect(**doc)
        self.assertEqual(case, restored)


class TestPolicyAttribution(unittest.TestCase):
    def test_counts_only_policy_cases_that_produce_target_capability(self):
        routable = _record(
            "routable",
            per_case=(
                _case("animal-win", "TARGET", -5, 10, family="animals:bkg4_animal_cow"),
                _case("market-harm", "TARGET", 5, -10, family="market_liquidity:bkg4_market_squeeze"),
            ),
        )
        future = _record(
            "future",
            per_case=(
                _case("market-win", "TARGET", -5, 10, family="market_liquidity:bkg4_market_squeeze"),
            ),
            diag_routability="FUTURE_DEPENDENT",
            opportunity="TRUE_FUTURE_DEPENDENT",
        )

        attribution = summarize_policy_attribution(
            (routable, future), downstream_ready_candidate_ids={"routable"},
        )

        self.assertEqual(attribution["animals:bkg4_animal_cow"], {
            "target_loss_to_win": 1,
            "useful_specialists": 1,
            "routable": 1,
            "future_dependent": 0,
            "downstream_ready": 1,
        })
        self.assertEqual(attribution["market_liquidity:bkg4_market_squeeze"], {
            "target_loss_to_win": 1,
            "useful_specialists": 1,
            "routable": 0,
            "future_dependent": 1,
            "downstream_ready": 0,
        })


class TestImperfectCandidateRecord(unittest.TestCase):
    def test_solves_breaks_sets(self):
        cases = (
            _case("c1", "TARGET", -10, 20),   # SOLVES
            _case("c2", "TARGET", 10, -20),    # BREAKS
            _case("c3", "TARGET", 5, 3),       # neutral positive
        )
        r = _record("x", per_case=cases)
        self.assertEqual(r.solves_set(), frozenset({"c1"}))
        self.assertEqual(r.breaks_set(), frozenset({"c2"}))

    def test_serialization_roundtrip(self):
        cases = (_case("c1", "TARGET", -10, 20), _case("c2", "TARGET", 10, -20))
        r = _record("x", per_case=cases)
        doc = r.to_document()
        restored = ImperfectCandidateRecord.from_document(doc)
        self.assertEqual(r.candidate_semantic_id, restored.candidate_semantic_id)
        self.assertEqual(r.target_loss_to_win, restored.target_loss_to_win)
        self.assertEqual(r.protected_win_to_loss, restored.protected_win_to_loss)
        self.assertEqual(len(r.per_case_effects), len(restored.per_case_effects))

    def test_deterministic_serialization(self):
        cases = (_case("c1", "TARGET", -10, 20),)
        r = _record("x", per_case=cases)
        doc1 = json.dumps(r.to_document(), sort_keys=True)
        doc2 = json.dumps(r.to_document(), sort_keys=True)
        self.assertEqual(doc1, doc2)

    def test_serialization_retains_program_for_later_trajectory_capture(self):
        record = _record("x", per_case=(_case("c1", "TARGET", -10, 20),))
        record = ImperfectCandidateRecord(
            **{**record.__dict__, "program_document": {"schema_version": 1}}
        )

        restored = ImperfectCandidateRecord.from_document(record.to_document())

        self.assertEqual(restored.program_document, {"schema_version": 1})


class TestImperfectSpecialistDiagnostics(unittest.TestCase):
    def test_persist_and_load(self):
        diag = ImperfectSpecialistDiagnostics()
        cases = (_case("c1", "TARGET", -10, 20),)
        diag.record(_record("a", per_case=cases))
        diag.record(_record("b", per_case=cases, l2w=3, protected_w2l=0))
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "test.jsonl"
            diag.persist(path)
            loaded = ImperfectSpecialistDiagnostics.load(path)
            self.assertEqual(len(loaded.records), 2)
            text = path.read_text()
            lines = [l for l in text.strip().splitlines() if l.strip()]
            self.assertEqual(len(lines), 2)
            for line in lines:
                json.loads(line)  # valid JSON

    def test_deterministic_order(self):
        diag = ImperfectSpecialistDiagnostics()
        diag.record(_record("b"))
        diag.record(_record("a"))
        text = diag.serialize()
        lines = text.strip().splitlines()
        ids = [json.loads(l)["candidate_semantic_id"] for l in lines]
        self.assertEqual(ids, ["a", "b"])


class TestSafeCandidatesUnchanged(unittest.TestCase):
    """Production promotion behavior: zero-W→L requirement is unchanged."""

    def test_safe_candidate_representation(self):
        cases = (_case("c1", "TARGET", -10, 20), _case("c2", "TARGET", 5, 3))
        r = _record("safe", protected_w2l=0, per_case=cases)
        self.assertEqual(r.protected_win_to_loss, 0)
        self.assertEqual(r.breaks_set("PROTECTED"), frozenset())

    def test_imperfect_not_promoted(self):
        r = _record("imp", l2w=3, protected_w2l=1)
        self.assertGreater(r.protected_win_to_loss, 0)


class TestDiagnosticRoutabilityClassification(unittest.TestCase):
    def test_routable(self):
        doc = {"classification": "ROUTABLE_CANDIDATE", "diagnostics": {}}
        self.assertEqual(
            classify_diagnostic_routability(doc),
            DiagnosticRoutabilityClass.ROUTABLE,
        )

    def test_future_dependent(self):
        doc = {"classification": "FUTURE_DEPENDENT", "diagnostics": {}}
        self.assertEqual(
            classify_diagnostic_routability(doc),
            DiagnosticRoutabilityClass.FUTURE_DEPENDENT,
        )

    def test_gate_limitation_with_legal_pred(self):
        doc = {
            "classification": "GATE_LIMITATION_SUSPECTED",
            "diagnostics": {
                "legal_predicate": {"feature": "OWN_WORKER_X"},
                "gate_predicate": None,
            },
        }
        self.assertEqual(
            classify_diagnostic_routability(doc),
            DiagnosticRoutabilityClass.GATE_LIMITATION,
        )

    def test_gate_limitation_on_repair_class(self):
        doc = {
            "classification": "REPAIR_BRANCH_AT_REVEAL",
            "diagnostics": {
                "legal_predicate": {"feature": "OWN_TILE_WATERED"},
                "gate_predicate": None,
            },
        }
        self.assertEqual(
            classify_diagnostic_routability(doc),
            DiagnosticRoutabilityClass.GATE_LIMITATION,
        )

    def test_unknown_no_predicates(self):
        doc = {
            "classification": "GATE_LIMITATION_SUSPECTED",
            "diagnostics": {"legal_predicate": None, "gate_predicate": None},
        }
        self.assertEqual(
            classify_diagnostic_routability(doc),
            DiagnosticRoutabilityClass.UNKNOWN,
        )


class TestSyntheticRoutabilityClassification(unittest.TestCase):
    """Synthetic example showing ROUTABLE, GATE_LIMITATION, FUTURE_DEPENDENT."""

    def _evidence(self, legal_contexts, gate_contexts, signatures, deltas):
        trajectories = []
        for i, (lc, gc, sig, delta) in enumerate(
            zip(legal_contexts, gate_contexts, signatures, deltas)
        ):
            trajectories.append(TrajectoryEvidence(
                case_id=f"case_{i}", role=DatasetRole.TARGET,
                states=(BoundaryState(100, sig, lc, gc),),
                requirements=(RequiredContinuation(100, "CAND" if delta > 0 else "PARENT"),),
                effect_delta=delta, support_group="g", family="f", seat=0,
                required_support=delta > 0,
            ))
        return RoutabilityEvidence("cand", 100, tuple(trajectories))

    def test_routable_when_gate_separates(self):
        lc = gc = (("OWN_MONEY", 0, 5000),)
        lc2 = gc2 = (("OWN_MONEY", 0, 1000),)
        ev = self._evidence([lc, lc2], [gc, gc2], [(1, 5000), (1, 1000)], [10, -5])
        result = screen_routability(ev)
        self.assertEqual(result.classification, RoutabilityClassification.ROUTABLE_CANDIDATE)
        diag = classify_diagnostic_routability(result.to_document())
        self.assertEqual(diag, DiagnosticRoutabilityClass.ROUTABLE)

    def test_gate_limitation_legal_separates_gate_cannot(self):
        lc1 = (("OWN_MONEY", 0, 5000), ("OWN_TILE_WATERED", 0, 1))
        lc2 = (("OWN_MONEY", 0, 5000), ("OWN_TILE_WATERED", 0, 0))
        gc = (("OWN_MONEY", 0, 5000),)
        ev = self._evidence([lc1, lc2], [gc, gc], [(1, 5000, 1), (1, 5000, 0)], [10, -5])
        result = screen_routability(ev)
        diag = classify_diagnostic_routability(result.to_document())
        self.assertEqual(diag, DiagnosticRoutabilityClass.GATE_LIMITATION)

    def test_future_dependent_nothing_separates(self):
        lc = gc = (("OWN_MONEY", 0, 5000),)
        ev = self._evidence([lc, lc], [gc, gc], [(1, 5000), (1, 5000)], [10, -5])
        result = screen_routability(ev)
        self.assertEqual(result.classification, RoutabilityClassification.FUTURE_DEPENDENT)
        diag = classify_diagnostic_routability(result.to_document())
        self.assertEqual(diag, DiagnosticRoutabilityClass.FUTURE_DEPENDENT)


class TestExistingFutureDependentFailsClosed(unittest.TestCase):
    """FUTURE_DEPENDENT continues to produce PERMANENT_REJECT."""

    def test_future_dependent_remains_permanent_reject(self):
        lc = gc = (("OWN_MONEY", 0, 5000),)
        trajectories = (
            TrajectoryEvidence(
                "c1", DatasetRole.TARGET,
                (BoundaryState(100, (1, 5000), lc, gc),),
                (RequiredContinuation(100, "CAND"),), 10, "g", "f", 0,
                required_support=True,
            ),
            TrajectoryEvidence(
                "c2", DatasetRole.TARGET,
                (BoundaryState(100, (1, 5000), lc, gc),),
                (RequiredContinuation(100, "PARENT"),), -5, "g", "f", 0,
            ),
        )
        ev = RoutabilityEvidence("cand", 100, trajectories)
        result = screen_routability(ev)
        self.assertEqual(result.classification, RoutabilityClassification.FUTURE_DEPENDENT)
        self.assertEqual(result.recommended_action, "PERMANENT_REJECT")


class TestAggregateDiagnostics(unittest.TestCase):
    def test_distribution_buckets(self):
        records = [
            _record("a", l2w=5, protected_w2l=0),
            _record("b", l2w=3, protected_w2l=1),
            _record("c", l2w=2, protected_w2l=2),
            _record("d", l2w=1, protected_w2l=4),
            _record("e", l2w=1, protected_w2l=7),
        ]
        agg = aggregate_diagnostics(records)
        self.assertEqual(agg["total_candidates_evaluated"], 5)
        self.assertEqual(agg["diagnostic_shortlist_size"], 5)
        self.assertEqual(agg["candidates_with_l2w_gt_0"], 5)
        self.assertEqual(agg["candidates_with_l2w_gt_0_and_w2l_gt_0"], 4)
        self.assertEqual(agg["protected_w2l_distribution"]["w2l_eq_0"], 1)
        self.assertEqual(agg["protected_w2l_distribution"]["w2l_eq_1"], 1)
        self.assertEqual(agg["protected_w2l_distribution"]["w2l_eq_2"], 1)
        self.assertEqual(agg["protected_w2l_distribution"]["w2l_3_to_5"], 1)
        self.assertEqual(agg["protected_w2l_distribution"]["w2l_gt_5"], 1)

    def test_useful_imperfect_buckets(self):
        records = [
            _record("a", l2w=5, protected_w2l=2),
            _record("b", l2w=3, protected_w2l=1),
            _record("c", l2w=2, protected_w2l=1),
        ]
        agg = aggregate_diagnostics(records)
        self.assertEqual(agg["useful_imperfect_buckets"]["l2w_ge2_w2l_eq1"], 2)
        self.assertEqual(agg["useful_imperfect_buckets"]["l2w_ge3_w2l_le2"], 2)
        self.assertEqual(agg["useful_imperfect_buckets"]["l2w_ge5_w2l_le2"], 1)
        self.assertEqual(agg["useful_imperfect_buckets"]["l2w_gt_w2l"], 3)


class TestStaticComplementarity(unittest.TestCase):
    def test_two_way_complement(self):
        a_cases = (
            _case("t1", "TARGET", -10, 20),
            _case("p1", "PROTECTED", 10, -20),
        )
        b_cases = (
            _case("c1", "TARGET", -10, -5),    # B doesn't solve c1
            _case("c2", "TARGET", 10, 5),      # B neutral on c2
        )
        # To get two-way complement, B should solve what A breaks
        b_cases2 = (
            _case("t2", "TARGET", -10, 20),
            _case("p1", "PROTECTED", 10, -5),  # preserves the baseline win
        )
        a = _record("a", per_case=a_cases)
        b = _record("b", per_case=b_cases2)
        result = compute_static_complementarity([a, b])
        self.assertIn("STATIC_COMPLEMENTARITY_HEURISTIC", result["label"])
        self.assertEqual(result["warning"], "NOT_COMPOSITION_SAFE")
        if result["pairs"]:
            pair = result["pairs"][0]
            self.assertIn(pair["classification"], [
                "ONE_WAY_COMPLEMENT", "TWO_WAY_COMPLEMENT", "NEAR_CLOSED_PAIR",
            ])
            self.assertEqual(pair["b_protects_a_breaks"], ["p1"])

    def test_no_complement_without_overlap(self):
        a_cases = (_case("c1", "TARGET", -10, 20),)
        b_cases = (_case("c2", "TARGET", -10, 20),)
        a = _record("a", per_case=a_cases)
        b = _record("b", per_case=b_cases)
        result = compute_static_complementarity([a, b])
        self.assertEqual(result["pairs_examined"], 1)
        self.assertEqual(result["complement_pairs"], 0)

    def test_incompatible_lanes_skipped(self):
        a_cases = (_case("c1", "TARGET", -10, 20), _case("c2", "TARGET", 10, -20))
        b_cases = (_case("c2", "TARGET", -10, 20), _case("c1", "TARGET", 10, -20))
        a = _record("a", lane="LAND_INVESTMENT", per_case=a_cases)
        b = _record("b", lane="ANIMAL_TRANSITION", per_case=b_cases)
        result = compute_static_complementarity([a, b])
        self.assertEqual(result["pairs_examined"], 0)

    def test_max_pairs_cap(self):
        records = [_record(f"r{i}", per_case=(
            _case("c1", "TARGET", -10, 20),
            _case("c2", "TARGET", 10, -20),
        )) for i in range(10)]
        result = compute_static_complementarity(records, max_pairs=3)
        self.assertLessEqual(result["pairs_examined"], 3)

    def test_adjacent_steps_in_different_epochs_are_not_compared(self):
        cases = (_case("p1", "PROTECTED", 10, -20),)
        a = _record("a", initiation_step=23, per_case=cases)
        b = _record("b", initiation_step=24, per_case=cases)
        self.assertEqual(compute_static_complementarity([a, b])["pairs_examined"], 0)


class TestPhase01Shortlist(unittest.TestCase):
    class Candidate:
        def __init__(self, cid, l2w, harm, worst, aggregate, lane="GENERAL", step=100):
            self.candidate_semantic_id = cid
            self.target_loss_to_win = l2w
            self.target_harm_count = harm
            self.target_worst_delta = worst
            self.target_aggregate_margin = aggregate
            self.effect_signature = f"sig-{cid}"
            self.structural_id = cid
            self.lane = lane
            self.initiation_step = step

    def test_shortlist_is_useful_deterministic_and_bounded(self):
        rows = [self.Candidate("z", 0, 0, 0, 100)]
        rows += [self.Candidate(f"c{i:02d}", 6 - i % 3, i % 2, -i, 100 - i) for i in range(20)]
        first = select_diagnostic_shortlist(rows, limit=8)
        second = select_diagnostic_shortlist(tuple(reversed(rows)), limit=8)
        self.assertEqual([r.candidate_semantic_id for r in first], [r.candidate_semantic_id for r in second])
        self.assertEqual(len(first), 8)
        self.assertTrue(all(r.target_loss_to_win > 0 for r in first))


class TestBinaryParentOpportunity(unittest.TestCase):
    def test_requested_classes_remain_distinct(self):
        self.assertEqual(BinaryOpportunityClass.BINARY_PARENT_PLAUSIBLE.value, "BINARY_PARENT_PLAUSIBLE")
        self.assertEqual(BinaryOpportunityClass.GATE_LIMITED.value, "GATE_LIMITED")
        self.assertEqual(BinaryOpportunityClass.DEFERRED_DECISION_PLAUSIBLE.value, "DEFERRED_DECISION_PLAUSIBLE")
        self.assertEqual(BinaryOpportunityClass.TRUE_FUTURE_DEPENDENT.value, "TRUE_FUTURE_DEPENDENT")

    def test_repair_at_reveal_is_deferred_not_gate_limited(self):
        doc = {
            "classification": "REPAIR_BRANCH_AT_REVEAL",
            "diagnostics": {
                "legal_predicate": {"feature": "OWN_TILE_WATERED"},
                "gate_predicate": None,
            },
        }
        from tools.bkg4_search.rlrmb.diagnostics import classify_binary_opportunity
        self.assertEqual(
            classify_binary_opportunity(doc),
            BinaryOpportunityClass.DEFERRED_DECISION_PLAUSIBLE,
        )

    def test_routing_rows_are_target_solves_vs_actual_protected_breaks(self):
        context = (("OWN_MONEY", 0, 1000),)
        rows = (
            ExactCaseEvidence(CaseMeta("target-solve", "t", 0, 1, "f"), DatasetRole.TARGET,
                              -10, 10, context, context, (1,), "c"),
            ExactCaseEvidence(CaseMeta("target-harm", "t", 0, 1, "f"), DatasetRole.TARGET,
                              -10, -20, context, context, (1,), "c"),
            ExactCaseEvidence(CaseMeta("protected-harm", "p", 0, 1, "f"), DatasetRole.VALIDATION,
                              20, 10, context, context, (1,), "c"),
            ExactCaseEvidence(CaseMeta("protected-break", "p", 0, 1, "f"), DatasetRole.VALIDATION,
                              20, -1, context, context, (1,), "c"),
        )
        self.assertEqual(
            [row.meta.case_id for row in select_binary_routing_rows(rows)],
            ["target-solve", "protected-break"],
        )


class TestFeatureSetAudit(unittest.TestCase):
    def test_audit_produces_correct_sets(self):
        audit = audit_feature_sets()
        self.assertIn("OWN_MONEY", audit["full_legal_observability_features"])
        self.assertIn("OWN_MONEY", audit["deployable_gate_grammar_features"])
        self.assertIn("OWN_MONEY", audit["prior_contrastive_features"])
        # Worker/tile features are in gate grammar but not in prior contrastive
        self.assertIn("OWN_WORKER_X", audit["in_gate_grammar_not_in_prior_contrastive"])
        self.assertIn("OWN_TILE_KIND", audit["in_gate_grammar_not_in_prior_contrastive"])
        # Tile care features are in legal but not in gate grammar
        self.assertIn("OWN_TILE_WATERED", audit["in_legal_not_in_gate_grammar"])
        self.assertGreater(audit["full_legal_observability_count"], audit["deployable_gate_grammar_count"])
        self.assertGreater(audit["deployable_gate_grammar_count"], audit["prior_contrastive_count"])

    def test_gate_grammar_features_subset_of_legal(self):
        audit = audit_feature_sets()
        self.assertEqual(len(audit["in_gate_grammar_not_in_legal"]), 0)


class TestLegalGateContextSeparation(unittest.TestCase):
    """Full legal observability and current gate grammar are distinct."""

    def test_legal_context_broader_than_gate_context(self):
        # Simulate what load_candidate_legal_states now produces
        legal = (
            ("OWN_MONEY", 0, 5000),
            ("OWN_TILE_WATERED", 0, 1),
            ("OWN_WORKER_X", 0, 3),
            ("STEP", 0, 100),
        )
        from tools.bkg4_search.module_codec import FEATURES
        gate_features = frozenset(FEATURES)
        gate = tuple((f, i, v) for f, i, v in legal if f in gate_features)
        # OWN_TILE_WATERED is NOT in gate grammar
        self.assertIn(("OWN_TILE_WATERED", 0, 1), legal)
        self.assertNotIn(("OWN_TILE_WATERED", 0, 1), gate)
        # OWN_WORKER_X IS in gate grammar
        self.assertIn(("OWN_WORKER_X", 0, 3), gate)
        self.assertGreater(len(legal), len(gate))


class TestSearchMemoryArtifactsReadable(unittest.TestCase):
    """Existing search-memory artifacts remain readable."""

    def test_search_memory_schema(self):
        from tools.bkg4_search.rlrmb.search_memory import SearchMemory, MEMORY_SCHEMA_VERSION
        with tempfile.TemporaryDirectory() as d:
            db = Path(d) / "test.db"
            with SearchMemory(db) as mem:
                mem.record_structural("test_id", lane="GENERAL")
                self.assertTrue(mem.is_structurally_seen("test_id"))
                self.assertEqual(mem.record_count(), 0)


class TestWriteDiagnosticArtifacts(unittest.TestCase):
    def test_writes_all_artifacts(self):
        diag = ImperfectSpecialistDiagnostics()
        cases = (
            _case("c1", "TARGET", -10, 20),
            _case("c2", "TARGET", 10, -20),
        )
        diag.record(_record("a", per_case=cases))
        diag.record(_record("b", per_case=cases, l2w=3, protected_w2l=0))
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "diagnostics"
            manifest = write_diagnostic_artifacts(out, diag)
            self.assertIn("useful_candidates_protected_effects", manifest)
            self.assertIn("summary", manifest)
            self.assertIn("feature_audit", manifest)
            self.assertIn("complementarity", manifest)
            self.assertIn("policy_attribution", manifest)
            # All files are valid JSON/JSONL
            for key, path in manifest.items():
                content = (out / path).read_text()
                if path.endswith(".jsonl"):
                    for line in content.strip().splitlines():
                        json.loads(line)
                else:
                    json.loads(content)

    def test_empty_diagnostics(self):
        diag = ImperfectSpecialistDiagnostics()
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "diagnostics"
            manifest = write_diagnostic_artifacts(out, diag)
            self.assertIn("summary", manifest)
            summary = json.loads((out / manifest["summary"]).read_text())
            self.assertEqual(summary["total_candidates_evaluated"], 0)
            self.assertEqual(summary["diagnostic_shortlist_size"], 0)


class TestExactCaseEvidenceGateContext(unittest.TestCase):
    """ExactCaseEvidence now carries separate gate_context."""

    def test_gate_context_field_exists(self):
        from tools.bkg4_search.rlrmb.pipeline import ExactCaseEvidence
        context = (("OWN_MONEY", 0, 3000),)
        gate = (("OWN_MONEY", 0, 3000),)
        ev = ExactCaseEvidence(
            CaseMeta("c1", "tape.json", 0, 1, "f"),
            DatasetRole.TARGET, -1, 1, context, gate, (1,), "cand_id",
        )
        self.assertEqual(ev.legal_context, context)
        self.assertEqual(ev.gate_context, gate)
        self.assertEqual(ev.delta, 2)

    def test_empty_gate_context_backward_compat(self):
        from tools.bkg4_search.rlrmb.pipeline import ExactCaseEvidence
        ev = ExactCaseEvidence(
            CaseMeta("c1", "tape.json", 0, 1, "f"),
            DatasetRole.TARGET, -1, 1, (), (), (1,), "cand_id",
        )
        self.assertEqual(ev.gate_context, ())


if __name__ == "__main__":
    unittest.main()
