from __future__ import annotations

import unittest
import json
from pathlib import Path
from unittest.mock import patch

from kaggriculture_bonsai.routing.cegis_adapter import (
    ADAPTER_SCHEMA_VERSION,
    AdapterCase,
    AdapterDisposition,
    CegisAdapterInput,
    CegisAdapterResult,
    adapt_case,
    run_hardened_cegis,
)
from kaggriculture_bonsai.routing.routability import (
    RoutabilityClassification,
    RoutabilityResult,
)
from kaggriculture_bonsai.search.schema import DatasetRole


def routing(classification: RoutabilityClassification) -> RoutabilityResult:
    actions = {
        RoutabilityClassification.ROUTABLE_CANDIDATE: "ENTER_HARDENED_CEGIS",
        RoutabilityClassification.REPAIR_LATER_CHECKPOINT: "RESAMPLE_CAUSAL_CHECKPOINT",
        RoutabilityClassification.REPAIR_COMMON_PREFIX: "RETAIN_COMMON_PREFIX",
        RoutabilityClassification.REPAIR_BRANCH_AT_REVEAL: "ADD_BOUNDED_LEGAL_BRANCH",
        RoutabilityClassification.REPAIR_ABORT_RECOVERY: "ADD_BOUNDED_ABORT_RECOVERY",
        RoutabilityClassification.FUTURE_DEPENDENT: "PERMANENT_REJECT",
        RoutabilityClassification.GATE_LIMITATION_SUSPECTED: "REVIEW_GATE_VOCABULARY",
    }
    scores = {
        RoutabilityClassification.ROUTABLE_CANDIDATE: 100,
        RoutabilityClassification.REPAIR_LATER_CHECKPOINT: 55,
        RoutabilityClassification.REPAIR_COMMON_PREFIX: 55,
        RoutabilityClassification.REPAIR_BRANCH_AT_REVEAL: 65,
        RoutabilityClassification.REPAIR_ABORT_RECOVERY: 50,
        RoutabilityClassification.FUTURE_DEPENDENT: 0,
        RoutabilityClassification.GATE_LIMITATION_SUSPECTED: 25,
    }
    return RoutabilityResult(
        "a" * 64, classification, "fixture", (), ("g",), None, None,
        actions[classification], {"acceptance_authority": False}, ("TARGET",), scores[classification],
    )


def case(
    case_id: str,
    role: DatasetRole,
    baseline: int,
    delta: int,
    value: int,
    signature: int,
    *,
    land: int = 2,
    group: str = "g",
    required_positive: bool = False,
    required_l2w: bool = False,
) -> AdapterCase:
    return AdapterCase(
        case_id=case_id,
        role=role,
        baseline_margin=baseline,
        margin_delta=delta,
        legal_context=(("OPP_UNITS", 0, value), ("OPP_LAND", 0, land)),
        exact_signature=(signature,),
        support_group=group,
        family=group,
        seat=0,
        required_target_positive=required_positive,
        required_target_l2w=required_l2w,
        evidence_id=f"e:{case_id}",
    )


def clean_input(*, routability=None, cases=None, reactive=False) -> CegisAdapterInput:
    rows = cases or (
        case("target", DatasetRole.TARGET, -5, 10, 8, 1, required_positive=True, required_l2w=True),
        case("transfer-a", DatasetRole.TRANSFER, -4, 8, 8, 2, group="a"),
        case("transfer-b", DatasetRole.TRANSFER, -3, 6, 8, 3, group="b"),
        case("harm", DatasetRole.TRANSFER, 10, -9, 7, 4, group="h"),
        case("validation", DatasetRole.VALIDATION, 10, -2, 8, 5, land=1, group="v"),
    )
    return CegisAdapterInput(
        candidate_id="a" * 64,
        program_semantic_id="p" * 64,
        program_schema_version=2 if reactive else 1,
        internal_decision_count=1 if reactive else 0,
        cases=tuple(rows),
        routability=routability or routing(RoutabilityClassification.ROUTABLE_CANDIDATE),
        min_transfer_positive=2,
        min_transfer_independent=2,
        lineage=("beam:1",),
    )


class CegisAdapterTests(unittest.TestCase):
    def test_exact_case_conversion_preserves_hardened_fields(self):
        row = case("target", DatasetRole.TARGET, -5, 10, 8, 9, required_positive=True, required_l2w=True)
        converted = adapt_case(row)
        self.assertEqual(converted.case_id, "target")
        self.assertEqual((converted.baseline_margin, converted.macro_margin, converted.delta), (-5, 5, 10))
        self.assertEqual(converted.context, {("OPP_LAND", 0): 2, ("OPP_UNITS", 0): 8})
        self.assertEqual(converted.exact_signature, (9,))
        self.assertEqual(converted.dataset_role, "target_capability")
        self.assertTrue(converted.required_target_positive)
        self.assertTrue(converted.required_target_l2w)

    def test_roles_and_missing_required_evidence_fail_closed(self):
        for role, expected in (
            (DatasetRole.TRANSFER, "transfer"),
            (DatasetRole.VALIDATION, "validation"),
            (DatasetRole.SEALED, "sealed_holdout"),
        ):
            with self.subTest(role=role):
                self.assertEqual(adapt_case(case("x", role, 1, 1, 8, 1)).dataset_role, expected)
        rows = tuple(row for row in clean_input().cases if row.role is not DatasetRole.TARGET)
        with self.assertRaises(ValueError):
            CegisAdapterInput(**{**clean_input().__dict__, "cases": rows})
        with self.assertRaises(ValueError):
            adapt_case(case("x", DatasetRole.CERTIFICATION, 1, 1, 8, 1))

    def test_routable_invokes_real_hardened_cegis_and_accepts(self):
        result = run_hardened_cegis(clean_input())
        self.assertEqual(result.disposition, AdapterDisposition.ENTER_CEGIS)
        self.assertTrue(result.cegis_invoked)
        self.assertTrue(result.accepted, result.to_document())
        self.assertEqual(result.cegis_stop_reason, "SAFE_CONVERGENCE")
        self.assertTrue(result.target_preserved)
        self.assertTrue(result.transfer_support["accepted"])

    def test_future_and_repairs_never_invoke_cegis(self):
        expected = {
            RoutabilityClassification.FUTURE_DEPENDENT: AdapterDisposition.PERMANENT_REJECT,
            RoutabilityClassification.REPAIR_LATER_CHECKPOINT: AdapterDisposition.RETURN_TO_REPAIR,
            RoutabilityClassification.REPAIR_COMMON_PREFIX: AdapterDisposition.RETURN_TO_REPAIR,
            RoutabilityClassification.REPAIR_BRANCH_AT_REVEAL: AdapterDisposition.RETURN_TO_REPAIR,
            RoutabilityClassification.REPAIR_ABORT_RECOVERY: AdapterDisposition.RETURN_TO_REPAIR,
            RoutabilityClassification.GATE_LIMITATION_SUSPECTED: AdapterDisposition.DIAGNOSTIC_ONLY,
        }
        with patch("kaggriculture_bonsai.routing.cegis_adapter._run_cegis", side_effect=AssertionError("called")):
            for classification, disposition in expected.items():
                with self.subTest(classification=classification):
                    result = run_hardened_cegis(clean_input(routability=routing(classification)))
                    self.assertEqual(result.disposition, disposition)
                    self.assertFalse(result.cegis_invoked)

    def test_reactive_program_is_opaque_outer_gate_subject(self):
        result = run_hardened_cegis(clean_input(reactive=True))
        self.assertTrue(result.accepted)
        self.assertEqual(result.program_schema_version, 2)
        self.assertEqual(result.internal_decision_count, 1)
        self.assertEqual(result.diagnostics["cegis_scope"], "OUTER_INITIATION_ONLY")

    def test_exact_mixed_state_and_required_target_loss_reject(self):
        mixed = (
            case("target", DatasetRole.TARGET, -5, 10, 8, 1, required_positive=True, required_l2w=True),
            case("harm", DatasetRole.TRANSFER, 10, -9, 8, 1),
            case("transfer", DatasetRole.TRANSFER, -3, 6, 8, 2),
            case("validation", DatasetRole.VALIDATION, 1, 0, 8, 3),
        )
        result = run_hardened_cegis(clean_input(cases=mixed))
        self.assertFalse(result.accepted)
        self.assertIn(result.rejection_reason, {"MIXED_LEGAL_STATE", "TARGET_CAPABILITY_LOST"})

    def test_independent_transfer_and_validation_counterexamples_remain_active(self):
        nontransfer = tuple(
            row for row in clean_input().cases if row.case_id != "transfer-b"
        )
        result = run_hardened_cegis(CegisAdapterInput(**{
            **clean_input(cases=nontransfer).__dict__, "min_transfer_positive": 2,
        }))
        self.assertFalse(result.accepted)
        self.assertEqual(result.rejection_reason, "NON_TRANSFERABLE")

        result = run_hardened_cegis(clean_input())
        added = [case_id for row in result.cegis_trace for case_id in row.get("newly_added_harms", ())]
        self.assertIn("validation", added)

    def test_threshold_and_sealed_are_downstream_frozen_checks(self):
        sealed = case("sealed", DatasetRole.SEALED, 10, -9, 8, 8)
        result = run_hardened_cegis(CegisAdapterInput(**{
            **clean_input().__dict__, "cases": clean_input().cases + (sealed,),
        }))
        self.assertFalse(result.accepted)
        self.assertEqual(result.rejection_reason, "HOLDOUT_FAILURE")
        self.assertEqual(result.sealed_status, "FAILED")
        self.assertTrue(result.threshold_robustness)
        self.assertNotIn("sealed_holdout", {role for row in result.cegis_trace for role in row["synthesis_dataset_roles"]})

    def test_candidate50_generic_regression_and_historical_compatibility(self):
        candidate50 = (
            case("target", DatasetRole.TARGET, -5, 10, 170, 50, required_positive=True, required_l2w=True),
            case("harm", DatasetRole.TRANSFER, 10, -9, 170, 50),
            case("validation", DatasetRole.VALIDATION, 10, -1, 170, 50),
        )
        result = run_hardened_cegis(clean_input(cases=candidate50))
        self.assertFalse(result.accepted)
        self.assertIn(result.rejection_reason, {"MIXED_LEGAL_STATE", "TARGET_CAPABILITY_LOST"})

        historical = run_hardened_cegis(clean_input())
        self.assertTrue(historical.accepted)
        self.assertGreaterEqual(historical.transfer_support["independent_positive_support"], 2)

    def test_serialization_is_deterministic_and_versioned(self):
        first = run_hardened_cegis(clean_input())
        second = run_hardened_cegis(clean_input())
        self.assertEqual(first.canonical_json(), second.canonical_json())
        self.assertEqual(CegisAdapterResult.from_document(first.to_document()), first)
        document = first.to_document()
        document["schema_version"] = ADAPTER_SCHEMA_VERSION + 1
        with self.assertRaises(ValueError):
            CegisAdapterResult.from_document(document)

if __name__ == "__main__":
    unittest.main()
