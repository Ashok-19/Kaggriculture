from __future__ import annotations

import json
import subprocess
import sys
import unittest

from tools.bkg4_search.rlrmb.failures import (
    FAILURE_ACTIONS,
    FailureClass,
    RepairAction,
    SearchFailure,
)
from tools.bkg4_search.rlrmb.schema import (
    RLRMB_SCHEMA_VERSION,
    DatasetRole,
    canonical_json,
    stable_identity,
    validate_runtime_identity_safe,
    validate_schema_version,
)


class ImportIsolationTests(unittest.TestCase):
    def test_package_import_is_isolated(self) -> None:
        script = """
import json
import sys
import tools.bkg4_search.rlrmb
forbidden = [
    name for name in sys.modules
    if "bonsai_editformer" in name
    or "distill_route" in name
    or name.startswith("torch")
    or name.startswith("cupy")
]
print(json.dumps(forbidden))
"""
        result = subprocess.run(
            [sys.executable, "-c", script],
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertEqual(json.loads(result.stdout), [])


class SchemaTests(unittest.TestCase):
    def test_canonical_json_is_deterministic(self) -> None:
        first = {"z": [{"b": 2, "a": "caf\u00e9"}], "a": {"y": 1, "x": 0}}
        second = {"a": {"x": 0, "y": 1}, "z": [{"a": "caf\u00e9", "b": 2}]}
        self.assertEqual(canonical_json(first), canonical_json(second))
        self.assertIn("caf\u00e9", canonical_json(first))
        self.assertNotIn(" ", canonical_json(first))

    def test_stable_identity_is_semantic_and_namespace_separated(self) -> None:
        first = stable_identity("candidate", {"b": 2, "a": 1}, [3, 4])
        reordered = stable_identity("candidate", {"a": 1, "b": 2}, [3, 4])
        changed = stable_identity("candidate", {"a": 1, "b": 9}, [3, 4])
        other_namespace = stable_identity("run", {"a": 1, "b": 2}, [3, 4])
        self.assertEqual(first, reordered)
        self.assertNotEqual(first, changed)
        self.assertNotEqual(first, other_namespace)

    def test_dataset_roles_round_trip_and_unknown_rejects(self) -> None:
        expected = {"TARGET", "TRANSFER", "VALIDATION", "SEALED", "CERTIFICATION"}
        self.assertEqual({role.value for role in DatasetRole}, expected)
        for role in DatasetRole:
            self.assertIs(DatasetRole.parse(role.value), role)
        with self.assertRaises(ValueError):
            DatasetRole.parse("DISCOVERY")
        with self.assertRaises(TypeError):
            DatasetRole.parse(DatasetRole.TARGET)

    def test_schema_version_fails_closed(self) -> None:
        self.assertEqual(validate_schema_version(RLRMB_SCHEMA_VERSION), RLRMB_SCHEMA_VERSION)
        for unsupported in (0, 2, True, "1", None):
            with self.subTest(unsupported=unsupported):
                with self.assertRaises((TypeError, ValueError)):
                    validate_schema_version(unsupported)

    def test_identity_leakage_is_rejected(self) -> None:
        for field in (
            "opponent_name",
            "submission_id",
            "replay_id",
            "episode_id",
            "tape_id",
        ):
            with self.subTest(field=field):
                with self.assertRaises(ValueError):
                    validate_runtime_identity_safe({field: "identity"})

    def test_own_seed_is_legal_but_offline_seed_provenance_is_not(self) -> None:
        validate_runtime_identity_safe({"feature": "OWN_SEED", "threshold": 4})
        validate_runtime_identity_safe({"OWN_SEED": 4})
        for field in ("replay_seed", "episode_seed"):
            with self.subTest(field=field):
                with self.assertRaises(ValueError):
                    validate_runtime_identity_safe({field: 17})

    def test_nested_identity_leakage_is_rejected(self) -> None:
        payload = {"phases": [{"termination": {"metadata": {"replay_id": "r1"}}}]}
        with self.assertRaisesRegex(ValueError, "replay_id"):
            validate_runtime_identity_safe(payload)


class FailureTests(unittest.TestCase):
    def test_failure_taxonomy_is_complete(self) -> None:
        expected = {
            "NO_HEADROOM",
            "BAD_CHECKPOINT",
            "BAD_HORIZON",
            "OPEN_LOOP_FRAGILITY",
            "MIXED_INITIATION",
            "NON_TRANSFERABLE",
            "DONOR_LIMITATION",
            "REPRESENTATION_LIMITATION",
            "TARGET_CAPABILITY_LOST",
            "GATE_LIMITATION",
            "FUTURE_DEPENDENT",
            "SEARCH_BUDGET_EXHAUSTED",
            "CUDA_FAULT",
            "SEMANTIC_MISMATCH",
        }
        self.assertEqual({failure.value for failure in FailureClass}, expected)

    def test_every_failure_has_exactly_one_action(self) -> None:
        self.assertEqual(set(FAILURE_ACTIONS), set(FailureClass))
        self.assertTrue(all(isinstance(action, RepairAction) for action in FAILURE_ACTIONS.values()))

    def test_failure_action_mapping(self) -> None:
        expected = {
            FailureClass.NO_HEADROOM: RepairAction.ABANDON_BASIN,
            FailureClass.BAD_CHECKPOINT: RepairAction.RESAMPLE_CHECKPOINT,
            FailureClass.BAD_HORIZON: RepairAction.ADJUST_HORIZON,
            FailureClass.OPEN_LOOP_FRAGILITY: RepairAction.REACTIVE_REPAIR,
            FailureClass.MIXED_INITIATION: RepairAction.REACTIVE_REPAIR,
            FailureClass.NON_TRANSFERABLE: RepairAction.SEMANTIC_REPAIR,
            FailureClass.DONOR_LIMITATION: RepairAction.EXTEND_REPRESENTATION,
            FailureClass.REPRESENTATION_LIMITATION: RepairAction.EXTEND_REPRESENTATION,
            FailureClass.TARGET_CAPABILITY_LOST: RepairAction.PRESERVE_TARGET,
            FailureClass.GATE_LIMITATION: RepairAction.REVIEW_GATE_LIMITATION,
            FailureClass.FUTURE_DEPENDENT: RepairAction.PERMANENT_REJECT,
            FailureClass.SEARCH_BUDGET_EXHAUSTED: RepairAction.SWITCH_BASIN_OR_OPERATOR,
            FailureClass.CUDA_FAULT: RepairAction.INFRASTRUCTURE_STOP,
            FailureClass.SEMANTIC_MISMATCH: RepairAction.INFRASTRUCTURE_STOP,
        }
        self.assertEqual(FAILURE_ACTIONS, expected)

    def test_failure_record_is_immutable_and_deterministic(self) -> None:
        first = SearchFailure(
            candidate_id="candidate-1",
            failure_class=FailureClass.NON_TRANSFERABLE,
            stage="transfer",
            evidence_ids=("evidence-b", "evidence-a", "evidence-a"),
        )
        second = SearchFailure(
            candidate_id="candidate-1",
            failure_class=FailureClass.NON_TRANSFERABLE,
            stage="transfer",
            evidence_ids=("evidence-a", "evidence-b"),
        )
        self.assertEqual(first.evidence_ids, ("evidence-a", "evidence-b"))
        self.assertEqual(first.canonical_json(), second.canonical_json())
        self.assertEqual(first.suggested_action, RepairAction.SEMANTIC_REPAIR)
        with self.assertRaises(AttributeError):
            first.stage = "changed"  # type: ignore[misc]

    def test_failure_record_validates_required_fields(self) -> None:
        with self.assertRaises(ValueError):
            SearchFailure("", FailureClass.NO_HEADROOM, "search")
        with self.assertRaises(ValueError):
            SearchFailure("candidate-1", FailureClass.NO_HEADROOM, "")


if __name__ == "__main__":
    unittest.main()
