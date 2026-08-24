from __future__ import annotations

import copy
import hashlib
import unittest

from tools.bkg4_search.bonsai_editformer.invariant_data import invariant_corpus_arrays
from tools.bkg4_search.edit_search.fitness import StateEvaluation
from tools.bkg4_search.invariant_module import (
    AbstractAction,
    GuardAtom,
    InvariantRule,
    InvariantStrategicModule,
    LiftedGuard,
    ModuleSketch,
)
from tools.bkg4_search.invariant_schema import make_corpus_record, validate_specialist_corpus
from tools.bkg4_search.invariant_search import pair_relative_case_rows
from tools.bkg4_search.invariant_specialists import SpecialistLineage


def _route(value: int, route: int = 70) -> InvariantStrategicModule:
    return InvariantStrategicModule((InvariantRule(
        ModuleSketch("route", "INSTANT", "ROUTE_MODE"), "GLOBAL",
        LiftedGuard((GuardAtom("OWN_MONEY", "GE", value),)),
        AbstractAction("ROUTE_OVERRIDE", route_id=route),
    ),))


def _evaluation(module: InvariantStrategicModule, *, base: int, trace: int) -> StateEvaluation:
    return StateEvaluation(
        module.semantic_id(), "specialist/label", ("c0", "c1"),
        (base, base + 100), (1, 2), (3, 4), (trace, trace + 1),
    )


def _lineage_dict(
    lineage: SpecialistLineage,
    *,
    root_semantic_id: str,
    parent_evidence: dict | None,
) -> dict:
    return {
        "lineage_schema_version": 2,
        "root_controller": lineage.root_controller,
        "root_semantic_id": root_semantic_id,
        "parent_semantic_id": lineage.parent_semantic_id,
        "operator": lineage.operator,
        "depth": lineage.depth,
        "source_margin_delta": lineage.source_margin_delta,
        "lineage_id": lineage.lineage_id,
        "source_goal_id": lineage.source_goal_id,
        "source_goal_path": [lineage.source_goal_id] if lineage.source_goal_id else [],
        "structural_mechanisms": list(lineage.structural_mechanisms),
        "retrospective_goal_id": lineage.retrospective_goal_id,
        "parent_evidence": parent_evidence,
    }


def _ancestor_node(lineage: SpecialistLineage, *, root_semantic_id: str) -> dict:
    return {
        "semantic_id": lineage.module.semantic_id(),
        "canonical_ast": lineage.module.canonical(),
        "depth": lineage.depth,
        "parent_semantic_id": lineage.parent_semantic_id,
        "root_semantic_id": root_semantic_id,
        "lineage_id": lineage.lineage_id,
        "operator": lineage.operator,
    }


def _parent_evidence(
    ancestry: tuple[SpecialistLineage, ...],
    parent_eval: StateEvaluation,
    *,
    root_semantic_id: str,
) -> dict:
    return {
        "evidence_version": 1,
        "ancestry": [_ancestor_node(row, root_semantic_id=root_semantic_id) for row in ancestry],
        "exact_evaluation": parent_eval.to_dict(),
    }


def _record(
    lineage: SpecialistLineage,
    *,
    root_semantic_id: str,
    source_rows: list[dict],
    parent_evidence: dict | None,
) -> dict:
    return make_corpus_record(
        lineage.module,
        labels={
            "exact_data_role": "specialist_discovery",
            "source_quality": "strong_specialist",
            "specialist_categories": ["strong_specialist"],
            "specialist_tier": "STRONG_SPECIALIST",
            "activation_count": 4,
            "action_change_count": 2,
            "useful_groups": 2,
            "harmed_groups": [],
            "target_cohorts": ["cohort"],
            "group_outcomes": [
                {"group_id": "g0", "useful": True},
                {"group_id": "g1", "useful": True},
            ],
            "cohort_outcomes": [{"cohort": "cohort", "useful_groups": 2}],
            "source_relative_case_outcomes": source_rows,
            "source_relative_mean_delta": sum(float(row.get("margin_delta", 0)) for row in source_rows) / max(1, len(source_rows)),
            "source_relative_l2w": 0,
            "source_relative_w2l": 0,
            "composition": {},
            "composition_relationships": {"compatible": [], "conflicting": []},
            "lineage": _lineage_dict(
                lineage, root_semantic_id=root_semantic_id, parent_evidence=parent_evidence,
            ),
        },
        provenance={
            "exact_evaluator": True,
            "data_role": "specialist_discovery",
            "parent": "parent-controller",
            "controller": "parent-controller",
            "candidate_abi": "B4IM1/v1",
            "evaluator_abi": "B4IC1/v1",
            "feature_registry": 1,
        },
    )


def _fixture() -> tuple[dict, dict, dict]:
    root_module = _route(500, 70)
    child_module = _route(750, 212)
    grand_module = _route(1000, 70)
    root = SpecialistLineage.root(root_module, parent_controller="parent-controller")
    child = root.child(child_module, operator="local_structural_neighborhood", source_margin_delta=10.0)
    grand = child.child(grand_module, operator="local_structural_neighborhood", source_margin_delta=20.0)
    root_sid = root_module.semantic_id()
    root_eval = _evaluation(root_module, base=100, trace=10)
    child_eval = _evaluation(child_module, base=150, trace=20)
    grand_eval = _evaluation(grand_module, base=200, trace=30)
    root_record = _record(
        root, root_semantic_id=root_sid,
        source_rows=[{"case_id": "root", "margin_delta": 0}], parent_evidence=None,
    )
    child_record = _record(
        child, root_semantic_id=root_sid,
        source_rows=[dict(row) for row in pair_relative_case_rows(root_eval, child_eval)],
        parent_evidence=_parent_evidence((root,), root_eval, root_semantic_id=root_sid),
    )
    grand_record = _record(
        grand, root_semantic_id=root_sid,
        source_rows=[dict(row) for row in pair_relative_case_rows(child_eval, grand_eval)],
        parent_evidence=_parent_evidence((root, child), child_eval, root_semantic_id=root_sid),
    )
    return root_record, child_record, grand_record


class InvariantLineageIntegrityTests(unittest.TestCase):
    def test_valid_internal_root_child_grandchild_lineage_passes(self):
        records = _fixture()
        report = validate_specialist_corpus(records)
        self.assertEqual(report["rows"], 3)
        self.assertEqual(report["lineage_roots"], 1)
        self.assertEqual(report["external_parent_rows"], 0)
        arrays = invariant_corpus_arrays(records)
        self.assertEqual(arrays["b4im_cat"].shape[0], 3)

    def test_valid_external_exact_labeled_parent_passes(self):
        root, child, _ = _fixture()
        del root
        report = validate_specialist_corpus((child,))
        self.assertEqual(report["external_parent_rows"], 1)
        invariant_corpus_arrays((child,))

    def test_valid_nonretained_depth_one_parent_for_grandchild_passes(self):
        _, _, grand = _fixture()
        report = validate_specialist_corpus((grand,))
        self.assertEqual(report["external_parent_rows"], 1)
        invariant_corpus_arrays((grand,))

    def test_lineage_corruption_matrix_fails_closed(self):
        root, child, grand = _fixture()
        cases: list[tuple[str, tuple[dict, ...]]] = []

        bad = copy.deepcopy(root)
        bad["target_labels"]["lineage"]["parent_semantic_id"] = child["semantic_id"]
        cases.append(("depth0_with_parent", (bad, child)))

        bad = copy.deepcopy(child)
        bad["target_labels"]["lineage"]["parent_semantic_id"] = None
        cases.append(("depth1_without_parent", (root, bad)))

        bad = copy.deepcopy(child)
        bad["target_labels"]["lineage"]["parent_semantic_id"] = "bogus-parent"
        cases.append(("bogus_parent", (root, bad)))

        bad = copy.deepcopy(child)
        bad["target_labels"]["lineage"]["depth"] = 2
        cases.append(("wrong_depth", (root, bad)))

        bad = copy.deepcopy(child)
        bad["target_labels"]["lineage"]["parent_semantic_id"] = child["semantic_id"]
        cases.append(("self_parent", (root, bad)))

        a = copy.deepcopy(root)
        b = copy.deepcopy(child)
        a["target_labels"]["lineage"]["depth"] = 2
        a["target_labels"]["lineage"]["parent_semantic_id"] = b["semantic_id"]
        b["target_labels"]["lineage"]["parent_semantic_id"] = a["semantic_id"]
        cases.append(("two_row_cycle", (a, b)))

        a = copy.deepcopy(root)
        b = copy.deepcopy(child)
        c = copy.deepcopy(grand)
        a["target_labels"]["lineage"]["depth"] = 3
        a["target_labels"]["lineage"]["parent_semantic_id"] = c["semantic_id"]
        cases.append(("deep_cycle", (a, b, c)))

        bad = copy.deepcopy(child)
        bad["target_labels"]["lineage"]["root_semantic_id"] = "wrong-root"
        cases.append(("wrong_root", (root, bad)))

        bad = copy.deepcopy(root)
        bad["target_labels"]["lineage"]["root_semantic_id"] = child["semantic_id"]
        cases.append(("depth0_wrong_root", (bad, child)))

        bad = copy.deepcopy(grand)
        bad["target_labels"]["lineage"]["root_semantic_id"] = child["semantic_id"]
        cases.append(("child_root_differs_from_parent", (root, child, bad)))

        bad = copy.deepcopy(child)
        bad["target_labels"]["lineage"]["parent_evidence"] = None
        cases.append(("external_parent_evidence_missing", (bad,)))

        bad = copy.deepcopy(child)
        evidence = bad["target_labels"]["lineage"]["parent_evidence"]
        evidence["ancestry"][-1]["canonical_ast"] = _route(333, 70).canonical()
        cases.append(("external_parent_semantic_mismatch", (bad,)))

        bad = copy.deepcopy(child)
        evidence = bad["target_labels"]["lineage"]["parent_evidence"]
        evidence["exact_evaluation"]["margins"][0] += 1
        cases.append(("external_parent_exact_label_mismatch", (bad,)))

        for name, records in cases:
            with self.subTest(name=name):
                with self.assertRaises(ValueError):
                    validate_specialist_corpus(records)

    def test_validator_attack_is_rejected_at_corpus_and_model_boundaries(self):
        root, _, _ = _fixture()
        bad = copy.deepcopy(root)
        lineage = bad["target_labels"]["lineage"]
        lineage["depth"] = 2
        lineage["parent_semantic_id"] = "bogus-parent-semantic-id"
        lineage["root_semantic_id"] = "bogus-root-semantic-id"
        with self.assertRaises(ValueError):
            validate_specialist_corpus((bad,))
        with self.assertRaises(ValueError):
            invariant_corpus_arrays((bad,))

    def test_stale_lineage_schema_fails_closed(self):
        root, _, _ = _fixture()
        stale = copy.deepcopy(root)
        stale["target_labels"]["lineage"]["lineage_schema_version"] = 1
        with self.assertRaises(ValueError):
            validate_specialist_corpus((stale,))

    def test_lineage_id_is_verified_not_trusted(self):
        root, _, _ = _fixture()
        bad = copy.deepcopy(root)
        bad["target_labels"]["lineage"]["lineage_id"] = hashlib.sha256(b"bogus").hexdigest()
        with self.assertRaises(ValueError):
            validate_specialist_corpus((bad,))


if __name__ == "__main__":
    unittest.main()
