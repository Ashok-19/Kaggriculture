from __future__ import annotations

import copy
import json
import unittest

from tools.bkg4_search.invariant_module import (
    AbstractAction,
    GuardAtom,
    InvariantRule,
    InvariantStrategicModule,
    LiftedGuard,
    ModuleSketch,
)
from tools.bkg4_search.invariant_schema import (
    invariant_model_input,
    make_corpus_record,
    module_from_model_input,
    validate_corpus_records,
    validate_quality_gated_corpus,
    validate_specialist_corpus,
)
from tools.bkg4_search.invariant_specialists import SpecialistLineage, analyze_specialist
from tools.bkg4_search.invariant_groups import GroupOutcome


def representative_module() -> InvariantStrategicModule:
    return InvariantStrategicModule((
        InvariantRule(
            ModuleSketch("engine_mechanic", "INSTANT", "ENGINE_MECHANIC"),
            "GLOBAL",
            LiftedGuard((
                GuardAtom("MARKET_PRICE", "LE", 1, arg=6),
                GuardAtom("OWN_SHED", "GE", 2, arg=6),
            )),
            AbstractAction("MARKET_INTENT", market_op=6, item=6, quantity=2, authority=5, priority=220),
        ),
        InvariantRule(
            ModuleSketch("route", "COMMITTED", "ROUTE_MODE"),
            "GLOBAL",
            LiftedGuard((GuardAtom("OWN_MONEY", "LE", 1000),)),
            AbstractAction("COMMITTED_ROUTE", route_id=70),
            LiftedGuard((GuardAtom("OWN_MONEY", "GE", 2500),)),
        ),
    ))


def _root_lineage_labels(lineage: SpecialistLineage) -> dict:
    return {
        "lineage_schema_version": 2,
        "root_controller": lineage.root_controller,
        "root_semantic_id": lineage.root_semantic_id,
        "parent_semantic_id": lineage.parent_semantic_id,
        "operator": lineage.operator,
        "depth": lineage.depth,
        "source_margin_delta": lineage.source_margin_delta,
        "lineage_id": lineage.lineage_id,
        "source_goal_id": lineage.source_goal_id,
        "source_goal_path": list(lineage.source_goal_path),
        "structural_mechanisms": list(lineage.structural_mechanisms),
        "retrospective_goal_id": lineage.retrospective_goal_id,
        "parent_evidence": None,
    }


class InvariantSchemaTests(unittest.TestCase):
    def test_canonical_ast_round_trips_losslessly_through_model_tokens(self):
        module = representative_module()
        model_input = invariant_model_input(module)
        restored = module_from_model_input(model_input)
        self.assertEqual(restored.canonical(), module.canonical())
        self.assertEqual(restored.semantic_id(), module.semantic_id())

    def test_model_input_excludes_identity_and_outcome_leakage(self):
        record = make_corpus_record(
            representative_module(),
            labels={"mean_margin_delta": 12.5, "certification_passed": False},
            provenance={
                "parent": "w33_s172_expansion_v1", "evaluator_abi": "B4IC1/v1",
                "tape_path": "allowed-only-in-provenance", "seed": 42,
            },
        )
        text = repr(record["model_input"]).lower()
        for forbidden in (
            "tape_path", "tape_id", "replay_id", "seed", "opponent", "recursion",
            "certification", "holdout", "outcome", "margin_delta",
        ):
            self.assertNotIn(forbidden, text)
        report = validate_corpus_records([record])
        self.assertEqual(report["rows"], 1)
        self.assertEqual(report["unique_semantics"], 1)

    def test_validator_rejects_forbidden_field_in_model_input(self):
        record = make_corpus_record(
            representative_module(), labels={},
            provenance={"parent": "p", "evaluator_abi": "B4IC1/v1"},
        )
        record["model_input"]["seed"] = 7
        with self.assertRaisesRegex(ValueError, "forbidden model input"):
            validate_corpus_records([record])

    def test_model_input_rejects_all_reported_leakage_aliases(self):
        aliases = (
            "heldout_result", "search_rank", "useful_group_count",
            "group_support", "evaluator_margin",
        )
        for alias in aliases:
            with self.subTest(alias=alias):
                record = make_corpus_record(
                    representative_module(), labels={},
                    provenance={"parent": "p", "evaluator_abi": "B4IC1/v1"},
                )
                record["model_input"][alias] = 1
                with self.assertRaisesRegex(ValueError, "model input"):
                    validate_corpus_records([record])

    def test_tokenized_ast_rejects_hidden_noncanonical_fields(self):
        model_input = invariant_model_input(representative_module())
        model_input["tokens"][-1:-1] = ['K:"search_rank"', "I:7"]
        with self.assertRaisesRegex(ValueError, "forbidden|noncanonical"):
            module_from_model_input(model_input)

    def test_stale_noncanonical_guard_order_fails_closed_on_resume(self):
        module = InvariantStrategicModule((InvariantRule(
            ModuleSketch("route", "INSTANT", "ROUTE_MODE"), "GLOBAL",
            LiftedGuard((GuardAtom("DAY", "GE", 1), GuardAtom("OWN_MONEY", "GE", 500))),
            AbstractAction("ROUTE_OVERRIDE", route_id=70),
        ),))
        model_input = invariant_model_input(module)
        stale_ast = module.canonical()
        stale_ast["rules"][0]["guard"]["atoms"].reverse()
        from tools.bkg4_search.invariant_schema import _encode
        stale_tokens = []
        _encode(stale_ast, stale_tokens)
        model_input["tokens"] = stale_tokens
        with self.assertRaisesRegex(ValueError, "noncanonical"):
            module_from_model_input(model_input)

    def test_corpus_record_rejects_unknown_top_level_fields(self):
        record = make_corpus_record(
            representative_module(), labels={},
            provenance={"parent": "p", "evaluator_abi": "B4IC1/v1"},
        )
        record["training_hint"] = "future outcome"
        with self.assertRaisesRegex(ValueError, "corpus row fields"):
            validate_corpus_records([record])

    def test_quality_gate_requires_exact_crossfit_positive_evidence(self):
        labels = {
            "fitness": {"unsafe_groups": 0, "useful_groups": 3},
            "qualification": {"production": True},
            "activation_count": 8,
            "action_change_count": 4,
            "group_outcomes": [],
            "exact_data_role": "discovery",
            "source_quality": "crossfit_positive",
            "crossfit_fold_support": 4,
            "outer_fold_support": 4,
        }
        provenance = {
            "parent": "p", "controller": "c", "evaluator_abi": "B4IC1/v1",
            "candidate_abi": "B4IM1/v1", "feature_registry": 1,
            "exact_evaluator": True, "data_role": "discovery",
        }
        record = make_corpus_record(
            representative_module(), labels=labels, provenance=provenance,
        )
        report = validate_quality_gated_corpus((record,))
        self.assertEqual(report["crossfit_positive"], 1)
        self.assertEqual(report["positive_structural_skeletons"], 1)
        with self.assertRaisesRegex(ValueError, "duplicate semantics"):
            validate_quality_gated_corpus((record, copy.deepcopy(record)))

        bad_label = copy.deepcopy(record)
        bad_label["target_labels"]["fitness"]["unsafe_groups"] = 1
        with self.assertRaisesRegex(ValueError, "safe and production-qualified"):
            validate_quality_gated_corpus((bad_label,))

        bad_provenance = copy.deepcopy(record)
        bad_provenance["provenance"]["exact_evaluator"] = False
        with self.assertRaisesRegex(ValueError, "exact CUDA provenance"):
            validate_quality_gated_corpus((bad_provenance,))

    def test_specialist_corpus_retains_mixed_downside_as_structured_labels(self):
        module = representative_module()
        evidence = analyze_specialist(
            module,
            (
                GroupOutcome("help-a", False, False, True, True, 1, 4000),
                GroupOutcome("help-b", False, False, True, True, 0, 2500),
                GroupOutcome("harm-c", True, True, True, False, 0, -6000),
            ),
            cohort_by_group={"help-a": "regime-a", "help-b": "regime-a", "harm-c": "regime-b"},
            activation_count=30, action_change_count=12,
        )
        lineage = SpecialistLineage.root(module, parent_controller="parent")
        record = make_corpus_record(
            module,
            labels={
                "exact_data_role": "specialist_discovery",
                "source_quality": "elite_specialist",
                "specialist_categories": ["elite_specialist", "high_risk_specialist", "generalization_boundary", "counter_specialist", "margin_specialist"],
                "specialist_tier": evidence.tier,
                "activation_count": evidence.activation_count,
                "action_change_count": evidence.action_change_count,
                "useful_groups": evidence.useful_groups,
                "harmed_groups": list(evidence.harmed_groups),
                "target_cohorts": list(evidence.target_cohorts),
                "group_outcomes": [row.__dict__ for row in evidence.outcomes],
                "cohort_outcomes": [row.__dict__ for row in evidence.cohorts],
                "source_relative_case_outcomes": [{"margin_delta": 1}],
                "source_relative_mean_delta": 1.0, "source_relative_l2w": 0, "source_relative_w2l": 0,
                "composition": evidence.composition.__dict__,
                "composition_relationships": {"compatible": [], "conflicting": ["other"]},
                "lineage": _root_lineage_labels(lineage),
            },
            provenance={
                "exact_evaluator": True, "data_role": "specialist_discovery",
                "parent": "parent", "controller": "parent", "candidate_abi": "B4IM1/v1",
                "evaluator_abi": "B4IC1/v1", "feature_registry": 1,
            },
        )
        report = validate_specialist_corpus((record,))
        self.assertEqual(report["elite_specialist"], 1)
        self.assertEqual(report["mixed_risk_rows"], 1)
        self.assertEqual(report["category_high_risk_specialist"], 1)
        self.assertEqual(report["category_generalization_boundary"], 1)
        encoded = json.dumps(record["model_input"], sort_keys=True)
        self.assertNotIn("regime-a", encoded)
        self.assertNotIn("help-a", encoded)

    def test_specialist_corpus_rejects_fake_elite_without_repeat_support(self):
        module = representative_module()
        record = make_corpus_record(
            module,
            labels={
                "exact_data_role": "specialist_discovery", "source_quality": "elite_specialist",
                "specialist_tier": "ELITE_SPECIALIST", "activation_count": 1, "action_change_count": 1,
                "useful_groups": 1, "harmed_groups": [], "target_cohorts": ["x"],
                "specialist_categories": ["elite_specialist"],
                "group_outcomes": [{"group_id": "one", "useful": True}],
                "cohort_outcomes": [], "source_relative_case_outcomes": [{"margin_delta": 1}],
                "source_relative_mean_delta": 1.0, "source_relative_l2w": 0, "source_relative_w2l": 0,
                "composition": {}, "composition_relationships": {"compatible": [], "conflicting": []},
                "lineage": _root_lineage_labels(SpecialistLineage.root(module, parent_controller="parent")),
            },
            provenance={
                "exact_evaluator": True, "data_role": "specialist_discovery", "parent": "parent", "controller": "parent",
                "candidate_abi": "B4IM1/v1", "evaluator_abi": "B4IC1/v1", "feature_registry": 1,
            },
        )
        with self.assertRaisesRegex(ValueError, "repeat support"):
            validate_specialist_corpus((record,))


if __name__ == "__main__":
    unittest.main()
