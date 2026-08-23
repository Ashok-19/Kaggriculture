from __future__ import annotations

import copy
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
)


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


if __name__ == "__main__":
    unittest.main()
