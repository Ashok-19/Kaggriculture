from __future__ import annotations

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
)


def representative_module() -> InvariantStrategicModule:
    return InvariantStrategicModule((
        InvariantRule(
            ModuleSketch("market", "INSTANT", "ENGINE_MECHANIC"),
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


if __name__ == "__main__":
    unittest.main()
