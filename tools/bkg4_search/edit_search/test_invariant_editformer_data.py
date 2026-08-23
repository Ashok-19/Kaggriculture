from __future__ import annotations

import unittest

import numpy as np

from tools.bkg4_search.bonsai_editformer.invariant_data import (
    B4IM_CAT_COLS,
    B4IM_NUM_COLS,
    B4IM_TOKEN_CAP,
    invariant_corpus_arrays,
    invariant_grouped_split,
    invariant_module_arrays,
    invariant_numeric_batch,
    module_from_invariant_arrays,
)
from tools.bkg4_search.invariant_module import (
    AbstractAction,
    GuardAtom,
    InvariantRule,
    InvariantStrategicModule,
    LiftedGuard,
    ModuleSketch,
)
from tools.bkg4_search.invariant_schema import make_corpus_record
from tools.bkg4_search.invariant_search import default_search_goals, emit_skeletons


def representative_module() -> InvariantStrategicModule:
    return InvariantStrategicModule((
        InvariantRule(
            ModuleSketch("route", "COMMITTED", "ROUTE_MODE"), "GLOBAL",
            LiftedGuard((GuardAtom("OWN_MONEY", "LE", 700),)),
            AbstractAction("COMMITTED_ROUTE", route_id=70),
            LiftedGuard((GuardAtom("OWN_MONEY", "GE", 1400),)),
        ),
        InvariantRule(
            ModuleSketch("recovery", "INSTANT", "RECOVER_RESOURCE_INVARIANT"), "EACH_OWN_UNIT",
            LiftedGuard((
                GuardAtom("OWN_TILE_HAS_ANIMAL", "EQ", 1),
                GuardAtom("OWN_INVENTORY", "GE", 1, arg=0),
            )),
            AbstractAction("UNIT_REWRITE", match_op=0, unit_op=15),
        ),
        InvariantRule(
            ModuleSketch("market", "INSTANT", "CREATE_FINANCING_RUNWAY"), "GLOBAL",
            LiftedGuard((GuardAtom("MARKET_PRICE", "LE", 1, arg=6), GuardAtom("OWN_SHED", "GE", 2, arg=6))),
            AbstractAction("MARKET_INTENT", market_op=6, item=6, quantity=2, authority=5, priority=220),
        ),
    ))


class InvariantEditFormerDataTests(unittest.TestCase):
    def test_fixed_shape_arrays_round_trip_losslessly(self):
        modules = (representative_module(), InvariantStrategicModule((InvariantRule(
            ModuleSketch("route", "INSTANT", "ROUTE_MODE"), "GLOBAL",
            LiftedGuard((GuardAtom("OWN_LAND", "LE", 2),)),
            AbstractAction("ROUTE_OVERRIDE", route_id=29),
        ),)))
        for module in modules:
            encoded = invariant_module_arrays(module)
            self.assertEqual(encoded["b4im_cat"].shape, (B4IM_TOKEN_CAP, B4IM_CAT_COLS))
            self.assertEqual(encoded["b4im_num"].shape, (B4IM_TOKEN_CAP, B4IM_NUM_COLS))
            rebuilt = module_from_invariant_arrays(
                encoded["b4im_cat"], encoded["b4im_num"], encoded["b4im_mask"],
            )
            self.assertEqual(rebuilt.canonical(), module.canonical())
            self.assertEqual(rebuilt.semantic_id(), module.semantic_id())

    def test_array_decoder_rejects_hidden_padded_payload(self):
        encoded = invariant_module_arrays(representative_module())
        cat = encoded["b4im_cat"].copy()
        cat[np.flatnonzero(~encoded["b4im_mask"])[0], 0] = 7
        with self.assertRaisesRegex(ValueError, "padded"):
            module_from_invariant_arrays(cat, encoded["b4im_num"], encoded["b4im_mask"])

    def test_model_arrays_collapse_reversed_guard_order(self):
        first = InvariantStrategicModule((InvariantRule(
            ModuleSketch("route", "INSTANT", "ROUTE_MODE"), "GLOBAL",
            LiftedGuard((GuardAtom("DAY", "GE", 1), GuardAtom("OWN_MONEY", "GE", 500))),
            AbstractAction("ROUTE_OVERRIDE", route_id=70),
        ),))
        reversed_ = InvariantStrategicModule((InvariantRule(
            ModuleSketch("route", "INSTANT", "ROUTE_MODE"), "GLOBAL",
            LiftedGuard((GuardAtom("OWN_MONEY", "GE", 500), GuardAtom("DAY", "GE", 1))),
            AbstractAction("ROUTE_OVERRIDE", route_id=70),
        ),))
        a = invariant_module_arrays(first)
        b = invariant_module_arrays(reversed_)
        for key in ("b4im_cat", "b4im_num", "b4im_mask"):
            self.assertTrue(np.array_equal(a[key], b[key]))

    def test_corpus_adapter_batches_and_keeps_skeletons_split_disjoint(self):
        modules = []
        goals = default_search_goals((29, 64, 70, 211))
        for index, goal in enumerate(goals):
            modules.extend(emit_skeletons(goal, n=8, seed=100 + index, route_ids=(29, 64, 70, 211)))
        records = [make_corpus_record(
            module,
            labels={"useful_groups": index % 3, "production_qualified": index % 2},
            provenance={"parent": "parent", "evaluator_abi": "B4IC1/v1"},
        ) for index, module in enumerate(modules)]
        arrays = invariant_corpus_arrays(records)
        self.assertEqual(arrays["b4im_cat"].shape[0], len(records))
        batch = invariant_numeric_batch(arrays, (0, len(records) - 1))
        self.assertEqual(batch["b4im_cat"].shape, (2, B4IM_TOKEN_CAP, B4IM_CAT_COLS))
        self.assertTrue(np.all(batch["b4im_mask"][:, 0]))
        split = invariant_grouped_split(records, validation_fraction=0.25)
        self.assertFalse(set(split.train_indices) & set(split.validation_indices))
        self.assertFalse(set(split.train_skeletons) & set(split.validation_skeletons))
        self.assertEqual(set(split.train_indices) | set(split.validation_indices), set(range(len(records))))


if __name__ == "__main__":
    unittest.main()
