from __future__ import annotations

import unittest

from tools.bkg4_search.bonsai_editformer.actor_grammar import EDIT_OPERATIONS, apply_edit
from tools.bkg4_search.module_codec import ConditionGene, ModuleGenome, RuleGene


class CandidateGenerationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            from tools.edit_corpus import candidates
        except ImportError as exc:
            raise AssertionError(f"candidate generator is not implemented yet: {exc}") from exc
        cls.g = candidates

    def parent(self):
        return ModuleGenome(
            144,
            (
                RuleGene((ConditionGene("SHOP_AT", "EQ", 0, 4), ConditionGene("OWN_MONEY", "LE", 0, 3000)), 213),
                RuleGene((ConditionGene("MARKET_PRICE", "GE", 3, 120),), 214),
            ),
        )

    def context(self):
        return [
            {"feature": "OWN_MONEY", "index": 0, "value": 2500},
            {"feature": "OPP_MONEY", "index": 0, "value": 4000},
            {"feature": "MARKET_PRICE", "index": 3, "value": 172},
            {"feature": "MARKET_INVENTORY", "index": 3, "value": 20},
            {"feature": "STEP", "index": 0, "value": 192},
        ]

    def test_generation_is_deterministic_semantically_unique_and_legal(self):
        parent = self.parent()
        a = self.g.generate_typed_candidates(parent, self.context(), n=96, seed=991, route_ids=self.g.DEFAULT_ROUTE_IDS)
        b = self.g.generate_typed_candidates(parent, self.context(), n=96, seed=991, route_ids=self.g.DEFAULT_ROUTE_IDS)
        self.assertEqual([x.edit_id for x in a], [x.edit_id for x in b])
        self.assertEqual(len({x.edit_id for x in a}), len(a))
        self.assertEqual(len({x.child_semantic_id for x in a}), len(a))
        self.assertGreaterEqual(len(a), 72)
        for row in a:
            child = apply_edit(parent, row.action, self.g.DEFAULT_ROUTE_IDS)
            self.assertEqual(child.semantic_id(), row.child_semantic_id)
            self.assertEqual(row.parent_semantic_id, parent.semantic_id())
            self.assertIn(row.action.operation, EDIT_OPERATIONS)

    def test_generation_preserves_explicit_what_where_how_fields(self):
        rows = self.g.generate_typed_candidates(self.parent(), self.context(), n=48, seed=42, route_ids=self.g.DEFAULT_ROUTE_IDS)
        for row in rows:
            payload = row.action_payload
            self.assertEqual(payload["operation"], row.action.operation)
            self.assertEqual(payload["target_node"], row.action.target_node)
            self.assertIn("target_kind", payload)
            self.assertIn("target_rule_index", payload)
            self.assertIn("target_condition_index", payload)
            self.assertIn("feature", payload)
            self.assertIn("cmp", payload)
            self.assertIn("value", payload)
            self.assertIn("route", payload)
            self.assertIn("priority", payload)
            self.assertIn("prefix_step", payload)

    def test_generation_spans_many_operation_families_and_diagnostic_proposal_classes(self):
        rows = self.g.generate_typed_candidates(self.parent(), self.context(), n=160, seed=77, route_ids=self.g.DEFAULT_ROUTE_IDS)
        ops = {x.action.operation for x in rows}
        self.assertGreaterEqual(len(ops), 9)
        self.assertIn("CHANGE_THRESHOLD", ops)
        self.assertIn("CHANGE_ROUTE", ops)
        self.assertIn("REPLACE_SUBTREE", ops)
        self.assertIn("REMOVE_REDUNDANT_BRANCH", ops)
        classes = {x.proposal_class for x in rows}
        self.assertIn("observed_threshold", classes)
        self.assertIn("broad_probe", classes)
        self.assertIn("inert_probe", classes)

    def test_context_thresholds_are_used_without_identity_fields(self):
        context = self.context() + [
            {"feature": "team_name", "index": 0, "value": 123},
            {"feature": "seed", "index": 0, "value": 999},
        ]
        pool = self.g.observed_conditions(context)
        self.assertTrue(any(c.feature == "OWN_MONEY" and c.value == 2500 for c in pool))
        self.assertFalse(any(c.feature in {"team_name", "seed"} for c in pool))


if __name__ == "__main__":
    unittest.main()
