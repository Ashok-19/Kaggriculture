from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tools.bkg4_search.bonsai_editformer.actor_grammar import EditAction
from tools.bkg4_search.module_codec import ConditionGene, ModuleGenome, RuleGene


ROUTES = (22, 23, 29, 32, 34, 46, 63, 64, 70, 77, 79, 211, 212, 213, 214)


def parent2() -> ModuleGenome:
    return ModuleGenome(
        144,
        (
            RuleGene((ConditionGene("SHOP_AT", "EQ", 0, 4), ConditionGene("OWN_MONEY", "LE", 0, 3000)), 213),
            RuleGene((ConditionGene("MARKET_PRICE", "GE", 3, 120),), 214),
        ),
    )


class EditSearchSchemaTests(unittest.TestCase):
    def test_multi_edit_program_reconstructs_state_relative_targets(self):
        from tools.bkg4_search.edit_search.schema import EditProgram
        p = parent2()
        actions = (
            EditAction("CHANGE_THRESHOLD", 5, feature="OWN_MONEY", cmp="LE", index=0, value=150),
            EditAction("CHANGE_PRIORITY", 6, priority=0),
            EditAction("CHANGE_ROUTE", 3, route=70),
        )
        program = EditProgram.build(p, actions, ROUTES, emitter="lns", seed=17, generation_depth=3)
        child = program.reconstruct(p, ROUTES)
        self.assertEqual(child.semantic_id(), program.child_semantic_id)
        self.assertEqual(len(program.atoms), 3)
        self.assertEqual(len(program.intermediate_semantic_ids), 3)
        self.assertEqual([a.target_kind for a in program.atoms], ["condition", "rule", "rule"])
        self.assertEqual([a.target_rule_index for a in program.atoms], [0, 1, 0])
        self.assertEqual(program.program_length, 3)
        self.assertEqual(program.generation_depth, 3)
        self.assertEqual(program.program_id, EditProgram.build(p, actions, ROUTES, emitter="lns", seed=17, generation_depth=3).program_id)

    def test_reconstruction_rejects_wrong_parent(self):
        from tools.bkg4_search.edit_search.schema import EditProgram
        p = parent2()
        program = EditProgram.build(p, (EditAction("CHANGE_WINDOW", 0, prefix_step=216),), ROUTES, emitter="systematic", seed=1)
        other = ModuleGenome(145, p.rules)
        with self.assertRaisesRegex(ValueError, "source parent"):
            program.reconstruct(other, ROUTES)

    def test_all_current_typed_operations_are_reconstructable_on_legal_parent(self):
        from tools.bkg4_search.edit_search.schema import EditProgram
        p = parent2()
        actions = {
            "ADD_SPECIALIST": EditAction("ADD_SPECIALIST", 0, feature="DAY", cmp="GE", value=4, route=22),
            "ADD_CONDITION": EditAction("ADD_CONDITION", 3, feature="OPP_MONEY", cmp="GE", value=5000),
            "CHANGE_THRESHOLD": EditAction("CHANGE_THRESHOLD", 4, feature="SHOP_AT", cmp="EQ", index=0, value=5),
            "CHANGE_ROUTE": EditAction("CHANGE_ROUTE", 3, route=70),
            "CHANGE_PRIORITY": EditAction("CHANGE_PRIORITY", 3, priority=1),
            "CHANGE_WINDOW": EditAction("CHANGE_WINDOW", 0, prefix_step=200),
            "INSERT_RECOVERY": EditAction("INSERT_RECOVERY", 0, feature="OWN_MONEY", cmp="LE", value=500, route=29),
            "REPLACE_MACRO": EditAction("REPLACE_MACRO", 3, route=64),
            "SPLIT_LANE": EditAction("SPLIT_LANE", 6, feature="OPP_UNITS", cmp="GE", value=5, route=46),
            "COMPOSE_PREDICATES": EditAction("COMPOSE_PREDICATES", 6, feature="OPP_MONEY", cmp="GE", value=3000),
            "REPLACE_SUBTREE": EditAction("REPLACE_SUBTREE", 6, feature="DAY", cmp="GE", value=5, route=77),
            "REMOVE_REDUNDANT_BRANCH": EditAction("REMOVE_REDUNDANT_BRANCH", 6),
        }
        for name, action in actions.items():
            with self.subTest(operation=name):
                program = EditProgram.build(p, (action,), ROUTES, emitter="test", seed=3)
                self.assertEqual(program.reconstruct(p, ROUTES).semantic_id(), program.child_semantic_id)
                self.assertEqual(program.atoms[0].operation, name)

    def test_canonical_edit_actions_reject_malformed_typed_fields_instead_of_normalizing(self):
        from tools.bkg4_search.edit_search.schema import EditProgram
        p = parent2()
        bad = (
            EditAction("ADD_SPECIALIST", 0, feature="MARKET_PRICE", cmp="GE", index=99, value=120, route=22),
            EditAction("CHANGE_PRIORITY", 3, priority=99),
            EditAction("CHANGE_WINDOW", 0, prefix_step=999),
            EditAction("ADD_SPECIALIST", 0, feature="OWN_MONEY", cmp="GE", index=0, value=1000, route=22, condition_count=3),
            EditAction("CHANGE_THRESHOLD", 4, feature="MARKET_PRICE", cmp="GE", index=99, value=150),
        )
        for action in bad:
            with self.subTest(action=action):
                with self.assertRaises(ValueError):
                    EditProgram.build(p, (action,), ROUTES, emitter="malformed", seed=1)

    def test_parent_profile_loads_champion_assets_without_hardcoding_current_identity(self):
        from tools.bkg4_search.edit_search.schema import ParentProfile
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for name in ("controller.bkgc", "routes.bkv6", "legacy_routes.bkv6"):
                (root / name).write_bytes(b"fixture")
            identity = {
                "combined_agent_identity": "future-parent-id",
                "bkg4_abi": 4,
                "feature_registry": 1,
                "controller_semantic_hash": "future-controller",
                "v6_route_pack": {"route_ids": [22, 70, 214]},
                "runtime_reconciler": {"defer_slots": [5, 6], "route_override_slot": 19},
            }
            (root / "identity.json").write_text(json.dumps(identity))
            profile = ParentProfile.from_directory(root)
            self.assertEqual(profile.combined_agent_identity, "future-parent-id")
            self.assertEqual(profile.route_ids, (22, 70, 214))
            self.assertEqual(profile.feed_slot, 5)
            self.assertEqual(profile.feed_count, 2)
            self.assertEqual(profile.strategic_route_slot, 19)
            self.assertEqual(profile.controller_path, root / "controller.bkgc")


    def test_format_breaking_hardening_uses_new_search_corpus_and_emitter_versions(self):
        from tools.bkg4_search.edit_search import CORPUS_SCHEMA_VERSION, EMITTER_REGISTRY_VERSION, SEARCH_STATE_SCHEMA_VERSION
        self.assertGreaterEqual(SEARCH_STATE_SCHEMA_VERSION, 3)
        self.assertGreaterEqual(CORPUS_SCHEMA_VERSION, 3)
        self.assertGreaterEqual(EMITTER_REGISTRY_VERSION, 3)

if __name__ == "__main__":
    unittest.main(verbosity=2)
