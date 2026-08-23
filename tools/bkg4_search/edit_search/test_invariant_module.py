from __future__ import annotations

import unittest

from tools.bkg4_search.invariant_module import (
    B4IM1_CANDIDATE_SIZE,
    AbstractAction,
    GuardAtom,
    InvariantRule,
    InvariantStrategicModule,
    LiftedGuard,
    ModuleSketch,
    distill_invariant_module,
    pack_candidate,
    unpack_candidate,
)
from tools.bkg4_search.invariant_search import structural_skeleton


class InvariantModuleTests(unittest.TestCase):
    def feed_deferral(self) -> InvariantStrategicModule:
        return InvariantStrategicModule((
            InvariantRule(
                sketch=ModuleSketch("animal_lifecycle", "INSTANT", "PRESERVE_PRODUCTION_SAVE_FEED"),
                domain="EACH_OWN_UNIT",
                guard=LiftedGuard((
                    GuardAtom("OWN_TILE_HAS_ANIMAL", "EQ", 1),
                    GuardAtom("OWN_TILE_ITEM", "EQ", 10),  # COW
                    GuardAtom("OWN_TILE_CONSECUTIVE_UNFED", "EQ", 0),
                    GuardAtom("OWN_TILE_PENDING_CARE_BONUS", "EQ", 0),
                )),
                action=AbstractAction("UNIT_REWRITE", match_op=15, unit_op=0),  # FEED -> PASS
            ),
        ))

    def test_canonical_identity_ignores_metadata_order(self):
        a = self.feed_deferral()
        b = InvariantStrategicModule.from_dict(a.canonical())
        self.assertEqual(a.semantic_id(), b.semantic_id())
        self.assertEqual(a.canonical(), b.canonical())

    def test_guard_conjunction_order_collapses_to_one_canonical_program(self):
        def route(atoms):
            return InvariantStrategicModule((InvariantRule(
                ModuleSketch("route", "INSTANT", "ROUTE_MODE"), "GLOBAL",
                LiftedGuard(tuple(atoms)), AbstractAction("ROUTE_OVERRIDE", route_id=70),
            ),))

        first = route((GuardAtom("DAY", "GE", 1), GuardAtom("OWN_MONEY", "GE", 500)))
        reversed_ = route((GuardAtom("OWN_MONEY", "GE", 500), GuardAtom("DAY", "GE", 1)))
        self.assertEqual(first.canonical(), reversed_.canonical())
        self.assertEqual(pack_candidate(first), pack_candidate(reversed_))
        self.assertEqual(first.semantic_id(), reversed_.semantic_id())
        self.assertEqual(structural_skeleton(first), structural_skeleton(reversed_))

    def test_canonicalization_is_idempotent(self):
        module = InvariantStrategicModule((InvariantRule(
            ModuleSketch("route", "INSTANT", "ROUTE_MODE"), "GLOBAL",
            LiftedGuard((GuardAtom("OWN_MONEY", "GE", 500), GuardAtom("DAY", "GE", 1))),
            AbstractAction("ROUTE_OVERRIDE", route_id=70),
        ),))
        once = module.canonicalized()
        twice = once.canonicalized()
        self.assertEqual(once, twice)
        self.assertEqual(once.canonical(), module.canonical())

    def test_duplicate_atoms_still_fail_under_canonicalization(self):
        atom = GuardAtom("DAY", "GE", 1)
        module = InvariantStrategicModule((InvariantRule(
            ModuleSketch("route", "INSTANT", "ROUTE_MODE"), "GLOBAL",
            LiftedGuard((atom, atom)), AbstractAction("ROUTE_OVERRIDE", route_id=70),
        ),))
        with self.assertRaisesRegex(ValueError, "duplicate guard atom"):
            module.canonicalized()

    def test_lifted_bound_unit_rejects_ignored_concrete_arg_alias(self):
        module = InvariantStrategicModule((InvariantRule(
            ModuleSketch("maintenance", "INSTANT", "REDUCE_MAINTENANCE_BACKLOG"),
            "EACH_OWN_UNIT", LiftedGuard((GuardAtom("OWN_WORKER_X", "EQ", 4, arg=17),)),
            AbstractAction("UNIT_REWRITE", match_op=15, unit_op=0),
        ),))
        with self.assertRaisesRegex(ValueError, "bound unit.*arg|noncanonical"):
            module.validate()

    def test_ordinary_validation_rejects_concrete_worker_and_tile_identity(self):
        fixtures = (
            GuardAtom("OWN_WORKER_X", "EQ", 4, arg=17),
            GuardAtom("OWN_WORKER_TILE", "EQ", 44, arg=3),
            GuardAtom("OWN_TILE_KIND", "EQ", 1, arg=17),
        )
        for atom in fixtures:
            with self.subTest(atom=atom):
                module = InvariantStrategicModule((InvariantRule(
                    ModuleSketch("route", "INSTANT", "ROUTE_MODE"), "GLOBAL",
                    LiftedGuard((atom,)), AbstractAction("ROUTE_OVERRIDE", route_id=70),
                ),))
                with self.assertRaisesRegex(ValueError, "identity"):
                    module.validate()

    def test_ordinary_identity_guard_hardening_preserves_legal_args(self):
        legal = (
            InvariantStrategicModule((InvariantRule(
                ModuleSketch("market", "INSTANT", "CREATE_FINANCING_RUNWAY"), "GLOBAL",
                LiftedGuard((GuardAtom("MARKET_PRICE", "GE", 10, arg=6), GuardAtom("OWN_SHED", "GE", 1, arg=6))),
                AbstractAction("MARKET_INTENT", market_op=6, item=6, quantity=1, authority=5, priority=200),
            ),)),
            InvariantStrategicModule((InvariantRule(
                ModuleSketch("maintenance", "INSTANT", "REDUCE_MAINTENANCE_BACKLOG"), "EACH_OWN_UNIT",
                LiftedGuard((GuardAtom("OWN_WORKER_X", "GE", 4, arg=0),)),
                AbstractAction("UNIT_REWRITE", match_op=15, unit_op=0),
            ),)),
            InvariantStrategicModule((InvariantRule(
                ModuleSketch("recovery", "INSTANT", "RECOVER_RESOURCE_INVARIANT"), "EACH_OWN_UNIT",
                LiftedGuard((GuardAtom("OWN_INVENTORY", "GE", 1, arg=0),)),
                AbstractAction("UNIT_REWRITE", match_op=17, unit_op=15),
            ),)),
        )
        for module in legal:
            module.validate()

    def test_b4im1_is_fixed_256_byte_roundtrip(self):
        module = self.feed_deferral()
        raw = pack_candidate(module)
        self.assertEqual(B4IM1_CANDIDATE_SIZE, 256)
        self.assertEqual(len(raw), 256)
        restored = unpack_candidate(raw)
        self.assertEqual(module.canonical(), restored.canonical())
        self.assertEqual(module.semantic_id(), restored.semantic_id())

    def test_per_unit_rule_has_no_concrete_unit_identity(self):
        module = self.feed_deferral()
        text = str(module.canonical())
        self.assertIn("EACH_OWN_UNIT", text)
        self.assertNotIn("unit_id", text)
        self.assertNotIn("tile_id", text)
        module.validate(production=True, multi_tape_support=2)

    def test_production_rejects_exact_step_patch(self):
        module = InvariantStrategicModule((
            InvariantRule(
                sketch=ModuleSketch("route", "INSTANT", "ROUTE_MODE"),
                domain="GLOBAL",
                guard=LiftedGuard((GuardAtom("STEP", "EQ", 192),)),
                action=AbstractAction("ROUTE_OVERRIDE", route_id=70),
            ),
        ))
        with self.assertRaisesRegex(ValueError, "exact STEP"):
            module.validate(production=True, multi_tape_support=3)
        module.validate(production=False, multi_tape_support=1)

    def test_production_requires_multiple_tapes(self):
        with self.assertRaisesRegex(ValueError, "multi-tape"):
            self.feed_deferral().validate(production=True, multi_tape_support=1)

    def test_lifted_unit_rewrite_distills_by_static_unit_expansion(self):
        distilled = distill_invariant_module(self.feed_deferral(), max_units=13, route_slot="slot_19")
        self.assertEqual(distilled["schema_version"], 4)
        self.assertEqual(distilled["feature_registry_version"], 1)
        self.assertEqual(distilled["runtime_reconciler"]["kind"], "invariant_unit_rewrite_v1")
        slots = distilled["runtime_reconciler"]["rewrite_slots"]
        self.assertEqual(len(slots), 13)
        self.assertEqual(len(set(slots)), 13)
        tree_text = str(distilled["tree"])
        self.assertIn("OWN_WORKER_TILE", tree_text)
        self.assertIn("OWN_TILE_CONSECUTIVE_UNFED", tree_text)
        self.assertNotIn("FEATURE_REGISTRY_VERSION = 2", tree_text)

    def test_lifted_unit_feature_distillation_binds_static_unit(self):
        module = InvariantStrategicModule((
            InvariantRule(
                sketch=ModuleSketch("maintenance", "INSTANT", "REDUCE_MAINTENANCE_BACKLOG"),
                domain="EACH_OWN_UNIT",
                guard=LiftedGuard((GuardAtom("OWN_WORKER_X", "GE", 4),)),
                action=AbstractAction("UNIT_REWRITE", match_op=15, unit_op=0),
            ),
        ))
        distilled = distill_invariant_module(module, max_units=2, route_slot="slot_19")
        text = str(distilled["tree"])
        self.assertIn("'arg': {'const': 0, 'type': 'UNIT'}", text)
        self.assertIn("'arg': {'const': 1, 'type': 'UNIT'}", text)

    def test_lifted_inventory_distillation_binds_unit_and_item(self):
        module = InvariantStrategicModule((InvariantRule(
            ModuleSketch("recovery", "INSTANT", "RECOVER_RESOURCE_INVARIANT"),
            "EACH_OWN_UNIT",
            LiftedGuard((GuardAtom("OWN_INVENTORY", "GE", 1, arg=0),)),
            AbstractAction("UNIT_REWRITE", match_op=17, unit_op=15, unit_arg=0),
        ),))
        distilled = distill_invariant_module(module, max_units=2, route_slot="slot_19")
        text = str(distilled["tree"])
        self.assertIn("'args': [{'const': 0, 'type': 'UNIT'}, {'const': 0, 'type': 'ITEM'}]", text)
        self.assertIn("'args': [{'const': 1, 'type': 'UNIT'}, {'const': 0, 'type': 'ITEM'}]", text)

    def test_market_intent_distills_to_ordinary_bkg4_intent(self):
        module = InvariantStrategicModule((
            InvariantRule(
                sketch=ModuleSketch("market", "INSTANT", "CREATE_FINANCING_RUNWAY"),
                domain="GLOBAL",
                guard=LiftedGuard((GuardAtom("OWN_SHED", "GE", 2, arg=5),)),
                action=AbstractAction("MARKET_INTENT", market_op=6, item=5, quantity=2, authority=3, priority=20),
            ),
        ))
        distilled = distill_invariant_module(module, max_units=13, route_slot="slot_19")
        self.assertIn("EmitMarketIntent", str(distilled["tree"]))

    def test_committed_route_has_explicit_exit_guard(self):
        module = InvariantStrategicModule((
            InvariantRule(
                sketch=ModuleSketch("expansion", "COMMITTED", "ACCELERATE_EXPANSION"),
                domain="GLOBAL",
                guard=LiftedGuard((GuardAtom("OWN_MONEY", "GE", 6000),)),
                exit_guard=LiftedGuard((GuardAtom("OWN_MONEY", "LE", 2500),)),
                action=AbstractAction("COMMITTED_ROUTE", route_id=70),
            ),
        ))
        module.validate(production=True, multi_tape_support=2)
        distilled = distill_invariant_module(module, max_units=13, route_slot="slot_19")
        self.assertIn("invariant_mode", str(distilled["blackboard"]))
        self.assertIn("OWN_MONEY", str(distilled["tree"]))

    def test_schema_rejects_domain_action_and_control_combinations_without_semantics(self):
        invalid = (
            InvariantRule(
                ModuleSketch("route", "INSTANT", "ROUTE_MODE"),
                "EACH_OWN_UNIT", LiftedGuard(), AbstractAction("ROUTE_OVERRIDE", route_id=70),
            ),
            InvariantRule(
                ModuleSketch("recovery", "INSTANT", "RECOVER_RESOURCE_INVARIANT"),
                "GLOBAL", LiftedGuard(), AbstractAction("UNIT_REWRITE", match_op=17, unit_op=15),
            ),
            InvariantRule(
                ModuleSketch("route", "COMMITTED", "ROUTE_MODE"),
                "GLOBAL", LiftedGuard(), AbstractAction("ROUTE_OVERRIDE", route_id=70),
                LiftedGuard((GuardAtom("OWN_MONEY", "GE", 1),)),
            ),
        )
        for rule in invalid:
            with self.subTest(rule=rule):
                with self.assertRaisesRegex(ValueError, "domain/action|control/action"):
                    InvariantStrategicModule((rule,)).validate()

    def test_schema_rejects_noncanonical_sketch_and_action_payloads(self):
        bad_sketch = InvariantStrategicModule((InvariantRule(
            ModuleSketch("arbitrary-domain", "INSTANT", "ROUTE_MODE"),
            "GLOBAL", LiftedGuard(), AbstractAction("ROUTE_OVERRIDE", route_id=70),
        ),))
        with self.assertRaisesRegex(ValueError, "effect domain"):
            bad_sketch.validate()

        bad_feed = InvariantStrategicModule((InvariantRule(
            ModuleSketch("recovery", "INSTANT", "RECOVER_RESOURCE_INVARIANT"),
            "EACH_OWN_UNIT", LiftedGuard(),
            AbstractAction("UNIT_REWRITE", match_op=17, unit_op=15, unit_arg=6, quantity=2),
        ),))
        with self.assertRaisesRegex(ValueError, "payload"):
            bad_feed.validate()

        noncanonical = (
            AbstractAction("NOOP", route_id=70),
            AbstractAction("ROUTE_OVERRIDE", route_id=70, match_op=1),
            AbstractAction("UNIT_REWRITE", match_op=17, unit_op=0, market_op=1),
            AbstractAction("MARKET_INTENT", market_op=1, route_id=70),
        )
        for action in noncanonical:
            with self.subTest(action=action):
                with self.assertRaisesRegex(ValueError, "noncanonical"):
                    action.validate()

    def test_unpack_rejects_nonzero_inactive_payload(self):
        raw = bytearray(pack_candidate(self.feed_deferral()))
        raw[-1] = 1
        with self.assertRaisesRegex(ValueError, "noncanonical"):
            unpack_candidate(bytes(raw))

    def test_global_inventory_guard_is_rejected_because_it_cannot_bind_unit_and_item(self):
        module = InvariantStrategicModule((InvariantRule(
            ModuleSketch("route", "INSTANT", "ROUTE_MODE"), "GLOBAL",
            LiftedGuard((GuardAtom("OWN_INVENTORY", "GE", 1, arg=0),)),
            AbstractAction("ROUTE_OVERRIDE", route_id=70),
        ),))
        with self.assertRaisesRegex(ValueError, "OWN_INVENTORY.*bound unit"):
            module.validate()


if __name__ == "__main__":
    unittest.main()
