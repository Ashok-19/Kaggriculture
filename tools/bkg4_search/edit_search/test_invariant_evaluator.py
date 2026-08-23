from __future__ import annotations

import copy
import unittest

from tools.bkg4_search.invariant_module import (
    AbstractAction,
    GuardAtom,
    InvariantRule,
    InvariantRuntimeState,
    InvariantStrategicModule,
    LiftedGuard,
    ModuleSketch,
    apply_reference_module,
)


class TinyObservation:
    def __init__(self, unit_tiles, tile_items, unfed=None, pending=None, money=0, shed=None, inventories=None):
        self.unit_tiles = list(unit_tiles)
        self.tile_items = dict(tile_items)
        self.unfed = dict(unfed or {})
        self.pending = dict(pending or {})
        self.money = int(money)
        self.shed = dict(shed or {})
        self.inventories = [dict(x) for x in (inventories or [{} for _ in unit_tiles])]

    def feature(self, name: str, arg: int = 0, arg1: int = 0) -> int:
        if name == "OWN_UNITS": return len(self.unit_tiles)
        if name == "OWN_MONEY": return self.money
        if name == "OWN_WORKER_TILE": return self.unit_tiles[arg]
        if name == "OWN_TILE_HAS_ANIMAL": return int(self.tile_items.get(arg, -1) in (9, 10, 11))
        if name == "OWN_TILE_ITEM": return int(self.tile_items.get(arg, -1))
        if name == "OWN_TILE_CONSECUTIVE_UNFED": return int(self.unfed.get(arg, 0))
        if name == "OWN_TILE_PENDING_CARE_BONUS": return int(self.pending.get(arg, 0))
        if name == "OWN_SHED": return int(self.shed.get(arg, 0))
        if name == "OWN_INVENTORY": return int(self.inventories[arg].get(arg1, 0))
        raise KeyError((name, arg, arg1))


def feed_module():
    return InvariantStrategicModule((InvariantRule(
        ModuleSketch("animal_lifecycle", "INSTANT", "PRESERVE_PRODUCTION_SAVE_FEED"),
        "EACH_OWN_UNIT",
        LiftedGuard((
            GuardAtom("OWN_TILE_HAS_ANIMAL", "EQ", 1),
            GuardAtom("OWN_TILE_ITEM", "EQ", 10),
            GuardAtom("OWN_TILE_CONSECUTIVE_UNFED", "EQ", 0),
            GuardAtom("OWN_TILE_PENDING_CARE_BONUS", "EQ", 0),
        )),
        AbstractAction("UNIT_REWRITE", match_op=15, unit_op=0),
    ),))


class InvariantReferenceEvaluatorTests(unittest.TestCase):
    def test_empty_module_is_exact_action_noop(self):
        action = {"units": [[15, 0, 1], [10, 0, 1]], "market": [[6, 5, 2]]}
        out, stats = apply_reference_module(InvariantStrategicModule(), TinyObservation([], {}), copy.deepcopy(action), InvariantRuntimeState())
        self.assertEqual(out, action)
        self.assertEqual(stats.action_changes, 0)
        self.assertEqual(stats.activations, 0)

    def test_lifted_feed_rule_tracks_semantics_not_unit_identity(self):
        module = feed_module()
        obs_a = TinyObservation([7, 8], {7: 10, 8: -1})
        act_a = {"units": [[15, 0, 1], [15, 0, 1]], "market": []}
        out_a, _ = apply_reference_module(module, obs_a, copy.deepcopy(act_a), InvariantRuntimeState())
        self.assertEqual(out_a["units"], [[0, 0, 1], [15, 0, 1]])

        # Same abstract world with concrete unit identities permuted.
        obs_b = TinyObservation([8, 7], {7: 10, 8: -1})
        act_b = {"units": [[15, 0, 1], [15, 0, 1]], "market": []}
        out_b, _ = apply_reference_module(module, obs_b, copy.deepcopy(act_b), InvariantRuntimeState())
        self.assertEqual(out_b["units"], [[15, 0, 1], [0, 0, 1]])

    def test_lifted_inventory_guard_uses_the_bound_units_inventory(self):
        module = InvariantStrategicModule((InvariantRule(
            ModuleSketch("recovery", "INSTANT", "RECOVER_RESOURCE_INVARIANT"),
            "EACH_OWN_UNIT",
            LiftedGuard((GuardAtom("OWN_INVENTORY", "GE", 1, arg=0),)),
            AbstractAction("UNIT_REWRITE", match_op=17, unit_op=15, unit_arg=0),
        ),))
        observation = TinyObservation(
            [7, 8], {7: 10, 8: 10}, inventories=[{}, {0: 1}],
        )
        action = {"units": [[17, 0, 1], [17, 0, 1]], "market": []}
        out, _ = apply_reference_module(module, observation, copy.deepcopy(action), InvariantRuntimeState())
        self.assertEqual(out["units"], [[17, 0, 1], [15, 0, 1]])

    def test_market_intent_is_bounded_and_precedes_route_market(self):
        module = InvariantStrategicModule((InvariantRule(
            ModuleSketch("market", "INSTANT", "CREATE_FINANCING_RUNWAY"),
            "GLOBAL",
            LiftedGuard((GuardAtom("OWN_SHED", "GE", 2, arg=5),)),
            AbstractAction("MARKET_INTENT", market_op=6, item=5, quantity=2, authority=3, priority=20),
        ),))
        action = {"units": [[0, 0, 1]], "market": [[1, 0, 1]]}
        out, stats = apply_reference_module(module, TinyObservation([0], {}, shed={5: 3}), copy.deepcopy(action), InvariantRuntimeState(), max_market_orders=10)
        self.assertEqual(out["market"][0], [6, 5, 2])
        self.assertEqual(out["market"][1], [1, 0, 1])
        self.assertEqual(stats.activations, 1)
        self.assertEqual(stats.action_changes, 1)

    def test_multiple_market_intents_follow_bkg4_authority_priority_order(self):
        module = InvariantStrategicModule((
            InvariantRule(ModuleSketch("market", "INSTANT", "CREATE_FINANCING_RUNWAY"), "GLOBAL", LiftedGuard(),
                          AbstractAction("MARKET_INTENT", market_op=6, item=5, quantity=1, authority=2, priority=250)),
            InvariantRule(ModuleSketch("market", "INSTANT", "AVOID_OVERBUY"), "GLOBAL", LiftedGuard(),
                          AbstractAction("MARKET_INTENT", market_op=6, item=6, quantity=1, authority=3, priority=1)),
        ))
        action = {"units": [[0, 0, 1]], "market": []}
        out, _ = apply_reference_module(module, TinyObservation([0], {}), copy.deepcopy(action), InvariantRuntimeState())
        self.assertEqual(out["market"], [[6, 6, 1], [6, 5, 1]])

    def test_committed_route_latches_then_invalidates(self):
        module = InvariantStrategicModule((InvariantRule(
            ModuleSketch("expansion", "COMMITTED", "ACCELERATE_EXPANSION"),
            "GLOBAL",
            LiftedGuard((GuardAtom("OWN_MONEY", "GE", 6000),)),
            AbstractAction("COMMITTED_ROUTE", route_id=70),
            LiftedGuard((GuardAtom("OWN_MONEY", "LE", 2500),)),
        ),))
        state = InvariantRuntimeState()
        base = {"units": [[0, 0, 1]], "market": [], "route": 22}
        a, _ = apply_reference_module(module, TinyObservation([0], {}, money=7000), copy.deepcopy(base), state)
        self.assertEqual(a["route"], 70)
        b, _ = apply_reference_module(module, TinyObservation([0], {}, money=4000), copy.deepcopy(base), state)
        self.assertEqual(b["route"], 70)
        c, _ = apply_reference_module(module, TinyObservation([0], {}, money=2000), copy.deepcopy(base), state)
        self.assertEqual(c["route"], 22)


if __name__ == "__main__":
    unittest.main()
