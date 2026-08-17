from __future__ import annotations

import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "python"))
import bonsai_kg as bkg


def P(tree, bb=None, subtrees=None):
    out = {"schema_version": 3, "blackboard": bb or [], "tree": tree}
    if subtrees is not None:
        out["subtrees"] = subtrees
    return out


def syn(step=0, units=1, x=4, y=4, extra=None):
    values = {
        (bkg.FEATURES["STEP"], 0): step,
        (bkg.FEATURES["OWN_UNITS"], 0): units,
        (bkg.FEATURES["OWN_WORKER_X"], 0): x,
        (bkg.FEATURES["OWN_WORKER_Y"], 0): y,
        (bkg.FEATURES["OWN_WORKER_TILE"], 0): y * 10 + x,
    }
    if extra:
        values.update(extra)
    return bkg.SyntheticObservation(values)


def market(tree, bb=None, step=0):
    return bkg.VM(bkg.compile_policy(P(tree, bb))).tick(syn(step)).action.market


class BonsaiKGV3Finalization(unittest.TestCase):
    def vm(self, tree, bb=None, subtrees=None):
        return bkg.VM(bkg.compile_policy(P(tree, bb, subtrees)))

    def test_01_parallel_halt_cancels_running_market_intent(self):
        tree = {"type": "Parallel", "success_threshold": 1, "failure_threshold": 2, "children": [
            {"type": "Sequence", "children": [
                {"type": "EmitMarketIntent", "action": "HIRE", "priority": 99},
                {"type": "WaitTurns", "turns": 5},
            ]},
            {"type": "Success"},
        ]}
        r = self.vm(tree).tick(syn())
        self.assertEqual(r.status, bkg.STATUS_SUCCESS)
        self.assertEqual(r.action.market, [])

    def test_02_parallel_halt_cancels_running_unit_intent(self):
        tree = {"type": "Parallel", "success_threshold": 1, "failure_threshold": 2, "children": [
            {"type": "Sequence", "children": [
                {"type": "EmitUnitIntent", "unit": 0, "action": "NORTH", "priority": 99},
                {"type": "WaitTurns", "turns": 5},
            ]},
            {"type": "Success"},
        ]}
        r = self.vm(tree).tick(syn())
        self.assertEqual(r.status, bkg.STATUS_SUCCESS)
        self.assertEqual(r.action.units[0][0], bkg.UNIT_OPS["PASS"])

    def test_03_nested_parallel_cancellation_removes_descendant_intent(self):
        inner = {"type": "Parallel", "success_threshold": 2, "failure_threshold": 2, "children": [
            {"type": "Sequence", "children": [
                {"type": "EmitMarketIntent", "action": "HIRE", "priority": 88},
                {"type": "WaitTurns", "turns": 5},
            ]},
            {"type": "WaitTurns", "turns": 5},
        ]}
        tree = {"type": "Parallel", "success_threshold": 1, "failure_threshold": 2,
                "children": [inner, {"type": "Success"}]}
        r = self.vm(tree).tick(syn())
        self.assertEqual((r.status, r.action.market), (bkg.STATUS_SUCCESS, []))

    def test_04_running_parallel_keeps_running_child_intent_this_turn(self):
        tree = {"type": "Parallel", "success_threshold": 2, "failure_threshold": 2, "children": [
            {"type": "Sequence", "children": [
                {"type": "EmitMarketIntent", "action": "HIRE", "priority": 77},
                {"type": "WaitTurns", "turns": 5},
            ]},
            {"type": "WaitTurns", "turns": 5},
        ]}
        r = self.vm(tree).tick(syn())
        self.assertEqual(r.status, bkg.STATUS_RUNNING)
        self.assertEqual(r.action.market, [(bkg.MARKET_OPS["HIRE"], 0, 1)])

    def test_05_successful_parallel_child_intent_survives_success(self):
        tree = {"type": "Parallel", "success_threshold": 1, "failure_threshold": 2, "children": [
            {"type": "EmitMarketIntent", "action": "HIRE", "priority": 20},
            {"type": "WaitTurns", "turns": 5},
        ]}
        r = self.vm(tree).tick(syn())
        self.assertEqual(r.status, bkg.STATUS_SUCCESS)
        self.assertEqual(r.action.market, [(bkg.MARKET_OPS["HIRE"], 0, 1)])

    def test_06_parent_parallel_failure_discards_all_intents(self):
        tree = {"type": "Parallel", "success_threshold": 2, "failure_threshold": 1, "children": [
            {"type": "Sequence", "children": [
                {"type": "EmitMarketIntent", "action": "HIRE", "priority": 20},
                {"type": "WaitTurns", "turns": 5},
            ]},
            {"type": "Failure"},
        ]}
        r = self.vm(tree).tick(syn())
        self.assertEqual((r.status, r.action.market), (bkg.STATUS_FAILURE, []))

    def test_07_halt_cancels_intent_but_runs_on_halted_effect(self):
        bb = [{"name": "released", "type": "I32", "initial": 0}]
        running = {"type": "Sequence", "lifecycle": {
            "on_halted": [{"op": "set", "slot": "released", "value": 1}]
        }, "children": [
            {"type": "EmitMarketIntent", "action": "HIRE", "priority": 99},
            {"type": "WaitTurns", "turns": 5},
        ]}
        tree = {"type": "Parallel", "success_threshold": 1, "failure_threshold": 2,
                "children": [running, {"type": "Success"}]}
        r = self.vm(tree, bb).tick(syn())
        self.assertEqual(r.action.market, [])
        self.assertEqual(r.blackboard, (1,))

    def test_08_move_and_water_reusable_subtree(self):
        subtrees = [{
            "name": "MoveAndWater",
            "parameters": {
                "worker": {"type": "UNIT", "mode": "in"},
                "target": {"type": "TILE", "mode": "in"},
            },
            "tree": {"type": "Sequence", "children": [
                {"type": "MoveUnitTo", "unit": {"binding": "worker"}, "target": {"binding": "target"}},
                {"type": "Condition", "lhs": {"feature": "OWN_TILE_WATERED", "arg": {"binding": "target"}}, "op": "EQ", "rhs": 0},
                {"type": "EmitUnitIntent", "unit": {"binding": "worker"}, "action": "WATER"},
            ]},
        }]
        bb = [{"name": "worker", "type": "UNIT", "initial": 0}, {"name": "target", "type": "TILE", "initial": 44}]
        tree = {"type": "Subtree", "subtree": "MoveAndWater", "bindings": {
            "worker": {"blackboard": "worker"}, "target": {"blackboard": "target"}}}
        vm = self.vm(tree, bb, subtrees)
        at_target = syn(extra={(bkg.FEATURES["OWN_TILE_WATERED"], 44): 0})
        r = vm.tick(at_target)
        self.assertEqual(r.status, bkg.STATUS_SUCCESS)
        self.assertEqual(r.action.units[0][0], bkg.UNIT_OPS["WATER"])

    def test_09_output_ports_are_distinct_and_nested_forwarding_works(self):
        subtrees = [
            {"name": "ProduceTarget", "parameters": {"target": {"type": "TILE", "mode": "out"}},
             "tree": {"type": "SetBlackboard", "slot": {"binding": "target"}, "value": 44}},
            {"name": "ForwardTarget", "parameters": {"dst": {"type": "TILE", "mode": "out"}},
             "tree": {"type": "Subtree", "subtree": "ProduceTarget", "bindings": {"target": {"binding": "dst"}}}},
        ]
        bb = [{"name": "a", "type": "TILE", "initial": -1}, {"name": "b", "type": "TILE", "initial": -1}]
        tree = {"type": "Sequence", "children": [
            {"type": "Subtree", "subtree": "ProduceTarget", "bindings": {"target": {"blackboard": "a"}}},
            {"type": "Subtree", "subtree": "ForwardTarget", "bindings": {"dst": {"blackboard": "b"}}},
            {"type": "Condition", "lhs": {"feature": "OWN_TILE_WATERED", "arg": {"blackboard": "a"}}, "op": "EQ", "rhs": 0},
            {"type": "Condition", "lhs": {"feature": "OWN_TILE_WATERED", "arg": {"blackboard": "b"}}, "op": "EQ", "rhs": 0},
        ]}
        r = self.vm(tree, bb, subtrees).tick(syn())
        self.assertEqual(r.status, bkg.STATUS_SUCCESS)
        self.assertEqual(r.blackboard, (44, 44))
        cp = bkg.compile_policy(P(tree, bb, subtrees))
        self.assertNotIn("subtrees", bkg.decompile_policy(cp))

    def test_10_output_port_rejects_const_feature_and_type_mismatch(self):
        subtrees = [{"name": "Out", "parameters": {"x": {"type": "TILE", "mode": "out"}},
                     "tree": {"type": "SetBlackboard", "slot": {"binding": "x"}, "value": 44}}]
        for actual in (44, {"feature": "OWN_WORKER_TILE", "arg": 0}, {"blackboard": "wrong"}):
            bb = [{"name": "wrong", "type": "UNIT", "initial": 0}]
            with self.assertRaises(bkg.PolicyError):
                bkg.compile_policy(P({"type": "Subtree", "subtree": "Out", "bindings": {"x": actual}}, bb, subtrees))

    def test_11_inout_port_reads_and_writes_same_caller_slot(self):
        subtrees = [{"name": "Bump", "parameters": {"x": {"type": "I32", "mode": "inout"}},
                     "tree": {"type": "AddBlackboard", "slot": {"binding": "x"}, "value": 2}}]
        bb = [{"name": "counter", "type": "I32", "initial": 3}]
        tree = {"type": "Subtree", "subtree": "Bump", "bindings": {"x": {"blackboard": "counter"}}}
        r = self.vm(tree, bb, subtrees).tick(syn())
        self.assertEqual(r.blackboard, (5,))

    def test_12_dynamic_market_item_quantity_valid_and_invalid_fail_locally(self):
        bb = [{"name": "product", "type": "ITEM", "initial": bkg.ITEM_ID["WHEAT"]},
              {"name": "qty", "type": "I32", "initial": 3}]
        tree = {"type": "EmitMarketIntent", "action": "SELL", "item": {"blackboard": "product"}, "quantity": {"blackboard": "qty"}}
        vm = self.vm(tree, bb)
        good = vm.tick(syn())
        self.assertEqual(good.action.market, [(bkg.MARKET_OPS["SELL"], bkg.ITEM_ID["WHEAT"], 3)])
        vm.state.blackboard[0] = bkg.ITEM_ID["GOOSE"]
        bad_item = vm.tick(syn(1))
        self.assertEqual((bad_item.status, bad_item.action.market, bad_item.fault), (bkg.STATUS_FAILURE, [], 0))
        vm.state.blackboard[0] = bkg.ITEM_ID["WHEAT"]
        vm.state.blackboard[1] = 0
        bad_qty = vm.tick(syn(2))
        self.assertEqual((bad_qty.status, bad_qty.action.market, bad_qty.fault), (bkg.STATUS_FAILURE, [], 0))

    def test_13_dynamic_tile_lifecycle_indices(self):
        bb = [{"name": "target", "type": "TILE", "initial": 44}]
        tree = {"type": "Sequence", "children": [
            {"type": "Condition", "lhs": {"feature": "OWN_TILE_CONSECUTIVE_UNWATERED", "arg": {"blackboard": "target"}}, "op": "EQ", "rhs": 3},
            {"type": "Condition", "lhs": {"feature": "OWN_TILE_MAX_LIFESPAN_STEP", "arg": {"blackboard": "target"}}, "op": "EQ", "rhs": 91},
        ]}
        extra = {(bkg.FEATURES["OWN_TILE_CONSECUTIVE_UNWATERED"],44):3, (bkg.FEATURES["OWN_TILE_MAX_LIFESPAN_STEP"],44):91}
        self.assertEqual(self.vm(tree, bb).tick(syn(extra=extra)).status, bkg.STATUS_SUCCESS)

    def test_14_dynamic_worker_feature_index(self):
        bb = [{"name": "worker", "type": "UNIT", "initial": 0}]
        tree = {"type": "Condition", "lhs": {"feature": "OWN_WORKER_TILE", "arg": {"blackboard": "worker"}}, "op": "EQ", "rhs": {"const": 44, "type": "TILE"}}
        self.assertEqual(self.vm(tree, bb).tick(syn()).status, bkg.STATUS_SUCCESS)

    def test_15_dynamic_two_index_own_inventory(self):
        bb = [{"name": "worker", "type": "UNIT", "initial": 0},
              {"name": "item", "type": "ITEM", "initial": bkg.ITEM_ID["WHEAT"]}]
        tree = {"type": "Condition", "lhs": {"feature": "OWN_INVENTORY", "args": [
            {"blackboard": "worker"}, {"blackboard": "item"}]}, "op": "EQ", "rhs": 7}
        extra = {(bkg.FEATURES["OWN_INVENTORY"], 0, bkg.ITEM_ID["WHEAT"]): 7}
        self.assertEqual(self.vm(tree, bb).tick(syn(extra=extra)).status, bkg.STATUS_SUCCESS)

    def test_16_dynamic_unit_none_and_missing_fail_without_fault(self):
        bb = [{"name": "worker", "type": "UNIT", "initial": -1}]
        tree = {"type": "EmitUnitIntent", "unit": {"blackboard": "worker"}, "action": "WATER"}
        vm = self.vm(tree, bb)
        r = vm.tick(syn())
        self.assertEqual((r.status, r.action.units[0][0], r.fault), (bkg.STATUS_FAILURE, bkg.UNIT_OPS["PASS"], 0))
        vm.state.blackboard[0] = 1
        r = vm.tick(syn(1, units=1))
        self.assertEqual((r.status, r.fault), (bkg.STATUS_FAILURE, 0))

    def test_17_bkg3_roundtrip_preserves_semantic_hash(self):
        bb = [{"name":"worker","type":"UNIT","initial":0},{"name":"item","type":"ITEM","initial":0}]
        tree = {"type":"Condition","lhs":{"feature":"OWN_INVENTORY","args":[{"blackboard":"worker"},{"blackboard":"item"}]},"op":"GE","rhs":1}
        cp = bkg.compile_policy(P(tree, bb))
        rebuilt = bkg.compile_policy(bkg.decompile_policy(cp))
        self.assertEqual(cp.semantic_hash, rebuilt.semantic_hash)
        self.assertEqual(bkg.semantic_payload_bytes(cp), bkg.semantic_payload_bytes(rebuilt))

    def test_18_feature_arg_512_exact_and_513_reject(self):
        inv = {"feature": "OWN_INVENTORY", "args": [0, "WHEAT"]}
        lifecycle = {
            "failure_if": {"lhs": inv, "op": "EQ", "rhs": inv},
            "success_if": {"lhs": inv, "op": "EQ", "rhs": inv},
            "skip_if": {"lhs": inv, "op": "EQ", "rhs": inv},
            "while": {"lhs": inv, "op": "EQ", "rhs": inv},
        }
        tree = {"type": "Sequence", "children": [{"type": "Success", "lifecycle": lifecycle} for _ in range(32)]}
        cp = bkg.compile_policy(P(tree))
        self.assertEqual((len(cp.values), len(cp.feature_args), len(cp.predicates)), (256, 512, 128))
        import copy
        bad = copy.deepcopy(cp)
        bad.feature_args.append(bkg.FeatureArgRef(bkg.FEATURE_ARG_KINDS["CONST"], bkg.BB_TYPES["UNIT"], 0, 0))
        with self.assertRaisesRegex(bkg.PolicyError, "metadata count"):
            bkg.validate_compiled(bad)

    def test_19_reactive_sequence_revisited_retry_success_keeps_market_intent(self):
        retry = {"type": "Retry", "n": 2, "child": {"type": "Sequence", "children": [
            {"type": "Condition", "lhs": {"feature": "STEP"}, "op": "GE", "rhs": 1},
            {"type": "EmitMarketIntent", "action": "HIRE", "priority": 90},
        ]}}
        vm = self.vm({"type": "ReactiveSequence", "children": [retry, {"type": "WaitTurns", "turns": 5}]})
        r0 = vm.tick(syn(0))
        self.assertEqual((r0.status, r0.action.market), (bkg.STATUS_RUNNING, []))
        r = vm.tick(syn(1))
        self.assertEqual(r.status, bkg.STATUS_RUNNING)
        self.assertEqual(r.action.market, [(bkg.MARKET_OPS["HIRE"], 0, 1)])

    def test_20_reactive_sequence_revisited_retry_success_keeps_unit_intent(self):
        retry = {"type": "Retry", "n": 2, "child": {"type": "Sequence", "children": [
            {"type": "Condition", "lhs": {"feature": "STEP"}, "op": "GE", "rhs": 1},
            {"type": "EmitUnitIntent", "unit": 0, "action": "NORTH", "priority": 90},
        ]}}
        vm = self.vm({"type": "ReactiveSequence", "children": [retry, {"type": "WaitTurns", "turns": 5}]})
        vm.tick(syn(0))
        r = vm.tick(syn(1))
        self.assertEqual(r.status, bkg.STATUS_RUNNING)
        self.assertEqual(r.action.units[0][0], bkg.UNIT_OPS["NORTH"])

    def test_21_reactive_sequence_natural_completion_to_root_success_keeps_intent(self):
        retry = {"type": "Retry", "n": 2, "child": {"type": "Sequence", "children": [
            {"type": "Condition", "lhs": {"feature": "STEP"}, "op": "GE", "rhs": 1},
            {"type": "EmitMarketIntent", "action": "HIRE", "priority": 90},
        ]}}
        vm = self.vm({"type": "ReactiveSequence", "children": [retry, {"type": "Success"}]})
        vm.tick(syn(0))
        r = vm.tick(syn(1))
        self.assertEqual(r.status, bkg.STATUS_SUCCESS)
        self.assertEqual(r.action.market, [(bkg.MARKET_OPS["HIRE"], 0, 1)])

    def test_22_reactive_natural_terminal_child_does_not_run_on_halted(self):
        bb = [{"name": "halted", "type": "I32", "initial": 0}]
        retry = {"type": "Retry", "n": 2, "lifecycle": {
            "on_halted": [{"op": "set", "slot": "halted", "value": 1}]
        }, "child": {"type": "Sequence", "children": [
            {"type": "Condition", "lhs": {"feature": "STEP"}, "op": "GE", "rhs": 1},
            {"type": "EmitMarketIntent", "action": "HIRE", "priority": 90},
        ]}}
        vm = self.vm({"type": "ReactiveSequence", "children": [retry, {"type": "WaitTurns", "turns": 5}]}, bb)
        vm.tick(syn(0))
        r = vm.tick(syn(1), trace=True)
        self.assertEqual(r.blackboard, (0,))
        self.assertFalse(any(event == "halt" and node == 1 for _, node, event, _ in r.trace))
        self.assertEqual(r.action.market, [(bkg.MARKET_OPS["HIRE"], 0, 1)])

    def test_23_reactive_genuine_preemption_halts_old_running_branch(self):
        bb = [{"name": "released", "type": "I32", "initial": 0}]
        old = {"type": "Sequence", "lifecycle": {
            "on_halted": [{"op": "set", "slot": "released", "value": 1}]
        }, "children": [
            {"type": "EmitMarketIntent", "action": "HIRE", "priority": 80},
            {"type": "WaitTurns", "turns": 5},
        ]}
        tree = {"type": "ReactiveSequence", "children": [
            {"type": "Condition", "lhs": {"feature": "STEP"}, "op": "LT", "rhs": 1}, old]}
        vm = self.vm(tree, bb)
        r0 = vm.tick(syn(0))
        self.assertEqual((r0.status, r0.action.market, r0.blackboard),
                         (bkg.STATUS_RUNNING, [(bkg.MARKET_OPS["HIRE"], 0, 1)], (0,)))
        r1 = vm.tick(syn(1), trace=True)
        self.assertEqual((r1.status, r1.action.market, r1.blackboard), (bkg.STATUS_FAILURE, [], (1,)))
        self.assertTrue(any(event == "halt" for _, _, event, _ in r1.trace))

    def test_24_reactive_selector_revisited_terminal_child_is_not_stale_halted(self):
        bb = [{"name": "halted", "type": "I32", "initial": 0}]
        retry = {"type": "Retry", "n": 2, "lifecycle": {
            "on_halted": [{"op": "set", "slot": "halted", "value": 1}]
        }, "child": {"type": "Failure"}}
        tree = {"type": "ReactiveSelector", "children": [
            retry, {"type": "EmitMarketIntent", "action": "HIRE", "priority": 70}]}
        vm = self.vm(tree, bb)
        self.assertEqual(vm.tick(syn(0)).status, bkg.STATUS_RUNNING)
        r = vm.tick(syn(1), trace=True)
        self.assertEqual((r.status, r.action.market, r.blackboard),
                         (bkg.STATUS_SUCCESS, [(bkg.MARKET_OPS["HIRE"], 0, 1)], (0,)))
        self.assertFalse(any(event == "halt" and node == 1 for _, node, event, _ in r.trace))

    def test_25_tile_item_uses_item_none_for_absent_logical_item(self):
        own_tiles = [[None for _ in range(10)] for _ in range(10)]
        opp_tiles = [[None for _ in range(10)] for _ in range(10)]
        own_tiles[0][1] = "LOCKED"
        own_tiles[0][2] = {"kind": "WEED"}
        own_tiles[0][3] = {"kind": "COOP"}
        own_tiles[0][4] = {"kind": "PASTURE", "animal": None}
        own_tiles[0][5] = {"kind": "COOP", "animal": None}  # escaped/currently absent animal
        own_tiles[0][6] = {"kind": "PLANT", "crop": "WHEAT"}
        own_tiles[0][7] = {"kind": "COOP", "animal": "GOOSE"}
        opp_tiles[0][6] = {"kind": "PLANT", "crop": "WHEAT"}
        opp_tiles[0][7] = {"kind": "COOP", "animal": "GOOSE"}
        obs = bkg.DictObservation({"player": 0, "farms": [{"tiles": own_tiles}, {"tiles": opp_tiles}]})
        f = bkg.FEATURES
        for tile in (0, 1, 2, 3, 4, 5, -1, 100):
            self.assertEqual(obs.feature(f["OWN_TILE_ITEM"], tile), -1)
        self.assertNotEqual(obs.feature(f["OWN_TILE_ITEM"], 0), bkg.ITEM_ID["WHEAT"])
        self.assertEqual(obs.feature(f["OWN_TILE_ITEM"], 6), bkg.ITEM_ID["WHEAT"])
        self.assertEqual(obs.feature(f["OWN_TILE_ITEM"], 7), bkg.ITEM_ID["GOOSE"])
        self.assertEqual(obs.feature(f["OPP_TILE_ITEM"], 6), bkg.ITEM_ID["WHEAT"])
        self.assertEqual(obs.feature(f["OPP_TILE_ITEM"], 7), bkg.ITEM_ID["GOOSE"])
        self.assertEqual(obs.feature(f["OPP_TILE_ITEM"], 0), -1)

    def test_26_sell_of_empty_tile_item_fails_without_fault_or_intent(self):
        tiles = [[None for _ in range(10)] for _ in range(10)]
        obs = {"player": 0, "farms": [{"tiles": tiles}, {"tiles": tiles}]}
        tree = {"type": "EmitMarketIntent", "action": "SELL",
                "item": {"feature": "OWN_TILE_ITEM", "arg": 0}, "quantity": 1}
        r = self.vm(tree).tick(obs)
        self.assertEqual((r.status, r.action.market, r.fault), (bkg.STATUS_FAILURE, [], 0))

    def test_27_compiler_generated_action_ir_roundtrips_canonically(self):
        import tempfile
        bb = [
            {"name": "item", "type": "ITEM", "initial": bkg.ITEM_ID["WHEAT"]},
            {"name": "qty", "type": "I32", "initial": 2},
        ]
        policies = [
            P({"type": "EmitUnitIntent", "unit": 0, "action": "WATER"}),
            P({"type": "EmitUnitIntent", "unit": 0, "action": "PLANT", "item": {"blackboard": "item"}}, bb),
            P({"type": "EmitUnitIntent", "unit": 0, "action": "PICKUP", "item": {"blackboard": "item"}, "quantity": {"blackboard": "qty"}}, bb),
            P({"type": "EmitUnitIntent", "unit": 0, "action": "PLACE", "item": "WHEAT", "quantity": 1}),
            P({"type": "EmitMarketIntent", "action": "HIRE"}),
            P({"type": "EmitMarketIntent", "action": "BUY_LAND"}),
            P({"type": "EmitMarketIntent", "action": "BUY_SEED", "item": {"blackboard": "item"}, "quantity": {"blackboard": "qty"}}, bb),
            P({"type": "EmitMarketIntent", "action": "BUY_PRODUCT", "item": "FERTILIZER", "quantity": 1}),
            P({"type": "EmitMarketIntent", "action": "BUY_ANIMAL", "item": "GOOSE", "quantity": 1}),
            P({"type": "EmitMarketIntent", "action": "SELL", "item": {"blackboard": "item"}, "quantity": {"blackboard": "qty"}}, bb),
        ]
        with tempfile.TemporaryDirectory() as td:
            for i, src in enumerate(policies):
                cp = bkg.compile_policy(src)
                path = Path(td) / f"op_{i}.bkgc"
                bkg.save_compiled(cp, path)
                loaded = bkg.load_compiled(path)
                rebuilt = bkg.compile_policy(bkg.decompile_policy(loaded))
                self.assertEqual(loaded.semantic_hash, rebuilt.semantic_hash)
                self.assertEqual(bkg.semantic_payload_bytes(loaded), bkg.semantic_payload_bytes(rebuilt))


if __name__ == "__main__":
    unittest.main(verbosity=2)
