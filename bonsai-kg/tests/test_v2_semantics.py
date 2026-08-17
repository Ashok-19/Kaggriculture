from __future__ import annotations

import json
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE.parent / "python"))
import bonsai_kg as bkg

NATIVE = ROOT / "scratch/bonsai_kg/validate_compiled"


def P(tree, bb=None, **extra):
    out = {"schema_version": 3, "blackboard": bb or [], "tree": tree}
    out.update(extra)
    return out


def syn(step=0, units=1, x=4, y=4, extra=None):
    values = {
        (bkg.FEATURES["STEP"], 0): step,
        (bkg.FEATURES["OWN_UNITS"], 0): units,
        (bkg.FEATURES["OWN_WORKER_X"], 0): x,
        (bkg.FEATURES["OWN_WORKER_Y"], 0): y,
    }
    if extra:
        values.update(extra)
    return bkg.SyntheticObservation(values)


def all_pass(action):
    return all(op == bkg.UNIT_OPS["PASS"] for op, _, _ in action.units) and not action.market


class BonsaiKGV2Semantics(unittest.TestCase):
    def vm(self, tree, bb=None, **extra):
        return bkg.VM(bkg.compile_policy(P(tree, bb, **extra)))

    def test_01_failed_selector_market_branch_rolls_back(self):
        tree = {"type": "Selector", "children": [
            {"type": "Sequence", "children": [
                {"type": "EmitMarketIntent", "action": "BUY_SEED", "item": "WHEAT", "priority": 100},
                {"type": "Failure"},
            ]},
            {"type": "EmitMarketIntent", "action": "SELL", "item": "WHEAT", "quantity": 1, "priority": 50},
        ]}
        r = self.vm(tree).tick(syn())
        self.assertEqual(r.status, bkg.STATUS_SUCCESS)
        self.assertEqual(r.action.market, [(bkg.MARKET_OPS["SELL"], bkg.ITEM_ID["WHEAT"], 1)])

    def test_02_failed_selector_unit_branch_rolls_back(self):
        tree = {"type": "Selector", "children": [
            {"type": "Sequence", "children": [
                {"type": "EmitUnitIntent", "unit": 0, "action": "NORTH", "priority": 200},
                {"type": "Failure"},
            ]},
            {"type": "EmitUnitIntent", "unit": 0, "action": "SOUTH", "priority": 100},
        ]}
        r = self.vm(tree).tick(syn())
        self.assertEqual(r.action.units[0][0], bkg.UNIT_OPS["SOUTH"])

    def test_03_sequence_emit_then_failure_rolls_back(self):
        tree = {"type": "Sequence", "children": [
            {"type": "EmitMarketIntent", "action": "HIRE", "priority": 1},
            {"type": "Failure"},
        ]}
        r = self.vm(tree).tick(syn())
        self.assertEqual(r.status, bkg.STATUS_FAILURE)
        self.assertTrue(all_pass(r.action))

    def test_04_emit_then_running_commits_current_tick(self):
        tree = {"type": "Sequence", "children": [
            {"type": "EmitMarketIntent", "action": "HIRE", "priority": 1},
            {"type": "WaitTurns", "turns": 3},
        ]}
        v = self.vm(tree)
        r0 = v.tick(syn(0))
        self.assertEqual(r0.status, bkg.STATUS_RUNNING)
        self.assertEqual(r0.action.market, [(bkg.MARKET_OPS["HIRE"], 0, 1)])
        r1 = v.tick(syn(1))
        self.assertEqual(r1.status, bkg.STATUS_RUNNING)
        self.assertEqual(r1.action.market, [])

    def test_05_force_failure_and_invert_rollback_child_success_intents(self):
        emit = {"type": "EmitMarketIntent", "action": "HIRE", "priority": 1}
        for tree in (
            {"type": "ForceFailure", "child": emit},
            {"type": "Invert", "child": emit},
        ):
            r = self.vm(tree).tick(syn())
            self.assertEqual(r.status, bkg.STATUS_FAILURE)
            self.assertTrue(all_pass(r.action))

    def test_06_parallel_child_rollback_does_not_leak(self):
        tree = {"type": "Parallel", "success_threshold": 1, "failure_threshold": 2, "children": [
            {"type": "Sequence", "children": [
                {"type": "EmitMarketIntent", "action": "HIRE", "priority": 99},
                {"type": "Failure"},
            ]},
            {"type": "Success"},
        ]}
        r = self.vm(tree).tick(syn())
        self.assertEqual(r.status, bkg.STATUS_SUCCESS)
        self.assertTrue(all_pass(r.action))

    def test_07_parallel_parent_failure_rolls_back_running_child_intent(self):
        tree = {"type": "Parallel", "success_threshold": 2, "failure_threshold": 1, "children": [
            {"type": "Sequence", "children": [
                {"type": "EmitMarketIntent", "action": "HIRE", "priority": 99},
                {"type": "WaitTurns", "turns": 5},
            ]},
            {"type": "Failure"},
        ]}
        r = self.vm(tree).tick(syn())
        self.assertEqual(r.status, bkg.STATUS_FAILURE)
        self.assertTrue(all_pass(r.action))

    def test_08_fault_after_emit_discards_entire_tick(self):
        bb = [{"name": "x", "type": "I32", "initial": 2147483647}]
        tree = {"type": "Sequence", "children": [
            {"type": "EmitMarketIntent", "action": "HIRE", "priority": 1},
            {"type": "AddBlackboard", "slot": "x", "value": 1},
        ]}
        r = self.vm(tree, bb).tick(syn())
        self.assertEqual((r.status, r.fault), (bkg.STATUS_FAILURE, 2))
        self.assertTrue(all_pass(r.action))

    def test_09_fault_is_sticky_until_reset(self):
        bb = [{"name": "x", "type": "I32", "initial": 2147483647}]
        v = self.vm({"type": "AddBlackboard", "slot": "x", "value": 1}, bb)
        for step in range(3):
            r = v.tick(syn(step))
            self.assertEqual((r.status, r.fault), (bkg.STATUS_FAILURE, 2))
            self.assertTrue(all_pass(r.action))
        self.assertEqual(v.state.ticks, 3)
        v.reset()
        self.assertEqual(v.state.fault, 0)
        self.assertEqual(v.state.ticks, 0)
        self.assertEqual(v.tick(syn(0)).fault, 2)

    def test_10_parallel_never_returns_running_without_running_child(self):
        skipped = {"type": "Success", "lifecycle": {
            "skip_if": {"lhs": {"feature": "ALWAYS"}, "op": "EQ", "rhs": 1}
        }}
        tree = {"type": "Parallel", "success_threshold": 2, "failure_threshold": 2,
                "children": [{"type": "Success"}, skipped, skipped]}
        v = self.vm(tree)
        self.assertEqual([v.tick(syn(i)).status for i in range(3)], [bkg.STATUS_FAILURE] * 3)

    def test_11_none_sentinels_and_dynamic_move_target(self):
        none_bb = [
            {"name": "u", "type": "UNIT", "initial": -1},
            {"name": "t", "type": "TILE", "initial": -1},
        ]
        r = self.vm({"type": "MoveUnitTo", "unit": {"blackboard": "u"}, "target": {"blackboard": "t"}}, none_bb).tick(syn())
        self.assertEqual(r.status, bkg.STATUS_FAILURE)
        self.assertTrue(all_pass(r.action))

        dyn_bb = [
            {"name": "u", "type": "UNIT", "initial": 0},
            {"name": "t", "type": "TILE", "initial": 45},
        ]
        r = self.vm({"type": "MoveUnitTo", "unit": {"blackboard": "u"}, "target": {"blackboard": "t"}}, dyn_bb).tick(syn(x=4, y=4))
        self.assertEqual(r.status, bkg.STATUS_RUNNING)
        self.assertEqual(r.action.units[0][0], bkg.UNIT_OPS["EAST"])

    def test_12_dynamic_blackboard_and_lifecycle_value_refs(self):
        bb = [{"name": "x", "type": "I32", "initial": 0}]
        r = self.vm({"type": "SetBlackboard", "slot": "x", "value": {"feature": "STEP"}}, bb).tick(syn(7))
        self.assertEqual(r.blackboard, (7,))
        r = self.vm({"type": "Success", "lifecycle": {
            "on_success": [{"op": "set", "slot": "x", "value": {"feature": "STEP"}}]
        }}, bb).tick(syn(9))
        self.assertEqual(r.blackboard, (9,))

    def test_13_semantic_hash_ignores_authoring_names_but_tracks_behavior(self):
        a = P({"type": "SetBlackboard", "name": "node_a", "slot": "a", "value": 5},
              [{"name": "a", "type": "I32", "initial": 0}], name="policy_a")
        b = P({"type": "SetBlackboard", "name": "node_b", "slot": "b", "value": 5},
              [{"name": "b", "type": "I32", "initial": 0}], name="policy_b")
        c = P({"type": "SetBlackboard", "name": "node_b", "slot": "b", "value": 6},
              [{"name": "b", "type": "I32", "initial": 0}], name="policy_b")
        ha, hb, hc = map(bkg.policy_hash, (a, b, c))
        self.assertEqual(ha, hb)
        self.assertNotEqual(ha, hc)
        stale = json.loads(json.dumps(a)); stale["semantic_hash"] = "00" * 32
        with self.assertRaises(bkg.PolicyError):
            bkg.compile_policy(stale)

    def test_14_binary_semantic_tamper_rejected_python_and_native(self):
        self.assertTrue(NATIVE.exists(), f"missing native validator: {NATIVE}")
        cp = bkg.compile_policy(P({"type": "WaitTurns", "turns": 2}))
        with tempfile.TemporaryDirectory() as td:
            good = Path(td) / "good.bkgc"; bad = Path(td) / "bad.bkgc"
            bkg.save_compiled(cp, good)
            native = subprocess.run([str(NATIVE), str(good)], cwd=ROOT, text=True, capture_output=True)
            self.assertEqual(native.returncode, 0, native.stderr)
            self.assertEqual(native.stdout.strip().split()[-1], cp.semantic_hash)
            raw = bytearray(good.read_bytes())
            struct.pack_into("<i", raw, bkg.HEADER.size + 12, 3)  # root WaitTurns p0, still structurally valid
            bad.write_bytes(raw)
            with self.assertRaisesRegex(bkg.PolicyError, "semantic hash mismatch"):
                bkg.load_compiled(bad)
            native_bad = subprocess.run([str(NATIVE), str(bad)], cwd=ROOT, text=True, capture_output=True)
            self.assertNotEqual(native_bad.returncode, 0)
            self.assertIn("semantic hash mismatch", native_bad.stderr)

    def test_15_decompile_recompile_preserves_semantic_hash(self):
        src = P({"type": "Sequence", "children": [
            {"type": "SetBlackboard", "slot": "x", "value": {"feature": "STEP"}},
            {"type": "MoveUnitTo", "unit": {"const": 0, "type": "UNIT"}, "target": {"blackboard": "t"}},
        ]}, [
            {"name": "x", "type": "I32", "initial": 0},
            {"name": "t", "type": "TILE", "initial": 45},
        ])
        cp = bkg.compile_policy(src)
        rebuilt = bkg.compile_policy(bkg.decompile_policy(cp))
        self.assertEqual(cp.semantic_hash, rebuilt.semantic_hash)
        self.assertEqual(bkg.semantic_payload_bytes(cp), bkg.semantic_payload_bytes(rebuilt))

    def test_16_subtree_expands_twice_to_strict_runtime_tree(self):
        src = P(
            {"type": "Sequence", "children": [
                {"type": "Subtree", "subtree": "move", "bindings": {"unit": {"const": 0, "type": "UNIT"}, "target": {"blackboard": "t1"}}},
                {"type": "Subtree", "subtree": "move", "bindings": {"unit": {"const": 0, "type": "UNIT"}, "target": {"blackboard": "t2"}}},
            ]},
            [{"name": "t1", "type": "TILE", "initial": 45}, {"name": "t2", "type": "TILE", "initial": 46}],
            subtrees=[{"name": "move", "parameters": {"unit": "UNIT", "target": "TILE"},
                       "tree": {"type": "MoveUnitTo", "unit": {"binding": "unit"}, "target": {"binding": "target"}}}],
        )
        cp = bkg.compile_policy(src)
        self.assertEqual(len(cp.nodes), 3)
        decompiled = bkg.decompile_policy(cp)
        self.assertNotIn("subtrees", decompiled)
        self.assertEqual([x["type"] for x in decompiled["tree"]["children"]], ["MoveUnitTo", "MoveUnitTo"])

        recursive = P({"type": "Success"}, subtrees=[
            {"name": "a", "parameters": {}, "tree": {"type": "Subtree", "subtree": "b", "bindings": {}}},
            {"name": "b", "parameters": {}, "tree": {"type": "Subtree", "subtree": "a", "bindings": {}}},
        ])
        with self.assertRaisesRegex(bkg.PolicyError, "recursive subtree"):
            bkg.compile_policy(recursive)

    def test_17_board_size_contract_rejects_non_10x10_observation(self):
        bad_tiles = [[None] * 10 for _ in range(9)]
        good_tiles = [[None] * 10 for _ in range(10)]
        raw = {"player": 0, "farms": [{"tiles": bad_tiles}, {"tiles": good_tiles}]}
        with self.assertRaisesRegex(bkg.PolicyError, "boardSize=10"):
            bkg.DictObservation(raw)


if __name__ == "__main__":
    unittest.main(verbosity=2)
