from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "python"))
import bonsai_kg as bkg


def P(tree, bb=None):
    return {"schema_version": 3, "blackboard": bb or [], "tree": tree}


def wrap_depth(depth: int):
    node = {"type": "Success"}
    for _ in range(depth - 1):
        node = {"type": "Invert", "child": node}
    return node


def four_preds():
    return {
        "failure_if": {"lhs": {"feature": "STEP"}, "op": "EQ", "rhs": 1000},
        "success_if": {"lhs": {"feature": "STEP"}, "op": "EQ", "rhs": 1001},
        "skip_if": {"lhs": {"feature": "STEP"}, "op": "EQ", "rhs": 1002},
        "while": {"lhs": {"feature": "STEP"}, "op": "LT", "rhs": 1003},
    }


class BonsaiKGV2Limits(unittest.TestCase):
    def assert_rejects(self, policy, text=None):
        with self.assertRaises(bkg.PolicyError) as cm:
            bkg.compile_policy(policy)
        if text is not None:
            self.assertIn(text, str(cm.exception))

    def test_01_depth_24_exact_25_reject(self):
        self.assertEqual(len(bkg.compile_policy(P(wrap_depth(24))).nodes), 24)
        self.assert_rejects(P(wrap_depth(25)), "depth")

    def test_02_nodes_128_exact_129_reject(self):
        exact = P({"type": "Sequence", "children": [{"type": "Success"} for _ in range(127)]})
        self.assertEqual(len(bkg.compile_policy(exact).nodes), 128)
        above = P({"type": "Sequence", "children": [{"type": "Success"} for _ in range(128)]})
        self.assert_rejects(above, "node limit")

    def test_03_state_slots_48_exact_49_reject(self):
        exact = P({"type": "Sequence", "children": [{"type": "WaitTurns", "turns": 1} for _ in range(47)]})
        self.assertEqual(bkg.compile_policy(exact).state_count, 48)
        above = P({"type": "Sequence", "children": [{"type": "WaitTurns", "turns": 1} for _ in range(48)]})
        self.assert_rejects(above, "state slot")

    def test_04_blackboard_32_exact_33_reject(self):
        def bb(n): return [{"name": f"x{i}", "type": "I32", "initial": i} for i in range(n)]
        self.assertEqual(len(bkg.compile_policy(P({"type": "Success"}, bb(32))).bb), 32)
        self.assert_rejects(P({"type": "Success"}, bb(33)), "blackboard")

    def test_05_parallel_children_16_exact_17_reject(self):
        exact = P({"type": "Parallel", "success_threshold": 16, "failure_threshold": 16,
                   "children": [{"type": "Success"} for _ in range(16)]})
        self.assertEqual(bkg.compile_policy(exact).nodes[0].child_count, 16)
        above = P({"type": "Parallel", "success_threshold": 17, "failure_threshold": 17,
                   "children": [{"type": "Success"} for _ in range(17)]})
        self.assert_rejects(above, "2..16")

    def test_06_retry_repeat_timeout_wait_bounds(self):
        for typ, field, high in (("Retry", "n", 255), ("Repeat", "n", 255), ("TurnTimeout", "turns", 720)):
            tree = {"type": typ, field: high, "child": {"type": "Success"}}
            bkg.compile_policy(P(tree))
            bad = copy.deepcopy(tree); bad[field] = high + 1
            self.assert_rejects(P(bad))
        bkg.compile_policy(P({"type": "WaitTurns", "turns": 720}))
        self.assert_rejects(P({"type": "WaitTurns", "turns": 721}))

    def test_07_predicates_128_and_value_refs_256_exact(self):
        children = [{"type": "Success", "lifecycle": four_preds()} for _ in range(32)]
        cp = bkg.compile_policy(P({"type": "Sequence", "children": children}))
        self.assertEqual(len(cp.predicates), 128)
        self.assertEqual(len(cp.values), 256)

        pred_above = children + [{"type": "Success", "lifecycle": {
            "failure_if": {"lhs": {"feature": "STEP"}, "op": "EQ", "rhs": 9}
        }}]
        self.assert_rejects(P({"type": "Sequence", "children": pred_above}), "predicate limit")

        value_above = children + [{"type": "SetBlackboard", "slot": "x", "value": 1}]
        self.assert_rejects(P({"type": "Sequence", "children": value_above},
                              [{"name": "x", "type": "I32", "initial": 0}]), "ValueRef limit")

    def test_08_effects_128_exact_129_reject(self):
        bb = [{"name": "x", "type": "I32", "initial": 0}]
        def tree(n):
            return {"type": "Success", "lifecycle": {"on_success": [{"op": "clear", "slot": "x"} for _ in range(n)]}}
        self.assertEqual(len(bkg.compile_policy(P(tree(128), bb)).effects), 128)
        self.assert_rejects(P(tree(129), bb), "effect limit")

    def test_09_lifecycle_effective_bound_is_state_limit_48(self):
        # MAX_LIFECYCLE=64 is a defensive raw-table cap. A valid tree cannot reach it because
        # every lifecycle node consumes a unique runtime state slot and MAX_STATE=48.
        exact_children = [{"type": "Success", "lifecycle": {}} for _ in range(47)]
        exact = {"type": "Sequence", "lifecycle": {}, "children": exact_children}
        cp = bkg.compile_policy(P(exact))
        self.assertEqual((len(cp.lifecycles), cp.state_count), (48, 48))
        above = {"type": "Sequence", "lifecycle": {},
                 "children": exact_children + [{"type": "Success", "lifecycle": {}}]}
        self.assert_rejects(P(above), "state slot")
        self.assertEqual(bkg.MAX_LIFECYCLE, 64)

    def test_10_child_reference_raw_cap_is_dominated_by_tree_node_cap(self):
        # In a strict 128-node tree, canonical child refs are exactly nodes-1 <=127.
        cp = bkg.compile_policy(P({"type": "Sequence", "children": [{"type": "Success"} for _ in range(127)]}))
        self.assertEqual(len(cp.child_refs), 127)
        unused = copy.deepcopy(cp); unused.child_refs.extend([0] * (bkg.MAX_CHILD_REFS - len(unused.child_refs)))
        with self.assertRaisesRegex(bkg.PolicyError, "child-reference"):
            bkg.validate_compiled(unused)
        over = copy.deepcopy(cp); over.child_refs.extend([0] * (bkg.MAX_CHILD_REFS + 1 - len(over.child_refs)))
        with self.assertRaisesRegex(bkg.PolicyError, "child"):
            bkg.validate_compiled(over)

    def test_11_deferred_intent_capacity_covers_max_reachable_emit_leaves(self):
        # Strict node cap means one composite root + at most 127 emitting leaves in one tick.
        tree = {"type": "Sequence", "children": [
            {"type": "EmitMarketIntent", "action": "HIRE", "priority": i % 256} for i in range(127)
        ]}
        cp = bkg.compile_policy(P(tree))
        self.assertEqual(len(cp.nodes), 128)
        obs = bkg.SyntheticObservation({(bkg.FEATURES["OWN_UNITS"], 0): 1})
        r = bkg.VM(cp).tick(obs, max_market_orders=16)
        self.assertEqual(r.fault, 0)
        self.assertEqual(len(r.action.market), 16)
        self.assertEqual(bkg.MAX_DEFERRED_INTENTS, 128)
        too_many = {"type": "Sequence", "children": [
            {"type": "EmitMarketIntent", "action": "HIRE"} for _ in range(128)
        ]}
        self.assert_rejects(P(too_many), "node limit")


if __name__ == "__main__":
    unittest.main(verbosity=2)
