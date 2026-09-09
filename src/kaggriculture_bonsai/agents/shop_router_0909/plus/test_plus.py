"""Candidate integrity: exact parent fallback, legal-state routing, gate bounds."""
import contextlib
import copy
import io
import json
import unittest
from pathlib import Path

from kaggriculture_bonsai.runtime import bonsai_kg_v4 as bkg

from ..reference_port.agent import ShopRouter0909Agent
from .agent import ACTIVE, DONOR_ROUTES, ShopRouter0909Plus, controller, route_table


def _engine():
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        from kaggle_environments import make
    return make


def _active():
    return json.loads(ACTIVE.read_text())["candidate"] if ACTIVE.exists() else None


class TestParentFallback(unittest.TestCase):
    def test_donor_table_without_gate_is_the_parent(self):
        make = _engine()
        parent, plus = ShopRouter0909Agent(), ShopRouter0909Plus({})
        steps = 0
        for seat in (0, 1):
            def paired(obs, cfg=None):
                nonlocal steps
                expected = parent(copy.deepcopy(obs), cfg)
                self.assertEqual(plus(copy.deepcopy(obs), cfg), expected)
                steps += 1
                return expected
            agents = ["random", "random"]
            agents[seat] = paired
            env = make("kaggriculture", configuration={"episodeSteps": 720, "seed": 4243}, debug=True)
            env.run(agents)
        self.assertEqual(steps, 2 * 719)

    def test_default_table_is_exactly_the_donor_table(self):
        self.assertEqual(route_table(), {k: v for k, v in DONOR_ROUTES.items() if v})


class TestControllerShape(unittest.TestCase):
    def test_routing_reads_only_shop_and_step_features(self):
        candidate = _active() or {}
        spec = controller(route_table(candidate.get("routes")), candidate.get("endgame"))
        allowed = {"STEP", "SHOP_AT"}
        found = set()

        def walk(node):
            if isinstance(node, dict):
                if "feature" in node:
                    found.add(node["feature"])
                for value in node.values():
                    walk(value)
            elif isinstance(node, list):
                for value in node:
                    walk(value)
        walk(spec["tree"])
        self.assertTrue(found)
        self.assertEqual(found - allowed, set())

    def test_controller_compiles_without_node_state(self):
        candidate = _active() or {}
        compiled = bkg.compile_policy(controller(route_table(candidate.get("routes")), candidate.get("endgame")))
        self.assertEqual(compiled.state_count, 0)
        self.assertLessEqual(len(compiled.nodes), 128)


class TestActiveCandidate(unittest.TestCase):
    def test_active_candidate_matches_recorded_identity(self):
        if not ACTIVE.exists():
            self.skipTest("no frozen candidate yet")
        record = json.loads(ACTIVE.read_text())
        agent = ShopRouter0909Plus(record["candidate"])
        self.assertEqual(agent.identity["semantic_hash"], record["gate_semantic_hash"])
        self.assertEqual(agent.identity["nodes"], record["gate_nodes"])


if __name__ == "__main__":
    unittest.main()
