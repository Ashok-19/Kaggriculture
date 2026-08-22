from __future__ import annotations

import json
import struct
import tempfile
import unittest
from pathlib import Path

from tools.bkg4_search.edit_search.schema import ParentProfile
from tools.bkg4_search.module_codec import ConditionGene, ModuleGenome, RuleGene


class EditSearchParityHarnessTests(unittest.TestCase):
    def test_representative_overlays_are_nonempty_legal_and_structurally_distinct(self):
        from tools.bkg4_search.edit_search.parity import representative_overlays

        routes = (22, 23, 29, 32, 34, 46, 63, 64, 70, 77, 79, 211, 212, 213, 214)
        rows = representative_overlays(routes)
        self.assertGreaterEqual(len(rows), 3)
        self.assertTrue(all(g.rules for g in rows))
        self.assertEqual(len({g.semantic_id() for g in rows}), len(rows))
        self.assertTrue(any(len(g.rules) >= 2 for g in rows))
        for g in rows:
            g.validate(routes)

    def test_b4tp1_transcode_preserves_normalized_actions_and_uint64_seed(self):
        from tools.bkg4_search.edit_search.parity import read_b4tp1, write_b4tp1_from_json

        payload = {
            "source": {"seat": 1, "seed": (1 << 64) - 3},
            "actions": [
                {"farmer": ["PASS"], "hands": [], "market": [["HIRE"]]},
                {"farmer": ["PICKUP", "WHEAT", 2], "hands": [["WEST"]], "market": [["SELL", "WHEAT", 1]]},
            ],
        }
        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / "tape.json"
            dst = Path(td) / "tape.b4tp"
            src.write_text(json.dumps(payload), encoding="utf-8")
            write_b4tp1_from_json(src, dst)
            got = read_b4tp1(dst)
        self.assertEqual(got["source_seat"], 1)
        self.assertEqual(got["source_seed"], (1 << 64) - 3)
        self.assertEqual(got["actions"][0]["market"], [(1, 0, 1)])
        self.assertEqual(got["actions"][1]["units"][0], (5, 0, 2))
        self.assertEqual(got["actions"][1]["market"], [(6, 0, 1)])

    def test_native_cpu_overlay_evaluator_compiles(self):
        import shutil
        import subprocess

        gxx = shutil.which("g++")
        if gxx is None:
            self.skipTest("g++ unavailable")
        root = Path(__file__).resolve().parents[3]
        source = root / "bonsai-kg" / "tests" / "bkg4_edit_overlay_cpp.cpp"
        proc = subprocess.run(
            [gxx, "-std=c++17", "-fsyntax-only", str(source)],
            cwd=root,
            capture_output=True,
            text=True,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)

    def test_materialized_overlay_appends_to_supplied_parent_not_frozen_base(self):
        from tools.bkg4_search.edit_search.parity import materialized_controller_source

        parent = {
            "name": "future-parent",
            "blackboard": [{"name": "slot_1", "type": "U16", "initial": 22}],
            "tree": {"type": "Sequence", "children": [{"type": "SetBlackboard", "slot": "slot_1", "value": 23}]},
        }
        g = ModuleGenome(37, (RuleGene((ConditionGene("OWN_MONEY", "GE", 0, 1000),), 70),))
        parent["blackboard"].append({"name": "effective_route", "type": "U16", "initial": 0})
        source = materialized_controller_source(parent, g, (22, 23, 70), route_slot="effective_route")
        self.assertEqual(source["name"], "future-parent")
        self.assertEqual(source["tree"]["children"][0]["value"], 23)
        module = source["tree"]["children"][-1]
        self.assertEqual(module["provenance"]["genome_id"], g.semantic_id())
        set_route = module["then"]["children"][0]["children"][-1]
        self.assertEqual(set_route["slot"], "effective_route")


if __name__ == "__main__":
    unittest.main()
