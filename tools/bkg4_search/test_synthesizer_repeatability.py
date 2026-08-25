from __future__ import annotations

import importlib.util
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from tools.bkg4_search.bounded_macro import (
    BoundedRouteMacro,
    bounded_macro_windows_overlap,
    validate_nonoverlapping_macro_windows,
)
from tools.bkg4_search.module_codec import ConditionGene
import tools.bkg4_search.distill_route410_wheat_liquidity_expansion as d410
import tools.bkg4_search.distill_route435_macro_cegis as d435


class SynthesizerRepeatabilityTests(unittest.TestCase):
    def test_route435_parent_recompiles_identically(self):
        root = Path(__file__).resolve().parents[2]
        build_path = root / "bonsai-kg/integrations/bkg4_champion/build_assets.py"
        spec = importlib.util.spec_from_file_location("bkg4_build_assets_p03_test", build_path)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        parent = json.loads(d410.PARENT.read_text(encoding="utf-8"))
        cp = mod.build_controller(parent)
        self.assertEqual(cp.semantic_hash, "f47ee7034e553e9a835486b283f4f819d857afdc95143a29d84cb07753b7c286")
        self.assertEqual((len(cp.nodes), len(cp.bb), cp.state_count), (363, 24, 0))

    def test_route410_fragment_matches_archive_and_is_bounded(self):
        if not d410.DONOR_PACK.exists():
            self.skipTest("local top-route archive unavailable")
        route = d410.build_route()
        donor = d410._donor_actions()
        base = d410._base_actions()
        self.assertEqual(route["actions"][119], base[119])
        self.assertEqual(route["actions"][120:138], donor[120:138])
        self.assertEqual(route["actions"][138], base[138])
        self.assertEqual(route["active_window"], [120, 137])

    def test_route410_gate_is_compact_public_conjunction(self):
        m = d410.macro()
        self.assertEqual(
            [(c.feature, c.index, c.cmp, c.value) for c in m.gate_clauses[0]],
            [
                ("MARKET_INVENTORY", 0, "LE", 9937),
                ("OPP_MONEY", 0, "LE", 120),
                ("MARKET_INVENTORY", 6, "GE", 9990),
            ],
        )
        text = json.dumps(m.canonical(), sort_keys=True).lower()
        for forbidden in ("opponent", "team", "submission", "replay", "episode", "seed", "tape"):
            self.assertNotIn(forbidden, text)

    def test_two_synthesized_macro_windows_are_explicitly_disjoint(self):
        new = d410.macro()
        old = d435.macro()
        self.assertFalse(bounded_macro_windows_overlap(new, old))
        validate_nonoverlapping_macro_windows((new, old))
        overlap = BoundedRouteMacro(
            "overlap_fixture",
            500,
            22,
            130,
            40,
            ((ConditionGene("OPP_MONEY", "GE", 0, 1),),),
            {"source_kind": "fixture"},
        )
        self.assertTrue(bounded_macro_windows_overlap(new, overlap))
        with self.assertRaisesRegex(ValueError, "overlap"):
            validate_nonoverlapping_macro_windows((new, overlap))

    def test_composed_module_contains_both_macros_without_priority_overlap(self):
        module = d410.build_module()
        routes = {int(x["route_id"]): x["name"] for x in module["strategic_routes"]}
        self.assertEqual(routes[216], d435.macro().name)
        self.assertEqual(routes[217], d410.macro().name)
        module_names = [
            x.get("module") for x in module["tree"]["children"] if isinstance(x, dict)
        ]
        self.assertLess(module_names.index(d435.macro().name), module_names.index(d410.macro().name))
        self.assertEqual(module["manifest"]["parent"], "route435_pizza_glut_rebalance_v1")
        self.assertEqual(module["manifest"]["active_window"], [120, 137])

    def test_cpu_cuda_decisions_match_for_both_synthesized_macros_when_available(self):
        root = Path(__file__).resolve().parents[2]
        cpp = root / "scratch/bonsai_kg/v4_cpp_differential_runner"
        cuda = root / "scratch/bonsai_kg/v4_cuda_differential_runner"
        controller = root / "bonsai-kg/integrations/bkg4_champion/champions/route410_wheat_liquidity_expansion_v1/controller.bkgc"
        if not (cpp.exists() and cuda.exists() and controller.exists()):
            self.skipTest("local differential runners or P0.3 controller unavailable")
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            seq = td / "two_macros.bks"
            frames = []
            for step in range(119, 206):
                frames.append((
                    (1, 0, 0, step),       # STEP
                    (12, 0, 0, 9934),      # WHEAT inventory
                    (5, 0, 0, 115),        # OPP_MONEY
                    (12, 6, 0, 9995),      # MILK inventory
                    (18, 1, 0, 5),         # SHOP_AT[1] == PIZZA
                    (7, 0, 0, 8),          # OPP_UNITS
                    (9, 0, 0, 2),          # OPP_LAND
                    (12, 3, 0, 9993),      # STRAWBERRY inventory
                ))
            with seq.open("w", encoding="utf-8") as f:
                f.write(f"BKS2 2 {len(frames)}\n")
                for frame in frames:
                    f.write(f"{len(frame)}\n")
                    for fid, a0, a1, value in frame:
                        f.write(f"{fid} {a0} {a1} {value}\n")
            manifest = td / "manifest.txt"
            manifest.write_text(f"{controller} {seq}\n", encoding="utf-8")
            cpu = subprocess.run([str(cpp), str(manifest)], text=True, capture_output=True)
            gpu = subprocess.run([str(cuda), str(manifest)], text=True, capture_output=True)
            if gpu.returncode != 0:
                self.skipTest(f"CUDA differential runner unavailable: {gpu.stderr.strip()}")
            self.assertEqual(cpu.returncode, 0, cpu.stderr)
            self.assertEqual(cpu.stdout, gpu.stdout)
            rows = cpu.stdout.splitlines()
            self.assertEqual(len(rows), 87)
            self.assertIn(",217,", rows[1])       # step120
            self.assertIn(",217,", rows[18])      # step137
            self.assertNotIn(",217,", rows[19])   # step138
            self.assertIn(",216,", rows[37])      # step156
            self.assertIn(",216,", rows[84])      # step203
            self.assertNotIn(",216,", rows[85])   # step204


if __name__ == "__main__":
    unittest.main(verbosity=2)
