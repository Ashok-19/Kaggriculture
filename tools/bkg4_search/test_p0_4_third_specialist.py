from __future__ import annotations

import importlib.util
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from tools.bkg4_search.bounded_macro import BoundedRouteMacro, ExactRoutePreemption
from tools.bkg4_search.module_codec import ConditionGene
import tools.bkg4_search.distill_route437_wheat_scarcity_sheep_shift as d437


class ThirdSpecialistSynthesisTests(unittest.TestCase):
    def test_exact_preemption_suppresses_incumbent_latch_on_authorization(self):
        macro = BoundedRouteMacro(
            "route437_wheat_scarcity_sheep_shift_v1",
            218,
            64,
            264,
            24,
            ((ConditionGene("OPP_MONEY", "GE", 0, 15750),),),
            {"source_kind": "fixture"},
        )
        rule = ExactRoutePreemption(
            preemptor_name=macro.name,
            preemptor_route_id=218,
            preempted_name="fert_wool_expansion_v1",
            preempted_route_id=213,
            preempted_active_slot="fert_wool_expansion_v1_active",
            start_step=264,
            end_step=287,
            evidence_kind="exact_cuda_route_authority_conflict",
        )
        tree = rule.specialist_tree(macro, route_override_slot="route_override")
        authorize = tree["children"][0]["then"]["child"]["children"]
        self.assertEqual(authorize[-2], {
            "type": "SetBlackboard",
            "slot": "fert_wool_expansion_v1_active",
            "value": 0,
        })
        self.assertEqual(authorize[-1]["slot"], macro.active_slot)
        self.assertEqual(rule.canonical()["resume_policy"], "both_windows_end_together")

    def test_exact_preemption_rejects_unproven_window_or_preemptor(self):
        macro = BoundedRouteMacro(
            "candidate", 218, 64, 264, 24,
            ((ConditionGene("OPP_MONEY", "GE", 0, 1),),),
            {"source_kind": "fixture"},
        )
        wrong_window = ExactRoutePreemption("candidate", 218, "incumbent", 213, "incumbent_active", 264, 286, "fixture")
        with self.assertRaisesRegex(ValueError, "window"):
            wrong_window.validate(macro)
        wrong_owner = ExactRoutePreemption("other", 218, "incumbent", 213, "incumbent_active", 264, 287, "fixture")
        with self.assertRaisesRegex(ValueError, "preemptor"):
            wrong_owner.validate(macro)

    def test_route437_gate_is_compact_legal_and_causal(self):
        macro = d437.macro()
        self.assertEqual((macro.route_id, macro.base_route_id, macro.start_step, macro.end_step), (218, 64, 264, 287))
        self.assertEqual(
            [(c.feature, c.index, c.cmp, c.value) for c in macro.gate_clauses[0]],
            [
                ("MARKET_PRICE", 0, "GE", 38),
                ("OPP_MONEY", 0, "GE", 15750),
                ("SHOP_AT", 2, "NE", 1),
            ],
        )
        text = json.dumps(macro.canonical(), sort_keys=True).lower()
        for forbidden in ("team", "submission", "replay", "episode", "seed", "tape"):
            self.assertNotIn(forbidden, text)

    def test_route437_fragment_is_exact_and_bounded(self):
        if not d437.DONOR.exists():
            self.skipTest("local route437 donor unavailable")
        route = d437.build_route()
        donor = d437._donor_actions()
        base = d437._base_actions()
        self.assertEqual(route["actions"][263], base[263])
        self.assertEqual(route["actions"][264:288], donor[264:288])
        self.assertEqual(route["actions"][288], base[288])
        self.assertEqual(route["active_window"], [264, 287])

    def test_composed_module_encodes_route213_preemption(self):
        module = d437.build_module()
        routes = {int(row["route_id"]): row["name"] for row in module["strategic_routes"]}
        self.assertEqual(routes[213], "fert_wool_expansion_v1")
        self.assertEqual(routes[216], "route435_pizza_glut_rebalance_v1")
        self.assertEqual(routes[217], "route410_wheat_liquidity_expansion_v1")
        self.assertEqual(routes[218], d437.macro().name)
        modules = [row.get("module") for row in module["tree"]["children"] if isinstance(row, dict)]
        self.assertEqual(modules[-1], d437.macro().name)
        self.assertEqual(module["manifest"]["route_authority_preemption"], d437.preemption().canonical())

    def test_composed_module_compiles_with_one_new_bool_slot(self):
        root = Path(__file__).resolve().parents[2]
        build_path = root / "bonsai-kg/integrations/bkg4_champion/build_assets.py"
        spec = importlib.util.spec_from_file_location("bkg4_build_assets_p04_test", build_path)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        build_assets = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(build_assets)
        compiled = build_assets.build_controller(d437.build_module())
        self.assertEqual(len(compiled.bb), 26)
        self.assertEqual(compiled.state_count, 0)

    def test_cpu_cuda_three_macro_boundaries_and_route213_preemption_match(self):
        root = Path(__file__).resolve().parents[2]
        cpp = root / "scratch/bonsai_kg/v4_cpp_differential_runner"
        cuda = root / "scratch/bonsai_kg/v4_cuda_differential_runner"
        controller = root / "bonsai-kg/integrations/bkg4_champion/champions/route437_wheat_scarcity_sheep_shift_v1/controller.bkgc"
        if not (cpp.exists() and cuda.exists() and controller.exists()):
            self.skipTest("local differential runners or P0.4 controller unavailable")
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            frames = []
            for step in range(119, 290):
                frames.append((
                    (1, 0, 0, step),                         # STEP
                    (12, 0, 0, 9934),                        # WHEAT inventory
                    (12, 6, 0, 9995),                        # MILK inventory
                    (5, 0, 0, 115 if step == 120 else 15750 if step == 264 else 5000),
                    (18, 1, 0, 5),                           # second shop is PIZZA
                    (7, 0, 0, 8),                            # opponent units
                    (9, 0, 0, 2),                            # opponent land
                    (12, 3, 0, 9993),                        # STRAWBERRY inventory
                    (11, 0, 0, 38),                          # WHEAT price
                    (18, 2, 0, 0),                           # third shop is not BRUNCH
                    (12, 8, 0, 10105 if step == 264 else 10000),
                    (12, 7, 0, 9930 if step == 264 else 10000),
                ))
            sequence = td / "three_macros.bks"
            with sequence.open("w", encoding="utf-8") as handle:
                handle.write(f"BKS2 2 {len(frames)}\n")
                for frame in frames:
                    handle.write(f"{len(frame)}\n")
                    for fid, arg0, arg1, value in frame:
                        handle.write(f"{fid} {arg0} {arg1} {value}\n")
            manifest = td / "manifest.txt"
            manifest.write_text(f"{controller} {sequence}\n", encoding="utf-8")
            cpu = subprocess.run([str(cpp), str(manifest)], text=True, capture_output=True)
            gpu = subprocess.run([str(cuda), str(manifest)], text=True, capture_output=True)
            if gpu.returncode != 0:
                self.skipTest(f"CUDA differential runner unavailable: {gpu.stderr.strip()}")
            self.assertEqual(cpu.returncode, 0, cpu.stderr)
            self.assertEqual(cpu.stdout, gpu.stdout)
            rows = cpu.stdout.splitlines()
            self.assertEqual(len(rows), 171)
            self.assertIn(",217,", rows[1])       # step120
            self.assertIn(",217,", rows[18])      # step137
            self.assertNotIn(",217,", rows[19])   # step138
            self.assertIn(",216,", rows[37])      # step156
            self.assertIn(",216,", rows[84])      # step203
            self.assertNotIn(",216,", rows[85])   # step204
            self.assertIn(",218,", rows[145])     # step264 preempts route213
            self.assertNotIn(",213,", rows[145])
            self.assertIn(",218,", rows[168])     # step287
            self.assertNotIn(",218,", rows[169])  # step288 returns to parent


if __name__ == "__main__":
    unittest.main(verbosity=2)
