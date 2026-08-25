from __future__ import annotations

import json
import unittest
from pathlib import Path

from tools.bkg4_search.bounded_macro import BoundedRouteMacro
from tools.bkg4_search.cuda_io import FRAGMENT_RESULT_HEAD, RESULT, load_fragment_results
from tools.bkg4_search.macro_initiation_cegis import MacroEffectCase, choose_best_gate, evaluate_gate, selected_mixed_exact_cells
from tools.bkg4_search.module_codec import ConditionGene


def _ctx(**values: int):
    return {(name, 0): value for name, value in values.items()}


class BoundedMacroCegisTests(unittest.TestCase):
    def macro(self) -> BoundedRouteMacro:
        return BoundedRouteMacro(
            name="test_macro",
            route_id=216,
            base_route_id=22,
            start_step=156,
            duration=48,
            gate_clauses=((ConditionGene("OPP_UNITS", "GE", 0, 7),),),
            provenance={"source_kind": "fixture", "source_route_id": 435, "fragment_sha256": "abc"},
        )

    def test_macro_is_windowed_latched_and_returns_to_parent(self):
        m = self.macro(); tree = m.specialist_tree(route_override_slot="route_override")
        self.assertEqual((m.start_step, m.end_step), (156, 203))
        text = json.dumps(tree, sort_keys=True)
        self.assertIn('"rhs": 156', text)
        self.assertIn('"rhs": 203', text)
        self.assertIn('"slot": "route_override"', text)
        self.assertIn('"value": 216', text)
        self.assertEqual(text.count('"value": 216'), 1)

    def test_route_document_changes_only_bounded_window(self):
        m = self.macro(); base=[{"farmer":["PASS"],"hands":[],"market":[]} for _ in range(719)]; donor=[{"farmer":["NORTH"],"hands":[],"market":[]} for _ in range(719)]
        route=m.route_document(base,donor)
        self.assertEqual(route["actions"][155],base[155]); self.assertEqual(route["actions"][156],donor[156])
        self.assertEqual(route["actions"][203],donor[203]); self.assertEqual(route["actions"][204],base[204])

    def test_macro_representation_is_deterministic_and_identity_free(self):
        m=self.macro(); self.assertEqual(m.semantic_id,self.macro().semantic_id)
        text=json.dumps(m.canonical(),sort_keys=True).lower()
        for forbidden in ("opponent","team","submission","replay","episode","seed","tape"):
            self.assertNotIn(forbidden,text)
        with self.assertRaises(ValueError):
            BoundedRouteMacro("bad",216,22,156,48,((ConditionGene("OPP_UNITS","GE",0,7),),),{"seed":1}).validate()
        with self.assertRaises(ValueError):
            BoundedRouteMacro("bad",216,22,156,48,((ConditionGene("OPP_UNITS","GE",0,7),),),{"source":"episode_123"}).validate()

    def test_inactive_gate_does_not_authorize_macro(self):
        gate=((ConditionGene("OPP_UNITS","GE",0,7),),)
        cases=[MacroEffectCase("c",100,200,_ctx(OPP_UNITS=5))]
        self.assertEqual(evaluate_gate(gate,cases).activated,0)

    def test_cegis_rejects_harmful_gate_and_counterexample_changes_choice(self):
        broad=((ConditionGene("OPP_UNITS","GE",0,7),),)
        narrow=((ConditionGene("OPP_UNITS","GE",0,7),ConditionGene("OPP_LAND","GE",0,2)),)
        discovery=[MacroEffectCase("p",-5,10,_ctx(OPP_UNITS=8,OPP_LAND=2),support_group="p")]
        first=choose_best_gate((broad,narrow),discovery); self.assertIsNotNone(first); self.assertEqual(first[1].complexity,1)
        with_counterexample=discovery+[MacroEffectCase("h",100,50,_ctx(OPP_UNITS=8,OPP_LAND=1),support_group="h")]
        second=choose_best_gate((broad,narrow),with_counterexample); self.assertIsNotNone(second)
        self.assertEqual(second[0],narrow); self.assertEqual(second[1].harmful,0)

    def test_protected_win_to_loss_dominates_average_margin(self):
        risky=((ConditionGene("OPP_UNITS","GE",0,1),),); safe=((ConditionGene("OPP_UNITS","GE",0,8),),)
        cases=[MacroEffectCase("gain",-100,1000,_ctx(OPP_UNITS=8)),MacroEffectCase("win",10,-1,_ctx(OPP_UNITS=1))]
        chosen=choose_best_gate((risky,safe),cases); self.assertIsNotNone(chosen); self.assertEqual(chosen[0],safe); self.assertEqual(chosen[1].win_to_loss,0)

    def test_mixed_identical_legal_state_cannot_be_safe(self):
        gate=((ConditionGene("OPP_UNITS","GE",0,7),),); sig=(1,2,3)
        cases=[MacroEffectCase("p",-5,10,_ctx(OPP_UNITS=8),sig),MacroEffectCase("h",10,-5,_ctx(OPP_UNITS=8),sig)]
        self.assertEqual(selected_mixed_exact_cells(gate,cases),(('h','p'),))
        self.assertIsNone(choose_best_gate((gate,),cases))

    def test_route435_fragment_matches_p0_1_archive(self):
        import tools.bkg4_search.distill_route435_macro_cegis as d
        if not d.DONOR_PACK.exists(): self.skipTest("local route435 archive unavailable")
        route=d.build_route(); donor=d._donor_actions(); base=d._base_actions()
        self.assertEqual(route["actions"][155],base[155])
        self.assertEqual(route["actions"][156:204],donor[156:204])
        self.assertEqual(route["actions"][204],base[204])
        self.assertEqual(route["active_window"],[156,203])

    def test_route435_final_gate_is_four_condition_public_conjunction(self):
        import tools.bkg4_search.distill_route435_macro_cegis as d
        m=d.macro(); self.assertEqual(len(m.gate_clauses),1); self.assertEqual(len(m.gate_clauses[0]),4)
        self.assertEqual([(c.feature,c.index,c.cmp,c.value) for c in m.gate_clauses[0]],[('SHOP_AT',1,'EQ',5),('OPP_UNITS',0,'GE',7),('OPP_LAND',0,'GE',2),('MARKET_INVENTORY',3,'GE',9990)])

    def test_cuda_fragment_result_parser_handles_macro_rows(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/"macro.b4fr"
            raw=bytearray(FRAGMENT_RESULT_HEAD.pack(b"B4FR1\0\0\0",1,2,2,0,2))
            for row in ((100,50,0,0),(90,80,0,0),(140,50,7,0),(95,80,4,0)):
                raw += RESULT.pack(*row)
            path.write_bytes(raw)
            meta,rows=load_fragment_results(path)
            self.assertEqual(meta["candidates"],2)
            self.assertEqual(rows[1][0],(140,50,7,0))
            self.assertEqual(rows[1][1][3],0)

    def test_cpu_cuda_bounded_macro_decisions_match_when_runners_available(self):
        import subprocess
        import tempfile

        root = Path(__file__).resolve().parents[2]
        cpp = root / "scratch/bonsai_kg/v4_cpp_differential_runner"
        cuda = root / "scratch/bonsai_kg/v4_cuda_differential_runner"
        controller = root / "bonsai-kg/integrations/bkg4_champion/champions/route435_pizza_glut_rebalance_v1/controller.bkgc"
        if not (cpp.exists() and cuda.exists() and controller.exists()):
            self.skipTest("local BKG4 differential runners or P0.2 controller unavailable")
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            seq = td / "macro.bks"
            frames = []
            for step in range(155, 206):
                frames.append((
                    (1, 0, 0, step),       # STEP
                    (18, 1, 0, 5),         # SHOP_AT[1] == PIZZA_SHOP
                    (7, 0, 0, 8),          # OPP_UNITS
                    (9, 0, 0, 2),          # OPP_LAND
                    (12, 3, 0, 9993),      # STRAWBERRY market inventory
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
            self.assertIn(",216,", rows[1])      # STEP 156 authorizes route 216.
            self.assertIn(",216,", rows[48])     # STEP 203 remains inside the window.
            self.assertNotIn(",216,", rows[49])  # STEP 204 returns to parent routing.


if __name__ == "__main__": unittest.main(verbosity=2)
