from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path

from kaggriculture_bonsai.league.selfplay_generator import (
    SelfplayPolicy,
    generate_selfplay_tapes,
)
from kaggriculture_bonsai.search.aspe import SpecialistProgram
from kaggriculture_bonsai.search.checkpoints import Lane
from kaggriculture_bonsai.search.evaluator import (
    decode_device_results,
    encode_device_batch,
    evaluate_reference_case,
)
from kaggriculture_bonsai.search.ir import (
    ActionFamily,
    ActionSource,
    Continuation,
    ContinuationKind,
    DecisionPoint,
    Phase,
    PublicCondition,
    Termination,
    TerminationType,
)
from kaggriculture_bonsai.search.motifs import motif_from_actions


class SpecialistProgramParityTests(unittest.TestCase):
    def test_reactive_specialist_body_matches_cuda(self):
        root = Path.cwd()
        binary = root / "workspace/build/cuda/rlrmb_eval"
        self.assertTrue(binary.exists(), "run scripts/build_cuda.py before parity tests")
        assets = root / "assets/champion"
        south = motif_from_actions(
            [{"farmer": ["SOUTH"], "hands": [], "market": []}],
            (ActionFamily.FARMER,),
        )
        east = motif_from_actions(
            [{"farmer": ["EAST"], "hands": [], "market": []}],
            (ActionFamily.FARMER,),
        )
        phases = tuple(
            Phase(
                ActionSource.SEMANTIC_SPLICE,
                (ActionFamily.FARMER,),
                Termination(TerminationType.FIXED_CAP, 1),
                motif_id=motif.semantic_id,
            )
            for motif in (south, east)
        )
        decision = DecisionPoint(
            0,
            PublicCondition("OWN_MONEY", "GE", value=0),
            Continuation(ContinuationKind.PHASE, 1),
            Continuation(ContinuationKind.ABORT_TO_PARENT),
        )
        specialist = SpecialistProgram(
            lane=Lane.GENERAL,
            activation_step=0,
            initiation_clauses=((PublicCondition("OWN_MONEY", "GE", value=0),),),
            phases=phases,
            max_total_steps=2,
            decisions=(decision,),
        )
        program = specialist.to_macro_program()
        motifs = {south.semantic_id: south, east.semantic_id: east}

        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            tape = generate_selfplay_tapes(
                work / "tapes",
                seeds=(271828,),
                policies=(SelfplayPolicy("mirror", "champion"),),
                seats=(0,),
                rng_seed=11,
            )[0]
            cases = work / "cases.tsv"
            cases.write_text(f"{tape.path}\t0\t271828\n", encoding="utf-8")
            expected = evaluate_reference_case(
                program,
                motifs,
                opponent_tape=str(tape.path),
                seed=271828,
                seat=0,
                asset_dir=str(assets),
            )
            batch = work / "programs.rlrb"
            batch.write_bytes(encode_device_batch((program,), motifs))
            output = work / "results.rlr"
            run = subprocess.run(
                [
                    str(binary),
                    "--controller", str(assets / "controller.bkgc"),
                    "--routes", str(assets / "routes.bkv6"),
                    "--legacy-routes", str(assets / "legacy_routes.bkv6"),
                    "--cases", str(cases),
                    "--programs", str(batch),
                    "--out", str(output),
                    "--feed-slot", "5",
                    "--feed-count", "13",
                    "--strategic-route-slot", "19",
                ],
                cwd=root,
                capture_output=True,
                text=True,
            )
            self.assertIn(run.returncode, (0, 1), run.stderr)
            _, _, rows = decode_device_results(output.read_bytes())
            actual = rows[0]
            self.assertEqual(
                (
                    expected.own_money,
                    expected.opp_money,
                    expected.action_changes,
                    expected.fault,
                    expected.final_phase,
                    expected.end_reason,
                    expected.complete,
                ),
                (
                    actual.own_money,
                    actual.opp_money,
                    actual.action_changes,
                    actual.fault,
                    actual.final_phase,
                    actual.end_reason,
                    actual.complete,
                ),
            )


if __name__ == "__main__":
    unittest.main()
