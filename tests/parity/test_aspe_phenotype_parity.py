from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path

from kaggriculture_bonsai.search.aspe import (
    ScreeningClassification,
    SpecialistProgram,
    classify_sentinel_results,
    encode_specialist_device_batch,
    run_cuda_sentinel,
    select_build_sentinel,
)
from kaggriculture_bonsai.search.aspe.phenotype import SentinelPanel
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


class AspePhenotypeParityTests(unittest.TestCase):
    def test_cuda_gate_and_telemetry_match_cpu_without_changing_outcomes(self):
        root = Path.cwd()
        binary = root / "workspace/build/cuda/rlrmb_eval"
        self.assertTrue(binary.exists(), "run scripts/build_cuda.py before parity tests")
        assets = root / "assets/champion"
        full_panel = select_build_sentinel(root / "workspace/aspe/baseline/manifest.json")
        panel = SentinelPanel(full_panel.source_panel_identity, full_panel.cases[:1])
        south = motif_from_actions(
            [{"farmer": ["SOUTH"], "hands": [], "market": []}],
            (ActionFamily.FARMER,),
        )
        east = motif_from_actions(
            [{"farmer": ["EAST"], "hands": [], "market": []}],
            (ActionFamily.FARMER,),
        )
        phases = tuple(Phase(
            ActionSource.SEMANTIC_SPLICE, (ActionFamily.FARMER,),
            Termination(TerminationType.FIXED_CAP, 1), motif_id=motif.semantic_id,
        ) for motif in (south, east))
        decision = DecisionPoint(
            0, PublicCondition("OWN_MONEY", "GE", value=0),
            Continuation(ContinuationKind.PHASE, 1),
            Continuation(ContinuationKind.ABORT_TO_PARENT),
        )
        active = SpecialistProgram(
            Lane.GENERAL, 0, ((PublicCondition("OWN_MONEY", "GE", value=0),),),
            phases, 2, (decision,),
        )
        inactive = SpecialistProgram(
            Lane.GENERAL, 0, ((PublicCondition("OWN_MONEY", "LE", value=-1),),),
            phases, 2, (decision,),
        )
        same_execution = SpecialistProgram(
            Lane.GENERAL, 0, ((PublicCondition("OPP_MONEY", "GE", value=0),),),
            phases, 2, (decision,),
        )
        motifs = {south.semantic_id: south, east.semantic_id: east}
        candidate_rows, baseline_rows = run_cuda_sentinel(
            (active, active, same_execution, inactive), motifs, panel,
            binary=binary, asset_dir=assets,
        )
        actual = candidate_rows[0]
        case = panel.cases[0]
        expected = evaluate_reference_case(
            active.to_macro_program(), motifs, opponent_tape=case.tape,
            seed=case.seed, seat=case.seat, asset_dir=str(assets),
            initiation_clauses=active.initiation_clauses,
        )
        self.assertEqual(
            (
                actual.own_money, actual.opp_money, actual.action_changes, actual.fault,
                actual.activated, actual.final_phase, actual.end_reason, actual.complete,
                actual.first_activation_step, actual.last_activation_step,
                actual.activation_count, actual.specialist_duration, actual.phase_path,
                actual.phase_usage, actual.branch_path, actual.completion_reason,
                actual.family_farmer_changes, actual.family_hands_changes,
                actual.family_market_changes, actual.action_trace_hash, actual.state_deltas,
            ),
            (
                expected.own_money, expected.opp_money, expected.action_changes, expected.fault,
                expected.activated, expected.final_phase, expected.end_reason, expected.complete,
                expected.first_activation_step, expected.last_activation_step,
                expected.activation_count, expected.specialist_duration, expected.phase_path,
                expected.phase_usage, expected.branch_path, expected.completion_reason,
                expected.family_farmer_changes, expected.family_hands_changes,
                expected.family_market_changes, expected.action_trace_hash, expected.state_deltas,
            ),
        )
        self.assertEqual((actual.activation_count, actual.phase_usage, actual.branch_path), (2, 0b11, 1))
        self.assertEqual(actual.specialist_duration, 2)
        classified = classify_sentinel_results(
            (active, active, same_execution, inactive), panel, candidate_rows, baseline_rows,
        )
        self.assertEqual(classified[1].duplicate_level, "STRUCTURAL")
        self.assertEqual(classified[2].duplicate_level, "EXECUTION")
        self.assertEqual(classified[3].classification, ScreeningClassification.ZERO_ACTIVATION)
        self.assertFalse(candidate_rows[3].activated)
        self.assertEqual(candidate_rows[3].action_changes, 0)
        self.assertEqual(
            (candidate_rows[3].own_money, candidate_rows[3].opp_money),
            (baseline_rows[0].own_money, baseline_rows[0].opp_money),
        )

        # The same body under the legacy v2 unconditional encoding must produce
        # the same game result; v3 adds observation-only gate/telemetry fields.
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            cases = work / "cases.tsv"
            cases.write_text(f"{case.tape}\t{case.seat}\t{case.seed}\n", encoding="utf-8")
            batch = work / "programs.rlrb"
            batch.write_bytes(encode_device_batch((active.to_macro_program(),), motifs))
            output = work / "results.rlr"
            run = subprocess.run([
                str(binary), "--controller", str(assets / "controller.bkgc"),
                "--routes", str(assets / "routes.bkv6"),
                "--legacy-routes", str(assets / "legacy_routes.bkv6"),
                "--cases", str(cases), "--programs", str(batch), "--out", str(output),
                "--feed-slot", "5", "--feed-count", "13", "--strategic-route-slot", "19",
            ], capture_output=True, text=True, cwd=root)
            self.assertIn(run.returncode, (0, 1), run.stderr)
            legacy = decode_device_results(output.read_bytes())[2][0]
        self.assertEqual(
            (legacy.own_money, legacy.opp_money, legacy.action_changes, legacy.fault),
            (actual.own_money, actual.opp_money, actual.action_changes, actual.fault),
        )


if __name__ == "__main__":
    unittest.main()
