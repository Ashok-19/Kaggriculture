from __future__ import annotations

import unittest
import struct
import subprocess
import tempfile
import time
from pathlib import Path

from kaggriculture_bonsai.search.checkpoints import Checkpoint, EventType, Lane
from kaggriculture_bonsai.search.evaluator import (
    FaultCode,
    ReferenceInterpreter,
    TerminationReason,
    decode_program_batch,
    decode_device_results,
    encode_device_batch,
    encode_program_batch,
    evaluate_reference_case,
    merge_action_families,
)
from kaggriculture_bonsai.search.ir import (
    ActionFamily,
    ActionSource,
    Continuation,
    ContinuationKind,
    DecisionPoint,
    MacroProgram,
    Phase,
    PublicCondition,
    Termination,
    TerminationType,
)
from kaggriculture_bonsai.search.motifs import Motif, motif_from_actions, motif_from_route_window


PASS = {"farmer": ["PASS"], "hands": [], "market": []}
PARENT = {"farmer": ["NORTH"], "hands": [["WEST"], ["EAST"]], "market": [["SELL", "WHEAT", 2], ["HIRE"]]}
SOURCE = {"farmer": ["SOUTH"], "hands": [["FEED"], ["CARE"]], "market": [["BUY_LAND"], ["BUY_PRODUCT", "WHEAT", 3]]}


def obs(step: int, *, shops=(), money=3000, hires=0, land=1, seeds=(0, 0, 0, 0, 0)):
    return {
        "step": step,
        "day": step // 24,
        "hour": step % 24,
        "player": 0,
        "farms": [
            {"money": money, "hands": [[0, 0]] * hires, "hires_today": hires, "unlocked_quadrants": ["NW"] * land, "tiles": []},
            {"money": 3000, "hands": [], "hires_today": 0, "unlocked_quadrants": ["NW"], "tiles": []},
        ],
        "private": {"seeds": dict(zip(("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON"), seeds)), "shed": {}, "inventories": []},
        "market": {"inventory": {}, "prices": {}},
        "town": {"unlocked_shops": list(shops)},
    }


def checkpoint(step=0):
    return Checkpoint(step, Lane.GENERAL, (EventType.FIRST_EFFECT_DIVERGENCE,))


def motif(action=SOURCE, families=tuple(ActionFamily), count=4):
    return motif_from_actions([action] * count, families)


def phase(source=ActionSource.DONOR_BLOCK, families=tuple(ActionFamily), termination=None, motif_id="a" * 64, **kwargs):
    if source is ActionSource.PARENT:
        families, motif_id = (), None
    return Phase(source, families, termination or Termination(TerminationType.FIXED_CAP, 1), motif_id=motif_id, **kwargs)


def program(*phases, step=0, cap=16):
    return MacroProgram(Lane.GENERAL, checkpoint(step), phases, cap)


def reactive(phases, decisions, *, recovery=None, step=0, cap=16):
    return MacroProgram(
        Lane.GENERAL, checkpoint(step), tuple(phases), cap,
        schema_version=2, decisions=tuple(decisions), recovery_phase=recovery,
    )


class EvaluatorParityTests(unittest.TestCase):
    def test_reactive_true_false_branch_timing_and_latching(self):
        motifs = [motif({**PASS, "farmer": [direction]}, count=2) for direction in ("NORTH", "SOUTH", "EAST")]
        phases = tuple(phase(families=(ActionFamily.FARMER,), motif_id=item.semantic_id) for item in motifs)
        decision = DecisionPoint(
            0, PublicCondition("OWN_MONEY", "GE", value=3000),
            Continuation(ContinuationKind.PHASE, 1), Continuation(ContinuationKind.PHASE, 2),
        )
        for money, expected in ((3000, ["NORTH", "SOUTH"]), (2999, ["NORTH", "EAST"])):
            with self.subTest(money=money):
                runner = ReferenceInterpreter(reactive(phases, (decision,), cap=2), {item.semantic_id: item for item in motifs})
                seen = []
                for step in range(2):
                    seen.append(runner.action(obs(step, money=money), PARENT)["farmer"][0])
                    runner.observe(obs(step + 1, money=money))
                self.assertEqual(seen, expected)

    def test_reactive_two_decisions_abort_and_recovery(self):
        motifs = [motif({**PASS, "farmer": [direction]}, count=2) for direction in ("NORTH", "SOUTH", "EAST", "WEST")]
        phases = tuple(phase(families=(ActionFamily.FARMER,), motif_id=item.semantic_id) for item in motifs[:3])
        first = DecisionPoint(0, PublicCondition("DAY", "GE", value=0), Continuation(ContinuationKind.PHASE, 1), Continuation(ContinuationKind.ABORT_TO_PARENT))
        second = DecisionPoint(1, PublicCondition("OWN_MONEY", "GE", value=3000), Continuation(ContinuationKind.PHASE, 2), Continuation(ContinuationKind.RECOVERY))
        lookup = {item.semantic_id: item for item in motifs}
        normal = ReferenceInterpreter(reactive(phases, (first, second), recovery=phase(families=(ActionFamily.FARMER,), motif_id=motifs[3].semantic_id), cap=3), lookup)
        recovered = ReferenceInterpreter(reactive(phases, (first, second), recovery=phase(families=(ActionFamily.FARMER,), motif_id=motifs[3].semantic_id), cap=3), lookup)
        normal_seen = []
        recovered_seen = []
        for step in range(3):
            normal_seen.append(normal.action(obs(step, money=3000), PARENT)["farmer"][0]); normal.observe(obs(step + 1, money=3000))
            recovered_seen.append(recovered.action(obs(step, money=1), PARENT)["farmer"][0]); recovered.observe(obs(step + 1, money=1))
        self.assertEqual(normal_seen, ["NORTH", "SOUTH", "EAST"])
        self.assertEqual(recovered_seen, ["NORTH", "SOUTH", "WEST"])
        self.assertEqual(recovered.action(obs(3, money=1), PARENT), PARENT)

        abort = DecisionPoint(0, PublicCondition("OWN_MONEY", "GE", value=3000), Continuation(ContinuationKind.PHASE, 1), Continuation(ContinuationKind.ABORT_TO_PARENT))
        stopped = ReferenceInterpreter(reactive(phases[:2], (abort,), cap=2), lookup)
        stopped.action(obs(0, money=1), PARENT); stopped.observe(obs(1, money=1))
        self.assertEqual(stopped.action(obs(1, money=1), PARENT), PARENT)
        self.assertTrue(stopped.complete)

    def test_reactive_global_cap_precedes_branch(self):
        m = motif(count=1)
        phases = (phase(motif_id=m.semantic_id), phase(motif_id=m.semantic_id))
        decision = DecisionPoint(0, PublicCondition("DAY", "GE", value=0), Continuation(ContinuationKind.PHASE, 1), Continuation(ContinuationKind.ABORT_TO_PARENT))
        runner = ReferenceInterpreter(reactive(phases, (decision,), cap=1), {m.semantic_id: m})
        runner.action(obs(0), PARENT); runner.observe(obs(1))
        self.assertTrue(runner.complete)
        self.assertEqual(runner.action(obs(1), PARENT), PARENT)

    def test_reactive_device_encoding_is_versioned_and_includes_recovery_motif(self):
        main = motif({**PASS, "farmer": ["NORTH"]}, count=1)
        recovery = motif({**PASS, "farmer": ["WEST"]}, count=1)
        phases = (phase(families=(ActionFamily.FARMER,), motif_id=main.semantic_id),)
        decision = DecisionPoint(0, PublicCondition("OWN_MONEY", "GE", value=3000), Continuation(ContinuationKind.ABORT_TO_PARENT), Continuation(ContinuationKind.RECOVERY))
        candidate = reactive(phases, (decision,), recovery=phase(families=(ActionFamily.FARMER,), motif_id=recovery.semantic_id), cap=2)
        payload = encode_device_batch((candidate,), {main.semantic_id: main, recovery.semantic_id: recovery})
        magic, version, programs, phases_count, motifs_count, actions_count = struct.unpack_from("<IIIIII", payload)
        self.assertEqual((magic, version, programs, phases_count, motifs_count, actions_count), (0x524C5234, 2, 1, 2, 2, 2))

    def test_host_encoding_round_trip_and_version_rejection(self):
        m = motif()
        original = program(phase(motif_id=m.semantic_id), cap=1)
        encoded = encode_program_batch([original], {m.semantic_id: m})
        decoded, motifs = decode_program_batch(encoded)
        self.assertEqual(decoded[0].to_document(), original.to_document())
        self.assertEqual(motifs[m.semantic_id].semantic_document(), m.semantic_document())
        malformed = bytearray(encoded)
        malformed[8:12] = (99).to_bytes(4, "little")
        with self.assertRaises(ValueError):
            decode_program_batch(bytes(malformed))
        device = encode_device_batch([original], {m.semantic_id: m})
        self.assertEqual(struct.unpack_from("<II", device), (0x524C5234, 1))

    def test_each_mask_merges_exact_families_without_reordering(self):
        for selected in (
            (ActionFamily.FARMER,), (ActionFamily.HANDS,), (ActionFamily.MARKET,),
            (ActionFamily.FARMER, ActionFamily.HANDS),
            (ActionFamily.FARMER, ActionFamily.MARKET),
            (ActionFamily.HANDS, ActionFamily.MARKET), tuple(ActionFamily),
        ):
            with self.subTest(selected=selected):
                merged = merge_action_families(PARENT, SOURCE, selected)
                for family, key in ((ActionFamily.FARMER, "farmer"), (ActionFamily.HANDS, "hands"), (ActionFamily.MARKET, "market")):
                    self.assertEqual(merged[key], SOURCE[key] if family in selected else PARENT[key])
        self.assertEqual(merge_action_families(PARENT, SOURCE, (ActionFamily.MARKET,))["market"], SOURCE["market"])
        self.assertEqual(merge_action_families(PARENT, SOURCE, (ActionFamily.HANDS,))["hands"], SOURCE["hands"])

    def test_fixed_cap_off_by_one_and_return_to_parent(self):
        m = motif(count=2)
        runner = ReferenceInterpreter(program(phase(termination=Termination(TerminationType.FIXED_CAP, 2), motif_id=m.semantic_id), cap=2), {m.semantic_id: m})
        self.assertEqual(runner.action(obs(0), PARENT), SOURCE)
        runner.observe(obs(1))
        self.assertEqual(runner.action(obs(1), PARENT), SOURCE)
        runner.observe(obs(2))
        self.assertEqual(runner.action(obs(2), PARENT), PARENT)
        self.assertEqual(runner.last_termination, TerminationReason.FIXED_CAP)

    def test_parent_only_exactness(self):
        runner = ReferenceInterpreter(program(phase(ActionSource.PARENT), cap=1), {})
        self.assertEqual(runner.action(obs(0), PARENT), PARENT)
        runner.observe(obs(1))
        self.assertEqual(runner.action(obs(1), PARENT), PARENT)
        self.assertEqual(runner.action_changes, 0)

    def test_next_eod_includes_hour_23_action(self):
        m = motif(count=4)
        runner = ReferenceInterpreter(program(phase(termination=Termination(TerminationType.NEXT_EOD, 4), motif_id=m.semantic_id), step=22, cap=4), {m.semantic_id: m})
        runner.action(obs(22), PARENT); runner.observe(obs(23))
        self.assertTrue(runner.active)
        runner.action(obs(23), PARENT); runner.observe(obs(24))
        self.assertFalse(runner.active)
        self.assertEqual(runner.last_termination, TerminationReason.NEXT_EOD)

    def test_shop_reveal_uses_observable_multiset_growth(self):
        m = motif(count=3)
        runner = ReferenceInterpreter(program(phase(termination=Termination(TerminationType.NEXT_SHOP_REVEAL, 3), motif_id=m.semantic_id), cap=3), {m.semantic_id: m})
        runner.action(obs(0, shops=("BAKERY",)), PARENT); runner.observe(obs(1, shops=("BAKERY",)))
        self.assertTrue(runner.active)
        runner.action(obs(1, shops=("BAKERY",)), PARENT); runner.observe(obs(2, shops=("BAKERY", "BAKERY")))
        self.assertFalse(runner.active)
        self.assertEqual(runner.last_termination, TerminationReason.NEXT_SHOP_REVEAL)

    def test_purchase_requires_observed_effect_not_attempt(self):
        m = motif(count=3)
        term = Termination(TerminationType.PURCHASE_COMPLETED, 3, objective="BUY_LAND", target_count=1)
        runner = ReferenceInterpreter(program(phase(termination=term, motif_id=m.semantic_id), cap=3), {m.semantic_id: m})
        runner.action(obs(0, land=1), PARENT); runner.observe(obs(1, land=1))
        self.assertTrue(runner.active)
        runner.action(obs(1, land=1), PARENT); runner.observe(obs(2, land=2))
        self.assertFalse(runner.active)

    def test_target_public_state_and_own_seed(self):
        m = motif(count=3)
        term = Termination(TerminationType.TARGET_PUBLIC_STATE, 3, condition=PublicCondition("OWN_SEED", "GE", 0, 2))
        runner = ReferenceInterpreter(program(phase(termination=term, motif_id=m.semantic_id), cap=3), {m.semantic_id: m})
        runner.action(obs(0, seeds=(0, 0, 0, 0, 0)), PARENT); runner.observe(obs(1, seeds=(2, 0, 0, 0, 0)))
        self.assertFalse(runner.active)
        self.assertEqual(runner.last_termination, TerminationReason.TARGET_PUBLIC_STATE)

    def test_parent_compatible_is_exact_action_compatibility(self):
        same = motif(PARENT, count=2)
        term = Termination(TerminationType.PARENT_COMPATIBLE, 2)
        runner = ReferenceInterpreter(program(phase(termination=term, motif_id=same.semantic_id), cap=2), {same.semantic_id: same})
        runner.action(obs(0), PARENT); runner.observe(obs(1))
        self.assertFalse(runner.active)
        self.assertEqual(runner.last_termination, TerminationReason.PARENT_COMPATIBLE)

    def test_max_steps_and_event_precedence(self):
        m = motif(count=1)
        term = Termination(TerminationType.NEXT_EOD, 1)
        runner = ReferenceInterpreter(program(phase(termination=term, motif_id=m.semantic_id), step=23, cap=1), {m.semantic_id: m})
        runner.action(obs(23), PARENT); runner.observe(obs(24))
        self.assertEqual(runner.last_termination, TerminationReason.NEXT_EOD)
        hard = ReferenceInterpreter(program(phase(termination=Termination(TerminationType.MAX_STEPS, 1), motif_id=m.semantic_id), cap=1), {m.semantic_id: m})
        hard.action(obs(0), PARENT); hard.observe(obs(1))
        self.assertEqual(hard.last_termination, TerminationReason.MAX_STEPS)

    def test_two_and_four_phase_transition(self):
        motifs = [motif({**PASS, "farmer": [direction]}, count=1) for direction in ("NORTH", "SOUTH", "EAST", "WEST")]
        phases = [phase(families=(ActionFamily.FARMER,), motif_id=item.semantic_id) for item in motifs]
        runner = ReferenceInterpreter(program(*phases, cap=4), {item.semantic_id: item for item in motifs})
        seen = []
        for step in range(4):
            seen.append(runner.action(obs(step), PARENT)["farmer"])
            runner.observe(obs(step + 1))
        self.assertEqual(seen, [["NORTH"], ["SOUTH"], ["EAST"], ["WEST"]])
        self.assertFalse(runner.active)

    def test_motif_exhaustion_and_unsupported_primitive_fault(self):
        m = motif(count=1)
        runner = ReferenceInterpreter(program(phase(termination=Termination(TerminationType.FIXED_CAP, 2), motif_id=m.semantic_id), cap=2), {m.semantic_id: m})
        runner.action(obs(0), PARENT); runner.observe(obs(1))
        self.assertEqual(runner.action(obs(1), PARENT), PARENT)
        self.assertEqual(runner.fault, FaultCode.MOTIF_EXHAUSTED)
        bad = phase(ActionSource.PARAMETERIZED_PRIMITIVE, families=(ActionFamily.MARKET,), motif_id=None, primitive_name="UNKNOWN")
        primitive = ReferenceInterpreter(program(bad, cap=1), {})
        self.assertEqual(primitive.action(obs(0), PARENT), PARENT)
        self.assertEqual(primitive.fault, FaultCode.UNSUPPORTED_PRIMITIVE)
        market_only = motif_from_actions([SOURCE], (ActionFamily.MARKET,))
        incompatible = program(phase(families=(ActionFamily.FARMER,), motif_id=market_only.semantic_id), cap=1)
        with self.assertRaises(ValueError):
            encode_device_batch((incompatible,), {market_only.semantic_id: market_only})

    def test_supported_primitive_and_deterministic_repeat(self):
        item = phase(ActionSource.PARAMETERIZED_PRIMITIVE, families=(ActionFamily.MARKET,), motif_id=None, primitive_name="LIQUIDITY_REBALANCE", parameters={"enabled": True, "mode": "WHEAT", "quantity": 2})
        candidate = program(item, cap=1)
        first = ReferenceInterpreter(candidate, {}).action(obs(0), PARENT)
        second = ReferenceInterpreter(candidate, {}).action(obs(0), PARENT)
        self.assertEqual(first, second)
        self.assertEqual(first["market"], [["BUY_PRODUCT", "WHEAT", 2]])

    def test_horizon_boundary(self):
        m = motif(count=1)
        runner = ReferenceInterpreter(program(phase(motif_id=m.semantic_id), step=718, cap=1), {m.semantic_id: m})
        runner.action(obs(718), PARENT); runner.observe(obs(719))
        self.assertFalse(runner.active)

    def test_fresh_cpu_cuda_parity(self):
        from kaggriculture_bonsai.league.selfplay_generator import SelfplayPolicy, generate_selfplay_tapes

        root = Path.cwd()
        binary = root / "workspace/build/cuda/rlrmb_eval"
        self.assertTrue(binary.exists(), "run scripts/build_cuda.py before parity tests")
        asset = root / "assets/champion"
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            tape = generate_selfplay_tapes(
                work / "tapes", seeds=(314159,),
                policies=(SelfplayPolicy("mirror", "champion"),), seats=(0,), rng_seed=7,
            )[0]
            cases = work / "cases.tsv"
            cases.write_text(f"{tape.path}\t0\t314159\n", encoding="utf-8")
            candidate = program(phase(ActionSource.PARENT), cap=1)
            expected = evaluate_reference_case(
                candidate, {}, opponent_tape=str(tape.path), seat=0,
                seed=314159, asset_dir=str(asset),
            )
            batch = work / "programs.rlrb"
            batch.write_bytes(encode_device_batch((candidate,), {}))
            output = work / "results.rlr"
            command = [
                str(binary), "--controller", str(asset / "controller.bkgc"),
                "--routes", str(asset / "routes.bkv6"),
                "--legacy-routes", str(asset / "legacy_routes.bkv6"),
                "--cases", str(cases), "--programs", str(batch), "--out", str(output),
                "--feed-slot", "5", "--feed-count", "13", "--strategic-route-slot", "19",
            ]
            run = subprocess.run(command, cwd=root, capture_output=True, text=True)
            self.assertIn(run.returncode, (0, 1), run.stderr)
            _, _, rows = decode_device_results(output.read_bytes())
            actual = rows[0]
            self.assertEqual(
                (expected.own_money, expected.opp_money, expected.action_changes,
                 expected.fault, expected.end_reason),
                (actual.own_money, actual.opp_money, actual.action_changes,
                 actual.fault, actual.end_reason),
            )


if __name__ == "__main__":
    unittest.main()
