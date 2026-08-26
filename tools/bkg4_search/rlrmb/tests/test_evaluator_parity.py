from __future__ import annotations

import unittest
import struct
import subprocess
import tempfile
import time
from pathlib import Path

from tools.bkg4_search.rlrmb.checkpoints import Checkpoint, EventType, Lane
from tools.bkg4_search.rlrmb.evaluator import (
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
from tools.bkg4_search.rlrmb.ir import (
    ActionFamily,
    ActionSource,
    MacroProgram,
    Phase,
    PublicCondition,
    Termination,
    TerminationType,
)
from tools.bkg4_search.rlrmb.motifs import Motif, motif_from_actions, motif_from_route_window


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


class EvaluatorParityTests(unittest.TestCase):
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

    @unittest.skipUnless(Path("scratch/cuda-engine-build/rlrmb_eval").exists(), "RLRMB CUDA evaluator not built")
    def test_exact_cpp_reference_cuda_and_fragment_oracle(self):
        root = Path.cwd()
        asset = root / "bonsai-kg/integrations/bkg4_champion/champions/bounded_automatic_composer_v1"
        case_source = root / "scratch/p0_7_large_scale_synthesis/stage_a_hardened/artifacts/536dd7c9f4f5b393c99ccf3f/family_00_target.tsv"
        route = root / "scratch/p0_1_macro_search/donors/route_410.json"
        if not case_source.exists() or not route.exists():
            self.skipTest("stored Phase-4 parity evidence unavailable")
        case_lines = case_source.read_text(encoding="utf-8").splitlines()[:2]
        parsed_cases = [line.split("\t") for line in case_lines]
        m = motif_from_route_window(route, 120, 18, tuple(ActionFamily))
        donor = program(phase(termination=Termination(TerminationType.FIXED_CAP, 18), motif_id=m.semantic_id), step=120, cap=18)
        parent = program(phase(ActionSource.PARENT), cap=1)
        expected = [
            evaluate_reference_case(item, {m.semantic_id: m}, opponent_tape=fields[0], seat=int(fields[1]), seed=int(fields[2]), asset_dir=str(asset))
            for item in (parent, donor) for fields in parsed_cases
        ]
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            cases = work / "cases.tsv"; cases.write_text("\n".join(case_lines) + "\n", encoding="utf-8")
            programs = work / "programs.rlrb"; programs.write_bytes(encode_device_batch((parent, donor), {m.semantic_id: m}))
            output = work / "results.rlr"
            command = [str(root / "scratch/cuda-engine-build/rlrmb_eval"), "--controller", str(asset / "controller.bkgc"), "--routes", str(asset / "routes.bkv6"), "--legacy-routes", str(asset / "legacy_routes.bkv6"), "--cases", str(cases), "--programs", str(programs), "--out", str(output), "--feed-slot", "5", "--feed-count", "13", "--strategic-route-slot", "19"]
            subprocess.run(command, cwd=root, check=True, capture_output=True, text=True)
            _, _, actual = decode_device_results(output.read_bytes())
            for cpu, gpu in zip(expected, actual):
                self.assertEqual((cpu.own_money, cpu.opp_money, cpu.action_changes, cpu.fault, cpu.end_reason), (gpu.own_money, gpu.opp_money, gpu.action_changes, gpu.fault, gpu.end_reason))
            self.assertEqual([row.semantic_id for row in actual], [parent.semantic_id] * 2 + [donor.semantic_id] * 2)
            donors = work / "donors.txt"; donors.write_text(str(route) + "\n", encoding="utf-8")
            fragment = work / "fragment.b4fr"
            subprocess.run([str(root / "scratch/cuda-engine-build/bkg4_fragment_eval"), "--controller", str(asset / "controller.bkgc"), "--routes", str(asset / "routes.bkv6"), "--legacy-routes", str(asset / "legacy_routes.bkv6"), "--cases", str(cases), "--donors", str(donors), "--out", str(fragment), "--starts", "120", "--durations", "18", "--feed-slot", "5", "--feed-count", "13", "--strategic-route-slot", "19"], cwd=root, check=True, capture_output=True, text=True)
            raw = fragment.read_bytes()
            old = [struct.unpack_from("<iiHH", raw, 28 + 12 * index) for index in (2, 3)]
            for gpu, oracle in zip(actual[2:], old):
                self.assertEqual((gpu.own_money, gpu.opp_money, gpu.action_changes, int(gpu.fault)), oracle)
            malformed = bytearray(programs.read_bytes()); malformed[4:8] = (99).to_bytes(4, "little")
            bad = work / "bad.rlrb"; bad.write_bytes(malformed)
            failed = subprocess.run([str(root / "scratch/cuda-engine-build/rlrmb_eval"), "--controller", str(asset / "controller.bkgc"), "--routes", str(asset / "routes.bkv6"), "--legacy-routes", str(asset / "legacy_routes.bkv6"), "--cases", str(cases), "--programs", str(bad), "--out", str(output)], cwd=root, capture_output=True, text=True)
            self.assertNotEqual(failed.returncode, 0)
            self.assertIn("version", failed.stderr)

    @unittest.skipUnless(Path("scratch/cuda-engine-build/rlrmb_eval").exists(), "RLRMB CUDA evaluator not built")
    def test_full_cpu_cuda_golden_matrix(self):
        root = Path.cwd()
        asset = root / "bonsai-kg/integrations/bkg4_champion/champions/bounded_automatic_composer_v1"
        case_source = root / "scratch/p0_7_large_scale_synthesis/stage_a_hardened/artifacts/536dd7c9f4f5b393c99ccf3f/family_00_target.tsv"
        if not case_source.exists():
            self.skipTest("stored Phase-4 parity case unavailable")
        case_line = case_source.read_text(encoding="utf-8").splitlines()[0]
        parsed_cases = [case_line.split("\t")]
        shared = motif_from_actions([SOURCE] * 100, tuple(ActionFamily))
        motifs = {shared.semantic_id: shared}
        candidates = []
        for mask in (
            (ActionFamily.FARMER,), (ActionFamily.HANDS,), (ActionFamily.MARKET,),
            (ActionFamily.FARMER, ActionFamily.HANDS), (ActionFamily.FARMER, ActionFamily.MARKET),
            (ActionFamily.HANDS, ActionFamily.MARKET), tuple(ActionFamily),
        ):
            candidates.append(program(phase(families=mask, motif_id=shared.semantic_id), cap=1))
        candidates.extend((
            program(phase(ActionSource.PARENT), cap=1),
            program(phase(ActionSource.SEMANTIC_SPLICE, families=(ActionFamily.MARKET,), motif_id=shared.semantic_id), cap=1),
            program(phase(ActionSource.PARAMETERIZED_PRIMITIVE, families=(ActionFamily.MARKET,), motif_id=None, primitive_name="LIQUIDITY_REBALANCE", parameters={"enabled": True, "mode": "WHEAT", "quantity": 2}), cap=1),
            program(phase(termination=Termination(TerminationType.NEXT_EOD, 3), motif_id=shared.semantic_id), step=22, cap=3),
            program(phase(termination=Termination(TerminationType.NEXT_SHOP_REVEAL, 4), motif_id=shared.semantic_id), step=70, cap=4),
            program(phase(termination=Termination(TerminationType.PURCHASE_COMPLETED, 3, objective="BUY_LAND", target_count=1), motif_id=shared.semantic_id), cap=3),
            program(phase(termination=Termination(TerminationType.TARGET_PUBLIC_STATE, 3, condition=PublicCondition("STEP", "GE", value=1)), motif_id=shared.semantic_id), cap=3),
            program(phase(ActionSource.PARENT, termination=Termination(TerminationType.PARENT_COMPATIBLE, 2)), cap=2),
            program(phase(termination=Termination(TerminationType.MAX_STEPS, 2), motif_id=shared.semantic_id), cap=2),
            program(phase(motif_id=shared.semantic_id), phase(motif_id=shared.semantic_id), cap=2),
            program(*(phase(motif_id=shared.semantic_id) for _ in range(4)), cap=4),
            program(phase(motif_id=shared.semantic_id), step=718, cap=1),
        ))
        short = motif_from_actions([SOURCE], tuple(ActionFamily)); motifs[short.semantic_id] = short
        candidates.extend((
            program(phase(termination=Termination(TerminationType.FIXED_CAP, 2), motif_id=short.semantic_id), cap=2),
            program(phase(ActionSource.PARAMETERIZED_PRIMITIVE, families=(ActionFamily.MARKET,), motif_id=None, primitive_name="UNKNOWN"), cap=1),
        ))
        for route, start, duration, lane in ((435, 156, 48, Lane.WORKER_PRODUCTION), (410, 120, 18, Lane.MARKET_LIQUIDITY), (437, 264, 24, Lane.ANIMAL_TRANSITION)):
            historical = motif_from_route_window(root / f"scratch/p0_1_macro_search/donors/route_{route}.json", start, duration, tuple(ActionFamily))
            motifs[historical.semantic_id] = historical
            candidates.append(MacroProgram(lane, Checkpoint(start, lane, (EventType.FIRST_EFFECT_DIVERGENCE,)), (phase(termination=Termination(TerminationType.FIXED_CAP, duration), motif_id=historical.semantic_id),), duration))
        expected = [
            evaluate_reference_case(item, motifs, opponent_tape=fields[0], seat=int(fields[1]), seed=int(fields[2]), asset_dir=str(asset))
            for item in candidates for fields in parsed_cases
        ]
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            cases = work / "cases.tsv"; cases.write_text(case_line + "\n", encoding="utf-8")
            batch = work / "matrix.rlrb"; batch.write_bytes(encode_device_batch(candidates, motifs))
            output = work / "matrix.rlr"
            command = [str(root / "scratch/cuda-engine-build/rlrmb_eval"), "--controller", str(asset / "controller.bkgc"), "--routes", str(asset / "routes.bkv6"), "--legacy-routes", str(asset / "legacy_routes.bkv6"), "--cases", str(cases), "--programs", str(batch), "--out", str(output), "--feed-slot", "5", "--feed-count", "13", "--strategic-route-slot", "19"]
            started = time.perf_counter()
            first_run = subprocess.run(command, cwd=root, capture_output=True, text=True)
            host_seconds = time.perf_counter() - started
            self.assertEqual(first_run.returncode, 1)
            print(f"phase4_batch_profile host_s={host_seconds:.6f} {first_run.stdout.strip()}")
            _, _, actual = decode_device_results(output.read_bytes())
            first_bytes = output.read_bytes()
            self.assertEqual(subprocess.run(command, cwd=root, capture_output=True, text=True).returncode, 1)
            self.assertEqual(output.read_bytes(), first_bytes)
            reversed_batch = work / "reverse.rlrb"; reversed_batch.write_bytes(encode_device_batch(tuple(reversed(candidates)), motifs))
            reverse_output = work / "reverse.rlr"
            reverse_command = list(command); reverse_command[reverse_command.index(str(batch))] = str(reversed_batch); reverse_command[reverse_command.index(str(output))] = str(reverse_output)
            self.assertEqual(subprocess.run(reverse_command, cwd=root, capture_output=True, text=True).returncode, 1)
            _, _, reverse_rows = decode_device_results(reverse_output.read_bytes())
            self.assertEqual(tuple(reversed(reverse_rows)), actual)
            profile_cases = work / "profile_cases.tsv"
            profile_cases.write_text("\n".join(case_source.read_text(encoding="utf-8").splitlines()[:2]) + "\n", encoding="utf-8")
            profile_output = work / "profile.rlr"
            profile_command = list(command); profile_command[profile_command.index(str(cases))] = str(profile_cases); profile_command[profile_command.index(str(output))] = str(profile_output)
            started = time.perf_counter(); profile_run = subprocess.run(profile_command, cwd=root, capture_output=True, text=True); profile_host = time.perf_counter() - started
            self.assertEqual(profile_run.returncode, 1)
            count, case_count, profile_rows = decode_device_results(profile_output.read_bytes())
            self.assertEqual((count, case_count, len(profile_rows)), (24, 2, 48))
            print(f"phase4_batch_2d_profile host_s={profile_host:.6f} {profile_run.stdout.strip()}")
        self.assertEqual(len(actual), len(expected))
        for index, (cpu, gpu) in enumerate(zip(expected, actual)):
            with self.subTest(candidate=index):
                self.assertEqual(gpu.semantic_id, candidates[index].semantic_id)
                if cpu.fault is not FaultCode.NONE:
                    self.assertEqual(cpu.fault, gpu.fault)
                else:
                    self.assertEqual((cpu.own_money, cpu.opp_money, cpu.action_changes, cpu.fault, cpu.final_phase, cpu.end_reason, cpu.complete), (gpu.own_money, gpu.opp_money, gpu.action_changes, gpu.fault, gpu.final_phase, gpu.end_reason, gpu.complete))


if __name__ == "__main__":
    unittest.main()
