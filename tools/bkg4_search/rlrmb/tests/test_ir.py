from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from tools.bkg4_search.rlrmb.checkpoints import Checkpoint, EventType, Lane
from tools.bkg4_search.rlrmb.ir import (
    ActionFamily,
    ActionSource,
    FallbackPolicy,
    MacroProgram,
    Phase,
    PublicCondition,
    Termination,
    TerminationType,
)
from tools.bkg4_search.rlrmb.motifs import motif_from_actions, motif_from_route_window


DONOR_ROOT = Path("scratch/p0_1_macro_search/donors")


def checkpoint(step: int = 120, lane: Lane = Lane.GENERAL) -> Checkpoint:
    return Checkpoint(step, lane, (EventType.FIRST_EFFECT_DIVERGENCE,))


def fixed(cap: int = 12) -> Termination:
    return Termination(TerminationType.FIXED_CAP, max_steps=cap)


def donor_phase(motif_id: str = "a" * 64, cap: int = 12) -> Phase:
    return Phase(
        ActionSource.DONOR_BLOCK,
        (ActionFamily.FARMER, ActionFamily.MARKET),
        fixed(cap),
        motif_id=motif_id,
    )


def program(*phases: Phase, step: int = 120, lane: Lane = Lane.GENERAL, cap: int = 48) -> MacroProgram:
    return MacroProgram(lane, checkpoint(step, lane), phases, cap)


class OptionProgramIRTests(unittest.TestCase):
    def test_minimal_valid_program_is_deterministic(self) -> None:
        item = program(donor_phase(), cap=12)
        self.assertEqual(item, item)
        self.assertEqual(item.semantic_id, item.semantic_id)
        self.assertEqual(item.to_document(), item.to_document())

    def test_four_phases_are_valid_and_five_reject(self) -> None:
        self.assertEqual(len(program(*(donor_phase(cap=6) for _ in range(4)), cap=24).phases), 4)
        with self.assertRaises(ValueError):
            program(*(donor_phase(cap=6) for _ in range(5)), cap=30)

    def test_zero_phases_rejects(self) -> None:
        with self.assertRaises(ValueError):
            program(cap=12)

    def test_round_trip_preserves_semantics_and_id(self) -> None:
        original = program(donor_phase(), cap=12)
        restored = MacroProgram.from_document(original.to_document())
        self.assertEqual(restored.to_document(), original.to_document())
        self.assertEqual(restored.semantic_id, original.semantic_id)

    def test_phase_order_is_semantic(self) -> None:
        first = donor_phase("a" * 64)
        second = donor_phase("b" * 64)
        self.assertNotEqual(program(first, second).semantic_id, program(second, first).semantic_id)

    def test_material_program_changes_change_identity(self) -> None:
        base_phase = donor_phase("a" * 64)
        base = program(base_phase, cap=12)
        variants = (
            program(donor_phase("b" * 64), cap=12),
            program(Phase(ActionSource.SEMANTIC_SPLICE, base_phase.action_families, fixed(), motif_id="a" * 64), cap=12),
            program(Phase(ActionSource.DONOR_BLOCK, (ActionFamily.HANDS,), fixed(), motif_id="a" * 64), cap=12),
            program(Phase(ActionSource.DONOR_BLOCK, base_phase.action_families, Termination(TerminationType.NEXT_EOD, 12), motif_id="a" * 64), cap=12),
            program(base_phase, step=121, cap=12),
            program(base_phase, lane=Lane.MARKET_LIQUIDITY, cap=12),
            program(base_phase, cap=13),
        )
        self.assertTrue(all(item.semantic_id != base.semantic_id for item in variants))

    def test_mask_order_is_not_semantic(self) -> None:
        left = Phase(ActionSource.DONOR_BLOCK, (ActionFamily.MARKET, ActionFamily.FARMER), fixed(), motif_id="a" * 64)
        right = Phase(ActionSource.DONOR_BLOCK, (ActionFamily.FARMER, ActionFamily.MARKET), fixed(), motif_id="a" * 64)
        self.assertEqual(program(left).semantic_id, program(right).semantic_id)

    def test_mask_precedence_is_explicit(self) -> None:
        phase = Phase(ActionSource.SEMANTIC_SPLICE, (ActionFamily.MARKET,), fixed(), motif_id="a" * 64)
        self.assertEqual(
            phase.family_resolution(),
            {"FARMER": "PARENT", "HANDS": "PARENT", "MARKET": "PHASE"},
        )

    def test_parent_source_needs_no_reference_and_overrides_nothing(self) -> None:
        phase = Phase(ActionSource.PARENT, (), fixed())
        self.assertTrue(program(phase, cap=12))
        self.assertEqual(set(phase.family_resolution().values()), {"PARENT"})
        with self.assertRaises(ValueError):
            Phase(ActionSource.PARENT, (ActionFamily.MARKET,), fixed())
        with self.assertRaises(ValueError):
            Phase(ActionSource.PARENT, (), fixed(), motif_id="a" * 64)

    def test_donor_and_splice_require_semantic_motif_reference(self) -> None:
        for source in (ActionSource.DONOR_BLOCK, ActionSource.SEMANTIC_SPLICE):
            with self.subTest(source=source), self.assertRaises(ValueError):
                Phase(source, (ActionFamily.MARKET,), fixed())
        with self.assertRaises(ValueError):
            Phase(ActionSource.DONOR_BLOCK, (ActionFamily.MARKET,), fixed(), motif_id="route435")

    def test_parameterized_primitive_has_bounded_scalar_parameters(self) -> None:
        phase = Phase(
            ActionSource.PARAMETERIZED_PRIMITIVE,
            (ActionFamily.MARKET,),
            fixed(),
            primitive_name="LIQUIDITY_REBALANCE",
            parameters={"quantity": 2, "enabled": True, "mode": "WHEAT"},
        )
        self.assertTrue(program(phase, cap=12))
        changed = Phase(
            ActionSource.PARAMETERIZED_PRIMITIVE,
            (ActionFamily.MARKET,),
            fixed(),
            primitive_name="LIQUIDITY_REBALANCE",
            parameters={"quantity": 3, "enabled": True, "mode": "WHEAT"},
        )
        self.assertNotEqual(program(phase, cap=12).semantic_id, program(changed, cap=12).semantic_id)
        with self.assertRaises((TypeError, ValueError)):
            Phase(ActionSource.PARAMETERIZED_PRIMITIVE, (ActionFamily.MARKET,), fixed(), primitive_name="X", parameters={"ratio": 1.5})
        with self.assertRaises((TypeError, ValueError)):
            Phase(ActionSource.PARAMETERIZED_PRIMITIVE, (ActionFamily.MARKET,), fixed(), primitive_name="X", parameters={"nested": {"x": 1}})
        with self.assertRaises(ValueError):
            Phase(ActionSource.PARAMETERIZED_PRIMITIVE, (ActionFamily.MARKET,), fixed(), primitive_name="X", parameters={"mode": "REPLAY_ID"})

    def test_all_termination_types_validate(self) -> None:
        cases = (
            Termination(TerminationType.FIXED_CAP, max_steps=6),
            Termination(TerminationType.NEXT_EOD, max_steps=24),
            Termination(TerminationType.NEXT_SHOP_REVEAL, max_steps=72),
            Termination(TerminationType.PURCHASE_COMPLETED, max_steps=24, objective="BUY_LAND", target_count=1),
            Termination(TerminationType.TARGET_PUBLIC_STATE, max_steps=24, condition=PublicCondition("OWN_MONEY", "GE", value=4000)),
            Termination(TerminationType.PARENT_COMPATIBLE, max_steps=24),
            Termination(TerminationType.MAX_STEPS, max_steps=24),
        )
        self.assertEqual({item.kind for item in cases}, set(TerminationType))
        for item in cases:
            self.assertEqual(Termination.from_document(item.to_document()), item)

    def test_boundedness_and_remaining_horizon_fail_closed(self) -> None:
        with self.assertRaises(ValueError):
            fixed(0)
        with self.assertRaises(ValueError):
            program(donor_phase(cap=13), cap=12)
        with self.assertRaises(ValueError):
            program(donor_phase(cap=19), step=700, cap=20)
        self.assertTrue(program(donor_phase(cap=19), step=700, cap=19))

    def test_lane_and_checkpoint_must_match(self) -> None:
        with self.assertRaises(ValueError):
            MacroProgram(Lane.GENERAL, checkpoint(lane=Lane.MARKET_LIQUIDITY), (donor_phase(),), 12)

    def test_identity_leakage_and_unsupported_sources_reject(self) -> None:
        with self.assertRaises(ValueError):
            Phase(ActionSource.PARAMETERIZED_PRIMITIVE, (ActionFamily.MARKET,), fixed(), primitive_name="X", parameters={"replay_id": "R1"})
        with self.assertRaises(ValueError):
            PublicCondition("replay_id", "EQ", value=1)
        with self.assertRaises(ValueError):
            motif_from_actions([{"farmer": ["PASS"], "episode_id": "e1"}], (ActionFamily.FARMER,))
        document = donor_phase().to_document()
        document["action_source"] = "UNKNOWN"
        with self.assertRaises(ValueError):
            Phase.from_document(document)

    def test_own_seed_is_a_legal_public_condition(self) -> None:
        condition = PublicCondition("OWN_SEED", "GE", index=0, value=2)
        term = Termination(TerminationType.TARGET_PUBLIC_STATE, max_steps=12, condition=condition)
        self.assertEqual(term.condition, condition)

    def test_branch_abort_and_recovery_documents_reject(self) -> None:
        base = program(donor_phase(), cap=12).to_document()
        for field in ("branches", "abort", "recovery"):
            with self.subTest(field=field):
                malformed = dict(base)
                malformed[field] = []
                with self.assertRaises(ValueError):
                    MacroProgram.from_document(malformed)

    def test_unknown_schema_and_termination_reject(self) -> None:
        with self.assertRaises(ValueError):
            MacroProgram(Lane.GENERAL, checkpoint(), (donor_phase(),), 12, schema_version=2)
        document = fixed().to_document()
        document["kind"] = "FOREVER"
        with self.assertRaises(ValueError):
            Termination.from_document(document)

    def test_motif_path_independence_and_content_sensitivity(self) -> None:
        actions = [{"farmer": ["PASS"], "market": [["HIRE"]]}]
        left = motif_from_actions(actions, (ActionFamily.FARMER, ActionFamily.MARKET), provenance=("source=/a/route.json",))
        right = motif_from_actions(actions, (ActionFamily.MARKET, ActionFamily.FARMER), provenance=("source=/b/route.json",))
        changed = motif_from_actions([{"farmer": ["EAST"], "market": [["HIRE"]]}], (ActionFamily.FARMER, ActionFamily.MARKET))
        self.assertEqual(left.semantic_id, right.semantic_id)
        self.assertNotEqual(left.semantic_id, changed.semantic_id)
        self.assertNotIn("/a/route.json", json.dumps(left.semantic_document()))

    def test_identical_route_windows_share_motif_identity(self) -> None:
        route = {"actions": [{"market": [["HIRE"]]}, {"farmer": ["PASS"]}]}
        with tempfile.TemporaryDirectory() as directory:
            left = Path(directory) / "left.json"
            right = Path(directory) / "right.json"
            left.write_text(json.dumps(route), encoding="utf-8")
            right.write_text(json.dumps(route), encoding="utf-8")
            a = motif_from_route_window(left, 0, 2, (ActionFamily.MARKET,))
            b = motif_from_route_window(right, 0, 2, (ActionFamily.MARKET,))
        self.assertEqual(a.semantic_id, b.semantic_id)

    def _assert_historical_motif(self, route: int, start: int, duration: int, lane: Lane) -> None:
        path = DONOR_ROOT / f"route_{route}.json"
        if not path.exists():
            self.skipTest(f"historical donor unavailable: {path}")
        motif = motif_from_route_window(path, start, duration, tuple(ActionFamily))
        candidate = MacroProgram(
            lane,
            checkpoint(start, lane),
            (Phase(ActionSource.DONOR_BLOCK, tuple(ActionFamily), fixed(duration), motif_id=motif.semantic_id),),
            duration,
            provenance=(f"route={route}", str(path)),
        )
        restored = MacroProgram.from_document(candidate.to_document())
        self.assertEqual(restored.semantic_id, candidate.semantic_id)
        self.assertNotIn(f"route_{route}", json.dumps(candidate.to_document()))

    def test_route435_motif_representation(self) -> None:
        self._assert_historical_motif(435, 156, 48, Lane.WORKER_PRODUCTION)

    def test_route410_motif_representation(self) -> None:
        self._assert_historical_motif(410, 120, 18, Lane.MARKET_LIQUIDITY)

    def test_route437_motif_representation(self) -> None:
        self._assert_historical_motif(437, 264, 24, Lane.ANIMAL_TRANSITION)

    def test_multi_phase_program_escapes_single_fragment_manifold(self) -> None:
        market = motif_from_actions(
            [{"market": [["BUY_PRODUCT", "WHEAT", 2]]}],
            (ActionFamily.MARKET,),
        )
        hands = motif_from_actions(
            [{"hands": [["FEED"]]}],
            (ActionFamily.HANDS,),
        )
        phases = (
            Phase(ActionSource.SEMANTIC_SPLICE, (ActionFamily.MARKET,), Termination(TerminationType.NEXT_EOD, 12), motif_id=market.semantic_id),
            Phase(ActionSource.DONOR_BLOCK, (ActionFamily.HANDS,), Termination(TerminationType.PURCHASE_COMPLETED, 12, objective="BUY_LAND", target_count=1), motif_id=hands.semantic_id),
        )
        candidate = program(*phases, cap=24)
        self.assertEqual(len(candidate.phases), 2)
        self.assertNotEqual(candidate.phases[0].motif_id, candidate.phases[1].motif_id)
        self.assertNotEqual(candidate.phases[0].action_families, candidate.phases[1].action_families)
        self.assertEqual(candidate.semantic_id, MacroProgram.from_document(candidate.to_document()).semantic_id)


if __name__ == "__main__":
    unittest.main()
