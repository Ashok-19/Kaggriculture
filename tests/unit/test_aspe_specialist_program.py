from __future__ import annotations

import json
import struct
import unittest

from kaggriculture_bonsai.composition.bonsai_composer import (
    StrategicAuthority,
    compose_specialists,
)
from kaggriculture_bonsai.routing.materialize import MaterializedInterpreter
from kaggriculture_bonsai.runtime import bonsai_kg_v4
from kaggriculture_bonsai.search.aspe import (
    ASPE_PUBLIC_FEATURES,
    SPECIALIST_PROGRAM_SCHEMA_VERSION,
    SpecialistProgram,
    initiation_matches,
    materialize_specialist,
)
from kaggriculture_bonsai.search.checkpoints import Lane
from kaggriculture_bonsai.search.evaluator import (
    TerminationReason,
    decode_program_batch,
    encode_device_batch,
    encode_program_batch,
)
from kaggriculture_bonsai.search.ir import (
    ActionFamily,
    ActionSource,
    Continuation,
    ContinuationKind,
    DecisionPoint,
    FallbackPolicy,
    Phase,
    PublicCondition,
    Termination,
    TerminationType,
)
from kaggriculture_bonsai.search.motifs import motif_from_actions

PASS = {"farmer": ["PASS"], "hands": [], "market": []}
PARENT = {
    "farmer": ["NORTH"],
    "hands": [["WEST"]],
    "market": [["SELL", "WHEAT", 2]],
}


def observation(
    step: int,
    *,
    own_money: int = 4000,
    opp_money: int = 3000,
    own_units: int = 2,
    opp_units: int = 1,
    own_land: int = 2,
    opp_land: int = 1,
    own_hires: int = 1,
    opp_hires: int = 0,
    shops=("BAKERY", "PIZZA_SHOP"),
    wheat_price: int = 31,
    wheat_inventory: int = 8,
    wheat_shed: int = 3,
    wheat_seed: int = 2,
):
    return {
        "step": step,
        "day": step // 24,
        "hour": step % 24,
        "player": 0,
        "farms": [
            {
                "money": own_money,
                "farmer": [0, 0],
                "hands": [[0, 0]] * (own_units - 1),
                "hires_today": own_hires,
                "unlocked_quadrants": ["NW"] * own_land,
                "tiles": [],
            },
            {
                "money": opp_money,
                "farmer": [0, 0],
                "hands": [[0, 0]] * (opp_units - 1),
                "hires_today": opp_hires,
                "unlocked_quadrants": ["NW"] * opp_land,
                "tiles": [],
            },
        ],
        "private": {
            "seeds": {"WHEAT": wheat_seed},
            "shed": {"WHEAT": wheat_shed},
            "inventories": [],
        },
        "market": {
            "prices": {"WHEAT": wheat_price},
            "inventory": {"WHEAT": wheat_inventory},
        },
        "town": {"unlocked_shops": list(shops)},
    }


def motif(direction: str, *, count: int = 4):
    return motif_from_actions(
        [{"farmer": [direction], "hands": [["CARE"]], "market": [["BUY_LAND"]]}] * count,
        tuple(ActionFamily),
    )


def motif_phase(item, *, families=(ActionFamily.FARMER,), cap=1, termination=None):
    return Phase(
        ActionSource.SEMANTIC_SPLICE,
        families,
        termination or Termination(TerminationType.FIXED_CAP, cap),
        motif_id=item.semantic_id,
    )


def market_phase(*, cap=2):
    return Phase(
        ActionSource.PARAMETERIZED_PRIMITIVE,
        (ActionFamily.MARKET,),
        Termination(TerminationType.FIXED_CAP, cap),
        primitive_name="LIQUIDITY_REBALANCE",
        parameters={"enabled": True, "mode": "WHEAT", "quantity": 2},
    )


def specialist(
    *phases,
    clauses=None,
    decisions=(),
    recovery=None,
    cap=None,
    step=48,
):
    return SpecialistProgram(
        lane=Lane.MARKET_LIQUIDITY,
        activation_step=step,
        initiation_clauses=(
            ((PublicCondition("OWN_MONEY", "GE", value=3000),),)
            if clauses is None else clauses
        ),
        phases=tuple(phases) or (market_phase(),),
        max_total_steps=cap or sum(phase.termination.max_steps for phase in phases or (market_phase(),)),
        decisions=tuple(decisions),
        recovery_phase=recovery,
    )


def materialized(program, motifs=()):
    lookup = {item.semantic_id: item for item in motifs}
    artifact = materialize_specialist(
        program,
        lookup,
        route_id=500,
        base_route_id=64,
        route_source="aspe/phase1/fixture.route.json",
        authority="strategic_route",
        authority_slot="slot_1",
    )
    return artifact, MaterializedInterpreter(artifact, lookup)


class SpecialistProgramTests(unittest.TestCase):
    def test_serialization_roundtrip_and_equivalent_identity(self):
        conditions = (
            PublicCondition("OWN_MONEY", "GE", value=3000),
            PublicCondition("MARKET_INVENTORY", "LE", index=0, value=8),
        )
        first = specialist(
            market_phase(),
            clauses=(conditions, (PublicCondition("SHOP_SEEN", "GE", index=5, value=1),)),
        )
        equivalent = specialist(
            market_phase(),
            clauses=((PublicCondition("SHOP_SEEN", "GE", index=5, value=1),), tuple(reversed(conditions))),
        )
        restored = SpecialistProgram.from_json(first.canonical_json())
        self.assertEqual(restored, first)
        self.assertEqual(restored.semantic_id, first.semantic_id)
        self.assertEqual(equivalent.semantic_id, first.semantic_id)
        self.assertEqual(first.schema_version, SPECIALIST_PROGRAM_SCHEMA_VERSION)

    def test_all_v1_predicates_evaluate_on_legal_public_state(self):
        conditions = {
            "STEP": PublicCondition("STEP", "GE", value=48),
            "DAY": PublicCondition("DAY", "GE", value=2),
            "HOUR": PublicCondition("HOUR", "EQ", value=0),
            "SHOP_AT": PublicCondition("SHOP_AT", "EQ", index=1, value=5),
            "SHOP_SEEN": PublicCondition("SHOP_SEEN", "GE", index=5, value=1),
            "OWN_MONEY": PublicCondition("OWN_MONEY", "GE", value=4000),
            "OPP_MONEY": PublicCondition("OPP_MONEY", "LE", value=3000),
            "OWN_UNITS": PublicCondition("OWN_UNITS", "GE", value=2),
            "OPP_UNITS": PublicCondition("OPP_UNITS", "LE", value=1),
            "OWN_LAND": PublicCondition("OWN_LAND", "GE", value=2),
            "OPP_LAND": PublicCondition("OPP_LAND", "LE", value=1),
            "OWN_HIRES_TODAY": PublicCondition("OWN_HIRES_TODAY", "GE", value=1),
            "OPP_HIRES_TODAY": PublicCondition("OPP_HIRES_TODAY", "LE", value=0),
            "MARKET_PRICE": PublicCondition("MARKET_PRICE", "GE", index=0, value=31),
            "MARKET_INVENTORY": PublicCondition("MARKET_INVENTORY", "GE", index=0, value=8),
            "OWN_SHED": PublicCondition("OWN_SHED", "GE", index=0, value=3),
            "OWN_SEED": PublicCondition("OWN_SEED", "GE", index=0, value=2),
        }
        self.assertEqual(set(conditions), ASPE_PUBLIC_FEATURES)
        for feature, condition in conditions.items():
            with self.subTest(feature=feature):
                program = specialist(clauses=((condition,),))
                self.assertTrue(initiation_matches(program, observation(48)))
                self.assertFalse(initiation_matches(program, observation(49)))

    def test_initiation_false_is_exact_parent_and_true_changes_only_market(self):
        program = specialist(clauses=((PublicCondition("OWN_MONEY", "GE", value=5000),),))
        _, inactive = materialized(program)
        self.assertEqual(inactive.action(observation(48, own_money=4999), PARENT), PARENT)
        self.assertFalse(inactive.active)
        self.assertEqual(inactive.action(observation(49, own_money=6000), PARENT), PARENT)

        _, active = materialized(program)
        changed = active.action(observation(48, own_money=5000), PARENT)
        self.assertEqual(changed["farmer"], PARENT["farmer"])
        self.assertEqual(changed["hands"], PARENT["hands"])
        self.assertEqual(changed["market"], [["BUY_PRODUCT", "WHEAT", 2]])

    def test_three_phase_transition_expresses_capital_land_option(self):
        items = tuple(motif(direction) for direction in ("SOUTH", "EAST", "WEST"))
        phases = tuple(motif_phase(item) for item in items)
        program = specialist(*phases, cap=3)
        _, runner = materialized(program, items)
        seen = []
        for step in range(48, 51):
            seen.append(runner.action(observation(step), PARENT)["farmer"][0])
            runner.observe(observation(step + 1))
        self.assertEqual(seen, ["SOUTH", "EAST", "WEST"])
        self.assertEqual(runner.action(observation(51), PARENT), PARENT)
        self.assertTrue(runner.complete)

    def test_reactive_branch_selects_materially_different_phase(self):
        items = tuple(motif(direction) for direction in ("SOUTH", "EAST", "WEST"))
        phases = tuple(motif_phase(item) for item in items)
        decision = DecisionPoint(
            0,
            PublicCondition("OWN_MONEY", "GE", value=3500),
            Continuation(ContinuationKind.PHASE, 1),
            Continuation(ContinuationKind.PHASE, 2),
        )
        program = specialist(*phases, decisions=(decision,), cap=2)
        for money, expected in ((4000, "EAST"), (3000, "WEST")):
            with self.subTest(money=money):
                _, runner = materialized(program, items)
                runner.action(observation(48, own_money=money), PARENT)
                runner.observe(observation(49, own_money=money))
                self.assertEqual(runner.action(observation(49, own_money=money), PARENT)["farmer"][0], expected)

    def test_public_state_termination_returns_to_parent_early(self):
        item = motif("SOUTH", count=4)
        term = Termination(
            TerminationType.TARGET_PUBLIC_STATE,
            4,
            condition=PublicCondition("OWN_LAND", "GE", value=3),
        )
        program = specialist(motif_phase(item, cap=4, termination=term), cap=4)
        _, runner = materialized(program, (item,))
        self.assertEqual(runner.action(observation(48, own_land=2), PARENT)["farmer"], ["SOUTH"])
        runner.observe(observation(49, own_land=3))
        self.assertEqual(runner.action(observation(49, own_land=3), PARENT), PARENT)
        self.assertEqual(runner._runner.last_termination, TerminationReason.TARGET_PUBLIC_STATE)

    def test_abort_and_bounded_recovery_both_finish_at_exact_parent(self):
        south, west = motif("SOUTH"), motif("WEST")
        first = motif_phase(south)
        abort = DecisionPoint(
            0,
            PublicCondition("OWN_MONEY", "GE", value=3500),
            Continuation(ContinuationKind.PHASE, 1),
            Continuation(ContinuationKind.ABORT_TO_PARENT),
        )
        aborting = specialist(first, motif_phase(west), decisions=(abort,), cap=2)
        _, runner = materialized(aborting, (south, west))
        runner.action(observation(48, own_money=3000), PARENT)
        runner.observe(observation(49, own_money=3000))
        self.assertEqual(runner.action(observation(49), PARENT), PARENT)
        self.assertTrue(runner.complete)

        recover = DecisionPoint(
            0,
            PublicCondition("OWN_MONEY", "GE", value=3500),
            Continuation(ContinuationKind.ABORT_TO_PARENT),
            Continuation(ContinuationKind.RECOVERY),
        )
        recovering = specialist(first, decisions=(recover,), recovery=motif_phase(west), cap=2)
        _, runner = materialized(recovering, (south, west))
        runner.action(observation(48, own_money=3000), PARENT)
        runner.observe(observation(49, own_money=3000))
        self.assertEqual(runner.action(observation(49), PARENT)["farmer"], ["WEST"])
        runner.observe(observation(50))
        self.assertEqual(runner.action(observation(50), PARENT), PARENT)

    def test_malformed_programs_fail_closed(self):
        base = PublicCondition("OWN_MONEY", "GE", value=1)
        with self.assertRaisesRegex(ValueError, "1..6 clauses"):
            specialist(clauses=())
        with self.assertRaisesRegex(ValueError, "1..6 clauses"):
            specialist(clauses=((base,),) * 7)
        with self.assertRaisesRegex(ValueError, "1..4 conditions"):
            specialist(clauses=((base,) * 5,))
        with self.assertRaisesRegex(ValueError, "1..3 body phases"):
            specialist(*(market_phase(cap=1) for _ in range(4)), cap=4)
        with self.assertRaisesRegex(ValueError, "non-strategic feature"):
            specialist(clauses=((PublicCondition("OWN_WORKER_X", "GE", index=0, value=0),),))
        with self.assertRaisesRegex(ValueError, "exact STEP equality"):
            specialist(clauses=((PublicCondition("STEP", "EQ", value=48),),))
        document = specialist().to_document()
        document["replay_id"] = "forbidden"
        with self.assertRaisesRegex(ValueError, "unsupported SpecialistProgram fields"):
            SpecialistProgram.from_document(document)
        document = json.loads(specialist().canonical_json())
        document["parent_fallback"] = "CONTINUE_SPECIALIST"
        with self.assertRaisesRegex(ValueError, "invalid SpecialistProgram lane or fallback"):
            SpecialistProgram.from_document(document)

    def test_host_device_and_bkg4_materialization_are_deterministic(self):
        item = motif("SOUTH")
        decision = DecisionPoint(
            0,
            PublicCondition("MARKET_INVENTORY", "GE", index=0, value=4),
            Continuation(ContinuationKind.PHASE, 1),
            Continuation(ContinuationKind.ABORT_TO_PARENT),
        )
        program = specialist(motif_phase(item), market_phase(cap=1), decisions=(decision,), cap=2)
        macro = program.to_macro_program()
        lookup = {item.semantic_id: item}
        host = encode_program_batch((macro,), lookup)
        restored, restored_motifs = decode_program_batch(host)
        self.assertEqual(restored[0].to_document(), macro.to_document())
        self.assertEqual(restored_motifs[item.semantic_id], item)
        first_device = encode_device_batch((macro,), lookup)
        second_device = encode_device_batch((macro,), lookup)
        self.assertEqual(first_device, second_device)
        self.assertEqual(struct.unpack_from("<II", first_device), (0x524C5234, 2))
        first, _ = materialized(program, (item,))
        second, _ = materialized(program, (item,))
        self.assertEqual(first.canonical_json(), second.canonical_json())
        self.assertEqual(first.materialization_id, second.materialization_id)
        self.assertEqual(first.state_mapping["parent_return"], "CLEAR_AND_RETURN_TO_PARENT")
        self.assertEqual(first.specialist.end_policy, FallbackPolicy.RETURN_TO_PARENT.value)

    def test_six_clause_gate_materializes_and_execution_repeats_exactly(self):
        clauses = tuple(
            (PublicCondition("OWN_MONEY", "GE", value=threshold),)
            for threshold in (1000, 2000, 3000, 4000, 5000, 6000)
        )
        program = specialist(clauses=clauses)
        first_artifact, first = materialized(program)
        second_artifact, second = materialized(program)
        self.assertEqual(len(first_artifact.outer_gate), 6)
        self.assertEqual(first_artifact.materialization_id, second_artifact.materialization_id)
        first_actions = []
        second_actions = []
        for step in (48, 49):
            current = observation(step, own_money=4000)
            first_actions.append(first.action(current, PARENT))
            second_actions.append(second.action(current, PARENT))
            first.observe(observation(step + 1, own_money=4000))
            second.observe(observation(step + 1, own_money=4000))
        self.assertEqual(first_actions, second_actions)
        self.assertEqual(first.action(observation(50), PARENT), PARENT)
        parent = {
            "schema_version": 4,
            "name": "aspe_phase1_parent",
            "blackboard": [{"name": "slot_1", "type": "I32", "initial": 0, "owner": "parent"}],
            "tree": {"type": "Sequence", "children": []},
            "strategic_routes": [],
        }
        composed = compose_specialists(
            parent,
            (first_artifact.specialist,),
            (),
            strategic_authorities=(StrategicAuthority("strategic_route", "slot_1"),),
            output_name="aspe_phase1_fixture",
        )
        executable = {
            key: value for key, value in composed.module.items()
            if key in {
                "schema_version", "name", "feature_registry_version", "enums",
                "blackboard", "subtrees", "tree", "provenance",
            }
        }
        self.assertGreater(len(bonsai_kg_v4.compile_policy(executable).nodes), 0)


if __name__ == "__main__":
    unittest.main()
