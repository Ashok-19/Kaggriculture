from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

from kaggriculture_bonsai.composition.bonsai_composer import StrategicAuthority, compose_specialists
from kaggriculture_bonsai.composition.module_codec import ConditionGene
from kaggriculture_bonsai.search.checkpoints import Checkpoint, EventType, Lane
from kaggriculture_bonsai.search.evaluator import ReferenceInterpreter
from kaggriculture_bonsai.search.ir import (
    ActionFamily, ActionSource, Continuation, ContinuationKind, DecisionPoint,
    MacroProgram, Phase, PublicCondition, Termination, TerminationType,
)
from kaggriculture_bonsai.routing.materialize import (
    MATERIALIZATION_SCHEMA_VERSION, MaterializationInput, MaterializedInterpreter,
    materialize_program,
)
from kaggriculture_bonsai.search.motifs import motif_from_actions

PASS = {"farmer": ["PASS"], "hands": [], "market": []}
PARENT = {"farmer": ["NORTH"], "hands": [["WEST"]], "market": [["SELL", "WHEAT", 2]]}


def obs(step, *, money=3000, land=1, hires=0, shops=()):
    return {
        "step": step, "day": step // 24, "hour": step % 24, "player": 0,
        "farms": [
            {"money": money, "hands": [[0, 0]] * hires, "hires_today": hires,
             "unlocked_quadrants": ["NW"] * land, "tiles": []},
            {"money": 3000, "hands": [], "hires_today": 0,
             "unlocked_quadrants": ["NW"], "tiles": []},
        ],
        "private": {"seeds": {}, "shed": {}, "inventories": []},
        "market": {"inventory": {}, "prices": {}},
        "town": {"unlocked_shops": list(shops)},
    }


def phase(motif_id, direction, *, families=(ActionFamily.FARMER,), cap=1):
    del direction
    return Phase(ActionSource.SEMANTIC_SPLICE, families, Termination(TerminationType.FIXED_CAP, cap), motif_id=motif_id)


class MaterializeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.motifs = {}
        for direction in ("SOUTH", "EAST", "WEST"):
            motif = motif_from_actions(
                [{"farmer": [direction], "hands": [["CARE"]], "market": [["HIRE"]]}] * 4,
                tuple(ActionFamily),
            )
            cls.motifs[direction] = motif

    def program(self, *, reactive=False):
        south, east, west = (self.motifs[x] for x in ("SOUTH", "EAST", "WEST"))
        phases = (
            phase(south.semantic_id, "SOUTH", families=(ActionFamily.FARMER,), cap=1),
            phase(east.semantic_id, "EAST", families=(ActionFamily.HANDS, ActionFamily.MARKET), cap=1),
            phase(west.semantic_id, "WEST", families=(ActionFamily.FARMER,), cap=1),
        )
        checkpoint = Checkpoint(0, Lane.GENERAL, (EventType.FIRST_EFFECT_DIVERGENCE,))
        if not reactive:
            return MacroProgram(Lane.GENERAL, checkpoint, phases[:2], 2)
        decision = DecisionPoint(
            0, PublicCondition("OWN_MONEY", "GE", value=3000),
            Continuation(ContinuationKind.PHASE, 1),
            Continuation(ContinuationKind.ABORT_TO_PARENT),
        )
        return MacroProgram(Lane.GENERAL, checkpoint, phases, 3, schema_version=2, decisions=(decision,), recovery_phase=None)

    def source(self, program=None, gate=None):
        lookup = {item.semantic_id: item for item in self.motifs.values()}
        return MaterializationInput(
            candidate_id="c" * 64,
            program=program or self.program(),
            outer_gate=gate or ((ConditionGene("OWN_MONEY", "GE", 0, 3000),),),
            motifs=lookup,
            route_id=500,
            base_route_id=64,
            route_source="workspace/materialized_fixture.json",
            authority="strategic_route",
            authority_slot="slot_1",
        )

    def run_pair(self, program, observations):
        source = self.source(program, gate=((ConditionGene("OWN_MONEY", "GE", 0, 0),),))
        artifact = materialize_program(source)
        direct = ReferenceInterpreter(program, source.motifs)
        materialized = MaterializedInterpreter(artifact, source.motifs)
        direct_actions = []
        materialized_actions = []
        for before, after in observations:
            direct_actions.append(direct.action(before, PARENT)); direct.observe(after)
            materialized_actions.append(materialized.action(before, PARENT)); materialized.observe(after)
        return artifact, direct_actions, materialized_actions, materialized

    def test_deterministic_materialization_serialization_and_semantic_id(self):
        first = materialize_program(self.source())
        second = materialize_program(self.source())
        self.assertEqual(first.canonical_json(), second.canonical_json())
        self.assertEqual(first.materialization_id, second.materialization_id)
        changed = materialize_program(self.source(gate=((ConditionGene("OWN_MONEY", "GE", 0, 2999),),)))
        self.assertNotEqual(first.materialization_id, changed.materialization_id)
        self.assertEqual(first.schema_version, MATERIALIZATION_SCHEMA_VERSION)

    def test_outer_gate_false_is_exact_parent_and_true_latches(self):
        artifact = materialize_program(self.source())
        inactive = MaterializedInterpreter(artifact, self.source().motifs)
        self.assertEqual(inactive.action(obs(0, money=1), PARENT), PARENT)
        self.assertFalse(inactive.active)
        active = MaterializedInterpreter(artifact, self.source().motifs)
        self.assertNotEqual(active.action(obs(0, money=3000), PARENT), PARENT)
        self.assertTrue(active.active)
        self.assertTrue(active.gate_latched)

    def test_v1_phase_masks_transition_and_parent_return_match_reference(self):
        artifact, direct, materialized, runner = self.run_pair(
            self.program(), ((obs(0), obs(1)), (obs(1), obs(2))),
        )
        self.assertEqual(materialized, direct)
        self.assertEqual(materialized[0], {"farmer": ["SOUTH"], "hands": [["WEST"]], "market": [["SELL", "WHEAT", 2]]})
        self.assertEqual(materialized[1], {"farmer": ["NORTH"], "hands": [["CARE"]], "market": [["HIRE"]]})
        self.assertEqual(runner.action(obs(2), PARENT), PARENT)
        self.assertTrue(runner.complete)
        self.assertGreater(artifact.node_count, 0)

    def test_all_termination_descriptors_have_bounded_bkg4_mapping(self):
        motif = self.motifs["SOUTH"]
        terms = (
            Termination(TerminationType.FIXED_CAP, 2),
            Termination(TerminationType.NEXT_EOD, 24),
            Termination(TerminationType.NEXT_SHOP_REVEAL, 48),
            Termination(TerminationType.PURCHASE_COMPLETED, 8, objective="BUY_LAND", target_count=1),
            Termination(TerminationType.TARGET_PUBLIC_STATE, 8, condition=PublicCondition("OWN_MONEY", "GE", value=4000)),
            Termination(TerminationType.PARENT_COMPATIBLE, 8),
            Termination(TerminationType.MAX_STEPS, 8),
        )
        for term in terms:
            with self.subTest(term=term.kind):
                p = Phase(ActionSource.DONOR_BLOCK, tuple(ActionFamily), term, motif_id=motif.semantic_id)
                program = MacroProgram(Lane.GENERAL, Checkpoint(0, Lane.GENERAL, (EventType.REGRET_PEAK,)), (p,), term.max_steps)
                artifact = materialize_program(self.source(program))
                self.assertEqual(artifact.state_mapping["terminations"][0]["kind"], term.kind.value)

    def test_v2_true_false_abort_latch_and_recovery_match_reference(self):
        reactive = self.program(reactive=True)
        _, direct, materialized, runner = self.run_pair(reactive, ((obs(0, money=3000), obs(1, money=3000)), (obs(1, money=1), obs(2, money=1))))
        self.assertEqual(materialized, direct)
        self.assertEqual(materialized[1]["hands"], [["CARE"]])
        self.assertTrue(runner.branch_latches)

        _, direct, materialized, runner = self.run_pair(reactive, ((obs(0, money=1), obs(1, money=1)),))
        self.assertEqual(materialized, direct)
        self.assertEqual(runner.action(obs(1, money=3000), PARENT), PARENT)
        self.assertTrue(runner.complete)

        south, west = self.motifs["SOUTH"], self.motifs["WEST"]
        first = phase(south.semantic_id, "SOUTH", cap=1)
        recovery = phase(west.semantic_id, "WEST", cap=1)
        decision = DecisionPoint(0, PublicCondition("OWN_MONEY", "GE", value=3000), Continuation(ContinuationKind.ABORT_TO_PARENT), Continuation(ContinuationKind.RECOVERY))
        recovered = MacroProgram(Lane.GENERAL, Checkpoint(0, Lane.GENERAL, (EventType.REGRET_PEAK,)), (first,), 2, schema_version=2, decisions=(decision,), recovery_phase=recovery)
        _, direct, materialized, runner = self.run_pair(recovered, ((obs(0, money=1), obs(1, money=1)), (obs(1, money=1), obs(2, money=1))))
        self.assertEqual(materialized, direct)
        self.assertEqual([row["farmer"][0] for row in materialized], ["SOUTH", "WEST"])
        self.assertEqual(runner.action(obs(2), PARENT), PARENT)

    def test_invalid_state_mapping_and_blackboard_collisions_fail_closed(self):
        source = self.source()
        with self.assertRaises(ValueError):
            MaterializationInput(**{**source.__dict__, "reserved_blackboard": ("rlrmb_cccccccccc_active",)})
        artifact = materialize_program(source)
        names = [row["name"] for row in artifact.blackboard]
        self.assertEqual(len(names), len(set(names)))

    def test_existing_composer_accepts_declaration_and_rejects_conflict(self):
        artifact = materialize_program(self.source())
        parent = {
            "name": "fixture_parent", "blackboard": [{"name": "slot_1", "type": "I32", "initial": 0, "owner": "parent"}],
            "tree": {"type": "Sequence", "children": []}, "strategic_routes": [],
        }
        result = compose_specialists(
            parent, (artifact.specialist,), (),
            strategic_authorities=(StrategicAuthority("strategic_route", "slot_1"),),
            output_name="rlrmb_fixture",
        )
        self.assertEqual(result.report["composition_order"], [artifact.specialist.name])
        with self.assertRaises(ValueError):
            compose_specialists(parent, (artifact.specialist, artifact.specialist), (), strategic_authorities=(StrategicAuthority("strategic_route", "slot_1"),), output_name="bad")

    def test_materialized_bkg4_source_compiles_with_current_composer(self):
        artifact = materialize_program(self.source())
        from kaggriculture_bonsai.runtime import bonsai_kg_v4

        parent = {
            "schema_version": 4,
            "name": "fixture_parent",
            "blackboard": [{"name": "slot_1", "type": "I32", "initial": 0, "owner": "parent"}],
            "tree": {"type": "Sequence", "children": []},
            "strategic_routes": [],
        }
        result = compose_specialists(
            parent, (artifact.specialist,), (),
            strategic_authorities=(StrategicAuthority("strategic_route", "slot_1"),),
            output_name="materialized_fixture",
        )
        executable = {
            key: value for key, value in result.module.items()
            if key in {"schema_version", "name", "feature_registry_version", "enums", "blackboard", "subtrees", "tree", "provenance"}
        }
        compiled = bonsai_kg_v4.compile_policy(executable)
        self.assertGreater(len(compiled.nodes), 0)


if __name__ == "__main__":
    unittest.main()
