from __future__ import annotations

import unittest

from tools.bkg4_search.rlrmb.beam import ExpansionOperator
from tools.bkg4_search.rlrmb.checkpoints import Checkpoint, EventType, Lane
from tools.bkg4_search.rlrmb.effects import EffectBlock, EffectVector
from tools.bkg4_search.rlrmb.failures import FailureClass
from tools.bkg4_search.rlrmb.ir import ActionFamily, ActionSource, MacroProgram, Phase, Termination, TerminationType
from tools.bkg4_search.rlrmb.lns import (
    DestroyOperator, OperatorTracker, RepairDisposition, RepairOutcome,
    accept_repair, deduplicate_repairs, destroy_descriptors, failure_policy,
    generate_repairs,
)
from tools.bkg4_search.rlrmb.motifs import motif_from_actions
from tools.bkg4_search.rlrmb.schema import DatasetRole


def fixture():
    actions = [{"farmer":["NORTH"],"hands":[["SOUTH"]],"market":[["BUY_PRODUCT","WHEAT",1]]}]*8
    other = [{"farmer":["EAST"],"hands":[["WEST"]],"market":[["SELL","WHEAT",1]]}]*8
    a=motif_from_actions(actions,tuple(ActionFamily)); b=motif_from_actions(other,tuple(ActionFamily))
    cp=Checkpoint(120,Lane.MARKET_LIQUIDITY,(EventType.REGRET_PEAK,))
    phases=(Phase(ActionSource.SEMANTIC_SPLICE,tuple(ActionFamily),Termination(TerminationType.FIXED_CAP,4),a.semantic_id),
            Phase(ActionSource.PARAMETERIZED_PRIMITIVE,(ActionFamily.MARKET,),Termination(TerminationType.FIXED_CAP,2),primitive_name="LIQUIDITY_REBALANCE",parameters={"enabled":True,"mode":"WHEAT","quantity":2}))
    return MacroProgram(cp.lane,cp,phases,6),{a.semantic_id:a,b.semantic_id:b},(Checkpoint(122,cp.lane,(EventType.EOD,)),Checkpoint(126,cp.lane,(EventType.SHOP_REVEAL,)))


def effects(target, transfer, baseline=(-100,-100)):
    return EffectVector((EffectBlock(DatasetRole.TARGET,"t",baseline,target),EffectBlock(DatasetRole.TRANSFER,"x",baseline,transfer)))


class LNSTests(unittest.TestCase):
    def test_descriptors_and_failure_dispositions_are_deterministic(self):
        p,_,_=fixture()
        self.assertEqual(destroy_descriptors(p,FailureClass.NON_TRANSFERABLE),destroy_descriptors(p,FailureClass.NON_TRANSFERABLE))
        self.assertEqual(failure_policy(FailureClass.BAD_CHECKPOINT).disposition,RepairDisposition.RETURN_TO_CHECKPOINT_SEARCH)
        self.assertEqual(failure_policy(FailureClass.OPEN_LOOP_FRAGILITY).disposition,RepairDisposition.DEFER_TO_REACTIVE_REPAIR)
        self.assertEqual(failure_policy(FailureClass.MIXED_INITIATION).disposition,RepairDisposition.DEFER_TO_REACTIVE_REPAIR)
        self.assertEqual(failure_policy(FailureClass.FUTURE_DEPENDENT).disposition,RepairDisposition.NO_REPAIR)
        self.assertEqual(failure_policy(FailureClass.BAD_HORIZON).operators[0],DestroyOperator.DESTROY_EVENT_BOUNDARY)

    def test_phase_source_boundary_parameter_isolation(self):
        p,lib,events=fixture()
        for op in (DestroyOperator.DESTROY_PHASE,DestroyOperator.DESTROY_DONOR_OR_MOTIF_SOURCE,
                   DestroyOperator.DESTROY_EVENT_BOUNDARY,DestroyOperator.DESTROY_NUMERIC_PARAMETER_GROUP):
            descriptor=next(d for d in destroy_descriptors(p,FailureClass.NON_TRANSFERABLE) if d.operator is op)
            repairs=generate_repairs(p,descriptor,lib,events)
            self.assertTrue(repairs)
            for repaired in repairs:
                if len(repaired.phases)==len(p.phases):
                    for i in range(len(p.phases)):
                        if i not in descriptor.phase_indices:
                            self.assertEqual(repaired.phases[i],p.phases[i])
            if op is DestroyOperator.DESTROY_EVENT_BOUNDARY:
                self.assertEqual(repairs[0].phases[0].motif_id,p.phases[0].motif_id)
                self.assertEqual(repairs[0].phases[0].action_families,p.phases[0].action_families)
            if op is DestroyOperator.DESTROY_DONOR_OR_MOTIF_SOURCE:
                self.assertEqual(repairs[0].phases[0].termination,p.phases[0].termination)

    def test_single_and_coupled_family_isolation(self):
        p,lib,events=fixture()
        for family in ActionFamily:
            descriptor=next(d for d in destroy_descriptors(p,FailureClass.NON_TRANSFERABLE)
                            if d.operator is DestroyOperator.DESTROY_ACTION_FAMILY_IN_PHASE and d.families==(family,))
            repairs=generate_repairs(p,descriptor,lib,events)
            for repaired in repairs:
                self.assertTrue(set(p.phases[0].action_families)^set(repaired.phases[0].action_families) <= {family})
        coupled=next(d for d in destroy_descriptors(p,FailureClass.NON_TRANSFERABLE)
                     if d.operator is DestroyOperator.DESTROY_COUPLED_ACTION_FAMILIES)
        repairs=generate_repairs(p,coupled,lib,events)
        self.assertTrue(repairs)
        self.assertTrue(all(ActionFamily.FARMER in r.phases[0].action_families for r in repairs))

    def test_structural_dedup(self):
        p,_,_=fixture(); unique,removed=deduplicate_repairs((p,p))
        self.assertEqual((len(unique),removed),(1,1))

    def test_target_preservation_transfer_improvement_and_aggregate_trap(self):
        parent=effects((150,150),(0,-20))
        improved=effects((150,160),(10,10))
        lost=effects((-100,100),(1_000_000,1_000_000))
        self.assertEqual(accept_repair(parent,improved).outcome,RepairOutcome.ACCEPTED_TRANSFER_IMPROVEMENT)
        self.assertEqual(accept_repair(parent,lost).outcome,RepairOutcome.TARGET_CAPABILITY_LOST)

    def test_coupled_ablation_lesson(self):
        parent=effects((150,150),(-20,-20))
        market=effects((150,150),(-10,-20)); hands=effects((150,150),(-20,-10)); both=effects((150,150),(10,10))
        self.assertNotEqual(accept_repair(parent,market).outcome,RepairOutcome.ACCEPTED_TRANSFER_IMPROVEMENT)
        self.assertNotEqual(accept_repair(parent,hands).outcome,RepairOutcome.ACCEPTED_TRANSFER_IMPROVEMENT)
        self.assertEqual(accept_repair(parent,both).outcome,RepairOutcome.ACCEPTED_TRANSFER_IMPROVEMENT)

    def test_operator_tracker_is_deterministic(self):
        tracker=OperatorTracker(); op=DestroyOperator.DESTROY_EVENT_BOUNDARY
        tracker.record(op,valid=True,evaluated=True,target_preserved=True,transfer_improved=True,archive_success=True)
        self.assertEqual(tracker.order((op,DestroyOperator.DESTROY_PHASE)),tracker.order((DestroyOperator.DESTROY_PHASE,op)))
        self.assertEqual(tracker.metrics()[op.value]["attempts"],1)


if __name__ == "__main__": unittest.main()
