"""Deterministic semantic large-neighborhood repair for bounded Option-Programs."""
from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, Iterable, Mapping, Sequence

from .archive import Archive, InsertResult
from .beam import BatchEvaluation, ExpansionOperator, EvaluatedProgram, _entry, expand_program
from .checkpoints import Checkpoint
from .effects import EffectBlock, EffectVector
from .failures import FailureClass
from .ir import ActionFamily, ActionSource, MacroProgram
from .motifs import Motif
from .schema import DatasetRole, stable_identity

LNS_SCHEMA_VERSION=1

class DestroyOperator(str,Enum):
    DESTROY_PHASE="DESTROY_PHASE"
    DESTROY_ACTION_FAMILY_IN_PHASE="DESTROY_ACTION_FAMILY_IN_PHASE"
    DESTROY_EVENT_BOUNDARY="DESTROY_EVENT_BOUNDARY"
    DESTROY_NUMERIC_PARAMETER_GROUP="DESTROY_NUMERIC_PARAMETER_GROUP"
    DESTROY_DONOR_OR_MOTIF_SOURCE="DESTROY_DONOR_OR_MOTIF_SOURCE"
    DESTROY_COUPLED_ACTION_FAMILIES="DESTROY_COUPLED_ACTION_FAMILIES"

class RepairDisposition(str,Enum):
    REPAIR="REPAIR"
    RETURN_TO_CHECKPOINT_SEARCH="RETURN_TO_CHECKPOINT_SEARCH"
    DEFER_TO_REACTIVE_REPAIR="DEFER_TO_REACTIVE_REPAIR"
    NO_REPAIR="NO_REPAIR"

class RepairOutcome(str,Enum):
    ACCEPTED_TRANSFER_IMPROVEMENT="ACCEPTED_TRANSFER_IMPROVEMENT"
    ACCEPTED_EFFECT_NOVELTY="ACCEPTED_EFFECT_NOVELTY"
    TARGET_CAPABILITY_LOST="TARGET_CAPABILITY_LOST"
    TRANSFER_NOT_IMPROVED="TRANSFER_NOT_IMPROVED"

@dataclass(frozen=True)
class FailurePolicy:
    disposition: RepairDisposition
    operators: tuple[DestroyOperator,...]=()

_NON_TRANSFERABLE=(DestroyOperator.DESTROY_ACTION_FAMILY_IN_PHASE,DestroyOperator.DESTROY_DONOR_OR_MOTIF_SOURCE,
 DestroyOperator.DESTROY_EVENT_BOUNDARY,DestroyOperator.DESTROY_PHASE,DestroyOperator.DESTROY_COUPLED_ACTION_FAMILIES)

def failure_policy(failure:FailureClass)->FailurePolicy:
    if failure is FailureClass.BAD_CHECKPOINT:return FailurePolicy(RepairDisposition.RETURN_TO_CHECKPOINT_SEARCH)
    if failure in {FailureClass.OPEN_LOOP_FRAGILITY,FailureClass.MIXED_INITIATION}:return FailurePolicy(RepairDisposition.DEFER_TO_REACTIVE_REPAIR)
    if failure is FailureClass.FUTURE_DEPENDENT:return FailurePolicy(RepairDisposition.NO_REPAIR)
    if failure is FailureClass.BAD_HORIZON:return FailurePolicy(RepairDisposition.REPAIR,(DestroyOperator.DESTROY_EVENT_BOUNDARY,))
    if failure in {FailureClass.DONOR_LIMITATION,FailureClass.REPRESENTATION_LIMITATION}:
        return FailurePolicy(RepairDisposition.REPAIR,(DestroyOperator.DESTROY_DONOR_OR_MOTIF_SOURCE,DestroyOperator.DESTROY_PHASE))
    if failure is FailureClass.TARGET_CAPABILITY_LOST:
        return FailurePolicy(RepairDisposition.REPAIR,(DestroyOperator.DESTROY_ACTION_FAMILY_IN_PHASE,DestroyOperator.DESTROY_PHASE))
    return FailurePolicy(RepairDisposition.REPAIR,_NON_TRANSFERABLE)

@dataclass(frozen=True)
class DestroyDescriptor:
    source_id:str
    failure:FailureClass
    operator:DestroyOperator
    phase_indices:tuple[int,...]
    families:tuple[ActionFamily,...]=()
    parameter_keys:tuple[str,...]=()
    schema_version:int=LNS_SCHEMA_VERSION
    @property
    def semantic_id(self):return stable_identity("RLRMB-LNS-DESTROY-V1",self.to_document())
    def to_document(self):return {"schema_version":self.schema_version,"source_id":self.source_id,"failure":self.failure.value,
      "operator":self.operator.value,"phase_indices":list(self.phase_indices),"families":[x.value for x in self.families],"parameter_keys":list(self.parameter_keys)}

def destroy_descriptors(program:MacroProgram,failure:FailureClass)->tuple[DestroyDescriptor,...]:
    policy=failure_policy(failure)
    if policy.disposition is not RepairDisposition.REPAIR:return ()
    out=[]
    for op in policy.operators:
      for i,p in enumerate(program.phases):
        if op is DestroyOperator.DESTROY_ACTION_FAMILY_IN_PHASE:
          out.extend(DestroyDescriptor(program.semantic_id,failure,op,(i,),(f,)) for f in p.action_families)
        elif op is DestroyOperator.DESTROY_COUPLED_ACTION_FAMILIES:
          if set(p.action_families)&{ActionFamily.HANDS,ActionFamily.MARKET}:out.append(DestroyDescriptor(program.semantic_id,failure,op,(i,),(ActionFamily.HANDS,ActionFamily.MARKET)))
        elif op is DestroyOperator.DESTROY_NUMERIC_PARAMETER_GROUP:
          if p.parameters:out.append(DestroyDescriptor(program.semantic_id,failure,op,(i,),parameter_keys=tuple(k for k,_ in p.parameters)))
        elif op is DestroyOperator.DESTROY_DONOR_OR_MOTIF_SOURCE:
          if p.action_source in {ActionSource.DONOR_BLOCK,ActionSource.SEMANTIC_SPLICE}:out.append(DestroyDescriptor(program.semantic_id,failure,op,(i,)))
        else: out.append(DestroyDescriptor(program.semantic_id,failure,op,(i,)))
    # Numeric repair is always available for primitive phases even when not a default priority.
    for i,p in enumerate(program.phases):
      if p.parameters:out.append(DestroyDescriptor(program.semantic_id,failure,DestroyOperator.DESTROY_NUMERIC_PARAMETER_GROUP,(i,),parameter_keys=tuple(k for k,_ in p.parameters)))
    return tuple(sorted({d.semantic_id:d for d in out}.values(),key=lambda d:(policy.operators.index(d.operator) if d.operator in policy.operators else 99,d.phase_indices,d.families,d.semantic_id)))

def _outside_equal(parent:MacroProgram,child:MacroProgram,d:DestroyDescriptor)->bool:
    if len(parent.phases)!=len(child.phases):return d.operator is DestroyOperator.DESTROY_PHASE
    return all(parent.phases[i]==child.phases[i] for i in range(len(parent.phases)) if i not in d.phase_indices)

def generate_repairs(program:MacroProgram,d:DestroyDescriptor,motifs:Mapping[str,Motif],events:Sequence[Checkpoint])->tuple[MacroProgram,...]:
    i=d.phase_indices[0]; candidates=[]
    mapping={DestroyOperator.DESTROY_EVENT_BOUNDARY:(ExpansionOperator.TRUNCATE_AT_EVENT,ExpansionOperator.EXTEND_TO_EVENT),
      DestroyOperator.DESTROY_NUMERIC_PARAMETER_GROUP:(ExpansionOperator.MUTATE_BOUNDED_NUMERIC_PARAMETER,),
      DestroyOperator.DESTROY_DONOR_OR_MOTIF_SOURCE:(ExpansionOperator.SWAP_SEMANTIC_DONOR_OR_MOTIF_PHASE,),
      DestroyOperator.DESTROY_PHASE:(ExpansionOperator.REPLACE_PHASE_SOURCE,)}
    if d.operator in {DestroyOperator.DESTROY_ACTION_FAMILY_IN_PHASE,DestroyOperator.DESTROY_COUPLED_ACTION_FAMILIES}:
      for c in expand_program(program,ExpansionOperator.CHANGE_ACTION_FAMILY_MASK,motifs,events):
        if len(c.phases)==len(program.phases) and set(c.phases[i].action_families)^set(program.phases[i].action_families)<=set(d.families):candidates.append(c)
    else:
      for op in mapping.get(d.operator,()):
        for c in expand_program(program,op,motifs,events):
          changed=[j for j in range(min(len(program.phases),len(c.phases))) if program.phases[j]!=c.phases[j]]
          if changed==[i] and _outside_equal(program,c,d):candidates.append(c)
      if d.operator is DestroyOperator.DESTROY_PHASE and len(program.phases)>1:
        phases=program.phases[:i]+program.phases[i+1:]; total=sum(p.termination.max_steps for p in phases)
        candidates.append(MacroProgram(program.lane,program.initiation,phases,total))
    return tuple({c.semantic_id:c for c in sorted(candidates,key=lambda c:c.semantic_id)}.values())

def deduplicate_repairs(programs:Iterable[MacroProgram]):
    values=tuple(programs); unique={p.semantic_id:p for p in values}; return tuple(unique[k] for k in sorted(unique)),len(values)-len(unique)

def _block(v:EffectVector,role:DatasetRole)->EffectBlock:return next(b for b in v.blocks if b.role is role)
def _summary(b:EffectBlock):return (sum(b.win_to_loss),sum(x<0 for x in b.margin_deltas),sum(b.loss_to_win),sum(x>0 for x in b.candidate_margins),min(b.margin_deltas),sum(b.margin_deltas))

@dataclass(frozen=True)
class RepairDecision:
    outcome:RepairOutcome
    target_before:tuple[int,...]
    target_after:tuple[int,...]
    transfer_before:tuple[int,...]
    transfer_after:tuple[int,...]

def accept_repair(parent:EffectVector,repaired:EffectVector,worst_target_floor:int=-(1<<31))->RepairDecision:
    pt,rt=_block(parent,DatasetRole.TARGET),_block(repaired,DatasetRole.TARGET); px,rx=_block(parent,DatasetRole.TRANSFER),_block(repaired,DatasetRole.TRANSFER)
    required=[i for i,x in enumerate(pt.candidate_margins) if x>0]
    lost=any(rt.candidate_margins[i]<=0 for i in required) or any(a and not b for a,b in zip(pt.loss_to_win,rt.loss_to_win)) or sum(rt.win_to_loss)>sum(pt.win_to_loss) or min(rt.margin_deltas)<worst_target_floor
    tb,ta,xb,xa=tuple(pt.margin_deltas),tuple(rt.margin_deltas),tuple(px.margin_deltas),tuple(rx.margin_deltas)
    if lost:return RepairDecision(RepairOutcome.TARGET_CAPABILITY_LOST,tb,ta,xb,xa)
    before,after=_summary(px),_summary(rx)
    # Aggregate is recorded but cannot establish improvement by itself.
    improved=after[:5] < before[:5] if False else ((-after[0],-after[1],after[2],after[3],after[4]) > (-before[0],-before[1],before[2],before[3],before[4]))
    return RepairDecision(RepairOutcome.ACCEPTED_TRANSFER_IMPROVEMENT if improved else RepairOutcome.TRANSFER_NOT_IMPROVED,tb,ta,xb,xa)

@dataclass
class _Counts:
    attempts:int=0; valid:int=0; evaluated:int=0; target:int=0; transfer:int=0; archive:int=0
class OperatorTracker:
    def __init__(self):self._counts={op:_Counts() for op in DestroyOperator}
    def record(self,op,*,valid=False,evaluated=False,target_preserved=False,transfer_improved=False,archive_success=False):
      c=self._counts[op]; c.attempts+=1;c.valid+=valid;c.evaluated+=evaluated;c.target+=target_preserved;c.transfer+=transfer_improved;c.archive+=archive_success
    def order(self,operators):
      return tuple(sorted(set(operators),key=lambda op:(-self._counts[op].transfer,-self._counts[op].archive,-self._counts[op].target,self._counts[op].attempts,op.value)))
    def metrics(self):
      return {op.value:{"attempts":c.attempts,"structurally_valid":c.valid,"cuda_evaluated":c.evaluated,"target_preserving":c.target,"transfer_improving":c.transfer,"archive_success":c.archive,
       "target_preserving_rate":0 if not c.attempts else c.target/c.attempts,"transfer_improvement_rate":0 if not c.attempts else c.transfer/c.attempts,"archive_success_rate":0 if not c.attempts else c.archive/c.attempts} for op,c in self._counts.items()}

@dataclass(frozen=True)
class RepairRecord:
    ancestor_id:str; parent_id:str; failure:FailureClass; destroy:DestroyDescriptor
    repair_operator:str; repaired_id:str; round:int; decision:RepairDecision
    archive_result:InsertResult|None; cuda_games:int
    def to_document(self):return {"schema_version":LNS_SCHEMA_VERSION,"ancestor_id":self.ancestor_id,"parent_id":self.parent_id,
      "failure":self.failure.value,"destroy":self.destroy.to_document(),"repair_operator":self.repair_operator,
      "repaired_id":self.repaired_id,"round":self.round,"target_before":list(self.decision.target_before),
      "target_after":list(self.decision.target_after),"transfer_before":list(self.decision.transfer_before),
      "transfer_after":list(self.decision.transfer_after),"outcome":self.decision.outcome.value,
      "archive_result":None if self.archive_result is None else self.archive_result.value,"cuda_games":self.cuda_games}

class DualPanelEvaluator:
    """Compose two unchanged Phase-6 CUDA adapters into role-separated exact effects."""
    def __init__(self,target,transfer):self.target=target;self.transfer=transfer;self.case_count=target.case_count+transfer.case_count
    def projected_games(self,n):return self.target.projected_games(n)+self.transfer.projected_games(n)
    def __call__(self,programs,motifs):
      t=self.target(programs,motifs);x=self.transfer(programs,motifs);rows=[]
      for a,b in zip(t.programs,x.programs):
        tb=_block(a.effect,DatasetRole.TARGET);xb=_block(b.effect,DatasetRole.TARGET)
        transfer=EffectBlock(DatasetRole.TRANSFER,xb.panel_id,xb.baseline_margins,xb.margin_deltas,xb.action_changes)
        rows.append(EvaluatedProgram(a.program,EffectVector((tb,transfer)),a.valid and b.valid,a.fault or b.fault))
      return BatchEvaluation(tuple(rows),self.case_count,t.games+x.games,t.launches+x.launches,t.host_seconds+x.host_seconds,t.cuda_ms+x.cuda_ms)

@dataclass(frozen=True)
class LNSResult:
    archive:Archive; records:tuple[RepairRecord,...]; metrics:dict[str,Any]

def run_lns(parent:MacroProgram,failure:FailureClass,motifs:Mapping[str,Motif],events:Sequence[Checkpoint],evaluator,
            *,archive:Archive|None=None,max_repairs:int=16)->LNSResult:
    archive=archive or Archive(); descriptors=destroy_descriptors(parent,failure); raw=[]
    for d in descriptors:
      for p in generate_repairs(parent,d,motifs,events):raw.append((d,p))
    unique_programs,removed=deduplicate_repairs(p for _,p in raw)
    unique_programs=unique_programs[:max_repairs]; by_id={p.semantic_id:p for p in unique_programs}
    descriptor_by_id={p.semantic_id:d for d,p in raw if p.semantic_id in by_id}
    if not unique_programs:return LNSResult(archive,(),{"raw_repairs":len(raw),"structural_duplicates_removed":removed,"cuda_evaluated_repairs":0})
    batch=evaluator((parent,*unique_programs),motifs); parent_effect=batch.programs[0].effect;records=[];tracker=OperatorTracker()
    for evaluated in batch.programs[1:]:
      d=descriptor_by_id[evaluated.program.semantic_id]
      decision=accept_repair(parent_effect,evaluated.effect); insertion=None
      if decision.outcome is RepairOutcome.ACCEPTED_TRANSFER_IMPROVEMENT and evaluated.valid:
        insertion=archive.insert(_entry(evaluated.program,evaluated.effect))
      tracker.record(d.operator,valid=True,evaluated=True,
                     target_preserved=decision.outcome is not RepairOutcome.TARGET_CAPABILITY_LOST,
                     transfer_improved=decision.outcome is RepairOutcome.ACCEPTED_TRANSFER_IMPROVEMENT,
                     archive_success=insertion in {InsertResult.INSERTED_NEW_CELL,InsertResult.REPLACED_ELITE})
      records.append(RepairRecord(parent.semantic_id,parent.semantic_id,failure,d,"BEAM_CONSTRAINED_"+d.operator.value,
                                  evaluated.program.semantic_id,1,decision,insertion,evaluator.case_count))
    metrics={"source_candidates":1,"failure_classes":{failure.value:1},"destroy_attempts":len(descriptors),
      "raw_repairs":len(raw),"structural_duplicates_removed":removed,"cuda_evaluated_repairs":len(unique_programs),
      "target_preserving_repairs":sum(r.decision.outcome is not RepairOutcome.TARGET_CAPABILITY_LOST for r in records),
      "target_capability_lost_repairs":sum(r.decision.outcome is RepairOutcome.TARGET_CAPABILITY_LOST for r in records),
      "transfer_improving_repairs":sum(r.decision.outcome is RepairOutcome.ACCEPTED_TRANSFER_IMPROVEMENT for r in records),
      "cuda_games":batch.games,"evaluator_launches":batch.launches,"host_seconds":batch.host_seconds,"cuda_ms":batch.cuda_ms,
      "operator_metrics":tracker.metrics(),"retained_effect_cells":archive.metrics()["occupied_cells"]}
    return LNSResult(archive,tuple(records),metrics)
