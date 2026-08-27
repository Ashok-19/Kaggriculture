import json
from pathlib import Path
import tempfile
import unittest
from tools.bkg4_search.rlrmb.pipeline import CaseMeta, ExactCaseEvidence, build_routability_evidence, build_cegis_input, emit_candidate_route
from tools.bkg4_search.rlrmb.routability import RoutabilityClassification, RoutabilityResult
from tools.bkg4_search.rlrmb.schema import DatasetRole
from tools.bkg4_search.rlrmb.ir import ActionSource, Phase
from tools.bkg4_search.rlrmb.tests.test_ir import checkpoint, fixed, program
from tools.bkg4_search.rlrmb.motifs import motif_from_actions
from tools.bkg4_search.rlrmb.ir import ActionFamily, MacroProgram
from tools.bkg4_search.rlrmb.p13_equal_budget import (
    BudgetLedger, GAME_CATEGORIES, _effect_identity, dispatch_repair, repair_disposition,
    run_campaign, select_promotion_pool, select_refinement_candidates, validate_config,
)
from tools.bkg4_search.rlrmb.effects import EffectBlock, EffectVector, effect_signature
from tools.bkg4_search.rlrmb.beam import BeamCandidate, SeedSource


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.program = program(Phase(ActionSource.PARENT, (), fixed(1)), cap=1)
        self.context = (("OWN_MONEY", 0, 3000),)
    def row(self, name, role, delta, signature=(1,), candidate=None):
        return ExactCaseEvidence(CaseMeta(name, "tape.json", 0, 1, name), role, -1, -1 + delta,
                                 self.context, signature, candidate or self.program.semantic_id)
    def test_candidate_specific_capture_binding_and_role_conversion(self):
        rows=(self.row("t",DatasetRole.TARGET,2),self.row("x",DatasetRole.TRANSFER,2,(2,)),self.row("v",DatasetRole.VALIDATION,0,(3,)))
        evidence=build_routability_evidence(self.program,rows)
        self.assertEqual({x.role for x in evidence.trajectories},{DatasetRole.TARGET,DatasetRole.TRANSFER,DatasetRole.VALIDATION})
        bad=self.row("bad",DatasetRole.TARGET,2,candidate="other")
        with self.assertRaises(ValueError): build_routability_evidence(self.program,(bad,))
    def test_adapter_has_no_historical_gate_and_sealed_fails_closed(self):
        rows=(self.row("t",DatasetRole.TARGET,2),self.row("x",DatasetRole.TRANSFER,2,(2,)),self.row("v",DatasetRole.VALIDATION,0,(3,)))
        routing=RoutabilityResult(self.program.semantic_id,RoutabilityClassification.ROUTABLE_CANDIDATE,"ok",(),(),None,None,"ENTER_HARDENED_CEGIS",{},("TARGET","TRANSFER","VALIDATION"),100)
        adapted=build_cegis_input(self.program,rows,routing)
        self.assertFalse(hasattr(adapted,"gate"))
        with self.assertRaises(ValueError): build_routability_evidence(self.program,(self.row("s",DatasetRole.SEALED,1),))

    def test_arbitrary_candidate_route_is_derived_from_semantic_content(self):
        motif = motif_from_actions(({"farmer": ["SOUTH"], "hands": [], "market": []},), tuple(ActionFamily))
        candidate = program(Phase(ActionSource.DONOR_BLOCK, tuple(ActionFamily), fixed(1), motif_id=motif.semantic_id), step=1, cap=1)
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory) / "parent.json"
            parent.write_text(json.dumps({"actions": [{"farmer": ["PASS"], "hands": [], "market": []}] * 3}))
            output = emit_candidate_route(candidate, {motif.semantic_id: motif}, parent, Path(directory) / "candidate.json")
            actions = json.loads(output.read_text())["actions"]
            self.assertEqual(actions[0]["farmer"], ["PASS"])
            self.assertEqual(actions[1]["farmer"], ["SOUTH"])

    def test_phase13_equal_budget_config_and_ledger_are_exact(self):
        config_path = Path("tools/bkg4_search/rlrmb/configs/p0_7_equal_budget_v1.json")
        config = validate_config(config_path)
        self.assertEqual(config["budget"]["total_cuda_games"], 265226)
        caps = {row["id"]: row["cuda_games"] for row in config["families"]}
        ledger = BudgetLedger(265226, caps, config["budget"]["per_family_categories"])
        family = config["families"][0]["id"]
        for category in GAME_CATEGORIES:
            ledger.consume(family, category, config["budget"]["per_family_categories"][category])
        self.assertEqual(ledger.by_family[family], 66306)
        with self.assertRaises(RuntimeError):
            ledger.consume(family, "target_beam", 1)

    def test_phase13_validate_only_writes_complete_empty_run_contract(self):
        config_path = Path("tools/bkg4_search/rlrmb/configs/p0_7_equal_budget_v1.json")
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "dry_run"
            self.assertEqual(run_campaign(config_path, output, validate_only=True), 0)
            self.assertEqual(json.loads((output / "metrics.json").read_text())["total_cuda_games"], 0)
            for name in ("config.json", "run.json", "metrics.json", "failures.json", "archive_summary.json"):
                self.assertTrue((output / name).exists())

    def test_phase13_effect_identity_uses_validated_signature_contract(self):
        effect = EffectVector((EffectBlock(DatasetRole.TARGET, "panel", (0,), (1,), (1,)),))
        self.assertEqual(_effect_identity(effect), effect_signature(effect))

    def candidate(self, step, deltas, families=(ActionFamily.MARKET,), provenance=()):
        motif_id=(f"{step:04x}" + "a" * 64)[:64]
        phase=Phase(ActionSource.SEMANTIC_SPLICE, families, fixed(1),motif_id=motif_id)
        point=checkpoint(step)
        value=MacroProgram(point.lane,point,(phase,),1,provenance=provenance)
        baseline=tuple(-1 for _ in deltas); candidate=tuple(base+delta for base,delta in zip(baseline,deltas))
        effect=EffectVector((EffectBlock(DatasetRole.TARGET,"panel",baseline,candidate,tuple(deltas)),))
        return BeamCandidate.seed(value,SeedSource.PARENT_ONLY).__class__(
            value,value.initiation,0,SeedSource.PARENT_ONLY,
            BeamCandidate.seed(value,SeedSource.PARENT_ONLY).lineage,effect=effect,
        )

    def risk(self, candidate, classification, score, diagnostics=None):
        return RoutabilityResult(candidate.program.semantic_id,classification,"risk",(),(),None,None,
                                 {
                                     RoutabilityClassification.ROUTABLE_CANDIDATE:"ENTER_HARDENED_CEGIS",
                                     RoutabilityClassification.REPAIR_LATER_CHECKPOINT:"RESAMPLE_CAUSAL_CHECKPOINT",
                                     RoutabilityClassification.REPAIR_COMMON_PREFIX:"RETAIN_COMMON_PREFIX",
                                     RoutabilityClassification.REPAIR_BRANCH_AT_REVEAL:"ADD_BOUNDED_LEGAL_BRANCH",
                                     RoutabilityClassification.REPAIR_ABORT_RECOVERY:"ADD_BOUNDED_ABORT_RECOVERY",
                                     RoutabilityClassification.FUTURE_DEPENDENT:"PERMANENT_REJECT",
                                     RoutabilityClassification.GATE_LIMITATION_SUSPECTED:"REVIEW_GATE_VOCABULARY",
                                 }[classification],diagnostics or {},("TARGET",),score)

    def test_routability_aware_refinement_rejects_aggregate_trap_and_keeps_diversity(self):
        unsafe=self.candidate(1,(10000,10000),families=tuple(ActionFamily))
        safe_a=self.candidate(120,(2,0)); safe_b=self.candidate(193,(0,2))
        risks={unsafe.program.semantic_id:self.risk(unsafe,RoutabilityClassification.FUTURE_DEPENDENT,0),
               safe_a.program.semantic_id:self.risk(safe_a,RoutabilityClassification.ROUTABLE_CANDIDATE,100),
               safe_b.program.semantic_id:self.risk(safe_b,RoutabilityClassification.ROUTABLE_CANDIDATE,100)}
        selected=select_refinement_candidates((unsafe,safe_b,safe_a),risks,2)
        self.assertEqual({x.program.semantic_id for x in selected},{safe_a.program.semantic_id,safe_b.program.semantic_id})
        self.assertEqual(len({_effect_identity(x.effect) for x in selected}),2)
        self.assertEqual(selected,select_refinement_candidates((safe_a,unsafe,safe_b),risks,2))

    def test_promotion_pool_is_checkpoint_diverse_and_provenance_neutral(self):
        early=self.candidate(1,(2,0),provenance=("route-a",))
        duplicate=self.candidate(1,(2,0),provenance=("route-b",))
        late=self.candidate(120,(1,0))
        self.assertEqual(early.program.semantic_id,duplicate.program.semantic_id)
        pool=select_promotion_pool((duplicate,late,early),{early.program.semantic_id,late.program.semantic_id},2)
        self.assertEqual({x.program.initiation.step for x in pool},{1,120})

    def test_failure_driven_repair_dispositions_are_exact(self):
        expected={
            RoutabilityClassification.REPAIR_LATER_CHECKPOINT:"RETURN_TO_CHECKPOINT_SEARCH",
            RoutabilityClassification.REPAIR_COMMON_PREFIX:"RETAIN_COMMON_PREFIX",
            RoutabilityClassification.REPAIR_BRANCH_AT_REVEAL:"BOUNDED_REACTIVE_BRANCH",
            RoutabilityClassification.REPAIR_ABORT_RECOVERY:"BOUNDED_ABORT_RECOVERY",
            RoutabilityClassification.FUTURE_DEPENDENT:"ABANDON_BASIN",
        }
        self.assertEqual({key:repair_disposition(key) for key in expected},expected)
        calls=[]
        handlers={value:(lambda value=value:calls.append(value) or value) for value in expected.values()
                  if value != "ABANDON_BASIN"}
        for classification,disposition in expected.items():
            self.assertEqual(dispatch_repair(classification,handlers),disposition)
        self.assertNotIn("ABANDON_BASIN",calls)

    def test_exact_collisions_require_a_legal_gate_predicate_before_refinement(self):
        mixed=self.candidate(25,(3,0),families=(ActionFamily.FARMER,ActionFamily.HANDS))
        gated=self.candidate(120,(2,0))
        risks={
            mixed.program.semantic_id:self.risk(mixed,RoutabilityClassification.ROUTABLE_CANDIDATE,100,
                                                {"exact_collision_count":8,"gate_predicate":None}),
            gated.program.semantic_id:self.risk(gated,RoutabilityClassification.ROUTABLE_CANDIDATE,100,
                                                {"exact_collision_count":8,"gate_predicate":{"feature":"OWN_MONEY"}}),
        }
        self.assertEqual(select_refinement_candidates((mixed,gated),risks,2),(gated,))


if __name__ == "__main__": unittest.main()
