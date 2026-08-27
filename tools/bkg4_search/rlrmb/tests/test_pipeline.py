import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from tools.bkg4_search.rlrmb.pipeline import (
    CaseMeta, ExactCaseEvidence, EXIT_ACCEPTED, EXIT_INFRASTRUCTURE, EXIT_STRATEGIC,
    LegalStateCaptureCache, TerminalVerdict, build_routability_evidence, build_cegis_input,
    capture_candidate_legal_states, emit_candidate_route, run_production_pipeline,
)
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
from tools.bkg4_search.rlrmb.cli import _parser, main as cli_main


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

    def test_outer_initiation_capture_is_reused_without_losing_candidate_binding(self):
        other=program(Phase(ActionSource.PARENT,(),fixed(2)),cap=2)
        cache=LegalStateCaptureCache()
        def fake_run(command,**unused):
            Path(command[command.index("--out")+1]).write_bytes(b"capture")
            Path(command[command.index("--truth-out")+1]).write_bytes(b"truth")
        with tempfile.TemporaryDirectory() as directory, \
             patch("tools.bkg4_search.rlrmb.pipeline.subprocess.run",side_effect=fake_run), \
             patch("tools.bkg4_search.rlrmb.pipeline.load_candidate_legal_states",side_effect=lambda p,r,*args:r):
            root=Path(directory);cases=root/"cases.tsv";cases.write_text("unused\n")
            first=(self.row("a",DatasetRole.TARGET,1,candidate=self.program.semantic_id),)
            second=(self.row("b",DatasetRole.TARGET,1,candidate=other.semantic_id),)
            capture_candidate_legal_states(self.program,first,cases,root/"one",capture_binary="capture",asset_dir=root,cache=cache)
            capture_candidate_legal_states(other,second,cases,root/"two",capture_binary="capture",asset_dir=root,cache=cache)
            self.assertEqual((cache.launches,cache.reuses),(1,1))
            self.assertEqual((root/"one"/f"{self.program.semantic_id}.b4is").stat().st_ino,
                             (root/"two"/f"{other.semantic_id}.b4is").stat().st_ino)
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

    @staticmethod
    def production_engine(metrics):
        def run(config, output, validate_only, smoke):
            output.mkdir(parents=True, exist_ok=True)
            source=json.loads(config.read_text())
            for name,value in {
                "config.json":source,
                "run.json":{"verdict":"CAMPAIGN_COMPLETE"},
                "metrics.json":metrics,
                "failures.json":{"counts":{}},
                "archive_summary.json":{"families":{}},
            }.items():
                (output/name).write_text(json.dumps(value))
            if metrics.get("accepted_specialists"):
                documents={
                    "routability":{"classification":"ROUTABLE_CANDIDATE"},
                    "cegis":{"terminal_result":"SAFE_CONVERGENCE","accepted":True},
                    "materialization":{"materialization_id":"synthetic-materialization"},
                    "composition":{"unresolved_conflicts":[]},
                    "parity":{"semantic_mismatches":0},
                }
                for directory,document in documents.items():
                    path=output/directory;path.mkdir();(path/"result.json").write_text(json.dumps(document))
            return 0
        return run

    def production_metrics(self, accepted=0, refined=0, remaining=10, mismatches=0):
        return {"total_cuda_games":20,"accepted_specialists":accepted,
                "accepted_candidate_ids":["candidate"] if accepted else [],
                "refined_candidates":refined,"semantic_mismatches":mismatches,
                "budget":{"remaining":remaining},"families":{}}

    def test_production_terminal_verdicts_and_artifacts(self):
        config=Path("tools/bkg4_search/rlrmb/configs/p0_7_equal_budget_v1.json")
        with tempfile.TemporaryDirectory() as directory:
            accepted=Path(directory)/"accepted"
            code=run_production_pipeline(config,accepted,engine=self.production_engine(self.production_metrics(accepted=1)))
            self.assertEqual(code,EXIT_ACCEPTED)
            run=json.loads((accepted/"run.json").read_text())
            self.assertEqual(run["terminal_verdict"],TerminalVerdict.ACCEPTED_SPECIALIST.value)
            self.assertTrue((accepted/"checkpoints.json").exists())
            for name in ("routability","cegis","materialization","composition","parity"):
                self.assertTrue((accepted/name/"result.json").exists())

            rejected=Path(directory)/"rejected"
            code=run_production_pipeline(config,rejected,engine=self.production_engine(self.production_metrics(refined=1)))
            self.assertEqual(code,EXIT_STRATEGIC)
            self.assertEqual(json.loads((rejected/"run.json").read_text())["terminal_verdict"],
                             TerminalVerdict.STRATEGIC_REJECTION.value)

    def test_production_infrastructure_failure_is_fail_closed(self):
        config=Path("tools/bkg4_search/rlrmb/configs/p0_7_equal_budget_v1.json")
        def broken(*args): raise RuntimeError("CUDA fixture fault")
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory)/"broken"
            self.assertEqual(run_production_pipeline(config,output,engine=broken),EXIT_INFRASTRUCTURE)
            self.assertEqual(json.loads((output/"run.json").read_text())["terminal_verdict"],
                             TerminalVerdict.INFRASTRUCTURE_FAILURE.value)
            self.assertTrue((output/"traceback.log").exists())

    def test_production_semantic_mismatch_is_infrastructure_failure(self):
        config=Path("tools/bkg4_search/rlrmb/configs/p0_7_equal_budget_v1.json")
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory)/"mismatch"
            code=run_production_pipeline(config,output,engine=self.production_engine(self.production_metrics(mismatches=1)))
            self.assertEqual(code,EXIT_INFRASTRUCTURE)

    def test_production_run_identity_is_deterministic(self):
        config=Path("tools/bkg4_search/rlrmb/configs/p0_7_equal_budget_v1.json")
        with tempfile.TemporaryDirectory() as directory:
            ids=[]
            for name in ("one","two"):
                output=Path(directory)/name
                self.assertEqual(run_production_pipeline(config,output,engine=self.production_engine(self.production_metrics(accepted=1))),0)
                ids.append(json.loads((output/"run.json").read_text())["run_id"])
            self.assertEqual(ids[0],ids[1])

    def test_production_resume_is_explicitly_unsupported(self):
        config=Path("tools/bkg4_search/rlrmb/configs/p0_7_equal_budget_v1.json")
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory)/"resume"
            self.assertEqual(run_production_pipeline(config,output,resume=True),EXIT_INFRASTRUCTURE)

    def test_cli_validation_writes_command_and_exit_code_without_answer_controls(self):
        config="tools/bkg4_search/rlrmb/configs/p0_7_equal_budget_v1.json"
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory)/"validate"
            self.assertEqual(cli_main(("--config",config,"--output",str(output),"--validate-config")),0)
            self.assertEqual((output/"exit_code.txt").read_text().strip(),"0")
            self.assertIn("--validate-config",(output/"command.txt").read_text())
        options={option for action in _parser()._actions for option in action.option_strings}
        self.assertTrue({"--config","--output","--validate-config","--smoke","--resume"}.issubset(options))
        self.assertFalse({"--selected-route","--accepted-candidate","--final-gate","--force-accept"}&options)


if __name__ == "__main__": unittest.main()
