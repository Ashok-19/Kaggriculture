import json
from pathlib import Path
import tempfile
import unittest

from tools.bkg4_search.rlrmb.cloud_runner import (
    _extend_route_pack, build_working_assets, compact_generation_artifacts, generation_config,
    _parser, initial_state, load_cloud_config, run_cloud,
)
from tools.bkg4_search.rlrmb.materialize import MaterializationInput, materialize_program
from tools.bkg4_search.rlrmb.tests.test_ir import program, Phase, ActionSource, fixed
from tools.bkg4_search.module_codec import ConditionGene
from tools.bkg4_search.bonsai_composer import StrategicAuthority, compose_specialists


ROOT=Path(__file__).resolve().parents[4]
CONFIG=ROOT/"tools/bkg4_search/rlrmb/configs/phase16_lightning_v1.json"


class CloudTests(unittest.TestCase):
    def test_mode_config_is_bounded_and_deterministic(self):
        config=load_cloud_config(CONFIG);state=initial_state(config)
        source_snapshot=json.dumps(config,sort_keys=True)
        first=generation_config(config,state,"pilot");second=generation_config(config,state,"pilot")
        self.assertEqual(first,second)
        self.assertEqual(len(first["families"]),1)
        self.assertEqual(first["families"][0]["id"],"family_02_offline_0")
        self.assertEqual(sum(first["budget"]["per_family_categories"].values()),150000)
        self.assertEqual(first["search"]["checkpoint_batch_size"],8)
        self.assertEqual(first["host_execution"]["producer_workers"],8)
        self.assertEqual(first["host_execution"]["queue_depth"],2)
        self.assertEqual(first["host_execution"]["start_method"],"spawn")
        self.assertEqual(config["cloud"]["expected_hardware"]["system_ram_gb"],128)
        tuned=generation_config(config,state,"pilot",operational_overrides={
            "host_workers":4,"queue_depth":3,"checkpoint_batch_size":4,
        })
        self.assertEqual(tuned["host_execution"]["producer_workers"],4)
        self.assertEqual(tuned["host_execution"]["queue_depth"],3)
        self.assertEqual(tuned["search"]["checkpoint_batch_size"],4)
        self.assertEqual(json.dumps(config,sort_keys=True),source_snapshot)
        self.assertEqual(initial_state(config)["config_identity"],state["config_identity"])
        options={option for action in _parser()._actions for option in action.option_strings}
        self.assertTrue({"--host-workers","--queue-depth","--checkpoint-batch-size","--telemetry-interval-seconds"}.issubset(options))

    def test_route_pack_extension_preserves_existing_and_candidate_mapping(self):
        asset=ROOT/"bonsai-kg/integrations/bkg4_champion/champions/bounded_automatic_composer_v1"
        route=ROOT/"scratch/p0_1_macro_search/donors/route_64.json"
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory)/"routes.bkv6"
            _extend_route_pack(asset/"routes.bkv6",route,999,output)
            self.assertGreater(output.stat().st_size,(asset/"routes.bkv6").stat().st_size)
            with self.assertRaises(ValueError):
                _extend_route_pack(asset/"routes.bkv6",route,64,Path(directory)/"collision.bkv6")

    def test_composed_candidate_builds_next_working_bkg4_assets(self):
        asset=ROOT/"bonsai-kg/integrations/bkg4_champion/champions/bounded_automatic_composer_v1"
        candidate=program(Phase(ActionSource.PARENT,(),fixed(1)),step=1,cap=1)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);deployment=root/"deployment";deployment.mkdir()
            route=deployment/f"{candidate.semantic_id}.route.json"
            route.write_bytes((ROOT/"scratch/p0_1_macro_search/donors/route_64.json").read_bytes())
            artifact=materialize_program(MaterializationInput(candidate.semantic_id,candidate,
                ((ConditionGene("OWN_MONEY","GE",0,0),),),{},999,64,str(route),"strategic_route_override","day2_capital_specialist_v1_route_override"))
            parent=json.loads((asset/"controller.json").read_text())
            composed=compose_specialists(parent,(artifact.specialist,),(),
                strategic_authorities=(StrategicAuthority("strategic_route_override","day2_capital_specialist_v1_route_override"),),output_name="cloud_fixture")
            (deployment/"composed_module.json").write_text(json.dumps(composed.module))
            (deployment/"specialist.json").write_text(json.dumps(artifact.specialist.canonical()))
            identity=build_working_assets(asset,deployment,root/"next")
            self.assertEqual(identity["candidate_id"],candidate.semantic_id)
            for name in ("controller.bkgc","controller.json","routes.bkv6","legacy_routes.bkv6","identity.json"):
                self.assertTrue((root/"next"/name).exists())

    def test_committed_generation_is_atomic_and_resume_safe(self):
        config=load_cloud_config(CONFIG)
        candidate="a"*64
        def accepted(runtime,output):
            output.mkdir(parents=True)
            family=json.loads(runtime.read_text())["families"][0]["id"]
            deployment=output/"families"/family/candidate/"deployment";deployment.mkdir(parents=True)
            for name,value in (("composed_module.json",{}),("specialist.json",{}),(f"{candidate}.route.json",{"actions":[]})):
                (deployment/name).write_text(json.dumps(value))
            (output/"metrics.json").write_text(json.dumps({"total_cuda_games":10,"accepted_candidate_ids":[candidate],
                "protected_win_to_loss":0,"semantic_mismatches":0}))
            return 0
        def assets(parent,deployment,destination):
            destination.mkdir();(destination/"controller.bkgc").write_bytes(b"ok")
            (destination/"controller.json").write_text("{}")
            (destination/"routes.bkv6").write_bytes(b"ok")
            (destination/"legacy_routes.bkv6").write_bytes(b"ok")
            return {"candidate_id":candidate,"nodes":1,"state_slots":1}
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory)/"run"
            self.assertEqual(run_cloud(CONFIG,output,mode="pilot",max_generations=1,
                                       pipeline_runner=accepted,asset_builder=assets),0)
            state=json.loads((output/"generation_state.json").read_text())
            self.assertEqual((state["generation"],state["cuda_games_consumed"]),(1,10))
            self.assertTrue((output/"generations/generation_0001/working_assets/controller.bkgc").exists())
            self.assertFalse(any(output.glob("*.pending")))

    def test_partial_infrastructure_failure_never_commits(self):
        def failed(runtime,output): return 2
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory)/"run"
            self.assertEqual(run_cloud(CONFIG,output,mode="pilot",max_generations=1,pipeline_runner=failed),2)
            self.assertFalse((output/"generation_state.json").exists())
            self.assertFalse((output/"generations").exists())

    def test_compaction_keeps_only_accepted_exact_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            run=Path(directory);family=run/"families"/"frontier"
            for name in ("prepromotion","accepted","rejected"):
                path=family/name;path.mkdir(parents=True);(path/"evidence.bin").write_bytes(b"exact")
            compact_generation_artifacts(run,("accepted",))
            self.assertTrue((family/"accepted/evidence.bin").exists())
            self.assertFalse((family/"prepromotion").exists())
            self.assertFalse((family/"rejected").exists())


if __name__=="__main__": unittest.main()
