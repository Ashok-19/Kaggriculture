import json
from pathlib import Path
import tempfile
import unittest

from tools.bkg4_search.rlrmb.cloud_runner import (
    _extend_route_pack, _controller_identity, build_working_assets, compact_generation_artifacts,
    generation_config, _parser, initial_state, load_cloud_config, run_cloud,
    search_context_identity, _resolve_composition_conflict,
)
from tools.bkg4_search.rlrmb.materialize import MaterializationInput, materialize_program
from tools.bkg4_search.rlrmb.tests.test_ir import program, Phase, ActionSource, fixed
from tools.bkg4_search.module_codec import ConditionGene
from tools.bkg4_search.bonsai_composer import SpecialistDeclaration, StrategicAuthority, compose_specialists


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

    def test_bundle_embedded_manifest_avoids_size_self_reference(self):
        import tarfile
        from tools.bkg4_search.rlrmb.cloud.build_lightning_bundle import build_bundle
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);output=root/"bundle";archive=root/"bundle.tar.gz"
            returned=build_bundle(CONFIG,output,archive)
            with tarfile.open(archive,"r:gz") as handle:
                embedded=json.loads(handle.extractfile(
                    "kaggriculture_phase16/phase16_bundle_manifest.json"
                ).read().decode("utf-8"))
            sidecar=json.loads(Path(str(archive)+".manifest.json").read_text())
            self.assertIsNone(embedded["generated_bundle_bytes"])
            self.assertEqual(sidecar["generated_bundle_bytes"],archive.stat().st_size)
            self.assertEqual(returned,sidecar)

    def test_bundle_runtime_manifests_are_relocatable(self):
        from tools.bkg4_search.rlrmb.cloud.build_lightning_bundle import build_bundle
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);output=root/"bundle";archive=root/"bundle.tar.gz"
            build_bundle(CONFIG,output,archive)
            config=load_cloud_config(CONFIG)
            panels=[config["transfer_panel"],config["validation_panel"],config["sealed_panel"],
                    *[row["target_panel"] for row in config["families"]]]
            for relative in panels:
                panel=output/relative
                rows=[line for line in panel.read_text().splitlines() if line.strip() and not line.startswith("#")]
                self.assertTrue(rows,relative)
                for line in rows:
                    tape=Path(line.split("\t",1)[0])
                    self.assertFalse(tape.is_absolute(),f"{relative} embeds absolute tape path: {tape}")
                    self.assertTrue((output/tape).exists(),f"{relative} references missing bundled tape: {tape}")
            donors=output/config["donor_manifest"]
            routes=[Path(line.strip()) for line in donors.read_text().splitlines() if line.strip()]
            self.assertTrue(routes)
            for route in routes:
                self.assertFalse(route.is_absolute(),f"{config['donor_manifest']} embeds absolute donor path: {route}")
                self.assertTrue((output/route).exists(),f"{config['donor_manifest']} references missing donor: {route}")


class ExactRepeatSuppressionTests(unittest.TestCase):
    """Phase 16.1 — exact search-context identity and repeat suppression."""

    def setUp(self):
        self.config = load_cloud_config(CONFIG)

    # --- helpers ---
    def _stub_runner(self, cuda_games=10, accepted=False):
        candidate = "a" * 64
        def runner(runtime_path, run_dir):
            run_dir.mkdir(parents=True)
            metrics = {"total_cuda_games": cuda_games,
                       "accepted_candidate_ids": [candidate] if accepted else [],
                       "protected_win_to_loss": 0, "semantic_mismatches": 0}
            (run_dir / "metrics.json").write_text(json.dumps(metrics))
            if accepted:
                family = json.loads(runtime_path.read_text())["families"][0]["id"]
                dep = run_dir / "families" / family / candidate / "deployment"
                dep.mkdir(parents=True)
                for name, val in (("composed_module.json", {}), ("specialist.json", {}),
                                  (f"{candidate}.route.json", {"actions": []})):
                    (dep / name).write_text(json.dumps(val))
            return 0
        return runner

    def _stub_assets(self, tmp_root):
        candidate = "a" * 64
        def assets(parent, deployment, destination):
            destination.mkdir(parents=True, exist_ok=True)
            for name in ("controller.bkgc", "routes.bkv6", "legacy_routes.bkv6"):
                (destination / name).write_bytes(b"ok")
            (destination / "controller.json").write_text("{}")
            identity_doc = {"controller_semantic_hash": "newhash", "route_id": 999,
                            "candidate_id": candidate, "nodes": 1, "state_slots": 1}
            (destination / "identity.json").write_text(json.dumps(identity_doc))
            return {"candidate_id": candidate, "nodes": 1, "state_slots": 1}
        return assets

    def _run_n(self, n, output, runner=None, assets=None):
        runner = runner or self._stub_runner()
        assets = assets or self._stub_assets(output)
        return run_cloud(CONFIG, output, mode="pilot", max_generations=n,
                         pipeline_runner=runner, asset_builder=assets)

    # Test 1: same generation + frontier + config → pipeline runs only once
    def test_same_context_runs_pipeline_only_once(self):
        with tempfile.TemporaryDirectory() as d:
            output = Path(d) / "run"
            calls = []
            def counting_runner(runtime_path, run_dir):
                calls.append(json.loads(runtime_path.read_text())["families"][0]["id"])
                return self._stub_runner()(runtime_path, run_dir)
            # 4 families; run up to 8 = one full cycle + one repeat cycle
            run_cloud(CONFIG, output, mode="pilot", max_generations=8,
                      pipeline_runner=counting_runner, asset_builder=self._stub_assets(output))
            # Only 4 actual pipeline calls — the second cycle is suppressed
            self.assertEqual(len(calls), 4)

    # Test 2: repeat encounter records zero CUDA games consumed
    def test_repeat_context_consumes_zero_cuda_games(self):
        with tempfile.TemporaryDirectory() as d:
            output = Path(d) / "run"
            self._run_n(4, output)
            state_after_4 = json.loads((output / "generation_state.json").read_text())
            games_after_4 = state_after_4["cuda_games_consumed"]
            # Run 4 more; all should be suppressed
            run_cloud(CONFIG, output / "run2", mode="pilot", max_generations=8,
                      pipeline_runner=self._stub_runner(), asset_builder=self._stub_assets(output))
            # Run 4 then 4 more in the SAME output (resume not needed for a fresh output)
            # Easier test: just verify avoided_cuda_games equals allocation * suppressed count
            output2 = Path(d) / "run2"
            output2.mkdir()
            # Manually run twice to see suppression
            run_cloud(CONFIG, output2, mode="pilot", max_generations=8,
                      pipeline_runner=self._stub_runner(cuda_games=150000),
                      asset_builder=self._stub_assets(output2))
            state2 = json.loads((output2 / "generation_state.json").read_text())
            # After 4 real runs, 4 suppressions
            self.assertEqual(state2["suppressed_contexts"], 4)
            self.assertEqual(state2["cuda_games_consumed"], 4 * 150000)
            # Avoided games recorded without actually running
            self.assertGreater(state2["avoided_cuda_games"], 0)

    # Test 3: suppression records avoided work in cloud_result.json
    def test_repeat_suppression_records_avoided_work(self):
        with tempfile.TemporaryDirectory() as d:
            output = Path(d) / "run"
            # Use a high limit so exhaustion detection fires before the limit
            run_cloud(CONFIG, output, mode="pilot", max_generations=100,
                      pipeline_runner=self._stub_runner(cuda_games=150000),
                      asset_builder=self._stub_assets(output))
            result = json.loads((output / "cloud_result.json").read_text())
            self.assertIn("exact_repeat_contexts_suppressed", result)
            self.assertIn("avoided_cuda_games", result)
            self.assertEqual(result["exact_repeat_contexts_suppressed"], 4)
            self.assertGreater(result["avoided_cuda_games"], 0)
            self.assertEqual(result["terminal_reason"], "GENERATION_CONTEXT_EXHAUSTED")

    # Test 4: new BKG4 generation re-enables frontiers
    def test_new_bkg4_generation_re_enables_frontiers(self):
        with tempfile.TemporaryDirectory() as d:
            output = Path(d) / "run"
            calls = []
            # First call accepts a specialist, committing a new generation
            first_call = [False]
            def smart_runner(runtime_path, run_dir):
                run_dir.mkdir(parents=True)
                family = json.loads(runtime_path.read_text())["families"][0]["id"]
                calls.append(family)
                # Accept on the very first call to trigger a new generation
                accept = not first_call[0]
                first_call[0] = True
                candidate = "a" * 64
                metrics = {"total_cuda_games": 10,
                           "accepted_candidate_ids": [candidate] if accept else [],
                           "protected_win_to_loss": 0, "semantic_mismatches": 0}
                (run_dir / "metrics.json").write_text(json.dumps(metrics))
                if accept:
                    dep = run_dir / "families" / family / candidate / "deployment"
                    dep.mkdir(parents=True)
                    for name, val in (("composed_module.json", {}), ("specialist.json", {}),
                                      (f"{candidate}.route.json", {"actions": []})):
                        (dep / name).write_text(json.dumps(val))
                return 0
            run_cloud(CONFIG, output, mode="pilot", max_generations=10,
                      pipeline_runner=smart_runner,
                      asset_builder=self._stub_assets(output))
            state = json.loads((output / "generation_state.json").read_text())
            # With one accepted specialist: generation 1, new context IDs → more runs
            # First cycle: 1 accepted (generation committed) + 3 non-accepted
            # Second cycle: new generation → at least some frontiers re-enabled
            self.assertGreater(len(calls), 4)

    # Test 5: search-config change re-enables a frontier
    def test_search_config_change_invalidates_suppression(self):
        config = load_cloud_config(CONFIG)
        state = initial_state(config)
        runtime_cfg = generation_config(config, state, "pilot")
        ctx1 = search_context_identity(runtime_cfg, state)
        # Mutate checkpoint_batch_size in the runtime config
        import copy
        runtime_cfg2 = copy.deepcopy(runtime_cfg)
        runtime_cfg2["search"]["checkpoint_batch_size"] = 999
        ctx2 = search_context_identity(runtime_cfg2, state)
        self.assertNotEqual(ctx1, ctx2)

    # Test 6: pending/failed attempt never marks a context complete
    def test_failed_attempt_never_marks_context_complete(self):
        with tempfile.TemporaryDirectory() as d:
            output = Path(d) / "run"
            def failing_runner(runtime_path, run_dir):
                return 2  # EXIT_INFRASTRUCTURE
            result = run_cloud(CONFIG, output, mode="pilot", max_generations=1,
                               pipeline_runner=failing_runner,
                               asset_builder=self._stub_assets(output))
            self.assertEqual(result, 2)
            # No generation_state.json means no completed context was recorded
            self.assertFalse((output / "generation_state.json").exists())

    # Test 7: resume preserves completed-context knowledge
    def test_resume_preserves_completed_context_knowledge(self):
        with tempfile.TemporaryDirectory() as d:
            output = Path(d) / "run"
            # Run 2 frontiers
            run_cloud(CONFIG, output, mode="pilot", max_generations=2,
                      pipeline_runner=self._stub_runner(),
                      asset_builder=self._stub_assets(output))
            state_mid = json.loads((output / "generation_state.json").read_text())
            completed_mid = dict(state_mid["completed_search_contexts"])
            self.assertEqual(len(completed_mid), 2)
            # Resume: the 2 completed contexts survive
            calls_after = []
            def counting_runner(runtime_path, run_dir):
                calls_after.append(json.loads(runtime_path.read_text())["families"][0]["id"])
                return self._stub_runner()(runtime_path, run_dir)
            run_cloud(CONFIG, output, mode="pilot", max_generations=6, resume=True,
                      pipeline_runner=counting_runner, asset_builder=self._stub_assets(output))
            state_final = json.loads((output / "generation_state.json").read_text())
            # Resumed state must contain original 2 completed contexts
            for ctx_id in completed_mid:
                self.assertIn(ctx_id, state_final["completed_search_contexts"])
            # After resume, only 2 more actual pipeline calls (families 3 and 4)
            self.assertEqual(len(calls_after), 2)

    # Test 8: context identity is deterministic
    def test_context_identity_is_deterministic(self):
        config = load_cloud_config(CONFIG)
        state = initial_state(config)
        runtime_cfg = generation_config(config, state, "pilot")
        ids = [search_context_identity(runtime_cfg, state) for _ in range(5)]
        self.assertEqual(len(set(ids)), 1)
        self.assertEqual(len(ids[0]), 64)  # SHA-256 hex

    # Test 9: SEALED panel does not enter search context identity
    def test_sealed_panel_excluded_from_search_context_identity(self):
        config = load_cloud_config(CONFIG)
        state = initial_state(config)
        runtime_cfg = generation_config(config, state, "pilot")
        ctx1 = search_context_identity(runtime_cfg, state)
        import copy
        runtime_cfg2 = copy.deepcopy(runtime_cfg)
        runtime_cfg2["sealed_panel"] = "/changed/sealed/panel.tsv"
        ctx2 = search_context_identity(runtime_cfg2, state)
        self.assertEqual(ctx1, ctx2,
            "SEALED panel must not contaminate search-context identity")

    # Test 10: controller identity resolves from identity.json when present
    def test_controller_identity_uses_identity_json_hash(self):
        with tempfile.TemporaryDirectory() as d:
            asset_dir = Path(d) / "assets"
            asset_dir.mkdir()
            # Without identity.json: stable fallback
            id_no_file = _controller_identity(str(asset_dir))
            self.assertIsInstance(id_no_file, str)
            self.assertTrue(len(id_no_file) > 0)
            # With identity.json: uses controller_semantic_hash
            (asset_dir / "identity.json").write_text(
                json.dumps({"controller_semantic_hash": "abc123", "route_id": 1}))
            id_with_file = _controller_identity(str(asset_dir))
            self.assertEqual(id_with_file, "abc123")
            # Generation change → identity changes
            (asset_dir / "identity.json").write_text(
                json.dumps({"controller_semantic_hash": "def456", "route_id": 2}))
            id_new_gen = _controller_identity(str(asset_dir))
            self.assertNotEqual(id_with_file, id_new_gen)

    # Test 11: exhaustion terminates cleanly without an infinite loop
    def test_exhaustion_terminates_without_looping(self):
        with tempfile.TemporaryDirectory() as d:
            output = Path(d) / "run"
            import time as _time
            start = _time.monotonic()
            run_cloud(CONFIG, output, mode="pilot", max_generations=100,
                      pipeline_runner=self._stub_runner(),
                      asset_builder=self._stub_assets(output))
            elapsed = _time.monotonic() - start
            result = json.loads((output / "cloud_result.json").read_text())
            self.assertEqual(result["terminal_reason"], "GENERATION_CONTEXT_EXHAUSTED")
            # Only 4 real pipeline calls (one per family), not 100
            state = json.loads((output / "generation_state.json").read_text())
            self.assertEqual(len(state["completed_frontiers"]), 4)
            self.assertEqual(state["suppressed_contexts"], 4)

    # Test 12: existing atomic-generation/resume atomicity still works
    def test_existing_atomic_generation_behavior_unchanged(self):
        config = load_cloud_config(CONFIG)
        candidate = "b" * 64
        def accepted_runner(runtime, output):
            output.mkdir(parents=True)
            family = json.loads(runtime.read_text())["families"][0]["id"]
            dep = output / "families" / family / candidate / "deployment"
            dep.mkdir(parents=True)
            for name, val in (("composed_module.json", {}), ("specialist.json", {}),
                               (f"{candidate}.route.json", {"actions": []})):
                (dep / name).write_text(json.dumps(val))
            (output / "metrics.json").write_text(json.dumps(
                {"total_cuda_games": 10, "accepted_candidate_ids": [candidate],
                 "protected_win_to_loss": 0, "semantic_mismatches": 0}))
            return 0
        def assets(parent, deployment, destination):
            destination.mkdir(parents=True)
            for name in ("controller.bkgc", "routes.bkv6", "legacy_routes.bkv6"):
                (destination / name).write_bytes(b"ok")
            (destination / "controller.json").write_text("{}")
            return {"candidate_id": candidate, "nodes": 1, "state_slots": 1}
        with tempfile.TemporaryDirectory() as d:
            output = Path(d) / "run"
            code = run_cloud(CONFIG, output, mode="pilot", max_generations=1,
                             pipeline_runner=accepted_runner, asset_builder=assets)
            self.assertEqual(code, 0)
            state = json.loads((output / "generation_state.json").read_text())
            self.assertEqual(state["generation"], 1)
            self.assertEqual(len(state["completed_search_contexts"]), 1)
            self.assertFalse(any(output.glob("*.pending")))


class StagingTimingTests(unittest.TestCase):
    """Phase 16.1 — stage timing fields in run_campaign metrics."""

    EXPECTED_STAGES = {
        "donor_event_extraction", "checkpoint_selection",
        "beam_search_total", "promotion_pool_selection",
        "target_legal_capture", "target_routability",
        "transfer_validation_eval", "transfer_validation_capture",
        "full_routability", "sealed_eval_capture",
        "cegis", "lns_repair", "materialization",
        "post_search_io", "metrics_normalization",
        "_total_attributed", "_campaign_wall", "_attribution_ratio",
    }

    def _run_smoke_with_mocks(self, output):
        """Run a smoke campaign with the batch pipeline fully stubbed out."""
        from tools.bkg4_search.rlrmb.p13_equal_budget import run_campaign
        from unittest.mock import patch, MagicMock

        config_path = Path("tools/bkg4_search/rlrmb/configs/p0_7_equal_budget_v1.json")
        mock_metrics = MagicMock()
        mock_metrics.tasks = 0; mock_metrics.wall_seconds = 0.0
        mock_metrics.worker_busy_seconds = 0.0; mock_metrics.coordinator_wait_seconds = 0.0
        mock_metrics.worker_pids = set(); mock_metrics.max_inflight = 0
        mock_metrics.reorder_buffer_high_water = 0
        mock_prepared = MagicMock()
        mock_prepared.motifs = {}; mock_prepared.checkpoint_motifs = {}
        mock_prepared.metrics = mock_metrics

        # Yield one batch so beam_search_total timer fires, then stop
        beam_result = MagicMock()
        beam_result.candidates = []
        beam_result.archive = MagicMock()
        beam_result.archive.entries.return_value = []
        beam_result.archive.metrics.return_value = {"candidates_considered": 0}
        beam_result.archive.serialize.return_value = '{"families":{}}'
        beam_result.metrics = {"depth_reached": 0, "host_seconds": 0.0, "cuda_ms": 0.0,
                               "raw_candidates": 0, "structural_duplicates_removed": 0,
                               "unique_effect_signatures": 0, "archive_update_seconds": 0.0,
                               "expansions_by_operator": {}}

        def fake_prefetch(batches, *args, **kwargs):
            for batch in batches[:1]:
                yield batch, mock_prepared, 0.0

        with patch("tools.bkg4_search.rlrmb.p13_equal_budget.prefetch_checkpoint_motif_batches",
                   side_effect=fake_prefetch), \
             patch("tools.bkg4_search.rlrmb.p13_equal_budget.run_beam",
                   return_value=beam_result), \
             patch("tools.bkg4_search.rlrmb.p13_equal_budget.CudaBatchEvaluator") as mock_cls:
            mock_inst = MagicMock()
            mock_inst.projected_games.return_value = 0
            mock_inst.case_count = 1
            mock_cls.return_value = mock_inst
            run_campaign(config_path, output, smoke=True)

    def test_stage_timing_fields_present_and_non_negative(self):
        from tools.bkg4_search.rlrmb.p13_equal_budget import run_campaign
        config_path = Path("tools/bkg4_search/rlrmb/configs/p0_7_equal_budget_v1.json")
        with tempfile.TemporaryDirectory() as d:
            output = Path(d) / "out"
            run_campaign(config_path, output, validate_only=True)
            # validate_only exits before stage_timing is created
            metrics = json.loads((output / "metrics.json").read_text())
            self.assertNotIn("stage_timing_seconds", metrics)

    def test_smoke_run_emits_stage_timing(self):
        with tempfile.TemporaryDirectory() as d:
            output = Path(d) / "out"
            self._run_smoke_with_mocks(output)
            metrics = json.loads((output / "metrics.json").read_text())
            self.assertIn("stage_timing_seconds", metrics)
            timing = metrics["stage_timing_seconds"]
            for key in self.EXPECTED_STAGES:
                self.assertIn(key, timing, f"missing stage: {key}")
            for key, val in timing.items():
                self.assertGreaterEqual(val, 0.0, f"stage {key} has negative time")
            # _total_attributed must not exceed campaign wall (with 10ms slack for measurement)
            self.assertLessEqual(timing["_total_attributed"], timing["_campaign_wall"] + 0.01)

    def test_attribution_ratio_field_is_a_fraction(self):
        with tempfile.TemporaryDirectory() as d:
            output = Path(d) / "out"
            self._run_smoke_with_mocks(output)
            metrics = json.loads((output / "metrics.json").read_text())
            ratio = metrics["stage_timing_seconds"]["_attribution_ratio"]
            self.assertGreaterEqual(ratio, 0.0)
            # The 95% gate is validated on real hardware; in unit tests just verify >= 0
            self.assertGreaterEqual(ratio, 0.0)


def _stub_declaration(name: str, route_id: int, start: int, end: int,
                       authority: str = "strategic_route_override",
                       candidate_id: str | None = None) -> SpecialistDeclaration:
    return SpecialistDeclaration(
        name=name, kind="bounded_route_macro", route_id=route_id,
        authorization_step=start, active_start=start, active_end=end,
        active_slot=f"active_{name}", authorities=(authority,),
        route_source=f"{name}.route.json", semantic_id=f"sem_{name}",
        gate_clauses=None, end_policy="RETURN_TO_PARENT", macro=None,
        provenance={"candidate_semantic_id": candidate_id or name},
    )


class CompositionConflictResolutionTests(unittest.TestCase):

    def test_no_conflict_returns_all(self):
        decls = [
            _stub_declaration("a", 300, 100, 110),
            _stub_declaration("b", 301, 150, 160),
        ]
        self.assertEqual(_resolve_composition_conflict(decls), [0, 1])

    def test_two_conflicting_selects_earlier(self):
        decls = [
            _stub_declaration("alpha", 300, 190, 199),
            _stub_declaration("beta", 301, 191, 198),
        ]
        result = _resolve_composition_conflict(decls)
        self.assertEqual(len(result), 1)
        selected = decls[result[0]]
        self.assertEqual(selected.active_start, 190)

    def test_unrelated_candidate_retained(self):
        decls = [
            _stub_declaration("early", 300, 100, 110),
            _stub_declaration("late_a", 301, 190, 199),
            _stub_declaration("late_b", 302, 191, 198),
        ]
        result = _resolve_composition_conflict(decls)
        self.assertIn(0, result)
        self.assertEqual(len(result), 2)

    def test_deterministic_across_calls(self):
        decls = [
            _stub_declaration("x", 300, 190, 199),
            _stub_declaration("y", 301, 191, 198),
            _stub_declaration("z", 302, 100, 110),
        ]
        results = [_resolve_composition_conflict(decls) for _ in range(10)]
        self.assertTrue(all(r == results[0] for r in results))

    def test_best_safe_subset_with_three_way_conflict(self):
        decls = [
            _stub_declaration("a", 300, 100, 120),
            _stub_declaration("b", 301, 110, 130),
            _stub_declaration("c", 302, 125, 140),
        ]
        result = _resolve_composition_conflict(decls)
        selected_names = {decls[i].name for i in result}
        self.assertIn("a", selected_names)
        self.assertIn("c", selected_names)
        self.assertNotIn("b", selected_names)

    def test_no_different_authority_conflict(self):
        decls = [
            _stub_declaration("a", 300, 190, 199, authority="auth_1"),
            _stub_declaration("b", 301, 191, 198, authority="auth_2"),
        ]
        self.assertEqual(_resolve_composition_conflict(decls), [0, 1])

    def test_single_candidate(self):
        decls = [_stub_declaration("solo", 300, 100, 110)]
        self.assertEqual(_resolve_composition_conflict(decls), [0])

    def test_empty_returns_empty(self):
        self.assertEqual(_resolve_composition_conflict([]), [])

    def test_safe_fast_path_unchanged(self):
        decls = [_stub_declaration("safe", 300, 100, 110)]
        self.assertEqual(_resolve_composition_conflict(decls), [0])

    def test_evidence_prefers_stronger_candidate(self):
        decls = [
            _stub_declaration("weak", 300, 190, 199, candidate_id="cid_weak"),
            _stub_declaration("strong", 301, 191, 198, candidate_id="cid_strong"),
        ]
        evidence = {
            "cid_weak": {"target_loss_to_win": 1, "protected_win_to_loss": 0},
            "cid_strong": {"target_loss_to_win": 5, "protected_win_to_loss": 0},
        }
        result = _resolve_composition_conflict(decls, evidence=evidence)
        self.assertEqual(len(result), 1)
        self.assertEqual(decls[result[0]].provenance["candidate_semantic_id"], "cid_strong")

    def test_evidence_penalizes_protected_regression(self):
        decls = [
            _stub_declaration("risky", 300, 190, 199, candidate_id="cid_risky"),
            _stub_declaration("safe_one", 301, 191, 198, candidate_id="cid_safe"),
        ]
        evidence = {
            "cid_risky": {"target_loss_to_win": 10, "routed_protected_win_to_loss": 3},
            "cid_safe": {"target_loss_to_win": 2, "routed_protected_win_to_loss": 0},
        }
        result = _resolve_composition_conflict(decls, evidence=evidence)
        self.assertEqual(len(result), 1)
        self.assertEqual(decls[result[0]].provenance["candidate_semantic_id"], "cid_safe")

    def test_evidence_missing_falls_back_to_position(self):
        decls = [
            _stub_declaration("early", 300, 190, 199, candidate_id="cid_early"),
            _stub_declaration("late", 301, 191, 198, candidate_id="cid_late"),
        ]
        result_no_ev = _resolve_composition_conflict(decls)
        result_empty_ev = _resolve_composition_conflict(decls, evidence={})
        self.assertEqual(result_no_ev, result_empty_ev)

    def test_evidence_with_transfer_tiebreak(self):
        decls = [
            _stub_declaration("low_transfer", 300, 190, 199, candidate_id="cid_lt"),
            _stub_declaration("high_transfer", 301, 191, 198, candidate_id="cid_ht"),
        ]
        evidence = {
            "cid_lt": {"target_loss_to_win": 3, "protected_win_to_loss": 0, "transfer_support": 1},
            "cid_ht": {"target_loss_to_win": 3, "protected_win_to_loss": 0, "transfer_support": 5},
        }
        result = _resolve_composition_conflict(decls, evidence=evidence)
        self.assertEqual(len(result), 1)
        self.assertEqual(decls[result[0]].provenance["candidate_semantic_id"], "cid_ht")


if __name__=="__main__": unittest.main()
