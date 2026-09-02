from __future__ import annotations

import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path

from tools.bkg4_search.bonsai_composer import StrategicAuthority, compose_specialists
from tools.bkg4_search.module_codec import ConditionGene
from tools.bkg4_search.rlrmb.cloud.build_lightning_bundle import build_bundle
from tools.bkg4_search.rlrmb.cloud_runner import (
    build_working_assets,
    compact_generation_artifacts,
    load_cloud_config,
    run_cloud,
)
from tools.bkg4_search.rlrmb.cegis_adapter import AdapterDisposition, CegisAdapterResult
from tools.bkg4_search.rlrmb.materialize import MaterializationInput, materialize_program
from tools.bkg4_search.rlrmb.pipeline import materialize_certified_candidate, run_production_pipeline
from tools.bkg4_search.rlrmb.tests.test_ir import ActionSource, Phase, fixed, program


ROOT = Path(__file__).resolve().parents[4]
CONFIG = ROOT / "tools/bkg4_search/rlrmb/configs/phase16_lightning_v1.json"
PARENT_ROUTE = ROOT / "scratch/p0_1_macro_search/donors/route_64.json"


def _program_at(step: int):
    return program(Phase(ActionSource.PARENT, (), fixed(1)), step=step, cap=1)


def _asset_dir(runtime_config: dict) -> Path:
    path = Path(runtime_config["asset_dir"])
    return path if path.is_absolute() else ROOT / path


def _parent_route(runtime_config: dict) -> Path:
    path = Path(runtime_config["parent_route"])
    return path if path.is_absolute() else ROOT / path


def _write_accepted_deployment(run_dir: Path, runtime_config: dict, family_id: str, candidate, route_id: int) -> None:
    deployment = run_dir / "families" / family_id / candidate.semantic_id / "deployment"
    deployment.mkdir(parents=True)
    route = deployment / f"{candidate.semantic_id}.route.json"
    shutil.copy2(_parent_route(runtime_config), route)
    authority_name = runtime_config["deployment"]["authority_name"]
    authority_slot = runtime_config["deployment"]["authority_slot"]
    artifact = materialize_program(MaterializationInput(
        candidate.semantic_id,
        candidate,
        ((ConditionGene("OWN_MONEY", "GE", 0, 0),),),
        {},
        route_id,
        64,
        route.name,  # production materialization must persist a deployment-relative source
        authority_name,
        authority_slot,
    ))
    parent = json.loads((_asset_dir(runtime_config) / "controller.json").read_text(encoding="utf-8"))
    composed = compose_specialists(
        parent,
        (artifact.specialist,),
        (),
        strategic_authorities=(StrategicAuthority(authority_name, authority_slot),),
        output_name=f"fixture_{candidate.semantic_id[:12]}",
    )
    (deployment / "materialization.json").write_text(artifact.canonical_json() + "\n", encoding="utf-8")
    (deployment / "specialist.json").write_text(
        json.dumps(artifact.specialist.canonical(), sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    (deployment / "composed_module.json").write_text(
        json.dumps(composed.module, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _json_strings(value):
    if isinstance(value, dict):
        for nested in value.values():
            yield from _json_strings(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from _json_strings(nested)
    elif isinstance(value, str):
        yield value


class PipelineHardeningTests(unittest.TestCase):
    def test_multi_accept_commit_resume_and_route_ids_are_atomic_and_collision_free(self):
        first_a, first_b, second = _program_at(1), _program_at(10), _program_at(20)
        calls = 0
        seen_route_bases: list[int] = []

        def runner(runtime_path: Path, run_dir: Path) -> int:
            nonlocal calls
            cfg = json.loads(runtime_path.read_text(encoding="utf-8"))
            family_id = cfg["families"][0]["id"]
            route_base = int(cfg["deployment"]["route_id_base"])
            seen_route_bases.append(route_base)
            run_dir.mkdir(parents=True)
            # Materialization order is deliberately opposite semantic sort order;
            # cloud commit must rebind authoritative route IDs deterministically.
            accepted = (first_b, first_a) if calls == 0 else (second,)
            for offset, candidate in enumerate(accepted):
                _write_accepted_deployment(run_dir, cfg, family_id, candidate, route_base + offset)
            rejected = run_dir / "families" / family_id / ("f" * 64)
            rejected.mkdir(parents=True)
            (rejected / "evidence.json").write_text("{}\n", encoding="utf-8")
            (run_dir / "metrics.json").write_text(json.dumps({
                "total_cuda_games": 17,
                "accepted_candidate_ids": [candidate.semantic_id for candidate in reversed(accepted)],
                "protected_win_to_loss": 0,
                "semantic_mismatches": 0,
            }), encoding="utf-8")
            (run_dir / "run.json").write_text(json.dumps({
                "artifact_references": [
                    f"families/{family_id}/{accepted[0].semantic_id}/deployment/specialist.json",
                    f"families/{family_id}/{'f' * 64}/evidence.json",
                ]
            }), encoding="utf-8")
            calls += 1
            return 0

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "campaign"
            self.assertEqual(run_cloud(
                CONFIG, output, mode="pilot", max_generations=1,
                pipeline_runner=runner, asset_builder=build_working_assets,
                telemetry_interval_seconds=60,
            ), 0)
            state = json.loads((output / "generation_state.json").read_text(encoding="utf-8"))
            self.assertEqual(state["generation"], 1)
            self.assertEqual(state["accepted_specialists"], sorted([first_a.semantic_id, first_b.semantic_id]))
            self.assertEqual(state["cuda_games_consumed"], 17)
            first_commit = state["commits"][-1]
            self.assertEqual(first_commit["candidate_ids"], sorted([first_a.semantic_id, first_b.semantic_id]))
            self.assertEqual(first_commit["route_ids"], [1000, 1001])

            generation = output / "generations/generation_0001"
            controller = json.loads((generation / "working_assets/controller.json").read_text(encoding="utf-8"))
            route_ids = {int(row["route_id"]) for row in controller["strategic_routes"]}
            self.assertTrue({1000, 1001}.issubset(route_ids))
            expected_route_ids = dict(zip(sorted([first_a.semantic_id, first_b.semantic_id]), [1000, 1001]))
            specialist_manifest = json.loads(
                (generation / "run/generation_deployment/specialists.json").read_text(encoding="utf-8")
            )
            manifest_rows = {row["candidate_id"]: row for row in specialist_manifest["specialists"]}
            identity = json.loads((generation / "working_assets/identity.json").read_text(encoding="utf-8"))
            self.assertEqual(identity["route_ids"], [1000, 1001])
            self.assertEqual(first_commit["route_ids"], identity["route_ids"])
            for candidate_id, authoritative_route_id in expected_route_ids.items():
                self.assertEqual(int(manifest_rows[candidate_id]["route_id"]), authoritative_route_id)
                self.assertEqual(int(manifest_rows[candidate_id]["specialist"]["route_id"]), authoritative_route_id)
                deployment = next(generation.glob(f"run/families/*/{candidate_id}/deployment"))
                specialist = json.loads((deployment / "specialist.json").read_text(encoding="utf-8"))
                materialization = json.loads((deployment / "materialization.json").read_text(encoding="utf-8"))
                self.assertEqual(int(specialist["route_id"]), authoritative_route_id)
                self.assertEqual(int(materialization["module"]["strategic_routes"][0]["route_id"]), authoritative_route_id)
                self.assertEqual(specialist["provenance"]["candidate_semantic_id"], candidate_id)
                self.assertEqual(specialist["provenance"]["program_semantic_id"], candidate_id)
                self.assertEqual(materialization["candidate_id"], candidate_id)
                self.assertEqual(specialist["route_source"], f"{candidate_id}.route.json")
                self.assertTrue((deployment / specialist["route_source"]).exists())
            run_manifest = json.loads((generation / "run/run.json").read_text(encoding="utf-8"))
            for reference in run_manifest["artifact_references"]:
                self.assertTrue((generation / "run" / reference).exists(), reference)
            for json_path in generation.rglob("*.json"):
                document = json.loads(json_path.read_text(encoding="utf-8"))
                self.assertFalse(any(".pending" in value for value in _json_strings(document)), json_path)

            self.assertEqual(run_cloud(
                CONFIG, output, mode="pilot", resume=True, max_generations=1,
                pipeline_runner=runner, asset_builder=build_working_assets,
                telemetry_interval_seconds=60,
            ), 0)
            final = json.loads((output / "generation_state.json").read_text(encoding="utf-8"))
            self.assertEqual(final["generation"], 2)
            self.assertEqual(final["cuda_games_consumed"], 34)
            self.assertEqual(seen_route_bases, [1000, 1002])
            self.assertEqual(final["commits"][-1]["route_ids"], [1002])
            self.assertEqual(len(set(sum((row["route_ids"] for row in final["commits"]), []))), 3)
            self.assertEqual(final["commits"][0]["protected_win_to_loss"], 0)
            self.assertEqual(final["commits"][0]["semantic_mismatches"], 0)

    def test_incompatible_multi_accept_resolves_compatible_subset(self):
        left, right = _program_at(7), _program_at(7)
        right = program(Phase(ActionSource.PARENT, (), fixed(1)), step=7, cap=2)

        def runner(runtime_path: Path, run_dir: Path) -> int:
            cfg = json.loads(runtime_path.read_text(encoding="utf-8"))
            family_id = cfg["families"][0]["id"]
            route_base = int(cfg["deployment"]["route_id_base"])
            run_dir.mkdir(parents=True)
            _write_accepted_deployment(run_dir, cfg, family_id, left, route_base)
            _write_accepted_deployment(run_dir, cfg, family_id, right, route_base + 1)
            (run_dir / "metrics.json").write_text(json.dumps({
                "total_cuda_games": 9,
                "accepted_candidate_ids": [left.semantic_id, right.semantic_id],
                "protected_win_to_loss": 0,
                "semantic_mismatches": 0,
            }), encoding="utf-8")
            return 0

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "campaign"
            self.assertEqual(run_cloud(
                CONFIG, output, mode="pilot", max_generations=1,
                pipeline_runner=runner, asset_builder=build_working_assets,
                telemetry_interval_seconds=60,
            ), 0)
            state = json.loads((output / "generation_state.json").read_text(encoding="utf-8"))
            self.assertEqual(state["generation"], 1)
            self.assertEqual(len(state["commits"]), 1)
            self.assertEqual(len(state["commits"][0]["route_ids"]), 1)
            self.assertEqual(state["commits"][0]["protected_win_to_loss"], 0)
            self.assertEqual(state["commits"][0]["semantic_mismatches"], 0)
            generation = output / "generations" / "generation_0001"
            self.assertTrue(generation.exists())
            conflict = json.loads(
                (generation / "run" / "composition_conflict.json").read_text(encoding="utf-8")
            )
            self.assertEqual(conflict["status"], "RESOLVED_COMPATIBLE_SUBSET")
            self.assertEqual(len(conflict["selected_candidate_ids"]), 1)

    def test_compaction_refreshes_manifest_to_only_retained_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            family = run / "families/frontier"
            accepted = family / ("a" * 64)
            rejected = family / ("b" * 64)
            accepted.mkdir(parents=True)
            rejected.mkdir(parents=True)
            (accepted / "keep.json").write_text("{}\n", encoding="utf-8")
            (rejected / "drop.json").write_text("{}\n", encoding="utf-8")
            (run / "run.json").write_text(json.dumps({
                "artifact_references": [
                    f"families/frontier/{'a' * 64}/keep.json",
                    f"families/frontier/{'b' * 64}/drop.json",
                ]
            }), encoding="utf-8")
            compact_generation_artifacts(run, ("a" * 64,))
            # The production compactor must leave the manifest self-consistent itself.
            manifest = json.loads((run / "run.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["artifact_references"], [f"families/frontier/{'a' * 64}/keep.json"])
            self.assertTrue(all((run / ref).exists() for ref in manifest["artifact_references"]))

    def test_production_materialization_never_embeds_pending_route_path(self):
        candidate = _program_at(5)
        result = CegisAdapterResult(
            candidate_id=candidate.semantic_id,
            program_semantic_id=candidate.semantic_id,
            program_schema_version=candidate.schema_version,
            internal_decision_count=0,
            disposition=AdapterDisposition.ENTER_CEGIS,
            cegis_invoked=True,
            accepted=True,
            rejection_reason=None,
            cegis_stop_reason="SAFE_CONVERGENCE",
            gate=[[{"feature": "OWN_MONEY", "cmp": "GE", "index": 0, "value": 0}]],
            target_preserved=True,
            transfer_support={},
            validation_counterexamples=(),
            threshold_robustness=(),
            sealed_status="PASS",
            evidence_ids=(),
            cegis_trace=(),
            diagnostics={},
        )
        with tempfile.TemporaryDirectory() as directory:
            deployment = Path(directory) / "attempt_0000.pending/run/families/f/candidate/deployment"
            artifact, _ = materialize_certified_candidate(
                candidate,
                {},
                result,
                parent_route=PARENT_ROUTE,
                output_dir=deployment,
                route_id=1000,
            )
            specialist = json.loads((deployment / "specialist.json").read_text(encoding="utf-8"))
            materialization = json.loads((deployment / "materialization.json").read_text(encoding="utf-8"))
            self.assertEqual(specialist["route_source"], f"{candidate.semantic_id}.route.json")
            self.assertNotIn(".pending", specialist["route_source"])
            self.assertNotIn(".pending", json.dumps(materialization, sort_keys=True))
            self.assertEqual(artifact.specialist.route_source, specialist["route_source"])

    def test_bundle_source_head_reaches_run_artifact_without_git_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle = root / "bundle"
            archive = root / "bundle.tar.gz"
            manifest = build_bundle(CONFIG, bundle, archive)
            self.assertFalse((bundle / ".git").exists())
            bundled_config = bundle / "tools/bkg4_search/rlrmb/configs/phase16_lightning_v1.json"
            campaign = bundle / "scratch/hardening-provenance-run"
            # lightning_bootstrap.sh creates these generated binaries before a run.
            generated = bundle / "scratch/cuda-engine-build"
            generated.mkdir(parents=True, exist_ok=True)
            (generated / "rlrmb_eval").write_bytes(b"fixture")
            (generated / "bkg4_information_state_capture").write_bytes(b"fixture")

            def engine(config_path: Path, output: Path, validate_only: bool, smoke: bool) -> int:
                output.mkdir(parents=True, exist_ok=True)
                (output / "metrics.json").write_text(json.dumps({
                    "total_cuda_games": 3,
                    "accepted_specialists": 0,
                    "accepted_candidate_ids": [],
                    "refined_candidates": 1,
                    "semantic_mismatches": 0,
                    "protected_win_to_loss": 0,
                    "budget": {"remaining": 1},
                    "families": {},
                }), encoding="utf-8")
                (output / "failures.json").write_text(json.dumps({"counts": {}}), encoding="utf-8")
                (output / "archive_summary.json").write_text(json.dumps({"families": {}}), encoding="utf-8")
                return 0

            def pipeline_runner(runtime_path: Path, run_dir: Path) -> int:
                return run_production_pipeline(runtime_path, run_dir, engine=engine)

            previous = Path.cwd()
            os.chdir(bundle)
            try:
                self.assertEqual(run_cloud(
                    bundled_config, campaign, mode="pilot", max_generations=1,
                    pipeline_runner=pipeline_runner, asset_builder=build_working_assets,
                    telemetry_interval_seconds=60,
                ), 0)
            finally:
                os.chdir(previous)
            run = json.loads((campaign / "completed_attempts/attempt_0000/run/run.json").read_text(encoding="utf-8"))
            self.assertEqual(run["repository_head"], manifest["source_head"])
            self.assertEqual(run["source_provenance"]["source_head"], manifest["source_head"])
            self.assertEqual(run["source_provenance"]["kind"], "bundle_manifest")


if __name__ == "__main__":
    unittest.main()
