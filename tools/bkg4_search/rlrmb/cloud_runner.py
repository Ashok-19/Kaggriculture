"""Committed-generation cloud scaling over the authoritative RLRMB pipeline."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import struct
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from .pipeline import EXIT_INFRASTRUCTURE, run_production_pipeline
from .schema import stable_identity

CLOUD_STATE_VERSION = 1


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_cloud_config(path: Path) -> dict[str, Any]:
    document = _read(path)
    if document.get("schema_version") != CLOUD_STATE_VERSION:
        raise ValueError("unsupported Phase-16 cloud config version")
    base_ref = document.get("base_config")
    if not base_ref:
        return document
    root = Path(__file__).resolve().parents[3]
    base = _read(root / str(base_ref))
    cloud = dict(document.get("cloud") or {})
    order = tuple(cloud.pop("frontier_order", ()))
    if order:
        by_id = {row["id"]: row for row in base["families"]}
        if set(order) != set(by_id):
            raise ValueError("Phase-16 frontier order must name every configured family exactly once")
        base["families"] = [by_id[name] for name in order]
    base["cloud"] = cloud
    return base


def _write_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def initial_state(config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": CLOUD_STATE_VERSION,
        "config_identity": stable_identity("RLRMB-P16-CONFIG-V1", config),
        "generation": 0,
        "next_frontier": 0,
        "cuda_games_consumed": 0,
        "working_asset_dir": str(config["asset_dir"]),
        "working_parent_route": str(config["parent_route"]),
        "accepted_specialists": [],
        "completed_frontiers": [],
        "commits": [],
    }


def load_state(output: Path, config: Mapping[str, Any], resume: bool) -> dict[str, Any]:
    path = output / "generation_state.json"
    if not path.exists():
        if resume:
            raise ValueError("resume requested without a committed generation state")
        return initial_state(config)
    if not resume:
        raise ValueError("output already contains a committed cloud campaign")
    state = _read(path)
    if state.get("schema_version") != CLOUD_STATE_VERSION:
        raise ValueError("unsupported cloud checkpoint version")
    if state.get("config_identity") != stable_identity("RLRMB-P16-CONFIG-V1", config):
        raise ValueError("cloud checkpoint config identity mismatch")
    return state


def _mode_policy(config: Mapping[str, Any], mode: str) -> Mapping[str, Any]:
    cloud = config.get("cloud") or {}
    policy = (cloud.get("modes") or {}).get(mode)
    if not isinstance(policy, Mapping):
        raise ValueError(f"cloud mode is not configured: {mode}")
    return policy


def generation_config(config: Mapping[str, Any], state: Mapping[str, Any], mode: str) -> dict[str, Any]:
    policy = _mode_policy(config, mode)
    families = tuple(config["families"])
    index = int(state["next_frontier"]) % len(families)
    selected = dict(families[index])
    allocation = int(policy["cuda_games_per_generation"])
    categories = dict(policy["per_generation_categories"])
    if sum(categories.values()) > allocation:
        raise ValueError("generation CUDA category caps exceed generation allocation")
    selected["cuda_games"] = allocation
    runtime = {key: value for key, value in config.items() if key != "cloud"}
    runtime["families"] = [selected]
    runtime["budget"] = {"total_cuda_games": allocation, "per_family_categories": categories}
    runtime["asset_dir"] = state["working_asset_dir"]
    runtime["parent_route"] = state["working_parent_route"]
    runtime["search"] = dict(runtime["search"])
    runtime["search"]["checkpoint_batch_size"] = int(policy["checkpoint_batch_size"])
    runtime["deployment"] = dict(runtime["deployment"])
    runtime["deployment"]["route_id_base"] = int(config["cloud"]["route_id_base"]) + int(state["generation"])
    runtime["deployment"]["authority_name"] = str(config["cloud"]["authority_name"])
    runtime["deployment"]["authority_slot"] = str(config["cloud"]["authority_slot"])
    return runtime


def _load_bkg4() -> Any:
    root = Path(__file__).resolve().parents[3]
    path = root / "bonsai-kg" / "python"
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
    import bonsai_kg_v4
    return bonsai_kg_v4


def _extend_route_pack(base: Path, route_json: Path, route_id: int, destination: Path) -> None:
    header = struct.Struct("<8sHHI32s")
    descriptor = struct.Struct("<HHI")
    raw = base.read_bytes()
    magic, version, route_count, action_count, _ = header.unpack_from(raw, 0)
    descriptors = [descriptor.unpack_from(raw, header.size + i * descriptor.size) for i in range(route_count)]
    if route_id in {row[0] for row in descriptors}:
        raise ValueError("working route pack route-id collision")
    records = raw[header.size + route_count * descriptor.size:]
    module_root = Path(__file__).resolve().parents[3]
    stateful = module_root / "bonsai-kg" / "integrations" / "stateful_v6"
    if str(stateful) not in sys.path:
        sys.path.insert(0, str(stateful))
    from build_assets import encode_action
    actions = _read(route_json)["actions"]
    new_records = b"".join(encode_action(dict(action)) for action in actions)
    descriptors.append((route_id, len(actions), action_count))
    payload = b"".join(descriptor.pack(*row) for row in descriptors) + records + new_records
    digest = hashlib.sha256(payload).digest()
    destination.write_bytes(header.pack(magic, version, len(descriptors), action_count + len(actions), digest) + payload)


def build_working_assets(parent_assets: Path, deployment: Path, destination: Path) -> dict[str, Any]:
    """Compile one fully composed module and extend the current route pack."""
    destination.mkdir(parents=True, exist_ok=False)
    module = _read(deployment / "composed_module.json")
    bkg4 = _load_bkg4()
    # Composer metadata is needed by the next composition generation, but is
    # deliberately outside the BKG4 compiler schema.  Compile only executable
    # policy fields while retaining the complete module as controller.json.
    executable_fields = {
        "schema_version", "name", "feature_registry_version", "enums", "blackboard",
        "subtrees", "tree", "provenance",
    }
    executable = {key: value for key, value in module.items() if key in executable_fields}
    compiled = bkg4.compile_policy(executable)
    bkg4.save_compiled(compiled, destination / "controller.bkgc")
    (destination / "controller.json").write_text(json.dumps(module, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    specialist = _read(deployment / "specialist.json")
    route_id = int(specialist["route_id"])
    route_json = next(deployment.glob("*.route.json"))
    _extend_route_pack(parent_assets / "routes.bkv6", route_json, route_id, destination / "routes.bkv6")
    shutil.copy2(parent_assets / "legacy_routes.bkv6", destination / "legacy_routes.bkv6")
    identity = {"schema_version": 1, "controller_semantic_hash": compiled.semantic_hash,
                "route_id": route_id, "candidate_id": specialist["provenance"]["candidate_semantic_id"],
                "nodes": len(compiled.nodes), "state_slots": compiled.state_count}
    _write_atomic(destination / "identity.json", identity)
    return identity


def _accepted_deployment(run_dir: Path, candidate_id: str) -> Path:
    matches = sorted(run_dir.glob(f"families/*/{candidate_id}/deployment"))
    if len(matches) != 1:
        raise ValueError(f"accepted candidate deployment is missing or ambiguous: {candidate_id}")
    return matches[0]


def compact_generation_artifacts(run_dir: Path, accepted: Sequence[str]) -> None:
    """Discard bulk search matrices while preserving exact accepted evidence."""
    keep = set(accepted)
    families = run_dir / "families"
    if not families.exists():
        return
    for family in families.iterdir():
        if not family.is_dir():
            continue
        for child in family.iterdir():
            if child.is_dir() and (child.name == "prepromotion" or child.name not in keep):
                shutil.rmtree(child)


def run_cloud(config_path: Path, output: Path, *, mode: str, resume: bool = False,
              max_generations: int | None = None, pipeline_runner: Any = run_production_pipeline,
              asset_builder: Any = build_working_assets) -> int:
    config = load_cloud_config(config_path)
    policy = _mode_policy(config, mode)
    output.mkdir(parents=True, exist_ok=True)
    state = load_state(output, config, resume)
    limit = int(max_generations or policy["max_generations"])
    started = time.monotonic()
    wall_limit = int(policy["wall_time_limit_seconds"])
    for _ in range(limit):
        if time.monotonic() - started >= wall_limit:
            break
        attempt = len(state["completed_frontiers"])
        pending = output / f"attempt_{attempt:04d}.pending"
        if pending.exists():
            shutil.rmtree(pending)
        runtime_config = generation_config(config, state, mode)
        pending.mkdir(parents=True)
        runtime_path = pending / "config.json"
        _write_atomic(runtime_path, runtime_config)
        run_dir = pending / "run"
        code = pipeline_runner(runtime_path, run_dir)
        if code == EXIT_INFRASTRUCTURE:
            return EXIT_INFRASTRUCTURE
        metrics = _read(run_dir / "metrics.json")
        state["cuda_games_consumed"] += int(metrics.get("total_cuda_games", 0))
        family_id = runtime_config["families"][0]["id"]
        accepted = tuple(sorted(metrics.get("accepted_candidate_ids", ())))
        state["completed_frontiers"].append(family_id)
        state["next_frontier"] += 1
        if accepted:
            candidate_id = accepted[0]
            deployment = _accepted_deployment(run_dir, candidate_id)
            next_generation = int(state["generation"]) + 1
            assets = pending / "working_assets"
            identity = asset_builder(Path(state["working_asset_dir"]), deployment, assets)
            if int(metrics.get("protected_win_to_loss", 0)) or int(metrics.get("semantic_mismatches", 0)):
                raise RuntimeError("accepted generation failed protected regression/parity")
            route = next(deployment.glob("*.route.json"))
            compact_generation_artifacts(run_dir, accepted)
            committed = output / "generations" / f"generation_{next_generation:04d}"
            committed.parent.mkdir(parents=True, exist_ok=True)
            os.replace(pending, committed)
            state["generation"] = next_generation
            state["working_asset_dir"] = str((committed / "working_assets").resolve())
            route_copy = committed / "working_parent.route.json"
            shutil.copy2(committed / "run" / deployment.relative_to(run_dir) / route.name, route_copy)
            state["working_parent_route"] = str(route_copy.resolve())
            state["accepted_specialists"].append(candidate_id)
            state["commits"].append({"generation": next_generation, "candidate_id": candidate_id,
                                     "family_id": family_id, "identity": identity,
                                     "cuda_games": int(metrics.get("total_cuda_games", 0))})
        else:
            compact_generation_artifacts(run_dir, ())
            completed = output / "completed_attempts" / f"attempt_{attempt:04d}"
            completed.parent.mkdir(parents=True, exist_ok=True)
            os.replace(pending, completed)
        _write_atomic(output / "generation_state.json", state)
    result = {"schema_version": 1, "mode": mode, "wall_seconds": time.monotonic() - started,
              "terminal_verdict": "PROMOTION_READY" if state["accepted_specialists"] else "NOT_PROMOTION_READY",
              **state}
    _write_atomic(output / "cloud_result.json", result)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--mode", choices=("pilot", "campaign"), required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-generations", type=int)
    args = parser.parse_args(argv)
    try:
        return run_cloud(args.config, args.output, mode=args.mode, resume=args.resume,
                         max_generations=args.max_generations)
    except Exception as exc:
        print(f"RLRMB_CLOUD_INFRASTRUCTURE_FAILURE: {exc}", file=sys.stderr)
        return EXIT_INFRASTRUCTURE


if __name__ == "__main__":
    raise SystemExit(main())
