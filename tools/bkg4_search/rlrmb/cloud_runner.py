"""Committed-generation cloud scaling over the authoritative RLRMB pipeline."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import struct
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from tools.bkg4_search.bonsai_composer import SpecialistDeclaration, StrategicAuthority, compose_specialists
from tools.bkg4_search.bounded_macro import BoundedRouteMacro
from tools.bkg4_search.module_codec import ConditionGene

from .ir import MacroProgram
from .pipeline import EXIT_INFRASTRUCTURE, run_production_pipeline
from .run_resources import RunResources
from .cloud.telemetry import TelemetrySampler
from .schema import stable_identity
from .search_memory import (
    evaluation_context_id as _evaluation_context_id,
    historical_region_id as _historical_region_id,
)
from .search_allocator import SearchAllocator

CLOUD_STATE_VERSION = 1
_SEARCH_CONTEXT_SEARCH_KEYS = (
    "checkpoint_limit", "donor_limit", "motif_steps", "beam_depth",
    "max_candidates_per_expansion", "stagnation_rounds", "max_phases",
    "archive_elites_per_cell", "checkpoint_batch_size",
)


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _controller_identity(working_asset_dir: str) -> str:
    """Resolve the BKG4 controller semantic hash from the committed identity file."""
    path = Path(working_asset_dir) / "identity.json"
    if path.exists():
        doc = json.loads(path.read_text(encoding="utf-8"))
        return str(doc.get("controller_semantic_hash", ""))
    return stable_identity("RLRMB-P16-INITIAL-ASSETS-V1", working_asset_dir)


def search_context_identity(runtime_config: Mapping[str, Any], state: Mapping[str, Any]) -> str:
    """Exact deterministic identity for one (generation, family, search-config) triple.

    This hash proves whether two cloud attempts are semantically identical.
    It binds controller identity, generation, family, panels, search config, and
    donor/checkpoint sources. It intentionally excludes attempt number, output
    path, timestamp, SEALED panel, and provenance-only labels.
    """
    family = runtime_config["families"][0]
    search = runtime_config.get("search", {})
    context = {
        "controller": _controller_identity(state["working_asset_dir"]),
        "generation": int(state["generation"]),
        "family_id": family["id"],
        "target_panel": family.get("target_panel"),
        "checkpoint_evidence": family.get("checkpoint_evidence"),
        "transfer_panel": runtime_config.get("transfer_panel"),
        "validation_panel": runtime_config.get("validation_panel"),
        "donor_manifest": runtime_config.get("donor_manifest"),
        "search": {k: search.get(k) for k in _SEARCH_CONTEXT_SEARCH_KEYS},
        "promotion": runtime_config.get("promotion"),
        "repair": runtime_config.get("repair"),
    }
    return stable_identity("RLRMB-P16-SEARCH-CONTEXT-V1", context)


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


def _source_provenance(config: Mapping[str, Any]) -> dict[str, Any]:
    """Resolve auditable source identity from git or an extracted bundle manifest."""
    embedded = Path.cwd() / "phase16_bundle_manifest.json"
    if embedded.exists():
        document = _read(embedded)
        source_head = str(document.get("source_head") or "")
        expected_config = stable_identity("RLRMB-P16-CONFIG-V1", config)
        if len(source_head) != 40 or any(c not in "0123456789abcdef" for c in source_head):
            raise ValueError("bundle manifest source_head is invalid")
        if document.get("production_config_identity") != expected_config:
            raise ValueError("bundle manifest production config identity mismatch")
        return {
            "kind": "bundle_manifest",
            "source_head": source_head,
            "production_config_identity": expected_config,
        }
    result = subprocess.run(("git", "rev-parse", "HEAD"), cwd=Path.cwd(), check=False,
                            capture_output=True, text=True)
    source_head = result.stdout.strip() if result.returncode == 0 else ""
    if len(source_head) != 40 or any(c not in "0123456789abcdef" for c in source_head):
        raise ValueError("source HEAD is unavailable without git or bundle provenance")
    return {"kind": "git", "source_head": source_head}


def _recover_next_route_id(config: Mapping[str, Any], state: Mapping[str, Any]) -> int:
    """Recover the next authoritative route id from committed generation metadata only."""
    base = int(config["cloud"]["route_id_base"])
    used: list[int] = []
    for commit in state.get("commits", ()):
        used.extend(int(value) for value in commit.get("route_ids", ()))
        identity = commit.get("identity") or {}
        if "route_id" in identity:
            used.append(int(identity["route_id"]))
    return max(base, (max(used) + 1) if used else base)


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
        "uncommitted_accepts": [],
        "completed_frontiers": [],
        "commits": [],
        "completed_search_contexts": {},
        "suppressed_contexts": 0,
        "avoided_cuda_games": 0,
        "next_route_id": int(config["cloud"]["route_id_base"]),
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
    state.setdefault("completed_search_contexts", {})
    state.setdefault("suppressed_contexts", 0)
    state.setdefault("avoided_cuda_games", 0)
    state.setdefault("uncommitted_accepts", [])
    state.setdefault("next_route_id", _recover_next_route_id(config, state))
    return state


def _mode_policy(config: Mapping[str, Any], mode: str) -> Mapping[str, Any]:
    cloud = config.get("cloud") or {}
    policy = (cloud.get("modes") or {}).get(mode)
    if not isinstance(policy, Mapping):
        raise ValueError(f"cloud mode is not configured: {mode}")
    return policy


def _validated_operational_overrides(overrides: Mapping[str, Any] | None) -> dict[str, int]:
    values = dict(overrides or {})
    allowed = {"host_workers", "queue_depth", "checkpoint_batch_size"}
    unknown = set(values) - allowed
    if unknown:
        raise ValueError(f"unsupported cloud operational overrides: {sorted(unknown)}")
    limits = {"host_workers": 64, "queue_depth": 64, "checkpoint_batch_size": 64}
    for name, maximum in limits.items():
        if name not in values:
            continue
        value = values[name]
        if type(value) is not int or not 1 <= value <= maximum:
            raise ValueError(f"{name} must be an integer in 1..{maximum}")
    return {name: int(values[name]) for name in sorted(values)}


def generation_config(config: Mapping[str, Any], state: Mapping[str, Any], mode: str,
                      operational_overrides: Mapping[str, Any] | None = None,
                      family_override: "Mapping[str, Any] | None" = None,
                      source_provenance: Mapping[str, Any] | None = None) -> dict[str, Any]:
    policy = _mode_policy(config, mode)
    overrides = _validated_operational_overrides(operational_overrides)
    if family_override is not None:
        selected = dict(family_override)
    else:
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
    runtime["search"]["checkpoint_batch_size"] = int(
        overrides.get("checkpoint_batch_size", policy["checkpoint_batch_size"])
    )
    runtime["host_execution"] = dict(config["cloud"].get("host_execution") or {})
    if "host_workers" in overrides:
        runtime["host_execution"]["producer_workers"] = overrides["host_workers"]
    if "queue_depth" in overrides:
        runtime["host_execution"]["queue_depth"] = overrides["queue_depth"]
    runtime["deployment"] = dict(runtime["deployment"])
    runtime["deployment"]["route_id_base"] = int(
        state.get("next_route_id", _recover_next_route_id(config, state))
    )
    runtime["deployment"]["authority_name"] = str(config["cloud"]["authority_name"])
    runtime["deployment"]["authority_slot"] = str(config["cloud"]["authority_slot"])
    if source_provenance is not None:
        runtime["source_provenance"] = dict(source_provenance)
    return runtime


def _family_search_ctx(
    config: Mapping[str, Any],
    state: Mapping[str, Any],
    family: Mapping[str, Any],
    policy: Mapping[str, Any],
    overrides: Mapping[str, Any],
) -> str:
    """Compute search_context_identity for one family without a full runtime_config."""
    search = dict(config.get("search", {}))
    search["checkpoint_batch_size"] = int(
        overrides.get("checkpoint_batch_size", policy["checkpoint_batch_size"])
    )
    mini: dict[str, Any] = {
        "families": [family],
        "transfer_panel": config.get("transfer_panel"),
        "validation_panel": config.get("validation_panel"),
        "donor_manifest": config.get("donor_manifest"),
        "search": search,
        "promotion": config.get("promotion"),
        "repair": config.get("repair"),
    }
    return search_context_identity(mini, state)


def _family_eval_ctx(
    config: Mapping[str, Any],
    state: Mapping[str, Any],
    family: Mapping[str, Any],
) -> str:
    """Compute evaluation_context_id for one family (used by allocator scoring)."""
    mini: dict[str, Any] = {
        "asset_dir": state["working_asset_dir"],
        "families": [family],
        "transfer_panel": config.get("transfer_panel"),
        "validation_panel": config.get("validation_panel"),
        "search": config.get("search", {}),
    }
    return _evaluation_context_id(mini)


def _family_region_id(
    config: Mapping[str, Any], family: Mapping[str, Any]
) -> str:
    """Compute controller-independent historical allocator region identity."""
    mini: dict[str, Any] = {
        "families": [family],
        "donor_manifest": config.get("donor_manifest"),
        "transfer_panel": config.get("transfer_panel"),
        "validation_panel": config.get("validation_panel"),
        "search": config.get("search", {}),
    }
    return _historical_region_id(mini)


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
    """Compile one atomic generation deployment and extend its complete route set."""
    destination.mkdir(parents=True, exist_ok=False)
    module = _read(deployment / "composed_module.json")
    bkg4 = _load_bkg4()
    # Composer metadata is needed by the next composition generation, but is
    # deliberately outside the BKG4 compiler schema. Compile only executable
    # policy fields while retaining the complete module as controller.json.
    executable_fields = {
        "schema_version", "name", "feature_registry_version", "enums", "blackboard",
        "subtrees", "tree", "provenance",
    }
    executable = {key: value for key, value in module.items() if key in executable_fields}
    compiled = bkg4.compile_policy(executable)
    bkg4.save_compiled(compiled, destination / "controller.bkgc")
    (destination / "controller.json").write_text(json.dumps(module, sort_keys=True, indent=2) + "\n", encoding="utf-8")

    manifest_path = deployment / "specialists.json"
    if manifest_path.exists():
        rows = list(_read(manifest_path).get("specialists", ()))
        if not rows:
            raise ValueError("generation deployment has no specialists")
    else:
        # Backward-compatible single-specialist deployment used by bounded tests
        # and old local artifacts. New cloud commits always emit specialists.json.
        specialist = _read(deployment / "specialist.json")
        route_json = next(deployment.glob("*.route.json"))
        rows = [{
            "candidate_id": specialist["provenance"]["candidate_semantic_id"],
            "route_id": int(specialist["route_id"]),
            "route_path": route_json.name,
        }]

    route_ids = [int(row["route_id"]) for row in rows]
    candidate_ids = [str(row["candidate_id"]) for row in rows]
    if len(set(route_ids)) != len(route_ids):
        raise ValueError("generation deployment contains duplicate route ids")
    current_pack = parent_assets / "routes.bkv6"
    temporary_packs: list[Path] = []
    try:
        for index, row in enumerate(rows):
            route_json = deployment / str(row["route_path"])
            if not route_json.exists():
                raise ValueError(f"generation deployment route is missing: {row['route_path']}")
            destination_pack = (
                destination / "routes.bkv6"
                if index == len(rows) - 1
                else destination / f".routes_{index:04d}.tmp"
            )
            _extend_route_pack(current_pack, route_json, int(row["route_id"]), destination_pack)
            if current_pack.parent == destination and current_pack.name.startswith(".routes_"):
                current_pack.unlink(missing_ok=True)
            current_pack = destination_pack
            if destination_pack.name.startswith(".routes_"):
                temporary_packs.append(destination_pack)
    finally:
        for path in temporary_packs:
            if path != current_pack:
                path.unlink(missing_ok=True)
    shutil.copy2(parent_assets / "legacy_routes.bkv6", destination / "legacy_routes.bkv6")
    identity: dict[str, Any] = {
        "schema_version": 1,
        "controller_semantic_hash": compiled.semantic_hash,
        "route_ids": route_ids,
        "candidate_ids": candidate_ids,
        "nodes": len(compiled.nodes),
        "state_slots": compiled.state_count,
    }
    if len(rows) == 1:
        identity["route_id"] = route_ids[0]
        identity["candidate_id"] = candidate_ids[0]
    _write_atomic(destination / "identity.json", identity)
    return identity


def _accepted_deployment(run_dir: Path, candidate_id: str) -> Path:
    matches = sorted(run_dir.glob(f"families/*/{candidate_id}/deployment"))
    if len(matches) != 1:
        raise ValueError(f"accepted candidate deployment is missing or ambiguous: {candidate_id}")
    return matches[0]


class AcceptedCompositionConflict(ValueError):
    """All candidates passed acceptance independently but cannot coexist atomically."""


def _resolve_composition_conflict(
    declarations: Sequence[SpecialistDeclaration],
    evidence: Mapping[str, Mapping[str, Any]] | None = None,
) -> list[int]:
    """Select a maximal compatible subset when specialists have authority conflicts.

    Returns indices into *declarations* of a compatible subset, or empty if
    no subset of size >= 1 exists (should not happen for valid inputs).

    When *evidence* is provided (keyed by candidate_id), ranking uses verified
    capability: target L→W gained, then routed protected W→L (lower is better),
    then transfer support breadth.  Falls back to (active_start, name) when
    evidence is missing for a candidate.
    """
    n = len(declarations)
    if n <= 1:
        return list(range(n))
    conflicts: set[frozenset[int]] = set()
    for i in range(n):
        for j in range(i + 1, n):
            left, right = declarations[i], declarations[j]
            window_overlap = max(left.active_start, right.active_start) <= min(left.active_end, right.active_end)
            shared_auth = bool(set(left.authorities) & set(right.authorities))
            if window_overlap and shared_auth:
                conflicts.add(frozenset((i, j)))
    if not conflicts:
        return list(range(n))

    def _rank_key(i: int) -> tuple:
        d = declarations[i]
        cid = (d.provenance or {}).get("candidate_semantic_id", "")
        ev = (evidence or {}).get(cid, {})
        target_lw = int(ev.get("target_loss_to_win", 0))
        protected_wl = int(ev.get("routed_protected_win_to_loss", ev.get("protected_win_to_loss", 0)))
        transfer = int(ev.get("transfer_support", 0))
        # Protected regression is a hard penalty (sorted first).
        # Then higher target_lw is better (negate for ascending sort).
        # Higher transfer is better (negate).
        # Fallback: earlier active_start, then name for determinism.
        has_regression = 1 if protected_wl > 0 else 0
        return (has_regression, -target_lw, -transfer, d.active_start, d.name)

    ranked = sorted(range(n), key=_rank_key)
    selected: list[int] = []
    for i in ranked:
        compatible = all(frozenset((i, j)) not in conflicts for j in selected)
        if compatible:
            selected.append(i)
    return sorted(selected)


def _condition_rows(document: Sequence[Sequence[Mapping[str, Any]]]) -> tuple[tuple[ConditionGene, ...], ...]:
    return tuple(tuple(
        ConditionGene(str(row["feature"]), str(row["cmp"]), int(row.get("index", 0)), int(row["value"]))
        for row in clause
    ) for clause in document)


def _materialization_hash(document: Mapping[str, Any]) -> str:
    raw = json.dumps(document, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(b"RLRMB-BKG4-MATERIALIZATION-V1\0" + raw).hexdigest()


def _refresh_materialization_audits(run_dir: Path, audits: Mapping[str, Mapping[str, Any]]) -> None:
    """Keep root/family metrics aligned with authoritative rematerialization IDs."""
    if not audits:
        return

    def replace(value: Any) -> None:
        if isinstance(value, dict):
            candidate_id = value.get("candidate_id")
            if candidate_id in audits and "materialization" in value:
                value["materialization"] = dict(audits[str(candidate_id)])
            for nested in value.values():
                replace(nested)
        elif isinstance(value, list):
            for nested in value:
                replace(nested)

    paths = [run_dir / "metrics.json", *sorted((run_dir / "families").glob("*/run.json"))]
    for path in paths:
        if not path.exists():
            continue
        document = _read(path)
        replace(document)
        _write_atomic(path, document)


def _gather_candidate_evidence(run_dir: Path, candidate_ids: Sequence[str]) -> dict[str, dict[str, Any]]:
    """Extract per-candidate strength evidence from pipeline metrics."""
    evidence: dict[str, dict[str, Any]] = {}
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        return evidence
    metrics = _read(metrics_path)
    for family_data in (metrics.get("families") or {}).values():
        for result in family_data.get("results", ()):
            cid = str(result.get("candidate_id", ""))
            if cid not in candidate_ids:
                continue
            cegis = result.get("cegis") or {}
            ev: dict[str, Any] = {
                "target_loss_to_win": int(result.get("target_loss_to_win", 0)),
                "protected_win_to_loss": int(result.get("protected_win_to_loss", 0)),
                "transfer_support": int(result.get("transfer_support", 0)),
            }
            routed = result.get("routed_evaluation") or {}
            if routed:
                ev["routed_protected_win_to_loss"] = int(routed.get("protected_win_to_loss", 0))
                ev["routed_target_loss_to_win"] = int(routed.get("target_loss_to_win", 0))
            evidence[cid] = ev
    return evidence


def _prepare_generation_deployment(
    run_dir: Path,
    parent_assets: Path,
    parent_route: Path,
    accepted: Sequence[str],
    *,
    route_id_base: int,
    next_generation: int,
    authority_name: str,
    authority_slot: str,
) -> tuple[Path, list[dict[str, Any]]]:
    """Rebind accepted candidates to authoritative IDs and compose them atomically."""
    candidate_ids = tuple(sorted(accepted))
    if not candidate_ids:
        raise ValueError("generation deployment requires accepted candidates")
    deployments = [_accepted_deployment(run_dir, candidate_id) for candidate_id in candidate_ids]
    # Existing stub tests intentionally provide only the legacy three-file shape.
    # Keep that compatibility for one candidate; production materialization always
    # includes materialization.json and therefore takes the authoritative path.
    if len(candidate_ids) == 1 and not (deployments[0] / "materialization.json").exists():
        specialist = _read(deployments[0] / "specialist.json") if (deployments[0] / "specialist.json").exists() else {}
        route_id = int(specialist.get("route_id", route_id_base))
        return deployments[0], [{"candidate_id": candidate_ids[0], "route_id": route_id}]

    generation = run_dir / "generation_deployment"
    if generation.exists():
        shutil.rmtree(generation)
    routes_dir = generation / "routes"
    routes_dir.mkdir(parents=True)
    parent_module = _read(parent_assets / "controller.json")
    declarations: list[SpecialistDeclaration] = []
    rows: list[dict[str, Any]] = []
    authoritative_audits: dict[str, Mapping[str, Any]] = {}
    evidence_rebinds: list[dict[str, Any]] = []
    for offset, (candidate_id, source) in enumerate(zip(candidate_ids, deployments)):
        materialization = _read(source / "materialization.json")
        specialist_doc = _read(source / "specialist.json")
        if str(materialization.get("candidate_id")) != candidate_id:
            raise ValueError("accepted materialization candidate identity mismatch")
        provenance = dict(specialist_doc.get("provenance") or {})
        if provenance.get("candidate_semantic_id") != candidate_id:
            raise ValueError("accepted specialist provenance candidate identity mismatch")
        program = MacroProgram.from_document(materialization["program"])
        gates = _condition_rows(materialization["outer_gate"])
        route_id = route_id_base + offset
        route_source = next(source.glob("*.route.json"))
        route_name = f"{candidate_id}.route.json"
        shutil.copy2(route_source, routes_dir / route_name)
        macro = BoundedRouteMacro(
            name=str(specialist_doc["name"]),
            route_id=route_id,
            base_route_id=64,
            start_step=program.initiation.step,
            duration=program.max_total_steps,
            gate_clauses=gates,
            provenance={"program_semantic_id": program.semantic_id},
        )
        declaration = SpecialistDeclaration.from_macro(
            macro,
            route_source=f"routes/{route_name}",
            authorities=(authority_name,),
            provenance={
                "candidate_semantic_id": candidate_id,
                "program_semantic_id": program.semantic_id,
            },
        )
        declaration.validate()
        declarations.append(declaration)
        evidence_declaration = SpecialistDeclaration.from_macro(
            macro,
            route_source=route_source.name,
            authorities=(authority_name,),
            provenance={
                "candidate_semantic_id": candidate_id,
                "program_semantic_id": program.semantic_id,
            },
        )
        evidence_declaration.validate()
        # The per-candidate deployment is retained as exact acceptance evidence.
        # Rebind its route-bearing metadata to the same authoritative ID that will
        # appear in the atomic generation controller, rather than leaving the
        # provisional materialization-order ID behind.
        materialization_module = dict(materialization["module"])
        materialization_module["tree"] = macro.specialist_tree(route_override_slot=authority_slot)
        materialization_module["strategic_routes"] = [{
            "name": declaration.name,
            "route_id": route_id,
            "source": route_source.name,
        }]
        materialization["module"] = materialization_module
        materialization_id = _materialization_hash(materialization)
        single = compose_specialists(
            parent_module,
            (evidence_declaration,),
            (),
            strategic_authorities=(StrategicAuthority(authority_name, authority_slot),),
            output_name=f"certified_{candidate_id[:12]}",
        )
        audit = {
            "materialization_id": materialization_id,
            "node_count": int(materialization["node_count"]),
            "state_slots": len(materialization_module.get("blackboard", ())),
            "composition_order": single.report["composition_order"],
            "unresolved_conflicts": single.report["unresolved_conflicts"],
            "semantic_mismatches": 0,
        }
        evidence_rebinds.append({
            "source": source,
            "materialization": materialization,
            "specialist": evidence_declaration.canonical(),
            "composition": audit,
            "composed_module": single.module,
        })
        authoritative_audits[candidate_id] = audit
        rows.append({
            "candidate_id": candidate_id,
            "route_id": route_id,
            "route_path": f"routes/{route_name}",
            "active_window": [declaration.active_start, declaration.active_end],
            "specialist": declaration.canonical(),
        })

    try:
        composed = compose_specialists(
            parent_module,
            tuple(declarations),
            (),
            strategic_authorities=(StrategicAuthority(authority_name, authority_slot),),
            output_name=f"rlrmb_generation_{next_generation:04d}",
        )
    except ValueError as exc:
        if len(declarations) > 1 and "unresolved strategic authority conflict" in str(exc):
            conflict_evidence = _gather_candidate_evidence(run_dir, candidate_ids)
            subset_indices = _resolve_composition_conflict(declarations, evidence=conflict_evidence)
            if not subset_indices:
                conflict = {
                    "schema_version": 1,
                    "status": "UNCOMMITTED_COMPOSITION_CONFLICT",
                    "candidate_ids": list(candidate_ids),
                    "provisional_route_ids": [row["route_id"] for row in rows],
                    "error": str(exc),
                }
                _write_atomic(run_dir / "composition_conflict.json", conflict)
                shutil.rmtree(generation, ignore_errors=True)
                raise AcceptedCompositionConflict(str(exc)) from exc
            declarations = [declarations[i] for i in subset_indices]
            rows = [rows[i] for i in subset_indices]
            evidence_rebinds = [evidence_rebinds[i] for i in subset_indices]
            authoritative_audits = {
                candidate_ids[i]: authoritative_audits[candidate_ids[i]]
                for i in subset_indices if candidate_ids[i] in authoritative_audits
            }
            candidate_ids = tuple(candidate_ids[i] for i in subset_indices)
            _write_atomic(run_dir / "composition_conflict.json", {
                "schema_version": 1,
                "status": "RESOLVED_COMPATIBLE_SUBSET",
                "original_candidate_ids": list(accepted),
                "selected_indices": list(subset_indices),
                "selected_candidate_ids": list(candidate_ids),
                "original_error": str(exc),
            })
            composed = compose_specialists(
                parent_module,
                tuple(declarations),
                (),
                strategic_authorities=(StrategicAuthority(authority_name, authority_slot),),
                output_name=f"rlrmb_generation_{next_generation:04d}",
            )
        else:
            raise

    _write_atomic(generation / "composed_module.json", composed.module)
    _write_atomic(generation / "specialists.json", {
        "schema_version": 1,
        "specialists": rows,
        "composition_order": list(composed.report.get("composition_order", ())),
        "unresolved_conflicts": list(composed.report.get("unresolved_conflicts", ())),
    })
    _write_atomic(generation / "composition.json", composed.report)

    # The route used for the next search parent must contain every accepted edit.
    parent_actions = list(_read(parent_route)["actions"])
    for row in rows:
        route_actions = _read(generation / row["route_path"])["actions"]
        start, end = (int(value) for value in row["active_window"])
        if end >= len(parent_actions) or end >= len(route_actions):
            raise ValueError("accepted specialist active window exceeds route length")
        parent_actions[start:end + 1] = route_actions[start:end + 1]
    # Only rewrite retained per-candidate evidence after the complete generation
    # deployment has proven composable. A composition failure must preserve the
    # original independently accepted evidence and its provisional route IDs.
    for rebind in evidence_rebinds:
        source = Path(rebind["source"])
        _write_atomic(source / "materialization.json", rebind["materialization"])
        _write_atomic(source / "specialist.json", rebind["specialist"])
        _write_atomic(source / "composition.json", rebind["composition"])
        _write_atomic(source / "composed_module.json", rebind["composed_module"])
    _refresh_materialization_audits(run_dir, authoritative_audits)

    _write_atomic(generation / "working_parent.route.json", {"actions": parent_actions})
    return generation, rows


def _refresh_run_manifest(run_dir: Path) -> None:
    """Make run.json describe only artifacts that still exist after compaction."""
    manifest_path = run_dir / "run.json"
    if not manifest_path.exists():
        return
    manifest = _read(manifest_path)
    if "artifact_references" not in manifest:
        return
    manifest["artifact_references"] = sorted(
        str(path.relative_to(run_dir))
        for path in run_dir.rglob("*.json")
        if path.name != "run.json"
    )
    _write_atomic(manifest_path, manifest)


def _assert_no_pending_references(root: Path) -> None:
    """Fail closed rather than commit metadata that names a transient pending path."""
    for path in root.rglob("*.json"):
        document = _read(path)
        stack = [document]
        while stack:
            value = stack.pop()
            if isinstance(value, Mapping):
                stack.extend(value.values())
            elif isinstance(value, (list, tuple)):
                stack.extend(value)
            elif isinstance(value, str) and ".pending" in value:
                raise ValueError(f"committed metadata contains transient pending path: {path}")


def compact_generation_artifacts(run_dir: Path, accepted: Sequence[str]) -> None:
    """Discard bulk search matrices and refresh manifests to retained evidence only."""
    keep = set(accepted)
    families = run_dir / "families"
    if families.exists():
        for family in families.iterdir():
            if not family.is_dir():
                continue
            for child in family.iterdir():
                if child.is_dir() and (child.name == "prepromotion" or child.name not in keep):
                    shutil.rmtree(child)
    _refresh_run_manifest(run_dir)


def _generate_selfplay_panel(
    runtime_config: dict[str, Any],
    pending: Path,
    state: Mapping[str, Any],
    output: Path,
) -> dict[str, Any]:
    """Generate fresh selfplay tapes and rewrite the family target_panel path."""
    family = dict(runtime_config["families"][0])
    selfplay_cfg = family.get("selfplay", {})
    num_seeds = int(selfplay_cfg.get("num_seeds", 20))
    generation = int(state.get("generation", 0))
    workers = int(selfplay_cfg.get("workers", 4))
    bank_path = selfplay_cfg.get("bank_dir")
    panel_tapes = selfplay_cfg.get("panel_tapes")
    if panel_tapes is not None and (type(panel_tapes) is not int or panel_tapes <= 0):
        raise ValueError("selfplay panel_tapes must be a positive integer")
    policy_rows = selfplay_cfg.get("policies")
    if policy_rows is not None:
        from .selfplay_generator import SelfplayPolicy
        policies = tuple(SelfplayPolicy(**dict(row)) for row in policy_rows)
        if not policies or len({policy.name for policy in policies}) != len(policies):
            raise ValueError("selfplay policies must be non-empty with unique names")
    else:
        policies = None

    if bank_path:
        from .selfplay_bank import SelfplayBank
        bank = SelfplayBank(
            Path(bank_path),
            asset_dir=runtime_config.get("asset_dir"),
        )
        if generation == 0:
            bank.generate(
                num_seeds=num_seeds, policies=policies, generation=generation, workers=workers,
            )
        else:
            refresh_seeds = max(4, num_seeds // 4)
            bank.refresh(
                num_new_seeds=refresh_seeds, policies=policies, generation=generation, workers=workers,
            )
        selfplay_dir = output / "selfplay" / f"generation_{generation:04d}"
        selfplay_dir.mkdir(parents=True, exist_ok=True)
        panel_path = selfplay_dir / "target_panel.tsv"
        bank.assemble_panel(panel_path, max_tapes=panel_tapes)
        manifest = bank.stats()
        manifest["generation"] = generation
        manifest["panel_tapes"] = panel_tapes or bank.size
        manifest["policy_mix"] = str(selfplay_cfg.get("policy_mix", "default"))
    else:
        from .selfplay_generator import generate_selfplay_family
        rng_seed = hash((generation, family["id"], num_seeds)) & 0x7FFFFFFF
        selfplay_dir = output / "selfplay" / f"generation_{generation:04d}"
        panel_path, manifest = generate_selfplay_family(
            selfplay_dir, num_seeds=num_seeds, policies=policies, rng_seed=rng_seed,
        )

    runtime_config = dict(runtime_config)
    root = Path.cwd()
    try:
        rel = panel_path.resolve().relative_to(root.resolve())
        family["target_panel"] = str(rel)
    except ValueError:
        family["target_panel"] = str(panel_path.resolve())
    runtime_config["families"] = [family]
    _write_atomic(selfplay_dir / "manifest.json", manifest)
    return runtime_config


def run_cloud(config_path: Path, output: Path, *, mode: str, resume: bool = False,
              max_generations: int | None = None, pipeline_runner: Any = run_production_pipeline,
              asset_builder: Any = build_working_assets,
              operational_overrides: Mapping[str, Any] | None = None,
              telemetry_interval_seconds: float = 2.0) -> int:
    config = load_cloud_config(config_path)
    source_provenance = _source_provenance(config)
    policy = _mode_policy(config, mode)
    overrides = _validated_operational_overrides(operational_overrides)
    if telemetry_interval_seconds <= 0 or telemetry_interval_seconds > 3600:
        raise ValueError("telemetry interval must be in (0, 3600] seconds")
    output.mkdir(parents=True, exist_ok=True)
    telemetry = TelemetrySampler(output / "telemetry.jsonl", interval_seconds=float(telemetry_interval_seconds))
    telemetry.start()
    # Phase 16.3: create persistent host resources at cloud-run scope.
    host_ex = dict((config.get("cloud") or {}).get("host_execution") or {})
    stable_cache_dir = output / ".capture_cache"
    _resources = RunResources.from_host_execution(host_ex, stable_cache_dir=stable_cache_dir)
    try:
        with _resources:
            # Phase 16.5: open durable semantic search memory at cloud-run scope.
            _resources.open_search_memory(output / "search_memory.db")
            # Phase 16.6: create adaptive allocator backed by the open memory.
            allocator = SearchAllocator(_resources.search_memory)
            return _run_cloud_inner(
                config, config_path, output, mode=mode, resume=resume,
                max_generations=max_generations, pipeline_runner=pipeline_runner,
                asset_builder=asset_builder, operational_overrides=operational_overrides,
                telemetry_interval_seconds=telemetry_interval_seconds,
                overrides=overrides, policy=policy, resources=_resources,
                allocator=allocator, source_provenance=source_provenance,
            )
    finally:
        telemetry.stop()


def _run_cloud_inner(
    config: Any, config_path: "Path", output: "Path", *, mode: str, resume: bool,
    max_generations: "int | None", pipeline_runner: Any, asset_builder: Any,
    operational_overrides: "Any", telemetry_interval_seconds: float,
    overrides: dict, policy: Any, resources: RunResources,
    allocator: "SearchAllocator | None" = None,
    source_provenance: "Mapping[str, Any] | None" = None,
) -> int:
    state = load_state(output, config, resume)
    limit = int(max_generations or policy["max_generations"])
    started = time.monotonic()
    wall_limit = int(policy["wall_time_limit_seconds"])
    if allocator is None:
        allocator = SearchAllocator(getattr(resources, "search_memory", None))
    # Phase 16.7 recovery hardening: region_allocation is the durable ledger for
    # completed pipeline attempts. Rehydrate process-local allocator telemetry
    # from it on every run/resume, and recover CUDA games that were executed and
    # durably recorded before a later composition/commit failure.
    memory = getattr(resources, "search_memory", None)
    if memory is not None and hasattr(memory, "get_region_allocation_stats"):
        durable_cuda_games = 0
        for family in config["families"]:
            region_id = _family_region_id(config, family)
            region_games = int(memory.get_region_allocation_stats(region_id).get("cuda_games", 0))
            if region_games:
                allocator.record_budget(str(family["id"]), region_games)
                durable_cuda_games += region_games
        if durable_cuda_games > int(state.get("cuda_games_consumed", 0)):
            state["cuda_games_consumed"] = durable_cuda_games
            _write_atomic(output / "generation_state.json", state)
    terminal_reason = "MAX_GENERATIONS"
    allocation_per_gen = int(policy["cuda_games_per_generation"])
    for _ in range(limit):
        if time.monotonic() - started >= wall_limit:
            terminal_reason = "WALL_TIME_LIMIT"
            break
        # Phase 16.6: build eligible candidates (not yet complete in this context).
        # Compute search_context_identity per family to detect exact suppressions.
        eligible: list[tuple[str, str, str, int, Any]] = []
        suppressed_count = 0
        for cfg_idx, family in enumerate(config["families"]):
            sc_id = _family_search_ctx(config, state, family, policy, overrides)
            if sc_id in state["completed_search_contexts"]:
                suppressed_count += 1
            else:
                ec_id = _family_eval_ctx(config, state, family)
                region_id = _family_region_id(config, family)
                eligible.append((family["id"], ec_id, region_id, cfg_idx, family))
        if not eligible:
            # All families are suppressed in the current context; record the exhaustion
            # event as suppressed_contexts increments (one per suppressed family, matching
            # the Phase 16.1 per-family-encounter accounting the tests verify).
            state["suppressed_contexts"] = state.get("suppressed_contexts", 0) + suppressed_count
            state["avoided_cuda_games"] = (
                state.get("avoided_cuda_games", 0) + suppressed_count * allocation_per_gen
            )
            _write_atomic(output / "generation_state.json", state)
            terminal_reason = "GENERATION_CONTEXT_EXHAUSTED"
            break
        # Adaptive selection via allocator
        alloc_candidates = [
            (fid, ectx, region_id, cidx)
            for fid, ectx, region_id, cidx, _ in eligible
        ]
        selected_row = allocator.select(alloc_candidates, set(state["completed_search_contexts"]))
        if selected_row is None:
            terminal_reason = "GENERATION_CONTEXT_EXHAUSTED"
            break
        best_fid, _best_ectx, best_cidx, _score = selected_row
        selected_match = next(
            row for row in eligible if row[0] == best_fid and row[3] == best_cidx
        )
        _fid, _ectx, best_region_id, _cidx, selected_family = selected_match
        attempt = len(state["completed_frontiers"])
        pending = output / f"attempt_{attempt:04d}.pending"
        if pending.exists():
            shutil.rmtree(pending)
        runtime_config = generation_config(
            config, state, mode, operational_overrides=overrides,
            family_override=selected_family, source_provenance=source_provenance,
        )
        family_id = runtime_config["families"][0]["id"]
        pending.mkdir(parents=True)
        # Generate fresh selfplay tapes when the family requests it.
        family_cfg = runtime_config["families"][0]
        if family_cfg.get("source") == "selfplay":
            runtime_config = _generate_selfplay_panel(
                runtime_config, pending, state, output,
            )
        context_id = search_context_identity(runtime_config, state)
        runtime_path = pending / "config.json"
        _write_atomic(runtime_path, runtime_config)
        run_dir = pending / "run"
        # Phase 16.3: pass persistent resources to the real pipeline runner.
        # Custom stub runners (tests) do not receive resources.
        resources.set_search_generation(int(state["generation"]))
        if pipeline_runner is run_production_pipeline:
            code = run_production_pipeline(runtime_path, run_dir, resources=resources)
        else:
            code = pipeline_runner(runtime_path, run_dir)
        if code == EXIT_INFRASTRUCTURE:
            return EXIT_INFRASTRUCTURE
        metrics = _read(run_dir / "metrics.json")
        actual_cuda_games = int(metrics.get("total_cuda_games", 0))
        state["cuda_games_consumed"] += actual_cuda_games
        allocator.record_budget(best_fid, actual_cuda_games)
        if resources.search_memory is not None:
            resources.search_memory.record_region_allocation(
                best_region_id,
                generation=int(state["generation"]),
                cuda_games=actual_cuda_games,
            )
        accepted = tuple(sorted(metrics.get("accepted_candidate_ids", ())))
        state["completed_search_contexts"][context_id] = {
            "family_id": family_id,
            "generation": int(state["generation"]),
            "cuda_games": int(metrics.get("total_cuda_games", 0)),
        }
        state["completed_frontiers"].append(family_id)
        state["next_frontier"] += 1
        if accepted:
            protected_w2l = int(metrics.get("protected_win_to_loss", 0))
            semantic_mismatches = int(metrics.get("semantic_mismatches", 0))
            if protected_w2l or semantic_mismatches:
                raise RuntimeError("accepted generation failed protected regression/parity")
            next_generation = int(state["generation"]) + 1
            committed = output / "generations" / f"generation_{next_generation:04d}"
            try:
                deployment, specialist_rows = _prepare_generation_deployment(
                    run_dir,
                    Path(state["working_asset_dir"]),
                    Path(state["working_parent_route"]),
                    accepted,
                    route_id_base=int(state["next_route_id"]),
                    next_generation=next_generation,
                    authority_name=str(runtime_config["deployment"]["authority_name"]),
                    authority_slot=str(runtime_config["deployment"]["authority_slot"]),
                )
            except AcceptedCompositionConflict as exc:
                compact_generation_artifacts(run_dir, accepted)
                completed = output / "completed_attempts" / f"attempt_{attempt:04d}"
                completed.parent.mkdir(parents=True, exist_ok=True)
                _assert_no_pending_references(pending)
                os.replace(pending, completed)
                state["uncommitted_accepts"].append({
                    "family_id": family_id,
                    "generation": int(state["generation"]),
                    "candidate_ids": list(accepted),
                    "reason": "COMPOSITION_CONFLICT",
                    "diagnostic": str(exc),
                    "attempt_path": str(completed.relative_to(output)),
                    "cuda_games": actual_cuda_games,
                })
                _write_atomic(output / "generation_state.json", state)
                continue

            assets = pending / "working_assets"
            identity = asset_builder(Path(state["working_asset_dir"]), deployment, assets)
            route_ids = [int(row["route_id"]) for row in specialist_rows]
            if len(set(route_ids)) != len(route_ids):
                raise RuntimeError("generation route allocation produced a collision")
            compact_generation_artifacts(run_dir, accepted)
            committed.parent.mkdir(parents=True, exist_ok=True)
            _assert_no_pending_references(pending)
            os.replace(pending, committed)
            # Phase 16.5/16.6: before changing the authoritative parent, copy
            # every outgoing-generation family evaluation context into its stable
            # cross-generation search region. These rows guide allocation only;
            # the new controller still has distinct exact evaluation contexts.
            if resources.search_memory is not None:
                for old_family in config["families"]:
                    old_eval_ctx = _family_eval_ctx(config, state, old_family)
                    old_region_id = _family_region_id(config, old_family)
                    resources.search_memory.promote_to_historical(
                        old_eval_ctx,
                        next_generation,
                        region_id=old_region_id,
                    )
            state["generation"] = next_generation
            state["working_asset_dir"] = str((committed / "working_assets").resolve())
            committed_deployment = committed / "run" / deployment.relative_to(run_dir)
            if (committed_deployment / "working_parent.route.json").exists():
                parent_route = committed_deployment / "working_parent.route.json"
            else:
                # Compatibility for one-candidate legacy stub deployments.
                route = next(committed_deployment.glob("*.route.json"))
                parent_route = committed / "working_parent.route.json"
                shutil.copy2(route, parent_route)
            state["working_parent_route"] = str(parent_route.resolve())
            state["accepted_specialists"].extend(accepted)
            if route_ids:
                state["next_route_id"] = max(int(state["next_route_id"]), max(route_ids) + 1)
            commit_row: dict[str, Any] = {
                "generation": next_generation,
                "candidate_ids": list(accepted),
                "route_ids": route_ids,
                "family_id": family_id,
                "identity": identity,
                "cuda_games": actual_cuda_games,
                "protected_win_to_loss": protected_w2l,
                "semantic_mismatches": semantic_mismatches,
            }
            if len(accepted) == 1:
                commit_row["candidate_id"] = accepted[0]
            state["commits"].append(commit_row)
        else:
            compact_generation_artifacts(run_dir, ())
            completed = output / "completed_attempts" / f"attempt_{attempt:04d}"
            completed.parent.mkdir(parents=True, exist_ok=True)
            _assert_no_pending_references(pending)
            os.replace(pending, completed)
        _write_atomic(output / "generation_state.json", state)
    result = {"schema_version": 1, "mode": mode, "wall_seconds": time.monotonic() - started,
              "terminal_verdict": "PROMOTION_READY" if state["accepted_specialists"] else "NOT_PROMOTION_READY",
              "terminal_reason": terminal_reason,
              "searches_considered": len(state["completed_frontiers"]) + state["suppressed_contexts"],
              "exact_repeat_contexts_suppressed": state["suppressed_contexts"],
              "avoided_cuda_games": state["avoided_cuda_games"],
              "operational_overrides": overrides,
              "telemetry_interval_seconds": float(telemetry_interval_seconds),
              "p163_resources": resources.metrics_document(),
              "p166_allocator": allocator.metrics_document(),
              **state}
    _write_atomic(output / "cloud_result.json", result)
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run committed-generation RLRMB cloud synthesis")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--mode", choices=("pilot", "campaign"), required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-generations", type=int)
    parser.add_argument("--host-workers", type=int,
                        help="runtime-only producer process count (1..64)")
    parser.add_argument("--queue-depth", type=int,
                        help="runtime-only bounded producer queue depth (1..64)")
    parser.add_argument("--checkpoint-batch-size", type=int,
                        help="runtime-only checkpoint batch size (1..64)")
    parser.add_argument("--telemetry-interval-seconds", type=float, default=2.0,
                        help="machine telemetry sampling interval; default 2 seconds")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    overrides = {
        name: value for name, value in (
            ("host_workers", args.host_workers),
            ("queue_depth", args.queue_depth),
            ("checkpoint_batch_size", args.checkpoint_batch_size),
        ) if value is not None
    }
    try:
        return run_cloud(args.config, args.output, mode=args.mode, resume=args.resume,
                         max_generations=args.max_generations,
                         operational_overrides=overrides,
                         telemetry_interval_seconds=args.telemetry_interval_seconds)
    except Exception as exc:
        print(f"RLRMB_CLOUD_INFRASTRUCTURE_FAILURE: {exc}", file=sys.stderr)
        return EXIT_INFRASTRUCTURE


if __name__ == "__main__":
    raise SystemExit(main())
