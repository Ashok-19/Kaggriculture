from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from tools.bkg4_search.module_codec import genome_from_json
from tools.edit_corpus.corpus import assert_model_payload_safe, canonical_json, derive_case_label, derive_edit_labels, stable_identity

from . import CORPUS_SCHEMA_VERSION, EDIT_PROGRAM_SCHEMA_VERSION, GRAMMAR_ID
from .archive import SearchArchive, behavior_descriptor, structural_descriptor
from .fitness import PairEvaluation
from .certification import CertificationClass, DiscoveryQuality, TransferFailure, discovery_quality, program_contains_temporal
from .panels import CaseBank
from .schema import EditProgram, ParentProfile


MODEL_FEATURE_COLUMNS = (
    "model_context_json",
    "context_probe_steps",
    "public_context_width",
    "public_context_registry_width",
    "public_context_values",
)
PROGRAM_MODEL_FEATURE_COLUMNS = (
    "parent_genome_json",
    "edit_program_json",
)

_MODEL_CONTEXT_KEYS = frozenset({"context_schema", "seat", "probe_steps", "frame_width", "registry_width"})


def archive_transaction_digest(archive: SearchArchive) -> str:
    """Digest post-evaluation search truth while ignoring commit-ledger bookkeeping."""
    raw = json.loads(json.dumps(archive.to_dict()))
    runtime = dict(raw.get("runtime_state") or {})
    runtime["exported_generations"] = []
    runtime.pop("generation_transactions", None)
    raw["runtime_state"] = runtime
    return hashlib.sha256(canonical_json(raw).encode()).hexdigest()


def generation_transaction_id(archive: SearchArchive, generation: int, program_ids: Sequence[str]) -> tuple[str, str]:
    digest = archive_transaction_digest(archive)
    tx = stable_identity(
        "BKG4-EDIT-GENERATION-TX-V1",
        archive.base_identity,
        archive.run_id,
        int(generation),
        digest,
        tuple(sorted(str(x) for x in program_ids)),
    )
    return tx, digest


def _f(name: str, typ: pa.DataType) -> pa.Field:
    return pa.field(name, typ, nullable=False)


BASE_SCHEMA = pa.schema([
    _f("base_id", pa.string()), _f("combined_agent_identity", pa.string()), _f("controller_semantic_hash", pa.string()),
    _f("bkg4_abi", pa.uint16()), _f("feature_registry", pa.uint16()), _f("route_ids", pa.list_(pa.uint16())),
    _f("controller_structure_json", pa.string()),
])
CASE_SCHEMA = pa.schema([
    _f("case_id", pa.string()), _f("seat", pa.int8()), _f("model_context_json", pa.string()), _f("baseline_margin", pa.int64()),
    _f("context_probe_steps", pa.list_(pa.int16())), _f("public_context_width", pa.int32()), _f("public_context_registry_width", pa.int32()),
    _f("public_context_values", pa.list_(pa.int64())), _f("protected", pa.bool_()), _f("prov_seed", pa.uint64()),
    _f("prov_family", pa.string()), _f("prov_role", pa.string()), _f("prov_tape_path", pa.string()), _f("prov_own_history", pa.bool_()),
])
PARENT_SCHEMA = pa.schema([
    _f("parent_id", pa.string()), _f("base_id", pa.string()), _f("overlay_semantic_id", pa.string()),
    _f("overlay_genome_json", pa.string()), _f("overlay_prefix_step", pa.uint16()), _f("overlay_rule_count", pa.uint8()),
])
PROGRAM_SCHEMA = pa.schema([
    _f("program_id", pa.string()), _f("base_id", pa.string()), _f("parent_id", pa.string()), _f("child_parent_id", pa.string()),
    _f("parent_overlay_semantic_id", pa.string()), _f("child_overlay_semantic_id", pa.string()),
    _f("parent_genome_json", pa.string()), _f("child_genome_json", pa.string()), _f("edit_program_json", pa.string()),
    _f("operations_json", pa.string()), _f("targets_json", pa.string()), _f("program_length", pa.uint16()), _f("generation_depth", pa.int32()),
    _f("parent_rule_count", pa.uint8()), _f("child_rule_count", pa.uint8()), _f("parent_prefix_step", pa.uint16()), _f("child_prefix_step", pa.uint16()),
    _f("sample_category", pa.string()), _f("discovery_quality", pa.string()), _f("certification_class", pa.string()), _f("transfer_failure", pa.string()),
    _f("certification_panel_count", pa.int16()), _f("cert_strict_safe_retention", pa.float64()), _f("cert_activation_retention", pa.float64()),
    _f("cert_behavior_retention", pa.float64()), _f("cert_safe_useful_retention", pa.float64()), _f("cert_elite_retention", pa.float64()), _f("cert_l2w_retention", pa.float64()),
    _f("cert_worst_protected_delta", pa.int64()), _f("cert_mean_margin_delta", pa.float64()), _f("temporal_edit", pa.bool_()),
    _f("strict_safe", pa.bool_()), _f("exact_noop", pa.bool_()), _f("behavior_changing", pa.bool_()),
    _f("l2w_count", pa.int32()), _f("w2l_count", pa.int32()), _f("new_loss_count", pa.int32()), _f("severe_regression_count", pa.int32()),
    _f("losses_before", pa.int32()), _f("losses_after", pa.int32()), _f("fragile_wins_before", pa.int32()), _f("fragile_wins_after", pa.int32()),
    _f("worst_candidate_margin", pa.int64()), _f("worst_protected_win_delta", pa.int64()), _f("mean_margin_delta", pa.float64()),
    _f("action_change_total", pa.int64()), _f("activation_total", pa.int64()), _f("label_count", pa.int32()), _f("source_label_count", pa.int32()), _f("root_label_count", pa.int32()),
    _f("root_strict_safe", pa.bool_()), _f("root_behavior_changing", pa.bool_()), _f("root_l2w_count", pa.int32()), _f("root_w2l_count", pa.int32()),
    _f("root_new_loss_count", pa.int32()), _f("root_severe_regression_count", pa.int32()), _f("root_worst_candidate_margin", pa.int64()),
    _f("root_worst_protected_win_delta", pa.int64()), _f("root_mean_margin_delta", pa.float64()),
    _f("exact_program_group", pa.string()), _f("parent_family_group", pa.string()), _f("parent_structural_group", pa.string()),
    _f("child_semantic_group", pa.string()), _f("action_skeleton_group", pa.string()), _f("operation_target_group", pa.string()),
    _f("lineage_group", pa.string()), _f("run_temporal_group", pa.string()), _f("target_holdout_group", pa.string()), _f("temporal_regime_group", pa.string()),
    _f("search_generation_group", pa.string()), _f("certification_panel_group", pa.string()), _f("search_emitter", pa.string()), _f("search_seed", pa.int64()),
    _f("search_run_id", pa.string()), _f("search_generation", pa.int32()), _f("search_structural_descriptor_json", pa.string()),
    _f("search_behavior_descriptor_json", pa.string()),
])
LABEL_SCHEMA = pa.schema([
    _f("baseline_margin", pa.int64()), _f("candidate_margin", pa.int64()), _f("margin_delta", pa.int64()),
    _f("baseline_outcome", pa.string()), _f("candidate_outcome", pa.string()), _f("l2w", pa.bool_()), _f("w2l", pa.bool_()), _f("new_loss", pa.bool_()),
    _f("protected", pa.bool_()), _f("action_changes", pa.uint16()), _f("activations", pa.uint16()),
    _f("parent_action_trace_hash", pa.uint64()), _f("candidate_action_trace_hash", pa.uint64()), _f("action_trace_equal", pa.bool_()),
    _f("behavior_changing", pa.bool_()), _f("case_id", pa.string()), _f("parent_action_changes_from_base", pa.uint16()),
    _f("candidate_action_changes_from_base", pa.uint16()), _f("parent_activations", pa.uint16()), _f("candidate_activations", pa.uint16()),
    _f("program_id", pa.string()), _f("edit_id", pa.string()), _f("parent_overlay_semantic_id", pa.string()),
    _f("child_overlay_semantic_id", pa.string()), _f("search_generation", pa.int32()), _f("label_role", pa.string()),
])


def _validate_model_context(payload: Any) -> None:
    if not isinstance(payload, dict):
        raise ValueError("model_context_json must encode an object")
    unknown = set(payload) - _MODEL_CONTEXT_KEYS
    if unknown:
        raise ValueError(f"unknown/non-pre-outcome model context keys: {sorted(unknown)}")
    if "context_schema" in payload and not isinstance(payload["context_schema"], str):
        raise ValueError("context_schema must be a string")
    if "seat" in payload and (type(payload["seat"]) is not int or payload["seat"] not in (0, 1)):
        raise ValueError("model context seat must be 0 or 1")
    for key in ("frame_width", "registry_width"):
        if key in payload and (type(payload[key]) is not int or int(payload[key]) < 0):
            raise ValueError(f"{key} must be a non-negative integer")
    if "probe_steps" in payload:
        steps = payload["probe_steps"]
        if not isinstance(steps, list) or any(type(x) is not int or not 0 <= x < 720 for x in steps):
            raise ValueError("probe_steps must be legal public step integers")


def _table(rows: list[dict[str, Any]], schema: pa.Schema) -> pa.Table:
    table = pa.Table.from_pylist(rows, schema=schema)
    if not table.schema.equals(schema, check_metadata=False):
        raise ValueError("Arrow schema drift")
    for field in schema:
        if not field.nullable and table[field.name].null_count:
            raise ValueError(f"non-null corpus field contains nulls: {field.name}")
    return table


def _write_rows(path: Path, rows: list[dict[str, Any]], schema: pa.Schema) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(_table(rows, schema), path, compression="zstd")


def _read_rows(path: Path) -> list[dict[str, Any]]:
    return pq.read_table(path).to_pylist()


def _hash_group(namespace: str, value: Any) -> str:
    return stable_identity(namespace, value)


def _action_skeleton(program: EditProgram) -> dict[str, Any]:
    atoms = []
    for atom in program.atoms:
        a = dict(atom.action)
        atoms.append({
            "operation": atom.operation,
            "target_kind": atom.target_kind,
            "target_rule_index": atom.target_rule_index,
            "target_condition_index": atom.target_condition_index,
            "feature": a.get("feature"),
            "cmp": a.get("cmp"),
            "index": int(a.get("index", 0)),
            "feature2": a.get("feature2"),
            "cmp2": a.get("cmp2"),
            "index2": int(a.get("index2", 0)),
            "condition_count": int(a.get("condition_count", 0)),
        })
    return {"atoms": atoms, "length": program.program_length}


def _operation_target(program: EditProgram) -> list[dict[str, Any]]:
    return [
        {
            "operation": atom.operation,
            "target_kind": atom.target_kind,
            "target_node": atom.target_node,
            "target_rule_index": atom.target_rule_index,
            "target_condition_index": atom.target_condition_index,
        }
        for atom in program.atoms
    ]


def _lineage_root(program: EditProgram, archive: SearchArchive) -> str:
    current = program
    seen: set[str] = set()
    while current.parent_program_id:
        if current.program_id in seen:
            raise ValueError("search lineage cycle detected")
        seen.add(current.program_id)
        parent = archive.programs.get(current.parent_program_id)
        if parent is None:
            break
        current = parent
    return current.program_id


def _quality_category(
    *,
    discovery: DiscoveryQuality,
    certification: CertificationClass,
    root_aggregate: Mapping[str, Any],
    emitter: str = "",
) -> str:
    if certification in {CertificationClass.CERTIFIED_ELITE_FLIP, CertificationClass.CERTIFIED_ELITE_MARGIN}:
        return "certified_elite"
    if certification == CertificationClass.CERTIFIED_SAFE_USEFUL:
        return "certified_safe_useful"
    if certification == CertificationClass.GENERALIZATION_HARD_NEGATIVE:
        return "generalization_hard_negative"
    if certification == CertificationClass.GENERALIZATION_BOUNDARY:
        return "generalization_boundary"
    if bool(root_aggregate["exact_noop"]):
        return "controlled_noop"
    if str(emitter) == "cegis_repair":
        return "near_feasible_repair"
    if discovery == DiscoveryQuality.DISCOVERY_BOUNDARY:
        return "near_feasible_repair"
    if str(emitter) == "certified_neighborhood":
        return "local_contrastive_neighbor"
    if discovery in {
        DiscoveryQuality.DISCOVERY_ELITE_FLIP,
        DiscoveryQuality.DISCOVERY_ELITE_MARGIN,
        DiscoveryQuality.DISCOVERY_SAFE_USEFUL,
    }:
        return "discovery_only_positive"
    if int(root_aggregate["new_loss_count"]) > 0 or int(root_aggregate["w2l_count"]) > 0 or int(root_aggregate["severe_regression_count"]) > 0 or float(root_aggregate["mean_margin_delta"]) < 0.0:
        return "hard_negative"
    if bool(root_aggregate["behavior_changing"]):
        return "structural_diversity"
    return "structural_diversity"


def program_sample_category(archive: SearchArchive, program_id: str, root_pair: PairEvaluation) -> str:
    program = archive.programs[str(program_id)]
    dq = discovery_quality(root_pair)
    vector = archive.certification_vector(str(program_id))
    cls = CertificationClass.UNCERTIFIED if vector is None else vector.certification_class
    return _quality_category(
        discovery=dq, certification=cls,
        root_aggregate=root_pair.certified_aggregate(), emitter=program.emitter,
    )


def select_high_information_program_ids(
    rows: Mapping[str, Mapping[str, Any]], *, max_noop_fraction: float = 0.10,
    max_easy_hard_negative_fraction: float = 0.30,
    max_structural_fraction: float = 0.15,
) -> tuple[str, ...]:
    """Select a deterministic high-information training slice from exact evaluated rows.

    Rare certified/generalization/repair/local-neighbor evidence is never discarded.
    Once any such premium evidence exists, easy catastrophic negatives and structural
    filler are fraction-capped instead of overwhelming the corpus. Negative-only
    diagnostic generations retain all non-noops so evidence is never silently erased.
    """
    noop_frac = float(max_noop_fraction)
    hard_frac = float(max_easy_hard_negative_fraction)
    structural_frac = float(max_structural_fraction)
    for name, value in (("max_noop_fraction", noop_frac), ("max_easy_hard_negative_fraction", hard_frac), ("max_structural_fraction", structural_frac)):
        if not 0.0 <= value < 1.0:
            raise ValueError(f"{name} must be in [0,1)")

    easy_categories = {"hard_negative", "structural_diversity", "controlled_noop"}
    premium = sorted(
        str(pid) for pid, row in rows.items()
        if str(row.get("sample_category")) not in easy_categories
    )
    hard = [
        (str(pid), row) for pid, row in rows.items()
        if str(row.get("sample_category")) == "hard_negative"
    ]
    structural = sorted(
        str(pid) for pid, row in rows.items()
        if str(row.get("sample_category")) == "structural_diversity"
    )
    noops = sorted(
        str(pid) for pid, row in rows.items()
        if str(row.get("sample_category")) == "controlled_noop"
    )

    # Without premium evidence, preserve the full non-noop diagnostic surface and
    # only enforce the no-op ceiling. Search must create positives rather than have
    # the sampler fabricate a balanced-looking corpus by deleting all negatives.
    if not premium:
        non_noop = sorted(pid for pid, row in rows.items() if str(row.get("sample_category")) != "controlled_noop")
        if not non_noop or noop_frac <= 0.0:
            return tuple(non_noop)
        max_noops = int((len(non_noop) * noop_frac) // max(1e-12, 1.0 - noop_frac))
        return tuple(sorted((*non_noop, *noops[:max_noops])))

    selected = list(premium)
    if structural_frac > 0.0 and structural:
        max_structural = int((len(selected) * structural_frac) // max(1e-12, 1.0 - structural_frac))
        selected.extend(structural[:max_structural])

    def hard_key(item: tuple[str, Mapping[str, Any]]) -> tuple[float, ...]:
        pid, row = item
        violation_count = (
            int(row.get("new_loss_count", 0))
            + int(row.get("w2l_count", 0))
            + int(row.get("severe_regression_count", 0))
        )
        return (
            float(violation_count),
            abs(float(row.get("mean_margin_delta", 0.0))),
            str(pid),
        )

    hard.sort(key=hard_key)
    if hard_frac > 0.0 and hard:
        max_hard = int((len(selected) * hard_frac) // max(1e-12, 1.0 - hard_frac))
        selected.extend(pid for pid, _ in hard[:max_hard])

    if noop_frac > 0.0 and noops:
        max_noops = int((len(selected) * noop_frac) // max(1e-12, 1.0 - noop_frac))
        selected.extend(noops[:max_noops])
    return tuple(sorted(dict.fromkeys(selected)))


def _temporal_regime_group(program: EditProgram) -> str:
    temporal = []
    for atom in program.atoms:
        action = dict(atom.action)
        if atom.operation == "CHANGE_WINDOW":
            temporal.append((atom.operation, int(action.get("prefix_step", -1))))
        for suffix in ("", "2"):
            feature = str(action.get(f"feature{suffix}", ""))
            if feature in {"STEP", "DAY", "HOUR"}:
                temporal.append((feature, str(action.get(f"cmp{suffix}", "")), int(action.get(f"value{suffix}", 0))))
    return _hash_group("BKG4-TEMPORAL-REGIME-GROUP-V1", temporal or [("semantic",)])


def _certification_defaults() -> dict[str, Any]:
    return {
        "certification_class": CertificationClass.UNCERTIFIED.value,
        "transfer_failure": TransferFailure.NOT_CERTIFIED.value,
        "certification_panel_count": 0,
        "cert_strict_safe_retention": 0.0,
        "cert_activation_retention": 0.0,
        "cert_behavior_retention": 0.0,
        "cert_safe_useful_retention": 0.0,
        "cert_elite_retention": 0.0,
        "cert_l2w_retention": 0.0,
        "cert_worst_protected_delta": 0,
        "cert_mean_margin_delta": 0.0,
    }


def _parent_id(base_identity: str, semantic_id: str) -> str:
    return stable_identity("BKG4-EDIT-SEARCH-PARENT-V1", str(base_identity), str(semantic_id))


class SearchCorpusStore:
    """Append-only exact Stage-2 corpus shards derived from persistent search archives."""

    def __init__(self, root: Path):
        self.root = root

    @classmethod
    def create(
        cls,
        path: str | Path,
        *,
        profile: ParentProfile,
        case_bank: CaseBank,
        archive: SearchArchive,
    ) -> "SearchCorpusStore":
        root = Path(path)
        if root.exists() and any(root.iterdir()):
            raise FileExistsError(f"search corpus directory is not empty: {root}")
        root.mkdir(parents=True, exist_ok=True)
        (root / "shards").mkdir(exist_ok=True)
        if archive.base_identity != profile.combined_agent_identity:
            raise ValueError("archive/base parent identity mismatch")
        if tuple(archive.route_ids) != tuple(profile.route_ids):
            raise ValueError("archive/profile route identity mismatch")
        controller_json = profile.root / "controller.json"
        if not controller_json.is_file():
            raise FileNotFoundError(f"missing canonical parent controller JSON: {controller_json}")
        controller_structure = json.loads(controller_json.read_text(encoding="utf-8"))
        base_rows = [{
            "base_id": profile.combined_agent_identity,
            "combined_agent_identity": profile.combined_agent_identity,
            "controller_semantic_hash": profile.controller_semantic_hash,
            "bkg4_abi": profile.bkg4_abi,
            "feature_registry": profile.feature_registry,
            "route_ids": list(profile.route_ids),
            "controller_structure_json": canonical_json(controller_structure),
        }]
        _write_rows(root / "base_structures.parquet", base_rows, BASE_SCHEMA)
        case_rows: list[dict[str, Any]] = []
        for case in sorted(case_bank.rows, key=lambda x: x.case_id):
            model_context = json.loads(case.model_context_json)
            assert_model_payload_safe(model_context)
            _validate_model_context(model_context)
            if not 0 <= int(case.seed) < (1 << 64):
                raise ValueError(f"case seed is outside uint64 ABI range: {case.case_id}")
            if not case.context_probe_steps or case.public_context_width <= 0 or case.public_context_registry_width <= 0:
                raise ValueError(f"ranker corpus case lacks public context: {case.case_id}")
            case_rows.append({
                "case_id": case.case_id,
                "seat": case.seat,
                "model_context_json": case.model_context_json,
                "baseline_margin": case.baseline_margin,
                "context_probe_steps": list(case.context_probe_steps),
                "public_context_width": case.public_context_width,
                "public_context_registry_width": case.public_context_registry_width,
                "public_context_values": list(case.public_context_values),
                "protected": case.protected,
                "prov_seed": case.seed,
                "prov_family": case.family,
                "prov_role": case.role,
                "prov_tape_path": case.tape_path,
                "prov_own_history": case.own_history,
            })
        _write_rows(root / "cases.parquet", case_rows, CASE_SCHEMA)
        store = cls(root)
        store._write_parents(archive)
        manifest = {
            "corpus_schema_version": CORPUS_SCHEMA_VERSION,
            "edit_program_schema_version": EDIT_PROGRAM_SCHEMA_VERSION,
            "grammar_id": GRAMMAR_ID,
            "bkg4_abi": profile.bkg4_abi,
            "feature_registry": profile.feature_registry,
            "cuda_result_abi": "B4EC1/v1",
            "base_identity": profile.combined_agent_identity,
            "root_semantic_id": archive.root_semantic_id,
            "case_bank_identity": case_bank.identity(),
            "route_ids": list(profile.route_ids),
            "model_feature_columns": list(MODEL_FEATURE_COLUMNS),
            "program_model_feature_columns": list(PROGRAM_MODEL_FEATURE_COLUMNS),
            "provenance_prefixes": ["prov_", "search_"],
            "case_roles": {
                "search_training": {
                    "identity": case_bank.identity(),
                    "materialized_for_training": True,
                    "role_counts": dict(sorted(Counter(str(x.role) for x in case_bank.rows).items())),
                },
                "certification": {**dict(archive.external_case_banks.get("certification") or {}), "materialized_for_training": False},
                "ranker_validation": {**dict(archive.external_case_banks.get("ranker_validation") or {}), "materialized_for_training": False},
            },
            "label_contract": {
                "source_discovery": "exact source-parent-relative B4EC1 outcome",
                "root_discovery": "exact root-relative B4EC1 outcome",
                "certification": "aggregate independent B4EC1 transfer vector; label/provenance only",
            },
            "max_controlled_noop_fraction": 0.10,
            "generations": [],
            "counts": {"programs": 0, "labels": 0, "parents": len(archive.states), "cases": len(case_bank)},
        }
        store._write_manifest(manifest)
        return store

    def _manifest(self) -> dict[str, Any]:
        path = self.root / "corpus_manifest.json"
        if not path.is_file():
            raise FileNotFoundError(f"missing corpus manifest: {path}")
        return json.loads(path.read_text(encoding="utf-8"))

    def _write_manifest(self, manifest: Mapping[str, Any]) -> None:
        target = self.root / "corpus_manifest.json"
        tmp = target.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(dict(manifest), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        tmp.replace(target)

    @staticmethod
    def _parent_rows(archive: SearchArchive) -> list[dict[str, Any]]:
        rows = []
        for sid in sorted(archive.states):
            genome = archive.genome(sid)
            rows.append({
                "parent_id": _parent_id(archive.base_identity, sid),
                "base_id": archive.base_identity,
                "overlay_semantic_id": sid,
                "overlay_genome_json": canonical_json(genome.canonical()),
                "overlay_prefix_step": genome.prefix_step,
                "overlay_rule_count": len(genome.rules),
            })
        return rows

    def _write_parents(self, archive: SearchArchive, path: Path | None = None) -> None:
        _write_rows(self.root / "parents.parquet" if path is None else path, self._parent_rows(archive), PARENT_SCHEMA)

    @staticmethod
    def _assert_parquet_schema(path: Path, expected: pa.Schema) -> None:
        if not path.is_file():
            raise ValueError(f"missing corpus parquet: {path}")
        actual = pq.ParquetFile(path).schema_arrow
        if not actual.equals(expected, check_metadata=False):
            raise ValueError(f"corpus Arrow schema mismatch for {path.name}: {actual} != {expected}")

    def _validate_layout(self, manifest: Mapping[str, Any], *, require_generation: bool) -> None:
        for name, schema in (("base_structures.parquet", BASE_SCHEMA), ("cases.parquet", CASE_SCHEMA), ("parents.parquet", PARENT_SCHEMA)):
            self._assert_parquet_schema(self.root / name, schema)
        generations = list(manifest.get("generations", ()))
        if require_generation and not generations:
            raise ValueError("corpus has no complete exported generation")
        expected_programs: set[str] = set()
        expected_labels: set[str] = set()
        seen_generations: set[int] = set()
        for row in generations:
            generation = int(row["generation"])
            if generation < 0 or generation in seen_generations:
                raise ValueError("duplicate/invalid corpus generation in manifest")
            seen_generations.add(generation)
            expected_p = f"shards/programs_g{generation:06d}.parquet"
            expected_l = f"shards/labels_g{generation:06d}.parquet"
            if str(row.get("program_shard")) != expected_p or str(row.get("label_shard")) != expected_l:
                raise ValueError("manifest generation shard names are non-canonical")
            expected_programs.add(expected_p); expected_labels.add(expected_l)
        actual_programs = {str(p.relative_to(self.root)) for p in (self.root / "shards").glob("programs_g*.parquet")}
        actual_labels = {str(p.relative_to(self.root)) for p in (self.root / "shards").glob("labels_g*.parquet")}
        if actual_programs != expected_programs:
            raise ValueError(f"orphan/missing program shards: actual={sorted(actual_programs)} expected={sorted(expected_programs)}")
        if actual_labels != expected_labels:
            raise ValueError(f"orphan/missing label shards: actual={sorted(actual_labels)} expected={sorted(expected_labels)}")
        for rel in sorted(expected_programs): self._assert_parquet_schema(self.root / rel, PROGRAM_SCHEMA)
        for rel in sorted(expected_labels): self._assert_parquet_schema(self.root / rel, LABEL_SCHEMA)

    def append_exact(
        self,
        *,
        generation: int,
        archive: SearchArchive,
        program_pairs: Mapping[str, PairEvaluation],
        root_pairs: Mapping[str, PairEvaluation] | None = None,
    ) -> dict[str, int]:
        manifest = self._manifest()
        if int(manifest["corpus_schema_version"]) != CORPUS_SCHEMA_VERSION or str(manifest["grammar_id"]) != GRAMMAR_ID:
            raise ValueError("incompatible search corpus schema/grammar")
        self._validate_layout(manifest, require_generation=False)
        generation = int(generation)
        if generation < 0:
            raise ValueError("generation must be non-negative")
        if any(int(x["generation"]) == generation for x in manifest.get("generations", [])):
            raise ValueError(f"generation already exported: {generation}")
        if not program_pairs:
            raise ValueError("cannot export an empty exact generation")
        explicit_root_pairs = root_pairs is not None
        root_pairs_map = dict(root_pairs or {})
        if explicit_root_pairs and set(root_pairs_map) != set(program_pairs):
            raise ValueError("source/root discovery program identities differ")
        transaction_id, archive_digest = generation_transaction_id(archive, generation, tuple(program_pairs))
        transaction_root = self.root / "transactions" / f"g{generation:06d}_{transaction_id[:16]}"
        transaction_root.mkdir(parents=True, exist_ok=False)
        snapshot_path = transaction_root / "archive.after.json"
        snapshot_path.write_text(json.dumps(archive.to_dict(), sort_keys=True, indent=2) + "\n", encoding="utf-8")
        tx_meta = {
            "schema_version": 1,
            "generation": generation,
            "transaction_id": transaction_id,
            "archive_digest": archive_digest,
            "status": "prepared",
        }
        (transaction_root / "transaction.json").write_text(json.dumps(tx_meta, sort_keys=True, indent=2) + "\n", encoding="utf-8")

        case_rows_by_id = {str(x["case_id"]): x for x in _read_rows(self.root / "cases.parquet")}
        case_ids = set(case_rows_by_id)
        program_rows: list[dict[str, Any]] = []
        label_rows: list[dict[str, Any]] = []
        for program_id in sorted(program_pairs):
            pair = program_pairs[program_id]
            root_pair = root_pairs_map.get(program_id, pair)
            program = archive.programs.get(program_id)
            if program is None:
                raise ValueError(f"export references unknown search program {program_id}")
            source = archive.genome(program.source_parent_semantic_id)
            child = program.reconstruct(source, archive.route_ids)
            if child.semantic_id() != pair.child_semantic_id or source.semantic_id() != pair.parent_semantic_id:
                raise ValueError("source exact pair identities do not match reconstructable EditProgram")
            if root_pair.child_semantic_id != child.semantic_id():
                raise ValueError("root exact pair child identity differs from reconstructable EditProgram")
            if explicit_root_pairs and root_pair.parent_semantic_id != archive.root_semantic_id:
                raise ValueError("root discovery pair is not root-relative")
            if tuple(str(r["case_id"]) for r in root_pair.case_rows) != tuple(str(r["case_id"]) for r in pair.case_rows):
                raise ValueError("source/root discovery case alignment mismatch")
            if len(pair.case_rows) != len({str(r["case_id"]) for r in pair.case_rows}):
                raise ValueError("duplicate edit-program × case relation")
            missing = {str(r["case_id"]) for r in pair.case_rows} - case_ids
            if missing:
                raise ValueError(f"exact labels reference unknown cases: {sorted(missing)[:5]}")
            for raw in pair.case_rows:
                cid = str(raw["case_id"])
                if bool(raw.get("protected", True)) != bool(case_rows_by_id[cid]["protected"]):
                    raise ValueError(f"exact label protected flag differs from case bank: {cid}")
            agg = pair.certified_aggregate()
            root_agg = root_pair.certified_aggregate()
            dq = discovery_quality(root_pair)
            vector = archive.certification_vector(program_id)
            cert = _certification_defaults()
            if vector is not None:
                cert = {
                    "certification_class": vector.certification_class.value,
                    "transfer_failure": vector.transfer_failure.value,
                    "certification_panel_count": int(vector.panel_count),
                    "cert_strict_safe_retention": float(vector.strict_safe_retention),
                    "cert_activation_retention": float(vector.activation_retention),
                    "cert_behavior_retention": float(vector.behavior_retention),
                    "cert_safe_useful_retention": float(vector.safe_useful_retention),
                    "cert_elite_retention": float(vector.elite_retention),
                    "cert_l2w_retention": float(vector.l2w_retention),
                    "cert_worst_protected_delta": int(vector.worst_certification_protected_delta),
                    "cert_mean_margin_delta": float(vector.certification_mean_margin_delta),
                }
            cert_class = CertificationClass(str(cert["certification_class"]))
            category = _quality_category(
                discovery=dq, certification=cert_class, root_aggregate=root_agg, emitter=program.emitter,
            )
            skeleton = _action_skeleton(program)
            op_target = _operation_target(program)
            program_rows.append({
                "program_id": program.program_id,
                "base_id": archive.base_identity,
                "parent_id": _parent_id(archive.base_identity, program.source_parent_semantic_id),
                "child_parent_id": _parent_id(archive.base_identity, program.child_semantic_id),
                "parent_overlay_semantic_id": program.source_parent_semantic_id,
                "child_overlay_semantic_id": program.child_semantic_id,
                "parent_genome_json": canonical_json(source.canonical()),
                "child_genome_json": canonical_json(child.canonical()),
                "edit_program_json": canonical_json(program.canonical(include_provenance=False)),
                "operations_json": canonical_json([a.operation for a in program.atoms]),
                "targets_json": canonical_json(op_target),
                "program_length": program.program_length,
                "generation_depth": program.generation_depth,
                "parent_rule_count": len(source.rules),
                "child_rule_count": len(child.rules),
                "parent_prefix_step": source.prefix_step,
                "child_prefix_step": child.prefix_step,
                "sample_category": category,
                "discovery_quality": dq.value,
                **cert,
                "temporal_edit": bool(program_contains_temporal(program)),
                "strict_safe": bool(agg["strict_safe"]),
                "exact_noop": bool(agg["exact_noop"]),
                "behavior_changing": bool(agg["behavior_changing"]),
                "l2w_count": int(agg["l2w_count"]),
                "w2l_count": int(agg["w2l_count"]),
                "new_loss_count": int(agg["new_loss_count"]),
                "severe_regression_count": int(agg["severe_regression_count"]),
                "losses_before": int(agg["losses_before"]),
                "losses_after": int(agg["losses_after"]),
                "fragile_wins_before": int(agg["fragile_wins_before"]),
                "fragile_wins_after": int(agg["fragile_wins_after"]),
                "worst_candidate_margin": int(agg["worst_candidate_margin"]),
                "worst_protected_win_delta": int(agg["worst_protected_win_delta"]),
                "mean_margin_delta": float(agg["mean_margin_delta"]),
                "action_change_total": int(agg["action_change_total"]),
                "activation_total": int(agg["activation_total"]),
                "label_count": len(pair.case_rows) + (len(root_pair.case_rows) if explicit_root_pairs else 0),
                "source_label_count": len(pair.case_rows),
                "root_label_count": len(root_pair.case_rows) if explicit_root_pairs else 0,
                "root_strict_safe": bool(root_agg["strict_safe"]),
                "root_behavior_changing": bool(root_agg["behavior_changing"]),
                "root_l2w_count": int(root_agg["l2w_count"]),
                "root_w2l_count": int(root_agg["w2l_count"]),
                "root_new_loss_count": int(root_agg["new_loss_count"]),
                "root_severe_regression_count": int(root_agg["severe_regression_count"]),
                "root_worst_candidate_margin": int(root_agg["worst_candidate_margin"]),
                "root_worst_protected_win_delta": int(root_agg["worst_protected_win_delta"]),
                "root_mean_margin_delta": float(root_agg["mean_margin_delta"]),
                "exact_program_group": program.program_id,
                "parent_family_group": program.source_parent_semantic_id,
                "parent_structural_group": _hash_group("BKG4-PARENT-STRUCTURE-GROUP-V1", {
                    "prefix": source.prefix_step,
                    "rules": [
                        {"features": [c.feature for c in r.conditions], "cmps": [c.cmp for c in r.conditions], "route": r.route}
                        for r in source.rules
                    ],
                }),
                "child_semantic_group": program.child_semantic_id,
                "action_skeleton_group": _hash_group("BKG4-EDIT-SKELETON-GROUP-V1", skeleton),
                "operation_target_group": _hash_group("BKG4-OP-TARGET-GROUP-V1", op_target),
                "lineage_group": _lineage_root(program, archive),
                "run_temporal_group": str(program.run_id or archive.run_id),
                "target_holdout_group": _hash_group("BKG4-TARGET-HOLDOUT-GROUP-V1", op_target),
                "temporal_regime_group": _temporal_regime_group(program),
                "search_generation_group": _hash_group("BKG4-SEARCH-GENERATION-GROUP-V1", {"run": str(program.run_id or archive.run_id), "generation": generation}),
                "certification_panel_group": _hash_group("BKG4-CERTIFICATION-PANEL-GROUP-V1", dict(archive.external_case_banks.get("certification") or {"identity":"uncertified"})),
                "search_emitter": program.emitter,
                "search_seed": program.seed,
                "search_run_id": str(program.run_id or archive.run_id),
                "search_generation": generation,
                "search_structural_descriptor_json": canonical_json(structural_descriptor(program, child)),
                "search_behavior_descriptor_json": canonical_json(behavior_descriptor(pair)),
            })
            for role, role_pair in (("source_discovery", pair), ("root_discovery", root_pair)):
                if role == "root_discovery" and not explicit_root_pairs:
                    continue
                for raw in role_pair.case_rows:
                    row = dict(raw)
                    row["program_id"] = program.program_id
                    row["edit_id"] = program.program_id
                    row["parent_overlay_semantic_id"] = role_pair.parent_semantic_id
                    row["child_overlay_semantic_id"] = program.child_semantic_id
                    row["search_generation"] = generation
                    row["label_role"] = role
                    label_rows.append(row)

        label_rows.sort(key=lambda r: (str(r["program_id"]), str(r["case_id"]), str(r["label_role"])))
        pair_keys = [(str(r["program_id"]), str(r["case_id"]), str(r["label_role"])) for r in label_rows]
        if len(set(pair_keys)) != len(pair_keys):
            raise ValueError("duplicate edit-program × case relation")

        shards = self.root / "shards"
        program_path = shards / f"programs_g{generation:06d}.parquet"
        label_path = shards / f"labels_g{generation:06d}.parquet"
        if program_path.exists() or label_path.exists():
            raise ValueError(f"orphan/canonical files already exist for generation {generation}")
        staging = shards / ".staging" / f"g{generation:06d}"
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir(parents=True, exist_ok=True)
        staged_program = staging / program_path.name
        staged_label = staging / label_path.name
        staged_parents = staging / "parents.parquet"
        prior_parents = staging / "parents.previous.parquet"
        parent_path = self.root / "parents.parquet"
        parent_replaced = False
        published_program = False
        published_label = False
        try:
            _write_rows(staged_program, program_rows, PROGRAM_SCHEMA)
            _write_rows(staged_label, label_rows, LABEL_SCHEMA)
            self._write_parents(archive, staged_parents)
            self._assert_parquet_schema(staged_program, PROGRAM_SCHEMA)
            self._assert_parquet_schema(staged_label, LABEL_SCHEMA)
            self._assert_parquet_schema(staged_parents, PARENT_SCHEMA)
            if pq.ParquetFile(staged_program).metadata.num_rows != len(program_rows) or pq.ParquetFile(staged_label).metadata.num_rows != len(label_rows):
                raise ValueError("staged generation row-count mismatch")

            next_manifest = json.loads(json.dumps(manifest))
            next_manifest["generations"].append({
                "generation": generation,
                "program_shard": str(program_path.relative_to(self.root)),
                "label_shard": str(label_path.relative_to(self.root)),
                "program_rows": len(program_rows),
                "label_rows": len(label_rows),
                "transaction_id": transaction_id,
                "archive_digest": archive_digest,
                "transaction_dir": str(transaction_root.relative_to(self.root)),
            })
            next_manifest["generations"].sort(key=lambda x: int(x["generation"]))
            next_manifest["counts"] = {
                "programs": sum(int(x["program_rows"]) for x in next_manifest["generations"]),
                "labels": sum(int(x["label_rows"]) for x in next_manifest["generations"]),
                "parents": len(archive.states),
                "cases": len(case_ids),
            }

            shutil.copy2(parent_path, prior_parents)
            shutil.copy2(parent_path, transaction_root / "parents.before.parquet")
            os.replace(staged_program, program_path); published_program = True
            os.replace(staged_label, label_path); published_label = True
            os.replace(staged_parents, parent_path); parent_replaced = True
            # Prepare every recoverable artifact before the manifest. The manifest is
            # the sole commit record and is the final fallible write in the transaction.
            tx_meta["status"] = "ready_to_commit"
            (transaction_root / "transaction.json").write_text(json.dumps(tx_meta, sort_keys=True, indent=2) + "\n", encoding="utf-8")
            self._write_manifest(next_manifest)
        except Exception:
            if published_program:
                program_path.unlink(missing_ok=True)
            if published_label:
                label_path.unlink(missing_ok=True)
            if parent_replaced and prior_parents.exists():
                os.replace(prior_parents, parent_path)
            shutil.rmtree(staging, ignore_errors=True)
            shutil.rmtree(transaction_root, ignore_errors=True)
            raise
        shutil.rmtree(staging, ignore_errors=True)
        self._validate_layout(self._manifest(), require_generation=True)
        return {"programs": len(program_rows), "labels": len(label_rows), "transaction_id": transaction_id, "archive_digest": archive_digest}


    def recover_archive(self, archive_path: str | Path) -> dict[str, Any]:
        """Recover or reject the corpus/archive pair using immutable generation snapshots.

        Corpus manifest publication is the cross-artifact commit intent. If it exists but
        the canonical archive is stale, the post-evaluation archive snapshot is restored.
        Prepared transactions that never reached the corpus manifest are rolled back.
        """
        manifest = self._manifest()
        generations = sorted(list(manifest.get("generations", ())), key=lambda x: int(x["generation"]))
        committed_by_generation = {int(x["generation"]): x for x in generations}
        tx_root = self.root / "transactions"
        rolled_back: list[int] = []
        tx_root.mkdir(parents=True, exist_ok=True)

        # A process can die after canonical shard/parents publication but before the
        # manifest commit. Such a prepared transaction is not committed and is reverted.
        for directory in sorted(p for p in tx_root.iterdir() if p.is_dir()):
            meta_path = directory / "transaction.json"
            if not meta_path.is_file():
                raise ValueError(f"generation transaction missing metadata: {directory}")
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            generation = int(meta.get("generation", -1))
            if generation in committed_by_generation:
                continue
            if generation < 0:
                raise ValueError("invalid prepared generation transaction")
            (self.root / "shards" / f"programs_g{generation:06d}.parquet").unlink(missing_ok=True)
            (self.root / "shards" / f"labels_g{generation:06d}.parquet").unlink(missing_ok=True)
            before = directory / "parents.before.parquet"
            if before.is_file():
                shutil.copy2(before, self.root / "parents.parquet")
            shutil.rmtree(self.root / "shards" / ".staging" / f"g{generation:06d}", ignore_errors=True)
            shutil.rmtree(directory)
            rolled_back.append(generation)

        self._validate_layout(manifest, require_generation=bool(generations))
        if not generations:
            return {"recovered_generations": [], "rolled_back_generations": sorted(rolled_back), "committed_generations": []}

        snapshots: dict[int, SearchArchive] = {}
        for row in generations:
            generation = int(row["generation"])
            transaction_id = str(row.get("transaction_id", ""))
            archive_digest = str(row.get("archive_digest", ""))
            transaction_dir = str(row.get("transaction_dir", ""))
            if not transaction_id or len(archive_digest) != 64 or not transaction_dir:
                raise ValueError("committed corpus generation lacks immutable archive transaction identity")
            directory = self.root / transaction_dir
            meta_path = directory / "transaction.json"
            snapshot_path = directory / "archive.after.json"
            if not meta_path.is_file() or not snapshot_path.is_file():
                raise ValueError("committed generation transaction snapshot is missing")
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if int(meta.get("generation", -1)) != generation or str(meta.get("transaction_id")) != transaction_id or str(meta.get("archive_digest")) != archive_digest:
                raise ValueError("generation transaction metadata disagrees with corpus manifest")
            snapshot = SearchArchive.load(snapshot_path)
            if snapshot.base_identity != str(manifest["base_identity"]):
                raise ValueError("generation transaction archive belongs to a different BKG4 parent")
            if archive_transaction_digest(snapshot) != archive_digest:
                raise ValueError("generation transaction archive digest mismatch")
            program_ids = [str(x["program_id"]) for x in pq.read_table(self.root / str(row["program_shard"]), columns=["program_id"]).to_pylist()]
            expected_tx, _ = generation_transaction_id(snapshot, generation, program_ids)
            if expected_tx != transaction_id:
                raise ValueError("generation transaction identity does not reconstruct")
            snapshots[generation] = snapshot

        archive_file = Path(archive_path)
        current = SearchArchive.load(archive_file) if archive_file.is_file() else None
        if current is not None:
            if current.base_identity != str(manifest["base_identity"]):
                raise ValueError("canonical archive/corpus parent identity mismatch")
            manifest_ids = {int(x["generation"]): str(x["transaction_id"]) for x in generations}
            for generation, transaction_id in current.runtime_state.generation_transactions.items():
                if manifest_ids.get(int(generation)) != str(transaction_id):
                    raise ValueError("canonical archive contains a generation transaction absent/different in corpus")

        latest_generation = int(generations[-1]["generation"])
        latest = snapshots[latest_generation]
        recovered = []
        current_tx = {} if current is None else dict(current.runtime_state.generation_transactions)
        for row in generations:
            generation = int(row["generation"]); transaction_id = str(row["transaction_id"])
            if current_tx.get(generation) != transaction_id:
                recovered.append(generation)
            latest.mark_exported_generation(generation, transaction_id)
        latest.save(archive_file)

        # Commit marker is deliberately last. A missing marker is recoverable because
        # corpus manifest + immutable snapshot are already sufficient to reconstruct.
        for row in generations:
            directory = self.root / str(row["transaction_dir"])
            meta_path = directory / "transaction.json"
            meta = json.loads(meta_path.read_text(encoding="utf-8")); meta["status"] = "committed"
            meta_path.write_text(json.dumps(meta, sort_keys=True, indent=2) + "\n", encoding="utf-8")
            (directory / "COMMITTED").write_text(str(row["transaction_id"]) + "\n", encoding="utf-8")
        return {
            "recovered_generations": sorted(recovered),
            "rolled_back_generations": sorted(rolled_back),
            "committed_generations": [int(x["generation"]) for x in generations],
        }


def _validate_model_boundary(manifest: Mapping[str, Any], case_rows: list[dict[str, Any]]) -> None:
    features = tuple(str(x) for x in manifest.get("model_feature_columns", ()))
    program_features = tuple(str(x) for x in manifest.get("program_model_feature_columns", ()))
    if features != MODEL_FEATURE_COLUMNS:
        raise ValueError(f"model feature contract mismatch/unknown feature: {features}")
    if program_features != PROGRAM_MODEL_FEATURE_COLUMNS:
        raise ValueError(f"program model feature contract mismatch/unknown feature: {program_features}")
    for row in case_rows:
        payload = json.loads(str(row["model_context_json"]))
        assert_model_payload_safe(payload)
        _validate_model_context(payload)


def _authoritative_case_row(row: Mapping[str, Any]) -> dict[str, Any]:
    expected = derive_case_label(
        int(row["baseline_margin"]), int(row["candidate_margin"]), int(row["action_changes"]), int(row["activations"]),
        protected=bool(row["protected"]), parent_action_trace_hash=int(row["parent_action_trace_hash"]),
        candidate_action_trace_hash=int(row["candidate_action_trace_hash"]),
    )
    expected["case_id"] = str(row["case_id"])
    return expected


def _validate_case_label(row: Mapping[str, Any], case_row: Mapping[str, Any]) -> dict[str, Any]:
    expected = _authoritative_case_row(row)
    for key in ("baseline_margin", "candidate_margin", "margin_delta", "baseline_outcome", "candidate_outcome", "l2w", "w2l", "new_loss", "protected", "action_changes", "activations", "parent_action_trace_hash", "candidate_action_trace_hash", "action_trace_equal", "behavior_changing"):
        if row[key] != expected[key]:
            raise ValueError(f"label recomputation mismatch: {key}")
    if int(row["candidate_action_changes_from_base"]) != int(row["action_changes"]):
        raise ValueError("label recomputation mismatch: candidate_action_changes_from_base")
    if int(row["candidate_activations"]) != int(row["activations"]):
        raise ValueError("label recomputation mismatch: candidate_activations")
    if bool(row["protected"]) != bool(case_row["protected"]):
        raise ValueError("label protected flag differs from authoritative case row")
    return expected


def _validate_certification_semantics(program_row: Mapping[str, Any]) -> CertificationClass:
    cls = CertificationClass(str(program_row["certification_class"]))
    TransferFailure(str(program_row["transfer_failure"]))
    panel_count = int(program_row["certification_panel_count"])
    safe = float(program_row["cert_strict_safe_retention"])
    active = float(program_row["cert_activation_retention"])
    behavior = float(program_row["cert_behavior_retention"])
    useful = float(program_row["cert_safe_useful_retention"])
    elite = float(program_row["cert_elite_retention"])
    l2w = float(program_row["cert_l2w_retention"])
    fractions = (safe, active, behavior, useful, elite, l2w)
    if panel_count < 0 or any(not 0.0 <= x <= 1.0 for x in fractions):
        raise ValueError("invalid certification vector range")
    if cls == CertificationClass.UNCERTIFIED:
        if panel_count != 0 or any(x != 0.0 for x in fractions):
            raise ValueError("uncertified program carries certification outcomes")
    else:
        if panel_count <= 0:
            raise ValueError("certified/generalization program lacks certification panels")
        if cls == CertificationClass.GENERALIZATION_HARD_NEGATIVE and not safe < 1.0:
            raise ValueError("generalization hard negative must have unsafe transfer")
        if cls == CertificationClass.GENERALIZATION_BOUNDARY and not (safe == 1.0 and (behavior < 0.50 or useful < 0.40)):
            raise ValueError("generalization boundary semantics mismatch")
        if cls in {CertificationClass.CERTIFIED_ELITE_FLIP, CertificationClass.CERTIFIED_ELITE_MARGIN, CertificationClass.CERTIFIED_SAFE_USEFUL, CertificationClass.CERTIFIED_SAFE_INCREMENTAL} and safe != 1.0:
            raise ValueError("certified positive must be safe on every certification panel")
        if cls == CertificationClass.CERTIFIED_ELITE_FLIP and not (behavior >= 0.50 and useful >= 0.40 and elite >= 0.25 and l2w > 0.0):
            raise ValueError("certified elite flip semantics mismatch")
        if cls == CertificationClass.CERTIFIED_ELITE_MARGIN and not (behavior >= 0.50 and useful >= 0.40 and elite >= 0.25):
            raise ValueError("certified elite margin semantics mismatch")
        if cls == CertificationClass.CERTIFIED_SAFE_USEFUL and not (behavior >= 0.50 and useful >= 0.40):
            raise ValueError("certified safe-useful semantics mismatch")
    return cls


def _validate_program_aggregate(program_row: Mapping[str, Any], exact_rows: list[dict[str, Any]]) -> None:
    source_rows = [r for r in exact_rows if str(r.get("label_role")) == "source_discovery"]
    root_rows = [r for r in exact_rows if str(r.get("label_role")) == "root_discovery"]
    if len(source_rows) != int(program_row["source_label_count"]) or len(root_rows) != int(program_row["root_label_count"]):
        raise ValueError("program aggregate recomputation mismatch: role label counts")
    if not source_rows or any(str(r.get("label_role")) not in {"source_discovery", "root_discovery"} for r in exact_rows):
        raise ValueError("invalid/empty exact discovery label roles")
    aggregate = derive_edit_labels(source_rows)
    root_aggregate = derive_edit_labels(root_rows or source_rows)
    checks = {
        "l2w_count": int(aggregate["l2w_count"]), "w2l_count": int(aggregate["w2l_count"]),
        "new_loss_count": int(aggregate["new_loss_count"]), "severe_regression_count": int(aggregate["severe_regression_count"]),
        "losses_before": int(aggregate["losses_before"]), "losses_after": int(aggregate["losses_after"]),
        "fragile_wins_before": int(aggregate["fragile_wins_before"]), "fragile_wins_after": int(aggregate["fragile_wins_after"]),
        "worst_candidate_margin": int(aggregate["worst_candidate_margin"]),
        "worst_protected_win_delta": int(aggregate["worst_protected_win_delta"]),
        "action_change_total": int(aggregate["action_change_total"]), "activation_total": int(aggregate["activation_total"]),
        "strict_safe": bool(aggregate["strict_safe"]), "exact_noop": bool(aggregate["exact_noop"]),
        "behavior_changing": bool(aggregate["behavior_changing"]),
        "root_strict_safe": bool(root_aggregate["strict_safe"]),
        "root_behavior_changing": bool(root_aggregate["behavior_changing"]),
        "root_l2w_count": int(root_aggregate["l2w_count"]), "root_w2l_count": int(root_aggregate["w2l_count"]),
        "root_new_loss_count": int(root_aggregate["new_loss_count"]), "root_severe_regression_count": int(root_aggregate["severe_regression_count"]),
        "root_worst_candidate_margin": int(root_aggregate["worst_candidate_margin"]),
        "root_worst_protected_win_delta": int(root_aggregate["worst_protected_win_delta"]),
    }
    for key, expected in checks.items():
        if program_row[key] != expected:
            raise ValueError(f"program aggregate recomputation mismatch: {key}")
    if not np.isclose(float(program_row["mean_margin_delta"]), float(aggregate["mean_margin_delta"]), rtol=0.0, atol=1e-12):
        raise ValueError("program aggregate recomputation mismatch: mean_margin_delta")
    if not np.isclose(float(program_row["root_mean_margin_delta"]), float(root_aggregate["mean_margin_delta"]), rtol=0.0, atol=1e-12):
        raise ValueError("program aggregate recomputation mismatch: root_mean_margin_delta")
    if int(program_row["label_count"]) != len(exact_rows):
        raise ValueError("program aggregate recomputation mismatch: label_count")
    synthetic = PairEvaluation(
        parent_semantic_id="root", child_semantic_id="child", panel_id="validated",
        case_rows=tuple(root_rows or source_rows), aggregate={},
    )
    dq = discovery_quality(synthetic)
    if str(program_row["discovery_quality"]) != dq.value:
        raise ValueError("program discovery-quality recomputation mismatch")
    cls = _validate_certification_semantics(program_row)
    expected_category = _quality_category(
        discovery=dq, certification=cls, root_aggregate=root_aggregate,
        emitter=str(program_row["search_emitter"]),
    )
    if str(program_row["sample_category"]) != expected_category:
        raise ValueError("program aggregate recomputation mismatch: sample_category")


def validate_export(path: str | Path) -> dict[str, Any]:
    root = Path(path)
    manifest_path = root / "corpus_manifest.json"
    if not manifest_path.is_file():
        raise ValueError("missing corpus manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if int(manifest.get("corpus_schema_version", -1)) != CORPUS_SCHEMA_VERSION:
        raise ValueError("incompatible corpus schema")
    if int(manifest.get("edit_program_schema_version", -1)) != EDIT_PROGRAM_SCHEMA_VERSION or str(manifest.get("grammar_id")) != GRAMMAR_ID:
        raise ValueError("incompatible edit-program schema/grammar")
    store = SearchCorpusStore(root)
    store._validate_layout(manifest, require_generation=True)
    route_ids = tuple(int(x) for x in manifest["route_ids"])
    if not route_ids or len(set(route_ids)) != len(route_ids):
        raise ValueError("invalid corpus route identity")

    base_rows = _read_rows(root / "base_structures.parquet")
    if len(base_rows) != 1 or str(base_rows[0]["base_id"]) != str(manifest["base_identity"]):
        raise ValueError("base-structure/manifest identity mismatch")
    if tuple(int(x) for x in base_rows[0]["route_ids"]) != route_ids:
        raise ValueError("base-structure route identity mismatch")

    case_rows = _read_rows(root / "cases.parquet")
    case_by_id = {str(x["case_id"]): x for x in case_rows}
    if len(case_by_id) != len(case_rows) or not case_rows:
        raise ValueError("duplicate/empty case identity in corpus")
    _validate_model_boundary(manifest, case_rows)
    for row in case_rows:
        if not 0 <= int(row["prov_seed"]) < (1 << 64):
            raise ValueError("case provenance seed outside uint64 range")

    parents_rows = _read_rows(root / "parents.parquet")
    parents: dict[str, Any] = {}
    for row in parents_rows:
        if str(row["base_id"]) != str(manifest["base_identity"]):
            raise ValueError("parent/base identity mismatch")
        genome_obj = json.loads(str(row["overlay_genome_json"]))
        genome = genome_from_json(genome_obj); genome.validate(route_ids)
        sid = genome.semantic_id()
        if sid != str(row["overlay_semantic_id"]) or str(row["parent_id"]) != _parent_id(str(manifest["base_identity"]), sid):
            raise ValueError("parent semantic/canonical identity mismatch")
        if sid in parents:
            raise ValueError("duplicate parent semantic identity")
        parents[sid] = genome
    if not parents:
        raise ValueError("corpus has no parent states")

    all_programs: dict[str, dict[str, Any]] = {}
    expected_labels: dict[str, int] = {}
    program_generation: dict[str, int] = {}
    operation_counts: Counter[str] = Counter()
    target_counts: Counter[str] = Counter()
    action_groups: Counter[str] = Counter()
    child_groups: Counter[str] = Counter()
    reconstruction_failures = 0
    generation_rows = list(manifest.get("generations", ()))
    for gen in generation_rows:
        generation = int(gen["generation"])
        ppath = root / str(gen["program_shard"])
        lpath = root / str(gen["label_shard"])
        pf = pq.ParquetFile(ppath); lf = pq.ParquetFile(lpath)
        if pf.metadata.num_rows != int(gen["program_rows"]) or lf.metadata.num_rows != int(gen["label_rows"]):
            raise ValueError("manifest/parquet row-count mismatch")
        for row in pf.read().to_pylist():
            pid = str(row["program_id"])
            if pid in all_programs:
                raise ValueError("duplicate exact program identity across corpus shards")
            if int(row["search_generation"]) != generation:
                raise ValueError("program search-generation mismatch")
            source_sid = str(row["parent_overlay_semantic_id"]); child_sid = str(row["child_overlay_semantic_id"])
            if source_sid not in parents or child_sid not in parents:
                raise ValueError("program references unresolved parent/child state")
            if str(row["parent_id"]) != _parent_id(str(manifest["base_identity"]), source_sid) or str(row["child_parent_id"]) != _parent_id(str(manifest["base_identity"]), child_sid):
                raise ValueError("program parent identity group mismatch")
            program = EditProgram.from_dict(json.loads(str(row["edit_program_json"])))
            try:
                rebuilt = program.reconstruct(parents[source_sid], route_ids)
            except Exception as exc:
                reconstruction_failures += 1
                raise ValueError(f"program reconstruction failure: {pid}: {exc}") from exc
            if pid != program.program_id or rebuilt.semantic_id() != child_sid or child_sid != program.child_semantic_id or source_sid != program.source_parent_semantic_id:
                reconstruction_failures += 1
                raise ValueError("program reconstruction child/source identity mismatch")
            if canonical_json(rebuilt.canonical()) != canonical_json(json.loads(str(row["child_genome_json"]))):
                reconstruction_failures += 1
                raise ValueError("recorded child genome differs from reconstructed child")
            if canonical_json(parents[source_sid].canonical()) != canonical_json(json.loads(str(row["parent_genome_json"]))):
                raise ValueError("recorded parent genome differs from resolved parent")
            if str(row["exact_program_group"]) != pid or str(row["child_semantic_group"]) != child_sid or str(row["parent_family_group"]) != source_sid:
                raise ValueError("program split-group identity mismatch")
            all_programs[pid] = row; expected_labels[pid] = int(row["label_count"]); program_generation[pid] = generation
            for atom in program.atoms:
                operation_counts[atom.operation] += 1; target_counts[atom.target_kind] += 1
            action_groups[str(row["action_skeleton_group"])] += 1; child_groups[child_sid] += 1

    if not all_programs:
        raise ValueError("corpus has no exact programs")

    observed_labels: Counter[str] = Counter()
    total_labels = 0
    for gen in generation_rows:
        generation = int(gen["generation"]); lpath = root / str(gen["label_shard"])
        last_key: tuple[str, str, str] | None = None
        current_pid: str | None = None
        current_exact_rows: list[dict[str, Any]] = []
        def finish_program() -> None:
            nonlocal current_pid, current_exact_rows
            if current_pid is None:
                return
            _validate_program_aggregate(all_programs[current_pid], current_exact_rows)
            current_pid = None; current_exact_rows = []
        for batch in pq.ParquetFile(lpath).iter_batches(batch_size=65536):
            for row in batch.to_pylist():
                pid = str(row["program_id"]); cid = str(row["case_id"]); role = str(row["label_role"]); key = (pid, cid, role)
                if last_key is not None and key <= last_key:
                    raise ValueError("duplicate or unsorted edit-program × case relation")
                last_key = key
                if pid not in all_programs or program_generation[pid] != generation:
                    raise ValueError("label references unknown/wrong-generation exact program")
                if cid not in case_by_id:
                    raise ValueError("label references unknown case")
                if str(row["edit_id"]) != pid or int(row["search_generation"]) != generation:
                    raise ValueError("label program/generation identity mismatch")
                if role not in {"source_discovery", "root_discovery"}:
                    raise ValueError("unknown exact label role")
                expected_parent = str(all_programs[pid]["parent_overlay_semantic_id"]) if role == "source_discovery" else str(manifest["root_semantic_id"])
                if str(row["parent_overlay_semantic_id"]) != expected_parent:
                    raise ValueError(f"{role} label parent semantic identity mismatch")
                if str(row["child_overlay_semantic_id"]) != str(all_programs[pid]["child_overlay_semantic_id"]):
                    raise ValueError("label child semantic identity mismatch")
                if current_pid is not None and pid != current_pid:
                    finish_program()
                if current_pid is None:
                    current_pid = pid
                exact = _validate_case_label(row, case_by_id[cid])
                exact["label_role"] = role
                current_exact_rows.append(exact); observed_labels[pid] += 1; total_labels += 1
        finish_program()

    for pid, expected in expected_labels.items():
        if observed_labels[pid] != expected:
            raise ValueError(f"program label participation mismatch: {pid}: {observed_labels[pid]} != {expected}")
    counts = dict(manifest.get("counts") or {})
    expected_global = {"programs": len(all_programs), "labels": total_labels, "parents": len(parents), "cases": len(case_by_id)}
    if any(int(counts.get(k, -1)) != v for k, v in expected_global.items()):
        raise ValueError(f"manifest global counts disagree with validated corpus: {counts} != {expected_global}")

    category_counts = Counter(str(row["sample_category"]) for row in all_programs.values())
    noop_rate = category_counts.get("controlled_noop", 0) / max(1, len(all_programs))
    semantic_child_duplicate_rate = sum(max(0, n - 1) for n in child_groups.values()) / max(1, len(all_programs))
    action_skeleton_duplicate_rate = sum(max(0, n - 1) for n in action_groups.values()) / max(1, len(all_programs))
    return {
        "programs": len(all_programs), "labels": total_labels, "cases": len(case_by_id), "parents": len(parents),
        "generations": len(generation_rows), "reconstruction_failures": reconstruction_failures, "referential_integrity_failures": 0,
        "exact_program_duplicate_rate": 0.0, "child_semantic_duplicate_rate": semantic_child_duplicate_rate,
        "action_skeleton_duplicate_rate": action_skeleton_duplicate_rate, "operation_counts": dict(sorted(operation_counts.items())),
        "target_kind_counts": dict(sorted(target_counts.items())), "model_feature_columns": list(manifest["model_feature_columns"]),
        "program_model_feature_columns": list(manifest["program_model_feature_columns"]),
        "uint64_trace_schema": True, "complete_generation_layout": True, "aggregates_recomputed": True, "model_boundary_allowlisted": True,
        "certified_elite": int(category_counts.get("certified_elite", 0)),
        "certified_safe_useful": int(category_counts.get("certified_safe_useful", 0)),
        "discovery_only_positive": int(category_counts.get("discovery_only_positive", 0)),
        "generalization_boundary": int(category_counts.get("generalization_boundary", 0)),
        "generalization_hard_negative": int(category_counts.get("generalization_hard_negative", 0)),
        "near_feasible_repair": int(category_counts.get("near_feasible_repair", 0)),
        "hard_negative": int(category_counts.get("hard_negative", 0)),
        "structural_diversity": int(category_counts.get("structural_diversity", 0)),
        "controlled_noop": int(category_counts.get("controlled_noop", 0)),
        "controlled_noop_rate": float(noop_rate),
        "category_counts": dict(sorted(category_counts.items())),
    }
