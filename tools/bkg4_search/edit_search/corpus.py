from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from tools.bkg4_search.module_codec import genome_from_json
from tools.edit_corpus.corpus import assert_model_payload_safe, canonical_json, stable_identity

from . import CORPUS_SCHEMA_VERSION, EDIT_PROGRAM_SCHEMA_VERSION, GRAMMAR_ID
from .archive import SearchArchive, behavior_descriptor, structural_descriptor
from .fitness import PairEvaluation
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


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path, compression="zstd")


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


def _sample_category(pair: PairEvaluation) -> str:
    a = pair.aggregate
    behavior = bool(a["behavior_changing"])
    safe = bool(a["strict_safe"])
    useful = behavior and (int(a["l2w_count"]) > 0 or float(a["mean_margin_delta"]) > 0.0)
    if safe and useful:
        return "strict_safe_useful"
    if safe and behavior:
        return "safe_incremental"
    if bool(a["exact_noop"]):
        return "controlled_noop"
    if int(a["w2l_count"]) > 0 or int(a["severe_regression_count"]) > 0 or float(a["mean_margin_delta"]) < 0.0:
        return "hard_negative"
    if behavior:
        return "boundary"
    return "novel_diverse"


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
        _write_rows(root / "base_structures.parquet", base_rows)
        case_rows: list[dict[str, Any]] = []
        for case in sorted(case_bank.rows, key=lambda x: x.case_id):
            assert_model_payload_safe(json.loads(case.model_context_json))
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
        _write_rows(root / "cases.parquet", case_rows)
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
            "case_bank_identity": case_bank.identity(),
            "route_ids": list(profile.route_ids),
            "model_feature_columns": list(MODEL_FEATURE_COLUMNS),
            "program_model_feature_columns": list(PROGRAM_MODEL_FEATURE_COLUMNS),
            "provenance_prefixes": ["prov_", "search_"],
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

    def _write_parents(self, archive: SearchArchive) -> None:
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
        _write_rows(self.root / "parents.parquet", rows)

    def append_exact(
        self,
        *,
        generation: int,
        archive: SearchArchive,
        program_pairs: Mapping[str, PairEvaluation],
    ) -> dict[str, int]:
        manifest = self._manifest()
        if int(manifest["corpus_schema_version"]) != CORPUS_SCHEMA_VERSION or str(manifest["grammar_id"]) != GRAMMAR_ID:
            raise ValueError("incompatible search corpus schema/grammar")
        generation = int(generation)
        if generation < 0:
            raise ValueError("generation must be non-negative")
        if any(int(x["generation"]) == generation for x in manifest.get("generations", [])):
            raise ValueError(f"generation already exported: {generation}")
        if not program_pairs:
            raise ValueError("cannot export an empty exact generation")
        case_ids = {str(x["case_id"]) for x in _read_rows(self.root / "cases.parquet")}
        program_rows: list[dict[str, Any]] = []
        label_rows: list[dict[str, Any]] = []
        for program_id in sorted(program_pairs):
            pair = program_pairs[program_id]
            program = archive.programs.get(program_id)
            if program is None:
                raise ValueError(f"export references unknown search program {program_id}")
            source = archive.genome(program.source_parent_semantic_id)
            child = program.reconstruct(source, archive.route_ids)
            if child.semantic_id() != pair.child_semantic_id or source.semantic_id() != pair.parent_semantic_id:
                raise ValueError("exact pair identities do not match reconstructable EditProgram")
            if len(pair.case_rows) != len({str(r["case_id"]) for r in pair.case_rows}):
                raise ValueError("duplicate edit-program × case relation")
            missing = {str(r["case_id"]) for r in pair.case_rows} - case_ids
            if missing:
                raise ValueError(f"exact labels reference unknown cases: {sorted(missing)[:5]}")
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
                "sample_category": _sample_category(pair),
                "strict_safe": bool(pair.aggregate["strict_safe"]),
                "exact_noop": bool(pair.aggregate["exact_noop"]),
                "behavior_changing": bool(pair.aggregate["behavior_changing"]),
                "l2w_count": int(pair.aggregate["l2w_count"]),
                "w2l_count": int(pair.aggregate["w2l_count"]),
                "severe_regression_count": int(pair.aggregate["severe_regression_count"]),
                "losses_after": int(pair.aggregate["losses_after"]),
                "fragile_wins_after": int(pair.aggregate["fragile_wins_after"]),
                "worst_candidate_margin": int(pair.aggregate["worst_candidate_margin"]),
                "worst_protected_win_delta": int(pair.aggregate["worst_protected_win_delta"]),
                "mean_margin_delta": float(pair.aggregate["mean_margin_delta"]),
                "label_count": len(pair.case_rows),
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
                "search_emitter": program.emitter,
                "search_seed": program.seed,
                "search_run_id": str(program.run_id or archive.run_id),
                "search_generation": generation,
                "search_structural_descriptor_json": canonical_json(structural_descriptor(program, child)),
                "search_behavior_descriptor_json": canonical_json(behavior_descriptor(pair)),
            })
            for raw in pair.case_rows:
                row = dict(raw)
                row["program_id"] = program.program_id
                row["edit_id"] = program.program_id
                row["parent_overlay_semantic_id"] = program.source_parent_semantic_id
                row["child_overlay_semantic_id"] = program.child_semantic_id
                row["search_generation"] = generation
                label_rows.append(row)
        pair_keys = [(str(r["program_id"]), str(r["case_id"])) for r in label_rows]
        if len(set(pair_keys)) != len(pair_keys):
            raise ValueError("duplicate edit-program × case relation")
        program_path = self.root / "shards" / f"programs_g{generation:06d}.parquet"
        label_path = self.root / "shards" / f"labels_g{generation:06d}.parquet"
        _write_rows(program_path, program_rows)
        _write_rows(label_path, sorted(label_rows, key=lambda r: (str(r["program_id"]), str(r["case_id"]))))
        self._write_parents(archive)
        manifest["generations"].append({
            "generation": generation,
            "program_shard": str(program_path.relative_to(self.root)),
            "label_shard": str(label_path.relative_to(self.root)),
            "program_rows": len(program_rows),
            "label_rows": len(label_rows),
        })
        manifest["generations"].sort(key=lambda x: int(x["generation"]))
        manifest["counts"] = {
            "programs": sum(int(x["program_rows"]) for x in manifest["generations"]),
            "labels": sum(int(x["label_rows"]) for x in manifest["generations"]),
            "parents": len(archive.states),
            "cases": len(case_ids),
        }
        self._write_manifest(manifest)
        return {"programs": len(program_rows), "labels": len(label_rows)}


def _validate_model_boundary(manifest: Mapping[str, Any], case_rows: list[dict[str, Any]]) -> None:
    features = [str(x) for x in manifest.get("model_feature_columns", ())]
    program_features = [str(x) for x in manifest.get("program_model_feature_columns", ())]
    forbidden = [x for x in features + program_features if x.startswith("prov_") or x.startswith("search_")]
    if forbidden:
        raise ValueError(f"provenance/search leakage in model feature columns: {forbidden}")
    for row in case_rows:
        assert_model_payload_safe(json.loads(str(row["model_context_json"])))


def validate_export(path: str | Path) -> dict[str, Any]:
    root = Path(path)
    manifest = json.loads((root / "corpus_manifest.json").read_text(encoding="utf-8"))
    if int(manifest.get("corpus_schema_version", -1)) != CORPUS_SCHEMA_VERSION:
        raise ValueError("incompatible corpus schema")
    if int(manifest.get("edit_program_schema_version", -1)) != EDIT_PROGRAM_SCHEMA_VERSION or str(manifest.get("grammar_id")) != GRAMMAR_ID:
        raise ValueError("incompatible edit-program schema/grammar")
    route_ids = tuple(int(x) for x in manifest["route_ids"])
    case_rows = _read_rows(root / "cases.parquet")
    case_ids = {str(x["case_id"]) for x in case_rows}
    if len(case_ids) != len(case_rows):
        raise ValueError("duplicate case identity in corpus")
    _validate_model_boundary(manifest, case_rows)
    parents_rows = _read_rows(root / "parents.parquet")
    parents: dict[str, Any] = {}
    for row in parents_rows:
        genome_obj = json.loads(str(row["overlay_genome_json"]))
        genome = genome_from_json(genome_obj)
        genome.validate(route_ids)
        sid = genome.semantic_id()
        if sid != str(row["overlay_semantic_id"]):
            raise ValueError("parent semantic identity mismatch")
        if sid in parents:
            raise ValueError("duplicate parent semantic identity")
        parents[sid] = genome
    if not parents:
        raise ValueError("corpus has no parent states")

    all_programs: dict[str, dict[str, Any]] = {}
    expected_labels: dict[str, int] = {}
    operation_counts: Counter[str] = Counter()
    target_counts: Counter[str] = Counter()
    action_groups: Counter[str] = Counter()
    child_groups: Counter[str] = Counter()
    reconstruction_failures = 0
    generation_rows = manifest.get("generations", [])
    for gen in generation_rows:
        ppath = root / str(gen["program_shard"])
        lpath = root / str(gen["label_shard"])
        if not ppath.is_file() or not lpath.is_file():
            raise ValueError("manifest references missing corpus shard")
        pf = pq.ParquetFile(ppath)
        lf = pq.ParquetFile(lpath)
        if pf.metadata.num_rows != int(gen["program_rows"]) or lf.metadata.num_rows != int(gen["label_rows"]):
            raise ValueError("manifest/parquet row-count mismatch")
        for row in pf.read().to_pylist():
            pid = str(row["program_id"])
            if pid in all_programs:
                raise ValueError("duplicate exact program identity across corpus shards")
            source_sid = str(row["parent_overlay_semantic_id"])
            child_sid = str(row["child_overlay_semantic_id"])
            if source_sid not in parents or child_sid not in parents:
                raise ValueError("program references unresolved parent/child state")
            program = EditProgram.from_dict(json.loads(str(row["edit_program_json"])))
            try:
                rebuilt = program.reconstruct(parents[source_sid], route_ids)
            except Exception as exc:
                reconstruction_failures += 1
                raise ValueError(f"program reconstruction failure: {pid}: {exc}") from exc
            if rebuilt.semantic_id() != child_sid or child_sid != program.child_semantic_id or source_sid != program.source_parent_semantic_id:
                reconstruction_failures += 1
                raise ValueError("program reconstruction child/source identity mismatch")
            if canonical_json(rebuilt.canonical()) != canonical_json(json.loads(str(row["child_genome_json"]))):
                reconstruction_failures += 1
                raise ValueError("recorded child genome differs from reconstructed child")
            if canonical_json(parents[source_sid].canonical()) != canonical_json(json.loads(str(row["parent_genome_json"]))):
                raise ValueError("recorded parent genome differs from resolved parent")
            all_programs[pid] = row
            expected_labels[pid] = int(row["label_count"])
            for atom in program.atoms:
                operation_counts[atom.operation] += 1
                target_counts[atom.target_kind] += 1
            action_groups[str(row["action_skeleton_group"])] += 1
            child_groups[child_sid] += 1

    observed_labels: Counter[str] = Counter()
    total_labels = 0
    for gen in generation_rows:
        lpath = root / str(gen["label_shard"])
        last_key: tuple[str, str] | None = None
        for batch in pq.ParquetFile(lpath).iter_batches(batch_size=65536):
            d = batch.to_pydict()
            pids = [str(x) for x in d["program_id"]]
            cids = [str(x) for x in d["case_id"]]
            for pid, cid in zip(pids, cids):
                key = (pid, cid)
                if last_key is not None and key <= last_key:
                    raise ValueError("duplicate or unsorted edit-program × case relation")
                last_key = key
                if pid not in all_programs:
                    raise ValueError("label references unknown exact program")
                if cid not in case_ids:
                    raise ValueError("label references unknown case")
                observed_labels[pid] += 1
            b = np.asarray(d["baseline_margin"], dtype=np.int64)
            c = np.asarray(d["candidate_margin"], dtype=np.int64)
            delta = np.asarray(d["margin_delta"], dtype=np.int64)
            l2w = np.asarray(d["l2w"], dtype=bool)
            w2l = np.asarray(d["w2l"], dtype=bool)
            trace_equal = np.asarray(d["action_trace_equal"], dtype=bool)
            behavior = np.asarray(d["behavior_changing"], dtype=bool)
            ph = np.asarray(d["parent_action_trace_hash"], dtype=np.uint64)
            ch = np.asarray(d["candidate_action_trace_hash"], dtype=np.uint64)
            if not np.array_equal(delta, c - b):
                raise ValueError("label recomputation mismatch: margin_delta")
            if not np.array_equal(l2w, (b < 0) & (c > 0)):
                raise ValueError("label recomputation mismatch: l2w")
            if not np.array_equal(w2l, (b > 0) & (c <= 0)):
                raise ValueError("label recomputation mismatch: w2l")
            if not np.array_equal(trace_equal, ph == ch):
                raise ValueError("label recomputation mismatch: action_trace_equal")
            if not np.array_equal(behavior, (delta != 0) | (~trace_equal)):
                raise ValueError("label recomputation mismatch: behavior_changing")
            total_labels += len(pids)
    for pid, expected in expected_labels.items():
        if observed_labels[pid] != expected:
            raise ValueError(f"program label participation mismatch: {pid}: {observed_labels[pid]} != {expected}")
    if total_labels != int(manifest["counts"]["labels"]) or len(all_programs) != int(manifest["counts"]["programs"]):
        raise ValueError("manifest global counts disagree with validated corpus")
    exact_duplicate_rate = 0.0
    semantic_child_duplicate_rate = (
        sum(max(0, n - 1) for n in child_groups.values()) / max(1, len(all_programs))
    )
    action_skeleton_duplicate_rate = (
        sum(max(0, n - 1) for n in action_groups.values()) / max(1, len(all_programs))
    )
    return {
        "programs": len(all_programs),
        "labels": total_labels,
        "cases": len(case_ids),
        "parents": len(parents),
        "reconstruction_failures": reconstruction_failures,
        "referential_integrity_failures": 0,
        "exact_program_duplicate_rate": exact_duplicate_rate,
        "child_semantic_duplicate_rate": semantic_child_duplicate_rate,
        "action_skeleton_duplicate_rate": action_skeleton_duplicate_rate,
        "operation_counts": dict(sorted(operation_counts.items())),
        "target_kind_counts": dict(sorted(target_counts.items())),
        "model_feature_columns": list(manifest["model_feature_columns"]),
        "program_model_feature_columns": list(manifest["program_model_feature_columns"]),
    }
