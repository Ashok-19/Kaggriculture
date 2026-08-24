from __future__ import annotations

import gzip
import hashlib
import json
import os
import shutil
import tarfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from tools.bkg4_search.bonsai_editformer.invariant_data import (
    invariant_corpus_arrays,
    module_from_model_input,
)
from tools.bkg4_search.invariant_schema import validate_specialist_corpus
from tools.bkg4_search.invariant_search import structural_skeleton


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_extract_tar(archive: str | Path, destination: str | Path) -> Path:
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with tarfile.open(archive, "r") as bundle:
        for member in bundle.getmembers():
            target = (destination / member.name).resolve()
            if target != root and root not in target.parents:
                raise ValueError(f"unsafe archive path: {member.name}")
            if not member.isfile():
                raise ValueError(f"unsafe non-file archive member: {member.name}")
        bundle.extractall(destination)
    return destination


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _counter(values: Iterable[str]) -> dict[str, int]:
    return dict(sorted(Counter(values).items()))


def index_row(record: Mapping[str, Any], row_position: int = 0) -> dict[str, Any]:
    module = module_from_model_input(record["model_input"])
    semantic_id = module.semantic_id()
    if semantic_id != str(record["semantic_id"]):
        raise ValueError("record semantic ID does not match canonical model input")
    skeleton = repr(structural_skeleton(module))
    labels = record["target_labels"]
    lineage = labels["lineage"]
    rules = module.rules
    mechanisms = sorted({str(rule.sketch.effect_class) for rule in rules})
    action_families = sorted({str(rule.action.kind) for rule in rules})
    control_modes = sorted({str(rule.sketch.control_pattern) for rule in rules})
    qd_niche = repr((
        tuple(mechanisms), tuple(action_families), tuple(control_modes),
        str(labels.get("risk_profile", "unknown")),
        str(labels.get("support_breadth", "unknown")), int(lineage["depth"]),
    ))
    return {
        "semantic_id": semantic_id,
        "ast_sha256": hashlib.sha256(_json_bytes(record["canonical_ast"])).hexdigest(),
        "skeleton": skeleton,
        "skeleton_id": hashlib.sha256(skeleton.encode("utf-8")).hexdigest(),
        "lineage_group": str(lineage["root_semantic_id"]),
        "lineage_id": str(lineage["lineage_id"]),
        "tier": str(labels["specialist_tier"]),
        "source_quality": str(labels["source_quality"]),
        "depth": int(lineage["depth"]),
        "operator": str(lineage["operator"]),
        "source_goal": lineage.get("source_goal_id"),
        "mechanisms": mechanisms,
        "action_families": action_families,
        "control_modes": control_modes,
        "guard_families": sorted({str(atom.feature).split("_", 1)[0] for rule in rules for atom in (*rule.guard.atoms, *rule.exit_guard.atoms)}),
        "qd_niche": qd_niche,
        "row_position": int(row_position),
    }


def index_rows(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [index_row(record, position) for position, record in enumerate(records)]


@dataclass(frozen=True)
class ScalableSplit:
    train_indices: tuple[int, ...]
    validation_indices: tuple[int, ...]
    train_skeletons: tuple[str, ...]
    validation_skeletons: tuple[str, ...]


def scalable_grouped_split(
    rows: Sequence[Mapping[str, Any]], *, validation_fraction: float = 0.20,
) -> ScalableSplit:
    if not 0.10 <= float(validation_fraction) <= 0.50:
        raise ValueError("validation_fraction must be between 0.10 and 0.50")
    buckets = max(2, round(1.0 / float(validation_fraction)))
    groups: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        groups.setdefault(str(row["skeleton"]), []).append(index)
    train: list[int] = []
    validation: list[int] = []
    train_keys: list[str] = []
    validation_keys: list[str] = []
    for key, indices in sorted(groups.items()):
        held_out = int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) % buckets == 0
        (validation if held_out else train).extend(indices)
        (validation_keys if held_out else train_keys).append(key)
    if not train or not validation:
        raise ValueError("structural split requires at least one train and validation skeleton")
    return ScalableSplit(
        tuple(sorted(train)), tuple(sorted(validation)), tuple(train_keys), tuple(validation_keys),
    )


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_jsonl_gz(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with gzip.open(path, "wt", encoding="utf-8", compresslevel=6) as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")


def write_validated_shard(
    shard: str | Path,
    records: Sequence[Mapping[str, Any]],
    *,
    island: int,
    shard_id: int,
    seed: int,
    screen_panel: Mapping[str, Any],
    label_panel: Mapping[str, Any],
    runtime: Mapping[str, Any],
    counters: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if not records:
        raise ValueError("cannot commit an empty corpus shard")
    shard = Path(shard)
    pending = shard.with_name(shard.name + ".pending")
    if pending.exists():
        shutil.rmtree(pending)
    pending.mkdir(parents=True)
    validation = validate_specialist_corpus(records)
    arrays = invariant_corpus_arrays(records)
    compact = index_rows(records)
    if len({row["semantic_id"] for row in compact}) != len(compact):
        raise ValueError("shard contains duplicate semantic IDs")
    _write_jsonl_gz(pending / "metadata.jsonl.gz", records)
    _write_jsonl_gz(pending / "index.jsonl.gz", compact)
    np.savez_compressed(pending / "training_arrays.npz", **arrays)
    labels = [record["target_labels"] for record in records]
    lineage = [row["lineage"] for row in labels]
    skeleton_counts = Counter(str(row["skeleton_id"]) for row in compact)
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "island": int(island),
        "shard_id": int(shard_id),
        "seed": int(seed),
        "screen_panel": dict(screen_panel),
        "label_panel": dict(label_panel),
        "runtime": dict(runtime),
        "validation": validation,
        "counts": {
            "rows": len(records),
            "semantic_ids": len(compact),
            "skeletons": len({row["skeleton_id"] for row in compact}),
            "qd_niches": len({str(row["qd_niche"]) for row in compact}),
            "tiers": _counter(str(row["specialist_tier"]) for row in labels),
            "l2w": sum(int(row.get("l2w_groups", 0)) > 0 for row in labels),
            "top10_skeleton_concentration": sum(count for _, count in skeleton_counts.most_common(10)) / len(records),
            "depths": _counter(str(row["depth"]) for row in lineage),
            "operators": _counter(str(row["operator"]) for row in lineage),
            "source_goals": _counter(str(row.get("source_goal_id") or "none") for row in lineage),
        },
        "counters": dict(counters or {}),
    }
    files = {}
    for name in ("metadata.jsonl.gz", "index.jsonl.gz", "training_arrays.npz"):
        path = pending / name
        files[name] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    manifest["files"] = files
    _write_json(pending / "shard_manifest.json", manifest)
    if shard.exists():
        raise FileExistsError(shard)
    os.replace(pending, shard)
    return manifest


def _read_gzip_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _canonical_duplicate_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (-int(row.get("label_cases", 0)), str(row["shard_path"]), int(row["row_position"]))


def merge_shards(output_root: str | Path, *, validation_fraction: float = 0.20) -> dict[str, Any]:
    root = Path(output_root)
    shard_manifests = sorted(root.glob("island*/shard-*/shard_manifest.json"))
    if not shard_manifests:
        raise ValueError("no validated shard manifests found")
    unique: dict[str, dict[str, Any]] = {}
    duplicate_log: list[dict[str, Any]] = []
    for manifest_path in shard_manifests:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        shard_rel = manifest_path.parent.relative_to(root).as_posix()
        label_cases = int(manifest["label_panel"].get("cases", 0))
        for row in _read_gzip_jsonl(manifest_path.parent / "index.jsonl.gz"):
            row.update({"shard_path": shard_rel, "label_cases": label_cases, "secondary_evidence": []})
            sid = str(row["semantic_id"])
            previous = unique.get(sid)
            if previous is None:
                unique[sid] = row
                continue
            if previous["ast_sha256"] != row["ast_sha256"]:
                raise ValueError(f"semantic ID collision with contradictory AST: {sid}")
            winner, secondary = sorted((previous, row), key=_canonical_duplicate_key)
            evidence = {
                "shard_path": secondary["shard_path"],
                "row_position": secondary["row_position"],
                "label_cases": secondary["label_cases"],
            }
            winner["secondary_evidence"] = list(winner.get("secondary_evidence", [])) + list(secondary.get("secondary_evidence", [])) + [evidence]
            unique[sid] = winner
            duplicate_log.append({"semantic_id": sid, "canonical": winner["shard_path"], "secondary": evidence})
    rows = sorted(unique.values(), key=lambda row: row["semantic_id"])
    _write_jsonl_gz(root / "global_index.jsonl.gz", rows)
    _write_jsonl_gz(root / "duplicate_log.jsonl.gz", duplicate_log)
    split_report: dict[str, Any] = {"available": False}
    try:
        split = scalable_grouped_split(rows, validation_fraction=validation_fraction)
    except ValueError:
        split = None
    if split is not None:
        split_root = root / "split"
        split_root.mkdir(exist_ok=True)
        with gzip.open(split_root / "train_ids.txt.gz", "wt", encoding="utf-8") as handle:
            handle.writelines(rows[i]["semantic_id"] + "\n" for i in split.train_indices)
        with gzip.open(split_root / "validation_ids.txt.gz", "wt", encoding="utf-8") as handle:
            handle.writelines(rows[i]["semantic_id"] + "\n" for i in split.validation_indices)
        split_report = {
            "available": True,
            "train_rows": len(split.train_indices),
            "validation_rows": len(split.validation_indices),
            "train_skeletons": len(split.train_skeletons),
            "validation_skeletons": len(split.validation_skeletons),
            "skeleton_overlap": len(set(split.train_skeletons) & set(split.validation_skeletons)),
        }
        _write_json(split_root / "split_manifest.json", split_report)
    report = {
        "schema_version": 1,
        "shards": len(shard_manifests),
        "indexed_rows": len(rows) + len(duplicate_log),
        "deduplicated_rows": len(rows),
        "duplicates": len(duplicate_log),
        "skeletons": len({row["skeleton_id"] for row in rows}),
        "tiers": _counter(str(row["tier"]) for row in rows),
        "split": split_report,
    }
    _write_json(root / "summary.json", report)
    return report
