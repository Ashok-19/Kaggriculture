from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

from tools.edit_corpus.corpus import SCHEMA_VERSION, assert_model_payload_safe, canonical_json, derive_case_label, derive_edit_labels

CORPUS_SCHEMA_VERSION = 1
SHARD_SCHEMA_VERSION = 1


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(canonical_json(value) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _write_parquet_atomic(path: Path, rows: Sequence[Mapping[str, Any]], *, sort_cols: Sequence[str] = ()) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([dict(x) for x in rows])
    if len(df) and sort_cols:
        cols = [x for x in sort_cols if x in df.columns]
        if cols:
            df = df.sort_values(cols, kind="mergesort").reset_index(drop=True)
    tmp = path.with_name(path.name + ".tmp")
    df.to_parquet(tmp, index=False, compression="zstd")
    os.replace(tmp, path)


def _jsonable(value: Any) -> Any:
    """Normalize pandas/NumPy/Arrow-loaded values for deterministic JSON comparison."""
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    # pyarrow list columns commonly round-trip through pandas as numpy.ndarray.
    if hasattr(value, "tolist") and value.__class__.__module__.startswith("numpy"):
        return _jsonable(value.tolist())
    if hasattr(value, "item") and value.__class__.__module__.startswith("numpy"):
        return _jsonable(value.item())
    if hasattr(value, "isoformat") and value.__class__.__module__.startswith(("pandas", "datetime")):
        return value.isoformat()
    try:
        missing = pd.isna(value)
        if isinstance(missing, bool) and missing:
            return None
    except (TypeError, ValueError):
        pass
    return value


def _read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [_jsonable(x) for x in pd.read_parquet(path).to_dict("records")]


def initialize_empty_corpus(root: str | Path) -> Path:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    (root / "shards").mkdir(exist_ok=True)
    if not (root / "processed_shards.json").exists():
        _atomic_json(root / "processed_shards.json", {})
    return root


def source_rows_from_archive(archive_root: str | Path) -> list[dict[str, Any]]:
    root = Path(archive_root)
    processed_path = root / "processed_days.json"
    if not processed_path.exists():
        raise RuntimeError(f"replay archive processed_days.json missing: {processed_path}")
    processed = json.loads(processed_path.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for date, meta in sorted(processed.items()):
        if str(meta.get("status")) != "complete":
            continue
        shard = str(meta.get("replay_shard") or f"replays_{date}.parquet")
        source_sha = str(meta.get("replay_shard_sha256") or "")
        if not source_sha:
            raise RuntimeError(f"completed replay archive day {date} lacks replay_shard_sha256")
        if not (root / shard).is_file():
            raise RuntimeError(f"completed replay archive day {date} is missing shard {shard}")
        rows.append({"date": str(date), "replay_shard": shard, "replay_shard_sha256": source_sha})
    return rows


def _processed(root: Path) -> dict[str, Any]:
    path = root / "processed_shards.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _audit_case_rows(cases: Sequence[Mapping[str, Any]]) -> None:
    for row in cases:
        if "model_context_json" not in row:
            raise ValueError("case row is missing model_context_json")
        try:
            payload = json.loads(str(row["model_context_json"]))
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid model_context_json for case {row.get('case_id')}: {exc}") from exc
        if not isinstance(payload, dict):
            raise ValueError("model_context_json must encode an object")
        assert_model_payload_safe(payload)


def commit_day(
    root: str | Path,
    *,
    date: str,
    source_sha: str,
    cases: Sequence[Mapping[str, Any]],
    edits: Sequence[Mapping[str, Any]],
    labels: Sequence[Mapping[str, Any]],
    source_shard: str | None = None,
) -> bool:
    root = initialize_empty_corpus(root)
    date = str(date)
    source_sha = str(source_sha)
    processed = _processed(root)
    prev = processed.get(date)
    if prev is not None and str(prev.get("status")) == "complete":
        if str(prev.get("source_replay_shard_sha256")) != source_sha:
            raise RuntimeError(f"source replay shard for {date} changed after processing")
        return False
    _audit_case_rows(cases)
    for label in labels:
        if "edit_id" not in label or "case_id" not in label:
            raise ValueError("every label row requires edit_id and case_id")
    case_path = root / "shards" / f"cases_{date}.parquet"
    edit_path = root / "shards" / f"edits_{date}.parquet"
    label_path = root / "shards" / f"labels_{date}.parquet"
    if any(p.exists() for p in (case_path, edit_path, label_path)):
        raise RuntimeError(f"uncommitted shard files already exist for {date}")
    _write_parquet_atomic(case_path, cases, sort_cols=("case_id",))
    _write_parquet_atomic(edit_path, edits, sort_cols=("edit_id",))
    _write_parquet_atomic(label_path, labels, sort_cols=("edit_id", "case_id"))
    processed[date] = {
        "status": "complete",
        "source_replay_shard": str(source_shard or f"replays_{date}.parquet"),
        "source_replay_shard_sha256": source_sha,
        "case_rows": len(cases),
        "edit_rows": len(edits),
        "label_rows": len(labels),
        "case_shard": case_path.relative_to(root).as_posix(),
        "edit_shard": edit_path.relative_to(root).as_posix(),
        "label_shard": label_path.relative_to(root).as_posix(),
        "case_shard_sha256": sha256_file(case_path),
        "edit_shard_sha256": sha256_file(edit_path),
        "label_shard_sha256": sha256_file(label_path),
    }
    _atomic_json(root / "processed_shards.json", processed)
    return True


def _merge_shards(root: Path, pattern: str, key_cols: Sequence[str], out_name: str) -> int:
    frames = []
    for path in sorted((root / "shards").glob(pattern)):
        frames.append(pd.read_parquet(path))
    if frames:
        df = pd.concat(frames, ignore_index=True, sort=False)
    else:
        df = pd.DataFrame()
    if len(df) and key_cols:
        missing = [x for x in key_cols if x not in df.columns]
        if missing:
            raise RuntimeError(f"{out_name} shard schema missing keys: {missing}")
        # Exact duplicate rows are harmless; conflicting duplicate identities are not.
        serialized = df.apply(lambda row: canonical_json(_jsonable(row.to_dict())), axis=1)
        temp = df.copy()
        temp["__row_json"] = serialized
        for key, grp in temp.groupby(list(key_cols), dropna=False, sort=False):
            if grp["__row_json"].nunique() > 1:
                raise RuntimeError(f"conflicting duplicate identity in {out_name}: {key}")
        df = df.drop_duplicates(subset=list(key_cols), keep="first")
        df = df.sort_values(list(key_cols), kind="mergesort").reset_index(drop=True)
    tmp = root / (out_name + ".tmp")
    df.to_parquet(tmp, index=False, compression="zstd")
    os.replace(tmp, root / out_name)
    return int(len(df))


def _rebuild_global_targets(root: Path) -> int:
    path = root / "edit_case_labels.parquet"
    labels = pd.read_parquet(path) if path.exists() else pd.DataFrame()
    rows: list[dict[str, Any]] = []
    if len(labels):
        if "edit_id" not in labels.columns:
            raise RuntimeError("edit_case_labels.parquet lacks edit_id")
        for edit_id, grp in labels.groupby("edit_id", sort=True, dropna=False):
            normalized = []
            for raw in grp.to_dict("records"):
                if all(k in raw for k in ("margin_delta", "l2w", "w2l", "action_changes", "activations")):
                    normalized.append(raw)
                    continue
                normalized.append(derive_case_label(
                    int(raw["baseline_margin"]), int(raw["candidate_margin"]),
                    int(raw.get("action_changes", 0)), int(raw.get("activations", 0)),
                    protected=bool(raw.get("protected", True)),
                ))
            rows.append({"edit_id": str(edit_id), **derive_edit_labels(normalized)})
    _write_parquet_atomic(root / "edit_targets.parquet", rows, sort_cols=("edit_id",))
    return len(rows)


def consolidate_corpus(root: str | Path) -> dict[str, int]:
    root = initialize_empty_corpus(root)
    counts = {
        "cases": _merge_shards(root, "cases_*.parquet", ("case_id",), "cases.parquet"),
        "edits": _merge_shards(root, "edits_*.parquet", ("edit_id",), "edits.parquet"),
        "labels": _merge_shards(root, "labels_*.parquet", ("edit_id", "case_id"), "edit_case_labels.parquet"),
    }
    _rebuild_global_targets(root)
    return counts


def _upsert_table(path: Path, incoming: Sequence[Mapping[str, Any]], key: str) -> None:
    rows = _read_rows(path)
    by_id: dict[str, dict[str, Any]] = {str(row[key]): row for row in rows}
    for raw in incoming:
        row = dict(raw)
        rid = str(row[key])
        old = by_id.get(rid)
        if old is not None and canonical_json(old) != canonical_json(row):
            raise RuntimeError(f"conflicting {key} definition: {rid}")
        by_id[rid] = row
    _write_parquet_atomic(path, [by_id[k] for k in sorted(by_id)], sort_cols=(key,))


def upsert_structures(
    root: str | Path,
    base_structures: Sequence[Mapping[str, Any]],
    parents: Sequence[Mapping[str, Any]],
) -> None:
    root = initialize_empty_corpus(root)
    _upsert_table(root / "base_structures.parquet", base_structures, "base_id")
    _upsert_table(root / "parents.parquet", parents, "parent_id")


def _manifest_payload(manifest: Mapping[str, Any]) -> bytes:
    body = {k: v for k, v in manifest.items() if k != "manifest_sha256"}
    return canonical_json(body).encode("utf-8")


def _counts_from_tables(root: Path) -> dict[str, Any]:
    cases = pd.read_parquet(root / "cases.parquet") if (root / "cases.parquet").exists() else pd.DataFrame()
    edits = pd.read_parquet(root / "edits.parquet") if (root / "edits.parquet").exists() else pd.DataFrame()
    labels = pd.read_parquet(root / "edit_case_labels.parquet") if (root / "edit_case_labels.parquet").exists() else pd.DataFrame()
    targets = pd.read_parquet(root / "edit_targets.parquet") if (root / "edit_targets.parquet").exists() else pd.DataFrame()
    op_counts = Counter(str(x) for x in edits["operation"].tolist()) if "operation" in edits.columns else Counter()
    label_counts = {}
    for col in ("l2w_count", "w2l_count", "severe_regression_count"):
        label_counts[col] = int(targets[col].sum()) if col in targets.columns and len(targets) else 0
    for col in ("strict_safe", "exact_noop", "behavior_changing"):
        label_counts[col] = int(targets[col].astype(bool).sum()) if col in targets.columns and len(targets) else 0
    shard_root = root / "shards"
    raw_case_rows = sum(len(pd.read_parquet(p)) for p in sorted(shard_root.glob("cases_*.parquet"))) if shard_root.exists() else 0
    raw_edit_rows = sum(len(pd.read_parquet(p)) for p in sorted(shard_root.glob("edits_*.parquet"))) if shard_root.exists() else 0
    raw_label_rows = sum(len(pd.read_parquet(p)) for p in sorted(shard_root.glob("labels_*.parquet"))) if shard_root.exists() else 0
    return {
        "case_rows": int(len(cases)),
        "edit_rows": int(len(edits)),
        "edit_target_rows": int(len(targets)),
        "edit_case_rows": int(len(labels)),
        "operation_counts": dict(sorted(op_counts.items())),
        "label_counts": label_counts,
        "dedup_counts": {
            "case_rows_removed": int(raw_case_rows - len(cases)),
            "semantic_edit_rows_removed": int(raw_edit_rows - len(edits)),
            "edit_case_rows_removed": int(raw_label_rows - len(labels)),
        },
    }


def write_corpus_manifest(
    root: str | Path,
    *,
    replay_archive_version: str,
    engine_identity: str,
    generator_version: str,
    edit_grammar_version: str = "bonsai_edit_actor_grammar_v1",
) -> dict[str, Any]:
    root = initialize_empty_corpus(root)
    processed = _processed(root)
    files: dict[str, Any] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name.endswith(".tmp") or path.name in {"corpus_manifest.json", "dataset-metadata.json"}:
            continue
        rel = path.relative_to(root).as_posix()
        files[rel] = {"bytes": int(path.stat().st_size), "sha256": sha256_file(path)}
    manifest: dict[str, Any] = {
        "schema_version": CORPUS_SCHEMA_VERSION,
        "row_schema_version": SCHEMA_VERSION,
        "shard_schema_version": SHARD_SCHEMA_VERSION,
        "replay_archive_version": str(replay_archive_version),
        "processed_dates": sorted(date for date, meta in processed.items() if str(meta.get("status")) == "complete"),
        "bkg4_parent_identities": sorted({str(x) for x in pd.read_parquet(root / "parents.parquet")["parent_id"].tolist()}) if (root / "parents.parquet").exists() and "parent_id" in pd.read_parquet(root / "parents.parquet").columns else [],
        "cuda_engine_identity": str(engine_identity),
        "candidate_generator_version": str(generator_version),
        "edit_grammar_version": str(edit_grammar_version),
        "counts": _counts_from_tables(root),
        "files": files,
    }
    manifest["manifest_sha256"] = hashlib.sha256(_manifest_payload(manifest)).hexdigest()
    _atomic_json(root / "corpus_manifest.json", manifest)
    return manifest


def validate_corpus(root: str | Path) -> dict[str, Any]:
    root = Path(root)
    path = root / "corpus_manifest.json"
    if not path.exists():
        raise RuntimeError(f"corpus manifest missing: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if int(manifest.get("schema_version", -1)) != CORPUS_SCHEMA_VERSION:
        raise RuntimeError(f"corpus schema mismatch: {manifest.get('schema_version')}")
    actual = hashlib.sha256(_manifest_payload(manifest)).hexdigest()
    if str(manifest.get("manifest_sha256")) != actual:
        raise RuntimeError("corpus manifest hash mismatch")
    for rel, meta in (manifest.get("files") or {}).items():
        p = root / rel
        if not p.is_file():
            raise RuntimeError(f"corpus file missing: {rel}")
        if int(p.stat().st_size) != int(meta["bytes"]):
            raise RuntimeError(f"corpus file size mismatch: {rel}")
        if sha256_file(p) != str(meta["sha256"]):
            raise RuntimeError(f"corpus file hash mismatch: {rel}")
    return manifest


def copy_prior_corpus(prior_root: str | Path, stage_root: str | Path) -> dict[str, Any]:
    prior = Path(prior_root)
    manifest = validate_corpus(prior)
    stage = Path(stage_root)
    if stage.exists():
        shutil.rmtree(stage)
    shutil.copytree(prior, stage)
    return manifest
