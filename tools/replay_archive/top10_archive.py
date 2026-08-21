#!/usr/bin/env python3
"""Core helpers for the Kaggriculture daily top-10 replay archive."""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Iterable, Mapping

SCHEMA_VERSION = 3
_TEAM_NAMES_KEY_RE = re.compile(rb'"TeamNames"\s*:\s*')


def extract_team_names_from_prefix(prefix: bytes | str) -> tuple[str, ...]:
    """Extract replay info.TeamNames from a small JSON prefix.

    Kaggriculture replay files place TeamNames in the header before the large steps
    array, so reading only the first few KiB is sufficient for participant discovery.
    """
    raw = prefix.encode("utf-8") if isinstance(prefix, str) else bytes(prefix)
    match = _TEAM_NAMES_KEY_RE.search(raw)
    if match is None:
        raise RuntimeError("replay prefix does not contain TeamNames")
    try:
        # Decode from the value start and let the JSON parser determine where the
        # array ends. A regex cannot safely find the closing bracket because `]`
        # is legal inside a quoted team name. Ignore only a potentially truncated
        # UTF-8 code point at the far end of the replay prefix.
        tail = raw[match.end():].decode("utf-8", errors="ignore")
        names, _ = json.JSONDecoder().raw_decode(tail)
    except Exception as exc:
        raise RuntimeError(f"invalid TeamNames header: {exc}") from exc
    if not isinstance(names, list) or len(names) < 2 or any(not isinstance(x, str) or not x for x in names):
        raise RuntimeError(f"invalid TeamNames value: {names!r}")
    return tuple(names)


def replay_entry_map(zf) -> dict[int, Any]:
    """Map numeric replay JSON filenames to ZIP members, keeping the first duplicate."""
    out: dict[int, Any] = {}
    for info in zf.infolist():
        name = Path(info.filename).name
        if not name.endswith(".json"):
            continue
        stem = name[:-5]
        if stem.isdigit():
            out.setdefault(int(stem), info)
    return out


def scan_team_names_from_zip(
    zf,
    manifest_rows: Iterable[Mapping[str, Any]],
    *,
    header_scan_bytes: int = 64 * 1024,
) -> tuple[dict[int, Any], dict[int, tuple[str, ...]], list[dict[str, Any]]]:
    """Scan replay headers, skipping source rows whose replay member is unavailable/bad."""
    entries = replay_entry_map(zf)
    names_by_episode: dict[int, tuple[str, ...]] = {}
    skipped: list[dict[str, Any]] = []
    for row in manifest_rows:
        try:
            episode_id = int(row["episode_id"])
        except Exception:
            continue
        info = entries.get(episode_id)
        if info is None:
            skipped.append({"episode_id": episode_id, "reason": "missing_member"})
            continue
        try:
            with zf.open(info) as f:
                prefix = f.read(int(header_scan_bytes))
        except Exception:
            skipped.append({"episode_id": episode_id, "reason": "unreadable_header"})
            continue
        try:
            names_by_episode[episode_id] = extract_team_names_from_prefix(prefix)
        except Exception:
            skipped.append({"episode_id": episode_id, "reason": "invalid_header"})
    return entries, names_by_episode, skipped


def read_replay_bytes_from_zip(
    zf,
    entries: Mapping[int, Any],
    episode_id: int,
) -> tuple[bytes | None, str | None]:
    """Read a selected replay body without letting one bad ZIP member abort the day."""
    episode_id = int(episode_id)
    info = entries.get(episode_id)
    if info is None:
        return None, "missing_member"
    try:
        raw = zf.read(info)
    except Exception:
        return None, "unreadable_member"
    try:
        raw.decode("utf-8")
    except UnicodeDecodeError:
        return None, "invalid_utf8"
    return raw, None


def rank_daily_teams(
    manifest_rows: Iterable[Mapping[str, Any]],
    team_names_by_episode: Mapping[int, Iterable[str]],
    n: int = 10,
) -> list[dict[str, Any]]:
    """Rank teams from official daily replay score evidence.

    The daily manifest gives the exact pairwise post-episode rating evidence as
    avg_score/min_score.  We do not assume replay ordering is a player ranking.
    Instead, every team is discovered from replay headers and ranked by its strongest
    observed average-agent-rating episode that day.  min_score and the implied upper
    agent score are deterministic tie-break/evidence fields.
    """
    if n < 1:
        raise ValueError("n must be positive")
    stats: dict[str, dict[str, Any]] = {}
    seen_episode_ids: set[int] = set()
    for raw in manifest_rows:
        episode_id = int(raw["episode_id"])
        if episode_id in seen_episode_ids:
            raise RuntimeError(f"duplicate episode id in daily manifest: {episode_id}")
        seen_episode_ids.add(episode_id)
        if episode_id not in team_names_by_episode:
            continue
        try:
            avg_score = float(raw["avg_score"])
            min_score = float(raw["min_score"])
        except (KeyError, TypeError, ValueError):
            continue
        if not math.isfinite(avg_score) or not math.isfinite(min_score):
            continue
        upper_score = 2.0 * avg_score - min_score
        names = tuple(dict.fromkeys(str(x) for x in team_names_by_episode[episode_id] if str(x)))
        if not names:
            continue
        for name in names:
            row = stats.setdefault(
                name,
                {
                    "team_name": name,
                    "appearances": 0,
                    "score_proxy": float("-inf"),
                    "max_min_score": float("-inf"),
                    "max_agent_score_upper": float("-inf"),
                    "evidence_episode_id": None,
                },
            )
            row["appearances"] += 1
            candidate = (avg_score, min_score, upper_score, -episode_id)
            incumbent = (
                float(row["score_proxy"]),
                float(row["max_min_score"]),
                float(row["max_agent_score_upper"]),
                -int(row["evidence_episode_id"]) if row["evidence_episode_id"] is not None else float("-inf"),
            )
            if candidate > incumbent:
                row["score_proxy"] = avg_score
                row["max_min_score"] = min_score
                row["max_agent_score_upper"] = upper_score
                row["evidence_episode_id"] = episode_id

    ranked = sorted(
        stats.values(),
        key=lambda r: (
            -float(r["score_proxy"]),
            -float(r["max_min_score"]),
            -float(r["max_agent_score_upper"]),
            str(r["team_name"]),
        ),
    )
    out: list[dict[str, Any]] = []
    for rank, row in enumerate(ranked[:n], 1):
        item = dict(row)
        item["rank"] = rank
        item["ranking_basis"] = "official daily replay score evidence: max observed avg_score by team"
        out.append(item)
    return out


def select_episode_ids_by_team_names(
    team_names_by_episode: Mapping[int, Iterable[str]],
    selected_team_names: set[str],
) -> list[int]:
    selected: list[int] = []
    for episode_id, names in team_names_by_episode.items():
        if selected_team_names.intersection(str(x) for x in names):
            selected.append(int(episode_id))
    return sorted(set(selected))


def source_row_digest(row: Mapping[str, Any]) -> str:
    canonical = {str(k): "" if v is None else str(v) for k, v in sorted(row.items(), key=lambda kv: str(kv[0]))}
    payload = json.dumps(canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def pending_source_days(index_rows: Iterable[Mapping[str, Any]], processed: Mapping[str, Any]) -> list[dict[str, Any]]:
    pending: list[dict[str, Any]] = []
    for raw in sorted(index_rows, key=lambda r: str(r["date"])):
        row = dict(raw)
        date = str(row["date"])
        digest = source_row_digest(row)
        prev = processed.get(date)
        if prev is None or prev.get("status") != "complete":
            pending.append(row)
            continue
        if str(prev.get("source_row_sha256")) != digest:
            raise RuntimeError(f"source index row for {date} changed after processing")
    return pending


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as f:
        f.write(payload)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _manifest_payload(manifest: Mapping[str, Any]) -> bytes:
    body = {k: v for k, v in manifest.items() if k != "manifest_sha256"}
    return json.dumps(body, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def write_archive_manifest(
    root: Path,
    *,
    processed_dates: list[str],
    episode_count: int,
    top10_row_count: int,
    source_index_manifest_sha256: str,
) -> dict[str, Any]:
    files: dict[str, Any] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name.endswith(".tmp") or path.name in {"archive_manifest.json", "dataset-metadata.json"}:
            continue
        rel = path.relative_to(root).as_posix()
        files[rel] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "source_index": "kaggle/kaggriculture-episodes-index",
        "ranking_source": "official daily replay ZIP headers + manifest score evidence",
        "processed_dates": sorted(processed_dates),
        "episode_count": int(episode_count),
        "top10_row_count": int(top10_row_count),
        "source_index_manifest_sha256": source_index_manifest_sha256,
        "files": files,
    }
    manifest["manifest_sha256"] = hashlib.sha256(_manifest_payload(manifest)).hexdigest()
    atomic_write_json(root / "archive_manifest.json", manifest)
    return manifest


def validate_archive(root: Path) -> dict[str, Any]:
    path = root / "archive_manifest.json"
    if not path.exists():
        raise RuntimeError(f"archive manifest missing: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if int(manifest.get("schema_version", -1)) != SCHEMA_VERSION:
        raise RuntimeError(f"archive schema mismatch: expected {SCHEMA_VERSION}, got {manifest.get('schema_version')}")
    expected_manifest_sha = str(manifest.get("manifest_sha256", ""))
    actual_manifest_sha = hashlib.sha256(_manifest_payload(manifest)).hexdigest()
    if expected_manifest_sha != actual_manifest_sha:
        raise RuntimeError("archive manifest hash mismatch")
    for rel, meta in (manifest.get("files") or {}).items():
        file_path = root / rel
        if not file_path.exists():
            raise RuntimeError(f"archive file missing: {rel}")
        if sha256_file(file_path) != str(meta["sha256"]):
            raise RuntimeError(f"archive file hash mismatch: {rel}")
        if file_path.stat().st_size != int(meta["bytes"]):
            raise RuntimeError(f"archive file size mismatch: {rel}")
    return manifest
