#!/usr/bin/env python3
"""Core helpers for the Kaggriculture daily top-10 replay archive."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping

SCHEMA_VERSION = 1


def _get(mapping: Mapping[str, Any], *names: str, default: Any = None) -> Any:
    for name in names:
        if name in mapping:
            return mapping[name]
    return default


def _require(mapping: Mapping[str, Any], *names: str) -> Any:
    value = _get(mapping, *names, default=None)
    if value is None:
        raise RuntimeError(f"missing required field: {'/'.join(names)}")
    return value


def normalize_episode_payload(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    container: Mapping[str, Any] = payload
    nested = _get(payload, "result", "Result", default=None)
    if isinstance(nested, Mapping):
        container = nested
    raw_episodes = _get(container, "episodes", "Episodes", default=[])
    if not isinstance(raw_episodes, list):
        raise RuntimeError("episode payload must contain an episodes list")
    out: list[dict[str, Any]] = []
    for raw in raw_episodes:
        if not isinstance(raw, Mapping):
            raise RuntimeError("episode entry must be an object")
        agents_raw = _get(raw, "agents", "Agents", default=[])
        if not isinstance(agents_raw, list) or not agents_raw:
            raise RuntimeError("episode entry has no agents")
        agents: list[dict[str, Any]] = []
        for i, a in enumerate(agents_raw):
            if not isinstance(a, Mapping):
                raise RuntimeError("episode agent must be an object")
            updated = _get(a, "updatedScore", "UpdatedScore", "updated_score", default=None)
            if updated is None:
                raise RuntimeError("episode metadata lacks updatedScore; refusing to infer leaderboard rank")
            index = _get(a, "index", "Index", default=i)
            agents.append(
                {
                    "team_id": int(_require(a, "teamId", "TeamId", "team_id")),
                    "team_name": str(_get(a, "teamName", "TeamName", "team_name", default="")),
                    "submission_id": int(_require(a, "submissionId", "SubmissionId", "submission_id")),
                    "initial_score": None if _get(a, "initialScore", "InitialScore", "initial_score", default=None) is None else float(_get(a, "initialScore", "InitialScore", "initial_score")),
                    "updated_score": float(updated),
                    "reward": _get(a, "reward", "Reward", default=None),
                    "index": int(index) if index is not None else i,
                }
            )
        episode_id = int(_require(raw, "id", "Id", "episodeId", "EpisodeId", "episode_id"))
        create_time = str(_get(raw, "createTime", "CreateTime", "create_time", default=""))
        end_time = str(_get(raw, "endTime", "EndTime", "end_time", default=create_time))
        out.append(
            {
                "episode_id": episode_id,
                "create_time": create_time,
                "end_time": end_time,
                "state": _get(raw, "state", "State", default=None),
                "type": _get(raw, "type", "Type", default=None),
                "agents": agents,
            }
        )
    out.sort(key=lambda e: (e["end_time"] or e["create_time"], e["episode_id"]))
    return out


def merge_episode_metadata(*sources: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Merge normalized episode metadata by episode id, rejecting disagreements."""
    by_id: dict[int, dict[str, Any]] = {}
    for source in sources:
        for raw in source:
            row = dict(raw)
            episode_id = int(row["episode_id"])
            previous = by_id.get(episode_id)
            if previous is not None and previous != row:
                raise RuntimeError(f"conflicting episode metadata for episode {episode_id}")
            by_id[episode_id] = row
    return sorted(by_id.values(), key=lambda e: (str(e.get("end_time") or e.get("create_time") or ""), int(e["episode_id"])))


def apply_episode_updates(state: dict[str, Any], episodes: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Apply chronological skill-rating updates in place and return ``state``."""
    for ep in sorted(episodes, key=lambda e: (str(e.get("end_time") or e.get("create_time") or ""), int(e["episode_id"]))):
        ts = str(ep.get("end_time") or ep.get("create_time") or "")
        for agent in ep["agents"]:
            team_id = int(agent["team_id"])
            submission_id = int(agent["submission_id"])
            team_key = str(team_id)
            team = state.setdefault(team_key, {"team_id": team_id, "team_name": agent.get("team_name", ""), "submissions": {}})
            if agent.get("team_name"):
                team["team_name"] = str(agent["team_name"])
            submissions = team.setdefault("submissions", {})
            sub_key = str(submission_id)
            sub = submissions.get(sub_key)
            if sub is None:
                submissions[sub_key] = {
                    "submission_id": submission_id,
                    "first_seen": ts,
                    "last_seen": ts,
                    "score": float(agent["updated_score"]),
                    "last_episode_id": int(ep["episode_id"]),
                }
            else:
                sub["last_seen"] = ts
                sub["score"] = float(agent["updated_score"])
                sub["last_episode_id"] = int(ep["episode_id"])
    return state


def daily_top_teams(state: Mapping[str, Any], n: int = 10) -> list[dict[str, Any]]:
    if n < 1:
        raise ValueError("n must be positive")
    ranked: list[dict[str, Any]] = []
    for team in state.values():
        submissions = list((team.get("submissions") or {}).values())
        if not submissions:
            continue
        newest_two = sorted(submissions, key=lambda s: (str(s.get("first_seen", "")), int(s["submission_id"])), reverse=True)[:2]
        best = max(newest_two, key=lambda s: (float(s["score"]), -int(s["submission_id"])))
        ranked.append({"team_id": int(team["team_id"]), "team_name": str(team.get("team_name", "")), "submission_id": int(best["submission_id"]), "score": float(best["score"])})
    ranked.sort(key=lambda r: (-r["score"], r["team_id"]))
    for rank, row in enumerate(ranked[:n], 1):
        row["rank"] = rank
    return ranked[:n]


def select_episode_ids(episodes: Iterable[Mapping[str, Any]], top_team_ids: set[int]) -> list[int]:
    selected: set[int] = set()
    for ep in episodes:
        if any(int(a["team_id"]) in top_team_ids for a in ep["agents"]):
            selected.add(int(ep["episode_id"]))
    return sorted(selected)


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


def merge_episode_rows(existing: Iterable[Mapping[str, Any]], incoming: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_id: dict[int, dict[str, Any]] = {}
    for source in (existing, incoming):
        for raw in source:
            row = dict(raw)
            episode_id = int(row["episode_id"])
            previous = by_id.get(episode_id)
            if previous is not None and previous != row:
                raise RuntimeError(f"conflicting episode metadata for episode {episode_id}")
            by_id[episode_id] = row
    return [by_id[k] for k in sorted(by_id)]


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


def write_archive_manifest(root: Path, *, processed_dates: list[str], episode_count: int, top10_row_count: int) -> dict[str, Any]:
    files: dict[str, Any] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name.endswith(".tmp") or path.name == "archive_manifest.json":
            continue
        rel = path.relative_to(root).as_posix()
        files[rel] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "source_index": "kaggle/kaggriculture-episodes-index",
        "processed_dates": sorted(processed_dates),
        "episode_count": int(episode_count),
        "top10_row_count": int(top10_row_count),
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
