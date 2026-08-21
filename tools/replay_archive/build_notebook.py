#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "notebooks/kaggriculture_top10_replay_archive"
OUT_NOTEBOOK = OUT_DIR / "kaggriculture_top10_replay_archive.ipynb"
OUT_META = OUT_DIR / "kernel-metadata.json"
HELPER = ROOT / "tools/replay_archive/top10_archive.py"


def _src(text: str) -> list[str]:
    return [line + "\n" for line in text.strip("\n").splitlines()]


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": _src(text)}


def code(text: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": _src(text)}


def render_notebook() -> str:
    helper_source = HELPER.read_text(encoding="utf-8")
    cells = [
        md("""
# Kaggriculture Daily Top-10 Replay Archive Maintainer

This notebook has one job: maintain a compact, append-only archive of official Kaggriculture episodes involving the teams that finish each UTC source day in the reconstructed top 10. It never ranks teams from daily replay ordering or final game rewards. Ranking uses per-agent simulation skill-rating `updatedScore` metadata, and the notebook fails closed if that metadata is unavailable.
"""),
        code(r'''
# Configuration. Kaggle notebook credentials/API access are used implicitly; no explicit authentication call.
from pathlib import Path
from copy import deepcopy
from datetime import datetime, timezone
import csv, hashlib, io, json, os, shutil, subprocess, tempfile, time

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

COMPETITION = "kaggriculture"
INDEX_DATASET = "kaggle/kaggriculture-episodes-index"
ARCHIVE_DATASET = "ashok205/kaggriculture-top10-replay-archive"
ARCHIVE_TITLE = "Kaggriculture Daily Top-10 Replay Archive"
TOP_N = 10
EPISODE_METADATA_BATCH = 40
METADATA_PAUSE_SECONDS = 0.35
TEAM_HISTORY_PAUSE_SECONDS = 0.75
MAX_METADATA_RETRIES = 5
LEADERBOARD_CANDIDATES = 50
PUBLISH = True

STAGE = Path("/kaggle/working/kaggriculture_top10_replay_archive")
TMP = Path("/kaggle/working/kaggriculture_top10_replay_tmp")
for p in (STAGE, TMP):
    if p.exists(): shutil.rmtree(p)
    p.mkdir(parents=True, exist_ok=True)

print({"index": INDEX_DATASET, "archive": ARCHIVE_DATASET, "top_n": TOP_N, "publish": PUBLISH})
'''),
        code(helper_source),
        code(r'''
# Kaggle/runtime I/O. Daily datasets are touched only for manifest.csv; replay JSON is fetched only after top-10 selection.
def run(cmd, *, timeout=3600, check=True):
    print("$", " ".join(map(str, cmd)), flush=True)
    p = subprocess.run([str(x) for x in cmd], text=True, capture_output=True, timeout=timeout)
    if check and p.returncode != 0:
        raise RuntimeError(f"command failed ({p.returncode}): {' '.join(map(str, cmd))}\n{p.stdout}\n{p.stderr}")
    return p


def locate_archive_input():
    candidates = [
        Path("/kaggle/input/kaggriculture-top10-replay-archive"),
        Path("/kaggle/input/datasets/ashok205/kaggriculture-top10-replay-archive"),
    ]
    for p in candidates:
        if (p / "archive_manifest.json").exists(): return p
    hits = [p.parent for p in Path("/kaggle/input").rglob("archive_manifest.json") if "top10" in str(p).lower()]
    if len(hits) > 1:
        raise RuntimeError(f"ambiguous prior archive mounts: {hits}")
    return hits[0] if hits else None


def locate_index_manifest():
    # Required attached input; this exact Kaggle mount path is also what drives the data-update schedule.
    path = Path("/kaggle/input/datasets/kaggle/kaggriculture-episodes-index/manifest.csv")
    if path.exists():
        return path
    raise RuntimeError(
        "kaggle/kaggriculture-episodes-index is not attached as a notebook input; "
        f"expected {path}"
    )


def csv_rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        return [dict(r) for r in csv.DictReader(f)]


def download_daily_manifest(slug, date):
    out = TMP / f"manifest_{date}"
    if out.exists(): shutil.rmtree(out)
    out.mkdir(parents=True)
    run(["kaggle", "datasets", "download", f"kaggle/{slug}", "-f", "manifest.csv", "-p", out, "--unzip", "-o", "-q"])
    path = out / "manifest.csv"
    if not path.exists(): raise RuntimeError(f"daily manifest missing for {date} {slug}")
    return path, csv_rows(path)


def resolve_episode_metadata(episode_ids):
    # Kaggle's simulation EpisodeService carries initialScore/updatedScore. Never fall back to replay ordering.
    from kaggle_environments import api as episode_api
    wanted = [int(x) for x in episode_ids]
    out = []
    for start in range(0, len(wanted), EPISODE_METADATA_BATCH):
        batch = wanted[start:start + EPISODE_METADATA_BATCH]
        last_error = None
        for attempt in range(MAX_METADATA_RETRIES):
            try:
                payload = episode_api.list_episodes(batch)
                rows = normalize_episode_payload(payload)
                got = {int(r["episode_id"]) for r in rows}
                if got != set(batch):
                    raise RuntimeError(f"EpisodeService returned {len(got)}/{len(batch)} requested IDs")
                out.extend(rows)
                last_error = None
                break
            except Exception as e:
                last_error = e
                time.sleep(min(8.0, 0.75 * (2 ** attempt)))
        if last_error is not None:
            raise RuntimeError(f"could not resolve score-bearing episode metadata for IDs {batch[:3]}...: {last_error}")
        time.sleep(METADATA_PAUSE_SECONDS)
    return sorted(out, key=lambda e: (e["end_time"] or e["create_time"], e["episode_id"]))


def resolve_team_history(team_id):
    # Full public episode history is the ranking source. Daily replay ordering is never used for rank.
    from kaggle_environments import api as episode_api
    last_error = None
    for attempt in range(MAX_METADATA_RETRIES):
        try:
            rows = normalize_episode_payload(episode_api.list_episodes_for_team(int(team_id)))
            rows = [e for e in rows if "VALIDATION" not in str(e.get("type") or "").upper()]
            last_error = None
            time.sleep(TEAM_HISTORY_PAUSE_SECONDS)
            return rows
        except Exception as e:
            last_error = e
            time.sleep(min(12.0, 1.0 * (2 ** attempt)))
    raise RuntimeError(f"could not resolve full score history for team {team_id}: {last_error}")


def current_leaderboard_team_ids(limit=LEADERBOARD_CANDIDATES):
    # Current leaderboard is only a discovery guard for new teams, not a historical-rank substitute.
    p = run(["kaggle", "competitions", "leaderboard", COMPETITION, "-v"])
    rows = list(csv.DictReader(io.StringIO(p.stdout)))
    out = []
    for r in rows:
        raw = r.get("teamId") or r.get("TeamId") or r.get("team_id")
        if raw not in (None, ""):
            out.append(int(raw))
        if len(out) >= limit:
            break
    if not out:
        raise RuntimeError("could not read team IDs from Kaggriculture leaderboard")
    return out


def episode_date(ep):
    ts = str(ep.get("end_time") or ep.get("create_time") or "")
    if len(ts) < 10:
        raise RuntimeError(f"episode {ep.get('episode_id')} has no usable timestamp")
    return ts[:10]


def read_rating_history(path):
    path = Path(path)
    if not path.exists(): return []
    df = pd.read_parquet(path, columns=["metadata_json"])
    return [json.loads(x) for x in df["metadata_json"].tolist()]


def write_rating_history(path, episodes):
    rows = [{
        "episode_id": int(e["episode_id"]),
        "end_time": str(e.get("end_time") or e.get("create_time") or ""),
        "metadata_json": json.dumps(e, sort_keys=True, ensure_ascii=False, separators=(",", ":")),
    } for e in episodes]
    write_rows_parquet_atomic(path, rows, ["end_time", "episode_id"])


def download_replay(episode_id, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    run(["kaggle", "competitions", "replay", str(int(episode_id)), "-p", out_dir, "-q"], timeout=600)
    expected = out_dir / f"episode-{int(episode_id)}-replay.json"
    if expected.exists(): return expected
    hits = list(out_dir.glob(f"*{int(episode_id)}*.json"))
    if len(hits) != 1: raise RuntimeError(f"cannot locate downloaded replay for episode {episode_id}: {hits}")
    return hits[0]


def read_parquet_rows(path):
    if not Path(path).exists(): return []
    return pd.read_parquet(path).to_dict("records")


def write_rows_parquet_atomic(path, rows, sort_cols):
    path = Path(path); tmp = path.with_name(path.name + ".tmp")
    df = pd.DataFrame(rows)
    if len(df):
        df = df.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)
    df.to_parquet(tmp, index=False, compression="zstd")
    os.replace(tmp, path)


def write_replay_shard(date, episode_ids):
    final = STAGE / f"replays_{date}.parquet"
    if final.exists(): raise RuntimeError(f"day shard already exists for unprocessed day: {final.name}")
    tmp = TMP / f"replays_{date}.parquet"
    schema = pa.schema([("episode_id", pa.int64()), ("replay_json", pa.large_string())])
    writer = pq.ParquetWriter(tmp, schema, compression="zstd", compression_level=10)
    replay_hashes = {}
    replay_dir = TMP / f"raw_{date}"
    replay_dir.mkdir(parents=True, exist_ok=True)
    try:
        for i, episode_id in enumerate(sorted(set(map(int, episode_ids))), 1):
            p = download_replay(episode_id, replay_dir)
            raw = p.read_bytes()
            obj = json.loads(raw)
            recorded = int((obj.get("info") or {}).get("EpisodeId", episode_id))
            if recorded != episode_id: raise RuntimeError(f"replay EpisodeId mismatch: wanted {episode_id}, got {recorded}")
            replay_hashes[episode_id] = hashlib.sha256(raw).hexdigest()
            txt = raw.decode("utf-8")
            writer.write_table(pa.Table.from_arrays([pa.array([episode_id], type=pa.int64()), pa.array([txt], type=pa.large_string())], schema=schema))
            p.unlink(missing_ok=True)
            if i % 25 == 0: print(f"  replay {i}/{len(episode_ids)}", flush=True)
    finally:
        writer.close()
    os.replace(tmp, final)
    check = pq.read_table(final, columns=["episode_id"]).column("episode_id").to_pylist()
    if check != sorted(set(map(int, episode_ids))): raise RuntimeError(f"day shard verification failed for {date}")
    return final, replay_hashes
'''),
        code(r'''
# Pure idempotency smoke before touching Kaggle sources.
_fixture = {"episodes": [
    {"id": 1, "endTime": "2026-01-01T01:00:00Z", "agents": [
        {"teamId": 1, "teamName": "A", "submissionId": 11, "updatedScore": 1600},
        {"teamId": 2, "teamName": "B", "submissionId": 21, "updatedScore": 1500}]},
    {"id": 2, "endTime": "2026-01-01T02:00:00Z", "agents": [
        {"teamId": 3, "teamName": "C", "submissionId": 31, "updatedScore": 1400},
        {"teamId": 1, "teamName": "A", "submissionId": 11, "updatedScore": 1610}]},
]}
_e = normalize_episode_payload(_fixture); _s = {}; apply_episode_updates(_s, _e)
assert [x["team_id"] for x in daily_top_teams(_s, 2)] == [1, 2]
assert select_episode_ids(_e, {1, 2}) == [1, 2]
_x = [{"episode_id": 1, "date": "2026-01-01", "replay_sha256": "a"}]
assert merge_episode_rows(_x, _x) == _x
print("PURE IDEMPOTENCY SMOKE PASSED")
'''),
        code(r'''
# Load and validate previous compact archive; otherwise initialize first-run state.
prior = locate_archive_input()
if prior is not None:
    prior_manifest = validate_archive(prior)
    for p in prior.iterdir():
        if p.is_file(): shutil.copy2(p, STAGE / p.name)
    print("validated prior archive", prior, "dates", len(prior_manifest.get("processed_dates", [])))
else:
    print("no prior archive_manifest.json: first-run historical bootstrap")

rating_state_path = STAGE / "rating_state.json"
processed_path = STAGE / "processed_days.json"
top10_path = STAGE / "top10_history.parquet"
episodes_path = STAGE / "episodes.parquet"
rating_history_path = STAGE / "rating_episodes.parquet"
rating_state = json.loads(rating_state_path.read_text()) if rating_state_path.exists() else {}
processed = json.loads(processed_path.read_text()) if processed_path.exists() else {}
top10_rows = read_parquet_rows(top10_path)
episode_rows = read_parquet_rows(episodes_path)
rating_history = read_rating_history(rating_history_path)

index_path = locate_index_manifest()
index_rows = csv_rows(index_path)
index_sha256 = sha256_file(index_path)
pending = pending_source_days(index_rows, processed)
print(f"official index: {len(index_rows)} days sha256={index_sha256}")
print("pending dates:", [r["date"] for r in pending])
'''),
        code(r'''
# Pre-resolve small source manifests/episode metadata for all pending dates, then reconstruct rankings
# from full team histories. Source dataset row order is never a ranking input.
pending_sources = {}
candidate_team_ids = {int(k) for k in rating_state.keys()}
for source_row in pending:
    date = str(source_row["date"]); slug = str(source_row["daily_dataset_slug"])
    daily_manifest_path, daily_rows = download_daily_manifest(slug, date)
    daily_manifest_sha = sha256_file(daily_manifest_path)
    ids = [int(r["episode_id"]) for r in daily_rows]
    if len(ids) != len(set(ids)): raise RuntimeError(f"duplicate episode IDs in source daily manifest {date}")
    metadata = resolve_episode_metadata(ids)
    for ep in metadata:
        for a in ep["agents"]: candidate_team_ids.add(int(a["team_id"]))
    pending_sources[date] = {
        "source_row": source_row, "slug": slug, "daily_rows": daily_rows,
        "daily_manifest_sha": daily_manifest_sha, "metadata": metadata,
    }

if pending:
    candidate_team_ids.update(current_leaderboard_team_ids())
    print(f"ranking candidate teams: {len(candidate_team_ids)}")
    for i, team_id in enumerate(sorted(candidate_team_ids), 1):
        rating_history = merge_episode_metadata(rating_history, resolve_team_history(team_id))
        if i % 25 == 0: print(f"  team histories {i}/{len(candidate_team_ids)}", flush=True)
    write_rating_history(rating_history_path, rating_history)
    print(f"full rating-history episodes cached: {len(rating_history)}")

# Apply every full-history public rating update through each UTC day boundary before ranking the top 10.
processed_this_run = []
last_processed = max(processed) if processed else ""
max_pending = max((str(r["date"]) for r in pending), default="")
ranking_events = [e for e in rating_history if (not last_processed or episode_date(e) > last_processed) and (not max_pending or episode_date(e) <= max_pending)]
ranking_events.sort(key=lambda e: (e["end_time"] or e["create_time"], e["episode_id"]))
ranking_cursor = 0

for source_row in pending:
    date = str(source_row["date"]); src = pending_sources[date]; slug = src["slug"]
    print(f"\n=== {date} {slug} ===", flush=True)
    while ranking_cursor < len(ranking_events) and episode_date(ranking_events[ranking_cursor]) <= date:
        apply_episode_updates(rating_state, [ranking_events[ranking_cursor]])
        ranking_cursor += 1

    top = daily_top_teams(rating_state, TOP_N)
    if len(top) != TOP_N: raise RuntimeError(f"only {len(top)} ranked teams available for {date}; refusing partial top-10")
    top_ids = {int(r["team_id"]) for r in top}; rank_by_team = {int(r["team_id"]): int(r["rank"]) for r in top}
    metadata = src["metadata"]; meta_by_id = {int(e["episode_id"]): e for e in metadata}
    selected_ids = select_episode_ids(metadata, top_ids)
    print("top10:", [(r["rank"], r["team_name"], round(r["score"], 1)) for r in top])
    print(f"selected {len(selected_ids)}/{len(metadata)} source episodes involving those teams")

    shard, replay_hashes = write_replay_shard(date, selected_ids)
    source_manifest_by_id = {int(r["episode_id"]): r for r in src["daily_rows"]}
    incoming_episode_rows = []
    for episode_id in selected_ids:
        ep = meta_by_id[episode_id]; agents = ep["agents"]
        for agent in agents:
            tid = int(agent["team_id"])
            if tid not in top_ids: continue
            opponents = [a for a in agents if int(a["team_id"]) != tid or int(a["submission_id"]) != int(agent["submission_id"])]
            opp = opponents[0] if opponents else {}; sm = source_manifest_by_id[episode_id]
            incoming_episode_rows.append({
                "date": date, "episode_id": episode_id, "team_id": tid, "team_name": agent.get("team_name", ""),
                "daily_rank": rank_by_team[tid], "submission_id": int(agent["submission_id"]), "seat": int(agent.get("index", 0)),
                "opponent_team_id": int(opp["team_id"]) if opp else None, "opponent": opp.get("team_name", "") if opp else "",
                "source_daily_dataset_slug": slug, "source_index_row_sha256": source_row_digest(source_row),
                "source_daily_manifest_sha256": src["daily_manifest_sha"], "source_avg_score": float(sm["avg_score"]),
                "source_min_score": float(sm["min_score"]), "source_sum_score": float(sm["sum_score"]),
                "replay_sha256": replay_hashes[episode_id], "replay_shard": shard.name,
            })

    incoming_top = [dict(r, date=date, ranking_basis="full public episode updatedScore history; best of newest two submissions") for r in top]
    top10_rows = top10_rows + incoming_top
    episode_rows = episode_rows + incoming_episode_rows
    top_keys = [(r["date"], int(r["rank"])) for r in top10_rows]
    if len(top_keys) != len(set(top_keys)): raise RuntimeError("duplicate date/rank in top10 history")
    ep_keys = [(r["date"], int(r["episode_id"]), int(r["team_id"])) for r in episode_rows]
    if len(ep_keys) != len(set(ep_keys)): raise RuntimeError("duplicate episode/team metadata row")

    processed[date] = {
        "status": "complete", "source_row_sha256": source_row_digest(source_row), "daily_dataset_slug": slug,
        "daily_manifest_sha256": src["daily_manifest_sha"], "daily_manifest_episode_count": len(src["daily_rows"]),
        "selected_episode_count": len(selected_ids), "ranking_candidate_team_count": len(candidate_team_ids),
        "replay_shard": shard.name, "replay_shard_sha256": sha256_file(shard),
    }
    write_rows_parquet_atomic(top10_path, top10_rows, ["date", "rank"])
    write_rows_parquet_atomic(episodes_path, episode_rows, ["date", "episode_id", "team_id"])
    atomic_write_json(rating_state_path, rating_state)
    atomic_write_json(processed_path, processed)
    processed_this_run.append(date)

# Final corpus manifest after all dates succeeded.
manifest = write_archive_manifest(
    STAGE,
    processed_dates=sorted(processed),
    episode_count=len({int(r["episode_id"]) for r in episode_rows}),
    top10_row_count=len(top10_rows),
)
manifest["source_index_manifest_sha256"] = index_sha256
manifest["ranking_history_episode_count"] = len(rating_history)
manifest["generated_utc"] = datetime.now(timezone.utc).isoformat()
manifest["manifest_sha256"] = hashlib.sha256(_manifest_payload(manifest)).hexdigest()
atomic_write_json(STAGE / "archive_manifest.json", manifest)
validate_archive(STAGE)
print("ARCHIVE VALIDATED", {"dates": len(processed), "episodes": manifest["episode_count"], "new_dates": processed_this_run})
'''),
        code(r'''
# Publish a new version of the SAME private dataset only when new dates were committed.
metadata = {
    "title": ARCHIVE_TITLE,
    "id": ARCHIVE_DATASET,
    "licenses": [{"name": "CC0-1.0"}],
    "subtitle": "Compact daily top-10 Kaggriculture replay corpus with exact provenance",
    "description": "Maintained by a scheduled notebook from kaggle/kaggriculture-episodes-index. Ranking uses simulation skill-rating updatedScore metadata, not replay ordering.",
}
(STAGE / "dataset-metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
if PUBLISH and processed_this_run:
    msg = f"top10 archive through {max(processed_this_run)}; +{len(processed_this_run)} day(s)"
    run(["kaggle", "datasets", "version", "-p", STAGE, "-m", msg, "-q", "-t", "-r", "skip"], timeout=7200)
    print("PUBLISHED", ARCHIVE_DATASET, msg)
elif not processed_this_run:
    print("UP TO DATE: no dataset version created")
else:
    print("PUBLISH=False: validated archive left in", STAGE)
'''),
        code(r'''
# Compact final report.
print(json.dumps({
    "archive_dataset": ARCHIVE_DATASET,
    "processed_dates_total": len(processed),
    "processed_dates_this_run": processed_this_run,
    "unique_archived_episodes": len({int(r["episode_id"]) for r in episode_rows}),
    "top10_rows": len(top10_rows),
    "archive_manifest_sha256": manifest["manifest_sha256"],
    "working_bytes": sum(p.stat().st_size for p in STAGE.iterdir() if p.is_file()),
}, indent=2))
'''),
    ]
    nb = {
        "cells": cells,
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3"}},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    return json.dumps(nb, ensure_ascii=False, sort_keys=True, indent=1) + "\n"


def render_kernel_metadata() -> str:
    meta = {
        "id": "ashok205/kaggriculture-top10-replay-archive-maintainer",
        "title": "Kaggriculture Top10 Replay Archive Maintainer",
        "code_file": "kaggriculture_top10_replay_archive.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": False,
        "enable_tpu": False,
        "enable_internet": True,
        "dataset_sources": ["kaggle/kaggriculture-episodes-index", "ashok205/kaggriculture-top10-replay-archive"],
        "kernel_sources": [],
        "competition_sources": ["kaggriculture"],
        "model_sources": [],
    }
    return json.dumps(meta, ensure_ascii=False, sort_keys=True, indent=2) + "\n"


def _validate_notebook(text: str) -> None:
    nb = json.loads(text)
    if nb.get("nbformat") != 4:
        raise RuntimeError("unexpected notebook format")
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"<cell-{i}>", "exec")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    notebook = render_notebook(); metadata = render_kernel_metadata(); _validate_notebook(notebook)
    if args.check:
        if not OUT_NOTEBOOK.exists() or OUT_NOTEBOOK.read_text(encoding="utf-8") != notebook:
            raise SystemExit(f"generated notebook is stale: {OUT_NOTEBOOK}")
        if not OUT_META.exists() or OUT_META.read_text(encoding="utf-8") != metadata:
            raise SystemExit(f"generated kernel metadata is stale: {OUT_META}")
        print("notebook generation check passed")
        return 0
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_NOTEBOOK.write_text(notebook, encoding="utf-8")
    OUT_META.write_text(metadata, encoding="utf-8")
    print(OUT_NOTEBOOK)
    print(OUT_META)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
