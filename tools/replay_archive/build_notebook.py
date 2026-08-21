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

This notebook has one job: maintain a compact, append-only archive of official Kaggriculture episodes involving the teams that finish each UTC source day in the reconstructed top 10. It never ranks teams from daily replay ordering or final game rewards. Ranking is reconstructed from the attached Meta Kaggle `EpisodeAgents.UpdatedScore` history joined through `Submissions`, `Teams`, and `Episodes`; the notebook fails closed if Meta Kaggle has not caught up to the attached Episodes Index.
"""),
        code(r'''
# Configuration. Kaggle notebook credentials/API access are used implicitly; no explicit authentication call.
from pathlib import Path
from datetime import datetime, timezone, timedelta
import csv, hashlib, json, os, shutil, subprocess

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

COMPETITION = "kaggriculture"
COMPETITION_ID = 147734
INDEX_DATASET = "kaggle/kaggriculture-episodes-index"
META_KAGGLE_DATASET = "kaggle/meta-kaggle"
ARCHIVE_DATASET = "ashok205/kaggriculture-top10-replay-archive"
ARCHIVE_TITLE = "Kaggriculture Daily Top-10 Replay Archive"
TOP_N = 10
META_CSV_CHUNK_ROWS = 500_000
PUBLISH = True

STAGE = Path("/kaggle/working/kaggriculture_top10_replay_archive")
TMP = Path("/kaggle/working/kaggriculture_top10_replay_tmp")
for p in (STAGE, TMP):
    if p.exists(): shutil.rmtree(p)
    p.mkdir(parents=True, exist_ok=True)

print({"index": INDEX_DATASET, "ranking_source": META_KAGGLE_DATASET, "archive": ARCHIVE_DATASET, "top_n": TOP_N, "publish": PUBLISH})
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


def locate_meta_kaggle_input():
    root = Path("/kaggle/input/datasets/kaggle/meta-kaggle")
    required = ["Teams.csv", "Submissions.csv", "Episodes.csv", "EpisodeAgents.csv"]
    missing = [name for name in required if not (root / name).exists()]
    if missing:
        raise RuntimeError(
            "kaggle/meta-kaggle is not attached as a notebook input or is incomplete; "
            f"missing {missing} under {root}"
        )
    return root


def _csv_chunks(path, usecols):
    return pd.read_csv(
        path,
        usecols=usecols,
        chunksize=META_CSV_CHUNK_ROWS,
        low_memory=False,
    )


def _numeric(series):
    return pd.to_numeric(series, errors="coerce")


def extract_meta_kaggle_rating_rows(meta_root, *, min_date, max_date, source_episode_ids):
    """Stream Meta Kaggle and materialize only Kaggriculture rating rows for the requested date range."""
    print(f"Meta Kaggle extraction {min_date}..{max_date}", flush=True)

    team_names = {}
    for chunk in _csv_chunks(meta_root / "Teams.csv", ["Id", "CompetitionId", "TeamName"]):
        comp = _numeric(chunk["CompetitionId"])
        part = chunk.loc[comp.eq(COMPETITION_ID), ["Id", "TeamName"]]
        for row in part.itertuples(index=False):
            if pd.isna(row.Id):
                continue
            team_names[int(row.Id)] = "" if pd.isna(row.TeamName) else str(row.TeamName)
    if not team_names:
        raise RuntimeError(f"Meta Kaggle contains no teams for Kaggriculture competition id {COMPETITION_ID}")
    team_ids = set(team_names)
    print(f"  teams: {len(team_ids)}", flush=True)

    submission_team = {}
    submission_date = {}
    for chunk in _csv_chunks(meta_root / "Submissions.csv", ["Id", "TeamId", "SubmissionDate"]):
        tids = _numeric(chunk["TeamId"])
        mask = tids.isin(team_ids)
        if not mask.any():
            continue
        part = chunk.loc[mask, ["Id", "TeamId", "SubmissionDate"]].copy()
        part["team_id_num"] = tids.loc[mask].astype("int64")
        for row in part.itertuples(index=False):
            if pd.isna(row.Id):
                continue
            sid = int(row.Id)
            submission_team[sid] = int(row.team_id_num)
            submission_date[sid] = "" if pd.isna(row.SubmissionDate) else str(row.SubmissionDate)
    if not submission_team:
        raise RuntimeError("Meta Kaggle contains no Kaggriculture submissions")
    submission_ids = set(submission_team)
    print(f"  submissions: {len(submission_ids)}", flush=True)

    episode_info = {}
    for chunk in _csv_chunks(meta_root / "Episodes.csv", ["Id", "Type", "CompetitionId", "CreateTime", "EndTime"]):
        comp = _numeric(chunk["CompetitionId"])
        ids_numeric = _numeric(chunk["Id"])
        create = chunk["CreateTime"].fillna("").astype(str)
        end = chunk["EndTime"].fillna(create).astype(str)
        dates = end.str.slice(0, 10)
        ranking_range = dates.ge(min_date) & dates.le(max_date)
        source_membership = ids_numeric.isin(source_episode_ids)
        mask = comp.eq(COMPETITION_ID) & (ranking_range | source_membership)
        if not mask.any():
            continue
        part = chunk.loc[mask, ["Id", "Type"]].copy()
        part["create_time_norm"] = create.loc[mask]
        part["end_time_norm"] = end.loc[mask]
        for row in part.itertuples(index=False):
            if pd.isna(row.Id):
                continue
            episode_info[int(row.Id)] = {
                "create_time": str(row.create_time_norm),
                "end_time": str(row.end_time_norm),
                "episode_type": "" if pd.isna(row.Type) else str(row.Type),
            }
    if not episode_info:
        raise RuntimeError(f"Meta Kaggle has no Kaggriculture episodes in {min_date}..{max_date}")
    episode_ids = set(episode_info)
    print(f"  episodes in date range: {len(episode_ids)}", flush=True)

    event_schema = pa.schema([
        ("episode_id", pa.int64()),
        ("create_time", pa.string()),
        ("end_time", pa.string()),
        ("episode_type", pa.string()),
        ("seat", pa.int32()),
        ("reward", pa.float64()),
        ("agent_state", pa.string()),
        ("submission_id", pa.int64()),
        ("submission_date", pa.string()),
        ("team_id", pa.int64()),
        ("team_name", pa.string()),
        ("initial_score", pa.float64()),
        ("updated_score", pa.float64()),
    ])
    event_columns = [f.name for f in event_schema]
    writers = {}
    shard_paths = {}
    source_parts = []
    total_event_rows = 0

    sub_team = submission_team
    sub_date = submission_date
    ep_create = {k: v["create_time"] for k, v in episode_info.items()}
    ep_end = {k: v["end_time"] for k, v in episode_info.items()}
    ep_type = {k: v["episode_type"] for k, v in episode_info.items()}

    try:
        for chunk in _csv_chunks(
            meta_root / "EpisodeAgents.csv",
            ["EpisodeId", "Index", "Reward", "State", "SubmissionId", "InitialScore", "UpdatedScore"],
        ):
            eids = _numeric(chunk["EpisodeId"])
            sids = _numeric(chunk["SubmissionId"])
            mask = eids.isin(episode_ids) & sids.isin(submission_ids)
            if not mask.any():
                continue
            part = chunk.loc[mask].copy()
            part["episode_id"] = eids.loc[mask].astype("int64")
            part["submission_id"] = sids.loc[mask].astype("int64")
            part["seat"] = _numeric(part["Index"]).fillna(0).astype("int32")
            part["reward"] = _numeric(part["Reward"]).astype("float64")
            part["initial_score"] = _numeric(part["InitialScore"]).astype("float64")
            part["updated_score"] = _numeric(part["UpdatedScore"]).astype("float64")
            part = part.loc[part["updated_score"].notna()].copy()
            if part.empty:
                continue
            part["team_id"] = part["submission_id"].map(sub_team).astype("int64")
            part["team_name"] = part["team_id"].map(team_names).fillna("").astype(str)
            part["submission_date"] = part["submission_id"].map(sub_date).fillna("").astype(str)
            part["create_time"] = part["episode_id"].map(ep_create).astype(str)
            part["end_time"] = part["episode_id"].map(ep_end).astype(str)
            part["episode_type"] = part["episode_id"].map(ep_type).fillna("").astype(str)
            part["agent_state"] = part["State"].fillna("").astype(str)
            out = part[event_columns].copy()
            out["event_date"] = out["end_time"].str.slice(0, 10)

            for event_date, group in out.groupby("event_date", sort=False):
                if not (min_date <= str(event_date) <= max_date):
                    continue
                if event_date not in writers:
                    path = TMP / f"meta_rating_events_{event_date}.parquet"
                    shard_paths[event_date] = path
                    writers[event_date] = pq.ParquetWriter(path, event_schema, compression="zstd", compression_level=6)
                table = pa.Table.from_pandas(group[event_columns], schema=event_schema, preserve_index=False, safe=False)
                writers[event_date].write_table(table)
                total_event_rows += len(group)

            selected = out.loc[out["episode_id"].isin(source_episode_ids), event_columns]
            if not selected.empty:
                source_parts.append(selected.copy())
    finally:
        for writer in writers.values():
            writer.close()

    if total_event_rows == 0:
        raise RuntimeError("Meta Kaggle extraction produced zero score-bearing Kaggriculture rating rows")

    if source_parts:
        source_df = pd.concat(source_parts, ignore_index=True)
    else:
        source_df = pd.DataFrame(columns=event_columns)
    found_source_ids = set(map(int, source_df["episode_id"].tolist())) if len(source_df) else set()
    missing_source_ids = sorted(set(map(int, source_episode_ids)) - found_source_ids)
    counts = source_df.groupby("episode_id").size().to_dict() if len(source_df) else {}
    bad_counts = sorted((int(eid), int(count)) for eid, count in counts.items() if int(count) != 2)
    if missing_source_ids or bad_counts:
        raise RuntimeError(
            "Meta Kaggle has not caught up to the attached episodes index; refusing an incomplete daily archive. "
            f"missing episode IDs={missing_source_ids[:10]} bad_agent_counts={bad_counts[:10]}"
        )

    provenance = {
        "dataset": META_KAGGLE_DATASET,
        "competition_id": COMPETITION_ID,
        "min_date": min_date,
        "max_date": max_date,
        "team_count": len(team_ids),
        "submission_count": len(submission_ids),
        "episode_metadata_count": len(episode_ids),
        "rating_event_row_count": int(total_event_rows),
        "source_episode_count": len(source_episode_ids),
        "files": {
            name: {"bytes": int((meta_root / name).stat().st_size), "mtime_ns": int((meta_root / name).stat().st_mtime_ns)}
            for name in ("Teams.csv", "Submissions.csv", "Episodes.csv", "EpisodeAgents.csv")
        },
    }
    return shard_paths, source_df, provenance

def episode_date(ep):
    ts = str(ep.get("end_time") or ep.get("create_time") or "")
    if len(ts) < 10:
        raise RuntimeError(f"episode {ep.get('episode_id')} has no usable timestamp")
    return ts[:10]



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
meta_provenance_path = STAGE / "meta_kaggle_provenance.json"
rating_state = json.loads(rating_state_path.read_text()) if rating_state_path.exists() else {}
processed = json.loads(processed_path.read_text()) if processed_path.exists() else {}
top10_rows = read_parquet_rows(top10_path)
episode_rows = read_parquet_rows(episodes_path)

index_path = locate_index_manifest()
index_rows = csv_rows(index_path)
index_sha256 = sha256_file(index_path)
pending = pending_source_days(index_rows, processed)
print(f"official index: {len(index_rows)} days sha256={index_sha256}")
print("pending dates:", [r["date"] for r in pending])
'''),
        code(r'''
# Download only tiny daily manifest.csv files first. Meta Kaggle supplies team identity and rating history;
# daily replay ordering is never used as a ranking signal.
pending_sources = {}
source_episode_ids = set()
for source_row in pending:
    date = str(source_row["date"]); slug = str(source_row["daily_dataset_slug"])
    daily_manifest_path, daily_rows = download_daily_manifest(slug, date)
    daily_manifest_sha = sha256_file(daily_manifest_path)
    ids = [int(r["episode_id"]) for r in daily_rows]
    if len(ids) != len(set(ids)):
        raise RuntimeError(f"duplicate episode IDs in source daily manifest {date}")
    source_episode_ids.update(ids)
    pending_sources[date] = {
        "source_row": source_row,
        "slug": slug,
        "daily_rows": daily_rows,
        "daily_manifest_sha": daily_manifest_sha,
        "episode_ids": ids,
    }

processed_this_run = []
meta_by_id = {}
event_shards = {}
if pending:
    meta_root = locate_meta_kaggle_input()
    last_processed = max(processed) if processed else ""
    if last_processed:
        event_min_date = (datetime.strptime(last_processed, "%Y-%m-%d").date() + timedelta(days=1)).isoformat()
    else:
        event_min_date = min(str(r["date"]) for r in index_rows)
    max_pending = max(str(r["date"]) for r in pending)
    if event_min_date > min(str(r["date"]) for r in pending):
        raise RuntimeError("archive has a non-contiguous processed-date state; refusing to reconstruct ranking from a gap")

    event_shards, source_rating_rows, meta_provenance = extract_meta_kaggle_rating_rows(
        meta_root,
        min_date=event_min_date,
        max_date=max_pending,
        source_episode_ids=source_episode_ids,
    )
    atomic_write_json(meta_provenance_path, meta_provenance)
    source_metadata = rating_rows_to_episode_metadata(source_rating_rows.to_dict("records"))
    meta_by_id = {int(e["episode_id"]): e for e in source_metadata}
    missing = sorted(source_episode_ids - set(meta_by_id))
    if missing:
        raise RuntimeError(f"Meta Kaggle source metadata missing indexed episode IDs: {missing[:10]}")

# Apply every Meta Kaggle UpdatedScore event through each UTC source-day boundary.
event_dates = sorted(event_shards)
event_cursor = 0
for source_row in pending:
    date = str(source_row["date"]); src = pending_sources[date]; slug = src["slug"]
    print(f"\n=== {date} {slug} ===", flush=True)
    while event_cursor < len(event_dates) and event_dates[event_cursor] <= date:
        event_date = event_dates[event_cursor]
        day_events = pd.read_parquet(event_shards[event_date])
        apply_rating_rows(rating_state, day_events.to_dict("records"))
        print(f"  applied {len(day_events)} rating rows from {event_date}", flush=True)
        event_cursor += 1

    top = daily_top_teams(rating_state, TOP_N)
    if len(top) != TOP_N:
        raise RuntimeError(f"only {len(top)} ranked teams available for {date}; refusing partial top-10")
    top_ids = {int(r["team_id"]) for r in top}
    rank_by_team = {int(r["team_id"]): int(r["rank"]) for r in top}
    metadata = [meta_by_id[eid] for eid in src["episode_ids"]]
    daily_meta_by_id = {int(e["episode_id"]): e for e in metadata}
    selected_ids = select_episode_ids(metadata, top_ids)
    print("top10:", [(r["rank"], r["team_name"], round(r["score"], 1)) for r in top])
    print(f"selected {len(selected_ids)}/{len(metadata)} source episodes involving those teams")

    shard, replay_hashes = write_replay_shard(date, selected_ids)
    source_manifest_by_id = {int(r["episode_id"]): r for r in src["daily_rows"]}
    incoming_episode_rows = []
    for episode_id in selected_ids:
        ep = daily_meta_by_id[episode_id]; agents = ep["agents"]
        for agent in agents:
            tid = int(agent["team_id"])
            if tid not in top_ids:
                continue
            opponents = [a for a in agents if int(a["team_id"]) != tid or int(a["submission_id"]) != int(agent["submission_id"])]
            opp = opponents[0] if opponents else {}; sm = source_manifest_by_id[episode_id]
            incoming_episode_rows.append({
                "date": date,
                "episode_id": episode_id,
                "team_id": tid,
                "team_name": agent.get("team_name", ""),
                "daily_rank": rank_by_team[tid],
                "submission_id": int(agent["submission_id"]),
                "seat": int(agent.get("index", 0)),
                "opponent_team_id": int(opp["team_id"]) if opp else None,
                "opponent": opp.get("team_name", "") if opp else "",
                "source_daily_dataset_slug": slug,
                "source_index_row_sha256": source_row_digest(source_row),
                "source_daily_manifest_sha256": src["daily_manifest_sha"],
                "source_avg_score": float(sm["avg_score"]),
                "source_min_score": float(sm["min_score"]),
                "source_sum_score": float(sm["sum_score"]),
                "replay_sha256": replay_hashes[episode_id],
                "replay_shard": shard.name,
            })

    incoming_top = [dict(
        r,
        date=date,
        ranking_basis="Meta Kaggle EpisodeAgents.UpdatedScore at UTC day boundary; best of newest two Submissions.SubmissionDate",
    ) for r in top]
    top10_rows = top10_rows + incoming_top
    episode_rows = episode_rows + incoming_episode_rows
    top_keys = [(r["date"], int(r["rank"])) for r in top10_rows]
    if len(top_keys) != len(set(top_keys)):
        raise RuntimeError("duplicate date/rank in top10 history")
    ep_keys = [(r["date"], int(r["episode_id"]), int(r["team_id"])) for r in episode_rows]
    if len(ep_keys) != len(set(ep_keys)):
        raise RuntimeError("duplicate episode/team metadata row")

    processed[date] = {
        "status": "complete",
        "source_row_sha256": source_row_digest(source_row),
        "daily_dataset_slug": slug,
        "daily_manifest_sha256": src["daily_manifest_sha"],
        "daily_manifest_episode_count": len(src["daily_rows"]),
        "selected_episode_count": len(selected_ids),
        "ranking_team_count": len(rating_state),
        "ranking_source": META_KAGGLE_DATASET,
        "replay_shard": shard.name,
        "replay_shard_sha256": sha256_file(shard),
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
manifest["ranking_source"] = META_KAGGLE_DATASET
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
    "description": "Maintained from kaggle/kaggriculture-episodes-index with historical leaderboard reconstruction from kaggle/meta-kaggle EpisodeAgents.UpdatedScore; replay ordering is never used as rank.",
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
        "dataset_sources": ["kaggle/kaggriculture-episodes-index", "kaggle/meta-kaggle", "ashok205/kaggriculture-top10-replay-archive"],
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
