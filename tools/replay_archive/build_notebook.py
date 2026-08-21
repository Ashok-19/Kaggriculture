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

This notebook has one job: maintain a compact append-only archive of official Kaggriculture daily replay data involving the ten strongest teams observed in each source day.

It does **not** use daily file ordering as a player ranking. For each official daily dataset it reads `manifest.csv`, scans only the small replay headers to recover `TeamNames`, ranks teams from that day's official rating evidence, stores full replay bodies only for the selected top-10 teams, then deletes the large temporary ZIP before processing the next day.

The attached `kaggle/kaggriculture-episodes-index` dataset is the authoritative day list and the notebook's **Data updates (once daily)** scheduling trigger.
"""),
        code(r'''
from pathlib import Path
from datetime import datetime, timezone
import csv, hashlib, io, json, os, shutil, subprocess, zipfile

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

INDEX_DATASET = "kaggle/kaggriculture-episodes-index"
ARCHIVE_DATASET = "ashok205/kaggriculture-top10-replay-archive"
ARCHIVE_TITLE = "Kaggriculture Daily Top-10 Replay Archive"
TOP_N = 10
HEADER_SCAN_BYTES = 64 * 1024
PUBLISH = True

STAGE = Path("/kaggle/working/kaggriculture_top10_replay_archive")
TMP = Path("/kaggle/working/kaggriculture_top10_replay_tmp")
for p in (STAGE, TMP):
    if p.exists():
        shutil.rmtree(p)
    p.mkdir(parents=True, exist_ok=True)

print({"index": INDEX_DATASET, "archive": ARCHIVE_DATASET, "top_n": TOP_N, "publish": PUBLISH})
'''),
        code(helper_source),
        code(r'''
def run(cmd, *, timeout=7200):
    print("$", " ".join(map(str, cmd)), flush=True)
    p = subprocess.run([str(x) for x in cmd], text=True, capture_output=True, timeout=timeout)
    if p.returncode != 0:
        raise RuntimeError(f"command failed ({p.returncode}): {' '.join(map(str, cmd))}\n{p.stdout}\n{p.stderr}")
    if p.stdout.strip():
        print(p.stdout.strip())
    if p.stderr.strip():
        print(p.stderr.strip())
    return p


def locate_index_manifest():
    path = Path("/kaggle/input/datasets/kaggle/kaggriculture-episodes-index/manifest.csv")
    if not path.exists():
        raise RuntimeError(
            "kaggle/kaggriculture-episodes-index is not attached as a notebook input; "
            f"expected {path}"
        )
    return path


def locate_archive_input():
    candidates = [
        Path("/kaggle/input/datasets/ashok205/kaggriculture-top10-replay-archive"),
        Path("/kaggle/input/kaggriculture-top10-replay-archive"),
    ]
    for p in candidates:
        if (p / "archive_manifest.json").exists():
            return p
    hits = [p.parent for p in Path("/kaggle/input").rglob("archive_manifest.json") if "top10" in str(p).lower()]
    if len(hits) > 1:
        raise RuntimeError(f"ambiguous prior archive mounts: {hits}")
    return hits[0] if hits else None


def csv_rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        return [dict(r) for r in csv.DictReader(f)]


def csv_rows_from_bytes(raw):
    text = raw.decode("utf-8-sig")
    return [dict(r) for r in csv.DictReader(io.StringIO(text))]


def read_parquet_rows(path):
    if not Path(path).exists():
        return []
    return pd.read_parquet(path).to_dict("records")


def write_rows_parquet_atomic(path, rows, sort_cols):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    df = pd.DataFrame(rows)
    if len(df):
        df = df.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)
    df.to_parquet(tmp, index=False, compression="zstd")
    os.replace(tmp, path)


def download_daily_zip(slug, date):
    day_tmp = TMP / f"day_{date}"
    if day_tmp.exists():
        shutil.rmtree(day_tmp)
    day_tmp.mkdir(parents=True)
    run(["kaggle", "datasets", "download", f"kaggle/{slug}", "-p", day_tmp, "-o", "-q"])
    zips = list(day_tmp.glob("*.zip"))
    if len(zips) != 1:
        raise RuntimeError(f"expected one daily ZIP for {date}, found {zips}")
    if zips[0].stat().st_size <= 0:
        raise RuntimeError(f"empty daily ZIP for {date}")
    return day_tmp, zips[0]


def replay_entry_map(zf):
    out = {}
    for info in zf.infolist():
        name = Path(info.filename).name
        if not name.endswith(".json"):
            continue
        stem = name[:-5]
        if stem.isdigit():
            out[int(stem)] = info
    return out


def scan_team_names(zf, manifest_rows):
    entries = replay_entry_map(zf)
    names_by_episode = {}
    for i, row in enumerate(manifest_rows, 1):
        episode_id = int(row["episode_id"])
        info = entries.get(episode_id)
        if info is None:
            raise RuntimeError(f"daily ZIP missing replay JSON for episode {episode_id}")
        with zf.open(info) as f:
            prefix = f.read(HEADER_SCAN_BYTES)
        names_by_episode[episode_id] = extract_team_names_from_prefix(prefix)
        if i % 200 == 0:
            print(f"  headers {i}/{len(manifest_rows)}", flush=True)
    return entries, names_by_episode


def write_replay_shard_from_zip(date, zf, entries, selected_ids):
    final = STAGE / f"replays_{date}.parquet"
    if final.exists():
        raise RuntimeError(f"day shard already exists for unprocessed day: {final.name}")
    tmp = TMP / f"replays_{date}.parquet"
    schema = pa.schema([("episode_id", pa.int64()), ("replay_json", pa.large_string())])
    writer = pq.ParquetWriter(tmp, schema, compression="zstd", compression_level=10)
    replay_hashes = {}
    ordered = sorted(set(map(int, selected_ids)))
    try:
        for i, episode_id in enumerate(ordered, 1):
            info = entries.get(episode_id)
            if info is None:
                raise RuntimeError(f"selected replay {episode_id} missing from ZIP")
            raw = zf.read(info)
            replay_hashes[episode_id] = hashlib.sha256(raw).hexdigest()
            writer.write_table(pa.Table.from_arrays([
                pa.array([episode_id], type=pa.int64()),
                pa.array([raw.decode("utf-8")], type=pa.large_string()),
            ], schema=schema))
            if i % 25 == 0:
                print(f"  stored replays {i}/{len(ordered)}", flush=True)
    finally:
        writer.close()
    os.replace(tmp, final)
    check = pq.read_table(final, columns=["episode_id"]).column("episode_id").to_pylist()
    if check != ordered:
        raise RuntimeError(f"day shard verification failed for {date}")
    return final, replay_hashes
'''),
        code(r'''
# Tiny deterministic smoke test before touching Kaggle sources.
_smoke_rows = [
    {"episode_id": "3", "avg_score": "1000", "min_score": "900"},
    {"episode_id": "1", "avg_score": "1300", "min_score": "1250"},
    {"episode_id": "2", "avg_score": "1200", "min_score": "1180"},
]
_smoke_names = {1: ("A", "B"), 2: ("C", "D"), 3: ("E", "F")}
_smoke_top = rank_daily_teams(_smoke_rows, _smoke_names, 4)
assert [r["team_name"] for r in _smoke_top] == ["A", "B", "C", "D"]
assert select_episode_ids_by_team_names(_smoke_names, {"A"}) == [1]
assert extract_team_names_from_prefix(b'{"info":{"TeamNames":["A","B"]},"steps":[') == ("A", "B")
print("PURE SMOKE PASSED")
'''),
        code(r'''
# Load the previous published archive if it exists.
prior = locate_archive_input()
if prior is not None:
    prior_manifest = validate_archive(prior)
    for p in prior.iterdir():
        if p.is_file():
            shutil.copy2(p, STAGE / p.name)
    print("validated prior archive", prior, "dates", len(prior_manifest.get("processed_dates", [])))
else:
    print("no prior archive manifest: first-run historical bootstrap")

processed_path = STAGE / "processed_days.json"
top10_path = STAGE / "top10_history.parquet"
episodes_path = STAGE / "episodes.parquet"
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
processed_this_run = []

for source_row in pending:
    date = str(source_row["date"])
    slug = str(source_row["daily_dataset_slug"])
    print(f"\n=== {date} {slug} ===", flush=True)
    day_tmp, zip_path = download_daily_zip(slug, date)
    zip_bytes = zip_path.stat().st_size

    try:
        with zipfile.ZipFile(zip_path) as zf:
            if "manifest.csv" not in zf.namelist():
                raise RuntimeError(f"daily ZIP {date} has no manifest.csv")
            daily_manifest_raw = zf.read("manifest.csv")
            daily_manifest_sha = sha256_bytes(daily_manifest_raw)
            daily_rows = csv_rows_from_bytes(daily_manifest_raw)
            expected_count = int(source_row["episode_count"])
            if len(daily_rows) != expected_count:
                raise RuntimeError(f"daily manifest count mismatch for {date}: index={expected_count} zip={len(daily_rows)}")
            ids = [int(r["episode_id"]) for r in daily_rows]
            if len(ids) != len(set(ids)):
                raise RuntimeError(f"duplicate episode IDs in source daily manifest {date}")

            entries, names_by_episode = scan_team_names(zf, daily_rows)
            top = rank_daily_teams(daily_rows, names_by_episode, TOP_N)
            if len(top) != TOP_N:
                raise RuntimeError(f"only {len(top)} teams found in {date}; refusing partial top-10")
            rank_by_name = {str(r["team_name"]): int(r["rank"]) for r in top}
            top_by_name = {str(r["team_name"]): r for r in top}
            top_names = set(rank_by_name)
            selected_ids = select_episode_ids_by_team_names(names_by_episode, top_names)
            if not selected_ids:
                raise RuntimeError(f"no top-10 episodes selected for {date}")

            print("top10:", [(r["rank"], r["team_name"], round(float(r["score_proxy"]), 1)) for r in top])
            print(f"selected {len(selected_ids)}/{len(daily_rows)} source episodes")

            shard, replay_hashes = write_replay_shard_from_zip(date, zf, entries, selected_ids)
            manifest_by_id = {int(r["episode_id"]): r for r in daily_rows}
            incoming_episode_rows = []
            for episode_id in selected_ids:
                participants = tuple(names_by_episode[episode_id])
                matched = list(dict.fromkeys(name for name in participants if name in top_names))
                sm = manifest_by_id[episode_id]
                avg_score = float(sm["avg_score"])
                min_score = float(sm["min_score"])
                for team_name in matched:
                    opponents = [name for name in participants if name != team_name]
                    evidence = top_by_name[team_name]
                    incoming_episode_rows.append({
                        "date": date,
                        "episode_id": episode_id,
                        "team_name": team_name,
                        "daily_rank": rank_by_name[team_name],
                        "daily_score_proxy": float(evidence["score_proxy"]),
                        "opponent": " | ".join(opponents),
                        "participants_json": json.dumps(list(participants), ensure_ascii=False, separators=(",", ":")),
                        "source_daily_dataset_slug": slug,
                        "source_index_row_sha256": source_row_digest(source_row),
                        "source_daily_manifest_sha256": daily_manifest_sha,
                        "source_avg_score": avg_score,
                        "source_min_score": min_score,
                        "source_agent_score_upper": 2.0 * avg_score - min_score,
                        "source_sum_score": float(sm["sum_score"]) if sm.get("sum_score") not in (None, "") else None,
                        "replay_sha256": replay_hashes[episode_id],
                        "replay_shard": shard.name,
                    })

            incoming_top = [dict(r, date=date) for r in top]
            top10_rows.extend(incoming_top)
            episode_rows.extend(incoming_episode_rows)

            top_keys = [(str(r["date"]), int(r["rank"])) for r in top10_rows]
            if len(top_keys) != len(set(top_keys)):
                raise RuntimeError("duplicate date/rank in top10 history")
            ep_keys = [(str(r["date"]), int(r["episode_id"]), str(r["team_name"])) for r in episode_rows]
            if len(ep_keys) != len(set(ep_keys)):
                raise RuntimeError("duplicate date/episode/team metadata row")

            processed[date] = {
                "status": "complete",
                "source_row_sha256": source_row_digest(source_row),
                "daily_dataset_slug": slug,
                "daily_manifest_sha256": daily_manifest_sha,
                "daily_manifest_episode_count": len(daily_rows),
                "selected_episode_count": len(selected_ids),
                "ranking_basis": "official daily replay score evidence: max observed avg_score by team",
                "temporary_zip_bytes": int(zip_bytes),
                "replay_shard": shard.name,
                "replay_shard_sha256": sha256_file(shard),
            }

        write_rows_parquet_atomic(top10_path, top10_rows, ["date", "rank"])
        write_rows_parquet_atomic(episodes_path, episode_rows, ["date", "episode_id", "team_name"])
        atomic_write_json(processed_path, processed)
        processed_this_run.append(date)
    finally:
        shutil.rmtree(day_tmp, ignore_errors=True)

metadata = {
    "title": ARCHIVE_TITLE,
    "id": ARCHIVE_DATASET,
    "licenses": [{"name": "CC0-1.0"}],
    "subtitle": "Compact daily Kaggriculture top-10 replay corpus with deterministic provenance",
    "description": "Maintained from the official Kaggriculture Episodes Index and daily replay datasets. Player selection is derived from replay TeamNames plus official daily score evidence; daily file order is never used as a player ranking.",
}
(STAGE / "dataset-metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

manifest = write_archive_manifest(
    STAGE,
    processed_dates=sorted(processed),
    episode_count=len({int(r["episode_id"]) for r in episode_rows}),
    top10_row_count=len(top10_rows),
    source_index_manifest_sha256=index_sha256,
)
validate_archive(STAGE)
print("ARCHIVE VALIDATED", {"dates": len(processed), "episodes": manifest["episode_count"], "new_dates": processed_this_run})
'''),
        code(r'''
# Publish only when at least one new source day was committed.
if PUBLISH and processed_this_run:
    msg = f"top10 archive through {max(processed_this_run)}; +{len(processed_this_run)} day(s)"
    run(["kaggle", "datasets", "version", "-p", STAGE, "-m", msg, "-q", "-t", "-r", "skip"])
    print("PUBLISHED", ARCHIVE_DATASET, msg)
elif not processed_this_run:
    print("UP TO DATE: no dataset version created")
else:
    print("PUBLISH=False: validated archive left in", STAGE)
'''),
        code(r'''
print(json.dumps({
    "archive_dataset": ARCHIVE_DATASET,
    "processed_dates_total": len(processed),
    "processed_dates_this_run": processed_this_run,
    "unique_archived_episodes": len({int(r["episode_id"]) for r in episode_rows}),
    "top10_rows": len(top10_rows),
    "archive_manifest_sha256": manifest["manifest_sha256"],
    "working_archive_bytes": sum(p.stat().st_size for p in STAGE.iterdir() if p.is_file()),
}, indent=2))
'''),
    ]
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3"},
        },
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
        "dataset_sources": [
            "kaggle/kaggriculture-episodes-index",
            "ashok205/kaggriculture-top10-replay-archive",
        ],
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
    notebook = render_notebook()
    metadata = render_kernel_metadata()
    _validate_notebook(notebook)
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
