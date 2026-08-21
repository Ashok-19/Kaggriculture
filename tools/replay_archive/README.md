# Kaggriculture Daily Top-10 Replay Archive

This subsystem only maintains replay data. It does not train BonsaiEditFormer/rankers and does not generate or optimize BKG4.

## Ranking semantics

The notebook never treats the daily replay dataset as a player ranking. For the historical bootstrap it scans every official source-day `manifest.csv` only to discover the source episode universe and candidate team IDs. It then resolves each candidate team's **full public episode history** through Kaggle simulation episode metadata and replays each agent's `updatedScore` skill-rating changes chronologically. The current leaderboard top rows are used only as an extra candidate-discovery guard. At each UTC day boundary, a team's reconstructed leaderboard score is the best score among its two newest observed submissions, matching Kaggriculture's two-tracked-submission/best-bot leaderboard rule. Daily replay ordering and daily `avg_score` are never ranking inputs.

If Kaggle's score-bearing simulation EpisodeService response does not contain `updatedScore`, the notebook aborts and publishes nothing. It never substitutes replay row order, `avg_score`, game reward, or final money as player rank.

## Archive files

The maintained Kaggle Dataset is flat to avoid directory-upload ambiguity:

- `archive_manifest.json` — corpus schema/counts, processed dates, per-file SHA-256/size, source-index hash, generation timestamp.
- `rating_state.json` — latest per-team/per-submission rating state through the latest processed UTC day.
- `rating_episodes.parquet` — compact normalized full public episode metadata used to reconstruct/audit rating history; no replay step payloads.
- `processed_days.json` — source-index row digest, daily-manifest digest, source slug, counts, replay-shard hash, completion status for each date.
- `top10_history.parquet` — `date`, `rank`, team ID/name, selected submission ID, reconstructed score, ranking basis.
- `episodes.parquet` — one row per archived `(date, episode_id, top10_team_id)` participation with rank, opponent, source dataset/manifest provenance, source score fields, replay SHA-256, and shard name.
- `replays_YYYY-MM-DD.parquet` — immutable replay JSON rows compressed with Parquet Zstd level 10.

Each replay is stored once per day shard even if both players are top-10. `episodes.parquet` may have two participation rows for such a replay so both daily ranks are preserved.

## First run

Create the private persistent dataset once:

```bash
kaggle datasets create -p tools/replay_archive/dataset_seed -q -t -r skip
kaggle datasets status ashok205/kaggriculture-top10-replay-archive
```

Generate/verify the notebook locally:

```bash
python3 tools/replay_archive/build_notebook.py
python3 tools/replay_archive/build_notebook.py --check
python3 -m unittest tools.replay_archive.test_top10_archive tools.replay_archive.test_build_notebook -v
```

Push the private CPU notebook. `kaggle kernels push` uploads and runs it:

```bash
kaggle kernels push -p notebooks/kaggriculture_top10_replay_archive -t 43200
kaggle kernels status ashok205/kaggriculture-top10-replay-archive-maintainer
```

The notebook metadata already attaches:

- `kaggle/kaggriculture-episodes-index`
- `ashok205/kaggriculture-top10-replay-archive`
- competition source `kaggriculture`

and enables internet while keeping GPU/TPU disabled.

No explicit `api.authenticate()` call exists in the notebook; it relies on Kaggle notebook API access.

## Incremental run

On later runs the latest custom archive dataset is mounted, validated by SHA-256, and copied to `/kaggle/working`. The notebook compares the current official index with `processed_days.json`:

- exact completed source row -> skip;
- changed row for an already completed date -> abort as provenance corruption;
- new date -> process chronologically.

Only each new daily `manifest.csv` is downloaded from the large daily replay dataset. Its episode metadata plus a small current-leaderboard candidate guard determine which team histories need refreshing; rankings are then reconstructed from those full public histories through the UTC boundary. Full replay JSON is downloaded only after that day's top-10 is known and only for source episodes containing at least one of those ten teams.

After all pending days succeed, the complete compact archive is validated and a new version of `ashok205/kaggriculture-top10-replay-archive` is published with:

```text
kaggle datasets version -p /kaggle/working/kaggriculture_top10_replay_archive -m <message> -q -t -r skip
```

The notebook performs this command itself. If there are no new dates it creates no dataset version.

## Failure/recovery behavior

A date is written into `processed_days.json` only after its score metadata is resolved, top-10 is ranked, all selected replays are downloaded and hashed, the Zstd shard is verified, and metadata/state files are atomically replaced. Dataset publication happens only after every pending date in that run succeeds and the whole staged archive validates.

If a run dies mid-day, no new dataset version is published. The next scheduled run starts again from the last successfully published archive version and retries the unfinished date. This makes the published corpus resumable and corruption-safe without retaining temporary raw replay JSON.

## Scheduling

Keep `kaggle/kaggriculture-episodes-index` attached as a notebook input. In Kaggle, open **Schedule a notebook to run** and choose **Data updates (once daily)**; Kaggle will trigger the notebook when that attached index dataset updates. Keep the custom archive dataset attached too so each run sees its latest published state.

Useful checks:

```bash
kaggle kernels status ashok205/kaggriculture-top10-replay-archive-maintainer
kaggle kernels files ashok205/kaggriculture-top10-replay-archive-maintainer
kaggle datasets status ashok205/kaggriculture-top10-replay-archive
kaggle datasets files ashok205/kaggriculture-top10-replay-archive
```
