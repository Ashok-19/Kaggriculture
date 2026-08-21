# Kaggriculture Daily Top-10 Replay Archive

This subsystem only maintains replay data. It does not train BonsaiEditFormer/rankers and does not generate or optimize BKG4.

## Ranking semantics

The notebook never treats the daily replay dataset as a player ranking. The official daily dataset is used only as the source episode universe via its small `manifest.csv`.

Historical leaderboard state is reconstructed from the attached `kaggle/meta-kaggle` dataset:

- `Teams.csv` identifies Kaggriculture teams.
- `Submissions.csv` maps submissions to teams and supplies `SubmissionDate`.
- `Episodes.csv` supplies episode timestamps.
- `EpisodeAgents.csv` supplies `InitialScore` / `UpdatedScore` for each acting submission.

The notebook streams those CSVs in bounded pandas chunks and retains only Kaggriculture rows in the requested unprocessed date range. At each UTC source-day boundary it applies `UpdatedScore` events chronologically, takes the better score among each team's two newest submissions, and ranks teams by score descending with stable team-ID tie breaking. Daily replay ordering, daily `avg_score`, game reward, and final money are never ranking inputs.

If Meta Kaggle has not yet caught up to every episode ID in a newly published Episodes Index day, the run fails closed and publishes nothing. The next scheduled run retries every still-unprocessed date.

## Required Kaggle notebook inputs

Keep these datasets attached:

- `kaggle/kaggriculture-episodes-index` — source-day trigger and replay episode universe.
- `kaggle/meta-kaggle` — historical rating/team/submission metadata.
- `ashok205/kaggriculture-top10-replay-archive` — previous compact archive version.

The competition source `kaggriculture` is also attached for replay downloads.

Exact mounted index path used by the notebook:

```text
/kaggle/input/datasets/kaggle/kaggriculture-episodes-index/manifest.csv
```

Meta Kaggle is read from:

```text
/kaggle/input/datasets/kaggle/meta-kaggle/
```

No explicit authentication call is used inside the notebook.

## Archive files

The maintained dataset is flat:

- `archive_manifest.json` — schema/counts, processed dates, per-file SHA-256/size, source-index hash, generation timestamp, ranking source.
- `rating_state.json` — latest per-team/per-submission rating state through the latest processed UTC day.
- `meta_kaggle_provenance.json` — Meta Kaggle file sizes/mtimes plus filtered team/submission/episode/rating-row counts for the current extraction.
- `processed_days.json` — source-row digest, daily-manifest digest, source slug, counts, ranking source, replay-shard hash, completion status.
- `top10_history.parquet` — one row per date/rank with team ID/name, selected submission, reconstructed score, ranking basis.
- `episodes.parquet` — one row per archived `(date, episode_id, top10_team_id)` participation with rank, opponent, source provenance, replay SHA-256 and shard.
- `replays_YYYY-MM-DD.parquet` — selected replay JSON compressed with Parquet Zstd level 10.

Full Meta Kaggle CSVs are never copied into `/kaggle/working`; only compact temporary per-day Kaggriculture rating shards are materialized during a run and discarded with the notebook session.

## First run

Create the private persistent dataset once:

```bash
kaggle datasets create -p tools/replay_archive/dataset_seed -q -t -r skip
kaggle datasets status ashok205/kaggriculture-top10-replay-archive
```

Generate and verify locally:

```bash
python3 tools/replay_archive/build_notebook.py
python3 tools/replay_archive/build_notebook.py --check
python3 -m unittest tools.replay_archive.test_top10_archive tools.replay_archive.test_build_notebook -v
```

If using CLI notebook push:

```bash
kaggle kernels push -p notebooks/kaggriculture_top10_replay_archive -t 43200
kaggle kernels status ashok205/kaggriculture-top10-replay-archive-maintainer
```

If uploading the `.ipynb` manually, add both `Kaggriculture Episodes Index` and `Meta Kaggle` as notebook inputs, plus the custom archive dataset.

## Incremental behavior

On later runs the notebook validates the previous archive and compares the attached official index against `processed_days.json`:

- completed identical source row: skip;
- completed source row changed: abort;
- unseen date: process chronologically.

For pending days it downloads only each day's small `manifest.csv`, streams Meta Kaggle for the required date range, reconstructs daily top-10 state, and then downloads full replay JSON only for source episodes containing at least one of those ten teams.

A day is marked complete only after ranking, selected replay downloads, hashes, Parquet shard verification, and atomic state/metadata writes succeed. Dataset publication occurs only after all pending days in that run succeed and the staged corpus validates.

## Scheduling

Keep `kaggle/kaggriculture-episodes-index` attached and choose **Schedule a notebook to run → Data updates (once daily)**. The Episodes Index remains the scheduling trigger. Meta Kaggle is a required ranking input but does not need to be the trigger.

Useful checks:

```bash
kaggle kernels status ashok205/kaggriculture-top10-replay-archive-maintainer
kaggle kernels files ashok205/kaggriculture-top10-replay-archive-maintainer
kaggle datasets status ashok205/kaggriculture-top10-replay-archive
kaggle datasets files ashok205/kaggriculture-top10-replay-archive
```
