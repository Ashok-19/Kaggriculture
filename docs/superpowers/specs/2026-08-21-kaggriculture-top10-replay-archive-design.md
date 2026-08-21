# Kaggriculture Daily Top-10 Replay Archive Design

## Goal

Maintain one compact Kaggle Dataset containing official daily replay episodes involving teams that rank in the reconstructed Kaggriculture top 10 for each source day, with deterministic provenance and incremental updates.

## Ranking source

Do **not** infer rank from daily replay ordering, `avg_score`, reward, or final money.

The Episodes Index provides only the daily episode universe. Historical leaderboard state is reconstructed from attached `kaggle/meta-kaggle` tables:

- `Teams.csv`: Kaggriculture team identity.
- `Submissions.csv`: submission→team mapping and `SubmissionDate`.
- `Episodes.csv`: episode timestamps.
- `EpisodeAgents.csv`: per-agent `InitialScore` and `UpdatedScore`.

The notebook streams these CSVs in bounded chunks, filters to competition ID `147734`, and materializes only compact rating events for the unprocessed date range. Rating events are applied chronologically through each UTC source-day boundary. A team's leaderboard score is the better score among its two newest submissions as of that boundary. Teams are ranked by score descending with a stable team-ID tie break.

If Meta Kaggle does not yet contain complete metadata for every episode ID in a pending source-day manifest, the notebook fails closed and publishes nothing.

## Source discovery

The required schedule-trigger input is `kaggle/kaggriculture-episodes-index`, mounted at:

`/kaggle/input/datasets/kaggle/kaggriculture-episodes-index/manifest.csv`

Each daily replay dataset is accessed only for `manifest.csv`; large replay datasets are never bulk extracted.

Meta Kaggle is a second required attached input under:

`/kaggle/input/datasets/kaggle/meta-kaggle/`

## Replay selection

After reconstructing a day's top 10, select every episode ID in that day's official manifest whose Meta Kaggle agent metadata contains at least one top-10 team ID. Deduplicate deterministically by integer episode ID. Only then download full replay JSON for selected IDs.

## Archive layout

- `archive_manifest.json`: schema version, source index, counts, per-file hashes/sizes, processed dates, ranking source, generation timestamp.
- `rating_state.json`: latest per-team/per-submission rating state through the latest processed day.
- `meta_kaggle_provenance.json`: Meta Kaggle source-file sizes/mtimes and filtered extraction counts.
- `processed_days.json`: source-row digest, daily-manifest digest, source slug, selected counts, ranking source, completion state, replay-shard hash.
- `top10_history.parquet`: date/rank, team ID/name, reconstructed score, selected submission, ranking basis.
- `episodes.parquet`: archived participation metadata with opponent/source/replay provenance.
- `replays_YYYY-MM-DD.parquet`: Zstd-compressed selected replay JSON.

Full Meta Kaggle files remain read-only inputs and are never copied into notebook working storage. Temporary filtered rating shards are bounded to Kaggriculture and the pending date interval.

## Incremental update

The notebook validates the latest custom archive, compares index-row digests against `processed_days.json`, and processes unseen dates chronologically. Existing completed identical rows are skipped; mutation of a completed source row aborts.

For an incremental run, `rating_state.json` resumes from the latest completed date and only rating events after that date through the newest pending source date are applied. A date is marked complete only after ranking, replay download, hashing, shard verification, and atomic metadata/state writes succeed.

## Publication and recovery

All prior compact archive files plus new day shards are staged before publishing a new version of the same custom dataset. If any pending date fails, no dataset version is published, so the next run restarts from the last valid archive and retries all pending dates.

No explicit Kaggle authentication call is used inside the notebook.

## Scheduling

Keep the Episodes Index attached and use Kaggle's **Data updates (once daily)** schedule trigger. Meta Kaggle and the custom archive dataset stay attached as required inputs; Episodes Index remains the trigger.

## Non-goals

No BonsaiEditFormer/ranker training, BKG4 editing, policy optimization, CUDA search, or competition submission work occurs here.
