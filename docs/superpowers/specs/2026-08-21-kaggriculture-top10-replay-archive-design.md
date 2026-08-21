# Kaggriculture Daily Top-10 Replay Archive Design

## Goal

Maintain one compact Kaggle Dataset containing all official daily replay episodes involving the teams that finish each UTC source day in the Kaggriculture top 10, with deterministic provenance and incremental updates.

## Ranking source

Do **not** infer player rank from the order of files or rows in the daily replay dataset. The daily replay archive is only the source episode universe.

For each source day, read only its small `manifest.csv` to discover source episode IDs and candidate teams. Resolve candidate teams' **full public episode histories** through Kaggle simulation episode metadata. That metadata exposes `teamId`, `submissionId`, `initialScore`, and `updatedScore`; `updatedScore` is the skill rating after that match. Replay all public rating updates chronologically through each UTC day boundary. Kaggriculture tracks a team's newest two submissions and shows the better-scoring one on the leaderboard, so the reconstructed daily team score is the maximum latest rating among the two newest submissions observed for that team as of the end of that UTC day. Rank by score descending, then stable `team_id` tie-break. Current leaderboard rows are used only to widen candidate discovery, never as historical snapshots.

The archive fails closed if score-bearing episode metadata is unavailable or fewer than ten ranked teams can be established; it never silently substitutes episode order, average episode score, reward, or final money for leaderboard rank.

## Source discovery

The official index is `kaggle/kaggriculture-episodes-index`. Its `manifest.csv` supplies `date`, `daily_dataset_slug`, URL, episode count, bytes, and score-summary provenance. Each daily dataset is accessed only for `manifest.csv`; the ~20 GiB expanded replay dataset is never extracted wholesale.

## Replay selection

After reconstructing the end-of-day top 10, select every episode ID in that day's official daily manifest whose metadata contains at least one of those ten team IDs. Deduplicate by integer episode ID. Download replay JSON only for selected IDs using the Kaggle simulation replay command/API available inside the notebook.

## Archive layout

The custom dataset contains:

- `archive_manifest.json`: schema version, source index, run timestamp, corpus counts, file hashes, processed date range, and provenance.
- `rating_state.json`: latest per-submission score/team/first-seen/last-seen state through the latest processed day.
- `rating_episodes.parquet`: compact normalized full public episode metadata used to reconstruct/audit ranking history.
- `processed_days.json`: source day, source slug, source index-row digest, daily-manifest digest, episode counts, selected counts, and completion status.
- `top10_history.parquet`: one row per date/rank with team ID/name, reconstructed leaderboard score, and selected active submission ID.
- `episodes.parquet`: one row per archived episode with date, episode ID, selected top-10 team(s), ranks, opponent(s), source daily slug, source manifest metrics, replay SHA-256, and shard path.
- `replays_YYYY-MM-DD.parquet`: Zstd-compressed rows containing `episode_id` and raw replay JSON.

Parquet replay shards use Zstd compression. Daily immutable shards avoid rewriting replay contents during a normal incremental run. The archive is flat so Kaggle dataset version uploads do not depend on directory-upload modes.

## Incremental update

The notebook mounts the latest version of the custom archive when available, validates its manifest/hashes/state, and copies it to a staging output. It downloads the latest official index manifest and compares source-row digests with `processed_days.json`.

- already processed + identical digest: skip;
- already processed + changed digest: fail closed rather than silently mixing provenance;
- unseen date: process chronologically.

A date is marked complete only after metadata resolution, top-10 ranking, replay download, replay hashing, Parquet shard write, and shard verification succeed. State/metadata files are written to temporary paths and atomically replaced after the day succeeds. Therefore a partial failure leaves the previous completed archive usable and the failed date pending on rerun.

## Publication

Every successful run publishes a new version of the **same** custom Kaggle Dataset. Because Kaggle dataset versions replace the visible file set, the notebook stages all prior compact archive files plus new immutable day shards before `kaggle datasets version`. It never drops previous days.

No explicit `api.authenticate()` call is used; Kaggle notebook API access is assumed as requested.

## Validation

Local tests cover score-state/ranking, two-newest-submission leaderboard semantics, deterministic episode selection, processed-day source mutation rejection, and idempotent manifest merge. The generated notebook contains a dry-run self-check before any publication path and verifies that applying the same day twice creates no new episode IDs.

## Non-goals

No BonsaiEditFormer training, ranker training, BKG4 edit generation, policy optimization, CUDA search, or submission work occurs in this notebook.
