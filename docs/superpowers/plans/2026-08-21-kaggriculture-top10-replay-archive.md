# Kaggriculture Top-10 Replay Archive Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a deterministic, resumable Kaggle notebook that archives only official daily episodes involving that day's reconstructed top-10 Kaggriculture teams.

**Architecture:** A small tested Python helper owns pure ranking/state/provenance logic and runtime Kaggle operations. A notebook builder embeds that helper into a standalone Kaggle notebook. The notebook uses official daily episode IDs for source discovery/selection, refreshes full public episode histories for discovered candidate teams, ranks teams from chronological `updatedScore` history using the best of their two newest submissions at each UTC day boundary, writes immutable Zstd Parquet replay shards, preserves previous archive state, and versions one private Kaggle Dataset.

**Tech Stack:** Python standard library, pandas, pyarrow, Kaggle CLI, requests/kaggle-environments EpisodeService metadata inside Kaggle.

**Spec:** `docs/superpowers/specs/2026-08-21-kaggriculture-top10-replay-archive-design.md`

## Global Constraints

- Never rank players from daily replay file order or `avg_score` ordering.
- Never bulk-extract the ~20 GiB daily replay datasets.
- No explicit Kaggle authentication call inside the notebook.
- Fail closed if score-bearing episode metadata is unavailable.
- Preserve all previous archived days when publishing a new dataset version.
- No RL/ranker/BKG4 training or optimization.
- Keep generated local test/runtime artifacts under ignored scratch paths.

---

### Task 1: Pure ranking and incremental-state core

**Files:**
- Create: `tools/replay_archive/top10_archive.py`
- Test: `tools/replay_archive/test_top10_archive.py`

**Interfaces:**
- `normalize_episode_payload(payload: dict) -> list[dict]`
- `apply_episode_updates(state: dict, episodes: list[dict]) -> dict`
- `daily_top_teams(state: dict, n: int = 10) -> list[dict]`
- `select_episode_ids(episodes: list[dict], top_team_ids: set[int]) -> list[int]`
- `pending_source_days(index_rows: list[dict], processed: dict) -> list[dict]`
- `merge_episode_rows(existing: list[dict], incoming: list[dict]) -> list[dict]`

- [ ] Write tests for camel/Pascal episode fields, rating updates, newest-two-submission semantics, deterministic selection, source mutation rejection, and idempotent merges.
- [ ] Run `python3 -m unittest tools.replay_archive.test_top10_archive -v` and verify failure because the helper does not yet exist.
- [ ] Implement the minimum pure functions.
- [ ] Re-run the focused tests until green.

### Task 2: Runtime archive writer and Kaggle I/O

**Files:**
- Modify: `tools/replay_archive/top10_archive.py`
- Test: `tools/replay_archive/test_top10_archive.py`

**Interfaces:**
- `sha256_file(path: Path) -> str`
- `source_row_digest(row: Mapping[str, object]) -> str`
- `validate_archive(root: Path) -> dict`
- `write_archive_manifest(root: Path, ...) -> dict`
- runtime helpers for downloading one dataset manifest, resolving episode metadata in bounded batches, downloading one replay, and writing a verified daily Parquet shard.

- [ ] Add failing tests for stable source digests, archive-manifest hashing, and partial-day state not being committed.
- [ ] Run focused tests and confirm the new assertions fail.
- [ ] Implement atomic JSON writes, hash verification, bounded retries/backoff, and strict metadata validation.
- [ ] Re-run focused tests.

### Task 3: Standalone Kaggle notebook generator

**Files:**
- Create: `tools/replay_archive/build_notebook.py`
- Create: `notebooks/kaggriculture_top10_replay_archive/kaggriculture_top10_replay_archive.ipynb`
- Create: `notebooks/kaggriculture_top10_replay_archive/kernel-metadata.json`

**Interfaces:**
- Builder reads `top10_archive.py` and embeds it into the notebook, so the notebook has no repo dependency at runtime.
- Notebook config exposes only dataset owner/slug, top-N=10, metadata batch size, and publication/dry-run switches.

- [ ] Generate notebook cells for configuration, source/archive discovery, bootstrap/incremental processing, validation, publication, and summary.
- [ ] Generate private CPU notebook metadata with internet enabled, official episodes-index dataset source, custom archive dataset source, and Kaggriculture competition source.
- [ ] Run the builder twice and verify identical notebook bytes.
- [ ] Parse the notebook JSON and compile every code cell.

### Task 4: Tiny idempotency dry run

**Files:**
- Test only; output under `scratch/replay_archive_dryrun/`.

- [ ] Run a synthetic two-day fixture through the pure ranking/selection/merge path.
- [ ] Apply the second day twice and assert the second application adds zero episode IDs and leaves processed-day/top-10 state unchanged.
- [ ] Remove/leave fixture output under ignored `scratch/` only.

### Task 5: Documentation, commands, and clean-tree gate

**Files:**
- Create: `tools/replay_archive/README.md`
- Modify: `.gitignore` only if needed for exact local result/archive artifacts.

- [ ] Document dataset schema, compression, first-run bootstrap, incremental semantics, corruption recovery, and schedule setup.
- [ ] Document exact Kaggle CLI commands for creating the private dataset, pushing/updating the notebook, checking status/output, and manually running one archive version.
- [ ] Run `python3 -m unittest tools.replay_archive.test_top10_archive -v`.
- [ ] Run `python3 tools/replay_archive/build_notebook.py --check`.
- [ ] Run `git diff --check` and inspect `git status --short`.
- [ ] Ensure every task-generated file is tracked or ignored and no unrelated file was removed.
