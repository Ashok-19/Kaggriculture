# B4IM Dual-T4 Corpus Deployment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Package and publish the frozen B4IM factory and add a production dual-T4 Kaggle corpus notebook.

**Architecture:** An allowlisted, integrity-manifested support bundle feeds two pinned long-lived shard workers. A streaming merger deduplicates semantic IDs and creates the exact structural grouped split without loading rich corpus rows globally.

**Tech Stack:** Python 3, unittest, NumPy, gzip/JSONL/NPZ, CUDA/NVCC, Kaggle CLI, Jupyter notebook JSON.

**Spec:** `docs/superpowers/specs/2026-08-24-b4im-dual-t4-corpus-deployment-design.md`

## Global Constraints

- Preserve B4IM1/v1 and B4IC1/v1 behavior; do not redesign the generator.
- Compile the Kaggle evaluator as native `sm_75` with `fast_math=false`.
- Use only discovery-role offline cases and preserve 64/400 tape-disjoint panels.
- Run no large local campaign, no training, no competition submission, no Git push.

---

### Task 1: Deployment Contracts

**Files:**
- Create: `tools/bkg4_search/test_kaggle_invariant_corpus.py`
- Create: `tools/bkg4_search/kaggle_invariant_corpus.py`

- [ ] Write failing tests for portable bank validation, safe extraction, shard serialization, semantic dedup, and reference-equivalent grouped splitting.
- [ ] Run the focused tests and confirm they fail because deployment APIs are absent.
- [ ] Implement the minimal shared contracts and rerun the focused tests.

### Task 2: Support Packager And Smoke

**Files:**
- Create: `tools/bkg4_search/package_kaggle_invariant_corpus_support.py`
- Test: `tools/bkg4_search/test_kaggle_invariant_corpus.py`

- [ ] Add failing inventory, provenance, portable path, and smoke fixture tests.
- [ ] Implement allowlisted staging, tape rewriting, exact fixture generation, manifests, and reproducible tar creation.
- [ ] Build the ignored staging dataset and validate imports, assets, banks, panels, and hashes.

### Task 3: Long-Lived Worker And Merger

**Files:**
- Create: `tools/bkg4_search/kaggle_invariant_corpus_runner.py`
- Create: `tools/bkg4_search/merge_invariant_corpus_shards.py`
- Modify: `tools/bkg4_search/generate_invariant_specialists.py`

- [ ] Add failing tests for skip-saturation, deadline/disk admission, worker GPU mapping, cleanup, merge conflicts, and split output.
- [ ] Add a backward-compatible saturation-report flag.
- [ ] Implement calibration/smoke/worker commands, durable shard commits, and streaming final merge/reporting.
- [ ] Run dry-run orchestration and the validated 384-row serialization/split equivalence checks.

### Task 4: Notebook

**Files:**
- Create: `tools/bkg4_search/build_kaggle_invariant_corpus_notebook.py`
- Create: `notebooks/kaggriculture_b4im_elite_corpus_dual_t4/kaggriculture_b4im_elite_corpus_dual_t4.ipynb`
- Create: `notebooks/kaggriculture_b4im_elite_corpus_dual_t4/kernel-metadata.json`

- [ ] Add notebook contract tests for the configuration, hardware gate, support validation, `sm_75`, both-GPU smoke/calibration, workers, monitoring, merge, publication option, and final report.
- [ ] Generate a compact production notebook and compile every code cell.
- [ ] Simulate materialization and worker launch path assumptions under scratch.

### Task 5: Publish And Finish

- [ ] Query the intended Kaggle dataset slug.
- [ ] Create a new private dataset or version the matching existing private dataset.
- [ ] Verify the published version, privacy, size, and three support marker files.
- [ ] Run focused and invariant regression tests, py_compile, notebook checks, `git diff --check`, and repository hygiene checks.
- [ ] Commit source/tests/notebook/docs, verify the worktree is clean, and do not push.
