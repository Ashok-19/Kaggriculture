# Kaggriculture CUDA-Labeled Edit Corpus Builder Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Build and smoke-test a second Kaggle notebook plus minimal support dataset that incrementally converts the top-10 replay archive into a canonical exact-CUDA BonsaiEditFormer edit corpus without training or modifying the champion.

**Architecture:** Reuse the existing typed `EditAction`/`ModuleGenome` grammar, replay-to-native-tape conventions, current BKG4 runtime, CUDA codecs and Kaggle publication patterns. Add only a thin current-BKG4 typed-overlay CUDA evaluator and a canonical corpus runtime that stores structured parent/edit/case/outcome records rather than current model tensors.

**Tech Stack:** Python 3, unittest, pandas/pyarrow Parquet, CUDA C++17/nvcc, existing `bonsai-kg` BKG4 ABI-4 runtime, Kaggle CLI/notebook metadata.

**Spec:** `docs/superpowers/specs/2026-08-21-kaggriculture-edit-corpus-builder-design.md`

## Global Constraints

- Do not train BonsaiEditFormer.
- Do not modify/promote the champion BKG4.
- Do not submit to the competition.
- Large-scale evaluation is CUDA only; local validation is bounded to a tiny fixture.
- Exact current CUDA truth is the only outcome label source.
- Team/opponent names, replay IDs and seed IDs are provenance only and never model features.
- The replay archive is streamed one date shard at a time.
- The corpus must be deterministic, resumable, idempotent, corruption-safe and append-only by date.
- Keep every repository file tracked or ignored and finish with a clean worktree.

---

### Task 1: Canonical corpus core and replay-derived cases

**Files:**
- Create: `tools/edit_corpus/__init__.py`
- Create: `tools/edit_corpus/corpus.py`
- Create: `tools/edit_corpus/test_corpus.py`

**Interfaces:**
- Produces `replay_to_native_tapes(replay_json, provenance) -> list[dict]`.
- Produces `build_case_candidates(...) -> list[dict]` and `select_representative_cases(...) -> list[dict]`.
- Produces `canonical_edit_identity(parent_id, action, child_id) -> str`.
- Produces `derive_case_label(...)` and `derive_edit_labels(...)`.
- Produces `pending_replay_shards(...)`, manifest writer/validator and leakage audit helpers.

- [x] **Step 1: Write failing tests** covering replay action offset, both-seat case construction, deterministic representative selection, feature/provenance separation, edit identity, exact label counts and processed-shard idempotency.
- [x] **Step 2: Run** `python3 -m unittest tools.edit_corpus.test_corpus -v` and confirm failures are for missing corpus APIs.
- [x] **Step 3: Implement minimal `corpus.py`** with pure deterministic helpers only; no CUDA subprocess logic.
- [x] **Step 4: Re-run** `python3 -m unittest tools.edit_corpus.test_corpus -v` and confirm green.

### Task 2: Typed candidate generator

**Files:**
- Create: `tools/edit_corpus/candidates.py`
- Create: `tools/edit_corpus/test_candidates.py`
- Reuse without modification: `tools/bkg4_search/bonsai_editformer/actor_grammar.py`
- Reuse without modification: `tools/bkg4_search/module_codec.py`

**Interfaces:**
- Produces `generate_typed_candidates(parent, context_rows, n, seed, route_ids) -> list[GeneratedEdit]`.
- `GeneratedEdit` contains the exact `EditAction`, parent/child semantic IDs, canonical action JSON and resulting genome JSON.

- [x] **Step 1: Write failing tests** requiring deterministic generation, all generated edits to pass `apply_edit`, explicit target/location fields, semantic dedup, multiple operation families and inclusion of deliberately broad/negative-oriented candidates.
- [x] **Step 2: Run** `python3 -m unittest tools.edit_corpus.test_candidates -v` and verify RED.
- [x] **Step 3: Implement generator** using `valid_operation_mask`, `target_masks`, `apply_edit`, observed context thresholds and existing `random_condition`/route IDs.
- [x] **Step 4: Re-run tests** and verify GREEN.

### Task 3: Exact current-BKG4 CUDA typed-overlay evaluator

**Files:**
- Create: `cuda-engine/src/bkg4_edit_corpus_eval.cu`
- Create: `tools/edit_corpus/cuda_results.py`
- Create: `tools/edit_corpus/test_cuda_results.py`

**Interfaces:**
- Evaluator CLI: `--controller --routes --legacy-routes --cases --genomes --out --strategic-route-slot --feed-slot --feed-count --tpb`.
- Result format: `B4EC1`, version 1, dimensions genomes x cases, records `(own_money:int32, opp_money:int32, action_changes:uint16, activations:uint16, flags:uint16, reserved:uint16)`.
- Python `load_edit_results(path) -> list[list[EditCudaResult]]` validates magic/version/dimensions/trailing bytes/flags.

- [x] **Step 1: Write failing binary-parser tests** for valid and corrupt result files.
- [x] **Step 2: Run** `python3 -m unittest tools.edit_corpus.test_cuda_results -v` and verify RED.
- [x] **Step 3: Implement the parser** and get parser tests green.
- [x] **Step 4: Implement CUDA evaluator** by composing current BKG4 tick/materialization/legacy/exploit/feed code with the existing ModuleGenome condition ABI; do not alter simulator semantics.
- [x] **Step 5: Compile only the new evaluator** with local nvcc using the same include/runtime paths as existing BKG4 evaluators.
- [x] **Step 6: Run bounded fixture parity**: empty genome vs existing current BKG4 evaluator on a few known cases must produce identical margins; then evaluate a few non-empty typed genomes and require zero flags.

### Task 4: Minimal versioned Kaggle support bundle

**Files:**
- Create: `tools/edit_corpus/build_support_dataset.py`
- Create: `tools/edit_corpus/test_support_dataset.py`
- Create: `tools/edit_corpus/support_dataset_seed/dataset-metadata.json`
- Create: `tools/edit_corpus/support_dataset_seed/README.md`

**Interfaces:**
- CLI: `python3 tools/edit_corpus/build_support_dataset.py --output scratch/kaggriculture_edit_corpus_support --force`.
- Output includes `support_manifest.json` plus only required runtime/source/assets.

- [x] **Step 1: Write failing tests** for required path inventory, exclusion of unrelated repo/scratch content, manifest versions and deterministic relative-path ordering.
- [x] **Step 2: Run** `python3 -m unittest tools.edit_corpus.test_support_dataset -v` and verify RED.
- [x] **Step 3: Implement packager** with an explicit allowlist and fail-closed missing-file checks.
- [x] **Step 4: Build a local bundle** and run tests green; inspect resulting size/path inventory.

### Task 5: Kaggle corpus runtime and incremental persistence

**Files:**
- Create: `tools/edit_corpus/kaggle_runtime.py`
- Create: `tools/edit_corpus/test_kaggle_runtime.py`

**Interfaces:**
- Locates attached replay archive, prior corpus and support bundle.
- Streams one replay Parquet shard/date.
- Writes/validates `base_structures.parquet`, `parents.parquet`, date case/edit/label shards, consolidated indexes, `processed_shards.json`, `corpus_manifest.json`.
- Calls compiled CUDA evaluator in bounded candidate chunks.

- [x] **Step 1: Write failing tests** using a tiny synthetic archive/prior corpus fixture for first-run, rerun idempotency, changed-source rejection, prior-row preservation and no provenance leakage into `model_context_json`.
- [x] **Step 2: Run** `python3 -m unittest tools.edit_corpus.test_kaggle_runtime -v` and verify RED.
- [x] **Step 3: Implement runtime** with deterministic one-day transactions and temporary-directory cleanup.
- [x] **Step 4: Re-run tests** and verify GREEN.

### Task 6: Deterministic Kaggle notebook generator

**Files:**
- Create: `tools/edit_corpus/build_notebook.py`
- Create: `tools/edit_corpus/test_build_notebook.py`
- Create: `notebooks/kaggriculture_edit_corpus_builder/kaggriculture_edit_corpus_builder.ipynb`
- Create: `notebooks/kaggriculture_edit_corpus_builder/kernel-metadata.json`
- Create: `tools/edit_corpus/corpus_dataset_seed/dataset-metadata.json`
- Create: `tools/edit_corpus/corpus_dataset_seed/README.md`

**Interfaces:**
- Notebook inputs: replay archive, support dataset, prior corpus dataset.
- GPU enabled, internet enabled for dataset publication.
- `build_notebook.py --check` validates deterministic generated notebook/metadata.

- [x] **Step 1: Write failing notebook tests** for deterministic/compilable cells, GPU metadata, exact three dataset inputs, one-shard-at-a-time processing, CUDA compilation/evaluation, prior-corpus validation and dataset version publication only on new dates.
- [x] **Step 2: Run** `python3 -m unittest tools.edit_corpus.test_build_notebook -v` and verify RED.
- [x] **Step 3: Implement generator** embedding/reusing `corpus.py`, `candidates.py`, `cuda_results.py` and `kaggle_runtime.py` from the support mount, not by duplicating simulator semantics.
- [x] **Step 4: Generate notebook and run** `python3 tools/edit_corpus/build_notebook.py --check` plus tests green.

### Task 7: Documentation, bounded validation and clean finish

**Files:**
- Create: `tools/edit_corpus/README.md`
- Update only if required: `.gitignore` for generated local support bundle path.

**Interfaces:**
- README documents exact support/corpus dataset create/version commands, notebook push/setup inputs, schema and incremental behavior.

- [x] **Step 1: Document exact commands** for support dataset build/create/version, corpus seed creation, notebook generation/tests/push and Kaggle-side one-shard smoke mode.
- [x] **Step 2: Run full bounded validation:** all `tools.edit_corpus` unit tests, notebook `--check`, support-bundle build, CUDA compile/parity smoke when local CUDA is available, and existing replay-archive regression tests.
- [x] **Step 3: Run** `git diff --check` and inspect `git status --short --branch`; remove/ignore only generated task artifacts.
- [x] **Step 4: Review changes** against every requested deliverable and confirm no champion/training/submission files changed.
- [x] **Step 5: Commit the completed infrastructure locally** with one focused commit and leave the repository clean. Do not push.
