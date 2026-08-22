# Permanent BKG4 Edit Search Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Replace the shallow one-generation proposal path with a reusable recursive typed BKG4 module-edit search subsystem that persists QD stepping stones, uses exact CUDA as truth, learns from broader-panel counterexamples, and exports strict exact-labeled ranker corpora.

**Architecture:** A new `tools/bkg4_search/edit_search/` package owns versioned EditProgram/state identities, emitters, QD archive, case-panel manager, stagnation/escalation, exact CUDA batching, search orchestration, corpus export/validation and benchmark CLI. It reuses the existing `ModuleGenome`, typed `EditAction/apply_edit`, `B4GM1` packer, `B4EC1` parser and exact label derivation instead of modifying the old corpus builder.

**Tech Stack:** Python 3, dataclasses/json/argparse, NumPy/PyArrow for bounded corpus output, existing C++/CUDA `bkg4_edit_corpus_eval.cu`, unittest.

**Spec:** `docs/superpowers/specs/2026-08-22-bkg4-permanent-edit-search-design.md`

## Global Constraints

- BKG4/Bonsai remains the final deterministic policy backbone; do not create BKG5.
- Exact CUDA is the search/label source of truth; no Python per-game evaluator.
- Large-scale search is not run locally in this task.
- Production champion is never modified/promoted automatically.
- Legal observation registry only; no seed/team/replay/opponent-private executable features.
- Search-state, grammar, emitter, result ABI, BKG4 ABI, feature registry and corpus schemas are independently versioned.
- Concrete candidates are parent-specific; motifs may be re-instantiated on a future parent and must be reevaluated.
- Repo must finish clean; the pre-existing downloaded corpus is left untouched and ignored as local validated data.

---

### Task 1: Canonical EditProgram and parent profiles

**Files:**
- Create: `tools/bkg4_search/edit_search/__init__.py`
- Create: `tools/bkg4_search/edit_search/schema.py`
- Test: `tools/bkg4_search/edit_search/test_schema.py`

**Interfaces:**
- `ParentProfile.from_directory(path) -> ParentProfile`
- `EditAtom.from_action(parent, action) -> EditAtom`
- `EditProgram.build(parent, actions, route_ids, emitter, seed, ...) -> EditProgram`
- `EditProgram.reconstruct(parent, route_ids) -> ModuleGenome`

- [x] Write tests proving single/multi-edit reconstruction, state-relative targets, invalid target rejection, stable IDs, future-parent profile loading, and all legal operation families including `CHANGE_PRIORITY` on a 2-rule parent.
- [x] Run `python3 -m unittest tools.bkg4_search.edit_search.test_schema -v` and confirm RED because package/API does not exist.
- [x] Implement the minimum canonical versioned representation and parent-profile loader.
- [x] Run the same test and confirm GREEN.

### Task 2: Exact fitness, descriptors, persistent QD archive and hard cases

**Files:**
- Create: `tools/bkg4_search/edit_search/fitness.py`
- Create: `tools/bkg4_search/edit_search/archive.py`
- Test: `tools/bkg4_search/edit_search/test_archive.py`

**Interfaces:**
- `StateEvaluation`, `PairEvaluation.from_state_results(...)`
- `LexicographicFitness.from_pair(...)`
- `SearchArchive.add_state/add_program/consider_evaluation/save/load`
- `SearchArchive.record_counterexamples(...)`

- [x] Write failing tests for no-compensation lexicographic ordering, production-vs-exploration archive separation, deterministic niche replacement, checkpoint/resume, hard-case persistence and exact semantic dedup.
- [x] Run test and confirm RED.
- [x] Implement the minimal archive and stable structural/behavior descriptors.
- [x] Run test and confirm GREEN.

### Task 3: Generic case-bank and informative screening-panel manager

**Files:**
- Create: `tools/bkg4_search/edit_search/panels.py`
- Test: `tools/bkg4_search/edit_search/test_panels.py`

**Interfaces:**
- `CaseRecord`, `CaseBank.load_jsonl/load_tsv_sidecar`
- `CasePanelManager.select_screen(...)`
- `CasePanelManager.compare_screen_to_full(...)`
- `CasePanelManager.refresh_required(...)`

- [x] Write failing tests for both-seat/family/hard-loss/fragile/protected/random-anchor mixture, counterexample priority, deterministic selection and Stage-1 recall diagnostics.
- [x] Implement panel selection using case metadata and disagreement/counterexample statistics only.
- [x] Verify GREEN.

### Task 4: Common emitter interface and recursive proposal set

**Files:**
- Create: `tools/bkg4_search/edit_search/emitters.py`
- Test: `tools/bkg4_search/edit_search/test_emitters.py`

**Interfaces:**
- `EmitterContext`
- `SearchEmitter.propose(...)`
- `SystematicEmitter`, `LocalRefinementEmitter`, `StructuralEmitter`, `RecombinationEmitter`, `LargeNeighborhoodEmitter`, `EvolvabilityMCTSEmitter`, `RandomTypedEmitter`

- [x] Write failing tests proving deterministic output, WHAT+WHERE+HOW completeness, zero/one/2+ rule parent support, non-216 prefix search, all applicable operations, recursive children as legal parents, recombination/LNS multi-edit programs, and random exploration floor.
- [x] Implement data-derived threshold neighborhoods from observed values/quantiles + existing predicates + local offsets and meaningful legal constants.
- [x] Implement semantic recombination through typed edits only; no byte splicing.
- [x] Implement bounded destroy/repair and MCTS derivations using existing `EvolvabilityTree` principles.
- [x] Verify GREEN.

### Task 5: Scheduler, lexicase specialists and anti-stagnation escalation

**Files:**
- Create: `tools/bkg4_search/edit_search/search.py`
- Test: `tools/bkg4_search/edit_search/test_search.py`

**Interfaces:**
- `FixedEmitterScheduler` (adaptive allocation remains omitted unless a future equal-budget ablation beats fixed allocation)
- `lexicase_parent(...)`
- `StagnationMonitor.observe(...)`
- `SearchEngine.propose_generation(...)`

- [x] Write failing tests for many-elite parent selection, specialist preservation, fixed nonzero random floor, rising duplicate-rate stagnation detection, real escalation of parent/emitter/depth regime, and deterministic resume.
- [x] Implement fixed scheduler as default; omit adaptive UCB allocation because no equal-budget ablation demonstrated a win, preserving explicit exploration floors.
- [x] Implement escalation levels 0..6 from the spec.
- [x] Verify GREEN.

### Task 6: Exact CUDA batch evaluator and multi-fidelity search loop

**Files:**
- Create: `tools/bkg4_search/edit_search/evaluator.py`
- Extend: `tools/bkg4_search/edit_search/search.py`
- Test: `tools/bkg4_search/edit_search/test_evaluator.py`

**Interfaces:**
- `CudaEvaluatorProfile`
- `ExactCudaEvaluator.evaluate(states, panel, work_dir) -> dict[semantic_id, StateEvaluation]`
- `SearchEngine.run_round(...)`

- [x] Write failing parser/command tests proving one batched evaluator invocation per state batch, B4EC1 corruption rejection, parent+child result alignment, and no CPU evaluator fallback.
- [x] Implement B4GM1 packing + exact evaluator subprocess + B4EC1 loading.
- [x] Implement Stage-1 evaluation, archive selection, Stage-2 evaluation, counterexample feedback and panel-recall accounting.
- [x] Verify GREEN.
- [x] Compile/run one tiny existing exact CUDA smoke when GPU/assets are available; never launch a large campaign.

### Task 7: Exact corpus exporter and validator

**Files:**
- Create: `tools/bkg4_search/edit_search/corpus.py`
- Test: `tools/bkg4_search/edit_search/test_corpus.py`

**Interfaces:**
- `export_corpus(archive, case_bank, out_dir, certification_panel=...)`
- `validate_export(out_dir)`

- [x] Write failing tests for full reconstruction, parent/child/case resolution, exact label recomputation, duplicate edit×case rejection, leakage boundary, logical sample categories and split grouping keys.
- [x] Implement Parquet exporter containing parent/child structures, full EditProgram, exact per-case labels, trace/effect fields, `search_*` provenance, descriptors and split groups.
- [x] Implement fail-closed validator.
- [x] Verify GREEN.

### Task 8: CLI, persistence lifecycle and documentation

**Files:**
- Create: `tools/bkg4_search/edit_search/cli.py`
- Create: `tools/bkg4_search/edit_search/README.md`
- Test: `tools/bkg4_search/edit_search/test_cli.py`

**Interfaces:**
- `init`, `propose`, `search`, `resume`, `validate`, `export-corpus`, `benchmark`, `anti-saturation`

- [x] Write failing tests for init/resume/version incompatibility and parent-change motif re-instantiation safety.
- [x] Implement exact commands and atomic state persistence.
- [x] Document current route-module semantic ceiling and future backend extension boundary.
- [x] Verify GREEN.

### Task 9: Equal-budget benchmark and anti-saturation harness

**Files:**
- Create: `tools/bkg4_search/edit_search/benchmark.py`
- Test: `tools/bkg4_search/edit_search/test_benchmark.py`

**Interfaces:**
- `benchmark_proposers(...)`
- `run_anti_saturation(...)`

- [x] Write failing tests for equal candidate/evaluation budgets and required metrics: uniqueness, duplicate rate, operation/target/parent depth, QD growth, useful-safe/L2W/W2L/severe rates, best target/floor improvements, novelty trajectory and escalation events.
- [x] Implement baselines: old `generate_typed_candidates`, random typed, fixed single-parent mutation, permanent search proposer.
- [x] Verify GREEN.
- [x] Run a bounded fresh held-out exact CUDA comparison using a small existing case bank; if the first configuration does not materially improve structural coverage/search yield, iterate emitter allocation/selection based on evidence without expanding local game count excessively.
- [x] Run bounded anti-saturation evidence and record duplicate/archive/novelty/emitter/escalation trajectories.

### Task 10: Full verification and clean integration

**Files:**
- Modify only where verification reveals a required compatibility fix.

- [x] Run all new `edit_search` tests.
- [x] Run existing `tools/edit_corpus` and `bonsai-kg/tests/test_bkg4_search.py` regressions to ensure no old path regressed.
- [x] Run bounded CUDA smoke/parity command(s) and confirm `bad=0`.
- [x] Run `repo_git_review` and inspect every diff for champion/ABI/data accidents.
- [x] Ensure generated benchmark/search artifacts are under ignored `scratch/` or removed; permanent source/docs/tests remain tracked.
- [x] Commit the reviewed permanent implementation so the repo is clean except for the pre-existing untracked downloaded corpus.
