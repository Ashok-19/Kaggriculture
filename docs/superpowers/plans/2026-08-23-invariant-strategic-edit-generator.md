# Invariant Strategic Edit Generator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and validate a permanent cross-trajectory invariant strategic BKG4 edit generator with resident CUDA evaluation and ranker-consumable semantic edits.

**Architecture:** Keep the existing BKG4 parent and legacy `ModuleGenome` intact. Add a bounded canonical `B4IM1` mini-IR with lifted guards and strategic macros, compile it to ordinary BKG4, evaluate it resident on CUDA, and search it with tape-grouped robust objectives and discovery-only cross-fit.

**Tech Stack:** Python 3, C++17, CUDA C++, BKG4 schema/ABI 4, unittest, existing edit-search/corpus infrastructure.

**Spec:** `docs/superpowers/specs/2026-08-23-invariant-strategic-edit-generator.md`

## Global Constraints

- BKG4 schema = 4, ABI = 4.
- Legacy ModuleGenome remains supported but is not the primary backend.
- Exact game search/evaluation uses CUDA; no heavy CPU game campaign.
- Held-out certification is label-only and never influences search.
- No champion promotion, Kaggle submission, push, or BonsaiEditFormer training.
- Empty/no-op invariant module must be exactly equivalent to the frozen current parent.
- Python / CPU C++ / CUDA / distilled BKG4 semantics must agree exactly.

---

### Task 1: Canonical invariant module grammar

**Files:**
- Create: `tools/bkg4_search/invariant_module.py`
- Test: `tools/bkg4_search/edit_search/test_invariant_module.py`

**Interfaces:**
- Produces `InvariantStrategicModule`, `InvariantRule`, `LiftedGuard`, `GuardAtom`, `AbstractAction`, `ModuleSketch`, `B4IM1` pack/unpack, semantic IDs, anti-patch validation and ordinary-BKG4 distillation.

- [ ] Write tests for canonicalization, semantic IDs, bounds, identity-free production validation, lifted unit binding, no-op, B4IM1 fixed ABI and exact round-trip.
- [ ] Run those tests and confirm missing-module failure.
- [ ] Implement the minimal canonical grammar and packer.
- [ ] Run the tests to green.
- [ ] Add distillation tests for per-unit compile-time expansion and stateful route mode.
- [ ] Implement BKG4 distillation using registry-v1 only.
- [ ] Re-run tests.

### Task 2: Grouped objectives and constant solver

**Files:**
- Create: `tools/bkg4_search/invariant_groups.py`
- Test: `tools/bkg4_search/edit_search/test_invariant_groups.py`

**Interfaces:**
- Produces tape grouping, rotating tape-disjoint folds, robust group fitness, production multi-tape qualification, boundary/interval constant solver and grouped counterexample classes.

- [ ] Write failing tests for seat-coherent tape grouping, fold disjointness, singleton restriction, worst-group ordering, broad interval preference and certification exclusion.
- [ ] Run tests red.
- [ ] Implement grouping/fitness/solver.
- [ ] Run tests green.

### Task 3: Resident CPU/CUDA B4IM1 execution

**Files:**
- Create: `bonsai-kg/include/bkg4_invariant_module.hpp`
- Create: `bonsai-kg/tests/bkg4_invariant_overlay_cpp.cpp`
- Create: `cuda-engine/src/bkg4_invariant_edit_eval.cu`
- Modify: `cuda-engine/Makefile`
- Create: `tools/bkg4_search/invariant_evaluator.py`
- Test: `tools/bkg4_search/edit_search/test_invariant_evaluator.py`

**Interfaces:**
- CPU/CUDA share an exact packed `B4IM1` layout and rule/action semantics.
- Python evaluator wrapper returns existing `StateEvaluation` objects.

- [ ] Add Python ABI/wrapper tests that fail because evaluator support is absent.
- [ ] Implement shared packed ABI and CPU reference evaluator.
- [ ] Implement resident candidate×case CUDA evaluator with existing base/controller/tapes resident.
- [ ] Build bounded binaries and verify empty and behavior-changing CPU/CUDA parity.
- [ ] Add mini-IR ↔ distilled-BKG4 parity for representative macros.

### Task 4: Invariant search backend

**Files:**
- Create: `tools/bkg4_search/invariant_search.py`
- Test: `tools/bkg4_search/edit_search/test_invariant_search.py`

**Interfaces:**
- Produces mechanism-oriented goals, sketch emitters, semantic QD descriptors, recombination compatibility, LNS, bounded structural MCTS choices, grouped CEGIS and capability escalation.

- [ ] Write failing tests for multi-tape production support, anti-patch restrictions, semantic recombination conflicts, grouped CEGIS repair, skeleton-first constant refinement and capability-level escalation.
- [ ] Implement deterministic emitters/search archive sufficient for bounded production search.
- [ ] Run tests green.

### Task 5: Corpus and BonsaiEditFormer invariant encoding

**Files:**
- Create: `tools/bkg4_search/invariant_schema.py`
- Modify: `tools/bkg4_search/bonsai_editformer/data.py`
- Test: `tools/bkg4_search/edit_search/test_invariant_schema.py`

**Interfaces:**
- Produces lossless semantic edit programs and canonical invariant AST tokens. Outcome/certification/group fields remain label-only.

- [ ] Write failing reconstruction, token round-trip and leakage-rejection tests.
- [ ] Implement versioned invariant edit program and tokenization.
- [ ] Run tests green.

### Task 6: Bounded architecture spike and five-split validation

**Files:**
- Create: `tools/bkg4_search/validate_invariant_generator.py`
- Create final evidence under ignored `scratch/bkg4_invariant_search/` and summarize durable results in `tools/bkg4_search/INVARIANT_STRATEGIC_GENERATOR_VALIDATION.md`.

**Interfaces:**
- Runs existing parent/case assets through exact CUDA only for game evaluation; compares legacy and invariant arms at equal candidate×case budget.

- [ ] Identify independent tapes and mechanism cohorts from the current CaseBank.
- [ ] Run a small 2–3-family B4IM1 spike on grouped discovery plus untouched mini-holdout.
- [ ] If a useful family transfers, anchor the bounded invariant search; otherwise inspect engine mechanics and test another bounded action family, never scalar-route widening.
- [ ] Run up to five independent grouped splits at bounded GPU cost and compute retention/new-loss metrics.
- [ ] Run equal-budget random/fixed/current-legacy/invariant comparison.
- [ ] Record candidate bytes, GPU memory plan, games/s and host/CUDA walltime.

### Task 7: Regression, review, and local commit

**Files:**
- Update: `tools/bkg4_search/INVARIANT_STRATEGIC_GENERATOR_VALIDATION.md`

- [ ] Run all new invariant tests.
- [ ] Run existing `edit_search`, `edit_corpus`, `bonsai-kg` search tests, compileall and `git diff --check`.
- [ ] Run fresh CPU/CUDA parity after final code.
- [ ] Review git diff and ensure no untracked files.
- [ ] Create a reviewed local-only commit; do not push.
- [ ] Release the task claim.
