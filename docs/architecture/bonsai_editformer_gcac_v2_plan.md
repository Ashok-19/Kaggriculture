# BonsaiEditFormer-GCAC-v2 Implementation Plan

**Goal:** Add semantic failure-goal conditioning and target-direction supervision to the existing 5M BonsaiEditFormer Actor-Critic without changing the proven critic/shared encoder or running meaningful local training.

**Architecture:** Preserve every proven critic parameter/output exactly. Add small `actor_goal_*` adapters, goal-conditioned actor queries, append-only `actor_direction_*` per-case heads, compact exact-CUDA goal relabels, support-aware operation masks, deterministic balanced validation, fresh24 per-failure generation, and signed directional shortlisting. Exact CUDA remains the reward oracle.

**Tech Stack:** Python 3.10, JAX/Flax/Optax/Orbax, NumPy, existing BKG4/CUDA binary formats, Bash/uv Lightning bundle workflow.

**Spec:** `docs/architecture/bonsai_editformer_gcac_v2_design.md`

## Global constraints

- Keep `BonsaiEditFormer-5M-v1` shared encoder and existing critic heads bit-for-bit frozen in Stage-B2.
- No Stage-C joint training; no 20M/50M scaling.
- No opponent/team/replay/seed/case-ID embedding in the model.
- Learned operation support gate is data-driven; default minimum safe-positive goal examples is 32.
- Exact per-case truth comes only from historical/current CUDA results.
- No long local training and no large local CUDA campaign.
- Reuse `tools/bkg4_search/bonsai_editformer/scripts/make_actor_lightning_bundle.sh` and `/teamspace/studios/this_studio/` cloud layout.
- Do not alter BKG4 champion/default/VM/ABI, push, or submit.

---

### Task 1: Goal-relabeled dataset and support/split guards

**Files:**
- Modify: `tools/bkg4_search/bonsai_editformer/actor_data.py`
- Create: `tools/bkg4_search/bonsai_editformer/test_gcac_v2.py`

**Produces:** compact transition table plus goal-example table; support manifest; mutation-group split; balanced validation helper.

- [ ] Write failing unit tests for goal class assignment, same-mutation relabel split integrity, support mask derivation, balanced validation class coverage, and required-class failure.
- [ ] Run the focused tests with the established JAX venv and confirm RED failures caused by missing GCAC-v2 APIs.
- [ ] Implement `GoalClass`, goal relabel construction, compact transition/example arrays, AWR-style bounded weights, operation support statistics/mask, mutation-group split mapping, and deterministic balanced validation selection.
- [ ] Re-run focused tests and confirm GREEN.

### Task 2: Goal-conditioned model and frozen-critic boundary

**Files:**
- Modify: `tools/bkg4_search/bonsai_editformer/actor_model.py`
- Modify: `tools/bkg4_search/bonsai_editformer/actor_cli.py`
- Test: `tools/bkg4_search/bonsai_editformer/test_gcac_v2.py`

**Produces:** `BonsaiEditFormer-GCAC-v2`, `goal_mask`, `goal_spec`, fully goal-conditioned actor, append-only directional case heads, actor-only optimizer mask/digest behavior.

- [ ] Add failing tests proving same parent/different goal changes operation, pointer, and semantic logits; proving critic outputs match the frozen critic exactly; and proving only `actor_*` parameters are trainable.
- [ ] Run and confirm RED.
- [ ] Add masked semantic goal pooling from frozen case tokens, goal-conditioned actor context/pointer/semantic heads, and `actor_direction_improve/flip/regress` per-case heads.
- [ ] Update batch construction and actor/directional losses to consume compact relabel examples and child candidate features while preserving critic digest exactness.
- [ ] Re-run tests and confirm GREEN.

### Task 3: Directional generation, fresh24 goal context, and shortlist

**Files:**
- Modify: `tools/bkg4_search/bonsai_editformer/data.py`
- Modify: `tools/bkg4_search/bonsai_editformer/actor_generate.py`
- Modify: `tools/bkg4_search/bonsai_editformer/actor_cli.py`
- Test: `tools/bkg4_search/bonsai_editformer/test_gcac_v2.py`

**Produces:** externally guarded fresh24 semantic feature context; per-failure actor generation; persisted signed predictions; directional shortlist.

- [ ] Add failing tests for exact goal/case alignment, unsupported-op suppression, per-goal generation metadata, signed shortlist ordering, and candidate→goal→CUDA bookkeeping.
- [ ] Run and confirm RED.
- [ ] Add guarded external-case feature/cache helpers and goal-context manifest loading.
- [ ] Add support-aware per-goal actor sampling and global canonical deduplication.
- [ ] Persist goal case IDs only as bookkeeping metadata; persist signed target delta and directional probabilities.
- [ ] Replace absolute-effect primary selection with target-direction → flip → safety → dominance ordering; retain absolute effect only as diagnostics/tie-break.
- [ ] Re-run tests and confirm GREEN.

### Task 4: Balanced evaluation and directional CUDA report

**Files:**
- Modify: `tools/bkg4_search/bonsai_editformer/actor_cli.py`
- Test: `tools/bkg4_search/bonsai_editformer/test_gcac_v2.py`

**Produces:** honest held-out Stage-B2 metrics and mandatory per-loss CUDA attribution.

- [ ] Add failing tests for balanced held-out evaluation counts/negative presence and per-goal CUDA aggregate calculations.
- [ ] Run and confirm RED.
- [ ] Replace `val[:eval_samples]` with deterministic stratified held-out selection.
- [ ] Report goal-aware actor likelihood, operation/target accuracy, signed-delta correlation, improve/L→W/W→L classification metrics and class counts.
- [ ] Extend CUDA campaign report with target-improved/worsened deltas, per-loss generated/changed/improved/best/flip counts, goal coverage, and existing safety metrics.
- [ ] Re-run tests and confirm GREEN.

### Task 5: End-to-end local GCAC-v2 smoke

**Files:**
- Modify: `tools/bkg4_search/bonsai_editformer/actor_cli.py`
- Test: `tools/bkg4_search/bonsai_editformer/test_gcac_v2.py`

**Produces:** one bounded `smoke` command covering required architecture invariants without meaningful training.

- [ ] Add smoke assertions for goal embedding/mask, goal-sensitive logits, support masks, relabel categories, leakage guard, validation negatives, few-step loss decrease, directional-head parameter change, frozen critic digest, checkpoint roundtrip, typed validity/uniqueness, different-goal candidate distributions, signed shortlist order, bookkeeping/alignment.
- [ ] Run dataset construction on a bounded smoke subset.
- [ ] Run 4–8 gradient updates only and verify the required smoke gates.
- [ ] Run generation/compiler/canonicalization smoke only; no large CUDA.

### Task 6: Lightning AI handoff and exact commands

**Files:**
- Modify: `tools/bkg4_search/bonsai_editformer/scripts/make_actor_lightning_bundle.sh`
- Create: `tools/bkg4_search/bonsai_editformer/scripts/gcac_v2_cloud_run.sh`
- Update: `tools/bkg4_search/bonsai_editformer/README.md`

**Produces:** source/data/checkpoint/provenance bundle and real cloud commands for verify, optional smoke, Stage-B2 train, balanced evaluation, per-failure generation, six-lane benchmark, exact CUDA, package, resume.

- [ ] Add a bundle dry-run/validation test that verifies every required local source/artifact path and records hashes/provenance.
- [ ] Update the existing actor bundle script to include GCAC-v2 dataset/manifest/fresh24 goal context and generated command runner without changing deployment method.
- [ ] Add `gcac_v2_cloud_run.sh` with actual argparse options exposed by `actor_cli.py`; root defaults to `/teamspace/studios/this_studio/kaggriculture` via `BONSAI_REMOTE_ROOT`.
- [ ] Build the local bundle, inspect archive contents/hash, and dry-run each cloud command path locally where possible.

### Task 7: Final verification, review, and local commit

**Files:** all changed source/docs/scripts/tests.

- [ ] Run focused GCAC-v2 unit tests from the established JAX venv.
- [ ] Run the complete bounded actor smoke.
- [ ] Run syntax/import checks and script `bash -n` checks.
- [ ] Build and inspect the Lightning bundle; record bytes/SHA/source HEAD and verify no result evidence was deleted.
- [ ] Run `git diff`/`git status`, confirm only intended tracked changes plus pre-existing `run/` and result tar untracked evidence.
- [ ] Review and locally commit the persistent implementation/handoff files; do not push.
- [ ] Return exact copy-paste Lightning commands and stop with the required GCAC-v2 handoff status.
