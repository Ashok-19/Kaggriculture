# BonsaiEditFormer

Offline JAX/Flax/Optax infrastructure for synthesizing deterministic BKG4 specialists. Neural models are search-time only; accepted edits compile/materialize into ordinary BKG4 and no neural runtime is shipped to Kaggle.

## Proven critic

`BonsaiEditFormer-5M-v1` remains the frozen structural edit critic/ranker:

- d=288, heads=6, FFN=960, tree/case/fusion blocks=3/1/1
- 4,991,412 parameters
- structural tree attention + typed edit encoding + legal-state activation context
- exact-CUDA regression, loss-to-win, margin, worst-win, ranking and signed per-case-delta objectives

The frozen current400 experiment found both strict-safe edits inside neural top-16. Its parameters, names, shapes and outputs are preserved exactly by GCAC-v2.

Scaling presets remain available for later experiments but are not part of Stage-B2:

- `20m`: 19,742,698 critic/ranker parameters
- `50m`: 53,558,719 critic/ranker parameters

## BonsaiEditFormer-GCAC-v2

GCAC-v2 fixes the specific failure of Actor-Critic v1: typed editing worked, but 0/64 Actor+Critic fresh24 candidates improved even one of B371's six losses.

The final 5M-scale synthesis model has **6,187,787 total parameters**:

- frozen proven critic/shared representation: **4,991,412**
- trainable actor/goal/directional parameters: **1,196,375**

The shared structural encoder is not replaced. New `actor_*` components consume stop-gradient copies of the proven tree/case/fused representations.

### Semantic failure goal

Each synthesis request supplies:

- `goal_mask[B, case_cap]`: one requested legal-state case token or a semantic cluster;
- `goal_spec[B,4]`: loss-repair / future margin-expansion mode, desired positive direction, and optional margin floor.

There is no learned opponent, team, replay, seed or case-ID embedding. Goal attention pools the actual legal-state case representations selected by `goal_mask`.

The goal conditions every important editor decision:

1. operation distribution;
2. operation-conditioned tree pointer;
3. typed feature/comparator/route/threshold/window/priority arguments.

Append-only `actor_direction_*` heads predict pair-relative signed case delta, case improvement, loss-to-win and win-to-loss risk. The existing critic heads remain frozen.

## Exact-CUDA hindsight goal relabeling

Stage-B2 uses the fixed schema-2 dataset produced by `actor_cli.py prepare-data`. Each exact historical parent→child mutation is relabeled against every parent losing case using the mutation's real CUDA margins.

Goal classes are lexicographic:

1. safe loss→win;
2. safe target improvement;
3. unsafe improvement;
4. no effect;
5. target harm;
6. severe regression.

Only classes 1–2 receive positive imitation likelihood. AWR-style weighting is applied only after safety classification; regressions cannot be cancelled by aggregate gains. All relabels of one semantic mutation inherit the same train/validation side.

The current fixed full dataset contains 1,054,511 goal examples from 60,000 semantic mutations, with 156,010 safe-positive and 898,501 negative/unsafe/no-effect examples.

### Learned operation support gate

The default minimum is 32 safe-positive goal examples. On the current exact corpus that masks the learned actor for:

- `INSERT_RECOVERY` — zero support;
- `REPLACE_MACRO` — zero support;
- `SPLIT_LANE` — 20 safe-positive goal examples.

Grammar support remains in code for future active learning. The threshold and observed counts are persisted in the dataset manifest.

## Balanced goal-aware validation

The old positives-first `val[:2048]` evaluation is removed. GCAC-v2 deterministically stratifies held-out goal examples across target-positive, target-harm, no-effect, severe-regression and operation strata. Evaluation fails if required negative classes are absent.

Metrics include useful-vs-hard-negative actor likelihood, positive operation/target accuracy, target signed-delta correlation, target-improvement and loss-to-win calibration, and per-case regression calibration.

## Per-failure generation and directional shortlist

Fresh24 B371 generation is explicit per failure. The six requested losses are represented by their legal-state case tokens, candidates are generated round-robin across goals, then globally canonical-deduplicated. Candidate metadata preserves goal provenance for exact CUDA attribution.

The frozen critic's signed `case_delta` normalization is exactly:

`case_delta = tanh(raw_margin_delta / 5000)`.

GCAC-v2 no longer treats `mean(abs(case_delta))` as primary efficacy. LOSS_REPAIR shortlist priority is:

1. positive signed target direction / improvement confidence;
2. predicted target L→W;
3. low global and protected-win regression risk;
4. worst-win / margin dominance;
5. only then absolute effect, diversity and complexity tie-breakers.

Exact CUDA remains the only strategic truth.

## Lightning AI Stage-B2

Meaningful training stays on Lightning AI. The tracked runner is:

`tools/bkg4_search/bonsai_editformer/scripts/gcac_v2_cloud_run.sh`

The tracked bundle builder remains the established workflow entry point:

`tools/bkg4_search/bonsai_editformer/scripts/make_actor_lightning_bundle.sh`

The bundle ships the exact hashed schema-2 dataset, proven critic checkpoint, frozen feature caches, guarded fresh24 semantic context, B371 baseline, fresh24 tapes, current400 identity data and CUDA evaluator source. Cloud Stage-B2 keeps the proven critic/shared encoder frozen.

No champion/default/VM/ABI promotion is performed by this workflow.
