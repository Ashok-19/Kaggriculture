# BonsaiEditFormer-GCAC-v2 — Goal-Conditioned Directionality Design

Date: 2026-08-20

## Problem being fixed

BonsaiEditFormer-ActorCritic-v1 already learned valid typed BKG4 editing, while the frozen BonsaiEditFormer-5M critic remained useful for regression/effect screening. The decisive fresh24 failure is directionality: under an equal 64-candidate CUDA budget, Actor+Critic produced zero candidates that improved even one of B371's six baseline losses. The current actor is conditioned on a generic fused parent/context representation, and the current shortlist collapses signed per-case `case_delta` predictions into `mean(abs(case_delta))`. It therefore answers "will this edit matter?" better than "will this edit improve this requested failure?"

The fix must preserve the proven structural encoder, critic heads, typed grammar, CUDA truth pipeline, and frozen Stage-B workflow. It must add the missing variable: a semantic repair goal.

## Research mapping

Primary references reviewed before implementation:

- Schaul et al., **Universal Value Function Approximators**, ICML 2015. A single approximator conditions on both state and goal and can generalize to unseen goals. Mapping: represent a target repair as a semantic goal embedding derived from legal case-state tokens; condition every actor decision on it rather than learning a categorical case identity.
- Andrychowicz et al., **Hindsight Experience Replay**, 2017. Off-policy experience can be relabeled with goals that were actually achieved. Mapping: relabel each exact-CUDA parent→child mutation by the failure cases whose margins it actually improved, left unchanged, or harmed. Never invent a repair label.
- Ghosh et al., **Learning to Reach Goals via Iterated Supervised Learning / GCSL**, ICLR 2021. Goal-conditioned behavior can be learned by supervised likelihood after relabeling trajectories with achieved goals. Mapping: keep Stage-B2 supervised and stable instead of introducing joint Actor+Critic RL; train `(parent, semantic failure goal) -> typed edit`.
- Peng et al., **Advantage-Weighted Regression**, 2019. Policy maximum likelihood can be weighted by off-policy advantage. Mapping: behavior-clone only lexicographically safe target-improving edits and weight them by a clipped exponential goal advantage; use target-specific harmful edits as negatives.
- Kostrikov, Nair & Levine, **Implicit Q-Learning**, 2021. Offline policy extraction should avoid relying on unconstrained out-of-distribution actions and uses advantage-weighted behavioral cloning. Mapping: mask learned edit operations with insufficient positive goal support during Stage-B2 rather than letting unsupported heads masquerade as experts.
- Dinella et al., **Hoppity: Learning Graph Transformations to Detect and Fix Bugs in Programs**, ICLR 2020. Program repair is decomposed into graph-location prediction and graph edits. Mapping: preserve the hierarchical `operation -> target pointer -> typed arguments` actor.
- Tarlow et al., **Graph2Diff**, 2019. Structured program/diagnostic graphs are mapped to diffs containing tokens and pointers to code locations. Mapping: combine the BKG4 structural representation with explicit failure diagnostics/goals, while retaining pointer-based tree localization.
- Mankowitz et al., **AlphaDev**, Nature 2023. A learned policy/value model guides search through legal program transformations, while exact correctness/performance evaluation remains authoritative. Mapping: retain legality pruning and use GCAC only to prioritize candidates for exact CUDA; do not replace the exact evaluator with neural reward.
- Bengio et al., **GFlowNet Foundations**, JMLR 2023, and Jain et al., **Multi-Objective GFlowNets**, 2022. Conditional GFlowNets are useful for diverse high-reward combinatorial generation. Mapping: defer them. Current evidence says directionality, not diversity, is the bottleneck; stochastic typed sampling and deduplication are sufficient for this iteration.

## Architecture

Architecture name: `BonsaiEditFormer-GCAC-v2`.

### Frozen proven path

The existing critic/shared path keeps the same parameter names, shapes, and outputs:

- structural tree Transformer and relative structural bias;
- edit-delta representation;
- legal-state case encoder and activation context;
- cross-fusion;
- `regression_logit`, `flip_logit`, `margin`, `worst_win`, `rank_residual`, `rank`, and signed `case_delta`.

Critic checkpoint import must remain exact. The Stage-B2 optimizer must update only parameters whose names begin with `actor_`. Shared encoder and proven critic parameters remain bit-for-bit frozen and digest checked.

### Semantic goal input

Each model batch gains:

- `goal_mask: bool[B, case_cap]`: marks one requested case or a semantic cluster among the existing legal case tokens;
- `goal_spec: float32[B, 4]`: `[loss_repair, margin_expansion, desired_direction, desired_margin_floor_norm]`.

No opponent/team/replay/seed/case-ID embedding is added. Case indices may exist only outside the model for joining labels and auditing.

The goal embedding is derived from already-encoded legal case tokens. A small trainable `actor_goal_*` attention-pooling adapter uses `goal_mask` plus `goal_spec`; it cannot see an arbitrary identity ID.

### Goal-conditioned actor

Use stop-gradient copies of frozen shared representations for the trainable actor path.

- Operation head: `P(op | frozen_fused, goal)`.
- Target pointer: query contains actor context + operation embedding + goal; keys remain frozen tree-token projections through trainable actor pointer layers.
- Semantic heads: condition on frozen fused context + goal + selected target token + selected operation embedding.

Thus goal information affects operation, location, and typed arguments rather than only a final score.

### Append-only directional case heads

Add small trainable `actor_direction_*` heads without changing any proven critic head:

- per-case improve logit `P(delta > 0)`;
- per-case L→W logit;
- per-case W→L regression logit.

These heads operate on candidate case representations plus the semantic goal embedding. They are trained against exact per-case CUDA labels using candidate/child feature batches, while the shared representation remains frozen. They are selection/calibration aids only; exact CUDA still decides truth.

## Goal-relabeled data

Keep the 60,000 inferred parent→child semantic mutations as the base transition table. Do not duplicate large action/target-mask tensors per goal. Store one transition table plus a compact goal-example table that maps each goal example to a transition.

For every base mutation, compare parent and child exact margins on their common case panel. Every relabel uses a parent case whose parent margin is negative. Raw target delta is `child_margin - parent_margin`.

Goal classes are lexicographic:

1. `SAFE_FLIP`: target becomes a win, with zero parent-win→loss and zero parent-win delta below -3000.
2. `SAFE_IMPROVE`: target delta > 0 with the same safety constraints.
3. `UNSAFE_IMPROVE`: target delta > 0 but a protected/current parent win is lost or suffers a severe regression.
4. `NO_EFFECT`: target delta == 0 and no severe safety event.
5. `TARGET_HARM`: target delta < 0 without a more severe safety event.
6. `SEVERE_REGRESSION`: any parent win becomes non-win or loses more than 3000 margin.

Only classes 1–2 receive positive imitation likelihood. Classes 3–6 supply target-specific contrastive/quality supervision. This prevents aggregate gains from cancelling regressions.

Positive likelihood weight is AWR-style but only *after* lexicographic safety gating. A configurable class bonus plus clipped target-delta magnitude defines goal advantage; `exp(advantage / beta)` is clipped to a configurable maximum. Negative weights are separately bounded and increase for target harm/severe regression. All formulas and hyperparameters are recorded in the manifest.

All relabels of the same parent→child semantic mutation inherit the same train/validation side. This prevents goal-A/train and goal-B/validation leakage for the same edit.

## Operation support gate

The current exact 60k corpus was re-audited on 2026-08-20. Goal-example safe-positive support includes:

- ADD_CONDITION 61,961
- ADD_SPECIALIST 3,127
- CHANGE_PRIORITY 43
- CHANGE_ROUTE 36,838
- CHANGE_THRESHOLD 22,001
- CHANGE_WINDOW 17,611
- COMPOSE_PREDICATES 434
- REMOVE_REDUNDANT_BRANCH 8,773
- REPLACE_SUBTREE 5,202
- SPLIT_LANE 20
- INSERT_RECOVERY 0 inferred transitions
- REPLACE_MACRO 0 inferred transitions

Default Stage-B2 minimum safe-positive support is 32. Therefore the learned actor masks INSERT_RECOVERY, REPLACE_MACRO, and SPLIT_LANE by evidence, not by name. The threshold is configurable and the grammar remains intact for future active learning.

## Balanced validation

Validation must be sampled deterministically from held-out mutation groups and must include, where data permits:

- safe target-positive examples;
- target-harm examples;
- target no-effect examples;
- severe W→L/reg3 examples;
- operation strata.

Evaluation fails loudly if required negative classes are absent. Metrics include positive op/target accuracy, useful-vs-hard-negative actor likelihood, goal target signed-delta correlation, improve classification, L→W classification, W→L classification, and exact class counts.

## Directional generation and shortlist

B371 generation is explicitly per fresh24 failure goal. The model sees semantic fresh24 case tokens, one `goal_mask` per requested loss, and no identity ID. Candidate metadata records which case bookkeeping index requested it so CUDA attribution is exact.

The existing `case_delta` target is exactly `tanh(raw_margin_delta / 5000)`. Directional scoring therefore preserves sign and can reconstruct an approximate raw delta as `5000 * atanh(clipped_prediction)` for diagnostics. It does not treat a historical `.15` absolute-effect threshold as a semantic constant.

LOSS_REPAIR shortlist order:

1. positive predicted target direction / improve confidence;
2. predicted L→W value;
3. low global and per-win regression risk;
4. worst-win / margin dominance;
5. only then absolute-effect activity, diversity, and complexity tie-breaks.

A high `|case_delta|` with negative target sign is never considered effective repair.

## Fresh24 goal context and alignment

Build a tiny ignored fresh24 information-state cache for the same 24 rows and step-264..287 probes used by B371. Its manifest binds:

- exact cases manifest/hash;
- exact B371 baseline result/hash;
- parent semantic identity;
- capture hash;
- probe schedule and BB width;
- the six loss bookkeeping indices.

Load-time guards reject case-count, probe-order, parent, or hash mismatch. IDs are bookkeeping only and are never embedded.

## Pass/fail experiment

Stay at 5M scale plus small adapters. Freeze the proven encoder/critic and train Stage-B2 on Lightning AI. Then repeat the six equal-CUDA-budget lanes. Minimum architecture repair is at least one Actor+directional-critic candidate with positive exact delta on the failure it was explicitly asked to repair. Strong pass is a fresh actor-generated L→W; preferred pass is strict-safe L→W with zero W→L and zero reg3.

Do not unfreeze or scale if explicit goal conditioning still yields zero positive target deltas. Diagnose grammar capacity, corpus support, goal representation, critic calibration, pointer reach, or need for multi-edit search first.
