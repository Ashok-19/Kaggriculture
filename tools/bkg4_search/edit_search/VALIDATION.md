# Permanent BKG4 Edit Search — Final Mixed-Safety Validation

Date: 2026-08-23

Status: **READY for large-scale exact-CUDA BonsaiEditFormer edit-corpus generation.**

This status does **not** mean every search-panel production niche generalizes to unseen cases, and it does not authorize champion promotion. The final disjoint holdout explicitly demonstrates why a separate promotion filter remains mandatory. No champion asset, Kaggle submission, or ranker training was performed in this hardening task.

## Parent and format identity

```text
parent combined identity:
6359ab400f9d128945cdbba5df7178a33b88c643ddd9b488299d66e14daa6925

controller semantic hash:
8c5d7cd0e32574cbe349d5964946a9e2d6e7a75d766708f5d82308cd8507a034

root ModuleGenome semantic id:
766224c843a42274275e2e0ac332c2512ab1a85377365b8dd093309e11b81cf2

SEARCH_STATE_SCHEMA_VERSION = 3
EDIT_PROGRAM_SCHEMA_VERSION  = 2
GRAMMAR_ID                   = bkg4_module_genome_edit_v2
EMITTER_REGISTRY_VERSION     = 3
QD_DESCRIPTOR_SCHEMA_VERSION = 2
CORPUS_SCHEMA_VERSION        = 3
```

## What the final hardening changed

The earlier post-validation implementation closed the original twelve correctness blockers (uint64 traces/transactions, tie→loss safety, aggregate recomputation, leakage, resume state, same-root benchmark, semantic QD dedup, real escalation, dangling refs, typed-index rejection, MCTS exact quality, non-empty parity). The final mixed-safety pass added the remaining production requirements:

1. **Authoritative CaseRegistry** — certifications/counterexamples must reference registered CaseBank cases and registered panel membership; ghost cases/panels fail closed.
2. **Cross-artifact transaction identity** — archive generation ledger, corpus manifest, immutable archive snapshot, and exact program shard are cryptographically tied. Recovery rolls back prepared transactions or reconstructs committed canonical state; generation-number copying is not accepted as repair.
3. **Root/source hard safety** — production requires root feasibility; source-relative safety is separately retained for supervision. Unsafe states compete in a repair population by distance to feasibility before reward.
4. **Production/repair/research separation** — exploration/novelty cannot silently become normal production parents or mask strategic stagnation.
5. **Mixed production CaseBank** — all-loss/random validation is rejected. Production requires losses, fragile/strong protected wins, both seats, and multiple strategy families.
6. **SearchGoals** — failure clusters use legal public R0 state only. Frame-coherent contrastive predicates, protected guards, temporal STEP localization, and counterexamples define specialized edit goals.
7. **Guard-safe contrastive behavior** — if public predicates cannot exclude protected guards, the contrastive emitter emits nothing rather than manufacturing a misleading specialist.
8. **Temporal specialists using existing ABI** — `ADD_SPECIALIST` + `ADD_CONDITION` represents three/four-predicate rules without changing B4GM1 CPU/CUDA layout.
9. **Multi-goal scheduling** — one parent's contrastive budget spans multiple failure goals, removing single-goal seed dependence.
10. **Contribution-aware emitters** — exact safe-useful/archive contribution changes later budgets while preserving a permanent random floor. Contrastive capacity cannot consume generic refill.
11. **Active Stage-1 recall governance** — detection recall is separate from Stage-2 selection recall; periodic full audits can recover missed candidates into the constrained archive; screen/Stage-2 budgets widen after recall misses.
12. **Persistent goal caching and bitset scorer** — frame-coherent conjunction search is cached per `(CaseBank identity, goal_id)` across rounds; cache identity changes when public bank contents or goal/counterexamples change.
13. **Shared typed pools** — repeated emitter/parent typed-neighborhood construction no longer dominates the simple/root-heavy proposal path.

## Mixed discovery and sealed holdout panels

The final search bank is:

```text
scratch/bkg4_edit_search_validation/v3_mixed_panels/rich_search64.jsonl
identity: fe9999bc19df1b36567ed734119b700a4992f9d76ac8b786e2029dd6861d8ead
cases: 64
losses: 20
fragile wins: 16
strong wins: 28
protected wins: 44
public probes/case: 12 (steps 264..286, even steps)
```

The untouched holdout is:

```text
scratch/bkg4_edit_search_validation/v3_mixed_panels/rich_holdout48.jsonl
identity: e5abbeaec9658dad3172d04ad2439302fd810b2fc02489672379669dc6f2b287
cases: 48
losses: 14
fragile wins: 10
strong wins: 24
search/holdout case overlap: 0
```

The public context is aligned with the current140 parent capture and contains only legal registry features; hidden replay truth is not used by SearchGoal synthesis.

## Direct temporal-specialist exact-CUDA proof

Before trusting recursive search, the new mechanism was isolated from scheduler noise. One deterministic real failure goal generated 96 route/timing specialist variants and exact CUDA evaluated all 96 against the full mixed search panel.

```text
candidates:                 96
hard-feasible:              96 / 96
safe-useful:                26
elite-tier:                 20
programs with L2W:           7
new losses:                  0
W2L:                         0
severe regressions:          0
best L2W:                    2
best mean margin delta: +1317.65625
protected damage on best:    0
```

This is the key evidence that the contrastive subsystem is not merely producing valid/random mutations: temporally localized legal edits can generate exact safe strategic gains.

## Six-generation persistent exact-CUDA lifecycle

Scratch lifecycle:

```text
scratch/bkg4_edit_search_validation/v3_rich_search_424320/
```

Every generation proposed 64 unique candidates and periodic audit evaluated all 64 on the full 64-case panel. The archive/corpus transaction ledger contains exactly generations 1..6.

Strategic progression:

| Generation | Effective screen / Stage-2 | Production niches after round | Production insertions | Repair insertions | Safe-useful | Source parents | Operation families | Detection recall useful / L2W / danger |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 16 / 20 | 0 | 0 | 37 | 0 | 1 | 6 | 1.00 / 0.944 / 1.00 |
| 2 | 16 / 20 | 5 | 5 | 16 | 1 | 2 | 7 | 1.00 / 0.882 / 1.00 |
| 3 | 24 / 30 | 36 | 33 | 23 | 9 | 8 | 11 | 1.00 / 1.00 / 1.00 |
| 4 | 24 / 30 | 75 | 41 | 20 | 11 | 16 | 11 | 1.00 / 1.00 / 1.00 |
| 5 | 24 / 30 | 111 | 40 | 17 | 14 | 15 | 11 | 0.50 / 1.00 / 1.00 |
| 6 | 36 / 45 | 154 | 43 | 16 | 13 | 16 | 11 | 1.00 / 1.00 / 1.00 |

G2's L2W recall miss widened the funnel to 24/30. G5 later exposed a useful-recall miss and widened it again to 36/45. G6 restored all three detection recalls to 1.0. This is active governance, not a diagnostic-only metric.

The archive after G6 contains:

```text
states:                 385
programs:               384
production niches:      154
exploration niches:     324
motifs:                   39
exact certifications:   446
exported generations: [1,2,3,4,5,6]
```

## Final exact corpus integrity and supervision distribution

Strict `validate_export` on the six-generation corpus returned:

```text
generations:                     6
programs:                      384
labels:                     24,576
parents:                       385
cases:                          64
complete_generation_layout:    true
aggregates_recomputed:          true
model_boundary_allowlisted:    true
uint64_trace_schema:            true
reconstruction_failures:        0
referential_integrity_failures: 0
child_semantic_duplicate_rate:  0.0
exact_program_duplicate_rate:   0.0
```

Operation coverage spans all current relevant families in this lifecycle, including `CHANGE_PRIORITY` once recursive multi-rule parents exist.

Persisted source-relative sample categories:

```text
hard_negative:        151
controlled_noop:      119
safe_incremental:      74
strict_safe_useful:    40
```

Recomputing exact source-parent→child safety tiers splits the 40 useful positives into:

```text
STRICT_SAFE_USEFUL: 29
ELITE_MARGIN:        8
ELITE_FLIP:          3
```

Additional distribution facts:

```text
source-feasible programs: 233 / 384
programs with at least one L2W: 92 / 384
program lengths: 325 x1-edit, 54 x2-edit, 5 x3-edit
emitter families represented: 9 / 9
```

The corpus is intentionally contrastive: useful/elite positives coexist with boundary/hard-negative/no-op examples. It is not a corpus of uniformly winning candidates and should not be filtered into one-sided labels before ranker training.

## Disjoint held-out retention

After G6, every one of the 154 unique search-panel production states was exact-CUDA evaluated against the sealed 48-case holdout.

```text
search-panel production states: 154
held-out hard-feasible:           78
held-out safe-useful:             28
held-out elite:                    0
held-out L2W programs:             0
held-out hard-negative:           76
```

The strongest retained family has:

```text
held-out mean margin delta: +118.6875
held-out worst protected delta: 0
held-out protected-tail mean:   0
```

Interpretation: the search system produces genuine transferable positives, but **production-safe on the discovery panel is not equivalent to globally promotion-safe**. The 76 held-out failures are useful negative evidence for ranker/generalization work and are the reason champion promotion must keep a disjoint exact holdout and official-engine canary.

## Equal-budget mixed proposer benchmark

The final multi-goal search semantics were compared with historical shallow, random typed, and fixed systematic mutation. Every arm used the same root and 64 candidates × 64 mixed cases per seed. Three independent deterministic seeds passed the material-improvement gate.

Permanent vs historical shallow:

| Seed | Old new-loss rate | Permanent new-loss rate | Old top-quartile W2L | Permanent W2L | Permanent safe-useful /1k | Permanent strict-safe L2W /1k | Permanent max depth | Permanent source parents |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 424340 | 0.796875 | **0.46875** | 0.1875 | **0.0** | 125.0 | 0.0 | 3 | 6 |
| 424341 | 0.828125 | **0.50000** | 0.3125 | **0.0** | 0.0 | 0.0 | 4 | 6 |
| 424342 | 0.812500 | **0.53125** | 0.2500 | **0.0** | 140.625 | 125.0 | 3 | 4 |

All 3/3 runs passed safety, yield, recursion, and diversity gates. The benchmark is intentionally reported without hiding the zero-yield seed: search remains stochastic/deterministic-seed dependent, but it is materially safer and structurally richer than the old shallow generator, and two seeds produced useful positives where the old/random/fixed arms produced none.

## Host proposal throughput

The original validator measured the 33-state permanent proposer at about 3.22 unique/s for a 64-candidate request. Shared typed pools on the same workload measured:

```text
64 candidates:  20.73 unique/s
128 candidates: 27.02 unique/s
256 candidates: 51.65 unique/s
```

The richer 12-probe contrastive scorer initially took 16.65 s for one real SearchGoal. Frame bitsets reduced that to 1.41 s (~11.8×). Persistent `(CaseBank identity, goal_id)` caching removes repeated synthesis across rounds.

On the final 193-state rich archive, a cold/warm 64-proposal pair measured:

```text
cold: 3.22 unique/s, 14 source parents, 12 operation families
warm: 4.88 unique/s, 19 source parents, 11 operation families
```

The rich archive deliberately pays more host cost for multi-parent structural diversity; exact game truth remains batched CUDA.

## Fresh non-empty CPU C++↔CUDA parity

Final command used three structurally distinct non-empty overlays and four real current140 cases.

```text
non-empty overlays:      3
cases:                   4
comparisons:            12
CPU faults:              0
CUDA faults:             0
CPU↔CUDA mismatches:     0
CUDA action changes:  5800
CUDA activations:     8480
B4EC1 bad:               0
```

The nonzero action changes/activations prove this is not inert-overlay parity.

The current parent native differential parity was also rerun through official Python-generated public observations:

```text
cases:              2
VM records:      1438
C++ mismatches:     0
CUDA mismatches:    0
```

## Fresh regression gates

```text
python3 -m unittest discover -s tools/bkg4_search/edit_search -p 'test_*.py' -v
→ 93 / 93 passed

python3 -m unittest discover -s tools/edit_corpus -p 'test_*.py' -v
→ 59 / 59 passed

python3 -m unittest -v bonsai-kg/tests/test_bkg4_search.py
→ 4 / 4 passed
```

The terminal `compileall`, `git diff --check`, reviewed local commit, and clean-status checks are performed after this document update so their results are not fabricated in advance.

## Readiness verdict

**Corpus generation: READY.**

The system now supplies the properties required for large-scale BonsaiEditFormer data generation:

- exact CUDA labels and CPU/CUDA translation parity;
- hard safety constraints plus useful repair/negative supervision;
- mixed safety panels rather than all-loss validation;
- legal public failure-directed contrastive edits with temporal localization;
- recursive multi-parent structural search and motif/recombination reuse;
- active Stage-1 recall governance;
- crash-consistent cross-artifact transactions and strict corpus recomputation;
- diverse positive, negative, boundary/no-op supervision rather than random valid mutations only;
- disjoint holdout evidence that quantifies generalization instead of assuming it.

**Champion promotion: NOT AUTOMATICALLY READY.** A concrete candidate must still pass a disjoint exact holdout, CPU/CUDA parity as applicable, and the mandatory 64/128-game official Python-engine canary before any ReCurSiON champion/submission change.


# Terminal certified-elite generator validation — 2026-08-23

This section supersedes older edit-search readiness measurements above. It validates the permanent schema-v4 search/corpus path against the unchanged BKG4 parent:

```text
parent identity:     6359ab400f9d128945cdbba5df7178a33b88c643ddd9b488299d66e14daa6925
controller hash:     8c5d7cd0e32574cbe349d5964946a9e2d6e7a75d766708f5d82308cd8507a034
root ModuleGenome:   766224c843a42274275e2e0ac332c2512ab1a85377365b8dd093309e11b81cf2
```

No champion asset was modified during this validation.

## Multi-temporal evidence roles

The final search uses 35 searchable/training cases (25 discovery + 10 rotating certification-repair), 10 disjoint held-out certification cases, and 16 frozen ranker-validation cases. Public context spans 22 probe steps from 48 through 672 across early, early-mid, mid, late-mid, and late strategic windows. Certification and frozen validation cases are not materialized as model-input rows.

## Real CUDA CEGIS

A bounded real B4EC1 repair campaign on promising unsafe discovery candidates produced:

```text
selected unsafe sources:      10
sources reaching feasible:    10 / 10
exact repair attempts:       312
less-infeasible descendants: 311
feasible descendants:        185
safe-useful descendants:      27
```

Later recursive production generations also produced repair→feasible and repair→elite transitions. CEGIS therefore no longer stalls at “less infeasible”; it reaches exact root feasibility on real CUDA truth.

## Five-seed equal-budget discovery benchmark

Fresh seeds (none are from the prohibited historical seed set):

```text
961101  961113  961127  961139  961151
```

Each proposer received the same root, the same rich 64-case discovery CaseBank, 64 candidates, and 4096 exact candidate×case CUDA evaluations.

```text
                         safe-useful/1k   median   elite/1k   safe L→W/1k   new-loss   W→L     severe
old shallow                    6.25        0.00      3.13          3.13       80.00%   20.00%   79.69%
random typed                    9.38        0.00      6.25          3.13       40.94%    0.00%   41.25%
fixed systematic               0.00        0.00      0.00          0.00       62.19%    0.00%   62.19%
permanent recursive          156.25      140.63    156.25        150.00       39.69%    0.00%   39.69%
```

Permanent safe-useful yield by seed:

```text
140.625  125.000  171.875  203.125  140.625
```

Permanent beats random typed on safe-useful yield in 5/5 seeds. The final safety-aware scheduler/refill accounting reduced permanent raw new-loss from the earlier 61.25% measurement to 39.69%, slightly below random typed at 40.94%. The aspirational 0.75×-random raw-proposal target was not reached; this is not used to weaken feasibility rules. Production admission still requires exact root feasibility through the audited multi-fidelity funnel.

## Held-out certification cohort

Across 136 discovery elites exported in the final four-generation corpus:

```text
mean strict-safe retention:     99.26%
mean activation retention:      58.09%
mean behavior retention:        58.09%
mean safe-useful retention:     48.90%
mean elite retention:           40.44%
certified positives:           125 / 136
certified elites:              109 / 136
```

Certification classes among those discovery elites:

```text
certified_elite_margin:        109
certified_safe_useful:          16
generalization_boundary:         9
generalization_hard_negative:    2
```

The cohort clears the 80% strict-safety, 50% behavior, and 40% safe-useful transfer gates. These held-out elites are margin elites; held-out L→W retention is zero and is not relabeled as a flip.

## Audited production funnel and throughput

Final G4 used a balanced 34/35 screen:

```text
target losses:       16
fragile wins:         8
strong wins:         10
counterexamples:      2
seat 0 / seat 1:  17 / 17
```

Corrected Stage-1 detection truth contains only programs that existed before Stage-1. CEGIS/certified-neighborhood descendants created later are not incorrectly counted as Stage-1 misses.

```text
useful recall:              1.000
L→W recall:                 1.000
danger recall:              1.000
target-repair recall:       1.000
selection useful recall:    1.000
selection L→W recall:       1.000
selection danger recall:    1.000
recall refresh required:    false
```

G4 walltime:

```text
proposal wall:          6.221 s
exact CUDA evaluation: 29.343 s
corpus transaction:     1.698 s
recovery:               3.562 s
round wall:            40.823 s
proposal fraction:      15.24%
unique proposals/s:     10.29
```

The permanent proposal path is below the <=25% round-wall requirement. Earlier mature generation profiling was reduced from roughly 30.2 s / 32 proposals to the current production rate through persistent SearchGoal cache, per-state MCTS typed-pool caching, bounded parent typed pools, and one immutable base `EmitterContext` per generation.

## Final high-information corpus

Strict `export-corpus` validation of the recovered G1–G4 corpus:

```text
programs:                         407
labels:                        28,490
cases materialized:                35
certified elites:                  128
certified safe-useful:              38
generalization boundaries:          15
generalization hard negatives:       2
near-feasible repairs:                6
local contrastive neighbors:         98
hard negatives:                     110
controlled no-ops:                   10  (2.46%)
reconstruction failures:              0
referential-integrity failures:       0
exact program duplicate rate:       0.0
child semantic duplicate rate:      0.0
```

The final corpus therefore contains dense certified-positive and local-contrast supervision, explicit transfer failures, repair examples, and a controlled no-op rate far below the 10% ceiling instead of the earlier hard-negative/no-op saturation.

## Real CUDA anti-saturation recovery

A low-budget 14-generation campaign on the mixed 64-case discovery panel used only eight proposals/generation with stagnation patience 1. Generations 1–6 produced zero safe-useful candidates and zero production niches. Quality escalation then activated:

```text
g7   level 1   wider contrastive semantic thresholds
g9   level 2   alternate failure clusters / broader QD parents
g10  level 3   stronger non-temporal guard / structural search
g11  level 4   stronger CEGIS / large-neighborhood repair
```

Recovery followed rather than continued novelty-only churn:

```text
g10 production niches: 1
g12 safe-useful: 1
g13 safe-useful: 2
g14 safe-useful: 2
final production niches: 12
```

This is real CUDA stall→escalation→safe-useful recovery evidence.

## Final parity

Expanded non-empty ModuleGenome parity now includes explicit temporal 3-condition and 4-condition rules in addition to the prior 1/2-condition and multi-rule overlays:

```text
non-empty overlays:       5
real cases:               4
comparisons:             20
CPU faults:               0
CUDA faults:              0
CPU↔CUDA mismatches:      0
CUDA action changes:   7838
CUDA activations:     11360
B4EC1 bad:                0
```

Official-Python-generated native differential parity was rerun on two seeds × both seats against the exact current parent assets:

```text
cases:                 4
VM records:         2876
C++ mismatches:        0
CUDA mismatches:       0
policy hash: 8c5d7cd0e32574cbe349d5964946a9e2d6e7a75d766708f5d82308cd8507a034
```

The missing `jsonschema` import required by the vendored Kaggle environment was supplied only through an ignored scratch-local dependency target; system Python and committed dependency metadata were not modified.

## Terminal regression matrix

Fresh commands after the final implementation changes:

```text
python3 -m unittest discover -s tools/bkg4_search/edit_search -p 'test_*.py' -v
→ 137 / 137 passed

python3 -m unittest discover -s tools/edit_corpus -p 'test_*.py' -v
→ 59 / 59 passed

python3 -m unittest -v bonsai-kg/tests/test_bkg4_search.py
→ 4 / 4 passed

python3 -m compileall -q tools/bkg4_search/edit_search
→ exit 0

git diff --check
→ exit 0
```

This validates the permanent edit-search/corpus generator. It does **not** promote or change the ReCurSiON champion. Any future champion/submission still requires the separate mandatory 64/128-game official-Python canary against current, historical, and own-history agents.
