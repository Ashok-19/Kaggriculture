# Permanent BKG4 Edit Search — Post-Independent-Validation Evidence

Date: 2026-08-22

Status of this document: **post-hardening validation evidence** for the permanent BKG4 edit-search infrastructure. This task does not promote a champion, submit to Kaggle, or train BonsaiEditFormer.

The independent review of commit `9e6e5d6e5b3dddd15cfc798ff3edc061ba125d43` returned `PERMANENT EDIT SEARCH NOT VALIDATED` despite the core recursive/CUDA architecture being real. The failures were correctness and validation failures around persistence, corpus integrity, safety semantics, benchmarking, escalation, typed input validation, MCTS wiring, and non-empty CPU↔CUDA translation.

All twelve blocker classes were hardened in place. The recursive/QD architecture, exact-CUDA evaluation path, CEGIS, lexicase selection, motif transfer, batching, and permanent random exploration floor were preserved.

## Blocker closure

| # | Independent blocker | Closure |
| --- | --- | --- |
| 1 | uint64 trace hashes / non-transactional Stage-2 corpus export | Explicit Arrow schemas use `uint64` for B4EC1 trace hashes and correct unsigned ABI types. Generation shards are staged, validated, atomically published, and the manifest is updated last. Orphans/missing shards fail validation. |
| 2 | tie→loss could enter production QD | Safety semantics now count any parent non-loss → child loss as a new loss. `LexicographicFitness` and archive insertion both enforce the invariant. |
| 3 | persisted corpus aggregates were trusted | Validator recomputes per-case labels, every persisted aggregate, `strict_safe`, exact-noop/behavior flags, and deterministic sample category from exact label rows. Any mismatch fails. |
| 4 | future-truth/model-feature leakage | The model boundary is a recursively enforced allowlist. Unknown declared model columns or unknown nested context keys fail. Outcome/search/provenance fields remain bookkeeping-only. |
| 5 | checkpoint/resume lost stagnation/search regime | `SearchRuntimeState` persists generation/export markers, proposal history, last proposal stats, stagnation counters/level, archive/QD/novelty/motifs/hard cases/certifications. Deterministic next-proposal equivalence is regression-tested. |
| 6 | old-vs-permanent benchmark used different roots | Historical shallow generation receives a benchmark-local injected `ModuleGenome` root. All arms assert the same semantic root before evaluation and report `same_root_parent=true`. |
| 7 | convergent EditPrograms inflated QD | Production QD, exploration QD, and novelty occupancy are keyed by resulting semantic policy state. Alternate program lineage may remain metadata but cannot create duplicate policy occupancy. |
| 8 | escalation level 1 was decorative | Level 1 reserves real wider threshold/window/route neighborhoods. Controlled identical-seed tests show changed proposals and greater window reach. |
| 9 | dangling QD references loaded | Archive load performs fail-closed referential/version/semantic integrity checks across QD, novelty, certifications, lineage, motifs, root/parent identity, grammar, and descriptor schema. |
| 10 | malformed typed feature indices were normalized | Canonical external EditPrograms and `ConditionGene`s reject illegal feature-specific indices, priority, route/prefix, target and condition-count semantics. Semantic validation never clips/modulos illegal input. |
| 11 | production MCTS quality map was empty | `SearchEngine` now derives MCTS quality only from compatible exact full-panel state certifications and passes it into production `EmitterContext`. Incompatible/stale panels are excluded; no-quality fallback remains exploratory. |
| 12 | only empty-overlay CPU↔CUDA parity existed | A permanent non-empty parity harness now feeds the same B4GM1 overlays and B4TP1 opponent tapes through B4EC1 CUDA and an authoritative CPU C++ BKG4/route/legacy/exploit/feed stack, comparing final money and action-trace hashes. |

Schema/grammar versions were bumped to prevent silently interpreting incompatible pre-hardening persisted state:

```text
SEARCH_STATE_SCHEMA_VERSION = 2
EDIT_PROGRAM_SCHEMA_VERSION  = 2
GRAMMAR_ID                   = bkg4_module_genome_edit_v2
EMITTER_REGISTRY_VERSION     = 2
QD_DESCRIPTOR_SCHEMA_VERSION = 2
CORPUS_SCHEMA_VERSION        = 2
```

## Fresh real two-generation exact-CUDA lifecycle

Fresh scratch root:

```text
scratch/bkg4_edit_search_validation/final_lifecycle_424260/
```

Parent:

```text
combined identity:
6359ab400f9d128945cdbba5df7178a33b88c643ddd9b488299d66e14daa6925

controller semantic hash:
8c5d7cd0e32574cbe349d5964946a9e2d6e7a75d766708f5d82308cd8507a034

root ModuleGenome semantic id:
766224c843a42274275e2e0ac332c2512ab1a85377365b8dd093309e11b81cf2
```

The fresh lifecycle used 16 real tape/seed/seat cases, legal allowlisted public model context, the standalone exact evaluator at `scratch/cuda-engine-build/bkg4_edit_corpus_eval`, 16 proposals/generation, a 4-case Stage-1 screen, 8-case full panel, and six exact Stage-2 finalists.

Generation 1:

```text
proposed:             16 / 16 unique
source parents:        1
operation families:    4
Stage-2 programs:      6
production niches:     4
exploration niches:    5
exact certifications:  7
exported generations: [1]
```

Immediate corpus validation after G1:

```text
export_valid:                  true
complete_generation_layout:    true
uint64_trace_schema:           true
aggregates_recomputed:         true
model_boundary_allowlisted:    true
programs:                      6
labels:                       48
reconstruction_failures:       0
referential_integrity_failures:0
```

The process-style resume then loaded the archive with `resumable=true`, restored G1's proposal/runtime state and exact certifications, and continued at generation 2.

Generation 2:

```text
proposed:              16 / 16 unique
source parents:         5
operation families:     6
quality panel:          final-lifecycle-424260/g1/full
exact quality states:   6
Stage-2 programs:       6
production niches:      8
exploration niches:    10
exact certifications:  17
exported generations: [1, 2]
```

Final corpus validation:

```text
export_valid:                   true
generations:                    2
programs:                      12
labels:                        96
parents:                       33
cases:                         16
uint64_trace_schema:            true
aggregates_recomputed:          true
model_boundary_allowlisted:     true
child_semantic_duplicate_rate:  0.0
exact_program_duplicate_rate:   0.0
reconstruction_failures:        0
referential_integrity_failures: 0
```

Final archive validation also returned `valid=true` with `[1,2]` in `runtime_state.exported_generations` and the G2 exact-quality panel/state count preserved.

## Fresh real-corpus corruption attacks

Copies of the final two-generation corpus were corrupted independently under:

```text
scratch/bkg4_edit_search_validation/final_corruptions_424260/
```

Every previously accepted corruption failed closed:

| Corruption | Result |
| --- | --- |
| program shard without corresponding label shard | rejected: orphan/missing label shards |
| label shard without corresponding program shard | rejected: orphan/missing program shards |
| unmanifested program shard | rejected: actual/expected program shard mismatch |
| manifest references deleted label shard | rejected: actual/expected label shard mismatch |
| `l2w_count` tamper | rejected: aggregate recomputation mismatch |
| `strict_safe` tamper | rejected: aggregate recomputation mismatch |
| `sample_category` tamper | rejected: aggregate recomputation mismatch |
| add declared `candidate_margin` model feature | rejected: model feature contract mismatch |
| nested `screening_rank` in `model_context_json` | rejected: unknown/non-pre-outcome context key |

The uint64 regression also round-trips values above `2^63-1` and `(2^64)-1` exactly in Parquet.

## Fresh fair equal-budget benchmark

All four arms use the same exact parent semantic identity and root truth:

```text
same_root_parent: true
root_truth_shared: true
root semantic id:
766224c843a42274275e2e0ac332c2512ab1a85377365b8dd093309e11b81cf2
```

Each arm receives exactly 64 children × 16 cases = **1,024 exact child×case CUDA evaluations** per seed. Three deterministic seeds were rerun with `--require-material-improvement`: `424251`, `424252`, and `424253`. All three exited 0 and passed all four material gates: safety, yield, recursive search, and structural diversity.

Permanent recursive versus historical old-shallow:

| Seed | Safe-useful /1k old → permanent | Strict-safe L→W /1k old → permanent | Useful diversity old → permanent | Source parents permanent | Max depth permanent | New-loss / W→L / severe permanent |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 424251 | 343.75 → **593.75** | 234.375 → **468.75** | 11 → **16** | 12 | 5 | 0 / 0 / 0 |
| 424252 | 343.75 → **625.0** | 203.125 → **453.125** | 11 → **16** | 12 | 5 | 0 / 0 / 0 |
| 424253 | 359.375 → **640.625** | 218.75 → **500.0** | 10 → **16** | 11 | 5 | 0 / 0 / 0 |

The permanent arm covered root/rule/condition targets on every seed. It reached 10 operation families on seed 424251 and all 11 historical shallow operation families on 424252/424253. The benchmark no longer relies on the invalid pre-hardening 5×/4.4× headline.

The benchmark exposed and caused one additional search-quality fix during hardening: one-slot recursive emitter allocations were starving later operation families because typed candidates were globally operation-sorted. Candidate selection now uses a bounded deterministic operation+target-family round-robin, preserving legal grammar diversity even when many parents receive one proposal slot. This is permanent proposer behavior, not benchmark-only filtering.

## Anti-saturation / escalation evidence

Two different questions are validated separately.

### 1. Does the final exact-CUDA search collapse under a narrow neighborhood?

A fresh 8-generation exact-CUDA stress used only eight legal public observation rows, 16 exact full-panel cases, and 24 candidates/generation. It **did not** trigger automatic escalation because it did not meet stagnation criteria:

```text
unique candidates:       24 / 24 every generation
source parents:           10–14 after G1
operation families:        9–12 after G1
underfill:                 0.0 every generation
duplicate rate after G1:   0.107–0.500
archive states after G8: 193
production niches:       141
exploration niches:      154
```

`--require-escalation` correctly returned nonzero for this healthy run. Escalation is not forced merely to satisfy a report.

### 2. Does escalation alter real search behavior when stagnation exists?

A fresh deterministic plateau oracle (`seed=424271`) removed progress/novelty while still exercising the real permanent proposer/archive/stagnation machinery. With patience 1, the monitor escalated once per generation through all six regimes:

```text
G1 → level 1: wider scalar/window/route neighborhoods
G2 → level 2: broader QD parents and structural mutation
G3 → level 3: semantic recombination across distant niches
G4 → level 4: large-neighborhood destroy/repair
G5 → level 5: deeper multi-edit evolvability/MCTS
G6 → level 6: novelty/rare niches and hard-case refresh
```

The run still filled 20/20 unique proposals every generation and expanded source-parent count from 1 to 15.

A separate identical-parent/identical-generation/identical-seed comparison (`generation=7`, `seed=515151`) showed level 1 is behaviorally real:

```text
level-0 proposals: 64
level-1 proposals: 64
overlap:           37
level-0 only:      27
level-1 only:      27
max window distance: 372 → 575
```

Level 1 also changed route/operation coverage (`CHANGE_PRIORITY` appeared at level 1 in this controlled sample). Therefore the regime transition is not a decorative counter change.

## Production MCTS exact-quality evidence

The final lifecycle supplies direct production evidence: generation 2 reports

```text
quality_panel_id:    final-lifecycle-424260/g1/full
quality_state_count: 6
```

Those are persisted exact Stage-2 full-panel state certifications from generation 1. Regression tests additionally prove:

- high/low exact quality changes MCTS branch preference,
- absent quality falls back to exploration,
- incompatible panels are ignored,
- save/resume reconstructs the same quality map.

No neural or future outcome score enters production MCTS selection.

## Fresh non-empty CPU C++ ↔ CUDA parity

Permanent command:

```bash
python3 -m tools.bkg4_search.edit_search.parity \
  --parent-dir bonsai-kg/integrations/bkg4_champion/champions/w33_s172_expansion_v1 \
  --cases-tsv scratch/bkg4_edit_search_validation/heldout16_cases7.tsv \
  --cuda-evaluator scratch/cuda-engine-build/bkg4_edit_corpus_eval \
  --work-dir scratch/bkg4_edit_search_validation/final_nonempty_parity_424260 \
  --case-count 4 \
  --report scratch/bkg4_edit_search_validation/final_nonempty_parity_424260/report.json
```

The harness uses three legal, structurally distinct, **non-empty** ModuleGenomes. Opponent tapes are normalized once into B4TP1 and the same bytes feed CPU and CUDA. The CPU path reuses the authoritative BKG4 VM, C++ simulator, extended route pack, frozen legacy overlay semantics, engine exploits, and feed reconciler; it is not a second Python simulator.

Fresh result:

```text
non-empty overlays:     3
cases:                  4
full comparisons:      12
CPU faults:             0
CUDA faults:            0
CPU↔CUDA mismatches:    0
CUDA action changes: 4140
CUDA activations:     8480
B4EC1 bad:              0
```

The nonzero action-change/activation counts prove this is not an empty/inert-overlay parity test.

The original base-champion native differential parity was also rerun:

```text
cases:              2
VM records:      1438
C++ mismatches:     0
CUDA mismatches:    0
```

## Fresh regression gates

Final commands and results before commit:

```text
python3 -m unittest discover -s tools/bkg4_search/edit_search -p 'test_*.py' -v
→ 54 / 54 passed

python3 -m unittest discover -s tools/edit_corpus -p 'test_*.py' -v
→ 59 / 59 passed

python3 bonsai-kg/tests/test_bkg4_search.py
→ 4 / 4 passed

python3 -m compileall -q tools/bkg4_search/edit_search
→ exit 0

uv run --with kaggle-environments==1.32.7 python \
  bonsai-kg/tests/bkg4_champion_native_parity.py --seeds 1 \
  --assets bonsai-kg/integrations/bkg4_champion/champions/w33_s172_expansion_v1
→ 2 cases / 1,438 VM records / 0 C++ mismatches / 0 CUDA mismatches
```

`git diff --check`, reviewed local commit, and final clean-status/HEAD are terminal repository gates and are recorded in the completion report rather than hard-coded here before the commit exists.

## Promotion status

No champion asset was modified or promoted. No Kaggle push/submission occurred. Therefore the official Python 64/128-game **promotion canary** is intentionally not run in this infrastructure-hardening task.

When a concrete searched candidate is selected for champion promotion, the mandatory sequence remains:

1. exact CUDA certification,
2. non-empty CPU C++ translation/parity of that exact candidate,
3. 64/128-game official Python-engine canary,
4. only then serialized champion promotion/submission.
