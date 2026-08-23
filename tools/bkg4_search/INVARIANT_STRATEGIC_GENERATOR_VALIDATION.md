# Invariant Strategic Generator Validation

Validation date: 2026-08-23

All game labels below came from the resident exact CUDA evaluators. Search read only the
discovery bank. Certification was evaluated after the selected semantic ID was frozen, and
ranker validation was never passed to search or corpus generation.

## Generator Verdict

The B4IM generator passes its system gate. This does not promote the concrete candidate:
label-only certification rejected it, so no champion or official-engine campaign followed.

## Representation And Search

- ABI/grammar: `B4IM1/v1`, 256 bytes, at most 3 rules and 5 total guards per rule.
- Domains: `GLOBAL`, `EACH_OWN_UNIT`; controls: `INSTANT`, `COMMITTED`.
- Actions: route override, committed route, lifted unit rewrite, and market intent.
- Python and native validation reject unsupported domain/action/control pairs, global unbound
  inventory guards, noncanonical active fields, nonzero inactive payloads, exact
  high-cardinality production patches, exact non-mechanic steps, and concrete identities.
- Search uses ten identity-free mechanism goals, structural QD descriptors that ignore scalar
  jitter, a seven-level ladder, tape-group fitness, exact public-context threshold evidence,
  and real queued grouped CEGIS repair.
- Internal cross-fit searches one discovery partition, freezes top K, and labels the disjoint
  selection partition exactly once. Selection cannot mutate search state.

## New Strategic Edit

The selected two-rule discovery edit is:

1. When public wool price is at most 192, use parent route 70.
2. On day 1, when own money is at least 20, use parent route 212. This later rule has overlap
   priority and removes the broad day-one route's exact resource failure.

On all 16 discovery tape groups it produced 16 safe, 15 behavior-changing, 11 useful, and
one L->W group. Mean group delta was +2024.15625, worst delta was -2531, and there were zero
new-loss or severe groups. The rules contain no tape, seed, replay, team, or opponent identity.

## Five Outer Splits

Command surface: `python3 -m tools.bkg4_search.validate_invariant_generator` with seed 91001,
64 candidates, one round, frozen top 8, and 25% internal selection.

| Fold | Internal transferable | Outer safe-useful | Outer production | Best useful | Best L->W |
|---:|---:|---:|---:|---:|---:|
| 0 | 4 | 4 | 0 | 1 | 0 |
| 1 | 4 | 4 | 4 | 3 | 0 |
| 2 | 4 | 4 | 4 | 2 | 1 |
| 3 | 4 | 4 | 4 | 2 | 0 |
| 4 | 4 | 4 | 2 | 1 | 0 |

Result: 5/5 outer folds contained a selectable safe-useful B4IM candidate (required 4/5).
The same semantic elite transferred in all five folds; one outer fold contained L->W.

## Equal-Budget Benchmark

Each arm evaluated exactly 64 children on the same outer-fold train bank, froze its top 8,
and evaluated those eight once on the same tape-disjoint outer bank.

| Arm | Held-out safe-useful / 8 | Held-out production | Held-out L->W |
|---|---:|---:|---:|
| random typed legacy | 0 | 0 | 0 |
| fixed systematic legacy | 0 | 0 | 0 |
| permanent ModuleGenome | 0 | 0 | 0 |
| invariant strategic B4IM | 4 | 0 | 0 |

This does not claim B4IM production qualification on the small four-tape benchmark partition.
It establishes superior selectable held-out transfer at equal child budget.

## Certification

Frozen semantic ID: `8fdddf13638a44bafb361219264dbc04a7f32406974da585a919ac58458b0a4e`.

Label-only certification produced 13/15 safe groups, 6 useful groups, 2 new-loss/severe
groups, 0 L->W, mean delta -2908.53, and worst delta -26632. It failed production
qualification. The candidate was not promoted, the labels did not feed search/corpus
selection, and no official Python campaign was run.

## Exact Corpus

Fresh ignored artifact:
`scratch/bkg4_invariant_search/final_validation_v3/corpus/invariant_exact_cuda_corpus.jsonl`.

- 128 rows, 128 unique semantic IDs, 0 duplicates
- 41 constant-free structural skeletons
- 4 cross-fit-positive rows across 2 positive skeletons
- 2 discovery-boundary, 54 unsafe-negative, 41 inert-negative, 27 safe-negative rows
- exact `B4IC1/v1` labels and complete parent/controller/ABI/registry provenance
- certification and ranker-validation absent from labels and model input

The quality validator rejects duplicate semantics, false exact provenance, malformed source
classes, and any positive that is unsafe, unqualified, inert, or supported by fewer than two
cross-fit folds.

## Parity And Throughput

- Empty B4IM versus B4EC: 4/4 binary result payloads exact; zero changes/activations/faults.
- Active CPU C++ versus CUDA: 16/16 exact money, trace, flags, and activations; 5,947 action
  changes and 11,504 activations.
- Distilled ordinary CPU BKG4 versus CUDA: 16/16 exact for route override, commitment, lifted
  rewrite, and market intent.
- Python runtime executes all four distilled families while preserving incumbent feed slots
  5..17 and strategic route slot 19.
- CUDA contains only `sm_86` cubins. The durable run executed 16,373 games in 13,223.612 GPU
  ms (1,238 effective games/s across 29 launches), with zero evaluator faults.

## Regressions

- edit-search discovery: 212 passed
- existing edit-corpus: 59 passed
- Bonsai/BKG unit tests: 150 passed
- focused feed distillation: 4 games, 2,876 ticks, 20 rewrites, 0 mismatches
- B4IM schema/model tests cover exact AST-array-AST round trip, padding and leakage attacks,
  quality-label attacks, structural split isolation, and JAX CUDA init/apply
- native invariant semantics, overlay, and distilled runners compiled and passed
- modified Python files compile under their intended Python runtimes

Machine-readable evidence is retained in ignored scratch at
`scratch/bkg4_invariant_search/final_validation_v3/report.json`.
