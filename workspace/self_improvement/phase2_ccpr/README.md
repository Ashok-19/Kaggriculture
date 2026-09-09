# Phase 2: Minimal CCPR

**PHASE 2 FAIL - MINIMAL CCPR NOT VALIDATED.**

Counterexample refinement improved BUILD evidence, but no CCPR-derived candidate
survived the frozen INTERNAL_DEV panel. The independent control retained one.
Dominant bottleneck: **generalization**, specifically small-BUILD predicate
separation that did not predict the effects of live activation on fresh games.
No further search, DEV tuning, fallback architecture, or Phase 3 was attempted.

Starting HEAD was `e5673f7f66817c82ae77486ae34ad653486fd5ca`, clean `main`.
The ending commit is the commit containing this report. Parent, historical gold
agents and champion assets are unchanged. Nothing was promoted, pushed or submitted.

## Bounded Design

The current user's narrower Phase-2 scope overrides the older master plan's
48,000-FGE comparison/replication and promotion requirements. Both tracked plans
and Phase-1 evidence were consulted. No separate local planning file was found
by the requested filename search in docs, workspace and local agent directories.

48 reachable roots: three fresh seeds (29091001-29091003), both seats, two
internal members, and checkpoints 335, 359, 575, 647. Members are unchanged
Phase-1 PARENT self-play and PARENT with fixed sell top-ups at 576-703. Parent
margins range from -1719 to +2542, including ties and near-losses. Checkpoints
cover daily boundaries around an earlier production window and late liquidation;
335 and 575 broaden the Phase-1 windows. Four checkpoints of one game are
correlated, not four independent trajectories. There are 12 distinct games.

Each of 12 initial lineages has a 48-turn body: sell top-up, market liquidation,
or value-order sorting, each at one checkpoint. Both arms have the same body,
duration and legal-predicate vocabulary and the same root observations. No body
was learned from a public opponent, and no downstream panel feeds proposal logic.

- CONTROL: five independent duration proposals, 48/24/16/8/4, fixed before results.
  This is a simple, useful Phase-1-informed control, not random harmful actions.
  Unlabeled threshold enumeration follows durations in its declared ordering but
  is not reached within five evaluations. Thus this is not an exhaustive comparison
  against all independent predicate-search schedules.
- CCPR: initial hypothesis plus at most four refinements. With mixed effects, rank
  single observable conditions by excluded activated non-beneficial roots, retained
  positive margin and complexity. Require at least two positive seeds and both
  seats. Permit at most two conjuncts. Reevaluate the actual compiled gate; a root
  truth table is only a proposal heuristic, never substituted for continuation.
  If no admissible WHEN proposal remains, shorten STOP, changing no other field.
- The incumbent uses W->L, W->T, T->L, L->W, T->W, margin, support and size in
  lexicographic order. Non-beneficial activated states include zero deltas.
  A smaller equivalent program is a ranking improvement but is not counted as
  improved terminal evidence in the lineage examples.

Both arms received the same five-evaluation ceiling per lineage. CCPR saturated
one lineage early: 55 candidates/660 continuations versus control 60/720.
Discovery suffix work was 219.6 versus 240 FGE. Common captures cost 48 full
executions and are included in the all-in ledger. No asymmetric free evaluation
cache was used: both initial arms were evaluated and charged separately.

## Results

| Metric | CONTROL | CCPR |
| --- | ---: | ---: |
| Candidate evaluations | 60 | 55 |
| Useful unique terminal-effect patterns | 7 | 10 |
| Safe candidates, including inactive/non-useful | 24 | 23 |
| Safe useful unique terminal-effect patterns | 5 | 8 |
| Legalized shortlist | 4 | 6 |
| Fresh BUILD survivors | 2 | 4 |
| INTERNAL_DEV survivors | 1 | 0 |
| Discovery L->W / T->W | 0 / 40 | 0 / 48 |
| Discovery W->T / W->L / T->L | 0 / 120 / 126 | 0 / 120 / 116 |
| All shortlisted DEV L->W / T->W | 0 / 22 | 0 / 18 |
| All shortlisted DEV W->T / W->L / T->L | 0 / 0 / 10 | 0 / 0 / 14 |

Effect-pattern counts are **not** counts of distinct strategic capabilities.
Search remained within three Phase-1 action mechanisms; safe gains came from two
market mechanisms. CCPR's higher BUILD count is not sufficient for a PASS.

Fresh BUILD used 29091101-29091103; DEV used 29092001-29092004, both seats and
both members. At most six distinct effect representatives per arm were frozen
before either panel. All were evaluated, including those rejected by fresh BUILD;
there was no DEV-driven alteration or rescue. CCPR had more shortlist opportunities
than control and still had zero survivors.

The control survivor is value sorting at steps 359-362, six BKG4 nodes, +88 fresh
BUILD margin and +176 DEV margin, 8 DEV T->W, no DEV W->L/W->T/T->L. This is a
known Phase-1-style effect, not evidence that Phase 2 created a new capability.

## Concrete Refinements

1. **Earlier top-up, 359:** the 48-turn initial body had 8 positive roots and 4
   activated non-beneficial roots. `MARKET_PRICE(WOOL) <= 1` retained 8 positives
   on two seeds/both seats, removed all four counterexamples, and changed net
   BUILD margin from +2760 to +388. The refinement deliberately trades aggregate
   gain for safety. Fresh BUILD +1252; DEV +844 but **4 T->L**, so rejected.
2. **Late top-up, 647:** shortening 48 to 24 turns changed net margin from -3388
   to +1228 and counterexamples from 12 to 6. Then `OWN_MONEY <= 74390` changed
   +1228 to +1616 and counterexamples from 6 to 2. This seven-node candidate was
   first in the frozen BUILD ranking. Fresh BUILD +690; DEV **-1886 and 2 T->L**.
3. **Late ordering, 647:** `MARKET_INVENTORY(WOOL) >= 10053` changed net -228 to
   +128 and counterexamples 6 to 2. Adding `MARKET_INVENTORY(TOMATO) >= 9919`
   retained +112 with zero BUILD counterexamples (8 positive roots, two seeds,
   both seats). The eight-node gate produced **zero value on fresh BUILD and DEV**.

These are actual evaluated child programs, not inferred improvements from labels.
There were 43 refinement evaluations and 460 activated non-beneficial case
instances across CCPR evaluations, with reuse/correlation across hypotheses.
Other lineages shortened losing liquidation bodies without producing useful
specialists; reducing a large loss is not counted as safe capability.

## Implementation And Correctness

One small Python module contains ordered proposals, counterexample extraction,
compact hypothesis memory, gate materialization, evaluation orchestration and
reporting. It reuses the existing BKG4 compiler and the exact Phase-1 body
implementations. No SpecialistProgram redesign or new action vocabulary was
needed: the existing stateless BKG4 override path is used. Conditions are checked
on current legal state on each active turn, not latched at a hidden snapshot ID.
This also means root separation alone cannot prove whole-window safety.

The Phase-1 native harness was extended, not duplicated:

- Existing simulator/two-controller snapshot copying and restore remain intact.
- Added a narrow adapter for existing BKG4 features and 12 root feature values.
- Added capture-only and disk-reuse modes, with binary/case/parent/route/arity
  identity and snapshot-file digest checks in the caller. Cross-context reuse is
  refused. Payloads remain ignored same-build artifacts, not a new archive format.
- No persistent specialist state is introduced. The repair gate has one output
  blackboard slot, no node state, 6-8 nodes and 4-48 turn maximum windows. PARENT
  ticks on every step, and is the exact inactive/terminated fallback.

Corrected fidelity, cached replay, every shortlisted legal program's CPU/CUDA and
fork/full integration, and reused Phase-1 tests all pass with **zero semantic
mismatches**. Four focused tests passed. Test simulation work is charged separately
in `summary.json`. No official-engine or public-cohort run was needed for this
failed architecture experiment; no release claim is made.

One initial fidelity fixture incorrectly compared a step-647 PARENT snapshot
against a full game whose repair could activate at 359. It correctly failed
fork/full comparison while CPU/CUDA agreed. The fixture was corrected to use
identical pre-activation prefixes before search. Its CPU/CUDA execution costs and
an earlier CPU run before sandbox GPU access was granted remain in `ledger.json`.
That invalid-comparator assertion is not concealed as a simulator bug or omitted
from compute. The first blocked GPU launch simulated no turns.

All-in work: **863,564 simulated turns / 1199.3944 FGE**, far below 9600. The
dominant uncertainty is now generalization, not throughput. More compute was not
spent after DEV falsified the claimed advantage. This experiment does not prove
all possible CCPR implementations fail; it falsifies this minimal frozen loop.

## Reproduce

Use `.venv/bin/python -m kaggriculture_bonsai.search.phase2_ccpr` from the repository
root. Compile the existing Phase-1 source into `workspace/build/phase2_ccpr/cpu`
and `cuda`, using the Phase-1 README's include paths, C++17/O2, CPU
`-ffp-contract=off`, and CUDA `--fmad=false -arch=sm_86` flags. Native versions are
g++ 11.4.0 and nvcc 11.5.119 on the RTX 3050 Laptop GPU.

Fresh experiment order: `freeze`, `fidelity`, `search`, `validate`, `integration`,
then focused tests and `summary`. Existing frozen search/shortlist artifacts are
protected from accidental overwrite. For an isolated reproduction, import the
module through `kaggriculture_bonsai`, set its `OUT` to a new `/tmp/` directory,
then call these same functions in order. Do not overwrite this DEV-exposed epoch.
Pass `PHASE1_TEST_EVIDENCE=<new OUT>/phase1_regression.json` when reusing
`tests/test_phase1_fork_repair.py`; point its ignored `cpu_gate`/`cuda_gate` binaries
at the rebuilt harness. Also run `tests/test_phase2_ccpr.py`.

`config.json`, `roots.json`, `hypotheses.json`, `lineages.json`, `legalized.json`,
`validation.json`, `integration.json`, `ledger.json` and `summary.json` contain
the frozen identities, actual paired outcomes, refinement ancestry and verdict.
`ACCEPTED_FOR_DEV` means selected before DEV, **not** accepted after DEV.

**Did counterexample refinement materially outperform independent proposal at
comparable budget? NO. Do not proceed to Phase 3.**
