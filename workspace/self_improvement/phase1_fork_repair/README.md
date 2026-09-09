# Phase 1: Exact Fork-and-Repair

**PASS under the bounded Phase-1 criteria. Stop here; no Phase 2 was implemented.**

Starting HEAD: `11e5543169510a935416e8e6f16f32201b0109e3`, clean `main`.
The final commit is the commit containing this evidence. Nothing was promoted,
pushed, or submitted. The parent, gold agents, and champion assets are unchanged.

## Result

- 12 distinct fresh internal discovery games, two checkpoints each, 24 roots.
  The simple harness regenerated each game for each root: 24 capture executions,
  explicitly charged rather than disguised as 12 executions.
- Checkpoints 359 and 647, just before daily/shop refresh and the final route
  transition. Parent margins ranged from -180 to +1733; these were contested
  games, not dominant wins. No public/HOLDOUT opponent supplied BUILD evidence.
- 60 bounded repairs (10 action effects x six durations) plus PARENT, yielding
  1,464 paired continuations. No early truncation or score proxy was used.
- Two supported repair mechanisms: sell-quantity top-up and sell-order
  prioritization. Both benefited 12 earlier roots, three seeds, both seats and
  both internal members. Ordering gains were only 11 margin per discovery case.
- Four frozen six-node BKG4 gates were legalized before fresh BUILD or DEV.
  All four retained positive net margin on both panels, with zero negative
  margin cases, W->T or W->L. They represent **two mechanisms, not four new
  strategic capabilities**.
- The first BUILD-ranked candidate, `legal_6_359`, tops up eligible product
  sales during steps 359-406. Fresh BUILD: +1272 aggregate margin, 12/12 positive,
  6 T->W. INTERNAL_DEV: +4584, 16/16 positive, 8 T->W. No L->W in these panels.
- All 1,464 matched uninterrupted interventions agreed with their forks.
  Naive work: 1,069,872 turns. Fork work: 333,480 turns. Savings: **68.83%**;
  all 15 candidates with any positive matrix entry were retained. Recorded
  process wall times were 10.87 versus 6.76 seconds, including table/setup costs.
- All-in work, including diagnostic reruns, CPU checks, fresh full-game panels,
  matched controls and 16 official-engine games: **1,813,608 turns / 2518.90 FGE**.

The raw discovery matrix was not safe: 1 L->W, 78 T->W, 0 W->T and 473 W->L.
Later-root top-ups often hurt. None of those harmful later gates was legalized.
Only frozen legal survivors carry the bounded zero-regression claim.

## Snapshot Contract

`native/cuda/src/phase1_fork_repair.cu` reuses the existing C++ `Sim` state and
CUDA `Snapshot`, `save_snapshot` and `load_snapshot`. It does not use the generic
kernel that resets controllers from observations. Each 49,488-byte snapshot owns:

- Simulator config, counters, terminal flag, both farms, every tile, worker,
  inventory and insertion order, market state, shop history and economic counters.
- Both complete `shop_router_0909::State` objects: blackboards, VM metadata,
  initialization/last-step/day fields, queue contents/counts/heads, advanced-sale
  amounts and due/suppression timing. This parent has 85 nodes, two blackboard
  slots and zero node-state slots. Layout checks reject other parent state shapes.
- RNG provenance is config seed plus day. The native engine reseeds its daily
  generator and consumes draws according to the *modified* state. No future
  shops, weeds or opponent actions are replayed. Device tables are rebound.

The disk round trip compares the complete payload bytes before restoring it.
These temporary files are same-build ABI payloads, not a cross-version snapshot
format. They stay in ignored build/temp directories. Tracked root identities use
logical field hashes, excluding struct padding and pointer addresses. Context is
the manifest plus seed, seat, member and checkpoint, not the hash alone.

Fidelity compares uninterrupted and restored terminal rewards and per-turn
logical simulator, controller and action fingerprints, on CPU and CUDA. The pure
observation adapter reads the captured simulator fields. Initial fixtures cover
three seeds, both seats, both members and checkpoints 23/71/143/359/647/705, including
modified actions and return to parent. A focused additional fixture covers route
12 on fresh BUILD. All comparisons had zero observed mismatches. Initial checks
were repeated after adding disk reload; both runs remain in the cost ledger.

No candidate-local persistent state is omitted: these repair bodies and their
gates are stateless. The gates use only a legal STEP range and existing BKG4
blackboard output. PARENT ticks normally on every step before an action override.
The native and Python integrations return its unmodified action outside the gate.
They never route on seed, seat, opponent identity, hidden state or snapshot ID.

## Integration And Limits

The Python integration reuses `ShopRouter0909Plus._repairs`; the native integration
reuses the parent's top-up implementation and mirrors its stable ordering repair.
No SpecialistProgram grammar/executor, production route pack, VM or database was
added. The allowed existing BKG4 override path was sufficient for these bodies.

CPU/CUDA legal integration passed. Sixteen pinned official 1.32.7 games matched
native terminal rewards, with all parent runtime states checked against an
independent parent on the same observations and 11,296 inactive/terminated fallback
actions checked exactly. Five focused pytest checks passed, including existing
parent shape/identity tests. This is not a release canary or promotion certificate.

The evidence supports a narrow primitive, not a new economic strategy or strong
opponent coverage. Discovery used route 0; fresh BUILD additionally reached route
12, and DEV route 4. Internal members were PARENT and a fixed early-top-up variant
of PARENT. Seats/checkpoints within one seed are correlated, not independent
samples. The improvements are mostly tie-breaking market effects; some improve
margin by reducing the opponent's money more than their own. No unseen-regression
rate, public-cohort strength, or first-place probability is inferred.

`report.json` is the machine-readable verdict and all-in ledger. `roots.json`,
`candidates.json`, `paired.json`, `ranking.json`, `legal_shortlist.json`,
`legal_build.json`, `dev.json`, `efficiency.json`, and the focused parity/official
artifacts retain terminal evidence only, not enormous per-turn traces.

No separate supplementary local fork/repair plan was found in the repository,
workspace, local agent directories or conventional user plan directories. Both
tracked first-place plans were read; the user's prompt controlled conflicts.

## Reproduction

Use the installed project virtual environment. From the repository root:

```bash
rtk mkdir -p workspace/build/phase1_fork_repair
g++ -std=c++17 -O2 -ffp-contract=off -x c++ -Inative/cpp/include -Inative/bkg4/include -Inative/bkg4/stateful native/cuda/src/phase1_fork_repair.cu -o workspace/build/phase1_fork_repair/cpu_gate
nvcc -std=c++17 -O2 --fmad=false -arch=sm_86 -Inative/cuda/include -Inative/bkg4/include -Inative/bkg4/stateful native/cuda/src/phase1_fork_repair.cu -o workspace/build/phase1_fork_repair/cuda_gate
```

Run the module `kaggriculture_bonsai.search.phase1_fork_repair` using
`.venv/bin/python -m`, with `--out /tmp/phase1-reproduction`, in this order:
`freeze`, `fidelity`, `matrix`, `full_matrix`, `shortlist`, `integration`,
`legal_build`, `dev`, `official`. Do not use RTK for GPU-running commands.
The runner refuses native work beyond 3,200 all-in FGE and verifies parent hashes.
Repeated evaluation artifacts get new names and remain charged. Reproduction with
the final source needs only one fidelity run, so it need not reproduce the
implementation-time extra fidelity expense. Compile time was not instrumented.

The focused tests run with:

```bash
PHASE1_TEST_EVIDENCE=/tmp/phase1-focused-test.json .venv/bin/python -m pytest -q tests/test_phase1_fork_repair.py src/kaggriculture_bonsai/agents/shop_router_0909/plus/test_plus.py::TestControllerShape src/kaggriculture_bonsai/agents/shop_router_0909/plus/test_plus.py::TestActiveCandidate
```

**Phase 2 is deliberately not implemented. No champion change is authorized by
this experiment.**
