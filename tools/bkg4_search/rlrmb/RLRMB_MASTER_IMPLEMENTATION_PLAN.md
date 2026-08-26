# ReCurSiON RLRMB Candidate Generation System Implementation Plan

> **For agentic workers:** Execute this plan phase-by-phase. Every phase has a hard validation gate. Do not skip forward because later work looks more interesting. Preserve the existing validated CEGIS/composer as downstream authorities until this plan explicitly integrates with them.

**Goal:** Build a clean, CUDA-first candidate-generation system that discovers genuinely new, coherent, transferable, legally routable Bonsai/BKG4 specialist capabilities at materially higher yield than the old donor-route × start × duration search.

**Architecture:** The new system is **Regret-Localized Reactive Macro Beam Search (RLRMB)**: regret/event-localized checkpoints feed a bounded Option-Program candidate IR; a hierarchical diversity-aware beam constructs programs; semantic Large-Neighborhood Search repairs promising candidates; an effect-vector quality-diversity archive prevents behavioral collapse; a legal-routability pre-screen reduces wasted CEGIS work; exact survivors are handed to the already-hardened CEGIS and existing automatic composer. Heavy game evaluation remains CUDA-first and the final deployed policy remains deterministic BKG4.

**Tech Stack:** Python orchestration, existing BKG4 compiler/materializer, existing CUDA Kaggriculture simulator, CUDA/C++ exact parity infrastructure, compact binary/JSON artifacts, `unittest`-style focused tests consistent with the current repository.

**Spec / strategic authority:**

- `tools/bkg4_search/BONSAI_FIRST_PLACE_MASTER_PLAN.md` — read-only strategic contract.
- `tools/bkg4_search/SYNTHESIZER_ARCHITECTURE.md` — current validated downstream synthesis path.
- `tools/bkg4_search/P0_6_CEGIS_FOUNDATION_HARDENING.md` — current CEGIS contract.
- P0.7 hardened Stage-A run `536dd7c9f4f5b393c99ccf3f` — empirical baseline: 265,226 CUDA games, 88 refined candidates, 0 accepted, 49 `MIXED_LEGAL_STATE`, 38 `NON_TRANSFERABLE`, 1 `TARGET_CAPABILITY_LOST`.
- Candidate-generation deep-research conclusion — build regret-localized reactive option-program search using hierarchical diverse beam + semantic LNS + effect-vector QD + routability pre-screen + CUDA batching.

## Global Constraints

1. Final runtime backbone remains deterministic Bonsai/BKG4.
2. Specialists are first-class; do not optimize for one globally dominant route.
3. Runtime routing must use legal current state only; no opponent/team/submission/replay/episode/tape identity, no future shop realization, no hidden unavailable opponent state, and no seed fingerprinting.
4. Large search/evaluation must run on CUDA. Avoid heavy CPU/RAM workloads.
5. CUDA candidate semantics must translate exactly to CPU C++ and BKG4 materialization semantics.
6. Before any actual champion promotion/submission, run a bounded 64/128-game official Python-engine canary. Do not use the official Python engine for large search.
7. Preserve the validated P0.6 CEGIS acceptance contract: target capability, independent transfer, frozen sealed holdout, exact legal-state ambiguity, threshold robustness, broad certification, parity.
8. Do not weaken safety gates to obtain an accepted specialist.
9. Do not revive candidate 50/route219 as a special case.
10. Do not turn this into EditFormer, corpus generation, PSRO, pure RL, runtime MCTS, or an arbitrary universal program synthesizer.
11. Heavy market ideas must account for whether the shared market also benefits the opponent.
12. Repository hygiene: every new permanent file is tracked; generated search outputs live under ignored `scratch/`; no new untracked junk.
13. Do not delete old evidence or historical reproduction files until the cleanup phase has proven they are redundant or safely archived.
14. The existing read-only master plan must not be modified.
15. Major gains only. Every phase must have a falsifiable strategic purpose.

---

# 1. Why this system is being built

The previous autonomous generator is validated as an orchestration/safety system but its candidate manifold is too narrow.

Current production search effectively performs:

```text
existing donor route
× fixed numeric start
× fixed numeric duration
→ copy donor actions verbatim inside the window
→ exact CUDA evaluate
→ adjust nearby start/duration
→ transfer
→ CEGIS
```

This cannot directly generate:

- novel strategic action sequences;
- multi-stage capabilities;
- semantic partial-route repair;
- multi-donor phase composition;
- event/state termination;
- an internal public-state branch;
- an abort/recovery path;
- a capability that safely waits for new public information before committing.

The hardened Stage-A result demonstrates that this is now the dominant bottleneck:

```text
265,226 CUDA games
88 refined attempts
0 accepted
49 MIXED_LEGAL_STATE
38 NON_TRANSFERABLE
1 TARGET_CAPABILITY_LOST
0 downstream composition/certification/parity failures
```

The donor bank also has substantial local behavioral duplication. Therefore raw donor file count is not a reliable diversity measure.

RLRMB changes **what CUDA is allowed to search** rather than merely spending more CUDA on the old manifold.

---

# 2. Definition of success

The new generator is validated only when it achieves all of the following.

## 2.1 Representation success

It can express, evaluate and materialize:

- 1–4 coherent strategic phases;
- parent/donor/semantic-primitive action sources;
- action-family masks;
- fixed-cap and event/state termination;
- deterministic return to parent;
- later, at most two bounded internal legal-state branches and explicit abort/recovery.

## 2.2 Search success

It can construct capabilities not present as one contiguous donor fragment.

## 2.3 Diversity success

Behaviorally equivalent candidates are deduplicated before expensive downstream work, and strategically distinct effect vectors survive even when they have lower aggregate margin than another candidate.

## 2.4 Routing success

Mixed-state evidence drives repair actions such as later initiation, common-prefix truncation, branch-at-reveal or abort/recovery rather than blindly trying another donor fragment.

## 2.5 Historical rediscovery success

The generator can rediscover behaviorally equivalent capabilities for the known route435, route410 and route437 specialist regimes without hard-coding their exact accepted route/start/gate tuples.

## 2.6 Equal-budget success

Against the old P0.7 Stage-A baseline at no more than 265,226 CUDA games on the same four frontier families, the first full RLRMB comparison must target:

- at least 1 autonomously accepted specialist;
- at least 6 target-L→W candidates per 100k CUDA games;
- `MIXED_LEGAL_STATE` among refined/CEGIS candidates ≤35%;
- `NON_TRANSFERABLE` ≤30%;
- unique effect signatures among refined candidates ≥85%;
- at least 3 distinct target-solving effect clusters across at least 2 families;
- 0 protected composed W→L;
- 0 CPU/CUDA semantic mismatches;
- 0 sealed-holdout leakage into generation.

If these targets are missed, diagnose which layer failed before scaling the budget.

---

# 3. Clean isolation boundary

All new Python-side RLRMB production code belongs under:

```text
tools/bkg4_search/rlrmb/
```

Do not scatter new `run_*`, `analyze_*`, `make_*`, `probe_*` or route-specific scripts directly into `tools/bkg4_search/`.

## 3.1 Planned directory structure

```text
tools/bkg4_search/rlrmb/
    __init__.py
    README.md
    RLRMB_MASTER_IMPLEMENTATION_PLAN.md

    schema.py
    failures.py
    canonical.py
    checkpoints.py
    motifs.py
    ir.py
    effects.py
    archive.py
    beam.py
    lns.py
    routability.py
    evaluator.py
    cegis_adapter.py
    materialize.py
    pipeline.py
    cli.py

    cuda/
        rlrmb_ir.cuh
        rlrmb_eval.cu
        rlrmb_io.hpp
        build.py

    configs/
        rediscovery_v1.json
        p0_7_equal_budget_v1.json

    tests/
        __init__.py
        test_schema.py
        test_canonical.py
        test_checkpoints.py
        test_ir.py
        test_effects.py
        test_archive.py
        test_beam.py
        test_lns.py
        test_routability.py
        test_evaluator_parity.py
        test_cegis_adapter.py
        test_materialize.py
        test_pipeline.py
        test_candidate50_regression.py
```

This is a target structure. A phase should create only the files it needs; do not create empty placeholder files for future phases.

## 3.2 External code RLRMB may reuse

RLRMB may import validated stable interfaces from outside its folder rather than copy them:

- `tools/bkg4_search/macro_initiation_cegis.py`
- `tools/bkg4_search/bonsai_synthesizer.py` acceptance/report helpers where a narrow adapter is sufficient
- `tools/bkg4_search/bonsai_composer.py`
- `tools/bkg4_search/information_state_io.py`
- `tools/bkg4_search/module_codec.py`
- `tools/bkg4_search/cuda_io.py`
- existing BKG4 compiler/materializer/build-assets path
- `cuda-engine/include/**` exact simulator/runtime headers

If a legacy helper is too entangled, wrap it. Do not copy large modules into `rlrmb/` just to achieve visual isolation.

## 3.3 Code RLRMB must not depend on for production search

Do not make the new production path depend on:

- route435/410/437 historical distillers/runners;
- `bonsai_editformer/**`;
- route-specific terminal JSON as production logic;
- old broad experimental scripts with implicit global state;
- old donor×start×duration `build_search_grid()` as the primary candidate constructor.

Historical artifacts may be loaded as test evidence only.

---

# 4. Core data contracts

These contracts should be implemented early and remain stable.

## 4.1 `FailureClass`

Machine-readable search outcomes:

```text
NO_HEADROOM
BAD_CHECKPOINT
BAD_HORIZON
OPEN_LOOP_FRAGILITY
MIXED_INITIATION
NON_TRANSFERABLE
DONOR_LIMITATION
REPRESENTATION_LIMITATION
TARGET_CAPABILITY_LOST
GATE_LIMITATION
FUTURE_DEPENDENT
SEARCH_BUDGET_EXHAUSTED
CUDA_FAULT
SEMANTIC_MISMATCH
```

Every rejection should state:

- candidate semantic ID;
- failure class;
- pipeline stage;
- evidence IDs;
- suggested next search operator when one is justified.

## 4.2 `Checkpoint`

Minimum fields:

```text
case/family reference
step
lane
reason(s)
legal state signature reference
regret/severity evidence
causal event type
recurrence support
both-seat support
```

Offline provenance may retain case IDs. Runtime candidate predicates may not use them.

## 4.3 `MacroProgram`

Initial v1 representation:

```text
MacroProgram
  lane_id
  initiation checkpoint descriptor
  1..4 phases
  max_total_steps
  deterministic parent fallback

Phase
  action_source
  source/motif reference
  action_family_mask
  termination
  parameter vector
```

Reactive v2 adds at most two internal decision points and explicit abort/recovery.

## 4.4 `EffectVector`

Minimum effect dimensions:

- exact target margin delta/sign vector;
- target L→W/W→L flags;
- micro-transfer delta/sign vector;
- money delta;
- market inventory/price deltas for material products;
- worker-count profile change;
- land change;
- crop/animal production-mix change;
- purchase/investment events;
- phase/termination class;
- action-change count.

Do not make every raw observation field an archive dimension.

## 4.5 `ArchiveCellKey`

Initial key:

```text
lane
× initiation/economic-state cluster
× effect cluster
× temporal phase
```

Within a cell, use safety-first lexicographic ranking rather than aggregate money alone.

---

# 5. Search objective

All search layers use the same high-level lexicographic ordering unless a phase explicitly tests another ordering:

1. semantic/runtime validity;
2. no W→L on the current protected/search panel;
3. target L→W;
4. required target positive support;
5. independent positive support;
6. both-seat recurrence;
7. worst active/target delta;
8. legal-routability / low mixed-state risk;
9. strategic effect novelty;
10. program simplicity;
11. aggregate margin only as a late tiebreaker.

Do not let one huge margin improvement dominate several newly created losses.

---

# 6. Artifact discipline

Permanent source/tests/configs/docs live under `tools/bkg4_search/rlrmb/`.

Generated runs live under:

```text
scratch/rlrmb/<run_id>/
```

Each material run should contain:

```text
config.json
run.json
command.txt
exit_code.txt
metrics.json
archive_summary.json
failures.json
artifacts/
```

Large binary CUDA results, state captures and traces stay under ignored scratch.

A permanent compact experiment result may later be appended to the existing `tools/bkg4_search/EXPERIMENT_LEDGER.md`, but RLRMB should not create another global ledger unless a real need emerges.

---

# 7. Phase map

| Phase | Deliverable | Primary proof | Scale permission |
|---|---|---|---|
| 0 | Isolated package contract + baselines | clean import/test boundary | no game search |
| 1 | Structural/effect canonicalization | duplicate old candidates collapse correctly | tiny stored evidence only |
| 2 | Regret/event checkpoint selector | known specialist checkpoints ranked from causal events | bounded stored traces |
| 3 | Option-Program IR v1 | deterministic round-trip + legality | no large games |
| 4 | Exact evaluator v1 | CUDA ↔ CPU semantics for IR primitives | small CUDA parity |
| 5 | Effect-vector QD archive | diverse effects retained, duplicates rejected | stored/synthetic evidence |
| 6 | Hierarchical diverse beam | constructs non-donor-equivalent programs | small CUDA search |
| 7 | Semantic LNS repair | repairs partial capabilities without random action corruption | small CUDA search |
| 8 | Reactive IR v2 | branch/abort/recovery exactness | bounded family0 probe |
| 9 | Routability pre-screen | classifies repairable mixed vs future-dependent | bounded exact state |
| 10 | Hardened CEGIS adapter | RLRMB candidate enters existing CEGIS unchanged | bounded integration |
| 11 | BKG4 materialization | candidate compiles exactly into Bonsai | CPU/CUDA/BKG4 parity |
| 12 | Historical rediscovery P1 | route435/410/437-like capability recovery | medium CUDA |
| 13 | Equal-budget P2 | beat P0.7 candidate-quality baseline | ≤265,226 CUDA games |
| 14 | Production autonomous pipeline | one clean CLI from weakness to accepted specialist | campaign-scale only after P2 |
| 15 | Deep repository cleanup | new path authoritative; clutter classified/removed safely | no strategic game search |
| 16 | Scale / successor Stage-A | new safely solved regimes per GPU-hour improves | large CUDA allowed |

No phase may silently skip its decision gate.

---

# Phase 0 — Isolated package contract and baseline freeze

## Objective

Create the clean namespace and freeze the comparison baseline without changing candidate semantics.

## Files

- Create `tools/bkg4_search/rlrmb/__init__.py`
- Create `tools/bkg4_search/rlrmb/README.md`
- Create `tools/bkg4_search/rlrmb/schema.py`
- Create `tools/bkg4_search/rlrmb/failures.py`
- Create `tools/bkg4_search/rlrmb/tests/__init__.py`
- Create `tools/bkg4_search/rlrmb/tests/test_schema.py`

## Interfaces

`schema.py` should define immutable run/candidate identity primitives and explicit dataset roles without importing old search orchestration.

`failures.py` owns the RLRMB failure taxonomy and suggested repair operator classes.

## Tasks

- [ ] Read current P0.7 run artifacts and record baseline metrics in `README.md`: 265,226 CUDA games, 88 refined attempts, 49 mixed, 38 non-transferable, 1 target-capability-lost, 0 accepted.
- [ ] Document the production boundary: RLRMB generates capability candidates; hardened CEGIS remains routing authority; composer remains composition authority.
- [ ] Define typed dataset roles: `TARGET`, `TRANSFER`, `VALIDATION`, `SEALED`, `CERTIFICATION`.
- [ ] Define stable candidate/run semantic-ID rules that exclude offline identity from runtime program content.
- [ ] Add fail-closed validation for unknown schema versions and illegal runtime identity fields.
- [ ] Add focused tests for deterministic canonical serialization and forbidden identity fields.
- [ ] Verify importing `tools.bkg4_search.rlrmb` does not import `bonsai_editformer` or historical route-specific scripts.

## Validation

Run only focused unit tests.

**Gate:** Phase 0 passes only if the new namespace is self-contained, deterministic and has no game-scale behavior change.

**Recommended commit:** `Create isolated RLRMB search foundation`

---

# Phase 1 — Structural and effect canonicalization

## Objective

Stop treating syntactically different but behaviorally equivalent candidates as independent strategic ideas.

## Files

- Create `canonical.py`
- Create `effects.py`
- Create `tests/test_canonical.py`
- Create `tests/test_effects.py`

## Required functions

Plan for interfaces equivalent to:

```text
canonical_program(program) -> canonical JSON-like structure
program_semantic_id(program) -> stable string
canonical_effect_vector(exact_results) -> EffectVector
effect_signature(effect_vector) -> stable string
```

## Tasks

- [ ] Implement structural canonicalization rules for source aliases, no-op masks, phase ordering, unused parameters and equivalent fixed/event terminations.
- [ ] Implement effect signatures from exact margin/economic effects; do not include opponent/replay identity.
- [ ] Build a read-only analyzer over the stored P0.7 candidate artifacts to reproduce the research finding that 88 attempts collapse to far fewer unique effect signatures.
- [ ] Build a donor-window dedup analyzer that groups locally identical action fragments by `(start, duration)`.
- [ ] Add tests showing exact whole-route or local-fragment duplicates collapse.
- [ ] Add tests showing strategically different effects remain distinct even if aggregate margin is equal.
- [ ] Ensure semantic ID changes when a material phase/action/termination changes.

## Decision gate

Phase 1 must demonstrate that the old search's nominal candidate count overstates strategic diversity and that the new canonicalization removes this redundancy deterministically.

Do not launch a new large search yet.

**Recommended commit:** `Add RLRMB candidate and effect canonicalization`

---

# Phase 2 — Regret/event checkpoint selector

## Objective

Replace uniform numeric start-grid dominance with causally meaningful initiation proposals.

## Files

- Create `checkpoints.py`
- Create `tests/test_checkpoints.py`

## Checkpoint proposal sources

A checkpoint may be proposed from:

- high negative baseline margin / regret peak;
- parent-vs-strong-policy action divergence;
- first irreversible investment decision;
- `BUY_LAND`, `HIRE`, animal/structure transition;
- shop reveal;
- day/EOD boundary;
- large market inventory/price change;
- production/payoff event;
- first observable divergence between positive and harmful effect trajectories;
- parent route/phase change.

## Tasks

- [ ] Define deterministic event extraction from stored legal trajectories/state captures.
- [ ] Rank checkpoints by recurrence, both-seat support, target severity/regret, causal novelty and proximity to actionable divergence.
- [ ] Include small before/at/after offsets around causal events where exact semantics justify them.
- [ ] Deduplicate checkpoints whose legal/economic states are equivalent for search purposes.
- [ ] Preserve lane labels such as market/liquidity, worker/production, land/investment, crop transition, animal transition, endgame/liquidation and mechanic/exploit.
- [ ] Add a fallback sparse numeric grid only when a target family has insufficient event evidence; the fallback must not become the dominant proposal source.
- [ ] Test against historical route435/410/437 evidence to verify their useful regions are reachable from the event/regret proposal set.

## Decision gate

At least the known historical specialist regimes must receive useful checkpoint proposals without hard-coding exact accepted steps.

**Recommended commit:** `Add regret and event checkpoint selection`

---

# Phase 3 — Option-Program IR v1

## Objective

Create the smallest new candidate language that can express coherent multi-phase programs without branches.

## Files

- Create `ir.py`
- Create `motifs.py`
- Create `tests/test_ir.py`

## v1 grammar

`MacroProgram`:

- exactly one lane ID;
- one initiation checkpoint;
- 1–4 phases;
- max total execution cap ≤ a validated bounded horizon;
- deterministic return-to-parent fallback.

`Phase` action source:

- `PARENT` — retain parent behavior;
- `DONOR_BLOCK` — use a donor/motif block;
- `SEMANTIC_SPLICE` — apply one coherent action-family replacement;
- `PARAMETERIZED_PRIMITIVE` — bounded strategy primitive supported by exact runtime semantics.

`Phase` action-family mask initially supports independent control of:

- farmer/unit movement/action family;
- hands/worker action family;
- market order family.

Do not invent dozens of fine-grained masks before evidence requires them.

`Termination` v1:

- fixed cap;
- next EOD;
- next shop reveal;
- purchase/investment completed;
- target public state threshold;
- parent-compatible state;
- max steps.

## Tasks

- [ ] Implement strict schema validation and canonical serialization.
- [ ] Define exact precedence when parent and override action families are combined.
- [ ] Define invalid-action behavior by inheriting official/existing engine semantics rather than creating new semantics.
- [ ] Define deterministic phase transition order.
- [ ] Define event termination using only legally observable current state.
- [ ] Implement motif extraction adapters for historical donor blocks without baking route IDs into the IR schema.
- [ ] Add round-trip tests: object → canonical document → object → identical semantic ID.
- [ ] Add rejection tests for >4 phases, invalid termination, identity predicates and unsupported action sources.

## Decision gate

IR v1 must be able to encode behaviorally equivalent forms of the three known historical specialist macro motifs while also representing at least one multi-phase program that is impossible as one contiguous donor fragment.

**Recommended commit:** `Define bounded RLRMB option-program IR`

---

# Phase 4 — Exact CPU/CUDA evaluator v1

## Objective

Evaluate Option-Programs directly and efficiently without translating every candidate back into thousands of host-side route files.

## Files

- Create `evaluator.py`
- Create `cuda/rlrmb_ir.cuh`
- Create `cuda/rlrmb_io.hpp`
- Create `cuda/rlrmb_eval.cu`
- Create `cuda/build.py`
- Create `tests/test_evaluator_parity.py`

## Architecture

The device evaluator should consume structure-of-arrays candidate buffers and evaluate conceptual `candidate × case` batches against the existing exact CUDA simulator.

Keep on device where practical:

- base route/action data;
- motif/donor blocks;
- program phase records;
- termination descriptors;
- compact parameters;
- case/tape data reused across the batch.

Return compact outputs:

- final margins/rewards;
- action-change count;
- L→W/W→L flags derivable host-side;
- selected economic effect summaries;
- phase/termination status;
- fault code.

Do not return full per-turn traces for every losing candidate.

## Tasks

- [ ] Implement a CPU/reference interpreter for IR semantics in Python or bounded C++ suitable for unit/parity tests.
- [ ] Implement a compact binary or fixed-layout RLRMB program format used identically by host and CUDA.
- [ ] Implement CUDA decoding/execution for IR v1.
- [ ] Reuse `cuda-engine/include/**` simulator semantics rather than reimplementing Kaggriculture physics.
- [ ] Add tiny golden parity fixtures covering each action source, each termination class and each action-family mask combination.
- [ ] Compare parent-only RLRMB program against unchanged parent result.
- [ ] Compare one donor-block RLRMB program against the existing fragment evaluator on the same `(start,duration)` behavior.
- [ ] Verify deterministic repeated runs.
- [ ] Ensure faults fail closed and carry candidate IDs.
- [ ] Profile only small batches at this phase; confirm batching reduces subprocess/file overhead relative to per-candidate orchestration.

## Decision gate

Zero semantic mismatches across golden fixtures and exact equivalence with the existing fragment evaluator for representations the old evaluator can express.

No medium/large campaign until this passes.

**Recommended commit:** `Implement exact CUDA RLRMB evaluator v1`

---

# Phase 5 — Effect-vector Quality-Diversity archive

## Objective

Preserve strategically distinct capabilities rather than one global score maximum.

## Files

- Create `archive.py`
- Create `tests/test_archive.py`

## Archive rules

Cell key starts as:

```text
lane × initiation cluster × effect cluster × temporal phase
```

Within a cell, compare candidates lexicographically by:

1. semantic validity;
2. W→L/harm;
3. target L→W;
4. required target positives;
5. independent support;
6. both-seat support;
7. worst delta;
8. routability score when available;
9. simplicity;
10. aggregate margin.

## Tasks

- [ ] Implement deterministic effect clustering with explicit versioning.
- [ ] Ensure candidates with identical effect signatures cannot fill multiple archive cells solely because their donor/source IDs differ.
- [ ] Preserve different target-response vectors even when syntax is similar.
- [ ] Track archive occupancy, novelty rate and replacement reason.
- [ ] Implement bounded per-cell capacity only if one elite per cell is empirically insufficient; default to one elite plus one optional repair candidate.
- [ ] Add serialization so archives can persist under `scratch/rlrmb/...` and resume deterministically.
- [ ] Test insertion-order determinism.

## Decision gate

On stored Stage-A evidence, the archive must retain strategically distinct candidates and collapse redundant donor-fragment variants in a way consistent with exact effect signatures.

**Recommended commit:** `Add effect-vector QD archive`

---

# Phase 6 — Hierarchical diverse beam search

## Objective

Make hierarchical beam the primary constructive search algorithm.

## Files

- Create `beam.py`
- Create `tests/test_beam.py`

## Search structure

```text
frontier weakness
→ ranked checkpoints
→ program seeds
→ one semantic expansion
→ large CUDA batch
→ target/effect evaluation
→ QD archive update
→ retain diverse partial programs
→ repeat until bounded phase/depth limit
```

## Seed sources

- parent-only skeleton;
- historical motif blocks that are legal at the checkpoint;
- action-family masked donor phases;
- parameterized primitives for proven strategic lanes;
- later, reduced-subgame teacher outputs.

## Expansion operators v1

- insert phase;
- replace phase source;
- shift phase boundary to adjacent event;
- change action-family mask;
- truncate at event;
- extend to event;
- mutate bounded numeric parameter;
- swap one semantic donor/motif phase.

No random per-turn action mutation.

## Tasks

- [ ] Implement deterministic beam expansion order.
- [ ] Evaluate each expansion batch in one/few CUDA invocations.
- [ ] Use structural dedup before CUDA and effect dedup after CUDA.
- [ ] Maintain per-lane/effect cells rather than one top-K scalar list.
- [ ] Enforce per-family CUDA budgets and search-depth caps.
- [ ] Stop an exhausted basin when archive novelty and target-quality improvement stall for configured rounds.
- [ ] Emit machine-readable expansion counts, dedup counts, CUDA games and archive changes.
- [ ] Add synthetic tests proving a lower-aggregate but strategically distinct candidate survives.
- [ ] Add a small real test proving beam can construct a two-phase program not present as a contiguous donor fragment.

## Decision gate

The beam must generate and preserve strategically novel multi-phase programs under a bounded CUDA probe without regressing evaluator exactness.

**Recommended commit:** `Implement RLRMB hierarchical diverse beam`

---

# Phase 7 — Semantic Large-Neighborhood Search repair

## Objective

Repair promising candidates by changing coherent strategic regions, not random actions.

## Files

- Create `lns.py`
- Create `tests/test_lns.py`

## Destroy operators

- one phase;
- one action family inside one phase;
- one event boundary;
- one numeric parameter group;
- one donor/motif source.

## Repair operators

Use the same structured beam constructor to rebuild the destroyed region from legal alternatives.

## Adaptive operator weights

Track operator outcomes by failure class:

- `NON_TRANSFERABLE` should prioritize mask/truncate/phase-replacement repairs;
- `BAD_HORIZON` should prioritize extend/truncate-to-event;
- `BAD_CHECKPOINT` should return to checkpoint search rather than mutate internal actions;
- `OPEN_LOOP_FRAGILITY` should later route to reactive Phase 8 operators;
- `FUTURE_DEPENDENT` receives no repair budget.

## Tasks

- [ ] Implement deterministic destroy descriptors.
- [ ] Ensure destroy/repair never mutates outside the declared neighborhood.
- [ ] Track causal lineage from parent candidate → destroy operator → repaired candidate.
- [ ] Use exact CUDA evaluation for repaired batches.
- [ ] Adapt operator selection using evidence across runs while preserving deterministic behavior under a fixed seed/config.
- [ ] Test that a market-only repair leaves unit behavior byte/semantically unchanged.
- [ ] Test that a worker-only repair leaves market behavior unchanged.
- [ ] Reproduce known historical ablation lesson: sometimes combined market+worker change is required, so LNS must support coupled neighborhoods.

## Decision gate

At least one stored/non-transferable or synthetic candidate must improve transfer through semantic repair without losing its original target capability.

**Recommended commit:** `Add semantic LNS candidate repair`

---

# Phase 8 — Reactive IR v2: bounded branch, abort and recovery

## Objective

Address mixed/open-loop-fragile candidates when the correct continuation becomes distinguishable after a legal observable event.

## Files

- Modify `ir.py`
- Modify CUDA IR/evaluator files
- Create/extend `tests/test_ir.py`
- Extend `tests/test_evaluator_parity.py`

## v2 limits

- maximum two internal decision points;
- predicates evaluated only at phase/event boundaries, not arbitrary every-turn general trees;
- predicates use existing legal BKG4 features or explicitly validated derived causal features;
- one explicit abort-to-parent path;
- one bounded recovery phase where evidence requires it;
- total phases/branches remain capped to avoid combinatorial explosion.

## Repair sequence for mixed candidates

When exact evidence is mixed:

```text
1. try later causal checkpoint
2. try shorter common prefix
3. locate first legal state divergence before the first action that must differ
4. branch at that public-state/event boundary
5. add abort/recovery only if the capability itself creates state drift
6. if decisive action differs before any legal distinction exists → FUTURE_DEPENDENT
```

## Tasks

- [ ] Extend canonical IR with versioned branch/abort/recovery records.
- [ ] Define branch semantics exactly and fail closed for missing feature inputs.
- [ ] Extend CUDA and CPU reference interpreters.
- [ ] Add golden parity tests for true/false branch, abort and recovery.
- [ ] Use the strong family0 mixed-state basin as the first bounded empirical test; do not hard-code route IDs into the production representation.
- [ ] Determine whether the family0 capability becomes legally routable with later initiation/common-prefix branching or is truly future-dependent.
- [ ] Record the result even if the correct outcome is permanent rejection.

## Decision gate

Reactive v2 is validated if exact CUDA evidence shows either:

A. at least one previously mixed strong capability becomes legally distinguishable and retains meaningful target/transfer value; or

B. the system proves the tested strong basin is future-dependent while preserving exact semantics and bounded search behavior.

If neither occurs, do not increase branch grammar further yet.

**Recommended commit:** `Add bounded reactive macro semantics`

---

# Phase 9 — Legal-routability pre-screen

## Objective

Reduce CEGIS waste without turning candidate generation into gate overfitting.

## Files

- Create `routability.py`
- Create `tests/test_routability.py`

## Allowed data

Only discovery/target/transfer-style generation partitions may influence the pre-screen.

`SEALED` must never influence generation or repair.

## Tests performed by the pre-screen

### Exact consistency

At initiation and every internal decision boundary:

- group by full exact legal signature;
- detect positive/harmful collision;
- determine whether the trajectories become legally distinguishable before the first decision that requires different action.

### Group-held-out separability

A tiny legal-state classifier/predicate search may be used as a **ranking diagnostic only** across held-out trajectory/family/seat groups.

It must not replace final hardened CEGIS.

### Repair classification

Return one of:

```text
ROUTABLE_CANDIDATE
REPAIR_LATER_CHECKPOINT
REPAIR_COMMON_PREFIX
REPAIR_BRANCH_AT_REVEAL
REPAIR_ABORT_RECOVERY
FUTURE_DEPENDENT
GATE_LIMITATION_SUSPECTED
```

## Tasks

- [ ] Implement exact mixed-sign grouping using the same legal-state definition as hardened CEGIS where possible.
- [ ] Implement decision-time divergence analysis for reactive programs.
- [ ] Add group-held-out separability score with strict prohibition on sealed data.
- [ ] Add tests proving identity/replay fields cannot improve routability score.
- [ ] Add tests that candidate50 remains non-promotable.
- [ ] Add tests distinguishing a repairable post-reveal branch case from a true future-dependent case.

## Decision gate

The pre-screen must correctly route known synthetic/legal cases into repair vs permanent rejection and must never accept a candidate on its own.

**Recommended commit:** `Add legal-routability pre-screen`

---

# Phase 10 — Adapter to existing hardened CEGIS

## Objective

Pass RLRMB capabilities into the validated routing system without creating a second CEGIS implementation.

## Files

- Create `cegis_adapter.py`
- Create `tests/test_cegis_adapter.py`

## Boundary

RLRMB owns:

- candidate program;
- exact target effects;
- transfer effects;
- validation effects;
- legal state captures at initiation/internal decisions;
- routability diagnostics;
- source/effect lineage.

Existing hardened CEGIS remains responsible for the **outer initiation gate** and current acceptance contract.

## Tasks

- [ ] Convert exact RLRMB candidate effects into `MacroEffectCase` inputs while preserving dataset roles and required target-positive/L→W semantics.
- [ ] Reuse `run_cegis()` rather than copying it.
- [ ] Keep validation counterexample feedback active.
- [ ] Preserve exact mixed legal-state fail-closed logic.
- [ ] Preserve gated independent transfer requirements.
- [ ] Preserve frozen sealed holdout behavior.
- [ ] Preserve threshold robustness.
- [ ] Ensure reactive internal branches are treated as part of candidate semantics; CEGIS controls only outer entry unless later evidence requires a separate contract.
- [ ] Add regression proving candidate50 remains rejected.
- [ ] Add historical gate/effect compatibility tests for route435/410 where stored fixtures permit.

## Decision gate

A synthetic/known RLRMB candidate can traverse the existing CEGIS contract with no weakening and deterministic results.

**Recommended commit:** `Integrate RLRMB with hardened CEGIS`

---

# Phase 11 — Deterministic BKG4 materialization and composition

## Objective

Compile accepted Option-Programs into ordinary BKG4 and reuse the validated composer.

## Files

- Create `materialize.py`
- Create `tests/test_materialize.py`

## Materialization model

Conceptually:

```text
outer CEGIS gate
→ set specialist active latch
→ phase selector/state
→ optional legal internal branch
→ route/action-family behavior
→ event/abort/termination
→ clear specialist state
→ return parent
```

## Tasks

- [ ] Define exact mapping from IR phase/branch state to BKG4 nodes/blackboard slots.
- [ ] Reuse existing features/conditions and route override primitives wherever possible.
- [ ] Allocate blackboard/state ownership explicitly.
- [ ] Do not introduce ABI changes unless the current VM genuinely cannot represent the validated candidate.
- [ ] Produce a normal `SpecialistDeclaration`/composer input or the smallest backward-compatible extension required.
- [ ] Run fail-closed composition conflict checks.
- [ ] Compare CUDA Option-Program evaluator result with materialized BKG4 result on active and inactive cases.
- [ ] Compare CPU C++ and CUDA differential execution around initiation, branch, abort, termination and return-to-parent boundaries.
- [ ] Confirm inactive cases are exactly parent behavior.

## Decision gate

Zero CPU/CUDA/BKG4 mismatches on all IR primitives and at least one complete candidate through materialization/composition.

**Recommended commit:** `Materialize RLRMB programs into Bonsai`

---

# Phase 12 — P1 historical rediscovery benchmark

## Objective

Prove the new representation/search can discover the kind of capability that historically required manual specialist work.

## Files

- Create `configs/rediscovery_v1.json`
- Create `pipeline.py` if not already created by earlier phases
- Create `tests/test_pipeline.py` for deterministic small fixtures
- Store large run output only in `scratch/rlrmb/rediscovery_*`

## Targets

Use route435, route410 and route437 regimes as **withheld behavior benchmarks**.

Do not seed the exact accepted `(route,start,duration,gate)` tuple.

Historical motifs/donor blocks may remain in the searchable source library only if the exact target answer is not directly injected.

## Required result

For each regime, RLRMB should discover a behaviorally equivalent coherent capability that:

- improves the intended target regime;
- transfers independently;
- is legally routable;
- survives CEGIS;
- materializes exactly;
- has zero protected W→L on the bounded promotion panel.

Exact syntax/route ID does not need to match history.

## Tasks

- [ ] Define leakage-safe rediscovery fixtures.
- [ ] Run bounded CUDA search per regime.
- [ ] Compare search path, effect cluster and macro motif against historical specialist evidence.
- [ ] Record games consumed and number of unique effect candidates.
- [ ] Require 3/3 behaviorally recovered before calling P1 complete.

## Decision gate

**PASS:** 3/3 historical specialist regimes recovered autonomously under the new candidate representation.

**FAIL:** Diagnose representation vs checkpoint vs search-operator failure. Do not proceed to equal-budget scale while 3/3 recovery is missing unless evidence proves one fixture cannot be made leakage-safe/reproducible.

**Recommended commit after permanent report:** `Validate RLRMB specialist rediscovery`

---

# Phase 13 — P2 equal-budget head-to-head against P0.7

## Objective

Prove a material candidate-generation improvement under the same game budget rather than by brute force.

## Files

- Create `configs/p0_7_equal_budget_v1.json`
- Generated results under `scratch/rlrmb/p0_7_equal_budget_*`
- Add one permanent compact report under `tools/bkg4_search/rlrmb/` after completion.

## Fixed comparison baseline

Old P0.7:

```text
CUDA games: 265,226
refined attempts: 88
accepted: 0
mixed: 49 / 55.7%
non-transferable: 38 / 43.2%
target-capability-lost: 1
```

## Rules

- Same four frontier families where reproducible.
- ≤265,226 CUDA games.
- No weakened CEGIS/holdout/certification.
- No candidate-50 special handling.
- Search may use richer RLRMB representations/operators.
- Sealed evidence stays sealed.

## Success targets

- ≥1 accepted autonomous specialist;
- ≥6 target-L→W candidates / 100k CUDA games;
- mixed-state rejection ≤35%;
- non-transferable rejection ≤30%;
- ≥85% unique effect signatures among refined candidates;
- ≥3 target-solving effect clusters across ≥2 families;
- 0 protected W→L;
- 0 semantic mismatch.

## Decision gate

If RLRMB does not materially beat the old generator, do not scale. Use the failure taxonomy to identify whether the issue is:

- candidate representation;
- beam objective;
- event checkpointing;
- repair operators;
- routability screen;
- actual absence of strategic headroom.

**Recommended commit after validated report:** `Validate RLRMB equal-budget candidate search`

---

# Phase 14 — Production autonomous RLRMB pipeline

## Objective

Create one authoritative clean entry point for future capability generation.

## Files

- Create/finalize `pipeline.py`
- Create `cli.py`
- Create `README.md` usage section
- Extend `tests/test_pipeline.py`

## Authoritative production flow

```text
current BKG4
→ frontier/regret family
→ causal checkpoint selector
→ Option-Program seeds
→ hierarchical diverse beam
→ CUDA evaluation
→ effect QD archive
→ semantic LNS repair
→ reactive repair when mixed
→ routability pre-screen
→ independent transfer
→ hardened CEGIS
→ BKG4 materialization
→ automatic composer
→ target-capability preservation
→ sealed/broad certification
→ CPU/CUDA parity
→ candidate specialist artifact
```

## CLI requirements

One deterministic config-driven CLI. It must not expose manual injection fields such as final gate, selected route, selected threshold or forced accepted candidate.

Required output:

- run ID;
- terminal verdict/reason;
- frontier selected;
- checkpoint proposals;
- beam/QD metrics;
- failure taxonomy counts;
- exact CUDA game count/kernel time/host time;
- accepted specialist artifacts when present.

Exit semantics should distinguish accepted, strategic rejection and infrastructure failure.

## Decision gate

One end-to-end bounded autonomous run produces a complete auditable result without route-specific orchestration scripts.

**Recommended commit:** `Establish authoritative RLRMB synthesis pipeline`

---

# Phase 15 — Deep repository cleanup after RLRMB becomes authoritative

## Objective

Reduce repository clutter without deleting unique evidence, while making the new production path unmistakable.

This phase occurs **after** RLRMB P1/P2 validation and production-path establishment. Cleaning earlier risks deleting evidence or helpers still needed during construction.

## 15.1 Classification first

Classify relevant repository content into:

```text
AUTHORITATIVE_RUNTIME
AUTHORITATIVE_RLRMB
REUSABLE_ENGINE_SUPPORT
HISTORICAL_REPRODUCTION
UNIQUE_EVIDENCE
PAUSED_RESEARCH
REPRODUCIBLE_GENERATED
OBSOLETE_DUPLICATE
UNKNOWN
```

Nothing in `UNKNOWN` may be deleted.

## 15.2 Dependency audit

Before deleting any tracked source:

- search imports/references;
- inspect report/ledger references;
- inspect build files;
- inspect tests;
- inspect whether it is the only reproduction path for an accepted specialist or historical result;
- inspect whether scratch/data contains unique replay/search evidence.

## 15.3 High-confidence deletion targets

Delete only after verification:

- ignored caches and `__pycache__`;
- reproducible local environments;
- stale build products that are cheap to regenerate;
- duplicate generated outputs superseded by permanent reports and not referenced;
- tracked experimental scripts proven unused, duplicated and nonessential for historical reproduction;
- dead test fixtures whose exact evidence is duplicated by an authoritative retained fixture.

## 15.4 Protected by default

Do not blindly delete:

- `.b4fr` exact result evidence;
- `.b4is` legal-state captures;
- replay/native tape data;
- accepted specialist routes/modules/composition evidence;
- P0.1–P0.7 reports;
- current/historical opponent archives;
- donor/motif source libraries;
- engine parity fixtures;
- historical route435/410/437 reproduction paths until replacement evidence is proven complete;
- `data/edit_corpus` merely because that lane is paused.

## 15.5 Desired post-cleanup structure

Future agents should see three obvious categories:

1. **Runtime/engine infrastructure** — BKG4, C++/CUDA exact semantics.
2. **Current search system** — `tools/bkg4_search/rlrmb/` plus explicitly documented stable adapters.
3. **Historical/archive material** — clearly marked reproduction/evidence, not mixed with production instructions.

If broad file moves would break many report links/imports, prefer documentation/classification plus deletion of proven junk over churn-heavy reorganization.

## 15.6 Repo-size audit

Use bounded directory-level size inspection, not pathological full-filesystem scans.

Record before/after size and every deletion class.

Never use broad destructive commands such as:

```text
git clean -fdx
rm -rf scratch/*
```

## Decision gate

Cleanup is complete only when:

- authoritative RLRMB path is documented;
- every new permanent RLRMB file is tracked;
- no new untracked junk exists;
- unrelated pre-existing user files are preserved unless separately resolved;
- no unique evidence was lost;
- focused regression tests still pass;
- repository size/clutter is materially reduced where safe.

**Recommended commit:** `Clean legacy BKG4 search repository safely`

---

# Phase 16 — Scale and successor Stage-A campaign

## Objective

Use the validated new generator to expand the Bonsai strategic repertoire.

Large CUDA search is authorized only after Phases 12–14 pass.

## Search policy

- prioritize unresolved high-regret recurrent weaknesses;
- maintain lane/effect diversity;
- use failure taxonomy to change operator/basin instead of repeatedly increasing budget;
- promote only through hardened CEGIS/composer/certification;
- retain old champion and historical agents for regression/adversarial panels;
- record new solved regimes per GPU-hour.

## Promotion validation

For any candidate intended to become part of the champion:

1. exact CUDA discovery/transfer/CEGIS/certification;
2. exact CPU C++/CUDA/BKG4 semantic parity;
3. broad protected regression;
4. when preparing an actual champion/submission, 64 or 128 official Python-engine games;
5. only then champion promotion/submission consideration.

---

# 8. Detailed failure-driven search policy

The pipeline must not merely log failures. It should alter the next search action.

| Failure | Meaning | Next action |
|---|---|---|
| `NO_HEADROOM` | no tested representation improves target | abandon basin or switch lane/teacher |
| `BAD_CHECKPOINT` | capability appears around a different causal boundary | resample event checkpoints |
| `BAD_HORIZON` | payoff/recovery outside current phase | extend/truncate to causal event |
| `OPEN_LOOP_FRAGILITY` | later legal state changes correct action | reactive repair |
| `MIXED_INITIATION` | same current legal state has conflicting macro outcome | later start/common prefix/branch analysis |
| `NON_TRANSFERABLE` | source-specific baggage or state drift | semantic LNS/mask/phase replacement |
| `DONOR_LIMITATION` | needed behavior absent from source library | parameterized primitive/multi-phase synthesis |
| `REPRESENTATION_LIMITATION` | required strategy needs unsupported bounded structure | extend IR only with concrete evidence |
| `TARGET_CAPABILITY_LOST` | repair abandoned original purpose | constrain target support during search |
| `GATE_LIMITATION` | candidate appears causally separable outside current gate vocabulary | consider evidence-backed CEGIS feature/grammar extension |
| `FUTURE_DEPENDENT` | decisive difference precedes all legal distinction | permanent rejection |
| `SEARCH_BUDGET_EXHAUSTED` | basin stagnated under current operators | freeze archive and switch basin/operator |
| `CUDA_FAULT` | evaluator/runtime fault | infrastructure stop, not strategic rejection |
| `SEMANTIC_MISMATCH` | CUDA/CPU/BKG4 disagree | block promotion and fix parity |

---

# 9. Candidate-source strategy

RLRMB should use multiple structured seed sources, but one search algorithm.

## 9.1 Historical motif library

Mine reusable blocks from strong routes based on semantic/economic behavior, not route ID count.

Examples of motif classes:

- liquidity creation;
- labor expansion;
- land purchase preparation;
- crop-system transition;
- animal-system transition;
- market liquidation;
- premium-product glut response;
- EOD inventory recovery;
- late-game conversion.

A motif should include preconditions/effects as offline descriptors, but runtime routing remains CEGIS/legal-state based.

## 9.2 Parameterized primitives

Add only primitives demonstrated by evidence, e.g. bounded market/investment decisions whose exact semantics are straightforward.

Do not build a giant generic DSL first.

## 9.3 Reduced-subgame teachers

Only after RLRMB core validation, allow exact/approximate subgame teachers for:

- endgame liquidation;
- market-order sequencing;
- investment/land break-even;
- short worker scheduling;
- production planning.

Teacher output must be distilled into the same Option-Program IR.

---

# 10. CUDA performance targets

The previous Stage-A run spent roughly 153 s in recorded GPU kernels inside roughly 1,324 s host wall time. RLRMB should improve the host/kernel ratio as part of candidate throughput.

Track:

- CUDA games/s;
- candidates per evaluator launch;
- average cases per candidate;
- host wall time;
- kernel time;
- serialization/I/O time where measurable;
- dedup ratio before CUDA;
- compaction ratio after screening;
- state-capture calls per refined candidate.

Do not optimize kernel microseconds while Python launches hundreds of redundant subprocesses.

Initial engineering target:

> batch candidate × case evaluation through a small number of persistent/broad CUDA invocations per search layer, returning compact results only.

---

# 11. Test hierarchy

Every phase should use the cheapest test that can falsify it.

## Tier 0 — pure unit

Schema, canonicalization, archive, deterministic expansion, failure taxonomy.

## Tier 1 — exact synthetic microcases

IR interpretation, event termination, branch behavior, abort/recovery.

## Tier 2 — CUDA/CPU parity

Small case/candidate batches only.

## Tier 3 — stored historical evidence

Route435/410/437 and candidate50 regression fixtures.

## Tier 4 — bounded fresh CUDA

Small candidate-generation probes.

## Tier 5 — P1 rediscovery

Medium CUDA.

## Tier 6 — equal-budget P2

≤265,226 CUDA games.

## Tier 7 — actual champion validation

Broad CUDA + bounded official Python canary.

Do not jump to Tier 6 to debug a Tier 0 bug.

---

# 12. Required permanent metrics

Every material RLRMB run should report:

## Search/control

- frontier families considered;
- checkpoints proposed/selected;
- checkpoint reason distribution;
- beam depth/width;
- expansion counts by operator;
- LNS attempts/success by operator;
- CUDA game count.

## Diversity

- structural candidates generated;
- structural duplicates removed;
- unique exact/effect signatures;
- archive occupancy;
- archive replacements;
- target-solving effect clusters.

## Strategic quality

- target positives;
- target L→W;
- target W→L;
- independent transfer support;
- both-seat support;
- worst delta;
- accepted specialists.

## Routing

- routable candidates;
- mixed-state candidates;
- repair classifications;
- CEGIS entry rate;
- CEGIS convergence rate;
- future-dependent rejections.

## Runtime

- host time;
- CUDA kernel time;
- games/s;
- host/kernel ratio;
- semantic mismatch count.

Primary productivity metric:

```text
new safely solved regimes / GPU-hour
```

Secondary metrics such as raw candidate count or average margin may not override it.

---

# 13. Anti-overfitting rules

1. Effect/QD descriptors may use offline family/case provenance for bookkeeping but runtime IR predicates may not.
2. Candidate ranking may use target/transfer evidence; sealed holdout remains untouched until its formal downstream role.
3. If a candidate requires a specific future shop realization that is not yet observable, redesign initiation/branch timing or reject it.
4. Sharp numeric thresholds are not automatically invalid, but must survive the existing threshold/transfer discipline.
5. A candidate should not be called diverse merely because it has a different donor ID, phase serialization or route filename.
6. Both seats should be used where the evidence panel supports them.
7. Do not optimize one family by sacrificing already-solved families.

---

# 14. What must not be built during initial RLRMB implementation

Do not build these before evidence requires them:

- arbitrary 20–100-node free-form Bonsai subtree evolution;
- per-turn 719-step MCTS;
- a learned neural value/proposal model;
- an LLM program mutator;
- EditFormer integration;
- PSRO/adversarial league integration;
- general RL training;
- more than two internal reactive branches;
- a universal predicate DSL;
- multi-GPU orchestration if the current machine/config does not already support it cleanly;
- a generic distributed scheduler.

The first objective is to prove that **small structured reactive programs** materially outperform fixed donor fragments.

---

# 15. Review gates after every phase

Before moving forward, the implementing agent must answer:

1. What exact limitation did this phase address?
2. What test would have falsified the design?
3. Did that test pass?
4. Did CPU/CUDA/BKG4 semantics remain exact where touched?
5. Did the phase create any new untracked or generated junk outside ignored scratch?
6. Did it modify the read-only master plan? It must not.
7. Did it weaken CEGIS, holdout, certification or protected-win standards? It must not.
8. Is the next phase still justified by evidence?

If any answer is unclear, stop and repair the phase before proceeding.

---

# 16. Commit strategy

Use local commits at meaningful phase gates, not after every trivial line edit.

Recommended sequence:

```text
Create isolated RLRMB search foundation
Add RLRMB candidate and effect canonicalization
Add regret and event checkpoint selection
Define bounded RLRMB option-program IR
Implement exact CUDA RLRMB evaluator v1
Add effect-vector QD archive
Implement RLRMB hierarchical diverse beam
Add semantic LNS candidate repair
Add bounded reactive macro semantics
Add legal-routability pre-screen
Integrate RLRMB with hardened CEGIS
Materialize RLRMB programs into Bonsai
Validate RLRMB specialist rediscovery
Validate RLRMB equal-budget candidate search
Establish authoritative RLRMB synthesis pipeline
Clean legacy BKG4 search repository safely
```

Do not push unless explicitly requested.

---

# 17. Final authoritative-path target

After the implementation and cleanup phases, a future agent should be able to understand the production search path from this small graph:

```text
tools/bkg4_search/rlrmb/cli.py
    ↓
regret/event checkpoints
    ↓
Option-Program IR
    ↓
hierarchical diverse beam
    ↓
CUDA candidate evaluator
    ↓
effect-vector QD archive
    ↓
semantic LNS / bounded reactive repair
    ↓
routability pre-screen
    ↓
existing hardened CEGIS
    ↓
RLRMB materializer
    ↓
existing automatic composer
    ↓
certification + parity
    ↓
accepted deterministic BKG4 specialist
```

Historical route-specific scripts remain reproduction evidence, not production instructions.

---

# 18. Terminal completion criteria

The RLRMB construction program is complete only when all statements below are true:

- [ ] The new system lives under a clean isolated `tools/bkg4_search/rlrmb/` namespace.
- [ ] Candidate diversity is measured by canonical structure/effects, not donor filenames.
- [ ] Checkpoints are primarily regret/event localized.
- [ ] Candidate IR supports bounded multi-phase programs.
- [ ] CUDA directly evaluates the new IR in large batches.
- [ ] CPU/CUDA exact parity exists for every IR primitive.
- [ ] QD archive preserves strategically distinct effect classes.
- [ ] Hierarchical beam constructs programs outside the old contiguous-donor manifold.
- [ ] Semantic LNS repairs promising programs coherently.
- [ ] Bounded reactive branches can respond to legal newly observed information.
- [ ] Mixed states are classified into repairable vs future-dependent.
- [ ] Hardened CEGIS remains the final outer initiation-gate authority.
- [ ] BKG4 materialization/composition is exact.
- [ ] route435/410/437 regimes are rediscovered behaviorally without direct answer injection.
- [ ] Equal-budget search materially beats the 265,226-game P0.7 baseline.
- [ ] At least one new specialist is autonomously accepted with zero protected W→L.
- [ ] Repository cleanup is completed only after the new path is authoritative and unique evidence is protected.
- [ ] Before champion/submission promotion, bounded official Python-engine validation is performed.

The strategic end state is not merely a more sophisticated searcher. It is a **clean, exact, autonomous capability factory** that continuously converts hard Kaggriculture weaknesses into new deterministic Bonsai specialists at materially better solved-regime yield per GPU-hour.
