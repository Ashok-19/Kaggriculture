# ReCurSiON Kaggriculture — Deterministic First-Place Execution Plan

## Authority and current stopping point

This is the **authoritative implementation roadmap** for the Kaggriculture project.

Research rationale and source references live in `docs/FIRST_PLACE_MASTER_PLAN.md`.
This document converts that research into repo-specific implementation phases, gates,
artifacts, and stop conditions.

Current verified planning baseline before this document:

- branch: `main`
- HEAD: `23da91177abfabcb462ebeddaacec18d68aa06f2`
- champion: `assets/champion/`
- deterministic runtime: Bonsai/BKG4
- primary planned discovery architecture: **Counterexample-Guided Continuation Program Repair (CCPR)**
- sole fallback discovery architecture after falsification: **Deterministic Semantic Quality-Diversity Evolution (DSQD)**
- neural/RL/BC/PPO/learned-proposer/LLM inner-loop systems: **out of scope**

The project is currently stopped **before new Phase 1**.

Completed infrastructure from the former ASPE work is retained:

| Foundation | Commit | Status | Current role |
| --- | --- | --- | --- |
| Frozen baseline context | `fac2253` | complete | immutable historical comparison context |
| SpecialistProgram v1 | `9726fea` | complete | compact deterministic specialist/option IR |
| CUDA phenotype telemetry + sentinel | `99f8bf1` | complete | cheap behavioral/effect screening and diagnostics |
| Persistent semantic archive | `57fe3e1` | complete | search-only program/phenotype/effect/failure memory |

These are **foundation layers**, not evidence that the old ASPE search hypothesis worked.
No typed-evolution campaign was ever authorized after the architecture review.

---

# 1. Non-negotiable project contract

Every phase and every agent must preserve these rules.

1. **Bonsai/BKG4 remains the final runtime.**
2. **The automated pipeline is deterministic/reproducible.** Fixed-seed exploration is allowed only when ordering, batching, tie-breaking, restart state, and results are reproducible.
3. **No neural policy, value model, learned proposer, behavioral cloning, PPO, RL-over-edits, or LLM in the automated search loop.**
4. **CPU/CUDA semantics must match exactly.** Any semantic mismatch invalidates dependent evidence.
5. **CUDA is the large-scale evaluation engine.** CPU is for bounded reference checks, symbolic/control logic, official-engine checks, and small tests.
6. **BUILD is internal only.** Fresh self-play, internal adversarial policies, internally generated reachable challenge states, and legal mechanics witnesses may construct BUILD.
7. **External/public/historical opponent data is HOLDOUT evidence only.** Never route on opponent identity or use named public opponents as BUILD targets.
8. **INTERNAL_DEV is independent from BUILD.** No DEV-derived feature, label, threshold, or search preference may feed back into discovery.
9. **TRANSFER, VALIDATION, certification CEGIS, and SEALED remain downstream firewalls.**
10. **SEALED is not searchable.** Search never receives case-level SEALED information.
11. **Final protected W→L must remain zero.** No phase may weaken this.
12. **Exact PARENT fallback is mandatory** when a specialist is inactive, terminated, aborted, recovering/completed, or faulted.
13. **Binary routing is the default composition mechanism** until measured portfolio pressure justifies something more complex.
14. **Large-scale campaigns are forbidden before bounded gates pass.** More games do not substitute for better information efficiency.
15. **Every candidate promotion must eventually survive the official Python/Kaggle canary**: 64 games normally, 128 for broad semantic/runtime changes.
16. **`assets/champion/` is the sole authoritative champion.** Never overwrite it from an incomplete experiment.
17. **`workspace/` stores compact tracked evidence.** Every non-ignored created file must be committed or deleted.
18. **Never create `scratch/`.** Existing ignored build/cache paths such as `workspace/build/` are disposable only.
19. **Do not store required evidence under a path component named `build/`**, because generic build paths are ignored.
20. **Do not push or submit unless explicitly requested.**

---

# 2. Current repository assets to reuse

## 2.1 Frozen comparison context

`workspace/aspe/baseline/` is the frozen historical comparison context.

Important properties already established:

- context identity: `f24636a2ea94ee066488b0c40b17362852b5786b4b66f4b7d96406e8ba862563`
- champion identity: `357dbb9489b26a8fb424de592da92e45e2c0ac28d8a450a5e6e1b5791ade9d29`
- league: 8 internal members / strategic lanes
- BUILD: 48 regimes = 3 seeds × 2 seats × 8 members
- INTERNAL_DEV: 24 independent cases
- TRANSFER: 24 cases
- VALIDATION: 24 cases
- SEALED: 24 cases
- zero seed overlap and zero case-identity overlap across panel pairs

Historical equal-budget baselines at 9,600 discovery games:

- RLRMB: 197 candidates, 86 effect phenotypes, 0 BUILD-eligible, 0 cross-regime, 0 DEV, 0 commits.
- Direct Bonsai: 198 candidates, 34 effect phenotypes, 13 limited-harm BUILD candidates collapsing to one strategic phenotype, 0 cross-regime, DEV 0/1 passed, 0 commits.
- semantic mismatches: 0.

Use this context for historical comparison and Phase-2 controlled falsification. Do not mutate it.

## 2.2 SpecialistProgram v1

Current implementation:

- `src/kaggriculture_bonsai/search/aspe/genome.py`
- `src/kaggriculture_bonsai/search/aspe/semantics.py`

It supports:

- 1–6 initiation DNF clauses,
- 1–4 conditions per clause,
- 1–3 bounded body phases,
- up to 2 forward reactive decisions,
- optional bounded recovery,
- semantic/event/public-state/hard-cap termination,
- exact RETURN_TO_PARENT.

Allowed strategic features include time, shops, money, units, land, hires, market price/inventory, shed and seed state. Exact `STEP == X` initiation and coordinate-dominant routing are intentionally rejected.

**Policy:** v1 is sufficient for the next falsification. Extend it only after a concrete legal representation witness shows that a beneficial response cannot be represented or safely separated.

## 2.3 Phenotype telemetry and sentinel

Current implementation:

- `src/kaggriculture_bonsai/search/aspe/phenotype.py`
- CUDA evaluator/result ABI support in `native/cuda/`
- frozen sentinel: `workspace/aspe/sentinel/`

The sentinel is **SCREENING_ONLY**. It can identify zero activation, execution duplicates, effect duplicates and obvious harms. It is never final acceptance authority.

Empirical phenotype equality is not a theorem. New challenges may split previously equivalent classes.

## 2.4 Semantic memory

Current implementation:

- `src/kaggriculture_bonsai/search/aspe/memory.py`
- tracked archive: `workspace/aspe/memory/archive.json`

Authority is always `SEARCH_MEMORY_ONLY`.

It already separates:

- program identity,
- execution phenotype identity,
- strategic-effect identity,
- lineage/proposal metadata,
- failures,
- BUILD/DEV/downstream evidence where permitted by its firewall.

Future CCPR phases should **repurpose** this archive into active BUILD-only counterexample and proposal memory without giving it promotion authority.

## 2.5 Existing snapshot primitives — important repo-specific fact

The repo already contains low-level snapshot work. Phase 1 must extend and validate it rather than reimplement it from zero.

Existing CPU pieces:

- `native/cpp/include/sim.hpp` is designed for trivially copyable simulator state and snapshot/fork use.
- `native/cpp/src/pybridge.cpp` has `Snapshot`, `Game::snapshot()`, and `run_snapshot_candidates()`.
- `src/kaggriculture_bonsai/evaluation/cpp_oracle.py` has `snapshot_vs_tape()` and `branch_tape_candidates()`.

Existing CUDA pieces:

- `native/cuda/include/sim.cuh` defines portable simulation `Snapshot`, save/load support, and cached RNG support.
- `native/cuda/include/policy_runner.cuh` contains `run_policy_snapshot_grid_kernel()`.

Critical current limitation:

> The existing generic CUDA snapshot kernel resets both policy states from the snapshot observation by default.

Its own comments explicitly state that stateful-prefix research should snapshot/copy `PolicyState` arrays alongside game snapshots.

The current CPU Python bridge snapshot also clones simulator state, while existing branching helpers are primarily tape continuations. That is not yet sufficient for CCPR, because exact continuation experiments require the full execution state of both deterministic controllers.

**Therefore new Phase 1 begins with controller-state-faithful continuation support, not basic simulator snapshot invention.**

---

# 3. Pipeline architecture to build

```text
assets/champion + internal capability league
                  |
                  v
        fresh deterministic self-play
                  |
                  v
      reachable challenge/state frontier <---- mechanics opportunity track
                  |                                  |
                  v                                  |
      select recoverable-regret witness              |
                  |                                  |
                  +---------------+------------------+
                                  v
                  CCPR deterministic synthesizer
        [typed bounded holes + paired exact continuations]
                                  ^
                                  |
       BUILD-only counterexamples / caches / archive / library
                                  |
                                  v
                    legal full-game generalization
                                  |
                                  v
                 BUILD -> independent INTERNAL_DEV
                                  |
                                  v
                       binary PARENT routing
                                  |
                                  v
        TRANSFER -> VALIDATION -> certification CEGIS
                                  |
                                  v
                  locked whole composition -> SEALED
                                  |
                                  v
                 materialization + CPU/CUDA parity
                                  |
                                  v
                    64/128 official Python canary
                                  |
                                  v
                     atomic certified champion
                                  |
                                  +------> repeat
```

Primary objective is not games/second. It is:

> **safe strategic capability gained per exact simulation budget and engineering cycle.**

---

# 4. Shared measurement contract

Every bounded search experiment must report the following separately.

## Proposal funnel

- raw proposals,
- canonical programs,
- structurally unique programs,
- execution phenotypes,
- strategic-effect phenotypes,
- target-family activations,
- changed legal actions,
- BUILD-eligible useful phenotypes,
- cross-regime useful phenotypes,
- DEV survivors,
- downstream survivors,
- certified safe specialists.

## Generalization support

Report support across:

- independent root seeds,
- both seats where the challenge family supports both,
- multiple internal members/policies,
- multiple trajectories,
- challenge families/coverage clusters.

Do not count many forks from one root as many independent examples.

## Compute ledger

Record:

- exact simulated turns,
- full games,
- prefix-generation turns,
- suffix/continuation turns,
- reference PARENT continuations,
- failed and cached evaluations,
- CUDA kernel/GPU time where available,
- CPU synthesis/checking time,
- wall time,
- compile/materialization time,
- peak relevant host/device memory,
- official-engine checks separately.

Define:

`FGE = total simulated game turns / 720`

FGE is a simulation-work normalization only. It is not an independence count and not a runtime-equivalence claim.

## Useful cross-regime phenotype

A counted useful cross-regime phenotype must satisfy existing BUILD eligibility and show beneficial full-game W/L support on at least:

- 2 independent root seeds,
- both seats when available,
- more than one internal member when the challenge family has such coverage.

Margin-only improvement is a stepping stone unless existing eligibility already authorizes more.

## Safety metrics

Always report:

- semantic mismatches,
- faults,
- protected W→L,
- W/D/L transitions,
- margin tails when relevant,
- fallback mismatches,
- official/native discrepancies when official checks are run.

---

# 5. Common experiment/repository contract

Every phase that runs a bounded experiment should leave one compact machine-readable result under a tracked path such as:

`workspace/first_place/<phase-or-run>/`

The manifest/result must include:

- schema/version,
- source HEAD,
- engine/config identity,
- champion identity,
- internal member identities,
- panel/root provenance,
- feature/grammar version,
- compiler/native ABI version where relevant,
- deterministic queue/tie-break rules,
- budgets,
- exact cost ledger,
- funnel metrics,
- parity/mismatch counts,
- pass/fail/defer verdict.

Large disposable traces, binaries and temporary caches belong only in already ignored paths. Keep compact recipes, hashes and minimized witnesses in Git.

Every phase ends with:

1. focused tests,
2. compact evidence,
3. explicit gate verdict,
4. local commit,
5. clean worktree.

A failed gate is a valid phase result. Do not implement later complexity to conceal it.

---

# 6. Phase 1 — Exact Controller-State Fork-and-Repair Value Experiment

## Objective

Prove or falsify the core premise that exact reachable-state continuations provide useful causal credit and improve information efficiency before CCPR itself is built.

This is the **only authorized next implementation phase**.

## Repo-specific implementation scope

Reuse the existing snapshot infrastructure and add only what is missing for exact policy continuation.

Primary areas to inspect/extend:

- `native/cpp/include/sim.hpp`
- `native/cpp/src/pybridge.cpp`
- `native/cuda/include/sim.cuh`
- `native/cuda/include/policy_runner.cuh`
- existing CUDA evaluation entrypoints under `native/cuda/src/`
- `src/kaggriculture_bonsai/evaluation/cpp_oracle.py`
- SpecialistProgram executor/evaluator paths
- nearest snapshot/parity tests

Required new contract:

A decision-boundary continuation snapshot must preserve enough state to reproduce uninterrupted play exactly, including:

- simulator state,
- config and terminal/game counters,
- RNG provenance/state needed by the real transition semantics,
- both controller runtime states,
- BKG4 blackboard/node/runtime meta state,
- compatibility/legacy overlay state where active,
- active SpecialistProgram state if applicable,
- exact engine/interpreter/ABI version identity.

Do not expose hidden snapshot-only information to runtime SpecialistProgram predicates.

## Important correctness distinction

Maintain three separate concepts:

1. **Exact continuation identity** — all simulator/controller state required to resume identically.
2. **Search abstraction** — descriptors used to group/compare challenge states.
3. **Runtime observation** — legal information a submitted deterministic policy may read.

Never use exact snapshot identity, seed, hidden state, or opponent identity as a runtime gate.

## Continuation semantics

For reachable state `s` and candidate option `p`:

```text
Delta_terminal(s,p)
  = final_margin(continue from s with p then PARENT)
  - final_margin(continue from s with PARENT)
```

Opponent actions must be recomputed from the modified state using the frozen internal opponent policy and its correct execution state. Do not replay an old future action tape and call that adversarial continuation evidence.

Short 24/96-turn horizons are diagnostics only. Terminal continuation is the strategic credit signal.

Action-dependent randomness must follow the real engine after the fork; branches are not required to share future events merely because the pre-fork seed was the same.

## Bounded experiment

Use 12 fresh internal root games:

- 3 BUILD-only fresh root seeds,
- both seats,
- 2 internal members chosen before the run.

Select two decision-boundary checkpoints per root game:

- one earlier investment/transition checkpoint,
- one later recovery/liquidation/transition checkpoint.

Total target: 24 checkpoints.

Evaluate up to 64 predeclared compact SpecialistProgram-v1 bodies/variants across those checkpoints, with a hard total phase budget of **≤3,200 native FGE** including prefixes, suffixes and full-game follow-up.

The intervention matrix should include:

- exact PARENT identity control,
- no-activation/fallback control,
- normal one-phase interventions,
- bounded parameter changes,
- termination variants,
- reactive variants where already expressible,
- deliberately infeasible-action/control cases.

For informative candidates, compare:

- exact forked continuation,
- matched uninterrupted-prefix intervention,
- PARENT reference,
- 24-turn diagnostic,
- 96-turn diagnostic,
- terminal outcome.

At most 12 informative bodies may proceed to ordinary legal full-game activation using existing gate/materialization machinery.

Do not query INTERNAL_DEV during body discovery.

## Fidelity tests

Must explicitly test:

- PARENT snapshot continuation == uninterrupted PARENT,
- modified candidate continuation == uninterrupted equivalent intervention,
- both seats,
- day boundary,
- shop/randomness boundary,
- active controller state,
- active SpecialistProgram phase/branch state where feasible,
- termination and return-to-parent,
- restart/reload determinism,
- CPU/CUDA equivalence,
- stateful policy continuation rather than reset-from-observation behavior.

## Metrics

- snapshot bytes and serialization identity,
- CPU restore equality,
- CUDA restore equality,
- action trace equality where expected,
- terminal margin equality,
- controller-state equality,
- faults/mismatches,
- beneficial body count,
- beneficial strategic-effect phenotype count,
- support across roots/seats,
- fork-to-full-game retention,
- actual simulated turns saved/added,
- early diagnostic false-negative rate.

## Pass gate

Phase 1 passes only if all are true:

1. **Zero snapshot/restart/fallback/CPU-CUDA semantic mismatches.**
2. At least **2 strategically distinct body effects** show positive paired terminal benefit across at least 2 root seeds and both seats in their relevant family.
3. At least **1** retains positive net paired terminal benefit when executed through an ordinary legal full-game SpecialistProgram gate on BUILD.
4. Shared-prefix/selective evaluation retains at least **90%** of beneficial matrix candidates while avoiding at least **30%** of simulated turns versus the matched uninterrupted-prefix comparison, counting all prefix/reference costs.
5. No DEV/SEALED leakage occurred.

These are falsification thresholds, not expected guaranteed results.

## Failure branches

### Fidelity failure

Fix snapshot/controller-state semantics inside Phase 1. No search claim survives a mismatch.

### No beneficial interventions

Permit one preregistered alternate checkpoint/motif matrix within the same ≤3,200-FGE total budget. If still empty, stop and report that the repair-density premise is unsupported.

### Local fork benefit but legal full-game activation loses it

Stop before CCPR. Record a concrete activation/representation/generalization witness. Do not hide it by building a larger search.

### Fidelity passes but turn-saving/information-efficiency gate fails

Snapshots remain a debugging/late-game tool, not a mandatory search primitive. Phase 2 must use matched ordinary full-game controls and the DSQD fallback becomes more relevant.

## Non-goals

Do not implement:

- CCPR best-first synthesis,
- DSQD,
- automatic challenge frontier,
- SMT solver,
- new VM,
- large grammar expansion,
- global routing,
- learned proposal logic,
- large campaign.

## Deliverables

- versioned exact continuation/snapshot contract,
- CPU/CUDA stateful continuation support,
- focused parity/restart tests,
- deterministic experiment entrypoint/config,
- compact `workspace/first_place/phase1...` evidence,
- intervention matrix summary,
- full cost ledger,
- legal-gate retention report,
- explicit `PASS` or `FAIL` verdict.

---

# 7. Phase 2 — Minimal CCPR + Equal-Budget Falsification

**Prerequisite:** Phase 1 passes its strategic and fidelity gates.

## Objective

Test whether counterexample-guided deterministic program repair actually produces generalizable specialists more efficiently than simpler matched controls.

## Implementation scope

Implement the smallest CCPR oracle needed for a fair experiment.

Recommended package direction:

`src/kaggriculture_bonsai/search/ccpr/`

Keep responsibilities minimal, likely:

- typed sketch/hole representation,
- deterministic best-first queue,
- BUILD-only witness/counterexample table,
- predicate threshold/bitset enumeration,
- exact evaluation cache,
- proposal/failure scoring from integer counts,
- integration adapters to SpecialistProgram v1, phenotype telemetry and semantic memory.

Do not rename or duplicate existing SpecialistProgram/phenotype/archive semantics.

## Search loop

For one BUILD challenge family:

1. select a recoverable-regret or disagreement witness,
2. start before the suspected causal failure,
3. create a tiny bounded sketch,
4. enumerate candidate holes deterministically,
5. batch exact continuation checks,
6. classify the failure,
7. modify only implicated holes/gates,
8. broaden support across roots/members,
9. promote only a bounded shortlist to full BUILD,
10. freeze shortlist before DEV.

Initial body sketches should prefer:

- 1 phase,
- one bounded termination,
- exact PARENT fallback,
- parent/motif/primitive sources already supported,
- second phase or reactive branch only after evidence requires it.

## Deterministic proposal ordering

Use stable queue ordering. A suitable initial tuple is:

1. unresolved witness-class count,
2. repair compatibility,
3. quantized rule cost,
4. program size,
5. canonical semantic ID.

No wall-clock time or worker completion order may change future proposals.

At fixed batch boundaries, maintain exact integer statistics for context/operator attempts, informative distinct outcomes, and BUILD successes. Rational comparisons must be deterministic; do not call them calibrated probabilities.

## Counterexample scope

Counterexamples are body-and-context specific.

Examples:

- zero activation → refine gate or challenge selection,
- resource infeasible → preparation/recovery hole,
- harmful phase → remove/shorten/repair that phase,
- bad termination → earlier/later event termination,
- wrong body on a legal observation → body-specific negative gate witness,
- contradictory outcomes under same abstraction → test legal feature refinement or require robustness,
- exact complete-program/context repeat → cache,
- proved contradiction → permanent local prune,
- sampled phenotype duplicate → reversible deprioritization, not theorem.

Never convert one failing body into a ban on all specialists for that state.

## Solver policy

Start without a general SMT dependency.

Use deterministic enumeration/bitsets for small predicate holes. Add one pinned single-thread solver only if measured evidence shows predicate/hole solving is the bottleneck. Do not encode the full game in SMT.

## Main experiment

### Frozen context

Run CCPR with **9,600 FGE** on the frozen Phase-0 context.

Run two matched 9,600-FGE controls:

1. same grammar/order but whole-game evaluation without targeted fork credit,
2. CCPR with active counterexample/count reuse disabled.

Historical RLRMB/Direct results may be shown only if engine/config identity matches. Otherwise label them historical and use a bounded same-version comparator before superiority claims.

### Fresh replication

After the frozen comparison, lock CCPR design.

Run two fresh independently seeded contexts, each **9,600 FGE**, with all downstream panels generated/frozen before discovery and disjoint by root/seed/case identity.

Each arm exposes at most 3 preregistered BUILD-selected candidates to its DEV screen.

## Metrics

- useful cross-regime phenotypes / 10k FGE,
- cost to first useful phenotype,
- cost to first DEV survivor,
- effect diversity,
- exact-repeat rate,
- repeated-failure rate,
- counterexample reuse rate,
- fork-to-full retention,
- CPU synthesis overhead,
- GPU utilization/evaluation cost,
- downstream survivors,
- certified safe gains.

## Pass gate

CCPR becomes the primary production oracle only if:

1. each of the **two fresh replication contexts** produces at least 2 distinct cross-regime useful phenotypes,
2. each fresh context produces at least 1 existing-gate DEV survivor,
3. at least 1 final composition survives the full downstream pipeline and official canary,
4. targeted continuation credit and active memory show clear value over matched controls after accounting for overhead,
5. if a matched control has positive yield, CCPR targets at least 2× useful yield/FGE or ≤50% cost to first independent survivor,
6. all parity/firewall/protected gates remain unchanged.

Do not claim an infinite speedup when a control has zero survivors. Report absolute counts and costs.

## Failure branches

### No generalizable bodies

Reject CCPR as primary. Run the single DSQD fallback described below.

### BUILD improves but DEV collapses

Stop and diagnose BUILD-only abstraction/generalization. Do not tune on DEV.

### Concrete v1 representation witness

Stop and design the smallest v2 change. Rerun an equivalent bounded comparison. Do not expand grammar speculatively.

### DEV survives but routing fails

Keep the body as `SEARCH_MEMORY_ONLY`; diagnose routability before scaling.

## Deliverables

- minimal CCPR package,
- deterministic restart/checkpoint state,
- exact cost ledger,
- matched ablation results,
- two fresh replications,
- architecture verdict,
- any certified specialist/composition produced,
- clean commit.

---

# 8. Conditional fallback — DSQD only after CCPR falsification

This is not a parallel permanent oracle.

## Objective

Test whether counterexample decomposition is the wrong inductive bias while keeping the same deterministic representation, evaluator, telemetry, archive, safety gates, and validated snapshot infrastructure.

## Scope

Implement deterministic typed semantic QD with:

- fixed ordered mutation neighborhoods,
- small sparse niches/local competition,
- bounded stepping-stone retention,
- stable deterministic migration/reseeding,
- semantic archive integration,
- no neural proposer.

## Budget and gate

First run: 9,600 FGE on one primary-failed context.

Only if it shows cross-regime useful signal may it proceed to two fresh replications, for a maximum of 28,800 discovery FGE before downstream work.

Pass criteria are the same as CCPR Phase 2. More novelty alone is failure.

If CCPR and DSQD both fail, stop scaling and commission a bounded representation/state-abstraction research task from concrete witnesses. Do not invent a third speculative production oracle.

---

# 9. Phase 3 — Reachable Challenge Frontier + Active Search Memory

**Prerequisite:** primary oracle selected by Phase 2 or DSQD fallback.

## Objective

Escape the fixed eight-lane basin and make later search depend on what prior evaluations actually taught us.

## Scope

Build a deterministic reachable challenge archive containing compact recipes/provenance for state families.

Each challenge family should retain:

- exact root/prefix provenance,
- legal observable descriptors,
- search abstraction version,
- recoverable-regret evidence,
- relevant transition/coverage tags,
- one or more representative reachable witnesses,
- successful controls where available,
- known response/effect disagreements,
- current status: solved / partially solved / unsolved / stale.

The eight current lanes remain reporting tags and initial strata, not permanent archive cells.

## Challenge sources

- fresh champion/internal-league games,
- retained payoff-diverse internal checkpoints,
- deterministic edits to internal programs/prefixes near selected events,
- existing options inserted earlier to reach new economic basins,
- internal pairings exposing payoff-profile gaps,
- confirmed mechanics witnesses,
- combinations of known options/resource regimes.

Arbitrary hand-edited states are mechanics/debug evidence until reproduced by legal play.

## Active memory

Repurpose `SemanticArchive` into a BUILD-only query surface for the proposer.

Use failure evidence to:

- avoid exact repeats,
- refine body-specific gates,
- prioritize missing prerequisites,
- change implicated holes,
- reduce repeated failure classes,
- version/invalidate stale context-specific constraints when parent/engine/grammar/abstraction changes.

Historical DEV/downstream evidence may remain verifier provenance but must not be available to proposal ranking.

## Automatic family splitting

When one search abstraction contains conflicting responses over enough independent roots, test a deterministic split on a **legal observable feature**.

Suggested minimum split evidence:

- at least 8 independent roots in the parent cell,
- conflicting response signs,
- both child cells retain independent support,
- held-out BUILD roots show lower aliasing.

Never split by seed, seat identity, opponent identity, hidden private state not observable to the runtime, or replay fingerprint.

## Experiment

Compare expanding-frontier search against fixed-lane search over matched budgets.

Use fixed-parent epochs of 2,400 FGE each and a capped fresh-root generation budget. Include one later parent-refresh epoch to prove cache invalidation and continued novelty.

## Pass gate

- ≥3 supported new challenge clusters beyond the initial taxonomy,
- ≥2 cross-regime useful phenotypes arising from those clusters,
- ≥1 fully certified new capability,
- ≥25% reduction in repeated avoidable failure rate versus passive-memory control at matched coverage,
- later useful yield ≥50% of early post-warm-up yield,
- zero panel/root leakage.

## Failure branch

If coverage rises without capability, shift priority toward recoverable regret and causal predecessors. Do not grow archive size for its own sake.

## Deliverables

- versioned challenge schema,
- deterministic scheduler,
- BUILD-only active-memory interface,
- corpus minimization,
- restart/cache invalidation tests,
- fixed-vs-expanding ablation,
- compact evidence.

---

# 10. Phase 4 — Mechanics Differential Testing + Opportunity Discovery

## Objective

Make engine-mechanics discovery a first-class deterministic subsystem while ensuring native-clone bugs can never masquerade as strategic gains.

## Two queues

### Correctness queue

Compare CPU/CUDA/native behavior and adjudicate representative cases against the pinned official Python engine.

Any mismatch:

- freezes affected search evidence,
- blocks promotion,
- triggers correctness repair,
- invalidates dependent cached results as needed.

### Opportunity queue

Search unusual but legal and semantically equal transitions for reproducible strategic benefit.

Only legally reachable internal witnesses may become BUILD challenge families.

## Initial mechanics families

At minimum cover:

- shop-demand/pricing boundaries,
- market quote and sequential commit interactions,
- shed/carrying/capacity pressure,
- animal care/feed/production timing,
- fertilizer timing/refresh,
- multi-worker shared-resource contention,
- state-dependent randomness/order-of-operations coupling.

Use deterministic boundary schedules such as:

- zero,
- one,
- just below threshold,
- at threshold,
- just above threshold,
- adjacent day/event boundaries,
- short legal action-order permutations.

Do not assume action commutativity or seat symmetry unless preconditions prove it.

## Experiment

- 10k–100k short transition cases across at least six families,
- deliberate perturbed test fixtures to verify mismatch detection,
- minimize interesting witnesses,
- validate up to ten opportunity hypotheses across related reachable roots,
- run only promising legal opportunities through the normal discovery/generalization pipeline.

## Pass gate

Correctness harness passes if:

- seeded fixture faults are detected,
- real production mismatches are zero/resolved,
- ≥90% of preregistered reachable boundary categories are exercised or documented as unreachable.

Strategic mechanics search receives expanded budget only after at least one cross-root beneficial legal option is found.

No profitable exploit is required for the correctness harness itself to be useful.

## Deliverables

- minimized mechanics corpus,
- engine/config/version pins,
- parity regression tests,
- reachable opportunity ledger,
- explicit rules/provenance checklist,
- compact official adjudication evidence.

---

# 11. Phase 5 — Evidence-Driven Symbolic Vocabulary

**Prerequisite:** enough successful programs exist. If not, mark `DEFERRED`.

## Objective

Stop rediscovering the same useful multi-action structures from primitives while keeping the final runtime ordinary deterministic BKG4.

## Start condition

Do not begin until there are at least:

- 10 useful BUILD programs,
- 3 independent successful lineages/families.

## Scope

Mine reusable typed fragments through deterministic anti-unification/compression.

Candidate library entries must preserve:

- preconditions,
- action ordering,
- family ownership,
- termination semantics,
- parameters,
- source lineage/provenance,
- expanded BKG4 cost.

The library is a search-time vocabulary only. Submission assets contain expanded ordinary BKG4 behavior.

## Ranking

Use deterministic measures such as:

- corpus compression benefit,
- definition cost,
- expanded node/runtime cost,
- parameter count,
- unsupported preconditions,
- cross-lineage reuse.

Compression is not strategic success.

## Experiment

Mine at most 32 candidate components from at most 100–500 successful/relevant programs.

Run matched new-BUILD-task searches:

- 4,800 FGE with primitive grammar,
- 4,800 FGE with library + same primitive access.

Use tasks withheld from the induction corpus.

## Pass gate

- ≥2 components reused beneficially outside induction families,
- ≥25% lower cost to matched useful-phenotype yield, or ≥2 extra useful phenotypes if control has none,
- no meaningful loss of generalization,
- no material expanded runtime/node regression,
- ≥1 library-enabled candidate fully certified.

## Failure branch

Keep successful archive programs and primitive grammar; retire unproductive library preference. Do not force a vocabulary phase if the success corpus is too weak.

---

# 12. Phase 6 — Long-Horizon Option Composition + Minimal Representation Growth

## Objective

Discover profitable strategy sequences that a single option followed by PARENT cannot express or evaluate correctly.

## Primary credit

Use paired terminal outcomes. Intermediate resource/milestone deltas are explanatory/ranking evidence only.

For options A and B measure:

```text
PARENT
A
B
A -> B
B -> A   (when legal)
```

Context-specific interaction:

`interaction(A,B) = Delta(A->B) - Delta(A) - Delta(B)`

Do not assume global additivity.

## Scope

- bounded removal tests,
- ordered option-pair experiments,
- enabling-state retention,
- deterministic event-level beam search,
- exact state transpositions,
- depth 2–3 initially,
- beam width 16–64 initially.

Reserve a bounded quota for option pairs where neither component is standalone-positive, because investment/enabling actions may require a later production/liquidation option.

## Representation growth rule

Only extend SpecialistProgram after a concrete witness proves:

1. a legal beneficial response exists,
2. v1 cannot express it or cannot legally distinguish harmful/supporting states,
3. the proposed minimal extension resolves the witness.

Then:

- version the representation,
- prove CPU/CUDA/materialization parity,
- rerun a matched old/new grammar experiment.

No arbitrary loops, new VM, or coordinate memorization without evidence.

## Pass gate

- ≥2 repeatable cross-root option combinations outperform component-only controls,
- ≥1 fully certified improvement,
- exact compilation/fallback parity,
- any representation extension passes its own witness and matched comparison.

## Failure branch

If PARENT already realizes the gain, do not add a planner. If only a relaxed economic model succeeds but executable actions fail, reject the plan.

---

# 13. Phase 7 — Global Portfolio Composition, only under measured pressure

This phase is conditional and may remain `DEFERRED` indefinitely.

## Trigger

Consider it when at least one is true:

- ≥32 useful eligible specialists exist,
- measured activation overlap >10% of active decisions,
- rejection is materially conflict-driven,
- compiled runtime/node usage approaches 70% of confirmed limit,
- a smaller portfolio already demonstrates damaging higher-order interaction.

## Objective

Preserve safe marginal gains when sequential binary additions become limited by overlap, ordering, shadowing or size.

## Scope

- activation/conflict telemetry,
- sparse conflict/no-good representation,
- bounded selection + ordering variables,
- complete-composition native simulation,
- optional pure-predicate sharing if it reduces actual compiled BKG4 cost.

A solver may schedule/select finite candidate combinations, but simulator certification remains authority.

If using MaxSAT/CP-SAT:

- pin one backend/version,
- one worker,
- deterministic work limits,
- canonical tie-breaking,
- UNKNOWN/timeout = no proof.

## Critical interaction rule

Pairwise compatibility is not enough. A composition regression may be higher-order.

After a regression:

- minimize the causal selected set/order if possible,
- add a scoped no-good,
- if minimization is inconclusive, forbid only the full failing assignment/context rather than inventing pairwise conflict.

## Pass gate

Relative to unchanged binary sequential routing, global composition must either:

- add ≥1 otherwise blocked useful capability, or
- reduce real compiled node/runtime cost by ≥20% at identical behavior,

while preserving:

- all protected wins,
- zero semantic mismatch,
- whole-composition downstream certification,
- official canary.

If not, keep binary routing.

---

# 14. Phase 8 — Autonomous Certificate-Gated Champion Improvement

## Objective

Turn the validated discovery stack into repeatable deterministic champion-improvement epochs without contaminating evaluation firewalls.

## Epoch contract

Before each epoch freeze:

- champion parent identity,
- internal league identities,
- BUILD root namespace,
- INTERNAL_DEV roots,
- TRANSFER roots,
- VALIDATION roots,
- certification policy,
- SEALED identity/private verifier contract,
- grammar/library version,
- search abstraction version,
- proposal budget,
- shortlist size,
- stopping rule,
- deterministic scheduler config.

All panel/root identities must be disjoint by construction.

## Discovery permissions

The proposer/search process may read BUILD-only active memory.

A separate verifier process/interface handles downstream protected evidence. Do not expose downstream case details to proposal generation.

Repeatedly used DEV panels lose independent-estimation status after their frozen shortlist is tested. They may remain verifier-side historical regression cases, while future epochs receive fresh DEV.

## Promotion

Every proposed champion update must complete:

1. BUILD eligibility/generalization,
2. INTERNAL_DEV,
3. binary/global routing as applicable,
4. TRANSFER,
5. VALIDATION,
6. certification CEGIS,
7. locked composition,
8. SEALED,
9. materialization reproducibility,
10. CPU/CUDA parity,
11. official 64/128 canary,
12. packaged cold-start/runtime/size checks,
13. atomic champion asset update.

## Experiment

Run at least three bounded production epochs and deliberately exercise:

- one rejected candidate,
- one interrupted/restarted run,
- one parent refresh.

## Pass gate

- ≥3 strategically distinct fully certified improvements across the bounded epochs,
- identical selected candidates/materialized hashes after deterministic restart,
- new challenge/failure families still being discovered,
- zero firewall violations,
- zero protected regression,
- bounded archive/runtime growth.

## Scaling gate

No million-game campaign.

Before even authorizing large tranches, require at least:

- 5 strategically distinct safe improvements,
- across ≥3 production epochs,
- within a preregistered ≤100,000 all-in native-FGE production window,
- covering ≥4 independently supported challenge clusters,
- late-epoch useful yield ≥50% of early post-warm-up yield,
- measurable remaining frontier novelty,
- zero semantic mismatch/protected regression/official discrepancy.

Even after passing, scale only in 25k–100k-FGE tranches with saturation checks.

---

# 15. Phase 9 — First-Place-Ready Adversarial Audit and Release

## Objective

Produce the strongest deterministic submission package that survives fresh internal attack, optional external HOLDOUT evaluation, semantic translation checks and operational constraints.

## Freeze first

Before the final audit, freeze:

- champion candidate,
- route/composition assets,
- engine/config,
- internal audit budget,
- HOLDOUT policy,
- official canary cases,
- packaging configuration.

No repair occurs inside the final audit itself.

## Internal audit

Challenge the frozen portfolio with:

- fresh internally synthesized exploiters,
- payoff-diverse retained capabilities,
- rare mechanics families,
- cross-regime transition challenges,
- new reachable frontier states.

An unresolved high-impact counterexample blocks release or starts a new independent search epoch.

## External HOLDOUT

Available external agents may be evaluated only as independent HOLDOUT.

Never:

- route on identity,
- import their action traces into BUILD,
- synthesize an opponent-specific replay repair.

A HOLDOUT failure may block selection, but the repair must be rediscovered from independent internal evidence in a new epoch.

## Official release contract

Before submission authorization:

- zero native semantic mismatches,
- final protected W→L = 0,
- all required downstream gates pass,
- SEALED pass on locked composition,
- 64-game official Python/Kaggle canary,
- 128-game canary for broad semantic/runtime changes,
- actual packaged submission tested,
- deterministic cold start,
- runtime/memory/size compliance,
- exact package/asset identity recorded,
- current competition rules/deadline/submission-selection constraints verified from official material.

Any unexplained official/native difference blocks promotion.

## Deliverables

- final deterministic submission package,
- engine/config/policy/package identities,
- verifier certificate/digest,
- compact release evidence,
- rollback identity for prior champion,
- clean local commit.

Push/live submission still requires explicit user authorization.

---

# 16. Deterministic anti-saturation contract

Monitor discovery in fixed 2,400-FGE windows once production search exists.

| Signal | Trigger | Required response |
| --- | --- | --- |
| exact repeated program/context | >20% for two windows | fix caching/scheduler before more GPU |
| noninformative effect duplicates | >80% for two windows | shift quota to least-served challenge families / distant fragments |
| repeated avoidable failure class | >50% of rejections | add scoped counterexample/prerequisite repair and verify memory use |
| novelty without useful capability | 3 windows with new coverage but 0 useful cross-regime phenotype | prioritize recoverable regret and causal predecessors |
| coarse state aliasing | conflicting body signs with ≥8 independent roots | test one legal-feature partition refinement |
| delayed-payoff false negatives | terminal-positive audits repeatedly killed early | disable/revise that early screen |
| library lock-in | >75% proposals descend from one fragment family | restore primitive/unexplored-construction quota |
| champion-local trap | challengers become behaviorally redundant | refresh payoff-diverse internal capabilities / prefixes |
| portfolio overhead | certification dominates or useful additions are shadowed | freeze additions; consider Phase 7 only if trigger is met |
| sustained zero safe yield | 3 complete production epochs | stop scaling; one bounded response cycle, then architecture review |

Response order:

1. remove exact waste,
2. change challenge allocation,
3. refine justified legal state abstraction,
4. retrieve distant reusable components,
5. search earlier causal prefixes,
6. test enabling option combinations,
7. stop if safe yield still does not recover.

Do not respond to every plateau by expanding the grammar.

---

# 17. What must not be revived

Do not revive as primary architecture:

- old ASPE evolution-first sequencing,
- learned proposer phases,
- PPO/RL/BC/neural policies,
- BonsaiEditFormer/edit-corpus pipeline,
- replay imitation,
- RLRMB donor-window search as primary oracle,
- Direct step-local Bonsai search as primary oracle,
- whole-game monolithic SMT,
- primitive 720-turn MCTS,
- global router before conflict pressure exists,
- dense giant MAP-Elites grids,
- sample-equality treated as formal semantic equivalence,
- million-game search before repeated bounded proof.

RLRMB and Direct remain frozen diagnostic baselines and may supply limited seed/motif ideas only if that does not distort equal-budget comparisons.

---

# 18. Current file-level implementation map

## Keep/reuse

- `assets/champion/`
- `workspace/aspe/baseline/`
- `workspace/aspe/sentinel/`
- `workspace/aspe/memory/`
- `src/kaggriculture_bonsai/search/aspe/genome.py`
- `src/kaggriculture_bonsai/search/aspe/semantics.py`
- `src/kaggriculture_bonsai/search/aspe/phenotype.py`
- `src/kaggriculture_bonsai/search/aspe/memory.py`
- existing search IR/motifs/evaluator/canonicalization
- current league/self-play/BUILD/DEV generation
- current routing/materialization/CEGIS/downstream safety
- `native/cpp/` reference simulator
- `native/cuda/` evaluation simulator

## New code should be added only when its phase is active

Likely future areas:

- Phase 1: continuation/snapshot adapters and exact controller-state support,
- Phase 2: `search/ccpr/` minimal deterministic synthesizer,
- Phase 3: challenge frontier + active-memory layer,
- Phase 4: mechanics harness,
- Phase 5: symbolic library module,
- Phase 6: option-composition planner,
- Phase 7: conditional composition solver,
- Phase 8: autonomous epoch/promotion runner.

Do not pre-create empty packages for later phases.

---

# 19. Phase gate summary

| Phase | Build only after | Primary proof required | If it fails |
| --- | --- | --- | --- |
| 1. Exact fork/value | now | exact stateful continuation + useful legally retained fork signal | stop/repair fidelity or reject snapshot thesis |
| 2. CCPR falsification | Phase 1 PASS | replicated cross-regime + DEV + one fully certified composition | run DSQD fallback only |
| DSQD fallback | CCPR FAIL | same replication/certification standard | stop scaling; representation/abstraction research |
| 3. Challenge frontier | oracle selected | new challenge clusters convert to safe capability and reduce repeated failures | retune recoverable-regret allocation, not archive size |
| 4. Mechanics | Phase 1 infra available; can overlap after Phase 2 | correctness coverage; strategic budget only after cross-root opportunity | keep small correctness maintenance harness |
| 5. Symbolic vocabulary | ≥10 useful BUILD programs / ≥3 lineages | new-task reuse improves useful yield | defer/retire library preference |
| 6. Option composition | useful option corpus exists | certified joint effects unavailable to component controls | keep single-option oracle |
| 7. Global composition | measured portfolio pressure | more safe capability or ≥20% same-behavior cost reduction | keep binary routing |
| 8. Autonomous epochs | discovery/challenge/safety stack validated | repeated deterministic certified improvements | stop on saturation/firewall failure |
| 9. Final audit/release | champion frozen | no unresolved high-impact internal issue + complete release contract | ship prior certified champion or reopen epoch |

---

# 20. Definition of success

The pipeline is not successful because it:

- evaluates many games,
- creates many programs,
- fills an archive,
- finds many phenotype hashes,
- produces margin-only changes,
- adds code complexity.

It succeeds when it repeatedly converts exact internal evidence into:

```text
new reachable challenge
    -> new strategically distinct useful response
    -> cross-root / cross-seat / cross-member support
    -> independent DEV survival
    -> safe routing
    -> downstream certification
    -> zero protected regression
    -> official-engine agreement
    -> stronger deterministic champion
```

The first implementation decision is therefore unambiguous:

> **Execute Phase 1 only: validate exact controller-state forks and the fork-and-repair value thesis within ≤3,200 FGE. Do not implement CCPR until that phase passes.**
