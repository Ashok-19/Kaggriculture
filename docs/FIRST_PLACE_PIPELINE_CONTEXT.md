# ReCurSiON Kaggriculture — First-Place Deterministic Pipeline Context

Roadmap update: [FIRST_PLACE_MASTER_PLAN.md](FIRST_PLACE_MASTER_PLAN.md) now defines the proposed deterministic execution gates. This file remains the factual pre-research repository snapshot.

## Purpose of this file

This is the **single primary context document** for a new high-cost planner/researcher reviewing the Kaggriculture project.

The planner should be able to understand the current system, the evidence that led here, the completed ASPE work, the still-unproven future plan, the safety contract, and the open first-place questions **without reading the implementation source tree first**.

Source inspection is allowed only when a critical ambiguity cannot be resolved from this document or when the planner needs to verify one narrow factual claim. The planner should spend its expensive reasoning budget on architecture/research rather than reconstructing the repository.

This document describes the project state at repository HEAD `57fe3e1223fae0f6a57c6b61d677274cd96e56ca` before this document itself is committed.

---

# 1. Mission

Team: **ReCurSiON**

Competition: **Kaggriculture**

Primary objective:

> Build the strongest possible deterministic Bonsai/BKG4 agent and pursue **1st place**, not merely a safe incremental leaderboard improvement.

Long-term technical target:

> A deterministic Bonsai portfolio that contains many strong local specialists and routes to them safely enough to approach 100% win rate against the relevant strategic space, including external competitors and ReCurSiON's own historical agents.

The project is not trying to discover one universal global policy. Kaggriculture is treated as a huge combinatorial game with many strategic regimes. The Bonsai tree is valuable precisely because it can hold many specialized subtrees and preserve parent fallback outside their activation regions.

---

# 2. New top-level directive: deterministic pipeline only

The current user directive supersedes older experimental plans that proposed learned/neural search components.

From this point forward the intended first-place pipeline is:

- deterministic final policy,
- deterministic/reproducible search and evaluation,
- deterministic Bonsai/BKG4 specialists,
- no PPO,
- no behavioral cloning,
- no neural game-playing policy,
- no learned neural proposal model,
- no RL-over-edits,
- no neural surrogate as a required component of the primary pipeline.

A deterministic search system may still use sophisticated algorithmic mechanisms such as typed program evolution, deterministic population methods, fixed-seed pseudo-randomized exploration, CEGIS, constraint solving, novelty/QD-style archives, bandit-like operator statistics, best-first search, MCTS-like deterministic scheduling, superoptimization, program synthesis, causal/counterfactual search, or other symbolic methods if justified by evidence.

The planner is explicitly asked to determine the best deterministic architecture rather than inherit ASPE unchanged.

The existing `docs/ASPE_MASTER_PLAN.md` was written before this new directive. Its later learned-proposer phases are now **provisional/obsolete unless replaced by a deterministic mechanism**.

---

# 3. Repository shape

The repository was deliberately cleaned before the current search work.

Primary roots:

- `src/kaggriculture_bonsai/` — active Python package
- `native/bkg4/` — BKG4/native controller support
- `native/cuda/` — CUDA evaluator
- `native/cpp/` — C++ reference engine/oracle
- `assets/champion/` — sole authoritative champion
- `assets/baseline/` — bounded internal route/motif seeds
- `configs/` — current configs
- `scripts/` — current build/experiment entrypoints
- `tests/` — unit/integration/parity tests
- `workspace/` — **tracked compact experiment evidence**
- `external/` — ignored user-supplied HOLDOUT inputs

There is intentionally no old RL stack, replay-corpus stack, EditFormer, B4IM, PureJaxRL, Puffertank, notebook archaeology, old champion archive, or giant scratch tree in the active repository.

Repository hygiene policy:

- no `scratch/`,
- every non-ignored file created by an agent must be committed or deleted,
- `workspace/` is tracked when evidence matters,
- existing `.gitignore` exemptions remain ignored,
- experiment evidence must not be placed under a path component named `build/` because `build/` is intentionally ignored for disposable native/build caches,
- local commits only unless push/submission is explicitly requested.

---

# 4. Runtime architecture: Bonsai/BKG4

The final agent backbone is **Bonsai/BKG4**.

Conceptually:

```text
CURRENT/PARENT POLICY
        |
        +-- specialist A under safe state condition
        +-- specialist B under safe state condition
        +-- specialist C under safe state condition
        +-- ...
```

Important properties:

1. Specialists are allowed to be locally strong and globally inappropriate.
2. Outside a specialist's valid activation region, behavior must fall back exactly to PARENT.
3. The Bonsai capacity is for a portfolio of specialists, not for blindly evolving an arbitrary monolithic 65k-node controller.
4. New specialists should be compact and composable.
5. Final runtime remains deterministic and small enough for Kaggle execution.

Historical evidence already showed that binary specialist routing can successfully preserve capability while removing protected regressions, so local specialization is not merely theoretical.

---

# 5. Exact execution contract

The project maintains three semantic layers:

```text
CUDA search/evaluation engine
        ==
CPU C++ reference engine
        ==
final Bonsai/BKG4 runtime semantics
```

Hard requirements:

- semantic mismatches must remain **0**,
- accepted composition must preserve protected **W->L = 0**,
- large search/evaluation is done on CUDA,
- CPU C++ is the reference semantic oracle,
- before a real live submission/champion promotion, run a bounded 64/128-game official Python/Kaggle-environments canary to verify that native results survive translation to the official engine.

Do not run large official-Python campaigns. The official engine is a release canary, not the main search engine.

---

# 6. Construction data philosophy

The current architecture intentionally avoids opponent-specific replay construction.

BUILD policy construction comes from:

- fresh environment interaction,
- self-play,
- internal adversarial policies,
- internal frozen checkpoints,
- lane/mechanics specialists,
- deterministic challenge generation.

External/public/historical agents are **HOLDOUT/generalization evidence**, not BUILD identities.

This avoids overfitting to named opponents and preserves the ability to generalize to unseen agents.

If replay archives are ever needed later, they should be loaded externally as HOLDOUT evidence rather than restored as the central construction source.

---

# 7. Current challenge-generation system

The project already has a deterministic adversarial self-play/league layer.

The current internal league uses **8 members / 8 strategic lanes**:

1. MARKET_LIQUIDITY
2. WORKER_PRODUCTION
3. LAND_INVESTMENT
4. CROP_TRANSITION
5. ANIMAL_TRANSITION
6. ENDGAME_LIQUIDATION
7. MECHANIC_EXPLOIT
8. GENERAL

These lanes organize challenge coverage. They are not intended to define the entire behavior archive or force specialists to generalize across unrelated lanes.

A local specialist is allowed to specialize inside one capability domain. Generalization means it should work across multiple independent instances of its relevant regime: seeds, seats, checkpoints/trajectories, and related challenge contexts.

---

# 8. Frozen panel architecture

ASPE Phase 0 created a reproducible baseline at:

`workspace/aspe/baseline/`

Manifest schema:

`kaggriculture.aspe.baseline` version 1

Frozen context identity:

`f24636a2ea94ee066488b0c40b17362852b5786b4b66f4b7d96406e8ba862563`

Champion identity:

`357dbb9489b26a8fb424de592da92e45e2c0ac28d8a450a5e6e1b5791ade9d29`

Panel structure:

### BUILD

- 48 distinct regimes
- 3 seeds
- both seats
- all 8 league members
- all 8 lanes
- 16 cases per seed
- 24 cases per seat
- 6 cases per member/lane

### INTERNAL_DEV

- 24 cases
- 3 independent seeds
- both seats
- 4 rotated members/lanes
- 6 cases per selected member/lane

### TRANSFER

- 24 cases
- 3 unique independent seeds
- both seats
- 4 members/lanes

### VALIDATION

- 24 cases
- 3 unique independent seeds
- both seats
- 4 members/lanes

### SEALED

- 24 cases
- 3 unique independent seeds
- both seats
- 4 members/lanes

Cross-panel contract:

- zero seed overlap across all panel pairs,
- zero case-identity overlap across all panel pairs,
- INTERNAL_DEV is not a training/search target,
- SEALED remains a firewall and must not become readable search evidence.

The frozen context was regenerated deterministically and retained the same identity.

---

# 9. Downstream safety pipeline

Current high-level funnel:

```text
fresh self-play / adversarial league
        -> BUILD
        -> specialist discovery/search
        -> INTERNAL_DEV
        -> binary routing / routability
        -> TRANSFER
        -> VALIDATION
        -> CEGIS gate hardening
        -> SEALED
        -> materialization/parity
        -> Bonsai composition
        -> new champion
```

Important distinctions:

- BUILD is search/construction evidence.
- INTERNAL_DEV is independent early generalization screening.
- TRANSFER/VALIDATION/CEGIS/SEALED remain final downstream safety layers.
- Archive/search-memory membership never grants composition authority.
- The final accepted composition still requires zero protected regression and exact semantics.

---

# 10. Historical specialist/routing evidence

The current architecture did prove that imperfect specialists can be useful when routed safely.

A historically important binary specialist (`392b9eb2`) had approximately:

- raw protected W->L = 10,
- raw target L->W = 5,
- routed protected W->L = 0,
- routed target L->W = 4,
- semantic mismatch = 0,
- successful CEGIS/routing,
- compact gate based on legal state.

This demonstrated the core Bonsai thesis:

> A specialist does not need to be globally safe before routing. It needs a useful local capability and a legal observable region in which that capability can be activated without protected regression.

However, later campaigns showed that many candidate bodies are too trajectory-bound for a legal gate to rescue them.

---

# 11. Why the old search systems were rejected as primary oracles

## RLRMB donor-window search

RLRMB searched bounded donor-window/macros around failure trajectories.

It was deterministic, fast on CUDA, and sometimes useful, but later multi-regime experiments exposed a major limitation: candidates were frequently tied to exact seed/seat trajectories rather than reusable strategic state.

Frozen equal-budget baseline:

- discovery budget: 9,600 CUDA games
- candidates: 197
- effect phenotypes: 86
- BUILD-eligible specialists: 0
- cross-regime specialists: 0
- DEV survivors: 0
- downstream/safe commits: 0
- semantic mismatches: 0

Its useful-looking pattern had support from only one seed and one seat and also caused BUILD regressions.

## Direct localized Bonsai search

A direct oracle was implemented to test whether donor-window representation itself was the problem.

Frozen equal-budget baseline:

- discovery budget: 9,600 CUDA games
- candidates: 198
- effect phenotypes: 34
- limited-harm BUILD candidates: 13
- those 13 collapsed to essentially **one strategic phenotype**
- cross-regime specialists: 0
- DEV attempted/passed: 1/0
- DEV L->W/W->L: 0/8
- safe commits: 0
- semantic mismatches: 0

The dominant behavior suppressed/changed the incumbent farmer behavior around step 96. It worked on one seed/seat slice and regressed independent DEV badly.

Main lesson:

> The bottleneck was not raw CUDA throughput. Candidate hypotheses were too low-information, trajectory-bound, or phenotypically redundant.

---

# 12. Search-efficiency problem we are trying to solve

Historically the system could spend roughly hundreds of thousands to around a million game evaluations and obtain only one or two genuinely useful specialists, sometimes none.

This is far too inefficient for Kaggriculture's strategic space.

The desired system should move toward a regime where a small number of meaningful deterministic search cycles routinely produces new promising strategic phenotypes.

The long-run aspiration discussed during ASPE research was roughly:

> 1–2 new promising cross-regime specialist phenotypes per <=10 meaningful search cycles after warm-up.

This is an aspiration, not an established achievable rate.

The planner should define better quantitative targets if appropriate.

Primary metrics should emphasize:

- cross-regime useful phenotypes / 10k CUDA games,
- INTERNAL_DEV survivors / 10k CUDA games,
- downstream survivors / 10k CUDA games,
- safe composed specialists / search budget,
- strategically distinct phenotypes rather than syntax count,
- diminishing duplicate/repeated-failure rate across generations,
- sustained novelty/generalization rather than one-off lucky hits.

---

# 13. ASPE research conclusion that led to current implementation

The previous research proposed **Adaptive Semantic Program Evolution (ASPE)**.

The core hypothesis was:

> Search complete compact strategic programs—WHEN + WHAT + REACT + TERMINATE + FALLBACK—rather than trajectory-local action edits, and accumulate semantic memory so later search does not repeat the same failures.

The research ranked the bottlenecks roughly as:

1. state-conditioned representation / activation semantics,
2. proposal/search intelligence,
3. phenotype-level duplicate suppression,
4. reactive multi-step body search / temporal credit,
5. semantic archive quality.

ASPE is not yet validated as a superior search oracle. Only infrastructure Phases 0–3 are complete.

---

# 14. Completed ASPE Phase 0 — reproducible baseline

Commit:

`fac2253` — `Freeze reproducible ASPE baseline context`

Outcome:

- complete tracked baseline under `workspace/aspe/baseline/`,
- BUILD/DEV/TRANSFER/VALIDATION/SEALED all reproducible,
- disjoint seeds/cases,
- fixed RLRMB and Direct comparison metrics,
- deterministic regeneration proven,
- no search semantics changed.

This is now the authoritative context for equal-budget deterministic search comparisons.

---

# 15. Completed ASPE Phase 1 — SpecialistProgram v1

Commit:

`9726fea` — `Add ASPE SpecialistProgram v1`

Canonical schema:

`kaggriculture.aspe.specialist-program` version 1

Representation:

- 1–6 canonical DNF initiation clauses,
- 1–4 conditions each,
- 1–3 bounded body phases,
- up to 2 forward-only reactive decisions,
- optional bounded recovery phase,
- semantic/event/public-state/hard-cap termination,
- mandatory `RETURN_TO_PARENT`.

Supported strategic public features:

- STEP / DAY / HOUR,
- SHOP_AT / SHOP_SEEN,
- own/opponent money,
- own/opponent units,
- own/opponent land,
- own/opponent hires,
- market price,
- market inventory,
- own shed,
- own seed.

Intentional anti-overfit restrictions:

- worker/tile coordinates are not part of the primary strategic v1 representation,
- exact `STEP == X` predicates are rejected,
- no replay identity,
- no opponent identity,
- no seed fingerprint,
- no future/hidden state.

Semantics:

- false initiation -> exact canonical PARENT action,
- normal completion -> exact PARENT,
- abort -> exact PARENT,
- recovery completion -> exact PARENT,
- interpreter fault -> exact PARENT,
- phase overrides remain action-family scoped so unrelated parent actions remain untouched.

The implementation reused the existing condition grammar, phases/motifs, reactive MacroProgram v2, CPU interpreter, CUDA encoding, materialization path, bounded route macro, and Bonsai compiler rather than creating a new execution stack.

CPU/CUDA parity passed.

---

# 16. Completed ASPE Phase 2 — phenotype telemetry and sentinel

Commit:

`99f8bf1` — `Add ASPE phenotype sentinel screening`

Key addition:

> Search can now distinguish program syntax from actual execution behavior and strategic effect before allocating full BUILD budget.

CUDA telemetry:

- CUDA input IR v3 supports SpecialistProgram DNF initiation gates,
- result ABI v2 uses a compact 96-byte record,
- legacy v1/v2 compatibility retained.

Captured fields include:

- activation steps/count,
- duration,
- phase usage/path,
- reactive branch path,
- termination/completion reason,
- per-family action changes,
- action trace hash,
- faults,
- terminal margin,
- ten strategic state deltas.

Three separate identity layers now exist:

1. structural identity — canonical SpecialistProgram semantic ID,
2. execution phenotype — observed sentinel behavior/action trace,
3. strategic-effect phenotype — margins, L->W/W->L, support geometry, strategic deltas.

Frozen sentinel:

`workspace/aspe/sentinel/`

Identity:

`aab5fac507011fb2929e4b1b3837a58a8e386e936a553257739bf16f86c2d4bb`

Composition:

- 16 frozen BUILD cases,
- factor-balanced across multiple seeds,
- both seats,
- multiple league members,
- multiple lanes.

Screen classifications:

- ZERO_ACTIVATION
- PHENOTYPE_DUPLICATE
- EFFECT_DUPLICATE
- CLEARLY_HARMFUL
- NOVEL_OR_INFORMATIVE

Important rule:

> Uncertain/narrow candidates are retained. Sentinel screening is not an acceptance authority.

Telemetry-enabled v3 and legacy execution produced identical outcomes; parity remained exact.

---

# 17. Completed ASPE Phase 3 — persistent semantic memory

Current repository HEAD before this context document:

`57fe3e1223fae0f6a57c6b61d677274cd96e56ca`

Commit:

`57fe3e1` — `Add persistent ASPE semantic archive`

Archive schema:

`kaggriculture.aspe.semantic-archive` version 1

Evaluation-record schema:

`kaggriculture.aspe.evaluation-record` version 1

Location:

`workspace/aspe/memory/archive.json`

Format:

- compact canonical JSON,
- atomic temp-write,
- file fsync,
- atomic replace,
- directory fsync,
- default bound: 100,000 evaluations.

Authority is always:

`SEARCH_MEMORY_ONLY`

Stored evidence includes:

- canonical SpecialistPrograms,
- lineage,
- parent/generation/operator/mutation site/context,
- execution phenotype,
- strategic-effect phenotype,
- sentinel classification,
- support geometry,
- optional BUILD evidence,
- pre-SEALED downstream evidence,
- margins,
- L->W/W->L,
- seed/seat/member/lane support,
- faults/mismatches,
- failure reasons,
- explicit imperfect stepping-stone status.

Readable SEALED evidence is rejected by the archive.

Dedup hierarchy:

- exact program/context repeats can be no-ops,
- structurally new programs may resolve to old execution phenotypes,
- different execution phenotypes may still resolve to an old strategic effect.

Novelty v1:

- deterministic component-Hamming distance over activation, phase/branch, termination, action-family, trace, and duration descriptors,
- nearest-phenotype queries,
- local competitors,
- novelty fraction,
- first-seen-since-generation queries.

Failure memory is queryable by parent/operator/context.

A restart/integration fixture showed:

```text
4 structural programs
-> 3 execution phenotypes
-> 2 strategic effects
```

while retaining imperfect stepping stones and surviving restart/atomic replacement safely.

No Phase 4 evolution exists yet.

---

# 18. Current exact stopping point

The repository has implemented the **infrastructure necessary to test ASPE**, but has not yet tested the central search thesis.

Completed:

```text
Phase 0: frozen reproducible context
Phase 1: state-conditioned SpecialistProgram
Phase 2: CUDA phenotype telemetry + sentinel screening
Phase 3: persistent semantic search memory
```

Not yet implemented:

```text
Phase 4: typed semantic evolution/search
Phase 5: equal-budget falsification
```

Therefore there is currently **no evidence that ASPE discovers specialists better than RLRMB/Direct**.

This is the most important current truth.

---

# 19. Existing planned Phase 4 — deterministic typed semantic evolution

The current ASPE plan proposes a small deterministic symbolic evolution system as the first actual search oracle.

Candidate mutations include:

- ADD_CONDITION
- REMOVE_CONDITION
- CHANGE_FEATURE
- CHANGE_COMPARATOR
- CHANGE_THRESHOLD
- WIDEN_THRESHOLD
- NARROW_THRESHOLD
- ADD_PHASE
- REMOVE_PHASE
- REPLACE_PHASE
- CHANGE_ACTION_SOURCE
- CHANGE_ACTION_FAMILY
- CHANGE_PARAMETER
- ADD_BRANCH
- REMOVE_BRANCH
- CHANGE_BRANCH_PREDICATE
- CHANGE_TERMINATION
- compatible subtree swaps/reuse later

Proposed parent sources:

- useful/novel archive entries,
- promising imperfect stepping stones,
- occasional champion/baseline-derived seeds,
- distant phenotypes when a niche stagnates.

Proposed population principles:

- small number of islands/populations,
- local phenotype-aware competition,
- occasional migration,
- deterministic operator-success statistics/adaptation,
- explicit exploration,
- no global scalar fitness as the sole parent-selection criterion.

Proposed evaluation funnel:

```text
many typed symbolic proposals
-> validity + structural dedupe
-> 16-case or similarly balanced sentinel
-> execution/effect phenotype dedupe
-> promote informative candidates
-> exact remaining BUILD
-> update archive/failure memory
-> repeat
```

This phase is still only a plan and should be challenged by the superior planner.

---

# 20. Existing planned Phase 5 — equal-budget falsification

The current ASPE plan proposes testing the first deterministic ASPE search against the frozen RLRMB and Direct baselines.

Authoritative discovery budget:

**9,600 CUDA games**

Illustrative allocation only:

- several bounded cycles,
- many candidates get sentinel evaluation,
- a small informative subset receives remaining BUILD cases,
- exact total CUDA ledger remains comparable.

Primary comparison metrics:

- structural programs,
- execution phenotypes,
- strategic-effect phenotypes,
- useful BUILD phenotypes,
- cross-regime BUILD phenotypes,
- seed/seat/member support,
- INTERNAL_DEV attempts/passes,
- duplicate rate,
- repeated-failure rate,
- archive improvements,
- CUDA games/GPU time/wall time.

Initial success signal in the old plan:

- >=2 distinct cross-regime useful phenotypes within the equal budget,
- preferably >=1 INTERNAL_DEV survivor.

Because both frozen baselines achieved zero cross-regime specialists and zero DEV survivors, even one genuine repeatable DEV survivor would be material evidence.

Positive results should be replicated on fresh deterministic context seeds before declaring the architecture solved.

Failure interpretation:

- no cross-regime phenotype -> representation/search still inadequate; do not scale blindly,
- cross-regime BUILD but DEV collapse -> generalization/state abstraction problem,
- DEV survivors but routing failure -> routing becomes a plausible next bottleneck.

The superior planner may redesign this falsification if it can justify a better test.

---

# 21. Superseded old Phase 6–7 idea

The previous ASPE plan proposed:

- Phase 6: train a learned structured proposer from search history,
- Phase 7: use that learned proposer online with explicit control exploration.

Under the **new deterministic-only directive**, this is no longer the intended path.

The new planner must replace these ideas with deterministic mechanisms if cumulative search intelligence is still required.

Possible deterministic research directions to evaluate include, but are not mandates:

- deterministic contextual operator statistics,
- deterministic multi-armed/adaptive allocation,
- typed genetic programming,
- quality-diversity/local competition,
- deterministic island models,
- grammar induction from successful subtrees,
- reusable symbolic macro libraries,
- CEGIS-guided mutation,
- constraint/SMT-guided predicate synthesis,
- deterministic superoptimization,
- best-first/A*/beam over semantic program edits,
- deterministic MCTS/PUCT-like scheduling,
- program-equivalence pruning,
- state clustering/abstract interpretation,
- causal/counterfactual continuation search,
- Go-Explore-like deterministic state archives,
- novelty/failure-driven curriculum,
- automated challenge generation from exact counterexamples,
- portfolio/search-algorithm selection by empirical deterministic evidence.

The planner should choose the smallest coherent architecture that maximizes discovery quality and sustained progress. Do not combine every technique.

---

# 22. Existing planned anti-saturation ideas

The old plan proposed several non-neural ideas that remain relevant candidates:

### Small islands/populations

Maintain semantically different search populations rather than one global basin.

### Reseeding

When one population stagnates:

- reseed from distant elites,
- revive rare stepping stones,
- prioritize underexplored challenge clusters.

### Reusable symbolic library

Mine recurring successful lineages for useful:

- gates,
- body fragments,
- recovery options,
- strategic micro-policies.

Only promote a reusable primitive when it recurs independently. Runtime compilation must still expand it to ordinary deterministic BKG4 nodes.

### Archive aging without forgetting

Reduce active selection pressure on stale dominated elites while retaining provenance and revival ability.

### Challenge scheduling

Prioritize unresolved strategic regions using deterministic evidence such as:

- champion regret,
- unsolved duration,
- recent search progress,
- phenotype scarcity/novelty,
- repeated downstream failure class.

These mechanisms have not been validated yet.

---

# 23. Current autonomous target loop

If a future search oracle earns promotion, the intended production loop is roughly:

```text
CURRENT CHAMPION
      |
      v
DETERMINISTIC ADVERSARIAL SELF-PLAY / LEAGUE
      |
      v
UNSOLVED CHALLENGE BANK
      |
      v
DETERMINISTIC SPECIALIST SEARCH
      |
      v
CROSS-REGIME BUILD CANDIDATES
      |
      v
INTERNAL_DEV
      |
      v
BINARY ROUTING
      |
      v
TRANSFER -> VALIDATION -> CEGIS -> SEALED
      |
      v
BONSAI COMPOSITION
      |
      v
NEW CHAMPION
      |
      +----------------------------> repeat
```

The superior planner should decide whether this is sufficient for 1st place or whether major missing deterministic components exist.

---

# 24. Engine/mechanics opportunity

Kaggriculture has exploitable deterministic mechanics and quirks. Historical investigation identified areas such as:

- shop replacement/duplicate behavior,
- market scarcity/inventory pricing behavior,
- sequential commit/quote interactions,
- shed overflow/discard behavior,
- consecutive-unfed animal escape behavior,
- fertilizer refresh interactions,
- CARE/fed-production timing,
- other exact engine edge cases.

The project explicitly permits exploiting game mechanics/bugs when legal in the competition, analogous to high-level competitive game play.

The MECHANIC_EXPLOIT lane exists, but historically mechanics-specific search was underdeveloped relative to its possible value.

A first-place planner should decide whether a dedicated deterministic exploit-discovery subsystem is a missing major lever.

---

# 25. Known bottlenecks and uncertainties

The following are **not settled**:

## A. ASPE itself is unproven

Phases 0–3 are infrastructure. No typed semantic evolutionary campaign has run.

## B. Search representation may still be too weak

SpecialistProgram v1 is much richer than donor windows, but long-horizon strategies may require:

- larger temporal options,
- state snapshots/continuation search,
- richer bounded state machines,
- causal long-horizon credit,
- strategic macro composition.

## C. Challenge generation may eventually saturate

The current 8-lane adversarial league is much better than replay-only discovery, but 8 manually named lanes are tiny relative to the true game space.

The planner should decide whether challenge generation needs a deterministic open-ended mechanism that automatically creates new strategic regimes beyond the lane taxonomy.

## D. Current BUILD geometry may still be too small

48 regimes are excellent for bounded comparison but cannot represent the full game.

The planner should distinguish:

- small fixed experimental panel for fair oracle comparison,
- expanding deterministic challenge bank for production search,
- protected holdout panels for promotion.

## E. Search may need better temporal credit

Terminal win/loss/margin is authoritative but strategically sparse.

Phase 2 now exposes compact intermediate state deltas and action traces. The planner should determine whether deterministic counterfactuals, continuation evaluation, milestone values, or causal attribution can materially improve search without introducing neural value models.

## F. Search may repeat local optima despite memory

Phase 3 prevents exact repetition and supports novelty, but the actual parent/mutation/search scheduler does not exist yet.

## G. Safe composition at scale is unproven

Binary routing has worked for individual specialists, but a very large specialist portfolio may introduce:

- overlapping activation regions,
- route conflicts,
- interaction regressions,
- ordering problems,
- node-budget/latency concerns,
- accumulated CEGIS complexity.

The planner should consider whether composition/routing needs a future deterministic global verification layer once the portfolio grows large.

## H. We lack a convincing deterministic first-place convergence argument

The current architecture can improve the agent, but it has not proven that it can keep finding major capability indefinitely.

The superior planner's most important task is to design a deterministic search/challenge/composition loop whose productivity does not collapse after easy specialists are exhausted.

---

# 26. What historically did NOT work well

Do not casually revive these without overwhelming new evidence:

- PPO/neural RL as the main policy,
- behavioral cloning,
- replay-corpus imitation as the main construction method,
- neural rankers/EditFormer infrastructure,
- giant opponent-specific rule banks,
- global-policy optimization that ignores the Bonsai specialist philosophy,
- naive replay donor-window search as the sole oracle,
- exact-step/local action editing as the sole oracle,
- scaling CUDA games without improving proposal information,
- more screening layers that do not improve candidate quality,
- delayed routing when legal earlier signals are absent,
- microportfolio complexity before search produces good specialists,
- cosmetic infrastructure work with no expected capability gain.

---

# 27. What is already strong and should not be casually rebuilt

Unless the superior planner finds concrete evidence of a correctness/strategic flaw, preserve:

- deterministic Bonsai/BKG4 runtime,
- CUDA exact evaluator,
- CPU C++ oracle,
- exact CPU/CUDA semantic contract,
- Phase-0 frozen baseline infrastructure,
- SpecialistProgram v1 execution semantics,
- Phase-2 phenotype telemetry/sentinel,
- Phase-3 semantic archive,
- independent BUILD/DEV/downstream panels,
- binary parent/specialist fallback concept,
- CEGIS/legal gating,
- zero protected W->L final safety,
- clean repository structure.

The superior planner may change the **search architecture** aggressively if evidence supports it.

---

# 28. First-place planning questions that must be answered

A new planner should explicitly answer at least the following.

1. Is the current deterministic ASPE direction fundamentally the best response-oracle architecture, or should Phase 4 be replaced before implementation?
2. What deterministic search algorithm can increase useful-specialist yield by orders of magnitude rather than marginally?
3. How should the system explore the near-infinite trajectory/state/program space without collapsing into fewer than ten strategic basins?
4. How should challenge generation itself evolve beyond eight manually named lanes?
5. How should failures become actionable deterministic counterexamples rather than just archive rows?
6. What is the best deterministic way to perform long-horizon credit assignment?
7. Should search operate from full games, strategic state snapshots, forked continuations, or a hierarchy of all three?
8. Can constraint solving/CEGIS synthesize initiation gates and program structure more efficiently than mutation alone?
9. Can program superoptimization or enumerative synthesis exploit the exact simulator better than evolutionary search?
10. What role should deterministic QD/islands/novelty play, if any?
11. How should useful symbolic subtrees/macros be discovered, canonicalized, recombined, and promoted into a reusable search vocabulary?
12. How should the system detect and exploit deterministic engine mechanics systematically rather than manually?
13. How should production challenge panels expand over time without contaminating independent validation?
14. How should a very large specialist portfolio be globally checked for routing/composition interactions?
15. What stopping/saturation metrics tell us that a search strategy has exhausted its useful neighborhood?
16. What deterministic mechanism should replace the old neural learned-proposer idea for cumulative search intelligence?
17. What exact phase-by-phase sequence gives the highest expected probability of reaching 1st place quickly?
18. Which current phases should be deleted, merged, or reordered?
19. What bounded experiments can falsify each major assumption before expensive implementation?
20. What are the highest-leverage missing components that the current plan has not considered at all?

---

# 29. Research standard for the superior planner

The planner should not produce another stack of fashionable algorithms connected by thin glue.

Expected methodology:

1. Start from the actual evidence summarized here.
2. Challenge every existing assumption, including ASPE.
3. Search the web/research literature/source implementations broadly.
4. Study deterministic combinatorial-search domains outside RL:
   - program synthesis,
   - superoptimization,
   - SAT/SMT,
   - CEGIS,
   - automated planning,
   - game-tree search,
   - genetic programming,
   - quality diversity,
   - novelty search,
   - automated algorithm discovery,
   - compiler optimization,
   - fuzzing/adversarial test generation,
   - theorem proving,
   - constraint programming,
   - exact/heuristic combinatorial optimization.
5. Prefer algorithms that exploit the exact deterministic CUDA evaluator and state-forking opportunities.
6. Evaluate implementation/source code of the strongest candidates where useful.
7. Identify one coherent primary architecture, not ten equal alternatives.
8. Produce a phased implementation plan with explicit falsification gates.
9. Keep the implementation lightweight unless a complex subsystem has a demonstrated major role.
10. Optimize for **first-place capability gain per engineering week and per CUDA game**, not elegance.

---

# 30. Constraints for future execution agents

When the research plan is accepted, implementation agents should:

- follow the planner's phases sequentially,
- stay on `main`,
- verify HEAD/status,
- work locally,
- use CUDA for heavy search,
- avoid heavy CPU workloads,
- run focused tests rather than unrelated giant suites,
- commit each completed phase locally,
- keep the repository clean,
- never create `scratch/`,
- never push or submit unless explicitly asked,
- never relax final semantic/safety contracts to manufacture apparent search success.

---

# 31. Current decision boundary

The project is intentionally paused **before ASPE Phase 4**.

Do not assume Phase 4 should be implemented next merely because it is the next section in `docs/ASPE_MASTER_PLAN.md`.

The superior planner/researcher is being introduced precisely to answer:

> Given everything now known, what deterministic end-to-end architecture should ReCurSiON actually build from this point to maximize the probability of finishing 1st?

It may:

- keep Phase 4,
- redesign Phase 4,
- replace ASPE's search core,
- add missing deterministic subsystems,
- reorder future work,
- narrow the plan substantially,
- or propose a new coherent deterministic pipeline.

Completed Phases 0–3 should be treated as reusable infrastructure, not as sunk-cost constraints on the architecture.

---

# 32. Current concise state

```text
Bonsai/BKG4 runtime            STRONG / KEEP
CUDA exact evaluator           STRONG / KEEP
CPU reference parity           STRONG / KEEP
self-play/adversarial league   USEFUL, may need broader challenge evolution
BUILD/DEV safety separation    STRONG / KEEP
binary routing/CEGIS           PROVEN useful, future scale interactions unproven
RLRMB search                   BASELINE, weak as primary
Direct local search            BASELINE, weak as primary
SpecialistProgram v1           IMPLEMENTED, search value unproven
phenotype telemetry            IMPLEMENTED
semantic archive               IMPLEMENTED
ASPE evolution                 NOT IMPLEMENTED
ASPE superiority               NOT PROVEN
learned/neural proposer        NOW OUT OF SCOPE
first-place convergence plan   NEEDS NEW RESEARCH
```

The next planner should use this document as its main repository context and spend its effort finding the strongest deterministic path from this exact point forward.
