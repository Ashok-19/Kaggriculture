# Adaptive Semantic Program Evolution (ASPE) — Master Implementation Plan

## Status

- Team: **ReCurSiON**
- Architecture target: **deterministic Bonsai/BKG4 runtime, self-play discovery**
- Research baseline: `500268a` equal-budget RLRMB vs Direct-Bonsai bakeoff
- Planning baseline: `434027f`
- Historical implementation record: completed Phases 0-3 remain reusable infrastructure.
- Unimplemented Phases 4 onward are superseded by [FIRST_PLACE_MASTER_PLAN.md](FIRST_PLACE_MASTER_PLAN.md); learned/neural proposer phases are out of scope.
- This is an implementation plan, not evidence that ASPE already works.

## Strategic objective

The repository already has a strong deterministic runtime, exact CPU/CUDA evaluation, adversarial self-play, multi-regime BUILD, INTERNAL_DEV, binary routing, CEGIS, and downstream safety. The weak link is specialist proposal quality.

The goal of ASPE is to make the search system increasingly intelligent about **which compact deterministic specialist program to evaluate next**, while keeping the final submission entirely deterministic Bonsai/BKG4.

The target is not more syntactic candidates. The target is a much higher rate of:

1. genuinely different execution/strategic phenotypes,
2. cross-seed, both-seat, multi-member BUILD improvements,
3. INTERNAL_DEV survivors,
4. ultimately safe composed specialists.

The long-run aspiration is roughly 1–2 new promising cross-regime specialist phenotypes per <=10 meaningful adaptive search cycles after warm-up. Safe commits will naturally be rarer because all existing safety gates remain authoritative.

---

# Non-negotiable constraints

1. **Bonsai/BKG4 remains the final runtime.** Learned or stochastic machinery may guide search, but successful programs must materialize to deterministic BKG4.
2. **CPU/CUDA semantics must remain exact.** Any semantic mismatch is a correctness failure.
3. **Final protected W->L remains zero.** ASPE never weakens TRANSFER, VALIDATION, CEGIS, SEALED, routing, or composition safety.
4. **BUILD remains fresh internal self-play/adversarial evidence.** Public/historical opponents are not construction data.
5. **INTERNAL_DEV remains independent.** Search must not optimize against fixed DEV cases.
6. **Parent fallback is mandatory.** An inactive or aborted specialist returns exactly to incumbent behavior.
7. **Do not search arbitrary 65k-node trees.** Bonsai capacity is for the final portfolio; individual specialists remain compact.
8. **CUDA is the large-scale evaluator.** Do not replace cheap exact batched evaluation with heavy Python simulation.
9. **No `scratch/` directory.** Do not recreate it.
10. **Repository hygiene is mandatory.** Every non-ignored file created by an agent must be committed or deleted before finishing. Existing `.gitignore` exemptions remain the only allowed untracked/generated locations.
11. **`workspace/` is tracked evidence.** Use it for compact reproducible experiment evidence. Do not leave arbitrary temporary files there.
12. **Do not put experiment evidence in a directory named `build/`.** The current `build/` ignore pattern is intentionally exempt from tracking and is reserved for disposable native/build caches such as `workspace/build/`.
13. **Do not train a learned proposer before the representation thesis is validated.** Phase A must prove useful cross-regime specialists first.
14. **RLRMB and Direct-Bonsai remain baselines until ASPE wins on evidence.** Do not delete them during early phases.

---

# Baseline evidence that ASPE must beat

The equal-budget bakeoff used the same champion, league, BUILD, INTERNAL_DEV and downstream firewall for both current oracles.

## Frozen search context

- League: 8 internal members / 8 strategic lanes
- BUILD: 48 regimes = 3 seeds x 2 seats x 8 members
- INTERNAL_DEV: 24 cases from independent seeds and rotated members
- Discovery budget: 9,600 CUDA games/oracle
- Conditional downstream allowance: 864 games/oracle

## RLRMB

- 197 candidates
- 86 unique effects
- best raw BUILD L->W: 8
- BUILD-eligible specialists: 0
- cross-regime specialists: 0
- DEV survivors: 0
- safe commits: 0

The useful-looking effect was tied to one seed and one seat and also caused BUILD W->L.

## Direct Bonsai

- 198 candidates
- 34 unique effects
- 13 limited-harm BUILD candidates
- all useful candidates collapsed to one strategic phenotype
- cross-regime specialists: 0
- DEV attempted/passed: 1/0
- DEV L->W/W->L: 0/8
- safe commits: 0

The dominant phenotype suppressed/changed the incumbent around step 96, worked only on one seed/seat slice, and generalized catastrophically.

## Required metric vocabulary

Every ASPE phase must distinguish at least:

- raw proposals,
- structurally unique programs,
- execution phenotypes,
- strategic-effect phenotypes,
- useful BUILD phenotypes,
- cross-regime BUILD phenotypes,
- INTERNAL_DEV survivors,
- TRANSFER survivors,
- safe commits,
- CUDA games,
- GPU time,
- wall time,
- phenotype duplicate rate,
- repeated-failure rate,
- archive improvements.

Primary discovery efficiency metrics:

- **cross-regime useful phenotypes / 10k CUDA games**,
- **DEV survivors / 10k CUDA games**.

Raw candidate count and raw effect-signature count are diagnostic only.

---

# Phase overview

| Phase | Purpose | Heavy search? | Gate to continue |
|---|---|---:|---|
| 0 | Reproducible ASPE baseline + contracts | No | Complete, tracked, deterministic context |
| 1 | SpecialistProgram v1 representation | No | Exact fallback + CPU semantics + round-trip tests |
| 2 | CUDA phenotype telemetry + sentinel funnel | Tiny | Exact parity and phenotype-collapse detection |
| 3 | Persistent semantic memory + archive | No/very small | Correct dedupe, lineage, replayability |
| 4 | Phase-A typed semantic evolution | Bounded CUDA | Produces broad, non-trivial phenotypes |
| 5 | Equal-budget Phase-A falsification | 9.6k games/run | Cross-regime phenotype(s), ideally DEV survivor |
| 6 | Learned proposer dataset + offline model | No large games | > control enrichment on held-out search history |
| 7 | Online adaptive proposer | Bounded CUDA | Clear enrichment while preserving exploration |
| 8 | Anti-saturation: islands, reuse, curriculum | Bounded/medium CUDA | Sustained novelty/generalization over generations |
| 9 | Production autonomous growth integration | CUDA campaign | ASPE safely outperforms old oracles end-to-end |
| 10 | Scale, hardening and submission readiness | Large CUDA only after proof | Stable growth + parity + official canary |

Phases are sequential decision gates. Do not implement later sophistication to rescue a failed earlier premise.

---

# Phase 0 — Freeze the ASPE experimental contract

## Goal

Create one complete, reproducible baseline context that future ASPE runs can consume without depending on ignored/missing files.

## Why this phase exists

The current `workspace/oracle_bakeoff/context/build/` evidence was affected by the generic `build/` ignore pattern. DEV/TRANSFER/VALIDATION/SEALED evidence is tracked, but the exact BUILD context is not fully represented in Git. ASPE needs a clean baseline before changing search.

## Work

1. Reuse the current champion from `assets/champion/`.
2. Recreate the equal-context internal league and all panels using the existing deterministic generators.
3. Store the new frozen context under a tracked path that does **not** contain a `build/` path component, for example:

   `workspace/aspe/baseline/context/build_panel/`

4. Keep all seeds disjoint between BUILD, INTERNAL_DEV, TRANSFER, VALIDATION and SEALED.
5. Record one compact manifest containing:
   - champion identity,
   - league member identities,
   - panel identities,
   - seeds/seats/members/lanes,
   - tool/config version,
   - source HEAD,
   - exact case counts.
6. Record the current RLRMB/Direct baseline summary without rerunning unnecessary large experiments if the existing results are sufficient.
7. Add a schema/version for ASPE experiment manifests.

## Repository targets

Primarily:

- `workspace/aspe/baseline/`
- `configs/` only if a clean reusable ASPE baseline config is required
- small helper code only if existing deterministic context generation cannot produce the manifest cleanly

## Verification

- all expected evidence paths are tracked or intentionally ignored by existing `.gitignore`,
- BUILD and DEV have zero seed/case overlap,
- manifest regeneration is deterministic,
- champion identity is exact,
- no `scratch/` paths exist,
- clean Git status after commit.

## Exit gate

Do not begin Phase 1 until the frozen ASPE baseline is fully reproducible from current source and its required non-ignored evidence is committed.

---

# Phase 1 — SpecialistProgram v1: search the complete strategic hypothesis

## Goal

Replace trajectory-local candidate semantics with a compact state-conditioned specialist program that jointly expresses:

- **WHEN** to activate,
- **WHAT** to do,
- **HOW** to react while active,
- **WHEN** to terminate,
- **WHEN** to abort/fall back.

This phase tests representation only. Do not add learned proposal logic.

## SpecialistProgram v1 shape

Start deliberately small:

- 1–6 initiation clauses,
- 1–4 conditions per clause,
- 1–3 body phases,
- 0–2 reactive branch decisions,
- one semantic/bounded termination policy per phase,
- optional tiny finite-state memory only if already supported cleanly,
- mandatory exact PARENT fallback.

The first strategic predicate set should emphasize legal public state:

- STEP / DAY / HOUR,
- SHOP_AT / SHOP_SEEN,
- OWN/OPP_MONEY,
- OWN/OPP_UNITS,
- OWN/OPP_LAND,
- OWN/OPP_HIRES_TODAY,
- MARKET_PRICE,
- MARKET_INVENTORY,
- OWN_SHED,
- OWN_SEED.

Exact worker/tile coordinates should not dominate v1 strategic discovery. They may remain available later for proven tactical niches.

## Reuse before invention

Inspect and reuse the existing:

- `composition/module_codec.py`,
- `composition/macro_initiation_cegis.py`,
- `search/ir.py`,
- `search/motifs.py`,
- `search/evaluator.py`,
- current BKG4 compiler/materializer.

Do not create a second unrelated policy language if the current grammar can express the required semantics.

## Required invariants

1. No opponent identity, replay identity, tape identity or seed fingerprint may be a runtime predicate.
2. Inactive specialist = exact parent action.
3. Abort/termination fallback = exact parent behavior.
4. Program canonicalization is deterministic.
5. Program serialization round-trips exactly.
6. The same SpecialistProgram has identical CPU and CUDA semantics before any search campaign.
7. Existing champion behavior is unchanged when no ASPE program is active.

## Suggested code shape

Prefer a small package rather than spreading ASPE across current files:

- `src/kaggriculture_bonsai/search/aspe/genome.py`
- `src/kaggriculture_bonsai/search/aspe/semantics.py`

Only split further when a real responsibility requires it.

## Tests

Add focused tests for:

- condition canonicalization,
- program identity,
- parent fallback,
- phase/branch semantics,
- termination,
- invalid program rejection,
- compile/materialize round-trip,
- representative CPU parity.

## Exit gate

Phase 1 passes only when a compact state-conditioned reactive program is a first-class deterministic object with exact parent fallback and parity-safe semantics.

No search-quality claim is required yet.

---

# Phase 2 — Execution phenotype telemetry and cheap sentinel screening

## Goal

Stop spending full BUILD budget on syntactically different programs that behave identically.

## Three levels of identity

ASPE must distinguish:

### Structural identity

Same canonical program/AST.

### Execution phenotype

Same or near-equivalent behavior across a sentinel panel, including:

- activation pattern,
- first/last activation step,
- activation count,
- action-change pattern/family,
- phase/branch path,
- duration,
- termination reason.

### Strategic-effect phenotype

Same meaningful strategic result pattern, including:

- L->W/W->L bitmap,
- seed/seat/member support,
- useful state deltas,
- harmful state deltas.

## Sentinel panel

Use a small factor-balanced subset across:

- multiple seeds,
- both seats,
- multiple league members,
- multiple strategic lanes/challenge clusters.

Do not simply take the first N cases.

## CUDA telemetry

Extend the exact evaluator only as much as necessary to return compact phenotype fields. Useful v1 fields include:

- activated,
- first activation step,
- last activation step,
- activation count,
- action changes,
- phase/branch path summary,
- termination reason,
- selected strategic deltas at activation, +24, +72, termination/terminal where feasible.

Keep payloads compact and deterministic.

## Critical regression fixture

The prior Direct-Bonsai result found 13 syntactically different useful candidates that effectively represented one phenotype. Create a compact fixture/test demonstrating that the new phenotype pipeline collapses this kind of duplication before full BUILD allocation.

## Repository targets

Likely:

- `search/aspe/phenotype.py`
- existing `search/evaluator.py`
- `native/cuda/` only if exact telemetry needs native changes
- parity/unit tests

Do not create a separate simulator.

## Exit gate

Phase 2 passes when:

- CPU/CUDA results still match,
- semantic mismatches remain zero,
- known equivalent behavior collapses to the same execution phenotype,
- genuinely different behavior remains distinct,
- sentinel screening can reject duplicates/zero-value behavior without final safety authority.

---

# Phase 3 — Persistent semantic memory and search archive

## Goal

Make every evaluated candidate useful as future search evidence instead of becoming disposable log output.

## Authoritative search memory

For each evaluated program preserve compact structured evidence:

- program ID,
- parent ID,
- lineage/generation/island,
- challenge cluster/lane hint,
- canonical genome,
- proposal operator and mutation site,
- structural hash,
- execution phenotype,
- strategic-effect phenotype,
- BUILD case outcomes and support,
- intermediate telemetry,
- DEV/downstream result when available,
- failure classification,
- final composition outcome if ever reached.

## Storage policy

Because repository evidence is tracked:

- keep authoritative compact run/search evidence in tracked `workspace/aspe/...`,
- prefer append-only JSONL/compact deterministic shards/manifests for durable evidence,
- large disposable indexes, compiled binaries and caches may live only under paths already ignored by the current `.gitignore` (for example `workspace/build/`),
- do not commit giant redundant raw intermediates merely because they were generated,
- delete non-authoritative temporary non-ignored files before finishing a task.

## Semantic archive

The archive should use meaningful phenotype descriptors rather than only raw effect hashes or 8 manual lanes.

Initial descriptor components:

1. activation behavior,
2. intervention/action-family behavior,
3. duration/branch/termination behavior,
4. selected strategic state deltas,
5. outcome support geometry.

Manual strategic lanes remain useful challenge metadata, not the complete niche definition.

## Local competition

Retain multiple strategically different stepping stones even when they are not final-safe specialists. Search-archive admission does **not** grant composition authority.

## Exit gate

Phase 3 passes when:

- duplicate evaluations are recognized before repetition where possible,
- lineage is reconstructable,
- search evidence is deterministic/replayable,
- archive diversity corresponds to actual behavioral diversity,
- imperfect stepping stones can be retained without touching champion safety.

---

# Phase 4 — Phase-A ASPE: typed semantic evolution

## Goal

Test the core thesis with the smallest coherent search system:

**state-conditioned program representation + typed evolution + phenotype memory + exact CUDA evaluation.**

No neural proposer yet.

## Mutation vocabulary

Start with typed local operators such as:

- ADD_CONDITION,
- REMOVE_CONDITION,
- CHANGE_FEATURE,
- CHANGE_COMPARATOR,
- CHANGE_THRESHOLD,
- WIDEN_THRESHOLD,
- NARROW_THRESHOLD,
- ADD_PHASE,
- REMOVE_PHASE,
- REPLACE_PHASE,
- CHANGE_ACTION_SOURCE,
- CHANGE_ACTION_FAMILY,
- CHANGE_PARAMETER,
- ADD_BRANCH,
- REMOVE_BRANCH,
- CHANGE_BRANCH_PREDICATE,
- CHANGE_TERMINATION,
- SWAP_REUSABLE_SUBTREE when compatible.

Each mutation must preserve a valid typed program or fail cheaply before CUDA.

## Parent selection

Parents should come from:

- current useful/novel archive entries,
- promising partial stepping stones,
- occasional baseline/champion-derived seeds,
- deliberately distant phenotypes when a niche stagnates.

Do not use global scalar fitness alone.

## Initial population structure

Keep it lightweight:

- small number of islands (roughly 4–8 at most),
- local phenotype-aware competition,
- occasional migration of genuinely useful subtrees/programs,
- simple contextual operator-success statistics/bandits are allowed,
- no learned neural proposer.

## Evaluation funnel

A typical bounded cycle should look like:

1. produce many typed symbolic proposals,
2. structural validation/dedup,
3. small balanced CUDA sentinel,
4. execution/strategic phenotype dedup,
5. promote only informative candidates,
6. exact remaining BUILD cases,
7. update archive and negative memory,
8. repeat.

The exact counts may be tuned while preserving a fixed authoritative CUDA-game ledger.

## Negative learning without ML

Repeatedly useless operator/context combinations should receive less proposal budget. Preserve explicit exploration so rare novel mechanisms remain discoverable.

## Exit gate

Phase 4 passes technically when the Phase-A oracle can run end-to-end on a bounded budget and produce auditable structural/phenotype/archive metrics with zero semantic mismatch.

Search superiority is decided only in Phase 5.

---

# Phase 5 — Equal-budget Phase-A falsification

## Goal

Answer the decisive question:

> Can state-conditioned typed semantic evolution produce genuine multi-seed/both-seat specialists where RLRMB and Direct-Bonsai produced zero?

## Frozen comparison

Use the Phase-0 context and unchanged:

- champion,
- league,
- BUILD,
- INTERNAL_DEV,
- TRANSFER/VALIDATION/SEALED interfaces.

Do not tune ASPE against DEV.

## Budget

Match the prior discovery budget:

- **9,600 CUDA discovery games** for ASPE.

A suitable illustrative allocation is five cycles of approximately 1,920 games each, for example:

- 160 candidates x 8 sentinel cases = 1,280,
- 16 promoted x 40 remaining BUILD cases = 640,
- total 1,920/cycle.

Counts are not sacred; the 9,600-game ledger is.

## Required report

Compare ASPE directly with frozen RLRMB and Direct results on:

- raw proposals,
- structurally unique programs,
- execution phenotypes,
- strategic-effect phenotypes,
- BUILD L->W/W->L,
- seed/seat/member support,
- cross-regime useful phenotypes,
- DEV attempts/passes,
- duplicate rate,
- repeated-failure rate,
- archive improvements per cycle,
- CUDA games/GPU time/wall time.

## Initial success signal

Because both baselines achieved zero cross-regime specialists and zero DEV survivors, the first meaningful target is:

- >=2 distinct cross-regime useful phenotypes within the equal budget,
- and preferably >=1 INTERNAL_DEV survivor.

A single genuine DEV survivor is already important evidence, but should be replicated before declaring the architecture solved.

## Replication rule

If the first bounded ASPE run is positive, repeat on fresh deterministic context seeds before moving to learned proposal work. Do not claim victory from one lucky panel.

## Failure branches

### A. No cross-regime phenotype

Stop. Do **not** add a learned proposer. Representation/search semantics are still inadequate. Reopen only targeted research around:

- richer temporal options,
- state snapshot/continuation search,
- larger but still compact body composition,
- Go-Explore-style continuation mechanisms.

### B. Cross-regime BUILD improves but DEV still collapses

Improve state abstraction/generalization pressure using fresh BUILD diversity and better semantic descriptors. Do not leak DEV into training.

### C. DEV survivors appear but routing fails

Only then does routing become the next primary bottleneck.

## Exit gate

Phase 5 is the major architecture gate.

Proceed to Phase 6 only if Phase-A demonstrates repeatable cross-regime capability and at least credible DEV generalization under the fixed budget.

---

# Phase 6 — Learned proposer dataset and offline search-policy learning

## Prerequisite

Phase 5 must pass. Otherwise this phase is prohibited.

## Goal

Use accumulated exact evaluation history to learn which structured program edits are worth proposing next.

The model guides search only. It never enters the submission runtime and never certifies safety.

## Training examples

Construct examples from evaluated parent/candidate relationships containing:

- challenge descriptor,
- parent program features,
- parent phenotype,
- archive neighborhood,
- recent lineage/failures,
- chosen edit site/operator,
- predicate feature/comparator/threshold,
- action/body choice,
- termination/branch choice,
- exact resulting outcomes.

Prefer pairwise/ranking targets initially:

- candidate A > candidate B for this parent/challenge,
- productive operator > unproductive operator in this context,
- lower regression risk,
- higher archive improvement,
- higher probability of cross-regime support.

## Model scope

Keep the first proposer lightweight and structured. It may predict factorized distributions such as:

- parent,
- edit site,
- operator,
- predicate feature,
- comparator,
- threshold region,
- action source/family,
- parameter,
- branch,
- termination.

Optional heads may estimate:

- archive improvement,
- BUILD L->W likelihood,
- regression likelihood,
- cross-regime probability,
- novelty.

Do not train an end-to-end game-playing model.

## Offline evaluation

Hold out complete generations/contexts so the proposer cannot memorize individual search episodes.

Required metrics:

- enrichment of useful proposals over uniform/control proposals,
- top-k precision for productive edits,
- regression-risk ranking,
- novelty preservation,
- calibration where meaningful.

## Exit gate

Do not integrate online unless the proposer shows a clear held-out enrichment signal. A useful target is roughly >=2x enrichment over uniform/control after sufficient data, but evidence quality matters more than hitting a cosmetic number.

---

# Phase 7 — Online adaptive proposer with explicit exploration

## Goal

Make evaluation N+1 informed by the accumulated evidence from evaluations 1..N while preventing model collapse.

## Initial proposal allocation

Use an explicit mixture such as:

- 70–80% learned/high-score proposals,
- 10–20% novelty/uncertainty proposals,
- ~10% approximately uniform control proposals.

The control stream is mandatory. It proves whether learning is actually improving proposal quality.

## Online update loop

Per bounded cycle:

1. choose challenge/archive parents,
2. propose structured edits,
3. symbolic validation/dedup,
4. sentinel CUDA evaluation,
5. phenotype dedup,
6. exact BUILD evaluation for promoted candidates,
7. update ProgramDB/archive,
8. update proposer on newly labeled examples,
9. preserve control statistics.

## Required measurements

- learned vs control useful-phenotype rate,
- learned vs control cross-regime rate,
- learned vs control duplicate rate,
- learned vs control regression rate,
- cycle-1 vs later-cycle enrichment,
- archive improvement trend,
- proposal entropy/diversity.

## Exit gate

The learned path must beat control without collapsing diversity. If learned proposals become repetitive or no better than control, fall back to typed semantic GP/evolution and redesign the proposer separately.

---

# Phase 8 — Anti-saturation: sustained innovation

## Goal

Prevent ASPE from becoming another fixed-neighborhood search that works briefly and then stalls.

This phase adds only mechanisms justified by observed saturation.

## 8.1 Island/reseed policy

Maintain a small number of semantically different populations. When an island stagnates:

- reseed from distant elites,
- sample rare reusable components,
- prioritize underexplored challenge clusters,
- preserve successful stepping stones instead of resetting globally.

## 8.2 Challenge scheduling

The adversarial league remains the challenge generator. Prioritize challenge clusters using evidence such as:

- champion regret,
- unsolved duration,
- recent improvement potential,
- proposer uncertainty,
- strategic novelty.

Do not build a full POET-style environment generator unless the existing league itself is proven to be the bottleneck.

## 8.3 Reusable search library

Mine repeated successful lineages for recurring substructures:

- gates,
- body fragments,
- recovery options,
- strategic micro-policies.

Promote a fragment to a reusable **search primitive** only when it recurs across independent successful lineages/regimes.

Compilation must expand it back to ordinary deterministic BKG4 nodes. Runtime must not depend on the learned library.

## 8.4 Archive aging without forgetting

Reduce active selection pressure for stale dominated elites while retaining provenance and the ability to revive distant stepping stones.

## Exit gate

Across repeated generations, the system should continue producing new execution/strategic phenotypes rather than merely increasing syntax count. If phenotype yield and DEV yield flatten despite healthy challenge diversity, diagnose the representation/proposer again rather than scaling compute blindly.

---

# Phase 9 — Autonomous self-play Bonsai growth

## Goal

Make ASPE the primary response oracle inside the existing autonomous loop once it has earned that role experimentally.

## Target loop

```text
CURRENT CHAMPION
      |
      v
ADVERSARIAL SELF-PLAY / LEAGUE
      |
      v
UNSOLVED CHALLENGE BANK
      |
      v
ASPE PROGRAM SEARCH
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
      +---------------------------> repeat
```

## What stays unchanged

Do not redesign without concrete evidence:

- champion runtime,
- CUDA exact evaluator authority,
- C++ reference oracle,
- self-play/adversarial league,
- BUILD/DEV independence,
- binary routing,
- TRANSFER,
- VALIDATION,
- CEGIS,
- SEALED,
- zero protected W->L,
- composition.

## Oracle policy

RLRMB/Direct may remain as occasional baselines/seed generators if complementary evidence justifies them. ASPE becomes primary only if its end-to-end safe discovery rate is superior.

## Exit gate

Require repeatable autonomous generations with:

- no semantic mismatches,
- no protected W->L in accepted composition,
- measurable new cross-regime capability,
- at least some downstream conversions,
- clean restart/reproducibility behavior,
- bounded resource use.

---

# Phase 10 — Scale, harden, and prepare release

## Goal

Only after the search architecture is proven, optimize throughput and reliability for long campaigns.

## Scaling work

Then, and only then, consider:

- larger CUDA candidate batches,
- asynchronous proposal/evaluation queues,
- persistent evaluator improvements,
- compact ProgramDB sharding/indexing,
- long-run checkpointing,
- bounded multi-GPU execution if actually useful,
- search-memory retention policies.

Do not optimize Python micro-overhead before proposal quality is proven.

## Pre-submission contract

Before any live submission/champion promotion intended for Kaggle:

1. current champion must pass local CPU/CUDA parity,
2. accepted specialist composition must preserve protected W->L = 0,
3. semantic mismatches must be 0,
4. run a bounded 64/128-game official Python/Kaggle-environments canary,
5. confirm native search results translate exactly to official engine behavior,
6. package only the deterministic BKG4 runtime/assets.

The learned proposer, archive and search database are development tools only.

---

# Repository implementation map

ASPE should fit the cleaned repository rather than recreate old architecture sprawl.

## Retain unchanged unless correctness requires otherwise

- `src/kaggriculture_bonsai/league/`
- `src/kaggriculture_bonsai/routing/`
- `src/kaggriculture_bonsai/runtime/`
- `src/kaggriculture_bonsai/evaluation/`
- `src/kaggriculture_bonsai/composition/`
- `native/bkg4/`
- `native/cuda/`
- `native/cpp/`
- `assets/champion/`

## Primary new search area

Prefer:

`src/kaggriculture_bonsai/search/aspe/`

Start with only responsibilities that Phase 1–4 genuinely need, likely:

- `genome.py` — SpecialistProgram structure/canonical identity,
- `semantics.py` — compile/interpreter helpers if existing IR does not suffice,
- `phenotype.py` — execution/strategic descriptors and dedupe,
- `memory.py` — compact persistent search evidence/archive interface,
- `evolution.py` — typed mutation, parent selection, Phase-A loop.

Later only:

- `proposer.py` — learned structured proposal policy,
- `library.py` — reusable abstraction mining.

Do not create placeholders for later phases before they are needed.

## Scripts/configs

Add at most a small number of explicit current entrypoints, for example:

- `scripts/run_aspe_phase_a.py`
- later `scripts/run_aspe_campaign.py`
- `configs/aspe_phase_a.json`

Prefer extending the existing campaign entrypoint if that is cleaner than another script.

## Tests

Mirror current layout:

- unit tests for genome/canonicalization/mutations/phenotypes,
- integration tests for sentinel/archive/evolution,
- parity tests for CUDA/CPU SpecialistProgram execution.

---

# Evidence and Git policy for every phase

This project intentionally keeps the working tree auditable.

For every implementation phase:

1. verify HEAD/status first,
2. do not create `scratch/`,
3. use tracked `workspace/aspe/<phase-or-run>/` for compact evidence that must persist,
4. use only existing ignored paths for disposable caches/build outputs,
5. avoid path components named `build/` for evidence that must be tracked,
6. delete temporary non-ignored artifacts that are not worth retaining,
7. commit all remaining non-ignored files,
8. leave the repository clean,
9. do not push or submit unless explicitly requested.

Every bounded experiment committed to `workspace/aspe/` should include one concise machine-readable result/manifest sufficient for the next agent to understand:

- source HEAD,
- config identity,
- panel/context identity,
- CUDA game ledger,
- candidate/phenotype funnel,
- generalization metrics,
- semantic mismatch count,
- final verdict.

---

# Kill switches and anti-overengineering rules

Stop and reconsider instead of adding complexity when any of these occur:

1. **Phase 1 representation cannot be made parity-exact.** Fix semantics first.
2. **Phase 2 telemetry changes evaluator outcomes.** Revert/fix before search.
3. **Phase 4 mainly creates phenotype duplicates.** Improve representation/mutations; do not add more CUDA.
4. **Phase 5 yields zero cross-regime specialists.** Do not train a proposer.
5. **Learned proposer fails to beat explicit control.** Fall back to typed semantic evolution.
6. **BUILD improves but DEV repeatedly collapses.** Revisit state abstraction/generalization, not routing complexity.
7. **Search diversity grows only by syntax/effect hash.** Fix phenotype descriptors.
8. **A proposed subsystem has no measurable role in the current bottleneck.** Do not add it.
9. **Large campaign requested before bounded falsification passes.** Refuse the scale-up and finish the gate first.

---

# Definition of ASPE success

ASPE is not successful because its code exists, its archive is large, or it evaluates many candidates.

It succeeds when, under equal exact CUDA budgets, it demonstrates a repeatable improvement over RLRMB/Direct in the funnel that matters:

```text
strategically distinct proposals
        -> cross-regime BUILD phenotypes
        -> INTERNAL_DEV survivors
        -> downstream survivors
        -> safe composed specialists
```

The architecture is mature only when the search policy demonstrably becomes more productive from accumulated experience without sacrificing strategic diversity or exact safety.

Until that evidence exists, every phase remains falsifiable and reversible.
