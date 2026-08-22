# Permanent BKG4 Edit Search

`tools/bkg4_search/edit_search/` is ReCurSiON's permanent recursive typed structural-search system for the BKG4 Bonsai backbone. Its job is to produce diverse, exact-CUDA-labeled Bonsai edit programs that are useful for BonsaiEditFormer training and future BKG4 improvement. It is not a champion-promotion mechanism by itself.

## Non-negotiable invariants

- BKG4/Bonsai remains the deterministic policy backbone.
- Large search/evaluation campaigns use exact CUDA (`B4EC1/v1`). `ExactCudaEvaluator` has no CPU/Python game-evaluator fallback.
- CPU C++ and CUDA semantics must remain bit-exact. Run `parity.py` after any evaluator/grammar change.
- A final submission candidate still requires a 64/128-game official Python Kaggriculture canary before promotion.
- Safety is a hard feasibility constraint, not a scalar reward. Margin gain never compensates for a new loss, win-to-loss transition, or configured severe protected regression.
- Concrete search state is tied to one `ParentProfile`. Symbolic motifs may transfer to a new parent only by reconstruction and fresh exact evaluation.
- Model-visible context is recursively allowlisted public information only. Search provenance, screening scores, opponent identity, and future outcome truth never become model features.
- Search, corpus, and QD persistence fail closed on version, identity, referential-integrity, or transaction mismatches.

Current persistence versions:

```text
SEARCH_STATE_SCHEMA_VERSION = 3
EDIT_PROGRAM_SCHEMA_VERSION  = 2
GRAMMAR_ID                   = bkg4_module_genome_edit_v2
EMITTER_REGISTRY_VERSION     = 3
QD_DESCRIPTOR_SCHEMA_VERSION = 2
CORPUS_SCHEMA_VERSION        = 3
```

## Architecture

- `schema.py` — `ParentProfile`, canonical state-relative `EditProgram`/`EditAtom` identities, reconstruction.
- `safety.py` — hard root/source feasibility, violation distance, safety tiers, protected-tail diagnostics.
- `goals.py` — public-state `SearchGoal` construction and contrastive predicate synthesis.
- `emitters.py` — contrastive, systematic, local, structural, motif, recombination, LNS, evolvability/MCTS, and permanent random-typed emitters.
- `archive.py` — authoritative `CaseRegistry`, exact certifications, production/repair/research QD, novelty, motifs, emitter contribution stats, hard cases, runtime transaction ledger.
- `panels.py` — mixed production CaseBank validation, deterministic screens, hard-case prioritization, Stage-1 detection-recall governance.
- `fitness.py` — exact parent→child case labels and safety-first lexicographic fitness.
- `evaluator.py` — one batched B4GM1→B4EC1 CUDA invocation per exact state batch.
- `search.py` — constrained parent lanes, lexicase selection, multi-goal scheduling, multi-fidelity evaluation, contribution-aware emitter allocation, persistent goal caches, anti-stagnation escalation.
- `corpus.py` — crash-consistent exact Parquet transactions, immutable generation identities/snapshots, recovery, strict recomputation validation, model-boundary auditing.
- `benchmark.py` — same-root equal-budget proposer comparison and bounded anti-saturation harness.
- `parity.py` — non-empty ModuleGenome CPU C++↔CUDA parity on shared normalized B4TP1 tapes.
- `cli.py` — persistent init/resume/search/validation/benchmark lifecycle.

## Hard safety and parent lanes

Every exact candidate is assessed relative to both its immediate source parent and the archive root. Root feasibility controls production admission.

A candidate is infeasible if it introduces any of:

- a new loss,
- a win→loss transition,
- a configured severe protected regression.

Safety tiers are:

```text
ELITE_FLIP
ELITE_MARGIN
STRICT_SAFE_USEFUL
SAFE_INCREMENTAL
BOUNDARY
HARD_NEGATIVE
CONTROLLED_NOOP
```

The archive keeps three separate populations:

1. **production** — root-feasible behavior-changing QD elites;
2. **repair** — root-infeasible states ordered by distance to feasibility before reward/novelty;
3. **research/exploration** — novelty evidence that never silently becomes a normal production parent.

Normal parents are root + production elites. Repair-capable emitters may receive a bounded repair-parent mixture. Research novelty cannot reset strategic stagnation by itself.

## Mixed production CaseBank

Production search rejects an all-loss or otherwise degenerate CaseBank. A production bank must contain meaningful safety contrast, including losses, protected wins, fragile protected wins, strong protected wins, both seats, and multiple strategy families.

The final validation bank used 64 current cases:

```text
20 losses
16 fragile wins
28 strong wins
44 protected wins total
```

A disjoint 48-case holdout was kept out of SearchGoals/QD during discovery.

## Contrastive SearchGoals

`SearchGoal` clusters current failures using legal public state context and pairs them with protected/fragile guards. Contrastive synthesis is intentionally conservative:

- conjunction predicates must co-occur in the same probe frame;
- synthesis uses stable public regime features rather than worker/tile coordinate memorization;
- if no predicate excludes protected guards, the contrastive emitter emits nothing for that goal;
- when probe timing is available, specialists receive exact `STEP==t` or bounded `STEP` interval guards;
- the existing B4GM1 ABI already supports up to four predicates/rule, so three/four-condition specialists are represented as canonical multi-edit programs without an ABI change;
- a parent's scheduled contrastive budget is distributed across multiple failure goals rather than one seed-selected goal;
- guarded condition sets are cached per `(CaseBank identity, goal_id)` across generations and invalidated automatically when either identity changes.

The final direct exact-CUDA temporal-specialist proof evaluated 96 candidates on the 64-case mixed panel:

```text
hard-feasible:             96 / 96
safe-useful:               26
elite-tier:                20
programs with L2W:          7
new losses:                 0
W2L:                        0
severe regressions:         0
best L2W:                   2
best mean margin delta: +1317.65625
```

## Stage-1 / Stage-2 recall governance

Stage-1 detection recall and Stage-2 selection recall are separate diagnostics. A small Stage-2 budget is not incorrectly reported as poor Stage-1 detection.

Periodic full audits recompute exact truth for all proposals and promote any missed safe candidate into the constrained archive. If detection recall falls below the configured floor, the next generation widens the screen and Stage-2 budget. The final validation demonstrated the full feedback loop:

```text
16-case screen / 20 Stage-2 -> L2W recall miss -> widen to 24 / 30
24 / 30 -> later useful-recall miss -> widen to 36 / 45
36 / 45 -> useful=1.0, L2W=1.0, danger=1.0
```

## Contribution-aware emitter allocation

The scheduler keeps explicit base weights and a permanent random floor, but exact contribution now changes later allocation. Emitters with sufficient attempts are scaled by safe-useful yield and production/repair insertions. Contrastive specialists have a first-class budget and are excluded from generic refill so unavailable motif/recombination capacity cannot collapse a generation into one aggressive family.

Unused capacity is refilled by broader typed emitters. One-slot allocations round-robin operation/target families instead of being captured by alphabetic candidate order.

## Proposal throughput

Two host bottlenecks were removed without changing CUDA truth:

1. shared deterministic typed pools avoid regenerating the same large typed neighborhood for every emitter/parent;
2. contrastive conjunction scoring uses frame-coherent bitsets plus per-goal cross-generation caching.

On the validator's 33-state workload, shared typed pools changed 64-candidate proposal throughput from about **3.22 unique/s to 20.73 unique/s**; 128 and 256 candidate probes reached **27.02/s** and **51.65/s** respectively.

On the much richer final 193-state archive with 14–19 source parents and 11–12 operation families, a 64-proposal cold/warm pair measured **3.22/s → 4.88/s**. A single rich SearchGoal's synthesis fell from **16.65 s to 1.41 s (~11.8×)**. Large exact evaluation remains CUDA-resident; host proposal work is bounded and deterministic.

## Crash-consistent corpus transactions

Each exported generation has an immutable transaction identity tied to:

- parent/search identity,
- generation number,
- canonical archive digest,
- exact program IDs.

A transaction writes program/label/parent shards and an immutable `archive.after.json` snapshot before the corpus manifest commit point. Recovery:

- rolls back prepared/uncommitted process-death artifacts;
- reconstructs stale canonical archive state from a committed immutable snapshot;
- rejects corpus/archive transaction or digest mismatches;
- never repairs an archive by copying generation numbers alone.

Caught exceptions roll back the partial generation and permit an exact retry. An initialized zero-generation corpus is resumable, while strict final export validation still rejects a corpus with no complete generation.

## Production lifecycle

Create a fresh archive:

```bash
python3 -m tools.bkg4_search.edit_search.cli init \
  --parent-dir bonsai-kg/integrations/bkg4_champion/champions/w33_s172_expansion_v1 \
  --archive scratch/edit_search/run/search.json \
  --run-id recur-001 --seed 99173 --root-prefix 0
```

Run production search:

```bash
python3 -m tools.bkg4_search.edit_search.cli search \
  --archive scratch/edit_search/run/search.json \
  --parent-dir bonsai-kg/integrations/bkg4_champion/champions/w33_s172_expansion_v1 \
  --case-bank scratch/edit_search/case_bank.jsonl \
  --observations scratch/edit_search/observations.jsonl \
  --evaluator scratch/cuda-engine-build/bkg4_edit_corpus_eval \
  --corpus scratch/edit_search/run/corpus \
  --work-dir scratch/edit_search/run/work \
  --start-generation 1 --rounds 4 \
  --candidates 4096 --screen-size 128 --final-size 512 --stage2-budget 512 \
  --recall-audit-every 4 --seed 99173
```

For large corpus campaigns, use a large candidate pool with a 64/128-case CUDA screen and exact 512-case final labeling only for retained candidates; periodic full audits govern screen recall. Do not increase final-panel size merely to compensate for poor candidate throughput.

Validate the exact corpus:

```bash
python3 -m tools.bkg4_search.edit_search.cli validate-corpus \
  --corpus scratch/edit_search/run/corpus \
  --archive scratch/edit_search/run/search.json
```

## Equal-budget benchmark

The benchmark compares historical shallow, random typed, fixed systematic, and permanent recursive search under one shared root and identical child×case exact-CUDA budgets. The permanent arm receives the same mixed CaseBank/SearchGoals used in production; contrastive search is not silently disabled during benchmarking.

`--require-material-improvement` checks catastrophic safety, safe-useful/repair yield, genuine recursion/multi-parent behavior, and structural diversity. Use multiple deterministic seeds.

## Non-empty CPU C++↔CUDA parity

```bash
python3 -m tools.bkg4_search.edit_search.parity \
  --parent-dir bonsai-kg/integrations/bkg4_champion/champions/w33_s172_expansion_v1 \
  --cases-tsv scratch/bkg4_editformer_jax/current140_cases7.tsv \
  --cuda-evaluator scratch/cuda-engine-build/bkg4_edit_corpus_eval \
  --work-dir scratch/edit_search/nonempty_parity \
  --case-count 4
```

The harness uses three non-empty legal overlays and fails on any CPU fault, CUDA fault, final-money mismatch, or action-trace mismatch.

## Promotion gate

Search/corpus readiness is not champion readiness. Before any discovered program modifies the official ReCurSiON agent:

1. choose a candidate using exact search-panel **and disjoint held-out** evidence;
2. freeze its semantic identity/assets;
3. run exact CPU C++↔CUDA parity for that candidate/grammar;
4. run a 64/128-game official Python Kaggriculture canary against current, historical, and own-history opponents;
5. reject any regression or unexplained translation discrepancy;
6. only then use the serialized champion promotion/submission workflow.

See `VALIDATION.md` for the final bounded evidence supporting corpus-generation readiness.


## Permanent certified-elite generator (schema v4)

The production search now separates three evidence roles instead of treating discovery success as final truth:

- `search_training`: the discovery bank plus a rotating certification-repair slice; this is the only role visible to proposal/CEGIS and corpus model context;
- `certification`: a disjoint held-out slice used only to issue `CertificationVector` / `CertificationClass` evidence;
- `ranker_validation`: frozen by identity and never materialized into training rows.

Discovery quality and certification quality are separate labels. A discovery elite that is unsafe on certification becomes a generalization hard negative; safe-but-inert transfer becomes a generalization boundary. Certified elites require independent safe, active, useful transfer. Search exploitation is certification-aware: certified motifs/donors and certified MCTS states outrank discovery-only states, while unsafe states remain confined to repair/exploration lanes.

The permanent CEGIS cascade repairs root-safety counterexamples with public-state guards, then evaluates repaired descendants exactly. Certified-positive states receive a bounded exact neighborhood over route, threshold, priority, and temporal-window alternatives. High-information corpus sampling retains certified positives, generalization failures, near-feasible repairs, and local elite contrasts while capping controlled no-ops at 10% when alternatives exist.

Proposal generation is cache/batch oriented: deterministic contrastive SearchGoal condition sets persist across restart; MCTS typed neighborhoods cache per semantic state; typed pools are bounded/reused per parent; immutable observation context is built once per generation and parent/goal contexts are shallow derivatives.

The benchmark accepts either legacy `--cases-tsv` or rich `--case-bank`. `--production-targeted --target-screen-size N` is an optional diagnostic that conserves the exact child×case CUDA budget by spending part of the budget on screening and the remainder on full-panel finalists. The primary final discovery benchmark remains the raw equal-budget four-arm comparison, while production safety is governed by the audited adaptive multi-fidelity funnel.

Final measured evidence is recorded in `VALIDATION.md`.
