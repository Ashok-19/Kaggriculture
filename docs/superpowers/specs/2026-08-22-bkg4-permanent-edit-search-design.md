# Permanent BKG4 Edit Search Design

## Goal

Build one reusable, resumable, GPU-first program-search subsystem that takes a supplied BKG4 parent profile, recursively explores typed deterministic route-module descendants, preserves diverse stepping stones, feeds broader-panel counterexamples back into future screens, and exports exact CUDA-labeled ranker corpora without modifying the production champion.

## Evidence-driven boundary

BKG4 `SPEC_V4.md` explicitly requires large-policy search to operate on a frozen base plus bounded mutable semantic modules. The existing `ModuleGenome` is a legal route-selector module with 0..8 ordered rules, 1..4 legal public predicates per rule, and a 0..719 activation prefix. The current `actor_grammar.py` already defines the typed edit operations and deterministic `apply_edit()` semantics.

Therefore search-state schema v1 uses `ModuleGenome` as its concrete mutable backend. It does **not** claim that current `REPLACE_SUBTREE` is arbitrary BKG4 subtree replacement: in v1 it means replacing one route-module rule. `REPLACE_MACRO` currently means a route substitution. Future richer bounded BKG4 modules may implement the same search interfaces under a new grammar/backend version without changing BKG4 ABI 4 or the archive/search engine.

## Architecture

```text
ParentProfile + CaseBank + public context statistics
                 |
                 v
        persistent SearchArchive
                 |
        parent/QD/lexicase selection
                 |
      +----------+-----------+
      |          |           |
 systematic   local/struct   recombine/LNS/MCTS/random
      |          |           |
      +----------+-----------+
                 |
         canonical EditProgram
                 |
      typed apply + static validation
                 |
        semantic/pair dedup
                 |
                 v
        exact CUDA Stage-1 panel
                 |
      QD insertion + parent discovery
                 |
     selected boundary/negative/novel
                 |
                 v
        exact CUDA Stage-2 panel
                 |
   corpus rows + counterexample feedback
```

Python is lightweight orchestration only. It batches complete `ModuleGenome` states into the existing `B4GM1` format and invokes `bkg4_edit_corpus_eval.cu`; there is no per-game Python evaluation.

## Versioned identities

Independent versions are recorded for:

- search-state schema
- EditProgram schema
- grammar/backend
- emitter registry
- QD descriptor schema
- B4EC1 CUDA result ABI
- BKG4 ABI
- feature registry
- corpus schema

A parent profile records concrete controller/route/legacy assets, combined parent identity, legal route IDs, BKG4/registry versions, and compatibility slots. Concrete candidates are parent-specific. Motifs/templates are abstract and may be re-instantiated against a future parent before exact reevaluation.

## Candidate representation

`EditProgram` contains:

- source parent semantic ID
- one or more canonical `EditAction` atoms
- each atom's exact state-relative target kind/rule/condition location
- all typed semantic parameters
- intermediate semantic IDs
- final child semantic ID
- emitter/proposal provenance as bookkeeping

Targets are resolved against the intermediate state immediately before each atom. Applying the complete program must deterministically reproduce every intermediate ID and final child. Program identity hashes source parent + canonical atoms + intermediate/final identities.

## Exact evaluation semantics

`StateEvaluation` stores one exact result vector per semantic state on a named case panel. `PairEvaluation` derives parent→child labels from two state result vectors with the existing `derive_case_label()` / `derive_edit_labels()` semantics.

Production-quality comparison is lexicographic and never allows average margin to compensate for a new W→L or severe protected regression. The ordering is:

1. W→L / new-loss regressions
2. severe protected regressions
3. losses after
4. L→W improvements
5. fragile wins after
6. worst candidate margin
7. worst protected-win delta / target-loss improvement
8. mean margin delta
9. complexity

Search steering may additionally use novelty/evolvability, but promotion-quality ordering never becomes a weighted sum.

## Search archive

Persist separately:

- all known semantic states needed for reconstruction
- production QD elites
- exploratory/unsafe QD elites
- novelty representatives
- hard counterexamples
- reusable motifs
- champion/search lineage
- exact panel certifications
- evaluated semantic IDs and parent→child program IDs
- per-emitter attempts, unique proposals, archive insertions, useful-safe hits, failures
- stagnation/escalation history

QD niches are compact hashed descriptor tuples rather than a Cartesian grid. Structural descriptors include rule-count bucket, prefix/window bucket, operation family, target kind, feature family, program length, and complexity delta. Exact-evaluated behavioral descriptors add changed-case-count and action-trace signature buckets.

## Parent selection

Search never uses one incumbent only. Parent sources include production QD elites, exploratory elites, novelty representatives, direct descendants with evolvability evidence, and lexicase-selected case specialists. Unsafe stepping stones remain clearly separated from promotion-safe elites.

## Initial emitters

All emitters share one deterministic interface and return complete `EditProgram`s.

1. `systematic`: exhaustive/bounded local enumeration of legal targets, route alternatives, priority moves, prefix/window neighbors, comparator/threshold neighborhoods, condition additions and lane splits. Threshold candidates come from observed legal-state values/quantiles, existing thresholds, local offsets and domain constants.
2. `local_refine`: small semantic-coordinate mutations around many archive elites.
3. `structural`: specialist/recovery insertion, predicate composition, branch split/delete, bounded rule replacement.
4. `recombine`: combine compatible rules/motifs from two archive states and express the composition as a reconstructable typed program.
5. `lns`: bounded destroy/repair programs that may contain several atoms and are judged on the completed child rather than requiring each intermediate to improve.
6. `evolvability_mcts`: bounded typed derivation search using UCB with direct quality plus descendant-neighborhood/evolvability evidence; no neural policy required.
7. `random_typed`: permanent nonzero exploration floor/canary.

Adaptive emitter allocation is optional and off by default. A fixed scheduler is the reference; UCB allocation is retained only if an equal-budget benchmark beats the fixed allocation.

## Case panels and CEGIS feedback

A generic `CaseBank` accepts league/replay/synthetic/internal-adversary native tapes plus legal bookkeeping. Search does not depend on replay-derived edits.

`CasePanelManager` deterministically builds Stage-1 panels from current losses, fragile wins, protected wins, both seats, family diversity, hard counterexamples and random anchors. Broader-panel failures add exact case IDs to the hard-case archive. Future panel selection prioritizes these counterexamples. Repeated hard cases are counted/deduplicated by stable case identity; unique catastrophic cases are never silently discarded.

Stage-1 recall against Stage-2 truth is measured for useful-safe, L→W and dangerous-regression cases. Panel refresh is triggered when configured recall floors deteriorate.

## Anti-stagnation

A rolling monitor tracks duplicate-child rate, unique proposal rate, QD insertions, strict-safe useful hits, target-loss improvements, novelty, parent concentration and emitter yields. Consecutive exhausted windows advance an explicit escalation level:

0. ordinary neighborhoods
1. wider scalar/window neighborhoods
2. broader QD parent selection + more structural mutation
3. recombination from distant niches
4. LNS destroy/repair
5. deeper multi-edit/MCTS
6. novelty/rare-niche targeting + hard-case refresh

The monitor guarantees only operational regime change, not mathematical endless improvement.

## Corpus export

Only exact Stage-2 evaluations may enter exported training shards. Each exported program stores parent/child structures, complete canonical edit program, target/action payloads, public case context, exact margins/deltas/outcomes, trace/effect fields, search provenance, depth and QD descriptors.

Logical categories are explicit: strict-safe useful, safe incremental, boundary, hard negative, controlled no-op, novel/diverse.

Search provenance remains `search_*` bookkeeping and is excluded from model feature payloads. Export includes grouping keys for exact edit/program, parent family, child semantic ID, action skeleton, operation+target combination, lineage and run temporal holdout.

Validator fails closed on unresolved identities, reconstruction mismatch, malformed CUDA rows, duplicate edit×case keys, label recomputation mismatch, schema mismatch, leakage, or unstable canonical IDs.

## Bounded validation

Local validation consists of unit/integration tests, proposal-only anti-saturation tests, a tiny exact CUDA smoke, and a bounded equal-CUDA-budget benchmark against the old shallow generator, random typed exploration and a simple fixed mutator. No large local CPU campaign and no champion promotion are part of this task.
