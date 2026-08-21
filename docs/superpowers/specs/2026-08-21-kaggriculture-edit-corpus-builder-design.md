# Kaggriculture CUDA-Labeled BonsaiEditFormer Corpus Builder Design

## Objective

Build a second Kaggle notebook whose only job is to transform the persistent top-10 replay archive into a compact, exact-CUDA-labeled, incrementally maintained BonsaiEditFormer training corpus. It must not train a model, modify the champion, or submit to the competition.

## Core architecture

The pipeline is:

`top-10 replay archive -> replay-derived opponent tapes/cases -> current-BKG4 exact CUDA baseline -> representative hard-case panel -> typed legal edit generation -> exact CUDA candidate labels -> canonical structured corpus -> persistent Kaggle dataset version`

The corpus stores canonical structured supervision rather than current neural tensor caches. Future ranker code may tensorize the corpus differently without rerunning expensive CUDA labels.

## Reused implementation

Reuse these existing components rather than reimplementing semantics:

- `tools/bkg4_search/bonsai_editformer/actor_grammar.py` for the typed edit grammar, target references and legality checks.
- `tools/bkg4_search/module_codec.py` for canonical `ModuleGenome` semantics and semantic IDs.
- `tools/bkg4_search/campaign.py` for proven domain-aware condition/mutation generation where useful.
- `tools/bkg4_search/cuda_io.py` for the B4GM1 codec and result parsing conventions.
- `tools/bkg4_search/build_league.py` / `prepare_optimization_oracle.py` for replay-to-native-tape semantics (`replay.steps[t+1][seat].action`).
- `cuda-engine` and `bonsai-kg` current BKG4 runtime for exact labels.
- `tools/replay_archive` notebook generation and persistent-dataset publication patterns.

## Current parent

The initial base parent is the current 345-node BKG4 champion lineage at:

`bonsai-kg/integrations/bkg4_champion/champions/w33_s172_expansion_v1/`

The corpus records its combined identity, controller semantic identity, controller structure, route pack identity, BKG4 ABI and feature registry. The schema supports more base parents later.

A corpus parent is represented as:

- base BKG4 controller identity/structure;
- optional typed overlay `ModuleGenome` identity/structure;
- parent semantic identity derived from both.

This permits edits at typed overlay nodes while preserving the actual 345-node base-tree location via an explicit base insertion/location descriptor. The storage schema does not assume the current fixed single-predicate/current400 representation.

## Replay streaming and case construction

The replay archive is never unpacked wholesale. Process one `replays_YYYY-MM-DD.parquet` shard at a time and delete temporary native tapes/captures/results after the day is committed.

For each source day:

1. Read only that replay shard plus matching rows from `episodes.parquet` / `top10_history.parquet`.
2. Convert top-10 participant action streams to compact native tapes.
3. Exact-deduplicate identical action programs within the day.
4. Build both-seat evaluation cases using the replay seed only as simulator provenance, never as a model feature.
5. Run the current BKG4 baseline on CUDA.
6. Deterministically retain a diverse hard panel emphasizing losses, ties/near-losses, fragile wins, daily-rank/meta-period coverage, both seats, and a bounded sample of robust wins.
7. Capture legal public-state context at fixed strategic probes with the existing BKG4 information-state CUDA capture. Hidden opponent truth output is discarded and never enters the corpus.

Team name, replay ID, seed ID and source paths are provenance-only columns prefixed `prov_`; they are not part of model feature payloads.

## Candidate generation

Generate typed edits using the existing grammar and `apply_edit`. The generator intentionally spans all operation families that are legal for the current parent/overlay:

- ADD_SPECIALIST
- ADD_CONDITION
- CHANGE_THRESHOLD
- CHANGE_ROUTE
- CHANGE_PRIORITY
- CHANGE_WINDOW
- INSERT_RECOVERY
- REPLACE_MACRO
- SPLIT_LANE
- COMPOSE_PREDICATES
- REPLACE_SUBTREE
- REMOVE_REDUNDANT_BRANCH

Generation mixes:

- observed-state thresholds from legal public-state captures;
- deterministic threshold perturbations;
- domain-aware random conditions;
- route changes;
- specialist/lane/subtree mutations;
- deliberately broad or weak candidates likely to be neutral/harmful.

The admission pipeline does not pre-filter candidates for apparent quality. Negative/regressive/no-effect examples are required. Candidates are canonicalized and deduplicated by parent semantic identity + canonical `EditAction` + resulting child semantic identity.

## Exact CUDA evaluator

Add one minimal evaluator, `cuda-engine/src/bkg4_edit_corpus_eval.cu`, by composing the current `bkg4_champion_v4_eval.cu` runtime with the existing ModuleGenome condition decoder.

For each `(candidate, case)` it executes the exact current BKG4 controller, current route pack, legacy compatibility overlay, accepted engine exploits and feed reconciliation. A typed overlay may select a legal route; otherwise exact incumbent behavior is preserved.

The result ABI stores:

- final own money;
- final opponent money;
- action-change count versus the no-overlay action at candidate-trajectory states;
- activation count;
- runtime flags.

Genome zero is the empty overlay and provides exact baseline truth for the same panel.

## Labels

Every admitted edit row is derived only from exact CUDA results. Store raw outcomes plus derived labels:

Per case:

- baseline margin;
- candidate margin;
- signed delta;
- baseline/candidate outcome class;
- L->W;
- W->L;
- severe regression (`baseline > 0 && delta < -3000`);
- activation count;
- action-change count;
- behavior-changing flag.

Per edit aggregate:

- L->W count;
- W->L count;
- severe regression count;
- mean margin delta;
- worst protected-win delta;
- loss count before/after;
- fragile-win count before/after;
- worst candidate margin;
- strict-safe indicator;
- exact-no-op indicator;
- activation/effect totals;
- lexicographic utility components in project order.

The primary ordering remains:

1. eliminate losses;
2. avoid new losses;
3. eliminate fragile wins;
4. improve worst-case margin;
5. improve broader margin.

No replay heuristic substitutes for CUDA truth.

## Corpus storage

Persistent dataset: `ashok205/kaggriculture-bonsai-edit-corpus`.

Files:

- `corpus_manifest.json`: global schema/provenance/counts and file integrity metadata.
- `processed_shards.json`: replay date/shard completion markers and source shard identity.
- `base_structures.parquet`: one row per base BKG4 controller containing canonical controller structure and identities.
- `parents.parquet`: one row per base+overlay parent identity.
- `cases.parquet`: deduplicated legal case/context records; model-feature payload is separate from provenance payload.
- `edits.parquet`: one row per unique evaluated semantic edit and aggregate CUDA labels.
- `edit_case_labels.parquet`: one row per edit x case exact CUDA outcome.
- `shards/cases_YYYY-MM-DD.parquet`, `shards/edits_YYYY-MM-DD.parquet`, `shards/labels_YYYY-MM-DD.parquet`: append-only date shards used to avoid rewriting one huge table during processing; compact consolidated indexes are rebuilt only at publication.

Model-visible structured payloads are JSON/categorical/numeric semantic fields, not opaque current-model tensor arrays.

## Incremental behavior

First run processes every replay archive date chronologically. Subsequent runs:

1. validate prior corpus manifest;
2. compare replay archive processed dates/shard hashes to `processed_shards.json`;
3. fail if an already-processed source shard changed;
4. process only unseen dates;
5. deduplicate new parent/case/edit identities against retained corpus indexes;
6. mark a date complete only after its shard files validate;
7. publish a new version of the same dataset only when at least one new date is committed.

A failed date publishes nothing; a later run resumes from the last published dataset version.

## Support dataset

Persistent support dataset: `ashok205/kaggriculture-edit-corpus-support`.

A local packager copies only:

- exact required CUDA headers/sources;
- exact required BKG4 headers/CUDA adapter/runtime headers;
- current 345-node controller JSON/BKGC/routes/legacy routes/identities;
- typed edit grammar/module codec/CUDA codec/corpus runtime Python helpers;
- an explicit support manifest with source commit, component versions and useful content hashes.

The notebook compiles from this mounted support dataset. Rebuild/version it whenever any included CUDA/search/BKG4 source changes.

## Validation

Local/unit validation must cover:

- replay -> native tape action alignment;
- deterministic case selection and both-seat construction;
- no identity/replay/seed leakage in model feature payloads;
- typed edit legality and canonical dedup;
- exact label derivation including L->W/W->L/severe/no-op cases;
- processed-shard idempotency and mutation rejection;
- prior-corpus validation and incremental preservation;
- deterministic notebook generation and correct GPU/input metadata;
- support bundle contains the exact minimal required paths and rejects missing inputs.

Bounded CUDA validation must compile the new evaluator and compare its empty-overlay baseline with the existing current BKG4 CUDA evaluator on a small known local fixture. Candidate smoke evaluation uses only a few genomes/cases. No large campaign is run locally.

Kaggle-side smoke instructions then run one small replay shard/subset, verify GPU compilation/evaluation, rerun idempotency, and only then enable the full historical notebook run.
