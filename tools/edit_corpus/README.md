# Kaggriculture BonsaiEditFormer CUDA edit-corpus builder

This subsystem builds **training data only**. It does not train BonsaiEditFormer, modify/promote BKG4, or submit to Kaggle.

## Data flow

```text
ashok205/kaggriculture-top10-replay-archive
  -> one replays_YYYY-MM-DD.parquet shard at a time
  -> replay action tapes + both candidate seats
  -> exact current-BKG4 CUDA baseline margins
  -> deterministic hard/fragile representative cases
  -> legal/public BKG4 R0 information-state capture
  -> typed semantic EditAction candidates
  -> exact CUDA parent/child evaluation
  -> compact canonical Parquet corpus
  -> ashok205/kaggriculture-bonsai-edit-corpus
```

The runtime never expands the full replay archive into `/kaggle/working`. Only one source day is read at a time; temporary native tapes, capture files, genome packs and CUDA outputs for that day are deleted before the next day.

## Exact parent and evaluator

Initial base BKG4 parent:

```text
bonsai-kg/integrations/bkg4_champion/champions/w33_s172_expansion_v1
combined identity: 6359ab400f9d128945cdbba5df7178a33b88c643ddd9b488299d66e14daa6925
controller nodes: 345
BB slots: 23
BKG4 ABI: 4
feature registry: 1
```

`cuda-engine/src/bkg4_edit_corpus_eval.cu` reuses the current BKG4 VM, route materialization, legacy compatibility layer and accepted engine-exploit layer. It does not reimplement simulator semantics. `B4EC1/v1` stores exact own/opponent final money, activation/effect counters and an executable-action trace fingerprint per `(overlay, case)`. Parent-relative behavior/no-op labels compare parent and child trace fingerprints rather than changes relative to the empty overlay.

## Corpus schema

`base_structures.parquet` stores the complete base BKG4 controller structure once per base identity, including canonical controller JSON, BKG4/feature-registry versions and support-source provenance.

`parents.parquet` stores each evaluated parent/child overlay: `parent_id`, `base_id`, overlay semantic ID, canonical overlay genome JSON, prefix step and rule count. This keeps the full parent representation explicit instead of baking current400 tensors into the dataset.

`cases.parquet` stores one representative legal case/context row. Important columns include `case_id`, `date`, `seat`, exact `baseline_margin`, `model_context_json`, `context_probe_steps`, `public_context_width`, `public_context_registry_width`, and flattened `public_context_values`. `prov_*` columns hold replay/team/seed bookkeeping only and are never model features.

`edits.parquet` stores canonical semantic edit definitions: `edit_id`, parent/child parent IDs and overlay IDs, operation family, explicit target node/kind/rule/condition location, complete `action_json`, and canonical child genome JSON. This supports all currently typed operation families and future subtree/specialist expansion without rebuilding CUDA labels into a fixed tensor schema.

`edit_case_labels.parquet` stores exact parent-relative supervision per edit × case: baseline/parent margin, candidate/child margin, signed delta, outcome transition, L→W, W→L, protected flag, severe-regression inputs, parent/child activation/effect counts, action-trace hashes, and behavior-changing/no-op information.

`edit_targets.parquet` is recomputed globally from all accumulated case labels. It contains the lexicographic aggregate targets: losses after, new losses, fragile wins after, worst candidate margin, mean margin delta, L→W/W→L counts, severe regression count, worst protected-win delta, strict-safe, exact-no-op, and behavior-changing.

`shards/cases_DATE.parquet`, `shards/edits_DATE.parquet`, and `shards/labels_DATE.parquet` are append-only per-source-day transaction files. `processed_shards.json` is written only after all three are complete. `corpus_manifest.json` records source archive identity, processed dates, parent IDs, CUDA-engine identity, generator/grammar/schema versions, operation-family counts, label counts, semantic dedup counts and file integrity metadata.

## Leakage boundary

Team names, opponent names, replay/episode IDs, submission IDs and seeds must only live under provenance fields such as `prov_team_name`, `prov_opponent_name`, `prov_episode_id`, and `prov_seed`. `model_context_json` is recursively audited and rejects those keys. `public_context_values` comes only from `B4ISR01` legal/public capture. The hidden-truth capture produced by the diagnostic CUDA tool is deliberately not loaded by the corpus builder.

## Build and publish the support dataset

The support bundle is explicit-allowlist only and is currently about a few MiB.

```bash
python3 tools/edit_corpus/build_support_dataset.py \
  --output scratch/kaggriculture_edit_corpus_support \
  --force
```

Create the private support dataset once:

```bash
kaggle datasets create \
  -p scratch/kaggriculture_edit_corpus_support \
  -q -t -r skip
```

Whenever CUDA/search/edit-corpus code or current parent assets change, rebuild the bundle and publish a new version of the **same** dataset:

```bash
python3 tools/edit_corpus/build_support_dataset.py \
  --output scratch/kaggriculture_edit_corpus_support \
  --force

kaggle datasets version \
  -p scratch/kaggriculture_edit_corpus_support \
  -m "refresh exact CUDA/EditCorpus support" \
  -q -t -r skip
```

The packager writes `support_manifest.json` with exact included paths, file hashes, source commit, current parent identity, BKG4 ABI, feature registry, grammar/generator version and CUDA result ABI.

## Create the persistent corpus dataset once

The notebook needs the persistent corpus dataset attached even before the first populated run, so create the tiny seed version once:

```bash
kaggle datasets create \
  -p tools/edit_corpus/corpus_dataset_seed \
  -q -t -r skip
```

This creates:

```text
ashok205/kaggriculture-bonsai-edit-corpus
```

The first successful GPU builder run publishes the first populated version; later runs append new replay dates to the same dataset. Existing corpus data is validated before it is copied into the new staging directory.

## Generate and validate the notebook

```bash
python3 tools/edit_corpus/build_notebook.py
python3 tools/edit_corpus/build_notebook.py --check

python3 -m unittest discover \
  -s tools/edit_corpus \
  -p 'test_*.py' \
  -v
```

Generated notebook:

```text
notebooks/kaggriculture_edit_corpus_builder/kaggriculture_edit_corpus_builder.ipynb
```

Kaggle CLI push:

```bash
kaggle kernels push \
  -p notebooks/kaggriculture_edit_corpus_builder \
  -t 43200
```

The generated kernel metadata enables GPU + internet and attaches exactly:

```text
ashok205/kaggriculture-top10-replay-archive
ashok205/kaggriculture-edit-corpus-support
ashok205/kaggriculture-bonsai-edit-corpus
```

## Kaggle smoke mode

For the first remote validation, edit only the notebook configuration cell:

```python
SMOKE_ONLY = True
PUBLISH = False
```

That bounds the run to one replay date, four replay rows, at most eight representative cases, two seed parents and a small candidate set. It still compiles/runs the real CUDA evaluator and information-state capture. Confirm the final manifest validates, then restore:

```python
SMOKE_ONLY = False
PUBLISH = True
```

for the historical bootstrap.

## Incremental behavior

On every run the notebook validates/copies the latest attached corpus, reads `processed_days.json` from the replay archive, compares source replay-shard identities to `processed_shards.json`, and processes only unseen dates. A previously completed date whose source shard identity changes is rejected instead of silently overwritten. A run with no pending dates does not compile CUDA and does not publish a new corpus dataset version.

Each pending day is a transaction: replay streaming → exact baseline CUDA → case selection → public-state capture → edit generation → exact parent/child CUDA labels → three date shards → processed marker. Failure before the processed marker leaves the prior published corpus untouched. Publication occurs only after consolidation and `validate_corpus()` succeed.

## Local bounded CUDA parity smoke

The development validation uses the same current BKG4 cases as the existing fragment evaluator. The empty overlay from `bkg4_edit_corpus_eval.cu` must match the existing current-BKG4 CUDA evaluator margin-for-margin and report zero overlay changes/activations. Non-empty typed overlays must finish with zero evaluator fault flags. This is intentionally tiny; large campaigns belong on Kaggle CUDA.
