# Kaggriculture Edit Corpus Support Dataset

Seed/reference metadata for the private dataset `ashok205/kaggriculture-edit-corpus-support`.

Build the publishable allowlisted bundle:

```bash
python3 tools/edit_corpus/build_support_dataset.py \
  --output scratch/kaggriculture_edit_corpus_support \
  --force
```

Create the dataset once from the generated bundle:

```bash
kaggle datasets create \
  -p scratch/kaggriculture_edit_corpus_support \
  -q -t -r skip
```

Whenever the CUDA evaluator, BKG4 search/edit code, corpus runtime, or parent assets change, rebuild and version the same dataset:

```bash
python3 tools/edit_corpus/build_support_dataset.py \
  --output scratch/kaggriculture_edit_corpus_support \
  --force

kaggle datasets version \
  -p scratch/kaggriculture_edit_corpus_support \
  -m "refresh exact CUDA/EditCorpus support" \
  -q -t -r skip
```

Do not manually add unrelated repository files. The packager uses a fail-closed explicit allowlist and writes `support_manifest.json` with the exact runtime inventory/provenance.
