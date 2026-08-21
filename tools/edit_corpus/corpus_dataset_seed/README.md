# Kaggriculture BonsaiEditFormer CUDA Edit Corpus seed

This directory exists only to create the persistent private Kaggle dataset once.
The actual corpus is produced and versioned by the GPU notebook.

Create once:

```bash
kaggle datasets create -p tools/edit_corpus/corpus_dataset_seed -q -t -r skip
```

After creation, attach `ashok205/kaggriculture-bonsai-edit-corpus` as an input to the corpus-builder notebook. The first successful builder run publishes the first populated dataset version; later runs append new replay dates to the same dataset.
