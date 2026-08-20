# BonsaiEditFormer-5M-v1

Controlled JAX/Flax/Optax architecture experiment for direct BKG4 synthesis. The model ranks semantic `ModuleGenome` edits; it is never used at competition runtime.

The primary comparison reuses the frozen MinGRU corpora/splits and adds only deterministic, pre-counterfactual representation features: typed module-tree structure, explicit edit deltas, frozen-parent legal-state activation probes, and the existing heuristic as an optional learned residual input. Cached CUDA outcomes remain labels only.

Generated environments, activation caches, feature tensors, checkpoints, logs, and rankings live under `scratch/bkg4_editformer_jax/` and are ignored by git.
