# BonsaiEditFormer

Offline JAX/Flax/Optax infrastructure for synthesizing deterministic BKG4 specialists. Neural models are search-time only; accepted edits compile/materialize into ordinary BKG4 and no neural runtime is shipped to Kaggle.

## Proven critic

`BonsaiEditFormer-5M-v1` is the frozen structural edit critic/ranker:

- d=288, heads=6, FFN=960, tree/case/fusion blocks=3/1/1
- 4,991,412 parameters
- structural tree attention + typed edit encoding + legal-state activation context
- exact-CUDA regression, loss-to-win, margin, worst-win, ranking and per-case-delta objectives

The frozen current400 experiment found both strict-safe edits inside top-16 and is preserved as the critic initialization baseline.

Scaling presets remain available for later capacity experiments:

- `20m`: d=448, heads=7, FFN=1664, blocks=4/2/2, 19,742,698 params
- `50m`: d=704, heads=11, FFN=2560, blocks=5/2/2, 53,558,719 params

## Actor-Critic editor

`actor_model.py` extends the proven 5M network with a hierarchical typed editor while sharing the structural encoder and retaining the critic heads exactly. The 5M-ish feasibility model has 5,581,831 parameters.

The actor operates on the existing `ModuleGenome` grammar rather than bytes or arbitrary 65k-node trees. It supports bounded typed operations including specialist insertion, condition composition, threshold/route/window/priority edits, lane splits, recovery insertion, subtree replacement and redundant-branch removal. Target locations are selected through an operation-masked pointer head over encoded tree nodes.

Training is staged: import the frozen critic exactly, keep critic/shared parameters frozen, and warm-start only actor-specific heads from exact-CUDA historical parent→child mutations. CUDA remains the sole source of improvement truth.

## No-op-attractor protection

The first fresh generator comparison showed that scalar critic ranking could prefer edits that were safe only because they did nothing. `actor_generate.py` therefore uses a safety-constrained effect-aware shortlist:

1. frozen critic regression probability remains the primary safety gate;
2. legal-state probes measure candidate-vs-parent route divergence, loss-target coverage and winning-case exposure;
3. the already-trained per-case `case_delta` head provides a stronger predicted-effect magnitude;
4. safe + strongly effective + loss-targeted edits are exploited first;
5. a bounded exploration fraction may admit strongly effective but higher-risk edits;
6. weak/inert candidates are fallback only and their use is reported explicitly.

The selector never treats activation itself as reward. Exact CUDA evaluation still decides whether an edit is good. `cuda-campaign` rejects reward manifests without both seats across the configured minimum number of distinct seeds and now reports exact-parent/no-op counts directly.

Generator benchmarks persist the full scored actor/heuristic pools as well as the equal-budget CUDA lanes so effect false positives can be audited without regenerating proposals.

Generated environments, feature tensors, checkpoints, logs, rankings, comparison artifacts and cloud bundles belong under ignored artifact paths (`scratch/`, result ZIPs, etc.).
