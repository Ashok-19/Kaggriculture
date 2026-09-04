# Dependency Audit

The audit began at adversarial league generation, self-play, multi-regime BUILD,
INTERNAL_DEV, RLRMB, binary routing, CEGIS, composition, CUDA evaluation, the C++
oracle, and champion loading. Python imports, subprocess targets, native includes,
runtime assets, configs, and tests were traced before deletion.

## Keep and move

- RLRMB core moved from `tools/bkg4_search/rlrmb` into the `league`, `search`,
  and `routing` packages.
- The composer, macro codec, information-state codec, goal synthesis, and CEGIS
  helpers form the retained composition closure.
- BKG3/BKG4 interpreters, the current champion runtime, legacy overlays, route
  pack reader/encoder, and engine exploit layer form the Python runtime closure.
- Native closure consists of BKG3/BKG4 headers/adapters, stateful CUDA helpers,
  CUDA simulator/tape/evaluator headers, two CUDA entrypoints, and the C++
  simulator/pybind oracle.
- The bounded automatic composer controller and its two route packs became the
  sole `assets/champion`; routes 64 and 410 are the retained internal motif seeds.

## Delete

PureJaxRL, Puffer/PPO, Puffertank, neural training/checkpoints, BonsaiEditFormer,
edit corpora, B4IM, meta-search, replay archives, public agent/notebook copies,
old submissions, old champion snapshots, historical plans/results/configs, and
obsolete engine benchmarks were outside the live closure and removed. The full
local Kaggle-environments checkout was replaced by pinned package version 1.32.7.

All old `scratch/` dependencies were either replaced by canonical assets,
fresh/synthetic tests, `workspace/`, or the explicit `external/holdout/`
interface. Git history remains the archive.
