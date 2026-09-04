# Repository Layout

- `src/kaggriculture_bonsai/league/`: self-play, league, multi-regime BUILD,
  and INTERNAL_DEV.
- `src/kaggriculture_bonsai/search/`: RLRMB IR, evaluation, beam/QD, repair,
  campaign pipeline, checkpoints, and restart memory.
- `src/kaggriculture_bonsai/routing/`: diagnostics, binary routing, CEGIS,
  materialization, and downstream certification.
- `src/kaggriculture_bonsai/composition/`: BKG4 specialist synthesis and
  composition.
- `src/kaggriculture_bonsai/runtime/`: BKG3/BKG4 interpreters, route codec,
  compatibility overlays, and champion agent.
- `src/kaggriculture_bonsai/evaluation/`: C++ reference-oracle interface.
- `native/bkg4/`, `native/cuda/`, `native/cpp/`: minimal native dependency
  closure for BKG4 execution, CUDA evaluation, and CPU reference semantics.
- `assets/champion/`: the sole authoritative compiled champion and manifest.
- `assets/baseline/`: two internal route programs used as bounded motif seeds.
- `configs/`: current campaign template and donor manifest.
- `scripts/`: native and champion-asset build entrypoints.
- `tests/unit`, `tests/integration`, `tests/parity`: active behavioral signal.
- `workspace/`: ignored generated data, builds, banks, and runs.
- `external/`: ignored, user-provided HOLDOUT panels.
