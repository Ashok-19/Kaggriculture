# Repository Layout

## Active source and runtime

- `src/kaggriculture_bonsai/`: Python package, current agents, runtime integrations, and policy-generation code.
- `native/bkg4/`: Bonsai/BKG4 ABI, VM, adapters and compilation/runtime support.
- `native/cpp/`: C++ reference simulator and native bridges.
- `native/cuda/`: CUDA simulator/evaluators for large-scale exact search/evaluation.
- `assets/champion/`: sole authoritative released champion assets.
- `configs/`: current runtime/campaign configuration.
- `scripts/`: bounded build/evaluation/maintenance entry points.
- `tests/unit`, `tests/integration`, `tests/parity`: active behavioral signal.

## Evidence

- `workspace/`: tracked compact experiment evidence and reproducible run context. Disposable build/cache subpaths are allowed only where matched by `.gitignore`; every other workspace artifact must be committed or removed.
- `external/`: ignored user-provided external/HOLDOUT material.
- Never create `scratch/`.

## Current planning authority

- `docs/FINAL_LLM_LANE_EVOLUTION_PLAN.md`: authoritative discovery/execution roadmap.
- `docs/FINAL_POLICY_DISCOVERY_ARCHITECTURE.md`: retained last RL/learned-policy research alternative; not active authority.
- `docs/ARCHITECTURE.md`: compact current architecture summary.
- `docs/WORKFLOW.md`: active implementation/evaluation workflow.

Older deterministic/gold/service-tour planning documents have been retired from live authority; their historical contents remain recoverable from Git history and their experiment evidence remains under `workspace/`.
