# Workflow

## Development

Install the editable package, make scoped changes inside the matching package,
and run the nearest unit and integration tests. Never restore an old experiment
directory to satisfy a test; replace archaeological fixtures with fresh or
synthetic current fixtures.

## Bounded CUDA experiment

Build with `scripts/build_cuda.py`. Generate fresh self-play into `workspace/`,
assemble a multi-seed, both-seat, multi-member BUILD panel, and generate a
disjoint INTERNAL_DEV rotation. `workspace/` is tracked: keep compact reproducible
evidence, and commit or delete every non-ignored artifact before finishing. Never
create `scratch/`, and do not place evidence that must be tracked inside a path
component named `build/`. Validate the campaign configuration before running.
Keep experiments bounded and record CUDA games, protected W->L, and semantic
mismatches.

For an equal-context comparison of donor-window RLRMB and synthesized local
Bonsai rules, build the native binaries and run:

```bash
.venv/bin/python scripts/run_oracle_bakeoff.py \
  --output workspace/oracle_bakeoff --discovery-games 9600
```

This freezes one fresh-self-play context and gives both methods the same
discovery budget. `--reuse-context` repeats search while verifying every panel
hash. The bakeoff writes evidence under `workspace/` and never updates the
champion. RLRMB and Direct-Bonsai are now frozen diagnostic baselines, not the
primary discovery path.

All new discovery work must follow `docs/FIRST_PLACE_EXECUTION_PLAN.md`. The
immediate authorized task is Phase 1: exact controller-state continuation/fork
fidelity plus the bounded fork-and-repair value experiment. Do not implement CCPR,
DSQD, later challenge-frontier machinery, or large campaigns before their gates.

The frozen historical comparison context is at `workspace/aspe/baseline/`.
Verify its complete file hashes and deterministic manifest regeneration with:

```bash
.venv/bin/python scripts/freeze_aspe_baseline.py --verify
```

## Champion update

Only a candidate that completes routing, independent TRANSFER/VALIDATION,
CEGIS/SEALED safety, composition, and parity may replace `assets/champion/`.
Update all controller/route assets atomically and retain a compact hash manifest.
Never overwrite the champion from an incomplete campaign.

## Official canary

Official Python/Kaggle canaries are a later release step, not a BUILD input.
Package from the current runtime and champion only after local CPU/CUDA parity.
Do not submit or inspect public opponents during discovery.
