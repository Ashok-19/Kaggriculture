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
champion. New specialist-search work must follow the phased gates in
`docs/ASPE_MASTER_PLAN.md`; do not jump directly to the learned proposer.

The Phase-0 ASPE comparison context is frozen at `workspace/aspe/baseline/`.
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
