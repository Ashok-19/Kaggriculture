# Workflow

## Development

Install the editable package, make scoped changes inside the matching package,
and run the nearest unit and integration tests. Never restore an old experiment
directory to satisfy a test; replace archaeological fixtures with fresh or
synthetic current fixtures.

## Bounded CUDA experiment

Build with `scripts/build_cuda.py`. Generate fresh self-play into `workspace/`,
assemble a multi-seed, both-seat, multi-member BUILD panel, and generate a
disjoint INTERNAL_DEV rotation. Validate the campaign configuration before
running. Keep experiments bounded and record CUDA games, protected W->L, and
semantic mismatches.

## Champion update

Only a candidate that completes routing, independent TRANSFER/VALIDATION,
CEGIS/SEALED safety, composition, and parity may replace `assets/champion/`.
Update all controller/route assets atomically and retain a compact hash manifest.
Never overwrite the champion from an incomplete campaign.

## Official canary

Official Python/Kaggle canaries are a later release step, not a BUILD input.
Package from the current runtime and champion only after local CPU/CUDA parity.
Do not submit or inspect public opponents during discovery.
