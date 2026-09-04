# ReCurSiON Kaggriculture Bonsai

This repository develops one system: the self-play Bonsai/BKG4 agent used by
team ReCurSiON. Fresh self-play and an internal adversarial league discover
bounded specialists; independent screening and safety stages decide whether a
specialist may be composed into the current BKG4 controller.

The authoritative champion is `assets/champion/`. Generated tapes, binaries,
search state, and results belong in ignored `workspace/`. External opponent
panels are optional HOLDOUT inputs under ignored `external/`; they never define
BUILD policy.

## Setup

```bash
uv venv .venv
uv pip install --python .venv/bin/python -e '.[dev]'
```

Kaggriculture semantics are pinned to `kaggle-environments==1.32.7`. No local
framework checkout is required.

## Native builds

```bash
.venv/bin/python scripts/build_cpp_oracle.py
.venv/bin/python scripts/build_cuda.py
```

The scripts write only to `workspace/build/`. CUDA defaults to `sm_86`; pass
`--arch` for another supported device.

## Tiny self-play check

```bash
.venv/bin/python -c "from pathlib import Path; from kaggriculture_bonsai.league.selfplay_generator import SelfplayPolicy, generate_selfplay_tapes; print(generate_selfplay_tapes(Path('workspace/smoke/tapes'), seeds=(7,), policies=(SelfplayPolicy('mirror','champion'),), seats=(0,), rng_seed=7)[0])"
```

The campaign template is `configs/campaign.json`. First generate fresh BUILD
and INTERNAL_DEV panels in `workspace/panels/`, provide independent HOLDOUT
panels under `external/holdout/`, then validate or run through:

```bash
kaggriculture-bonsai --config configs/campaign.json --output workspace/runs/demo --validate-config
```

## Tests

```bash
.venv/bin/pytest -q tests/unit tests/integration
.venv/bin/pytest -q tests/parity
```

Build native binaries before parity tests. See [architecture](docs/ARCHITECTURE.md),
[layout](docs/REPO_LAYOUT.md), and [workflow](docs/WORKFLOW.md).
