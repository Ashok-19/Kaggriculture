# RLRMB

## Purpose

Regret-Localized Reactive Macro Beam Search (RLRMB) is the new authoritative
capability candidate-generation system under construction. It replaces the old
candidate manifold:

```text
donor x start x duration
```

with richer structured capability generation. Phase 0 establishes contracts only;
it does not evaluate or improve candidate quality.

## Frozen baseline

Hardened P0.7 Stage-A run:

```text
run_id = 536dd7c9f4f5b393c99ccf3f
CUDA games = 265,226
refined attempts = 88
accepted = 0
MIXED_LEGAL_STATE = 49
NON_TRANSFERABLE = 38
TARGET_CAPABILITY_LOST = 1
```

This frozen result is the future equal-budget comparison baseline. Phase 0 makes
no claim to improve it.

## Responsibility boundary

RLRMB owns capability candidate generation. It is not a second CEGIS
implementation.

The existing hardened system continues to own outer initiation CEGIS,
target-capability preservation, transfer validation, sealed holdout, threshold
robustness, automatic composition, broad certification, and CPU/CUDA parity.

## Current status

Phase 0 - isolated package contract and baseline freeze: complete after focused
validation. Phases 1 and later are not implemented.

## Generated outputs

Permanent source belongs under `tools/bkg4_search/rlrmb/`. Future generated runs
belong under `scratch/rlrmb/`.
