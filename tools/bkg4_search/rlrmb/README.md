# RLRMB

## Purpose

Regret-Localized Reactive Macro Beam Search (RLRMB) is the new authoritative
capability candidate-generation system under construction. It replaces the old
candidate manifold:

```text
donor x start x duration
```

with richer structured capability generation. Phases 0 and 1 establish contracts
only; they do not evaluate or improve candidate quality.

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

## Canonical identity contracts

Structural candidate IDs hash canonical semantic content, not donor route IDs,
filenames, artifact paths, or attempt numbers. Fragment-content IDs hash only the
ordered normalized actions in the selected window; candidate IDs separately retain
semantically relevant timing such as start and duration. Market-order order and hand
order remain significant.

Effect signatures identify exact ordered, role-separated panel response vectors.
They include baseline margins, per-case margin deltas, sign and L-to-W/W-to-L
patterns, and per-case action-change counts when available. Donor and attempt
provenance is excluded.

Read-only analysis of the hardened Stage-A run found 52 unique target effect
signatures and 54 unique target-plus-transfer signatures among 88 attempts. These
strict signatures include exact panel identity and action-change vectors. Omitting
action changes produces fewer classes, but would discard a required observed effect
dimension. Thus target effects contain 36 duplicates (40.9% of attempts), while
combined target-plus-transfer effects contain 34 duplicates (38.6%).

The same 88 attempts reduce to 50 structural candidates when identified by
canonical fragment content plus start and duration, leaving 38 duplicate structural
attempts.

The common 64-route donor bank contained the following unique local fragments:

```text
start 24, duration 24: 17
start 48, duration 24: 17
start 60, duration 12: 15
start 84, duration 12: 20
start 120, duration 18: 30
start 156, duration 48: 44
start 264, duration 24: 48
```

This confirms that nominal donor count substantially overstates local behavioral
diversity, especially in early windows. No search behavior changed in Phase 1.

## Current status

- Phase 0 - isolated package contract and baseline freeze: validated.
- Phase 1 - structural and effect canonicalization: validated.
- Phases 2 and later: not implemented.

## Generated outputs

Permanent source belongs under `tools/bkg4_search/rlrmb/`. Future generated runs
belong under `scratch/rlrmb/`.
