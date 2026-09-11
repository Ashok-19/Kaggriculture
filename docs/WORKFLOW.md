# Workflow

## Development

Install the editable package, make scoped changes inside the matching package, and run the nearest unit/integration/parity tests. Never restore archaeological experiment code merely to satisfy a test. Replace stale fixtures with current or synthetic evidence.

## LLM lane candidate workflow

All new strategic discovery follows `docs/FINAL_LLM_LANE_EVOLUTION_PLAN.md`.

Each agent task must:

1. verify HEAD/status and lane authority;
2. inspect the actual parent/source/engine needed for the lane;
3. read compact lane dead-end/counterexample evidence;
4. state one primary causal hypothesis;
5. implement an executable candidate rather than prose advice;
6. keep runtime information legal and deterministic;
7. run focused static/semantic checks;
8. leave large-scale evaluation to CUDA;
9. emit compact candidate metadata and evidence;
10. stop at the assigned proposal/revision budget.

The first authorized architecture step is the bounded **8-agent manual lane pilot**. Do not build a large autonomous LLM orchestration service before that pilot demonstrates useful proposal signal.

## Bounded CUDA funnel

Use a staged funnel rather than immediately evaluating every LLM proposal on thousands of games:

```text
compile / legality / focused semantic tests
-> 32-64 fresh BUILD games
-> 128-256 fresh CUDA confirmation if useful
-> up to 512-game multi-regime confirmation for survivors
-> freeze
-> INTERNAL_DEV
```

CUDA is mandatory for bulk evaluation. CPU/C++ is for reference checks and small parity work. Record all game/turn work and semantic mismatches.

## Evidence roles

BUILD is fresh internal self-play/adversarial evidence by default. Public/historical source may be exposed to a generation agent only when explicitly tagged `TRAINING_SOURCE`; it then loses independent holdout status for that epoch.

INTERNAL_DEV is independent from BUILD. TRANSFER/VALIDATION/CEGIS/SEALED remain downstream firewalls. Any panel whose result is used to design a later candidate is retired as unseen evidence.

Do not use leaderboard score as an evolutionary reward.

## Champion update

Only the complete composed BKG4 policy may replace `assets/champion/`, after routing/composition, independent transfer/validation, protected safety gates, CPU/CUDA parity, and the official 64/128-game Python canary. Individual lane specialists are never promoted directly.

## Repository hygiene

`workspace/` is tracked compact evidence. Keep accepted candidate summaries small; do not commit giant logs or transient binaries. Commit or remove every non-ignored artifact before finishing. Never create `scratch/`.
