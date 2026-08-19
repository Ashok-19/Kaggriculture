# BKG4 `animal_feed_deferral_v1` Promotion Report

Date: 2026-08-19
Team: ReCurSiON
Frozen VM: `8381e027646ba6494cbe4dd06a810ea55ec83575`
Canonical controller: `e9e66dfc9985a2111dc7c1b332fb569a02194528dd6512ccf3f154bd28abbc49`
Combined identity: `8cd92cc600537f5f0157925cd0b7d0de203e3b0d8187dd5389e02af96c9faf43`

## Decision

**Accept the module into the BKG4 champion lineage. Do not treat this as the final 100% champion or a submission decision.** BASE_0 remains preserved in `assets/`; the accepted challenger is frozen under `champions/feed_deferral_v1/`.

## Causal mechanic

The authoritative animal lifecycle permits one unfed day before escape, while scheduled base production still occurs. The minimized policy therefore authorizes deferring a route-planned cow `FEED` only when `consecutive_unfed == 0` and `pending_care_bonus == 0`. The BKG4 tree computes that authorization from legal observation features. A narrow reconciler only changes the already-materialized `FEED` to `PASS` when authorized.

No ABI/schema/VM change was made. The challenger has 256 nodes, 18 BB slots, 0 NodeState slots, depth 11, static max-visits 179, and 0 live intents.

## Promotion evidence

- Historical CUDA league: 336 games, 21 families, 152 seeds, both seats. Wins `243→251`, losses `93→85`, zero outcome regressions. Delta distribution: 330 positive, 5 zero, 1 negative (`-566`) without an outcome flip.
- Sealed current-agent holdout: 24/24 positive deltas, mean `+450.83`, zero outcome regressions.
- Canonical BKG4 CUDA distillation: 0/336 mismatches against the searched mechanic, with exactly 1,852 FEED rewrites.
- Tri-engine controller parity: 2,876 reachable tick records, 0 Python/C++ and 0 Python/CUDA mismatches across status, fault and full BB state.
- Historical ReCurSiON source regression: 8 official games against archived `f120_04ad004` and `v32_d52526d`; every matched margin improved by `+280…+477`, zero outcome regressions.
- Evolvability: 8/10 one-step semantic neighbors promotion-safe, six lexicographically identical elites, `v=0.49152`, `Q=0.48231`, brittleness `0.00921`.
- Hot path: p50/p95/p99 node visits `101/124/129`, max 132; zero persistent node state and zero intent journal pressure.
- Required official Python canary: 64/64 games completed across 32 opponent tapes and both seats. Result `37W/27L`, mean absolute margin `+10,216`, median `+3,908`; checkpoint SHA-256 `abe109f7f4d4bdfa561d74e678613999fadc675955c5ad0c6302a98b1ac29d5e`. This canary was not used for search feedback.

## What the 64-game canary means

The canary confirms authoritative Python execution of the mature canonical challenger. It is not evidence of 100% absolute strength: 27 losses remain, including one ReCurSiON tape loss and multiple Ryo Hasegawa losses. Therefore this module is a robust major improvement, not completion of the competition objective.

The next strategic work should attack those remaining hard lanes through bounded BKG4 semantic modules and CUDA search, while continuing engine-wide exploit discovery. Do not patch canary identities or seeds; use causal public/private state features and sealed holdouts.

## Freeze boundaries

- BKG4 ABI remains 4.
- Frozen VM commit remains unchanged.
- BASE_0 assets remain intact for regression.
- No Kaggle push/submission is part of this promotion.
- Large search remains CUDA-only; official Python remains a small final canary.
