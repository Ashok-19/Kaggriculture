# BKG4 Hard-League Specialist Promotion — 2026-08-20

## Verdict

`day2_capital_specialist_v1` is accepted on top of frozen `animal_feed_deferral_v1`.

The BKG4 VM/ABI/registry remain frozen at v4 / ABI 4 / registry-v1. Feed deferral is unchanged. No opponent identity, replay ID, seed, hidden future, or opponent private state is used by the production controller.

## Capability

The first specialist subtree recognizes a public economic commitment at step 26:

- `OPP_MONEY <= 139`
- `OPP_UNITS == 2`

It then authorizes route 211 only at steps `26..33` and `36`. The route was minimized by an exhaustive `2^16 = 65,536` CUDA action-mask search to mask `0x01ff`.

Two later recovery subtrees are nested under the specialist-active latch:

1. `BAKERY -> SMOOTHIE_SHOP`, steps 264–287: route 22.
2. `BRUNCH_SPOT -> BRUNCH_SPOT` plus own weed at tile 41, steps 264–287: route 46.

The strategic route authorization is an ephemeral BKG4 blackboard route value. It never overwrites the frozen controller route slot.

## Shared-market check

The accepted gain is not based on raising a shared product price. A temporary fresh regression was decomposed and showed our money falling more than the opponent's; it was therefore diagnosed as a private production/execution problem, not a shared-market opportunity. The recovery subtrees repair that private trajectory. Any future market idea must likewise demonstrate asymmetric value rather than merely lifting a price both players can realize.

## Search evidence

- 270,010 full CUDA games in the semantic opening/capital grid over the original 10 missing-capability losses.
- Every one of those 10 losses had some new-action-sequence headroom, proving the prior U/W/M route vocabulary was incomplete.
- 1,703,936 CUDA games in the exhaustive 65,536-mask day-2 specialist search over 26 discovery cases.
- 16 masks flipped losses with zero negative discovery deltas; `0x01ff` was the minimal elite representative.
- Entry thresholds 120/130/139/145/160 produce identical outcomes on screen61, shifted-current140 and historical336, so the gate is not an isolated exact threshold.

## Final compiled controller

- semantic hash: `74a4ddd75d93ee46cdad3fc78dbe1caf408678f669dea117685998069e85f6af`
- combined identity: `150ec96d6ff59b33855b7191c5501c298491a5f9727f6441519ec1f4c0d29142`
- nodes: 300
- BB: 20
- NodeState: 0
- max depth: 11
- static max visits: 220
- live intents: 0

Hot path over 2,876 reachable ticks:

- visits p50/p95/p99/max: 116 / 139 / 144 / 147
- feature evaluations p50/p95/p99/max: 44 / 66 / 71 / 74
- depth p50/p95/p99/max: 7 / 7 / 7 / 11

## Validation

- screen61: `24W/37L -> 26W/35L`, 2 loss-to-win flips, zero margin regressions.
- historical336: `251W/85L -> 251W/85L`, all 336 margins unchanged.
- fresh24: 10 positive / 14 unchanged / 0 negative deltas.
- sealed24: `12W/12L -> 14W/10L`, 2 loss-to-win flips, 0 negative deltas.
- current Ryo bank: 30 games / 15 seeds / both seats; 30 positive / 0 negative deltas, mean +1,400.43, floor `-20,143 -> -17,341`.
- fresh shifted leaderboard bank: 35 newly downloaded episodes, 70 tapes, 20 families, 140 both-seat games; `92W/48L` remains `92W/48L`, but 10 positive / 130 unchanged / 0 negative deltas. All ten fresh Ryo evaluations improve.
- source-backed ReCurSiON: 8 games, 0 outcome regressions, minimum margin delta +280.
- BKG4 VM parity: 0 Python/C++ and 0 Python/CUDA mismatches over 2,876 records.
- route-value materializer parity: 0 mismatches in 9 route/step/unit-count cases across Python/C++/CUDA.
- official Python canary: `39W/25L`, improving the accepted feed-deferral reference `37W/27L` by two wins.

## Competitive snapshot used

Live leaderboard refresh before the shifted holdout had tetsuya, Ryo Hasegawa, Kawashigi, Arman Tuganbaev, Izzoudine Mohamed KANTA, Crop Dusta, ReCurSiON, Xiaowenhao404, Subramanya N and Thomas Tschinkel among the leading/current strategies. Public submission IDs were resolved through Kaggle's team-public-submissions API, and up to five recent completed episodes per target submission were selected while preferring diverse opponents.

## Next optimization direction

This module demonstrates the intended 65k-node usage model: state-routed specialist subtrees plus nested recovery subtrees. The remaining hard league is not one family: current losses include Ryo, Arman, tetsuya, Crop Dusta, Izzoudine, Subramanya and others. Continue with new route/action-sequence capability discovery per causal state regime rather than global route replacement.
