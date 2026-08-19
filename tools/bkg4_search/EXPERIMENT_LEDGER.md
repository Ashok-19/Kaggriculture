# BKG4 Search Experiment Ledger — 2026-08-19

This ledger records the compact evidence needed to resume the BKG4 search campaign. Large raw results remain under ignored `scratch/bkg4_search/`; hashes are recorded in the champion ledger.

| Experiment | Scope | Result | Decision |
|---|---:|---|---|
| Full-controller FEED semantic grid | 1,344 CUDA neighbors | Strong basin found around one-day cow feed deferral | Continue + minimize |
| Cow-only, previous day fed, no pending care bonus | 336 historical games / 21 families / 152 seeds / both seats | 243→251 wins, 93→85 losses; 330 positive deltas, 5 ties, 1 non-outcome negative delta | **Accept** |
| GOOSE expansion | Semantic neighbor | Behaviorally redundant on current actuator manifold | Remove during minimization |
| SHEEP expansion | Semantic neighbor | Major regressions | **Reject** |
| FEED→HARVEST / FEED→COLLECT replacement actions | Grid neighbors | No outcome gain over simple FEED→PASS | **Reject as unnecessary complexity** |
| Wheat caps 4/8/16 | Full historical league | Weaker conversions and/or more negative deltas than uncapped rule | **Reject** |
| Wheat caps 32/64 | Full historical league | Behaviorally identical to uncapped candidate | Remove as redundant |
| Market cash-unlock/order exploitation | 1,728 CUDA cases after fixing unit-vs-currency prototype bug | Zero activation under current controller | **Reject for current manifold** |
| Canonical BKG4 distillation | 336 historical CUDA games | 0 mismatches against searched mechanic; 1,852 rewrites | **Accept** |
| Tri-engine BKG4 controller parity | 2,876 reachable ticks | 0 Python/C++ and 0 Python/CUDA mismatches | **Pass** |
| Source-backed ReCurSiON regression | 8 official games vs `f120_04ad004` and `v32_d52526d` | Every matched margin improved by +280…+477; 0 outcome regressions | **Pass** |
| Sealed current-agent holdout | 24 games | 24/24 positive deltas; mean +450.83; 0 outcome regressions | **Pass** |
| Evolvability neighborhood | 10 one-step semantic neighbors | 8/10 promotion-safe, 6 identical elites, σ(v−Q)=0.00921 | **Pass** |
| Required official Python canary | 64 games / 32 tapes / both seats | 37W/27L, no runtime failure; not used for retuning | **Execution gate pass; absolute-strength gap remains** |

## Accepted module

`animal_feed_deferral_v1` is accepted into the BKG4 strategic lineage. The BKG4 tree owns the legal causal authorization. The external reconciler is intentionally non-strategic: it only converts an already materialized route-pack `FEED` to `PASS` for an authorized unit.

The canonical policy is `e9e66dfc9985a2111dc7c1b332fb569a02194528dd6512ccf3f154bd28abbc49`, combined identity `8cd92cc600537f5f0157925cd0b7d0de203e3b0d8187dd5389e02af96c9faf43`.

## Engine-wide discovery backlog

These are hypotheses, not accepted exploits. Keep the engine-wide scope, but require count/headroom evidence and CUDA counterfactual search before adding any behavior: locked-tile movement/interactions, same-tile sequential multi-unit actions, atomic PLANT validation, unit-order-sensitive end-of-day inventory overflow, plant decay/harvest ordering, duplicate shop effects, low-price sales with weak shared-supply consequences, and other town/market ordering effects.

Do not turn any backlog item into a route/seed/opponent lookup. New behavior must remain causal, legal-observation-driven, bounded in the BKG4 grammar, and independently validated.
