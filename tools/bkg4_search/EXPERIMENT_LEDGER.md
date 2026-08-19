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

## Optimization-augmentation oracle campaign

| Experiment | Scope | Result | Decision |
|---|---:|---|---|
| Leakage-free optimization screen | 61 cases: all 27 canary losses + 10 lowest canary wins + 12 historical holdout + 12 fresh holdout | Source episode/seat dedup fixed before final oracle rerun; both seats represented | **Use as oracle screen** |
| ORACLE-U utility arbitration | 100,833 exact CUDA games | 8,591 active jobs; 16 hindsight loss→win flips; game best-delta mean +9,331.59, median +5,474, p90 +21,196, max +47,360 | **Headroom real; production selector not established** |
| ORACLE-W worker/task-stream reassignment | 93,879 exact CUDA games | 59,556 active jobs; 17 hindsight loss→win flips; game best-delta mean +10,302.52, median +8,464, p90 +27,387, max +47,243 | **Largest hindsight ceiling; heterogeneous choices** |
| ORACLE-M macro option commitment | 18,666 exact CUDA games | 14,897 active jobs; 13 hindsight loss→win flips; game best-delta mean +5,584.46, median +4,023, p90 +14,081, max +17,758 | **Material but third-ranked ceiling** |
| U current-public ambiguity | 183 official-engine frames at steps 96/120/144 | HOLD_BUYS step120: 6 identical-public-state classes with mixed-sign outcomes, 29 cases, max within-class delta span 60,410 | **Reject hidden-future interpretation** |
| U compact public gate search | 1–2 semantic predicates, discovery-selected, untouched historical/fresh holdout | Only discovery-safe flip gate: step144 + second PIZZA + opp money≥175; 4 discovery activations / 2 flips, **0 independent holdout activations** | **Reject: no holdout support** |
| W compact public gate search | 1–2 semantic predicates over transition frames | Best discovery-safe gate: step264, sells≤1, opp weeds≥1 → SWAP_10_12/24; fresh holdout 3 activations, 2 negative, sum −8,232, min −4,490 | **Reject: holdout regression** |
| M compact public gate search | 1–2 semantic predicates over transition frames | Best discovery-safe gate: step144, second SMOOTHIE, opp weeds=0 → route23/72; fresh holdout 6/6 negative, sum −93,100, min −36,103 | **Reject: severe holdout regression** |
| Semantic neighborhoods around U/W/M near-misses | threshold/eligibility perturbations | U nearest thresholds introduce discovery regressions or no activation; W neighbors remain fresh-negative; M neighbors remain 6/6 fresh-negative | **Evolvability does not rescue candidates** |
| Fixed optimizer decisions | all U/W/M candidate×probe choices on 61-screen | Only regression-free fixed choices have zero loss→win flips and trivial gains | **Reject as competitively immaterial** |

**Terminal verdict:** the three optimization proposals expose substantial *clairvoyant* regret, but no tested legal/current-observation selector converts that ceiling into broad promotion-safe behavior. Do not ship a utility arbiter, assignment solver, or macro planner from this campaign. Return to causal BKG4 module / engine-mechanic discovery until a new observable separator exists.

## Temporal information-state campaign

| Experiment | Scope | Result | Decision |
|---|---:|---|---|
| Complete R0 capture | 61 cases × 22 probes × 3,828 exact values | Full registry-v1 + 18 BB + 15 legal compatibility/exploit + 2 runtime values; zero CUDA faults | **Previous coarse ambiguity test corrected** |
| Exact R0 ambiguity | all active U/W/M screen decisions | Mixed cells persist: U 93, W 261, M 112; max spans 153,574 / 153,574 / 33,338 | **H0 not established by completeness alone** |
| H0 SMOOTHIE/high-cash HOLD_BUYS | threshold sweep 250..570 | Screen fresh pair flips 2 losses, but 0/336 historical + 0/24 independent fresh activations; only 2 sealed positives | **Reject as too narrow** |
| R1 semantic history capture | 1/4/24/72 deltas + own-action counts + event-since; 368 values | Mixed cells reduce to U 85, W 256, M 100; legal history genuinely splits some exact-R0 cells | **Information exists, policy value unproven** |
| H1 strawberry-growth SWAP_6_9 | broad historical336/fresh24/sealed24 CUDA | H24 growth>=2: historical +76,164 aggregate but 25 protected regressions + 6 wins lost; fresh 2/2 negative, -39,194, two wins lost | **Reject H1 promotion** |
| Temporal metamorphic test | strongest H1 diagnostic rule | 367/367 unrelated temporal perturbations preserved decision; causal history/context changes changed it | **Causal specificity pass, outcome gate fail** |
| Perfect hidden-truth upper bound | R0+R1 + true opponent 12-item totals + 5 seeds | Adds **zero** mixed-cell separation: 85/256/100 unchanged | **Hidden resource state not missing separator** |
| H2 perfect-truth policy | true strawberry seeds==0 -> SWAP_6_9 | historical336: 22 protected regressions, 6 wins lost; independent fresh24 no support | **Reject H2 before observer implementation** |
| Ryo trajectory diagnosis | 20 remaining Ryo canary losses | Multiple U/W/M headroom clusters; 16 early-land/cash-drawdown + 4 late-land/cash-positive strawberry-burst; 5 have no U/W/M flip | **No identity routing; new capability needed** |
| U/W/M-unflippable diagnosis | 10 canary losses | 5 Ryo + 4 Arman + 1 ReCurSiON; selector state cannot fix absent action headroom | **Search new lane/mechanic/capability** |

**Terminal verdict:** **TEMPORAL INFORMATION-STATE HYPOTHESIS TERMINALLY REJECTED.** R1 legal history is measurably informative, but no causal temporal selector is broad/holdout-safe; perfect hidden truth adds no representation separation and remains unsafe as a policy gate. No temporal module, observer, regime controller, active probe, ABI change, or champion change is promoted.
