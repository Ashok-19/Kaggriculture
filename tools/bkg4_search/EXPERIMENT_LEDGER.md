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

## 2026-08-20 — day2_capital_specialist_v1 PROMOTED

- Parent: frozen `animal_feed_deferral_v1` (`e9e66dfc...abbc49`).
- New semantic hash: `74a4ddd75d93ee46cdad3fc78dbe1caf408678f669dea117685998069e85f6af`.
- Combined identity: `150ec96d6ff59b33855b7191c5501c298491a5f9727f6441519ec1f4c0d29142`.
- 300 nodes / 20 BB / 0 NodeState; BKG4 VM/ABI/registry frozen.
- Capability: step-26 public low-cash/single-hire specialist route, minimized to mask `0x01ff`, with nested BAKERY→SMOOTHIE and BRUNCH+weed recovery subtrees.
- Search scale: 270,010 semantic opening games + 1,703,936 exhaustive day-2 mask games, plus recovery grids/broad validation.
- screen61: 24W/37L -> 26W/35L, 0 negative deltas.
- historical336: all terminal margins unchanged, 0 regressions.
- fresh24: 10 positive / 14 zero / 0 negative.
- sealed24: 12W/12L -> 14W/10L, 0 negative deltas.
- current Ryo30: 30/30 positive deltas across 15 seeds/both seats.
- fresh shifted-current140: 35 episodes / 70 tapes / 20 families / 140 games, 10 positive / 130 unchanged / 0 negative.
- source-backed ReCurSiON: 0 regressions.
- VM tri-engine: 0 mismatches / 2,876 records. Route-value materializer: 0 mismatches / 9 cases.
- official Python64: 39W/25L vs accepted parent reference 37W/27L.
- Shared-market audit: accepted gain is private execution/production; no rule is credited merely for lifting a shared market price.
- Machine evidence: `tools/bkg4_search/DAY2_CAPITAL_SPECIALIST_TERMINAL.json`.


## 2026-08-20 — ACCEPT hire4_capital_cycle_v1

Parent: `day2_capital_specialist_v1` (300 nodes). Candidate: `hire4_capital_cycle_v1` (315 nodes, BB21, state0).

Causal gate: at step 26, `OPP_HIRES_TODAY == 4 && OPP_MONEY >= 30` latches route 212 through step 47. The gate is legal/public and strategy-agnostic.

Shared-market check: the route contains a later `SELL FERTILIZER` + `BUY_PRODUCT WHEAT` sequence. Ablation removing that sequence destroys the capability; it finances the private animal/fertilizer production cycle. No credit is assigned to symmetric shared-price movement.

Promotion evidence:
- shifted current140: 92W/48L -> 94W/46L; 2 L->W, 0 W->L, 8 positive deltas, 0 negative deltas, +8,199 aggregate margin;
- historical336, screen61, fresh24, sealed24: all margins unchanged;
- source-backed ReCurSiON gate: 4 bounded games, 0 outcome regressions, min delta +347;
- controller parity: 2,876 ticks, 0 Python/C++, 0 Python/CUDA mismatches;
- route materialization parity including route 212: 12 cases, 0 mismatches;
- hot path: p50/p95/p99/max visits 123/146/151/154, static max 234, state0, live intents0;
- official Python canary64: 39W/25L, unchanged from the accepted 300-node parent.

Machine evidence: `tools/bkg4_search/HIRE4_CAPITAL_CYCLE_TERMINAL.json`.

The broader fragment remains research-only where exact public-state collisions mix positive and negative futures. Continue from the new 46-loss shifted-current frontier with additional specialized subtrees.


## 2026-08-20 — ACCEPT specialist stack routes 213 + 214

Parent `hire4_capital_cycle_v1` (315 nodes, 94W/46L shifted-current) was extended by two legal-state specialists.

- Route 213 `fert_wool_expansion_v1`: step264 gate `FERT_INV>=10105 && WOOL_INV<=9940`, 12/12 positive supported activations, 2 current L->W, zero negative deltas. Shared-market ablation: removing private HIRE/BUY_LAND expansion makes all supported cases worse.
- Route 214 `w33_s172_expansion_v1`: step192 gate `WHEAT_PRICE==33 && STRAWBERRY_PRICE==172`, 70/70 positive activations across 29 episodes / 22 families, 2 current L->W, zero negative deltas. Adjacent wheat-price regimes are mixed. Removing HIRE makes all 70 supported activations worse; removing SELL financing makes 68/70 worse.

Composed controller: 345 nodes, BB23, state0, semantic hash `8c5d7cd0e32574cbe349d5964946a9e2d6e7a75d766708f5d82308cd8507a034`, identity `6359ab400f9d128945cdbba5df7178a33b88c643ddd9b488299d66e14daa6925`.

Validation: current140 98W/42L (4 net flips vs 315 parent), historical336/screen61/fresh24/sealed24 no outcome regressions, controller parity 0 mismatches/2876 ticks, route materializer 0/18 mismatches, source-backed ReCurSiON 0/4 regressions, official Python canary64 39W/25L.

Machine evidence: `tools/bkg4_search/SPECIALIST_STACK_345_TERMINAL.json`.


## 2026-08-25 — P0.1 regret-guided counterfactual CUDA macro search

| Experiment | Scope | Result | Decision |
|---|---:|---|---|
| Current-parent hard-loss recheck | 10 terminal U/W/M-unflippable cases vs 345-node `w33_s172_expansion_v1` | All 10 remain losses; Ryo margins improve only +94…+98 vs old references | **Hard set still authoritative** |
| Coarse coherent donor-fragment search | 38 source-backed top-10 routes × 17 checkpoints × 24/48/72 horizons; live BKG4 rollout | 19,390 exact CUDA games, bad=0; 2 isolated Ryo L→W; early Izzoudine basin improves 9/10 but no Arman flip | **Deepen only evidence-backed basins** |
| Refined/hierarchical search | finer starts/horizons, public hybrids, two-stage route continuations | Large deficit reductions but no additional legal post-reveal flips | **Vocabulary/checkpoint limits matter** |
| Early-checkpoint all-route search | 64 coherent routes × steps 0/24/48/72 × horizons 24…192 | Arman cases 18/19 flip strongly at step48; later rejected after exact-state analysis | **Hindsight signal only** |
| Step48 complete legal-state capture | 61 cases × full registry/BB/compat/runtime | Support cells are mixed-sign under identical complete legal state | **Reject future-dependent early macro** |
| Post-reveal route435 transfer | step156 × 48, full 61-case screen | `PIZZA→PIZZA` 2/2 positive and 2/2 L→W; `PET→PIZZA` 2/2 positive with 1 L→W; other supported cohorts positive; ungated global use unsafe | **Capability discovery validated; gating deferred** |
| Route435 component ablation | 8 supported cases | full macro 8/8 positive, 3 L→W, 0 harmful; market-only 5/8 positive/3 harmful; units-only 4/8 positive/4 harmful | **Co-adapted multi-stage macro, not isolated edit** |

P0.1 terminal verdict: **VALIDATED.** Deeper CUDA counterfactual planning found a recurring post-reveal production/worker/market continuation that flips an authoritative U/W/M-unflippable hard loss and transfers across independent legal-state cohorts. The campaign also found a search-control failure: existing route79 can flip one hard case when searched at a longer/earlier window than the old ORACLE-M grid. The next stage should add a bounded macro/subtree representation plus initiation-set CEGIS; do not resume corpus/EditFormer generation for this result.

Permanent evidence: `tools/bkg4_search/P0_1_REGRET_GUIDED_MACRO_SEARCH.md`.


## 2026-08-25 — P0.2 bounded-macro initiation-set CEGIS

| Experiment | Scope | Result | Decision |
|---|---:|---|---|
| Bounded macro representation | route435 source-backed fragment, step156 ×48, ordinary BKG4 latch + route override | Compiles as 363 nodes / 24 BB / 0 state; route216 active only steps156–203; no ABI/VM change | **Representation accepted** |
| Initiation CEGIS iteration 0 | `SHOP_AT[1] == PIZZA_SHOP` | Unsafe: historical336 50 harmful activations / 32 W→L; additional current/fresh harms | **Reject broad demand-only gate** |
| Initiation CEGIS iteration 1 | add `OPP_UNITS >= 7 && OPP_LAND >= 2` | Removes expansion-mismatch harms, but 4 harmful screen61 states remain | **Refine on causal shared-market regime** |
| Initiation CEGIS iteration 2 | add `MARKET_INVENTORY[STRAWBERRY] >= 9990` | Core panels: 14 activations, 14 positive, 7 L→W, 0 harmful, 0 W→L; no selected mixed-sign identical legal-state cells | **Gate accepted for independent validation** |
| Frozen-gate fresh949 holdout | 80 previously unseen cases | 2 activations on both seats of a new episode: -1,824→+6,134 and -1,383→+6,043; 2/2 L→W, 0 harmful | **Independent positive transfer pass** |
| Additional top-10 holdouts | validation40 + sealed40 | 0 activations, therefore exact parent behavior and zero regressions | **Safe exclusion evidence; no positive support claimed** |
| Full composed CUDA certification | 745 panel records | 729/729 inactive cases exactly parent; 16/16 active cases exactly standalone route435; 0 mismatches/faults, 16 positive, 9 L→W, 0 harmful/W→L, worst delta +4,977 | **End-to-end composition pass** |
| Deduplicated support audit | 700 unique `(tape, seat, seed)` records | 10 unique activations across 5 source trajectories, all both-seat positive; 10/10 positive, 5 L→W, 0 harmful/W→L; 8 unique ReCurSiON self-history records untouched | **Promotion-safety evidence pass** |
| BKG4 CPU/CUDA bounded-decision parity | 2 legal-state trajectories × steps0–216 | 434/434 decision records exact; active case writes route216 at 156–203 and returns to parent at 204 | **Parity pass** |
| Focused regression suite | P0.2 + P0.1 parser + contrastive-goal tests | 24/24 passed after provenance hardening | **Implementation pass** |

Final public-state gate: `STEP == 156 && SHOP_AT[1] == PIZZA_SHOP && OPP_UNITS >= 7 && OPP_LAND >= 2 && MARKET_INVENTORY[STRAWBERRY] >= 9990`.

Accepted research lineage: `route435_pizza_glut_rebalance_v1`, semantic hash `f47ee7034e553e9a835486b283f4f819d857afdc95143a29d84cb07753b7c286`, combined identity `e2a2a0e91035c05c3183e691efee19441b6cee0d86506cbeb0e68a55768e53ce`. `default_runtime` remains `BKG4_BASE_0`; no submission performed.

P0.2 terminal verdict: **VALIDATED AND ACCEPTED INTO THE BKG4 RESEARCH LINEAGE.** Initiation-set CEGIS converted the P0.1 planning discovery into a deterministic, identity-free, bounded public-state specialist with independent both-seat L→W transfer and exact inactive-parent preservation. Continue with the next regret-search → bounded-macro → initiation-CEGIS cycle; do not infer broad coverage from the intentionally narrow activation support.

Permanent evidence: `tools/bkg4_search/P0_2_MACRO_INITIATION_CEGIS.md`.


## 2026-08-25 — P0.3 synthesizer repeatability: route410 wheat/liquidity expansion

| Experiment | Scope | Result | Decision |
|---|---:|---|---|
| Current P0.2 frontier refresh | hard10 + screen61 + shifted-current140 against 363-node route435 parent | hard10 9 losses; screen61 32 losses; current140 41 losses; severe new paired losses include Crop Dusta -35,110/-35,110 and Subramanya -30,048/-30,048 | **Target current loss frontier, not old static list** |
| Coarse regret-guided macro search | 64 coherent routes × 8 checkpoints × 4 horizons on 20 most severe current losses | 40,980 exact CUDA games, ~5,033 games/s, bad=0; route410@120x24 and route434@168x72 each expose repeated both-seat L→W basins | **Refine route410 first: cleaner composition and gate signal** |
| Route410 window refinement | starts108..132, horizons12..48 on current140 | step120×18 and ×24 both 6/6 positive / 2 L→W on support; ×18 has better worst delta; starts114/126 introduce support harms | **Minimize to step120×18** |
| CEGIS iteration 0 | `WHEAT inventory <= 9934` | core585: 6/6 positive, 2 L→W, 0 harmful; only three support tapes | **Promising causal drawdown regime, expand cautiously** |
| Simple-gate independent audit | `WHEAT <= 9934..9936` on validation40 + sealed40 + fresh949 | validation40 adds 2 harmful (-7,731 each); sealed40 adds 2 harmful **W→L**; combined core/holdouts 12 positive / 4 harmful / 6 L→W / 2 W→L | **Reject as final gate despite simplicity** |
| CEGIS iteration 1 | `WHEAT <= 9937 && OPP_MONEY <= 120` | core: 8/8 positive, 2 L→W, 0 harmful; untouched validation40 activates one both-seat trajectory at -7,731 delta each | **Reject and add counterexample** |
| CEGIS iteration 2 | add `MILK inventory >= 9990` | core+counterexample: 6/6 positive, 2 L→W, 0 harmful, no mixed exact legal cells; milk threshold 9984..9995 and opp-money 115..150 form stable safe plateaus | **Freeze gate** |
| Frozen sealed40 | one unseen both-seat trajectory | 2/2 positive, 0 harmful, +15,334/+21,227 | **Independent positive transfer** |
| Frozen fresh949 | two unseen episodes, both seats | 4/4 positive and 4/4 L→W, 0 harmful; deltas +18,467..+28,764 | **Independent L→W transfer pass** |
| Two-synthesized-specialist composition | route217 window120..137 + existing route216 window156..203 | windows disjoint; fail-closed overlap guard added; 0 dual-gate activations in 745 certification records | **Safe manual composition proof** |
| Full composed CUDA certification | 745 protected/current/holdout records | 733/733 inactive exact parent; 12/12 active exact route410 teacher; 12 positive, 6 L→W, 0 harmful/W→L, worst +322, bad=0 | **End-to-end pass** |
| Deduplicated regression audit | 700 unique `(tape, seat, seed)` records | 12 activations / 6 tapes, all both-seat pairs; 12/12 positive, 6 L→W, 0 harmful/W→L; 495 protected P0.2 wins have zero negative delta; 8 self-history records and all 10 route435 support records unchanged | **Promotion-safety pass** |
| Targeted component ablation | 12 final support records | full 12/12 positive, 6 L→W; market-only 4 harmful / 2 W→L; worker/production-only 10 harmful / 0 L→W | **Coordinated private program required; shared-market-only explanation rejected** |
| Final rebuilt CUDA reproduction | rebuilt P0.2 parent + 380-node P0.3 challenger on cert745, strategic route slot 19 | 1,490 games per controller, bad=0; parent 0/745 and candidate 0/745 mismatches vs persisted final rows | **Compiled-asset reproduction pass** |
| CPU/CUDA two-macro parity | steps119..205 on compiled 380-node challenger | 87/87 decision records exact; route217 returns at138, route216 returns at204 | **Parity pass** |
| Closure tests | focused P0.2/P0.3 + adjacent P0.1/contrastive | 22/22 focused + 13/13 adjacent = 35/35 pass | **Implementation/regression pass** |

Final P0.3 gate: `STEP == 120 && MARKET_INVENTORY[WHEAT] <= 9937 && OPP_MONEY <= 120 && MARKET_INVENTORY[MILK] >= 9990`.

Accepted research lineage: `route410_wheat_liquidity_expansion_v1`, route217, steps120..137; 380 nodes / 25 BB / 0 state, semantic hash `b2edd5dd76b936b6ea469009eb75b587531cef2f1be62939254b924f2e245edc`, combined identity `494bdfb4aa785088c3ebc23ac1fa2b71db8f51a26f861bea6480a7c457322413`. `default_runtime` remains unchanged; no submission performed.

P0.3 terminal verdict: **VALIDATED.** The planning → bounded macro → public-state CEGIS → deterministic BKG4 pipeline has now produced two materially distinct independently transferring specialists: route435's midgame strawberry-glut production rebalance and route410's day-5 wheat-drawdown liquidity/labor expansion. Continue Phase C with at least one more distinct specialist before making automatic composition the main implementation stage, unless the next natural capability exposes an actual overlap/resource conflict first.


## 2026-08-25 — P0.4 third specialist and exact route-authority preemption

| Experiment | Scope | Result | Decision |
|---|---:|---|---|
| Current P0.3 frontier refresh | screen61 + current140 against 380-node route410 parent | screen61 29W/32L; current140 101W/39L; worst -49,506 | **Search actual remaining frontier** |
| Coarse coherent CUDA search | 64 donors × 8 starts × 4 horizons on target20 | 40,980 games, bad=0; route437@264x24 gives 10 positive / 4 L→W / +55,859 on target20 | **Refine route437** |
| Window refinement | starts240..288 × durations12..48 on current140 | 264x24 remains best by L→W; ungated 32 positive / 102 harmful | **Specialist gate required** |
| Initial MELON/YARN gate | core585 then broad archive766 | core 8/8 positive / 3 L→W; archive 7 activations with 6 harmful, worst -13,331 | **Reject overfit gate** |
| Final counterexample-guided gate | `WHEAT price>=38 && OPP_MONEY>=15750 && SHOP_AT[2]!=BRUNCH` | core 4/4 positive / 3 L→W; fresh 2/2 positive; archive 2/2 positive; validation/sealed excluded; no harms/mixed cells | **Freeze and synthesize** |
| Threshold audit | money one-dollar sweep, WHEAT price/inventory neighbors, shop counterexamples | money safe begins at15,729; selected15,750; price37 mixed, price38 safe; admitting third Brunch adds 4 harms | **Boundaries explicit** |
| Causal CUDA ablation | 8 accepted support + 4 Brunch counterexamples | smaller MELON sale exactly unchanged; COW reversion makes 5/8 support harmful and aggregate -1,228; base WHEAT/fert timing remains 8/8 positive | **Private sheep allocation is causal** |
| Exact composition | route218 and route213 share steps264..287/slot19 | `ExactRoutePreemption` clears route213 latch only on route218 authorization; both windows end together | **Minimum precedence rule implemented** |
| Full compiled CUDA certification | cert745 + archive766, parent/challenger/teacher piecewise | 1,511 records: 1,503 inactive exact parent, 8 active exact teacher, 8 positive, 3 L→W, 0 harmful/W→L, worst +1,129, bad=0 | **End-to-end pass** |
| Protected preservation | 1,466 unique records | 969 parent wins with zero negative delta; self-history 0/8 activations; route217 0/12 changed, route216 0/16 changed, route213 0/51 changed | **Regression pass** |
| CPU/CUDA collision parity | steps119..289 including simultaneous route213/218 authorization | 171/171 exact; route217/216/218 boundaries correct; collision resolves to218; parent return at288 | **Parity pass** |
| Closure tests | focused P0.2/P0.3/P0.4 + adjacent P0.1/contrastive | 29/29 focused + 13/13 adjacent | **Implementation pass** |

Final P0.4 gate: `STEP == 264 && MARKET_PRICE[WHEAT] >= 38 && OPP_MONEY >= 15750 && SHOP_AT[2] != BRUNCH_SPOT`.

Accepted research lineage: `route437_wheat_scarcity_sheep_shift_v1`, route218, steps264..287; 398 nodes / 26 BB / 0 state, semantic hash `4e611d80f0bbb38bd696aa3dbb8a5e9e903be556e4ceafa233b5cb5685597be6`, combined identity `357dbb9489b26a8fb424de592da92e45e2c0ac28d8a450a5e6e1b5791ade9d29`. `default_runtime` remains unchanged; no submission performed.

P0.4 terminal verdict: **VALIDATED.** The third synthesized specialist independently transfers and exposes the first real multi-specialist authority conflict. Proceed to a narrow fail-closed composition stage that reproduces the proven route218-over-route213 rule; do not build a speculative universal scheduler.


## 2026-08-25 — P0.5 bounded automatic Bonsai composer

| Validation | Scope | Result | Decision |
|---|---:|---|---|
| Declarative composition | frozen accepted parent + route213/216/217/218 declarations + route218-over-route213 rule | deterministic module/report; one resolved window+authority+possible-gate conflict; zero unresolved conflicts; construction about 3 ms | **Reusable composer path accepted** |
| Structural oracle | automatic output vs manual P0.4 | BB, tree, routes and reconciler exact; controller and route-pack bytes exact; 398 nodes / 26 BB / semantic `4e611d80...`; combined identity unchanged | **Exact policy reproduction** |
| CUDA game equivalence | cert745 + archive766 | 1,511 records / 3,022 games; result files byte-identical; terminal, margin and activation mismatches 0; bad=0 | **Game-level pass** |
| Inactive-parent invariant | P0.4 certification universe | 1,503/1,503 inactive cases exact; eight active cases exact accepted teacher; protected wins unchanged | **Fallback invariant preserved** |
| CPU/CUDA collision | steps119..289, manual + automatic controllers | 342 decision rows; CPU/CUDA and manual/automatic mismatches 0; collision resolves route218 through287; parent at288 | **Authority semantics pass** |
| Fail-closed/determinism tests | composer + P0.2-P0.4 + adjacent search tests | 23/23 composer, 52/52 focused combined, 13/13 adjacent; reordered declarations and repeated CLI output identical | **Implementation pass** |

Composition semantic ID: `7d66b2f808437b24344d3c2132334540126c1fd34c84f3ed0e7abfe0650a823e`.

P0.5 terminal verdict: **VALIDATED.** Current composition is data rather than route-specific controller surgery. The bounded composer rejects undeclared authority overlap, bad ownership, invalid precedence and unsupported resume behavior, while producing the exact P0.4 executable policy. Proceed to P0.6 end-to-end synthesis orchestration; do not add a fourth specialist manually first.

Permanent evidence: `tools/bkg4_search/P0_5_AUTOMATIC_COMPOSER.md`.


## 2026-08-26 — P0.6 generic CEGIS foundation hardening

| Validation | Scope | Result | Decision |
|---|---:|---|---|
| Prior `run_cegis()` audit | fixed discovery/counterexample set + validation | validation harms only skipped pre-ranked gates; no feedback/resynthesis; no-harm branch could return despite validation harm | **Correctness defect confirmed** |
| Active counterexample TDD | synthetic exact legal-state fixtures | validation harm appended to evidence, resynthesis performed, safe convergence/mixed/no-separator/budget reasons explicit | **Active CEGIS pass** |
| Target-capability synthesis | required target positive + required target L->W fixtures | target rescue outranks unrelated transfer; zero required target support fails inside CEGIS as `TARGET_CAPABILITY_LOST` | **Original capability is a synthesis constraint** |
| Gated transfer contract | independent transfer fixtures | final gate must retain configured positive + independent support or reject `NON_TRANSFERABLE` | **Transfer cannot be discarded by target routing** |
| Frozen sealed holdout | explicit pure contract | sealed harm rejects `HOLDOUT_FAILURE`; sealed rows do not enter CEGIS evidence | **Leakage guard pass** |
| Exact ambiguity / identity | registry+BB+compat collision + identity tests | mixed exact legal state fails closed; identity-like features rejected; `OWN_SEED` legal exception preserved | **Legal-state pass** |
| Threshold trace | numeric-neighbor fixture | target support, support groups, W->L and first unsafe lower/upper boundaries recorded | **Auditability pass** |
| Candidate-50 real artifact | persisted Stage-A route212@24x24 audit | macro 6/6 positive, 4 L->W; old `OPP_MONEY<=22` activates 0/6 target states (`OPP_MONEY=170`) and remains `TARGET_CAPABILITY_LOST` | **Regression pass** |
| Historical gate evidence | all stored route435/410 exact CUDA effect/state panels | route435 14/14 positive, 7 L->W, 0 harm/W->L; route410 12/12 positive, 6 L->W, 0 harm/W->L; zero selected mixed cells | **Historical semantics preserved** |
| Bounded regression suite | synthesizer + P0.1 + P0.2/P0.3/P0.4 + P0.5 composer | **99/99 PASS**; available CPU/CUDA decision parity tests pass | **Foundation regression pass** |
| Repository cleanup | synthesis source tree only | removed self-ignored 4.3 GB EditFormer `.venv` + bytecode caches; no tracked historical/evidence file removed; scratch/data preserved | **Safe cleanup pass** |

Current authoritative path: `tools/bkg4_search/SYNTHESIZER_ARCHITECTURE.md`.
Permanent report: `tools/bkg4_search/P0_6_CEGIS_FOUNDATION_HARDENING.md`.

P0.6 CEGIS foundation verdict: **VALIDATED / READY for a corrected P0.7 Stage-A CUDA campaign.** No Stage B, large search, official Python, EditFormer/corpus work, PSRO, push or submission was performed.
