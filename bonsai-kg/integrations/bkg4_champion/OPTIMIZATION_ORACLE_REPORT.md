# BKG4 Optimization-Augmentation Oracle Campaign — Terminal Report

Date: 2026-08-19
Team: ReCurSiON
Starting HEAD: `3bcdd40a9b67d477ad181163a539a7c90c614faf`
Frozen BKG4 VM: `8381e027646ba6494cbe4dd06a810ea55ec83575`
Accepted controller: `e9e66dfc9985a2111dc7c1b332fb569a02194528dd6512ccf3f154bd28abbc49`
Accepted combined identity: `8cd92cc600537f5f0157925cd0b7d0de203e3b0d8187dd5389e02af96c9faf43`

## 1–6. Frozen state, repository, and fresh league

1. The campaign started from a clean `main` at the exact HEAD above.
2. BKG4 ABI/version/runtime were not modified. No BKG5 was created.
3. `animal_feed_deferral_v1` was not modified. No accepted champion asset was overwritten.
4. The live leaderboard was refreshed before oracle work. The leading snapshot was: tetsuya 3142.0, Ryo Hasegawa 3103.9, カワシギ 3066.6, Arman Tuganbaev 3043.6, ReCurSiON 2948.5, Xiaowenhao404 2933.1, Crop Dusta 2926.3, Kobe BRYANT 2921.9, peikopon 2914.6, Galaxantic 2911.1, followed closely by Thomas Tschinkel and Efe Can Celiksoy. Existing current replay banks covered Ryo, tetsuya, Arman, カワシギ, Kobe, Xiaowenhao404, peikopon, Galaxantic and ReCurSiON plus historical distinct families. The installed Kaggle CLI did not expose the `competitions episodes` subcommand documented in `AGENTS.md`, so no new Crop Dusta replay was fabricated or claimed.
5. The final oracle screen contained 61 cases: all 27 official-canary losses, the 10 lowest-margin canary wins, 12 untouched historical-holdout cases and 12 leakage-free fresh-holdout cases, both seats where available. Fresh holdout deduplication uses source episode/seat as well as serialized SHA.
6. The authoritative engine source was re-audited around `market_price`, `_process_market`, `_apply_unit_action`, `_town_consume`, `_daily_refresh_plants`, `_daily_refresh_animals`, and `_end_of_day`. The oracle uses exact CUDA engine continuation; no generic logistic approximation or invented collision rule was introduced.

## 7. Causal clustering of the 27 canary losses

Exact loss→win oracle coverage clusters:

- `U+W`: 7 losses
- `U+M`: 2
- `W+M`: 2
- `U+W+M`: 2
- `U only`: 1
- `M only`: 3
- none of U/W/M: 10

Thus market/capital arbitration can flip 12 of the 27 losses in hindsight, worker/task reassignment 11, macro commitment 9, with substantial overlap. Ten losses survive every tested bounded optimization oracle. The absolute remaining canary family losses stay dominated by Ryo Hasegawa, with additional Arman, peikopon and one historical ReCurSiON-tape loss.

## 8–11. ORACLE-U — utility arbitration

Methodology: at fixed public decision probes, clone exact CUDA continuation and compare the canonical BKG4 action with bounded legal alternatives. Local candidates include HARVEST/WATER/FEED/CARE/COLLECT_FERTILIZER/FERTILIZE on the first 13 proven route units. Market candidates include HOLD_SELL, HOLD_HIRE, HOLD_LAND, HOLD_BUYS, HOLD_ALL_MARKET, SELLS_FIRST, BUYS_FIRST and REVERSE_MARKET. Legacy/exploit/feed authority remains above the candidate mutation.

- CUDA games: **100,833**
- active candidate jobs: **8,591**
- activation states: **1,159**
- positive-regret states: **368**
- state regret mean / median / p90 / max: **+1,256.71 / 0 / +2,459.2 / +47,360**
- per-game best delta mean / median / p90 / max: **+9,331.59 / +5,474 / +21,196 / +47,360**
- hindsight loss→win flips: **16** on the 61-case screen; **12** among the 27 canary losses
- fresh holdout hindsight flips: **2**; historical holdout hindsight flips: **2**
- dominant hindsight choices: HOLD_BUYS 31 games, HOLD_SELL 14, UNIT0_HARVEST 7, UNIT0_CARE 4.

The problem is observability. At step 120, HOLD_BUYS has six classes of **identical current public/legal feature frames** containing both positive and negative outcomes; 29 cases lie in those ambiguous classes and the worst within-class counterfactual delta span is **60,410**. A larger arithmetic utility formula cannot distinguish futures that are identical in its allowed information.

The exhaustive one/two-predicate semantic gate search found one discovery-safe flipping gate: `step==144 && second_shop==PIZZA_SHOP && opp_money>=175 -> HOLD_BUYS`. It activates four discovery cases, flips two, and has minimum discovery delta +2,742—but **zero independent holdout activations**. Lower nearby money thresholds introduce discovery regressions; higher thresholds eliminate activation. HOLD_SELL produced no discovery-safe flipping gate with independent support.

Decision: **ORACLE-U TERMINALLY REJECTED FOR PRODUCTION**. Headroom is real, but the tested legal selector is not generalizable.

## 12–14. ORACLE-W — commitment-aware worker/task assignment

Methodology: the current GPManager remains the baseline. The oracle does not pretend Manhattan distance is the full task cost. Instead it performs exact 24-turn reassignment of the existing coherent Bonsai route task streams, testing all 78 pairwise swaps among the first 13 proven units plus stable left/right rotations. Exact continuation automatically prices travel, pickup detours, carried inventory, deadline effects, abandonment and downstream task interactions.

- CUDA games: **93,879**
- active candidate jobs: **59,556**
- activation states: **1,159**
- positive-regret states: **575**
- state regret mean / median / p90 / max: **+1,721.47 / 0 / +5,647.4 / +47,243**
- per-game best delta mean / median / p90 / max: **+10,302.52 / +8,464 / +27,387 / +47,243**
- hindsight loss→win flips: **17** screen / **11** canary
- fresh holdout hindsight flips: **2**; historical holdout hindsight flips: **4**
- all 30 paired-seat groups have a positive hindsight best delta in both seats.

The best swap is not stable: winning assignments are distributed across SWAP_1_2, SWAP_3_11, SWAP_0_3, SWAP_1_3, SWAP_0_2, SWAP_3_6, SWAP_0_8, SWAP_0_11 and others.

The only discovery-safe one/two-predicate gated swap with a discovery flip was `step==264 && market_orders_sell<=1 && opp_tiles_weed>=1 -> SWAP_10_12 for 24 turns`. On independent fresh holdout it activates three times, is negative twice, sums **−8,232**, and reaches **−4,490**. Semantic neighbors either remove activation or remain holdout-negative.

Decision: **ORACLE-W TERMINALLY REJECTED FOR PRODUCTION**. It has the largest hindsight ceiling but no stable observable assignment rule; adding Hungarian/JV/min-cost machinery would merely optimize an unvalidated cost model and is therefore unjustified.

## 15–17. ORACLE-M — selective macro-option planning

Methodology: at strategic transition probes `[72,144,216,288,360,432,504,576,648]`, compare coherent temporary commitments to each frozen Bonsai route `(22,23,29,32,34,46,63,64,70,77,79)` for 24, 72 or 144 turns. The controller continues ticking and higher legacy/exploit/feed authorities remain intact. No full-season LP or hidden-future production planner is introduced.

- CUDA games: **18,666**
- active candidate jobs: **14,897**
- activation states: **531**
- positive-regret states: **307**
- state regret mean / median / p90 / max: **+1,385.40 / +10 / +4,813 / +17,758**
- per-game best delta mean / median / p90 / max: **+5,584.46 / +4,023 / +14,081 / +17,758**
- hindsight loss→win flips: **13** screen / **9** canary
- fresh holdout hindsight flips: **0**; historical holdout hindsight flips: **4**.

The best discovery-safe public gate was `step==144 && second_shop==SMOOTHIE_SHOP && opp_tiles_weed==0 -> route23 for 72 turns`: four discovery activations, two flips, all nonnegative. On independent fresh holdout it activates six times and is **6/6 negative**, aggregate **−93,100**, minimum **−36,103**. Nearby semantic gates remain 6/6 fresh-negative.

Decision: **ORACLE-M TERMINALLY REJECTED FOR PRODUCTION**.

## 18–19. Cross-oracle ranking and winner decision

Hindsight ranking by robust competitive ceiling is **W > U > M**: W has 17 flips and +10.30k mean best delta; U 16 and +9.33k; M 13 and +5.58k. Implementation-risk ranking is the reverse: U is simplest, W requires a meaningful task-cost model, M carries state-manifold/commitment risk.

There is **no production winner**. W wins the clairvoyant oracle ranking, but its observable selector fails holdout. U is the most distillable idea but its public-state ambiguity is directly demonstrated. M is both lower-ceiling and severely holdout-brittle. The evidence-first rule therefore chooses rejection rather than implementing a hidden strategic optimizer.

## 20–29. Production subsystem, representation, semantics, and evolvability

20. No optimizer subsystem/module was added to the champion lineage.
21. Bonsai remains the complete strategic authority; nothing external bypasses it.
22. Oracle representation uses bounded integer candidate IDs, fixed route/unit enums and exact simulator actions. No production representation was introduced.
23. No floating utility/sigmoid/Gaussian/learned ranking code was added. The rejected gate search used integer comparisons only.
24. CUDA oracle implementation: `cuda-engine/src/bkg4_optimization_oracle_eval.cu`.
25. No CPU C++ optimizer was implemented because no branch survived the direct holdout promotion floor. Implementing dead production code would violate the campaign rule.
26. Python additions are experiment/capture/analyzer tooling only; the production Python agent is unchanged.
27. Search parameterization is semantic and bounded: U legal action family + public gate predicates; W worker-stream pair and commitment horizon; M frozen route + commitment horizon + public gate.
28. Evolvability/semantic-neighborhood result: the U safe threshold is isolated/no-holdout; W neighbors remain fresh-negative; M neighbors remain 6/6 fresh-negative. Neighborhood quality therefore confirms brittleness.
29. Search-vs-evolvability ablation terminates before tree selection: **both modes reject all candidates at the same direct holdout floor**. There is no legitimate λ=0 vs λ=1 shipping comparison after the promotion floor is violated; lineage-Q cannot turn a holdout regression into an acceptable candidate.

## 30–36. League, seats, holdout, metamorphic, minimization

30. Historical before/after: no production candidate survived, so the accepted champion remains unchanged. Hindsight oracle evidence is not reported as shipped improvement.
31. Current hard-family before/after: unchanged for the same reason. The official canary remains the authoritative absolute reference at 37W/27L.
32. ReCurSiON before/after: unchanged; own-history protection remains mandatory and no candidate was allowed to bypass it.
33. Independent holdout was decisive: U had no support; W regressed; M regressed severely.
34. Both-seat oracle consistency is strong at the clairvoyant level (U 29/30 paired groups with positive best delta, W 30/30, M 29/30) but does not rescue selector generalization.
35. Metamorphic/public-information check: identical allowed current frames can imply opposite U counterfactual signs. Therefore any supposed selector that changes between those frames would necessarily rely on disallowed information. This is a terminal anti-fingerprint check.
36. Minimization ended at rejection: no extra parameters/tasks/options were retained.

## 37–46. Canonical identity, hot path, performance, parity, official engine

37. Canonical controller semantic hash remains `e9e66dfc9985a2111dc7c1b332fb569a02194528dd6512ccf3f154bd28abbc49`.
38. Combined champion identity remains `8cd92cc600537f5f0157925cd0b7d0de203e3b0d8187dd5389e02af96c9faf43`.
39. Node count remains **256**.
40. Blackboard/state remain **18 / 0** slots; max live intents 0; static max visits 179.
41. Existing hot-path profile is unchanged: visits p50/p95/p99/max 101/124/129/132; depth 6/6/6/max11; feature evals 43/65/70/max73.
42. Corrected CUDA oracle throughput: U **5,840.53 games/s**, W **9,043.44 games/s**, M **3,766.64 games/s**. The oracle kernel compiled with 149 registers, 6,240-byte stack frame and zero PTX spills. Production CUDA cost impact is **zero** because the optimizers were rejected.
43. C++ production performance impact: **zero**.
44. Python production latency impact: **zero**.
45. No new tri-engine runtime path exists. Existing accepted champion parity remains 0 Python/C++ and 0 Python/CUDA mismatches over 2,876 reachable ticks; a final spot-check is run before commit.
46. No new official 64/128 canary is warranted because no candidate reached maturity. The unchanged authoritative canary remains **37W/27L**.

## 47–52. Rejections, remaining losses, repository, verdict

47. Rejected branches: U—irreducible current-public ambiguity/no holdout-supported gate; W—heterogeneous optimal assignment and holdout regression; M—lower ceiling plus catastrophic fresh generalization failure.
48. Remaining hard canary losses: Ryo Hasegawa 20, Arman Tuganbaev 4, peikopon 2, ReCurSiON tape 1.
49. Highest *clairvoyant* regret domains remain worker allocation and capital/market timing, but the next campaign should seek a new **causal observable separator** rather than attach a more complex optimizer to the same ambiguous states. Ten of 27 canary losses are not flipped by any tested U/W/M oracle and need different BKG4/engine-mechanic search.
50. Repository cleanliness is verified after tests and commit.
51. Local commit SHA is recorded after the terminal evidence commit. Nothing is pushed or submitted.
52. Final verdict: **OPTIMIZATION PROPOSALS TERMINALLY REJECTED**.

Machine-readable evidence: `tools/bkg4_search/OPTIMIZATION_ORACLE_TERMINAL.json`.
Raw large CUDA result artifacts remain under ignored `scratch/bkg4_search/optimization_oracle/` and are SHA-256 referenced by the terminal JSON.
