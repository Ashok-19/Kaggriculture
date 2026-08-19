# BKG4 Temporal Information-State Campaign — Terminal Report

Date: 2026-08-20  
Starting HEAD: `8c001a0f370b5980b2f30a526fa763bfee3d24ca`  
Frozen VM foundation: `8381e027646ba6494cbe4dd06a810ea55ec83575`  
Frozen accepted controller: `e9e66dfc9985a2111dc7c1b332fb569a02194528dd6512ccf3f154bd28abbc49`  
Frozen combined identity: `8cd92cc600537f5f0157925cd0b7d0de203e3b0d8187dd5389e02af96c9faf43`

## Terminal verdict

**TEMPORAL INFORMATION-STATE HYPOTHESIS TERMINALLY REJECTED**

The campaign tested the minimum legal information state in the required order: complete current legal state (R0), finite semantic legal history (R1), then an offline perfect-hidden-truth upper bound for set-valued inference (R2 upper bound). R1 contains real information, but the strongest causal temporal selector failed broad held-out game validation. Perfect opponent hidden totals/seeds added zero representation separation beyond R0+R1 and truth-gated policies remained unsafe. Therefore a production temporal module, hidden-state observer, semantic opponent regime, and active-information probe are not justified by current evidence.

No champion behavior changed. No BKG4 ABI, feature registry, VM semantics, feed-deferral module, legacy compatibility state, or accepted engine exploit was modified.

## Research interpretation

The cited finite-state-controller, predictive-state, stateful-BT, and imperfect-information literature supports legal history as a legitimate controller input. It does not imply that memory is useful in this game. The experiment therefore treated temporal state as a falsifiable representational hypothesis rather than an architectural requirement.

The observed result is consistent with H3: finite legal history can disambiguate some exact-current-state cells, but those distinctions do not robustly identify which U/W/M intervention should be taken. The remaining oracle variation is better explained by action-candidate insufficiency plus future opponent/environment continuation sensitivity than by a missing compact hidden resource variable.

## Correction to the previous ambiguity test

The previous optimization-oracle ambiguity frame was deliberately coarse. It did not enumerate exact worker positions, every spatial tile field, complete current BKG4 BB state, or all legal compatibility/exploit memory. This campaign replaces that diagnostic frame with the complete registry-v1 surface before making any partial-observability claim.

## R0 — complete current legal information state

The CUDA capture serializes **3,828 exact int32 values per decision state**:

- **3,793 registry-v1 values**: every legal feature over every valid constant argument used by registry v1.
- **18 current BKG4 blackboard slots** from the accepted feed-deferral controller.
- **15 legal persistent compatibility/exploit values**: frozen legacy public-prefix state plus accepted exploit latches.
- **2 runtime metadata values** (`fault`, tick count).

Registry enumeration includes scalar time/economy/unit/land/hire features; all 9 market prices/inventories; own shed/seeds; own per-unit inventory for 40 unit argument slots × 12 items; all shop positions/counts; all own/opponent quadrant flags; own/opponent worker X/Y/tile for 40 unit slots; and every own/opponent tile kind/item/yield/water/feed/care/fertilizer/lifecycle/animal field over all 100 tiles.

R0 capture: 61 cases × 22 probe steps, zero CUDA faults. Capture SHA-256 is recorded in `tools/bkg4_search/TEMPORAL_INFORMATION_STATE_TERMINAL.json`.

### R0 ambiguity

Across all active oracle decisions:

| Branch | R0 cells | Mixed-sign cells | Cases in mixed cells | Worst within-cell regret span |
|---|---:|---:|---:|---:|
| U | 6,005 | 93 | 392 | 153,574 |
| W | 46,558 | 261 | 862 | 153,574 |
| M | 11,494 | 112 | 361 | 33,338 |

Key target cells remain mixed under complete R0. Examples:

- U `HOLD_BUYS`: 17 mixed cells / 63 mixed cases / max span 60,410.
- U `HOLD_SELL`: 23 / 79 / 60,401.
- W `SWAP_0_3`: 14 / 78 / 82,457.
- W `SWAP_1_2`: 8 / 46 / 60,499.
- W `SWAP_6_9`: 3 / 8 / 17,495.
- M `ROUTE_23_H144`: 7 / 27 / 17,898.
- M `ROUTE_46_H144`: 9 / 33 / 17,898.
- M `ROUTE_77_H72`: 7 / 27 / 33,338.

Therefore the old coarse representation was not the sole cause of ambiguity.

## H0 — current-state separator attempt

Exact-R0 holdout reuse exposed one causal simplification for `HOLD_BUYS`: at step 120, first shop SMOOTHIE, high opponent public cash distinguished a repeated positive state. A threshold sweep (`OPP_MONEY >= 250..570`) was evaluated as a direct current-state policy.

Oracle screen support existed: 2 discovery activations (+6,057 aggregate) and 2 fresh-screen holdout activations producing 2 loss→win flips (+34,642 aggregate). But broad validation showed no recurrence in historical336 or independent fresh24; only two sealed24 cases activated and were positive (+6,057). This is insufficient breadth/evolvability for promotion.

**H0 rejected as sufficient.** Complete current state contains some narrow separators, but no major robust current-state promotion was established.

## R1 — finite legal semantic history

R1 contains **368 semantic values** only. No history hash, trajectory ID, replay action tape, seed, opponent identity, or raw 720-turn array is stored.

The capture uses legal consecutive observations plus our own emitted actions and prior deterministic controller state. It contains:

- 1/4/24/72-turn deltas for market inventory/price, own/opponent money, units, land, hires, own/opponent crop and animal counts, tile-kind counts, and shop count.
- route-at-minus-1/4/24/72 and validity flags.
- rolling own action counts over 1/4/24/72 for all unit and market operations.
- event-since counters for opponent land/unit/crop/animal/structure/weed changes, shop unlock, route change, and per-product market inventory direction/10000-threshold crossing.

R1 capture: 61 cases × 22 probes, zero CUDA faults.

### Memory-length ablation

| Branch | R0 mixed | +1 | +4 | +24 | +72 | +events |
|---|---:|---:|---:|---:|---:|---:|
| U | 93 | 93 | 91 | 90 | **85** | 85 |
| W | 261 | 261 | 261 | 256 | **256** | 256 |
| M | 112 | 112 | 104 | **100** | 100 | 100 |

Full R1 reduces mixed cells by 8.6% for U, 1.9% for W, and 10.7% for M. Event-since additions do not reduce exact-cell counts beyond the full 72-turn semantic representation.

The most recurrent informative temporal dimensions are opponent public money deltas, recent fertilizer-market inventory movement, and time since opponent weed-state change. These are real causal/public trajectory features, not identity proxies.

### Same-current-state / different-history evidence

Natural exact-R0 cells exist where legal histories differ and oracle signs differ. R1 splits 18 U mixed R0 cells (8 fully resolved), 23 W cells (5 fully resolved), and 16 M cells (12 fully resolved). Examples include identical current R0 but different 24/72-turn opponent-money and fertilizer-market trajectories.

This directly proves that legal history can carry information beyond the current registry snapshot. It does **not** prove a safe policy exists.

## H1 — temporal selector search and broad game validation

Bounded semantic predicate search found the strongest screen-local W hypothesis:

`first shop == PIZZA && opponent STRAWBERRY planted-tile count increased by >=2/3 over the prior 24 turns -> apply SWAP_6_9 stream reassignment at step 192 for a bounded horizon`.

The diagnostic rule passed the temporal metamorphic test: perturbing each of 367 unrelated temporal components preserved the decision 367/367 times; changing the causal strawberry-growth feature or current shop context changed the decision.

Screen-local support was strong enough to justify direct policy testing: discovery 2 positive activations; historical screen holdout 4 positive activations with 2 loss→win flips (+37,230); fresh screen holdout 2 positive activations (+8,860).

Broad CUDA policy validation rejected it:

- Historical336, growth>=2, H24: 45 activations, +76,164 aggregate margin, but **25 negative deltas, 25 protected margin regressions, 6 win→loss regressions**. Wins fell 251→250 despite 5 loss→win flips.
- Historical336, growth>=3, H24: 27 activations, -35,980 aggregate, 19 protected regressions, 5 win→loss regressions.
- Independent fresh24, both >=2 and >=3 H24: 2 activations, **2/2 negative**, -39,194 aggregate, two wins lost, minimum delta -19,597.
- 12/36/48-turn policy horizons were also evaluated and did not produce a safe neighborhood.

**H1 rejected for promotion.** History is informative but not a robust action selector under the tested causal grammar.

## R2 / H2 — perfect hidden-truth upper bound

Before implementing an interval observer, the campaign captured exact opponent private state **offline only** as a scientific upper bound:

- total holdings (`shed + all carried inventory`) for 12 items;
- exact opponent seed holdings for 5 crops.

These 17 values never enter production inference.

Adding perfect hidden truth on top of complete R0+R1 produced **zero additional equivalence-class separation**:

| Branch | R0+R1 mixed | R0+R1+perfect truth mixed |
|---|---:|---:|
| U | 85 | **85** |
| W | 256 | **256** |
| M | 100 | **100** |

Target mixed-cell counts and worst regret spans are also unchanged. Thus these strategically plausible hidden totals are either already determined by R0+R1 in this corpus or do not explain oracle-sign heterogeneity.

A direct perfect-truth policy upper bound was then tested. The strongest truth gate (`first shop PIZZA && true opponent STRAWBERRY seeds == 0 -> SWAP_6_9`) still failed historical336: 40 activations, 16 positive/22 negative deltas, +92,990 aggregate, but **22 protected regressions and 6 win→loss regressions**. It had no support in independent fresh24. A true-wheat-seed HOLD_BUYS gate had zero historical/fresh activation.

Because even perfect hidden truth is not a safe selector, a legal set-valued observer cannot improve the information ceiling. Building one would add complexity and estimator error without demonstrated decision value.

**H2 rejected.** Observer not implemented. Containment rate, interval width and exactness are therefore intentionally N/A rather than invented metrics.

## Ryo-loss diagnosis

The 20 remaining Ryo-family canary losses are not one homogeneous hidden regime. Their U/W/M oracle-headroom clusters are:

- U+W: 7
- NONE: 5
- U+M: 2
- W+M: 2
- M: 2
- U: 1
- U+W+M: 1

A compact public-trajectory diagnostic splits them into 16 early-land/cash-drawdown cases and 4 late-land/cash-positive strawberry-burst cases. This is mechanism-based analysis only; no opponent name/identity feature is used by any tested rule.

The temporal campaign promoted **0/20** Ryo losses because every candidate that recovered some Ryo oracle regret failed broader protected/fresh validation.

## Ten U/W/M-unflippable losses

Exactly 10 of the 27 original canary losses had no tested U/W/M oracle flip: 5 Ryo, 4 Arman, 1 ReCurSiON-history. Temporal information cannot repair these by selecting U/W/M because the candidate action family itself lacks sufficient headroom.

Their trajectories include both early-land/cash-drawdown and late-land/strawberry-growth patterns, reinforcing that the next gain must come from a new strategic capability (lane/production/market/tactical/recovery/endgame/engine-mechanic continuation), not a more elaborate selector over the same U/W/M actions.

## Semantic regimes and active information

No semantic opponent regime was promoted. The two Ryo trajectory groups above are diagnostic summaries only and have no production authority.

Active information probing was **not tested**. Its prerequisite failed: perfect hidden truth added no marginal representation separation and truth-gated policies were unsafe. A probe cannot justify its immediate cost when the hidden information it would reveal has no demonstrated safe downstream value.

## Performance / parity / lineage consequences

No temporal production path was added, so champion runtime cost is unchanged:

- controller nodes: **256**
- BB slots: **18**
- NodeState slots: **0**
- static max visits: **179**
- accepted hot-path profile remains p50/p95/p99/max node visits 101/124/129/132.

No temporal Python/C++/CUDA parity target exists because no temporal path was promoted. Existing accepted champion tri-engine parity remains the required frozen reference. CUDA experiment evaluators reported zero runtime-fault games.

The accepted official Python canary remains **37W / 27L**. It was not rerun because no production behavior changed and no promotion reached maturity.

## Rejected hypotheses / features

Rejected as promotion mechanisms in this campaign:

- exact R0 high-cash SMOOTHIE HOLD_BUYS gate: too narrow/no broad recurrence;
- opponent 24-turn strawberry-growth W gate: broad protected/fresh regressions;
- longer/shorter W horizons: no safe neighborhood;
- simple R1 event/cash/market-history gates: information gain without robust decision gain;
- perfect hidden wheat/strawberry seed gates: still unsafe and no marginal representation separation;
- explicit set-valued hidden-state observer: rejected before implementation by perfect-truth upper bound;
- semantic opponent regime controller: insufficient evidence;
- active information probing: prerequisites failed.

## Next strategic implication

Do not reopen generic utility arbitration, Hungarian/JV worker optimization, or macro-route MPC with the same snapshot/temporal selector formulation. The information-state campaign shows that more selector state is not the current bottleneck.

Return to **new causal BKG4 capability / engine-mechanic / lane discovery** with CUDA-scale counterfactual search, especially for the 10 U/W/M-unflippable losses and the Ryo regimes where no oracle candidate had headroom. Any future temporal revisit requires a new action family whose outcome actually becomes separable by legal history.

Canonical machine-readable evidence: `tools/bkg4_search/TEMPORAL_INFORMATION_STATE_TERMINAL.json`.
