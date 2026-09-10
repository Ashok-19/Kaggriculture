# Bounded executor/mechanics fallback for Shop Plus

**MECHANICS FALLBACK FAIL - CURRENT GOLD PLAN EXHAUSTED.**

Thirty-two bounded executor overlays were screened, confirmed and frozen against
the Shop Plus reference. None produced a material whole-policy improvement.
The underlying production route pack is byte-identical throughout, nothing was
promoted, and `assets/champion/` is unchanged.

Team: ReCurSiON. Starting HEAD `3ef51472eb33e713a2f7c403d6db7831a4662f09`, clean
`main`. This is the single small fallback permitted by
`docs/FINAL_GOLD_MEDAL_BONSAI_PLAN.md` after
`workspace/gold_route_search/phase01_replay_seeded` returned
`PHASE 1 FAIL - PRODUCTION ROUTE PATH NOT VALIDATED`.

## Reference identity (verified, unmodified)

| Property | Value |
| --- | --- |
| Agent | `shop_router_0909/plus` |
| Controller nodes / blackboard / node state | **85 / 2 / 0** |
| Controller bytes | **3,224** |
| Controller SHA-256 | `84bbfe9a...0b1cc450` |
| Controller semantic hash | `2207e9f0...613dd03b` |
| Route pack | **13 x 719** records, SHA-256 `f711680c...eebe696` |
| Arity table | SHA-256 `5e830564...bb882af` |

A candidate is only a deterministic overlay applied to the action the frozen
`shop_router_0909::tick` has already produced, and only for the searching seat.
The controller, the route pack and every opponent asset are untouched, so the
all-zero overlay is the reference by construction; `fidelity.json` verifies it.

## What the current executor actually does

Source: `src/kaggriculture_bonsai/agents/shop_router_0909/reference_port/native.hpp`
and `native/cpp/include/sim.hpp`.

| Question | Answer in source |
| --- | --- |
| DROP | Deposits the whole cargo in inventory-insertion order, and **destroys** whatever does not fit (`f.discarded[it] += amount - take`). Requires shed adjacency. |
| PLACE | Deposits a chosen item and quantity, clamped to cargo and to free room; the remainder stays carried. Non-destructive. |
| PICKUP | Takes `min(n, shed[item])` from the shed; requires adjacency. |
| Capacity projection | `project()` estimates the post-turn shed for shed-adjacent workers only, in the donor's worker/inventory order. It is a **sell-sizing** estimate; nothing ever uses it to avoid a discard. |
| Cargo priority | None. There is no value ordering anywhere in the executor. |
| Feed | `OP_FEED` consumes one WHEAT from the **acting worker's own cargo**, not the shed. Two consecutive unfed days make the animal escape. |
| Feed protection | Only indirect: `advance_sales` and `full_sell` skip WHEAT and FERTILIZER. The route tape's own WHEAT sales are never restrained. |
| Lookahead | `advance_sales` reads exactly the next tape step, gated by `step % 4` and `(step+1) % 72`. |
| Advanced-sale interaction | `subtract_advanced` removes the lead quantity from the next step's own SELL orders; cargo is not consulted. |
| Overlay precedence | `tape -> repair_weeds -> subtract_advanced -> advance_sales -> full_sell -> order clamp -> liquidate(718)`. |
| CPU/CUDA | One `BKG4_VM_HD` source shared by both; `native/cuda/src/mechanics_fallback.cu` compiles unchanged for host and device. |

Order of resolution matters: unit actions resolve before market orders in the
same step, so a sale can only free capacity for a **later** step's deposit.

### Measured weaknesses, and how large they are

Whole reference games, four native opponents, both seats:

| Weakness | Measured on the reference |
| --- | --- |
| Cargo destroyed by capacity | **7 units per game**, all at price 1, and all of it from the **overnight** `drop_inventories`, not from a DROP action |
| DROP actions with cargo exceeding free room | **0 over the whole panel** |
| Goods left unsold at the end | **0** |
| Empty coop/pasture at the end | ~1 per game |
| Feed at risk | none observed; Shop Plus sells wheat deliberately and buys feed back with `BUY_PRODUCT` |

That is the finding that decides this experiment: the two mechanics the plan
prioritised have almost no material to work on in this executor.

## Candidate set (frozen before any outcome was read)

`candidates.json`. Exactly **32**: **16 capacity**, **16 feed**, **0 combined**.
Parameters: `cap_mode cap_floor cap_room cap_look feed_mode feed_res feed_look`.

* **Family A, capacity-safe deposits.** `cap_mode` 1 replaces a destructive DROP
  with a partial `PLACE` of the highest-value carried item (canonical item order,
  so the choice never depends on inventory insertion order), keeping the rest as
  legal cargo, and PASSes rather than destroy when the shed is full;
  `cap_floor` is the units-at-risk activation threshold. `cap_room` adds one
  pre-emptive sale of the **cheapest** non-wheat, non-fertilizer product when
  free room is below the target and the tape schedules a deposit within
  `cap_look` steps.
* **Family B, projected feed preservation.** `feed_mode` 1 clamps the route's own
  WHEAT sales so forecast FEED demand plus `feed_res` stays in stock; mode 3
  converts an idle shed-adjacent PASS into a WHEAT `PICKUP` for a lane that will
  FEED within `feed_look`; mode 2 does both.

Nothing else changes. Step 718 is never overlaid; liquidation stays the frozen
terminal behaviour.

## Correctness

`fidelity.json`, `official.json`.

| Check | Result |
| --- | ---: |
| Zero-edit overlay versus the frozen executor (16 games, CPU and CUDA) | **0** semantic mismatches |
| Mechanics boundary panel, 32 overlays x 8 whole games, CPU versus CUDA | **0** mismatches over 256 games |
| Games in which a mechanic actually fired | 144 of 256 |
| Runtime/reset/invalid-state failures | **0** |
| Official engine (kaggle-environments 1.32.7) versus native, reference and candidate | 16 games, **0** mismatches |

Whole games cover the dawn reset, the shed-capacity boundary, mixed cargo,
multi-worker ordering and the terminal turn. `tests/test_mechanics_fallback.py`
pins the partial-PLACE rule at 99/100/full, the activation threshold, the
deposit-conditioned pre-emptive sale, the feed clamp arithmetic, the lane-scoped
top-up and the untouched terminal turn.

## BUILD screen

`build.json`. 32 contexts (4 seeds x 2 seats x 4 live native opponents: Shop
Plus, direct Shop, route-3 continuation, route-12 continuation), 1,056 games,
whole games to the final actionable turn.

* **2 of 32** candidates improved cohort-balanced utility; **0 of 32** improved
  the median paired margin.
* Semantic dedup by observed whole-game closure, not by parameter tuple:
  **14 of 16** capacity overlays are **identical to the reference** because the
  partial-deposit rule never activates. 17 distinct behaviours remain.
* Every feed candidate regresses. The clamp variants lose 8,000-9,600 coins of
  median margin: Shop Plus's wheat sales are revenue, not a mistake.

Two distinct finalists were frozen: `c07` (pre-emptive sale, room 20, lookahead
4) and `f28` (feed top-up only, reserve 2, horizon 24).

## Confirmation and INTERNAL_DEV

`confirmation.json`, `dev.json`. Confirmation: 32 fresh seeds, both seats, four
live native opponents, 256 contexts each. DEV: 64 fresh seeds, both seats, the
same four opponents, 512 paired games, evaluation only, no tuning after.

| | c07 confirmation | f28 confirmation | c07 DEV |
| --- | ---: | ---: | ---: |
| Utility change | +8.98 pt | -9.38 pt | **+10.35 pt** |
| One-sided 90% lower bound | +7.81 pt | -12.11 pt | **+9.57 pt** |
| Median paired margin | +5 | -42 | **+4.5** |
| Mean paired margin | | | **-2.4** |
| Seat 0 / seat 1 | +8.98 / +8.98 | -9.38 / -9.38 | +10.35 / +10.35 |
| Shop Plus / direct / route-3 / route-12 | +35.94 / 0 / 0 / 0 | -32.81 / -6.25 / 0 / +1.56 | **+41.41 / 0.00 / 0.00 / 0.00** |

DEV transitions: L->W 0, T->W **106**, W->T 0, W->L 0, T->L 0.

Every point of the headline gain is a tie converted into a win **inside the Shop
Plus mirror cohort**. In a mirror both seats play the same policy, so the paired
margin is exactly zero; one extra small sale breaks that symmetry by a few coins.
Against the three opponents that are not the subject itself the effect is exactly
**0.00 points**, and the mean paired margin is negative.

## Mechanics diagnostics (DEV, 512 games)

| Quantity | Value |
| --- | ---: |
| Overflow units prevented | 349 of 3,854 discarded |
| Discard value prevented | 349 coins over 512 games (~0.7 per game) |
| Feed reserve violations avoided | 0 (mechanic not in the frozen winner) |
| Starvation / escape events avoided | 0 (empty pens 534, unchanged) |
| Route-action failures avoided | 0 |
| Extra sales issued | 432 pre-emptive sales, 864 units |
| Final unsold inventory change | 0 |

The mechanic is real and it fires, but the economics it recovers are negligible.

## External transfer

`external.json`. The same four frozen local policies declared for the route
pilot - Gold Plus V8, Gold V25, the BKG4 champion and the direct Shop port - on
32 fresh seeds, both seats, 512 official-engine games. Gold V25, Gold Plus V8
and the direct Shop port share ancestry with the subject; this is a live
transfer check, not four independent strategic families. No external result was
used for tuning.

Per-opponent utility change: **0.00** for all four. Mean **0.00**, worst cohort
**0.00**. No collapse, and no transfer benefit either.

## Predeclared gates

| Gate | Result |
| --- | --- |
| DEV utility >= +3 pt with a positive 90% lower bound | PASS (+10.35, LB +9.57) |
| Positive paired margin overall | PASS (median +4.5; mean -2.4) |
| Effect attributable to the mechanic | PASS (349 units of discard prevented) |
| Benefit in both seats | PASS |
| Benefit against multiple native opponents | **FAIL (1 of 4; mirror only)** |
| No external-transfer collapse | PASS |
| Zero CPU/CUDA semantic mismatches | PASS |
| Zero official/native action mismatches | PASS |
| Zero runtime/reset/invalid-state failures | PASS |
| Route pack byte-identical | PASS |

## Work

| Quantity | Value |
| --- | ---: |
| Simulated turns | 3,433,944 |
| Native FGE | **4,776.0** (hard cap 5,000) |
| GPU | 3,376,424 turns in 37.5 s |
| Native CPU | 57,520 turns in 2.1 s |
| Official-engine games | 528 |

## Files

`manifest.json` frozen identity, seeds and candidate set. `candidates.json` the
32 overlays. `fidelity.json` zero-edit and boundary parity. `build.json` screen,
closures and dedup. `confirmation.json`, `dev.json`, `official.json`,
`external.json`, `ledger.json` every native batch, `summary.json` the gates and
the verdict.

## Conclusion

The bounded executor path is closed. Capacity-safe deposits have nothing to
recover in this executor, projected feed preservation is actively harmful to it,
and the one candidate that survives screening wins only against a copy of
itself. Per the plan, the existing deployable family is kept and no further
incremental search layer is added here.

**MECHANICS FALLBACK FAIL - CURRENT GOLD PLAN EXHAUSTED.**
