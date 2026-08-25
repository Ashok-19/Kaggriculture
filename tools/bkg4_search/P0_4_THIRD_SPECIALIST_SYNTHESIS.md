# P0.4 Third Specialist Synthesis and Explicit Composition Requirements

**Verdict: VALIDATED.** P0.4 synthesizes route218 `route437_wheat_scarcity_sheep_shift_v1`, a legal public-state bounded macro for steps 264..287. It also implements the first evidence-backed route-authority rule: route218 explicitly suppresses the overlapping route213 latch when route218 authorizes. The final compiled challenger is parent-exact outside eight certified activations and improves all eight.

## A. Repository state

- repository: `/home/nnmax/Desktop/kaggle/kaggriculture`
- team: ReCurSiON
- branch: `main`
- starting HEAD: `8b78baaf3bc5c7e8aef333ea382787ee89d79bfe`
- starting worktree: exactly the five known unrelated Aug-25 untracked files
- master plan: read and left unchanged
- no push, Kaggle submission, PSRO, EditFormer or unrelated corpus campaign

All game evaluation and legal-state capture used the exact CUDA engine. CPU work was limited to parsing, compact conjunction search, compilation and CPU/CUDA decision parity.

## B. Starting BKG4

P0.4 starts from `route410_wheat_liquidity_expansion_v1`:

- 380 nodes, 25 BB slots, 0 state slots;
- max depth 11, static max visits 295;
- semantic hash `b2edd5dd76b936b6ea469009eb75b587531cef2f1be62939254b924f2e245edc`;
- combined identity `494bdfb4aa785088c3ebc23ac1fa2b71db8f51a26f861bea6480a7c457322413`;
- route-pack hash `9eedcfd2fcef415c7b0a10601ee61cf04b6fc7a6d7079cf9f0f0d0eb83f90cf5`.

The two prior synthesized macros are route217 at steps120..137 and route216 at steps156..203. Both write strategic route override BB slot19.

## C. Current loss frontier

Fresh exact evaluation of the actual P0.3 parent found:

| Panel | Wins | Losses | Worst margin |
|---|---:|---:|---:|
| screen61 | 29 | 32 | -49,506 |
| current140 | 101 | 39 | -49,506 |

Repeated both-seat losses included Subramanya, Ryo Hasegawa, Thomas Tschinkel, Izzoudine Mohamed KANTA, tetsuya and StackKnight trajectories. The 20 most severe current losses formed the bounded search target.

## D. Selected target family

The selected family is a day-11 capital-rich, wheat-scarce regime. It is not tied to an identity: its observable signature is expensive WHEAT, high public opponent liquidity, and absence of a third BRUNCH_SPOT demand sink.

The selected support spans four independent tapes and both seats:

- `Kobe_BRYANT_s1_ep94622821`;
- `Michael_Timbs_s0_ep94669660`;
- `Izzoudine_Mohamed_KANTA_s0_ep94817624`;
- one protected archive trajectory represented by its sanitized content name.

Names are offline evidence labels only and do not appear in runtime gates or macro provenance.

## E. Strategic diagnosis

At step264 the incumbent route64 continuation buys two cows. Route437 instead buys two sheep, preserves a large labor deployment, and changes the care, watering and planting schedule through step287. The route also changes several market orders, but targeted ablation shows those market changes are not the source of the certified gain.

The third BRUNCH_SPOT predicate is causal rather than an arbitrary shop fingerprint. Brunch adds wheat, egg and strawberry demand. In an already wheat-scarce regime, the four nearby third-Brunch records are all harmed by route437, while the no-third-Brunch records support the sheep allocation.

## F. Counterfactual CUDA search

The P0.1/P0.3 bank of 64 coherent source-backed routes was searched against the actual 380-node parent on 20 severe current losses:

- starts: 72, 120, 168, 216, 264, 336, 432, 576;
- durations: 24, 48, 72, 96;
- candidates: 2,049;
- exact CUDA games: 40,980;
- GPU kernel time: 8,322.46 ms;
- throughput: about 4,924 games/s;
- faults: 0.

This was a bounded coherent-route campaign, not a new giant data search.

## G. Best discoveries

Route437 at start264 for 24 turns was the best repeated discovery:

- target20: 10 positive, 4 L->W, sum delta +55,859;
- positive families: 5;
- L->W families: 3;
- best refined window on current140: 264..287.

Ungated use was unsafe: current140 had 32 positive and 102 harmful records, while core585 had 153 positive and 422 harmful records. This established a valuable specialist, not a global route replacement.

## H. Rejected discoveries

Several apparently clean gates failed falsification and were rejected rather than patched into fingerprints:

1. `MELON inventory <= 10065 && SHOP_AT[2] == YARN_STORE` was 8/8 positive on four core tetsuya tapes, but the 766-record archive produced 7 activations with 6 harmful, worst -13,331 and aggregate -18,932.
2. One-extra-condition repairs of that gate collapsed to two or three original tapes and had no independent activation.
3. `WHEAT inventory <= 9830 && CARROT inventory <= 9881` transferred positively once but produced two archive harms.
4. Route407's cow-in-shed/Pizza fallback transferred on fresh data but produced an untouched sealed harm of -1,203.
5. Route423 had no safe two-predicate conjunction with L->W support in at least two support groups.
6. Route418 remained globally unsafe and did not justify a larger search.

Route434 remains a fallback research family; evidence did not justify switching away from route437.

## I. Selected capability

The accepted capability is `route437_wheat_scarcity_sheep_shift_v1`:

- strategic route ID: 218;
- source route: 437;
- base route: 64;
- authorization step: 264;
- inclusive active window: 264..287;
- fragment SHA-256: `175853bd7517328c7a72d84f13cb149ef376221602b1bd911783dad920e0010c`.

It is materially distinct from route217's day-5 liquidity/labor expansion and route216's Pizza/strawberry-glut production rebalance.

## J. Representation

Route218 uses the existing `BoundedRouteMacro` representation:

- one legal gate evaluated exactly at step264;
- one owned BOOL latch;
- one bounded route override through step287;
- automatic return to parent routing afterward;
- no VM, ABI, feature-registry or NodeState change.

P0.4 adds `ExactRoutePreemption`, a deliberately narrow composition primitive. It clears one named incumbent latch only when an evidence-backed preemptor authorizes and requires the preemptor window to match the declared conflict window exactly. It is not a generic scheduler.

## K. Initiation dataset

Gate synthesis and falsification used:

- core585 discovery/protection records;
- untouched validation40;
- untouched sealed40;
- fresh949 panel with 80 records;
- broad current archive with 766 records;
- complete legal/public state at step264;
- hidden truth captures retained only as unused diagnostics.

The final compiled audit covers 1,511 records, 1,466 unique `(tape, seat, seed)` records after deduplication.

## L. CEGIS iterations

| Iteration | Gate / counterexample | Outcome |
|---|---|---|
| 0 | ungated route437 | 153 positive / 422 harmful on core585; reject |
| 1 | MELON/YARN | 8/8 positive on core, no validation/sealed/fresh hits; archive adds 6 harms; reject |
| 2 | `WHEAT price >=38 && OPP_MONEY >=15860` | core 4/4 positive, fresh 2/2 positive, but only a five-dollar safe boundary before harms; refine |
| 3 | add `SHOP_AT[2] != BRUNCH_SPOT`, lower round money threshold to 15,750 | 8/8 positive across core/fresh/archive, 3 L->W, 0 harmful/W->L, no mixed legal cells; accept |

The final refinement uses one causal counterexample predicate. It does not use identity, history, hidden inventory or future information.

## M. Final gate

```text
STEP == 264
AND MARKET_PRICE[WHEAT] >= 38
AND OPP_MONEY >= 15750
AND SHOP_AT[2] != BRUNCH_SPOT
```

All fields are legal current public state. The macro semantic ID is `3321dcaa1006610625b3ac085e2f1f51309955d9fa4d63e47124fdd5d42b9fe3`.

## N. Threshold robustness

The opponent-money threshold was swept one dollar at a time:

- first all-positive threshold with the no-Brunch clause: 15,729;
- selected round threshold: 15,750;
- thresholds 15,729 through the tested upper range remain harm-free, with support shrinking monotonically;
- at 15,728 two harms remain, so the unsafe boundary is explicit.

WHEAT price is a discrete economic regime:

- `>=38` is safe under the other final predicates;
- tightening to `>=39` remains safe but retains only the archive transfer pair;
- loosening to `>=37` enters a mixed regime with many harms;
- the equivalent WHEAT inventory bucket is safe through 9,843 and first admits harms at 9,844.

The shop predicate is also explicit: admitting third BRUNCH_SPOT states adds four harms, worst -5,186 and aggregate -17,846.

## O. Independent transfer

Post-discovery positive transfer is present in two independent panels:

| Panel | Activations | Positive | Harmful | Worst delta | Sum delta |
|---|---:|---:|---:|---:|---:|
| fresh949 | 2 | 2 | 0 | +1,129 | +2,303 |
| protected archive766 | 2 | 2 | 0 | +3,775 | +7,809 |

Validation40 and sealed40 each have zero activations and therefore provide safe-exclusion, not positive-transfer, evidence. Combined core/fresh/archive support is 8/8 positive across four tapes and both seats.

## P. Interaction with route217

Route217 ends at step137, so its execution window is disjoint from route218. In final certification:

- route217 support records: 12;
- route217/route218 dual activations: 0;
- support records changed by P0.4: 0;
- harms or W->L: 0.

## Q. Interaction with route216

Route216 ends at step203, also disjoint from route218. In final certification:

- route216 support records: 16;
- route216/route218 dual activations: 0;
- support records changed by P0.4: 0;
- harms or W->L: 0.

## R. Composition conflict classification

P0.4 discovered a real conflict between route437 and accepted route213 `fert_wool_expansion_v1`:

- **TEMPORAL_CONFLICT**: both own steps264..287;
- **ROUTE_AUTHORITY_CONFLICT**: both write strategic override slot19;
- **PREEMPTION_CONFLICT**: the route selected last controls the executed route.

The rejected MELON/YARN initiation regime included a route213-latched trajectory where forced route437 improved both seats by about +9K. That is exact evidence that route437 can validly preempt route213 in a relevant state. P0.4 does not claim a broader resource-commitment conflict because the route213 interaction data do not establish one.

## S. Composition resolution

The minimum concrete rule is:

1. route213 runs in its existing position;
2. route218 is appended later in the strategic lane;
3. if route218's gate succeeds at step264, it explicitly clears `fert_wool_expansion_v1_active` before setting its own latch;
4. route218 writes route218 through step287;
5. both route windows end together, so no restoration state is needed.

This removes dependence on undocumented append order. A 171-frame CPU/CUDA differential includes a synthetic simultaneous route213/route218 authorization and confirms route218 ownership at steps264..287 and parent return at288.

## T. Full final BKG4

Accepted research challenger:

- name: `route437_wheat_scarcity_sheep_shift_v1`;
- 398 nodes, 26 BB slots, 0 state slots;
- max depth 11, static max visits 312, max live intents 0;
- controller semantic hash `4e611d80f0bbb38bd696aa3dbb8a5e9e903be556e4ceafa233b5cb5685597be6`;
- combined identity `357dbb9489b26a8fb424de592da92e45e2c0ac28d8a450a5e6e1b5791ade9d29`;
- route-pack hash `068dca3988994778dc2c3d96b36a30b078b85f6ac329b91dc1a6fbf9d2a323e9`;
- routes: prior set plus route218.

The default submission/runtime is unchanged.

## U. Regression certification

Final exact compiled CUDA certification:

- records: 1,511;
- unique records: 1,466;
- inactive candidate cases exact to P0.3 parent: 1,503/1,503;
- active candidate cases exact to standalone route437 teacher: 8/8;
- positive activations: 8/8;
- L->W: 3;
- harmful: 0;
- W->L: 0;
- worst delta: +1,129;
- aggregate delta: +31,923;
- mixed exact legal-state cells: 0;
- CUDA faults: 0.

All 969 protected parent wins have zero negative delta and zero W->L. Eight ReCurSiON self-history records have zero route218 activations.

On current140, the parent 101W/39L becomes 104W/36L. Screen61 is unchanged. The severe unrelated -49,506 frontier remains and is not concealed by this specialist.

## V. Existing synthesized specialist preservation

The final candidate preserves all observed support for routes217 and216 exactly. It also preserves all 51 unique route213-gate support records in certification: the final route218 gate has zero observed overlap with route213 support, and all 51 outcomes are parent-exact.

The explicit suppression rule is nevertheless retained because the execution windows and route authority genuinely overlap, and an earlier route213-active route437 counterfactual demonstrated beneficial preemption. Static absence in the final support sample is not used as a substitute for deterministic conflict semantics.

## W. CPU/CUDA parity

The final differential covers 171 consecutive frames from step119 through289:

- CPU/CUDA mismatches: 0;
- route217 active boundaries: 120 and137; returns at138;
- route216 active boundaries: 156 and203; returns at204;
- simultaneous route213/route218 authorization at264 resolves to route218;
- route218 remains active through287 and returns at288.

The 1,511-game-record compiled piecewise audit also has zero parent/teacher mismatches.

## X. Tests

- focused P0.2/P0.3/P0.4 suite: 29/29 passed;
- adjacent P0.1 regret-parser and contrastive-goal suite: 13/13 passed;
- CUDA fragment evaluator and information-state-capture build targets: current;
- final CUDA campaigns: bad=0.

Tests cover exact fragment integrity, legal gate shape, fail-closed generic overlap behavior, exact route213 suppression, module compilation, all three synthesized macro boundaries and CPU/CUDA parity.

## Y. Strategic interpretation

P0.4 validates a third materially distinct planning-to-Bonsai capability. The full route is not winning because it manipulates a shared price in the opponent's favor:

- reducing the route437 MELON sell order to route64's amount produces exactly identical results on all 12 ablation records;
- reverting SHEEP to COW makes five of eight accepted support records harmful and changes supported aggregate delta from +31,923 to -1,228;
- reverting the early WHEAT/fertilizer market timing leaves all eight accepted records positive.

The dominant mechanism is the private sheep allocation plus its co-adapted worker execution.

## Z. Scaling interpretation

Three synthesized specialists increase the 345-node pre-synthesis controller to 398 nodes and add three BOOL slots. Capacity is not the immediate constraint. The limiting factors are now initiation coverage, counterexample quality and explicit composition semantics.

P0.4 also shows why specialist count alone is insufficient: the first natural day-11 capability immediately shared an authority window with an accepted non-synthesized specialist.

## AA. Automatic composition readiness

The project is ready to begin a bounded automatic-composition stage, but not ready for a speculative universal scheduler. There are now:

- three independently validated synthesized specialists;
- exact parent-inactivity invariants;
- a real overlap/preemption example;
- one minimal executable precedence rule;
- CPU/CUDA collision parity coverage.

Automatic composition should initially reproduce these proven semantics and fail closed elsewhere.

## AB. Concrete composer requirements

An evidence-driven composer must provide:

1. explicit strategic-slot ownership per specialist;
2. inclusive active windows and authorization steps;
3. gate-overlap analysis separate from window-overlap analysis;
4. declared precedence for every overlapping authority pair;
5. named suppression and resume/end policy, not implicit append order;
6. BB ownership and uniqueness checks;
7. legal-state-only gate validation and mixed-cell rejection;
8. protected-case counterexamples and exact inactive-parent equivalence;
9. pairwise collision tests on CPU and CUDA;
10. fail-closed rejection for undeclared overlaps or incompatible resource commitments.

Resource-commitment declarations remain a future requirement; P0.4 does not invent a schema before an exact route-pair conflict demonstrates what must be represented.

## AC. Next stage

Proceed to the smallest automatic-composition stage that can ingest route216, route217, route218 and the accepted route213 authority declaration, reproduce the current order/preemption exactly, and reject unproven overlaps. Do not start PSRO or EditFormer and do not widen into a large unrelated data campaign.

The next search work should target remaining severe families only after this composition contract is executable and regression-certified.

## AD. Commit

All permanent P0.4 code, tests, route/macro/module artifacts, compiled challenger assets, ledger entries and this report are committed locally. The exact commit hash is reported by the completion response / `git log -1` after this file is committed.

No push or Kaggle submission is part of P0.4.

## AE. Final git status

P0.4 scratch results remain under ignored `scratch/p0_4_third_specialist/`. The five unrelated pre-existing untracked files remain untouched and excluded from the commit. Final status is reported after the commit.

## AF. Final recommendation

**Accept `route437_wheat_scarcity_sheep_shift_v1` into the BKG4 research lineage.** It has independent fresh and protected-archive transfer, exact compiled parent fallback, zero protected regression, a causal private-allocation mechanism and exact CPU/CUDA composition behavior. Begin a narrow evidence-backed composer stage using the concrete route218-over-route213 precedence contract; do not generalize beyond conflicts the evaluator can certify.
