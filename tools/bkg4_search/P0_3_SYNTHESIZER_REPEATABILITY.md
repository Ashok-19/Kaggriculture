# P0.3 SYNTHESIZER REPEATABILITY VALIDATED

Date: 2026-08-25
Team: ReCurSiON
Stage: second regret-discovered bounded macro + initiation-set CEGIS + two-specialist BKG4 composition

P0.3 validates repeatability of the deterministic Bonsai synthesizer. Starting from the accepted 363-node P0.2 route435 parent, regret-guided CUDA search found a second materially different strategic continuation, route410 around the day-5 boundary. The continuation was minimized to an 18-turn bounded macro, counterexample-gated using only legal current state, transferred to unseen both-seat trajectories, compiled into ordinary BKG4, composed with route435 without overlapping route authority, and certified with zero protected W→L.

The accepted research-lineage specialist is `route410_wheat_liquidity_expansion_v1`, strategic route217. It is not made the configured default runtime and is not submitted.

## A. Repository state

- Branch at P0.3 start: `main`.
- Start HEAD: `74f64510b2a2febcc2700b74b4b9c7bb9e8ce8e1`.
- The read-only strategic contract `tools/bkg4_search/BONSAI_FIRST_PLACE_MASTER_PLAN.md` was read completely and not modified.
- P0.3 originally began with the same five unrelated Aug-25 untracked files preserved through P0.1/P0.2: the paused dual-T4 corpus log plus the bakery/pizza/brunch module, route, distillation script and test. This continuation resumed an uncommitted P0.3 worktree on the same P0.2 HEAD; those five unrelated files remained untouched.
- All accepted P0.3 permanent code, module/route/macro assets, compiled challenger assets, ledgers and this report are committed locally at closure. No push or submission is part of this stage.

## B. Starting BKG4

The exact P0.2 parent is `route435_pizza_glut_rebalance_v1`:

- 363 nodes;
- 24 blackboard slots;
- 0 NodeState slots;
- max depth 11;
- static max visits 279;
- max live intents 0;
- controller semantic hash `f47ee7034e553e9a835486b283f4f819d857afdc95143a29d84cb07753b7c286`;
- combined identity `e2a2a0e91035c05c3183e691efee19441b6cee0d86506cbeb0e68a55768e53ce`;
- route-pack hash `ca28049cfbfcb57bbcfb8c74700881737ee3f6133e1ccd3d751bc85d1c415cee`.

A bounded recompilation test reproduces the P0.2 controller hash exactly. The existing synthesized route435 specialist remains route216, step156..203, with its established four-condition PIZZA/opponent-expansion/strawberry-glut gate.

## C. Current loss frontier

The loss frontier was re-evaluated against the current 363-node parent before planning.

- hard10: 9 losses / 1 win. P0.2 had solved the old route435 hard target; remaining losses include ReCurSiON -49,506, Arman -15,665/-15,530, Ryo -10,666/-10,588/-10,428/-9,043 and Arman -3,908/-3,908.
- screen61: 32 losses / 29 wins.
- shifted-current140: 41 losses / 99 wins.

The highest-value current paired losses included:

- Crop Dusta episode94674807: -35,110 in both seats;
- Subramanya episode94617605: -30,048 in both seats;
- Ryo episode94459964: -16,061/-17,341;
- Thomas and Izzoudine families with repeated -8k..-15k losses.

The 20 most severe shifted-current losses formed the initial regret queue. This target set was selected because it contained repeated both-seat severe deficits and several distinct strategy families, not because of opponent identity; identity was used only for offline evidence bookkeeping.

## D. Target hard-loss family

The selected second capability family emerged around the day-5 boundary, step120, in states where:

- shared WHEAT inventory is materially below its 10,000 baseline;
- both economies remain publicly liquidity constrained;
- MILK remains well supplied;
- the incumbent route64 is about to spend on a cow and four hires.

The most severe selected target pair was Crop Dusta episode94674807, both seats at -35,110 under P0.2.

Mechanically, the failure is capital allocation plus worker scheduling. The incumbent route64 buys a cow and four hires, then uses a comparatively less aggressive worker program. The discovered continuation instead sells available wheat/fertilizer, hires five workers, buys a sheep, and coordinates pasture/sheep placement, care/feed, watering and strawberry planting. When milk is already well supplied, the immediate cow commitment is less valuable than labor expansion and the alternative animal/production cycle.

This is distinct from route435, whose causal regime is a later step156 strawberry-glut production rebalance after opponent land/labor expansion.

## E. Counterfactual search

### Search control

A severity-balanced 20-case queue was searched first rather than repeating P0.1's broad 61-case grid.

Coarse checkpoints:

`72, 120, 168, 216, 264, 336, 432, 576`.

Coarse horizons:

`24, 48, 72, 96`.

The macro vocabulary was the existing 64-route coherent bank:

- current BKG4 strategic routes;
- public hybrid routes;
- source-backed top-10 teacher routes.

Candidates were route continuations, not arbitrary per-turn mutation strings. Before and after the fragment, the actual 363-node P0.2 BKG4 remained live.

The coarse grid contained 2,049 candidates including baseline and executed 40,980 exact CUDA games on the 20 severe targets. GPU kernel time was 8,142.58 ms, approximately 5,032.8 games/s, `bad=0`.

Refinement around route410 evaluated starts `108,114,120,126,132` with durations `12,18,24,30,36,48`: 4,340 exact CUDA games, 1,209.89 ms, approximately 3,587 games/s, `bad=0`.

Across the complete P0.3 terminal-game campaign, 55,580 exact CUDA games were executed, including 211 cheap baseline games repeated after initializing the persistent scratch output directory. The non-duplicated persisted scientific result set contains 55,369 games. Terminal-game CUDA kernel time was approximately 19.38 seconds. Complete legal-state captures added 745 episode rollouts and approximately 3.90 seconds of CUDA kernel time. Every accepted CUDA invocation reported `bad=0`.

No official Python engine and no heavy CPU game grid were used.

## F. Best planning discoveries

### route410 at step120

Source-backed route410 is from the archived rank-3 Subramanya N strategy. The source route has no clipped hand actions or market orders.

On the severe target20 grid, route410 step120 x24 produced repeated L→W:

- Crop Dusta episode94674807 seat0: -35,110 -> +3,911;
- same episode seat1: -35,110 -> +6,173;
- Izzoudine episode94669660 seat0: -8,859 -> +8,225;
- same episode seat1: -10,822 -> +2,750.

It also improved several other target losses without flipping them.

### route434 at step168 x72

A separate source-backed route434 continuation also showed strong teacher headroom:

- Crop Dusta episode94674807: both seats -35,110 -> +14,792/+14,146;
- tetsuya episode94508472: both seats -8,823 -> +2,128/+1,567.

However, route434's 168..239 window overlaps route435's 156..203 window, and its broad ungated transfer remained highly mixed. It was not selected once route410 demonstrated a cleaner compact initiation region and non-overlapping composition.

### route46 at step168 x72

Existing route46 flipped both seats of one Izzoudine trajectory but had weaker recurrence and broad transfer than the selected route410 basin. This remained search-control evidence rather than specialist #2.

## G. Rejected hindsight / unsafe discoveries

No P0.3 candidate was falsely promoted from an identical-complete-legal-state mixed-sign cell. Complete legal state was captured at both major candidate checkpoints, steps120 and168, and the final route410 gate selects no mixed exact legal-state cell.

Important rejected branches:

- route410 starts around 114 can produce more raw flips, but even on the support set one case becomes harmful; reject the earlier start.
- route410 starts around 126 are strongly unsafe on the same support regime; reject delayed initiation.
- route410 ungated is globally catastrophic across screen/historical/current panels, including many W→L; capability discovery is not global promotion.
- route434 is a strong teacher but was not selected because its ungated sign structure is mixed and its execution window overlaps the already accepted route435 macro.
- CEGIS iteration1 appeared perfect on the core 585 cases but failed on untouched validation40; this counterexample was incorporated rather than hidden.

Unlike P0.1's rejected step48 macro, P0.3 did not uncover a candidate proven to require hidden future information. The decisive route410 validation counterexample was legally separable.

## H. Selected capability

Final macro source: route410.

- start step: 120;
- duration: 18 turns;
- active window: 120..137 inclusive;
- source fragment SHA256: `954edb5db3adf1833c906dc9631b53c038a1e67c3fe818bb721f035b58ade0db`;
- compiled strategic route: 217;
- base route outside the fragment: route64.

The strategic motif is **day-5 wheat-drawdown liquidity-to-labor expansion with cow deferral**.

Relative to incumbent route64, the fragment immediately:

- sells wheat and fertilizer to release liquidity;
- hires five workers instead of four;
- does not make the incumbent immediate cow purchase;
- buys/places a sheep and builds the supporting pasture;
- coordinates the expanded workforce through care/feed, watering and strawberry planting;
- later sells additional fertilizer and rebuys wheat as part of the private production loop.

This is materially distinct from route435's later anti-strawberry-glut worker/market production rebalance. The public market conditions are selectors; the value comes from private capital allocation, labor capacity and animal/crop execution, not a symmetric shared-price movement.

## I. Representation

P0.3 reuses P0.2 `BoundedRouteMacro` unchanged for execution semantics:

1. evaluate authorization only at start step120;
2. latch one module-owned BOOL when the legal gate passes;
3. write route217 into the existing strategic route-override BB only at steps120..137;
4. stop writing at step138, automatically returning route authority to the P0.2 parent;
5. no reactivation later.

No BKG4 VM ABI, feature-registry ABI or NodeState change was needed.

One minimal composition helper was added to `bounded_macro.py`: `validate_nonoverlapping_macro_windows`. Until the future automatic composer has exact overlap evidence, two synthesized macros that can own the strategic route slot over overlapping step windows are rejected fail-closed. Route217 [120,137] and route216 [156,203] are disjoint and therefore compose without priority ambiguity.

## J. Initiation-state dataset

Core effect/counterexample corpus:

- screen61: 61;
- historical336: 336;
- current140: 140;
- fresh24: 24;
- sealed24: 24;
- total core: 585 cases.

Every core case was labeled with exact P0.2 baseline and route410 counterfactual outcome and had complete legal state captured at step120. Hidden-truth files emitted by the diagnostic capture binary were not used by synthesis.

Validation/holdout progression:

- untouched top-10 validation40: 40 cases; opened once after iteration1 was frozen and then converted into CEGIS counterexamples;
- untouched top-10 sealed40: 40 cases; opened only after iteration2 was frozen;
- untouched fresh949: 80 cases; opened only after iteration2 was frozen.

Final certification contains 745 panel records, 700 unique `(tape basename, seat, seed)` records after intentional overlap removal.

Final unique activation support:

- 12 activated records;
- six source trajectories/tapes;
- all six occur as both-seat pairs;
- 12 positive;
- 0 harmful;
- 6 L→W;
- 0 W→L.

## K. CEGIS iterations

### Iteration 0 — WHEAT drawdown only

Candidate:

`MARKET_INVENTORY[WHEAT] <= 9934`

Core result:

- 6 activations;
- 6 positive;
- 2 L→W;
- 0 harmful;
- 0 W→L;
- three support tapes.

Interpretation: material wheat drawdown is a real causal axis, but support was narrow. CEGIS expanded only to the nearest empirically useful boundary rather than adding arbitrary features.

A closure audit explicitly compared the simple wheat-only basin against the independent panels. `WHEAT <= 9934`, `<=9935`, and `<=9936` are behaviorally identical: validation40 contributes two harmful activations at -7,731 each, and sealed40 contributes two harmful W→L activations (worst -5,671). Across core + validation40 + sealed40 + fresh949, each simple gate gives 16 activations, 12 positive, 4 harmful, 6 L→W, **2 W→L**, and one selected mixed-sign exact legal-state cell. The simple rule therefore fails the primary safety objective and cannot be preferred merely for lower gate complexity.

### Iteration 1 — add opponent liquidity

Candidate:

`WHEAT inventory <= 9937`
`AND OPP_MONEY <= 120`.

Core result:

- 8 activations;
- 8 positive;
- 2 L→W;
- 0 harmful/W→L;
- four support tapes.

Untouched validation40 then produced the required counterexample: one both-seat top-10 trajectory at wheat9934 / opponent money86 was selected, but each seat lost 7,731 margin. Both remained wins, so W→L was still zero, but the safety objective rejects harmful activations even without outcome reversal.

The harmful validation state is not identical to the positive states in complete legal state; exact mixed-cell count remains zero. CEGIS therefore refined rather than rejecting the macro as future-dependent.

### Iteration 2 — add MILK supply state

Final clause:

`WHEAT inventory <= 9937`
`AND OPP_MONEY <= 120`
`AND MILK inventory >= 9990`.

The milk predicate is causally linked to the program: route410 skips the incumbent cow purchase and reallocates that capital to workers/sheep. When milk is already well supplied, cow commitment is less attractive.

Core + validation counterexample result:

- 6 activations;
- 6 positive;
- 2 L→W;
- 0 harmful;
- 0 W→L;
- worst delta +322;
- aggregate delta +132,402;
- three both-seat support tapes;
- zero selected mixed exact legal-state cells.

The gate was frozen here. No later holdout was used to retune it.

## L. Final gate

At exactly step120, authorize route217 for 18 turns iff:

`MARKET_INVENTORY[WHEAT] <= 9937`
`AND OPP_MONEY <= 120`
`AND MARKET_INVENTORY[MILK] >= 9990`.

Causal interpretation:

- WHEAT drawdown is evidence of staple-market pressure/demand;
- low public opponent money identifies a liquidity-constrained committed economy rather than a cash-rich state where the same expansion program is less compelling;
- well-supplied MILK says the parent route64's immediate cow commitment has weak scarcity value;
- step120 is the day-5 boundary where the donor and parent make the relevant irreversible capital/labor allocation.

The gate uses no team, opponent identity, submission, tape, episode, seed, replay or future shop information.

## M. Threshold / brittleness analysis

The final scalar thresholds were swept on the full core plus validation counterexample set.

### WHEAT inventory

With the other final predicates fixed:

- thresholds 9934, 9935 and 9936: four safe activations, two L→W;
- threshold 9937: six safe activations, two L→W;
- threshold 9938 and above: known harmful/W→L states enter immediately.

The wheat boundary is therefore empirically discrete rather than a broad arbitrary scalar plateau. The accepted value is the widest observed safe boundary; the neighboring unsafe regime is explicitly documented.

### Opponent money

With wheat<=9937 and milk>=9990:

- thresholds 80..110: two safe activations, no L→W;
- thresholds 115..150: identical six safe activations, two L→W.

120 is a round interior value inside this stable plateau.

### MILK inventory

With wheat<=9937 and opponent money<=120:

- thresholds 9978..9983 admit the validation harmful pair;
- thresholds 9984..9995 are behaviorally identical: six safe activations, two L→W;
- thresholds >=9996 deactivate all observed support.

9990 is a round interior threshold, not an exact observed fingerprint.

### Gate-necessity audit

Dropping any final axis reintroduces observed unsafe states. `WHEAT<=9937 AND MILK>=9990` without opponent liquidity gives eight core+validation hits, two harmful and two W→L (worst -24,919). `OPP_MONEY<=120 AND MILK>=9990` without wheat drawdown gives 28 hits, 20 harmful and eight W→L (worst -73,179). The three-predicate gate is therefore not retained for aggregate money or arbitrary complexity; every axis excludes a distinct observed counterexample class.

## N. Independent transfer

After iteration2 was frozen:

### sealed40

One previously unseen trajectory activated in both seats:

- seat0: +37,199 -> +52,533, delta +15,334;
- seat1: +38,685 -> +59,912, delta +21,227.

Result: 2/2 positive, 0 harmful, 0 W→L.

### fresh949

Two previously unseen episodes activated, both seats each:

- Crop Dusta episode94879309 seat0: -3,535 -> +23,686, delta +27,221, L→W;
- seat1: -3,416 -> +25,348, delta +28,764, L→W;
- Crop Dusta episode94767489 seat0: -5,891 -> +12,576, delta +18,467, L→W;
- seat1: -5,891 -> +12,576, delta +18,467, L→W.

Result: 4/4 positive, **4/4 L→W**, 0 harmful, 0 W→L.

These trajectories were not available to choose or refine the final gate.

## O. Composition with route435

The two planner-discovered synthesized specialists use the same existing strategic route-override BB but their execution windows are disjoint:

- route410 / route217: steps120..137;
- route435 / route216: steps156..203.

There is therefore an 18-step gap between route217 termination and route216 initiation. No nesting or preemption is required.

P0.3 adds an explicit fail-closed composition check: overlapping synthesized macro windows are rejected unless a future stage supplies deliberate precedence evidence. This prevents accidental append-order priority from becoming an implicit composer.

Across all 745 certification records:

- route410 final gate activations: 12;
- states also satisfying the route435 gate: 0;
- all 10 unique route435 support records are unchanged by P0.3;
- no route435 positive cohort loses margin or outcome.

This is the first exact coexistence proof for two planning-discovered synthesized Bonsai specialists.

## P. Full composed BKG4

Accepted research challenger: `route410_wheat_liquidity_expansion_v1`.

Compiled ordinary BKG4 identity:

- nodes: 380;
- BB slots: 25;
- state slots: 0;
- max depth: 11;
- static max visits: 295;
- max live intents: 0;
- controller semantic hash: `b2edd5dd76b936b6ea469009eb75b587531cef2f1be62939254b924f2e245edc`;
- combined identity: `494bdfb4aa785088c3ebc23ac1fa2b71db8f51a26f861bea6480a7c457322413`;
- route-pack hash: `9eedcfd2fcef415c7b0a10601ee61cf04b6fc7a6d7079cf9f0f0d0eb83f90cf5`;
- strategic route IDs include both 216 and217.

P0.3 adds only 17 controller nodes and one BOOL BB slot over P0.2. No ABI or NodeState capacity was consumed.

## Q. Regression

Exact composed CUDA certification used 745 records spanning the established protected panels and post-freeze holdouts.

Casewise semantics:

- 733 route410 gate misses: **733/733 exactly equal the P0.2 parent terminal result**;
- 12 route410 gate hits: **12/12 exactly equal the standalone route410 teacher intervention**;
- piecewise mismatches: 0;
- positive activations: 12/12;
- harmful activations: 0;
- L→W: 6;
- W→L: 0;
- worst activated delta: +322;
- aggregate activated delta: +261,882;
- CUDA faults: 0.

After freezing the gate, regenerating the macro/route/module and rebuilding the 380-node challenger, the final 745-case parent and candidate CUDA evaluations were rerun with the required strategic route materialization slot (`19`). Each rerun executed 1,490 exact games (baseline plus route410 teacher candidate), reported `bad=0`, and reproduced the persisted final parent/candidate terminal rows with **0/745 mismatches**. This guards the certification against stale compiled assets.

Panel breakdown:

- screen61: 0 activations, exact parent;
- historical336: 0 activations, exact parent;
- current140: 6 activations, 6 positive, 2 L→W, 0 harmful/W→L; parent 99W/41L -> candidate 101W/39L;
- fresh24: 0 activations;
- sealed24: 0 activations;
- validation40: 0 activations after CEGIS refinement;
- sealed40: 2 positive activations;
- fresh949: 4 positive activations, all 4 L→W.

Deduplicated by `(tape basename, seat, seed)`:

- 700 unique records;
- 12 activations / six tapes, all both-seat pairs;
- 12 positive;
- 0 harmful;
- 6 L→W;
- 0 W→L;
- worst +322;
- aggregate +261,882.

Protected-current-parent audit:

- 495 unique P0.2 parent wins;
- 0 W→L;
- **0 negative margin deltas among protected wins**;
- eight unique ReCurSiON self-history records, zero route410 activations and zero change;
- 10 unique route435 support records, zero route410 overlap and zero change.

## R. CPU/CUDA parity

The existing BKG4 v4 C++ and CUDA differential runners were exercised against the final 380-node controller across steps119..205 with legal decision-surface features sufficient to authorize both synthesized macros.

- records: 87;
- CPU/CUDA mismatches: 0;
- route217 active exactly at steps120..137;
- route217 absent from step138 onward;
- route216 active exactly at steps156..203;
- route216 absent at step204.

This proves the final two-macro route-authorization sequence has ordinary matching CPU C++ and CUDA BKG4 semantics.

## S. Tests

Focused implementation tests cover:

- bounded macro window/latch/fallback semantics;
- identity-free deterministic representation;
- harmful-gate CEGIS rejection;
- protected W→L lexicographic priority;
- exact mixed-state fail-closed behavior;
- route435 source-fragment and final gate preservation;
- exact P0.2 parent recompilation identity;
- route410 source-fragment and final gate;
- threshold-boundary semantics;
- explicit synthesized-window overlap rejection;
- CUDA result parsing;
- one-macro and two-macro CPU/CUDA decision parity.

The final focused P0.2/P0.3 suite passes **22/22**. Adjacent P0.1 regret-parser and contrastive-goal tests pass **13/13**, for 35 passing closure tests total.

## T. Strategic interpretation

**YES.** The planning -> coherent bounded macro -> public-state CEGIS -> deterministic BKG4 pipeline has now worked for more than one materially distinct strategic capability.

P0.2 synthesized a step156 strawberry-glut production rebalance. P0.3 independently discovered and synthesized a step120 wheat-drawdown liquidity/labor expansion that changes animal commitment and worker scheduling. Each has independent both-seat transfer, each compiles to ordinary BKG4, and the two coexist without protected regression.

This is stronger evidence than two syntactic route patches: the initiation regimes, timing, economic mechanism and action motifs are different.

A targeted 12-record component ablation also rejects a shared-market-only explanation. The full route410 continuation is 12/12 positive with 6 L→W and 0 W→L. Keeping route410 market orders but route64 worker/unit behavior produces 4 harmful activations and 2 W→L (worst -56,765); keeping route410 worker/production behavior but route64 market orders produces 10 harmful activations and no L→W (worst -12,505). The accepted gain requires the coordinated private capital/labor/animal program.

## U. Scaling implications

The first two synthesized specialists increase the historical 345-node parent to 380 nodes: +35 nodes and +2 BB slots total.

This supports the architectural thesis that thousands of BKG4 nodes can be populated by many compact validated specialists without tree capacity being the immediate bottleneck. It does **not** prove that growth remains linear or conflict-free to 1K, 5K or 10K+ nodes. At larger repertoire sizes, route-authority overlap, BB ownership, resource conflicts, initiation interactions, hot-path cost and regression scheduling will become material.

Evidence now supports continued Phase C specialist generation. Tree capacity remains abundant; validated capability discovery and composition semantics remain the limiting resources.

## V. Composition lessons

P0.3 contributes three concrete future-composer requirements:

1. **window conflict must be explicit** — synthesized route macros sharing a strategic route slot cannot rely on append order when their active windows overlap;
2. **gate overlap and window overlap are different questions** — P0.3 observed zero dual-gate activations, but the implementation still validates execution-window compatibility independently;
3. **parent-exact inactivity is a powerful composition invariant** — every inactive new specialist should reproduce the previous composed parent exactly, making conflict localization straightforward.

P0.3's two synthesized macros are deliberately easy to compose because their windows are disjoint. Automatic composition is not yet justified as the main stage solely from this benign case.

## W. Next stage recommendation

Synthesize **specialist #3** before making automatic composition infrastructure the main task, unless the next highest-value natural capability itself overlaps route216/217 or shares another scarce runtime resource.

The master plan says automatic composition becomes priority after several independently validated macro specialists exist. Two specialists now prove repeatability, but composition has only been tested in a disjoint-window case. A third distinct capability should either provide one more clean repeat or, preferably if evidence naturally leads there, expose the first genuine overlap/preemption conflict. That would give the automatic-composer stage a concrete contract rather than a speculative one.

Do not start PSRO, corpus generation or EditFormer from this result.

## X. Commit

The accepted P0.3 implementation is committed locally after final review. The exact commit hash is reported by the task-completion response / `git log -1` after this file is committed.

No push or Kaggle submission is part of P0.3.

## Y. Final git status

P0.3 permanent code, generated route/macro/module assets, compiled challenger assets, champion/experiment ledgers and this report are intended to be committed. P0.3 raw results remain under ignored `scratch/p0_3_repeatability/`.

The five unrelated pre-existing Aug-25 untracked files remain deliberately untouched and excluded.

## Z. Final recommendation

**Accept route410_wheat_liquidity_expansion_v1 into the BKG4 research lineage and continue one more regret-search -> bounded-macro -> CEGIS cycle.** P0.3 demonstrates that the deterministic Bonsai synthesizer is repeatable across a second strategically distinct capability, with strong post-freeze both-seat L→W transfer, exact parent fallback, zero protected W→L, and exact CPU/CUDA two-specialist semantics. The next experiment should seek specialist #3 and use any naturally occurring overlap/resource conflict to define the eventual automatic composition system from evidence rather than abstraction-first design.
