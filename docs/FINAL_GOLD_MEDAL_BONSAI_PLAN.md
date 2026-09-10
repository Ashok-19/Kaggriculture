# Final Gold-Medal Bonsai Plan

## Executive verdict

**Selected architecture: replay-seeded CUDA population-based production-route synthesis, with a small live best-response league, compiled into a BKG4 route portfolio and a bounded reactive executor (Option B).**

**THIS IS THE PLAN WE SHOULD EXECUTE.**

The teacher remains an expensive offline search system, not a neural policy and not the Bonsai tree. Public replays now supply a small, deduplicated set of production-job examples, timing priors and economic challenges instead of requiring all useful job content to be hand-designed. Search evolves their dependent full-season schedules, evaluates both agents live through the final actionable turn, and retains economically different successful routes. BKG4 executes the resulting route data, chooses among a few compatible continuations using legal observations, and handles bounded recovery. No replay lookup, planner or neural network ships.

The immediate starting executor and deployment reference are `shop_router_0909/plus`, verified at **85 nodes, 2 blackboard slots, 0 node-state slots, 3,224 controller bytes**, with **13 x 719 route records** in an **878,770-byte** pack. The direct Shop Router and Gold families are founding strategy material and comparison opponents, not sacred parents. The authoritative champion remains the different asset in `assets/champion/`; this document does not promote anything.

The strongest diagnosis is **a mismatch between the improvement representation and the competitive bottleneck, amplified by narrow evaluation**. Corrected CCPR searched gates and durations around four fixed mechanical interventions against only two closely related internal opponents. It improved proposal efficiency but could not invent a new production economy. Its best fresh-DEV improvement was +2,100 aggregate margin over 32 games, or +65.625 per game, with ten tie-to-win conversions. That is real, useful repair evidence, not evidence of a new leaderboard-strength strategy.

The decisive external clue is that a strong public Shop Router author describes recombining daily and three-day strategy sections and conducting approximately 21 million candidate-game comparisons. The policy we successfully ported is already the product of substantial route search. The next step is to own that discovery capability, while improving on its acknowledged fixed-opponent-tape limitation.[^1]

The primary investment is therefore a mutable route/job representation and efficient live CUDA continuations, not another specialist-gate campaign. Use stochastic proposals with reproducible experiment seeds; permit temporary and ordinary matchup regressions during discovery; select on diverse-league win utility and downside risk. Preserve exact simulator semantics, legal observations, deployment correctness, and catastrophic-failure protection without making every historical game an immutable strategic constraint.

**Single next experiment:** 128 replay-informed production proposals versus 128 uninformed production proposals using the same job compiler, legal support and compute. Replace the old local-repair arm, rather than adding a third arm. Use 32 discovery roots, full-horizon continuations, independent BUILD confirmation, one frozen winner per arm, and fresh internal/external evaluation. Cap it at 12,000 native full-game equivalents, two GPU-hours, two CPU-hours for independent Python opponents, and four engineering days. Mine at most 24 job records before this test; do not build a large strategy library or learned proposal model first.

The expected major upside is a stronger production economy across many matchups, rather than more wins against near-identical ancestors. The main risk is that useful new schedules remain hard to generate or fail when live market responses change. Top-10 status and a gold medal are objectives, not outcomes established by current evidence.

## Evidence scope and repository identity

The evidence cutoff is **2026-09-10**. Labels have the following meanings:

- **REPOSITORY FACT:** verified source, asset, or recorded experiment in the audited checkout; historical results were not rerun unless explicitly stated.
- **EXTERNAL EVIDENCE:** public engine/source, paper, implementation, or a clearly attributed public measurement.
- **INFERENCE:** a conclusion supported by evidence but not directly established by a controlled experiment.
- **PROPOSAL:** the selected design, budget, threshold, or prediction; not a measured result.

The audited branch was clean `main`, at **`7331794590efac169c06cf4c26b6c0625ce0ecd4`**, commit subject `Validate trajectory-diverse CCPR against independent control`. All identities and historical comparisons below refer to that input commit. The documentation commit containing this report is necessarily a later HEAD.

The focused replay-archive continuation started from clean `main` at **`dffa9b4013df653b8d7409027c1bc36f63bd420a`**. It changes the proposal-information source and pilot comparison, not the audited production assets or historical experimental conclusions. Its bounded dataset measurements are recorded in [the replay evidence manifest](../workspace/research/top10_replay_archive/evidence.json).

### Actual policy assets

| Policy | Controller nodes / BB / node state | Controller bytes | Instruction budget / depth | Interpretation |
| --- | ---: | ---: | ---: | --- |
| Authoritative `assets/champion/controller.bkgc` | 398 / 26 / 0 | 15,798 | 312 / 11 | Repository champion, not the isolated Shop parent |
| Direct `shop_router_0909/reference_port` | 61 / 1 / 0 | 2,376 | 64 / 7 | Exact public-policy port |
| `shop_router_0909/plus` | **85 / 2 / 0** | **3,224** | **83 / 7** | Strong current isolated deployment reference |
| `gold_v25` | 23 / 3 / 0 | 1,028 | 64 / 5 | Older direct public-derived route policy |
| Gold Plus recovery V8 | 23-node base + 16-node specialist; 4 total BB, 0 node state | 1,028 + 758 | Separate composed components | Actual accepted family is `BKG4_GOLD_V25_PLUS_RECOVERY_V8`, not an early README version |

Controller sizes were obtained by loading the compiled files with `kaggriculture_bonsai.runtime.bonsai_kg_v4.load_compiled`, not by counting JSON lines. Gold Plus also carries a 488-byte/player native recovery overlay and a 12,984-byte derived schedule seed-requirement cache. The latter refers to crop seeds, not hidden episode RNG information.

| Route data | Routes x actionable turns | Records | Bytes |
| --- | ---: | ---: | ---: |
| Shop Router | 13 x 719 | 9,347 | 878,770 |
| Shop arity metadata | 13 x 719 x 2 bytes | 9,347 entries | 18,694 |
| Gold V25 | 4 x 719 | 2,876 | 270,424 |
| Champion current pack | 18 x 719 | 12,942 | 1,216,740 |
| Champion legacy pack | 11 x 719 | 7,909 | 743,582 |

Important identity hashes:

```text
authoritative champion controller SHA256
36e5e83e5240d2371f19f0006107990e0be44e5f6de9d31ec8f9c2c59d41a5b2

Shop Plus controller SHA256
84bbfe9a32ccf4d202664500b28bba787dab18cb18180c539d70f1890b1cc450
Shop Plus controller semantic hash
2207e9f02c80e3b5971745426897a78cdfad16abf4d7694b716dfbc9613dd03b

direct Shop controller SHA256
b2d784244cf6c3b15979540da1be6f918dc48e1dbc38ba76314a4db0fdd22c97

Shop route pack SHA256
f711680c0c4514be70c97e64491f969ad31d79edd6a639b42cf950b98eebe696
Shop arity SHA256
5e8305647ac58067267515699c0dc15371e7d738d299512bfd26615ebfb882af

Gold V25 controller SHA256
842e04c8ada511806be0321f7b52408b8ce4a4a9fd33d753c9dcf5173b715f3c
Gold V25 route pack SHA256
27aa7e818348b9d8c7ce6bf45c8f83736fd5933e407a8a09956071864b8b1f22
```

**Metadata discrepancy:** `workspace/shop_router_0909/active_candidate.json` records `gate_sha256` beginning `9cedb97b`, which is not the actual frozen controller hash above. Its semantic hash and 85/2/0 footprint agree. The packaged controller and its `assets/identity.json` both match `84bbfe9a...`. This is provenance drift, not evidence of an action mismatch. Future experiments must pin actual bytes and runtime source, not rely on that stale field. No metadata was changed here.

### Existing submission artifacts

| Existing archive | Bytes | SHA256 |
| --- | ---: | --- |
| `workspace/gold_v25_submission/submission_gold_v25_bkg4.tar.gz` | 90,690 | `f963687d28c8be2154590e43e5302230a5865f8df8da55fca0b3e851c015e2b8` |
| `workspace/gold_v25_plus_submission/submission_gold_v25_plus_bkg4.tar.gz` | 101,248 | `3112783787fbfa0f18dca108306b01a66e62cd003f073a7cd356d5e71f62da98` |
| `workspace/shop_router_0909/plus_submission/submission_shop_router_0909_plus_bkg4.tar.gz` | 163,227 | `5fe4b5338f6f2b7d1ac1d6fa053a5a2b76defabf1afd0dea2c1aa8ce0566668b` |

The Shop archive contains the root entry point, isolated CPU shared library, runtime, controller, routes, arity, identity, license, and provenance. Archive existence and local canaries do **not** establish a remote submission, acceptance, rating, or rank.

**Access limits:** the live leaderboard and full competition-rules pages did not expose their substantive contents through the available public access. No authenticated ReCurSiON submission history or independently verified rank series was available. Public notebook titles, cached author ranks, and claimed scores are not current leaderboard measurements. The reported public-port-versus-descendant leaderboard reversal therefore remains an important observation to explain, not a causal result independently reproduced here. Existing SEALED contents were not consulted or repurposed.

## Why current improvement approach stalled

### What the experiments actually establish

| Evidence | Verified result | What it does not establish |
| --- | --- | --- |
| Phase 1 fork and repair | Exact snapshot/CPU/CUDA checks; 333,480 fork turns versus 1,069,872 naive turns, 68.83% saved on the matched comparison; four legalized fresh-DEV survivors | That the repair vocabulary can generate superior economies |
| Initial CCPR | 8 versus 5 safe useful BUILD patterns; 0 versus 1 DEV survivors; only three unique BUILD seeds | Generalization from many independent trajectories |
| Phase 2D | Whole-trajectory splitting and greater trajectory diversity improved fresh-DEV prediction from 4/10 to 8/10; one 7-node top-up gave +4,777 margin on 24 DEV contexts | Diversity alone caused the improvement; guard and ranking also changed |
| Corrected CCPR | 87 versus 136 proposal evaluations, 11 versus 7 safe useful BUILD candidates; first useful at 17 versus 27; one fresh-DEV mechanism in each arm | CCPR met its required two-family breadth gate, or CONTROL dominated every metric |
| Corrected CCPR best specialist | 431-478 top-up, tomato price >=69; 7 nodes, 1 BB, 400 bytes; +2,100 margin, 10 T->W, zero protected regressions on 32 DEV games | +2,100 per game, 10 independent strategic discoveries, or leaderboard gain |

Sources: [Phase 1](../workspace/self_improvement/phase1_fork_repair/README.md), [initial CCPR](../workspace/self_improvement/phase2_ccpr/README.md), [generalization diagnosis](../workspace/self_improvement/phase2d_generalization/README.md), [corrected CCPR](../workspace/self_improvement/phase2_ccpr_corrected/README.md).

Corrected CCPR had **48 unique BUILD seeds and 72 roots**, not 72 independent games. Each checkpoint had six discovery seeds and six different guard seeds. The opponents were still the parent and the parent with sell top-ups at steps 576-703. Discovery contained 12 ties, 11 wins, and one loss; guard contained 11 ties and 13 wins. That is useful near-mirror economic variation, but almost no hard losing-strategy distribution.

The 25 initial programs were six windows times four existing bodies, plus a reference. Refinements changed conditions and durations, with at most two predicates; they did not learn a new BODY, REACT state, production macro, or route. Consequently, increasing proposal intelligence could mostly decide **where an existing repair should run**, not what new strategy to execute.

### Strong deployable evidence is also concentrated

Shop Plus's recorded FINAL_TARGET panel is **384 games**: 128/0/0 versus its notebook donor, 124/0/4 versus Gold Plus, and 127/0/1 versus Gold V25. The pooled total is 379 wins and five losses. The 192-game DEV panel had been perfect. Gold Plus V8's 256/256 final result was against Gold V25, not 256 games sampled from a diverse competitive meta. These are strong local results with narrow opponent coverage. They are not medal estimates. [Shop provenance](../src/kaggriculture_bonsai/agents/shop_router_0909/PROVENANCE.md), [Gold Plus release](../workspace/gold_v25_plus/round8/release.json).

Shop Plus changes one observed shop-pair route choice and adds projected-shed sell top-up at 704-717. **Its route pack is unchanged.** It did not pass the then-required perfect-win final criterion and remains an isolated frozen composition. Conversely, rejecting it for five losses does not prove that relaxing that criterion would improve leaderboard rank.

The direct Shop port's recorded differential check covered 24 games / 17,256 acting steps with zero action/state/CPU/CUDA mismatches. Its official canary stopped at 42 of 64 matched cases when interrupted, so it must not be described as a completed 64-case certification. Shop Plus separately records 12 games / 8,628 native differential steps and a completed 32-case matched official canary (64 engine runs, 23,008 steps), with zero mismatches. This is strong evidence that the inherited executor is worth retaining, while leaving new mechanics and route regimes to be checked explicitly.

### Diagnosis ranked by evidential strength

| Explanation | Assessment | Evidence and implication |
| --- | --- | --- |
| Too-local repair vocabulary; production routes frozen | **Strongly supported** | No route/body generation in corrected CCPR; almost all gains are sell top-up or ordering. Search cannot discover decisions it cannot represent. |
| Opponent distribution too narrow and parent-centred | **Strongly supported** | Two near-identical internal members; historical finals mainly against donor relatives. More seeds do not create new strategic families. |
| Seed/root pseudoreplication | **Supported historically; partly corrected** | Original per-window samples drew from three seeds; whole-seed grouping improved matters. Corrected checkpoint sample size is still six seeds/panel. |
| Objective rewards tiny near-mirror tie flips | **Supported** | Ten T->W conversions can coexist with only +65.625 mean margin. Win utility is right for competition, but the challenge population determines what a win means. |
| Strict intermediate no-regression blocks useful exploration | **Supported as a mechanism** | Phase 2D's ungated body matched the guarded body's guard/DEV outcomes but was rejected for one -44 W->W discovery-margin change. Not proof of a net beneficial release tradeoff. |
| Threshold routing uses correlations instead of mechanisms | **Supported in specific cases** | Corrected 575 money-gated top-up turned +468 discovery into -937 guard; another ordering gate passed guard but lost a DEV tie. |
| Narrow self-play causes strategic collapse | **Plausible, not directly measured** | Population support was narrow by construction. There is no before/after strategic-entropy study proving an actual collapse. |
| Meta is fundamentally non-transitive | **Unestablished; public counter-evidence exists** | One public 14-agent study found no intransitive triple. Do not make Nash cycling the central explanation. |
| Seat aliasing or simulator errors explain the observed improvements | **Not supported by available checks** | Existing exact differential/canary evidence is substantial, although never a proof over every new action regime. |
| Public ports are empirically more robust than every descendant | **Unverified globally** | Local descendants beat their parents on recorded panels; independently verified remote longitudinal comparisons are missing. |
| Margin-insensitive optimization is the sole problem | **Rejected as a complete explanation** | Existing ranking already used margins and protected transitions. The issue is the value/coverage of those margins, not their total absence. |

**Conclusion:** the stall is best explained by optimizing a narrow residual problem after importing a strong economy, then using success on that residual problem as a proxy for competitive advancement. Simulator quality and repair efficiency are assets. They are not substitutes for economic innovation.

## What direct public ports teach us

### The useful policy is larger than its tree

The direct Shop port uses 61 controller nodes, 13 full tapes, daily weed-repair queues, projected inventory, next-turn sale advancement, order-slot-preserving sale subtraction, and final liquidation. Its BKG4 router observes the first two shops at step 144 and changes to a final continuation at 648. The controller is a selector over a much richer strategic object, not the policy's complete description. [Port implementation](../src/kaggriculture_bonsai/agents/shop_router_0909/reference_port/agent.py).

Record-level inspection found routes 3-12 identical to route 0 before step 144, but with **423-515 changed records** over their complete stored tapes. In contrast, raw route 1 first differs at step 70 and raw route 2 at 313. The controller does not execute every route from its beginning. Therefore the meaningful object is the **executed route composition plus overlay semantics**, not an independently replayed file selected arbitrarily mid-game.

All 13 stored Shop tapes contain **zero TOMATO PLANT requests**. Eleven have the same crop-request counts: 12 melon, 163 wheat, 33 strawberry, and 31 carrot. Routes 9 and 12 request 165 wheat and 28 carrot. These are raw schedule requests, not realized plantings or proof that tomato is profitable. They do show that a 13-route portfolio can contain much less production diversity than its route count suggests.

The replay archive extends this lesson: neither a large action-hash count nor a large collection of team names establishes many independent strategies. Mine realized production profiles first, then inspect the service and financing sequences behind distinctive profiles. A public trajectory can supply a missing job idea without supplying a deployable policy or evidence that the idea beats Shop Plus.

Animal purchase schedules differ more. Route 0 requests 8 cows, 6 sheep, and 3 geese; route 3 requests 6 cows and 11 sheep; route 12 requests 4 cows and 14 sheep. Purchases after step 144 include decisions at 150, 169-178, 195-226, and 241-265. There is a concrete, accessible search space of future herd allocation and hundreds of turns of associated service, not merely endgame sales.

### Public strategy structures inspected

| Family/source | Actual strategic structure | Architectural lesson and limit |
| --- | --- | --- |
| yhay81 Shop Router 0908/0909 | Complete tapes, causal shop-conditioned continuations, daily repair, advanced sales, terminal liquidation. The 0908 author reports mining 1,326 histories / 1,313 unique sequences, recombining one-day and three-day sections, and roughly 21 million candidate-game comparisons. | **Direct evidence for route synthesis as a serious search space.** Author warns that recorded opponent tapes do not model adaptive live opponents. These are author-reported search counts, not independently rerun benchmarks.[^1][^2] |
| aurax7 Reactive Router / local Gold-derived family | Four-route-style backbone with reactive machinery for worker alignment, weeds, market timing, budget/room guards, dead stock, and terminal handling; settings must be distinguished from enabled branches. | Preserve the executor, not just its nominal tape. A public code title does not certify its latest strength.[^3] |
| Seyamalam current `main.py`, attributed Kaito Fukami V18 lineage | Multiple episode-derived schedules; public-state prototype selection for board policy, separate market-policy selection, late lead-based abstention, and recovery liquidation. | A compact selector can use causal state and separate execution from trading. Shared episode ancestry prevents counting these as wholly independent strategies.[^4] |
| COK-ZhangZiliang current V10 code | Alternative herd schedules, early shop routing, public step-72 composition signature, confidence-gated market front-running, short recovery lookahead. | Observable opponent adaptation is feasible. Exact composition fingerprints may be brittle; use production regimes rather than competitor recognition.[^5] |
| lonespear V11 | Reactive task generation and assignment for production, feed, care, cash reserves, land, workers, and crop recovery rather than a large embedded tape pack. | A genuinely different behavioural challenge; greater runtime sophistication is not evidence of stronger play. README score claims are historical and low relative to claimed leading agents.[^6] |
| GzmCR reactive scheduler | Visible-opponent supply estimation, marginal herd selection, task scheduling and capacity checks. Its embedded market curves differ from the audited current engine. | Useful design evidence for opponent-state features; **not** a trusted current economic oracle or strong baseline without version reconciliation.[^7] |

The public source structures above were inspected; arbitrary notebook cells or downloaded agents were not executed. Attribution and compatible licenses remain requirements for any later ingestion.

### Meta and private-agent inference

A public report by sobameshi describes a round robin of 14 public implementations, 96 fresh seeds, and both seats. It reports no intransitive triple, 86 of 91 chronological pairs favouring the newer implementation, and rapid replacement of old replay openings. This is informative first-hand evidence but not an independently reproduced matrix or a sample of all private leaders.[^8]

**INFERENCE:** much of the public meta may be a ladder of increasingly productive economies, with tactical adaptation deciding close mirrors. Thus a robust, generally stronger schedule is a better first target than an elaborate anti-cycle router. Private leaders may discover more productive economies through search, hand-designed scheduling, learned teachers, or combinations; the visible evidence does not identify which training algorithm they use. Implement capabilities that produce and withstand new economies, not assumptions about private source code.

**Answer to the route question:** yes, the evidence supports moving the unit of improvement from tiny local repairs to coordinated full-season production alternatives. It does not imply deleting reactive repairs or assuming that arbitrary full-tape mutation will work.

## Top-10 Replay Archive Strategy

**Decision: Option B, replay-seeded route synthesis. YES, this materially improves the plan's proposal source and first experiment; it does not establish a measured competitive gain.** Keep the CUDA truth oracle, live league, small archive and BKG4 deployment. Do not build a replay-first mega-library or make imitation the primary teacher.

### Contents, provenance and limits

**EXTERNAL DATA, locally measured:** `ashok205/kaggriculture-top10-replay-archive`, dataset ID **11731942**, public **version 19**, CC0, updated **2026-09-09 23:19:37.640 UTC**, contains **45 files / 2,213,153,054 downloaded bytes** (2.213 GB, 2.061 GiB). There are 41 daily ZSTD-compressed Parquet replay shards, July 30 through **September 8**, plus four metadata files. All listed file sizes and the 44 payload hashes in the archive manifest were verified; the manifest is separately pinned. The official index version 42 already lists September 9, but this archive version does not. Coverage begins with the official published index, not a proven complete competition history.[^39][^40][^41]

The entire episode-ID index was read, not every full replay. It contains **16,168 distinct episodes and distinct recorded replay hashes**, with **zero duplicate IDs across shards**. `episodes.parquet` has **22,731 participant-selection rows**: 6,563 extra rows, or **28.87%**, are intentional second selected-team links, not duplicate replay bodies. There are **814 participant names**, **191 selected names**, 410 selected team-days, **5,171 unordered / 7,130 ordered name-pair matchups**. The most repeated pairing has 172 episodes. Names are not submission IDs or policy counts.

The 2.2 GB size is misleading for processing cost: replay column chunks expand to **495,925,263,181 bytes** (495.9 GB / 461.9 GiB, Parquet metadata accounting). They store one `episode_id:int64, replay_json:large_string` row per row group, compressed with ZSTD level 10. Read IDs/footers globally and selected row groups individually. Do not load all decoded JSON or build a feature-store service.

Collector source explains an important misnomer: each day's ten teams are chosen by **maximum observed pair-average rating**, with additional score evidence and name tie-breaking, not verified leaderboard positions. Both players receive the same pair-level score evidence. The source official daily archives already select completed games by pair-average rating under a 20-GiB/day cap. This archive then keeps games involving its ten proxy-selected teams. It retained 16,168 of 28,864 source-indexed episodes; one source header was unavailable, zero selected bodies were unavailable. The collector has no winner-only filter. Versions are cumulative copies plus new daily shards; never concatenate all 19 versions.[^40][^41]

**Bounded body audit:** deterministically select the minimum SHA256 of `gold-replay-audit-v1:<episode_id>` per selected team-day, then deduplicate the union. This yields **348 episodes, all 41 dates, 696 player trajectories**, including 527 trajectories belonging to that day's selected names. All 191 selected names are represented. Raw sampled bodies total **10,605,340,258 bytes**, decoded one game at a time and hash-checked. The sample has 291 selected-player wins, 231 losses and five ties; it is not a winners-only corpus. All 348 sampled games have 720 steps and DONE/DONE status. **Archive-wide timeout/error prevalence and seed uniqueness are unmeasured**, not zero by inference from this sample.

| Available in inspected bodies | Missing or not established |
| --- | --- |
| Both 719-action streams: farmer, ordered hands, ordered market lists; step rewards/statuses and final rewards | Source code, submission IDs, numeric team IDs, exact collection-time leaderboard rank, individual agent rating |
| Both public farm boards, land, cash, unit positions; plant/animal fields; markets, prices, shops | Agent memory, route ID, causal branch rationale, random policy state, native snapshot/controller state |
| Each seat's private shed, seeds and carried inventories; episode config; `info.EpisodeId`, names, replay seed | Opponent private state is **not live-observable**; replay seed is **not live-observable**; precise game timestamp absent from inspected schema |
| `module_version`, replay schema/version; daily collection date in index | Engine source hash; explicit default market curves when `marketParams={}`; proof that all archive versions share one body schema |

All sampled replay schemas are 1; archive metadata schema is 3. Body package versions are **1.32.2:34, 1.32.3:12, 1.32.4:16, 1.32.5:11, 1.32.6:64, 1.32.7:211**. Early configurations use town-center selling interval 12 versus current 24. Match both package/configuration and the audited engine semantics before interpreting old economics. Version/date alone does not establish an innovation's value.

**Legality:** public CC0 distribution and official publication support offline analysis; the full competition rules and linked host statements remain inaccessible through available public access. The previously cited participant reports host permission for replay-based submissions, but that is not independently retrieved host text.[^8] The plan permits public-data analysis, job proposals and legal-state opponent-regime modelling subject to the existing pre-release rules check. It does not authorize hidden/private access. Strip opponent-private fields, IDs, names, seed and future shops from all selector/proposal-model runtime features. Replay seeds may be offline provenance or reproduction inputs only. No live seed dependency or hidden-information route lookup is proposed.

### Measured strategic information

Behavioural grouping uses **realized** `(cow, sheep, goose, owned tiles, hands)` at observations 144/288/432/576, not team identity. Coarse regimes use the herd at 576: a species is dominant at >=60% of animal count; otherwise mixed; zero animals is separate. These are descriptive fingerprints, **not inferred independent policies**. Worker counts at day boundaries are zero because hands expire; midday/max-worker profiles are necessary for labour diversity.

| Selected-player sample | Jul 30-Aug 12 | Aug 13-26 | Aug 27-Sep 8 |
| --- | ---: | ---: | ---: |
| Trajectories | 164 | 187 | 176 |
| Distinct four-checkpoint asset profiles | 57 | 72 | 106 |
| Effective profile count, exp(Shannon entropy) | 19.0 | 27.4 | 63.8 |
| Most frequent herd at 576 | 8 cows/6 sheep: 76 | 10 cows/4 sheep: 53 | 9 cows/5 sheep: 26 |
| Any geese at 576 | 2 | 14 | 52 |
| Median requested wheat / melon / carrot plantings | 72 / 21 / 0 | 127 / 14 / 6 | 164 / 12 / 15 |

Across 527 selected trajectories there are **521 full-action hashes**, **450 farm-only hashes**, **129 first-24-action hashes**, and **224 asset profiles** (effective count 85.1). Across all 696 sampled players, 682 action sequences are unique: 14 extra exact copies, **2.01%**. Yet the five selected-player coarse regimes have effective count only **2.93**: 248 cow-heavy, 195 mixed, 76 sheep-heavy, six no-herd and two goose-heavy. Thus many action strings coexist with a few broad economic modes and substantial within-mode variation. Neither 521 nor 224 is a defensible count of independent strategic families.

The early 8-cow/6-sheep profile persists in sampled games through August 23 but has no late-period selected example. The goose increase also appears within package 1.32.7: 14/159 selected trajectories on August 15-26 versus 52/176 afterward. Late geese and sheep-heavy alternatives broaden supply patterns. Requested plantings and observed profiles identify mutation hypotheses, not executed yields or causal score gains. Changed engine, shops, opponents, eligibility and sampling all confound temporal comparisons. The data can mine **behavioural deltas**, not certify that B improved over A. Dates of first observation are upper bounds on when a strategy existed, not invention dates or proven diffusion chains.

### What can actually be recovered

**Routes: partially.** Every sampled realized 719-action sequence is recoverable, but not the generating policy. Align action `steps[t+1][seat].action` with pre-action observation `steps[t]`, excluding the initial dummy action; restore framework-shared fields before live feature comparison. Keep arity/order/zero orders intact. Historical weed responses and future route switches are not reusable unconditional actions. The first pilot translates jobs into the existing executor; it does not blindly replay complete archived tapes.

**Production jobs: yes, as observed examples with inferred contracts awaiting verification.** A detailed 41-game / 82-player pass follows tile identities and placement/planting days, actual hires, land changes and yield fields. Episode 106622723, seat 0, has a sheep at row 4/column 7 first present in observation 160, day 6, with first positive stored yield at observation 288. The same episode has 3-cow/15-sheep versus 1-cow/21-sheep economies at 576. This is concrete late-game material beyond the narrow inherited herd choices, not proof either beats Shop Plus. Crop `yield_units` can already be positive at planting, so it is not a maturity flag; derive harvest readiness from engine semantics and successful state transitions.

Mine boundaries around acquisition -> placement -> repeated service -> harvest -> cargo -> sale, with daily worker resets and land/financing dependencies. A record contains legal entry-state requirements, worker/site roles, ordered actions, time/material/cash requirements, persistent assets, measured state effects and an exit contract. Report requested orders separately from realized asset changes. Exact revenue attribution across simultaneous market interactions needs engine-aware accounting or subsequent CUDA ablation; a cash difference alone is not a job's profit. One narrow extractor can annotate the two existing job types; it cannot automatically infer all missing prerequisites or eliminate compiler design.

**Deltas and disagreement sets: yes, with limitations.** In the detailed pass, role-normalized public/own-private state contracts at steps 72/144/216 produce 66/73/77 groups among 82 trajectories; 12/9/5 groups repeat. Their next-72-turn investment requests disagree in 2/1/1 groups. These windows contain later observations and are not immediate action labels or causal preferences. A concrete step-144 matched-economy pair in episode 93191035 requests two cows versus two sheep at decision 168. Compare actual decision-time legal state, seat and feasible job closure before turning this into a counterfactual. Most exact noninitial states are unique; use contract-filtered retrieval, not claims of abundant matched-treatment data. Generate fresh live BUILD states for final CUDA comparison, never reconstruct an opponent's missing memory from the replay.

Public source remains a cross-check: the already inspected yhay81 and Kaito-lineage code establish tape/router/repair architectures, and their display-name families occur in the index. Display-name coincidence cannot map historical submissions to current source. No private source architecture or exact historical submission mapping is established here. Positive action/prefix and engine-matched source reproduction would be required before assigning ancestry; absence of a match proves little. Do not spend the pilot resolving every identity.

### Mining priorities and implementation choice

| Use | Value | Decision |
| --- | --- | --- |
| Lifecycle examples, dependency hints, timing/quantity priors | **HIGH** | Seed at most 24 auditable jobs, family-balanced; compile and mutate, do not copy labels blindly. |
| Economic regimes, saturation/close-loss challenges | **HIGH** | Convert into live challengers after the pilot; retain losses and rare economies, not just winners. |
| Route founders and temporal deltas | **MEDIUM** | Translate persistent-asset contracts; no immediate full-route import or causal ranking by date. |
| Disagreement retrieval | **MEDIUM** | Propose feasible alternatives from comparable legal states; CUDA supplies missing counterfactual returns. |
| Descriptor thresholds and proposal-frequency priors | **HIGH** | Freeze campaign thresholds on TRAINING only, cap clones and reserve >=20% uninformed exploration. |
| Selector examples / strategy embeddings | **MEDIUM** | Replay supplies legal state coverage, not the best-route label; train selectors on live counterfactual returns. |
| Raw-action BC / return-conditioned sequence policy | **LOW** | Joint-action compounding, clone bias, old engines and unavailable live features; not primary. |
| Offline-RL value targets, logged-win rewards as causal ranking | **REJECT** | Missing counterfactual support and changing opponent policies make confident action extrapolation unjustified. |
| Recorded opponent suffix as a live challenger; rank/seed identity routing | **REJECT** | Counterfactual inconsistency or unavailable runtime information. |

**Search-space reduction is plausible, not measured speedup.** In the late selected sample, 10th-90th percentile realized day-12 herd ranges are cows 4-9, sheep 3-11, geese 0-3; owned land is usually 75 tiles. These give practical centres, not hard limits: rare late examples reach 21 sheep and broader whole-sample extremes exist. Acquisition/service days, daily labour, observed shop conditions, reserve distributions and sale windows should be conditioned on entry contracts before weighting proposals. A toy independent box of 0-24 animals/species has 15,625 combinations versus 216 in these marginal ranges, about 72x fewer; **this is illustrative arithmetic, not an empirical feasible-space reduction or search acceleration**. Novel exploration and exact verification remain mandatory.

**Learned-model usefulness changed: yes, modestly at the macro-prior level; no change to the primary teacher.** There are at most 32,336 player trajectories and roughly 23.25 million action decisions if all games are full length, but far fewer independent economies. The sample supports retrieval/count-based priors in roughly 0.5-1 engineering day, with no learner GPU cost. A small state-to-job classifier or embedding could take 2-4 days plus approximately 0.5-2 GPU-hours *after* trustworthy macro labels exist; expected improvement is unmeasured. A raw sequence BC/offline-RL system still requires the expensive action/schema/causality work, not just a 2-GB download. Do not fund it before the simple prior wins. CompILE supports the concept of reusable trajectory segmentation; its public example is a toy sequence implementation, not our compiler. SPiRL supports skill-level priors rather than flat action priors, but its robotics results do not establish Kaggriculture gains.[^42][^43]

Choose one local streaming reader plus Parquet/JSONL summaries and a small JSON job list, with raw files ignored. **Option A** leaves useful demonstrated content unused; **Option C** delays falsification behind a large library and unsafe-composition risk; **Option D** adds learning before showing that labels/priors help. **Option B wins** because it changes only the information supplied to the existing search. Full-corpus body parsing is deferred unless a measured sampling gap blocks a valuable job. The fresh live two-arm experiment below, not archived wins, decides whether replay seeding stays.

## Engine mechanics findings

### Engine identity and scope

The installed official package is **`kaggle-environments==1.32.7`**. The inspected engine is `.venv/lib/python3.13/site-packages/kaggle_environments/envs/kaggriculture/kaggriculture.py`. Its SHA256 is:

```text
bc8a54879ef02c7ea64b8b333d6a976f0ea65c4949149d01f463f23bccee653e
```

The publicly fetched upstream `master` engine had identical bytes at the evidence cutoff. This checks the local engine against current public source; it does not prove the competition servers' pinned package. The host's balance-change discussion directs users to 1.32.7 and links the market-curve change.[^9][^10][^11]

Mechanics below are established from the Python interpreter and checked against corresponding C++/CUDA implementation paths. **CONFIRMED describes semantics, not a measured incremental score gain or formal competition-rules approval.** Existing parity evidence covers tested trajectories; unusual boundaries still need targeted differential tests before use.

### Turn and observation map

1. Both agents choose actions from the same preceding public game state and their own private state. Opponent farms, money, positions, crops/animals and their public fields are visible; opponent shed, seeds, carried inventory, pending orders, source identity, and episode seed are not.
2. The interpreter counts all supplied PLANT requests by crop for each player. If demand exceeds owned seeds, all that crop's PLANT requests are blocked. Even extra requests for nonexistent hands can enter this count.
3. Player 0's farmer and listed hands act in order, then player 1's. Farm operations do not directly compete for the same land. Own earlier worker operations can affect later workers through shed contents/capacity.
4. All unit actions precede market resolution. Newly purchased seeds/animals, hired workers, and purchased land cannot enable an earlier action in that same turn.
5. Only the first ten market list positions are processed. Market positions, including failed/zero/no-op entries, matter. At each position, hiring/land operations spend cash before per-unit trading; the two players' product quotes use the shared pre-commit inventory for that matching iteration, then commits occur.
6. Town consumption follows the market: unlocked shops consume at multiples of four steps and the centre at multiples of 24, including step 0. Prices are then relevant to subsequent actions; selling on the consumption turn does not get its later price update.
7. Plant decay follows trading. At `(step + 1) % 24 == 0`, daily refresh, weeds, automatic inventory deposits, worker reset and possibly shop unlock happen.
8. The next observation exposes updated day/hour, farms, market and shops. The framework supplies shared `step` to both live agents.
9. With 720 recorded states, actionable steps are **0 through 718**: **719 transitions**. Step 719 is not another agent opportunity. No final hour-23 daily processing occurs after action 718. Final reward is cash, not total asset value.

Source anchors: Python `_initialize`, `_apply_unit_action`, `_process_market`, `_town_consume`, `_daily_refresh_plants`, `_daily_refresh_animals`, `_end_of_day`, `interpreter`; native [C++ simulator](../native/cpp/include/sim.hpp) and [CUDA simulator](../native/cuda/include/sim.cuh).

### Economic and biological boundaries

| System | Source-established boundary | Planning consequence |
| --- | --- | --- |
| Market prices | Product-specific inventory curves, integer rounding, minimum price 1; initial inventory 10,000. Current carrot/tomato/egg scarcity uses hinge curves; wool/melon glut is nonlinear. | Quote actual marginal units. Average price times quantity is not exact revenue. Old price models can silently choose the wrong economy. |
| Buys | Only wheat/fertilizer product buys; seed/animal buys are distinct operations. Product buys are money/capacity constrained, not limited by positive town inventory. Buy uses post-buy inventory price. | Do not infer an infinite arbitrage loop from a sale quote. Feed purchasing can be an intentional part of the economy. |
| Shops | Eight shop types; duplicates drawn with replacement; at most eight instances; unlocks visible at steps 72, 144, ..., 576. Each copy consumes independently. Single-product shops pull two units of their product. | Count all observed copies, not just whether a shop exists or the first two names. Future shops are unknown. |
| Fertilizer and melon markets | Fertilizer has no town consumption sink; melon only has centre consumption. | Herd/crop scale can create severe glut. Marginal production and disposal timing matter more than nominal base prices. |
| Shed | Capacity 100; seeds are separate. DROP discards carried overflow; item insertion and worker order determine which goods fit. PLACE can transfer a chosen partial quantity while retaining excess. | Avoid waste before it happens; sorting SELL orders alone cannot recover goods already destroyed by DROP. |
| Access and movement | Four centre-adjacent access tiles work even while their quadrants are locked. Locked tiles can be traversed; workers do not collide. | Route geometry is much simpler than a collision-aware RTS planner; do not buy land merely to reach the shed. |
| Overnight | All workers' inventories deposit automatically, with overflow discarded; hands disappear, hire count resets, farmer returns to default spawn. Plants/animals/land persist. | Overnight is a useful movement reset, **not** an arbitrary strategy-switch reset. Cash/cargo and production contracts still have to match. |
| Hiring | Daily Fibonacci costs beginning 1, 1, 2, 3, 5; deterministic least-occupied spawn choice; new worker usable next turn. | Search cheap early hires, task packing and next-turn placement rather than a fixed hand count. |
| Wheat/carrot/melon | Different seed costs, ages and harvest windows; water bonuses occur in specified later-age windows, with fertilizer doubling that bonus. | Evaluate sowing, water, fertilizer, harvest and financing together, not as isolated actions. |
| Tomato/strawberry | Delayed first yield, recurring production limited to four production events, held-yield capacity four; watering/fertilizer and harvest cadence affect realized total. | They are not infinite income sources. A new crop macro must cover its full lifecycle and end-of-season timing. |
| Water and weeds | New plants begin with one consecutive unwatered count; failing planting-day water kills them at the next refresh. Later plants can tolerate one missed day. Empty owned tiles can spawn weeds; expired plants decay on the interpreter's step schedule. | A shifted PLANT without shifted WATER is not a valid economic experiment. Recovery needs to preserve deadlines, not simply queue indefinitely. |
| Animals | Goose/cow/sheep first-yield delays 4/8/6 days, intervals 1/2/3, held-yield caps 4/6/6. One unfed day is tolerated; two consecutive unfed days cause escape. | Long-horizon purchases need enough remaining production days and a viable feed/service schedule. |
| Care | Existing pending care is consumed on a fed production day, then today's fed-and-cared action contributes to later production. | Care on the yield day does not increase the yield just produced. Care schedules have delayed credit. |
| Animal fertilizer | Availability is refreshed as a boolean, not an accumulating stockpile, including after a tolerated unfed day. | Collection cadence matters; no unlimited hidden accumulation. |
| Termination | Cash only; unsold output, seeds, structures and animals receive no residual valuation. Last action precedes the final daily deposit/production that a naive 30-day schedule might expect. | Compile a route-specific terminal liquidation tail and forbid investments that only mature after the horizon. |

### RNG, seeding and state aliasing

At daily refresh the engine constructs `random.Random((seed * 1_000_003) ^ day)`. The stream is consumed by eligible empty tiles in player/row/column order, then by shop selection. Occupied tiles do not consume the same weed draw as empty tiles. A route changing occupancy can therefore change later shop selection and the other seat's weed realization **under the same initial seed**.

This is a major evaluation fact, not a recommended hidden-information exploit. Common seeds pair candidate and parent starting conditions; they do not guarantee identical exogenous future shop sequences after different actions. A checkpoint is reusable only for the same complete prefix state, including both live controllers. Teacher lookahead can simulate the hidden seed offline, but labels and runtime decisions must aggregate over independent seeds and use only the observation history available at the decision.

The official seed resolver removes the seed from configuration and records it in `env.info` for replays. A small local diagnostic using the official runner, `episodeSteps=5`, two PASS agents, and seed 123 confirmed: both live agents received steps 0-3; neither received a seed in observation/configuration; the seed existed in replay info. Raw `env.state[1].observation` lacked `step`, but the official runner restored shared fields. Therefore a public claim that seat 1 necessarily receives step 0 forever is **rejected for this runner**; a hand-written harness that passes raw state can create that bug.[^12]

Observable weed history might constrain a hypothetical RNG inference problem, but no practical seed-identification method, legal authorization, or robust gain is established. Another public discussion raises the same question without a verified host answer.[^13] Hidden seed recovery is excluded from the primary architecture. Reading replay-only seed data at runtime, process/environment internals, private opponent files, credentials, or hidden test information is explicitly rejected.

### Ranked mechanics opportunity table

`L` means ordinary game-action/public-observation use, subject to the competition's full rules and version check. `H` means host clarification required before any live use. `X` means excluded. Gains are qualitative incremental hypotheses unless an existing repair result is cited; no fabricated coin estimates are assigned.

| Rank | Mechanic / status | Why potentially profitable; estimated gain | Required observable inputs | Deterministic / seat-specific | CPU/CUDA support | Live legality | Complexity and decisive test |
| ---: | --- | --- | --- | --- | --- | --- | --- |
| 1 | **CONFIRMED:** productive-lifecycle and care timing | Reallocate hundreds of turns of output and labour; potentially high, unmeasured beyond existing routes | Own tiles/age/care/feed, cash, market, remaining turns | Exact conditional on state; not intrinsically seat-specific | Implemented in both engines | L | Medium. Full-season animal/crop job ablation with care-before-yield and final-horizon fixtures |
| 2 | **CONFIRMED:** shed overflow, partial PLACE, ordered PICKUP/DROP | Recover destroyed valuable output and protect feed; medium/high in overflow-heavy schedules | Own cargo/shed, positions, action order | Yes; own worker order | Implemented; boundary parity required | L | Low/medium. Capacity 99/100/101, mixed cargo insertion orders, earlier pickup freeing later deposit space |
| 3 | **CONFIRMED:** same-turn deposit then sale; next-turn purchase usability | Better liquidity and fewer missed production actions; medium; donor already exploits part | Own inventory/cash, intended worker actions and market slots | Yes; relative market orders matter | Implemented; donor parity covers normal paths | L | Low. Sale after DROP, buy then attempted same-turn plant/hire action, and affordability precedence |
| 4 | **CONFIRMED:** scarce-product curves and duplicate-shop demand | Different herd/crop economics and sale timing; potentially high, not proof tomato is best | Current inventories/prices, all shop counts, own/visible opponent production | Yes for known state; future shops uncertain | Current price logic in both engines | L | Medium. Per-unit quote boundaries, duplicate shops, matched live supply-shift opponents |
| 5 | **CONFIRMED:** 719-action horizon and no final daily settlement | Avoid stranded output and valueless late investment; meaningful but much already captured | Current step/day/hour, own production/cargo | Yes; both seats | Implemented; existing terminal repair | L | Low. Steps 695/704/717/718 and production nominally due at 719/720 |
| 6 | **CONFIRMED:** cheap daily hands and predictable spawn/reset | More productive worker-turns and cheaper logistics; potentially medium/high in new schedules | Own worker count/positions, cash, hire count, day/hour | Yes; own order | Implemented in both | L | Medium. HIRE slot order, spawn occupancy ties, first usable turn, end-of-day cancellation |
| 7 | **CONFIRMED:** weed delay and atomic planting | Prevent missing whole crop/feed chains; medium on affected trajectories | Own tiles, seeds, scheduled requests, worker count | Yes after observed weeds | Implemented; ordinary repair parity exists | L | Low/medium. Weed insertion before planting-day water; excess/nonexistent hand requests; short seed stock |
| 8 | **CONFIRMED:** order positions, no-op slots, marginal quotes | Avoid accidental adversarial market alignment; incremental close-match gains | Own stock and cash, public prices, previous visible opponent consequences | Yes for joint actions; opponent next action unknown | Implemented; test all slot boundaries | L | Low. Zero/failed slots versus compaction; split orders; both seats and buy/sell combinations |
| 9 | **PLAUSIBLE gain:** observe opponent production and adjust economic continuation | Avoid entering an already saturated product and forecast public supply; potentially medium/high | Public opponent tiles/ages/herd, land, cash, shop/market history | Deterministic classifier; not source identity | Generic features exist; Shop adapter needs wiring | L | Medium. Fresh live-family holdout and feature ablation; reject exact clone fingerprints |
| 10 | **CONFIRMED semantics, PLAUSIBLE gain:** tolerated single unfed day / production-care tradeoff | Labour/feed savings may exceed lost care in a narrow window; risky, unknown gain | Own animal state, pending bonus, inventory, days left | Yes; both seats | Present; needs targeted parity | L | Medium. Two-day escape, pending bonus reset, production cap, whole-season net gain; not default executor behaviour |
| 11 | **CONFIRMED semantics, PLAUSIBLE gain:** occupying tiles changes future RNG consumption | Could alter distributions of weeds/shops; profitable directed control unproved | Public board history; unknown seed remains unavailable | Conditional determinism; player-order dependence | Same shared RNG ordering in native paths | H for deliberate RNG manipulation | High. Distributional test across unseen seeds; **not selected** |
| 12 | **REJECTED:** automatic seat-0 market price priority | Same-slot quotes are computed before either commit; no universal first-seat sale advantage | Joint order context for offline check | Seat effects can arise elsewhere | Source agrees on simultaneous quoting | X as a claimed advantage | Confirm same-slot equal sales with both seats; do not build a seat-0 frontrun rule |
| 13 | **REJECTED:** free buy/sell cash loop or negative-quantity money creation | Post-buy pricing and quantity validation defeat the simple claim | Market quantities/prices | Yes | Source checks present | X; investigate any genuine bug with hosts | Boundary tests, not an architectural revenue source |
| 14 | **REJECTED:** infinite shed/animal/crop yield or terminal asset bonus | Capacity discards overflow; production is capped; final reward is cash | Own state | Yes | Source checks present | X as a claimed advantage | Test stated caps; never optimize a surrogate crediting unsold assets at termination |
| 15 | **REJECTED:** hidden seed/platform exploitation; universal missing seat-1 clock | Seed is not exposed; standard runner supplies shared clock | Would require hidden information or a broken custom harness | Not a legal policy feature | Official five-state clock/seed probe performed | X | Keep legal clock fallback and reset tests; no extraction, intrusion, network abuse or private-data access |

**Legality boundary:** public source and the ordinary action API establish technical feasibility, not blanket permission to exploit every bug. A public participant reports that hosts allowed public replay use, but the linked host statements and full rules were not independently retrievable.[^8] Phase 0 must record the actual current rules and any relevant host clarification. The selected architecture needs no controversial exploit: it works through ordinary production, scheduling, market orders and public observations. Suspected validation bugs or deliberate RNG manipulation are excluded unless explicitly cleared.

## External research findings

The relevant literature supports components, not a claim that their benchmark gains transfer numerically to Kaggriculture.

| Research / implementation | Established contribution | Decision for this system |
| --- | --- | --- |
| PSRO, Lanctot et al., 2017; 2024 survey; OpenSpiel `psro_v2` | Iteratively expand a restricted policy game using response oracles; the oracle can be search/planning/evolution rather than neural RL. | Retain a small payoff matrix and search responses to diverse mixtures. Do not claim global Nash convergence from approximate finite-league search.[^14][^15][^16] |
| AlphaStar league, 2019 | A league with differing training roles and historical policies helps discover and withstand strategies; supervised initialization was useful. | Borrow frozen anchors and exploiters. Do not copy its neural training scale or hundreds of agents.[^17] |
| NFSP, 2016 | Learns approximate best-response and average-policy components in imperfect-information self-play. | Relevant warning against only the latest self-play policy; not the shortest path to a compiled route agent here.[^18] |
| MAP-Elites, 2015; pyribs; QDax | Preserve high-quality solutions across behavioural descriptors rather than a single best genome. GPU-oriented QD implementations exist. | A 24-entry route archive is enough initially. Use realized production/cash descriptors; do not port the existing CUDA simulator to JAX just to use QDax.[^19][^20][^21] |
| Rolling-horizon evolution and two-player coevolutionary planning | Search action sequences with a forward model; compare against tree search; planning horizon and representation matter. | Use complete remaining-season evaluation and structured mutation. No need to ship rolling-horizon planning.[^22][^23] |
| Options / temporal abstraction, 1999 | Temporally extended closed-loop actions provide meaningful units of planning and learning. | Represent a productive lifecycle and its initiation/termination conditions, not merely a 72-row slice.[^24] |
| SKETCH / CEGIS; DreamCoder | Sketch completion and counterexamples can synthesize programs; learned reusable abstractions can improve program search. | Use a small typed job grammar and counterexamples to repair compiler/runtime contracts. Do not mistake predicate refinement over four fixed bodies for general program synthesis.[^25][^26][^27] |
| Policy distillation; DAgger; VIPER | Teacher policies can be compressed; student-induced state distribution matters; value-sensitive tree extraction can outperform plain imitation. | Learn only the small route selector if necessary. Value-weight labels and evaluate the actual compiled student; exact route compilation avoids most action-level distillation risk.[^28][^29][^30][^31] |
| EfficientZero V2, 2024 | Sample-efficient model-based RL across multiple control benchmarks. | Learned models are not inherently impractical, but learning dynamics we already simulate exactly adds error and engineering. No world model in the selected plan.[^32] |
| FunSearch, 2023; AlphaEvolve, 2025 | Evolutionary program proposal combined with executable evaluation can discover improved algorithms. | Supports separating an expressive proposal mechanism from an exact evaluator. An LLM-program-search service is not required; arbitrary code evolution is not the first experiment.[^33][^34][^35] |
| Reliable evaluation literature / `rliable`, 2021 | Report uncertainty, distributions and robust aggregate metrics rather than isolated point estimates. | Use seed-clustered paired intervals and cohort results. Reuse statistical ideas without adding a service or mandatory dependency.[^36] |
| OR-Tools CP-SAT | Mature integer constraint solving and scheduling infrastructure. | A possible local schedule feasibility tool, not the primary nonlinear two-agent game solver. No solver dependency before a measured decoder bottleneck.[^37] |

### Learned teacher reconsidered quantitatively

**PROPOSAL estimates, not Kaggriculture training measurements.** A full neural policy would need to emit coordinated actions for multiple workers plus ordered market transactions, respect atomic seeds and cash/capacity constraints, retain production commitments, and learn returns delayed by hundreds of turns. Flat action classification is a poor interface. A viable teacher would need the same macro/job abstraction proposed for search, action masks, and live-opponent rollouts.

| Learned role | Plausible starting data/compute | Implementation / first signal | Distillation and risk |
| --- | --- | --- | --- |
| Behaviour cloning a strong executor | 10,000-100,000 freshly generated games, roughly 7.19M-71.9M acting transitions; small model training roughly 2-12 GPU-hours after efficient data collection | 4-8 engineering days for structured labels, masks, recurrent state and executable evaluation | Can reproduce known behaviour, not inherently surpass it. A 0.1% independent consequential error rate over 719 turns would leave only about 49% entirely error-free games; independence is an illustration, not an observed rate. |
| Macro-level self-play RL teacher | Pilot 10M-100M environment transitions, then potentially 100M-1B; roughly 10-100+ GPU-hours including learner/inference, highly throughput-dependent | 2-4 weeks for a credible first policy/search comparison on this hardware | Higher ultimate upside, high initial-basin and credit-assignment risk; route extraction loses reactive decisions unless represented explicitly. |
| Learned continuation critic/proposal ranker | 100,000-1M evaluated route/context outcomes generated by the selected search; hours of small-model fitting | 2-5 additional days once useful outcomes exist | Could save oracle calls, but out-of-distribution overconfidence is dangerous. It must never replace final exact evaluation. **Not on the committed critical path.** |
| Small route selector | Hundreds to a few thousand independent information-state groups with exact route counterfactual returns | 1-2 days after a genuinely useful route portfolio exists; negligible model-training compute | Direct bounded decision-tree/BKG4 compilation. This limited supervised step is part of the chosen architecture. |

On the currently measured small-batch Shop workload, 10M game transitions correspond to 10M / 719 = **13,908 full-game equivalents**. At approximately 10.6 FGE/s that is about 22 minutes of evaluator wall time, before neural inference/training and data overhead. One billion transitions would be about 36.5 hours at the same evaluator rate. A full RL programme is therefore not rejected because simulation is impossible; it loses because action abstraction, training stability, faithful extraction, and engineering time are unproved while structured route search directly matches the existing deployment format.

Do not train a dynamics world model when exact C++/CUDA dynamics are available and cheap. Do not use public replay action cloning as an unqualified offline-RL dataset: logged trajectories have narrow action support, historical-engine confounding and adaptive-opponent mismatch. The archive makes cheap macro-frequency/retrieval priors worthwhile now; it does not supply counterfactual value labels. A future small critic still requires trustworthy live-search returns and is neither a parallel architecture nor the fallback. The replay-specific cost and model decision appears below.

## Architecture alternatives

The estimates below assume one engineer using the existing 4 GB RTX 3050 Laptop GPU. They are decision ranges, not promises or measured learning curves. Compute includes simulated games, not merely optimizer updates.

| Candidate | Strategic upside | Engineering / first signal | Compute scale | Local optimum / overfit risk | CUDA reuse / BKG4 fit | Decision |
| --- | --- | --- | --- | --- | --- | --- |
| Continue deterministic CCPR or DSQD over current SpecialistPrograms | Low-to-medium residual gains | 1-3 days | 1,000-20,000 FGE/test | High representation ceiling; narrow gates remain easy to fit | Excellent reuse, direct compilation | Reject as primary; keep repair tools |
| Single-incumbent full-route hill climbing / beam search | Medium/high if strong macro decoder | 3-7 days | 10,000-100,000 FGE | Beam/hill-climber can retain one economic basin; heuristic truncation creates horizon bias | High reuse, excellent fit | Serious but dominated by small population search at modest extra cost |
| Offline macro-MCTS / information-set planner | High possible ceiling | 1-3 weeks | 100,000+ continuations, irregular batches | Branch explosion, uncertain opponent action model; privileged-state determinization risk | Partial GPU reuse; extracting a small causal student is additional work | Do not select for first medal push |
| **Replay-seeded population production-route synthesis + small live best-response league** | **High practical upside: informed new economies plus robustness** | **2-4 days bounded first test; roughly 2 weeks to one complete loop if it passes** | **12,000-FGE pilot; 100,000-250,000 FGE per later campaign** | Medium: replay/grammar bias remains; cap clone weight and reserve novel proposals | **High CUDA reuse; nearly exact route compilation** | **SELECTED; replay priors must earn their place** |
| Neural BC/self-play teacher -> tree/routes | Potentially highest broad reactive ceiling | 2-4 weeks before a fair strong-baseline comparison | 10M-1B transitions; 10-100+ learner GPU-hours | High imitation basin, sparse long credit, distillation loss | Simulator useful, learner integration and student extraction substantial | Not primary |
| CP-SAT/SMT whole-season economic planner | Good feasibility/upper-bound insight; uncertain game strength | 1-3 weeks | CPU-heavy solve time plus live-simulation checks | Exact full game includes nonlinear prices, RNG-dependent occupancy and opponent policies; simplified optimum can be wrong | Poor direct GPU search reuse; route output fits well | Not primary; bounded feasibility only if needed later |
| Mechanics-first overlays on latest public port | Medium immediate, lower independent-strategy ceiling | 1-3 days | 1,000-10,000 FGE | Crowded public ideas; improvements may only win mirrors | Excellent reuse and compilation | **Only small fallback after the explicit early failure gate** |
| LLM/evolution/program-synthesis platform with learned proposal/value models | High theoretical upside | Several weeks of evaluator/proposal/security integration | Uncertain external inference cost plus large evaluation volume | Can optimize evaluator artifacts or generate unmaintainable programs | Mixed; compilation and parity can dominate | Too much infrastructure before evidence |

CMA-ES and Bayesian optimization are useful for low-dimensional numerical parameters such as reserve levels or sale offsets, but not the primary representation for variable-length, categorical, dependency-coupled production schedules. MAP-Elites-style retention can accompany discrete mutation without pretending a route is a smooth vector. Dynamic-programming approximations and economic upper bounds are useful diagnostics; with opponent-dependent prices and persistent assets, a small sufficient Markov state for exact season-level DP has not been established.

**Why the winner wins:** it searches the type of object already demonstrated to carry strong play, makes long-horizon improvements directly executable, spends compute on the existing exact simulator, and needs only a small population rather than a neural training stack. It also addresses the strongest public route-search limitation by keeping both opponents live. The committed system is one evolutionary search with full-horizon evaluation, not an ensemble of MCTS, RL, solvers and QD running indefinitely.

## Final chosen architecture

```text
Public replay index -> small typed-job/timing prior (TRAINING only)
                         |
Shop Plus executor + direct-route founders + small frozen league
                         |
           fresh seeds, both seats, difficult live opponents
                         |
     replay-seeded dependency-closed proposals + 20% novel exploration
                         |
           batched CUDA full-season continuation oracle
                         |
       small economically diverse archive + response search
                         |
        exact route compilation + bounded BKG4 selection
                         |
         fresh policy-level validation and release audit
                         |
                  champion N + 1
                         |
             frozen league anchor and new challenger target
                         +-----------------------> repeat
```

### 1. Parent and strategic centre

Use **Shop Plus as the initial executor, default behaviour, reset semantics and measured reference**. Use the direct Shop schedules as primary route founders; retain direct Gold/Gold Plus as alternative genetic material and historical comparison policies. `assets/champion/` stays the authoritative repository champion until a later explicit promotion.

Replay-derived economies become additional founder *descriptions*, not raw schedules spliced into the executor. The first pilot imports only post-144 production jobs into the existing causal composition. Earlier openings and full-route founders remain candidates for the same search after this gate, with full reset evaluation and explicit reconstruction of recovery semantics.

There is no requirement that every new genome remain a specialist attached to Shop Plus. A full superior policy may replace its economic schedule. Conversely, do not revert to a direct port solely because of an unverified leaderboard anecdote. The breeding population, not one immutable parent, becomes the strategic centre. Cross-family schedule transfer is admitted only after explicit action/state compatibility checks; raw Gold-to-Shop tape crossover is not assumed safe.

### 2. Teacher/search system

Run **population-based large-neighbourhood route search**. Maintain at most 24 economically distinct elites. A generation produces 128 proposals from frozen founders and archive elites, evaluates them through the full remaining game against live opponents, confirms a small shortlist on fresh BUILD contexts, and retains complementary successes. The same search engine also produces best responses to vulnerable league cohorts. There is no separate learned teacher.

Proposal sampling is stochastic and recorded. Up to 80% of proposals use family-balanced replay/elite job and timing priors; at least 20% ignore replay frequencies and explore the full declared legal job support. This is a mixture inside one search, not an imitation policy followed by an unrelated optimizer. Engine simulation, compiler output, runtime selection and deployment remain deterministic. Reproducibility comes from pinned inputs, proposal RNG state and measured results, not a ban on random search.

### 3. Search representation

A genome contains a small route portfolio, its existing causal selection skeleton, and **typed production jobs with dependency-closed edits**. Initial jobs cover animal acquisition/service and crop lifecycles. Later successful jobs may change expansion, worker allocation and liquidation timing. One edit may touch 24 or 72 contiguous turns, but its resource and production consequences are repaired through turn 718.

Replay mining proposes job parameters and observed dependencies; the compiler owns feasibility. Keep observed facts separate from inferred prerequisites. A recorded request is not proof of execution, and a successful observed lifecycle is not proof of profitability on today's engine. Missing service/financing dependencies cause rejection, not silent raw-action concatenation.

The minimal job record is:

```text
job kind; site/worker lane; start day; deadline/end day;
resource prerequisites; buy/pickup/place or plant dependencies;
service/harvest cadence; cargo/deposit/sale obligations;
fallback and termination condition
```

The compiler lowers a job into existing unit/market actions and bounded recovery parameters. A deterministic deadline-ordered scheduler reserves worker time, travel, money and materials; it does not solve arbitrary general planning. Existing tape work remains unchanged outside the selected dependency closure. Walking needs no collision planner in this engine. If the bounded scheduler cannot fit the closure, reject that proposal rather than silently emitting impossible work.

Examples of meaningful mutation are: replace a not-yet-purchased cow with sheep and update purchase, placement, service and sale assumptions for the rest of the season; bring a production site's acquisition forward with its financing and feed; replace a crop lifecycle with another complete lifecycle; stagger harvest/deposit work to avoid capacity loss; change production expansion and its hiring plan together. Moving a BUY_ANIMAL token alone is not a strategy mutation.

A starting generation allocates approximately 50% of proposals to lifecycle/subschedule mutation, 25% to capital or labour retiming, 15% to contract-compatible crossover, and 10% to quantities/reserves/sale timing. These are frozen starting proportions, not separate optimizer services. At least half of candidates must change production or labour allocation before the final quarter, rather than allowing the archive to fill with liquidation variants.

### 4. Challenge/opponent distribution

Cap the working league at **12 live policies**: four anchors/founders, four behaviourally different economic challengers, and four rotating historical/best-response entries. Do not count four source wrappers around one tape as four strategic families. Deduplicate by ancestry, executed opening/job structure and payoff similarity, not notebook name alone.

Use a simple search opponent mixture: 50% uniformly over structural cohorts, 25% over the current champion and recent strong anchors, 25% over empirically difficult opponents. Maintain a floor on each cohort; hard opponents cannot erase diversity. Targets include milk/wool/crop-heavy production, faster expansion, high-capacity-throughput production, different sales timing, reactive recovery and public-state adaptation. Small fixed stress policies can be useful even if weak overall.

Public agents may be declared **TRAINING** stimuli or founders. This is an explicit strategic change from pure internal BUILD: excluding strong observable strategies a priori is not justified by the evidence. Source identities remain provenance and offline bookkeeping only; they are never runtime features. Public agents used for training are never reported as independent HOLDOUT evidence. Hold out genuinely different source/behaviour families, newly generated adversarial responses, and fresh seeds. Do not move an existing protected panel into BUILD.

Replay-derived challenge regimes enter through executable public policies or newly synthesized live challengers. A historical market-saturation state motivates a supply challenge; its recorded opponent action suffix is not that challenger. Replays lack opponent controller memory, so they cannot directly populate exact native fork roots. Capture roots from fresh live BUILD games instead.

### 5. Objective

For candidate `p`, opponent `o`, seed `s` and seat `a`, let `u` be 1 for a win, 0.5 for a tie, 0 for a loss. Let `m` be final own cash minus opponent cash. Search uses paired differences from its actual founder/reference under the same initial seed and opponent, while allowing the subsequent worlds to diverge legitimately.

Use the following frozen starting objective within each archive cell:

```text
J(p) = 0.75 * cohort-weighted mean(u)
     + 0.25 * mean(u in the worst-performing quarter of cohorts)
```

For an N-context screen, rank by `floor(N * J) / N`, then by the mean paired cash-margin change clipped at +/-5,000 coins per game, then by candidate hash for reproducibility. This one-game-resolution grouping gives search a money signal without calling tiny sampled utility differences statistically significant. Confirmation reports unrounded utility and seed-clustered uncertainty. Also record own revenue, purchase/hire costs, lost overflow, failed production jobs and final unsold goods. This distinguishes a better economy from a tiny tie-breaking tactic without replacing the competition objective with unbounded money farming.

No per-case margin nondecrease is required during search. No best-response candidate has to beat every league member before it can enter the archive. Hard constraints are action legality, valid state access, resource/runtime bounds and exact semantic checks. Final release requires robust improvement and the explicit downside limits below, not 100% wins against a parent.

### 6. Diversity and non-transitivity

Use a 12-cell archive with at most two elites per cell: dominant realized revenue source (milk / wool / other), earlier/later capital commitment, and earlier/later cash realization. Freeze descriptor thresholds from founder BUILD behaviour for each campaign. Also retain the opponent-payoff vector; equivalent behaviour/payoffs do not become new capabilities because a predicate or hash changed.

This archive is an exploration aid, not a claim that these three descriptors perfectly capture strategy. A weak diverse route is not automatically deployable. Selection keeps quality within cells and tests whether alternatives add portfolio utility.

Maintain a seat-balanced 12x12 payoff summary and uncertainty. Initially use cohort-balanced/hard-opponent sampling, not a Nash solver. If statistically credible cycles appear, the **same** response-oracle loop can use a restricted-game mixture with a uniform exploration floor. A finite meta-game solution would protect only its included policies, not prove unexploitable competition play. Public evidence currently favours testing broad economic superiority first.

### 7. Long-horizon credit

Every serious production proposal is evaluated to action 718 with live opponents. No day-end cash surrogate ranks away a promising eight-day investment. Full-season suffix return is the teacher's credit assignment. Diagnostics attribute returns to production jobs by removing/reverting complete dependency closures, not by treating every action as independent.

Use saved roots at genuine decision boundaries such as 72, 144 and 216 only when the prefix and both controllers are identical. An edit before a root invalidates reuse of that root. Counterfactual alternatives share legitimate information at the decision; they do not select a different route for each hidden seed and then pretend that per-seed oracle is executable.

### 8. Mechanics exploitation

Mechanics enter as **tested macro semantics and proposal operators**: care/yield alignment, crop service cadence, pre-market cargo flow, partial deposits, hiring/financing order, shop-demand response and terminal maturity. Existing weed recovery and liquidation are retained initially. Gain claims require paired full-horizon ablation. No speculative RNG exploitation is a required capability.

### 9. Distillation

Most teacher output is compiled exactly as route data. There is no need to imitate thousands of action labels with a neural-to-tree pipeline. The small learned component is an optional bounded selector over fixed compatible continuations, trained from BUILD counterfactual returns on legal observations.

Use value-sensitive selection loss: choosing a route 2 coins worse is less serious than choosing one that loses a viable economy. Fit/prune a depth-limited decision tree or enumerate a small set of legal economic thresholds; compile its predicates and route IDs directly to BKG4. Student-induced trajectories must be evaluated, following the DAgger/VIPER lesson rather than reporting only teacher-label agreement.[^29][^30]

### 10. Routing

Keep the inherited first-two-shop selection as the starting selector. Add at most **two to four genuinely useful new continuations** in the first release. Additional branches are justified by out-of-sample portfolio value, not archive size. Binary predicates remain building blocks; a permanent two-policy ceiling is not mandatory.

Choose routes at common-prefix or explicitly contract-compatible boundaries. A contract includes persistent tile layout/ages, placed animals, land, money/reserves, seed/cargo state, controller state and pending recovery. A new day resetting workers does not make arbitrary routes compatible. Keep the selected route latched; use a bounded switch count and abstain when feature support or feasibility is insufficient.

Candidate routing inputs are all observed shop counts, current product inventory/marginal price, own feed/capacity reserve, visible opponent production mix and public economic trajectory. Introduce only the derived features actually used by an accepted selector. No source identity, public episode ID, hidden seed, unobserved future shop, or opponent private inventory.

### 11. Validation

Use **three data roles**, not a ladder of ten named panels: adaptive BUILD, fresh DEV for a frozen shortlist, and one release audit for a frozen complete policy. Fidelity tests are engineering checks, not another statistically reusable holdout. Whole seeds, trajectories and opponent families are grouped correctly; old exposed experiments remain historical evidence.

### 12. Promotion

Promote the complete deployed policy only after the fresh release audit establishes useful overall and difficult-cohort win improvement, bounded downside, and zero semantic/runtime failures. The authoritative champion changes only in a later implementation session through the repository's explicit promotion process. A specialist's standalone result or a 100% donor win rate cannot trigger promotion.

### 13. Autonomous improvement loop

Champion N contributes a frozen executor/reference and a league entry. The archive identifies weak cohorts and unused productive alternatives. Search produces new production schedules and best responses; BUILD tests and deduplicates them; a frozen student/portfolio faces fresh DEV and release evaluation. Champion N+1 becomes another frozen anchor, not the only opponent. Search resumes with fresh seeds and retained diverse founders. Failed release cases may become future training data only after being retired and replaced; they cannot remain advertised as unseen validation.

## Why this should beat the current baseline

**INFERENCE, to be falsified:** a compiler that can alter a production lifecycle can change the output and resource economy for hundreds of turns. The current four-body repair vocabulary cannot. Existing route differences, current unplaced-animal decisions, biological delays, and the public route-search construction all support the existence of a strategically relevant search space.

The causal argument is specific:

1. Search creates different productive capacity, timing and labour allocation, rather than only recovering missed sales from an unchanged schedule.
2. Full-horizon exact simulation prices those investments, including feed, hiring, scarce-market responses, overflow and terminal losses.
3. Live opponents respond to changed market and production conditions; fixed recorded tapes no longer create false certainty about counterfactual interactions.
4. A small archive preserves alternatives long enough for different economic conditions and opponents to reveal their value.
5. Exact route compilation avoids the error accumulation of a large action-level distilled policy.
6. Fresh policy-level comparisons measure broad utility rather than the number of safe local predicates discovered.

This is not an argument that a larger search budget automatically wins. The donor's reported route-search scale is already substantial. Our potential advantage is **coordinated new jobs, live counterfactual opponents, exact GPU search, and better evaluation**, not merely rerunning a smaller version of public tape selection. If those do not yield material fresh gains promptly, stop scaling.

## BKG4 deployment design

### Actual representational headroom

The generic BKG4 implementation supports 65,534 nodes, depth 48, 128 blackboard slots, 512 node-state slots and a 65,534 instruction-budget ceiling; normal visits above 4,096 trigger a warning. It supports stateful sequences/selectors, reactive conditions, typed blackboard writes/additions, waits, movement and unit/market intents. The generic feature registry has 59 legal feature names, including public opponent state. [Python ABI/compiler](../src/kaggriculture_bonsai/runtime/bonsai_kg_v4.py), [native ABI](../native/bkg4/include/bonsai_kg_v4_abi.hpp).

These are **format/VM maxima**, not competition runtime budgets or sensible targets. Shop Plus uses about 0.13% of the node maximum, but increasing node count is not an objective. The present adapter is much narrower than the generic VM: its route decision observation exposes STEP and the first two SHOP_AT values; the fork-repair adapter supports eleven feature IDs with twelve sampled feature values. Generic feature existence does not mean that the Shop deployment/search adapters already implement it. [Shop native adapter](../src/kaggriculture_bonsai/agents/shop_router_0909/reference_port/native.hpp), [fork evaluator](../native/cuda/src/phase1_fork_repair.cu).

The BKV6 record is 94 bytes, with at most 12 saved hands plus farmer and ten market positions. Route counts/IDs and per-route lengths use 16-bit fields; action count is 32-bit. The generic extended loader accepts variable route counts, but the isolated Shop adapter hard-codes 13 routes, 719 steps and a one/two-slot controller restriction. Changing packs and adding features therefore needs a **new narrowly scoped candidate adapter**, not a claim that any bigger controller is already plug-compatible. Do not modify the frozen reference adapter to make experiments pass. [Route format](../src/kaggriculture_bonsai/runtime/route_format.py), [Shop pack](../src/kaggriculture_bonsai/agents/shop_router_0909/reference_port/pack.py), [bridge restrictions](../src/kaggriculture_bonsai/agents/shop_router_0909/reference_port/bridge.cu).

**Replay-derived compatibility finding:** the detailed 82-player audit observes up to 15 actual hands, with 31 trajectories exceeding 12 at some point. Raw replay routes therefore need an explicit saved-hand arity check; truncating them is not distillation. The pilot uses existing feasible worker lanes and rejects unsupported imports. If a later valuable founder genuinely needs more lanes, represent the extra bounded jobs in BKG4/runtime intents or make a narrowly scoped, parity-tested route-format extension; do not rewrite the VM merely to ingest the archive. Day-boundary worker counts cannot diagnose this limit because hires reset daily.

### What lives where

| Component | Deployed responsibility | Initial design budget, not a VM limit |
| --- | --- | --- |
| Controller | Causal route selection, latched commitment, bounded overrides, recovery/terminal priority | Target <=256 nodes and <=512 visits; allow a measured exception rather than arbitrary node minimization |
| Routes | Full executable production/worker/market schedules and route-specific terminal tails | Keep the 13 founders plus at most 4 accepted new continuations initially; roughly 1.15 MB raw action data before metadata/duplicate-tail handling |
| Macros | Offline typed jobs compiled to route records; only genuinely conditional bounded micro-programs remain runtime | Two initial job families; no general online planner |
| Reactive overlays | Weed repair, legal quantity/capacity adjustment, feed preservation, delayed-work handling and liquidation | Reuse existing mechanics; every new overlay gets explicit precedence and parity tests |
| Blackboard/state | Route ID, commitment boundary, small regime summary, failure/recovery flags and bounded counters | Target <=16 BB slots and <=16 node-state slots; native queues/reserves counted separately |
| Derived features | Deterministic summaries from current legal own/public state, such as visible herd counts and observed shop pull | Add only features used by accepted rules; identical C++/CUDA/runtime definitions |

A 17-route raw payload is approximately 1.15 MB, far below a reason to optimize thousands of controller nodes. It does not establish a competition package-size allowance; verify the actual rules before release. The official local spec has `actTimeout=1` second. The policy should remain comfortably below this with measured cold-start and maximum-worker/overlay states, not use VM maxima as a timing guarantee.

**Composition rules:** preserve order-slot semantics and output arity; deduplicate or cancel advanced sales exactly; update future-action lookahead when routes change; clear only the state whose lifecycle ends; keep opponent state private; reset on a new episode/step rewind. Route 2 at 648 is not a universally valid terminal tail for a newly synthesized economy. A generated route must carry its own compatible continuation through 718, and zero-edit compilation must reproduce the inherited executor, including the overlay's cross-boundary lookahead.

**Are we using only a fraction of BKG4's power?** Yes, especially route data, stateful commitment and legal public-state selection. The appropriate response is not more arbitrary specialists: it is a search/compiler that discovers productive schedules and small conditional recovery programs, then uses those existing deployment capabilities.

## Training/search pipeline and CUDA budget

### Concrete discovery mechanism

1. Freeze executor, engine/source hashes, route ancestry, dataset version/manifest, replay TRAINING membership, proposal RNG, legal feature definitions, league members, cohort weights and disjoint data roles.
2. Index replay metadata and stratified behaviour with a local Parquet/JSONL table; retrieve a capped, current-regime, family-balanced job sample. Decode the **executed** founder composition into a job/route view and translate selected replay jobs into it. Preserve raw records and arity; attach only the dependencies required for the two initial job families. Verify a zero-edit round trip. No database service or neural model is needed.
3. Generate dependency-closed candidates. Schedule selected jobs against existing lanes, budgets and persistent state; emit compact route deltas plus any bounded overlay parameter changes. Reject structurally impossible candidates before simulation.
4. Evaluate candidate x context batches in CUDA from shared valid roots, with both policies active at every step. Use full returns and diagnostic counters. Candidate identity includes route delta, controller, overlay and source version.
5. Confirm a small, behaviourally diverse BUILD shortlist on independent full games. Retain quality by archive cell and payoff coverage, not by how many near-equivalent hashes exist.
6. Fit only a small selector over frozen compatible route alternatives. Re-evaluate the complete compiled executor from reset; do not add isolated specialist scores together.
7. Freeze the candidate before fresh DEV. The release audit is queried once for a frozen complete policy, not used to repair thresholds.

### Measured hardware and throughput

The available GPU is an **NVIDIA GeForce RTX 3050 Laptop GPU, 4,096 MiB**, with 3,756 MiB free at inspection and driver 595.84.

The corrected CCPR ledger records **293 CUDA calls, 898,712 simulated turns, and 117.5896 seconds**: approximately **10.615 full-game equivalents/second** using that experiment's `turns / 720` accounting. Its CPU ledger has 13 calls, 44,204 turns and 1.834 seconds, a different workload; this is not a fair CPU-versus-GPU speed comparison. [Corrected campaign evidence](../workspace/self_improvement/phase2_ccpr_corrected/).

The older oracle bakeoff reports 9,600 games in 1.548184 seconds of direct-Bonsai kernel time, and RLRMB end-to-end search throughput of approximately 2,382 games/second. Those were different policy/evaluator workloads. They show batch potential, **not** current Shop Plus throughput. The old RLRMB result also recorded ten CUDA faults; its performance number is not a correctness certificate. [Bakeoff result](../workspace/oracle_bakeoff/result.json).

Current fork search has one GPU thread per case, small case batches, a host-side loop over candidates and repeated host/device results traffic. The standalone action bridge launches a one-thread kernel per tick and is a parity interface, not a high-throughput full-game teacher. The existing capture harness can replay a full reference game per requested root. Avoid extrapolating kernel-only game rates to a new route-search campaign.

### Minimal throughput work, only after useful search exists

Use the existing exact simulator and stateful parent executor. Batch the **candidate x context** product, keep immutable packs and snapshots resident, upload route deltas or candidate pack views once per batch, and return compact outcomes. Start with 256-1,024 simultaneous continuations, not a simulator rewrite or an immediate million-game campaign.

The verified current fork record is **49,488 bytes**, including both controller states. 4,096 such records occupy approximately 193.3 MiB. A single 719-turn raw route is 67,586 bytes; 512 distinct single-route payloads occupy about 33 MiB, but 512 complete 13-route packs would be approximately 429 MiB. Shared base-plus-delta storage matters. Working states, stacks, simulator tables and new macro/overlay state add memory beyond these figures; a snapshot calculation alone is not a safe launch-size limit. Start below 1 GiB of explicit batch allocations and measure occupancy/spills and peak memory.

Planning budgets use **10 FGE/s** conservatively until the new workload is measured. At that rate 100,000 FGE is about 2.8 hours and 250,000 about 6.9 hours, excluding preprocessing and independent Python opponents. Use full-engine-turn accounting across captures, rejected candidates, reference games and retries. For new work standardize actual FGE as `executed transitions / 719`; retain `/720` only when reporting historical experiments exactly.

Independent unported public agents may run in the official Python engine for small DEV/release cohorts. Do not replace them with fixed action tapes to claim live-opponent GPU throughput. Promote a public challenger to the GPU league only after a separately verified native implementation or use a native internally synthesized behavioural challenger. This avoids making porting every public agent a prerequisite for useful route search.

## Meta/generalization strategy

### Three roles and one complete-policy decision

| Role | Contents and permitted use | Typical scale after the pilot |
| --- | --- | --- |
| BUILD | Fresh multi-regime seeds, live training league, archive/search feedback, grouped cross-fitting for selector construction | Adaptive budget of 100,000-250,000 FGE/campaign; candidate screens on 32-128 contexts, confirmation on 256+ |
| DEV | Frozen small shortlist; new seeds and withheld challengers/families; no gate repair on these labels | At most 4 complete policies x 64 independent seeds x both seats x 8 opponents; paired reference |
| RELEASE | One frozen complete candidate, new seeds, frozen family weights, targeted safety cases and official packaged-runtime canary | 512 independent seeds x both seats x 8 opponents = 8,192 games/policy, plus narrow canary cases; parent paired |

These are design scales, not evidence already collected. If many opponents are only available as Python agents, reserve the full 512-seed panel for the native league and a predeclared smaller independent-source stratum; report their uncertainty separately rather than pretending equal precision. Do not let slow Python execution turn BUILD into static-tape search.

No seed may cross data roles. Reusing a seed across opponents/seats within a role is useful pairing, but its outcomes form a cluster, not independent samples. Multiple checkpoints in one game are one trajectory. Opponents sharing production schedules form a source/behaviour cohort. Existing DEV/FINAL/SEALED artifacts do not become a new pristine release set because a directory is renamed.

**Replay role:** all inspected/mined archive material is TRAINING/research exposure, never a hidden local HOLDOUT. Record observed replay seeds for exclusion and generate independently salted BUILD/DEV/release seed lists; exact deduplication applies to all consumed replay bodies, without claiming knowledge of unparsed seeds. Group replay examples by episode, economic/action-prefix lineage and time/version, not random steps or team names. A held-out date slice can test a miner, but it does not certify competitive generalization. Public comparison agents with unknown replay ancestry are source-held-out *transfer checks*, not proven behaviour-family holdouts. Release still needs fresh live challengers/families with no mined ancestry; if independence cannot be established, disclose the gap rather than relabeling the archive.

Use cohort-balanced results as primary when real meta frequencies are unknown. Report a secondary plausible current-meta weighting only when based on legitimately observed and timestamped public information; cap clone-family weight and show weight sensitivity. Keep frozen hard-cohort and catastrophic stress results visible. Low-margin games are diagnostic, not the entire opponent distribution.

### Statistical and promotion rule

All thresholds here are **PROPOSALS to freeze before evaluation**. They replace universal parent-game monotonicity while retaining meaningful safety.

1. The candidate has zero action/state semantic mismatches in C++/CUDA/official canaries and zero invalid, timeout, missing-action or unsupported-feature failures. Any mismatch blocks promotion regardless of win rate.
2. On RELEASE, cohort-balanced paired win utility improves by at least **3 percentage points**, and the one-sided **95% seed-clustered confidence lower bound exceeds zero**. Use the frozen cohort weighting and whole-seed resampling, keeping both seats/opponents paired. Report an interval, not just pooled wins.
3. For every predeclared major cohort and each seat, the lower confidence bound on paired utility change must exceed **-5 percentage points**, with simultaneous/multiplicity-aware intervals. Small unsupported cohorts cannot silently pass; the verdict is insufficient evidence. This is statistical noninferiority, not zero losses.
4. The worst-quarter cohort utility must not decrease by more than **2 points** in point estimate, and all individual new losses must be summarized by cause and magnitude. A large broad gain may include ordinary matchup losses; it may not hide a newly broken economy in a small cohort.
5. Keep a small **fixed catastrophic firewall** defined before outcomes: representative legal-state mechanics/reset/overflow/terminal cases, plus full games with viable parent economies. Require no new runtime/production-collapse failure, and no parent win with margin >=5,000 turning into a candidate loss <=-5,000. A floor on final cash ratio (candidate at least 75% of paired parent on these cases) guards major collapse. This deliberately differs from treating every one-coin parent win as catastrophic.
6. Verify actual deployment packaging, source/license provenance, current competition rules and runtime budget before later promotion. Review only the frozen policy; no release-driven predicate edits.

A 3-point effect is a practical target, not a guaranteed detectable signal with every cohort mixture. With 512 seed clusters, uncertainty may still be too large, especially on rare regimes. If the frozen test is inconclusive, do not force a win: retain the incumbent. At most one predeclared independent replication for the identical policy is permitted; account for the two looks in the confidence rule. No repeated release fishing.

On saturated or zero-variance samples, do not treat a degenerate bootstrap interval as proof of zero population risk. Use a conservative bounded-outcome or exact paired-outcome interval, with the same seed grouping and multiplicity control, for the affected decision. No finite local panel certifies zero strategic regression on every possible competition game.

The changed safety definition is an explicit strategic decision: **zero semantic mismatch stays mandatory; universal per-game final protected W->L does not remain the objective.** Preserve all old evidence and do not retroactively relabel previous failures. Future implementation must reconcile the older governance document with this plan before changing promotion code.

### Detecting real leaderboard relevance

Local evaluation becomes a predictive model only after prospective calibration against legally observed competition results. In a later authorized execution phase, record immutable submission hash, engine version, opponent mix, game count, time window and settled outcomes. Compare mature results, not a fresh rating with an older well-sampled rating. A public rating analysis itself warns that its rating/matchmaking formulas are estimates, not authoritative platform specifications.[^38]

Do not optimize an assumed Elo equation or count notebook medals as agent performance. The smallest reliable process is a prospective log of complete-policy local predictions and subsequent real outcomes, with frozen candidates and realistic uncertainty. If two successive material local gains fail to transfer, change the **opponent distribution/strength evidence**, not add more validation tiers or another threshold-search campaign.

## Engine exploitation strategy

The first selected operators use ordinary and confirmed mechanics: coordinated care/harvest timing and crop/water/feed timing. Capacity-aware cargo flow and market-slot order are subsequent small operators within the same route search. Each new operator has a source anchor, legal inputs, a handful of exact boundary fixtures, one targeted counterfactual and fresh full-season evidence.

Distinguish three questions: does the engine do this, does the simulator match it, and does using it improve a competitive policy? A source-level peculiarity answers only the first. A successful synthetic fixture answers only part of the second. Neither licenses a leaderboard claim.

Keep a compact mechanics ledger rather than an exploit subsystem. For a suspect bug, preserve the minimal local reproducer, request host clarification before live use, and continue with ordinary mechanics. Do not explore seed inversion, hidden information, infrastructure behaviour or competitor files as part of the selected architecture.

## Previous assumptions: keep, change or uncertain

| Proposition | Verdict | Reason |
| --- | --- | --- |
| 1. Tiny local specialists are the right unit of improvement | **CHANGE** | Retain them for recovery; production lifecycles and full-season route closures become the discovery unit. |
| 2. Zero regression versus parent is mandatory at every intermediate step | **CHANGE** | It blocks investment/exploration and can reject harmless W->W margin tradeoffs. Use robust final utility and explicit catastrophic protection instead. |
| 3. Pure internal BUILD is superior to public training/search stimuli | **CHANGE** | No comparative evidence supports the prohibition. Declare public TRAINING material honestly, separate it from HOLDOUT, and preserve licenses. |
| 4. Deterministic-only search is preferable | **CHANGE** | Reproducible stochastic structured mutation is practical; deployment and engine semantics stay deterministic. No evidence establishes deterministic proposal superiority. |
| 5. Opponent-specific training necessarily causes harmful overfit | **CHANGE** | Best responses are useful inside a mixed population; failure comes from narrow evaluation or identity-dependent execution, not specificity alone. |
| 6. Full routes should remain frozen | **CHANGE** | The routes encode most of the economy and were themselves search-generated. Freeze references, not the entire future search space. |
| 7. Current-champion self-play is sufficient | **CHANGE** | Two near-mirror members did not supply strategic breadth. Preserve history, supply shifts, reactive challengers and response search. |
| 8. Binary routing is sufficient | **UNCERTAIN** | Binary gates remain adequate for the first accepted addition, but a permanent two-policy ceiling is unjustified. Expand only after portfolio value is measured. |
| 9. Existing BKG4 feature vocabulary is sufficient | **UNCERTAIN** | Generic legal features are rich; Shop/fork adapters expose only a subset and useful economic aggregates are absent. Wire minimal needed features rather than redesigning the VM. |
| 10. Current leaderboard evaluation methodology predicts rank | **CHANGE** | Current evidence is narrow and lacks a verified prospective local-to-remote calibration. Never equate 100% donor wins with top-10. |
| 11. CCPR/DSQD are the best available discovery approaches | **CHANGE** | CCPR's efficiency is established only inside a restricted vocabulary; DSQD was recommended but not run. Neither establishes superiority to route search. |
| 12. The current parent should remain the strategic centre | **CHANGE** | Keep Shop Plus as initial executor/reference, but breed from a population and permit a new dominant economy. |

### Retain, abandon and repurpose

**Retain:** BKG4 compiler/runtime, compact route packs, exact C++/CUDA engines, snapshot/fork correctness, live stateful opponents, donor attribution, known weed/capacity/terminal repairs, disjoint information roles and immutable historical evidence.

**Abandon from the future primary architecture:** endless WHEN-only repair campaigns, frozen-route doctrine, source-identity BUILD exclusion, champion-versus-itself as sufficient training, universal intermediate/final parent-game monotonicity, and node count as a proxy for strategic complexity. Do not resurrect the old neural stacks, RLRMB or Direct-Bonsai campaigns merely to fill an alternatives checklist.

**Repurpose:** CCPR witnesses diagnose failing production jobs and executor contracts; SpecialistPrograms express small accepted reactive components; binary routing is a compiler primitive; the CUDA fork oracle scores new schedules; quality-diversity retention holds production alternatives instead of predicate variants. Historical experimental code need not be deleted to stop funding its failed strategic role.

## Phased implementation plan

The phases below are **future work, not executed by this report**. They are sequential commitments behind falsification gates, not five architectures kept alive. Engineering estimates assume one engineer and existing components; stop at a failed gate rather than automatically proceeding.

### Phase 0: Establish the experiment contract

- **Objective:** make the first route experiment interpretable and legally deployable later, without a new research platform.
- **Minimal implementation:** one frozen manifest for byte identities, archive v19, current engine/rules, TRAINING exclusions, four native opponents, two equal-budget production arms and job operators. Reuse the research metadata index; inspect at most 64 selected current-regime replay bodies to annotate no more than 24 job records from at most eight economic founder profiles. Keep existing assets untouched; reconcile future governance and verify zero-edit execution.
- **Experiment:** validate logged action/pre-state alignment and a few complete observed lifecycles, then the existing small reset, capacity, atomic-seed, market-order and terminal fixtures. This is annotation/compilation fidelity, not search or replay-score replication.
- **Compute budget:** <=64 native FGE plus a small official canary; no full-corpus body scan, no model, <=one engineering day within the unchanged four-day pilot cap.
- **Pass:** at least eight nonduplicate, post-144, dependency-auditable job templates (at least four animal and four crop) are expressible in the shared compiler; pinned references agree, legal features are available, no unresolved ordinary-action rules issue or zero-edit discrepancy.
- **Failure:** any source/runtime mismatch or a need for an invasive VM/simulator rewrite. Stop and resolve identity/correctness before economic search.
- **Artifact:** compact frozen experiment manifest and fidelity summary, not a new agent or champion.

### Phase 1: Falsify replay-seeded route improvement

- **Objective:** test whether replay priors improve material fresh production discovery beyond equal-budget uninformed production search.
- **Minimal implementation:** two dependency-closed job operators, candidate-specific route data access, and a full-horizon native evaluation path. No neural model, full generic scheduler, multiway learned router or league service.
- **Experiment:** the single immediate experiment specified next.
- **Compute budget:** 12,000 native FGE, two GPU-hours, two CPU-hours for small independent-source checks; all-in engineering cap four days including Phase 0.
- **Pass:** the predeclared fresh-policy gain and production-attribution gates below; exact compiled/runtime agreement.
- **Failure:** no material fresh gain, only terminal improvement, or infrastructure exceeds the cap. If at least one arm passes the absolute production gate but replay priors do not establish incremental value, disable replay proposal weighting and retain the original route-search architecture. If neither passes, use the existing small mechanics fallback. Do not add another pilot arm or widen the budget after outcomes.
- **Artifact:** one frozen replay-informed candidate, one uninformed-production candidate, compact paired results and an explicit prior/production pass-fail decision. No automatic promotion.

### Phase 2: Build the small route league

- **Objective:** turn one productive search capability into repeated improvement without economic collapse.
- **Minimal implementation:** 24-entry archive, at most 12 live league members, 128-proposal generations, three operator types after the two pilot jobs prove useful, candidate x context GPU batching, payoff/cohort summaries and compact provenance. Reuse files/JSON and the existing build/cache layout; no database/service tier.
- **Experiment:** one 100,000-FGE campaign followed by an independent 100,000-FGE replication with fresh BUILD contexts and opponent mixtures. Include newly generated best responses and different production regimes.
- **Compute budget:** cap 250,000 native FGE and ten GPU-hours for both campaigns and confirmation at conservative measured throughput; approximately 3-5 engineering days.
- **Pass:** materially improved complete-policy DEV utility in the replication, more than one useful production/payoff alternative in the archive, and no dependence on one exact opponent/seat/seed cohort. Demonstrate measured batch speed, not a kernel-only claim.
- **Failure:** apparent gains disappear outside one founder family, population reduces to top-up variants, or speed/decoder cost prevents the budget from producing useful candidates.
- **Artifact:** compact elite route pack, frozen league manifest/payoff summary and reproducible campaign report. Do not store campaign traces or mass candidate output in tracked workspace.

### Phase 3: Compile and validate the portfolio

- **Objective:** convert discovered alternatives into a reliable small deployed policy.
- **Minimal implementation:** candidate adapter supporting required pack views and legal derived features, route contracts, a pruned selector with default/abstention, route-specific terminal tails, and explicit overlay precedence. Limit the first composition to 2-4 new continuations.
- **Experiment:** value-sensitive selector cross-fitting on BUILD, full closed-loop student evaluation on fresh DEV, counterfactual ablation of route choice versus executor changes, and official/C++/CUDA canaries.
- **Compute budget:** <=25,000 native FGE and four GPU-hours plus bounded independent Python checks; approximately 2-3 engineering days.
- **Pass:** compiled action semantics match exactly; closed-loop portfolio captures at least 80% of the **deployable causal teacher's** measured DEV improvement and retains a material positive improvement. Never use a hidden-seed per-case oracle as the teacher upper bound for this claim.
- **Failure:** routing destroys the economic gain, routes are not feasibly switchable, or feature/overlay growth exceeds the small runtime design without value. Do not compensate by adding more gates.
- **Artifact:** frozen candidate controller/routes/runtime identity and complete-policy DEV/canary report, ready for one release audit.

### Phase 4: Release and repeat

- **Objective:** promote only a defensible whole-policy improvement and test real competitive relevance.
- **Minimal implementation:** one release-audit command and compact prospective local/remote result log using existing packaging/promotion machinery when separately authorized. No new validation tiers.
- **Experiment:** frozen release panel, catastrophic firewall, official packaged-runtime checks; later authorized competition observation with adequate game count and timestamps. The report does not authorize a submission now.
- **Compute budget:** approximately 16,384 paired native full games for 8,192 games/policy, plus explicit external-source and canary costs; cap four GPU-hours and a predeclared CPU budget. Recheck if actual opponent availability changes the execution mix.
- **Pass:** all promotion rules above, followed by no contradictory adequately sampled competition evidence. Promote locally only through the explicit asset process; keep the previous champion reproducible.
- **Failure:** inconclusive/negative release utility, catastrophic regression, mismatch, runtime failure or two successive well-supported local gains without competitive transfer. Stop release or recalibrate the opponent evidence as appropriate.
- **Artifact:** authoritative champion update only after later approval/gates, compact release evidence and the next frozen league anchor. No automatic public submission or push.

## Immediate next experiment

**Name: replay-informed versus uninformed production-job proposals, with an unchanged causal parent selector.**

**Question:** does a small replay-derived job/timing prior discover materially better fresh production policies than the same structured search without that prior? This replaces, not supplements, the old production-versus-local-repair comparison. It tests both absolute economic improvement and the new information source with two arms.

### Frozen design

**Reference:** exact Shop Plus bytes and native executor identified above. First divergence no earlier than step 144; both agents remain live. The compiler preserves inherited causal route choices and overlay semantics. Candidate route changes include any affected continuation through 718; do not blindly force the old global route-2 tail after changing the economy. Zero-edit candidates must reproduce the original boundary lookahead and actions exactly.

**Mining scope:** reuse all 16,168 episode metadata records and the 348-game stratified research summaries. From current-version/configuration-compatible TRAINING examples, freeze at most 64 episode bodies, outcome-blind and balanced by economic profile/day rather than proxy rank. Inspect both seats, including losing trajectories. Select at most eight founder profiles and 24 auditable job records; keep at least eight nonduplicate templates across the two job families. No additional archive version, bulk full-body scan, large library or raw-route execution is required. Old-engine jobs are hypothesis material only and excluded from this pilot's prior. Record every selected episode/seat, template contract and rejection reason before CUDA returns.

**Shared production support:** 64 animal-job changes and 64 crop-job changes per arm. Animal jobs alter one or two **not-yet-acquired** placements after 144, or shift their acquisition/service timing by at most one day, updating buy/pickup/place, feed/care/harvest and output disposition together. Crop jobs alter one complete post-144 planting/service/harvest lifecycle or its timing by at most one day, including seed financing, water, fertilizer where used, cargo and feed consequences. Use existing feasible sites/worker lanes. Apply the typed transformation to the relevant inherited route branches; missing/incompatible jobs make a proposal invalid, not a fabricated effect. Both arms have identical grammar, ranges, eligibility, compiler and executor. The contrast is the proposal prior, not a more powerful compiler for one arm.

**Arm R, 128 replay-informed proposals:** 64 single-job instantiations guided by the mined templates; 38 structured timing/quantity mutations or contract-compatible two-job recombinations; 26 proposals sampled without replay frequencies from the full shared support. Allocate 32/19/13 to each production family. Thus **20.3% deliberately ignores the replay prior**. Templates guide which feasible commitment to alter and its service/timing pattern, not episode-specific action coordinates or future-shop answers. At least half the prior-guided proposals must differ from the inherited production job in more than sale timing.

The finite proposal schedule and RNG are frozen before returns. Enumerate eligible jobs and fixed alternatives; sample 128 unique valid candidates per arm with a cap of 512 compilation attempts per arm. Deduplicate actual compiled closures, not template names. The pilot need not invent an arbitrary full-farm scheduler. It must change production, not only sale quantities. This fixed, non-adaptive first generation tests whether the replay prior is worth retaining before funding population infrastructure. Record acceptance rates and candidate diversity as diagnostics, not substitutes for fresh policy wins.

**Arm U, 128 controls:** uninformed structured production proposals, uniformly sampled from the same declared feasible job/parameter support, with no replay frequencies, nearest-example selection or historical outcome weights. Use 64 animal and 64 crop proposals, the same mutation/recombination scope and independent frozen proposal RNG. Neither arm contains local-repair-only candidates; historical CCPR remains diagnostic evidence, not a third arm. Give both arms identical evaluation budgets, charge failed simulation attempts/reference games, and keep zero-edit checks separate from discoveries.

This is a conservative **proposal-prior ablation**, not a claim that the control's shared grammar was developed without ever seeing public play. It measures whether replay-guided choices add value after equalizing compiler capability; annotation time is charged to the common engineering budget and reported separately. Keep the control's support/proposal schedule frozen before selecting replay job examples.

**Four native discovery opponents:** Shop Plus; direct Shop; a fixed route-3 continuation opponent; and a fixed route-12 continuation opponent. The latter two use the same legal common opening to 144 and live reactive execution thereafter, not recorded opponent actions. Their different production schedules stress supply, although they remain related ancestry. Declare this limitation rather than calling four entries four independent public families. Add no opponent training to this pilot.

### Exact approximate scale

| Work | Panel / count | Charged work |
| --- | --- | ---: |
| Fidelity and no-edit checks | Phase 0 and candidate-boundary fixtures | <=64 native FGE plus small official checks |
| Discovery roots | 16 independent seeds; each assigned to one of four opponents, both seats; checkpoint 144 | **32 roots**, one per trajectory; <=32 reference games for capture |
| Arm R screening | 128 candidates x 32 contexts; suffix 144-718 inclusive | **4,096 continuations**, 2,355,200 transitions |
| Arm U screening | Same | **4,096 continuations**, 2,355,200 transitions |
| Independent BUILD confirmation | Best 2 candidates/arm fixed from discovery; 32 new seeds x both seats x 4 opponents | **1,024 candidate games + 256 paired reference games** |
| Fresh internal DEV | Exactly 1 candidate/arm frozen before DEV; 64 new seeds x both seats x same 4 opponents | **1,024 candidate games + 512 paired reference games** |
| External-policy check | Exactly those two candidates and parent; 32 different seeds x both seats x 4 frozen external policies | **768 official-engine games**, CPU; no CUDA static-tape substitute |

The discovery suffixes are approximately **6,551.3 FGE at /719**. Captures, native confirmation, internal DEV and fidelity bring the native total to approximately **9,463 FGE**, leaving room inside the **12,000-FGE** cap for required exact checks and small overhead. Record actual work, not only this estimate. GPU wall cap is two hours; independent Python checks have a two-CPU-hour aggregate budget. If external runtimes cannot fit, stop with an explicitly incomplete generalization result rather than dropping that task or declaring success.

The external policies remain frozen **Gold Plus V8**, the inspected **Seyamalam V21-style route/prototype policy**, **COK V10 mixed-herd policy**, and **lonespear V11 reactive scheduler**, subject to public availability/license and source-identity checks. Do not use these source implementations or their evaluation returns to construct pilot proposals; quarantine positively matched replay lineages from job mining. Unknown replay ancestry cannot be certified absent, so this is explicitly an **external live transfer check, not four independent HOLDOUT families**. The reactive agent is not known top-tier. Freeze any legally necessary structurally comparable replacement before candidate outcomes. Later release requires genuinely fresh challengers; the pilot does not certify medal strength.

### One-shot decision

Choose the best confirmed complete policy in each arm using BUILD only. Report internal/external paired win utility, both seats, per-opponent results, own money/margin distributions, actual production quantities, failed jobs, overflow and unsold stock. Use seed-clustered intervals; four discovery opponents sharing ancestry do not make 32 independent strategic regimes.

**Pass the architecture's first gate only if all of the following hold:**

- Arm R's frozen winner gains at least **5 percentage points** of cohort-balanced fresh internal win utility over Shop Plus, with a **one-sided 90% seed-clustered lower bound above zero**.
- It gains at least **3 points** more internal utility than the frozen uninformed-production winner in point estimate. Report the uncertainty of this difference; do not claim universal algorithm superiority or measured search-space speedup from one generation.
- The external panel's mean paired utility is nonnegative, and no external-policy cohort has a point loss exceeding **10 points**. This small panel is an early generalization veto, not release certification.
- Whole-job ablation shows the improvement requires a changed productive lifecycle or labour schedule. On the preregistered eligible/activated cohort, median final cash-margin improvement is at least **1,000 coins** and the positive economic effect occurs in both seats and against at least two native opponent entries. The eligible cohort is defined from decision-time legal state, never by whether the game improved.
- The compiled candidate has zero semantic mismatches, runtime faults or catastrophic-firewall failures. No output is selected using hidden seed knowledge or future shops.

The 1,000-coin threshold is a deliberately material search target, not a demonstrated requirement for gold. One strong production mechanism is enough to justify the next phase; do not recreate CCPR's arbitrary requirement that two named mechanisms must survive a tiny initial test. Diversity is required in the subsequent population and challenge distribution.

**Replay-prior failure:** no eight auditable templates within the mining cap, raw trajectories fail contract translation, or Arm R misses the incremental +3-point gate. Remove replay proposal weighting rather than funding a bigger miner or a neural rescue. If at least one arm meets the absolute +5-point, uncertainty, external, economic-attribution and exactness gates, that supports production search without establishing the prior's added value: retain the successful candidate as evidence, proceed with the original unweighted route-search architecture, and use new replay material only for challenge/context analysis. Failure before a valid comparison produces an engineering verdict, not evidence that uninformed search won. A failed post-144 pilot does not prove that all historical openings are useless.

**Production-path failure:** neither arm passes those absolute gates, only terminal tie-breaking survives, or most jobs require a general planner. Use the one small fallback below. Do not rescue a failed result by shrinking cohorts, moving thresholds, querying more DEV candidates or raising the budget after outcomes. No production-search experiment, candidate compilation or simulation in this design was run during the replay review.

## Kill criteria

1. **Early representation/engineering failure:** both production arms fail the absolute gate, or a fair result requires more than four engineering days. A failed replay prior alone removes that module if either arm passes the absolute production gate; it does not invalidate production-route search. Do not build a large mined library/league first.
2. **No material economic discovery:** 100,000-250,000 additional FGE after a successful pilot produces only near-mirror sale tricks, no repeatable broader economy improvement, or an archive of effective duplicates.
3. **Live-opponent dependence failure:** candidate gains require recorded opponent actions, hidden-seed-conditioned route selection, unavailable future shops, or a single exact production fingerprint.
4. **Compilation failure:** exact routes plus bounded reactive behaviour cannot preserve the deployable teacher's gain; improving imitation accuracy is not enough if closed-loop returns regress.
5. **Simulation or legality failure:** any uncorrected semantic mismatch, unsupported engine version, private information dependency, prohibited technique or unresolved rules condition necessary for the claimed advantage.
6. **Competitive relevance failure:** two successive meaningful and adequately supported local policy gains do not transfer to later observed competition play. Stop claiming that the existing league predicts rank; refresh strong opponent coverage before another release.

**One small fallback, only after the production-path gate fails:** retain Shop Plus and run one bounded, ordinary-mechanics executor improvement, prioritizing capacity-safe partial deposits and projected feed preservation, with at most 32 overlay candidates and 5,000 native FGE plus the same small independent-source checks. Removing an unhelpful replay-prior module from otherwise successful route search is an ablation, not a second parallel architecture. No route league, learned policy, MCTS, solver platform or indefinite CCPR campaign remains active alongside the mechanics fallback. If that also fails, keep the existing deployable family and report that the proposed medal path lacks supporting evidence; do not manufacture another open-ended roadmap.

## Gold-medal path

The current evidence establishes a compact, strong local route-agent family and excellent tools for exact experimentation. It does not establish current top-10 proximity. The largest reachable improvement is to discover a meaningfully more productive economy, make it survive live responses and independent seeds, and deploy it without distillation loss.

The practical path is: **one cheap replay-prior/production-route comparison; one small repeated route-search league; one causal compiled portfolio; one statistically defensible release; then repeat against stronger challenges**. Historical jobs reduce blank-page proposal design; novel mutations and live counterfactual verification must provide the gain beyond the copied public meta. A two-week implementation target is plausible only if the four-day pilot works and native throughput is adequate. It is not a schedule guarantee, and the competition deadline must be checked from an authoritative current page rather than a cached countdown.

Judge progress in opponent-adjusted full-policy wins and economic capability, not controller nodes, number of specialists, perfect records against relatives or volume of experiment artifacts. A top-10 candidate must eventually withstand strong unseen economies and prospective competition evidence. The architecture earns further compute only by producing those signals.

**Strategic authority:** this document supersedes conflicting methodological recommendations in `docs/FIRST_PLACE_EXECUTION_PLAN.md` for future discovery work: route freezing, deterministic-only discovery, pure-internal training doctrine and universal parent-game monotonicity are no longer the proposed strategy. It does not change existing policies, protected datasets, historical verdicts, deployment assets or production enforcement. The exactness, legal-observation and provenance boundaries remain mandatory. Future implementation must explicitly reconcile governance before changing code.

## Sources and reproducibility notes

External sources below were accessed or checked on 2026-09-10 unless a different publication date is stated. Mutable public code is identified by the retrieved source hashes where available. Cached discussion dates/ranks are not treated as exact live leaderboard timestamps. Literature transfer and compute estimates are analytical proposals, not empirical Kaggriculture results.

### Competitive and engine sources

[^1]: yhay81, [Shop Router 0908](https://www.kaggle.com/code/yhay81/shop-router-0908), Kaggle public notebook, retrieved version 2, kernel version ID 133520239; [public source endpoint](https://www.kaggle.com/api/v1/kernels/pull/yhay81/shop-router-0908). Notebook source SHA256 `b64b16559c441d5ff48891629df6bef519427d4780aa7b97d306449a796aa606`. Why it matters: primary author description of daily/three-day route recombination, large candidate evaluation and the limitation of static opponent tapes.

[^2]: yhay81, [Shop Router 0909](https://www.kaggle.com/code/yhay81/shop-router-0909), Kaggle public notebook, retrieved version 3, kernel version ID 133670867; [public source endpoint](https://www.kaggle.com/api/v1/kernels/pull/yhay81/shop-router-0909). Notebook source SHA256 `a36029e34448e823b6b11990ad06a86e255da6b629cbfb30035b542422592dd4`; extracted `main.py` SHA256 `d6d74997dc5b483db63d8e39cafa1afeec0f366824e75107e109123f111e866b`. Why it matters: actual source underlying the repository's strongest current route executor, not just a notebook score/title.

[^3]: aurax7, [Reactive Router](https://www.kaggle.com/code/aurax7/reactive-router), Kaggle public notebook; [public source endpoint](https://www.kaggle.com/api/v1/kernels/pull/aurax7/reactive-router), retrieved version 1, kernel version ID 133599234. Notebook source SHA256 `0e8b6606bf7bc843a8b69fddc9357f8da11e0c38f52453481ef8265a033b56a1`; published main SHA256 `53070a6adfaac7b91914a53e215bf0c521fff1d633db74abdf20bfcd13190175`. Why it matters: route-plus-reactive architecture; interpret with the local [Gold provenance](../src/kaggriculture_bonsai/agents/gold_v25/PROVENANCE.md).

[^4]: Seyamalam, [Kaggriculture `main.py`](https://github.com/Seyamalam/Kaggriculture/blob/main/main.py), GitHub mutable source; SHA256 `0cd14b653102d276c4f902fa3b8c6bd81d869b8ab64c422cb881b9d2346ec639`. Why it matters: inspected board/market prototype selection, late lead-based abstention and recovery sale code. Source attributes Kaito Fukami's [V18 closed-loop notebook](https://www.kaggle.com/code/kaitofukami/40-53-top-10-future-holdout-v18-closed-loop?scriptVersionId=340030138); title is not rank verification.

[^5]: COK-ZhangZiliang, [Kaggriculture `main.py`](https://github.com/COK-ZhangZiliang/Kaggriculture/blob/main/main.py), GitHub mutable source; SHA256 `56831f3c43c9727d90016b7a7a8d4eb51d1a4c08c1120d58f061d9176e8bc109`. Why it matters: actual V10 route mixing and public-state confidence/fingerprint logic; README's historical V2 score is not current-code performance.

[^6]: lonespear, [Kaggriculture `main.py`](https://github.com/lonespear/kaggriculture/blob/main/main.py), GitHub mutable source; SHA256 `eb5b5f59a8ec2d40b77cc99d4ffe3b932136fdcf9f6b6e168726b7f07ab47cb0`. Why it matters: a genuinely reactive task-based economy and documented historical failures; not a verified strong leaderboard opponent.

[^7]: GzmCR, [Kaggriculture `main.py`](https://github.com/GzmCR/Kaggriculture/blob/main/main.py), GitHub mutable source; SHA256 `e5fe6dc3a24412fb856b4c3eed16202dd1d8c23b2a7342c2fa5a20bce309129b`. Why it matters: visible-opponent supply and marginal herd heuristics; embedded price-model version mismatch limits economic conclusions.

[^8]: sobameshi, [I came here to build a farming agent. I ended up building a replay-measurement pipeline](https://www.kaggle.com/competitions/kaggriculture/discussion/739273), Kaggle public discussion, September 2026 context. Why it matters: first-hand public round-robin and replay-diffusion observations; attributed evidence against assuming public-meta non-transitivity. Raw matrix not independently reproduced. Its descriptions of host permissions are secondary reports, not a substitute for full rules.

[^9]: Kaggle, [official Kaggriculture interpreter](https://github.com/Kaggle/kaggle-environments/blob/master/kaggle_environments/envs/kaggriculture/kaggriculture.py), public engine source, byte-equal to the installed 1.32.7 file at cutoff. Why it matters: authoritative executable order, economics, RNG, biology and terminal semantics for the audited public version.

[^10]: Kaggle host, [Small balance change](https://www.kaggle.com/competitions/kaggriculture/discussion/735311), August 2026 discussion. Why it matters: explicit version guidance and changed resource economics; not a promise that no future patch can occur.

[^11]: Kaggle environments, [PR #1399: market balance change](https://github.com/Kaggle/kaggle-environments/pull/1399). Why it matters: primary implementation context for making underused resources situationally valuable. Also inspect [environment specification](https://github.com/Kaggle/kaggle-environments/blob/master/kaggle_environments/envs/kaggriculture/kaggriculture.json), including seed privacy and action/runtime configuration.

[^12]: Kaggle environments, [framework `core.py`](https://github.com/Kaggle/kaggle-environments/blob/master/kaggle_environments/core.py), particularly `__agent_runner` and `__get_shared_state`; [public missing-step claim](https://www.kaggle.com/competitions/kaggriculture/discussion/737545). Why it matters: source and the four-transition local probe distinguish raw replay/state layout from the actual live-agent observation. The discussion claim was checked, not assumed true.

[^13]: Picu, [Reverse engineering the seed](https://www.kaggle.com/competitions/kaggriculture/discussion/739388), Kaggle discussion, September 2026 context. Why it matters: unresolved public question, not a working exploit or host authorization. Full [competition rules](https://www.kaggle.com/competitions/kaggriculture/rules) and [leaderboard](https://www.kaggle.com/competitions/kaggriculture/leaderboard) were attempted but their substantive live contents were unavailable; no rule-specific permission or current rank is inferred from them.

### Research and implementation sources

[^14]: Lanctot et al., [A Unified Game-Theoretic Approach to Multiagent Reinforcement Learning](https://papers.nips.cc/paper_files/paper/2017/file/3323fe11e9595c09af38fe67567a9394-Paper.pdf), NeurIPS 2017. Why it matters: foundational PSRO population/response-oracle formulation.

[^15]: Bighashdel et al., [Policy Space Response Oracles: A Survey](https://arxiv.org/abs/2403.02227), 2024; [IJCAI proceedings](https://www.ijcai.org/proceedings/2024/0880.pdf). Why it matters: explicitly includes planning/search/evolutionary response oracles and discusses strategy exploration limits.

[^16]: Google DeepMind, [OpenSpiel PSRO implementation](https://github.com/google-deepmind/open_spiel/tree/master/open_spiel/python/algorithms/psro_v2) and [algorithm documentation](https://openspiel.readthedocs.io/en/latest/algorithms.html), maintained public implementation. Why it matters: concrete reference for small payoff/mixed-opponent bookkeeping, not a required simulator migration.

[^17]: Vinyals et al. / AlphaStar team, [Grandmaster level in StarCraft II using multi-agent reinforcement learning](https://deepmind.google/blog/alphastar-grandmaster-level-in-starcraft-ii-using-multi-agent-reinforcement-learning/), Google DeepMind, 2019-10-30; Nature paper DOI `10.1038/s41586-019-1724-z`. Why it matters: league roles, frozen history and supervised initialization; scale and game differences prevent direct cost/performance transfer.

[^18]: Heinrich and Silver, [Deep Reinforcement Learning from Self-Play in Imperfect-Information Games](https://arxiv.org/abs/1603.01121), 2016. Why it matters: NFSP's separation of response learning and average behaviour; not evidence for raw-action Kaggriculture RL efficiency.

[^19]: Mouret and Clune, [Illuminating search spaces by mapping elites](https://arxiv.org/abs/1504.04909), 2015. Why it matters: behavioural archive rather than one incumbent; quality and diversity are distinct goals.

[^20]: Tjanaka et al., [pyribs](https://github.com/icaros-usc/pyribs), GECCO 2023 library paper and maintained implementation. Why it matters: practical archive/emitter interfaces and CMA-ME references; its focus on continuous vectors does not make raw tapes a suitable CMA genome.

[^21]: Adaptive and Intelligent Robotics Lab, [QDax](https://github.com/adaptive-intelligent-robotics/QDax), accelerated quality-diversity implementation. Why it matters: evidence that large batched QD is practical; rewriting the existing CUDA engine in JAX is unnecessary for this plan.

[^22]: Gaina et al., [Rolling Horizon Evolutionary Algorithms for General Video Game Playing](https://arxiv.org/abs/2003.12331), 2020. Why it matters: concrete forward-model sequence-search family and horizon/parameter issues. Related authors' [implementation and experiment links](https://gaigresearch.github.io/2020/06/15/gaina2020onlinerhea/) provide reproducible planning context.

[^23]: [Rolling Horizon Coevolutionary Planning for Two-Player Video Games](https://arxiv.org/abs/1607.01730), 2016. Why it matters: opponent-aware evolutionary forward planning rather than static single-agent trajectory optimization.

[^24]: Sutton, Precup and Singh, [Between MDPs and semi-MDPs: A framework for temporal abstraction in reinforcement learning](https://www.sciencedirect.com/science/article/pii/S0004370299000521), Artificial Intelligence, 1999. Why it matters: temporally extended actions with initiation/execution/termination semantics.

[^25]: Solar-Lezama et al., [Combinatorial Sketching for Finite Programs](https://people.csail.mit.edu/asolar/papers/asplos06-final.pdf), ASPLOS 2006; [Program sketching](https://link.springer.com/article/10.1007/s10009-012-0249-7), 2013. Why it matters: CEGIS/program sketching is broader than the repository's particular gate-repair experiment.

[^26]: Ellis et al., [DreamCoder: Growing generalizable, interpretable knowledge with wake-sleep Bayesian program learning](https://arxiv.org/abs/2006.08381), preprint 2020 / PLDI 2021. Why it matters: learned reusable abstractions can change the effective search language; not a reason to build DreamCoder before a two-job test.

[^27]: Ellis et al., [DreamCoder implementation](https://github.com/ellisk42/ec). Why it matters: actual program-learning implementation and complexity reference; not an immediate project dependency.

[^28]: Rusu et al., [Policy Distillation](https://arxiv.org/abs/1511.06295), 2015 / ICLR 2016. Why it matters: separate discovery policy from deployed policy, with fidelity needing measurement.

[^29]: Ross, Gordon and Bagnell, [A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning](https://proceedings.mlr.press/v15/ross11a.html), AISTATS 2011. Why it matters: DAgger addresses learner-induced distribution shift, relevant to routed/compiled students.

[^30]: Bastani, Pu and Solar-Lezama, [Verifiable Reinforcement Learning via Policy Extraction](https://arxiv.org/abs/1805.08328), NeurIPS 2018. Why it matters: VIPER's value-sensitive decision-tree extraction; examples do not prove a compact full-action Kaggriculture student exists.

[^31]: Bastani, [VIPER implementation](https://github.com/obastani/viper). Why it matters: original executable extraction reference; use concepts, not its old dependency stack wholesale.

[^32]: Wang et al., [EfficientZero V2: Mastering Discrete and Continuous Control with Limited Data](https://arxiv.org/abs/2403.00564), 2024. Why it matters: credible modern model-based learning alternative, with no demonstrated advantage over this already-exact Kaggriculture simulator.

[^33]: Romera-Paredes et al. / Fawzi, [FunSearch: Making new discoveries in mathematical sciences using Large Language Models](https://deepmind.google/blog/funsearch-making-new-discoveries-in-mathematical-sciences-using-large-language-models/), Google DeepMind, 2023-12-14. Why it matters: executable evaluators plus population-based program proposals.

[^34]: Google DeepMind, [FunSearch implementation](https://github.com/google-deepmind/funsearch), accompanying Nature publication. Why it matters: concrete evaluation/program-search reference, not evidence that LLM calls are needed for route mutation.

[^35]: AlphaEvolve team, [AlphaEvolve: A Gemini-powered coding agent for designing advanced algorithms](https://deepmind.google/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/), Google DeepMind, 2025-05-14. Why it matters: recent evolutionary program discovery with automatic evaluators; gains and budgets do not transfer directly to competitive games.

[^36]: Agarwal et al., Deep Reinforcement Learning at the Edge of the Statistical Precipice, NeurIPS 2021; [authors' `rliable` library](https://github.com/google-research/rliable), archived 2025-10-15. Why it matters: uncertainty and distribution-aware evaluation rather than small-panel point estimates; archive status checked rather than assuming current maintenance.

[^37]: Google, [OR-Tools CP-SAT solver documentation](https://developers.google.com/optimization/cp/cp_solver), current documentation. Why it matters: proven integer constraint solver for narrowly bounded schedule feasibility; not a complete exact model of the economic game.

[^38]: Ryo Hasegawa, [1st Place (previously) submission strategy discussion](https://www.kaggle.com/competitions/kaggriculture/discussion/736219), August/September 2026 context. Why it matters: first-hand concerns about game count, opponent draw and rating maturity. Author explicitly qualifies the model as an estimate; no matchmaking formula, submission limit or final-ranking rule is adopted from it as an official fact.

### Local evidence and audit limits

Reproduction anchors are the input Git commit, asset hashes above, the four self-improvement reports, Shop provenance, Gold Plus V8 release evidence, route-format/VM sources and native simulator/evaluator sources linked in their respective sections. Historical counters were inspected as compact evidence; large campaigns were not rerun. Public API source retrieval and static code analysis provided the external policy observations; the ignored local Shop notebook corroborates the port's provenance.

The original architecture review's only new game execution was the four-transition official PASS-agent clock/seed diagnostic described above. The replay continuation downloaded the pinned public archive, indexed all replay IDs/footers and parsed a bounded body sample; it ran **no games, candidate search or training**. Large compressed files and disposable parsing summaries remain ignored under `build/top10_archive_research/`; only the revised plan and compact evidence manifest are committed. No production agent, search implementation, champion, submission package or protected panel was changed; no competition submission or push occurred. Phase 0 and the revised immediate experiment remain unexecuted.

### Replay-continuation sources

[^39]: ashok205 / NNMax, [Kaggriculture Daily Top-10 Replay Archive](https://www.kaggle.com/datasets/ashok205/kaggriculture-top10-replay-archive/versions/19), Kaggle public dataset, version 19, updated September 9, 2026; [public metadata endpoint](https://www.kaggle.com/api/v1/datasets/view/ashok205/kaggriculture-top10-replay-archive). Why it matters: actual public replay bodies, index, schema-3 manifest and daily provenance, not an assumed 2-GB corpus. Measurements and pinned hashes are in the repository evidence manifest.

[^40]: ashok205 / NNMax, [top10-replay-dataset-archive](https://www.kaggle.com/code/ashok205/top10-replay-dataset-archive), public collector notebook; [source endpoint](https://www.kaggle.com/api/v1/kernels/pull/ashok205/top10-replay-dataset-archive), retrieved metadata version 18, source SHA256 `1e2b739d81864141700f2c8992181fc6f0b7d959b5abaf09e4c500f66dfa79bd`. Why it matters: inspected pair-average proxy ranking, both-team episode links, ZSTD row groups, append-only/cumulative update logic and lack of winner filtering. Mutable notebook metadata and old cell timestamps do not independently pin the collector source used for every historical shard.

[^41]: Kaggle, [Kaggriculture episodes index](https://www.kaggle.com/datasets/kaggle/kaggriculture-episodes-index), official dataset, retrieved version 42, July 30-September 9, 2026; [September 8 official daily archive](https://www.kaggle.com/datasets/kaggle/kaggriculture-episodes-2026-09-08). Why it matters: primary description of completed-game selection by average agent rating under a 20-GiB daily cap, and the upstream coverage boundary. The top-10 archive is a further selected subset, not the complete competition match history.

[^42]: Kipf et al., [CompILE: Compositional Imitation Learning and Execution](https://proceedings.mlr.press/v97/kipf19a.html), ICML 2019; [author implementation](https://github.com/tkipf/compile). Why it matters: primary evidence that variable-length demonstrated behaviour can be segmented and reused; the repository example is a simplified sequence task, not a ready-made production-job miner.

[^43]: Pertsch, Lee and Lim, [Accelerating Reinforcement Learning with Learned Skill Priors](https://proceedings.mlr.press/v155/pertsch21a.html), CoRL 2020 / proceedings 2021; [official SPiRL implementation](https://github.com/clvrai/spirl). Why it matters: skill-level proposal priors can guide exploration more effectively than flat action priors in the studied domains. Transfer to this game is an inference; it motivates the cheap macro-prior contrast, not immediate adoption of its RL stack.
