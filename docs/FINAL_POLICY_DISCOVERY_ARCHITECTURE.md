# Final Policy Discovery Architecture

## Executive verdict

**Build BKG4-projected hierarchical Expert Iteration: learned strategic proposal/action-value models, exact CUDA full-horizon counterfactual improvement of closed-loop economic plans, and cost-sensitive BKG4 extraction inside every iteration.**

**THIS IS THE ARCHITECTURE WE SHOULD BUILD NEXT.**

This is a learning-and-search architecture, not raw-action PPO, replay cloning, a world model, or another route-mutation campaign. The learner controls a state-dependent economic plan: investment, production mix, land, labour, service priorities, financing, and realization of cash. A deterministic, BKG4-compatible executive handles movement and transaction mechanics. Search changes coordinated multi-day plans from the opening onward; it does not preserve the inherited worker lanes, post-144 restriction, or one/two-lifecycle edit boundary. The exact simulator supplies outcomes. Small learned models generalize them. The actual BKG4 student supplies an increasing fraction of subsequent training states.

The recommendation is a judgment about **practical research upside**, not a measured probability of winning gold. Its central uncertainty is whether a sufficiently expressive executable abstraction and a small causal student can preserve strategic improvements. Test that uncertainty before building a full league or expensive RL stack.

**One first experiment:** three bounded improvement/projection rounds, compared with equally supported unguided planning, on a new executable economic-plan representation. Maximum **30,000 native full-game equivalents (FGE), eight GPU-hours, six aggregate CPU-hours, and six engineering days**. Include an early representation gate, fresh whole-game evaluation, two student-correction rounds, and real non-mirror economic evidence. An offline perfect-information winner or a mirror-only utility gain cannot pass.

**One fallback:** replay-initialized hierarchical recurrent PPO over the same *validated* executable plan interface, without a learned-search critic. It receives one fixed 20-million-transition trial only if the primary algorithm fails its scientific gate. A broken action interface, illegality, or irrecoverable extraction failure stops both methods; switching optimizers cannot repair those defects.

The previous [Final Gold-Medal Bonsai Plan](FINAL_GOLD_MEDAL_BONSAI_PLAN.md) was **experimentally exhausted at `4bcf1e78d404f236ba5c2cfdd9ce038be9948e7c`**. Its production-route pilot failed at `3ef51472eb33e713a2f7c403d6db7831a4662f09`; its single mechanics fallback failed at the audited HEAD. That document remains unchanged historical evidence. Neither experiment promoted a policy.

### Evidence and authority

Repository inspection began on clean `main` at **`4bcf1e78d404f236ba5c2cfdd9ce038be9948e7c`**, team **ReCurSiON**. Evidence cutoff: **2026-09-10**. This report distinguishes recorded measurements, externally reported evidence, architectural inference, and proposed budgets. All future learning curves, controller sizes, thresholds, and schedules below are proposals unless explicitly identified as measurements.

This is the authoritative **future discovery decision**, explicitly superseding conflicting method-selection recommendations in `FINAL_GOLD_MEDAL_BONSAI_PLAN.md` and `FIRST_PLACE_EXECUTION_PLAN.md`. It permits learned guidance and declared public TRAINING material, and replaces universal strategic win-to-loss prohibition with robust policy-level promotion. It does **not** alter production enforcement, `AGENTS.md`, protected datasets, frozen diagnostic baselines, or `assets/champion/`. The next implementation session must explicitly reconcile those governance conflicts before executing new training. No old protected panel becomes BUILD by renaming it.

No training, new simulator campaign, production implementation, packaging, submission, push, or champion modification accompanies this decision. Historical results were inspected, not rerun. Lightweight checks loaded compiled controllers, read hardware information, and retrieved public research/source metadata. SEALED payloads were not read.

## Why the previous plan failed

### Verified experimental sequence

| Experiment and committed evidence | Actual result | Supported interpretation |
| --- | --- | --- |
| [Phase 1 exact fork/repair](../workspace/self_improvement/phase1_fork_repair/README.md) | 1,464 matched continuations; uninterrupted/fork results agreed. 333,480 versus 1,069,872 turns on the matched work comparison, 68.83% saved. Four legalized survivors, representing two mechanics. | Exact full-horizon continuation works for the tested controller layouts. Search infrastructure was productive; strategic breadth was not established. |
| [Original CCPR](../workspace/self_improvement/phase2_ccpr/README.md) | Eight versus five safe useful BUILD patterns, but zero versus one DEV survivors; each checkpoint fit relied on three independent seeds. | Proposal efficiency did not imply generalization. Checkpoints are not independent games. |
| [Phase 2D](../workspace/self_improvement/phase2d_generalization/README.md) | The combined trajectory-diverse/guarded protocol improved prediction on ten fixed hypotheses from 4/10 to 8/10; one top-up survived. | Correlation and selection mattered. This did not isolate diversity from the changed guard/ranking, or prove general CCPR superiority. |
| [Corrected CCPR](../workspace/self_improvement/phase2_ccpr_corrected/README.md) | 35.28% fewer discovery turns; one sell-top-up family survived, +2,100 aggregate margin over 32 DEV games, ten T-to-W and no L-to-W. | A genuine bounded repair success, but the required breadth failed. The search never learned a new body or production abstraction. |
| [Replay-seeded route pilot](../workspace/gold_route_search/phase01_replay_seeded/README.md), [machine-readable verdict](../workspace/gold_route_search/phase01_replay_seeded/summary.json) | 128 informed versus 128 uninformed candidates; neither arm had a discovery utility improvement. 52/128 closures overlapped. Byte-identical winners; DEV -16.40625 points, median paired margin -343, 68 W-to-L and 86 T-to-L in 512 games. Transfer mean -21.484375 points. | Reweighting a narrow support did not create a better economy. This was not a full population-learning experiment and not a test of all possible route search. |
| [Mechanics fallback](../workspace/self_improvement/mechanics_fallback/README.md), [machine-readable verdict](../workspace/self_improvement/mechanics_fallback/summary.json) | 32 overlays. No reference overflowing DROP opportunity on the diagnostic panel; feed protection damaged sell-and-buy-back financing. Winner DEV +10.3515625 points, entirely 106 mirror T-to-W; mean paired margin -2.3965. Every transfer cohort had zero utility benefit. | Correct mechanics can optimize an irrelevant weakness. Utility can amplify economically tiny changes in a tie-heavy test population. |

The route pilot's production-versus-sales-only ablation showed a +15.625-point *relative* production effect, but the complete modification still lost. That supports the importance of production; it does not establish that the proposed new production was good. Both route/mechanics experiments recorded zero CPU/CUDA and official/native mismatches. Their transfer opponents were frozen local substitutes with shared ancestry, **not four independently sourced hidden leader policies**.

### Strongest diagnosis

**High confidence:** the tested discovery representations could not reorganize enough of the economy, and the opponent/objective design made local effects look more important than they were. Corrected CCPR controlled WHEN/STOP around fixed interventions. Route proposals changed one or two acquisitions/lifecycles after step 144, with inherited sites and service lanes, at most one-day compatible retiming. Mechanics preserved the entire economy. None tested coordinated opening finance, land acquisition, workforce, spatial layout, and subsequent production under a genuinely new closed-loop policy.

**Moderate confidence:** the public economy is locally well optimized within those representations. Many route proposals had real and different economic effects, but mostly negative outcomes. Globally coordinated change plausibly requires crossing temporary cash or payoff valleys. Exact terminal evaluation solves *measuring* delayed consequences; it does not solve *proposing* coordinated decisions or generalizing the winning choices.

**Not established:** that all deterministic search is exhausted; that generic RL has a low intrinsic ceiling; that a single optimum defeats all opponents; that online opponent adaptation is essential everywhere; or that a new economy must require a neural runtime. There is no current, adequately documented, like-for-like raw-RL versus macro-RL versus search benchmark in the inspected active evidence. Earlier generic-RL impressions cannot substitute for one.

### Hypothesis discrimination

| Hypothesis | Evidence strength | Consequence for the next experiment |
| --- | --- | --- |
| Wrong representation / tightly coupled inherited schedule | Strong local evidence | Permit coordinated opening and spatial/labour changes; retain primitive support beneath macros. |
| Poor long-horizon credit | Structural engine evidence; no isolated learning ablation | Use undiscounted terminal continuations; search at least three successive planning decisions together. |
| Narrow opponents / misleading objective | Direct mirror and shared-ancestry evidence | Separate mirror results, score economic materiality, evaluate fresh live non-mirrors. |
| Replay correlation / public-meta bias | Audited coarse-regime concentration | Small balanced pretraining, no mandatory imitation constraint, novelty floor. |
| Need opponent adaptation | Visible public routers; conflicting public payoff studies | Include legal opponent features, but require a feature ablation before retaining complex memory/routing. |
| BKG4 itself limits policy quality | No evidence for current policies; untested for broad teacher | Test extraction at the first loop, not after a successful expensive teacher campaign. |
| Safety rules reject useful changes | Phase 2D rejects some positive DEV shadows; not causal proof of overall failure | Permit ordinary tradeoffs in discovery; keep semantic constraints absolute. |

## What this teaches us about policy discovery

The unit of progress must be a **closed-loop economic capability**, not a predicate, route hash, or candidate count. A new production profile is necessary diagnostic evidence but insufficient: the previous route pilot had over 100 distinct profiles per arm and no utility winner.

Three different bottlenecks must be measured independently:

1. **Reachability:** can the executive express and execute the proposed economy without inherited lane/tape assumptions?
2. **Improvement:** can the teacher find a materially better policy at a fixed compute budget against live opponents?
3. **Projection:** can the BKG4 student retain that advantage on states it actually visits?

A learner is valuable if it shares economic knowledge between different states and guides coordinated alternatives, not if it predicts the score of already-known losing route mutations. A search oracle is valuable if its outputs define legal policies, not if it chooses a different answer for each hidden seed. A league is valuable if it creates useful response pressure, not if it contains many wrappers around the same opening.

## Kaggriculture-specific learning/search challenges

The inspected [official schema](https://raw.githubusercontent.com/Kaggle/kaggle-environments/master/kaggle_environments/envs/kaggriculture/kaggriculture.json) specifies two players, 10x10 farms, 720 recorded steps, 24 turns/day, ten market orders/turn, a 100-unit shed, and final-money reward. The pinned native runtime executes actions 0 through 718. Public farms and markets are visible; only the acting player's private shed, seeds, and inventories are exposed. The seed is deliberately removed from agent configuration. These are runtime facts, not assumptions from replay contents.[^engine]

The native [CPU engine](../native/cpp/include/sim.hpp) and [CUDA engine](../native/cuda/include/sim.cuh) resolve ordered worker operations and then joint market slots. Simultaneous planting can fail through shared seed constraints; same-turn sales cannot retroactively free room for a preceding deposit. Hands expire daily. Crop maturity, animal feed/care, hiring finance, capacity, and prices couple otherwise separate actions. Raw actions are therefore not independent per-worker classification targets.

At 13 actors, even a toy choice of ten applicable actions each gives `10^13` joint combinations before items, quantities, sites, or market permutations. This is illustrative branching arithmetic, not a count of all legal actions. Independent action masks cannot enforce every joint resource constraint. An autoregressive decoder needs a provisional resource ledger and the same order as the engine.

An investment returning 500 turns later receives only `0.99^500 ~= 0.0066` of its terminal contribution under a naive per-turn discount. Undiscounted episodic returns avoid that distortion but not variance. Daily or three-day options reduce decision frequency, yet must preserve obligations beyond their nominal duration. A dawn reset of hands does not reset crops, livestock, cash, or market effects.

### Action representations compared

Branching ranges below are design estimates, not measured search statistics. "Compiler" includes feasibility and recovery, not just encoding bytes.

| Representation | Branching / decision horizon | Credit and compiler difficulty | New economies / BKG4 fit |
| --- | --- | --- | --- |
| Raw unit operations | Roughly 5-30 choices/actor before arguments; 719 ticks | Extreme joint credit; low syntax, high joint legality | Maximum primitive freedom; extracting coordinated 719-tick behaviour is hardest. |
| Ordered market actions | Up to ten categorical/quantity positions per tick | Moderate syntax, severe financing/price coupling | Cannot invent a farm economy if production is fixed; native intents fit. |
| Worker-task assignments | Tens to hundreds of tasks per free worker; event-driven | Assignment coupling remains; deterministic navigation reduces waste | Expressive scheduling; bounded target/task state fits BKG4. |
| Per-day plans | High-dimensional vector; about 30 decisions | Must coordinate shared cash, land, and labour; significant executive work | Strong compromise; not an open-loop day tape. |
| Production jobs | Tens/hundreds of site/type/timing choices | Dependencies and service obligations; prior grammar was too restrictive | Useful execution units, not fixed policy choices or fixed inherited lanes. |
| Economic target macros | Factorized categorical targets; 10-60 decisions | Long returns remain, but complete terminal labels are cheap enough to test | Can coordinate expansion/mix/labour; executable semantics must earn trust. |
| Options | Small sampled initiation/parameter choices; variable duration | Initiation/termination/interrupt rules explicit | Excellent if closed-loop and independently reusable; bad if renamed tapes. |
| Route segments | Tens/hundreds of fragments; 10-30 boundaries | Entry/exit contracts often incompatible | Strong execution precision, poor adaptation after an unplanned prefix. |
| Full routes | One huge object; 719-row coupling | Easy decoding, extremely hard global proposal | Exact deployment, but selecting existing routes cannot invent a new economy. |
| Latent learned skills | Continuous or large code space; duration learned | Expensive skill discovery; decoder and safety harder to certify | Potential abstraction discovery; a neural skill decoder does not disappear when its selector becomes a tree. |
| BKG4 subprograms | Enormous typed syntax space; finite-state horizon | Exact execution, difficult nonsmooth structure search | Direct deployability; unrestricted primitive support can escape repair vocabulary. |
| **State-conditioned economic plans + executable task options** | **Sample 32 complete plans; daily/event execution, strategic reconsideration every 72 turns initially** | **Learn joint plan proposals; evaluate to termination; reuse one deterministic executive** | **Selected: changes opening, targets, timing, spatial assignment, and later responses while keeping an extractable policy interface.** |

Do not independently train one competing RL agent per hand. The farm is one centralized decision maker with ordered sub-actions; multi-agent learning applies primarily to the two competing farms. Cooperative-MARL factorizations are informative, but their decomposition assumptions do not automatically hold for shared financing and capacity.[^combinatorial]

## Replay corpus implications

The [committed audit](../workspace/research/top10_replay_archive/evidence.json) established **16,168 unique episodes**, 41 daily shards spanning July 30 through September 8, and **2,213,153,054 downloaded bytes** in version 19. A current public metadata query again returned version 19, updated September 9 at 23:19:37.64 UTC, CC0. Its listing-level `totalBytes` is 2,200,387,730, different from the audit's file-summed downloaded bytes; use the pinned manifest/file sizes for accounting, not an assumption that metadata fields measure the same thing.[^archive]

Full replay column chunks expand to approximately **495.9 GB**. Stream selected row groups; never materialize the full decoded corpus on this 16-GB machine. The audited body sample comprised 348 episodes, with 211 compatible 1.32.7 examples and 137 older configurations. Across 527 proxy-selected player trajectories, 521 action hashes coexisted with an effective coarse economic-regime count of only 2.927. Proxy-top10 selection uses pair-rating evidence, not verified leaderboard rank or independent generating-policy count.

**Selected use:** small causal pretraining for economic-plan proposals and public supply representations; operational sequencing examples; rare-state coverage; and diagnostic challenge descriptors. The archive is not the reward oracle and not a source of restorable private controllers.

For the pilot, use at most **64 current-configuration episodes**, balanced by day and realized economic profile, both seats and both outcomes. Keep at most 20 day/decision records per player: **at most 2,560 records**. Extract only observed acquisition/planting/land/hiring/order transitions. They are partial labels for plan components; missing intent, future reserve targets, and unobserved cancelled jobs remain unlabeled. Do not invent a unique latent macro label from a complete future trajectory.

Align the next stored action with its pre-action observation. Mask opponent-private fields, episode/team/submission IDs, seed, future shops, future observations, and terminal results from policy inputs. Future public supply can be a supervised auxiliary *target*, never an input. Engine/configuration eligibility is checked before deriving economic labels. Group all samples from an episode together and deduplicate close action prefixes; cap source-family weight where ancestry is known. Unknown ancestry prevents claiming source-independent generalization.

Replay BC initializes only the proposal distribution. Exact online BUILD outcomes replace replay preferences. Replay auxiliary loss decays to zero by the second improvement round; at least half of initial search proposals ignore the learned replay prior. No permanent KL tether to public behaviour. Historical losing actions remain useful evidence about feasibility and distribution, not recommendations.

Decision Transformer, masked action prediction, next-macro prediction, and strategy embeddings are plausible pretraining tools. Return conditioning cannot manufacture unsupported counterfactuals, and a logged high score may reflect weak opposition or obsolete economics. IQL's supported-action improvement is a serious offline alternative, but exact fresh interventions offer a cleaner source of new-action value here. We therefore do not fit a replay-only bootstrapped Q-function or a full trajectory transformer as the primary learner.[^dt][^iql]

## Exact simulator implications

### What is already available

Phase 1 snapshots are **49,488 bytes** for its fixed Shop controller layout. They own game configuration/state, both complete controllers, queue positions, advanced-sale state, and inventory insertion order. CPU/CUDA logical hashes and full-horizon restoration checks passed on the recorded fixtures. The engine regenerates daily random draws according to the modified state rather than replaying future weeds or shops.

The [Phase 1 implementation](../native/cuda/src/phase1_fork_repair.cu) explicitly captures both controller states. In contrast, the generic [policy snapshot runner](../native/cuda/include/policy_runner.cuh) resets policies from observations by default, and [Python snapshot candidate helpers](../native/cpp/src/pybridge.cpp) include tape-based continuations. Neither is a drop-in exact continuation for a new learned/FSM executive. Generalize the explicit state-capture contract, not the assumption that a simulator snapshot alone is sufficient.

The [device observation view](../native/cuda/include/observation.cuh) is a useful existing boundary: it exposes public farms and only own private inventory. A new teacher/executive must consume that legal domain, not `Sim::State` directly. Native CPU and the official Python engine remain reference checks; large training/search batches remain CUDA.

### Exact dynamics are not an information-state oracle

A snapshot contains the opponent's private inventory, controller memory, and actual RNG seed. The exact simulator is a training evaluator, **not a license to deploy a clairvoyant planner**. These are different quantities:

```text
bad target:     argmax_a R(full_snapshot, a), independently for each hidden world
desired value:  E[R | legal history summary x, executable plan a, training mixture]
```

Primary design: force candidate plans in fresh training snapshots, record terminal outcomes, and regress their **expected** value from legal inputs. Candidate programs and their future branches read only legal observations. Cross-fit by whole seed before using the critic to choose a plan on that seed. Confirm a shared causal policy over fresh games; do not average per-world oracle maxima and call that a policy score.

For offline beam selection, every candidate is the same contingent program across its root group, and is ranked by aggregate returns before any per-root label is chosen. Grouping is a fixed BUILD abstraction, not a claim of exact Bayesian belief matching. Keep individual forced-action outcomes as noisy regression targets. This avoids needing an exact posterior over invisible private controllers in the first experiment.

A later online information-set MCTS would need a validated belief sampler conditioned on legal history, not arbitrary hidden-state replacement or knowledge of the true future seed. POMCP and work on strategy fusion explain why this extra step matters. We deliberately do **not** make online belief-tree search part of the selected deployment or pilot.[^pomcp][^fusion]

Exact dynamics do not make model-free RL pointless: direct policy gradients may use compute better when tree branching, model calls, and projection dominate. They do make learned *dynamics* an unattractive first investment. Learn policy, action value, and proposal quality, not a second approximate biology/market simulator.[^ez2][^tdmpc]

## BKG4 deployment implications

### Verified assets and actual headroom

| Asset | Nodes / BB / node state | Controller bytes | Role |
| --- | ---: | ---: | --- |
| `workspace/shop_router_0909/plus/controller.bkgc` | **85 / 2 / 0** | **3,224** | Initial competitive reference, not authoritative champion |
| `workspace/shop_router_0909/port/controller.bkgc` | 61 / 1 / 0 | 2,376 | Frozen direct public port |
| `gold_v25` recorded direct asset | 23 / 3 / 0 | 1,028 | Alternative public-derived economy; provenance inspected |
| `assets/champion/controller.bkgc` | **398 / 26 / 0** | **15,798** | Only authoritative repository champion |

The Shop and champion controller counts/bytes above were independently loaded during this review. Shop Plus SHA256 is `84bbfe9a32ccf4d202664500b28bba787dab18cb18180c539d70f1890b1cc450`; champion SHA256 is `36e5e83e5240d2371f19f0006107990e0be44e5f6de9d31ec8f9c2c59d41a5b2`. The Shop route pack remains 13 x 719, 878,770 bytes, as verified in the latest experiment manifests.

The [BKG4 ABI](../native/bkg4/include/bonsai_kg_v4_abi.hpp) supports reactive/stateful selectors and sequences, conditions, enum switches, blackboard set/add, lifecycle effects, waits/timeouts, movement, and unit/market intents. Limits include 65,534 nodes, depth 48, 128 BB slots, 512 node-state slots, 40 units, and 16 market intents. The official default accepts only ten market orders. VM limits are not sensible target sizes or competition performance guarantees.

**Critical adapter gap:** [Shop native execution](../src/kaggriculture_bonsai/agents/shop_router_0909/reference_port/native.hpp) exposes only STEP and two SHOP_AT arguments to its route controller, allocates two BB slots, discards emitted VM intents, and uses 13 actor slots. It cannot execute a general learned controller merely by loading a larger `.bkgc`. The new candidate needs a separate sized runtime/observation/action adapter. Do not modify the frozen port to make this true retroactively.

The existing [BKV6 route record](../src/kaggriculture_bonsai/runtime/route_format.py) is 94 bytes and carries 12 saved hands plus the farmer. The archive observed up to 15 actual hands. For this design, additional workers execute BKG4 task intents, not silently truncated route records. Keep the old route codec and VM ABI unchanged unless a separate measured requirement justifies a later format revision.

### Distillation influences the winner

The selected teacher and student share exactly the same executable economic-plan schema, task options, legal features, explicit commitments, and integer executive. Only high-level selection/proposal is learned. The student need not approximate thousands of raw worker/market decisions or reconstruct a neural skill decoder.

A tree over fixed route IDs is easy but too restrictive. A finite-state selector over parameterized economic plans can set crop/herd targets, allocate cash and workers, change service priority, and revisit these choices using current observations. Route packs remain useful for a certified unchanged opening or action subsequences whose entry state matches exactly; they are not the whole policy language.

A new plan is **not** allowed to resume the old route blindly after changing the farm. Safe fallback after divergence means servicing the assets actually present and using valid task options. Returning to a tape requires its full entry contract, including layout, age, inventory, workers, and executor memory. The current parent may be shadow-ticked for attribution, but its stale actions are not a universal recovery strategy.

## External research findings

### Findings that change this decision

| Work / primary implementation | Architectural consequence |
| --- | --- |
| Expert Iteration, Anthony et al. (2017) | Separate planning from generalization and repeat the improvement loop. Search must create a teacher-apprentice gap, and extraction must preserve it; neither can be assumed.[^exit] |
| AlphaZero; Gumbel planning; Sampled MuZero | Use learned proposal priors and sampled action subsets. Low-budget search can fail to improve; exact action values are a condition, not a guaranteed property of a learned critic. Do not import alternating perfect-information minimax into simultaneous farming.[^az][^gumbel][^sampled] |
| Options; CompILE; SPiRL | Make initiation, execution, and termination explicit. Sequence priors can help exploration, but learned segmentation/skill embeddings do not establish executable resource contracts.[^options][^compile][^spirl] |
| SkillTree (2024 preprint / AAAI 2025) | A tree at the strategic level is plausible. Its neural low-level skills are not automatically BKG4-compatible; keep our executive symbolic from the beginning.[^skilltree] |
| PPO; IMPALA/V-trace; R2D2; A3C/A2C lineage; SAC | These address optimization, stale actors, recurrence, or replay efficiency, not the choice of economic actions. A2C offers simplicity, IMPALA useful decoupling, R2D2 memory with replay-state complexity. SAC is better matched to continuous control than discontinuous integer purchase bundles.[^ppo][^impala][^r2d2][^a3c][^sac] |
| Decision Transformer; IQL | Replay-first learning is credible, but offline support and changing opponents limit extrapolation. Use online counterfactual labels and a removable prior rather than treating historical success as a causal value target.[^dt][^iql] |
| PSRO; 2024 survey; AlphaStar; NFSP; PBT | Retain historical opponents and explicit response pressure. A small restricted payoff matrix is useful; full AlphaStar scale, concurrent neural populations, and automatic hyperparameter tournaments are not needed.[^psro][^psrosurvey][^alphastar][^nfsp][^pbt] |
| DAgger; VIPER; policy distillation | Query the teacher on student-induced states and weight consequential errors. CART extraction is empirical, not a theorem that any teacher admits a compact tree.[^dagger][^viper][^distill] |
| PIRL; PROPEL | Teacher-to-program projection should happen during improvement, not only at the end. Their assumptions and control-domain results do not prove convergence for this finite, partly observed policy class.[^pirl][^propel] |
| Deep surrogate-assisted MAP-Elites; SAIL; pyribs | Learned ranking plus exact evaluation is a real alternative. It needs an expressive genotype and evolving data; putting a surrogate over failed local closures does not enlarge their support.[^dsame][^sail][^pyribs] |
| DreamCoder; FunSearch; AlphaEvolve (2025) | Search can discover reusable program structure, not only parameter values. Keep compositional primitive support and later compress repeated successful subprograms; do not start with a new LLM service or assume code evolution is a proven farming teacher.[^dreamcoder][^funsearch][^alphaevolve] |
| iCEM; TD-MPC2; EfficientZero V2 | Temporally coherent sampling and learned guidance are attractive. Learned dynamics and continuous-control speedups are unnecessary/unproven here; terminal exact rollouts remain authoritative.[^icem][^tdmpc][^ez2] |
| HOP (ICML 2024); POMCP; strategy-fusion work (2024) | Public-history opponent modelling is plausible; exact hidden snapshots are not legal information states. Avoid identity labels and belief-free perfect-information planning.[^hop][^pomcp][^fusion] |
| Statistical Precipice / rliable | Use paired uncertainty, distributions, and predeclared comparisons rather than the best of many noisy candidate means.[^rliable] |

Official implementations inspected include `google-deepmind/mctx`, OpenSpiel's `psro_v2`, `obastani/viper`, `clvrai/spirl`, and the surrogate-generation repository `icaros-usc/dsage`. The Hearthstone-specific source is **EvoStone2**, not DSAGE. These are algorithm/reference code, not drop-in dependencies. `mctx` requires JAX-compatible transitions; rewriting the exact CUDA engine merely to use it would discard a major existing asset. Use the established search/projection methods with a narrow native batch interface.[^mctx][^openspiel][^vipercode][^spirlcode][^dsage][^dsame]

The recent literature changes the component choices, not the evidence standard. No retrieved 2024-2026 work establishes a Kaggriculture-specific teacher, a compact-lossless BKG4 extraction guarantee, or a one-laptop route to AlphaStar-scale performance.

### Current public competitors and conflicting evidence

The public Shop Router 0908 notebook was freshly retrieved through its source endpoint: version 2, kernel version ID 133520239. The author describes 1,326 public histories, 1,313 unique sequences, daily/three-day recombination and approximately 21 million candidate-game comparisons in a refinement stage. The author explicitly warns that fixed opponent tapes do not model adaptive opponents. This is first-hand evidence that substantial offline strategy construction exists; it is not independent verification of that search count or private leaders' algorithms.[^shop]

A public PPO notebook was also retrieved: Alejandro Fonda's `kaggriculture-ppo-training`, version 20, kernel version ID 129745065. Its visible wrapper performs 20 BC episodes, then 300 PPO episodes against `random`, repeatedly evaluating eight seeds against `random`. It imports `PPONet` and update logic from a separate dataset. We did not execute it or inspect those hidden-to-this-review implementation details. This shows an actual learned training attempt, **not evidence of strong competitive raw-action RL**.[^publicppo]

Two particularly relevant first-hand discussions conflict in a useful way:

- dzjiann reports **960 games among six public strategies**, a three-policy cycle, and a plateau when PPO selects/switches existing trajectories. Its authorship/source report supports route-prefix constraints and matchup interactions as plausible limits. Identity-conditioned diagnostic accuracy is not a legal runtime feature or a proof of Nash equilibrium.[^metappo]
- sobameshi reports a **14-implementation, 96-seed, both-seat** comparison with no intransitive triple and 86/91 chronological pairs favouring the newer implementation. The source describes replay diffusion and repeated replacement by better economies. This supports a productivity ladder in that sample, not universal transitivity.[^replayladder]

Different policy populations, periods, and evaluation methods can produce both results. Do not average them into a universal claim about the meta. Measure the new live payoff matrix, preserve broad productive strength, and retain a bounded hard-opponent component.

The public "Is Pure Self-Play PPO viable?" thread contains both weak pure-self-play reports and a hybrid rule/PPO income claim, without a controlled strong-opponent comparison. A separate BC/PPO discussion reports worsening after additional learning. They motivate controls for throughput, initialization, and distribution shift; they do not falsify Kaggriculture-specific RL.[^ppothread][^bcnoise]

Current GitHub source was revisited for Seyamalam's replay/closed-loop selection family, COK-ZhangZiliang's route/meta-market policy, and lonespear's task scheduler. Their visible code establishes different execution structures, not current ranks. COK's live entry-point docstring still says V9 while surrounding historical documentation discusses later variants, another reason to pin bytes rather than titles. A repository topic named `reinforcement-learning` is not proof that its deployed policy was trained with RL.[^seyam][^cok][^lonespear]

**Access limits:** some Kaggle pages returned empty shells on direct access but had substantive indexed discussion text; notebook source APIs and public GitHub raw source were accessible. Current full rules, directly attributable host replay-permission statements, definitive deadline details, authenticated ReCurSiON submissions, and current leader ranks were not independently established. Do not infer private-team training methods, a guaranteed remaining schedule, or rules clearance from cached ranks/countdowns. Verify those release prerequisites through authoritative competition access before using the plan for a submission.

## Architecture candidates

### Serious alternatives

| ID | Architecture | Concrete learner / improvement mechanism | Main failure mode |
| --- | --- | --- | --- |
| A | Raw recurrent self-play RL | Entity encoder + GRU + autoregressive worker/market decoder, legal ledger, PPO or V-trace, episodic terminal utility | Must jointly discover navigation, resource schedules, and 100-500-turn investments; difficult raw-action extraction. |
| B | Replay-pretrained hierarchical RL + league | Small recurrent actor/critic over executable plan factors; masked BC then semi-MDP PPO; frozen opponent mixture | Macros may impose a ceiling; on-policy exploration and replay anchoring can plateau. **Fallback.** |
| C | **BKG4-projected exact Expert Iteration** | Small proposal/Q network; full-horizon contingent-plan beam improvement; cost-sensitive projection and student roll-ins | Executive expressiveness, biased critic targets, and projection loss. **Selected.** |
| D | Surrogate-assisted policy evolution/QD | Evolve complete contingent economic-policy genomes; online ensemble surrogate; exact confirmation; behaviour archive | Needs a useful genotype and enough truly different policy evaluations; surrogate can amplify blind spots. |
| E | Offline macro planner without learned guidance | Information-aware beam/CEM or macro-MCTS; terminal live rollouts; fit BKG4 policy from decisions | Repeats search across states, expensive branching/belief handling; lacks amortized generalization. |
| F | Direct learned/evolved BKG4 | Typed subtree crossover, threshold/enum/FSM mutation, optionally relaxed tree training then hardening | Exact deployability, but hard structure credit and combinatorial syntax search can recreate a narrow grammar. |
| G | Replay policy induction + counterfactual relabeling | Next-plan BC or small strategic transformer; tree extraction; exact one-step option reranking and DAgger | Efficient warm start, but logged support and single-step improvement can remain in the public basin. |
| H | Neural-guided abstraction/program synthesis | Learn a recognizer over typed executable programs; solve training tasks, compress reusable subprograms, repeat | Highest representation-development overhead; learning a new DSL may consume the available engineering window. |

### Decision scores

Scores are ordinal architectural judgments, **not measured algorithm results**. Five is favourable. In risk columns five means low risk; in effort/compute columns five means cheap/simple. No invented probability of gold or overly precise weighted total is assigned.

| ID | Strategic upside | Game fit | Long-credit handling | Sample efficiency | Engineering | Compute affordability | Replay value | Exact CUDA reuse |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A | 5 | 2 | 2 | 1 | 1 | 1 | 3 | 3 |
| B | 4 | 4 | 3 | 3 | 3 | 3 | 5 | 4 |
| C | 5 | 5 | 5 | 4 | 3 | 4 | 4 | 5 |
| D | 4 | 4 | 4 | 3 | 3 | 4 | 3 | 5 |
| E | 4 | 4 | 5 | 2 | 3 | 3 | 3 | 5 |
| F | 4 | 3 | 2 | 2 | 3 | 4 | 2 | 5 |
| G | 3 | 4 | 3 | 4 | 4 | 4 | 5 | 4 |
| H | 5 | 3 | 3 | 2 | 1 | 2 | 4 | 4 |

| ID | Avoid local basin | Avoid public imitation | Plausible beyond Shop | Easy distillation | BKG4 fidelity potential | First decisive signal |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| A | 3 | 5 | 3 | 1 | 2 | 3-6 engineering weeks |
| B | 3 | 2 | 4 | 3 | 4 | 2-3 weeks from scratch; shorter after C infrastructure |
| C | 4 | 4 | 4 | 4 | 5 | <=6 engineering days for bounded pilot |
| D | 3 | 4 | 3 | 4 | 5 | 1-2 weeks |
| E | 3 | 4 | 3 | 3 | 4 | 1-2 weeks |
| F | 2 | 4 | 3 | 5 | 5 | 1-2 weeks |
| G | 2 | 1 | 2 | 3 | 4 | 4-8 engineering days |
| H | 4 | 4 | 3 | 3 | 4 | 3-6 weeks |

### Distillation considered before selection

| ID | What BKG4 would need | Fidelity / shift correction / size risk |
| --- | --- | --- |
| A | Raw action trees plus a discretized recurrent memory, per-worker choices and ordered-market decoder | No credible compact-fidelity estimate; potentially tens of thousands of nodes. DAgger alone cannot reproduce arbitrary continuous memory. Quantize/extract memory and revalidate full games. |
| B | Plan-selection tree/FSM and the same task executive | Explicit commitments make state extractable; GRU needs removal or finite-state approximation. Target <=4,096 nodes; weighted DAgger before any promotion. |
| C | Cost-sensitive plan tree, exact executive, bounded explicit history | Best controlled projection risk: teacher and student share the executable action interface, and every round is tested after projection. Initial target <=2,048 nodes, hard pilot cap 4,096. |
| D | Native/BKG4 contingent genome or induced policy tree | Near-exact if genotype already uses typed BKG4 subprograms; otherwise same state-shift risk as C. Target 1,000-4,096 nodes. |
| E | A causal state-to-plan student, not a bag of per-seed trajectories | High planner gap may be impossible to retain; future observations/beliefs must be available to student. Target <=4,096 nodes is a gate, not a guarantee. |
| F | The learned program itself | No teacher/student loss, but relaxed-to-hard decisions can change behaviour. Quantized final program requires CPU/CUDA and whole-game checks; target 512-4,096 nodes. |
| G | State-to-plan tree with small explicit history | Moderate extraction cost; high unseen-state failure if trained only on replays. Live counterfactual DAgger mandatory; target <=2,048 nodes. |
| H | Learned abstractions lowered to typed BKG4 subtrees with bounded memory | Compiler validation and whole-program verification dominate. No trustworthy size estimate before synthesis; cap at VM-compatible practical budgets. |

All candidates need legal observation-only features, sized controller snapshots, deterministic integer deployed decisions, and identical CPU/CUDA semantics. These properties do not follow merely from naming a distillation algorithm.

Direct learning of BKG4 contents is genuinely possible. Fixed-tree thresholds and plan constants could use CEM or policy-gradient training of soft branches, followed by hard integer projection; route/option selectors could use categorical actor-critic updates; explicit FSM transitions could be learned from aggregated teacher trajectories. Learning subtree structure or using RL over program edits additionally requires a typed construction grammar, delayed whole-program credit, and control of invalid/redundant programs. A relaxed branch's gradient is not the derivative of the exact hardened VM. These approaches avoid an external neural runtime, but fixing today's tree/route vocabulary would recreate the demonstrated support restriction. The selected method therefore learns deployable contents through repeated cost-sensitive projection while allowing a broader teacher action space; unrestricted BKG4 syntax search is not the first investment.

**Why C wins:** it uses expensive simulation to answer strategically meaningful counterfactuals, shares the answers across states, and forces deployment compatibility to be tested while discovery is still cheap. Compared with B, terminal search can directly compare coordinated investment bundles rather than waiting for a policy gradient to explore them. Compared with D/F, it learns state/action consequences rather than assigning one scalar fitness to an enormous program. Compared with E, it amortizes improved decisions. Compared with G, the teacher improves beyond replay labels. Compared with H, it postpones general abstraction-language invention until executable tasks have demonstrated value.

## Final selected architecture

### 1. Policy representation

A **finite-state economic manager** selects a quantized plan vector. A deterministic task executive realizes that plan from the current farm. Decisions begin at step 0 with a default 72-turn interval; a selected 24-turn commitment or bounded feasibility interrupt can bring the next decision forward to a daily boundary. Daily execution refreshes worker assignments. Maximum 30 manager decisions/game in the pilot, including later stages of a selected bundle. No forced late switch to an incompatible ancestral route.

Plans are desired targets and priorities, not promises that a certain number of assets will be purchased regardless of cash. Existing assets are serviced from observations. A plan can change several investment/production factors together and can retain an earlier commitment while revising a later target.

### 2. Observation/state representation

Use a versioned legal feature schema shared by training, native execution, and extraction:

- Own/public opponent cash, unlocked quadrants, live units and hires today; day/hour/step and currently visible ordered shops.
- Current market inventory and integer price for every product; one- and three-day *past* public market summaries.
- Own private shed, seeds, total capacity use, worker positions and ordered carried inventory.
- Both public tile grids: kind, crop/species, placed/planted day, current yield, watered/fed/cared flags and visible deprivation counters.
- Explicit plan targets, pending task IDs, assigned sites, progress/timeout flags, previous observed public supply summary. No learned hidden controller state is required by the primary architecture.

Role-normalize own/opponent, but preserve worker order and board geometry. Own counts and public opponent herd/crop-age summaries are deterministic derived features; they are not new information. Their implementation must be identical on CPU/CUDA and reconstructible from the official observation. Feature IDs in the generic VM are not proof that a candidate adapter already supplies them.

Pilot model inputs use **up to 192 normalized scalar economic features**, selected before outcomes, plus a shared tile encoder: per-tile categorical/numeric inputs -> 32 units -> 32 units, pooled separately by owner/quadrant. Concatenate these with an encoded candidate bundle (plan factors, branch feature/threshold IDs, duration and stage) into a 256 -> 128 ReLU MLP. Use three small independent Q heads and a factorized proposal head, **target 0.3-0.8 million total parameters, hard cap one million**. The proposal head sees only the state encoding; Q additionally sees the bundle encoding. Bootstrap whole seeds for the Q heads, not separate rows from one seed. Record actual counts before training. No pixels, long-context transformer, or full-history recurrent model in the primary pilot.

The neural pooling is a training representation, not a deployable hidden feature. BKG4 splits use legal raw/derived integer features. Any useful distinction unavailable to the student must be added as an explicit causal feature within the existing resource budget or rejected by projection. Do not smuggle the encoder into the final runtime.

The existing feature registry is closed and validated: arbitrary new feature IDs cannot be supplied by an adapter without a version change. Keep registry v1 for the pilot. Expose at most 16 declared integer summaries through reserved, adapter-populated BB inputs, using existing legal features for everything else. Initial summaries are own/opponent counts for each livestock species (six), own active counts for each crop (five), own/opponent ready-crop counts (two), own unmet-feed count, last-day aggregate public supply change, and past three-day own-cash change. Any required historical values also consume declared persistent storage. The wider neural input does not imply 192 free student features; projection must demonstrate that this smaller legal interface suffices.

### 3. Action abstraction and executive

The initial factorized plan schema includes:

| Plan field | Proposed support and meaning |
| --- | --- |
| Livestock targets | Cows/sheep/geese each 0-24; joint occupied-site feasibility mask, not three independent purchases |
| Crop allocation | Desired active-site counts for all five crops, 0-64 each with joint land feasibility; allocation may differ by quadrant |
| Land | Target 1-4 quadrants; purchase order among still-locked quadrants remains the engine's supported order |
| Labour | Daily desired hands 0-15; hire up to available funds, with exact marginal hire cost |
| Cash reserve | Quantized 0, 50, 100, 200, 400, 800; explicitly available for search, not an invariant profit rule |
| Feed financing | Maintain inventory or deliberate sale/buy-back; no universal wheat-preservation rule |
| Service priority | A permutation of feed, water, care, harvest, expansion and planting priorities, with feasibility guards |
| Spatial allocation | Canonical site ranking chosen from shed-distance, quadrant-local and row/column sweeps; operational primitives can name any unlocked legal tile |
| Sales | Sell all available; sell excess over the current service/input reservation; or batch excess at day-end. Order products by fixed item order, reverse item order, current price, or current quantity, with item-order tie breaking. Final-step liquidation is explicit. |
| Commitment duration | 24 or 72 turns; existing assets survive plan changes and retain service obligations |

This is a large factorized action domain. The initial search samples **32 complete feasible plan bundles**, not all Cartesian products. The actual list of bundles is not a permanent menu of hand-authored economies. Joint resampling, crossover, and factor proposals can create combinations absent from Shop or replays. The pilot searches linked bundles at **three consecutive manager boundaries**, potentially changing 216 turns of decisions with consequences evaluated through 718.

The executive is deliberately operational, not a second hard-coded economic optimizer:

1. Enumerate tasks from current tiles, cargo, target deficits, and explicit plan priorities.
2. Reserve shared resources and assign eligible tasks to workers in a deterministic priority/deadline/distance/worker-index order. Tie-breaking is fixed. Do not assume worker actions commute.
3. Execute short reusable options: move to tile, clear weed, carry/pick up material, construct/place, plant, water, feed, care, harvest, deposit, and emit ordered financing/sale intents.
4. Each option has legal initiation predicates, a target, explicit success/failure, a timeout, and a deterministic recovery state. Consequential work is rechecked against the next real observation.
5. Clear daily hire/assignment state at dawn while retaining real asset obligations. Recompute from actual state, never from an expected replay row.

The action interface includes raw primitive operations beneath these options. Newly successful short sequences may later be factored into reusable BKG4 subtrees, but no additional skill-discovery service is part of the pilot. Fixed site-ranking choices and task arbitration are a remaining abstraction bias, explicitly tested at the representation gate. This is not claimed to span every possible optimal strategy.

The runtime boundary must be explicit: BKG4 selects investments, plan constants, priorities, sale modes and option/FSM state; a shared native mechanical helper enumerates feasible tasks, reserves resources and computes the declared stable assignment ordering. It does not select a profitable crop/herd mix or override strategy by an undocumented heuristic. The sized adapter executes manager selection, mechanical assignment and option execution in that order within one environment turn. If separate BKG4 manager/executor views are needed, declare their state ownership and count both controllers, copied plan fields, and all native persistent state against the joint resource contract. This is new adapter work, not an existing VM callback or a claim that BKG4 contains a general assignment solver. Neural teacher and compiled student use precisely the same helper and turn phases.

**The executive is new work.** Existing weed queues and route execution cannot simply be relabeled as it. Porting all existing economic heuristics would merely move manual strategy into a hidden scheduler. The first gate must demonstrate productive execution and coordinated changes with only mechanical scheduling decisions fixed.

### 4. Teacher/learner algorithm

Use **approximate rollout policy iteration with learned proposal guidance and frequent program projection**, an Expert Iteration adaptation rather than literal AlphaZero.

For frozen policy `pi_k` and a frozen live-opponent mixture:

```text
Q_k(x, a) = expected terminal outcome after executing contingent bundle a,
            then following pi_k from the resulting actual state.
```

The bundle is a small closed-loop program: up to three plan decisions, with optional binary branches on legal observations at later boundaries. Future branches cannot depend on their evaluation seed or a privileged continuation result. Beam width eight, maximum depth three, **32 evaluated proposals total per root-group/round**, not 32 new proposals at every depth. Allocate eight one-stage, eight two-stage and sixteen three-stage proposals; complete each prefix with the frozen policy for terminal evaluation. Proposal generation combines neural factors, independent factor resampling, and recombination of complete previously evaluated bundles. Every promoted beam member receives full-horizon exact evaluation, not a learned terminal score. Pilot branches compare current money with the cash-reserve support, legal species/crop counts with 0/1/2/4/8/16/32, feed deficit with zero, or current market inventory with 0/10/25/50/100. They use only registered features/declared summary BB, never a hidden-world identifier. The manager can still use all its permitted student inputs for root selection.

The frozen neural manager produces at most 16 feasible candidate bundles by deterministic beam decoding of the proposal factors, includes the current bundle as a comparator, and chooses the largest mean Q with lexicographic tie breaking. No simulator call, seed-conditioned randomness, or hidden opponent input is used for this causal teacher evaluation. It executes the selected bundle's later observation-conditioned stages, not just its first plan followed by an unrelated manager. A bundle ID, stage and timeout preserve that commitment; a declared feasibility interrupt terminates it. Both search and evaluation use these same interrupt rules. BKG4 stores bundle constants/branches in subtrees and only the active plan plus bundle ID/stage in BB, rather than allocating three complete plans in persistent memory.

At each round, collect roots from legal complete live trajectories. Force the declared alternatives and run both policies to termination. Fit the Q ensemble to **all** evaluated returns, including failures; hold folds out by whole seed and retain opponent-mixture/version tags as training bookkeeping, not actor inputs. Estimate expected action values before maximizing. Individual privileged-world argmax labels are prohibited.

Use a terminal training scalar:

```text
u = 1(win), 0.5(tie), 0(loss)
m = final own cash - final opponent cash
r = 0.9 * (2*u - 1) + 0.1 * m / (10000 + abs(m))
gamma = 1
```

The bounded smooth margin term provides resolution without a hard clipping plateau at large margins or equating tiny cash changes with broad competitiveness. There is no reward for unliquidated asset value, requested purchases, or survival alone. Final policy selection reports raw win utility and material economic effects separately; shaping cannot pass promotion by itself.

Train terminal-return regression with Huber loss and a within-root ranking loss, using a fixed 0.1 ranking coefficient. Fit proposal factors with return-weighted maximum likelihood on cross-fitted advantages: weight `exp(clip(A/0.2, -5, 5))`; cap any root's total contribution. Adam, learning rate `3e-4`, batch 256, at most 20 passes/round, BUILD-only early stopping. These starting hyperparameters are preregistration proposals, not tuned values. Q uncertainty determines which alternatives deserve evaluation, never final promotion.

Do not cut early investment rollouts at day-end because the critic predicts low cash. The first pilot uses **no learned leaf-value truncation**. Value learning estimates plan quality and expected outcome; full exact simulation remains the comparison oracle. This trades some simulation cost for a cleaner first falsification.

### 5. Replay initialization

Pretrain the proposal head on the masked component labels from the 64-episode sample, at most ten passes and one GPU-hour included in the pilot cap. Pretrain an auxiliary prediction of next-day *public* supply only where current-engine labels are valid. Do not pretrain a privileged value function. Round zero uses half unconditioned proposals; replay weight decays to zero by round two. Keep a no-replay initialization diagnostic on the same examples/search support, not a second full campaign.

### 6. Opponents and 7. self-play

Use frozen anchors plus asynchronously refreshed response policies; one learner trains at a time. Training identities belong only in the offline payoff table. Initial local anchors are Shop Plus, Gold V25, and the authoritative champion, explicitly declared TRAINING for this new epoch; all retain public ancestry caveats. Add a new live task-policy challenger with a different production/service profile, selected for mechanical validity before candidate outcomes, not falsely advertised as a strong independent opponent.

The same improvement loop produces generalist policies and periodically a best response to a hard mixture. Do not build separate AlphaStar main/exploiter/league-exploiter stacks. A maximum eight-entry league and a 16-entry behavioural archive are sufficient after the pilot passes. Public source policies assigned TRAINING are never called HOLDOUT later.

### 8. Long-horizon credit and 9. CUDA

Credit uses exact terminal returns for a complete investment/production bundle and gamma one. Three-boundary proposals can coordinate finance today, expansion next, and production later; the critic learns their expected consequences across games. Periodic from-start rollouts expose changes before step 144. Short-horizon scheduling feasibility does not substitute for terminal economic value.

CUDA evaluates both live policies and all full suffixes, captures complete controller state, and yields compact outcomes/features. CPU controls beam bookkeeping and model fitting orchestration, not large Python game loops. For training data generation, inference occurs at manager boundaries, not every worker action. Alternate batched simulation and small-model fitting on the 4-GB device rather than retaining two memory-hungry frameworks concurrently.

### 10. Exploration and 11. curriculum

At least 50% of pilot proposals ignore learned replay frequencies; at least 25% of the full proposal batch jointly resamples three or more plan fields, including one capital/labour and one production decision. Include early-root states, all crop/species categories, and delayed-payoff bundles. No selection by public team identity or fixed future shop pattern. Archive descriptors use realized production mix, acquisition timing, capital employed, and payoff vectors, not program hashes.

Curriculum changes which roots receive extra labels, not the definition of success. Start with legal execution/terminal realization fixtures, then allocation and investment, then full opening-to-terminal policies. Every strategic round still includes from-start games. Restored late states cannot by themselves pass a gate, and a successful option must prove its prerequisites can be reached without a privileged reset.

### 12. Teacher validation

Evaluate a **frozen legal neural manager plus the exact executive**, without per-game privileged search selection. Compare it with Shop Plus, the prior manager, and the unguided-planning control in fresh whole games. Separately report the offline search's training improvement. If only the latter is positive, the learned teacher has not passed.

### 13. Extraction, 14. correction, 15. runtime

Project each improvement round to a cost-sensitive decision tree/FSM over the same executable bundles. Lower comparisons, plan constants, future legal branches, latches, task options, and bounded state into BKG4. Run the student on new BUILD seeds, query/counterfactually label its encountered states, and refit twice at most in the pilot. Deployment contains integer BKG4 selection and mechanical task execution, optional certified route fragments, explicit memory, and no neural weights/search/retrieval.

### 16. Promotion and 17. iteration

Only the compiled whole policy can be promoted after independent robust evaluation and engineering checks. An accepted student becomes the new roll-in policy and one frozen league member. It does not erase previous challenging policies. The next iteration queries states induced by that student, creates new causal bundles, and repeats with a new untouched evaluation epoch.

## Why this can escape the Shop Plus basin

The earlier pilot could replace the fourth post-144 cow with a sheep while inheriting most of the financing, sites, workers, and service structure. This design can instead ask whether to hire a different workforce at the opening, delay or accelerate a land purchase, use a different crop cash bridge, select a different herd/site allocation, and revise later investment after observing production and demand. All those decisions can change in one proposal, and subsequent actions service the resulting farm rather than returning to a mismatched tape.

Concrete newly expressible hypotheses include:

- Finance earlier livestock through a different quick-return crop mix, with changed hiring and acquisition timing, then service it through maturity.
- Allocate land and workers differently between crop blocks and animal service rather than requiring a new lifecycle to fit an old idle lane.
- Establish tomato or a larger sheep/goose programme from a feasible opening, with coordinated working-capital and sale decisions, instead of changing a single planted item late.
- Delay investment when visible rival supply saturates its product; redirect later *uncommitted* capacity without pretending already placed animals can be freely converted.
- Use deliberate feed buy-back where profitable and inventory reserves elsewhere; neither is permanently imposed by a repair rule.
- Learn a more productive common opening even when the future opponent regime is not yet identifiable. Do not resolve opening uncertainty by illegal opponent recognition.

These are **testable possibilities, not profitability claims**. The simulator must establish their value. The escape mechanism is coordinated decisions plus new reachable states plus cross-state generalization, not merely a larger number of random proposals. Primitive task support and a learnable proposal distribution reduce, but do not eliminate, human abstraction bias.

## Training/search loop

```text
Small compatible replay sample -> removable proposal initialization
                                     |
Frozen BKG4 student + live league -> fresh legal roll-in trajectories
                                     |
Shared contingent plan proposals -> exact full-horizon CUDA returns
                                     |
Cross-fitted Q / proposal update -> improved causal neural manager
                                     |
Cost-sensitive BKG4 projection -> student-induced BUILD roll-ins
                                     |
Exact counterfactual correction -> frozen whole-policy DEV
                                     |
Independent release gates -> champion N+1 -> league / next epoch
```

No per-root winner is deployed by seed/hash lookup. No search score crosses from DEV into model fitting. Within BUILD, retain all failures, censor no economic collapses, and record forced-action support probabilities, model/checkpoint identity, executive version, seeds, paired references, and charged work. Recompute labels when the rollout policy changes materially; `Q_k` is not timeless ground truth.

The practical batch API should accept a policy/executive identity, candidate bundle IDs, root IDs, opponent controller states, and an explicit turn budget; return compact terminal outcomes and legal summaries. This is a proposed narrow interface, not an implementation commitment in this session. Avoid model training from full decoded replay JSON or per-tick host/device transfers.

## Opponent/league design

After the pilot, cap the league at eight frozen policies: three initial anchors, the latest accepted student, two older diverse students, one generalist response, and one difficult-mixture response. Deduplicate by economic trajectory and payoff similarity. Store a small seat-balanced payoff matrix with seed uncertainty. Retain a policy because it contributes a challenge, not because it has a different filename.

Sampling: 50% uniform over behavioural/ancestry cohorts, 25% latest accepted opponent, 25% hard-opponent sampling proportional to `(1 - win_rate)^2`, with a 5% floor per active cohort after normalization. Freeze this mixture during each improvement round so that return targets have a defined opponent distribution. Best-response training uses the same machinery and a declared target mixture, not a hidden identity feature.

This is PSRO-inspired restricted-population improvement, **not a Nash guarantee**. Initial selection does not need a meta-solver. A credible cycle justifies a restricted-game mixture in a later epoch, but it does not justify adding a permanent identity router. Deterministic BKG4 cannot be assumed to implement a true randomized equilibrium at identical initial observations; the chosen target is a robust deterministic adaptive policy. If reliable results show that randomized openings are essential, that is a new deployment decision, not seed-fingerprinting permission.

Public opponent features should describe economic effects: visible species/crop ages, land, workers, cash, and observed market supply trends. A three-head auxiliary public-supply forecast is permissible in training. Retain explicit runtime regime summaries only if removing them costs at least one utility point or material economic gain on a fresh BUILD confirmation with both seats. No source-name classifier. Existing binary routing remains the default composition mechanism; additional stateful decisions must show incremental out-of-sample value.

External validation initially uses newly pinned live COK and lonespear policies where licensing and current-engine execution checks succeed. They are structurally useful transfer checks, not certified leaders; known/unknown replay ancestry must be disclosed. Before claiming gold-candidate strength, acquire at least one genuinely strong fresh live non-Shop challenge. If it cannot be obtained, report the missing evidence rather than substitute more copies of local routes. Old TRANSFER/VALIDATION/CEGIS/SEALED panels remain isolated.

## BKG4 extraction/distillation

### Projection procedure

1. Collect state/plan/terminal-return data on mixed teacher/student BUILD trajectories. Estimate counterfactual plan regret with held-out-seed Q predictions and exact rollout confirmation.
2. Fit a cost-sensitive tree whose leaves select complete feasible bundle subprograms, containing plan vectors and any later legal branches. Use aggregate expected regret, not majority imitation alone; quantize split thresholds to integer runtime features before evaluation. Merge equivalent bundle choices and reuse constant tables where supported. BKG4 has no arbitrary subroutine-call opcode: shared logic requires a valid common FSM dispatch or duplicated tree nodes, both fully counted. Count the complete lowered controller, not only the selector tree.
3. Add explicit finite-state latches only for commitments, current option/target, and bounded causal history. The primary model has no recurrent hidden vector to extract.
4. Compile decisions and task options to generic BKG4 nodes; produce sized state storage and a new legal adapter. Stage route fragments only when their entire entry/exit contract is satisfied.
5. Evaluate native CPU/CUDA action/state equality from identical legal observations and matched full games. Confirm representative new mechanics and boundary behaviour with the official engine.
6. Execute the BKG4 student on new BUILD games. Label its own off-trajectory states with the fixed causal teacher/forced alternatives; aggregate and refit. At most two correction rounds in the pilot, three in a later epoch.

The VIPER lesson is cost-sensitive student roll-ins, not a promise that every Q-labelled tree is verifiable or optimal. Our legality/semantic checks can establish execution properties on checked paths; competitive generalization remains statistical.[^viper][^dagger]

### Practical resource contract

Initial target: **<=2,048 compiled nodes**, hard pilot cap **4,096**, depth <=24, maximum static visits <=4,096. Target total compiled controller <=256 KiB; measure rather than infer bytes from node count. Budget <=112 of 128 BB slots and <=128 node-state slots, including all executive state. A feasible allocation is 48 slots for target/task/phase of 16 actors, 32 plan fields, 16 history/summary slots, and 16 global control/reserve slots. Native auxiliary memory must also be listed and snapshotted; no uncounted persistent side channel.

Sixteen actors means the farmer plus fifteen hands, covering the audited high-labour examples without extending BKV6. Support up to the generic 40 only if later evidence justifies a revised storage contract. Keep <=128 outstanding intents and <=10 final market orders. Mechanical placement and joint resource masking must happen before action emission, with parity tests for ordering conflicts.

The Shop route pack may remain as a certified incumbent mode entered only from reset, with ordinary binary choice between it and the new economic executive. It is not an automatic rescue after an incompatible investment. A new compiled policy may eventually remove routes entirely; BKG4 remains the policy language either way. If complete routes are added later, each 719-row route costs about 67.6 KB before descriptors/arity; 32 such routes are about 2.16 MB, not thousands of free branches.

### Fidelity gates

Same plan plus same legal observation and explicit state must produce **exactly identical** executive actions and state transitions in teacher executive, BKG4 student executive, CPU, and CUDA. Manager plan disagreement is allowed only as measured approximation, not called a semantic mismatch. Compare closed-loop returns and value-weighted decision regret, not just 99% raw-action agreement.

Require the student to retain **>=80% of the teacher's measured positive paired utility gain**, lose <=1 point absolute utility, and retain >=80% of its material paired economic gain on the same fresh panel. When baseline utility is saturated, economic retention remains mandatory. A tiny or nonpositive teacher gain cannot satisfy a retention ratio. Two correction rounds without passing closes the pilot; do not double tree size indefinitely.

Reset/repeated-step behaviour, interrupted options, newly hired workers, dawn expiry, simultaneous purchases/planting, mixed cargo, full shed, market order limits, absent shops, final action 718, and unsupported engine configuration all require explicit tests. Freeze the shared integer executive before evaluating policy quality. Neural floating-point training need not be bit-identical; deployed actions and CPU/CUDA simulation must be.

## Validation and promotion

### Data flow

Preserve **multi-regime BUILD -> discovery oracle -> INTERNAL_DEV -> binary composition/routing -> TRANSFER/VALIDATION/CEGIS/SEALED -> final composition** as data-role boundaries. This plan changes the discovery oracle and strategic promotion criteria, not the disjointness firewall. Freeze the complete composed policy before its terminal audit. A downstream failure may inspire a future retired-and-replaced training epoch, never another claim of unseen performance on the same panel.

Use full games, live opponents, both seats, matched initial seeds, and independent controller instances. Bootstrap **whole seed blocks**, keeping seats/opponents paired; related opponents are cohorts, not independent replications. Run three independent learning/proposal RNG replications after the pilot before a release claim. Keep a frozen shortlist and no post-DEV model selection.

### Metrics

Primary competitive utility is `u = win + 0.5*tie`. Report cohort-balanced paired delta, both seats, per-cohort deltas, worst-quartile cohort utility, paired margin, own final cash, and the complete W/T/L transition table. Separate mirror and non-mirror results. Economic diagnostics include realized production, hiring/input expenditure, working-capital timing, failures, discards, and unsold terminal goods.

A +10-point utility gain on tiny mirror sales cannot qualify as strategic improvement. Likewise more own money against an idle opponent does not establish competitive value. Use both raw outcome and material economic evidence during discovery, then stronger live opponents for release.

### Proposed release thresholds

On **at least 512 fresh seeds**, both seats, and at least four meaningful live cohorts, require the complete BKG4 policy to satisfy:

- Cohort-balanced utility improvement >=3 percentage points over the declared strongest relevant reference, with a one-sided 95% seed-clustered lower bound >0.
- Non-mirror aggregate improvement with a positive 95% lower bound; no reliance on parent-mirror ties to supply the claim.
- Positive point improvement in both seats and at least two meaningful non-mirror cohorts; worst-cohort paired point regression no worse than -5 points, and worst-quartile cohort utility not lower.
- No unresolved runtime corruption, invalid feature access, CPU/CUDA discrepancy, or official/native semantic mismatch. Representative official-engine confirmation includes new executive states and both seats.
- Predeclared economic-collapse rate (final own cash below 50% of paired reference) does not increase by more than one point on the panel; any runtime-caused collapse is an engineering veto regardless of rate.
- Teacher/student fidelity gates pass, independent source/provenance/rules requirements are resolved, and the complete final composition passes its designated sealed stage.

If utility is ceiling-limited against the available pool, recruit stronger live challengers. A margin-only pilot pass may justify research, **not release promotion**. The authoritative champion and strongest isolated Shop reference must both be reported; replacing a weak incumbent is not evidence of surpassing the strongest available public-derived policy.

Ordinary W-to-L tradeoffs are allowed if these robust thresholds pass. Universal preservation of every historical win is not a sound general strategic objective. Zero **semantic** mismatch, illegal/private information use, and runtime corruption remain absolute. Quantify uncertainty and leave the previous champion intact until a separately authorized promotion.

## Compute and engineering budget

### Measured local resources

Hardware inspection: **NVIDIA GeForce RTX 3050 Laptop GPU, 4,096 MiB VRAM**, driver 595.84; approximately 3,756 MiB free at inspection. CPU: **Ryzen 7 4800H, eight cores / sixteen threads**. Memory: approximately **15 GiB RAM**, 9 GiB available at inspection. The project environment has `kaggle-environments==1.32.7`; PyTorch is **not installed** in that virtual environment. No learner dependency was added.

Recorded route pilot: 7,327,560 GPU turns in 21.268 recorded seconds, approximately **344,535 turns/s or 479 FGE/s**. Mechanics fallback: 3,376,424 GPU turns in 37.519 seconds, approximately **89,993 turns/s or 125 FGE/s**. These are different native batch workloads and recorded evaluator times, not neural training throughput or end-to-end campaign duration. The earlier gold plan cited only about 10.6 FGE/s on a small-batch Shop workload; none is a guarantee for a new executive.

The route pilot's 528 official-engine games took about 394 seconds wall on eight workers, roughly 0.88 aggregate CPU-hours by worker-wall accounting, not 394 CPU-seconds. Compilation, feature extraction, and model inference must be charged separately.

Use **FGE = simulated joint game turns / 719** from this plan onward. Old Phase 1/CCPR reports normalized by 720; preserve their historical numbers and compare raw turns when necessary. One joint transition is two policy decisions but one simulator turn. Forks reduce work, not the number of independent games.

### Candidate compute estimates

The following are order-of-magnitude budgets to obtain a meaningful first strategic result after basic interface work, **not promised sample complexity**. GPU-hours include learning/search/inference. CPU-hours are aggregate consumed time, not elapsed engineering time. No method is selected on an extrapolated fastest kernel number.

| ID | Joint transitions / FGE | GPU-hours | CPU-hours | Model / VRAM | Compact data | Engineering |
| --- | --- | ---: | ---: | --- | --- | --- |
| A | 100M-1B / 139k-1.39M | 50-300+ | 10-40 | 1-5M recurrent autoregressive; 3-4 GB tight | 5-20 GB bounded replay | 3-6 weeks; reject a hundreds-of-GPU-days scale-up |
| B | 20M-100M / 27.8k-139k | 12-48 | 4-12 | 0.5-1M actor/critic; <=3 GB | 1-5 GB | 2-3 weeks fresh; fallback reuses C interface |
| C | Pilot <=21.57M / <=30k; next 71.9M-179.75M / 100k-250k | Pilot <=8; next 12-36 | Pilot <=6; next 8-16 | <=1M; target <=2.5 GB | Pilot <=2 GB; next <=5 GB | <=6-day pilot; roughly 2-4 weeks total gated work |
| D | 35.95M-179.75M / 50k-250k | 8-36 | 6-16 | 0.2-1M surrogate ensemble; <=2.5 GB | 1-5 GB | 1-3 weeks |
| E | 71.9M-359.5M / 100k-500k | 12-60 | 8-24 | No learner; native search batches <=3 GB | <=2 GB summaries | 1-3 weeks |
| F | 35.95M-359.5M / 50k-500k | 8-60 | 8-24 | No model or <0.2M relaxed tree; <=3 GB | <=2 GB | 1-3 weeks |
| G | 7.19M-35.95M fresh / 10k-50k plus selected replay records | 4-16 | 4-12 | 0.3-2M model; <=3 GB | 1-5 GB streamed | 1-2 weeks |
| H | 71.9M-359.5M / 100k-500k plus synthesis | 20-80, external LLM cost unpriced | 20-80 | Local recognizer <=1M; larger code model not assumed | 2-10 GB | 3-6 weeks |

Native simulation at a conservative **five effective FGE/s** would consume 1.67 GPU-hours for 30,000 FGE; neural inference, data movement, and fitting must fit the remaining pilot allowance. At only one FGE/s, simulation alone takes 8.33 hours and misses the cap. Measure the complete batched path after the first 256 games; require at least five effective FGE/s and <=2.5 GiB peak device memory before the main experiment. This is an admission threshold, not observed throughput.

One million FP32 parameters need about 4 MB for weights and roughly 16 MB including gradients and two Adam moments; activations, CUDA context, simulator buffers, temporary tensors, and inference batching dominate VRAM. Start at 64-256 native worlds/batch and 256 training examples/batch. A thousand historical 49,488-byte snapshots are about 49.5 MB; new executive snapshots may be larger and must be measured. Store only selected roots and compact decision features, with bounded replay eviction. Do not use the compressed archive's 2.2-GB size as its decoded RAM estimate.

Expected engineering is dominated by the executable plan interface, sized snapshots/legal adapter, and BKG4 extraction. The neural network is comparatively small. Six engineering days is a strict first gate, not a claim that a complete research stack can be finished in six days. Subsequent work is authorized only in bounded tranches; verify the current competition deadline before accepting a multi-week path.

## Immediate falsification experiment

### Question and contract

**Can learned-guided exact improvement over coordinated causal economic plans produce a materially stronger whole-game policy than Shop Plus, and can a bounded BKG4 student retain it?**

This is one sequential experiment with a matched control and internal diagnostics, not a menu of research programmes. Freeze source/asset identities, plan schema, candidate budgets, seed manifests, evaluation endpoints, thresholds, and model initialization before returns. Do not implement it during this research session.

### Early representation gate

Spend at most two of the six engineering days and 1,000 of the 30,000 native FGE on the executive/adapter contract. Demonstrate all of the following:

- Untouched incumbent mode is action/state identical to Shop Plus.
- A task-executed reference economy completes live whole games with median own final cash at least 95% of the Shop reference on 32 BUILD seeds, both seats. This is a productivity prerequisite, not fidelity to the original route or proof of competitive quality.
- At least eight mechanically valid coordinated bundles change two or more of opening investment, land, labour, crop/herd allocation, or spatial layout, with at least four first divergences before step 144. Their eventual realized commitments differ, not only requested tokens.
- New task intent execution, controller-state restore, CPU/CUDA parity, and representative official canaries have zero semantic mismatch; unsupported operations fail explicitly.

Failure closes the architecture attempt without training a network. Do not rescue it by restricting changes back to post-144 inherited lanes. This gate directly tests the most important unproven assumption: the new executive can preserve competent production while admitting genuinely new policies.

### Models, examples and rounds

Use the <=1M-parameter proposal/Q ensemble specified above, Adam `3e-4`, batch 256, gamma one, terminal reward as specified. At most 64 replay episodes / 2,560 partial-label records; no full-corpus scan. Three improvement rounds and at most two BKG4 correction rounds; one frozen student per arm proceeds to DEV.

Each improvement round uses **64 fresh independent seed contexts**, one declared seat/opponent/checkpoint assignment per seed, balanced across both seats and four local TRAINING opponents. Initial checkpoints are 0, 72, 144, 216, with 16 roots each; subsequent rounds use the same time allocation on fresh student/teacher roll-ins. For the pilot, each checkpoint's 16 independent roots form one group, balanced over the eight seat/opponent combinations. All 32 candidate programs in that group run on all 16 roots. They may branch on legal economic features, but group construction and candidate selection cannot branch on opponent identity. Keep the deliberately coarse groups intact; do not split them into near-singleton hidden-world surrogates. These are multi-regime BUILD groups, not exact information-set posteriors.

**Learned arm:** 32 candidate contingent bundles/group, evaluated on every root, up to three linked planning decisions. Pool proposals from the group's legal state encodings, then freeze each stage's common candidate set before its outcomes. Half are independent support-covering proposals; the other half use the current proposal/Q guidance after round zero. Full-horizon outcomes supply at most **6,144 state/bundle/outcome rows** over three rounds, before correction data.

**Unguided control:** identical executive, action support, beam depth, roots/cost allowance, replay access, and BKG4 projection procedure, but no neural Q/proposal guidance. Uniform/recombination proposals replace guided sampling. It also receives up to 6,144 exact comparisons. Projection is shared to isolate the contribution of learned search guidance rather than compiler quality. Shared initial work is explicitly charged once as common infrastructure, not claimed as free per-arm evidence.

This pilot cannot support broad claims about stochastic optimizer variance from one run. Its purpose is high-information rejection or permission for replicated training, not final algorithm superiority.

### Work ledger and seed separation

Proposed seed namespaces, unused in the reviewed experiment manifests: BUILD 61010000-61019999, correction/confirmation 61020000-61029999, DEV 61030000-61039999, transfer 61040000-61049999, parity 61050000-61050999. Before execution, check against the full non-sealed registry without opening SEALED contents; regenerate if there is any reserved-range collision. Seed IDs are evaluator provenance only.

| Work item | Upper allowance |
| --- | ---: |
| Representation/fidelity preflight | 1,000 native FGE |
| Both arms: 3 x 64 x 32 full remaining-game comparisons | <=12,288 native FGE; exact turn accounting usually less |
| Prefix captures, paired parents, BUILD confirmation and model diagnostics | <=4,000 native FGE |
| Two bounded student-correction rounds, both arms | <=8,000 native FGE |
| Fresh native DEV, teacher/student/control/Shop references and head-to-head | <=4,000 native FGE |
| Reserved native correctness/failed-run allowance | 712 native FGE |
| **Total** | **30,000 FGE = 21,570,000 joint transitions maximum** |
| Live external/official checks | <=512 official-engine games, charged separately inside six aggregate CPU-hours |

Fresh native DEV uses 128 new seed blocks, both seats, the four frozen local opponents. Run teacher, learned-arm student, control student, and Shop reference on the common panel: 4,096 whole-game executions would slightly exceed the 4,000 DEV allocation **if all four receive the complete matrix**. Therefore preregister teacher on 64 of those seeds and the three deployed comparators on all 128: 512 + 3,072 = **3,584 executions**, leaving 256 direct student-versus-Shop games and 160 correctness/reference executions inside 4,000. Use the common 64-seed subpanel for teacher/student retention, not unmatched denominators.

Transfer uses 32 separate seeds, both seats, two newly pinned live public policies, and teacher/student/control/reference: **512 official games**. If canaries require additional official games, reduce the transfer allocation *before* starting so that the combined cap remains 512, retaining at least 24 seeds per endpoint. Report any endpoint unavailable or underpowered; do not relabel a substituted ancestor as independent HOLDOUT. No transfer result tunes the policy.

### Pass criteria

All engineering and representation checks above must pass. Then require:

1. **Absolute strength:** student-versus-Shop utility >=55% on 256 fresh direct games, with one-sided 90% seed-block lower bound >50%.
2. **Beyond mirror:** >=3-point paired utility improvement on the non-mirror native cohorts, positive one-sided 90% lower bound, and positive contribution in at least two non-mirror cohorts. If the reference already has >=95% utility there, use the predeclared ceiling exception: no negative utility delta, median paired margin >=1,000 coins, and median own final cash >=2% higher in both seats and at least two cohorts. This exception permits further research, not promotion.
3. **Causal materiality:** reverting the coordinated investment/production bundle, while retaining sale mechanics, removes at least half the economic gain on a predeclared BUILD attribution panel. At least one effective change begins before step 144. Terminal-only tie-breaking fails.
4. **Learned contribution:** at equal charged simulation budget, learned guidance improves retained-candidate terminal regret by >=20% on fresh BUILD-confirmation groups relative to unguided guidance, with positive seed-clustered 90% lower bound for the improvement. Require pairwise ranking accuracy >=65% on the material alternatives defined below. If the control has zero regret or neither finds positive alternatives, this gate does not pass by division convention. Report fresh policy comparison between arms even when economic gains, rather than utility, supply the early signal.
5. **Deployability:** BKG4 retains >=80% of positive teacher utility/economic improvement with <=1-point absolute utility loss on their common fresh subpanel, after at most two correction rounds, inside the resource cap.
6. **Transfer:** no negative aggregate utility delta, no cohort worse than -5 points in point estimate, and no material economic collapse against the two live transfer endpoints. This is an early veto, not a medal-strength certificate.

For the learned-contribution diagnostic, preregister four new 16-seed checkpoint groups. Freeze each arm's retained bundle before confirmation. Evaluate a common pool of up to eight bundles (both selections, reference, and up to five diversified retained alternatives) on all roots, charged inside the 4,000-FGE confirmation allowance. Define a group's empirical pool optimum as the largest **group-mean** terminal scalar, and regret as its difference from that arm's selected bundle; compare the mean group regrets. This is pooled finite-support regret, not global-optimum regret. Bootstrap entire seeds within groups and recompute the pooled maximum. For ranking accuracy, use whole-seed cross-fitted predictions and exact paired **group-mean** returns; a pair is material when scalar return differs by at least 0.05, or when paired utility is identical and mean margin differs by at least 500 coins. Require at least 20 such pairs across the four groups; insufficient support is a non-pass, not an excuse to add tiny mirror differences. The diagnostic never trains on its confirmation outcomes within this pilot.

A pass authorizes one replicated 100k-FGE tranche, not an unrestricted campaign. A marginal confidence interval is a non-pass at the fixed budget, not permission to keep querying DEV until significant.

## Kill criteria

- **Representation kill:** the new executive cannot maintain the reference productivity floor, express coordinated early changes, or fit BKG4 without truncating workers/state. Stop before a learner campaign.
- **Primary algorithm kill:** after three rounds / 30k FGE / eight GPU-hours, whichever comes first, the frozen teacher/student fails the absolute/non-mirror/materiality test, or learned guidance fails to improve useful search efficiency. Invoke the one fallback only if the common interface remains valid.
- **Information kill:** apparent improvement requires actual future seeds, invisible opponent state, per-world oracle selection, frozen opponent suffixes, or teacher inputs unavailable to BKG4. Invalidate dependent results; do not distill illegal labels and call the output causal.
- **Extraction kill:** after two pilot corrections, student retains less than 80% of meaningful gain or exceeds 4,096 nodes/112 BB/128 node-state slots. No indefinite tree growth. Shared extraction failure also blocks the fallback.
- **Throughput kill:** below five effective FGE/s on the complete batch path, >2.5 GiB peak VRAM, >six aggregate CPU-hours, or >six engineering days needed for a fair pilot. Report engineering failure separately from algorithm failure.
- **Public-basin kill:** three rounds produce no beneficial non-replay coordinated opening/production change, or the only gain is against mirror opponents. Better replay likelihood is not progress.
- **Post-pilot kill:** two independent 100k-FGE repetitions fail to preserve broad positive gain, or a later 250k-FGE ceiling produces no additional deployed improvement. Stop scaling; preserve the current champion.
- **Generalization kill:** source-diverse live checks systematically reject internal gains. Refresh evaluation once in a newly declared epoch before making further competitiveness claims; do not use exposed HOLDOUT as a permanent optimization target.

No kill criterion authorizes lowering semantic/legal standards. A reproducible engineering bug may be fixed inside the original time/compute cap; affected outcomes are discarded and reruns charged. No additional architecture is introduced after the declared fallback fails.

## One fallback

**Replay-initialized hierarchical recurrent PPO over the validated economic-plan executive.** This is not raw PPO or PPO choosing among existing Shop routes. It uses the same factorized investment/production action support and exact deterministic low-level executive, but learns from complete fresh episodes rather than a learned counterfactual critic and beam-improvement operator.

Trigger: the primary's decisive scientific gate fails while mechanical expressiveness, legal state access, snapshots, throughput, and initial BKG4 extraction are sound. Shared representation/legality/extraction failure ends this plan; a second optimizer is not a credible remedy. No parallel PPO campaign.

Specification: same entity/economic encoder, a 128-unit GRU and autoregressive masked categorical plan heads, <=1M parameters. Replay partial-label initialization as above; removable BC weight goes to zero by 10% of online work. Terminal scalar reward unchanged, gamma one, GAE lambda 0.95 **per manager decision**, full episodes with recurrent burn-in/state replay for minibatches, PPO clip 0.2, Adam `3e-4`, four update epochs, entropy coefficient 0.01 decaying to 0.001. Batch 128 complete native games; preserve action-component joint log probabilities and do not treat aborted/masked commands as independent sampled actions.

Train against the frozen small league mixture, updating it only at epoch boundaries. Limit the trial to **20 million joint environment transitions (~27,816 FGE), 12 GPU-hours, four aggregate CPU-hours, and three additional engineering days** using the primary's already built interfaces. Use two initialization seeds splitting the budget, not a hyperparameter sweep. This tests whether temporally coherent on-policy exploration can outperform biased/local search improvement.

Before deployment, replace GRU memory with the primary's explicit legal history/commitment state and a tree/FSM; at most 16 discrete memory states if additional memory is genuinely needed, inside the same BB/runtime budget. Quantization and memory-transition errors are real distillation risks. Two DAgger corrections and the same >=80% retention criterion apply. An excellent neural PPO policy with no faithful BKG4 realization fails.

Use newly frozen evaluation seeds; the primary's exposed DEV is no longer unseen. Require the same absolute strength, non-mirror/materiality, transfer, and engineering thresholds, except the primary-specific critic-efficiency gate. Failure at the fixed budget ends this discovery plan. No raw-action RL, surrogate evolution, route expansion, or mechanics campaign is quietly added afterward.

## Path from current agent to gold candidate

1. **Research decision complete:** preserve both failed gold-plan experiments, exact current assets, and this explicit methodological reset.
2. **Authorize only the six-day falsification experiment:** reconcile governance, validate current competition constraints, and establish the new executable plan/BKG4 contract. No full league first.
3. **Require teacher and student evidence together:** pass the bounded learned-versus-unguided comparison, early coordinated-change attribution, and live fresh tests. Otherwise invoke only the specified conditional fallback or stop.
4. **Replicate before scaling:** two independent seed/proposal runs and a third learning initialization, within the next capped research tranche; grow the league through the same response oracle and measure real behavioural diversity.
5. **Release only a robust deployed policy:** use stronger fresh opponents, protected downstream stages and official confirmation. Do not use a pilot's ceiling-exception margin gain as a promotion certificate.
6. **Iterate from the deployed student:** champion N+1 contributes new reachable states and an opponent, while earlier challenging policies remain. The discovery objective is new economic capability, not additional source wrappers or small mirror repairs.

Gold is not established by internal tests alone. Current leader-generation methods and ReCurSiON's verified leaderboard position remain unknown. The practical next investment is a system that can propose coordinated causal economies, learn from exact outcomes, and repeatedly produce executable BKG4 improvements. This plan earns further engineering only by demonstrating that loop.

## Sources

Repository sources are linked at their relevant claims and refer to the audited starting HEAD. External sources below were retrieved or checked on 2026-09-10; mutable source identities are explicitly pinned where retrieved through APIs. Discussion observations are author reports, not independently reproduced results. No external benchmark gain is used as a numerical forecast for Kaggriculture.

[^engine]: Kaggle, [official Kaggriculture schema](https://raw.githubusercontent.com/Kaggle/kaggle-environments/master/kaggle_environments/envs/kaggriculture/kaggriculture.json), current public source; local runtime pinned to package 1.32.7. Defines legal observation/action boundary and timeout/reward defaults.
[^archive]: ashok205, [Kaggriculture Top10 Replay Archive](https://www.kaggle.com/datasets/ashok205/kaggriculture-top10-replay-archive), version 19; [public metadata endpoint](https://www.kaggle.com/api/v1/datasets/list?search=kaggriculture-top10-replay-archive). Corpus counts/body statistics come from the linked local audit, not a new full decode.
[^exit]: Anthony, Tian and Barber, [Thinking Fast and Slow with Deep Learning and Tree Search](https://arxiv.org/html/1705.08439v2), NeurIPS 2017, sections 3 and 5. Planning/generalization loop and expert/apprentice gap.
[^az]: Silver et al., [Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm](https://arxiv.org/abs/1712.01815), 2017 AlphaZero preprint. Policy/value-guided exact-game search; not a direct model for partial observation.
[^gumbel]: Danihelka et al., [Policy Improvement by Planning with Gumbel](https://openreview.net/pdf?id=bERaNdoegnO), ICLR 2022. Few-simulation improvement and correctly evaluated action-value condition.
[^sampled]: Hubert et al., [Learning and Planning in Complex Action Spaces](https://arxiv.org/abs/2104.06303), ICML 2021. Planning over sampled action subsets, not full enumeration.
[^options]: Sutton, Precup and Singh, [Between MDPs and Semi-MDPs: A Framework for Temporal Abstraction in Reinforcement Learning](https://incompleteideas.net/609%20dropbox/other%20readings%20and%20resources/Options.pdf), Artificial Intelligence 1999. Closed-loop options and semi-MDP accounting.
[^compile]: Kipf et al., [CompILE: Compositional Imitation Learning and Execution](https://arxiv.org/abs/1812.01483), ICML 2019. Learned variable-length sequence abstractions.
[^spirl]: Pertsch, Lee and Lim, [Accelerating Reinforcement Learning with Learned Skill Priors](https://proceedings.mlr.press/v155/pertsch21a.html), CoRL 2020 / PMLR 2021. State-conditioned skill priors.
[^skilltree]: Wen et al., [SkillTree: Explainable Skill-Based Deep Reinforcement Learning for Long-Horizon Control Tasks](https://arxiv.org/abs/2411.12173), 2024 preprint / AAAI 2025. Strategic tree plus learned skill decoder; decoder-deployment caveat.
[^ppo]: Schulman et al., [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347), 2017. Bounded fallback optimization method.
[^impala]: Espeholt et al., [IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor-Learner Architectures](https://arxiv.org/abs/1802.01561), ICML 2018. Decoupled collection and V-trace, not a solution to action abstraction.
[^r2d2]: Kapturowski et al., [Recurrent Experience Replay in Distributed Reinforcement Learning](https://openreview.net/forum?id=r1lyTjAqYX), ICLR 2019. Recurrent state staleness and replay/burn-in costs.
[^a3c]: Mnih et al., [Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1602.01783), ICML 2016. Actor-critic lineage; parallel actors do not remove economic credit assignment.
[^sac]: Haarnoja et al., [Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor](https://arxiv.org/abs/1801.01290), ICML 2018. Relevant mainly to continuous/parameterized alternatives, not directly integer order lists.
[^dt]: Chen et al., [Decision Transformer: Reinforcement Learning via Sequence Modeling](https://arxiv.org/abs/2106.01345), NeurIPS 2021. Return-conditioned sequence learning and offline-support limitation for this application.
[^iql]: Kostrikov, Nair and Levine, [Offline Reinforcement Learning with Implicit Q-Learning](https://arxiv.org/abs/2110.06169), ICLR 2022. Supported-action offline learning/online fine-tuning alternative.
[^psro]: Lanctot et al., [A Unified Game-Theoretic Approach to Multiagent Reinforcement Learning](https://arxiv.org/abs/1711.00832), NeurIPS 2017. Response oracles and population generalization.
[^psrosurvey]: Bighashdel et al., [Policy Space Response Oracles: A Survey](https://arxiv.org/abs/2403.02227), 2024. Strategy exploration and finite-population limitations.
[^alphastar]: Vinyals et al., [Grandmaster Level in StarCraft II Using Multi-Agent Reinforcement Learning](https://www.nature.com/articles/s41586-019-1724-z), Nature 2019; architecture and pseudocode supplements. Initialization and league roles, not a compute template.
[^nfsp]: Heinrich and Silver, [Deep Reinforcement Learning from Self-Play in Imperfect-Information Games](https://arxiv.org/abs/1603.01121), 2016. Average-policy versus response learning.
[^pbt]: Jaderberg et al., [Population Based Training of Neural Networks](https://arxiv.org/abs/1711.09846), 2017. Hyperparameter/population adaptation is optional infrastructure, not the initial bottleneck.
[^dagger]: Ross, Gordon and Bagnell, [A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning](https://proceedings.mlr.press/v15/ross11a.html), AISTATS 2011. Student-induced distributions.
[^viper]: Bastani, Pu and Solar-Lezama, [Verifiable Reinforcement Learning via Policy Extraction](https://arxiv.org/html/1805.08328v2), NeurIPS 2018, section 2. Cost-sensitive tree extraction; empirical CART limitations.
[^distill]: Rusu et al., [Policy Distillation](https://arxiv.org/abs/1511.06295), 2015. Compression precedent, not guaranteed BKG4 equivalence.
[^pirl]: Verma et al., [Programmatically Interpretable Reinforcement Learning](https://arxiv.org/abs/1804.02477), ICML 2018. Neural guidance for symbolic policies.
[^propel]: Verma et al., [Imitation-Projected Programmatic Reinforcement Learning](https://arxiv.org/html/1907.05431v4), NeurIPS 2019. Iterative projection, with domain/assumption limits.
[^dsame]: Zhang, Fontaine, Hoover and Nikolaidis, [Deep Surrogate Assisted MAP-Elites for Automated Hearthstone Deckbuilding](https://arxiv.org/abs/2112.03534), 2021; [author implementation, EvoStone2](https://github.com/icaros-usc/EvoStone2). Online surrogate/QD alternative.
[^sail]: Gaier, Asteroth and Mouret, [Data-Efficient Exploration, Optimization, and Modeling of Diverse Designs through Surrogate-Assisted Illumination](https://arxiv.org/abs/1702.03713), 2017. Surrogate-assisted illumination.
[^pyribs]: ICAROS, [pyribs](https://github.com/icaros-usc/pyribs), official quality-diversity implementation. Reference for a small archive, not a required dependency.
[^dreamcoder]: Ellis et al., [DreamCoder: Bootstrapping Inductive Program Synthesis with Wake-Sleep Library Learning](https://people.csail.mit.edu/asolar/papers/EllisWNSMHCST21.pdf), PLDI 2021. Learn reusable program abstractions, not only thresholds.
[^funsearch]: Romera-Paredes et al., [Mathematical Discoveries from Program Search with Large Language Models](https://www.nature.com/articles/s41586-023-06924-6), published December 2023 / Nature 2024. Executable evaluation plus program proposals.
[^alphaevolve]: Google DeepMind, [AlphaEvolve: A Gemini-Powered Coding Agent for Designing Advanced Algorithms](https://deepmind.google/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/), May 14, 2025. Expressive code evolution; no farming or one-laptop guarantee.
[^icem]: Pinneri et al., [Sample-Efficient Cross-Entropy Method for Real-Time Planning](https://proceedings.mlr.press/v155/pinneri21a.html), CoRL 2020 / PMLR 2021. Coherent trajectory sampling and elite reuse.
[^tdmpc]: Hansen, Su and Wang, [TD-MPC2: Scalable, Robust World Models for Continuous Control](https://arxiv.org/abs/2310.16828), ICLR 2024. Learned planning comparison; learned dynamics not selected.
[^ez2]: Wang et al., [EfficientZero V2: Mastering Discrete and Continuous Control with Limited Data](https://arxiv.org/abs/2403.00564), 2024. Sample-efficient model-based alternative; no need to relearn exact dynamics.
[^hop]: Huang et al., [Efficient Adaptation in Mixed-Motive Environments via Hierarchical Opponent Modeling and Planning](https://proceedings.mlr.press/v235/huang24p.html), ICML 2024. Opponent-goal inference/planning precedent.
[^pomcp]: Silver and Veness, [Monte-Carlo Planning in Large POMDPs](https://papers.nips.cc/paper/4031-monte-carlo-planning-in-large-pomdps.pdf), NeurIPS 2010. Belief/history requirement for a simulator-based planner.
[^fusion]: Arjonilla, Saffidine and Cazenave, [Perfect Information Monte Carlo with Postponing Reasoning](https://arxiv.org/abs/2408.02380), 2024. Determinization and strategy-fusion risk.
[^rliable]: Agarwal et al., [Deep Reinforcement Learning at the Edge of the Statistical Precipice](https://arxiv.org/abs/2108.13264), NeurIPS 2021; [author implementation](https://agarwl.github.io/rliable/). Robust uncertainty reporting.
[^combinatorial]: Tkachuk et al., [Efficient Planning in Combinatorial Action Spaces with Applications to Cooperative Multi-Agent Reinforcement Learning](https://proceedings.mlr.press/v206/tkachuk23a/tkachuk23a.pdf), AISTATS 2023. Structural assumptions behind tractable joint-action planning.
[^mctx]: Google DeepMind, [mctx](https://github.com/google-deepmind/mctx), official JAX MCTS implementation and recurrent-function interface. Reference, not a reason to replace CUDA.
[^openspiel]: Google DeepMind, [OpenSpiel PSRO implementation](https://github.com/google-deepmind/open_spiel/tree/master/open_spiel/python/algorithms/psro_v2). Restricted population implementation reference.
[^vipercode]: Bastani et al., [VIPER implementation](https://github.com/obastani/viper). Tree extraction reference.
[^spirlcode]: Pertsch et al., [SPiRL implementation](https://github.com/clvrai/spirl). Skill-prior reference.
[^dsage]: ICAROS, [Deep Surrogate Assisted Generation of Environments](https://github.com/icaros-usc/dsage). Adjacent surrogate-guided generation implementation, distinct from Hearthstone EvoStone2.
[^shop]: Yusuke Hayashi / yhay81, [Shop Router 0908](https://www.kaggle.com/code/yhay81/shop-router-0908); [public source endpoint](https://www.kaggle.com/api/v1/kernels/pull/yhay81/shop-router-0908), version 2, kernel version ID 133520239. Freshly inspected author account of discovery methodology.
[^publicppo]: Alejandro Fonda, [Kaggriculture PPO Training](https://www.kaggle.com/code/alejandrofonda/kaggriculture-ppo-training); [public source endpoint](https://www.kaggle.com/api/v1/kernels/pull/alejandrofonda/kaggriculture-ppo-training), version 20, kernel version ID 129745065. Visible training/evaluation wrapper only.
[^metappo]: dzjiann, [RL of Meta Agent: 960 Matchups and a PPO Plateau](https://www.kaggle.com/competitions/kaggriculture/discussion/736439), public discussion, September 2026 context. Author-reported cycles and route-selector learning limits; indexed body accessible, direct page shell empty.
[^replayladder]: sobameshi, [I Came Here to Build a Farming Agent. I Ended Up Building a Replay-Measurement Pipeline](https://www.kaggle.com/competitions/kaggriculture/discussion/739273), public discussion, September 2026 context. Author-reported ladder/diffusion measurements; not host rules text.
[^ppothread]: NNMax and respondents, [Is Pure Self-Play PPO Viable?](https://www.kaggle.com/competitions/kaggriculture/discussion/734952), public discussion, August/September 2026 context. Uncontrolled experience reports, not comparative proof.
[^bcnoise]: [RL Seems to Have a Very Low Ceiling and Is Extremely Prone to Noise Interference](https://www.kaggle.com/competitions/kaggriculture/discussion/736917), public author report, September 2026 context. Motivates initialization/distribution controls, not a universal ceiling claim.
[^seyam]: Seyamalam, [current agent source](https://raw.githubusercontent.com/Seyamalam/Kaggriculture/main/main.py) and [repository evidence](https://github.com/Seyamalam/Kaggriculture). Public replay/closed-loop family; no independently verified current rank.
[^cok]: COK-ZhangZiliang, [current agent source](https://raw.githubusercontent.com/COK-ZhangZiliang/Kaggriculture/main/main.py). Route/meta-market implementation; versions and ancestry require pinning before evaluation.
[^lonespear]: lonespear, [current agent source](https://raw.githubusercontent.com/lonespear/kaggriculture/main/main.py). Reactive task-scheduler structure, not established leader performance.
