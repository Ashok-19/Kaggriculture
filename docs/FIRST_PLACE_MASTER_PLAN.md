# ReCurSiON: Deterministic First-Place Master Plan

## Status and Authority

- Research date: 2026-09-05.
- Audience: ReCurSiON research leads and future implementation agents.
- Verified repository: /home/nnmax/Desktop/kaggle/kaggriculture, branch main.
- Verified starting HEAD: b16d2a793e8ebbcf3d00cf23f9088599dcf2f08b; worktree initially clean.
- Status: proposed authoritative execution roadmap, with experimental decision gates. This document does not establish that its search architecture works.
- This roadmap supersedes unimplemented ASPE Phases 4 onward. Completed ASPE Phases 0-3 remain reusable infrastructure. The historical ASPE document remains the record of those phases, not an alternative execution schedule.
- Deterministic Bonsai/BKG4 remains the runtime. No neural policy, value function, proposer, behavioral cloning, PPO, RL-over-edits, or LLM in the automated search loop.
- No implementation was changed or search campaign run for this research.
- Primary repository evidence: [FIRST_PLACE_PIPELINE_CONTEXT.md](FIRST_PLACE_PIPELINE_CONTEXT.md). Source-tree reconstruction was deliberately avoided.

## 1. First-Place Verdict

**The current planned pipeline needs a major redesign of discovery, not a replacement of the runtime or safety stack.**

Typed mutations plus a semantic archive can reduce wasted syntax, but they do not, by themselves, supply better strategic questions, expose delayed consequences, discover economically feasible combinations, or produce counterexamples that constrain the next proposal. Increasing their population size is not an adequate first-place strategy.

**One primary architecture: Counterexample-Guided Continuation Program Repair (CCPR).**

CCPR is a deterministic best-first synthesizer of bounded symbolic options, driven by exact paired simulator continuations from reachable internal self-play states. It alternates body repair and legal predicate synthesis, turns BUILD failures into scoped constraints, and periodically checks complete games. The same synthesis engine constructs internal challengers under a different objective. A coverage-guided challenge archive, a mechanics-testing track, and a certified symbolic vocabulary supply new problems and reusable components.

**One fallback architecture: Deterministic Semantic Quality-Diversity Evolution (DSQD).**

Only activate DSQD after the primary fails its bounded falsification. It uses the same legal representation, telemetry, safety pipeline, and any independently validated snapshot infrastructure, but generates bodies through deterministic typed neighborhood schedules and niche-local selection rather than counterexample-guided best-first repair. It is not a second permanent primary oracle.

This choice is a research judgment, not a claim that a paper has solved Kaggriculture. Confidence is high that the missing layers matter, moderate that CCPR is the best first experiment, and low in any numerical probability of finishing first. Neither a near-100% win-rate claim nor a first-place probability can be supported by the current evidence.

### The five largest bottlenecks

| Rank | Bottleneck | Missing or weak? | Why it is strategically important |
| --- | --- | --- | --- |
| 1 | Decision-state experiments and causal credit | Missing as a demonstrated search subsystem | Whole-game scores confound where a change mattered, what enabled it, and whether the parent could exploit its result. |
| 2 | Hypothesis generation that changes after failure | Weak existing proposal systems; active refinement missing | RLRMB and Direct search generated almost no reusable capability. Passive memory does not eliminate the next bad hypothesis. |
| 3 | Reachable challenge diversity and information-aware state abstraction | Weak eight-lane organization; automatic frontier missing | A better solver of the same few regimes still saturates. Hidden-state or coarse-feature aliases also make apparently useful bodies unroutable. |
| 4 | Feasible strategic vocabulary and multi-option economics | Primitive representation exists; discovery and combination missing | Land, labor, liquidity, production and liquidation must work together. Independently attractive actions can form a losing economy. |
| 5 | Mechanics discovery plus cumulative portfolio assurance | Mechanics lane weak; large-scale assurance unproven | A profitable legal ordering effect can dominate many parameter edits, while independently safe specialists can regress after composition. |

A cross-cutting sixth risk is **experimental governance**: small repeatedly queried DEV/SEALED panels cannot support unlimited autonomous improvement without adaptive overfitting. Preserve the existing gates and strengthen their lifecycle, not their thresholds.

### What the repository evidence actually says

The frozen 9,600-game comparisons produced zero cross-regime specialists, zero DEV survivors, and zero safe commits for both RLRMB and Direct Bonsai. Direct's 13 limited-harm candidates collapsed to roughly one strategic phenotype; its DEV attempt produced 0 L->W and 8 W->L. These results diagnose low-information proposal search, not inadequate GPU throughput.

Conversely, the historical routed specialist that changed raw protected W->L from 10 to 0 while retaining 4 of 5 target L->W is real evidence for local specialization with fallback. It is not evidence that arbitrary bodies can be rescued by gates. ASPE Phases 0-3 establish reproducibility, execution support, and memory; none establishes superior specialist discovery. All numbers in this paragraph come from the repository context, not a new rerun.

## 2. Research Findings That Change the Design

### 2.1 Synthesis rather than blind variation, but not a monolithic SMT game

CEGIS explicitly reuses counterexamples to constrain subsequent candidates. Sketching separates high-level structure from holes solved by a synthesizer. This is the relevant architectural lesson: make the next candidate answer a known failure instead of merely differ from its parent. Kaggriculture lacks a tractable complete winning specification, so our simulation checker is an empirical counterexample oracle, not a proof of universal correctness. [Solar-Lezama, Program Synthesis By Sketching, 2008](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2008/EECS-2008-176.html)

Bounded enumeration is not inherently naive: CVC4SY combines optimized enumeration with solver-guided strategies. Use bitset enumeration first for small predicate/body holes; introduce one pinned solver only where it demonstrably helps. Do not encode the entire economy and 720-turn adversarial game into SMT. [Reynolds et al., CVC4SY, CAV 2019](https://theory.stanford.edu/~barrett/pubs/RBN%2B19-abstract.html), [cvc5 SyGuS implementation example](https://cvc5.github.io/docs/cvc5-1.3.0/examples/sygus-fun.html)

Probe learns grammar preferences from partial solutions during enumeration without requiring an offline corpus. Adopt the narrower, transparent mechanism of exact success counts and deterministic rule costs. Do not assume its input/output synthesis benchmarks establish long-horizon game performance. [Barke, Peleg and Polikarpova, OOPSLA 2020](https://arxiv.org/abs/2010.08663)

The 2025 preprint/POPL 2026 work on oriented metrics supports abstraction refinement and semantic pruning when a domain supplies the necessary structure. Its string/bitvector results do not give a distance-to-winning-program metric for Kaggriculture. Therefore use proved local equivalences and reversible empirical deduplication, not a fabricated admissible strategic bound. [Meyer and Tepe, POPL 2026](https://popl26.sigplan.org/details/POPL-2026-popl-research-papers/77/Oriented-Metrics-for-Bottom-Up-Enumerative-Synthesis)

### 2.2 Return to a state, then improve a continuation

Go-Explore separates returning to an archived state from exploring beyond it. Reuse that principle and reachable-state provenance, not its neural robustification machinery. Our replacement for robustification is legal symbolic generalization followed by independent full-game certification. [Ecoffet et al., First return, then explore, Nature 2021](https://arxiv.org/abs/2004.12919)

Classical rollout algorithms improve a base heuristic by evaluating possible continuations. Their guarantees require particular assumptions; finite state families, unknown opponents, imperfect option search, and compiled gates do not inherit those guarantees. The useful mechanism is exact comparison against PARENT, including a terminal continuation. [Bertsekas, Tsitsiklis and Wu, Journal of Heuristics 1997](https://www.mit.edu/~jnt/Papers/J066-97-rollout.pdf)

Width-based planning and general-policy sketches support decomposing long plans around features and subgoals. They justify trying shallow option composition before primitive-action trees. They do not establish low planning width for this game or make novelty pruning sound for winning. [Lipovetzky and Geffner, ECAI 2012](https://nirlipo.github.io/publication/lipovetzky-2012-width/), [Bonet and Geffner, JAIR 2024](https://publications.rwth-aachen.de/record/988540)

### 2.3 Treat challenge discovery like structured fuzzing

FairFuzz prioritizes rare branches and protects input components needed to reach them. Translate this into preserving a useful self-play prefix while varying the decisions around an underexplored transition. Rarity alone is insufficient; maintain separate priorities for reachability, recoverable regret, and coverage. [Lemieux and Sen, FairFuzz, ASE 2018](https://www.carolemieux.com/fairfuzz-ase18.pdf)

AFL++ provides target-specific mutator interfaces, while its evaluation also demonstrates that techniques can help some targets and hurt others. Use its testing principles and optional native harness integration, not its complete stochastic scheduling stack as an unexamined dependency. [Fioraldi et al., WOOT 2020](https://www.usenix.org/conference/woot20/presentation/fioraldi), [AFL++ implementation](https://github.com/AFLplusplus/AFLplusplus)

Fixed RNG seeds alone do not guarantee reproducibility. The LLVM Entropic implementation discussion specifically removed time-based weighting because it broke fixed-seed determinism. Our budgets and scheduling decisions must therefore use evaluation counts, deterministic work limits, and ordered batch commits, not elapsed time or worker completion order. [LLVM Entropic implementation discussion, 2020](https://reviews.llvm.org/D73776?id=241686)

### 2.4 Preserve diversity without confusing it with progress

MAP-Elites preserves different high-performing behaviors; CVT-MAP-Elites addresses the exponential growth of a Cartesian descriptor grid. Keep a bounded sparse archive with local competition rather than building a giant grid over dozens of state features. [Mouret and Clune, 2015](https://arxiv.org/abs/1504.04909), [Vassiliades et al., 2016](https://arxiv.org/abs/1610.05729)

Nontransitive games can cycle through policies that beat their immediate predecessors. Keep internal historical capabilities and payoff-profile diversity, not merely the latest champion and its strongest current exploiter. A finite internal payoff table is coverage evidence, not a Nash-equilibrium or exploitability certificate against all possible opponents. [Balduzzi et al., Open-ended Learning in Symmetric Zero-sum Games, 2019](https://arxiv.org/abs/1901.08106)

Minimal-criterion coevolution motivates keeping challenging but attainable tasks. Our attainability evidence is a concrete reachable witness and, when available, an improving response. Maintain a small frontier for currently unsolved tasks rather than declaring them impossible because bounded search failed. [Brant and Stanley, GECCO 2017](https://stars.library.ucf.edu/scopus2015/7496/)

### 2.5 Learn a symbolic library, not a neural proposer

Stitch discovers reusable abstractions by corpus-guided compression. Adapt typed anti-unification and minimum-description-length ranking to successful internal programs, then demand new-task improvement: compressing syntax is not enough. Library functions are offline search conveniences expanded into ordinary BKG4. [Bowers et al., Top-Down Synthesis for Library Learning, POPL 2023](https://arxiv.org/abs/2211.16605), [Stitch implementation](https://github.com/mlb2251/stitch)

Large-neighborhood search supplies a useful controlled escape: remove a causally implicated group of choices, then solve that bounded hole. Its adaptive operator allocation is relevant; its transportation results are not a game-performance forecast. [Ropke and Pisinger, Transportation Science 2006](https://pubsonline.informs.org/doi/epdf/10.1287/trsc.1050.0135)

### 2.6 Current Kaggriculture mechanics and evidence limits

The official README describes a 720-turn season, terminal bank-money scoring, private inventories, duplicate shop instances, capacity-limited deposits, and different market curves. These favor observation-conditioned economic options over fixed calendars. Its description of immediate unchanged-market buy/sell round trips provides a useful negative control, not evidence for free arbitrage. [Official Kaggriculture README, accessed 2026-09-05](https://raw.githubusercontent.com/Kaggle/kaggle-environments/master/kaggle_environments/envs/kaggriculture/README.md)

The latest engine-file commit returned by the official GitHub API was **28b6d8af3ce73926b3d0fda1410c1ddd8384ab8c**, dated 2026-08-15. It introduced scarcity-side hinge behavior for carrot, tomato and egg. Early-August strategy conclusions may therefore be stale. This pins the engine file inspected, not the live competition installation or every package dependency. [Official change #1399](https://github.com/Kaggle/kaggle-environments/commit/28b6d8af3ce73926b3d0fda1410c1ddd8384ab8c)

**Critical source findings:** in that engine, daily weed draws occur conditionally on empty tiles, and the same daily RNG subsequently draws a shop. Thus an action-dependent draw count can change subsequent events. Market units are quoted before sequential commits. Animal production applies existing CARE credit before banking the current day's credit. The README's narrative is not a substitute for these transition details. Forks must reproduce the pinned implementation, not freeze future shops or assume that same-seed branches share identical events. This is a fidelity requirement and an inferred coupling, not a demonstrated profitable exploit. [Pinned official engine: _spawn_weeds, _end_of_day, _process_market, _daily_refresh_animals](https://raw.githubusercontent.com/Kaggle/kaggle-environments/28b6d8af3ce73926b3d0fda1410c1ddd8384ab8c/kaggle_environments/envs/kaggriculture/kaggriculture.py)

A public deterministic implementation reports that workforce assignment, economic interactions, and apparently ineffective but unreachable actions affected its development. Treat this as unreplicated strategic-space evidence, not as BUILD data, transferable numerical results, or a current leaderboard claim. Do not import its agents, parameters, or opponent labels into construction. [lonespear public development report, dated 2026-08-07](https://raw.githubusercontent.com/lonespear/kaggriculture/main/README.md)

The supplied [lane discussion 734212](https://www.kaggle.com/competitions/kaggriculture/discussion/734212), [competition rules](https://www.kaggle.com/competitions/kaggriculture/rules), and [overview](https://www.kaggle.com/competitions/kaggriculture/overview) returned page shells with no readable content. Exact-ID search did not recover the lane discussion. The installed Kaggle CLI lacks the documented pages command. Consequently:
- This report does not claim to have read or verified the lane discussion's method.
- Current deadline, submission-selection policy, resource limits, and exploit permissions require official confirmation before release.
- In-game mechanics research may proceed locally; no undocumented bug-dependent release is authorized by this report.
- A secondary timeline mirror is not sufficient to certify competition deadlines.
- Native parity reported by the repository is preserved as evidence; no claim is made that this research independently verified its alignment to today's live engine.

## 3. Hard Architecture Comparison

These are qualitative engineering judgments for this repository, not head-to-head experimental results. All rows assume the same legal observation boundary and final BKG4 compilation. High expressivity does not imply efficient discovery.

### Search and credit

| Architecture | Candidate expressivity | Search-space efficiency | Long-horizon reasoning | Failure reuse |
| --- | --- | --- | --- | --- |
| **CCPR: best-first counterexample-guided continuation repair** | Bounded reactive options; later compiled option combinations | Strong potential from localized holes, exact witnesses, gate bitsets and prefix reuse; vulnerable to bad abstractions | Paired terminal continuations plus event-level option composition | Active body-specific negatives, gate refinement, scoped no-goods and causal neighborhood selection |
| **DSQD: deterministic typed semantic evolution** | Same DSL; flexible joint changes without a full specification | Moderate; semantic screening helps but most variation can remain low-information | Requires added continuation credit; evolution alone does not solve delay | Counts, parent selection and context-conditioned operators; weaker direct elimination |
| Pure bottom-up enumeration / version spaces | Complete within a finite bounded grammar | Excellent for tiny predicates; combinatorial for multi-phase bodies and constants | None intrinsically; terminal evaluations remain expensive | Strong for exact input/output constraints; strategic examples are conditional rather than exact target actions |
| Monolithic SMT / SyGuS / bounded game solving | High within a faithful bounded encoding | Attractive for arithmetic guards; poor expected scaling for the complete engine | Explicit horizon encoding grows rapidly; opponent uncertainty multiplies it | Strong logical clauses when specifications are formal |
| Deterministic MCTS over programs or actions | Broad grammar/tree search | Tree statistics allocate effort, but many partial programs have uninformative completions | Deep lookahead possible; branching and deceptive rollout values remain | Visit/value statistics; counterexample reuse requires additional machinery |
| Deterministic LNS / tabu program repair | Broad local neighborhoods, including joint parameter/body edits | Good when the right block to destroy is known | Better multi-step edits, but no intrinsic terminal credit | Tabu and operator history; excellent as a CCPR neighborhood, incomplete as the whole system |
| Equality saturation / superoptimization | Equivalent rewrites or bounded alternative implementations | Strong for canonicalization and small implementation optimization | Does not discover a better strategic objective merely by rewriting equivalents | Equivalence proofs and concrete failures; wrong abstraction for whole-game policy improvement |
| HTN / width-based / best-first strategic planning | Rich subgoal structures, constrained by macro vocabulary | Potentially strong with an accurate abstraction; large modeling cost | Best intrinsic temporal structure among alternatives | Failed subgoals refine abstractions; successful plans become options |

### Reuse, diversity and scaling

| Architecture | Successful component reuse | Diversity and anti-saturation | CUDA batching | Reproducibility | Relative engineering cost | Likely Kaggriculture failure |
| --- | --- | --- | --- | --- | --- | --- |
| CCPR | Explicit typed library and cached option continuations | New reachable challenges, bounded exploration quota, abstraction refinement | High for concrete candidate x state-family checks; CPU synthesis remains serial/bounded | Intrinsic with fixed order and deterministic solver limits | Medium initially; high only for gated later layers | Local wins fail to survive legal activation or changed economics |
| DSQD | Subtree/motif reuse | Strong niche preservation; still dependent on descriptors and proposal quality | High | Deterministic schedules preferred; seeded variation requires pinned execution | Low-medium on existing infrastructure | More distinct syntax/effects but no independent useful specialist |
| Pure enumeration | Excellent subterm tables | Exhaustive only within small bounds; vocabulary expansion costly | Moderate-high for body batches; gate enumeration mainly CPU | High | Low for gates, high for full bodies | Mostly redundant or never-activated programs |
| Monolithic SMT | Solver lemmas and synthesized terms | No automatic strategic diversity objective | Low for symbolic solving; high only after concretization | Pinned single-thread runs and resource limits required | Very high for the whole engine | Timeout, wrong model, or seed-specific perfect-information plans |
| MCTS | Transpositions/subtree reuse | Exploration bonus is not strategic novelty | Medium; adaptive tree expansion fragments batches | Fixed tie-breaks, simulations and ordered updates | Medium-high | Search statistics amplify misleading short completions |
| LNS / tabu | Good local block reuse | Escapes some local optima; challenge generation still absent | High for bounded repaired neighborhoods | High under fixed schedules | Medium | Repairs the wrong neighborhood repeatedly |
| Equality saturation / superoptimization | Excellent exact expression reuse | Equivalence classes preserve behavior, not strategic novelty | Low for e-graph/SMT; batch test candidates possible | High if all extraction/rewrite choices pinned | Medium locally, very high globally | Unsound rewrite of stateful actions or no new capability |
| HTN / width planning | Explicit options and subgoals | New subgoals can broaden behavior; fixed manual decomposition saturates | Moderate for frontier continuations | High under deterministic queue rules | High before a reusable option library exists | Optimizes an abstract economy that cannot be executed by workers |

### Selection and rejected promises

CCPR wins the initial decision because it uses all five available assets together: exact evaluation, a compact option representation, phenotype telemetry, persistent memory, and PARENT fallback. It creates a tractable *local specification* from paired terminal consequences instead of pretending the entire game has a cheap symbolic specification.

DSQD is the fallback because it is cheaper to build, uses the same infrastructure, and can tolerate strategic effects that resist local symbolic decomposition. It is not selected merely because ASPE work already exists.

Do not import guarantees of geometric semantic GP into a stateful game. Its classic constructions concern function-output semantics; averaging or interpolating terminal-effect vectors does not construct a corresponding legal reactive controller. [Moraglio, Krawiec and Johnson, PPSN 2012](https://citeseerx.ist.psu.edu/document?doi=050e6c0dbbe93a9e5c8f1491e868761406054e96&repid=rep1&type=pdf)

STOKE shows that stochastic test-guided search can be useful for loop-free binary optimization, not that large seeded policy mutation is the best specialist oracle. Retain bounded superoptimization ideas for minimization after discovery. [Schkufza, Sharma and Aiken, 2012/ASPLOS 2013](https://arxiv.org/abs/1211.0557)

Equality saturation is for justified equalities; observed equality on a sentinel is not a valid rewrite rule. Apply it only if simple canonicalization is insufficient and a proved local rewrite set exists. [Willsey et al., egg, POPL 2021](https://arxiv.org/abs/2004.03082)

Recent work is not automatically admissible under the directive. The AAAI 2024 MCTS SyGuS system uses learned policy/value functions, and the 2025 meta-QD work parameterizes competition with neural attention. Their reported gains do not justify removing those models and claiming equivalent performance. Neither is a primary component here. [Parsert and Polgreen, AAAI 2024](https://ojs.aaai.org/index.php/AAAI/article/view/28938), [Faldor, Lange and Cully, 2025](https://arxiv.org/abs/2502.02190)

A*, branch-and-bound and partial-order reductions are useful only where bounds or action independence are justified. Economic milestones and observed margin are not admissible heuristics. Maintain a non-pruned exploration queue so heuristic ranking cannot permanently erase delayed-payoff candidates. [Fast Downward's explicit heuristic-property contracts](https://www.fast-downward.org/latest/documentation/search/Evaluator/)

## 4. The Selected End-to-End System

~~~~text
assets/champion + internal capability league
                  |
        fresh deterministic self-play
                  |
      reachable challenge/state archive <--- legal mechanics fuzzing
                  |                               |
      family selection + regret witnesses        | confirmed opportunity
                  |                               |
         CCPR specialist synthesizer <------------+
  [best-first typed holes + exact forked continuations]
                  ^
                  | BUILD-only counterexamples, counts,
                  | exact caches, successful symbolic library
                  |
       legal initiation/body generalization
                  |
        full BUILD -> independent INTERNAL_DEV
                  |
        existing binary routing / PARENT
                  |
        TRANSFER -> VALIDATION -> certification CEGIS
                  |
      locked whole composition -> SEALED verifier
                  |
       materialization + native parity + official canary
                  |
      atomic certified champion update / local evidence
                  |
           new internal challenges
                  +-------------------------------> repeat
~~~~

Global route solving appears only when measured portfolio conflicts justify it. It replaces the *route construction step*, never the gates below it. External agents enter a separate HOLDOUT evaluation path, not the arrow back into construction.

### 4.1 The search unit: a reachable decision-state family

A challenge is not a seed label or an opponent identity. It is a family of reachable internal game states with:
- Legal observable descriptors, an event/decision boundary, and an identified uncertainty or failure.
- Original internal policy hashes, configuration, seed provenance, seat and full replayable prefix.
- Full simulator state and controller execution state, kept separate from features the program may read.
- At least one reference PARENT continuation and, when discovered, a better concrete response.
- Related states from distinct trajectory roots, including successful controls and near-boundary cases.

Capture at complete decision boundaries initially, not halfway through a joint-action commit. Snapshot contents must include both private states, all game counters and terminal status, market/town state, configuration, deterministic randomness provenance, both controllers' execution memory, and the exact interpreter/ABI version. Hidden simulator information is allowed for executing the environment, never as a runtime predicate, action selector, descriptor-derived seed fingerprint, or per-instance plan lookup.

A family starts with existing observable bins: time remaining, liquidity, production mix, labor scale, land, market scarcity, observed shop demand, and inventory pressure. Richer diagnostic aggregates may be computed offline, but adding one to the policy DSL requires its own legal-feature and compilation parity test.

Maintain three distinct notions:
1. **Exact state identity** for memoized execution, including controller memory and randomness.
2. **Search abstraction** for prioritizing related challenges.
3. **Runtime information** available to the deterministic policy.

Never merge these. Two states that look identical to a gate may require different responses because the gate omits a legal feature, the body needs reactive behavior, or genuinely hidden information differs. The first two can justify refinement; the third requires robust behavior or fallback, not a hidden-state gate.

Returning to an internal snapshot is a search acceleration, not replay-corpus policy construction. Every strategic witness must have a fresh internal provenance chain. Arbitrary hand-edited states belong to engine testing until reconstructed by legal play.

### 4.2 Exact continuation semantics

For state s and candidate option p, measure:

~~~~text
Delta_terminal(s, p, opponents)
  = final_money_margin(s, p then PARENT, reactive_opponents)
  - final_money_margin(s, PARENT, reactive_opponents)
~~~~

Also retain W/D/L changes, action effects and intermediate resource deltas. Use multiple frozen internal opponent policies as separate strata; recompute their actions on the modified state. Replaying the opponent's original future actions would measure an intervention against a script, not a response against a policy.

The parent must continue from the correct execution state. Restoring only the farm while resetting the parent or active specialist is invalid. A forced one-time intervention used to diagnose value is not a deployable gate; later full-game execution must use ordinary SpecialistProgram activation, termination, possible reactivation and fallback semantics.

Use horizons at event boundaries, initially up to 24 and 96 turns for information, and the actual terminal horizon for strategic value. An option may terminate after a few turns while its evaluation continues hundreds of turns under PARENT. No early money penalty or milestone score may hard-reject an otherwise legal investment solely for having delayed returns.

Use paired seeds, but allow action-dependent future randomness to diverge according to the engine. Report this as total effect under the real transition system. A diagnostic experiment that holds future events fixed is explicitly a different, non-certifying model and is not needed in the primary implementation.

Prefix reuse saves work, but does not automatically save orders of magnitude. For N candidates at turn t and terminal horizon T:

~~~~text
full prefix cost       ~ N * T
shared prefix cost     ~ t + N * (T - t)
limiting prefix speedup ~ T / (T - t)
~~~~

At t=360 and T=720 the limiting saving is about 2x, not 100x. Orders-of-magnitude improvement in useful-specialist yield would have to come from better questions, early informative rejection, reuse, and generalization, not this arithmetic. Early-game decisions still need nearly complete continuations.

### 4.3 CCPR proposal generation, precisely

Inputs are a locked parent, one BUILD challenge family, a bounded typed grammar, a BUILD counterexample set, and context-versioned memory. Output is a compact program plus evidence, never a champion.

1. **Select a repair question.** Use measured recoverable regret, an underexplored transition, or disagreement among existing options. Begin before the suspected cause, not automatically at the terminal loss. If late repair cannot help, scan earlier checkpoints in a predetermined coarse-to-fine order.
2. **Choose a small sketch.** Start with one body phase, mandatory fallback and one termination rule. Holes specify action-family source, bounded parameter values, event termination and, only as needed, a branch or second phase. Start from PARENT, internal motifs or causally relevant archive components.
3. **Enumerate by a deterministic best-first queue.** Order by repair compatibility, unresolved witness classes, quantized rule cost, program size and canonical ID. Evaluate completed programs; evaluate partial programs only with an explicit PARENT completion. That completion is a heuristic estimate, not a bound on every extension.
4. **Solve small predicates separately.** Enumerate legal feature thresholds using BUILD values and engine-defined boundaries. Evaluate predicates as bitsets. Canonicalize contradictory/redundant clauses. Prefer the fewest clauses/literals and widest empirically supported region, subject to body-specific negative examples.
5. **Batch exact checks.** Start on the target family plus nearby negative controls, then execute diagnostic horizons or terminal suffixes as justified by Phase 1. Retain uncertain and delayed-payoff candidates. Promote informative candidates to complete BUILD.
6. **Refine on a concrete failure.** Determine whether the problem is initiation, infeasible resources, wrong body choice, missing reaction, termination, parent continuation, or an unobservable distinction. Change the implicated hole or gate; do not blindly mutate all fields.
7. **Broaden support before DEV.** Test the same program over other roots and members in the relevant regime, not cloned continuations counted as independent evidence. A candidate that only works at one injected checkpoint remains a diagnostic result.
8. **Freeze a bounded shortlist.** Only then query INTERNAL_DEV. No DEV row, derivative feature, or operator-success label enters the synthesizer.
9. **Send survivors through existing routing and certification.** All acceptance authority stays downstream.

A BUILD counterexample is **body-and-context-specific**. A losing outcome for body A does not mean the state must exclude every specialist. Unrouted specialists may remain useful despite raw harms; the current limited-harm eligibility rules still apply. Do not impose a new global-safe-body requirement that would discard the historically proven routing opportunity.

Start without a general solver dependency. If predicate enumeration becomes the bottleneck, use one pinned single-thread SAT/SMT/MaxSAT implementation for finite truth-table constraints, discrete thresholds, and bounded grammar holes. cvc5 is suitable for SyGuS-shaped holes; a MaxSAT backend can solve weighted coverage with hard exclusions. The experiment chooses the smallest necessary integration, not a suite of competing solvers.

### 4.4 How failure changes evaluation number 1,000,000

Maintain the following BUILD-only records:

| Evidence | Deterministic response | Scope and safeguard |
| --- | --- | --- |
| Exact repeated program/context evaluation | Return cached result | Full engine/parent/opponent/state/ABI context must match |
| Initiation does not fire in target family | Solve a different legal gate or select another family | ZERO_ACTIVATION on one sentinel is not a global rejection |
| Wrong body at a known legal observation | Add body-specific negative bitset and refine gate/body | Preserve contradictory labels as an abstraction issue, not a vote |
| Missing input, budget, capacity or action prerequisite | Prioritize a recovery/preparation hole | A diagnostic resource heuristic cannot prove impossibility |
| Harmful phase or late termination | Re-evaluate with that phase removed/ended earlier | Verify by paired intervention before attributing causality |
| Repeated harmful complete candidate | Exact no-good or bounded tabu entry | Do not prune all extensions of a bad partial program |
| Proven gate contradiction or local semantic equivalence | Permanent grammar pruning under the proof's assumptions | No proof from hashes or sampled effects |
| Repeated partial successes | Lower the cost of contributing grammar rules | Exact integer counts, lineage caps and exploration floor |
| Same abstract state, inconsistent outcomes | Split on legal features or require reactive robustness | Never split on hidden state, seed, seat identity or opponent identity |

For each context/operator record attempts a, informative distinct outcomes i, and generalizable BUILD successes u. A simple initial ordering statistic is the exact rational (u+1)/(a+2), with separate novelty and failure counters; compare by integer cross-multiplication. Do not call these calibrated success probabilities. Update only at fixed batch boundaries and cap contributions from one lineage so hundreds of clones do not masquerade as hundreds of independent successes.

Use a predeclared 64-proposal schedule: 32 witness-directed repairs, 16 least-served coverage-family proposals, 8 lowest-complexity unexplored constructions, and 8 archive-component recombinations. These are queues within one synthesizer, not separate search architectures. The split is a starting configuration to ablate, not a permanent magic ratio.

Scoped constraints, improved grammar order and new components make later search different. Changing parent, engine, representation or challenge abstraction versions invalidates dependent empirical bans and effect caches. Retain old evidence as provenance, not as current truth.

### 4.5 Long-horizon credit and combinations

**Primary credit: paired terminal option advantage.** Intermediate telemetry explains why an option may be promising; it never replaces terminal outcomes or final zero-regression certification.

For a successful option, run bounded removal tests over its phases/action families. For a promising pair A, B, measure all four continuations PARENT, A, B, and A followed by B:

~~~~text
interaction(A, B) = Delta(A then B) - Delta(A) - Delta(B)
~~~~

This is context-specific interaction evidence, not a general additive decomposition. Store the exact family and policies. Include an ordering reversal when legal, because A then B need not equal B then A. Do not approximate thousands of options by a globally additive value table.

To avoid the parent-continuation trap, allocate a bounded quota to option pairs even when neither standalone option is profitable. A capital investment may only pay with a later production and liquidation option. Retain reversible enabling states and positive joint effects as stepping stones, not as certified specialists.

Later, search event-level option sequences with a deterministic beam, initially width 16-64 and depth 2-3. Each frontier node is an exact forked state plus controller context. Land expansion, labor allocation and selling schedules must be executable, not merely feasible in a relaxed economic model. Use resource bounds for ranking unless their admissibility is proved.

Backward induction is allowed on a small fixed continuation graph or narrow endgame subproblem. It is not a claim to solve the whole partially observed adversarial game. Primitive-action MCTS across all workers for 720 turns is not the next phase.

### 4.6 Reusable vocabulary and representation verdict

**SpecialistProgram v1 is sufficient for the first bounded falsification; it is not established as sufficient for first place.**

Its DNF initiation, few phases, bounded reactions, recovery and fallback are a sensible compact option IR. Its documented strategic feature list omits many potentially important aggregate notions: urgent work, crop age/yield distribution, carried-versus-shed inventory pressure, feed runway, service distance and pending productive work. Existing body motifs may encode some of these effects; this report did not inspect their implementation and does not assert otherwise.

Do not immediately add coordinates, arbitrary loops, a new VM, or large persistent memory. First produce a **representation witness**: an improving legal action/option experiment that v1 cannot express or cannot separate from harmful cases using its allowed observations. An alias in a coarse search descriptor alone is not such a witness.

If witnesses justify it, add the smallest deterministic observable aggregate or bounded control construct, prove CPU/CUDA/materialization agreement, and rerun the equal-budget experiment under a new version. Spatial reasoning should use legal task/region summaries and stable local ordering, not seed-specific tile addresses.

Library induction protocol:
1. Use successful BUILD programs from at least three independent lineages and more than one root family. Do not mine DEV/SEALED winners as training data.
2. Canonicalize fragments and anti-unify typed differences into parameters.
3. Rank by corpus compression minus definition cost, expanded node cost, parameter count and unsupported preconditions.
4. Preserve action ordering and family ownership. Do not infer commutativity from similar terminal margins.
5. Test reuse on new BUILD tasks withheld from the induction corpus; compare against the same search without the library.
6. Promote only fragments that improve new-task search efficiency; retain a primitive-only exploration floor.
7. Expand each call to ordinary BKG4 and certify it like any other candidate. No dynamic library learner runs in the submission.

Initial domain names such as market recovery or capital preservation are descriptive tags, not hard-coded eternal templates. Successful primitives get their operational meaning from preconditions, effects, termination, and repeated evidence.

### 4.7 The open-ended challenge generator

Retain the eight lanes as reporting labels and initial coverage strata. They are not archive cells and not a permanent enumeration of the strategy space.

Use one bounded reachable-state archive with several witness roles: cheapest reproducible prefix, highest measured recoverable regret, rare transition, and unsolved frontier. Keep multiple witnesses when private context or public geometry differs materially; never reduce an entire strategic region to one lucky seed.

Challenge proposals come from:
- Fresh internal games with parent, current internal adversaries and capability-preserving checkpoints.
- Deterministic edits to internal programs or short action prefixes near selected events.
- Pairings that expose internal payoff-profile gaps, including challengers that are not globally strongest.
- Existing options inserted earlier in a legal prefix to reach new economic basins.
- Confirmed reachable mechanics witnesses.
- Targeted combinations of known options and previously incompatible resource regimes.

Coverage descriptors include semantic state bins, action-family x precondition outcomes, selected state-transition pairs, mechanics branch/boundary coverage, and option effect classes. Use trace hashes for exact identity, not as distances that reward arbitrary trajectory differences.

Automatic lane splitting is **counterexample-driven partition refinement**: after a cell has enough independent roots, test a deterministic legal-feature split that separates conflicting responses or regret. Require both children to have independent support and reduced aliasing on additional BUILD roots. Version the abstraction; do not claim novelty merely because bins were renamed.

Corpus minimization is deterministic set cover over rare coverage and failure witnesses, with seed/seat/root diversity constraints. Retain compact prefixes/snapshots outside git and commit their recipes, hashes and minimized evidence. There is no reason to commit every rollout.

### 4.8 Reproducibility and reversible semantic compression

An experiment manifest pins engine/configuration, parent and internal members, panel/root provenance, grammar/library, feature abstraction, compiler and flags, CUDA architecture, solver version/settings, batch size, queue tie-breaks and work budgets. Aggregate GPU results in canonical candidate/case order. Hash-table iteration, parallel completion order, floating-point reductions, adaptive batch sizes and timeouts must not decide which candidate is generated next. Preserve the existing engine arithmetic; do not quantize simulation values to make parity appear easier.

Serialize canonical state fields, not uninitialized struct padding. Hash matches used for exact execution reuse require collision-safe record equality or a verified canonical identity. Store a deterministic restart checkpoint at each committed batch; an interrupted batch is replayed, not partly incorporated. Acceptance materialization must reproduce the same bytes from source and manifest.

A sentinel/effect duplicate is an empirical compression decision, not a theorem. Retain regenerable alternative representatives and the context that made them appear equal. Re-probe them when a new challenge splits the class; audit a fixed one-in-twenty of deferred candidates in canonical order within the normal budget. Preserve rare activations and enabling states. Only proved contradictions/equivalences or identical complete context evaluations justify permanent pruning.

## 5. Mechanics Discovery Is a First-Class Component

**Yes: build a dedicated mechanics differential-testing and opportunity-discovery subsystem.** It shares the state-forking harness and challenge archive; it is not a separate large Python search engine.

There are two queues with different authority:

1. **Correctness queue:** compare native CPU and CUDA transitions; use bounded official-engine checks to adjudicate. A mismatch freezes affected results, triggers repair and invalidates dependent evidence. A profitable CUDA-only behavior is a bug, never a specialist.
2. **Opportunity queue:** search unusual but equal, legal engine transitions for terminal advantage and cross-root reproducibility. Only reachable internal witnesses can become BUILD challenges.

Proposed initial experiment families, not claims of profitable exploits:

| Family | Structured test | Strategic follow-through |
| --- | --- | --- |
| Shop-demand and pricing boundaries | Reach duplicate-demand histories; test inventory on both sides of price-shape boundaries | Evaluate production switching and sale timing under reactive internal opponents |
| Market quote/commit interactions | Permute legal order slots, quantities, buys/sells and seats around resource limits | Distinguish a true marginal advantage from a same-turn accounting illusion |
| Shed pressure and liquidation | Vary deposit timing, carried items, capacity thresholds and sale-before-purchase order | Measure terminal preservation, not merely a transient inventory increase |
| Animal survival and care | Enumerate legal feed/care/harvest orderings around daily and production boundaries | Include upkeep opportunity cost and terminal conversion |
| Fertilizer timing | Test refresh, repeat application and harvest/production boundary orderings | Retain only reproducible net gains after labor and purchase costs |
| Multi-worker resource contention | Test shared prerequisite demand and action ordering under constrained stock | Synthesize coordination conditions rather than a one-trace action sequence |
| State-dependent randomness | Compare reachable occupancy-changing interventions across seeds and seats | Fidelity and robustness tests only; no hidden-seed inference or future-event routing |

Generate input variations in a stable order around exact boundaries: zero, one, just below/at/above capacity, adjacent time boundaries, and short legal permutations. Preserve the prefix needed to reach the rare event. Delta-minimize action sequences while replaying reachability, then generalize the witness.

Metamorphic checks must have valid preconditions. Reordering independent actions is not valid if they share inventory or pricing; seat-swapping is not automatically a symmetry of sequential processing and RNG consumption. Failure of an assumed symmetry can be a useful test result, but not automatically an engine bug.

Illegal-state mutation remains useful for robustness tests, never evidence of strategic gain. Avoid malformed-input crashes, denial-of-service strategies, resource-exhaustion tricks, environment inspection outside legal observations, or hidden-seed recovery. Rules-dependent opportunities require an explicit release decision supported by current official rules.

## 6. Routing, Composition and Certification

### Keep binary routing first

For a small portfolio, retain the existing parent/specialist routing path and exact fallback. Search should produce bodies with stable legal initiation regions; it must not outsource every generalization failure to a more complicated router.

Certify the **whole materialized composition**, not only each component. Effects can change which states are visited, which options activate, how an active option terminates, and what the parent does after fallback. A rule that never overlaps another at initiation can still interact economically with it later.

Track:
- Activation, overlap, shadowing and reactivation by specialist and legal state family.
- Pairwise and selected higher-order intervention effects.
- Node count, duplicated gates, runtime work, memory and worst-case action size.
- Marginal new protected wins per added node and per certification cost.
- A permanent protected-game manifest relative to the certified ancestor chain, plus the immediate parent.

Protected games cannot disappear merely because a new parent, lane taxonomy or archive eviction makes them inconvenient. A finite protected corpus provides an empirical contract; it is not universal non-regression over all possible games.

### Introduce global composition only on evidence

Trigger the global route phase when at least 32 useful eligible specialists exist and either measured activation overlap exceeds 10% of active decisions, candidate rejection is substantially conflict-driven, or compiled runtime/node usage approaches 70% of the confirmed limit. These are proposed engineering triggers, not new safety thresholds. A measured smaller-portfolio interaction can trigger it earlier.

Use a conflict graph to schedule checks, not as a complete model of strategic interaction. Synthesize a bounded ordered set of existing gates/options with:
- Hard legal-feature, fallback, size and known protected-counterexample constraints.
- Soft goals for independent useful coverage, margin robustness and compactness.
- Deterministic priority/order variables.
- No simultaneous override of the same action family unless explicitly supported and separately certified.

Weighted MaxSAT or CP-SAT is appropriate for this finite selection/ordering problem; native continuation checks validate its behavioral assumptions. Use one backend, a pinned version, one worker, fixed deterministic work limits and canonical tie-breaking. UNKNOWN or timeout is not a proof. [PySAT RC2 implementation](https://pysathq.github.io/docs/html/api/examples/rc2.html), [OR-Tools CP-SAT documentation](https://developers.google.com/optimization/cp/cp_solver)

After a composed regression, minimize the causal set of selected options/order decisions and add a scoped no-good. If a minimal cause is not established, forbid only that full assignment under the relevant context; do not invent a pairwise incompatibility. Higher-order clauses are allowed. Gains on a static state-by-specialist table are not additive full-game payoffs.

Compile shared pure predicates into a DAG/decision diagram only if it reduces the real expanded BKG4 cost. Variable ordering can make decision diagrams large; acyclic pure-predicate sharing must not change stateful option execution or fallback. [Bryant, Graph-Based Algorithms for Boolean Function Manipulation, 1986](https://www.cs.cmu.edu/~bryant/pubdir/ieeetc86.pdf)

### Two different CEGIS loops, two firewalls

- **Discovery CEGIS:** only BUILD states and internal generated witnesses. It changes specialist bodies and provisional initiation.
- **Certification CEGIS:** the existing downstream gate-hardening authority, after TRANSFER and VALIDATION. Its counterexamples may modify only the candidate under certification according to existing rules; they never train the discovery grammar or body proposer.

Any certification edit invalidates affected earlier checks. Re-run the complete modified route on the required BUILD/DEV screening and TRANSFER/VALIDATION/protected checks before SEALED. Rechecks are safety tests of a frozen edit, not permission for a DEV-guided body search.

### Panel lifecycle for autonomous improvement

Frozen Phase-0 panels remain unchanged for historical comparison. New production epochs have disjoint seed namespaces, case identities, and trajectory roots across BUILD, INTERNAL_DEV, TRANSFER, VALIDATION, certification CEGIS and SEALED. Never move a used downstream case into BUILD.

Before searching, freeze epoch manifests, feature schema, shortlist quota, stopping rule and evaluation budgets. Discovery reads BUILD only. A separate verifier can read protected/downstream evidence. The semantic archive's historical ability to store pre-SEALED evidence must not imply permission for the proposal engine to query it; compartmentalize it or enforce a BUILD-only query surface.

Repeated querying creates adaptive selection pressure even if the search never sees raw labels. After an epoch's frozen shortlist is screened, retire that DEV panel from independent-estimation duty; retain it only as a verifier-side regression suite. Future epochs use fresh disjoint panels. Report how many candidates and epochs were tested. The 24-case panels are useful gates, not precise estimates of rare regression risk. [Blum and Hardt, The Ladder, ICML 2015](https://proceedings.mlr.press/v37/blum15.html)

SEALED is checked only after the whole composition is locked. The search receives no examples, descriptors, margins, or per-case outcomes. A designated verifier may return the promotion verdict and a certificate digest. Rejected compositions do not get repeatedly tuned against the same SEALED panel. New attempts require a new preregistered epoch and fresh sealing; retain prior protected cases privately for regression checks.

### Non-negotiable release contract

Every promotion, including an early experimental one, must satisfy:
1. Deterministic source/config/engine/panel/program/compiler identities and reproducible asset materialization.
2. Native CPU/CUDA agreement for state transitions, observation extraction, action selection, option execution and compiled runtime. Semantic mismatches = 0.
3. Exact PARENT fallback on inactive, terminated, aborted and fault paths; no corrupted persistent state or discarded unrelated action families.
4. Full required downstream checks and final protected W->L = 0 for the entire composition. Report W/D/L transitions and margin tails as well; never quietly relax existing tie or harm rules.
5. SEALED authorization at the designated stage, with no searchable SEALED contents.
6. A fixed **64-game official Python/Kaggle canary**, expanded to **128 games** for broader semantic changes or a new release engine. Both seats and representative internal policy families; compare native/official outcomes and trace diagnostics. Any unexplained discrepancy blocks promotion.
7. Test the actual packaged submission, dependency availability, legal action schema, runtime/memory/size limits and deterministic cold starts. Pin the official environment/config used and verify live competition requirements.
8. Hash and inspect the final package; preserve the previous champion for rollback through reproducible manifests. Only assets/champion is authoritative.
9. Atomic promotion after all gates, not before. Local commit and clean worktree; push and live submission require separate explicit authorization.

The canary is not a statistically strong proof of generalization and not a replacement for native search or protected certification. A canary mismatch is not repaired by discarding the failing seed.

## 7. Measurement and Compute Strategy

### Count information, not simulator calls alone

Record all of the following separately:
- Raw proposals, canonical programs, execution phenotypes and strategic-effect phenotypes.
- Target-family activations and changed legal actions.
- Distinct reachable roots, seeds, seats and internal member support.
- Full games, snapshot-prefix generation, suffix rollouts, short continuations and exact simulated turns.
- GPU time, wall time, CPU synthesis/checking time, compile time, peak host/device memory and storage.
- BUILD-eligible useful phenotypes, cross-regime useful phenotypes, DEV survivors, downstream survivors and certified composed gains.
- Exact-repeat rate, empirical-effect duplicate rate, repeated-failure-class rate and sentinel false-negative audits.
- Full-game generalization retention of fork-selected candidates, benefit-per-node, and cost per safe improvement.

Define **FGE = total simulated game turns / 720**, retaining actual transition counts and the pinned engine's terminal convention. Count both policies' shared game transition once, not twice. A 96-turn fork is not a full game and not an independent generalization sample. Count PARENT reference continuations, prefix regeneration, failed checks and any uncached reruns. Also provide an all-in ledger including downstream and CPU/official work.

Compare the frozen 9,600-game baselines using both raw full-game counts and the new 9,600-FGE budget. FGE normalizes simulation work, not runtime cost: short batches, restoration, telemetry and compilation can change effective throughput. Include a time-matched view using measurements, without allowing wall time to affect deterministic search order.

A **cross-regime useful phenotype** must satisfy the repository's existing BUILD eligibility rules and additionally demonstrate beneficial full-game W/L support on at least two independent seed roots, both seats, and more than one internal member when the regime has such coverage. It need not win in unrelated lanes. Any new proposed gate is conjunctive with existing requirements, never a weakening. Margin-only progress is labeled a stepping stone, not a counted useful specialist.

Two specialists count as strategically distinct only if their action-family/phase effects or independently supported activation regimes differ meaningfully. Hash differences, neighboring thresholds and minute terminal deltas are not sufficient.

### Budget accounting and statistical interpretation

A fixed-seed experiment is reproducible, not automatically statistically independent. Cluster uncertainty calculations by independent root seed, with seats and checkpoints grouped under the root. Report per-context results and paired W/D/L counts; do not treat thousands of forks as thousands of independent games.

There is no valid infinite speedup ratio when a comparator yields zero survivors. Report absolute differences, repeated contexts and an upper bound on cost per observed success. Historical claims of one or two specialists per very large campaign remain coarse context, not a calibrated control distribution.

Do not promote by repeatedly peeking until a significance threshold happens to pass. Lock candidate order, endpoints and budgets. Small-panel gates enforce observed safety; broader unseen-risk estimates need new independent cases and explicit uncertainty.

### Hardware and storage

A read-only hardware query during this research reported **NVIDIA GeForce RTX 3050 Laptop GPU, 4,096 MiB**. No current games/second benchmark was run. Initial designs must fit this device; larger hardware is optional after architecture evidence.

Illustrative engineering allowances, not measured performance forecasts:

| Component | Evaluation allowance per bounded experiment | CUDA suitability | CPU overhead | Memory/storage allowance | Build complexity / expected leverage |
| --- | --- | --- | --- | --- | --- |
| Snapshot fidelity and repair pilot | <=3,200 FGE, roughly 1,500 candidate-state suffixes plus controls | High once state restore is native | Serialization, hashing and small reference checks | Start 24 snapshots; later 4k snapshots at a provisional 8-64 KiB each is 32-256 MiB | 3-5 engineering days; potentially very high |
| CCPR oracle | 9,600 FGE/context; 1k-20k concrete proposals depending on continuation cost | High for batches; not symbolic enumeration | Bitsets, queue and canonicalization; aim below 25% of end-to-end time | 10k-100k compact records; size measured before scaling | 5-8 days for bounded version; very high if generalization survives |
| Challenge frontier and active memory | 2k-10k fresh full games/epoch plus budgeted forks | High | Clustering, set cover, scoped witness updates | 1k-10k retained states; 0.1-2 GiB provisional host budget | 3-5 days; high sustained leverage |
| Mechanics testing | 10k-100k short transition tests; <=2k FGE for opportunity validation | High for transition corpus | Small C++/official reference and minimization | 100-1k minimized witnesses, usually far smaller than campaign logs | 2-4 days; potentially very high but uncertain |
| Library induction | CPU enumeration over <=100-500 programs; two matched 4,800-FGE searches | High for validation only | Bounded anti-unification and ranking | <=32 proposed components initially; <=256 MiB working cap | 2-4 days; high only after repeated successes |
| Option-pair planning | 10k-50k suffix evaluations within <=9,600 FGE/context | High with grouped horizons | Beam management and preconditions | 16-64 frontier states/root, bounded cache | 3-6 days; high for enabling combinations |
| Global route synthesis | Initially <=5k-20k FGE for candidate compositions | High for checker | CPU solver plus bitsets; explicit work cap | Sparse conflicts; do not allocate every specialist x state x order combination | 4-7 days, conditional; low value before real portfolio pressure |
| Certification and release | Initial 0.9k-5k native games/candidate bundle; grows with protected corpus; 64/128 official games/release | High natively; official canary CPU only | Reference checks, packaging and audit | Compact manifests/certificates; full traces regenerated on demand | 2-4 days automation; mandatory safety leverage |

These allowances overlap because subsystems share evaluation infrastructure; they are not promises that all development fits one month. CPU time estimates must be calibrated by the pilot rather than invented. A solver exceeding its count budget returns no solution; a maintenance wall-clock watchdog may abort an experiment but may not select a different winner.

Keep a conservative device working-set cap, initially around 1 GiB for fork state/output pools, leaving room for the existing evaluator and driver. Measure actual state size, spills and launch occupancy; adjust batch size deterministically from a pinned configuration, not live timing feedback.

The existing 96-byte result ABI can remain the normal summary path. Emit detailed traces only for sampled audits or minimized witnesses. Preserve canonical JSON as an interchange format. If 100,000-record whole-file rewriting becomes material, introduce an append-only compact log plus deterministic index/checkpoints behind the existing archive API; do not prematurely build a database service.

### Scaling gate

No million-game campaign before all of the following:
- Primary or fallback has passed its bounded comparison and fresh-context replication.
- At least five strategically distinct safe improvements have been obtained over at least three production epochs within a preregistered <=100,000 all-in native-FGE production window, with earlier research/control costs and official canary costs also reported separately.
- Improvements cover at least four independently supported challenge clusters, not five threshold variants.
- Late-epoch useful yield is at least half the early post-warm-up yield and remaining frontier novelty is measurable.
- No semantic mismatch, final protected W->L, firewall violation, or unexplained official discrepancy.
- Engineering time and archive/runtime growth remain bounded.

Even then, authorize 25k-100k-FGE tranches with saturation checks, not a blind million-game run. Failure to meet this gate means improve the search or stop, not rename margin deltas as safe specialists.

## 8. Implementation-Ready Phase Sequence

Phases below replace the unimplemented ASPE sequence. They are gates, not a shopping list. Each phase ends with a compact evidence manifest, a pass/fail/defer decision, focused tests and a local commit. Do not implement a later research subsystem to conceal failure of an earlier premise.

The release contract in Section 6 applies from the **first** proposed champion promotion. Phase 9 automates/finalizes release readiness; it does not postpone canaries or safety until then. Every phase uses the existing package import surface and current repository hygiene rules.

### Phase 1: Exact Fork-and-Repair Value Experiment

**Objective.** Determine whether reachable-state continuations expose reusable repair opportunities at lower information cost than repeating whole prefixes. This is the single next implementation phase.

**Implementation scope.**
- Add the smallest native snapshot/restore and continuation entrypoint, reusing the existing state layout and SpecialistProgram executor. Establish the actual terminal convention and pinned official/native engine identity.
- Capture complete decision-boundary state and both controllers' context. Restore on CPU and CUDA and compare to uninterrupted execution.
- Build a deterministic intervention matrix using existing v1 bodies and tiny legal parameter variants. No adaptive synthesizer yet.
- Include action-dependent randomness, day boundaries, active-option state and fallback in focused fidelity tests.

**Explicit non-goals.** No new VM, general SMT solver, automatic curriculum, full evolution implementation, global router, permanent schema expansion or large campaign. No DEV/SEALED access.

**Experiment.**
1. Use 12 fresh internal root games: three BUILD seeds x both seats x two internal members. Select two checkpoint types per game, balancing earlier investment and later recovery/transition decisions.
2. Evaluate up to 64 predeclared compact v1 bodies across these 24 checkpoints: up to 1,536 terminal suffix checks plus matched uninterrupted-prefix checks of the same injected interventions.
3. Include identity/PARENT, no-activation, termination and deliberately infeasible-action controls. The uninterrupted intervention control is a harness experiment, not a legal submission gate.
4. Inspect 24/96-turn diagnostics against the terminal matrix. Audit potential early-screen false negatives, especially delayed investments.
5. For at most 12 informative bodies, use the existing legal gate machinery to test ordinary execution on full BUILD. Do not count forced checkpoint success as a deployed capability.

**Metrics.** Full-state/action/outcome equality; controller-state and restart equality; bytes/snapshot; restore overhead; actual turns and GPU time; candidate activation; distinct beneficial effects; terminal-value retention across roots; retention under ordinary legal initiation.

**Pass gate.**
- Zero parity/fallback/restart mismatches, including modified-action continuations rather than only PARENT.
- At least two distinct body effects show positive paired terminal margin across at least two seeds and both seats within the relevant family.
- At least one retains positive net paired terminal margin when run with a legal gate on full BUILD; label margin-only evidence explicitly.
- Shared-prefix/selective evaluation can retain at least 90% of the matrix's beneficial candidates while avoiding at least 30% of simulated turns versus matched uninterrupted-prefix checks, counting reference/prefix costs. These are proposed minimum leverage thresholds, not predicted results.

**Failure branch.**
- Fidelity failure: fix within the phase; no strategic claim survives a semantic mismatch.
- No beneficial matrix entries: the efficiency/retention result is inconclusive, not 100% recall. Permit one preregistered alternate checkpoint/motif matrix within the total budget; then fail the strategic thesis if still empty.
- Benefits exist but legal full-game initiation loses them: record an activation/representation witness and stop before building a bigger oracle.
- If fidelity works but leverage fails, do not make snapshots mandatory for search. Activate the fallback experiment with ordinary full games; retain snapshots only for debugging or narrow late-game work.

**Compute budget.** <=3,200 native FGE including prefixes, suffixes, controls and up to 576 full BUILD games for the shortlist; CPU reference checks bounded to a representative subset plus boundary fixtures. At most a few dozen short official transition fixtures to resolve engine-version uncertainty, not a large Python campaign.

**Deliverable.** Reproducible snapshot contract, round-trip/parity fixtures, intervention matrix summary, exact cost ledger, legal-gate retention report and an explicit pass/fail. Commit compact evidence under workspace/ without campaign traces.

### Phase 2: Minimal CCPR and Equal-Budget Falsification

**Objective.** Prove that counterexample-guided specialist generation, not merely prefix caching, produces independent reusable capabilities.

**Implementation scope.** Bounded best-first typed holes, BUILD-only witness tables, predicate bitsets, exact evaluation caches, deterministic queue updates and complete-game promotion. Reuse telemetry/sentinel/archive interfaces. Initially keep the v1 grammar fixed.

**Explicit non-goals.** No automatic library induction, global route solving, neural proposer, large populations, arbitrary loops or hand-tuning from DEV. Do not implement DSQD unless this phase fails.

**Experiment.**
- Run CCPR with a 9,600-FGE discovery budget on the frozen context, retaining its identities unchanged. Freeze the same parent across comparison and replication; do not let an early promotion alter later arms.
- Run two matched 9,600-FGE controls on that context: the same grammar/search ordering using full-game evaluation without targeted fork credit, and CCPR with active counterexample/count reuse disabled. Match initial grammar, candidates where applicable, roots and shortlist quotas; report differences in reachable candidates rather than pretending all adaptive arms see identical sequences.
- Retain the historical RLRMB/Direct results as controls only if the engine/config are identical. If a version mismatch is found, label them historical and rerun a bounded same-version comparator before claiming superiority.
- Lock the algorithm after this comparison, then replicate CCPR on two fresh independently seeded contexts, each 9,600 FGE. Generate their disjoint downstream panels before search.
- Each arm exposes at most three preregistered, BUILD-selected candidates to its independent DEV screen. The existing 864-game conditional downstream allowance is retained where sufficient; exhaustion is reported as censored, never silently charged to another arm.
- A final locked composition may undergo the full release contract using a separately declared certification allowance.

**Metrics.** Useful cross-regime phenotypes and DEV survivors per 10k FGE; exact-repeat/failure reuse; effect diversity; fork-to-full retention; full-game counts; GPU/CPU/wall cost; protected transitions; safe marginal gains.

**Pass gate.**
- At least two distinct cross-regime useful phenotypes and at least one existing-gate DEV survivor in each of the two fresh replication contexts. The frozen context remains a comparison/diagnostic context, not an untouched generalization sample.
- At least one final safe composition survives the complete downstream flow and official canary.
- Positive absolute useful/DEV yield versus the zero-yield historical baselines, plus evidence that targeted credit and active memory are worth their overhead. Where a control has positive yield, target at least 2x useful yield/FGE or half the cost to its first independent survivor. Where it has zero, report counts and replication, not an infinite ratio.
- All existing safety thresholds, semantic equality and firewalls pass unchanged.

**Failure branch.**
- No generalizable body despite successful Phase 1: reject CCPR as the primary oracle and run the single DSQD fallback below.
- BUILD succeeds but DEV collapses: reject the generalization thesis; inspect BUILD-only alias/reachability/activation evidence, not DEV examples. Allow one bounded representation diagnosis, not repeated DEV-driven tuning.
- A concrete unsupported v1 capability is proven: stop with that witness. A minimal extension requires a new version, a new preregistered experiment and equivalent comparator expressivity. This is an explicit strategic revision, not an automatic pass into later phases.
- DEV survives but legal routing fails: preserve the body as SEARCH_MEMORY_ONLY and diagnose routability before scaling. Do not count it as a safe specialist.

**Compute budget.** 48,000 discovery FGE for three primary contexts and two controls, plus bounded conditional downstream and separately reported release work. Calibrate expected GPU hours from Phase 1. No million-game experiment is justified.

**Deliverable.** Minimal oracle, deterministic restart/ledger tests, matched ablations, three-context report, explicit architecture verdict and any fully certified compact specialist.

### Conditional Fallback: DSQD, Only After Primary Falsification

**Objective.** Test whether counterexample decomposition is the wrong inductive bias while the representation and evaluator remain useful.

**Implementation scope.** Deterministic typed neighborhood enumeration, small niche-local populations, age-based stepping-stone retention and stable migration. Use the existing semantic archive. Variation uses a fixed ordered schedule; any seeded permutation is pinned in the manifest.

**Explicit non-goals.** No return to RLRMB as primary, no learned proposer, no simultaneous permanent CCPR/DSQD ensemble and no larger budget to hide poorer information efficiency.

**Experiment.** Same fixed grammar and eligibility rules, 9,600 FGE on one primary-failed context; continue to two fresh contexts only after useful cross-regime signal. Use independently validated snapshots if they helped, otherwise ordinary full games. Compare the same useful/DEV/certified metrics.

**Metrics.** Same Phase-2 ledger, plus niche occupancy, lineage concentration and novelty-to-generalization conversion.

**Pass gate.** The same Phase-2 cross-regime/DEV replication and full certification requirements. A mere increase in novel phenotypes is a failure.

**Failure branch.** If both architectures fail, halt search scaling and issue a bounded representation/strategic-abstraction research task with concrete witnesses. Do not create a third speculative production oracle or claim the roadmap's empirical thesis succeeded.

**Compute budget.** 9,600 FGE initially; at most 28,800 discovery FGE if the first run justifies replication, plus declared downstream work.

**Deliverable.** One alternative-oracle verdict. If DSQD wins, it replaces the CCPR proposal generator in subsequent phases; the challenge, evidence and safety contracts remain.

### Phase 3: Reachable Challenge Frontier and Active Memory

**Objective.** Escape the fixed eight-lane/small-panel basin and make search improve as evidence accumulates.

**Implementation scope.** Reachable state/prefix archive, BUILD-only data access, deterministic coverage/regret scheduling, body-specific failure constraints, abstraction split witnesses, corpus minimization and capability-preserving internal league refresh.

**Explicit non-goals.** No public-opponent construction, arbitrary state injection as strategic evidence, high-dimensional dense MAP grid, full Nash solver or unbounded archive service.

**Experiment.** Compare an expanding frontier against the same oracle on fixed lanes for four sequential fixed-parent epochs, with equal 2,400-FGE discovery allocations per epoch and a separate capped fresh-root generation ledger. Then perform one champion-refresh epoch to test cache invalidation and new-state discovery. Freeze DEV shortlists and independent panels per epoch.

**Metrics.** New reachable transition/effect classes, useful phenotypes per epoch, repeated-failure reduction, cross-root support, novel-challenge-to-safe-gain conversion, stale-cache rejection and archive replay determinism.

**Pass gate.**
- At least three new supported challenge clusters beyond the initial taxonomy, defined from BUILD mechanics/response witnesses before independent screening, not from arbitrary names or downstream outcomes.
- At least two cross-regime useful phenotypes arising from those clusters and at least one fully certified new capability.
- At least 25% lower repeated-failure rate at matched novelty coverage than passive-memory control.
- Later-epoch useful yield is at least half the first two post-warm-up epochs' yield; if the denominator is zero, this metric cannot pass.
- No panel/root leakage or loss of protected historical capability.

**Failure branch.** If coverage rises without useful responses, change challenge selection toward recoverable regret and causal predecessors, not archive size. If abstraction splits only increase names, revert that split policy, not evidence. If memory harms exploration, increase its fixed exploration floor and rerun one bounded BUILD-only ablation.

**Compute budget.** Cap at 25k FGE: 19,200 for the eight discovery epochs, 2,400 for the parent-refresh check, and at most 3,400 for fresh roots. Downstream is separately capped. Start with 1k-4k archive states and a modest host-memory cap.

**Deliverable.** Versioned challenge schema, prefix recipes, scheduler/restart tests, active-memory ablation and sustained-yield report. The original eight lanes remain readable reporting fields.

### Phase 4: Mechanics Differential Fuzzing and Opportunity Validation

**Objective.** Discover legal transition opportunities and catch clone-specific false wins before they poison search memory.

**Implementation scope.** Structured native transition harness, mechanics coverage tags, boundary/permutation schedules, reachability-aware minimization, correctness/opportunity queues and bounded official adjudication.

**Explicit non-goals.** No infrastructure attacks, illegal-state strategy promotion, hidden-seed exploitation, production Python fuzz campaigns or requirement to find a profitable bug.

**Experiment.** Test 10k-100k short state/action cases across at least six mechanics families. Include deliberately perturbed test doubles to check mismatch detection without changing production semantics. Validate up to ten minimized opportunity hypotheses on related reachable roots and run only promising legal options through the existing oracle.

**Metrics.** Coverage of preregistered boundaries/interactions; detected fixture faults; equal native transitions; minimized witness size; reachable opportunities; terminal and full-game generalization yield.

**Pass gate.**
- All seeded mismatch classes detected; real production mismatches resolved to zero before resuming dependent search.
- At least 90% of reachable preregistered boundary categories exercised, with unreachability or missing coverage documented.
- Every claimed opportunity has a replayable legal witness and agreement with bounded official checks.
- Strategic funding expands only after one cross-root beneficial option; absence of profit does not fail the correctness harness but limits it to a small maintenance quota.

**Failure branch.** A mismatch quarantines the affected engine/version results. A fabricated-state-only opportunity is retired from strategy search. No profitable opportunities after the budget: keep regression coverage and reduce exploration allocation rather than forcing an exploit narrative.

**Compute budget.** <=2k native FGE for long opportunity validation beyond the short test corpus; CPU reference checks sampled by coverage, plus <=64 short official fixtures. These are not 64 full canary games.

**Deliverable.** Compact minimized mechanics corpus, source/config pins, parity regression tests, a rules/provenance checklist and a factual opportunity ledger.

### Phase 5: Evidence-Driven Symbolic Vocabulary

**Objective.** Reduce repeated rediscovery of useful multi-action components without hard-coding a permanent manual strategy taxonomy.

**Implementation scope.** Typed fragment extraction, anti-unification, compression ranking, bounded parameterization, new-task reuse tests and ordinary BKG4 expansion.

**Explicit non-goals.** No neural library learner, DreamCoder-style neural recognition stack, recursive runtime calls, automatic broad DSL expansion or treating compression as strategic gain.

**Experiment.** Start only after at least ten useful BUILD programs from three independent lineages/families exist. Mine at most 32 components from at most 100-500 programs. Compare two 4,800-FGE new-BUILD-task searches with and without the library under identical primitive access and independent final screening.

**Metrics.** Expanded program size, compile/parity agreement, tasks solved, useful phenotypes/FGE, reuse across lineages, induction CPU cost and library-disabled exploration retention.

**Pass gate.** At least two components reused beneficially outside their induction families; at least 25% lower cost to matched useful-phenotype yield or two additional useful phenotypes when the control has none; no material increase in expanded runtime cost or loss of generalization. Confirm one library-enabled candidate through full certification before making the library default.

**Failure branch.** Keep the primitive grammar and useful archive programs. Remove unproductive abstractions from proposal preference without deleting provenance. If the prerequisite corpus does not exist, mark the phase DEFERRED; do not mine a library of failures merely to complete a milestone.

**Compute budget.** <=9,600 native FGE for comparison, plus declared certification. CPU-only induction with deterministic node/work and memory caps.

**Deliverable.** Small versioned macro vocabulary, expansion tests, provenance/assumption cards and new-task ablation.

### Phase 6: Long-Horizon Option Composition and Minimal Representation Growth

**Objective.** Discover jointly profitable economies that one-option/PARENT continuation misses.

**Implementation scope.** Four-way option interaction tests, event-level beam search, exact state transpositions, enabling-state retention and only witness-justified legal feature/control extensions.

**Explicit non-goals.** No full 720-turn primitive tree, learned value function, general-purpose HTN environment rewrite or approximate economic solver as an acceptance authority.

**Experiment.** Preselect up to 20 BUILD families showing likely enabling dependencies or isolated-option failure. Compare standalone options against ordered pairs and shallow triples under equal budgets. Include at least one earlier investment and one terminal conversion family. Use a fixed 16-64 beam and depth 2-3; compile the final combinations before generalization tests.

**Metrics.** Positive interaction beyond standalone effects, retained delayed-payoff candidates, legal action feasibility, full-game gain, expanded node count, observation-alias resolution and search cost.

**Pass gate.** At least two repeatable cross-root combinations whose component-only controls do not achieve the same gain; at least one fully certified improvement; exact compilation/fallback parity. A representation extension additionally needs a concrete previously unrepresentable legal witness and matched old/new grammar comparison.

**Failure branch.** If the parent already realizes the benefits, do not add a planner. If only a relaxed resource model succeeds, reject the plan. If compiled combinations exceed the node/runtime cap, reduce the option vocabulary or target one smaller witness, not a new runtime.

**Compute budget.** <=9,600 native FGE/context; allow one fresh confirmation context only after signal. CPU queue work bounded; exact fork state memory scales with beam width, not all game histories.

**Deliverable.** Interaction evidence, a bounded deterministic option planner if justified, and minimal versioned representation changes only when proven necessary.

### Phase 7: Global Portfolio Composition, Conditional on Pressure

**Objective.** Preserve safe marginal gains when sequential binary additions suffer from overlap, order dependence or size growth.

**Implementation scope.** Activation/conflict telemetry, finite option/order selection, conservative conflict clauses, complete-composition checker and optional pure-predicate sharing.

**Explicit non-goals.** No routing by opponent identity, unconstrained hierarchy, simultaneous action blending, exhaustive all-subset verification or replacing empirical gates with solver satisfiability.

**Experiment.** Trigger using Section 6's measured pressure. On the same frozen candidate set, compare current sequential binary composition with bounded global selection/order. Test adversarially generated overlapping activation sequences, triples where pairwise checks pass, parent refresh and materialized execution.

**Metrics.** Safe marginal wins, protected regressions, shadowing, higher-order counterexamples, expanded nodes, worst-case runtime and certification cost.

**Pass gate.** Preserve all prior protected wins; gain at least one otherwise blocked useful capability or reduce compiled node/runtime cost by at least 20% at identical behavior; zero semantic mismatches and final W->L; whole composition passes independent downstream and official canary.

**Failure branch.** Keep sequential binary routing and reject the conflicting addition. Solver timeout or larger decision diagrams does not justify loosening gates. If trigger conditions never occur, record DEFERRED and keep the simpler router.

**Compute budget.** Initially 5k-20k native FGE; CPU solver bounded by deterministic work and one worker. Grow only if the marginal benefit exceeds verification cost.

**Deliverable.** Finite composition specification, conflict witnesses, deterministic solver/compile tests and an explicit comparison against unchanged binary routing.

### Phase 8: Autonomous Champion Improvement With Exposure Control

**Objective.** Turn the validated components into a repeatable improvement loop without corrupting protected evaluation or manufacturing progress.

**Implementation scope.** Epoch manifests, BUILD-only proposer permissions, deterministic scheduling and restarts, atomic certificate-gated promotion, cache invalidation, panel rotation, protected-case retention, budget/saturation stops and rollback manifests.

**Explicit non-goals.** No auto-push, auto-submit, continuous SEALED peeking, endless campaigns, hidden manual candidate selection or changing thresholds during a run.

**Experiment.** Execute at least three bounded production epochs. Exercise one rejected candidate, one interrupted/restarted run and one parent refresh. Each epoch freezes its shortlist and independent panels before search; release checks apply to every promotion.

**Metrics.** Certified distinct gains per all-in FGE and engineering week; certificate identity under restart; exposure counts; frontier growth; protected retention; archive/runtime growth; phase-specific useful yield.

**Pass gate.** At least three distinct fully certified improvements across the epochs, with new failure families still being discovered; identical selected candidates and materialized hashes after restart; zero firewall violations and regressions. The stronger five-improvement/100k-FGE gate in Section 7 is additionally required before large scaling.

**Failure branch.** A firewall or certificate failure disables promotion immediately. Saturation invokes Section 9's fixed response ladder, not unlimited runtime. No independent progress after the bounded responses: stop the epoch and retain the last certified champion.

**Compute budget.** 10k-25k native discovery FGE/epoch initially; separate bounded certification and 64/128 official games per promoted release. The <=100k scaling criterion is measured over a preregistered production window; report all earlier research/control costs separately.

**Deliverable.** Autonomous local runner, auditable promotion journal, restart/rejection/firewall tests, three-epoch evidence and a clean reproducible repository.

### Phase 9: First-Place-Ready Release and Adversarial Audit

**Objective.** Make the strongest certified portfolio deployable and robust to internal blind spots, translation errors and operational mistakes.

**Implementation scope.** Fresh internal challenger audit, optional independent external HOLDOUT evaluation, stress testing of the actual package, deterministic materialization and final release manifest.

**Explicit non-goals.** No claim that internal dominance proves first place, no external identity construction, no large official-engine search and no live submission without approval.

**Experiment.**
- Freeze the final portfolio; challenge it with fresh internal synthesized exploiters, retained payoff-diverse capabilities, rare mechanics families and cross-regime transitions.
- Evaluate available external agents only as HOLDOUT. Do not route by their identity or use their action traces to construct a repair.
- Complete native parity, all protected/downstream checks, SEALED authorization, 64/128 official canary and packaged cold-start/runtime tests.
- Verify current rules, deadlines, team submission-selection policy, package format and resource limits from official competition material before submission authorization.

**Metrics.** Fresh-root W/D/L, worst-family and tail margins, internal best-response gaps, HOLDOUT outcomes with uncertainty, protected W->L, parity, runtime/size/memory and artifact identity.

**Pass gate.** All hard release conditions pass; no unresolved high-impact internal counterexample within the declared audit budget; exact final package hash matches the certified asset. External coverage gaps and statistical uncertainty are disclosed rather than converted into a universal guarantee.

**Failure branch.** Any correctness/protected failure blocks release. A strategic HOLDOUT failure can block selection but never becomes a BUILD replay; a new search epoch must generate independent internal evidence. Rules-dependent exploitation stays disabled until cleared. Ship the prior certified champion if time prevents completing another full funnel.

**Compute budget.** Initially 5k-20k fresh native audit games, increased only through the scaling gate; exactly declared 64/128-game official canary; bounded CPU package and trace checks.

**Deliverable.** Final deterministic submission package, engine/config/policy hashes, private verifier certificate, compact public-safe release evidence, local commit and clean worktree. No push or submission by the planning agent.

## 9. Concrete Anti-Saturation Contract

Monitor fixed windows of 2,400 discovery FGE, grouped into epochs. Thresholds below are initial operational triggers to preregister and later revise only between experiments.

| Indicator | Trigger | Deterministic response |
| --- | --- | --- |
| Repeated syntax/context | >20% exact repeats in two windows | Repair caching/scheduler first; do not spend GPU to reconfirm known rows |
| Phenotypic repetition | >80% noninformative effect duplicates in two windows | Move quota to least-served challenge cells and distant typed fragments |
| Failure repetition | Same avoidable failure class >50% of rejected checks | Add a scoped witness constraint or repair prerequisite hole; audit whether counts are actually used |
| Novelty without capability | New coverage grows, no cross-regime useful phenotype for three windows | Prioritize recoverable regret and causal predecessors; stop reward for pure coverage volume |
| Coarse state aliasing | Conflicting body signs in a cell with >=8 independent roots | Test a legal-feature split; preserve hidden uncertainty rather than fingerprinting it |
| Delayed-payoff attrition | Terminal-positive audit candidates repeatedly rejected early | Disable that early screen; reserve terminal checks for enabling investments |
| Library lock-in | >75% new proposals descend from one fragment family | Restore primitive/least-cost construction quota and retire stale rule preference |
| Champion-local trap | Internal challengers become behaviorally redundant | Pair payoff-diverse internal capabilities and vary reachable prefixes, not named external agents |
| Portfolio overhead | Certification dominates useful search or additions are shadowed | Freeze additions; minimize or test Phase 7 if its trigger is met |
| Sustained zero safe yield | No new safe gain across three complete production epochs | Stop scaling; one bounded response cycle, then architecture/representation review |

Apply responses in this order: fix duplicates -> move challenge allocation -> refine a justified state abstraction -> retrieve a distant reusable component -> search an earlier causal prefix -> test an enabling option combination -> stop if useful yield does not recover. This order prevents every plateau from triggering a larger grammar.

Keep at least one witness for rare mechanics and protected capabilities even when they are currently unprofitable. Keep a bounded unsolved frontier, but avoid spending the entire budget on irreversible losses with no known path to improvement. A measured local-regret witness is a lower bound on available improvement, not a proof that unsolved states lack a response.

No finite archive or search schedule guarantees indefinite novelty. The objective is measurable expansion of reachable strategic capability with bounded regression and engineering cost. A long-lived local optimum remains possible; the stopping rule is part of the architecture, not an embarrassment to conceal.

## 10. Infrastructure Disposition and Work to Stop

| Area | Verdict | Concrete treatment |
| --- | --- | --- |
| Frozen ASPE Phase-0 baseline | KEEP | Immutable historical comparison; new engine/config epochs get new identities |
| SpecialistProgram v1 | MODIFY | Keep execution contract and initial grammar; extend only for concrete legal representation witnesses |
| Phenotype telemetry | KEEP | Preserve ABI summaries; add sampled causal/transition diagnostics only as needed |
| Sentinel screening | MODIFY | Keep frozen sentinel for comparisons; add target-family probes and false-negative audits; no global equivalence inference |
| Semantic archive | MODIFY | Repurpose passive failure rows into BUILD-only active constraints, rule counts and versioned reuse; maintain SEARCH_MEMORY_ONLY authority |
| Typed semantic evolution | REPLACE | CCPR is primary; deterministic semantic evolution exists only as the failed-primary fallback |
| Islands / QD | MODIFY | Keep bounded local competition, archive diversity and least-served scheduling; defer independent populations unless fallback needs them |
| Learned proposer phases | DROP | Remove neural proposal datasets/training/online serving from the execution path |
| RLRMB | KEEP | Frozen diagnostic baseline; not primary discovery |
| Direct Bonsai | KEEP | Frozen diagnostic baseline; not primary discovery |
| Adversarial league | MODIFY | Preserve current capabilities; add reachable coverage/regret challenges and payoff-profile diversity |
| BUILD / INTERNAL_DEV architecture | KEEP | Preserve separation; strengthen epoch lifecycle and trajectory-root disjointness |
| Binary routing | KEEP | Proven default; global optimization only under measured conflict/size pressure |
| CEGIS | MODIFY | Retain certification role; add strictly separate BUILD-only discovery refinement |
| TRANSFER / VALIDATION / SEALED | KEEP | No weaker thresholds, no data feedback to body generation; whole composition certified |
| CUDA / C++ reference engines | KEEP | Extend state restore; preserve exact semantics and quarantine mismatches |
| Engine exploit discovery | MODIFY | Promote weak mechanics lane into first-class correctness/opportunity testing |
| Autonomous champion loop | MODIFY | Certificate-gated epochs with exposure accounting, restarts and stop rules |
| Global MaxSAT route selection | DEFER | Only after Phase-7 trigger; no extra router now |
| General equality-saturation engine | DEFER | Simple canonicalization first; require proved rewrites and measured benefit |
| Whole-game SMT / primitive deep MCTS | DROP | Not primary components; bounded predicate/endgame subproblems remain allowed |

Stop immediately:
- Unconditional implementation of old ASPE Phase 4 followed by scaling before a fork/value falsification.
- Learned-proposer ASPE Phases 6-7 and any neural/LLM inner-loop substitute.
- Counting syntax, trace hashes or occupied bins as useful specialist yield.
- Permanently rejecting a body because it did not activate on 16 sentinel cases.
- Treating effects equal on a finite panel as globally equivalent programs.
- Reusing DEV/TRANSFER/VALIDATION/SEALED evidence as proposal-training memory.
- Increasing routing complexity to compensate for wholly trajectory-bound bodies.
- Planning a million-game campaign before repeated safe capability-per-budget evidence.

None of the completed infrastructure is retired merely because the search hypothesis changed.

Using the completed-infrastructure vocabulary explicitly: frozen baseline = KEEP; SpecialistProgram v1 = MODIFY only on witnesses; phenotype telemetry = KEEP; sentinel = MODIFY; semantic archive = REPURPOSE into active, context-scoped search memory while preserving its existing interface and authority. RETIRE applies to none of Phases 0-3.

## 11. Direct Answers to the Sixteen Required Questions

1. **Top five bottlenecks?** Decision-state credit; active synthesis intelligence; reachable challenge/state abstraction; executable vocabulary and combinations; mechanics discovery and portfolio assurance.
2. **Missing versus weak?** State forking/causal experiments, active refinement, library induction and global assurance are missing as demonstrated capabilities. Proposal quality, eight-lane challenge generation and the mechanics lane are weak existing areas.
3. **Is v1 expressive enough?** For the next falsification, yes as a sensible initial hypothesis. For first place, unproven and likely incomplete in legal aggregate features and longer conditional coordination. Extend only on witnesses.
4. **Is typed semantic evolution the right next oracle?** Not as the default next step. First test exact forked repair value; then CCPR. DSQD is the one fallback if the primary fails.
5. **Should snapshots become central?** Yes, conditional on Phase 1. They improve the experimental unit and credit, not just speed. Full games remain indispensable for activation and generalization.
6. **Use CEGIS/SMT inside generation?** Yes to BUILD-only empirical CEGIS and bounded predicate/body-hole constraints. No to a monolithic symbolic game model or a claim of universal winning proofs.
7. **Escape eight lanes?** Maintain reachable state/transition coverage, regret witnesses, internal adversarial prefix edits and counterexample-driven legal partition refinement. Lanes become reporting tags.
8. **How do failures improve search?** Scoped negatives/no-goods, causal hole selection, exact caches, version-space reduction where justified, integer rule counts and a nonzero exploration quota.
9. **Best long-horizon credit?** Paired terminal option advantage against reactive internal policies, supplemented by removal and ordered-combination interventions. Milestones only explain and rank.
10. **Discover reusable primitives?** Typed anti-unification/compression from repeated internal successes, validated on new BUILD tasks and expanded into ordinary BKG4.
11. **Dedicated engine fuzzing?** Yes, two queues: correctness mismatches block search; equal legal opportunities must prove reachability and cross-root value.
12. **Large portfolio composition?** Keep binary routing until measured pressure; then bounded global selection/order with conflict clauses and complete-composition simulation, followed by unchanged certification.
13. **Exact anti-saturation mechanisms?** Fixed metric windows and a deterministic response ladder spanning challenge allocation, legal abstraction refinement, primitive/library exploration, earlier interventions and option combinations.
14. **What stops?** Old unproven evolution-first sequencing, neural proposers, premature million-game scaling, sample-equivalence pruning and premature global routing.
15. **Immediate next implementation?** Phase 1: the <=3,200-FGE exact fork-and-repair value experiment. Nothing else is authorized by its numbering.
16. **Full sequence?** Fork/value -> CCPR falsification (DSQD only on failure) -> challenge frontier/active memory -> mechanics -> symbolic vocabulary -> option combinations/minimal representation -> conditional global composition -> autonomous certified epochs -> final adversarial release.

## 12. Falsification, Execution Priority and First-Place End State

### The key falsification

The recommended primary is wrong for this repository if exact forked repairs repeatedly look useful locally but cannot produce legal, independent full-game survivors at the bounded Phase-2 budget, or if counterexample-driven refinement does no better than the matched simpler controls after accounting for CPU/engineering overhead.

That result must not be explained away by citing Go-Explore, increasing grammar size, or spending another million games. Run the one fallback. If both fail, fundamental representation/abstraction work is required before more discovery infrastructure.

### Engineering-week priority

The first executable milestone is deliberately only Phase 1. Next is the smallest end-to-end CCPR demonstration through existing safety and canary machinery. This can produce a useful champion before automatic library learning or global routing exists.

The longer roadmap is conditional. Phases 5 and 7 can remain explicitly deferred when their prerequisites are absent. Basic randomness/version checks and correctness fixtures belong in Phase 1 even though the full mechanics subsystem comes later. Release obligations are never deferred.

The allowances in Section 7 imply several engineering weeks, potentially longer for a single implementation agent. Verify the competition deadline before scheduling. If the available window is short, prioritize Phases 1-3, the bounded high-value part of Phase 4, and the mandatory release contract. Freeze feature work with enough time for full certification and a rollback release; do not rush a global solver or new runtime into the final days.

### First-place-ready is a measurable system, not a leaderboard promise

The end state is a deterministic champion whose capability expands through:
- New reachable internal challenges rather than repeated named-opponent targets.
- Compact reproducible responses with independently supported legal activation.
- Successful enabling combinations and reused components rather than endless primitive rediscovery.
- A mechanics harness that discovers opportunities and prevents clone-specific false victories.
- A growing protected capability set and complete-composition certificates.
- Independent adversarial and optional external evidence that can block a bad release without becoming construction data.
- A clean, resource-compliant, officially canaried deterministic package.

This is a plausible path to sustained strategic improvement because the loop can expand the **problems**, the **response vocabulary**, and the **combinations**, while failure changes future search. It is stronger than accumulating five specialists in a fixed basin. It still cannot guarantee first place against unknown future agents, finite evaluation, unobserved information, or competition changes.

## Appendix A. Claim-to-Source and Uncertainty Ledger

Sources were accessed on 2026-09-05. Publication dates below refer to the actual paper or identified commit, not search-engine crawl labels. The numbered sections contain the recommendations; this ledger records what the evidence does and does not support.

| Claim family | Primary evidence / date | Confidence and limitation |
| --- | --- | --- |
| Current repository architecture/results | [Pipeline context](FIRST_PLACE_PIPELINE_CONTEXT.md), verified HEAD b16d2a7 | High as repository-provided evidence; no campaign rerun |
| Exact current file-level engine mechanisms | [Official engine at 28b6d8a](https://raw.githubusercontent.com/Kaggle/kaggle-environments/28b6d8af3ce73926b3d0fda1410c1ddd8384ab8c/kaggle_environments/envs/kaggriculture/kaggriculture.py), 2026-08-15 file commit | High for inspected source; live deployment alignment unverified |
| Recent economic regime change | [Kaggle change #1399](https://github.com/Kaggle/kaggle-environments/commit/28b6d8af3ce73926b3d0fda1410c1ddd8384ab8c), 2026-08-15 | High for change; no current winning strategy inferred |
| Game-level observation/scoring description | [Kaggle README](https://raw.githubusercontent.com/Kaggle/kaggle-environments/master/kaggle_environments/envs/kaggriculture/README.md), current retrieval | High for documentation; exact order adjudicated by pinned source |
| Public deterministic strategic-space evidence | [lonespear report](https://raw.githubusercontent.com/lonespear/kaggriculture/main/README.md), 2026-08-07; [COK-ZhangZiliang baseline](https://github.com/COK-ZhangZiliang/Kaggriculture), retrieved 2026-09-05 | Anecdotal development evidence; not current rank, BUILD source or replicated performance |
| CEGIS and sketches | [Solar-Lezama, Berkeley thesis](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2008/EECS-2008-176.html), 2008 | Strong algorithmic foundation; complete game specification absent |
| Efficient bounded term search | [Reynolds et al., CVC4SY](https://theory.stanford.edu/~barrett/pubs/RBN%2B19-abstract.html), CAV 2019; [cvc5 example](https://cvc5.github.io/docs/cvc5-1.3.0/examples/sygus-fun.html), versioned docs | Strong for local synthesis; no whole-game scaling claim |
| Online symbolic search ordering | [Barke et al., Probe](https://arxiv.org/abs/2010.08663), OOPSLA 2020 | Supports partial-solution grammar adaptation; game transfer untested |
| Semantic factorization/refinement | [Meyer and Tepe](https://popl26.sigplan.org/details/POPL-2026-popl-research-papers/77/Oriented-Metrics-for-Bottom-Up-Enumerative-Synthesis), preprint 2025 / POPL 2026 | Strong in structured synthesis domains; no justified strategic metric here |
| Return-then-explore principle | [Ecoffet et al.](https://arxiv.org/abs/2004.12919), Nature 2021 | Supports state-return exploration; neural robustification not adopted |
| Base-policy continuation reasoning | [Bertsekas et al.](https://www.mit.edu/~jnt/Papers/J066-97-rollout.pdf), 1997 | Supports rollout mechanism; its improvement theorem not asserted for compiled family repairs |
| Subgoal/feature planning | [Lipovetzky and Geffner](https://nirlipo.github.io/publication/lipovetzky-2012-width/), 2012; [Bonet and Geffner](https://publications.rwth-aachen.de/record/988540), 2024 | Supports option decomposition; game width unknown |
| Library discovery | [Bowers et al.](https://arxiv.org/abs/2211.16605), POPL 2023; [Stitch](https://github.com/mlb2251/stitch), implementation | Supports corpus-guided abstraction; strategic gain requires new-task tests |
| Adaptive bounded neighborhoods | [Ropke and Pisinger](https://pubsonline.informs.org/doi/epdf/10.1287/trsc.1050.0135), 2006 | Useful mechanism; transportation results not transferred numerically |
| Rare-event structured fuzzing | [Lemieux and Sen](https://www.carolemieux.com/fairfuzz-ase18.pdf), ASE 2018; [Fioraldi et al.](https://www.usenix.org/conference/woot20/presentation/fioraldi), WOOT 2020 | Supports coverage/mutation design; rare coverage need not be valuable strategy |
| Fuzzing reproducibility | [LLVM Entropic discussion](https://reviews.llvm.org/D73776?id=241686), 2020; [libFuzzer docs](https://llvm.org/docs/LibFuzzer.html), current retrieval | Supports avoiding time-based scheduling; our full determinism must be tested |
| QD and archive scaling | [Mouret and Clune](https://arxiv.org/abs/1504.04909), 2015; [Vassiliades et al.](https://arxiv.org/abs/1610.05729), 2016 | Supports diversity retention and bounded cells; no non-saturation guarantee |
| Attainable coevolutionary challenges | [Brant and Stanley](https://stars.library.ucf.edu/scopus2015/7496/), GECCO 2017 | Supports minimal criteria; bounded failure is not impossibility |
| Nontransitive population coverage | [Balduzzi et al.](https://arxiv.org/abs/1901.08106), 2019 | Supports preserving strategic diversity; full exploitability not measured |
| Semantic GP limits | [Moraglio et al.](https://citeseerx.ist.psu.edu/document?doi=050e6c0dbbe93a9e5c8f1491e868761406054e96&repid=rep1&type=pdf), 2012 | Function-semantic results do not imply smooth reactive-policy fitness |
| Superoptimization / equalities | [Schkufza et al.](https://arxiv.org/abs/1211.0557), 2012/2013; [Willsey et al.](https://arxiv.org/abs/2004.03082), POPL 2021 | Local optimization evidence; not a strategy-discovery oracle |
| Finite route constraints and compression | [RC2](https://pysathq.github.io/docs/html/api/examples/rc2.html), current docs; [CP-SAT](https://developers.google.com/optimization/cp/cp_solver), current docs; [Bryant](https://www.cs.cmu.edu/~bryant/pubdir/ieeetc86.pdf), 1986 | Suitable finite subproblems; neither additive payoff nor small BDD guaranteed |
| Adaptive holdout risk | [Blum and Hardt](https://proceedings.mlr.press/v37/blum15.html), ICML 2015 | Strong warning about repeated exposure; this plan does not claim Ladder's theorem for game panels |
| Recent neural alternatives excluded | [Parsert and Polgreen](https://ojs.aaai.org/index.php/AAAI/article/view/28938), 2024; [Faldor et al.](https://arxiv.org/abs/2502.02190), 2025 | Their full methods violate the primary-component directive |
| Claimed first-place probability / speedup | No direct evidence | Unknown; all leverage gates and compute allowances are proposed experiments |

### Remaining consequential gaps

| Gap | Why it matters | Resolution |
| --- | --- | --- |
| Snapshot support and complete controller state in the current native APIs | Determines implementation effort and fidelity | Narrow Phase-1 implementation inspection; no unsupported claim that a public fork API already exists |
| Exact deployed official engine/package/config | Native clones can agree with each other but not the competition | Pin and compare in Phase 1; confirm again before release |
| Lane discussion content and official competition rules | Cannot certify its advice, deadlines or exploit permissions | Obtain readable official material; do not infer permission from engine behavior |
| Useful repair density and legal-gate retention | Central architecture hypothesis | Phase 1, then three-context Phase 2 |
| v1's effective body/feature ceiling | Could make a good local planner impossible to compile | Representation witnesses, not speculative VM expansion |
| Long-horizon and large-portfolio interaction frequency | Determines whether later complexity pays | Phases 6-7 only with measured prerequisites |
| True external strategic distribution | Internal wins may not generalize | Independent HOLDOUT audit, uncertainty reporting, no identity construction |

### Research procedure and stopping decision

Discovery covered official Kaggriculture source/documentation, public deterministic implementation reports, CEGIS/enumeration/solver implementations, superoptimization, planning/rollouts, symbolic library discovery, semantic GP/QD, coevolution, fuzzing and adaptive validation. Follow-up concentrated on counterexample reuse, source-level event ordering/randomness, recent 2024-2026 alternatives, and the reproducibility/guarantee limits of the shortlisted methods.

Consequential implementation claims were checked against primary source or official tool documentation. Official engine commit metadata was retrieved directly from GitHub after a sandbox DNS failure was retried with approval. No Local-Mcp was used. No subagent research was incorporated; research and synthesis were completed directly after the user requested solo work.

Research stopped when each major design decision had either primary support or an explicit empirical gap and a bounded test. More broad paper collection is unlikely to decide whether this particular v1 representation has dense, routable repair opportunities; that uncertainty requires the next experiment, not another literature list.

**Execution decision: proceed to Phase 1 only. The complete deterministic roadmap is ready for gated execution; its first-place and information-efficiency hypotheses remain falsifiable.**
