# ReCurSiON BonsaiEvolve — Final LLM-Guided Lane Evolution Plan

## Executive decision

**The active discovery architecture is now LLM-guided lane evolution over executable Bonsai/BKG4 policies, with exact native simulation as the sole policy-quality oracle.**

General-purpose LLM agents are used for what the deterministic systems repeatedly failed to provide: broad semantic hypothesis generation, source-level reasoning, causal debugging, structural program edits, and cross-mechanism composition. They are **not** trusted evaluators. Every proposal must compile, execute legally, and survive exact CUDA/native evaluation before it can enter a lane archive. Final release authority remains the complete composed BKG4 policy, not an LLM judgment.

This plan deliberately reverses the previous workflow:

```text
old:
    humans choose a narrow policy representation
        -> deterministic search explores only that representation
        -> repeated local-optimum / representation failures

new:
    strong general LLM agents inspect code + evidence
        -> propose executable strategic programs and mechanisms
        -> exact simulator selects what is real
        -> repeated successful structures become the representation
```

The runtime remains Bonsai/BKG4. CUDA remains the large-scale truth oracle. CPU/C++ remains the bounded reference. Official Python remains the final semantic canary. The authoritative champion remains `assets/champion/` until an independently validated composed candidate replaces it.

The previous RL-oriented plan, `docs/FINAL_POLICY_DISCOVERY_ARCHITECTURE.md`, is intentionally retained as the last learned-policy alternative and research reference. It is **not** the active execution plan because its proposed executable interface was subsequently falsified. All older live planning documents are retired from authority; their historical contents remain recoverable from Git history and their experiment evidence remains under `workspace/`.

**THIS IS THE ARCHITECTURE WE SHOULD BUILD NEXT.**

---

## 1. Evidence forcing the change

Current planning baseline: clean `main` at `47273c4c4c29c53eea962bf650c52f76479102e6` (`Diagnose service-tour transfer failure`) when this plan was written.

The project has now falsified several distinct hand-designed discovery spaces:

- Local symbolic repair/CCPR improved search efficiency but did not produce broad new economic capability.
- Replay-seeded and uninformed production-route search both failed to find a discovery-utility improvement in 128-candidate arms.
- The mechanics fallback produced a mirror-tie effect rather than broad strengthening.
- The generic economic-plan/task-executive representation retained only about 13% of Shop-level median cash on its decisive preflight.
- The more sophisticated resource-contracted service-tour representation could reproduce about 98% of Shop cash in one regime, but only about 54% median retention across the multi-configuration paired diagnostic.
- The committed postmortem at `workspace/policy_discovery/service_tour_postmortem/summary.json` classifies that failure as **B — PROGRAMME-BODY / EXECUTION INCOMPATIBILITY**: tour completion was only about 76.5–77.1%, production was broadly lower, the first persistent failure preceded Shop route selection, and no small programme-level counterfactual recovered the gap across contexts.

The shared lesson is not that exact simulation, Bonsai, or structured policy is wrong. The stronger diagnosis is:

> **We repeatedly constrain search by choosing the policy language before we know what the winning causal structures are.**

The missing component is a proposal mechanism capable of making large, coherent, source-aware conceptual jumps while still being grounded by exact evaluation.

Generic RL remains an open research alternative, not an established solution. The retained RL plan correctly identifies long-horizon credit and action-representation problems, but its first executable interface failed before the learner was tested. There is currently insufficient evidence to justify training a large learned policy stack before trying a cheaper proposal source that already has broad program reasoning capability.

---

## 2. Why LLM-guided evolution fits Kaggriculture

Kaggriculture has an unusually favorable separation between **hard-to-invent** and **easy-to-check**:

- Inventing a strong opening, financing sequence, workforce schedule, route composition, exploit, or causal router is difficult.
- Running that concrete policy in the exact simulator and measuring the result is comparatively cheap.
- We already have strong source policies, exact CPU/CUDA engines, BKG4 compilation, historical failures, replay diagnostics, and bounded safety gates.

This matches the useful design principle behind systems such as FunSearch and AlphaEvolve: use an LLM as a creative program proposer and an automated evaluator as the authority. FunSearch additionally demonstrates the value of program skeletons and diversity-preserving islands; AlphaEvolve demonstrates that LLM-driven evolutionary program improvement can operate over larger codebases when automated evaluators provide objective feedback. The 2024 LLM-guided programmatic-policy work similarly supports using pretrained programming knowledge to guide program search rather than relying on blind mutation alone.

Primary references:

- FunSearch, Nature 2024: https://www.nature.com/articles/s41586-023-06924-6
- AlphaEvolve, Google DeepMind 2025/2026: https://deepmind.google/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/
- LLM-GS for programmatic policy search, 2024: https://arxiv.org/abs/2405.16450
- Project lane approach: https://www.kaggle.com/competitions/kaggriculture/discussion/734212

These sources motivate components; none proves that an LLM will discover a winning Kaggriculture policy. The architecture is therefore gated by a small first experiment before any large automation investment.

---

## 3. Non-negotiable contract

1. **Bonsai/BKG4 remains the final runtime.** No LLM, neural model, retrieval service, or cloud dependency exists in the submitted agent.
2. **LLMs propose; engines decide.** No candidate survives because an agent says it is better.
3. **CUDA is mandatory for large-scale evaluation.** CPU is for bounded reference/parity work. The official Python engine is used before release.
4. **CPU/CUDA semantics must match exactly.** Any unresolved semantic mismatch invalidates dependent results.
5. **Every strategic proposal must become executable code/data.** Prose-only ideas do not enter an archive.
6. **Lane specialization is allowed and encouraged.** A useful specialist does not need to be globally safe by itself.
7. **The composed final policy is the regression unit.** Specialists are allowed to lose outside their activation region; final routing/composition must satisfy the protected release gates.
8. **Runtime routing may use only legal causal observations.** No opponent identity, submission name, episode ID, hidden seed, future observation, or private opponent state.
9. **No adaptive use of INTERNAL_DEV/TRANSFER/VALIDATION/SEALED.** Once a candidate sees a downstream result, it is frozen for that evaluation epoch. SEALED remains unavailable to generation agents.
10. **Source-informed discovery must be declared.** Any public/historical policy source shown to an LLM as a target becomes `TRAINING_SOURCE` for that epoch and cannot later be claimed as an independent holdout.
11. **Maintain genuinely untouched transfer evidence.** At least one source-diverse family plus independent internal adversarial policies must remain unseen by proposal agents for each release epoch.
12. **Final protected win-to-loss must remain zero unless an explicit future governance decision changes that standard.**
13. **Use the official 64/128-game canary before submission.** Broad runtime/semantic changes use 128.
14. **`assets/champion/` is immutable during discovery.** Promotion is a separate reviewed step.
15. **Repository hygiene is mandatory.** No `scratch/`; every non-ignored created file is committed or removed.
16. **Do not build an elaborate autonomous LLM platform before the bounded manual-agent pilot proves that LLM proposals contain useful strategic signal.**

---

## 4. The core architecture: BonsaiEvolve

```text
                       frozen champion + lane elites
                                  |
                   compact lane-specific evidence packet
                                  |
                   multiple independent LLM agents
                     / invent / critique / combine \
                                  |
                        executable candidate patch
                                  |
             compile -> legality -> CPU/CUDA semantic checks
                                  |
                       exact CUDA BUILD evaluation
                                  |
                 scored + behaviorally described candidate
                                  |
                    lane-specific diverse elite archive
                                  |
                         fresh INTERNAL_DEV gate
                                  |
                       router/composition lane
                                  |
                 complete composed-policy validation
                                  |
          TRANSFER -> VALIDATION -> CEGIS/SEALED -> official
                                  |
                         reviewed champion promotion
```

The system searches **programs**, not scalar parameter vectors. An LLM may create or replace a BKG4 subtree, rewrite a route/program, change an opening, introduce bounded state, alter market logic, compose two mechanisms, or restructure a strategy. It may also propose a new reusable abstraction when repeated evidence justifies it.

We do **not** first invent a universal new DSL. Existing BKG4/runtime/source mechanisms are the initial substrate. Useful abstractions are formalized only after successful candidates demonstrate that they recur.

---

## 5. Lane architecture

A single global fitness pool would rapidly collapse toward the current strongest family. We instead maintain independent lanes with separate archives and objectives. This preserves specialists and forces exploration of strategically different mechanisms.

### Lane A — Opening / finance / labour

Focus: steps 0–144 and the causal consequences of opening capital allocation.

Typical hypotheses:

- different ordered financing sequences;
- daily hire timing and spawn geometry;
- early crop bridge versus livestock acquisition;
- delayed/accelerated land;
- capital reserve/realization interaction.

A candidate can survive this lane if it creates a materially stronger opening regime even if it is not yet globally dominant.

### Lane B — Crops / land / spatial economy

Focus:

- crop mixture and lifecycle design;
- fertilizer/water/renewal programmes;
- tomato/perennial structures;
- quadrant acquisition and spatial service geometry;
- new crop-heavy or mixed economies.

### Lane C — Livestock / production

Focus:

- cow/sheep/goose programmes;
- herd composition;
- service/harvest coordination;
- feed financing;
- labour allocation and product realization.

### Lane D — Market / realization / working capital

Focus:

- ordered buy/sell sequences;
- advanced sale and debt handling;
- product reservation versus liquidation;
- market saturation response;
- terminal realization;
- financing exploits that are legal under the engine.

### Lane E — Mechanics / exploit / recovery

Focus on legal engine mechanics and robust runtime behavior:

- action ordering;
- hire spawn behavior;
- dawn deposit;
- inventory insertion order;
- planting resource coupling;
- market ordering;
- bounded recovery and failure handling.

This lane must show actual economic value. A mechanically elegant change with no strategic effect does not survive merely because it is correct.

### Lane F — Source-informed / best-response specialists

Agents may inspect declared `TRAINING_SOURCE` policies and source code to identify assumptions, vulnerabilities, and strong reusable mechanisms. The output must be a causal executable policy, never an identity lookup.

A specialist can target a known economic family, but its activation predicate must be expressible using legal runtime state such as visible herd/crops, cash, land, workforce, market inventory/prices, and causal history.

Any source exposed to this lane loses holdout status for that epoch.

### Lane G — Novel full-policy / generalist

This lane encourages large coherent changes that do not fit one local mechanism lane. Agents may rewrite openings and downstream economics together, borrow validated mechanisms across lanes, or propose an entirely new full-policy structure.

The purpose is to prevent the lane system itself from becoming another restrictive representation.

### Lane H — Router / composition

This lane receives already demonstrated specialists and generalists. It does not invent speculative economic bodies first.

Tasks:

- synthesize legal causal activation predicates;
- resolve overlaps/conflicts;
- preserve exact fallback;
- build nested binary routing when possible;
- combine multiple specialists into one BKG4 policy;
- find composition regressions and repair routing rather than weakening the specialist unnecessarily.

Binary routing remains the default because it is already proven and auditable. More complex stateful routing requires measured incremental value.

---

## 6. What an LLM agent receives

Agents should receive a **small, high-information packet**, not thousands of raw replays.

Every lane task packet should contain:

- exact parent/champion identity and relevant source paths;
- lane charter and allowed strategic scope;
- current lane elites and their semantic summaries;
- compact payoff/utility matrix;
- fresh BUILD failure clusters;
- first consequential divergences for representative cases;
- cash/production/market trajectories where relevant;
- engine/runtime source snippets needed for the mechanism;
- current BKG4/resource limits;
- prior failed hypotheses for the lane;
- explicit forbidden data and evaluation roles;
- a fixed evaluation budget.

For source-informed lanes, include only sources explicitly declared `TRAINING_SOURCE`.

Agents should be instructed to inspect actual repository source before editing. Replays are supplementary evidence, not a substitute for source when source exists.

---

## 7. Agent roles

We should exploit model diversity and role diversity rather than ask every agent to perform the same mutation.

### Inventor
Proposes a substantive causal mechanism and implements it.

### Source analyst
Studies engine/agent source and extracts assumptions, vulnerabilities, or reusable mechanisms.

### Counterexample analyst
Explains why a candidate lost from synchronized economic traces and proposes a bounded causal repair.

### BKG4 implementer
Translates a strategic idea into clean deployable BKG4/runtime logic without hiding strategy in opaque helpers.

### Adversarial reviewer
Attempts to falsify the candidate before expensive evaluation: missing obligations, illegal information, state corruption, overfit predicates, brittle timing, composition conflicts.

### Composer/router
Combines demonstrated lane elites and searches for legal activation conditions.

### Simplifier
After a successful candidate exists, reduces unnecessary code/state while proving behavior/strength retention. Simplification is never performed before there is a successful mechanism worth preserving.

Initially these are **prompt roles for existing general agents**, not separate services to engineer.

---

## 8. Candidate contract

Every generated candidate must have a compact record:

```text
candidate_id
lane
parent_id
agent/model/session identity (for provenance only)
hypothesis
mechanism category
files changed
semantic diff summary
expected causal effect
known risks
training sources shown
BUILD seed/panel identity
BKG4/controller/route hashes
resource usage
behavior descriptor
BUILD results
DEV status
composition status
```

A candidate must be reproducible from a fixed parent plus its committed patch/source. We do not keep giant logs.

Recommended tracked archive root after Phase 1 proves useful:

```text
workspace/llm_lane_evolution/
  manifest.json
  lanes/<lane>/archive.jsonl
  candidates/<accepted-id>/summary.json
  candidates/<accepted-id>/proposal.md
  candidates/<accepted-id>/metrics.json
```

Failed candidates normally retain only a compact semantic/failure record. Do not commit thousands of transient candidate binaries.

---

## 9. Candidate evaluation: LLM creativity, deterministic authority

### Gate 0 — static validity

Before simulation:

- compile/lower successfully;
- obey legal observation boundary;
- no prohibited file/data access;
- fit declared BKG4/runtime limits;
- deterministic reset/state ownership;
- no unbounded native strategic helper;
- focused unit/parity checks for new semantics.

Invalid candidates die immediately.

### Gate 1 — tiny lane BUILD screen

Use fresh internal BUILD contexts appropriate to the lane and both seats. This is a cheap causal screen, not promotion.

Require:

- the intended behavior actually occurs;
- no runtime faults;
- no catastrophic collapse across the small general support panel;
- measurable lane-target effect or clearly new strategic phenotype.

A candidate with no realized semantic difference is rejected even if hashes differ.

### Gate 2 — CUDA lane confirmation

Promising candidates receive a larger fresh CUDA evaluation across:

- lane-target contexts;
- multi-regime internal self-play;
- latest internal challengers;
- both seats;
- matched seeds.

Archive useful specialists even when they are not global winners, provided their target capability is real and they are safe behind fallback.

### Gate 3 — INTERNAL_DEV

Freeze candidate code and routing hypothesis before INTERNAL_DEV. DEV results may accept/reject the frozen candidate but may not tune it. If DEV evidence is later used to design a new candidate, that DEV panel is retired and replaced for the new epoch.

### Gate 4 — composition

Useful bodies enter the router lane. Test the **complete composed policy** against fresh internal opponents and historical ReCurSiON policies.

This is where regression control becomes strict.

### Gate 5 — downstream firewalls

The complete frozen composition proceeds through:

```text
TRANSFER -> VALIDATION -> CEGIS/SEALED -> official Python canary
```

No LLM sees SEALED case-level evidence.

---

## 10. Lane fitness: do not collapse specialists into one scalar

Each lane archive uses lexicographic survival:

1. semantic/legality correctness;
2. lane-target capability;
3. multi-context robustness inside the lane;
4. bounded general support / fallback safety;
5. behavioral novelty versus existing elites;
6. resource/complexity cost as a tie-breaker.

Do **not** require every specialist to beat the champion globally. That would destroy the main advantage of Bonsai.

Examples:

- A sheep-heavy body that strongly beats a visible sheep/crop regime but loses elsewhere can survive as a specialist.
- A market exploit that only applies under a causal inventory/price condition can survive if the router can activate it safely.
- A generalist candidate must meet a much broader support standard because it has no narrow fallback region.

The final composed policy, not the individual body, is held to release-level regression requirements.

---

## 11. Diversity preservation

Use **lane archives as islands**. Within each lane retain several elites with distinct behavior descriptors rather than only the current best score.

Useful descriptors include:

- opening acquisition vector;
- day-3/day-6 workforce;
- land timing;
- crop mix;
- herd mix;
- spatial footprint;
- working-capital trajectory;
- product realization profile;
- payoff vector against internal challenger cohorts.

Candidate code hash is not a diversity descriptor.

Periodically allow cross-lane recombination by a composer agent. Recombination is not automatic; it must preserve contracts/semantics and is evaluated as a new candidate.

---

## 12. Source-code use and anti-overfitting governance

The project gains value from letting intelligent agents read strong public source code, but this must be separated from validation claims.

### TRAINING_SOURCE

A public/historical source may be explicitly assigned to the source-informed lane. Agents may inspect it, reason about it, and design best responses or borrow mechanisms. That policy is then permanently marked `TRAINING_SOURCE` for the current epoch.

### HOLDOUT_SOURCE

Policies not shown to generation agents remain independent transfer evidence. Their source, detailed traces, and case-level results are withheld until the frozen candidate reaches the appropriate downstream gate.

### Runtime restriction

Even a source-specific specialist must route on causal legal state, never on source identity. If its win requires a fingerprint that has no strategic causal meaning, reject it.

### Leaderboard restriction

The public/hidden leaderboard is **release evidence, not the optimizer**. Do not repeatedly submit variants and feed leaderboard deltas back to LLMs as a mutation reward. A submission result that influences later design becomes exposed evidence and cannot still be described as independent validation.

We cannot guarantee zero regression against unknown hidden opponents. The practical defense is broad internal adversarial pressure, source-diverse withheld transfer, own-history evaluation, causal routing, and very limited final leaderboard queries.

---

## 13. Historical ReCurSiON and adversarial pressure

Every serious composition must face:

- the current champion;
- strong isolated Shop/Gold-derived references;
- all relevant historical submitted ReCurSiON policies available locally;
- current internal best-response/adversarial policies;
- diverse lane elites that expose different economies;
- source-diverse HOLDOUT policies at the transfer stage.

Historical ReCurSiON is never treated as an external opponent family for runtime identity routing. It is simply regression pressure.

The long-run objective remains the strongest possible composed agent, ideally reaching 100% against known meaningful policies, including historical ReCurSiON. Do not fake that objective by overfitting a fixed small panel.

---

## 14. Router/composition rules

Specialists make Bonsai valuable only if routing is causal and safe.

The router agent receives:

- candidate bodies already validated in their lanes;
- state/effect telemetry from BUILD only;
- overlapping activation examples;
- parent/generalist fallback;
- route-conflict and failure cases.

It may propose predicates over legal state such as:

- public opponent herd/crop state;
- own economic state;
- land/workforce;
- public market inventory/prices;
- asset maturity/deprivation;
- bounded causal history maintained by the policy.

It may not use:

- opponent name/source/team;
- replay hash;
- episode/seed identity;
- hidden/private opponent state;
- future shop/price information.

Default composition is nested binary routing:

```text
if specialist_A_safe_and_useful(state):
    A
else:
    parent_or_next_router
```

A specialist may have internal state, but exact PARENT fallback and obligation handling must be explicit. If composition creates regressions, first diagnose routing/interaction; do not automatically weaken a valuable specialist.

---

## 15. First experiment — manual LLM lane pilot

Do **not** build a large autonomous agent farm first.

The first question is simpler:

> Can strong general LLM agents, given source and compact causal evidence, produce executable BKG4 changes with useful strategic effects that our deterministic proposal systems did not generate?

### Scope

Run a manual/serialized pilot with **8 independent agent attempts** across four high-information lanes:

- 2 agents: Opening / finance / labour
- 2 agents: Market / realization / mechanics
- 2 agents: Source-informed / best-response specialist
- 2 agents: Novel full-policy / generalist

Each attempt gets:

- one fixed parent;
- one lane packet;
- one primary proposal;
- at most one causal revision after BUILD feedback;
- no INTERNAL_DEV feedback during revision.

Agents may inspect source, engine code and declared TRAINING_SOURCE material. They must implement their own proposal cleanly rather than hand back prose.

### Parent set

Use a small fixed seed set of strong executable parents chosen before outcomes:

- authoritative champion for composition/generalist work;
- isolated Shop Plus where its structure is the relevant donor;
- at most one other verified strong current internal/public-derived executable family if needed.

Do not let each agent cherry-pick whichever parent makes its result look best.

### Evaluation budget

Keep the pilot bounded to approximately **<=10,000 native FGE total**, overwhelmingly CUDA, including revisions and confirmations. This is a hard research ceiling, not a target to consume.

Suggested funnel per candidate:

```text
static + semantic tests
-> 32-64 fresh BUILD games
-> if real effect: 128-256 fresh CUDA games
-> if still useful: 512-game multi-regime confirmation
```

Do not spend the 512-game stage on obvious failures.

### Pilot pass criteria

The LLM direction earns Phase 2 only if all of the following are true:

1. At least **6/8** attempts compile and produce the intended nontrivial behavioral change.
2. At least **4/8** are behaviorally distinct from their parent and from each other, not cosmetic rewrites.
3. At least **2 different lanes** produce a candidate with a material causal improvement on fresh BUILD confirmation.
4. At least **one** useful candidate is a strategic mechanism that was not already expressible as the small local mutation tested in the failed deterministic campaigns.
5. At least **one** candidate or routed specialist survives a fresh INTERNAL_DEV gate after freezing.
6. CPU/CUDA semantics for any new runtime behavior are exact.
7. The experiment remains within the budget and leaves the champion unchanged.

This does **not** require the first eight agents to produce the final winning agent. It requires evidence that LLM semantic proposals have better information content than the exhausted deterministic proposal generators.

### Pilot kill criteria

Stop/reconsider if:

- agents mostly rediscover known failed tweaks;
- proposals repeatedly require illegal/private information;
- valid proposals are overwhelmingly cosmetic;
- no two lanes produce useful fresh BUILD effects;
- the only positive effects disappear immediately outside the exact targeted case;
- agent-generated code is so broad/opaque that causal evaluation becomes impractical;
- the evaluator cannot keep the loop exact and cheap.

One prompt/context redesign is allowed if the first packet design clearly starved agents of necessary evidence. Do not indefinitely blame prompting for a failed approach.

---

## 16. Phase 2 — lane archive and iterative agent evolution

Only after the manual pilot passes:

1. Formalize `workspace/llm_lane_evolution/` candidate memory.
2. Create standard lane task-packet builders from existing telemetry.
3. Maintain 2–4 diverse elites per active lane.
4. Run small batches of independent LLM proposal sessions.
5. Give revision agents exact BUILD counterexamples and parent/candidate diffs.
6. Periodically assign composer agents to combine successful mechanisms.
7. Use CUDA for hundreds/thousands of games only after cheap screens.
8. Refresh BUILD/adversarial contexts regularly so the population cannot optimize one frozen panel forever.

Do not fine-tune a local model initially. The advantage being tested is the reasoning ability of strong general models.

Automation should reduce clerical work, not replace scientific gates.

---

## 17. Phase 3 — adversarial lane generation

Once several useful policies exist, use LLM agents not only to strengthen the main agent but to build **challengers**.

Challenge agents receive the current composed candidate and are asked to construct legal counter-strategies. Their objective is to expose weaknesses, not become the final policy.

Useful challengers enter the internal adversarial league and future BUILD mixtures.

This creates a closed improvement loop:

```text
main candidate
    -> LLM challenger agents invent counters
    -> exact CUDA finds real weaknesses
    -> lane agents repair or add specialists
    -> router composes
    -> new main candidate
```

Do not use challenger identity at runtime. The challenger exists to generate strategic pressure and reachable failure states.

---

## 18. Phase 4 — release composition and no-regression discipline

Before any champion promotion, freeze the complete composed BKG4 policy and run the full safety sequence.

Minimum release evidence should include:

- broad internal multi-regime BUILD history;
- fresh INTERNAL_DEV;
- all relevant historical ReCurSiON policies;
- current adversarial league;
- independent source-diverse TRANSFER;
- VALIDATION;
- CEGIS/SEALED according to existing protected-data governance;
- exact CPU/CUDA parity;
- official Python 64/128 canary;
- complete BKG4/resource/runtime audit.

**Final protected W→L remains zero.** A specialist may be individually imperfect, but it may enter the final policy only through a composition that preserves the protected wins.

Leaderboard submission is a final release action, not a search generation.

---

## 19. Prompting protocol for optimization agents

Every optimization agent prompt should force the following workflow:

1. Verify repo HEAD/status and read current lane authority.
2. Inspect the exact parent implementation and relevant engine/source.
3. Read compact prior failures for its lane.
4. State one primary causal hypothesis before editing.
5. Explain why the hypothesis is not a previously failed local edit.
6. Implement the smallest coherent change capable of testing the hypothesis — which may still be structurally large if the mechanism requires coordinated changes.
7. Keep runtime information legal and deterministic.
8. Add only focused semantic tests.
9. Run bounded local checks; large evaluation belongs to CUDA.
10. Produce a machine-readable candidate summary.
11. Never promote or modify `assets/champion/`.
12. Stop when its assigned candidate/revision budget is exhausted.

The prompt should explicitly invite the agent to reconsider its approach, inspect source, and use web/research papers when stuck. It should not prescribe a narrow mutation grammar.

---

## 20. Memory: what agents should learn from past failures

Every lane packet should include a short `KNOWN_DEAD_ENDS` section. At minimum:

- RLRMB/Direct equal-budget search found no generalizable committed specialist in the tested representation.
- CCPR/local repair produced only narrow surviving effects.
- Post-144 route/lifecycle mutation did not create a stronger economy.
- Replay-informed route proposals were no better than uninformed proposals.
- Capacity/feed mechanics patches did not yield broad strength; feed preservation specifically harmed deliberate financing behavior.
- Generic target/task execution retained only ~13% of Shop productivity.
- Service-tour programmes reached ~98% in one regime but only ~54% multi-configuration retention and were killed after the postmortem showed body-level execution incompatibility.

Agents are allowed to revisit a mechanism only if they can explain what material assumption is different.

---

## 21. What not to build yet

Until the manual pilot passes, do not build:

- an autonomous distributed LLM scheduler;
- an LLM fine-tuning corpus;
- a new general policy DSL;
- a neural value/proposal model;
- a new world model;
- a large vector database;
- an AlphaStar-scale league service;
- a million-candidate evolutionary campaign;
- complex multi-agent messaging infrastructure.

Use the existing general agents manually/serially first. The point is to test **proposal intelligence**, not infrastructure.

---

## 22. Why this can escape the previous local optima

The deterministic systems required us to decide in advance what a legal mutation looked like. As a result, they could not propose a mechanism outside the hand-authored grammar.

An LLM agent can simultaneously reason about:

- an opponent's source code;
- BKG4 structure;
- engine ordering;
- financing;
- worker scheduling;
- crop/livestock biology;
- market behavior;
- prior failed experiments;
- a counterexample trace;

and then rewrite several interacting strategic mechanisms coherently.

For example, it may conclude that a herd change requires a new crop bridge, hire pattern, land date, service schedule, and realization sequence together. The evaluator then determines whether that theory is real.

This does not remove representation risk. It moves representation discovery **inside the creative loop** instead of freezing it before search. Repeated successful mechanisms can later be standardized as Bonsai primitives or templates.

---

## 23. Main risks

### LLM hallucination
Mitigation: compile and exact evaluation. No semantic claim is accepted from prose.

### Overfitting a named opponent
Mitigation: declared TRAINING_SOURCE status, causal runtime predicates, fresh internal support, withheld transfer families, final composition gates.

### Candidate bloat
Mitigation: resource accounting and a simplifier stage only after strength is demonstrated. Do not optimize for elegance before value.

### Duplicate ideas
Mitigation: semantic lane archive and explicit dead-end memory.

### Hidden strategy in native helpers
Mitigation: code review; strategic choice must remain visible in BKG4/program source. Native helpers may implement deterministic mechanics only.

### Agents converging on the same public solution
Mitigation: role/lane diversity, independent contexts, island archives, novelty descriptors, and explicit novel full-policy lane.

### Adaptive validation overfit
Mitigation: strict BUILD/DEV/TRANSFER/VALIDATION/SEALED lifecycle and retirement of any panel whose results feed later design.

### False confidence from leaderboard
Mitigation: leaderboard is not an optimization reward; limited release submissions only.

---

## 24. Success definition

The first success is **not** rank 1. It is evidence that the LLM proposal mechanism generates strategically useful executable novelty that the deterministic generators missed.

The system is mature when it can repeatedly:

1. discover new lane capabilities;
2. preserve them as specialists or generalists;
3. generate adversarial counters;
4. synthesize causal routers;
5. compose without protected regression;
6. continue improving against its own historical agents and strong external families.

The long-run aspiration remains a Bonsai portfolio that wins against all meaningful known policies, including historical ReCurSiON, while generalizing to hidden leaderboard opponents.

No architecture can guarantee zero hidden-leaderboard regression. The engineering objective is to make hidden regression increasingly unlikely by widening causal policy diversity and adversarial validation rather than fitting the leaderboard itself.

---

## 25. Immediate next action

**Do not build an autonomous BonsaiEvolve service yet.**

Create the Phase-1 manual LLM lane pilot described in Section 15. The next implementation session should only:

- create the minimal lane/candidate evidence format;
- freeze the first four lane packets;
- define the exact 8-agent task protocol;
- define the bounded CUDA funnel;
- run the first agent proposals through that funnel;
- stop with a PASS/FAIL verdict on whether LLM-guided lane evolution supplies useful strategic proposal signal.

If the pilot passes, automate the clerical loop. If it fails, retain the last RL plan as the next serious research alternative rather than restarting deterministic mutation search under another name.

**FINAL VERDICT: BUILD THE BOUNDED LLM-GUIDED LANE PILOT NEXT.**
