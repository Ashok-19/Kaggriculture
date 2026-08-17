# Bonsai-KG Deterministic Learned Option-Value Architecture

## Status

**Proof-of-concept architecture proposal. Not implemented. BKG3 remains frozen.**

This document records a reviewed Kaggriculture-specific architecture direction for future ReCurSiON research. It deliberately separates confirmed properties of the current repository from proposed work and longer-term hypotheses.

### Read this first

1. **Bonsai-KG remains the policy and control backbone.** This proposal does not replace it with end-to-end RL.
2. Learning is restricted to **option value / ranking at bounded strategic decision points** chosen by Bonsai.
3. A learned component does **not** emit arbitrary primitive game actions; Bonsai defines admissible candidates and executes the selected macro/subtree.
4. The first runtime representation should use **deterministic integer inference**, not ordinary floating-point neural inference.
5. The first integration should sit **immediately outside frozen BKG3**, not add a new VM opcode or casually create BKG4.
6. Any counterfactual search branch must clone **both simulator state and Bonsai `RuntimeState`**. Simulator-only snapshots are invalid for stateful-policy continuation.
7. The first experiment is **not model training**. It is an **oracle-headroom study** at the existing stateful-v6 stage-144/stage-216 route decisions.
8. Train a learned selector only if a perfect per-state choice among the existing route candidates shows substantial global headroom.
9. Future training/evaluation must use broad current, historical, emerging, and ReCurSiON populations, both seats, and broad seeds; opponent identity is never a production feature.
10. CPU/CUDA learned inference must satisfy an exact arithmetic contract: **`score_cpu == score_cuda` bit-for-bit**.
11. Before any Kaggle submission, the actual candidate must still pass the required official-Python 64/128-game promotion canary.
12. The longer-term hypothesis is a **Bonsai-gated universal option critic**, not dozens of independent commodity agents.

---

## 1. Motivation

Kaggriculture is a 720-turn, two-player economic control problem with coupled production, worker capacity, land, shops, crop/livestock lifecycle constraints, and a nonlinear shared market. Strategic value can depend on interactions among:

- crop and livestock portfolio;
- capital timing;
- land expansion;
- daily worker capacity and hiring;
- watering, feeding, care, harvesting, and routing obligations;
- shop demand revealed during the season;
- market inventory and nonlinear prices;
- opponent public economic state;
- liquidation timing and terminal cash.

Bonsai-KG is already a strong fit for legality, safety, temporal execution, causal routing, and persistent control state. The proposed learned layer is not intended to teach the agent how to move, water, feed, harvest, or obey game rules. Its intended role is narrower:

> Given a legal state, current Bonsai memory, a decision context, and a bounded candidate option, estimate the future strategic value of that option well enough to rank the candidates.

The motivation is therefore a **representation and value-estimation problem at selected high-regret decisions**, not a claim that the current behavior-tree VM is memoryless or generally incapable. `RULES.md` currently states that Bonsai capacity is not a practical bottleneck and that search quality, feature/action/expert expressiveness, and economic basin coverage are often the limiting factors. This proposal respects that conclusion by requiring an oracle-headroom gate before any learned selector is trained.

---

## 2. Existing Bonsai-KG foundation — confirmed current implementation

Authoritative source: [`../../bonsai-kg/SPEC.md`](../../bonsai-kg/SPEC.md), [`../../bonsai-kg/include/bonsai_kg.hpp`](../../bonsai-kg/include/bonsai_kg.hpp), and [`../../bonsai-kg/include/bonsai_kg_abi.hpp`](../../bonsai-kg/include/bonsai_kg_abi.hpp).

### 2.1 BKG3 is stateful

BKG3 is not a memoryless decision tree. Confirmed current mechanisms include:

- per-simulation `RuntimeState`;
- typed persistent blackboard;
- stateful `Sequence` and `Selector`;
- reactive sequence/selector with deterministic preemption;
- `If` branch latching while running;
- lifecycle predicates/effects;
- temporal controls including `WaitTurns` and `TurnTimeout`;
- persistent node state;
- deterministic deferred-intent arbitration.

Day boundaries do not implicitly reset policy state. `RuntimeState` persists until reset for a fresh episode.

**Architectural consequence:** Bonsai owns execution memory and temporal control. A learned option-value model does not automatically need recurrent memory if its input exposes the legal current state plus the relevant Bonsai-owned memory. This does **not** eliminate the separate problem of estimating long-horizon consequences.

### 2.2 BKG3 is finite and deliberately constrained

Current hard limits include:

| Resource | BKG3 limit |
|---|---:|
| nodes | 128 |
| depth | 24 |
| child refs | 256 |
| blackboard slots | 32 |
| runtime state slots | 48 |
| ValueRefs | 256 |
| FeatureArgRefs | 512 |
| predicates | 128 |
| lifecycle descriptors | 64 |
| effects | 128 |
| deferred intents | 128 |

The current checked-in stateful-v6 compatibility controller records:

- `controller_nodes = 99`;
- `controller_state_slots = 48`.

Source: [`../../bonsai-kg/integrations/stateful_v6/assets/identity.json`](../../bonsai-kg/integrations/stateful_v6/assets/identity.json).

The fact that stateful-v6 already consumes all 48 runtime state slots is important for integration design. It does **not** prove that Bonsai tree capacity in general is the strategic bottleneck. It does mean that compiling a substantial learned forest/scorer directly into ordinary BKG3 `If`/`Condition` structure is a poor **first** integration strategy for this compatibility baseline.

### 2.3 BKG3 deliberately has no arbitrary model callback

The VM design explicitly excludes runtime plugins, arbitrary scripts/callbacks, reflection, dynamic allocation, and arbitrary expression evaluation. Portable policy code sees only the restricted legal feature interface.

Current portable ValueRef sources are:

- `CONST`;
- `BB`;
- `LEGAL_FEATURE`.

Therefore adding a node/opcode such as `NeuralModel`, `Advisor`, `ScoreOptions`, or `LearnedPolicyCallback` would be a real VM semantic and serialization expansion, not a cosmetic integration.

**POC rule:** do not modify BKG3 semantics or ABI merely to test whether learned option ranking has useful headroom. Do not casually create BKG4.

### 2.4 Legal observation and privacy boundary

The frozen registry permits legal current-observation information including own/opponent public money, land, units, hires, public farm state, shared market state, shops, own private shed/seeds/inventories, and public worker/tile lifecycle fields.

It explicitly excludes opponent private state, RNG/future RNG, future shops, replay/episode/submission identifiers, opponent/team identity, ReCurSiON identity, and native hidden caches/statistics.

Any future learned advisor must preserve this boundary. Its production inputs should be derived from:

> **legal BKG3 observation features + Bonsai-owned persistent state**

It should not receive unrestricted raw `Sim` access in production.

### 2.5 Exact execution contract

BKG3 policy arithmetic is integer. C++ and CUDA share native interpreter semantics, and promotion requires exact cross-engine behavior rather than reward-only agreement.

That standard must extend to any learned runtime component. Ordinary CPU/GPU floating-point inference is not presumed bit-identical.

---

## 3. Architecture principles — proposed POC

The proposed hierarchy is:

```text
BONSAI-KG
  deterministic hierarchical executive
  state / memory / safety / legality / lifecycle
                  |
                  v
        bounded strategic decision
                  |
                  v
     deterministic learned option-value
             / ranking model
                  |
                  v
      deterministic scores + argmax
                  |
                  v
         Bonsai latches choice
                  |
                  v
      Bonsai macro/subtree executes
                  |
                  v
           final game action
```

The core principle is:

> **Bonsai decides where learning is allowed.**

The learned component is an advisor/value estimator, not a general autonomous agent.

### 3.1 What Bonsai remains responsible for

Bonsai owns:

- when a strategic decision is opened;
- candidate generation or candidate-set selection;
- candidate legality and admissibility;
- safety and lifecycle invariants;
- persistent episode state;
- strategic phase management;
- latching and commitment;
- switching/re-evaluation rules;
- fallback behavior;
- tactical/macro execution;
- final primitive game action production.

### 3.2 What the learned component is responsible for

The learned component answers a bounded question such as:

```text
Value(
    legal_current_state,
    relevant_bonsai_memory,
    decision_context,
    candidate_option
)
```

or directly produces a vector of deterministic candidate scores for one fixed decision domain.

It does not:

- invent arbitrary actions;
- bypass legality checks;
- own the complete 720-turn controller;
- replace Bonsai state/lifecycle logic;
- access hidden identity or future information;
- decide where it may be invoked.

---

## 4. Decision-oriented option models

Specialists should be organized around **strategic decisions**, not commodity names.

### 4.1 Avoid independent commodity agents

Do not begin with independent agents such as:

```text
cow model
milk model
wheat model
strawberry model
...
```

Kaggriculture's economy is strongly coupled. A cow investment affects or depends on pasture occupation, capital, wheat/feed demand, worker actions, care timing, milk production, shed capacity, market pressure, land opportunity cost, shop demand, and remaining season length. Independent commodity objectives can therefore conflict.

### 4.2 Preferred decision domains

Potential bounded decision domains include:

- production portfolio;
- livestock investment;
- crop investment;
- capital allocation;
- land expansion;
- workforce/hiring;
- market liquidation;
- fertilizer allocation;
- worker/task allocation;
- endgame liquidation.

Resource-specific scores can exist internally, but should contribute to a joint option decision rather than operate as autonomous resource policies.

### 4.3 Option contract

A candidate option should be a meaningful Bonsai-executable strategic commitment or macro, not an unconstrained primitive-action proposal. Example:

```text
Decision: LIVESTOCK_PORTFOLIO
Candidates:
  0 = preserve current livestock plan
  1 = goose-biased expansion
  2 = cow-biased expansion
  3 = sheep-biased expansion
  4 = mixed expansion
```

Bonsai determines which candidates are legal/admissible in the current context and what happens after one is chosen.

---

## 5. Legal state and opponent features

Opponent information may be used only when it is legal, causal, and economically meaningful.

Useful public opponent context may include:

- money;
- unlocked land/quadrants;
- worker count and public worker positions;
- public crop composition;
- public livestock composition;
- public tile lifecycle state;
- visible production capacity or pressure derived from public state;
- interaction with current shared market state.

Never use:

- opponent/team name or id;
- submission id;
- replay id;
- episode id;
- seed;
- future shops;
- future RNG;
- hidden/private opponent shed, seeds, or worker inventories.

Prefer economically meaningful aggregates over identity-like fingerprints. If two states are legally indistinguishable at a commitment point, the deterministic production decision must remain identical.

---

## 6. State and memory

### 6.1 Bonsai owns persistent execution memory

A specialist can be memoryless as a function if its input contains the state needed for its decision, for example:

```text
legal observation features
+ relevant Bonsai blackboard values
+ strategic phase
+ current/previous commitment
+ decision-specific context
```

This reduces the need for a recurrent learned controller.

### 6.2 Snapshot search must clone Bonsai state

This is a hard requirement from the existing BKG3 semantics.

For any counterfactual continuation, copy:

```text
simulator snapshot
+
Bonsai RuntimeState
```

A simulator-only snapshot is insufficient because the policy may have latched branches, node cursors, blackboard commitments, and temporal state that affect future actions.

If a future advisor ever owns persistent runtime state, that state must also be cloned for counterfactual continuation. The preferred first advisor is stateless at runtime apart from immutable model parameters, which avoids adding another snapshot-state domain.

---

## 7. Deterministic inference contract

### 7.1 Exactness requirement

Future serving inference must satisfy an explicit CPU/CUDA parity contract:

```text
features_cpu == features_cuda
score_cpu == score_cuda
argmax_cpu == argmax_cuda
```

bit-for-bit for every accepted input.

### 7.2 Recommended first representation

The first model representation should be an **integer boosted or oblivious decision forest with vector-valued leaf outputs**.

Conceptually:

```text
input features: int32

split:
    feature[k] <= integer_threshold

leaf:
    int16/int32 score contribution
    for each candidate option

accumulation:
    int64

choice:
    deterministic argmax
    fixed tie-breaking order
```

The exact file format and runtime API are intentionally not defined here; that belongs to a later POC design only after oracle headroom is demonstrated.

### 7.3 Why this is the preferred first model

Advantages:

- represents nonlinear feature interactions;
- uses exact integer comparison/arithmetic;
- avoids runtime floating point;
- permits cheap CUDA inference;
- is straightforward to differential-test against C++;
- is auditable at feature/split/leaf level;
- can define explicit overflow/saturation behavior;
- needs no softmax, sigmoid, or stochastic sampling;
- supports vector scores across a bounded candidate set.

A quantized tiny MLP can be evaluated later if evidence shows the forest representation is insufficient, but it should not be the first POC by default.

### 7.4 Required arithmetic details before implementation

A real serving contract must eventually freeze:

- feature integer domains/quantization;
- split comparison semantics;
- leaf score types;
- accumulation width;
- overflow/saturation rules;
- candidate masking semantics;
- deterministic tie-breaking;
- invalid/missing feature behavior;
- model semantic/hash identity.

None of these are implemented by this document.

---

## 8. Training with CUDA counterfactual evaluation

### 8.1 Primary training philosophy

The preferred initial learning pipeline is:

```text
exact CUDA counterfactual search
            ->
option-value / ranking labels
            ->
compact deterministic model distillation
```

This is closer to approximate policy improvement over **Bonsai-defined options** than primitive-action end-to-end RL.

### 8.2 Counterfactual state construction

At a strategic decision state `S` with candidates `A, B, C, D`:

1. capture the exact simulator state;
2. capture the exact Bonsai `RuntimeState`;
3. clone both for each candidate;
4. force/apply the candidate option through the integration boundary;
5. continue with the intended continuation policy;
6. evaluate across broad seeds/opponents where the experiment design permits;
7. aggregate future outcomes into option statistics.

Large-scale branch evaluation belongs in the exact CUDA engine. C++/official Python are correctness and promotion tools, not the default large-search engine.

### 8.3 What the model should not relearn

Do not spend model capacity rediscovering:

- legal movement;
- watering/feeding survival rules;
- basic harvesting;
- shed/worker mechanics;
- action legality;
- the complete primitive action space.

Those remain in deterministic execution logic. The learned model should focus on strategic opportunity cost among competent options.

---

## 9. Value and ranking targets

### 9.1 Primitive outcome

The natural primitive terminal statistic is:

```text
terminal_margin = own_final_money - opponent_final_money
```

Raw own terminal money alone should not be the sole objective because the game is competitive and winning depends on the opponent's final cash.

### 9.2 Preferred learning emphasis

Training should emphasize **relative candidate value**, especially pairwise/listwise ranking within the same decision state.

Useful target form:

```text
ADV(S, A) = Value(S, A) - Value(S, current_Bonsai_choice)
```

Paired same-state comparisons reduce irrelevant baseline variation and directly answer whether an alternative is better than the incumbent choice.

### 9.3 Rollout statistics to retain

For each state/candidate pair, retain enough aggregate information to audit both central value and downside:

- win probability;
- loss probability;
- mean terminal margin;
- median terminal margin;
- lower quantile(s);
- CVaR or an equivalent tail-risk measure;
- paired advantage versus the current Bonsai choice where applicable.

### 9.4 Strategic objective

The promotion objective should preserve this priority:

1. winning first;
2. catastrophic-loss avoidance;
3. then margin quality.

A model that raises mean money while creating rare large losses is not automatically an improvement.

The exact scalarization/ranking loss is a later experimental question. It must be validated against tournament outcomes rather than chosen for ML convenience.

---

## 10. Generalization and anti-overfitting strategy

Any learned option-value system must be trained and evaluated against a broad population, not one selected rival.

Required controls include:

- current top agents;
- historical top agents;
- emerging strong agents;
- historical ReCurSiON agents;
- both seats;
- broad seed coverage;
- balanced economic/shop regimes;
- held-out seed ranges when provenance and experiment design make that meaningful;
- refreshed emerging strategies;
- leave-family-out validation.

Examples of leave-family-out tests:

```text
train without one strong external family -> evaluate on that family
train without one ReCurSiON historical family -> evaluate on it
train before a fresh population refresh -> evaluate on newly refreshed agents
```

If gains disappear under family withholding, treat the model as meta memorization until disproven.

Production features must remain causal legal state, never identity proxies.

---

## 11. Initial integration boundary with BKG3

### 11.1 First integration should remain outside the VM

The first learned scorer should sit **immediately outside frozen BKG3**.

Conceptually:

```text
BKG3 controller
+
deterministic advisor model
+
macro/expert assets
```

The advisor can participate at a narrow integration/controller boundary so that Bonsai can latch and execute the chosen bounded option, but the exact API is intentionally deferred until the oracle-headroom gate passes.

### 11.2 Why not add a VM primitive first

Adding a model primitive would immediately create questions about:

- ABI records;
- serialization;
- semantic hashing;
- validation;
- feature access;
- CPU/CUDA execution;
- failure semantics;
- snapshot semantics;
- backward compatibility.

None of that is justified before proving that learned selection has substantial strategic headroom.

### 11.3 Why not compile the first model into ordinary BKG3 nodes

BKG3 is intentionally finite, and the current stateful-v6 controller already uses 99/128 nodes and 48/48 state slots. A learned forest expanded into many ordinary conditions would consume scarce structural budget, obscure the model/BT boundary, and make exact model identity harder to reason about.

This is an integration argument for the first POC, not a claim that all future learned logic must permanently remain external.

### 11.4 Future identity concept

If the architecture is eventually implemented, a complete agent identity should include at least:

```text
BKG3 semantic hash
+
advisor model hash
+
macro/expert asset hash
```

The exact artifact format is not mandated yet.

---

## 12. Stateful-v6 as the first controlled experiment — confirmed baseline, proposed test

Authoritative integration source: [`../../bonsai-kg/integrations/stateful_v6/README.md`](../../bonsai-kg/integrations/stateful_v6/README.md) and its checked-in assets.

### 12.1 Confirmed current structure

The compatibility controller currently provides:

```text
pre-144 route 64
    ->
stage-144 threshold decision
    ->
route/leaf state stored in BKG3 blackboard
    ->
optional one-shot stage-216 refinement
    ->
fixed route expert/tape execution
```

The frozen BKG3 controller owns stage-144 selection, remembered leaf/route state, stage-216 refinement, and latching. The immutable route pack is an actuator and does not choose a route.

Current route IDs are exactly:

```text
29, 64, 32, 79, 46, 63, 22, 34, 77, 70, 23
```

This provides an unusually clean controlled boundary because downstream route execution can remain unchanged while selector representation is studied.

### 12.2 Do not train the learned router first

The actual first experiment is the **oracle-headroom gate** described in the companion POC document:

[`bonsai_kg_learned_option_value_poc.md`](bonsai_kg_learned_option_value_poc.md)

If perfect state-dependent selection among these eleven existing route experts offers only small global improvement, selector representation is not the primary bottleneck. The option/expert library must be improved instead.

---

## 13. Failure modes and mitigations

### 13.1 Candidate-set ceiling

**Failure:** the advisor selects perfectly among weak/insufficient options but cannot create a winning continuation.

**Mitigation:** run oracle headroom before training. If oracle gain is small, stop selector work and expand/improve the option library or intervene earlier.

### 13.2 Counterfactual label leakage

**Failure:** labels or production features accidentally depend on future shops/RNG, seed, replay identity, or other information unavailable at decision time.

**Mitigation:** separate label-generation-only future outcomes from serving features; enforce the legal BKG3 feature/privacy boundary in the model input pipeline.

### 13.3 Replay-response bias

**Failure:** the learned model memorizes responses to a narrow set of realized replay trajectories rather than general economic states.

**Mitigation:** broad counterfactual continuations, fresh seeds, multiple opponent families, causal public-state features, and family-withheld validation.

### 13.4 Policy distribution shift

**Failure:** once the learned router changes earlier decisions, later states move outside the training distribution.

**Mitigation:** iterative data regeneration under the updated policy, explicit coverage diagnostics, and deterministic fallback for poorly covered regions.

### 13.5 Repeated re-evaluation / switching

**Failure:** a value advisor oscillates among options on small score changes and destroys coherent macro execution.

**Mitigation:** Bonsai owns decision timing, commitment, latching, switching conditions, and hysteresis if later justified. The advisor does not decide when to re-open a decision.

### 13.6 Target misalignment

**Failure:** maximizing mean own money or mean margin reduces actual win rate or creates bad tails.

**Mitigation:** retain win/loss and tail-risk statistics; prioritize winning and catastrophic-loss avoidance before average-margin optimization.

### 13.7 Catastrophic-tail blindness

**Failure:** a candidate has attractive mean value but rare severe losses.

**Mitigation:** track lower quantiles/CVaR or equivalent downside statistics and include robust promotion gates, not mean-only acceptance.

### 13.8 Specialist conflicts

**Failure:** independently optimized resource or subsystem specialists fight over capital, land, workers, inventory, or market timing.

**Mitigation:** organize learning around joint decisions/options; prefer a shared option critic over autonomous commodity agents.

### 13.9 Search-label / continuation-policy bias

**Failure:** `Q(S,A)` reflects quirks of the continuation policy used after forcing `A`, rather than the true usefulness of option `A` under the intended production controller.

**Mitigation:** define continuation policy explicitly; use paired comparisons; re-evaluate after policy updates; distinguish option quality from downstream-controller weakness.

### 13.10 Float-to-integer quantization rank flips

**Failure:** a model trained in floating point changes candidate ordering when converted to integer serving form.

**Mitigation:** train/evaluate with the serving quantization contract in the loop or perform explicit post-quantization ranking audits before accepting the model.

### 13.11 Training-serving feature skew

**Failure:** offline feature extraction differs subtly from production C++/CUDA feature semantics.

**Mitigation:** derive serving features from the same legal feature definitions and maintain exact cross-engine feature parity tests.

### 13.12 Opponent-family memorization

**Failure:** the model learns brittle family fingerprints rather than economic structure.

**Mitigation:** no identity features/proxies; leave-family-out validation; current+historical+self population diversity; fresh-agent evaluation.

### 13.13 State aliasing

**Failure:** two strategically different states collapse to the same advisor feature representation.

**Mitigation:** diagnose contradictory labels at identical feature vectors; add only legal causal features or Bonsai memory needed to disambiguate; if futures are truly indistinguishable, optimize robust option value rather than invent an identity classifier.

### 13.14 Macro drift after retraining

**Failure:** an advisor trained for one version of an option/expert library becomes invalid after macros change.

**Mitigation:** hash/version option assets with the model, regenerate labels when option semantics change, and reject incompatible model/macro identities.

### 13.15 CPU/CUDA inference divergence

**Failure:** serving scores differ by engine due to arithmetic, overflow, comparison, or feature mismatch.

**Mitigation:** integer arithmetic contract, fixed tie-breaking, semantic hashing, randomized differential tests, and fail-closed model loading.

---

## 14. Long-term hypothesis: Bonsai-gated universal option critic

The likely higher-ceiling direction is **not** dozens of independent specialists. It is a shared critic conditioned on decision type and candidate option:

```text
Q(
    legal_state,
    Bonsai_memory,
    decision_type,
    candidate_option
)
```

Possible decision types:

```text
PRODUCTION_PORTFOLIO
CAPITAL_ALLOCATION
LIVESTOCK_INVESTMENT
LAND_EXPANSION
WORKFORCE
LIQUIDATION
ENDGAME_LIQUIDATION
```

Bonsai still defines:

- when the decision exists;
- the bounded candidate set;
- legality and safety;
- commitment/switching policy;
- macro execution;
- final game actions.

The shared critic estimates option value and can share learned representation across coupled economic decisions.

### 14.1 Fallback / coverage concept

A future design should preserve a deterministic fallback concept:

```text
if advisor coverage/confidence is inadequate:
    use deterministic Bonsai fallback
```

No exact confidence or OOD mechanism is defined here. It must be evidence-driven and deterministic if implemented.

---

## 15. Promotion requirements

A later learned-option candidate is not promoted because it improves one local screen.

Required evaluation dimensions include:

- current top population;
- historical top population;
- emerging strong agents;
- historical ReCurSiON agents;
- both seats;
- broad seeds;
- family-withheld/generalization tests;
- exact CPU/CUDA feature/inference/action parity.

Before any Kaggle submission, the actual candidate must still pass the project's separate official Python-engine 64/128-game correctness/promotion canary.

Large-scale search/data generation belongs in CUDA. Official Python and C++ remain important independent correctness gates.

---

## 16. Explicit non-goals

This proposal does **not** authorize:

- replacing Bonsai-KG with pure/end-to-end RL;
- training a primitive-action policy;
- adding arbitrary model callbacks to BKG3;
- changing BKG3 ABI for the first POC;
- giving a learned advisor raw unrestricted simulator access in production;
- using opponent/team/submission/replay identity;
- using seed, hidden opponent state, future shops, or future RNG;
- training separate autonomous agents for every crop/animal/product;
- accepting floating-point CPU/CUDA divergence;
- training a learned v6 router before proving oracle headroom;
- treating a learned selector as useful if the current expert candidate set has little oracle headroom;
- shipping runtime search merely because search is useful as an offline teacher.

---

## 17. Architectural status summary

### Confirmed current implementation

- BKG3 is stateful and deterministic.
- It has persistent `RuntimeState`, typed blackboard, stateful/reactive composites, lifecycle semantics, temporal controls, and deterministic intent arbitration.
- It is finite and callback-free.
- Portable observation access is legal-feature-restricted.
- Snapshot continuation requires copying `RuntimeState` with simulator state.
- Current stateful-v6 owns threshold-based stage-144/stage-216 route selection and latching, with eleven immutable route experts.
- Current v6 compiled identity records 99 nodes and 48 runtime state slots.

### Proposed POC

- First measure oracle headroom among the existing v6 route candidates at stage 144/216.
- If substantial headroom exists, distill option ranking into a deterministic integer model outside BKG3.
- Keep downstream Bonsai/expert execution unchanged for the controlled experiment.

### Long-term hypothesis

- Expand from one route-ranker to a Bonsai-gated universal option critic over several strategic decision types.
- Preserve Bonsai as router, memory, safety layer, commitment manager, and final executor.

---

## 18. Authoritative repository references

Read these when reviewing or implementing any future POC:

- [`../../RULES.md`](../../RULES.md) — durable research/promotion rules, CUDA-first search, causal lane constraints.
- [`../../README.md`](../../README.md) — Kaggriculture mechanics, 720-turn horizon, shops, market, observations, and win condition.
- [`../../bonsai-kg/SPEC.md`](../../bonsai-kg/SPEC.md) — authoritative BKG3 semantic specification.
- [`../../bonsai-kg/include/bonsai_kg.hpp`](../../bonsai-kg/include/bonsai_kg.hpp) — native host/device evaluator.
- [`../../bonsai-kg/include/bonsai_kg_abi.hpp`](../../bonsai-kg/include/bonsai_kg_abi.hpp) — ABI, limits, `RuntimeState`, features, node/value types.
- [`../../bonsai-kg/integrations/stateful_v6/README.md`](../../bonsai-kg/integrations/stateful_v6/README.md) — current compatibility-controller architecture.
- [`../../bonsai-kg/integrations/stateful_v6/assets/identity.json`](../../bonsai-kg/integrations/stateful_v6/assets/identity.json) — exact controller node/state counts and route-pack identity.
- [`../../cpp-engine/include/sim.hpp`](../../cpp-engine/include/sim.hpp) — exact C++ simulator implementation.
- [`../../cuda-engine/include/sim.cuh`](../../cuda-engine/include/sim.cuh) — exact CUDA simulator implementation.
- [`bonsai_kg_learned_option_value_poc.md`](bonsai_kg_learned_option_value_poc.md) — operational first experiment and go/no-go gate.
