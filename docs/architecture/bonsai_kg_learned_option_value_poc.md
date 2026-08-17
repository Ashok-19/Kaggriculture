# Learned Option-Value POC: Stateful-v6 Route Oracle

## Status

**NOT IMPLEMENTED YET.**

This document defines the first permitted proof-of-concept for the architecture in [`bonsai_kg_learned_option_value.md`](bonsai_kg_learned_option_value.md).

It is intentionally an **oracle-headroom study before any model training**.

No advisor runtime, learned forest, dataset pipeline, CUDA evaluator, BKG3 change, or benchmark result is created by this document.

---

## 1. Purpose

Determine whether **selector representation is actually limiting the current stateful-v6 compatibility controller**.

The question is not:

> Can we train a model that mimics or slightly perturbs the existing selector?

The question is:

> If we had perfect state-dependent choice among the eleven existing v6 route experts at the current decision boundaries, how much could global results improve?

If the answer is "very little", then a learned route selector solves the wrong problem. The candidate route/expert set is the bottleneck and should be improved before any learned ranking work.

---

## 2. Confirmed current baseline

Authoritative files:

- [`../../bonsai-kg/integrations/stateful_v6/README.md`](../../bonsai-kg/integrations/stateful_v6/README.md)
- [`../../bonsai-kg/integrations/stateful_v6/assets/identity.json`](../../bonsai-kg/integrations/stateful_v6/assets/identity.json)
- [`../../bonsai-kg/integrations/stateful_v6/assets/controller.json`](../../bonsai-kg/integrations/stateful_v6/assets/controller.json)

The translated compatibility structure is:

```text
pre-144 route 64
    ->
stage-144 BKG3 threshold decision
    ->
leaf/route latch in BKG3 blackboard
    ->
optional one-shot stage-216 refinement
    ->
immutable route expert/tape execution
```

The route pack is an actuator only; it never chooses the route.

Current route candidate IDs are:

```text
29
64
32
79
46
63
22
34
77
70
23
```

The compiled controller identity records:

```text
controller_nodes       = 99
controller_state_slots = 48
```

This is a clean A/B boundary because downstream expert execution can remain identical while only route selection is studied.

---

## 3. Phase 0 only: oracle headroom

### 3.1 Decision snapshots

Collect/evaluate snapshots at the existing route decision boundaries:

- stage 144;
- stage 216 where the current controller permits/refines a route decision.

The exact legal snapshot extraction and forcing interface must be designed later from source. Do not infer it from this document alone.

### 3.2 Snapshot state that must be cloned

For every branch, copy:

```text
exact simulator snapshot
+
exact Bonsai BKG3 RuntimeState
```

This is mandatory because BKG3 is stateful. Simulator state alone does not preserve latched route/leaf state, node state, blackboard state, temporal state, or other controller continuation semantics.

If any later advisor owns persistent runtime state, that state must also be cloned. The Phase-0 oracle has no learned advisor state.

### 3.3 Candidate forcing

For each legal/current candidate route at the decision point:

1. start from the same simulator + BKG3 state pair;
2. force the candidate route through a controlled experimental mechanism;
3. preserve all unrelated controller/expert behavior;
4. continue the game in the exact CUDA engine;
5. record terminal outcomes and robust statistics.

The forcing mechanism is experimental scaffolding only. It must not mutate frozen BKG3 semantics or the checked-in stateful-v6 integration.

### 3.4 Why CUDA

This experiment requires many same-state continuations across candidates, opponents, seats, and seeds. Per `RULES.md`, exact same-state counterfactual continuation/oracle work belongs in the CUDA engine.

Do not substitute a heavy CPU/Python brute-force implementation.

---

## 4. Required evaluation population

The headroom result must represent global tournament strength, not one selected opponent.

Required future population dimensions:

- current top agents;
- historical top agents;
- emerging strong agents;
- historical ReCurSiON agents;
- both seats;
- broad seeds/economic regimes.

The exact current population must be refreshed when the experiment is actually run. This document intentionally contains no mutable leaderboard snapshot.

Opponent identity is an evaluation grouping only, never a production selector feature.

---

## 5. Outcome statistics

The primitive terminal statistic is:

```text
margin = own_final_money - opponent_final_money
```

For each snapshot/candidate, retain enough information to judge winning and downside rather than mean money alone:

- win/loss outcome;
- terminal margin;
- mean margin over repeated continuations where applicable;
- median margin;
- lower quantile(s);
- CVaR or equivalent downside statistic if enough samples exist.

For the incumbent current selector choice `B(S)`, compute paired candidate advantage where useful:

```text
ADV(S, A) = Value(S, A) - Value(S, B(S))
```

The oracle itself should select candidates under a robust tournament-aligned criterion. The exact scalarization is an experimental design decision; do not default blindly to maximum mean own money.

---

## 6. Core comparison

For every evaluated decision state, compare:

```text
CURRENT
    existing v6 threshold selector

versus

ORACLE
    best candidate from the same existing route set
    according to the defined future-outcome criterion
```

Aggregate this comparison across the global population.

The experiment must answer:

1. How often does the current selector already choose the oracle-best candidate?
2. When it is wrong, how large is the paired loss?
3. Are mistakes concentrated in a small economic regime or broadly distributed?
4. Does the oracle convert actual losses/ties to wins, or only improve already-positive margins?
5. Does oracle improvement survive both seats and family withholding?
6. Are large gains attributable to selector error, or to continuation-policy artifacts after forcing a route?

---

## 7. Go / no-go rule

### NO-GO: small oracle headroom

If perfect candidate-route selection produces only small global improvement:

```text
STOP learned-router work.
```

Interpretation:

```text
existing expert/macro candidate set is the bottleneck,
not selector representation.
```

Next work should improve/create coherent options, alter earlier strategic basins, or expand the legal option library. Do not train a model to solve a selector problem that the oracle says has little value.

### GO: substantial oracle headroom

If the oracle produces a **major, robust global gain** with improved win conversion and acceptable downside across both seats/populations:

```text
proceed to a learned route-value / ranking POC
```

The first learned representation should default to the deterministic integer forest contract described in the primary architecture document.

"Substantial" is deliberately not assigned a speculative numeric threshold here. The actual gate should be chosen from tournament-relevant evidence when Phase 0 is executed.

---

## 8. What follows only after a GO result

A later learned POC may then:

1. construct legal serving features from BKG3-observable state plus relevant Bonsai blackboard state;
2. generate paired/listwise route-value labels from exact counterfactual evaluation;
3. split data by seed/opponent family/regime with explicit holdouts;
4. train a compact integer-compatible route ranker;
5. verify post-quantization ranking stability;
6. implement exact C++/CUDA deterministic inference outside BKG3;
7. keep downstream route execution unchanged;
8. compare learned selection against both incumbent and oracle ceilings;
9. run family-withheld/generalization tests;
10. only after correctness and global strength gates, consider broader option-value decisions.

None of those steps are authorized by the present documentation-only task.

---

## 9. Evidence required before model implementation

Before implementing any learned route scorer, Phase 0 should provide evidence for all of the following:

### 9.1 Headroom exists

The oracle among the existing candidates materially improves global tournament outcomes.

### 9.2 Headroom is selector-addressable

The gain comes from choosing a different existing route at the current legal decision boundary, not from hidden future information or changes to downstream behavior.

### 9.3 Features can plausibly distinguish useful choices

States with different oracle choices should differ in legal causal state or relevant Bonsai memory. If conflicting futures are identical under every legal serving feature, the correct problem is robust/minimax option value, not identity classification.

### 9.4 Gain generalizes

Headroom should not be isolated to one opponent family, one seat, or one narrow seed block.

### 9.5 Downside is acceptable

Oracle improvement should not come from sacrificing catastrophic tails for mean margin.

### 9.6 Continuation semantics are valid

The branch evaluator must continue from exact simulator + BKG3 state and preserve intended route/macro semantics.

---

## 10. Leakage and generalization controls

Phase 0 and any later learned POC must keep a strict boundary between:

### Allowed serving information

- legal current BKG3 observation features;
- relevant Bonsai-owned persistent state;
- deterministic decision context/candidate id.

### Label-generation-only information

- terminal outcome of a counterfactual continuation;
- future events observed only after branching.

### Never valid as serving features

- opponent/team identity;
- submission/replay/episode id;
- seed;
- future shops;
- future RNG;
- hidden/private opponent state;
- replay fingerprints designed to recover identity.

Recommended later validation includes leave-family-out tests and fresh-population evaluation.

---

## 11. Risks specific to the v6 oracle

### 11.1 Candidate library may already dominate selector error

The eleven fixed routes may simply lack the continuation needed for unresolved regimes. This is exactly what the oracle gate is designed to detect.

### 11.2 Route forcing can invalidate controller state if done incorrectly

Changing only a route id without preserving the corresponding BKG3 latch/leaf/refinement semantics could create an impossible state.

**Requirement:** design forcing from the authoritative v6 controller/runtime source and verify that each counterfactual is a valid controlled continuation.

### 11.3 Stage-216 semantics are conditional

The current controller performs an optional one-shot refinement, not an unconstrained second independent strategy reset.

**Requirement:** preserve current eligibility/decision semantics when defining the candidate set and oracle comparison.

### 11.4 Continuation-policy bias

A route can look weak because the fixed downstream expert is weak in a state it was not designed for.

**Requirement:** interpret oracle value as value of the **existing route expert under the existing downstream execution**, not an abstract intrinsic value of a route id.

### 11.5 Oracle labels can be unlearnable from legal current state

Different future shop/RNG realizations can make different options best even from identical current legal observations.

**Requirement:** use robust expected/minimax/distributional option value over admissible future uncertainty rather than leaking future realization into serving features.

---

## 12. Correctness requirements for the eventual evaluator

When Phase 0 is later implemented, it must establish at least:

- exact simulator snapshot clone semantics;
- exact BKG3 `RuntimeState` clone semantics;
- no mutation of frozen checked-in v6 assets;
- both-seat correctness;
- deterministic candidate forcing;
- exact route identity and execution continuity;
- reproducible outcome aggregation;
- no hidden serving features.

Any future learned inference then adds a separate exact C++/CUDA parity gate.

---

## 13. Non-goals of this POC

Phase 0 is **not**:

- model training;
- an end-to-end RL experiment;
- primitive-action MCTS as the runtime policy;
- a BKG3 ABI extension;
- a new Bonsai opcode;
- a new route/expert library;
- a leaderboard-specific classifier;
- a justification to use team/submission identity;
- a benchmark of speculative model architectures.

It is one controlled question:

> **How much value is available from perfect selection among the eleven existing stateful-v6 route experts at the existing stage-144/stage-216 decision boundaries?**

---

## 14. Decision tree for the future session

```text
Read authoritative BKG3 + stateful-v6 source
              |
              v
Implement exact CUDA oracle harness
without modifying frozen BKG3/v6 assets
              |
              v
Clone Sim + RuntimeState at 144/216
              |
              v
Evaluate all valid existing route candidates
across broad population / seats / seeds
              |
              v
Compare incumbent selector vs perfect-route oracle
              |
       +------+------+
       |             |
small headroom   major headroom
       |             |
       v             v
STOP selector    design integer
learning         learned ranker POC
       |
       v
improve option/expert library
or intervene earlier
```

---

## 15. References

- [`bonsai_kg_learned_option_value.md`](bonsai_kg_learned_option_value.md) — primary architecture and long-term direction.
- [`../../RULES.md`](../../RULES.md) — CUDA-first research, causal lanes, promotion constraints.
- [`../../README.md`](../../README.md) — Kaggriculture mechanics and legal observation context.
- [`../../bonsai-kg/SPEC.md`](../../bonsai-kg/SPEC.md) — BKG3 state/snapshot/privacy semantics.
- [`../../bonsai-kg/include/bonsai_kg_abi.hpp`](../../bonsai-kg/include/bonsai_kg_abi.hpp) — `RuntimeState` and hard limits.
- [`../../bonsai-kg/integrations/stateful_v6/README.md`](../../bonsai-kg/integrations/stateful_v6/README.md) — compatibility-controller architecture.
- [`../../bonsai-kg/integrations/stateful_v6/assets/identity.json`](../../bonsai-kg/integrations/stateful_v6/assets/identity.json) — exact route set and controller identity.
- [`../../cpp-engine/include/sim.hpp`](../../cpp-engine/include/sim.hpp) — exact C++ simulator.
- [`../../cuda-engine/include/sim.cuh`](../../cuda-engine/include/sim.cuh) — exact CUDA simulator.
