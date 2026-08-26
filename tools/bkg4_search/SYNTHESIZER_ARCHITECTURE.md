# Authoritative Bonsai Synthesizer Architecture

This is the compact navigation guide for the current autonomous specialist-synthesis path. Strategic authority remains `BONSAI_FIRST_PLACE_MASTER_PLAN.md`; this file does not amend or replace it.

## Authoritative flow

```text
bonsai_synthesizer.py
  -> regret_macro_search.py / exact CUDA fragment evidence
  -> exact initiation-state capture (information_state_io.py)
  -> run_cegis() in bonsai_synthesizer.py
       -> macro_initiation_cegis.py
            -> edit_search/goals.py contrastive condition synthesis
       -> exact validation-panel counterexample feedback
  -> BoundedRouteMacro (bounded_macro.py)
  -> SpecialistDeclaration / bounded automatic composition (bonsai_composer.py)
  -> target-capability preservation
  -> frozen sealed holdout
  -> broad CUDA certification
  -> CPU/CUDA parity
```

`run_cegis()` is the one active CEGIS orchestration path. `macro_initiation_cegis.py` owns the reusable legal-state effect model, scoring/evaluation, exact-state ambiguity checks and compact conjunction synthesis. Do not create a parallel route-specific CEGIS implementation for new specialists.

## CORE

- `bonsai_synthesizer.py` — autonomous orchestration, CUDA panel execution, active counterexample loop, dataset roles, target/transfer/holdout contracts, threshold audit and certification flow.
- `macro_initiation_cegis.py` — `MacroEffectCase`, `GateScore`, legal gate evaluation, mixed exact-state detection, contrastive conjunction enumeration and ranking.
- `bounded_macro.py` — deterministic bounded route macro, latch/window semantics, ordinary Bonsai gate compilation and exact bounded preemption representation.
- `bonsai_composer.py` — validated P0.5 fail-closed composer. Do not redesign it as part of gate synthesis.

## REUSABLE_SUPPORT

- `edit_search/goals.py` — contrastive public-state condition generation used by compact CEGIS.
- `information_state_io.py` — exact legal initiation-state capture. Exact signature is registry + Bonsai blackboard + compatibility state; hidden truth is separate diagnostic material, not a gate input.
- `cuda_io.py` — exact CUDA fragment/result binary I/O.
- `module_codec.py` — feature/condition/module codec.
- `regret_macro_search.py` — P0.1 coherent CUDA macro-search result handling; search architecture is outside this CEGIS hardening surface.
- `p0_6_autonomous_synthesizer_v1.json` — declarative autonomous-synthesis configuration and panel roles.
- focused synthesis/composer regression tests.

## Current CEGIS semantics

The engine is bounded but active:

1. Original target effects and transfer effects form initial synthesis evidence.
2. Original target-positive rows are marked required; target L->W rows are separately marked required L->W.
3. A compact conjunction of at most four legal predicates is synthesized from current evidence.
4. The candidate gate is evaluated over the exact CUDA validation/counterexample pool.
5. Newly gate-active harmful or W->L validation rows are appended to CEGIS evidence.
6. Gate synthesis is rerun from the enlarged set.
7. The loop terminates on safe convergence, mixed exact legal state, no compact legal separator, target-capability loss, or deterministic iteration-budget exhaustion.
8. The final gate must retain configured independent positive transfer support.
9. The sealed holdout is checked only after gate freeze and is never silently recycled into CEGIS training evidence.
10. Threshold neighbors are audited before composition.
11. The existing post-composition `TargetCapabilityReport` remains an independent acceptance check.

For a fixed macro/case, exact CUDA forced-macro outcome does not change when the gate is resynthesized; only gate activation changes. The validation panel therefore needs one exact CUDA effect evaluation for the fixed macro, after which active CEGIS can add exact harmful rows without replaying identical games every symbolic iteration.

## Dataset roles

- `target_capability` — original discovery states that justified the macro. At least some discovery-positive support is required; if discovery contained L->W, at least one required L->W must survive.
- `transfer` — independent macro-transfer evidence. The final gate must preserve configured positive/independent support.
- `validation` — exact active-Counterexample source. Gate-active harms can enter CEGIS evidence.
- `sealed_holdout` — untouched frozen-gate check. Failure rejects; it is not training data.
- certification/archive/current-top10 — post-composition broad safety and protected-win regression evidence.

If a future process deliberately tunes on a failed sealed case, it must create a fresh independent holdout. The current synthesizer does not silently reclassify sealed evidence.

## Legal state and identity

Runtime routing uses current legal state only. Exact ambiguity uses the full captured legal signature: registry + BB + compat. If the same exact legal state contains both positive and harmful macro outcomes, CEGIS fails closed because no legal current-state gate can separate them.

Opponent/team/submission/replay/episode/tape/offline-seed identity is forbidden. `OWN_SEED` is legal own private runtime information and remains explicitly distinct from offline replay/episode seed identity.

## Gate grammar and macro state machine

Current autonomous synthesis is Level 1 only: one conjunction, at most four predicates. `BoundedRouteMacro` can compile multiple clauses as a Selector/DNF, but CEGIS does not synthesize DNF yet because current validated evidence does not establish a need for the extra grammar.

The macro has a start gate at a fixed step, a latched fixed-duration active window and deterministic return to parent. There is no synthesized termination predicate, abort path or recovery state machine yet. Add those only after a concrete failure demonstrates the bounded representation is the blocker.

## HISTORICAL_REPRODUCTION

`P0_1_REGRET_GUIDED_MACRO_SEARCH.md` through `P0_6_AUTONOMOUS_SYNTHESIZER.md`, `distill_route435_macro_cegis.py`, `run_route435_initiation_cegis.py`, `distill_route410_macro_cegis.py`, `run_route410_initiation_cegis.py`, `distill_route410_wheat_liquidity_expansion.py`, `distill_route437_wheat_scarcity_sheep_shift.py`, their focused tests, and accepted terminal JSON evidence are historical reproduction fixtures. Keep them; do not use them as the production path for new specialists.

## EXPERIMENTAL

Standalone `analyze_*`, `make_*`, `prepare_*`, `probe_*`, `campaign.py`, `build_league.py`, invariant-search/factory/generator utilities, and the broader `edit_search/` framework other than the `goals.py` dependency are research tooling unless a current plan explicitly calls for them. They are not imported by the P0.6 authoritative synthesis path.

## PAUSED_EDITFORMER

`bonsai_editformer/**` is paused for this lane and must not be confused with the autonomous Bonsai specialist synthesizer. Its local `.venv` is disposable environment state, not source/evidence.

## OBSOLETE

No tracked synthesis-relevant file was proven obsolete strongly enough to delete in the P0.6 CEGIS hardening pass.

## UNKNOWN

No file on the synthesis-critical dependency path remains unclassified. Unrelated pre-existing untracked user work is outside this classification and must not be treated as synthesis clutter.

## Rule for future specialist synthesis

Start new autonomous specialist work at `bonsai_synthesizer.py` with the declarative P0.6/P0.7-style config and exact CUDA evidence. Do not copy a `run_routeXXX_*` or `distill_routeXXX_*` historical script into a new production synthesis path.
