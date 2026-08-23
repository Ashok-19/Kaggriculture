# Invariant Strategic Edit Generator Design

## Objective

Replace the trajectory-brittle scalar route-selector grammar as the primary permanent BKG4 edit generator with a bounded, canonical, cross-trajectory strategic module representation. Legacy `ModuleGenome` remains available as a regression baseline and cheap local emitter.

## Constraints

- Preserve BKG4 schema 4 and ABI 4.
- Keep the large current BKG4 parent frozen; mutate only a small semantic module.
- Use legal current observations only. Never encode tape, seed, team, opponent identity, replay identity, hidden state, RNG state, or future information.
- Large game evaluation is exact resident CUDA. CPU is for bounded tests, compilation, proposal/solver work, and parity.
- Certification is label-only and cannot guide search.
- Every search construct must have Python reference semantics, CPU C++ semantics, CUDA semantics, and a mechanical ordinary-BKG4 distillation.
- No champion promotion, Kaggle submission, or ranker training in this task.

## Root cause

`ModuleGenome` is a 420-byte `B4GM1` route selector: at most eight rules, four scalar conditions per rule, then an existing route ID. Its easiest optimization path is a trajectory-local threshold or time patch. The underlying BKG4 VM already supports stateful control, blackboard binding, direct unit/market intents, recovery, hysteresis and 65k nodes, so VM capacity is not the bottleneck. The temporal-information-state campaign also showed that more selector state does not repair missing action capabilities.

## Representation

Introduce `InvariantStrategicModule` with explicit grammar identity `bkg4_invariant_strategic_module_v1` and compact resident ABI `B4IM1/v1`. A module is a bounded sequence of semantic rules. Each rule has:

1. a `ModuleSketch` (`domain`, `control_pattern`, `effect_class`),
2. a lifted guard over legal observation facts,
3. an abstract action macro,
4. bounded constants/state parameters.

Initial lifted domains are `GLOBAL` and `EACH_OWN_UNIT`. `EACH_OWN_UNIT` binds a unit and its current tile, so one rule represents all equivalent concrete units. The initial action families are:

- `ROUTE_OVERRIDE`: existing route as a basic/legacy action family;
- `UNIT_REWRITE`: suppress or replace a matching planned unit operation for all bound units satisfying the lifted guard;
- `MARKET_INTENT`: guarded bounded market order emitted through ordinary BKG4 intent semantics;
- `COMMITTED_ROUTE`: stateful guarded route mode with an invariant exit condition.

The first implementation uses feature-registry v1. Per-unit/tile relational tests compile by finite static expansion using `OWN_UNITS`, `OWN_WORKER_TILE`, blackboard tile bindings and existing tile features. No registry-v2 aggregate is introduced until a measured cross-tape failure requires one.

## Direct-action integration

Current champion integration intentionally ignores BKG4 `ActionResult4` and materializes the route pack. For invariant modules the smallest compatible integration is:

1. tick BKG4 and retain `ActionResult4`;
2. materialize the selected route exactly as today;
3. apply frozen legacy and engine-exploit overlays exactly as today;
4. apply explicitly authorized unit rewrites from the invariant module;
5. merge direct BKG4 unit claims only where they are non-PASS;
6. prepend bounded BKG4 market claims, respecting the engine market-order limit;
7. preserve existing feed reconciliation.

With an empty/no-op invariant module, the action result contains only PASS/no market claims and the final action trace is exactly unchanged.

## Search and generalization

Search is skeleton-first and constant-second. Structural mutations choose domain, control pattern, lifted predicates and action family. Constants are solved from discovery-only, tape-grouped evidence using observed boundaries/intervals and broad-margin preference rather than arbitrary int32 brute force.

Discovery cases are grouped by trajectory/tape. Production specialists require multi-tape support; singleton mechanisms remain exploration/CEGIS seeds. Fitness is lexicographic over group safety, useful-group count, worst-group delta, lower quantile, median and mean. Discovery-only rotating tape folds provide internal cross-fit regularization. Final certification remains frozen and label-only.

CEGIS counterexamples carry a failure class (`guard_too_broad`, `wrong_action_effect`, `unstable_constant`, `resource_invariant`, `action_conflict`, `lifecycle_missing`) and repairs alter semantic skeletons rather than adding tape-local equality patches. Anti-patch validation rejects arbitrary exact STEP/high-cardinality equalities and fixed unit/tile identities for production unless mechanically justified and multi-tape supported.

## Corpus and BonsaiEditFormer boundary

Every invariant edit serializes the canonical semantic AST: sketch, domain, guard, action macro, parameters/state, source parent and child semantic ID. Outcome fields (discovery group support, cross-fit, certification) remain labels/provenance and are not candidate input. The ranker receives canonical module-AST tokens, not tape/group/seed/opponent identifiers.

## Acceptance

The architecture is only `READY` after repeated tape-disjoint useful transfer. The final validation compares random typed legacy, fixed systematic legacy, current permanent ModuleGenome search and the invariant generator at equal exact-CUDA budget across five grouped splits when the case bank permits. Empty-module, tri-engine, and BKG4-distillation parity require zero mismatch.
