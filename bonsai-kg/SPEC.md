# Bonsai-KG Semantic Specification

Status: core specification, schema version 2, compiled ABI version 2.

Bonsai-KG is the Kaggriculture-native behavior-tree policy VM used by future ReCurSiON challenger agents. It is intentionally separate from the frozen champion lineage and from the upstream `bonsai/` reference implementation.

## 1. Design boundary

Bonsai-KG is a compiled, deterministic, turn-based behavior-tree VM. One Kaggriculture agent call is one Bonsai-KG tick. A declarative Policy is immutable and may be shared by many simulations. Each simulation owns independent RuntimeState.

The VM is not a generic robotics BT framework. It has no dynamic plugins, arbitrary callbacks, wall-clock timers, strings in the hot loop, reflection, runtime allocation, or hidden simulator access.

The legal native observation boundary is the existing CUDA `ObservationView`: public game state for both players plus only the acting player's private shed, seeds, and worker inventories. The C++ and Python adapters expose the same registry. Policy code must never receive simulator state, RNG state, replay ids, team/opponent identity, future shop values, or opponent-private values.

## 2. Evidence-derived upstream semantics

Sollimann/bonsai is the conceptual reference for declarative Behavior vs transient State, depth-first left-to-right traversal, three leaf statuses, stateful Sequence/Select, and `memory(false)` reactive composites.

BehaviorTree.CPP is used only to resolve missing lifecycle/typed-data concepts. In particular, its current source establishes that `_failureIf`, `_successIf`, and `_skipIf` are checked only when a node is idle/skipped, `_while` is also checked while running and can halt a node, and lifecycle scripts distinguish success, failure, halt, and generic terminal postconditions. Its Parallel node ticks children deterministically in order and remembers completed children.

Bonsai-KG adopts the useful semantics while replacing scripts/callbacks with finite predicates and finite typed-blackboard effects.

## 3. Status model

Leaf/task contract:

- `SUCCESS`
- `FAILURE`
- `RUNNING`

The native VM also has an internal `SKIPPED` control result because generic skip semantics cannot be represented compositionally with only the three leaf statuses. `SKIPPED` is not a legal task return and is produced only by lifecycle preconditions. Sequence and Selector both ignore skipped children. A skipped root is normalized to external `SUCCESS` and emits no action by itself.

No user-authored node may return IDLE.

## 4. Tree and execution model

Policies are structurally acyclic trees. Children are stored as compact indices in a contiguous child-reference table. Traversal is deterministic depth-first, left-to-right.

All runtime policy arithmetic is integer arithmetic. Feature values used by policy comparisons are signed 32-bit integers. Kaggriculture money is stored as `double` by the exact native engines but every competition-baseline mutation is integral; the policy feature registry exposes money only after a checked/in-spec integral conversion to `int32`. Python uses the same conversion. No policy decision depends on floating-point tolerance or fast math.

Every tick has an instruction/node-visit budget stored in the compiled policy. Budget exhaustion is a deterministic VM fault: the root result is `FAILURE`, action output is PASS/empty market, and the fault flag is set. A nonzero fault is fatal and sticky: later ticks return the same fault and empty action without evaluating the tree until `reset(policy)` is called. Valid compiled policies are additionally bounded by schema limits and one-body-evaluation-per-turn loop semantics.

The portable contract requires `boardSize == 10`. TILE values use the fixed row-major encoding `y * 10 + x`; loaders require the compiled board-size field to be 10, Python observations require two 10 by 10 public tile grids, and the native adapters statically bind to the engines' 10 by 10 board. A differently sized game must be rejected rather than truncated or reinterpreted.

## 5. Stateful and reactive composites

### Sequence

Stateful. Resume at the currently RUNNING child. `SUCCESS` advances to the next child in the same tick. `FAILURE` immediately fails and resets the cursor. `RUNNING` stores the cursor. Skipped children are ignored. All children success => `SUCCESS`, cursor reset.

### Selector

Stateful priority selector. Resume at the currently RUNNING child. `FAILURE` advances in the same tick. `SUCCESS` immediately succeeds and resets the cursor. `RUNNING` stores the cursor. Skipped children are ignored. All children fail/skip => `FAILURE`, cursor reset.

### ReactiveSequence

Memoryless with explicit preemption tracking. Starts at child zero every tick. Success/skip advances; failure fails. Running stores that child as the active branch. If a previously RUNNING later child is no longer the active child because an earlier guard/branch changes outcome, the previous subtree is halted deterministically before the parent returns.

### ReactiveSelector

Memoryless priority selector. Starts at child zero every tick. Failure/skip advances; success succeeds. Running stores that child as the active branch. When a different higher-priority child takes over, the previous RUNNING subtree is halted deterministically before the parent returns.

Reactive preemption cleanup is deterministic but, like common BT implementations, selection of the new branch necessarily occurs by ticking it; cleanup of the previously running branch occurs immediately once takeover is known. Policies that require a reservation to be released before evaluating an emergency action should place that reservation validity in an explicit guard/blackboard predicate rather than rely on unspecified ordering.

### If

Exactly three children: condition, then, else. The condition is evaluated until it returns terminal. Once it selects then/else, that branch is latched while RUNNING and the condition is not re-evaluated. Terminal branch result resets the latch. A skipped condition yields internal `SKIPPED`.

### Parallel

Threshold Parallel is VM-supported but disabled in search grammar v2.

Parameters: `success_threshold` and `failure_threshold`, each in `[1, child_count]`, child count <= 16.

Children are ticked in ascending child order; there is no OS/GPU concurrency. SUCCESS and FAILURE children are remembered for the current Parallel activation. Threshold success/failure and success impossibility are checked after each child, with children skipped on the current tick excluded from the remaining possible successes. On any terminal result, any RUNNING siblings are halted and all Parallel state is reset. If no SUCCESS/FAILURE child has been remembered and all children skip on the current activation, Parallel returns terminal internal `SKIPPED` and resets. If terminal children were remembered but the remaining children all skip and no threshold can be reached, the activation deterministically returns `FAILURE` and resets. Parallel returns `RUNNING` only when at least one unresolved child actually returned `RUNNING` on the current tick and success remains possible.

## 6. Decorators and turn controls

### Invert

SUCCESS <-> FAILURE. RUNNING and SKIPPED unchanged.

### ForceSuccess / ForceFailure

RUNNING and SKIPPED unchanged. Any terminal non-skipped child result maps to the forced result.

### TurnTimeout(N)

N >= 1. First child tick records the current game step. Before each later child tick, if `current_step - start_step >= N`, the child is halted and the decorator returns FAILURE. Child terminal result clears timeout state. No wall-clock time is used.

### Retry(N)

N >= 1. At most one child attempt is evaluated per game turn. Child SUCCESS => SUCCESS and counter reset. Child RUNNING => RUNNING. Child FAILURE increments attempts; the Nth failure => FAILURE, otherwise child is reset and Retry returns RUNNING until the next game turn. SKIPPED propagates without consuming an attempt.

### Repeat(N)

N >= 1. At most one completed body cycle is counted per game turn. Child FAILURE => FAILURE and reset. Child RUNNING => RUNNING. Child SUCCESS increments completed cycles; the Nth success => SUCCESS, otherwise child is reset and Repeat returns RUNNING until the next game turn. SKIPPED propagates without consuming a cycle.

### KeepRunningUntilFailure

At most one child evaluation per game turn. Child FAILURE => SUCCESS. Child RUNNING => RUNNING. Child SUCCESS resets the child and returns RUNNING so the next cycle begins next game turn. SKIPPED propagates.

### WaitTurns(N)

Stateful leaf used for temporal commitments/tests. First activation stores current step and returns RUNNING. It returns SUCCESS once `current_step - start_step >= N`. N >= 1.

Explicit `While`/`WhileAll` nodes are deliberately not in ABI v2. Reactive composites plus lifecycle `_while` cover the required causal re-evaluation without introducing zero-time loop ambiguity. The compiled ABI reserves node ids for future additive extensions.

## 7. Lifecycle preconditions and effects

Any node may have a lifecycle descriptor. Lifecycle nodes receive a runtime slot even if their core node is otherwise stateless so the VM can distinguish inactive from RUNNING.

Precondition order is fixed:

1. `failure_if`
2. `success_if`
3. `skip_if`
4. `while`

`failure_if`, `success_if`, and `skip_if` are evaluated only when the node is not already RUNNING. `while` is evaluated on initial entry and while RUNNING. A false initial `while` produces internal SKIPPED. A false `while` on a RUNNING node halts its active subtree, executes halt cleanup, then returns SKIPPED.

Lifecycle predicates are finite, statically typed comparisons over ValueRefs; no arbitrary code is allowed.

Terminal hooks execute in this order:

- SUCCESS: `on_success`, then `post`
- FAILURE: `on_failure`, then `post`
- halt/preemption: `on_halted` only
- SKIPPED: no post hook

Effects are finite typed-blackboard operations: `set`, `clear`, `add`. `set` and `add` consume ValueRefs; `clear` restores the slot's declared initial value. The compiler rejects slot/type misuse and arithmetic on non-I32 slots.

## 8. Typed blackboard

Authoring-level slots have names and static types. Compilation resolves names to integer indices. Runtime has no hashes or strings.

ABI v2 blackboard types:

- `BOOL` (0/1 only)
- `I32`
- `U16` (0..65535)
- `ENUM8` (0..255)
- `TILE` (`NONE=-1` or 0..99)
- `UNIT` (`NONE=-1` or 0..39)

Runtime storage is one 32-bit word per slot; type metadata is immutable policy data. Maximum v2 slots: 32.

### ValueRef

Every runtime-readable value is compiled to a statically typed ValueRef. The portable ValueRef types are `I32`, `TILE`, and `UNIT`; BOOL, U16, and ENUM8 blackboard slots are read through the I32 type while their declaration constraints still govern writes. ValueRef sources are exactly:

- `CONST`: an inline signed 32-bit constant, with an explicit or context-inferred type;
- `BB`: a resolved blackboard slot, whose effective read type must match the context;
- `LEGAL_FEATURE`: an id and validated constant argument from the legal feature registry.

Authoring shorthands such as an integer condition RHS remain valid and normalize to a typed CONST ValueRef. TILE and UNIT constants and slots accept the `-1` NONE sentinel. There are no implicit TILE/UNIT/I32 conversions. Comparisons require equal effective types. All unused ValueRef fields are zero in canonical compiled IR.

Lifetimes are distinct:

- blackboard: episode-persistent and reset only on explicit episode reset;
- node execution state: episode-persistent transient control state, reset/halted by BT lifecycle;
- deferred intent journal and arbiter scratch: tick-local Evaluator storage, never part of persistent RuntimeState, discarded after every root tick.

## 9. Feature registry and legality

Feature ids are fixed by ABI v2 and implemented in all adapters. Parametric features validate their argument range.

Core registry (all values are I32 unless marked otherwise):

- ALWAYS
- STEP, DAY, HOUR
- OWN_MONEY, OPP_MONEY
- OWN_UNITS, OPP_UNITS
- OWN_LAND, OPP_LAND
- OWN_HIRES_TODAY, OPP_HIRES_TODAY
- OWN_QUADRANT_UNLOCKED(quadrant), OPP_QUADRANT_UNLOCKED(quadrant)
- MARKET_PRICE(product), MARKET_INVENTORY(product)
- OWN_SHED(item), OWN_SHED_TOTAL
- OWN_SEED(crop)
- OWN_INVENTORY(packed unit/item)
- SHOP_COUNT, SHOP_AT(index), SHOP_SEEN(shop)
- OWN_WORKER_X/Y(unit), OPP_WORKER_X/Y(unit)
- OWN_WORKER_TILE(unit), OPP_WORKER_TILE(unit) (`TILE`, NONE for a missing unit)
- OWN_TILE_KIND(tile), OPP_TILE_KIND(tile)
- OWN_TILE_ITEM(tile), OPP_TILE_ITEM(tile)
- OWN_TILE_PLANTED_DAY(tile), OPP_TILE_PLANTED_DAY(tile)
- OWN_TILE_YIELD(tile), OPP_TILE_YIELD(tile)
- OWN_TILE_WATERED(tile), OPP_TILE_WATERED(tile)
- OWN_TILE_CONSECUTIVE_UNWATERED(tile), OPP_TILE_CONSECUTIVE_UNWATERED(tile)
- OWN_TILE_MAX_LIFESPAN_STEP(tile), OPP_TILE_MAX_LIFESPAN_STEP(tile)
- OWN_TILE_FERTILIZED_UNTIL_DAY(tile), OPP_TILE_FERTILIZED_UNTIL_DAY(tile)
- OWN_TILE_PLACED_DAY(tile), OPP_TILE_PLACED_DAY(tile)
- OWN_TILE_FED(tile), OPP_TILE_FED(tile)
- OWN_TILE_CONSECUTIVE_UNFED(tile), OPP_TILE_CONSECUTIVE_UNFED(tile)
- OWN_TILE_CARED(tile), OPP_TILE_CARED(tile)
- OWN_TILE_FERT_AVAILABLE(tile), OPP_TILE_FERT_AVAILABLE(tile)
- OWN_TILE_PENDING_CARE_BONUS(tile), OPP_TILE_PENDING_CARE_BONUS(tile)
- OWN_TILE_HAS_ANIMAL(tile), OPP_TILE_HAS_ANIMAL(tile)

`OWN_INVENTORY` encodes arg as `(unit << 8) | item`.

Quadrant arguments are `0=NW`, `1=NE`, `2=SW`, `3=SE`. Tile-kind/item encodings and absent-field defaults are fixed by the ABI and identical in the official, C++, and CUDA adapters. `HAS_ANIMAL` makes an empty COOP/PASTURE distinguishable from an occupied one even when `TILE_ITEM`'s empty default is zero.

Field-by-field legality against the repository `README.md` Observation Format is explicit:

- top-level `day` and `hour` are direct features; `step` is the engine/official turn counter used by the environment and temporal nodes; `player` is not exposed as a policy feature because own/opponent addressing is already normalized by the adapter;
- each public farm's `money`, `hires_today`, worker positions, and unlocked-quadrant state are exposed directly or by deterministic derived features (`UNITS`, `LAND`, `WORKER_TILE`, `QUADRANT_UNLOCKED`);
- every documented public tile discriminator/value/lifecycle field is represented: kind, crop/animal item, occupancy, planted/placed day, watered/fed/cared flags, consecutive-unwatered/unfed, yield, max lifespan step, fertilized-until day, fertilizer availability, and pending care bonus;
- shared market inventory/prices and town unlocked-shop multiplicity/order are exposed;
- acting-player private `shed`, `seeds`, and per-worker inventories are exposed; opponent private values are deliberately absent;
- RNG seed/state, replay/submission/team/opponent identity, future observations/shops, and native-only caches/statistics such as decay queues, produced/discarded/sold counters, revenue and spend are excluded.

No registry entry can address opponent shed/seeds/inventory or hidden simulator state.

## 10. Leaf/task primitives

ABI v2 portable leaves:

- `Success`
- `Failure`
- `Condition`
- `SetBlackboard`
- `AddBlackboard`
- `EmitUnitIntent`
- `EmitMarketIntent`
- `WaitTurns`
- `MoveUnitTo`

`Condition` compares two same-typed ValueRefs using EQ, NE, LT, LE, GT, GE. `SetBlackboard`, `AddBlackboard`, lifecycle `set`/`add` effects, and `MoveUnitTo` operands consume typed ValueRefs. Instant `EmitUnitIntent` and `EmitMarketIntent` retain canonical compile-time unit/op/item/quantity/priority operands in ABI v2; dynamic action operands are intentionally deferred until evidence shows they are strategically necessary.

`MoveUnitTo(unit,target,priority)` is a representative true multi-turn spatial task. `unit` is a UNIT ValueRef and `target` is a TILE ValueRef, so both can change between ticks through blackboard or legal-feature reads. NONE or a currently missing unit/target fails without emission. At the target it succeeds. Otherwise it emits exactly one Manhattan move intent (X direction before Y direction for deterministic tie behavior) and returns RUNNING. The legacy authoring shorthand `unit` integer plus `x`/`y` integers normalizes to UNIT/TILE CONST references and retains its valid behavior. Higher-level strategic tasks such as expansion/crop/animal cycles are expected to be authored as reusable subtrees; they are not hard-coded into the VM.

Native C++ and CUDA execute the same shared interpreter implementation, so a new portable primitive changes one native code path plus the independent Python oracle, not separate C++ and CUDA semantics.

## 11. Intent and action arbitration

Every tick starts with an empty deferred IntentBuffer with raw capacity 128. Leaves append candidates; no action arbitration or materialization occurs during traversal. Each node evaluation records an intent checkpoint. A node returning `FAILURE` or internal `SKIPPED` rolls the buffer back to its entry checkpoint, so failed selector branches, failed sequence prefixes, skipped lifecycle branches, and skipped Parallel activations cannot leak actions. SUCCESS and RUNNING commit their descendants' candidates to the parent transaction. Raw-capacity exhaustion is a sticky fatal VM fault.

After traversal, a root `FAILURE`, a raw root `SKIPPED`, or any VM fault discards the complete buffer. Only a root SUCCESS or RUNNING is arbitrated. Root SKIPPED is then normalized to external SUCCESS with PASS/empty market output. Intents remain tick-local and are never carried across ticks.

Unit channels are indexed exactly like the native engine (`0` farmer, `1..` hired hands). A unit channel accepts one winning intent. Winner order is:

1. higher explicit priority (0..255),
2. lower deterministic emission sequence number (DFS order).

Market intents are ordered only in the post-traversal arbitration pass: descending priority, then ascending emission sequence. Final action emits the first `min(environment_max_orders, 16)` entries. Thus environment-significant market order ordering is explicit and portable.

A subtree may emit multiple non-conflicting unit and market intents in one tick, including through Parallel. There are no racing writes and Parallel is not physical concurrency.

A constant unit index outside the ABI capacity is rejected at compile time. If an in-capacity `EmitUnitIntent` references a unit that does not exist in the current own observation, the leaf emits nothing and returns `FAILURE`, allowing normal BT fallback. PICKUP/PLACE unit quantities are `1..32767`; unit actions whose quantity is semantically ignored require canonical quantity `1`. Market quantities are `1..65535`, with HIRE/BUY_LAND canonically fixed to quantity `1`. Validation rejects invalid constant op/item/quantity forms. Finalization sets action unit count from current own public unit count, fills unspecified units with PASS, and never exceeds native fixed capacities.

## 12. Serialization and compiled ABI

Human/interchange format is strict JSON, schema version 2. Authoring-only names, formatting, subtree definitions, and unused definitions do not identify executable semantics. Compilation first expands reusable subtrees, resolves bindings and names, lowers shorthands to ValueRefs, assigns dense tables, and zeroes every unused field.

Compiled format `.bkgc` is little-endian ABI version 2:

- fixed 64-byte header with magic `BKG2`, ABI version, semantic SHA-256, root id, table counts, state-slot count, instruction budget, ValueRef count, required board size, and canonical static `max_intents` bound;
- 32-byte Node records;
- uint16 child-reference table;
- 8-byte blackboard declarations;
- 12-byte ValueRef records;
- 12-byte Predicate records containing two ValueRef indices, a comparison, and zero reserved bytes;
- 24-byte Lifecycle records;
- 8-byte Effect records; SET/ADD store a ValueRef table index in the signed 32-bit value field, while CLEAR requires canonical zero.

The header hash is SHA-256 over an explicit normalized little-endian semantic stream. The stream commits the ABI/domain prefix, board size, root, instruction/state counts, every table count, canonical `max_intents`, and every executable field in table order. `max_intents` is exactly the number of potentially emitting runtime nodes (`EmitUnitIntent`, `EmitMarketIntent`, `MoveUnitTo`) in the strict compiled tree; loaders recompute and reject any different value. It excludes the stored hash, source/slot/subtree/node names, debug ids, and all padding/reserved bytes. Both Python and host C++ loaders reconstruct that stream from parsed records, recompute the digest, and reject a mismatch before returning a PolicyView. CUDA receives only a host-validated image.

Strings and JSON are absent from the runtime hot loop. PolicyView is immutable pointers/counts over those tables.

Node records contain compact type/flags, child range, optional state slot, optional lifecycle slot, an explicit reserved word, four integer parameters, and a debug id. Stateless nodes without lifecycle consume no node-state slot. Every record type has an explicit allowed-field mask: unused flags, parameters, reserved words, and padding must be zero; childless nodes use `first_child=0`. Ranges must use the unique canonical empty encoding `(start=0,count=0)`. Debug ids are excluded from the semantic hash but canonical compiled files require `debug_id == node_index`; authoring node names therefore cannot perturb executable identity.

## 13. Reusable authoring subtrees

Schema v2 permits lightweight top-level reusable subtree definitions. A definition has a unique name, a finite map of typed `I32`/`TILE`/`UNIT` parameters, and one tree body. A `Subtree` authoring node names a definition and supplies exactly one same-typed ValueRef binding for each parameter. The body may refer to those values with `{"binding":"name"}` anywhere a ValueRef is accepted; binding a caller blackboard slot is the supported typed remapping mechanism.

Subtrees are authoring sugar only. The compiler expands each use into independent nodes before enforcing the ordinary node/depth/state/table limits, and compiled IR remains a strict single-parent tree with no calls or shared state. Unknown/missing/extra/type-mismatched bindings are errors. Direct or indirect recursion is rejected even if the recursive definition is otherwise unused; unused nonrecursive definitions do not affect executable bytes or the semantic hash.

## 14. Runtime reset

`reset(policy)` is the only episode-start/reset primitive. It zeros node execution state and fault/counters and initializes blackboard slots from declarations. The deferred intent journal is tick-local Evaluator scratch and therefore does not survive a tick at all. No blackboard or node state may survive reset.

Python agent integration must call reset at a new episode and on observed step rollback/restart. Native batch policy runners reset once after simulator initialization. Snapshot continuation research must snapshot/copy Bonsai-KG RuntimeState alongside game state if continuation of a stateful prefix is required; resetting from a game snapshot intentionally means starting a fresh policy state.

Day boundaries do not implicitly reset the Bonsai-KG blackboard or node state. Policies use observable DAY/HOUR/STEP if day-specific cleanup is desired.

## 15. Limits and search grammar separation

ABI v2 hard limits:

- nodes: 128
- depth: 24
- child references: 256
- blackboard slots: 32
- stateful/lifecycle runtime slots: 48
- lifecycle predicates: 128
- lifecycle descriptors: 64
- effects: 128
- ValueRefs: 256
- Parallel children: 16
- Retry/Repeat N: 1..255
- TurnTimeout/WaitTurns: 1..720
- market intent capacity: 16
- raw deferred intent capacity: 128 compact 8-byte records; strict 128-node trees can reach at most 127 emitting leaves
- unit channels: 40
- static `max_intents`: canonical executable bound in `0..128`; `0` selects the journal/arbitration-free native execution path

The VM capability is larger than search grammar v2. Search grammar v2 initially enables Sequence, Selector, ReactiveSequence, ReactiveSelector, If, Invert, ForceSuccess, ForceFailure, Condition, SetBlackboard, AddBlackboard, EmitUnitIntent, EmitMarketIntent, WaitTurns, MoveUnitTo, and expanded Subtree seeds. It keeps Parallel, Retry, Repeat, KeepRunningUntilFailure, and TurnTimeout available to hand-authored/seeded policies but disabled for blind initial mutation. This can be relaxed by evidence without an ABI redesign.

## 16. Validator fail-closed rules

Compilation rejects at least:

- wrong schema version;
- missing/invalid root tree;
- unknown node/feature/blackboard/effect type;
- empty composites where semantics would be ambiguous;
- wrong decorator/If child cardinality;
- depth/node/child/state/table bounds;
- invalid predicate source, comparison, argument, or blackboard type;
- invalid ValueRef kind/type/source, type mismatch, NONE/range misuse, or nonzero unused field;
- invalid lifecycle/effect slot/type;
- invalid Retry/Repeat/timeout/wait threshold;
- invalid Parallel thresholds/count;
- invalid action opcode/item/quantity/priority/unit/tile constant;
- any feature not in the explicit legal registry;
- nonzero header/record padding, reserved data, or node fields outside the type's allowed mask;
- semantic hash mismatch;
- compiled board size other than 10;
- authoring subtree recursion or invalid typed binding/remapping;
- any node not marked portable in ABI v2.

Malformed compiled files are rejected by host loaders before execution. CUDA receives only validated compiled tables.

## 17. Debug/introspection

Debug tracing is opt-in. Python can emit per-node events containing step, debug id, node type, result, and halt events. The shared native VM accepts an optional bounded TraceBuffer; a null pointer is the production CUDA path. Trace overflow truncates trace data only and does not alter policy semantics.

Activation/result/halt counts and RUNNING duration are derived from traces for small runs. No per-node production counter arrays are required in GPU RuntimeState.

## 18. Reference implementation and verification entry points

The implementation is isolated under `bonsai-kg/`:

- `python/bonsai_kg.py`: authoring validator/compiler, canonical hashing, binary load/save, independent reference VM, decompiler, official-agent wrapper.
- `include/bonsai_kg_abi.hpp`: compact portable POD ABI and runtime-state declarations.
- `include/bonsai_kg.hpp`: shared host/device fixed-stack native interpreter.
- `include/bonsai_kg_io.hpp`: strict host `.bkgc` loader/validator.
- `cpp/kag_adapter.hpp`: exact C++ legal feature/action adapter.
- `cuda/kag_adapter.cuh`: exact CUDA `ObservationView` legal feature/action adapter.
- `cuda/policy.cuh`: production `policy_runner.cuh` protocol adapter.
- `tests/`: golden semantics, randomized differential tests, malformed-binary tests, real feature-registry parity, official/C++/CUDA game canaries, memory/throughput probes.

Generated executables/policies/corpora live under ignored `scratch/bonsai_kg/`.

Reproducible repository-local verification:

```text
make -C bonsai-kg test
make -C bonsai-kg bench
```

The first command is the correctness gate. The benchmark command is evidence only and does not replace semantic parity tests.

## 19. Parity contract

The specification is authoritative; Python is the readable independent oracle; C++ and CUDA compile the same native core.

A portable policy is promotion-eligible only when the same compiled policy hash produces identical per-tick root status, fault state, blackboard, relevant node state, intent/final action and halt behavior in Python/C++/CUDA differential tests. Game-level reward parity alone is insufficient.

Before any future Kaggle submission, policy behavior must additionally pass the project's official Python 64/128-game canary requirements. Bonsai-KG foundation work itself does not submit.
