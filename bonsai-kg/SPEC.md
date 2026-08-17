# Bonsai-KG Semantic Specification

Status: foundation specification, schema version 1, compiled ABI version 1.

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

Every tick has an instruction/node-visit budget stored in the compiled policy. Budget exhaustion is a deterministic VM fault: the root result is `FAILURE`, action output is PASS/empty market, and the fault flag is set. Valid compiled policies are additionally bounded by schema limits and one-body-evaluation-per-turn loop semantics.

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

Threshold Parallel is VM-supported but disabled in search grammar v1.

Parameters: `success_threshold` and `failure_threshold`, each in `[1, child_count]`, child count <= 16.

Children are ticked in ascending child order; there is no OS/GPU concurrency. Terminal children are remembered for the current Parallel activation. RUNNING children are tracked. Threshold success/failure is checked after each child. On terminal Parallel result, any RUNNING siblings are halted and Parallel state is reset. If all currently tickable children skip and no threshold is reached, Parallel returns internal `SKIPPED`; otherwise it returns `RUNNING`.

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

Explicit `While`/`WhileAll` nodes are deliberately not in ABI v1. Reactive composites plus lifecycle `_while` cover the required causal re-evaluation without introducing zero-time loop ambiguity. The compiled ABI reserves node ids for future additive extensions.

## 7. Lifecycle preconditions and effects

Any node may have a lifecycle descriptor. Lifecycle nodes receive a runtime slot even if their core node is otherwise stateless so the VM can distinguish inactive from RUNNING.

Precondition order is fixed:

1. `failure_if`
2. `success_if`
3. `skip_if`
4. `while`

`failure_if`, `success_if`, and `skip_if` are evaluated only when the node is not already RUNNING. `while` is evaluated on initial entry and while RUNNING. A false initial `while` produces internal SKIPPED. A false `while` on a RUNNING node halts its active subtree, executes halt cleanup, then returns SKIPPED.

Lifecycle predicates are finite comparisons over a legal feature or typed blackboard slot; no arbitrary code is allowed.

Terminal hooks execute in this order:

- SUCCESS: `on_success`, then `post`
- FAILURE: `on_failure`, then `post`
- halt/preemption: `on_halted` only
- SKIPPED: no post hook

Effects are finite typed-blackboard operations: `set`, `clear`, `add`. `clear` restores the slot's declared initial value. The compiler rejects slot/type misuse and arithmetic on non-I32 slots.

## 8. Typed blackboard

Authoring-level slots have names and static types. Compilation resolves names to integer indices. Runtime has no hashes or strings.

ABI v1 types:

- `BOOL` (0/1 only)
- `I32`
- `U16` (0..65535)
- `ENUM8` (0..255)
- `TILE` (0..99 or -1 sentinel when explicitly allowed by a task)
- `UNIT` (0..39 or -1 sentinel when explicitly allowed by a task)

Runtime storage is one 32-bit word per slot; type metadata is immutable policy data. Maximum v1 slots: 32.

Lifetimes are distinct:

- blackboard: episode-persistent and reset only on explicit episode reset;
- node execution state: episode-persistent transient control state, reset/halted by BT lifecycle;
- intent arbiter: tick-local and cleared before every root tick.

## 9. Feature registry and legality

Feature ids are fixed by ABI v1 and implemented in all adapters. Parametric features validate their argument range.

Core registry:

- ALWAYS
- STEP, DAY, HOUR
- OWN_MONEY, OPP_MONEY
- OWN_UNITS, OPP_UNITS
- OWN_LAND, OPP_LAND
- OWN_HIRES_TODAY
- MARKET_PRICE(product), MARKET_INVENTORY(product)
- OWN_SHED(item), OWN_SHED_TOTAL
- OWN_SEED(crop)
- OWN_INVENTORY(packed unit/item)
- SHOP_COUNT, SHOP_AT(index), SHOP_SEEN(shop)
- OWN_WORKER_X/Y(unit), OPP_WORKER_X/Y(unit)
- OWN_TILE_KIND(tile), OPP_TILE_KIND(tile)
- OWN_TILE_ITEM(tile), OPP_TILE_ITEM(tile)
- OWN_TILE_YIELD(tile), OPP_TILE_YIELD(tile)
- OWN_TILE_WATERED(tile), OPP_TILE_WATERED(tile)
- OWN_TILE_FED(tile), OPP_TILE_FED(tile)
- OWN_TILE_CARED(tile), OPP_TILE_CARED(tile)
- OWN_TILE_FERT_AVAILABLE(tile), OPP_TILE_FERT_AVAILABLE(tile)

`OWN_INVENTORY` encodes arg as `(unit << 8) | item`.

No registry entry exposes opponent shed/seeds/inventory, RNG seed/state, replay/submission/team identity, future observations, or future shops.

## 10. Leaf/task primitives

ABI v1 portable leaves:

- `Success`
- `Failure`
- `Condition`
- `SetBlackboard`
- `AddBlackboard`
- `EmitUnitIntent`
- `EmitMarketIntent`
- `WaitTurns`
- `MoveUnitTo`

`Condition` compares one legal feature or blackboard slot to an integer using EQ, NE, LT, LE, GT, GE.

`MoveUnitTo(unit,x,y,priority)` is a representative true multi-turn spatial task. If the unit does not exist it fails. At the target it succeeds. Otherwise it emits exactly one Manhattan move intent (X direction before Y direction for deterministic tie behavior) and returns RUNNING. Higher-level strategic tasks such as expansion/crop/animal cycles are expected to be authored as reusable subtrees or later portable task opcodes; they are not hard-coded into the foundation.

Native C++ and CUDA execute the same shared interpreter implementation, so a new portable primitive changes one native code path plus the independent Python oracle, not separate C++ and CUDA semantics.

## 11. Intent and action arbitration

Every tick starts with an empty IntentBuffer.

Unit channels are indexed exactly like the native engine (`0` farmer, `1..` hired hands). A unit channel accepts one winning intent. Winner order is:

1. higher explicit priority (0..255),
2. lower deterministic emission sequence number (DFS order).

Market intents are maintained in the same total order: descending priority, then ascending emission sequence. Final action emits the first `min(environment_max_orders, 16)` entries. Thus environment-significant market order ordering is explicit and portable.

A subtree may emit multiple non-conflicting unit and market intents in one tick, including through Parallel. There are no racing writes and Parallel is not physical concurrency.

A constant unit index outside the ABI capacity is rejected at compile time. If an in-capacity `EmitUnitIntent` references a unit that does not exist in the current own observation, the leaf emits nothing and returns `FAILURE`, allowing normal BT fallback. Validation rejects invalid constant op/item/quantity forms. Finalization sets action unit count from current own public unit count, fills unspecified units with PASS, and never exceeds native fixed capacities.

## 12. Serialization and compiled ABI

Human/interchange format is canonical JSON, schema version 1. Canonicalization removes any supplied `canonical_policy_hash`, serializes with sorted keys and compact separators, and hashes UTF-8 bytes with SHA-256. The resulting hash identifies the authoring policy semantics.

Compiled format `.bkgc` is little-endian ABI version 1:

- fixed 64-byte header with magic `BKG1`, schema version, SHA-256, root id, table counts, state-slot count and instruction budget;
- 32-byte Node records;
- uint16 child-reference table;
- 8-byte blackboard declarations;
- 12-byte Predicate records;
- 24-byte Lifecycle records;
- 8-byte Effect records.

Strings and JSON are absent from the runtime hot loop. PolicyView is immutable pointers/counts over those tables.

Node records contain compact type/flags, child range, optional state slot, optional lifecycle slot, four integer parameters, and stable debug id. Stateless nodes without lifecycle consume no node-state slot.

## 13. Runtime reset

`reset(policy)` is the only episode-start/reset primitive. It zeros node execution state and fault/counters, initializes blackboard slots from declarations, and clears tick-local intents. No blackboard or node state may survive reset.

Python agent integration must call reset at a new episode and on observed step rollback/restart. Native batch policy runners reset once after simulator initialization. Snapshot continuation research must snapshot/copy Bonsai-KG RuntimeState alongside game state if continuation of a stateful prefix is required; resetting from a game snapshot intentionally means starting a fresh policy state.

Day boundaries do not implicitly reset the Bonsai-KG blackboard or node state. Policies use observable DAY/HOUR/STEP if day-specific cleanup is desired.

## 14. Limits and search grammar separation

ABI v1 hard limits:

- nodes: 128
- depth: 24
- child references: 256
- blackboard slots: 32
- stateful/lifecycle runtime slots: 48
- lifecycle predicates: 128
- lifecycle descriptors: 64
- effects: 128
- Parallel children: 16
- Retry/Repeat N: 1..255
- TurnTimeout/WaitTurns: 1..720
- market intent capacity: 16
- unit channels: 40

The VM capability is larger than search grammar v1. Search grammar v1 initially enables Sequence, Selector, ReactiveSequence, ReactiveSelector, If, Invert, ForceSuccess, ForceFailure, Condition, SetBlackboard, AddBlackboard, EmitUnitIntent, EmitMarketIntent, WaitTurns, and MoveUnitTo. It keeps Parallel, Retry, Repeat, KeepRunningUntilFailure, and TurnTimeout available to hand-authored/seeded policies but disabled for blind initial mutation. This can be relaxed by evidence without an ABI redesign.

## 15. Validator fail-closed rules

Compilation rejects at least:

- wrong schema version;
- missing/invalid root tree;
- unknown node/feature/blackboard/effect type;
- empty composites where semantics would be ambiguous;
- wrong decorator/If child cardinality;
- depth/node/child/state/table bounds;
- invalid predicate source, comparison, argument, or blackboard type;
- invalid lifecycle/effect slot/type;
- invalid Retry/Repeat/timeout/wait threshold;
- invalid Parallel thresholds/count;
- invalid action opcode/item/quantity/priority/unit/tile constant;
- any feature not in the explicit legal registry;
- any node not marked portable in ABI v1.

Malformed compiled files are rejected by host loaders before execution. CUDA receives only validated compiled tables.

## 16. Debug/introspection

Debug tracing is opt-in. Python can emit per-node events containing step, debug id, node type, result, and halt events. The shared native VM accepts an optional bounded TraceBuffer; a null pointer is the production CUDA path. Trace overflow truncates trace data only and does not alter policy semantics.

Activation/result/halt counts and RUNNING duration are derived from traces for small runs. No per-node production counter arrays are required in GPU RuntimeState.

## 17. Reference implementation and verification entry points

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

## 18. Parity contract

The specification is authoritative; Python is the readable independent oracle; C++ and CUDA compile the same native core.

A portable policy is promotion-eligible only when the same compiled policy hash produces identical per-tick root status, fault state, blackboard, relevant node state, intent/final action and halt behavior in Python/C++/CUDA differential tests. Game-level reward parity alone is insufficient.

Before any future Kaggle submission, policy behavior must additionally pass the project's official Python 64/128-game canary requirements. Bonsai-KG foundation work itself does not submit.
