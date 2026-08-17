# Bonsai-KG Semantic Specification

Status: foundation finalization specification, schema version 3, compiled ABI version 3 (`BKG3`).

Bonsai-KG is the Kaggriculture-native compiled behavior-tree VM used by future ReCurSiON challenger agents. It is isolated from the frozen champion lineage and from upstream `bonsai/`.

## 1. Design boundary

One Kaggriculture agent call is one Bonsai-KG tick. Immutable compiled policy data may be shared across simulations; each simulation owns an independent `RuntimeState`.

The VM is intentionally finite and CUDA-friendly. It has no runtime plugins, arbitrary scripts/callbacks, strings in the hot loop, reflection, wall-clock timers, dynamic allocation, unbounded recursion, or arbitrary expression evaluator.

Portable policy/task code receives only a restricted legal observation provider exposing:

```text
feature(feature_id, arg0, arg1)
```

The production C++ backing `Sim` and CUDA backing `ObservationView` are private implementation details of their adapters and are not accessible to policy/task code. There is no public raw-state accessor.

## 2. Status model

Task statuses are:

- `SUCCESS`
- `FAILURE`
- `RUNNING`

Internal `SKIPPED` exists only for lifecycle skip/while composition. It is not a user leaf result. Sequence and Selector ignore skipped children. Root `SKIPPED` normalizes to external `SUCCESS` with PASS/empty market output.

A VM fault is separate from task failure. Faults are fatal and sticky until `reset(policy)`.

## 3. Tree and execution model

Compiled runtime structure is a strict single-parent tree in canonical preorder. `root == 0`. Every `Node` stores `subtree_end`, the exclusive preorder node id after its complete subtree. Thus node `i` owns the interval:

```text
[i, nodes[i].subtree_end)
```

The loader proves preorder, parent uniqueness, reachability, child ranges, depth, and exact `subtree_end` values.

Traversal is deterministic depth-first, left-to-right. Every tick has a finite instruction/node-visit budget. Budget exhaustion sets a sticky fault, returns external `FAILURE`, and materializes no game action.

All policy arithmetic is integer. Money is read through the existing checked integral conversion used by all three engines. `boardSize == 10` is a hard portable contract. TILE encoding is row-major `y * 10 + x`.

## 4. Stateful and reactive composites

### Sequence

Stateful. Resume the RUNNING child. SUCCESS/SKIPPED advances in the same tick. FAILURE resets and fails. RUNNING stores the cursor. All children complete => SUCCESS and reset.

### Selector

Stateful priority selector. Resume the RUNNING child. FAILURE/SKIPPED advances in the same tick. SUCCESS resets and succeeds. RUNNING stores the cursor. Exhaustion => FAILURE and reset.

### ReactiveSequence / ReactiveSelector

Restart from child zero every tick and remember the previously RUNNING child. If evaluation selects another branch, the old RUNNING subtree is halted deterministically.

ReactiveSequence advances on SUCCESS/SKIPPED and stops on FAILURE/RUNNING. ReactiveSelector advances on FAILURE/SKIPPED and stops on SUCCESS/RUNNING.

The remembered RUNNING child is halt-eligible only while traversal has not yet revisited that child on the current tick. Once that exact child is evaluated again, it has either completed normally or remains the active RUNNING branch; it must not receive a later stale HALT merely because traversal continues to another child. A genuine reactive preemption occurs only when an earlier child changes outcome and stops/selects before the remembered old branch is reached. Normal terminal completion runs normal terminal hooks and never `on_halted`.

### If

Exactly condition/then/else. The selected branch is latched while RUNNING. Terminal branch result clears the latch. A skipped condition yields internal SKIPPED.

### Parallel

Threshold Parallel is VM-supported but remains disabled in blind search grammar v3.

- child count: 2..16
- success/failure thresholds: 1..child_count
- deterministic ascending child order
- SUCCESS/FAILURE children remembered during an activation
- RUNNING children tracked
- skipped current-tick children are excluded from remaining possible successes
- terminal threshold or success-impossibility halts all RUNNING siblings and resets Parallel state
- all-skipped activation with no remembered terminals => SKIPPED
- remembered terminals + no RUNNING child + unmet/impossible threshold => FAILURE
- Parallel returns RUNNING only if at least one unresolved child actually remains RUNNING.

Critically, halting a RUNNING sibling also cancels every current-tick deferred action intent emitted by that sibling subtree before final action materialization. See section 10.

## 5. Decorators and temporal controls

`Invert`: SUCCESS <-> FAILURE; RUNNING/SKIPPED unchanged.

`ForceSuccess` / `ForceFailure`: terminal non-skipped result maps to the forced result; RUNNING/SKIPPED unchanged.

`TurnTimeout(N)`: game-step timeout only. On timeout the active child is halted, then FAILURE is returned.

`Retry(N)`, `Repeat(N)`, `KeepRunningUntilFailure`: at most one body evaluation/completion attempt per game turn. No same-turn unbounded loops.

`WaitTurns(N)`: first activation stores STEP and returns RUNNING; succeeds once elapsed turns >= N.

Explicit generic While/WhileAll opcodes are not part of BKG3. Reactive composites plus lifecycle `while` provide bounded causal re-evaluation.

## 6. Lifecycle semantics

Any node may have a lifecycle descriptor. Preconditions are checked in this fixed order:

1. `failure_if`
2. `success_if`
3. `skip_if`
4. `while`

`failure_if`, `success_if`, and `skip_if` are checked only when the node is not already RUNNING. `while` is also checked while RUNNING. False RUNNING `while` halts the active subtree and returns SKIPPED.

Terminal effects:

- SUCCESS: `on_success`, then `post`
- FAILURE: `on_failure`, then `post`
- halt/preemption: `on_halted` only
- SKIPPED: no post

Effects are finite typed blackboard `set`, `clear`, and `add` operations. `set`/`add` consume ValueRefs. `clear` restores the declaration initial value. `add` is I32-only.

`on_halted` blackboard effects still execute when same-tick action intents are cancelled. Blackboard state is not globally transactional.

## 7. Typed blackboard

Blackboard declaration types:

- `BOOL`: 0/1
- `I32`
- `U16`: 0..65535
- `ENUM8`: 0..255
- `TILE`: NONE=-1 or 0..99
- `UNIT`: NONE=-1 or 0..39
- `ITEM`: NONE=-1 or WHEAT..SHEEP (0..11)

Runtime storage remains one signed 32-bit word per slot. Maximum slots: 32.

BOOL/U16/ENUM8 are read as effective I32 ValueRefs while their declaration constraints still govern writes. TILE, UNIT, and ITEM remain distinct portable ValueRef types.

## 8. ValueRef architecture

Portable ValueRef types:

- `I32`
- `TILE`
- `UNIT`
- `ITEM`

ValueRef sources:

- `CONST`
- `BB`
- `LEGAL_FEATURE`

There are no implicit TILE/UNIT/ITEM/I32 conversions. Comparisons require matching effective types.

TILE, UNIT, and ITEM accept NONE=-1. Runtime operation consumers decide whether NONE is locally valid; malformed compiled type metadata is rejected before execution.

`OWN_TILE_ITEM(tile)` and `OPP_TILE_ITEM(tile)` use ITEM NONE (`-1`) whenever the tile has no logical crop/animal. A PLANT returns its crop ITEM id; an occupied COOP/PASTURE returns its animal ITEM id. EMPTY, LOCKED, WEED, empty animal structures, structures after an animal has escaped, TILE NONE, and out-of-range TILE indices return `-1`, never WHEAT/0.

A legal-feature ValueRef does not recursively contain arbitrary ValueRefs as indices. Instead it points to a compact contiguous `FeatureArgRef` range.

## 9. Dynamic legal feature indices

`FeatureArgRef` is an 8-byte immutable compiled record. Its source is restricted to:

- `CONST`
- `BB`

This deliberately prevents recursive feature-expression graphs while permitting reusable dynamic ports.

Feature signatures are statically checked by compiler and loaders:

- `MARKET_PRICE(item)`, `MARKET_INVENTORY(item)`: ITEM
- `OWN_SHED(item)`: ITEM
- `OWN_SEED(item)`: ITEM; runtime feature semantics naturally return zero outside crop ids
- `OWN_INVENTORY(unit,item)`: UNIT, ITEM
- `SHOP_AT(index)`, `SHOP_SEEN(index)`: I32
- own/opponent quadrant unlocked: I32
- own/opponent worker X/Y/TILE: UNIT
- every own/opponent tile public field: TILE
- all remaining features: no arguments.

BKG3 uses two logical arguments for `OWN_INVENTORY`; packed `(unit << 8) | item` is not the runtime representation. The compiler accepts the old static packed authoring shorthand only as compatibility sugar where applicable.

Dynamic examples that are valid BKG3 programs include:

```text
OWN_TILE_WATERED[target_tile]
OWN_TILE_CONSECUTIVE_UNWATERED[target_tile]
OWN_WORKER_TILE[worker]
MARKET_PRICE[item]
OWN_INVENTORY[worker,item]
```

NONE/out-of-range runtime indices are deterministic and follow the registered feature adapter's absent-value semantics; they never expose hidden memory.

## 10. Deferred action transaction and halt cancellation

Game-action leaves never write directly to the environment action. They append to a tick-local deferred intent journal of at most 128 compact 8-byte records.

Each node frame records an entry journal checkpoint. FAILURE or SKIPPED rolls the journal back to that checkpoint. Root FAILURE/SKIPPED/fault discards the full journal. SUCCESS/RUNNING descendants commit to their parent transaction.

Each deferred intent additionally records the emitting source node id. `halt_node(root)` uses the canonical preorder interval `[root, subtree_end)` to compact the current journal and remove every intent emitted by the halted subtree before it performs normal active-path cleanup.

Therefore the general invariant is:

> If a subtree is halted before final action materialization, every current-tick deferred game-action intent emitted by that subtree is cancelled.

This applies uniformly to Parallel threshold completion, reactive preemption, timeout, lifecycle `while`, and future halt sources that use the shared halt primitive.

Journal compaction does not roll back blackboard writes. `on_halted` effects still execute. Native frame rollback clamps stale pre-halt checkpoints to the compacted journal length so compaction can never resurrect stale records.

After traversal, unit arbitration chooses highest priority then earliest deterministic emission order. Market arbitration sorts descending priority then ascending emission order and applies the environment cap after ordering.

## 11. Dynamic action operands

Action opcodes remain compile-time constants. Priority remains compile-time constant.

`EmitUnitIntent` operands:

- unit: UNIT ValueRef
- action/opcode: constant
- item: ITEM ValueRef for PLANT/PICKUP/PLACE; canonical NONE otherwise
- quantity: I32 ValueRef for PICKUP/PLACE; canonical constant 1 otherwise
- priority: constant 0..255

`EmitMarketIntent` operands:

- action/opcode: constant
- item: ITEM ValueRef for BUY_SEED/BUY_PRODUCT/BUY_ANIMAL/SELL; canonical NONE for HIRE/BUY_LAND
- quantity: I32 ValueRef; HIRE/BUY_LAND require canonical constant 1
- priority: constant 0..255

Runtime values that are correctly typed but invalid for the selected operation produce local leaf FAILURE and emit nothing. Examples include UNIT NONE, missing worker, SELL of an animal id, BUY_SEED of a non-crop, quantity <=0, or quantity beyond the operation range. These are not VM faults.

Operation subsets:

- PLANT: crop ids 0..4
- PICKUP/PLACE: item ids 0..11; quantity 1..32767
- BUY_SEED: crop ids 0..4
- BUY_PRODUCT: WHEAT or FERTILIZER
- BUY_ANIMAL: GOOSE/COW/SHEEP
- SELL: product/fertilizer ids 0..8
- market quantity: 1..65535.

Compiler and binary loaders require the same canonical hidden/static operands. HIRE/BUY_LAND must store exact CONST ITEM NONE and exact CONST I32 quantity 1. PLANT must store exact CONST I32 quantity 1. Unit actions other than PLANT/PICKUP/PLACE must store exact CONST ITEM NONE and exact CONST I32 quantity 1. For operations that permit dynamic ITEM/quantity, dynamic typed refs remain valid, while compiler-known-invalid CONST domains are rejected during compilation/loading. This keeps every accepted binary faithfully decompilable to an authoring form with the same executable identity.

`MoveUnitTo` consumes dynamic UNIT and TILE ValueRefs and emits one Manhattan move, X before Y, while RUNNING.

## 12. Reusable subtree ports

Subtrees are compile-time authoring constructs only. BKG3 supports typed parameters with modes:

- `IN`: readable source; may bind a compatible CONST, BB, or legal-feature ValueRef
- `OUT`: write-only destination; must bind a mutable caller blackboard slot of exactly the declared type
- `INOUT`: readable and writable; must bind a mutable caller blackboard slot of exactly the declared type.

Destination bindings resolve wherever a destination slot is meaningful, including `SetBlackboard`, `AddBlackboard`, and lifecycle set/add/clear effects.

Nested forwarding is statically checked. OUT may forward only to an outer writable port; INOUT requires an outer INOUT port. Reading an OUT-only port is a `PolicyError`.

After expansion there are no runtime subtree calls, port names, binding maps, strings, or implicit local contexts. Each instantiation becomes an independent strict tree fragment and concrete blackboard/value references. Recursion is rejected.

## 13. Legal observation registry and privacy boundary

The registry contains only legal current-observation information:

- ALWAYS, STEP, DAY, HOUR
- own/opponent public money, units, land/quadrants, hires_today
- current shared market price/inventory
- current shared shops
- acting player's private shed, seeds, per-worker inventory
- own/opponent public worker positions/tile ids
- own/opponent public tile kind/item/yield and all documented plant/animal lifecycle fields.

No registry entry exposes:

- opponent private shed
- opponent private seeds
- opponent private worker inventories
- RNG seed/state or future RNG
- future shop sequence
- replay/episode/submission id
- opponent/team identity
- ReCurSiON identity
- native-only hidden caches/statistics.

`CppObservation` keeps its `Sim*` and player privately. `CudaObservation` keeps its `ObservationView` privately. Portable task code can obtain game information only through `feature(...)`. Compile-time regression guards verify the raw backing members are inaccessible.

## 14. Serialization and BKG3 ABI

Human authoring format: strict JSON schema version 3.

Compiled `.bkgc`: little-endian `BKG3`, ABI version 3.

Fixed record sizes:

- Header: 64 B
- Node: 32 B
- BbDecl: 8 B
- ValueRef: 12 B
- FeatureArgRef: 8 B
- Predicate: 12 B
- Lifecycle: 24 B
- Effect: 8 B.

The 64-byte header stores magic/version/header size, semantic SHA-256, root, table/state counts, instruction budget, and required board size. `max_intents` is not serialized in BKG3; host loading derives the number of potentially emitting nodes and places it in the runtime `PolicyView` for the journal-free fast path when zero.

Node's compact `subtree_end` field is part of executable semantics/canonical representation and is hashed.

## 15. Semantic hash and canonical identity

SHA-256 is computed over an explicit normalized little-endian executable stream with domain prefix `BKG3SEM\0`.

The stream includes:

- ABI/domain version
- board size
- root/counts/state/instruction budget
- derived static emitting-node count
- every behavior-affecting Node field including `subtree_end`
- child references
- blackboard declarations/initials
- ValueRefs
- FeatureArgRefs
- predicates
- lifecycle descriptors
- effects.

It excludes the stored hash itself, authoring names, subtree names, formatting, and debug ids. Canonical compiled files still require debug id == node index.

Canonical slot/table ordering and canonical preorder are part of executable-program identity. Reordering blackboard/table declarations while consistently changing references may therefore change the hash. BKG3 does not attempt general mathematical program equivalence.

Python and host C++ independently reconstruct the normalized stream and recompute SHA-256. CUDA receives only host-validated tables.

## 16. Reset and persistent state

`reset(policy)` initializes blackboard declarations, zeros node state, visits/fault/tick counters, and begins a fresh episode policy state.

Day boundaries do not implicitly reset Bonsai-KG state. Snapshot continuation must copy `RuntimeState` alongside simulator state if policy continuation is intended.

The deferred journal, arbitration claims, evaluator stack, and halt stack are tick-local and are not persistent `RuntimeState`.

## 17. Hard limits and search-grammar separation

BKG3 limits:

- nodes: 128
- depth: 24
- child refs: 256
- blackboard slots: 32
- runtime state slots: 48
- ValueRefs: 256
- FeatureArgRefs: 512
- predicates: 128
- lifecycle descriptors: 64
- effects: 128
- Parallel children: 16
- Retry/Repeat: 1..255
- TurnTimeout/WaitTurns: 1..720
- deferred intent journal: 128
- market materialization: 16
- unit channels: 40.

VM capability is intentionally broader than search grammar v3. Dynamic UNIT/ITEM/quantity operands, dynamic feature indices, typed ports, Parallel, and bounded decorators may exist in seeded/hand-authored macros without forcing blind evolution to mutate the entire space. `schema/search_grammar_v3.json` is the current conservative mutation contract.

## 18. Validation and verification contract

Compiler/loader failures are fail-closed. They reject malformed types, invalid table ranges/aliases, noncanonical empty ranges, invalid state/lifecycle reuse, malformed feature signatures, invalid action ValueRef types, wrong board size, wrong preorder/subtree intervals, recursion, unreachable nodes, cycles, trailing bytes, nonzero padding/reserved bytes, and semantic hash mismatch.

Runtime operation-domain invalidity that can legitimately arise from dynamic typed values is a leaf FAILURE, not a corrupt-policy fault.

Static CONST writes to a blackboard destination are additionally checked against the destination's declared type, not only its effective ValueRef type. This applies to `SetBlackboard` and lifecycle SET effects, so for example CONST I32 `2` cannot be hidden in a BOOL destination encoding. Hash-valid binaries that violate these compiler-known canonical constraints are rejected before execution.

Primary verification entry points:

```text
make -C bonsai-kg test
make -C bonsai-kg bench
```

The correctness gate includes retained semantic suites, BKG3 finalization acceptance tests, malformed-binary checks, 512-policy Python/C++/CUDA randomized differential testing, permanent three-scenario two-seat official feature parity, and both-seat official/C++/CUDA integrated game parity.

Generated executables and corpora remain under ignored `scratch/bonsai_kg/`.

## 19. Promotion contract

Python is the readable independent oracle; C++ and CUDA share one host/device native interpreter.

A portable policy is promotion-eligible only if the same compiled executable identity produces exact cross-engine status, fault, action, blackboard, and allocated node-state behavior under differential testing. Game reward equality alone is insufficient.

Before any future Kaggle submission, the project additionally requires the separate official Python 64/128-game promotion canary for the actual candidate. Foundation finalization itself does not create or submit a strategic agent.
