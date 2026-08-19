# Bonsai-KG v4 Semantic Specification

Status: BKG4 foundation implementation. BKG3 (`BKG3`, schema/ABI 3) is frozen and remains a separate runtime.

## 1. Design boundary

BKG4 is a finite, statically compiled, integer-deterministic behavior-tree VM for Kaggriculture. It is not a scripting language and not a state-action lookup engine. Runtime policy code can obtain game information only through the versioned legal feature registry. There are no runtime plugins, callbacks, reflection, raw simulator pointers, strings in the hot loop, dynamic allocation inside CUDA kernels, unbounded loops, recursion, floating-point strategy expressions, state hashes, replay identities, opponent identities, RNG state, or future observations.

BKG4 feature-registry version 1 is executable-identical to the frozen BKG3 legal observation surface. VM ABI and feature-registry version are separate identities so future deterministic legal aggregates do not require a BKG5 structural redesign.

## 2. Structural ABI

- magic: `BKG4`
- schema: 4
- ABI: 4
- feature registry: 1
- board size: 10 semantic constant
- root: node 0
- node id: `uint16_t`
- `NONE_U16`: `0xFFFF`
- maximum nodes: 65,534
- valid ids: 0..65,533
- maximum exclusive `subtree_end`: 65,534 (`0xFFFE`)
- strict single-parent tree only
- canonical preorder
- subtree-contiguous layout
- `n_child_refs == n_nodes - 1`
- no runtime DAG or subtree calls

The fixed 64-byte header stores actual counts and independently checkable execution proofs: node/state/BB/metadata counts, instruction budget, maximum live intents, maximum structural depth, and feature-registry version. `Node4` remains 32 bytes.

## 3. Runtime memory

Structural capacity is static shared program memory. It does not imply per-environment storage.

Persistent logical state is:

- `RuntimeMeta4 { fault, ticks }`
- `blackboard[n_bb]`
- `NodeState4[n_state]`

`n_bb` and `n_state` are actual compiled counts. BKG4 contains no `NodeState[MAX_NODES]`, no `NodeState[MAX_STATE]` environment object, and no per-node visitation array in production. CUDA allocates persistent slabs before kernels; physical AoS/SoA layout is not semantic.

Hard ceilings are 128 BB slots and 512 NodeState slots. NodeState is interpreter continuation state; long-horizon strategic commitments should live in named semantic blackboard slots.

## 4. Compiler-proven state elision

The compiler computes bottom-up `may_return_running` and `needs_cross_tick_state`. A composite receives a persistent slot only if its executable subtree can survive a tick and the node must remember continuation/preemption state. Lifecycle metadata alone does not force state on a synchronous node. A lifecycle on a node that may run does require active-state tracking.

Examples:

- synchronous Sequence of Conditions/Set/Success: zero persistent state
- Sequence containing WaitTurns: Sequence cursor + WaitTurns timer state
- synchronous SwitchEnum: zero state
- SwitchEnum with a possibly RUNNING branch: state to remember/preempt prior selection
- Retry/Repeat with count >1: state even when child is synchronous
- MoveUnitTo: no intrinsic state; lifecycle may require active state

Native loaders independently recompute this proof and reject missing, extra, aliased, or non-dense state assignments.

## 5. Control semantics and BehaviorTree.CPP relationship

BKG4 borrows established BT semantics but does not depend on BehaviorTree.CPP.

### Sequence / Selector

Stateful only when required by state-elision analysis. Resume a RUNNING child. These MATCH standard stateful Sequence/Fallback behavior at the semantic level, with BKG-specific `SKIPPED` handling and deferred action transactions.

### ReactiveSequence / ReactiveSelector

Re-evaluate from child zero every tick and halt a previously RUNNING child when earlier feedback selects/stops elsewhere. MATCH the mature reactive-control principle; BKG-specific halt also cancels current-tick intents from the halted subtree.

### If

Condition/then/else with selected branch latched while RUNNING. MATCHES BehaviorTree.CPP-style non-reactive `IfThenElse`: once a branch is RUNNING the condition is not used to switch branches until the activation terminates.

### ReactiveIf

Exactly condition/then/else (omitted else compiles to Failure). The condition must be synchronous and is evaluated every tick. If its boolean outcome changes while the old branch is RUNNING, BKG4 halts the old branch, cancels its current-tick intents, applies `on_halted`, then executes the newly selected branch. This MATCHES the reactive switching principle of BehaviorTree.CPP `WhileDoElse`, while deliberately restricting the condition to normal finite Bonsai nodes and requiring it to be synchronous.

### SwitchEnum

Direct semantic dispatch over a *named enum blackboard slot only*. Children are compiled densely in enum-value order plus a default safety branch. Selected-child lookup is constant-time. A changed selection preempts a prior RUNNING child before the new branch executes. This MATCHES BehaviorTree.CPP Switch preemption behavior but DELIBERATELY DIFFERS by forbidding arbitrary I32/string/hash lookup keys and runtime ports.

### Parallel

2..16 children, deterministic ascending evaluation order, explicit success/failure thresholds, RUNNING-child tracking and deterministic halt on terminal threshold/impossibility. It follows the established sequential-per-tick Parallel model; BKG4 additionally uses transactional intent cancellation and fixed 16-bit activation masks.

### Decorators

Invert, ForceSuccess, ForceFailure, TurnTimeout, Retry, Repeat, KeepRunningUntilFailure and WaitTurns retain BKG3 discrete-turn semantics. Retry/Repeat perform at most one body completion attempt per game tick. BKG4 does not import asynchronous nodes, general Script, plugins, dynamic RunOnce nodes or arbitrary callback decorators.

## 6. Lifecycle semantics

Lifecycle order remains:

1. `failure_if`
2. `success_if`
3. `skip_if`
4. `while`

The first three are evaluated only on non-active entry. `while` is also re-evaluated while RUNNING and preempts on false. SUCCESS executes `on_success` then `post`; FAILURE executes `on_failure` then `post`; preemption executes `on_halted`; SKIPPED has no terminal post. Blackboard effects are not transactionally rolled back.

## 7. Named enums

Authoring may define named enum types. They erase to `ENUM8` storage but retain a compact `enum_tag` and `enum_size` in `BbDecl4`; CONST/BB ValueRefs carry the enum tag. The compiler/loader reject cross-enum comparisons, writes and subtree bindings. Named enums support EQ/NE comparisons only. Runtime has no enum strings.

`SwitchEnum` accepts only a named enum BB slot. There is intentionally no `SwitchI32`.

## 8. Legal feature registry

Registry v1 contains the same current-observation features as frozen BKG3: time, public money/labor/land, shared market, acting-player private shed/seeds/inventory, current shops, public worker positions, and documented public tile lifecycle state. It exposes no opponent-private inventory/shed/seeds, RNG, future shop/weed sequence, replay/submission/team identity, whole-board hash, arbitrary byte data or native engine pointer.

A future registry version may add only deterministic pure integer features derived exclusively from the current legal observation and implemented identically in Python/C++/CUDA.

## 9. Immutable metadata and interning

ValueRefs, contiguous FeatureArg sequences, lifecycle predicates, lifecycle descriptors and immutable effect sequences are deterministically interned by first canonical occurrence. Immutable aliases are legal. Mutable NodeState is never shared. Host loaders validate all referenced metadata and reject unused table records.

Metadata uses uint16 addressability (up to 65,534 records per table); practical policy budgets are compiler/search concerns rather than another small ABI ceiling.

## 10. Deferred intent transaction

Game-action leaves append to a tick-local deferred journal. Structural capacity and live journal capacity are independent.

`DeferredIntent4` is 12 bytes and stores:

- `uint16 priority_key`
- `uint16 source node`
- `uint16 emission order`
- `uint16 quantity`
- operation/argument/subject bytes

The compiler computes a conservative safe maximum simultaneously live journal requirement, considering rollback, branch exclusivity and Parallel concurrency. Header `max_live_intents` must equal the independently recomputed proof and may not exceed 128.

Every evaluator frame records its entry journal length. FAILURE/SKIPPED/fault rolls back to that checkpoint. `halt_node(root)` removes intents whose 16-bit source lies in `[root, subtree_end)`. Emission order is 16-bit so rolled-back attempts may exceed 255 without tie-order corruption.

CUDA scratch classes are 0/8/16/32/64/128 intents; depth classes are 16/24/32/48. Only scratch layout is template-specialized. The interpreter body is shared, preventing a code-generation cross-product while preserving compile-time local-memory sizing.

## 11. Authority and arbitration

Action leaves have a compiler-visible authority class plus local priority:

`FALLBACK < TACTICAL < DOMAIN < COMMITTED_STRATEGY < RECOVERY < INVARIANT`

`priority_key = (authority << 8) | local_priority`.

Unit arbitration chooses highest key then earliest emission. Market arbitration is descending key then ascending emission and applies the environment market-order cap after ordering. Converted BKG3 leaves receive one uniform authority class, preserving existing relative priority behavior.

Authority is a final deterministic conflict layer, not a substitute for coherent tree gating.

## 12. Static execution proof

For each subtree the compiler derives at least:

- may run / needs state
- maximum structural depth
- conservative maximum node visits per tick
- conservative maximum live intents
- maximum emissions per tick
- BB read/write sets
- legal feature dependencies
- action domains
- authority classes

No same-tick unbounded loop exists, so visit/emission bounds fit the 16-bit structural ceiling. Instruction budget is independent of total policy size and must be >= the static visit proof. Policies above the normal 4096-visit engineering target are linted for review.

Large linear Selector dispatch receives `LARGE_LINEAR_DISPATCH` / SwitchEnum guidance.

## 13. Compile-time reuse and strategic patterns

Subtree/module reuse remains compile-time expansion into independent strict tree fragments. Typed IN/OUT/INOUT bindings are checked before expansion. No runtime DAG exists.

`python/bonsai_kg_v4_patterns.py` supplies authoring helpers for Ensure, Maintain, RecoveryFallback, DeadlineGuard, transactional intent groups, RunOnce, HysteresisBand and GuardedCommitment. They compile to ordinary BKG4 nodes and BB state.

## 14. Module contracts and anti-memorization linting

The compiler produces a non-hot manifest with module intervals/counts, depth/visit/state/intent bounds, BB reads/writes, feature dependencies, action domains, authorities, may-run state and optional provenance. BB declarations have semantic owners; foreign/multi-module writers are linted.

The compiler warns on large linear dispatch and suspicious exact equalities on high-cardinality features such as STEP/money/market quantities. Exact mechanically justified temporal rules remain legal. Whole-state hash features, arbitrary callbacks, identity/seed/future-state features and arbitrary-I32 Switch routing are impossible in the BKG4 executable vocabulary.

Policy promotion is expected to use multi-seed/both-seat/held-out/regression/metamorphic evidence and module provenance; VM capacity is not permission to add replay patches.

## 15. Semantic identity and loader contract

Semantic SHA-256 domain is `BKG4SEM\0`. The normalized little-endian stream includes ABI/registry identity, actual counts/proofs, board semantic constant, all behavior-affecting node fields including `aux`, child refs, BB enum metadata/initials, ValueRefs, FeatureArgs, predicates, lifecycles and effects. Source names, formatting, module provenance and comments are excluded.

Python and C++ independently reconstruct the stream. CUDA receives only host-validated tables.

Loaders fail closed on malformed identity/counts, wrong registry, wrong exact file size/trailing data, enum/type/action errors, strict-tree violations, preorder/subtree errors, state-elision disagreement, proof under-reporting, illegal features, metadata corruption/padding, invalid authority and semantic-hash mismatch.

## 16. BKG3 compatibility

BKG3 files, headers, compiler, runtime, tests, stateful-v6 integration and assets are not reinterpreted or widened. BKG4 provides semantic source/decompile conversion: BKG3 -> authoring v4 -> BKG4 compiler. This is used only for equivalence validation until an intentional strategy migration occurs.

## 17. Search architecture

Production serialization is always one canonical strict tree. Large-policy search operates on a frozen base and a bounded mutable semantic module; it must not replicate the full 10k-60k policy image per candidate. Search grammars are domain-restricted. Winning modules are compiled into a complete canonical policy, minimized/ablated, then rerun through exact parity and regression gates.

## 18. Performance principle

Address-space capacity, actual policy size and hot execution size are independent. A 65k policy with a 40-node hot path should pay principally for those visited nodes plus compact static-table/cache effects. The benchmark suite separately varies cold policy size, hot visits, actual state count, journal class and depth class.

The defining invariant of BKG4 is:

> 65k increases how much verified strategic knowledge a policy may contain, not how much state or work every environment must pay for.
