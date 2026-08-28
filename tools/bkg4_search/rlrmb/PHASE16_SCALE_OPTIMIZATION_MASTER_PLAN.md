# ReCurSiON RLRMB Phase-16 Scale Optimization Master Plan

> **For agentic workers:** Execute this roadmap strictly phase-by-phase. Each phase has a hard validation gate. Do not begin the next phase until the current phase has passed its falsifiable acceptance criteria. This document extends the already-validated RLRMB Phase 16 after the first real Lightning pilot; it does **not** replace or modify `RLRMB_MASTER_IMPLEMENTATION_PLAN.md` or `BONSAI_FIRST_PLACE_MASTER_PLAN.md`.

**Goal:** Convert the working Phase-16 Lightning RLRMB system into a high-duty-cycle, persistent-memory, adaptive autonomous search system that maximizes **new safely solved regimes per cloud/GPU hour** while preserving exact deterministic BKG4 semantics.

**Architecture:** Optimization proceeds in seven ordered steps. First stop deterministic repeated work, then parallelize the actual CPU bottleneck, then persist expensive invariant resources, then optimize cloud duty cycle/CUDA batching, then add durable semantic search memory, then make search allocation adapt from that memory, and finally prove repeated multi-generation autonomous BKG4 improvement under a bounded cloud stress campaign.

**Tech Stack:** Python orchestration, `ProcessPoolExecutor`-based deterministic host workers, existing RLRMB Option-Program IR, existing effect-vector QD archive, existing exact CUDA evaluator, CPU C++/BKG4 parity tooling, existing hardened CEGIS/composer/materializer, compact JSON artifacts, optional Python stdlib `sqlite3` only if Phase 16.5 evidence shows indexed durable memory is required.

**Strategic authority:**

- `tools/bkg4_search/BONSAI_FIRST_PLACE_MASTER_PLAN.md` — read-only strategic contract.
- `tools/bkg4_search/rlrmb/RLRMB_MASTER_IMPLEMENTATION_PLAN.md` — completed RLRMB construction plan and validation history.
- `tools/bkg4_search/rlrmb/configs/phase16_lightning_v1.json` — current cloud-scale configuration.
- `tools/bkg4_search/rlrmb/PHASE16_CLOUD_READINESS.json` — pre-pilot bundle/readiness evidence.
- `scratch/phase16_1h_run/` — first real Lightning pilot evidence; treat as source evidence, not as production input.

---

# 0. Frozen post-pilot baseline

The first real Lightning pilot proved that the end-to-end Phase-16 pipeline works strategically but wastes most cloud wall time.

Validated strategic result:

```text
cloud wall                  3913.069 s
CUDA games                  1,988,964
committed BKG4 generations  1
accepted specialists        1
terminal verdict            PROMOTION_READY
semantic mismatches         0
protected W->L              0
```

Measured performance diagnosis:

```text
CPU candidate generation                 ~2547 s  (~65.1%)
per-attempt uninstrumented/cert overhead ~941 s  (~24.1%)
legal-state capture                       ~288 s  (~7.4%)
CUDA evaluator host wall                  ~135 s  (~3.5%)
CUDA kernel time                           ~66 s  (~1.7%)
overall cloud throughput                  ~508 games/cloud-second
attempt-local evaluator throughput        ~12k-19k games/second
```

Hardware evidence:

```text
GPU average utilization  ~6.75%
GPU 0% samples           ~89.4%
VRAM high-water          ~9.6 GiB / 96 GiB
CPU average              ~9%
RAM high-water           ~5.3% / 128 GiB
```

The key diagnosis is not "CUDA is slow". The active evaluator is already fast. The current system is dominated by:

1. single-threaded Python candidate construction;
2. deterministic re-search of unchanged frontiers with effectively no cross-attempt memory;
3. repeated process/cache/input setup;
4. only then smaller CUDA launch/context and batch-size inefficiencies.

The first run also proved a second strategic weakness: the outer loop repeatedly cycles the same four families and reproduces essentially identical candidate/effect results because the search state resets on each attempt.

This roadmap exists to close those bottlenecks in the correct order.

---

# 1. Global constraints for Phases 16.1-16.7

These apply to every phase.

1. **Bonsai/BKG4 remains the final runtime agent.** Search intelligence may improve offline; deployed behavior remains deterministic BKG4.
2. **CUDA remains the heavy evaluator.** Do not move large game evaluation to CPU merely to increase CPU utilization.
3. **CPU/CUDA/BKG4 semantics must remain exact.** Required mismatch count is always `0`.
4. **Do not weaken acceptance gates.** Routability, transfer, validation, hardened CEGIS, SEALED discipline, composition, protected regression and parity remain authoritative.
5. **SEALED never enters generation, memory-driven proposal ranking or adaptive allocator training.** It remains downstream certification-only.
6. **No live champion auto-promotion.** Cloud search may produce `PROMOTION_READY`; actual champion replacement remains deliberate and requires the official Python canary.
7. **No large search locally.** Local work is limited to unit tests, parity, deterministic fixtures, bounded profiling and bundle self-tests. Large validation belongs on Lightning/CUDA.
8. **Determinism is non-negotiable.** Worker scheduling, cache hit order, restart order, database row order and filesystem order must not affect semantic candidate IDs, QD state, acceptance order or final BKG4.
9. **Cached/reused work does not consume fictional CUDA budget.** The CUDA ledger records games actually executed. Reuse metrics separately record `avoided_cuda_games` or equivalent. Never charge cached outcomes as if games ran.
10. **Generation changes invalidate context-dependent evidence.** Candidate structural identity may persist, but margins, EffectVectors, legal-state captures and certification outcomes are authoritative only under the exact BKG4/panel/config context that produced them.
11. **No opponent-identity runtime logic.** Offline family labels may guide search bookkeeping only.
12. **Major gains only.** Do not chase JSON formatting, tiny hash savings or cosmetic refactors while P0/P1 bottlenecks remain.
13. **Repository hygiene:** permanent source/docs/tests are tracked; generated databases/runs/bundles remain under ignored scratch/output locations. Preserve the five pre-existing unrelated untracked files.
14. **Do not modify the read-only master plans.** This file is the post-pilot execution roadmap.
15. **Primary productivity metric:** `new safely solved regimes / GPU-hour` and, operationally, `new semantic/effect territory / cloud-hour`. Raw games/s is secondary.

---

# 2. Phase dependency graph

```text
16.1  STOP EXACT REPETITION + COMPLETE TIMING
  |
  v
16.2  PARALLELIZE CANDIDATE CONSTRUCTION
  |
  v
16.3  PERSIST HOST RESOURCES + INVARIANT CACHES
  |
  v
16.4  CLOUD DUTY-CYCLE + CUDA/VRAM TUNING
  |
  v
16.5  DURABLE SEMANTIC SEARCH MEMORY
  |
  v
16.6  ADAPTIVE / SELF-IMPROVING SEARCH ALLOCATION
  |
  v
16.7  MULTI-GENERATION STRESS + LARGE-CAMPAIGN READINESS
```

Do not merge 16.1 and 16.6. Exact-repeat suppression is a correctness/efficiency layer. Adaptive search is a later policy layer that intentionally chooses new work.

Do not optimize GPU batch size before 16.1-16.3 have removed the dominant host waste.

---

# 3. Phase 16.1 — Exact-repeat suppression and full stage timing

## 3.1 Purpose

Stop the cloud runner from executing the same deterministic frontier search repeatedly under an unchanged BKG4/search context, while instrumenting the remaining unexplained per-attempt wall time precisely.

Current failure mode:

```text
same BKG4 generation
+ same family
+ same target/transfer/validation inputs
+ same search config
+ deterministic candidate generation
=
identical search repeated again
```

The first pilot effectively ran 19 attempts while only four family contexts were unique in generation 0/1 cycles.

## 3.2 Required behavior

Introduce an exact **search-context identity** that is sufficient to prove deterministic equivalence of an attempt. At minimum it must bind:

- working BKG4 generation / controller semantic identity;
- family ID;
- relevant target/transfer/validation panel identities;
- search/beam/LNS/reactive configuration identity;
- donor/motif source identity already represented by config/manifest identity;
- relevant checkpoint-selection configuration;
- production config identity.

Do not key it on attempt number, output path, wall clock or provenance-only fields.

For a completed exact context:

```text
same context encountered again
-> do not call the search pipeline again
-> do not consume CUDA games
-> record exact-repeat suppression
-> advance scheduling to genuinely different work
```

If all eligible fixed frontiers in the current generation are exhausted and there is no changed search context, terminate the fixed policy cleanly with an explicit exhaustion reason rather than looping forever.

A newly committed BKG4 generation changes the authoritative evaluation context and makes those frontiers eligible again.

## 3.3 Persistence boundary

The exact-context ledger must survive cloud `--resume` from the last committed state. A restart must not forget that an unchanged frontier was already completed.

Only completed/committed attempt outcomes enter this ledger. `.pending` work never does.

## 3.4 Timing instrumentation

Add exact stage timers around `run_campaign()` and production normalization so the previous ~941-second coarse remainder becomes attributable.

Required aggregate keys should cover at least:

```text
config_validation
donor_event_extraction
checkpoint_selection
host_pool_acquire
motif_preparation
beam_search_total
promotion_pool_selection
target_legal_capture
target_routability
transfer_validation_eval
transfer_validation_capture
full_routability
sealed_eval_capture
cegis
lns_repair
materialization
post_search_io
metrics_normalization
production_summary
```

Names may follow existing conventions, but the final stage accounting must be explicit and machine-readable.

## 3.5 Primary files

Expected implementation surface:

- `tools/bkg4_search/rlrmb/cloud_runner.py`
- `tools/bkg4_search/rlrmb/p13_equal_budget.py`
- `tools/bkg4_search/rlrmb/pipeline.py`
- `tools/bkg4_search/rlrmb/tests/test_cloud.py`
- `tools/bkg4_search/rlrmb/tests/test_pipeline.py`

Do not touch `beam.py` candidate semantics in this phase.

## 3.6 Required tests

Must prove:

1. same generation + same frontier + same config calls the expensive pipeline only once;
2. repeat encounter records a cache/exhaustion event and executes `0` new CUDA games;
3. BKG4 generation change invalidates context-dependent repeat suppression;
4. changed search config invalidates the context;
5. partial/pending attempt never marks a context complete;
6. resume retains completed-context memory;
7. exact-repeat suppression cannot reuse SEALED-dependent generation evidence;
8. stage timers are present and non-negative;
9. total measured stage wall reconciles with attempt wall within a documented small instrumentation remainder;
10. existing cloud atomic-generation tests still pass.

## 3.7 Validation gate

Phase 16.1 passes only when:

- an unchanged four-family fixed cycle cannot execute a fifth redundant full search;
- repeated deterministic contexts consume zero additional CUDA games;
- changed generation/config correctly re-enables search;
- `0` semantic mismatches;
- `0` protected W->L;
- stage timing accounts for at least **95%** of attempt wall on a bounded fixture/probe, or every remaining gap is explicitly identified;
- full RLRMB + CEGIS/composer regressions remain green.

## 3.8 Expected strategic gain

This is expected to produce the largest immediate effective-throughput gain because it stops computing known answers. It should be fixed before CPU parallelization so later speedups accelerate novel work, not duplicate work.

## 3.9 Recommended commit

```text
Stop deterministic Phase16 repeat search
```

---

# 4. Phase 16.2 — Parallelize the actual CPU candidate-construction bottleneck

## 4.1 Purpose

Use the 16-vCPU Lightning host for the measured 65% wall-time bottleneck:

- `generate_seeds()`;
- `expand_program()`;
- structurally safe portions of candidate construction/dedup preparation.

The existing process pool currently accelerates donor motif preparation, which represented only a tiny fraction of cloud wall time.

## 4.2 Required execution model

Keep one authoritative coordinator for:

- CUDA budget authorization;
- QD archive mutation;
- deterministic result ordering;
- promotion ordering;
- BKG4 generation commits.

Workers may construct independent candidate shards.

Recommended task boundaries:

### Seed generation

Partition by checkpoint or another deterministic semantic shard.

```text
checkpoint shard -> worker -> ordered candidate records
```

Coordinator merges and sorts by the same canonical order used by the serial implementation.

### Expansion

Partition by deterministic `(parent semantic ID, expansion operator)` or similarly bounded independent work item.

```text
parent/operator shard -> worker -> generated MacroPrograms
```

Coordinator performs the authoritative merge/dedup/order step.

Do not let worker completion order influence beam survival.

## 4.3 IPC discipline

Prefer compact immutable payloads. Avoid repeatedly pickling giant archive/object graphs.

Workers should receive only what the pure constructor needs, such as:

- checkpoint or compact checkpoint document;
- parent `MacroProgram`;
- operator;
- bounded motif subset/library descriptors;
- max phase/config fields.

Do not introduce shared memory unless measured IPC becomes a new material bottleneck.

## 4.4 QD/archive ownership

The authoritative `Archive` remains coordinator-owned.

Workers do **not** race-write QD cells.

Parallel candidate construction must produce the exact same canonical candidate sequence as the serial path before evaluation/archive insertion.

## 4.5 Primary files

Expected implementation surface:

- `tools/bkg4_search/rlrmb/beam.py`
- `tools/bkg4_search/rlrmb/host_pipeline.py`
- `tools/bkg4_search/rlrmb/p13_equal_budget.py`
- `tools/bkg4_search/rlrmb/tests/test_beam.py`
- `tools/bkg4_search/rlrmb/tests/test_host_pipeline.py`

Operational worker controls may be extended through:

- `tools/bkg4_search/rlrmb/cloud_runner.py`
- `tools/bkg4_search/rlrmb/configs/phase16_lightning_v1.json`

only if needed.

## 4.6 Required tests

For the same bounded inputs, compare worker counts `1`, `2`, and `4` and require identical:

- generated candidate semantic IDs;
- candidate order;
- lineages;
- duplicate counts;
- CUDA candidate sequence;
- EffectVectors;
- QD archive serialization;
- terminal reason;
- game ledger.

Also force out-of-order worker completion and prove canonical output is unchanged.

A worker exception must fail closed before CUDA budget or archive state is advanced.

## 4.7 Local performance gate

Use an identical bounded candidate-construction workload.

Phase passes local engineering only if:

- multiprocessing reduces candidate-preparation wall materially;
- the improvement is not erased by IPC/process overhead;
- serial fallback remains exact;
- no large CUDA campaign is used to prove it.

Do not accept a ~2-5% micro-win. If 4 workers cannot materially beat serial construction, profile IPC/task granularity before moving on.

## 4.8 Lightning performance gate

Run a short cloud probe after local correctness.

Required outcome:

- candidate-generation wall drops by at least **3x** versus the frozen single-thread baseline on comparable unique-family work, or profiling proves a different new bottleneck has taken over;
- multiple vCPUs are measurably active during candidate construction;
- `0` semantic mismatches;
- no search-result change attributable solely to worker count.

Do not require 100% CPU usage. The requirement is that host candidate construction no longer monopolizes one core for most of cloud wall.

## 4.9 Recommended commit

```text
Parallelize RLRMB candidate construction
```

---

# 5. Phase 16.3 — Persistent host resources and generation-scoped invariant caches

## 5.1 Purpose

Remove repeated setup/recomputation that remains after exact-repeat suppression and candidate parallelization.

Inspector evidence showed repeated costs from:

- process-pool creation/destruction;
- legal-state capture cache reset;
- donor event extraction;
- checkpoint selection;
- motif preparation;
- repeated input loading.

## 5.2 Process pool lifetime

Create the host process pool at cloud-run scope, not per attempt.

The pool may live across attempts and frontiers, but workers must remain stateless with respect to attempt-specific strategic state.

Shutdown must be explicit and clean on:

- normal completion;
- infrastructure exception;
- keyboard/termination path where supported.

Do not create a distributed worker service.

## 5.3 Legal-state capture cache

Promote `LegalStateCaptureCache` from attempt-local to a safe broader lifetime.

Cache keys must bind every semantic input that changes capture meaning, including at minimum:

- working asset/controller identity;
- case panel identity/path under the validated bundle;
- probe steps;
- feed/route slots and related capture ABI parameters.

Do not reuse captures across BKG4 generation changes unless the key proves the controller/route assets are identical.

Cross-attempt cache hits must be byte/semantic-equivalent to fresh captures.

## 5.4 Invariant frontier inputs

Cache at the narrowest safe scope:

### Run-level if truly invariant

- donor manifest parse;
- donor route action decoding where immutable;
- static panel metadata.

### Generation/family-level

- `_generic_events()` results;
- deterministic checkpoint selection;
- motif batches whose semantic inputs are unchanged.

Do not cache dynamic candidate evaluation outcomes here; Phase 16.5 owns durable semantic search memory.

## 5.5 Primary files

Expected implementation surface:

- `tools/bkg4_search/rlrmb/cloud_runner.py`
- `tools/bkg4_search/rlrmb/p13_equal_budget.py`
- `tools/bkg4_search/rlrmb/pipeline.py`
- `tools/bkg4_search/rlrmb/host_pipeline.py`
- cloud/pipeline tests.

## 5.6 Required tests

Must prove:

1. process pool is created once for a multi-attempt cloud fixture;
2. same safe capture key reuses prior capture;
3. changed generation/controller identity forces capture miss;
4. changed panel/probe steps force capture miss;
5. donor events/checkpoints are computed once per valid semantic scope;
6. cache results equal fresh results;
7. resume does not trust ephemeral in-memory caches as authoritative state;
8. process workers terminate cleanly;
9. no stale cache changes candidate/routing/CEGIS outcomes.

## 5.7 Performance gate

On a bounded multi-attempt fixture/probe:

- process-spawn overhead should become approximately one pool startup per cloud run;
- repeated legal capture launch count should fall materially when semantic keys repeat;
- repeated donor/event/checkpoint preparation should become negligible;
- no new single stage may silently replace candidate generation as an unexplained P0 bottleneck.

## 5.8 Recommended commit

```text
Persist Phase16 host resources and caches
```

---

# 6. Phase 16.4 — Cloud duty-cycle optimization and CUDA/VRAM tuning

## 6.1 Purpose

Only after Phases 16.1-16.3 remove the dominant host waste, re-profile the complete cloud timeline and optimize the new top bottleneck.

This phase is evidence-driven. Do not assume the old bottleneck ranking remains true after earlier fixes.

## 6.2 Re-profile first

A short Lightning probe must report:

- total cloud wall;
- stage wall breakdown;
- candidate-generation wall;
- legal capture wall;
- CUDA evaluator wall;
- CUDA kernel wall;
- subprocess/context startup wall;
- games/cloud-second;
- candidates/cloud-second;
- GPU utilization timeline;
- GPU 0% fraction;
- VRAM high-water;
- machine CPU utilization;
- process-pool utilization;
- queue/backpressure metrics.

At least 95% of wall must be attributed to named stages.

## 6.3 CUDA batching policy

If CUDA evaluator/context overhead has become material, tune:

- checkpoint batch size;
- max candidates per expansion;
- candidate x case batch dimensions;
- number of evaluator launches.

Use the RTX PRO 6000's 96 GB VRAM only where larger resident batches increase useful throughput.

Do not optimize for maximum VRAM occupancy itself.

## 6.4 Persistent evaluator process

A persistent CUDA evaluator process is allowed only if profiling shows repeated CUDA context/process startup is now a material fraction of wall.

If implemented, it must preserve:

- one controlled GPU owner;
- exact request/result association;
- deterministic ordering;
- explicit crash/fail-closed behavior;
- central budget authorization before each batch;
- identical CPU/CUDA semantics.

Do not create many competing CUDA worker processes on one GPU.

## 6.5 CPU/GPU overlap

Overlap only independent work.

Do not speculatively compute depth `N+1` candidates that semantically depend on depth `N` archive results.

Safe overlap may include:

- preparing independent checkpoint/family inputs;
- serialization of the next already-authorized batch;
- parsing/summary work for the previous batch;
- legal independent preprocessing.

Preserve deterministic commit/order barriers.

## 6.6 Performance gate

Use a frozen performance probe containing only unique search contexts.

Compare against the post-16.3 system, not against the old repetitive 3913-second run alone.

Required outcome:

- cloud throughput improves materially;
- no P0 idle gap remains unexplained;
- GPU is not starved during periods where ready CUDA work exists;
- evaluator launch/context overhead is no longer a leading cost;
- `0` mismatches and protected regressions.

For historical reference, the inspector projected ~3,600-5,700 games/cloud-second after all P0/P1 fixes. Treat that as a hypothesis, not a hard promise. A measured multi-X gain is required; do not falsify metrics to reach the projection.

## 6.7 Low-value work explicitly deferred unless newly justified

Do not prioritize:

- JSON compact formatting;
- telemetry optimization;
- RAM filling;
- multi-GPU orchestration;
- kernel micro-optimization;
- async filesystem frameworks.

## 6.8 Recommended commit

```text
Raise RLRMB cloud search duty cycle
```

---

# 7. Phase 16.5 — Durable semantic search memory

## 7.1 Purpose

Move beyond exact-attempt caching and give the search system a durable semantic memory of what it has tried, what happened, and which regions are exhausted or promising.

This is the foundation for a genuinely cumulative multi-day search.

## 7.2 Memory is not a replay cache

The memory must represent search knowledge, not opponent identity or raw replay memorization.

Core records should include compact semantic/search outcomes such as:

- candidate structural semantic ID;
- family/lane/checkpoint semantic context;
- generation/evaluation-context identity;
- operator/lineage;
- target EffectVector/signature;
- transfer/validation outcome where reached;
- routability classification;
- CEGIS outcome/rejection class;
- repair disposition;
- accepted specialist relation;
- CUDA cost;
- novelty/QD cell information;
- terminal failure class;
- timestamp/order for audit only, never semantic policy.

## 7.3 Context-scoped authority

Distinguish three forms of memory.

### Structural memory — context-independent

Facts such as "this canonical `MacroProgram` semantic ID has already been generated" may persist globally.

### Evaluation memory — context-dependent

Margins, EffectVectors and legal-state/certification outcomes are authoritative only under the exact evaluation-context identity:

```text
working BKG4 semantic identity
+ panel identities
+ simulator/IR/config identity
```

A new BKG4 generation must not blindly reuse old margins as current truth.

### Historical prior memory — cross-generation guidance

Old outcomes may inform future search priority, but cannot be treated as exact current evaluation evidence after the parent changes.

This distinction is mandatory.

## 7.4 Persisted knowledge classes

The system must remember at least:

- evaluated structural candidates;
- exact effect signatures;
- QD/effect cells observed;
- routability failures;
- non-transferable candidates;
- FUTURE_DEPENDENT terminal basins;
- bad checkpoint/bad horizon evidence;
- repair successes/failures;
- accepted capabilities;
- operator/motif statistics;
- exhausted exact semantic regions.

## 7.5 Storage choice

Choose the smallest storage system that survives expected campaign scale.

Preferred decision procedure:

1. estimate record volume for a multi-day campaign;
2. benchmark compact append/query requirements;
3. use canonical JSON/JSONL if it remains comfortably small and queryable;
4. use Python stdlib `sqlite3` if indexed lookups, transactional resume and millions of records justify it.

Do not introduce an external database service.

If SQLite is used:

- schema must be versioned;
- write transactions must commit only completed semantic records;
- accepted-generation state remains authoritative in `generation_state.json`/generation artifacts;
- memory corruption/failure must fail closed rather than silently bias search;
- emit a compact canonical summary for audit.

## 7.6 Memory-driven duplicate avoidance

Before expensive generation/evaluation, the system should be able to answer:

```text
Have we already generated this semantic candidate?
Have we already evaluated it in this exact context?
Did this basin repeatedly fail for a terminal reason?
Is this effect signature already represented?
```

Actions differ by answer:

- exact evaluated same-context candidate -> reuse/skip CUDA;
- structurally seen but new evaluation context -> candidate may require reevaluation;
- terminal FUTURE_DEPENDENT same-context basin -> suppress unless context changes materially;
- duplicate effect but strategically simpler/routable alternative -> allow QD policy to decide, do not blindly suppress all same-effect candidates.

## 7.7 Resume contract

Memory must survive restart consistently with the last committed generation.

Never persist a partial authoritative candidate result before its evaluator/result association is complete.

The memory may contain completed rejected search evidence beyond accepted BKG4 generations, but its transaction boundary must make crash semantics explicit.

## 7.8 SEALED firewall

SEALED outcomes must not become proposal/training features in search memory.

Certification may store a downstream audit fact that a frozen candidate passed/failed final certification, but adaptive generation must never read SEALED-derived labels.

Add explicit tests for this firewall.

## 7.9 Primary files

Expected new module:

- `tools/bkg4_search/rlrmb/search_memory.py`

Likely integration:

- `cloud_runner.py`
- `p13_equal_budget.py`
- `beam.py`
- `archive.py`
- `failures.py`
- new `tests/test_search_memory.py`

Do not overload `cloud_runner.py` with the entire memory implementation.

## 7.10 Validation gate

Phase passes only when:

1. same-context evaluated semantic candidates are not reevaluated;
2. new BKG4 generation invalidates exact outcome reuse while retaining historical priors;
3. failure classes persist and query correctly;
4. restart reproduces the same memory-visible state;
5. insertion/query order does not alter search semantics;
6. SEALED cannot leak into generation memory;
7. memory compaction/size is bounded enough for a multi-day run;
8. `0` exactness regressions;
9. bounded search demonstrates a materially lower repeated-semantic evaluation rate than Phase 16.4.

## 7.11 Recommended commit

```text
Add persistent RLRMB semantic search memory
```

---

# 8. Phase 16.6 — Adaptive self-improving search allocation

## 8.1 Purpose

Use accumulated search memory to choose **where to search next and how**, rather than fixed round-robin frontier cycling.

This is the phase where RLRMB becomes a self-improving search controller.

It does not replace RLRMB candidate semantics, CEGIS or Bonsai. It improves search allocation.

## 8.2 Initial policy should be deterministic and interpretable

Do **not** begin with a neural controller.

Start with a deterministic evidence-driven allocator using statistics from Phase 16.5.

The allocator may score search opportunities using factors such as:

- unresolved loss/regret severity;
- semantic novelty opportunity;
- historical target L->W yield;
- transfer/routability survival rate;
- CEGIS survival rate;
- new effect/QD-cell yield;
- CUDA games spent;
- repeated duplicate rate;
- terminal-failure rate;
- basin exhaustion;
- operator/motif yield;
- recency/generation change;
- exploration bonus for insufficiently tested regions.

The objective should approximate:

```text
expected new safely solved regimes / GPU-hour
```

not average margin.

## 8.3 Frontier allocator

Replace unconditional fixed round-robin selection with a deterministic priority queue/policy over eligible search opportunities.

A frontier that is exact-context exhausted should receive no budget until one of these changes:

- BKG4 generation;
- search neighborhood/operator allocation;
- new hard-case evidence;
- new legal checkpoint/motif context;
- explicit exploration rule justifies a broader/new neighborhood.

Do not merely add randomness to make repeated attempts different.

## 8.4 Operator/motif adaptation

Track which operators/motifs produce:

- novel structures;
- novel effects;
- target L->W;
- transfer-safe candidates;
- routable candidates;
- CEGIS entries/acceptances;
- repeated terminal failures.

Use these statistics to allocate expansion budget while maintaining an exploration floor for under-tested valid operators.

A simple deterministic UCB-style or optimistic score is acceptable if evidence supports it. Avoid stochastic policy choices unless reproducibility is explicitly preserved through a semantic seed contract.

## 8.5 Basin exhaustion

Introduce explicit bounded basin state.

Examples:

- `UNSEEN`
- `ACTIVE`
- `LOW_YIELD`
- `EXHAUSTED_EXACT_CONTEXT`
- `TERMINAL_FUTURE_DEPENDENT`
- `REPAIR_PENDING`
- `REOPENED_NEW_GENERATION`

Names may differ, but the system must stop repeatedly spending budget in regions whose deterministic neighborhood has been exhausted.

Exhaustion is contextual, not eternal. A stronger parent or newly discovered motif/checkpoint can reopen a region.

## 8.6 Failure-driven policy

Preserve existing repair taxonomy and use it as search-control feedback:

```text
BAD_CHECKPOINT          -> alternative checkpoint allocation
BAD_HORIZON             -> boundary/horizon search
NON_TRANSFERABLE        -> semantic LNS / different motif or mask neighborhood
REPAIR_COMMON_PREFIX    -> common-prefix repair allocation
REPAIR_BRANCH_AT_REVEAL -> bounded reactive allocation
REPAIR_ABORT_RECOVERY   -> recovery allocation
FUTURE_DEPENDENT        -> suppress exact basin
TARGET_CAPABILITY_LOST  -> target-preserving repair
```

Do not weaken terminal classifications merely to keep a basin active.

## 8.7 No holdout contamination

Adaptive statistics may use generation/search evidence from TARGET, TRANSFER and VALIDATION according to existing roles.

SEALED remains unavailable to allocator scoring.

## 8.8 Primary files

Prefer a separate focused module such as:

- `tools/bkg4_search/rlrmb/search_policy.py`

with memory/query interfaces from `search_memory.py`.

Likely integration:

- `cloud_runner.py`
- `p13_equal_budget.py`
- `beam.py`
- tests under `rlrmb/tests/`.

Do not embed a large adaptive policy directly into config parsing.

## 8.9 Required synthetic tests

Construct deterministic histories proving:

1. productive frontier/operator receives more budget than repeatedly sterile one;
2. unseen region still gets exploration budget;
3. exact exhausted basin gets zero repeated budget;
4. generation change can reopen stale context;
5. terminal FUTURE_DEPENDENT receives no exact repeat budget;
6. NON_TRANSFERABLE redirects toward valid repair rather than global abandonment;
7. changing record insertion order does not change allocator decision;
8. restart produces same next search decision;
9. SEALED data cannot affect score;
10. adaptive allocation never modifies acceptance thresholds/gates.

## 8.10 Equal-budget cloud comparison

Run fixed-policy and adaptive-policy bounded searches with the same CUDA game ceiling on comparable unresolved frontier evidence.

Primary comparison metrics:

- new semantic candidates not previously evaluated;
- new effect signatures/QD cells;
- target-solving effect clusters;
- transferable/routable candidates;
- safely solved regimes;
- protected W->L;
- semantic mismatch;
- cloud/GPU hours.

Phase 16.6 requires a **material** gain. A useful default gate is at least a 25% improvement in one primary novelty/solved-regime productivity metric with no material regression in the others and no safety regression. If evidence suggests a better threshold, document it before the comparison rather than moving the goalpost afterward.

Accepted-specialist count is valuable but too sparse to be the only gate for a short comparison.

## 8.11 Recommended commit

```text
Adapt RLRMB search from accumulated experience
```

---

# 9. Phase 16.7 — Multi-generation stress validation and large-campaign readiness

## 9.1 Purpose

Prove that the optimized, memory-backed adaptive system can operate for hours, keep making novel progress, survive restarts, and repeatedly improve the evolving BKG4 without semantic or protected-regression failure.

This is the final gate before serious large-scale autonomous campaigns.

## 9.2 Required validation sequence

### A. Local bounded correctness

Run:

- full RLRMB unit/regression suite;
- hardened CEGIS/composer tests;
- search-memory tests;
- adaptive-policy tests;
- multiprocessing determinism tests;
- CPU/CUDA parity;
- materialized BKG4 parity;
- bundle isolated self-test;
- config validation.

No large local search.

### B. Short Lightning throughput probe

Confirm actual hardware behavior after all optimizations.

Report:

- games/cloud-second;
- candidates/cloud-second;
- stage wall fractions;
- CPU candidate generation speedup;
- GPU activity/idle timeline;
- VRAM high-water;
- duplicate/avoided evaluation rate;
- memory query/write overhead;
- new semantic/effect yield.

### C. Bounded multi-hour adaptive pilot

Run a real unresolved-frontier search long enough to exercise:

- multiple frontier decisions;
- memory updates;
- basin exhaustion/reopening;
- at least one resume/restart drill;
- BKG4 generation advancement if safe specialists are discovered.

Do not require the system to fabricate an acceptance merely to pass infrastructure validation.

### D. Multi-generation mechanics stress

Use deterministic fixtures to prove many sequential generation commits even if real search only finds one specialist during the pilot.

For real cloud evidence, prefer naturally achieving at least two consecutive accepted-generation transitions before a multi-day campaign. If strategic scarcity prevents this, the report must separate:

- mechanical multi-generation correctness;
- search quality/yield.

Do not weaken acceptance to force G2/G3.

## 9.3 Large-campaign readiness criteria

The system is `LARGE_CAMPAIGN_READY` only if all are true:

1. no exact deterministic frontier repetition;
2. persistent search memory survives resume;
3. candidate generation uses host multiprocessing materially;
4. no dominant unexplained cloud wall-time gap remains;
5. adaptive allocator spends budget on new/eligible semantic territory rather than fixed cycling;
6. same-context duplicate CUDA evaluations are rare and explained;
7. CPU/CUDA/BKG4 mismatches = `0`;
8. protected W->L = `0` for committed generations;
9. SEALED generation use = `0`;
10. pending attempts cannot advance BKG4;
11. accepted specialists automatically materialize/compose into the next parent;
12. memory/context invalidation is correct across BKG4 generations;
13. restart from last committed generation is deterministic;
14. cloud artifacts remain compact enough for multi-day execution;
15. throughput is materially multi-X better than the original ~508 games/cloud-second baseline on a comparable unique-search probe;
16. search productivity is measured primarily by novel/safely solved regimes per GPU-hour, not raw repeated games.

## 9.4 Promotion boundary remains separate

Even after Phase 16.7, a large campaign's final candidate is not automatically the live champion.

Actual promotion requires:

```text
broad protected CUDA regression
-> 0 protected W->L
-> 0 CPU/CUDA/BKG4 mismatch
-> 64/128 official Python-engine canary
-> deliberate live champion promotion
```

## 9.5 Final verdicts

Phase 16.7 report must use one of:

```text
RLRMB LARGE CAMPAIGN READY
RLRMB LARGE CAMPAIGN NOT READY
```

If not ready, identify the single highest remaining blocker rather than inventing another broad redesign phase.

## 9.6 Recommended commit

```text
Validate RLRMB adaptive multi-generation scaling
```

---

# 10. Permanent metrics added by this roadmap

The final system should report these in addition to existing RLRMB metrics.

## Exact-repeat / memory

- search-contexts considered;
- exact contexts executed;
- exact-repeat contexts suppressed;
- semantic candidates skipped from same-context memory;
- avoided CUDA games;
- memory hits/misses by knowledge class;
- evaluation-context invalidations;
- exhausted basins;
- reopened basins.

## Host parallelism

- candidate-generation tasks;
- worker busy seconds;
- candidate-generation wall;
- coordinator merge/dedup wall;
- IPC/serialization wall where measurable;
- worker count;
- worker pool creations;
- worker failures.

## Cloud duty cycle

- named stage wall seconds;
- accounted wall percentage;
- CUDA evaluator wall;
- CUDA kernel wall;
- evaluator launches;
- context/startup wall where measurable;
- games/cloud-second;
- candidates/cloud-second;
- GPU idle while ready work exists;
- GPU utilization timeline summary;
- VRAM high-water;
- aggregate CPU utilization.

## Adaptive policy

- frontier/basin scores;
- budget allocated by frontier/operator;
- exploration vs exploitation allocation;
- novel structural yield / 100k games;
- novel effect yield / 100k games;
- transfer/routability survival by operator;
- CEGIS progression by operator/basin;
- safely solved regimes / GPU-hour.

Metrics must remain compact; do not log one verbose timing document per candidate.

---

# 11. Falsification / stop rules

This roadmap is not permission to keep adding infrastructure indefinitely.

Stop and reconsider if:

- exact-repeat suppression changes strategic results for identical contexts;
- candidate multiprocessing cannot beat serial execution after task-granularity/IPC tuning;
- memory lookups become a new dominant bottleneck;
- adaptive search improves novelty but materially reduces transferable/safely routable yield;
- search-memory logic starts using stale cross-generation margins as current truth;
- SEALED leaks into generation decisions;
- a phase causes any semantic mismatch;
- a phase creates protected W->L in committed BKG4;
- complexity grows without a measured multi-X or strategic-productivity gain.

When a phase fails its gate, repair that phase or revise its design with evidence. Do not skip forward.

---

# 12. Phase-by-phase execution protocol

For every future implementation session:

1. Read this file and the two authoritative master plans.
2. Verify branch/HEAD/status.
3. State the exact phase being executed.
4. Inspect the actual current implementation; do not rely only on this roadmap.
5. Write/extend focused tests before changing behavior.
6. Implement only the current phase.
7. Run the cheapest falsifying tests first.
8. Run full RLRMB/CEGIS/parity gates before claiming completion.
9. Use only bounded local performance work.
10. If the phase requires real hardware proof, build the updated Lightning bundle and provide the exact short cloud command.
11. Record before/after metrics against the frozen or immediately previous baseline.
12. Commit at the phase gate.
13. Leave the repository clean except the five protected pre-existing untracked files.
14. Stop. Do not begin the next phase in the same implementation session unless the user explicitly asks.

---

# 13. Recommended phase commit sequence

```text
16.1  Stop deterministic Phase16 repeat search
16.2  Parallelize RLRMB candidate construction
16.3  Persist Phase16 host resources and caches
16.4  Raise RLRMB cloud search duty cycle
16.5  Add persistent RLRMB semantic search memory
16.6  Adapt RLRMB search from accumulated experience
16.7  Validate RLRMB adaptive multi-generation scaling
```

Do not push unless explicitly requested.

---

# 14. Strategic end state

After Phase 16.7, the search loop should no longer resemble:

```text
pick family 0
search deterministic neighborhood
forget
pick family 1
search deterministic neighborhood
forget
...
cycle back
repeat identical work
```

It should behave as:

```text
current BKG4 generation
    |
    v
persistent search memory
    |
    +--> what is already known/exhausted?
    +--> what failed and why?
    +--> what operators/motifs worked?
    +--> what semantic/effect regions remain novel?
    |
    v
adaptive deterministic allocator
    |
    v
parallel candidate construction on CPU
    |
    v
broad exact CUDA evaluation
    |
    v
QD / routability / transfer / CEGIS / SEALED
    |
    +--> rejection/failure updates memory
    |
    +--> safe specialist
              |
              v
        compose BKG4 generation N+1
              |
              v
        invalidate context-dependent truth
        retain historical search experience
              |
              v
        search the new frontier intelligently
```

The intended outcome is a **cumulative autonomous capability factory**: each cloud hour should add new information even when it does not add a specialist, and every accepted specialist should make the next search context stronger rather than causing the system to forget what it learned.

That is the post-pilot Phase-16 path from a working RLRMB pipeline to a robust large-scale search system suitable for the ReCurSiON first-place attempt.
