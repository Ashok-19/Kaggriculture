# Dual-T4 Kaggriculture GPU-Resident Memetic Meta Search

`notebooks/kaggle_dual_t4_meta_search.ipynb` is a 10-hour, two-T4 discovery pipeline for compact Bonsai-compatible strategic programs. The long search is no longer a Python-orchestrated generational GA.

## Architecture

There are exactly two independent top-level search universes:

- **GPU0 / exploit:** strongly anchored to verified 4096-byte historical Bonsai structural priors, lower mutation amplitude, larger local-refinement lanes and slower restarts.
- **GPU1 / explore:** receives only one weak anchor, then uses substantially stronger structural mutation and faster stagnation resets to seek a different economic basin.

Each top-level process owns one `CUDA_VISIBLE_DEVICES` value and never communicates with the other during the primary search. There is no NCCL, peer-to-peer population exchange, shared population lock, or periodic cross-GPU merge. The two result sets meet only in the final screen.

Python is a supervisor. One compiled `bonsai_memetic_search` process per GPU owns the expensive loop:

```text
resident population
    -> exact CUDA simulation
    -> on-device family/regression fitness
    -> QD niche archive
    -> on-device island selection
    -> stagnation / duplicate-signature reset
    -> module crossover + structural mutation
    -> GPU local-neighborhood refinement lanes
    -> next resident generation
```

There is no per-generation Python callback and no reward/fitness round trip to Python. CUDA launches for multiple generations are queued from compiled C++; the host synchronizes only at sparse epoch/checkpoint boundaries measured in minutes.

## Compact search representation

The hot search uses `SearchGenome`, statically fixed at **256 bytes**. It is not JSON, a pointer tree, or a 719-action replay. It contains strategic degrees of freedom only:

- phase timing;
- compact farm-layout weights and layout seeds;
- land/labor/livestock/cash targets;
- phase-specific execution mode and actuator route choices;
- planner horizons and buffers;
- market floors/reserves/liquidation controls;
- task priorities;
- up to four compact public-observation reactive rules.

Every genome expands deterministically to the existing `GPManager` execution backbone and distills deterministically to a standard **4096-byte `GPProgram`**. Movement legality, watering, feeding, harvesting, shed interaction and other execution mechanics remain the shared competent Bonsai primitives; evolution searches strategy around them rather than rediscovering low-level legality.

The compact quantization is a search-space parameterization, not a simulator approximation. Promoted candidates are always evaluated with the exact CUDA Kaggriculture simulator.

## Official-engine exploitation axis

Engine exploitation is a primary search axis, not a side experiment. The bundle contains the exact official `kaggriculture.py` used to derive these mechanics and locks its SHA-256 in `bundle_manifest.json`; the notebook aborts if that source changes. Code is authoritative over documentation.

The current engine-aware grammar exposes only legal current observations and exact deterministic consequences of the official implementation:

- shared market inventory and current market prices;
- our private shed inventory;
- both farms' public crop/animal/geometry exposure;
- current unlocked shop instances, including duplicates;
- market queue ordering, batch size and relative withholding pressure;
- the special `$1` SELL branch, where a successful sale pays/removes our stock but does not increase shared market supply;
- town-consumption timing after `_process_market`;
- empty-space/geometry pressure that can legally alter end-of-day weed RNG consumption and therefore the later shop draw.

No runtime gene can access opponent-private inventory, episode/replay identity, the seed, future RNG, or future shops. Geometry/RNG effects are discovered only through exact offline simulation and expressed at runtime as deterministic farm structure.

The 256-byte genome dedicates compact bytes to an engine profile, market batch/order bias and packed relative-withholding/empty-space pressure. New engine-aware leaf kinds preserve the fixed 4096-byte `GPProgram` ABI. Search-vs-distilled parity is forced through these engine leaf modes before either long T4 run begins.

Sub-island allocation is intentional:

- GPU0/exploit: **12 of 32** default sub-islands are engine-mechanics refinement lanes;
- GPU1/explore: **20 of 32** default sub-islands are de-novo/adversarial engine-mechanics lanes.

The hot-loop objective remains relative. Rewards are candidate money minus opponent money; family loss/regression/downside constraints dominate ranking. Each offspring also carries the common-case mean of its parent/reference, so the GPU computes a `marginal_delta = offspring margin - parent margin` on the same family-balanced hard core. This is only a downstream tie-breaker after safety criteria and suppresses changes that merely raise both players together.

## Exactness gates

Simulation remains pinned to repository semantics for `kaggle-environments==1.32.7`. Builds use `-O3` and never use `--use_fast_math`.

Before either long-lived T4 process starts, the notebook requires both:

1. the existing BKG3-v6 CUDA fingerprint/tick smoke in `v6_smoke_reference.json`;
2. a full-reward parity gate comparing a `SearchGenome` executor against its distilled standard `GPProgram` on the core development cases.

Any mismatch aborts the notebook.

The independent `bonsai_search` executable remains the final-screen evaluator. This gives the final candidates an independent exact `GPProgram` execution path rather than trusting only the search interpreter.

### Replay-faithful seed policy

Primary fitness no longer uses one global simulation seed. Every development tape carries its canonical `source.seed`, materialized as `source_seed` in `dev_manifest.tsv`, and each `TierCase` evaluates both candidate seats under that tape's recorded seed. The 168-tape bank currently contains **152 unique recorded seeds**; the 12-tape hot core contains **12 distinct seeds**, so the engine-geometry/RNG search cannot optimize one shared shop/weed trajectory without increasing game count.

The incumbent regression reference is seed-consistent as well. `baseline_dev_recorded_seed.tsv` contains **336 BKG3-v6 historical CUDA mode-0 cases** (168 tapes × both seats) evaluated on the same per-tape recorded seeds. Bundle construction, resident loading and final merge all reject any baseline row whose seed disagrees with the corresponding development tape. The old 2184000-only baseline is not used for v6 fitness/regression decisions.

## Development data boundary

`prepare_input_bundle.py` builds a development-only Kaggle Dataset and asserts:

- 168 development tapes / 21 families;
- 84 frozen validation tapes excluded;
- 50 blind continuity tapes excluded;
- 28 family-holdout tapes excluded;
- Kawashigi and Ueddy present in the hard core;
- 152 unique recorded development seeds and 12/12 distinct Stage-1 seeds;
- 336 seed-consistent BKG3-v6 baseline/regression cases;
- no forbidden tape content hash copied into the bundle;
- no fast-math setting.

The entire allowed corpus is parsed once before the resident search clock starts, encoded as native actions/descriptors and copied independently to each GPU. No JSON replay parsing occurs in the hot loop.

## Device-resident multi-fidelity search

Every resident candidate is repeatedly screened on the exact **core12 × both seats** evidence through step 120, with each tape evaluated under its own recorded `source.seed`. This is the high-volume search fitness and includes both hard families and seed-consistent incumbent-win regression guards. Replay-faithful evidence exposes 19 baseline-losing development families, so the 12-family hot core reserves Kawashigi/Ueddy and fills the remaining slots by loss count/downside severity; the 21/42/168 promotion tiers retain complete family coverage.

Broader evidence is deliberately **not** run over the million-scale resident population. The 256-entry quality-diversity archive is the promotion boundary:

- every 8 generations: QD elites -> one representative from all 21 families, both seats, step 240;
- every 32 generations: QD elites -> 42 tapes, both seats, step 480;
- every 128 generations: QD elites -> all 168 development tapes, both seats, full step 720.

Certified higher-fidelity niche elites are stored separately and are preferred restart material after stagnation. Thus expensive evidence feeds back into search without turning a full-development pass into the dominant workload.

The final ranking objective is family/downside oriented, not mean-margin oriented: fewer losing families, fewer losses, fewer incumbent-win regressions, stronger worst-family result, stronger floor, then global wins/margin.

## Sub-islands, diversity and local refinement

Each GPU maintains exactly **32 production logical sub-islands**; autotuning selects launch geometry but cannot change this strategic topology. Every sub-island has independent RNG state, a local champion, stagnation count and reset history. GPU0 reserves 12/32 for engine mechanics and GPU1 reserves 20/32.

Diversity has two layers:

- selection breaks fitness ties toward less crowded strategic niches;
- a 256-cell quality-diversity archive hashes strategic descriptors such as land/livestock/cash/execution/market regimes.

A compact strategic behavior signature detects duplicate champions. Repeated stagnation or duplicate convergence triggers a strong island kick. Reset sources prefer higher-fidelity certified QD niches, then ordinary QD niches, then structural priors; this avoids repeatedly reseeding every island from the global champion.

Local policy relaxation is discrete and GPU-side. A configurable prefix of each sub-island population is generated as small one-/two-coordinate neighborhoods around the current local elite. The rest of the population performs module-level crossover and broader mutation. GPU0 allocates more slots to these local neighborhoods; GPU1 allocates fewer and uses stronger structural changes.

## VRAM policy

`VRAM_TARGET` defaults to **0.95** of safely available memory, not 95% of nominal card bytes. It is an upper safety bound, **not** a population-size target. Each resident engine:

1. creates the CUDA context and immutable simulator lookup tables;
2. loads development routes/corpus;
3. queries live free VRAM;
4. subtracts configurable emergency headroom (default 512 MiB);
5. receives an explicit production population derived from measured exact games/sec and the target generation latency;
6. verifies that population fits inside the safe VRAM budget and backs off only if allocation itself fails.

The memory is used for resident genomes, decoded manager caches, double candidate populations, simulator states, reward buffers, per-family accumulators, RNG state, island state, QD archives and promotion/certification workspaces. Allocating unused VRAM is not a performance objective: once the T4 is compute-bound, excessively large resident populations only lengthen the evolutionary feedback loop and can defeat progress watchdogs.

The notebook parses `RESIDENT_MEMORY` from each process and aborts if the resident population differs from the per-GPU tuned production population or exceeds the safe memory bound. A separate utilization burst proves both T4s are actually compute-bound. Real T4 allocation and throughput values are reported only after the Kaggle run; local-GPU numbers are not substituted.

## T4 autotuning

Each T4 is calibrated independently **and concurrently** before the primary search. `autotune.py` benchmarks one representative 8192-candidate / 32-sub-island exact core generation at each supported block size and selects the highest measured games/sec configuration. The 8192 batch is deliberately the smallest calibration population that preserves the production 32-island topology. Production population is then derived from measured games/sec, the target generation latency, and a conservative safety factor, rounded to a 32×256 candidate quantum and capped by safe VRAM. Each calibration trial has a hard timeout. It records:

- T4 model and free/total memory;
- compute capability;
- threads/block;
- exact games/sec and candidates/sec;
- 256-byte genome / manager / simulator sizes;
- safe-memory budget;
- estimated full-resident population and core-generation time;
- throughput-targeted production population and its predicted core-generation time;
- production allocation fraction of the safe VRAM budget.

Small benchmark populations are only for launch-geometry throughput measurement. Production uses the explicit throughput-targeted population, not `population=0` full-VRAM autosizing. The watchdog derives a per-GPU stale limit from that predicted production generation time, with the configured stale window retained as a minimum floor.

CUDA Graphs/persistent single-kernel execution are intentionally not assumed to be wins. The first optimized implementation already removes Python/host fitness orchestration and queues repeated CUDA work in compiled C++. Further graph/kernel fusion should be driven by real T4 profiling, register pressure and launch-overhead evidence rather than sophistication for its own sake.

## Checkpoint and logging cadence

Resident processes emit one compact heartbeat after an epoch and checkpoint roughly every three minutes when generation duration allows. Checkpoints contain sub-island champions/search counters, QD niche programs and multi-fidelity certified niche programs, not full simulator populations.

SIGTERM/SIGINT is handled as a graceful-stop request: the engine finishes the current safe epoch, writes the final compact checkpoint/archive and exits. Resume is therefore **elite/QD continuation**, not a bit-exact restoration of the entire resident population/RNG stream.

Notebook telemetry is sampled every five minutes using `nvidia-smi` and records GPU utilization, memory-controller utilization, VRAM, power and temperature. The warm-up gate is followed by a long-run watchdog: checkpoint generation progress has a 20-minute stale limit, while GPU idleness requires three consecutive monitor cycles whose three-sample utilization burst remains below 5%. A single transient 0% sample never triggers a restart. Each universe may be gracefully terminated and resumed from its compact checkpoint **once**; a second stall fails the run rather than wasting the remaining GPU budget. There is no per-game or per-candidate notebook output.

## Absolute 10-hour lifecycle

At Cell 1 the notebook records one monotonic `HARD_EXIT_DEADLINE = start + 10h`.

Setup performs hardware probe, bundle audit, compilation, exactness gates and T4 autotuning. Search receives only the walltime left after reserving:

- 15 minutes for graceful shutdown/orchestration;
- 25 minutes for final screening and export, including a protected 3-minute export guard.

Both search engines have independent internal monotonic deadlines. Near the search-stop deadline the notebook requests graceful process-group shutdown; no manual action is required.

The final exact screen canonical-deduplicates exported `GPProgram`s and shards candidate programs between GPU0 and GPU1. It remains replay-faithful: all 168 tapes × both seats use each tape's recorded seed. Before the full screen, both T4s run a small full-720-step evaluator probe; measured aggregate games/sec and remaining wall time determine how many candidates can safely finish, with a conservative safety factor and a requirement to preserve both search universes. Every evaluator runs in its own process group under an absolute deadline. If screening or its throughput probe overruns, the whole child group is terminated before the protected export guard. Artifact compression is separately deadline-bounded, so the 10-hour limit is enforced inside the final-screen/export subprocesses rather than checked only after they return.

## Input bundle

Build locally:

```bash
python3 tools/meta_search/prepare_input_bundle.py \
  --output scratch/kaggriculture_meta_search_bundle \
  --force
```

The bundle includes:

```text
cuda-engine/src/bonsai_memetic_search.cu
cuda-engine/src/bonsai_search.cu
cuda-engine/include/...
bonsai-kg/...
actuators/...
populations/...
priors/*.bin
tools/meta_search/...
bundle_manifest.json
```

The three bundled `priors/*.bin` files are verified fixed-size historical 4096-byte `GPProgram` records. They are structural initialization priors, not claims of being the current champion.

## Kaggle setup

1. Attach the private Dataset `ashok205/kaggriculture-meta-search`, latest version.
2. Import `notebooks/kaggle_dual_t4_meta_search.ipynb`.
3. Select **GPU T4 ×2**.
4. Run all cells in order.

The notebook recognizes both common Kaggle input layouts, including:

```text
/kaggle/input/datasets/ashok205/kaggriculture-meta-search
```

and falls back to recursively locating a unique `bundle_manifest.json`.

## Outputs

```text
/kaggle/working/meta_search/
  config.json
  hardware.json
  utilization_samples.jsonl
  island0/
    tuning.json
    command.json
    resident_search.log
    resident_checkpoint.bin
    checkpoint.json
    subisland_*_best.bin
    niche_*_best.bin
    niche_archive.jsonl
    certified_niche_*_L*.bin
    certified_archive.jsonl
    resident_summary.json
    summary.json
  island1/
    ...same independent files...
  merged/
    deduplicated_elites.jsonl
    final_screen_commands.json
    final_screen_gpu0.log
    final_screen_gpu1.log
    final_screen_results.jsonl
    nondominated.jsonl
    top_candidates.jsonl
    top_candidates/*.bin
    summary.json
  final_report.json
```

The notebook packages all of this as:

```text
/kaggle/working/kaggriculture_dual_t4_meta_search_results.tar.gz
```

The discovery run does not touch frozen validation, modify the champion, push Git, or submit to Kaggle. Any candidate considered for promotion must still be reproduced locally in exact CUDA/C++, then tested on the required 64/128-game official Python-engine canary before submission work begins.
