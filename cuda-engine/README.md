# Kaggriculture CUDA engine

Native CUDA C++ port of the pinned Kaggriculture 1.32.7 simulator. The runtime simulator has no Python bridge and no CPU simulation fallback: episode state, CPython-compatible MT19937, market/town processing, farm transitions, day refresh, snapshots, replay-tape execution, and device policies execute in CUDA device code.

A normal native C++ host process still performs unavoidable CUDA orchestration (file parsing, allocations, kernel launches, result copies). It does **not** simulate game turns on the CPU. The proven CPU/Python engines are development oracles only and are not linked into the production CUDA runners.

## Target

- GPU used for development: NVIDIA GeForce RTX 3050 Laptop GPU, 4 GiB, compute capability 8.6.
- Driver: 595.84.
- Default compiler: `/usr/bin/nvcc`, CUDA 11.5.
- Default target: `sm_86`.
- Pinned game semantics: kaggle-environments 1.32.7 / Kaggle commit `28b6d8af3ce73926b3d0fda1410c1ddd8384ab8c`.
- Fast math is intentionally forbidden. Integer market prices depend on exact-enough double transcendental/rounding behavior and are exhaustively parity-tested.

## Layout

- `include/pyrandom.cuh`: CPython-compatible MT19937 plus exact cached-day stream reader.
- `include/sim.cuh`: complete device simulator, snapshots, and GPU-resident market tables.
- `include/runtime.cuh`: CUDA allocation/table initialization/error handling; no CPU simulator.
- `include/tape_io.hpp`: dependency-free parser for the established replay-tape JSON schema (`farmer`/`hands`/`market`); no Python or third-party JSON runtime.
- `include/observation.cuh`: zero-copy legal seat-specific observation views. Both farms' public state is visible; only the acting player's shed, seeds, and worker inventories are exposed privately.
- `include/policy_runner.cuh`: generic device-only full-episode and snapshot policy rollout templates for rule policies, learned selectors, or neural inference code.
- `include/logical_hash.hpp`: ABI-independent logical-state fingerprint used by development parity tools.
- `src/validate.cu`: complete reference episodes in one GPU kernel against pinned official checkpoints.
- `src/price_probe.cu`: batched CUDA market-price probe for exhaustive 1.32.7 parity.
- `src/state_parity.cu` / `src/cpu_state_hash.cpp`: development-only deep logical-state comparison against the proven CPU oracle.
- `src/snapshot_validate.cu`: cross-kernel save/load/resume validation at causal boundaries including 72/120/144/168, for cached and uncached RNG paths.
- `src/bench.cu`: one-thread-per-environment full-episode throughput benchmark.
- `src/branch_bench.cu`: synthetic GPU snapshot continuation grid benchmark.
- `src/tape_bench.cu`: native full-episode replay-tape cohort runner with current-state worker padding/truncation semantics.
- `src/tape_branch.cu`: native base-prefix + opponent + arbitrary candidate-tape branching across seed snapshots; supports `@candidate-list.txt` and reports wins, mean margin, and floor.
- `src/policy_smoke.cu`: all-device legal-observation policy rollout smoke test.
- `tools/run_state_parity.sh`: deterministic deep-state comparison harness.

Generated binaries/profiles/disassembly live under `scratch/cuda-engine-build/` and are not source artifacts.

## Build

```bash
make -C cuda-engine -j2
```

The project hard two-worker rule still applies to CPU simulation/research processes. CUDA kernels intentionally use the GPU in parallel; do not combine them with broad CPU multiprocessing sweeps.

## Correctness gates

Run the full suite with:

```bash
make -C cuda-engine check
```

Equivalent individual gates:

```bash
make -C cuda-engine price-parity
make -C cuda-engine validate-parity
make -C cuda-engine state-parity
make -C cuda-engine snapshot-parity
make -C cuda-engine policy-smoke
```

The frozen-engine proof established during development is:

- **270,009 / 270,009** integer market-price states exact against pinned kaggle-environments 1.32.7 (9 products, inventory -10000..20000).
- **15 / 15** complete 719-transition official-reference episodes exact.
- **10,800 / 10,800** logical state checkpoints exact (15 traces × 720 states), including meaningful farm tiles, worker positions/inventories, private shed/seeds, accounting, market, shops, and temporal fields.
- Snapshot save/load/resume exact at **16 boundaries** per trace (`0,1,23,24,71,72,119,120,143,144,167,168,300,600,718,719`) across all 15 traces.
- Both ordinary and cached-RNG snapshot resumes land on the identical logical final state.
- Native live-tape execution matched the proven CPU compiled-tape engine seed-for-seed on Kawashigi vs Hax for seeds 62000..62004.
- Real live split/resume self-continuations reproduced continuous results at prefixes 72, 120, 168, 300, and 600, covering both candidate seats.
- The device-policy legal-observation smoke gate runs 256 pass-vs-pass episodes entirely through the GPU policy API and ends exactly at `(3000,3000)`.

`price-parity` invokes the existing Python checker only as a development oracle. `state-parity` builds a separate CPU-oracle hash emitter only for comparison. Neither Python nor the CPU simulator is linked into the CUDA runtime/benchmark/research executables.

### Important exactness implementation details

- Replay tapes are padded/truncated every turn to the **current simulated farm's worker count**, not the worker count recorded in the source replay. This matches the proven CPU replay semantics on divergent seeds/branches.
- `Snapshot` contains only `{Config, State}`. Runtime table pointers are excluded and rebound on load, so snapshots are safe across kernels and clonable into large continuation grids.
- CUDA 11.5 did not reliably propagate `Tile`'s negative sentinel default members through a deeply nested aggregate reset. All device tile construction is therefore explicit via `make_tile()`. Deep-state parity caught and verifies this.

## Snapshot RNG acceleration

For a seed/state snapshot evaluated against many candidate continuations, every candidate consumes the same daily CPython RNG stream until its public trajectory changes. Repeating the full MT19937 seed/mix/twist work in every branch was wasteful.

The retained exact optimization precomputes the first 624 **tempered CPython MT19937 outputs** for each `(snapshot, day)` once on the GPU. Candidate continuations read that shared stream directly. `CachedDayRandom` retains an exact slow fallback that reconstructs the stream if an unusually long `randbelow()` rejection sequence ever consumes beyond 624 words, so this is not an approximation.

This cache materially improved saturated snapshot branching and passed cached-vs-uncached snapshot parity at every tested boundary.

A different experiment that precomputed all daily RNG streams for independent full-episode seed sweeps was rejected: the cache cost/memory traffic made the real Kaw/Hax workload slower. Independent-seed full episodes therefore use the ordinary exact MT path.

## Native replay-tape research path

`tape_bench` evaluates two replay policies over a contiguous seed cohort entirely in CUDA after native C++ parsing:

```bash
scratch/cuda-engine-build/tape_bench TAPE0.json TAPE1.json 62000 4096 64
```

`tape_branch` builds exact base-policy prefixes over many seeds, snapshots them, optionally builds shared post-prefix daily RNG streams, then evaluates every snapshot against every candidate continuation:

```bash
scratch/cuda-engine-build/tape_branch \
  BASE0.json BASE1.json \
  0 168 62000 96 \
  @candidate-list.txt
```

Arguments after the seed cohort are candidate replay tapes. Any argument beginning with `@` expands a newline-delimited tape list. The runner reports per-candidate win count, mean margin, and floor. Candidate-major results are still just research evidence; promotion requires the project's legal public-state cohort construction and full holdout protocol.

For the RTX 3050 and a 28-candidate market-heavy continuation population, **64 threads/block** and roughly **96 seed snapshots** were the best measured local geometry. `tape_branch` defaults to 64 threads/block and accepts `KAGCUDA_TPB` as a native override for other GPUs.

## Measured performance regimes

These measurements are workload-specific. Do not quote one number as a universal CUDA engine speed.

### Lightweight fixed action stream

A 4,096-environment synthetic/starter-style batch reached roughly:

- **~98k complete 719-turn episodes/s**
- **~70M individual environment transitions/s**

The final CPU core is about 8,775 episodes/s single-thread on its lightweight benchmark, so this regime is roughly an order of magnitude faster on the RTX 3050.

### Real market-heavy full replay policies

Kawashigi vs Hax over 4,096 fresh seeds measured around:

- **~26k complete episodes/s**
- **~18.8M transitions/s**

The final CPU PGO fixed-tape path is about 6.5k episodes/s using the allowed two CPU workers, so the GPU is roughly 4× faster for this real full-episode cohort workload.

### Real market-heavy 28-candidate continuation search

With exact Kaw/Hax-derived prefixes at step 168 and the genuine 28-tape donor population:

- 96 snapshots × 28 candidates = 2,688 continuations/launch.
- Three adjacent frozen-source runs measured **33.64k, 35.59k, and 35.64k continuations/s branch-only**; median **35.59k**.
- Median branch time was about **75.5 ms**.
- Median throughput including prefix construction + shared RNG-cache construction was **32.95k continuations/s**.

The CPU snapshot continuation engine is around 5.3k continuations/s median on its 28-donor workload. CUDA therefore provides roughly a 6–7× saturated continuation-throughput advantage when many seed/state snapshots are evaluated together.

A single `1 snapshot × 28 candidates` launch is underfilled and was only around **4.4k continuations/s**, slightly below the CPU snapshot path. Use CPU for tiny latency-sensitive branch sets; use CUDA for the intended population-search/RL workload.

### Lightweight snapshot grid

Synthetic/light continuation grids can be much faster than market-heavy replay cohorts. A 96×28 run exceeded **270k continuations/s (~150M transitions/s)**. This is useful for upper-bound characterization but should not be used as the expected live-meta throughput number.

## Evidence-driven performance decisions

Retained:

- one independent environment/continuation per CUDA thread;
- per-environment `Sim` state in GPU global memory (required for tractable nvcc code generation and persistent snapshots);
- exact market price cache/prefix tables;
- shared tempered-MT output cache for snapshot cohorts;
- compact snapshot-major candidate scheduling;
- default 64 threads/block for the current RTX 3050 real donor workload;
- no `--use_fast_math`.

Tested and rejected on the genuine market-heavy workload:

- precomputing all daily RNG streams for independent seed sweeps;
- explicit cache-all (`-dlcm=ca`) global-load policy;
- register caps at 64 and 72 (introduced spills without a reliable gain);
- moving the 624-word MT scratch state into global `Sim` storage (shrunk the local stack but badly regressed full-episode throughput);
- padding 28 candidates to 32 lanes so one snapshot occupied one warp;
- scalarizing two-player market temporaries (higher register pressure, slower real branches);
- `__restrict__` buffer hints;
- `--extra-device-vectorization`;
- broader compiler loop-unrolling style changes were not adopted without real-workload proof.

The remaining branch kernel has a sizeable local frame largely associated with the mathematically exact rare MT fallback. Removing that frame did not improve real branch throughput and hurt independent-seed episodes, so stack size alone is not a valid optimization target.

## Hybrid rule + RL use

The CUDA engine is not only a replay evaluator. `ObservationView` + `policy_runner.cuh` provide the legal all-device boundary needed for hybrid research:

- deterministic high-confidence rule logic can execute directly on GPU;
- a learned value/selector model can choose among rule-generated continuations;
- residual RL can learn when to override the heuristic baseline;
- thousands of self-play environments can stay on device;
- exact snapshots can oversample rare failure cohorts;
- exploiters/adversarial populations can search for weaknesses in the current policy;
- large `(state, candidate, final_reward)` datasets can be generated for value learning/distillation.

The intended end-game architecture is **rules + lane experts + learned value/selector + deterministic fallback**, trained/evaluated in this CUDA world. The final Kaggle submission can then be a distilled CPU-compatible policy; the large CUDA research/training system does not need to ship with the submission.

## Practical guidance

- Use the CPU snapshot path for one tiny 28-candidate latency-sensitive state.
- Use CUDA when the workload is `many snapshots/seeds × many candidates/policies`; this is the regime that unlocks the GPU.
- Keep exact validation mandatory after any simulator/RNG/market change.
- If the simulator changes, regenerate/re-run all parity gates before trusting snapshots or RL data.
- Do not infer strategic v33 promotion evidence from synthetic or Kaw/Hax performance benchmarks; run the actual legal PET/ICE public-state cohorts and holdouts.
