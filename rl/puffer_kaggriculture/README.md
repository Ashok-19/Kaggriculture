# Kaggriculture PufferLib RL teacher

This directory is an offline-only RL feasibility integration. It does not modify BKG4, its VM/ABI, the production champion, or Kaggle submission code.

## Environment and policy boundary

- Reuses `cpp-engine/include/sim.hpp`, the parity-validated competition simulator. The build copies the exact current `sim.hpp` and `pyrandom.hpp` into the Puffer Ocean environment and records their hashes.
- Uses PufferLib 4.0 native static CPU/OpenMP vectorization plus pinned CPU/GPU transfers and the native CUDA MinGRU/PPO/Muon backend.
- Observation is a normalized 6,377-float representation containing public state for both farms plus only the learner seat's private shed, seeds, and carried inventories. Opponent-private state is never encoded.
- Action space has 44 discrete heads: one state-conditioned legal primitive-action head for every possible unit index (40), plus four ordered market heads. Every active worker can act; there is no worker cap.
- Full Puffer horizon is 720 slots: 719 real Kaggriculture decisions plus one terminal flush slot required by Puffer 4.0's `reward[t+1]` advantage convention. The flush action is ignored and the next rollout starts from a reset simulator with reset MinGRU state.
- Reward remains terminal-only and bounded: `0.9 * win_sign + 0.1 * tanh(margin / 5000)`.

## Local teacher configuration

The performance-validated local teacher is a 3-layer MinGRU with hidden size 1,664:

`6377*1664 + 3*(3*1664*1664) + (9100+1)*1664 = 50,675,456` trainable parameters.

The local RTX 3050 configuration uses 64 agents, one rollout buffer, two environment threads, horizon 720, minibatch 5,760, BF16 native Puffer precision, and `cudagraphs=1`. Replay ratio stays at 1.0: throughput is not inflated by skipping optimizer work.

## PufferLib 4.0 compatibility/performance patches

`scripts/patch_puffer_backend.py` applies only pinned-SHA native-backend patches before building:

- raises `MAX_ATN_HEADS` from 16 to 64 for the 44-head Kaggriculture policy;
- adds the cleanup null guard required when graph allocations are absent;
- keeps CUDA graphs for rollout inference but disables Puffer 4.0 train-graph capture. The installed train graph captures a fixed selected minibatch/gradient path; later graph replays change weights while reusing stale training work, so it is not valid for RL updates;
- replaces the serial discrete sampler with one block per agent / one thread per action head. Philox draws, per-head softmax arithmetic, and final log-prob summation retain original head order.

The parallel sampler was checked against the original serial sampler on a 64-agent full rollout. SHA-256 hashes were identical for actions, logprobs, values, rewards, and terminals, and episode aggregates were exactly identical. This is an execution-only optimization.

A head-parallel PPO-loss prototype was also tested. It produced the exact same post-update checkpoint hash as the serial PPO kernel but was slower, so it was rejected and is not part of the build.

## Measured local throughput

The original 69.45M / 8-agent / minibatch-720 configuration delivered roughly 1.1k end-to-end SPS and spent about 94% of rollout time in GPU inference/sampling. The environment itself can exceed ~200k SPS, so it was not the bottleneck.

The accepted performance path increased real end-to-end training throughput by roughly an order of magnitude. Repeated synchronized 50.68M runs with 64 agents and minibatch 5,760 are around 11k-12k SPS on the local RTX 3050, with finite losses and about 3.33 GiB VRAM used. The head-parallel sampler reduced sampler kernel share from 12.7% of traced GPU time to about 0.4%; remaining time is dominated by BF16 GEMMs and real training work.

Use `scripts/bench_native.py` for synchronized native throughput measurements. Generated checkpoints, logs, and raw benchmark artifacts belong under the ignored RL artifact directories.
