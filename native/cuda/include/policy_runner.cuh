#pragma once
#include "observation.cuh"

namespace kagcuda {

// Device policy protocol:
//
//   struct MyPolicyState { ... per-environment mutable state ... };
//   struct MyPolicy {
//     __device__ void reset(PolicyState&, const ObservationView&) const;
//     __device__ void act(PolicyState&, const ObservationView&, Action&) const;
//   };
//
// The policy never receives Sim/State and therefore cannot accidentally read
// opponent-private state. Action{} is zero/pass by default; policy code fills
// only the desired fields. Neural policies can store device weight pointers in
// the policy object; rule policies can be plain POD structs.

template <class Policy0, class PolicyState0, class Policy1, class PolicyState1>
__global__ void run_policy_batch_kernel(Config base, DeviceTables tables, int n_envs,
                                        Policy0 policy0, PolicyState0* state0,
                                        Policy1 policy1, PolicyState1* state1,
                                        Sim* sims, double* rewards) {
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= n_envs) return;

    Config cfg = base;
    cfg.seed = base.seed + static_cast<uint64_t>(id);
    Sim& sim = sims[id];
    sim.init(cfg, &tables);

    PolicyState0& ps0 = state0[id];
    PolicyState1& ps1 = state1[id];
    policy0.reset(ps0, make_observation(sim, 0));
    policy1.reset(ps1, make_observation(sim, 1));

    while (!sim.st.done) {
        Action a0{};
        Action a1{};
        // The official callable sees the current number of workers. Match that
        // shape even if a policy leaves trailing unit actions untouched.
        a0.n_units = sim.st.farms[0].n_units;
        a1.n_units = sim.st.farms[1].n_units;
        policy0.act(ps0, make_observation(sim, 0), a0);
        policy1.act(ps1, make_observation(sim, 1), a1);
        sim.step_with_unit_counts(a0, sim.st.farms[0].n_units,
                                  a1, sim.st.farms[1].n_units);
    }

    rewards[id * 2] = sim.st.farms[0].money;
    rewards[id * 2 + 1] = sim.st.farms[1].money;
}

// Snapshot continuation counterpart. Policies are reset from the snapshot
// observation by default. Stateful-prefix research can instead snapshot/copy
// PolicyState arrays alongside game Snapshot arrays and launch a custom kernel
// using the same ObservationView + Sim::step_with_unit_counts_cached API.
template <class Policy0, class PolicyState0, class Policy1, class PolicyState1>
__global__ void run_policy_snapshot_grid_kernel(DeviceTables tables,
                                                const Snapshot* snapshots, int n_snapshots,
                                                int policies_per_snapshot,
                                                Policy0 policy0, PolicyState0* state0,
                                                Policy1 policy1, PolicyState1* state1,
                                                const uint32_t* rng_cache,
                                                int cache_first_day, int cache_n_days,
                                                Sim* sims, double* rewards) {
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = n_snapshots * policies_per_snapshot;
    if (id >= total) return;
    const int snapshot_id = id / policies_per_snapshot;

    const uint32_t* cache = nullptr;
    if (rng_cache && cache_n_days > 0)
        cache = rng_cache + static_cast<size_t>(snapshot_id) * cache_n_days * PyRandom::N;

    Sim& sim = sims[id];
    sim.load_snapshot(snapshots[snapshot_id], &tables, cache, cache_first_day, cache_n_days);
    PolicyState0& ps0 = state0[id];
    PolicyState1& ps1 = state1[id];
    policy0.reset(ps0, make_observation(sim, 0));
    policy1.reset(ps1, make_observation(sim, 1));

    while (!sim.st.done) {
        Action a0{};
        Action a1{};
        a0.n_units = sim.st.farms[0].n_units;
        a1.n_units = sim.st.farms[1].n_units;
        policy0.act(ps0, make_observation(sim, 0), a0);
        policy1.act(ps1, make_observation(sim, 1), a1);
        if (cache)
            sim.step_with_unit_counts_cached(a0, sim.st.farms[0].n_units,
                                             a1, sim.st.farms[1].n_units);
        else
            sim.step_with_unit_counts(a0, sim.st.farms[0].n_units,
                                      a1, sim.st.farms[1].n_units);
    }
    rewards[id * 2] = sim.st.farms[0].money;
    rewards[id * 2 + 1] = sim.st.farms[1].money;
}

}  // namespace kagcuda
