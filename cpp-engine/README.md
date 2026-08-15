# Kaggriculture C++ engine

Maintained local C++ port of the Kaggriculture environment, extracted from
`public-notebooks/4000x-environment-speedup-kaggriculture.ipynb` and updated for
the competition host's 2026-08-15 balance change.

## Authoritative environment

This directory targets **kaggle-environments 1.32.7**, specifically official
Kaggle commit:

`28b6d8af3ce73926b3d0fda1410c1ddd8384ab8c`

That is PR #1399 (`Make underused resources situational`). The shipped
`kaggriculture.py` SHA-256 is:

`bc8a54879ef02c7ea64b8b333d6a976f0ea65c4949149d01f463f23bccee653e`

The 1.32.7 delta from the notebook's 1.32.6 baseline is market pricing:

- add `hinge` scarcity shape with gain 8.0;
- CARROT below-I0: `hinge`, target 1.00;
- TOMATO below-I0: `hinge`, target 0.40;
- EGG below-I0: `hinge`, target 0.40.

The hinge is linear up to its resource-specific knee `T`, then adds a quadratic
scarcity term. This is why the old notebook engine must not be used unchanged.

## Layout

- `include/pyrandom.hpp` — CPython-compatible RNG used by weeds/shop unlocks.
- `include/sim.hpp` — header-only environment simulation.
- `src/validate.cpp` — exact full-trajectory validator.
- `src/bench.cpp` — raw C++ throughput benchmark.
- `src/price_probe.cpp` — market-price parity probe.
- `src/pybridge.cpp` — native Python binding used by local Python-agent arenas.
- `tools/build_pybridge.py` — builds the binding into `scratch/cpp-engine-build/`.
- `tools/export_trace.py` — exports exact actions/truth from official 1.32.7.
- `tools/check_price_parity.py` — exhaustive integer price comparison.
- `tools/run_full_parity.py` — 5 pairings × 3 seeds full-episode parity gate.
- `tools/check_fastpaths.py` — compiled-tape, shared-public, and snapshot-branch parity gate.
- `tools/bench_python.py` — minimal Python-environment throughput baseline.

Compiled binaries and generated traces are intentionally kept outside this
source tree under `scratch/cpp-engine-build/`.

## Build and parity gates

From the repository root:

```bash
make -C cpp-engine
make -C cpp-engine price-parity
make -C cpp-engine full-parity
make -C cpp-engine fastpath-parity
```

No C++ result is valid project evidence unless the parity gates pass against the
current pinned official engine.

## Scope and switching policy

The maintained Python bridge exports the full observation surface used by the
current research corpus: public farms/tiles/workers, market inventory/prices,
town shops, day/hour, and player-private shed/seeds/per-worker inventories.
Validation completed against the refreshed 1.32.7 official oracle:

- 270,009/270,009 integer market-price states exact;
- 15/15 full 719-action reference episodes exact at every compared state;
- complete no-op observation trees exact for both seats across 719 turns;
- v32 vs One-For-All current-agent rewards exact across representative paired seats;
- complete v32/OFA 719-action streams exact on a dynamic game;
- v33 Hax/Wang replay-fixture rewards exact across both seats.

The optimized raw standalone simulator now runs about **8,775 complete episodes/s**
on one CPU thread on a dynamic 719-step trace, roughly **6.31 million environment
transitions/s**. The original maintained port was about 3,600-3,800 eps/s on this
machine. The retained core optimizations are exact derived-state/algorithmic changes:
market-price caching and batched independent orders, scheduled plant decay, fused
end-of-day board traversal, redundant inventory-clear removal, cheaper whole-inventory
drops, and cached/specialized CPython MT19937 seed initialization.

There are three different research throughput regimes:

1. **Arbitrary Python agents.** `arena.py --backend cpp --reuse-agents --shared-public-observation`
   is the fastest verified current-agent mode. `--reuse-agents` is valid only for
   agents proven to reset all episode state at step 0; `--shared-public` is valid only
   for agents proven not to mutate nested public observation objects. On v32 vs
   One-For-All, seeds 60200-60209, both seats, two workers, the complete 20-game table
   is exact. With the same agent-reuse optimization on the official backend, the
   measured wall times were 19.40 s official versus 8.73 s C++, about **2.22x fair
   backend speedup**. Python policy execution is now about 90% of loaded-game time,
   so simulator micro-optimization cannot multiply this regime much further.
2. **Compiled replay/tape batches.** `cpp_oracle.run_tapes_batch(...)` parses each tape
   once and executes the entire seed batch in C++. It reproduces the existing replay
   wrapper semantics, including dynamic hand padding/truncation. On a market-heavy
   Kaw-vs-Hax 1,024-seed panel, the portable bridge is about 3.0k eps/s with one
   worker and about 6.0k eps/s with two; the locally profile-guided build is about
   **3.3k / 6.5k eps/s** with one/two workers respectively. The pre-native
   Python replay wrappers measured about 6 eps/s on the same class of workload.
3. **Snapshot continuation search.** `snapshot_vs_tape(...)` runs the causal Python
   prefix once and clones the exact C++ state; `branch_tape_candidates(...)` then
   evaluates compiled donor continuations natively. Snapshot workers use a thread-local
   exact seeded-MT cache for repeated future `(seed, day)` states. The real 28-donor
   step-168 Hax search runs around **5.3k continuations/s median** on the local PGO
   build, with repeated outputs exact.

All native worker counts are hard-limited to **1 or 2**. Never bypass that cap.

### Optional local PGO

The default bridge build is profile-free and uses the portable optimized flags
`-O3 -march=native -flto -fipa-pta -fno-semantic-interposition -fvisibility=hidden`.
For this workstation only, `tools/build_pybridge.py` also supports:

```bash
KAG_CPP_PGO=generate uv run --python 3.10 --with pybind11 python cpp-engine/tools/build_pybridge.py
# Run representative replay batches, snapshot branches, and current-agent games.
KAG_CPP_PGO=use uv run --python 3.10 --with pybind11 python cpp-engine/tools/build_pybridge.py
python3 cpp-engine/tools/check_fastpaths.py
```

PGO data lives only under `scratch/cpp-engine-build/pgo/`; it is machine/workload
specific and must never be treated as source or committed. Re-train it after C++
source changes. The final trained profile improved the market-heavy batch from about
6.0k to about **6.5k eps/s** with two workers. Do not use `-Ofast` or `-ffast-math`:
exact floating-point market pricing and rounding are part of the parity contract.

`agent_system/tools/official_oracle.py` remains the final correctness oracle and
is pinned, content-hash verified, to the same 1.32.7 source.
