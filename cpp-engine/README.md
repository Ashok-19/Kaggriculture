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
- `tools/bench_python.py` — minimal Python-environment throughput baseline.

Compiled binaries and generated traces are intentionally kept outside this
source tree under `scratch/cpp-engine-build/`.

## Build and parity gates

From the repository root:

```bash
make -C cpp-engine
make -C cpp-engine price-parity
make -C cpp-engine full-parity
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

On this workstation the raw environment runs at about **3,600 eps/s** versus
**0.811 eps/s** for `kaggle-environments==1.32.7`. With full current Python agent
logic, Python policy execution is now the dominant cost: in a direct v32 vs
One-For-All timing, C++ observation construction plus action parsing/physics was
about 17% of game time and the two Python policies about 83%.

`agent_system/tools/arena.py` uses the C++ backend by default. Use
`--backend official` for correctness canaries. For agents explicitly verified to
reset all episode state at step 0, `--reuse-agents` keeps the loaded modules alive
inside each worker. On v32 vs One-For-All, seeds 60200-60209, both seats and two
workers, it reproduced the exact 20-game result table while reducing wall time
from 26.12 s official Python to 10.14 s C++ (about **2.58x end-to-end**). Do not
use this mode for an unverified stateful agent. The arena rejects more than two
workers to enforce the local compute cap.

`agent_system/tools/official_oracle.py` remains the final correctness oracle and
is pinned, content-hash verified, to the same 1.32.7 source.
