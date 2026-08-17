# Stateful Bonsai v6 — BKG3 compatibility translation

This package is the first post-freeze Bonsai-KG compatibility baseline. It translates the historical research controller in `scratch/bonsai_cpu/stateful_core_v6/main.py` without changing its strategy.

## Architecture

- `controller.json` / `controller.bkgc`: genuine frozen BKG3 control policy. It owns stage-144 selection, remembered leaf/route state, one-shot stage-216 refinement, and latching.
- `routes.bkv6`: immutable compatibility expert tape compiled from the eleven authoritative `route<ID>.json` files. It is an actuator only; it never chooses a route.
- `runtime.py`, `cpp_runtime.hpp`, `cuda_runtime.cuh`: thin wrappers that reset per episode, execute the same BKG3 controller, read `selected_route` from BKG3 blackboard, and materialize the corresponding tape action with the historical dynamic hand-count rule.

No BKG3 opcode, ABI record, observation feature, status rule, or runtime layout is changed by this integration.

## BKG3 blackboard

| slot | name | type | initial |
|---|---|---|---|
| 0 | `leaf` | ENUM8 | 255 / NONE |
| 1 | `route` | ENUM8 | 64 |
| 2 | `stage144_done` | BOOL | 0 |
| 3 | `stage216_done` | BOOL | 0 |

Leaf encoding matches the historical CUDA reconstruction exactly: `A1=0, A2=1, B=2, C1=3, C2=4, D1=5, D2=6, E=7, F=8, G=9, H=10, I=11, J1=12, J2=13, K=14, NONE=255`.

Route ids are stored directly as their historical integer ids.

## Identity

`assets/identity.json` records:

- old Python oracle SHA-256;
- old CUDA reconstruction SHA-256;
- every source route JSON SHA-256;
- BKG3 controller semantic hash;
- route-pack full SHA-256 and payload SHA-256;
- a combined compatibility-agent identity.

Python, C++ and CUDA reject controller/route assets whose hashes do not match the checked-in identity.

## Rebuild and review

From repository root:

```bash
make -C bonsai-kg v6-build
make -C bonsai-kg v6-test
make -C bonsai-kg v6-canary
make -C bonsai-kg v6-official-canary
```

`v6-test` rebuilds/validates assets, checks every stage-144/stage-216 selector branch through Python/C++/CUDA, checks all 7,909 route records against source/Python/C++/CUDA, and runs the focused Python boundary/reset suite.

`v6-canary` runs 64 complete exact-engine legal-history games (32 per seat) and compares old Python v6 -> translated Python -> exact C++ -> CUDA at every tick. It is a translation test, not a win-rate benchmark.

`v6-official-canary` runs the same 64-game comparison through `kaggle-environments==1.32.7`. The installed 1.32.7 `kaggriculture.py` is verified byte-for-byte identical to the repository copy under `kaggle-environments-master/kaggle_environments/envs/kaggriculture/kaggriculture.py` before freeze evidence is reported.
