# Kaggriculture Agent System

This directory is the development and runtime system for the post-v27 Kaggriculture agent.
It is intentionally separate from `public-agents/` and the research database.

## Design

The system has four layers:

1. **Official-engine oracle** — Python tooling pins the official Kaggle `kaggriculture.py` revision and runs exact local games/counterfactuals.
2. **Native decision core** — C++ implements exact market pricing, economic-regime inference, opponent market residual tracking, market MPC, route-selection signals, and later batch search.
3. **Runtime gateway** — `submission/main.py` converts the Kaggle observation to a fixed C ABI frame and applies native decisions to a coherent production route. It never uses opponent identity fields.
4. **Promotion arena** — paired-seat, multi-seed, multi-family, chronological evaluation. A candidate is not promoted merely because it improves one leaderboard-like screen.

The public v27 route remains a baseline/donor, not the architecture. The target is a low-entropy production portfolio plus closed-loop market and economic routing.

## Non-negotiable promotion gates

- Valid Kaggle action schema on every turn.
- No opponent names, team IDs, submission IDs, or private opponent state.
- Exact market-price parity with the official engine.
- Both-seat evaluation.
- 100% wins against v27 on the designated v27 domination panel before a candidate may be called `v27-dominant`.
- Broader population tests against current route families after the v27 gate.
- Chronological future episodes remain untouched until policy freeze.

`100%` is a measured promotion criterion, not an assumption or a claim made before the arena passes.

## Build

```bash
cmake -S agent_system -B agent_system/build -DCMAKE_BUILD_TYPE=Release
cmake --build agent_system/build -j
```

The shared library is `agent_system/build/libkag_agent.so`.

## Runtime ABI

`KagFrame` is a bounded POD representation of public/own-private observations. The native library retains only per-seat controller state (market residual history, route confidence, and sale debt). `KagDecision` contains a route/regime recommendation and a bounded market intervention.

The Python gateway remains capable of a safe Python fallback if the native library is absent; submission packaging will fail closed if the promoted policy requires native execution but the library is missing.
