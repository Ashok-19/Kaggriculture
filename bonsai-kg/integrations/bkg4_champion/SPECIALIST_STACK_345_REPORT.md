# BKG4 Specialist Stack — 345 Nodes

Accepted locally on 2026-08-20 for team ReCurSiON.

The 315-node parent is extended by two independent state-routed subtrees: route 213 (`fert_wool_expansion_v1`) and route 214 (`w33_s172_expansion_v1`). The resulting controller is 345 nodes, 23 BB slots, zero NodeState slots, and leaves the frozen BKG4 VM/ABI unchanged.

Route 213 activates at step 264 when fertilizer market inventory is at least 10105 and wool inventory at most 9940. Across current, historical and screen validation it has 12/12 positive activations over 6 episodes and 4 families, with two current loss-to-win flips. Removing private expansion makes all 12 supported activations worse.

Route 214 activates at step 192 when WHEAT price is exactly 33 and STRAWBERRY price exactly 172. This discrete market state is broad rather than identity-like: 70/70 supported activations are positive across 29 episodes and 22 families, including screen and sealed holdouts, with two current loss-to-win flips. Adjacent WHEAT-price regimes are mixed, so the exact discrete boundary is deliberately not smoothed. Removing hires makes all 70 supported cases worse; removing SELL financing makes 68/70 worse. The market book is therefore a lane selector, not the source of symmetric value.

The exact composed controller moves the shifted current league from 94W/46L to 98W/42L with zero outcome regression. Historical336, screen61, fresh24 and sealed24 have no negative outcome regression. Python/C++/CUDA controller parity is exact over 2,876 reachable ticks; route materialization parity is 0 mismatches across 18 cases; source-backed ReCurSiON regression is 0/4. Hot-path p99 visits are 165 with zero state slots and zero live intents. The official Python canary remains 39W/25L.

This is still far from the 100% objective. The next search starts from the remaining 42 fresh-current losses and continues adding legal-state specialist islands rather than broad global routes.
