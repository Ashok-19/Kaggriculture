# Service-tour transfer postmortem

## Verdict

**SERVICE-TOUR POSTMORTEM — KILL REPRESENTATION FAMILY**

Primary classification: **B — PROGRAMME-BODY / EXECUTION INCOMPATIBILITY**.

The postmortem used only the four already-exposed BUILD seeds from the failed preflight, both seats, and the same four frozen opponent configurations. No INTERNAL_DEV, TRANSFER, VALIDATION, CEGIS or SEALED data was opened.

The dominant failure is not a missing Shop route selector. Six of eight contexts keep Shop route 0 at step 144, yet P0 still loses 22-72% of paired productivity. The first persistent economic divergence occurs much earlier at step 65: P0 has four fertilizer units stranded in worker cargo and repeatedly requests an unaffordable cow, whereas Shop realizes four fertilizer units and immediately finances that cow. More importantly, this is not an isolated finance bug: P0 completes only about 76.5-77.1% of its declared day-level service obligations, accumulates 375-389 carried-over obligations by day 28, and broadly under-produces wheat, melon, strawberry and animal products across every context.

Three bounded diagnostic counterfactuals fail the salvage criterion. Deferring post-opening animals by one day helps one Shop-Plus context but has a median gap recovery of only 6.65% and strongly regresses route-12. One extra daily hire regresses all contexts. Extending the existing completed-tour realization trigger from land financing to unresolved livestock capital helps one seat of one Shop-Plus context, hurts direct Shop, and recovers only 1.8-3.1% in the route-3/12 stress contexts. No legal small programme-level mechanism recovers >=50% of the gap across multiple relevant contexts.

Public market pressure is an amplifier, not the root cause: milk/wool/strawberry/wheat prices and inventories diverge substantially later by opponent, but P0's service backlog and missed day-2 acquisition begin before route selection and before those market regimes separate. Salvaging this family would require rebuilding worker/service bodies and their realization coordination by regime, which violates the intended small programme-selection salvage hypothesis and approaches multiple complete low-level schedules.
