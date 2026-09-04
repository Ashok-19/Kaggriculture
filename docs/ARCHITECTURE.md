# Current Architecture

## Discovery

Fresh BKG4 self-play supplies environment-only trajectories across independent
seeds and both seats. The internal adversarial league adds bounded Option-Program
exploiters spanning strategic lanes. Multi-regime BUILD explicitly measures
support across seeds, seats, members, lanes, and trajectories so a single-path
specialist cannot dominate discovery.

## Screening and search

INTERNAL_DEV is generated independently and screens discovery candidates before
expensive downstream evaluation. RLRMB and Direct-Bonsai remain the current
search baselines, but research at `500268a` found both unable to produce
cross-regime specialists under the equal-budget bakeoff. The planned replacement
is Adaptive Semantic Program Evolution (ASPE), staged in
`docs/ASPE_MASTER_PLAN.md`. Until its Phase-A falsification passes, it is not the
production oracle. CUDA evaluates exact batches; the C++ engine remains the
reference physics oracle.

## Routing and safety

Candidates may enter the existing binary PARENT/specialist routing path.
TRANSFER and VALIDATION measure independent generalization. CEGIS hardens legal
gates, and SEALED remains unavailable to search. A candidate is composable only
after exact materialization/parity checks, zero semantic mismatches, and zero
protected win-to-loss.

## Runtime

The Python and native BKG4 implementations consume the same compiled controller
and route packs from `assets/champion/`. Compatibility overlays for the frozen
legacy behavior remain explicit runtime modules; new strategy belongs in the
controller and composed specialist routes.
