# BKG4 Champion Lineage

This directory is the strategic BKG4 lineage. It does **not** modify ABI/schema 4.

`BKG4_BASE_0` is deliberately conservative:

- the frozen stateful-v6 BKG3 controller is semantically converted to BKG4;
- the frozen stateful-v6 route pack remains the normal actuator;
- the already-validated H4/public-shop compatibility overlays are preserved in a separate immutable route pack;
- the accepted `FERT81 -> CARROT-v3 -> EOD shed-overflow` mechanics overlay remains frozen external behavior.

New strategic search modules are compiled into the BKG4 controller after the frozen v6 decision stages. Search metadata, lineage statistics and evolvability values never enter the production ABI.

Build BASE_0 locally with:

```text
python3 bonsai-kg/integrations/bkg4_champion/build_assets.py
```

The build records SHA-256 provenance for every compatibility tape. Runtime operation uses only tracked assets after the build.

## First promoted strategic module

`animal_feed_deferral_v1` was discovered from the authoritative engine's animal refresh semantics and distilled into the BKG4 controller. The tree computes per-unit causal authorization from legal lifecycle features; the reconciler has no strategic logic and only replaces an already materialized `FEED` with `PASS` for a tree-authorized unit.

The minimized rule applies only to cows with `consecutive_unfed == 0` and `pending_care_bonus == 0`. Search showed goose expansion was redundant on the current actuator manifold, sheep expansion was unsafe, tighter wheat guards were weaker, and harvest/collect replacements were unnecessary.

Canonical challenger assets live under `champions/feed_deferral_v1/`; BASE_0 assets remain intact for regression.

## Promotion rule

A searched module is not a champion merely because CUDA search ranks it highly. Promotion requires direct regression gates, descendant/evolvability evidence, both seats, unseen seeds/families, ablation/minimization, exact C++/CUDA parity, and only then a small official-Python canary when warranted.
