# B4IM Dual-T4 Corpus Deployment Design

## Goal

Turn the frozen B4IM1/B4IC1 specialist factory at source commit
`c012c2c6c1de18fc250cf9e5425c3aea1f1d273c` into a portable Kaggle corpus-production
appliance. The deployment consists of one private support dataset and one importable
dual-T4 notebook. It does not change the B4IM grammar, retrain a model, promote a
champion, run official Python games, or submit to the competition.

## Support Product

The support dataset is an allowlisted tar bundle plus `support_manifest.json` and
`dataset-metadata.json`. It contains the transitive Python and CUDA runtime closure,
the validated `w33_s172_expansion_v1` BKG4 parent, three portable discovery banks,
their 253 tapes, and an exact B4IC smoke fixture. Bank tape paths are rewritten to
`../tapes/<content-sha256>.json`; the staged bundle is independently loaded through
`load_offline_specialist_bank()` and checked for unique cases, unique cohort ownership,
discovery-only roles, resolved tapes, and a legal 64/400 tape-disjoint split.

The manifest records B4IM1/v1, B4IC1/v1, BKG4 ABI 4, feature registry 1, no fast math,
the canonical parent identity, bank/tape summaries, the `sm_75` runtime requirement,
and per-file SHA256/size integrity metadata. The smoke fixture contains empty,
route-override, committed-route, lifted each-own-unit rewrite, and market-intent
candidates with exact money, trace, activation, action-change, and fault expectations.

## Kaggle Runtime

The notebook safely materializes either Kaggle's expanded archive or the intact tar
under `/kaggle/working`, validates every file, requires two T4s, and compiles the exact
evaluator once using `ARCH=sm_75`, an installed host compiler, and no fast math. It
inspects the cubin and runs the packaged exact smoke independently with
`CUDA_VISIBLE_DEVICES=0` and `1`, each using logical device 0.

Calibration runs representative candidate-by-64 batches concurrently for TPB 64, 128,
and 256. Production then launches one long-lived Python worker per physical T4. Each
worker owns an island directory and a disjoint seed namespace, repeatedly executes
bounded specialist-factory shards, and never falls back to CPU game evaluation.

## Shards And Finalization

Each shard uses the existing specialist-aware funnel and capability ladder, including
depth-2 recursion, CEGIS, MCTS, LNS, recombination, composition, local neighborhoods,
and ablations. Expensive saturation reporting is disabled per shard without weakening
corpus or lineage validation. A successful shard is revalidated, converted to compressed
NPZ arrays, written as gzip JSONL plus a manifest/index, and stripped of disposable
raw evaluator intermediates.

Workers share one absolute deadline, stop before the final reserve or disk floor, and
emit heartbeat/progress state. The notebook monitors low-frequency GPU utilization and
terminates workers gracefully. Final consolidation streams shard indexes, detects
semantic duplicates and contradictory ASTs, selects a deterministic canonical row while
preserving secondary-panel evidence, and emits a lightweight global index.

The grouped split hashes the exact `repr(structural_skeleton(module))` key used by
`invariant_grouped_split()`. An equivalence test against the validated 384-row corpus
must match the reference split exactly. Final validation requires one semantic ID per
training row and zero train/validation skeleton overlap.

## Verification

Local verification is bounded to packaging, imports, path/provenance checks, fixture
validation, shard serialization, dedup/split tests, orchestration dry-run, notebook cell
compilation, relevant invariant regressions, Python compilation, and Git hygiene. No
large local CPU/GPU campaign is part of this deployment.
