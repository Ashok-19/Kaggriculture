# Dual-T4 Kaggriculture Bonsai Meta Search

This directory packages the development-only CUDA meta-discovery campaign used by `notebooks/kaggle_dual_t4_meta_search.ipynb`.

## Purpose

The notebook runs two independent exact-CUDA program-search islands for roughly one Kaggle 10-hour session:

- GPU 0: exploitation-biased, strongly preserving proven v6 actuator basins and stronger selection pressure.
- GPU 1: exploration-biased, lower selection pressure, fewer route-seeded individuals, extra structural mutation.

The islands do not exchange populations during the main search. They write separate checkpoints and elite archives and are merged only after the search phase.

The candidate representation is the existing fixed-size `GPProgram`/`GPManager` Bonsai search grammar in `cuda-engine/include/bonsai_gp.cuh`: ordered public-observation predicates plus modular route/manager leaves and an evolvable 11-phase economic blueprint. Candidate action selection cannot reference opponent identity, submission/replay IDs, episode IDs, seeds, hidden inventory, or future RNG.

## Exact semantics

Simulation uses the repository CUDA engine pinned to the Kaggriculture `kaggle-environments==1.32.7` semantics. Search builds use `-O3` and explicitly do not use `--use_fast_math`.

Early search stages approximate fitness only by using fewer development tapes and stopping exact trajectories earlier. The simulator transition semantics themselves are not approximated.

Before a long run, the notebook compiles and executes the exact BKG3-v6 CUDA compatibility runner and compares its stdout SHA-256 against `v6_smoke_reference.json`. Any mismatch aborts the campaign.

## Development-only input bundle

Build the upload directory locally from repository root:

```bash
python3 tools/meta_search/prepare_input_bundle.py \
  --output scratch/kaggriculture_meta_search_bundle \
  --force
```

The builder requires and verifies:

- 168 development tapes across 21 priority families;
- 84 frozen validation tapes excluded;
- 50 blind elite/continuity tapes excluded;
- 28 family holdouts excluded.

It hashes forbidden tape contents and aborts if any forbidden tape is copied into the bundle. It never copies the source population directories wholesale.

The generated directory is ignored through the repository-wide `scratch/` rule. Upload the **contents** of that directory to a private Kaggle Dataset named:

```text
kaggriculture-meta-search
```

Expected top-level bundle contents include `cuda-engine/`, `bonsai-kg/`, `tools/`, `actuators/`, `populations/`, `bundle_manifest.json`, and `README.md`.

## Search funnel

Stage 0 is the in-memory grammar normalization/legality filter applied during generation and mutation.

The default compiled search funnel is benchmark-driven at runtime through free-VRAM sizing, but its development evidence grows deliberately:

1. `stage1_massive_discovery`: 12 baseline-derived hard/guard families, both seats, exact trajectories through step 120; 55% of search walltime.
2. `stage2_family_core`: one hard representative from each of all 21 families, both seats, through step 240; 20%.
3. `stage3_medium`: two representatives per family (42 tapes), both seats, through step 480; 15%.
4. `stage4_full_development`: all 168 development tapes, both seats, full 720-step semantics; 10%.

After stages 1–3, each island also runs its current champion through an infrequent full 168-tape × both-seat, 720-step development gate. This detects tiny-core overfit before the final hours without exposing frozen data.

5. Final: exact-deduplicated elites from both islands receive the same full 168-tape × both-seat development screen at seed `2184000`.

The final rank prioritizes, in order: losing-family count, total losses, regressions versus exact v6 baseline, worst-family win rate, downside floor, overall win rate, mean margin, then program complexity.

## Kaggle notebook setup

1. Create/upload the private Kaggle Dataset above.
2. Import `notebooks/kaggle_dual_t4_meta_search.ipynb` as a Kaggle notebook.
3. Attach the `kaggriculture-meta-search` Dataset as notebook input.
4. In notebook settings, choose the accelerator that exposes **2 × NVIDIA T4** GPUs. The hardware probe deliberately aborts on fewer than two GPUs; with `REQUIRE_T4=True` it also aborts on another GPU model.
5. Internet is not required. The notebook compiles entirely from the attached bundle.
6. Run all cells in order. Do not manually alternate between GPU cells; both islands are launched concurrently and monitored by one orchestration loop.

The central configuration cell exposes walltime, island seeds, VRAM target, checkpoint cadence, final-screen candidate count, and resume mode. Each GPU is independently autotuned across multiple candidate batch sizes and CUDA block sizes before the campaign. The measured turn-step rate is converted into a per-stage candidates/hour estimate and a checkpoint-friendly population cap; the CUDA executable independently derives the safe population ceiling from live free VRAM. The production population is therefore the smaller of the throughput-driven cap and the 93%-free-VRAM ceiling, with automatic OOM fallback to lower fractions. This prioritizes sustained useful throughput and bounded generations over merely filling memory.

## Walltime and failure behavior

The nominal search budget is 9h20m, but the notebook subtracts measured setup time plus a 10-minute orchestration/development-gate reserve and a 30-minute final-screen/export reserve before launching. It aborts rather than overrun if too little walltime remains. Each search stage checks monotonic walltime after complete generations.

Every island has isolated output and archive directories. If one island fails, the monitor leaves the other running and may restart only the failed island once from its latest `checkpoint_best.bin`. The default archives/checkpoints every completed generation and writes atomically. A forced notebook guard may lose only the currently incomplete generation.

## Resume

Within the same Kaggle working session, set `RESUME=True` before rerunning the notebook and existing island checkpoints will be reused. If resuming in a fresh Kaggle session, attach a Dataset containing a previous `meta_search/` output tree, set `RESUME_DATASET_MOUNT` to that Dataset mount name, and set `RESUME=True`.

Resume is elite-checkpoint continuation, not bit-exact replay of the entire evolutionary population: completed stages are skipped and an incomplete/downstream stage is reseeded from its last canonical champion using the recorded deterministic search seed schedule. The checkpoint records RNG state for audit, but the current campaign does not reconstruct the full prior population/RNG stream.

## Outputs

The notebook writes:

```text
/kaggle/working/meta_search/
  config.json
  hardware.json
  island0/
    tuning.json
    stage*/archive/
    summary.json
  island1/
    tuning.json
    stage*/archive/
    summary.json
  merged/
    deduplicated_elites.jsonl
    final_screen_commands.json
    final_screen_gpu0.log
    final_screen_gpu1.log
    final_screen_results.jsonl
    nondominated.jsonl
    top_candidates.jsonl
    top_candidates/*.bin
    summary.json
    reproduction_commands.txt
```

and packages the complete compact result tree as:

```text
/kaggle/working/kaggriculture_dual_t4_meta_search_results.tar.gz
```

The final common screen is split across both T4s, but every candidate is evaluated on the same development cases; this is only parallel partitioning of candidates, not a change in evidence.

## Recover and import results

After the Kaggle run completes, download `kaggriculture_dual_t4_meta_search_results.tar.gz` from notebook outputs. Locally import it under a run-specific ignored directory, for example:

```bash
mkdir -p scratch/bonsai_meta_search_runs/2026-08-18-dual-t4
cd scratch/bonsai_meta_search_runs/2026-08-18-dual-t4
tar -xzf /path/to/kaggriculture_dual_t4_meta_search_results.tar.gz
```

Alternatively use the tracked safe importer:

```bash
python3 tools/meta_search/import_results.py /path/to/kaggriculture_dual_t4_meta_search_results.tar.gz \
  --run-id 2026-08-18-dual-t4
```

A future local session should inspect `meta_search/merged/top_candidates.jsonl` and its canonical 4096-byte candidate binaries, reproduce selected candidates in exact CUDA/C++, and distill only causal improvements into BKG3. Only after that should the 84 validation tapes, 50 blind tapes, or 28 family holdouts be exposed. Before any submission, still run the required 64/128-game official Python-engine canary.

The discovery notebook does not modify the champion, push Git, or submit to Kaggle.
