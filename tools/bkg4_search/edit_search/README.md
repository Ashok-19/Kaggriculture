# Permanent BKG4 Edit Search

`tools/bkg4_search/edit_search/` is the reusable recursive typed structural-search subsystem for ReCurSiON's BKG4 Bonsai agent.

It is intentionally separate from the old replay-derived one-generation edit-corpus builder. The permanent engine accepts a concrete BKG4 parent profile, keeps a persistent quality-diversity archive of exact semantic states, proposes fully specified typed edit programs, evaluates them with exact B4EC1 CUDA truth, feeds broader-panel counterexamples back into future screening, and exports strict exact-labeled BonsaiEditFormer training corpora.

## Non-negotiable invariants

- BKG4/Bonsai remains the deterministic policy backbone. This package does not create BKG5 or replace the champion automatically.
- Search/evaluation truth is exact CUDA (`B4EC1/v1`). There is no CPU or Python game-evaluator fallback in `ExactCudaEvaluator`.
- Concrete search states are parent-specific. A future BKG4 parent gets a fresh archive; only symbolic motifs may be re-instantiated, and every re-instantiated motif must be reevaluated.
- Production ranking is safety-first and lexicographic. Aggregate margin gains cannot compensate for new losses or severe protected-win regressions.
- Search preserves unsafe or specialist stepping stones only in exploration/QD structures; unsafe states do not enter the production-safe archive.
- Model inputs are restricted to legal/public state context plus parent structure/EditProgram semantics. `prov_*`, `search_*`, candidate screening scores and final outcome labels are never model feature columns.
- Large campaigns belong on CUDA. Before any candidate is promoted to a submission, validate the final deterministic CPU C++ artifact and run a small 64/128-game official Python-engine canary.

## Architecture

The permanent package is split into narrow components:

- `schema.py` — parent profiles and canonical state-relative `EditProgram`/`EditAtom` identities.
- `emitters.py` — systematic, local, structural, motif, semantic recombination, large-neighborhood, evolvability/MCTS and permanent random-typed emitters.
- `archive.py` — persistent production/exploration QD niches, novelty archive, motifs, exact certifications and hard counterexamples.
- `panels.py` — deterministic mixed screening panels and hard-case prioritization.
- `fitness.py` — exact parent→child labels and safety-first lexicographic fitness.
- `evaluator.py` — one batched B4GM1→B4EC1 exact CUDA invocation per state batch; no fallback.
- `search.py` — recursive parent selection, lexicase specialists, multi-fidelity Stage 1/Stage 2 evaluation and anti-stagnation escalation.
- `corpus.py` — append-only exact Parquet export plus fail-closed reconstruction/referential/label validation.
- `benchmark.py` — equal-budget proposer comparison and anti-saturation harness.
- `cli.py` — persistent lifecycle and validation commands.

The default fixed emitter scheduler is deliberate. An adaptive UCB allocator was not retained because no fixed-vs-adaptive ablation demonstrated a benefit. Exploration floors therefore remain explicit and reproducible.

## Current semantic ceiling

The current backend searches the frozen BKG4 portable route-selector `ModuleGenome`: up to eight ordered rules, up to four legal/public predicates per rule, one frozen route-bank choice per rule, and a searchable activation prefix/window. It can recursively produce 2+ rule parents and all applicable typed operations, including `CHANGE_PRIORITY` once a state has multiple rules.

This is a semantic ceiling of the **current backend**, not of the permanent search infrastructure. If a future BKG4 architecture adds a richer portable module grammar, add a separately versioned grammar/EditProgram backend and evaluator packing contract. Do not silently reinterpret existing `GRAMMAR_ID`, search-state, QD descriptor or corpus schema versions.

## Build the exact evaluator

The permanent wrapper uses `cuda-engine/src/bkg4_edit_corpus_eval.cu` and the same B4EC1 ABI used by the edit-corpus builder.

```bash
make -C cuda-engine bkg4-edit-corpus-eval
```

The target intentionally exposes separate B4EC flags because this evaluator is also compiled as a standalone support-dataset source. On local CUDA 11.5 installations whose default GCC is too new, select a supported host compiler explicitly; for an RTX 3050 (`sm_86`) local validation build:

```bash
make -C cuda-engine -B bkg4-edit-corpus-eval \
  B4EC_HOST_CXX=g++-9 \
  B4EC_NVCCFLAGS='-O3 -std=c++17 -lineinfo -arch=sm_86'
```

Do not enable `--use_fast_math`; market/parity truth requires exact arithmetic behavior.

## Persistent lifecycle

All commands are run from the repository root.

Create a fresh concrete archive for a parent:

```bash
python3 -m tools.bkg4_search.edit_search.cli init \
  --parent-dir bonsai-kg/integrations/bkg4_champion/champions/w33_s172_expansion_v1 \
  --archive scratch/edit_search/run/search.json \
  --run-id recur-001 \
  --seed 99173 \
  --root-prefix 0
```

Inspect/resume it:

```bash
python3 -m tools.bkg4_search.edit_search.cli validate --archive scratch/edit_search/run/search.json
python3 -m tools.bkg4_search.edit_search.cli resume --archive scratch/edit_search/run/search.json
```

Proposal-only debugging is available without evaluation:

```bash
python3 -m tools.bkg4_search.edit_search.cli propose \
  --archive scratch/edit_search/run/search.json \
  --observations scratch/edit_search/observations.jsonl \
  --count 256 --generation 1 --seed 99173 \
  --out scratch/edit_search/run/proposals_g1.jsonl
```

A production search round requires a `CaseBank` JSONL with legal/public ranker context and the exact evaluator:

```bash
python3 -m tools.bkg4_search.edit_search.cli search \
  --archive scratch/edit_search/run/search.json \
  --parent-dir bonsai-kg/integrations/bkg4_champion/champions/w33_s172_expansion_v1 \
  --case-bank scratch/edit_search/case_bank.jsonl \
  --observations scratch/edit_search/observations.jsonl \
  --evaluator scratch/cuda-engine-build/bkg4_edit_corpus_eval \
  --corpus scratch/edit_search/run/corpus \
  --work-dir scratch/edit_search/run/work \
  --start-generation 1 --rounds 1 \
  --candidates 4096 --screen-size 128 --final-size 512 --stage2-budget 512 \
  --seed 99173
```

The search command persists the archive after every completed generation and appends only exact Stage-2 rows to the corpus. Reusing an already-exported generation fails closed.

Validate an exported exact corpus:

```bash
python3 -m tools.bkg4_search.edit_search.cli validate-corpus \
  --corpus scratch/edit_search/run/corpus \
  --archive scratch/edit_search/run/search.json
```

## Equal-budget benchmark

The benchmark compares four proposal systems with identical **child × case exact-CUDA budgets**:

1. historical shallow `build_edit_plan`,
2. random typed single-parent mutation,
3. fixed systematic single-parent mutation,
4. permanent recursive/QD search.

The root evaluation is shared and does not bias one arm's child budget.

```bash
python3 -m tools.bkg4_search.edit_search.cli benchmark \
  --parent-dir bonsai-kg/integrations/bkg4_champion/champions/w33_s172_expansion_v1 \
  --cases-tsv scratch/bkg4_edit_search_validation/heldout16_cases7.tsv \
  --observations scratch/bkg4_edit_search_validation/current140_observations.jsonl \
  --evaluator scratch/cuda-engine-build/bkg4_edit_corpus_eval \
  --work-dir scratch/bkg4_edit_search_validation/cli_benchmark \
  --candidate-budget 64 --seed 99173 --memory-fraction 0.55 \
  --report scratch/bkg4_edit_search_validation/cli_benchmark_report.json \
  --require-material-improvement
```

`--require-material-improvement` returns nonzero unless permanent recursive search beats the old shallow path on at least two structural/search dimensions, does not reduce safe-useful yield, and does not worsen top-candidate W→L rate.

## Anti-saturation harness

The anti-saturation command first exact-scores the supplied B4EC1 cases to establish current root margins, then runs a bounded persistent recursive search while recording raw proposal pressure, duplicate rejection, archive/QD growth, emitter yield, source-parent diversity, operation coverage and escalation events.

```bash
python3 -m tools.bkg4_search.edit_search.cli anti-saturation \
  --parent-dir bonsai-kg/integrations/bkg4_champion/champions/w33_s172_expansion_v1 \
  --cases-tsv scratch/bkg4_edit_search_validation/heldout16_cases7.tsv \
  --observations scratch/bkg4_edit_search_validation/current140_observations.jsonl \
  --evaluator scratch/cuda-engine-build/bkg4_edit_corpus_eval \
  --work-dir scratch/bkg4_edit_search_validation/cli_anti_saturation \
  --generations 8 --candidates-per-generation 24 \
  --screen-size 8 --final-size 16 --stagnation-patience 1 \
  --seed 99173 --memory-fraction 0.55 \
  --report scratch/bkg4_edit_search_validation/cli_anti_saturation_report.json \
  --require-escalation
```

Escalation levels expand the search regime in order: wider scalar/window/route neighborhoods; broader QD parents/structural edits; semantic recombination; large-neighborhood destroy/repair; deeper evolvability/MCTS; then novelty/rare-niche/hard-case refresh. A permanent random-typed floor remains nonzero.

## New-parent procedure

When BKG4 changes:

1. Create a fresh archive with the new `ParentProfile`.
2. Optionally pass `--motifs-from OLD_ARCHIVE` during `init` to copy symbolic motif templates only.
3. Re-instantiate those motifs against targets valid in the new parent.
4. Reevaluate every concrete child with exact CUDA. Never import old concrete elite fitness as truth for the new parent.
5. Run the equal-budget benchmark and anti-saturation gate on a current held-out league panel before scaling.

## Promotion gate

Search is not promotion. Before an exact-CUDA discovery modifies the official ReCurSiON champion:

1. Freeze the exact candidate semantic identity and assets.
2. Confirm the candidate result translates exactly to the CPU C++ engine using the existing BKG4 parity harness.
3. Run a 64- or 128-game official Python Kaggriculture canary against a representative current/historical/own-history panel.
4. Reject any unexplained CUDA↔CPU↔official-engine discrepancy or regression.
5. Only then consider a champion write/submission under the repository's serialized promotion workflow.

See `VALIDATION.md` for the bounded evidence supporting the current implementation.
