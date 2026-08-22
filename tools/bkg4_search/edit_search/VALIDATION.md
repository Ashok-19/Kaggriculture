# Permanent BKG4 Edit Search Validation

Date: 2026-08-22

## Scope

This validation covers the permanent recursive edit-search infrastructure only. It does **not** promote or modify the BKG4 champion, submit to Kaggle, train BonsaiEditFormer, or run a large local search campaign.

The objective of the bounded validation is to prove:

- canonical typed EditPrograms reconstruct exactly,
- persistent QD state and hard cases survive resume,
- recursive search can use evaluated descendants as future parents,
- exact CUDA is the only production evaluation path,
- broader-panel failures become counterexamples,
- exported corpus rows reconstruct/recompute exactly,
- the permanent proposer materially improves search yield over the old shallow generator at equal exact-CUDA budget,
- increasing duplicate pressure causes actual search-regime escalation instead of silent neighborhood exhaustion.

## Exact evaluator build

Local machine:

```text
GPU: NVIDIA GeForce RTX 3050 Laptop GPU (sm_86)
CUDA compiler: nvcc 11.5
```

The system default GCC 11 is incompatible with this local nvcc release for the current C++ standard-library headers. The Makefile target therefore supports an explicit host compiler. The exact local validation build was:

```bash
make -C cuda-engine -B bkg4-edit-corpus-eval \
  B4EC_HOST_CXX=g++-9 \
  B4EC_NVCCFLAGS='-O3 -std=c++17 -lineinfo -arch=sm_86'
```

Result: successful build of `scratch/cuda-engine-build/bkg4_edit_corpus_eval`.

The evaluator source/ABI is unchanged; this Makefile addition only makes the existing B4EC1 evaluator a reproducible target and exposes host/compiler flags.

## Exact CUDA empty-overlay smoke

Parent:

```text
combined identity: 6359ab400f9d128945cdbba5df7178a33b88c643ddd9b488299d66e14daa6925
controller hash:   8c5d7cd0e32574cbe349d5964946a9e2d6e7a75d766708f5d82308cd8507a034
BKG4 ABI:          4
feature registry:  1
```

Four-case bounded smoke:

```text
B4EC1 genomes=1 cases=4 games=4
gpu_ms=268.37
games_s=14.9048
action_changes=0
activations=0
bad=0
```

Empty-overlay margins exactly reproduced the previously established fixture:

```text
[13116, 3422, 19729, 14006]
```

The current 140-case bank root scan also completed with exact CUDA:

```text
B4EC1 genomes=1 cases=140 games=140
gpu_ms=498.271
games_s=280.971
action_changes=0
activations=0
bad=0
```

That root scan contained 42 incumbent losses and 32 wins within the 5,000-margin fragile band. Proposal thresholds were derived only from the public R0 capture; 3,157 legal observed threshold rows were available for the held-out benchmark.

## Fresh first-class CLI equal-budget benchmark

Command:

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

Result: exit code 0, `material_improvement=true`.

The held-out panel has 16 current hard-loss cases balanced 8/8 by seat across 9 strategy families. Every arm receives exactly 64 evaluated children × 16 cases = **1,024 exact child×case evaluations**. The root truth is shared.

| Metric | Old shallow | Random typed | Fixed systematic | Permanent recursive |
| --- | ---: | ---: | ---: | ---: |
| strict-safe L→W / 1,000 | 78.125 | 187.5 | 453.125 | **343.75** |
| safe-useful / 1,000 | 93.75 | 265.625 | 484.375 | **468.75** |
| target-loss improving / 1,000 | 609.375 | 421.875 | 828.125 | **593.75** |
| top-quartile W→L rate | 0 | 0 | 0 | **0** |
| useful structural diversity | 4 | 1 | 1 | **8** |
| maximum generation depth | 2 | 1 | 1 | **5** |
| operation coverage | 10 | 2 | 1 | 6 |
| QD structural niches | 42 | 16 | 5 | 24 |

The permanent recursive arm therefore produced **4.4×** the strict-safe L→W candidate rate of the old shallow generator and **2.0×** the useful structural diversity while preserving zero top-quartile W→L. Its safe-useful rate was **5.0×** the old shallow rate (468.75 vs 93.75 per 1,000).

The fixed systematic arm is intentionally retained as evidence: on this narrow all-loss panel it has higher raw repair yield, but only one operation family, one source-parent structural depth and one useful structural shape. It is an exploitation component, not a replacement for persistent recursive/QD search.

## Fresh first-class CLI anti-saturation probe

Command:

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

Result: exit code 0. Root scoring confirmed 16/16 cases are losses for this deliberately hard panel.

Duplicate pressure rose while the engine continued filling every requested generation:

```text
G1  duplicate=20.51%  unique=24/24  level=0
G2  duplicate=43.18%  unique=24/24  level=0
G3  duplicate=58.62%  unique=24/24  level=0
G4  duplicate=67.95%  unique=24/24  level=0
G5  duplicate=74.47%  unique=24/24  level=1
G6  duplicate=77.27%  unique=24/24  level=2
G7  duplicate=81.34%  unique=24/24  level=3
G8  duplicate=81.34%  unique=24/24  level=4
```

Observed escalation events:

```text
level 1: wider scalar/window/route neighborhoods
level 2: broader QD parents and structural mutation
level 3: semantic recombination across distant niches
level 4: large-neighborhood destroy/repair
```

Final persistent search state:

```text
states:             193
production niches:  44
exploration niches: 87
novelty entries:     7
final level:          4
```

Source-parent diversity expanded from one root in generation 1 to as many as 15 different archived parents. Operation coverage reached seven families during the run. Underfill remained 0.0 for every generation.

This demonstrates the intended operational anti-saturation property: duplicate pressure is measured, triggers concrete changes in the search regime, and does not silently collapse candidate throughput. It is not a claim of mathematical open-endedness.

## Training-corpus boundary

The exact search-corpus exporter stores:

- reconstructable parent and child `ModuleGenome` structures,
- full canonical EditProgram JSON,
- operation/target semantics,
- exact per-case margins/flip/regression/trace-effect labels,
- sample category and split grouping keys,
- search provenance and QD descriptors in `search_*` fields.

Only the declared legal/public case model columns plus `parent_genome_json` and `edit_program_json` are model feature columns. The validator rejects provenance/search fields in the declared model feature boundary, unresolved parent/child/case references, duplicate program×case rows, reconstruction mismatches and exact label recomputation errors.

## Promotion status

No BKG4 champion asset was modified or promoted by this implementation task. Therefore an official Python 64/128-game **promotion** canary is not applicable to a candidate yet.

When the search produces a concrete candidate selected for promotion, the mandatory sequence remains:

1. exact CUDA certification,
2. exact CPU C++ translation/parity,
3. 64/128-game official Python-engine validation,
4. only then champion promotion/submission.

## Final regression commands

The terminal integration gate for this implementation is:

```bash
python3 -m unittest discover -s tools/bkg4_search/edit_search -p 'test_*.py' -v
python3 -m unittest discover -s tools/edit_corpus -p 'test_*.py' -v
python3 -m unittest bonsai-kg/tests/test_bkg4_search.py -v
python3 bonsai-kg/tests/bkg4_champion_native_parity.py --seeds 1 --assets bonsai-kg/integrations/bkg4_champion/champions/w33_s172_expansion_v1
python3 -m compileall -q tools/bkg4_search/edit_search
git diff --check
```

Fresh terminal results for this implementation pass:

```text
permanent edit-search suite: 32/32 passed
legacy edit-corpus suite:     59/59 passed
existing BKG4 search suite:    4/4 passed
Python compileall:             exit 0
git diff --check:              exit 0
native differential parity:    2 cases / 1,438 VM records, C++ mismatches=0, CUDA mismatches=0
```

The native parity command was run through `uv run --with kaggle-environments==1.32.7` because the system Python lacked `jsonschema`; the first plain-Python attempt failed before any simulation executed.
