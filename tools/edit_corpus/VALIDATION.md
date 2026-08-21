# Edit-corpus builder validation report

Date: 2026-08-21

## Scope

Validation is intentionally bounded. No ranker training, champion modification, competition submission, or large local CPU campaign was run.

## Python regression suite

Command:

```bash
python3 -m unittest discover -s tools/edit_corpus -p 'test_*.py' -v
```

Result: **36/36 passed**.

Coverage includes replay action-offset semantics, both-seat case construction, deterministic hard/fragile selection, provenance/model-feature separation, typed edit legality and semantic dedup, broad operation-family generation, B4EC1 parsing/corruption checks, parent-relative action-trace behavior labels, append/idempotency/source-mutation rejection, NumPy/Parquet list-column round trips, global target recomputation, semantic edit dedup counts, support allowlist packaging, deterministic notebook generation, GPU/input metadata and publication gating.

## Existing replay-archive regression suite

Command:

```bash
python3 -m unittest \
  tools.replay_archive.test_top10_archive \
  tools.replay_archive.test_build_notebook \
  -v
```

Result: **19/19 passed**.

## Notebook generation

Command:

```bash
python3 tools/edit_corpus/build_notebook.py --check
```

Result: `notebook generation check passed`.

The generated notebook has GPU + internet enabled and exactly three dataset inputs: replay archive, edit-corpus support bundle, and persistent edit corpus.

## Support bundle

Final pre-commit working-tree bundle size: **2,339,154 bytes** (~2.23 MiB), with explicit allowlisted runtime/source/assets only. Both CUDA programs compiled directly from the packaged paths using the same commands used by the notebook:

```text
nvcc -O3 -std=c++17 -lineinfo ... bkg4_edit_corpus_eval.cu
nvcc -O3 -std=c++17 -lineinfo ... bkg4_information_state_capture.cu
```

Both builds succeeded.

## Exact CUDA evaluator parity

Tiny fixture: 4 known current-BKG4 cases, 3 overlay genomes, 12 CUDA games total.

Packaged edit evaluator result:

```text
B4EC1 genomes=3 cases=4 games=12
gpu_ms=337.912
action_changes=1884
activations=2370
bad=0
```

For the empty overlay, margins exactly matched the existing proven current-BKG4 CUDA fragment evaluator:

```text
[13116, 3422, 19729, 14006]
```

The empty overlay reported zero overlay action changes and zero activations. Executable-action trace fingerprints were produced per case for parent-relative no-op/behavior labels.

## Legal/public information-state capture

Packaged capture tool result on 4 cases × 2 probes:

```text
r0 cases=4 probes=2 width=3833 registry=3793 bb=23 compat=15 meta=2
gpu_ms=275.448
bad=0
output_MiB=0.116974
```

`load_public_context()` parsed 4 contexts and 953 observed threshold tuples. The corpus path reads only `B4ISR01` legal/public capture. The separate hidden-truth diagnostic output is never loaded into corpus/model context.

## End-to-end replay -> exact-CUDA corpus smoke

A real archived replay (`episode 94579514`) was placed into a one-day synthetic archive wrapper and run through the actual builder transaction with bounded settings.

Completed transaction before consolidation:

```text
source date: 2026-08-20
representative cases: 4
semantic edits: 8
exact CUDA edit x case labels: 32
parents persisted: 9
processed day marker: complete
```

After fixing the discovered NumPy/Parquet list canonicalization edge case, the already-produced shards consolidated and validated successfully:

```text
cases.parquet: 4 rows
edits.parquet: 8 rows
edit_case_labels.parquet: 32 rows
edit_targets.parquet: 8 rows
```

Smoke operation distribution:

```text
ADD_SPECIALIST: 3
ADD_CONDITION: 1
CHANGE_ROUTE: 1
CHANGE_THRESHOLD: 1
CHANGE_WINDOW: 1
INSERT_RECOVERY: 1
```

The small smoke intentionally does not try to cover every grammar family; the full deterministic generator tests exercise broader legal families and the normal notebook uses substantially larger candidate counts.

The smoke contained both positive and negative supervision, including L->W, W->L, severe regressions, strict-safe edits, behavior-changing edits and exact parent-relative no-ops.

## Idempotency / incremental preservation

The completed smoke corpus was used as a prior corpus and rerun against the same replay archive. Result:

```text
pending replay dates []
processed_this_run []
cases 4
edits 8
labels 32
```

A second true no-change rerun reproduced the same corpus manifest SHA exactly:

```text
35bf69ce814fb1ac4c068391df0666cc444b2f8f556be1298a7b762f7d81b82a
```

Unit tests additionally cover appending a new date while retaining prior rows and reject mutation of an already-processed source shard.

## Storage behavior

The builder reads one `replays_YYYY-MM-DD.parquet` source shard at a time directly from the mounted replay dataset. Temporary native tapes, case TSVs, capture binaries/results and genome/result chunks live only under the per-run temporary directory and the per-day directory is deleted in a `finally` block. An up-to-date run discovers zero pending dates before CUDA compilation, so it does not waste GPU/compiler time.

## Remote Kaggle smoke

The generated notebook contains `SMOKE_ONLY=True/False` and `PUBLISH=True/False` switches specifically for a bounded Kaggle-GPU validation. The local implementation has been tested on an NVIDIA CUDA GPU and the generated kernel metadata requests a Kaggle GPU. A remote Kaggle execution could not be launched from the current local credentials: even the read-only command `kaggle datasets status ashok205/kaggriculture-edit-corpus-support` returned HTTP 403. No remote write was attempted after that permission failure. Use the exact smoke instructions in `tools/edit_corpus/README.md` once Kaggle dataset access is available; the generated kernel metadata requests GPU and the packaged CUDA tools have already passed local exact-parity smoke tests.
