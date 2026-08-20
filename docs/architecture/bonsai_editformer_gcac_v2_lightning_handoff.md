# BonsaiEditFormer-GCAC-v2 — Lightning Stage-B2 Handoff

Date: 2026-08-20

This handoff starts **training only**. It does not authorize Stage-C joint training, 20M/50M scaling, champion changes, pushes or Kaggle submissions.

## Architecture identity

- Architecture: `BonsaiEditFormer-GCAC-v2`
- Total parameters: `6,187,787`
- Proven frozen critic/shared parameters: `4,991,412`
- Trainable actor/goal/directional parameters: `1,196,375`
- Frozen shared digest: `514efa27ef9df1057d3b8809c3cd4651032b29f933ccbece36e66d9e24854909`
- B371 semantic parent: `b3714ce680b84a74233b053c5bc32f48a37e3c257b46b4e131a0f784b4534286`
- Implementation base before GCAC-v2 edits: `25a7ba41057c2e7d5f0dad39804fbaa6dafee198`
- Exact bundle source HEAD is written into `.source_provenance.env` by the bundle builder.

## Fixed Stage-B2 dataset

Path in bundle:

`scratch/bkg4_editformer_jax/actor/gcac_v2_transitions_full.npz`

- bytes on local filesystem: ~132 MiB
- dataset SHA-256: `e5ac00d899f811774b400dc84ae5e5508b5884610969e78ea611e2cbd1eb098c`
- manifest SHA-256: `e688c7ae261add7e9a623a77569825ab7480a4c4db7e784bb1ae7c2fb10176e2`
- transitions: 60,000
- hindsight failure-goal examples: 1,054,511
- train examples: 841,515
- validation examples: 212,996
- safe-positive examples: 156,010
- non-positive/unsafe/no-effect examples: 898,501
- relabel leakage: 0

Goal classes:

- safe L→W: 82,857
- safe improve: 73,153
- unsafe improve: 241,072
- no effect: 221,053
- target harm: 43,486
- severe regression: 392,890

Split identities:

- train example hash: `f0b041fb64f7caeb21705dbcc83f501ff1cb955ffb3ab977b428c627b6861b37`
- val example hash: `7ad16674911085f5846e981e4bfe783e44f333bd11660678de90a56dd9a8603e`
- transition-train hash: `30b158ff7f2d0ce9d3d5f463045469da70731f8ec3787a85d48c28a9f6af5244`
- transition-val hash: `d8587fba791b9f08a151b043b2f16782bdcf5088f892875af1ff58cbeff98e8b`

Default learned-operation gate is 32 safe-positive goal examples. It masks only:

- `INSERT_RECOVERY`
- `REPLACE_MACRO`
- `SPLIT_LANE`

The grammar still supports them for later active learning.

## Fresh24 semantic failure context

Path in bundle:

`scratch/bkg4_editformer_jax/gcac_v2/fresh24_goal_context.json`

The capture is the frozen 345-node champion legal public-state trajectory at steps 264..287. B371/candidate ModuleGenome activation is evaluated offline over those states. This is intentionally the same kind of frozen-state activation context used by the proven critic; it is **not** claimed to be a B371-mutated trajectory capture.

- capture SHA-256: `d6b74db20146784bfdf394fde018dd3026eb7181380e9fa58a3b690306738efc`
- cases: 24
- unique seeds: 6
- both seats per seed: true
- B371 loss rows: `2, 3, 4, 5, 16, 17`
- exact loss margins: `-3204, -7518, -16061, -17341, -3908, -3908`
- identity features embedded in model: false

## Local verification completed

Focused GCAC-v2 unit suite: 14/14 pass after final pair-relative scoring regression coverage.

Four-update bounded GPU smoke:

- loss: `14.9733458 -> 2.5443902`
- finite gradients: true
- critic frozen exact: true
- checkpoint roundtrip exact: true
- same parent / different goal changes operation, pointer and semantic logits: true
- balanced validation 512 rows:
  - target-positive 160
  - target-harm 80
  - no-effect 100
  - severe-regression 90
  - unsafe-improve 82
  - operation families 10
- typed grammar smoke: 32/32 valid and unique

Directional-head digest changed across those four updates while the shared digest remained unchanged:

- init directional digest: `740237cfa79a3c17b38c63ca84fd12044ca83b1f46edd4bb228ee13d58dc52a1`
- smoke directional digest: `40435acacdd10577fa1b01560949dc9087d1e88908565877fab28c48dbd77eaf`
- directional parameter elements: 104,548

Small fresh24 generation smoke:

- requested 48
- global unique after per-goal dedup: 46
- all 6 goals covered
- masked unsupported operations generated: 0
- strategic positive-repair claims: none; four local updates are not meaningful training.

## Lightning layout

Expected repo after bundle extraction:

`/teamspace/studios/this_studio/kaggriculture`

Persistent result root:

`/teamspace/studios/this_studio/bonsai_gcac_v2`

The exact runner is:

`tools/bkg4_search/bonsai_editformer/scripts/gcac_v2_cloud_run.sh`

## Exact Stage commands

Run from `/teamspace/studios/this_studio/kaggriculture` after extracting the prepared bundle.

### A. Environment verification

```bash
./tools/bkg4_search/bonsai_editformer/scripts/gcac_v2_cloud_run.sh verify
```

### B. Optional short cloud smoke

```bash
./tools/bkg4_search/bonsai_editformer/scripts/gcac_v2_cloud_run.sh smoke
```

### C. Stage-B2 goal-conditioned actor training

```bash
./tools/bkg4_search/bonsai_editformer/scripts/gcac_v2_cloud_run.sh train
```

Training configuration is fixed in the runner: 5M frozen critic backbone, batch 1024, 65,536 class-prioritized goal examples/epoch, 10 epochs = 640 optimizer updates, LR `8e-4`, BF16/cuDNN, actor/goal/directional parameters only.

### D. Balanced held-out evaluation

```bash
./tools/bkg4_search/bonsai_editformer/scripts/gcac_v2_cloud_run.sh evaluate
```

### E. Per-failure generation

```bash
./tools/bkg4_search/bonsai_editformer/scripts/gcac_v2_cloud_run.sh generate
```

### F. Six-lane generation benchmark

```bash
./tools/bkg4_search/bonsai_editformer/scripts/gcac_v2_cloud_run.sh benchmark
```

Uses 64 CUDA candidates/lane and 4× proposal pool for actor/heuristic critic lanes.

### G. Exact CUDA campaign

```bash
./tools/bkg4_search/bonsai_editformer/scripts/gcac_v2_cloud_run.sh cuda
```

The runner compiles the exact evaluator with the locally validated command:

`nvcc -O3 -std=c++17`

and evaluates the same six fresh seeds / both seats.

### H. Result packaging

```bash
./tools/bkg4_search/bonsai_editformer/scripts/gcac_v2_cloud_run.sh package
```

Expected package:

`/teamspace/studios/this_studio/bonsai_gcac_v2/packages/bonsai_gcac_v2_results.tar.gz`

The package retains the complete final GCAC-v2 checkpoint for reproducibility/resume; critic-initialization and intermediate checkpoints are reduced to metadata.

### I. Resume one additional Stage-B2 epoch

Only use if a Stage-B2 run was interrupted or an explicitly justified extra frozen-actor epoch is needed:

```bash
./tools/bkg4_search/bonsai_editformer/scripts/gcac_v2_cloud_run.sh resume
```

Do not use resume to hide a failed directionality result.

## Decision gate after download

Minimum architecture repair:

- `actor_critic.target_improving_candidates > 0` for explicitly requested B371 failures.

Strong pass:

- at least one fresh actor-generated L→W.

Preferred pass:

- at least one strict-safe actor L→W with `W→L = 0` and `reg3 = 0`.

If every explicitly targeted loss still has zero positive actor delta, do not unfreeze or scale. Diagnose grammar capacity, corpus repair support, goal representation, signed critic calibration, pointer reach, or need for multi-edit search first.
