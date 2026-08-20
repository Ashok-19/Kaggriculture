#!/usr/bin/env bash
set -euo pipefail

MODEL_SIZE="${1:?usage: cloud_run.sh <20m|50m> <verify|smoke|benchmark|train|evaluate|package|all>}"
STAGE="${2:?usage: cloud_run.sh <20m|50m> <verify|smoke|benchmark|train|evaluate|package|all>}"
case "$MODEL_SIZE" in 20m|50m) ;; *) echo "invalid model size: $MODEL_SIZE" >&2; exit 2 ;; esac

ROOT="${BONSAI_REMOTE_ROOT:-/home/nnmax/Desktop/kaggle/kaggriculture}"
PERSIST_ROOT="${BONSAI_PERSIST_ROOT:-/teamspace/studios/this_studio/bonsai_editformer_scaling}"
RUN="$PERSIST_ROOT/runs/editformer${MODEL_SIZE}_v1"
PACKAGES="$PERSIST_ROOT/packages"
cd "$ROOT"
[[ -f .source_provenance.env ]] && source .source_provenance.env

export UV_PROJECT_ENVIRONMENT="$ROOT/scratch/bkg4_editformer_jax/venv"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export JAX_COMPILATION_CACHE_DIR="$ROOT/scratch/bkg4_editformer_jax/jax_compilation_cache"
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0
export PYTHONUNBUFFERED=1
mkdir -p "$RUN" "$PACKAGES" "$JAX_COMPILATION_CACHE_DIR"

CLI=(uv run --project tools/bkg4_search/bonsai_editformer python -u tools/bkg4_search/bonsai_editformer/cli.py)

verify() {
  "${CLI[@]}" stack-check --model-size "$MODEL_SIZE" --attention-impl cudnn | tee "$RUN/stack_check.log"
}

smoke() {
  "${CLI[@]}" smoke --model-size "$MODEL_SIZE" --attention-impl cudnn --limit 32 --batch-size 2 --steps 1 --seed 73 | tee "$RUN/smoke.log"
}

benchmark() {
  "${CLI[@]}" benchmark --model-size "$MODEL_SIZE" --attention-impl cudnn --batch-size 1024 --warmup 1 --steps 3 --gpu-sample-ms 100 --seed 73 --out "$RUN/benchmark.json" 2>&1 | tee "$RUN/benchmark.log"
}

train() {
  "${CLI[@]}" train-historical \
    --model-size "$MODEL_SIZE" --attention-impl cudnn --seed 73 \
    --batch-size 1024 --effective-batch-size 1024 --eval-batch 1024 \
    --epochs 10 --early-epoch 2 --middle-epoch 6 \
    --lr 3e-4 --weight-decay 1e-4 --clip 2.0 \
    --w-reg 6 --w-flip 4 --w-margin 1 --w-worst 3 --w-rank 6 --w-case 1.5 \
    --split primary --out "$RUN" 2>&1 | tee "$RUN/historical.log"

  "${CLI[@]}" calibrate \
    --model-size "$MODEL_SIZE" --attention-impl cudnn --seed 73 \
    --batch-size 128 --effective-batch-size 128 --eval-batch 400 \
    --epochs 20 --lr 1e-4 --weight-decay 1e-4 --clip 2.0 \
    --w-reg 6 --w-flip 4 --w-margin 1 --w-worst 3 --w-rank 6 --w-case 1.5 \
    --split primary --checkpoint "$RUN/history_final" --out "$RUN" 2>&1 | tee "$RUN/calibration.log"
}

evaluate() {
  "${CLI[@]}" evaluate --model-size "$MODEL_SIZE" --attention-impl cudnn --seed 73 \
    --checkpoint "$RUN/final" --batch-size 400 \
    --out "$RUN/current_rankings.jsonl" --report-out "$RUN/evaluation_result.json" 2>&1 | tee "$RUN/evaluation.log"
  "${CLI[@]}" compare --model-size "$MODEL_SIZE" --rankings "$RUN/current_rankings.jsonl" \
    --out "$RUN/search_comparison.json" --seed 123 --random-repeats 20 2>&1 | tee "$RUN/comparison.log"
}

package_results() {
  RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
  "${CLI[@]}" package-results --model-size "$MODEL_SIZE" --run-dir "$RUN" --run-id "$RUN_ID" --package-dir "$PACKAGES"
}

case "$STAGE" in
  verify) verify ;;
  smoke) smoke ;;
  benchmark) benchmark ;;
  train) train ;;
  evaluate) evaluate ;;
  package) package_results ;;
  all) verify; smoke; benchmark; train; evaluate; package_results ;;
  *) echo "invalid stage: $STAGE" >&2; exit 2 ;;
esac

printf 'persistent_run=%s\n' "$RUN"
printf 'packages=%s\n' "$PACKAGES"
