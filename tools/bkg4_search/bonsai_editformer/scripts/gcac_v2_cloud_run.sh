#!/usr/bin/env bash
set -euo pipefail

STAGE="${1:?usage: gcac_v2_cloud_run.sh <verify|smoke|train|evaluate|generate|benchmark|cuda|package|resume>}"
ROOT="${BONSAI_REMOTE_ROOT:-/teamspace/studios/this_studio/kaggriculture}"
PERSIST_ROOT="${BONSAI_PERSIST_ROOT:-/teamspace/studios/this_studio/bonsai_gcac_v2}"
RUN="$PERSIST_ROOT/run"
PACKAGES="$PERSIST_ROOT/packages"
DATASET="$ROOT/scratch/bkg4_editformer_jax/actor/gcac_v2_transitions_full.npz"
GOAL_CONTEXT="$ROOT/scratch/bkg4_editformer_jax/gcac_v2/fresh24_goal_context.json"
CRITIC="$ROOT/bonsai_editformer5m_results/final"
INIT="$RUN/critic_initialized"
CUDA_EXE="$ROOT/scratch/bkg4_editformer_jax/gcac_v2/bkg4_current_module_eval"
CUDA_SRC="$ROOT/scratch/bkg4_rl_synthesis_aug20/bkg4_current_module_eval.cu"
CASES="$ROOT/run/fresh24_cases.tsv"
CASES_MANIFEST="$ROOT/run/fresh24_cases.manifest.json"
CHAMP="$ROOT/bonsai-kg/integrations/bkg4_champion/champions/w33_s172_expansion_v1"

cd "$ROOT"
[[ -f .source_provenance.env ]] && source .source_provenance.env
export UV_PROJECT_ENVIRONMENT="$ROOT/scratch/bkg4_editformer_jax/venv"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export JAX_COMPILATION_CACHE_DIR="$ROOT/scratch/bkg4_editformer_jax/jax_compilation_cache"
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0
export PYTHONUNBUFFERED=1
mkdir -p "$RUN" "$PACKAGES" "$JAX_COMPILATION_CACHE_DIR" "$(dirname "$CUDA_EXE")"

ACTOR=(uv run --project tools/bkg4_search/bonsai_editformer python -u tools/bkg4_search/bonsai_editformer/actor_cli.py)
CRITIC_CLI=(uv run --project tools/bkg4_search/bonsai_editformer python -u tools/bkg4_search/bonsai_editformer/cli.py)

init_critic() {
  if [[ ! -f "$INIT/_CHECKPOINT_METADATA" ]]; then
    "${ACTOR[@]}" init-critic \
      --dataset "$DATASET" --attention-impl cudnn --batch-size 32 --seed 73 \
      --critic-checkpoint "$CRITIC" --out "$INIT" | tee "$RUN/init_critic.log"
  fi
}

compile_cuda() {
  if [[ ! -x "$CUDA_EXE" || "$CUDA_SRC" -nt "$CUDA_EXE" ]]; then
    nvcc -O3 -std=c++17 -o "$CUDA_EXE" "$CUDA_SRC"
  fi
}

verify() {
  test -f "$DATASET"
  test -f "${DATASET%.npz}.manifest.json"
  test -f "$GOAL_CONTEXT"
  test -d "$CRITIC"
  test -f "$CASES"
  test -f "$CASES_MANIFEST"
  uv run --project tools/bkg4_search/bonsai_editformer python -c \
    "from gcac_context import load_goal_context_manifest; m=load_goal_context_manifest('scratch/bkg4_editformer_jax/gcac_v2/fresh24_goal_context.json', expected_parent='b3714ce680b84a74233b053c5bc32f48a37e3c257b46b4e131a0f784b4534286'); print('goal_context_ok', m['loss_case_ids'], m['capture_sha256'])"
  "${CRITIC_CLI[@]}" stack-check --model-size 5m --attention-impl cudnn | tee "$RUN/stack_check.log"
  init_critic
  compile_cuda
  "$CUDA_EXE" >/dev/null 2>&1 || rc=$?; [[ "${rc:-2}" -eq 2 ]]
  printf 'verify_ok source_head=%s dataset=%s goal_context=%s\n' "${BONSAI_SOURCE_HEAD:-unknown}" "$DATASET" "$GOAL_CONTEXT"
}

smoke() {
  "${ACTOR[@]}" smoke \
    --dataset "$DATASET" --attention-impl cudnn --batch-size 32 --seed 73 \
    --critic-checkpoint "$CRITIC" --steps 2 --eval-samples 512 \
    --direction-loss-weight 0.5 --lr 1e-3 --weight-decay 1e-4 --clip 2.0 \
    2>&1 | tee "$RUN/smoke.log"
}

train() {
  init_critic
  "${ACTOR[@]}" train-actor \
    --dataset "$DATASET" --attention-impl cudnn --batch-size 1024 --seed 73 \
    --checkpoint "$INIT" --out "$RUN" --epochs 10 --early-epoch 2 --middle-epoch 6 \
    --eval-samples 1024 --examples-per-epoch 65536 --direction-loss-weight 0.5 \
    --lr 8e-4 --weight-decay 1e-4 --clip 2.0 \
    2>&1 | tee "$RUN/actor_training.log"
}

evaluate() {
  "${ACTOR[@]}" evaluate-actor \
    --dataset "$DATASET" --attention-impl cudnn --batch-size 1024 --seed 73 \
    --checkpoint "$RUN/final" --eval-samples 1024 --out "$RUN/actor_evaluation_result.json" \
    2>&1 | tee "$RUN/actor_evaluation.log"
}

generate() {
  "${ACTOR[@]}" evaluate-generation \
    --checkpoint "$RUN/final" --dataset "$DATASET" --goal-context "$GOAL_CONTEXT" \
    --attention-impl cudnn --base-semantic b3714ce680b84a74233b053c5bc32f48a37e3c257b46b4e131a0f784b4534286 \
    --samples 512 --seed 73 --temperature 0.9 --topk 8 --diversity-penalty 0.35 \
    --out "$RUN/generated512.jsonl" --report "$RUN/generation_evaluation.json" \
    2>&1 | tee "$RUN/generation.log"
}

benchmark() {
  "${ACTOR[@]}" generator-benchmark \
    --checkpoint "$RUN/final" --dataset "$DATASET" --goal-context "$GOAL_CONTEXT" \
    --attention-impl cudnn --base-semantic b3714ce680b84a74233b053c5bc32f48a37e3c257b46b4e131a0f784b4534286 \
    --seed 73 --temperature 0.9 --topk 8 --diversity-penalty 0.35 \
    --out "$RUN/generator_benchmark" --cuda-budget 64 --proposal-multiplier 4 --critic-batch 256 \
    --min-improve-prob 0.5 --max-regression-prob 0.35 --max-protected-regress-prob 0.35 \
    --operation-diversity-penalty 0.25 \
    2>&1 | tee "$RUN/generator_benchmark.log"
}

cuda_campaign() {
  compile_cuda
  "${ACTOR[@]}" cuda-campaign \
    --lanes "$RUN/generator_benchmark" \
    --base-parent "$RUN/generator_benchmark/base_parent.json" \
    --cases "$CASES" --cases-manifest "$CASES_MANIFEST" \
    --exe "$CUDA_EXE" --controller "$CHAMP/controller.bkgc" --routes "$CHAMP/routes.bkv6" \
    --legacy-routes "$CHAMP/legacy_routes.bkv6" --out "$RUN/cuda_campaign" --gpu 0 --min-seeds 6 \
    2>&1 | tee "$RUN/cuda_campaign.log"
}

package_results() {
  "${ACTOR[@]}" package-results --run-dir "$RUN" --package "$PACKAGES/bonsai_gcac_v2_results.tar.gz" \
    2>&1 | tee "$RUN/package.log"
}

resume() {
  "${ACTOR[@]}" train-actor \
    --dataset "$DATASET" --attention-impl cudnn --batch-size 1024 --seed 73 \
    --checkpoint "$RUN/final" --out "$RUN" --epochs 1 --early-epoch 2 --middle-epoch 6 \
    --eval-samples 1024 --examples-per-epoch 65536 --direction-loss-weight 0.5 \
    --lr 8e-4 --weight-decay 1e-4 --clip 2.0 --resume \
    2>&1 | tee "$RUN/resume.log"
}

case "$STAGE" in
  verify) verify ;;
  smoke) smoke ;;
  train) train ;;
  evaluate) evaluate ;;
  generate) generate ;;
  benchmark) benchmark ;;
  cuda) cuda_campaign ;;
  package) package_results ;;
  resume) resume ;;
  *) echo "invalid stage: $STAGE" >&2; exit 2 ;;
esac

printf 'persistent_run=%s\npackages=%s\n' "$RUN" "$PACKAGES"
