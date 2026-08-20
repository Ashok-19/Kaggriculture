#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$ROOT"

# Source snapshots are created from HEAD. Refuse to silently omit uncommitted tracked work.
git diff --quiet
git diff --cached --quiet

BUNDLE_ROOT="$ROOT/scratch/editformer_gcac_v2_bundle"
PAYLOAD="$BUNDLE_ROOT/kaggriculture"
ARCHIVE="$ROOT/scratch/bonsai_editformer_gcac_v2_bundle.tgz"
rm -rf "$BUNDLE_ROOT" "$ARCHIVE"
mkdir -p "$PAYLOAD"

HEAD_SHA="$(git rev-parse HEAD)"
BRANCH="$(git branch --show-current)"
PUREJAXRL_TREE="$(git rev-parse HEAD:purejaxrl)"

# Exact tracked source. The neural model remains offline-only; BKG4/CUDA assets are
# included for deterministic proposal compilation/evaluation after training.
git archive HEAD \
  tools/bkg4_search \
  docs/architecture \
  bonsai-kg/python \
  bonsai-kg/include \
  bonsai-kg/cuda \
  bonsai-kg/integrations/bkg4_champion \
  bonsai-kg/integrations/stateful_v6 \
  cuda-engine/include \
  cuda-engine/Makefile \
  | tar -x -C "$PAYLOAD"

cat > "$PAYLOAD/.source_provenance.env" <<EOF
export BONSAI_SOURCE_HEAD=$HEAD_SHA
export BONSAI_SOURCE_BRANCH=$BRANCH
export BONSAI_PUREJAXRL_TREE_SHA=$PUREJAXRL_TREE
EOF

# Fixed Stage-B2 supervision. Cloud training must consume this exact file, never relabel
# the historical corpus independently.
DATASET="scratch/bkg4_editformer_jax/actor/gcac_v2_transitions_full.npz"
DATASET_MANIFEST="${DATASET%.npz}.manifest.json"
mkdir -p "$PAYLOAD/scratch/bkg4_editformer_jax/actor"
cp "$DATASET" "$DATASET_MANIFEST" "$PAYLOAD/scratch/bkg4_editformer_jax/actor/"

# Frozen feature caches needed by Stage-B2 parent/child batches and B371 lookup.
mkdir -p "$PAYLOAD/scratch/bkg4_editformer_jax/features"
cp scratch/bkg4_editformer_jax/features/historical_full.npz \
   scratch/bkg4_editformer_jax/features/current_full.npz \
   scratch/bkg4_editformer_jax/features/manifest_full.json \
   "$PAYLOAD/scratch/bkg4_editformer_jax/features/"

# Proven BonsaiEditFormer-5M critic checkpoint.
mkdir -p "$PAYLOAD/bonsai_editformer5m_results"
cp -a bonsai_editformer5m_results/final "$PAYLOAD/bonsai_editformer5m_results/"

# Frozen current400 identity/truth used only to recover the B371 ModuleGenome exactly.
mkdir -p "$PAYLOAD/scratch/bkg4_rl_synthesis_aug20" "$PAYLOAD/scratch/bkg4_search/current_shifted_aug20"
cp scratch/bkg4_rl_synthesis_aug20/single_predicates.b4gm \
   scratch/bkg4_rl_synthesis_aug20/single_predicates_current140.b4rs \
   scratch/bkg4_rl_synthesis_aug20/single_predicates_meta.jsonl \
   scratch/bkg4_rl_synthesis_aug20/bkg4_current_module_eval.cu \
   "$PAYLOAD/scratch/bkg4_rl_synthesis_aug20/"
cp scratch/bkg4_search/current_shifted_aug20/cases.tsv "$PAYLOAD/scratch/bkg4_search/current_shifted_aug20/"

# Exact fresh24 semantic goal context. The capture is frozen-345 public-state context;
# B371/candidate ModuleGenome activations are evaluated offline over these legal states.
mkdir -p "$PAYLOAD/scratch/bkg4_editformer_jax/gcac_v2"
cp scratch/bkg4_editformer_jax/gcac_v2/fresh24_goal_context.json \
   scratch/bkg4_editformer_jax/gcac_v2/fresh24_r0_264.b4is \
   scratch/bkg4_editformer_jax/gcac_v2/fresh24_cases7.tsv \
   scratch/bkg4_editformer_jax/gcac_v2/probes264_287.txt \
   "$PAYLOAD/scratch/bkg4_editformer_jax/gcac_v2/"

# Exact B371 fresh24 baseline and fair six-seed/both-seat reward case order.
mkdir -p "$PAYLOAD/run/cuda_campaign" "$PAYLOAD/scratch/bkg4_search/fresh_promotion24"
cp run/fresh24_cases.tsv run/fresh24_cases.manifest.json "$PAYLOAD/run/"
cp run/cuda_campaign/actor_critic.b4rs "$PAYLOAD/run/cuda_campaign/"
cp -a scratch/bkg4_search/fresh_promotion24/tapes "$PAYLOAD/scratch/bkg4_search/fresh_promotion24/"

# Validate payload identities before archiving.
DATASET_SHA="$(sha256sum "$DATASET" | awk '{print $1}')"
MANIFEST_DATASET_SHA="$(python3 -c 'import json;print(json.load(open("scratch/bkg4_editformer_jax/actor/gcac_v2_transitions_full.manifest.json"))["dataset_sha256"])')"
[[ "$DATASET_SHA" == "$MANIFEST_DATASET_SHA" ]]
GOAL_CAPTURE_SHA="$(sha256sum scratch/bkg4_editformer_jax/gcac_v2/fresh24_r0_264.b4is | awk '{print $1}')"
MANIFEST_GOAL_SHA="$(python3 -c 'import json;print(json.load(open("scratch/bkg4_editformer_jax/gcac_v2/fresh24_goal_context.json"))["capture_sha256"])')"
[[ "$GOAL_CAPTURE_SHA" == "$MANIFEST_GOAL_SHA" ]]

# Ensure the runner itself is executable in the extracted bundle.
chmod +x "$PAYLOAD/tools/bkg4_search/bonsai_editformer/scripts/gcac_v2_cloud_run.sh"

tar -C "$BUNDLE_ROOT" -czf "$ARCHIVE" kaggriculture
printf 'archive=%s\n' "$ARCHIVE"
printf 'sha256=%s\n' "$(sha256sum "$ARCHIVE" | awk '{print $1}')"
printf 'bytes=%s\n' "$(stat -c %s "$ARCHIVE")"
printf 'source_head=%s\n' "$HEAD_SHA"
printf 'dataset_sha256=%s\n' "$DATASET_SHA"
printf 'goal_capture_sha256=%s\n' "$GOAL_CAPTURE_SHA"
