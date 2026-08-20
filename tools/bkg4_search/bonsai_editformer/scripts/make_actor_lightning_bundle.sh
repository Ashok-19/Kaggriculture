#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$ROOT"

BUNDLE_ROOT="$ROOT/scratch/editformer_actor_bundle"
PAYLOAD="$BUNDLE_ROOT/kaggriculture"
ARCHIVE="$ROOT/scratch/bonsai_editformer_actorcritic_bundle.tgz"
rm -rf "$BUNDLE_ROOT" "$ARCHIVE"
mkdir -p "$PAYLOAD"

HEAD_SHA="$(git rev-parse HEAD)"
BRANCH="$(git branch --show-current)"
PUREJAXRL_TREE="$(git rev-parse HEAD:purejaxrl)"

# Tracked source needed by Actor-Critic training, typed generation, BKG4 compilation,
# and the exact current-345 CUDA module evaluator.
git archive HEAD \
  tools/bkg4_search \
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

# Frozen representation caches used by actor supervision and critic scoring.
mkdir -p "$PAYLOAD/scratch/bkg4_editformer_jax/features"
cp scratch/bkg4_editformer_jax/features/historical_full.npz \
   scratch/bkg4_editformer_jax/features/current_full.npz \
   scratch/bkg4_editformer_jax/features/manifest_full.json \
   "$PAYLOAD/scratch/bkg4_editformer_jax/features/"
cp scratch/bkg4_editformer_jax/historical40_r0.b4is \
   scratch/bkg4_editformer_jax/current140_r0_264.b4is \
   "$PAYLOAD/scratch/bkg4_editformer_jax/"

# Proven BonsaiEditFormer-5M critic checkpoint.
mkdir -p "$PAYLOAD/bonsai_editformer5m_results"
cp -a bonsai_editformer5m_results/final "$PAYLOAD/bonsai_editformer5m_results/"

# Historical exact-CUDA mutation corpus and current-345 400-candidate truth.
mkdir -p "$PAYLOAD/scratch/bkg4_search" "$PAYLOAD/scratch/bkg4_rl_synthesis_aug20"
cp -a scratch/bkg4_search/campaign_broad_evo4096 \
      scratch/bkg4_search/campaign_guarded_evo4096 \
      scratch/bkg4_search/campaign_p144_broad_evo4096 \
      scratch/bkg4_search/campaign_evo2048_v2 \
      scratch/bkg4_search/campaign_greedy2048_v2 \
      "$PAYLOAD/scratch/bkg4_search/"
cp scratch/bkg4_search/cuda_combined40.tsv "$PAYLOAD/scratch/bkg4_search/"
mkdir -p "$PAYLOAD/scratch/bkg4_search/current_shifted_aug20"
cp scratch/bkg4_search/current_shifted_aug20/cases.tsv "$PAYLOAD/scratch/bkg4_search/current_shifted_aug20/"
cp scratch/bkg4_rl_synthesis_aug20/single_predicates.b4gm \
   scratch/bkg4_rl_synthesis_aug20/single_predicates_current140.b4rs \
   scratch/bkg4_rl_synthesis_aug20/single_predicates_meta.jsonl \
   "$PAYLOAD/scratch/bkg4_rl_synthesis_aug20/"

# Current-345 exact CUDA evaluator source. Recompile natively on Lightning; never copy
# the local GPU binary as the cloud truth executable.
cp scratch/bkg4_rl_synthesis_aug20/bkg4_current_module_eval.cu \
   "$PAYLOAD/scratch/bkg4_rl_synthesis_aug20/"

# Fresh anti-overfit reward league: 6 seeds x both seats = 24 cases.
mkdir -p "$PAYLOAD/scratch/bkg4_search/base0_fresh_promotion24_p144"
cp scratch/bkg4_search/base0_fresh_promotion24_p144/cases.jsonl \
   "$PAYLOAD/scratch/bkg4_search/base0_fresh_promotion24_p144/"
mkdir -p "$PAYLOAD/scratch/bkg4_search/fresh_promotion24"
cp -a scratch/bkg4_search/fresh_promotion24/tapes \
      "$PAYLOAD/scratch/bkg4_search/fresh_promotion24/"

# Preserve absolute BASE_0 tape paths used by the historical data guards.
while IFS=$'\t' read -r base_tape _; do
  [[ -n "$base_tape" ]] || continue
  rel="${base_tape#$ROOT/}"
  mkdir -p "$PAYLOAD/$(dirname "$rel")"
  cp "$base_tape" "$PAYLOAD/$rel"
done < scratch/bkg4_search/cuda_combined40.tsv

tar -C "$BUNDLE_ROOT" -czf "$ARCHIVE" kaggriculture
printf 'archive=%s\n' "$ARCHIVE"
printf 'sha256=%s\n' "$(sha256sum "$ARCHIVE" | awk '{print $1}')"
printf 'bytes=%s\n' "$(stat -c %s "$ARCHIVE")"
printf 'source_head=%s\n' "$HEAD_SHA"
