#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$ROOT"

BUNDLE_ROOT="$ROOT/scratch/editformer_scaling_bundle"
PAYLOAD="$BUNDLE_ROOT/kaggriculture"
ARCHIVE="$ROOT/scratch/bonsai_editformer_scaling_bundle.tgz"
rm -rf "$BUNDLE_ROOT" "$ARCHIVE"
mkdir -p "$PAYLOAD"

HEAD_SHA="$(git rev-parse HEAD)"
BRANCH="$(git branch --show-current)"
PUREJAXRL_TREE="$(git rev-parse HEAD:purejaxrl)"

# Source snapshot: only files required by the ranker plus the frozen parent metadata.
git archive HEAD \
  tools/bkg4_search/bonsai_editformer \
  tools/bkg4_search/information_state_io.py \
  bonsai-kg/python \
  bonsai-kg/integrations/bkg4_champion/assets/identity.json \
  bonsai-kg/integrations/bkg4_champion/champions/w33_s172_expansion_v1/identity.json \
  | tar -x -C "$PAYLOAD"

cat > "$PAYLOAD/.source_provenance.env" <<EOF
export BONSAI_SOURCE_HEAD=$HEAD_SHA
export BONSAI_SOURCE_BRANCH=$BRANCH
export BONSAI_PUREJAXRL_TREE_SHA=$PUREJAXRL_TREE
EOF

# Frozen full feature tensors are representation-identical across 5M/20M/50M because token caps/schema stay fixed.
mkdir -p "$PAYLOAD/scratch/bkg4_editformer_jax/features"
cp scratch/bkg4_editformer_jax/features/historical_full.npz \
   scratch/bkg4_editformer_jax/features/current_full.npz \
   scratch/bkg4_editformer_jax/features/manifest_full.json \
   "$PAYLOAD/scratch/bkg4_editformer_jax/features/"
cp scratch/bkg4_editformer_jax/historical40_r0.b4is \
   scratch/bkg4_editformer_jax/current140_r0_264.b4is \
   "$PAYLOAD/scratch/bkg4_editformer_jax/"

# Frozen raw corpora remain present because the training/evaluation guards reconstruct row identities/splits.
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

# Preserve the absolute BASE_0 tape paths referenced by cuda_combined40.tsv.
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
