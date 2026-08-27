#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$ROOT"
export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null || [[ "$(uv --version)" != "uv 0.11.3"* ]]; then
  python3 -m pip install --user --disable-pip-version-check "uv==0.11.3"
fi
CAP="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n1 | tr -d '.')"
ARCH="sm_${CAP}"
mkdir -p scratch/cuda-engine-build
python3 tools/bkg4_search/rlrmb/cuda/build.py --arch "$ARCH" --output scratch/cuda-engine-build/rlrmb_eval
make -C cuda-engine ARCH="$ARCH" BUILD_DIR=../scratch/cuda-engine-build bkg4-information-state-capture
make -C cuda-engine ARCH="$ARCH" BUILD_DIR=../scratch/cuda-engine-build bkg4-fragment-eval
python3 - <<'PY'
import json
from pathlib import Path
from tools.bkg4_search.rlrmb.cloud_runner import generation_config, initial_state, load_cloud_config
source=Path('tools/bkg4_search/rlrmb/configs/phase16_lightning_v1.json')
config=load_cloud_config(source)
Path('scratch/phase16_runtime_validation.json').write_text(
    json.dumps(generation_config(config,initial_state(config),'pilot'),sort_keys=True,indent=2)+'\n')
PY
python3 -m tools.bkg4_search.rlrmb.cli \
  --config scratch/phase16_runtime_validation.json \
  --output scratch/bootstrap_config_validation --validate-config
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python3 -m unittest \
  tools.bkg4_search.rlrmb.tests.test_evaluator_parity \
  tools.bkg4_search.rlrmb.tests.test_materialize \
  tools.bkg4_search.rlrmb.tests.test_cloud
set +e
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python3 -m tools.bkg4_search.rlrmb.cli \
  --config scratch/phase16_runtime_validation.json \
  --output scratch/bootstrap_smoke --smoke
SMOKE_STATUS=$?
set -e
if [[ "$SMOKE_STATUS" -ne 0 && "$SMOKE_STATUS" -ne 1 ]]; then exit "$SMOKE_STATUS"; fi
echo "RLRMB Lightning bootstrap complete: $ARCH"
