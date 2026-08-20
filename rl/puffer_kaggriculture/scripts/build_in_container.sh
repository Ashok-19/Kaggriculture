#!/bin/bash
set -euo pipefail
. /puffertank/venv/bin/activate

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PUFFER=/puffertank/pufferlib
EXPECTED_PUFFER_SHA='d21a161a4009''4ec7ba7578e9''cf8779b73f6c218c'

actual_sha="$(git -C "$PUFFER" rev-parse HEAD)"
if [[ "$actual_sha" != "$EXPECTED_PUFFER_SHA" ]]; then
    echo "Refusing to build against unexpected PufferLib SHA: $actual_sha" >&2
    exit 2
fi

ENV_DST="$PUFFER/ocean/kaggriculture"
rm -rf "$ENV_DST"
mkdir -p "$ENV_DST"
cp "$ROOT/rl/puffer_kaggriculture/ocean/kaggriculture/kaggriculture.h" "$ENV_DST/"
cp "$ROOT/rl/puffer_kaggriculture/ocean/kaggriculture/binding.c" "$ENV_DST/"
cp "$ROOT/rl/puffer_kaggriculture/config/kaggriculture.ini" "$PUFFER/config/kaggriculture.ini"
cp "$ROOT/cpp-engine/include/sim.hpp" "$ENV_DST/sim.hpp"
cp "$ROOT/cpp-engine/include/pyrandom.hpp" "$ENV_DST/pyrandom.hpp"

python "$ROOT/rl/puffer_kaggriculture/scripts/patch_puffer_backend.py" "$PUFFER"

cd "$PUFFER"
CC='clang++ -x c++' bash build.sh kaggriculture

echo "PUFFERLIB_SHA=$actual_sha"
echo "SIM_SHA=$(sha256sum "$ROOT/cpp-engine/include/sim.hpp" | awk '{print $1}')"
echo "PYRANDOM_SHA=$(sha256sum "$ROOT/cpp-engine/include/pyrandom.hpp" | awk '{print $1}')"
python - <<'PY'
import pufferlib._C as C
print('COMPILED_ENV', C.env_name)
print('PRECISION_BYTES', C.precision_bytes)
PY
