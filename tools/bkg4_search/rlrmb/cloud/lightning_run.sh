#!/usr/bin/env bash
set -euo pipefail
MODE="${1:?usage: lightning_run.sh pilot|campaign OUTPUT [--resume]}"
OUTPUT="${2:?usage: lightning_run.sh pilot|campaign OUTPUT [--resume]}"
RESUME="${3:-}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$ROOT"
mkdir -p "$OUTPUT"
CMD=(python3 -m tools.bkg4_search.rlrmb.cloud_runner --config tools/bkg4_search/rlrmb/configs/phase16_lightning_v1.json --output "$OUTPUT" --mode "$MODE")
if [[ "$RESUME" == "--resume" ]]; then CMD+=(--resume); fi
printf '%q ' "${CMD[@]}" > "$OUTPUT/command.txt"; printf '\n' >> "$OUTPUT/command.txt"
set +e
"${CMD[@]}" > >(tee "$OUTPUT/stdout.log") 2> >(tee "$OUTPUT/stderr.log" >&2)
STATUS=$?
set -e
printf '%s\n' "$STATUS" > "$OUTPUT/exit_code.txt"
exit "$STATUS"
