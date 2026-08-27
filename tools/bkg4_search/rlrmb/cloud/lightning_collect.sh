#!/usr/bin/env bash
set -euo pipefail
RUN="${1:?usage: lightning_collect.sh RUN_DIR OUTPUT_TAR_GZ}"
DEST="${2:?usage: lightning_collect.sh RUN_DIR OUTPUT_TAR_GZ}"
FILES=(generation_state.json cloud_result.json command.txt exit_code.txt stdout.log stderr.log)
[[ -f "$RUN/telemetry.jsonl" ]] && FILES+=(telemetry.jsonl)
[[ -d "$RUN/generations" ]] && FILES+=(generations)
[[ -d "$RUN/completed_attempts" ]] && FILES+=(completed_attempts)
tar -czf "$DEST" -C "$RUN" "${FILES[@]}"
echo "$DEST"
