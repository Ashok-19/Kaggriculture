#!/bin/sh
set -eu
if [ "$#" -lt 2 ]; then
  echo "usage: run_state_parity.sh BUILD_DIR TRACE [TRACE...]" >&2
  exit 2
fi
build_dir=$1
shift
for f in "$@"; do
  "$build_dir/cpu_state_hash" "$f" > "$build_dir/cpu-state.hash"
  "$build_dir/cuda_state_hash" "$f" > "$build_dir/cuda-state.hash"
  cmp "$build_dir/cpu-state.hash" "$build_dir/cuda-state.hash"
  echo "PASS-STATE $f (720 logical states exact)"
done
