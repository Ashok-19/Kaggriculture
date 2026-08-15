"""Generate and validate 15 full episodes against kaggle-environments 1.32.7."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from export_trace import export

PAIRS = [
    ("starter", "random"),
    ("random", "starter"),
    ("starter", "starter"),
    ("random", "random"),
    ("pass", "starter"),
]
SEEDS = [11, 22, 33]


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: run_full_parity.py VALIDATE OUT_DIR")
    validate = Path(sys.argv[1]).resolve()
    out_dir = Path(sys.argv[2]).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    traces = []
    for pair in PAIRS:
        for seed in SEEDS:
            traces.append(export(seed, pair, out_dir))
    subprocess.run([str(validate), *map(str, traces)], check=True)
    print(f"PASS full-environment parity: {len(traces)}/{len(traces)} traces")


if __name__ == "__main__":
    main()
