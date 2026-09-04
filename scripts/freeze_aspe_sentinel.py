#!/usr/bin/env python3
"""Regenerate the tracked ASPE Phase-2 BUILD sentinel contract."""
from __future__ import annotations

import argparse
from pathlib import Path

from kaggriculture_bonsai.search.aspe import select_build_sentinel, write_sentinel_evidence


parser = argparse.ArgumentParser()
parser.add_argument("--baseline", type=Path, default=Path("workspace/aspe/baseline/manifest.json"))
parser.add_argument("--output", type=Path, default=Path("workspace/aspe/sentinel"))
args = parser.parse_args()

panel = select_build_sentinel(args.baseline)
write_sentinel_evidence(panel, args.output)
print(f"sentinel={panel.identity} cases={len(panel.cases)} output={args.output}")
