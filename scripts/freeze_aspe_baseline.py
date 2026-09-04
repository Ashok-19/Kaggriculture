"""Create or verify the tracked ASPE Phase-0 baseline context."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from kaggriculture_bonsai.league.aspe_baseline import (
    build_manifest,
    freeze_aspe_baseline,
    validate_manifest,
)
from kaggriculture_bonsai.paths import PROJECT_ROOT


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("workspace/aspe/baseline"))
    parser.add_argument("--source-head")
    parser.add_argument("--replace", action="store_true")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    output = PROJECT_ROOT / args.output if not args.output.is_absolute() else args.output
    if args.verify:
        path = output / "manifest.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        validate_manifest(manifest)
        regenerated = build_manifest(output, source_head=manifest["source_head"])
        if regenerated != manifest:
            raise RuntimeError("ASPE baseline manifest does not regenerate identically")
        print(json.dumps({"context_identity": manifest["context_identity"], "verified": True}))
        return 0
    manifest = freeze_aspe_baseline(
        args.output, source_head=args.source_head, replace=args.replace,
    )
    print(json.dumps({"context_identity": manifest["context_identity"], "generated": True}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

