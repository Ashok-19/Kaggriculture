from __future__ import annotations

import argparse
import json
from pathlib import Path

from tools.bkg4_search.mechanics_codec import MechanicGenome, grid, neighborhood, save_genomes

ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--grid", action="store_true")
    ap.add_argument("--from-json", type=Path)
    ap.add_argument("--neighbors", action="store_true")
    args = ap.parse_args()
    out = (ROOT / args.out).resolve() if not args.out.is_absolute() else args.out
    if args.grid:
        gs = grid()
    elif args.from_json:
        src = json.loads(((ROOT / args.from_json).resolve() if not args.from_json.is_absolute() else args.from_json).read_text())
        x = src.get("genome", src)
        g = MechanicGenome(**x)
        gs = [MechanicGenome(), g]
        if args.neighbors:
            gs.extend(neighborhood(g))
    else:
        gs = [MechanicGenome()]
    # Stable de-duplication by semantic id; baseline remains first.
    seen = set(); ordered = []
    for g in gs:
        k = g.semantic_id()
        if k not in seen:
            seen.add(k); ordered.append(g)
    save_genomes(ordered, out)
    manifest = out.with_suffix(out.suffix + ".json")
    manifest.write_text(json.dumps({"schema": 1, "count": len(ordered), "genomes": [{"id": g.semantic_id(), "genome": g.canonical()} for g in ordered]}, sort_keys=True, indent=2) + "\n")
    print(f"genomes={len(ordered)} out={out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
