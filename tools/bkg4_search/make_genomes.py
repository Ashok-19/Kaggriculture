from __future__ import annotations

import argparse
from pathlib import Path

from tools.bkg4_search.cuda_io import pack_genomes
from tools.bkg4_search.module_codec import ConditionGene, ModuleGenome, RuleGene

ROUTES = (22, 23, 29, 32, 34, 46, 63, 64, 70, 77, 79)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", type=int, default=72)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--route-oracle", action="store_true")
    ap.add_argument("--from-json", type=Path, action="append", default=[])
    args = ap.parse_args()
    genomes = [ModuleGenome(args.prefix, ())]
    if args.route_oracle:
        genomes.extend(
            ModuleGenome(
                args.prefix,
                (RuleGene((ConditionGene("STEP", "GE", 0, 0),), route),),
            )
            for route in ROUTES
        )
    if args.from_json:
        from tools.bkg4_search.module_codec import load_genome
        genomes.extend(load_genome(p) for p in args.from_json)
    pack_genomes(genomes, args.out)
    print(f"genomes={len(genomes)} out={args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
