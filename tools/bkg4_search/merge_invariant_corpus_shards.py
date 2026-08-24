#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from tools.bkg4_search.kaggle_invariant_corpus import merge_shards


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Globally deduplicate B4IM shards and create a structural grouped split")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validation-fraction", type=float, default=0.20)
    args = parser.parse_args(argv)
    print(json.dumps(merge_shards(args.output, validation_fraction=args.validation_fraction), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
