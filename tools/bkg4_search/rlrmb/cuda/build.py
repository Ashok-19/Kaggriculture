#!/usr/bin/env python3
"""Build only the RLRMB Phase-4 CUDA evaluator."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[4]
SOURCE = Path(__file__).with_name("rlrmb_eval.cu")
DEFAULT_OUTPUT = ROOT / "scratch" / "cuda-engine-build" / "rlrmb_eval"


def build_command(output: Path, *, arch: str = "sm_86") -> list[str]:
    nvcc = shutil.which("nvcc") or "/usr/bin/nvcc"
    return [nvcc, "-O3", "-std=c++17", f"-arch={arch}", "-lineinfo", f"-I{ROOT / 'cuda-engine' / 'include'}", "-o", str(output), str(SOURCE)]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--arch", default="sm_86")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    command = build_command(args.output, arch=args.arch)
    print(" ".join(command))
    subprocess.run(command, cwd=ROOT, check=True)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
