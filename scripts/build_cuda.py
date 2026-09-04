#!/usr/bin/env python3
"""Build the active RLRMB CUDA evaluator and state-capture helper."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "native" / "cuda" / "src"
DEFAULT_OUTPUT_DIR = ROOT / "workspace" / "build" / "cuda"


def build_command(source: Path, output: Path, *, arch: str = "sm_86") -> list[str]:
    nvcc = shutil.which("nvcc") or "/usr/bin/nvcc"
    includes = (
        ROOT,
        ROOT / "native" / "cuda" / "include",
        ROOT / "native" / "bkg4" / "include",
        ROOT / "native" / "bkg4" / "cuda",
        ROOT / "native" / "bkg4" / "stateful",
    )
    return [
        nvcc, "-O3", "-std=c++17", f"-arch={arch}", "-lineinfo",
        *(f"-I{path}" for path in includes), "-o", str(output), str(source),
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--arch", default="sm_86")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name in ("rlrmb_eval", "bkg4_information_state_capture"):
        output = args.output_dir / name
        command = build_command(SOURCE_DIR / f"{name}.cu", output, arch=args.arch)
        print(" ".join(command))
        subprocess.run(command, cwd=ROOT, check=True)
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
