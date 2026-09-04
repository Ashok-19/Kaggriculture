"""Authoritative config-driven RLRMB production entry point."""
from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path
from typing import Sequence

from kaggriculture_bonsai.search.pipeline import EXIT_INFRASTRUCTURE, run_production_pipeline


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run bounded autonomous RLRMB synthesis")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--validate-config", action="store_true",
                        help="validate inputs and write a zero-game audit run")
    parser.add_argument("--smoke", action="store_true",
                        help="run the validated tiny bounded production fixture")
    parser.add_argument("--resume", action="store_true",
                        help="fail closed; deterministic resume is not yet supported")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.validate_config and args.smoke:
        parser.error("--validate-config and --smoke are mutually exclusive")
    command = (sys.executable, "-m", "kaggriculture_bonsai.cli",
               "--config", str(args.config), "--output", str(args.output))
    if args.validate_config:
        command += ("--validate-config",)
    if args.smoke:
        command += ("--smoke",)
    if args.resume:
        command += ("--resume",)
    print(f"RLRMB production output: {args.output}")
    print("RLRMB stage: frontier -> checkpoints -> beam/QD -> repair -> routability -> CEGIS -> materialization -> parity")
    code = run_production_pipeline(args.config, args.output, validate_only=args.validate_config,
                                   smoke=args.smoke, resume=args.resume)
    try:
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "command.txt").write_text(" ".join(shlex.quote(part) for part in command) + "\n", encoding="utf-8")
        (args.output / "exit_code.txt").write_text(f"{code}\n", encoding="utf-8")
    except OSError as exc:
        print(f"RLRMB infrastructure failure while writing audit artifacts: {exc}", file=sys.stderr)
        return EXIT_INFRASTRUCTURE
    print(f"RLRMB production exit code: {code}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
