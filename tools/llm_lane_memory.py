#!/usr/bin/env python3
"""Stream/query the BonsaiEvolve common report memory CSV."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import deque
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = ROOT / "workspace/llm_lane_evolution/memory/reports.csv"
FIELDS = [
    "report_id",
    "recorded_at",
    "round",
    "agent_id",
    "lane",
    "branch",
    "start_head",
    "report_commit",
    "candidate_id",
    "verdict",
    "retain",
    "valid_executable",
    "nontrivial",
    "build_improvement",
    "routable",
    "mechanism",
    "compact_summary",
    "key_lesson",
    "proposal_ref",
    "summary_ref",
    "metrics_ref",
    "tags",
]
BOOL_FIELDS = {"retain", "valid_executable", "nontrivial", "build_improvement", "routable"}
MAX_SUMMARY = 800
MAX_LESSON = 600


def iter_rows(path: Path) -> Iterable[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != FIELDS:
            raise ValueError(f"unexpected CSV header: {reader.fieldnames!r}")
        yield from reader


def as_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "y"}


def emit(rows: Iterable[dict[str, str]], fmt: str) -> None:
    if fmt == "jsonl":
        for row in rows:
            print(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
        return
    writer = csv.DictWriter(sys.stdout, fieldnames=FIELDS, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow(row)


def cmd_validate(path: Path) -> int:
    seen: set[str] = set()
    count = 0
    for line_no, row in enumerate(iter_rows(path), start=2):
        count += 1
        report_id = row["report_id"].strip()
        if not report_id:
            raise ValueError(f"line {line_no}: empty report_id")
        if report_id in seen:
            raise ValueError(f"line {line_no}: duplicate report_id {report_id}")
        seen.add(report_id)
        for field in BOOL_FIELDS:
            if row[field].strip().lower() not in {"true", "false"}:
                raise ValueError(f"line {line_no}: {field} must be true/false")
        if len(row["compact_summary"]) > MAX_SUMMARY:
            raise ValueError(f"line {line_no}: compact_summary exceeds {MAX_SUMMARY} chars")
        if len(row["key_lesson"]) > MAX_LESSON:
            raise ValueError(f"line {line_no}: key_lesson exceeds {MAX_LESSON} chars")
    print(json.dumps({"ok": True, "rows": count, "path": str(path.relative_to(ROOT))}))
    return 0


def cmd_tail(path: Path, limit: int, fmt: str) -> int:
    rows = deque(iter_rows(path), maxlen=limit)
    emit(rows, fmt)
    return 0


def cmd_show(path: Path, report_id: str, fmt: str) -> int:
    for row in iter_rows(path):
        if row["report_id"] == report_id:
            emit([row], fmt)
            return 0
    print(f"report not found: {report_id}", file=sys.stderr)
    return 1


def cmd_search(path: Path, args: argparse.Namespace) -> int:
    needle = (args.text or "").casefold()
    results: list[dict[str, str]] = []
    for row in iter_rows(path):
        if args.lane and row["lane"] != args.lane:
            continue
        if args.agent and row["agent_id"] != args.agent:
            continue
        if args.verdict and row["verdict"] != args.verdict:
            continue
        if args.retain and as_bool(row["retain"]) != (args.retain == "yes"):
            continue
        if args.build_improvement and as_bool(row["build_improvement"]) != (args.build_improvement == "yes"):
            continue
        if needle:
            haystack = " ".join(
                row[field]
                for field in ("lane", "candidate_id", "mechanism", "compact_summary", "key_lesson", "tags")
            ).casefold()
            if needle not in haystack:
                continue
        results.append(row)
        if len(results) >= args.limit:
            break
    emit(results, args.format)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="memory CSV path")
    sub = parser.add_subparsers(dest="command", required=True)

    validate = sub.add_parser("validate", help="validate schema and compactness")
    validate.set_defaults(func=lambda path, _args: cmd_validate(path))

    tail = sub.add_parser("tail", help="show newest rows without loading the whole CSV")
    tail.add_argument("--limit", type=int, default=20)
    tail.add_argument("--format", choices=("jsonl", "csv"), default="jsonl")
    tail.set_defaults(func=lambda path, args: cmd_tail(path, args.limit, args.format))

    show = sub.add_parser("show", help="show one report by report_id")
    show.add_argument("report_id")
    show.add_argument("--format", choices=("jsonl", "csv"), default="jsonl")
    show.set_defaults(func=lambda path, args: cmd_show(path, args.report_id, args.format))

    search = sub.add_parser("search", help="stream-filter compact report rows")
    search.add_argument("--lane")
    search.add_argument("--agent")
    search.add_argument("--verdict")
    search.add_argument("--retain", choices=("yes", "no"))
    search.add_argument("--build-improvement", choices=("yes", "no"))
    search.add_argument("--text")
    search.add_argument("--limit", type=int, default=20)
    search.add_argument("--format", choices=("jsonl", "csv"), default="jsonl")
    search.set_defaults(func=cmd_search)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    path = args.csv
    if not path.is_absolute():
        path = (ROOT / path).resolve()
    if not path.exists():
        parser.error(f"CSV does not exist: {path}")
    if getattr(args, "limit", 1) <= 0:
        parser.error("--limit must be positive")
    try:
        return args.func(path, args)
    except (OSError, ValueError, csv.Error) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
