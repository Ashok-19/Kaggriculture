#!/usr/bin/env python3
"""Safely import a Kaggle dual-T4 result archive into ignored local scratch."""
from __future__ import annotations
import argparse, json, tarfile
from datetime import datetime, timezone
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("archive", type=Path)
    ap.add_argument("--run-id")
    ap.add_argument("--output-root", type=Path, default=Path("scratch/bonsai_meta_search_runs"))
    args = ap.parse_args()
    repo = Path(__file__).resolve().parents[2]
    archive = args.archive.resolve()
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-dual-t4")
    root = args.output_root if args.output_root.is_absolute() else repo / args.output_root
    dest = root / run_id
    if dest.exists(): raise RuntimeError(f"destination exists: {dest}")
    dest.mkdir(parents=True)
    with tarfile.open(archive, "r:gz") as tf:
        base = dest.resolve()
        for member in tf.getmembers():
            if member.issym() or member.islnk() or member.isdev():
                raise RuntimeError(f"unsafe archive member type: {member.name}")
            target = (dest / member.name).resolve()
            if target != base and base not in target.parents:
                raise RuntimeError(f"unsafe archive path: {member.name}")
        tf.extractall(dest)
    summary = dest / "meta_search/merged/summary.json"
    tops = dest / "meta_search/merged/top_candidates.jsonl"
    if not summary.exists() or not tops.exists():
        raise RuntimeError("archive is missing merged summary/top-candidate artifacts")
    info = {"imported_to": str(dest), "summary": json.loads(summary.read_text()),
            "next": "Reproduce selected candidates in exact CUDA/C++ and distill to BKG3 before exposing blind validation."}
    print(json.dumps(info, indent=2, ensure_ascii=False))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
