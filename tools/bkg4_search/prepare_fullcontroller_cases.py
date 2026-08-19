from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def from_recorded(path: Path) -> list[dict]:
    rows = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    return [
        {
            "tape": str(r["tape"]), "team": str(r["team"]), "seat": int(r["seat"]),
            "seed": int(r["seed"]), "protected": int(r.get("protected", 1)),
            "own_history": int(r.get("own_history", 0)),
        }
        for r in rows
    ]


def from_dev_manifest(path: Path, both_seats: bool) -> list[dict]:
    src = list(csv.DictReader(path.open(encoding="utf-8"), delimiter="\t"))
    rows = []
    for r in src:
        tape = ROOT / "scratch" / "kaggriculture_meta_search_bundle" / r["bundle_path"]
        seats = (0, 1) if both_seats else (1 - int(r["source_seat"]),)
        for seat in seats:
            rows.append({
                "tape": str(tape.resolve()), "team": r["team"], "seat": seat,
                "seed": int(r["source_seed"]), "protected": 1, "own_history": 0,
            })
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--recorded", type=Path)
    group.add_argument("--dev-manifest", type=Path)
    ap.add_argument("--both-seats", action="store_true")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.recorded:
        src = (ROOT / args.recorded).resolve() if not args.recorded.is_absolute() else args.recorded
        rows = from_recorded(src)
    else:
        src = (ROOT / args.dev_manifest).resolve() if not args.dev_manifest.is_absolute() else args.dev_manifest
        rows = from_dev_manifest(src, args.both_seats)
    out = (ROOT / args.out).resolve() if not args.out.is_absolute() else args.out
    fams = {name: i for i, name in enumerate(sorted({r["team"] for r in rows}))}
    lines = []
    for r in rows:
        lines.append("\t".join((
            str(r["tape"]), str(int(r["seat"])), str(int(r["seed"])), str(fams[r["team"]]),
            str(int(r.get("protected", 1))), str(int(r.get("own_history", 0))),
        )))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    case_json = out.with_suffix(out.suffix + ".cases.jsonl")
    case_json.write_text("".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows), encoding="utf-8")
    meta = {
        "schema": 1, "cases": len(rows), "families": fams, "source": str(src),
        "both_seats": bool(args.both_seats), "unique_seeds": len({r["seed"] for r in rows}),
    }
    out.with_suffix(out.suffix + ".json").write_text(json.dumps(meta, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
    print(json.dumps(meta, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
