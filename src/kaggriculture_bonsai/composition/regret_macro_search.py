from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from kaggriculture_bonsai.composition.cuda_io import load_fragment_results


def load_grid(path: str | Path) -> dict[int, dict]:
    rows: dict[int, dict] = {}
    with Path(path).open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            cid = int(row["id"])
            if cid in rows:
                raise ValueError(f"duplicate candidate id {cid}")
            rows[cid] = {
                "id": cid,
                "donor": int(row["donor"]),
                "start": int(row["start"]),
                "duration": int(row["duration"]),
                "path": row["path"],
            }
    if not rows:
        raise ValueError("empty fragment grid")
    return rows


def load_case_records(path: str | Path) -> list[dict]:
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if p.suffix.lower() == ".json":
        data = json.loads(text)
        if not isinstance(data, list):
            raise ValueError("case JSON must be a list")
        return [dict(x) for x in data]
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def sanitize_native_tape(src: str | Path, dst: str | Path) -> None:
    data = json.loads(Path(src).read_text(encoding="utf-8"))
    actions = data.get("actions") if isinstance(data, dict) else data
    if not isinstance(actions, list) or not actions:
        raise ValueError("native tape must contain a non-empty actions list")
    out = Path(dst)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"actions": actions}, separators=(",", ":")) + "\n", encoding="utf-8")


def _margin(row: tuple[int, int, int, int]) -> int:
    return int(row[0]) - int(row[1])


def summarize_fragment_run(results: str | Path, grid: str | Path, cases: str | Path) -> dict:
    meta, rows = load_fragment_results(results)
    grid_rows = load_grid(grid)
    case_rows = load_case_records(cases)
    if len(case_rows) != meta["cases"]:
        raise ValueError(f"case count mismatch {len(case_rows)} != {meta['cases']}")
    if not (meta["start_index"] <= 0 < meta["start_index"] + meta["candidates"]):
        raise ValueError("fragment result chunk does not include candidate 0 baseline")
    base_local = -meta["start_index"]
    baseline = rows[base_local]
    if any(r[3] for r in baseline):
        raise ValueError("baseline contains CUDA/runtime faults")
    baseline_margins = [_margin(r) for r in baseline]

    candidate_stats = []
    for li, candidate_rows in enumerate(rows):
        cid = meta["start_index"] + li
        if cid == 0:
            continue
        if cid not in grid_rows:
            raise ValueError(f"candidate {cid} missing from grid")
        if any(r[3] for r in candidate_rows):
            raise ValueError(f"candidate {cid} contains CUDA/runtime faults")
        margins = [_margin(r) for r in candidate_rows]
        deltas = [m - b for m, b in zip(margins, baseline_margins)]
        candidate_stats.append({
            **grid_rows[cid],
            "positive": sum(d > 0 for d in deltas),
            "neutral": sum(d == 0 for d in deltas),
            "harmful": sum(d < 0 for d in deltas),
            "loss_to_win": sum(b < 0 < m for b, m in zip(baseline_margins, margins)),
            "win_to_loss": sum(b > 0 > m for b, m in zip(baseline_margins, margins)),
            "sum_delta": sum(deltas),
            "min_delta": min(deltas),
            "max_delta": max(deltas),
            "margins": margins,
            "deltas": deltas,
            "changes": [int(r[2]) for r in candidate_rows],
        })

    best_per_case = []
    for ci, (case, base) in enumerate(zip(case_rows, baseline_margins)):
        best_score = (0, 0, base)
        best = None
        for cand in candidate_stats:
            margin = cand["margins"][ci]
            delta = cand["deltas"][ci]
            score = (int(base < 0 < margin), delta, margin)
            if score > best_score:
                best_score = score
                best = (cand, margin, delta)
        row = {
            "case": ci,
            "baseline_margin": base,
            "best_candidate_id": 0,
            "best_margin": base,
            "best_delta": 0,
            "loss_to_win": False,
            "candidate": {"donor": -1, "start": 0, "duration": 0, "path": "BASE"},
            **case,
        }
        if best is not None:
            cand, margin, delta = best
            row.update({
                "best_candidate_id": cand["id"],
                "best_margin": margin,
                "best_delta": delta,
                "loss_to_win": base < 0 < margin,
                "candidate": {k: cand[k] for k in ("donor", "start", "duration", "path")},
            })
        best_per_case.append(row)

    candidate_stats.sort(key=lambda x: (x["loss_to_win"], x["positive"], x["sum_delta"]), reverse=True)
    return {
        "schema": 1,
        "result": str(results),
        "grid": str(grid),
        "cases_source": str(cases),
        "meta": meta,
        "baseline_margins": baseline_margins,
        "candidate_stats": candidate_stats,
        "best_per_case": best_per_case,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Small host-side helper for regret-guided BKG4 CUDA fragment search.")
    sub = ap.add_subparsers(dest="command", required=True)
    s = sub.add_parser("summarize", help="summarize a B4FR1 result chunk against candidate-0 baseline")
    s.add_argument("--results", required=True, type=Path)
    s.add_argument("--grid", required=True, type=Path)
    s.add_argument("--cases", required=True, type=Path)
    s.add_argument("--out", type=Path)
    t = sub.add_parser("sanitize-tape", help="strip metadata so the native CUDA tape parser sees only actions")
    t.add_argument("input", type=Path)
    t.add_argument("output", type=Path)
    args = ap.parse_args()
    if args.command == "sanitize-tape":
        sanitize_native_tape(args.input, args.output)
        return 0
    payload = summarize_fragment_run(args.results, args.grid, args.cases)
    text = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
