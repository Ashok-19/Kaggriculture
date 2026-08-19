from __future__ import annotations

import argparse
import json
from pathlib import Path

from tools.bkg4_search.build_league import replay_tapes

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EPISODES = (94460852, 94459964, 94452931, 94545578, 94541153, 94459078)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--replays", type=Path, default=Path("scratch/bkg4_search/fresh_holdout_aug19"))
    ap.add_argument("--episode", type=int, action="append", default=[])
    ap.add_argument("--out", type=Path, default=Path("scratch/bkg4_search/fresh_promotion24"))
    args = ap.parse_args()
    replay_dir = (ROOT / args.replays).resolve() if not args.replays.is_absolute() else args.replays
    out = (ROOT / args.out).resolve() if not args.out.is_absolute() else args.out
    tape_dir = out / "tapes"
    tape_dir.mkdir(parents=True, exist_ok=True)
    episodes = tuple(args.episode) if args.episode else DEFAULT_EPISODES

    cases = []
    summary = []
    for eid in episodes:
        replay = replay_dir / f"episode-{eid}-replay.json"
        if not replay.exists():
            raise FileNotFoundError(replay)
        obj = json.loads(replay.read_text(encoding="utf-8"))
        shops144 = list(((obj["steps"][144][0].get("observation") or {}).get("town") or {}).get("unlocked_shops") or [])
        rows = replay_tapes(replay, tape_dir)
        for row in rows:
            for candidate_seat in (0, 1):
                cases.append({
                    "tape": row["path"],
                    "team": row["team"],
                    "seed": int(row["seed"]),
                    "seat": candidate_seat,
                    "protected": 1,
                    "own_history": int(row["team"] == "ReCurSiON"),
                    "source": "fresh_promotion_holdout",
                    "source_episode": eid,
                    "opponent_source_seat": int(row["seat"]),
                    "shops144": shops144,
                    "pet_seen_144": "PET_CAFE" in shops144,
                })
        summary.append({"episode": eid, "seed": int(rows[0]["seed"]), "teams": [r["team"] for r in rows], "shops144": shops144})

    cases_path = out / "cases.jsonl"
    cases_path.write_text("".join(json.dumps(x, ensure_ascii=False, sort_keys=True) + "\n" for x in cases), encoding="utf-8")
    report = {
        "schema": 1,
        "episodes": summary,
        "cases": len(cases),
        "unique_seeds": len({x["seed"] for x in cases}),
        "families": sorted({x["team"] for x in cases}),
        "pet_cases": sum(bool(x["pet_seen_144"]) for x in cases),
        "own_history_cases": sum(int(x["own_history"]) for x in cases),
    }
    (out / "manifest.json").write_text(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
