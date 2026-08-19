from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tape_meta(path: Path, fallback_team: str | None = None, fallback_seed: int | None = None) -> dict:
    obj = json.loads(path.read_text(encoding="utf-8"))
    actions = obj.get("actions") if isinstance(obj, dict) else None
    if not isinstance(actions, list) or not actions:
        raise ValueError(f"not a native tape: {path}")
    src = obj.get("source") or {}
    seed = src.get("seed", fallback_seed)
    if seed is None:
        raise ValueError(f"missing seed: {path}")
    return {
        "path": str(path.resolve()),
        "sha256": sha256(path),
        "team": str(src.get("team") or fallback_team or path.stem),
        "seed": int(seed),
        "episode_id": int(src.get("episode_id") or 0),
        "source_seat": int(src.get("seat")) if src.get("seat") is not None else None,
    }


def replay_to_tapes(replay_path: Path, out_dir: Path) -> list[dict]:
    replay = json.loads(replay_path.read_text(encoding="utf-8"))
    steps = replay.get("steps") or []
    info = replay.get("info") or {}
    teams = list(info.get("TeamNames") or ["seat0", "seat1"])
    seed = int(info.get("seed"))
    episode = int(info.get("EpisodeId") or 0)
    if len(steps) < 2 or any(len(x) != 2 for x in steps):
        raise ValueError(f"invalid replay: {replay_path}")
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for seat in (0, 1):
        actions = [
            dict(step[seat].get("action") or {"farmer": ["PASS"], "hands": [], "market": []})
            for step in steps[1:]
        ]
        obj = {
            "source": {
                "episode_id": episode,
                "seat": seat,
                "team": teams[seat],
                "opponent": teams[1 - seat],
                "seed": seed,
                "bank_role": "optimization_oracle_fresh_holdout",
            },
            "actions": actions,
        }
        name = teams[seat].replace("/", "_").replace(" ", "_")
        path = out_dir / f"{name}_s{seat}_ep{episode}.json"
        path.write_text(json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
        rows.append(tape_meta(path))
    return rows


def build_sha_index() -> dict[str, dict]:
    index: dict[str, dict] = {}
    tape_dirs = (
        ROOT / "scratch/bkg4_search/animal_sealed24/tapes",
        ROOT / "scratch/bkg4_search/fresh_promotion24/tapes",
        ROOT / "scratch/bkg4_search/league_aug19/fresh_tapes",
    )
    for d in tape_dirs:
        if not d.is_dir():
            continue
        for p in sorted(d.glob("*.json")):
            try:
                m = tape_meta(p)
            except Exception:
                continue
            index[m["sha256"]] = m

    bundle = ROOT / "scratch/kaggriculture_meta_search_bundle"
    manifest = bundle / "populations/dev_manifest.tsv"
    if manifest.exists():
        for r in csv.DictReader(manifest.open(encoding="utf-8"), delimiter="\t"):
            p = bundle / r["bundle_path"]
            if not p.exists():
                continue
            m = tape_meta(p, r["team"], int(r["source_seed"]))
            index[m["sha256"]] = m
    return index


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, default=Path("scratch/bkg4_search/official_canary64_checkpoint.jsonl"))
    ap.add_argument("--bundle", type=Path, default=Path("scratch/kaggriculture_meta_search_bundle"))
    ap.add_argument("--fresh-replays", type=Path, default=Path("scratch/bkg4_search/fresh_holdout_aug19"))
    ap.add_argument("--out", type=Path, default=Path("scratch/bkg4_search/optimization_oracle"))
    ap.add_argument("--historical-families", type=int, default=21)
    ap.add_argument("--fresh-holdout-episodes", type=int, default=8)
    args = ap.parse_args()

    checkpoint = (ROOT / args.checkpoint).resolve() if not args.checkpoint.is_absolute() else args.checkpoint
    bundle = (ROOT / args.bundle).resolve() if not args.bundle.is_absolute() else args.bundle
    fresh_replays = (ROOT / args.fresh_replays).resolve() if not args.fresh_replays.is_absolute() else args.fresh_replays
    out = (ROOT / args.out).resolve() if not args.out.is_absolute() else args.out
    out.mkdir(parents=True, exist_ok=True)

    sha_index = build_sha_index()
    canary = [json.loads(x) for x in checkpoint.read_text(encoding="utf-8").splitlines() if x.strip()]
    if len(canary) != 64 or {int(x["game_index"]) for x in canary} != set(range(64)):
        raise RuntimeError("official canary checkpoint must contain exact games 0..63")

    rows: list[dict] = []
    used_sha: set[str] = set()
    used_sources: set[tuple[int, int, str]] = set()
    for r in sorted(canary, key=lambda x: int(x["game_index"])):
        h = str(r["tape_sha256"])
        if h not in sha_index:
            raise FileNotFoundError(f"cannot resolve canary tape sha {h}")
        t = sha_index[h]
        used_sha.add(h)
        if t.get("episode_id") and t.get("source_seat") is not None:
            used_sources.add((int(t["episode_id"]), int(t["source_seat"]), str(t["team"])))
        rows.append({
            "tape": t["path"], "team": str(r["team"]), "seat": int(r["seat"]), "seed": int(r["seed"]),
            "protected": 1, "own_history": int(str(r["team"]) == "ReCurSiON"), "role": "canary_discovery",
            "reference_margin": int(round(float(r["margin"]))), "source_id": f"canary_game_{int(r['game_index'])}",
        })

    # Historical holdout: one untouched tape per family, both candidate seats. The
    # oracle never receives the family label; it is analysis metadata only.
    manifest = list(csv.DictReader((bundle / "populations/dev_manifest.tsv").open(encoding="utf-8"), delimiter="\t"))
    by_team: dict[str, list[dict]] = defaultdict(list)
    for r in manifest:
        p = bundle / r["bundle_path"]
        h = sha256(p)
        if h not in used_sha:
            by_team[r["team"]].append({**r, "path": p, "sha256": h})
    historical_added = 0
    for team in sorted(by_team)[: max(0, args.historical_families)]:
        r = sorted(by_team[team], key=lambda z: (int(z["source_seed"]), z["bundle_path"]))[0]
        used_sha.add(r["sha256"])
        for seat in (0, 1):
            rows.append({
                "tape": str(r["path"].resolve()), "team": team, "seat": seat, "seed": int(r["source_seed"]),
                "protected": 1, "own_history": int(team == "ReCurSiON"), "role": "historical_holdout",
                "reference_margin": None, "source_id": f"hist_{team}_{r['source_seed']}_s{seat}",
            })
            historical_added += 1

    # Fresh holdout: convert raw recent replays that are not already represented by
    # the canary tape hashes. This stays under scratch and is never a policy input.
    fresh_tape_dir = out / "fresh_tapes"
    fresh_added = 0
    episode_count = 0
    if fresh_replays.is_dir():
        for replay in sorted(fresh_replays.glob("episode-*-replay.json"), reverse=True):
            if episode_count >= max(0, args.fresh_holdout_episodes):
                break
            candidates = replay_to_tapes(replay, fresh_tape_dir)
            novel = [
                m for m in candidates
                if m["sha256"] not in used_sha
                and (int(m.get("episode_id") or 0), int(m.get("source_seat") or 0), str(m["team"])) not in used_sources
            ]
            if not novel:
                continue
            episode_count += 1
            for m in novel:
                used_sha.add(m["sha256"])
                used_sources.add((int(m.get("episode_id") or 0), int(m.get("source_seat") or 0), str(m["team"])))
                for seat in (0, 1):
                    rows.append({
                        "tape": m["path"], "team": m["team"], "seat": seat, "seed": m["seed"],
                        "protected": 1, "own_history": int(m["team"] == "ReCurSiON"), "role": "fresh_holdout",
                        "reference_margin": None, "source_id": f"fresh_{Path(m['path']).stem}_candseat{seat}",
                    })
                    fresh_added += 1

    families = {name: i for i, name in enumerate(sorted({r["team"] for r in rows}))}
    roles = {name: i for i, name in enumerate(sorted({r["role"] for r in rows}))}
    tsv = out / "cases.tsv"
    with tsv.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write("\t".join(map(str, (
                r["tape"], r["seat"], r["seed"], families[r["team"]], r["protected"], r["own_history"], roles[r["role"]]
            ))) + "\n")
    meta = out / "cases.jsonl"
    meta.write_text("".join(json.dumps({**r, "family_id": families[r["team"]], "role_id": roles[r["role"]]}, ensure_ascii=False, sort_keys=True) + "\n" for r in rows), encoding="utf-8")

    # Dense utility/assignment probes and sparse strategic transition probes. These
    # are fixed before seeing oracle outcomes.
    uw_probes = list(range(24, 697, 12))
    m_probes = [72, 144, 216, 288, 360, 432, 504, 576, 648]
    (out / "probes_uw.txt").write_text("\n".join(map(str, uw_probes)) + "\n", encoding="utf-8")
    (out / "probes_m.txt").write_text("\n".join(map(str, m_probes)) + "\n", encoding="utf-8")
    # Deterministic high-value screen: every canary loss, the ten lowest-margin
    # canary wins, and twelve cases from each untouched holdout role. Selection is
    # fixed before oracle outcomes are read.
    canary_rows = rows[:64]
    screen = [r for r in canary_rows if int(r["reference_margin"] or 0) < 0]
    screen.extend(sorted((r for r in canary_rows if int(r["reference_margin"] or 0) > 0), key=lambda r: int(r["reference_margin"]))[:10])
    screen.extend([r for r in rows if r["role"] == "historical_holdout"][:12])
    screen.extend([r for r in rows if r["role"] == "fresh_holdout"][:12])
    screen_families = {name: i for i, name in enumerate(sorted({r["team"] for r in screen}))}
    screen_roles = {name: i for i, name in enumerate(sorted({r["role"] for r in screen}))}
    (out / "screen_cases.jsonl").write_text("".join(json.dumps({**r, "family_id": screen_families[r["team"]], "role_id": screen_roles[r["role"]]}, ensure_ascii=False, sort_keys=True) + "\n" for r in screen), encoding="utf-8")
    (out / "screen_cases.tsv").write_text("".join("\t".join(map(str, (r["tape"], r["seat"], r["seed"], screen_families[r["team"]], r["protected"], r["own_history"], screen_roles[r["role"]]))) + "\n" for r in screen), encoding="utf-8")
    screen_probes = [48,72,96,120,144,168,192,216,240,264,288,336,384,432,480,528,576,624,672]
    (out / "screen_probes_uw.txt").write_text("\n".join(map(str, screen_probes)) + "\n", encoding="utf-8")

    summary = {
        "schema": 1,
        "cases": len(rows),
        "canary_cases": 64,
        "canary_losses": sum(int(float(r.get("margin", 0))) < 0 for r in canary),
        "historical_holdout_cases": historical_added,
        "fresh_holdout_cases": fresh_added,
        "families": families,
        "roles": roles,
        "uw_probe_steps": uw_probes,
        "macro_probe_steps": m_probes,
        "policy_inputs_exclude": ["team", "family_id", "role", "source_id", "reference_margin"],
        "screen_cases": len(screen),
        "screen_fresh_unique_sources": len({(r.get("source_id"), r["seat"]) for r in screen if r["role"] == "fresh_holdout"}),
    }
    (out / "manifest.json").write_text(json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
