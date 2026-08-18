#!/usr/bin/env python3
"""Merge two CUDA search islands and run one common development-only final screen."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import struct
import subprocess
from collections import defaultdict
from pathlib import Path

CASE_RE = re.compile(r"CASE_REPORT pi=(\d+) opponent=(\d+) seat=(\d+) seed=(\d+) margin=([-+0-9.eE]+)")
GEN_RE = re.compile(r"g(\d+)_r(\d+)\.bin$")
MANAGER_OFFSET = 4 + 96 * 40  # GPProgram header + 96 GPRule records.


def origin_info(path: Path) -> dict:
    parts = path.parts
    island = next((p for p in parts if p.startswith("island")), "unknown")
    m = GEN_RE.search(path.name)
    if m:
        stage = next((p for p in parts if p.startswith("stage")), "legacy")
        generation, rank = int(m.group(1)), int(m.group(2))
        sm = re.match(r"stage(\d+)", stage)
        stage_order = int(sm.group(1)) if sm else 1
    elif path.name.startswith("certified_niche_"):
        stage, generation, rank, stage_order = "resident-certified", -1, 0, 103
    elif path.name.startswith("niche_"):
        stage, generation, rank, stage_order = "resident-qd", -1, 0, 102
    elif path.name.startswith("subisland_"):
        stage, generation, rank, stage_order = "resident-subisland", -1, 0, 101
    else:
        stage, generation, rank, stage_order = "resident", -1, 0, 100
    return {"island": island, "stage": stage, "generation": generation, "archive_rank": rank, "stage_order": stage_order}


def descriptor(data: bytes) -> dict:
    if len(data) != 4096:
        return {"program_bytes": len(data), "n_rules": data[0] if data else None, "layout": "unknown"}
    m = memoryview(data)[MANAGER_OFFSET:]
    n_rules = data[0]
    phase_land = list(m[119:130]); phase_hires = list(m[130:141])
    phase_cow = list(m[141:152]); phase_sheep = list(m[152:163]); phase_goose = list(m[163:174])
    late_cash = struct.unpack_from("<H", m, 174 + 10 * 2)[0]
    return {
        "program_bytes": len(data), "n_rules": n_rules,
        "final_land": phase_land[-1], "max_daily_hires": max(phase_hires),
        "final_cows": phase_cow[-1], "final_sheep": phase_sheep[-1], "final_geese": phase_goose[-1],
        "late_cash_reserve": late_cash,
        "niche": f"r{min(n_rules,8)}-l{phase_land[-1]}-h{max(phase_hires)//3}-c{phase_cow[-1]//3}-s{phase_sheep[-1]//3}-k{late_cash//1000}",
    }


def baseline_metrics(rows: list[dict], families: list[str]) -> dict:
    losses = 0; fam = defaultdict(lambda: [0, 0, 0]); margins = []
    for r in rows:
        c = r["class"]; idx = int(r["route_index"]); family = families[idx]
        if c == "W": fam[family][0] += 1
        elif c == "T": fam[family][1] += 1
        else: fam[family][2] += 1; losses += 1
        margins.append(float(r["margin"]))
    return {"losses": losses, "losing_families": sum(v[2] > 0 for v in fam.values()),
            "floor": min(margins), "mean_margin": sum(margins) / len(margins)}


def candidate_metrics(pi: int, cases: list[tuple[int, int, float]], families: list[str], baseline_by_case: dict[tuple[int, int], str]) -> dict:
    fam = defaultdict(lambda: {"w": 0, "t": 0, "l": 0, "margins": []})
    w = t = l = regressions = conversions = 0; margins = []
    for oi, seat, margin in cases:
        cls = "W" if margin > 0 else "T" if margin == 0 else "L"
        if cls == "W": w += 1
        elif cls == "T": t += 1
        else: l += 1
        b = baseline_by_case[(oi, seat)]
        regressions += int(b == "W" and cls == "L")
        conversions += int(b == "L" and cls == "W")
        f = fam[families[oi]]; f[cls.lower()] += 1; f["margins"].append(margin); margins.append(margin)
    family_rows = {}
    for name, x in fam.items():
        n = x["w"] + x["t"] + x["l"]
        family_rows[name] = {"w": x["w"], "t": x["t"], "l": x["l"], "wr": x["w"] / n if n else 0.0,
                             "floor": min(x["margins"]), "mean_margin": sum(x["margins"]) / n}
    worst_wr = min(x["wr"] for x in family_rows.values())
    return {"candidate_index": pi, "wins": w, "ties": t, "losses": l, "cases": len(cases),
            "wr": w / len(cases), "losing_families": sum(x["l"] > 0 for x in family_rows.values()),
            "worst_family_wr": worst_wr, "regressions_vs_baseline": regressions,
            "converted_baseline_losses": conversions, "floor": min(margins),
            "mean_margin": sum(margins) / len(margins), "families": family_rows}


def rank_key(r: dict):
    return (r["losing_families"], r["losses"], r["regressions_vs_baseline"], -r["worst_family_wr"],
            -r["floor"], -r["wr"], -r["mean_margin"], r["descriptor"].get("n_rules", 255), r["candidate_id"])


def dominates(a: dict, b: dict) -> bool:
    amin = (a["losing_families"], a["losses"], a["regressions_vs_baseline"])
    bmin = (b["losing_families"], b["losses"], b["regressions_vs_baseline"])
    amax = (a["worst_family_wr"], a["floor"], a["wr"], a["mean_margin"])
    bmax = (b["worst_family_wr"], b["floor"], b["wr"], b["mean_margin"])
    no_worse = all(x <= y for x, y in zip(amin, bmin)) and all(x >= y for x, y in zip(amax, bmax))
    strictly = any(x < y for x, y in zip(amin, bmin)) or any(x > y for x, y in zip(amax, bmax))
    return no_worse and strictly


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", type=Path, required=True)
    ap.add_argument("--bundle", type=Path, required=True)
    ap.add_argument("--search-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--max-candidates", type=int, default=256)
    ap.add_argument("--seed", type=int, default=2184000)
    ap.add_argument("--tpb", type=int, default=128)
    ap.add_argument("--gpus", default="0,1")
    ap.add_argument("--tuning0", type=Path)
    ap.add_argument("--tuning1", type=Path)
    args = ap.parse_args()
    bundle = args.bundle.resolve(); root = args.search_root.resolve(); out = args.output.resolve(); out.mkdir(parents=True, exist_ok=True)

    manifest = list(csv.DictReader((bundle / "populations/dev_manifest.tsv").open(encoding="utf-8"), delimiter="\t"))
    if len(manifest) != 168 or any(r["bank_role"] != "dev" for r in manifest):
        raise RuntimeError("final screen requires exactly the 168 development tapes")
    families = [r["team"] for r in manifest]
    baseline = list(csv.DictReader((bundle / "populations/baseline_dev_cases.tsv").open(encoding="utf-8"), delimiter="\t"))
    baseline_by_case = {(int(r["route_index"]), int(r["seat"])): r["class"] for r in baseline}
    if len(baseline_by_case) != 336:
        raise RuntimeError("baseline development evidence must contain 168 tapes x both seats")

    # Exact-deduplicate canonical 4096-byte program records while preserving ancestry/origins.
    unique: dict[str, dict] = {}
    candidate_paths = list(root.glob("island*/stage*/archive/*.bin"))
    candidate_paths += list(root.glob("island*/subisland_*_best.bin"))
    candidate_paths += list(root.glob("island*/niche_*_best.bin"))
    candidate_paths += list(root.glob("island*/certified_niche_*_L*.bin"))
    for p in sorted(set(candidate_paths)):
        data = p.read_bytes()
        if len(data) != 4096:
            continue
        h = hashlib.sha256(data).hexdigest(); info = origin_info(p)
        rec = unique.setdefault(h, {"sha256": h, "candidate_id": h[:16], "path": str(p), "origins": [], "descriptor": descriptor(data)})
        rec["origins"].append({**info, "path": str(p)})
        old = origin_info(Path(rec["path"]))
        if (info["stage_order"], info["generation"], -info["archive_rank"]) > (old["stage_order"], old["generation"], -old["archive_rank"]):
            rec["path"] = str(p)
    if not unique:
        raise RuntimeError("no island elite binaries found")

    # Screening priority: later funnel stages first, then latest generation/top archive rank, while guaranteeing both islands.
    records = list(unique.values())
    def pre_key(r):
        best = max((o["stage_order"], o["generation"], -o["archive_rank"]) for o in r["origins"])
        return (-best[0], -best[1], -best[2], r["candidate_id"])
    records.sort(key=pre_key)
    selected = []
    for island in ("island0", "island1"):
        for r in records:
            if any(o["island"] == island for o in r["origins"]): selected.append(r); break
    seen = {r["sha256"] for r in selected}
    for r in records:
        if r["sha256"] not in seen:
            selected.append(r); seen.add(r["sha256"])
        if len(selected) >= args.max_candidates: break

    with (out / "deduplicated_elites.jsonl").open("w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    gpus = [x.strip() for x in args.gpus.split(",") if x.strip()]
    if not gpus: gpus = ["0"]
    tunings = {}
    for gpu, path in (("0", args.tuning0), ("1", args.tuning1)):
        if path and path.exists(): tunings[gpu] = int(json.loads(path.read_text())["best_tpb"])
    chunks = [[] for _ in gpus]
    for i, r in enumerate(selected): chunks[i % len(gpus)].append((i, r))
    commands = []; processes = []
    for gpu, chunk in zip(gpus, chunks):
        if not chunk: continue
        tpb = tunings.get(gpu, args.tpb)
        cmd = [str(args.binary), "--routes", str(bundle / "actuators/v6_routes.txt"),
               "--opponent-routes", str(bundle / "populations/dev168.txt"), "--opponents", "168",
               "--generations", "1", "--seeds", "1", "--seed-start", str(args.seed), "--fitness", "wins",
               "--max-step", "720", "--report-cases", "--tpb", str(tpb), "--island-mode", "exploit"]
        for _, r in chunk: cmd += ["--population-program", r["path"]]
        log = out / f"final_screen_gpu{gpu}.log"
        env = dict(__import__("os").environ); env["CUDA_VISIBLE_DEVICES"] = gpu
        fh = log.open("w")
        proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env)
        commands.append({"gpu": gpu, "tpb": tpb, "global_indices": [i for i, _ in chunk], "command": cmd, "log": str(log)})
        processes.append((proc, fh, log, chunk))
    (out / "final_screen_commands.json").write_text(json.dumps(commands, indent=2) + "\n")
    failures = []
    for proc, fh, log, chunk in processes:
        rc = proc.wait(); fh.close()
        if rc != 0: failures.append((rc, str(log)))
    if failures:
        raise RuntimeError(f"final common CUDA screen failed: {failures}")

    case_map: dict[int, list[tuple[int, int, float]]] = defaultdict(list)
    for _, (_, _, log, chunk) in zip(commands, processes):
        local_to_global = {local: global_i for local, (global_i, _) in enumerate(chunk)}
        for line in log.read_text(errors="replace").splitlines():
            m = CASE_RE.search(line)
            if m:
                gi = local_to_global[int(m.group(1))]
                case_map[gi].append((int(m.group(2)), int(m.group(3)), float(m.group(5))))
    if any(len(case_map[i]) != 336 for i in range(len(selected))):
        bad = {i: len(case_map[i]) for i in range(len(selected)) if len(case_map[i]) != 336}
        raise RuntimeError(f"incomplete final screen cases: {bad}")

    results = []
    for i, rec in enumerate(selected):
        met = candidate_metrics(i, case_map[i], families, baseline_by_case)
        results.append({**rec, **met})
    results.sort(key=rank_key)
    for rank, r in enumerate(results): r["rank"] = rank + 1

    dominated_ids = set()
    for i, b in enumerate(results):
        for j, a in enumerate(results):
            if i != j and dominates(a, b): dominated_ids.add(b["candidate_id"]); break
    nondominated = [r for r in results if r["candidate_id"] not in dominated_ids]

    # Preserve global strength plus lightweight strategy niches and both island ancestries.
    top = results[:32]
    top_ids = {r["candidate_id"] for r in top}
    best_by_niche = {}
    for r in results:
        key = (r["descriptor"].get("niche"), tuple(sorted({o["island"] for o in r["origins"]})))
        best_by_niche.setdefault(key, r)
    for r in best_by_niche.values():
        if r["candidate_id"] not in top_ids:
            top.append(r); top_ids.add(r["candidate_id"])
        if len(top) >= 64: break

    with (out / "final_screen_results.jsonl").open("w") as f:
        for r in results: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with (out / "nondominated.jsonl").open("w") as f:
        for r in nondominated: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    top_dir = out / "top_candidates"; top_dir.mkdir(exist_ok=True)
    with (out / "top_candidates.jsonl").open("w") as f:
        for r in top:
            dst = top_dir / f"rank{r['rank']:03d}_{r['candidate_id']}.bin"; shutil.copy2(r["path"], dst)
            row = dict(r); row["exported_binary"] = str(dst.relative_to(out)); f.write(json.dumps(row, ensure_ascii=False) + "\n")

    base = baseline_metrics(baseline, families); best = results[0]
    if best["losing_families"] <= base["losing_families"] - 2 and best["losses"] <= base["losses"] - 5 and best["regressions_vs_baseline"] <= 2:
        classification = "A. MAJOR NEW META BASIN FOUND"
    elif best["losses"] < base["losses"] or best["losing_families"] < base["losing_families"]:
        classification = "B. STRONG LOCAL IMPROVEMENTS ONLY"
    else:
        classification = "C. SEARCH COLLAPSED TO INCUMBENT BASIN"
    summary = {"unique_elites": len(records), "screened": len(selected), "nondominated": len(nondominated),
               "baseline": base, "best_candidate": {k: best[k] for k in ("candidate_id","rank","wins","ties","losses","losing_families","regressions_vs_baseline","converted_baseline_losses","worst_family_wr","floor","mean_margin")},
               "classification": classification}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    (out / "reproduction_commands.txt").write_text("Final screen commands are stored exactly in final_screen_commands.json.\nCandidate binaries are canonical 4096-byte GPProgram records; SHA-256 is the candidate identity.\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
