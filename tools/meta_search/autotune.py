#!/usr/bin/env python3
"""Per-GPU T4 autotuning for the resident Kaggriculture memetic search engine."""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path

HB_RE = re.compile(
    r"HEARTBEAT .*?generation_seconds=([0-9.eE+-]+).*?candidates_per_sec=([0-9.eE+-]+).*?games_per_sec=([0-9.eE+-]+)"
)
MEM_RE = re.compile(
    r"RESIDENT_MEMORY .*?safe_available=(\d+).*?safe_budget=(\d+).*?population=(\d+).*?per_candidate=(\d+).*?"
    r"allocated_fraction_of_safe=([0-9.eE+-]+).*?sizeof_SearchGenome=(\d+).*?sizeof_GPManager=(\d+).*?sizeof_Sim=(\d+)"
)


def gpu_info() -> dict:
    q = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,memory.free,compute_cap,power.limit",
        "--format=csv,noheader,nounits",
    ]
    p = subprocess.run(q, text=True, capture_output=True, check=True)
    lines = p.stdout.strip().splitlines()
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0].strip()
    physical = int(visible) if visible.isdigit() else 0
    line = lines[physical] if physical < len(lines) else lines[0]
    idx, name, total, free, cap, power = [x.strip() for x in line.split(",")]
    return {
        "physical_index": int(idx),
        "name": name,
        "memory_total_mib": int(total),
        "memory_free_mib": int(free),
        "compute_capability": cap,
        "power_limit_w": float(power),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", type=Path, required=True)
    ap.add_argument("--bundle", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--populations", default="4096,16384,65536")
    ap.add_argument("--tpb", default="64,128,256")
    ap.add_argument("--sub-islands", type=int, default=32)
    ap.add_argument("--engine-sub-islands", type=int, default=12)
    ap.add_argument("--universe", choices=("exploit", "explore"), default="exploit")
    ap.add_argument("--seed", type=int, default=730001)
    ap.add_argument("--vram-target", type=float, default=0.95)
    ap.add_argument("--reserve-mib", type=float, default=512.0)
    ap.add_argument("--target-generation-seconds", type=float, default=240.0)
    args = ap.parse_args()

    binary = args.binary.resolve()
    bundle = args.bundle.resolve()
    pops = [int(x) for x in args.populations.split(",") if x]
    tpbs = [int(x) for x in args.tpb.split(",") if x]
    trials: list[dict] = []

    # Benchmark exact 24-case core generations. The production run does NOT use
    # these small populations; it autosizes its resident pool from live safe VRAM.
    for population in pops:
        sub_islands = min(args.sub_islands, max(4, population // 256))
        for tpb in tpbs:
            with tempfile.TemporaryDirectory(prefix="kg_memetic_tune_") as td:
                cmd = [
                    str(binary), "--bundle", str(bundle), "--output", td,
                    "--population", str(population), "--sub-islands", str(sub_islands),
                    "--engine-sub-islands", str(min(args.engine_sub_islands, sub_islands)),
                    "--universe", args.universe,
                    "--local-per-island", "16", "--tpb", str(tpb), "--benchmark",
                    "--walltime-seconds", "900", "--search-seed", str(args.seed + population * 17 + tpb),
                    "--seed-start", "2184000", "--vram-target", str(args.vram_target),
                    "--reserve-mib", str(args.reserve_mib),
                ]
                p = subprocess.run(cmd, text=True, capture_output=True)
            hb = HB_RE.search(p.stdout)
            mm = MEM_RE.search(p.stdout)
            row = {
                "population": population,
                "sub_islands": sub_islands,
                "tpb": tpb,
                "returncode": p.returncode,
                "generation_seconds": float(hb.group(1)) if hb else None,
                "candidates_per_sec": float(hb.group(2)) if hb else 0.0,
                "games_per_sec": float(hb.group(3)) if hb else 0.0,
                "safe_available_bytes": int(mm.group(1)) if mm else None,
                "safe_budget_bytes": int(mm.group(2)) if mm else None,
                "per_candidate_bytes": int(mm.group(4)) if mm else None,
                "allocated_fraction_of_safe": float(mm.group(5)) if mm else None,
                "sizeof_SearchGenome": int(mm.group(6)) if mm else None,
                "sizeof_GPManager": int(mm.group(7)) if mm else None,
                "sizeof_Sim": int(mm.group(8)) if mm else None,
                "stderr_tail": p.stderr[-2000:],
            }
            trials.append(row)
            print(json.dumps(row), flush=True)

    good = [x for x in trials if x["returncode"] == 0 and x["games_per_sec"] > 0]
    if not good:
        raise RuntimeError(f"all resident autotune trials failed: {trials}")

    # Throughput is authoritative; candidate/s breaks near-ties. Prefer larger
    # measured batches when throughput is within 1% so launch overhead is amortized.
    peak = max(x["games_per_sec"] for x in good)
    near = [x for x in good if x["games_per_sec"] >= 0.99 * peak]
    best = max(near, key=lambda x: (x["population"], x["candidates_per_sec"]))
    per_candidate = next(x["per_candidate_bytes"] for x in good if x["per_candidate_bytes"])
    safe_budget = next(x["safe_budget_bytes"] for x in reversed(good) if x["safe_budget_bytes"])
    estimated_full_population = int(safe_budget // per_candidate)
    estimated_full_generation_seconds = (
        estimated_full_population * 24.0 / best["games_per_sec"] if best["games_per_sec"] > 0 else None
    )
    result = {
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu": gpu_info(),
        "universe": args.universe,
        "requested_engine_sub_islands": args.engine_sub_islands,
        "best_tpb": best["tpb"],
        "best_measured_population": best["population"],
        "best_sub_islands": best["sub_islands"],
        "best_games_per_sec": best["games_per_sec"],
        "best_candidates_per_sec": best["candidates_per_sec"],
        "per_candidate_bytes": per_candidate,
        "vram_target": args.vram_target,
        "estimated_full_vram_population": estimated_full_population,
        "estimated_full_vram_core_generation_seconds": estimated_full_generation_seconds,
        "target_generation_seconds": args.target_generation_seconds,
        "production_population": 0,
        "production_population_semantics": "0 means resident engine autosizes to the live 95%-safe-VRAM budget",
        "trials": trials,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
