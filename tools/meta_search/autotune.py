#!/usr/bin/env python3
"""Short per-GPU launch-geometry benchmark for the dual-T4 campaign."""
from __future__ import annotations
import argparse, json, os, re, subprocess, time
from pathlib import Path

PERF_RE = re.compile(r"games_per_sec=([0-9.eE+-]+)")
MEM_RE = re.compile(r"sizeof_GPProgram=(\d+) sizeof_Sim=(\d+)")


def gpu_info() -> dict:
    q = ["nvidia-smi", "--query-gpu=name,memory.total,memory.free,compute_cap", "--format=csv,noheader,nounits"]
    p = subprocess.run(q, text=True, capture_output=True, check=True)
    lines = p.stdout.strip().splitlines()
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0].strip()
    physical = int(visible) if visible.isdigit() else 0
    line = lines[physical] if physical < len(lines) else lines[0]
    name, total, free, cap = [x.strip() for x in line.split(",")]
    return {"name": name, "memory_total_mib": int(total), "memory_free_mib": int(free), "compute_capability": cap}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", type=Path, required=True)
    ap.add_argument("--bundle", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--populations", default="512,2048,8192,32768")
    ap.add_argument("--tpb", default="64,128,256")
    ap.add_argument("--seed", type=int, default=730001)
    args = ap.parse_args()
    bundle = args.bundle.resolve(); binary = args.binary.resolve()
    candidate = bundle / "actuators/v6_routes.txt"
    opponents = bundle / "populations/stage1_core12.txt"
    trials = []
    populations = [int(x) for x in args.populations.split(",") if x]
    for population in populations:
        for tpb in [int(x) for x in args.tpb.split(",") if x]:
            cmd = [str(binary), "--routes", str(candidate), "--opponent-routes", str(opponents),
                   "--population", str(population), "--generations", "1", "--seeds", "1",
                   "--opponents", "3", "--max-step", "72", "--elite", "8", "--tpb", str(tpb),
                   "--search-seed", str(args.seed + population * 17 + tpb), "--seed-start", "2184000", "--fitness", "robust",
                   "--island-mode", "exploit"]
            t0 = time.monotonic()
            p = subprocess.run(cmd, text=True, capture_output=True)
            dt = time.monotonic() - t0
            m = PERF_RE.findall(p.stdout); mm = MEM_RE.findall(p.stdout)
            trials.append({"population": population, "tpb": tpb, "returncode": p.returncode, "elapsed_seconds": dt,
                           "games_per_sec": float(m[-1]) if m else 0.0,
                           "turn_steps_per_sec": (float(m[-1]) * 72.0) if m else 0.0,
                           "sizeof_GPProgram": int(mm[-1][0]) if mm else None,
                           "sizeof_Sim": int(mm[-1][1]) if mm else None,
                           "stderr_tail": p.stderr[-2000:]})
    good = [x for x in trials if x["returncode"] == 0 and x["games_per_sec"] > 0]
    if not good:
        raise RuntimeError(f"all autotune trials failed: {trials}")
    best = max(good, key=lambda x: x["games_per_sec"])
    result = {"cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "gpu": gpu_info(),
              "tested_populations": populations, "best_population": best["population"], "best_tpb": best["tpb"],
              "best_games_per_sec": best["games_per_sec"], "best_turn_steps_per_sec": best["turn_steps_per_sec"],
              "trials": trials}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
