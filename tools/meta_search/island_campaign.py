#!/usr/bin/env python3
"""Supervise one independent long-lived resident CUDA memetic search universe."""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
from pathlib import Path

_CHILD: subprocess.Popen | None = None


def stop_child(signum=None, frame=None):
    # bonsai_memetic_search treats SIGTERM/SIGINT as a graceful checkpoint request.
    global _CHILD
    if _CHILD is not None and _CHILD.poll() is None:
        _CHILD.terminate()


def main() -> int:
    global _CHILD
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", type=Path, required=True)
    ap.add_argument("--bundle", type=Path, required=True)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--tuning", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--island", type=int, choices=(0, 1), required=True)
    ap.add_argument("--walltime-seconds", type=float, required=True)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    signal.signal(signal.SIGTERM, stop_child)
    signal.signal(signal.SIGINT, stop_child)

    cfg = json.loads(args.config.read_text())
    tuning = json.loads(args.tuning.read_text())
    bundle = args.bundle.resolve()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)

    mode = "exploit" if args.island == 0 else "explore"
    mode_cfg = cfg[mode]
    search_seed = int(cfg["island0_seed"] if args.island == 0 else cfg["island1_seed"])
    tpb = int(tuning["best_tpb"])
    # Strategic topology is configuration, not an autotune variable. Small
    # benchmark populations may use fewer temporary sub-islands, but the long
    # production universe always preserves the intended 32-island grammar split.
    sub_islands = int(cfg["default_sub_islands"])
    engine_sub_islands = int(mode_cfg["engine_sub_islands"])
    if not (4 <= sub_islands <= 64 and 0 <= engine_sub_islands <= sub_islands):
        raise RuntimeError(f"invalid production island topology: {sub_islands=} {engine_sub_islands=}")

    cmd = [
        str(args.binary),
        "--bundle", str(bundle),
        "--output", str(out),
        "--universe", mode,
        "--sub-islands", str(sub_islands),
        "--engine-sub-islands", str(engine_sub_islands),
        "--tpb", str(tpb),
        "--epoch-generations", str(cfg["epoch_generations"]),
        "--stagnation-limit", str(mode_cfg["stagnation_limit"]),
        "--local-per-island", str(mode_cfg["local_per_island"]),
        "--promotion-family-every", str(cfg["promotion"]["family21_every_generations"]),
        "--promotion-medium-every", str(cfg["promotion"]["medium42_every_generations"]),
        "--promotion-full-every", str(cfg["promotion"]["full168_every_generations"]),
        "--walltime-seconds", str(args.walltime_seconds),
        "--vram-target", str(cfg["vram_target_fraction"]),
        "--reserve-mib", str(cfg["vram_reserve_mib"]),
        "--checkpoint-interval-seconds", str(cfg["checkpoint_interval_seconds"]),
        "--search-seed", str(search_seed),
    ]

    # GPU0 starts from all strong historical structural priors. GPU1 receives only
    # one anchor and is otherwise deliberately de-novo/high-mutation.
    prior_dir = bundle / "priors"
    priors = sorted(prior_dir.glob("*.bin")) if prior_dir.exists() else []
    use_priors = priors if args.island == 0 else priors[:1]
    for p in use_priors:
        cmd += ["--seed-program", str(p)]
    if args.resume:
        cmd.append("--resume")

    record = {
        "island": args.island,
        "mode": mode,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "search_seed": search_seed,
        "tpb": tpb,
        "sub_islands": sub_islands,
        "engine_sub_islands": engine_sub_islands,
        "engine_sub_island_fraction": engine_sub_islands / sub_islands,
        "walltime_seconds": args.walltime_seconds,
        "priors": [str(p) for p in use_priors],
        "command": cmd,
    }
    (out / "command.json").write_text(json.dumps(record, indent=2) + "\n")

    log_path = out / "resident_search.log"
    with log_path.open("a" if args.resume else "w") as log:
        _CHILD = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, text=True)
        rc = _CHILD.wait()
        _CHILD = None

    summary_path = out / "resident_summary.json"
    wrapper_summary = {
        **record,
        "status": "complete" if rc == 0 else "failed",
        "returncode": rc,
        "resident_summary": json.loads(summary_path.read_text()) if summary_path.exists() else None,
        "log": str(log_path),
    }
    (out / "summary.json").write_text(json.dumps(wrapper_summary, indent=2) + "\n")
    return int(rc)


if __name__ == "__main__":
    raise SystemExit(main())
