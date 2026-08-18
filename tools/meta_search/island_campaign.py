#!/usr/bin/env python3
"""Run one independent long-lived Kaggriculture CUDA search island."""
from __future__ import annotations
import argparse, json, os, signal, subprocess, time
from pathlib import Path

_CHILD: subprocess.Popen | None = None


def stop_child(signum=None, frame=None):
    global _CHILD
    if _CHILD is not None and _CHILD.poll() is None:
        _CHILD.terminate()


def read_json(path: Path):
    return json.loads(path.read_text())


def main() -> int:
    global _CHILD
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", type=Path, required=True)
    ap.add_argument("--bundle", type=Path, required=True)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--tuning", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--island", type=int, choices=(0, 1), required=True)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    signal.signal(signal.SIGTERM, stop_child); signal.signal(signal.SIGINT, stop_child)
    cfg = read_json(args.config); tuning = read_json(args.tuning)
    bundle = args.bundle.resolve(); out = args.output.resolve(); out.mkdir(parents=True, exist_ok=True)
    mode = "exploit" if args.island == 0 else "explore"
    search_seed = int(cfg["island0_seed"] if args.island == 0 else cfg["island1_seed"])
    tpb = int(tuning["best_tpb"])
    candidate_routes = bundle / "actuators/v6_routes.txt"
    total_search = float(cfg["search_walltime_seconds"])
    summary = {"island": args.island, "mode": mode, "search_seed": search_seed,
               "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "tpb": tpb, "stages": [], "development_gates": []}
    previous_best: Path | None = None

    for si, stage in enumerate(cfg["stages"]):
        stage_out = out / stage["name"]; archive = stage_out / "archive"; stage_out.mkdir(parents=True, exist_ok=True)
        done_file = stage_out / "stage_done.json"
        if args.resume and done_file.exists():
            done = read_json(done_file); summary["stages"].append(done)
            best = archive / "checkpoint_best.bin"
            if best.exists(): previous_best = best
            continue
        if args.resume and (archive / "checkpoint_best.bin").exists():
            previous_best = archive / "checkpoint_best.bin"
        stage_seconds = max(30.0, total_search * float(stage["walltime_fraction"]))
        opponent_count = sum(1 for x in (bundle / stage["route_list"]).read_text().splitlines() if x.strip())
        measured_turn_rate = float(tuning.get("best_turn_steps_per_sec", 0.0))
        stage_denominator = opponent_count * 2 * int(stage["max_step"])
        estimated_candidates_per_hour = measured_turn_rate * 3600.0 / stage_denominator if measured_turn_rate > 0.0 else 0.0
        stage_cap = int(cfg.get("population_cap", 0))
        if measured_turn_rate > 0.0:
            target_gen_seconds = float(stage.get("target_generation_seconds", 180.0))
            predicted = max(int(cfg["elite"]), int(measured_turn_rate * target_gen_seconds / stage_denominator))
            stage_cap = predicted if stage_cap <= 0 else min(stage_cap, predicted)
        print(f"ISLAND {args.island} PLAN {stage['name']} opponents={opponent_count} measured_turn_steps_per_sec={measured_turn_rate:.1f} estimated_candidates_per_hour={estimated_candidates_per_hour:.1f} population_cap={stage_cap or 'VRAM_ONLY'}", flush=True)
        fractions = [float(cfg["vram_target_fraction"])] + [float(x) for x in cfg.get("vram_fallback_fractions", [])]
        rc = None; used_fraction = None; started = time.time()
        for attempt, fraction in enumerate(fractions):
            archive.mkdir(parents=True, exist_ok=True)
            cmd = [str(args.binary), "--routes", str(candidate_routes),
                   "--opponent-routes", str(bundle / stage["route_list"]),
                   "--opponents", str(opponent_count),
                   "--generations", "1000000000", "--seeds", str(stage["seeds"]),
                   "--seed-start", str(cfg["seed_start"]), "--elite", str(cfg["elite"]),
                   "--tpb", str(tpb), "--search-seed", str(search_seed + si * 1000003 + attempt * 7919),
                   "--fitness", stage["fitness"], "--max-step", str(stage["max_step"]),
                   "--island-mode", mode, "--walltime-seconds", str(stage_seconds),
                   "--auto-vram-fraction", str(fraction), "--archive-dir", str(archive),
                   "--archive-count", str(cfg["archive_count"]),
                   "--archive-interval", str(cfg["archive_interval_generations"]),
                   "--out", str(stage_out / "best.txt")]
            if stage_cap > 0: cmd += ["--population-cap", str(stage_cap)]
            if previous_best is not None and previous_best.exists(): cmd += ["--seed-program", str(previous_best)]
            (stage_out / "command.json").write_text(json.dumps(cmd, indent=2) + "\n")
            log_path = stage_out / f"search_attempt{attempt}.log"
            print(f"ISLAND {args.island} START {stage['name']} fraction={fraction} tpb={tpb}", flush=True)
            with log_path.open("w") as log:
                _CHILD = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, text=True)
                rc = _CHILD.wait(); _CHILD = None
            if rc == 0:
                used_fraction = fraction; break
            tail = log_path.read_text(errors="replace")[-8000:]
            if "out of memory" not in tail.lower() and "cudaErrorMemoryAllocation" not in tail:
                break
            print(f"ISLAND {args.island} OOM {stage['name']} fraction={fraction}; retrying lower fraction", flush=True)
        elapsed = time.time() - started
        if rc != 0:
            fail = {"name": stage["name"], "returncode": rc, "elapsed_seconds": elapsed, "status": "failed"}
            summary["stages"].append(fail); (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
            return int(rc or 2)
        previous_best = archive / "checkpoint_best.bin"
        if si + 1 < len(cfg["stages"]):
            gate_log = stage_out / "development_gate.log"
            gate_cmd = [str(args.binary), "--routes", str(candidate_routes),
                        "--opponent-routes", str(bundle / "populations/dev168.txt"), "--opponents", "168",
                        "--load", str(previous_best), "--seeds", "1", "--seed-start", str(cfg["seed_start"]),
                        "--max-step", "720", "--fitness", "balanced", "--tpb", str(tpb),
                        "--island-mode", mode, "--report-cases"]
            (stage_out / "development_gate_command.json").write_text(json.dumps(gate_cmd, indent=2) + "\n")
            gate_started = time.time()
            with gate_log.open("w") as log:
                _CHILD = subprocess.Popen(gate_cmd, stdout=log, stderr=subprocess.STDOUT, text=True)
                gate_rc = _CHILD.wait(); _CHILD = None
            gate_elapsed = time.time() - gate_started
            gate = {"after_stage": stage["name"], "status": "complete" if gate_rc == 0 else "failed",
                    "elapsed_seconds": gate_elapsed, "returncode": gate_rc, "log": str(gate_log)}
            summary["development_gates"].append(gate)
            if gate_rc != 0:
                (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
                return int(gate_rc or 2)
            print(f"ISLAND {args.island} DEV_GATE {stage['name']} elapsed={gate_elapsed:.1f}", flush=True)
        done = {"name": stage["name"], "status": "complete", "elapsed_seconds": elapsed,
                "vram_fraction": used_fraction, "population_cap": stage_cap, "measured_turn_steps_per_sec": measured_turn_rate,
                "estimated_candidates_per_hour": estimated_candidates_per_hour,
                "max_step": stage["max_step"], "opponent_routes": stage["route_list"],
                "checkpoint_best": str(previous_best)}
        done_file.write_text(json.dumps(done, indent=2) + "\n")
        summary["stages"].append(done)
        print(f"ISLAND {args.island} DONE {stage['name']} elapsed={elapsed:.1f}", flush=True)
    summary["status"] = "complete"; (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
