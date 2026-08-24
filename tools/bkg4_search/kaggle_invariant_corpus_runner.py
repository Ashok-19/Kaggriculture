#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from tools.bkg4_search.edit_search.schema import ParentProfile
from tools.bkg4_search.generate_invariant_specialists import (
    _bank_specs,
    combine_offline_banks,
    parser as generator_parser,
    run as run_generator,
)
from tools.bkg4_search.invariant_evaluator import InvariantExactCudaEvaluator, load_invariant_results
from tools.bkg4_search.invariant_factory import load_offline_specialist_bank, split_specialist_panels
from tools.bkg4_search.invariant_search import InvariantSearchBackend
from tools.bkg4_search.invariant_factory import SpecialistFactory
from tools.bkg4_search.kaggle_invariant_corpus import sha256_file, write_validated_shard


def load_support(support: str | Path) -> tuple[Path, dict[str, Any]]:
    support = Path(support).resolve()
    manifest_path = support / "support_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    required = {
        "schema_version": 1,
        "source_commit": "c012c2c6c1de18fc250cf9e5425c3aea1f1d273c",
        "candidate_abi": "B4IM1/v1",
        "result_abi": "B4IC1/v1",
        "bkg4_abi": 4,
        "feature_registry": 1,
        "fast_math": False,
    }
    for key, value in required.items():
        if manifest.get(key) != value:
            raise ValueError(f"support provenance mismatch for {key}: {manifest.get(key)!r} != {value!r}")
    selection = manifest.get("selection", {})
    if selection.get("policy") != "specialist_quality_diversity_v1" or int(selection.get("screen_cases", 0)) != 128 or int(selection.get("label_cases", 0)) != 512:
        raise ValueError("support does not contain the corrected specialist-aware 128/512 selection pipeline")
    for rel, meta in manifest["files"].items():
        path = support / rel
        if not path.is_file() or path.stat().st_size != int(meta["bytes"]) or sha256_file(path) != str(meta["sha256"]):
            raise ValueError(f"support file integrity mismatch: {rel}")
    parent = support / manifest["parent"]["path"]
    profile = ParentProfile.from_directory(parent)
    if profile.combined_agent_identity != manifest["parent"]["canonical_identity"]:
        raise ValueError("support parent canonical identity mismatch")
    banks = _load_banks(support, manifest)
    if len(banks.cases) != int(manifest["bank_validation"]["cases"]):
        raise ValueError("support bank case count mismatch")
    panel = manifest["bank_validation"]
    split_specialist_panels(banks.cases, screen_cases=int(panel["screen_cases"]), label_cases=int(panel["label_cases"]), seed=20260824)
    return support, manifest


def _load_banks(support: Path, manifest: Mapping[str, Any]):
    loaded = [
        load_offline_specialist_bank(support / rel, cohort_kind=Path(rel).stem)
        for rel in manifest["banks"]
    ]
    return combine_offline_banks(loaded)


def _panel_record(support: Path, manifest: Mapping[str, Any], seed: int, screen_cases: int, label_cases: int) -> tuple[dict[str, Any], dict[str, Any]]:
    bank = _load_banks(support, manifest)
    screen, label = split_specialist_panels(bank.cases, screen_cases=screen_cases, label_cases=label_cases, seed=seed)

    def record(name: str, cases) -> dict[str, Any]:
        ids = sorted(case.case_id for case in cases)
        import hashlib
        digest = hashlib.sha256("\n".join(ids).encode()).hexdigest()
        return {"id": f"{name}:{digest}", "cases": len(cases), "tapes": len({case.tape_path for case in cases}), "case_ids_sha256": digest}

    return record("screen", screen), record("label", label)


def run_smoke(support: Path, manifest: Mapping[str, Any], *, binary: Path, device: int, output: Path) -> dict[str, Any]:
    smoke = support / "smoke"
    parent = support / manifest["parent"]["path"]
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    result = output / "results.b4ic"
    cmd = [
        str(binary), "--controller", str(parent / "controller.bkgc"),
        "--routes", str(parent / "routes.bkv6"), "--legacy-routes", str(parent / "legacy_routes.bkv6"),
        "--cases", str(smoke / "cases.tsv"), "--candidates", str(smoke / "candidates.b4im"),
        "--out", str(result), "--feed-slot", "5", "--feed-count", "13",
        "--strategic-route-slot", "19", "--tpb", "64", "--memory-fraction", "0.50",
        "--device", str(device),
    ]
    proc = subprocess.run(cmd, cwd=smoke, text=True, capture_output=True)
    (output / "cuda.log").write_text(proc.stdout + proc.stderr, encoding="utf-8")
    if proc.returncode:
        raise RuntimeError(f"exact smoke evaluator failed: {(proc.stderr or proc.stdout)[-4000:]}")
    actual = [[row.__dict__ for row in rows] for rows in load_invariant_results(result)]
    expected = json.loads((smoke / "expected.json").read_text(encoding="utf-8"))
    if actual != expected["expected_records"]:
        raise RuntimeError("exact B4IC smoke mismatch")
    if any(row["flags"] for rows in actual for row in rows):
        raise RuntimeError("exact B4IC smoke returned a nonzero fault flag")
    report = {"ok": True, "device": int(device), "candidates": len(actual), "cases": len(actual[0]), "records": sum(map(len, actual)), "log": proc.stdout.strip()}
    (output / "smoke_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def calibrate(
    support: Path, manifest: Mapping[str, Any], *, binary: Path, device: int,
    output: Path, candidates: int = 1024, seed: int = 20260824,
) -> dict[str, Any]:
    output = Path(output).resolve()
    profile = ParentProfile.from_directory(support / manifest["parent"]["path"])
    bank = _load_banks(support, manifest)
    panel = manifest["bank_validation"]
    screen, _ = split_specialist_panels(bank.cases, screen_cases=int(panel["screen_cases"]), label_cases=int(panel["label_cases"]), seed=seed)
    producer = InvariantExactCudaEvaluator(profile, binary=binary, device=device, tpb=64, memory_fraction=0.50)
    modules = SpecialistFactory(InvariantSearchBackend(route_ids=profile.route_ids, evaluator=producer), root_controller=profile.combined_agent_identity).propose(count=candidates, seed=seed).modules
    choices = []
    output.mkdir(parents=True, exist_ok=True)
    for tpb in (64, 128, 256):
        evaluator = InvariantExactCudaEvaluator(profile, binary=binary, device=device, tpb=tpb, memory_fraction=0.70)
        evaluator.evaluate(modules, screen, work_dir=output / f"tpb-{tpb}", panel_id=f"calibrate-{tpb}")
        profile_row = evaluator.profile_history[-1]
        choices.append({"tpb": tpb, "memory_fraction": 0.70, **profile_row})
    valid = [row for row in choices if row.get("gpu_games_per_second") is not None]
    if not valid:
        raise RuntimeError("calibration produced no CUDA throughput profile")
    chosen = max(valid, key=lambda row: float(row["gpu_games_per_second"]))
    report = {"device": device, "representative_candidates": len(modules), "screen_cases": len(screen), "choices": choices, "chosen_tpb": chosen["tpb"], "chosen_memory_fraction": chosen["memory_fraction"], "kernel_games_per_second": chosen["gpu_games_per_second"]}
    (output / "calibration.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def _generator_args(args: argparse.Namespace, shard_seed: int, work: Path) -> argparse.Namespace:
    values = [
        "--parent", str(args.support / args.manifest["parent"]["path"]),
        "--evaluator", str(args.binary), "--work-dir", str(work),
        "--proposals", str(args.proposals), "--retain", str(args.retain),
        "--label-retain", str(args.label_retain),
        "--screen-cases", str(args.screen_cases), "--label-cases", str(args.label_cases),
        "--recursive", str(args.recursive), "--recursive-rounds", str(args.recursive_rounds),
        "--repairs", str(args.repairs), "--ablations", str(args.ablations),
        "--followup-retain", str(args.followup_retain), "--seed", str(shard_seed),
        "--followup-label-retain", str(args.followup_label_retain),
        "--device", "0", "--tpb", str(args.tpb), "--memory-fraction", str(args.memory_fraction),
        "--skip-saturation",
    ]
    for rel in args.manifest["banks"]:
        values.extend(("--bank", f"{Path(rel).stem}={args.support / rel}"))
    return generator_parser().parse_args(values)


def run_worker(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output).resolve()
    island_root = output / f"island{args.island}"
    island_root.mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        report = {"dry_run": True, "island": args.island, "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "seed_base": args.base_seed, "support": str(args.support)}
        (island_root / "dry_run.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        return report
    shard_id = len(list(island_root.glob("shard-*/shard_manifest.json")))
    durations: list[float] = []
    totals = {"raw_proposals": 0, "screened": 0, "screen_retained": 0, "exact_labeled": 0, "labeled": 0, "exact_games": 0, "faults": 0, "shards": 0}
    worker_started = time.monotonic()
    while shard_id < args.max_shards:
        remaining = float(args.generation_deadline_monotonic) - time.monotonic()
        predicted = max(float(args.minimum_shard_seconds), (max(durations[-3:]) * 1.25 if durations else 0.0))
        free = shutil.disk_usage(output).free
        if remaining < predicted or free < int(args.disk_floor_bytes):
            break
        seed = int(args.base_seed) + int(args.island) * 1_000_000_000 + shard_id * 1_000_003
        work = island_root / f".work-shard-{shard_id:04d}"
        if work.exists():
            shutil.rmtree(work)
        started = time.monotonic()
        report = run_generator(_generator_args(args, seed, work))
        corpus = work / "corpus/invariant_specialists_exact_cuda.jsonl"
        records = [json.loads(line) for line in corpus.read_text(encoding="utf-8").splitlines() if line.strip()]
        screen_panel, label_panel = _panel_record(args.support, args.manifest, seed, args.screen_cases, args.label_cases)
        rounds = report["rounds"]
        counters = {
            "raw_proposals": sum(int(row["proposals"]) for row in rounds),
            "screened": sum(int(row["proposals"]) for row in rounds),
            "screen_retained": sum(int(row["screen_retained"]) for row in rounds),
            "exact_labeled": sum(int(row["exact_labeled"]) for row in rounds),
            "labeled": len(records),
            "exact_games": int(report["throughput"]["exact_games"]),
            "faults": 0,
            "elapsed_seconds": float(report["throughput"]["elapsed_seconds"]),
        }
        runtime = {
            "source_commit": args.manifest["source_commit"], "implementation_commit": args.manifest["implementation_commit"],
            "candidate_abi": report["candidate_abi"], "evaluator_abi": report["evaluator_abi"],
            "bkg4_abi": 4, "feature_registry": 1, "exact_cuda": True, "fast_math": False,
            "island": args.island, "logical_device": 0, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "tpb": args.tpb, "memory_fraction": args.memory_fraction, "cuda_profiles": report["throughput"]["cuda_profiles"],
        }
        write_validated_shard(
            island_root / f"shard-{shard_id:04d}", records, island=args.island, shard_id=shard_id,
            seed=seed, screen_panel=screen_panel, label_panel=label_panel, runtime=runtime, counters=counters,
        )
        shutil.rmtree(work)
        duration = time.monotonic() - started
        durations.append(duration)
        for key in totals:
            if key == "shards":
                continue
            totals[key] += int(counters.get(key, 0))
        totals["shards"] += 1
        worker_elapsed = max(time.monotonic() - worker_started, 1e-9)
        heartbeat = {
            "island": args.island, "shard": shard_id, "seed": seed,
            "duration_seconds": duration, "elapsed_seconds": worker_elapsed,
            "remaining_seconds": float(args.generation_deadline_monotonic) - time.monotonic(),
            "free_disk_bytes": shutil.disk_usage(output).free, "totals": totals,
            "edits_per_second": totals["labeled"] / worker_elapsed,
            "games_per_second": totals["exact_games"] / worker_elapsed,
        }
        (island_root / "heartbeat.json").write_text(json.dumps(heartbeat, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        shard_id += 1
    worker_elapsed = max(time.monotonic() - worker_started, 1e-9)
    final = {
        "island": args.island, "finished": True, "totals": totals, "durations": durations,
        "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "elapsed_seconds": worker_elapsed,
        "edits_per_second": totals["labeled"] / worker_elapsed,
        "games_per_second": totals["exact_games"] / worker_elapsed,
    }
    (island_root / "worker_report.json").write_text(json.dumps(final, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return final


def parser() -> argparse.ArgumentParser:
    out = argparse.ArgumentParser(description="Kaggle dual-T4 B4IM corpus worker and exact gates")
    commands = out.add_subparsers(dest="command", required=True)
    for name in ("run-smoke", "calibrate"):
        cmd = commands.add_parser(name)
        cmd.add_argument("--support", type=Path, required=True)
        cmd.add_argument("--binary", type=Path, required=True)
        cmd.add_argument("--device", type=int, default=0)
        cmd.add_argument("--output", type=Path, required=True)
        if name == "calibrate":
            cmd.add_argument("--candidates", type=int, default=1024)
            cmd.add_argument("--seed", type=int, default=20260824)
    worker = commands.add_parser("worker")
    worker.add_argument("--support", type=Path, required=True)
    worker.add_argument("--binary", type=Path, required=True)
    worker.add_argument("--output", type=Path, required=True)
    worker.add_argument("--island", type=int, choices=(0, 1), required=True)
    worker.add_argument("--base-seed", type=int, default=2026082400)
    worker.add_argument("--generation-deadline-monotonic", type=float, required=True)
    worker.add_argument("--proposals", type=int, default=20_000)
    worker.add_argument("--retain", type=int, default=2_048)
    worker.add_argument("--label-retain", type=int, default=4_096)
    worker.add_argument("--screen-cases", type=int, default=128)
    worker.add_argument("--label-cases", type=int, default=512)
    worker.add_argument("--recursive", type=int, default=4_096)
    worker.add_argument("--recursive-rounds", type=int, default=2)
    worker.add_argument("--repairs", type=int, default=1_024)
    worker.add_argument("--ablations", type=int, default=1_024)
    worker.add_argument("--followup-retain", type=int, default=1_024)
    worker.add_argument("--followup-label-retain", type=int, default=2_048)
    worker.add_argument("--tpb", type=int, default=64)
    worker.add_argument("--memory-fraction", type=float, default=0.70)
    worker.add_argument("--disk-floor-bytes", type=int, default=12 * 1024**3)
    worker.add_argument("--minimum-shard-seconds", type=float, default=900.0)
    worker.add_argument("--max-shards", type=int, default=10_000)
    worker.add_argument("--dry-run", action="store_true")
    return out


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    support, manifest = load_support(args.support)
    args.support, args.manifest = support, manifest
    if args.command == "run-smoke":
        result = run_smoke(support, manifest, binary=args.binary.resolve(), device=args.device, output=args.output)
    elif args.command == "calibrate":
        result = calibrate(support, manifest, binary=args.binary.resolve(), device=args.device, output=args.output, candidates=args.candidates, seed=args.seed)
    else:
        result = run_worker(args)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
