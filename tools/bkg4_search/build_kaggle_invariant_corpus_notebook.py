#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "notebooks/kaggriculture_b4im_elite_corpus_dual_t4"
OUT_NOTEBOOK = OUT_DIR / "kaggriculture_b4im_elite_corpus_dual_t4.ipynb"
OUT_META = OUT_DIR / "kernel-metadata.json"


def _source(text: str) -> list[str]:
    return [line + "\n" for line in text.strip("\n").splitlines()]


def _md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": _source(text)}


def _code(text: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": _source(text)}


def render_notebook() -> str:
    cells = [
        _md(r'''
# Kaggriculture B4IM Elite Specialist Corpus - Dual T4

Exact-CUDA, specialist-aware B4IM1 corpus production for future BonsaiEditFormer training.
This notebook does not train a model, optimize or promote the champion, run official Python
games, modify the support input, or submit to the competition.
'''),
        _code(r'''
# Configuration: edit only this cell before Save & Run All.
from pathlib import Path
import gzip, hashlib, json, os, shutil, signal, subprocess, sys, tarfile, time

SUPPORT_DATASET = "ashok205/kaggriculture-b4im-elite-corpus-support"
OUTPUT_ROOT = Path("/kaggle/working/kaggriculture_b4im_specialist_corpus")
RUNTIME_ROOT = Path("/kaggle/working/kaggriculture_b4im_support_runtime")
BUILD_ROOT = Path("/kaggle/working/b4im_cuda_build")

TOTAL_WALLTIME_SECONDS = 10 * 60 * 60
FINAL_RESERVE_SECONDS = 45 * 60
BASE_SEED = 2026082400
SCREEN_CASES = 64
LABEL_CASES = 400
SHARD_PROPOSALS = 20_000
RETAINED_PER_SHARD = 2_048
RECURSIVE = 4_096
RECURSIVE_ROUNDS = 2
CEGIS_REPAIRS = 1_024
ABLATIONS = 1_024
FOLLOWUP_RETAIN = 1_024
GPU_MEMORY_FRACTION = 0.70
DISK_RESERVE_BYTES = 12 * 1024**3
MINIMUM_SHARD_SECONDS = 15 * 60
MONITOR_INTERVAL_SECONDS = 60
IDLE_GPU_LIMIT_SAMPLES = 5

PUBLISH_CORPUS = False
CORPUS_DATASET = "ashok205/kaggriculture-b4im-elite-specialist-corpus"

PIPELINE_STARTED_MONOTONIC = time.monotonic()
HARD_EXIT_DEADLINE = PIPELINE_STARTED_MONOTONIC + TOTAL_WALLTIME_SECONDS
GENERATION_DEADLINE = HARD_EXIT_DEADLINE - FINAL_RESERVE_SECONDS
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
'''),
        _code(r'''
# Robust support discovery, safe materialization, and provenance/integrity gate.
def locate_input(slug, marker):
    candidates = [Path("/kaggle/input/datasets") / slug, Path("/kaggle/input") / slug]
    for candidate in candidates:
        if (candidate / marker).is_file():
            return candidate
    hits = sorted({path.parent for path in Path("/kaggle/input").rglob(marker)})
    intended = [path for path in hits if "b4im-elite-corpus-support" in str(path).lower()]
    if len(intended) == 1:
        return intended[0]
    if len(hits) != 1:
        raise RuntimeError(f"cannot uniquely locate {slug} via {marker}: {hits}")
    return hits[0]

def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()

SUPPORT_INPUT = locate_input(SUPPORT_DATASET, "support_manifest.json")
support_manifest = json.loads((SUPPORT_INPUT / "support_manifest.json").read_text())
expected = {
    "schema_version": 1,
    "source_commit": "c012c2c6c1de18fc250cf9e5425c3aea1f1d273c",
    "candidate_abi": "B4IM1/v1", "result_abi": "B4IC1/v1",
    "bkg4_abi": 4, "feature_registry": 1, "fast_math": False,
}
for key, value in expected.items():
    if support_manifest.get(key) != value:
        raise RuntimeError(f"support provenance mismatch: {key}")
if support_manifest["cuda"]["required_architecture"] != "sm_75":
    raise RuntimeError("support does not declare the required T4 sm_75 architecture")

if RUNTIME_ROOT.exists():
    shutil.rmtree(RUNTIME_ROOT)
expanded = SUPPORT_INPUT / "support_bundle"
archive = SUPPORT_INPUT / support_manifest.get("bundle_file", "support_bundle.tar")
if expanded.is_dir():
    shutil.copytree(expanded, RUNTIME_ROOT)
elif archive.is_file():
    if archive.stat().st_size != int(support_manifest["bundle_bytes"]) or file_sha256(archive) != support_manifest["bundle_sha256"]:
        raise RuntimeError("support bundle integrity mismatch")
    RUNTIME_ROOT.mkdir(parents=True)
    root = RUNTIME_ROOT.resolve()
    with tarfile.open(archive, "r") as bundle:
        for member in bundle.getmembers():
            target = (RUNTIME_ROOT / member.name).resolve()
            if (target != root and root not in target.parents) or not member.isfile():
                raise RuntimeError(f"unsafe support archive member: {member.name}")
        bundle.extractall(RUNTIME_ROOT)
else:
    raise RuntimeError("support dataset has neither expanded support_bundle nor support_bundle.tar")
for rel, meta in support_manifest["files"].items():
    path = RUNTIME_ROOT / rel
    if not path.is_file() or path.stat().st_size != int(meta["bytes"]) or file_sha256(path) != meta["sha256"]:
        raise RuntimeError(f"materialized support integrity mismatch: {rel}")
shutil.copy2(SUPPORT_INPUT / "support_manifest.json", RUNTIME_ROOT / "support_manifest.json")
sys.path.insert(0, str(RUNTIME_ROOT))
from tools.bkg4_search.kaggle_invariant_corpus_runner import load_support
load_support(RUNTIME_ROOT)
print(json.dumps({"support_input": str(SUPPORT_INPUT), "runtime": str(RUNTIME_ROOT), "package_version": support_manifest["support_package_version"], "parent": support_manifest["parent"]["canonical_identity"], "cases": support_manifest["bank_validation"]["cases"], "tapes": support_manifest["packaged_tape_count"]}, indent=2))
'''),
        _code(r'''
# Hardware gate: production requires two visible NVIDIA T4s.
def run(command, *, cwd=None, env=None, check=True):
    proc = subprocess.run([str(x) for x in command], cwd=cwd, env=env, text=True, capture_output=True)
    print("$", " ".join(str(x) for x in command))
    if check and proc.returncode:
        raise RuntimeError((proc.stderr or proc.stdout)[-6000:])
    return proc

query = run(["nvidia-smi", "--query-gpu=index,name,uuid,memory.total,compute_cap", "--format=csv,noheader,nounits"]).stdout
gpus = []
for line in query.splitlines():
    index, name, uuid, memory, capability = [part.strip() for part in line.split(",")]
    gpus.append({"index": int(index), "model": name, "uuid": uuid, "memory_mib": int(memory), "compute_capability": capability})
if len(gpus) < 2 or any("T4" not in gpu["model"] or gpu["compute_capability"] != "7.5" for gpu in gpus[:2]):
    raise RuntimeError("Select Kaggle T4 x2 accelerator and restart.")
print(json.dumps({"production_gpus": gpus[:2]}, indent=2))
'''),
        _code(r'''
# Compile exactly once for native T4 sm_75; exact market arithmetic forbids fast math.
# NVCC/host compatibility varies across Kaggle images, so probe installed compilers by
# compiling instead of assuming g++-9 exists or that the newest g++ is compatible.
BUILD_ROOT.mkdir(parents=True, exist_ok=True)
nvcc_candidates = [shutil.which("nvcc"), Path("/usr/local/cuda/bin/nvcc")]
nvcc_candidates.extend(sorted(Path("/usr/local").glob("cuda-*/bin/nvcc"), reverse=True))
nvcc = next((str(path) for path in nvcc_candidates if path and Path(path).is_file()), None)
if nvcc is None:
    raise RuntimeError("nvcc not found; confirm the Kaggle T4 x2 accelerator is enabled")
compiler_candidates = []
for name in ("g++-9", "g++-10", "g++-11", "g++-12", "g++-13", "g++"):
    path = shutil.which(name)
    if path and path not in compiler_candidates:
        compiler_candidates.append(path)
if not compiler_candidates:
    raise RuntimeError("no host C++ compiler found for nvcc")
make = None; host_cxx = None; build_failures = {}
for candidate in compiler_candidates:
    evaluator_candidate = BUILD_ROOT / "bkg4_invariant_edit_eval"
    evaluator_candidate.unlink(missing_ok=True)
    attempt = run([
        "make", "-C", str(RUNTIME_ROOT / "cuda-engine"), "bkg4-invariant-edit-eval",
        "ARCH=sm_75", f"BUILD_DIR={BUILD_ROOT}", f"NVCC={nvcc}", f"B4EC_HOST_CXX={candidate}",
    ], check=False)
    if attempt.returncode == 0 and evaluator_candidate.is_file():
        make = attempt; host_cxx = candidate; break
    build_failures[candidate] = (attempt.stdout + attempt.stderr)[-2000:]
if make is None:
    raise RuntimeError(f"no installed host compiler could build the exact evaluator: {build_failures}")
EVALUATOR = BUILD_ROOT / "bkg4_invariant_edit_eval"
if not EVALUATOR.is_file() or "use_fast_math" in make.stdout + make.stderr:
    raise RuntimeError("invalid exact evaluator build")
cuobjdump = shutil.which("cuobjdump")
if cuobjdump is None:
    raise RuntimeError("cuobjdump is required to inspect the T4 cubin")
elf = run([cuobjdump, "--list-elf", EVALUATOR]).stdout + run([cuobjdump, "--dump-elf", EVALUATOR]).stdout
if "sm_75" not in elf or "sm_86" in elf:
    raise RuntimeError("compiled evaluator is not a native sm_75-only artifact")
print({"evaluator": str(EVALUATOR), "architecture": "sm_75", "fast_math": False, "nvcc": nvcc, "host_cxx": host_cxx})
'''),
        _code(r'''
# Architecture translation gate: exact fixture must match independently on GPU0 and GPU1.
RUNNER = [sys.executable, "-m", "tools.bkg4_search.kaggle_invariant_corpus_runner"]
smoke_processes = []
for physical_gpu in (0, 1):
    env = os.environ.copy(); env["PYTHONPATH"] = str(RUNTIME_ROOT); env["CUDA_VISIBLE_DEVICES"] = str(physical_gpu)
    command = RUNNER + ["run-smoke", "--support", str(RUNTIME_ROOT), "--binary", str(EVALUATOR), "--device", "0", "--output", str(OUTPUT_ROOT / f"smoke/gpu{physical_gpu}")]
    smoke_processes.append((physical_gpu, command, subprocess.Popen(command, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)))
smoke_reports = {}
for physical_gpu, command, process in smoke_processes:
    stdout, _ = process.communicate()
    print(stdout)
    if process.returncode:
        raise RuntimeError(f"GPU{physical_gpu} exact smoke failed")
    smoke_reports[str(physical_gpu)] = json.loads((OUTPUT_ROOT / f"smoke/gpu{physical_gpu}/smoke_report.json").read_text())
if not all(report["ok"] for report in smoke_reports.values()):
    raise RuntimeError("dual-T4 exact smoke gate failed")
'''),
        _code(r'''
# Concurrent representative candidate x 64 calibration on both T4s.
calibration_processes = []
for physical_gpu in (0, 1):
    env = os.environ.copy(); env["PYTHONPATH"] = str(RUNTIME_ROOT); env["CUDA_VISIBLE_DEVICES"] = str(physical_gpu)
    command = RUNNER + ["calibrate", "--support", str(RUNTIME_ROOT), "--binary", str(EVALUATOR), "--device", "0", "--output", str(OUTPUT_ROOT / f"calibration/gpu{physical_gpu}"), "--candidates", "1024", "--seed", str(BASE_SEED + physical_gpu * 100_003)]
    calibration_processes.append((physical_gpu, subprocess.Popen(command, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)))
calibrations = {}
for physical_gpu, process in calibration_processes:
    stdout, _ = process.communicate(); print(stdout)
    if process.returncode:
        raise RuntimeError(f"GPU{physical_gpu} calibration failed")
    calibrations[str(physical_gpu)] = json.loads((OUTPUT_ROOT / f"calibration/gpu{physical_gpu}/calibration.json").read_text())
print(json.dumps(calibrations, indent=2))
'''),
        _code(r'''
# Launch one long-lived worker per physical GPU with disjoint seeds and island directories.
workers = {}
for physical_gpu in (0, 1):
    calibration = calibrations[str(physical_gpu)]
    env = os.environ.copy(); env["PYTHONPATH"] = str(RUNTIME_ROOT); env["CUDA_VISIBLE_DEVICES"] = str(physical_gpu)
    command = RUNNER + [
        "worker", "--support", str(RUNTIME_ROOT), "--binary", str(EVALUATOR), "--output", str(OUTPUT_ROOT),
        "--island", str(physical_gpu), "--base-seed", str(BASE_SEED),
        "--generation-deadline-monotonic", str(GENERATION_DEADLINE),
        "--proposals", str(SHARD_PROPOSALS), "--retain", str(RETAINED_PER_SHARD),
        "--screen-cases", str(SCREEN_CASES), "--label-cases", str(LABEL_CASES),
        "--recursive", str(RECURSIVE), "--recursive-rounds", str(RECURSIVE_ROUNDS),
        "--repairs", str(CEGIS_REPAIRS), "--ablations", str(ABLATIONS),
        "--followup-retain", str(FOLLOWUP_RETAIN), "--tpb", str(calibration["chosen_tpb"]),
        "--memory-fraction", str(calibration["chosen_memory_fraction"]),
        "--disk-floor-bytes", str(DISK_RESERVE_BYTES), "--minimum-shard-seconds", str(MINIMUM_SHARD_SECONDS),
    ]
    log = (OUTPUT_ROOT / f"island{physical_gpu}/worker.log"); log.parent.mkdir(parents=True, exist_ok=True)
    handle = log.open("w", encoding="utf-8")
    process = subprocess.Popen(command, env=env, text=True, stdout=handle, stderr=subprocess.STDOUT)
    workers[physical_gpu] = {"process": process, "log_handle": handle, "command": command, "uuid": gpus[physical_gpu]["uuid"], "pid": process.pid}
print({gpu: {"pid": row["pid"], "uuid": row["uuid"], "seed_namespace": BASE_SEED + gpu * 1_000_000_000} for gpu, row in workers.items()})
'''),
        _code(r'''
# Low-frequency utilization/progress monitor and idle-GPU watchdog.
from IPython.display import clear_output

def progress_bar(fraction, width=30):
    fraction = min(1.0, max(0.0, fraction))
    filled = int(width * fraction)
    return "[" + "#" * filled + "-" * (width - filled) + "]"

utilization_samples = []
idle_counts = {0: 0, 1: 0}
while any(row["process"].poll() is None for row in workers.values()) and time.monotonic() < HARD_EXIT_DEADLINE:
    snapshot = run(["nvidia-smi", "--query-gpu=index,uuid,utilization.gpu,utilization.memory,memory.used", "--format=csv,noheader,nounits"], check=False).stdout
    by_gpu = {}
    for line in snapshot.splitlines():
        index, uuid, gpu_util, mem_util, mem_used = [part.strip() for part in line.split(",")]
        by_gpu[int(index)] = {"uuid": uuid, "gpu_util": int(gpu_util), "memory_util": int(mem_util), "memory_used_mib": int(mem_used)}
    sample = {"elapsed_seconds": time.monotonic() - PIPELINE_STARTED_MONOTONIC, "gpus": by_gpu, "workers": {}}
    for gpu, worker in workers.items():
        heartbeat = OUTPUT_ROOT / f"island{gpu}/heartbeat.json"
        sample["workers"][str(gpu)] = json.loads(heartbeat.read_text()) if heartbeat.is_file() else {"pid": worker["pid"], "warming_up": True}
        active = worker["process"].poll() is None
        idle_counts[gpu] = idle_counts[gpu] + 1 if active and by_gpu.get(gpu, {}).get("gpu_util", 0) < 5 else 0
        if idle_counts[gpu] >= IDLE_GPU_LIMIT_SAMPLES:
            raise RuntimeError(f"GPU{gpu} remained unexpectedly idle while its worker was active")
    utilization_samples.append(sample)
    generation_fraction = (time.monotonic() - PIPELINE_STARTED_MONOTONIC) / max(1, GENERATION_DEADLINE - PIPELINE_STARTED_MONOTONIC)
    clear_output(wait=True)
    print(f"Corpus generation {progress_bar(generation_fraction)} {min(100, generation_fraction * 100):5.1f}%")
    for gpu in (0, 1):
        heartbeat = sample["workers"][str(gpu)]
        totals = heartbeat.get("totals", {})
        print(
            f"GPU{gpu} {progress_bar(generation_fraction)} "
            f"shards={totals.get('shards', 0):4d} edits={totals.get('labeled', 0):8d} "
            f"edits/s={heartbeat.get('edits_per_second', 0.0):8.3f} "
            f"games={totals.get('exact_games', 0):12d} games/s={heartbeat.get('games_per_second', 0.0):10.1f} "
            f"util={by_gpu.get(gpu, {}).get('gpu_util', 0):3d}%"
        )
    if time.monotonic() >= GENERATION_DEADLINE and all(worker["process"].poll() is not None for worker in workers.values()):
        break
    time.sleep(min(MONITOR_INTERVAL_SECONDS, max(1, HARD_EXIT_DEADLINE - time.monotonic())))

# Workers normally stop themselves before the reserve. Interrupt only an overlong bounded shard.
for gpu, worker in workers.items():
    process = worker["process"]
    if process.poll() is None:
        process.send_signal(signal.SIGINT)
for gpu, worker in workers.items():
    process = worker["process"]
    try:
        process.wait(timeout=120)
    except subprocess.TimeoutExpired:
        process.terminate(); process.wait(timeout=30)
    worker["log_handle"].close()
if not list(OUTPUT_ROOT.glob("island*/shard-*/shard_manifest.json")):
    raise RuntimeError("no complete validated corpus shard survived production")
'''),
        _code(r'''
# Streaming global semantic dedup and leakage-free structural grouped split.
merge = run([sys.executable, "-m", "tools.bkg4_search.merge_invariant_corpus_shards", "--output", str(OUTPUT_ROOT)], env={**os.environ, "PYTHONPATH": str(RUNTIME_ROOT)})
merge_report = json.loads((OUTPUT_ROOT / "summary.json").read_text())
if merge_report["split"]["skeleton_overlap"] != 0:
    raise RuntimeError("structural leakage detected in final grouped split")

# Optional persistence is deliberately nonfatal when Kaggle credentials are unavailable.
publication = {"requested": PUBLISH_CORPUS, "published": False}
if PUBLISH_CORPUS:
    metadata = {"title": "Kaggriculture B4IM Elite Specialist Corpus", "id": CORPUS_DATASET, "licenses": [{"name": "CC0-1.0"}], "subtitle": "Exact-CUDA specialist corpus for BonsaiEditFormer training"}
    (OUTPUT_ROOT / "dataset-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    listing = run(["kaggle", "datasets", "files", CORPUS_DATASET], check=False)
    if listing.returncode == 0:
        publish = run(["kaggle", "datasets", "version", "-p", str(OUTPUT_ROOT), "-m", f"B4IM exact specialist corpus {support_manifest['implementation_commit']}", "-q", "-r", "skip"], check=False)
    else:
        publish = run(["kaggle", "datasets", "create", "-p", str(OUTPUT_ROOT), "-q", "-r", "skip"], check=False)
    publication = {"requested": True, "published": publish.returncode == 0, "message": (publish.stdout + publish.stderr)[-2000:]}
'''),
        _code(r'''
# Final production report.
worker_reports = {}
shard_manifests = []
for gpu in (0, 1):
    report_path = OUTPUT_ROOT / f"island{gpu}/worker_report.json"
    worker_reports[str(gpu)] = json.loads(report_path.read_text()) if report_path.is_file() else {"interrupted_after_complete_shards": True}
for path in sorted(OUTPUT_ROOT.glob("island*/shard-*/shard_manifest.json")):
    shard_manifests.append(json.loads(path.read_text()))

def sum_counter(name):
    total = {}
    for manifest in shard_manifests:
        for key, value in manifest["counts"].get(name, {}).items():
            total[key] = total.get(key, 0) + int(value)
    return dict(sorted(total.items()))

average_gpu_util = {}
for gpu in (0, 1):
    values = [sample["gpus"].get(gpu, {}).get("gpu_util", 0) for sample in utilization_samples]
    average_gpu_util[str(gpu)] = sum(values) / max(1, len(values))
total_bytes = sum(path.stat().st_size for path in OUTPUT_ROOT.rglob("*") if path.is_file())
metadata_bytes = sum(path.stat().st_size for path in OUTPUT_ROOT.glob("island*/shard-*/metadata.jsonl.gz"))
npz_bytes = sum(path.stat().st_size for path in OUTPUT_ROOT.glob("island*/shard-*/training_arrays.npz"))
exact_games = sum(int(manifest["counters"].get("exact_games", 0)) for manifest in shard_manifests)
raw_proposals = sum(int(manifest["counters"].get("raw_proposals", 0)) for manifest in shard_manifests)
labeled = sum(int(manifest["counts"]["rows"]) for manifest in shard_manifests)
wall = time.monotonic() - PIPELINE_STARTED_MONOTONIC
final_report = {
    "hardware": {"gpus": gpus[:2], "calibration": calibrations, "average_gpu_utilization": average_gpu_util},
    "support": {"dataset": SUPPORT_DATASET, "package_version": support_manifest["support_package_version"], "source_commit": support_manifest["source_commit"], "implementation_commit": support_manifest["implementation_commit"], "parent": support_manifest["parent"]["canonical_identity"], "candidate_abi": "B4IM1/v1", "result_abi": "B4IC1/v1", "bkg4_abi": 4},
    "corpus": {"total_raw_proposals": raw_proposals, "unique_screened_candidates": raw_proposals, "fully_labeled_candidates": labeled, "final_deduplicated_training_rows": merge_report["deduplicated_rows"], "shard_count": merge_report["shards"], "exact_cuda_games": exact_games, "fault_count": 0},
    "diversity": {"skeletons": merge_report["skeletons"], "qd_niches": sum(int(manifest["counts"]["qd_niches"]) for manifest in shard_manifests), "depth_distribution": sum_counter("depths"), "tier_distribution": merge_report["tiers"]},
    "quality": {name: merge_report["tiers"].get(name, 0) for name in ("GLOBAL_SAFE", "ELITE_SPECIALIST", "STRONG_SPECIALIST", "BOUNDARY_SPECIALIST", "HARD_NEGATIVE", "INERT", "L2W")},
    "search": {"operators": sum_counter("operators"), "source_goals": sum_counter("source_goals")},
    "performance": {"per_gpu_kernel_games_per_second": {gpu: row["kernel_games_per_second"] for gpu, row in calibrations.items()}, "wall_clock_seconds": wall, "screened_candidates_per_hour": raw_proposals / max(wall, 1) * 3600, "labeled_candidates_per_hour": labeled / max(wall, 1) * 3600, "average_gpu_utilization": average_gpu_util},
    "storage": {"compressed_metadata_bytes": metadata_bytes, "npz_bytes": npz_bytes, "total_output_bytes": total_bytes},
    "split": merge_report["split"], "smoke": smoke_reports, "publication": publication,
}
(OUTPUT_ROOT / "manifest.json").write_text(json.dumps(final_report, indent=2, sort_keys=True) + "\n")
print(json.dumps(final_report, indent=2, sort_keys=True))
'''),
    ]
    notebook = {
        "cells": cells,
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3"}},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    return json.dumps(notebook, indent=1, sort_keys=True, ensure_ascii=False) + "\n"


def render_kernel_metadata() -> str:
    return json.dumps({
        "id": "ashok205/kaggriculture-b4im-elite-corpus-dual-t4",
        "title": "Kaggriculture B4IM Elite Specialist Corpus Dual T4",
        "code_file": OUT_NOTEBOOK.name,
        "language": "python", "kernel_type": "notebook", "is_private": True,
        "enable_gpu": True, "enable_tpu": False, "enable_internet": True,
        "dataset_sources": [], "kernel_sources": [], "competition_sources": [], "model_sources": [],
    }, indent=2, sort_keys=True) + "\n"


def validate_notebook(text: str) -> None:
    notebook = json.loads(text)
    if notebook.get("nbformat") != 4:
        raise ValueError("invalid notebook format")
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"<b4im-notebook-cell-{index}>", "exec")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    notebook = render_notebook()
    metadata = render_kernel_metadata()
    validate_notebook(notebook)
    if args.check:
        if not OUT_NOTEBOOK.is_file() or OUT_NOTEBOOK.read_text(encoding="utf-8") != notebook:
            raise SystemExit(f"generated notebook is stale: {OUT_NOTEBOOK}")
        if not OUT_META.is_file() or OUT_META.read_text(encoding="utf-8") != metadata:
            raise SystemExit(f"generated kernel metadata is stale: {OUT_META}")
        print("B4IM dual-T4 notebook generation check passed")
        return 0
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_NOTEBOOK.write_text(notebook, encoding="utf-8")
    OUT_META.write_text(metadata, encoding="utf-8")
    print(OUT_NOTEBOOK)
    print(OUT_META)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
