#!/usr/bin/env python3
"""Generate the Kaggle-ready GPU-resident dual-T4 memetic meta-search notebook."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "notebooks/kaggle_dual_t4_meta_search.ipynb"


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": [x + "\n" for x in text.strip().splitlines()]}


def code(text: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [x + "\n" for x in text.strip().splitlines()]}


cells = [
md("""
# Kaggriculture Dual-T4 GPU-Resident Memetic Meta Search

Two independent exact-CUDA Bonsai search universes. GPU0 is exploitation-biased; GPU1 is exploration-biased. The expensive mutation → simulation → family scoring → QD selection → stagnation/reset → local-refinement loop runs inside compiled CUDA/C++, with sparse host supervision only. Frozen validation, blind, and family-holdout data are excluded from the attached development bundle.
"""),
code(r'''
# CELL 1 — absolute pipeline deadline and central configuration
from pathlib import Path
import hashlib, json, os, re, shutil, subprocess, tarfile, time

NOTEBOOK_STARTED = time.monotonic()
TOTAL_WALLTIME_SECONDS = 10 * 60 * 60
HARD_EXIT_DEADLINE = NOTEBOOK_STARTED + TOTAL_WALLTIME_SECONDS
FINAL_RESERVE_SECONDS = 25 * 60
ORCHESTRATION_RESERVE_SECONDS = 15 * 60
VRAM_TARGET = 0.95
VRAM_RESERVE_MIB = 512
MONITOR_INTERVAL_SECONDS = 5 * 60
FINAL_SCREEN_MAX_CANDIDATES = 256
ISLAND0_SEED = 340034
ISLAND1_SEED = 9340034
RESUME = False
RESUME_DATASET_MOUNT = None
REQUIRE_T4 = True

WORK_ROOT = Path("/kaggle/working/meta_search_runtime")
OUTPUT_ROOT = Path("/kaggle/working/meta_search")
BUILD_ROOT = Path("/kaggle/working/meta_search_build")

assert FINAL_RESERVE_SECONDS + ORCHESTRATION_RESERVE_SECONDS < TOTAL_WALLTIME_SECONDS
assert 0.90 <= VRAM_TARGET <= 0.97
print("hard 10-hour deadline initialized")
'''),
code(r'''
# CELL 2 — hardware/environment probe

def run(cmd, *, env=None, cwd=None, check=True, capture=True):
    print("$", " ".join(map(str, cmd)))
    return subprocess.run([str(x) for x in cmd], env=env, cwd=cwd, check=check,
                          text=True, capture_output=capture)

print(run(["nvidia-smi"]).stdout)
print(run(["nvcc", "--version"]).stdout)
q = run(["nvidia-smi", "--query-gpu=index,uuid,name,memory.total,memory.free,compute_cap", "--format=csv,noheader,nounits"]).stdout.strip().splitlines()
gpus = []
for line in q:
    idx, uuid, name, total, free, cap = [x.strip() for x in line.split(",")]
    gpus.append({"index": int(idx), "uuid": uuid, "name": name, "memory_total_mib": int(total),
                 "memory_free_mib": int(free), "compute_capability": cap})
assert len(gpus) >= 2, f"need two GPUs, detected {len(gpus)}"
if REQUIRE_T4:
    assert all("T4" in g["name"] for g in gpus[:2]), f"expected 2xT4, got {[g['name'] for g in gpus[:2]]}"
assert gpus[0]["compute_capability"] == gpus[1]["compute_capability"], "two search GPUs must share one compiled architecture"
if OUTPUT_ROOT.exists() and not RESUME: shutil.rmtree(OUTPUT_ROOT)
BUILD_ROOT.mkdir(parents=True, exist_ok=True)
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
(OUTPUT_ROOT / "hardware.json").write_text(json.dumps({"gpus": gpus[:2], "nvidia_smi": run(["nvidia-smi"]).stdout}, indent=2) + "\n")
print(json.dumps(gpus[:2], indent=2))
'''),
code(r'''
# CELL 3 — robustly locate/copy the latest private development-only Dataset
candidates = [
    Path("/kaggle/input/datasets/ashok205/kaggriculture-meta-search"),
    Path("/kaggle/input/kaggriculture-meta-search"),
]
input_root = next((p for p in candidates if (p / "bundle_manifest.json").exists()), None)
if input_root is None:
    matches = list(Path("/kaggle/input").rglob("bundle_manifest.json"))
    assert len(matches) == 1, f"cannot uniquely locate bundle_manifest.json: {matches}"
    input_root = matches[0].parent
print("input bundle:", input_root)

if WORK_ROOT.exists() and not RESUME: shutil.rmtree(WORK_ROOT)
if not WORK_ROOT.exists(): shutil.copytree(input_root, WORK_ROOT)
manifest = json.loads((WORK_ROOT / "bundle_manifest.json").read_text())
assert manifest["dev_tapes"] == 168 and manifest["families"] == 21
assert manifest["validation_excluded"] == 84
assert manifest["blind_excluded"] == 50
assert manifest["family_holdout_excluded"] == 28
assert manifest["fast_math"] is False
assert manifest["resident_search_genome_bytes"] == 256
assert manifest["engine_exploitation_search"] is True
assert manifest["engine_sub_islands"] == {"exploit": 12, "explore": 20, "total_default": 32}
engine_source = WORK_ROOT / manifest["official_engine_path"]
assert engine_source.exists(), f"official engine source missing from bundle: {engine_source}"
assert hashlib.sha256(engine_source.read_bytes()).hexdigest() == manifest["official_engine_sha256"], "official Kaggriculture engine source hash mismatch"
assert "Kawashigi" in manifest["stage1_families"] and "Ueddy" in manifest["stage1_families"]

if RESUME and RESUME_DATASET_MOUNT:
    rr = Path("/kaggle/input") / RESUME_DATASET_MOUNT / "meta_search"
    if not rr.exists():
        hits = list(Path("/kaggle/input").rglob(f"{RESUME_DATASET_MOUNT}/meta_search"))
        assert len(hits) == 1, f"resume root not found: {RESUME_DATASET_MOUNT}"
        rr = hits[0]
    for island in ("island0", "island1"):
        src = rr / island
        if src.exists(): shutil.copytree(src, OUTPUT_ROOT / island, dirs_exist_ok=True)

cfg = json.loads((WORK_ROOT / "tools/meta_search/default_config.json").read_text())
cfg.update({
    "total_walltime_seconds": TOTAL_WALLTIME_SECONDS,
    "final_reserve_seconds": FINAL_RESERVE_SECONDS,
    "orchestration_reserve_seconds": ORCHESTRATION_RESERVE_SECONDS,
    "island0_seed": ISLAND0_SEED,
    "island1_seed": ISLAND1_SEED,
    "vram_target_fraction": VRAM_TARGET,
    "vram_reserve_mib": VRAM_RESERVE_MIB,
    "final_screen_max_candidates": FINAL_SCREEN_MAX_CANDIDATES,
    "monitor_interval_seconds": MONITOR_INTERVAL_SECONDS,
})
(OUTPUT_ROOT / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
print(json.dumps(manifest, indent=2, ensure_ascii=False))
'''),
code(r'''
# CELL 4 — compile the resident search engine, independent exact evaluator, and BKG3-v6 smoke
cc = gpus[0]["compute_capability"].replace(".", "")
arch = [f"-gencode=arch=compute_{cc},code=sm_{cc}"]
common = ["nvcc", "-O3", "-std=c++17", "-lineinfo", "-Xptxas=-v", *arch]
assert all("fast_math" not in x and "fast-math" not in x for x in common)

resident_bin = BUILD_ROOT / "bonsai_memetic_search"
evaluator_bin = BUILD_ROOT / "bonsai_search"
v6_smoke_bin = BUILD_ROOT / "stateful_v6_game_cuda"
run([*common, "-I", str(WORK_ROOT / "cuda-engine/include"), "-o", str(resident_bin),
     str(WORK_ROOT / "cuda-engine/src/bonsai_memetic_search.cu")], capture=False)
run([*common, "-I", str(WORK_ROOT / "cuda-engine/include"), "-o", str(evaluator_bin),
     str(WORK_ROOT / "cuda-engine/src/bonsai_search.cu")], capture=False)
run([*common, "-o", str(v6_smoke_bin), str(WORK_ROOT / "bonsai-kg/tests/stateful_v6_game_cuda.cu")], capture=False)
assert resident_bin.exists() and evaluator_bin.exists() and v6_smoke_bin.exists()
print("compiled resident/evaluator/smoke binaries")
'''),
code(r'''
# CELL 5 — exact semantics gates: BKG3-v6 fingerprint + SearchGenome→GPProgram parity
ref = json.loads((WORK_ROOT / "tools/meta_search/v6_smoke_reference.json").read_text())
cmd = [str(v6_smoke_bin), str(WORK_ROOT / ref["controller"]), str(WORK_ROOT / ref["route_pack"]),
       str(ref["seed_start"]), str(ref["count"]), str(ref["seat"]), str(ref["steps"])]
p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
digest = hashlib.sha256(p.stdout).hexdigest(); stderr_lines = p.stderr.decode().strip().splitlines()
assert digest == ref["stdout_sha256"], "ABORT: BKG3-v6 CUDA fingerprint mismatch"
assert stderr_lines == ref["expected_stderr_lines"], "ABORT: BKG3-v6 tick/fault smoke mismatch"
print("EXACT BKG3-v6 CUDA SMOKE PASSED", digest)

prior = WORK_ROOT / "priors" / manifest["search_priors"][0]
env = os.environ.copy(); env["CUDA_VISIBLE_DEVICES"] = "0"
parity_cmd = [str(resident_bin), "--bundle", str(WORK_ROOT), "--output", str(OUTPUT_ROOT / "parity"),
              "--sub-islands", "8", "--parity-check", "--seed-program", str(prior)]
p = subprocess.run(parity_cmd, env=env, text=True, capture_output=True)
print(p.stdout)
if p.returncode != 0:
    print(p.stderr); raise RuntimeError("SearchGenome distillation parity failed")
assert "engine_mode=1" in p.stdout and "mismatches=0" in p.stdout
print("ENGINE-MODE SEARCH-GENOME DISTILLATION PARITY PASSED")
'''),
code(r'''
# CELL 6 — independent exact resident-engine autotuning on GPU0/GPU1
for gpu in (0, 1):
    env = os.environ.copy(); env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    out = OUTPUT_ROOT / f"island{gpu}"; out.mkdir(parents=True, exist_ok=True)
    tuning = out / "tuning.json"
    mode = "exploit" if gpu == 0 else "explore"
    mode_cfg = cfg[mode]
    cmd = ["python3", str(WORK_ROOT / "tools/meta_search/autotune.py"),
           "--binary", str(resident_bin), "--bundle", str(WORK_ROOT), "--output", str(tuning),
           "--populations", ",".join(map(str, cfg["autotune_populations"])),
           "--tpb", ",".join(map(str, cfg["autotune_tpb"])),
           "--sub-islands", str(cfg["default_sub_islands"]),
           "--engine-sub-islands", str(mode_cfg["engine_sub_islands"]),
           "--universe", mode,
           "--vram-target", str(VRAM_TARGET), "--reserve-mib", str(VRAM_RESERVE_MIB),
           "--target-generation-seconds", str(cfg["autotune_target_generation_seconds"]),
           "--seed", str(770001 + gpu * 10000)]
    p = subprocess.run(cmd, env=env, cwd=WORK_ROOT, text=True, capture_output=True)
    print(p.stdout[-8000:])
    if p.returncode != 0:
        print(p.stderr); raise RuntimeError(f"GPU{gpu} resident autotune failed")
print("both resident T4 autotuners passed")
'''),
code(r'''
# CELL 7 — compute the remaining search budget from one monotonic absolute deadline
now = time.monotonic()
remaining = HARD_EXIT_DEADLINE - now
search_seconds = remaining - FINAL_RESERVE_SECONDS - ORCHESTRATION_RESERVE_SECONDS
assert search_seconds >= 60 * 60, f"only {search_seconds/60:.1f} minutes remain for search; restart the notebook cleanly"
MERGE_START_DEADLINE = HARD_EXIT_DEADLINE - FINAL_RESERVE_SECONDS
SEARCH_STOP_DEADLINE = MERGE_START_DEADLINE - ORCHESTRATION_RESERVE_SECONDS
search_seconds = max(0.0, SEARCH_STOP_DEADLINE - time.monotonic())
print(f"setup_elapsed={(time.monotonic()-NOTEBOOK_STARTED)/60:.1f}m search_budget={search_seconds/3600:.2f}h "
      f"shutdown_reserve={ORCHESTRATION_RESERVE_SECONDS/60:.0f}m final_reserve={FINAL_RESERVE_SECONDS/60:.0f}m")
'''),
code(r'''
# CELL 8 — launch exactly one long-lived compiled search universe per T4
campaigns = {}

def launch_island(gpu: int, *, resume: bool, walltime: float):
    env = os.environ.copy(); env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    out = OUTPUT_ROOT / f"island{gpu}"; out.mkdir(parents=True, exist_ok=True)
    console = (out / "campaign_console.log").open("a" if resume else "w")
    cmd = ["python3", str(WORK_ROOT / "tools/meta_search/island_campaign.py"),
           "--binary", str(resident_bin), "--bundle", str(WORK_ROOT),
           "--config", str(OUTPUT_ROOT / "config.json"), "--tuning", str(out / "tuning.json"),
           "--output", str(out), "--island", str(gpu), "--walltime-seconds", str(max(60.0, walltime))]
    if resume: cmd.append("--resume")
    proc = subprocess.Popen(cmd, env=env, cwd=WORK_ROOT, stdout=console, stderr=subprocess.STDOUT, start_new_session=True)
    return {"proc": proc, "console": console, "env": env, "cmd": cmd, "restarts": 0}

for gpu in (0, 1):
    campaigns[gpu] = launch_island(gpu, resume=RESUME, walltime=search_seconds)
    print(f"launched GPU{gpu} universe pid={campaigns[gpu]['proc'].pid}")
'''),
code(r'''
# CELL 9 — warm-up gate: both physical T4s active and resident allocation near safe target
expected_uuids = {g["uuid"] for g in gpus[:2]}
mem_re = re.compile(r"allocated_fraction_of_safe=([0-9.eE+-]+)")
allocation_seen = {}
for _ in range(36):
    for gpu, item in campaigns.items():
        rc = item["proc"].poll()
        assert rc is None, f"GPU{gpu} universe exited early rc={rc}; inspect campaign_console.log/resident_search.log"
        log = OUTPUT_ROOT / f"island{gpu}/resident_search.log"
        if log.exists():
            m = mem_re.search(log.read_text(errors="replace"))
            if m: allocation_seen[gpu] = float(m.group(1))
    apps = run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory", "--format=csv,noheader"]).stdout
    active = {line.split(",", 1)[0].strip() for line in apps.splitlines() if line.strip()}
    if expected_uuids.issubset(active) and len(allocation_seen) == 2: break
    time.sleep(5)
print("safe allocation fractions:", allocation_seen)
assert len(allocation_seen) == 2, "resident memory plan did not become observable on both GPUs"
assert all(0.90 <= x <= 0.99 for x in allocation_seen.values()), f"resident allocation missed safe target: {allocation_seen}"

# Prove the processes are computing, not merely holding VRAM while CPU orchestration spins.
util_samples = {0: [], 1: []}
for _ in range(6):
    rows = run(["nvidia-smi", "--query-gpu=index,utilization.gpu", "--format=csv,noheader,nounits"]).stdout.splitlines()
    for row in rows[:2]:
        gi, util = [x.strip() for x in row.split(",")]
        if int(gi) in util_samples: util_samples[int(gi)].append(int(float(util)))
    time.sleep(3)
print("warmup GPU utilization samples:", util_samples)
for gpu, vals in util_samples.items():
    assert vals and max(vals) >= 50 and sum(vals)/len(vals) >= 20, f"GPU{gpu} is not sustaining CUDA work: {vals}"
print("BOTH T4 RESIDENT UNIVERSES ACTIVE AND COMPUTE-BOUND")
'''),
code(r'''
# CELL 10 — sparse health/utilization monitor; no generation-level Python orchestration
samples = []
last_generation = {0: -1, 1: -1}
while time.monotonic() < SEARCH_STOP_DEADLINE:
    now = time.monotonic()
    alive = 0
    for gpu, item in list(campaigns.items()):
        rc = item["proc"].poll()
        if rc is None:
            alive += 1; continue
        if rc != 0 and item["restarts"] < 1 and now + 30 * 60 < SEARCH_STOP_DEADLINE:
            item["console"].close()
            rem = SEARCH_STOP_DEADLINE - now
            replacement = launch_island(gpu, resume=True, walltime=rem)
            replacement["restarts"] = item["restarts"] + 1
            campaigns[gpu] = replacement; alive += 1
            print(f"GPU{gpu} restarted once from compact checkpoint")
    if alive == 0: break

    telemetry = run(["nvidia-smi",
        "--query-gpu=index,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu",
        "--format=csv,noheader,nounits"]).stdout.strip()
    sample = {"elapsed_seconds": now - NOTEBOOK_STARTED, "telemetry": telemetry, "islands": {}}
    for gpu in (0, 1):
        cp = OUTPUT_ROOT / f"island{gpu}/checkpoint.json"
        if cp.exists():
            x = json.loads(cp.read_text()); sample["islands"][str(gpu)] = x
            last_generation[gpu] = int(x.get("generation", last_generation[gpu]))
    samples.append(sample)
    (OUTPUT_ROOT / "utilization_samples.jsonl").write_text("".join(json.dumps(x) + "\n" for x in samples))
    print(f"elapsed={(now-NOTEBOOK_STARTED)/3600:.2f}h generations={last_generation}\n{telemetry}")
    time.sleep(min(MONITOR_INTERVAL_SECONDS, max(1, SEARCH_STOP_DEADLINE - time.monotonic())))
'''),
code(r'''
# CELL 11 — graceful stop guard; resident engines checkpoint themselves before final merge
for gpu, item in campaigns.items():
    if item["proc"].poll() is None:
        print(f"requesting graceful stop GPU{gpu}")
        item["proc"].terminate()
shutdown_deadline = MERGE_START_DEADLINE
for gpu, item in campaigns.items():
    proc = item["proc"]
    while proc.poll() is None and time.monotonic() < shutdown_deadline:
        time.sleep(5)
    if proc.poll() is None:
        print(f"GPU{gpu} exceeded graceful shutdown reserve; forcing termination")
        proc.kill(); proc.wait()
    item["console"].close()
    print(f"GPU{gpu} supervisor rc={proc.returncode}")
assert time.monotonic() < HARD_EXIT_DEADLINE - 5 * 60, "insufficient hard-deadline reserve for final screen"
'''),
code(r'''
# CELL 12 — canonical dedup inventory from sub-island, QD-niche and certified archives
bins = []
for pat in ("island*/subisland_*_best.bin", "island*/niche_*_best.bin", "island*/certified_niche_*_L*.bin"):
    bins.extend(OUTPUT_ROOT.glob(pat))
hashes = {}
for p in bins:
    if p.stat().st_size != 4096: continue
    hashes.setdefault(hashlib.sha256(p.read_bytes()).hexdigest(), []).append(str(p))
print(f"resident GPProgram exports={len(bins)} canonical_unique={len(hashes)}")
assert hashes, "no resident elite programs available for final merge"
'''),
code(r'''
# CELL 13 — final common 168-tape × both-seat exact screen, sharded across BOTH T4s
merged = OUTPUT_ROOT / "merged"; merged.mkdir(parents=True, exist_ok=True)
cmd = ["python3", str(WORK_ROOT / "tools/meta_search/merge_elites.py"),
       "--binary", str(evaluator_bin), "--bundle", str(WORK_ROOT), "--search-root", str(OUTPUT_ROOT),
       "--output", str(merged), "--max-candidates", str(FINAL_SCREEN_MAX_CANDIDATES),
       "--seed", str(cfg["final_screen_seed"]), "--gpus", "0,1",
       "--tuning0", str(OUTPUT_ROOT / "island0/tuning.json"), "--tuning1", str(OUTPUT_ROOT / "island1/tuning.json")]
p = subprocess.run(cmd, cwd=WORK_ROOT, text=True, capture_output=True)
print(p.stdout)
if p.returncode != 0:
    print(p.stderr); raise RuntimeError("dual-GPU final common development screen failed")
assert time.monotonic() < HARD_EXIT_DEADLINE, "hard 10-hour deadline exceeded during final screen"
'''),
code(r'''
# CELL 14 — utilization/search report, artifact export, normal auto-exit
report = {"hardware": gpus[:2], "islands": {}, "merged": json.loads((OUTPUT_ROOT / "merged/summary.json").read_text())}
for gpu in (0, 1):
    t = json.loads((OUTPUT_ROOT / f"island{gpu}/tuning.json").read_text())
    s = OUTPUT_ROOT / f"island{gpu}/resident_summary.json"
    report["islands"][str(gpu)] = {"tuning": t, "search": json.loads(s.read_text()) if s.exists() else None}
report["total_pipeline_seconds"] = time.monotonic() - NOTEBOOK_STARTED
(OUTPUT_ROOT / "final_report.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2)[:12000])

archive = Path("/kaggle/working/kaggriculture_dual_t4_meta_search_results.tar.gz")
with tarfile.open(archive, "w:gz") as tf: tf.add(OUTPUT_ROOT, arcname="meta_search")
print("RESULT_ARCHIVE", archive)
print(f"TOTAL_PIPELINE_HOURS={(time.monotonic()-NOTEBOOK_STARTED)/3600:.3f}")
print("AUTOEXIT: search, dual-GPU final screen and artifact export complete")
'''),
]

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3"},
        "kaggle": {"accelerator": "gpu"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(nb, indent=1) + "\n", encoding="utf-8")
print(OUT)
