#!/usr/bin/env python3
"""Generate the Kaggle-ready dual-T4 meta-search notebook."""
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
# Kaggriculture Dual-T4 Bonsai Meta Search

Batch research notebook for two independent exact-CUDA Bonsai search islands. GPU 0 is exploitation-biased; GPU 1 is exploration-biased. No validation/blind/family-holdout tapes are present in the input bundle. The notebook aborts before the long run if the exact BKG3-v6 CUDA smoke fingerprint fails.
"""),
code(r'''
# CELL 1 — central configuration
from pathlib import Path
import hashlib, json, os, shutil, subprocess, tarfile, time

NOTEBOOK_STARTED = time.monotonic()
INPUT_DATASET_MOUNT = "kaggriculture-meta-search"
WORK_ROOT = Path("/kaggle/working/meta_search_runtime")
OUTPUT_ROOT = Path("/kaggle/working/meta_search")
BUILD_ROOT = Path("/kaggle/working/meta_search_build")

TOTAL_WALLTIME_SECONDS = 10 * 60 * 60
SEARCH_WALLTIME_SECONDS = 9 * 60 * 60 + 20 * 60
ISLAND0_SEED = 340034
ISLAND1_SEED = 9340034
VRAM_TARGET_FRACTION = 0.93
VRAM_FALLBACK_FRACTIONS = [0.90, 0.86, 0.80]
CHECKPOINT_INTERVAL_GENERATIONS = 1
LOG_INTERVAL_SECONDS = 300
FINAL_SCREEN_MAX_CANDIDATES = 256
FINAL_RESERVE_SECONDS = 30 * 60
ORCHESTRATION_RESERVE_SECONDS = 10 * 60
RESUME = False
RESUME_DATASET_MOUNT = None  # e.g. "previous-kaggriculture-meta-search-results" for a fresh-session resume
REQUIRE_T4 = True

assert 0.0 < VRAM_TARGET_FRACTION < 0.98
assert SEARCH_WALLTIME_SECONDS < TOTAL_WALLTIME_SECONDS
print("configuration loaded")
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
    gpus.append({"index": int(idx), "uuid": uuid, "name": name, "memory_total_mib": int(total), "memory_free_mib": int(free), "compute_capability": cap})
assert len(gpus) >= 2, f"need two GPUs, detected {len(gpus)}"
if REQUIRE_T4:
    assert all("T4" in g["name"] for g in gpus[:2]), f"expected 2xT4, got {[g['name'] for g in gpus[:2]]}"
if OUTPUT_ROOT.exists() and not RESUME:
    shutil.rmtree(OUTPUT_ROOT)
BUILD_ROOT.mkdir(parents=True, exist_ok=True)
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
(OUTPUT_ROOT / "hardware.json").write_text(json.dumps({"gpus": gpus, "nvidia_smi": run(["nvidia-smi"]).stdout}, indent=2) + "\n")
print(json.dumps(gpus[:2], indent=2))
'''),
code(r'''
# CELL 3 — locate/copy the read-only input bundle and lock development-only config
input_root = Path("/kaggle/input") / INPUT_DATASET_MOUNT
if not (input_root / "bundle_manifest.json").exists():
    matches = list(Path("/kaggle/input").glob("*/bundle_manifest.json"))
    assert len(matches) == 1, f"cannot uniquely locate bundle_manifest.json: {matches}"
    input_root = matches[0].parent
print("input bundle:", input_root)

if WORK_ROOT.exists() and not RESUME:
    shutil.rmtree(WORK_ROOT)
if not WORK_ROOT.exists():
    shutil.copytree(input_root, WORK_ROOT)
manifest = json.loads((WORK_ROOT / "bundle_manifest.json").read_text())
if RESUME and RESUME_DATASET_MOUNT:
    resume_root = Path("/kaggle/input") / RESUME_DATASET_MOUNT / "meta_search"
    assert resume_root.exists(), f"resume tree not found: {resume_root}"
    for island in ("island0", "island1"):
        src = resume_root / island
        if src.exists(): shutil.copytree(src, OUTPUT_ROOT / island, dirs_exist_ok=True)
    print("restored island checkpoints from", resume_root)
assert manifest["dev_tapes"] == 168 and manifest["families"] == 21
assert manifest["validation_excluded"] == 84
assert manifest["blind_excluded"] == 50
assert manifest["family_holdout_excluded"] == 28
assert manifest["fast_math"] is False

cfg = json.loads((WORK_ROOT / "tools/meta_search/default_config.json").read_text())
cfg.update({
    "total_walltime_seconds": TOTAL_WALLTIME_SECONDS,
    "search_walltime_seconds": SEARCH_WALLTIME_SECONDS,
    "island0_seed": ISLAND0_SEED,
    "island1_seed": ISLAND1_SEED,
    "vram_target_fraction": VRAM_TARGET_FRACTION,
    "vram_fallback_fractions": VRAM_FALLBACK_FRACTIONS,
    "archive_interval_generations": CHECKPOINT_INTERVAL_GENERATIONS,
    "final_screen_max_candidates": FINAL_SCREEN_MAX_CANDIDATES,
})
(OUTPUT_ROOT / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
print(json.dumps(manifest, indent=2))
'''),
code(r'''
# CELL 4 — compile exact CUDA search + BKG3-v6 smoke runner
caps = sorted({g["compute_capability"].replace(".", "") for g in gpus[:2]})
arch_flags = []
for cc in caps:
    arch_flags += [f"-gencode=arch=compute_{cc},code=sm_{cc}"]
common = ["nvcc", "-O3", "-std=c++17", "-lineinfo", "--compiler-options=-O3,-Wall", *arch_flags]
assert all("fast_math" not in x and "fast-math" not in x for x in common)

search_bin = BUILD_ROOT / "bonsai_search"
v6_smoke_bin = BUILD_ROOT / "stateful_v6_game_cuda"
run([*common, "-I", str(WORK_ROOT / "cuda-engine/include"),
     "-o", str(search_bin), str(WORK_ROOT / "cuda-engine/src/bonsai_search.cu")], capture=False)
run([*common, "-o", str(v6_smoke_bin), str(WORK_ROOT / "bonsai-kg/tests/stateful_v6_game_cuda.cu")], capture=False)
assert search_bin.exists() and v6_smoke_bin.exists()
print("compiled:", search_bin, v6_smoke_bin)
'''),
code(r'''
# CELL 5 — exact BKG3-v6 CUDA smoke gate; abort on any mismatch
ref = json.loads((WORK_ROOT / "tools/meta_search/v6_smoke_reference.json").read_text())
cmd = [str(v6_smoke_bin), str(WORK_ROOT / ref["controller"]), str(WORK_ROOT / ref["route_pack"]),
       str(ref["seed_start"]), str(ref["count"]), str(ref["seat"]), str(ref["steps"])]
p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
digest = hashlib.sha256(p.stdout).hexdigest()
stderr_lines = p.stderr.decode().strip().splitlines()
print("BKG3-v6 smoke sha256:", digest)
print("\n".join(stderr_lines))
assert digest == ref["stdout_sha256"], "ABORT: exact BKG3-v6 CUDA smoke fingerprint mismatch"
assert stderr_lines == ref["expected_stderr_lines"], "ABORT: BKG3-v6 fault/tick smoke mismatch"
print("EXACT BKG3-v6 CUDA SMOKE PASSED")
'''),
code(r'''
# CELL 6 — independently auto-tune GPU0 and GPU1
for gpu in (0, 1):
    env = os.environ.copy(); env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    tuning = OUTPUT_ROOT / f"island{gpu}" / "tuning.json"
    tuning.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["python3", str(WORK_ROOT / "tools/meta_search/autotune.py"),
           "--binary", str(search_bin), "--bundle", str(WORK_ROOT), "--output", str(tuning),
           "--populations", "512,2048,8192,32768", "--seed", str(770001 + gpu * 10000)]
    p = subprocess.run(cmd, env=env, cwd=WORK_ROOT, text=True, capture_output=True)
    print(p.stdout)
    if p.returncode != 0:
        print(p.stderr)
        raise RuntimeError(f"GPU{gpu} autotune failed")
print("per-GPU autotuning complete")
'''),
code(r'''
# CELL 7 — initialize isolated search directories
for gpu in (0, 1):
    d = OUTPUT_ROOT / f"island{gpu}"
    d.mkdir(parents=True, exist_ok=True)
(OUTPUT_ROOT / "merged").mkdir(parents=True, exist_ok=True)
print("output root:", OUTPUT_ROOT)
'''),
code(r'''
# CELL 8 — launch BOTH independent islands non-blocking
setup_elapsed = time.monotonic() - NOTEBOOK_STARTED
remaining = TOTAL_WALLTIME_SECONDS - setup_elapsed
assert remaining > FINAL_RESERVE_SECONDS + ORCHESTRATION_RESERVE_SECONDS + 30 * 60, "insufficient notebook walltime remains for a safe long search"
effective_search = min(SEARCH_WALLTIME_SECONDS, remaining - FINAL_RESERVE_SECONDS - ORCHESTRATION_RESERVE_SECONDS)
cfg["search_walltime_seconds"] = int(effective_search)
(OUTPUT_ROOT / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
print(f"setup_elapsed={setup_elapsed/60:.1f}m effective_search={effective_search/3600:.2f}h orchestration_reserve={ORCHESTRATION_RESERVE_SECONDS/60:.0f}m final_reserve={FINAL_RESERVE_SECONDS/60:.0f}m")
campaigns = {}
launch_time = time.monotonic()
for gpu in (0, 1):
    env = os.environ.copy(); env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    out = OUTPUT_ROOT / f"island{gpu}"
    console = (out / "campaign_console.log").open("a" if RESUME else "w")
    cmd = ["python3", str(WORK_ROOT / "tools/meta_search/island_campaign.py"),
           "--binary", str(search_bin), "--bundle", str(WORK_ROOT),
           "--config", str(OUTPUT_ROOT / "config.json"),
           "--tuning", str(out / "tuning.json"), "--output", str(out), "--island", str(gpu)]
    if RESUME: cmd.append("--resume")
    proc = subprocess.Popen(cmd, env=env, cwd=WORK_ROOT, stdout=console, stderr=subprocess.STDOUT, start_new_session=True)
    campaigns[gpu] = {"proc": proc, "console": console, "cmd": cmd, "env": env, "restarts": 0}
    print(f"launched island{gpu} pid={proc.pid}")
'''),
code(r'''
# CELL 9 — verify BOTH physical GPUs actually have compute processes before the long monitor
expected_uuids = {g["uuid"] for g in gpus[:2]}
active_uuids = set()
compute_text = ""
for _ in range(12):
    for gpu, item in campaigns.items():
        rc = item["proc"].poll()
        assert rc is None, f"island{gpu} exited early rc={rc}; inspect {OUTPUT_ROOT / f'island{gpu}/campaign_console.log'}"
    compute_text = run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory", "--format=csv,noheader"]).stdout
    active_uuids = {line.split(",", 1)[0].strip() for line in compute_text.splitlines() if line.strip()}
    if expected_uuids.issubset(active_uuids): break
    time.sleep(5)
print(compute_text)
assert expected_uuids.issubset(active_uuids), f"ABORT: both GPUs are not active; expected {expected_uuids}, active {active_uuids}"
print("both independent GPU search processes are active")
'''),
code(r'''
# CELL 10 — monitor independently; restart only the failed island once from its checkpoint
monitor_deadline = launch_time + effective_search + ORCHESTRATION_RESERVE_SECONDS
last_log = 0.0
while True:
    alive = 0
    now = time.monotonic()
    for gpu, item in campaigns.items():
        rc = item["proc"].poll()
        if rc is None:
            alive += 1
            continue
        if rc != 0 and item["restarts"] < 1 and now + 30 * 60 < monitor_deadline:
            item["console"].close()
            item["console"] = (OUTPUT_ROOT / f"island{gpu}" / "campaign_console.log").open("a")
            cmd = list(item["cmd"])
            if "--resume" not in cmd: cmd.append("--resume")
            item["proc"] = subprocess.Popen(cmd, env=item["env"], cwd=WORK_ROOT, stdout=item["console"], stderr=subprocess.STDOUT, start_new_session=True)
            item["restarts"] += 1; alive += 1
            print(f"restarted island{gpu} from checkpoint pid={item['proc'].pid}")
    if now - last_log >= LOG_INTERVAL_SECONDS:
        last_log = now
        elapsed = now - launch_time
        print(f"elapsed={elapsed/3600:.2f}h alive={alive}")
        print(run(["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"]).stdout)
        for gpu in (0, 1):
            cp = OUTPUT_ROOT / f"island{gpu}"
            checkpoints = sorted(cp.glob("stage*/archive/checkpoint.json"))
            if checkpoints:
                print(f"island{gpu} latest:", checkpoints[-1].read_text().strip())
    if alive == 0 or now >= monitor_deadline:
        break
    time.sleep(60)
'''),
code(r'''
# CELL 11 — graceful shutdown guard
for gpu, item in campaigns.items():
    proc = item["proc"]
    if proc.poll() is None:
        print(f"terminating island{gpu} at notebook deadline guard")
        proc.terminate()
for gpu, item in campaigns.items():
    proc = item["proc"]
    if proc.poll() is None:
        try: proc.wait(timeout=180)
        except subprocess.TimeoutExpired: proc.kill(); proc.wait()
    item["console"].close()
    print(f"island{gpu} rc={proc.returncode}")
# A prior completed generation checkpoint remains valid even if a guard termination interrupted the next generation.
'''),
code(r'''
# CELL 12 — CPU-side pre-merge inventory/canonical dedup count
all_bins = sorted(OUTPUT_ROOT.glob("island*/stage*/archive/*.bin"))
hashes = {}
for p in all_bins:
    h = hashlib.sha256(p.read_bytes()).hexdigest()
    hashes.setdefault(h, []).append(str(p))
print(f"archive binaries={len(all_bins)} unique canonical programs={len(hashes)}")
assert hashes, "no elite programs available to merge"
'''),
code(r'''
# CELL 13 — deterministic merge + common full 168-development-tape screen on BOTH GPUs
merged = OUTPUT_ROOT / "merged"
cmd = ["python3", str(WORK_ROOT / "tools/meta_search/merge_elites.py"),
       "--binary", str(search_bin), "--bundle", str(WORK_ROOT), "--search-root", str(OUTPUT_ROOT),
       "--output", str(merged), "--max-candidates", str(FINAL_SCREEN_MAX_CANDIDATES),
       "--seed", str(cfg["final_screen_seed"]), "--gpus", "0,1",
       "--tuning0", str(OUTPUT_ROOT / "island0/tuning.json"), "--tuning1", str(OUTPUT_ROOT / "island1/tuning.json")]
p = subprocess.run(cmd, cwd=WORK_ROOT, text=True, capture_output=True)
print(p.stdout)
if p.returncode != 0:
    print(p.stderr)
    raise RuntimeError("final common development screen failed")
'''),
code(r'''
# CELL 14 — summarize and export compact result bundle
summary = json.loads((OUTPUT_ROOT / "merged/summary.json").read_text())
print(json.dumps(summary, indent=2, ensure_ascii=False))
for gpu in (0, 1):
    p = OUTPUT_ROOT / f"island{gpu}/summary.json"
    if p.exists(): print(f"island{gpu}:", p.read_text())

archive = Path("/kaggle/working/kaggriculture_dual_t4_meta_search_results.tar.gz")
with tarfile.open(archive, "w:gz") as tf:
    tf.add(OUTPUT_ROOT, arcname="meta_search")
print("RESULT_ARCHIVE", archive)
print("Download this archive and import it locally under scratch/bonsai_meta_search_runs/<run-id>/ before any blind validation or champion promotion.")
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
