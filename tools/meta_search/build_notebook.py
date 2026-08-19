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
import csv, hashlib, json, os, re, shutil, signal, subprocess, time

NOTEBOOK_STARTED = time.monotonic()
TOTAL_WALLTIME_SECONDS = 10 * 60 * 60
HARD_EXIT_DEADLINE = NOTEBOOK_STARTED + TOTAL_WALLTIME_SECONDS
FINAL_RESERVE_SECONDS = 25 * 60
ORCHESTRATION_RESERVE_SECONDS = 15 * 60
FINAL_EXPORT_GUARD_SECONDS = 3 * 60
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
assert 60 <= FINAL_EXPORT_GUARD_SECONDS < FINAL_RESERVE_SECONDS
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
assert manifest["production_population_policy"] == "throughput_targeted", "v8+ dataset required: production population policy mismatch"
assert manifest["autotune_population"] == 8192, "v8+ dataset required: autotune population mismatch"
assert manifest["watchdog_generation_aware"] is True, "v8+ dataset required: watchdog policy mismatch"
assert manifest["recorded_seed_fitness"] is True
assert manifest["stage1_recorded_seed_count"] == 12
assert manifest["recorded_source_seed_count"] >= 12
assert manifest["baseline_seed_policy"] == "recorded_per_tape"
assert manifest["baseline_cases"] == 336
baseline_source = WORK_ROOT / "tools/meta_search/baseline_dev_recorded_seed.tsv"
assert baseline_source.exists(), "recorded-seed baseline provenance file missing"
assert hashlib.sha256(baseline_source.read_bytes()).hexdigest() == manifest["baseline_source_sha256"], "recorded-seed baseline source hash mismatch"
dev_rows = list(csv.DictReader((WORK_ROOT / "populations/dev_manifest.tsv").open(encoding="utf-8"), delimiter="\t"))
baseline_rows = list(csv.DictReader((WORK_ROOT / "populations/baseline_dev_cases.tsv").open(encoding="utf-8"), delimiter="\t"))
assert len(dev_rows) == 168 and len(baseline_rows) == 336
assert len({int(r["source_seed"]) for r in dev_rows}) == manifest["recorded_source_seed_count"]
seen_baseline_cases = set()
for r in baseline_rows:
    oi, seat = int(r["route_index"]), int(r["seat"])
    assert 0 <= oi < 168 and seat in (0, 1)
    assert int(r["seed"]) == int(dev_rows[oi]["source_seed"]), f"baseline seed mismatch route={oi}"
    seen_baseline_cases.add((oi, seat))
assert len(seen_baseline_cases) == 336
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
    "final_export_guard_seconds": FINAL_EXPORT_GUARD_SECONDS,
})
(OUTPUT_ROOT / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
print(json.dumps(manifest, indent=2, ensure_ascii=False))
'''),
code(r'''
# CELL 4 — compile only search-critical binaries; final evaluator compiles in background after search launch
cc = gpus[0]["compute_capability"].replace(".", "")
arch = [f"-gencode=arch=compute_{cc},code=sm_{cc}"]
common = ["nvcc", "-O3", "-std=c++17", "-lineinfo", "-Xptxas=-v", *arch]
assert all("fast_math" not in x and "fast-math" not in x for x in common)

resident_bin = BUILD_ROOT / "bonsai_memetic_search"
evaluator_bin = BUILD_ROOT / "bonsai_search"
v6_smoke_bin = BUILD_ROOT / "stateful_v6_game_cuda"
if evaluator_bin.exists(): evaluator_bin.unlink()
evaluator_compile_cmd = [*common, "-I", str(WORK_ROOT / "cuda-engine/include"), "-o", str(evaluator_bin),
                         str(WORK_ROOT / "cuda-engine/src/bonsai_search.cu")]
run([*common, "-I", str(WORK_ROOT / "cuda-engine/include"), "-o", str(resident_bin),
     str(WORK_ROOT / "cuda-engine/src/bonsai_memetic_search.cu")], capture=False)
run([*common, "-o", str(v6_smoke_bin), str(WORK_ROOT / "bonsai-kg/tests/stateful_v6_game_cuda.cu")], capture=False)
assert resident_bin.exists() and v6_smoke_bin.exists() and not evaluator_bin.exists()
print("compiled resident/smoke binaries; final evaluator compile deferred until T4 search is active")
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
              "--sub-islands", "8", "--engine-sub-islands", "8",
              "--parity-check", "--seed-program", str(prior)]
p = subprocess.run(parity_cmd, env=env, text=True, capture_output=True)
print(p.stdout)
if p.returncode != 0:
    print(p.stderr); raise RuntimeError("SearchGenome distillation parity failed")
assert "engine_mode=1" in p.stdout and "mismatches=0" in p.stdout
print("ENGINE-MODE SEARCH-GENOME DISTILLATION PARITY PASSED")
'''),
code(r'''
# CELL 6 — concurrent exact resident-engine launch-geometry calibration on GPU0/GPU1
# The 8192-candidate calibration is the smallest representative batch that preserves
# all 32 production sub-islands. Production population is derived from measured T4
# throughput and target generation latency, with safe VRAM used only as an upper cap.
assert cfg["autotune_populations"] == [8192]
autotune_jobs = []
for gpu in (0, 1):
    env = os.environ.copy(); env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    out = OUTPUT_ROOT / f"island{gpu}"; out.mkdir(parents=True, exist_ok=True)
    tuning = out / "tuning.json"
    mode = "exploit" if gpu == 0 else "explore"
    mode_cfg = cfg[mode]
    cmd = ["python3", str(WORK_ROOT / "tools/meta_search/autotune.py"),
           "--binary", str(resident_bin), "--bundle", str(WORK_ROOT), "--output", str(tuning),
           "--populations", "8192",
           "--tpb", ",".join(map(str, cfg["autotune_tpb"])),
           "--sub-islands", str(cfg["default_sub_islands"]),
           "--engine-sub-islands", str(mode_cfg["engine_sub_islands"]),
           "--universe", mode,
           "--vram-target", str(VRAM_TARGET), "--reserve-mib", str(VRAM_RESERVE_MIB),
           "--target-generation-seconds", str(cfg["autotune_target_generation_seconds"]),
           "--production-generation-safety-factor", str(cfg["production_generation_safety_factor"]),
           "--trial-timeout-seconds", str(cfg["autotune_max_generation_seconds"]),
           "--seed", str(770001 + gpu * 10000)]
    p = subprocess.Popen(cmd, env=env, cwd=WORK_ROOT, text=True,
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
    autotune_jobs.append((gpu, p, cmd))

for gpu, p, cmd in autotune_jobs:
    stdout, stderr = p.communicate()
    print(f"GPU{gpu} autotune:\n{stdout[-8000:]}")
    if p.returncode != 0:
        print(stderr); raise RuntimeError(f"GPU{gpu} resident autotune failed: {cmd}")
    tuning_data = json.loads((OUTPUT_ROOT / f"island{gpu}/tuning.json").read_text())
    production_population = int(tuning_data["production_population"])
    expected_generation = float(tuning_data["estimated_production_core_generation_seconds"])
    assert production_population >= cfg["default_sub_islands"] * 256
    assert production_population % cfg["default_sub_islands"] == 0
    assert 0 < expected_generation <= cfg["autotune_target_generation_seconds"]
    print(f"GPU{gpu} production population={production_population} expected_core_generation={expected_generation:.1f}s "
          f"safe_fraction={tuning_data['production_allocated_fraction_of_safe']:.3f}")
print("both resident T4 autotuners passed concurrently with bounded production generation latency")
'''),
code(r'''
# CELL 7 — compute the remaining search budget from one monotonic absolute deadline
now = time.monotonic()
remaining = HARD_EXIT_DEADLINE - now
search_seconds = remaining - FINAL_RESERVE_SECONDS - ORCHESTRATION_RESERVE_SECONDS
assert search_seconds >= 60 * 60, f"only {search_seconds/60:.1f} minutes remain for search; restart the notebook cleanly"
MERGE_START_DEADLINE = HARD_EXIT_DEADLINE - FINAL_RESERVE_SECONDS
SEARCH_STOP_DEADLINE = MERGE_START_DEADLINE - ORCHESTRATION_RESERVE_SECONDS
FINAL_SCREEN_DEADLINE = HARD_EXIT_DEADLINE - FINAL_EXPORT_GUARD_SECONDS
assert SEARCH_STOP_DEADLINE < MERGE_START_DEADLINE < FINAL_SCREEN_DEADLINE < HARD_EXIT_DEADLINE
search_seconds = max(0.0, SEARCH_STOP_DEADLINE - time.monotonic())
print(f"setup_elapsed={(time.monotonic()-NOTEBOOK_STARTED)/60:.1f}m search_budget={search_seconds/3600:.2f}h "
      f"shutdown_reserve={ORCHESTRATION_RESERVE_SECONDS/60:.0f}m final_reserve={FINAL_RESERVE_SECONDS/60:.0f}m "
      f"export_guard={FINAL_EXPORT_GUARD_SECONDS/60:.0f}m")
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
    return {"proc": proc, "console": console, "env": env, "cmd": cmd, "restarts": 0, "watchdog_stopped": False}

def terminate_campaign(item, grace_seconds: float):
    proc = item["proc"]
    if proc.poll() is not None:
        return
    try: os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError: return
    deadline = time.monotonic() + max(0.0, grace_seconds)
    while proc.poll() is None and time.monotonic() < deadline:
        time.sleep(1)
    if proc.poll() is None:
        try: os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError: pass
        proc.wait(timeout=10)

for gpu in (0, 1):
    campaigns[gpu] = launch_island(gpu, resume=RESUME, walltime=search_seconds)
    print(f"launched GPU{gpu} universe pid={campaigns[gpu]['proc'].pid}")
'''),
code(r'''
# CELL 9 — warm-up gate: exact throughput-sized populations active and both T4s compute-bound
expected_uuids = {g["uuid"] for g in gpus[:2]}
mem_re = re.compile(r"RESIDENT_MEMORY .*?population=(\d+).*?allocated_fraction_of_safe=([0-9.eE+-]+)")
resident_seen = {}
for _ in range(36):
    for gpu, item in campaigns.items():
        rc = item["proc"].poll()
        assert rc is None, f"GPU{gpu} universe exited early rc={rc}; inspect campaign_console.log/resident_search.log"
        log = OUTPUT_ROOT / f"island{gpu}/resident_search.log"
        if log.exists():
            m = mem_re.search(log.read_text(errors="replace"))
            if m: resident_seen[gpu] = {"population": int(m.group(1)), "safe_fraction": float(m.group(2))}
    apps = run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory", "--format=csv,noheader"]).stdout
    active = {line.split(",", 1)[0].strip() for line in apps.splitlines() if line.strip()}
    if expected_uuids.issubset(active) and len(resident_seen) == 2: break
    time.sleep(5)
print("resident memory plans:", resident_seen)
assert len(resident_seen) == 2, "resident memory plan did not become observable on both GPUs"
for gpu in (0, 1):
    tuning_data = json.loads((OUTPUT_ROOT / f"island{gpu}/tuning.json").read_text())
    expected_population = int(tuning_data["production_population"])
    seen = resident_seen[gpu]
    assert seen["population"] == expected_population, f"GPU{gpu} production population mismatch: {seen} vs {expected_population}"
    assert 0.0 < seen["safe_fraction"] < 0.99, f"GPU{gpu} unsafe resident allocation: {seen}"

# Prove the processes are computing, not merely holding memory while CPU orchestration spins.
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

# The independent final-screen evaluator is expensive to compile on Kaggle (~26m in
# the previous run) but is not needed during search. Compile it now on CPU while both
# T4 universes continue searching, hiding that setup cost behind useful GPU work.
evaluator_compile_log_path = OUTPUT_ROOT / "bonsai_search_compile.log"
evaluator_compile_log = evaluator_compile_log_path.open("w")
evaluator_compile_proc = subprocess.Popen(evaluator_compile_cmd, cwd=WORK_ROOT,
    stdout=evaluator_compile_log, stderr=subprocess.STDOUT, start_new_session=True)
print(f"background evaluator compile pid={evaluator_compile_proc.pid}; T4 search continues concurrently")
'''),
code(r'''
# CELL 10 — sparse health/utilization monitor + one-shot stale/GPU-idle watchdog
samples = []
last_generation = {0: -1, 1: -1}
last_progress_time = {0: time.monotonic(), 1: time.monotonic()}
low_util_streak = {0: 0, 1: 0}
WATCHDOG_STALE_SECONDS = float(cfg["watchdog_stale_seconds"])
WATCHDOG_GENERATION_MULTIPLIER = float(cfg["watchdog_generation_multiplier"])
WATCHDOG_GENERATION_MARGIN_SECONDS = float(cfg["watchdog_generation_margin_seconds"])
EXPECTED_GENERATION_SECONDS = {
    gpu: float(json.loads((OUTPUT_ROOT / f"island{gpu}/tuning.json").read_text())["estimated_production_core_generation_seconds"])
    for gpu in (0, 1)
}
WATCHDOG_STALE_LIMIT = {
    gpu: max(WATCHDOG_STALE_SECONDS,
             EXPECTED_GENERATION_SECONDS[gpu] * WATCHDOG_GENERATION_MULTIPLIER + WATCHDOG_GENERATION_MARGIN_SECONDS)
    for gpu in (0, 1)
}
print("watchdog expected generation seconds:", EXPECTED_GENERATION_SECONDS,
      "adaptive stale limits:", WATCHDOG_STALE_LIMIT)
WATCHDOG_LOW_GPU_UTIL_PCT = float(cfg["watchdog_low_gpu_util_pct"])
WATCHDOG_LOW_GPU_SAMPLES = int(cfg["watchdog_low_gpu_samples"])
WATCHDOG_RESTART_GRACE_SECONDS = float(cfg["watchdog_restart_grace_seconds"])
WATCHDOG_MIN_RESTART_REMAINING_SECONDS = float(cfg["watchdog_min_restart_remaining_seconds"])

def burst_gpu_utilization():
    vals = {0: [], 1: []}
    for k in range(3):
        rows = run(["nvidia-smi", "--query-gpu=index,utilization.gpu", "--format=csv,noheader,nounits"]).stdout.splitlines()
        for row in rows[:2]:
            gi, util = [x.strip() for x in row.split(",")]
            gi = int(gi)
            if gi in vals: vals[gi].append(float(util))
        if k < 2: time.sleep(2)
    return vals, {g: max(v) if v else 0.0 for g, v in vals.items()}

def restart_or_stop(gpu: int, reason: str, now: float):
    item = campaigns[gpu]
    remaining = SEARCH_STOP_DEADLINE - now
    if item["restarts"] < 1 and remaining >= WATCHDOG_MIN_RESTART_REMAINING_SECONDS:
        print(f"WATCHDOG GPU{gpu}: {reason}; graceful checkpointed restart")
        terminate_campaign(item, WATCHDOG_RESTART_GRACE_SECONDS)
        item["console"].close()
        replacement = launch_island(gpu, resume=True, walltime=remaining)
        replacement["restarts"] = item["restarts"] + 1
        campaigns[gpu] = replacement
        last_generation[gpu] = -1
        last_progress_time[gpu] = time.monotonic()
        low_util_streak[gpu] = 0
        return "restarted"
    if remaining < WATCHDOG_MIN_RESTART_REMAINING_SECONDS:
        print(f"WATCHDOG GPU{gpu}: {reason}; too near search deadline to restart, preserving last checkpoint")
        terminate_campaign(item, WATCHDOG_RESTART_GRACE_SECONDS)
        item["console"].close(); item["watchdog_stopped"] = True
        return "stopped"
    terminate_campaign(item, WATCHDOG_RESTART_GRACE_SECONDS)
    item["console"].close()
    raise RuntimeError(f"GPU{gpu} stalled again after its one allowed restart: {reason}")

while time.monotonic() < SEARCH_STOP_DEADLINE:
    now = time.monotonic()
    # First handle actual process exits. Any substantially early exit gets one resume.
    for gpu, item in list(campaigns.items()):
        rc = item["proc"].poll()
        if rc is None or item.get("watchdog_stopped"): continue
        if now + WATCHDOG_MIN_RESTART_REMAINING_SECONDS < SEARCH_STOP_DEADLINE and item["restarts"] < 1:
            item["console"].close()
            replacement = launch_island(gpu, resume=True, walltime=SEARCH_STOP_DEADLINE-now)
            replacement["restarts"] = item["restarts"] + 1
            campaigns[gpu] = replacement
            last_generation[gpu] = -1; last_progress_time[gpu] = now; low_util_streak[gpu] = 0
            print(f"GPU{gpu} exited rc={rc}; restarted once from compact checkpoint")
        elif rc != 0 and item["restarts"] >= 1:
            item["console"].close()
            raise RuntimeError(f"GPU{gpu} failed again after restart rc={rc}")

    telemetry = run(["nvidia-smi",
        "--query-gpu=index,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu",
        "--format=csv,noheader,nounits"]).stdout.strip()
    util_burst, effective_util = burst_gpu_utilization()
    sample = {"elapsed_seconds": now - NOTEBOOK_STARTED, "telemetry": telemetry,
              "utilization_burst": util_burst, "effective_gpu_util": effective_util, "islands": {}}

    # Checkpoint generation is the progress heartbeat. It is intentionally sparse and
    # independent of Python orchestration; a changed generation resets the stale timer.
    for gpu in (0, 1):
        cp = OUTPUT_ROOT / f"island{gpu}/checkpoint.json"
        if cp.exists():
            x = json.loads(cp.read_text()); sample["islands"][str(gpu)] = x
            gen = int(x.get("generation", -1))
            if gen != last_generation[gpu]:
                last_generation[gpu] = gen
                last_progress_time[gpu] = now
        if campaigns[gpu]["proc"].poll() is None:
            if effective_util[gpu] < WATCHDOG_LOW_GPU_UTIL_PCT:
                low_util_streak[gpu] += 1
            else:
                low_util_streak[gpu] = 0

    evaluator_rc = evaluator_compile_proc.poll()
    sample["evaluator_compile_returncode"] = evaluator_rc
    if evaluator_rc is not None and evaluator_rc != 0:
        evaluator_compile_log.flush()
        tail = evaluator_compile_log_path.read_text(errors="replace")[-8000:]
        for item in campaigns.values(): terminate_campaign(item, WATCHDOG_RESTART_GRACE_SECONDS)
        raise RuntimeError(f"background bonsai_search compile failed rc={evaluator_rc}:\n{tail}")

    samples.append(sample)
    (OUTPUT_ROOT / "utilization_samples.jsonl").write_text("".join(json.dumps(x) + "\n" for x in samples))
    print(f"elapsed={(now-NOTEBOOK_STARTED)/3600:.2f}h generations={last_generation} effective_util={effective_util} "
          f"low_streak={low_util_streak} evaluator_compile_rc={evaluator_rc}\n{telemetry}")

    # Do not react to one transient 0% nvidia-smi sample. A watchdog event requires
    # either a full stale window with no checkpoint generation progress, or multiple
    # consecutive burst-sampled near-idle monitor cycles.
    for gpu in (0, 1):
        item = campaigns[gpu]
        if item["proc"].poll() is not None or item.get("watchdog_stopped"): continue
        stale_for = now - last_progress_time[gpu]
        stale = stale_for >= WATCHDOG_STALE_LIMIT[gpu]
        low = low_util_streak[gpu] >= WATCHDOG_LOW_GPU_SAMPLES
        if stale or low:
            reason = f"no_generation_progress={stale_for:.0f}s" if stale else f"low_gpu_util_streak={low_util_streak[gpu]} effective={effective_util[gpu]:.1f}%"
            restart_or_stop(gpu, reason, now)

    if all(item["proc"].poll() is not None or item.get("watchdog_stopped") for item in campaigns.values()):
        break
    time.sleep(min(MONITOR_INTERVAL_SECONDS, max(1, SEARCH_STOP_DEADLINE - time.monotonic())))
'''),
code(r'''
# CELL 11 — graceful stop guard; resident engines checkpoint themselves before final merge
for gpu, item in campaigns.items():
    if item["proc"].poll() is None:
        print(f"requesting graceful stop GPU{gpu}")
        try: os.killpg(item["proc"].pid, signal.SIGTERM)
        except ProcessLookupError: pass
shutdown_deadline = MERGE_START_DEADLINE
for gpu, item in campaigns.items():
    proc = item["proc"]
    while proc.poll() is None and time.monotonic() < shutdown_deadline:
        time.sleep(2)
    if proc.poll() is None:
        print(f"GPU{gpu} exceeded graceful shutdown reserve; forcing process-group termination")
        try: os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError: pass
        proc.wait(timeout=10)
    if not item["console"].closed: item["console"].close()
    print(f"GPU{gpu} supervisor rc={proc.returncode}")
assert time.monotonic() < MERGE_START_DEADLINE + 5, "graceful shutdown consumed final-screen reserve"

# The background evaluator compile should have finished hours ago. Give it only the
# remaining orchestration reserve if it is unexpectedly still running.
if evaluator_compile_proc.poll() is None:
    remaining_compile = max(1.0, MERGE_START_DEADLINE - time.monotonic())
    try:
        evaluator_compile_proc.wait(timeout=remaining_compile)
    except subprocess.TimeoutExpired:
        try: os.killpg(evaluator_compile_proc.pid, signal.SIGTERM)
        except ProcessLookupError: pass
        time.sleep(2)
        if evaluator_compile_proc.poll() is None:
            try: os.killpg(evaluator_compile_proc.pid, signal.SIGKILL)
            except ProcessLookupError: pass
        evaluator_compile_proc.wait(timeout=10)
evaluator_compile_log.close()
if evaluator_compile_proc.returncode != 0 or not evaluator_bin.exists():
    tail = evaluator_compile_log_path.read_text(errors="replace")[-8000:]
    raise RuntimeError(f"final evaluator compile unavailable rc={evaluator_compile_proc.returncode}:\n{tail}")
print("background final evaluator compile PASSED")
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
# CELL 13 — deadline-bounded replay-faithful 168-tape × both-seat exact screen on BOTH T4s
merged = OUTPUT_ROOT / "merged"; merged.mkdir(parents=True, exist_ok=True)
merge_deadline_epoch = time.time() + max(0.0, FINAL_SCREEN_DEADLINE - time.monotonic())
cmd = ["python3", str(WORK_ROOT / "tools/meta_search/merge_elites.py"),
       "--binary", str(evaluator_bin), "--bundle", str(WORK_ROOT), "--search-root", str(OUTPUT_ROOT),
       "--output", str(merged), "--max-candidates", str(FINAL_SCREEN_MAX_CANDIDATES),
       "--deadline-epoch", str(merge_deadline_epoch),
       "--deadline-safety-seconds", str(cfg["final_screen_deadline_safety_seconds"]),
       "--probe-candidates-per-gpu", str(cfg["final_screen_probe_candidates_per_gpu"]),
       "--probe-opponents", str(cfg["final_screen_probe_opponents"]),
       "--probe-max-seconds", str(cfg["final_screen_probe_max_seconds"]),
       "--throughput-safety-factor", str(cfg["final_screen_throughput_safety_factor"]),
       "--gpus", "0,1", "--tuning0", str(OUTPUT_ROOT / "island0/tuning.json"),
       "--tuning1", str(OUTPUT_ROOT / "island1/tuning.json")]
p = subprocess.Popen(cmd, cwd=WORK_ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
try:
    stdout, stderr = p.communicate(timeout=max(1.0, FINAL_SCREEN_DEADLINE - time.monotonic()))
except subprocess.TimeoutExpired:
    try: os.killpg(p.pid, signal.SIGTERM)
    except ProcessLookupError: pass
    time.sleep(2)
    if p.poll() is None:
        try: os.killpg(p.pid, signal.SIGKILL)
        except ProcessLookupError: pass
    stdout, stderr = p.communicate()
    print(stdout); print(stderr)
    raise RuntimeError("final merge exceeded its absolute screen deadline; process group terminated")
print(stdout)
if p.returncode != 0:
    print(stderr); raise RuntimeError("dual-GPU replay-faithful final development screen failed")
assert time.monotonic() < FINAL_SCREEN_DEADLINE, "final screen consumed the protected export guard"
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
export = subprocess.Popen(["tar", "-czf", str(archive), "-C", str(OUTPUT_ROOT.parent), OUTPUT_ROOT.name], start_new_session=True)
try:
    export.wait(timeout=max(1.0, HARD_EXIT_DEADLINE - time.monotonic() - 10.0))
except subprocess.TimeoutExpired:
    try: os.killpg(export.pid, signal.SIGKILL)
    except ProcessLookupError: pass
    export.wait(timeout=5)
    raise RuntimeError("artifact export exceeded hard 10-hour deadline reserve")
if export.returncode != 0: raise RuntimeError(f"artifact export failed rc={export.returncode}")
assert time.monotonic() < HARD_EXIT_DEADLINE, "hard 10-hour deadline exceeded during export"
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
