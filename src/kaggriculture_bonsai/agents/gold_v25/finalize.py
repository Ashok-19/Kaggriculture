"""Check the bounded evidence and write the isolated candidate validation manifest."""
import ctypes
import importlib.metadata
import json
import subprocess

from .extract import ASSETS, HERE, ROOT, SOURCE_SHA256, source, sha
from .native import NativeAgent
from .validate import canonical

STARTING_HEAD = "f2df0832e3a7c3f4411549aa19a59c7d0e4ca1d7"


def main():
    assert sha(source().encode()) == SOURCE_SHA256
    identity = json.loads((ASSETS / "identity.json").read_text())
    panel = json.loads((ASSETS / "differential.json").read_text())
    canary = json.loads((ASSETS / "canary.json").read_text())
    assert panel["games"] == 16 and panel["acting_steps"] == 16 * 719
    assert set(panel["routes"]) == {"0", "1", "2", "3"}
    assert canary["matched_cases"] == 64 and canary["engine_runs"] == 128
    assert canary["acting_steps_compared"] == 64 * 719
    for result in (panel, canary):
        for key, value in result.items():
            if "mismatches" in key or key == "faults":
                assert value == 0, (key, value)
        assert all(row["status"] == ["DONE", "DONE"] for row in result["cases"])
    commands = [
        [str(ROOT / ".venv/bin/python"), "-m", "unittest", "kaggriculture_bonsai.agents.gold_v25.test_gold_v25"],
        [str(ROOT / ".venv/bin/python"), "-m", "pytest", "-q", "tests/unit/test_v4_semantics.py", "tests/parity/test_evaluator_parity.py"],
    ]
    tests = []
    for command in commands:
        run = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=True)
        tests.append({"command": command[1:], "returncode": run.returncode, "summary": (run.stdout + run.stderr).strip()})
    subprocess.run(["git", "diff", "--exit-code", STARTING_HEAD, "--", "assets/champion", "assets/baseline"], cwd=ROOT, check=True)
    protected = subprocess.check_output(["git", "ls-files", "assets/champion", "assets/baseline"], cwd=ROOT, text=True).splitlines()
    protected_hashes = {p: sha((ROOT / p).read_bytes()) for p in protected}
    binaries = {}
    for backend in ("cpu", "cuda"):
        path = f"/tmp/gold_v25_native/gold_{backend}.so"
        agent = NativeAgent(path, gpu=backend == "cuda")
        try:
            binaries[backend] = {"sha256": sha(open(path, "rb").read()), "state_bytes_per_player": agent.lib.gold_size(2)}
        finally:
            agent.close()
    distribution = importlib.metadata.distribution("kaggle-environments")
    assert distribution.version == "1.32.7"
    engine_source = distribution.locate_file("kaggle_environments/envs/kaggriculture/kaggriculture.py")
    source_hashes = {str(p.relative_to(ROOT)): sha(p.read_bytes()) for p in sorted(HERE.iterdir()) if p.is_file()}
    validation = {
        "candidate": "BKG4_GOLD_V25", "status": "EXACT_BASELINE_READY_NOT_PROMOTED",
        "starting_head": STARTING_HEAD, "source_files": source_hashes,
        "official_engine": {"version": distribution.version, "source_sha256": sha(engine_source.read_bytes())},
        "differential_panel_identity": sha(canonical([{k: r[k] for k in ("opponent", "seed", "seat")} for r in panel["cases"]])),
        "canary_panel_identity": sha(canonical([{k: r[k] for k in ("opponent", "seed", "seat")} for r in canary["cases"]])),
        "differential_games": 16, "differential_acting_steps": 16 * 719,
        "differential_action_mismatches": 0, "differential_state_mismatches": 0,
        "cpu_cuda_mismatches": 0, "official_canary_matched_cases": 64,
        "official_canary_engine_runs": 128, "official_canary_acting_steps": 64 * 719,
        "official_canary_action_mismatches": 0, "official_canary_terminal_mismatches": 0,
        "crashes_faults": 0, "tests": tests, "binaries": binaries,
        "evidence_sha256": {name: sha((ASSETS / name).read_bytes()) for name in ("differential.json", "canary.json")},
        "protected_assets_unchanged": True, "protected_assets": protected_hashes,
        "generic_runtime_changes": [], "strategy_improvements": [],
        "representation_corrections": ["Restore omitted PICKUP/PLACE quantity arity from isolated sidecar", "Preserve empty market order slots as []"],
    }
    (ASSETS / "validation.json").write_text(json.dumps(validation, indent=2) + "\n")
    identity.update({"status": validation["status"], "validation_manifest": "validation.json",
                     "validation_manifest_sha256": sha((ASSETS / "validation.json").read_bytes()),
                     "validation_panel_identity": validation["differential_panel_identity"],
                     "differential_mismatch_count": 0, "cpu_cuda_mismatch_count": 0,
                     "official_canary": {"matched_cases": 64, "engine_runs": 128, "result": "EXACT", "faults": 0}})
    (ASSETS / "identity.json").write_text(json.dumps(identity, indent=2) + "\n")
    print(json.dumps({k: v for k, v in validation.items() if k not in ("source_files", "protected_assets", "tests")}, indent=2))


if __name__ == "__main__":
    main()
