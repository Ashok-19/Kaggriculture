"""Focused package checks; this file is not included in the Kaggle archive."""
import copy
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
EXTRACTED = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(EXTRACTED))
spec = importlib.util.spec_from_file_location("packaged_main", EXTRACTED / "main.py")
packaged = importlib.util.module_from_spec(spec)
spec.loader.exec_module(packaged)
sys.path.insert(0, str(ROOT / "src"))
from kaggriculture_bonsai.agents.gold_v25.extract import reference
from kaggriculture_bonsai.agents.gold_v25.validate import engine, opponent, play


def paired(label, seed, seat):
    make = engine()
    ref = reference()
    expected = []
    def original(obs, cfg=None):
        action = ref["agent"](obs, cfg)
        expected.append((copy.deepcopy(obs), copy.deepcopy(action)))
        return action
    first = play(make, original, label, seed, seat)
    count = 0
    def candidate(obs, cfg=None):
        nonlocal count
        old_obs, old_action = expected[count]
        action = packaged.agent(obs, cfg)
        assert obs == old_obs and action == old_action, (label, seed, seat, count)
        count += 1
        return action
    second = play(make, candidate, label, seed, seat)
    assert count == 719 and first == second
    return count, first


def main():
    cases_only = sys.argv[2] if len(sys.argv) > 2 else "full"
    total = 0
    for label in ("reference", "champion"):
        for seed in (7, 1234, 271828, 314159):
            for seat in (0, 1):
                n, result = paired(label, seed, seat)
                total += n
                print("differential", label, seed, seat, result, flush=True)
    if cases_only == "differential":
        print(json.dumps({"differential_games": 16, "differential_action_steps": total, "mismatches": 0}), flush=True)
        return
    rows = []
    for label in ("reference", "champion", "starter", "reference"):
        offset = len(rows)
        for seed in range(8000 + offset, 8008 + offset):
            for seat in (0, 1):
                n, result = paired(label, seed, seat)
                rows.append(result)
                print("canary", len(rows), label, seed, seat, result, flush=True)
    print(json.dumps({"differential_games": 16, "differential_action_steps": total, "canary_cases": len(rows), "canary_action_steps": len(rows) * 719, "mismatches": 0}), flush=True)


if __name__ == "__main__":
    main()
