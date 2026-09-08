"""Bounded differential panel and paired official-engine canary. No search."""
from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import importlib.metadata
import io
import json
from collections import Counter
from pathlib import Path

from .agent import GoldV25Agent
from .extract import ASSETS, SOURCE_SHA256, reference, sha
from .native import NativeAgent, state_summary


def canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()


class Audit:
    def __init__(self, cpu, cuda):
        self.ref = reference()
        self.python = GoldV25Agent()
        self.cpu = NativeAgent(cpu)
        self.cuda = NativeAgent(cuda, gpu=True)
        self.steps = 0
        self.routes = Counter()
        self.events = Counter()
        self.trace = hashlib.sha256()

    def close(self):
        self.cpu.close()
        self.cuda.close()

    def __call__(self, obs, cfg=None):
        expected = self.ref["agent"](obs, cfg)
        p = int(obs.get("player", 0))
        state = self.ref["_IMPL"].chassis.players[p]
        summary = state_summary(state)
        for label, candidate in (("python", self.python), ("cpu", self.cpu), ("cuda", self.cuda)):
            actual = candidate(obs, cfg)
            actual_state = candidate.chassis.players[p] if label == "python" else candidate.state
            wanted = state if label == "python" else summary
            if actual != expected or actual_state != wanted:
                mismatch = {"backend": label, "step": obs["step"], "observation": obs, "expected": expected, "actual": actual,
                            "expected_state": wanted, "actual_state": actual_state}
                Path("/tmp/gold_v25_mismatch.json").write_text(json.dumps(mismatch, indent=2))
                raise AssertionError(f"{label} mismatch at step {obs['step']}; /tmp/gold_v25_mismatch.json")
        self.steps += 1
        self.routes[state["route"]] += 1
        if state["pending"]:
            self.events["pending_weed"] += 1
        if state["sell_state"]["suppress"]:
            self.events["sell_lead"] += 1
        if obs["step"] >= 718:
            self.events["terminal_rescue"] += 1
        self.trace.update(canonical([obs["step"], expected, summary]))
        return expected


def engine():
    if importlib.metadata.version("kaggle-environments") != "1.32.7":
        raise RuntimeError("official engine must be pinned to 1.32.7")
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        from kaggle_environments import make
    return make


def opponent(label):
    if label == "starter":
        return label
    if label == "reference":
        return reference()["agent"]
    if label == "champion":
        from kaggriculture_bonsai.runtime.champion import BKG4ChampionAgent
        return BKG4ChampionAgent()
    raise ValueError(label)


def play(make, subject, label, seed, seat):
    agents = [opponent(label), opponent(label)]
    agents[seat] = subject
    env = make("kaggriculture", configuration={"episodeSteps": 720, "seed": seed}, debug=True)
    # Pinned Agent.act(debug=True) propagates exceptions immediately.
    env.run(agents)
    final = env.state
    status = [s.status for s in final]
    if status != ["DONE", "DONE"]:
        raise AssertionError((label, seed, seat, status))
    return {"rewards": [s.reward for s in final], "status": status}


def run_panel(cpu, cuda):
    make = engine()
    rows, routes, events = [], Counter(), Counter()
    steps = 0
    for label in ("reference", "champion"):
        for seed in (7, 1234, 271828, 314159):
            for seat in (0, 1):
                audit = Audit(cpu, cuda)
                try:
                    result = play(make, audit, label, seed, seat)
                    # Engine wrappers may catch exceptions; exactly 719 audited
                    # calls and zero fallback diagnostics are mandatory.
                    assert audit.steps == 719, audit.steps
                    assert not any(audit.ref["_IMPL"].chassis.diagnostics.values())
                    assert not any(audit.python.chassis.diagnostics.values())
                    rows.append({"opponent": label, "seed": seed, "seat": seat, **result, "trace_sha256": audit.trace.hexdigest(), "routes": dict(audit.routes), "events": dict(audit.events)})
                    steps += audit.steps
                    routes.update(audit.routes)
                    events.update(audit.events)
                    print(f"panel {len(rows)}/16 {label} seed={seed} seat={seat} {result['rewards']}", flush=True)
                finally:
                    audit.close()
    assert set(routes) == {0, 1, 2, 3}, routes
    assert events["pending_weed"] and events["sell_lead"] and events["terminal_rescue"], events
    result = {"panel": "gold_v25_differential_v1", "games": len(rows), "acting_steps": steps,
              "action_mismatches": 0, "state_mismatches": 0, "cpu_cuda_mismatches": 0,
              "routes": dict(routes), "events": dict(events), "cases": rows}
    (ASSETS / "differential.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def canary(cpu):
    make = engine()
    rows = []
    # 64 matched cases, independently simulated reference and CPU-native arms.
    for label in ("reference", "champion", "starter", "reference"):
        offset = len(rows)
        for seed in range(8000 + offset, 8008 + offset):
            for seat in (0, 1):
                ref = reference()
                expected = []
                def original(obs, cfg=None):
                    action = ref["agent"](obs, cfg)
                    expected.append((copy.deepcopy(obs), copy.deepcopy(action), copy.deepcopy(ref["_IMPL"].chassis.players[seat])))
                    return action
                a = play(make, original, label, seed, seat)
                native = NativeAgent(cpu)
                count = 0
                digest = hashlib.sha256()
                def translated(obs, cfg=None):
                    nonlocal count
                    old_obs, old_action, old_state = expected[count]
                    action = native(obs, cfg)
                    if obs != old_obs or action != old_action or native.state != state_summary(old_state):
                        raise AssertionError(f"canary discrepancy {label} {seed} {seat} step={count}")
                    digest.update(canonical(action))
                    count += 1
                    return action
                try:
                    b = play(make, translated, label, seed, seat)
                    assert count == len(expected) == 719, (count, len(expected))
                    assert a == b, (a, b)
                    assert not any(ref["_IMPL"].chassis.diagnostics.values())
                finally:
                    native.close()
                rows.append({"opponent": label, "seed": seed, "seat": seat, **a, "action_trace_sha256": digest.hexdigest()})
                print(f"canary {len(rows)}/64 {label} seed={seed} seat={seat} {a['rewards']}", flush=True)
    result = {"panel": "gold_v25_official_canary_v1", "engine": "1.32.7", "matched_cases": 64, "engine_runs": 128,
              "acting_steps_compared": 64 * 719, "action_mismatches": 0, "state_mismatches": 0,
              "observation_mismatches": 0, "terminal_mismatches": 0, "faults": 0, "cases": rows}
    (ASSETS / "canary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("mode", choices=("panel", "canary"))
    parser.add_argument("--cpu", type=Path, default=Path("/tmp/gold_v25_native/gold_cpu.so"))
    parser.add_argument("--cuda", type=Path, default=Path("/tmp/gold_v25_native/gold_cuda.so"))
    args = parser.parse_args()
    run_panel(args.cpu, args.cuda) if args.mode == "panel" else canary(args.cpu)
