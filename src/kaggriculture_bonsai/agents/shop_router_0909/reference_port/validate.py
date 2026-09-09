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

from .agent import ShopRouter0909Agent
from .extract import ASSETS, reference
from .native import BUILD, NativeAgent, state_summary

MISMATCH_DUMP = Path("/tmp/shop_router_0909_mismatch.json")


def canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()


def engine():
    if importlib.metadata.version("kaggle-environments") != "1.32.7":
        raise RuntimeError("official engine must be pinned to 1.32.7")
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        from kaggle_environments import make
    return make


def opponent(label):
    if label in ("random", "starter"):
        return label
    if label == "reference":
        return reference()["agent"]
    if label == "port":
        return ShopRouter0909Agent()
    if label == "gold_v25":
        from kaggriculture_bonsai.agents.gold_v25.agent import GoldV25Agent
        return GoldV25Agent()
    raise ValueError(label)


def play(make, subject, label, seed, seat):
    agents = [opponent(label), opponent(label)]
    agents[seat] = subject
    env = make("kaggriculture", configuration={"episodeSteps": 720, "seed": seed}, debug=True)
    env.run(agents)
    status = [s.status for s in env.state]
    if status != ["DONE", "DONE"]:
        raise AssertionError((label, seed, seat, status))
    return {"rewards": [s.reward for s in env.state], "status": status}


class Audit:
    """One donor arm plus the Python, CPU and CUDA port arms on identical inputs."""

    def __init__(self, cpu, cuda):
        self.ref = reference()
        self.python = ShopRouter0909Agent()
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
        expected = self.ref["agent"](copy.deepcopy(obs), cfg)
        player = int(obs["player"])
        donor_state = self.ref["_POLICY"].players[player]
        summary = state_summary(donor_state)
        for label, candidate in (("python", self.python), ("cpu", self.cpu), ("cuda", self.cuda)):
            actual = candidate(copy.deepcopy(obs), cfg)
            actual_state = (state_summary(candidate.players[player]) if label == "python" else candidate.state)
            if actual != expected or actual_state != summary:
                MISMATCH_DUMP.write_text(json.dumps(
                    {"backend": label, "step": obs["step"], "observation": obs, "expected": expected,
                     "actual": actual, "expected_state": summary, "actual_state": actual_state},
                    indent=2, default=str))
                raise AssertionError(f"{label} mismatch at step {obs['step']}; {MISMATCH_DUMP}")
        self.steps += 1
        self.routes[summary["route"]] += 1
        if summary["queues"]:
            self.events["weed_queue"] += 1
        if summary["advanced"]:
            self.events["sale_lead"] += 1
        if obs["step"] == 718:
            self.events["liquidate"] += 1
        self.trace.update(canonical([obs["step"], expected, summary]))
        return expected


# Twenty-four full games: every donor plan reached at least once, both seats,
# and only frozen deterministic opponents. Disjoint from every later panel.
PANEL = (
    ("reference", 0, 0),  # seat-0 plan 0
    ("port", 0, 1),  # seat-0 plan 0
    ("gold_v25", 55, 0),  # seat-0 plan 1
    ("reference", 55, 1),  # seat-0 plan 1
    ("port", 3, 0),  # seat-0 plan 3
    ("gold_v25", 3, 1),  # seat-0 plan 3
    ("reference", 120, 0),  # seat-0 plan 4
    ("port", 120, 1),  # seat-0 plan 4
    ("gold_v25", 45, 0),  # seat-0 plan 5
    ("reference", 45, 1),  # seat-0 plan 5
    ("port", 153, 0),  # seat-0 plan 6
    ("gold_v25", 153, 1),  # seat-0 plan 6
    ("reference", 127, 0),  # seat-0 plan 7
    ("port", 127, 1),  # seat-0 plan 7
    ("gold_v25", 17, 0),  # seat-0 plan 8
    ("reference", 17, 1),  # seat-0 plan 8
    ("port", 2, 0),  # seat-0 plan 9
    ("gold_v25", 2, 1),  # seat-0 plan 9
    ("reference", 38, 0),  # seat-0 plan 10
    ("port", 38, 1),  # seat-0 plan 10
    ("gold_v25", 25, 0),  # seat-0 plan 11
    ("reference", 25, 1),  # seat-0 plan 11
    ("port", 150, 0),  # seat-0 plan 12
    ("gold_v25", 150, 1),  # seat-0 plan 12
)


def run_panel(cpu, cuda, out=ASSETS):
    make = engine()
    rows, routes, events, steps = [], Counter(), Counter(), 0
    for label, seed, seat in PANEL:
        audit = Audit(cpu, cuda)
        try:
            result = play(make, audit, label, seed, seat)
            assert audit.steps == 719, audit.steps
            rows.append({"opponent": label, "seed": seed, "seat": seat, **result,
                         "trace_sha256": audit.trace.hexdigest(), "routes": dict(audit.routes),
                         "events": dict(audit.events)})
            steps += audit.steps
            routes.update(audit.routes)
            events.update(audit.events)
            print(f"panel {len(rows)}/{len(PANEL)} {label} seed={seed} seat={seat} {result['rewards']}", flush=True)
        finally:
            audit.close()
    assert set(routes) == set(range(13)), sorted(routes)
    assert events["weed_queue"] and events["sale_lead"] and events["liquidate"], events
    payload = {"panel": "shop_router_0909_differential_v1", "games": len(rows), "acting_steps": steps,
               "action_mismatches": 0, "state_mismatches": 0, "cpu_cuda_mismatches": 0,
               "routes": dict(sorted(routes.items())), "events": dict(events), "cases": rows}
    Path(out).mkdir(parents=True, exist_ok=True)
    (Path(out) / "differential.json").write_text(json.dumps(payload, indent=2) + "\n")
    return payload


# Sixty-four matched cases (128 official-engine runs) on canary-only seeds. Both
# arms replay the same game independently, so the opponent must be deterministic.
CANARY = tuple(("reference", seed, seat) for seed in range(9000, 9016) for seat in (0, 1)) + \
         tuple(("gold_v25", seed, seat) for seed in range(9100, 9116) for seat in (0, 1))


def canary(cpu, out=ASSETS, cases=CANARY):
    """Matched official-engine arms: donor Python versus the CPU-native port."""
    make = engine()
    rows = []
    for label, seed, seat in cases:
        ref = reference()
        expected = []

        def original(obs, cfg=None):
            action = ref["agent"](obs, cfg)
            expected.append((copy.deepcopy(obs), copy.deepcopy(action),
                             state_summary(ref["_POLICY"].players[int(obs["player"])])))
            return action

        first = play(make, original, label, seed, seat)
        native = NativeAgent(cpu)
        count, digest = 0, hashlib.sha256()

        def translated(obs, cfg=None):
            nonlocal count
            old_obs, old_action, old_state = expected[count]
            action = native(obs, cfg)
            if obs != old_obs or action != old_action or native.state != old_state:
                raise AssertionError(f"canary discrepancy {label} {seed} {seat} step={count}")
            digest.update(canonical(action))
            count += 1
            return action

        try:
            second = play(make, translated, label, seed, seat)
            assert count == len(expected) == 719, (count, len(expected))
            assert first == second, (first, second)
        finally:
            native.close()
        rows.append({"opponent": label, "seed": seed, "seat": seat, **first,
                     "action_trace_sha256": digest.hexdigest()})
        print(f"canary {len(rows)}/{len(cases)} {label} seed={seed} seat={seat} {first['rewards']}", flush=True)
    payload = {"panel": "shop_router_0909_official_canary_v1", "engine": "1.32.7",
               "matched_cases": len(rows), "engine_runs": 2 * len(rows),
               "acting_steps_compared": len(rows) * 719, "action_mismatches": 0,
               "state_mismatches": 0, "observation_mismatches": 0, "terminal_mismatches": 0,
               "faults": 0, "cases": rows}
    Path(out).mkdir(parents=True, exist_ok=True)
    (Path(out) / "canary.json").write_text(json.dumps(payload, indent=2) + "\n")
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("mode", choices=("panel", "canary"))
    parser.add_argument("--cpu", type=Path, default=BUILD / "shop_cpu.so")
    parser.add_argument("--cuda", type=Path, default=BUILD / "shop_cuda.so")
    args = parser.parse_args()
    result = run_panel(args.cpu, args.cuda) if args.mode == "panel" else canary(args.cpu)
    print(json.dumps({k: v for k, v in result.items() if k != "cases"}, indent=2))
