"""CPU/CUDA parity for the frozen candidate, plus its official-engine canary.

The candidate reuses the parent's route pack and native tick; only the compiled
controller differs, so the same differential machinery applies unchanged.
"""
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

from kaggriculture_bonsai.runtime import bonsai_kg_v4 as bkg

from ..reference_port.extract import ASSETS, sha
from ..reference_port.native import BUILD, NativeAgent, state_summary
from ..reference_port.validate import canonical, engine, opponent, play
from .agent import ACTIVE, ShopRouter0909Plus

EVIDENCE = ASSETS.parent / "plus"
CONTROLLER = EVIDENCE / "controller.bkgc"

# Twelve full games on seeds disjoint from BUILD, INTERNAL_DEV and FINAL_TARGET.
PANEL = tuple(("reference", seed, seat) for seed in (0, 3, 45, 150) for seat in (0, 1)) + \
        tuple(("gold_v25", seed, seat) for seed in (17, 127) for seat in (0, 1))


def save_controller(spec, path=CONTROLLER):
    agent = ShopRouter0909Plus(spec)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    bkg.save_compiled(agent.compiled, path)
    return path, sha(Path(path).read_bytes()), agent


class Audit:
    """Python candidate versus the CPU and CUDA candidate on identical inputs."""

    def __init__(self, spec, cpu, cuda, controller, digest):
        self.python = ShopRouter0909Plus(spec)
        self.cpu = NativeAgent(cpu, controller=controller, controller_sha256=digest)
        self.cuda = NativeAgent(cuda, gpu=True, controller=controller, controller_sha256=digest)
        self.steps = 0
        self.gated = 0
        self.routes = Counter()
        self.trace = hashlib.sha256()

    def close(self):
        self.cpu.close()
        self.cuda.close()

    def __call__(self, obs, cfg=None):
        expected = self.python(copy.deepcopy(obs), cfg)
        player = int(obs["player"])
        summary = state_summary(self.python.players[player])
        for label, candidate in (("cpu", self.cpu), ("cuda", self.cuda)):
            actual = candidate(copy.deepcopy(obs), cfg)
            if actual != expected or candidate.state != summary:
                Path("/tmp/shop_router_0909_plus_mismatch.json").write_text(json.dumps(
                    {"backend": label, "step": obs["step"], "observation": obs, "expected": expected,
                     "actual": actual, "expected_state": summary, "actual_state": candidate.state},
                    indent=2, default=str))
                raise AssertionError(f"{label} mismatch at step {obs['step']}")
        self.steps += 1
        self.routes[summary["route"]] += 1
        self.gated += bool(summary["endgame"])
        self.trace.update(canonical([obs["step"], expected, summary]))
        return expected


def run_panel(spec, cpu, cuda, out=EVIDENCE):
    make = engine()
    controller, digest, _ = save_controller(spec)
    rows, steps, gated = [], 0, 0
    for label, seed, seat in PANEL:
        audit = Audit(spec, cpu, cuda, controller, digest)
        try:
            result = play(make, audit, label, seed, seat)
            assert audit.steps == 719, audit.steps
            rows.append({"opponent": label, "seed": seed, "seat": seat, **result,
                         "trace_sha256": audit.trace.hexdigest(), "gated_steps": audit.gated,
                         "routes": dict(audit.routes)})
            steps += audit.steps
            gated += audit.gated
            print(f"plus panel {len(rows)}/{len(PANEL)} {label} seed={seed} seat={seat} {result['rewards']}", flush=True)
        finally:
            audit.close()
    assert gated, "endgame gate never fired"
    payload = {"panel": "shop_router_0909_plus_parity_v1", "games": len(rows), "acting_steps": steps,
               "controller_sha256": digest, "action_mismatches": 0, "state_mismatches": 0,
               "cpu_cuda_mismatches": 0, "faults": 0, "gated_steps": gated, "cases": rows}
    Path(out).mkdir(parents=True, exist_ok=True)
    (Path(out) / "parity.json").write_text(json.dumps(payload, indent=2) + "\n")
    return payload


# Thirty-two matched cases (64 official-engine runs) on canary-only seeds.
CANARY = tuple(("reference", seed, seat) for seed in range(9200, 9208) for seat in (0, 1)) + \
         tuple(("gold_v25", seed, seat) for seed in range(9300, 9308) for seat in (0, 1))


def canary(spec, cpu, out=EVIDENCE, cases=CANARY):
    """Matched official-engine arms: Python candidate versus the CPU candidate."""
    if importlib.metadata.version("kaggle-environments") != "1.32.7":
        raise RuntimeError("official engine must be pinned to 1.32.7")
    make = engine()
    controller, digest, _ = save_controller(spec)
    rows = []
    for label, seed, seat in cases:
        python = ShopRouter0909Plus(spec)
        expected = []

        def original(obs, cfg=None):
            action = python(obs, cfg)
            expected.append((copy.deepcopy(obs), copy.deepcopy(action),
                             state_summary(python.players[int(obs["player"])])))
            return action

        first = play(make, original, label, seed, seat)
        native = NativeAgent(cpu, controller=controller, controller_sha256=digest)
        count, trace = 0, hashlib.sha256()

        def translated(obs, cfg=None):
            nonlocal count
            old_obs, old_action, old_state = expected[count]
            action = native(obs, cfg)
            if obs != old_obs or action != old_action or native.state != old_state:
                raise AssertionError(f"canary discrepancy {label} {seed} {seat} step={count}")
            trace.update(canonical(action))
            count += 1
            return action

        try:
            second = play(make, translated, label, seed, seat)
            assert count == len(expected) == 719, (count, len(expected))
            assert first == second, (first, second)
        finally:
            native.close()
        rows.append({"opponent": label, "seed": seed, "seat": seat, **first,
                     "action_trace_sha256": trace.hexdigest()})
        print(f"plus canary {len(rows)}/{len(cases)} {label} seed={seed} seat={seat} {first['rewards']}", flush=True)
    payload = {"panel": "shop_router_0909_plus_official_canary_v1", "engine": "1.32.7",
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
    spec = json.loads(ACTIVE.read_text())["candidate"]
    result = run_panel(spec, args.cpu, args.cuda) if args.mode == "panel" else canary(spec, args.cpu)
    print(json.dumps({k: v for k, v in result.items() if k != "cases"}, indent=2))
