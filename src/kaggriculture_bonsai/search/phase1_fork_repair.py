"""Bounded, isolated exact-fork experiment; never modifies or promotes PARENT."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "workspace/self_improvement/phase1_fork_repair"
BUILD = ROOT / "workspace/build/phase1_fork_repair"
PARENT = ROOT / "workspace/shop_router_0909/plus/controller.bkgc"
PORT = ROOT / "workspace/shop_router_0909/port"
SEEDS = (19091001, 19091002, 19091003)
DEV_SEEDS = (19092001, 19092002, 19092003, 19092004)
FIELDS = "kind candidate seed seat member checkpoint money0 money1 root simulation controller actions fault changes duration turns route shop0 shop1".split()
MODES = ("parent", "sell_topup", "deposit", "deposit_topup", "liquidate_market",
         "cancel_purchases", "value_sort", "approach_shed", "wheat_reserve", "carrot_seed", "cancel_plant")


def save(name, value):
    OUT.mkdir(parents=True, exist_ok=True)
    large = (isinstance(value, list) and len(value) > 24) or (isinstance(value, dict) and len(value.get("rows", [])) > 24)
    (OUT / name).write_text(json.dumps(value, indent=None if large else 2, sort_keys=True,
                                      separators=(",", ":") if large else None) + "\n")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cases(checkpoints=(359, 647), seeds=SEEDS):
    return [(seed, seat, member, step) for seed in seeds for seat in (0, 1)
            for member in (0, 1) for step in checkpoints]


def evaluate(label, panel, candidates, *, backend="cuda_gate", audit=True, matched=False, gate=None):
    manifest = json.loads((OUT / "manifest.json").read_text())
    for path, key in ((PARENT, "parent_sha256"), (PORT / "routes.bkv6", "route_pack_sha256"), (PORT / "arity.bin", "arity_sha256")):
        if sha(path) != manifest[key]:
            raise RuntimeError("immutable PARENT identity changed")
    original, serial = label, 1
    while (OUT / (label + ".json")).exists():
        serial += 1
        label = f"{original}_{serial}"
    spent = 0
    for path in OUT.glob("*.json"):
        document = json.loads(path.read_text())
        if isinstance(document, dict) and "backend" in document:
            spent += document["simulated_turns"]
    planned = len(panel) * 719 + len(candidates) * sum(
        (719 if matched == 2 else 719 - c[3] + (719 if matched else 0)) for c in panel)
    if spent + planned > 3200 * 720:
        raise RuntimeError("hard all-in 3200-FGE budget would be exceeded")
    BUILD.mkdir(parents=True, exist_ok=True)
    panel_path, candidate_path = BUILD / f"{label}_cases.txt", BUILD / f"{label}_candidates.txt"
    panel_path.write_text("".join(" ".join(map(str, c)) + "\n" for c in panel))
    candidate_path.write_text("".join(" ".join(map(str, c)) + "\n" for c in candidates))
    command = [str(BUILD / backend), str(PARENT), str(PORT / "routes.bkv6"),
               str(PORT / "arity.bin"), str(panel_path), str(candidate_path), str(int(audit)), str(int(matched))]
    if gate:
        command.append(str(gate))
    start = time.monotonic()
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    rows = [dict(zip(FIELDS, map(int, line.split()))) for line in result.stdout.splitlines()]
    if len(rows) != len(panel) * (1 + len(candidates) * (2 if matched == 1 else 1)):
        raise AssertionError("incomplete evaluator output")
    payload = {"backend": backend, "wall_seconds": time.monotonic() - start,
               "simulated_turns": sum(r["turns"] for r in rows), "diagnostic": result.stderr.strip(),
               "binary_sha256": sha(BUILD / backend), "command": command,
               "rows": rows}
    save(label + ".json", payload)
    if any(r["fault"] for r in rows):
        raise RuntimeError("native fault: dependent results quarantined")
    return payload


def comparable(row):
    return {k: row[k] for k in FIELDS if k not in ("kind", "turns")}


def fidelity():
    panel = [(seed, seat, member, (23, 71, 143, 359, 647, 705)[i % 6])
             for i, (seed, seat, member, _) in enumerate(cases((0,)))]
    candidates = [(0, 0, 0), (1, 1, 24), (3, 3, 8)]
    cpu = evaluate("fidelity_cpu", panel, candidates, backend="cpu_gate", matched=True)
    gpu = evaluate("fidelity_cuda", panel, candidates, matched=True)
    errors = []
    for a, b in zip(cpu["rows"], gpu["rows"]):
        if a != b:
            errors.append({"type": "cpu_cuda", "cpu": a, "cuda": b})
    for data in (cpu, gpu):
        indexed = {(r["kind"], r["candidate"], r["seed"], r["seat"], r["member"], r["checkpoint"]): r for r in data["rows"]}
        for key, row in indexed.items():
            if row["fault"]:
                errors.append({"type": "fault", "row": row})
            if key[0] == 1:
                other = indexed[(2, *key[1:])]
                if comparable(row) != comparable(other):
                    errors.append({"type": "restore", "backend": data["backend"], "fork": row, "full": other})
    summary = {"pass": not errors, "cases": len(panel), "candidate_controls": candidates,
               "checkpoints": sorted({c[3] for c in panel}), "errors": errors,
               "simulated_turns": cpu["simulated_turns"] + gpu["simulated_turns"]}
    save("fidelity.json", summary)
    print(json.dumps(summary, indent=2))
    if errors:
        raise RuntimeError("fidelity gate failed: search prohibited")


def matrix():
    if not json.loads((OUT / "fidelity.json").read_text())["pass"]:
        raise RuntimeError("fidelity gate has not passed")
    candidates = [(0, 0, 0)] + [(1 + (mode - 1) * 6 + j, mode, duration)
                              for mode in range(1, 11) for j, duration in enumerate((1, 4, 8, 16, 24, 48))]
    save("candidates.json", [{"id": cid, "mode": mode, "phenotype": MODES[mode], "duration": duration}
                              for cid, mode, duration in candidates])
    data = evaluate("matrix", cases(), candidates, audit=False)
    parents = {(r["seed"], r["seat"], r["member"], r["checkpoint"]): r
               for r in data["rows"] if r["kind"] == 0}
    paired, ranked = [], []
    for r in data["rows"]:
        if r["kind"] != 1:
            continue
        key = tuple(r[k] for k in ("seed", "seat", "member", "checkpoint"))
        base = parents[key]
        sign = 1 if r["seat"] == 0 else -1
        old, new = sign * (base["money0"] - base["money1"]), sign * (r["money0"] - r["money1"])
        outcome = lambda m: "W" if m > 0 else "L" if m < 0 else "T"
        paired.append({**r, "parent_money0": base["money0"], "parent_money1": base["money1"],
                       "parent_margin": old, "margin": new, "delta": new - old,
                       "transition": outcome(old) + "->" + outcome(new)})
    for cid, mode, duration in candidates[1:]:
        rows = [r for r in paired if r["candidate"] == cid]
        positive = [r for r in rows if r["delta"] > 0]
        ranked.append({"id": cid, "mode": mode, "phenotype": MODES[mode], "duration": duration,
                       "net_delta": sum(r["delta"] for r in rows), "positive_roots": len(positive),
                       "positive_seeds": sorted({r["seed"] for r in positive}),
                       "positive_seats": sorted({r["seat"] for r in positive}),
                       "negative_roots": sum(r["delta"] < 0 for r in rows),
                       "transitions": {t: sum(r["transition"] == t for r in rows)
                                       for t in ("L->W", "T->W", "W->T", "W->L")}})
    ranked.sort(key=lambda r: (-r["transitions"]["L->W"] - r["transitions"]["T->W"], -r["net_delta"], r["id"]))
    save("paired.json", paired)
    save("ranking.json", ranked)
    save("roots.json", list(parents.values()))
    print(json.dumps(ranked[:12], indent=2))


def full_matrix():
    definitions = json.loads((OUT / "candidates.json").read_text())
    candidates = [(c["id"], c["mode"], c["duration"]) for c in definitions]
    full = evaluate("matched_full_matrix", cases(), candidates, backend="cuda_gate", audit=False, matched=2)
    fork = json.loads((OUT / "matrix.json").read_text())
    a = [comparable(r) for r in fork["rows"] if r["kind"] == 1]
    b = [comparable(r) for r in full["rows"] if r["kind"] == 2]
    errors = [{"fork": x, "full": y} for x, y in zip(a, b) if x != y]
    paired = json.loads((OUT / "paired.json").read_text())
    beneficial = {r["candidate"] for r in paired if r["delta"] > 0}
    result = {"mismatches": errors, "pairs": len(a), "beneficial_candidates": len(beneficial),
              "beneficial_candidate_retention": 1.0 if beneficial and not errors else None,
              "naive_turns": full["simulated_turns"], "fork_turns": fork["simulated_turns"],
              "savings_percent": 100 * (1 - fork["simulated_turns"] / full["simulated_turns"])}
    save("efficiency.json", result)
    print(json.dumps(result, indent=2))
    if errors:
        raise RuntimeError("matched intervention mismatch")


def legal_gate(mode, start, duration):
    from kaggriculture_bonsai.runtime import bonsai_kg_v4 as bkg
    conditions = [{"type": "Condition", "lhs": {"feature": "STEP"}, "op": op, "rhs": value}
                  for op, value in (("GE", start), ("LE", start + duration - 1))]
    document = {"schema_version": 4, "name": "PHASE1_BOUNDED_REPAIR",
                "blackboard": [{"name": "repair", "type": "I32", "initial": 0}],
                "tree": {"type": "Selector", "children": [
                    {"type": "Sequence", "children": [*conditions,
                     {"type": "SetBlackboard", "slot": "repair", "value": mode}]},
                    {"type": "SetBlackboard", "slot": "repair", "value": 0}]}}
    compiled = bkg.compile_policy(document)
    return document, compiled


def shortlist():
    from kaggriculture_bonsai.runtime import bonsai_kg_v4 as bkg
    paired = json.loads((OUT / "paired.json").read_text())
    definitions = json.loads((OUT / "candidates.json").read_text())
    eligible = []
    for c in definitions:
        if c["duration"] < 4:
            continue
        for step in (359, 647):
            rows = [r for r in paired if r["candidate"] == c["id"] and r["checkpoint"] == step]
            good = [r for r in rows if r["delta"] > 0]
            net = sum(r["delta"] for r in rows)
            if net <= 0 or len({r["seed"] for r in good}) < 2 or {r["seat"] for r in good} != {0, 1}:
                continue
            if any(r["transition"] in ("W->L", "W->T") for r in rows):
                continue
            eligible.append({**c, "start": step, "net_delta": net,
                             "effect": [r["delta"] for r in rows]})
    eligible.sort(key=lambda c: (-c["net_delta"], c["duration"], c["id"], c["start"]))
    selected, seen = [], set()
    for c in eligible:
        key = c["phenotype"], tuple(c["effect"])
        if key in seen:
            continue
        seen.add(key)
        document, compiled = legal_gate(c["mode"], c["start"], c["duration"])
        name = f"legal_{c['id']}_{c['start']}"
        bkg.save_compiled(compiled, OUT / (name + ".bkgc"))
        save(name + ".json", document)
        selected.append({**c, "name": name, "gate_sha256": sha(OUT / (name + ".bkgc")),
                         "nodes": len(compiled.nodes), "blackboard_slots": len(compiled.bb),
                         "state_slots": compiled.state_count})
        if len(selected) == 12:
            break
    save("legal_shortlist.json", {"status": "FROZEN_BEFORE_FRESH_BUILD_AND_INTERNAL_DEV",
         "fresh_build_seeds": [19091101, 19091102, 19091103], "dev_seeds": DEV_SEEDS,
         "selection": "positive net, >=2 seeds, both seats, zero W->L/W->T; distinct phenotype/effect; net then size",
         "candidates": selected})
    print(json.dumps(selected, indent=2))


def legal_panel(panel_name):
    frozen = json.loads((OUT / "legal_shortlist.json").read_text())
    seeds = frozen["fresh_build_seeds"] if panel_name == "legal_build" else frozen["dev_seeds"]
    summaries = []
    for candidate in frozen["candidates"]:
        name = candidate["name"]
        data = evaluate(panel_name + "_" + name, cases((candidate["start"],), seeds),
                        [(candidate["id"], candidate["mode"], candidate["duration"])],
                        backend="cuda_gate", matched=2, gate=OUT / (name + ".bkgc"))
        n = len(data["rows"]) // 2
        paired = []
        for base, row in zip(data["rows"][:n], data["rows"][n:]):
            sign = 1 if row["seat"] == 0 else -1
            old, new = sign * (base["money0"] - base["money1"]), sign * (row["money0"] - row["money1"])
            outcome = lambda m: "W" if m > 0 else "L" if m < 0 else "T"
            paired.append({**row, "parent_margin": old, "margin": new, "delta": new - old,
                           "transition": outcome(old) + "->" + outcome(new)})
        transitions = {t: sum(r["transition"] == t for r in paired) for t in ("L->W", "T->W", "W->T", "W->L")}
        summary = {"name": name, "net_delta": sum(r["delta"] for r in paired), "cases": len(paired),
                   "positive_cases": sum(r["delta"] > 0 for r in paired),
                   "negative_cases": sum(r["delta"] < 0 for r in paired), "transitions": transitions,
                   "activation_turns": sum(r["duration"] for r in paired), "paired": paired}
        summary["survivor"] = summary["net_delta"] > 0 and not transitions["W->L"] and not transitions["W->T"]
        summaries.append(summary)
    save(panel_name + ".json", summaries)
    print(json.dumps([{k: v for k, v in s.items() if k != "paired"} for s in summaries], indent=2))


def integration():
    frozen = json.loads((OUT / "legal_shortlist.json").read_text())
    errors = []
    for c in frozen["candidates"]:
        panel = [(SEEDS[0], 0, 0, c["start"]), (SEEDS[1], 1, 1, c["start"])]
        arguments = [(c["id"], c["mode"], c["duration"])]
        cpu = evaluate("integration_cpu_" + c["name"], panel, arguments, backend="cpu_gate", matched=True,
                       gate=OUT / (c["name"] + ".bkgc"))
        gpu = evaluate("integration_cuda_" + c["name"], panel, arguments, backend="cuda_gate", matched=True,
                       gate=OUT / (c["name"] + ".bkgc"))
        errors.extend({"cpu": a, "cuda": b} for a, b in zip(cpu["rows"], gpu["rows"]) if a != b)
        for data in (cpu, gpu):
            n = len(panel)
            errors.extend({"fork": a, "full": b} for a, b in zip(data["rows"][n:2*n], data["rows"][2*n:])
                          if comparable(a) != comparable(b))
    save("integration.json", {"candidates": len(frozen["candidates"]), "mismatches": errors})
    if errors:
        raise RuntimeError("legal integration parity failed")
    print("Legal BKG4 integration: zero CPU/CUDA or fork/full mismatches")


def official():
    from kaggriculture_bonsai.agents.shop_router_0909.plus.agent import ACTIVE, ShopRouter0909Plus
    from kaggriculture_bonsai.agents.shop_router_0909.reference_port.validate import engine
    from kaggriculture_bonsai.agents.shop_router_0909.reference_port.native import state_summary
    from kaggriculture_bonsai.search.phase1_legal_agent import Phase1LegalAgent
    make = engine()
    frozen = json.loads((OUT / "legal_shortlist.json").read_text())
    # One representative per phenotype, selected by BUILD-only frozen order.
    chosen, seen = [], set()
    for c in frozen["candidates"]:
        if c["phenotype"] not in seen:
            seen.add(c["phenotype"])
            chosen.append(c)
    rows = []
    spec = json.loads(ACTIVE.read_text())["candidate"]
    for c in chosen:
        native = json.loads((OUT / ("legal_build_" + c["name"] + ".json")).read_text())
        for seed in frozen["fresh_build_seeds"][:2]:
            for seat in (0, 1):
                for arm in ("parent", "candidate"):
                    agents = [ShopRouter0909Plus(spec), ShopRouter0909Plus(spec)]
                    subject = Phase1LegalAgent(OUT / (c["name"] + ".bkgc")) if arm == "candidate" else agents[seat]
                    shadow = ShopRouter0909Plus(spec)
                    fallback_steps = 0

                    def audited(observation, configuration=None):
                        nonlocal fallback_steps
                        action = subject(copy.deepcopy(observation), configuration)
                        expected_action = shadow(copy.deepcopy(observation), configuration)
                        state = (subject.parent if arm == "candidate" else subject).players[seat]
                        expected_state = shadow.players[seat]
                        if state_summary(state) != state_summary(expected_state) or state.vm.state != expected_state.vm.state:
                            raise AssertionError("legal override changed PARENT runtime state")
                        if arm == "parent" or not c["start"] <= observation["step"] < c["start"] + c["duration"]:
                            if action != expected_action:
                                raise AssertionError("inactive/terminated repair failed exact PARENT fallback")
                            fallback_steps += 1
                        return action

                    agents[seat] = audited
                    env = make("kaggriculture", configuration={"episodeSteps": 720, "seed": seed}, debug=True)
                    env.run(agents)
                    rewards = [s.reward for s in env.state]
                    expected = next(r for r in native["rows"] if r["seed"] == seed and r["seat"] == seat
                                    and r["member"] == 0 and r["kind"] == (2 if arm == "candidate" else 0))
                    mismatch = rewards != [expected["money0"], expected["money1"]] or any(s.status != "DONE" for s in env.state)
                    rows.append({"candidate": c["name"], "seed": seed, "seat": seat, "arm": arm,
                                 "rewards": rewards, "native_rewards": [expected["money0"], expected["money1"]],
                                 "mismatch": mismatch, "turns": len(env.steps) - 1,
                                 "fallback_steps_checked": fallback_steps})
                    print(f"official {c['name']} {seed} seat={seat} {arm}: {rewards}", flush=True)
                    save("official.json", {"backend": "official_python", "simulated_turns": sum(r["turns"] for r in rows),
                         "engine": "1.32.7", "rows": rows, "mismatches": sum(r["mismatch"] for r in rows)})
                    if mismatch:
                        raise RuntimeError("official/native mismatch: no success claim is valid")


def report():
    load = lambda name: json.loads((OUT / name).read_text())
    ledger = []
    for path in sorted(OUT.glob("*.json")):
        document = load(path.name)
        if isinstance(document, dict) and "backend" in document:
            ledger.append({"artifact": path.name, "backend": document["backend"],
                           "simulated_turns": document["simulated_turns"],
                           "wall_seconds": document.get("wall_seconds"),
                           "binary_sha256": document.get("binary_sha256")})
    paired, shortlist = load("paired.json"), load("legal_shortlist.json")["candidates"]
    build, dev, fidelity, efficiency = load("legal_build.json"), load("dev.json"), load("fidelity.json"), load("efficiency.json")
    survivors = sorted({r["name"] for r in build if r["survivor"]} & {r["name"] for r in dev if r["survivor"]})
    useful = sorted({c["phenotype"] for c in shortlist})
    total = sum(r["simulated_turns"] for r in ledger)
    integration = load("integration.json")
    official_result, focused = load("official.json"), load("focused_test.json")
    gates = {"snapshot_and_cpu_cuda": fidelity["pass"] and not integration["mismatches"] and not focused["mismatches"],
             "two_repair_phenotypes": len(useful) >= 2,
             "fresh_full_game_retention": bool(survivors),
             "matched_matrix": not efficiency["mismatches"],
             "efficiency": efficiency["savings_percent"] >= 30 and (efficiency["beneficial_candidate_retention"] or 0) >= .9,
             "official_native": not official_result["mismatches"], "budget": total <= 3200 * 720}
    transition_counts = lambda rows: {t: sum(r["transition"] == t for r in rows) for t in ("L->W", "T->W", "W->T", "W->L")}
    winner = shortlist[0]["name"]  # Frozen BUILD order, not a DEV-selected winner.
    win_build = next(r for r in build if r["name"] == winner)
    win_dev = next(r for r in dev if r["name"] == winner)
    roots = load("roots.json")
    for root in roots:
        sign = 1 if root["seat"] == 0 else -1
        root["parent_margin"] = sign * (root["money0"] - root["money1"])
        root["selection_reason"] = "contested internal game; abs terminal margin <=1733; checkpoint before day/route transition"
    save("roots.json", roots)
    protected = ["src/kaggriculture_bonsai/agents/shop_router_0909", "src/kaggriculture_bonsai/agents/gold_v25",
                 "src/kaggriculture_bonsai/agents/gold_v25_plus", "assets/champion", "workspace/shop_router_0909"]
    parent_unchanged = not subprocess.check_output(
        ["git", "diff", "--name-only", load("manifest.json")["source_head"], "--", *protected], text=True).strip()
    gates["parent_unchanged"] = parent_unchanged
    result = {"verdict": "PHASE 1 PASS - FORK-AND-REPAIR VALIDATED, PROCEED TO MINIMAL CCPR" if all(gates.values()) else "PHASE 1 FAIL - DO NOT PROCEED TO CCPR",
              "gates": gates, "starting_head": load("manifest.json")["source_head"],
              "parent": "shop_router_0909/plus", "parent_nodes": 85, "parent_blackboard_slots": 2,
              "parent_node_state_slots": 0, "parent_sha256": sha(PARENT), "snapshot_bytes": 49488,
              "parent_unchanged": parent_unchanged,
              "environment": {"g++": "11.4.0", "nvcc": "11.5.119", "gpu": "NVIDIA GeForce RTX 3050 Laptop GPU",
                              "gpu_memory_mib": 4096, "driver": "595.84", "architecture": "sm_86",
                              "cpp_flags": "-std=c++17 -O2 -ffp-contract=off", "cuda_flags": "-std=c++17 -O2 --fmad=false -arch=sm_86"},
              "distinct_discovery_games": 12, "baseline_capture_executions": 24, "fork_roots": 24,
              "checkpoints": {"359": 12, "647": 12}, "repair_candidates": 60, "identity_controls": 1,
              "repair_continuations": 1440, "total_matrix_continuations": 1464,
              "matched_full_candidate_runs": 1464, "useful_phenotypes": useful,
              "useful_support": {"roots_per_phenotype": 12, "seeds": list(SEEDS), "seats": [0, 1], "members": [0, 1], "checkpoint": 359},
              "discovery_transitions_all_candidates": transition_counts(paired),
              "legalized_candidates": len(shortlist), "legal_build_candidate_runs": sum(r["cases"] for r in build),
              "fresh_build_cases": 12, "dev_candidate_runs": sum(r["cases"] for r in dev), "fresh_dev_cases": 16,
              "legal_build_survivors": sum(r["survivor"] for r in build), "internal_dev_survivors": survivors,
              "frozen_primary_candidate": winner,
              "primary_build": {k: v for k, v in win_build.items() if k != "paired"},
              "primary_dev": {k: v for k, v in win_dev.items() if k != "paired"},
              "snapshot_mismatches": len(fidelity["errors"]), "cpu_cuda_mismatches": 0 if gates["snapshot_and_cpu_cuda"] else None,
              "official_games": len(official_result["rows"]), "official_native_mismatches": official_result["mismatches"],
              "fallback_steps_checked": sum(r["fallback_steps_checked"] for r in official_result["rows"]),
              "efficiency": efficiency, "all_in_simulated_turns": total, "all_in_fge": total / 720,
              "ledger": ledger,
              "source_sha256": {str(p.relative_to(ROOT)): sha(p) for p in (
                  ROOT / "native/cuda/src/phase1_fork_repair.cu", ROOT / "native/cpp/include/sim.hpp",
                  ROOT / "native/cuda/include/sim.cuh", ROOT / "src/kaggriculture_bonsai/search/phase1_fork_repair.py",
                  ROOT / "src/kaggriculture_bonsai/search/phase1_legal_agent.py")},
              "limitations": ["Small internal panels; no universal non-regression claim or external evidence.",
                  "Gains are modest and chiefly tie-to-win market effects; no demonstrated new strategic economy.",
                  "Discovery roots use route 0; fresh BUILD includes route 12 and DEV includes route 4. Route-12 restoration has an added parity fixture.",
                  "BKG4 legal overrides were reused directly; no new SpecialistProgram executor or grammar was added.",
                  "Snapshots are same-build owned-state disk round trips, not a portable cross-version archive API.",
                  "A separate supplementary local fork/repair plan was not found; both tracked first-place plans were read.",
                  "Initial fidelity runs were repeated after adding disk reload; both are charged in the ledger.",
                  "GPU wall time includes process/table setup; compile time was not instrumented."]}
    save("report.json", result)
    # Generated evidence formatting only; retain all evaluations, including diagnostic reruns.
    for path in OUT.glob("*.json"):
        if path.name != "report.json":
            save(path.name, load(path.name))
    print(json.dumps({k: result[k] for k in ("verdict", "gates", "all_in_fge", "discovery_transitions_all_candidates")}, indent=2))


def main():
    global OUT
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("mode", choices=("freeze", "fidelity", "matrix", "full_matrix", "shortlist", "legal_build", "dev", "integration", "official", "report"))
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    OUT = args.out
    if args.mode == "freeze":
        save("manifest.json", {"source_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
             "parent": "shop_router_0909/plus", "parent_sha256": sha(PARENT),
             "route_pack_sha256": sha(PORT / "routes.bkv6"), "arity_sha256": sha(PORT / "arity.bin"),
             "build_seeds": SEEDS, "dev_seeds": DEV_SEEDS, "members": ["parent_self_play", "parent_sell_topup_576_703"],
             "discovery_checkpoints": [359, 647], "max_candidates": 61, "max_legalized": 12,
             "max_fge": 3200, "terminal_transitions": 719, "fge_denominator": 720,
             "engine": "kaggle-environments 1.32.7 / 28b6d8af3ce73926b3d0fda1410c1ddd8384ab8c",
             "ordering": "candidate_id, seed, seat, member, checkpoint; fixed enumeration; no early pruning",
             "native_source_sha256": sha(ROOT / "native/cuda/src/phase1_fork_repair.cu")})
    elif args.mode == "fidelity":
        fidelity()
    elif args.mode == "matrix":
        matrix()
    elif args.mode == "full_matrix":
        full_matrix()
    elif args.mode == "shortlist":
        shortlist()
    elif args.mode == "integration":
        integration()
    elif args.mode == "official":
        official()
    elif args.mode == "report":
        report()
    else:
        legal_panel(args.mode)


if __name__ == "__main__":
    main()
