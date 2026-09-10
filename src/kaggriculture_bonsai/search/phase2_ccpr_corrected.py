"""Frozen trajectory-diverse comparison; no post-guard or DEV repair."""
from __future__ import annotations

import argparse
import itertools
import subprocess

from kaggriculture_bonsai.search import phase2_ccpr as c
from kaggriculture_bonsai.search import phase2d_generalization as d

OUT = c.ROOT / "workspace/self_improvement/phase2_ccpr_corrected"
STEPS = (335, 359, 431, 575, 647, 695)
MODES = {1: "sell_topup", 2: "worker_deposit", 4: "market_liquidation", 6: "value_sort"}
ARMS = ("ccpr", "control")


def configure():
    d.OUT = OUT
    d.configure()


def groups():
    result = []
    # Each block has twelve two-root games and twelve single-root games.
    for i in range(48):
        j = i % 24
        if j < 12:
            k = j // 4
            checkpoints = (STEPS[2 * k], STEPS[2 * k + 1])
        else:
            checkpoints = (STEPS[(j - 12) // 2],)
        result.append({"seed": 49091001 + i, "seat": j % 2, "member": (j // 2) % 2,
                       "split": "discovery" if i < 24 else "guard", "checkpoints": checkpoints})
    return result


def panel(split, start):
    return [(g["seed"], g["seat"], g["member"], start) for g in c.load("groups.json")
            if g["split"] == split and start in g["checkpoints"]]


def control_plan(initial, roots):
    """Independent broad-coverage gates; outcome fields never select a proposal."""
    vocabulary = c.vocabulary(roots)
    pools = {1: [], 2: []}
    for size in pools:
        for predicates in itertools.combinations(vocabulary, size):
            combined = initial["predicates"] + [list(p) for p in predicates]
            if len(combined) > 2 or len({p[0] for p in combined}) != len(combined):
                continue
            kept = [r for r in roots if c.matches(r["features"], combined)]
            if not (2 <= len(kept) < len(roots)) or {r["seat"] for r in kept} != {0, 1}:
                continue
            p = c.program(initial["start"], initial["mode"], initial["duration"], combined)
            pools[size].append((-len(kept), c.identity(p), p))
        pools[size].sort(key=lambda x: (x[0], x[1]))
    durations = [v for v in (24, 16, 8, 4) if v < initial["duration"]]
    singles = [p for _, _, p in pools[1]]
    doubles = [p for _, _, p in pools[2]]
    ordered = [initial]
    # Interleave STOP, WHEN, STOP, STOP, two-clause WHEN, STOP.
    if durations:
        ordered.append(c.program(initial["start"], initial["mode"], durations[0], initial["predicates"]))
    ordered.extend(singles[:1])
    ordered.extend(c.program(initial["start"], initial["mode"], v, initial["predicates"]) for v in durations[1:3])
    ordered.extend(doubles[:1] or singles[1:2])
    ordered.extend(c.program(initial["start"], initial["mode"], v, initial["predicates"]) for v in durations[3:])
    ordered.extend(singles[1:] + doubles[1:])
    unique = {}
    for p in ordered:
        unique.setdefault(c.identity(p), p)
    return list(unique.values())[:7]


def stop_reason(rows):
    if d.group_metrics(rows)["eligible"]:
        return "SAFE_USEFUL"
    if all(r["delta"] == 0 for r in rows):
        return "ZERO_VALUE"
    return None


def freeze():
    if (OUT / "config.json").exists():
        raise RuntimeError("epoch already frozen")
    parent = c.bkg.load_compiled(c.PARENT)
    assert (len(parent.nodes), len(parent.bb), parent.state_count) == (85, 2, 0)
    initial = [c.program(start, mode, min(48, 719 - start)) for start in STEPS for mode in MODES]
    initial.append(c.program(359, 1, 48, [(6, "LE", 241)]))
    c.save("config.json", {"head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "parent_sha256": c.sha(c.PARENT), "routes_sha256": c.sha(c.PORT / "routes.bkv6"),
        "arity_sha256": c.sha(c.PORT / "arity.bin"), "parent_nodes": len(parent.nodes),
        "parent_blackboard": len(parent.bb), "parent_node_state": parent.state_count,
        "initial_hypotheses": initial, "reference_id": c.identity(initial[-1]), "steps": STEPS,
        "effects": MODES, "max_refinements": 6, "evaluations_per_lineage": 7, "shortlist_per_arm": 8,
        "budget_fge": 9600, "dev_seeds": list(range(49092001, 49092033)),
        "stopping": "both arms independently stop at first group-safe useful incumbent, zero terminal effect, exhaustion or seven evaluations; realized budgets reported, no padding",
        "control": "same initial programs and unlabeled quartile vocabulary; independent duration/one-clause/two-clause schedule, broad observable support then semantic hash; freeze before outcomes",
        "ccpr": "unchanged WHEN-first/STOP proposer; unchanged Phase-2D seed-group safety/rank; max two clauses",
        "selection": "same group eligibility, one best per lineage, distinct effect vectors; one representative per action mechanism first, then rank; freeze <=8 per arm before guard; no replacements",
        "pass": ">=2 newly refined CCPR DEV survivors spanning >=2 action mechanisms, >=3 improved causal lineages, fresh outcome flip, multi-seed/seat/member support, zero parity failures, material advantage without worse shortlist safety",
        "advantage": "more unique DEV survivors or mechanisms at no worse survivor/FGE rate; or more unique DEV flips at no greater node/duration maxima; or >=2x best survivor net at comparable footprint and >=control flips; shared initial/reference survivors cannot alone prove CCPR",
        "source_sha256": {str(p.relative_to(c.ROOT)): c.sha(p) for p in (
            c.ROOT / "native/cuda/src/phase1_fork_repair.cu",
            c.ROOT / "src/kaggriculture_bonsai/search/phase2_ccpr.py",
            c.ROOT / "src/kaggriculture_bonsai/search/phase2d_generalization.py",
            c.ROOT / "src/kaggriculture_bonsai/search/phase2_ccpr_corrected.py")},
        "binaries": {b: c.sha(c.BUILD / b) for b in ("cpu", "cuda")}})
    c.save("groups.json", groups())
    c.save("ledger.json", [])


def fidelity():
    if (OUT / "fidelity.json").exists():
        raise RuntimeError("fidelity already recorded")
    results = [d.check_integration(c.program(start, mode, 8), f"fidelity_{mode}")
               for start, mode in ((359, 1), (431, 2), (695, 4), (575, 6))]
    c.save("fidelity.json", results)


def discovery():
    assert all(r["mismatches"] == 0 for r in c.load("fidelity.json"))
    if (OUT / "control_plan.json").exists():
        raise RuntimeError("discovery frozen; no repeated/exposed-epoch search")
    roots = {start: d.evaluate(c.program(start, 0), panel("discovery", start), f"capture_discovery_{start}", kind=4)
             for start in STEPS}
    c.save("roots.json", {"discovery": roots})
    initial = c.load("config.json")["initial_hypotheses"]
    plans = {c.identity(p): control_plan(p, roots[p["start"]]) for p in initial}
    c.save("control_plan.json", plans)  # Written before the first candidate outcome.
    records, lineages = [], []
    for p0 in initial:
        origin = c.identity(p0)
        base = roots[p0["start"]]
        vocabulary = c.vocabulary(base)
        for arm in ARMS:
            current, current_rows, tried, chain = p0, None, set(), []
            status = "SATURATED"
            for generation in range(7):
                ancestor, witnesses, operation = None, [], "INITIAL"
                if generation == 0:
                    p = p0
                elif arm == "control":
                    if generation >= len(plans[origin]):
                        break
                    p, operation = plans[origin][generation], "INDEPENDENT"
                else:
                    proposal = c.refine(current, current_rows, vocabulary, tried)
                    if proposal is None:
                        break
                    p, operation, witnesses = proposal
                    ancestor = c.identity(current)
                pid = c.identity(p)
                tried.add(pid)
                rows = c.pair(d.evaluate(p, panel("discovery", p["start"]), f"{arm}_discovery_{origin[:8]}_{generation}"), base)
                improved = current_rows is not None and d.rank(rows, p) < d.rank(current_rows, current)
                records.append({"arm": arm, "origin": origin, "id": pid, "program": p, "generation": generation,
                    "operation": operation, "parent_hypothesis": ancestor, "counterexample_witnesses": witnesses,
                    "improved": improved, "metrics": d.group_metrics(rows), "rows": rows})
                chain.append(pid)
                if current_rows is None or improved:
                    current, current_rows = p, rows
                status = stop_reason(current_rows) or "SATURATED"
                if status != "SATURATED":
                    break
            lineages.append({"arm": arm, "origin": origin, "hypotheses": chain, "best": c.identity(current), "status": status})
    c.save("hypotheses.json", records)
    c.save("lineages.json", lineages)
    selected = []
    for arm in ARMS:
        eligible = [r for r in records if r["arm"] == arm and r["metrics"]["eligible"]]
        eligible.sort(key=lambda r: d.rank(r["rows"], r["program"]))
        best_by_mode = []
        for mode in MODES:
            best_by_mode.extend(next(([r] for r in eligible if r["program"]["mode"] == mode), []))
        seen = set()
        for r in best_by_mode + eligible:
            effect = (r["program"]["mode"], r["program"]["start"], tuple(x["delta"] for x in r["rows"]))
            if effect in seen:
                continue
            seen.add(effect)
            selected.append({k: r[k] for k in ("arm", "origin", "id", "program", "generation")})
            if len(seen) == 8:
                break
    c.save("frozen_shortlist.json", selected)
    print("discovery", [(a, sum(r["arm"] == a for r in records), sum(s["arm"] == a for s in selected)) for a in ARMS], flush=True)


def guard():
    if (OUT / "guard.json").exists():
        raise RuntimeError("guard already opened")
    selected = c.load("frozen_shortlist.json")
    roots = c.load("roots.json")
    roots["guard"] = {str(start): d.evaluate(c.program(start, 0), panel("guard", start), f"capture_guard_{start}", kind=4)
                      for start in STEPS}
    c.save("roots.json", roots)
    results = []
    for r in selected:
        p = r["program"]
        rows = c.pair(d.evaluate(p, panel("guard", p["start"]), f"{r['arm']}_guard_{r['id'][:10]}"), roots["guard"][str(p["start"])])
        results.append({**r, "metrics": d.group_metrics(rows), "rows": rows})
    c.save("guard.json", results)
    c.save("legalized.json", [{k: r[k] for k in ("arm", "origin", "id", "program", "generation")}
                              for r in results if r["metrics"]["eligible"]])
    print("guard survivors", [(a, sum(r["arm"] == a and r["metrics"]["eligible"] for r in results)) for a in ARMS], flush=True)


def validate():
    if (OUT / "dev.json").exists():
        raise RuntimeError("DEV already opened; no retuning")
    integrations, build, dev = [], [], []
    selected = c.load("legalized.json")
    for r in selected:
        p = r["program"]
        integrations.append({"arm": r["arm"], **d.check_integration(p, f"integration_{r['arm']}_{r['id'][:10]}")})
        cases = panel("guard", p["start"])
        full = d.evaluate(p, cases, f"{r['arm']}_legal_build_{r['id'][:10]}", kind=2, audit=True)
        n = len(cases)
        paired = c.pair(full[n:], full[:n])
        fork = next(g["rows"] for g in c.load("guard.json") if g["arm"] == r["arm"] and g["id"] == r["id"])
        keys = ("money0", "money1", "delta", "changes", "duration")
        assert [[x[k] for k in keys] for x in paired] == [[x[k] for k in keys] for x in fork]
        build.append({**r, "metrics": d.group_metrics(paired), "rows": paired})
        _, document, compiled = c.compile_gate(p)
        c.bkg.save_compiled(compiled, OUT / (r["id"] + ".bkgc"))
        integrations[-1].update({"nodes": len(compiled.nodes), "blackboard": len(compiled.bb),
                                "node_state": compiled.state_count, "gate": document})
    c.save("integration.json", integrations)
    c.save("legal_build.json", build)
    for r in selected:
        p = r["program"]
        cases = [(seed, i % 2, (i // 2) % 2, p["start"]) for i, seed in enumerate(c.load("config.json")["dev_seeds"])]
        full = d.evaluate(p, cases, f"{r['arm']}_dev_{r['id'][:10]}", kind=2, audit=True)
        n = len(cases)
        paired = c.pair(full[n:], full[:n])
        m = d.group_metrics(paired)
        dev.append({**r, "metrics": m, "survivor": m["eligible"], "rows": paired})
        print("DEV", r["arm"], r["id"][:10], m["net"], m["transitions"], m["eligible"], flush=True)
    c.save("dev.json", dev)


def summary():
    records, dev, ledger = (c.load(n) for n in ("hypotheses.json", "dev.json", "ledger.json"))
    results, examples = {}, []
    for arm in ARMS:
        rr = [r for r in records if r["arm"] == arm]
        vv = [r for r in dev if r["arm"] == arm]
        survivors = [r for r in vv if r["survivor"]]
        turns = sum(l["turns"] for l in ledger if l["label"].startswith(arm + "_discovery_"))
        effects = {(r["program"]["mode"], tuple(x["delta"] for x in r["rows"])) for r in survivors}
        flips = {(x["seed"], x["seat"], x["member"]) for r in survivors for x in r["rows"] if x["transition"] in ("L->W", "T->W")}
        results[arm] = {"candidates": len(rr), "continuations": sum(len(r["rows"]) for r in rr),
            "discovery_turns": turns, "discovery_fge": turns / 720,
            "safe_useful_build": sum(r["metrics"]["eligible"] for r in rr),
            "guard_tested": sum(r["arm"] == arm for r in c.load("guard.json")),
            "guard_survivors": sum(r["arm"] == arm for r in c.load("legalized.json")),
            "legalized": len(vv), "dev_survivors": len(survivors), "unique_dev_effects": len(effects),
            "mechanisms": sorted({MODES[r["program"]["mode"]] for r in survivors}),
            "newly_refined_survivors": [r["id"] for r in survivors if r["generation"] > 0 and arm == "ccpr"],
            "unique_dev_flips": len(flips), "best_net": max((r["metrics"]["net"] for r in survivors), default=0),
            "max_nodes": max((6 + len(r["program"]["predicates"]) for r in survivors), default=0),
            "max_duration": max((r["program"]["duration"] for r in survivors), default=0),
            "dev_transitions": {t: sum(r["metrics"]["transitions"][t] for r in vv) for t in c.TRANSITIONS},
            "survivor_transitions": {t: sum(r["metrics"]["transitions"][t] for r in survivors) for t in c.TRANSITIONS},
            "discovery_transitions": {t: sum(r["metrics"]["transitions"][t] for r in rr) for t in c.TRANSITIONS}}
    for r in records:
        if r["arm"] != "ccpr" or not r["improved"]:
            continue
        parent = next(x for x in records if x["arm"] == "ccpr" and x["origin"] == r["origin"] and x["id"] == r["parent_hypothesis"])
        if [x["delta"] for x in parent["rows"]] != [x["delta"] for x in r["rows"]]:
            examples.append({"origin": r["origin"], "id": r["id"], "operation": r["operation"], "program": r["program"],
                             "before": parent["metrics"], "after": r["metrics"], "witnesses": r["counterexample_witnesses"]})
    a, b = results["ccpr"], results["control"]
    no_worse_safety = all(a["dev_transitions"][t] <= b["dev_transitions"][t] for t in ("W->L", "W->T", "T->L"))
    yield_better = a["unique_dev_effects"] > b["unique_dev_effects"] and a["unique_dev_effects"] * b["discovery_turns"] >= b["unique_dev_effects"] * a["discovery_turns"]
    footprint = a["max_nodes"] <= b["max_nodes"] + 1 and a["max_duration"] <= max(b["max_duration"], 4)
    advantage = no_worse_safety and (yield_better or
        (footprint and a["unique_dev_flips"] > b["unique_dev_flips"]) or
        (footprint and a["best_net"] >= max(1, 2 * b["best_net"]) and a["unique_dev_flips"] >= b["unique_dev_flips"]))
    derived = [r for r in dev if r["arm"] == "ccpr" and r["generation"] > 0 and r["survivor"]]
    support = any({x["seat"] for x in r["rows"] if x["delta"] > 0} == {0, 1} and
                  {x["member"] for x in r["rows"] if x["delta"] > 0} == {0, 1} for r in derived)
    correct = all(r["mismatches"] == 0 for r in c.load("fidelity.json") + c.load("integration.json"))
    passed = correct and advantage and support and len(derived) >= 2 and len({r["program"]["mode"] for r in derived}) >= 2 and len({e["origin"] for e in examples}) >= 3 and a["unique_dev_flips"] > 0
    total = sum(l["turns"] for l in ledger) + d.regression_turns()
    assert total <= 9600 * 720
    c.save("summary.json", {"starting_head": c.load("config.json")["head"], "parent_nodes": c.load("config.json")["parent_nodes"],
        "arms": results, "initial_hypotheses_per_arm": 25, "refinements": sum(r["arm"] == "ccpr" and r["generation"] > 0 for r in records),
        "counterexample_instances": sum(r["metrics"]["counterexamples"] for r in records if r["arm"] == "ccpr"),
        "improved_lineages": len({e["origin"] for e in examples}), "examples": examples,
        "snapshot_mismatches": sum(r["mismatches"] for r in c.load("fidelity.json")),
        "cpu_cuda_mismatches": sum(r["mismatches"] for r in c.load("integration.json")),
        "simulated_turns": total, "fge": total / 720, "regression_turns": d.regression_turns(),
        "material_advantage": advantage, "derived_families": sorted({MODES[r["program"]["mode"]] for r in derived}),
        "pass": passed, "verdict": "PHASE 2 PASS - TRAJECTORY-DIVERSE CCPR VALIDATED, PROCEED TO REACHABLE CHALLENGE FRONTIER" if passed else "PHASE 2 FAIL - CCPR NOT SUPERIOR, MOVE TO DSQD"})
    print("PASS" if passed else "FAIL", results, "all-in FGE", total / 720, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "fidelity", "discovery", "guard", "validate", "summary"))
    args = parser.parse_args()
    configure()
    globals()[args.stage]()
