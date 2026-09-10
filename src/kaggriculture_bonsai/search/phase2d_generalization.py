"""One bounded trajectory-diversity diagnosis, reusing the frozen CCPR loop."""
from __future__ import annotations

import argparse
import json
import subprocess
from collections import Counter

from kaggriculture_bonsai.search import phase2_ccpr as ccpr

OLD = ccpr.ROOT / "workspace/self_improvement/phase2_ccpr"
OUT = ccpr.ROOT / "workspace/self_improvement/phase2d_generalization"
LIMIT = 2000 * 720


def configure():
    ccpr.OUT = OUT


def old(name):
    return json.loads((OLD / name).read_text())


def regression_turns():
    path = OUT / "phase1_regression.json"
    return json.loads(path.read_text())["simulated_turns"] if path.exists() else 0


def trajectories():
    games = []
    for i in range(32):
        j = i % 16
        steps = (335, 359) if j < 4 else (575, 647) if j < 8 else (359,) if j < 12 else (647,)
        games.append({"seed": 39091001 + i, "seat": j % 2, "member": (j // 2) % 2,
                      "split": "discovery" if i < 16 else "guard", "checkpoints": steps})
    return games


def cases(split, start):
    return [(g["seed"], g["seat"], g["member"], start)
            for g in ccpr.load("trajectory_groups.json")
            if g["split"] == split and start in g["checkpoints"]]


def group_metrics(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault(row["seed"], []).append(row)
    groups = {str(seed): ccpr.metrics(values) for seed, values in sorted(grouped.items())}
    metrics = ccpr.metrics(rows)
    worst = min((g["net"] for g in groups.values()), default=0)
    positive = sum(g["net"] > 0 for g in groups.values())
    return {**metrics, "groups": groups, "worst_group_net": worst,
            "positive_groups": positive, "negative_groups": sum(g["net"] < 0 for g in groups.values()),
            "eligible": metrics["safe"] and metrics["useful"] and worst >= 0 and positive >= 2}


def rank(rows, program):
    m = group_metrics(rows)
    t = m["transitions"]
    return (t["W->L"], t["W->T"], t["T->L"], m["negative_groups"], -m["worst_group_net"],
            -t["L->W"], -t["T->W"], -m["positive_groups"], -m["net"],
            len(program["predicates"]), program["duration"], ccpr.identity(program))


def evaluate(program, panel, label, **kwargs):
    # The old evaluator's larger cap remains untouched; enforce this epoch's cap before every call.
    kind = kwargs.get("kind", 3)
    suffix = sum(719 - c[3] for c in panel)
    planned = len(panel) * 719 if kind == 4 else suffix if kind == 3 else len(panel) * 1438 + (suffix if kind == 1 else 0)
    limit = ccpr.load("config.json").get("budget_fge", 2000) * 720
    if sum(r["turns"] for r in ccpr.load("ledger.json")) + regression_turns() + planned > limit:
        raise RuntimeError("hard configured all-in FGE budget")
    rows = ccpr.evaluate(program, panel, label, **kwargs)
    expected = len(panel) * (3 if kind == 1 else 2 if kind == 2 else 1)
    if len(rows) != expected:
        raise RuntimeError("incomplete evaluator output")
    return rows


def freeze():
    if (OUT / "config.json").exists():
        raise RuntimeError("epoch already frozen")
    selected = [r for r in old("legalized.json") if r["arm"] == "ccpr"]
    history = old("hypotheses.json")
    extra = [ccpr.program(335, 1, 8), ccpr.program(359, 1), ccpr.program(647, 1)]
    for p in extra:
        r = next(r for r in history if r["arm"] == "ccpr" and r["id"] == ccpr.identity(p))
        selected.append({k: r[k] for k in ("arm", "id", "program", "lineage", "generation")})
    selected.append(next(r for r in old("legalized.json") if r["id"].startswith("f66e6485")))
    ccpr.save("config.json", {"head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "parent_sha256": ccpr.sha(ccpr.PARENT), "routes_sha256": ccpr.sha(ccpr.PORT / "routes.bkv6"),
        "arity_sha256": ccpr.sha(ccpr.PORT / "arity.bin"), "starting_hypotheses": selected,
        "max_additional_refinements": 3, "budget_fge": 2000,
        "dev_seeds": list(range(39092001, 39092025)),
        "selection": "one discovery-ranked incumbent per lineage; zero protected regressions and no negative seed groups on discovery AND guard; >=2 positive seeds and both seats on each panel",
        "firewall": "freeze all incumbents before guard; no guard-driven refinement; originals are mandatory DEV shadow controls, never rescued after guard rejection",
        "control": "unchanged Phase-2 independent-proposal survivor; no new independent search",
        "recovery": "multiple old brittle candidates rejected, >=1 newly refined eligible DEV survivor, and a survivor strictly better than control on L->W,T->W,positive seed groups,net (lexicographic), with zero protected DEV regression",
        "root_selection": "fixed known daily/economic leverage windows; no result-based seed or checkpoint replacement",
        "source_sha256": {str(path.relative_to(ccpr.ROOT)): ccpr.sha(path) for path in (
            ccpr.ROOT / "native/cuda/src/phase1_fork_repair.cu",
            ccpr.ROOT / "src/kaggriculture_bonsai/search/phase2_ccpr.py")},
        "binaries": {b: ccpr.sha(ccpr.BUILD / b) for b in ("cpu", "cuda")}})
    ccpr.save("trajectory_groups.json", trajectories())
    ccpr.save("ledger.json", [])


def check_integration(p, label):
    panel = [(39093001, 0, 0, p["start"]), (39093002, 1, 1, p["start"])]
    cpu = evaluate(p, panel, label + "_cpu", backend="cpu", kind=1, audit=True)
    gpu = evaluate(p, panel, label + "_cuda", kind=1, audit=True)
    assert cpu == gpu, "CPU/CUDA mismatch"
    for a, b in zip(gpu[2:4], gpu[4:]):
        assert {k: v for k, v in a.items() if k not in ("kind", "turns")} == {
            k: v for k, v in b.items() if k not in ("kind", "turns")}, "snapshot/full-game mismatch"
    reused = evaluate(p, panel, label + "_reuse", audit=True)
    assert reused == gpu[2:4], "disk snapshot mismatch"
    return {"id": ccpr.identity(p), "mismatches": 0, "rows": gpu}


def fidelity():
    if (OUT / "fidelity.json").exists():
        raise RuntimeError("fidelity already recorded")
    ccpr.save("fidelity.json", check_integration(ccpr.program(359, 1, 24, [(1, "GE", 2)]), "fidelity"))


def discovery():
    assert ccpr.load("fidelity.json")["mismatches"] == 0
    if (OUT / "candidate_results.json").exists():
        raise RuntimeError("discovery already run; no exposed-epoch retuning")
    parents = {}
    for start in ccpr.STEPS:
        parents[start] = evaluate(ccpr.program(start, 0), cases("discovery", start), f"capture_discovery_{start}", kind=4)
    ccpr.save("roots.json", {"discovery": parents})
    records, finalists = [], []
    for origin in ccpr.load("config.json")["starting_hypotheses"]:
        p = origin["program"]
        base = parents[p["start"]]
        vocab = ccpr.vocabulary(base)
        current, current_rows, tried = p, None, set()
        depth = 0 if origin["arm"] == "control" else 3
        for generation in range(depth + 1):
            operation, witnesses, ancestor = "INITIAL", [], None
            if generation:
                proposal = ccpr.refine(current, current_rows, vocab, tried)
                if proposal is None:
                    break
                p, operation, witnesses = proposal
                ancestor = ccpr.identity(current)
            pid = ccpr.identity(p)
            tried.add(pid)
            rows = ccpr.pair(evaluate(p, cases("discovery", p["start"]), f"discovery_{origin['id'][:8]}_{generation}"), base)
            improved = current_rows is not None and rank(rows, p) < rank(current_rows, current)
            records.append({"origin": origin["id"], "arm": origin["arm"], "id": pid, "program": p,
                            "generation": generation, "operation": operation, "parent_hypothesis": ancestor,
                            "counterexample_witnesses": witnesses, "improved": improved,
                            "metrics": group_metrics(rows), "rows": rows})
            if current_rows is None or improved:
                current, current_rows = p, rows
        finalists.append({"origin": origin["id"], "arm": origin["arm"], "id": ccpr.identity(current), "program": current})
    ccpr.save("candidate_results.json", records)
    ccpr.save("frozen_incumbents.json", finalists)
    print("discovery", len(records), "evaluations; frozen incumbents", len(finalists), flush=True)


def guard():
    if (OUT / "guard.json").exists():
        raise RuntimeError("guard already opened")
    records = ccpr.load("candidate_results.json")
    finalists = ccpr.load("frozen_incumbents.json")
    # Only original hypotheses and pre-guard incumbents are screened, not a new search over guard labels.
    frozen = {r["id"]: r for r in records if r["generation"] == 0}
    for f in finalists:
        frozen[f["id"]] = next(r for r in records if r["id"] == f["id"])
    ccpr.save("guard_manifest.json", sorted(frozen))
    roots = ccpr.load("roots.json")
    roots["guard"] = {}
    for start in ccpr.STEPS:
        roots["guard"][str(start)] = evaluate(ccpr.program(start, 0), cases("guard", start), f"capture_guard_{start}", kind=4)
    ccpr.save("roots.json", roots)
    results = []
    for pid, r in sorted(frozen.items()):
        p = r["program"]
        rows = ccpr.pair(evaluate(p, cases("guard", p["start"]), f"guard_{pid[:10]}"), roots["guard"][str(p["start"])])
        m = group_metrics(rows)
        results.append({"id": pid, "program": p, "metrics": m, "rows": rows,
                        "discovery_eligible": r["metrics"]["eligible"], "accepted": r["metrics"]["eligible"] and m["eligible"]})
    ccpr.save("guard.json", results)
    passed = {r["id"] for r in results if r["accepted"]}
    selected = [f for f in finalists if f["id"] in passed]
    ccpr.save("legalized.json", selected)
    # DEV shadow measurements cannot change the frozen acceptance decisions.
    dev = {r["id"]: {"id": r["id"], "program": r["program"]} for r in records if r["generation"] == 0}
    dev.update({f["id"]: {"id": f["id"], "program": f["program"]} for f in selected})
    ccpr.save("dev_manifest.json", list(dev.values()))
    print("guard accepted", len(passed), "prospective shortlist", len(selected), "DEV including shadows", len(dev), flush=True)


def validate():
    if (OUT / "dev.json").exists():
        raise RuntimeError("DEV already opened; no further refinement")
    integrations, results = [], []
    selected = {r["id"] for r in ccpr.load("legalized.json")}
    for r in ccpr.load("dev_manifest.json"):
        p = r["program"]
        integrations.append(check_integration(p, "integration_" + r["id"][:10]))
        ccpr.save("integration.json", integrations)
        if r["id"] in selected:
            panel = cases("guard", p["start"])
            full = evaluate(p, panel, "legal_build_" + r["id"][:10], kind=2, audit=True)
            n = len(panel)
            paired = ccpr.pair(full[n:], full[:n])
            fork = next(g["rows"] for g in ccpr.load("guard.json") if g["id"] == r["id"])
            assert [(x["money0"], x["money1"], x["delta"], x["duration"], x["changes"]) for x in paired] == [
                (x["money0"], x["money1"], x["delta"], x["duration"], x["changes"]) for x in fork]
            results.append({**r, "split": "legal_build", "metrics": group_metrics(paired), "rows": paired})
        panel = [(seed, i % 2, (i // 2) % 2, p["start"])
                 for i, seed in enumerate(ccpr.load("config.json")["dev_seeds"])]
        full = evaluate(p, panel, "dev_" + r["id"][:10], kind=2, audit=True)
        n = len(panel)
        paired = ccpr.pair(full[n:], full[:n])
        m = group_metrics(paired)
        results.append({**r, "split": "dev", "selected": r["id"] in selected,
                        "metrics": m, "survivor": r["id"] in selected and m["eligible"], "rows": paired})
        ccpr.save("dev.json", results)
        print("DEV", r["id"][:12], "selected", r["id"] in selected, "net", m["net"], "safe", m["safe"], flush=True)


def summary():
    cfg = ccpr.load("config.json")
    records, guards, dev = (ccpr.load(n) for n in ("candidate_results.json", "guard.json", "dev.json"))
    comparison = []
    for origin in cfg["starting_hypotheses"]:
        pid = origin["id"]
        hist = next(r for r in old("hypotheses.json") if r["id"] == pid and r["arm"] == origin["arm"])
        prior = next((r for r in old("validation.json") if r["id"] == pid and r["arm"] == origin["arm"] and r["split"] == "dev"), None)
        new = next(r for r in records if r["id"] == pid and r["generation"] == 0)
        guard_row = next(r for r in guards if r["id"] == pid)
        dev_row = next(r for r in dev if r["id"] == pid and r["split"] == "dev")
        comparison.append({"id": pid, "arm": origin["arm"], "program": origin["program"], "old_build": hist["metrics"],
            "old_dev": prior["metrics"] if prior else None, "discovery": new["metrics"], "guard": guard_row["metrics"],
            "new_build_accepted": guard_row["accepted"], "dev": dev_row["metrics"]})
    old_brittle = [c for c in comparison if c["arm"] == "ccpr" and c["old_dev"] is not None
                  and not (c["old_dev"]["safe"] and c["old_dev"]["useful"])]
    rejected = sum(not c["new_build_accepted"] for c in old_brittle)
    selected = ccpr.load("legalized.json")
    ccpr_ids = {r["id"] for r in selected if r["arm"] == "ccpr"}
    survivors = [r for r in dev if r["split"] == "dev" and r["survivor"] and r["id"] in ccpr_ids]
    new_ids = {r["id"] for r in records if r["generation"] > 0} - {r["id"] for r in cfg["starting_hypotheses"]}
    control = next(r for r in dev if r["split"] == "dev" and r["id"] == cfg["starting_hypotheses"][-1]["id"])
    def value(r):
        m = r["metrics"]
        return (m["transitions"]["L->W"], m["transitions"]["T->W"], m["positive_groups"], m["net"])
    advantage = any(value(r) > value(control) for r in survivors) if control["survivor"] else bool(survivors)
    recovered = rejected >= 2 and any(r["id"] in new_ids for r in survivors) and advantage
    total = sum(r["turns"] for r in ccpr.load("ledger.json")) + regression_turns()
    old_correct = sum((c["old_build"]["safe"] and c["old_build"]["useful"]) == c["dev"]["eligible"] for c in comparison)
    new_correct = sum(c["new_build_accepted"] == c["dev"]["eligible"] for c in comparison)
    routing = []
    for pid in sorted({r["id"] for r in selected}):
        p = next(r["program"] for r in selected if r["id"] == pid)
        _, document, compiled = ccpr.compile_gate(p)
        path = OUT / (pid + ".bkgc")
        ccpr.bkg.save_compiled(compiled, path)
        routing.append({"id": pid, "program": p, "gate": document, "nodes": len(compiled.nodes),
                        "blackboard": len(compiled.bb), "node_state": compiled.state_count,
                        "bytes": path.stat().st_size, "sha256": ccpr.sha(path)})
    result = {"starting_head": cfg["head"], "parent": "shop_router_0909/plus", "parent_nodes": 85,
        "independent_build_games": 32, "build_seeds": [g["seed"] for g in trajectories()], "both_seats": True,
        "roots": 48, "roots_per_game": dict(Counter(len(g["checkpoints"]) for g in trajectories())),
        "starting_ccpr_hypotheses": 9, "control_hypotheses": 1,
        "additional_refinements": sum(r["generation"] > 0 for r in records),
        "discovery_continuations": sum(len(r["rows"]) for r in records), "guard_continuations": sum(len(r["rows"]) for r in guards),
        "simulated_turns": total, "fge": total / 720, "phase1_regression_turns": regression_turns(),
        "snapshot_mismatches": ccpr.load("fidelity.json")["mismatches"],
        "cpu_cuda_mismatches": sum(r["mismatches"] for r in ccpr.load("integration.json")),
        "comparisons": comparison, "old_brittle_candidates": len(old_brittle), "brittle_rejected_before_dev": rejected,
        "guard_passing_programs": sum(g["accepted"] for g in guards), "legalized_lineages": len(selected),
        "legalized_unique_programs": len({r["id"] for r in selected}), "ccpr_dev_survivors": len(survivors),
        "control_dev_survivors": int(control["survivor"]), "selected_dev_transitions": {
            t: sum(r["metrics"]["transitions"][t] for r in dev if r["split"] == "dev" and r["selected"])
            for t in ccpr.TRANSITIONS},
        "old_build_correct_predictions": old_correct, "new_build_correct_predictions": new_correct,
        "prediction_denominator": len(comparison), "diversified_build_predicts_dev_better": new_correct > old_correct,
        "ccpr_beats_control": advantage, "recoverable": recovered, "routing": routing,
        "implementation_sha256": ccpr.sha(ccpr.ROOT / "src/kaggriculture_bonsai/search/phase2d_generalization.py"),
        "parent_unchanged": ccpr.sha(ccpr.PARENT) == cfg["parent_sha256"] and ccpr.sha(ccpr.PORT / "routes.bkv6") == cfg["routes_sha256"],
        "verdict": "PHASE 2D PASS - CCPR GENERALIZATION RECOVERED, REDO PHASE 2 WITH TRAJECTORY-DIVERSE BUILD" if recovered else
                   "PHASE 2D FAIL - CCPR GENERALIZATION FAILURE CONFIRMED, MOVE TO DSQD"}
    assert total <= LIMIT
    ccpr.save("summary.json", result)
    print(result["verdict"], "FGE", result["fge"], "brittle rejected", rejected, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "fidelity", "discovery", "guard", "validate", "summary"))
    args = parser.parse_args()
    configure()
    globals()[args.stage]()
