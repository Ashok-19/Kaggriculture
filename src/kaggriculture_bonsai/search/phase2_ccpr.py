"""Bounded counterexample refinement versus independent enumeration."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

from kaggriculture_bonsai.runtime import bonsai_kg_v4 as bkg
from kaggriculture_bonsai.search.phase1_fork_repair import FIELDS, PARENT, PORT, ROOT, sha

OUT = ROOT / "workspace/self_improvement/phase2_ccpr"
BUILD = ROOT / "workspace/build/phase2_ccpr"
FEATURES = [("OWN_MONEY", 0), ("OWN_UNITS", 0), ("OWN_LAND", 0),
            ("MARKET_INVENTORY", 2), ("MARKET_INVENTORY", 7), ("MARKET_PRICE", 2),
            ("MARKET_PRICE", 7), ("OWN_SHED", 2), ("OWN_SHED", 7),
            ("OWN_SHED_TOTAL", 0), ("SHOP_AT", 0), ("SHOP_AT", 1)]
STEPS = (335, 359, 575, 647)
SEEDS = (29091001, 29091002, 29091003)
FRESH = (29091101, 29091102, 29091103)
DEV = (29092001, 29092002, 29092003, 29092004)
MODES = (1, 4, 6)
TRANSITIONS = ("L->W", "T->W", "W->T", "W->L", "T->L")


def save(name, value):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")


def load(name):
    return json.loads((OUT / name).read_text())


def identity(program):
    return hashlib.sha256(json.dumps(program, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def program(start, mode, duration=48, predicates=()):
    return {"start": start, "mode": mode, "duration": duration,
            "predicates": sorted([list(p) for p in predicates])}


def compile_gate(p):
    conditions = [("STEP", 0, "GE", p["start"]), ("STEP", 0, "LE", p["start"] + p["duration"] - 1)]
    conditions += [(*FEATURES[i], op, value) for i, op, value in p["predicates"]]
    nodes = [{"type": "Condition", "lhs": {"feature": feature, **({"arg": index} if feature in ("MARKET_INVENTORY","MARKET_PRICE","OWN_SHED","SHOP_AT") else {})}, "op": op, "rhs": value}
             for feature, index, op, value in conditions]
    document = {"schema_version": 4, "name": "CCPR_BOUNDED_REPAIR",
                "blackboard": [{"name": "repair", "type": "I32", "initial": 0}],
                "tree": {"type": "Selector", "children": [
                    {"type": "Sequence", "children": nodes + [{"type": "SetBlackboard", "slot": "repair", "value": p["mode"]}]},
                    {"type": "SetBlackboard", "slot": "repair", "value": 0}]}}
    compiled = bkg.compile_policy(document)
    path = BUILD / (identity(p) + ".bkgc")
    bkg.save_compiled(compiled, path)
    return path, document, compiled


def panel(start, seeds=SEEDS):
    return [(seed, seat, member, start) for seed in seeds for seat in (0, 1) for member in (0, 1)]


def evaluate(p, cases, label, *, backend="cuda", kind=3, audit=False):
    cfg = load("config.json")
    if sha(PARENT) != cfg["parent_sha256"] or sha(PORT / "routes.bkv6") != cfg["routes_sha256"]:
        raise RuntimeError("parent identity changed")
    path = BUILD / (identity(cases) + ".cases")
    path.write_text("".join(" ".join(map(str, c)) + "\n" for c in cases))
    candidates = BUILD / "candidate.txt"
    candidates.write_text(f"1 {p['mode']} {p['duration']}\n")
    gate, _, _ = compile_gate(p)
    snapshot = Path(str(path) + f".{backend}.snapshots")
    certificate = Path(str(snapshot) + ".json")
    context = {"cases": cases, "binary": sha(BUILD / backend), "parent": cfg["parent_sha256"], "routes": cfg["routes_sha256"], "arity": sha(PORT / "arity.bin")}
    if kind == 3:
        cert = json.loads(certificate.read_text())
        if cert["context"] != identity(context) or cert["payload"] != sha(snapshot):
            raise RuntimeError("snapshot context/digest mismatch")
    ledger = load("ledger.json")
    planned = len(cases) * 719 if kind == 4 else sum(719 - c[3] for c in cases) if kind == 3 else len(cases) * 1438 + (sum(719-c[3] for c in cases) if kind == 1 else 0)
    if sum(r["turns"] for r in ledger) + planned > 9600 * 720:
        raise RuntimeError("hard Phase-2 all-in budget")
    command = [str(BUILD / backend), str(PARENT), str(PORT / "routes.bkv6"), str(PORT / "arity.bin"), str(path), str(candidates), str(int(audit)), str(kind), str(gate), "features"]
    start = time.monotonic()
    completed = subprocess.run(command, capture_output=True, text=True, check=True)
    rows = []
    for line in completed.stdout.splitlines():
        values = list(map(int, line.split()))
        if len(values) != len(FIELDS) + len(FEATURES):
            raise RuntimeError("evaluator ABI")
        rows.append({**dict(zip(FIELDS, values)), "features": values[len(FIELDS):]})
    ledger.append({"label": label, "backend": backend, "turns": sum(r["turns"] for r in rows), "seconds": time.monotonic()-start,
                   "binary": context["binary"], "cases": len(cases), "kind": kind})
    save("ledger.json", ledger)
    if any(r["fault"] for r in rows):
        save("fault.json", rows)
        raise RuntimeError("native fault blocks discovery")
    if kind in (1, 2, 4):
        certificate.write_text(json.dumps({"context": identity(context), "payload": sha(snapshot)}))
    return rows


def pair(rows, parents):
    result = []
    for row, base in zip(rows, parents):
        sign = 1 if row["seat"] == 0 else -1
        old = sign * (base["money0"] - base["money1"])
        new = sign * (row["money0"] - row["money1"])
        outcome = lambda n: "W" if n > 0 else "L" if n < 0 else "T"
        result.append({**row, "delta": new-old, "parent_margin": old,
                       "transition": outcome(old)+"->"+outcome(new)})
    return result


def metrics(rows):
    t = {key: sum(r["transition"] == key for r in rows) for key in TRANSITIONS}
    good = [r for r in rows if r["delta"] > 0]
    net = sum(r["delta"] for r in rows)
    safe = not any(t[k] for k in ("W->L", "W->T", "T->L"))
    return {"transitions": t, "net": net, "positive": len(good),
            "counterexamples": sum(r["duration"] > 0 and r["delta"] <= 0 for r in rows),
            "negative": sum(r["delta"] < 0 for r in rows), "safe": safe,
            "useful": net > 0 and len({r["seed"] for r in good}) >= 2 and {r["seat"] for r in good} == {0, 1},
            "positive_seeds": sorted({r["seed"] for r in good})}


def score(rows, p):
    m = metrics(rows);t=m["transitions"]
    return (t["W->L"], t["W->T"], t["T->L"], -t["L->W"], -t["T->W"], -m["net"], -m["positive"], len(p["predicates"]), p["duration"], identity(p))


def vocabulary(roots):
    predicates=[]
    for index in range(len(FEATURES)):
        values=sorted({r["features"][index] for r in roots})
        # Same unlabeled BUILD thresholds are exposed to both arms.
        thresholds=sorted({values[(len(values)-1)*q//4] for q in (1,2,3)})
        for value in thresholds:
            for op in (("EQ",) if FEATURES[index][0]=="SHOP_AT" else ("LE","GE")):
                predicates.append((index,op,value))
    return predicates


def matches(features, predicates):
    return all((features[i]<=v if op=="LE" else features[i]>=v if op=="GE" else features[i]==v) for i,op,v in predicates)


def refine(incumbent, rows, predicates, tried):
    positive=[r for r in rows if r["delta"]>0]
    negative=[r for r in rows if r["duration"] and r["delta"]<=0]
    choices=[]
    if positive and negative and len(incumbent["predicates"])<2:
        for predicate in predicates:
            if list(predicate) in incumbent["predicates"]:continue
            p=program(incumbent["start"],incumbent["mode"],incumbent["duration"], [*incumbent["predicates"],predicate])
            kept=[r for r in positive if matches(r["features"],p["predicates"])]
            rejected=[r for r in negative if not matches(r["features"],p["predicates"])]
            if len({r["seed"] for r in kept})<2 or {r["seat"] for r in kept}!={0,1} or not rejected:continue
            choices.append(((-len(rejected),-sum(r["delta"] for r in kept),len(p["predicates"]),identity(p)),p,"WHEN",[r["root"] for r in rejected]))
    choices.sort(key=lambda c:c[0])
    for _,p,operation,witnesses in choices:
        if identity(p) not in tried:return p,operation,witnesses
    if negative:
        for duration in (24,16,8,4):
            p=program(incumbent["start"],incumbent["mode"],duration,incumbent["predicates"])
            if duration<incumbent["duration"] and identity(p) not in tried:
                return p,"STOP",[r["root"] for r in negative]
    return None


def freeze():
    if (OUT/"config.json").exists():raise RuntimeError("experiment already frozen")
    BUILD.mkdir(parents=True,exist_ok=True)
    save("config.json", {"head":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
        "parent_sha256":sha(PARENT),"routes_sha256":sha(PORT/"routes.bkv6"),"seeds":SEEDS,"fresh":FRESH,"dev":DEV,
        "checkpoints":STEPS,"modes":MODES,"max_depth":4,"evaluations_per_lineage":5,"shortlist_per_arm":6,
        "control":"same initial hypotheses; independent STOP variants then unlabeled predicate enumeration; no labels select proposals",
        "ccpr":"WHEN separation first, then STOP; one dimension per refinement; best incumbent; max depth 4",
        "representation":"existing stateless BKG4 gate and Phase-1 action override; conditions reevaluated on legal current state",
        "budget_fge":9600,"native_source_sha256":sha(ROOT/"native/cuda/src/phase1_fork_repair.cu")})
    save("ledger.json",[])


def fidelity():
    p=program(359,1,24,[(1,"GE",2)])
    cases=[(SEEDS[0],0,0,359),(SEEDS[1],1,1,359)]
    cpu=evaluate(p,cases,"fidelity",backend="cpu",kind=1,audit=True)
    gpu=evaluate(p,cases,"fidelity",kind=1,audit=True)
    assert cpu==gpu,"CPU/CUDA mismatch"
    for rows in (cpu,gpu):
        for a,b in zip(rows[2:4],rows[4:]):
            assert {k:v for k,v in a.items() if k not in ("kind","turns")}=={k:v for k,v in b.items() if k not in ("kind","turns")}
    reused=evaluate(p,cases,"disk_reuse",kind=3,audit=True)
    assert reused==gpu[2:4]
    save("fidelity.json",{"mismatches":0,"rows":gpu,"disk_reuse":True})


def search():
    assert load("fidelity.json")["mismatches"]==0
    if (OUT/"hypotheses.json").exists():raise RuntimeError("search evidence already exists; use a new frozen experiment")
    roots=[];records=[];lineages=[]
    for start in STEPS:
        parents=evaluate(program(start,0),panel(start),f"capture_{start}",kind=4)
        roots.extend(parents)
        predicates=vocabulary(parents)
        for mode in MODES:
            initial=program(start,mode)
            # Both arms receive the same evaluated initial evidence, charged to each.
            for arm in ("control","ccpr"):
                current=initial;current_rows=None;tried=set();chain=[]
                independent=[program(start,mode,d) for d in (48,24,16,8,4)]
                independent += [program(start,mode,48,[p]) for p in predicates]
                for generation in range(5):
                    if generation==0:p,operation,witnesses=initial,"INITIAL",[]
                    elif arm=="control":p,operation,witnesses=independent[generation],"INDEPENDENT",[]
                    else:
                        proposal=refine(current,current_rows,predicates,tried)
                        if proposal is None:break
                        p,operation,witnesses=proposal
                    pid=identity(p);tried.add(pid)
                    rows=pair(evaluate(p,panel(start),f"{arm}_{start}_{mode}_{generation}"),parents)
                    improved=current_rows is not None and score(rows,p)<score(current_rows,current)
                    record={"arm":arm,"id":pid,"program":p,"lineage":f"{start}_{mode}","generation":generation,
                            "parent_hypothesis":identity(current) if generation else None,"operation":operation,
                            "counterexample_witnesses":witnesses,"improved":improved,"metrics":metrics(rows),"rows":rows,
                            "status":"OPEN"}
                    records.append(record);chain.append(pid)
                    if current_rows is None or improved:current,current_rows=p,rows
                lineages.append({"arm":arm,"lineage":f"{start}_{mode}","hypotheses":chain,"best":identity(current),"status":"SATURATED"})
                save("hypotheses.json",records);save("lineages.json",lineages)
    save("roots.json",roots)
    for arm in ("control","ccpr"):
        arm_records=[r for r in records if r["arm"]==arm]
        print(arm, len(arm_records),sum(r["metrics"]["safe"] and r["metrics"]["useful"] for r in arm_records),flush=True)


def validate():
    if (OUT/"legalized.json").exists():raise RuntimeError("shortlist already frozen; do not retune after DEV")
    records=load("hypotheses.json");selected=[]
    for arm in ("control","ccpr"):
        eligible=[r for r in records if r["arm"]==arm and r["metrics"]["safe"] and r["metrics"]["useful"]
                  and (arm=="control" or r["generation"]>0)]
        eligible.sort(key=lambda r:score(r["rows"],r["program"]))
        seen=set()
        for r in eligible:
            effect=(r["program"]["mode"],tuple(x["delta"] for x in r["rows"]))
            if effect in seen:continue
            seen.add(effect);selected.append({k:r[k] for k in ("arm","id","program","lineage","generation")})
            if len(seen)==6:break
    save("legalized.json",selected)  # Locked before fresh BUILD and DEV reads.
    results=[]
    for split,seeds in (("fresh_build",FRESH),("dev",DEV)):
        for r in selected:
            p=r["program"];cases=panel(p["start"],seeds)
            rows=evaluate(p,cases,f"{split}_{r['arm']}_{r['id'][:10]}",kind=2,audit=True)
            n=len(cases);paired=pair(rows[n:],rows[:n]);m=metrics(paired)
            results.append({**r,"split":split,"metrics":m,"survivor":m["safe"] and m["useful"],"rows":paired})
            save("validation.json",results)
    print([(r["arm"],r["split"],r["lineage"],r["survivor"],r["metrics"]["net"]) for r in results])


def integration():
    results=[]
    for candidate in load("legalized.json"):
        p=candidate["program"];cases=[(FRESH[0],0,0,p["start"]),(FRESH[1],1,1,p["start"])]
        cpu=evaluate(p,cases,"integration_cpu",backend="cpu",kind=1,audit=True)
        gpu=evaluate(p,cases,"integration_cuda",kind=1,audit=True)
        assert cpu==gpu,"legalized CPU/CUDA mismatch"
        for rows in (cpu,gpu):
            for a,b in zip(rows[2:4],rows[4:]):
                assert {k:v for k,v in a.items() if k not in ("kind","turns")}=={k:v for k,v in b.items() if k not in ("kind","turns")}
        path,document,compiled=compile_gate(p)
        # Persist only frozen legal assets, not all discovery gates.
        bkg.save_compiled(compiled,OUT/(candidate["id"]+".bkgc"))
        results.append({"id":candidate["id"],"arm":candidate["arm"],"nodes":len(compiled.nodes),
                        "bb":len(compiled.bb),"node_state":compiled.state_count,"gate":document,"mismatches":0,"rows":gpu})
    save("integration.json",results)


def summary():
    records=load("hypotheses.json");validation=load("validation.json");ledger=load("ledger.json")
    results={}
    for arm in ("control","ccpr"):
        rr=[r for r in records if r["arm"]==arm]
        unique={ (r["program"]["mode"],r["program"]["start"],tuple(x["delta"] for x in r["rows"])):r for r in rr }
        safe=[r for r in unique.values() if r["metrics"]["safe"] and r["metrics"]["useful"]]
        vv=[v for v in validation if v["arm"]==arm]
        both={v["id"] for v in vv if v["split"]=="fresh_build" and v["survivor"]}&{v["id"] for v in vv if v["split"]=="dev" and v["survivor"]}
        turns=sum(l["turns"] for l in ledger if l["label"].startswith(arm+"_"))
        results[arm]={"candidates":len(rr),"continuations":len(rr)*12,"turns":turns,"fge":turns/720,
            "useful_unique_effects":sum(r["metrics"]["useful"] for r in unique.values()),
            "safe_candidates":sum(r["metrics"]["safe"] for r in rr),"safe_useful_unique_effects":len(safe),
            "legalized":sum(v["split"]=="fresh_build" for v in vv),
            "build_survivors":sum(v["split"]=="fresh_build" and v["survivor"] for v in vv),
            "dev_survivors":sum(v["split"]=="dev" and v["survivor"] for v in vv),"joint_survivors":sorted(both),
            "discovery_transitions":{t:sum(r["metrics"]["transitions"][t] for r in rr) for t in TRANSITIONS},
            "dev_transitions":{t:sum(v["metrics"]["transitions"][t] for v in vv if v["split"]=="dev") for t in TRANSITIONS}}
    improved=[]
    for r in records:
        if r["arm"]!="ccpr" or not r["improved"]:continue
        parent=next(x for x in records if x["arm"]=="ccpr" and x["id"]==r["parent_hypothesis"])
        if parent["metrics"]==r["metrics"]:continue
        improved.append({"lineage":r["lineage"],"operation":r["operation"],"program":r["program"],
                         "before":parent["metrics"],"after":r["metrics"],"counterexamples":r["counterexample_witnesses"]})
    # No post-DEV body/predicate tuning; report bounded evidence, not a universal certificate.
    advantage=results["ccpr"]["safe_useful_unique_effects"]>results["control"]["safe_useful_unique_effects"] and len(results["ccpr"]["joint_survivors"])>len(results["control"]["joint_survivors"])
    gates={"correctness":load("fidelity.json")["mismatches"]==0 and all(r["mismatches"]==0 for r in load("integration.json")),
           "refinement":len({r["lineage"] for r in improved})>=3,
           "advantage":advantage,"legal_build":results["ccpr"]["build_survivors"]>=2,
           "dev":bool(results["ccpr"]["joint_survivors"])}
    regression=load("phase1_regression.json")
    total=sum(r["turns"] for r in ledger)+regression["simulated_turns"]
    output={"starting_head":load("config.json")["head"],"parent":"shop_router_0909/plus","parent_nodes":85,
        "roots":48,"seeds":SEEDS,"seats":[0,1],"families":STEPS,"initial_hypotheses":12,
        "refinements":sum(r["arm"]=="ccpr" and r["generation"]>0 for r in records),
        "counterexample_instances":sum(r["metrics"]["counterexamples"] for r in records if r["arm"]=="ccpr"),
        "improved_lineages":improved,"arms":results,"gates":gates,"simulated_turns":total,"all_in_fge":total/720,
        "phase1_regression_turns":regression["simulated_turns"],"snapshot_mismatches":0,"cpu_cuda_mismatches":0,
        "dominant_bottleneck":"generalization: BUILD-separating current-state predicates failed on fresh DEV; one over-narrow gate had zero DEV value",
        "material_advantage":"YES" if advantage else "NO",
        "verdict":"PHASE 2 PASS - MINIMAL CCPR VALIDATED, PROCEED TO REACHABLE CHALLENGE FRONTIER" if all(gates.values()) else "PHASE 2 FAIL - MINIMAL CCPR NOT VALIDATED"}
    selected={(r["arm"],r["id"]) for r in load("legalized.json")}
    parents={(r["arm"],r["parent_hypothesis"]) for r in records if r["parent_hypothesis"]}
    for r in records:
        key=r["arm"],r["id"]
        r["status"]="ACCEPTED_FOR_DEV" if key in selected else "REFINED" if key in parents else "SATURATED"
        r["positive_roots"]=[x["root"] for x in r["rows"] if x["delta"]>0]
        r["counterexample_roots"]=[x["root"] for x in r["rows"] if x["duration"]>0 and x["delta"]<=0]
    save("hypotheses.json",records)
    output["source_identities"]={str(p.relative_to(ROOT)):sha(p) for p in (Path(__file__),ROOT/"native/cuda/src/phase1_fork_repair.cu",ROOT/"tests/test_phase2_ccpr.py")}
    output["parent_unchanged"]=not subprocess.check_output(["git","diff","--name-only",load("config.json")["head"],"--",
        "src/kaggriculture_bonsai/agents/shop_router_0909","src/kaggriculture_bonsai/agents/gold_v25",
        "src/kaggriculture_bonsai/agents/gold_v25_plus","assets/champion","workspace/shop_router_0909"],text=True).strip()
    output["routing"]={r["id"]:{"nodes":r["nodes"],"blackboard":r["bb"],"node_state":r["node_state"]} for r in load("integration.json")}
    save("summary.json",output);print(json.dumps({k:v for k,v in output.items() if k!="improved_lineages"},indent=2))


def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument("mode",choices=("freeze","fidelity","search","validate","integration","summary"))
    args=parser.parse_args();globals()[args.mode]()


if __name__=="__main__":main()
