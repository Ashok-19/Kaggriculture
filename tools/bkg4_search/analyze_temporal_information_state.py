from __future__ import annotations

import collections
import hashlib
import json
import statistics
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.bkg4_search.information_state_io import (  # noqa: E402
    HIDDEN_TRUTH_NAMES,
    load_hidden_truth,
    load_r0,
    load_r1,
    r1_manifest,
    registry_manifest,
)
from tools.bkg4_search.optimization_oracle_io import (  # noqa: E402
    load as load_oracle,
    m_candidate_name,
    u_candidate_name,
    w_candidate_name,
)

SCRATCH = ROOT / "scratch/bkg4_search"
OROOT = SCRATCH / "optimization_oracle"
OUT = ROOT / "tools/bkg4_search/TEMPORAL_INFORMATION_STATE_TERMINAL.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_cases() -> list[dict]:
    return [json.loads(x) for x in (OROOT / "screen_cases.jsonl").read_text().splitlines() if x.strip()]


def load_manifest(path: Path) -> list[tuple[int, int, int]]:
    out = []
    for line in path.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        z = line.split("\t")
        out.append((int(z[3]), int(z[4]), int(z[5])))
    return out


def load_result(path: Path, row_struct: struct.Struct) -> tuple[int, int, list[tuple]]:
    raw = path.read_bytes()
    magic, version, candidates, cases = struct.unpack_from("<8sIII", raw, 0)
    if version != 1:
        raise ValueError((path, version))
    rows = [x for x in row_struct.iter_unpack(raw[20:])]
    if len(rows) != candidates * cases:
        raise ValueError((path, len(rows), candidates, cases))
    return candidates, cases, rows


def score_result(path: Path, manifest: Path, names: list[str], row_struct: struct.Struct) -> dict:
    ng, nc, rows = load_result(path, row_struct)
    if ng != len(names):
        raise ValueError((path, ng, len(names)))
    meta = load_manifest(manifest)
    if len(meta) != nc:
        raise ValueError((manifest, len(meta), nc))
    base = rows[:nc]
    bm = [x[0] - x[1] for x in base]
    out = {
        "artifact": str(path.relative_to(ROOT)),
        "artifact_sha256": sha256(path),
        "cases": nc,
        "base_wins": sum(x > 0 for x in bm),
        "base_losses": sum(x < 0 for x in bm),
        "candidates": {},
    }
    for gi, name in enumerate(names):
        r = rows[gi * nc:(gi + 1) * nc]
        m = [x[0] - x[1] for x in r]
        d = [m[i] - bm[i] for i in range(nc)]
        # H0 has rewrites at index2/flags3. H1/H2 have rewrites2, activations3, flags4.
        activations = sum((x[3] > 0 if len(x) >= 6 else x[2] > 0) for x in r)
        rewrites = sum(x[2] for x in r)
        flags = sum((x[4] != 0 if len(x) >= 6 else x[3] != 0) for x in r)
        out["candidates"][name] = {
            "wins": sum(x > 0 for x in m),
            "losses": sum(x < 0 for x in m),
            "activations": activations,
            "rewrites": rewrites,
            "positive_deltas": sum(x > 0 for x in d),
            "negative_deltas": sum(x < 0 for x in d),
            "sum_delta": sum(d),
            "mean_delta": statistics.mean(d),
            "min_delta": min(d),
            "max_delta": max(d),
            "loss_to_win_flips": sum(bm[i] < 0 < m[i] for i in range(nc)),
            "win_to_loss_regressions": sum(bm[i] > 0 >= m[i] for i in range(nc)),
            "protected_margin_regressions": sum(meta[i][1] and d[i] < 0 for i in range(nc)),
            "own_history_margin_regressions": sum(meta[i][2] and d[i] < 0 for i in range(nc)),
            "runtime_fault_games": flags,
        }
    return out


def branches():
    uw = [int(x) for x in (OROOT / "screen_probes_uw.txt").read_text().splitlines() if x.strip()]
    mp = [int(x) for x in (OROOT / "probes_m.txt").read_text().splitlines() if x.strip()]
    routes = (22, 23, 29, 32, 34, 46, 63, 64, 70, 77, 79)
    hz = (24, 72, 144)
    return [
        ("U", load_oracle(OROOT / "oracle_u_screen.b4or"), uw, u_candidate_name),
        ("W", load_oracle(OROOT / "oracle_w_screen_h24.b4or"), uw, w_candidate_name),
        ("M", load_oracle(OROOT / "oracle_m_screen.b4or"), mp, lambda i: m_candidate_name(i, routes, hz)),
    ]


def build_state_cache(r0, r1, truth) -> dict:
    """Intern exact frames once; bytes equality is exact i32 representation equality.

    This keeps the terminal analyzer lightweight instead of rebuilding 3,828-value
    Python tuples inside every oracle-candidate loop.  These keys are offline
    equivalence-analysis artifacts only; no hash/fingerprint enters production.
    """
    probe_index = {step: i for i, step in enumerate(r0.probes)}
    r0_keys = [[r0.frame(ci, pi).tobytes() for pi in range(len(r0.probes))] for ci in range(r0.cases)]
    r1_values = [[tuple(int(x) for x in r1.frame(ci, pi)) for pi in range(len(r1.probes))] for ci in range(r1.cases)]
    truth_keys = [[truth.frame(ci, pi).tobytes() for pi in range(len(truth.probes))] for ci in range(truth.cases)]
    return {"probe_index": probe_index, "r0": r0_keys, "r1": r1_values, "truth": truth_keys}


def ambiguity(r0, r1, truth, cache) -> dict:
    subsets = {
        "R0": [],
        "R0+1": list(range(55)) + [220, 224] + list(range(228, 252)),
        "R0+4": list(range(110)) + [220, 221, 224, 225] + list(range(228, 276)),
        "R0+24": list(range(165)) + [220, 221, 222, 224, 225, 226] + list(range(228, 300)),
        "R0+72": list(range(220)) + list(range(220, 228)) + list(range(228, 324)),
        "R0+events": list(range(368)),
    }
    targets = {
        "U": {"HOLD_BUYS", "HOLD_SELL"},
        "W": {"SWAP_0_3", "SWAP_1_2", "SWAP_6_9"},
        "M": {"ROUTE_23_H144", "ROUTE_46_H144", "ROUTE_77_H72"},
    }
    result = {}
    for bn, oracle, probes, label in branches():
        result[bn] = {}
        for sn, inds in subsets.items():
            cells = mixed = mixed_cases = max_span = 0
            target = {}
            for cand in range(1, oracle.candidates):
                groups: dict[tuple, list[int]] = collections.defaultdict(list)
                for ci in range(oracle.cases):
                    for pi, step in enumerate(probes):
                        x = oracle.rows[ci][pi][cand]
                        if not x[2]:
                            continue
                        b = oracle.rows[ci][pi][0]
                        d = x[0] - x[1] - (b[0] - b[1])
                        ri = cache["probe_index"][step]
                        if sn == "R0": sig = cache["r0"][ci][ri]
                        else: sig = (cache["r0"][ci][ri], tuple(cache["r1"][ci][ri][i] for i in inds))
                        groups[sig].append(d)
                cm = cc = sp = 0
                cells += len(groups)
                for vals in groups.values():
                    if any(v > 0 for v in vals) and any(v < 0 for v in vals):
                        cm += 1; cc += len(vals); sp = max(sp, max(vals) - min(vals))
                mixed += cm; mixed_cases += cc; max_span = max(max_span, sp)
                nm = label(cand)
                if nm in targets[bn]:
                    target[nm] = {"mixed_cells": cm, "mixed_cases": cc, "max_span": sp, "cells": len(groups)}
            result[bn][sn] = {
                "cells": cells, "mixed_cells": mixed, "mixed_cases": mixed_cases,
                "max_within_cell_regret_span": max_span, "targets": target,
            }
        # Perfect truth upper bound on top of full semantic R1.
        cells = mixed = mixed_cases = max_span = 0; target = {}
        for cand in range(1, oracle.candidates):
            groups = collections.defaultdict(list)
            for ci in range(oracle.cases):
                for pi, step in enumerate(probes):
                    x = oracle.rows[ci][pi][cand]
                    if not x[2]: continue
                    b = oracle.rows[ci][pi][0]; d = x[0] - x[1] - (b[0] - b[1])
                    ri = cache["probe_index"][step]
                    groups[(cache["r0"][ci][ri], cache["r1"][ci][ri], cache["truth"][ci][ri])].append(d)
            cm = cc = sp = 0; cells += len(groups)
            for vals in groups.values():
                if any(v > 0 for v in vals) and any(v < 0 for v in vals):
                    cm += 1; cc += len(vals); sp = max(sp, max(vals) - min(vals))
            mixed += cm; mixed_cases += cc; max_span = max(max_span, sp)
            nm = label(cand)
            if nm in targets[bn]: target[nm] = {"mixed_cells": cm, "mixed_cases": cc, "max_span": sp, "cells": len(groups)}
        result[bn]["R0+R1+TRUTH"] = {"cells": cells, "mixed_cells": mixed, "mixed_cases": mixed_cases, "max_within_cell_regret_span": max_span, "targets": target}
    return result


def feature_resolution(r0, r1, cache) -> dict:
    names = r1_manifest(); out = {}
    for bn, oracle, probes, _ in branches():
        split = collections.Counter(); resolve = collections.Counter(); total = 0
        for cand in range(1, oracle.candidates):
            groups = collections.defaultdict(list)
            for ci in range(oracle.cases):
                for pi, step in enumerate(probes):
                    x = oracle.rows[ci][pi][cand]
                    if not x[2]: continue
                    b = oracle.rows[ci][pi][0]; d = x[0] - x[1] - (b[0] - b[1]); ri = r0.probes.index(step)
                    groups[cache["r0"][ci][ri]].append((ci, ri, d))
            for vals in groups.values():
                if not (any(x[2] > 0 for x in vals) and any(x[2] < 0 for x in vals)): continue
                total += 1
                for fi, name in enumerate(names):
                    g = collections.defaultdict(list)
                    for ci, ri, d in vals: g[cache["r1"][ci][ri][fi]].append(d)
                    if len(g) > 1: split[name] += 1
                    if len(g) > 1 and all(not (any(d > 0 for d in ds) and any(d < 0 for d in ds)) for ds in g.values()):
                        resolve[name] += 1
        out[bn] = {"r0_mixed_cells": total, "top_split_features": split.most_common(12), "top_fully_resolving_features": resolve.most_common(12)}
    return out


def paired_histories(r0, r1, cache) -> dict:
    names = r1_manifest(); out = {}
    for bn, oracle, probes, label in branches():
        split_cells = fully_resolved = 0; examples = []
        for cand in range(1, oracle.candidates):
            groups = collections.defaultdict(list)
            for ci in range(oracle.cases):
                for pi, step in enumerate(probes):
                    x = oracle.rows[ci][pi][cand]
                    if not x[2]: continue
                    b = oracle.rows[ci][pi][0]; d = x[0] - x[1] - (b[0] - b[1]); ri = r0.probes.index(step)
                    groups[cache["r0"][ci][ri]].append((ci, step, d, cache["r1"][ci][ri]))
            for vals in groups.values():
                if not (any(x[2] > 0 for x in vals) and any(x[2] < 0 for x in vals)): continue
                hist = collections.defaultdict(list)
                for x in vals: hist[x[3]].append(x)
                if len(hist) <= 1: continue
                split_cells += 1
                if all(not (any(x[2] > 0 for x in vv) and any(x[2] < 0 for x in vv)) for vv in hist.values()): fully_resolved += 1
                pos = [x for x in vals if x[2] > 0]; neg = [x for x in vals if x[2] < 0]
                if pos and neg:
                    a, b = pos[0], neg[0]
                    diffs = [(names[i], a[3][i], b[3][i]) for i in range(r1.width) if a[3][i] != b[3][i]]
                    examples.append({"choice": label(cand), "step": a[1], "positive_case": a[0], "positive_delta": a[2], "negative_case": b[0], "negative_delta": b[2], "history_diff_count": len(diffs), "first_history_differences": diffs[:12]})
        examples.sort(key=lambda x: abs(x["positive_delta"] - x["negative_delta"]), reverse=True)
        out[bn] = {"r0_mixed_cells_split_by_r1": split_cells, "r0_mixed_cells_fully_resolved_by_r1": fully_resolved, "examples": examples[:5]}
    return out


def screen_gate_metrics(r0, r1, truth, cases) -> dict:
    rm = registry_manifest(); ridx = {x: i for i, x in enumerate(rm)}; rn = r1_manifest(); rni = {x: i for i, x in enumerate(rn)}
    u = load_oracle(OROOT / "oracle_u_screen.b4or"); w = load_oracle(OROOT / "oracle_w_screen_h24.b4or")
    probes = [int(x) for x in (OROOT / "screen_probes_uw.txt").read_text().splitlines() if x.strip()]
    uc = next(i for i in range(u.candidates) if u_candidate_name(i) == "HOLD_BUYS")
    wc = next(i for i in range(w.candidates) if w_candidate_name(i) == "SWAP_6_9")

    def collect(oracle, cand, step, predicate):
        pi = probes.index(step); ri = r0.probes.index(step); by = collections.defaultdict(lambda: {"n": 0, "negative": 0, "sum_delta": 0, "min_delta": 0, "loss_to_win_flips": 0})
        for ci, c in enumerate(cases):
            if not predicate(ci, ri): continue
            x = oracle.rows[ci][pi][cand]
            if not x[2]: continue
            b = oracle.rows[ci][pi][0]; bm = b[0] - b[1]; m = x[0] - x[1]; d = m - bm; s = by[c["role"]]
            s["n"] += 1; s["negative"] += d < 0; s["sum_delta"] += d; s["min_delta"] = min(s["min_delta"], d); s["loss_to_win_flips"] += bm < 0 < m
        return dict(by)

    h0 = collect(u, uc, 120, lambda ci, ri: int(r0.frame(ci, ri)[ridx[("SHOP_AT", (0,))]]) == 6 and int(r0.frame(ci, ri)[ridx[("OPP_MONEY", ())]]) >= 250)
    h1 = collect(w, wc, 192, lambda ci, ri: int(r0.frame(ci, ri)[ridx[("SHOP_AT", (0,))]]) == 5 and int(r1.frame(ci, ri)[rni["delta24.opp_crop_3"]]) >= 2)
    h2 = collect(w, wc, 192, lambda ci, ri: int(r0.frame(ci, ri)[ridx[("SHOP_AT", (0,))]]) == 5 and int(truth.frame(ci, ri)[15]) <= 0)
    return {"H0_current_gate": h0, "H1_temporal_gate": h1, "H2_perfect_truth_gate": h2}


def hard_loss_analysis(r0, r1, truth, cases) -> dict:
    previous = json.loads((ROOT / "tools/bkg4_search/OPTIMIZATION_ORACLE_TERMINAL.json").read_text())
    rows = previous["canary_loss_clusters"]["rows"]
    rm = registry_manifest(); ri = {x: i for i, x in enumerate(rm)}; rn = r1_manifest(); rni = {x: i for i, x in enumerate(rn)}
    shops = ["BAKERY", "BRUNCH_SPOT", "FARMERS_MARKET", "ICE_CREAM_SHOP", "PET_CAFE", "PIZZA_SHOP", "SMOOTHIE_SHOP", "YARN_STORE"]

    def feat(ci, step):
        p = r0.probes.index(step); f = r0.frame(ci, p); t = r1.frame(ci, p); h = truth.frame(ci, p)
        s0 = int(f[ri[("SHOP_AT", (0,))]]); s1 = int(f[ri[("SHOP_AT", (1,))]])
        return {
            "shop0": shops[s0] if s0 >= 0 else None, "shop1": shops[s1] if s1 >= 0 else None,
            "opp_money": int(f[ri[("OPP_MONEY", ())]]), "opp_land": int(f[ri[("OPP_LAND", ())]]), "opp_units": int(f[ri[("OPP_UNITS", ())]]),
            "own_money": int(f[ri[("OWN_MONEY", ())]]), "own_land": int(f[ri[("OWN_LAND", ())]]), "route": int(f[len(rm) + 1]),
            "delta24_opp_money": int(t[rni["delta24.opp_money"]]), "delta24_opp_strawberry": int(t[rni["delta24.opp_crop_3"]]),
            "since_opp_land_increase": int(t[rni["since_opp_land_increase"]]), "truth_opp_wheat": int(h[0]), "truth_opp_strawberry_seed": int(h[15]),
        }

    ryo = [x for x in rows if cases[x["case"]]["team"] == "Ryo Hasegawa"]
    regimes = collections.Counter()
    ryo_rows = []
    for x in ryo:
        f144, f192 = feat(x["case"], 144), feat(x["case"], 192)
        regime = "early_land_then_cash_drawdown" if f144["opp_land"] >= 2 and f192["delta24_opp_money"] <= 0 else "late_land_cash_positive_strawberry_burst"
        regimes[regime] += 1
        ryo_rows.append({"case": x["case"], "seat": cases[x["case"]]["seat"], "base_margin": x["base_margin"], "oracle_cluster": x["cluster"], "trajectory_regime": regime, "at144": f144, "at192": f192})
    none = [x for x in rows if x["cluster"] == "NONE"]
    return {
        "ryo_count": len(ryo), "ryo_oracle_cluster_counts": dict(collections.Counter(x["cluster"] for x in ryo)),
        "ryo_public_trajectory_regimes": dict(regimes), "ryo_rows": ryo_rows,
        "unflippable_count": len(none), "unflippable_team_counts": dict(collections.Counter(cases[x["case"]]["team"] for x in none)),
        "unflippable_rows": [{"case": x["case"], "team": cases[x["case"]]["team"], "seat": cases[x["case"]]["seat"], "base_margin": x["base_margin"], "at144": feat(x["case"], 144), "at192": feat(x["case"], 192)} for x in none],
    }


def metamorphic(r0, r1) -> dict:
    names = r1_manifest(); idx = names.index("delta24.opp_crop_3")
    # Rule diagnostic only: first shop PIZZA and delta24 opponent strawberry count >=2.
    # Perturb every unrelated temporal component independently; decision must remain unchanged.
    chosen = None
    rm = registry_manifest(); ridx = {x: i for i, x in enumerate(rm)}
    for ci in range(r1.cases):
        pi = r0.probes.index(192)
        if int(r0.frame(ci, pi)[ridx[("SHOP_AT", (0,))]]) == 5 and int(r1.frame(ci, pi)[idx]) >= 2:
            chosen = (ci, pi); break
    if chosen is None: raise RuntimeError("no temporal metamorphic source state")
    ci, pi = chosen; base = [int(x) for x in r1.frame(ci, pi)]
    pred = lambda vec, shop=5: shop == 5 and vec[idx] >= 2
    original = pred(base)
    preserved = 0
    for j in range(len(base)):
        if j == idx: continue
        v = list(base); v[j] += 12345 if v[j] < 1000 else -123
        if pred(v) == original: preserved += 1
    relevant = list(base); relevant[idx] = 1
    return {
        "diagnostic_rule": "first_shop==PIZZA && delta24(opponent STRAWBERRY planted count)>=2",
        "irrelevant_temporal_components_perturbed": len(base) - 1,
        "irrelevant_perturbations_preserved_decision": preserved,
        "causal_feature_change_changed_decision": pred(relevant) != original,
        "current_shop_context_change_changed_decision": pred(base, shop=6) != original,
    }


def main() -> None:
    cases = load_cases(); r0 = load_r0(OROOT / "r0_complete.b4is"); r1 = load_r1(OROOT / "r1_semantic.b4is"); truth = load_hidden_truth(OROOT / "hidden_truth.b4is")
    if not (r0.cases == r1.cases == truth.cases == len(cases) and r0.probes == r1.probes == truth.probes): raise ValueError("capture alignment")
    cache = build_state_cache(r0, r1, truth)
    h0_names = ["BASE"] + [f"OPP_MONEY_GE_{x}" for x in (250, 300, 350, 400, 450, 500, 525, 550, 570)]
    h1_names = ["BASE"] + [f"STRAW_GROWTH_GE_{t}_H{h}" for t in (1, 2, 3) for h in (12, 24, 36, 48)]
    h2_names = ["BASE", "TRUE_WHEAT_SEED0_HOLD_BUYS", "TRUE_STRAW_SEED0_SWAP69_H24", "TRUE_STRAW_SEED0_SWAP69_H48"]
    row_h0 = struct.Struct("<iiHH"); row_hx = struct.Struct("<iiHHHH")
    leagues = {
        "H0": {
            "historical336": score_result(SCRATCH / "h0_hold_buys_historical336.b4h0", SCRATCH / "fullcontroller_historical336.tsv", h0_names, row_h0),
            "fresh24": score_result(SCRATCH / "h0_hold_buys_fresh24.b4h0", SCRATCH / "fullcontroller_fresh24.tsv", h0_names, row_h0),
            "sealed24": score_result(SCRATCH / "h0_hold_buys_sealed24.b4h0", SCRATCH / "fullcontroller_animal_sealed24.tsv", h0_names, row_h0),
        },
        "H1": {
            "historical336": score_result(SCRATCH / "h1_temporal_swap_historical336.b4h1", SCRATCH / "fullcontroller_historical336.tsv", h1_names, row_hx),
            "fresh24": score_result(SCRATCH / "h1_temporal_swap_fresh24.b4h1", SCRATCH / "fullcontroller_fresh24.tsv", h1_names, row_hx),
            "sealed24": score_result(SCRATCH / "h1_temporal_swap_sealed24.b4h1", SCRATCH / "fullcontroller_animal_sealed24.tsv", h1_names, row_hx),
        },
        "H2": {
            "historical336": score_result(SCRATCH / "h2_truth_historical336.b4h2", SCRATCH / "fullcontroller_historical336.tsv", h2_names, row_hx),
            "fresh24": score_result(SCRATCH / "h2_truth_fresh24.b4h2", SCRATCH / "fullcontroller_fresh24.tsv", h2_names, row_hx),
            "sealed24": score_result(SCRATCH / "h2_truth_sealed24.b4h2", SCRATCH / "fullcontroller_animal_sealed24.tsv", h2_names, row_hx),
        },
    }
    data = {
        "schema": 1,
        "final_verdict": "TEMPORAL INFORMATION-STATE HYPOTHESIS TERMINALLY REJECTED",
        "r0": {"cases": r0.cases, "probes": len(r0.probes), "width": r0.width, "registry_width": r0.registry_width, "bb_width": r0.bb_width, "compat_width": r0.compat_width, "meta_width": r0.meta_width, "artifact_sha256": sha256(r0.path)},
        "r1": {"width": r1.width, "artifact_sha256": sha256(r1.path), "horizons": [1, 4, 24, 72], "semantic_only": True},
        "h2_truth": {"width": truth.width, "artifact_sha256": sha256(truth.path), "variables": list(HIDDEN_TRUTH_NAMES), "production_input": False, "observer_status": "NOT_IMPLEMENTED_PERFECT_TRUTH_UPPER_BOUND_FAILED", "containment_rate": None, "interval_width": None, "exactness_rate": None},
        "ambiguity": ambiguity(r0, r1, truth, cache),
        "temporal_feature_resolution": feature_resolution(r0, r1, cache),
        "paired_histories": paired_histories(r0, r1, cache),
        "screen_gates": screen_gate_metrics(r0, r1, truth, cases),
        "league_validation": leagues,
        "metamorphic": metamorphic(r0, r1),
        "hard_losses": hard_loss_analysis(r0, r1, truth, cases),
        "active_information_probe": {"tested": False, "reason": "R2 perfect-truth upper bound added zero marginal representation separation and truth-gated policies were unsafe; active probing prerequisites were not met."},
        "promotion": {"promoted": False, "controller_hash": None, "combined_identity": None, "new_nodes": 0, "new_bb": 0, "new_state": 0},
    }
    OUT.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"out": str(OUT), "verdict": data["final_verdict"], "R0_U_mixed": data["ambiguity"]["U"]["R0"]["mixed_cells"], "R1_U_mixed": data["ambiguity"]["U"]["R0+events"]["mixed_cells"], "R2_U_mixed": data["ambiguity"]["U"]["R0+R1+TRUTH"]["mixed_cells"], "R0_W_mixed": data["ambiguity"]["W"]["R0"]["mixed_cells"], "R1_W_mixed": data["ambiguity"]["W"]["R0+events"]["mixed_cells"], "R2_W_mixed": data["ambiguity"]["W"]["R0+R1+TRUTH"]["mixed_cells"], "R0_M_mixed": data["ambiguity"]["M"]["R0"]["mixed_cells"], "R1_M_mixed": data["ambiguity"]["M"]["R0+events"]["mixed_cells"], "R2_M_mixed": data["ambiguity"]["M"]["R0+R1+TRUTH"]["mixed_cells"]}, sort_keys=True))


if __name__ == "__main__":
    main()
