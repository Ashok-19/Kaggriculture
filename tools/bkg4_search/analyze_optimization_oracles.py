from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.bkg4_search.optimization_oracle_io import (
    load,
    m_candidate_name,
    u_candidate_name,
    w_candidate_name,
)

ROOT = Path(__file__).resolve().parents[2]
ROUTES = (22, 23, 29, 32, 34, 46, 63, 64, 70, 77, 79)
HORIZONS = (24, 72, 144)


def qtile(xs: list[int], q: float) -> float:
    if not xs:
        return 0.0
    ys = sorted(xs)
    pos = q * (len(ys) - 1)
    lo = int(pos)
    hi = min(len(ys) - 1, lo + 1)
    f = pos - lo
    return ys[lo] * (1.0 - f) + ys[hi] * f


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def branch_summary(path: Path, cases: list[dict], probes: list[int], labeler) -> tuple[dict, list[int], list[int], list[tuple[int, int]]]:
    r = load(path)
    if r.cases != len(cases) or r.probes != len(probes):
        raise RuntimeError(f"shape mismatch: {path}")
    base: list[int] = []
    best: list[int] = []
    choices: list[tuple[int, int]] = []
    state_regrets: list[int] = []
    active_jobs = 0
    alternative_regression_jobs = 0
    top = Counter()
    for ci in range(len(cases)):
        b0 = None
        game_best = -(1 << 60)
        game_choice = (0, 0)
        for pi, _step in enumerate(probes):
            row = r.rows[ci][pi]
            bm = row[0][0] - row[0][1]
            if row[0][3]:
                raise RuntimeError("baseline runtime flag")
            if b0 is None:
                b0 = bm
            elif bm != b0:
                raise RuntimeError("baseline drift across probes")
            state_best = bm
            any_active = False
            for cand, x in enumerate(row[1:], 1):
                if x[3]:
                    raise RuntimeError(f"candidate runtime flag: {path} case={ci} probe={pi} cand={cand}")
                if not x[2]:
                    continue
                active_jobs += 1
                any_active = True
                margin = x[0] - x[1]
                alternative_regression_jobs += margin < bm
                state_best = max(state_best, margin)
                if margin > game_best:
                    game_best = margin
                    game_choice = (pi, cand)
            if any_active:
                state_regrets.append(state_best - bm)
        assert b0 is not None
        if game_best < b0:
            game_best = b0
            game_choice = (0, 0)
        base.append(b0)
        best.append(game_best)
        choices.append(game_choice)
        if game_choice[1]:
            top[labeler(game_choice[1])] += 1
    delta = [g - b for g, b in zip(best, base)]
    roles = defaultdict(list)
    for i, case in enumerate(cases):
        roles[case["role"]].append(i)
    role_stats = {}
    for role, idx in roles.items():
        role_stats[role] = {
            "cases": len(idx),
            "base_losses": sum(base[i] < 0 for i in idx),
            "loss_to_win_flips": sum(base[i] < 0 < best[i] for i in idx),
            "positive_best_delta": sum(delta[i] > 0 for i in idx),
            "mean_best_delta": statistics.mean(delta[i] for i in idx),
        }
    groups = defaultdict(list)
    for i, case in enumerate(cases):
        groups[(case["tape"], case["seed"], case["team"])].append(i)
    seat_pairs = [idx for idx in groups.values() if {cases[i]["seat"] for i in idx} == {0, 1}]
    report = {
        "artifact": str(path.relative_to(ROOT)),
        "artifact_sha256": sha256(path),
        "cases": len(cases),
        "baseline_wins": sum(x > 0 for x in base),
        "baseline_losses": sum(x < 0 for x in base),
        "activation_jobs": active_jobs,
        "activation_states": len(state_regrets),
        "positive_regret_states": sum(x > 0 for x in state_regrets),
        "state_regret_mean": statistics.mean(state_regrets),
        "state_regret_median": statistics.median(state_regrets),
        "state_regret_p90": qtile(state_regrets, 0.90),
        "state_regret_max": max(state_regrets),
        "game_positive_best_delta": sum(x > 0 for x in delta),
        "game_best_delta_mean": statistics.mean(delta),
        "game_best_delta_median": statistics.median(delta),
        "game_best_delta_p90": qtile(delta, 0.90),
        "game_best_delta_max": max(delta),
        "loss_to_win_flips": sum(b < 0 < g for b, g in zip(base, best)),
        "remaining_losses": sum(g <= 0 for g in best),
        "alternative_regression_jobs": alternative_regression_jobs,
        "paired_seat_groups": len(seat_pairs),
        "both_seats_positive_best_delta": sum(all(delta[i] > 0 for i in idx) for idx in seat_pairs),
        "role_stats": role_stats,
        "top_hindsight_choices": top.most_common(12),
    }
    return report, base, best, choices


def public_ambiguity(path: Path, cases: list[dict], oracle_path: Path, probes: list[int], target: str) -> dict:
    frames = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    r = load(oracle_path)
    cand = next(i for i in range(r.candidates) if u_candidate_name(i) == target)
    out = {}
    for step in (96, 120, 144):
        pi = probes.index(step)
        groups = defaultdict(list)
        for f in frames:
            if f["step"] != step:
                continue
            ci = int(f["case"])
            signature = tuple(sorted((k, json.dumps(v, sort_keys=True)) for k, v in f.items() if k not in {"case", "role", "seat"}))
            b = r.rows[ci][pi][0]
            x = r.rows[ci][pi][cand]
            bm = b[0] - b[1]
            delta = (x[0] - x[1]) - bm
            groups[signature].append(delta)
        ambiguous = [v for v in groups.values() if any(x > 0 for x in v) and any(x < 0 for x in v)]
        out[str(step)] = {
            "unique_public_frames": len(groups),
            "ambiguous_frames": len(ambiguous),
            "ambiguous_cases": sum(len(v) for v in ambiguous),
            "max_within_frame_delta_span": max((max(v) - min(v) for v in ambiguous), default=0),
        }
    return out


def loss_clusters(cases: list[dict], branches: dict[str, tuple], probes_by_branch: dict[str, list[int]], labelers: dict[str, object]) -> dict:
    cluster = Counter()
    rows = []
    for ci, case in enumerate(cases):
        if case["role"] != "canary_discovery" or int(case.get("reference_margin") or 0) >= 0:
            continue
        flips = []
        detail = {}
        for name, (r, base, _best, _choice) in branches.items():
            b = base[ci]
            best_margin = b
            best_step = None
            best_name = None
            for pi, step in enumerate(probes_by_branch[name]):
                for cand, x in enumerate(r.rows[ci][pi][1:], 1):
                    if not x[2]:
                        continue
                    margin = x[0] - x[1]
                    if margin > best_margin:
                        best_margin = margin
                        best_step = step
                        best_name = labelers[name](cand)
            if best_margin > 0:
                flips.append(name)
            detail[name] = {"best_margin": best_margin, "delta": best_margin - b, "step": best_step, "choice": best_name}
        key = "+".join(flips) if flips else "NONE"
        cluster[key] += 1
        rows.append({"case": ci, "base_margin": int(case["reference_margin"]), "cluster": key, **detail})
    if len(rows) != 27:
        raise RuntimeError(f"expected 27 canary losses, got {len(rows)}")
    return {"clusters": dict(sorted(cluster.items())), "rows": rows}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("scratch/bkg4_search/optimization_oracle"))
    ap.add_argument("--out", type=Path, default=Path("tools/bkg4_search/OPTIMIZATION_ORACLE_TERMINAL.json"))
    args = ap.parse_args()
    root = (ROOT / args.root).resolve() if not args.root.is_absolute() else args.root
    out = (ROOT / args.out).resolve() if not args.out.is_absolute() else args.out
    cases = [json.loads(x) for x in (root / "screen_cases.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    pu = [int(x) for x in (root / "screen_probes_uw.txt").read_text().splitlines() if x.strip()]
    pm = [int(x) for x in (root / "probes_m.txt").read_text().splitlines() if x.strip()]
    up = root / "oracle_u_screen.b4or"
    wp = root / "oracle_w_screen_h24.b4or"
    mp = root / "oracle_m_screen.b4or"
    u_report, u_base, u_best, u_choice = branch_summary(up, cases, pu, u_candidate_name)
    w_report, w_base, w_best, w_choice = branch_summary(wp, cases, pu, w_candidate_name)
    m_label = lambda i: m_candidate_name(i, ROUTES, HORIZONS)
    m_report, m_base, m_best, m_choice = branch_summary(mp, cases, pm, m_label)
    if u_base != w_base or u_base != m_base:
        raise RuntimeError("cross-oracle baseline mismatch")
    branches = {
        "U": (load(up), u_base, u_best, u_choice),
        "W": (load(wp), w_base, w_best, w_choice),
        "M": (load(mp), m_base, m_best, m_choice),
    }
    payload = {
        "schema": 1,
        "screen_manifest_sha256": sha256(root / "screen_cases.jsonl"),
        "screen_cases": len(cases),
        "screen_roles": dict(Counter(c["role"] for c in cases)),
        "oracles": {"U": u_report, "W": w_report, "M": m_report},
        "U_public_ambiguity": {
            "HOLD_BUYS": public_ambiguity(root / "public_features_96_120_144.jsonl", cases, up, pu, "HOLD_BUYS"),
            "HOLD_SELL": public_ambiguity(root / "public_features_96_120_144.jsonl", cases, up, pu, "HOLD_SELL"),
        },
        "canary_loss_clusters": loss_clusters(cases, branches, {"U": pu, "W": pu, "M": pm}, {"U": u_candidate_name, "W": w_candidate_name, "M": m_label}),
        "distillation_terminal": {
            "U": {
                "best_discovery_safe_gate": "step==144 AND second_shop==PIZZA_SHOP AND opp_money>=175 -> HOLD_BUYS",
                "discovery_activations": 4,
                "discovery_flips": 2,
                "discovery_min_delta": 2742,
                "independent_holdout_activations": 0,
                "status": "REJECTED_NO_HOLDOUT_SUPPORT",
            },
            "W": {
                "best_discovery_safe_gate": "step==264 AND market_orders_sell<=1 AND opp_tiles_weed>=1 -> SWAP_10_12 for 24 turns",
                "discovery_activations": 4,
                "discovery_flips": 1,
                "fresh_holdout_activations": 3,
                "fresh_holdout_negative": 2,
                "fresh_holdout_sum_delta": -8232,
                "fresh_holdout_min_delta": -4490,
                "status": "REJECTED_HOLDOUT_REGRESSION",
            },
            "M": {
                "best_discovery_safe_gate": "step==144 AND second_shop==SMOOTHIE_SHOP AND opp_tiles_weed==0 -> route23 for 72 turns",
                "discovery_activations": 4,
                "discovery_flips": 2,
                "fresh_holdout_activations": 6,
                "fresh_holdout_negative": 6,
                "fresh_holdout_sum_delta": -93100,
                "fresh_holdout_min_delta": -36103,
                "status": "REJECTED_HOLDOUT_REGRESSION",
            },
        },
        "final_verdict": "OPTIMIZATION_PROPOSALS_TERMINALLY_REJECTED",
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(out), "verdict": payload["final_verdict"], "oracles": {k: {"flips": v["loss_to_win_flips"], "mean": v["game_best_delta_mean"]} for k, v in payload["oracles"].items()}}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
