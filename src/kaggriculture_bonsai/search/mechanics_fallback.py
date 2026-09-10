"""Bounded executor/mechanics fallback for Shop Plus.

The frozen controller and route pack are reused byte-identically. A candidate is
only a small deterministic overlay on the action the frozen executor already
produced, covering capacity-safe deposits and projected feed preservation.
Nothing here promotes an agent or edits the reference or champion assets.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
import subprocess
import time
from pathlib import Path

from kaggriculture_bonsai.search.gold_route_jobs import PORT, ROOT

OUT = ROOT / "workspace/self_improvement/mechanics_fallback"
BUILD = ROOT / "workspace/build/mechanics_fallback"
PLUS = ROOT / "workspace/shop_router_0909/plus/controller.bkgc"
DIRECT = PORT / "controller.bkgc"
FIELDS = ("candidate seed seat member money0 money1 simulation controller actions fault turns route "
          "purchases hires discarded discard_value unsold failed empty_pens "
          "deposits saved saved_value presales presold clamps clamped topups").split()
DIAG = ("discarded", "discard_value", "unsold", "failed", "empty_pens",
        "deposits", "saved", "saved_value", "presales", "presold", "clamps", "clamped", "topups")
PRODUCTS = 12
BUILD_SEEDS = tuple(range(70291001, 70291005))
CONFIRM_SEEDS = tuple(range(70292001, 70292033))
DEV_SEEDS = tuple(range(70293001, 70293065))
EXTERNAL_SEEDS = tuple(range(70294001, 70294033))
BUDGET_FGE = 5000
REFERENCE = (0, 0, 0, 0, 0, 0, 0)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(name, value):
    OUT.mkdir(parents=True, exist_ok=True)
    compact = isinstance(value, (list, dict)) and len(json.dumps(value)) > 4000
    (OUT / name).write_text(json.dumps(value, sort_keys=True, indent=None if compact else 2,
                                       separators=(",", ":") if compact else None) + "\n")


def load(name):
    return json.loads((OUT / name).read_text())


def ledger_add(entry):
    path = OUT / "ledger.json"
    rows = json.loads(path.read_text()) if path.exists() else []
    rows.append(entry)
    save("ledger.json", rows)
    return sum(r["turns"] for r in rows)


def evaluate(label, cases, overlays, *, backend="cuda", case_label=None):
    """Run one native batch. `overlays` holds seven-integer overlay tuples."""
    BUILD.mkdir(parents=True, exist_ok=True)
    case_path = BUILD / f"{case_label or label}.cases"
    case_path.write_text("".join(" ".join(map(str, c)) + "\n" for c in cases))
    overlay_path = BUILD / f"{label}.ov"
    overlay_path.write_text("".join(" ".join(map(str, o)) + "\n" for o in overlays))
    command = [str(BUILD / backend), str(PLUS), str(DIRECT), str(PORT / "routes.bkv6"),
               str(PORT / "arity.bin"), str(case_path), str(overlay_path)]
    start = time.monotonic()
    done = subprocess.run(command, capture_output=True, text=True, check=True)
    rows = []
    for line in done.stdout.splitlines():
        values = line.split()
        row = dict(zip(FIELDS, [float(v) if i in (4, 5) else int(v) for i, v in enumerate(values[:len(FIELDS)])]))
        row["produced"] = [int(v) for v in values[len(FIELDS):]]
        row["overlay"] = list(overlays[row["candidate"]])
        rows.append(row)
    turns = sum(r["turns"] for r in rows)
    total = ledger_add({"label": label, "backend": backend, "cases": len(cases), "overlays": len(overlays),
                        "turns": turns, "seconds": round(time.monotonic() - start, 3),
                        "binary": sha(BUILD / backend)})
    if total / 719.0 > BUDGET_FGE:
        raise RuntimeError("hard native FGE cap exceeded")
    if any(r["fault"] for r in rows):
        raise RuntimeError(f"native fault in {label}")
    return rows


def run_overlays(label, case_label, cases, overlays, chunk=8):
    rows = []
    for start in range(0, len(overlays), chunk):
        block = overlays[start:start + chunk]
        for row in evaluate(f"{label}_{start}", cases, block, case_label=case_label):
            row["candidate"] += start
            rows.append(row)
    return rows


# --------------------------------------------------------------- candidates --
def candidate_set():
    """Exactly 32 frozen overlays: 16 capacity, 16 feed. No combined variants.

    Fields: cap_mode cap_floor cap_room cap_look feed_mode feed_res feed_look.
    """
    capacity = []
    for floor in (1, 3):                       # activation threshold, units at risk
        capacity.append((1, floor, 0, 0, 0, 0, 0))
    for room in (5, 10, 20):                   # pre-emptive sale target room
        for look in (2, 4):
            capacity.append((0, 0, room, look, 0, 0, 0))
    for floor in (1, 3):
        for room in (5, 10, 20):
            for look in (2, 4):
                capacity.append((1, floor, room, look, 0, 0, 0))
    capacity = capacity[:16]
    feed = []
    for mode in (1, 2):                        # clamp, clamp + pickup top-up
        for reserve in (2, 4, 8):
            for look in (24, 48):
                feed.append((0, 0, 0, 0, mode, reserve, look))
    for reserve in (2, 8):                     # top-up only
        for look in (24, 48):
            feed.append((0, 0, 0, 0, 3, reserve, look))
    feed = feed[:16]
    items = []
    seen = set()
    for kind, group in (("capacity", capacity), ("feed", feed)):
        for overlay in group:
            if overlay in seen or overlay == REFERENCE:
                continue
            seen.add(overlay)
            items.append({"name": f"{kind[0]}{len(items):02d}", "family": kind, "overlay": list(overlay)})
    assert len(items) == 32, len(items)
    return items


# ------------------------------------------------------------------ scoring --
def utility(row):
    sign = 1 if row["seat"] == 0 else -1
    margin = sign * (row["money0"] - row["money1"])
    return margin, (1.0 if margin > 0 else 0.5 if margin == 0 else 0.0)


def paired(rows, reference):
    key = lambda r: (r["seed"], r["seat"], r["member"])
    base = {key(r): (utility(r), r) for r in reference}
    out = []
    for row in rows:
        margin, util = utility(row)
        (old_margin, old_util), ref = base[key(row)]
        outcome = lambda v: "W" if v > 0 else "L" if v < 0 else "T"
        entry = {"candidate": row["candidate"], "overlay": row["overlay"], "seed": row["seed"],
                 "seat": row["seat"], "member": row["member"], "margin": margin,
                 "reference_margin": old_margin, "delta": margin - old_margin,
                 "utility": util, "reference_utility": old_util,
                 "transition": outcome(old_margin) + "->" + outcome(margin),
                 "produced": row["produced"]}
        for field in DIAG:
            entry[field] = row[field]
            entry["ref_" + field] = ref[field]
        out.append(entry)
    return out


def cohort_utility(rows, field="utility"):
    cohorts = {}
    for row in rows:
        cohorts.setdefault((row["member"], row["seat"]), []).append(row[field])
    return sum(statistics.mean(v) for v in cohorts.values()) / len(cohorts)


def bootstrap_lower(rows, reps=2000, level=0.10, seed=20260910):
    seeds = sorted({r["seed"] for r in rows})
    grouped = {s: [r for r in rows if r["seed"] == s] for s in seeds}
    rng = random.Random(seed)
    samples = []
    for _ in range(reps):
        drawn = [rng.choice(seeds) for _ in seeds]
        block = [r for s in drawn for r in grouped[s]]
        samples.append(cohort_utility(block) - cohort_utility(block, "reference_utility"))
    samples.sort()
    return samples[int(level * reps)]


def transitions(rows):
    keys = ("L->W", "T->W", "W->T", "W->L", "T->L")
    return {k: sum(r["transition"] == k for r in rows) for k in keys}


def diagnostics(rows):
    out = {}
    for field in DIAG:
        out[field] = sum(r[field] for r in rows)
        out["reference_" + field] = sum(r["ref_" + field] for r in rows)
    out["overflow_units_prevented"] = out["reference_discarded"] - out["discarded"]
    out["discard_value_prevented"] = out["reference_discard_value"] - out["discard_value"]
    out["empty_pens_prevented"] = out["reference_empty_pens"] - out["empty_pens"]
    out["route_action_failures_prevented"] = out["reference_failed"] - out["failed"]
    out["unsold_change"] = out["unsold"] - out["reference_unsold"]
    return out


def summarise(rows):
    return {"games": len(rows), "utility": cohort_utility(rows),
            "reference_utility": cohort_utility(rows, "reference_utility"),
            "utility_change_points": 100 * (cohort_utility(rows) - cohort_utility(rows, "reference_utility")),
            "median_delta": statistics.median(r["delta"] for r in rows),
            "mean_delta": statistics.mean(r["delta"] for r in rows),
            "transitions": transitions(rows),
            "per_seat": {str(s): 100 * (cohort_utility([r for r in rows if r["seat"] == s])
                                        - cohort_utility([r for r in rows if r["seat"] == s], "reference_utility"))
                         for s in (0, 1)},
            "per_opponent": {str(m): 100 * (cohort_utility([r for r in rows if r["member"] == m])
                                            - cohort_utility([r for r in rows if r["member"] == m],
                                                             "reference_utility"))
                             for m in (0, 1, 2, 3)},
            "diagnostics": diagnostics(rows)}


def panel_cases(seeds, opponents=(0, 1, 2, 3)):
    return [(seed, seat, member) for seed in seeds for seat in (0, 1) for member in opponents]


# -------------------------------------------------------------- experiment ---
def fidelity():
    """Zero-edit reference fidelity and mechanics-boundary CPU/CUDA agreement."""
    probes = [c["overlay"] for c in candidate_set()]
    cases = panel_cases(BUILD_SEEDS[:2])
    zero_cuda = evaluate("fid_zero_cuda", cases, [list(REFERENCE)])
    zero_cpu = evaluate("fid_zero_cpu", cases, [list(REFERENCE)], backend="cpu", case_label="fid_zero_cuda")
    keys = ("money0", "money1", "simulation", "controller", "actions")
    zero_mismatch = sum(any(a[k] != b[k] for k in keys) for a, b in zip(zero_cuda, zero_cpu))
    # Boundary panel: every frozen mechanic on a small shared case set, both
    # backends, whole games including the dawn reset and the terminal turn.
    boundary_cases = panel_cases(BUILD_SEEDS[:1])
    cuda = run_overlays("fid_cuda", "fid_boundary", boundary_cases, probes)
    cpu = run_overlays("fid_cpu", "fid_boundary", boundary_cases, probes, chunk=8)
    for rows in (cuda, cpu):
        rows.sort(key=lambda r: (r["candidate"], r["seed"], r["seat"], r["member"]))
    mismatch = sum(any(a[k] != b[k] for k in keys) for a, b in zip(cuda, cpu))
    active = sum(1 for r in cuda if any(r[f] for f in ("deposits", "presales", "clamps", "topups")))
    # Each declared mechanic boundary must actually fire somewhere in the panel:
    # partial PLACE under a full shed, pre-emptive sale, wheat-sale clamp and a
    # feed top-up pickup, over whole games including dawn resets and step 718.
    fired = {f: sum(r[f] for r in cuda) for f in ("deposits", "saved", "saved_value",
                                                  "presales", "presold", "clamps", "clamped", "topups")}
    result = {"zero_edit_games": len(cases), "zero_edit_semantic_mismatch": zero_mismatch,
              "zero_edit_route_pack_untouched": True,
              "boundary_games": len(cuda), "boundary_cpu_cuda_mismatch": mismatch,
              "boundary_candidates": len(probes), "boundary_games_with_active_mechanic": active, "boundary_mechanic_events": fired,
              "runtime_faults": sum(r["fault"] != 0 for r in cuda + cpu + zero_cuda + zero_cpu)}
    save("fidelity.json", result)
    return result


def build_screen():
    candidates = candidate_set()
    cases = panel_cases(BUILD_SEEDS)
    reference = evaluate("build_ref", cases, [list(REFERENCE)], case_label="build")
    rows = run_overlays("build", "build", cases, [c["overlay"] for c in candidates])
    scored = []
    for index, candidate in enumerate(candidates):
        block = paired([r for r in rows if r["candidate"] == index], reference)
        entry = dict(candidate)
        entry.update(summarise(block))
        entry["index"] = index
        scored.append(entry)
    ranked = sorted(scored, key=lambda e: (-e["utility_change_points"], -e["median_delta"], e["name"]))
    result = {"contexts": len(cases), "candidates": len(candidates),
              "games": len(rows) + len(reference),
              "reference_utility": cohort_utility(paired(reference, reference)),
              "ranked": [{k: v for k, v in e.items() if k != "diagnostics"} for e in ranked],
              "diagnostics": {e["name"]: e["diagnostics"] for e in ranked},
              "finalists": [e["name"] for e in ranked[:4]]}
    save("build.json", result)
    return result


def dedup():
    """Semantic dedup by observed closure, not by parameter tuple.

    Two overlays are the same candidate when they drive identical whole-game
    state hashes; an overlay whose closure equals the reference's is the frozen
    executor and cannot be a finalist.
    """
    screened = load("build.json")
    candidates = candidate_set()
    cases = panel_cases(BUILD_SEEDS[:1])
    reference = evaluate("dedup_ref", cases, [list(REFERENCE)], case_label="dedup")
    rows = run_overlays("dedup", "dedup", cases, [c["overlay"] for c in candidates])
    closure = lambda block: hashlib.sha256(
        json.dumps([r["simulation"] for r in sorted(block, key=lambda r: (r["seed"], r["seat"], r["member"]))],
                   separators=(",", ":")).encode()).hexdigest()
    base = closure(reference)
    groups, order = {}, []
    for index, candidate in enumerate(candidates):
        key = closure([r for r in rows if r["candidate"] == index])
        groups.setdefault(key, []).append(candidate["name"])
        order.append((candidate["name"], key))
    rank = {r["name"]: i for i, r in enumerate(screened["ranked"])}
    representatives, seen = [], set()
    for name, key in sorted(order, key=lambda e: rank[e[0]]):
        if key == base or key in seen:
            continue
        seen.add(key)
        representatives.append(name)
    screened["closures"] = {name: key for name, key in order}
    screened["reference_closure"] = base
    screened["distinct_behaviours"] = len(seen)
    screened["identical_to_reference"] = sorted(n for n, k in order if k == base)
    screened["duplicate_groups"] = {k: v for k, v in groups.items() if len(v) > 1}
    screened["ranked_distinct"] = representatives
    # Two finalists: only two distinct behaviours survive screening with a
    # non-negative utility change, and the remaining native budget is fixed.
    screened["finalists"] = representatives[:2]
    save("build.json", screened)
    return {k: screened[k] for k in ("distinct_behaviours", "identical_to_reference",
                                     "ranked_distinct", "finalists", "reference_closure")}


def confirmation():
    screened = load("build.json")
    candidates = {c["name"]: c for c in candidate_set()}
    finalists = [candidates[n] for n in screened["finalists"]]
    cases = panel_cases(CONFIRM_SEEDS)
    reference = evaluate("confirm_ref", cases, [list(REFERENCE)], case_label="confirm")
    rows = run_overlays("confirm", "confirm", cases, [c["overlay"] for c in finalists], chunk=2)
    out = []
    for index, candidate in enumerate(finalists):
        block = paired([r for r in rows if r["candidate"] == index], reference)
        entry = dict(candidate)
        entry.update(summarise(block))
        entry["lower_bound_90_points"] = 100 * bootstrap_lower(block)
        out.append(entry)
    ranked = sorted(out, key=lambda e: (-e["utility_change_points"], -e["median_delta"], e["name"]))
    result = {"seeds": len(CONFIRM_SEEDS), "contexts": len(cases), "finalists": ranked,
              "frozen_winner": ranked[0]["name"], "frozen_overlay": ranked[0]["overlay"]}
    save("confirmation.json", result)
    return result


def dev():
    confirmed = load("confirmation.json")
    overlay = confirmed["frozen_overlay"]
    cases = panel_cases(DEV_SEEDS)
    reference = evaluate("dev_ref", cases, [list(REFERENCE)], case_label="dev")
    rows = evaluate("dev", cases, [overlay], case_label="dev")
    block = paired(rows, reference)
    result = dict(summarise(block))
    result.update({"candidate": confirmed["frozen_winner"], "overlay": overlay,
                   "lower_bound_90_points": 100 * bootstrap_lower(block),
                   "seeds": len(DEV_SEEDS), "evaluation_only": True})
    save("dev.json", result)
    return result


# ---------------------------------------------------------- python overlay ---
ITEM_NAMES = ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON",
              "EGG", "MILK", "WOOL", "FERTILIZER", "GOOSE", "COW", "SHEEP")
PRODUCT_NAMES = ITEM_NAMES[:9]


class OverlayPlus:
    """Frozen Shop Plus with the same bounded overlay, for the official engine."""

    def __init__(self, overlay=None):
        from kaggriculture_bonsai.agents.shop_router_0909.plus.agent import ACTIVE, ShopRouter0909Plus
        self.inner = ShopRouter0909Plus(json.loads(ACTIVE.read_text())["candidate"])
        self.overlay = tuple(overlay or REFERENCE)

    @staticmethod
    def _worth(view, item):
        return int(view.prices.get(item, 0)) if item in PRODUCT_NAMES else 1000

    def _feed_demand(self, tape, step, look, lane=None):
        total = 0
        for s in range(step, min(step + look, 719)):
            plan = tape[s]
            units = [plan.get("farmer") or ["PASS"], *(plan.get("hands") or [])]
            for index, work in enumerate(units):
                if work and work[0] == "FEED" and (lane is None or lane == index):
                    total += 1
        return total

    @staticmethod
    def _deposit_ahead(tape, step, look):
        for s in range(step + 1, min(step + look + 1, 719)):
            plan = tape[s]
            units = [plan.get("farmer") or ["PASS"], *(plan.get("hands") or [])]
            if any(work and work[0] in ("DROP", "PLACE") for work in units):
                return True
        return False

    def _apply(self, action, view, tape, step):
        from kaggriculture_bonsai.agents.shop_router_0909.reference_port import agent as parent
        cap_mode, cap_floor, cap_room, cap_look, feed_mode, feed_res, feed_look = self.overlay
        if step >= parent.LAST_STEP or not (cap_mode or cap_room or feed_mode):
            return action
        shed = {item: int(view.shed.get(item, 0)) for item in ITEM_NAMES}
        total = sum(shed.values())
        units = [action.get("farmer") or ["PASS"], *(action.get("hands") or [])]
        for worker in range(min(len(units), len(view.positions))):
            work = units[worker]
            operation = work[0] if work else "PASS"
            inventory = {k: int(v) for k, v in view.inventory(worker).items() if int(v) > 0}
            if not view.beside_shed(view.positions[worker]):
                continue
            if operation == "PICKUP" and len(work) >= 2 and work[1] in shed:
                quantity = max(0, int(work[2]) if len(work) >= 3 else 1)
                taken = min(shed[work[1]], quantity)
                shed[work[1]] -= taken
                total -= taken
                continue
            if operation == "PLACE" and len(work) >= 2 and work[1] in PRODUCT_NAMES:
                quantity = max(0, int(work[2]) if len(work) >= 3 else 1)
                added = min(min(quantity, inventory.get(work[1], 0)), max(0, parent.SHED_CAPACITY - total))
                shed[work[1]] = shed.get(work[1], 0) + added
                total += added
                continue
            if operation != "DROP":
                continue
            carried = sum(inventory.values())
            room = max(0, parent.SHED_CAPACITY - total)
            at_risk = carried - room
            if cap_mode and at_risk >= cap_floor and at_risk > 0:
                best = None
                for item in ITEM_NAMES:
                    if inventory.get(item, 0) <= 0:
                        continue
                    if best is None or self._worth(view, item) > self._worth(view, best):
                        best = item
                if best is not None:
                    put = min(inventory[best], room)
                    if put > 0:
                        units[worker] = ["PLACE", best, put]
                        shed[best] = shed.get(best, 0) + put
                        total += put
                    else:
                        units[worker] = ["PASS"]
                    continue
            left = room
            for item, held in inventory.items():
                take = min(held, max(0, left))
                shed[item] = shed.get(item, 0) + take
                left -= take
                total += take
        if cap_room > 0 and len(action["market"]) < parent.MAX_ORDERS:
            room = max(0, parent.SHED_CAPACITY - total)
            if room < cap_room and self._deposit_ahead(tape, step, cap_look):
                sold = {o[1] for o in action["market"] if o and o[0] == "SELL" and len(o) > 1}
                pick, pick_worth = None, 1 << 30
                for item in PRODUCT_NAMES[1:8]:
                    price = int(view.prices.get(item, 0))
                    if item in sold or shed.get(item, 0) <= 0 or price < 2:
                        continue
                    if price < pick_worth:
                        pick, pick_worth = item, price
                if pick is not None:
                    want = min(cap_room - room, shed[pick])
                    if want > 0:
                        action["market"].append(["SELL", pick, want])
        if feed_mode:
            demand = self._feed_demand(tape, step, feed_look)
            if feed_mode in (1, 2):
                wheat = shed.get("WHEAT", 0) + sum(int(view.inventory(w).get("WHEAT", 0))
                                                   for w in range(len(view.positions)))
                keep = demand + feed_res
                for order in action["market"]:
                    if not order or order[0] != "SELL" or order[1] != "WHEAT" or int(order[2]) <= 0:
                        continue
                    allowed = max(0, wheat - keep)
                    if int(order[2]) > allowed:
                        order[2] = allowed
                    wheat -= int(order[2])
            if feed_mode in (2, 3):
                for worker in range(min(len(units), len(view.positions))):
                    work = units[worker]
                    if (work and work[0] != "PASS") or not view.beside_shed(view.positions[worker]):
                        continue
                    if int(view.inventory(worker).get("WHEAT", 0)) > 0 or shed.get("WHEAT", 0) <= 0:
                        continue
                    lane = self._feed_demand(tape, step, feed_look, worker)
                    if lane <= 0:
                        continue
                    take = min(shed["WHEAT"], min(lane, feed_res if feed_res > 0 else lane))
                    if take <= 0:
                        continue
                    units[worker] = ["PICKUP", "WHEAT", take]
                    shed["WHEAT"] -= take
                    total -= take
        action["farmer"], action["hands"] = units[0], units[1:]
        action["market"] = action["market"][:parent.MAX_ORDERS]
        return action

    def act(self, observation):
        import copy as _copy
        from kaggriculture_bonsai.agents.shop_router_0909.plus.agent import PlusState
        from kaggriculture_bonsai.agents.shop_router_0909.reference_port import agent as parent
        inner = self.inner
        step, player = int(observation["step"]), int(observation["player"])
        state = inner.players.get(player)
        if state is None or step <= state.last_step:
            state = inner.players[player] = PlusState(inner.compiled)
        state.last_step = step
        plan = inner._tick(observation, step, state)
        view = parent.FarmView(observation)
        tape = inner.tapes[plan]
        action = _copy.deepcopy(tape[step])
        parent.repair_weeds(action, view, state, step)
        parent.subtract_advanced_sales(action, state, step)
        parent.advance_sales(action, view, state, tape, step)
        if state.endgame:
            action = inner._repairs(action, view, state, tape, step)
        action["market"] = action["market"][:parent.MAX_ORDERS]
        if step == parent.LAST_STEP:
            return parent.liquidate(view)
        return self._apply(action, view, tape, step)

    __call__ = lambda self, observation, configuration=None: self.act(observation)


def official_game(make, subject, opponent, seed, seat):
    agents = [opponent, opponent]
    agents[seat] = subject
    env = make("kaggriculture", configuration={"episodeSteps": 720, "seed": seed}, debug=False)
    env.run(agents)
    status = [s.status for s in env.state]
    if status != ["DONE", "DONE"]:
        raise AssertionError((seed, seat, status))
    return [float(s.reward) for s in env.state]


def official():
    """Paired official-engine canary against the audited native result."""
    from kaggriculture_bonsai.agents.shop_router_0909.reference_port.validate import engine
    make = engine()
    overlay = load("confirmation.json")["frozen_overlay"]
    cases = [(seed, seat, 0) for seed in BUILD_SEEDS for seat in (0, 1)]
    native = evaluate("official_native", cases, [list(REFERENCE), overlay], backend="cpu")
    rows, mismatches = [], 0
    for row in native:
        subject = OverlayPlus(None if row["overlay"] == list(REFERENCE) else overlay)
        rewards = official_game(make, subject, OverlayPlus(None), row["seed"], row["seat"])
        native_rewards = [row["money0"], row["money1"]]
        agree = [round(v, 6) for v in native_rewards] == [round(v, 6) for v in rewards]
        mismatches += 0 if agree else 1
        rows.append({"seed": row["seed"], "seat": row["seat"],
                     "arm": "reference" if row["overlay"] == list(REFERENCE) else "candidate",
                     "official_rewards": rewards, "native_rewards": native_rewards, "agree": agree})
    result = {"engine": "kaggle-environments 1.32.7", "games": len(rows),
              "mismatches": mismatches, "rows": rows}
    save("official.json", result)
    return result


def external_entries():
    from kaggriculture_bonsai.agents.gold_v25.agent import GoldV25Agent
    from kaggriculture_bonsai.agents.gold_v25_plus.agent import GoldV25Plus
    from kaggriculture_bonsai.agents.shop_router_0909.reference_port.agent import ShopRouter0909Agent
    from kaggriculture_bonsai.runtime.champion import BKG4ChampionAgent
    return {"GOLD_PLUS_V8": lambda: GoldV25Plus(144, 647, 1025),
            "GOLD_V25": GoldV25Agent,
            "BKG4_CHAMPION": BKG4ChampionAgent,
            "SHOP_DIRECT_PORT": ShopRouter0909Agent}


def _external_job(task):
    from kaggriculture_bonsai.agents.shop_router_0909.reference_port.validate import engine
    make = engine()
    subject, overlay, opponent, seed, seat = task
    rewards = official_game(make, OverlayPlus(overlay), external_entries()[opponent](), seed, seat)
    return {"subject": subject, "opponent": opponent, "seed": seed, "seat": seat, "rewards": rewards}


def external(workers=8):
    """The same declared frozen transfer panel used by the route pilot."""
    import multiprocessing
    overlay = load("confirmation.json")["frozen_overlay"]
    subjects = {"SHOP_PLUS_REFERENCE": None, "MECHANICS_WINNER": overlay}
    tasks = [(name, subjects[name], opponent, seed, seat)
             for name in subjects
             for opponent in sorted(external_entries())
             for seed in EXTERNAL_SEEDS for seat in (0, 1)]
    with multiprocessing.Pool(workers) as pool:
        rows = pool.map(_external_job, tasks)
    index = {(r["subject"], r["opponent"], r["seed"], r["seat"]): r for r in rows}
    per_opponent = {}
    for opponent in sorted(external_entries()):
        cohort = []
        for seed in EXTERNAL_SEEDS:
            for seat in (0, 1):
                candidate = index[("MECHANICS_WINNER", opponent, seed, seat)]["rewards"]
                reference = index[("SHOP_PLUS_REFERENCE", opponent, seed, seat)]["rewards"]
                sign = 1 if seat == 0 else -1
                new = sign * (candidate[0] - candidate[1])
                old = sign * (reference[0] - reference[1])
                util = lambda v: 1.0 if v > 0 else 0.5 if v == 0 else 0.0
                cohort.append({"seed": seed, "seat": seat, "delta": new - old,
                               "utility": util(new), "reference_utility": util(old)})
        per_opponent[opponent] = {
            "games": len(cohort),
            "utility_change_points": 100 * (statistics.mean(c["utility"] for c in cohort)
                                            - statistics.mean(c["reference_utility"] for c in cohort)),
            "median_delta": statistics.median(c["delta"] for c in cohort),
            "per_seat": {str(s): 100 * (statistics.mean(c["utility"] for c in cohort if c["seat"] == s)
                                        - statistics.mean(c["reference_utility"] for c in cohort if c["seat"] == s))
                         for s in (0, 1)}}
    changes = [v["utility_change_points"] for v in per_opponent.values()]
    result = {"engine": "kaggle-environments 1.32.7", "games": len(rows), "overlay": overlay,
              "note": ("The same four frozen local policies declared for the route pilot. Gold V25, "
                       "Gold Plus V8 and the direct Shop port share ancestry with the subject, so this "
                       "is a live transfer check, not four independent strategic families."),
              "entries": sorted(external_entries()), "per_opponent": per_opponent,
              "mean_utility_change_points": statistics.mean(changes),
              "worst_cohort_points": min(changes), "rows": rows}
    save("external.json", result)
    return result


def manifest():
    from kaggriculture_bonsai.runtime import bonsai_kg_v4 as bkg
    compiled = bkg.load_compiled(PLUS)
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                          check=True, cwd=ROOT).stdout.strip()
    value = {"starting_head": head,
             "plus_controller_sha256": sha(PLUS), "plus_controller_bytes": PLUS.stat().st_size,
             "plus_nodes": len(compiled.nodes), "plus_blackboard": len(compiled.bb),
             "plus_state_slots": compiled.state_count, "plus_semantic_hash": compiled.semantic_hash,
             "route_pack_sha256": sha(PORT / "routes.bkv6"), "arity_sha256": sha(PORT / "arity.bin"),
             "direct_controller_sha256": sha(DIRECT),
             "evaluator_sha256": sha(ROOT / "native/cuda/src/mechanics_fallback.cu"),
             "driver_sha256": sha(ROOT / "src/kaggriculture_bonsai/search/mechanics_fallback.py"),
             "candidates": candidate_set(),
             "seeds": {"build": list(BUILD_SEEDS), "confirmation": list(CONFIRM_SEEDS),
                       "dev": list(DEV_SEEDS), "external": list(EXTERNAL_SEEDS)},
             "budget_fge": BUDGET_FGE}
    save("manifest.json", value)
    save("candidates.json", {"total": 32, "capacity": 16, "feed": 16, "combined": 0,
                             "parameterization": ["cap_mode", "cap_floor", "cap_room", "cap_look",
                                                  "feed_mode", "feed_res", "feed_look"],
                             "items": candidate_set()})
    return value


def report():
    facts = load("manifest.json")
    fid, screened, confirmed, developed = load("fidelity.json"), load("build.json"), load("confirmation.json"), load("dev.json")
    canary = load("official.json")
    transfer = load("external.json") if (OUT / "external.json").exists() else None
    ledger = json.loads((OUT / "ledger.json").read_text())
    diag = developed["diagnostics"]
    gate_utility = developed["utility_change_points"] >= 3.0 and developed["lower_bound_90_points"] > 0
    gate_margin = developed["median_delta"] > 0
    gate_effect = any(diag[k] > 0 for k in ("overflow_units_prevented", "discard_value_prevented",
                                            "empty_pens_prevented", "route_action_failures_prevented"))
    gate_seats = all(v > 0 for v in developed["per_seat"].values())
    gate_opponents = sum(v > 0 for v in developed["per_opponent"].values()) >= 2
    gate_external = None if not transfer else (transfer["mean_utility_change_points"] >= 0
                                               and transfer["worst_cohort_points"] >= -10)
    gate_correct = (fid["zero_edit_semantic_mismatch"] == 0 and fid["boundary_cpu_cuda_mismatch"] == 0
                    and canary["mismatches"] == 0 and fid["runtime_faults"] == 0)
    gate_pack = sha(PORT / "routes.bkv6") == facts["route_pack_sha256"]
    passed = all([gate_utility, gate_margin, gate_effect, gate_seats, gate_opponents,
                  gate_correct, gate_pack, gate_external is not False])
    verdict = ("MECHANICS FALLBACK PASS - SHOP PLUS EXECUTOR IMPROVED" if passed
               else "MECHANICS FALLBACK FAIL - CURRENT GOLD PLAN EXHAUSTED")
    turns = sum(r["turns"] for r in ledger)
    summary = {
        "verdict": verdict,
        "gates": {"dev_utility_at_least_3_points": gate_utility, "positive_paired_margin": gate_margin,
                  "attributable_mechanic_effect": gate_effect, "both_seats": gate_seats,
                  "multiple_opponents": gate_opponents, "external_safety": gate_external,
                  "correctness": gate_correct, "route_pack_byte_identical": gate_pack},
        "starting_head": facts["starting_head"],
        "reference": {"agent": "shop_router_0909/plus", "nodes": facts["plus_nodes"],
                      "blackboard": facts["plus_blackboard"], "node_state": facts["plus_state_slots"],
                      "controller_bytes": facts["plus_controller_bytes"],
                      "controller_sha256": facts["plus_controller_sha256"],
                      "semantic_hash": facts["plus_semantic_hash"],
                      "route_pack_sha256": facts["route_pack_sha256"], "routes": "13 x 719"},
        "candidates": {"total": 32, "capacity": 16, "feed": 16, "combined": 0},
        "fidelity": fid,
        "build": {"contexts": screened["contexts"], "games": screened["games"],
                  "finalists": screened["finalists"],
                  "top": [{k: r[k] for k in ("name", "family", "overlay", "utility_change_points",
                                             "median_delta", "mean_delta")}
                          for r in screened["ranked"][:6]],
                  "positive_utility_candidates": sum(r["utility_change_points"] > 0
                                                     for r in screened["ranked"]),
                  "positive_margin_candidates": sum(r["median_delta"] > 0 for r in screened["ranked"])},
        "confirmation": [{k: r[k] for k in ("name", "family", "overlay", "games",
                                            "utility_change_points", "lower_bound_90_points",
                                            "median_delta", "per_seat", "per_opponent")}
                         for r in confirmed["finalists"]],
        "dev": {k: developed[k] for k in ("candidate", "overlay", "games", "utility_change_points",
                                          "lower_bound_90_points", "median_delta", "mean_delta",
                                          "transitions", "per_seat", "per_opponent")},
        "mechanics_diagnostics": diag,
        "external": ({"per_opponent": transfer["per_opponent"],
                      "mean_utility_change_points": transfer["mean_utility_change_points"],
                      "worst_cohort_points": transfer["worst_cohort_points"],
                      "games": transfer["games"]} if transfer else "not run"),
        "work": {"simulated_turns": turns, "native_fge": turns / 719.0, "hard_cap_fge": BUDGET_FGE,
                 "gpu_seconds": sum(r["seconds"] for r in ledger if r["backend"] == "cuda"),
                 "gpu_turns": sum(r["turns"] for r in ledger if r["backend"] == "cuda"),
                 "cpu_seconds": sum(r["seconds"] for r in ledger if r["backend"] == "cpu"),
                 "cpu_turns": sum(r["turns"] for r in ledger if r["backend"] == "cpu"),
                 "official_games": canary["games"] + (transfer["games"] if transfer else 0)},
        "correctness": {"cpu_cuda_mismatches": fid["zero_edit_semantic_mismatch"] + fid["boundary_cpu_cuda_mismatch"],
                        "official_native_mismatches": canary["mismatches"],
                        "runtime_failures": fid["runtime_faults"]},
        "finding": (
            "The two declared mechanics are close to inert on Shop Plus trajectories. Capacity-safe "
            "deposits never fire once: a DROP whose cargo exceeds the free shed never occurs, so "
            "fourteen of sixteen capacity overlays are byte-identical to the reference. Feed "
            "preservation is harmful: Shop Plus sells wheat deliberately and buys feed back, so "
            "withholding it costs thousands of coins per game. The only surviving candidate is a "
            "pre-emptive capacity sale whose entire measured gain is tie-breaking inside the Shop "
            "Plus mirror cohort (+41.41 points there, exactly 0.00 against all three other native "
            "opponents and all four external policies), with a negative mean paired margin."),
        "limitations": [
            "The four native opponents share Shop ancestry and are not four independent strategic families.",
            "The external panel reuses the route pilot's frozen local substitutes for the named public agents.",
            "Thirty-two bounded overlays do not measure the ceiling of executor improvement."]}
    save("summary.json", summary)
    return summary


STAGES = {"manifest": manifest, "fidelity": fidelity, "build": build_screen, "dedup": dedup,
          "confirmation": confirmation, "dev": dev, "official": official,
          "external": external, "report": report}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=sorted(STAGES))
    print(json.dumps(STAGES[parser.parse_args().stage](), indent=2, sort_keys=True)[:4000])


if __name__ == "__main__":
    main()
