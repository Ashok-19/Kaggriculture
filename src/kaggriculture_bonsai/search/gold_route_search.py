"""Phase 0/1 of the replay-seeded production-route pilot.

Two equal-budget arms propose dependency-closed production jobs against the
frozen Shop Plus executor: Arm R uses a small replay-derived prior, Arm U uses
the same declared support without it. Nothing here promotes an agent or edits
the reference or champion assets.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
import subprocess
import time
from pathlib import Path

from kaggriculture_bonsai.runtime import bonsai_kg_v4 as bkg
from kaggriculture_bonsai.search import gold_route_jobs as J
from kaggriculture_bonsai.search.gold_route_jobs import ANIMALS, CROPS, ITEMS, PORT, ROOT

OUT = ROOT / "workspace/gold_route_search/phase01_replay_seeded"
BUILD = ROOT / "workspace/build/gold_route_search"
ARCHIVE = ROOT / "build/top10_archive_research"
PLUS = ROOT / "workspace/shop_router_0909/plus/controller.bkgc"
DIRECT = PORT / "controller.bkgc"
FIELDS = ("candidate seed seat member checkpoint money0 money1 root simulation controller actions "
          "fault turns route shop0 shop1 purchases hires discarded unsold failed").split()
PRODUCTS = 12
ANIMAL_SPECIES = (9, 10, 11)
CROP_SPECIES = (0, 1, 2, 3, 4)
DISCOVERY_SEEDS = tuple(range(70191001, 70191017))
BUILD_SEEDS = tuple(range(70192001, 70192033))
DEV_SEEDS = tuple(range(70193001, 70193065))
EXTERNAL_SEEDS = tuple(range(70194001, 70194033))
PROPOSAL_SEED = 70190909
BUDGET_FGE = 12000


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(name, value):
    OUT.mkdir(parents=True, exist_ok=True)
    compact = isinstance(value, (list, dict)) and len(json.dumps(value)) > 4000
    (OUT / name).write_text(json.dumps(value, sort_keys=True, indent=None if compact else 2,
                                       separators=(",", ":") if compact else None) + "\n")


def load(name):
    return json.loads((OUT / name).read_text())


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def ledger_add(entry):
    path = OUT / "ledger.json"
    rows = json.loads(path.read_text()) if path.exists() else []
    rows.append(entry)
    save("ledger.json", rows)
    return sum(r["turns"] for r in rows)


def spent_fge():
    path = OUT / "ledger.json"
    rows = json.loads(path.read_text()) if path.exists() else []
    return sum(r["turns"] for r in rows) / 719.0


def evaluate(label, cases, packs, kind, *, backend="cuda", audit=False, case_label=None):
    """Run one native batch. `packs` holds pack stems, "-" for the zero edit."""
    BUILD.mkdir(parents=True, exist_ok=True)
    case_path = BUILD / f"{case_label or label}.cases"
    case_path.write_text("".join(" ".join(map(str, c)) + "\n" for c in cases))
    (BUILD / f"{label}.cand").write_text("\n".join(packs) + "\n")
    command = [str(BUILD / backend), str(PLUS), str(DIRECT), str(PORT / "routes.bkv6"),
               str(PORT / "arity.bin"), str(case_path), str(BUILD / f"{label}.cand"), str(kind), str(int(audit))]
    start = time.monotonic()
    done = subprocess.run(command, capture_output=True, text=True, check=True)
    rows = []
    for line in done.stdout.splitlines():
        values = line.split()
        row = dict(zip(FIELDS, [float(v) if i in (5, 6) else int(v) for i, v in enumerate(values[:len(FIELDS)])]))
        row["produced"] = [int(v) for v in values[len(FIELDS):]]
        row["pack"] = packs[row["candidate"]]
        rows.append(row)
    turns = sum(r["turns"] for r in rows)
    total = ledger_add({"label": label, "backend": backend, "kind": kind, "cases": len(cases),
                        "packs": len(packs), "turns": turns, "seconds": round(time.monotonic() - start, 3),
                        "binary": sha(BUILD / backend)})
    if total / 719.0 > BUDGET_FGE:
        raise RuntimeError("hard native FGE cap exceeded")
    if any(r["fault"] for r in rows):
        raise RuntimeError(f"native fault in {label}")
    return rows


def route_views():
    """Exact per-route job inventories from thirteen annotated reference games."""
    views = {}
    for route in range(13):
        dump = BUILD / f"dump{route}.txt"
        if not dump.exists():
            base = J.Pack(PORT / "routes.bkv6", PORT / "arity.bin")
            broadcast = base.clone()
            for target in range(13):
                for step in range(J.STEPS):
                    broadcast.put(target, step, base.get(route, step))
            stem = BUILD / f"bcast{route}"
            stem.with_suffix(".bkv6").write_bytes(broadcast.to_bytes())
            stem.with_suffix(".arity").write_bytes(broadcast.arity_bytes())
            (BUILD / f"dump{route}.cases").write_text(f"{DISCOVERY_SEEDS[0]} 0 0 144\n")
            (BUILD / f"dump{route}.cand").write_text(str(stem) + "\n")
            command = [str(BUILD / "cpu"), str(PLUS), str(DIRECT), str(PORT / "routes.bkv6"),
                       str(PORT / "arity.bin"), str(BUILD / f"dump{route}.cases"),
                       str(BUILD / f"dump{route}.cand"), "5", "0"]
            done = subprocess.run(command, capture_output=True, text=True, check=True)
            dump.write_text(done.stdout)
            ledger_add({"label": f"dump{route}", "backend": "cpu", "kind": 5, "cases": 1,
                        "packs": 1, "turns": 719, "seconds": 0.0, "binary": sha(BUILD / "cpu")})
        view = J.RouteView(route, dump)
        J.annotate_shares(view)
        views[route] = view
    return views


REPLAY_TAG = "gold-route-pilot-v1"
SPECIES_BY_NAME = {"GOOSE": 9, "COW": 10, "SHEEP": 11,
                   "WHEAT": 0, "CARROT": 1, "TOMATO": 2, "STRAWBERRY": 3, "MELON": 4}


def replay_selection(limit=64):
    """Outcome-blind, current-configuration replay bodies, balanced by day."""
    rows = [json.loads(line) for line in (ARCHIVE / "sample_summaries.jsonl").read_text().splitlines()]
    eligible = [r for r in rows if r["module_version"] == "1.32.7"
                and r["configuration"].get("townCenterSellInterval") == 24
                and r["configuration"].get("turnsPerDay") == 24
                and r["configuration"].get("shedCapacity") == 100
                and r["configuration"].get("startingMoney") == 3000
                and r["steps"] == 720]
    by_date = {}
    for row in eligible:
        key = hashlib.sha256(f"{REPLAY_TAG}:{row['episode_id']}".encode()).hexdigest()
        by_date.setdefault(row["date"], []).append((key, row["episode_id"], row["date"]))
    for value in by_date.values():
        value.sort()
    chosen, index = [], 0
    while len(chosen) < limit and any(len(v) > index for v in by_date.values()):
        for date in sorted(by_date):
            if len(by_date[date]) > index and len(chosen) < limit:
                chosen.append(by_date[date][index])
        index += 1
    return {"eligible": len(eligible), "rejected_old_configuration": len(rows) - len(eligible),
            "selected": [{"episode_id": e, "date": d} for _, e, d in chosen]}


def replay_bodies(selection):
    """One decode pass over the selected bodies only; no full-corpus scan."""
    import pyarrow.parquet as pq
    wanted = {row["episode_id"]: row["date"] for row in selection["selected"]}
    trajectories = []
    for date in sorted(set(wanted.values())):
        shard = ARCHIVE / f"data/replays_{date}.parquet"
        handle = pq.ParquetFile(shard)
        ids = handle.read(columns=["episode_id"]).column(0).to_pylist()
        for index, episode in enumerate(ids):
            if wanted.get(episode) != date:
                continue
            replay = json.loads(handle.read_row_group(index).to_pylist()[0]["replay_json"])
            for seat in (0, 1):
                trajectories.append(trajectory(replay, seat, episode, date))
    return trajectories


def trajectory(replay, seat, episode, date):
    """Realized post-checkpoint lifecycles and a coarse economic profile."""
    steps = replay["steps"]
    lifecycles, live = [], {}
    for index, state in enumerate(steps):
        farm = state[seat]["observation"]["farms"][seat]
        for y, row in enumerate(farm["tiles"]):
            for x, tile in enumerate(row):
                if not isinstance(tile, dict):
                    continue
                kind = tile.get("animal") or tile.get("crop")
                if not kind:
                    continue
                key = (y, x, kind, tile.get("placed_day", tile.get("planted_day")))
                if key not in live:
                    live[key] = {"cell": [x, y], "type": kind, "entry": index,
                                 "day": key[-1], "family": "ANIMAL" if tile.get("animal") else "CROP"}
                    lifecycles.append(live[key])
                record = live[key]
                if tile.get("yield_units", 0) > 0 and "first_yield" not in record:
                    record["first_yield"] = index
                record["last_seen"] = index
    herd, crops = {}, {}
    farm = steps[576][seat]["observation"]["farms"][seat]
    for row in farm["tiles"]:
        for tile in row:
            if isinstance(tile, dict):
                if tile.get("animal"):
                    herd[tile["animal"]] = herd.get(tile["animal"], 0) + 1
                elif tile.get("crop"):
                    crops[tile["crop"]] = crops.get(tile["crop"], 0) + 1
    return {"episode_id": episode, "date": date, "seat": seat, "herd_576": herd, "crops_576": crops,
            "lifecycles": [r for r in lifecycles if r["entry"] >= 144]}


def profile_key(row):
    total = sum(row["herd_576"].values())
    if not total:
        regime = "NO_HERD"
    else:
        best = max(row["herd_576"], key=lambda k: row["herd_576"][k])
        regime = best if row["herd_576"][best] * 10 >= total * 6 else "MIXED"
    crop_total = sum(row["crops_576"].values()) or 1
    best_crop = max(row["crops_576"], key=lambda k: row["crops_576"][k]) if row["crops_576"] else "NONE"
    return f"{regime}/{best_crop}/{'HIGH' if crop_total >= 12 else 'LOW'}"


def mine_replay_jobs(trajectories, max_profiles=8, max_records=24):
    """At most eight founder profiles and twenty-four auditable job records."""
    groups = {}
    for row in trajectories:
        groups.setdefault(profile_key(row), []).append(row)
    ordered = sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0]))[:max_profiles]
    pool, rejections = [], []
    for profile, rows in ordered:
        for row in sorted(rows, key=lambda r: (r["episode_id"], r["seat"])):
            for life in sorted(row["lifecycles"], key=lambda l: (l["entry"], l["cell"])):
                species = SPECIES_BY_NAME.get(life["type"])
                if species is None:
                    rejections.append({"reason": "unknown item", "type": life["type"]})
                    continue
                if "first_yield" not in life:
                    rejections.append({"reason": "no observed yield transition",
                                       "episode_id": row["episode_id"], "type": life["type"]})
                    continue
                pool.append({"profile": profile, "episode_id": row["episode_id"], "seat": row["seat"],
                             "family": life["family"], "species": species, "item": life["type"],
                             "entry_step": life["entry"], "entry_day": life["day"],
                             "first_yield_step": life["first_yield"],
                             "yield_delay_steps": life["first_yield"] - life["entry"],
                             "site": life["cell"], "last_seen_step": life["last_seen"],
                             "product": ITEMS[ANIMALS[species]["product"]] if life["family"] == "ANIMAL" else life["type"]})
    # Balanced, outcome-blind selection: new contract shapes first, then coverage.
    def key(record):
        return (record["family"], record["species"], record["entry_day"] // 3,
                min(3, record["yield_delay_steps"] // 48))
    records, seen_keys = [], set()
    quota = {"ANIMAL": max_records // 2, "CROP": max_records - max_records // 2}
    for wave in (0, 1):
        for family in ("ANIMAL", "CROP"):
            taken = sum(r["family"] == family for r in records)
            for record in pool:
                if record["family"] != family or taken >= quota[family] or len(records) >= max_records:
                    continue
                if wave == 0 and key(record) in seen_keys:
                    continue
                if record in records:
                    continue
                seen_keys.add(key(record))
                records.append(record)
                taken += 1
    templates, seen = [], set()
    for record in records:
        shape = key(record)
        if shape in seen:
            continue
        seen.add(shape)
        templates.append({"template": f"{record['family']}_{record['item']}_D{record['entry_day']}",
                          "family": record["family"], "species": record["species"],
                          "entry_day": record["entry_day"], "yield_delay_steps": record["yield_delay_steps"],
                          "profile": record["profile"], "source_episode": record["episode_id"],
                          "source_seat": record["seat"]})
    return {"profiles": [p for p, _ in ordered], "records": records, "templates": templates,
            "rejections": rejections[:24]}


SHIFTS = (-24, 0, 24)


def support(views):
    """The declared feasible job support. Both arms sample from exactly this."""
    canonical = views[0]
    animal = [(s, i) for f, s, i in canonical.support("ANIMAL")]
    crop = [(s, i) for f, s, i in canonical.support("CROP")]
    crop = [key for key in crop if key[1] < 12]  # bounded, auditable planting ordinals
    return {"ANIMAL": animal, "CROP": crop}


def replay_prior(mined):
    """Small non-identifying prior: which species and timings replays realize."""
    weights = {"ANIMAL": {}, "CROP": {}}
    days = {"ANIMAL": [], "CROP": []}
    for record in mined["records"]:
        weights[record["family"]][record["species"]] = weights[record["family"]].get(record["species"], 0) + 1
        days[record["family"]].append(record["entry_day"])
    return {"species_weight": {f: {str(k): v for k, v in w.items()} for f, w in weights.items()},
            "median_entry_day": {f: (statistics.median(v) if v else None) for f, v in days.items()},
            "excluded_features": ["episode_id", "team", "seed", "opponent_private_state", "future_shops"]}


def choose(rng, options, weights=None):
    if not weights:
        return rng.choice(options)
    total = sum(weights)
    if total <= 0:
        return rng.choice(options)
    pick = rng.random() * total
    for option, weight in zip(options, weights):
        pick -= weight
        if pick <= 0:
            return option
    return options[-1]


def sample_operation(rng, family, declared, prior, informed, views):
    species, ordinal = rng.choice(declared[family])
    targets = [t for t in (ANIMAL_SPECIES if family == "ANIMAL" else CROP_SPECIES) if t != species]
    if informed:
        weight = prior["species_weight"][family]
        targets_weight = [weight.get(str(t), 0) + 0.25 for t in targets]
        target = choose(rng, targets, targets_weight)
        job = views[0].job(family, species, ordinal)
        anchor = job["planted_day"]
        median = prior["median_entry_day"][family]
        preference = [1.0 + (1.5 if median is not None and
                             abs(anchor + s // 24 - median) < abs(anchor - median) else 0.0) for s in SHIFTS]
        shift = choose(rng, SHIFTS, preference)
    else:
        target = rng.choice(targets)
        shift = rng.choice(SHIFTS)
    if target == species:
        return None
    return {"family": family, "species": species, "ordinal": ordinal, "target": target, "shift": shift}


def proposal_schedule(prior, declared, views):
    """Frozen, equal-complexity schedules; only the proposal prior differs."""
    plan = []
    for arm in ("R", "U"):
        rng = random.Random(f"{PROPOSAL_SEED}:{arm}")
        rows = []
        # 90 single-operation and 38 two-operation slots in both arms.
        classes = ([("template", "ANIMAL", True)] * 32 + [("template", "CROP", True)] * 32
                   + [("recombination", "ANIMAL", True)] * 19 + [("recombination", "CROP", True)] * 19
                   + [("uninformed", "ANIMAL", False)] * 13 + [("uninformed", "CROP", False)] * 13)
        if arm == "U":
            classes = [(name, family, False) for name, family, _ in classes]
        for index, (name, family, informed) in enumerate(classes):
            rows.append({"arm": arm, "slot": index, "class": name, "family": family, "informed": informed})
        plan.append({"arm": arm, "slots": rows, "rng": f"{PROPOSAL_SEED}:{arm}"})
        del rng
    return plan


def compile_candidate(operations, views, base):
    pack = base.clone()
    closure, applied = [], 0
    for route in range(13):
        view = views[route]
        editor = J.Editor(pack, route, view)
        for operation in operations:
            job = view.job(operation["family"], operation["species"], operation["ordinal"])
            if job is None:
                continue
            if operation["family"] == "ANIMAL":
                J.animal_convert(editor, job, operation["target"], operation["shift"])
            else:
                J.crop_convert(editor, job, operation["target"], operation["shift"])
            applied += 1
        closure.extend(editor.closure)
    if not applied:
        raise J.InvalidCandidate("no inherited job matched the proposal")
    return pack, closure, applied


def run_packs(label, case_label, cases, packs, kind, chunk=32):
    rows = []
    for start in range(0, len(packs), chunk):
        block = packs[start:start + chunk]
        part = evaluate(f"{label}_{start}", cases, block, kind, case_label=case_label)
        for row in part:
            row["candidate"] += start
        rows.extend(part)
    return rows


def utility(row):
    sign = 1 if row["seat"] == 0 else -1
    margin = sign * (row["money0"] - row["money1"])
    return margin, (1.0 if margin > 0 else 0.5 if margin == 0 else 0.0)


def paired(rows, reference):
    key = lambda r: (r["seed"], r["seat"], r["member"])
    base = {key(r): utility(r) for r in reference}
    out = []
    for row in rows:
        margin, util = utility(row)
        old_margin, old_util = base[key(row)]
        outcome = lambda v: "W" if v > 0 else "L" if v < 0 else "T"
        out.append({"candidate": row["candidate"], "pack": row["pack"], "seed": row["seed"], "seat": row["seat"],
                    "member": row["member"], "margin": margin, "reference_margin": old_margin,
                    "delta": margin - old_margin, "utility": util, "reference_utility": old_util,
                    "transition": outcome(old_margin) + "->" + outcome(margin),
                    "produced": row["produced"], "purchases": row["purchases"], "hires": row["hires"],
                    "unsold": row["unsold"], "discarded": row["discarded"], "failed": row["failed"]})
    return out


def cohort_utility(rows, field="utility"):
    """Equal weight per (opponent, seat) cohort, never per raw game."""
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


def summarise(rows):
    return {"games": len(rows), "utility": cohort_utility(rows),
            "reference_utility": cohort_utility(rows, "reference_utility"),
            "utility_change_points": 100 * (cohort_utility(rows) - cohort_utility(rows, "reference_utility")),
            "median_delta": statistics.median(r["delta"] for r in rows),
            "mean_delta": statistics.mean(r["delta"] for r in rows),
            "transitions": transitions(rows),
            "produced_total": [sum(r["produced"][k] for r in rows) for k in range(PRODUCTS)],
            "reference_free": True}


def panel_cases(seeds, checkpoint=144, opponents=(0, 1, 2, 3), assign=False):
    if assign:
        return [(seed, seat, opponents[index % len(opponents)], checkpoint)
                for index, seed in enumerate(seeds) for seat in (0, 1)]
    return [(seed, seat, member, checkpoint)
            for seed in seeds for seat in (0, 1) for member in opponents]


def phase0():
    """Pin identities, mine the bounded replay library and prove zero-edit fidelity."""
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    controller = bkg.load_compiled(PLUS)
    footprint = (len(controller.nodes), len(controller.bb), controller.state_count)
    if footprint != (85, 2, 0):
        raise RuntimeError(f"reference footprint drift {footprint}")
    selection = replay_selection()
    mined = mine_replay_jobs(replay_bodies(selection))
    animal = sum(t["family"] == "ANIMAL" for t in mined["templates"])
    crop = sum(t["family"] == "CROP" for t in mined["templates"])
    gate = len(mined["templates"]) >= 8 and animal >= 4 and crop >= 4
    save("replay_jobs.json", {"selection": selection, "profiles": mined["profiles"],
                              "records": mined["records"], "templates": mined["templates"],
                              "rejections": mined["rejections"], "usable_templates": len(mined["templates"]),
                              "animal_templates": animal, "crop_templates": crop, "gate": gate})
    views = route_views()
    declared = support(views)
    base = J.Pack(PORT / "routes.bkv6", PORT / "arity.bin")
    zero = base.clone()
    stem = BUILD / "zero_edit"
    stem.with_suffix(".bkv6").write_bytes(zero.to_bytes())
    stem.with_suffix(".arity").write_bytes(zero.arity_bytes())
    identical = (zero.to_bytes() == (PORT / "routes.bkv6").read_bytes()
                 and zero.arity_bytes() == (PORT / "arity.bin").read_bytes())
    cases = panel_cases(DISCOVERY_SEEDS[:4], assign=True)
    cuda = evaluate("fidelity_cuda", cases, ["-", str(stem)], 0, audit=True)
    cpu = evaluate("fidelity_cpu", cases, ["-", str(stem)], 0, backend="cpu", audit=True, case_label="fidelity_cuda")
    mismatch = 0
    for a, b in zip(cuda, cpu):
        if any(a[k] != b[k] for k in ("money0", "money1", "simulation", "controller", "actions", "root")):
            mismatch += 1
    reference = [r for r in cuda if r["pack"] == "-"]
    edited = [r for r in cuda if r["pack"] != "-"]
    zero_mismatch = sum(any(a[k] != b[k] for k in ("money0", "money1", "simulation", "controller", "actions"))
                        for a, b in zip(reference, edited))
    lifecycles = replay_lifecycle_check(load("replay_jobs.json"))
    fidelity = {"head": head, "reference_footprint": {"nodes": 85, "blackboard": 2, "node_state": 0},
                "zero_edit_bytes_identical": identical, "zero_edit_semantic_mismatch": zero_mismatch,
                "cpu_cuda_mismatch": mismatch, "checked_games": len(cases),
                "declared_support": {k: len(v) for k, v in declared.items()},
                "route_job_counts": {str(r): len(views[r].jobs) for r in views},
                "replay_lifecycle_check": lifecycles,
                "gate": bool(identical and zero_mismatch == 0 and mismatch == 0 and gate
                             and lifecycles["mismatches"] == 0)}
    save("fidelity.json", fidelity)
    save("manifest.json", {
        "experiment": "gold_route_search/phase01_replay_seeded", "team": "ReCurSiON",
        "starting_head": head, "engine": "audited native CPU/CUDA sim (kaggle-environments 1.32.7 semantics)",
        "reference_agent": "shop_router_0909/plus",
        "plus_controller_sha256": sha(PLUS), "plus_controller_semantic": controller.semantic_hash,
        "direct_controller_sha256": sha(DIRECT),
        "route_pack_sha256": sha(PORT / "routes.bkv6"), "arity_sha256": sha(PORT / "arity.bin"),
        "native_source_sha256": sha(ROOT / "native/cuda/src/gold_route_search.cu"),
        "job_module_sha256": sha(ROOT / "src/kaggriculture_bonsai/search/gold_route_jobs.py"),
        "driver_sha256": sha(ROOT / "src/kaggriculture_bonsai/search/gold_route_search.py"),
        "binaries": {name: sha(BUILD / name) for name in ("cpu", "cuda")},
        "replay_archive": {"dataset": "ashok205/kaggriculture-top10-replay-archive", "version": 19,
                           "bodies_inspected": len(selection["selected"]),
                           "index": "workspace/research/top10_replay_archive/evidence.json"},
        "proposal_rng_seed": PROPOSAL_SEED,
        "seeds": {"discovery": list(DISCOVERY_SEEDS), "build": list(BUILD_SEEDS),
                  "dev": list(DEV_SEEDS), "external": list(EXTERNAL_SEEDS)},
        "opponents": {"0": "shop_router_0909/plus", "1": "shop_router_0909/reference_port",
                      "2": "shop plus executor on broadcast route 3", "3": "shop plus executor on broadcast route 12"},
        "candidate_generation": {"proposals_per_arm": 128, "max_compilations_per_arm": 512,
                                 "families": ["ANIMAL", "CROP"], "shifts": list(SHIFTS),
                                 "declared_support": {k: [list(v) for v in vs] for k, vs in declared.items()},
                                 "structure": "90 single-operation and 38 two-operation slots per arm"},
        "training_exclusions": ["episode_id", "team_name", "replay_seed", "opponent_private_state",
                                "future_observations", "future_shop_sequence"],
        "budget_fge": BUDGET_FGE})
    return fidelity


def replay_lifecycle_check(document):
    """Do the mined replay lifecycles obey current engine biology?"""
    mismatches, checked = [], 0
    for record in document["records"][:12]:
        checked += 1
        delay_days = record["yield_delay_steps"] / 24.0
        if record["family"] == "ANIMAL":
            expected = ANIMALS[record["species"]]["first"]
        else:
            expected = CROPS[record["species"]]["first"]
        if delay_days > expected + 4 or delay_days < -1:
            mismatches.append({"episode_id": record["episode_id"], "item": record["item"],
                               "observed_delay_days": round(delay_days, 2), "engine_first_yield_day": expected})
    return {"checked": checked, "mismatches": len(mismatches), "detail": mismatches}


def propose():
    """Freeze both proposal schedules and compile them before any outcome."""
    document = load("replay_jobs.json")
    prior = replay_prior({"records": document["records"]})
    views = route_views()
    declared = support(views)
    base = J.Pack(PORT / "routes.bkv6", PORT / "arity.bin")
    plan = proposal_schedule(prior, declared, views)
    arms = {}
    for entry in plan:
        arm = entry["arm"]
        rng = random.Random(entry["rng"])
        accepted, digests, attempts, rejected = [], set(), 0, {}
        for slot in entry["slots"]:
            while attempts < 512 and len(accepted) <= slot["slot"]:
                attempts += 1
                operations = []
                first = sample_operation(rng, slot["family"], declared, prior, slot["informed"], views)
                if first is None:
                    rejected["null"] = rejected.get("null", 0) + 1
                    continue
                operations.append(first)
                if slot["class"] == "recombination":
                    other = "CROP" if slot["family"] == "ANIMAL" else "ANIMAL"
                    second = sample_operation(rng, other, declared, prior, slot["informed"], views)
                    if second is not None:
                        operations.append(second)
                try:
                    pack, closure, applied = compile_candidate(operations, views, base)
                except J.InvalidCandidate as error:
                    rejected[str(error)] = rejected.get(str(error), 0) + 1
                    continue
                digest = hashlib.sha256(pack.to_bytes() + pack.arity_bytes()).hexdigest()
                if digest in digests:
                    rejected["duplicate closure"] = rejected.get("duplicate closure", 0) + 1
                    continue
                digests.add(digest)
                stem = BUILD / f"cand_{arm}_{len(accepted):03d}"
                stem.with_suffix(".bkv6").write_bytes(pack.to_bytes())
                stem.with_suffix(".arity").write_bytes(pack.arity_bytes())
                accepted.append({"arm": arm, "index": len(accepted), "class": slot["class"],
                                 "family": slot["family"], "informed": slot["informed"],
                                 "operations": operations, "routes_edited": applied,
                                 "edits": len(closure), "closure_sha256": digest, "pack": str(stem)})
            if len(accepted) >= 128:
                break
        arms[arm] = {"proposals": accepted, "attempts": attempts, "rejected": rejected,
                     "valid": len(accepted)}
    save("proposal_schedule.json", {"prior": prior, "arms": arms,
                                    "note": "frozen before any candidate evaluation"})
    return arms


def arm_packs(arms, arm):
    return [p["pack"] for p in arms[arm]["proposals"]]


def discovery():
    arms = load("proposal_schedule.json")["arms"]
    cases = panel_cases(DISCOVERY_SEEDS, assign=True)
    reference = evaluate("roots", cases, ["-"], 0, case_label="roots")
    result = {"cases": cases, "roots": len(cases)}
    for arm in ("R", "U"):
        packs = arm_packs(arms, arm)
        rows = run_packs(f"discovery_{arm}", "roots", cases, packs, 3)
        rows = paired(rows, reference)
        ranked = []
        for index in range(len(packs)):
            block = [r for r in rows if r["pack"] == packs[index]]
            ranked.append({"index": index, "pack": packs[index], **summarise(block)})
        ranked.sort(key=lambda r: (-r["utility"], -r["median_delta"]))
        result[arm] = {"candidates": ranked, "continuations": len(rows),
                       "finalists": [r["pack"] for r in ranked[:2]]}
    save("discovery.json", result)
    return result


def confirmation():
    plan = load("discovery.json")
    cases = panel_cases(BUILD_SEEDS)
    reference = evaluate("confirm_roots", cases, ["-"], 0, case_label="confirm")
    result = {}
    for arm in ("R", "U"):
        packs = plan[arm]["finalists"]
        rows = paired(run_packs(f"confirm_{arm}", "confirm", cases, packs, 0), reference)
        ranked = [{"pack": pack, **summarise([r for r in rows if r["pack"] == pack])} for pack in packs]
        ranked.sort(key=lambda r: (-r["utility"], -r["median_delta"]))
        result[arm] = {"finalists": ranked, "winner": ranked[0]["pack"]}
    save("confirmation.json", result)
    return result


def dev():
    winners = load("confirmation.json")
    cases = panel_cases(DEV_SEEDS)
    reference = evaluate("dev_roots", cases, ["-"], 0, case_label="dev")
    result = {}
    rows_by_arm = {}
    for arm in ("R", "U"):
        pack = winners[arm]["winner"]
        rows = paired(run_packs(f"dev_{arm}", "dev", cases, [pack], 0), reference)
        rows_by_arm[arm] = rows
        result[arm] = {"pack": pack, **summarise(rows),
                       "lower_bound_90": bootstrap_lower(rows),
                       "per_opponent": {str(m): summarise([r for r in rows if r["member"] == m])
                                        for m in (0, 1, 2, 3)},
                       "per_seat": {str(s): summarise([r for r in rows if r["seat"] == s]) for s in (0, 1)}}
    difference = result["R"]["utility_change_points"] - result["U"]["utility_change_points"]
    result["arm_difference_points"] = difference
    result["arm_difference_lower_bound_90"] = paired_arm_bootstrap(rows_by_arm["R"], rows_by_arm["U"])
    save("dev.json", result)
    return result


def paired_arm_bootstrap(rows_r, rows_u, reps=2000, level=0.10, seed=20260911):
    seeds = sorted({r["seed"] for r in rows_r})
    group_r = {s: [r for r in rows_r if r["seed"] == s] for s in seeds}
    group_u = {s: [r for r in rows_u if r["seed"] == s] for s in seeds}
    rng = random.Random(seed)
    samples = []
    for _ in range(reps):
        drawn = [rng.choice(seeds) for _ in seeds]
        block_r = [r for s in drawn for r in group_r[s]]
        block_u = [r for s in drawn for r in group_u[s]]
        samples.append((cohort_utility(block_r) - cohort_utility(block_r, "reference_utility"))
                       - (cohort_utility(block_u) - cohort_utility(block_u, "reference_utility")))
    samples.sort()
    return samples[int(level * reps)]


def python_plus(pack_stem=None):
    """The frozen Plus executor, optionally reading a candidate route pack."""
    from kaggriculture_bonsai.agents.shop_router_0909.plus.agent import ACTIVE, ShopRouter0909Plus
    from kaggriculture_bonsai.agents.shop_router_0909.reference_port.pack import ShopRoutePack, Tape
    agent = ShopRouter0909Plus(json.loads(ACTIVE.read_text())["candidate"])
    if pack_stem:
        path = Path(str(pack_stem) + ".bkv6")
        identity = {"route_pack": {"pack_sha256": sha(path), "route_ids": list(range(13))}}
        arity = Path(str(pack_stem) + ".arity").read_bytes()
        pack = ShopRoutePack(path, identity, arity)
        agent.tapes = [Tape(pack, route) for route in range(13)]
    return agent


def external_entries():
    from kaggriculture_bonsai.agents.gold_v25.agent import GoldV25Agent
    from kaggriculture_bonsai.agents.gold_v25_plus.agent import GoldV25Plus
    from kaggriculture_bonsai.agents.shop_router_0909.reference_port.agent import ShopRouter0909Agent
    from kaggriculture_bonsai.runtime.champion import BKG4ChampionAgent
    return {"GOLD_PLUS_V8": lambda: GoldV25Plus(144, 647, 1025),
            "GOLD_V25": GoldV25Agent,
            "BKG4_CHAMPION": BKG4ChampionAgent,
            "SHOP_DIRECT_PORT": ShopRouter0909Agent}


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
    """Small paired official-engine canary against the audited native result."""
    from kaggriculture_bonsai.agents.shop_router_0909.reference_port.validate import engine
    make = engine()
    winner = load("confirmation.json")["R"]["winner"]
    cases = [(seed, seat, 0, 144) for seed in DISCOVERY_SEEDS[:4] for seat in (0, 1)]
    native = evaluate("official_ref", cases, ["-", winner], 0, backend="cpu")
    rows, mismatches = [], 0
    for row in native:
        rewards = official_game(make, python_plus(None if row["pack"] == "-" else winner),
                                python_plus(None), row["seed"], row["seat"])
        native_rewards = [row["money0"], row["money1"]]
        agree = [round(v, 6) for v in native_rewards] == [round(v, 6) for v in rewards]
        mismatches += 0 if agree else 1
        rows.append({"seed": row["seed"], "seat": row["seat"],
                     "arm": "reference" if row["pack"] == "-" else "candidate",
                     "official_rewards": rewards, "native_rewards": native_rewards, "agree": agree})
    result = {"engine": "kaggle-environments 1.32.7", "games": len(rows),
              "mismatches": mismatches, "rows": rows}
    save("official.json", result)
    return result


def _external_job(task):
    from kaggriculture_bonsai.agents.shop_router_0909.reference_port.validate import engine
    make = engine()
    subject, stem, opponent, seed, seat = task
    rewards = official_game(make, python_plus(stem), external_entries()[opponent](), seed, seat)
    return {"subject": subject, "opponent": opponent, "seed": seed, "seat": seat, "rewards": rewards}


def external(workers=8):
    """Declared frozen transfer panel. Substitutions were fixed before results."""
    import multiprocessing
    confirmed = load("confirmation.json")
    packs = {"SHOP_PLUS_REFERENCE": None, "ARM_R_WINNER": confirmed["R"]["winner"],
             "ARM_U_WINNER": confirmed["U"]["winner"]}
    digest = {name: (sha(Path(str(stem) + ".bkv6")) if stem else sha(PORT / "routes.bkv6"))
              for name, stem in packs.items()}
    unique, alias = {}, {}
    for name, value in digest.items():
        alias[name] = unique.setdefault(value, name)
    tasks = [(alias[name], packs[alias[name]], opponent, seed, seat)
             for name in packs if alias[name] == name
             for opponent in sorted(external_entries())
             for seed in EXTERNAL_SEEDS for seat in (0, 1)]
    with multiprocessing.Pool(workers) as pool:
        rows = pool.map(_external_job, tasks)
    index = {(r["subject"], r["opponent"], r["seed"], r["seat"]): r for r in rows}
    summary = {}
    for name in ("ARM_R_WINNER", "ARM_U_WINNER"):
        source = alias[name]
        per_opponent = {}
        for opponent in sorted(external_entries()):
            cohort = []
            for seed in EXTERNAL_SEEDS:
                for seat in (0, 1):
                    candidate = index[(source, opponent, seed, seat)]["rewards"]
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
        summary[name] = {"identical_to": source, "per_opponent": per_opponent,
                         "mean_paired_utility_change_points": statistics.mean(changes),
                         "worst_cohort_regression_points": min(changes)}
    result = {"engine": "kaggle-environments 1.32.7", "games": len(rows),
              "planned_games": 768, "unique_subject_packs": len(unique),
              "substitution": ("Named external policies other than Gold Plus V8 were unavailable locally; "
                               "four frozen structurally comparable local policies were declared before results. "
                               "This is a live transfer check, not four independent external families."),
              "entries": sorted(external_entries()), "summary": summary, "rows": rows}
    save("external.json", result)
    return result


def sales_only_pack(operations, views, base, stem):
    """The same output obligations without the changed production commitment."""
    pack = base.clone()
    for route in range(13):
        view = views[route]
        editor = J.Editor(pack, route, view)
        for operation in operations:
            job = view.job(operation["family"], operation["species"], operation["ordinal"])
            if job is None:
                continue
            if operation["family"] == "ANIMAL":
                old, new = ANIMALS[operation["species"]]["product"], ANIMALS[operation["target"]]["product"]
            else:
                old, new = operation["species"], operation["target"]
            start = job["harvests"][0] if job["harvests"] else job.get("place", job.get("plant"))["step"]
            editor.remap_sales(start, old, new, job["share"])
    Path(str(stem) + ".bkv6").write_bytes(pack.to_bytes())
    Path(str(stem) + ".arity").write_bytes(pack.arity_bytes())
    return str(stem)


def ablation():
    """Whole-job ablation: production commitment versus terminal sales alone."""
    schedule = load("proposal_schedule.json")
    winners = load("confirmation.json")
    views = route_views()
    base = J.Pack(PORT / "routes.bkv6", PORT / "arity.bin")
    result = {}
    for arm in ("R", "U"):
        pack = winners[arm]["winner"]
        proposal = next(p for p in schedule["arms"][arm]["proposals"] if p["pack"] == pack)
        stem = sales_only_pack(proposal["operations"], views, base, BUILD / f"ablate_{arm}")
        cases = panel_cases(DEV_SEEDS[:16])
        reference = evaluate("ablate_ref", cases, ["-"], 0, case_label="ablate")
        rows = paired(run_packs(f"ablate_{arm}", "ablate", cases, [pack, stem], 0), reference)
        full = [r for r in rows if r["pack"] == pack]
        sales = [r for r in rows if r["pack"] == stem]
        result[arm] = {"operations": proposal["operations"], "pack": pack,
                       "full_candidate": summarise(full), "sales_only": summarise(sales),
                       "production_effect_points": (summarise(full)["utility_change_points"]
                                                    - summarise(sales)["utility_change_points"]),
                       "median_paired_margin_full": statistics.median(r["delta"] for r in full),
                       "seat_support": {str(s): statistics.median(r["delta"] for r in full if r["seat"] == s)
                                        for s in (0, 1)},
                       "opponent_support": {str(m): statistics.median(r["delta"] for r in full if r["member"] == m)
                                            for m in (0, 1, 2, 3)}}
    save("ablation.json", result)
    return result


def candidate_parity():
    """CPU/CUDA agreement on the new route semantics, not only the zero edit."""
    winner = load("confirmation.json")["R"]["winner"]
    cases = panel_cases(DISCOVERY_SEEDS[:4], assign=True)
    cuda = evaluate("parity_cuda", cases, [winner], 0, audit=True)
    cpu = evaluate("parity_cpu", cases, [winner], 0, backend="cpu", audit=True, case_label="parity_cuda")
    keys = ("money0", "money1", "simulation", "controller", "actions", "root")
    mismatch = sum(any(a[k] != b[k] for k in keys) for a, b in zip(cuda, cpu))
    result = {"games": len(cases), "cpu_cuda_mismatch": mismatch, "pack": winner}
    save("parity.json", result)
    return result


def report():
    """Predeclared gates, then exactly one verdict. No threshold is moved."""
    manifest, fidelity = load("manifest.json"), load("fidelity.json")
    replay, schedule = load("replay_jobs.json"), load("proposal_schedule.json")
    found, confirmed, developed = load("discovery.json"), load("confirmation.json"), load("dev.json")
    ablate, parity, canary = load("ablation.json"), load("parity.json"), load("official.json")
    transfer = load("external.json") if (OUT / "external.json").exists() else None
    ledger = json.loads((OUT / "ledger.json").read_text())
    absolute = (developed["R"]["utility_change_points"] >= 5.0
                and developed["R"]["lower_bound_90"] > 0)
    prior = developed["arm_difference_points"] >= 3.0
    economic = (ablate["R"]["median_paired_margin_full"] >= 1000
                and all(v >= 0 for v in ablate["R"]["seat_support"].values())
                and sum(v >= 0 for v in ablate["R"]["opponent_support"].values()) >= 2)
    external_gate = None
    if transfer:
        best = transfer["summary"]["ARM_R_WINNER"]
        external_gate = (best["mean_paired_utility_change_points"] >= 0
                         and best["worst_cohort_regression_points"] >= -10)
    correctness = (fidelity["zero_edit_semantic_mismatch"] == 0 and fidelity["cpu_cuda_mismatch"] == 0
                   and parity["cpu_cuda_mismatch"] == 0 and canary["mismatches"] == 0)
    any_absolute = any(developed[arm]["utility_change_points"] >= 5.0
                       and developed[arm]["lower_bound_90"] > 0 for arm in ("R", "U"))
    if absolute and prior and economic and correctness and external_gate is not False:
        verdict = "PHASE 1 PASS - REPLAY-SEEDED PRODUCTION ROUTE SEARCH VALIDATED, PROCEED TO SMALL ROUTE LEAGUE"
    elif any_absolute and economic and correctness:
        verdict = "PHASE 1 PARTIAL PASS - PRODUCTION ROUTE SEARCH VALIDATED, REPLAY PRIOR NOT VALIDATED"
    else:
        verdict = "PHASE 1 FAIL - PRODUCTION ROUTE PATH NOT VALIDATED"
    turns = sum(row["turns"] for row in ledger)
    summary = {
        "verdict": verdict,
        "gates": {"absolute_production": absolute, "replay_prior_increment": prior,
                  "material_economic_change": economic, "external_safety": external_gate,
                  "correctness": correctness},
        "starting_head": manifest["starting_head"],
        "reference": {"agent": "shop_router_0909/plus", "nodes": 85, "blackboard": 2, "node_state": 0,
                      "controller_sha256": manifest["plus_controller_sha256"],
                      "controller_bytes": 3224, "routes": "13 x 719",
                      "route_pack_sha256": manifest["route_pack_sha256"]},
        "replay": {"archive_version": 19, "bodies_inspected": len(replay["selection"]["selected"]),
                   "founder_profiles": len(replay["profiles"]), "job_records": len(replay["records"]),
                   "usable_templates": replay["usable_templates"],
                   "animal_templates": replay["animal_templates"], "crop_templates": replay["crop_templates"]},
        "phase0": {"zero_edit_semantic_mismatch": fidelity["zero_edit_semantic_mismatch"],
                   "zero_edit_bytes_identical": fidelity["zero_edit_bytes_identical"],
                   "cpu_cuda_mismatch": fidelity["cpu_cuda_mismatch"],
                   "replay_lifecycle_mismatches": fidelity["replay_lifecycle_check"]["mismatches"],
                   "official_games": canary["games"], "official_mismatches": canary["mismatches"]},
        "arms": {arm: {"valid_proposals": schedule["arms"][arm]["valid"],
                       "compilation_attempts": schedule["arms"][arm]["attempts"],
                       "rejections": schedule["arms"][arm]["rejected"],
                       "discovery_continuations": found[arm]["continuations"],
                       "build_finalists": [f["pack"].split("/")[-1] for f in confirmed[arm]["finalists"]],
                       "dev_winner": developed[arm]["pack"].split("/")[-1],
                       "dev_utility_change_points": developed[arm]["utility_change_points"],
                       "dev_lower_bound_90_points": 100 * developed[arm]["lower_bound_90"],
                       "dev_median_paired_margin": developed[arm]["median_delta"],
                       "dev_transitions": developed[arm]["transitions"],
                       "dev_per_opponent_points": {k: v["utility_change_points"]
                                                   for k, v in developed[arm]["per_opponent"].items()},
                       "dev_per_seat_points": {k: v["utility_change_points"]
                                               for k, v in developed[arm]["per_seat"].items()}}
                 for arm in ("R", "U")},
        "arm_comparison": {"utility_difference_points": developed["arm_difference_points"],
                           "difference_lower_bound_90_points": 100 * developed["arm_difference_lower_bound_90"],
                           "winner_closure_identical": (confirmed["R"]["winner"].split("/")[-1]
                                                        != confirmed["U"]["winner"].split("/")[-1]
                                                        and sha(Path(confirmed["R"]["winner"] + ".bkv6"))
                                                        == sha(Path(confirmed["U"]["winner"] + ".bkv6")))},
        "ablation": {arm: {k: v for k, v in ablate[arm].items() if k not in ("full_candidate", "sales_only")}
                     for arm in ("R", "U")},
        "external": transfer["summary"] if transfer else "not run",
        "work": {"simulated_turns": turns, "native_fge": turns / 719.0, "hard_cap_fge": BUDGET_FGE,
                 "gpu_seconds": sum(r["seconds"] for r in ledger if r["backend"] == "cuda"),
                 "cpu_seconds": sum(r["seconds"] for r in ledger if r["backend"] == "cpu"),
                 "official_games": canary["games"] + (transfer["games"] if transfer else 0)},
        "correctness": {"cpu_cuda_mismatches": fidelity["cpu_cuda_mismatch"] + parity["cpu_cuda_mismatch"],
                        "official_native_mismatches": canary["mismatches"], "runtime_faults": 0,
                        "hidden_information_dependency": False},
        "limitations": [
            "Four native discovery opponents share Shop ancestry; they are not four independent strategic families.",
            "The external panel substitutes frozen local policies for the unavailable named public agents.",
            "One fixed proposal generation does not measure the ceiling of production-route search."]}
    save("summary.json", summary)
    return summary
