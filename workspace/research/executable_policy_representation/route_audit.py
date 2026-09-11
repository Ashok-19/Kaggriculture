"""Read-only schedule statistics, not a policy compiler or a simulator.

Run from the repository root with .venv/bin/python and the installed package.
All counts describe requested actions, not realized production or feasibility.
"""

import hashlib
import json
from collections import Counter
from pathlib import Path

from kaggriculture_bonsai.agents.shop_router_0909.reference_port.pack import ShopRoutePack


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def actors(action):
    return [action["farmer"], *action["hands"]]


def read_routes(directory):
    directory = Path(directory)
    identity = json.loads((directory / "identity.json").read_text())
    pack = ShopRoutePack(directory / "routes.bkv6", identity,
                         (directory / "arity.bin").read_bytes())
    return [[pack.raw_action(route, step) for step in range(719)]
            for route in range(13)]


def analyze(routes):
    base = routes[0]
    days = [base[start:start + 24] for start in range(0, 719, 24)]
    unit_counts = Counter(unit[0] for action in base for unit in actors(action))
    purchases = [[step, order] for step, action in enumerate(base)
                 for order in action["market"]
                 if order and order[0] in ("BUY_LAND", "BUY_ANIMAL")]
    hires = [[day, [hour for hour, action in enumerate(actions)
                    for order in action["market"] if order and order[0] == "HIRE"]]
             for day, actions in enumerate(days)]
    lane_days = []
    for day, actions in enumerate(days[:29]):
        for worker in range(max(map(lambda a: len(actors(a)), actions))):
            lane = [actors(a)[worker] if worker < len(actors(a)) else None
                    for a in actions]
            lane_days.append((day, worker, canonical(lane)))
    lane_hist = Counter(row[2] for row in lane_days)
    motifs = Counter()
    motif_examples = {}
    for day, worker, encoded in lane_days:
        lane = json.loads(encoded)
        for hour in range(21):
            fragment = lane[hour:hour + 4]
            if any(unit is None for unit in fragment):
                continue
            if not any(unit[0] in ("FEED", "CARE", "HARVEST", "WATER",
                                   "COLLECT_FERTILIZER", "PLANT") for unit in fragment):
                continue
            key = canonical(fragment)
            motifs[key] += 1
            motif_examples.setdefault(key, [day, worker, hour])
    lags = {}
    for lag in (24, 72):
        comparisons = [(actors(base[t]), actors(base[t + lag]))
                       for t in range(144, 648 - lag)]
        den = sum(max(len(left), len(right)) for left, right in comparisons)
        same = sum(sum(a == b for a, b in zip(left, right))
                   for left, right in comparisons)
        lags[str(lag)] = {"equal_slots": same, "total_slots": den,
                          "fraction": round(same / den, 6)}
    families = []
    for rid, route in enumerate(routes):
        different = [t for t in range(719) if route[t] != base[t]]
        board_diff = [t for t in range(719) if actors(route[t]) != actors(base[t])]
        plants = Counter(unit[1] for action in route for unit in actors(action)
                         if unit[0] == "PLANT")
        buys = Counter()
        for action in route:
            for order in action["market"]:
                if order and order[0] == "BUY_ANIMAL":
                    buys[order[1]] += order[2]
        families.append({"route": rid, "different_joint_steps": len(different),
                         "first_difference": different[0] if different else None,
                         "different_board_steps": len(board_diff),
                         "first_board_difference": board_diff[0] if board_diff else None,
                         "plants": dict(sorted(plants.items())),
                         "animal_purchase_quantities": dict(sorted(buys.items()))})
    # Full-day equality is deliberately literal: no claim of contract equivalence.
    all_days = [canonical(route[t:t + 24]) for route in routes for t in range(0, 696, 24)]
    all_three_days = [canonical(route[t:t + 72]) for route in routes for t in range(0, 648, 72)]
    return {
        "scope": "13 Shop tapes, zero simulation, raw requests; route 0 is a structural reference, not an asserted best route",
        "route_pack_sha256": hashlib.sha256(Path("workspace/shop_router_0909/port/routes.bkv6").read_bytes()).hexdigest(),
        "route_count": len(routes), "steps_per_route": 719,
        "common_prefix_steps": next(t for t in range(719) if any(r[t] != base[t] for r in routes)),
        "route0_unit_operation_counts": dict(sorted(unit_counts.items())),
        "route0_total_hire_requests": sum(len(hours) for _, hours in hires),
        "route0_hires_by_day": [{"day": day, "hour_counts": dict(sorted(Counter(hours).items()))}
                               for day, hours in hires],
        "route0_market_hour_counts": dict(sorted(Counter(
            step % 24 for step, action in enumerate(base)
            for order in action["market"] if order).items())),
        "route0_sell_hour_counts": dict(sorted(Counter(
            step % 24 for step, action in enumerate(base)
            for order in action["market"] if order and order[0] == "SELL").items())),
        "route0_asset_purchase_orders": purchases,
        "route0_first_day_market": [[t, a["market"]] for t, a in enumerate(base[:24]) if a["market"]],
        "route0_complete_lane_days": len(lane_days),
        "route0_unique_complete_lane_days": len(lane_hist),
        "route0_lane_days_in_repeated_classes": sum(n for n in lane_hist.values() if n > 1),
        "route0_lag_equality_steps144_to647": lags,
        "route0_top_service_four_action_motifs": [
            {"actions": json.loads(k), "count": n,
             "first_day_worker_hour": motif_examples[k]}
            for k, n in motifs.most_common(10)],
        "all_routes_full_day_count": len(all_days),
        "all_routes_unique_full_days": len(set(all_days)),
        "all_routes_three_day_count": len(all_three_days),
        "all_routes_unique_three_days": len(set(all_three_days)),
        "cross_route_differences": families,
    }


if __name__ == "__main__":
    print(json.dumps(analyze(read_routes("workspace/shop_router_0909/port")), indent=2))
