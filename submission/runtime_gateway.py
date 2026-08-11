"""Kaggriculture submission gateway.

The gateway preserves the proven v27 safety controller, exposes bounded
observation features to the native C++ core, and applies only interventions
that have passed explicit promotion gates. No opponent identity fields are used.
"""
from __future__ import annotations

import copy
import ctypes
import json
import math
from pathlib import Path

PRODUCTS = ("WHEAT","CARROT","TOMATO","STRAWBERRY","MELON","EGG","MILK","WOOL","FERTILIZER")
CROPS = ("WHEAT","CARROT","TOMATO","STRAWBERRY","MELON")
ANIMALS = ("GOOSE","COW","SHEEP")
SHOPS = ("BAKERY","PIZZA_SHOP","BRUNCH_SPOT","YARN_STORE","ICE_CREAM_SHOP","PET_CAFE","SMOOTHIE_SHOP","FARMERS_MARKET")
LOOKAHEAD = 4
WEED_REPLAY_STEPS = 8
SEED_TRIM = {619: 1, 620: 1, 633: 2, 641: 1}
# Native market MPC remains a development/search feature until it clears the
# exact arena. Shipping infrastructure must not silently promote an unverified
# controller merely because libkag_agent.so is present.
ENABLE_NATIVE_MARKET = False
# Promoted adaptive-controller parameters. These are implemented locally from
# publicly observable mechanisms and independently revalidated in our exact arena.
YARN_SWITCH_STEP = 161
YARN_PURCHASE_STEP = 192
YARN_HAND_INDEX = 4
YARN_HAND_START = 193
YARN_HAND_END = 212
WOOL_GATES = ((480, 170), (600, 120), (672, 80), (719, 1))
WOOL_PRESSURE = 78
WOOL_BATCH = 16
WOOL_GAP = 6
RC2_CHECKPOINTS = (216, 240, 264)
RC2_DISTANCE_MAX = 8
RC2_RELAY_LEAD = 3
RC2_RELAY_FIRST = 278
RC2_RELAY_LAST = 662
PRICE_FLOOR = 1
DEMAND_ALPHA = 0.25
EVAC_HORIZON = 3
EVAC_PRODUCTS = ("WOOL", "MILK", "STRAWBERRY")
EVAC_BASE_VALUE = {"EGG": 50, "MILK": 160, "WOOL": 200, "CARROT": 30, "TOMATO": 40, "STRAWBERRY": 200, "MELON": 250}
EVAC_SAFE_ACTIONS = ("PASS", "COLLECT_FERTILIZER")
SHED_ACCESS = ((4, 4), (5, 4), (4, 5), (5, 5))
WHEAT_SHIFT_QTY = 3
WHEAT_SHIFT_LEAD = 1
WHEAT_SHIFT_CASH_RESERVE = 2000
WHEAT_SELL_SHIFT_LEAD = 1
WHEAT_SELL_SHIFT_CAP = 24
MOVE_DELTA = {"NORTH": (0, -1), "SOUTH": (0, 1), "WEST": (-1, 0), "EAST": (1, 0)}
MARKET_PARAMS = {
    "WHEAT": (25,10000,400,"sqrt",0.8,"log",0.2),
    "CARROT": (35,10000,450,"log",0.2,"sqrt",0.7),
    "TOMATO": (60,10000,200,"linear",0.4,"sqrt",0.6),
    "STRAWBERRY": (120,10000,100,"sqrt",0.7,"linear",1.6),
    "MELON": (250,10000,300,"log",0.2,"sq",3.6),
    "EGG": (50,10000,332,"linear",0.4,"log",0.2),
    "MILK": (160,10000,122,"sqrt",0.6,"linear",1.6),
    "WOOL": (200,10000,105,"log",0.2,"sq",3.2),
    "FERTILIZER": (100,10000,200,"linear",0.4,"linear",0.4),
}
SHOP_PRODUCTS = {
    "BAKERY": ("EGG","WHEAT"), "PIZZA_SHOP": ("MILK","TOMATO","WHEAT"),
    "BRUNCH_SPOT": ("EGG","WHEAT","STRAWBERRY"), "YARN_STORE": ("WOOL",),
    "ICE_CREAM_SHOP": ("STRAWBERRY","MILK","WHEAT"), "PET_CAFE": ("CARROT",),
    "SMOOTHIE_SHOP": ("STRAWBERRY","MILK"),
    "FARMERS_MARKET": ("WHEAT","CARROT","TOMATO","STRAWBERRY"),
}


def _get(o, k, d=None):
    if isinstance(o, dict): return o.get(k, d)
    g = getattr(o, "get", None)
    if callable(g): return g(k, d)
    return getattr(o, k, d)


def _copy_action(a):
    a = copy.deepcopy(a or {})
    return {"farmer": list(a.get("farmer") or ["PASS"]),
            "hands": [list(x or ["PASS"]) for x in (a.get("hands") or [])],
            "market": [list(x) for x in (a.get("market") or [])]}


ROOT = Path(__file__).resolve().parent
_ROUTE_PAYLOAD = json.loads((ROOT / "routes.json").read_text())
ROUTES = {k: v["actions"] for k, v in _ROUTE_PAYLOAD["routes"].items()}
ROUTE_NAMES = ("v27", "lev", "jince", "breaking")
# Promoted production backbone: Lev's stronger opening is structurally compatible
# with v27 at step 174. The suffix keeps v27's robust late production schedule.
PRIMARY_ROUTE = ROUTES["lev"][:174] + ROUTES["v27"][174:]


def _fertilizer_pickups(action):
    total = 0
    for unit_action in [action.get("farmer", ["PASS"]), *(action.get("hands", []) or [])]:
        if (
            isinstance(unit_action, (list, tuple))
            and len(unit_action) >= 3
            and unit_action[0] == "PICKUP"
            and unit_action[1] == "FERTILIZER"
        ):
            total += max(0, int(unit_action[2]))
    return total


_FERT_PICKUP_SUFFIX = [0] * 720
for _step in range(718, -1, -1):
    _FERT_PICKUP_SUFFIX[_step] = _FERT_PICKUP_SUFFIX[_step + 1] + _fertilizer_pickups(ROUTES["v27"][_step])


class KagFrameC(ctypes.Structure):
    _fields_ = [
        ("step", ctypes.c_int32), ("player", ctypes.c_int32),
        ("turns_per_day", ctypes.c_int32), ("town_shop_sell_interval", ctypes.c_int32),
        ("town_center_sell_interval", ctypes.c_int32),
        ("own_money", ctypes.c_double), ("opp_money", ctypes.c_double),
        ("own_quadrants", ctypes.c_int32), ("opp_quadrants", ctypes.c_int32),
        ("own_hands", ctypes.c_int32), ("opp_hands", ctypes.c_int32),
        ("own_crops", ctypes.c_int32 * 5), ("opp_crops", ctypes.c_int32 * 5),
        ("own_animals", ctypes.c_int32 * 3), ("opp_animals", ctypes.c_int32 * 3),
        ("shed", ctypes.c_int32 * 9), ("seeds", ctypes.c_int32 * 5),
        ("market_inventory", ctypes.c_int32 * 9), ("market_prices", ctypes.c_int32 * 9),
        ("shop_counts", ctypes.c_int32 * 8),
        ("planned_sell", (ctypes.c_int32 * 9) * 5),
        ("planned_market_slots", ctypes.c_int32 * 5),
    ]


class KagDecisionC(ctypes.Structure):
    _fields_ = [
        ("regime", ctypes.c_int32), ("route_id", ctypes.c_int32),
        ("route_confidence", ctypes.c_double), ("v27_similarity", ctypes.c_double),
        ("preempt_item", ctypes.c_int32), ("preempt_qty", ctypes.c_int32),
        ("preempt_horizon", ctypes.c_int32), ("market_mode", ctypes.c_int32),
        ("estimated_edge", ctypes.c_double),
    ]


def _load_native():
    path = ROOT / "libkag_agent.so"
    if not path.exists(): return None
    lib = ctypes.CDLL(str(path))
    lib.kag_agent_abi_version.restype = ctypes.c_int
    if lib.kag_agent_abi_version() != 1: return None
    lib.kag_agent_decide.argtypes = [ctypes.POINTER(KagFrameC)]
    lib.kag_agent_decide.restype = KagDecisionC
    lib.kag_agent_reset.argtypes = [ctypes.c_int32]
    return lib


_NATIVE = _load_native()
def _fresh_state():
    return {
        "last": -1,
        "route": "v27",
        "debt": {},
        "weed": {},
        "yarn": None,
        "wool_last_sale": -1000,
        "rc2_checks": {},
        "rc2_locked": False,
        "rc2_due_step": -1,
        "rc2_due": 0,
        "fert_excess_debt": 0,
        "livestock_extra": {},
        "livestock_lag": {},
        "evac_plan": None,
        "wheat_buy_debt": {},
        "cross_borrow": {},
        "wheat_sell_debt": {},
    }


_STATE = {0: _fresh_state(), 1: _fresh_state()}


def _farm(obs, seat):
    farms = list(_get(obs, "farms", []) or [])
    return farms[seat] if seat < len(farms) else {}


def _align(action, obs):
    seat = int(_get(obs, "player", 0) or 0); farm = _farm(obs, seat)
    n = len(_get(farm, "hands", []) or []); hands = list(action.get("hands") or [])
    if len(hands) < n: hands += [["PASS"] for _ in range(n - len(hands))]
    action["hands"] = [list(x or ["PASS"]) for x in hands[:n]]
    return action


def _tile_at(farm, position):
    try: return (_get(farm, "tiles", []) or [])[int(position[1])][int(position[0])]
    except (IndexError, TypeError, ValueError): return "LOCKED"


def _trace_actor_action(route, step, actor):
    trace = route[min(max(int(step), 0), len(route) - 1)] or {}
    if actor == "farmer": return list(trace.get("farmer") or ["PASS"])
    hands = trace.get("hands", []) or []
    return list(hands[actor] if actor < len(hands) else ["PASS"])


def _weed_repair(obs, action, route, step, state):
    action = _align(_copy_action(action), obs)
    farm = _farm(obs, int(_get(obs, "player", 0) or 0))
    positions = [_get(farm, "farmer"), *list(_get(farm, "hands", []) or [])]
    unit_actions = [action.get("farmer", ["PASS"]), *list(action.get("hands") or [])]
    active = state["weed"]
    for actor, tx in list(active.items()):
        idx = 0 if actor == "farmer" else int(actor) + 1
        if idx >= len(unit_actions):
            active.pop(actor, None)
            continue
        age = step - tx["start"]
        intended = tx["intended"]
        strawberry = (
            len(intended) >= 2
            and intended[0] == "PLANT"
            and intended[1] == "STRAWBERRY"
        )
        dependency_chain = (
            strawberry
            or bool(intended and intended[0] == "BUILD_PASTURE")
            or bool(len(intended) >= 2 and intended[0] == "PLANT" and intended[1] == "WHEAT")
        )
        if age == 1:
            unit_actions[idx] = list(intended)
        elif dependency_chain and age >= 2:
            # STRAWBERRY is valuable enough that dropping the tail after a fixed
            # replay window creates persistent production deficits. Keep the
            # exact actor one turn behind until a scheduled PASS absorbs the lag.
            unit_actions[idx] = _trace_actor_action(route, step - 1, actor)
            if _trace_actor_action(route, step, actor)[0] == "PASS" or age > 24:
                active.pop(actor, None)
        elif 2 <= age <= 1 + WEED_REPLAY_STEPS:
            unit_actions[idx] = _trace_actor_action(route, step - 1, actor)
        else:
            active.pop(actor, None)
    for idx, (position, intended) in enumerate(zip(positions, unit_actions)):
        actor = "farmer" if idx == 0 else idx - 1
        if actor in active or not isinstance(intended, list) or not intended: continue
        if intended[0] not in ("BUILD_PASTURE", "PLANT"): continue
        tile = _tile_at(farm, position)
        if isinstance(tile, dict) and tile.get("kind") == "WEED":
            active[actor] = {"start": step, "intended": list(intended)}
            unit_actions[idx] = ["DIG"]
    action["farmer"] = unit_actions[0] if unit_actions else ["PASS"]
    action["hands"] = unit_actions[1:]
    return _align(action, obs)


def _cow_count(farm):
    total = 0
    for row in (_get(farm, "tiles", []) or []):
        for tile in row:
            if isinstance(tile, dict) and tile.get("animal") == "COW":
                total += 1
    return total


def _apply_stranded_cow_recovery(action, obs, step, state, route):
    """Recover a publicly visible missing cow without changing route movement."""
    action = _align(_copy_action(action), obs)
    seat = int(_get(obs, "player", 0) or 0)
    farms = list(_get(obs, "farms", []) or [])
    if len(farms) < 2:
        return action
    own, opp = farms[seat], farms[1 - seat]
    extra = state["livestock_extra"]
    lag = state["livestock_lag"]

    # Farm hands are dismissed/re-hired at day boundaries; actor indices must
    # never leak across that identity reset.
    if step % 24 == 0:
        extra.clear()
        lag.clear()

    positions = [_get(own, "farmer"), *list(_get(own, "hands", []) or [])]
    unit_actions = [action.get("farmer", ["PASS"]), *list(action.get("hands") or [])]

    # Replay the displaced action one turn behind. A scheduled PASS is explicit
    # slack: execute the delayed previous action there and rejoin the route.
    for actor in list(lag):
        idx = 0 if actor == "farmer" else int(actor) + 1
        if idx >= len(unit_actions):
            lag.pop(actor, None)
            extra.pop(actor, None)
            continue
        unit_actions[idx] = _trace_actor_action(route, step - 1, actor)
        if _trace_actor_action(route, step, actor)[0] == "PASS":
            lag.pop(actor, None)

    shops = tuple(_get(_get(obs, "town", {}) or {}, "unlocked_shops", []) or [])
    if "YARN_STORE" not in shops:
        own_cows, opp_cows = _cow_count(own), _cow_count(opp)
        shed = _get(_get(obs, "private", {}) or {}, "shed", {}) or {}
        available = max(0, int(_get(shed, "COW", 0) or 0))

        # Piggy-back on an already scheduled COW pickup, so movement is unchanged.
        for idx, raw in enumerate(unit_actions):
            actor = "farmer" if idx == 0 else idx - 1
            order = list(raw or ["PASS"])
            if actor in lag or actor in state["weed"] or actor in extra:
                continue
            if len(order) >= 2 and order[0] == "PICKUP" and order[1] == "COW":
                requested = max(1, int(order[2]) if len(order) >= 3 else 1)
                if own_cows < opp_cows and available > requested:
                    if len(order) < 3:
                        order.append(requested + 1)
                    else:
                        order[2] = requested + 1
                    unit_actions[idx] = order
                    extra[actor] = "COW"
                    break

        # Insert placement only when that carrying actor naturally traverses an
        # empty pasture. One later PASS absorbs the displaced route action.
        for idx, (position, raw) in enumerate(zip(positions, unit_actions)):
            actor = "farmer" if idx == 0 else idx - 1
            if extra.get(actor) != "COW" or actor in lag:
                continue
            tile = _tile_at(own, position)
            order = list(raw or ["PASS"])
            if (
                isinstance(tile, dict)
                and tile.get("kind") == "PASTURE"
                and not tile.get("animal")
                and order[0] != "PLACE"
            ):
                unit_actions[idx] = ["PLACE", "COW"]
                extra.pop(actor, None)
                lag[actor] = 1
                break

    action["farmer"] = unit_actions[0] if unit_actions else ["PASS"]
    action["hands"] = unit_actions[1:]
    return _align(action, obs)


def _shape(name, value):
    x = max(0.0, float(value))
    if name == "linear": return x
    if name == "sq": return x * x
    if name == "sqrt": return math.sqrt(x)
    if name == "log": return math.log1p(x)
    if name == "log10": return math.log10(1.0 + x)
    raise ValueError(name)


def _market_price(item, inventory):
    base, eq, scale, bf, bt, af, at = MARKET_PARAMS[item]
    if inventory < eq:
        price = base + bt * base / _shape(bf, scale) * _shape(bf, eq - inventory)
    else:
        price = base - at * base / _shape(af, scale) * _shape(af, inventory - eq)
    return max(PRICE_FLOOR, int(round(price)))


def _is_sell(o):
    return isinstance(o, (list, tuple)) and len(o) >= 3 and o[0] == "SELL" and o[1] in MARKET_PARAMS


def _demand_per_day(obs, cfg, item):
    shops = list(_get(_get(obs, "town", {}) or {}, "unlocked_shops", []) or [])
    tpd = int(_get(cfg, "turnsPerDay", 24) or 24); interval = max(1, int(_get(cfg, "townShopSellInterval", 4) or 4))
    demand = 0.0
    for shop in shops:
        products = SHOP_PRODUCTS.get(shop, ())
        if item in products: demand += (tpd / interval) * (2 if len(products) == 1 else 1)
    if item != "FERTILIZER": demand += tpd / max(1, int(_get(cfg, "townCenterSellInterval", 24) or 24))
    return demand


def _sell_order_score(obs, cfg, order):
    if not _is_sell(order): return float("-inf")
    item, q = str(order[1]), max(0, int(order[2])); market = _get(obs, "market", {}) or {}
    inv = _get(market, "inventory", {}) or {}; prices = _get(market, "prices", {}) or {}
    current_inv = int(_get(inv, item, 10000) or 0); quote = float(_get(prices, item, _market_price(item, current_inv)) or 0)
    later = float(_market_price(item, current_inv + q)); impact = float(q) * max(0.0, quote - later)
    if impact <= 0: return impact
    demand = max(0.25, _demand_per_day(obs, cfg, item)); excess = max(0.0, current_inv + q - 10000)
    urgency = min(1.0, (excess / demand) / 10.0)
    return impact * (1.0 + DEMAND_ALPHA * urgency)


def _rank_sell_slots(obs, action, cfg):
    action = _copy_action(action); market = list(action.get("market") or [])
    rows = [(_sell_order_score(obs, cfg, o), -i, list(o)) for i, o in enumerate(market) if _is_sell(o)]
    if len(rows) < 2: return action
    rows.sort(reverse=True); ranked = iter(row[2] for row in rows)
    action["market"] = [next(ranked) if _is_sell(o) else o for o in market]
    return action


def _yarn_route(obs, step, state):
    if step < YARN_SWITCH_STEP:
        return False
    if state.get("yarn") is None:
        shops = tuple(_get(_get(obs, "town", {}) or {}, "unlocked_shops", []) or [])
        blocked = {"SMOOTHIE_SHOP", "PIZZA_SHOP"}
        state["yarn"] = "YARN_STORE" in shops and not any(shop in blocked for shop in shops)
    return bool(state["yarn"])


def _swap_cow_for_sheep(order):
    order = list(order or ["PASS"])
    if len(order) >= 2 and order[1] == "COW":
        order[1] = "SHEEP"
    return order


def _apply_yarn_livestock(action, obs, step, state):
    action = _copy_action(action)
    if not _yarn_route(obs, step, state):
        return action
    if step == YARN_PURCHASE_STEP:
        action["market"] = [_swap_cow_for_sheep(o) for o in action.get("market", [])]
    hands = [list(o or ["PASS"]) for o in action.get("hands", [])]
    if YARN_HAND_START <= step <= YARN_HAND_END and YARN_HAND_INDEX < len(hands):
        hands[YARN_HAND_INDEX] = _swap_cow_for_sheep(hands[YARN_HAND_INDEX])
    action["hands"] = hands
    return action


def _wool_gate(step):
    for end_step, gate in WOOL_GATES:
        if step < end_step:
            return gate
    return 1


def _apply_wool_release(action, obs, step, state):
    action = _copy_action(action)
    if not _yarn_route(obs, step, state):
        return action
    market_orders = [list(o) for o in action.get("market", [])]
    if any(len(o) >= 2 and o[0] == "SELL" and o[1] == "WOOL" for o in market_orders):
        return action
    shed = _get(_get(obs, "private", {}) or {}, "shed", {}) or {}
    wool = max(0, int(_get(shed, "WOOL", 0) or 0))
    if wool <= 0 or len(market_orders) >= 10:
        return action
    total = sum(max(0, int(value or 0)) for value in shed.values())
    prices = _get(_get(obs, "market", {}) or {}, "prices", {}) or {}
    price = max(0, int(_get(prices, "WOOL", 0) or 0))
    terminal = step >= 713
    pressure = total >= WOOL_PRESSURE
    price_ok = price >= _wool_gate(step)
    gap_ok = step - int(state.get("wool_last_sale", -1000)) >= WOOL_GAP
    if not terminal and not pressure and (not price_ok or not gap_ok):
        return action
    quantity = wool if terminal else min(wool, WOOL_BATCH)
    if pressure:
        quantity = min(wool, max(quantity, total - (WOOL_PRESSURE - 12)))
    market_orders.append(["SELL", "WOOL", max(1, int(quantity))])
    action["market"] = market_orders[:10]
    state["wool_last_sale"] = step
    return action


def _public_signature(farm):
    keys = ("WHEAT","CARROT","TOMATO","STRAWBERRY","MELON","COW","SHEEP","GOOSE","PASTURE","COOP","WEED")
    counts = {key: 0 for key in keys}
    for row in (_get(farm, "tiles", []) or []):
        for tile in row if isinstance(row, list) else [row]:
            if not isinstance(tile, dict):
                continue
            for field in ("crop", "animal", "kind"):
                value = str(tile.get(field, "")).upper()
                if value in counts:
                    counts[value] += 1
                    break
    return (
        len(_get(farm, "hands", []) or []),
        len(_get(farm, "unlocked_quadrants", []) or []),
        tuple(counts[key] for key in sorted(counts)),
    )


def _route_distance(obs):
    farms = list(_get(obs, "farms", []) or [])
    if len(farms) < 2:
        return 10**9
    left, right = _public_signature(farms[0]), _public_signature(farms[1])
    return abs(left[0] - right[0]) + 3 * abs(left[1] - right[1]) + sum(abs(a - b) for a, b in zip(left[2], right[2]))


def _update_rc2_lock(obs, step, state):
    checks = state["rc2_checks"]
    if step in RC2_CHECKPOINTS and step not in checks:
        checks[step] = _route_distance(obs) <= RC2_DISTANCE_MAX
        if all(checkpoint in checks for checkpoint in RC2_CHECKPOINTS):
            state["rc2_locked"] = all(checks.values())


def _repay_rc2(action, state, step):
    due_step = int(state.get("rc2_due_step", -1))
    if due_step != step:
        if 0 <= due_step < step:
            state["rc2_due_step"], state["rc2_due"] = -1, 0
        return action
    remaining = max(0, int(state.get("rc2_due", 0)))
    market = []
    for raw in action.get("market", []):
        order = list(raw)
        if remaining > 0 and len(order) >= 3 and order[0] == "SELL" and order[1] == "FERTILIZER":
            requested = max(0, int(order[2])); reduction = min(requested, remaining)
            requested -= reduction; remaining -= reduction
            if requested <= 0:
                continue
            order[2] = requested
        market.append(order)
    action["market"] = market
    state["rc2_due_step"], state["rc2_due"] = -1, 0
    return action


def _future_relay_fertilizer(step):
    future_step = step + RC2_RELAY_LEAD
    if not (RC2_RELAY_FIRST <= step <= RC2_RELAY_LAST and 281 <= future_step <= 665 and future_step % 24 == 17 and future_step < 719):
        return 0
    return sum(
        max(0, int(order[2]))
        for order in ROUTES["v27"][future_step].get("market", [])
        if len(order) >= 3 and order[0] == "SELL" and order[1] == "FERTILIZER"
    )


def _apply_rc2_relay(action, obs, step, state, cfg):
    action = _copy_action(action)
    _update_rc2_lock(obs, step, state)
    action = _repay_rc2(action, state, step)
    if not state.get("rc2_locked") or state.get("rc2_due", 0):
        return action
    target = _future_relay_fertilizer(step)
    if target <= 0 or len(action.get("market", [])) >= 10:
        return action
    shed = _get(_get(obs, "private", {}) or {}, "shed", {}) or {}
    available = max(0, int(_get(shed, "FERTILIZER", 0) or 0))
    for order in action.get("market", []):
        if len(order) >= 3 and order[0] == "SELL" and order[1] == "FERTILIZER":
            available = max(0, available - max(0, int(order[2])))
    quantity = min(target, available)
    if quantity <= 0:
        return action
    action["market"].append(["SELL", "FERTILIZER", quantity])
    state["rc2_due_step"] = step + RC2_RELAY_LEAD
    state["rc2_due"] = quantity
    return _rank_sell_slots(obs, action, cfg)


def _repay_fertilizer_excess(action, state):
    debt = max(0, int(state.get("fert_excess_debt", 0)))
    market = []
    for raw in action.get("market", []):
        order = list(raw)
        if debt > 0 and len(order) >= 3 and order[0] == "SELL" and order[1] == "FERTILIZER":
            requested = max(0, int(order[2])); reduction = min(requested, debt)
            requested -= reduction; debt -= reduction
            if requested <= 0:
                continue
            order[2] = requested
        market.append(order)
    action["market"] = market
    state["fert_excess_debt"] = debt
    return action


def _apply_fertilizer_excess(action, obs, step, state, cfg):
    action = _repay_fertilizer_excess(_copy_action(action), state)
    if len(action.get("market", [])) >= 10:
        return action
    shed = _get(_get(obs, "private", {}) or {}, "shed", {}) or {}
    stock = max(0, int(_get(shed, "FERTILIZER", 0) or 0))
    planned_now = sum(
        max(0, int(order[2]))
        for order in action.get("market", [])
        if len(order) >= 3 and order[0] == "SELL" and order[1] == "FERTILIZER"
    )
    reserve = _FERT_PICKUP_SUFFIX[min(719, step + 1)]
    excess = max(0, stock - planned_now - reserve)
    if excess <= 0:
        return action
    action["market"].append(["SELL", "FERTILIZER", excess])
    state["fert_excess_debt"] = int(state.get("fert_excess_debt", 0)) + excess
    return _rank_sell_slots(obs, action, cfg)


def _count_farm(farm):
    crops = {k: 0 for k in CROPS}; animals = {k: 0 for k in ANIMALS}
    for row in (_get(farm, "tiles", []) or []):
        for tile in row:
            if not isinstance(tile, dict): continue
            if tile.get("kind") == "PLANT" and tile.get("crop") in crops: crops[tile["crop"]] += 1
            if tile.get("animal") in animals: animals[tile["animal"]] += 1
    return crops, animals


def _planned_sells(route, step):
    out = [[0] * len(PRODUCTS) for _ in range(LOOKAHEAD + 1)]; slots = [0] * (LOOKAHEAD + 1)
    for h in range(LOOKAHEAD + 1):
        m = (route[min(718, step + h)].get("market", []) or []); slots[h] = len(m)
        for o in m:
            if _is_sell(o): out[h][PRODUCTS.index(o[1])] += max(0, int(o[2]))
    return out, slots


def _actor_action(route, step, idx):
    trace = route[min(max(int(step), 0), len(route) - 1)] or {}
    if idx == 0:
        return list(trace.get("farmer") or ["PASS"])
    hands = trace.get("hands", []) or []
    return list(hands[idx - 1] if idx - 1 < len(hands) else ["PASS"])


def _set_actor_action(action, idx, command):
    if idx == 0:
        action["farmer"] = list(command)
        return
    hands = list(action.get("hands") or [])
    if idx - 1 < len(hands):
        hands[idx - 1] = list(command)
    action["hands"] = hands


def _unit_positions(farm):
    return [_get(farm, "farmer"), *list(_get(farm, "hands", []) or [])]


def _private_inventories(obs):
    return list(_get(_get(obs, "private", {}) or {}, "inventories", []) or [])


def _distance(a, b):
    return abs(int(a[0]) - int(b[0])) + abs(int(a[1]) - int(b[1]))


def _step_toward(a, b):
    x, y = int(a[0]), int(a[1]); tx, ty = int(b[0]), int(b[1])
    if x < tx: return ["EAST"]
    if x > tx: return ["WEST"]
    if y < ty: return ["SOUTH"]
    if y > ty: return ["NORTH"]
    return ["PASS"]


def _choose_dayend_evacuation(obs, action, step, route):
    mod = int(step) % 24
    if mod < 24 - EVAC_HORIZON:
        return None
    seat = int(_get(obs, "player", 0) or 0); farm = _farm(obs, seat)
    positions = _unit_positions(farm); inventories = _private_inventories(obs)
    remaining = 24 - mod; best = None
    for idx, (position, inventory) in enumerate(zip(positions, inventories)):
        if not position or not isinstance(inventory, dict):
            continue
        live = action.get("farmer") if idx == 0 else ((action.get("hands") or [])[idx - 1] if idx - 1 < len(action.get("hands") or []) else ["PASS"])
        canonical = _actor_action(route, step, idx)
        if list(live or ["PASS"]) != canonical:
            continue
        end = min(len(route), step + remaining)
        future = [_actor_action(route, t, idx) for t in range(step, end)]
        if any(not a or a[0] not in EVAC_SAFE_ACTIONS for a in future):
            continue
        target = min(SHED_ACCESS, key=lambda q: _distance(position, q)); dist = _distance(position, target)
        if dist + 1 > remaining:
            continue
        lost_fertilizer = sum(1 for a in future if a[0] == "COLLECT_FERTILIZER")
        for product in EVAC_PRODUCTS:
            quantity = max(0, int(inventory.get(product, 0) or 0))
            if quantity <= 0:
                continue
            score = quantity * EVAC_BASE_VALUE.get(product, 1) - 100 * lost_fertilizer
            if best is None or score > best["score"]:
                best = {"actor": idx, "product": product, "qty": quantity, "target": target, "score": score}
    return best


def _apply_dayend_evacuation(action, obs, step, state, route):
    action = _copy_action(action)
    if step % 24 == 0:
        state["evac_plan"] = None
    plan = state.get("evac_plan")
    if plan is None:
        plan = _choose_dayend_evacuation(obs, action, step, route)
        state["evac_plan"] = plan
    if plan is None:
        return action
    seat = int(_get(obs, "player", 0) or 0); farm = _farm(obs, seat); positions = _unit_positions(farm)
    idx = int(plan["actor"])
    if idx >= len(positions):
        state["evac_plan"] = None
        return action
    position, target = positions[idx], plan["target"]
    if tuple(position) != tuple(target):
        _set_actor_action(action, idx, _step_toward(position, target))
        return _align(action, obs)
    if len(action.get("market") or []) >= 10:
        return action
    private = _get(obs, "private", {}) or {}; shed = _get(private, "shed", {}) or {}
    free = max(0, 100 - sum(max(0, int(v or 0)) for v in shed.values()))
    inventories = _private_inventories(obs); inventory = inventories[idx] if idx < len(inventories) and isinstance(inventories[idx], dict) else {}
    quantity = min(int(plan["qty"]), max(0, int(inventory.get(plan["product"], 0) or 0)), free)
    if quantity <= 0:
        state["evac_plan"] = None
        return action
    _set_actor_action(action, idx, ["PLACE", plan["product"], quantity])
    action["market"] = list(action.get("market") or []) + [["SELL", plan["product"], quantity]]
    state["evac_plan"] = None
    return _align(action, obs)


def _future_wheat_buy(route, step):
    future = step + WHEAT_SHIFT_LEAD
    if future >= len(route):
        return 0
    return sum(max(0, int(o[2])) for o in (route[future].get("market") or []) if len(o) >= 3 and o[0] == "BUY_PRODUCT" and o[1] == "WHEAT")


def _reduce_wheat_buy(action, quantity):
    action = _copy_action(action); out = []; remaining = max(0, int(quantity))
    for raw in action.get("market", []):
        order = list(raw)
        if remaining > 0 and len(order) >= 3 and order[0] == "BUY_PRODUCT" and order[1] == "WHEAT":
            reduction = min(remaining, max(0, int(order[2]))); order[2] = int(order[2]) - reduction; remaining -= reduction
            if order[2] <= 0:
                continue
        out.append(order)
    action["market"] = out
    return action


def _apply_wheat_purchase_shift(action, obs, step, state, route):
    action = _copy_action(action); debt = state["wheat_buy_debt"]
    due = max(0, int(debt.pop(step, 0)))
    if due:
        action = _reduce_wheat_buy(action, due)
    future = _future_wheat_buy(route, step)
    if future <= 0 or len(action.get("market") or []) >= 10:
        return action
    seat = int(_get(obs, "player", 0) or 0); money = int(_get(_farm(obs, seat), "money", 0) or 0)
    quantity = min(WHEAT_SHIFT_QTY, future)
    if quantity <= 0 or money < WHEAT_SHIFT_CASH_RESERVE:
        return action
    action["market"] = list(action.get("market") or []) + [["BUY_PRODUCT", "WHEAT", quantity]]
    due_step = step + WHEAT_SHIFT_LEAD; debt[due_step] = int(debt.get(due_step, 0)) + quantity
    return action


def _apply_pass_weed_dig(action, obs):
    action = _align(_copy_action(action), obs)
    seat = int(_get(obs, "player", 0) or 0); farm = _farm(obs, seat)
    positions = _unit_positions(farm)
    unit_actions = [action.get("farmer", ["PASS"]), *list(action.get("hands") or [])]
    for idx, (position, command) in enumerate(zip(positions, unit_actions)):
        if not isinstance(command, list) or not command or command[0] != "PASS":
            continue
        tile = _tile_at(farm, position)
        if isinstance(tile, dict) and tile.get("kind") == "WEED":
            unit_actions[idx] = ["DIG"]
    action["farmer"] = unit_actions[0] if unit_actions else ["PASS"]; action["hands"] = unit_actions[1:]
    return _align(action, obs)


def _movement_destination(position, command):
    if not position or not isinstance(command, list) or not command or command[0] not in MOVE_DELTA:
        return None
    dx, dy = MOVE_DELTA[command[0]]
    return (int(position[0]) + dx, int(position[1]) + dy)


def _apply_cross_worker_preclear(action, obs, step, state, route):
    action = _align(_copy_action(action), obs)
    seat = int(_get(obs, "player", 0) or 0); farm = _farm(obs, seat)
    positions = _unit_positions(farm)
    unit_actions = [action.get("farmer", ["PASS"]), *list(action.get("hands") or [])]
    borrowed = state["cross_borrow"]

    # A cleaner may borrow exactly its immediately-following canonical PASS.
    # Repay that single displaced movement before considering a new preclear.
    for idx, command in list(borrowed.items()):
        if idx < len(unit_actions) and unit_actions[idx] == ["PASS"]:
            unit_actions[idx] = list(command)
        borrowed.pop(idx, None)

    incoming = {}
    for idx, (position, command) in enumerate(zip(positions, unit_actions)):
        destination = _movement_destination(position, command)
        if destination is not None:
            incoming.setdefault(destination, []).append(idx)

    if step + 1 < len(route):
        for cleaner, (position, command) in enumerate(zip(positions, unit_actions)):
            if cleaner in borrowed or not position or not isinstance(command, list) or not command or command[0] not in MOVE_DELTA:
                continue
            tile = _tile_at(farm, position)
            if not isinstance(tile, dict) or tile.get("kind") != "WEED":
                continue
            if _actor_action(route, step + 1, cleaner) != ["PASS"]:
                continue
            here = (int(position[0]), int(position[1])); route_critical = False
            for incoming_actor in incoming.get(here, []):
                next_command = _actor_action(route, step + 1, incoming_actor)
                if next_command and next_command[0] in ("PLANT", "BUILD_PASTURE"):
                    route_critical = True
                    break
            if not route_critical:
                continue
            borrowed[cleaner] = list(command)
            unit_actions[cleaner] = ["DIG"]

    action["farmer"] = unit_actions[0] if unit_actions else ["PASS"]
    action["hands"] = unit_actions[1:]
    return _align(action, obs)


def _wheat_pickups(action):
    total = 0
    for command in [action.get("farmer", ["PASS"]), *(action.get("hands", []) or [])]:
        if isinstance(command, list) and len(command) >= 3 and command[0] == "PICKUP" and command[1] == "WHEAT":
            total += max(0, int(command[2]))
    return total


def _future_wheat_sell(route, step):
    future = step + WHEAT_SELL_SHIFT_LEAD
    if future >= len(route):
        return 0
    return sum(max(0, int(o[2])) for o in (route[future].get("market") or []) if len(o) >= 3 and o[0] == "SELL" and o[1] == "WHEAT")


def _reduce_wheat_sell(action, quantity):
    action = _copy_action(action); out = []; remaining = max(0, int(quantity))
    for raw in action.get("market", []):
        order = list(raw)
        if remaining > 0 and len(order) >= 3 and order[0] == "SELL" and order[1] == "WHEAT":
            reduction = min(remaining, max(0, int(order[2])))
            order[2] = int(order[2]) - reduction; remaining -= reduction
            if order[2] <= 0:
                continue
        out.append(order)
    action["market"] = out
    return action


def _apply_wheat_sell_shift(action, obs, step, state, route):
    action = _copy_action(action); debt = state["wheat_sell_debt"]
    due = max(0, int(debt.pop(step, 0)))
    if due:
        action = _reduce_wheat_sell(action, due)
    future_sale = _future_wheat_sell(route, step)
    if future_sale <= 0 or len(action.get("market") or []) >= 10:
        return action
    private = _get(obs, "private", {}) or {}; shed = _get(private, "shed", {}) or {}
    stock = max(0, int(_get(shed, "WHEAT", 0) or 0))
    # Unit actions execute before market orders. Reserve every WHEAT unit needed
    # by current and next-turn pickups before moving any planned sale forward.
    reserve = _wheat_pickups(action)
    future = step + WHEAT_SELL_SHIFT_LEAD
    if future < len(route):
        reserve += _wheat_pickups(route[future])
    quantity = min(WHEAT_SELL_SHIFT_CAP, future_sale, max(0, stock - reserve))
    if quantity <= 0:
        return action
    action["market"] = list(action.get("market") or []) + [["SELL", "WHEAT", quantity]]
    debt[future] = int(debt.get(future, 0)) + quantity
    return action


def _make_frame(obs, cfg, route):
    f = KagFrameC(); seat = int(_get(obs, "player", 0) or 0); opp = 1 - seat
    farms = list(_get(obs, "farms", []) or []); own, other = farms[seat], farms[opp]
    oc, oa = _count_farm(own); xc, xa = _count_farm(other)
    f.step = int(_get(obs, "step", 0) or 0); f.player = seat
    f.turns_per_day = int(_get(cfg, "turnsPerDay", 24) or 24)
    f.town_shop_sell_interval = int(_get(cfg, "townShopSellInterval", 4) or 4)
    f.town_center_sell_interval = int(_get(cfg, "townCenterSellInterval", 24) or 24)
    f.own_money = float(_get(own, "money", 0) or 0); f.opp_money = float(_get(other, "money", 0) or 0)
    f.own_quadrants = len(_get(own, "unlocked_quadrants", []) or []); f.opp_quadrants = len(_get(other, "unlocked_quadrants", []) or [])
    f.own_hands = len(_get(own, "hands", []) or []); f.opp_hands = len(_get(other, "hands", []) or [])
    for i, k in enumerate(CROPS): f.own_crops[i], f.opp_crops[i] = oc[k], xc[k]
    for i, k in enumerate(ANIMALS): f.own_animals[i], f.opp_animals[i] = oa[k], xa[k]
    priv = _get(obs, "private", {}) or {}; shed = _get(priv, "shed", {}) or {}; seeds = _get(priv, "seeds", {}) or {}
    market = _get(obs, "market", {}) or {}; mi = _get(market, "inventory", {}) or {}; mp = _get(market, "prices", {}) or {}
    for i, k in enumerate(PRODUCTS):
        f.shed[i] = int(_get(shed, k, 0) or 0); f.market_inventory[i] = int(_get(mi, k, 10000) or 10000); f.market_prices[i] = int(_get(mp, k, 1) or 1)
    for i, k in enumerate(CROPS): f.seeds[i] = int(_get(seeds, k, 0) or 0)
    counts = {k: 0 for k in SHOPS}
    for shop in (_get(_get(obs, "town", {}) or {}, "unlocked_shops", []) or []):
        if shop in counts: counts[shop] += 1
    for i, k in enumerate(SHOPS): f.shop_counts[i] = counts[k]
    ps, sl = _planned_sells(route, f.step)
    for h in range(LOOKAHEAD + 1):
        for p in range(len(PRODUCTS)): f.planned_sell[h][p] = ps[h][p]
        f.planned_market_slots[h] = sl[h]
    return f


def _apply_seed_trim(action, step):
    cut = int(SEED_TRIM.get(int(step), 0))
    if cut <= 0:
        return action
    action = _copy_action(action)
    market = []
    for order in action.get("market", []):
        order = list(order)
        if (
            cut > 0
            and len(order) >= 3
            and order[0] == "BUY_SEED"
            and order[1] == "WHEAT"
        ):
            quantity = max(0, int(order[2]))
            removed = min(quantity, cut)
            quantity -= removed
            cut -= removed
            if quantity <= 0:
                continue
            order[2] = quantity
        market.append(order)
    action["market"] = market
    return action


def _apply_debt(market, debt):
    out = []
    for o in market:
        if len(o) >= 3 and o[0] == "SELL" and debt.get(o[1], 0) > 0:
            q = int(o[2]); paid = min(q, debt[o[1]]); debt[o[1]] -= paid; q -= paid
            if q > 0: out.append(["SELL", o[1], q])
        else: out.append(o)
    return out


def _apply_native_market(obs, action, decision, state):
    action = _copy_action(action); action["market"] = _apply_debt(action["market"], state["debt"])
    item_i = int(decision.preempt_item)
    if 0 <= item_i < len(PRODUCTS) and int(decision.preempt_qty) > 0 and len(action["market"]) < 10:
        item = PRODUCTS[item_i]
        if not any(len(o) >= 3 and o[0] == "SELL" and o[1] == item for o in action["market"]):
            q = min(int(decision.preempt_qty), int(_get(_get(_get(obs, "private", {}) or {}, "shed", {}) or {}, item, 0) or 0))
            if q > 0:
                action["market"].append(["SELL", item, q]); state["debt"][item] = state["debt"].get(item, 0) + q
    return action


def agent(obs, configuration=None):
    try:
        seat = int(_get(obs, "player", 0) or 0); step = min(718, max(0, int(_get(obs, "step", 0) or 0))); st = _STATE[seat]
        if step == 0 or step < st["last"]:
            st.clear(); st.update(_fresh_state())
            if _NATIVE is not None: _NATIVE.kag_agent_reset(seat)
        st["last"] = step; route = PRIMARY_ROUTE
        action = _weed_repair(obs, _copy_action(route[step]), route, step, st)
        action = _apply_stranded_cow_recovery(action, obs, step, st, route)
        action = _apply_yarn_livestock(action, obs, step, st)
        action = _apply_wool_release(action, obs, step, st)
        action = _rank_sell_slots(obs, action, configuration)
        action = _apply_rc2_relay(action, obs, step, st, configuration)
        action = _apply_fertilizer_excess(action, obs, step, st, configuration)
        action = _apply_seed_trim(action, step)
        if _NATIVE is not None:
            decision = _NATIVE.kag_agent_decide(ctypes.byref(_make_frame(obs, configuration, route)))
            # Coherent route switching stays disabled until a continuation passes
            # exact paired-seat promotion. Native market interventions are kept
            # research-only until their own hard promotion gate clears.
            if ENABLE_NATIVE_MARKET:
                action = _rank_sell_slots(obs, _apply_native_market(obs, action, decision, st), configuration)
        action = _apply_dayend_evacuation(action, obs, step, st, route)
        action = _apply_wheat_purchase_shift(action, obs, step, st, route)
        action = _apply_pass_weed_dig(action, obs)
        action = _apply_cross_worker_preclear(action, obs, step, st, route)
        action = _apply_wheat_sell_shift(action, obs, step, st, route)
        return _align(action, obs)
    except Exception:
        seat = int(_get(obs, "player", 0) or 0); farm = _farm(obs, seat)
        return {"farmer": ["PASS"], "hands": [["PASS"] for _ in (_get(farm, "hands", []) or [])], "market": []}
