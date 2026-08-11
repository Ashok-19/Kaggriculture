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
PRICE_FLOOR = 1
DEMAND_ALPHA = 0.25
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
_STATE = {
    0: {"last": -1, "route": "v27", "debt": {}, "weed": {}},
    1: {"last": -1, "route": "v27", "debt": {}, "weed": {}},
}


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
        if idx >= len(unit_actions): active.pop(actor, None); continue
        age = step - tx["start"]
        if age == 1: unit_actions[idx] = list(tx["intended"])
        elif 2 <= age <= 1 + WEED_REPLAY_STEPS: unit_actions[idx] = _trace_actor_action(route, step - 1, actor)
        else: active.pop(actor, None)
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
            st.clear(); st.update({"last": -1, "route": "v27", "debt": {}, "weed": {}})
            if _NATIVE is not None: _NATIVE.kag_agent_reset(seat)
        st["last"] = step; route = ROUTES[st["route"]]
        action = _weed_repair(obs, _copy_action(route[step]), route, step, st)
        action = _apply_seed_trim(action, step)
        action = _rank_sell_slots(obs, action, configuration)
        if _NATIVE is not None:
            decision = _NATIVE.kag_agent_decide(ctypes.byref(_make_frame(obs, configuration, route)))
            # Coherent route switching stays disabled until a continuation passes
            # exact paired-seat promotion. Native market interventions are kept
            # research-only until their own hard promotion gate clears.
            if ENABLE_NATIVE_MARKET:
                action = _rank_sell_slots(obs, _apply_native_market(obs, action, decision, st), configuration)
        return _align(action, obs)
    except Exception:
        seat = int(_get(obs, "player", 0) or 0); farm = _farm(obs, seat)
        return {"farmer": ["PASS"], "hands": [["PASS"] for _ in (_get(farm, "hands", []) or [])], "market": []}
