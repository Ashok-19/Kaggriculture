"""Route-pack job grammar for the replay-seeded production-route pilot.

The frozen Shop Plus executor is never modified. A candidate is a complete
alternative route pack; only the searching seat reads it. Jobs are mined from
the inherited tapes with an intent-level lane/tile model that mirrors the audited
engine rules, so every transformation can be checked for dependency closure
before a game is simulated.
"""
from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path

from kaggriculture_bonsai.runtime import bonsai_kg as bkg
from kaggriculture_bonsai.runtime.route_format import ACTION_SIZE, DESC, HEADER, MAX_MARKET, MAX_SAVED_HANDS, ORDER, PACK_MAGIC, PACK_VERSION, UNIT

ROOT = Path(__file__).resolve().parents[3]
PORT = ROOT / "workspace/shop_router_0909/port"
STEPS, ROUTES, BOARD, SHED_CAPACITY = 719, 13, 10, 100
U = bkg.UNIT_OPS
M = bkg.MARKET_OPS
ITEMS = bkg.ITEMS
CROPS = {0: dict(seed=10, first=2, last=4, interval=0, cap=6, ongoing=False),
         1: dict(seed=20, first=2, last=3, interval=0, cap=4, ongoing=False),
         2: dict(seed=50, first=8, last=8, interval=1, cap=4, ongoing=True),
         3: dict(seed=100, first=10, last=10, interval=2, cap=4, ongoing=True),
         4: dict(seed=80, first=10, last=12, interval=0, cap=6, ongoing=False)}
ANIMALS = {9: dict(cost=300, coop=True, first=4, interval=1, cap=4, product=5),
           10: dict(cost=400, coop=False, first=8, interval=2, cap=6, product=6),
           11: dict(cost=500, coop=False, first=6, interval=3, cap=6, product=7)}
ACCESS = ((4, 4), (5, 4), (4, 5), (5, 5))
T_EMPTY, T_LOCKED, T_WEED, T_COOP, T_PASTURE, T_PLANT = range(6)


def quadrant(x, y):
    h = BOARD // 2
    return (0 if x < h else 1) + (0 if y < h else 2)


class Pack:
    """Mutable in-memory BKV6 route pack plus its short-form arity table."""

    def __init__(self, pack_path, arity_path):
        raw = Path(pack_path).read_bytes()
        magic, version, route_count, action_count, _ = HEADER.unpack_from(raw, 0)
        if magic != PACK_MAGIC or version != PACK_VERSION or route_count != ROUTES:
            raise ValueError("unexpected route pack header")
        self.descs = [DESC.unpack_from(raw, HEADER.size + i * DESC.size) for i in range(route_count)]
        base = HEADER.size + route_count * DESC.size
        self.action_count = action_count
        self.body = bytearray(raw[base:])
        self.arity = list(struct.unpack("<%dH" % (ROUTES * STEPS), Path(arity_path).read_bytes()))
        self.first = {d[0]: d[2] for d in self.descs}

    def clone(self):
        other = object.__new__(Pack)
        other.descs, other.action_count = list(self.descs), self.action_count
        other.body, other.arity, other.first = bytearray(self.body), list(self.arity), dict(self.first)
        return other

    def index(self, route, step):
        return self.first[route] + step

    def get(self, route, step):
        off = self.index(route, step) * ACTION_SIZE
        saved, n_market = struct.unpack_from("<BB", self.body, off)
        p = off + 2
        units = []
        for _ in range(1 + MAX_SAVED_HANDS):
            units.append(list(UNIT.unpack_from(self.body, p)))
            p += UNIT.size
        market = []
        for _ in range(MAX_MARKET):
            market.append(list(ORDER.unpack_from(self.body, p)))
            p += ORDER.size
        return {"saved": saved, "n_market": n_market, "units": units, "market": market,
                "arity": self.arity[route * STEPS + step]}

    def put(self, route, step, record):
        off = self.index(route, step) * ACTION_SIZE
        struct.pack_into("<BB", self.body, off, record["saved"], record["n_market"])
        p = off + 2
        for unit in record["units"]:
            UNIT.pack_into(self.body, p, unit[0], unit[1], unit[2])
            p += UNIT.size
        for order in record["market"]:
            ORDER.pack_into(self.body, p, order[0], order[1], max(0, order[2]))
            p += ORDER.size
        self.arity[route * STEPS + step] = record["arity"]

    def to_bytes(self):
        payload = bytes(self.body)
        head = HEADER.pack(PACK_MAGIC, PACK_VERSION, ROUTES, self.action_count,
                           hashlib.sha256(b"".join(DESC.pack(*d) for d in self.descs) + payload).digest())
        # The descriptor block is hashed with the payload exactly as it is stored.
        return head + b"".join(DESC.pack(*d) for d in self.descs) + payload

    def arity_bytes(self):
        return struct.pack("<%dH" % (ROUTES * STEPS), *self.arity)


def unit_arity(op, short):
    if op == U["PLANT"]:
        return 2
    if op in (U["PICKUP"], U["PLACE"]):
        return 2 if short else 3
    return 1


class Trace:
    """Intent-level replay of one tape: lane positions, tiles, seeds and shed.

    It follows the audited engine order (unit actions, then market orders, then
    plant decay, then the dawn reset) but assumes no weed interruption, which is
    exactly the schedule the donor tape encodes.
    """

    def __init__(self, pack, route, upto=STEPS):
        self.events = []
        self.tiles = {}
        self.seeds = [0] * 5
        self.shed = [0] * 12
        self.money = 3000
        h = BOARD // 2
        for y in range(BOARD):
            for x in range(BOARD):
                self.tiles[(x, y)] = {"kind": T_EMPTY if quadrant(x, y) == 0 else T_LOCKED, "what": -1,
                                      "animal": False, "planted": -1, "yield": 0, "watered": -1,
                                      "fert_until": -2, "life": -1}
        pos = [(h - 1, h - 1)]
        inv = [dict()]
        for step in range(upto):
            day = step // 24
            record = pack.get(route, step)
            n = 1 + record["saved"]
            demand = {}
            for i in range(n):
                op, arg, _ = record["units"][i]
                if op == U["PLANT"]:
                    demand[arg] = demand.get(arg, 0) + 1
            blocked = {c for c, d in demand.items() if d > self.seeds[c]}
            for i in range(min(n, len(pos))):
                op, arg, count = record["units"][i]
                short = bool(record["arity"][0] if False else (pack.arity[route * STEPS + step] >> i) & 1)
                quantity = count if unit_arity(op, short) >= 3 else 1
                self.apply(step, i, pos, inv, op, arg, quantity, day, blocked)
            for slot in range(record["n_market"]):
                op, item, count = record["market"][slot]
                if op == M["HIRE"]:
                    if len(pos) < 40:
                        occupancy = [sum(pos[u] == cell for u in range(len(pos))) for cell in ACCESS]
                        pos.append(ACCESS[occupancy.index(min(occupancy))])
                        inv.append(dict())
                    continue
                self.market(step, slot, op, item, count, pos, len(pos))
            self.decay(step)
            if (step + 1) % 24 == 0:
                for i in range(len(pos)):
                    for item, amount in inv[i].items():
                        room = max(0, SHED_CAPACITY - sum(self.shed))
                        take = min(amount, room)
                        self.shed[item] += take
                for tile in self.tiles.values():
                    tile["watered"] = -1
                    if tile["animal"]:
                        info = ANIMALS[tile["what"]]
                        age = (day + 1) - tile["planted"]
                        if age >= info["first"] and (age - info["first"]) % max(1, info["interval"]) == 0:
                            tile["yield"] = min(info["cap"], tile["yield"] + 1)
                pos, inv = [(h - 1, h - 1)], [dict()]

    def apply(self, step, lane, pos, inv, op, arg, quantity, day, blocked):
        x, y = pos[lane]
        tile = self.tiles[(x, y)]
        record = {"step": step, "lane": lane, "op": op, "arg": arg, "n": quantity, "tile": [x, y], "ok": False}
        if op in (U["NORTH"], U["SOUTH"], U["EAST"], U["WEST"]):
            nx = x + (op == U["EAST"]) - (op == U["WEST"])
            ny = y + (op == U["SOUTH"]) - (op == U["NORTH"])
            if 0 <= nx < BOARD and 0 <= ny < BOARD:
                pos[lane] = (nx, ny)
            return
        if op == U["PASS"]:
            return
        shed_side = (x in (4, 5)) and (y in (4, 5))
        if op == U["DROP"]:
            if shed_side:
                for item, amount in inv[lane].items():
                    room = max(0, SHED_CAPACITY - sum(self.shed))
                    self.shed[item] += min(amount, room)
                inv[lane] = {}
                record["ok"] = True
        elif op == U["PICKUP"]:
            if shed_side and quantity > 0:
                take = min(quantity, self.shed[arg])
                if take > 0:
                    self.shed[arg] -= take
                    inv[lane][arg] = inv[lane].get(arg, 0) + take
                    record["ok"] = True
        elif op == U["PLACE"]:
            if arg >= 9:
                want = T_COOP if ANIMALS[arg]["coop"] else T_PASTURE
                if tile["kind"] == want and not tile["animal"] and inv[lane].get(arg, 0) > 0:
                    inv[lane][arg] -= 1
                    tile.update(kind=want, what=arg, animal=True, planted=day, yield_=0)
                    tile["yield"] = 0
                    record["ok"] = True
            elif shed_side:
                take = min(quantity, inv[lane].get(arg, 0), max(0, SHED_CAPACITY - sum(self.shed)))
                if take > 0:
                    inv[lane][arg] -= take
                    self.shed[arg] += take
                    record["ok"] = True
        elif tile["kind"] != T_LOCKED:
            info = CROPS.get(tile["what"]) if tile["kind"] == T_PLANT else None
            if op == U["PLANT"]:
                if arg < 5 and tile["kind"] == T_EMPTY and arg not in blocked and self.seeds[arg] > 0:
                    self.seeds[arg] -= 1
                    crop = CROPS[arg]
                    tile.update(kind=T_PLANT, what=arg, animal=False, planted=day,
                                yield_=0, fert_until=-2)
                    tile["yield"] = 0 if crop["ongoing"] else 1
                    tile["life"] = -1 if crop["ongoing"] else (day + crop["last"] + 1) * 24
                    record["ok"] = True
            elif op == U["WATER"]:
                if info and tile["watered"] != day:
                    tile["watered"] = day
                    record["ok"] = True
                    if not info["ongoing"]:
                        age = day - tile["planted"]
                        if (info["last"] + 1) // 2 <= age <= info["last"]:
                            bonus = 2 if tile["fert_until"] >= day else 1
                            tile["yield"] = min(info["cap"], tile["yield"] + bonus)
            elif op == U["FERTILIZE"]:
                if info and inv[lane].get(8, 0) > 0:
                    inv[lane][8] -= 1
                    tile["fert_until"] = max(tile["fert_until"], day + 2)
                    record["ok"] = True
            elif op == U["HARVEST"]:
                if tile["kind"] in (T_PLANT, T_COOP, T_PASTURE) and tile["yield"] > 0:
                    if tile["kind"] == T_PLANT:
                        if day - tile["planted"] >= info["first"]:
                            inv[lane][tile["what"]] = inv[lane].get(tile["what"], 0) + tile["yield"]
                            record["ok"], record["units"] = True, tile["yield"]
                            tile["yield"] = 0
                            if not info["ongoing"]:
                                self.tiles[(x, y)] = dict(tile, kind=T_EMPTY, what=-1, animal=False,
                                                          planted=-1, life=-1, watered=-1, fert_until=-2)
                                self.tiles[(x, y)]["yield"] = 0
                    elif tile["animal"]:
                        product = ANIMALS[tile["what"]]["product"]
                        inv[lane][product] = inv[lane].get(product, 0) + tile["yield"]
                        record["ok"], record["units"] = True, tile["yield"]
                        tile["yield"] = 0
            elif op == U["DIG"]:
                if tile["kind"] != T_EMPTY and not tile["animal"]:
                    self.tiles[(x, y)] = dict(tile, kind=T_EMPTY, what=-1, planted=-1, life=-1)
                    self.tiles[(x, y)]["yield"] = 0
                    record["ok"] = True
            elif op in (U["BUILD_COOP"], U["BUILD_PASTURE"]):
                if tile["kind"] == T_EMPTY:
                    tile.update(kind=T_COOP if op == U["BUILD_COOP"] else T_PASTURE, what=-1, animal=False)
                    record["ok"] = True
            elif op == U["FEED"]:
                if tile["animal"] and inv[lane].get(0, 0) > 0:
                    inv[lane][0] -= 1
                    record["ok"] = True
            elif op == U["COLLECT_FERTILIZER"]:
                if tile["animal"]:
                    inv[lane][8] = inv[lane].get(8, 0) + 1
                    record["ok"] = True
            elif op == U["CARE"]:
                record["ok"] = bool(tile["animal"])
        if op not in (U["PASS"],):
            self.events.append(record)

    def market(self, step, slot, op, item, count, pos, n_units):
        record = {"step": step, "slot": slot, "op": op, "item": item, "n": count, "market": True}
        if op == M["BUY_SEED"] and item < 5:
            self.seeds[item] += count
        elif op == M["BUY_ANIMAL"] and item >= 9:
            room = max(0, SHED_CAPACITY - sum(self.shed))
            self.shed[item] += min(count, room)
        elif op == M["BUY_PRODUCT"] and item < 9:
            room = max(0, SHED_CAPACITY - sum(self.shed))
            self.shed[item] += min(count, room)
        elif op == M["SELL"] and item < 9:
            self.shed[item] = max(0, self.shed[item] - count)
        if op:
            self.events.append(record)

    def decay(self, step):
        for key, tile in self.tiles.items():
            if tile["kind"] == T_PLANT and tile["life"] >= 0 and step + 1 >= tile["life"]:
                self.tiles[key] = dict(tile, kind=T_EMPTY, what=-1, planted=-1, life=-1, watered=-1, fert_until=-2)
                self.tiles[key]["yield"] = 0


def parse_dump(path):
    """Exact engine annotation stream produced by the host evaluator."""
    events, tiles, state = [], {}, {}
    for line in Path(path).read_text().splitlines():
        parts = line.split()
        if parts[0] == "E":
            step, lane, op, arg, n, x, y, kind, what, animal, yield_units, planted = map(int, parts[1:])
            events.append({"step": step, "lane": lane, "op": op, "arg": arg, "n": n, "tile": (x, y),
                           "kind": kind, "what": what, "animal": animal, "yield": yield_units, "planted": planted})
        elif parts[0] == "M":
            step, slot, op, item, n = map(int, parts[1:])
            events.append({"step": step, "slot": slot, "op": op, "item": item, "n": n, "market": True})
        elif parts[0] == "T":
            step, x, y, kind, what, animal, yield_units, planted = map(int, parts[1:])
            tiles.setdefault((x, y), []).append((step, kind, what, animal, yield_units, planted))
        elif parts[0] == "S":
            values = list(map(int, parts[1:]))
            state[values[0]] = {"money": values[1], "units": values[2], "seeds": values[3:8],
                                "shed": values[8:20], "produced": values[20:32]}
    return events, tiles, state


def mine_jobs(events, tiles, state, after=144):
    """Auditable post-checkpoint production jobs with their dependency closure."""
    by_step_lane = {(e["step"], e["lane"]): e for e in events if "lane" in e}
    orders = [e for e in events if e.get("market")]
    jobs = []
    for cell, history in sorted(tiles.items()):
        seen_animal, seen_crop = False, False
        for index, (step, kind, what, animal, _, planted) in enumerate(history):
            previous = history[index - 1] if index else None
            if animal and (previous is None or not previous[3] or previous[2] != what):
                seen_animal = True
                if step - 1 < after:
                    continue
                jobs.append(animal_job(cell, step - 1, what, planted, events, by_step_lane, orders, history))
            if kind == T_PLANT and (previous is None or previous[1] != T_PLANT or previous[5] != planted):
                seen_crop = True
                if step - 1 < after:
                    continue
                jobs.append(crop_job(cell, step - 1, what, planted, events, by_step_lane, orders, history))
        del seen_animal, seen_crop
    return [j for j in jobs if j]


def _service(cell, start, events, stop=STEPS):
    return [{"step": e["step"], "lane": e["lane"], "op": e["op"]} for e in events
            if "lane" in e and e["tile"] == cell and start < e["step"] < stop
            and e["op"] in (U["FEED"], U["CARE"], U["COLLECT_FERTILIZER"], U["HARVEST"],
                            U["WATER"], U["FERTILIZE"], U["PLANT"], U["DIG"])]


def animal_job(cell, step, species, planted, events, by_step_lane, orders, history):
    place = next((e for e in events if "lane" in e and e["step"] == step and e["op"] == U["PLACE"]
                  and e["arg"] == species and e["tile"] == cell), None)
    if place is None:
        return None
    pickup = next((e for e in reversed([e for e in events if "lane" in e and e["op"] == U["PICKUP"]
                                        and e["arg"] == species and e["step"] <= step
                                        and e["lane"] == place["lane"]])), None)
    purchase = next((e for e in reversed([o for o in orders if o["op"] == M["BUY_ANIMAL"]
                                          and o["item"] == species
                                          and (pickup is None or o["step"] <= pickup["step"])])), None)
    build = next((e for e in reversed([e for e in events if "lane" in e and e["tile"] == cell
                                       and e["op"] in (U["BUILD_COOP"], U["BUILD_PASTURE"])
                                       and e["step"] < step])), None)
    if pickup is None or purchase is None:
        return None
    return {"family": "ANIMAL", "cell": list(cell), "species": species, "product": ANIMALS[species]["product"],
            "place": {"step": place["step"], "lane": place["lane"]},
            "pickup": {"step": pickup["step"], "lane": pickup["lane"], "n": pickup["n"]},
            "purchase": {"step": purchase["step"], "slot": purchase["slot"], "n": purchase["n"]},
            "build": None if build is None else {"step": build["step"], "lane": build["lane"], "op": build["op"]},
            "planted_day": planted, "service": _service(cell, step, events),
            "harvests": [s["step"] for s in _service(cell, step, events) if s["op"] == U["HARVEST"]]}


def crop_job(cell, step, crop, planted, events, by_step_lane, orders, history):
    plant = next((e for e in events if "lane" in e and e["step"] == step and e["op"] == U["PLANT"]
                  and e["arg"] == crop and e["tile"] == cell), None)
    if plant is None:
        return None
    purchase = next((e for e in reversed([o for o in orders if o["op"] == M["BUY_SEED"]
                                          and o["item"] == crop and o["step"] < step])), None)
    if purchase is None:
        return None
    live = {h[0] for h in history if h[1] == T_PLANT and h[5] == planted}
    cleared = next((t for t in range(step + 1, STEPS) if t not in live), STEPS)
    service = _service(cell, step, events, cleared)
    return {"family": "CROP", "cell": list(cell), "species": crop, "product": crop,
            "plant": {"step": plant["step"], "lane": plant["lane"]},
            "purchase": {"step": purchase["step"], "slot": purchase["slot"], "n": purchase["n"]},
            "planted_day": planted, "cleared_step": cleared, "service": service,
            "harvests": [s["step"] for s in service if s["op"] == U["HARVEST"]],
            "waters": [s["step"] for s in service if s["op"] == U["WATER"]]}


class InvalidCandidate(RuntimeError):
    pass


class RouteView:
    """Mined jobs and exact lane geometry for one inherited route."""

    def __init__(self, route, dump_path):
        self.route = route
        self.events, self.tiles, self.state = parse_dump(dump_path)
        self.jobs = mine_jobs(self.events, self.tiles, self.state)
        self.position = {(e["step"], e["lane"]): e["tile"] for e in self.events if "lane" in e}
        self.lane_op = {(e["step"], e["lane"]): e["op"] for e in self.events if "lane" in e}
        self.ordinal = {}
        for family in ("ANIMAL", "CROP"):
            for species in sorted({j["species"] for j in self.jobs if j["family"] == family}):
                chosen = sorted([j for j in self.jobs if j["family"] == family and j["species"] == species],
                                key=lambda j: j.get("place", j.get("plant"))["step"])
                for index, job in enumerate(chosen):
                    self.ordinal[(family, species, index)] = job

    def job(self, family, species, index):
        return self.ordinal.get((family, species, index))

    def support(self, family):
        return sorted(key for key in self.ordinal if key[0] == family)


class Editor:
    """Applies dependency-closed job edits to one route of a candidate pack."""

    def __init__(self, pack, route, view):
        self.pack, self.route, self.view = pack, route, view
        self.closure = []

    def unit(self, step, lane, op, arg, count, short=True):
        record = self.pack.get(self.route, step)
        if lane > record["saved"]:
            raise InvalidCandidate("lane outside stored hand count")
        record["units"][lane] = [op, arg, count]
        mask = 1 << lane
        record["arity"] = (record["arity"] | mask) if short else (record["arity"] & ~mask)
        self.pack.put(self.route, step, record)
        self.closure.append(("unit", self.route, step, lane, op, arg, count, int(short)))

    def order(self, step, slot, op, item, count):
        record = self.pack.get(self.route, step)
        if slot >= MAX_MARKET:
            raise InvalidCandidate("market slot outside format limit")
        record["market"][slot] = [op, item, max(0, count)]
        record["n_market"] = max(record["n_market"], slot + 1)
        self.pack.put(self.route, step, record)
        self.closure.append(("order", self.route, step, slot, op, item, max(0, count)))

    def merge_order(self, step, op, item, count):
        """Add `count` units to a matching order, or take a free slot."""
        if count <= 0:
            return
        record = self.pack.get(self.route, step)
        for slot in range(record["n_market"]):
            existing = record["market"][slot]
            if existing[0] == op and existing[1] == item:
                self.order(step, slot, op, item, existing[2] + count)
                return
        if record["n_market"] < MAX_MARKET:
            self.order(step, record["n_market"], op, item, count)
            return
        # A full order book defers the dependent order to the next free turn.
        for later in range(step + 1, min(step + 5, STEPS)):
            record = self.pack.get(self.route, later)
            if record["n_market"] < MAX_MARKET:
                self.order(later, record["n_market"], op, item, count)
                return
        raise InvalidCandidate("no free market slot for the dependent order")

    def reduce_order(self, step, slot, op, item, count):
        record = self.pack.get(self.route, step)
        existing = record["market"][slot]
        if existing[0] != op or existing[1] != item:
            raise InvalidCandidate("financing order moved")
        remaining = existing[2] - count
        self.order(step, slot, op if remaining > 0 else 0, item if remaining > 0 else 0, max(0, remaining))

    def remap_sales(self, start, old_item, new_item, share):
        """Move a proportional part of the planned output obligation."""
        moved = 0
        for step in range(start, STEPS):
            record = self.pack.get(self.route, step)
            for slot in range(record["n_market"]):
                op, item, count = record["market"][slot]
                if op != M["SELL"] or item != old_item or count <= 0:
                    continue
                take = min(count, max(1, int(round(count * share))))
                self.order(step, slot, M["SELL"], old_item, count - take)
                try:
                    self.merge_order(step, M["SELL"], new_item, take)
                except InvalidCandidate:
                    continue
                moved += take
        return moved


PASSABLE = {U["PASS"], U["NORTH"], U["SOUTH"], U["EAST"], U["WEST"]}


def free_slot(view, lane, anchor, direction, span=24, shed=False):
    """Nearest idle step in the same lane standing on the same tile.

    A movement or PASS slot may be reused; anything else already carries work.
    """
    tile = view.position.get((anchor, lane))
    if tile is None:
        raise InvalidCandidate("job step is outside the annotated trace")
    order = range(anchor - 1, anchor - span - 1, -1) if direction < 0 else range(anchor + 1, anchor + span + 1)
    for step in order:
        if not 144 <= step < STEPS:
            break
        where = view.position.get((step, lane))
        if where is None:
            continue
        if shed:
            if where[0] not in (4, 5) or where[1] not in (4, 5):
                continue
        elif where != tile:
            continue
        if view.lane_op.get((step, lane)) not in PASSABLE:
            continue
        return step
    raise InvalidCandidate("no compatible slot within one day")


def animal_convert(editor, job, target, shift=0):
    """Move one head of a placement job to another species, closing purchase,
    structure, pickup, placement and output obligations."""
    species = job["species"]
    if target == species or target not in ANIMALS:
        raise InvalidCandidate("null animal conversion")
    want_coop = ANIMALS[target]["coop"]
    if want_coop != ANIMALS[species]["coop"]:
        if job["build"] is None:
            raise InvalidCandidate("no auditable structure to repurpose")
        editor.unit(job["build"]["step"], job["build"]["lane"],
                    U["BUILD_COOP"] if want_coop else U["BUILD_PASTURE"], 0, 1)
    purchase, pickup, place = job["purchase"], job["pickup"], job["place"]
    steps = {"purchase": purchase["step"], "pickup": pickup["step"], "place": place["step"]}
    if shift:
        direction = -1 if shift < 0 else 1
        steps["pickup"] = free_slot(editor.view, pickup["lane"], pickup["step"], direction, shed=True)
        steps["place"] = free_slot(editor.view, place["lane"], place["step"], direction)
        if steps["place"] <= steps["pickup"]:
            raise InvalidCandidate("retimed placement would precede its pickup")
        steps["purchase"] = max(144, min(purchase["step"] + shift, steps["pickup"] - 1))
        editor.unit(pickup["step"], pickup["lane"], U["PASS"], 0, 1)
        editor.unit(place["step"], place["lane"], U["PASS"], 0, 1)
    editor.reduce_order(purchase["step"], purchase["slot"], M["BUY_ANIMAL"], species, 1)
    editor.merge_order(steps["purchase"], M["BUY_ANIMAL"], target, 1)
    editor.unit(steps["pickup"], pickup["lane"], U["PICKUP"], target, 1)
    editor.unit(steps["place"], place["lane"], U["PLACE"], target, 1)
    first = job["harvests"][0] if job["harvests"] else steps["place"]
    editor.remap_sales(first, ANIMALS[species]["product"], ANIMALS[target]["product"], job["share"])
    return {"kind": "ANIMAL_CONVERT", "cell": job["cell"], "from": ITEMS[species], "to": ITEMS[target],
            "shift": shift, "place_step": steps["place"]}


def crop_convert(editor, job, target, shift=0):
    """Move one planting lifecycle to another crop, closing seed financing,
    service cadence, feed replacement and output obligations."""
    species = job["species"]
    if target == species or target not in CROPS:
        raise InvalidCandidate("null crop conversion")
    plant, purchase = job["plant"], job["purchase"]
    step = plant["step"]
    if shift:
        step = free_slot(editor.view, plant["lane"], plant["step"], -1 if shift < 0 else 1)
        editor.unit(plant["step"], plant["lane"], U["PASS"], 0, 1)
    editor.reduce_order(purchase["step"], purchase["slot"], M["BUY_SEED"], species, 1)
    editor.merge_order(max(144, min(step - 1, purchase["step"])), M["BUY_SEED"], target, 1)
    editor.unit(step, plant["lane"], U["PLANT"], target, 1)
    old, new = CROPS[species], CROPS[target]
    day = step // 24
    lost = 0
    for service in job["service"]:
        if service["op"] != U["HARVEST"]:
            continue
        age = service["step"] // 24 - day
        ready = age >= new["first"] and (new["ongoing"] or age <= new["last"] + 1)
        if not ready:
            lost += 1
    if old["ongoing"] and not new["ongoing"]:
        # A finite crop clears its tile: give the freed service slots back to a replant.
        replanted = 0
        for service in job["service"]:
            if service["op"] in (U["WATER"], U["HARVEST"]) and service["step"] > step + new["last"] * 24 and replanted < 2:
                editor.unit(service["step"], service["lane"], U["PLANT"], target, 1)
                editor.merge_order(service["step"] - 1, M["BUY_SEED"], target, 1)
                replanted += 1
    if not old["ongoing"] and new["ongoing"]:
        for service in job["service"]:
            if service["op"] == U["PLANT"]:
                editor.unit(service["step"], service["lane"], U["HARVEST"], 0, 1)
    if species == 0 and lost:
        # WHEAT is the feed item; replace the lost feed rather than starving the herd.
        editor.merge_order(job["harvests"][0] if job["harvests"] else step, M["BUY_PRODUCT"], 0,
                           min(20, 3 * max(1, lost)))
    editor.remap_sales(job["harvests"][0] if job["harvests"] else step, species, target, job["share"])
    return {"kind": "CROP_CONVERT", "cell": job["cell"], "from": ITEMS[species], "to": ITEMS[target],
            "shift": shift, "plant_step": step, "lost_harvests": lost}


def annotate_shares(view):
    """Each job's share of its species' planned output, used for obligations."""
    for family in ("ANIMAL", "CROP"):
        for species in sorted({j["species"] for j in view.jobs if j["family"] == family}):
            chosen = [j for j in view.jobs if j["family"] == family and j["species"] == species]
            for job in chosen:
                job["share"] = 1.0 / max(1, len(chosen))
