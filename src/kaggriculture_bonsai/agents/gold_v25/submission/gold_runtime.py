"""Self-contained CPU loader for the frozen BKG4 GOLD V25 submission."""
import ctypes as c
import hashlib
import json
from pathlib import Path

ITEMS = ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL", "FERTILIZER", "GOOSE", "COW", "SHEEP")
ITEM_ID = {name: i for i, name in enumerate(ITEMS)}
UNIT_NAMES = {i: name for i, name in enumerate(("PASS", "NORTH", "SOUTH", "EAST", "WEST", "PICKUP", "DROP", "PLACE", "PLANT", "WATER", "HARVEST", "FERTILIZE", "DIG", "BUILD_COOP", "BUILD_PASTURE", "FEED", "COLLECT_FERTILIZER", "CARE"))}
ORDER_NAMES = {0: "NONE", 1: "HIRE", 2: "BUY_LAND", 3: "BUY_SEED", 4: "BUY_PRODUCT", 5: "BUY_ANIMAL", 6: "SELL"}
PRODUCTS = ITEMS[:9]
I = c.c_int32


class Unit(c.Structure):
    _fields_ = [(name, I) for name in ("op", "arg", "n", "arity")]


class Order(c.Structure):
    _fields_ = [(name, I) for name in ("op", "item", "n")]


class Tile(c.Structure):
    _fields_ = [(name, I) for name in ("kind", "animal", "watered", "yield_units", "fed", "fertilizer", "cared")]


class Worker(c.Structure):
    _fields_ = [(name, I) for name in ("x", "y", "inv_count")] + [("keys", I * 12), ("inv", I * 12), ("tile", Tile)]


class Observation(c.Structure):
    _fields_ = [(name, I) for name in ("step", "player", "board", "n_units", "yarn", "egg")] + [(name, I * 12) for name in ("shed", "seeds", "prices")] + [("workers", Worker * 40)]


class Action(c.Structure):
    _fields_ = [("n_units", I), ("n_market", I), ("units", Unit * 40), ("market", Order * 10)]


class Result(c.Structure):
    _fields_ = [("action", Action)] + [(name, I) for name in ("last_step", "route", "day6", "day27", "due")] + [("suppress", I * 12), ("fault", I), ("pending_counts", I * 40), ("pending_hash", c.c_uint64)]


def _get(value, key, default=None):
    if isinstance(value, dict):
        return value.get(key, default)
    getter = getattr(value, "get", None)
    if callable(getter):
        return getter(key, default)
    return getattr(value, key, default)


def _int(value, default=0):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _tile_at(tiles, pos):
    try:
        return tiles[int(pos[1])][int(pos[0])]
    except (TypeError, ValueError, IndexError):
        return "LOCKED"


def _adjacent(pos, board):
    half = board // 2
    return pos[0] in (half - 1, half) and pos[1] in (half - 1, half)


def encode_observation(obs):
    out = Observation()
    out.step = _int(_get(obs, "step", 0))
    out.player = _int(_get(obs, "player", 0))
    farms = list(_get(obs, "farms", []) or [])
    farm = farms[out.player] if 0 <= out.player < len(farms) else {}
    tiles = _get(farm, "tiles", []) or []
    out.board = len(tiles) or 10
    hands = list(_get(farm, "hands", []) or [])
    positions = [_get(farm, "farmer", None)] + hands
    out.n_units = len(positions)
    if not 1 <= out.n_units <= 40:
        raise ValueError("native BKG4 unit limit")
    town = _get(obs, "town", {}) or {}
    out.yarn = int("YARN_STORE" in (_get(town, "unlocked_shops", []) or []))
    private = _get(obs, "private", {}) or {}
    shed = _get(private, "shed", {}) or {}
    seeds = _get(private, "seeds", {}) or {}
    inventories = list(_get(private, "inventories", []) or [])
    market = _get(obs, "market", {}) or {}
    prices = _get(market, "prices", {}) or {}
    out.egg = _int(_get(_get(market, "inventory", {}) or {}, "EGG", 10000))
    for k, name in enumerate(ITEMS):
        out.shed[k] = max(0, _int(_get(shed, name, 0)))
        out.seeds[k] = _int(_get(seeds, name, 0))
        out.prices[k] = _int(_get(prices, name, 0))
    for i, pos in enumerate(positions):
        if not isinstance(pos, (list, tuple)) or len(pos) < 2:
            raise ValueError("invalid worker position")
        worker = out.workers[i]
        worker.x, worker.y = _int(pos[0]), _int(pos[1])
        inventory = dict(inventories[i] or {}) if i < len(inventories) else {}
        worker.inv_count = len(inventory)
        for j, (name, qty) in enumerate(inventory.items()):
            if j >= 12 or name not in ITEM_ID:
                raise ValueError("invalid inventory")
            worker.keys[j], worker.inv[ITEM_ID[name]] = ITEM_ID[name], _int(qty)
        tile = _tile_at(tiles, pos)
        worker.tile.kind = -1 if tile == "LOCKED" else 0
        if isinstance(tile, dict):
            kind = _get(tile, "kind")
            worker.tile.kind = {"WEED": 1, "PLANT": 2, "COOP": 3, "PASTURE": 4}.get(kind, 5)
            worker.tile.animal = int(_get(tile, "animal") is not None)
            worker.tile.watered = int(bool(_get(tile, "watered_today")))
            worker.tile.yield_units = _int(_get(tile, "yield_units", 0))
            worker.tile.fed = int(bool(_get(tile, "fed_today")))
            worker.tile.fertilizer = int(bool(_get(tile, "fertilizer_available")))
            worker.tile.cared = int(bool(_get(tile, "cared_today")))
    return out


def action_object(action):
    def unit(item):
        result = [UNIT_NAMES[item.op]]
        if item.arity >= 2:
            result.append(ITEMS[item.arg])
        if item.arity >= 3:
            result.append(item.n)
        return result
    units = [unit(item) for item in action.units[:action.n_units]]
    market = []
    for item in action.market[:action.n_market]:
        if item.op == 0:
            market.append([])
        elif item.op in (1, 2):
            market.append([ORDER_NAMES[item.op]])
        else:
            market.append([ORDER_NAMES[item.op], ITEMS[item.item], item.n])
    return {"farmer": units[0], "hands": units[1:], "market": market}


class GoldRuntime:
    def __init__(self, root=None):
        root = Path(root) if root else Path(__file__).resolve().parent
        assets = root / "assets"
        identity = json.loads((assets / "identity.json").read_text())
        for filename, key in (("controller.bkgc", "controller_sha256"), ("routes.bkv6", "route_pack"), ("arity.bin", "arity_sha256")):
            data = (assets / filename).read_bytes()
            expected = identity[key]["pack_sha256"] if filename == "routes.bkv6" else identity[key]
            if hashlib.sha256(data).hexdigest() != expected:
                raise ValueError("packaged gold-v25 asset hash mismatch")
        self.lib = c.CDLL(str(root / "gold_cpu.so"))
        self.lib.gold_error.restype = c.c_char_p
        self.lib.gold_create.argtypes = [c.c_char_p] * 4 + [I]
        self.lib.gold_create.restype = c.c_void_p
        self.lib.gold_destroy.argtypes = [c.c_void_p]
        self.lib.gold_tick.argtypes = [c.c_void_p, c.POINTER(Observation), c.POINTER(Result)]
        if self.lib.gold_size(0) != c.sizeof(Observation) or self.lib.gold_size(1) != c.sizeof(Result):
            raise ValueError("gold-v25 bridge ABI mismatch")
        self.handle = self.lib.gold_create(str(assets / "controller.bkgc").encode(), str(assets / "routes.bkv6").encode(), identity["route_pack"]["pack_sha256"].encode(), str(assets / "arity.bin").encode(), 0)
        if not self.handle:
            raise RuntimeError(self.lib.gold_error().decode())

    def __call__(self, observation, configuration=None):
        encoded, result = encode_observation(observation), Result()
        if self.lib.gold_tick(self.handle, c.byref(encoded), c.byref(result)):
            raise RuntimeError(self.lib.gold_error().decode())
        if result.fault:
            raise RuntimeError("gold-v25 native fault")
        return action_object(result.action)
