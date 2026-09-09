"""Self-contained CPU runtime adapter for SHOP_ROUTER_0909_PLUS."""
import ctypes as c
import hashlib
import json
from pathlib import Path

ITEMS = ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL", "FERTILIZER", "GOOSE", "COW", "SHEEP")
ITEM_ID = {name: i for i, name in enumerate(ITEMS)}
SHOPS = {"BAKERY": 0, "BAKERY_SHOP": 0, "BRUNCH": 1, "BRUNCH_SPOT": 1, "FARMERS_MARKET": 2, "ICE_CREAM": 3, "ICE_CREAM_SHOP": 3, "PET_CAFE": 4, "PIZZA": 5, "PIZZA_SHOP": 5, "SMOOTHIE": 6, "SMOOTHIE_SHOP": 6, "YARN": 7, "YARN_STORE": 7}
UNIT_NAMES = {i: name for i, name in enumerate(("PASS", "NORTH", "SOUTH", "EAST", "WEST", "PICKUP", "DROP", "PLACE", "PLANT", "WATER", "HARVEST", "FERTILIZE", "DIG", "BUILD_COOP", "BUILD_PASTURE", "FEED", "COLLECT_FERTILIZER", "CARE"))}
ORDER_NAMES = {0: "NONE", 1: "HIRE", 2: "BUY_LAND", 3: "BUY_SEED", 4: "BUY_PRODUCT", 5: "BUY_ANIMAL", 6: "SELL"}
I = c.c_int32


class Unit(c.Structure):
    _fields_ = [(name, I) for name in ("op", "arg", "n", "arity")]


class Order(c.Structure):
    _fields_ = [(name, I) for name in ("op", "item", "n")]


class Worker(c.Structure):
    _fields_ = [(name, I) for name in ("x", "y", "weed", "inv_count")] + [("keys", I * 12), ("inv", I * 12)]


class Observation(c.Structure):
    _fields_ = [(name, I) for name in ("step", "player", "board", "n_positions", "shop0", "shop1")] + [(name, I * 12) for name in ("shed", "shed_present", "prices")] + [("workers", Worker * 40)]


class Action(c.Structure):
    _fields_ = [("n_units", I), ("n_market", I), ("units", Unit * 13), ("market", Order * 10)]


class Result(c.Structure):
    _fields_ = [("action", Action)] + [(name, I) for name in ("last_step", "route", "day", "due", "endgame", "fault")] + [("advanced", I * 12), ("queue_counts", I * 40), ("queue_hash", c.c_uint64)]


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


def _tile(tiles, position):
    try:
        return tiles[int(position[1])][int(position[0])]
    except (TypeError, ValueError, IndexError):
        return None


def encode_observation(observation):
    out = Observation()
    out.step, out.player = _int(_get(observation, "step")), _int(_get(observation, "player"))
    farms = list(_get(observation, "farms", []) or [])
    farm = farms[out.player] if 0 <= out.player < len(farms) else {}
    tiles = _get(farm, "tiles", []) or []
    out.board = len(tiles) or 10
    positions = [_get(farm, "farmer", None)] + list(_get(farm, "hands", []) or [])
    out.n_positions = len(positions)
    if not 1 <= out.n_positions <= 13:
        raise ValueError("shop-router native unit limit")
    shops = list(_get(_get(observation, "town", {}) or {}, "unlocked_shops", []) or [])
    out.shop0 = SHOPS.get(str(shops[0]), -1) if len(shops) > 0 else -1
    out.shop1 = SHOPS.get(str(shops[1]), -1) if len(shops) > 1 else -1
    private = _get(observation, "private", {}) or {}
    shed = _get(private, "shed", {}) or {}
    prices = _get(_get(observation, "market", {}) or {}, "prices", {}) or {}
    inventories = list(_get(private, "inventories", []) or [])
    for name, quantity in dict(shed).items():
        if name not in ITEM_ID:
            raise ValueError("unknown shed item")
        k = ITEM_ID[name]
        out.shed[k], out.shed_present[k] = max(0, _int(quantity)), 1
    for name, price in dict(prices).items():
        if name in ITEM_ID:
            out.prices[ITEM_ID[name]] = _int(price)
    for i, position in enumerate(positions):
        if not isinstance(position, (list, tuple)) or len(position) < 2:
            raise ValueError("invalid worker position")
        worker = out.workers[i]
        worker.x, worker.y = _int(position[0]), _int(position[1])
        tile = _tile(tiles, position)
        worker.weed = int(isinstance(tile, dict) and _get(tile, "kind") == "WEED")
        inventory = dict(inventories[i] or {}) if i < len(inventories) else {}
        worker.inv_count = len(inventory)
        for j, (name, quantity) in enumerate(inventory.items()):
            if j >= 12 or name not in ITEM_ID:
                raise ValueError("invalid worker inventory")
            k = ITEM_ID[name]
            worker.keys[j], worker.inv[k] = k, _int(quantity)
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
    for order in action.market[:action.n_market]:
        if order.op == 0:
            market.append([])
        elif order.op in (1, 2):
            market.append([ORDER_NAMES[order.op]])
        else:
            market.append([ORDER_NAMES[order.op], ITEMS[order.item], order.n])
    return {"farmer": units[0], "hands": units[1:], "market": market}


class GoldRuntime:
    def __init__(self, root=None):
        root = Path(root) if root else Path(__file__).resolve().parent
        assets = root / "assets"
        identity = json.loads((assets / "identity.json").read_text())
        for filename, expected in (("controller.bkgc", identity["controller_sha256"]), ("routes.bkv6", identity["route_pack_sha256"]), ("arity.bin", identity["arity_sha256"])):
            if hashlib.sha256((assets / filename).read_bytes()).hexdigest() != expected:
                raise ValueError("shop-router asset hash mismatch")
        self.lib = c.CDLL(str(root / "shop_cpu.so"))
        self.lib.shop_error.restype = c.c_char_p
        self.lib.shop_create.argtypes = [c.c_char_p] * 4 + [I]
        self.lib.shop_create.restype = c.c_void_p
        self.lib.shop_destroy.argtypes = [c.c_void_p]
        self.lib.shop_tick.argtypes = [c.c_void_p, c.POINTER(Observation), c.POINTER(Result)]
        if self.lib.shop_size(0) != c.sizeof(Observation) or self.lib.shop_size(1) != c.sizeof(Result):
            raise ValueError("shop-router bridge ABI mismatch")
        self.handle = self.lib.shop_create(str(assets / "controller.bkgc").encode(), str(assets / "routes.bkv6").encode(), identity["route_pack_sha256"].encode(), str(assets / "arity.bin").encode(), 0)
        if not self.handle:
            raise RuntimeError(self.lib.shop_error().decode())

    def __call__(self, observation, configuration=None):
        encoded, result = encode_observation(observation), Result()
        if self.lib.shop_tick(self.handle, c.byref(encoded), c.byref(result)):
            raise RuntimeError(self.lib.shop_error().decode())
        if result.fault:
            raise RuntimeError("shop-router native fault")
        return action_object(result.action)
