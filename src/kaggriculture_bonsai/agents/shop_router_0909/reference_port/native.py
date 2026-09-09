"""Thin observation/serialization adapter to the native CPU/CUDA exact port."""
import ctypes as c
import json
import shutil
import subprocess
from pathlib import Path

from kaggriculture_bonsai.runtime import bonsai_kg_v4 as bkg

from . import agent as donor
from .extract import ASSETS, HERE, ROOT, sha
from .pack import ROUTES, STEPS

I = c.c_int32
ITEMS, UNITS, TAPE_UNITS, MAX_ORDERS = 12, 40, 13, 10
BUILD = Path("/tmp/shop_router_0909_native")


class Unit(c.Structure):
    _fields_ = [(k, I) for k in ("op", "arg", "n", "arity")]


class Order(c.Structure):
    _fields_ = [(k, I) for k in ("op", "item", "n")]


class Worker(c.Structure):
    _fields_ = [(k, I) for k in ("x", "y", "weed", "inv_count")] + [("keys", I * ITEMS), ("inv", I * ITEMS)]


class Observation(c.Structure):
    _fields_ = ([(k, I) for k in ("step", "player", "board", "n_positions", "shop0", "shop1")]
                + [(k, I * ITEMS) for k in ("shed", "shed_present", "prices")]
                + [("workers", Worker * UNITS)])


class Action(c.Structure):
    _fields_ = [("n_units", I), ("n_market", I), ("units", Unit * TAPE_UNITS), ("market", Order * MAX_ORDERS)]


class Result(c.Structure):
    _fields_ = ([("action", Action)] + [(k, I) for k in ("last_step", "route", "day", "due", "endgame", "fault")]
                + [("advanced", I * ITEMS), ("queue_counts", I * UNITS), ("queue_hash", c.c_uint64)])


def compile_native(directory=BUILD, cuda=True):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    includes = [ROOT / "native/bkg4/include", ROOT / "native/bkg4/stateful"]
    compiler = (shutil.which("nvcc") or "/usr/bin/nvcc") if cuda else "g++"
    output = directory / ("shop_cuda.so" if cuda else "shop_cpu.so")
    command = [compiler, "-std=c++17", "-O2", "-shared"]
    command += ["-arch=sm_86", "-Xcompiler", "-fPIC"] if cuda else ["-fPIC", "-x", "c++"]
    command += [f"-I{p}" for p in includes] + [str(HERE / "bridge.cu"), "-o", str(output)]
    subprocess.run(command, check=True)
    return output


def encode_observation(obs):
    out = Observation()
    out.step, out.player = int(obs["step"]), int(obs["player"])
    view = donor.FarmView(obs)
    out.board = len(view.tiles)
    out.n_positions = len(view.positions)
    if not 1 <= out.n_positions <= TAPE_UNITS:
        raise ValueError("native port unit limit")
    shops = obs["town"]["unlocked_shops"]
    out.shop0 = bkg.SHOP_ID.get(str(shops[0]), -1) if len(shops) > 0 else -1
    out.shop1 = bkg.SHOP_ID.get(str(shops[1]), -1) if len(shops) > 1 else -1
    for name, qty in view.shed.items():
        if name not in bkg.ITEM_ID:
            raise ValueError(f"shed item outside the native domain: {name}")
        out.shed[bkg.ITEM_ID[name]], out.shed_present[bkg.ITEM_ID[name]] = int(qty), 1
    for name, price in view.prices.items():
        if name in bkg.ITEM_ID:
            out.prices[bkg.ITEM_ID[name]] = int(price)
    for i, position in enumerate(view.positions):
        w = out.workers[i]
        w.x, w.y = int(position[0]), int(position[1])
        tile = view.tiles[w.y][w.x]
        w.weed = int(isinstance(tile, dict) and tile.get("kind") == "WEED")
        inventory = view.inventory(i)
        w.inv_count = len(inventory)
        for j, (name, qty) in enumerate(inventory.items()):
            k = bkg.ITEM_ID[name]
            w.keys[j], w.inv[k] = k, int(qty)
    return out


UNIT_NAMES = {v: k for k, v in bkg.UNIT_OPS.items()}
ORDER_NAMES = {v: k for k, v in bkg.MARKET_OPS.items()}


def action_object(action):
    def unit(u):
        values = [UNIT_NAMES[u.op]]
        if u.arity >= 2:
            values.append(bkg.ITEMS[u.arg])
        if u.arity >= 3:
            values.append(u.n)
        return values
    units = [unit(u) for u in action.units[:action.n_units]]
    market = []
    for o in action.market[:action.n_market]:
        market.append([] if o.op == 0 else
                      [ORDER_NAMES[o.op]] if o.op in (bkg.MARKET_OPS["HIRE"], bkg.MARKET_OPS["BUY_LAND"]) else
                      [ORDER_NAMES[o.op], bkg.ITEMS[o.item], o.n])
    return {"farmer": units[0], "hands": units[1:], "market": market}


def queue_hash(queues):
    h = 14695981039346656037

    def word(value):
        nonlocal h
        for b in (value & 0xFFFFFFFF).to_bytes(4, "little"):
            h = ((h ^ b) * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    for i in range(UNITS):
        entries = queues.get(i, [])
        word(len(entries))
        for a in entries:
            arity = len(a)
            word(bkg.UNIT_OPS[a[0]])
            word(bkg.ITEM_ID[a[1]] if arity >= 2 else 0)
            word(int(a[2]) if arity >= 3 else 1)
            word(arity)
    return h


def state_summary(state):
    """Donor state, in the native Result's vocabulary."""
    return {"last_step": state.last_step, "route": state.plan, "day": state.day,
            "due": state.sale_due_step, "endgame": getattr(state, "endgame", 0),
            "advanced": {k: v for k, v in state.advanced_sales.items() if v},
            "queues": {i: list(q) for i, q in state.queues.items() if q},
            "queue_hash": queue_hash({i: list(q) for i, q in state.queues.items()}),
            "fault": 0}


class NativeAgent:
    def __init__(self, library, gpu=False, assets=ASSETS, controller=None, controller_sha256=None):
        """`controller` swaps in a candidate policy over the same frozen routes."""
        identity = json.loads((assets / "identity.json").read_text())
        if sha((assets / "arity.bin").read_bytes()) != identity["arity_sha256"]:
            raise ValueError("shop-router-0909 native asset hash mismatch")
        policy = Path(controller) if controller else assets / "controller.bkgc"
        expected = controller_sha256 if controller else identity["controller_sha256"]
        if expected is not None and sha(policy.read_bytes()) != expected:
            raise ValueError("shop-router-0909 native asset hash mismatch")
        self.lib = c.CDLL(str(library))
        self.lib.shop_error.restype = c.c_char_p
        self.lib.shop_create.argtypes = [c.c_char_p] * 4 + [c.c_int]
        self.lib.shop_create.restype = c.c_void_p
        self.lib.shop_destroy.argtypes = [c.c_void_p]
        self.lib.shop_tick.argtypes = [c.c_void_p, c.POINTER(Observation), c.POINTER(Result)]
        self.lib.shop_queues.argtypes = [c.c_void_p, I, c.POINTER(I), I]
        if self.lib.shop_size(0) != c.sizeof(Observation) or self.lib.shop_size(1) != c.sizeof(Result):
            raise ValueError("shop-router-0909 bridge ABI mismatch")
        self.handle = self.lib.shop_create(
            str(policy).encode(), str(assets / "routes.bkv6").encode(),
            identity["route_pack"]["pack_sha256"].encode(), str(assets / "arity.bin").encode(), gpu)
        if not self.handle:
            raise RuntimeError(self.lib.shop_error().decode())
        self.state = None

    def close(self):
        if self.handle:
            self.lib.shop_destroy(self.handle)
            self.handle = None

    def __call__(self, obs, configuration=None):
        encoded, out = encode_observation(obs), Result()
        if self.lib.shop_tick(self.handle, c.byref(encoded), c.byref(out)):
            raise RuntimeError(self.lib.shop_error().decode())
        queues = {}
        size = sum(out.queue_counts) * 4
        if size:
            buffer = (I * size)()
            if self.lib.shop_queues(self.handle, encoded.player, buffer, size) != size:
                raise RuntimeError(self.lib.shop_error().decode())
            offset = 0
            for i, count in enumerate(out.queue_counts):
                if count:
                    queues[i] = []
                for _ in range(count):
                    op, arg, n, arity = buffer[offset:offset + 4]
                    entry = [UNIT_NAMES[op]]
                    if arity >= 2:
                        entry.append(bkg.ITEMS[arg])
                    if arity >= 3:
                        entry.append(n)
                    queues[i].append(entry)
                    offset += 4
        self.state = {"last_step": out.last_step, "route": out.route, "day": out.day, "due": out.due,
                      "endgame": out.endgame,
                      "advanced": {bkg.ITEMS[i]: n for i, n in enumerate(out.advanced) if n},
                      "queues": queues, "queue_hash": out.queue_hash, "fault": out.fault}
        if out.fault:
            raise RuntimeError(f"shop-router-0909 native fault {out.fault}")
        return action_object(out.action)


if __name__ == "__main__":
    print(compile_native(cuda=False))
    print(compile_native(cuda=True))
