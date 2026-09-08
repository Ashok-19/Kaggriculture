"""Thin observation/serialization adapter to the native CPU/CUDA gold-v25 agent."""
import ctypes as c
import json
import shutil
import subprocess
from pathlib import Path

from kaggriculture_bonsai.runtime import bonsai_kg_v4 as bkg
from . import overlay as donor
from .extract import ASSETS, HERE, ROOT, sha

I = c.c_int32


class Unit(c.Structure):
    _fields_ = [(k, I) for k in ("op", "arg", "n", "arity")]


class Order(c.Structure):
    _fields_ = [(k, I) for k in ("op", "item", "n")]


class Tile(c.Structure):
    _fields_ = [(k, I) for k in ("kind", "animal", "watered", "yield_units", "fed", "fertilizer", "cared")]


class Worker(c.Structure):
    _fields_ = [(k, I) for k in ("x", "y", "inv_count")] + [("keys", I * 12), ("inv", I * 12), ("tile", Tile)]


class Observation(c.Structure):
    _fields_ = [(k, I) for k in ("step", "player", "board", "n_units", "yarn", "egg")] + [(k, I * 12) for k in ("shed", "seeds", "prices")] + [("workers", Worker * 40)]


class Action(c.Structure):
    _fields_ = [("n_units", I), ("n_market", I), ("units", Unit * 40), ("market", Order * 10)]


class Result(c.Structure):
    _fields_ = [("action", Action)] + [(k, I) for k in ("last_step", "route", "day6", "day27", "due")] + [("suppress", I * 12), ("fault", I), ("pending_counts", I * 40), ("pending_hash", c.c_uint64)]


def compile_native(directory=Path("/tmp/gold_v25_native"), cuda=True):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    includes = [ROOT / "native/bkg4/include", ROOT / "native/bkg4/stateful"]
    compiler = (shutil.which("nvcc") or "/usr/bin/nvcc") if cuda else "g++"
    output = directory / ("gold_cuda.so" if cuda else "gold_cpu.so")
    command = [compiler, "-std=c++17", "-O2", "-shared"]
    command += ["-arch=sm_86", "-Xcompiler", "-fPIC"] if cuda else ["-fPIC", "-x", "c++"]
    command += [f"-I{p}" for p in includes] + [str(HERE / "bridge.cu"), "-o", str(output)]
    subprocess.run(command, check=True)
    return output


def encode_observation(obs):
    out = Observation()
    out.step = donor._step_of(obs)
    out.player = donor._int(donor._get(obs, "player", 0))
    view = donor._View(obs, out.player, donor.DEFAULT_SETTINGS)
    out.board, out.n_units = view.board, len(view.positions)
    if out.n_units > 40 or out.n_units < 1:
        raise ValueError("native BKG4 unit limit")
    out.yarn = "YARN_STORE" in (donor._get(donor._get(obs, "town", {}), "unlocked_shops", []) or [])
    inv = donor._get(donor._get(obs, "market", {}), "inventory", {}) or {}
    out.egg = donor._int(donor._get(inv, "EGG", 10000))
    for k, name in enumerate(bkg.ITEMS):
        out.shed[k] = view.shed.get(name, 0)
        out.seeds[k] = donor._int(view.seeds.get(name, 0))
        out.prices[k] = view.prices.get(name, 0)
    for i, pos in enumerate(view.positions):
        w = out.workers[i]
        w.x, w.y = map(int, pos)
        inventory = view.inv(i)
        w.inv_count = len(inventory)
        for j, (name, qty) in enumerate(inventory.items()):
            k = bkg.ITEM_ID[name]
            w.keys[j], w.inv[k] = k, donor._int(qty)
        tile = donor._tile_at(view.tiles, pos)
        w.tile.kind = -1 if tile == "LOCKED" else 0
        if isinstance(tile, dict):
            w.tile.kind = {"WEED": 1, "PLANT": 2, "COOP": 3, "PASTURE": 4}.get(tile.get("kind"), 5)
            w.tile.animal = tile.get("animal") is not None
            w.tile.watered = bool(tile.get("watered_today"))
            w.tile.yield_units = donor._int(tile.get("yield_units", 0))
            w.tile.fed = bool(tile.get("fed_today"))
            w.tile.fertilizer = bool(tile.get("fertilizer_available"))
            w.tile.cared = bool(tile.get("cared_today"))
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
        market.append([] if o.op == 0 else [ORDER_NAMES[o.op]] if o.op in (1, 2) else [ORDER_NAMES[o.op], bkg.ITEMS[o.item], o.n])
    return {"farmer": units[0], "hands": units[1:], "market": market}


def pending_hash(pending):
    h = 14695981039346656037
    def word(value):
        nonlocal h
        for b in (value & 0xFFFFFFFF).to_bytes(4, "little"):
            h = ((h ^ b) * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    for i in range(40):
        queue = pending.get(i, [])
        word(len(queue))
        for (x, y), a in queue:
            for value in (x, y, bkg.UNIT_OPS[a[0]], bkg.ITEM_ID[a[1]] if len(a) >= 2 else 0, int(a[2]) if len(a) >= 3 else 1, len(a)):
                word(value)
    return h


def state_summary(st):
    router = st["router_state"]
    return {"last_step": st["last_step"], "route": st["route"], "day6": bool(router.get("day6")), "day27": bool(router.get("day27")), "due": st["sell_state"]["due_step"], "suppress": st["sell_state"]["suppress"], "pending_hash": pending_hash(st["pending"]), "pending": st["pending"], "fault": 0}


class NativeAgent:
    def __init__(self, library, gpu=False, assets=ASSETS):
        identity = json.loads((assets / "identity.json").read_text())
        for filename, expected in (("controller.bkgc", identity["controller_sha256"]), ("arity.bin", identity["arity_sha256"])):
            if sha((assets / filename).read_bytes()) != expected:
                raise ValueError("gold-v25 native asset hash mismatch")
        self.lib = c.CDLL(str(library))
        self.lib.gold_error.restype = c.c_char_p
        self.lib.gold_create.argtypes = [c.c_char_p] * 4 + [c.c_int]
        self.lib.gold_create.restype = c.c_void_p
        self.lib.gold_destroy.argtypes = [c.c_void_p]
        self.lib.gold_tick.argtypes = [c.c_void_p, c.POINTER(Observation), c.POINTER(Result)]
        self.lib.gold_pending.argtypes = [c.c_void_p, I, c.POINTER(I), I]
        if self.lib.gold_size(0) != c.sizeof(Observation) or self.lib.gold_size(1) != c.sizeof(Result):
            raise ValueError("gold-v25 bridge ABI mismatch")
        self.handle = self.lib.gold_create(str(assets / "controller.bkgc").encode(), str(assets / "routes.bkv6").encode(), identity["route_pack"]["pack_sha256"].encode(), str(assets / "arity.bin").encode(), gpu)
        if not self.handle:
            raise RuntimeError(self.lib.gold_error().decode())
        self.state = None

    def close(self):
        if self.handle:
            self.lib.gold_destroy(self.handle)
            self.handle = None

    def __call__(self, obs, configuration=None):
        encoded, out = encode_observation(obs), Result()
        if self.lib.gold_tick(self.handle, c.byref(encoded), c.byref(out)):
            raise RuntimeError(self.lib.gold_error().decode())
        self.state = {k: getattr(out, k) for k in ("last_step", "route", "day6", "day27", "due", "pending_hash", "fault")}
        self.state["suppress"] = {bkg.ITEMS[i]: n for i, n in enumerate(out.suppress) if n}
        pending = {}
        size = sum(out.pending_counts) * 6
        if size:
            buffer = (I * size)()
            if self.lib.gold_pending(self.handle, encoded.player, buffer, size) != size:
                raise RuntimeError(self.lib.gold_error().decode())
            offset = 0
            for i, count in enumerate(out.pending_counts):
                if count:
                    pending[i] = []
                for _ in range(count):
                    x, y, op, arg, n, arity = buffer[offset:offset + 6]
                    action = [UNIT_NAMES[op]]
                    if arity >= 2:
                        action.append(bkg.ITEMS[arg])
                    if arity >= 3:
                        action.append(n)
                    pending[i].append(((x, y), action))
                    offset += 6
        self.state["pending"] = pending
        if out.fault:
            raise RuntimeError(f"gold-v25 native fault {out.fault}")
        return action_object(out.action)


if __name__ == "__main__":
    print(compile_native(cuda=False))
    print(compile_native(cuda=True))
