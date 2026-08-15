#!/usr/bin/env python3
"""Fast local runner using the parity-validated Kaggriculture C++ engine.

The C++ physics targets kaggle-environments 1.32.7. Python agents are still
called normally each turn; only environment state transitions/observation
construction move to C++.
"""
from __future__ import annotations

import copy
import importlib.util
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BUILD = ROOT / "scratch" / "cpp-engine-build"
BRIDGE_SRC = ROOT / "cpp-engine" / "src" / "pybridge.cpp"
SIM_SRC = ROOT / "cpp-engine" / "include" / "sim.hpp"
RNG_SRC = ROOT / "cpp-engine" / "include" / "pyrandom.hpp"
BUILD_SCRIPT = ROOT / "cpp-engine" / "tools" / "build_pybridge.py"


class Attr(dict):
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    __setattr__ = dict.__setitem__


def _structify(value):
    """Match the official oracle's container shape.

    Only the top-level observation supports attribute access. Nested farms,
    market, town, private state, tiles, and inventories remain ordinary Python
    dict/list objects in the official interpreter; recursively wrapping them in
    Attr makes deepcopy dramatically more expensive without adding compatibility.
    """
    if isinstance(value, dict):
        return Attr(value)
    return value


DEFAULT_CONFIG = Attr({
    "episodeSteps": 720,
    "actTimeout": 1,
    "boardSize": 10,
    "startingMoney": 3000,
    "maxMarketOrdersPerTurn": 10,
    "turnsPerDay": 24,
    "shedCapacity": 100,
    "weedSpawnChance": 0.005,
    "townShopUnlockInterval": 3,
    "townShopSellInterval": 4,
    "townCenterSellInterval": 24,
    "seed": None,
    "farmHandCostMult": 1,
    "marketParams": {},
})


def _extension_path() -> Path | None:
    candidates = list(BUILD.glob("kaggriculture_cpp*.so"))
    if not candidates:
        return None
    tag = f"cpython-{sys.version_info.major}{sys.version_info.minor}"
    exact = [p for p in candidates if tag in p.name]
    return exact[0] if exact else None


def _ensure_bridge() -> Path:
    ext = _extension_path()
    newest_source = max(p.stat().st_mtime for p in (BRIDGE_SRC, SIM_SRC, RNG_SRC, BUILD_SCRIPT))
    if ext is not None and ext.stat().st_mtime >= newest_source:
        return ext
    pyver = f"{sys.version_info.major}.{sys.version_info.minor}"
    subprocess.run([
        "uv", "run", "--python", pyver, "--with", "pybind11", "python", str(BUILD_SCRIPT)
    ], cwd=ROOT, check=True)
    ext = _extension_path()
    if ext is None:
        raise RuntimeError(f"C++ bridge build did not produce an extension for Python {pyver}")
    return ext


def ensure_bridge() -> Path:
    """Build/verify the native extension once before multiprocessing spawn."""
    return _ensure_bridge()


def _load_bridge():
    path = _ensure_bridge()
    spec = importlib.util.spec_from_file_location("kaggriculture_cpp", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load C++ bridge {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class MiniState:
    def __init__(self, observation, reward=0.0, status="ACTIVE"):
        self.observation = observation
        self.action = {}
        self.reward = reward
        self.status = status


def run(agent0, agent1, seed: int, config=None, capture: bool = False):
    bridge = _load_bridge()
    cfg = Attr(DEFAULT_CONFIG.copy())
    if config:
        cfg.update(config)
    cfg["seed"] = None
    game_config = dict(cfg)
    game_config.pop("seed", None)
    game = bridge.Game(int(seed), game_config)
    frames = []

    while not game.done:
        obs = [_structify(game.observation(0)), _structify(game.observation(1))]
        if capture:
            frames.append([MiniState(copy.deepcopy(obs[0])), MiniState(copy.deepcopy(obs[1]))])
        # bridge.Game.observation() materializes a fresh Python tree detached
        # from the native Sim state. Passing it directly is therefore safe even
        # if an agent mutates its observation, and avoids the expensive defensive
        # deepcopy required by official_oracle's shared interpreter state.
        actions = [
            agent0(obs[0], Attr(cfg.copy())),
            agent1(obs[1], Attr(cfg.copy())),
        ]
        game.step(actions[0], actions[1])

    rewards = [float(x) for x in game.rewards()]
    final_obs = [_structify(game.observation(0)), _structify(game.observation(1))]
    state = [
        MiniState(final_obs[0], rewards[0], "DONE"),
        MiniState(final_obs[1], rewards[1], "DONE"),
    ]
    return rewards, frames, state
