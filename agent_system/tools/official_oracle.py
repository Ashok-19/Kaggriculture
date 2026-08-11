#!/usr/bin/env python3
"""Pinned minimal runner around Kaggle's official Kaggriculture interpreter.

This deliberately uses the official Python interpreter as the correctness oracle
instead of re-implementing stochastic town/weed semantics in the search core.
"""
from __future__ import annotations

import copy
import types
import urllib.request
from pathlib import Path

PIN = "bded87b0d7879078c726a93a4884d044f79c4eed"
BASE = f"https://raw.githubusercontent.com/Kaggle/kaggle-environments/{PIN}/kaggle_environments/envs/kaggriculture"
CACHE = Path("agent_system/.cache/official")


class Attr(dict):
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    __setattr__ = dict.__setitem__


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


def ensure_official_source() -> tuple[Path, Path]:
    CACHE.mkdir(parents=True, exist_ok=True)
    py = CACHE / "kaggriculture.py"
    js = CACHE / "kaggriculture.json"
    if not py.exists():
        py.write_bytes(urllib.request.urlopen(f"{BASE}/kaggriculture.py", timeout=30).read())
    if not js.exists():
        js.write_bytes(urllib.request.urlopen(f"{BASE}/kaggriculture.json", timeout=30).read())
    return py, js


def load_engine():
    py, js = ensure_official_source()
    source = py.read_text()
    source = source.replace(
        "from kaggle_environments.utils import resolve_episode_seed",
        '''def resolve_episode_seed(env):\n    seed = env.configuration.get("seed", None)\n    if seed is None:\n        seed = env.info.get("seed", 0)\n    seed = int(seed)\n    env.info["seed"] = seed\n    env.configuration["seed"] = None\n    return seed''',
    )
    # The official module loads kaggriculture.json relative to __file__.
    expected_json = py.with_name("kaggriculture.json")
    if expected_json != js and not expected_json.exists():
        expected_json.write_bytes(js.read_bytes())
    mod = types.ModuleType("official_kaggriculture")
    mod.__file__ = str(py.resolve())
    exec(compile(source, mod.__file__, "exec"), mod.__dict__)
    return mod


class MiniEnv:
    def __init__(self, seed: int, config=None):
        self.configuration = Attr(DEFAULT_CONFIG.copy())
        if config:
            self.configuration.update(config)
        self.configuration["seed"] = int(seed)
        self.info = {"seed": int(seed)}
        self.done = False


class MiniState:
    def __init__(self, player: int):
        self.observation = Attr({"step": 0, "player": player})
        self.action = {}
        self.reward = 0.0
        self.status = "ACTIVE"


def run(agent0, agent1, seed: int, config=None, capture: bool = False):
    engine = load_engine()
    env = MiniEnv(seed, config)
    state = [MiniState(0), MiniState(1)]
    engine.interpreter(state, env)
    frames = []
    for step in range(int(env.configuration.episodeSteps) - 1):
        for i in (0, 1):
            state[i].observation.step = step
        if capture:
            frames.append(copy.deepcopy(state))
        cfg = Attr(env.configuration.copy())
        actions = []
        for i, agent in enumerate((agent0, agent1)):
            actions.append(agent(copy.deepcopy(state[i].observation), cfg))
        state[0].action, state[1].action = actions
        engine.interpreter(state, env)
        if all(s.status == "DONE" for s in state):
            env.done = True
            break
    rewards = [
        float(s.reward) if s.status == "DONE" else float(state[0].observation.farms[i]["money"])
        for i, s in enumerate(state)
    ]
    return rewards, frames, state
