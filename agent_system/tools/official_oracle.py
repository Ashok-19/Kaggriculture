#!/usr/bin/env python3
"""Pinned minimal runner around Kaggle's official Kaggriculture interpreter.

This deliberately uses the official Python interpreter as the correctness oracle
instead of re-implementing stochastic town/weed semantics in the search core.
The pin below is the PR #1399 merge shipped by kaggle-environments 1.32.7.
"""
from __future__ import annotations

import copy
import hashlib
import types
import urllib.request
from pathlib import Path

PIN = "28b6d8af3ce73926b3d0fda1410c1ddd8384ab8c"
BASE = f"https://raw.githubusercontent.com/Kaggle/kaggle-environments/{PIN}/kaggle_environments/envs/kaggriculture"
CACHE = Path("agent_system/.cache/official")
SOURCE_SHA256 = {
    "kaggriculture.py": "bc8a54879ef02c7ea64b8b333d6a976f0ea65c4949149d01f463f23bccee653e",
    "kaggriculture.json": "a82c89c1a2315b93f39775d8e025471a01b738647c9772658368ee6b1b6f4867",
}


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


def _ensure_pinned_file(path: Path) -> None:
    expected = SOURCE_SHA256[path.name]
    if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() == expected:
        return
    data = urllib.request.urlopen(f"{BASE}/{path.name}", timeout=30).read()
    actual = hashlib.sha256(data).hexdigest()
    if actual != expected:
        raise RuntimeError(
            f"official source hash mismatch for {path.name}: {actual} != {expected}"
        )
    path.write_bytes(data)


def ensure_official_source() -> tuple[Path, Path]:
    CACHE.mkdir(parents=True, exist_ok=True)
    py = CACHE / "kaggriculture.py"
    js = CACHE / "kaggriculture.json"
    _ensure_pinned_file(py)
    _ensure_pinned_file(js)
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
