#!/usr/bin/env python3
"""Fast local runner using the parity-validated Kaggriculture C++ engine.

The C++ physics targets kaggle-environments 1.32.7. Python agents are still
called normally each turn; only environment state transitions/observation
construction move to C++.
"""
from __future__ import annotations

import copy
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BUILD = ROOT / "scratch" / "cpp-engine-build"
BRIDGE_SRC = ROOT / "cpp-engine" / "src" / "pybridge.cpp"
SIM_SRC = ROOT / "cpp-engine" / "include" / "sim.hpp"
RNG_SRC = ROOT / "cpp-engine" / "include" / "pyrandom.hpp"
BUILD_SCRIPT = ROOT / "cpp-engine" / "tools" / "build_pybridge.py"
_BRIDGE = None
_COMPILED_TAPE_CACHE = {}


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
        "uv", "run", "--python", pyver, "--with", "pybind11==3.1.0", "python", str(BUILD_SCRIPT)
    ], cwd=ROOT, check=True)
    ext = _extension_path()
    if ext is None:
        raise RuntimeError(f"C++ bridge build did not produce an extension for Python {pyver}")
    return ext


def ensure_bridge() -> Path:
    """Build/verify the native extension once before multiprocessing spawn."""
    return _ensure_bridge()


def _load_bridge():
    global _BRIDGE
    if _BRIDGE is not None:
        return _BRIDGE
    path = _ensure_bridge()
    spec = importlib.util.spec_from_file_location("kaggriculture_cpp", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load C++ bridge {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _COMPILED_TAPE_CACHE.clear()
    _BRIDGE = mod
    return mod


def load_tape_actions(tape):
    """Return the raw 719-step action list from a replay tape path/object."""
    if isinstance(tape, (str, Path)):
        path = Path(tape)
        if path.suffix == ".py":
            path = path.with_suffix(".json")
        payload = json.loads(path.read_text())
    else:
        payload = tape
    if isinstance(payload, dict):
        payload = payload.get("actions")
    if not isinstance(payload, list) or not payload:
        raise ValueError("tape must contain a non-empty actions list")
    return payload


class MiniState:
    def __init__(self, observation, reward=0.0, status="ACTIVE"):
        self.observation = observation
        self.action = {}
        self.reward = reward
        self.status = status


def compile_tape(tape):
    """Compile a replay tape into native Action structs, caching file-backed tapes."""
    bridge = _load_bridge()
    if isinstance(tape, (str, Path)):
        path = Path(tape)
        if path.suffix == ".py":
            path = path.with_suffix(".json")
        path = path.resolve()
        st = path.stat()
        key = (str(path), st.st_mtime_ns, st.st_size)
        compiled = _COMPILED_TAPE_CACHE.get(key)
        if compiled is None:
            compiled = bridge.CompiledTape(load_tape_actions(path))
            _COMPILED_TAPE_CACHE[key] = compiled
        return compiled
    return bridge.CompiledTape(load_tape_actions(tape))


def run_tapes_batch(tape0, tape1, seeds, config=None, workers: int = 1):
    """Pure-C++ fixed-tape cohort runner; workers is hard-limited to 1 or 2."""
    if workers not in (1, 2):
        raise ValueError("workers must be 1 or 2 on this local workstation")
    seed_values = [int(seed) for seed in seeds]
    if not seed_values:
        return []
    bridge = _load_bridge()
    compiled0 = compile_tape(tape0)
    compiled1 = compile_tape(tape1)
    game_config = dict(config or {})
    game_config.pop("seed", None)

    return [[float(x) for x in row]
            for row in bridge.run_compiled_tapes_batch(
                seed_values, compiled0, compiled1, game_config, workers)]


def snapshot_vs_tape(agent, tape, seed: int, candidate_player: int, stop_step: int,
                     config=None):
    """Run a Python candidate prefix against a native replay tape and clone C++ state."""
    if candidate_player not in (0, 1):
        raise ValueError("candidate_player must be 0 or 1")
    if stop_step < 0 or stop_step > int(DEFAULT_CONFIG["episodeSteps"]) - 1:
        raise ValueError("stop_step must be within the playable 0..719 prefix range")
    bridge = _load_bridge()
    cfg = Attr(DEFAULT_CONFIG.copy())
    if config:
        cfg.update(config)
    cfg["seed"] = None
    game_config = dict(cfg)
    game_config.pop("seed", None)
    game = bridge.Game(int(seed), game_config)
    game.load_compiled_tape(compile_tape(tape), 1 - candidate_player)
    for _ in range(stop_step):
        if game.done:
            break
        obs = _structify(game.observation(candidate_player))
        action = agent(obs, Attr(cfg.copy()))
        game.step_with_tape(action)
    return game.snapshot()


def branch_tape_candidates(snapshot, candidate_tapes, opponent_tape, candidate_player: int,
                           workers: int = 1):
    """Evaluate compiled tape continuations from one exact C++ snapshot."""
    if candidate_player not in (0, 1):
        raise ValueError("candidate_player must be 0 or 1")
    if workers not in (1, 2):
        raise ValueError("workers must be 1 or 2 on this local workstation")
    bridge = _load_bridge()
    compiled = [compile_tape(tape) for tape in candidate_tapes]
    opponent = compile_tape(opponent_tape)
    return [[float(x) for x in row]
            for row in bridge.run_snapshot_candidates(
                snapshot, compiled, opponent, candidate_player, workers)]


def run(agent0, agent1, seed: int, config=None, capture: bool = False,
        shared_public: bool = False):
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
        if shared_public:
            pair = game.observations_shared()
            obs = [_structify(pair[0]), _structify(pair[1])]
        else:
            obs = [_structify(game.observation(0)), _structify(game.observation(1))]
        if capture:
            frames.append([MiniState(copy.deepcopy(obs[0])), MiniState(copy.deepcopy(obs[1]))])
        # bridge.Game.observation() materializes a fresh Python tree detached
        # from the native Sim state. Passing it directly is therefore safe even
        # if an agent mutates its observation, and avoids the expensive defensive
        # deepcopy required by official_oracle's shared interpreter state.
        # Match the official oracle: one fresh configuration mapping per step,
        # shared by the two agent calls.
        step_cfg = Attr(cfg.copy())
        actions = [
            agent0(obs[0], step_cfg),
            agent1(obs[1], step_cfg),
        ]
        game.step(actions[0], actions[1])

    rewards = [float(x) for x in game.rewards()]
    final_obs = [_structify(game.observation(0)), _structify(game.observation(1))]
    state = [
        MiniState(final_obs[0], rewards[0], "DONE"),
        MiniState(final_obs[1], rewards[1], "DONE"),
    ]
    return rewards, frames, state


def run_vs_tape(agent, tape, seed: int, candidate_player: int, config=None, capture: bool = False):
    """Run one Python agent against a replay tape with the tape side native in C++.

    Recorded hand actions are padded/truncated inside C++ to the current tape
    player's farm-hand count, matching the existing Python replay wrappers.
    Only the candidate observation is materialized during the episode.
    """
    if candidate_player not in (0, 1):
        raise ValueError("candidate_player must be 0 or 1")
    bridge = _load_bridge()
    cfg = Attr(DEFAULT_CONFIG.copy())
    if config:
        cfg.update(config)
    cfg["seed"] = None
    game_config = dict(cfg)
    game_config.pop("seed", None)
    game = bridge.Game(int(seed), game_config)
    compiled = compile_tape(tape)
    game.load_compiled_tape(compiled, 1 - candidate_player)
    frames = []

    while not game.done:
        obs = _structify(game.observation(candidate_player))
        if capture:
            frames.append(MiniState(copy.deepcopy(obs)))
        action = agent(obs, Attr(cfg.copy()))
        game.step_with_tape(action)

    rewards = [float(x) for x in game.rewards()]
    final_obs = [_structify(game.observation(0)), _structify(game.observation(1))]
    state = [
        MiniState(final_obs[0], rewards[0], "DONE"),
        MiniState(final_obs[1], rewards[1], "DONE"),
    ]
    return rewards, frames, state
