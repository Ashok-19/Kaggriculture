"""Fresh self-play tape generation for autonomous Bonsai discovery.

Generates diverse game tapes by running BKG4 against itself and policy
variants, producing the same tape format consumed by the existing CUDA
evaluation pipeline (``load_case_panel`` reads a TSV of tape/seat/seed).
"""
from __future__ import annotations

import copy
import hashlib
import json
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

_KAG_ENV = Path(__file__).resolve().parents[3] / "kaggle-environments-master"
_CHAMP = Path(__file__).resolve().parents[3] / "bonsai-kg" / "integrations" / "bkg4_champion"

SELFPLAY_SCHEMA_VERSION = 1
EPISODE_STEPS = 720


@dataclass(frozen=True)
class SelfplayPolicy:
    name: str
    kind: str  # "champion", "noisy", "greedy_market", "passive", "random_early"
    noise_rate: float = 0.0
    market_bias: float = 0.0
    passive_until: int = 0
    random_prefix: int = 0


DEFAULT_POLICIES = (
    SelfplayPolicy("bkg4_mirror", "champion"),
    SelfplayPolicy("bkg4_noisy_010", "noisy", noise_rate=0.10),
    SelfplayPolicy("bkg4_passive_72", "passive", passive_until=72),
)


def _ensure_paths():
    for _p in (_KAG_ENV, _CHAMP):
        s = str(_p)
        if s not in sys.path:
            sys.path.insert(0, s)


def _make_agent(policy: SelfplayPolicy, rng: random.Random):
    _ensure_paths()
    from runtime import BKG4ChampionAgent  # type: ignore[import]

    agent = BKG4ChampionAgent()
    if policy.kind == "champion":
        return agent

    def wrapped(obs: Dict[str, Any], config: Any = None) -> Dict[str, Any]:
        step = int(obs.get("step", 0) or 0)
        if policy.kind == "passive" and step < policy.passive_until:
            return {"farmer": ["PASS"], "hands": [], "market": []}
        if policy.kind == "random_early" and step < policy.random_prefix:
            action = agent(obs, config)
            if rng.random() < 0.5:
                action["market"] = []
            return action
        action = agent(obs, config)
        if policy.kind == "noisy" and rng.random() < policy.noise_rate:
            action["market"] = []
        if policy.kind == "greedy_market" and rng.random() < policy.market_bias:
            market = action.get("market", [])
            sell_ops = [op for op in market if isinstance(op, list) and op and op[0] == "SELL"]
            if sell_ops:
                action["market"] = sell_ops
        return action

    return wrapped


@dataclass(frozen=True)
class GeneratedTape:
    path: Path
    seat: int
    seed: int
    policy_name: str
    opponent_policy: str
    margin: float
    reward: float
    opponent_reward: float


def _tape_identity(seed: int, seat: int, policy: str, opponent: str) -> str:
    raw = f"selfplay-v1:{seed}:{seat}:{policy}:{opponent}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def generate_selfplay_tapes(
    output_dir: Path,
    *,
    seeds: Sequence[int],
    policies: Sequence[SelfplayPolicy] | None = None,
    seats: Sequence[int] = (0, 1),
    rng_seed: int = 0,
) -> list[GeneratedTape]:
    _ensure_paths()
    from kaggle_environments import make  # type: ignore[import]

    output_dir.mkdir(parents=True, exist_ok=True)
    if policies is None:
        policies = list(DEFAULT_POLICIES)
    rng = random.Random(rng_seed)
    tapes: list[GeneratedTape] = []
    for game_seed in seeds:
        for seat in seats:
            for opponent_policy in policies:
                candidate_rng = random.Random(rng.randint(0, 2**31))
                opponent_rng = random.Random(rng.randint(0, 2**31))
                champion_agent = _make_agent(
                    SelfplayPolicy("bkg4_candidate", "champion"), candidate_rng
                )
                opp_agent = _make_agent(opponent_policy, opponent_rng)
                candidate_actions: list[dict] = []

                def candidate_fn(obs, config=None):
                    a = champion_agent(obs, config)
                    candidate_actions.append(copy.deepcopy(a))
                    return a

                opp_actions: list[dict] = []

                def opponent_fn(obs, config=None):
                    a = opp_agent(obs, config)
                    opp_actions.append(copy.deepcopy(a))
                    return a

                env = make(
                    "kaggriculture",
                    configuration={"episodeSteps": EPISODE_STEPS, "seed": game_seed},
                    debug=False,
                )
                players = (
                    [candidate_fn, opponent_fn]
                    if seat == 0
                    else [opponent_fn, candidate_fn]
                )
                env.run(players)
                final = env.steps[-1]
                reward = float(final[seat].reward)
                opp_reward = float(final[1 - seat].reward)
                tape_id = _tape_identity(
                    game_seed, seat, "bkg4_candidate", opponent_policy.name
                )
                tape_actions = candidate_actions if seat == 0 else candidate_actions
                opp_tape_actions = opp_actions if seat == 0 else opp_actions
                for record_seat, actions, name in (
                    (seat, tape_actions, "bkg4_candidate"),
                    (1 - seat, opp_tape_actions, opponent_policy.name),
                ):
                    file_id = _tape_identity(
                        game_seed, record_seat, name, opponent_policy.name
                    )
                    tape_path = output_dir / f"selfplay_{file_id}.json"
                    if not tape_path.exists():
                        document = {
                            "source": {
                                "kind": "selfplay",
                                "schema_version": SELFPLAY_SCHEMA_VERSION,
                                "seed": game_seed,
                                "seat": record_seat,
                                "policy": name,
                                "opponent_policy": opponent_policy.name,
                                "reward": float(final[record_seat].reward),
                                "opponent_reward": float(
                                    final[1 - record_seat].reward
                                ),
                            },
                            "actions": actions,
                        }
                        tape_path.write_text(
                            json.dumps(document, ensure_ascii=False, separators=(",", ":")) + "\n",
                            encoding="utf-8",
                        )
                tape = GeneratedTape(
                    path=output_dir / f"selfplay_{tape_id}.json",
                    seat=seat,
                    seed=game_seed,
                    policy_name="bkg4_candidate",
                    opponent_policy=opponent_policy.name,
                    margin=reward - opp_reward,
                    reward=reward,
                    opponent_reward=opp_reward,
                )
                tapes.append(tape)
    return tapes


def write_target_panel(
    tapes: Sequence[GeneratedTape],
    panel_path: Path,
    *,
    family_id: int = 0,
    protected_flag: int = 0,
    own_history: int = 0,
    role: int = 0,
) -> int:
    """Write a 7-column TSV panel compatible with all CUDA binaries.

    Columns: tape_path, seat, seed, family, protected_flag, own_history, role.
    """
    panel_path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for tape in tapes:
        lines.append(
            f"{tape.path}\t{tape.seat}\t{tape.seed}"
            f"\t{family_id}\t{protected_flag}\t{own_history}\t{role}"
        )
    panel_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(lines)


def generate_diverse_seeds(
    count: int,
    *,
    rng_seed: int = 0,
    existing: Sequence[int] = (),
) -> list[int]:
    rng = random.Random(rng_seed)
    used = set(existing)
    seeds: list[int] = []
    while len(seeds) < count:
        s = rng.randint(1, 2**31 - 1)
        if s not in used:
            seeds.append(s)
            used.add(s)
    return seeds


def generate_selfplay_family(
    output_dir: Path,
    *,
    num_seeds: int = 20,
    policies: Sequence[SelfplayPolicy] | None = None,
    seats: Sequence[int] = (0,),
    rng_seed: int = 0,
) -> tuple[Path, dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    tapes_dir = output_dir / "tapes"
    panel_path = output_dir / "target_panel.tsv"
    seeds = generate_diverse_seeds(num_seeds, rng_seed=rng_seed)
    started = time.monotonic()
    tapes = generate_selfplay_tapes(
        tapes_dir,
        seeds=seeds,
        policies=policies,
        seats=seats,
        rng_seed=rng_seed,
    )
    elapsed = time.monotonic() - started
    count = write_target_panel(tapes, panel_path)
    margins = [t.margin for t in tapes]
    policy_counts: dict[str, int] = {}
    for t in tapes:
        policy_counts[t.opponent_policy] = policy_counts.get(t.opponent_policy, 0) + 1
    manifest = {
        "schema_version": SELFPLAY_SCHEMA_VERSION,
        "source": "selfplay",
        "num_seeds": num_seeds,
        "num_tapes": count,
        "policies": policy_counts,
        "generation_seconds": round(elapsed, 1),
        "margin_stats": {
            "mean": round(sum(margins) / max(1, len(margins)), 1),
            "min": round(min(margins), 1) if margins else 0,
            "max": round(max(margins), 1) if margins else 0,
        },
        "panel_path": str(panel_path),
        "tapes_dir": str(tapes_dir),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    return panel_path, manifest
