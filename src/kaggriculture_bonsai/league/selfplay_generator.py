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
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

SELFPLAY_SCHEMA_VERSION = 2
EPISODE_STEPS = 720


@dataclass(frozen=True)
class SelfplayPolicy:
    name: str
    kind: str
    family: str = "baseline"
    noise_rate: float = 0.0
    market_bias: float = 0.0
    passive_until: int = 0
    random_prefix: int = 0
    land_target: int = 0
    crop: str = "WHEAT"
    animal: str = ""
    checkpoint_asset_dir: str = ""
    strategic_lane: str = "GENERAL"
    generation: int = 0
    parent_checkpoint: str = ""
    member_type: str = "hand_designed"
    exploiter_program: str = ""
    mutation_origin: str = ""


DEFAULT_POLICIES = (
    # Retain the proven baseline, but give each retained tape an explicit regime.
    SelfplayPolicy("bkg4_mirror", "champion", family="baseline"),
    SelfplayPolicy("bkg4_noisy_010", "noisy", family="bkg4_mutation", noise_rate=0.10),
    SelfplayPolicy("bkg4_passive_72", "passive", family="capital_timing", passive_until=72),
    # Deliberate economic regimes. These are not named opponents and never read HOLDOUT data.
    SelfplayPolicy("bkg4_market_squeeze", "market_squeeze", family="market_liquidity"),
    SelfplayPolicy("bkg4_cash_hoard_144", "cash_hoarder", family="capital_timing", passive_until=144),
    SelfplayPolicy("bkg4_early_land", "early_expansion", family="land_investment", land_target=4, passive_until=288),
    SelfplayPolicy("bkg4_late_land_288", "late_expansion", family="land_investment", land_target=4, passive_until=288),
    SelfplayPolicy("bkg4_farm_strawberry", "farm_heavy", family="farming", crop="STRAWBERRY"),
    SelfplayPolicy("bkg4_animal_cow", "animal_heavy", family="animals", animal="COW"),
    SelfplayPolicy("bkg4_inventory_hoard", "inventory_hoarder", family="inventory"),
    SelfplayPolicy("bkg4_liquidator_432", "liquidator", family="inventory", passive_until=432),
    SelfplayPolicy("bkg4_mutated_025", "mutated_bkg4", family="bkg4_mutation", noise_rate=0.25),
)


def _own_state(obs: Mapping[str, Any]) -> tuple[Mapping[str, Any], Mapping[str, Any], Any]:
    player = int(obs.get("player", 0) or 0)
    farms = obs.get("farms") or []
    farm = farms[player] if 0 <= player < len(farms) and isinstance(farms[player], Mapping) else {}
    private = obs.get("private") if isinstance(obs.get("private"), Mapping) else {}
    farmer = farm.get("farmer") or [0, 0]
    tile = None
    tiles = farm.get("tiles") or []
    if len(farmer) >= 2:
        x, y = int(farmer[0]), int(farmer[1])
        if 0 <= y < len(tiles) and 0 <= x < len(tiles[y]):
            tile = tiles[y][x]
    return farm, private, tile


def _sales_from_shed(private: Mapping[str, Any]) -> list[list[Any]]:
    shed = private.get("shed") if isinstance(private.get("shed"), Mapping) else {}
    return [["SELL", item, int(count)] for item, count in sorted(shed.items()) if int(count) > 0]


def _prepend_market(market: list[list[Any]], orders: list[list[Any]]) -> list[list[Any]]:
    return (orders + market)[:10]


def apply_selfplay_policy(
    policy: SelfplayPolicy,
    obs: Mapping[str, Any],
    action: Mapping[str, Any],
    rng: random.Random,
) -> Dict[str, Any]:
    """Apply one deterministic, environment-only regime to a BKG4 action.

    The wrapper sees exactly the live observation available to an agent. It never
    accesses replay panels, holdout cases, or opponent identity.
    """
    result: Dict[str, Any] = copy.deepcopy(dict(action))
    result.setdefault("farmer", ["PASS"])
    result.setdefault("hands", [])
    market = [list(order) for order in result.get("market", []) if isinstance(order, list)]
    step = int(obs.get("step", 0) or 0)
    farm, private, tile = _own_state(obs)
    seeds = private.get("seeds") if isinstance(private.get("seeds"), Mapping) else {}
    unlocked = farm.get("unlocked_quadrants") or []

    if policy.kind == "passive" and step < policy.passive_until:
        return {"farmer": ["PASS"], "hands": [], "market": []}
    if policy.kind == "random_early" and step < policy.random_prefix:
        result["market"] = [] if rng.random() < 0.5 else market
        return result
    if policy.kind == "noisy" and rng.random() < policy.noise_rate:
        market = []
    elif policy.kind == "greedy_market" and rng.random() < policy.market_bias:
        market = [order for order in market if order and order[0] == "SELL"]
    elif policy.kind == "mutated_bkg4" and rng.random() < policy.noise_rate:
        if rng.random() < 0.5:
            market = []
        else:
            result["farmer"] = ["PASS"]

    if policy.kind == "cash_hoarder" and step < policy.passive_until:
        market = [order for order in market if order and order[0] == "SELL"]
    elif policy.kind == "inventory_hoarder":
        market = [order for order in market if order and order[0] != "SELL"]
    elif policy.kind == "liquidator" and step >= policy.passive_until:
        market = _sales_from_shed(private)
    elif policy.kind == "market_squeeze":
        buys = [["BUY_PRODUCT", "WHEAT", 12], ["BUY_SEED", "CARROT", 4]]
        market = _prepend_market([order for order in market if order and order[0] != "SELL"], buys)
    elif policy.kind == "early_expansion" and step < policy.passive_until:
        if len(unlocked) < policy.land_target:
            market = _prepend_market(market, [["BUY_LAND"]])
    elif policy.kind == "late_expansion" and step >= policy.passive_until:
        if len(unlocked) < policy.land_target:
            market = _prepend_market(market, [["BUY_LAND"]])
    elif policy.kind == "farm_heavy":
        market = _prepend_market(market, [["BUY_SEED", policy.crop, 4]])
        if tile is None and int(seeds.get(policy.crop, 0)) > 0:
            result["farmer"] = ["PLANT", policy.crop]
    elif policy.kind == "animal_heavy":
        animal = policy.animal or "COW"
        structure = "COOP" if animal == "GOOSE" else "PASTURE"
        market = _prepend_market(market, [["BUY_ANIMAL", animal, 1], ["BUY_PRODUCT", "WHEAT", 12]])
        shed = private.get("shed") if isinstance(private.get("shed"), Mapping) else {}
        if tile is None and 24 <= step < 288:
            result["farmer"] = [f"BUILD_{structure}"]
        elif isinstance(tile, Mapping) and tile.get("kind") == structure:
            if int(shed.get(animal, 0)) > 0 and not tile.get("animal"):
                result["farmer"] = ["PLACE", animal, 1]
            elif tile.get("animal") and not tile.get("fed_today"):
                result["farmer"] = ["FEED"]
            elif tile.get("animal") and not tile.get("cared_today"):
                result["farmer"] = ["CARE"]

    result["market"] = market[:10]
    return result


def _make_agent(policy: SelfplayPolicy, rng: random.Random):
    from kaggriculture_bonsai.runtime import BKG4ChampionAgent

    agent = BKG4ChampionAgent(policy.checkpoint_asset_dir or None)
    if policy.kind == "champion":
        return agent

    if policy.kind == "option_program":
        from kaggriculture_bonsai.search.evaluator import ReferenceInterpreter, decode_program_batch

        programs, motifs = decode_program_batch(Path(policy.exploiter_program).read_bytes())
        if len(programs) != 1:
            raise ValueError("league exploiter bundle must contain exactly one program")
        interpreter = ReferenceInterpreter(programs[0], motifs)
        had_action = False

        def option_program(obs: Dict[str, Any], config: Any = None) -> Dict[str, Any]:
            nonlocal had_action, interpreter
            step = int(obs.get("step", 0) or 0)
            if step == 0:
                interpreter = ReferenceInterpreter(programs[0], motifs)
                had_action = False
            if had_action:
                interpreter.observe(obs)
            action = interpreter.action(obs, agent(obs, config))
            had_action = True
            return action

        return option_program

    def wrapped(obs: Dict[str, Any], config: Any = None) -> Dict[str, Any]:
        return apply_selfplay_policy(policy, obs, agent(obs, config), rng)

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
    from kaggle_environments import make

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
                                "policy_family": opponent_policy.family,
                                "policy_kind": opponent_policy.kind,
                                "strategic_lane": opponent_policy.strategic_lane,
                                "league_generation": opponent_policy.generation,
                                "parent_checkpoint": opponent_policy.parent_checkpoint,
                                "member_type": opponent_policy.member_type,
                                "exploiter_program": opponent_policy.exploiter_program,
                                "mutation_origin": opponent_policy.mutation_origin,
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
