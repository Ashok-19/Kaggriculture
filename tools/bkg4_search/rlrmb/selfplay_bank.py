"""Persistent self-play bank for autonomous Bonsai discovery.

Manages a reusable bank of fresh environment-generated cases with:
- Parallel game generation (multiprocessing)
- Incremental refresh (add new seeds, retain diverse older regimes)
- Provenance tracking per tape
- Panel assembly from bank contents
"""
from __future__ import annotations

import json
import multiprocessing
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from .selfplay_generator import (
    DEFAULT_POLICIES,
    GeneratedTape,
    SelfplayPolicy,
    generate_diverse_seeds,
    write_target_panel,
)

BANK_SCHEMA_VERSION = 1


@dataclass
class BankEntry:
    tape_path: str
    seat: int
    seed: int
    policy_name: str
    opponent_policy: str
    margin: float
    reward: float
    opponent_reward: float
    generation: int
    bkg4_version: str
    policy_family: str = "baseline"
    policy_kind: str = "champion"
    regime_metadata: dict[str, Any] = field(default_factory=dict)


def _run_single_game(args: tuple) -> list[dict[str, Any]]:
    """Run a single self-play game in a worker process. Returns serializable tape info."""
    import copy
    import hashlib
    import json as _json
    import random
    import sys

    output_dir, game_seed, seat, policy_dict, rng_state_int = args

    kag_env = Path(__file__).resolve().parents[3] / "kaggle-environments-master"
    champ = Path(__file__).resolve().parents[3] / "bonsai-kg" / "integrations" / "bkg4_champion"
    for p in (kag_env, champ):
        s = str(p)
        if s not in sys.path:
            sys.path.insert(0, s)

    from kaggle_environments import make  # type: ignore[import]
    from runtime import BKG4ChampionAgent  # type: ignore[import]
    from .selfplay_generator import _make_agent

    policy = SelfplayPolicy(**policy_dict)
    rng = random.Random(rng_state_int)

    candidate_rng = random.Random(rng.randint(0, 2**31))
    opponent_rng = random.Random(rng.randint(0, 2**31))

    champion_agent = BKG4ChampionAgent()
    opp_agent = _make_agent(policy, opponent_rng)

    candidate_actions: list[dict] = []
    opp_actions: list[dict] = []

    def candidate_fn(obs, config=None):
        a = champion_agent(obs, config)
        candidate_actions.append(copy.deepcopy(a))
        return a

    def opponent_fn(obs, config=None):
        a = opp_agent(obs, config)
        opp_actions.append(copy.deepcopy(a))
        return a

    env = make(
        "kaggriculture",
        configuration={"episodeSteps": 720, "seed": game_seed},
        debug=False,
    )
    players = [candidate_fn, opponent_fn] if seat == 0 else [opponent_fn, candidate_fn]
    env.run(players)
    final = env.steps[-1]
    reward = float(final[seat].reward)
    opp_reward = float(final[1 - seat].reward)

    def tape_id(s, st, pol_name, opp_name):
        raw = f"selfplay-v1:{s}:{st}:{pol_name}:{opp_name}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    results = []
    for record_seat, actions, name in (
        (seat, candidate_actions, "bkg4_candidate"),
        (1 - seat, opp_actions, policy.name),
    ):
        fid = tape_id(game_seed, record_seat, name, policy.name)
        tape_path = Path(output_dir) / f"selfplay_{fid}.json"
        if not tape_path.exists():
            document = {
                "source": {
                    "kind": "selfplay",
                "schema_version": 2,
                    "seed": game_seed,
                    "seat": record_seat,
                    "policy": name,
                    "opponent_policy": policy.name,
                    "reward": float(final[record_seat].reward),
                "opponent_reward": float(final[1 - record_seat].reward),
                "policy_family": policy.family,
                "policy_kind": policy.kind,
                "regime_metadata": {
                    "noise_rate": policy.noise_rate,
                    "passive_until": policy.passive_until,
                    "land_target": policy.land_target,
                    "crop": policy.crop,
                    "animal": policy.animal,
                    "checkpoint_asset_dir": policy.checkpoint_asset_dir,
                },
                },
                "actions": actions,
            }
            tape_path.write_text(
                _json.dumps(document, ensure_ascii=False, separators=(",", ":")) + "\n",
                encoding="utf-8",
            )

    cand_id = tape_id(game_seed, seat, "bkg4_candidate", policy.name)
    results.append({
        "path": str(Path(output_dir) / f"selfplay_{cand_id}.json"),
        "seat": seat,
        "seed": game_seed,
        "policy_name": "bkg4_candidate",
        "opponent_policy": policy.name,
        "margin": reward - opp_reward,
        "reward": reward,
        "opponent_reward": opp_reward,
        "policy_family": policy.family,
        "policy_kind": policy.kind,
        "regime_metadata": {
            "noise_rate": policy.noise_rate,
            "passive_until": policy.passive_until,
            "land_target": policy.land_target,
            "crop": policy.crop,
            "animal": policy.animal,
            "checkpoint_asset_dir": policy.checkpoint_asset_dir,
        },
    })
    return results


def _get_bkg4_version(asset_dir: str | None = None) -> str:
    if asset_dir:
        identity_path = Path(asset_dir) / "identity.json"
        if identity_path.exists():
            doc = json.loads(identity_path.read_text(encoding="utf-8"))
            return str(doc.get("controller_semantic_hash", "unknown"))
    return "current"


class SelfplayBank:
    def __init__(self, bank_dir: Path, *, asset_dir: str | None = None):
        self.bank_dir = bank_dir
        self.tapes_dir = bank_dir / "tapes"
        self.index_path = bank_dir / "index.json"
        self.asset_dir = asset_dir
        self._entries: list[BankEntry] = []
        self._load()

    def _load(self):
        if self.index_path.exists():
            doc = json.loads(self.index_path.read_text(encoding="utf-8"))
            known = {policy.name: policy for policy in DEFAULT_POLICIES}
            entries = []
            for stored in doc.get("entries", []):
                row = dict(stored)
                legacy_policy = known.get(str(row.get("opponent_policy", "")))
                if "policy_family" not in row:
                    row["policy_family"] = legacy_policy.family if legacy_policy else "legacy"
                if "policy_kind" not in row:
                    row["policy_kind"] = legacy_policy.kind if legacy_policy else "unknown"
                if "regime_metadata" not in row:
                    row["regime_metadata"] = {"legacy_schema": 1}
                entries.append(BankEntry(**row))
            self._entries = entries
        else:
            self._entries = []

    def _save(self):
        self.bank_dir.mkdir(parents=True, exist_ok=True)
        doc = {
            "schema_version": BANK_SCHEMA_VERSION,
            "num_entries": len(self._entries),
            "entries": [
                {
                    "tape_path": e.tape_path,
                    "seat": e.seat,
                    "seed": e.seed,
                    "policy_name": e.policy_name,
                    "opponent_policy": e.opponent_policy,
                    "margin": e.margin,
                    "reward": e.reward,
                    "opponent_reward": e.opponent_reward,
                    "generation": e.generation,
                    "bkg4_version": e.bkg4_version,
                    "policy_family": e.policy_family,
                    "policy_kind": e.policy_kind,
                    "regime_metadata": e.regime_metadata,
                }
                for e in self._entries
            ],
        }
        self.index_path.write_text(
            json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )

    @property
    def entries(self) -> list[BankEntry]:
        return list(self._entries)

    @property
    def size(self) -> int:
        return len(self._entries)

    def existing_seeds(self) -> set[int]:
        return {e.seed for e in self._entries}

    def existing_keys(self) -> set[tuple[int, int, str]]:
        return {(e.seed, e.seat, e.opponent_policy) for e in self._entries}

    def generate(
        self,
        *,
        num_seeds: int,
        policies: Sequence[SelfplayPolicy] | None = None,
        seats: Sequence[int] = (0, 1),
        rng_seed: int = 0,
        generation: int = 0,
        workers: int = 4,
    ) -> dict[str, Any]:
        self.tapes_dir.mkdir(parents=True, exist_ok=True)
        if policies is None:
            policies = list(DEFAULT_POLICIES)

        existing = self.existing_seeds()
        seeds = generate_diverse_seeds(num_seeds, rng_seed=rng_seed, existing=list(existing))
        existing_keys = self.existing_keys()

        import random
        rng = random.Random(rng_seed)
        tasks = []
        for game_seed in seeds:
            for seat in seats:
                for policy in policies:
                    if (game_seed, seat, policy.name) in existing_keys:
                        continue
                    rng_state = rng.randint(0, 2**31)
                    tasks.append((
                        str(self.tapes_dir),
                        game_seed,
                        seat,
                        {
                            "name": policy.name,
                            "kind": policy.kind,
                            "noise_rate": policy.noise_rate,
                            "market_bias": policy.market_bias,
                            "passive_until": policy.passive_until,
                            "random_prefix": policy.random_prefix,
                            "family": policy.family,
                            "land_target": policy.land_target,
                            "crop": policy.crop,
                            "animal": policy.animal,
                            "checkpoint_asset_dir": policy.checkpoint_asset_dir,
                        },
                        rng_state,
                    ))

        if not tasks:
            return {"new_tapes": 0, "total": self.size, "generation_seconds": 0.0}

        bkg4_version = _get_bkg4_version(self.asset_dir)
        bounded_workers = min(workers, len(tasks))

        started = time.monotonic()
        new_entries = []

        if bounded_workers <= 1:
            for task in tasks:
                results = _run_single_game(task)
                for r in results:
                    new_entries.append(BankEntry(
                        tape_path=r["path"], seat=r["seat"], seed=r["seed"],
                        policy_name=r["policy_name"], opponent_policy=r["opponent_policy"],
                        margin=r["margin"], reward=r["reward"],
                        opponent_reward=r["opponent_reward"],
                        generation=generation, bkg4_version=bkg4_version,
                        policy_family=r["policy_family"], policy_kind=r["policy_kind"],
                        regime_metadata=r["regime_metadata"],
                    ))
        else:
            ctx = multiprocessing.get_context("spawn")
            with ctx.Pool(bounded_workers) as pool:
                for results in pool.imap_unordered(_run_single_game, tasks):
                    for r in results:
                        new_entries.append(BankEntry(
                            tape_path=r["path"], seat=r["seat"], seed=r["seed"],
                            policy_name=r["policy_name"], opponent_policy=r["opponent_policy"],
                            margin=r["margin"], reward=r["reward"],
                            opponent_reward=r["opponent_reward"],
                            generation=generation, bkg4_version=bkg4_version,
                            policy_family=r["policy_family"], policy_kind=r["policy_kind"],
                            regime_metadata=r["regime_metadata"],
                        ))

        elapsed = time.monotonic() - started
        self._entries.extend(new_entries)
        self._save()

        return {
            "new_tapes": len(new_entries),
            "total": self.size,
            "generation_seconds": round(elapsed, 1),
            "seeds_generated": len(seeds),
            "tasks_run": len(tasks),
            "workers": bounded_workers,
            "throughput_games_per_second": round(len(tasks) / max(0.1, elapsed), 2),
        }

    def refresh(
        self,
        *,
        num_new_seeds: int,
        policies: Sequence[SelfplayPolicy] | None = None,
        seats: Sequence[int] = (0, 1),
        generation: int = 0,
        workers: int = 4,
    ) -> dict[str, Any]:
        rng_seed = hash(("refresh", generation)) & 0x7FFFFFFF
        return self.generate(
            num_seeds=num_new_seeds, policies=policies, seats=seats,
            rng_seed=rng_seed, generation=generation, workers=workers,
        )

    def assemble_panel(
        self,
        panel_path: Path,
        *,
        max_tapes: int | None = None,
        policy_filter: set[str] | None = None,
    ) -> int:
        entries = self._entries
        if policy_filter:
            entries = [e for e in entries if e.opponent_policy in policy_filter]
        missing = [entry.tape_path for entry in entries if not Path(entry.tape_path).is_file()]
        if missing:
            raise ValueError(
                f"selfplay bank has {len(missing)} missing tape payload(s); "
                "restore the bank before assembling BUILD"
            )
        if max_tapes and len(entries) > max_tapes:
            # Round-robin policy groups instead of allowing an older, plentiful
            # baseline regime to crowd fresh strategic variants out of BUILD.
            by_policy: dict[str, list[BankEntry]] = {}
            for entry in entries:
                by_policy.setdefault(entry.opponent_policy, []).append(entry)
            for group in by_policy.values():
                group.sort(key=lambda entry: (entry.generation, entry.seed, entry.seat), reverse=True)
            entries = []
            while len(entries) < max_tapes and any(by_policy.values()):
                for policy in sorted(by_policy):
                    if by_policy[policy] and len(entries) < max_tapes:
                        entries.append(by_policy[policy].pop(0))

        tapes = [
            GeneratedTape(
                path=Path(e.tape_path), seat=e.seat, seed=e.seed,
                policy_name=e.policy_name, opponent_policy=e.opponent_policy,
                margin=e.margin, reward=e.reward, opponent_reward=e.opponent_reward,
            )
            for e in entries
        ]
        return write_target_panel(tapes, panel_path)

    def stats(self) -> dict[str, Any]:
        if not self._entries:
            return {"size": 0}
        margins = [e.margin for e in self._entries]
        policy_counts: dict[str, int] = {}
        gen_counts: dict[int, int] = {}
        version_counts: dict[str, int] = {}
        family_counts: dict[str, int] = {}
        for e in self._entries:
            policy_counts[e.opponent_policy] = policy_counts.get(e.opponent_policy, 0) + 1
            gen_counts[e.generation] = gen_counts.get(e.generation, 0) + 1
            version_counts[e.bkg4_version] = version_counts.get(e.bkg4_version, 0) + 1
            family_counts[e.policy_family] = family_counts.get(e.policy_family, 0) + 1
        return {
            "size": len(self._entries),
            "unique_seeds": len(self.existing_seeds()),
            "policies": policy_counts,
            "generations": gen_counts,
            "bkg4_versions": version_counts,
            "policy_families": family_counts,
            "margin_min": round(min(margins), 1),
            "margin_max": round(max(margins), 1),
            "margin_mean": round(sum(margins) / len(margins), 1),
        }
