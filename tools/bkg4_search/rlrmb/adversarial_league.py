"""Small internal adversarial league for fresh Bonsai weakness discovery.

The league is deliberately not a promotion system.  It searches deterministic
Option-Programs with the existing CUDA evaluator, materializes the best internal
opponents, and emits lane-balanced self-play cases for the unchanged RLRMB
specialist and Phase-1 routing pipeline.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .beam import CudaBatchEvaluator
from .checkpoints import Checkpoint, EventType, Lane
from .evaluator import encode_program_batch
from .ir import ActionFamily, ActionSource, MacroProgram, Phase, Termination, TerminationType
from .motifs import Motif, motif_from_route_window
from .selfplay_generator import SelfplayPolicy, generate_selfplay_tapes

LEAGUE_SCHEMA_VERSION = 1
ALL_LANES = tuple(Lane)


@dataclass(frozen=True)
class LeagueMember:
    policy_id: str
    parent: str
    checkpoint: str
    generation: int
    member_type: str
    strategic_lane: str
    mutation_origin: str = ""
    exploiter_origin: str = ""
    active: bool = True

    def __post_init__(self) -> None:
        if not self.policy_id or self.generation < 0:
            raise ValueError("league members require an ID and non-negative generation")
        if self.strategic_lane not in {lane.value for lane in ALL_LANES}:
            raise ValueError(f"unknown strategic lane: {self.strategic_lane}")


class LeagueManager:
    """Persistent, bounded population of useful internal BKG4 policies."""

    def __init__(self, path: Path, *, max_members: int = 24) -> None:
        if max_members < len(ALL_LANES):
            raise ValueError("league capacity must permit at least one member per lane")
        self.path = path
        self.max_members = max_members
        self.members: list[LeagueMember] = []
        if path.exists():
            document = json.loads(path.read_text(encoding="utf-8"))
            if document.get("schema_version") != LEAGUE_SCHEMA_VERSION:
                raise ValueError("unsupported league schema version")
            self.members = [LeagueMember(**row) for row in document.get("members", [])]

    def add(self, member: LeagueMember) -> None:
        self.members = [row for row in self.members if row.policy_id != member.policy_id]
        self.members.append(member)
        if len(self.members) > self.max_members:
            protected = {Lane(row.strategic_lane) for row in self.members if row.active}
            removable = sorted(
                self.members,
                key=lambda row: (row.member_type == "champion", row.active,
                                 row.generation, row.policy_id),
            )
            for row in removable:
                lane = Lane(row.strategic_lane)
                same_lane = [item for item in self.members if item.active and item.strategic_lane == lane.value]
                if row.member_type != "champion" and (len(same_lane) > 1 or lane not in protected):
                    self.members.remove(row)
                    break
            if len(self.members) > self.max_members:
                raise ValueError("league capacity cannot preserve champion and lane coverage")
        self.save()

    def freeze_previous_champion(self, *, policy_id: str, checkpoint: str, generation: int) -> None:
        self.add(LeagueMember(policy_id, "BKG4", checkpoint, generation,
                              "frozen_checkpoint", Lane.GENERAL.value))

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        document = {
            "schema_version": LEAGUE_SCHEMA_VERSION,
            "max_members": self.max_members,
            "members": [asdict(row) for row in sorted(self.members, key=lambda row: row.policy_id)],
        }
        self.path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def stats(self) -> dict[str, Any]:
        lanes = {lane.value: 0 for lane in ALL_LANES}
        types: dict[str, int] = {}
        for row in self.members:
            lanes[row.strategic_lane] += 1
            types[row.member_type] = types.get(row.member_type, 0) + 1
        return {"members": len(self.members), "lanes": lanes, "types": types}


@dataclass(frozen=True)
class ExploiterResult:
    policy: SelfplayPolicy
    program: MacroProgram
    cuda_games: int
    gpu_ms: float
    wins: int
    mean_margin: float
    worst_margin: int


def _lane_starts(lane: Lane) -> tuple[int, ...]:
    return {
        Lane.ANIMAL_TRANSITION: (144, 264),
        Lane.MARKET_LIQUIDITY: (96, 192),
        Lane.CROP_TRANSITION: (48, 216),
        Lane.LAND_INVESTMENT: (24, 120),
        Lane.WORKER_PRODUCTION: (72, 168),
        Lane.ENDGAME_LIQUIDATION: (576, 648),
        Lane.MECHANIC_EXPLOIT: (0, 336),
        Lane.GENERAL: (0, 288),
    }[lane]


def make_bounded_exploiter_candidates(
    donor_routes: Sequence[Path], *, routes_per_lane: int = 2, duration: int = 24,
) -> tuple[tuple[MacroProgram, ...], dict[str, Motif]]:
    """Create a small, explicitly lane-balanced deterministic search population."""
    routes = [path for path in donor_routes if path.is_file()][:routes_per_lane]
    if not routes:
        raise ValueError("exploiter search requires at least one internal donor route")
    programs: list[MacroProgram] = []
    motifs: dict[str, Motif] = {}
    for lane in ALL_LANES:
        for route, start in zip(routes, _lane_starts(lane)):
            cap = min(duration, 719 - start)
            motif = motif_from_route_window(
                route, start, cap, tuple(ActionFamily),
                provenance=("internal_league_cuda_exploiter", lane.value),
            )
            motifs[motif.semantic_id] = motif
            checkpoint = Checkpoint(start, lane, (EventType.FIRST_EFFECT_DIVERGENCE,))
            phase = Phase(ActionSource.DONOR_BLOCK, tuple(ActionFamily),
                          Termination(TerminationType.FIXED_CAP, cap), motif.semantic_id)
            programs.append(MacroProgram(lane, checkpoint, (phase,), cap,
                                         provenance=("internal_league_cuda_exploiter",)))
    return tuple(programs), motifs


def search_cuda_exploiters(
    *, binary: Path, asset_dir: Path, cases_path: Path, donor_routes: Sequence[Path],
    output_dir: Path, generation: int = 0, routes_per_lane: int = 2,
) -> tuple[list[ExploiterResult], dict[str, Any]]:
    """Select one champion-exposing deterministic program per strategic lane."""
    programs, motifs = make_bounded_exploiter_candidates(
        donor_routes, routes_per_lane=routes_per_lane,
    )
    started = time.monotonic()
    evaluator = CudaBatchEvaluator(binary=binary, asset_dir=asset_dir, cases_path=cases_path)
    batch = evaluator(programs, motifs)
    elapsed = time.monotonic() - started
    output_dir.mkdir(parents=True, exist_ok=True)
    selected: list[ExploiterResult] = []
    for lane in ALL_LANES:
        lane_rows = [row for row in batch.programs if row.program.lane is lane]
        ranked = []
        for row in lane_rows:
            block = row.effect.blocks[0]
            margins = tuple(base + delta for base, delta in zip(block.baseline_margins, block.margin_deltas))
            ranked.append((sum(value > 0 for value in margins), sum(margins) / len(margins),
                           min(margins), row.program, margins))
        wins, mean, worst, program, _ = max(ranked, key=lambda item: (item[0], item[1], item[2], item[3].semantic_id))
        bundle = output_dir / f"exploiter_{lane.value.lower()}.rlrb"
        used = {phase.motif_id for phase in program.phases if phase.motif_id}
        bundle.write_bytes(encode_program_batch((program,), {key: motifs[key] for key in used}))
        policy_id = f"league_g{generation}_{lane.value.lower()}_{program.semantic_id[:10]}"
        policy = SelfplayPolicy(
            policy_id, "option_program", family="adversarial_league",
            checkpoint_asset_dir=str(asset_dir), strategic_lane=lane.value,
            generation=generation, parent_checkpoint=str(asset_dir),
            member_type="cuda_exploiter", exploiter_program=str(bundle),
            mutation_origin="bounded_internal_donor_window",
        )
        selected.append(ExploiterResult(policy, program, batch.games,
                                        batch.cuda_ms, wins, mean, worst))
    summary = {
        "schema_version": LEAGUE_SCHEMA_VERSION,
        "objective": "maximize wins then mean margin against current BKG4",
        "candidate_programs": len(programs),
        "cases": batch.cases,
        "cuda_games": batch.games,
        "gpu_ms": batch.cuda_ms,
        "host_seconds": round(elapsed, 3),
        "selected": [{"policy_id": row.policy.name, "lane": row.program.lane.value,
                      "wins": row.wins, "mean_margin": round(row.mean_margin, 2),
                      "worst_margin": row.worst_margin} for row in selected],
    }
    (output_dir / "exploiter_search.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return selected, summary


def bootstrap_internal_league(
    manager: LeagueManager, *, asset_dir: Path, frozen_checkpoints: Sequence[Path],
    exploiters: Sequence[ExploiterResult], generation: int,
) -> None:
    manager.add(LeagueMember("bkg4_current", "BKG4", str(asset_dir), generation,
                             "champion", Lane.GENERAL.value))
    for index, checkpoint in enumerate(frozen_checkpoints):
        manager.add(LeagueMember(f"bkg4_frozen_{index}", "BKG4", str(checkpoint),
                                 max(0, generation - index - 1), "frozen_checkpoint",
                                 Lane.GENERAL.value))
    for result in exploiters:
        manager.add(LeagueMember(
            result.policy.name, "bkg4_current", result.policy.exploiter_program,
            generation, "cuda_exploiter", result.program.lane.value,
            mutation_origin=result.policy.mutation_origin,
            exploiter_origin="rlrmb_eval:win_mean_margin",
        ))


def generate_league_build_cases(
    output_dir: Path, *, seeds: Sequence[int], exploiters: Sequence[ExploiterResult],
    rng_seed: int = 0,
) -> tuple[Path, dict[str, Any]]:
    """Generate fresh champion failures, keeping BUILD provenance internal-only."""
    tapes = generate_selfplay_tapes(
        output_dir / "tapes", seeds=seeds,
        policies=[row.policy for row in exploiters], seats=(0, 1), rng_seed=rng_seed,
    )
    by_lane: dict[str, list[Any]] = {lane.value: [] for lane in ALL_LANES}
    lane_by_policy = {row.policy.name: row.program.lane.value for row in exploiters}
    for tape in tapes:
        by_lane[lane_by_policy[tape.opponent_policy]].append(tape)
    selected = []
    quota = max(1, min((len(rows) for rows in by_lane.values() if rows), default=1))
    for lane in ALL_LANES:
        rows = sorted(by_lane[lane.value], key=lambda row: (row.margin, row.seed, row.seat))
        selected.extend(rows[:quota])
    panel = output_dir / "build_panel.tsv"
    from .selfplay_generator import write_target_panel
    write_target_panel(selected, panel)
    summary = {
        "build_only": True,
        "holdout_sources": "EXCLUDED",
        "fresh_cases": len(selected),
        "champion_losses": sum(row.margin < 0 for row in selected),
        "high_regret_cases": sum(row.margin < -1000 for row in selected),
        "lane_coverage": {lane: len(rows[:quota]) for lane, rows in by_lane.items()},
        "panel": str(panel),
    }
    (output_dir / "build_manifest.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return panel, summary
