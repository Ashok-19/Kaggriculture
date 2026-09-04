"""Bounded direct search over localized Bonsai action-rule edits.

Unlike RLRMB donor motifs, every action block in this module is synthesized
from the legal Kaggriculture action grammar.  Programs still use the existing
Option-Program ABI so CUDA evaluation, CEGIS, materialization, and composition
remain identical between search oracles.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence

from kaggriculture_bonsai.league.build_regimes import BuildRegime, BuildSupport, build_support

from .beam import BatchEvaluation
from .checkpoints import Checkpoint
from .effects import EffectVector, effect_signature
from .ir import ActionFamily, ActionSource, MacroProgram, Phase, Termination, TerminationType
from .motifs import Motif, motif_from_actions
from .schema import DatasetRole, stable_identity


@dataclass(frozen=True)
class DirectRuleTemplate:
    """One compiler-ready action leaf and the action families it owns."""

    name: str
    action: Mapping[str, Any]
    families: tuple[ActionFamily, ...]


@dataclass(frozen=True)
class DirectSearchConfig:
    cuda_game_budget: int
    initial_candidates: int = 120
    elite_count: int = 16
    max_leaf_steps: int = 2

    def __post_init__(self) -> None:
        if self.cuda_game_budget <= 0 or self.initial_candidates <= 0 or self.elite_count <= 0:
            raise ValueError("direct-search bounds must be positive")
        if self.max_leaf_steps not in (1, 2):
            raise ValueError("direct search supports one- or two-leaf edits")


@dataclass(frozen=True)
class DirectCandidate:
    program: MacroProgram
    template_names: tuple[str, ...]
    depth: int
    effect: EffectVector | None = None
    build_support: BuildSupport | None = None
    cuda_games: int = 0
    fault: str | None = None


@dataclass(frozen=True)
class DirectSearchResult:
    candidates: tuple[DirectCandidate, ...]
    motifs: Mapping[str, Motif]
    metrics: Mapping[str, Any]


def _action(*, farmer: Sequence[Any] = ("PASS",), market: Sequence[Sequence[Any]] = ()) -> dict[str, Any]:
    return {"farmer": list(farmer), "hands": [], "market": [list(row) for row in market]}


def direct_rule_templates() -> tuple[DirectRuleTemplate, ...]:
    """Return a compact action grammar, independent of routes and opponents."""
    rows: list[DirectRuleTemplate] = []

    def farmer(name: str, op: str, *args: Any) -> None:
        rows.append(DirectRuleTemplate(name, _action(farmer=(op, *args)), (ActionFamily.FARMER,)))

    def market(name: str, op: str, *args: Any) -> None:
        rows.append(DirectRuleTemplate(name, _action(market=((op, *args),)), (ActionFamily.MARKET,)))

    for op in ("NORTH", "SOUTH", "EAST", "WEST", "WATER", "HARVEST", "FERTILIZE", "DIG",
               "BUILD_COOP", "BUILD_PASTURE", "FEED", "COLLECT_FERTILIZER", "CARE", "DROP"):
        farmer(f"farmer_{op.lower()}", op)
    for crop in ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON"):
        farmer(f"plant_{crop.lower()}", "PLANT", crop)
    for op in ("HIRE", "BUY_LAND"):
        market(f"market_{op.lower()}", op)
    for crop in ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON"):
        for quantity in (1, 4):
            market(f"buy_seed_{crop.lower()}_{quantity}", "BUY_SEED", crop, quantity)
    for item in ("WHEAT", "FERTILIZER"):
        for quantity in (1, 4):
            market(f"buy_product_{item.lower()}_{quantity}", "BUY_PRODUCT", item, quantity)
    for animal in ("GOOSE", "COW", "SHEEP"):
        market(f"buy_animal_{animal.lower()}", "BUY_ANIMAL", animal, 1)
    for item in ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL"):
        market(f"sell_{item.lower()}_4", "SELL", item, 4)
    return tuple(rows)


def _program(checkpoint: Checkpoint, templates: Sequence[DirectRuleTemplate], motifs: dict[str, Motif]) -> DirectCandidate:
    actions = tuple(row.action for row in templates)
    families = tuple(sorted({family for row in templates for family in row.families}, key=lambda row: row.value))
    motif = motif_from_actions(
        actions, families,
        provenance=("direct_bonsai_action_grammar", *(f"leaf={row.name}" for row in templates)),
    )
    motifs[motif.semantic_id] = motif
    cap = len(actions)
    phase = Phase(
        ActionSource.SEMANTIC_SPLICE, families,
        Termination(TerminationType.FIXED_CAP, cap), motif.semantic_id,
    )
    program = MacroProgram(
        checkpoint.lane, checkpoint, (phase,), cap,
        provenance=("direct_localized_bonsai_search",),
    )
    return DirectCandidate(program, tuple(row.name for row in templates), cap - 1)


def generate_direct_seeds(
    checkpoints: Sequence[Checkpoint], templates: Sequence[DirectRuleTemplate] | None = None,
) -> tuple[tuple[DirectCandidate, ...], dict[str, Motif]]:
    """Generate a round-robin population with uniform checkpoint coverage."""
    checkpoints = tuple(sorted(checkpoints, key=lambda row: row.rank_key()))
    templates = tuple(templates or direct_rule_templates())
    if not checkpoints or not templates:
        raise ValueError("direct search requires checkpoints and rule templates")
    motifs: dict[str, Motif] = {}
    candidates: list[DirectCandidate] = []
    for offset in range(len(templates)):
        for index, checkpoint in enumerate(checkpoints):
            candidates.append(_program(checkpoint, (templates[(offset + index) % len(templates)],), motifs))
    unique = {row.program.semantic_id: row for row in candidates}
    return tuple(unique[key] for key in dict.fromkeys(row.program.semantic_id for row in candidates)), motifs


def _rank(candidate: DirectCandidate) -> tuple[Any, ...]:
    support = candidate.build_support
    if support is None or candidate.effect is None:
        return (1, 1, 0, 0, 0, 0, candidate.program.semantic_id)
    block = next(row for row in candidate.effect.blocks if row.role is DatasetRole.TARGET)
    return (
        -int(support.cross_regime_supported),
        support.win_to_loss,
        -support.loss_to_win,
        -support.seed_support,
        -support.member_support,
        -sum(block.margin_deltas),
        candidate.program.semantic_id,
    )


def _evaluate(
    candidates: Sequence[DirectCandidate], motifs: Mapping[str, Motif], evaluator: Any,
    regimes: Sequence[BuildRegime],
) -> tuple[tuple[DirectCandidate, ...], BatchEvaluation]:
    batch = evaluator(tuple(row.program for row in candidates), motifs)
    if len(batch.programs) != len(candidates):
        raise RuntimeError("direct evaluator changed candidate count")
    result: list[DirectCandidate] = []
    for source, evaluated in zip(candidates, batch.programs):
        if source.program.semantic_id != evaluated.program.semantic_id:
            raise RuntimeError("SEMANTIC_MISMATCH: direct evaluator reordered candidates")
        if not evaluated.valid:
            result.append(replace(source, cuda_games=batch.cases, fault=evaluated.fault))
            continue
        block = next(row for row in evaluated.effect.blocks if row.role is DatasetRole.TARGET)
        result.append(replace(
            source, effect=evaluated.effect, build_support=build_support(block, regimes),
            cuda_games=batch.cases,
        ))
    return tuple(result), batch


def _diverse_elites(candidates: Sequence[DirectCandidate], limit: int) -> tuple[DirectCandidate, ...]:
    ranked = sorted((row for row in candidates if row.effect is not None), key=_rank)
    selected: list[DirectCandidate] = []
    cells: set[tuple[str, int, str]] = set()
    for row in ranked:
        cell = (
            row.program.lane.value,
            row.program.initiation.step // 24,
            effect_signature(row.effect),
        )
        if cell in cells:
            continue
        cells.add(cell)
        selected.append(row)
        if len(selected) == limit:
            break
    return tuple(selected)


def run_direct_search(
    checkpoints: Sequence[Checkpoint], evaluator: Any, regimes: Sequence[BuildRegime],
    config: DirectSearchConfig,
) -> DirectSearchResult:
    """Run a two-level, QD-retained direct Bonsai edit beam."""
    started = time.monotonic()
    seeds, motifs = generate_direct_seeds(checkpoints)
    cases = int(evaluator.case_count)
    max_total_candidates = config.cuda_game_budget // cases - 1
    if max_total_candidates < 1:
        raise ValueError("CUDA budget cannot evaluate one direct candidate plus parent")
    initial_count = min(len(seeds), config.initial_candidates, max_total_candidates)
    initial, batch0 = _evaluate(seeds[:initial_count], motifs, evaluator, regimes)
    games = batch0.games
    gpu_ms = batch0.cuda_ms
    launches = batch0.launches
    all_candidates = list(initial)

    expansion_count = 0
    if config.max_leaf_steps == 2:
        remaining_slots = max(0, (config.cuda_game_budget - games) // cases - 1)
        if remaining_slots:
            templates = direct_rule_templates()
            expansions: list[DirectCandidate] = []
            for elite in _diverse_elites(initial, config.elite_count):
                first = next(row for row in templates if row.name == elite.template_names[0])
                compatible = [row for row in templates if row.families == first.families]
                for second in compatible:
                    expansions.append(_program(elite.program.initiation, (first, second), motifs))
            known = {row.program.semantic_id for row in initial}
            unique = {row.program.semantic_id: row for row in expansions if row.program.semantic_id not in known}
            pending = tuple(unique[key] for key in dict.fromkeys(row.program.semantic_id for row in expansions)
                            if key in unique)[:remaining_slots]
            if pending:
                evaluated, batch1 = _evaluate(pending, motifs, evaluator, regimes)
                all_candidates.extend(evaluated)
                expansion_count = len(evaluated)
                games += batch1.games
                gpu_ms += batch1.cuda_ms
                launches += batch1.launches

    valid = tuple(row for row in all_candidates if row.effect is not None)
    target_blocks = [next(block for block in row.effect.blocks if block.role is DatasetRole.TARGET)
                     for row in valid]
    metrics = {
        "schema_version": 1,
        "representation": "DIRECT_LOCALIZED_BONSAI_RULE_V1",
        "template_count": len(direct_rule_templates()),
        "checkpoints": len(checkpoints),
        "initial_candidates": len(initial),
        "expanded_candidates": expansion_count,
        "cuda_candidates_evaluated": len(all_candidates),
        "cuda_cases": cases,
        "cuda_games": games,
        "cuda_budget": config.cuda_game_budget,
        "evaluator_launches": launches,
        "cuda_ms": round(gpu_ms, 3),
        "wall_seconds": round(time.monotonic() - started, 3),
        "faults": sum(row.fault is not None for row in all_candidates),
        "semantic_mismatches": 0,
        "unique_effect_signatures": len({effect_signature(row.effect) for row in valid}),
        "target_loss_to_win": max((sum(block.loss_to_win) for block in target_blocks), default=0),
        "target_win_to_loss": min((sum(block.win_to_loss) for block in target_blocks), default=0),
        "useful_build_specialists": sum(
            bool(row.build_support and row.build_support.loss_to_win > 0
                 and row.build_support.win_to_loss <= 2) for row in valid
        ),
        "cross_regime_supported_specialists": sum(
            bool(row.build_support and row.build_support.cross_regime_supported) for row in valid
        ),
        "search_identity": stable_identity(
            "DIRECT-LOCALIZED-BONSAI-SEARCH-V1", config.__dict__,
            [row.program.semantic_id for row in all_candidates],
        ),
    }
    return DirectSearchResult(tuple(all_candidates), motifs, metrics)
