"""Deterministic hierarchical, archive-backed Option-Program beam search."""

from __future__ import annotations

import re
import subprocess
import tempfile
import time
from dataclasses import dataclass, replace
from enum import Enum
from itertools import chain as _iterchain
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from .archive import Archive, ArchiveEntry, InsertResult, make_cell_key
from .checkpoints import Checkpoint
from .effects import EffectBlock, EffectVector, effect_signature
from .evaluator import FaultCode, decode_device_results, encode_device_batch
from .failures import FailureClass
from .ir import (
    ACTION_HORIZON,
    MAX_PHASES,
    ActionFamily,
    ActionSource,
    MacroProgram,
    Phase,
    Termination,
    TerminationType,
)
from .motifs import Motif
from .schema import DatasetRole, stable_identity

BEAM_SCHEMA_VERSION = 1


class SeedSource(str, Enum):
    PARENT_ONLY = "PARENT_ONLY"
    HISTORICAL_MOTIF = "HISTORICAL_MOTIF"
    MASKED_DONOR_PHASE = "MASKED_DONOR_PHASE"
    PARAMETERIZED_PRIMITIVE = "PARAMETERIZED_PRIMITIVE"


class ExpansionOperator(str, Enum):
    INSERT_PHASE = "INSERT_PHASE"
    REPLACE_PHASE_SOURCE = "REPLACE_PHASE_SOURCE"
    SHIFT_PHASE_BOUNDARY_TO_ADJACENT_EVENT = "SHIFT_PHASE_BOUNDARY_TO_ADJACENT_EVENT"
    CHANGE_ACTION_FAMILY_MASK = "CHANGE_ACTION_FAMILY_MASK"
    TRUNCATE_AT_EVENT = "TRUNCATE_AT_EVENT"
    EXTEND_TO_EVENT = "EXTEND_TO_EVENT"
    MUTATE_BOUNDED_NUMERIC_PARAMETER = "MUTATE_BOUNDED_NUMERIC_PARAMETER"
    SWAP_SEMANTIC_DONOR_OR_MOTIF_PHASE = "SWAP_SEMANTIC_DONOR_OR_MOTIF_PHASE"


ALL_MASKS = (
    (ActionFamily.FARMER,),
    (ActionFamily.HANDS,),
    (ActionFamily.MARKET,),
    (ActionFamily.FARMER, ActionFamily.HANDS),
    (ActionFamily.FARMER, ActionFamily.MARKET),
    (ActionFamily.HANDS, ActionFamily.MARKET),
    tuple(ActionFamily),
)


@dataclass(frozen=True)
class BeamConfig:
    max_depth: int = 2
    max_phases: int = MAX_PHASES
    cuda_game_budget: int = 128
    max_candidates_per_expansion: int = 32
    stagnation_rounds: int = 2
    minimum_novelty_improvement: int = 1
    minimum_target_quality_improvement: int = 1
    allowed_operators: tuple[ExpansionOperator, ...] = tuple(ExpansionOperator)
    schema_version: int = BEAM_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != BEAM_SCHEMA_VERSION:
            raise ValueError(f"unsupported beam schema version: {self.schema_version!r}")
        for field in ("max_depth", "max_phases", "cuda_game_budget", "max_candidates_per_expansion", "stagnation_rounds"):
            if type(getattr(self, field)) is not int or getattr(self, field) <= 0:
                raise ValueError(f"{field} must be a positive integer")
        if self.max_phases > MAX_PHASES:
            raise ValueError(f"max_phases cannot exceed {MAX_PHASES}")
        for field in ("minimum_novelty_improvement", "minimum_target_quality_improvement"):
            if type(getattr(self, field)) is not int or getattr(self, field) < 0:
                raise ValueError(f"{field} must be a non-negative integer")
        try:
            operators = tuple(ExpansionOperator(item) for item in self.allowed_operators)
        except (TypeError, ValueError) as exc:
            raise ValueError("unknown beam expansion operator") from exc
        if not operators:
            raise ValueError("at least one expansion operator is required")
        object.__setattr__(self, "allowed_operators", tuple(sorted(set(operators), key=lambda item: item.value)))

    @property
    def semantic_id(self) -> str:
        return stable_identity("RLRMB-BEAM-CONFIG-V1", {
            "schema_version": self.schema_version, "max_depth": self.max_depth,
            "max_phases": self.max_phases, "cuda_game_budget": self.cuda_game_budget,
            "max_candidates_per_expansion": self.max_candidates_per_expansion,
            "stagnation_rounds": self.stagnation_rounds,
            "minimum_novelty_improvement": self.minimum_novelty_improvement,
            "minimum_target_quality_improvement": self.minimum_target_quality_improvement,
            "allowed_operators": [item.value for item in self.allowed_operators],
        })


@dataclass(frozen=True)
class Lineage:
    parent_id: str | None
    operator: str
    parameters: tuple[tuple[str, Any], ...]
    result_id: str
    depth: int

    def to_document(self) -> dict[str, Any]:
        return {"parent_id": self.parent_id, "operator": self.operator,
                "parameters": dict(self.parameters), "result_id": self.result_id, "depth": self.depth}


@dataclass(frozen=True)
class BeamCandidate:
    program: MacroProgram
    checkpoint: Checkpoint
    depth: int
    seed_source: SeedSource
    lineage: Lineage
    effect: EffectVector | None = None
    archive_result: InsertResult | None = None
    cuda_games: int = 0
    fault: str | None = None
    failure_class: FailureClass | None = None
    family_ref: str | None = None

    @classmethod
    def seed(cls, program: MacroProgram, source: SeedSource) -> "BeamCandidate":
        return cls(program, program.initiation, 0, source,
                   Lineage(None, source.value, (), program.semantic_id, 0))


@dataclass(frozen=True)
class EvaluatedProgram:
    program: MacroProgram
    effect: EffectVector
    valid: bool
    fault: str | None


@dataclass(frozen=True)
class BatchEvaluation:
    programs: tuple[EvaluatedProgram, ...]
    cases: int
    games: int
    launches: int
    host_seconds: float
    cuda_ms: float


def _motifs(motif_library: Mapping[str, Motif]) -> tuple[Motif, ...]:
    values = tuple(sorted(motif_library.values(), key=lambda item: item.semantic_id))
    if any(key != value.semantic_id for key, value in motif_library.items()):
        raise ValueError("motif library keys must be semantic motif IDs")
    return values


def _phase_for_motif(motif: Motif, families: tuple[ActionFamily, ...], *, donor: bool = False) -> Phase:
    cap = min(motif.source_length, 8)
    return Phase(ActionSource.DONOR_BLOCK if donor else ActionSource.SEMANTIC_SPLICE,
                 families, Termination(TerminationType.FIXED_CAP, cap), motif.semantic_id)


def generate_seeds(
    checkpoints: Sequence[Checkpoint], motif_library: Mapping[str, Motif], config: BeamConfig
) -> tuple[BeamCandidate, ...]:
    generated: list[BeamCandidate] = []
    for checkpoint in sorted(checkpoints, key=lambda item: item.rank_key()):
        remaining = ACTION_HORIZON - checkpoint.step
        parent = MacroProgram(checkpoint.lane, checkpoint,
                              (Phase(ActionSource.PARENT, (), Termination(TerminationType.MAX_STEPS, 1)),), 1)
        generated.append(BeamCandidate.seed(parent, SeedSource.PARENT_ONLY))
        for motif in _motifs(motif_library):
            full = tuple(motif.supported_families)
            cap = min(motif.source_length, 8, remaining)
            donor_phase = replace(_phase_for_motif(motif, full, donor=True),
                                  termination=Termination(TerminationType.FIXED_CAP, cap))
            generated.append(BeamCandidate.seed(
                MacroProgram(checkpoint.lane, checkpoint, (donor_phase,), cap), SeedSource.HISTORICAL_MOTIF))
            for mask in ALL_MASKS:
                if set(mask).issubset(motif.supported_families):
                    masked = replace(_phase_for_motif(motif, mask),
                                     termination=Termination(TerminationType.FIXED_CAP, cap))
                    generated.append(BeamCandidate.seed(
                        MacroProgram(checkpoint.lane, checkpoint, (masked,), cap), SeedSource.MASKED_DONOR_PHASE))
        primitive_cap = min(2, remaining)
        primitive = Phase(ActionSource.PARAMETERIZED_PRIMITIVE, (ActionFamily.MARKET,),
                          Termination(TerminationType.FIXED_CAP, primitive_cap), primitive_name="LIQUIDITY_REBALANCE",
                          parameters={"enabled": True, "mode": "WHEAT", "quantity": 1})
        generated.append(BeamCandidate.seed(
            MacroProgram(checkpoint.lane, checkpoint, (primitive,), primitive_cap), SeedSource.PARAMETERIZED_PRIMITIVE))
    return tuple(sorted(generated, key=lambda item: (item.seed_source.value, item.checkpoint.rank_key(), item.program.semantic_id)))


def _program_with_phases(program: MacroProgram, phases: Sequence[Phase]) -> MacroProgram | None:
    total = sum(phase.termination.max_steps for phase in phases)
    if not 1 <= len(phases) <= MAX_PHASES or total > ACTION_HORIZON - program.initiation.step:
        return None
    return MacroProgram(program.lane, program.initiation, tuple(phases), max(program.max_total_steps, total),
                        provenance=program.provenance)


def _replace_phase(program: MacroProgram, index: int, phase: Phase) -> MacroProgram | None:
    phases = list(program.phases); phases[index] = phase
    return _program_with_phases(program, phases)


def _event_caps(program: MacroProgram, phase_index: int, events: Sequence[Checkpoint]) -> tuple[int, ...]:
    start = program.initiation.step + sum(item.termination.max_steps for item in program.phases[:phase_index])
    boundary = start + program.phases[phase_index].termination.max_steps
    candidates = sorted({event.step - start for event in events if event.step > start and event.step != boundary})
    before = [cap for cap in candidates if cap < program.phases[phase_index].termination.max_steps]
    after = [cap for cap in candidates if cap > program.phases[phase_index].termination.max_steps]
    return tuple(([before[-1]] if before else []) + ([after[0]] if after else []))


def expand_program(
    program: MacroProgram, operator: ExpansionOperator, motif_library: Mapping[str, Motif],
    adjacent_events: Sequence[Checkpoint], *, max_phases: int = MAX_PHASES,
) -> tuple[MacroProgram, ...]:
    operator = ExpansionOperator(operator)
    generated: list[MacroProgram] = []
    library = _motifs(motif_library)
    if operator is ExpansionOperator.INSERT_PHASE:
        if len(program.phases) >= max_phases:
            return ()
        for position in range(len(program.phases) + 1):
            for motif in library:
                candidate = _program_with_phases(
                    program, program.phases[:position] + (_phase_for_motif(motif, motif.supported_families),) + program.phases[position:])
                if candidate is not None:
                    generated.append(candidate)
    elif operator in {ExpansionOperator.REPLACE_PHASE_SOURCE, ExpansionOperator.SWAP_SEMANTIC_DONOR_OR_MOTIF_PHASE}:
        for index, old in enumerate(program.phases):
            if old.action_source not in {ActionSource.DONOR_BLOCK, ActionSource.SEMANTIC_SPLICE}:
                continue
            for motif in library:
                if (motif.semantic_id == old.motif_id
                        or not set(old.action_families).issubset(motif.supported_families)
                        or motif.source_length < old.termination.max_steps):
                    continue
                source = old.action_source if operator is ExpansionOperator.SWAP_SEMANTIC_DONOR_OR_MOTIF_PHASE else ActionSource.SEMANTIC_SPLICE
                candidate = _replace_phase(program, index, replace(old, action_source=source, motif_id=motif.semantic_id))
                if candidate is not None:
                    generated.append(candidate)
    elif operator is ExpansionOperator.CHANGE_ACTION_FAMILY_MASK:
        by_id = {item.semantic_id: item for item in library}
        for index, old in enumerate(program.phases):
            supported = (ActionFamily.MARKET,) if old.action_source is ActionSource.PARAMETERIZED_PRIMITIVE else (
                by_id[old.motif_id].supported_families if old.motif_id in by_id else ())
            for mask in ALL_MASKS:
                if mask != old.action_families and set(mask).issubset(supported):
                    candidate = _replace_phase(program, index, replace(old, action_families=mask))
                    if candidate is not None:
                        generated.append(candidate)
    elif operator in {ExpansionOperator.SHIFT_PHASE_BOUNDARY_TO_ADJACENT_EVENT,
                      ExpansionOperator.TRUNCATE_AT_EVENT, ExpansionOperator.EXTEND_TO_EVENT}:
        for index, old in enumerate(program.phases):
            for cap in _event_caps(program, index, adjacent_events):
                if operator is ExpansionOperator.TRUNCATE_AT_EVENT and cap >= old.termination.max_steps:
                    continue
                if operator is ExpansionOperator.EXTEND_TO_EVENT and cap <= old.termination.max_steps:
                    continue
                candidate = _replace_phase(program, index, replace(old, termination=replace(old.termination, max_steps=cap)))
                if candidate is not None:
                    generated.append(candidate)
    elif operator is ExpansionOperator.MUTATE_BOUNDED_NUMERIC_PARAMETER:
        for index, old in enumerate(program.phases):
            if old.action_source is not ActionSource.PARAMETERIZED_PRIMITIVE:
                continue
            parameters = dict(old.parameters)
            for key in sorted(parameters):
                value = parameters[key]
                if type(value) is int:
                    for mutated in (value - 1, value + 1):
                        if 1 <= mutated <= 10:
                            changed = dict(parameters); changed[key] = mutated
                            candidate = _replace_phase(program, index, replace(old, parameters=changed))
                            if candidate is not None:
                                generated.append(candidate)
    unique = {item.semantic_id: item for item in generated if item.semantic_id != program.semantic_id}
    return tuple(unique[key] for key in sorted(unique))


def deduplicate_candidates(candidates: Iterable[BeamCandidate]) -> tuple[tuple[BeamCandidate, ...], int]:
    values = tuple(candidates)
    unique: dict[str, BeamCandidate] = {}
    for candidate in sorted(values, key=lambda item: (item.program.semantic_id, item.lineage.operator, item.lineage.parent_id or "")):
        unique.setdefault(candidate.program.semantic_id, candidate)
    return tuple(unique.values()), len(values) - len(unique)


def _entry(program: MacroProgram, effect: EffectVector, candidate_id: str | None = None) -> ArchiveEntry:
    target = next((block for block in effect.blocks if block.role is DatasetRole.TARGET), None)
    if target is None:
        raise ValueError("beam effects require a TARGET block")
    return ArchiveEntry(
        candidate_id=candidate_id or program.semantic_id, program_id=program.semantic_id,
        effect_signature=effect_signature(effect), cell_key=make_cell_key(program.initiation, effect),
        semantic_valid=True, target_win_to_loss=sum(target.win_to_loss),
        target_harm_count=sum(value < 0 for value in target.margin_deltas),
        target_loss_to_win=sum(target.loss_to_win),
        required_target_positive_support=sum(value > 0 for value in target.candidate_margins),
        independent_positive_support=0, both_seat_support=0, worst_delta=min(target.margin_deltas),
        routability=None, program_simplicity=len(program.phases) + sum(len(phase.parameters) for phase in program.phases),
        aggregate_margin=sum(target.margin_deltas), loss_to_win_pattern=target.loss_to_win,
    )


@dataclass(frozen=True)
class BeamResult:
    archive: Archive
    candidates: tuple[BeamCandidate, ...]
    metrics: dict[str, Any]
    terminal_reason: str

    def entry_for_effect(self, program: MacroProgram, effect: EffectVector, label: str,
                         candidate_id: str | None = None) -> ArchiveEntry:
        return _entry(program, effect, candidate_id or f"{program.semantic_id}-{label}")


def _quality(archive: Archive) -> tuple[int, int, int]:
    safe = [entry for entry in archive.entries() if entry.target_win_to_loss == 0]
    return max(((entry.target_loss_to_win, entry.required_target_positive_support, entry.worst_delta)
                for entry in safe), default=(0, 0, -(1 << 31)))


def _lineage_parameters(parent: MacroProgram, result: MacroProgram) -> tuple[tuple[str, Any], ...]:
    changed = [index for index in range(max(len(parent.phases), len(result.phases)))
               if index >= len(parent.phases) or index >= len(result.phases)
               or parent.phases[index] != result.phases[index]]
    return (
        ("changed_phase_positions", ",".join(map(str, changed))),
        ("result_phase_count", len(result.phases)),
        ("result_phase_digest", stable_identity(
            "RLRMB-LINEAGE-PHASE-EDIT-V1", [phase.to_document() for phase in result.phases]
        )),
        ("result_max_total_steps", result.max_total_steps),
    )


# ---------------------------------------------------------------------------
# Phase 16.2 — parallel candidate construction workers (batched for IPC efficiency)
#
# Tasks are coarsened to min(pool.workers, n_items) batches so that IPC
# round-trip cost is amortized over a much larger work unit per task.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _SeedTask:
    """Generate seeds for a batch of checkpoints in one worker call.

    ``per_cp_motifs[i]`` is the motif tuple for ``checkpoints[i]``; if the
    tuple is empty the shared ``shared_motifs`` are used for that checkpoint.
    """
    checkpoints: tuple[Checkpoint, ...]
    per_cp_motifs: tuple[tuple[Motif, ...], ...]  # parallel to checkpoints
    shared_motifs: tuple[Motif, ...]
    config: BeamConfig


def _seed_task_worker(task: "_SeedTask") -> "tuple[BeamCandidate, ...]":
    """Generate seeds for all checkpoints in the batch; return a pre-sorted tuple."""
    shared_lib: dict[str, Motif] = {m.semantic_id: m for m in task.shared_motifs}
    result: list[BeamCandidate] = []
    for i, cp in enumerate(task.checkpoints):
        if i < len(task.per_cp_motifs) and task.per_cp_motifs[i]:
            lib: dict[str, Motif] = {m.semantic_id: m for m in task.per_cp_motifs[i]}
        else:
            lib = shared_lib
        result.extend(generate_seeds((cp,), lib, task.config))
    return tuple(sorted(
        result,
        key=lambda item: (item.seed_source.value, item.checkpoint.rank_key(), item.program.semantic_id),
    ))


@dataclass(frozen=True)
class _ExpansionTask:
    """Expand a batch of parents across all operators in one worker call.

    ``per_parent_motifs[i]`` is the motif tuple for ``parents[i]``; if the
    tuple is empty the shared ``shared_motifs`` are used for that parent.
    """
    parents: tuple[MacroProgram, ...]
    parent_seed_sources: tuple[SeedSource, ...]  # parallel to parents
    per_parent_motifs: tuple[tuple[Motif, ...], ...]  # parallel to parents
    shared_motifs: tuple[Motif, ...]
    operators: tuple[ExpansionOperator, ...]
    adjacent_events: tuple[Checkpoint, ...]
    max_phases: int
    depth: int
    family_ref: str


def _expand_task_worker(
    task: "_ExpansionTask",
) -> "tuple[tuple[str, tuple[BeamCandidate, ...]], ...]":
    """Expand all parents in the batch; return flat (operator_value, candidates) pairs."""
    shared_lib: dict[str, Motif] = {m.semantic_id: m for m in task.shared_motifs}
    result: list[tuple[str, tuple[BeamCandidate, ...]]] = []
    for pi, parent in enumerate(task.parents):
        if pi < len(task.per_parent_motifs) and task.per_parent_motifs[pi]:
            lib: dict[str, Motif] = {m.semantic_id: m for m in task.per_parent_motifs[pi]}
        else:
            lib = shared_lib
        seed_src = task.parent_seed_sources[pi] if pi < len(task.parent_seed_sources) else SeedSource.HISTORICAL_MOTIF
        for operator in task.operators:
            programs = expand_program(
                parent, operator, lib, task.adjacent_events, max_phases=task.max_phases
            )
            candidates: list[BeamCandidate] = []
            for item in programs:
                lineage = Lineage(
                    parent.semantic_id, operator.value,
                    _lineage_parameters(parent, item), item.semantic_id, task.depth,
                )
                candidates.append(
                    BeamCandidate(
                        item, item.initiation, task.depth, seed_src, lineage,
                        family_ref=task.family_ref,
                    )
                )
            result.append((operator.value, tuple(candidates)))
    return tuple(result)


def _chunk(items: tuple, n_workers: int) -> list[tuple]:
    """Split ``items`` into at most ``n_workers`` contiguous chunks."""
    if not items:
        return []
    n_tasks = min(n_workers, len(items))
    chunk_size = max(1, (len(items) + n_tasks - 1) // n_tasks)
    return [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]


def run_beam(
    family_ref: str, checkpoints: Sequence[Checkpoint], motif_library: Mapping[str, Motif],
    evaluator: Callable[[Sequence[MacroProgram], Mapping[str, Motif]], BatchEvaluation],
    config: BeamConfig, archive: Archive | None = None,
    checkpoint_motifs: Mapping[int, Mapping[str, Motif]] | None = None,
    candidate_pool: Any | None = None,
) -> BeamResult:
    if type(family_ref) is not str or not family_ref:
        raise ValueError("family_ref must be non-empty offline bookkeeping")
    archive = archive or Archive()
    _cp_pool_tasks = 0; _cp_pool_busy = 0.0; _cp_pool_wall = 0.0
    _ex_pool_tasks = 0; _ex_pool_busy = 0.0; _ex_pool_wall = 0.0
    preparation_started = time.perf_counter()
    _shared_motifs = tuple(motif_library.values())
    if checkpoint_motifs is None:
        if candidate_pool is not None:
            _sorted_cps = tuple(sorted(checkpoints, key=lambda item: item.rank_key()))
            _cp_chunks = _chunk(_sorted_cps, candidate_pool.workers)
            _seed_tasks = [
                _SeedTask(chunk, (), _shared_motifs, config) for chunk in _cp_chunks
            ]
            _seed_batches, _sm = candidate_pool.map_ordered(_seed_task_worker, _seed_tasks)
            seeds = tuple(sorted(
                _iterchain.from_iterable(_seed_batches),
                key=lambda item: (item.seed_source.value, item.checkpoint.rank_key(), item.program.semantic_id),
            ))
            _cp_pool_tasks += _sm.tasks; _cp_pool_busy += _sm.worker_busy_seconds; _cp_pool_wall += _sm.wall_seconds
        else:
            seeds = generate_seeds(checkpoints, motif_library, config)
    else:
        missing = [checkpoint.step for checkpoint in checkpoints if checkpoint.step not in checkpoint_motifs]
        if missing:
            raise ValueError(f"checkpoint motif mapping is incomplete: {missing}")
        if candidate_pool is not None:
            _sorted_cps = tuple(checkpoints)
            _cp_chunks = _chunk(_sorted_cps, candidate_pool.workers)
            _seed_tasks = [
                _SeedTask(
                    chunk,
                    tuple(tuple(checkpoint_motifs[cp.step].values()) for cp in chunk),
                    _shared_motifs,
                    config,
                )
                for chunk in _cp_chunks
            ]
            _seed_batches, _sm = candidate_pool.map_ordered(_seed_task_worker, _seed_tasks)
            seeds = tuple(sorted(
                _iterchain.from_iterable(_seed_batches),
                key=lambda item: (item.seed_source.value, item.checkpoint.rank_key(), item.program.semantic_id),
            ))
            _cp_pool_tasks += _sm.tasks; _cp_pool_busy += _sm.worker_busy_seconds; _cp_pool_wall += _sm.wall_seconds
        else:
            seeds = tuple(item for checkpoint in checkpoints
                          for item in generate_seeds((checkpoint,), checkpoint_motifs[checkpoint.step], config))
    raw_candidates = len(seeds)
    dedup_started = time.perf_counter()
    pending, duplicates = deduplicate_candidates(seeds)
    structural_dedup_seconds = time.perf_counter() - dedup_started
    host_preparation_seconds = time.perf_counter() - preparation_started
    all_candidates: list[BeamCandidate] = []
    program_states: dict[str, BeamCandidate] = {}
    games = launches = evaluated_count = 0
    host_seconds = cuda_ms = 0.0
    archive_update_seconds = serialization_seconds = result_parsing_seconds = subprocess_seconds = 0.0
    expansions_by_operator = {item.value: 0 for item in ExpansionOperator}
    expansions_by_depth: dict[str, int] = {}
    unique_effects: set[str] = set()
    stagnant = 0
    terminal = "MAX_DEPTH"
    depth_reached = 0

    for depth in range(config.max_depth + 1):
        if depth > 0:
            expansion_started = time.perf_counter()
            parents = tuple(sorted(program_states.values(), key=lambda item: (
                item.program.lane.value, item.effect and effect_signature(item.effect), item.program.semantic_id)))
            if candidate_pool is not None:
                _parent_chunks = _chunk(parents, candidate_pool.workers)
                _cp_map = checkpoint_motifs or {}
                _exp_tasks = [
                    _ExpansionTask(
                        tuple(p.program for p in chunk),
                        tuple(p.seed_source for p in chunk),
                        tuple(
                            tuple(_cp_map.get(p.program.initiation.step, motif_library).values())
                            for p in chunk
                        ),
                        _shared_motifs,
                        config.allowed_operators,
                        tuple(checkpoints), config.max_phases, depth, family_ref,
                    )
                    for chunk in _parent_chunks
                ]
                _exp_results, _em = candidate_pool.map_ordered(_expand_task_worker, _exp_tasks)
                _ex_pool_tasks += _em.tasks; _ex_pool_busy += _em.worker_busy_seconds; _ex_pool_wall += _em.wall_seconds
                raw: list[BeamCandidate] = []
                for task_output in _exp_results:
                    for op_value, cands in task_output:
                        expansions_by_operator[op_value] += len(cands)
                        raw.extend(cands)
            else:
                raw = []
                for parent in parents:
                    expansion_motifs = (checkpoint_motifs or {}).get(parent.program.initiation.step, motif_library)
                    for operator in config.allowed_operators:
                        programs = expand_program(parent.program, operator, expansion_motifs, checkpoints, max_phases=config.max_phases)
                        expansions_by_operator[operator.value] += len(programs)
                        for item in programs:
                            lineage = Lineage(parent.program.semantic_id, operator.value,
                                              _lineage_parameters(parent.program, item), item.semantic_id, depth)
                            raw.append(BeamCandidate(item, item.initiation, depth, parent.seed_source, lineage,
                                                     family_ref=family_ref))
            raw_candidates += len(raw)
            expansions_by_depth[str(depth)] = len(raw)
            dedup_started = time.perf_counter()
            pending, removed = deduplicate_candidates(raw)
            structural_dedup_seconds += time.perf_counter() - dedup_started
            host_preparation_seconds += time.perf_counter() - expansion_started
            duplicates += removed
            if not pending:
                terminal = "NO_EXPANSIONS"
                break
        known = {item.program.semantic_id for item in all_candidates}
        duplicates += sum(item.program.semantic_id in known for item in pending)
        pending = tuple(item for item in pending if item.program.semantic_id not in known)
        case_count = getattr(evaluator, "case_count", 1)
        projected = getattr(evaluator, "projected_games", lambda count: count * case_count)
        allowed = min(len(pending), config.max_candidates_per_expansion)
        while allowed and games + projected(allowed) > config.cuda_game_budget:
            allowed -= 1
        pending = pending[:allowed]
        if not pending:
            terminal = "BUDGET_EXHAUSTED"
            break
        before_cells = archive.metrics()["occupied_cells"]
        before_effects = len(unique_effects)
        before_quality = _quality(archive)
        batch = evaluator(tuple(item.program for item in pending), motif_library)
        profile = getattr(evaluator, "last_profile", {})
        serialization_seconds += float(profile.get("serialization_seconds", 0.0))
        result_parsing_seconds += float(profile.get("result_parsing_seconds", 0.0))
        subprocess_seconds += float(profile.get("subprocess_seconds", batch.host_seconds))
        if games + batch.games > config.cuda_game_budget:
            raise RuntimeError("evaluator exceeded projected CUDA game budget")
        games += batch.games; launches += batch.launches; evaluated_count += len(batch.programs)
        host_seconds += batch.host_seconds; cuda_ms += batch.cuda_ms
        if len(batch.programs) != len(pending):
            raise RuntimeError("evaluator result count mismatch")
        archive_started = time.perf_counter()
        for source, evaluated in zip(pending, batch.programs):
            if evaluated.program.semantic_id != source.program.semantic_id:
                raise RuntimeError("SEMANTIC_MISMATCH: evaluator reordered candidates")
            if not evaluated.valid:
                state = replace(source, fault=evaluated.fault, failure_class=FailureClass.CUDA_FAULT,
                                cuda_games=case_count)
            else:
                entry = _entry(source.program, evaluated.effect)
                insertion = archive.insert(entry)
                unique_effects.add(entry.effect_signature)
                state = replace(source, effect=evaluated.effect, archive_result=insertion, cuda_games=case_count)
            all_candidates.append(state)
        archive_update_seconds += time.perf_counter() - archive_started
        elite_ids = {entry.program_id for entry in archive.entries()}
        program_states = {item.program.semantic_id: item for item in all_candidates
                          if item.effect is not None and item.program.semantic_id in elite_ids}
        depth_reached = depth
        novelty = (archive.metrics()["occupied_cells"] - before_cells) + (len(unique_effects) - before_effects)
        quality = _quality(archive)
        quality_improvement = int(quality > before_quality)
        if novelty < config.minimum_novelty_improvement and quality_improvement < config.minimum_target_quality_improvement:
            stagnant += 1
        else:
            stagnant = 0
        if stagnant >= config.stagnation_rounds:
            terminal = "STAGNATION"
            break
        if games >= config.cuda_game_budget:
            terminal = "BUDGET_EXHAUSTED"
            break

    archive_metrics = archive.metrics()
    target_blocks = [next(block for block in item.effect.blocks if block.role is DatasetRole.TARGET)
                     for item in all_candidates if item.effect is not None]
    metrics = {
        "schema_version": BEAM_SCHEMA_VERSION,
        "config_id": config.semantic_id,
        "checkpoints_considered": len(checkpoints), "seeds_generated": len(seeds),
        "expansions_by_operator": expansions_by_operator, "expansions_by_depth": expansions_by_depth,
        "raw_candidates": raw_candidates, "structural_duplicates_removed": duplicates,
        "unique_structural_ids": len({item.program.semantic_id for item in all_candidates}),
        "cuda_candidates_evaluated": evaluated_count, "cuda_cases": getattr(evaluator, "case_count", 1),
        "cuda_games": games,
        "unique_effect_signatures": len(unique_effects), "archive_occupied_cells": archive_metrics["occupied_cells"],
        "archive_insertions": archive_metrics["inserted"], "archive_replacements": archive_metrics["replacements"],
        "novelty_rate": archive_metrics["novelty_rate"], "target_solving_cells": archive_metrics["target_solving_cells"],
        "target_positives": max((sum(value > 0 for value in block.candidate_margins) for block in target_blocks), default=0),
        "target_loss_to_win": max((sum(block.loss_to_win) for block in target_blocks), default=0),
        "target_win_to_loss": min((sum(block.win_to_loss) for block in target_blocks), default=0),
        "best_worst_delta": max((min(block.margin_deltas) for block in target_blocks), default=0),
        "distinct_loss_to_win_patterns": archive_metrics["distinct_retained_l_to_w_patterns"],
        "evaluator_launches": launches, "host_seconds": host_seconds, "cuda_ms": cuda_ms,
        "host_preparation_seconds": host_preparation_seconds,
        "structural_dedup_seconds": structural_dedup_seconds,
        "archive_update_seconds": archive_update_seconds,
        "serialization_seconds": serialization_seconds,
        "result_parsing_seconds": result_parsing_seconds,
        "subprocess_seconds": subprocess_seconds,
        "games_per_second": 0.0 if host_seconds <= 0 else games / host_seconds,
        "depth_reached": depth_reached, "terminal_reason": terminal,
        "failure_counts": {
            failure.value: sum(item.failure_class is failure for item in all_candidates)
            for failure in (FailureClass.CUDA_FAULT, FailureClass.SEMANTIC_MISMATCH)
        },
        "candidate_pool_workers": getattr(candidate_pool, "workers", 0) if candidate_pool is not None else 0,
        "candidate_pool_tasks": _cp_pool_tasks + _ex_pool_tasks,
        "candidate_pool_worker_busy_seconds": _cp_pool_busy + _ex_pool_busy,
        "candidate_pool_wall_seconds": _cp_pool_wall + _ex_pool_wall,
    }
    return BeamResult(archive, tuple(all_candidates), metrics, terminal)


def _process_cuda_results(
    programs: Sequence[MacroProgram],
    rows: tuple,
    cases: int,
    gpu_ms: float,
    encode_seconds: float,
    subprocess_seconds: float,
    parse_seconds: float,
) -> BatchEvaluation:
    """Shared post-processing for both legacy and persistent evaluators."""
    baseline_rows = rows[:cases]
    baseline = tuple(row.own_money - row.opp_money for row in baseline_rows)
    panel_id = stable_identity("RLRMB-BEAM-TARGET-PANEL-V1", [f"slot-{i:04d}" for i in range(cases)], baseline)
    evaluated = []
    for index, program in enumerate(programs, start=1):
        selected = rows[index * cases:(index + 1) * cases]
        if any(row.semantic_id != program.semantic_id for row in selected):
            raise RuntimeError("SEMANTIC_MISMATCH: CUDA semantic ID mismatch")
        margins = tuple(row.own_money - row.opp_money for row in selected)
        block = EffectBlock(DatasetRole.TARGET, panel_id, baseline,
                            tuple(value - base for value, base in zip(margins, baseline)),
                            tuple(row.action_changes for row in selected))
        faults = tuple(row.fault for row in selected if row.fault is not FaultCode.NONE)
        evaluated.append(EvaluatedProgram(program, EffectVector((block,)), not faults,
                                          None if not faults else faults[0].name))
    total_programs = 1 + len(programs)  # parent + candidates
    return BatchEvaluation(tuple(evaluated), cases, total_programs * cases, 1,
                           encode_seconds + subprocess_seconds + parse_seconds, gpu_ms)


class CudaBatchEvaluator:
    """One-invocation candidate-by-case adapter over the unchanged Phase-4 binary."""

    def __init__(self, *, binary: str | Path, asset_dir: str | Path, cases_path: str | Path,
                 feed_slot: int = 5, feed_count: int = 13, strategic_route_slot: int = 19) -> None:
        self.binary = Path(binary); self.asset_dir = Path(asset_dir); self.cases_path = Path(cases_path)
        if not self.binary.exists() or not self.cases_path.exists():
            raise ValueError("CUDA evaluator binary and case panel must exist")
        self.case_count = len([line for line in self.cases_path.read_text(encoding="utf-8").splitlines() if line.strip()])
        if self.case_count <= 0:
            raise ValueError("CUDA case panel must be non-empty")
        self.feed_slot = feed_slot; self.feed_count = feed_count; self.strategic_route_slot = strategic_route_slot
        self.last_profile: dict[str, float] = {}

    def projected_games(self, candidate_count: int) -> int:
        return (candidate_count + 1) * self.case_count

    def __call__(self, programs: Sequence[MacroProgram], motif_library: Mapping[str, Motif]) -> BatchEvaluation:
        if not programs:
            return BatchEvaluation((), self.case_count, 0, 0, 0.0, 0.0)
        first = programs[0]
        parent = MacroProgram(first.lane, first.initiation,
                              (Phase(ActionSource.PARENT, (), Termination(TerminationType.MAX_STEPS, 1)),), 1)
        batch_programs = (parent, *programs)
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory); packed = work / "programs.rlrb"; output = work / "results.rlr"
            encode_started = time.perf_counter()
            packed.write_bytes(encode_device_batch(batch_programs, motif_library))
            encode_seconds = time.perf_counter() - encode_started
            command = [str(self.binary), "--controller", str(self.asset_dir / "controller.bkgc"),
                       "--routes", str(self.asset_dir / "routes.bkv6"), "--legacy-routes", str(self.asset_dir / "legacy_routes.bkv6"),
                       "--cases", str(self.cases_path), "--programs", str(packed), "--out", str(output),
                       "--feed-slot", str(self.feed_slot), "--feed-count", str(self.feed_count),
                       "--strategic-route-slot", str(self.strategic_route_slot)]
            started = time.perf_counter()
            completed = subprocess.run(command, capture_output=True, text=True)
            subprocess_seconds = time.perf_counter() - started
            if not output.exists():
                raise RuntimeError(f"CUDA evaluator failed: {completed.stderr.strip()}")
            parse_started = time.perf_counter()
            count, cases, rows = decode_device_results(output.read_bytes())
            if (count, cases) != (len(batch_programs), self.case_count):
                raise RuntimeError("CUDA result dimensions mismatch")
            match = re.search(r"gpu_ms=([0-9.]+)", completed.stdout)
            gpu_ms = float(match.group(1)) if match else 0.0
            parse_seconds = time.perf_counter() - parse_started
            self.last_profile = {"serialization_seconds": encode_seconds,
                                 "subprocess_seconds": subprocess_seconds,
                                 "result_parsing_seconds": parse_seconds}
        return _process_cuda_results(programs, rows, cases, gpu_ms, encode_seconds, subprocess_seconds, parse_seconds)


# ---------------------------------------------------------------------------
# Phase 16.4 — Persistent CUDA evaluator process
#
# Eliminates per-batch CUDA context startup by keeping the evaluator process
# alive across all beam depths and families within a single cloud run.
# The subprocess loads controller/routes once; case panels are cached by path
# inside the process. Request/result association uses an explicit integer ID.
# ---------------------------------------------------------------------------

import io as _io
import struct as _struct

_RESULT_HEADER_FMT = _struct.Struct("<8sIII")  # magic, version, candidates, cases
_RESULT_RECORD_FMT = _struct.Struct("<iiHHBBBB32s")


class PersistentCudaEvaluator:
    """Persistent CUDA subprocess reused across all evaluator calls in a cloud run.

    The binary is invoked once with ``--persist``; it loads controller/routes/
    legacy-routes once and accepts batches from stdin until a ``SHUTDOWN``
    sentinel. Case panels are cached inside the process by absolute path, so
    only the first request for each panel incurs tape-loading cost.

    Satisfies the Phase-16.4 hard constraints:
    * Exactly one GPU owner (single subprocess, one CUDA context).
    * Deterministic request/result association (monotonic integer request IDs).
    * Fail-closed: if the subprocess dies, the next call raises RuntimeError.
    * Central budget authorization is unchanged: ``AccountedEvaluator`` checks
      the ledger before each ``__call__``; the subprocess never executes work
      before that check completes in the coordinator.
    * Clean shutdown via ``SHUTDOWN`` sentinel or ``stop()`` / ``__exit__``.
    """

    def __init__(self, *, binary: str | Path, asset_dir: str | Path,
                 feed_slot: int = 5, feed_count: int = 13,
                 strategic_route_slot: int = 19) -> None:
        self.binary = Path(binary)
        self.asset_dir = Path(asset_dir)
        self._feed_slot = feed_slot
        self._feed_count = feed_count
        self._strategic_route_slot = strategic_route_slot
        self._proc: "subprocess.Popen | None" = None
        self._next_id: int = 0
        # Metrics
        self.evaluator_process_starts: int = 0
        self.evaluator_requests: int = 0
        self.persistent_evaluator_reuses: int = 0
        self.total_serialization_seconds: float = 0.0
        self.total_result_parsing_seconds: float = 0.0
        self.total_evaluator_host_seconds: float = 0.0

    def _build_command(self) -> list[str]:
        return [
            str(self.binary),
            "--controller", str(self.asset_dir / "controller.bkgc"),
            "--routes", str(self.asset_dir / "routes.bkv6"),
            "--legacy-routes", str(self.asset_dir / "legacy_routes.bkv6"),
            "--persist",
            "--feed-slot", str(self._feed_slot),
            "--feed-count", str(self._feed_count),
            "--strategic-route-slot", str(self._strategic_route_slot),
        ]

    def start(self) -> None:
        """Start the persistent subprocess (called lazily on first request)."""
        self._proc = subprocess.Popen(
            self._build_command(),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self.evaluator_process_starts += 1

    def _check_alive(self) -> None:
        """Raise if the subprocess has exited; start it if it was never started."""
        if self._proc is not None and self._proc.poll() is not None:
            stderr = b""
            try:
                stderr = self._proc.stderr.read()
            except Exception:
                pass
            self._proc = None
            raise RuntimeError(
                f"persistent CUDA evaluator process exited unexpectedly: "
                f"{stderr.decode('utf-8', errors='replace').strip()}"
            )
        if self._proc is None:
            self.start()

    def submit_batch(
        self,
        cases_path: Path,
        programs: Sequence[MacroProgram],
        motif_library: Mapping[str, Motif],
        case_count: int,
    ) -> BatchEvaluation:
        """Submit one batch to the persistent process and return the result."""
        self._check_alive()
        first = programs[0]
        parent = MacroProgram(first.lane, first.initiation,
                              (Phase(ActionSource.PARENT, (), Termination(TerminationType.MAX_STEPS, 1)),), 1)
        batch_programs = (parent, *programs)

        encode_started = time.perf_counter()
        batch_bytes = encode_device_batch(batch_programs, motif_library)
        encode_seconds = time.perf_counter() - encode_started

        rid = self._next_id
        self._next_id = (self._next_id + 1) & 0xFFFFFFFF
        header_line = f"BATCH {rid} {len(batch_bytes)} {cases_path.resolve()}\n".encode("utf-8")

        host_started = time.perf_counter()
        self._proc.stdin.write(header_line + batch_bytes)  # type: ignore[union-attr]
        self._proc.stdin.flush()  # type: ignore[union-attr]

        response_line = self._proc.stdout.readline()  # type: ignore[union-attr]
        if not response_line:
            self._proc = None
            raise RuntimeError("persistent CUDA evaluator closed stdout unexpectedly")

        parts = response_line.decode("utf-8").split()
        if len(parts) < 4 or parts[0] != "RESULT":
            self._proc = None
            raise RuntimeError(f"persistent evaluator bad response header: {response_line!r}")
        if int(parts[1]) != rid:
            self._proc = None
            raise RuntimeError(f"persistent evaluator request/result ID mismatch: sent {rid}, got {parts[1]}")
        result_bytes = int(parts[2])
        gpu_ms = float(parts[3])

        raw_result = self._proc.stdout.read(result_bytes)  # type: ignore[union-attr]
        if len(raw_result) != result_bytes:
            self._proc = None
            raise RuntimeError("persistent evaluator returned short result")

        parse_started = time.perf_counter()
        host_seconds = parse_started - host_started
        count, cases, rows = decode_device_results(raw_result)
        if (count, cases) != (len(batch_programs), case_count):
            raise RuntimeError("persistent evaluator result dimensions mismatch")
        parse_seconds = time.perf_counter() - parse_started

        self.evaluator_requests += 1
        if self.evaluator_requests > 1:
            self.persistent_evaluator_reuses += 1
        self.total_serialization_seconds += encode_seconds
        self.total_result_parsing_seconds += parse_seconds
        self.total_evaluator_host_seconds += host_seconds

        return _process_cuda_results(programs, rows, cases, gpu_ms, encode_seconds, host_seconds, parse_seconds)

    def bind(self, cases_path: "str | Path", *,
             feed_slot: int = 5, feed_count: int = 13,
             strategic_route_slot: int = 19) -> "_PersistentBoundEvaluator":
        """Return a family-scoped wrapper with the same API as CudaBatchEvaluator."""
        cases_path = Path(cases_path)
        case_count = len([ln for ln in cases_path.read_text(encoding="utf-8").splitlines() if ln.strip()])
        return _PersistentBoundEvaluator(self, cases_path, case_count)

    def stop(self) -> None:
        """Send shutdown sentinel and wait for the subprocess to exit."""
        if self._proc is not None:
            try:
                self._proc.stdin.write(b"SHUTDOWN\n")  # type: ignore[union-attr]
                self._proc.stdin.flush()  # type: ignore[union-attr]
                self._proc.wait(timeout=15)
            except Exception:
                try:
                    self._proc.kill()
                except Exception:
                    pass
            finally:
                self._proc = None

    def __enter__(self) -> "PersistentCudaEvaluator":
        self.start()
        return self

    def __exit__(self, *args: object) -> None:
        self.stop()

    def metrics_document(self) -> dict[str, Any]:
        return {
            "evaluator_process_starts": self.evaluator_process_starts,
            "evaluator_requests": self.evaluator_requests,
            "persistent_evaluator_reuses": self.persistent_evaluator_reuses,
            "total_serialization_seconds": round(self.total_serialization_seconds, 6),
            "total_result_parsing_seconds": round(self.total_result_parsing_seconds, 6),
            "total_evaluator_host_seconds": round(self.total_evaluator_host_seconds, 6),
        }


class _PersistentBoundEvaluator:
    """Drop-in replacement for CudaBatchEvaluator, delegates to a shared PersistentCudaEvaluator.

    This allows the same persistent process to serve multiple case panels
    (families) while satisfying the single-GPU-owner constraint.
    """

    def __init__(self, persistent: PersistentCudaEvaluator, cases_path: Path, case_count: int) -> None:
        self._persistent = persistent
        self.cases_path = cases_path
        self.case_count = case_count
        self.last_profile: dict[str, float] = {}

    def projected_games(self, candidate_count: int) -> int:
        return (candidate_count + 1) * self.case_count

    def __call__(self, programs: Sequence[MacroProgram], motif_library: Mapping[str, Motif]) -> BatchEvaluation:
        if not programs:
            return BatchEvaluation((), self.case_count, 0, 0, 0.0, 0.0)
        result = self._persistent.submit_batch(self.cases_path, programs, motif_library, self.case_count)
        self.last_profile = {
            "serialization_seconds": self._persistent.total_serialization_seconds,
            "subprocess_seconds": self._persistent.total_evaluator_host_seconds,
            "result_parsing_seconds": self._persistent.total_result_parsing_seconds,
        }
        return result
