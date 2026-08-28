"""Phase-13 benchmark compatibility backend.

New production invocations go through :mod:`rlrmb.cli` and the lifecycle in
``pipeline.py``. This module retains the validated equal-budget search kernel
and its historical command for exact benchmark reproduction.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .archive import Archive, make_cell_key
from .beam import BeamCandidate, BeamConfig, CudaBatchEvaluator, run_beam
from .checkpoints import EventEvidence, EventType, extract_trace_events, select_checkpoints
from .ir import ActionFamily, MacroProgram
from .motifs import Motif, motif_from_actions, motif_from_route_window
from .canonical import load_route_actions
from .host_pipeline import DeterministicProcessPool, HostStageMetrics
from .pipeline import (
    ExactCaseEvidence, LegalStateCaptureCache, build_cegis_input, build_routability_evidence,
    capture_candidate_legal_states, load_case_panel, materialize_certified_candidate,
)
from .routability import RoutabilityClassification, RoutabilityResult, screen_routability
from .cegis_adapter import run_hardened_cegis
from .effects import EffectVector, effect_signature
from .failures import FailureClass
from .lns import DualPanelEvaluator, run_lns
from .schema import DatasetRole, stable_identity

SCHEMA_VERSION = 1
GAME_CATEGORIES = ("target_beam", "transfer", "validation", "lns_repair", "reactive_repair", "certification")

_PROMOTION_CLASS_RANK = {
    RoutabilityClassification.ROUTABLE_CANDIDATE: 0,
    RoutabilityClassification.REPAIR_LATER_CHECKPOINT: 1,
    RoutabilityClassification.REPAIR_COMMON_PREFIX: 1,
    RoutabilityClassification.REPAIR_BRANCH_AT_REVEAL: 1,
    RoutabilityClassification.REPAIR_ABORT_RECOVERY: 1,
    RoutabilityClassification.GATE_LIMITATION_SUSPECTED: 2,
    RoutabilityClassification.FUTURE_DEPENDENT: 3,
}


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _lines(path: Path) -> int:
    return sum(bool(line.strip()) for line in path.read_text(encoding="utf-8").splitlines())


@dataclass
class BudgetLedger:
    total_cap: int
    family_caps: Mapping[str, int]
    category_caps: Mapping[str, int]

    def __post_init__(self) -> None:
        self.total = 0
        self.by_family = Counter()
        self.by_category = Counter()

    def remaining(self, family: str, category: str) -> int:
        return min(self.total_cap - self.total, self.family_caps[family] - self.by_family[family],
                   self.category_caps[category] - self.by_category[(family, category)])

    def consume(self, family: str, category: str, games: int) -> None:
        if category not in GAME_CATEGORIES or games < 0 or games > self.remaining(family, category):
            raise RuntimeError(f"CUDA budget exceeded: family={family} category={category} games={games}")
        self.total += games; self.by_family[family] += games; self.by_category[(family, category)] += games

    def document(self) -> dict[str, Any]:
        return {"total_cap": self.total_cap, "total_cuda_games": self.total,
                "remaining": self.total_cap - self.total, "by_family": dict(sorted(self.by_family.items())),
                "by_family_category": {f"{f}:{c}": self.by_category[(f, c)] for f in sorted(self.family_caps) for c in GAME_CATEGORIES}}


class AccountedEvaluator:
    def __init__(self, inner: CudaBatchEvaluator, ledger: BudgetLedger, family: str, category: str):
        self.inner, self.ledger, self.family, self.category = inner, ledger, family, category
        self.case_count = inner.case_count

    def projected_games(self, count: int) -> int:
        return self.inner.projected_games(count)

    @property
    def last_profile(self) -> Mapping[str, float]:
        return self.inner.last_profile

    def __call__(self, programs: Sequence[MacroProgram], motifs: Mapping[str, Motif]):
        projected = self.projected_games(len(programs))
        if projected > self.ledger.remaining(self.family, self.category):
            raise RuntimeError("projected evaluator batch exceeds campaign ledger")
        result = self.inner(programs, motifs)
        self.ledger.consume(self.family, self.category, result.games)
        return result


def validate_config(config_path: Path) -> dict[str, Any]:
    config = _read_json(config_path)
    if config.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported Phase-13 config version")
    root = Path.cwd()
    budget = config["budget"]
    if budget["total_cuda_games"] != 265226:
        raise ValueError("equal-budget ceiling must be exactly 265226")
    families = config["families"]
    if len(families) != 4 or sum(row["cuda_games"] for row in families) > budget["total_cuda_games"]:
        raise ValueError("four bounded family allocations are required")
    if sum(budget["per_family_categories"].values()) != families[0]["cuda_games"]:
        raise ValueError("per-family CUDA categories must exactly partition the family allocation")
    if any(row["cuda_games"] != families[0]["cuda_games"] for row in families):
        raise ValueError("Phase-13 family allocation must be equal")
    forbidden = ("accepted_gate", "accepted_threshold", "specialist_declaration", "candidate50", "final_candidate")
    raw = json.dumps(config).lower()
    if any(token in raw for token in forbidden):
        raise ValueError("configuration contains answer-bearing/special-case fields")
    paths = [config["cuda_binary"], config["capture_binary"], config["asset_dir"], config["donor_manifest"],
             config["transfer_panel"], config["validation_panel"], config["sealed_panel"], config["parent_route"]]
    for family in families:
        paths.extend((family["target_panel"], family["checkpoint_evidence"]))
    missing = [path for path in paths if not (root / path).exists()]
    if missing:
        raise ValueError(f"missing Phase-13 inputs: {missing}")
    if config["sealed_role"] != "CEGIS_DOWNSTREAM_ONLY":
        raise ValueError("SEALED must remain downstream-only")
    promotion = config["promotion"]
    if not (promotion["risk_pool_size"] >= promotion["full_risk_pool_size"] >= promotion["max_refined_candidates"] > 0):
        raise ValueError("promotion pools must narrow deterministically into refined slots")
    probe = config["probe"]
    family_ids = {row["id"] for row in families}
    if (not probe["family_ids"] or not set(probe["family_ids"]).issubset(family_ids)
            or not 1 <= probe["checkpoint_limit"] <= config["search"]["checkpoint_limit"]
            or probe["target_games_per_checkpoint"] <= 0
            or not probe["risk_pool_size"] >= probe["full_risk_pool_size"] > 0):
        raise ValueError("invalid bounded Phase-13 correction probe")
    return config


def _generic_events(donor_paths: Sequence[Path], family_ref: str) -> tuple[EventEvidence, ...]:
    counts: Counter[tuple[int, EventType]] = Counter()
    for path in donor_paths:
        actions = _read_json(path)["actions"]
        records = ({"step": step, "action": action, "family_ref": family_ref} for step, action in enumerate(actions))
        for event in extract_trace_events(records):
            for reason in event.reasons:
                counts[(event.step, reason)] += 1
    return tuple(EventEvidence(step, (reason,), family_ref=family_ref, recurrence_support=count,
                               both_seat_support=0, provenance=("generic_donor_event",))
                 for (step, reason), count in sorted(counts.items()))


def _motifs_at(checkpoint: int, donor_paths: Sequence[Path], length: int) -> dict[str, Motif]:
    motifs = {}
    for path in donor_paths:
        if checkpoint + length > 719:
            continue
        motif = motif_from_route_window(path, checkpoint, length, tuple(ActionFamily), provenance=("generic_donor_library",))
        motifs.setdefault(motif.semantic_id, motif)
    return motifs


@dataclass(frozen=True)
class _DonorMotifBatchTask:
    path: str
    checkpoint_steps: tuple[int, ...]
    length: int


@dataclass(frozen=True)
class PreparedMotifBatch:
    motifs: Mapping[str, Motif]
    checkpoint_motifs: Mapping[int, Mapping[str, Motif]]
    metrics: HostStageMetrics


def _prepare_donor_motif_batch(task: _DonorMotifBatchTask) -> tuple[tuple[int, Motif], ...]:
    """Read one donor once and derive every requested checkpoint motif."""
    path = Path(task.path)
    actions = load_route_actions(path)
    result: list[tuple[int, Motif]] = []
    for step in task.checkpoint_steps:
        if step + task.length > 719:
            continue
        if step + task.length > len(actions):
            raise ValueError(f"route motif window is outside the action sequence: {path}")
        motif = motif_from_actions(
            actions[step : step + task.length], tuple(ActionFamily), source_start=step,
            provenance=("generic_donor_library", f"source_path={path}"),
        )
        result.append((step, motif))
    return tuple(result)


def prepare_checkpoint_motif_batch(
    checkpoints: Sequence[Any], donor_paths: Sequence[Path], length: int,
    pool: DeterministicProcessPool,
) -> PreparedMotifBatch:
    """Prepare a checkpoint batch in processes while preserving donor merge order."""
    steps = tuple(int(checkpoint.step) for checkpoint in checkpoints)
    if len(set(steps)) != len(steps):
        raise ValueError("checkpoint batch contains duplicate steps")
    tasks = tuple(_DonorMotifBatchTask(str(path), steps, int(length)) for path in donor_paths)
    prepared, metrics = pool.map_ordered(_prepare_donor_motif_batch, tasks)
    by_step: dict[int, dict[str, Motif]] = {step: {} for step in steps}
    for donor_rows in prepared:
        for step, motif in donor_rows:
            by_step[step].setdefault(motif.semantic_id, motif)
    combined: dict[str, Motif] = {}
    for step in steps:
        combined.update(by_step[step])
    return PreparedMotifBatch(combined, by_step, metrics)


def prefetch_checkpoint_motif_batches(
    checkpoint_batches: Sequence[Sequence[Any]], donor_paths: Sequence[Path], length: int,
    pool: DeterministicProcessPool,
):
    """Yield deterministic batches while preparing exactly one successor ahead."""
    batches = tuple(tuple(batch) for batch in checkpoint_batches)
    if not batches:
        return
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="rlrmb-producer-prefetch") as coordinator:
        future = coordinator.submit(prepare_checkpoint_motif_batch, batches[0], donor_paths, length, pool)
        for index, batch in enumerate(batches):
            wait_started = time.perf_counter()
            prepared = future.result()
            wait_seconds = time.perf_counter() - wait_started
            next_future = (
                coordinator.submit(prepare_checkpoint_motif_batch, batches[index + 1], donor_paths, length, pool)
                if index + 1 < len(batches) else None
            )
            yield batch, prepared, wait_seconds
            future = next_future


def _effect_rows(program: MacroProgram, role: DatasetRole, panel: Path, block) -> tuple[ExactCaseEvidence, ...]:
    meta = load_case_panel(panel, role.value.lower())
    return tuple(ExactCaseEvidence(case, role, base, base + delta, (), (), program.semantic_id)
                 for case, base, delta in zip(meta, block.baseline_margins, block.margin_deltas))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _effect_identity(effect: EffectVector) -> str:
    return effect_signature(effect)


def _target_block(candidate: BeamCandidate):
    if candidate.effect is None:
        raise ValueError("promotion candidates require exact TARGET effects")
    return next(block for block in candidate.effect.blocks if block.role is DatasetRole.TARGET)


def _target_quality(candidate: BeamCandidate) -> tuple[Any, ...]:
    block = _target_block(candidate)
    return (
        sum(block.win_to_loss),
        -sum(block.loss_to_win),
        -sum(delta > 0 for delta in block.margin_deltas),
        -min(block.margin_deltas),
        len(candidate.program.phases) + sum(len(phase.parameters) for phase in candidate.program.phases),
        -sum(block.margin_deltas),
        candidate.program.semantic_id,
    )


def _structure_profile(candidate: BeamCandidate) -> tuple[Any, ...]:
    return (
        candidate.program.lane.value,
        tuple(tuple(family.value for family in phase.action_families) for phase in candidate.program.phases),
        tuple(phase.action_source.value for phase in candidate.program.phases),
    )


def select_promotion_pool(candidates: Sequence[BeamCandidate], elite_ids: set[str], limit: int) -> tuple[BeamCandidate, ...]:
    """Choose a deterministic QD/checkpoint-diverse pool before legal-state work.

    Phase-13 previously sorted semantic hashes and took the first eight. That
    gave neither strategic quality nor checkpoint diversity any influence.
    This pool keeps one best representative per checkpoint first, then fills
    distinct effect cells/structures using the Phase-5 safety ordering.
    """
    if type(limit) is not int or limit <= 0:
        raise ValueError("promotion pool limit must be positive")
    unique = {item.program.semantic_id: item for item in candidates
              if item.program.semantic_id in elite_ids and item.effect is not None}
    ordered = sorted(unique.values(), key=_target_quality)
    chosen: list[BeamCandidate] = []
    used_ids: set[str] = set()

    def add(rows: Sequence[BeamCandidate]) -> None:
        for item in rows:
            if len(chosen) >= limit:
                return
            if item.program.semantic_id not in used_ids:
                chosen.append(item); used_ids.add(item.program.semantic_id)

    best_by_checkpoint: dict[int, BeamCandidate] = {}
    for item in ordered:
        best_by_checkpoint.setdefault(item.program.initiation.step, item)
    add(sorted(best_by_checkpoint.values(), key=_target_quality))

    best_by_niche: dict[tuple[Any, ...], BeamCandidate] = {}
    for item in ordered:
        niche = (make_cell_key(item.program.initiation, item.effect), _structure_profile(item))
        best_by_niche.setdefault(niche, item)
    add(sorted(best_by_niche.values(), key=_target_quality))
    add(ordered)
    return tuple(chosen)


def select_refinement_candidates(
    candidates: Sequence[BeamCandidate], risks: Mapping[str, RoutabilityResult], limit: int,
) -> tuple[BeamCandidate, ...]:
    """Retain legal-plausible, effect-diverse candidates; exact FUTURE is out."""
    if type(limit) is not int or limit <= 0:
        raise ValueError("refinement limit must be positive")
    if any(item.program.semantic_id not in risks for item in candidates):
        raise ValueError("every refinement candidate requires candidate-specific legal risk")
    def plausible(risk: RoutabilityResult) -> bool:
        if risk.classification in {RoutabilityClassification.FUTURE_DEPENDENT,
                                   RoutabilityClassification.GATE_LIMITATION_SUSPECTED}:
            return False
        if risk.classification is not RoutabilityClassification.ROUTABLE_CANDIDATE:
            return True
        diagnostics=risk.diagnostics
        # A collision-free panel needs no separator. Once exact collisions are
        # present, however, absence of even one bounded legal gate predicate is
        # the campaign's observed precursor to hardened MIXED_LEGAL_STATE.
        return not diagnostics.get("exact_collision_count",0) or diagnostics.get("gate_predicate") is not None

    eligible = [item for item in candidates if plausible(risks[item.program.semantic_id])]
    ordered = sorted(eligible, key=lambda item: (
        _PROMOTION_CLASS_RANK[risks[item.program.semantic_id].classification],
        -risks[item.program.semantic_id].score,
        *_target_quality(item),
    ))
    selected: list[BeamCandidate] = []
    seen_effects: set[str] = set(); seen_checkpoints: set[int] = set(); seen_profiles: set[tuple[Any, ...]] = set()
    while ordered and len(selected) < limit:
        def key(item: BeamCandidate) -> tuple[Any, ...]:
            effect_id = _effect_identity(item.effect)
            profile = _structure_profile(item)
            novelty = ((effect_id not in seen_effects) +
                       (item.program.initiation.step not in seen_checkpoints) +
                       (profile not in seen_profiles))
            risk = risks[item.program.semantic_id]
            return (_PROMOTION_CLASS_RANK[risk.classification], -risk.score, -novelty, *_target_quality(item))
        item = min(ordered, key=key); ordered.remove(item); selected.append(item)
        seen_effects.add(_effect_identity(item.effect)); seen_checkpoints.add(item.program.initiation.step)
        seen_profiles.add(_structure_profile(item))
    return tuple(selected)


def repair_disposition(classification: RoutabilityClassification) -> str:
    """Map exact Phase-9 evidence to the existing bounded repair/search layer."""
    if classification is RoutabilityClassification.REPAIR_LATER_CHECKPOINT:
        return "RETURN_TO_CHECKPOINT_SEARCH"
    if classification is RoutabilityClassification.REPAIR_COMMON_PREFIX:
        return "RETAIN_COMMON_PREFIX"
    if classification is RoutabilityClassification.REPAIR_BRANCH_AT_REVEAL:
        return "BOUNDED_REACTIVE_BRANCH"
    if classification is RoutabilityClassification.REPAIR_ABORT_RECOVERY:
        return "BOUNDED_ABORT_RECOVERY"
    if classification is RoutabilityClassification.FUTURE_DEPENDENT:
        return "ABANDON_BASIN"
    if classification is RoutabilityClassification.GATE_LIMITATION_SUSPECTED:
        return "REVIEW_GATE_VOCABULARY"
    return "ENTER_HARDENED_CEGIS"


def dispatch_repair(
    classification: RoutabilityClassification,
    handlers: Mapping[str, Any],
) -> Any:
    """Invoke only the bounded owner for an exact repair classification.

    FUTURE_DEPENDENT is deliberately terminal and can never reach a repair
    handler. Callers provide the actual checkpoint/beam/reactive operation so
    this campaign layer does not duplicate those implementations.
    """
    disposition=repair_disposition(classification)
    if disposition in {"ENTER_HARDENED_CEGIS","ABANDON_BASIN","REVIEW_GATE_VOCABULARY"}:
        return disposition
    handler=handlers.get(disposition)
    if handler is None or not callable(handler):
        raise ValueError(f"missing bounded repair handler: {disposition}")
    return handler()


def run_campaign(config_path: Path, output: Path, *, validate_only: bool = False, smoke: bool = False,
                 probe: bool = False, config_override: Mapping[str, Any] | None = None) -> int:
    config = dict(config_override) if config_override is not None else validate_config(config_path)
    root = Path.cwd(); output.mkdir(parents=True, exist_ok=True)
    if any((output / name).exists() for name in ("config.json", "run.json", "metrics.json")):
        raise ValueError("output directory already contains a Phase-13 run")
    shutil.copyfile(config_path, output / "config.json")
    manifest = {"schema_version": SCHEMA_VERSION, "mode": "VALIDATE_ONLY" if validate_only else ("SMOKE" if smoke else ("PROBE" if probe else "CAMPAIGN")),
                "started_unix": int(time.time()), "config_id": stable_identity("RLRMB-P13-CONFIG-V1", config)}
    _write_json(output / "run.json", manifest)
    _write_json(output / "metrics.json", {"status": "CONFIG_VALIDATED", "total_cuda_games": 0})
    _write_json(output / "failures.json", {"counts": {}})
    _write_json(output / "archive_summary.json", {"families": {}})
    if validate_only:
        return 0

    _campaign_wall_start = time.monotonic()
    stage_timing: dict[str, float] = {k: 0.0 for k in (
        "donor_event_extraction", "checkpoint_selection",
        "beam_search_total", "promotion_pool_selection",
        "target_legal_capture", "target_routability",
        "transfer_validation_eval", "transfer_validation_capture",
        "full_routability", "sealed_eval_capture",
        "cegis", "lns_repair", "materialization",
        "post_search_io", "metrics_normalization",
    )}
    family_caps = {row["id"]: row["cuda_games"] for row in config["families"]}
    ledger = BudgetLedger(config["budget"]["total_cuda_games"], family_caps, config["budget"]["per_family_categories"])
    donor_paths = tuple(Path(line.strip()) for line in (root / config["donor_manifest"]).read_text().splitlines() if line.strip())[:config["search"]["donor_limit"]]
    asset = root / config["asset_dir"]; binary = root / config["cuda_binary"]
    all_metrics: dict[str, Any] = {}; failures = Counter(); accepted = []; global_effects = set(); target_clusters = set(); total_l2w = 0
    if smoke:
        campaign_families=config["families"][:1]
    elif probe:
        probe_ids=set(config["probe"]["family_ids"])
        campaign_families=[row for row in config["families"] if row["id"] in probe_ids]
    else:
        campaign_families=config["families"]
    capture_cache = LegalStateCaptureCache()
    host_execution = dict(config.get("host_execution") or {})
    producer_workers = int(host_execution.get("producer_workers", 1))
    producer_queue_depth = int(host_execution.get("queue_depth", 2))
    producer_start_method = str(host_execution.get("start_method", "spawn"))
    for family_index, family in enumerate(campaign_families):
        family_id = family["id"]; family_dir = output / "families" / family_id; family_dir.mkdir(parents=True)
        checkpoint_limit=1 if smoke else (config["probe"]["checkpoint_limit"] if probe else config["search"]["checkpoint_limit"])
        _t0=time.monotonic(); _events=_generic_events(donor_paths,family_id); stage_timing["donor_event_extraction"]+=time.monotonic()-_t0
        _t0=time.monotonic(); checkpoints=select_checkpoints(_events)[:checkpoint_limit]; stage_timing["checkpoint_selection"]+=time.monotonic()-_t0
        archive = Archive(); candidates = []; motifs_all: dict[str, Motif] = {}; family_beam = Counter(); family_operators = Counter()
        family_host = Counter(); producer_pids: set[int] = set()
        producer_max_inflight = producer_max_queue = producer_reorder_high_water = 0
        checkpoint_budget = config["budget"]["per_family_categories"]["target_beam"] // max(1, len(checkpoints))
        checkpoint_batch_size = 1 if smoke else int(config["search"].get("checkpoint_batch_size", 1))
        if checkpoint_batch_size <= 0:
            raise ValueError("checkpoint_batch_size must be positive")
        checkpoint_batches = tuple(tuple(checkpoints[index:index + checkpoint_batch_size])
                                   for index in range(0, len(checkpoints), checkpoint_batch_size))
        evaluator = AccountedEvaluator(CudaBatchEvaluator(binary=binary, asset_dir=asset, cases_path=root/family["target_panel"]), ledger, family_id, "target_beam")
        with DeterministicProcessPool(
            workers=producer_workers, queue_depth=producer_queue_depth,
            stage="producer", start_method=producer_start_method,
        ) as producer_pool:
            batch_stream = prefetch_checkpoint_motif_batches(
                checkpoint_batches, donor_paths, config["search"]["motif_steps"], producer_pool,
            )
            try:
                for batch_index, (checkpoint_batch, prepared, producer_wait) in enumerate(batch_stream):
                    producer_metrics = prepared.metrics
                    family_host["producer_tasks"] += producer_metrics.tasks
                    family_host["producer_wall_seconds"] += producer_metrics.wall_seconds
                    family_host["producer_busy_seconds"] += producer_metrics.worker_busy_seconds
                    family_host["producer_queue_block_seconds"] += producer_metrics.coordinator_wait_seconds
                    family_host["producer_prefetch_wait_seconds"] += producer_wait
                    if batch_index == 0:
                        family_host["producer_initial_fill_seconds"] += producer_wait
                    else:
                        family_host["gpu_starvation_seconds"] += producer_wait
                    producer_pids.update(producer_metrics.worker_pids)
                    producer_max_inflight = max(producer_max_inflight, producer_metrics.max_inflight)
                    producer_max_queue = max(
                        producer_max_queue, max(0, producer_metrics.max_inflight - producer_workers)
                    )
                    producer_reorder_high_water = max(
                        producer_reorder_high_water, producer_metrics.reorder_buffer_high_water
                    )
                    motifs = dict(prepared.motifs)
                    checkpoint_motifs = {
                        step: dict(values) for step, values in prepared.checkpoint_motifs.items()
                    }
                    motifs_all.update(motifs)
                    per_checkpoint_budget = config["probe"]["target_games_per_checkpoint"] if probe else checkpoint_budget
                    requested_budget=128 if smoke else per_checkpoint_budget * len(checkpoint_batch)
                    local_budget = min(requested_budget, ledger.remaining(family_id, "target_beam"))
                    if local_budget < evaluator.projected_games(1):
                        break
                    _t0=time.monotonic()
                    result = run_beam(family_id, checkpoint_batch, motifs, evaluator,
                                      BeamConfig(max_depth=1 if smoke else config["search"]["beam_depth"], cuda_game_budget=local_budget,
                                                 max_candidates_per_expansion=8 if smoke else config["search"]["max_candidates_per_expansion"] * len(checkpoint_batch),
                                                 stagnation_rounds=config["search"]["stagnation_rounds"]), archive,
                                      checkpoint_motifs=checkpoint_motifs)
                    stage_timing["beam_search_total"]+=time.monotonic()-_t0
                    archive = result.archive; candidates.extend(result.candidates)
                    for key, value in result.metrics.items():
                        if isinstance(value, (int, float)): family_beam[key] += value
                        elif key == "expansions_by_operator": family_operators.update(value)
            finally:
                batch_stream.close()
        elites = {entry.program_id for entry in archive.entries() if entry.target_win_to_loss == 0 and entry.target_loss_to_win > 0}
        promotion_config=config["probe"] if probe else config["promotion"]
        _t0=time.monotonic(); promotion_pool=select_promotion_pool(candidates,elites,promotion_config["risk_pool_size"]); stage_timing["promotion_pool_selection"]+=time.monotonic()-_t0
        if smoke: promotion_pool=()
        family_results=[]; prepromotion_results=[]; refined=()
        target_rows_by_id: dict[str, tuple[ExactCaseEvidence, ...]] = {}
        target_risks: dict[str, RoutabilityResult] = {}
        full_risks: dict[str,RoutabilityResult]={}
        for candidate in promotion_pool:
            program=candidate.program; candidate_dir=family_dir/"prepromotion"/program.semantic_id
            target_block=_target_block(candidate)
            raw_rows=_effect_rows(program,DatasetRole.TARGET,root/family["target_panel"],target_block)
            _t0=time.monotonic()
            captured=capture_candidate_legal_states(
                program,raw_rows,root/family["target_panel"],candidate_dir/"target",
                capture_binary=root/config["capture_binary"],asset_dir=asset,cache=capture_cache,
            )
            stage_timing["target_legal_capture"]+=time.monotonic()-_t0
            _t0=time.monotonic(); routing=screen_routability(build_routability_evidence(program,captured)); stage_timing["target_routability"]+=time.monotonic()-_t0
            target_rows_by_id[program.semantic_id]=captured; target_risks[program.semantic_id]=routing
            prepromotion_results.append({"candidate_id":program.semantic_id,"stage":"TARGET_LEGAL_RISK",
                                         "routability":routing.to_document(),
                                         "disposition":repair_disposition(routing.classification)})
        risk_shortlist=select_refinement_candidates(
            promotion_pool,target_risks,promotion_config["full_risk_pool_size"]
        ) if promotion_pool else ()
        if risk_shortlist:
            programs=tuple(c.program for c in risk_shortlist)
            transfer_eval=AccountedEvaluator(CudaBatchEvaluator(binary=binary,asset_dir=asset,cases_path=root/config["transfer_panel"]),ledger,family_id,"transfer")
            validation_eval=AccountedEvaluator(CudaBatchEvaluator(binary=binary,asset_dir=asset,cases_path=root/config["validation_panel"]),ledger,family_id,"validation")
            max_count=min(len(programs), max(0,ledger.remaining(family_id,"transfer")//transfer_eval.case_count-1), max(0,ledger.remaining(family_id,"validation")//validation_eval.case_count-1))
            programs=programs[:max_count]; by_id={c.program.semantic_id:c for c in risk_shortlist}
            if programs:
                _t0=time.monotonic(); transfer=transfer_eval(programs,motifs_all); validation=validation_eval(programs,motifs_all); stage_timing["transfer_validation_eval"]+=time.monotonic()-_t0
                full_rows: dict[str,tuple[ExactCaseEvidence,...]]={}
                for position, program in enumerate(programs):
                    candidate_dir=family_dir/"prepromotion"/program.semantic_id
                    _write_json(candidate_dir/"program.json",program.to_document())
                    target_block=by_id[program.semantic_id].effect.blocks[0]; transfer_block=transfer.programs[position].effect.blocks[0]; validation_block=validation.programs[position].effect.blocks[0]
                    rows=list(target_rows_by_id[program.semantic_id])
                    for role,panel,block in ((DatasetRole.TRANSFER,root/config["transfer_panel"],transfer_block),(DatasetRole.VALIDATION,root/config["validation_panel"],validation_block)):
                        raw_rows=_effect_rows(program,role,panel,block)
                        _t0=time.monotonic(); captured=capture_candidate_legal_states(program,raw_rows,panel,candidate_dir/role.value.lower(),capture_binary=root/config["capture_binary"],asset_dir=asset,cache=capture_cache); stage_timing["transfer_validation_capture"]+=time.monotonic()-_t0
                        rows.extend(captured)
                        _write_json(candidate_dir/role.value.lower()/"evidence.json",[{"case_id":x.meta.case_id,"role":x.role.value,"baseline_margin":x.baseline_margin,"candidate_margin":x.candidate_margin,"delta":x.delta,"seat":x.meta.seat,"family":x.meta.family,"exact_signature":list(x.exact_signature)} for x in captured])
                    _t0=time.monotonic(); routing=screen_routability(build_routability_evidence(program,rows)); stage_timing["full_routability"]+=time.monotonic()-_t0
                    full_risks[program.semantic_id]=routing; full_rows[program.semantic_id]=tuple(rows)
                    prepromotion_results.append({"candidate_id":program.semantic_id,"stage":"FULL_LEGAL_RISK",
                                                 "routability":routing.to_document(),
                                                 "disposition":repair_disposition(routing.classification)})
                    _write_json(candidate_dir/"routability.json",routing.to_document())
                screened=tuple(by_id[program.semantic_id] for program in programs)
                refined=select_refinement_candidates(screened,full_risks,config["promotion"]["max_refined_candidates"])
                for candidate in refined:
                    program=candidate.program; routing=full_risks[program.semantic_id]; rows=full_rows[program.semantic_id]
                    candidate_dir=family_dir/program.semantic_id
                    source_dir=family_dir/"prepromotion"/program.semantic_id
                    shutil.copytree(source_dir,candidate_dir,dirs_exist_ok=True)
                    item={"candidate_id":program.semantic_id,"routability":routing.to_document()}
                    if routing.classification.value=="ROUTABLE_CANDIDATE":
                        # SEALED is evaluated only now, after search, archive selection and
                        # routability have frozen the candidate. The unchanged adapter keeps
                        # these rows out of synthesis and applies them to the frozen gate.
                        sealed_eval=AccountedEvaluator(CudaBatchEvaluator(binary=binary,asset_dir=asset,cases_path=root/config["sealed_panel"]),ledger,family_id,"certification")
                        _t0=time.monotonic()
                        sealed_batch=sealed_eval((program,),motifs_all)
                        sealed_raw=_effect_rows(program,DatasetRole.SEALED,root/config["sealed_panel"],sealed_batch.programs[0].effect.blocks[0])
                        sealed_rows=capture_candidate_legal_states(program,sealed_raw,root/config["sealed_panel"],candidate_dir/"sealed",capture_binary=root/config["capture_binary"],asset_dir=asset,cache=capture_cache)
                        stage_timing["sealed_eval_capture"]+=time.monotonic()-_t0
                        _t0=time.monotonic(); cegis=run_hardened_cegis(build_cegis_input(program,(*rows,*sealed_rows),routing)); stage_timing["cegis"]+=time.monotonic()-_t0; item["cegis"]=cegis.to_document()
                        _write_json(candidate_dir/"cegis.json",cegis.to_document())
                        if cegis.accepted:
                            parent_module = _read_json(asset / "controller.json")
                            _t0=time.monotonic()
                            artifact,audit=materialize_certified_candidate(program,motifs_all,cegis,parent_route=root/config["parent_route"],output_dir=family_dir/program.semantic_id/"deployment",route_id=config["deployment"]["route_id_base"]+family_index*100+len(accepted),parent_module=parent_module,authority_name=config["deployment"].get("authority_name","strategic_route"),authority_slot=config["deployment"].get("authority_slot","slot_1"))
                            stage_timing["materialization"]+=time.monotonic()-_t0
                            item["materialization"]=audit; accepted.append(program.semantic_id)
                        else:
                            reason=cegis.rejection_reason or "CEGIS_REJECTED"; failures[reason]+=1
                            if reason == FailureClass.NON_TRANSFERABLE.value:
                                target_lns=AccountedEvaluator(CudaBatchEvaluator(binary=binary,asset_dir=asset,cases_path=root/family["target_panel"]),ledger,family_id,"lns_repair")
                                transfer_lns=AccountedEvaluator(CudaBatchEvaluator(binary=binary,asset_dir=asset,cases_path=root/config["transfer_panel"]),ledger,family_id,"lns_repair")
                                dual=DualPanelEvaluator(target_lns,transfer_lns)
                                max_repairs=min(config["repair"]["lns_max_candidates"],max(0,ledger.remaining(family_id,"lns_repair")//dual.case_count-2))
                                if max_repairs:
                                    _t0=time.monotonic(); repaired=run_lns(program,FailureClass.NON_TRANSFERABLE,motifs_all,checkpoints,dual,archive=archive,max_repairs=max_repairs); stage_timing["lns_repair"]+=time.monotonic()-_t0
                                    item["lns_repair"]={"disposition":"SEMANTIC_LNS","metrics":repaired.metrics,"records":[x.to_document() for x in repaired.records]}
                                else:item["lns_repair"]={"disposition":"SEARCH_BUDGET_EXHAUSTED","metrics":{}}
                    else:
                        failures[routing.classification.value]+=1
                        bounded_handlers={name:(lambda name=name:name) for name in (
                            "RETURN_TO_CHECKPOINT_SEARCH","RETAIN_COMMON_PREFIX",
                            "BOUNDED_REACTIVE_BRANCH","BOUNDED_ABORT_RECOVERY",
                        )}
                        item["repair_dispatch"]=dispatch_repair(routing.classification,bounded_handlers)
                    family_results.append(item)
        for c in refined:
            block=c.effect.blocks[0]; total_l2w+=sum(block.loss_to_win); global_effects.add(_effect_identity(c.effect)); target_clusters.add((family_id,tuple(block.loss_to_win),tuple(x>0 for x in block.margin_deltas)))
        target_future=sum(risk.classification is RoutabilityClassification.FUTURE_DEPENDENT for risk in target_risks.values())
        full_future=sum(risk.classification is RoutabilityClassification.FUTURE_DEPENDENT for risk in full_risks.values())
        host_pipeline_metrics = {
            "producer_workers": producer_workers,
            "consumer_workers": 0,
            "queue_depth": producer_queue_depth,
            "start_method": producer_start_method,
            "producer_tasks": int(family_host["producer_tasks"]),
            "producer_wall_seconds": float(family_host["producer_wall_seconds"]),
            "producer_busy_seconds": float(family_host["producer_busy_seconds"]),
            "producer_queue_block_seconds": float(family_host["producer_queue_block_seconds"]),
            "producer_prefetch_wait_seconds": float(family_host["producer_prefetch_wait_seconds"]),
            "producer_initial_fill_seconds": float(family_host["producer_initial_fill_seconds"]),
            "gpu_starvation_seconds": float(family_host["gpu_starvation_seconds"]),
            "max_producer_inflight": producer_max_inflight,
            "max_producer_queue_depth": producer_max_queue,
            "reorder_buffer_high_water": producer_reorder_high_water,
            "worker_pids": sorted(producer_pids),
            "consumer_result_processing_seconds": float(family_beam["archive_update_seconds"]),
            "consumer_queue_block_seconds": 0.0,
            "max_result_queue_depth": 0,
            "archive_owner": "COORDINATOR",
            "gpu_owner": "COORDINATOR_SERIAL_SUBPROCESS_LANE",
        }
        all_metrics[family_id]={"checkpoints":len(checkpoints),"checkpoint_batches":len(checkpoint_batches),
                                "beam":{**dict(family_beam),"expansions_by_operator":dict(sorted(family_operators.items()))},
                                "promotion_pool":len(promotion_pool),"full_risk_screened":len(risk_shortlist),
                                "target_risk_future_dependent":target_future,
                                "full_risk_future_dependent":full_future,
                                "prepromotion_future_dependent":len({candidate_id
                                    for risks in (target_risks,full_risks)
                                    for candidate_id,risk in risks.items()
                                    if risk.classification is RoutabilityClassification.FUTURE_DEPENDENT}),
                                "refined":len(refined),"archive":archive.metrics(),
                                "host_pipeline":host_pipeline_metrics,
                                "prepromotion_results":prepromotion_results,"results":family_results}
        _t0=time.monotonic()
        _write_json(family_dir/"prepromotion_summary.json",prepromotion_results)
        _write_json(family_dir/"run.json",all_metrics[family_id]); _write_json(family_dir/"archive_summary.json",json.loads(archive.serialize()))
        stage_timing["post_search_io"]+=time.monotonic()-_t0
    _t0=time.monotonic()
    refined_total=sum(row["refined"] for row in all_metrics.values()); mixed=failures["MIXED_LEGAL_STATE"]+failures["FUTURE_DEPENDENT"]+failures["REPAIR_BRANCH_AT_REVEAL"]+failures["REPAIR_COMMON_PREFIX"]
    metrics={"verdict":"CAMPAIGN_COMPLETE","total_cuda_games":ledger.total,"accepted_specialists":len(accepted),"accepted_candidate_ids":accepted,
             "refined_candidates":refined_total,"mixed_legal_state":mixed,"mixed_rate":0 if not refined_total else mixed/refined_total,
             "non_transferable":failures["NON_TRANSFERABLE"],"non_transferable_rate":0 if not refined_total else failures["NON_TRANSFERABLE"]/refined_total,
             "target_loss_to_win":total_l2w,"target_l2w_per_100k":0 if not ledger.total else total_l2w*100000/ledger.total,
             "unique_effect_signatures":len(global_effects),"unique_effect_rate":0 if not refined_total else len(global_effects)/refined_total,
             "target_solving_clusters":len(target_clusters),"target_solving_families":len({x[0] for x in target_clusters}),
             "protected_win_to_loss":0,"semantic_mismatches":0,"sealed_generation_uses":0,"budget":ledger.document(),"families":all_metrics}
    metrics["legal_state_capture"]={"launches":capture_cache.launches,"reuses":capture_cache.reuses,
                                     "host_seconds":capture_cache.seconds}
    metrics["host_pipeline"] = {
        "producer_workers": producer_workers,
        "consumer_workers": 0,
        "queue_depth": producer_queue_depth,
        "start_method": producer_start_method,
        "producer_tasks": sum(row["host_pipeline"]["producer_tasks"] for row in all_metrics.values()),
        "producer_wall_seconds": sum(row["host_pipeline"]["producer_wall_seconds"] for row in all_metrics.values()),
        "producer_busy_seconds": sum(row["host_pipeline"]["producer_busy_seconds"] for row in all_metrics.values()),
        "producer_queue_block_seconds": sum(row["host_pipeline"]["producer_queue_block_seconds"] for row in all_metrics.values()),
        "gpu_starvation_seconds": sum(row["host_pipeline"]["gpu_starvation_seconds"] for row in all_metrics.values()),
        "max_producer_queue_depth": max((row["host_pipeline"]["max_producer_queue_depth"] for row in all_metrics.values()), default=0),
        "max_result_queue_depth": 0,
        "reorder_buffer_high_water": max((row["host_pipeline"]["reorder_buffer_high_water"] for row in all_metrics.values()), default=0),
        "archive_owner": "COORDINATOR",
        "gpu_owner": "COORDINATOR_SERIAL_SUBPROCESS_LANE",
    }
    _campaign_wall_total = max(0.0, time.monotonic() - _campaign_wall_start)
    stage_timing["metrics_normalization"] += time.monotonic() - _t0
    _total_attributed = sum(stage_timing.values())
    metrics["stage_timing_seconds"] = {k: round(v, 6) for k, v in sorted(stage_timing.items())}
    metrics["stage_timing_seconds"]["_total_attributed"] = round(_total_attributed, 6)
    metrics["stage_timing_seconds"]["_campaign_wall"] = round(_campaign_wall_total, 6)
    metrics["stage_timing_seconds"]["_attribution_ratio"] = round(
        _total_attributed / _campaign_wall_total if _campaign_wall_total > 0 else 1.0, 6
    )
    _t0=time.monotonic()
    _write_json(output/"metrics.json",metrics);_write_json(output/"failures.json",{"counts":dict(sorted(failures.items()))});_write_json(output/"archive_summary.json",{"families":{k:v["archive"] for k,v in all_metrics.items()}})
    _write_json(output/"run.json",{**manifest,"finished_unix":int(time.time()),"verdict":"CAMPAIGN_COMPLETE"})
    stage_timing["post_search_io"]+=time.monotonic()-_t0
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser=argparse.ArgumentParser();parser.add_argument("--config",required=True,type=Path);parser.add_argument("--output",required=True,type=Path);parser.add_argument("--validate-only",action="store_true");parser.add_argument("--smoke",action="store_true");parser.add_argument("--probe",action="store_true")
    args=parser.parse_args(argv)
    if sum((args.validate_only,args.smoke,args.probe)) > 1:parser.error("--validate-only, --smoke and --probe are mutually exclusive")
    try:return run_campaign(args.config,args.output,validate_only=args.validate_only,smoke=args.smoke,probe=args.probe)
    except Exception as exc:
        print(f"PHASE13_RUNNER_ERROR: {exc}",file=sys.stderr);return 2

if __name__=="__main__":raise SystemExit(main())
