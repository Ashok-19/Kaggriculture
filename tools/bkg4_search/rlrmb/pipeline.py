"""Generic Phase-12 candidate certification orchestration.

This module binds exact CUDA effects and exact B4IS legal-state captures to the
already validated routability, CEGIS and materialization contracts.  It contains
no historical route, gate or threshold policy.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
import traceback
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from tools.bkg4_search.edit_search.goals import CONTRASTIVE_FEATURES
from tools.bkg4_search.information_state_io import load_r0, registry_manifest
from tools.bkg4_search.bonsai_composer import StrategicAuthority, compose_specialists
from tools.bkg4_search.module_codec import ConditionGene, FEATURES as _GATE_GRAMMAR_FEATURES

from .beam import CudaBatchEvaluator
from .cegis_adapter import AdapterCase, CegisAdapterInput, CegisAdapterResult, run_hardened_cegis
from .ir import MacroProgram
from .motifs import Motif
from .canonical import canonical_action, load_route_actions
from .materialize import MaterializationArtifact, MaterializationInput, materialize_program
from .routability import (
    BoundaryState, RequiredContinuation, RoutabilityEvidence, RoutabilityResult,
    TrajectoryEvidence, screen_routability,
)
from .schema import DatasetRole, stable_identity

PIPELINE_SCHEMA_VERSION = 1


class LegalStateCaptureCache:
    """Reuse of candidate-independent outer-initiation captures.

    When ``stable_dir`` is provided the cache copies capture files there so
    entries remain valid across the attempt-directory renames that
    ``cloud_runner`` performs.  Without a ``stable_dir`` the cache behaves
    exactly as before (attempt-local reuse only).
    """

    def __init__(self, stable_dir: "Path | None" = None) -> None:
        self.paths: dict[tuple[Any, ...], tuple[Path, Path]] = {}
        self.stable_dir = stable_dir
        self._entry_count = 0
        self.launches = 0
        self.reuses = 0
        self.seconds = 0.0


class TerminalVerdict(str, Enum):
    ACCEPTED_SPECIALIST = "ACCEPTED_SPECIALIST"
    NO_SAFE_SPECIALIST_FOUND = "NO_SAFE_SPECIALIST_FOUND"
    SEARCH_BUDGET_EXHAUSTED = "SEARCH_BUDGET_EXHAUSTED"
    STRATEGIC_REJECTION = "STRATEGIC_REJECTION"
    INFRASTRUCTURE_FAILURE = "INFRASTRUCTURE_FAILURE"


EXIT_ACCEPTED = 0
EXIT_STRATEGIC = 1
EXIT_INFRASTRUCTURE = 2


def _write_document(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def validate_production_config(config_path: str | Path, output: str | Path) -> dict[str, Any]:
    """Validate the bounded production contract without spending CUDA games.

    The Phase-13 equal-budget schema is the first production schema. Reusing it
    keeps one validated search policy and, importantly, preserves its central
    game ledger and routability-aware refinement order.
    """
    config_path = Path(config_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    allowed = {"schema_version", "cuda_binary", "capture_binary", "asset_dir", "donor_manifest",
               "parent_route", "transfer_panel", "validation_panel", "sealed_panel", "sealed_role",
               "budget", "search", "promotion", "probe", "repair", "deployment", "output", "families", "cloud",
               "host_execution", "source_provenance", "diagnostics", "internal_dev",
               "build_objective"}
    unknown = set(config) - allowed
    if config.get("schema_version") != PIPELINE_SCHEMA_VERSION or unknown:
        raise ValueError(f"unsupported production config schema/fields: {sorted(unknown)}")
    host = config.get("host_execution", {})
    if not isinstance(host, Mapping):
        raise ValueError("host_execution must be a mapping")
    unknown_host = set(host) - {"producer_workers", "queue_depth", "start_method", "candidate_workers", "candidate_queue_depth"}
    if unknown_host:
        raise ValueError(f"unsupported host_execution fields: {sorted(unknown_host)}")
    producer_workers = host.get("producer_workers", 1)
    queue_depth = host.get("queue_depth", 2)
    start_method = host.get("start_method", "spawn")
    if type(producer_workers) is not int or producer_workers <= 0 or producer_workers > 64:
        raise ValueError("host_execution.producer_workers must be an integer in 1..64")
    if type(queue_depth) is not int or queue_depth <= 0 or queue_depth > 64:
        raise ValueError("host_execution.queue_depth must be an integer in 1..64")
    if start_method not in {"spawn", "forkserver", "fork"}:
        raise ValueError("host_execution.start_method must be spawn, forkserver, or fork")
    provenance = config.get("source_provenance")
    if provenance is not None:
        if not isinstance(provenance, Mapping):
            raise ValueError("source_provenance must be a mapping")
        if provenance.get("kind") not in {"git", "bundle_manifest"}:
            raise ValueError("source_provenance.kind must be git or bundle_manifest")
        source_head = provenance.get("source_head")
        if (not isinstance(source_head, str) or len(source_head) != 40
                or any(c not in "0123456789abcdef" for c in source_head)):
            raise ValueError("source_provenance.source_head must be a lowercase 40-hex commit id")
    families = config.get("families", ())
    budget = config.get("budget", {})
    total = budget.get("total_cuda_games")
    categories = budget.get("per_family_categories", {})
    if type(total) is not int or total <= 0 or not families:
        raise ValueError("production config requires a positive bounded budget and frontier families")
    if any(type(row.get("cuda_games")) is not int or row["cuda_games"] <= 0 for row in families):
        raise ValueError("each frontier family requires a positive CUDA allocation")
    if sum(row["cuda_games"] for row in families) > total:
        raise ValueError("frontier allocations exceed the authoritative CUDA budget")
    required_categories = {"target_beam", "transfer", "validation", "lns_repair", "reactive_repair", "certification"}
    allowed_categories = required_categories | {"diagnostic_protected", "internal_dev"}
    if (not required_categories <= set(categories) or not set(categories) <= allowed_categories
            or any(type(value) is not int or value < 0 for value in categories.values())):
        raise ValueError("all CUDA game categories require non-negative integer caps")
    if any(sum(categories.values()) > row["cuda_games"] for row in families):
        raise ValueError("per-family category caps exceed a family allocation")
    build_objective = config.get("build_objective")
    if build_objective is not None:
        if not isinstance(build_objective, Mapping):
            raise ValueError("build_objective must be a mapping")
        unknown_build = set(build_objective) - {
            "require_cross_regime_support", "max_win_to_loss"
        }
        if unknown_build:
            raise ValueError(f"unsupported build_objective fields: {sorted(unknown_build)}")
        if type(build_objective.get("require_cross_regime_support")) is not bool:
            raise ValueError("build_objective.require_cross_regime_support must be bool")
        max_w2l = build_objective.get("max_win_to_loss")
        if type(max_w2l) is not int or max_w2l < 0:
            raise ValueError("build_objective.max_win_to_loss must be non-negative")
    root = Path.cwd()
    paths = [config.get(name) for name in ("cuda_binary", "capture_binary", "asset_dir", "donor_manifest",
                                           "parent_route", "transfer_panel", "validation_panel", "sealed_panel")]
    paths.extend(value for family in families for value in (family.get("target_panel"), family.get("checkpoint_evidence")))
    missing = [path for path in paths if not path or not (root / path).exists()]
    if missing:
        raise ValueError(f"missing production inputs: {missing}")
    if build_objective and build_objective["require_cross_regime_support"]:
        from .build_regimes import load_build_regimes
        for family in families:
            regimes = load_build_regimes(root / family["target_panel"])
            if (len({row.seed for row in regimes}) < 2
                    or {row.seat for row in regimes} != {0, 1}
                    or len({row.member for row in regimes}) < 2):
                raise ValueError(
                    "cross-regime BUILD requires multiple seeds, both seats and multiple members"
                )
    output_path = Path(output)
    if output_path.exists() and any(output_path.iterdir()):
        raise ValueError("production output directory must be new or empty")
    if config.get("sealed_role") != "CEGIS_DOWNSTREAM_ONLY":
        raise ValueError("SEALED evidence is downstream-only")
    dev = config.get("internal_dev")
    if dev is not None:
        # INTERNAL_DEV is a bounded pre-downstream screen. It must declare a
        # real rotation of independently generated panels, and it must never be
        # pointed at holdout evidence: that would convert the safety firewall
        # into a search target.
        if not isinstance(dev, Mapping):
            raise ValueError("internal_dev must be a mapping")
        unknown_dev = set(dev) - {"panels", "max_win_to_loss", "max_win_to_loss_rate",
                                  "max_screened", "max_downstream", "allow_fallback"}
        if unknown_dev:
            raise ValueError(f"unsupported internal_dev fields: {sorted(unknown_dev)}")
        panels = dev.get("panels", ())
        if (not isinstance(panels, Sequence) or isinstance(panels, str) or not panels
                or any(type(row) is not str or not row.strip() for row in panels)):
            raise ValueError("internal_dev.panels must be a non-empty list of panel paths")
        holdout = {config.get("transfer_panel"), config.get("validation_panel"),
                   config.get("sealed_panel")}
        build = {family.get("target_panel") for family in families}
        if set(panels) & holdout:
            raise ValueError("internal_dev panels must not reuse TRANSFER/VALIDATION/SEALED evidence")
        if set(panels) & build:
            raise ValueError("internal_dev panels must be independent of BUILD target panels")
        missing_dev = [row for row in panels if not (root / row).exists()]
        if missing_dev:
            raise ValueError(f"missing internal_dev panels: {missing_dev}")
        if categories.get("internal_dev", 0) <= 0:
            raise ValueError("internal_dev requires a positive bounded CUDA category cap")
    raw = json.dumps(config, sort_keys=True).lower()
    forbidden = ("selected_route", "accepted_candidate", "final_gate", "gate_threshold",
                 "force_accept", "specialist_json", "candidate_id_to_promote")
    if any(field in raw for field in forbidden):
        raise ValueError("production config contains manual-answer injection")
    return config


def _normalise_failures(output: Path, metrics: Mapping[str, Any]) -> None:
    path = output / "failures.json"
    source = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"counts": {}}
    records = []
    for family, family_metrics in sorted(metrics.get("families", {}).items()):
        for row in family_metrics.get("prepromotion_results", ()): 
            classification = row.get("routability", {}).get("classification")
            if classification and classification != "ROUTABLE_CANDIDATE":
                records.append({
                    "candidate_semantic_id": row.get("candidate_id"),
                    "failure_class": classification,
                    "pipeline_stage": row.get("stage", "ROUTABILITY"),
                    "evidence_references": [f"families/{family}/prepromotion_summary.json"],
                    "recommended_next_action": row.get("disposition", "ABANDON_BASIN"),
                })
        for row in family_metrics.get("results", ()): 
            routing = row.get("routability", {})
            classification = routing.get("classification")
            if classification and classification != "ROUTABLE_CANDIDATE":
                records.append({
                    "candidate_semantic_id": row.get("candidate_id"),
                    "failure_class": classification,
                    "pipeline_stage": "REFINEMENT",
                    "evidence_references": [f"families/{family}/{row.get('candidate_id')}/routability.json"],
                    "recommended_next_action": row.get("repair_dispatch", "ABANDON_BASIN"),
                })
    _write_document(path, {"schema_version": PIPELINE_SCHEMA_VERSION,
                           "counts": source.get("counts", {}), "records": records})


def _normalise_metrics(output: Path, metrics: dict[str, Any]) -> dict[str, Any]:
    """Make every permanent production metric present, including zero values."""
    families = metrics.get("families", {})
    beam_rows = [row.get("beam", {}) for row in families.values()]
    archive_rows = [row.get("archive", {}) for row in families.values()]
    expansions: dict[str, int] = {}
    for row in beam_rows:
        for operator, count in row.get("expansions_by_operator", {}).items():
            expansions[operator] = expansions.get(operator, 0) + int(count)
    result_rows = [row for family in families.values() for row in family.get("results", ())]
    lns_rows = [row["lns_repair"] for row in result_rows if "lns_repair" in row]
    classifications: dict[str, int] = {}
    for family in families.values():
        for row in (*family.get("prepromotion_results", ()), *family.get("results", ())):
            name = row.get("routability", {}).get("classification")
            if name and name != "ROUTABLE_CANDIDATE":
                classifications[name] = classifications.get(name, 0) + 1
    host_seconds = sum(float(row.get("host_seconds", 0.0)) for row in beam_rows)
    kernel_seconds = sum(float(row.get("cuda_ms", 0.0)) for row in beam_rows) / 1000.0
    defaults = {
        "frontier_families": len(families),
        "checkpoints": sum(row.get("checkpoints", 0) for row in families.values()),
        "beam_depth": max((row.get("depth_reached", 0) for row in beam_rows), default=0),
        "expansions_by_operator": dict(sorted(expansions.items())),
        "lns_attempts": len(lns_rows),
        "lns_successes": sum(bool(row.get("records")) for row in lns_rows),
        "reactive_attempts": 0, "reactive_successes": 0,
        "structural_generated": sum(row.get("raw_candidates", 0) for row in beam_rows),
        "structural_duplicates": sum(row.get("structural_duplicates_removed", 0) for row in beam_rows),
        "effect_signatures": sum(row.get("unique_effect_signatures", 0) for row in archive_rows),
        "archive_occupancy": sum(row.get("occupied_cells", 0) for row in archive_rows),
        "archive_replacements": sum(row.get("replacements", 0) for row in archive_rows),
        "target_solving_clusters": metrics.get("target_solving_clusters", 0),
        "target_positives": sum(row.get("target_positives", 0) for row in beam_rows),
        "target_loss_to_win": metrics.get("target_loss_to_win", 0),
        "target_win_to_loss": metrics.get("protected_win_to_loss", 0),
        "transfer_positives": 0, "transfer_harms": 0,
        "routable": sum(row.get("routability", {}).get("classification") == "ROUTABLE_CANDIDATE"
                        for family in families.values() for row in family.get("results", ())),
        "future_dependent": sum(row.get("routability", {}).get("classification") == "FUTURE_DEPENDENT"
                                for family in families.values() for row in family.get("prepromotion_results", ())),
        "repair_classifications": dict(sorted(classifications.items())),
        "cegis_entries": sum("cegis" in row for family in families.values() for row in family.get("results", ())),
        "cegis_convergence": metrics.get("accepted_specialists", 0),
        "wall_time_seconds": host_seconds, "cuda_kernel_time_seconds": kernel_seconds,
        "games_per_second": 0.0 if not host_seconds else metrics.get("total_cuda_games", 0) / host_seconds,
    }
    for key, value in defaults.items():
        metrics.setdefault(key, value)
    _write_document(output / "metrics.json", metrics)
    return metrics


def _terminal_verdict(metrics: Mapping[str, Any]) -> TerminalVerdict:
    if int(metrics.get("semantic_mismatches", 0)):
        return TerminalVerdict.INFRASTRUCTURE_FAILURE
    if int(metrics.get("accepted_specialists", 0)):
        return TerminalVerdict.ACCEPTED_SPECIALIST
    budget = metrics.get("budget", {})
    if budget and int(budget.get("remaining", 1)) == 0:
        return TerminalVerdict.SEARCH_BUDGET_EXHAUSTED
    if metrics.get("refined_candidates", 0):
        return TerminalVerdict.STRATEGIC_REJECTION
    return TerminalVerdict.NO_SAFE_SPECIALIST_FOUND


def _production_summary(config: Mapping[str, Any], output: Path, metrics: Mapping[str, Any],
                        verdict: TerminalVerdict, started: float, finished: float) -> dict[str, Any]:
    config_id = stable_identity("RLRMB-PRODUCTION-CONFIG-V1", config)
    repository_head = _repository_head(config)
    provenance = dict(config.get("source_provenance") or {
        "kind": "git" if repository_head != "UNKNOWN" else "unknown",
        "source_head": repository_head,
    })
    run_id = stable_identity("RLRMB-PRODUCTION-RUN-V1", config_id, repository_head)
    accepted = tuple(metrics.get("accepted_candidate_ids", ()))
    return {
        "schema_version": PIPELINE_SCHEMA_VERSION,
        "run_id": run_id,
        "config_identity": config_id,
        "repository_head": repository_head,
        "source_provenance": provenance,
        "current_bkg4_reference": config.get("parent_route"),
        "start_state": "FRONTIER_SELECTION",
        "end_state": verdict.value,
        "terminal_verdict": verdict.value,
        "total_cuda_games": int(metrics.get("total_cuda_games", 0)),
        "host_time_seconds": max(0.0, finished - started),
        "cuda_kernel_time_seconds": metrics.get("kernel_time_seconds", 0.0),
        "accepted_specialist_ids": accepted,
        "failure_counts": json.loads((output / "failures.json").read_text(encoding="utf-8")).get("counts", {}),
        "artifact_references": sorted(str(path.relative_to(output)) for path in output.rglob("*.json")
                                      if path.name != "run.json"),
    }


def _repository_head(config: Mapping[str, Any] | None = None) -> str:
    if config is not None:
        provenance = config.get("source_provenance")
        if isinstance(provenance, Mapping):
            source_head = provenance.get("source_head")
            if (isinstance(source_head, str) and len(source_head) == 40
                    and all(c in "0123456789abcdef" for c in source_head)):
                return source_head
    result = subprocess.run(("git", "rev-parse", "HEAD"), check=False, capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else "UNKNOWN"


ProductionEngine = Callable[[Path, Path, bool, bool], int]


def _validated_engine(config: Path, output: Path, validate_only: bool, smoke: bool,
                       resources: Any = None) -> int:
    from .p13_equal_budget import run_campaign
    validated = validate_production_config(config, output)
    return run_campaign(config, output, validate_only=validate_only, smoke=smoke,
                        config_override=validated, resources=resources)


def run_production_pipeline(config_path: str | Path, output: str | Path, *,
                            validate_only: bool = False, smoke: bool = False,
                            resume: bool = False, engine: ProductionEngine | None = None,
                            resources: Any = None) -> int:
    """Run the authoritative autonomous synthesis lifecycle.

    Strategic outcomes are completed runs (exit 1); malformed evidence,
    evaluator faults and semantic mismatches are infrastructure failures (2).
    Accepted specialists return 0. Resume deliberately fails closed because the
    validated budget ledger has no durable transaction protocol yet.
    """
    output_path = Path(output)
    started = time.monotonic()
    try:
        if resume:
            raise ValueError("deterministic resume is not supported")
        config = validate_production_config(config_path, output_path)
        if engine is None:
            code = _validated_engine(Path(config_path), output_path, validate_only, smoke,
                                     resources)
        else:
            code = engine(Path(config_path), output_path, validate_only, smoke)
        if code != 0:
            raise RuntimeError(f"autonomous synthesis backend returned {code}")
        metrics = json.loads((output_path / "metrics.json").read_text(encoding="utf-8"))
        if validate_only:
            run = json.loads((output_path / "run.json").read_text(encoding="utf-8"))
            repository_head = _repository_head(config)
            run.update({
                "schema_version": PIPELINE_SCHEMA_VERSION,
                "terminal_verdict": "CONFIG_VALIDATED",
                "repository_head": repository_head,
                "source_provenance": dict(config.get("source_provenance") or {
                    "kind": "git" if repository_head != "UNKNOWN" else "unknown",
                    "source_head": repository_head,
                }),
            })
            _write_document(output_path / "run.json", run)
            return EXIT_ACCEPTED
        metrics = _normalise_metrics(output_path, metrics)
        _normalise_failures(output_path, metrics)
        verdict = _terminal_verdict(metrics)
        finished = time.monotonic()
        _write_document(output_path / "run.json",
                        _production_summary(config, output_path, metrics, verdict, started, finished))
        checkpoints = {family: {"proposed_selected": row.get("checkpoints", 0)}
                       for family, row in sorted(metrics.get("families", {}).items())}
        _write_document(output_path / "checkpoints.json", {"schema_version": 1, "families": checkpoints})
        return EXIT_INFRASTRUCTURE if verdict is TerminalVerdict.INFRASTRUCTURE_FAILURE else (
            EXIT_ACCEPTED if verdict is TerminalVerdict.ACCEPTED_SPECIALIST else EXIT_STRATEGIC)
    except Exception as exc:
        output_path.mkdir(parents=True, exist_ok=True)
        config_source = Path(config_path)
        if config_source.exists() and not (output_path / "config.json").exists():
            try:
                _write_document(output_path / "config.json", json.loads(config_source.read_text(encoding="utf-8")))
            except (OSError, json.JSONDecodeError):
                pass
        if not (output_path / "metrics.json").exists():
            _write_document(output_path / "metrics.json", {"schema_version": 1, "total_cuda_games": 0,
                                                            "semantic_mismatches": 0})
        if not (output_path / "archive_summary.json").exists():
            _write_document(output_path / "archive_summary.json", {"schema_version": 1, "families": {}})
        failure = {"schema_version": PIPELINE_SCHEMA_VERSION, "counts": {"INFRASTRUCTURE_FAILURE": 1},
                   "records": [{"candidate_semantic_id": None, "failure_class": "INFRASTRUCTURE_FAILURE",
                                "pipeline_stage": "PIPELINE", "evidence_references": [],
                                "recommended_next_action": "FIX_INFRASTRUCTURE",
                                "diagnostic": str(exc)}]}
        _write_document(output_path / "failures.json", failure)
        try:
            failure_config = json.loads(config_source.read_text(encoding="utf-8")) if config_source.exists() else {}
        except (OSError, json.JSONDecodeError):
            failure_config = {}
        repository_head = _repository_head(failure_config)
        _write_document(output_path / "run.json", {"schema_version": PIPELINE_SCHEMA_VERSION,
                        "run_id": stable_identity("RLRMB-PRODUCTION-FAILED-RUN-V1", str(output_path)),
                        "terminal_verdict": TerminalVerdict.INFRASTRUCTURE_FAILURE.value,
                        "repository_head": repository_head,
                        "source_provenance": dict(failure_config.get("source_provenance") or {
                            "kind": "git" if repository_head != "UNKNOWN" else "unknown",
                            "source_head": repository_head,
                        }),
                        "error": str(exc)})
        (output_path / "traceback.log").write_text(traceback.format_exc(), encoding="utf-8")
        return EXIT_INFRASTRUCTURE


@dataclass(frozen=True)
class CaseMeta:
    case_id: str
    tape: str
    seat: int
    seed: int
    family: str


@dataclass(frozen=True)
class ExactCaseEvidence:
    meta: CaseMeta
    role: DatasetRole
    baseline_margin: int
    candidate_margin: int
    legal_context: tuple[tuple[str, int, int], ...]
    gate_context: tuple[tuple[str, int, int], ...]
    exact_signature: tuple[int, ...]
    candidate_id: str

    @property
    def delta(self) -> int:
        return self.candidate_margin - self.baseline_margin


def load_case_panel(path: str | Path, label: str) -> tuple[CaseMeta, ...]:
    rows = []
    for index, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines()):
        if not line.strip() or line.startswith("#"):
            continue
        fields = line.split("\t")
        if len(fields) < 3:
            raise ValueError("case panel row requires tape, seat and seed")
        tape = Path(fields[0])
        rows.append(CaseMeta(
            f"{label}:{index}", fields[0], int(fields[1]), int(fields[2]),
            _case_family_from_tape(tape),
        ))
    if not rows:
        raise ValueError("case panel is empty")
    return tuple(rows)


def _case_family_from_tape(tape: Path) -> str:
    """Use fresh self-play policy provenance without assigning replay identities."""
    try:
        source = json.loads(tape.read_text(encoding="utf-8")).get("source", {})
    except (OSError, ValueError, TypeError):
        return tape.stem
    if source.get("kind") != "selfplay":
        return tape.stem
    family = source.get("policy_family")
    policy = source.get("opponent_policy")
    if isinstance(family, str) and family and isinstance(policy, str) and policy:
        return f"{family}:{policy}"
    return tape.stem


def evaluate_candidate_panel(program: MacroProgram, motifs: Mapping[str, Motif], role: DatasetRole,
                             cases_path: str | Path, *, binary: str | Path, asset_dir: str | Path) -> tuple[ExactCaseEvidence, ...]:
    if role is DatasetRole.SEALED:
        raise ValueError("SEALED cannot enter generation-time candidate evaluation")
    meta = load_case_panel(cases_path, role.value.lower())
    batch = CudaBatchEvaluator(binary=binary, asset_dir=asset_dir, cases_path=cases_path)((program,), motifs)
    block = batch.programs[0].effect.blocks[0]
    return tuple(ExactCaseEvidence(row, role, base, base + delta, (), (), (), program.semantic_id)
                 for row, base, delta in zip(meta, block.baseline_margins, block.margin_deltas))


def capture_candidate_legal_states(program: MacroProgram, rows: Sequence[ExactCaseEvidence], cases_path: str | Path,
                                   out_dir: str | Path, *, capture_binary: str | Path, asset_dir: str | Path,
                                   feed_slot: int = 5, feed_count: int = 13, strategic_route_slot: int = 19,
                                   cache: LegalStateCaptureCache | None = None) -> tuple[ExactCaseEvidence, ...]:
    if any(row.candidate_id != program.semantic_id for row in rows):
        raise ValueError("legal-state evidence is bound to a different candidate")
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    probes = out_dir / f"{program.semantic_id}.probes"; capture = out_dir / f"{program.semantic_id}.b4is"
    truth = out_dir / f"{program.semantic_id}.truth_unused.b4is"
    probe_steps = (program.initiation.step, *[program.initiation.step + sum(p.termination.max_steps for p in program.phases[:d.after_phase + 1]) for d in program.decisions])
    probes.write_text("\n".join(map(str, probe_steps)) + "\n", encoding="utf-8")
    asset_dir = Path(asset_dir); cases_path = Path(cases_path)
    cache_key = (str(asset_dir.resolve()), str(cases_path.resolve()), probe_steps,
                 feed_slot, feed_count, strategic_route_slot)
    reusable = cache is not None and not program.decisions and cache_key in cache.paths
    if reusable:
        source_capture, source_truth = cache.paths[cache_key]
        try:
            os.link(source_capture, capture)
            os.link(source_truth, truth)
        except OSError:
            shutil.copy2(source_capture, capture)
            shutil.copy2(source_truth, truth)
        cache.reuses += 1
    else:
        command = [str(capture_binary), "--controller", str(asset_dir / "controller.bkgc"), "--routes", str(asset_dir / "routes.bkv6"),
                   "--legacy-routes", str(asset_dir / "legacy_routes.bkv6"), "--cases", str(cases_path), "--probes", str(probes),
                   "--out", str(capture), "--truth-out", str(truth), "--feed-slot", str(feed_slot), "--feed-count", str(feed_count),
                   "--strategic-route-slot", str(strategic_route_slot)]
        started = time.monotonic()
        subprocess.run(command, check=True, capture_output=True, text=True)
        if cache is not None:
            cache.launches += 1
            cache.seconds += time.monotonic() - started
            if not program.decisions:
                if cache.stable_dir is not None:
                    # Copy to a cache-owned stable location so the entry
                    # survives attempt-level directory renames.
                    cache.stable_dir.mkdir(parents=True, exist_ok=True)
                    eid = cache._entry_count; cache._entry_count += 1
                    sc = cache.stable_dir / f"c{eid:06d}.b4is"
                    st = cache.stable_dir / f"c{eid:06d}_t.b4is"
                    shutil.copy2(capture, sc)
                    shutil.copy2(truth, st)
                    cache.paths[cache_key] = (sc, st)
                else:
                    cache.paths[cache_key] = (capture, truth)
    manifest = capture.with_suffix(".manifest.json")
    manifest.write_text(json.dumps({"schema_version": PIPELINE_SCHEMA_VERSION, "candidate_id": program.semantic_id,
                                    "cases_path": str(Path(cases_path)), "probe_steps": probe_steps}, sort_keys=True) + "\n",
                        encoding="utf-8")
    return load_candidate_legal_states(program, rows, cases_path, capture)


def load_candidate_legal_states(program: MacroProgram, rows: Sequence[ExactCaseEvidence], cases_path: str | Path,
                                capture: str | Path) -> tuple[ExactCaseEvidence, ...]:
    """Bind a persisted exact capture only when its candidate/case manifest agrees."""
    capture = Path(capture)
    manifest_path = capture.with_suffix(".manifest.json")
    if not manifest_path.exists():
        raise ValueError("candidate-specific capture manifest is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != PIPELINE_SCHEMA_VERSION or manifest.get("candidate_id") != program.semantic_id:
        raise ValueError("legal-state capture belongs to a different candidate/version")
    if manifest.get("cases_path") != str(Path(cases_path)):
        raise ValueError("legal-state capture belongs to a different case panel")
    data = load_r0(capture); registry = registry_manifest()
    if data.cases != len(rows) or data.probes[0] != program.initiation.step:
        raise ValueError("candidate capture shape/boundary mismatch")
    _gate_features = frozenset(_GATE_GRAMMAR_FEATURES)
    result = []
    for index, row in enumerate(rows):
        frame = data.frame(index, 0); signature = tuple(int(x) for x in frame[:data.registry_width + data.bb_width + data.compat_width])
        legal_context = tuple((feature, int(args[0]) if args else 0, int(frame[ri])) for ri, (feature, args) in enumerate(registry)
                              if len(args) <= 1)
        gate_context = tuple((feature, idx, val) for feature, idx, val in legal_context
                             if feature in _gate_features)
        result.append(ExactCaseEvidence(row.meta, row.role, row.baseline_margin, row.candidate_margin, legal_context, gate_context, signature, row.candidate_id))
    return tuple(result)


def build_routability_evidence(program: MacroProgram, rows: Sequence[ExactCaseEvidence]) -> RoutabilityEvidence:
    if any(row.candidate_id != program.semantic_id for row in rows):
        raise ValueError("evidence is bound to a different candidate")
    if any(row.role is DatasetRole.SEALED for row in rows):
        raise ValueError("SEALED cannot enter routability evidence")
    trajectories = []
    for row in rows:
        if not row.exact_signature or not row.legal_context:
            raise ValueError("routability requires exact captured legal state")
        continuation = program.semantic_id if row.delta > 0 else "PARENT"
        trajectories.append(TrajectoryEvidence(
            row.meta.case_id, row.role,
            (BoundaryState(program.initiation.step, row.exact_signature, row.legal_context, row.gate_context),),
            (RequiredContinuation(program.initiation.step, continuation),), row.delta,
            row.meta.family, row.meta.family, row.meta.seat, required_support=row.role is DatasetRole.TARGET and row.delta > 0,
        ))
    return RoutabilityEvidence(program.semantic_id, program.initiation.step, tuple(trajectories))


def build_cegis_input(program: MacroProgram, rows: Sequence[ExactCaseEvidence], routing: RoutabilityResult) -> CegisAdapterInput:
    cases = []
    target_positive = [row for row in rows if row.role is DatasetRole.TARGET and row.delta > 0]
    for row in rows:
        required = row in target_positive
        l2w = row.baseline_margin < 0 < row.candidate_margin
        cases.append(AdapterCase(row.meta.case_id, row.role, row.baseline_margin, row.delta, row.legal_context,
                                 row.exact_signature, row.meta.family, row.meta.family, row.meta.seat,
                                 required, required and l2w, stable_identity("RLRMB-PIPELINE-EVIDENCE-V1", row.meta.case_id, row.candidate_id)))
    return CegisAdapterInput(program.semantic_id, program.semantic_id, program.schema_version, len(program.decisions), tuple(cases), routing)


def certify_candidate(program: MacroProgram, rows: Sequence[ExactCaseEvidence]) -> tuple[RoutabilityResult, CegisAdapterResult | None]:
    routing = screen_routability(build_routability_evidence(program, rows))
    if routing.classification.value != "ROUTABLE_CANDIDATE":
        return routing, None
    return routing, run_hardened_cegis(build_cegis_input(program, rows, routing))


def _gate_conditions(gate: Any) -> tuple[tuple[ConditionGene, ...], ...]:
    if not gate:
        raise ValueError("accepted CEGIS result has no outer gate")
    return tuple(tuple(ConditionGene(str(c["feature"]), str(c["cmp"]), int(c.get("index", 0)), int(c["value"]))
                       for c in clause) for clause in gate)


def emit_candidate_route(program: MacroProgram, motifs: Mapping[str, Motif], parent_route: str | Path,
                         destination: str | Path) -> Path:
    """Emit the deterministic actuator route for a fixed-boundary v1 program.

    This is a transport artifact derived from semantic motif content, never a
    historical specialist declaration. State/event/reactive programs continue
    to use their explicit state mapping and therefore fail closed here.
    """
    if program.schema_version != 1 or program.decisions or any(p.termination.kind.value != "FIXED_CAP" for p in program.phases):
        raise ValueError("static route emission requires an open-loop fixed-boundary program")
    actions = [canonical_action(a) for a in load_route_actions(parent_route)]
    cursor = program.initiation.step
    for phase in program.phases:
        if phase.action_source.value == "PARENT":
            cursor += phase.termination.max_steps
            continue
        motif = motifs.get(phase.motif_id or "")
        if motif is None or motif.source_length < phase.termination.max_steps:
            raise ValueError("route emission is missing finite motif content")
        source_actions = motif.semantic_document()["action_block"]
        selected = {family.value for family in phase.action_families}
        for offset in range(phase.termination.max_steps):
            index = cursor + offset
            if index >= len(actions):
                raise ValueError("program exceeds parent route")
            parent, source = actions[index], source_actions[offset]
            actions[index] = {
                "farmer": source["farmer"] if "FARMER" in selected else parent["farmer"],
                "hands": source["hands"] if "HANDS" in selected else parent["hands"],
                "market": source["market"] if "MARKET" in selected else parent["market"],
            }
        cursor += phase.termination.max_steps
    destination = Path(destination); destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps({"actions": actions}, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    return destination


def materialize_certified_candidate(program: MacroProgram, motifs: Mapping[str, Motif], result: CegisAdapterResult,
                                    *, parent_route: str | Path, output_dir: str | Path, route_id: int,
                                    base_route_id: int = 64,
                                    parent_module: Mapping[str, Any] | None = None,
                                    authority_name: str = "strategic_route",
                                    authority_slot: str = "day2_capital_specialist_v1_route_override") -> tuple[MaterializationArtifact, Mapping[str, Any]]:
    if result.candidate_id != program.semantic_id or not result.accepted:
        raise ValueError("materialization requires the matching accepted CEGIS result")
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    route_path = emit_candidate_route(program, motifs, parent_route, output_dir / f"{program.semantic_id}.route.json")
    # Route source is deployment-relative so materialization/composer metadata
    # survives cloud attempt-directory atomic renames without stale .pending paths.
    source = MaterializationInput(program.semantic_id, program, _gate_conditions(result.gate), motifs, route_id,
                                  base_route_id, route_path.name, authority_name, authority_slot)
    artifact = materialize_program(source)
    parent = dict(parent_module) if parent_module is not None else {
        "name": "rlrmb_certification_parent",
        "blackboard": [{"name": authority_slot, "type": "I32", "initial": 0, "owner": "parent"}],
        "tree": {"type": "Sequence", "children": []}, "strategic_routes": [],
    }
    composed = compose_specialists(parent, (artifact.specialist,), (),
                                   strategic_authorities=(StrategicAuthority(authority_name, authority_slot),),
                                   output_name=f"certified_{program.semantic_id[:12]}")
    audit = {"materialization_id": artifact.materialization_id, "node_count": artifact.node_count,
             "state_slots": len(artifact.blackboard), "composition_order": composed.report["composition_order"],
             "unresolved_conflicts": composed.report["unresolved_conflicts"], "semantic_mismatches": 0}
    (output_dir / "materialization.json").write_text(artifact.canonical_json() + "\n", encoding="utf-8")
    (output_dir / "composition.json").write_text(json.dumps(audit, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    (output_dir / "composed_module.json").write_text(
        json.dumps(composed.module, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    (output_dir / "specialist.json").write_text(
        json.dumps(artifact.specialist.canonical(), sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return artifact, audit
