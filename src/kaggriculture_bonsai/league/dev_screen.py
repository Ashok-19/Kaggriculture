"""INTERNAL_DEV cross-regime generalization screen.

RLRMB discovery previously ranked and promoted specialists using BUILD evidence
alone, so a candidate that merely memorized the BUILD/league failure states was
indistinguishable from one that had learned a transferable correction.  Every
such candidate then consumed full downstream TRANSFER budget before the true
holdout firewall rejected it.

This module inserts a *screen* — not a second promotion architecture, and not a
holdout — between BUILD discovery and the unchanged Phase-1 downstream path:

    BUILD discovery -> INTERNAL_DEV screen -> routing/CEGIS -> TRANSFER/VALIDATION/SEALED

INTERNAL_DEV panels are generated independently from BUILD by
``adversarial_league.generate_league_dev_cases`` (different seeds, seats,
league member subset and tapes) and rotate deterministically per generation so
candidates cannot permanently specialize to one DEV panel.  The screen has no
authority to accept anything: it can only reorder candidates and withhold
downstream budget from candidates with obvious broad cross-regime regressions.
TRANSFER, VALIDATION, SEALED and final composition remain the sole safety
authority and are unchanged.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence

from kaggriculture_bonsai.search.effects import EffectBlock

DEV_SCREEN_SCHEMA_VERSION = 1
DEV_ROLE = "INTERNAL_DEV"


class DevScreenStatus(str, Enum):
    """Outcome of the bounded cross-regime screen for one candidate."""

    DEV_PASS = "DEV_PASS"
    DEV_WEAK_PASS = "DEV_WEAK_PASS"
    DEV_NO_CROSS_REGIME_EFFECT = "DEV_NO_CROSS_REGIME_EFFECT"
    DEV_REGRESSION = "DEV_REGRESSION"
    DEV_FALLBACK_ADMITTED = "DEV_FALLBACK_ADMITTED"
    DEV_UNAVAILABLE = "DEV_UNAVAILABLE"


#: Statuses that may consume downstream TRANSFER budget.
DOWNSTREAM_ELIGIBLE = frozenset({
    DevScreenStatus.DEV_PASS,
    DevScreenStatus.DEV_WEAK_PASS,
    DevScreenStatus.DEV_FALLBACK_ADMITTED,
    DevScreenStatus.DEV_UNAVAILABLE,
})

#: Rejection reasons, kept distinct so failure analysis can separate a
#: BUILD-only overfit from an actively harmful cross-regime candidate.
REJECTION_REASONS = {
    DevScreenStatus.DEV_NO_CROSS_REGIME_EFFECT: "BUILD_ONLY_OVERFIT",
    DevScreenStatus.DEV_REGRESSION: "DEV_REGRESSION",
}


@dataclass(frozen=True)
class DevScreenPolicy:
    """Bounded, deterministic screening thresholds.

    The thresholds are deliberately permissive: DEV is a screen, and requiring
    perfection here would collapse discovery without adding any safety that
    TRANSFER/VALIDATION/SEALED do not already provide.
    """

    max_win_to_loss: int = 2
    max_win_to_loss_rate: float = 0.25
    max_screened: int = 8
    max_downstream: int = 3
    allow_fallback: bool = True

    def __post_init__(self) -> None:
        if self.max_win_to_loss < 0 or not 0.0 <= self.max_win_to_loss_rate <= 1.0:
            raise ValueError("DEV regression thresholds must be non-negative and rate in [0,1]")
        if self.max_screened < 1 or self.max_downstream < 1:
            raise ValueError("DEV screen bounds must be positive")

    @classmethod
    def from_config(cls, config: Mapping[str, Any] | None) -> "DevScreenPolicy":
        config = dict(config or {})
        return cls(
            max_win_to_loss=int(config.get("max_win_to_loss", 2)),
            max_win_to_loss_rate=float(config.get("max_win_to_loss_rate", 0.25)),
            max_screened=int(config.get("max_screened", 8)),
            max_downstream=int(config.get("max_downstream", 3)),
            allow_fallback=bool(config.get("allow_fallback", True)),
        )


@dataclass(frozen=True)
class DevPanelSelection:
    """Deterministic per-generation choice of one INTERNAL_DEV panel."""

    panel: str
    generation: int
    rotation_index: int
    available_panels: tuple[str, ...]

    def to_document(self) -> dict[str, Any]:
        return {
            "panel": self.panel,
            "generation": self.generation,
            "rotation_index": self.rotation_index,
            "available_panels": list(self.available_panels),
        }


def select_dev_panel(panels: Sequence[str], generation: int) -> DevPanelSelection:
    """Rotate DEV panels deterministically so no panel becomes a fixed target."""
    rows = tuple(str(row) for row in panels if str(row).strip())
    if not rows:
        raise ValueError("INTERNAL_DEV rotation requires at least one panel")
    if generation < 0:
        raise ValueError("generation must be non-negative")
    index = generation % len(rows)
    return DevPanelSelection(rows[index], generation, index, rows)


def assert_dev_independence(build_panels: Sequence[Path], dev_panel: Path) -> dict[str, Any]:
    """Fail closed if a DEV panel reuses any BUILD case.

    DEV must measure generalization, so overlap with the discovery distribution
    would silently reintroduce exactly the BUILD-only bias it screens for.
    """
    def _cases(path: Path) -> set[str]:
        if not path.exists():
            return set()
        return {
            line.split("\t")[0].strip()
            for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
        }

    def _seeds(path: Path) -> set[int]:
        if not path.exists():
            return set()
        seeds = set()
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip() or line.startswith("#"):
                continue
            fields = line.split("\t")
            if len(fields) >= 3:
                seeds.add(int(fields[2]))
        return seeds

    dev_cases = _cases(Path(dev_panel))
    if not dev_cases:
        raise ValueError("INTERNAL_DEV panel is empty")
    build_cases: set[str] = set()
    for path in build_panels:
        build_cases |= _cases(Path(path))
    overlap = sorted(build_cases & dev_cases)
    if overlap:
        raise ValueError(f"INTERNAL_DEV panel reuses BUILD cases: {overlap[:3]}")
    build_seeds = set().union(*(_seeds(Path(path)) for path in build_panels))
    dev_seeds = _seeds(Path(dev_panel))
    seed_overlap = sorted(build_seeds & dev_seeds)
    if seed_overlap:
        raise ValueError(f"INTERNAL_DEV panel reuses BUILD seeds: {seed_overlap[:3]}")
    return {
        "dev_cases": len(dev_cases),
        "build_cases": len(build_cases),
        "overlap": 0,
        "seed_overlap": 0,
        "independent": True,
    }


@dataclass(frozen=True)
class DevScreenResult:
    candidate_semantic_id: str
    family: str
    lane: str
    status: DevScreenStatus
    dev_panel: str
    generation: int
    rotation_index: int
    dev_cases: int
    dev_activated: int
    dev_loss_to_win: int
    dev_win_to_loss: int
    dev_positive: int
    dev_harmful: int
    dev_aggregate_margin: int
    dev_worst_delta: int
    build_loss_to_win: int
    build_protected_win_to_loss: int
    provenance: dict[str, Any] = field(default_factory=dict)

    @property
    def win_to_loss_rate(self) -> float:
        return self.dev_win_to_loss / self.dev_activated if self.dev_activated else 0.0

    @property
    def downstream_eligible(self) -> bool:
        return self.status in DOWNSTREAM_ELIGIBLE

    @property
    def rejection_reason(self) -> str:
        return REJECTION_REASONS.get(self.status, "")

    def to_document(self) -> dict[str, Any]:
        return {
            "schema_version": DEV_SCREEN_SCHEMA_VERSION,
            "role": DEV_ROLE,
            "candidate_semantic_id": self.candidate_semantic_id,
            "family": self.family,
            "lane": self.lane,
            "status": self.status.value,
            "downstream_eligible": self.downstream_eligible,
            "rejection_reason": self.rejection_reason,
            "dev_panel": self.dev_panel,
            "generation": self.generation,
            "rotation_index": self.rotation_index,
            "dev_cases": self.dev_cases,
            "dev_activated": self.dev_activated,
            "dev_loss_to_win": self.dev_loss_to_win,
            "dev_win_to_loss": self.dev_win_to_loss,
            "dev_win_to_loss_rate": round(self.win_to_loss_rate, 6),
            "dev_positive": self.dev_positive,
            "dev_harmful": self.dev_harmful,
            "dev_aggregate_margin": self.dev_aggregate_margin,
            "dev_worst_delta": self.dev_worst_delta,
            "build_loss_to_win": self.build_loss_to_win,
            "build_protected_win_to_loss": self.build_protected_win_to_loss,
            "provenance": dict(sorted(self.provenance.items())),
        }


def dev_block_metrics(block: EffectBlock) -> dict[str, int]:
    """Exact INTERNAL_DEV outcome counts, without conflating harm and W->L."""
    if not isinstance(block, EffectBlock):
        raise TypeError("DEV metrics require an EffectBlock")
    baselines = block.baseline_margins
    deltas = block.margin_deltas
    candidates = block.candidate_margins
    return {
        "dev_cases": len(deltas),
        "dev_activated": sum(1 for delta in deltas if delta != 0),
        "dev_loss_to_win": sum(1 for base, cand in zip(baselines, candidates) if base < 0 < cand),
        "dev_win_to_loss": sum(1 for base, cand in zip(baselines, candidates) if base > 0 > cand),
        "dev_positive": sum(1 for delta in deltas if delta > 0),
        "dev_harmful": sum(1 for delta in deltas if delta < 0),
        "dev_aggregate_margin": sum(deltas),
        "dev_worst_delta": min(deltas),
    }


def classify_dev_outcome(metrics: Mapping[str, int], policy: DevScreenPolicy) -> DevScreenStatus:
    """Classify a candidate's cross-regime behaviour.

    An inert candidate is not "safe": it is the signature of a BUILD-only
    overfit, because a genuine correction touches the same weakness class in an
    independently generated regime.
    """
    activated = int(metrics["dev_activated"])
    win_to_loss = int(metrics["dev_win_to_loss"])
    rate = win_to_loss / activated if activated else 0.0
    if activated == 0:
        return DevScreenStatus.DEV_NO_CROSS_REGIME_EFFECT
    if win_to_loss > policy.max_win_to_loss or rate > policy.max_win_to_loss_rate:
        return DevScreenStatus.DEV_REGRESSION
    if int(metrics["dev_loss_to_win"]) >= 1 and int(metrics["dev_aggregate_margin"]) >= 0:
        return DevScreenStatus.DEV_PASS
    return DevScreenStatus.DEV_WEAK_PASS


_STATUS_RANK = {
    DevScreenStatus.DEV_PASS: 0,
    DevScreenStatus.DEV_WEAK_PASS: 1,
    DevScreenStatus.DEV_FALLBACK_ADMITTED: 2,
    DevScreenStatus.DEV_UNAVAILABLE: 3,
    DevScreenStatus.DEV_NO_CROSS_REGIME_EFFECT: 4,
    DevScreenStatus.DEV_REGRESSION: 5,
}


def rank_dev_results(results: Sequence[DevScreenResult]) -> tuple[DevScreenResult, ...]:
    """Deterministic ranking: cross-regime evidence first, BUILD gain second."""
    return tuple(sorted(results, key=lambda row: (
        _STATUS_RANK[row.status],
        row.dev_win_to_loss,
        -row.dev_loss_to_win,
        -row.dev_aggregate_margin,
        -row.build_loss_to_win,
        row.candidate_semantic_id,
    )))


def apply_dev_screen(
    results: Sequence[DevScreenResult], policy: DevScreenPolicy,
) -> tuple[tuple[DevScreenResult, ...], tuple[DevScreenResult, ...]]:
    """Return (admitted, withheld) downstream candidates.

    Candidates with obvious broad DEV regressions never consume downstream
    budget.  If the screen would otherwise admit nobody, a single bounded
    fallback slot keeps discovery alive — but a DEV_REGRESSION is never the
    fallback, since that is the failure mode this screen exists to stop.
    """
    ranked = rank_dev_results(results)
    admitted = [row for row in ranked if row.downstream_eligible][:policy.max_downstream]
    if not admitted and policy.allow_fallback:
        fallback = [row for row in ranked if row.status is not DevScreenStatus.DEV_REGRESSION]
        if fallback:
            row = fallback[0]
            admitted = [DevScreenResult(**{
                **{key: getattr(row, key) for key in row.__dataclass_fields__},
                "status": DevScreenStatus.DEV_FALLBACK_ADMITTED,
            })]
    admitted_ids = {row.candidate_semantic_id for row in admitted}
    withheld = tuple(row for row in ranked if row.candidate_semantic_id not in admitted_ids)
    return tuple(admitted), withheld


class DevScreenMetrics:
    """Bounded counters for the DEV screen, reported alongside Phase-1 metrics."""

    def __init__(self) -> None:
        self.by_status: dict[str, int] = {status.value: 0 for status in DevScreenStatus}
        self.attempted = 0
        self.admitted = 0
        self.withheld = 0
        self.cuda_games = 0

    def observe(self, result: DevScreenResult) -> None:
        self.attempted += 1
        self.by_status[result.status.value] += 1

    def observe_admission(self, admitted: Sequence[DevScreenResult],
                          withheld: Sequence[DevScreenResult]) -> None:
        self.admitted = len(admitted)
        self.withheld = len(withheld)
        for row in admitted:
            if row.status is DevScreenStatus.DEV_FALLBACK_ADMITTED:
                self.by_status[row.status.value] += 1

    def to_document(self) -> dict[str, Any]:
        passes = (self.by_status[DevScreenStatus.DEV_PASS.value]
                  + self.by_status[DevScreenStatus.DEV_WEAK_PASS.value])
        failures = (self.by_status[DevScreenStatus.DEV_REGRESSION.value]
                    + self.by_status[DevScreenStatus.DEV_NO_CROSS_REGIME_EFFECT.value])
        return {
            "schema_version": DEV_SCREEN_SCHEMA_VERSION,
            "role": DEV_ROLE,
            "authority": "SCREENING_ONLY",
            "dev_attempted": self.attempted,
            "dev_passes": passes,
            "dev_failures": failures,
            "dev_admitted_downstream": self.admitted,
            "dev_withheld_downstream": self.withheld,
            "dev_cuda_games": self.cuda_games,
            "by_status": dict(sorted(self.by_status.items())),
        }


def write_dev_screen_artifact(
    directory: Path, selection: DevPanelSelection, policy: DevScreenPolicy,
    results: Sequence[DevScreenResult], admitted: Sequence[DevScreenResult],
    metrics: DevScreenMetrics, independence: Mapping[str, Any],
    parent_generation_context: str = "",
) -> dict[str, Any]:
    directory.mkdir(parents=True, exist_ok=True)
    document = {
        "schema_version": DEV_SCREEN_SCHEMA_VERSION,
        "role": DEV_ROLE,
        "authority": "SCREENING_ONLY",
        "holdout_authority_unchanged": True,
        "parent_generation_context": parent_generation_context,
        "panel_selection": selection.to_document(),
        "independence": dict(sorted(independence.items())),
        "policy": {
            "max_win_to_loss": policy.max_win_to_loss,
            "max_win_to_loss_rate": policy.max_win_to_loss_rate,
            "max_screened": policy.max_screened,
            "max_downstream": policy.max_downstream,
            "allow_fallback": policy.allow_fallback,
        },
        "metrics": metrics.to_document(),
        "admitted": [row.candidate_semantic_id for row in admitted],
        "results": [row.to_document() for row in rank_dev_results(results)],
    }
    (directory / "dev_screen.json").write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return document
