from __future__ import annotations

import argparse
import copy
import dataclasses
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from tools.bkg4_search.bonsai_composer import (
    COMPOSER_KIND,
    SpecialistDeclaration,
    StrategicAuthority,
    compose_declaration,
)
from tools.bkg4_search.bounded_macro import BoundedRouteMacro
from tools.bkg4_search.cuda_io import load_fragment_results
from tools.bkg4_search.information_state_io import R0Capture, load_r0, registry_manifest
from tools.bkg4_search.macro_initiation_cegis import (
    GateScore,
    MacroEffectCase,
    enumerate_safe_conjunctions,
    evaluate_gate,
    gate_hits,
    selected_mixed_exact_cells,
)
from tools.bkg4_search.module_codec import ConditionGene
from tools.bkg4_search.regret_macro_search import load_grid

ROOT = Path(__file__).resolve().parents[2]

ACCEPTED = "ACCEPTED"
REJECTED = "REJECTED"
# Backward-compatible internal name; terminal machine verdict is the required ACCEPTED.
ACCEPTED_RESEARCH_CANDIDATE = ACCEPTED
NO_SAFE_SPECIALIST_FOUND = "NO_SAFE_SPECIALIST_FOUND"
SEARCH_BUDGET_EXHAUSTED = "SEARCH_BUDGET_EXHAUSTED"
INFRASTRUCTURE_FAILURE = "INFRASTRUCTURE_FAILURE"

NO_COUNTERFACTUAL_GAIN = "NO_COUNTERFACTUAL_GAIN"
NON_TRANSFERABLE = "NON_TRANSFERABLE"
MIXED_LEGAL_STATE = "MIXED_LEGAL_STATE"
HINDSIGHT_DEPENDENT = "HINDSIGHT_DEPENDENT"
NO_SAFE_GATE = "NO_SAFE_GATE"
HOLDOUT_FAILURE = "HOLDOUT_FAILURE"
THRESHOLD_BRITTLE = "THRESHOLD_BRITTLE"
COMPOSITION_CONFLICT = "COMPOSITION_CONFLICT"
PROTECTED_REGRESSION = "PROTECTED_REGRESSION"
CPU_CUDA_MISMATCH = "CPU_CUDA_MISMATCH"
CUDA_FAULT = "CUDA_FAULT"
TARGET_CAPABILITY_LOST = "TARGET_CAPABILITY_LOST"

FAILURE_REASONS = {
    NO_COUNTERFACTUAL_GAIN,
    NON_TRANSFERABLE,
    MIXED_LEGAL_STATE,
    HINDSIGHT_DEPENDENT,
    NO_SAFE_GATE,
    HOLDOUT_FAILURE,
    THRESHOLD_BRITTLE,
    COMPOSITION_CONFLICT,
    PROTECTED_REGRESSION,
    CPU_CUDA_MISMATCH,
    CUDA_FAULT,
    TARGET_CAPABILITY_LOST,
    SEARCH_BUDGET_EXHAUSTED,
}

_FORBIDDEN_RUNTIME_TERMS = (
    "opponent_team",
    "team",
    "submission",
    "replay",
    "episode",
    "seed",
    "tape",
    "filename",
    "rank",
)

# P0.6 intentionally begins with compact high-level public state. Full legal-state
# signatures still include every registry/BB/compat value for hindsight detection;
# only gate proposal is restricted to this causal/auditable vocabulary.
_P0_6_GATE_FEATURES = frozenset({
    "DAY", "HOUR", "OWN_MONEY", "OPP_MONEY", "OWN_UNITS", "OPP_UNITS",
    "OWN_LAND", "OPP_LAND", "OWN_HIRES_TODAY", "OPP_HIRES_TODAY",
    "OWN_SHED_TOTAL", "SHOP_COUNT", "MARKET_PRICE", "MARKET_INVENTORY",
    "OWN_SHED", "OWN_SEED", "SHOP_AT", "SHOP_SEEN",
    "OWN_QUADRANT_UNLOCKED", "OPP_QUADRANT_UNLOCKED",
})
_P0_6_CEGIS_FEATURE_BUDGET = 24
_P0_6_CEGIS_SINGLETON_TARGET_BUDGET = 4


def _sha(value: Any, prefix: bytes = b"P0.6\0") -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(prefix + raw).hexdigest()


def _margin(row: Sequence[int]) -> int:
    return int(row[0]) - int(row[1])


def _route_id_from_path(path: str | Path) -> int:
    match = re.search(r"route[_-]?(\d+)", Path(path).stem, re.IGNORECASE)
    if not match:
        raise ValueError(f"cannot infer source route id from {path}")
    return int(match.group(1))


def _jsonable_condition(c: ConditionGene) -> dict[str, Any]:
    return {"feature": c.feature, "cmp": c.cmp, "index": int(c.index), "value": int(c.value)}


def _canonical_gate(gate: Sequence[Sequence[ConditionGene]]) -> list[list[dict[str, Any]]]:
    return [[_jsonable_condition(c) for c in clause] for clause in gate]


def _identity_like_runtime_feature(feature: str) -> bool:
    lowered = feature.lower()
    return any(
        term in lowered
        for term in _FORBIDDEN_RUNTIME_TERMS
        if not (term == "seed" and feature == "OWN_SEED")
    )


def validate_identity_free_gate(gate: Sequence[Sequence[ConditionGene]]) -> None:
    for clause in gate:
        if not clause:
            raise ValueError("gate clause cannot be empty")
        for condition in clause:
            if _identity_like_runtime_feature(condition.feature):
                raise ValueError(f"identity-like runtime feature forbidden: {condition.feature}")
            condition.validate()


@dataclass(frozen=True)
class CaseRecord:
    case_id: str
    path: str
    seat: int
    seed: int
    family: int
    protected: int
    own_history: int
    role: int

    @property
    def trajectory_key(self) -> tuple[str, int]:
        return self.path, int(self.seed)

    @classmethod
    def from_fields(cls, fields: Sequence[str], index: int) -> "CaseRecord":
        if len(fields) not in (6, 7):
            raise ValueError("case TSV row requires six or seven fields")
        return cls(
            case_id=f"case_{index:04d}",
            path=str(fields[0]),
            seat=int(fields[1]),
            seed=int(fields[2]),
            family=int(fields[3]),
            protected=int(fields[4]),
            own_history=int(fields[5]),
            role=int(fields[6]) if len(fields) == 7 else 0,
        )

    def tsv(self) -> str:
        return "\t".join(
            map(str, (self.path, self.seat, self.seed, self.family, self.protected, self.own_history, self.role))
        )


def load_case_bank(path: str | Path) -> list[CaseRecord]:
    rows: list[CaseRecord] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        rows.append(CaseRecord.from_fields(line.split("\t"), len(rows)))
    if not rows:
        raise ValueError(f"empty case bank: {path}")
    return rows


@dataclass(frozen=True)
class FrontierEntry:
    family: int
    case_ids: tuple[str, ...]
    margins: tuple[int, ...]
    losses: int
    severity: int
    independent_trajectories: int
    both_seat_trajectories: int
    priority_score: tuple[int, ...]
    grouping_features: tuple[str, ...] = ("offline_family", "severity", "recurrence", "both_seat_recurrence")

    def canonical(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def rank_loss_frontier(
    cases: Sequence[CaseRecord],
    baseline_margins: Sequence[int],
    *,
    max_families: int,
) -> list[FrontierEntry]:
    if len(cases) != len(baseline_margins):
        raise ValueError("case/margin count mismatch")
    grouped: dict[int, list[tuple[CaseRecord, int]]] = defaultdict(list)
    for row, margin in zip(cases, baseline_margins):
        if int(margin) < 0:
            grouped[int(row.family)].append((row, int(margin)))
    entries: list[FrontierEntry] = []
    for family, rows in grouped.items():
        rows.sort(key=lambda x: (x[0].case_id, x[0].path, x[0].seat, x[0].seed))
        severity = sum(-margin for _, margin in rows)
        by_trajectory: dict[tuple[str, int], set[int]] = defaultdict(set)
        for row, _ in rows:
            by_trajectory[row.trajectory_key].add(int(row.seat))
        both = sum(seats == {0, 1} for seats in by_trajectory.values())
        independent = len(by_trajectory)
        # Both-seat recurrence and recurrence are deliberately ahead of raw money severity.
        score = (both, len(rows), independent, severity, -int(family))
        entries.append(
            FrontierEntry(
                family=int(family),
                case_ids=tuple(row.case_id for row, _ in rows),
                margins=tuple(margin for _, margin in rows),
                losses=len(rows),
                severity=severity,
                independent_trajectories=independent,
                both_seat_trajectories=both,
                priority_score=score,
            )
        )
    entries.sort(key=lambda x: x.priority_score, reverse=True)
    return entries[: max(0, int(max_families))]


@dataclass(frozen=True)
class TargetSelection:
    family: int
    cases: tuple[CaseRecord, ...]
    baseline_margins: tuple[int, ...]
    grouping_features: tuple[str, ...]
    independent_support: int
    both_seat_support: int
    search_priority_score: tuple[int, ...]

    def canonical(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "case_ids": [x.case_id for x in self.cases],
            "baseline_margins": list(self.baseline_margins),
            "grouping_features": list(self.grouping_features),
            "independent_trajectories": self.independent_support,
            "both_seat_support": self.both_seat_support,
            "search_priority_score": list(self.search_priority_score),
        }


def build_target_cases(
    frontier: FrontierEntry,
    cases: Sequence[CaseRecord],
    baseline_margins: Sequence[int],
    *,
    max_cases: int,
) -> TargetSelection:
    by_id = {row.case_id: (row, int(margin)) for row, margin in zip(cases, baseline_margins)}
    selected = [by_id[cid] for cid in frontier.case_ids if cid in by_id and by_id[cid][1] < 0]
    selected.sort(key=lambda x: (x[1], x[0].case_id))
    selected = selected[:max_cases]
    trajectories: dict[tuple[str, int], set[int]] = defaultdict(set)
    for row, _ in selected:
        trajectories[row.trajectory_key].add(row.seat)
    return TargetSelection(
        family=frontier.family,
        cases=tuple(row for row, _ in selected),
        baseline_margins=tuple(m for _, m in selected),
        grouping_features=frontier.grouping_features,
        independent_support=len(trajectories),
        both_seat_support=sum(v == {0, 1} for v in trajectories.values()),
        search_priority_score=frontier.priority_score,
    )


@dataclass(frozen=True)
class SearchGridRow:
    candidate_id: int
    donor_index: int
    donor_path: str
    donor_route_id: int
    start: int
    duration: int

    def canonical(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def build_search_grid(
    donor_paths: Sequence[str],
    starts: Sequence[int],
    durations: Sequence[int],
    *,
    max_candidates: int,
) -> list[SearchGridRow]:
    rows = [SearchGridRow(0, -1, "BASE", -1, 0, 0)]
    for donor_index, donor_path in enumerate(donor_paths):
        try:
            route_id = _route_id_from_path(donor_path)
        except ValueError:
            route_id = donor_index
        for start in sorted({int(x) for x in starts}):
            for duration in sorted({int(x) for x in durations}):
                if not 0 <= start < 719 or duration <= 0 or start + duration > 719:
                    continue
                rows.append(SearchGridRow(len(rows), donor_index, str(donor_path), route_id, start, duration))
    if len(rows) > int(max_candidates):
        raise ValueError(f"search grid exceeds candidate budget: {len(rows)} > {max_candidates}")
    return rows


@dataclass(frozen=True)
class CandidateEvidence:
    candidate_id: int
    donor_route_id: int
    donor_path: str
    start: int
    duration: int
    baseline_margins: tuple[int, ...]
    deltas: tuple[int, ...]
    action_changes: tuple[int, ...]
    positive: int = 0
    harmful: int = 0
    loss_to_win: int = 0
    win_to_loss: int = 0
    independent_positive_support: int = 0
    both_seat_positive_support: int = 0
    sum_delta: int = 0
    worst_delta: int = 0

    @property
    def margins(self) -> tuple[int, ...]:
        return tuple(a + b for a, b in zip(self.baseline_margins, self.deltas))

    def transferable(self, *, min_positive: int, min_independent: int) -> bool:
        if self.positive < min_positive or self.independent_positive_support < min_independent:
            return False
        return self.loss_to_win > 0 or self.positive >= max(min_positive + 1, 3)

    def canonical(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass(frozen=True)
class TargetCapabilityReport:
    accepted: bool
    failure_reason: str | None
    case_ids: tuple[str, ...]
    gate_hits: tuple[bool, ...]
    parent_margins: tuple[int, ...]
    teacher_margins: tuple[int, ...]
    final_margins: tuple[int, ...]
    final_deltas: tuple[int, ...]
    records: int
    activations: int
    positive: int
    harmful: int
    loss_to_win: int
    win_to_loss: int
    discovery_positive: int
    discovery_loss_to_win: int
    preserved_positive: int
    preserved_loss_to_win: int
    independent_positive_support: int
    both_seat_positive_support: int
    worst_delta: int
    aggregate_delta: int

    def canonical(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def assess_target_capability(
    cases: Sequence[CaseRecord],
    discovery: CandidateEvidence,
    final_margins: Sequence[int],
    gate_active: Sequence[bool],
) -> TargetCapabilityReport:
    if not (len(cases) == len(discovery.baseline_margins) == len(discovery.deltas) == len(final_margins) == len(gate_active)):
        raise ValueError("target capability count mismatch")
    parent = tuple(int(x) for x in discovery.baseline_margins)
    teacher = tuple(int(p + d) for p, d in zip(parent, discovery.deltas))
    final = tuple(int(x) for x in final_margins)
    active = tuple(bool(x) for x in gate_active)
    deltas = tuple(int(f - p) for p, f in zip(parent, final))
    discovery_positive_mask = tuple(int(d) > 0 for d in discovery.deltas)
    discovery_l2w_mask = tuple(p < 0 < t for p, t in zip(parent, teacher))
    preserved_positive_mask = tuple(
        a and dp and delta > 0
        for a, dp, delta in zip(active, discovery_positive_mask, deltas)
    )
    preserved_l2w_mask = tuple(
        a and dl2w and p < 0 < f
        for a, dl2w, p, f in zip(active, discovery_l2w_mask, parent, final)
    )
    positive_trajectories: dict[tuple[str, int], set[int]] = defaultdict(set)
    for row, keep in zip(cases, preserved_positive_mask):
        if keep:
            positive_trajectories[row.trajectory_key].add(int(row.seat))
    harmful = sum(delta < 0 for delta in deltas)
    win_to_loss = sum(p > 0 > f for p, f in zip(parent, final))
    preserved_positive = sum(preserved_positive_mask)
    preserved_l2w = sum(preserved_l2w_mask)
    requires_l2w = any(discovery_l2w_mask)
    accepted = (
        harmful == 0
        and win_to_loss == 0
        and preserved_positive > 0
        and (not requires_l2w or preserved_l2w > 0)
    )
    return TargetCapabilityReport(
        accepted=accepted,
        failure_reason=None if accepted else TARGET_CAPABILITY_LOST,
        case_ids=tuple(row.case_id for row in cases),
        gate_hits=active,
        parent_margins=parent,
        teacher_margins=teacher,
        final_margins=final,
        final_deltas=deltas,
        records=len(cases),
        activations=sum(active),
        positive=sum(delta > 0 for delta in deltas),
        harmful=harmful,
        loss_to_win=sum(p < 0 < f for p, f in zip(parent, final)),
        win_to_loss=win_to_loss,
        discovery_positive=sum(discovery_positive_mask),
        discovery_loss_to_win=sum(discovery_l2w_mask),
        preserved_positive=preserved_positive,
        preserved_loss_to_win=preserved_l2w,
        independent_positive_support=len(positive_trajectories),
        both_seat_positive_support=sum(seats == {0, 1} for seats in positive_trajectories.values()),
        worst_delta=min(deltas, default=0),
        aggregate_delta=sum(deltas),
    )


def _candidate_with_support(candidate: CandidateEvidence, cases: Sequence[CaseRecord]) -> CandidateEvidence:
    if len(candidate.deltas) != len(cases) or len(candidate.baseline_margins) != len(cases):
        raise ValueError("candidate/case count mismatch")
    positive_indices = [i for i, delta in enumerate(candidate.deltas) if delta > 0]
    positive_trajectories: dict[tuple[str, int], set[int]] = defaultdict(set)
    for i in positive_indices:
        positive_trajectories[cases[i].trajectory_key].add(cases[i].seat)
    margins = candidate.margins
    return dataclasses.replace(
        candidate,
        positive=len(positive_indices),
        harmful=sum(x < 0 for x in candidate.deltas),
        loss_to_win=sum(b < 0 < m for b, m in zip(candidate.baseline_margins, margins)),
        win_to_loss=sum(b > 0 > m for b, m in zip(candidate.baseline_margins, margins)),
        independent_positive_support=len(positive_trajectories),
        both_seat_positive_support=sum(v == {0, 1} for v in positive_trajectories.values()),
        sum_delta=sum(candidate.deltas),
        worst_delta=min(candidate.deltas, default=0),
    )


def rank_candidates(candidates: Sequence[CandidateEvidence], cases: Sequence[CaseRecord]) -> list[CandidateEvidence]:
    ranked = [_candidate_with_support(c, cases) for c in candidates]
    ranked.sort(
        key=lambda c: (
            -c.loss_to_win,
            -c.positive,
            -c.independent_positive_support,
            -c.both_seat_positive_support,
            -c.sum_delta,
            -c.worst_delta,
            sum(c.action_changes),
            c.duration,
            c.start,
            c.donor_route_id,
            c.candidate_id,
        )
    )
    return ranked


def detect_mixed_legal_states(cases: Sequence[MacroEffectCase]) -> tuple[tuple[str, ...], ...]:
    cells: dict[tuple[int, ...], list[MacroEffectCase]] = defaultdict(list)
    for row in cases:
        if row.exact_signature:
            cells[row.exact_signature].append(row)
    mixed = []
    for rows in cells.values():
        if any(x.delta > 0 for x in rows) and any(x.delta < 0 for x in rows):
            mixed.append(tuple(sorted(x.case_id for x in rows)))
    return tuple(sorted(mixed))


@dataclass(frozen=True)
class CegisIteration:
    iteration: int
    gate: tuple[tuple[ConditionGene, ...], ...]
    score: GateScore
    counterexample_source: str

    def canonical(self) -> dict[str, Any]:
        return {
            "iteration": self.iteration,
            "gate": _canonical_gate(self.gate),
            "score": dataclasses.asdict(self.score),
            "counterexample_source": self.counterexample_source,
        }


@dataclass(frozen=True)
class CegisResult:
    final_gate: tuple[tuple[ConditionGene, ...], ...] | None
    final_score: GateScore
    trace: tuple[CegisIteration, ...]
    mixed_cells: tuple[tuple[str, ...], ...] = ()
    holdout_score: GateScore | None = None

    def canonical(self) -> dict[str, Any]:
        return {
            "final_gate": None if self.final_gate is None else _canonical_gate(self.final_gate),
            "final_score": dataclasses.asdict(self.final_score),
            "trace": [x.canonical() for x in self.trace],
            "mixed_cells": [list(x) for x in self.mixed_cells],
            "holdout_score": None if self.holdout_score is None else dataclasses.asdict(self.holdout_score),
        }


def run_cegis(
    discovery: Sequence[MacroEffectCase],
    counterexamples: Sequence[MacroEffectCase],
    validation: Sequence[MacroEffectCase],
    *,
    max_conditions: int,
    max_iterations: int,
) -> CegisResult:
    # Positive discovery rows and harmful counterexamples are explicit datasets. Neutral
    # rows remain available for support scoring but do not drive contrastive synthesis.
    all_training = list(discovery) + list(counterexamples)
    mixed = detect_mixed_legal_states(all_training)
    empty = GateScore(0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    positives = [x for x in all_training if x.delta > 0]
    harmful = [x for x in all_training if x.delta < 0]
    if not positives:
        return CegisResult(None, empty, (), mixed)
    if not harmful:
        # No observed harm: still derive a legal current-state gate from the positive
        # support by using one deterministic equality/range condition shared by support.
        contexts = [x.context for x in positives]
        shared = sorted(set.intersection(*(set(ctx) for ctx in contexts))) if contexts else []
        candidates: list[tuple[ConditionGene, ...]] = []
        for feature, index in shared:
            vals = [int(ctx[(feature, index)]) for ctx in contexts]
            if len(set(vals)) == 1:
                candidates.append((ConditionGene(feature, "EQ", index, vals[0]),))
            else:
                candidates.append((ConditionGene(feature, "GE", index, min(vals)),))
                candidates.append((ConditionGene(feature, "LE", index, max(vals)),))
        candidates = [x for x in candidates if not _identity_like_runtime_feature(x[0].feature)]
        if not candidates:
            return CegisResult(None, empty, (), mixed)
        scored = [((clause,), evaluate_gate((clause,), all_training)) for clause in candidates]
        scored.sort(key=lambda row: row[1].lexicographic_key())
        gate, score = scored[0]
        validate_identity_free_gate(gate)
        holdout = evaluate_gate(gate, validation) if validation else None
        return CegisResult(gate, score, (CegisIteration(0, gate, score, "discovery"),), mixed, holdout)

    ranked = enumerate_safe_conjunctions(
        all_training,
        max_conditions=max_conditions,
        limit=128,
        feature_budget=_P0_6_CEGIS_FEATURE_BUDGET,
        singleton_l2w_budget=_P0_6_CEGIS_SINGLETON_TARGET_BUDGET,
    )
    trace: list[CegisIteration] = []
    for iteration, (clause, score) in enumerate(ranked[: max_iterations]):
        gate = (tuple(clause),)
        validate_identity_free_gate(gate)
        trace.append(CegisIteration(iteration, gate, score, "counterexample" if harmful else "discovery"))
        holdout = evaluate_gate(gate, validation) if validation else None
        if holdout is not None and (holdout.harmful or holdout.win_to_loss):
            continue
        if selected_mixed_exact_cells(gate, all_training):
            continue
        return CegisResult(gate, score, tuple(trace), mixed, holdout)
    return CegisResult(None, empty, tuple(trace), mixed)


def threshold_robustness(
    gate: Sequence[Sequence[ConditionGene]],
    cases: Sequence[MacroEffectCase],
    *,
    radius: int = 2,
) -> list[dict[str, Any]]:
    validate_identity_free_gate(gate)
    rows: list[dict[str, Any]] = []
    offsets = [0]
    for step in range(1, radius + 1):
        offsets.extend((-step, step))
    for ci, clause in enumerate(gate):
        for pi, condition in enumerate(clause):
            if condition.cmp not in {"GE", "LE", "EQ", "NE"}:
                continue
            for offset in offsets:
                changed = ConditionGene(condition.feature, condition.cmp, condition.index, int(condition.value) + offset)
                new_gate = [list(c) for c in gate]
                new_gate[ci][pi] = changed
                canonical = tuple(tuple(c) for c in new_gate)
                score = evaluate_gate(canonical, cases)
                rows.append(
                    {
                        "clause": ci,
                        "predicate": pi,
                        "feature": condition.feature,
                        "index": int(condition.index),
                        "cmp": condition.cmp,
                        "selected": int(condition.value),
                        "value": int(changed.value),
                        "offset": offset,
                        "safe": score.harmful == 0 and score.win_to_loss == 0,
                        "activated": score.activated,
                        "positive": score.positive,
                        "harmful": score.harmful,
                        "loss_to_win": score.loss_to_win,
                        "worst_delta": score.worst_delta,
                    }
                )
    return rows


def allocate_route_id(existing: Iterable[int], *, minimum: int = 216) -> int:
    used = {int(x) for x in existing}
    candidate = int(minimum)
    while candidate in used:
        candidate += 1
    if candidate > 65535:
        raise ValueError("no strategic route id available")
    return candidate


def make_specialist_name(candidate: CandidateEvidence, gate: Sequence[Sequence[ConditionGene]]) -> str:
    semantic = _sha(
        {
            "source_route": candidate.donor_route_id,
            "start": candidate.start,
            "duration": candidate.duration,
            "gate": _canonical_gate(gate),
        },
        b"P0.6-SPECIALIST-NAME\0",
    )
    return f"synthesized_route{candidate.donor_route_id}_s{candidate.start}_{semantic[:10]}"


def build_synthesized_macro(
    candidate: CandidateEvidence,
    gate: Sequence[Sequence[ConditionGene]],
    *,
    route_id: int,
    base_route_id: int,
    fragment_hash: str,
) -> BoundedRouteMacro:
    validate_identity_free_gate(gate)
    macro = BoundedRouteMacro(
        name=make_specialist_name(candidate, gate),
        route_id=int(route_id),
        base_route_id=int(base_route_id),
        start_step=int(candidate.start),
        duration=int(candidate.duration),
        gate_clauses=tuple(tuple(c for c in clause) for clause in gate),
        provenance={
            "source_kind": "autonomous_synthesis",
            "source_route_id": int(candidate.donor_route_id),
            "fragment_sha256": str(fragment_hash),
            "stage": "P0.6",
        },
    )
    macro.validate()
    return macro


def build_specialist_declaration(
    macro: BoundedRouteMacro,
    route_source: str,
    macro_source: str,
) -> SpecialistDeclaration:
    return SpecialistDeclaration.from_macro(
        macro,
        route_source=str(route_source),
        authorities=("strategic_route_override",),
        macro_source=str(macro_source),
        provenance={"acceptance": "P0.6_autonomous_candidate"},
    )


def extend_composition_declaration(base: Mapping[str, Any], specialist: SpecialistDeclaration) -> dict[str, Any]:
    specialist.validate()
    out = copy.deepcopy(dict(base))
    if int(out.get("schema", 0)) != 1 or out.get("kind") != COMPOSER_KIND:
        raise ValueError("unsupported base composition declaration")
    if any(int(x["route_id"]) == int(specialist.route_id) for x in out.get("specialists", ())):
        raise ValueError("route id already exists in composition")
    out.setdefault("specialists", []).append(specialist.canonical())
    out["output_name"] = f"{str(out.get('output_name') or 'bkg4')}_plus_{specialist.name}"
    out["provenance"] = {
        **dict(out.get("provenance") or {}),
        "p0_6_extension": "autonomous_synthesizer",
    }
    return out


@dataclass(frozen=True)
class PanelSpec:
    name: str
    path: str
    role: str

    def canonical(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass(frozen=True)
class SynthesisBudget:
    max_target_families: int = 3
    max_candidate_routes: int = 8
    max_coarse_candidates: int = 2049
    max_refinement_candidates: int = 257
    max_cuda_games: int = 500_000
    max_cegis_iterations: int = 16
    max_conditions: int = 4
    max_certification_cases: int = 1600

    def validate(self) -> None:
        for field in dataclasses.fields(self):
            if int(getattr(self, field.name)) <= 0:
                raise ValueError(f"budget {field.name} must be positive")
        if self.max_conditions > 4:
            raise ValueError("P0.6 compact gate grammar is limited to four conditions")


@dataclass(frozen=True)
class SynthesisConfig:
    controller: str
    routes: str
    legacy_routes: str
    composition: str
    donor_bank: str
    frontier_panel: str
    transfer_panel: str
    validation_panel: str
    sealed_panel: str
    certification_panel: str
    archive_panel: str
    output_root: str
    fragment_eval_bin: str
    state_capture_bin: str
    build_assets_script: str
    cpp_differential_runner: str
    cuda_differential_runner: str
    strategic_route_slot: int = 19
    feed_slot: int = 5
    feed_count: int = 13
    coarse_starts: tuple[int, ...] = (48, 96, 120, 144, 168, 216, 264, 336)
    coarse_durations: tuple[int, ...] = (12, 24, 48)
    refine_step_radius: int = 12
    refine_duration_radius: int = 12
    max_target_cases: int = 24
    min_transfer_positive: int = 2
    min_transfer_independent: int = 2
    threshold_radius: int = 2
    budget: SynthesisBudget = field(default_factory=SynthesisBudget)
    extra_falsification_panels: tuple[PanelSpec, ...] = ()

    def validate(self) -> None:
        self.budget.validate()
        if self.max_target_cases <= 0:
            raise ValueError("max_target_cases")
        if self.min_transfer_positive <= 0 or self.min_transfer_independent <= 0:
            raise ValueError("transfer minima")
        if self.threshold_radius < 1:
            raise ValueError("threshold radius")

    def canonical(self) -> dict[str, Any]:
        return {
            "controller": self.controller,
            "routes": self.routes,
            "legacy_routes": self.legacy_routes,
            "composition": self.composition,
            "donor_bank": self.donor_bank,
            "frontier_panel": self.frontier_panel,
            "transfer_panel": self.transfer_panel,
            "validation_panel": self.validation_panel,
            "sealed_panel": self.sealed_panel,
            "certification_panel": self.certification_panel,
            "archive_panel": self.archive_panel,
            "output_root": self.output_root,
            "fragment_eval_bin": self.fragment_eval_bin,
            "state_capture_bin": self.state_capture_bin,
            "build_assets_script": self.build_assets_script,
            "cpp_differential_runner": self.cpp_differential_runner,
            "cuda_differential_runner": self.cuda_differential_runner,
            "strategic_route_slot": self.strategic_route_slot,
            "feed_slot": self.feed_slot,
            "feed_count": self.feed_count,
            "coarse_starts": list(self.coarse_starts),
            "coarse_durations": list(self.coarse_durations),
            "refine_step_radius": self.refine_step_radius,
            "refine_duration_radius": self.refine_duration_radius,
            "max_target_cases": self.max_target_cases,
            "min_transfer_positive": self.min_transfer_positive,
            "min_transfer_independent": self.min_transfer_independent,
            "threshold_radius": self.threshold_radius,
            "budget": dataclasses.asdict(self.budget),
            "extra_falsification_panels": [x.canonical() for x in self.extra_falsification_panels],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SynthesisConfig":
        forbidden = {"final_gate", "selected_route", "selected_start", "selected_duration", "selected_threshold"}
        present = forbidden & set(value)
        if present:
            raise ValueError(f"manual strategy/gate injection is forbidden: {sorted(present)}")
        fields = {f.name for f in dataclasses.fields(cls)}
        unknown = set(value) - fields
        if unknown:
            raise ValueError(f"unknown synthesizer config fields: {sorted(unknown)}")
        payload = dict(value)
        if "budget" in payload and not isinstance(payload["budget"], SynthesisBudget):
            payload["budget"] = SynthesisBudget(**dict(payload["budget"]))
        if "extra_falsification_panels" in payload:
            payload["extra_falsification_panels"] = tuple(
                x if isinstance(x, PanelSpec) else PanelSpec(**x) for x in payload["extra_falsification_panels"]
            )
        for name in ("coarse_starts", "coarse_durations"):
            if name in payload:
                payload[name] = tuple(int(x) for x in payload[name])
        out = cls(**payload)
        out.validate()
        return out

    @classmethod
    def load(cls, path: str | Path) -> "SynthesisConfig":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    @classmethod
    def synthetic(cls) -> "SynthesisConfig":
        return cls(
            controller="scratch/controller.bkgc",
            routes="scratch/routes.bkv6",
            legacy_routes="scratch/legacy_routes.bkv6",
            composition="scratch/composition.json",
            donor_bank="scratch/donors.txt",
            frontier_panel="scratch/frontier.tsv",
            transfer_panel="scratch/transfer.tsv",
            validation_panel="scratch/validation.tsv",
            sealed_panel="scratch/sealed.tsv",
            certification_panel="scratch/cert.tsv",
            archive_panel="scratch/archive.tsv",
            output_root="scratch/p0_6_synthetic",
            fragment_eval_bin="scratch/cuda-engine-build/bkg4_fragment_eval",
            state_capture_bin="scratch/cuda-engine-build/bkg4_information_state_capture",
            build_assets_script="bonsai-kg/integrations/bkg4_champion/build_assets.py",
            cpp_differential_runner="scratch/bonsai_kg/v4_cpp_differential_runner",
            cuda_differential_runner="scratch/bonsai_kg/v4_cuda_differential_runner",
            budget=SynthesisBudget(max_candidate_routes=4),
            extra_falsification_panels=(PanelSpec("self_history", "scratch/self.tsv", "self_history"),),
        )


@dataclass(frozen=True)
class CommandRequest:
    argv: tuple[str, ...]
    probes: tuple[int, ...] = ()


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def build_capture_request(config: SynthesisConfig, cases: Path, step: int, out_dir: Path) -> CommandRequest:
    binary = _resolve(config.state_capture_bin)
    out = out_dir / f"step{int(step)}.b4is"
    truth = out_dir / f"step{int(step)}_truth_unused.b4is"
    probes = out_dir / f"probe_{int(step)}.txt"
    return CommandRequest(
        (
            str(binary),
            "--controller", str(_resolve(config.controller)),
            "--routes", str(_resolve(config.routes)),
            "--legacy-routes", str(_resolve(config.legacy_routes)),
            "--cases", str(cases),
            "--probes", str(probes),
            "--out", str(out),
            "--truth-out", str(truth),
            "--feed-slot", str(config.feed_slot),
            "--feed-count", str(config.feed_count),
            "--strategic-route-slot", str(config.strategic_route_slot),
        ),
        probes=(int(step),),
    )


def plan_falsification_panels(config: SynthesisConfig) -> list[PanelSpec]:
    rows = [
        PanelSpec("certification", config.certification_panel, "protected"),
        PanelSpec("archive", config.archive_panel, "current_archive"),
        *config.extra_falsification_panels,
        PanelSpec("sealed", config.sealed_panel, "sealed"),
    ]
    # deterministic with the sealed holdout always last
    body = sorted(rows[:-1], key=lambda x: (x.role, x.name, x.path))
    return body + [rows[-1]]


@dataclass(frozen=True)
class CertificationReport:
    accepted: bool
    failure_reason: str | None
    records: int
    positive: int
    harmful: int
    loss_to_win: int
    win_to_loss: int
    worst_delta: int
    aggregate_delta: int
    protected_negative_delta: int

    def canonical(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def assess_certification(
    cases: Sequence[CaseRecord],
    parent_margins: Sequence[int],
    candidate_margins: Sequence[int],
) -> CertificationReport:
    if not (len(cases) == len(parent_margins) == len(candidate_margins)):
        raise ValueError("certification count mismatch")
    deltas = [int(c) - int(p) for p, c in zip(parent_margins, candidate_margins)]
    protected_negative = sum(
        bool(row.protected) and parent > 0 and delta < 0
        for row, parent, delta in zip(cases, parent_margins, deltas)
    )
    w2l = sum(p > 0 > c for p, c in zip(parent_margins, candidate_margins))
    failure = PROTECTED_REGRESSION if protected_negative or w2l else None
    return CertificationReport(
        accepted=failure is None,
        failure_reason=failure,
        records=len(cases),
        positive=sum(x > 0 for x in deltas),
        harmful=sum(x < 0 for x in deltas),
        loss_to_win=sum(p < 0 < c for p, c in zip(parent_margins, candidate_margins)),
        win_to_loss=w2l,
        worst_delta=min(deltas, default=0),
        aggregate_delta=sum(deltas),
        protected_negative_delta=protected_negative,
    )


@dataclass(frozen=True)
class ParityReport:
    accepted: bool
    failure_reason: str | None
    frames: int
    mismatches: int

    def canonical(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def compare_parity_rows(cpu_rows: Sequence[str], cuda_rows: Sequence[str]) -> ParityReport:
    n = max(len(cpu_rows), len(cuda_rows))
    mismatch = abs(len(cpu_rows) - len(cuda_rows)) + sum(a != b for a, b in zip(cpu_rows, cuda_rows))
    return ParityReport(mismatch == 0, None if mismatch == 0 else CPU_CUDA_MISMATCH, n, mismatch)


@dataclass
class SynthesisResult:
    run_id: str
    verdict: str
    terminal_reason: str | None = None
    target_frontier: list[dict[str, Any]] = field(default_factory=list)
    selected_family: dict[str, Any] | None = None
    candidate_ranking: list[dict[str, Any]] = field(default_factory=list)
    rejections: list[dict[str, Any]] = field(default_factory=list)
    candidate_attempts: list[dict[str, Any]] = field(default_factory=list)
    selected_candidate: dict[str, Any] | None = None
    transfer: dict[str, Any] | None = None
    cegis: dict[str, Any] | None = None
    final_gate: list[list[dict[str, Any]]] | None = None
    threshold_sweep: list[dict[str, Any]] = field(default_factory=list)
    macro: dict[str, Any] | None = None
    specialist: dict[str, Any] | None = None
    composition: dict[str, Any] | None = None
    target_capability: dict[str, Any] | None = None
    falsification: dict[str, Any] | None = None
    certification: dict[str, Any] | None = None
    parity: dict[str, Any] | None = None
    final_identity: dict[str, Any] | None = None
    runtime: dict[str, Any] = field(default_factory=dict)
    autonomy_proof: bool = True
    artifacts: dict[str, str] = field(default_factory=dict)

    def canonical(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


class SyntheticBackend:
    """Deterministic integration fixture proving retry/budget semantics.

    The real backend below owns CUDA/process orchestration. This fixture intentionally
    contains no strategy fields and therefore cannot inject a route or gate.
    """

    def __init__(self, *, candidate_outcomes: Sequence[Mapping[str, Any]]):
        self.outcomes = [dict(x) for x in candidate_outcomes]
        self.candidates_attempted = 0

    def try_candidate(self, index: int, config: SynthesisConfig) -> Mapping[str, Any]:
        self.candidates_attempted += 1
        if index >= len(self.outcomes):
            return {"failure": NON_TRANSFERABLE}
        return self.outcomes[index]


def synthesize_next_specialist(config: SynthesisConfig, *, backend: Any | None = None) -> SynthesisResult:
    config.validate()
    run_id = _sha(config.canonical(), b"P0.6-RUN\0")[:24]
    if backend is None:
        backend = CudaSynthesisBackend()
    if hasattr(backend, "run") and not isinstance(backend, SyntheticBackend):
        return backend.run(config, run_id=run_id)
    result = SynthesisResult(run_id=run_id, verdict=REJECTED, terminal_reason=SEARCH_BUDGET_EXHAUSTED)
    max_attempts = min(config.budget.max_candidate_routes, len(getattr(backend, "outcomes", ())))
    for index in range(max_attempts):
        outcome = dict(backend.try_candidate(index, config))
        failure = outcome.get("failure")
        if failure:
            rejection = {"candidate_index": index, "reason": str(failure)}
            result.rejections.append(rejection)
            result.candidate_attempts.append({"candidate_index": index, "status": REJECTED, "rejection": dict(rejection)})
            continue
        if outcome.get("accepted"):
            result.verdict = ACCEPTED
            result.terminal_reason = None
            result.selected_candidate = {"candidate_index": index}
            result.candidate_attempts.append({"candidate_index": index, "status": ACCEPTED, "rejection": None})
            return result
        rejection = {"candidate_index": index, "reason": NO_COUNTERFACTUAL_GAIN}
        result.rejections.append(rejection)
        result.candidate_attempts.append({"candidate_index": index, "status": REJECTED, "rejection": dict(rejection)})
    return result


# ---------------------------- real CUDA backend ----------------------------


@dataclass
class _RunLedger:
    cuda_games: int = 0
    cuda_ms: float = 0.0
    host_started: float = field(default_factory=time.perf_counter)
    command_log: list[dict[str, Any]] = field(default_factory=list)

    def record(self, argv: Sequence[str], stdout: str, elapsed_s: float) -> None:
        self.command_log.append({"argv": list(argv), "stdout": stdout.strip(), "elapsed_s": round(elapsed_s, 6)})
        match = re.search(r"games=(\d+).*gpu_ms=([0-9.]+)", stdout)
        if match:
            self.cuda_games += int(match.group(1))
            self.cuda_ms += float(match.group(2))


def _run(argv: Sequence[str], ledger: _RunLedger, *, cwd: Path = ROOT) -> str:
    started = time.perf_counter()
    proc = subprocess.run(list(argv), cwd=cwd, text=True, capture_output=True)
    elapsed = time.perf_counter() - started
    ledger.record(argv, proc.stdout, elapsed)
    if proc.returncode != 0:
        raise RuntimeError(f"command failed ({proc.returncode}): {' '.join(argv)}\n{proc.stderr.strip()}")
    return proc.stdout


def _read_donors(path: Path) -> list[str]:
    rows = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip() and not line.startswith("#")]
    if not rows:
        raise ValueError("empty donor bank")
    return rows


def _write_cases(path: Path, cases: Sequence[CaseRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(x.tsv() for x in cases) + "\n", encoding="utf-8")


def _fragment_command(
    config: SynthesisConfig,
    *,
    controller: Path,
    routes: Path,
    legacy_routes: Path,
    cases: Path,
    donors: Path,
    out: Path,
    grid: Path,
    starts: Sequence[int],
    durations: Sequence[int],
) -> tuple[str, ...]:
    return (
        str(_resolve(config.fragment_eval_bin)),
        "--controller", str(controller),
        "--routes", str(routes),
        "--legacy-routes", str(legacy_routes),
        "--cases", str(cases),
        "--donors", str(donors),
        "--out", str(out),
        "--grid-out", str(grid),
        "--starts", ",".join(map(str, sorted(set(starts)))),
        "--durations", ",".join(map(str, sorted(set(durations)))),
        "--feed-slot", str(config.feed_slot),
        "--feed-count", str(config.feed_count),
        "--strategic-route-slot", str(config.strategic_route_slot),
    )


def _candidate_evidence_from_result(
    results: Path,
    grid: Path,
    cases: Sequence[CaseRecord],
) -> list[CandidateEvidence]:
    meta, rows = load_fragment_results(results)
    grid_rows = load_grid(grid)
    if meta["start_index"] != 0 or len(rows) != meta["candidates"]:
        raise ValueError("P0.6 currently requires one complete fragment result chunk")
    baseline = tuple(_margin(x) for x in rows[0])
    if any(x[3] for x in rows[0]):
        raise RuntimeError(CUDA_FAULT)
    out: list[CandidateEvidence] = []
    for cid in range(1, len(rows)):
        stat = grid_rows[cid]
        candidate_rows = rows[cid]
        if any(x[3] for x in candidate_rows):
            continue
        margins = tuple(_margin(x) for x in candidate_rows)
        deltas = tuple(m - b for m, b in zip(margins, baseline))
        donor_path = str(stat["path"])
        out.append(
            CandidateEvidence(
                cid,
                _route_id_from_path(donor_path),
                donor_path,
                int(stat["start"]),
                int(stat["duration"]),
                baseline,
                deltas,
                tuple(int(x[2]) for x in candidate_rows),
            )
        )
    return rank_candidates(out, cases)


def _capture_contexts(capture: R0Capture) -> tuple[list[dict[tuple[str, int], int]], list[tuple[int, ...]], list[int]]:
    registry = registry_manifest()
    contexts: list[dict[tuple[str, int], int]] = []
    signatures: list[tuple[int, ...]] = []
    base_routes: list[int] = []
    for ci in range(capture.cases):
        frame = capture.frame(ci, 0)
        # The compact CEGIS grammar carries one feature index. Exclude registry rows
        # with two arguments (for example per-unit-per-item inventory) rather than
        # collapsing them onto an ambiguous key. They remain present in the exact
        # legal-state signature used for hindsight/mixed-cell rejection.
        contexts.append({
            (name, args[0] if args else 0): int(frame[index])
            for index, (name, args) in enumerate(registry)
            if len(args) <= 1 and name in _P0_6_GATE_FEATURES
        })
        signatures.append(tuple(int(x) for x in frame[: capture.registry_width + capture.bb_width + capture.compat_width]))
        base_routes.append(int(frame[capture.registry_width + 1]))
    return contexts, signatures, base_routes


def _macro_effect_cases(
    label: str,
    cases: Sequence[CaseRecord],
    evidence: CandidateEvidence,
    capture: R0Capture,
) -> tuple[list[MacroEffectCase], list[int]]:
    contexts, signatures, base_routes = _capture_contexts(capture)
    if len(cases) != len(evidence.deltas) or capture.cases != len(cases):
        raise ValueError("effect/capture count mismatch")
    rows = []
    for i, case in enumerate(cases):
        rows.append(
            MacroEffectCase(
                f"{label}:{case.case_id}",
                int(evidence.baseline_margins[i]),
                int(evidence.baseline_margins[i] + evidence.deltas[i]),
                contexts[i],
                signatures[i],
                support_group=f"{case.family}:{Path(case.path).name}",
            )
        )
    return rows, base_routes


def _modal_base_route(base_routes: Sequence[int], effect_rows: Sequence[MacroEffectCase], gate: Sequence[Sequence[ConditionGene]]) -> int:
    active = [route for route, row in zip(base_routes, effect_rows) if gate_hits(gate, row.context) and row.delta > 0]
    if not active:
        active = list(base_routes)
    counts = Counter(active)
    return sorted(counts.items(), key=lambda x: (-x[1], x[0]))[0][0]


def _read_actions(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    actions = value.get("actions") if isinstance(value, dict) else None
    if not isinstance(actions, list) or len(actions) != 719:
        raise ValueError(f"route source must contain 719 actions: {path}")
    return [dict(x) for x in actions]


def _find_route_source(donors: Sequence[str], route_id: int) -> Path:
    matches = [Path(x) for x in donors if _route_id_from_path(x) == int(route_id)]
    if not matches:
        raise ValueError(f"base route {route_id} unavailable in coherent route bank")
    return matches[0]


def _import_build_assets(path: Path):
    spec = importlib.util.spec_from_file_location("p0_6_build_assets", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import BKG4 build_assets")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _result_baseline(path: Path) -> tuple[list[tuple[int, int, int, int]], dict[str, Any]]:
    meta, rows = load_fragment_results(path)
    if not rows or meta["start_index"] != 0:
        raise ValueError("baseline result missing candidate 0")
    return list(rows[0]), meta


class CudaSynthesisBackend:
    def run(self, config: SynthesisConfig, *, run_id: str) -> SynthesisResult:
        config.validate()
        ledger = _RunLedger()
        out_root = _resolve(config.output_root) / run_id
        out_root.mkdir(parents=True, exist_ok=True)
        result = SynthesisResult(run_id=run_id, verdict=REJECTED, terminal_reason=NO_SAFE_SPECIALIST_FOUND)
        result.artifacts["run_root"] = str(out_root.relative_to(ROOT) if out_root.is_relative_to(ROOT) else out_root)
        try:
            controller = _resolve(config.controller)
            routes = _resolve(config.routes)
            legacy = _resolve(config.legacy_routes)
            composition_path = _resolve(config.composition)
            donors_path = _resolve(config.donor_bank)
            donors = _read_donors(donors_path)[: config.budget.max_candidate_routes * 8]
            if not all(p.exists() for p in (controller, routes, legacy, composition_path, donors_path)):
                raise FileNotFoundError("one or more P0.6 configured inputs do not exist")

            # 1) exact current frontier baseline on CUDA. One donor/start/duration is enough
            # to obtain candidate-0 baseline; search itself occurs after target selection.
            frontier_cases = load_case_bank(_resolve(config.frontier_panel))
            baseline_donors = out_root / "baseline_donor.txt"
            baseline_donors.write_text(donors[0] + "\n", encoding="utf-8")
            baseline_out = out_root / "frontier_baseline.b4fr"
            baseline_grid = out_root / "frontier_baseline_grid.tsv"
            _run(
                _fragment_command(
                    config,
                    controller=controller,
                    routes=routes,
                    legacy_routes=legacy,
                    cases=_resolve(config.frontier_panel),
                    donors=baseline_donors,
                    out=baseline_out,
                    grid=baseline_grid,
                    starts=(config.coarse_starts[0],),
                    durations=(config.coarse_durations[0],),
                ),
                ledger,
            )
            base_rows, _ = _result_baseline(baseline_out)
            baseline_margins = [_margin(x) for x in base_rows]
            frontier = rank_loss_frontier(frontier_cases, baseline_margins, max_families=config.budget.max_target_families)
            result.target_frontier = [x.canonical() for x in frontier]
            if not frontier:
                result.verdict = REJECTED
                result.terminal_reason = NO_SAFE_SPECIALIST_FOUND
                return self._finish(result, ledger, out_root)

            composition_base = json.loads(composition_path.read_text(encoding="utf-8"))
            existing_routes = {int(x["route_id"]) for x in composition_base.get("specialists", ())}
            route_id = allocate_route_id(existing_routes)
            attempts = 0
            global_candidate_rank: list[dict[str, Any]] = []

            for family_rank, family in enumerate(frontier):
                target = build_target_cases(family, frontier_cases, baseline_margins, max_cases=config.max_target_cases)
                target_path = out_root / f"family_{family_rank:02d}_target.tsv"
                _write_cases(target_path, target.cases)
                result.selected_family = target.canonical()

                # 2) hierarchical CUDA search: coarse full coherent bank.
                donor_subset = donors[: max(1, min(len(donors), 64))]
                expected = 1 + len(donor_subset) * len(set(config.coarse_starts)) * len(set(config.coarse_durations))
                if expected > config.budget.max_coarse_candidates:
                    max_donors = max(1, (config.budget.max_coarse_candidates - 1) // (len(set(config.coarse_starts)) * len(set(config.coarse_durations))))
                    donor_subset = donor_subset[:max_donors]
                coarse_donors = out_root / f"family_{family_rank:02d}_coarse_donors.txt"
                coarse_donors.write_text("\n".join(donor_subset) + "\n", encoding="utf-8")
                coarse_out = out_root / f"family_{family_rank:02d}_coarse.b4fr"
                coarse_grid = out_root / f"family_{family_rank:02d}_coarse_grid.tsv"
                coarse_candidates = 1 + len(donor_subset) * len(set(config.coarse_starts)) * len(set(config.coarse_durations))
                projected_games = coarse_candidates * len(target.cases)
                if ledger.cuda_games + projected_games > config.budget.max_cuda_games:
                    result.rejections.append({"family": family.family, "reason": SEARCH_BUDGET_EXHAUSTED, "stage": "coarse_search"})
                    break
                _run(
                    _fragment_command(
                        config,
                        controller=controller,
                        routes=routes,
                        legacy_routes=legacy,
                        cases=target_path,
                        donors=coarse_donors,
                        out=coarse_out,
                        grid=coarse_grid,
                        starts=config.coarse_starts,
                        durations=config.coarse_durations,
                    ),
                    ledger,
                )
                coarse_ranked = _candidate_evidence_from_result(coarse_out, coarse_grid, target.cases)
                if not coarse_ranked or coarse_ranked[0].positive == 0:
                    result.rejections.append({"family": family.family, "reason": NO_COUNTERFACTUAL_GAIN, "stage": "coarse_search"})
                    continue

                # 3) refine top basins with only their donor routes and nearby starts/durations.
                top_basins: list[CandidateEvidence] = []
                seen_donor: set[int] = set()
                for candidate in coarse_ranked:
                    if candidate.donor_route_id not in seen_donor:
                        top_basins.append(candidate)
                        seen_donor.add(candidate.donor_route_id)
                    if len(top_basins) >= min(4, config.budget.max_candidate_routes):
                        break
                refine_donors_list = [x.donor_path for x in top_basins]
                refine_starts = sorted({
                    max(0, min(718, c.start + offset))
                    for c in top_basins
                    for offset in (-config.refine_step_radius, -config.refine_step_radius // 2, 0, config.refine_step_radius // 2, config.refine_step_radius)
                })
                refine_durations = sorted({
                    max(1, min(719 - c.start, c.duration + offset))
                    for c in top_basins
                    for offset in (-config.refine_duration_radius, 0, config.refine_duration_radius)
                    if c.duration + offset > 0
                })
                while 1 + len(refine_donors_list) * len(refine_starts) * len(refine_durations) > config.budget.max_refinement_candidates:
                    if len(refine_starts) > 3:
                        refine_starts = refine_starts[::2]
                    elif len(refine_durations) > 2:
                        refine_durations = refine_durations[:2]
                    elif len(refine_donors_list) > 1:
                        refine_donors_list = refine_donors_list[:-1]
                    else:
                        break
                refine_donors = out_root / f"family_{family_rank:02d}_refine_donors.txt"
                refine_donors.write_text("\n".join(refine_donors_list) + "\n", encoding="utf-8")
                refine_out = out_root / f"family_{family_rank:02d}_refine.b4fr"
                refine_grid = out_root / f"family_{family_rank:02d}_refine_grid.tsv"
                refine_candidates_count = 1 + len(refine_donors_list) * len(refine_starts) * len(refine_durations)
                projected_games = refine_candidates_count * len(target.cases)
                if ledger.cuda_games + projected_games > config.budget.max_cuda_games:
                    result.rejections.append({"family": family.family, "reason": SEARCH_BUDGET_EXHAUSTED, "stage": "refine_search"})
                    break
                _run(
                    _fragment_command(
                        config,
                        controller=controller,
                        routes=routes,
                        legacy_routes=legacy,
                        cases=target_path,
                        donors=refine_donors,
                        out=refine_out,
                        grid=refine_grid,
                        starts=refine_starts,
                        durations=refine_durations,
                    ),
                    ledger,
                )
                ranked = _candidate_evidence_from_result(refine_out, refine_grid, target.cases)
                global_candidate_rank.extend([x.canonical() for x in ranked[:10]])
                result.candidate_ranking = global_candidate_rank[:30]

                for candidate in ranked[: config.budget.max_candidate_routes]:
                    if attempts >= config.budget.max_candidate_routes:
                        break
                    attempts += 1
                    rejection = self._attempt_candidate(
                        config,
                        candidate,
                        target,
                        controller,
                        routes,
                        legacy,
                        composition_base,
                        donors,
                        route_id,
                        out_root / f"candidate_{attempts:02d}",
                        ledger,
                        result,
                    )
                    if rejection is None:
                        result.candidate_attempts.append(self._attempt_snapshot(candidate, result, None))
                        return self._finish(result, ledger, out_root)
                    rejection_record = {
                        "candidate_id": candidate.candidate_id,
                        "source_route": candidate.donor_route_id,
                        "start": candidate.start,
                        "duration": candidate.duration,
                        **rejection,
                    }
                    result.rejections.append(rejection_record)
                    result.candidate_attempts.append(self._attempt_snapshot(candidate, result, rejection_record))
                if attempts >= config.budget.max_candidate_routes:
                    break

            result.verdict = REJECTED
            if attempts >= config.budget.max_candidate_routes or any(row.get("reason") == SEARCH_BUDGET_EXHAUSTED for row in result.rejections):
                result.terminal_reason = SEARCH_BUDGET_EXHAUSTED
            else:
                result.terminal_reason = NO_SAFE_SPECIALIST_FOUND
            return self._finish(result, ledger, out_root)
        except Exception as exc:
            result.verdict = INFRASTRUCTURE_FAILURE
            result.terminal_reason = INFRASTRUCTURE_FAILURE
            result.rejections.append({"reason": INFRASTRUCTURE_FAILURE, "detail": f"{type(exc).__name__}: {exc}"})
            return self._finish(result, ledger, out_root)

    @staticmethod
    def _attempt_snapshot(
        candidate: CandidateEvidence,
        result: SynthesisResult,
        rejection: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        return {
            "candidate": candidate.canonical(),
            "status": ACCEPTED if rejection is None else REJECTED,
            "rejection": None if rejection is None else copy.deepcopy(dict(rejection)),
            "transfer": copy.deepcopy(result.transfer),
            "cegis": copy.deepcopy(result.cegis),
            "final_gate": copy.deepcopy(result.final_gate),
            "threshold_sweep": copy.deepcopy(result.threshold_sweep),
            "macro": copy.deepcopy(result.macro),
            "specialist": copy.deepcopy(result.specialist),
            "composition": copy.deepcopy(result.composition),
            "target_capability": copy.deepcopy(result.target_capability),
            "falsification": copy.deepcopy(result.falsification),
            "certification": copy.deepcopy(result.certification),
            "parity": copy.deepcopy(result.parity),
        }

    def _attempt_candidate(
        self,
        config: SynthesisConfig,
        candidate: CandidateEvidence,
        target: TargetSelection,
        controller: Path,
        routes: Path,
        legacy: Path,
        composition_base: Mapping[str, Any],
        donors: Sequence[str],
        route_id: int,
        out_dir: Path,
        ledger: _RunLedger,
        result: SynthesisResult,
    ) -> dict[str, Any] | None:
        out_dir.mkdir(parents=True, exist_ok=True)
        # Candidate-scoped evidence must never leak from a previous retry.
        result.transfer = None
        result.cegis = None
        result.final_gate = None
        result.threshold_sweep = []
        result.macro = None
        result.specialist = None
        result.composition = None
        result.target_capability = None
        result.falsification = None
        result.certification = None
        result.parity = None
        result.final_identity = None
        source = Path(candidate.donor_path)
        exact_donors = out_dir / "donor.txt"
        exact_donors.write_text(str(source) + "\n", encoding="utf-8")

        def exact_panel(panel_path: str, label: str) -> tuple[CandidateEvidence, list[CaseRecord], Path, R0Capture, list[MacroEffectCase], list[int]]:
            cases_path = _resolve(panel_path)
            cases = load_case_bank(cases_path)
            if ledger.cuda_games + 2 * len(cases) > config.budget.max_cuda_games:
                raise RuntimeError(SEARCH_BUDGET_EXHAUSTED)
            out = out_dir / f"{label}.b4fr"
            grid = out_dir / f"{label}_grid.tsv"
            _run(
                _fragment_command(
                    config,
                    controller=controller,
                    routes=routes,
                    legacy_routes=legacy,
                    cases=cases_path,
                    donors=exact_donors,
                    out=out,
                    grid=grid,
                    starts=(candidate.start,),
                    durations=(candidate.duration,),
                ), ledger,
            )
            ranked = _candidate_evidence_from_result(out, grid, cases)
            if len(ranked) != 1:
                raise RuntimeError("exact panel did not produce one donor candidate")
            evidence = ranked[0]
            capture_dir = out_dir / f"{label}_capture"
            capture_dir.mkdir(parents=True, exist_ok=True)
            req = build_capture_request(config, cases_path, candidate.start, capture_dir)
            probe_path = Path(req.argv[req.argv.index("--probes") + 1])
            probe_path.write_text(f"{candidate.start}\n", encoding="utf-8")
            _run(req.argv, ledger)
            capture_path = Path(req.argv[req.argv.index("--out") + 1])
            capture = load_r0(capture_path)
            effect, base_routes = _macro_effect_cases(label, cases, evidence, capture)
            return evidence, cases, out, capture, effect, base_routes

        transfer_e, transfer_cases, _, _, transfer_rows, transfer_routes = exact_panel(config.transfer_panel, "transfer")
        transfer_ranked = _candidate_with_support(transfer_e, transfer_cases)
        result.transfer = transfer_ranked.canonical()
        if not transfer_ranked.transferable(min_positive=config.min_transfer_positive, min_independent=config.min_transfer_independent):
            return {"reason": NON_TRANSFERABLE, "stage": "transfer"}

        validation_e, validation_cases, _, _, validation_rows, validation_routes = exact_panel(config.validation_panel, "validation")
        sealed_e, sealed_cases, _, _, sealed_rows, sealed_routes = exact_panel(config.sealed_panel, "sealed_effect")

        positives = [x for x in transfer_rows if x.delta > 0]
        harmful = [x for x in transfer_rows if x.delta < 0]
        mixed = detect_mixed_legal_states(transfer_rows + validation_rows)
        if positives and all(any(x.case_id in cell for cell in mixed) for x in positives):
            return {"reason": MIXED_LEGAL_STATE, "stage": "legal_state", "mixed_cells": [list(x) for x in mixed]}

        cegis = run_cegis(
            positives,
            harmful,
            validation_rows,
            max_conditions=config.budget.max_conditions,
            max_iterations=config.budget.max_cegis_iterations,
        )
        result.cegis = cegis.canonical()
        if cegis.final_gate is None:
            return {"reason": NO_SAFE_GATE, "stage": "cegis", "mixed_cells": [list(x) for x in mixed]}
        gate = cegis.final_gate
        holdout = evaluate_gate(gate, sealed_rows)
        if holdout.harmful or holdout.win_to_loss:
            return {"reason": HOLDOUT_FAILURE, "stage": "sealed_holdout", "score": dataclasses.asdict(holdout)}
        if holdout.positive == 0 and cegis.final_score.positive <= 1:
            return {"reason": NON_TRANSFERABLE, "stage": "sealed_holdout", "detail": "no recurring positive holdout support"}

        threshold = threshold_robustness(gate, transfer_rows + validation_rows, radius=config.threshold_radius)
        result.threshold_sweep = threshold
        selected_rows = [x for x in threshold if x["offset"] == 0]
        if any(not x["safe"] for x in selected_rows):
            return {"reason": THRESHOLD_BRITTLE, "stage": "threshold"}
        unsafe_neighbors = [x for x in threshold if abs(x["offset"]) == 1 and not x["safe"]]
        if unsafe_neighbors and len(unsafe_neighbors) == len([x for x in threshold if abs(x["offset"]) == 1]):
            return {"reason": THRESHOLD_BRITTLE, "stage": "threshold", "detail": "all immediate neighbors unsafe"}

        base_route = _modal_base_route(
            transfer_routes + validation_routes + sealed_routes,
            transfer_rows + validation_rows + sealed_rows,
            gate,
        )
        base_source = _find_route_source(donors, base_route)
        donor_actions = _read_actions(source)
        base_actions = _read_actions(base_source)
        fragment_blob = json.dumps(donor_actions[candidate.start:candidate.start + candidate.duration], sort_keys=True, separators=(",", ":")).encode("utf-8")
        fragment_hash = hashlib.sha256(fragment_blob).hexdigest()
        macro = build_synthesized_macro(candidate, gate, route_id=route_id, base_route_id=base_route, fragment_hash=fragment_hash)
        result.final_gate = _canonical_gate(gate)
        result.macro = macro.canonical()

        macro_path = out_dir / "macro.json"
        route_path = out_dir / "route.json"
        macro_path.write_text(json.dumps(macro.canonical(), sort_keys=True, indent=2) + "\n", encoding="utf-8")
        route_path.write_text(json.dumps(macro.route_document(base_actions, donor_actions), sort_keys=True, indent=2) + "\n", encoding="utf-8")
        rel_macro = str(macro_path.relative_to(ROOT))
        rel_route = str(route_path.relative_to(ROOT))
        specialist = build_specialist_declaration(macro, rel_route, rel_macro)
        result.specialist = specialist.canonical()
        candidate_composition = extend_composition_declaration(composition_base, specialist)
        composition_path = out_dir / "composition.json"
        composition_path.write_text(json.dumps(candidate_composition, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        try:
            composition = compose_declaration(candidate_composition, repo_root=ROOT)
        except ValueError as exc:
            return {"reason": COMPOSITION_CONFLICT, "stage": "composition", "detail": str(exc)}
        result.composition = {
            "semantic_id": composition.semantic_id,
            "report": composition.report,
            "elapsed_ms": composition.elapsed_ms,
        }
        module_path = out_dir / "module.json"
        module_path.write_text(json.dumps(composition.module, sort_keys=True, indent=2) + "\n", encoding="utf-8")

        assets_dir = out_dir / "assets"
        build_assets = _import_build_assets(_resolve(config.build_assets_script))
        identity = build_assets.build(assets_dir, module_path)
        candidate_controller = assets_dir / "controller.bkgc"
        candidate_routes = assets_dir / "routes.bkv6"
        candidate_legacy = assets_dir / "legacy_routes.bkv6"
        result.final_identity = identity

        # Capability preservation is a separate acceptance axis from broad safety.
        # Re-run the final compiled controller on the exact discovery target cases;
        # a safe gate that no longer reaches the capability that justified synthesis
        # is not an accepted specialist.
        target_capability_path = out_dir / "target_capability.tsv"
        _write_cases(target_capability_path, target.cases)
        if ledger.cuda_games + 2 * len(target.cases) > config.budget.max_cuda_games:
            return {"reason": SEARCH_BUDGET_EXHAUSTED, "stage": "target_capability"}
        target_candidate_out = out_dir / "target_capability_candidate.b4fr"
        target_candidate_grid = out_dir / "target_capability_candidate_grid.tsv"
        _run(
            _fragment_command(
                config,
                controller=candidate_controller,
                routes=candidate_routes,
                legacy_routes=candidate_legacy,
                cases=target_capability_path,
                donors=exact_donors,
                out=target_candidate_out,
                grid=target_candidate_grid,
                starts=(candidate.start,),
                durations=(candidate.duration,),
            ),
            ledger,
        )
        _, target_candidate_rows = load_fragment_results(target_candidate_out)
        if not target_candidate_rows or any(row[3] for row in target_candidate_rows[0]):
            return {"reason": CUDA_FAULT, "stage": "target_capability"}
        target_capture_dir = out_dir / "target_capability_capture"
        target_capture_dir.mkdir(parents=True, exist_ok=True)
        target_req = build_capture_request(config, target_capability_path, candidate.start, target_capture_dir)
        Path(target_req.argv[target_req.argv.index("--probes") + 1]).write_text(f"{candidate.start}\n", encoding="utf-8")
        _run(target_req.argv, ledger)
        target_cap = load_r0(Path(target_req.argv[target_req.argv.index("--out") + 1]))
        target_contexts, _, _ = _capture_contexts(target_cap)
        target_active = [gate_hits(gate, context) for context in target_contexts]
        target_final_margins = [_margin(row) for row in target_candidate_rows[0]]
        target_report = assess_target_capability(target.cases, candidate, target_final_margins, target_active)
        result.target_capability = target_report.canonical()
        if not target_report.accepted:
            return {"reason": TARGET_CAPABILITY_LOST, "stage": "target_capability", "report": target_report.canonical()}

        # Final exact certification: compare the composed candidate baseline to the
        # current parent and to the forced donor teacher on gate activations.
        panels = plan_falsification_panels(config)
        aggregate_parent: list[int] = []
        aggregate_candidate: list[int] = []
        aggregate_cases: list[CaseRecord] = []
        panel_reports: dict[str, Any] = {}
        teacher_mismatch = 0
        inactive_mismatch = 0
        total_activations = 0
        for panel in panels:
            cases_path = _resolve(panel.path)
            if not cases_path.exists():
                continue
            panel_cases = load_case_bank(cases_path)
            if len(aggregate_cases) + len(panel_cases) > config.budget.max_certification_cases:
                continue
            # Parent + forced teacher.
            parent_out = out_dir / f"cert_{panel.name}_parent.b4fr"
            parent_grid = out_dir / f"cert_{panel.name}_parent_grid.tsv"
            _run(_fragment_command(config, controller=controller, routes=routes, legacy_routes=legacy, cases=cases_path, donors=exact_donors, out=parent_out, grid=parent_grid, starts=(candidate.start,), durations=(candidate.duration,)), ledger)
            parent_meta, parent_rows = load_fragment_results(parent_out)
            if any(r[3] for row in parent_rows for r in row):
                return {"reason": CUDA_FAULT, "stage": "certification"}
            # Candidate + a harmless donor entry only to obtain candidate-0 baseline.
            cand_out = out_dir / f"cert_{panel.name}_candidate.b4fr"
            cand_grid = out_dir / f"cert_{panel.name}_candidate_grid.tsv"
            _run(_fragment_command(config, controller=candidate_controller, routes=candidate_routes, legacy_routes=candidate_legacy, cases=cases_path, donors=exact_donors, out=cand_out, grid=cand_grid, starts=(candidate.start,), durations=(candidate.duration,)), ledger)
            _, cand_rows = load_fragment_results(cand_out)
            if any(r[3] for r in cand_rows[0]):
                return {"reason": CUDA_FAULT, "stage": "certification"}

            capture_dir = out_dir / f"cert_{panel.name}_capture"
            capture_dir.mkdir(parents=True, exist_ok=True)
            req = build_capture_request(config, cases_path, candidate.start, capture_dir)
            Path(req.argv[req.argv.index("--probes") + 1]).write_text(f"{candidate.start}\n", encoding="utf-8")
            _run(req.argv, ledger)
            cap = load_r0(Path(req.argv[req.argv.index("--out") + 1]))
            contexts, _, _ = _capture_contexts(cap)
            active = [gate_hits(gate, c) for c in contexts]
            total_activations += sum(active)
            p = parent_rows[0]
            teacher = parent_rows[1]
            c = cand_rows[0]
            for flag, pr, tr, cr in zip(active, p, teacher, c):
                pk = (pr[0], pr[1], pr[3])
                tk = (tr[0], tr[1], tr[3])
                ck = (cr[0], cr[1], cr[3])
                if flag:
                    teacher_mismatch += ck != tk
                else:
                    inactive_mismatch += ck != pk
            pm = [_margin(x) for x in p]
            cm = [_margin(x) for x in c]
            cert = assess_certification(panel_cases, pm, cm)
            panel_reports[panel.name] = {
                **cert.canonical(),
                "role": panel.role,
                "activations": sum(active),
            }
            aggregate_cases.extend(panel_cases)
            aggregate_parent.extend(pm)
            aggregate_candidate.extend(cm)

        full_cert = assess_certification(aggregate_cases, aggregate_parent, aggregate_candidate)
        result.falsification = {
            "panels": panel_reports,
            "activations": total_activations,
            "inactive_parent_mismatch": inactive_mismatch,
            "active_teacher_mismatch": teacher_mismatch,
        }
        result.certification = full_cert.canonical()
        if not full_cert.accepted:
            return {"reason": full_cert.failure_reason or PROTECTED_REGRESSION, "stage": "certification"}
        if inactive_mismatch or teacher_mismatch:
            return {"reason": PROTECTED_REGRESSION, "stage": "teacher_equivalence", "inactive_mismatch": inactive_mismatch, "teacher_mismatch": teacher_mismatch}

        parity = self._parity(config, macro, candidate_controller, transfer_rows + validation_rows + sealed_rows, out_dir, ledger)
        result.parity = parity.canonical()
        if not parity.accepted:
            return {"reason": CPU_CUDA_MISMATCH, "stage": "parity", "mismatches": parity.mismatches}

        result.selected_candidate = candidate.canonical()
        result.verdict = ACCEPTED
        result.terminal_reason = None
        result.artifacts.update({
            "macro": rel_macro,
            "route": rel_route,
            "composition": str(composition_path.relative_to(ROOT)),
            "module": str(module_path.relative_to(ROOT)),
            "assets": str(assets_dir.relative_to(ROOT)),
            "target_capability": str(target_candidate_out.relative_to(ROOT)),
        })
        return None

    def _parity(
        self,
        config: SynthesisConfig,
        macro: BoundedRouteMacro,
        controller: Path,
        effect_rows: Sequence[MacroEffectCase],
        out_dir: Path,
        ledger: _RunLedger,
    ) -> ParityReport:
        active = next((row for row in effect_rows if gate_hits(macro.gate_clauses, row.context)), None)
        inactive = next((row for row in effect_rows if not gate_hits(macro.gate_clauses, row.context)), None)
        samples = [x for x in (active, inactive) if x is not None]
        if not samples:
            return ParityReport(False, CPU_CUDA_MISMATCH, 0, 1)
        feature_ids = _feature_id_map()
        sequences = []
        for si, sample in enumerate(samples):
            seq = out_dir / f"parity_{si}.bks"
            frames = []
            for step in range(max(0, macro.start_step - 1), min(719, macro.end_step + 2)):
                features = [(feature_ids["STEP"], 0, 0, step)]
                for clause in macro.gate_clauses:
                    for condition in clause:
                        key = (condition.feature, int(condition.index))
                        if key not in sample.context:
                            continue
                        fid = feature_ids.get(condition.feature)
                        if fid is None:
                            raise ValueError(f"no differential feature id for {condition.feature}")
                        features.append((fid, int(condition.index), 0, int(sample.context[key])))
                frames.append(features)
            with seq.open("w", encoding="utf-8") as handle:
                handle.write(f"BKS2 2 {len(frames)}\n")
                for frame in frames:
                    handle.write(f"{len(frame)}\n")
                    for fid, a0, a1, value in frame:
                        handle.write(f"{fid} {a0} {a1} {value}\n")
            sequences.append(seq)
        manifest = out_dir / "parity_manifest.txt"
        manifest.write_text("".join(f"{controller} {seq}\n" for seq in sequences), encoding="utf-8")
        cpu = subprocess.run([str(_resolve(config.cpp_differential_runner)), str(manifest)], cwd=ROOT, text=True, capture_output=True)
        cuda = subprocess.run([str(_resolve(config.cuda_differential_runner)), str(manifest)], cwd=ROOT, text=True, capture_output=True)
        ledger.command_log.append({"argv": [str(_resolve(config.cpp_differential_runner)), str(manifest)], "stdout": cpu.stdout.strip(), "elapsed_s": 0})
        ledger.command_log.append({"argv": [str(_resolve(config.cuda_differential_runner)), str(manifest)], "stdout": cuda.stdout.strip(), "elapsed_s": 0})
        if cpu.returncode != 0 or cuda.returncode != 0:
            return ParityReport(False, CPU_CUDA_MISMATCH, 0, 1)
        return compare_parity_rows(cpu.stdout.splitlines(), cuda.stdout.splitlines())

    def _finish(self, result: SynthesisResult, ledger: _RunLedger, out_root: Path) -> SynthesisResult:
        result.runtime = {
            "host_seconds": round(time.perf_counter() - ledger.host_started, 6),
            "cuda_games": ledger.cuda_games,
            "cuda_ms": round(ledger.cuda_ms, 3),
            "commands": ledger.command_log,
        }
        artifact = out_root / "synthesis_run.json"
        result.artifacts["run_report"] = str(artifact.relative_to(ROOT) if artifact.is_relative_to(ROOT) else artifact)
        artifact.write_text(json.dumps(result.canonical(), sort_keys=True, indent=2) + "\n", encoding="utf-8")
        return result


def _feature_id_map() -> dict[str, int]:
    # IDs are the frozen BKG4 feature-registry v1 identifiers used by BKS2 differential
    # runners. Load them from the Python compiler when possible; retain exact v1 scalar
    # fallback values for the common compact-gate features.
    try:
        sys_path = ROOT / "bonsai-kg/python"
        import sys
        if str(sys_path) not in sys.path:
            sys.path.insert(0, str(sys_path))
        import bonsai_kg_v4 as bkg4  # type: ignore
        mapping = getattr(bkg4, "FEATURES", None)
        if isinstance(mapping, Mapping):
            return {str(k): int(v) for k, v in mapping.items()}
    except Exception:
        pass
    return {
        "ALWAYS": 0,
        "STEP": 1,
        "DAY": 2,
        "HOUR": 3,
        "OWN_MONEY": 4,
        "OPP_MONEY": 5,
        "OWN_UNITS": 6,
        "OPP_UNITS": 7,
        "OWN_LAND": 8,
        "OPP_LAND": 9,
        "OWN_HIRES_TODAY": 10,
        "MARKET_PRICE": 11,
        "MARKET_INVENTORY": 12,
        "OWN_SHED_TOTAL": 13,
        "SHOP_COUNT": 14,
        "OPP_HIRES_TODAY": 15,
        "OWN_SHED": 16,
        "OWN_SEED": 17,
        "SHOP_AT": 18,
        "SHOP_SEEN": 19,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="P0.6 bounded end-to-end autonomous Bonsai specialist synthesizer")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, help="optional copy of the final machine-readable run result")
    args = parser.parse_args()
    config = SynthesisConfig.load(args.config)
    result = synthesize_next_specialist(config)
    text = json.dumps(result.canonical(), sort_keys=True, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    print(json.dumps({"run_id": result.run_id, "verdict": result.verdict, "rejections": len(result.rejections), "cuda_games": result.runtime.get("cuda_games", 0)}, sort_keys=True))
    return 0 if result.verdict == ACCEPTED_RESEARCH_CANDIDATE else 2


if __name__ == "__main__":
    raise SystemExit(main())
