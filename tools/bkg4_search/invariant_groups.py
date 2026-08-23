from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass
from statistics import median
from typing import Iterable, Mapping, Sequence

from tools.bkg4_search.edit_search.panels import CaseRecord


@dataclass(frozen=True)
class TapeFold:
    fold: int
    train_tapes: tuple[str, ...]
    validation_tapes: tuple[str, ...]
    train_case_ids: tuple[str, ...]
    validation_case_ids: tuple[str, ...]


@dataclass(frozen=True)
class GroupOutcome:
    group_id: str
    new_loss: bool
    severe_regression: bool
    behavior_changed: bool
    useful: bool
    l2w: int
    margin_delta: float


@dataclass(frozen=True)
class ProductionQualification:
    production: bool
    safe_groups: int
    behavior_groups: int
    useful_groups: int
    l2w_groups: int
    total_groups: int
    reason: str


@dataclass(frozen=True)
class RobustGroupFitness:
    groups_with_new_loss: int
    groups_with_severe_regression: int
    unsafe_groups: int
    safe_groups: int
    behavior_groups: int
    useful_groups: int
    l2w_groups: int
    worst_group_delta: float
    lower_quartile_delta: float
    median_group_delta: float
    mean_group_delta: float

    @classmethod
    def from_outcomes(cls, outcomes: Sequence[GroupOutcome]) -> "RobustGroupFitness":
        if not outcomes:
            raise ValueError("group fitness requires at least one outcome")
        xs = sorted(float(x.margin_delta) for x in outcomes)
        qidx = max(0, int(math.floor(0.25 * (len(xs) - 1))))
        new = sum(bool(x.new_loss) for x in outcomes)
        severe = sum(bool(x.severe_regression) for x in outcomes)
        unsafe = sum(bool(x.new_loss or x.severe_regression) for x in outcomes)
        return cls(
            groups_with_new_loss=new,
            groups_with_severe_regression=severe,
            unsafe_groups=unsafe,
            safe_groups=len(outcomes) - unsafe,
            behavior_groups=sum(bool(x.behavior_changed) for x in outcomes),
            useful_groups=sum(bool(x.useful) for x in outcomes),
            l2w_groups=sum(int(x.l2w) > 0 for x in outcomes),
            worst_group_delta=min(xs),
            lower_quartile_delta=xs[qidx],
            median_group_delta=float(median(xs)),
            mean_group_delta=sum(xs) / len(xs),
        )

    def key(self) -> tuple:
        """Ascending lexicographic key: hard group safety first, then broad useful support."""
        return (
            int(self.groups_with_new_loss),
            int(self.groups_with_severe_regression),
            int(self.unsafe_groups),
            -int(self.safe_groups),
            -int(self.useful_groups),
            -int(self.l2w_groups),
            -int(self.behavior_groups),
            -float(self.worst_group_delta),
            -float(self.lower_quartile_delta),
            -float(self.median_group_delta),
            -float(self.mean_group_delta),
        )


@dataclass(frozen=True)
class ThresholdSolution:
    comparator: str
    threshold: int
    interval_lo: int
    interval_hi: int
    support_groups: int
    beneficial_hits: int
    harmed_protected: int
    boundary_margin: int
    score: tuple


def tape_group_id(case: CaseRecord) -> str:
    # The evaluator tape is the independent trajectory identity. Seat is intentionally excluded.
    return str(case.tape_path)


def grouped_cases(cases: Iterable[CaseRecord]) -> dict[str, tuple[CaseRecord, ...]]:
    out: dict[str, list[CaseRecord]] = {}
    for case in cases:
        out.setdefault(tape_group_id(case), []).append(case)
    return {key: tuple(sorted(rows, key=lambda x: (int(x.seat), str(x.case_id)))) for key, rows in sorted(out.items())}


def rotating_tape_folds(
    cases: Sequence[CaseRecord],
    *,
    n_folds: int = 4,
    seed: int = 0,
    discovery_only: bool = True,
) -> tuple[TapeFold, ...]:
    if discovery_only:
        bad = [x.case_id for x in cases if "cert" in str(x.role).lower() or "holdout" in str(x.role).lower() or "validation" in str(x.role).lower()]
        if bad:
            raise ValueError(f"certification/holdout cases cannot enter discovery folds: {bad[:4]}")
    groups = grouped_cases(cases)
    if not groups:
        raise ValueError("cannot fold empty case set")
    n_folds = max(2, min(int(n_folds), len(groups)))
    tapes = sorted(groups)
    # Stable seeded ordering independent of Python hash randomization.
    tapes.sort(key=lambda x: hashlib.sha256(f"{int(seed)}\0{x}".encode()).digest())
    buckets = [[] for _ in range(n_folds)]
    for i, tape in enumerate(tapes):
        buckets[i % n_folds].append(tape)
    folds: list[TapeFold] = []
    for fi, validation in enumerate(buckets):
        val = tuple(sorted(validation))
        train = tuple(sorted(t for t in tapes if t not in set(val)))
        folds.append(TapeFold(
            fold=fi,
            train_tapes=train,
            validation_tapes=val,
            train_case_ids=tuple(x.case_id for t in train for x in groups[t]),
            validation_case_ids=tuple(x.case_id for t in val for x in groups[t]),
        ))
    return tuple(folds)


def production_qualification(
    outcomes: Sequence[GroupOutcome],
    *,
    min_useful_groups: int = 2,
    min_behavior_groups: int | None = None,
) -> ProductionQualification:
    total = len(outcomes)
    safe = sum(not (x.new_loss or x.severe_regression) for x in outcomes)
    behavior = sum(bool(x.behavior_changed) for x in outcomes)
    useful = sum(bool(x.useful) for x in outcomes)
    l2w = sum(int(x.l2w) > 0 for x in outcomes)
    need_behavior = int(min_useful_groups if min_behavior_groups is None else min_behavior_groups)
    reason = "ok"
    production = True
    if total < int(min_useful_groups):
        production = False; reason = "insufficient independent tape groups"
    elif safe != total:
        production = False; reason = "unsafe discovery group"
    elif useful < int(min_useful_groups):
        production = False; reason = "insufficient useful tape-group support"
    elif behavior < need_behavior:
        production = False; reason = "insufficient behavior-changing tape-group support"
    return ProductionQualification(production, safe, behavior, useful, l2w, total, reason)


def _passes(value: int, threshold: int, comparator: str) -> bool:
    if comparator == "GE": return int(value) >= int(threshold)
    if comparator == "GT": return int(value) > int(threshold)
    if comparator == "LE": return int(value) <= int(threshold)
    if comparator == "LT": return int(value) < int(threshold)
    raise ValueError("constant solver supports GE/GT/LE/LT")


def solve_cross_tape_threshold(
    samples_by_group: Mapping[str, Sequence[tuple[int, bool, bool]]],
    *,
    comparator: str,
) -> ThresholdSolution:
    """Solve one bounded scalar constant from multi-tape discovery evidence.

    Each sample is `(value, beneficial, protected_negative)`. Hard priority is zero harmed
    protected examples, then number of distinct groups with a beneficial hit. Ties prefer
    wider equivalent intervals and larger distance from the nearest labeled boundary.
    Candidate constants come only from observed decision boundaries, never arbitrary int32s.
    """
    if comparator not in {"GE", "GT", "LE", "LT"}:
        raise ValueError("unsupported threshold comparator")
    rows = [(str(g), int(v), bool(pos), bool(prot)) for g, xs in samples_by_group.items() for v, pos, prot in xs]
    if not rows:
        raise ValueError("threshold solver requires samples")
    values = sorted(set(v for _, v, _, _ in rows))
    candidates = set(values)
    for a, b in zip(values, values[1:]):
        candidates.add(a + (b - a) // 2)
        if a < b:
            candidates.add(a + 1)
            candidates.add(b - 1)
    # Include one just outside the observed range for strict comparisons.
    candidates.add(values[0] - 1); candidates.add(values[-1] + 1)

    scored: list[tuple[tuple, int, int, int, int]] = []
    for t in sorted(candidates):
        harmed = sum(prot and _passes(v, t, comparator) for _, v, _, prot in rows)
        groups = {g for g, v, pos, _ in rows if pos and _passes(v, t, comparator)}
        hits = sum(pos and _passes(v, t, comparator) for _, v, pos, _ in rows)
        # Distance from threshold to the closest observed value discourages exact memorization.
        boundary = min(abs(v - t) for _, v, _, _ in rows)
        scored.append(((int(harmed), -len(groups), -int(hits), -int(boundary)), t, len(groups), int(hits), int(harmed)))
    scored.sort(key=lambda x: (x[0], x[1]))
    best_key, best_t, support, hits, harmed = scored[0]

    # Find the complete integer interval with exactly the same primary classification
    # (harmed protected, support groups, beneficial hits). This is the stable threshold basin.
    signature = (harmed, support, hits)
    equivalent = []
    lo_scan = values[0] - 2
    hi_scan = values[-1] + 2
    for t in range(lo_scan, hi_scan + 1):
        h = sum(prot and _passes(v, t, comparator) for _, v, _, prot in rows)
        gs = {g for g, v, pos, _ in rows if pos and _passes(v, t, comparator)}
        bh = sum(pos and _passes(v, t, comparator) for _, v, pos, _ in rows)
        if (int(h), len(gs), int(bh)) == signature:
            equivalent.append(t)
    if equivalent:
        interval_lo, interval_hi = min(equivalent), max(equivalent)
        # Prefer the center of the broad stable interval, with deterministic lower tie.
        chosen = (interval_lo + interval_hi) // 2
    else:
        interval_lo = interval_hi = best_t
        chosen = best_t
    boundary_margin = min(abs(v - chosen) for _, v, _, _ in rows)
    score = (harmed, -support, -hits, -(interval_hi - interval_lo), -boundary_margin)
    return ThresholdSolution(comparator, chosen, interval_lo, interval_hi, support, hits, harmed, boundary_margin, score)


def group_outcomes_from_case_rows(case_rows: Sequence[Mapping], case_to_tape: Mapping[str, str]) -> tuple[GroupOutcome, ...]:
    """Aggregate exact pair rows into trajectory-level robust outcomes.

    Expected row keys are the existing edit-search pair fields: `case_id`, `baseline_margin`,
    `candidate_margin`, `margin_delta`, and optionally `action_changed`/`activation_count`.
    """
    buckets: dict[str, list[Mapping]] = {}
    for row in case_rows:
        cid = str(row["case_id"])
        if cid not in case_to_tape:
            raise ValueError(f"case {cid} missing tape grouping")
        buckets.setdefault(str(case_to_tape[cid]), []).append(row)
    out = []
    for tape, rows in sorted(buckets.items()):
        new_loss = any(int(r["baseline_margin"]) > 0 and int(r["candidate_margin"]) <= 0 for r in rows)
        severe = any(int(r["baseline_margin"]) > 0 and int(r["margin_delta"]) < -3000 for r in rows)
        changed = any(bool(r.get("action_changed", False)) or int(r["margin_delta"]) != 0 for r in rows)
        useful = (not new_loss and not severe and changed and any(int(r["margin_delta"]) > 0 for r in rows))
        l2w = sum(int(r["baseline_margin"]) < 0 < int(r["candidate_margin"]) for r in rows)
        delta = sum(float(r["margin_delta"]) for r in rows) / len(rows)
        out.append(GroupOutcome(tape, new_loss, severe, changed, useful, l2w, delta))
    return tuple(out)
