from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from tools.bkg4_search.edit_search.goals import SearchGoal, synthesize_contrastive_conditions
from tools.bkg4_search.module_codec import ConditionGene


@dataclass(frozen=True)
class MacroEffectCase:
    case_id: str
    baseline_margin: int
    macro_margin: int
    context: Mapping[tuple[str, int], int]
    exact_signature: tuple[int, ...] = ()
    support_group: str = ""
    dataset_role: str = "discovery"
    required_target_positive: bool = False
    required_target_l2w: bool = False

    @property
    def delta(self) -> int:
        return int(self.macro_margin) - int(self.baseline_margin)

    @property
    def loss_to_win(self) -> bool:
        return self.baseline_margin < 0 < self.macro_margin

    @property
    def win_to_loss(self) -> bool:
        return self.baseline_margin > 0 > self.macro_margin


@dataclass(frozen=True)
class GateScore:
    activated: int
    positive: int
    neutral: int
    harmful: int
    loss_to_win: int
    win_to_loss: int
    worst_delta: int
    aggregate_delta: int
    support_groups: int
    complexity: int
    target_positive: int = 0
    target_loss_to_win: int = 0

    def lexicographic_key(self) -> tuple[int, ...]:
        # Strict promotion ordering: safety, original target rescue, transfer,
        # worst-case margin, simplicity, then aggregate money.
        return (
            self.win_to_loss,
            self.harmful,
            -self.target_loss_to_win,
            -self.target_positive,
            -self.loss_to_win,
            -self.positive,
            -self.support_groups,
            -self.worst_delta,
            self.complexity,
            -self.aggregate_delta,
        )


def condition_hits(condition: ConditionGene, context: Mapping[tuple[str, int], int]) -> bool:
    value = context.get((condition.feature, int(condition.index)))
    if value is None:
        return False
    if condition.cmp == "EQ": return value == condition.value
    if condition.cmp == "NE": return value != condition.value
    if condition.cmp == "LE": return value <= condition.value
    if condition.cmp == "GE": return value >= condition.value
    raise ValueError(condition.cmp)


def gate_hits(clauses: Sequence[Sequence[ConditionGene]], context: Mapping[tuple[str, int], int]) -> bool:
    return any(clause and all(condition_hits(c, context) for c in clause) for clause in clauses)


def evaluate_gate(clauses: Sequence[Sequence[ConditionGene]], cases: Sequence[MacroEffectCase]) -> GateScore:
    active = [case for case in cases if gate_hits(clauses, case.context)]
    deltas = [case.delta for case in active]
    return GateScore(
        activated=len(active),
        positive=sum(x > 0 for x in deltas),
        neutral=sum(x == 0 for x in deltas),
        harmful=sum(x < 0 for x in deltas),
        loss_to_win=sum(case.loss_to_win for case in active),
        win_to_loss=sum(case.win_to_loss for case in active),
        worst_delta=min(deltas, default=0),
        aggregate_delta=sum(deltas),
        support_groups=len({case.support_group for case in active if case.support_group}),
        complexity=sum(len(clause) for clause in clauses),
        target_positive=sum(case.required_target_positive and case.delta > 0 for case in active),
        target_loss_to_win=sum(case.required_target_l2w and case.loss_to_win for case in active),
    )


def selected_mixed_exact_cells(clauses: Sequence[Sequence[ConditionGene]], cases: Sequence[MacroEffectCase]) -> tuple[tuple[str, ...], ...]:
    cells: dict[tuple[int, ...], list[MacroEffectCase]] = {}
    for case in cases:
        if case.exact_signature and gate_hits(clauses, case.context):
            cells.setdefault(case.exact_signature, []).append(case)
    mixed = []
    for rows in cells.values():
        if any(x.delta > 0 for x in rows) and any(x.delta < 0 for x in rows):
            mixed.append(tuple(sorted(x.case_id for x in rows)))
    return tuple(sorted(mixed))


def _contexts(cases: Sequence[MacroEffectCase]) -> dict[str, tuple[dict[str, int], ...]]:
    return {
        case.case_id: tuple(
            {"probe_index": 0, "probe_step": 0, "feature": feature, "index": index, "value": int(value)}
            for (feature, index), value in sorted(case.context.items())
        )
        for case in cases
    }


def enumerate_safe_conjunctions(
    cases: Sequence[MacroEffectCase],
    *,
    max_conditions: int = 4,
    limit: int = 256,
    feature_budget: int = 128,
    singleton_l2w_budget: int | None = None,
) -> list[tuple[tuple[ConditionGene, ...], GateScore]]:
    """Enumerate compact harm-excluding conjunctions using the existing BKG4 grammar.

    Candidate generation is contrastive; final ranking is effect-aware and therefore
    preserves L->W before positive count or average money.
    """
    if not cases:
        return []
    positive = [x.case_id for x in cases if x.delta > 0]
    harmful = [x.case_id for x in cases if x.delta < 0]
    l2w = [x.case_id for x in cases if x.loss_to_win]
    required_target_positive = [x.case_id for x in cases if x.required_target_positive and x.delta > 0]
    required_target_l2w = [x.case_id for x in cases if x.required_target_l2w and x.loss_to_win]
    if not positive or not harmful:
        return []
    contexts = _contexts(cases)
    target_sets: list[tuple[str, ...]] = []
    for target_ids in (required_target_l2w, required_target_positive):
        target_tuple = tuple(target_ids)
        if target_tuple and target_tuple not in target_sets:
            target_sets.append(target_tuple)
    # Preserve the historical bounded proposal schedule exactly for non-target rows;
    # required-target proposal sets are additive, not a rewrite of the old search.
    target_sets.append(tuple(positive))
    if l2w:
        target_sets.append(tuple(l2w))
        singleton_ids = required_target_l2w + [x for x in l2w if x not in required_target_l2w]
        if singleton_l2w_budget is not None:
            singleton_ids = singleton_ids[: max(0, int(singleton_l2w_budget))]
        target_sets.extend((case_id,) for case_id in singleton_ids)
    candidates: dict[tuple[ConditionGene, ...], GateScore] = {}
    for target_ids in target_sets:
        goal = SearchGoal(tuple(target_ids), tuple(harmful), (), tuple(harmful))
        rows = synthesize_contrastive_conditions(
            goal,
            contexts,
            max_conditions=max_conditions,
            limit=limit,
            feature_budget=feature_budget,
            guard_excluding_only=True,
            allow_temporal=False,
        )
        for conditions in rows:
            score = evaluate_gate((conditions,), cases)
            if score.harmful or score.win_to_loss:
                continue
            if selected_mixed_exact_cells((conditions,), cases):
                continue
            old = candidates.get(conditions)
            if old is None or score.lexicographic_key() < old.lexicographic_key():
                candidates[conditions] = score
    return sorted(candidates.items(), key=lambda row: row[1].lexicographic_key())


def refine_conjunction(
    base_clause: Sequence[ConditionGene],
    cases: Sequence[MacroEffectCase],
    *,
    max_added_conditions: int = 1,
    limit: int = 256,
    feature_budget: int = 128,
) -> list[tuple[tuple[ConditionGene, ...], GateScore]]:
    """CEGIS refinement for one active conjunction.

    Only cases currently selected by ``base_clause`` participate in contrastive
    refinement. Positive rows in exact mixed-sign legal-state cells are excluded
    from targets rather than being rescued with hidden/future information.
    """
    base = tuple(base_clause)
    active = [case for case in cases if all(condition_hits(c, case.context) for c in base)]
    harmful = [case for case in active if case.delta < 0]
    if not harmful:
        return [(base, evaluate_gate((base,), cases))]
    mixed_ids: set[str] = set()
    cells: dict[tuple[int, ...], list[MacroEffectCase]] = {}
    for case in active:
        if case.exact_signature:
            cells.setdefault(case.exact_signature, []).append(case)
    for rows in cells.values():
        if any(x.delta > 0 for x in rows) and any(x.delta < 0 for x in rows):
            mixed_ids.update(x.case_id for x in rows)
    targets = [case for case in active if case.delta > 0 and case.case_id not in mixed_ids]
    if not targets:
        return []
    contexts = _contexts(active)
    goal = SearchGoal(
        tuple(x.case_id for x in targets),
        tuple(x.case_id for x in harmful),
        (),
        tuple(x.case_id for x in harmful),
    )
    extras = synthesize_contrastive_conditions(
        goal,
        contexts,
        max_conditions=max_added_conditions,
        limit=limit,
        feature_budget=feature_budget,
        guard_excluding_only=True,
        allow_temporal=False,
    )
    ranked: dict[tuple[ConditionGene, ...], GateScore] = {}
    for extra in extras:
        clause = base + tuple(c for c in extra if c not in base)
        if len(clause) > len(base) + max_added_conditions:
            continue
        score = evaluate_gate((clause,), cases)
        if score.harmful or score.win_to_loss:
            continue
        if selected_mixed_exact_cells((clause,), cases):
            continue
        ranked[clause] = score
    return sorted(ranked.items(), key=lambda row: row[1].lexicographic_key())


def choose_best_gate(candidate_clauses: Sequence[Sequence[Sequence[ConditionGene]]], cases: Sequence[MacroEffectCase]) -> tuple[tuple[tuple[ConditionGene, ...], ...], GateScore] | None:
    ranked = []
    for gate in candidate_clauses:
        clauses = tuple(tuple(c for c in clause) for clause in gate)
        score = evaluate_gate(clauses, cases)
        if selected_mixed_exact_cells(clauses, cases):
            continue
        ranked.append((score.lexicographic_key(), clauses, score))
    if not ranked:
        return None
    _, clauses, score = min(ranked, key=lambda row: row[0])
    return clauses, score
