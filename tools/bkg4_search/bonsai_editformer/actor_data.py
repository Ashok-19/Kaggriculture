from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from enum import IntEnum
from pathlib import Path
from typing import Iterable

import numpy as np

from actor_grammar import (
    CMP_INDEX,
    EDIT_OPERATIONS,
    FEATURE_INDEX,
    OP_INDEX,
    EditAction,
    node_refs,
    normalize_value,
    target_masks,
)
from data import CMP_NAME, FEATURE_NAME, OUT, ROUTES, decode_genome, history_rows
from tools.bkg4_search.module_codec import ConditionGene, ModuleGenome, RuleGene

ROUTE_INDEX = {r: i for i, r in enumerate(ROUTES)}
ACTOR_DATA_DIR = OUT / "actor"


class GoalClass(IntEnum):
    """Lexicographic per-failure outcome class for goal-conditioned supervision."""

    SEVERE_REGRESSION = -2
    TARGET_HARM = -1
    NO_EFFECT = 0
    UNSAFE_IMPROVE = 1
    SAFE_IMPROVE = 2
    SAFE_FLIP = 3


def classify_goal_outcome(*, parent_margin: int, child_margin: int, w2l: int, reg3: int) -> GoalClass:
    """Classify one requested failure without allowing aggregate gains to hide regressions."""
    delta = int(child_margin) - int(parent_margin)
    unsafe = int(w2l) > 0 or int(reg3) > 0
    if delta > 0:
        if unsafe:
            return GoalClass.UNSAFE_IMPROVE
        return GoalClass.SAFE_FLIP if int(parent_margin) < 0 < int(child_margin) else GoalClass.SAFE_IMPROVE
    if unsafe:
        return GoalClass.SEVERE_REGRESSION
    if delta < 0:
        return GoalClass.TARGET_HARM
    return GoalClass.NO_EFFECT


def goal_example_weight(
    goal_class: GoalClass | int,
    *,
    target_delta_raw: int,
    beta: float = 2.0,
    max_positive: float = 12.0,
    max_negative: float = 8.0,
) -> dict[str, float | bool]:
    """Return bounded class-gated supervision weight; magnitude never overrides safety."""
    cls = GoalClass(int(goal_class))
    beta = max(1e-6, float(beta))
    delta = int(target_delta_raw)
    if cls in (GoalClass.SAFE_FLIP, GoalClass.SAFE_IMPROVE):
        class_bonus = 4.0 if cls == GoalClass.SAFE_FLIP else 2.0
        magnitude = min(4.0, max(0.0, delta / 5000.0))
        advantage = class_bonus + magnitude
        weight = min(float(max_positive), math.exp(advantage / beta))
        return {"positive": True, "advantage": float(advantage), "weight": float(weight)}
    if cls == GoalClass.UNSAFE_IMPROVE:
        weight = 4.0 + min(2.0, max(0.0, delta / 5000.0))
    elif cls == GoalClass.SEVERE_REGRESSION:
        weight = 5.0 + min(2.0, abs(delta) / 5000.0)
    elif cls == GoalClass.TARGET_HARM:
        weight = 2.0 + min(2.0, abs(delta) / 5000.0)
    else:
        weight = 0.5
    return {"positive": False, "advantage": 0.0, "weight": float(min(float(max_negative), weight))}


def relabel_failure_goals(parent_margins: np.ndarray, child_margins: np.ndarray) -> tuple[list[dict], dict[str, int]]:
    """Hindsight-relabel one exact mutation against every parent losing case."""
    parent = np.asarray(parent_margins, np.int32)
    child = np.asarray(child_margins, np.int32)
    if parent.shape != child.shape:
        raise ValueError(f"parent/child margin shape mismatch: {parent.shape} != {child.shape}")
    delta = child - parent
    parent_win = parent > 0
    w2l = int(np.sum(parent_win & (child <= 0)))
    reg3 = int(np.sum(parent_win & (delta < -3000)))
    examples: list[dict] = []
    for case_id in np.flatnonzero(parent < 0):
        raw_delta = int(delta[case_id])
        cls = classify_goal_outcome(
            parent_margin=int(parent[case_id]), child_margin=int(child[case_id]), w2l=w2l, reg3=reg3,
        )
        examples.append({
            "goal_case_id": int(case_id),
            "goal_class": cls,
            "parent_margin": int(parent[case_id]),
            "child_margin": int(child[case_id]),
            "target_delta_raw": raw_delta,
            "target_delta_norm": float(math.tanh(raw_delta / 5000.0)),
            "target_improve": bool(raw_delta > 0),
            "target_flip": bool(parent[case_id] < 0 < child[case_id]),
            "w2l": w2l,
            "reg3": reg3,
        })
    return examples, {"w2l": w2l, "reg3": reg3}


def operation_support_mask(operation_support: dict, *, min_safe_positive: int) -> np.ndarray:
    """Return the learned-policy operation mask from exact safe-positive goal support."""
    threshold = max(0, int(min_safe_positive))
    return np.asarray([
        int(operation_support.get(name, {}).get("safe_positive_examples", 0)) >= threshold
        for name in EDIT_OPERATIONS
    ], dtype=bool)


def assert_no_relabel_leakage(transition_idx: np.ndarray, train_idx: np.ndarray, val_idx: np.ndarray) -> None:
    """Require every relabel of one semantic mutation to stay on one split side."""
    transition_idx = np.asarray(transition_idx, np.int64)
    train_transitions = set(int(x) for x in transition_idx[np.asarray(train_idx, np.int64)])
    val_transitions = set(int(x) for x in transition_idx[np.asarray(val_idx, np.int64)])
    crossing = train_transitions & val_transitions
    if crossing:
        sample = sorted(crossing)[:8]
        raise ValueError(f"relabel leakage across train/validation transition groups: {sample}")


def _validation_class_counts(ids: np.ndarray, goal_class: np.ndarray, actor_op: np.ndarray) -> dict[str, int]:
    cls = np.asarray(goal_class, np.int8)[ids]
    ops = np.asarray(actor_op, np.int32)[ids]
    return {
        "n": int(len(ids)),
        "target_positive": int(np.sum(np.isin(cls, [int(GoalClass.SAFE_FLIP), int(GoalClass.SAFE_IMPROVE)]))),
        "target_harm": int(np.sum(cls == int(GoalClass.TARGET_HARM))),
        "no_effect": int(np.sum(cls == int(GoalClass.NO_EFFECT))),
        "severe_regression": int(np.sum(cls == int(GoalClass.SEVERE_REGRESSION))),
        "unsafe_improve": int(np.sum(cls == int(GoalClass.UNSAFE_IMPROVE))),
        "operation_families": int(len(set(int(x) for x in ops))),
    }


def stratified_validation_indices(
    candidate_ids: np.ndarray,
    goal_class: np.ndarray,
    actor_op: np.ndarray,
    *,
    max_samples: int,
) -> tuple[np.ndarray, dict[str, int]]:
    """Deterministically select held-out examples across direction and operation strata."""
    candidate_ids = np.asarray(candidate_ids, np.int32)
    goal_class = np.asarray(goal_class, np.int8)
    actor_op = np.asarray(actor_op, np.int32)
    required = {
        "target_positive": np.isin(goal_class[candidate_ids], [int(GoalClass.SAFE_FLIP), int(GoalClass.SAFE_IMPROVE)]),
        "target_harm": goal_class[candidate_ids] == int(GoalClass.TARGET_HARM),
        "no_effect": goal_class[candidate_ids] == int(GoalClass.NO_EFFECT),
        "severe_regression": goal_class[candidate_ids] == int(GoalClass.SEVERE_REGRESSION),
    }
    missing = [name for name, mask in required.items() if not np.any(mask)]
    if missing:
        raise ValueError(f"required validation class absent: {', '.join(missing)}")
    budget = min(max(1, int(max_samples)), len(candidate_ids))
    # Round-robin over (direction class, operation) groups, preserving stable index order.
    groups: dict[tuple[int, int], list[int]] = defaultdict(list)
    for idx in sorted(int(x) for x in candidate_ids):
        groups[(int(goal_class[idx]), int(actor_op[idx]))].append(idx)
    keys = sorted(groups)
    selected: list[int] = []
    depth = 0
    while len(selected) < budget:
        added = False
        for key in keys:
            xs = groups[key]
            if depth < len(xs):
                selected.append(xs[depth]); added = True
                if len(selected) >= budget:
                    break
        if not added:
            break
        depth += 1
    ids = np.asarray(selected, np.int32)
    counts = _validation_class_counts(ids, goal_class, actor_op)
    selected_missing = [name for name in ("target_positive", "target_harm", "no_effect", "severe_regression") if counts[name] <= 0]
    if selected_missing:
        raise ValueError(f"required validation class missing after stratification: {', '.join(selected_missing)}")
    return ids, counts


@dataclass(frozen=True)
class Transition:
    parent: int
    child: int
    action: EditAction
    advantage: float
    positive: bool
    weight: float


def row_genome(row) -> ModuleGenome:
    d = decode_genome(row.blob)
    rules = []
    for rr in d.rules:
        # Historical search corpora can contain repeated identical predicates in a
        # conjunction. They are semantically redundant (A AND A == A) but the typed
        # ModuleGenome grammar deliberately rejects duplicate conditions. Normalize
        # only exact duplicates here, preserving first occurrence/order and therefore
        # the evaluated policy semantics used by the attached CUDA labels.
        conds = []
        seen = set()
        for fid, cmp_code, index, value in rr["conds"]:
            cond = ConditionGene(FEATURE_NAME[fid], CMP_NAME[cmp_code], int(index), int(value))
            if cond in seen:
                continue
            seen.add(cond)
            conds.append(cond)
        rules.append(RuleGene(tuple(conds), int(rr["route"])))
    g = ModuleGenome(int(d.prefix), tuple(rules))
    g.validate(ROUTES)
    return g


def genome_key(g: ModuleGenome):
    return (
        int(g.prefix_step),
        tuple((tuple((c.feature, c.cmp, int(c.index), int(c.value)) for c in r.conditions), int(r.route)) for r in g.rules),
    )


def row_quality(row) -> float:
    l = row.labels
    return (
        30.0 * float(l["safe"])
        + 7.0 * float(l["l2w"])
        - 18.0 * float(l["w2l"])
        - 8.0 * float(l["reg3"])
        + float(l["mean_delta"]) / 2000.0
    )


def rule_node_index(g: ModuleGenome, ri: int) -> int:
    pos = 3
    for j, r in enumerate(g.rules):
        if j == ri:
            return pos
        pos += 1 + len(r.conditions)
    raise IndexError(ri)


def condition_node_index(g: ModuleGenome, ri: int, ci: int) -> int:
    return rule_node_index(g, ri) + 1 + ci


def _rule_action(operation: str, target: int, rule: RuleGene, *, priority: int = 0) -> EditAction:
    c1 = rule.conditions[0]
    c2 = rule.conditions[1] if len(rule.conditions) > 1 else c1
    return EditAction(
        operation=operation, target_node=target,
        feature=c1.feature, cmp=c1.cmp, index=c1.index, value=c1.value,
        route=rule.route,
        feature2=c2.feature, cmp2=c2.cmp, index2=c2.index, value2=c2.value,
        condition_count=min(2, len(rule.conditions)), priority=priority,
    )


def infer_action(parent: ModuleGenome, child: ModuleGenome) -> EditAction | None:
    if parent.prefix_step != child.prefix_step and parent.rules == child.rules:
        return EditAction("CHANGE_WINDOW", 0, prefix_step=child.prefix_step)

    # Exact insertion/removal of one complete rule.
    if len(child.rules) == len(parent.rules) + 1:
        for ci in range(len(child.rules)):
            if child.rules[:ci] + child.rules[ci + 1:] == parent.rules:
                added = child.rules[ci]
                target = 0 if ci == 0 else rule_node_index(parent, min(ci - 1, len(parent.rules) - 1))
                op = "ADD_SPECIALIST"
                if parent.rules and ci < len(parent.rules):
                    base = parent.rules[ci]
                    if set(base.conditions).issubset(set(added.conditions)) and added.conditions != base.conditions:
                        op = "SPLIT_LANE"
                        target = rule_node_index(parent, ci)
                return _rule_action(op, target, added, priority=ci)
    if len(parent.rules) == len(child.rules) + 1:
        for pi in range(len(parent.rules)):
            if parent.rules[:pi] + parent.rules[pi + 1:] == child.rules:
                return EditAction("REMOVE_REDUNDANT_BRANCH", rule_node_index(parent, pi), priority=pi)

    if len(parent.rules) != len(child.rules):
        return None

    diff_rules = [i for i, (a, b) in enumerate(zip(parent.rules, child.rules)) if a != b]
    if not diff_rules:
        return None

    # Pure priority move: same multiset of rules.
    if sorted(parent.rules, key=repr) == sorted(child.rules, key=repr):
        for ri, r in enumerate(parent.rules):
            if ri >= len(child.rules) or child.rules[ri] != r:
                new_pos = child.rules.index(r)
                return EditAction("CHANGE_PRIORITY", rule_node_index(parent, ri), priority=new_pos)

    if len(diff_rules) != 1:
        return None
    ri = diff_rules[0]
    a, b = parent.rules[ri], child.rules[ri]
    target_rule = rule_node_index(parent, ri)

    if a.conditions == b.conditions and a.route != b.route:
        return _rule_action("CHANGE_ROUTE", target_rule, b)

    if a.route == b.route and len(b.conditions) == len(a.conditions) + 1:
        for ci in range(len(b.conditions)):
            if b.conditions[:ci] + b.conditions[ci + 1:] == a.conditions:
                added = b.conditions[ci]
                op = "COMPOSE_PREDICATES" if len(a.conditions) >= 2 else "ADD_CONDITION"
                return EditAction(op, target_rule, feature=added.feature, cmp=added.cmp, index=added.index, value=added.value,
                                  route=b.route, condition_count=1)

    if a.route == b.route and len(a.conditions) == len(b.conditions):
        diffs = [i for i, (x, y) in enumerate(zip(a.conditions, b.conditions)) if x != y]
        if len(diffs) == 1:
            ci = diffs[0]; x, y = a.conditions[ci], b.conditions[ci]
            if (x.feature, x.cmp, x.index) == (y.feature, y.cmp, y.index) and x.value != y.value:
                return EditAction("CHANGE_THRESHOLD", condition_node_index(parent, ri, ci), feature=x.feature, cmp=x.cmp,
                                  index=x.index, value=y.value, route=b.route)

    # Bounded rule replacement captures changes that are more than one local scalar edit.
    if b.conditions:
        return _rule_action("REPLACE_SUBTREE", target_rule, b, priority=ri)
    return None


def _direct_candidate_pairs(genomes: list[ModuleGenome]) -> set[tuple[int, int]]:
    exact = {genome_key(g): i for i, g in enumerate(genomes)}
    pairs: set[tuple[int, int]] = set()
    for ci, g in enumerate(genomes):
        # Inverse rule/condition edits find exact evaluated parents without O(N^2).
        for ri in range(len(g.rules)):
            pg = ModuleGenome(g.prefix_step, g.rules[:ri] + g.rules[ri + 1:])
            pi = exact.get(genome_key(pg))
            if pi is not None: pairs.add((pi, ci)); pairs.add((ci, pi))
            r = g.rules[ri]
            if len(r.conditions) > 1:
                for cj in range(len(r.conditions)):
                    nr = RuleGene(r.conditions[:cj] + r.conditions[cj + 1:], r.route)
                    pg = ModuleGenome(g.prefix_step, g.rules[:ri] + (nr,) + g.rules[ri + 1:])
                    pi = exact.get(genome_key(pg))
                    if pi is not None: pairs.add((pi, ci)); pairs.add((ci, pi))
    return pairs


def _group_candidate_pairs(genomes: list[ModuleGenome], qualities: np.ndarray, max_group: int = 12) -> set[tuple[int, int]]:
    groups: dict[tuple, list[int]] = defaultdict(list)
    for i, g in enumerate(genomes):
        groups[("prefix", tuple(g.rules))].append(i)
        groups[("priority", g.prefix_step, tuple(sorted(g.rules, key=repr)))].append(i)
        for ri, r in enumerate(g.rules):
            groups[("route", g.prefix_step, ri, g.rules[:ri], r.conditions, g.rules[ri + 1:])].append(i)
            groups[("subtree", g.prefix_step, ri, g.rules[:ri], g.rules[ri + 1:])].append(i)
            for ci, c in enumerate(r.conditions):
                groups[("value", g.prefix_step, ri, ci, g.rules[:ri], r.route, r.conditions[:ci],
                        (c.feature, c.cmp, c.index), r.conditions[ci + 1:], g.rules[ri + 1:])].append(i)
    pairs: set[tuple[int, int]] = set()
    for ids in groups.values():
        ids = list(dict.fromkeys(ids))
        if len(ids) < 2:
            continue
        ids = sorted(ids, key=lambda x: (-qualities[x], x))[:max_group]
        best = ids[0]
        for other in ids[1:]:
            pairs.add((other, best)); pairs.add((best, other))
    return pairs


def build_transitions(limit_rows: int | None = None, max_transitions: int = 60_000) -> tuple[list[Transition], list[ModuleGenome], list]:
    rows = history_rows()[0]
    if limit_rows is not None:
        rows = rows[:limit_rows]
    genomes = [row_genome(r) for r in rows]
    qualities = np.asarray([row_quality(r) for r in rows], np.float32)
    pairs = _direct_candidate_pairs(genomes) | _group_candidate_pairs(genomes, qualities)
    transitions: list[Transition] = []
    for pi, ci in sorted(pairs):
        action = infer_action(genomes[pi], genomes[ci])
        if action is None:
            continue
        adv = float(qualities[ci] - qualities[pi])
        child = rows[ci].labels
        catastrophic = bool(child["w2l"] > rows[pi].labels["w2l"] or child["reg3"] > rows[pi].labels["reg3"])
        positive = adv > 0.05 and not catastrophic
        if positive:
            weight = min(12.0, 1.0 + max(0.0, adv) + 4.0 * float(child["safe"]) + 0.5 * float(child["l2w"]))
        else:
            weight = min(6.0, 0.5 + abs(min(0.0, adv)) + 2.0 * float(catastrophic))
        transitions.append(Transition(pi, ci, action, adv, positive, float(weight)))
    transitions.sort(key=lambda t: (not t.positive, -t.weight, t.parent, t.child, t.action.operation))
    return transitions[:max_transitions], genomes, rows


def split_transition_indices(transitions: list[Transition]) -> tuple[np.ndarray, np.ndarray]:
    train, val = [], []
    for i, t in enumerate(transitions):
        key = f"{t.parent}:{t.child}:{t.action.operation}".encode()
        bucket = int(hashlib.sha256(key).hexdigest()[:8], 16) % 5
        (val if bucket == 0 else train).append(i)
    return np.asarray(train, np.int32), np.asarray(val, np.int32)


def _action_arrays(transitions: list[Transition], genomes: list[ModuleGenome], tree_cap: int = 48) -> dict[str, np.ndarray]:
    n = len(transitions)
    a = {
        "parent_idx": np.zeros(n, np.int32), "child_idx": np.zeros(n, np.int32),
        "actor_op": np.zeros(n, np.int32), "actor_target": np.zeros(n, np.int32),
        "actor_target_mask": np.zeros((n, len(EDIT_OPERATIONS), tree_cap), bool),
        "feature": np.zeros(n, np.int32), "cmp": np.zeros(n, np.int32), "route": np.zeros(n, np.int32), "index": np.zeros(n, np.int32),
        "feature2": np.zeros(n, np.int32), "cmp2": np.zeros(n, np.int32), "index2": np.zeros(n, np.int32),
        "condition_count": np.zeros(n, np.int32), "priority": np.zeros(n, np.int32),
        "value_norm": np.zeros(n, np.float32), "value2_norm": np.zeros(n, np.float32), "prefix_norm": np.zeros(n, np.float32),
        "legacy_positive": np.zeros(n, np.float32), "legacy_weight": np.zeros(n, np.float32), "legacy_advantage": np.zeros(n, np.float32),
    }
    for i, t in enumerate(transitions):
        x = t.action; parent = genomes[t.parent]
        a["parent_idx"][i] = t.parent; a["child_idx"][i] = t.child; a["actor_op"][i] = OP_INDEX[x.operation]; a["actor_target"][i] = x.target_node
        a["actor_target_mask"][i] = np.asarray(target_masks(parent, tree_cap), bool)
        a["feature"][i] = FEATURE_INDEX[x.feature]; a["cmp"][i] = CMP_INDEX[x.cmp]; a["route"][i] = ROUTE_INDEX.get(x.route, 0); a["index"][i] = max(0, min(99, x.index))
        a["feature2"][i] = FEATURE_INDEX[x.feature2]; a["cmp2"][i] = CMP_INDEX[x.cmp2]; a["index2"][i] = max(0, min(99, x.index2))
        a["condition_count"][i] = max(0, min(1, x.condition_count - 1)); a["priority"][i] = max(0, min(7, x.priority))
        a["value_norm"][i] = normalize_value(x.feature, x.value); a["value2_norm"][i] = normalize_value(x.feature2, x.value2)
        a["prefix_norm"][i] = 2.0 * max(0, min(719, x.prefix_step)) / 719.0 - 1.0
        a["legacy_positive"][i] = float(t.positive); a["legacy_weight"][i] = t.weight; a["legacy_advantage"][i] = t.advantage
    return a


def goal_supervision_arrays(
    transitions: list[Transition],
    rows: list,
    *,
    case_cap: int = 48,
    awr_beta: float = 2.0,
    max_positive_weight: float = 12.0,
    max_negative_weight: float = 8.0,
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, int]]]:
    """Build pair-relative labels once per mutation plus compact hindsight goal examples."""
    nt = len(transitions)
    pair_parent_margin = np.zeros((nt, case_cap), np.int32)
    pair_child_margin = np.zeros((nt, case_cap), np.int32)
    pair_case_delta_raw = np.zeros((nt, case_cap), np.int32)
    pair_case_delta_norm = np.zeros((nt, case_cap), np.float16)
    pair_case_mask = np.zeros((nt, case_cap), bool)
    pair_case_improve = np.zeros((nt, case_cap), bool)
    pair_case_flip = np.zeros((nt, case_cap), bool)
    pair_case_regress = np.zeros((nt, case_cap), bool)
    pair_case_reg3 = np.zeros((nt, case_cap), bool)

    support: dict[str, dict[str, int]] = {
        name: {
            "transitions": 0,
            "goal_examples": 0,
            "target_positive_examples": 0,
            "safe_positive_examples": 0,
            "safe_l2w_examples": 0,
            "target_harm_examples": 0,
            "target_noop_examples": 0,
            "harmful_examples": 0,
            "severe_regression_examples": 0,
        }
        for name in EDIT_OPERATIONS
    }
    ex_transition: list[int] = []
    ex_goal_case: list[int] = []
    ex_goal_class: list[int] = []
    ex_target_delta_raw: list[int] = []
    ex_target_delta_norm: list[float] = []
    ex_target_improve: list[bool] = []
    ex_target_flip: list[bool] = []
    ex_positive: list[float] = []
    ex_weight: list[float] = []
    ex_advantage: list[float] = []
    ex_w2l: list[int] = []
    ex_reg3: list[int] = []

    for ti, t in enumerate(transitions):
        parent = rows[t.parent]
        child = rows[t.child]
        nc = min(int(parent.nc) if hasattr(parent, "nc") else len(parent.margins),
                 int(child.nc) if hasattr(child, "nc") else len(child.margins), case_cap)
        if not np.array_equal(np.asarray(parent.baseline[:nc]), np.asarray(child.baseline[:nc])):
            raise ValueError(f"transition {ti} parent/child baseline mismatch")
        pm = np.asarray(parent.margins[:nc], np.int32)
        cm = np.asarray(child.margins[:nc], np.int32)
        delta = cm - pm
        pair_parent_margin[ti, :nc] = pm
        pair_child_margin[ti, :nc] = cm
        pair_case_delta_raw[ti, :nc] = delta
        pair_case_delta_norm[ti, :nc] = np.tanh(delta.astype(np.float64) / 5000.0).astype(np.float16)
        pair_case_mask[ti, :nc] = True
        pair_case_improve[ti, :nc] = delta > 0
        pair_case_flip[ti, :nc] = (pm < 0) & (cm > 0)
        pair_case_regress[ti, :nc] = (pm > 0) & (cm <= 0)
        pair_case_reg3[ti, :nc] = (pm > 0) & (delta < -3000)

        examples, summary = relabel_failure_goals(pm, cm)
        op_support = support[t.action.operation]
        op_support["transitions"] += 1
        for ex in examples:
            cls = GoalClass(int(ex["goal_class"]))
            wt = goal_example_weight(
                cls,
                target_delta_raw=int(ex["target_delta_raw"]),
                beta=awr_beta,
                max_positive=max_positive_weight,
                max_negative=max_negative_weight,
            )
            ex_transition.append(ti)
            ex_goal_case.append(int(ex["goal_case_id"]))
            ex_goal_class.append(int(cls))
            ex_target_delta_raw.append(int(ex["target_delta_raw"]))
            ex_target_delta_norm.append(float(ex["target_delta_norm"]))
            ex_target_improve.append(bool(ex["target_improve"]))
            ex_target_flip.append(bool(ex["target_flip"]))
            ex_positive.append(float(bool(wt["positive"])))
            ex_weight.append(float(wt["weight"]))
            ex_advantage.append(float(wt["advantage"]))
            ex_w2l.append(int(summary["w2l"]))
            ex_reg3.append(int(summary["reg3"]))
            op_support["goal_examples"] += 1
            op_support["target_positive_examples"] += int(int(ex["target_delta_raw"]) > 0)
            op_support["safe_positive_examples"] += int(cls in (GoalClass.SAFE_FLIP, GoalClass.SAFE_IMPROVE))
            op_support["safe_l2w_examples"] += int(cls == GoalClass.SAFE_FLIP)
            op_support["target_harm_examples"] += int(int(ex["target_delta_raw"]) < 0)
            op_support["target_noop_examples"] += int(int(ex["target_delta_raw"]) == 0)
            op_support["severe_regression_examples"] += int(summary["w2l"] > 0 or summary["reg3"] > 0)
            op_support["harmful_examples"] += int(
                int(ex["target_delta_raw"]) < 0 or summary["w2l"] > 0 or summary["reg3"] > 0
            )

    arrays = {
        "pair_parent_margin": pair_parent_margin,
        "pair_child_margin": pair_child_margin,
        "pair_case_delta_raw": pair_case_delta_raw,
        "pair_case_delta_norm": pair_case_delta_norm,
        "pair_case_mask": pair_case_mask,
        "pair_case_improve": pair_case_improve,
        "pair_case_flip": pair_case_flip,
        "pair_case_regress": pair_case_regress,
        "pair_case_reg3": pair_case_reg3,
        "transition_idx": np.asarray(ex_transition, np.int32),
        "goal_case_id": np.asarray(ex_goal_case, np.int16),
        "goal_class": np.asarray(ex_goal_class, np.int8),
        "target_delta_raw": np.asarray(ex_target_delta_raw, np.int32),
        "target_delta_norm": np.asarray(ex_target_delta_norm, np.float32),
        "target_improve": np.asarray(ex_target_improve, bool),
        "target_flip": np.asarray(ex_target_flip, bool),
        "positive": np.asarray(ex_positive, np.float32),
        "weight": np.asarray(ex_weight, np.float32),
        "advantage": np.asarray(ex_advantage, np.float32),
        "w2l": np.asarray(ex_w2l, np.int16),
        "reg3": np.asarray(ex_reg3, np.int16),
        "goal_mode": np.zeros(len(ex_transition), np.int8),  # 0=LOSS_REPAIR; 1 reserved for MARGIN_EXPANSION.
        "desired_margin_floor": np.zeros(len(ex_transition), np.float32),
    }
    return arrays, support


def prepare_actor_dataset(out: Path | None = None, *, limit_rows: int | None = None, max_transitions: int = 60_000,
                          min_safe_positive: int = 32, awr_beta: float = 2.0,
                          max_positive_weight: float = 12.0, max_negative_weight: float = 8.0) -> dict:
    out = Path(out or ACTOR_DATA_DIR / ("transitions_full.npz" if limit_rows is None else f"transitions_smoke{limit_rows}.npz"))
    out.parent.mkdir(parents=True, exist_ok=True)
    transitions, genomes, rows = build_transitions(limit_rows, max_transitions)
    arrays = _action_arrays(transitions, genomes)
    goal_arrays, operation_support = goal_supervision_arrays(
        transitions, rows, case_cap=48, awr_beta=awr_beta,
        max_positive_weight=max_positive_weight, max_negative_weight=max_negative_weight,
    )
    arrays.update(goal_arrays)
    transition_train, transition_val = split_transition_indices(transitions)
    transition_side = np.zeros(len(transitions), np.int8)
    transition_side[transition_train] = 1
    transition_side[transition_val] = 2
    ex_side = transition_side[arrays["transition_idx"]]
    train = np.flatnonzero(ex_side == 1).astype(np.int32)
    val = np.flatnonzero(ex_side == 2).astype(np.int32)
    assert_no_relabel_leakage(arrays["transition_idx"], train, val)
    arrays["transition_train_idx"] = transition_train
    arrays["transition_val_idx"] = transition_val
    arrays["train_idx"] = train
    arrays["val_idx"] = val
    learned_mask = operation_support_mask(operation_support, min_safe_positive=min_safe_positive)
    arrays["learned_operation_mask"] = learned_mask
    np.savez(out, **arrays)
    ops = {name: 0 for name in EDIT_OPERATIONS}
    positive_ops = {name: 0 for name in EDIT_OPERATIONS}
    for t in transitions:
        ops[t.action.operation] += 1
        positive_ops[t.action.operation] += int(t.positive)
    goal_counts = {name: int(np.sum(arrays["goal_class"] == int(cls))) for name, cls in (
        ("safe_flip", GoalClass.SAFE_FLIP), ("safe_improve", GoalClass.SAFE_IMPROVE),
        ("unsafe_improve", GoalClass.UNSAFE_IMPROVE), ("no_effect", GoalClass.NO_EFFECT),
        ("target_harm", GoalClass.TARGET_HARM), ("severe_regression", GoalClass.SEVERE_REGRESSION),
    )}
    sha = hashlib.sha256(out.read_bytes()).hexdigest()
    manifest = {
        "schema": 2, "architecture": "BonsaiEditFormer-GCAC-v2", "dataset": str(out), "dataset_sha256": sha,
        "rows": len(rows), "transitions": len(transitions), "goal_examples": len(arrays["transition_idx"]),
        "train": len(train), "val": len(val),
        "positive": int(np.sum(arrays["positive"] > 0)), "negative": int(np.sum(arrays["positive"] <= 0)),
        "goal_classes": goal_counts,
        "operations": ops, "legacy_positive_operations": positive_ops,
        "operation_support": operation_support,
        "min_safe_positive_support": int(min_safe_positive),
        "learned_operation_mask": {name: bool(learned_mask[i]) for i, name in enumerate(EDIT_OPERATIONS)},
        "awr": {"beta": float(awr_beta), "max_positive_weight": float(max_positive_weight), "max_negative_weight": float(max_negative_weight)},
        "transition_train_hash": hashlib.sha256(transition_train.tobytes()).hexdigest(),
        "transition_val_hash": hashlib.sha256(transition_val.tobytes()).hexdigest(),
        "train_hash": hashlib.sha256(train.tobytes()).hexdigest(), "val_hash": hashlib.sha256(val.tobytes()).hexdigest(),
        "relabel_leakage": 0,
        "case_delta_normalization": "tanh(raw_margin_delta / 5000)",
        "source": "historical exact-CUDA ModuleGenome parent-child transitions with hindsight failure-goal relabeling",
    }
    out.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def load_actor_dataset(path: str | Path) -> dict[str, np.ndarray]:
    return dict(np.load(Path(path), allow_pickle=False))
