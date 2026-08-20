from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import asdict, dataclass
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
        conds = []
        for fid, cmp_code, index, value in rr["conds"]:
            conds.append(ConditionGene(FEATURE_NAME[fid], CMP_NAME[cmp_code], int(index), int(value)))
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
        "positive": np.zeros(n, np.float32), "weight": np.zeros(n, np.float32), "advantage": np.zeros(n, np.float32),
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
        a["positive"][i] = float(t.positive); a["weight"][i] = t.weight; a["advantage"][i] = t.advantage
    return a


def prepare_actor_dataset(out: Path | None = None, *, limit_rows: int | None = None, max_transitions: int = 60_000) -> dict:
    out = Path(out or ACTOR_DATA_DIR / ("transitions_full.npz" if limit_rows is None else f"transitions_smoke{limit_rows}.npz"))
    out.parent.mkdir(parents=True, exist_ok=True)
    transitions, genomes, rows = build_transitions(limit_rows, max_transitions)
    arrays = _action_arrays(transitions, genomes)
    train, val = split_transition_indices(transitions)
    arrays["train_idx"] = train; arrays["val_idx"] = val
    np.savez(out, **arrays)
    ops = {name: 0 for name in EDIT_OPERATIONS}
    positive_ops = {name: 0 for name in EDIT_OPERATIONS}
    for t in transitions:
        ops[t.action.operation] += 1
        positive_ops[t.action.operation] += int(t.positive)
    sha = hashlib.sha256(out.read_bytes()).hexdigest()
    manifest = {
        "schema": 1, "dataset": str(out), "dataset_sha256": sha,
        "rows": len(rows), "transitions": len(transitions), "train": len(train), "val": len(val),
        "positive": sum(int(t.positive) for t in transitions), "negative": sum(int(not t.positive) for t in transitions),
        "operations": ops, "positive_operations": positive_ops,
        "train_hash": hashlib.sha256(train.tobytes()).hexdigest(), "val_hash": hashlib.sha256(val.tobytes()).hexdigest(),
        "source": "historical exact-CUDA ModuleGenome parent-child transitions",
    }
    out.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def load_actor_dataset(path: str | Path) -> dict[str, np.ndarray]:
    return dict(np.load(Path(path), allow_pickle=False))
