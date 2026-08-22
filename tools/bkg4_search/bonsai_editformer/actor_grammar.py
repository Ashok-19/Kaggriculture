from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from tools.bkg4_search.module_codec import (
    COMPARATORS,
    FEATURES,
    ITEM_INDEX_FEATURES,
    MAX_CONDS,
    MAX_RULES,
    SHOP_INDEX_FEATURES,
    TILE_INDEX_FEATURES,
    UNIT_INDEX_FEATURES,
    ConditionGene,
    ModuleGenome,
    RuleGene,
)

EDIT_OPERATIONS = (
    "ADD_SPECIALIST",
    "ADD_CONDITION",
    "CHANGE_THRESHOLD",
    "CHANGE_ROUTE",
    "CHANGE_PRIORITY",
    "CHANGE_WINDOW",
    "INSERT_RECOVERY",
    "REPLACE_MACRO",
    "SPLIT_LANE",
    "COMPOSE_PREDICATES",
    "REPLACE_SUBTREE",
    "REMOVE_REDUNDANT_BRANCH",
)
OP_INDEX = {name: i for i, name in enumerate(EDIT_OPERATIONS)}
N_OPS = len(EDIT_OPERATIONS)

TARGET_ROOT = "root"
TARGET_RULE = "rule"
TARGET_CONDITION = "condition"

OP_TARGET_KIND = {
    "ADD_SPECIALIST": (TARGET_ROOT, TARGET_RULE),
    "ADD_CONDITION": (TARGET_RULE,),
    "CHANGE_THRESHOLD": (TARGET_CONDITION,),
    "CHANGE_ROUTE": (TARGET_RULE,),
    "CHANGE_PRIORITY": (TARGET_RULE,),
    "CHANGE_WINDOW": (TARGET_ROOT,),
    "INSERT_RECOVERY": (TARGET_ROOT, TARGET_RULE),
    "REPLACE_MACRO": (TARGET_RULE,),
    "SPLIT_LANE": (TARGET_RULE,),
    "COMPOSE_PREDICATES": (TARGET_RULE,),
    "REPLACE_SUBTREE": (TARGET_RULE,),
    "REMOVE_REDUNDANT_BRANCH": (TARGET_RULE,),
}

FEATURE_INDEX = {name: i for i, name in enumerate(FEATURES)}
CMP_INDEX = {name: i for i, name in enumerate(COMPARATORS)}


@dataclass(frozen=True)
class TargetRef:
    node_index: int
    kind: str
    rule_index: int = -1
    condition_index: int = -1


@dataclass(frozen=True)
class EditAction:
    operation: str
    target_node: int
    feature: str = "STEP"
    cmp: str = "GE"
    index: int = 0
    value: int = 0
    route: int = 0
    feature2: str = "STEP"
    cmp2: str = "GE"
    index2: int = 0
    value2: int = 0
    condition_count: int = 1
    priority: int = 0
    prefix_step: int = 0


def node_refs(genome: ModuleGenome, tree_cap: int = 48) -> list[TargetRef | None]:
    refs: list[TargetRef | None] = [None] * tree_cap
    refs[0] = TargetRef(0, TARGET_ROOT)
    pos = 3
    for ri, rule in enumerate(genome.rules):
        if pos >= tree_cap:
            break
        refs[pos] = TargetRef(pos, TARGET_RULE, ri, -1)
        pos += 1
        for ci, _ in enumerate(rule.conditions):
            if pos >= tree_cap:
                break
            refs[pos] = TargetRef(pos, TARGET_CONDITION, ri, ci)
            pos += 1
    return refs


def target_masks(genome: ModuleGenome, tree_cap: int = 48) -> list[list[bool]]:
    refs = node_refs(genome, tree_cap)
    out = [[False] * tree_cap for _ in range(N_OPS)]
    for op_name, oi in OP_INDEX.items():
        for i, ref in enumerate(refs):
            if ref is None or ref.kind not in OP_TARGET_KIND[op_name]:
                continue
            allow = True
            if op_name in ("ADD_SPECIALIST", "INSERT_RECOVERY"):
                allow = len(genome.rules) < MAX_RULES
            elif op_name in ("ADD_CONDITION", "COMPOSE_PREDICATES"):
                allow = ref.kind == TARGET_RULE and len(genome.rules[ref.rule_index].conditions) < MAX_CONDS
            elif op_name in ("CHANGE_PRIORITY", "REMOVE_REDUNDANT_BRANCH"):
                allow = ref.kind == TARGET_RULE and len(genome.rules) >= 1
            elif op_name == "SPLIT_LANE":
                allow = ref.kind == TARGET_RULE and len(genome.rules) < MAX_RULES and len(genome.rules[ref.rule_index].conditions) < MAX_CONDS
            out[oi][i] = bool(allow)
        if not any(out[oi]):
            # Keep logits finite even for impossible ops; the op sampler masks them separately.
            out[oi][0] = True
    return out


def valid_operation_mask(genome: ModuleGenome) -> list[bool]:
    masks = target_masks(genome)
    valid = []
    for op_name, oi in OP_INDEX.items():
        real = any(masks[oi][i] for i, ref in enumerate(node_refs(genome)) if ref is not None and (i != 0 or ref.kind == TARGET_ROOT))
        if op_name in ("ADD_SPECIALIST", "INSERT_RECOVERY"):
            real = len(genome.rules) < MAX_RULES
        elif op_name in ("ADD_CONDITION", "COMPOSE_PREDICATES", "CHANGE_ROUTE", "REPLACE_MACRO", "CHANGE_PRIORITY", "REPLACE_SUBTREE", "REMOVE_REDUNDANT_BRANCH", "SPLIT_LANE"):
            real = bool(genome.rules) and real
        elif op_name == "CHANGE_THRESHOLD":
            real = any(r.conditions for r in genome.rules)
        valid.append(bool(real))
    return valid


def legal_index(feature: str, index: int) -> int:
    index = int(index)
    if feature == "OWN_SEED":
        return max(0, min(4, index))
    if feature in ITEM_INDEX_FEATURES:
        return max(0, min(11, index))
    if feature in SHOP_INDEX_FEATURES:
        return max(0, min(7, index))
    if feature in UNIT_INDEX_FEATURES:
        return max(0, min(39, index))
    if feature in TILE_INDEX_FEATURES:
        return max(0, min(99, index))
    return 0


def value_bounds(feature: str) -> tuple[int, int]:
    if feature == "STEP": return 0, 719
    if feature == "DAY": return 0, 30
    if feature == "HOUR": return 0, 23
    if feature == "SHOP_AT": return 0, 7
    if feature == "SHOP_SEEN": return 0, 1
    if feature in ("OWN_MONEY", "OPP_MONEY"): return 0, 100_000
    if feature in ("OWN_UNITS", "OPP_UNITS", "OWN_HIRES_TODAY", "OPP_HIRES_TODAY"): return 0, 40
    if feature in ("OWN_LAND", "OPP_LAND"): return 0, 100
    if feature == "MARKET_PRICE": return 0, 100_000
    if feature == "MARKET_INVENTORY": return 0, 100_000
    if feature == "OWN_SHED": return 0, 100_000
    if feature == "OWN_SEED": return 0, 100_000
    if feature in UNIT_INDEX_FEATURES: return 0, 9
    if feature in TILE_INDEX_FEATURES: return 0, 100_000
    return -100_000, 100_000


def normalize_value(feature: str, value: int) -> float:
    lo, hi = value_bounds(feature)
    value = max(lo, min(hi, int(value)))
    if lo >= 0 and hi > 1000:
        return float(math.log1p(value) / math.log1p(hi) * 2.0 - 1.0)
    return float(2.0 * (value - lo) / max(1, hi - lo) - 1.0)


def denormalize_value(feature: str, value_norm: float) -> int:
    lo, hi = value_bounds(feature)
    x = max(-1.0, min(1.0, float(value_norm)))
    if lo >= 0 and hi > 1000:
        v = math.expm1((x + 1.0) * 0.5 * math.log1p(hi))
    else:
        v = lo + (x + 1.0) * 0.5 * (hi - lo)
    return int(round(max(lo, min(hi, v))))


def _condition(feature: str, cmp: str, index: int, value: int) -> ConditionGene:
    # Canonical semantic inputs are validated, never clipped/normalized. Neural adapters
    # may normalize numeric tensors before constructing an EditAction, but apply_edit is
    # the authoritative typed boundary and must reject malformed actions.
    if feature == "SHOP_SEEN" and int(value) != 1:
        raise ValueError("SHOP_SEEN canonical predicate value must be 1")
    c = ConditionGene(str(feature), str(cmp), int(index), int(value))
    c.validate()
    return c


def _new_conditions(a: EditAction) -> tuple[ConditionGene, ...]:
    count = int(a.condition_count)
    if not 1 <= count <= 2:
        raise ValueError("edit condition_count must be 1..2")
    out = [_condition(a.feature, a.cmp, a.index, a.value)]
    if count == 2:
        c2 = _condition(a.feature2, a.cmp2, a.index2, a.value2)
        if c2 != out[0]:
            out.append(c2)
    return tuple(out)


def apply_edit(parent: ModuleGenome, action: EditAction, route_ids: Sequence[int]) -> ModuleGenome:
    if action.operation not in OP_INDEX:
        raise ValueError(f"unknown edit operation {action.operation}")
    refs = node_refs(parent)
    if not 0 <= action.target_node < len(refs) or refs[action.target_node] is None:
        raise ValueError("invalid editor target node")
    ref = refs[action.target_node]
    assert ref is not None
    if ref.kind not in OP_TARGET_KIND[action.operation]:
        raise ValueError(f"target {ref.kind} invalid for {action.operation}")

    rules = list(parent.rules)
    op = action.operation
    route = int(action.route)
    if route not in route_ids and op in ("ADD_SPECIALIST", "INSERT_RECOVERY", "SPLIT_LANE", "CHANGE_ROUTE", "REPLACE_MACRO", "REPLACE_SUBTREE"):
        raise ValueError(f"route {route} is not legal")

    if op in ("ADD_SPECIALIST", "INSERT_RECOVERY"):
        if len(rules) >= MAX_RULES:
            raise ValueError("rule budget exhausted")
        pos = len(rules) if ref.kind == TARGET_ROOT else min(len(rules), ref.rule_index + 1)
        if op == "INSERT_RECOVERY":
            pos = 0
        rules.insert(pos, RuleGene(_new_conditions(action), route))
    elif op in ("ADD_CONDITION", "COMPOSE_PREDICATES"):
        ri = ref.rule_index; rule = rules[ri]; conds = list(rule.conditions)
        additions = list(_new_conditions(action))
        room = MAX_CONDS - len(conds)
        for c in additions[:room]:
            if c not in conds:
                conds.append(c)
        if tuple(conds) == rule.conditions:
            raise ValueError("edit did not add a condition")
        rules[ri] = RuleGene(tuple(conds), rule.route)
    elif op == "CHANGE_THRESHOLD":
        ri, ci = ref.rule_index, ref.condition_index; rule = rules[ri]; conds = list(rule.conditions); old = conds[ci]
        if (str(action.feature), str(action.cmp), int(action.index)) != (old.feature, old.cmp, old.index):
            raise ValueError("threshold edit predicate identity must match its target")
        conds[ci] = _condition(old.feature, old.cmp, old.index, action.value)
        if conds[ci] == old:
            raise ValueError("threshold edit is a no-op")
        rules[ri] = RuleGene(tuple(conds), rule.route)
    elif op in ("CHANGE_ROUTE", "REPLACE_MACRO"):
        ri = ref.rule_index; rule = rules[ri]
        if route == rule.route:
            raise ValueError("route edit is a no-op")
        rules[ri] = RuleGene(rule.conditions, route)
    elif op == "CHANGE_PRIORITY":
        ri = ref.rule_index; new_pos = int(action.priority)
        if not 0 <= new_pos < len(rules):
            raise ValueError("priority target out of range")
        if new_pos == ri:
            raise ValueError("priority edit is a no-op")
        rule = rules.pop(ri); rules.insert(new_pos, rule)
    elif op == "CHANGE_WINDOW":
        prefix = int(action.prefix_step)
        if not 0 <= prefix < 720:
            raise ValueError("prefix_step out of range")
        if prefix == parent.prefix_step:
            raise ValueError("window edit is a no-op")
        out = ModuleGenome(prefix, tuple(rules)); out.validate(route_ids); return out
    elif op == "SPLIT_LANE":
        if len(rules) >= MAX_RULES:
            raise ValueError("rule budget exhausted")
        ri = ref.rule_index; base = rules[ri]; conds = list(base.conditions)
        for c in _new_conditions(action):
            if c not in conds and len(conds) < MAX_CONDS:
                conds.append(c)
        if tuple(conds) == base.conditions:
            raise ValueError("split did not specialize the branch")
        rules.insert(ri, RuleGene(tuple(conds), route))
    elif op == "REPLACE_SUBTREE":
        rules[ref.rule_index] = RuleGene(_new_conditions(action), route)
    elif op == "REMOVE_REDUNDANT_BRANCH":
        del rules[ref.rule_index]
    else:
        raise AssertionError(op)

    out = ModuleGenome(parent.prefix_step, tuple(rules))
    out.validate(route_ids)
    if out.semantic_id() == parent.semantic_id():
        raise ValueError("editor produced semantic no-op")
    return out
