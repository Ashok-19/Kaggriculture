from __future__ import annotations

import dataclasses
import random
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from tools.bkg4_search.bonsai_editformer.actor_grammar import (
    EDIT_OPERATIONS,
    OP_INDEX,
    EditAction,
    apply_edit,
    node_refs,
    target_masks,
    valid_operation_mask,
)
from tools.bkg4_search.campaign import random_condition
from tools.bkg4_search.module_codec import FEATURES, ConditionGene, ModuleGenome
from tools.edit_corpus.corpus import canonical_edit_identity

DEFAULT_ROUTE_IDS = (22, 23, 29, 32, 34, 46, 63, 64, 70, 77, 79, 211, 212, 213, 214)


@dataclass(frozen=True)
class GeneratedEdit:
    edit_id: str
    parent_semantic_id: str
    child_semantic_id: str
    action: EditAction
    action_payload: dict[str, Any]
    child_genome: dict[str, Any]
    proposal_class: str


def observed_conditions(context_rows: Iterable[Mapping[str, Any]]) -> list[ConditionGene]:
    out: list[ConditionGene] = []
    seen: set[ConditionGene] = set()
    for row in context_rows:
        feature = str(row.get("feature", ""))
        if feature not in FEATURES:
            continue
        try:
            index = int(row.get("index", 0))
            value = int(round(float(row["value"])))
        except (KeyError, TypeError, ValueError, OverflowError):
            continue
        cmps = ("EQ",) if feature in ("SHOP_AT", "SHOP_SEEN") else ("LE", "GE")
        for cmp in cmps:
            try:
                c = ConditionGene(feature, cmp, index, 1 if feature == "SHOP_SEEN" else value)
                c.validate()
            except ValueError:
                continue
            if c not in seen:
                seen.add(c)
                out.append(c)
    return sorted(out, key=lambda c: (c.feature, c.cmp, c.index, c.value))


def _diagnostic_conditions(observed: Sequence[ConditionGene]) -> dict[str, list[ConditionGene]]:
    broad = [
        ConditionGene("DAY", "GE", 0, 0),
        ConditionGene("STEP", "GE", 0, 0),
        ConditionGene("OWN_MONEY", "GE", 0, 0),
    ]
    inert = [
        ConditionGene("OWN_MONEY", "GE", 0, 100_000),
        ConditionGene("MARKET_INVENTORY", "GE", 0, 100_000),
        ConditionGene("STEP", "EQ", 0, 719),
    ]
    return {
        "observed_threshold": list(observed) or [ConditionGene("OWN_MONEY", "LE", 0, 3000)],
        "broad_probe": broad,
        "inert_probe": inert,
    }


def _target_payload(parent: ModuleGenome, action: EditAction) -> dict[str, Any]:
    refs = node_refs(parent)
    ref = refs[action.target_node] if 0 <= action.target_node < len(refs) else None
    payload = dataclasses.asdict(action)
    payload.update({
        "target_kind": ref.kind if ref is not None else "unknown",
        "target_rule_index": int(ref.rule_index) if ref is not None else -1,
        "target_condition_index": int(ref.condition_index) if ref is not None else -1,
    })
    return payload


def _value_for_threshold(parent: ModuleGenome, target: int, cond: ConditionGene, round_i: int) -> int:
    ref = node_refs(parent)[target]
    if ref is None or ref.condition_index < 0:
        return int(cond.value)
    old = parent.rules[ref.rule_index].conditions[ref.condition_index]
    value = int(cond.value) if cond.feature == old.feature else int(old.value) + (1 if round_i % 2 == 0 else -1)
    if value == old.value:
        step = max(1, abs(int(old.value)) // 10 or 1)
        value = int(old.value) + (step if round_i % 2 == 0 else -step)
    return max(-(1 << 31), min((1 << 31) - 1, value))


def _action_for(
    parent: ModuleGenome,
    operation: str,
    target: int,
    c1: ConditionGene,
    c2: ConditionGene,
    route_ids: Sequence[int],
    rng: random.Random,
    round_i: int,
) -> EditAction:
    refs = node_refs(parent)
    ref = refs[target]
    route = int(route_ids[(round_i + target) % len(route_ids)])
    priority = round_i % max(1, len(parent.rules))
    prefix_step = (parent.prefix_step + 1 + (round_i * 37) % 719) % 720
    if prefix_step == parent.prefix_step:
        prefix_step = (prefix_step + 1) % 720

    if operation == "CHANGE_THRESHOLD":
        assert ref is not None and ref.condition_index >= 0
        old = parent.rules[ref.rule_index].conditions[ref.condition_index]
        return EditAction(
            operation=operation,
            target_node=target,
            feature=old.feature,
            cmp=old.cmp,
            index=old.index,
            value=_value_for_threshold(parent, target, c1, round_i),
            route=parent.rules[ref.rule_index].route,
        )
    if operation in ("CHANGE_ROUTE", "REPLACE_MACRO"):
        assert ref is not None and ref.rule_index >= 0
        old_route = parent.rules[ref.rule_index].route
        choices = [int(x) for x in route_ids if int(x) != int(old_route)]
        route = choices[(round_i + target) % len(choices)]
    if operation == "CHANGE_PRIORITY":
        assert ref is not None and ref.rule_index >= 0
        if len(parent.rules) > 1:
            priority = (ref.rule_index + 1 + round_i) % len(parent.rules)
            if priority == ref.rule_index:
                priority = (priority + 1) % len(parent.rules)
    if operation == "CHANGE_WINDOW":
        target = 0

    return EditAction(
        operation=operation,
        target_node=target,
        feature=c1.feature,
        cmp=c1.cmp,
        index=c1.index,
        value=c1.value,
        route=route,
        feature2=c2.feature,
        cmp2=c2.cmp,
        index2=c2.index,
        value2=c2.value,
        condition_count=1 + (round_i % 2),
        priority=priority,
        prefix_step=prefix_step,
    )


def generate_typed_candidates(
    parent: ModuleGenome,
    context_rows: Iterable[Mapping[str, Any]],
    *,
    n: int,
    seed: int,
    route_ids: Sequence[int] = DEFAULT_ROUTE_IDS,
) -> list[GeneratedEdit]:
    if n <= 0:
        return []
    route_ids = tuple(int(x) for x in route_ids)
    if not route_ids:
        raise ValueError("route_ids must not be empty")
    parent.validate(route_ids)
    rng = random.Random(int(seed))
    observed = observed_conditions(context_rows)
    diagnostics = _diagnostic_conditions(observed)
    proposal_classes = ("observed_threshold", "broad_probe", "inert_probe", "domain_random")
    op_valid = valid_operation_mask(parent)
    masks = target_masks(parent)
    valid_ops = [name for name in EDIT_OPERATIONS if op_valid[OP_INDEX[name]]]
    if not valid_ops:
        raise ValueError("parent has no legal typed edit operations")

    seen_children = {parent.semantic_id()}
    seen_edits: set[str] = set()
    out: list[GeneratedEdit] = []
    max_attempts = max(n * 80, len(valid_ops) * 16)
    for attempt in range(max_attempts):
        if len(out) >= n:
            break
        operation = valid_ops[attempt % len(valid_ops)]
        oi = OP_INDEX[operation]
        targets = [i for i, ok in enumerate(masks[oi]) if ok and node_refs(parent)[i] is not None]
        if not targets:
            continue
        target = targets[(attempt // len(valid_ops) + oi) % len(targets)]
        proposal_class = proposal_classes[(attempt // max(1, len(valid_ops))) % len(proposal_classes)]
        if proposal_class == "domain_random":
            c1 = random_condition(rng)
            c2 = random_condition(rng)
        else:
            pool = diagnostics[proposal_class]
            c1 = pool[(attempt + target) % len(pool)]
            c2 = pool[(attempt * 3 + target + 1) % len(pool)]
        action = _action_for(parent, operation, target, c1, c2, route_ids, rng, attempt)
        try:
            child = apply_edit(parent, action, route_ids)
        except ValueError:
            continue
        child_id = child.semantic_id()
        if child_id in seen_children:
            continue
        payload = _target_payload(parent, action)
        edit_id = canonical_edit_identity(parent.semantic_id(), payload, child_id)
        if edit_id in seen_edits:
            continue
        seen_children.add(child_id)
        seen_edits.add(edit_id)
        out.append(GeneratedEdit(
            edit_id=edit_id,
            parent_semantic_id=parent.semantic_id(),
            child_semantic_id=child_id,
            action=action,
            action_payload=payload,
            child_genome=child.canonical(),
            proposal_class=proposal_class,
        ))
    return out
