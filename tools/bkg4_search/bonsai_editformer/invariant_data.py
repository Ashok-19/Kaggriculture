from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from tools.bkg4_search.invariant_module import (
    AbstractAction,
    GuardAtom,
    InvariantRule,
    InvariantStrategicModule,
    LiftedGuard,
    ModuleSketch,
    bkg,
)
from tools.bkg4_search.invariant_schema import (
    module_from_model_input,
    validate_corpus_records,
    validate_specialist_corpus,
)
from tools.bkg4_search.invariant_search import structural_skeleton


B4IM_TOKEN_CAP = 19  # root + 3 * (rule + 5 entry/exit guards)
B4IM_CAT_COLS = 8
B4IM_NUM_COLS = 10
B4IM_LABEL_NAMES = (
    "regression", "flip", "margin_target", "worst_target", "rank_target",
    "safe", "l2w", "w2l", "mean_delta",
)

_DOMAIN = {"GLOBAL": 1, "EACH_OWN_UNIT": 2}
_DOMAIN_NAME = {value: key for key, value in _DOMAIN.items()}
_CONTROL = {"INSTANT": 1, "COMMITTED": 2}
_CONTROL_NAME = {value: key for key, value in _CONTROL.items()}
_EFFECTS = (
    "NONE", "ROUTE_MODE", "PRESERVE_PRODUCTION_SAVE_FEED", "CREATE_FINANCING_RUNWAY",
    "ACCELERATE_EXPANSION", "REDUCE_MAINTENANCE_BACKLOG", "RECOVER_RESOURCE_INVARIANT",
    "AVOID_OVERBUY", "ENGINE_MECHANIC",
)
_EFFECT = {name: index + 1 for index, name in enumerate(_EFFECTS)}
_EFFECT_NAME = {value: key for key, value in _EFFECT.items()}
_EFFECT_DOMAIN = {
    "NONE": "generic", "ROUTE_MODE": "route",
    "PRESERVE_PRODUCTION_SAVE_FEED": "animal_lifecycle",
    "CREATE_FINANCING_RUNWAY": "market", "ACCELERATE_EXPANSION": "expansion",
    "REDUCE_MAINTENANCE_BACKLOG": "maintenance",
    "RECOVER_RESOURCE_INVARIANT": "recovery", "AVOID_OVERBUY": "market",
    "ENGINE_MECHANIC": "engine_mechanic",
}
_ACTION_NAMES = ("NOOP", "ROUTE_OVERRIDE", "UNIT_REWRITE", "MARKET_INTENT", "COMMITTED_ROUTE")
_ACTION = {name: index + 1 for index, name in enumerate(_ACTION_NAMES)}
_ACTION_NAME = {value: key for key, value in _ACTION.items()}
_CMP_NAMES = ("EQ", "NE", "LT", "LE", "GT", "GE")
_CMP = {name: index + 1 for index, name in enumerate(_CMP_NAMES)}
_CMP_NAME = {value: key for key, value in _CMP.items()}
_FEATURE = {name: int(code) + 1 for name, code in bkg.FEATURES.items()}
_FEATURE_NAME = {value: key for key, value in _FEATURE.items()}


@dataclass(frozen=True)
class InvariantArraySplit:
    train_indices: tuple[int, ...]
    validation_indices: tuple[int, ...]
    train_skeletons: tuple[str, ...]
    validation_skeletons: tuple[str, ...]


def invariant_module_arrays(module: InvariantStrategicModule) -> dict[str, np.ndarray]:
    """Encode a canonical B4IM AST into fixed-shape, model-ready integer arrays."""
    module = module.canonicalized()
    cat = np.zeros((B4IM_TOKEN_CAP, B4IM_CAT_COLS), dtype=np.int16)
    num = np.zeros((B4IM_TOKEN_CAP, B4IM_NUM_COLS), dtype=np.int32)
    mask = np.zeros(B4IM_TOKEN_CAP, dtype=bool)
    parent = np.full(B4IM_TOKEN_CAP, -1, dtype=np.int16)
    depth = np.zeros(B4IM_TOKEN_CAP, dtype=np.int8)
    rule_ids = np.zeros(B4IM_TOKEN_CAP, dtype=np.int8)
    cat[0, 0] = 1
    num[0, 0] = len(module.rules)
    mask[0] = True
    pos = 1
    for rule_index, rule in enumerate(module.rules, 1):
        rule_pos = pos
        cat[pos] = (
            2, rule_index, _DOMAIN[rule.domain], _CONTROL[rule.sketch.control_pattern],
            _EFFECT[rule.sketch.effect_class], _ACTION[rule.action.kind], 0, 0,
        )
        action = rule.action
        num[pos] = (
            action.route_id, action.match_op, action.unit_op, action.unit_arg, action.quantity,
            action.market_op, action.item, action.authority, action.priority,
            len(rule.guard.atoms) + len(rule.exit_guard.atoms),
        )
        mask[pos] = True
        parent[pos] = 0
        depth[pos] = 1
        rule_ids[pos] = rule_index
        pos += 1
        for phase, atoms in ((1, rule.guard.atoms), (2, rule.exit_guard.atoms)):
            for ordinal, atom in enumerate(atoms, 1):
                cat[pos] = (
                    3, rule_index, phase, ordinal, _FEATURE[atom.feature], _CMP[atom.cmp], 0, 0,
                )
                num[pos, 0] = int(atom.value)
                num[pos, 1] = int(atom.arg)
                mask[pos] = True
                parent[pos] = rule_pos
                depth[pos] = 2
                rule_ids[pos] = rule_index
                pos += 1
    return {
        "b4im_cat": cat,
        "b4im_num": num,
        "b4im_mask": mask,
        "b4im_parent": parent,
        "b4im_depth": depth,
        "b4im_rule": rule_ids,
    }


def module_from_invariant_arrays(
    cat: np.ndarray,
    num: np.ndarray,
    mask: np.ndarray,
) -> InvariantStrategicModule:
    cat = np.asarray(cat)
    num = np.asarray(num)
    mask = np.asarray(mask)
    if cat.shape != (B4IM_TOKEN_CAP, B4IM_CAT_COLS):
        raise ValueError("invalid B4IM categorical array shape")
    if num.shape != (B4IM_TOKEN_CAP, B4IM_NUM_COLS) or mask.shape != (B4IM_TOKEN_CAP,):
        raise ValueError("invalid B4IM numeric/mask array shape")
    if cat.dtype.kind not in "iu" or num.dtype.kind not in "iu" or mask.dtype.kind != "b":
        raise ValueError("B4IM model arrays require integer categories/numerics and a boolean mask")
    active = np.flatnonzero(mask)
    if not len(active) or not np.array_equal(active, np.arange(len(active))):
        raise ValueError("B4IM token mask must be a non-empty contiguous prefix")
    if np.any(cat[len(active):]) or np.any(num[len(active):]):
        raise ValueError("B4IM padded model array payload must be zero")
    if tuple(int(x) for x in cat[0]) != (1, 0, 0, 0, 0, 0, 0, 0):
        raise ValueError("invalid B4IM model root token")
    rule_count = int(num[0, 0])
    if np.any(num[0, 1:]) or not 0 <= rule_count <= 3:
        raise ValueError("invalid B4IM model root payload")
    rules: list[InvariantRule] = []
    pos = 1
    for rule_index in range(1, rule_count + 1):
        row = tuple(int(x) for x in cat[pos])
        if row[0] != 2 or row[1] != rule_index or any(row[6:]):
            raise ValueError("invalid B4IM model rule token")
        action = AbstractAction(
            _ACTION_NAME[row[5]], route_id=int(num[pos, 0]), match_op=int(num[pos, 1]),
            unit_op=int(num[pos, 2]), unit_arg=int(num[pos, 3]), quantity=int(num[pos, 4]),
            market_op=int(num[pos, 5]), item=int(num[pos, 6]), authority=int(num[pos, 7]),
            priority=int(num[pos, 8]),
        )
        guard_count = int(num[pos, 9])
        pos += 1
        entry: list[GuardAtom] = []
        exit_: list[GuardAtom] = []
        for guard_ordinal in range(guard_count):
            if pos >= len(active):
                raise ValueError("truncated B4IM model guard tokens")
            grow = tuple(int(x) for x in cat[pos])
            if grow[0] != 3 or grow[1] != rule_index or grow[2] not in (1, 2) or any(grow[6:]):
                raise ValueError("invalid B4IM model guard token")
            target = entry if grow[2] == 1 else exit_
            if grow[3] != len(target) + 1 or grow[4] not in _FEATURE_NAME or grow[5] not in _CMP_NAME:
                raise ValueError("noncanonical B4IM model guard order/category")
            if np.any(num[pos, 2:]):
                raise ValueError("noncanonical B4IM model guard numeric payload")
            target.append(GuardAtom(_FEATURE_NAME[grow[4]], _CMP_NAME[grow[5]], int(num[pos, 0]), int(num[pos, 1])))
            pos += 1
        effect = _EFFECT_NAME[row[4]]
        rules.append(InvariantRule(
            ModuleSketch(_EFFECT_DOMAIN[effect], _CONTROL_NAME[row[3]], effect),
            _DOMAIN_NAME[row[2]], LiftedGuard(tuple(entry)), action, LiftedGuard(tuple(exit_)),
        ))
    if pos != len(active):
        raise ValueError("trailing B4IM model tokens")
    module = InvariantStrategicModule(tuple(rules))
    module.validate()
    canonical = invariant_module_arrays(module)
    if not (
        np.array_equal(cat, canonical["b4im_cat"])
        and np.array_equal(num, canonical["b4im_num"])
        and np.array_equal(mask, canonical["b4im_mask"])
    ):
        raise ValueError("noncanonical B4IM model arrays")
    return module


def _validate_model_corpus(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    specialist = [
        isinstance(record.get("target_labels"), Mapping)
        and record["target_labels"].get("exact_data_role") == "specialist_discovery"
        for record in records
    ]
    if any(specialist):
        if not all(specialist):
            raise ValueError("model corpus cannot mix specialist and non-specialist validation contracts")
        return validate_specialist_corpus(records)
    return validate_corpus_records(records)


def invariant_corpus_arrays(records: Sequence[Mapping[str, Any]]) -> dict[str, np.ndarray]:
    """Materialize validated invariant corpus rows for batching/ranker integration."""
    _validate_model_corpus(records)
    n = len(records)
    arrays = {
        "b4im_cat": np.zeros((n, B4IM_TOKEN_CAP, B4IM_CAT_COLS), np.int16),
        "b4im_num": np.zeros((n, B4IM_TOKEN_CAP, B4IM_NUM_COLS), np.int32),
        "b4im_mask": np.zeros((n, B4IM_TOKEN_CAP), bool),
        "b4im_parent": np.full((n, B4IM_TOKEN_CAP), -1, np.int16),
        "b4im_depth": np.zeros((n, B4IM_TOKEN_CAP), np.int8),
        "b4im_rule": np.zeros((n, B4IM_TOKEN_CAP), np.int8),
        "labels": np.zeros((n, len(B4IM_LABEL_NAMES)), np.float32),
        "case_features": np.zeros((n, 1, 22), np.float32),
        "case_mask": np.ones((n, 1), bool),
        "case_delta": np.zeros((n, 1), np.float32),
        "heuristic": np.zeros(n, np.float32),
        "group": np.zeros(n, np.int16),
        "semantic_id": np.zeros(n, dtype="S64"),
    }
    for index, record in enumerate(records):
        module = module_from_model_input(record["model_input"])
        encoded = invariant_module_arrays(module)
        for key, value in encoded.items():
            arrays[key][index] = value
        labels = record["target_labels"]
        arrays["labels"][index] = [float(labels.get(name, 0.0)) for name in B4IM_LABEL_NAMES]
        arrays["semantic_id"][index] = module.semantic_id().encode()
    return arrays


def invariant_numeric_batch(cache: Mapping[str, np.ndarray], ids: Sequence[int]) -> dict[str, np.ndarray]:
    indices = np.asarray(ids, dtype=np.int32)
    keys = (
        "b4im_cat", "b4im_num", "b4im_mask", "b4im_parent", "b4im_depth",
        "b4im_rule", "case_features", "case_mask", "case_delta", "heuristic", "labels", "group",
    )
    if any(key not in cache for key in keys):
        raise ValueError("invariant cache is missing required model arrays")
    return {key: np.asarray(cache[key])[indices] for key in keys}


def invariant_grouped_split(
    records: Sequence[Mapping[str, Any]],
    *,
    validation_fraction: float = 0.20,
) -> InvariantArraySplit:
    """Keep constant variants of one structural mechanism on the same side."""
    _validate_model_corpus(records)
    if not 0.10 <= float(validation_fraction) <= 0.50:
        raise ValueError("validation_fraction must be between 0.10 and 0.50")
    buckets = max(2, round(1.0 / float(validation_fraction)))
    groups: dict[str, list[int]] = {}
    for index, record in enumerate(records):
        module = module_from_model_input(record["model_input"])
        key = repr(structural_skeleton(module))
        groups.setdefault(key, []).append(index)
    train: list[int] = []
    validation: list[int] = []
    train_keys: list[str] = []
    validation_keys: list[str] = []
    for key, indices in sorted(groups.items()):
        import hashlib
        held_out = int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) % buckets == 0
        (validation if held_out else train).extend(indices)
        (validation_keys if held_out else train_keys).append(key)
    if not train or not validation:
        raise ValueError("structural split requires at least one train and validation skeleton")
    return InvariantArraySplit(
        tuple(sorted(train)), tuple(sorted(validation)), tuple(train_keys), tuple(validation_keys),
    )
