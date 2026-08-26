"""Structural canonicalization for current RLRMB candidate inputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .schema import stable_identity, validate_runtime_identity_safe

CANDIDATE_CANONICAL_VERSION = 1

_CANDIDATE_PROVENANCE_FIELDS = frozenset(
    {
        "artifact_directory",
        "attempt_number",
        "candidate_id",
        "donor_index",
        "donor_path",
        "donor_route_id",
        "source_path",
    }
)
_UNIT_NOARG_OPS = frozenset(
    {
        "PASS",
        "NORTH",
        "SOUTH",
        "EAST",
        "WEST",
        "DROP",
        "WATER",
        "HARVEST",
        "FERTILIZE",
        "DIG",
        "BUILD_COOP",
        "BUILD_PASTURE",
        "FEED",
        "COLLECT_FERTILIZER",
        "CARE",
    }
)
_UNIT_ITEM_OPS = frozenset({"PLANT"})
_UNIT_ITEM_COUNT_OPS = frozenset({"PICKUP", "PLACE"})
_MARKET_ATOMIC_OPS = frozenset({"HIRE", "BUY_LAND"})
_MARKET_COUNT_OPS = frozenset({"BUY_SEED", "BUY_PRODUCT", "BUY_ANIMAL", "SELL"})


def _exact_int(value: Any, path: str) -> int:
    if type(value) is not int:
        raise TypeError(f"non-canonical integer at {path}: {value!r}")
    return value


def _canonical_unit_action(value: Any, path: str) -> list[Any]:
    if not isinstance(value, (list, tuple)) or not value or type(value[0]) is not str:
        return ["PASS"]
    op = value[0]
    if op in _UNIT_NOARG_OPS:
        return [op]
    if op in _UNIT_ITEM_OPS:
        if len(value) < 2 or type(value[1]) is not str:
            return ["PASS"]
        return [op, value[1]]
    if op in _UNIT_ITEM_COUNT_OPS:
        if len(value) < 2 or type(value[1]) is not str:
            return ["PASS"]
        count = 1 if len(value) < 3 else _exact_int(value[2], f"{path}[2]")
        return [op, value[1], count]
    return ["PASS"]


def _canonical_market_order(value: Any, path: str) -> list[Any]:
    if not isinstance(value, (list, tuple)) or not value or type(value[0]) is not str:
        return []
    op = value[0]
    if op in _MARKET_ATOMIC_OPS:
        return [op]
    if op not in _MARKET_COUNT_OPS or len(value) < 3 or type(value[1]) is not str:
        return []
    count = _exact_int(value[2], f"{path}[2]")
    if count <= 0:
        return []
    return [op, value[1], count]


def canonical_action(action: Any) -> dict[str, Any]:
    """Mirror proven official action parsing while preserving unit/order positions."""
    if not isinstance(action, Mapping):
        return {"farmer": ["PASS"], "hands": [], "market": []}
    farmer = _canonical_unit_action(action.get("farmer"), "action.farmer")
    hands_value = action.get("hands")
    market_value = action.get("market")
    hands = (
        [_canonical_unit_action(value, f"action.hands[{index}]") for index, value in enumerate(hands_value)]
        if isinstance(hands_value, list)
        else []
    )
    market = (
        [_canonical_market_order(value, f"action.market[{index}]") for index, value in enumerate(market_value)]
        if isinstance(market_value, list)
        else []
    )
    return {"farmer": farmer, "hands": hands, "market": market}


def canonical_action_sequence(actions: Sequence[Any]) -> list[dict[str, Any]]:
    if not isinstance(actions, (list, tuple)):
        raise TypeError("action sequence must be a list or tuple")
    return [canonical_action(action) for action in actions]


def action_sequence_semantic_id(actions: Sequence[Any]) -> str:
    return stable_identity("RLRMB-ACTION-SEQUENCE-V1", canonical_action_sequence(actions))


def action_fragment_content_id(actions: Sequence[Any], start: int, duration: int) -> str:
    """Identify fragment content independently of source and temporal position."""
    start = _exact_int(start, "start")
    duration = _exact_int(duration, "duration")
    if start < 0 or duration <= 0 or start + duration > len(actions):
        raise ValueError("fragment window is outside the action sequence")
    return action_sequence_semantic_id(actions[start : start + duration])


def _canonical_value(value: Any, path: str) -> Any:
    if value is None or type(value) in (str, int, bool):
        return value
    if isinstance(value, float):
        raise TypeError(f"non-canonical numeric value at {path}: {value!r}")
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key in sorted(value):
            if type(key) is not str:
                raise TypeError(f"candidate key at {path} must be a string")
            nested = value[key]
            if key == "actions":
                result[key] = canonical_action_sequence(nested)
            else:
                result[key] = _canonical_value(nested, f"{path}.{key}")
        return result
    if isinstance(value, (list, tuple)):
        return [_canonical_value(nested, f"{path}[{index}]") for index, nested in enumerate(value)]
    raise TypeError(f"unsupported candidate value at {path}: {type(value).__name__}")


def canonical_candidate(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return semantic candidate structure with top-level source aliases removed."""
    if not isinstance(payload, Mapping):
        raise TypeError("candidate payload must be a mapping")
    semantic = {
        key: value
        for key, value in payload.items()
        if key not in _CANDIDATE_PROVENANCE_FIELDS
    }
    validate_runtime_identity_safe(semantic)
    canonical = _canonical_value(semantic, "candidate")
    return {"canonical_version": CANDIDATE_CANONICAL_VERSION, **canonical}


def candidate_semantic_id(payload: Mapping[str, Any]) -> str:
    return stable_identity("RLRMB-CANDIDATE-V1", canonical_candidate(payload))


def load_route_actions(path: str | Path) -> list[Any]:
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    actions = document.get("actions") if isinstance(document, Mapping) else document
    if not isinstance(actions, list):
        raise ValueError(f"route has no action list: {path}")
    return actions


def donor_fragment_diversity(
    donor_paths: Iterable[str | Path],
    windows: Iterable[tuple[int, int]],
) -> dict[tuple[int, int], int]:
    """Count distinct canonical fragment contents for each requested window."""
    normalized_windows = tuple(
        (_exact_int(start, "window.start"), _exact_int(duration, "window.duration"))
        for start, duration in windows
    )
    identities = {window: set() for window in normalized_windows}
    for path in donor_paths:
        actions = load_route_actions(path)
        for window in normalized_windows:
            identities[window].add(action_fragment_content_id(actions, *window))
    return {window: len(values) for window, values in identities.items()}
