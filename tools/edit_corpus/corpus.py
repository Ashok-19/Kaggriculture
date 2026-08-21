from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from typing import Any, Iterable, Mapping, Sequence

SCHEMA_VERSION = 1
DEFAULT_FRAGILE_MARGIN = 5_000
DEFAULT_SEVERE_DELTA = -3_000
PASS_ACTION = {"farmer": ["PASS"], "hands": [], "market": []}
_FORBIDDEN_MODEL_KEY_PARTS = (
    "team",
    "opponent",
    "seed",
    "episode",
    "replay",
    "submission",
    "agent_name",
    "prov_",
)


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def stable_identity(namespace: str, *parts: Any) -> str:
    h = hashlib.sha256()
    h.update(namespace.encode("utf-8"))
    h.update(b"\0")
    for part in parts:
        raw = part if isinstance(part, bytes) else canonical_json(part).encode("utf-8")
        h.update(len(raw).to_bytes(8, "little"))
        h.update(raw)
    return h.hexdigest()


_UNIT_NOARG_OPS = frozenset({
    "PASS", "NORTH", "SOUTH", "EAST", "WEST", "DROP", "WATER", "HARVEST",
    "FERTILIZE", "DIG", "BUILD_COOP", "BUILD_PASTURE", "FEED",
    "COLLECT_FERTILIZER", "CARE",
})
_UNIT_ITEM_OPS = frozenset({"PLANT"})
_UNIT_ITEM_COUNT_OPS = frozenset({"PICKUP", "PLACE"})
_MARKET_ATOMIC_OPS = frozenset({"HIRE", "BUY_LAND"})
_MARKET_COUNT_OPS = frozenset({"BUY_SEED", "BUY_PRODUCT", "BUY_ANIMAL", "SELL"})


def _official_int(value: Any) -> int | None:
    """Mirror the official engine's int(value) coercion, but fail closed."""
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return None


def _normalize_unit_action(value: Any) -> list[Any]:
    # Official _apply_unit_action silently no-ops non-list/empty/malformed actions.
    if not isinstance(value, (list, tuple)) or not value or not isinstance(value[0], str):
        return ["PASS"]
    op = value[0]
    if op in _UNIT_NOARG_OPS:
        return [op]
    if op in _UNIT_ITEM_OPS:
        if len(value) < 2 or not isinstance(value[1], str):
            return ["PASS"]
        return [op, value[1]]
    if op in _UNIT_ITEM_COUNT_OPS:
        if len(value) < 2 or not isinstance(value[1], str):
            return ["PASS"]
        n = 1 if len(value) < 3 else _official_int(value[2])
        if n is None:
            return ["PASS"]
        return [op, value[1], n]
    return ["PASS"]


def _normalize_market_order(value: Any) -> list[Any]:
    # Official _parse_order ignores malformed/non-list/unknown market orders.
    if not isinstance(value, (list, tuple)) or not value or not isinstance(value[0], str):
        return []
    op = value[0]
    if op in _MARKET_ATOMIC_OPS:
        return [op]
    if op not in _MARKET_COUNT_OPS or len(value) < 3 or not isinstance(value[1], str):
        return []
    n = _official_int(value[2])
    if n is None or n <= 0:
        return []
    return [op, value[1], n]


def _normalize_action(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return dict(PASS_ACTION)
    farmer = _normalize_unit_action(value.get("farmer"))
    hands_raw = value.get("hands")
    market_raw = value.get("market")
    hands = [_normalize_unit_action(x) for x in hands_raw] if isinstance(hands_raw, list) else []
    market = [_normalize_market_order(x) for x in market_raw] if isinstance(market_raw, list) else []
    return {"farmer": farmer, "hands": hands, "market": market}


def replay_to_native_tapes(replay_json: str | Mapping[str, Any], provenance: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    replay = json.loads(replay_json) if isinstance(replay_json, str) else dict(replay_json)
    steps = replay.get("steps") or []
    if len(steps) < 2 or any(not isinstance(step, list) or len(step) != 2 for step in steps):
        raise ValueError("Kaggriculture replay must contain two-agent steps including the initial placeholder")
    info = replay.get("info") or {}
    teams = list(info.get("TeamNames") or ["seat0", "seat1"])
    if len(teams) != 2:
        raise ValueError("Kaggriculture replay must contain exactly two team names")
    episode_id = int(info.get("EpisodeId") or (provenance or {}).get("episode_id") or 0)
    if not episode_id:
        raise ValueError("replay episode id is required for provenance")
    seed = int(info.get("seed") if info.get("seed") is not None else (provenance or {}).get("seed", 0))
    ranks = dict((provenance or {}).get("daily_rank_by_team") or {})
    rewards = replay.get("rewards") or [None, None]
    date = str((provenance or {}).get("date") or "")
    rows: list[dict[str, Any]] = []
    for seat in (0, 1):
        actions = [_normalize_action(step[seat].get("action") if isinstance(step[seat], Mapping) else None) for step in steps[1:]]
        tape_id = stable_identity("KAGGRICULTURE-NATIVE-TAPE-V1", actions)
        rows.append({
            "source_seat": seat,
            "actions": actions,
            "tape_semantic_id": tape_id,
            "prov_episode_id": episode_id,
            "prov_team_name": str(teams[seat]),
            "prov_opponent_name": str(teams[1 - seat]),
            "prov_seed": seed,
            "prov_date": date,
            "prov_daily_rank": int(ranks.get(str(teams[seat]), 999)),
            "prov_reward": rewards[seat] if seat < len(rewards) else None,
            "prov_opponent_reward": rewards[1 - seat] if 1 - seat < len(rewards) else None,
        })
    return rows


def assert_model_payload_safe(payload: Mapping[str, Any]) -> None:
    def walk(value: Any, path: str) -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                lk = str(key).lower()
                if any(part in lk for part in _FORBIDDEN_MODEL_KEY_PARTS):
                    raise ValueError(f"forbidden model feature key at {path}{key}: identity/private provenance leakage")
                walk(child, f"{path}{key}.")
        elif isinstance(value, list):
            for i, child in enumerate(value):
                walk(child, f"{path}{i}.")
    walk(payload, "")


def build_case_candidates(tapes: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    unique: dict[tuple[str, int, str], Mapping[str, Any]] = {}
    for tape in tapes:
        tid = str(tape["tape_semantic_id"])
        seed = int(tape["prov_seed"])
        date = str(tape.get("prov_date", ""))
        unique.setdefault((tid, seed, date), tape)
    rows: list[dict[str, Any]] = []
    for (tape_id, seed, date), tape in sorted(unique.items()):
        for candidate_seat in (0, 1):
            model_context = {"seat": candidate_seat}
            assert_model_payload_safe(model_context)
            case_id = stable_identity("BKG4-REPLAY-CASE-V1", date, tape_id, seed, candidate_seat)
            rows.append({
                "case_id": case_id,
                "tape_semantic_id": tape_id,
                "seat": candidate_seat,
                "model_context_json": canonical_json(model_context),
                "prov_seed": seed,
                "prov_episode_id": int(tape.get("prov_episode_id", 0)),
                "prov_source_seat": int(tape.get("source_seat", -1)),
                "prov_team_name": str(tape.get("prov_team_name", "")),
                "prov_opponent_name": str(tape.get("prov_opponent_name", "")),
                "prov_date": str(tape.get("prov_date", "")),
                "prov_daily_rank": int(tape.get("prov_daily_rank", 999)),
            })
    return rows


def _case_bucket(margin: int | float, fragile_margin: int) -> int:
    if margin < 0:
        return 0
    if margin <= fragile_margin:
        return 1
    return 2


def select_representative_cases(
    rows: Sequence[Mapping[str, Any]],
    *,
    max_cases: int,
    fragile_margin: int = DEFAULT_FRAGILE_MARGIN,
) -> list[dict[str, Any]]:
    if max_cases <= 0:
        return []
    groups: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for raw in rows:
        row = dict(raw)
        if "baseline_margin" not in row:
            raise ValueError("baseline_margin is required before representative-case selection")
        seat = int(row["seat"])
        if seat not in (0, 1):
            raise ValueError("candidate seat must be 0 or 1")
        bucket = _case_bucket(float(row["baseline_margin"]), int(fragile_margin))
        groups[(bucket, seat)].append(row)
    for key, items in groups.items():
        bucket, _ = key
        if bucket == 0:
            items.sort(key=lambda r: (float(r["baseline_margin"]), int(r.get("prov_daily_rank", 999)), str(r["case_id"])))
        else:
            items.sort(key=lambda r: (float(r["baseline_margin"]), int(r.get("prov_daily_rank", 999)), str(r["case_id"])))

    selected: list[dict[str, Any]] = []
    depth = 0
    ordered_groups = [(b, s) for b in (0, 1, 2) for s in (0, 1)]
    while len(selected) < min(max_cases, len(rows)):
        added = False
        for key in ordered_groups:
            items = groups.get(key, [])
            if depth < len(items):
                selected.append(items[depth])
                added = True
                if len(selected) >= max_cases:
                    break
        if not added:
            break
        depth += 1
    return sorted(selected, key=lambda r: str(r["case_id"]))


def select_screening_cases(
    final_rows: Sequence[Mapping[str, Any]],
    *,
    max_cases: int,
) -> list[dict[str, Any]]:
    """Choose a deterministic representative screening subset of the final panel."""
    if max_cases <= 0 or not final_rows:
        return []
    return select_representative_cases(
        final_rows,
        max_cases=min(int(max_cases), len(final_rows)),
    )


def canonical_edit_identity(parent_id: str, action: Mapping[str, Any], child_id: str) -> str:
    return stable_identity(
        "BONSAI-EDIT-CORPUS-EDIT-V1",
        str(parent_id),
        json.loads(canonical_json(dict(action))),
        str(child_id),
    )


def _outcome(margin: int) -> str:
    return "W" if margin > 0 else "L" if margin < 0 else "T"


def derive_case_label(
    baseline_margin: int,
    candidate_margin: int,
    action_changes: int,
    activations: int,
    *,
    protected: bool = True,
    parent_action_trace_hash: int | None = None,
    candidate_action_trace_hash: int | None = None,
) -> dict[str, Any]:
    b = int(baseline_margin)
    c = int(candidate_margin)
    delta = c - b
    changes = int(action_changes)
    acts = int(activations)
    if (parent_action_trace_hash is None) != (candidate_action_trace_hash is None):
        raise ValueError("parent/candidate action trace hashes must be supplied together")
    trace_equal = (
        int(parent_action_trace_hash) == int(candidate_action_trace_hash)
        if parent_action_trace_hash is not None
        else changes == 0
    )
    return {
        "baseline_margin": b,
        "candidate_margin": c,
        "margin_delta": delta,
        "baseline_outcome": _outcome(b),
        "candidate_outcome": _outcome(c),
        "l2w": bool(b < 0 and c > 0),
        "w2l": bool(b > 0 and c <= 0),
        "protected": bool(protected),
        "action_changes": changes,
        "activations": acts,
        "parent_action_trace_hash": None if parent_action_trace_hash is None else int(parent_action_trace_hash),
        "candidate_action_trace_hash": None if candidate_action_trace_hash is None else int(candidate_action_trace_hash),
        "action_trace_equal": bool(trace_equal),
        "behavior_changing": bool(delta != 0 or not trace_equal),
    }


def derive_edit_labels(
    case_rows: Sequence[Mapping[str, Any]],
    *,
    fragile_margin: int = DEFAULT_FRAGILE_MARGIN,
    severe_delta: int = DEFAULT_SEVERE_DELTA,
) -> dict[str, Any]:
    if not case_rows:
        raise ValueError("cannot derive edit labels without exact CUDA case rows")
    rows = [dict(x) for x in case_rows]
    for row in rows:
        for key in ("baseline_margin", "candidate_margin", "margin_delta", "action_changes", "activations"):
            if key not in row:
                raise ValueError(f"missing exact CUDA label field {key}")
    l2w = sum(bool(r.get("l2w")) for r in rows)
    w2l = sum(bool(r.get("w2l")) for r in rows)
    severe = sum(int(r["baseline_margin"]) > 0 and int(r["margin_delta"]) < int(severe_delta) for r in rows)
    losses_before = sum(int(r["baseline_margin"]) < 0 for r in rows)
    losses_after = sum(int(r["candidate_margin"]) < 0 for r in rows)
    fragile_before = sum(0 < int(r["baseline_margin"]) <= int(fragile_margin) for r in rows)
    fragile_after = sum(0 < int(r["candidate_margin"]) <= int(fragile_margin) for r in rows)
    deltas = [int(r["margin_delta"]) for r in rows]
    protected_deltas = [int(r["margin_delta"]) for r in rows if bool(r.get("protected", True)) and int(r["baseline_margin"]) > 0]
    action_change_total = sum(int(r["action_changes"]) for r in rows)
    activation_total = sum(int(r["activations"]) for r in rows)
    behavior_changing = any(
        bool(r.get("behavior_changing", int(r["margin_delta"]) != 0 or int(r["action_changes"]) > 0))
        for r in rows
    )
    exact_noop = all(
        int(r["margin_delta"]) == 0
        and bool(r.get("action_trace_equal", int(r["action_changes"]) == 0))
        for r in rows
    )
    mean_delta = sum(deltas) / len(deltas)
    worst_candidate_margin = min(int(r["candidate_margin"]) for r in rows)
    worst_protected_win_delta = min(protected_deltas) if protected_deltas else 0
    strict_safe = w2l == 0 and severe == 0 and losses_after <= losses_before
    return {
        "l2w_count": int(l2w),
        "w2l_count": int(w2l),
        "severe_regression_count": int(severe),
        "losses_before": int(losses_before),
        "losses_after": int(losses_after),
        "fragile_wins_before": int(fragile_before),
        "fragile_wins_after": int(fragile_after),
        "mean_margin_delta": float(mean_delta),
        "worst_protected_win_delta": int(worst_protected_win_delta),
        "worst_candidate_margin": int(worst_candidate_margin),
        "strict_safe": bool(strict_safe),
        "exact_noop": bool(exact_noop),
        "behavior_changing": bool(behavior_changing),
        "action_change_total": int(action_change_total),
        "activation_total": int(activation_total),
        "lex_losses_after": int(losses_after),
        "lex_new_losses": int(w2l),
        "lex_fragile_wins_after": int(fragile_after),
        "lex_worst_margin": int(worst_candidate_margin),
        "lex_mean_margin_delta": float(mean_delta),
    }


def pending_replay_shards(
    source_rows: Iterable[Mapping[str, Any]],
    processed: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    pending: list[dict[str, Any]] = []
    seen_dates: set[str] = set()
    for raw in sorted((dict(x) for x in source_rows), key=lambda x: str(x["date"])):
        date = str(raw["date"])
        if date in seen_dates:
            raise RuntimeError(f"duplicate replay archive date: {date}")
        seen_dates.add(date)
        source_sha = str(raw["replay_shard_sha256"])
        prev = processed.get(date)
        if prev is None or str(prev.get("status")) != "complete":
            pending.append(raw)
            continue
        if str(prev.get("source_replay_shard_sha256")) != source_sha:
            raise RuntimeError(f"source replay shard for {date} changed after processing")
    return pending
