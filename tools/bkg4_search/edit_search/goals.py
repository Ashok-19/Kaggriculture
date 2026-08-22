from __future__ import annotations

import hashlib
import itertools
import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from tools.bkg4_search.information_state_io import registry_manifest
from tools.bkg4_search.module_codec import FEATURES, ConditionGene

from .panels import CaseBank


# Contrastive specialists should explain strategic regimes, not memorize individual
# worker/tile coordinates. Exact CUDA remains authoritative for all candidate effects.
CONTRASTIVE_FEATURES = frozenset({
    "STEP", "DAY", "HOUR", "SHOP_AT", "SHOP_SEEN",
    "OWN_MONEY", "OPP_MONEY", "OWN_UNITS", "OPP_UNITS",
    "OWN_LAND", "OPP_LAND", "OWN_HIRES_TODAY", "OPP_HIRES_TODAY",
    "MARKET_PRICE", "MARKET_INVENTORY", "OWN_SHED", "OWN_SEED",
})


@dataclass(frozen=True)
class SearchGoal:
    target_case_ids: tuple[str, ...]
    protected_guard_ids: tuple[str, ...]
    fragile_win_ids: tuple[str, ...] = ()
    counterexample_ids: tuple[str, ...] = ()
    desired_repair_type: str = "repair_loss"

    def __post_init__(self) -> None:
        if not self.target_case_ids:
            raise ValueError("SearchGoal requires at least one target failure")
        if set(self.target_case_ids) & set(self.protected_guard_ids):
            raise ValueError("SearchGoal target and protected guard sets must be disjoint")

    @property
    def goal_id(self) -> str:
        payload = json.dumps({
            "target": sorted(self.target_case_ids),
            "guard": sorted(self.protected_guard_ids),
            "fragile": sorted(self.fragile_win_ids),
            "counterexamples": sorted(self.counterexample_ids),
            "type": self.desired_repair_type,
        }, sort_keys=True, separators=(",", ":"))
        return "goal:" + hashlib.sha256(("BKG4-SEARCH-GOAL-V1\0" + payload).encode()).hexdigest()


def decode_case_contexts(bank: CaseBank) -> dict[str, tuple[dict[str, int | str], ...]]:
    registry = registry_manifest()
    feature_set = set(CONTRASTIVE_FEATURES)
    out: dict[str, tuple[dict[str, int | str], ...]] = {}
    for case in bank:
        if not case.public_context_width or not case.context_probe_steps or not case.public_context_values:
            out[case.case_id] = ()
            continue
        if case.public_context_registry_width != len(registry):
            raise ValueError(f"case public registry width mismatch: {case.case_id}")
        rows: list[dict[str, int | str]] = []
        width = int(case.public_context_width)
        for pi, _step in enumerate(case.context_probe_steps):
            frame = case.public_context_values[pi * width:(pi + 1) * width]
            if len(frame) != width:
                raise ValueError(f"case public context frame is truncated: {case.case_id}")
            for ri, (name, args) in enumerate(registry):
                if ri >= len(frame) or name not in feature_set or len(args) > 1:
                    continue
                rows.append({
                    "probe_index": int(pi), "probe_step": int(_step),
                    "feature": str(name), "index": int(args[0]) if args else 0, "value": int(frame[ri]),
                })
        out[case.case_id] = tuple(rows)
    return out


def _values(contexts: Mapping[str, Sequence[Mapping[str, Any]]], case_ids: Sequence[str]) -> dict[tuple[str, int], list[int]]:
    out: dict[tuple[str, int], list[int]] = {}
    for cid in case_ids:
        for row in contexts.get(str(cid), ()):
            try:
                key = (str(row["feature"]), int(row.get("index", 0))); value = int(row["value"])
            except (KeyError, TypeError, ValueError):
                continue
            if key[0] in FEATURES:
                out.setdefault(key, []).append(value)
    return out


def _condition_hit(condition: ConditionGene, rows: Sequence[Mapping[str, Any]]) -> bool:
    vals = [int(r["value"]) for r in rows if str(r.get("feature")) == condition.feature and int(r.get("index", 0)) == condition.index]
    if not vals:
        return False
    if condition.cmp == "LE": return any(v <= condition.value for v in vals)
    if condition.cmp == "GE": return any(v >= condition.value for v in vals)
    if condition.cmp == "EQ": return any(v == condition.value for v in vals)
    return False

def _context_index(
    contexts: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, tuple[dict[tuple[str, int], tuple[int, ...]], ...]]:
    """Index legal public observations by probe frame, preserving conjunction semantics."""
    out: dict[str, tuple[dict[tuple[str, int], tuple[int, ...]], ...]] = {}
    for cid, rows in contexts.items():
        frames: dict[int, dict[tuple[str, int], list[int]]] = {}
        for row in rows:
            try:
                frame = int(row.get("probe_index", 0))
                key = (str(row["feature"]), int(row.get("index", 0)))
                value = int(row["value"])
            except (KeyError, TypeError, ValueError):
                continue
            frames.setdefault(frame, {}).setdefault(key, []).append(value)
        out[str(cid)] = tuple(
            {k: tuple(v) for k, v in grouped.items()}
            for _, grouped in sorted(frames.items())
        )
    return out


def _condition_hit_indexed(condition: ConditionGene, rows: Mapping[tuple[str, int], Sequence[int]]) -> bool:
    vals = rows.get((condition.feature, condition.index), ())
    if not vals:
        return False
    if condition.cmp == "LE": return any(v <= condition.value for v in vals)
    if condition.cmp == "GE": return any(v >= condition.value for v in vals)
    if condition.cmp == "EQ": return any(v == condition.value for v in vals)
    return False


def _conjunction_stats_indexed(
    conditions: Sequence[ConditionGene],
    goal: SearchGoal,
    contexts: Mapping[str, Sequence[Mapping[tuple[str, int], Sequence[int]]]],
) -> dict[str, int]:
    def hit(cid: str) -> bool:
        frames = contexts.get(cid, ())
        return bool(conditions) and any(
            all(_condition_hit_indexed(c, frame) for c in conditions)
            for frame in frames
        )
    return {
        "target_hits": sum(hit(x) for x in goal.target_case_ids),
        "guard_hits": sum(hit(x) for x in goal.protected_guard_ids),
        "fragile_hits": sum(hit(x) for x in goal.fragile_win_ids),
        "counterexample_hits": sum(hit(x) for x in goal.counterexample_ids),
    }


def matching_probe_steps(
    conditions: Sequence[ConditionGene],
    case_id: str,
    contexts: Mapping[str, Sequence[Mapping[str, Any]]],
) -> tuple[int, ...]:
    """Return public probe steps where all predicates co-occur for one case."""
    frames: dict[int, list[Mapping[str, Any]]] = {}
    frame_steps: dict[int, int] = {}
    for row in contexts.get(str(case_id), ()):
        frame = int(row.get("probe_index", 0))
        frames.setdefault(frame, []).append(row)
        frame_steps.setdefault(frame, int(row.get("probe_step", frame)))
    return tuple(
        frame_steps[frame]
        for frame, rows in sorted(frames.items())
        if conditions and all(_condition_hit(c, rows) for c in conditions)
    )


def conjunction_stats(
    conditions: Sequence[ConditionGene],
    goal: SearchGoal,
    contexts: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, int]:
    return _conjunction_stats_indexed(conditions, goal, _context_index(contexts))


def _representatives(values: Sequence[int], *, limit: int = 7) -> tuple[int, ...]:
    xs = sorted(set(int(x) for x in values))
    if len(xs) <= limit:
        return tuple(xs)
    picks = {0, len(xs) - 1}
    for q in (0.10, 0.25, 0.50, 0.75, 0.90):
        picks.add(int(round(q * (len(xs) - 1))))
    return tuple(xs[i] for i in sorted(picks))


def _condition_pool(
    goal: SearchGoal,
    indexed: Mapping[str, Sequence[Mapping[tuple[str, int], Sequence[int]]]],
    *,
    feature_budget: int,
    condition_budget: int,
) -> list[ConditionGene]:
    target: dict[tuple[str, int], list[int]] = {}
    guards: dict[tuple[str, int], list[int]] = {}
    for cid in goal.target_case_ids:
        for frame in indexed.get(cid, ()):
            for key, vals in frame.items(): target.setdefault(key, []).extend(int(x) for x in vals)
    for cid in goal.protected_guard_ids:
        for frame in indexed.get(cid, ()):
            for key, vals in frame.items(): guards.setdefault(key, []).extend(int(x) for x in vals)

    def feature_key(key: tuple[str, int]) -> tuple[Any, ...]:
        tv = sorted(target[key]); gv = sorted(guards[key])
        tmed = tv[len(tv)//2]; gmed = gv[len(gv)//2]
        span = max(1, max(tv + gv) - min(tv + gv))
        perfect = int(max(tv) < min(gv) or max(gv) < min(tv))
        gap = max(0, min(gv) - max(tv), min(tv) - max(gv))
        separation = abs(tmed - gmed) / span
        return (-perfect, -(gap / span), -separation, key[0], key[1])

    common = [key for key in set(target) & set(guards) if target[key] and guards[key]]
    common.sort(key=feature_key)
    common = common[:max(1, int(feature_budget))]
    scored: dict[ConditionGene, tuple[Any, ...]] = {}
    for feature, index in common:
        tv = target[(feature, index)]; gv = guards[(feature, index)]
        trep = _representatives(tv); grep = _representatives(gv)
        candidates = set(trep) | set(grep)
        # Only representative cross-class boundaries are needed; all-pairs raw
        # target×guard midpoints were the real-panel combinatorial bottleneck.
        for a in trep:
            for b in grep:
                if a != b:
                    lo, hi = sorted((a, b)); candidates.add((lo + hi) // 2)
        if feature in ("SHOP_AT", "SHOP_SEEN"):
            cmps = ("EQ",); candidates = set(trep)
        else:
            cmps = ("LE", "GE")
            for v in tuple(candidates): candidates.update((int(v) - 1, int(v) + 1))
        for value in sorted(candidates):
            for cmp in cmps:
                try:
                    c = ConditionGene(feature, cmp, index, 1 if feature == "SHOP_SEEN" else int(value)); c.validate()
                except ValueError:
                    continue
                diag = _conjunction_stats_indexed((c,), goal, indexed)
                if not diag["target_hits"]:
                    continue
                scored[c] = (
                    diag["counterexample_hits"], diag["guard_hits"], diag["fragile_hits"],
                    -diag["target_hits"], feature_key((feature, index)), c.feature, c.index, c.cmp, c.value,
                )
    return [c for c, _ in sorted(scored.items(), key=lambda kv: kv[1])[:max(1, int(condition_budget))]]


def synthesize_contrastive_conditions(
    goal: SearchGoal,
    contexts: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    max_conditions: int = 4,
    limit: int = 64,
    feature_budget: int = 64,
    guard_excluding_only: bool = False,
) -> list[tuple[ConditionGene, ...]]:
    max_conditions = max(1, min(4, int(max_conditions))); limit = max(1, int(limit))
    feature_budget = max(4, min(256, int(feature_budget)))
    indexed = _context_index(contexts)
    singles = _condition_pool(
        goal, indexed, feature_budget=feature_budget,
        condition_budget=max(32, min(192, limit * 4)),
    )
    if not singles: return []

    # Beam expansion used to rescan every case×probe frame for every candidate
    # conjunction. Cache each single predicate as an exact frame bitset instead;
    # conjunction is then integer-mask intersection, preserving same-frame semantics.
    frame_rows: list[Mapping[tuple[str, int], Sequence[int]]] = []
    case_masks: dict[str, int] = {}
    for cid in sorted(indexed):
        mask = 0
        for frame in indexed[cid]:
            bit = 1 << len(frame_rows)
            frame_rows.append(frame)
            mask |= bit
        case_masks[cid] = mask
    condition_masks: dict[ConditionGene, int] = {}
    for condition in singles:
        mask = 0
        for fi, frame in enumerate(frame_rows):
            if _condition_hit_indexed(condition, frame):
                mask |= 1 << fi
        condition_masks[condition] = mask
    stats_cache: dict[tuple[ConditionGene, ...], dict[str, int]] = {}

    def fast_stats(conds: tuple[ConditionGene, ...]) -> dict[str, int]:
        cached = stats_cache.get(conds)
        if cached is not None:
            return cached
        mask = -1
        for c in conds:
            mask &= condition_masks[c]
        def count(ids: Sequence[str]) -> int:
            return sum(bool(mask & case_masks.get(str(cid), 0)) for cid in ids)
        out = {
            "target_hits": count(goal.target_case_ids),
            "guard_hits": count(goal.protected_guard_ids),
            "fragile_hits": count(goal.fragile_win_ids),
            "counterexample_hits": count(goal.counterexample_ids),
        }
        stats_cache[conds] = out
        return out

    def key(conds: tuple[ConditionGene, ...]) -> tuple[Any, ...]:
        d = fast_stats(conds)
        return (
            d["counterexample_hits"], d["guard_hits"], d["fragile_hits"],
            -d["target_hits"], len(conds),
            tuple((c.feature, c.index, c.cmp, c.value) for c in conds),
        )

    candidates: dict[tuple[tuple[str, int, str, int], ...], tuple[ConditionGene, ...]] = {}
    beam = [(c,) for c in singles]
    for depth in range(1, max_conditions + 1):
        for conds in beam:
            ident = tuple((c.feature, c.index, c.cmp, c.value) for c in conds)
            candidates.setdefault(ident, conds)
        if depth == max_conditions: break
        parents = sorted(beam, key=key)[:max(16, min(64, limit * 2))]
        next_rows: dict[tuple[tuple[str, int, str, int], ...], tuple[ConditionGene, ...]] = {}
        for parent in parents:
            used = set(parent)
            for c in singles:
                if c in used: continue
                conds = tuple(sorted((*parent, c), key=lambda x: (x.feature, x.index, x.cmp, x.value)))
                ident = tuple((x.feature, x.index, x.cmp, x.value) for x in conds)
                if fast_stats(conds)["target_hits"]:
                    next_rows.setdefault(ident, conds)
        beam = sorted(next_rows.values(), key=key)[:max(32, min(128, limit * 4))]
        if not beam: break
    ranked = sorted(candidates.values(), key=key)
    if guard_excluding_only:
        ranked = [
            conds for conds in ranked
            if fast_stats(conds)["target_hits"] > 0 and fast_stats(conds)["guard_hits"] == 0
        ]
    return ranked[:limit]


def _cluster_signature(rows: Sequence[Mapping[str, Any]]) -> tuple[tuple[str, int, int], ...]:
    # Identity-free public regime signature. Coarse time/unit context groups coherent failures
    # without encoding opponent/replay provenance.
    wanted = {"DAY", "HOUR", "OWN_UNITS", "OPP_UNITS", "OWN_LAND", "OPP_LAND"}
    vals: dict[tuple[str, int], list[int]] = {}
    for row in rows:
        feature = str(row.get("feature", ""))
        if feature in wanted:
            vals.setdefault((feature, int(row.get("index", 0))), []).append(int(row["value"]))
    sig=[]
    for (feature,index), xs in sorted(vals.items()):
        xs=sorted(xs); sig.append((feature,index,xs[len(xs)//2]))
    return tuple(sig)


def build_search_goals(bank: CaseBank, *, counterexample_ids: Sequence[str] = (), max_goals: int = 12) -> tuple[SearchGoal, ...]:
    contexts = decode_case_contexts(bank)
    losses = [x for x in bank if x.baseline_margin < 0]
    guards = tuple(x.case_id for x in bank if x.protected and x.baseline_margin > 0)
    fragile = tuple(x.case_id for x in bank if x.protected and 0 < x.baseline_margin <= 5000)
    if not losses or not guards:
        return ()
    groups: dict[tuple[tuple[str, int, int], ...], list[str]] = {}
    for row in losses:
        groups.setdefault(_cluster_signature(contexts.get(row.case_id, ())), []).append(row.case_id)
    goals: list[SearchGoal] = []
    for _, ids in sorted(groups.items(), key=lambda kv: (len(kv[1]), kv[0])):
        goals.append(SearchGoal(tuple(sorted(ids)), guards, fragile, tuple(sorted(set(counterexample_ids) & set(guards)))))
    # Singleton specialists remain a bounded fallback for exceptional failures.
    for row in sorted(losses, key=lambda x: (x.baseline_margin, x.case_id)):
        if len(goals) >= max_goals: break
        goal = SearchGoal((row.case_id,), guards, fragile, tuple(sorted(set(counterexample_ids) & set(guards))))
        if goal.goal_id not in {x.goal_id for x in goals}: goals.append(goal)
    return tuple(goals[:max_goals])
