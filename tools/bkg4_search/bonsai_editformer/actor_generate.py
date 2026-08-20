from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import jax
import numpy as np

from actor_grammar import (
    COMPARATORS,
    EDIT_OPERATIONS,
    FEATURES,
    OP_INDEX,
    EditAction,
    apply_edit,
    denormalize_value,
    node_refs,
    target_masks,
    valid_operation_mask,
)
from actor_model import BonsaiEditActorCritic
from data import OUT, ROUTES, Row, _feature_arrays, current_rows, feature_arrays_from_capture, label_dict
from model import EditFormerConfig
from tools.bkg4_search.campaign import mutate, random_condition
from tools.bkg4_search.cuda_io import GENOME_SIZE, FILE_HEAD, pack_genomes
from tools.bkg4_search.module_codec import ModuleGenome
from gcac_context import case_meta_from_tsv


@dataclass(frozen=True)
class GeneratedCandidate:
    source: str
    semantic_id: str
    parent_semantic_id: str
    action: dict
    genome: dict
    actor_logit: float | None = None
    critic_score: float | None = None
    critic_regression_prob: float | None = None
    critic_flip_prob: float | None = None
    critic_margin_pred: float | None = None
    critic_worst_win_pred: float | None = None
    critic_effect_mean_abs: float | None = None
    critic_effect_cases: int | None = None
    probe_activated_cases: int | None = None
    probe_route_divergence_cases: int | None = None
    probe_loss_target_cases: int | None = None
    probe_win_exposure_cases: int | None = None
    probe_effect_score: float | None = None
    goal_case_ids: tuple[int, ...] | None = None
    generated_goal_case_ids: tuple[int, ...] | None = None
    goal_mode: str | None = None
    critic_goal_delta_norm: float | None = None
    critic_goal_delta_margin: float | None = None
    critic_goal_improve_prob: float | None = None
    critic_goal_flip_prob: float | None = None
    critic_protected_regress_prob_max: float | None = None
    critic_predicted_target_flips: int | None = None
    critic_predicted_win_crossings: int | None = None


def _repeat_batch(batch: dict, n: int) -> dict:
    out = {}
    for k, v in batch.items():
        x = np.asarray(v)
        if x.shape[0] != 1:
            raise ValueError(f"expected singleton parent batch for {k}, got {x.shape}")
        out[k] = np.repeat(x, n, axis=0)
    return out


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    e = np.exp(x)
    return e / max(1e-30, float(np.sum(e)))


def _sample_cat(logits: np.ndarray, rng: np.random.Generator, *, valid: np.ndarray | None = None,
                temperature: float = 1.0, topk: int = 0, penalty: np.ndarray | None = None) -> tuple[int, float]:
    z = np.asarray(logits, np.float64).copy()
    if penalty is not None:
        z -= np.asarray(penalty, np.float64)
    if valid is not None:
        z[~np.asarray(valid, bool)] = -1e30
    z /= max(1e-4, float(temperature))
    if topk and topk < len(z):
        keep = np.argpartition(z, -topk)[-topk:]
        m = np.ones(len(z), bool); m[keep] = False; z[m] = -1e30
    p = _softmax(z)
    if not np.isfinite(p).all() or p.sum() <= 0:
        raise ValueError("invalid categorical distribution")
    idx = int(rng.choice(len(z), p=p))
    return idx, float(logits[idx])


def _to_device(batch: dict) -> dict:
    import jax.numpy as jnp
    out = {}
    int_keys = {"tree_cat", "edit_cat", "tree_parent", "tree_depth", "tree_rule", "group", "actor_op", "actor_target"}
    bool_keys = {"tree_mask", "edit_mask", "case_mask", "actor_target_mask"}
    for k, v in batch.items():
        if k in int_keys: out[k] = jnp.asarray(v, jnp.int32)
        elif k in bool_keys: out[k] = jnp.asarray(v, bool)
        else: out[k] = jnp.asarray(v, jnp.float32)
    return out


def _sample_actions(model: BonsaiEditActorCritic, params, cfg: EditFormerConfig, parent: ModuleGenome, parent_batch: dict,
                    n: int, seed: int, temperature: float, topk: int, diversity_penalty: float,
                    route_ids: Sequence[int], *, goal_mask: np.ndarray | None = None,
                    goal_spec: np.ndarray | None = None, learned_operation_mask: np.ndarray | None = None) -> list[tuple[EditAction, float]]:
    rng = np.random.default_rng(seed)
    base = dict(parent_batch)
    base["actor_target_mask"] = np.asarray(target_masks(parent, cfg.tree_cap), bool)[None, ...]
    base["actor_op"] = np.zeros((1,), np.int32); base["actor_target"] = np.zeros((1,), np.int32)
    if goal_mask is not None:
        gm=np.asarray(goal_mask,bool).reshape(1,cfg.case_cap);base["goal_mask"]=gm
    else:
        base["goal_mask"]=np.asarray(base.get("case_mask",np.ones((1,cfg.case_cap),bool)),bool)
    if goal_spec is not None:
        base["goal_spec"]=np.asarray(goal_spec,np.float32).reshape(1,4)
    else:
        base["goal_spec"]=np.asarray([[1.0,0.0,1.0,0.0]],np.float32)
    first = model.apply({"params": params}, _to_device(base), train=False)
    op_logits = np.asarray(first["actor_op_logits"])[0]
    target_all = np.asarray(first["actor_target_logits_all"])[0]
    op_valid = np.asarray(valid_operation_mask(parent), bool)
    if learned_operation_mask is not None:
        lom=np.asarray(learned_operation_mask,bool)
        if lom.shape!=(len(EDIT_OPERATIONS),):raise ValueError(f"learned operation mask shape {lom.shape}")
        op_valid &= lom
    if not np.any(op_valid):raise ValueError("no learned actor operation remains valid for parent")
    masks = np.asarray(target_masks(parent, cfg.tree_cap), bool)
    refs = node_refs(parent, cfg.tree_cap)
    op_counts = np.zeros(len(EDIT_OPERATIONS), np.float64)
    target_counts = np.zeros(cfg.tree_cap, np.float64)
    ops, targets, actor_logits = [], [], []
    for _ in range(n):
        op, ol = _sample_cat(op_logits, rng, valid=op_valid, temperature=temperature, topk=topk,
                             penalty=diversity_penalty * op_counts)
        target, tl = _sample_cat(target_all[op], rng, valid=masks[op], temperature=temperature, topk=topk,
                                 penalty=diversity_penalty * target_counts)
        ops.append(op); targets.append(target); actor_logits.append(ol + tl)
        op_counts[op] += 1.0; target_counts[target] += 1.0

    second = _repeat_batch(base, n)
    second["actor_op"] = np.asarray(ops, np.int32); second["actor_target"] = np.asarray(targets, np.int32)
    out = model.apply({"params": params}, _to_device(second), train=False)
    arrays = {k: np.asarray(v) for k, v in out.items() if k.startswith("actor_")}
    route_valid = np.asarray([r in set(route_ids) for r in ROUTES], bool)
    actions = []
    for i, (op_i, target) in enumerate(zip(ops, targets)):
        op = EDIT_OPERATIONS[op_i]
        feature_i, fl = _sample_cat(arrays["actor_feature_logits"][i], rng, temperature=temperature, topk=topk)
        cmp_i, cl = _sample_cat(arrays["actor_cmp_logits"][i], rng, temperature=temperature, topk=topk)
        route_i, rl = _sample_cat(arrays["actor_route_logits"][i], rng, valid=route_valid, temperature=temperature, topk=topk)
        index_i, il = _sample_cat(arrays["actor_index_logits"][i], rng, temperature=temperature, topk=topk)
        feature2_i, _ = _sample_cat(arrays["actor_feature2_logits"][i], rng, temperature=temperature, topk=topk)
        cmp2_i, _ = _sample_cat(arrays["actor_cmp2_logits"][i], rng, temperature=temperature, topk=topk)
        index2_i, _ = _sample_cat(arrays["actor_index2_logits"][i], rng, temperature=temperature, topk=topk)
        cc_i, _ = _sample_cat(arrays["actor_condition_count_logits"][i], rng, temperature=temperature, topk=0)
        pr_i, _ = _sample_cat(arrays["actor_priority_logits"][i], rng, temperature=temperature, topk=0)
        feature = FEATURES[feature_i]; feature2 = FEATURES[feature2_i]
        # Threshold edits preserve the target predicate semantics; only the scalar moves.
        if op == "CHANGE_THRESHOLD":
            ref = refs[target]
            assert ref is not None
            old = parent.rules[ref.rule_index].conditions[ref.condition_index]
            feature = old.feature; cmp_i = COMPARATORS.index(old.cmp); index_i = old.index
        noise = float(rng.normal(0.0, 0.08 * max(0.25, temperature)))
        value_norm = float(np.clip(arrays["actor_value_norm"][i] + noise, -1, 1))
        value2_norm = float(np.clip(arrays["actor_value2_norm"][i] + noise, -1, 1))
        prefix_norm = float(np.clip(arrays["actor_prefix_norm"][i] + noise, -1, 1))
        action = EditAction(
            operation=op, target_node=int(target), feature=feature, cmp=COMPARATORS[cmp_i], index=int(index_i),
            value=denormalize_value(feature, value_norm), route=int(ROUTES[route_i]),
            feature2=feature2, cmp2=COMPARATORS[cmp2_i], index2=int(index2_i), value2=denormalize_value(feature2, value2_norm),
            condition_count=int(cc_i + 1), priority=int(pr_i), prefix_step=int(round((prefix_norm + 1.0) * 0.5 * 719)),
        )
        actions.append((action, float(actor_logits[i] + fl + cl + rl + il)))
    return actions


def actor_generate(model, params, cfg: EditFormerConfig, parent: ModuleGenome, parent_batch: dict, *, n: int,
                   seed: int = 73, temperature: float = 0.9, topk: int = 8, diversity_penalty: float = 0.35,
                   route_ids: Sequence[int] = ROUTES, max_rounds: int = 8,
                   goal_mask: np.ndarray | None = None, goal_spec: np.ndarray | None = None,
                   goal_case_ids: Sequence[int] = (), goal_mode: str = "LOSS_REPAIR",
                   learned_operation_mask: np.ndarray | None = None) -> list[GeneratedCandidate]:
    seen = {parent.semantic_id()}; result: list[GeneratedCandidate] = []
    for round_i in range(max_rounds):
        want = max(16, (n - len(result)) * 2)
        for action, logit in _sample_actions(model, params, cfg, parent, parent_batch, want, seed + round_i * 1009,
                                             temperature, topk, diversity_penalty, route_ids,
                                             goal_mask=goal_mask,goal_spec=goal_spec,learned_operation_mask=learned_operation_mask):
            try:
                g = apply_edit(parent, action, route_ids)
            except ValueError:
                continue
            sid = g.semantic_id()
            if sid in seen:
                continue
            seen.add(sid)
            result.append(GeneratedCandidate("actor", sid, parent.semantic_id(), asdict(action), g.canonical(), actor_logit=logit,
                                             goal_case_ids=tuple(int(x) for x in goal_case_ids),generated_goal_case_ids=tuple(int(x) for x in goal_case_ids),goal_mode=goal_mode))
            if len(result) >= n:
                return result
    return result


def random_typed_generate(parent: ModuleGenome, *, n: int, seed: int, route_ids: Sequence[int] = ROUTES) -> list[GeneratedCandidate]:
    rng = random.Random(seed); seen = {parent.semantic_id()}; out = []
    valid_ops = [i for i, ok in enumerate(valid_operation_mask(parent)) if ok]
    masks = target_masks(parent)
    for _ in range(n * 100):
        if len(out) >= n: break
        oi = rng.choice(valid_ops); op = EDIT_OPERATIONS[oi]; targets = [i for i, ok in enumerate(masks[oi]) if ok]
        target = rng.choice(targets); c1 = random_condition(rng); c2 = random_condition(rng)
        action = EditAction(op, target, c1.feature, c1.cmp, c1.index, c1.value, rng.choice(tuple(route_ids)),
                            c2.feature, c2.cmp, c2.index, c2.value, rng.choice((1, 2)), rng.randrange(8), rng.randrange(720))
        if op == "CHANGE_THRESHOLD":
            ref = node_refs(parent)[target]
            assert ref is not None
            old = parent.rules[ref.rule_index].conditions[ref.condition_index]
            action = EditAction(op, target, old.feature, old.cmp, old.index, random_condition(rng).value, old.route if hasattr(old, "route") else 0)
        try: g = apply_edit(parent, action, route_ids)
        except ValueError: continue
        if g.semantic_id() in seen: continue
        seen.add(g.semantic_id()); out.append(GeneratedCandidate("random_typed", g.semantic_id(), parent.semantic_id(), asdict(action), g.canonical()))
    return out


def heuristic_generate(parent: ModuleGenome, *, n: int, seed: int) -> list[GeneratedCandidate]:
    rng = random.Random(seed); seen = {parent.semantic_id()}; out = []
    for _ in range(n * 100):
        if len(out) >= n: break
        try: g = mutate(parent, rng); g.validate(ROUTES)
        except ValueError: continue
        sid = g.semantic_id()
        if sid in seen: continue
        seen.add(sid); out.append(GeneratedCandidate("heuristic_mutate", sid, parent.semantic_id(), {"operation": "EXISTING_MUTATE"}, g.canonical()))
    return out


def empirical_generate(parent: ModuleGenome, actor_data: dict[str, np.ndarray], *, n: int, seed: int) -> list[GeneratedCandidate]:
    # Empirical safe-positive operation-frequency baseline. Compact GCAC-v2 goal examples
    # map back to transition-level operations and cannot replay exact historical semantics.
    rng = random.Random(seed); tids=np.asarray(actor_data["transition_idx"],np.int32)
    ops=np.asarray(actor_data["actor_op"],np.int32)[tids];pos=np.asarray(actor_data["positive"],bool)
    counts = np.bincount(ops[pos], minlength=len(EDIT_OPERATIONS)).astype(float) + 0.1
    valid = np.asarray(valid_operation_mask(parent), bool); counts *= valid; counts /= counts.sum()
    masks = target_masks(parent); seen = {parent.semantic_id()}; out = []
    for _ in range(n * 100):
        if len(out) >= n: break
        oi = int(np.random.default_rng(seed + _).choice(len(counts), p=counts)); op = EDIT_OPERATIONS[oi]
        targets = [i for i, ok in enumerate(masks[oi]) if ok]; target = rng.choice(targets)
        c1, c2 = random_condition(rng), random_condition(rng)
        a = EditAction(op, target, c1.feature, c1.cmp, c1.index, c1.value, rng.choice(ROUTES), c2.feature, c2.cmp,
                       c2.index, c2.value, rng.choice((1, 2)), rng.randrange(8), rng.randrange(720))
        try: g = apply_edit(parent, a, ROUTES)
        except ValueError: continue
        sid = g.semantic_id()
        if sid in seen: continue
        seen.add(sid); out.append(GeneratedCandidate("empirical_ops", sid, parent.semantic_id(), asdict(a), g.canonical()))
    return out


def _genome_blobs(genomes: Sequence[ModuleGenome]) -> list[bytes]:
    p = OUT / "actor" / "tmp_feature_genomes.b4gm"; p.parent.mkdir(parents=True, exist_ok=True); pack_genomes(genomes, p)
    raw = p.read_bytes(); _, _, ng = FILE_HEAD.unpack_from(raw, 0)
    if ng != len(genomes): raise ValueError("packed genome count mismatch")
    return [raw[FILE_HEAD.size + i * GENOME_SIZE: FILE_HEAD.size + (i + 1) * GENOME_SIZE] for i in range(ng)]


def current_candidate_features(genomes: Sequence[ModuleGenome], cfg: EditFormerConfig,
                               baseline: np.ndarray | None = None) -> dict[str, np.ndarray]:
    base = np.asarray(current_rows()[0].baseline if baseline is None else baseline, np.int32).copy()
    blobs = _genome_blobs(genomes); rows = []
    for g, blob in zip(genomes, blobs):
        sha = hashlib.sha256(blob).hexdigest()
        rows.append(Row(len(base), blob, base.copy(), base.copy(), "actor_generated", sha, g.semantic_id(),
                        label_dict(base, base), {"heuristic_norm": 0.0, "heuristic_rank": 399}, 2))
    return _feature_arrays(rows, cfg)


def context_candidate_features(genomes: Sequence[ModuleGenome], cfg: EditFormerConfig,
                               context: dict) -> dict[str, np.ndarray]:
    """Build candidate features on a guarded external legal-state capture."""
    base = np.asarray(context["baseline_margins"], np.int32)
    blobs = _genome_blobs(genomes); rows = []
    for g, blob in zip(genomes, blobs):
        sha = hashlib.sha256(blob).hexdigest()
        rows.append(Row(len(base), blob, base.copy(), base.copy(), "gcac_goal_context", sha, g.semantic_id(),
                        label_dict(base, base), {"heuristic_norm": 0.0, "heuristic_rank": 399}, 2))
    return feature_arrays_from_capture(rows, cfg, context["capture"], case_meta_from_tsv(context["cases"]))


def candidate_features(genomes: Sequence[ModuleGenome], cfg: EditFormerConfig,
                       parent_margins: np.ndarray, context: dict | None = None) -> dict[str, np.ndarray]:
    if context is None:
        return current_candidate_features(genomes, cfg, baseline=parent_margins)
    expected = np.asarray(context["baseline_margins"], np.int32)
    if not np.array_equal(np.asarray(parent_margins, np.int32), expected):
        raise ValueError("goal-context baseline margins do not match requested synthesis parent")
    return context_candidate_features(genomes, cfg, context)


def probe_effect_profiles(parent: ModuleGenome, genomes: Sequence[ModuleGenome], cfg: EditFormerConfig,
                          parent_margins: np.ndarray, context: dict | None = None) -> list[dict[str, float | int]]:
    """Cheap legal-state effect proxy relative to the exact synthesis parent.

    This is intentionally not CUDA reward. It uses only frozen-parent public-state probes
    already present in the EditFormer feature pipeline. A candidate must still pass exact
    CUDA evaluation. Route divergence is more discriminative than raw activation because
    an activated branch that selects the same module route is often behaviorally inert.
    """
    parent_margins = np.asarray(parent_margins, np.int32)
    cache = candidate_features(genomes, cfg, parent_margins, context)
    parent_cache = candidate_features([parent], cfg, parent_margins, context)
    pmap = {
        int(cid): float(parent_cache["case_features"][0, j, 19])
        for j, cid in enumerate(parent_cache["case_ids"][0]) if int(cid) >= 0
    }
    profiles: list[dict[str, float | int]] = []
    for i in range(len(genomes)):
        activated = divergence = loss_target = win_exposure = valid = 0
        for j, cid0 in enumerate(cache["case_ids"][i]):
            cid = int(cid0)
            if cid < 0:
                continue
            valid += 1
            cf = cache["case_features"][i, j]
            activated += int(float(cf[4]) > 0.5)
            child_route = float(cf[19])
            parent_route = pmap.get(cid, 0.0)
            if abs(child_route - parent_route) > 1e-6:
                divergence += 1
                if parent_margins[cid] < 0:
                    loss_target += 1
                elif parent_margins[cid] > 0:
                    win_exposure += 1
        # Targeted change is useful; broad exposure to already-winning cases is risky.
        # Log scaling prevents broad overrides from dominating narrow specialists.
        effect_score = (
            1.25 * math.log1p(loss_target)
            - 0.75 * math.log1p(win_exposure)
            + 0.20 * math.log1p(divergence)
            + 0.05 * math.log1p(activated)
        )
        profiles.append({
            "activated_cases": activated,
            "route_divergence_cases": divergence,
            "loss_target_cases": loss_target,
            "win_exposure_cases": win_exposure,
            "effect_score": float(effect_score),
            "valid_probe_cases": valid,
        })
    return profiles


def critic_merit(c: GeneratedCandidate) -> float:
    return (
        float(c.critic_score if c.critic_score is not None else -1e9)
        - 4.0 * float(c.critic_regression_prob or 0.0)
        + 2.0 * float(c.critic_flip_prob or 0.0)
        + 0.50 * float(c.critic_margin_pred or 0.0)
        + 0.75 * float(c.critic_effect_mean_abs or 0.0)
    )


def _diverse_order(candidates: Sequence[GeneratedCandidate], diversity_penalty: float) -> list[GeneratedCandidate]:
    remaining = list(candidates); selected: list[GeneratedCandidate] = []; counts: dict[str, int] = {}
    while remaining:
        best_i = max(
            range(len(remaining)),
            key=lambda i: (
                critic_merit(remaining[i])
                + float(remaining[i].probe_effect_score or 0.0)
                - diversity_penalty * counts.get(str(remaining[i].action.get("operation")), 0),
                remaining[i].semantic_id,
            ),
        )
        c = remaining.pop(best_i); selected.append(c)
        op = str(c.action.get("operation")); counts[op] = counts.get(op, 0) + 1
    return selected


def directional_case_score(*, parent_margins: np.ndarray, case_ids: np.ndarray, case_mask: np.ndarray,
                           goal_mask: np.ndarray, direction_delta: np.ndarray,
                           improve_prob: np.ndarray, flip_prob: np.ndarray,
                           regress_prob: np.ndarray) -> dict[str, float | int]:
    """Summarize pair-relative directional heads for one candidate/goal.

    `direction_delta` is the append-only GCAC-v2 pair-relative head trained on
    child-minus-parent exact CUDA deltas. The proven frozen `case_delta` head is
    intentionally not an input here because its historical baseline semantics differ.
    """
    parent_margins = np.asarray(parent_margins, np.int32)
    case_ids = np.asarray(case_ids, np.int32)
    valid = np.asarray(case_mask, bool) & (case_ids >= 0)
    goal = np.asarray(goal_mask, bool) & valid
    dd = np.asarray(direction_delta, np.float32)
    ip = np.asarray(improve_prob, np.float32)
    fp = np.asarray(flip_prob, np.float32)
    rp = np.asarray(regress_prob, np.float32)
    if not np.any(goal):
        return {
            "goal_delta_norm": -1.0,
            "goal_delta_margin": -1e9,
            "goal_improve_prob": 0.0,
            "goal_flip_prob": 0.0,
            "protected_regress_prob_max": 1.0,
            "predicted_target_flips": 0,
            "predicted_win_crossings": 0,
        }
    raw = 5000.0 * np.arctanh(np.clip(dd, -0.999, 0.999))
    win_mask = np.asarray([
        bool(v and 0 <= int(cid) < len(parent_margins) and parent_margins[int(cid)] > 0)
        for cid, v in zip(case_ids, valid)
    ], bool)
    target_flips = 0
    win_crossings = 0
    for k, cid0 in enumerate(case_ids):
        cid = int(cid0)
        if cid < 0 or cid >= len(parent_margins) or not valid[k]:
            continue
        pred_margin = float(parent_margins[cid]) + float(raw[k])
        if goal[k] and parent_margins[cid] < 0 and pred_margin > 0:
            target_flips += 1
        if win_mask[k] and pred_margin <= 0:
            win_crossings += 1
    return {
        "goal_delta_norm": float(np.mean(dd[goal])),
        "goal_delta_margin": float(np.mean(raw[goal])),
        "goal_improve_prob": float(np.mean(ip[goal])),
        "goal_flip_prob": float(np.mean(fp[goal])),
        "protected_regress_prob_max": float(np.max(rp[win_mask])) if np.any(win_mask) else 0.0,
        "predicted_target_flips": int(target_flips),
        "predicted_win_crossings": int(win_crossings),
    }


def directional_shortlist(
    candidates: Sequence[GeneratedCandidate],
    budget: int,
    *,
    min_improve_prob: float = 0.50,
    max_regression_prob: float = 0.35,
    max_protected_regress_prob: float = 0.35,
    diversity_penalty: float = 0.25,
) -> tuple[list[GeneratedCandidate], dict[str, int | float]]:
    """LOSS_REPAIR shortlist: signed target direction first, absolute effect last."""
    xs = list(candidates)
    budget = max(0, min(int(budget), len(xs)))
    def delta(c): return float(c.critic_goal_delta_norm if c.critic_goal_delta_norm is not None else -1e9)
    def improve(c): return float(c.critic_goal_improve_prob if c.critic_goal_improve_prob is not None else 0.0)
    def flip(c): return float(c.critic_goal_flip_prob if c.critic_goal_flip_prob is not None else 0.0)
    def reg(c): return float(c.critic_regression_prob if c.critic_regression_prob is not None else 1.0)
    def preg(c): return float(c.critic_protected_regress_prob_max if c.critic_protected_regress_prob_max is not None else 1.0)
    def positive(c): return delta(c) > 0.0 and improve(c) >= float(min_improve_prob)
    def safe(c): return reg(c) <= float(max_regression_prob) and preg(c) <= float(max_protected_regress_prob)
    def likely_flip(c): return int(c.critic_predicted_target_flips or 0) > 0 or flip(c) >= 0.50

    # A positive target direction is non-negotiable for the exploitation frontier.
    # Within it, predicted flips and safety dominate magnitude. Absolute effect is last.
    pools = [
        [c for c in xs if positive(c) and likely_flip(c) and safe(c)],
        [c for c in xs if positive(c) and safe(c) and not likely_flip(c)],
        [c for c in xs if positive(c) and likely_flip(c) and not safe(c)],
        [c for c in xs if positive(c) and not likely_flip(c) and not safe(c)],
        [c for c in xs if not positive(c) and safe(c)],
        [c for c in xs if not positive(c) and not safe(c)],
    ]
    selected: list[GeneratedCandidate] = []
    used: set[str] = set()
    op_counts: dict[str, int] = {}
    for pool in pools:
        remaining = list(pool)
        while remaining and len(selected) < budget:
            best_i = max(range(len(remaining)), key=lambda i: (
                flip(remaining[i]),
                delta(remaining[i]),
                -reg(remaining[i]),
                -preg(remaining[i]),
                float(remaining[i].critic_worst_win_pred or 0.0),
                float(remaining[i].critic_margin_pred or 0.0),
                -diversity_penalty * op_counts.get(str(remaining[i].action.get("operation")), 0),
                float(remaining[i].critic_effect_mean_abs or 0.0),
                remaining[i].semantic_id,
            ))
            c = remaining.pop(best_i)
            if c.semantic_id in used:
                continue
            selected.append(c); used.add(c.semantic_id)
            op = str(c.action.get("operation")); op_counts[op] = op_counts.get(op, 0) + 1
        if len(selected) >= budget:
            break
    diag = {
        "input_candidates": len(xs), "budget": budget,
        "positive_direction_available": sum(positive(c) for c in xs),
        "safe_positive_direction_available": sum(positive(c) and safe(c) for c in xs),
        "predicted_flip_available": sum(positive(c) and likely_flip(c) for c in xs),
        "selected_positive_direction": sum(positive(c) for c in selected),
        "selected_safe_positive_direction": sum(positive(c) and safe(c) for c in selected),
        "selected_predicted_flip": sum(likely_flip(c) for c in selected),
        "selected_negative_or_uncertain_direction": sum(not positive(c) for c in selected),
        "mean_selected_goal_delta_norm": float(np.mean([delta(c) for c in selected])) if selected else 0.0,
        "mean_selected_regression_prob": float(np.mean([reg(c) for c in selected])) if selected else 0.0,
    }
    return selected, diag


def effect_aware_shortlist(candidates: Sequence[GeneratedCandidate], budget: int, *,
                           max_regression_prob: float = 0.35,
                           exploration_regression_prob: float = 0.75,
                           exploration_fraction: float = 0.15,
                           diversity_penalty: float = 0.25) -> tuple[list[GeneratedCandidate], dict[str, int | float]]:
    """Prefer safe edits predicted to touch parent losses; never maximize activity blindly.

    The old scalar ranker could achieve a high score by selecting edits that never changed
    the parent. This selector keeps regression risk as the primary gate, requires legal-state
    route divergence for the exploitation pool, and reserves only a small bounded fraction
    for uncertain-but-effectful exploration. Exact CUDA remains the sole reward source.
    """
    xs = list(candidates)
    def reg(c): return float(c.critic_regression_prob if c.critic_regression_prob is not None else 1.0)
    def probe_effect(c): return int(c.probe_route_divergence_cases or 0) > 0
    def targeted(c): return probe_effect(c) and int(c.probe_loss_target_cases or 0) > 0
    def strong_effect(c): return float(c.critic_effect_mean_abs or 0.0) >= 0.15
    safe_strong_targeted = [c for c in xs if targeted(c) and strong_effect(c) and reg(c) <= max_regression_prob]
    safe_weak_targeted = [c for c in xs if targeted(c) and not strong_effect(c) and reg(c) <= max_regression_prob]
    safe_effect = [c for c in xs if probe_effect(c) and not targeted(c) and reg(c) <= max_regression_prob]
    uncertain_strong_targeted = [c for c in xs if targeted(c) and strong_effect(c) and max_regression_prob < reg(c) <= exploration_regression_prob]
    safe_noeffect = [c for c in xs if not probe_effect(c) and reg(c) <= max_regression_prob]
    used: set[str] = set(); out: list[GeneratedCandidate] = []
    explore_n = min(budget, int(math.ceil(max(0.0, exploration_fraction) * budget)))
    exploit_n = max(0, budget - explore_n)
    # First take genuinely safe + strongly effective candidates. Do not allow weak
    # probe-only candidates to consume the budget before the explicit exploration quota.
    for c in _diverse_order(safe_strong_targeted, diversity_penalty):
        if len(out) >= exploit_n: break
        if c.semantic_id not in used: out.append(c); used.add(c.semantic_id)
    for c in _diverse_order(uncertain_strong_targeted, diversity_penalty):
        if len(out) >= budget or sum(max_regression_prob < reg(x) <= exploration_regression_prob and strong_effect(x) for x in out) >= explore_n: break
        if c.semantic_id not in used: out.append(c); used.add(c.semantic_id)
    for pool in (_diverse_order(safe_weak_targeted, diversity_penalty), _diverse_order(safe_effect, diversity_penalty)):
        for c in pool:
            if len(out) >= budget: break
            if c.semantic_id not in used: out.append(c); used.add(c.semantic_id)
        if len(out) >= budget: break
    # If the targeted/effectful frontier is too small, preserve equal CUDA budget but make
    # the fallback explicit in diagnostics instead of silently pretending no-ops are useful.
    remainder = sorted(xs, key=lambda c: (c.semantic_id in used, reg(c) > max_regression_prob, -critic_merit(c), c.semantic_id))
    for c in remainder:
        if len(out) >= budget: break
        if c.semantic_id not in used: out.append(c); used.add(c.semantic_id)
    diag = {
        "input_candidates": len(xs), "budget": budget,
        "safe_strong_targeted_available": len(safe_strong_targeted), "safe_weak_targeted_available": len(safe_weak_targeted),
        "safe_effect_available": len(safe_effect), "uncertain_strong_targeted_available": len(uncertain_strong_targeted), "safe_noeffect_available": len(safe_noeffect),
        "selected_route_divergent": sum(probe_effect(c) for c in out),
        "selected_loss_targeted": sum(targeted(c) for c in out),
        "selected_strong_effect": sum(strong_effect(c) for c in out),
        "selected_predicted_noeffect": sum(not probe_effect(c) for c in out),
        "selected_above_safety_gate": sum(reg(c) > max_regression_prob for c in out),
        "mean_selected_regression_prob": float(np.mean([reg(c) for c in out])) if out else 0.0,
        "no_safe_effective_frontier": len(safe_strong_targeted) == 0,
    }
    return out, diag


def critic_score_candidates(model, params, cfg: EditFormerConfig, candidates: list[GeneratedCandidate], *,
                            parent: ModuleGenome, parent_margins: np.ndarray,
                            batch_size: int = 256, context: dict | None = None) -> list[GeneratedCandidate]:
    from tools.bkg4_search.module_codec import genome_from_json
    genomes = [genome_from_json(c.genome) for c in candidates]
    parent_margins=np.asarray(parent_margins,np.int32)
    cache = candidate_features(genomes, cfg, parent_margins, context)
    profiles = probe_effect_profiles(parent, genomes, cfg, parent_margins, context)
    scores = []
    for s in range(0, len(candidates), batch_size):
        ids = np.arange(s, min(len(candidates), s + batch_size))
        b = {k: cache[k][ids] for k in ("tree_cat","tree_num","tree_mask","tree_parent","tree_depth","tree_rule","edit_cat","edit_num","edit_mask","case_features","case_mask","case_delta","heuristic","labels","group")}
        b["actor_op"] = np.zeros(len(ids), np.int32); b["actor_target"] = np.zeros(len(ids), np.int32)
        b["actor_target_mask"] = np.ones((len(ids), len(EDIT_OPERATIONS), cfg.tree_cap), bool)
        goal_masks=[];goal_specs=[]
        for ci in ids:
            requested=tuple(int(x) for x in (candidates[int(ci)].goal_case_ids or ()))
            if not requested:
                requested=tuple(int(x) for x in np.flatnonzero(parent_margins<0))
            mask=np.isin(np.asarray(cache["case_ids"][int(ci)],np.int32),np.asarray(requested,np.int32)) & np.asarray(cache["case_mask"][int(ci)],bool)
            if int(np.sum(mask))!=len(set(requested)):
                raise ValueError(f"goal case alignment failed for candidate {candidates[int(ci)].semantic_id}: requested={requested}")
            goal_masks.append(mask)
            mode=str(candidates[int(ci)].goal_mode or "LOSS_REPAIR")
            goal_specs.append([1.0 if mode=="LOSS_REPAIR" else 0.0,1.0 if mode=="MARGIN_EXPANSION" else 0.0,1.0,0.0])
        b["goal_mask"]=np.asarray(goal_masks,bool);b["goal_spec"]=np.asarray(goal_specs,np.float32)
        o = model.apply({"params": params}, _to_device(b), train=False)
        rank = np.asarray(o["rank_residual"]); reg = np.asarray(jax.nn.sigmoid(o["regression_logit"])); flip = np.asarray(jax.nn.sigmoid(o["flip_logit"]))
        margin = np.asarray(o["margin"]); worst = np.asarray(o["worst_win"]); case_delta = np.asarray(o["case_delta"], np.float32)
        direction_delta=np.asarray(o["actor_direction_delta"],np.float32)
        improve=np.asarray(jax.nn.sigmoid(o["actor_direction_improve_logit"]),np.float32)
        dflip=np.asarray(jax.nn.sigmoid(o["actor_direction_flip_logit"]),np.float32)
        dreg=np.asarray(jax.nn.sigmoid(o["actor_direction_regress_logit"]),np.float32)
        case_mask = np.asarray(cache["case_mask"][ids], bool);case_ids=np.asarray(cache["case_ids"][ids],np.int32);goal_mask=np.asarray(b["goal_mask"],bool)
        mean_abs = np.sum(np.abs(case_delta) * case_mask, axis=1) / np.maximum(1, np.sum(case_mask, axis=1))
        effect_cases = np.sum((np.abs(case_delta) >= 0.10) & case_mask, axis=1)
        for j in range(len(ids)):
            ds=directional_case_score(
                parent_margins=parent_margins,
                case_ids=case_ids[j],
                case_mask=case_mask[j],
                goal_mask=goal_mask[j],
                direction_delta=direction_delta[j],
                improve_prob=improve[j],
                flip_prob=dflip[j],
                regress_prob=dreg[j],
            )
            scores.append((float(rank[j]),float(reg[j]),float(flip[j]),float(margin[j]),float(worst[j]),float(mean_abs[j]),int(effect_cases[j]),
                           float(ds["goal_delta_norm"]),float(ds["goal_delta_margin"]),float(ds["goal_improve_prob"]),float(ds["goal_flip_prob"]),
                           float(ds["protected_regress_prob_max"]),int(ds["predicted_target_flips"]),int(ds["predicted_win_crossings"])))
    return [GeneratedCandidate(
        c.source, c.semantic_id, c.parent_semantic_id, c.action, c.genome, c.actor_logit,
        critic_score=sc[0], critic_regression_prob=sc[1], critic_flip_prob=sc[2],
        critic_margin_pred=sc[3], critic_worst_win_pred=sc[4], critic_effect_mean_abs=sc[5], critic_effect_cases=sc[6],
        probe_activated_cases=int(pr["activated_cases"]),
        probe_route_divergence_cases=int(pr["route_divergence_cases"]),
        probe_loss_target_cases=int(pr["loss_target_cases"]),
        probe_win_exposure_cases=int(pr["win_exposure_cases"]),
        probe_effect_score=float(pr["effect_score"]),
        goal_case_ids=tuple(int(x) for x in (c.goal_case_ids or ())),generated_goal_case_ids=tuple(int(x) for x in (c.generated_goal_case_ids or c.goal_case_ids or ())),goal_mode=c.goal_mode,
        critic_goal_delta_norm=sc[7],critic_goal_delta_margin=sc[8],critic_goal_improve_prob=sc[9],critic_goal_flip_prob=sc[10],
        critic_protected_regress_prob_max=sc[11],critic_predicted_target_flips=sc[12],critic_predicted_win_crossings=sc[13],
    ) for c, sc, pr in zip(candidates, scores, profiles)]


def write_candidates(path: str | Path, candidates: Sequence[GeneratedCandidate]) -> None:
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("".join(json.dumps(asdict(c), sort_keys=True) + "\n" for c in candidates))


def read_candidates(path: str | Path) -> list[GeneratedCandidate]:
    return [GeneratedCandidate(**json.loads(x)) for x in Path(path).read_text().splitlines() if x.strip()]
