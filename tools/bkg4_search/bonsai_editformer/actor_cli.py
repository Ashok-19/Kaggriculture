from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tarfile
import time
from collections import Counter
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
if str(HERE) not in sys.path: sys.path.insert(0, str(HERE))

from actor_data import (
    ACTOR_DATA_DIR,
    GoalClass,
    assert_no_relabel_leakage,
    load_actor_dataset,
    prepare_actor_dataset,
    row_genome,
    stratified_validation_indices,
)
from actor_generate import (
    GeneratedCandidate,
    actor_generate,
    critic_score_candidates,
    context_candidate_features,
    directional_shortlist,
    empirical_generate,
    heuristic_generate,
    random_typed_generate,
    read_candidates,
    write_candidates,
)
from actor_grammar import EDIT_OPERATIONS, FEATURES, OP_INDEX, apply_edit
from actor_model import (
    BonsaiEditActorCritic,
    actor_config_report,
    actor_param_mask,
    critic_param_digest,
    merge_critic_params,
)
from cli import environment_info, load_checkpoint, params_digest, save_checkpoint, tree_all_finite
from data import OUT, ROUTES, current_rows, history_rows, load_cache
from model import BonsaiEditFormer, config_for_size, count_params
from tools.bkg4_search.cuda_io import load_results, pack_genomes
from tools.bkg4_search.module_codec import ModuleGenome, distill_module, genome_from_json
from gcac_context import load_goal_context_manifest

ARCH = "BonsaiEditFormer-GCAC-v2"
EXPECTED_PARAMS_5M = 6_187_787
DEFAULT_BASE_SEMANTIC = "b3714ce680b84a74233b053c5bc32f48a37e3c257b46b4e131a0f784b4534286"
NUMERIC_KEYS = ("tree_cat","tree_num","tree_mask","tree_parent","tree_depth","tree_rule","edit_cat","edit_num","edit_mask","case_features","case_mask","case_delta","heuristic","labels","group")


def _to_jax(batch):
    import jax.numpy as jnp
    ints = {"tree_cat","edit_cat","tree_parent","tree_depth","tree_rule","group","actor_op","actor_target",
            "feature","cmp","route","index","feature2","cmp2","index2","condition_count","priority",
            "goal_case_id","goal_class","goal_mode"}
    bools = {"tree_mask","edit_mask","case_mask","actor_target_mask","actor_supported","goal_mask",
             "direction_mask","direction_improve","direction_flip","direction_regress","direction_reg3"}
    return {k:jnp.asarray(v, jnp.int32 if k in ints else bool if k in bools else jnp.float32) for k,v in batch.items()}


def _goal_mask(case_ids: np.ndarray, goal_case_ids: np.ndarray) -> np.ndarray:
    case_ids = np.asarray(case_ids, np.int32)
    goal_case_ids = np.asarray(goal_case_ids, np.int32)
    mask = case_ids == goal_case_ids[:, None]
    counts = np.sum(mask, axis=1)
    if np.any(counts != 1):
        bad = np.flatnonzero(counts != 1)[:8]
        raise ValueError(f"goal case missing/duplicated in case tokens for batch rows {bad.tolist()}: counts={counts[bad].tolist()}")
    return mask


def _gather_pair_by_case(ds, tids: np.ndarray, case_ids: np.ndarray, key: str, default=0):
    src = np.asarray(ds[key])[tids]
    valid = np.asarray(case_ids) >= 0
    safe = np.clip(np.asarray(case_ids, np.int32), 0, src.shape[1] - 1)
    gathered = np.take_along_axis(src, safe, axis=1)
    return np.where(valid, gathered, default)


def actor_batch(cache, ds, ids, *, child: bool = False):
    """Build one GCAC-v2 batch; ids address compact goal examples, not transitions."""
    ids = np.asarray(ids, np.int32)
    tids = np.asarray(ds["transition_idx"][ids], np.int32)
    row_idx = np.asarray(ds["child_idx" if child else "parent_idx"][tids], np.int32)
    b = {k: np.asarray(cache[k][row_idx]).copy() for k in NUMERIC_KEYS}
    case_ids = np.asarray(cache["case_ids"][row_idx], np.int32)
    parent_margin = _gather_pair_by_case(ds, tids, case_ids, "pair_parent_margin", 0).astype(np.int32)
    valid_case = (case_ids >= 0) & np.asarray(b["case_mask"], bool)
    # The frozen feature cache was built relative to the campaign baseline. Goal relabels
    # are relative to the semantic mutation parent, so patch only outcome fields 0..2.
    cf = np.asarray(b["case_features"], np.float32)
    cf[..., 0] = np.tanh(parent_margin / 20000.0)
    cf[..., 1] = (parent_margin > 0).astype(np.float32)
    cf[..., 2] = (parent_margin < 0).astype(np.float32)
    b["case_features"] = cf

    for k in ("actor_op","actor_target","actor_target_mask","feature","cmp","route","index","feature2","cmp2","index2",
              "condition_count","priority","value_norm","value2_norm","prefix_norm"):
        b[k] = np.asarray(ds[k][tids])
    for k in ("goal_case_id","goal_class","goal_mode","target_delta_raw","target_delta_norm","target_improve","target_flip",
              "positive","weight","advantage","w2l","reg3","desired_margin_floor"):
        b[k] = np.asarray(ds[k][ids])
    learned_mask = np.asarray(ds.get("learned_operation_mask", np.ones(len(EDIT_OPERATIONS), bool)), bool)
    op = np.asarray(b["actor_op"], np.int32)
    b["actor_supported"] = learned_mask[op]
    b["actor_operation_mask"] = np.broadcast_to(learned_mask[None, :], (len(ids), len(learned_mask))).copy()
    b["goal_mask"] = _goal_mask(case_ids, np.asarray(b["goal_case_id"], np.int32)) & valid_case
    mode = np.asarray(b["goal_mode"], np.int32)
    floor = np.asarray(b["desired_margin_floor"], np.float32)
    b["goal_spec"] = np.stack([
        (mode == 0).astype(np.float32), (mode == 1).astype(np.float32),
        np.ones(len(ids), np.float32), np.tanh(floor / 20000.0),
    ], axis=1)
    b["direction_mask"] = _gather_pair_by_case(ds, tids, case_ids, "pair_case_mask", False).astype(bool) & valid_case
    b["direction_delta_norm"] = _gather_pair_by_case(ds, tids, case_ids, "pair_case_delta_norm", 0.0).astype(np.float32)
    b["direction_delta_raw"] = _gather_pair_by_case(ds, tids, case_ids, "pair_case_delta_raw", 0).astype(np.float32)
    b["direction_improve"] = _gather_pair_by_case(ds, tids, case_ids, "pair_case_improve", False).astype(bool)
    b["direction_flip"] = _gather_pair_by_case(ds, tids, case_ids, "pair_case_flip", False).astype(bool)
    b["direction_regress"] = _gather_pair_by_case(ds, tids, case_ids, "pair_case_regress", False).astype(bool)
    b["direction_reg3"] = _gather_pair_by_case(ds, tids, case_ids, "pair_case_reg3", False).astype(bool)
    return _to_jax(b)


def init_actor(cfg, batch, seed=73):
    import jax
    model = BonsaiEditActorCritic(cfg); key = jax.random.PRNGKey(seed); key, drop = jax.random.split(key)
    params = model.init({"params":key,"dropout":drop}, batch, train=False)["params"]
    return model, params


def import_critic(cfg, batch, critic_checkpoint, seed=73):
    model, actor_params = init_actor(cfg, batch, seed)
    critic = load_checkpoint(critic_checkpoint)["params"]
    merged, report = merge_critic_params(actor_params, critic)
    return model, merged, critic, report


def _ce(logits, target):
    import optax
    return optax.softmax_cross_entropy_with_integer_labels(logits, target)


def actor_loss(outputs, batch):
    import jax.numpy as jnp, optax
    pos = batch["positive"]; weight = batch["weight"]
    supported = batch.get("actor_supported", jnp.ones_like(pos, dtype=bool)).astype(jnp.float32)
    pw = pos * weight * supported; denom = jnp.maximum(jnp.sum(pw), 1.0)
    op = batch["actor_op"]; target = batch["actor_target"]
    target_logits = outputs["actor_target_logits_all"][jnp.arange(op.shape[0]), op]
    losses = {
        "op": jnp.sum(_ce(outputs["actor_op_logits"], op) * pw) / denom,
        "target": jnp.sum(_ce(target_logits, target) * pw) / denom,
    }
    semantic_ops = jnp.isin(op, jnp.asarray([OP_INDEX[x] for x in ("ADD_SPECIALIST","ADD_CONDITION","INSERT_RECOVERY","SPLIT_LANE","COMPOSE_PREDICATES","REPLACE_SUBTREE")]))
    route_ops = jnp.isin(op, jnp.asarray([OP_INDEX[x] for x in ("ADD_SPECIALIST","INSERT_RECOVERY","SPLIT_LANE","CHANGE_ROUTE","REPLACE_MACRO","REPLACE_SUBTREE")]))
    value_ops = semantic_ops | (op == OP_INDEX["CHANGE_THRESHOLD"])
    priority_ops = op == OP_INDEX["CHANGE_PRIORITY"]
    prefix_ops = op == OP_INDEX["CHANGE_WINDOW"]
    def masked(loss_vec, mask):
        w = pw * mask.astype(jnp.float32); return jnp.sum(loss_vec * w) / jnp.maximum(jnp.sum(w), 1.0)
    losses["feature"] = masked(_ce(outputs["actor_feature_logits"], batch["feature"]), semantic_ops)
    losses["cmp"] = masked(_ce(outputs["actor_cmp_logits"], batch["cmp"]), semantic_ops)
    losses["index"] = masked(_ce(outputs["actor_index_logits"], batch["index"]), semantic_ops)
    losses["route"] = masked(_ce(outputs["actor_route_logits"], batch["route"]), route_ops)
    losses["condition_count"] = masked(_ce(outputs["actor_condition_count_logits"], batch["condition_count"]), semantic_ops)
    losses["feature2"] = masked(_ce(outputs["actor_feature2_logits"], batch["feature2"]), semantic_ops & (batch["condition_count"] > 0))
    losses["cmp2"] = masked(_ce(outputs["actor_cmp2_logits"], batch["cmp2"]), semantic_ops & (batch["condition_count"] > 0))
    losses["index2"] = masked(_ce(outputs["actor_index2_logits"], batch["index2"]), semantic_ops & (batch["condition_count"] > 0))
    losses["value"] = masked(optax.huber_loss(outputs["actor_value_norm"], batch["value_norm"], delta=0.15), value_ops)
    losses["value2"] = masked(optax.huber_loss(outputs["actor_value2_norm"], batch["value2_norm"], delta=0.15), semantic_ops & (batch["condition_count"] > 0))
    losses["priority"] = masked(_ce(outputs["actor_priority_logits"], batch["priority"]), priority_ops)
    losses["prefix"] = masked(optax.huber_loss(outputs["actor_prefix_norm"], batch["prefix_norm"], delta=0.15), prefix_ops)
    quality_target = pos.astype(jnp.float32)
    qw = weight * supported
    losses["quality"] = jnp.sum(optax.sigmoid_binary_cross_entropy(outputs["actor_proposal_quality_logit"], quality_target) * qw) / jnp.maximum(jnp.sum(qw), 1.0)
    # Target-specific harmful/no-effect edits reduce the exact operation+target likelihood
    # under this semantic failure goal. Unsupported operations are masked from Stage-B2.
    neg = (1.0 - pos) * jnp.minimum(weight, 4.0) * supported; nden = jnp.maximum(jnp.sum(neg), 1.0)
    op_p = jnp.exp(-_ce(outputs["actor_op_logits"], op)); target_p = jnp.exp(-_ce(target_logits, target))
    losses["negative"] = jnp.sum((-jnp.log(jnp.maximum(1e-5, 1.0 - op_p * target_p))) * neg) / nden
    total = losses["op"] + losses["target"] + 0.8*(losses["feature"]+losses["cmp"]+losses["route"]+losses["value"]) + \
            0.35*(losses["index"]+losses["condition_count"]+losses["feature2"]+losses["cmp2"]+losses["index2"]+losses["value2"]) + \
            0.5*(losses["priority"]+losses["prefix"]) + 0.5*losses["quality"] + 0.15*losses["negative"]
    return total, losses


def directional_loss(outputs, batch):
    """Pair-relative signed/calibration loss for append-only directional case heads."""
    import jax.numpy as jnp, optax
    mask = batch["direction_mask"].astype(jnp.float32)
    example_weight = batch.get("weight", jnp.ones((mask.shape[0],), jnp.float32)).astype(jnp.float32)
    w = mask * example_weight[:, None]
    denom = jnp.maximum(jnp.sum(w), 1.0)
    parts = {
        "delta": jnp.sum(optax.huber_loss(outputs["actor_direction_delta"], batch["direction_delta_norm"].astype(jnp.float32), delta=0.20) * w) / denom,
        "improve": jnp.sum(optax.sigmoid_binary_cross_entropy(outputs["actor_direction_improve_logit"], batch["direction_improve"].astype(jnp.float32)) * w) / denom,
        "flip": jnp.sum(optax.sigmoid_binary_cross_entropy(outputs["actor_direction_flip_logit"], batch["direction_flip"].astype(jnp.float32)) * w) / denom,
        "regress": jnp.sum(optax.sigmoid_binary_cross_entropy(outputs["actor_direction_regress_logit"], batch["direction_regress"].astype(jnp.float32)) * w) / denom,
    }
    total = parts["delta"] + 0.50 * parts["improve"] + 0.75 * parts["flip"] + 0.75 * parts["regress"]
    return total, parts


def _np_sigmoid(x):
    x = np.asarray(x, np.float64)
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(x)))


def _rankdata_np(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, np.float64)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), np.float64); ranks[order] = np.arange(len(x), dtype=np.float64)
    return ranks


def _corr_np(a, b, *, rank=False) -> float:
    a = np.asarray(a, np.float64); b = np.asarray(b, np.float64)
    if len(a) < 3 or np.std(a) <= 1e-12 or np.std(b) <= 1e-12:
        return 0.0
    if rank:
        a = _rankdata_np(a); b = _rankdata_np(b)
    return float(np.corrcoef(a, b)[0, 1])


def actor_metrics(outputs, batch):
    op = np.asarray(batch["actor_op"]); pos = np.asarray(batch["positive"]) > 0
    supported = np.asarray(batch.get("actor_supported", np.ones(len(op), bool)), bool)
    pred_op = np.argmax(np.asarray(outputs["actor_op_logits"]), axis=-1)
    tl = np.asarray(outputs["actor_target_logits_all"])[np.arange(len(op)), op]; pred_target = np.argmax(tl, axis=-1)
    op_logits = np.asarray(outputs["actor_op_logits"], np.float64)
    op_logp = op_logits - np.log(np.sum(np.exp(op_logits - np.max(op_logits, axis=1, keepdims=True)), axis=1))[:, None] - np.max(op_logits, axis=1, keepdims=True)
    # Stable target log-softmax for the teacher-forced operation.
    mx = np.max(tl, axis=1, keepdims=True); target_logp = tl - mx - np.log(np.sum(np.exp(tl - mx), axis=1, keepdims=True))
    target = np.asarray(batch["actor_target"], np.int32)
    joint = op_logp[np.arange(len(op)), op] + target_logp[np.arange(len(op)), target]
    hard = np.isin(np.asarray(batch["goal_class"], np.int32), [int(GoalClass.TARGET_HARM), int(GoalClass.SEVERE_REGRESSION), int(GoalClass.UNSAFE_IMPROVE)]) & supported
    useful = pos & supported
    return {
        "n": int(len(op)), "positive": int(pos.sum()), "supported_positive": int(useful.sum()), "hard_negative": int(hard.sum()),
        "op_accuracy_positive": float(np.mean(pred_op[useful] == op[useful])) if np.any(useful) else 0.0,
        "target_accuracy_positive": float(np.mean(pred_target[useful] == target[useful])) if np.any(useful) else 0.0,
        "quality_positive_mean": float(np.mean(_np_sigmoid(np.asarray(outputs["actor_proposal_quality_logit"])[useful]))) if np.any(useful) else 0.0,
        "quality_negative_mean": float(np.mean(_np_sigmoid(np.asarray(outputs["actor_proposal_quality_logit"])[hard]))) if np.any(hard) else 0.0,
        "joint_logp_useful_mean": float(np.mean(joint[useful])) if np.any(useful) else 0.0,
        "joint_logp_hard_negative_mean": float(np.mean(joint[hard])) if np.any(hard) else 0.0,
    }


def directional_metrics(outputs, batch):
    goal = np.asarray(batch["goal_mask"], bool)
    target_delta_pred = np.sum(np.asarray(outputs["actor_direction_delta"], np.float32) * goal, axis=1)
    target_delta_true = np.asarray(batch["target_delta_norm"], np.float32)
    improve_p = np.sum(_np_sigmoid(np.asarray(outputs["actor_direction_improve_logit"])) * goal, axis=1)
    flip_p = np.sum(_np_sigmoid(np.asarray(outputs["actor_direction_flip_logit"])) * goal, axis=1)
    improve_y = np.asarray(batch["target_improve"], bool)
    flip_y = np.asarray(batch["target_flip"], bool)
    dm = np.asarray(batch["direction_mask"], bool)
    regress_p = _np_sigmoid(np.asarray(outputs["actor_direction_regress_logit"]))[dm]
    regress_y = np.asarray(batch["direction_regress"], bool)[dm]
    return {
        "target_delta_pearson": _corr_np(target_delta_pred, target_delta_true),
        "target_delta_spearman": _corr_np(target_delta_pred, target_delta_true, rank=True),
        "target_delta_pred_mean": float(np.mean(target_delta_pred)) if len(target_delta_pred) else 0.0,
        "target_delta_true_mean": float(np.mean(target_delta_true)) if len(target_delta_true) else 0.0,
        "target_improve_accuracy": float(np.mean((improve_p >= 0.5) == improve_y)) if len(improve_y) else 0.0,
        "target_improve_prob_positive": float(np.mean(improve_p[improve_y])) if np.any(improve_y) else 0.0,
        "target_improve_prob_negative": float(np.mean(improve_p[~improve_y])) if np.any(~improve_y) else 0.0,
        "target_flip_accuracy": float(np.mean((flip_p >= 0.5) == flip_y)) if len(flip_y) else 0.0,
        "target_flip_prob_positive": float(np.mean(flip_p[flip_y])) if np.any(flip_y) else 0.0,
        "target_flip_prob_negative": float(np.mean(flip_p[~flip_y])) if np.any(~flip_y) else 0.0,
        "case_regress_accuracy": float(np.mean((regress_p >= 0.5) == regress_y)) if len(regress_y) else 0.0,
        "case_regress_prob_positive": float(np.mean(regress_p[regress_y])) if np.any(regress_y) else 0.0,
        "case_regress_prob_negative": float(np.mean(regress_p[~regress_y])) if np.any(~regress_y) else 0.0,
    }


TRAIN_CLASS_FRACTIONS = {
    GoalClass.SAFE_FLIP: 0.35,
    GoalClass.SAFE_IMPROVE: 0.30,
    GoalClass.TARGET_HARM: 0.10,
    GoalClass.SEVERE_REGRESSION: 0.10,
    GoalClass.NO_EFFECT: 0.10,
    GoalClass.UNSAFE_IMPROVE: 0.05,
}


def sample_training_examples(ds, train_ids, n: int, rng: np.random.Generator) -> np.ndarray:
    """Class-prioritized sampling keeps loss-repair signal dense without scalarizing safety."""
    train_ids = np.asarray(train_ids, np.int32)
    n = int(n)
    if n <= 0 or n >= len(train_ids):
        return rng.permutation(train_ids)
    cls = np.asarray(ds["goal_class"], np.int8)
    chunks = []
    allocated = 0
    items = list(TRAIN_CLASS_FRACTIONS.items())
    for j, (goal_cls, frac) in enumerate(items):
        want = n - allocated if j == len(items) - 1 else int(round(n * frac))
        pool = train_ids[cls[train_ids] == int(goal_cls)]
        if len(pool) == 0:
            continue
        chunks.append(rng.choice(pool, size=want, replace=len(pool) < want))
        allocated += want
    if allocated < n:
        chunks.append(rng.choice(train_ids, size=n - allocated, replace=len(train_ids) < (n - allocated)))
    out = np.concatenate(chunks)[:n].astype(np.int32)
    return rng.permutation(out)


def balanced_validation_examples(ds, val_ids, max_samples: int):
    val_ids = np.asarray(val_ids, np.int32)
    transition = np.asarray(ds["transition_idx"], np.int32)
    example_ops = np.asarray(ds["actor_op"], np.int32)[transition]
    ids, counts = stratified_validation_indices(
        val_ids, np.asarray(ds["goal_class"], np.int8), example_ops, max_samples=max_samples,
    )
    return ids, counts


def make_actor_tx(params, lr, wd, clip):
    import jax, optax
    mask = actor_param_mask(params)
    labels = jax.tree.map(lambda m: "actor" if bool(m) else "frozen", mask)
    return optax.multi_transform({"actor":optax.chain(optax.clip_by_global_norm(clip),optax.adamw(lr,weight_decay=wd)),"frozen":optax.set_to_zero()}, labels)


def cmd_prepare(args):
    report = prepare_actor_dataset(
        Path(args.out), limit_rows=args.limit_rows, max_transitions=args.max_transitions,
        min_safe_positive=args.min_safe_positive, awr_beta=args.awr_beta,
        max_positive_weight=args.max_positive_weight, max_negative_weight=args.max_negative_weight,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


def _load_train_context(args):
    ds = load_actor_dataset(args.dataset); cache = load_cache("historical", True); train = ds["train_idx"]; val = ds["val_idx"]
    sample = actor_batch(cache, ds, train[:min(args.batch_size, len(train))]); cfg = config_for_size("5m", attention_impl=args.attention_impl)
    return ds, cache, train, val, sample, cfg


def cmd_init_critic(args):
    ds,cache,train,val,sample,cfg = _load_train_context(args); model,params,critic,merge = import_critic(cfg,sample,args.critic_checkpoint,args.seed)
    count = count_params(params)
    if count != EXPECTED_PARAMS_5M: raise ValueError(f"GCAC-v2 param mismatch {count} != {EXPECTED_PARAMS_5M}")
    out = model.apply({"params":params}, sample, train=False); critic_model = BonsaiEditFormer(cfg); co = critic_model.apply({"params":critic}, sample, train=False)
    keys=("regression_logit","flip_logit","margin","worst_win","rank_residual","rank","case_delta")
    exact=all(np.array_equal(np.asarray(out[k]),np.asarray(co[k])) for k in keys)
    if not exact: raise ValueError("critic outputs changed after Actor-Critic import")
    meta={"architecture":ARCH,"parameter_count":count,"critic_checkpoint":str(Path(args.critic_checkpoint).resolve()),"critic_checkpoint_params_digest":params_digest(critic),"merge":merge,"critic_output_exact":exact,"environment":environment_info()}
    save_checkpoint(Path(args.out),params,None,meta)
    print(json.dumps(meta,indent=2,sort_keys=True))


def cmd_smoke(args):
    ds,cache,train,val,sample,cfg = _load_train_context(args); model,params,critic,merge = import_critic(cfg,sample,args.critic_checkpoint,args.seed)
    actual_count=count_params(params)
    if EXPECTED_PARAMS_5M and actual_count != EXPECTED_PARAMS_5M: raise ValueError(f"parameter count {actual_count} != {EXPECTED_PARAMS_5M}")
    before_critic = critic_param_digest(params)
    import jax, optax
    tx=make_actor_tx(params,args.lr,args.weight_decay,args.clip); state=tx.init(params)
    @jax.jit
    def step(p,s,pb,cb,key):
        def obj(q):
            key1,key2=jax.random.split(key)
            po=model.apply({"params":q},pb,train=True,rngs={"dropout":key1}); aloss,aparts=actor_loss(po,pb)
            co=model.apply({"params":q},cb,train=True,rngs={"dropout":key2}); dloss,dparts=directional_loss(co,cb)
            return aloss + args.direction_loss_weight*dloss, (aparts,dparts)
        (loss,parts),grads=jax.value_and_grad(obj,has_aux=True)(p); updates,s=tx.update(grads,s,p); p=optax.apply_updates(p,updates); return p,s,loss,parts,optax.global_norm(grads)
    rng=np.random.default_rng(args.seed)
    smoke_ids=sample_training_examples(ds,train,min(args.batch_size,len(train)),rng)
    batch=actor_batch(cache,ds,smoke_ids); child_batch=actor_batch(cache,ds,smoke_ids,child=True)
    key=jax.random.PRNGKey(args.seed+99); losses=[]; norms=[]
    for _ in range(args.steps):
        key,sk=jax.random.split(key);params,state,l,parts,gn=step(params,state,batch,child_batch,sk);l.block_until_ready();losses.append(float(l));norms.append(float(gn))
    after_critic=critic_param_digest(params)
    out=model.apply({"params":params},batch,train=False); child_out=model.apply({"params":params},child_batch,train=False)
    m=actor_metrics(out,{k:np.asarray(v) for k,v in batch.items()}); dm=directional_metrics(child_out,{k:np.asarray(v) for k,v in child_batch.items()})
    val_ids,val_counts=balanced_validation_examples(ds,val,min(args.eval_samples,len(val)))
    # Same semantic mutation with two different failure goals must alter all actor stages.
    goal_sensitive=False; goal_diffs={}
    trans=np.asarray(ds["transition_idx"],np.int32); valset=set(int(x) for x in train)
    for tid in np.unique(trans[train]):
        ex=[i for i in np.flatnonzero(trans==tid) if int(i) in valset]
        if len(ex)>=2 and int(ds["goal_case_id"][ex[0]])!=int(ds["goal_case_id"][ex[1]]):
            g0=actor_batch(cache,ds,[ex[0]]);g1=actor_batch(cache,ds,[ex[1]])
            o0=model.apply({"params":params},g0,train=False);o1=model.apply({"params":params},g1,train=False)
            for k in ("actor_op_logits","actor_target_logits_all","actor_feature_logits"):
                goal_diffs[k]=float(np.max(np.abs(np.asarray(o0[k])-np.asarray(o1[k]))))
            goal_sensitive=all(v>1e-7 for v in goal_diffs.values());break
    # Grammar/pack smoke remains independent of learned policy quality.
    _,parent,_=_current_parent(DEFAULT_BASE_SEMANTIC);generated=random_typed_generate(parent,n=32,seed=args.seed)
    genomes=[genome_from_json(x.genome) for x in generated];pack=OUT/"actor/smoke_generated.b4gm";pack.parent.mkdir(parents=True,exist_ok=True);pack_genomes(genomes,pack);pack_sha=hashlib.sha256(pack.read_bytes()).hexdigest()
    ck=OUT/"actor/smoke_actor_checkpoint";save_checkpoint(ck,params,state,{"architecture":ARCH,"smoke":True});restored=load_checkpoint(ck)["params"]
    report={"architecture":actor_config_report(cfg,params),"merge":merge,"losses":losses,"loss_decreased":losses[-1]<losses[0],"finite_gradients":bool(np.isfinite(norms).all()),"gradient_norms":norms,
            "critic_frozen_exact":before_critic==after_critic,"checkpoint_roundtrip":params_digest(params)==params_digest(restored),"actor_metrics":m,"directional_metrics":dm,
            "validation_class_counts":val_counts,"goal_sensitive":goal_sensitive,"goal_logit_max_abs_diffs":goal_diffs,
            "learned_operation_mask":{name:bool(ds["learned_operation_mask"][i]) for i,name in enumerate(EDIT_OPERATIONS)},
            "generated_valid":len(generated),"generated_unique":len({x.semantic_id for x in generated}),"packed_genomes_sha256":pack_sha,"environment":environment_info()}
    print(json.dumps(report,indent=2,sort_keys=True))
    required=(report["loss_decreased"] and report["finite_gradients"] and report["critic_frozen_exact"] and report["checkpoint_roundtrip"] and report["goal_sensitive"] and val_counts["target_harm"]>0 and val_counts["severe_regression"]>0 and len(generated)>=16)
    if not required: raise SystemExit(2)


def cmd_train(args):
    ds,cache,train,val,sample,cfg=_load_train_context(args);model,params=init_actor(cfg,sample,args.seed);init=load_checkpoint(args.checkpoint)
    from flax.core import freeze
    params=freeze(init["params"]);actual_count=count_params(params)
    if EXPECTED_PARAMS_5M and actual_count!=EXPECTED_PARAMS_5M:raise ValueError(f"parameter count {actual_count} != {EXPECTED_PARAMS_5M}")
    tx=make_actor_tx(params,args.lr,args.weight_decay,args.clip);state=tx.init(params);start_updates=0
    if args.resume and "opt_state" in init:state=init["opt_state"];start_updates=int(np.asarray(init.get("updates",0)))
    frozen_digest=critic_param_digest(params)
    import jax,optax
    @jax.jit
    def step(p,s,pb,cb,key):
        def obj(q):
            key1,key2=jax.random.split(key)
            po=model.apply({"params":q},pb,train=True,rngs={"dropout":key1});aloss,aparts=actor_loss(po,pb)
            co=model.apply({"params":q},cb,train=True,rngs={"dropout":key2});dloss,dparts=directional_loss(co,cb)
            return aloss+args.direction_loss_weight*dloss,(aparts,dparts)
        (loss,parts),grads=jax.value_and_grad(obj,has_aux=True)(p);updates,s=tx.update(grads,s,p);p=optax.apply_updates(p,updates);return p,s,loss,parts,optax.global_norm(grads)
    rng=np.random.default_rng(args.seed);key=jax.random.PRNGKey(args.seed+1000);outdir=Path(args.out);outdir.mkdir(parents=True,exist_ok=True);history=[];updates=start_updates;t0=time.perf_counter()
    val_ids,val_counts=balanced_validation_examples(ds,val,min(args.eval_samples,len(val)))
    def evaluate(epoch,tag):
        pb=actor_batch(cache,ds,val_ids);cb=actor_batch(cache,ds,val_ids,child=True)
        po=model.apply({"params":params},pb,train=False);co=model.apply({"params":params},cb,train=False)
        aloss,aparts=actor_loss(po,pb);dloss,dparts=directional_loss(co,cb);loss=aloss+args.direction_loss_weight*dloss
        rec={"epoch":epoch,"tag":tag,"loss":float(loss),"actor_loss":float(aloss),"directional_loss":float(dloss),
             "parts":{"actor":{k:float(v) for k,v in aparts.items()},"directional":{k:float(v) for k,v in dparts.items()}},
             "validation_class_counts":val_counts,"actor_metrics":actor_metrics(po,{k:np.asarray(v) for k,v in pb.items()}),
             "directional_metrics":directional_metrics(co,{k:np.asarray(v) for k,v in cb.items()})}
        print("METRIC",json.dumps(rec,sort_keys=True),flush=True);return rec
    history.append(evaluate(0,"init"))
    for ep in range(1,args.epochs+1):
        order=sample_training_examples(ds,train,args.examples_per_epoch,rng);el=[];eg=[]
        for s in range(0,len(order),args.batch_size):
            ids=order[s:s+args.batch_size]
            if len(ids)<args.batch_size:ids=np.resize(ids,args.batch_size)
            pb=actor_batch(cache,ds,ids);cb=actor_batch(cache,ds,ids,child=True);key,sk=jax.random.split(key);params,state,l,parts,gn=step(params,state,pb,cb,sk);l.block_until_ready();el.append(float(l));eg.append(float(gn));updates+=1
        print(json.dumps({"phase":"goal_conditioned_actor_warmstart","epoch":ep,"loss":float(np.mean(el)),"grad_norm":float(np.mean(eg)),"updates":updates,"examples":len(order)}),flush=True)
        if ep in (args.early_epoch,args.middle_epoch,args.epochs):
            tag="early" if ep==args.early_epoch else "middle" if ep==args.middle_epoch else "final";history.append(evaluate(ep,tag));save_checkpoint(outdir/tag,params,state,{"architecture":ARCH,"epoch":ep,"updates":updates,"critic_frozen_digest":frozen_digest},extra={"updates":jax.numpy.asarray(updates,jax.numpy.int32),"rng":key})
    if critic_param_digest(params)!=frozen_digest:raise ValueError("critic/shared encoder changed during frozen Stage-B2 warm-start")
    result={"architecture":ARCH,"params":count_params(params),"epochs":args.epochs,"updates":updates,"seconds":time.perf_counter()-t0,"critic_frozen_exact":True,"critic_digest":frozen_digest,
            "dataset":str(Path(args.dataset).resolve()),"dataset_sha256":hashlib.sha256(Path(args.dataset).read_bytes()).hexdigest(),"validation_class_counts":val_counts,
            "history":history,"environment":environment_info(),"config":vars(args)|{"fn":None}}
    result["config"].pop("fn",None);(outdir/"actor_training_result.json").write_text(json.dumps(result,indent=2,sort_keys=True,default=str)+"\n");print("RESULT",json.dumps(result,sort_keys=True,default=str))


def cmd_evaluate_actor(args):
    ds=load_actor_dataset(args.dataset);cache=load_cache("historical",True);val=np.asarray(ds["val_idx"],np.int32)
    ids,counts=balanced_validation_examples(ds,val,min(args.eval_samples,len(val)));cfg=config_for_size("5m",attention_impl=args.attention_impl);model=BonsaiEditActorCritic(cfg);ck=load_checkpoint(args.checkpoint);params=ck["params"]
    if count_params(params)!=EXPECTED_PARAMS_5M:raise ValueError(f"GCAC-v2 parameter mismatch {count_params(params)} != {EXPECTED_PARAMS_5M}")
    pb=actor_batch(cache,ds,ids);cb=actor_batch(cache,ds,ids,child=True);po=model.apply({"params":params},pb,train=False);co=model.apply({"params":params},cb,train=False)
    aloss,aparts=actor_loss(po,pb);dloss,dparts=directional_loss(co,cb)
    report={"architecture":ARCH,"params":count_params(params),"checkpoint":str(Path(args.checkpoint).resolve()),"dataset":str(Path(args.dataset).resolve()),"dataset_sha256":hashlib.sha256(Path(args.dataset).read_bytes()).hexdigest(),"validation_class_counts":counts,"actor_loss":float(aloss),"directional_loss":float(dloss),"actor_parts":{k:float(v) for k,v in aparts.items()},"directional_parts":{k:float(v) for k,v in dparts.items()},"actor_metrics":actor_metrics(po,{k:np.asarray(v) for k,v in pb.items()}),"directional_metrics":directional_metrics(co,{k:np.asarray(v) for k,v in cb.items()}),"critic_digest":critic_param_digest(params),"environment":environment_info()}
    out=Path(args.out);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n");print(json.dumps(report,indent=2,sort_keys=True))


def _current_parent(semantic_id):
    rows=current_rows();cache=load_cache("current",True)
    for i,r in enumerate(rows):
        if r.semantic_id==semantic_id:
            return i,row_genome(r),{k:cache[k][i:i+1] for k in NUMERIC_KEYS}
    raise ValueError(f"current parent semantic id not found: {semantic_id}")


def _candidate_copy(c: GeneratedCandidate, **updates) -> GeneratedCandidate:
    d=asdict(c);d.update(updates);return GeneratedCandidate(**d)


def _goal_context_parent(cfg, semantic_id: str, manifest_path: str):
    context=load_goal_context_manifest(manifest_path,expected_parent=semantic_id)
    _,parent,_=_current_parent(semantic_id)
    cache=context_candidate_features([parent],cfg,context)
    batch={k:cache[k][:1] for k in NUMERIC_KEYS}
    case_ids=np.asarray(cache["case_ids"][0],np.int32)
    for cid in context["loss_case_ids"]:
        if int(np.sum(case_ids==int(cid)))!=1:raise ValueError(f"goal loss {cid} missing/duplicated in semantic context")
    return context,parent,batch,case_ids


def _goal_mask_from_case_ids(case_ids: np.ndarray, goal_case_id: int) -> np.ndarray:
    mask=np.asarray(case_ids,np.int32)==int(goal_case_id)
    if int(np.sum(mask))!=1:raise ValueError(f"goal case {goal_case_id} not uniquely represented")
    return mask


def _generate_actor_goal_pool(model,params,cfg,parent,parent_batch,case_ids,loss_ids,learned_mask,*,n_total,seed,temperature,topk,diversity_penalty):
    loss_ids=[int(x) for x in loss_ids]
    per_goal=max(1,int(np.ceil(max(1,n_total)/max(1,len(loss_ids)))))
    by_goal={};raw=[]
    for gi,cid in enumerate(loss_ids):
        xs=actor_generate(model,params,cfg,parent,parent_batch,n=per_goal,seed=seed+gi*100003,temperature=temperature,topk=topk,
                          diversity_penalty=diversity_penalty,goal_mask=_goal_mask_from_case_ids(case_ids,cid),
                          goal_spec=np.asarray([1.,0.,1.,0.],np.float32),goal_case_ids=(cid,),goal_mode="LOSS_REPAIR",
                          learned_operation_mask=learned_mask)
        by_goal[cid]=len(xs);raw.append(xs)
    selected=[];seen={};depth=0
    while len(selected)<n_total:
        added=False
        for cid,xs in zip(loss_ids,raw):
            if depth>=len(xs):continue
            c=xs[depth];added=True
            if c.semantic_id in seen:
                j=seen[c.semantic_id];old=selected[j];goals=tuple(sorted(set(old.generated_goal_case_ids or old.goal_case_ids or ())|{cid}))
                selected[j]=_candidate_copy(old,generated_goal_case_ids=goals)
                continue
            seen[c.semantic_id]=len(selected);selected.append(c)
            if len(selected)>=n_total:break
        if not added:break
        depth+=1
    return selected,{"requested_total":int(n_total),"per_goal_requested":per_goal,"generated_per_goal":by_goal,"unique_after_global_dedup":len(selected)}


def _score_best_goal(model,params,cfg,candidates,parent,parent_margins,context,loss_ids,batch_size):
    expanded=[]
    for c in candidates:
        for cid in loss_ids:
            expanded.append(_candidate_copy(c,goal_case_ids=(int(cid),),generated_goal_case_ids=tuple(int(x) for x in (c.generated_goal_case_ids or c.goal_case_ids or (int(cid),))),goal_mode="LOSS_REPAIR"))
    scored=critic_score_candidates(model,params,cfg,expanded,parent=parent,parent_margins=parent_margins,batch_size=batch_size,context=context)
    best={}
    def key(c):
        delta=float(c.critic_goal_delta_norm) if c.critic_goal_delta_norm is not None else -9.0
        improve=float(c.critic_goal_improve_prob) if c.critic_goal_improve_prob is not None else 0.0
        flip=float(c.critic_goal_flip_prob) if c.critic_goal_flip_prob is not None else 0.0
        risk=float(c.critic_regression_prob) if c.critic_regression_prob is not None else 1.0
        protected=float(c.critic_protected_regress_prob_max) if c.critic_protected_regress_prob_max is not None else 1.0
        positive=delta>0 and improve>=.5
        return (int(positive),int(c.critic_predicted_target_flips or 0)>0,flip,-risk,-protected,delta)
    for c in scored:
        old=best.get(c.semantic_id)
        if old is None or key(c)>key(old):best[c.semantic_id]=c
    return [best[c.semantic_id] for c in candidates]


def cmd_generation_eval(args):
    cfg=config_for_size("5m",attention_impl=args.attention_impl);context,parent,parent_batch,case_ids=_goal_context_parent(cfg,args.base_semantic,args.goal_context)
    ck=load_checkpoint(args.checkpoint);model=BonsaiEditActorCritic(cfg);ds=load_actor_dataset(args.dataset);learned=np.asarray(ds["learned_operation_mask"],bool)
    t=time.perf_counter();cand,goal_diag=_generate_actor_goal_pool(model,ck["params"],cfg,parent,parent_batch,case_ids,context["loss_case_ids"],learned,n_total=args.samples,seed=args.seed,temperature=args.temperature,topk=args.topk,diversity_penalty=args.diversity_penalty);seconds=time.perf_counter()-t
    scored=critic_score_candidates(model,ck["params"],cfg,cand,parent=parent,parent_margins=np.asarray(context["baseline_margins"],np.int32),batch_size=min(256,max(1,len(cand))),context=context) if cand else []
    ops=Counter(x.action.get("operation") for x in scored);targets=len({(x.action.get("operation"),x.action.get("target_node")) for x in scored})
    per_goal={str(cid):{"generated":0,"retained_after_dedup":0,"predicted_positive_direction":0,"predicted_flip":0} for cid in context["loss_case_ids"]}
    for cid,n in goal_diag["generated_per_goal"].items():per_goal[str(cid)]["generated"]=int(n)
    for c in scored:
        for cid in (c.generated_goal_case_ids or c.goal_case_ids or ()):
            if str(cid) in per_goal:per_goal[str(cid)]["retained_after_dedup"]+=1
        for cid in (c.goal_case_ids or ()):
            if str(cid) in per_goal:
                per_goal[str(cid)]["predicted_positive_direction"]+=int(float(c.critic_goal_delta_norm or -1)>0 and float(c.critic_goal_improve_prob or 0)>=.5)
                per_goal[str(cid)]["predicted_flip"]+=int(int(c.critic_predicted_target_flips or 0)>0)
    report={"architecture":ARCH,"requested":args.samples,"valid_unique":len(scored),"valid_unique_rate":len(scored)/max(1,args.samples),"semantic_duplicate_rate_after_canonicalization":1-len({x.semantic_id for x in scored})/max(1,len(scored)),"operation_distribution":ops,"target_diversity":targets,"goal_coverage":sum(v["retained_after_dedup"]>0 for v in per_goal.values()),"per_goal":per_goal,"generation":goal_diag,"novel_vs_parent":sum(x.semantic_id!=parent.semantic_id() for x in scored),"seconds":seconds,"candidates_s":len(scored)/max(seconds,1e-9)}
    write_candidates(args.out,scored);Path(args.report).write_text(json.dumps(report,indent=2,sort_keys=True)+"\n");print(json.dumps(report,indent=2,sort_keys=True))


def cmd_generator_benchmark(args):
    cfg=config_for_size("5m",attention_impl=args.attention_impl);context,parent,parent_batch,case_ids=_goal_context_parent(cfg,args.base_semantic,args.goal_context);ck=load_checkpoint(args.checkpoint);model=BonsaiEditActorCritic(cfg);ds=load_actor_dataset(args.dataset);out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    parent_margins=np.asarray(context["baseline_margins"],np.int32);loss_ids=[int(x) for x in context["loss_case_ids"]];learned=np.asarray(ds["learned_operation_mask"],bool)
    proposal_n=max(args.cuda_budget,args.proposal_multiplier*args.cuda_budget)
    actor_pool,actor_generation=_generate_actor_goal_pool(model,ck["params"],cfg,parent,parent_batch,case_ids,loss_ids,learned,n_total=proposal_n,seed=args.seed,temperature=args.temperature,topk=args.topk,diversity_penalty=args.diversity_penalty)
    heuristic_pool=heuristic_generate(parent,n=proposal_n,seed=args.seed+11)
    all_goals=tuple(loss_ids)
    heuristic_raw=[_candidate_copy(c,goal_case_ids=all_goals,generated_goal_case_ids=all_goals,goal_mode="LOSS_REPAIR") for c in heuristic_pool]
    random_pool=[_candidate_copy(c,goal_case_ids=all_goals,generated_goal_case_ids=all_goals,goal_mode="LOSS_REPAIR") for c in random_typed_generate(parent,n=args.cuda_budget,seed=args.seed+23)]
    empirical_pool=[_candidate_copy(c,goal_case_ids=all_goals,generated_goal_case_ids=all_goals,goal_mode="LOSS_REPAIR") for c in empirical_generate(parent,ds,n=args.cuda_budget,seed=args.seed+37)]
    actor_scored=critic_score_candidates(model,ck["params"],cfg,actor_pool,parent=parent,parent_margins=parent_margins,batch_size=args.critic_batch,context=context)
    heuristic_raw_scored=critic_score_candidates(model,ck["params"],cfg,heuristic_raw,parent=parent,parent_margins=parent_margins,batch_size=args.critic_batch,context=context)
    heuristic_best=_score_best_goal(model,ck["params"],cfg,heuristic_pool,parent,parent_margins,context,loss_ids,args.critic_batch)
    actor_filtered,actor_diag=directional_shortlist(actor_scored,args.cuda_budget,min_improve_prob=args.min_improve_prob,max_regression_prob=args.max_regression_prob,max_protected_regress_prob=args.max_protected_regress_prob,diversity_penalty=args.operation_diversity_penalty)
    heuristic_filtered,heuristic_diag=directional_shortlist(heuristic_best,args.cuda_budget,min_improve_prob=args.min_improve_prob,max_regression_prob=args.max_regression_prob,max_protected_regress_prob=args.max_protected_regress_prob,diversity_penalty=args.operation_diversity_penalty)
    write_candidates(out/"actor_pool_scored.jsonl",actor_scored);write_candidates(out/"heuristic_pool_scored.jsonl",heuristic_best)
    lanes={"actor_raw":actor_scored[:args.cuda_budget],"actor_critic":actor_filtered,"heuristic_raw":heuristic_raw_scored[:args.cuda_budget],"heuristic_critic":heuristic_filtered,"random_typed":random_pool[:args.cuda_budget],"empirical_ops":empirical_pool[:args.cuda_budget]}
    parent_path=out/"base_parent.json";parent_path.write_text(json.dumps(parent.canonical(),indent=2,sort_keys=True)+"\n")
    summary={"architecture":ARCH,"base_parent":parent.semantic_id(),"goal_context":str(args.goal_context),"loss_case_ids":loss_ids,"actor_generation":actor_generation,"proposal_n":proposal_n,"cuda_budget_per_lane":args.cuda_budget,
             "selection":{"actor_critic":actor_diag,"heuristic_critic":heuristic_diag,"min_improve_prob":args.min_improve_prob,"max_regression_prob":args.max_regression_prob,"max_protected_regress_prob":args.max_protected_regress_prob,"operation_diversity_penalty":args.operation_diversity_penalty},"lanes":{}}
    for name,c in lanes.items():
        write_candidates(out/f"{name}.jsonl",c)
        summary["lanes"][name]={"count":len(c),"unique":len({x.semantic_id for x in c}),"operations":Counter(x.action.get('operation') for x in c),"target_node_diversity":len({(x.action.get('operation'),x.action.get('target_node')) for x in c}),"goal_coverage":len({int(g) for x in c for g in (x.goal_case_ids or ())}),
                                "predicted_positive_target_direction":sum(float(x.critic_goal_delta_norm or -1)>0 for x in c if x.critic_goal_delta_norm is not None),"predicted_target_flips":sum(int(x.critic_predicted_target_flips or 0)>0 for x in c),"predicted_win_crossings":sum(int(x.critic_predicted_win_crossings or 0)>0 for x in c),"mean_goal_delta_norm":float(np.mean([x.critic_goal_delta_norm for x in c if x.critic_goal_delta_norm is not None])) if any(x.critic_goal_delta_norm is not None for x in c) else None,"mean_regression_prob":float(np.mean([x.critic_regression_prob for x in c if x.critic_regression_prob is not None])) if any(x.critic_regression_prob is not None for x in c) else None}
    (out/"generator_benchmark.json").write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n");print(json.dumps(summary,indent=2,sort_keys=True))


def cmd_prepare_fresh(args):
    rows=[json.loads(x) for x in Path(args.source_jsonl).read_text().splitlines() if x.strip()];seeds=sorted({int(r['seed']) for r in rows});families={name:i for i,name in enumerate(sorted({str(r['team']) for r in rows}))};seat_by_seed={s:set() for s in seeds}
    for r in rows:seat_by_seed[int(r['seed'])].add(int(r['seat']))
    if len(seeds)<args.min_seeds or any(v!={0,1} for v in seat_by_seed.values()):raise ValueError(f"fresh CUDA reward must be both seats across >= {args.min_seeds} seeds: seeds={len(seeds)} seats={seat_by_seed}")
    out=Path(args.out);out.parent.mkdir(parents=True,exist_ok=True);lines=[]
    for r in rows:lines.append("\t".join([str(r['tape']),str(int(r['seat'])),str(int(r['seed'])),str(families[str(r['team'])]),str(int(r.get('protected',1))),str(int(r.get('own_history',0))) ]))
    out.write_text("\n".join(lines)+"\n");m={"cases":len(rows),"unique_seeds":len(seeds),"both_seats_per_seed":True,"seeds":seeds,"families":families,"sha256":hashlib.sha256(out.read_bytes()).hexdigest()};Path(args.manifest).write_text(json.dumps(m,indent=2,sort_keys=True)+"\n");print(json.dumps(m,indent=2,sort_keys=True))


def cuda_goal_metrics(candidate: GeneratedCandidate, baseline: np.ndarray, margins: np.ndarray) -> dict:
    baseline=np.asarray(baseline,np.int32);margins=np.asarray(margins,np.int32)
    if baseline.shape!=margins.shape:raise ValueError("CUDA candidate/baseline shape mismatch")
    loss_ids=[int(i) for i in np.flatnonzero(baseline<0)]
    goals=[int(x) for x in (candidate.goal_case_ids or loss_ids) if 0<=int(x)<len(baseline) and baseline[int(x)]<0]
    if not goals:goals=loss_ids
    delta=margins-baseline;td=np.asarray([delta[g] for g in goals],np.int32)
    return {"goal_case_ids":goals,"generated_goal_case_ids":[int(x) for x in (candidate.generated_goal_case_ids or candidate.goal_case_ids or goals)],"target_changed":bool(np.any(td!=0)),"target_improved":bool(np.any(td>0)),"target_worsened":bool(np.any(td<0)),"target_mean_delta":float(np.mean(td)),"target_best_delta":int(np.max(td)),"target_flip_count":int(np.sum([margins[g]>0 for g in goals]))}


def cmd_cuda(args):
    cases_manifest=json.loads(Path(args.cases_manifest).read_text());
    if not cases_manifest.get("both_seats_per_seed") or int(cases_manifest.get("unique_seeds",0))<args.min_seeds:raise ValueError("refusing single-seat/low-seed CUDA reward manifest")
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True);base=genome_from_json(json.loads(Path(args.base_parent).read_text()));report={"cases":cases_manifest,"lanes":{},"cuda_evaluator":str(Path(args.exe).resolve())}
    cuda_lane_names=("actor_raw","actor_critic","heuristic_raw","heuristic_critic","random_typed","empirical_ops")
    for lane_name in cuda_lane_names:
        lane_file=Path(args.lanes)/f"{lane_name}.jsonl"
        if not lane_file.exists(): raise FileNotFoundError(f"missing CUDA comparison lane: {lane_file}")
        candidates=read_candidates(lane_file);genomes=[base]+[genome_from_json(c.genome) for c in candidates];gm=out/f"{lane_file.stem}.b4gm";rs=out/f"{lane_file.stem}.b4rs";pack_genomes(genomes,gm)
        env=os.environ.copy();env["CUDA_VISIBLE_DEVICES"]=str(args.gpu);cmd=[args.exe,"--controller",args.controller,"--routes",args.routes,"--legacy-routes",args.legacy_routes,"--cases",args.cases,"--genomes",str(gm),"--out",str(rs),"--feed-slot","5","--feed-count","13","--tpb","64"]
        p=subprocess.run(cmd,cwd=ROOT,env=env,text=True,capture_output=True,check=False);(out/f"{lane_file.stem}.cuda.log").write_text(p.stdout+p.stderr)
        if p.returncode:raise RuntimeError(f"CUDA lane {lane_file.stem} failed: {p.stderr[-3000:]}")
        rr=load_results(rs);baseline=np.asarray([a-b for a,b,_,flags in rr[0]],np.int32)
        if any(int(flags) for _,_,_,flags in rr[0]):raise RuntimeError(f"CUDA baseline flags in {lane_name}")
        baseline_losses=[int(i) for i in np.flatnonzero(baseline<0)];safe=[];recs=[]
        for idx,(cand,row) in enumerate(zip(candidates,rr[1:]),1):
            if any(int(flags) for _,_,_,flags in row):raise RuntimeError(f"CUDA candidate flags lane={lane_name} candidate={idx}")
            m=np.asarray([a-b for a,b,_,_ in row],np.int32);delta=m-baseline;l2w=int(np.sum((baseline<0)&(m>0)));w2l=int(np.sum((baseline>0)&(m<=0)));reg3=int(np.sum((baseline>0)&(delta<-3000)));md=float(np.mean(delta));is_safe=l2w>0 and w2l==0 and reg3==0
            changed_cases=int(np.sum(m!=baseline));exact_parent_behavior=changed_cases==0
            gm=cuda_goal_metrics(cand,baseline,m)
            rec={"candidate_index":idx,"semantic_id":cand.semantic_id,"safe":is_safe,"l2w":l2w,"w2l":w2l,"reg3":reg3,"mean_delta":md,"changed_cases":changed_cases,"exact_parent_behavior":exact_parent_behavior,"source":cand.source,"action":cand.action,**gm,
                 "critic_regression_prob":cand.critic_regression_prob,"critic_flip_prob":cand.critic_flip_prob,"critic_margin_pred":cand.critic_margin_pred,"critic_effect_mean_abs":cand.critic_effect_mean_abs,"critic_effect_cases":cand.critic_effect_cases,"critic_goal_delta_norm":cand.critic_goal_delta_norm,"critic_goal_delta_margin":cand.critic_goal_delta_margin,"critic_goal_improve_prob":cand.critic_goal_improve_prob,"critic_goal_flip_prob":cand.critic_goal_flip_prob,"critic_protected_regress_prob_max":cand.critic_protected_regress_prob_max,
                 "probe_activated_cases":cand.probe_activated_cases,"probe_route_divergence_cases":cand.probe_route_divergence_cases,"probe_loss_target_cases":cand.probe_loss_target_cases,"probe_win_exposure_cases":cand.probe_win_exposure_cases,"probe_effect_score":cand.probe_effect_score};recs.append(rec)
            if is_safe:safe.append(rec)
        toks={};
        for tok in p.stdout.split():
            if '=' in tok:
                k,v=tok.split('=',1);toks[k]=v
        per_loss={}
        for cid in baseline_losses:
            targeted=[(c,r) for c,r in zip(candidates,recs) if cid in [int(x) for x in (c.goal_case_ids or baseline_losses)]]
            generated=[c for c in candidates if cid in [int(x) for x in (c.generated_goal_case_ids or c.goal_case_ids or baseline_losses)]]
            improvements=[];changed=improved=flipped=0
            for _,r in targeted:
                own,opp,_,_=rr[r["candidate_index"]][cid];margin=int(own)-int(opp);d=margin-int(baseline[cid]);improvements.append(d);changed+=int(d!=0);improved+=int(d>0);flipped+=int(margin>0)
            per_loss[str(cid)]={"generated_provenance_in_lane":len(generated),"evaluated_targeted":len(targeted),"changed_margin":changed,"improved":improved,"best_improvement":max(improvements) if improvements else 0,"flipped_to_win":flipped}
        lane={"candidates":len(candidates),"games":int(toks.get('games',len(genomes)*len(baseline))),"games_s":float(toks.get('games_s',0)),"safe_count":len(safe),"safe_per_1000_cuda_evaluated":1000*len(safe)/max(1,len(candidates)),"first_safe_candidate":safe[0]['candidate_index'] if safe else None,
              "target_improving_candidates":sum(r['target_improved'] for r in recs),"target_worsening_candidates":sum(r['target_worsened'] for r in recs),"mean_target_loss_delta":float(np.mean([r['target_mean_delta'] for r in recs])) if recs else 0.0,"best_target_loss_delta":max([r['target_best_delta'] for r in recs],default=0),
              "loss_to_win_candidates":sum(r['l2w']>0 for r in recs),"win_to_loss_candidates":sum(r['w2l']>0 for r in recs),"severe_winning_margin_regression_candidates":sum(r['reg3']>0 for r in recs),"positive_margin_candidates":sum(r['mean_delta']>0 for r in recs),"behavior_changing_candidates":sum(not r['exact_parent_behavior'] for r in recs),"exact_parent_behavior_candidates":sum(r['exact_parent_behavior'] for r in recs),"mean_changed_cases":float(np.mean([r['changed_cases'] for r in recs])) if recs else 0.0,
              "operation_distribution":dict(Counter(str(r['action'].get('operation')) for r in recs)),"target_node_diversity":len({(str(r['action'].get('operation')),r['action'].get('target_node')) for r in recs}),"goal_coverage":len({g for r in recs for g in r['goal_case_ids']}),"per_loss":per_loss,"safe_records":safe,"records":recs};report["lanes"][lane_file.stem]=lane
    (out/"cuda_campaign_result.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n");print(json.dumps(report,indent=2,sort_keys=True))


def cmd_package(args):
    run=Path(args.run_dir);required=["actor_training_result.json","actor_evaluation_result.json","generation_evaluation.json","generator_benchmark/generator_benchmark.json","cuda_campaign/cuda_campaign_result.json"];missing=[x for x in required if not(run/x).exists()]
    if missing:raise FileNotFoundError(missing)
    training=json.loads((run/required[0]).read_text());evaluation=json.loads((run/required[1]).read_text());generation=json.loads((run/required[2]).read_text());genbench=json.loads((run/required[3]).read_text());cuda=json.loads((run/required[4]).read_text());ac=cuda['lanes'].get('actor_critic',{});manifest={"schema":2,"architecture":ARCH,"params":training['params'],"created_utc":datetime.now(UTC).isoformat(),"critic_frozen_exact":training['critic_frozen_exact'],"actor_training":training,"actor_evaluation":evaluation,"generation":generation,"generator_benchmark":genbench,"cuda_campaign":cuda,"decision_gate":{"valid_generation":generation['valid_unique_rate']>=0.9,"all_failure_goals_covered":generation.get('goal_coverage',0)>=6,"target_direction_repaired":int(ac.get('target_improving_candidates',0))>0,"new_actor_l2w":int(ac.get('loss_to_win_candidates',0))>0,"new_strict_safe_actor_critic":int(ac.get('safe_count',0))>0,"zero_actor_critic_w2l":int(ac.get('win_to_loss_candidates',0))==0}}
    (run/"result_manifest.json").write_text(json.dumps(manifest,indent=2,sort_keys=True,default=str)+"\n");lines=[f"# {ARCH}","",f"- Parameters: **{training['params']:,}**",f"- Critic frozen exact: **{training['critic_frozen_exact']}**",f"- Valid unique actor generation: **{generation['valid_unique_rate']:.3f}**",f"- Failure-goal coverage: **{generation.get('goal_coverage',0)}/6**",f"- Operation families generated: **{len(generation['operation_distribution'])}**",""]
    for lane,x in cuda['lanes'].items():lines.append(f"- {lane}: target_improve={x.get('target_improving_candidates',0)}, best_target_delta={x.get('best_target_loss_delta',0)}, safe={x['safe_count']}, l2w={x['loss_to_win_candidates']}, w2l={x['win_to_loss_candidates']}, reg3={x.get('severe_winning_margin_regression_candidates',0)}, games/s={x['games_s']:.1f}")
    (run/"SUMMARY.md").write_text("\n".join(lines)+"\n");package=Path(args.package);package.parent.mkdir(parents=True,exist_ok=True)
    with tarfile.open(package,"w:gz",compresslevel=6) as tf:
        for p in run.rglob('*'):
            if not p.is_file():
                continue
            rel=p.relative_to(run)
            # Keep the final GCAC-v2 checkpoint fully reproducible. Intermediate and
            # critic-initialization checkpoints keep metadata only to bound package size.
            if rel.parts and rel.parts[0] in {"critic_initialized","early","middle"} and not p.name.startswith('_CHECKPOINT'):
                continue
            tf.add(p,arcname=f"{run.name}/{rel}")
    print(json.dumps({"package":str(package),"sha256":hashlib.sha256(package.read_bytes()).hexdigest(),"bytes":package.stat().st_size},indent=2))


def parser():
    ap=argparse.ArgumentParser(description=ARCH);sp=ap.add_subparsers(dest='cmd',required=True)
    p=sp.add_parser('prepare-data');p.add_argument('--out',default=str(ACTOR_DATA_DIR/'gcac_v2_transitions_full.npz'));p.add_argument('--limit-rows',type=int);p.add_argument('--max-transitions',type=int,default=60000);p.add_argument('--min-safe-positive',type=int,default=32);p.add_argument('--awr-beta',type=float,default=2.0);p.add_argument('--max-positive-weight',type=float,default=12.0);p.add_argument('--max-negative-weight',type=float,default=8.0);p.set_defaults(fn=cmd_prepare)
    def train_common(p):
        p.add_argument('--dataset',required=True);p.add_argument('--attention-impl',choices=('xla','cudnn'),default='cudnn');p.add_argument('--batch-size',type=int,default=512);p.add_argument('--seed',type=int,default=73)
    p=sp.add_parser('init-critic');train_common(p);p.add_argument('--critic-checkpoint',required=True);p.add_argument('--out',required=True);p.set_defaults(fn=cmd_init_critic)
    p=sp.add_parser('smoke');train_common(p);p.add_argument('--critic-checkpoint',required=True);p.add_argument('--steps',type=int,default=4);p.add_argument('--eval-samples',type=int,default=512);p.add_argument('--direction-loss-weight',type=float,default=.5);p.add_argument('--lr',type=float,default=1e-3);p.add_argument('--weight-decay',type=float,default=1e-4);p.add_argument('--clip',type=float,default=2.0);p.set_defaults(fn=cmd_smoke)
    p=sp.add_parser('train-actor');train_common(p);p.add_argument('--checkpoint',required=True);p.add_argument('--out',required=True);p.add_argument('--epochs',type=int,default=10);p.add_argument('--early-epoch',type=int,default=2);p.add_argument('--middle-epoch',type=int,default=6);p.add_argument('--eval-samples',type=int,default=4096);p.add_argument('--examples-per-epoch',type=int,default=65536);p.add_argument('--direction-loss-weight',type=float,default=.5);p.add_argument('--lr',type=float,default=8e-4);p.add_argument('--weight-decay',type=float,default=1e-4);p.add_argument('--clip',type=float,default=2.0);p.add_argument('--resume',action='store_true');p.set_defaults(fn=cmd_train)
    p=sp.add_parser('evaluate-actor');train_common(p);p.add_argument('--checkpoint',required=True);p.add_argument('--eval-samples',type=int,default=4096);p.add_argument('--out',required=True);p.set_defaults(fn=cmd_evaluate_actor)
    def gen_common(p):
        p.add_argument('--checkpoint',required=True);p.add_argument('--goal-context',required=True);p.add_argument('--attention-impl',choices=('xla','cudnn'),default='cudnn');p.add_argument('--base-semantic',default=DEFAULT_BASE_SEMANTIC);p.add_argument('--seed',type=int,default=73);p.add_argument('--temperature',type=float,default=.9);p.add_argument('--topk',type=int,default=8);p.add_argument('--diversity-penalty',type=float,default=.35)
    p=sp.add_parser('evaluate-generation');gen_common(p);p.add_argument('--dataset',required=True);p.add_argument('--samples',type=int,default=512);p.add_argument('--out',required=True);p.add_argument('--report',required=True);p.set_defaults(fn=cmd_generation_eval)
    p=sp.add_parser('generator-benchmark');gen_common(p);p.add_argument('--dataset',required=True);p.add_argument('--out',required=True);p.add_argument('--cuda-budget',type=int,default=128);p.add_argument('--proposal-multiplier',type=int,default=4);p.add_argument('--critic-batch',type=int,default=256);p.add_argument('--max-regression-prob',type=float,default=.35);p.add_argument('--max-protected-regress-prob',type=float,default=.35);p.add_argument('--min-improve-prob',type=float,default=.5);p.add_argument('--operation-diversity-penalty',type=float,default=.25);p.set_defaults(fn=cmd_generator_benchmark)
    p=sp.add_parser('prepare-fresh-cases');p.add_argument('--source-jsonl',required=True);p.add_argument('--out',required=True);p.add_argument('--manifest',required=True);p.add_argument('--min-seeds',type=int,default=6);p.set_defaults(fn=cmd_prepare_fresh)
    p=sp.add_parser('cuda-campaign');p.add_argument('--lanes',required=True);p.add_argument('--base-parent',required=True);p.add_argument('--cases',required=True);p.add_argument('--cases-manifest',required=True);p.add_argument('--exe',required=True);p.add_argument('--controller',required=True);p.add_argument('--routes',required=True);p.add_argument('--legacy-routes',required=True);p.add_argument('--out',required=True);p.add_argument('--gpu',type=int,default=0);p.add_argument('--min-seeds',type=int,default=6);p.set_defaults(fn=cmd_cuda)
    p=sp.add_parser('package-results');p.add_argument('--run-dir',required=True);p.add_argument('--package',required=True);p.set_defaults(fn=cmd_package)
    return ap


def main():
    args = parser().parse_args()
    return args.fn(args)

if __name__=='__main__': main()
