from __future__ import annotations

from dataclasses import asdict
from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp

from actor_grammar import EDIT_OPERATIONS, N_OPS
from model import (
    CASE_DIM,
    EDIT_CAT_COLS,
    EDIT_NUM_COLS,
    TREE_CAT_COLS,
    TREE_NUM_COLS,
    BonsaiEditFormer,
    CrossFusionBlock,
    EditFormerConfig,
    TransformerBlock,
    _dtype,
    count_params,
)


class BonsaiEditActorCritic(nn.Module):
    """BonsaiEditFormer-GCAC-v2: frozen critic plus goal-conditioned typed editor.

    Critic/shared-encoder submodule names intentionally match BonsaiEditFormer so a
    frozen 5M critic checkpoint can be merged exactly. Every new trainable adapter/head
    is named ``actor_*`` and consumes stop-gradient copies of proven representations.
    """

    cfg: EditFormerConfig

    @nn.compact
    def __call__(self, batch: dict[str, jax.Array], *, train: bool = False) -> dict[str, jax.Array]:
        cfg = self.cfg
        dtype = _dtype(cfg.compute_dtype)
        tc, tn = batch["tree_cat"], batch["tree_num"]
        tm = batch["tree_mask"].astype(bool)

        def emb(name, vocab, ids):
            return nn.Embed(vocab, cfg.d_model, dtype=dtype, param_dtype=jnp.float32, name=name)(ids)

        tree = (
            emb("tree_type", 8, tc[..., 0]) + emb("tree_feature", 40, tc[..., 1]) +
            emb("tree_cmp", 8, tc[..., 2]) + emb("tree_route", 20, tc[..., 3]) +
            emb("tree_parent_kind", 4, tc[..., 4]) + emb("tree_ordinal", 12, tc[..., 5]) +
            nn.Dense(cfg.d_model, dtype=dtype, param_dtype=jnp.float32, name="tree_num_proj")(tn)
        )
        tree = tree * tm[..., None]
        for i in range(cfg.tree_blocks):
            tree = TransformerBlock(cfg, f"tree{i}", structural=True)(
                tree, tm, batch["tree_parent"], batch["tree_depth"], batch["tree_rule"], train=train
            )
        tree_pool = tree[:, 0]

        ec, en = batch["edit_cat"], batch["edit_num"]
        em = batch["edit_mask"].astype(bool)
        if cfg.use_edit_delta:
            edit = (
                emb("edit_type", 8, ec[..., 0]) + emb("edit_feature", 40, ec[..., 1]) +
                emb("edit_cmp", 8, ec[..., 2]) + emb("edit_route", 20, ec[..., 3]) +
                emb("edit_ordinal", 12, ec[..., 4]) +
                nn.Dense(cfg.d_model, dtype=dtype, param_dtype=jnp.float32, name="edit_num_proj")(en)
            )
        else:
            edit = jnp.zeros((*ec.shape[:2], cfg.d_model), dtype=dtype)
        edit = edit * em[..., None]
        edit = nn.LayerNorm(dtype=jnp.float32, name="edit_ln")(edit).astype(dtype)
        edit = nn.Dense(cfg.ffn_dim // 2, dtype=dtype, param_dtype=jnp.float32, name="edit_ff1")(edit)
        edit = nn.gelu(edit)
        edit = nn.Dense(cfg.d_model, dtype=dtype, param_dtype=jnp.float32, name="edit_ff2")(edit)
        edit_pool = jnp.sum(edit * em[..., None], axis=1) / jnp.maximum(jnp.sum(em, axis=1, keepdims=True), 1)

        cm = batch["case_mask"].astype(bool)
        if cfg.use_activation_context:
            cases = nn.Dense(cfg.d_model, dtype=dtype, param_dtype=jnp.float32, name="case_proj")(batch["case_features"])
        else:
            cases = jnp.zeros((*batch["case_features"].shape[:2], cfg.d_model), dtype=dtype)
        cases = cases * cm[..., None]
        dummy_parent = jnp.full(cm.shape, -1, jnp.int32)
        dummy_depth = jnp.zeros(cm.shape, jnp.int32)
        dummy_rule = jnp.zeros(cm.shape, jnp.int32)
        for i in range(cfg.case_blocks):
            cases = TransformerBlock(cfg, f"case{i}", structural=False)(cases, cm, dummy_parent, dummy_depth, dummy_rule, train=train)

        q = jnp.stack([tree_pool, edit_pool], axis=1)
        qmask = jnp.ones(q.shape[:2], dtype=bool)
        for i in range(cfg.fusion_blocks):
            q = CrossFusionBlock(cfg, f"fusion{i}")(q, qmask, cases, cm, train=train)
        fused = nn.LayerNorm(dtype=jnp.float32, name="fused_ln")(jnp.mean(q, axis=1)).astype(dtype)

        # Critic heads: names and shapes are identical to the proven ranker.
        def scalar_head(name: str):
            h = nn.Dense(cfg.d_model // 2, dtype=dtype, param_dtype=jnp.float32, name=f"{name}_h")(fused)
            h = nn.gelu(h)
            return nn.Dense(1, dtype=jnp.float32, param_dtype=jnp.float32, name=f"{name}_o")(h.astype(jnp.float32))[:, 0]

        regression_logit = scalar_head("regression")
        flip_logit = scalar_head("flip")
        margin = jnp.tanh(scalar_head("margin"))
        worst_win = jnp.tanh(scalar_head("worst_win"))
        rank_residual = scalar_head("rank")
        heuristic = batch["heuristic"].astype(jnp.float32)
        final_rank = rank_residual + (heuristic if cfg.use_heuristic_residual else 0.0)
        case_delta = nn.Dense(1, dtype=jnp.float32, param_dtype=jnp.float32, name="case_delta")(cases.astype(jnp.float32))[..., 0]
        case_delta = jnp.tanh(case_delta)

        # GCAC-v2 goal path. Goal identity is never embedded: the mask selects actual
        # legal-state case tokens and goal_spec only describes desired outcome/direction.
        frozen_cases = jax.lax.stop_gradient(cases)
        frozen_tree = jax.lax.stop_gradient(tree)
        frozen_fused = jax.lax.stop_gradient(fused)
        bsz = fused.shape[0]
        goal_mask = batch.get("goal_mask", cm).astype(bool) & cm
        has_goal = jnp.any(goal_mask, axis=1, keepdims=True)
        goal_mask = jnp.where(has_goal, goal_mask, cm)
        goal_spec = batch.get("goal_spec", jnp.zeros((bsz, 4), jnp.float32)).astype(jnp.float32)
        goal_query = nn.Dense(cfg.d_model, dtype=dtype, param_dtype=jnp.float32, name="actor_goal_query")(goal_spec)
        goal_key = nn.Dense(cfg.d_model, use_bias=False, dtype=dtype, param_dtype=jnp.float32, name="actor_goal_key")(frozen_cases)
        goal_score = jnp.einsum("bd,btd->bt", goal_query.astype(jnp.float32), goal_key.astype(jnp.float32)) / jnp.sqrt(float(cfg.d_model))
        goal_score = jnp.where(goal_mask, goal_score, jnp.asarray(-1e9, jnp.float32))
        goal_weight = jax.nn.softmax(goal_score, axis=1) * goal_mask.astype(jnp.float32)
        goal_weight = goal_weight / jnp.maximum(jnp.sum(goal_weight, axis=1, keepdims=True), 1e-6)
        goal_pool = jnp.sum(frozen_cases.astype(jnp.float32) * goal_weight[..., None], axis=1)
        goal_spec_h = nn.Dense(cfg.d_model, dtype=jnp.float32, param_dtype=jnp.float32, name="actor_goal_spec")(goal_spec)
        goal = jnp.concatenate([goal_pool, goal_spec_h], axis=-1)
        goal = nn.Dense(cfg.d_model, dtype=dtype, param_dtype=jnp.float32, name="actor_goal_fuse")(goal)
        goal = nn.gelu(goal)
        goal = nn.LayerNorm(dtype=jnp.float32, name="actor_goal_ln")(goal).astype(dtype)

        # Hierarchical actor: goal conditions operation, pointer query, and all typed
        # semantic arguments. The second pass supplies sampled actor_op/actor_target.
        actor_input = jnp.concatenate([frozen_fused, goal], axis=-1)
        actor_h = nn.Dense(cfg.d_model, dtype=dtype, param_dtype=jnp.float32, name="actor_shared_h")(actor_input)
        actor_h = nn.gelu(actor_h)
        op_logits = nn.Dense(N_OPS, dtype=jnp.float32, param_dtype=jnp.float32, name="actor_op")(actor_h.astype(jnp.float32))

        op_embed = self.param("actor_op_embed", nn.initializers.normal(stddev=0.02), (N_OPS, cfg.d_model), jnp.float32).astype(dtype)
        ptr_key = nn.Dense(cfg.d_model, use_bias=False, dtype=dtype, param_dtype=jnp.float32, name="actor_ptr_key")(frozen_tree)
        ptr_base = nn.Dense(cfg.d_model, use_bias=False, dtype=dtype, param_dtype=jnp.float32, name="actor_ptr_query")(actor_h)
        ptr_goal = nn.Dense(cfg.d_model, use_bias=False, dtype=dtype, param_dtype=jnp.float32, name="actor_ptr_goal")(goal)
        ptr_q = ptr_base[:, None, :] + ptr_goal[:, None, :] + op_embed[None, :, :]
        target_logits_all = jnp.einsum("bod,btd->bot", ptr_q.astype(jnp.float32), ptr_key.astype(jnp.float32)) / jnp.sqrt(float(cfg.d_model))
        default_target_mask = jnp.broadcast_to(tm[:, None, :], target_logits_all.shape)
        target_mask = batch.get("actor_target_mask", default_target_mask).astype(bool)
        target_logits_all = jnp.where(target_mask, target_logits_all, jnp.asarray(-1e9, jnp.float32))

        selected_op = jnp.clip(batch.get("actor_op", jnp.zeros((bsz,), jnp.int32)).astype(jnp.int32), 0, N_OPS - 1)
        selected_target = jnp.clip(batch.get("actor_target", jnp.zeros((bsz,), jnp.int32)).astype(jnp.int32), 0, cfg.tree_cap - 1)
        selected_target_state = frozen_tree[jnp.arange(bsz), selected_target]
        selected_op_embed = op_embed[selected_op]
        sem = jnp.concatenate([frozen_fused, goal, selected_target_state, selected_op_embed], axis=-1)
        sem = nn.Dense(cfg.d_model, dtype=dtype, param_dtype=jnp.float32, name="actor_sem_h")(sem)
        sem = nn.gelu(sem)

        def cat_head(name: str, n: int):
            return nn.Dense(n, dtype=jnp.float32, param_dtype=jnp.float32, name=name)(sem.astype(jnp.float32))

        feature_logits = cat_head("actor_feature", 27)
        cmp_logits = cat_head("actor_cmp", 4)
        route_logits = cat_head("actor_route", 15)
        index_logits = cat_head("actor_index", 100)
        feature2_logits = cat_head("actor_feature2", 27)
        cmp2_logits = cat_head("actor_cmp2", 4)
        index2_logits = cat_head("actor_index2", 100)
        condition_count_logits = cat_head("actor_condition_count", 2)
        priority_logits = cat_head("actor_priority", 8)
        template_logits = cat_head("actor_template", 4)
        value_norm = jnp.tanh(nn.Dense(1, dtype=jnp.float32, param_dtype=jnp.float32, name="actor_value")(sem.astype(jnp.float32))[:, 0])
        value2_norm = jnp.tanh(nn.Dense(1, dtype=jnp.float32, param_dtype=jnp.float32, name="actor_value2")(sem.astype(jnp.float32))[:, 0])
        prefix_norm = jnp.tanh(nn.Dense(1, dtype=jnp.float32, param_dtype=jnp.float32, name="actor_prefix")(sem.astype(jnp.float32))[:, 0])
        proposal_quality_logit = nn.Dense(1, dtype=jnp.float32, param_dtype=jnp.float32, name="actor_quality")(sem.astype(jnp.float32))[:, 0]

        # Append-only candidate-aware directional heads. The case representation already
        # encodes the supplied baseline margin/outcome and candidate activation context.
        # Selecting these outputs with goal_mask provides target-specific interpretation.
        direction_goal = nn.Dense(cfg.d_model // 2, dtype=dtype, param_dtype=jnp.float32, name="actor_direction_goal")(goal)
        direction_goal = jnp.broadcast_to(direction_goal[:, None, :], (bsz, cfg.case_cap, cfg.d_model // 2))
        direction_input = jnp.concatenate([frozen_cases, direction_goal], axis=-1)
        direction_h = nn.Dense(cfg.d_model // 2, dtype=dtype, param_dtype=jnp.float32, name="actor_direction_h")(direction_input)
        direction_h = nn.gelu(direction_h)
        direction_delta = jnp.tanh(nn.Dense(1, dtype=jnp.float32, param_dtype=jnp.float32, name="actor_direction_delta")(direction_h.astype(jnp.float32))[..., 0])
        direction_improve = nn.Dense(1, dtype=jnp.float32, param_dtype=jnp.float32, name="actor_direction_improve")(direction_h.astype(jnp.float32))[..., 0]
        direction_flip = nn.Dense(1, dtype=jnp.float32, param_dtype=jnp.float32, name="actor_direction_flip")(direction_h.astype(jnp.float32))[..., 0]
        direction_regress = nn.Dense(1, dtype=jnp.float32, param_dtype=jnp.float32, name="actor_direction_regress")(direction_h.astype(jnp.float32))[..., 0]

        return {
            "regression_logit": regression_logit,
            "flip_logit": flip_logit,
            "margin": margin,
            "worst_win": worst_win,
            "rank_residual": rank_residual,
            "rank": final_rank,
            "case_delta": case_delta,
            "actor_goal_embedding": goal.astype(jnp.float32),
            "actor_goal_attention": goal_weight.astype(jnp.float32),
            "actor_direction_delta": direction_delta,
            "actor_direction_improve_logit": direction_improve,
            "actor_direction_flip_logit": direction_flip,
            "actor_direction_regress_logit": direction_regress,
            "actor_op_logits": op_logits,
            "actor_target_logits_all": target_logits_all,
            "actor_feature_logits": feature_logits,
            "actor_cmp_logits": cmp_logits,
            "actor_route_logits": route_logits,
            "actor_index_logits": index_logits,
            "actor_feature2_logits": feature2_logits,
            "actor_cmp2_logits": cmp2_logits,
            "actor_index2_logits": index2_logits,
            "actor_condition_count_logits": condition_count_logits,
            "actor_priority_logits": priority_logits,
            "actor_template_logits": template_logits,
            "actor_value_norm": value_norm,
            "actor_value2_norm": value2_norm,
            "actor_prefix_norm": prefix_norm,
            "actor_proposal_quality_logit": proposal_quality_logit,
        }


def merge_critic_params(actor_params: Any, critic_params: Any) -> tuple[Any, dict[str, int]]:
    """Merge all exact name+shape-compatible critic tensors into Actor-Critic params."""
    from flax import traverse_util
    from flax.core import freeze, unfreeze

    a = traverse_util.flatten_dict(unfreeze(actor_params), sep="/")
    c = traverse_util.flatten_dict(unfreeze(critic_params), sep="/")
    copied = 0
    copied_elems = 0
    critic_elems = 0
    incompatible = 0
    for key, value in c.items():
        critic_elems += int(value.size)
        av = a.get(key)
        if av is not None and tuple(av.shape) == tuple(value.shape):
            a[key] = value
            copied += 1
            copied_elems += int(value.size)
        else:
            incompatible += 1
    merged = traverse_util.unflatten_dict({tuple(k.split("/")): v for k, v in a.items()})
    return freeze(merged), {
        "critic_tensors": len(c),
        "copied_tensors": copied,
        "incompatible_tensors": incompatible,
        "critic_elements": critic_elems,
        "copied_elements": copied_elems,
    }


def actor_param_mask(params: Any) -> Any:
    """True only for actor-specific parameters; freezes critic and shared encoder."""
    from flax import traverse_util
    from flax.core import freeze, unfreeze

    flat = traverse_util.flatten_dict(unfreeze(params), sep="/")
    masked = {k: any(part.startswith("actor_") for part in k.split("/")) for k in flat}
    return freeze(traverse_util.unflatten_dict({tuple(k.split("/")): v for k, v in masked.items()}))


def critic_param_digest(params: Any) -> str:
    import hashlib
    from flax import traverse_util
    from flax.core import unfreeze

    flat = traverse_util.flatten_dict(unfreeze(params), sep="/")
    h = hashlib.sha256()
    for key in sorted(flat):
        if any(part.startswith("actor_") for part in key.split("/")):
            continue
        h.update(key.encode())
        h.update(jax.device_get(flat[key]).tobytes())
    return h.hexdigest()


def actor_config_report(cfg: EditFormerConfig, params: Any) -> dict[str, Any]:
    return {
        "architecture": "BonsaiEditFormer-GCAC-v2",
        "shared_encoder_config": asdict(cfg),
        "operations": list(EDIT_OPERATIONS),
        "parameters": count_params(params),
    }
