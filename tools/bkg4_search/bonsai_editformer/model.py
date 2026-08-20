from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp


N_REL = 10
N_DIST = 5
TREE_CAT_COLS = 6
TREE_NUM_COLS = 10
EDIT_CAT_COLS = 5
EDIT_NUM_COLS = 14
CASE_DIM = 22


@dataclass(frozen=True)
class EditFormerConfig:
    d_model: int = 288
    heads: int = 6
    ffn_dim: int = 960
    tree_blocks: int = 3
    case_blocks: int = 1
    fusion_blocks: int = 1
    tree_cap: int = 48
    edit_cap: int = 10
    case_cap: int = 48
    dropout: float = 0.05
    attention_impl: str = "xla"
    use_structural_bias: bool = True
    use_edit_delta: bool = True
    use_activation_context: bool = True
    use_rank_loss: bool = True
    use_heuristic_residual: bool = True
    compute_dtype: str = "bfloat16"

    @property
    def head_dim(self) -> int:
        if self.d_model % self.heads:
            raise ValueError("d_model must be divisible by heads")
        return self.d_model // self.heads


def _dtype(name: str):
    return jnp.bfloat16 if name == "bfloat16" else jnp.float32


def structural_relations(parent: jax.Array, depth: jax.Array, rule_id: jax.Array, mask: jax.Array):
    """Build direction-aware tree relations from compact structural metadata."""
    b, t = parent.shape
    idx = jnp.arange(t, dtype=parent.dtype)[None, :]
    p = parent
    p_safe = jnp.clip(p, 0, t - 1)
    gp = jnp.take_along_axis(p, p_safe, axis=1)
    qi = idx[:, :, None]
    kj = idx[:, None, :]
    pi = p[:, :, None]
    pj = p[:, None, :]
    gpi = gp[:, :, None]
    gpj = gp[:, None, :]
    ri = rule_id[:, :, None]
    rj = rule_id[:, None, :]
    di = depth[:, :, None]
    dj = depth[:, None, :]

    rel = jnp.full((b, t, t), 9, dtype=jnp.int32)  # other
    rel = jnp.where(qi == kj, 1, rel)  # self
    rel = jnp.where((pj == qi) & (qi != kj), 2, rel)  # query is parent of key
    rel = jnp.where((pi == kj) & (qi != kj), 3, rel)  # query is child of key
    rel = jnp.where((gpj == qi) & (qi != kj), 4, rel)  # query ancestor
    rel = jnp.where((gpi == kj) & (qi != kj), 5, rel)  # query descendant
    siblings = (pi == pj) & (pi >= 0) & (qi != kj)
    rel = jnp.where(siblings, 6, rel)
    same_rule = (ri == rj) & (ri > 0) & (qi != kj)
    rel = jnp.where(same_rule, 7, rel)
    root_context = ((qi <= 2) | (kj <= 2)) & (qi != kj)
    rel = jnp.where(root_context & (rel == 9), 8, rel)
    valid = mask[:, :, None] & mask[:, None, :]
    rel = jnp.where(valid, rel, 0)

    dist = jnp.minimum(jnp.abs(di - dj), 4).astype(jnp.int32)
    dist = jnp.where(siblings | same_rule, jnp.minimum(dist + 1, 4), dist)
    dist = jnp.where(valid, dist, 0)
    return rel, dist


class MultiHeadAttention(nn.Module):
    cfg: EditFormerConfig
    name_prefix: str

    @nn.compact
    def __call__(self, qx, kvx, qmask, kvmask, *, bias=None, train: bool):
        cfg = self.cfg
        dtype = _dtype(cfg.compute_dtype)
        hd = cfg.head_dim
        q = nn.Dense(cfg.d_model, use_bias=False, dtype=dtype, param_dtype=jnp.float32, name=f"{self.name_prefix}_q")(qx)
        k = nn.Dense(cfg.d_model, use_bias=False, dtype=dtype, param_dtype=jnp.float32, name=f"{self.name_prefix}_k")(kvx)
        v = nn.Dense(cfg.d_model, use_bias=False, dtype=dtype, param_dtype=jnp.float32, name=f"{self.name_prefix}_v")(kvx)
        q = q.reshape(q.shape[0], q.shape[1], cfg.heads, hd)
        k = k.reshape(k.shape[0], k.shape[1], cfg.heads, hd)
        v = v.reshape(v.shape[0], v.shape[1], cfg.heads, hd)
        mask = qmask[:, None, :, None] & kvmask[:, None, None, :]
        out = jax.nn.dot_product_attention(
            q, k, v, bias=bias, mask=mask,
            implementation=cfg.attention_impl,
        )
        out = out.reshape(out.shape[0], out.shape[1], cfg.d_model)
        out = nn.Dense(cfg.d_model, dtype=dtype, param_dtype=jnp.float32, name=f"{self.name_prefix}_o")(out)
        return nn.Dropout(rate=cfg.dropout, name=f"{self.name_prefix}_drop")(out, deterministic=not train)


class TransformerBlock(nn.Module):
    cfg: EditFormerConfig
    block_name: str
    structural: bool = False

    @nn.compact
    def __call__(self, x, mask, parent=None, depth=None, rule_id=None, *, train: bool):
        cfg = self.cfg
        y = nn.LayerNorm(dtype=jnp.float32, param_dtype=jnp.float32, name=f"{self.block_name}_ln1")(x).astype(_dtype(cfg.compute_dtype))
        bias = None
        if self.structural and cfg.use_structural_bias:
            rel, dist = structural_relations(parent, depth, rule_id, mask)
            rb = self.param(f"{self.block_name}_rel_bias", nn.initializers.zeros, (N_REL, cfg.heads), jnp.float32)
            db = self.param(f"{self.block_name}_dist_bias", nn.initializers.zeros, (N_DIST, cfg.heads), jnp.float32)
            bias = rb[rel] + db[dist]  # B,T,T,H
            bias = jnp.transpose(bias, (0, 3, 1, 2)).astype(y.dtype)
        y = MultiHeadAttention(cfg, f"{self.block_name}_attn")(y, y, mask, mask, bias=bias, train=train)
        x = x + y.astype(x.dtype)
        y = nn.LayerNorm(dtype=jnp.float32, param_dtype=jnp.float32, name=f"{self.block_name}_ln2")(x).astype(_dtype(cfg.compute_dtype))
        y = nn.Dense(cfg.ffn_dim, dtype=_dtype(cfg.compute_dtype), param_dtype=jnp.float32, name=f"{self.block_name}_ff1")(y)
        y = nn.gelu(y)
        y = nn.Dropout(rate=cfg.dropout, name=f"{self.block_name}_ffdrop1")(y, deterministic=not train)
        y = nn.Dense(cfg.d_model, dtype=_dtype(cfg.compute_dtype), param_dtype=jnp.float32, name=f"{self.block_name}_ff2")(y)
        y = nn.Dropout(rate=cfg.dropout, name=f"{self.block_name}_ffdrop2")(y, deterministic=not train)
        return x + y.astype(x.dtype)


class CrossFusionBlock(nn.Module):
    cfg: EditFormerConfig
    block_name: str

    @nn.compact
    def __call__(self, q, qmask, cases, cmask, *, train: bool):
        cfg = self.cfg
        y = nn.LayerNorm(dtype=jnp.float32, name=f"{self.block_name}_ln1")(q).astype(_dtype(cfg.compute_dtype))
        kv = nn.LayerNorm(dtype=jnp.float32, name=f"{self.block_name}_kvln")(cases).astype(_dtype(cfg.compute_dtype))
        y = MultiHeadAttention(cfg, f"{self.block_name}_cross")(y, kv, qmask, cmask, train=train)
        q = q + y.astype(q.dtype)
        y = nn.LayerNorm(dtype=jnp.float32, name=f"{self.block_name}_ln2")(q).astype(_dtype(cfg.compute_dtype))
        y = nn.Dense(cfg.ffn_dim, dtype=_dtype(cfg.compute_dtype), param_dtype=jnp.float32, name=f"{self.block_name}_ff1")(y)
        y = nn.gelu(y)
        y = nn.Dense(cfg.d_model, dtype=_dtype(cfg.compute_dtype), param_dtype=jnp.float32, name=f"{self.block_name}_ff2")(y)
        return q + y.astype(q.dtype)


class BonsaiEditFormer(nn.Module):
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
        return {
            "regression_logit": regression_logit,
            "flip_logit": flip_logit,
            "margin": margin,
            "worst_win": worst_win,
            "rank_residual": rank_residual,
            "rank": final_rank,
            "case_delta": case_delta,
        }


def count_params(params: Any) -> int:
    return sum(int(x.size) for x in jax.tree.leaves(params))
