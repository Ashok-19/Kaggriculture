"""Deterministic, hash-pinned donor extraction. Never execute notebook setup cells."""
from __future__ import annotations

import argparse
import ast
import copy
import functools
import base64
import gzip
import hashlib
import json
import struct
from pathlib import Path

from kaggriculture_bonsai.runtime import bonsai_kg_v4 as bkg
from kaggriculture_bonsai.runtime.route_codec import encode_action
from kaggriculture_bonsai.runtime.route_format import DESC, HEADER, PACK_MAGIC, PACK_VERSION
from .pack import ROUTES, STEPS, ShopRoutePack

NOTEBOOK_SHA256 = "1d7f6a17bad05dafe27002fb20a0a650603f571c2a3d0408431d1a7d0ab2701b"
SOURCE_SHA256 = "d6d74997dc5b483db63d8e39cafa1afeec0f366824e75107e109123f111e866b"
TAPES_SHA256 = "17d503f2fd20d59f9c0f14024d1e74a8add8bb9b5561d4d908b45deecb5495ef"
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
NOTEBOOK = ROOT / "public-notebooks/current-gold-spet-9.ipynb"
ASSETS = ROOT / "workspace/shop_router_0909/port"

# Donor shop identity is positional: the ordered first two unlocked shops.
YARN = bkg.SHOP_ID["YARN_STORE"]
YARN_SECOND = {"BAKERY": 3, "BRUNCH_SPOT": 4, "FARMERS_MARKET": 5, "ICE_CREAM_SHOP": 6,
               "PET_CAFE": 5, "PIZZA_SHOP": 7, "SMOOTHIE_SHOP": 8, "YARN_STORE": 12}
YARN_FIRST = {"BAKERY": 9, "BRUNCH_SPOT": 9, "FARMERS_MARKET": 1, "ICE_CREAM_SHOP": 9,
              "PET_CAFE": 10, "PIZZA_SHOP": 6, "SMOOTHIE_SHOP": 11}
ROUTE_STEP, FINAL_PLAN_STEP = 144, 648


def sha(data):
    return hashlib.sha256(data).hexdigest()


@functools.lru_cache(maxsize=2)
def _code_cells(notebook=NOTEBOOK):
    raw = Path(notebook).read_bytes()
    if sha(raw) != NOTEBOOK_SHA256:
        raise ValueError("pinned notebook SHA-256 mismatch")
    return tuple("".join(c["source"]) for c in json.loads(raw)["cells"] if c["cell_type"] == "code")


def source(notebook=NOTEBOOK):
    """The final Kaggle callable, exactly as the notebook writes it to main.py."""
    cells = _code_cells(notebook)
    sources = [c.split("\n", 1)[1].replace("\r\n", "\n") for c in cells if c.startswith("%%writefile main.py\n")]
    if len(sources) != 1 or sha(sources[0].encode()) != SOURCE_SHA256:
        raise ValueError("pinned reference main.py SHA-256 mismatch")
    return sources[0]


def tapes(notebook=NOTEBOOK):
    """Decode only the literal tape payload, never arbitrary notebook Python."""
    return copy.deepcopy(_tapes(notebook))


@functools.lru_cache(maxsize=2)
def _tapes(notebook):
    cell = next(c for c in _code_cells(notebook) if "tape_data" in c)
    assignment = next(n for n in ast.parse(cell).body
                      if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "tape_data" for t in n.targets))
    if not isinstance(assignment.value, ast.Constant) or not isinstance(assignment.value.value, str):
        raise ValueError("unexpected tape payload expression")
    raw = gzip.decompress(base64.b64decode(assignment.value.value))
    if sha(raw) != TAPES_SHA256:
        raise ValueError("pinned actions.json SHA-256 mismatch")
    decoded = json.loads(raw)
    if len(decoded) != ROUTES or any(len(t) != STEPS for t in decoded):
        raise ValueError("expected 13 complete, 719-turn action tapes")
    return decoded


@functools.lru_cache(maxsize=2)
def _donor_code(notebook):
    return compile(source(notebook), "shop_router_0909_reference_main.py", "exec")


def reference(notebook=NOTEBOOK):
    """The donor callable itself, backed by the donor's own decoded tapes."""
    namespace = {}
    exec(_donor_code(notebook), namespace)
    policy = namespace["Policy"].__new__(namespace["Policy"])
    # The donor deep-copies each tape entry before touching it and never writes
    # back, so instances may share one decoded copy.
    policy.tapes = _tapes(notebook)
    policy.players = {}
    namespace["_POLICY"] = policy
    return namespace


def controller():
    """Positional two-shop lookup latched at 144, final continuation latched at 648."""
    def cond(lhs, op, rhs):
        return {"type": "Condition", "lhs": lhs, "op": op, "rhs": rhs}
    def shop(index):
        return {"feature": "SHOP_AT", "arg": index}
    def put(value):
        return {"type": "SetBlackboard", "slot": "route", "value": value}
    def lookup(index, table):
        # Fails when no donor key matches, so the caller falls through to plan 0.
        return {"type": "Selector", "children": [
            {"type": "Sequence", "children": [cond(shop(index), "EQ", bkg.SHOP_ID[name]), put(plan)]}
            for name, plan in table.items()]}
    select = {"type": "Selector", "children": [
        {"type": "Sequence", "children": [cond(shop(1), "EQ", YARN), lookup(0, YARN_SECOND)]},
        {"type": "Sequence", "children": [cond(shop(0), "EQ", YARN), lookup(1, YARN_FIRST)]},
        put(0)]}
    tree = {"type": "Sequence", "children": [
        {"type": "If", "condition": cond({"feature": "STEP"}, "EQ", ROUTE_STEP), "then": select, "else": {"type": "Success"}},
        {"type": "If", "condition": cond({"feature": "STEP"}, "EQ", FINAL_PLAN_STEP), "then": put(2), "else": {"type": "Success"}}]}
    return {"schema_version": 4, "name": "BKG4_SHOP_ROUTER_0909",
            "blackboard": [{"name": "route", "type": "I32", "initial": 0}], "tree": tree}


def generate(out=ASSETS, notebook=NOTEBOOK):
    donor = tapes(notebook)
    payload = b"".join(DESC.pack(rid, STEPS, rid * STEPS) for rid in range(ROUTES))
    records = {rid: b"".join(encode_action(a) for a in donor[rid]) for rid in range(ROUTES)}
    payload += b"".join(records.values())
    actions = ROUTES * STEPS
    pack = HEADER.pack(PACK_MAGIC, PACK_VERSION, ROUTES, actions, hashlib.sha256(payload).digest()) + payload
    arity = b"".join(
        struct.pack("<H", sum(1 << i for i, u in enumerate([a["farmer"]] + a["hands"])
                              if u[0] in ("PICKUP", "PLACE") and len(u) == 2))
        for rid in range(ROUTES) for a in donor[rid])
    spec = controller()
    compiled = bkg.compile_policy(spec)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "routes.bkv6").write_bytes(pack)
    (out / "arity.bin").write_bytes(arity)
    bkg.save_compiled(compiled, out / "controller.bkgc")
    manifest = {
        "candidate": "BKG4_SHOP_ROUTER_0909", "status": "EXACT_PORT_FROZEN_PARENT",
        "source_notebook": str(NOTEBOOK.relative_to(ROOT)), "source_notebook_sha256": NOTEBOOK_SHA256,
        "source_agent_sha256": SOURCE_SHA256, "source_tapes_sha256": TAPES_SHA256,
        "official_engine_version": "1.32.7",
        "cpu_abi_version": bkg.ABI_VERSION, "cuda_abi_version": bkg.ABI_VERSION,
        "controller_sha256": sha((out / "controller.bkgc").read_bytes()),
        "controller_semantic_hash": compiled.semantic_hash,
        "controller_nodes": len(compiled.nodes), "controller_state_slots": compiled.state_count,
        "controller_blackboard_slots": len(compiled.bb),
        "arity_sha256": sha(arity),
        "route_pack": {"pack_sha256": sha(pack), "route_ids": list(range(ROUTES)),
                       "format": "BKV6RP1", "version": PACK_VERSION},
        "routes": [{"id": rid, "donor": "yhay81/shop-router-0909", "actions": STEPS,
                    "canonical_json_sha256": sha(json.dumps(donor[rid], sort_keys=True, separators=(",", ":")).encode()),
                    "packed_actions_sha256": sha(records[rid])} for rid in range(ROUTES)],
    }
    reader = ShopRoutePack(out / "routes.bkv6", manifest, arity)
    for rid, tape in enumerate(donor):
        for step, action in enumerate(tape):
            if reader.raw_action(rid, step) != action:
                raise ValueError(f"lossy route representation: route={rid}, step={step}")
    (out / "identity.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (out / "controller.json").write_text(json.dumps(spec, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--out", type=Path, default=ASSETS)
    print(json.dumps(generate(parser.parse_args().out), indent=2))
