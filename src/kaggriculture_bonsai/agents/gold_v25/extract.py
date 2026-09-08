"""Deterministic, hash-pinned donor extraction. Never execute notebook setup cells."""
from __future__ import annotations

import argparse
import ast
import base64
import hashlib
import json
import struct
import zlib
from pathlib import Path

from kaggriculture_bonsai.runtime import bonsai_kg_v4 as bkg
from kaggriculture_bonsai.runtime.route_codec import encode_action
from kaggriculture_bonsai.runtime.route_format import DESC, HEADER, PACK_MAGIC, PACK_VERSION
from .pack import GoldRoutePack

NOTEBOOK_SHA256 = "330085c464e992b4e1cfd4c0cbb7c4d5f97a67fd5c6f754c56bec5a6eed12820"
SOURCE_SHA256 = "bb3109331d4087d5e5890515f9b1d1e0d4f9ea6015ee3d1b172a93b8b1d8dbb4"
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
NOTEBOOK = ROOT / "public-notebooks/current-gold-sept-8.ipynb"
ASSETS = ROOT / "workspace/gold_v25_port"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def source(notebook=NOTEBOOK):
    raw = Path(notebook).read_bytes()
    if sha(raw) != NOTEBOOK_SHA256:
        raise ValueError("pinned notebook SHA-256 mismatch")
    cells = ["".join(c["source"]) for c in json.loads(raw)["cells"] if c["cell_type"] == "code"]
    sources = [c.split("\n", 1)[1].replace("\r\n", "\n") for c in cells if c.startswith("%%writefile main.py\n")]
    if len(sources) != 1 or sha(sources[0].encode()) != SOURCE_SHA256:
        raise ValueError("pinned reference main.py SHA-256 mismatch")
    return sources[0]


def reference():
    namespace = {}
    exec(compile(source(), "gold_v25_reference_main.py", "exec"), namespace)
    return namespace


def routes(src):
    tree = ast.parse(src)
    assignment = next(n for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "_PAYLOAD" for t in n.targets))
    # Decode only the literal payload, not arbitrary notebook Python.
    strings = [n.value for n in ast.walk(assignment.value) if isinstance(n, ast.Constant) and isinstance(n.value, str)]
    if len(strings) != 1:
        raise ValueError("unexpected payload expression")
    payload = json.loads(zlib.decompress(base64.b85decode(strings[0])))
    result = {0: payload["base"]}
    for rid, patch in payload["patches"].items():
        tape = list(result[0])
        for step, action in patch:
            tape[step] = action
        result[int(rid)] = tape
    if sorted(result) != [0, 1, 2, 3] or any(len(t) != 719 for t in result.values()):
        raise ValueError("unexpected route identities/counts")
    return result


def controller():
    def cond(lhs, op, rhs):
        return {"type": "Condition", "lhs": lhs, "op": op, "rhs": rhs}
    def put(slot, value):
        return {"type": "SetBlackboard", "slot": slot, "value": value}
    def branch(condition, yes, no):
        return {"type": "If", "condition": condition, "then": yes, "else": no}
    children = []
    for step, latch, predicate, yes, no in (
        (144, "day6", cond({"feature": "SHOP_SEEN", "arg": 7}, "EQ", 1), 1, 0),
        (648, "day27", cond({"feature": "MARKET_INVENTORY", "arg": "EGG"}, "LE", 9888), 2, 3),
    ):
        ready = {"type": "Sequence", "children": [cond({"feature": "STEP"}, "GE", step), cond({"blackboard": latch}, "EQ", 0)]}
        action = {"type": "Sequence", "children": [branch(predicate, put("route", yes), put("route", no)), put(latch, 1)]}
        children.append(branch(ready, action, {"type": "Success"}))
    return {"schema_version": 4, "name": "BKG4_GOLD_V25", "blackboard": [{"name": name, "type": "I32", "initial": 0} for name in ("route", "day6", "day27")], "tree": {"type": "Sequence", "children": children}}


def overlay_source(src):
    """Mechanical subset, retaining the original license and function text."""
    tree = ast.parse(src)
    functions = {"_get", "_int", "_step_of", "_shed_adjacent", "_tile_at", "_is_noop", "make_agent"}
    constants = {"PRODUCTS", "SEED_PRICE", "ANIMAL_COST", "ANIMAL_STRUCTURE", "LAND_PRICES", "MOVES", "FRONT_RUN_ITEMS", "LAST_ACT_STEP", "PASS_ACTION", "DEFAULT_SETTINGS"}
    methods = {"__init__", "_state", "_route_action", "act", "_hand_align", "_weed_repair", "_projected_shed", "_apply_suppression", "_add_sell", "_sell_lead"}
    lines = src.splitlines(keepends=True)
    output = "".join(line.rstrip() + "\n" for line in lines[:tree.body[0].lineno - 1]) + "\n# Mechanically extracted active chassis subset; regenerate with extract.py.\n"
    for n in tree.body:
        keep = isinstance(n, (ast.Import, ast.ImportFrom)) and (isinstance(n, ast.ImportFrom) or any(a.name == "copy" for a in n.names))
        keep |= isinstance(n, ast.FunctionDef) and n.name in functions
        keep |= isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in constants for t in n.targets)
        keep |= isinstance(n, ast.ClassDef) and n.name == "_View"
        if isinstance(n, ast.ClassDef) and n.name == "Chassis":
            output += "class Chassis:\n"
            for m in n.body:
                if isinstance(m, ast.FunctionDef) and m.name in methods:
                    start = min([m.lineno] + [d.lineno for d in m.decorator_list])
                    output += "".join(lines[start - 1:m.end_lineno]) + "\n"
        elif keep:
            output += "".join(lines[n.lineno - 1:n.end_lineno]) + "\n\n"
    return output.rstrip() + "\n"


def generate(out=ASSETS, write_overlay=False):
    src = source()
    tapes = routes(src)
    payload = b"".join(DESC.pack(rid, 719, rid * 719) for rid in range(4))
    records = {rid: b"".join(encode_action(a) for a in tapes[rid]) for rid in range(4)}
    payload += b"".join(records.values())
    pack = HEADER.pack(PACK_MAGIC, PACK_VERSION, 4, 2876, hashlib.sha256(payload).digest()) + payload
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "routes.bkv6").write_bytes(pack)
    arity = b"".join(struct.pack("<H", sum(1 << i for i, u in enumerate([a["farmer"]] + a["hands"]) if u[0] in ("PICKUP", "PLACE") and len(u) == 2)) for rid in range(4) for a in tapes[rid])
    (out / "arity.bin").write_bytes(arity)
    spec = controller()
    cp = bkg.compile_policy(spec)
    bkg.save_compiled(cp, out / "controller.bkgc")
    manifest = {
        "candidate": "BKG4_GOLD_V25", "status": "UNVALIDATED_NOT_PROMOTED",
        "source_notebook": str(NOTEBOOK.relative_to(ROOT)), "source_notebook_sha256": NOTEBOOK_SHA256,
        "source_agent_sha256": SOURCE_SHA256, "official_engine_version": "1.32.7",
        "cpu_abi_version": bkg.ABI_VERSION, "cuda_abi_version": bkg.ABI_VERSION,
        "controller_sha256": sha((out / "controller.bkgc").read_bytes()),
        "controller_semantic_hash": cp.semantic_hash,
        "arity_sha256": sha(arity),
        "controller_nodes": len(cp.nodes), "controller_state_slots": cp.state_count, "controller_blackboard_slots": len(cp.bb),
        "route_pack": {"pack_sha256": sha(pack), "route_ids": list(range(4)), "format": "BKV6RP1", "version": PACK_VERSION},
        "routes": [{"id": rid, "donor": "yhay81/shop-router-0908", "actions": 719,
                    "canonical_json_sha256": sha(json.dumps(tapes[rid], sort_keys=True, separators=(",", ":")).encode()),
                    "packed_actions_sha256": sha(records[rid])} for rid in range(4)],
    }
    reader = GoldRoutePack(out / "routes.bkv6", manifest, arity)
    for rid, tape in tapes.items():
        for step, action in enumerate(tape):
            if reader.raw_action(rid, step) != action:
                raise ValueError(f"lossy route representation: route={rid}, step={step}")
    (out / "identity.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (out / "controller.json").write_text(json.dumps(spec, indent=2) + "\n")
    if write_overlay:
        (HERE / "overlay.py").write_text(overlay_source(src))
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--out", type=Path, default=ASSETS)
    parser.add_argument("--write-overlay", action="store_true")
    args = parser.parse_args()
    print(json.dumps(generate(args.out, args.write_overlay), indent=2))
