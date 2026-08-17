from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

HERE = Path(__file__).resolve().parent
BKG_ROOT = HERE.parents[1]
REPO_ROOT = BKG_ROOT.parent
sys.path.insert(0, str(BKG_ROOT / "python"))
import bonsai_kg as bkg

ROUTE_IDS = (29, 64, 32, 79, 46, 63, 22, 34, 77, 70, 23)
LEAVES = ("A1", "A2", "B", "C1", "C2", "D1", "D2", "E", "F", "G", "H", "I", "J1", "J2", "K")
LEAF_ID = {name: i for i, name in enumerate(LEAVES)}
LEAF_NONE = 255
MAX_SAVED_HANDS = 12
MAX_MARKET = 10
PACK_MAGIC = b"BKV6RP1\0"
PACK_VERSION = 1
HEADER = struct.Struct("<8sHHI32s")
DESC = struct.Struct("<HHI")
UNIT = struct.Struct("<BBh")
ORDER = struct.Struct("<BBH")
ACTION_SIZE = 2 + UNIT.size * (1 + MAX_SAVED_HANDS) + ORDER.size * MAX_MARKET

UNIT_OPS = bkg.UNIT_OPS
MARKET_OPS = bkg.MARKET_OPS
ITEM_ID = bkg.ITEM_ID


def condition(feature: str | None = None, op: str = "EQ", rhs: int = 0, arg: int | None = None,
              bb: str | None = None) -> Dict[str, Any]:
    if bb is not None:
        lhs: Any = {"blackboard": bb}
    else:
        lhs = {"feature": feature}
        if arg is not None:
            lhs["arg"] = arg
    return {"type": "Condition", "lhs": lhs, "op": op, "rhs": rhs}


def iff(test: Dict[str, Any], yes: Dict[str, Any], no: Dict[str, Any]) -> Dict[str, Any]:
    return {"type": "If", "condition": test, "then": yes, "else": no}


def success() -> Dict[str, Any]:
    return {"type": "Success"}


def set_leaf_route(leaf: str, route: int) -> Dict[str, Any]:
    return {
        "type": "SetBlackboard", "slot": "leaf", "value": LEAF_ID[leaf],
        "lifecycle": {"on_success": [{"op": "set", "slot": "route", "value": route}]},
    }


def set_route(route: int) -> Dict[str, Any]:
    return {"type": "SetBlackboard", "slot": "route", "value": route}


def decision144() -> Dict[str, Any]:
    shop0 = lambda op, rhs: condition("SHOP_AT", op, rhs, 0)
    shop1 = lambda op, rhs: condition("SHOP_AT", op, rhs, 1)
    seen = lambda sid, op="LE", rhs=0: condition("SHOP_SEEN", op, rhs, sid)
    price = lambda item, op, rhs: condition("MARKET_PRICE", op, rhs, item)
    money = lambda own, op, rhs: condition("OWN_MONEY" if own else "OPP_MONEY", op, rhs)

    branch_no_yarn = iff(
        price(bkg.ITEM_ID["MILK"], "LE", 192),
        iff(
            shop1("LE", 2),
            iff(shop0("LE", 2), set_leaf_route("A1", 29), set_leaf_route("A2", 64)),
            iff(
                money(True, "LE", 310),
                set_leaf_route("B", 64),
                iff(shop1("LE", 5), set_leaf_route("C1", 64), set_leaf_route("C2", 32)),
            ),
        ),
        iff(
            seen(5),
            iff(
                shop1("LE", 2),
                iff(money(False, "LE", 56), set_leaf_route("D1", 79), set_leaf_route("D2", 32)),
                set_leaf_route("E", 64),
            ),
            iff(shop1("LE", 2), set_leaf_route("F", 70), set_leaf_route("G", 63)),
        ),
    )
    branch_yarn = iff(
        price(bkg.ITEM_ID["TOMATO"], "LE", 62),
        iff(
            shop1("LE", 3),
            set_leaf_route("H", 22),
            iff(
                shop0("LE", 5),
                set_leaf_route("I", 34),
                iff(seen(6), set_leaf_route("J1", 77), set_leaf_route("J2", 34)),
            ),
        ),
        set_leaf_route("K", 46),
    )
    root = iff(seen(7), branch_no_yarn, branch_yarn)
    root["lifecycle"] = {"on_success": [{"op": "set", "slot": "stage144_done", "value": 1}]}
    return root


def decision216() -> Dict[str, Any]:
    seen = lambda sid, op="LE", rhs=0: condition("SHOP_SEEN", op, rhs, sid)
    shop2 = lambda op, rhs: condition("SHOP_AT", op, rhs, 2)
    own_money = lambda op, rhs: condition("OWN_MONEY", op, rhs)
    leaf_is = lambda leaf: condition(op="EQ", rhs=LEAF_ID[leaf], bb="leaf")

    a1 = iff(
        seen(3),
        iff(seen(6), set_route(29), set_route(32)),
        iff(own_money("LE", 715), set_route(29), set_route(23)),
    )
    b = iff(shop2("LE", 6), set_route(64), set_route(46))
    c1 = iff(shop2("LE", 6), set_route(64), set_route(34))
    d2 = iff(shop2("LE", 6), set_route(32), set_route(46))
    e = iff(
        seen(4),
        iff(seen(5), set_route(79), set_route(63)),
        iff(shop2("LE", 6), set_route(64), set_route(46)),
    )
    root = iff(
        leaf_is("A1"), a1,
        iff(leaf_is("B"), b,
            iff(leaf_is("C1"), c1,
                iff(leaf_is("D2"), d2,
                    iff(leaf_is("E"), e, success())))),
    )
    root["lifecycle"] = {"on_success": [{"op": "set", "slot": "stage216_done", "value": 1}]}
    return root


def controller_policy() -> Dict[str, Any]:
    stage144 = iff(
        condition(op="EQ", rhs=1, bb="stage144_done"),
        success(),
        iff(condition("STEP", "LT", 144), success(), decision144()),
    )
    stage216 = iff(
        condition(op="EQ", rhs=1, bb="stage216_done"),
        success(),
        iff(condition("STEP", "LT", 216), success(), decision216()),
    )
    return {
        "schema_version": 3,
        "name": "stateful_core_v6_bkg3_compat_controller",
        "blackboard": [
            {"name": "leaf", "type": "ENUM8", "initial": LEAF_NONE},
            {"name": "route", "type": "ENUM8", "initial": 64},
            {"name": "stage144_done", "type": "BOOL", "initial": 0},
            {"name": "stage216_done", "type": "BOOL", "initial": 0},
        ],
        "tree": {"type": "Sequence", "children": [stage144, stage216]},
    }


def encode_unit(action: Any) -> bytes:
    a = list(action or ["PASS"])
    op_name = str(a[0]) if a else "PASS"
    if op_name not in UNIT_OPS:
        raise ValueError(f"unknown unit op {op_name}")
    op = UNIT_OPS[op_name]
    arg = 0
    n = 1
    if len(a) >= 2:
        if isinstance(a[1], str):
            if a[1] not in ITEM_ID:
                raise ValueError(f"unknown unit item {a[1]}")
            arg = ITEM_ID[a[1]]
        else:
            n = int(a[1])
    if len(a) >= 3:
        n = int(a[2])
    if not -32768 <= n <= 32767:
        raise ValueError(f"unit quantity out of int16 range: {n}")
    return UNIT.pack(op, arg, n)


def encode_order(order: Any) -> bytes:
    a = list(order or [])
    if not a:
        return ORDER.pack(0, 0, 0)
    op_name = str(a[0])
    if op_name not in MARKET_OPS:
        raise ValueError(f"unknown market op {op_name}")
    op = MARKET_OPS[op_name]
    if op_name in ("HIRE", "BUY_LAND"):
        return ORDER.pack(op, 0, 1)
    if len(a) < 3 or str(a[1]) not in ITEM_ID:
        raise ValueError(f"invalid market order {a}")
    n = int(a[2])
    if not 0 <= n <= 65535:
        raise ValueError(f"market quantity out of uint16 range: {n}")
    return ORDER.pack(op, ITEM_ID[str(a[1])], n)


def encode_action(action: Dict[str, Any]) -> bytes:
    hands = list(action.get("hands", []) or [])
    market = list(action.get("market", []) or [])
    if len(hands) > MAX_SAVED_HANDS or len(market) > MAX_MARKET:
        raise ValueError("route action exceeds compatibility pack capacity")
    out = bytearray(struct.pack("<BB", len(hands), len(market)))
    out += encode_unit(action.get("farmer") or ["PASS"])
    for i in range(MAX_SAVED_HANDS):
        out += encode_unit(hands[i]) if i < len(hands) else UNIT.pack(UNIT_OPS["PASS"], 0, 1)
    for i in range(MAX_MARKET):
        out += encode_order(market[i]) if i < len(market) else ORDER.pack(0, 0, 0)
    if len(out) != ACTION_SIZE:
        raise AssertionError((len(out), ACTION_SIZE))
    return bytes(out)


def build_route_pack(route_dir: Path) -> Tuple[bytes, Dict[str, Any]]:
    descs: List[Tuple[int, int, int]] = []
    records: List[bytes] = []
    source_files: List[Dict[str, Any]] = []
    first = 0
    for rid in ROUTE_IDS:
        path = route_dir / f"route{rid}.json"
        raw = path.read_bytes()
        obj = json.loads(raw)
        actions = obj.get("actions")
        if not isinstance(actions, list) or not actions:
            raise ValueError(f"route {rid} has no actions")
        descs.append((rid, len(actions), first))
        first += len(actions)
        records.extend(encode_action(a) for a in actions)
        source_files.append({"route_id": rid, "file": path.name, "sha256": hashlib.sha256(raw).hexdigest(), "actions": len(actions)})
    payload = bytearray()
    for d in descs:
        payload += DESC.pack(*d)
    for r in records:
        payload += r
    payload_hash = hashlib.sha256(payload).digest()
    pack = HEADER.pack(PACK_MAGIC, PACK_VERSION, len(descs), len(records), payload_hash) + payload
    meta = {
        "format": "BKV6RP1",
        "version": PACK_VERSION,
        "route_ids": list(ROUTE_IDS),
        "route_count": len(descs),
        "action_records": len(records),
        "action_record_bytes": ACTION_SIZE,
        "payload_sha256": payload_hash.hex(),
        "pack_sha256": hashlib.sha256(pack).hexdigest(),
        "source_files": source_files,
    }
    return bytes(pack), meta


def combined_identity(controller_hash: str, route_pack_hash: str) -> str:
    h = hashlib.sha256()
    h.update(b"BKG3-V6-COMPAT\0")
    h.update(bytes.fromhex(controller_hash))
    h.update(bytes.fromhex(route_pack_hash))
    return h.hexdigest()


def build(route_dir: Path, out_dir: Path) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    policy = controller_policy()
    controller_json = out_dir / "controller.json"
    controller_json.write_text(json.dumps(policy, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    compiled = bkg.compile_policy(policy)
    controller_bin = out_dir / "controller.bkgc"
    bkg.save_compiled(compiled, controller_bin)
    pack, route_meta = build_route_pack(route_dir)
    pack_path = out_dir / "routes.bkv6"
    pack_path.write_bytes(pack)
    identity = {
        "schema": 1,
        "oracle_python": "scratch/bonsai_cpu/stateful_core_v6/main.py",
        "oracle_python_sha256": hashlib.sha256((route_dir / "main.py").read_bytes()).hexdigest(),
        "old_cuda": "scratch/bonsai_next/v6_cuda_eval.cu",
        "old_cuda_sha256": hashlib.sha256((REPO_ROOT / "scratch/bonsai_next/v6_cuda_eval.cu").read_bytes()).hexdigest(),
        "controller_semantic_hash": compiled.semantic_hash,
        "controller_nodes": len(compiled.nodes),
        "controller_state_slots": compiled.state_count,
        "controller_compiled_bytes": compiled.compiled_bytes,
        "route_pack": route_meta,
        "combined_agent_identity": combined_identity(compiled.semantic_hash, route_meta["pack_sha256"]),
        "leaf_enum": {**LEAF_ID, "NONE": LEAF_NONE},
        "bb_slots": {"leaf": 0, "route": 1, "stage144_done": 2, "stage216_done": 3},
    }
    (out_dir / "identity.json").write_text(json.dumps(identity, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    identity_hpp = f'''#pragma once\nnamespace bonsaikg_v6 {{\ninline constexpr const char* CONTROLLER_SEMANTIC_HASH = "{compiled.semantic_hash}";\ninline constexpr const char* ROUTE_PACK_SHA256 = "{route_meta['pack_sha256']}";\ninline constexpr const char* COMBINED_AGENT_IDENTITY = "{identity['combined_agent_identity']}";\ninline constexpr int LEAF_NONE = {LEAF_NONE};\n}} // namespace bonsaikg_v6\n'''
    (out_dir / "identity.hpp").write_text(identity_hpp, encoding="utf-8")
    return identity


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--route-dir", type=Path, default=REPO_ROOT / "scratch/bonsai_cpu/stateful_core_v6")
    ap.add_argument("--out", type=Path, default=HERE / "assets")
    args = ap.parse_args()
    identity = build(args.route_dir.resolve(), args.out.resolve())
    print(json.dumps(identity, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
