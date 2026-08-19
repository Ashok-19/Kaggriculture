from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

HERE = Path(__file__).resolve().parent
BKG_ROOT = HERE.parents[1]
REPO_ROOT = BKG_ROOT.parent
V6 = BKG_ROOT / "integrations" / "stateful_v6"
sys.path.insert(0, str(BKG_ROOT / "python"))
sys.path.insert(0, str(V6))

import bonsai_kg as bkg3
import bonsai_kg_v4 as bkg4
from build_assets import ACTION_SIZE, DESC, HEADER, PACK_MAGIC, PACK_VERSION, encode_action

# These are the exact tapes used by the accepted H4/public-lane compatibility stack.
# They are packed into a small immutable binary so the runtime does not depend on scratch.
LEGACY_TAPES: Tuple[Tuple[int, str, str], ...] = (
    (200, "H4_P", "scratch/live_top10/aug16_1549_refresh/tapes_dev/Yubo-best_s1_ep93580448.json"),
    (201, "H4_A", "scratch/bonsai_kg_v6_reinforcement/population/native_tapes/0005_peikopon_s0_ep94018272.json"),
    (202, "H4_W86", "scratch/bonsai_kg_v6_reinforcement/population/native_tapes/3250_whitelok_s0_ep93993645.json"),
    (203, "H4_W253", "scratch/bonsai_kg_v6_reinforcement/population/native_tapes/0024_Jince_s0_ep94010384.json"),
    (204, "H4_FY", "scratch/bonsai_kg_v6_reinforcement/population/native_tapes/0051_Tomas_Escobar_Rivera_s1_ep94020940.json"),
    (205, "H4_ICE164", "scratch/bonsai_kg_v6_reinforcement/population/native_tapes/0046_wenyuan_Guo_s1_ep93977817.json"),
    (206, "H4_ICE120", "scratch/bonsai_kg_v6_reinforcement/population/native_tapes/0019_dk2lone_s1_ep94001610.json"),
    (207, "H4_FF17", "scratch/live_top10/aug16_latest_refresh/tapes_dev/junichiro_morita_s0_ep93427728.json"),
    (208, "EXT_D36", "scratch/live_top10/aug16_kawashigi_new55540317/tapes_dev20/kawashigi_new55540317_s1_ep93536446.json"),
    (209, "EXT_D85", "scratch/bonsai_kg_v6_reinforcement/population/native_tapes/0001_team_s0_ep94019141.json"),
    (210, "EXT_D72", "scratch/live_top10/aug16_2044_refresh/tapes_dev/Thomas_s0_ep93655746.json"),
)


def _read_actions(path: Path) -> List[Dict[str, Any]]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    actions = obj.get("actions")
    if not isinstance(actions, list) or not actions:
        raise ValueError(f"legacy tape has no actions: {path}")
    return actions


def build_legacy_pack() -> Tuple[bytes, Dict[str, Any]]:
    descs: List[Tuple[int, int, int]] = []
    records: List[bytes] = []
    sources: List[Dict[str, Any]] = []
    first = 0
    for route_id, name, rel in LEGACY_TAPES:
        path = REPO_ROOT / rel
        raw = path.read_bytes()
        actions = _read_actions(path)
        descs.append((route_id, len(actions), first))
        first += len(actions)
        for action in actions:
            normalized = dict(action)
            normalized["market"] = [([] if isinstance(o, list) and o and o[0] == "PASS" else o) for o in list(action.get("market") or [])]
            records.append(encode_action(normalized))
        sources.append({
            "route_id": route_id,
            "name": name,
            "source": rel,
            "sha256": hashlib.sha256(raw).hexdigest(),
            "actions": len(actions),
        })
    payload = bytearray()
    for desc in descs:
        payload += DESC.pack(*desc)
    for record in records:
        payload += record
    payload_hash = hashlib.sha256(payload).digest()
    pack = HEADER.pack(PACK_MAGIC, PACK_VERSION, len(descs), len(records), payload_hash) + payload
    return bytes(pack), {
        "format": "BKV6RP1",
        "version": PACK_VERSION,
        "route_ids": [x[0] for x in LEGACY_TAPES],
        "route_names": {str(x[0]): x[1] for x in LEGACY_TAPES},
        "route_count": len(descs),
        "action_records": len(records),
        "action_record_bytes": ACTION_SIZE,
        "payload_sha256": payload_hash.hex(),
        "pack_sha256": hashlib.sha256(pack).hexdigest(),
        "source_files": sources,
    }


def build_controller(module: Dict[str, Any] | None = None) -> bkg4.CompiledPolicy:
    source3 = bkg3.load_compiled(V6 / "assets" / "controller.bkgc")
    source4 = bkg4.bkg3_to_bkg4_source(source3)
    source4["name"] = "bkg4_champion_controller"
    source4["provenance"] = {
        "foundation": "stateful_v6/controller.bkgc",
        "foundation_semantic_hash": source3.semantic_hash,
        "bkg4_vm_commit": "8381e027646ba6494cbe4dd06a810ea55ec83575",
    }
    if module is not None:
        module_tree = module.get("tree") if isinstance(module, dict) and "tree" in module else module
        module_bb = module.get("blackboard", []) if isinstance(module, dict) else []
        if module_bb:
            source4.setdefault("blackboard", []).extend(module_bb)
        root = source4["tree"]
        if root.get("type") != "Sequence":
            root = {"type": "Sequence", "children": [root]}
            source4["tree"] = root
        root.setdefault("children", []).append(module_tree)
    return bkg4.compile_policy(source4)


def combined_identity(controller_hash: str, v6_pack_hash: str, legacy_pack_hash: str, runtime_profile: Dict[str, Any] | None = None) -> str:
    h = hashlib.sha256()
    h.update(b"BKG4-CHAMPION-BASE0\0")
    h.update(bytes.fromhex(controller_hash))
    h.update(bytes.fromhex(v6_pack_hash))
    h.update(bytes.fromhex(legacy_pack_hash))
    if runtime_profile is not None:
        h.update(b"\0BKG4-RUNTIME-RECONCILER\0")
        h.update(json.dumps(runtime_profile, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    return h.hexdigest()


def resolve_runtime_reconciler(module: Dict[str, Any] | None) -> Dict[str, Any] | None:
    if not module or not isinstance(module.get("runtime_reconciler"), dict):
        return None
    source3 = bkg3.load_compiled(V6 / "assets" / "controller.bkgc")
    base4 = bkg4.bkg3_to_bkg4_source(source3)
    names = [str(x["name"]) for x in base4.get("blackboard", [])]
    names.extend(str(x["name"]) for x in module.get("blackboard", []))
    if len(names) != len(set(names)):
        raise ValueError("runtime reconciler blackboard names are not unique")
    name_to_slot = {name: i for i, name in enumerate(names)}
    raw = dict(module["runtime_reconciler"])
    defer_names = list(raw.get("defer_slots") or [])
    if raw.get("kind") == "feed_deferral_v1":
        if not defer_names:
            raise ValueError("feed_deferral_v1 requires defer_slots")
        raw["defer_slot_names"] = defer_names
        raw["defer_slots"] = [name_to_slot[name] for name in defer_names]
    else:
        raise ValueError(f"unsupported runtime reconciler {raw.get('kind')!r}")
    return raw


def build(out_dir: Path, module_json: Path | None = None) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    module = json.loads(module_json.read_text(encoding="utf-8")) if module_json else None
    runtime_reconciler = resolve_runtime_reconciler(module)
    cp = build_controller(module)
    controller_path = out_dir / "controller.bkgc"
    bkg4.save_compiled(cp, controller_path, out_dir / "controller.manifest.json")
    (out_dir / "controller.json").write_text(
        json.dumps(bkg4.decompile_policy(cp), sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )

    legacy_pack, legacy_meta = build_legacy_pack()
    legacy_path = out_dir / "legacy_routes.bkv6"
    legacy_path.write_bytes(legacy_pack)

    v6_identity = json.loads((V6 / "assets" / "identity.json").read_text(encoding="utf-8"))
    v6_pack_hash = str(v6_identity["route_pack"]["pack_sha256"])
    identity = {
        "schema": 1,
        "lineage": "BKG4_BASE_0" if module is None else str((module.get("manifest") or {}).get("module", "BKG4_CHALLENGER")),
        "bkg4_abi": 4,
        "feature_registry": 1,
        "bkg4_vm_frozen_commit": "8381e027646ba6494cbe4dd06a810ea55ec83575",
        "controller_semantic_hash": cp.semantic_hash,
        "controller_nodes": len(cp.nodes),
        "controller_state_slots": cp.state_count,
        "controller_bb_slots": len(cp.bb),
        "controller_max_depth": cp.max_depth,
        "controller_max_visits": cp.analysis[0].max_visits,
        "controller_max_live_intents": cp.max_live_intents,
        "v6_route_pack": {
            "path": "../stateful_v6/assets/routes.bkv6",
            "pack_sha256": v6_pack_hash,
        },
        "legacy_route_pack": legacy_meta,
        "legacy_semantics": "h4_lowcash_p60_r86_r253_fy172_ice164_v120_ff17 + h4max_4branch",
        "engine_exploits": "FERT81 -> CARROT-v3 -> EOD shed-overflow rescue",
        "combined_agent_identity": combined_identity(cp.semantic_hash, v6_pack_hash, legacy_meta["pack_sha256"], runtime_reconciler),
        "module_source": (str(module_json.resolve().relative_to(REPO_ROOT)) if module_json and module_json.resolve().is_relative_to(REPO_ROOT) else str(module_json) if module_json else None),
        "module_manifest": module.get("manifest") if module else None,
        "runtime_reconciler": runtime_reconciler,
    }
    (out_dir / "identity.json").write_text(json.dumps(identity, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    (out_dir / "identity.hpp").write_text(
        "#pragma once\nnamespace bonsaikg4_champion {\n"
        f'inline constexpr const char* CONTROLLER_SEMANTIC_HASH = "{cp.semantic_hash}";\n'
        f'inline constexpr const char* LEGACY_ROUTE_PACK_SHA256 = "{legacy_meta["pack_sha256"]}";\n'
        f'inline constexpr const char* COMBINED_AGENT_IDENTITY = "{identity["combined_agent_identity"]}";\n'
        "}\n",
        encoding="utf-8",
    )
    return identity


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=HERE / "assets")
    ap.add_argument("--module-json", type=Path)
    args = ap.parse_args()
    print(json.dumps(build(args.out.resolve(), args.module_json.resolve() if args.module_json else None), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
