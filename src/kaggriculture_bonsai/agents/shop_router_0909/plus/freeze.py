"""Freeze one screened candidate as the active composition, with its identity."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .agent import ACTIVE, ShopRouter0909Plus
from ..reference_port.extract import ASSETS


def freeze(spec, status="SCREENED", path=ACTIVE, notes=None):
    agent = ShopRouter0909Plus(spec)
    identity = agent.identity
    parent = json.loads((ASSETS / "identity.json").read_text())
    payload = {
        "identity": "BKG4_SHOP_ROUTER_0909_PLUS",
        "parent": parent["candidate"],
        "parent_controller_semantic_hash": parent["controller_semantic_hash"],
        "parent_route_pack_sha256": parent["route_pack"]["pack_sha256"],
        "candidate": spec,
        "status": status,
        "gate_semantic_hash": identity["semantic_hash"],
        "gate_sha256": hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest(),
        "gate_nodes": identity["nodes"],
        "gate_blackboard_slots": identity["blackboard"],
        "gate_node_state_slots": identity["state_slots"],
        "route_pack_changes": 0,
        "routes": identity["routes"],
        "endgame": identity["endgame"],
        "runtime_features": ["STEP", "SHOP_AT"],
    }
    if notes:
        payload["notes"] = notes
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=2) + "\n")
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--status", default="SCREENED")
    parser.add_argument("--notes", default="")
    args = parser.parse_args()
    print(json.dumps(freeze(json.loads(args.spec), args.status, notes=args.notes or None), indent=2))
