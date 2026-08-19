from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MAX_PROVEN_UNITS = 13


def cond(feature: str, op: str, rhs, arg=None):
    lhs = {"feature": feature}
    if arg is not None:
        lhs["arg"] = arg
    return {"type": "Condition", "lhs": lhs, "op": op, "rhs": rhs}


def build_module() -> dict:
    bb = [{"name": "feed_probe_tile", "type": "TILE", "initial": -1, "owner": "animal_feed_deferral"}]
    bb += [
        {"name": f"feed_defer_{u}", "type": "BOOL", "initial": 0, "owner": "animal_feed_deferral"}
        for u in range(MAX_PROVEN_UNITS)
    ]
    children = [
        {"type": "SetBlackboard", "slot": f"feed_defer_{u}", "value": 0}
        for u in range(MAX_PROVEN_UNITS)
    ]
    for u in range(MAX_PROVEN_UNITS):
        tile = {"blackboard": "feed_probe_tile"}
        authorize = {
            "type": "Sequence",
            "children": [
                {"type": "SetBlackboard", "slot": "feed_probe_tile", "value": {"feature": "OWN_WORKER_TILE", "arg": u}},
                cond("OWN_TILE_HAS_ANIMAL", "EQ", 1, tile),
                cond("OWN_TILE_ITEM", "EQ", "COW", tile),
                cond("OWN_TILE_CONSECUTIVE_UNFED", "EQ", 0, tile),
                cond("OWN_TILE_PENDING_CARE_BONUS", "EQ", 0, tile),
                {"type": "SetBlackboard", "slot": f"feed_defer_{u}", "value": 1},
            ],
        }
        children.append({
            "type": "If",
            "condition": cond("OWN_UNITS", "GT", u),
            "then": {"type": "ForceSuccess", "child": authorize},
            "else": {"type": "Success"},
        })
    return {
        "schema_version": 4,
        "name": "animal_feed_deferral_v1",
        "blackboard": bb,
        "runtime_reconciler": {
            "kind": "feed_deferral_v1",
            "defer_slots": [f"feed_defer_{u}" for u in range(MAX_PROVEN_UNITS)],
            "max_units": MAX_PROVEN_UNITS,
            "materialization_rule": "replace route-pack FEED with PASS iff corresponding BKG4 authorization slot is true",
        },
        "manifest": {
            "module": "animal_feed_deferral_v1",
            "category": "engine_mechanics/animal_lifecycle",
            "causal_hypothesis": "An animal survives one unfed day and still receives scheduled base production; when a cow was fed on the previous day and has no pending care bonus, deferring a planned FEED saves wheat without sacrificing survival or care-bonus production.",
            "discovery": "authoritative _daily_refresh_animals source audit + full-controller CUDA semantic grid",
            "legal_features": ["OWN_UNITS", "OWN_WORKER_TILE", "OWN_TILE_HAS_ANIMAL", "OWN_TILE_ITEM", "OWN_TILE_CONSECUTIVE_UNFED", "OWN_TILE_PENDING_CARE_BONUS"],
            "search_genome_id": "7002da5a004ff7b2e9817176e7166a91487a53e0c52a0b9dbda04e14c95e7961",
            "minimized_from": "GOOSE|COW -> COW only; replacement variants collapsed to PASS; wheat caps rejected; pending-care guard retained",
            "search_campaign": "bkg4_engine_mechanics_feed_v1_20260819",
            "bkg4_vm_frozen_commit": "8381e027646ba6494cbe4dd06a810ea55ec83575",
        },
        "tree": {
            "type": "Sequence",
            "name": "AnimalFeedDeferralAuthorization",
            "module": "animal_feed_deferral_v1",
            "category": "engine_mechanics/animal_lifecycle",
            "provenance": {
                "source": "kaggriculture.py::_daily_refresh_animals",
                "candidate_id": "7002da5a004ff7b2e9817176e7166a91487a53e0c52a0b9dbda04e14c95e7961",
            },
            "children": children,
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("bonsai-kg/integrations/bkg4_champion/modules/animal_feed_deferral_v1.json"))
    args = ap.parse_args()
    out = (ROOT / args.out).resolve() if not args.out.is_absolute() else args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    module = build_module()
    out.write_text(json.dumps(module, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(out), "bb_slots": len(module["blackboard"]), "proven_units": MAX_PROVEN_UNITS}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
