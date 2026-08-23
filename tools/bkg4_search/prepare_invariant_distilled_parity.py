from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

from tools.bkg4_search.invariant_module import (
    AbstractAction,
    GuardAtom,
    InvariantRule,
    InvariantStrategicModule,
    LiftedGuard,
    ModuleSketch,
    compose_distilled_module,
    distill_invariant_module,
    pack_file,
)


ROOT = Path(__file__).resolve().parents[2]
CHAMPION = ROOT / "bonsai-kg" / "integrations" / "bkg4_champion"
PARENT_MODULE = CHAMPION / "modules" / "animal_feed_deferral_plus_day2_plus_hire4_plus_fert_wool_plus_w33_s172_v1.json"
ROUTE_SLOT = "day2_capital_specialist_v1_route_override"


def _build_assets_module():
    spec = importlib.util.spec_from_file_location("bkg4_parity_build_assets", CHAMPION / "build_assets.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load BKG4 build_assets")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def modules() -> dict[str, InvariantStrategicModule]:
    return {
        "route_override": InvariantStrategicModule((InvariantRule(
            ModuleSketch("route", "INSTANT", "ROUTE_MODE"), "GLOBAL", LiftedGuard(), AbstractAction("ROUTE_OVERRIDE", route_id=70),
        ),)),
        "committed_route": InvariantStrategicModule((InvariantRule(
            ModuleSketch("route", "COMMITTED", "ROUTE_MODE"), "GLOBAL", LiftedGuard((GuardAtom("OWN_MONEY", "GE", 0),)),
            AbstractAction("COMMITTED_ROUTE", route_id=70), LiftedGuard((GuardAtom("OWN_MONEY", "LT", 0),)),
        ),)),
        "unit_rewrite": InvariantStrategicModule((InvariantRule(
            ModuleSketch("maintenance", "INSTANT", "REDUCE_MAINTENANCE_BACKLOG"), "EACH_OWN_UNIT", LiftedGuard(),
            AbstractAction("UNIT_REWRITE", match_op=17, unit_op=0),
        ),)),
        "market_intent": InvariantStrategicModule((InvariantRule(
            ModuleSketch("market", "INSTANT", "CREATE_FINANCING_RUNWAY"), "GLOBAL", LiftedGuard(),
            AbstractAction("MARKET_INTENT", market_op=1, quantity=1, authority=4, priority=250),
        ),)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve(); out.mkdir(parents=True, exist_ok=True)
    parent = json.loads(PARENT_MODULE.read_text(encoding="utf-8"))
    build_assets = _build_assets_module()
    manifest = {"modules": {}}
    ordered = modules()
    (out / "candidates.b4im").write_bytes(pack_file(tuple(ordered.values())))
    for index, (name, module) in enumerate(ordered.items()):
        distilled = distill_invariant_module(module, max_units=13, route_slot=ROUTE_SLOT)
        composed = compose_distilled_module(parent, distilled)
        module_path = out / f"{name}.json"
        module_path.write_text(json.dumps(composed, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        assets = out / name
        identity = build_assets.build(assets, module_path)
        rewrites = []
        profile = identity.get("runtime_reconciler") or {}
        profiles = profile.get("reconcilers") or [profile]
        for reconciler in profiles:
            if reconciler.get("kind") == "invariant_unit_rewrite_v1":
                for row in reconciler["rewrites"]:
                    rewrites.append([row["slot"], row["unit"], 17, 0, row["arg"], row["quantity"]])
        manifest["modules"][name] = {
            "candidate_index": index,
            "semantic_id": module.semantic_id(),
            "assets": str(assets),
            "rewrites": rewrites,
        }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
