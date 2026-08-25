from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.bkg4_search.bounded_macro import BoundedRouteMacro
from tools.bkg4_search.module_codec import ConditionGene

CHAMP = ROOT / "bonsai-kg/integrations/bkg4_champion"
V6 = ROOT / "bonsai-kg/integrations/stateful_v6"
PARENT = CHAMP / "modules/animal_feed_deferral_plus_day2_plus_hire4_plus_fert_wool_plus_w33_s172_v1.json"
DONOR_PACK = ROOT / "scratch/public_targets/top10_parent_routes/routes_top10_parents.bkv6"
ROUTE_OUT = CHAMP / "modules/routes/route435_pizza_glut_rebalance_v1.json"
MACRO_OUT = CHAMP / "modules/macros/route435_pizza_glut_rebalance_v1.json"
MODULE_OUT = CHAMP / "modules/animal_feed_deferral_plus_day2_plus_hire4_plus_fert_wool_plus_w33_s172_plus_route435_pizza_glut_rebalance_v1.json"

ROUTE_ID = 216
DONOR_ROUTE_ID = 435
BASE_ROUTE_ID = 22
START_STEP = 156
DURATION = 48
FRAGMENT_SHA256 = "77ba16e50fc8634b3ea010e45a7cf308723df12cdbad215665a35fbef97f83ba"
ROUTE_OVERRIDE_SLOT = "day2_capital_specialist_v1_route_override"

GATE = (
    ConditionGene("SHOP_AT", "EQ", 1, 5),              # PIZZA_SHOP
    ConditionGene("OPP_UNITS", "GE", 0, 7),
    ConditionGene("OPP_LAND", "GE", 0, 2),
    ConditionGene("MARKET_INVENTORY", "GE", 3, 9990),  # STRAWBERRY near glut
)


def macro() -> BoundedRouteMacro:
    return BoundedRouteMacro(
        name="route435_pizza_glut_rebalance_v1",
        route_id=ROUTE_ID,
        base_route_id=BASE_ROUTE_ID,
        start_step=START_STEP,
        duration=DURATION,
        gate_clauses=(GATE,),
        provenance={
            "source_kind": "source_backed_route_archive",
            "source_route_id": DONOR_ROUTE_ID,
            "fragment_sha256": FRAGMENT_SHA256,
            "discovery": "P0.1_regret_guided_cuda_macro_search",
        },
    )


def _load_v6_runtime():
    spec = importlib.util.spec_from_file_location("stateful_v6_runtime_for_route435_macro", V6 / "runtime.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import stateful-v6 runtime")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _route_pack_identity(pack_path: Path) -> dict:
    m = _load_v6_runtime()
    raw = pack_path.read_bytes()
    magic, version, route_count, _actions, _payload = m.HEADER.unpack_from(raw, 0)
    if magic != m.PACK_MAGIC or version != m.PACK_VERSION:
        raise ValueError(f"invalid route pack: {pack_path}")
    route_ids = [int(m.DESC.unpack_from(raw, m.HEADER.size + i * m.DESC.size)[0]) for i in range(route_count)]
    return {"route_pack": {"pack_sha256": hashlib.sha256(raw).hexdigest(), "route_ids": route_ids}}


def _load_route_actions(pack_path: Path, route_id: int, identity: dict | None = None) -> list[dict]:
    m = _load_v6_runtime()
    pack = m.RoutePack(pack_path, identity or _route_pack_identity(pack_path))
    return [pack.action(route_id, step, int(pack._record(route_id, step)[0])) for step in range(719)]


def _base_actions() -> list[dict]:
    identity = json.loads((V6 / "assets/identity.json").read_text(encoding="utf-8"))
    return _load_route_actions(V6 / "assets/routes.bkv6", BASE_ROUTE_ID, identity)


def _donor_actions() -> list[dict]:
    return _load_route_actions(DONOR_PACK, DONOR_ROUTE_ID)


def build_route() -> dict:
    m = macro()
    donor = _donor_actions()
    raw = json.dumps(donor[START_STEP:START_STEP + DURATION], sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    if hashlib.sha256(raw).hexdigest() != FRAGMENT_SHA256:
        raise ValueError("route435 source fragment no longer matches P0.1 evidence")
    return m.route_document(_base_actions(), donor)


def build_module() -> dict:
    parent = json.loads(PARENT.read_text(encoding="utf-8"))
    out = copy.deepcopy(parent)
    m = macro()
    out["name"] = "animal_feed_deferral_plus_day2_plus_hire4_plus_fert_wool_plus_w33_s172_plus_route435_pizza_glut_rebalance_v1"
    if any(x.get("name") == m.active_slot for x in out.get("blackboard", [])):
        raise ValueError("route435 macro BB already exists")
    out.setdefault("blackboard", []).append(m.blackboard_decl())
    if out["tree"].get("type") != "Sequence":
        raise ValueError("expected accepted parent module sequence")
    out["tree"]["children"].append(m.specialist_tree(route_override_slot=ROUTE_OVERRIDE_SLOT))
    out.setdefault("strategic_routes", []).append({
        "route_id": ROUTE_ID,
        "name": m.name,
        "source": str(ROUTE_OUT.relative_to(ROOT)),
    })
    out["manifest"] = {
        "module": out["name"],
        "parent": "w33_s172_expansion_v1",
        "category": "midgame/bounded_macro_cegis",
        "macro_schema": "bounded_route_macro_v1",
        "macro_source": str(MACRO_OUT.relative_to(ROOT)),
        "macro_id": m.semantic_id,
        "route_id": ROUTE_ID,
        "active_window": [m.start_step, m.end_step],
        "entry_gate": "STEP == 156 and SHOP_AT[1] == PIZZA_SHOP and OPP_UNITS >= 7 and OPP_LAND >= 2 and MARKET_INVENTORY[STRAWBERRY] >= 9990",
        "legal_features": ["STEP", "SHOP_AT[1]", "OPP_UNITS", "OPP_LAND", "MARKET_INVENTORY[STRAWBERRY]"],
        "causal_hypothesis": (
            "When the second visible shop is PIZZA_SHOP, the opponent has committed to both labor and land expansion, and the shared "
            "strawberry market is already near glut, the current parent overcommits strawberries and under-converts its five-worker "
            "production system. The route435 fragment reduces strawberry commitment, rebalances animal/wheat timing, and uses workers "
            "more actively for 48 turns before returning to the parent."
        ),
        "priority": (
            "The macro subtree is appended after existing specialists. During steps 156-203 an authorized macro therefore owns the shared "
            "strategic route override. Existing latched specialists resume automatically after step203 because the macro stops writing the slot."
        ),
        "shared_market_note": "P0.1 component ablations show the capability requires coordinated private worker+market behavior; no gain is credited to shared-price movement alone.",
        "identity_rule": "No opponent/team/submission/replay/episode/seed/tape identity is present in runtime gate or macro representation.",
        "composition": [
            "animal_feed_deferral_v1",
            "day2_capital_specialist_v1",
            "hire4_capital_cycle_v1",
            "fert_wool_expansion_v1",
            "w33_s172_expansion_v1",
            m.name,
        ],
    }
    return out


def main() -> int:
    m = macro()
    route = build_route()
    module = build_module()
    ROUTE_OUT.parent.mkdir(parents=True, exist_ok=True)
    MACRO_OUT.parent.mkdir(parents=True, exist_ok=True)
    ROUTE_OUT.write_text(json.dumps(route, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    MACRO_OUT.write_text(json.dumps(m.canonical(), ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    MODULE_OUT.write_text(json.dumps(module, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"macro": str(MACRO_OUT.relative_to(ROOT)), "module": str(MODULE_OUT.relative_to(ROOT)), "route": str(ROUTE_OUT.relative_to(ROOT)), "macro_id": m.semantic_id, "route_id": ROUTE_ID, "window": [m.start_step, m.end_step]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
