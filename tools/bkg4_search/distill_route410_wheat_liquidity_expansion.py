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

from tools.bkg4_search.bounded_macro import BoundedRouteMacro, validate_nonoverlapping_macro_windows
from tools.bkg4_search.distill_route435_macro_cegis import macro as route435_macro
from tools.bkg4_search.module_codec import ConditionGene

CHAMP = ROOT / "bonsai-kg/integrations/bkg4_champion"
V6 = ROOT / "bonsai-kg/integrations/stateful_v6"
PARENT = CHAMP / "modules/animal_feed_deferral_plus_day2_plus_hire4_plus_fert_wool_plus_w33_s172_plus_route435_pizza_glut_rebalance_v1.json"
DONOR_PACK = ROOT / "scratch/public_targets/top10_parent_routes/routes_top10_parents.bkv6"
ROUTE_OUT = CHAMP / "modules/routes/route410_wheat_liquidity_expansion_v1.json"
MACRO_OUT = CHAMP / "modules/macros/route410_wheat_liquidity_expansion_v1.json"
MODULE_OUT = CHAMP / "modules/animal_feed_deferral_plus_day2_plus_hire4_plus_fert_wool_plus_w33_s172_plus_route435_plus_route410_wheat_liquidity_expansion_v1.json"

ROUTE_ID = 217
DONOR_ROUTE_ID = 410
BASE_ROUTE_ID = 64
START_STEP = 120
DURATION = 18
FRAGMENT_SHA256 = "954edb5db3adf1833c906dc9631b53c038a1e67c3fe818bb721f035b58ade0db"
ROUTE_OVERRIDE_SLOT = "day2_capital_specialist_v1_route_override"

GATE = (
    ConditionGene("MARKET_INVENTORY", "LE", 0, 9937),  # WHEAT drawdown
    ConditionGene("OPP_MONEY", "LE", 0, 120),
    ConditionGene("MARKET_INVENTORY", "GE", 6, 9990),  # MILK remains well supplied
)


def macro() -> BoundedRouteMacro:
    out = BoundedRouteMacro(
        name="route410_wheat_liquidity_expansion_v1",
        route_id=ROUTE_ID,
        base_route_id=BASE_ROUTE_ID,
        start_step=START_STEP,
        duration=DURATION,
        gate_clauses=(GATE,),
        provenance={
            "source_kind": "source_backed_route_archive",
            "source_route_id": DONOR_ROUTE_ID,
            "fragment_sha256": FRAGMENT_SHA256,
            "discovery": "P0.3_regret_guided_cuda_repeatability_search",
        },
    )
    validate_nonoverlapping_macro_windows((route435_macro(), out))
    return out


def _load_v6_runtime():
    spec = importlib.util.spec_from_file_location("stateful_v6_runtime_for_route410_macro", V6 / "runtime.py")
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
    raw = json.dumps(
        donor[START_STEP:START_STEP + DURATION],
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    if hashlib.sha256(raw).hexdigest() != FRAGMENT_SHA256:
        raise ValueError("route410 source fragment no longer matches P0.3 evidence")
    return m.route_document(_base_actions(), donor)


def build_module() -> dict:
    parent = json.loads(PARENT.read_text(encoding="utf-8"))
    out = copy.deepcopy(parent)
    m = macro()
    out["name"] = "animal_feed_deferral_plus_day2_plus_hire4_plus_fert_wool_plus_w33_s172_plus_route435_plus_route410_wheat_liquidity_expansion_v1"
    if any(x.get("name") == m.active_slot for x in out.get("blackboard", [])):
        raise ValueError("route410 macro BB already exists")
    out.setdefault("blackboard", []).append(m.blackboard_decl())
    if out["tree"].get("type") != "Sequence":
        raise ValueError("expected P0.2 parent module sequence")
    out["tree"]["children"].append(m.specialist_tree(route_override_slot=ROUTE_OVERRIDE_SLOT))
    out.setdefault("strategic_routes", []).append({
        "route_id": ROUTE_ID,
        "name": m.name,
        "source": str(ROUTE_OUT.relative_to(ROOT)),
    })
    out["manifest"] = {
        "module": out["name"],
        "parent": "route435_pizza_glut_rebalance_v1",
        "category": "day5/bounded_macro_cegis",
        "macro_schema": "bounded_route_macro_v1",
        "macro_source": str(MACRO_OUT.relative_to(ROOT)),
        "macro_id": m.semantic_id,
        "route_id": ROUTE_ID,
        "active_window": [m.start_step, m.end_step],
        "entry_gate": "STEP == 120 and MARKET_INVENTORY[WHEAT] <= 9937 and OPP_MONEY <= 120 and MARKET_INVENTORY[MILK] >= 9990",
        "legal_features": ["STEP", "MARKET_INVENTORY[WHEAT]", "OPP_MONEY", "MARKET_INVENTORY[MILK]"],
        "causal_hypothesis": (
            "At the day-5 boundary, material WHEAT drawdown indicates staple demand while low public opponent liquidity indicates a committed economy. "
            "When MILK remains well supplied, the incumbent route64 cow purchase is less attractive. The route410 fragment instead converts "
            "available wheat/fertilizer into five hires, delays the animal mix toward SHEEP, and uses the expanded labor force for care, watering "
            "and strawberry planting before returning to the P0.2 parent."
        ),
        "priority": (
            "This synthesized macro ends at step137. The existing route435 synthesized macro begins at step156, so their strategic route-override "
            "windows are disjoint. P0.3 fails closed on synthesized-macro window overlap rather than relying on append order."
        ),
        "shared_market_note": (
            "The gate reads shared market state, but the payoff comes from private capital allocation and worker/animal execution. The macro sells "
            "held wheat/fertilizer to finance hires and changes the private cow/sheep/labor plan; no gain is credited to shared-price movement alone."
        ),
        "identity_rule": "No opponent/team/submission/replay/episode/seed/tape identity is present in runtime gate or macro representation.",
        "composition": [
            "animal_feed_deferral_v1",
            "day2_capital_specialist_v1",
            "hire4_capital_cycle_v1",
            "fert_wool_expansion_v1",
            "w33_s172_expansion_v1",
            "route435_pizza_glut_rebalance_v1",
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
    print(json.dumps({
        "macro": str(MACRO_OUT.relative_to(ROOT)),
        "module": str(MODULE_OUT.relative_to(ROOT)),
        "route": str(ROUTE_OUT.relative_to(ROOT)),
        "macro_id": m.semantic_id,
        "route_id": ROUTE_ID,
        "window": [m.start_step, m.end_step],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
