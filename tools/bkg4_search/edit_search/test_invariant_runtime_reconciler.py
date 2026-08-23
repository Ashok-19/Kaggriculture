from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CHAMPION = ROOT / "bonsai-kg" / "integrations" / "bkg4_champion"
if str(CHAMPION) not in sys.path:
    sys.path.insert(0, str(CHAMPION))

from mechanics_reconciler import apply_invariant_unit_rewrites  # noqa: E402
from tools.bkg4_search.invariant_module import (  # noqa: E402
    AbstractAction,
    GuardAtom,
    InvariantRule,
    InvariantStrategicModule,
    LiftedGuard,
    ModuleSketch,
    compose_distilled_module,
    distill_invariant_module,
)


def _load_build_assets():
    spec = importlib.util.spec_from_file_location("bkg4_champion_build_assets_test", CHAMPION / "build_assets.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load build_assets")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_runtime():
    spec = importlib.util.spec_from_file_location("bkg4_champion_runtime_test", CHAMPION / "runtime.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load champion runtime")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _observation(step: int, *, money: int = 3000, hands: int = 0) -> dict:
    tiles = [[None for _ in range(10)] for _ in range(10)]
    prices = {name: 100 for name in (
        "WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL", "FERTILIZER",
    )}
    return {
        "step": step, "day": step // 24, "hour": step % 24, "player": 0,
        "farms": [
            {"money": money, "tiles": tiles, "farmer": [0, 0], "hands": [[0, 0] for _ in range(hands)], "unlocked_quadrants": ["NW"]},
            {"money": 3000, "tiles": tiles, "farmer": [0, 0], "hands": [], "unlocked_quadrants": ["NW"]},
        ],
        "town": {"unlocked_shops": []},
        "market": {"prices": prices, "inventory": {name: 100 for name in prices}},
        "private": {"shed": {}, "seeds": {}, "inventories": [{} for _ in range(hands + 1)]},
    }


class InvariantRuntimeReconcilerTests(unittest.TestCase):
    def test_generic_rewrite_changes_only_authorized_matching_unit(self):
        action = {"farmer": ["CARE"], "hands": [["FEED", "WHEAT"], ["CARE"]], "market": []}
        rewrites = [
            {"slot": 2, "unit": 0, "match_op": "CARE", "replacement_op": "PASS", "arg": 0, "quantity": 1},
            {"slot": 3, "unit": 1, "match_op": "CARE", "replacement_op": "PASS", "arg": 0, "quantity": 1},
            {"slot": 4, "unit": 2, "match_op": "CARE", "replacement_op": "PASS", "arg": 0, "quantity": 1},
        ]
        out = apply_invariant_unit_rewrites(action, [0, 0, 1, 1, 1], rewrites)
        self.assertEqual(out["farmer"], ["PASS"])
        self.assertEqual(out["hands"][0], ["FEED", "WHEAT"])
        self.assertEqual(out["hands"][1], ["PASS"])
        self.assertEqual(action["farmer"], ["CARE"])

    def test_generic_rewrite_emits_official_unit_action_shapes(self):
        action = {"farmer": ["CARE"], "hands": [["CARE"], ["CARE"], ["CARE"]], "market": []}
        rewrites = [
            {"slot": 0, "unit": 0, "match_op": "CARE", "replacement_op": "FEED", "arg": 0, "quantity": 1},
            {"slot": 1, "unit": 1, "match_op": "CARE", "replacement_op": "PLANT", "arg": 0, "quantity": 1},
            {"slot": 2, "unit": 2, "match_op": "CARE", "replacement_op": "PICKUP", "arg": 6, "quantity": 2},
            {"slot": 3, "unit": 3, "match_op": "CARE", "replacement_op": "DROP", "arg": 7, "quantity": 9},
        ]
        out = apply_invariant_unit_rewrites(action, [1, 1, 1, 1], rewrites)
        self.assertEqual(out["farmer"], ["FEED"])
        self.assertEqual(out["hands"], [
            ["PLANT", "WHEAT"], ["PICKUP", "MILK", 2], ["DROP"],
        ])

    def test_build_resolves_generic_rewrite_slot_names_to_indices(self):
        module = InvariantStrategicModule((InvariantRule(
            ModuleSketch("maintenance", "INSTANT", "REDUCE_MAINTENANCE_BACKLOG"),
            "EACH_OWN_UNIT", LiftedGuard(),
            AbstractAction("UNIT_REWRITE", match_op=17, unit_op=0),
        ),))
        distilled = distill_invariant_module(module, max_units=2)
        build_assets = _load_build_assets()
        resolved = build_assets.resolve_runtime_reconciler(distilled)
        self.assertEqual(resolved["kind"], "invariant_unit_rewrite_v1")
        self.assertEqual([x["slot"] for x in resolved["rewrites"]], [4, 5])
        self.assertEqual(resolved["rewrite_slot_names"], ["invariant_rewrite_r0_u0", "invariant_rewrite_r0_u1"])

    def test_composition_preserves_parent_feed_and_route_reconciliation(self):
        parent_path = CHAMPION / "modules" / "animal_feed_deferral_plus_day2_plus_hire4_plus_fert_wool_plus_w33_s172_v1.json"
        parent = json.loads(parent_path.read_text(encoding="utf-8"))
        module = InvariantStrategicModule((InvariantRule(
            ModuleSketch("maintenance", "INSTANT", "REDUCE_MAINTENANCE_BACKLOG"),
            "EACH_OWN_UNIT", LiftedGuard(),
            AbstractAction("UNIT_REWRITE", match_op=17, unit_op=0),
        ),))
        invariant = distill_invariant_module(
            module, max_units=2, route_slot="day2_capital_specialist_v1_route_override",
        )
        composed = compose_distilled_module(parent, invariant)
        self.assertEqual(composed["runtime_reconciler"]["kind"], "composite_v1")
        build_assets = _load_build_assets()
        resolved = build_assets.resolve_runtime_reconciler(composed)
        self.assertEqual(resolved["kind"], "composite_v1")
        self.assertEqual([x["kind"] for x in resolved["reconcilers"]], [
            "feed_deferral_route_value_v1", "invariant_unit_rewrite_v1",
        ])
        self.assertEqual(resolved["reconcilers"][0]["defer_slots"], list(range(5, 18)))
        self.assertEqual(resolved["reconcilers"][0]["route_override_slot"], 19)
        self.assertEqual(resolved["reconcilers"][1]["rewrite_slots"], [23, 24])

    def test_composed_assets_execute_all_distilled_action_families(self):
        parent_path = CHAMPION / "modules" / "animal_feed_deferral_plus_day2_plus_hire4_plus_fert_wool_plus_w33_s172_v1.json"
        parent = json.loads(parent_path.read_text(encoding="utf-8"))
        build_assets = _load_build_assets()
        champion_runtime = _load_runtime()

        modules = {
            "route": InvariantStrategicModule((InvariantRule(
                ModuleSketch("route", "INSTANT", "ROUTE_MODE"), "GLOBAL", LiftedGuard(),
                AbstractAction("ROUTE_OVERRIDE", route_id=70),
            ),)),
            "committed": InvariantStrategicModule((InvariantRule(
                ModuleSketch("route", "COMMITTED", "ROUTE_MODE"), "GLOBAL",
                LiftedGuard((GuardAtom("OWN_MONEY", "GE", 0),)),
                AbstractAction("COMMITTED_ROUTE", route_id=70),
                LiftedGuard((GuardAtom("OWN_MONEY", "LT", 0),)),
            ),)),
            "rewrite": InvariantStrategicModule((
                InvariantRule(ModuleSketch("route", "INSTANT", "ROUTE_MODE"), "GLOBAL", LiftedGuard(), AbstractAction("ROUTE_OVERRIDE", route_id=29)),
                InvariantRule(ModuleSketch("maintenance", "INSTANT", "REDUCE_MAINTENANCE_BACKLOG"), "EACH_OWN_UNIT", LiftedGuard(), AbstractAction("UNIT_REWRITE", match_op=17, unit_op=0)),
            )),
            "market": InvariantStrategicModule((InvariantRule(
                ModuleSketch("market", "INSTANT", "CREATE_FINANCING_RUNWAY"), "GLOBAL", LiftedGuard(),
                AbstractAction("MARKET_INTENT", market_op=1, quantity=1, authority=4, priority=250),
            ),)),
        }
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            actions = {}
            for name, module in modules.items():
                distilled = distill_invariant_module(
                    module, max_units=1, route_slot="day2_capital_specialist_v1_route_override",
                )
                composed = compose_distilled_module(parent, distilled)
                module_path = root / f"{name}.json"
                module_path.write_text(json.dumps(composed), encoding="utf-8")
                assets = root / name
                build_assets.build(assets, module_path)
                agent = champion_runtime.BKG4ChampionAgent(assets)
                actions[name] = agent(_observation(12 if name == "rewrite" else 0))

        self.assertEqual(actions["route"]["farmer"], ["BUILD_PASTURE"])
        self.assertEqual(actions["committed"], actions["route"])
        self.assertEqual(actions["rewrite"]["farmer"], ["PASS"])
        self.assertEqual(actions["market"]["market"][0], ["HIRE"])


if __name__ == "__main__":
    unittest.main()
