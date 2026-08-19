from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Dict

HERE = Path(__file__).resolve().parent
BKG_ROOT = HERE.parents[1]
V6 = BKG_ROOT / "integrations" / "stateful_v6"
for _path in (HERE, BKG_ROOT / "python", V6):
    _s = str(_path)
    if _s in sys.path:
        sys.path.remove(_s)
    sys.path.insert(0, _s)

import bonsai_kg_v4 as bkg4
from engine_exploits import EngineExploitState, apply_engine_exploits
from legacy_overlay import LegacyOverlayState, apply_legacy_overlay
from mechanics_reconciler import apply_feed_deferral

_V6_RUNTIME_SPEC = importlib.util.spec_from_file_location("bkg4_champion_v6_runtime", V6 / "runtime.py")
if _V6_RUNTIME_SPEC is None or _V6_RUNTIME_SPEC.loader is None:
    raise RuntimeError("cannot load frozen stateful-v6 runtime")
_V6_RUNTIME = importlib.util.module_from_spec(_V6_RUNTIME_SPEC)
_V6_RUNTIME_SPEC.loader.exec_module(_V6_RUNTIME)
RoutePack = _V6_RUNTIME.RoutePack


class CompatibilityError(RuntimeError):
    pass


class LegacyRoutePack(RoutePack):
    @staticmethod
    def _order_obj(op: int, item: int, n: int):
        if op == 0:
            return ["PASS"]
        return RoutePack._order_obj(op, item, n)


class BKG4ChampionAgent:
    """BKG4 controller with frozen compatibility overlays.

    Search-added strategy belongs in the BKG4 controller. The H4/public-prefix and
    accepted engine-mechanics layers remain exact external compatibility behavior in
    BASE_0 so architecture migration cannot cause a strategic regression.
    """

    def __init__(self, asset_dir: Path | str | None = None):
        self.asset_dir = Path(asset_dir) if asset_dir is not None else HERE / "assets"
        self.identity = json.loads((self.asset_dir / "identity.json").read_text(encoding="utf-8"))
        self.controller = bkg4.load_compiled(self.asset_dir / "controller.bkgc")
        if self.controller.semantic_hash != self.identity["controller_semantic_hash"]:
            raise CompatibilityError("BKG4 champion controller semantic hash mismatch")

        v6_asset = V6 / "assets"
        v6_identity = json.loads((v6_asset / "identity.json").read_text(encoding="utf-8"))
        route_meta = dict(self.identity["v6_route_pack"])
        expected_v6 = str(route_meta["pack_sha256"])
        if route_meta.get("path") == "routes.bkv6":
            route_path = self.asset_dir / "routes.bkv6"
            self.base_pack = RoutePack(route_path, {"route_pack": route_meta})
        else:
            if v6_identity["route_pack"]["pack_sha256"] != expected_v6:
                raise CompatibilityError("BKG4 champion frozen v6 route-pack identity mismatch")
            self.base_pack = RoutePack(v6_asset / "routes.bkv6", v6_identity)

        legacy_identity = {"route_pack": self.identity["legacy_route_pack"]}
        self.legacy_pack = LegacyRoutePack(self.asset_dir / "legacy_routes.bkv6", legacy_identity)
        if hashlib.sha256((self.asset_dir / "legacy_routes.bkv6").read_bytes()).hexdigest() != self.identity["legacy_route_pack"]["pack_sha256"]:
            raise CompatibilityError("BKG4 champion legacy route-pack hash mismatch")

        self.vm = bkg4.VM(self.controller)
        self.legacy = LegacyOverlayState()
        self.exploits = EngineExploitState()
        self.last_step = -1
        runtime_reconciler = self.identity.get("runtime_reconciler")
        self.route_override_slot = None
        self.route_override_id = None
        self.route_override_value = False
        if runtime_reconciler is None:
            self.feed_defer_slots = ()
        elif runtime_reconciler.get("kind") in ("feed_deferral_v1", "feed_deferral_route_override_v1", "feed_deferral_route_value_v1"):
            self.feed_defer_slots = tuple(int(x) for x in runtime_reconciler.get("defer_slots") or [])
            if runtime_reconciler.get("kind") == "feed_deferral_route_override_v1":
                self.route_override_slot = int(runtime_reconciler["route_override_slot"])
                self.route_override_id = int(runtime_reconciler["route_override_id"])
            elif runtime_reconciler.get("kind") == "feed_deferral_route_value_v1":
                self.route_override_slot = int(runtime_reconciler["route_override_slot"])
                self.route_override_value = True
        else:
            raise CompatibilityError(f"unsupported BKG4 runtime reconciler {runtime_reconciler.get('kind')!r}")

    @staticmethod
    def _hands(observation: Dict[str, Any]) -> int:
        p = int(observation.get("player", 0) or 0)
        farms = list(observation.get("farms") or [])
        own = farms[p] if 0 <= p < len(farms) and isinstance(farms[p], dict) else {}
        return len(own.get("hands") or [])

    def reset(self) -> None:
        self.vm.reset()
        self.legacy.reset()
        self.exploits.reset()
        self.last_step = -1

    def state(self) -> Dict[str, Any]:
        bb = self.vm.state.blackboard
        controller_route = int(bb[1]) if len(bb) > 1 else None
        route_override_raw = int(bb[self.route_override_slot]) if self.route_override_slot is not None and 0 <= self.route_override_slot < len(bb) else 0
        route_override_active = bool(route_override_raw)
        effective_route = (route_override_raw if self.route_override_value else self.route_override_id) if route_override_active else controller_route
        return {
            "route": effective_route,
            "controller_route": controller_route,
            "route_override_active": route_override_active,
            "h4_trigger": self.legacy.h4_trigger,
            "external_ok": self.legacy.external_ok,
            "external_route": self.legacy.external_route,
            "engine_high_labor": self.exploits.high_labor,
            "carrot_latched": self.exploits.carrot_latched,
            "feed_defer_units": [i for i, slot in enumerate(self.feed_defer_slots) if 0 <= slot < len(bb) and int(bb[slot])],
            "fault": self.vm.state.fault,
        }

    def __call__(self, observation: Dict[str, Any], configuration: Any = None) -> Dict[str, Any]:
        step = int(observation.get("step", 0) or 0)
        if step == 0 or step <= self.last_step:
            self.reset()
        self.last_step = step

        result = self.vm.tick(observation, max_market_orders=0, trace=False)
        if result.fault:
            raise CompatibilityError(f"BKG4 controller fault {result.fault}")
        if result.status != bkg4.STATUS_SUCCESS:
            raise CompatibilityError(f"BKG4 controller returned {bkg4.STATUS_NAMES[result.status]}")
        route = int(self.vm.state.blackboard[1])
        if self.route_override_slot is not None:
            override = int(self.vm.state.blackboard[self.route_override_slot])
            if override:
                route = override if self.route_override_value else int(self.route_override_id)
        hands = self._hands(observation)
        action = self.base_pack.action(route, step, hands)
        action = apply_legacy_overlay(observation, action, self.legacy_pack, self.legacy, hands)
        action = apply_engine_exploits(
            observation,
            configuration,
            action,
            self.exploits,
            external_ok=self.legacy.external_ok,
        )
        if self.feed_defer_slots:
            action = apply_feed_deferral(action, self.vm.state.blackboard, self.feed_defer_slots)
        return action


def load_default_agent() -> BKG4ChampionAgent:
    return BKG4ChampionAgent()
