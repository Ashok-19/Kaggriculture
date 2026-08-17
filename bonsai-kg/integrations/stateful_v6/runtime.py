from __future__ import annotations

import hashlib
import json
import struct
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
BKG_ROOT = HERE.parents[1]
sys.path.insert(0, str(BKG_ROOT / "python"))
import bonsai_kg as bkg
from build_assets import ACTION_SIZE, DESC, HEADER, LEAF_ID, LEAF_NONE, MAX_MARKET, MAX_SAVED_HANDS, ORDER, PACK_MAGIC, PACK_VERSION, UNIT

UNIT_NAMES = {v: k for k, v in bkg.UNIT_OPS.items()}
MARKET_NAMES = {v: k for k, v in bkg.MARKET_OPS.items()}
LEAF_NAMES = {v: k for k, v in LEAF_ID.items()}


class CompatibilityError(RuntimeError):
    pass


class RoutePack:
    def __init__(self, pack_path: Path, identity: Dict[str, Any]):
        self.pack_path = Path(pack_path)
        raw = self.pack_path.read_bytes()
        expected = str(identity["route_pack"]["pack_sha256"])
        actual = hashlib.sha256(raw).hexdigest()
        if actual != expected:
            raise CompatibilityError(f"route-pack SHA-256 mismatch: expected {expected}, got {actual}")
        if len(raw) < HEADER.size:
            raise CompatibilityError("route pack too short")
        magic, version, route_count, action_count, payload_hash = HEADER.unpack_from(raw, 0)
        if magic != PACK_MAGIC or version != PACK_VERSION:
            raise CompatibilityError("invalid route-pack header")
        payload = raw[HEADER.size:]
        if hashlib.sha256(payload).digest() != payload_hash:
            raise CompatibilityError("route-pack payload hash mismatch")
        if route_count != len(identity["route_pack"]["route_ids"]):
            raise CompatibilityError("route-pack route count mismatch")
        desc_bytes = route_count * DESC.size
        expected_size = HEADER.size + desc_bytes + action_count * ACTION_SIZE
        if len(raw) != expected_size:
            raise CompatibilityError("route-pack size/count mismatch")
        self._raw = raw
        self._records_offset = HEADER.size + desc_bytes
        self._routes: Dict[int, Tuple[int, int]] = {}
        for i in range(route_count):
            rid, steps, first = DESC.unpack_from(raw, HEADER.size + i * DESC.size)
            if rid in self._routes or first + steps > action_count:
                raise CompatibilityError("invalid route descriptor")
            self._routes[rid] = (first, steps)
        if set(self._routes) != set(identity["route_pack"]["route_ids"]):
            raise CompatibilityError("route-pack route ids mismatch")
        self.action_count = action_count

    def _record(self, route: int, step: int) -> memoryview:
        if route not in self._routes:
            raise CompatibilityError(f"unknown compatibility route {route}")
        first, steps = self._routes[route]
        if steps <= 0:
            raise CompatibilityError(f"empty compatibility route {route}")
        s = min(max(int(step), 0), steps - 1)
        off = self._records_offset + (first + s) * ACTION_SIZE
        return memoryview(self._raw)[off:off + ACTION_SIZE]

    @staticmethod
    def _unit_obj(op: int, arg: int, n: int) -> List[Any]:
        name = UNIT_NAMES.get(op)
        if name is None:
            raise CompatibilityError(f"invalid packed unit op {op}")
        if name in ("PICKUP", "PLACE"):
            return [name, bkg.ITEMS[arg], int(n)]
        if name == "PLANT":
            return [name, bkg.ITEMS[arg]]
        return [name]

    @staticmethod
    def _order_obj(op: int, item: int, n: int) -> List[Any]:
        name = MARKET_NAMES.get(op)
        if name is None:
            raise CompatibilityError(f"invalid packed market op {op}")
        if name in ("HIRE", "BUY_LAND"):
            return [name]
        return [name, bkg.ITEMS[item], int(n)]

    def action(self, route: int, step: int, current_hands: int) -> Dict[str, Any]:
        r = self._record(route, step)
        saved_hands, n_market = struct.unpack_from("<BB", r, 0)
        if saved_hands > MAX_SAVED_HANDS or n_market > MAX_MARKET:
            raise CompatibilityError("packed action count exceeds format limit")
        off = 2
        farmer = self._unit_obj(*UNIT.unpack_from(r, off)); off += UNIT.size
        saved: List[List[Any]] = []
        for i in range(MAX_SAVED_HANDS):
            u = self._unit_obj(*UNIT.unpack_from(r, off)); off += UNIT.size
            if i < saved_hands:
                saved.append(u)
        market: List[List[Any]] = []
        for i in range(MAX_MARKET):
            o = ORDER.unpack_from(r, off); off += ORDER.size
            if i < n_market:
                market.append(self._order_obj(*o))
        hands = [list(saved[i]) if i < len(saved) else ["PASS"] for i in range(max(0, int(current_hands)))]
        return {"farmer": list(farmer), "hands": hands, "market": market}


class StatefulV6BKG3Agent:
    def __init__(self, asset_dir: Path | str | None = None):
        self.asset_dir = Path(asset_dir) if asset_dir is not None else HERE / "assets"
        self.identity = json.loads((self.asset_dir / "identity.json").read_text(encoding="utf-8"))
        self.controller = bkg.load_compiled(self.asset_dir / "controller.bkgc")
        expected_controller = str(self.identity["controller_semantic_hash"])
        if self.controller.semantic_hash != expected_controller:
            raise CompatibilityError(
                f"controller semantic hash mismatch: expected {expected_controller}, got {self.controller.semantic_hash}"
            )
        self.pack = RoutePack(self.asset_dir / "routes.bkv6", self.identity)
        self.vm = bkg.VM(self.controller)
        self.last_step = -1

    def reset(self) -> None:
        self.vm.reset()
        self.last_step = -1

    @staticmethod
    def _hands(observation: Dict[str, Any]) -> int:
        p = int(observation.get("player", 0) or 0)
        farms = list(observation.get("farms", []) or [])
        own = farms[p] if 0 <= p < len(farms) and isinstance(farms[p], dict) else {}
        return len(own.get("hands", []) or [])

    def state(self) -> Dict[str, Any]:
        bb = self.vm.state.blackboard
        stage1 = bool(bb[2])
        leaf_id = int(bb[0])
        return {
            "leaf_id": leaf_id,
            "leaf": LEAF_NAMES.get(leaf_id) if stage1 else None,
            "route": int(bb[1]) if stage1 else None,
            "effective_route": int(bb[1]),
            "stage144_done": stage1,
            "stage216_done": bool(bb[3]),
            "fault": int(self.vm.state.fault),
        }

    def __call__(self, observation: Dict[str, Any], configuration: Any = None) -> Dict[str, Any]:
        del configuration
        step = int(observation.get("step", 0) or 0)
        if step == 0 or step <= self.last_step:
            self.vm.reset()
        self.last_step = step
        result = self.vm.tick(observation, max_market_orders=0, trace=False)
        if result.fault:
            raise CompatibilityError(f"BKG3 controller fault {result.fault}")
        if result.status != bkg.STATUS_SUCCESS:
            raise CompatibilityError(f"BKG3 controller returned unexpected status {bkg.STATUS_NAMES[result.status]}")
        route = int(self.vm.state.blackboard[1])
        return self.pack.action(route, step, self._hands(observation))


def load_default_agent() -> StatefulV6BKG3Agent:
    return StatefulV6BKG3Agent()
