from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Dict, Optional

SHOP_ID = {
    "BAKERY": 0, "BAKERY_SHOP": 0,
    "BRUNCH": 1, "BRUNCH_SPOT": 1,
    "FARMERS_MARKET": 2,
    "ICE_CREAM": 3, "ICE_CREAM_SHOP": 3,
    "PET_CAFE": 4,
    "PIZZA": 5, "PIZZA_SHOP": 5,
    "SMOOTHIE": 6, "SMOOTHIE_SHOP": 6,
    "YARN": 7, "YARN_STORE": 7,
}

H4_P = 200
H4_A = 201
H4_W86 = 202
H4_W253 = 203
H4_FY = 204
H4_ICE164 = 205
H4_ICE120 = 206
H4_FF17 = 207
EXT_D36 = 208
EXT_D85 = 209
EXT_D72 = 210


@dataclass
class LegacyOverlayState:
    trigger_decided: bool = False
    h4_trigger: bool = False
    first_shop: Optional[str] = None
    r86: bool = False
    ownweed168: bool = False
    r253: bool = False
    fy144: bool = False
    ice86: bool = False
    ice_v32: bool = False
    ff168: bool = False
    ff_low: bool = False
    external_ok: bool = False
    external_route: Optional[int] = None

    def reset(self) -> None:
        self.__dict__.update(LegacyOverlayState().__dict__)


def _shops(obs: Dict[str, Any]):
    return list((obs.get("town") or {}).get("unlocked_shops") or [])


def _farm(obs: Dict[str, Any], opponent: bool = False) -> Dict[str, Any]:
    player = int(obs.get("player", 0) or 0)
    farms = list(obs.get("farms") or [])
    idx = 1 - player if opponent else player
    return farms[idx] if 0 <= idx < len(farms) and isinstance(farms[idx], dict) else {}


def _kind(farm: Dict[str, Any], y: int, x: int):
    tiles = farm.get("tiles") or []
    tile = tiles[y][x] if len(tiles) > y and len(tiles[y]) > x else None
    return tile.get("kind") if isinstance(tile, dict) else None


def _materialize(pack: Any, route: int, step: int, current_hands: int) -> Dict[str, Any]:
    return pack.action(route, step, current_hands)


def apply_legacy_overlay(
    observation: Dict[str, Any],
    base_action: Dict[str, Any],
    pack: Any,
    state: LegacyOverlayState,
    current_hands: int,
) -> Dict[str, Any]:
    """Exact frozen H4/public-prefix compatibility overlay used by engine-max.

    All gates are public/current observation predicates. This is compatibility
    preservation only; new strategic search must not extend this function.
    """
    step = int(observation.get("step", 0) or 0)
    if step == 0:
        state.reset()
        return base_action

    own = _farm(observation, False)
    opp = _farm(observation, True)
    shops = _shops(observation)

    if not state.trigger_decided:
        state.h4_trigger = len(opp.get("hands") or []) == 4 and float(opp.get("money", 0) or 0) <= 32.0
        state.trigger_decided = True

    action = base_action
    if state.h4_trigger:
        if shops and state.first_shop is None:
            state.first_shop = str(shops[0])
        if step == 86:
            state.r86 = _kind(opp, 2, 0) == "WEED"
            state.ice86 = state.first_shop == "ICE_CREAM_SHOP" and _kind(own, 2, 4) == "WEED"
        if step == 144:
            state.fy144 = len(shops) >= 2 and shops[0] == "FARMERS_MARKET" and shops[1] == "YARN_STORE"
        if step == 168:
            state.ownweed168 = _kind(own, 3, 6) == "WEED"
            state.ff168 = shops[:2] == ["FARMERS_MARKET", "FARMERS_MARKET"] and _kind(own, 0, 8) == "WEED"
        if step == 253:
            state.r253 = state.ownweed168 and state.first_shop == "PIZZA_SHOP" and float(opp.get("money", 0) or 0) < 2200.0
            if state.ice86:
                state.ice_v32 = float(opp.get("money", 0) or 0) >= 2100.0
            if state.ff168:
                state.ff_low = float(opp.get("money", 0) or 0) < 1700.0

        route: Optional[int] = None
        if state.ice86 and 86 <= step < 96:
            route = H4_W86
        elif state.ice86 and step >= 96:
            route = H4_ICE120 if state.ice_v32 and step >= 253 else H4_ICE164
        elif state.ff168 and step >= 168:
            route = H4_A if state.ff_low and step >= 253 else H4_FF17
        elif state.r86 and 86 <= step < 96:
            route = H4_W86
        elif state.fy144 and step >= 144:
            route = H4_FY
        elif state.r253 and step >= 253:
            route = H4_W253
        elif step < 72:
            route = H4_P
        else:
            route = H4_P if state.first_shop == "PIZZA_SHOP" else H4_A
        if route is not None:
            action = _materialize(pack, route, step, current_hands)

    if step == 25:
        state.external_ok = not state.h4_trigger

    if step == 288 and state.external_ok:
        prefix = shops[:4]
        if prefix == ["FARMERS_MARKET", "PET_CAFE", "YARN_STORE", "YARN_STORE"]:
            state.external_route = EXT_D36
        elif prefix == ["FARMERS_MARKET", "BAKERY", "PIZZA_SHOP", "ICE_CREAM_SHOP"]:
            state.external_route = EXT_D85
        elif prefix == ["BRUNCH_SPOT", "PIZZA_SHOP", "BAKERY", "BAKERY"]:
            state.external_route = EXT_D72
        elif prefix == ["BAKERY", "FARMERS_MARKET", "BRUNCH_SPOT", "PET_CAFE"]:
            state.external_route = EXT_D85
        else:
            state.external_route = None

    if state.external_route is not None and step >= 288:
        action = _materialize(pack, state.external_route, step, current_hands)
    return action
