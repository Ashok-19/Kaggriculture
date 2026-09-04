from __future__ import annotations

import struct
from typing import Any, Mapping

from . import bonsai_kg as bkg
from .route_format import ACTION_SIZE, MAX_MARKET, MAX_SAVED_HANDS, ORDER, UNIT


def _encode_unit(action: Any) -> bytes:
    values = list(action or ["PASS"])
    name = str(values[0]) if values else "PASS"
    if name not in bkg.UNIT_OPS:
        raise ValueError(f"unknown unit op {name}")
    arg, count = 0, 1
    if len(values) >= 2:
        if isinstance(values[1], str):
            if values[1] not in bkg.ITEM_ID:
                raise ValueError(f"unknown unit item {values[1]}")
            arg = bkg.ITEM_ID[values[1]]
        else:
            count = int(values[1])
    if len(values) >= 3:
        count = int(values[2])
    if not -32768 <= count <= 32767:
        raise ValueError(f"unit quantity out of int16 range: {count}")
    return UNIT.pack(bkg.UNIT_OPS[name], arg, count)


def _encode_order(order: Any) -> bytes:
    values = list(order or [])
    if not values:
        return ORDER.pack(0, 0, 0)
    name = str(values[0])
    if name not in bkg.MARKET_OPS:
        raise ValueError(f"unknown market op {name}")
    if name in ("HIRE", "BUY_LAND"):
        return ORDER.pack(bkg.MARKET_OPS[name], 0, 1)
    if len(values) < 3 or str(values[1]) not in bkg.ITEM_ID:
        raise ValueError(f"invalid market order {values}")
    count = int(values[2])
    if not 0 <= count <= 65535:
        raise ValueError(f"market quantity out of uint16 range: {count}")
    return ORDER.pack(bkg.MARKET_OPS[name], bkg.ITEM_ID[str(values[1])], count)


def encode_action(action: Mapping[str, Any]) -> bytes:
    hands = list(action.get("hands", []) or [])
    market = list(action.get("market", []) or [])
    if len(hands) > MAX_SAVED_HANDS or len(market) > MAX_MARKET:
        raise ValueError("route action exceeds compatibility pack capacity")
    encoded = bytearray(struct.pack("<BB", len(hands), len(market)))
    encoded += _encode_unit(action.get("farmer") or ["PASS"])
    for index in range(MAX_SAVED_HANDS):
        encoded += _encode_unit(hands[index]) if index < len(hands) else UNIT.pack(bkg.UNIT_OPS["PASS"], 0, 1)
    for index in range(MAX_MARKET):
        encoded += _encode_order(market[index]) if index < len(market) else ORDER.pack(0, 0, 0)
    if len(encoded) != ACTION_SIZE:
        raise AssertionError((len(encoded), ACTION_SIZE))
    return bytes(encoded)
