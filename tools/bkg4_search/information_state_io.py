from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

HEADER = struct.Struct("<8s8I")
MAGIC_R0 = b"B4ISR01\0"


@dataclass(frozen=True)
class R0Capture:
    path: Path
    version: int
    cases: int
    probes: tuple[int, ...]
    width: int
    registry_width: int
    bb_width: int
    compat_width: int
    meta_width: int
    data: memoryview

    def frame(self, case: int, probe_index: int) -> memoryview:
        if not (0 <= case < self.cases and 0 <= probe_index < len(self.probes)):
            raise IndexError((case, probe_index))
        start = (case * len(self.probes) + probe_index) * self.width
        return self.data[start:start + self.width]


def load_r0(path: str | Path) -> R0Capture:
    p = Path(path)
    raw = p.read_bytes()
    if len(raw) < HEADER.size:
        raise ValueError("short R0 capture")
    magic, version, cases, nprobes, width, registry, bb, compat, meta = HEADER.unpack_from(raw, 0)
    if magic != MAGIC_R0 or version != 1:
        raise ValueError("invalid R0 capture header")
    off = HEADER.size
    need_probes = nprobes * 4
    if len(raw) < off + need_probes:
        raise ValueError("short probe table")
    probes = struct.unpack_from(f"<{nprobes}i", raw, off)
    off += need_probes
    expected = off + cases * nprobes * width * 4
    if len(raw) != expected:
        raise ValueError(f"R0 size mismatch {len(raw)} != {expected}")
    ints = memoryview(raw)[off:].cast("i")
    return R0Capture(p, version, cases, tuple(probes), width, registry, bb, compat, meta, ints)


def registry_manifest() -> list[tuple[str, tuple[int, ...]]]:
    """Exact registry-v1 enumeration order used by the CUDA R0 capture."""
    out: list[tuple[str, tuple[int, ...]]] = []
    for name in (
        "ALWAYS", "STEP", "DAY", "HOUR", "OWN_MONEY", "OPP_MONEY", "OWN_UNITS", "OPP_UNITS",
        "OWN_LAND", "OPP_LAND", "OWN_HIRES_TODAY", "OWN_SHED_TOTAL", "SHOP_COUNT", "OPP_HIRES_TODAY",
    ):
        out.append((name, ()))
    for name in ("MARKET_PRICE", "MARKET_INVENTORY"):
        for item in range(9): out.append((name, (item,)))
    for item in range(12): out.append(("OWN_SHED", (item,)))
    for item in range(5): out.append(("OWN_SEED", (item,)))
    for unit in range(40):
        for item in range(12): out.append(("OWN_INVENTORY", (unit, item)))
    for i in range(8): out.append(("SHOP_AT", (i,)))
    for i in range(8): out.append(("SHOP_SEEN", (i,)))
    for q in range(4): out.append(("OWN_QUADRANT_UNLOCKED", (q,)))
    for q in range(4): out.append(("OPP_QUADRANT_UNLOCKED", (q,)))
    for name in ("OWN_WORKER_X", "OWN_WORKER_Y", "OPP_WORKER_X", "OPP_WORKER_Y", "OWN_WORKER_TILE", "OPP_WORKER_TILE"):
        for unit in range(40): out.append((name, (unit,)))
    tile_names = (
        "OWN_TILE_KIND", "OPP_TILE_KIND", "OWN_TILE_ITEM", "OPP_TILE_ITEM", "OWN_TILE_YIELD", "OPP_TILE_YIELD",
        "OWN_TILE_WATERED", "OPP_TILE_WATERED", "OWN_TILE_FED", "OPP_TILE_FED", "OWN_TILE_CARED", "OPP_TILE_CARED",
        "OWN_TILE_FERT_AVAILABLE", "OPP_TILE_FERT_AVAILABLE",
        "OWN_TILE_PLANTED_DAY", "OPP_TILE_PLANTED_DAY", "OWN_TILE_CONSECUTIVE_UNWATERED", "OPP_TILE_CONSECUTIVE_UNWATERED",
        "OWN_TILE_MAX_LIFESPAN_STEP", "OPP_TILE_MAX_LIFESPAN_STEP", "OWN_TILE_FERTILIZED_UNTIL_DAY", "OPP_TILE_FERTILIZED_UNTIL_DAY",
        "OWN_TILE_PLACED_DAY", "OPP_TILE_PLACED_DAY", "OWN_TILE_CONSECUTIVE_UNFED", "OPP_TILE_CONSECUTIVE_UNFED",
        "OWN_TILE_PENDING_CARE_BONUS", "OPP_TILE_PENDING_CARE_BONUS", "OWN_TILE_HAS_ANIMAL", "OPP_TILE_HAS_ANIMAL",
    )
    for name in tile_names:
        for tile in range(100): out.append((name, (tile,)))
    return out


COMPAT_NAMES = (
    "legacy.trigger_decided", "legacy.h4_trigger", "legacy.first_shop", "legacy.r86", "legacy.ownweed168",
    "legacy.r253", "legacy.fy144", "legacy.ice86", "legacy.ice_v32", "legacy.ff168", "legacy.ff_low",
    "legacy.external_ok", "legacy.external_route", "exploit.high_labor", "exploit.carrot_latched",
)
META_NAMES = ("runtime.fault", "runtime.ticks")


HEADER_R1 = struct.Struct("<8s4I")
MAGIC_R1 = b"B4ISR11\0"


@dataclass(frozen=True)
class R1Capture:
    path: Path
    version: int
    cases: int
    probes: tuple[int, ...]
    width: int
    data: memoryview

    def frame(self, case: int, probe_index: int) -> memoryview:
        if not (0 <= case < self.cases and 0 <= probe_index < len(self.probes)):
            raise IndexError((case, probe_index))
        start = (case * len(self.probes) + probe_index) * self.width
        return self.data[start:start + self.width]


def load_r1(path: str | Path) -> R1Capture:
    p = Path(path); raw = p.read_bytes()
    if len(raw) < HEADER_R1.size: raise ValueError("short R1 capture")
    magic, version, cases, nprobes, width = HEADER_R1.unpack_from(raw, 0)
    if magic != MAGIC_R1 or version != 1: raise ValueError("invalid R1 capture header")
    off = HEADER_R1.size
    probes = struct.unpack_from(f"<{nprobes}i", raw, off); off += nprobes * 4
    expected = off + cases * nprobes * width * 4
    if len(raw) != expected: raise ValueError(f"R1 size mismatch {len(raw)} != {expected}")
    return R1Capture(p, version, cases, tuple(probes), width, memoryview(raw)[off:].cast("i"))


def r1_manifest() -> list[str]:
    sem = [f"market_inv_{i}" for i in range(9)] + [f"market_price_{i}" for i in range(9)] + [
        "own_money", "opp_money", "own_units", "opp_units", "own_land", "opp_land", "own_hires_today", "opp_hires_today",
    ] + [f"opp_crop_{i}" for i in range(5)] + [f"opp_animal_{i}" for i in range(3)] + [f"opp_kind_{i}" for i in range(6)] + [f"own_crop_{i}" for i in range(5)] + [f"own_animal_{i}" for i in range(3)] + [f"own_kind_{i}" for i in range(6)] + ["shop_count"]
    assert len(sem) == 55
    out=[]
    for h in (1,4,24,72): out += [f"delta{h}.{x}" for x in sem]
    for h in (1,4,24,72): out.append(f"route_at_minus_{h}")
    for h in (1,4,24,72): out.append(f"valid_{h}")
    unit_ops=("PASS","NORTH","SOUTH","EAST","WEST","PICKUP","DROP","PLACE","PLANT","WATER","HARVEST","FERTILIZE","DIG","BUILD_COOP","BUILD_PASTURE","FEED","COLLECT_FERTILIZER","CARE")
    market_ops=("HIRE","BUY_LAND","BUY_SEED","BUY_PRODUCT","BUY_ANIMAL","SELL")
    for h in (1,4,24,72): out += [f"own_action_count{h}.unit.{x}" for x in unit_ops] + [f"own_action_count{h}.market.{x}" for x in market_ops]
    out += ["since_opp_land_increase","since_opp_units_increase","since_opp_crop_change","since_opp_animal_change","since_opp_structure_change","since_opp_weed_change","since_shop_unlock","since_route_change"]
    for item in range(9): out += [f"since_market_inv_increase_{item}",f"since_market_inv_decrease_{item}",f"since_market_cross_below_10000_{item}",f"since_market_cross_above_10000_{item}"]
    assert len(out)==368
    return out


HIDDEN_TRUTH_NAMES = tuple([f"opp_total_item_{i}" for i in range(12)] + [f"opp_seed_{i}" for i in range(5)])
MAGIC_TRUTH = b"B4IST11\0"


def load_hidden_truth(path: str | Path) -> R0Capture:
    p = Path(path)
    raw = p.read_bytes()
    if len(raw) < HEADER.size:
        raise ValueError("short hidden-truth capture")
    magic, version, cases, nprobes, width, registry, bb, compat, meta = HEADER.unpack_from(raw, 0)
    if magic != MAGIC_TRUTH or version != 1:
        raise ValueError("invalid hidden-truth capture header")
    if width != 17 or registry != 17 or bb or compat or meta:
        raise ValueError("invalid hidden-truth diagnostic dimensions")
    off = HEADER.size
    probes = struct.unpack_from(f"<{nprobes}i", raw, off)
    off += nprobes * 4
    expected = off + cases * nprobes * width * 4
    if len(raw) != expected:
        raise ValueError(f"hidden-truth size mismatch {len(raw)} != {expected}")
    return R0Capture(p, version, cases, tuple(probes), width, registry, bb, compat, meta, memoryview(raw)[off:].cast("i"))
