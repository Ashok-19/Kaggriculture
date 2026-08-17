from __future__ import annotations

import copy
import hashlib
import json
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

SCHEMA_VERSION = 2
ABI_VERSION = 2
MAGIC = b"BKG2"
BOARD_SIZE = 10

MAX_NODES = 128
MAX_DEPTH = 24
MAX_CHILD_REFS = 256
MAX_BB = 32
MAX_STATE = 48
MAX_PRED = 128
MAX_LIFECYCLE = 64
MAX_EFFECT = 128
MAX_VALUE_REFS = 256
MAX_PARALLEL_CHILDREN = 16
MAX_UNITS = 40
MAX_MARKET_INTENTS = 16
MAX_DEFERRED_INTENTS = MAX_NODES
MAX_VISITS = 4096
NONE_U16 = 0xFFFF
ACTIVE_BIT = 1 << 31
RUNNING_MASK16 = 0xFFFF

STATUS_SUCCESS = 0
STATUS_FAILURE = 1
STATUS_RUNNING = 2
STATUS_SKIPPED = 3  # internal only
STATUS_NAMES = ("SUCCESS", "FAILURE", "RUNNING", "SKIPPED")

NODE_TYPES = {
    "Sequence": 0,
    "Selector": 1,
    "ReactiveSequence": 2,
    "ReactiveSelector": 3,
    "If": 4,
    "Parallel": 5,
    "Invert": 6,
    "ForceSuccess": 7,
    "ForceFailure": 8,
    "TurnTimeout": 9,
    "Retry": 10,
    "Repeat": 11,
    "KeepRunningUntilFailure": 12,
    "Success": 13,
    "Failure": 14,
    "Condition": 15,
    "SetBlackboard": 16,
    "AddBlackboard": 17,
    "EmitUnitIntent": 18,
    "EmitMarketIntent": 19,
    "WaitTurns": 20,
    "MoveUnitTo": 21,
}
NODE_NAMES = {v: k for k, v in NODE_TYPES.items()}

STATEFUL_TYPES = {
    NODE_TYPES[x]
    for x in (
        "Sequence", "Selector", "ReactiveSequence", "ReactiveSelector", "If",
        "Parallel", "TurnTimeout", "Retry", "Repeat", "WaitTurns"
    )
}
DECORATOR_TYPES = {
    NODE_TYPES[x]
    for x in (
        "Invert", "ForceSuccess", "ForceFailure", "TurnTimeout", "Retry", "Repeat",
        "KeepRunningUntilFailure"
    )
}

CMP = {"EQ": 0, "NE": 1, "LT": 2, "LE": 3, "GT": 4, "GE": 5}
CMP_NAMES = {v: k for k, v in CMP.items()}

BB_TYPES = {"BOOL": 0, "I32": 1, "U16": 2, "ENUM8": 3, "TILE": 4, "UNIT": 5}
BB_NAMES = {v: k for k, v in BB_TYPES.items()}
VALUE_TYPES = {name: BB_TYPES[name] for name in ("I32", "TILE", "UNIT")}
VALUE_KINDS = {"CONST": 0, "BB": 1, "LEGAL_FEATURE": 2}

EFFECT_OP = {"set": 0, "clear": 1, "add": 2}

ITEMS = ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL", "FERTILIZER", "GOOSE", "COW", "SHEEP")
ITEM_ID = {name: i for i, name in enumerate(ITEMS)}
N_PRODUCTS = 9
N_CROPS = 5
N_ANIMALS = 3
SHOPS = ("BAKERY", "BRUNCH_SPOT", "FARMERS_MARKET", "ICE_CREAM_SHOP", "PET_CAFE", "PIZZA_SHOP", "SMOOTHIE_SHOP", "YARN_STORE")
SHOP_ID = {name: i for i, name in enumerate(SHOPS)}
SHOP_ID.update({"BAKERY_SHOP": 0, "BRUNCH": 1, "ICE_CREAM": 3, "PET_CAFE": 4, "PIZZA": 5, "SMOOTHIE": 6, "YARN": 7})

FEATURES = {
    "ALWAYS": 0,
    "STEP": 1,
    "DAY": 2,
    "HOUR": 3,
    "OWN_MONEY": 4,
    "OPP_MONEY": 5,
    "OWN_UNITS": 6,
    "OPP_UNITS": 7,
    "OWN_LAND": 8,
    "OPP_LAND": 9,
    "OWN_HIRES_TODAY": 10,
    "MARKET_PRICE": 11,
    "MARKET_INVENTORY": 12,
    "OWN_SHED": 13,
    "OWN_SHED_TOTAL": 14,
    "OWN_SEED": 15,
    "OWN_INVENTORY": 16,
    "SHOP_COUNT": 17,
    "SHOP_AT": 18,
    "SHOP_SEEN": 19,
    "OWN_WORKER_X": 20,
    "OWN_WORKER_Y": 21,
    "OPP_WORKER_X": 22,
    "OPP_WORKER_Y": 23,
    "OWN_TILE_KIND": 24,
    "OPP_TILE_KIND": 25,
    "OWN_TILE_ITEM": 26,
    "OPP_TILE_ITEM": 27,
    "OWN_TILE_YIELD": 28,
    "OPP_TILE_YIELD": 29,
    "OWN_TILE_WATERED": 30,
    "OPP_TILE_WATERED": 31,
    "OWN_TILE_FED": 32,
    "OPP_TILE_FED": 33,
    "OWN_TILE_CARED": 34,
    "OPP_TILE_CARED": 35,
    "OWN_TILE_FERT_AVAILABLE": 36,
    "OPP_TILE_FERT_AVAILABLE": 37,
    "OPP_HIRES_TODAY": 38,
    "OWN_QUADRANT_UNLOCKED": 39,
    "OPP_QUADRANT_UNLOCKED": 40,
    "OWN_WORKER_TILE": 41,
    "OPP_WORKER_TILE": 42,
    "OWN_TILE_PLANTED_DAY": 43,
    "OPP_TILE_PLANTED_DAY": 44,
    "OWN_TILE_CONSECUTIVE_UNWATERED": 45,
    "OPP_TILE_CONSECUTIVE_UNWATERED": 46,
    "OWN_TILE_MAX_LIFESPAN_STEP": 47,
    "OPP_TILE_MAX_LIFESPAN_STEP": 48,
    "OWN_TILE_FERTILIZED_UNTIL_DAY": 49,
    "OPP_TILE_FERTILIZED_UNTIL_DAY": 50,
    "OWN_TILE_PLACED_DAY": 51,
    "OPP_TILE_PLACED_DAY": 52,
    "OWN_TILE_CONSECUTIVE_UNFED": 53,
    "OPP_TILE_CONSECUTIVE_UNFED": 54,
    "OWN_TILE_PENDING_CARE_BONUS": 55,
    "OPP_TILE_PENDING_CARE_BONUS": 56,
    "OWN_TILE_HAS_ANIMAL": 57,
    "OPP_TILE_HAS_ANIMAL": 58,
}
FEATURE_NAMES = {v: k for k, v in FEATURES.items()}
FEATURE_TYPES = {fid: BB_TYPES["I32"] for fid in FEATURE_NAMES}
FEATURE_TYPES[FEATURES["OWN_WORKER_TILE"]] = BB_TYPES["TILE"]
FEATURE_TYPES[FEATURES["OPP_WORKER_TILE"]] = BB_TYPES["TILE"]
TILE_FEATURES = {
    FEATURES[name] for name in FEATURE_NAMES.values()
    if name.startswith("OWN_TILE_") or name.startswith("OPP_TILE_")
}

UNIT_OPS = {
    "PASS": 0, "NORTH": 1, "SOUTH": 2, "EAST": 3, "WEST": 4,
    "PICKUP": 5, "DROP": 6, "PLACE": 7, "PLANT": 8, "WATER": 9,
    "HARVEST": 10, "FERTILIZE": 11, "DIG": 12, "BUILD_COOP": 13,
    "BUILD_PASTURE": 14, "FEED": 15, "COLLECT_FERTILIZER": 16, "CARE": 17,
}
MARKET_OPS = {"HIRE": 1, "BUY_LAND": 2, "BUY_SEED": 3, "BUY_PRODUCT": 4, "BUY_ANIMAL": 5, "SELL": 6}

HEADER = struct.Struct("<4sHH32sHHHHHHHHHHHH")
NODE_STRUCT = struct.Struct("<BBHHHH2siiiiI")
BB_STRUCT = struct.Struct("<B3si")
VALUE_STRUCT = struct.Struct("<BBHii")
PRED_STRUCT = struct.Struct("<HHB7s")
LIFECYCLE_STRUCT = struct.Struct("<12H")
EFFECT_STRUCT = struct.Struct("<BBHi")
assert HEADER.size == 64 and NODE_STRUCT.size == 32 and BB_STRUCT.size == 8
assert VALUE_STRUCT.size == 12 and PRED_STRUCT.size == 12 and LIFECYCLE_STRUCT.size == 24 and EFFECT_STRUCT.size == 8


class PolicyError(ValueError):
    pass


@dataclass
class Node:
    type: int
    flags: int = 0
    first_child: int = 0
    child_count: int = 0
    state_slot: int = NONE_U16
    lifecycle_slot: int = NONE_U16
    p0: int = 0
    p1: int = 0
    p2: int = 0
    p3: int = 0
    debug_id: int = 0


@dataclass
class BbDecl:
    type: int
    initial: int
    name: str = ""


@dataclass
class ValueRef:
    kind: int
    type: int
    id: int = 0
    arg: int = 0
    value: int = 0


@dataclass
class Predicate:
    lhs: int
    rhs: int
    cmp: int


@dataclass
class Effect:
    op: int
    slot: int
    value: int


@dataclass
class Lifecycle:
    failure_if: int = NONE_U16
    success_if: int = NONE_U16
    skip_if: int = NONE_U16
    while_pred: int = NONE_U16
    on_success: Tuple[int, int] = (0, 0)
    on_failure: Tuple[int, int] = (0, 0)
    on_halted: Tuple[int, int] = (0, 0)
    post: Tuple[int, int] = (0, 0)


@dataclass
class CompiledPolicy:
    semantic_hash: str
    root: int
    nodes: List[Node]
    child_refs: List[int]
    bb: List[BbDecl]
    values: List[ValueRef]
    predicates: List[Predicate]
    lifecycles: List[Lifecycle]
    effects: List[Effect]
    state_count: int
    instruction_budget: int
    board_size: int = BOARD_SIZE
    source: Optional[Dict[str, Any]] = None

    @property
    def canonical_hash(self) -> str:
        return self.semantic_hash

    @property
    def compiled_bytes(self) -> int:
        return (
            HEADER.size + len(self.nodes) * NODE_STRUCT.size + len(self.child_refs) * 2
            + len(self.bb) * BB_STRUCT.size + len(self.values) * VALUE_STRUCT.size
            + len(self.predicates) * PRED_STRUCT.size
            + len(self.lifecycles) * LIFECYCLE_STRUCT.size + len(self.effects) * EFFECT_STRUCT.size
        )


@dataclass
class NodeState:
    a: int = 0
    b: int = 0


@dataclass
class UnitClaim:
    priority: int
    order: int
    op: int
    arg: int
    n: int


@dataclass
class MarketClaim:
    priority: int
    order: int
    op: int
    item: int
    n: int


@dataclass
class DeferredIntent:
    kind: int  # 0 unit, 1 market
    priority: int
    op: int
    arg: int
    n: int
    subject: int
    order: int


@dataclass
class RuntimeState:
    blackboard: List[int]
    node_state: List[NodeState]
    intents: List[DeferredIntent] = field(default_factory=list)
    unit_claims: List[Optional[UnitClaim]] = field(default_factory=lambda: [None] * MAX_UNITS)
    market_claims: List[MarketClaim] = field(default_factory=list)
    emit_seq: int = 0
    visits: int = 0
    fault: int = 0
    ticks: int = 0


@dataclass
class ActionResult:
    n_units: int
    units: List[Tuple[int, int, int]]
    market: List[Tuple[int, int, int]]

    def compact(self) -> Tuple[Any, ...]:
        return (self.n_units, tuple(self.units), tuple(self.market))


@dataclass
class TickResult:
    status: int
    action: ActionResult
    fault: int
    blackboard: Tuple[int, ...]
    node_state: Tuple[Tuple[int, int], ...]
    trace: Tuple[Tuple[int, int, str, str], ...]


class SyntheticObservation:
    """Sparse deterministic feature source for differential tests."""

    def __init__(self, values: Optional[Dict[Tuple[int, int], int]] = None):
        self.values = values or {}

    def feature(self, feature_id: int, arg: int = 0) -> int:
        if feature_id == FEATURES["ALWAYS"]:
            return 1
        return int(self.values.get((feature_id, arg), self.values.get((feature_id, 0), 0)))


class DictObservation:
    """Official/pybridge observation adapter; exposes only the ABI-v2 legal registry."""

    def __init__(self, observation: Dict[str, Any]):
        self.obs = observation
        self.player = int(observation.get("player", 0) or 0)
        farms = observation.get("farms")
        if self.player not in (0, 1) or not isinstance(farms, (list, tuple)) or len(farms) != 2:
            raise PolicyError("observation requires player 0/1 and exactly two farms")
        for farm in farms:
            tiles = farm.get("tiles") if isinstance(farm, dict) else None
            if not isinstance(tiles, (list, tuple)) or len(tiles) != BOARD_SIZE or any(
                not isinstance(row, (list, tuple)) or len(row) != BOARD_SIZE for row in tiles
            ):
                raise PolicyError("Bonsai-KG requires boardSize=10 observations")

    @staticmethod
    def _units(farm: Dict[str, Any]) -> int:
        return 1 + len(list(farm.get("hands", []) or []))

    @staticmethod
    def _land(farm: Dict[str, Any]) -> int:
        return len(list(farm.get("unlocked_quadrants", []) or [])) or 1

    @staticmethod
    def _pos(farm: Dict[str, Any], unit: int) -> Tuple[int, int]:
        if unit == 0:
            p = farm.get("farmer", [-1, -1]) or [-1, -1]
        else:
            hands = list(farm.get("hands", []) or [])
            p = hands[unit - 1] if 0 <= unit - 1 < len(hands) else [-1, -1]
        return (int(p[0]), int(p[1])) if isinstance(p, (list, tuple)) and len(p) >= 2 else (-1, -1)

    @staticmethod
    def _tile(farm: Dict[str, Any], tile_idx: int) -> Any:
        if not 0 <= tile_idx < BOARD_SIZE * BOARD_SIZE:
            return None
        rows = list(farm.get("tiles", []) or [])
        y, x = divmod(tile_idx, BOARD_SIZE)
        if y >= len(rows) or not isinstance(rows[y], (list, tuple)) or x >= len(rows[y]):
            return None
        return rows[y][x]

    @staticmethod
    def _tile_kind(tile: Any) -> int:
        if tile is None:
            return 0
        if tile == "LOCKED":
            return 1
        if not isinstance(tile, dict):
            return 0
        return {"WEED": 2, "COOP": 3, "PASTURE": 4, "PLANT": 5}.get(str(tile.get("kind", "")), 0)

    @staticmethod
    def _tile_item(tile: Any) -> int:
        if not isinstance(tile, dict):
            return 0
        name = tile.get("crop", tile.get("animal", ""))
        return ITEM_ID.get(str(name), 0)

    def feature(self, feature_id: int, arg: int = 0) -> int:
        o = self.obs
        farms = list(o.get("farms", []) or [])
        own = farms[self.player] if self.player < len(farms) else {}
        opp_i = 1 - self.player
        opp = farms[opp_i] if opp_i < len(farms) else {}
        priv = dict(o.get("private", {}) or {})
        if feature_id == FEATURES["ALWAYS"]: return 1
        if feature_id == FEATURES["STEP"]: return int(o.get("step", 0) or 0)
        if feature_id == FEATURES["DAY"]: return int(o.get("day", int(o.get("step", 0) or 0) // 24) or 0)
        if feature_id == FEATURES["HOUR"]: return int(o.get("hour", int(o.get("step", 0) or 0) % 24) or 0)
        if feature_id == FEATURES["OWN_MONEY"]: return int(float(own.get("money", 0) or 0))
        if feature_id == FEATURES["OPP_MONEY"]: return int(float(opp.get("money", 0) or 0))
        if feature_id == FEATURES["OWN_UNITS"]: return self._units(own)
        if feature_id == FEATURES["OPP_UNITS"]: return self._units(opp)
        if feature_id == FEATURES["OWN_LAND"]: return self._land(own)
        if feature_id == FEATURES["OPP_LAND"]: return self._land(opp)
        if feature_id == FEATURES["OWN_HIRES_TODAY"]: return int(own.get("hires_today", 0) or 0)
        if feature_id == FEATURES["OPP_HIRES_TODAY"]: return int(opp.get("hires_today", 0) or 0)
        if feature_id in (FEATURES["OWN_QUADRANT_UNLOCKED"], FEATURES["OPP_QUADRANT_UNLOCKED"]):
            farm = own if feature_id == FEATURES["OWN_QUADRANT_UNLOCKED"] else opp
            quadrant = ("NW", "NE", "SW", "SE")[arg] if 0 <= arg < 4 else ""
            return int(quadrant in list(farm.get("unlocked_quadrants", []) or []))
        market = dict(o.get("market", {}) or {})
        if feature_id == FEATURES["MARKET_PRICE"]:
            return int(dict(market.get("prices", {}) or {}).get(ITEMS[arg] if 0 <= arg < N_PRODUCTS else "", 0) or 0)
        if feature_id == FEATURES["MARKET_INVENTORY"]:
            return int(dict(market.get("inventory", {}) or {}).get(ITEMS[arg] if 0 <= arg < N_PRODUCTS else "", 0) or 0)
        shed = dict(priv.get("shed", {}) or {})
        if feature_id == FEATURES["OWN_SHED"]:
            return int(shed.get(ITEMS[arg] if 0 <= arg < len(ITEMS) else "", 0) or 0)
        if feature_id == FEATURES["OWN_SHED_TOTAL"]:
            return sum(int(v or 0) for v in shed.values())
        seeds = dict(priv.get("seeds", {}) or {})
        if feature_id == FEATURES["OWN_SEED"]:
            return int(seeds.get(ITEMS[arg] if 0 <= arg < N_CROPS else "", 0) or 0)
        if feature_id == FEATURES["OWN_INVENTORY"]:
            unit, item = (arg >> 8) & 0xFF, arg & 0xFF
            invs = list(priv.get("inventories", []) or [])
            inv = dict(invs[unit] or {}) if 0 <= unit < len(invs) and isinstance(invs[unit], dict) else {}
            return int(inv.get(ITEMS[item] if 0 <= item < len(ITEMS) else "", 0) or 0)
        shops = list(dict(o.get("town", {}) or {}).get("unlocked_shops", []) or [])
        if feature_id == FEATURES["SHOP_COUNT"]: return len(shops)
        if feature_id == FEATURES["SHOP_AT"]:
            return SHOP_ID.get(str(shops[arg]), -1) if 0 <= arg < len(shops) else -1
        if feature_id == FEATURES["SHOP_SEEN"]:
            return sum(1 for s in shops if SHOP_ID.get(str(s), -1) == arg)
        if feature_id in (FEATURES["OWN_WORKER_X"], FEATURES["OWN_WORKER_Y"]):
            pos = self._pos(own, arg); return pos[feature_id == FEATURES["OWN_WORKER_Y"]]
        if feature_id in (FEATURES["OPP_WORKER_X"], FEATURES["OPP_WORKER_Y"]):
            pos = self._pos(opp, arg); return pos[feature_id == FEATURES["OPP_WORKER_Y"]]
        if feature_id in (FEATURES["OWN_WORKER_TILE"], FEATURES["OPP_WORKER_TILE"]):
            pos = self._pos(own if feature_id == FEATURES["OWN_WORKER_TILE"] else opp, arg)
            return pos[1] * BOARD_SIZE + pos[0] if 0 <= pos[0] < BOARD_SIZE and 0 <= pos[1] < BOARD_SIZE else -1
        own_tile_features = {
            FEATURES["OWN_TILE_KIND"], FEATURES["OWN_TILE_ITEM"], FEATURES["OWN_TILE_YIELD"],
            FEATURES["OWN_TILE_WATERED"], FEATURES["OWN_TILE_FED"], FEATURES["OWN_TILE_CARED"],
            FEATURES["OWN_TILE_FERT_AVAILABLE"], FEATURES["OWN_TILE_PLANTED_DAY"],
            FEATURES["OWN_TILE_CONSECUTIVE_UNWATERED"], FEATURES["OWN_TILE_MAX_LIFESPAN_STEP"],
            FEATURES["OWN_TILE_FERTILIZED_UNTIL_DAY"], FEATURES["OWN_TILE_PLACED_DAY"],
            FEATURES["OWN_TILE_CONSECUTIVE_UNFED"], FEATURES["OWN_TILE_PENDING_CARE_BONUS"],
            FEATURES["OWN_TILE_HAS_ANIMAL"]
        }
        tile_farm = own if feature_id in own_tile_features else opp
        tile = self._tile(tile_farm, arg)
        if feature_id in (FEATURES["OWN_TILE_KIND"], FEATURES["OPP_TILE_KIND"]): return self._tile_kind(tile)
        if feature_id in (FEATURES["OWN_TILE_ITEM"], FEATURES["OPP_TILE_ITEM"]): return self._tile_item(tile)
        if feature_id in (FEATURES["OWN_TILE_YIELD"], FEATURES["OPP_TILE_YIELD"]): return int(tile.get("yield_units", 0) or 0) if isinstance(tile, dict) else 0
        if feature_id in (FEATURES["OWN_TILE_WATERED"], FEATURES["OPP_TILE_WATERED"]): return int(bool(tile.get("watered_today", False))) if isinstance(tile, dict) else 0
        if feature_id in (FEATURES["OWN_TILE_FED"], FEATURES["OPP_TILE_FED"]): return int(bool(tile.get("fed_today", False))) if isinstance(tile, dict) else 0
        if feature_id in (FEATURES["OWN_TILE_CARED"], FEATURES["OPP_TILE_CARED"]): return int(bool(tile.get("cared_today", False))) if isinstance(tile, dict) else 0
        if feature_id in (FEATURES["OWN_TILE_FERT_AVAILABLE"], FEATURES["OPP_TILE_FERT_AVAILABLE"]): return int(bool(tile.get("fertilizer_available", False))) if isinstance(tile, dict) else 0
        if feature_id in (FEATURES["OWN_TILE_PLANTED_DAY"], FEATURES["OPP_TILE_PLANTED_DAY"]):
            return int(tile.get("planted_day", 0) or 0) if isinstance(tile, dict) and tile.get("kind") == "PLANT" else 0
        if feature_id in (FEATURES["OWN_TILE_CONSECUTIVE_UNWATERED"], FEATURES["OPP_TILE_CONSECUTIVE_UNWATERED"]):
            return int(tile.get("consecutive_unwatered", 0) or 0) if isinstance(tile, dict) and tile.get("kind") == "PLANT" else 0
        if feature_id in (FEATURES["OWN_TILE_MAX_LIFESPAN_STEP"], FEATURES["OPP_TILE_MAX_LIFESPAN_STEP"]):
            return int(tile.get("max_lifespan_step", -1)) if isinstance(tile, dict) and tile.get("kind") == "PLANT" else -1
        if feature_id in (FEATURES["OWN_TILE_FERTILIZED_UNTIL_DAY"], FEATURES["OPP_TILE_FERTILIZED_UNTIL_DAY"]):
            return int(tile.get("fertilized_until_day", -1)) if isinstance(tile, dict) and tile.get("kind") == "PLANT" else -1
        if feature_id in (FEATURES["OWN_TILE_PLACED_DAY"], FEATURES["OPP_TILE_PLACED_DAY"]):
            return int(tile.get("placed_day", 0) or 0) if isinstance(tile, dict) and tile.get("kind") in ("COOP", "PASTURE") else 0
        if feature_id in (FEATURES["OWN_TILE_CONSECUTIVE_UNFED"], FEATURES["OPP_TILE_CONSECUTIVE_UNFED"]):
            return int(tile.get("consecutive_unfed", 0) or 0) if isinstance(tile, dict) and tile.get("kind") in ("COOP", "PASTURE") else 0
        if feature_id in (FEATURES["OWN_TILE_PENDING_CARE_BONUS"], FEATURES["OPP_TILE_PENDING_CARE_BONUS"]):
            return int(tile.get("pending_care_bonus", 0) or 0) if isinstance(tile, dict) and tile.get("kind") in ("COOP", "PASTURE") else 0
        if feature_id in (FEATURES["OWN_TILE_HAS_ANIMAL"], FEATURES["OPP_TILE_HAS_ANIMAL"]):
            return int(bool(tile.get("animal"))) if isinstance(tile, dict) and tile.get("kind") in ("COOP", "PASTURE") else 0
        return 0


def canonical_payload(policy: Dict[str, Any]) -> bytes:
    obj = copy.deepcopy(policy)
    obj.pop("semantic_hash", None)
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def policy_hash(policy: Dict[str, Any]) -> str:
    return compile_policy(policy).semantic_hash


def canonical_policy(policy: Dict[str, Any]) -> Dict[str, Any]:
    obj = json.loads(canonical_payload(policy).decode("utf-8"))
    obj["semantic_hash"] = compile_policy(obj).semantic_hash
    return obj


def dumps_canonical(policy: Dict[str, Any], pretty: bool = False) -> str:
    obj = canonical_policy(policy)
    if pretty:
        return json.dumps(obj, sort_keys=True, indent=2, ensure_ascii=False) + "\n"
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _fit_i32(v: Any, what: str) -> int:
    if isinstance(v, bool):
        v = int(v)
    if not isinstance(v, int) or not -(1 << 31) <= v < (1 << 31):
        raise PolicyError(f"{what} must fit int32")
    return int(v)


def _validate_bb_value(bb_type: int, value: int, what: str) -> None:
    if bb_type == BB_TYPES["BOOL"] and value not in (0, 1): raise PolicyError(f"{what} BOOL must be 0/1")
    if bb_type == BB_TYPES["U16"] and not 0 <= value <= 65535: raise PolicyError(f"{what} U16 out of range")
    if bb_type == BB_TYPES["ENUM8"] and not 0 <= value <= 255: raise PolicyError(f"{what} ENUM8 out of range")
    if bb_type == BB_TYPES["TILE"] and not -1 <= value < BOARD_SIZE * BOARD_SIZE: raise PolicyError(f"{what} TILE out of range")
    if bb_type == BB_TYPES["UNIT"] and not -1 <= value < MAX_UNITS: raise PolicyError(f"{what} UNIT out of range")


def _feature_arg_ok(fid: int, arg: int) -> bool:
    if fid in (FEATURES["MARKET_PRICE"], FEATURES["MARKET_INVENTORY"]): return 0 <= arg < N_PRODUCTS
    if fid == FEATURES["OWN_SHED"]: return 0 <= arg < len(ITEMS)
    if fid == FEATURES["OWN_SEED"]: return 0 <= arg < N_CROPS
    if fid == FEATURES["OWN_INVENTORY"]:
        return 0 <= ((arg >> 8) & 0xFF) < MAX_UNITS and 0 <= (arg & 0xFF) < len(ITEMS) and arg <= 0xFFFF
    if fid == FEATURES["SHOP_AT"]: return 0 <= arg < 8
    if fid == FEATURES["SHOP_SEEN"]: return 0 <= arg < 8
    if fid in (FEATURES["OWN_QUADRANT_UNLOCKED"], FEATURES["OPP_QUADRANT_UNLOCKED"]): return 0 <= arg < 4
    if fid in {
        FEATURES["OWN_WORKER_X"], FEATURES["OWN_WORKER_Y"], FEATURES["OPP_WORKER_X"], FEATURES["OPP_WORKER_Y"],
        FEATURES["OWN_WORKER_TILE"], FEATURES["OPP_WORKER_TILE"],
    }: return 0 <= arg < MAX_UNITS
    if fid in TILE_FEATURES: return 0 <= arg < BOARD_SIZE * BOARD_SIZE
    return arg == 0


class Compiler:
    def __init__(self, policy: Dict[str, Any]):
        self.source = copy.deepcopy(policy)
        self.nodes: List[Node] = []
        self.child_refs: List[int] = []
        self.bb: List[BbDecl] = []
        self.bb_by_name: Dict[str, int] = {}
        self.values: List[ValueRef] = []
        self.predicates: List[Predicate] = []
        self.lifecycles: List[Lifecycle] = []
        self.effects: List[Effect] = []
        self.state_count = 0
        self.subtrees: Dict[str, Dict[str, Any]] = {}

    def compile(self) -> CompiledPolicy:
        p = self.source
        if not isinstance(p, dict) or p.get("schema_version") != SCHEMA_VERSION:
            raise PolicyError("schema_version must be 2")
        supplied = p.get("semantic_hash")
        allowed_top = {"schema_version", "name", "semantic_hash", "blackboard", "subtrees", "tree"}
        extra = set(p) - allowed_top
        if extra: raise PolicyError(f"unknown policy fields: {sorted(extra)}")
        self._compile_blackboard(p.get("blackboard", []))
        self._load_subtrees(p.get("subtrees", []))
        if "tree" not in p: raise PolicyError("missing tree")
        root = self._compile_node(self._expand(p["tree"], {}, ()), 1)
        if not self.nodes: raise PolicyError("empty policy")
        if len(self.nodes) > MAX_NODES or len(self.child_refs) > MAX_CHILD_REFS: raise PolicyError("compiled table limit exceeded")
        budget = max(64, min(MAX_VISITS, len(self.nodes) * 8 + len(self.child_refs) * 2))
        compiled = CompiledPolicy("", root, self.nodes, self.child_refs, self.bb, self.values,
                                  self.predicates, self.lifecycles, self.effects, self.state_count,
                                  budget, BOARD_SIZE, copy.deepcopy(p))
        validate_compiled(compiled)
        compiled.semantic_hash = hashlib.sha256(semantic_payload_bytes(compiled)).hexdigest()
        if supplied is not None and supplied != compiled.semantic_hash:
            raise PolicyError("semantic_hash does not match normalized executable policy")
        return compiled

    def _compile_blackboard(self, decls: Any) -> None:
        if not isinstance(decls, list) or len(decls) > MAX_BB: raise PolicyError("blackboard must be list with <=32 slots")
        for i, d in enumerate(decls):
            if not isinstance(d, dict): raise PolicyError("blackboard declaration must be object")
            if set(d) - {"name", "type", "initial"}: raise PolicyError("unknown blackboard declaration field")
            name = d.get("name")
            typ = d.get("type")
            if not isinstance(name, str) or not name or not name[0].isalpha() or not all(c.isalnum() or c == "_" for c in name):
                raise PolicyError("invalid blackboard slot name")
            if name in self.bb_by_name: raise PolicyError(f"duplicate blackboard slot {name}")
            if typ not in BB_TYPES: raise PolicyError(f"unknown blackboard type {typ}")
            initial = _fit_i32(d.get("initial", 0), f"blackboard {name} initial")
            _validate_bb_value(BB_TYPES[typ], initial, f"blackboard {name} initial")
            self.bb_by_name[name] = i
            self.bb.append(BbDecl(BB_TYPES[typ], initial, name))

    def _load_subtrees(self, defs: Any) -> None:
        if not isinstance(defs, list):
            raise PolicyError("subtrees must be a list")
        for d in defs:
            if not isinstance(d, dict) or set(d) != {"name", "parameters", "tree"}:
                raise PolicyError("subtree definition requires name/parameters/tree")
            name, params = d["name"], d["parameters"]
            if not isinstance(name, str) or not name or name in self.subtrees:
                raise PolicyError("invalid or duplicate subtree name")
            if not isinstance(params, dict) or any(
                not isinstance(k, str) or not k or v not in VALUE_TYPES for k, v in params.items()
            ):
                raise PolicyError(f"invalid parameters for subtree {name}")
            self.subtrees[name] = copy.deepcopy(d)

        graph: Dict[str, set[str]] = {name: set() for name in self.subtrees}
        def calls(obj: Any, out: set[str]) -> None:
            if isinstance(obj, dict):
                if obj.get("type") == "Subtree":
                    target = obj.get("subtree")
                    if target not in self.subtrees: raise PolicyError(f"unknown subtree {target}")
                    out.add(target)
                for value in obj.values(): calls(value, out)
            elif isinstance(obj, list):
                for value in obj: calls(value, out)
        for name, d in self.subtrees.items(): calls(d["tree"], graph[name])
        color: Dict[str, int] = {name: 0 for name in graph}
        def visit(name: str) -> None:
            if color[name] == 1: raise PolicyError(f"recursive subtree {name}")
            if color[name] == 2: return
            color[name] = 1
            for target in graph[name]: visit(target)
            color[name] = 2
        for name in graph: visit(name)

    def _expand(self, obj: Any, env: Dict[str, Any], stack: Tuple[str, ...]) -> Any:
        if isinstance(obj, list):
            return [self._expand(x, env, stack) for x in obj]
        if not isinstance(obj, dict):
            return copy.deepcopy(obj)
        if set(obj) == {"binding"}:
            name = obj["binding"]
            if name not in env: raise PolicyError(f"unknown subtree binding {name}")
            return copy.deepcopy(env[name])
        if obj.get("type") == "Subtree":
            if set(obj) - {"type", "name", "subtree", "bindings"}:
                raise PolicyError("unknown Subtree invocation field")
            name = obj.get("subtree")
            if name not in self.subtrees: raise PolicyError(f"unknown subtree {name}")
            if name in stack: raise PolicyError(f"recursive subtree {name}")
            definition = self.subtrees[name]
            params = definition["parameters"]
            bindings = obj.get("bindings")
            if not isinstance(bindings, dict) or set(bindings) != set(params):
                raise PolicyError(f"subtree {name} bindings must exactly match parameters")
            resolved = {key: self._expand(value, env, stack) for key, value in bindings.items()}
            for key, typ in params.items(): self._make_value_ref(resolved[key], VALUE_TYPES[typ], False)
            return self._expand(definition["tree"], resolved, stack + (name,))
        return {key: self._expand(value, env, stack) for key, value in obj.items()}

    def _alloc_state(self) -> int:
        if self.state_count >= MAX_STATE: raise PolicyError("state slot limit exceeded")
        out = self.state_count
        self.state_count += 1
        return out

    def _make_value_ref(self, obj: Any, expected_type: Optional[int] = None, allocate: bool = True) -> ValueRef:
        if isinstance(obj, (int, bool)):
            typ = expected_type if expected_type is not None else BB_TYPES["I32"]
            value = _fit_i32(obj, "constant ValueRef")
            _validate_bb_value(typ, value, "constant ValueRef")
            ref = ValueRef(VALUE_KINDS["CONST"], typ, 0, 0, value)
        elif isinstance(obj, dict) and "const" in obj:
            if set(obj) - {"const", "type"}: raise PolicyError("invalid CONST ValueRef fields")
            type_name = obj.get("type")
            typ = VALUE_TYPES.get(type_name) if type_name is not None else expected_type
            if typ not in VALUE_TYPES.values(): raise PolicyError("CONST ValueRef requires I32/TILE/UNIT type")
            value = _fit_i32(obj["const"], "constant ValueRef")
            _validate_bb_value(typ, value, "constant ValueRef")
            ref = ValueRef(VALUE_KINDS["CONST"], typ, 0, 0, value)
        elif isinstance(obj, dict) and "blackboard" in obj:
            if set(obj) != {"blackboard"}: raise PolicyError("invalid BB ValueRef fields")
            name = obj["blackboard"]
            if name not in self.bb_by_name: raise PolicyError(f"unknown blackboard slot {name}")
            slot = self.bb_by_name[name]
            declared = self.bb[slot].type
            typ = declared if declared in (BB_TYPES["TILE"], BB_TYPES["UNIT"]) else BB_TYPES["I32"]
            ref = ValueRef(VALUE_KINDS["BB"], typ, slot, 0, 0)
        elif isinstance(obj, dict) and "feature" in obj:
            if set(obj) - {"feature", "arg"}: raise PolicyError("invalid LEGAL_FEATURE ValueRef fields")
            name = obj["feature"]
            if name not in FEATURES: raise PolicyError(f"illegal/unknown feature {name}")
            arg = _fit_i32(obj.get("arg", 0), "feature arg")
            fid = FEATURES[name]
            if not _feature_arg_ok(fid, arg): raise PolicyError(f"feature arg out of range for {name}: {arg}")
            ref = ValueRef(VALUE_KINDS["LEGAL_FEATURE"], FEATURE_TYPES[fid], fid, arg, 0)
        else:
            raise PolicyError("ValueRef must be CONST, BB, or LEGAL_FEATURE")
        if expected_type is not None and ref.type != expected_type:
            raise PolicyError(f"ValueRef type mismatch: expected {BB_NAMES[expected_type]}, got {BB_NAMES[ref.type]}")
        if allocate:
            if len(self.values) >= MAX_VALUE_REFS: raise PolicyError("ValueRef limit exceeded")
            self.values.append(ref)
        return ref

    def _compile_value_ref(self, obj: Any, expected_type: Optional[int] = None) -> int:
        self._make_value_ref(obj, expected_type, True)
        return len(self.values) - 1

    def _compile_predicate(self, obj: Any) -> int:
        if len(self.predicates) >= MAX_PRED: raise PolicyError("predicate limit exceeded")
        if not isinstance(obj, dict) or set(obj) != {"lhs", "op", "rhs"}: raise PolicyError("predicate requires lhs/op/rhs")
        op = obj["op"]
        if op not in CMP: raise PolicyError(f"invalid comparison {op}")
        lhs = self._compile_value_ref(obj["lhs"])
        rhs = self._compile_value_ref(obj["rhs"], self.values[lhs].type)
        pred = Predicate(lhs, rhs, CMP[op])
        self.predicates.append(pred)
        return len(self.predicates) - 1

    def _compile_effects(self, values: Any, label: str) -> Tuple[int, int]:
        if values is None: return (0, 0)
        if not isinstance(values, list): raise PolicyError(f"{label} must be effect list")
        if not values: return (0, 0)
        start = len(self.effects)
        for e in values:
            if len(self.effects) >= MAX_EFFECT: raise PolicyError("effect limit exceeded")
            if not isinstance(e, dict) or set(e) - {"op", "slot", "value"}: raise PolicyError("invalid effect")
            op = e.get("op")
            slot_name = e.get("slot")
            if op not in EFFECT_OP or slot_name not in self.bb_by_name: raise PolicyError("invalid effect op/slot")
            slot = self.bb_by_name[slot_name]
            if op == "add" and self.bb[slot].type != BB_TYPES["I32"]: raise PolicyError("add is only valid for I32")
            if op == "set" and "value" not in e: raise PolicyError("set effect requires value")
            if op == "clear" and "value" in e: raise PolicyError("clear effect has no value")
            if op == "add" and "value" not in e: raise PolicyError("add effect requires value")
            if op == "clear":
                value_ref = 0
            else:
                declared = self.bb[slot].type
                expected = declared if declared in (BB_TYPES["TILE"], BB_TYPES["UNIT"]) else BB_TYPES["I32"]
                value_ref = self._compile_value_ref(e["value"], expected)
                ref = self.values[value_ref]
                if ref.kind == VALUE_KINDS["CONST"]:
                    _validate_bb_value(declared, ref.value, f"effect {op} {slot_name}")
            self.effects.append(Effect(EFFECT_OP[op], slot, value_ref))
        return (start, len(values))

    def _compile_lifecycle(self, obj: Any) -> int:
        if len(self.lifecycles) >= MAX_LIFECYCLE: raise PolicyError("lifecycle limit exceeded")
        if not isinstance(obj, dict): raise PolicyError("lifecycle must be object")
        allowed = {"failure_if", "success_if", "skip_if", "while", "on_success", "on_failure", "on_halted", "post"}
        extra = set(obj) - allowed
        if extra: raise PolicyError(f"unknown lifecycle fields {sorted(extra)}")
        def p(name: str) -> int:
            return self._compile_predicate(obj[name]) if name in obj else NONE_U16
        lc = Lifecycle(
            p("failure_if"), p("success_if"), p("skip_if"), p("while"),
            self._compile_effects(obj.get("on_success"), "on_success"),
            self._compile_effects(obj.get("on_failure"), "on_failure"),
            self._compile_effects(obj.get("on_halted"), "on_halted"),
            self._compile_effects(obj.get("post"), "post"),
        )
        self.lifecycles.append(lc)
        return len(self.lifecycles) - 1

    def _node_children(self, typ: str, obj: Dict[str, Any]) -> List[Any]:
        composites = {"Sequence", "Selector", "ReactiveSequence", "ReactiveSelector", "Parallel"}
        if typ in composites:
            children = obj.get("children")
            if not isinstance(children, list) or not children: raise PolicyError(f"{typ} requires non-empty children")
            if typ == "Parallel" and not 2 <= len(children) <= MAX_PARALLEL_CHILDREN: raise PolicyError("Parallel requires 2..16 children")
            return children
        if typ == "If":
            if not all(k in obj for k in ("condition", "then", "else")): raise PolicyError("If requires condition/then/else")
            return [obj["condition"], obj["then"], obj["else"]]
        if NODE_TYPES[typ] in DECORATOR_TYPES:
            if "child" not in obj: raise PolicyError(f"{typ} requires child")
            return [obj["child"]]
        return []

    def _compile_node(self, obj: Any, depth: int) -> int:
        if depth > MAX_DEPTH: raise PolicyError("tree depth exceeds 24")
        if len(self.nodes) >= MAX_NODES: raise PolicyError("node limit exceeds 128")
        if not isinstance(obj, dict): raise PolicyError("node must be object")
        typ = obj.get("type")
        if typ not in NODE_TYPES: raise PolicyError(f"unknown/non-portable node type {typ}")
        allowed_common = {"type", "name", "lifecycle"}
        type_fields = {
            "Sequence": {"children"}, "Selector": {"children"}, "ReactiveSequence": {"children"}, "ReactiveSelector": {"children"},
            "Parallel": {"children", "success_threshold", "failure_threshold"},
            "If": {"condition", "then", "else"},
            "Invert": {"child"}, "ForceSuccess": {"child"}, "ForceFailure": {"child"},
            "TurnTimeout": {"child", "turns"}, "Retry": {"child", "n"}, "Repeat": {"child", "n"},
            "KeepRunningUntilFailure": {"child"}, "Success": set(), "Failure": set(),
            "Condition": {"lhs", "op", "rhs"}, "SetBlackboard": {"slot", "value"},
            "AddBlackboard": {"slot", "value"},
            "EmitUnitIntent": {"unit", "action", "item", "quantity", "priority"},
            "EmitMarketIntent": {"action", "item", "quantity", "priority"},
            "WaitTurns": {"turns"}, "MoveUnitTo": {"unit", "target", "x", "y", "priority"},
        }
        extra = set(obj) - allowed_common - type_fields[typ]
        if extra: raise PolicyError(f"unknown fields for {typ}: {sorted(extra)}")
        idx = len(self.nodes)
        node = Node(type=NODE_TYPES[typ], debug_id=idx)
        self.nodes.append(node)
        children = self._node_children(typ, obj)
        child_ids = [self._compile_node(c, depth + 1) for c in children]
        if len(self.child_refs) + len(child_ids) > MAX_CHILD_REFS: raise PolicyError("child reference limit exceeded")
        node.first_child = len(self.child_refs) if child_ids else 0
        node.child_count = len(child_ids)
        self.child_refs.extend(child_ids)
        lifecycle_obj = obj.get("lifecycle")
        if node.type in STATEFUL_TYPES or lifecycle_obj is not None:
            node.state_slot = self._alloc_state()
        if lifecycle_obj is not None:
            node.lifecycle_slot = self._compile_lifecycle(lifecycle_obj)
        if typ == "Parallel":
            s = _fit_i32(obj.get("success_threshold", len(children)), "parallel success_threshold")
            f = _fit_i32(obj.get("failure_threshold", 1), "parallel failure_threshold")
            if not 1 <= s <= len(children) or not 1 <= f <= len(children): raise PolicyError("parallel threshold out of range")
            node.p0, node.p1 = s, f
        elif typ == "TurnTimeout":
            n = _fit_i32(obj.get("turns"), "timeout turns")
            if not 1 <= n <= 720: raise PolicyError("TurnTimeout turns must be 1..720")
            node.p0 = n
        elif typ in ("Retry", "Repeat"):
            n = _fit_i32(obj.get("n"), f"{typ} n")
            if not 1 <= n <= 255: raise PolicyError(f"{typ} n must be 1..255")
            node.p0 = n
        elif typ == "Condition":
            pred = self._compile_predicate({"lhs": obj.get("lhs"), "op": obj.get("op"), "rhs": obj.get("rhs")})
            p = self.predicates.pop()
            assert pred == len(self.predicates)
            node.flags, node.p0, node.p1 = p.cmp, p.lhs, p.rhs
        elif typ in ("SetBlackboard", "AddBlackboard"):
            name = obj.get("slot")
            if name not in self.bb_by_name: raise PolicyError(f"unknown blackboard slot {name}")
            slot = self.bb_by_name[name]
            declared = self.bb[slot].type
            if typ == "AddBlackboard" and declared != BB_TYPES["I32"]: raise PolicyError("AddBlackboard requires I32 slot")
            expected = declared if declared in (BB_TYPES["TILE"], BB_TYPES["UNIT"]) else BB_TYPES["I32"]
            value_ref = self._compile_value_ref(obj.get("value"), expected)
            ref = self.values[value_ref]
            if typ == "SetBlackboard" and ref.kind == VALUE_KINDS["CONST"]:
                _validate_bb_value(declared, ref.value, f"{typ} {name}")
            node.p0, node.p1 = slot, value_ref
        elif typ == "EmitUnitIntent":
            unit = _fit_i32(obj.get("unit", 0), "unit")
            if not 0 <= unit < MAX_UNITS: raise PolicyError("unit out of range")
            act = obj.get("action")
            if act not in UNIT_OPS: raise PolicyError(f"invalid unit action {act}")
            item = self._item_value(obj.get("item"), allow_none=True)
            qty = _fit_i32(obj.get("quantity", 1), "unit quantity")
            pri = _fit_i32(obj.get("priority", 0), "priority")
            if not 0 <= pri <= 255: raise PolicyError("unit priority out of range")
            if act in ("PICKUP", "PLACE"):
                if not 1 <= qty <= 32767: raise PolicyError("unit quantity out of range")
            else:
                if "quantity" in obj and qty != 1: raise PolicyError(f"{act} quantity is semantically ignored and must be 1")
                qty = 1
            if act == "PLANT" and not 0 <= item < N_CROPS: raise PolicyError("PLANT requires crop item")
            if act in ("PICKUP", "PLACE") and not 0 <= item < len(ITEMS): raise PolicyError(f"{act} requires item")
            if act not in ("PLANT", "PICKUP", "PLACE") and "item" in obj: raise PolicyError(f"{act} does not accept item")
            node.flags, node.p0, node.p1, node.p2, node.p3 = pri, unit, UNIT_OPS[act], max(item, 0), qty
        elif typ == "EmitMarketIntent":
            act = obj.get("action")
            if act not in MARKET_OPS: raise PolicyError(f"invalid market action {act}")
            item = self._item_value(obj.get("item"), allow_none=True)
            qty = _fit_i32(obj.get("quantity", 1), "market quantity")
            pri = _fit_i32(obj.get("priority", 0), "priority")
            if not 0 <= pri <= 255 or not 1 <= qty <= 65535: raise PolicyError("market priority/quantity out of range")
            if act in ("HIRE", "BUY_LAND"):
                if "item" in obj or ("quantity" in obj and qty != 1): raise PolicyError(f"{act} takes no item/quantity")
                item, qty = 0, 1
            elif act == "BUY_SEED" and not 0 <= item < N_CROPS: raise PolicyError("BUY_SEED requires crop")
            elif act == "BUY_PRODUCT" and item not in (ITEM_ID["WHEAT"], ITEM_ID["FERTILIZER"]): raise PolicyError("BUY_PRODUCT supports WHEAT/FERTILIZER")
            elif act == "BUY_ANIMAL" and not ITEM_ID["GOOSE"] <= item <= ITEM_ID["SHEEP"]: raise PolicyError("BUY_ANIMAL requires animal")
            elif act == "SELL" and not 0 <= item < N_PRODUCTS: raise PolicyError("SELL requires product")
            node.p0, node.p1, node.p2, node.p3 = MARKET_OPS[act], max(item, 0), qty, pri
        elif typ == "WaitTurns":
            n = _fit_i32(obj.get("turns"), "wait turns")
            if not 1 <= n <= 720: raise PolicyError("WaitTurns turns must be 1..720")
            node.p0 = n
        elif typ == "MoveUnitTo":
            unit = self._compile_value_ref(obj.get("unit", 0), BB_TYPES["UNIT"])
            if "target" in obj:
                if "x" in obj or "y" in obj: raise PolicyError("MoveUnitTo uses target or x/y, not both")
                target = self._compile_value_ref(obj["target"], BB_TYPES["TILE"])
            else:
                if "x" not in obj or "y" not in obj: raise PolicyError("MoveUnitTo requires target or x/y")
                x, y = _fit_i32(obj["x"], "move x"), _fit_i32(obj["y"], "move y")
                if not 0 <= x < BOARD_SIZE or not 0 <= y < BOARD_SIZE: raise PolicyError("MoveUnitTo x/y out of range")
                target = self._compile_value_ref({"const": y * BOARD_SIZE + x, "type": "TILE"}, BB_TYPES["TILE"])
            pri = _fit_i32(obj.get("priority", 0), "move priority")
            if not 0 <= pri <= 255:
                raise PolicyError("MoveUnitTo parameter out of range")
            node.p0, node.p1, node.p2 = unit, target, pri
        return idx

    @staticmethod
    def _item_value(value: Any, allow_none: bool = False) -> int:
        if value is None and allow_none: return -1
        if isinstance(value, str):
            if value not in ITEM_ID: raise PolicyError(f"unknown item {value}")
            return ITEM_ID[value]
        if isinstance(value, int) and 0 <= value < len(ITEMS): return value
        raise PolicyError(f"invalid item {value}")


def compile_policy(policy: Dict[str, Any]) -> CompiledPolicy:
    return Compiler(policy).compile()


def load_policy_json(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _static_max_intents(policy: CompiledPolicy) -> int:
    emitting = {NODE_TYPES["EmitUnitIntent"], NODE_TYPES["EmitMarketIntent"], NODE_TYPES["MoveUnitTo"]}
    return sum(1 for node in policy.nodes if node.type in emitting)


def _compiled_image(policy: CompiledPolicy, digest: bytes) -> bytes:
    data = bytearray(HEADER.pack(MAGIC, ABI_VERSION, HEADER.size, digest, policy.root,
                                 len(policy.nodes), len(policy.child_refs), policy.state_count,
                                 len(policy.bb), len(policy.values), len(policy.predicates),
                                 len(policy.lifecycles), len(policy.effects), policy.instruction_budget,
                                 policy.board_size, _static_max_intents(policy)))
    for n in policy.nodes:
        data += NODE_STRUCT.pack(n.type, n.flags, n.first_child, n.child_count, n.state_slot,
                                 n.lifecycle_slot, b"\0\0", n.p0, n.p1, n.p2, n.p3, n.debug_id)
    for c in policy.child_refs: data += struct.pack("<H", c)
    for b in policy.bb: data += BB_STRUCT.pack(b.type, b"\0\0\0", b.initial)
    for v in policy.values: data += VALUE_STRUCT.pack(v.kind, v.type, v.id, v.arg, v.value)
    for p in policy.predicates: data += PRED_STRUCT.pack(p.lhs, p.rhs, p.cmp, b"\0" * 7)
    for lc in policy.lifecycles:
        data += LIFECYCLE_STRUCT.pack(
            lc.failure_if, lc.success_if, lc.skip_if, lc.while_pred,
            lc.on_success[0], lc.on_success[1], lc.on_failure[0], lc.on_failure[1],
            lc.on_halted[0], lc.on_halted[1], lc.post[0], lc.post[1])
    for e in policy.effects: data += EFFECT_STRUCT.pack(e.op, e.slot, 0, e.value)
    return bytes(data)


def semantic_payload_bytes(policy: CompiledPolicy) -> bytes:
    """Normalized executable identity stream; excludes source/debug/padding bytes."""
    data = bytearray(b"BKG2SEM\0")
    def u8(v: int) -> None: data.extend(struct.pack("<B", v))
    def u16(v: int) -> None: data.extend(struct.pack("<H", v))
    def i32(v: int) -> None: data.extend(struct.pack("<i", v))
    for v in (
        ABI_VERSION, policy.board_size, policy.root, len(policy.nodes), len(policy.child_refs),
        policy.state_count, len(policy.bb), len(policy.values), len(policy.predicates),
        len(policy.lifecycles), len(policy.effects), policy.instruction_budget, _static_max_intents(policy),
    ):
        u16(v)
    for n in policy.nodes:
        u8(n.type); u8(n.flags)
        for v in (n.first_child, n.child_count, n.state_slot, n.lifecycle_slot): u16(v)
        for v in (n.p0, n.p1, n.p2, n.p3): i32(v)
    for c in policy.child_refs: u16(c)
    for b in policy.bb: u8(b.type); i32(b.initial)
    for v in policy.values:
        u8(v.kind); u8(v.type); u16(v.id); i32(v.arg); i32(v.value)
    for q in policy.predicates:
        u16(q.lhs); u16(q.rhs); u8(q.cmp)
    for lc in policy.lifecycles:
        for v in (
            lc.failure_if, lc.success_if, lc.skip_if, lc.while_pred,
            lc.on_success[0], lc.on_success[1], lc.on_failure[0], lc.on_failure[1],
            lc.on_halted[0], lc.on_halted[1], lc.post[0], lc.post[1],
        ): u16(v)
    for e in policy.effects:
        u8(e.op); u8(e.slot); i32(e.value)
    return bytes(data)


def save_compiled(policy: CompiledPolicy, path: str | Path) -> None:
    digest = hashlib.sha256(semantic_payload_bytes(policy)).digest()
    policy.semantic_hash = digest.hex()
    data = _compiled_image(policy, digest)
    Path(path).write_bytes(data)


def load_compiled(path: str | Path) -> CompiledPolicy:
    raw = Path(path).read_bytes()
    if len(raw) < HEADER.size: raise PolicyError("compiled file too short")
    fields = HEADER.unpack_from(raw, 0)
    magic, version, hsize, digest = fields[:4]
    if magic != MAGIC or version != ABI_VERSION or hsize != HEADER.size: raise PolicyError("invalid compiled header")
    root, nn, nc, ns, nb, nv, np, nl, ne, budget, board_size, max_intents = fields[4:]
    if max_intents > MAX_DEFERRED_INTENTS: raise PolicyError("compiled max-intents bound exceeds ABI limit")
    if nn > MAX_NODES or nc > MAX_CHILD_REFS or ns > MAX_STATE or nb > MAX_BB or nv > MAX_VALUE_REFS or np > MAX_PRED or nl > MAX_LIFECYCLE or ne > MAX_EFFECT:
        raise PolicyError("compiled counts exceed ABI limits")
    expected = HEADER.size + nn * NODE_STRUCT.size + nc * 2 + nb * BB_STRUCT.size + nv * VALUE_STRUCT.size + np * PRED_STRUCT.size + nl * LIFECYCLE_STRUCT.size + ne * EFFECT_STRUCT.size
    if len(raw) != expected: raise PolicyError("compiled file size/count mismatch")
    off = HEADER.size
    nodes = []
    for _ in range(nn):
        x = NODE_STRUCT.unpack_from(raw, off); off += NODE_STRUCT.size
        if x[6] != b"\0\0": raise PolicyError("compiled Node padding is nonzero")
        nodes.append(Node(*x[:6], *x[7:]))
    child_refs = list(struct.unpack_from(f"<{nc}H", raw, off)) if nc else []; off += nc * 2
    bb = []
    for _ in range(nb):
        t, pad, init = BB_STRUCT.unpack_from(raw, off); off += BB_STRUCT.size
        if pad != b"\0\0\0": raise PolicyError("compiled BbDecl padding is nonzero")
        bb.append(BbDecl(t, init))
    values = []
    for _ in range(nv):
        values.append(ValueRef(*VALUE_STRUCT.unpack_from(raw, off))); off += VALUE_STRUCT.size
    preds = []
    for _ in range(np):
        lhs, rhs, cmp_id, pad = PRED_STRUCT.unpack_from(raw, off); off += PRED_STRUCT.size
        if pad != b"\0" * 7: raise PolicyError("compiled Predicate padding is nonzero")
        preds.append(Predicate(lhs, rhs, cmp_id))
    lcs = []
    for _ in range(nl):
        x = LIFECYCLE_STRUCT.unpack_from(raw, off); off += LIFECYCLE_STRUCT.size
        lcs.append(Lifecycle(x[0], x[1], x[2], x[3], (x[4], x[5]), (x[6], x[7]), (x[8], x[9]), (x[10], x[11])))
    effects = []
    for _ in range(ne):
        op, slot, effect_reserved, value = EFFECT_STRUCT.unpack_from(raw, off); off += EFFECT_STRUCT.size
        if effect_reserved != 0: raise PolicyError("compiled Effect reserved field is nonzero")
        effects.append(Effect(op, slot, value))
    cp = CompiledPolicy(digest.hex(), root, nodes, child_refs, bb, values, preds, lcs, effects, ns, budget, board_size)
    validate_compiled(cp)
    if max_intents != _static_max_intents(cp): raise PolicyError("compiled max-intents bound is noncanonical")
    actual = hashlib.sha256(semantic_payload_bytes(cp)).digest()
    if actual != digest: raise PolicyError("compiled semantic hash mismatch")
    return cp


def validate_compiled(p: CompiledPolicy) -> None:
    if not p.nodes or not 0 <= p.root < len(p.nodes): raise PolicyError("compiled root invalid")
    if p.board_size != BOARD_SIZE: raise PolicyError("compiled board size must be 10")
    if len(p.nodes) > MAX_NODES or len(p.child_refs) > MAX_CHILD_REFS or p.state_count > MAX_STATE:
        raise PolicyError("compiled node/child/state count exceeds ABI limits")
    if len(p.bb) > MAX_BB or len(p.values) > MAX_VALUE_REFS or len(p.predicates) > MAX_PRED or len(p.lifecycles) > MAX_LIFECYCLE or len(p.effects) > MAX_EFFECT:
        raise PolicyError("compiled metadata count exceeds ABI limits")
    if not 1 <= p.instruction_budget <= MAX_VISITS: raise PolicyError("compiled instruction budget invalid")

    for i, b in enumerate(p.bb):
        if b.type not in BB_NAMES: raise PolicyError(f"compiled blackboard {i} type invalid")
        _validate_bb_value(b.type, b.initial, f"compiled blackboard {i} initial")

    for i, v in enumerate(p.values):
        if v.kind not in VALUE_KINDS.values() or v.type not in VALUE_TYPES.values():
            raise PolicyError(f"compiled ValueRef {i} kind/type invalid")
        if v.kind == VALUE_KINDS["CONST"]:
            if v.id != 0 or v.arg != 0: raise PolicyError(f"compiled CONST ValueRef {i} noncanonical")
            _validate_bb_value(v.type, v.value, f"compiled CONST ValueRef {i}")
        elif v.kind == VALUE_KINDS["BB"]:
            if not 0 <= v.id < len(p.bb) or v.arg != 0 or v.value != 0:
                raise PolicyError(f"compiled BB ValueRef {i} noncanonical")
            declared = p.bb[v.id].type
            effective = declared if declared in (BB_TYPES["TILE"], BB_TYPES["UNIT"]) else BB_TYPES["I32"]
            if v.type != effective: raise PolicyError(f"compiled BB ValueRef {i} type mismatch")
        else:
            if v.id not in FEATURE_NAMES or not _feature_arg_ok(v.id, v.arg) or v.value != 0:
                raise PolicyError(f"compiled LEGAL_FEATURE ValueRef {i} invalid")
            if v.type != FEATURE_TYPES[v.id]: raise PolicyError(f"compiled LEGAL_FEATURE ValueRef {i} type mismatch")

    for i, q in enumerate(p.predicates):
        if q.cmp not in CMP_NAMES or not 0 <= q.lhs < len(p.values) or not 0 <= q.rhs < len(p.values):
            raise PolicyError(f"compiled predicate {i} encoding invalid")
        if p.values[q.lhs].type != p.values[q.rhs].type:
            raise PolicyError(f"compiled predicate {i} type mismatch")

    effect_value_refs: set[int] = set()
    for i, e in enumerate(p.effects):
        if e.op not in EFFECT_OP.values() or not 0 <= e.slot < len(p.bb): raise PolicyError(f"compiled effect {i} invalid")
        if e.op == EFFECT_OP["clear"]:
            if e.value != 0: raise PolicyError(f"compiled effect {i} CLEAR value must be zero")
            continue
        if not 0 <= e.value < len(p.values): raise PolicyError(f"compiled effect {i} ValueRef invalid")
        declared = p.bb[e.slot].type
        expected = declared if declared in (BB_TYPES["TILE"], BB_TYPES["UNIT"]) else BB_TYPES["I32"]
        if p.values[e.value].type != expected: raise PolicyError(f"compiled effect {i} ValueRef type mismatch")
        if e.op == EFFECT_OP["add"] and declared != BB_TYPES["I32"]:
            raise PolicyError(f"compiled effect {i} ADD requires I32")
        effect_value_refs.add(e.value)

    used_predicates: set[int] = set()
    used_effects: set[int] = set()
    for i, lc in enumerate(p.lifecycles):
        for q in (lc.failure_if, lc.success_if, lc.skip_if, lc.while_pred):
            if q != NONE_U16 and not 0 <= q < len(p.predicates): raise PolicyError(f"compiled lifecycle {i} predicate invalid")
            if q != NONE_U16:
                if q in used_predicates: raise PolicyError("compiled lifecycle predicates must not alias")
                used_predicates.add(q)
        for start, count in (lc.on_success, lc.on_failure, lc.on_halted, lc.post):
            if start < 0 or count < 0 or start + count > len(p.effects): raise PolicyError(f"compiled lifecycle {i} effect range invalid")
            if count == 0 and start != 0: raise PolicyError(f"compiled lifecycle {i} empty effect range is noncanonical")
            for effect in range(start, start + count):
                if effect in used_effects: raise PolicyError("compiled lifecycle effects must not alias")
                used_effects.add(effect)
    if used_predicates != set(range(len(p.predicates))): raise PolicyError("compiled predicate table is not dense/used")
    if used_effects != set(range(len(p.effects))): raise PolicyError("compiled effect table is not dense/used")

    parents = [0] * len(p.nodes)
    child_ref_used = [0] * len(p.child_refs)
    used_state: set[int] = set()
    used_lifecycles: set[int] = set()
    used_values: set[int] = {ref for q in p.predicates for ref in (q.lhs, q.rhs)} | effect_value_refs
    leaf_types = set(NODE_TYPES.values()) - {NODE_TYPES[x] for x in ("Sequence", "Selector", "ReactiveSequence", "ReactiveSelector", "If", "Parallel")} - DECORATOR_TYPES
    for i, n in enumerate(p.nodes):
        if n.type not in NODE_NAMES: raise PolicyError(f"compiled node {i} type invalid")
        if n.debug_id != i: raise PolicyError(f"compiled node {i} debug id invalid")
        if n.first_child < 0 or n.child_count < 0 or n.first_child + n.child_count > len(p.child_refs):
            raise PolicyError(f"compiled node {i} child range invalid")
        if n.child_count == 0 and n.first_child != 0: raise PolicyError(f"compiled node {i} empty child range is noncanonical")
        for pos in range(n.first_child, n.first_child + n.child_count):
            if child_ref_used[pos]: raise PolicyError("compiled child-reference ranges overlap")
            child_ref_used[pos] = 1
        children = p.child_refs[n.first_child:n.first_child+n.child_count]
        for c in children:
            if not 0 <= c < len(p.nodes): raise PolicyError(f"compiled node {i} child invalid")
            parents[c] += 1
            if parents[c] > 1: raise PolicyError("compiled structure is not a tree")
        needs_state = n.type in STATEFUL_TYPES or n.lifecycle_slot != NONE_U16
        if needs_state and n.state_slot == NONE_U16: raise PolicyError(f"compiled node {i} missing state slot")
        if not needs_state and n.state_slot != NONE_U16: raise PolicyError(f"compiled node {i} has unnecessary state slot")
        if n.state_slot != NONE_U16:
            if not 0 <= n.state_slot < p.state_count: raise PolicyError(f"compiled node {i} state slot invalid")
            if n.state_slot in used_state: raise PolicyError(f"compiled node {i} aliases a state slot")
            used_state.add(n.state_slot)
        if n.lifecycle_slot != NONE_U16:
            if not 0 <= n.lifecycle_slot < len(p.lifecycles): raise PolicyError(f"compiled node {i} lifecycle invalid")
            if n.lifecycle_slot in used_lifecycles: raise PolicyError("compiled lifecycle descriptors must not alias")
            used_lifecycles.add(n.lifecycle_slot)

        if n.type in (NODE_TYPES["Sequence"], NODE_TYPES["Selector"], NODE_TYPES["ReactiveSequence"], NODE_TYPES["ReactiveSelector"]):
            if n.child_count < 1: raise PolicyError(f"compiled {NODE_NAMES[n.type]} is empty")
        elif n.type == NODE_TYPES["If"]:
            if n.child_count != 3: raise PolicyError("compiled If cardinality invalid")
        elif n.type == NODE_TYPES["Parallel"]:
            if not 2 <= n.child_count <= MAX_PARALLEL_CHILDREN or not 1 <= n.p0 <= n.child_count or not 1 <= n.p1 <= n.child_count:
                raise PolicyError("compiled Parallel parameters invalid")
        elif n.type in DECORATOR_TYPES:
            if n.child_count != 1: raise PolicyError(f"compiled {NODE_NAMES[n.type]} cardinality invalid")
        elif n.type in leaf_types and n.child_count != 0:
            raise PolicyError(f"compiled leaf {NODE_NAMES[n.type]} has children")

        if n.type not in (NODE_TYPES["Condition"], NODE_TYPES["EmitUnitIntent"]) and n.flags != 0:
            raise PolicyError(f"compiled {NODE_NAMES[n.type]} flags must be zero")
        allowed_params: Dict[int, set[int]] = {
            NODE_TYPES["Parallel"]: {0, 1}, NODE_TYPES["TurnTimeout"]: {0}, NODE_TYPES["Retry"]: {0},
            NODE_TYPES["Repeat"]: {0}, NODE_TYPES["Condition"]: {0, 1}, NODE_TYPES["SetBlackboard"]: {0, 1},
            NODE_TYPES["AddBlackboard"]: {0, 1}, NODE_TYPES["EmitUnitIntent"]: {0, 1, 2, 3},
            NODE_TYPES["EmitMarketIntent"]: {0, 1, 2, 3}, NODE_TYPES["WaitTurns"]: {0},
            NODE_TYPES["MoveUnitTo"]: {0, 1, 2},
        }
        params = (n.p0, n.p1, n.p2, n.p3)
        for param_i, value in enumerate(params):
            if param_i not in allowed_params.get(n.type, set()) and value != 0:
                raise PolicyError(f"compiled {NODE_NAMES[n.type]} unused p{param_i} must be zero")

        if n.type in (NODE_TYPES["TurnTimeout"], NODE_TYPES["WaitTurns"]) and not 1 <= n.p0 <= 720:
            raise PolicyError(f"compiled {NODE_NAMES[n.type]} turn bound invalid")
        if n.type in (NODE_TYPES["Retry"], NODE_TYPES["Repeat"]) and not 1 <= n.p0 <= 255:
            raise PolicyError(f"compiled {NODE_NAMES[n.type]} count invalid")
        if n.type == NODE_TYPES["Condition"]:
            if n.flags not in CMP_NAMES or not 0 <= n.p0 < len(p.values) or not 0 <= n.p1 < len(p.values):
                raise PolicyError("compiled Condition encoding invalid")
            if p.values[n.p0].type != p.values[n.p1].type: raise PolicyError("compiled Condition type mismatch")
            used_values.update((n.p0, n.p1))
        if n.type in (NODE_TYPES["SetBlackboard"], NODE_TYPES["AddBlackboard"]):
            if not 0 <= n.p0 < len(p.bb) or not 0 <= n.p1 < len(p.values):
                raise PolicyError(f"compiled {NODE_NAMES[n.type]} operand invalid")
            declared = p.bb[n.p0].type
            expected = declared if declared in (BB_TYPES["TILE"], BB_TYPES["UNIT"]) else BB_TYPES["I32"]
            if p.values[n.p1].type != expected: raise PolicyError(f"compiled {NODE_NAMES[n.type]} ValueRef type mismatch")
            if n.type == NODE_TYPES["AddBlackboard"] and declared != BB_TYPES["I32"]: raise PolicyError("compiled AddBlackboard requires I32")
            used_values.add(n.p1)
        if n.type == NODE_TYPES["EmitUnitIntent"]:
            if not 0 <= n.p0 < MAX_UNITS or n.p1 not in UNIT_OPS.values() or not 1 <= n.p3 <= 32767:
                raise PolicyError("compiled EmitUnitIntent parameters invalid")
            if n.p1 == UNIT_OPS["PLANT"] and not 0 <= n.p2 < N_CROPS: raise PolicyError("compiled PLANT item invalid")
            if n.p1 in (UNIT_OPS["PICKUP"], UNIT_OPS["PLACE"]) and not 0 <= n.p2 < len(ITEMS): raise PolicyError("compiled unit item invalid")
            if n.p1 not in (UNIT_OPS["PLANT"], UNIT_OPS["PICKUP"], UNIT_OPS["PLACE"]) and n.p2 != 0:
                raise PolicyError("compiled no-item unit action item must be zero")
            if n.p1 not in (UNIT_OPS["PICKUP"], UNIT_OPS["PLACE"]) and n.p3 != 1:
                raise PolicyError("compiled unit action ignored quantity must be canonical 1")
        if n.type == NODE_TYPES["EmitMarketIntent"]:
            if n.p0 not in MARKET_OPS.values() or not 1 <= n.p2 <= 65535 or not 0 <= n.p3 <= 255: raise PolicyError("compiled EmitMarketIntent parameters invalid")
            if n.p0 in (MARKET_OPS["HIRE"], MARKET_OPS["BUY_LAND"]) and (n.p1 != 0 or n.p2 != 1): raise PolicyError("compiled no-arg market order invalid")
            if n.p0 == MARKET_OPS["BUY_SEED"] and not 0 <= n.p1 < N_CROPS: raise PolicyError("compiled BUY_SEED item invalid")
            if n.p0 == MARKET_OPS["BUY_PRODUCT"] and n.p1 not in (ITEM_ID["WHEAT"], ITEM_ID["FERTILIZER"]): raise PolicyError("compiled BUY_PRODUCT item invalid")
            if n.p0 == MARKET_OPS["BUY_ANIMAL"] and not ITEM_ID["GOOSE"] <= n.p1 <= ITEM_ID["SHEEP"]: raise PolicyError("compiled BUY_ANIMAL item invalid")
            if n.p0 == MARKET_OPS["SELL"] and not 0 <= n.p1 < N_PRODUCTS: raise PolicyError("compiled SELL item invalid")
        if n.type == NODE_TYPES["MoveUnitTo"]:
            if not 0 <= n.p0 < len(p.values) or p.values[n.p0].type != BB_TYPES["UNIT"]:
                raise PolicyError("compiled MoveUnitTo unit ValueRef invalid")
            if not 0 <= n.p1 < len(p.values) or p.values[n.p1].type != BB_TYPES["TILE"] or not 0 <= n.p2 <= 255:
                raise PolicyError("compiled MoveUnitTo target/priority invalid")
            used_values.update((n.p0, n.p1))

    if child_ref_used != [1] * len(p.child_refs): raise PolicyError("compiled child-reference table is not dense/used")
    if used_lifecycles != set(range(len(p.lifecycles))): raise PolicyError("compiled lifecycle table is not dense/used")
    if len(used_state) != p.state_count or used_state != set(range(p.state_count)):
        raise PolicyError("compiled state slots are not dense/unique")
    if used_values != set(range(len(p.values))): raise PolicyError("compiled ValueRef table is not dense/used")
    if parents[p.root] != 0: raise PolicyError("compiled root has a parent")
    for i, count in enumerate(parents):
        if i != p.root and count != 1: raise PolicyError(f"compiled node {i} unreachable or parentless")

    color = [0] * len(p.nodes)
    stack: List[Tuple[int, int, int]] = [(p.root, 0, 1)]
    color[p.root] = 1
    while stack:
        node_id, next_child, depth = stack[-1]
        if depth > MAX_DEPTH: raise PolicyError("compiled depth invalid")
        n = p.nodes[node_id]
        children = p.child_refs[n.first_child:n.first_child+n.child_count]
        if next_child < len(children):
            child_id = children[next_child]
            stack[-1] = (node_id, next_child + 1, depth)
            if color[child_id] == 1: raise PolicyError("compiled cycle detected")
            if color[child_id] == 0:
                color[child_id] = 1
                stack.append((child_id, 0, depth + 1))
        else:
            color[node_id] = 2
            stack.pop()
    if any(c == 0 for c in color): raise PolicyError("compiled unreachable node")


class VM:
    def __init__(self, policy: CompiledPolicy):
        validate_compiled(policy)
        self.p = policy
        self.state = self.reset()
        self._trace: Optional[List[Tuple[int, int, str, str]]] = None
        self._obs: Any = None

    def reset(self) -> RuntimeState:
        self.state = RuntimeState([b.initial for b in self.p.bb], [NodeState() for _ in range(self.p.state_count)])
        return self.state

    def tick(self, observation: Any, max_market_orders: int = 10, trace: bool = False) -> TickResult:
        self._obs = observation if hasattr(observation, "feature") else DictObservation(observation)
        s = self.state
        s.intents = []
        s.unit_claims = [None] * MAX_UNITS
        s.market_claims = []
        s.emit_seq = 0
        s.visits = 0
        self._trace = [] if trace else None
        own_units = max(0, min(MAX_UNITS, self._feature(FEATURES["OWN_UNITS"], 0)))
        raw_status = STATUS_FAILURE if s.fault else self._eval_node(self.p.root)
        if s.fault or raw_status in (STATUS_FAILURE, STATUS_SKIPPED):
            s.intents = []
        if not s.fault and raw_status in (STATUS_SUCCESS, STATUS_RUNNING):
            self._arbitrate()
        status = STATUS_FAILURE if s.fault else STATUS_SUCCESS if raw_status == STATUS_SKIPPED else raw_status
        units: List[Tuple[int, int, int]] = []
        for i in range(own_units):
            c = s.unit_claims[i]
            units.append((c.op, c.arg, c.n) if c else (UNIT_OPS["PASS"], 0, 1))
        cap = max(0, min(MAX_MARKET_INTENTS, int(max_market_orders)))
        market = [(c.op, c.item, c.n) for c in s.market_claims[:cap]]
        action = ActionResult(own_units, units, market)
        s.ticks += 1
        return TickResult(status, action, s.fault, tuple(s.blackboard),
                          tuple((x.a & 0xFFFFFFFF, x.b & 0xFFFFFFFF) for x in s.node_state),
                          tuple(self._trace or ()))

    def _trace_event(self, node: int, status: int, event: str = "tick") -> None:
        if self._trace is not None:
            step = self._feature(FEATURES["STEP"], 0)
            self._trace.append((step, self.p.nodes[node].debug_id, event, STATUS_NAMES[status]))

    def _feature(self, fid: int, arg: int) -> int:
        return int(self._obs.feature(fid, arg))

    def _cmp(self, lhs: int, rhs: int, cmp_id: int) -> bool:
        return (lhs == rhs, lhs != rhs, lhs < rhs, lhs <= rhs, lhs > rhs, lhs >= rhs)[cmp_id]

    def _pred(self, idx: int) -> bool:
        p = self.p.predicates[idx]
        return self._cmp(self._value(p.lhs), self._value(p.rhs), p.cmp)

    def _value(self, idx: int) -> int:
        v = self.p.values[idx]
        if v.kind == VALUE_KINDS["CONST"]: return v.value
        if v.kind == VALUE_KINDS["BB"]: return self.state.blackboard[v.id]
        return self._feature(v.id, v.arg)

    def _set_fault(self, code: int) -> None:
        if self.state.fault == 0:
            self.state.fault = int(code)

    def _write_bb(self, slot: int, value: int) -> bool:
        try:
            _validate_bb_value(self.p.bb[slot].type, value, "runtime blackboard write")
        except PolicyError:
            self._set_fault(5)
            return False
        self.state.blackboard[slot] = int(value)
        return True

    def _effects(self, rng: Tuple[int, int]) -> None:
        start, count = rng
        for e in self.p.effects[start:start + count]:
            if self.state.fault:
                return
            if e.op == EFFECT_OP["set"]:
                if not self._write_bb(e.slot, self._value(e.value)): return
            elif e.op == EFFECT_OP["clear"]:
                self.state.blackboard[e.slot] = self.p.bb[e.slot].initial
            elif e.op == EFFECT_OP["add"]:
                v = self.state.blackboard[e.slot] + self._value(e.value)
                if v < -(1 << 31) or v >= (1 << 31):
                    self._set_fault(2)
                    return
                self.state.blackboard[e.slot] = v

    def _is_active(self, n: Node) -> bool:
        return n.state_slot != NONE_U16 and bool(self.state.node_state[n.state_slot].b & ACTIVE_BIT)

    def _set_active(self, n: Node, active: bool) -> None:
        if n.state_slot == NONE_U16: return
        st = self.state.node_state[n.state_slot]
        st.b = (st.b | ACTIVE_BIT) if active else (st.b & ~ACTIVE_BIT)

    def _eval_node(self, idx: int) -> int:
        if self.state.fault:
            return STATUS_FAILURE
        intent_checkpoint = len(self.state.intents)
        self.state.visits += 1
        if self.state.visits > self.p.instruction_budget:
            self._set_fault(1)
            del self.state.intents[intent_checkpoint:]
            return STATUS_FAILURE
        n = self.p.nodes[idx]
        lc = self.p.lifecycles[n.lifecycle_slot] if n.lifecycle_slot != NONE_U16 else None
        active = self._is_active(n)
        if lc is not None:
            if not active:
                if lc.failure_if != NONE_U16 and self._pred(lc.failure_if):
                    self._effects(lc.on_failure); self._effects(lc.post); self._trace_event(idx, STATUS_FAILURE, "precondition"); del self.state.intents[intent_checkpoint:]; return STATUS_FAILURE
                if lc.success_if != NONE_U16 and self._pred(lc.success_if):
                    self._effects(lc.on_success); self._effects(lc.post); self._trace_event(idx, STATUS_SUCCESS, "precondition"); return STATUS_SUCCESS
                if lc.skip_if != NONE_U16 and self._pred(lc.skip_if):
                    self._trace_event(idx, STATUS_SKIPPED, "precondition"); del self.state.intents[intent_checkpoint:]; return STATUS_SKIPPED
                if lc.while_pred != NONE_U16 and not self._pred(lc.while_pred):
                    self._trace_event(idx, STATUS_SKIPPED, "precondition"); del self.state.intents[intent_checkpoint:]; return STATUS_SKIPPED
            elif lc.while_pred != NONE_U16 and not self._pred(lc.while_pred):
                self._halt_node(idx)
                self._trace_event(idx, STATUS_SKIPPED, "while_halt")
                del self.state.intents[intent_checkpoint:]
                return STATUS_SKIPPED
        status = self._eval_core(idx, n)
        if self.state.fault:
            del self.state.intents[intent_checkpoint:]
            return STATUS_FAILURE
        if status == STATUS_RUNNING:
            self._set_active(n, True)
        else:
            self._set_active(n, False)
            if lc is not None and status == STATUS_SUCCESS:
                self._effects(lc.on_success); self._effects(lc.post)
            elif lc is not None and status == STATUS_FAILURE:
                self._effects(lc.on_failure); self._effects(lc.post)
        if status in (STATUS_FAILURE, STATUS_SKIPPED):
            del self.state.intents[intent_checkpoint:]
        self._trace_event(idx, status)
        return status

    def _children(self, n: Node) -> Sequence[int]:
        return self.p.child_refs[n.first_child:n.first_child + n.child_count]

    def _halt_node(self, idx: int) -> None:
        n = self.p.nodes[idx]
        was_active = self._is_active(n)
        children = self._children(n)
        st = self.state.node_state[n.state_slot] if n.state_slot != NONE_U16 else None
        if n.type in (NODE_TYPES["Sequence"], NODE_TYPES["Selector"]):
            if st and children and st.a < len(children): self._halt_node(children[st.a])
        elif n.type in (NODE_TYPES["ReactiveSequence"], NODE_TYPES["ReactiveSelector"]):
            if st and st.a and st.a - 1 < len(children): self._halt_node(children[st.a - 1])
        elif n.type == NODE_TYPES["If"]:
            if st and children:
                phase = st.a
                if phase == 0: self._halt_node(children[0])
                elif phase in (1, 2): self._halt_node(children[phase])
        elif n.type == NODE_TYPES["Parallel"]:
            if st:
                running = st.b & RUNNING_MASK16
                for i, c in enumerate(children):
                    if running & (1 << i): self._halt_node(c)
        elif n.type in DECORATOR_TYPES and children:
            self._halt_node(children[0])
        if st:
            st.a = 0; st.b = 0
        if was_active and n.lifecycle_slot != NONE_U16:
            self._effects(self.p.lifecycles[n.lifecycle_slot].on_halted)
        if was_active:
            self._trace_event(idx, STATUS_SKIPPED, "halt")

    def _eval_core(self, idx: int, n: Node) -> int:
        t = n.type
        children = self._children(n)
        st = self.state.node_state[n.state_slot] if n.state_slot != NONE_U16 else None
        if t == NODE_TYPES["Sequence"]:
            i = int(st.a)
            while i < len(children):
                r = self._eval_node(children[i])
                if r in (STATUS_SUCCESS, STATUS_SKIPPED): i += 1; st.a = i; continue
                if r == STATUS_RUNNING: st.a = i; return r
                st.a = 0; return STATUS_FAILURE
            st.a = 0; return STATUS_SUCCESS
        if t == NODE_TYPES["Selector"]:
            i = int(st.a)
            while i < len(children):
                r = self._eval_node(children[i])
                if r in (STATUS_FAILURE, STATUS_SKIPPED): i += 1; st.a = i; continue
                if r == STATUS_RUNNING: st.a = i; return r
                st.a = 0; return STATUS_SUCCESS
            st.a = 0; return STATUS_FAILURE
        if t in (NODE_TYPES["ReactiveSequence"], NODE_TYPES["ReactiveSelector"]):
            prev = int(st.a) - 1 if st.a else -1
            for i, c in enumerate(children):
                r = self._eval_node(c)
                if t == NODE_TYPES["ReactiveSequence"]:
                    if r in (STATUS_SUCCESS, STATUS_SKIPPED): continue
                    if prev >= 0 and prev != i: self._halt_node(children[prev])
                    st.a = i + 1 if r == STATUS_RUNNING else 0
                    return r
                else:
                    if r in (STATUS_FAILURE, STATUS_SKIPPED): continue
                    if prev >= 0 and prev != i: self._halt_node(children[prev])
                    st.a = i + 1 if r == STATUS_RUNNING else 0
                    return r
            if prev >= 0: self._halt_node(children[prev])
            st.a = 0
            return STATUS_SUCCESS if t == NODE_TYPES["ReactiveSequence"] else STATUS_FAILURE
        if t == NODE_TYPES["If"]:
            phase = int(st.a)
            if phase == 0:
                r = self._eval_node(children[0])
                if r == STATUS_RUNNING: return r
                if r == STATUS_SKIPPED: st.a = 0; return r
                phase = 1 if r == STATUS_SUCCESS else 2
                st.a = phase
            r = self._eval_node(children[phase])
            if r != STATUS_RUNNING: st.a = 0
            return r
        if t == NODE_TYPES["Parallel"]:
            success = st.a & 0xFFFF
            failure = (st.a >> 16) & 0xFFFF
            running = st.b & RUNNING_MASK16
            skipped = 0
            for i, c in enumerate(children):
                bit = 1 << i
                if success & bit or failure & bit:
                    continue
                r = self._eval_node(c)
                if self.state.fault:
                    return STATUS_FAILURE
                if r == STATUS_SUCCESS:
                    success |= bit; running &= ~bit
                elif r == STATUS_FAILURE:
                    failure |= bit; running &= ~bit
                elif r == STATUS_RUNNING:
                    running |= bit
                elif r == STATUS_SKIPPED:
                    skipped += 1; running &= ~bit
                st.a = success | (failure << 16)
                st.b = (st.b & ACTIVE_BIT) | running
                sc, fc = success.bit_count(), failure.bit_count()
                possible_successes = len(children) - fc - skipped
                if sc >= n.p0 or fc >= n.p1 or possible_successes < n.p0:
                    for j, cc in enumerate(children):
                        if running & (1 << j):
                            self._halt_node(cc)
                    out = STATUS_SUCCESS if sc >= n.p0 else STATUS_FAILURE
                    st.a = 0; st.b = 0
                    return out
            if running:
                return STATUS_RUNNING
            out = STATUS_SKIPPED if success == 0 and failure == 0 and skipped == len(children) else STATUS_FAILURE
            st.a = 0; st.b = 0
            return out
        if t == NODE_TYPES["Invert"]:
            r = self._eval_node(children[0])
            return STATUS_FAILURE if r == STATUS_SUCCESS else STATUS_SUCCESS if r == STATUS_FAILURE else r
        if t in (NODE_TYPES["ForceSuccess"], NODE_TYPES["ForceFailure"]):
            r = self._eval_node(children[0])
            if r in (STATUS_RUNNING, STATUS_SKIPPED): return r
            return STATUS_SUCCESS if t == NODE_TYPES["ForceSuccess"] else STATUS_FAILURE
        if t == NODE_TYPES["TurnTimeout"]:
            active = bool(st.b & ACTIVE_BIT)
            now = self._feature(FEATURES["STEP"], 0)
            if not active: st.a = now & 0xFFFFFFFF
            elif now - int(st.a) >= n.p0:
                self._halt_node(children[0]); st.a = 0; return STATUS_FAILURE
            r = self._eval_node(children[0])
            if r != STATUS_RUNNING: st.a = 0
            return r
        if t == NODE_TYPES["Retry"]:
            r = self._eval_node(children[0])
            if r == STATUS_SUCCESS: st.a = 0; return r
            if r == STATUS_FAILURE:
                st.a += 1
                if st.a >= n.p0: st.a = 0; return STATUS_FAILURE
                return STATUS_RUNNING
            return r
        if t == NODE_TYPES["Repeat"]:
            r = self._eval_node(children[0])
            if r == STATUS_FAILURE: st.a = 0; return r
            if r == STATUS_SUCCESS:
                st.a += 1
                if st.a >= n.p0: st.a = 0; return STATUS_SUCCESS
                return STATUS_RUNNING
            return r
        if t == NODE_TYPES["KeepRunningUntilFailure"]:
            r = self._eval_node(children[0])
            if r == STATUS_FAILURE: return STATUS_SUCCESS
            if r == STATUS_SUCCESS: return STATUS_RUNNING
            return r
        if t == NODE_TYPES["Success"]: return STATUS_SUCCESS
        if t == NODE_TYPES["Failure"]: return STATUS_FAILURE
        if t == NODE_TYPES["Condition"]:
            return STATUS_SUCCESS if self._cmp(self._value(n.p0), self._value(n.p1), n.flags) else STATUS_FAILURE
        if t == NODE_TYPES["SetBlackboard"]:
            return STATUS_SUCCESS if self._write_bb(n.p0, self._value(n.p1)) else STATUS_FAILURE
        if t == NODE_TYPES["AddBlackboard"]:
            v = self.state.blackboard[n.p0] + self._value(n.p1)
            if v < -(1 << 31) or v >= (1 << 31): self._set_fault(2); return STATUS_FAILURE
            self.state.blackboard[n.p0] = v; return STATUS_SUCCESS
        if t == NODE_TYPES["EmitUnitIntent"]:
            if n.p0 >= self._feature(FEATURES["OWN_UNITS"], 0): return STATUS_FAILURE
            self._emit_unit(n.p0, n.p1, n.p2, n.p3, n.flags); return STATUS_SUCCESS
        if t == NODE_TYPES["EmitMarketIntent"]:
            self._emit_market(n.p0, n.p1, n.p2, n.p3); return STATUS_SUCCESS
        if t == NODE_TYPES["WaitTurns"]:
            active = bool(st.b & ACTIVE_BIT)
            now = self._feature(FEATURES["STEP"], 0)
            if not active: st.a = now & 0xFFFFFFFF; return STATUS_RUNNING
            if now - int(st.a) >= n.p0: st.a = 0; return STATUS_SUCCESS
            return STATUS_RUNNING
        if t == NODE_TYPES["MoveUnitTo"]:
            unit, target = self._value(n.p0), self._value(n.p1)
            if unit < 0 or target < 0 or target >= BOARD_SIZE * BOARD_SIZE or unit >= self._feature(FEATURES["OWN_UNITS"], 0): return STATUS_FAILURE
            x = self._feature(FEATURES["OWN_WORKER_X"], unit)
            y = self._feature(FEATURES["OWN_WORKER_Y"], unit)
            tx, ty = target % BOARD_SIZE, target // BOARD_SIZE
            if x == tx and y == ty: return STATUS_SUCCESS
            op = UNIT_OPS["EAST"] if x < tx else UNIT_OPS["WEST"] if x > tx else UNIT_OPS["SOUTH"] if y < ty else UNIT_OPS["NORTH"]
            self._emit_unit(unit, op, 0, 1, n.p2)
            return STATUS_RUNNING
        self._set_fault(3)
        return STATUS_FAILURE

    def _emit_unit(self, unit: int, op: int, arg: int, quantity: int, priority: int) -> None:
        self._defer(DeferredIntent(0, priority, op, arg, quantity, unit, self.state.emit_seq))

    def _emit_market(self, op: int, item: int, quantity: int, priority: int) -> None:
        self._defer(DeferredIntent(1, priority, op, item, quantity, 0, self.state.emit_seq))

    def _defer(self, intent: DeferredIntent) -> None:
        self.state.emit_seq += 1
        if len(self.state.intents) >= MAX_DEFERRED_INTENTS:
            self._set_fault(4)
            return
        self.state.intents.append(intent)

    def _arbitrate(self) -> None:
        s = self.state
        for intent in s.intents:
            if intent.kind == 0:
                if not 0 <= intent.subject < MAX_UNITS: continue
                old = s.unit_claims[intent.subject]
                candidate = UnitClaim(intent.priority, intent.order, intent.op, intent.arg, intent.n)
                if old is None or intent.priority > old.priority or (intent.priority == old.priority and intent.order < old.order):
                    s.unit_claims[intent.subject] = candidate
            else:
                s.market_claims.append(MarketClaim(intent.priority, intent.order, intent.op, intent.arg, intent.n))
        s.market_claims.sort(key=lambda z: (-z.priority, z.order))
        del s.market_claims[MAX_MARKET_INTENTS:]


def decompile_policy(policy: CompiledPolicy) -> Dict[str, Any]:
    """Reconstruct a canonical authoring form from portable compiled IR.

    Debug-only node names are intentionally not in the runtime ABI. Blackboard names
    loaded from .bkgc are deterministically regenerated as slot_N; this does not change
    compiled runtime semantics.
    """
    slot_names = [b.name if b.name else f"slot_{i}" for i, b in enumerate(policy.bb)]

    def value_obj(index: int) -> Any:
        ref = policy.values[index]
        if ref.kind == VALUE_KINDS["CONST"]:
            return ref.value if ref.type == BB_TYPES["I32"] else {"const": ref.value, "type": BB_NAMES[ref.type]}
        if ref.kind == VALUE_KINDS["BB"]:
            return {"blackboard": slot_names[ref.id]}
        out: Dict[str, Any] = {"feature": FEATURE_NAMES[ref.id]}
        if ref.arg != 0 or ref.id in {
            FEATURES["MARKET_PRICE"], FEATURES["MARKET_INVENTORY"], FEATURES["OWN_SHED"],
            FEATURES["OWN_SEED"], FEATURES["OWN_INVENTORY"], FEATURES["SHOP_AT"],
            FEATURES["SHOP_SEEN"], FEATURES["OWN_WORKER_X"], FEATURES["OWN_WORKER_Y"],
            FEATURES["OPP_WORKER_X"], FEATURES["OPP_WORKER_Y"], FEATURES["OWN_WORKER_TILE"],
            FEATURES["OPP_WORKER_TILE"], FEATURES["OWN_QUADRANT_UNLOCKED"], FEATURES["OPP_QUADRANT_UNLOCKED"],
        } or ref.id in TILE_FEATURES:
            out["arg"] = ref.arg
        return out

    def pred_obj(index: int) -> Dict[str, Any]:
        q = policy.predicates[index]
        return {"lhs": value_obj(q.lhs), "op": CMP_NAMES[q.cmp], "rhs": value_obj(q.rhs)}

    effect_names = {v: k for k, v in EFFECT_OP.items()}
    def effect_range(rng: Tuple[int, int]) -> List[Dict[str, Any]]:
        start, count = rng
        result: List[Dict[str, Any]] = []
        for e in policy.effects[start:start + count]:
            item: Dict[str, Any] = {"op": effect_names[e.op], "slot": slot_names[e.slot]}
            if e.op != EFFECT_OP["clear"]:
                item["value"] = value_obj(e.value)
            result.append(item)
        return result

    def lifecycle_obj(index: int) -> Dict[str, Any]:
        lc = policy.lifecycles[index]
        out: Dict[str, Any] = {}
        for key, q in (("failure_if", lc.failure_if), ("success_if", lc.success_if),
                       ("skip_if", lc.skip_if), ("while", lc.while_pred)):
            if q != NONE_U16:
                out[key] = pred_obj(q)
        for key, rng in (("on_success", lc.on_success), ("on_failure", lc.on_failure),
                         ("on_halted", lc.on_halted), ("post", lc.post)):
            effects = effect_range(rng)
            if effects:
                out[key] = effects
        return out

    unit_names = {v: k for k, v in UNIT_OPS.items()}
    market_names = {v: k for k, v in MARKET_OPS.items()}

    def node_obj(index: int) -> Dict[str, Any]:
        n = policy.nodes[index]
        typ = NODE_NAMES[n.type]
        out: Dict[str, Any] = {"type": typ}
        children = policy.child_refs[n.first_child:n.first_child + n.child_count]
        if typ in ("Sequence", "Selector", "ReactiveSequence", "ReactiveSelector", "Parallel"):
            out["children"] = [node_obj(c) for c in children]
            if typ == "Parallel":
                out["success_threshold"] = n.p0
                out["failure_threshold"] = n.p1
        elif typ == "If":
            out["condition"], out["then"], out["else"] = (node_obj(children[0]), node_obj(children[1]), node_obj(children[2]))
        elif n.type in DECORATOR_TYPES:
            out["child"] = node_obj(children[0])
            if typ == "TurnTimeout": out["turns"] = n.p0
            elif typ in ("Retry", "Repeat"): out["n"] = n.p0
        elif typ == "Condition":
            out.update({"lhs": value_obj(n.p0), "op": CMP_NAMES[n.flags], "rhs": value_obj(n.p1)})
        elif typ in ("SetBlackboard", "AddBlackboard"):
            out.update({"slot": slot_names[n.p0], "value": value_obj(n.p1)})
        elif typ == "EmitUnitIntent":
            out.update({"unit": n.p0, "action": unit_names[n.p1], "priority": n.flags})
            if n.p1 in (UNIT_OPS["PICKUP"], UNIT_OPS["PLACE"], UNIT_OPS["PLANT"]): out["item"] = ITEMS[n.p2]
            if n.p3 != 1: out["quantity"] = n.p3
        elif typ == "EmitMarketIntent":
            out.update({"action": market_names[n.p0], "priority": n.p3})
            if n.p0 not in (MARKET_OPS["HIRE"], MARKET_OPS["BUY_LAND"]):
                out["item"] = ITEMS[n.p1]
                out["quantity"] = n.p2
        elif typ == "WaitTurns":
            out["turns"] = n.p0
        elif typ == "MoveUnitTo":
            out.update({"unit": value_obj(n.p0), "target": value_obj(n.p1), "priority": n.p2})
        if n.lifecycle_slot != NONE_U16:
            out["lifecycle"] = lifecycle_obj(n.lifecycle_slot)
        return out

    bb = [{"name": slot_names[i], "type": BB_NAMES[b.type], "initial": b.initial} for i, b in enumerate(policy.bb)]
    result: Dict[str, Any] = {"schema_version": SCHEMA_VERSION, "tree": node_obj(policy.root)}
    if bb:
        result["blackboard"] = bb
    return result

class KaggricultureAgent:
    """Reusable official-Python agent wrapper with explicit episode rollback reset."""

    def __init__(self, policy: CompiledPolicy | Dict[str, Any], max_market_orders: int = 10):
        self.compiled = policy if isinstance(policy, CompiledPolicy) else compile_policy(policy)
        self.vm = VM(self.compiled)
        self.max_market_orders = int(max_market_orders)
        self.last_step: Optional[int] = None

    def reset(self) -> None:
        self.vm.reset()
        self.last_step = None

    def __call__(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        fallback_step = int(observation.get("day", 0) or 0) * 24 + int(observation.get("hour", 0) or 0)
        step = int(observation.get("step", fallback_step) or 0)
        if self.last_step is not None and step <= self.last_step:
            self.vm.reset()
        self.last_step = step
        result = self.vm.tick(observation, max_market_orders=self.max_market_orders, trace=False)
        return to_official_action(result.action)


def to_official_action(action: ActionResult) -> Dict[str, Any]:
    unit_name = {v: k for k, v in UNIT_OPS.items()}
    market_name = {v: k for k, v in MARKET_OPS.items()}
    def unit(a: Tuple[int, int, int]) -> List[Any]:
        op, arg, n = a; name = unit_name.get(op, "PASS")
        if name in ("PICKUP", "PLACE", "PLANT"):
            return [name, ITEMS[arg], n] if name in ("PICKUP", "PLACE") else [name, ITEMS[arg]]
        return [name]
    def order(a: Tuple[int, int, int]) -> List[Any]:
        op, item, n = a; name = market_name.get(op, "")
        if name in ("HIRE", "BUY_LAND"): return [name]
        return [name, ITEMS[item], n]
    units = list(action.units)
    farmer = unit(units[0]) if units else ["PASS"]
    hands = [unit(x) for x in units[1:]]
    return {"farmer": farmer, "hands": hands, "market": [order(x) for x in action.market]}


def compile_file(src: str | Path, dst: str | Path) -> CompiledPolicy:
    p = compile_policy(load_policy_json(src))
    save_compiled(p, dst)
    return p


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Compile/inspect Bonsai-KG policy v1")
    ap.add_argument("policy")
    ap.add_argument("--out")
    ap.add_argument("--canonical", action="store_true")
    ap.add_argument("--decompile", action="store_true")
    args = ap.parse_args()
    path = Path(args.policy)
    if path.suffix == ".bkgc":
        if args.out:
            ap.error("--out is only valid when compiling authoring JSON")
        compiled = load_compiled(path)
        source = decompile_policy(compiled)
        if args.canonical or args.decompile:
            print(json.dumps(source, sort_keys=True, indent=2, ensure_ascii=False) + "\n", end="")
        print(json.dumps({"hash": compiled.canonical_hash, "nodes": len(compiled.nodes), "state_slots": compiled.state_count,
                          "compiled_bytes": compiled.compiled_bytes, "instruction_budget": compiled.instruction_budget}, sort_keys=True))
    else:
        source = load_policy_json(path)
        compiled = compile_policy(source)
        if args.out: save_compiled(compiled, args.out)
        if args.canonical: print(dumps_canonical(source, pretty=True), end="")
        if args.decompile: print(json.dumps(decompile_policy(compiled), sort_keys=True, indent=2, ensure_ascii=False) + "\n", end="")
        print(json.dumps({"hash": compiled.canonical_hash, "nodes": len(compiled.nodes), "state_slots": compiled.state_count,
                          "compiled_bytes": compiled.compiled_bytes, "instruction_budget": compiled.instruction_budget}, sort_keys=True))
