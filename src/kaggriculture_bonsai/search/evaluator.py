"""Auditable Option-Program v1 reference semantics and host encoding."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from enum import IntEnum
import json
import struct
from typing import Any, Mapping, Sequence

from .canonical import canonical_action
from .ir import (
    ActionFamily,
    ActionSource,
    ContinuationKind,
    MacroProgram,
    Phase,
    PublicCondition,
    TerminationType,
)
from .motifs import MOTIF_SCHEMA_VERSION, Motif
from .schema import canonical_json

EVALUATOR_FORMAT_VERSION = 1
_MAGIC = b"RLRMBE1\0"
_HEADER = struct.Struct("<8sII")
_DEVICE_HEADER = struct.Struct("<IIIIII")
_PROGRAM_RECORD = struct.Struct("<HHIBBBB32s")
_PHASE_RECORD = struct.Struct("<BBBBHHHHiiBBBBii")
_DECISION_RECORD = struct.Struct("<BBBBBBHHii")
_MOTIF_RECORD = struct.Struct("<IHBB")
_UNIT_RECORD = struct.Struct("<BBh")
_ORDER_RECORD = struct.Struct("<BBi")
_RESULT_HEADER = struct.Struct("<8sIII")
_RESULT_RECORD_V1 = struct.Struct("<iiHHBBBB32s")
_RESULT_RECORD_V2 = struct.Struct("<iiHHBBBBHHHHHHHHQBBBBhhhhhhhhhh32s")

_UNIT_OPS = ("PASS", "NORTH", "SOUTH", "EAST", "WEST", "PICKUP", "DROP", "PLACE", "PLANT", "WATER", "HARVEST", "FERTILIZE", "DIG", "BUILD_COOP", "BUILD_PASTURE", "FEED", "COLLECT_FERTILIZER", "CARE")
_MARKET_OPS = (None, "HIRE", "BUY_LAND", "BUY_SEED", "BUY_PRODUCT", "BUY_ANIMAL", "SELL")
_FEATURES = ("ALWAYS", "STEP", "DAY", "HOUR", "OWN_MONEY", "OPP_MONEY", "OWN_UNITS", "OPP_UNITS", "OWN_LAND", "OPP_LAND", "OWN_HIRES_TODAY", "MARKET_PRICE", "MARKET_INVENTORY", "OWN_SHED", "OWN_SHED_TOTAL", "OWN_SEED", "OWN_INVENTORY", "SHOP_COUNT", "SHOP_AT", "SHOP_SEEN", "OWN_WORKER_X", "OWN_WORKER_Y", "OPP_WORKER_X", "OPP_WORKER_Y", "OWN_TILE_KIND", "OPP_TILE_KIND", "OWN_TILE_ITEM", "OPP_TILE_ITEM", "OWN_TILE_YIELD", "OPP_TILE_YIELD", "OWN_TILE_WATERED", "OPP_TILE_WATERED", "OWN_TILE_FED", "OPP_TILE_FED", "OWN_TILE_CARED", "OPP_TILE_CARED", "OWN_TILE_FERT_AVAILABLE", "OPP_TILE_FERT_AVAILABLE", "OPP_HIRES_TODAY")


class FaultCode(IntEnum):
    NONE = 0
    INVALID_PROGRAM_VERSION = 1
    INVALID_PHASE_COUNT = 2
    INVALID_SOURCE = 3
    INVALID_MOTIF_REFERENCE = 4
    UNSUPPORTED_PRIMITIVE = 5
    INVALID_TERMINATION = 6
    MOTIF_EXHAUSTED = 7
    INVALID_FEATURE = 8
    HORIZON_VIOLATION = 9
    INTERNAL_ERROR = 10
    INVALID_DECISION = 11
    INVALID_CONTINUATION = 12
    INVALID_RECOVERY = 13


class TerminationReason(IntEnum):
    NONE = 0
    FIXED_CAP = 1
    NEXT_EOD = 2
    NEXT_SHOP_REVEAL = 3
    PURCHASE_COMPLETED = 4
    TARGET_PUBLIC_STATE = 5
    PARENT_COMPATIBLE = 6
    MAX_STEPS = 7
    PROGRAM_CAP = 8


class CompletionReason(IntEnum):
    NONE = 0
    FINISHED = 1
    ABORT = 2
    RECOVERY = 3
    PROGRAM_CAP = 4
    FAULT = 5


def merge_action_families(
    parent_action: Mapping[str, Any],
    source_action: Mapping[str, Any],
    selected: Sequence[ActionFamily],
) -> dict[str, Any]:
    """Select whole ordered action families; never merge or sort within a family."""
    parent = canonical_action(parent_action)
    source = canonical_action(source_action)
    mask = set(selected)
    hand_count = len(parent["hands"])
    source_hands = list(source["hands"][:hand_count])
    source_hands.extend([["PASS"]] * (hand_count - len(source_hands)))
    return {
        "farmer": source["farmer"] if ActionFamily.FARMER in mask else parent["farmer"],
        "hands": source_hands if ActionFamily.HANDS in mask else parent["hands"],
        "market": source["market"][:10] if ActionFamily.MARKET in mask else parent["market"][:10],
    }


def _motif_from_semantic(document: Mapping[str, Any]) -> Motif:
    if document.get("schema_version") != MOTIF_SCHEMA_VERSION:
        raise ValueError("unsupported motif schema version")
    return Motif(
        tuple(document["action_block"]),
        tuple(ActionFamily(value) for value in document["supported_families"]),
    )


def encode_program_batch(programs: Sequence[MacroProgram], motifs: Mapping[str, Motif]) -> bytes:
    """Encode a deterministic versioned host artifact consumed by the CUDA loader."""
    if not programs:
        raise ValueError("program batch must not be empty")
    referenced = {
        phase.motif_id
        for program in programs
        for phase in (*program.phases, *((program.recovery_phase,) if program.recovery_phase else ()))
        if phase.motif_id
    }
    missing = referenced - set(motifs)
    if missing:
        raise ValueError(f"missing motif references: {sorted(missing)}")
    for program in programs:
        for phase in (*program.phases, *((program.recovery_phase,) if program.recovery_phase else ())):
            if phase.motif_id and not set(phase.action_families).issubset(motifs[phase.motif_id].supported_families):
                raise ValueError("phase mask exceeds motif supported action families")
    payload = canonical_json(
        {
            "programs": [program.to_document() for program in programs],
            "motifs": {key: motifs[key].semantic_document() for key in sorted(referenced)},
        }
    ).encode("utf-8")
    return _HEADER.pack(_MAGIC, EVALUATOR_FORMAT_VERSION, len(payload)) + payload


def decode_program_batch(data: bytes) -> tuple[tuple[MacroProgram, ...], dict[str, Motif]]:
    if len(data) < _HEADER.size:
        raise ValueError("short evaluator batch")
    magic, version, length = _HEADER.unpack_from(data)
    if magic != _MAGIC:
        raise ValueError("invalid evaluator batch magic")
    if version != EVALUATOR_FORMAT_VERSION:
        raise ValueError(f"unsupported evaluator format version: {version}")
    if len(data) != _HEADER.size + length:
        raise ValueError("evaluator batch length mismatch")
    document = json.loads(data[_HEADER.size :].decode("utf-8"))
    programs = tuple(MacroProgram.from_document(item) for item in document["programs"])
    motifs = {key: _motif_from_semantic(item) for key, item in document["motifs"].items()}
    if any(motifs[key].semantic_id != key for key in motifs):
        raise ValueError("motif semantic ID mismatch")
    return programs, motifs


def _pack_action(action: Mapping[str, Any]) -> bytes:
    value = canonical_action(action)
    units = [value["farmer"], *value["hands"]]
    result = bytearray()
    for index in range(40):
        unit = units[index] if index < len(units) else ["PASS"]
        op = _UNIT_OPS.index(unit[0])
        arg = _ITEMS.index(unit[1]) if len(unit) > 1 and unit[1] in _ITEMS else 0
        count = int(unit[2]) if len(unit) > 2 else 1
        result.extend(_UNIT_RECORD.pack(op, arg, count))
    for index in range(10):
        order = value["market"][index] if index < len(value["market"]) else []
        op = _MARKET_OPS.index(order[0]) if order else 0
        item = _ITEMS.index(order[1]) if len(order) > 1 and order[1] in _ITEMS else 0
        count = int(order[2]) if len(order) > 2 else (1 if op else 0)
        result.extend(_ORDER_RECORD.pack(op, item, count))
    result.extend(bytes((min(len(units), 40), min(len(value["market"]), 10))))
    return bytes(result)


def encode_device_batch(
    programs: Sequence[MacroProgram],
    motifs: Mapping[str, Motif],
    *,
    initiation_clauses: Sequence[Sequence[Sequence[PublicCondition]]] | None = None,
    semantic_ids: Sequence[str] | None = None,
) -> bytes:
    """Flatten programs and shared motifs into the exact packed CUDA ABI."""
    if not programs:
        raise ValueError("program batch must not be empty")
    if initiation_clauses is not None and len(initiation_clauses) != len(programs):
        raise ValueError("specialist gate/program count mismatch")
    if semantic_ids is not None and len(semantic_ids) != len(programs):
        raise ValueError("semantic ID/program count mismatch")
    motif_ids = sorted({
        phase.motif_id
        for item in programs
        for phase in (*item.phases, *((item.recovery_phase,) if item.recovery_phase else ()))
        if phase.motif_id
    })
    if any(key not in motifs for key in motif_ids):
        raise ValueError("device batch is missing a referenced motif")
    for program in programs:
        for phase in (*program.phases, *((program.recovery_phase,) if program.recovery_phase else ())):
            if phase.motif_id and not set(phase.action_families).issubset(motifs[phase.motif_id].supported_families):
                raise ValueError("phase mask exceeds motif supported action families")
    motif_index = {key: index for index, key in enumerate(motif_ids)}
    actions: list[bytes] = []
    motif_records: list[bytes] = []
    for key in motif_ids:
        item = motifs[key]
        document = item.semantic_document()
        first = len(actions)
        actions.extend(_pack_action(action) for action in document["action_block"])
        mask = sum({ActionFamily.FARMER: 1, ActionFamily.HANDS: 2, ActionFamily.MARKET: 4}[family] for family in item.supported_families)
        motif_records.append(_MOTIF_RECORD.pack(first, len(document["action_block"]), mask, 0))
    phase_records: list[bytes] = []
    decision_records: list[bytes] = []
    program_records: list[bytes] = []
    sources = {source: index for index, source in enumerate(ActionSource)}
    terminations = {kind: index for index, kind in enumerate(TerminationType)}
    objectives = {None: 0, "BUY_LAND": 1, "HIRE": 2, "BUY_ANIMAL": 3, "BUY_SEED": 4, "BUY_PRODUCT": 5}
    continuation_kinds = {
        ContinuationKind.PHASE: 0,
        ContinuationKind.ABORT_TO_PARENT: 1,
        ContinuationKind.RECOVERY: 2,
    }

    def pack_phase(phase: Phase) -> bytes:
        condition = phase.termination.condition
        feature = _FEATURES.index(condition.feature) if condition else 0
        comparator = {"EQ": 0, "NE": 1, "LE": 2, "GE": 3}[condition.comparator] if condition else 0
        parameters = dict(phase.parameters)
        primitive = 1 if phase.primitive_name == "LIQUIDITY_REBALANCE" else (0 if phase.action_source is not ActionSource.PARAMETERIZED_PRIMITIVE else 255)
        primitive_item = _ITEMS.index(parameters.get("mode", "WHEAT")) if parameters.get("mode", "WHEAT") in _ITEMS else 255
        mask = sum({ActionFamily.FARMER: 1, ActionFamily.HANDS: 2, ActionFamily.MARKET: 4}[family] for family in phase.action_families)
        return _PHASE_RECORD.pack(
            sources[phase.action_source], mask, terminations[phase.termination.kind], primitive,
            phase.termination.max_steps, motif_index.get(phase.motif_id, 0xFFFF), feature, comparator,
            condition.index if condition else 0, condition.value if condition else 0,
            objectives[phase.termination.objective], primitive_item, int(parameters.get("enabled", False)), 0,
            phase.termination.target_count or 0, int(parameters.get("quantity", 0)),
        )

    for program_index, item in enumerate(programs):
        phase_first = len(phase_records)
        for phase in item.phases:
            phase_records.append(pack_phase(phase))
        if item.recovery_phase is not None:
            phase_records.append(pack_phase(item.recovery_phase))
        clauses = None if initiation_clauses is None else initiation_clauses[program_index]
        gate_count = 0
        if clauses is not None:
            if not 1 <= len(clauses) <= 6:
                raise ValueError("specialist gate requires 1..6 clauses")
            for clause_index, clause in enumerate(clauses):
                if not 1 <= len(clause) <= 4:
                    raise ValueError("specialist gate clauses require 1..4 conditions")
                for condition in clause:
                    phase_records.append(_PHASE_RECORD.pack(
                        0, 0, 0, 0, 0, 0, _FEATURES.index(condition.feature),
                        {"EQ": 0, "NE": 1, "LE": 2, "GE": 3}[condition.comparator],
                        condition.index, condition.value, 0, 0, 0, clause_index, 0, 0,
                    ))
                    gate_count += 1
        decision_first = len(decision_records)
        for decision in item.decisions:
            def target(continuation):
                return continuation.phase_index if continuation.phase_index is not None else 0xFF
            decision_records.append(_DECISION_RECORD.pack(
                decision.after_phase,
                continuation_kinds[decision.when_true.kind], target(decision.when_true),
                continuation_kinds[decision.when_false.kind], target(decision.when_false), 0,
                _FEATURES.index(decision.predicate.feature),
                {"EQ": 0, "NE": 1, "LE": 2, "GE": 3}[decision.predicate.comparator],
                decision.predicate.index, decision.predicate.value,
            ))
        if decision_first > 255:
            raise ValueError("device batch decision table exceeds v2 record capacity")
        initiation_word = item.initiation.step
        max_steps_word = item.max_total_steps
        if clauses is not None:
            initiation_word |= len(clauses) << 10
            max_steps_word |= gate_count << 10
        semantic_id = item.semantic_id if semantic_ids is None else semantic_ids[program_index]
        program_records.append(_PROGRAM_RECORD.pack(
            initiation_word, max_steps_word, phase_first, len(item.phases),
            list(type(item.lane)).index(item.lane) | (0x80 if item.recovery_phase is not None else 0),
            decision_first, len(item.decisions),
            bytes.fromhex(semantic_id),
        ))
    version = 3 if initiation_clauses is not None else (2 if any(item.schema_version == 2 for item in programs) else 1)
    return b"".join((
        _DEVICE_HEADER.pack(0x524C5234, version, len(programs), len(phase_records), len(motif_records), len(actions)),
        *program_records, *phase_records, *decision_records, *motif_records, *actions,
    ))


@dataclass(frozen=True)
class DeviceResult:
    own_money: int
    opp_money: int
    action_changes: int
    fault: FaultCode
    activated: bool
    final_phase: int
    end_reason: TerminationReason
    complete: bool
    semantic_id: str
    first_activation_step: int = -1
    last_activation_step: int = -1
    activation_count: int = 0
    specialist_duration: int = 0
    phase_path: int = 0
    phase_usage: int = 0
    branch_path: int = 0
    completion_reason: CompletionReason = CompletionReason.NONE
    family_farmer_changes: int = 0
    family_hands_changes: int = 0
    family_market_changes: int = 0
    action_trace_hash: int = 0
    state_deltas: tuple[int, ...] = (0,) * 10


def decode_device_results(data: bytes) -> tuple[int, int, tuple[DeviceResult, ...]]:
    if len(data) < _RESULT_HEADER.size:
        raise ValueError("short RLRMB result")
    magic, version, candidates, cases = _RESULT_HEADER.unpack_from(data)
    if magic != b"RLRMBR1\0" or version not in (1, 2):
        raise ValueError("unsupported RLRMB result")
    record = _RESULT_RECORD_V1 if version == 1 else _RESULT_RECORD_V2
    expected = _RESULT_HEADER.size + candidates * cases * record.size
    if len(data) != expected:
        raise ValueError("RLRMB result length mismatch")
    rows = []
    for offset in range(_RESULT_HEADER.size, len(data), record.size):
        values = record.unpack_from(data, offset)
        own, opp, changes, fault, active, phase, reason, complete = values[:8]
        if version == 1:
            rows.append(DeviceResult(own, opp, changes, FaultCode(fault), bool(active), phase, TerminationReason(reason), bool(complete), values[8].hex()))
            continue
        first, last, activations, duration, phase_path, farmer, hands, market = values[8:16]
        action_trace_hash = values[16]
        phase_usage, branch_path, completion, _reserved = values[17:21]
        deltas = tuple(values[21:31])
        rows.append(DeviceResult(
            own_money=own, opp_money=opp, action_changes=changes, fault=FaultCode(fault),
            activated=bool(active), final_phase=phase, end_reason=TerminationReason(reason),
            complete=bool(complete), semantic_id=values[31].hex(),
            first_activation_step=-1 if first == 0xFFFF else first,
            last_activation_step=-1 if last == 0xFFFF else last,
            activation_count=activations, specialist_duration=duration,
            phase_path=phase_path, phase_usage=phase_usage, branch_path=branch_path,
            completion_reason=CompletionReason(completion), family_farmer_changes=farmer,
            family_hands_changes=hands, family_market_changes=market,
            action_trace_hash=action_trace_hash, state_deltas=deltas,
        ))
    return candidates, cases, tuple(rows)


def _shops(observation: Mapping[str, Any]) -> Counter[str]:
    return Counter(observation.get("town", {}).get("unlocked_shops", ()))


def _own_farm(observation: Mapping[str, Any]) -> Mapping[str, Any]:
    return observation["farms"][observation["player"]]


def _opp_farm(observation: Mapping[str, Any]) -> Mapping[str, Any]:
    return observation["farms"][1 - observation["player"]]


_ITEMS = ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL", "FERTILIZER", "GOOSE", "COW", "SHEEP")
_SEEDS = _ITEMS[:5]
_SHOPS = ("BAKERY", "BRUNCH_SPOT", "FARMERS_MARKET", "ICE_CREAM_SHOP", "PET_CAFE", "PIZZA_SHOP", "SMOOTHIE_SHOP", "YARN_STORE")
_SHOP_IDS = {name: index for index, name in enumerate(_SHOPS)}
_SHOP_IDS.update({"BAKERY_SHOP": 0, "BRUNCH": 1, "ICE_CREAM": 3, "PIZZA": 5, "SMOOTHIE": 6, "YARN": 7})


def _public_feature(condition: PublicCondition, observation: Mapping[str, Any]) -> int:
    own, opp = _own_farm(observation), _opp_farm(observation)
    direct = {
        "STEP": observation["step"], "DAY": observation["day"], "HOUR": observation["hour"],
        "OWN_MONEY": own["money"], "OPP_MONEY": opp["money"],
        "OWN_UNITS": 1 + len(own.get("hands", ())), "OPP_UNITS": 1 + len(opp.get("hands", ())),
        "OWN_LAND": len(own.get("unlocked_quadrants", ())), "OPP_LAND": len(opp.get("unlocked_quadrants", ())),
        "OWN_HIRES_TODAY": own.get("hires_today", 0), "OPP_HIRES_TODAY": opp.get("hires_today", 0),
    }
    if condition.feature in direct:
        return int(direct[condition.feature])
    index = condition.index
    if condition.feature == "OWN_SEED":
        return int(observation["private"].get("seeds", {}).get(_SEEDS[index], 0))
    if condition.feature == "OWN_SHED":
        return int(observation["private"].get("shed", {}).get(_ITEMS[index], 0))
    if condition.feature in ("MARKET_PRICE", "MARKET_INVENTORY"):
        key = "prices" if condition.feature == "MARKET_PRICE" else "inventory"
        return 0 if index >= 9 else int(observation["market"].get(key, {}).get(_ITEMS[index], 0))
    if condition.feature == "SHOP_AT":
        shops = observation["town"].get("unlocked_shops", ())
        return -1 if index >= len(shops) else _SHOP_IDS.get(str(shops[index]), -1)
    if condition.feature == "SHOP_SEEN":
        return sum(1 for item in observation["town"].get("unlocked_shops", ()) if _SHOP_IDS.get(str(item), -1) == index)
    worker_features = {"OWN_WORKER_X", "OWN_WORKER_Y", "OPP_WORKER_X", "OPP_WORKER_Y"}
    if condition.feature in worker_features:
        farm = own if condition.feature.startswith("OWN") else opp
        positions = [farm.get("farmer", (-1, -1)), *farm.get("hands", ())]
        if index >= len(positions):
            return -1
        return int(positions[index][0 if condition.feature.endswith("_X") else 1])
    tile_features = {"OWN_TILE_KIND", "OPP_TILE_KIND", "OWN_TILE_ITEM", "OPP_TILE_ITEM", "OWN_TILE_YIELD", "OPP_TILE_YIELD"}
    if condition.feature in tile_features:
        farm = own if condition.feature.startswith("OWN") else opp
        tiles = farm.get("tiles", ())
        y, x = divmod(index, 10)
        if y >= len(tiles) or x >= len(tiles[y]):
            return 0
        tile = tiles[y][x]
        if condition.feature.endswith("_YIELD"):
            return int(tile.get("yield_units", 0)) if isinstance(tile, Mapping) else 0
        if condition.feature.endswith("_KIND"):
            names = {None: 0, "LOCKED": 1, "WEED": 2, "COOP": 3, "PASTURE": 4, "PLANT": 5}
            kind = tile.get("kind") if isinstance(tile, Mapping) else tile
            return names.get(kind, 0)
        if not isinstance(tile, Mapping):
            return -1
        name = tile.get("crop", tile.get("animal"))
        return _ITEMS.index(name) if name in _ITEMS else -1
    raise ValueError(f"unsupported public feature: {condition.feature}")


def _condition_matches(condition: PublicCondition, observation: Mapping[str, Any]) -> bool:
    lhs = _public_feature(condition, observation)
    return {"EQ": lhs == condition.value, "NE": lhs != condition.value, "LE": lhs <= condition.value, "GE": lhs >= condition.value}[condition.comparator]


def _purchase_metric(objective: str, observation: Mapping[str, Any]) -> int:
    own = _own_farm(observation)
    if objective == "BUY_LAND":
        return len(own.get("unlocked_quadrants", ()))
    if objective == "HIRE":
        return int(own.get("hires_today", 0))
    raise ValueError(f"unsupported purchase objective: {objective}")


@dataclass
class ReferenceInterpreter:
    """State machine driven by exact-engine pre/post observations.

    ``action`` runs on the current legal pre-action observation. ``observe`` runs
    exactly once after that action has been submitted to the engine. A triggering
    action belongs to the current phase; phase advancement affects the next turn.
    """

    program: MacroProgram
    motifs: Mapping[str, Motif]
    initiation_clauses: Sequence[Sequence[PublicCondition]] | None = None

    def __post_init__(self) -> None:
        self.active = False
        self.complete = False
        self.phase_index = 0
        self.phase_steps = 0
        self.program_steps = 0
        self.action_changes = 0
        self.fault = FaultCode.NONE
        self.last_termination = TerminationReason.NONE
        self._entry_day = 0
        self._entry_shops: Counter[str] = Counter()
        self._entry_purchase = 0
        self._pending_compatible = False
        self._had_action = False
        self._in_recovery = False
        self.first_activation_step = -1
        self.last_activation_step = -1
        self.activation_count = 0
        self.phase_path = 0
        self.phase_usage = 0
        self.branch_path = 0
        self.completion_reason = CompletionReason.NONE
        self.family_farmer_changes = 0
        self.family_hands_changes = 0
        self.family_market_changes = 0
        self.action_trace_hash = 0
        self.activation_observation: Mapping[str, Any] | None = None

    @property
    def phase(self) -> Phase:
        if self._in_recovery:
            if self.program.recovery_phase is None:
                raise ValueError("active recovery has no recovery phase")
            return self.program.recovery_phase
        return self.program.phases[self.phase_index]

    def _enter(self, observation: Mapping[str, Any]) -> None:
        self.active = True
        self.phase_steps = 0
        self._entry_day = int(observation["day"])
        self._entry_shops = _shops(observation)
        term = self.phase.termination
        self._entry_purchase = _purchase_metric(term.objective, observation) if term.objective else 0
        step = int(observation["step"])
        if self.first_activation_step < 0:
            self.first_activation_step = step
            self.activation_observation = observation
            self.action_trace_hash = 1469598103934665603
        self.last_activation_step = step
        self.activation_count += 1
        phase_code = len(self.program.phases) if self._in_recovery else self.phase_index
        self.phase_usage |= 1 << phase_code
        self.phase_path = (self.phase_path << 3) | (phase_code + 1)

    def _source_action(self) -> dict[str, Any]:
        phase = self.phase
        if phase.action_source is ActionSource.PARENT:
            return PASS
        if phase.action_source in (ActionSource.DONOR_BLOCK, ActionSource.SEMANTIC_SPLICE):
            motif = self.motifs.get(phase.motif_id)
            if motif is None:
                self.fault = FaultCode.INVALID_MOTIF_REFERENCE
                return PASS
            if not set(phase.action_families).issubset(motif.supported_families):
                self.fault = FaultCode.INVALID_MOTIF_REFERENCE
                return PASS
            block = motif.semantic_document()["action_block"]
            if self.phase_steps >= len(block):
                self.fault = FaultCode.MOTIF_EXHAUSTED
                return PASS
            return canonical_action(block[self.phase_steps])
        if phase.primitive_name == "LIQUIDITY_REBALANCE":
            params = dict(phase.parameters)
            if set(params) != {"enabled", "mode", "quantity"} or type(params["enabled"]) is not bool or type(params["quantity"]) is not int:
                self.fault = FaultCode.UNSUPPORTED_PRIMITIVE
                return PASS
            return {"farmer": ["PASS"], "hands": [], "market": [["BUY_PRODUCT", params["mode"], params["quantity"]]] if params["enabled"] else []}
        self.fault = FaultCode.UNSUPPORTED_PRIMITIVE
        return PASS

    def action(self, observation: Mapping[str, Any], parent_action: Mapping[str, Any]) -> dict[str, Any]:
        parent = canonical_action(parent_action)
        step = int(observation["step"])
        if self.fault is not FaultCode.NONE or self.complete or step < self.program.initiation.step:
            return parent
        if not self.active:
            if not self._had_action and self.initiation_clauses is not None:
                matched = any(
                    all(_condition_matches(condition, observation) for condition in clause)
                    for clause in self.initiation_clauses
                )
                if not matched:
                    self.complete = True
                    return parent
            if step != self.program.initiation.step and not self._had_action:
                self.fault = FaultCode.HORIZON_VIOLATION
                return parent
            self._enter(observation)
        source = parent if self.phase.action_source is ActionSource.PARENT else self._source_action()
        if self.fault is not FaultCode.NONE:
            return parent
        candidate = merge_action_families(parent, source, self.phase.action_families)
        self.action_trace_hash = _trace_action(self.action_trace_hash, candidate)
        if candidate["farmer"] != parent["farmer"]:
            self.family_farmer_changes += 1
        if candidate["hands"] != parent["hands"]:
            self.family_hands_changes += 1
        if candidate["market"] != parent["market"]:
            self.family_market_changes += 1
        self._pending_compatible = candidate == parent
        self._had_action = True
        self.last_activation_step = step
        if candidate != parent:
            self.action_changes += 1
        return candidate

    def observe(self, post_observation: Mapping[str, Any]) -> None:
        if not self.active or self.fault is not FaultCode.NONE or not self._had_action:
            return
        self.phase_steps += 1
        self.program_steps += 1
        term = self.phase.termination
        reason = TerminationReason.NONE
        if term.kind is TerminationType.NEXT_EOD and int(post_observation["day"]) > self._entry_day:
            reason = TerminationReason.NEXT_EOD
        elif term.kind is TerminationType.NEXT_SHOP_REVEAL and _shops(post_observation) - self._entry_shops:
            reason = TerminationReason.NEXT_SHOP_REVEAL
        elif term.kind is TerminationType.PURCHASE_COMPLETED and _purchase_metric(term.objective, post_observation) >= self._entry_purchase + term.target_count:
            reason = TerminationReason.PURCHASE_COMPLETED
        elif term.kind is TerminationType.TARGET_PUBLIC_STATE:
            try:
                matched = _condition_matches(term.condition, post_observation)
            except ValueError:
                self.fault = FaultCode.INVALID_FEATURE
                return
            if matched:
                reason = TerminationReason.TARGET_PUBLIC_STATE
        elif term.kind is TerminationType.PARENT_COMPATIBLE and self._pending_compatible:
            reason = TerminationReason.PARENT_COMPATIBLE
        if reason is TerminationReason.NONE and self.phase_steps >= term.max_steps:
            reason = TerminationReason.MAX_STEPS if term.kind is TerminationType.MAX_STEPS else TerminationReason.FIXED_CAP
        if reason is TerminationReason.NONE and self.program_steps >= self.program.max_total_steps:
            reason = TerminationReason.PROGRAM_CAP
        if reason is TerminationReason.NONE:
            return
        self.last_termination = reason
        if self._in_recovery or self.program_steps >= self.program.max_total_steps:
            self.active = False
            self.complete = True
            self.completion_reason = CompletionReason.RECOVERY if self._in_recovery else CompletionReason.PROGRAM_CAP
            return
        decision = next((item for item in self.program.decisions if item.after_phase == self.phase_index), None)
        if decision is not None:
            try:
                matched = _condition_matches(decision.predicate, post_observation)
            except (IndexError, KeyError, TypeError, ValueError):
                self.fault = FaultCode.INVALID_FEATURE
                return
            continuation = decision.when_true if matched else decision.when_false
            self.branch_path = (self.branch_path << 2) | (1 if matched else 2)
            if continuation.kind is ContinuationKind.ABORT_TO_PARENT:
                self.active = False
                self.complete = True
                self.completion_reason = CompletionReason.ABORT
                return
            if continuation.kind is ContinuationKind.RECOVERY:
                if self.program.recovery_phase is None:
                    self.fault = FaultCode.INVALID_RECOVERY
                    return
                self._in_recovery = True
                self.active = False
                self._had_action = True
                return
            if continuation.kind is ContinuationKind.PHASE:
                if continuation.phase_index is None or continuation.phase_index <= self.phase_index or continuation.phase_index >= len(self.program.phases):
                    self.fault = FaultCode.INVALID_CONTINUATION
                    return
                self.phase_index = continuation.phase_index
                self.active = False
                self._had_action = True
                return
            self.fault = FaultCode.INVALID_CONTINUATION
            return
        if self.phase_index + 1 == len(self.program.phases):
            self.active = False
            self.complete = True
            self.completion_reason = CompletionReason.FINISHED
            return
        self.phase_index += 1
        self.active = False
        self._had_action = True


PASS = {"farmer": ["PASS"], "hands": [], "market": []}


@dataclass(frozen=True)
class ReferenceResult:
    own_money: int
    opp_money: int
    action_changes: int
    fault: FaultCode
    activated: bool
    final_phase: int
    end_reason: TerminationReason
    complete: bool
    first_activation_step: int = -1
    last_activation_step: int = -1
    activation_count: int = 0
    specialist_duration: int = 0
    phase_path: int = 0
    phase_usage: int = 0
    branch_path: int = 0
    completion_reason: CompletionReason = CompletionReason.NONE
    family_farmer_changes: int = 0
    family_hands_changes: int = 0
    family_market_changes: int = 0
    action_trace_hash: int = 0
    state_deltas: tuple[int, ...] = (0,) * 10


def _strategic_snapshot(observation: Mapping[str, Any]) -> tuple[int, ...]:
    own, opp = _own_farm(observation), _opp_farm(observation)
    private = observation.get("private", {})
    return (
        int(own["money"]), int(opp["money"]), 1 + len(own.get("hands", ())),
        1 + len(opp.get("hands", ())), len(own.get("unlocked_quadrants", ())),
        len(opp.get("unlocked_quadrants", ())), int(own.get("hires_today", 0)),
        int(opp.get("hires_today", 0)), sum(map(int, private.get("shed", {}).values())),
        sum(map(int, private.get("seeds", {}).values())),
    )


def _trace_action(value: int, action: Mapping[str, Any]) -> int:
    canonical = canonical_action(action)

    def word(number: int) -> None:
        nonlocal value
        for byte in int(number).to_bytes(4, "little", signed=True):
            value = ((value ^ byte) * 1099511628211) & ((1 << 64) - 1)

    units = [canonical["farmer"], *canonical["hands"]]
    word(len(units))
    for unit in units:
        word(_UNIT_OPS.index(unit[0]))
        word(_ITEMS.index(unit[1]) if len(unit) > 1 and unit[1] in _ITEMS else 0)
        word(int(unit[2]) if len(unit) > 2 else 1)
    word(len(canonical["market"]))
    for order in canonical["market"]:
        word(_MARKET_OPS.index(order[0]))
        word(_ITEMS.index(order[1]) if len(order) > 1 and order[1] in _ITEMS else 0)
        word(int(order[2]) if len(order) > 2 else 1)
    return value


def evaluate_reference_case(
    program: MacroProgram,
    motifs: Mapping[str, Motif],
    *,
    opponent_tape: str,
    seed: int,
    seat: int,
    asset_dir: str,
    initiation_clauses: Sequence[Sequence[PublicCondition]] | None = None,
) -> ReferenceResult:
    """Run the interpreter over parity-validated C++ physics and CPU BKG4.

    The Python layer only selects/merges actions and advances the IR state
    machine. All game transitions and observations come from the native C++ oracle.
    """
    from kaggriculture_bonsai.evaluation.cpp_oracle import run_vs_tape
    from kaggriculture_bonsai.runtime import champion as champion_runtime

    parent = champion_runtime.BKG4ChampionAgent(asset_dir)
    interpreter = ReferenceInterpreter(program, motifs, initiation_clauses)
    had_action = False

    def candidate(observation: Mapping[str, Any], configuration: Any) -> dict[str, Any]:
        nonlocal had_action
        if had_action:
            interpreter.observe(observation)
        step = int(observation.get("step", 0) or 0)
        if step == 0 or step <= parent.last_step:
            parent.reset()
        parent.last_step = step
        tick = parent.vm.tick(observation, max_market_orders=10, trace=False)
        if tick.fault or tick.status != champion_runtime.bkg4.STATUS_SUCCESS:
            raise champion_runtime.CompatibilityError("CPU BKG4 controller fault")
        route = int(parent.vm.state.blackboard[1])
        if parent.route_override_slot is not None:
            override = int(parent.vm.state.blackboard[parent.route_override_slot])
            if override:
                route = override if parent.route_override_value else int(parent.route_override_id)
        hands = parent._hands(observation)
        base_action = parent.base_pack.action(route, step, hands)
        action = interpreter.action(observation, base_action)
        action = champion_runtime.apply_legacy_overlay(observation, action, parent.legacy_pack, parent.legacy, hands)
        action = champion_runtime.apply_engine_exploits(
            observation, configuration, action, parent.exploits,
            external_ok=parent.legacy.external_ok,
        )
        if parent.feed_defer_slots:
            action = champion_runtime.apply_feed_deferral(action, parent.vm.state.blackboard, parent.feed_defer_slots)
        had_action = True
        return action

    rewards, _, final = run_vs_tape(candidate, opponent_tape, int(seed), int(seat))
    if had_action:
        interpreter.observe(final[seat].observation)
    state_deltas = (0,) * 10
    if interpreter.activation_observation is not None:
        before = _strategic_snapshot(interpreter.activation_observation)
        after = _strategic_snapshot(final[seat].observation)
        state_deltas = tuple(max(-32768, min(32767, x - y)) for x, y in zip(after, before))
    return ReferenceResult(
        int(rewards[seat]), int(rewards[1-seat]), interpreter.action_changes,
        interpreter.fault, interpreter._had_action, interpreter.phase_index,
        interpreter.last_termination, interpreter.complete,
        interpreter.first_activation_step, interpreter.last_activation_step,
        interpreter.activation_count, interpreter.program_steps, interpreter.phase_path,
        interpreter.phase_usage, interpreter.branch_path,
        CompletionReason.FAULT if interpreter.fault else interpreter.completion_reason,
        interpreter.family_farmer_changes, interpreter.family_hands_changes,
        interpreter.family_market_changes, interpreter.action_trace_hash, state_deltas,
    )
