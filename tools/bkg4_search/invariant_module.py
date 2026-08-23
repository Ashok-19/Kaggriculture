from __future__ import annotations

import dataclasses
import copy
import hashlib
import json
import struct
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
BKG_PY = ROOT / "bonsai-kg" / "python"
if str(BKG_PY) not in sys.path:
    sys.path.insert(0, str(BKG_PY))
import bonsai_kg_v4 as bkg  # noqa: E402

GRAMMAR_ID = "bkg4_invariant_strategic_module_v1"
B4IM1_MAGIC = b"B4IM"
B4IM1_VERSION = 1
B4IM1_CANDIDATE_SIZE = 256
MAX_RULES = 3
MAX_GUARDS_TOTAL = 5

_DOMAINS = {"GLOBAL": 0, "EACH_OWN_UNIT": 1}
_DOMAIN_NAMES = {v: k for k, v in _DOMAINS.items()}
_CONTROLS = {"INSTANT": 0, "COMMITTED": 1}
_CONTROL_NAMES = {v: k for k, v in _CONTROLS.items()}
_ACTIONS = {"NOOP": 0, "ROUTE_OVERRIDE": 1, "UNIT_REWRITE": 2, "MARKET_INTENT": 3, "COMMITTED_ROUTE": 4}
_ACTION_NAMES = {v: k for k, v in _ACTIONS.items()}
_EFFECTS = {
    "NONE": 0,
    "ROUTE_MODE": 1,
    "PRESERVE_PRODUCTION_SAVE_FEED": 2,
    "CREATE_FINANCING_RUNWAY": 3,
    "ACCELERATE_EXPANSION": 4,
    "REDUCE_MAINTENANCE_BACKLOG": 5,
    "RECOVER_RESOURCE_INVARIANT": 6,
    "AVOID_OVERBUY": 7,
    "ENGINE_MECHANIC": 8,
}
_EFFECT_NAMES = {v: k for k, v in _EFFECTS.items()}
_EFFECT_DOMAINS = {
    "NONE": "generic",
    "ROUTE_MODE": "route",
    "PRESERVE_PRODUCTION_SAVE_FEED": "animal_lifecycle",
    "CREATE_FINANCING_RUNWAY": "market",
    "ACCELERATE_EXPANSION": "expansion",
    "REDUCE_MAINTENANCE_BACKLOG": "maintenance",
    "RECOVER_RESOURCE_INVARIANT": "recovery",
    "AVOID_OVERBUY": "market",
    "ENGINE_MECHANIC": "engine_mechanic",
}
_CMP = {"EQ": 0, "NE": 1, "LT": 2, "LE": 3, "GT": 4, "GE": 5}
_CMP_NAMES = {v: k for k, v in _CMP.items()}
_AUTH_NAMES = {v: k for k, v in bkg.AUTHORITY.items()}
_UNIT_OP_NAMES = {v: k for k, v in bkg.UNIT_OPS.items()}
_MARKET_OP_NAMES = {v: k for k, v in bkg.MARKET_OPS.items()}
_ITEM_NAMES = {v: k for k, v in bkg.ITEM_ID.items()}

_HEADER = struct.Struct("<4sBBHQ")  # 16 bytes
_RULE = struct.Struct("<8B5h2x")   # 20 bytes
_GUARD = struct.Struct("<HBBii")   # 12 bytes; 5 per rule -> 80 bytes/rule
assert _HEADER.size + MAX_RULES * (_RULE.size + MAX_GUARDS_TOTAL * _GUARD.size) == B4IM1_CANDIDATE_SIZE

_ITEM_ARG_FEATURES = {"MARKET_PRICE", "MARKET_INVENTORY", "OWN_SHED", "OWN_SEED", "OWN_INVENTORY"}
_UNIT_ARG_FEATURES = {"OWN_WORKER_X", "OWN_WORKER_Y", "OPP_WORKER_X", "OPP_WORKER_Y", "OWN_WORKER_TILE", "OPP_WORKER_TILE"}
_TILE_ARG_FEATURES = {
    name for name in bkg.FEATURES
    if name.startswith("OWN_TILE_") or name.startswith("OPP_TILE_")
}
_HIGH_CARD_EQ = {"OWN_MONEY", "OPP_MONEY", "MARKET_PRICE", "MARKET_INVENTORY"}


def _enum_value(table: Mapping[str, int], name: str, label: str) -> int:
    if name not in table:
        raise ValueError(f"unknown {label} {name}")
    return int(table[name])


@dataclass(frozen=True)
class ModuleSketch:
    domain: str
    control_pattern: str
    effect_class: str

    def validate(self) -> None:
        if self.control_pattern not in _CONTROLS:
            raise ValueError(f"unknown control pattern {self.control_pattern}")
        if self.effect_class not in _EFFECTS:
            raise ValueError(f"unknown effect class {self.effect_class}")
        if not self.domain or len(self.domain) > 64:
            raise ValueError("invalid sketch domain")


@dataclass(frozen=True)
class GuardAtom:
    feature: str
    cmp: str
    value: int
    arg: int = 0

    def validate(self, *, lifted_domain: str) -> None:
        if self.feature not in bkg.FEATURES:
            raise ValueError(f"feature-registry-v1 has no {self.feature}")
        if self.cmp not in _CMP:
            raise ValueError(f"invalid comparator {self.cmp}")
        if not -(1 << 31) <= int(self.value) < (1 << 31):
            raise ValueError("guard value must fit int32")
        if not -(1 << 31) <= int(self.arg) < (1 << 31):
            raise ValueError("guard arg must fit int32")
        if lifted_domain == "EACH_OWN_UNIT" and self.feature in _TILE_ARG_FEATURES:
            if int(self.arg) != 0:
                raise ValueError("lifted tile guard cannot carry a concrete tile id")
            return
        if self.feature in _UNIT_ARG_FEATURES and not 0 <= int(self.arg) < 40:
            raise ValueError("unit feature arg out of range")
        if self.feature in _TILE_ARG_FEATURES and not 0 <= int(self.arg) < 100:
            raise ValueError("tile feature arg out of range")
        if self.feature in _ITEM_ARG_FEATURES and not 0 <= int(self.arg) < 12:
            raise ValueError("item feature arg out of range")


@dataclass(frozen=True)
class LiftedGuard:
    atoms: tuple[GuardAtom, ...] = field(default_factory=tuple)

    def validate(self, *, lifted_domain: str) -> None:
        if len(self.atoms) > MAX_GUARDS_TOTAL:
            raise ValueError(f"guard exceeds {MAX_GUARDS_TOTAL} atoms")
        if len(set(self.atoms)) != len(self.atoms):
            raise ValueError("duplicate guard atom")
        for atom in self.atoms:
            atom.validate(lifted_domain=lifted_domain)


@dataclass(frozen=True)
class AbstractAction:
    kind: str
    route_id: int = 0
    match_op: int = 0
    unit_op: int = 0
    unit_arg: int = 0
    quantity: int = 1
    market_op: int = 0
    item: int = 0
    authority: int = 3
    priority: int = 0

    def validate(self) -> None:
        if self.kind not in _ACTIONS:
            raise ValueError(f"unknown abstract action {self.kind}")
        if self.kind in ("ROUTE_OVERRIDE", "COMMITTED_ROUTE") and not 0 <= int(self.route_id) <= 65535:
            raise ValueError("route_id out of range")
        if self.kind == "UNIT_REWRITE":
            if int(self.match_op) not in _UNIT_OP_NAMES or int(self.unit_op) not in _UNIT_OP_NAMES:
                raise ValueError("unit rewrite op out of range")
            if not -32768 <= int(self.quantity) <= 32767:
                raise ValueError("unit quantity out of int16 range")
        if self.kind == "MARKET_INTENT":
            if int(self.market_op) not in _MARKET_OP_NAMES:
                raise ValueError("market op out of range")
            if not 0 <= int(self.item) < 12:
                raise ValueError("market item out of range")
            if not -32768 <= int(self.quantity) <= 32767:
                raise ValueError("market quantity out of int16 range")
            if int(self.authority) not in _AUTH_NAMES:
                raise ValueError("authority out of range")
            if not 0 <= int(self.priority) <= 255:
                raise ValueError("priority out of range")


@dataclass(frozen=True)
class InvariantRule:
    sketch: ModuleSketch
    domain: str
    guard: LiftedGuard
    action: AbstractAction
    exit_guard: LiftedGuard = field(default_factory=LiftedGuard)

    def validate(self, *, production: bool, multi_tape_support: int) -> None:
        self.sketch.validate()
        if self.domain not in _DOMAINS:
            raise ValueError(f"unknown lifted domain {self.domain}")
        self.guard.validate(lifted_domain=self.domain)
        self.exit_guard.validate(lifted_domain=self.domain)
        if len(self.guard.atoms) + len(self.exit_guard.atoms) > MAX_GUARDS_TOTAL:
            raise ValueError(f"entry+exit guard exceeds {MAX_GUARDS_TOTAL} atoms")
        self.action.validate()
        if self.sketch.control_pattern == "COMMITTED" and not self.exit_guard.atoms:
            raise ValueError("committed control pattern requires exit guard")
        if self.action.kind == "COMMITTED_ROUTE" and self.sketch.control_pattern != "COMMITTED":
            raise ValueError("COMMITTED_ROUTE requires COMMITTED control pattern")
        if production:
            if int(multi_tape_support) < 2:
                raise ValueError("production invariant requires multi-tape support")
            for atom in (*self.guard.atoms, *self.exit_guard.atoms):
                if atom.feature == "STEP" and atom.cmp == "EQ" and self.sketch.effect_class != "ENGINE_MECHANIC":
                    raise ValueError("production invariant rejects exact STEP patch")
                if atom.feature in _HIGH_CARD_EQ and atom.cmp == "EQ":
                    raise ValueError("production invariant rejects high-cardinality exact equality")
                if self.domain == "GLOBAL" and atom.feature in (_UNIT_ARG_FEATURES | _TILE_ARG_FEATURES):
                    raise ValueError("production invariant rejects concrete unit/tile identity")


@dataclass(frozen=True)
class InvariantStrategicModule:
    rules: tuple[InvariantRule, ...] = field(default_factory=tuple)

    def validate(self, *, production: bool = False, multi_tape_support: int = 0) -> None:
        if len(self.rules) > MAX_RULES:
            raise ValueError(f"module exceeds {MAX_RULES} rules")
        for rule in self.rules:
            rule.validate(production=production, multi_tape_support=multi_tape_support)

    def canonical(self) -> dict[str, Any]:
        return {
            "grammar_id": GRAMMAR_ID,
            "schema_version": 1,
            "rules": [
                {
                    "sketch": dataclasses.asdict(r.sketch),
                    "domain": r.domain,
                    "guard": {"atoms": [dataclasses.asdict(a) for a in r.guard.atoms]},
                    "exit_guard": {"atoms": [dataclasses.asdict(a) for a in r.exit_guard.atoms]},
                    "action": dataclasses.asdict(r.action),
                }
                for r in self.rules
            ],
        }

    @classmethod
    def from_dict(cls, obj: Mapping[str, Any]) -> "InvariantStrategicModule":
        gid = obj.get("grammar_id", GRAMMAR_ID)
        if gid != GRAMMAR_ID:
            raise ValueError(f"incompatible invariant grammar {gid!r}")
        if int(obj.get("schema_version", 1)) != 1:
            raise ValueError("incompatible invariant module schema")
        rules = []
        for row in obj.get("rules", []):
            sketch = ModuleSketch(**dict(row["sketch"]))
            guard = LiftedGuard(tuple(GuardAtom(**dict(x)) for x in row.get("guard", {}).get("atoms", [])))
            exit_guard = LiftedGuard(tuple(GuardAtom(**dict(x)) for x in row.get("exit_guard", {}).get("atoms", [])))
            action = AbstractAction(**dict(row["action"]))
            rules.append(InvariantRule(sketch, str(row["domain"]), guard, action, exit_guard))
        out = cls(tuple(rules))
        out.validate()
        return out

    def semantic_id(self) -> str:
        raw = json.dumps(self.canonical(), sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(b"BKG4-INVARIANT-STRATEGIC-MODULE-V1\0" + raw).hexdigest()


@dataclass
class InvariantRuntimeState:
    committed_mask: int = 0

    def reset(self) -> None:
        self.committed_mask = 0


@dataclass(frozen=True)
class InvariantExecutionStats:
    activations: int = 0
    action_changes: int = 0


def _compare_value(lhs: int, rhs: int, cmp: str) -> bool:
    if cmp == "EQ": return int(lhs) == int(rhs)
    if cmp == "NE": return int(lhs) != int(rhs)
    if cmp == "LT": return int(lhs) < int(rhs)
    if cmp == "LE": return int(lhs) <= int(rhs)
    if cmp == "GT": return int(lhs) > int(rhs)
    if cmp == "GE": return int(lhs) >= int(rhs)
    raise ValueError(f"invalid comparator {cmp}")


def _reference_atom_value(observation: Any, atom: GuardAtom, domain: str, unit: int | None) -> int:
    if domain == "EACH_OWN_UNIT" and atom.feature in _TILE_ARG_FEATURES:
        if unit is None:
            raise ValueError("lifted tile feature requires bound unit")
        tile = int(observation.feature("OWN_WORKER_TILE", int(unit)))
        return int(observation.feature(atom.feature, tile))
    if domain == "EACH_OWN_UNIT" and atom.feature in _UNIT_ARG_FEATURES:
        if unit is None:
            raise ValueError("lifted unit feature requires bound unit")
        return int(observation.feature(atom.feature, int(unit)))
    if domain == "EACH_OWN_UNIT" and atom.feature == "OWN_INVENTORY":
        if unit is None:
            raise ValueError("lifted inventory feature requires bound unit")
        return int(observation.feature(atom.feature, int(unit), int(atom.arg)))
    return int(observation.feature(atom.feature, int(atom.arg)))


def _reference_guard_matches(observation: Any, guard: LiftedGuard, domain: str, unit: int | None = None) -> bool:
    return all(_compare_value(_reference_atom_value(observation, atom, domain, unit), atom.value, atom.cmp) for atom in guard.atoms)


def apply_reference_module(
    module: InvariantStrategicModule,
    observation: Any,
    action: dict[str, Any],
    state: InvariantRuntimeState,
    *,
    max_market_orders: int = 10,
) -> tuple[dict[str, Any], InvariantExecutionStats]:
    """Authoritative pure-Python executable semantics for one B4IM1 module tick.

    The caller supplies a legal-feature adapter with `feature(name, arg)`. `action` is the
    already planned route action represented by integer triples `[op,arg,n]` for units and
    market orders. Route actions may additionally carry a synthetic `route` field for the
    reference route-mode tests; the resident engines apply the returned route override before
    materialization and unit/market transformations afterwards.
    """
    module.validate()
    before = json.dumps(action, sort_keys=True, separators=(",", ":"))
    activations = 0
    units = action.setdefault("units", [])
    market = action.setdefault("market", [])
    pending_market: list[tuple[int, int, list[int]]] = []
    for ri, rule in enumerate(module.rules):
        a = rule.action
        if rule.domain == "GLOBAL":
            active = _reference_guard_matches(observation, rule.guard, rule.domain)
            if a.kind == "COMMITTED_ROUTE":
                bit = 1 << ri
                if active:
                    state.committed_mask |= bit
                if (state.committed_mask & bit) and _reference_guard_matches(observation, rule.exit_guard, rule.domain):
                    state.committed_mask &= ~bit
                if state.committed_mask & bit:
                    action["route"] = int(a.route_id)
                    activations += 1
            elif not active:
                continue
            elif a.kind == "ROUTE_OVERRIDE":
                action["route"] = int(a.route_id)
                activations += 1
            elif a.kind == "MARKET_INTENT":
                key = (int(a.authority) << 8) | int(a.priority)
                pending_market.append((key, ri, [int(a.market_op), int(a.item), int(a.quantity)]))
                activations += 1
            elif a.kind == "NOOP":
                continue
            else:
                raise ValueError(f"action {a.kind} requires lifted unit domain")
            continue

        if rule.domain != "EACH_OWN_UNIT":
            raise ValueError(f"unsupported reference lifted domain {rule.domain}")
        if a.kind != "UNIT_REWRITE":
            raise ValueError(f"lifted unit domain does not support {a.kind}")
        n_units = min(len(units), max(0, int(observation.feature("OWN_UNITS", 0))))
        matched_rule = False
        for unit in range(n_units):
            if not _reference_guard_matches(observation, rule.guard, rule.domain, unit):
                continue
            matched_rule = True
            row = units[unit]
            if isinstance(row, tuple):
                row = list(row); units[unit] = row
            if not isinstance(row, list) or not row:
                continue
            if int(row[0]) != int(a.match_op):
                continue
            while len(row) < 3:
                row.append(1 if len(row) == 2 else 0)
            row[0] = int(a.unit_op)
            row[1] = int(a.unit_arg)
            row[2] = int(a.quantity)
        if matched_rule:
            activations += 1
    if pending_market:
        pending_market.sort(key=lambda x: (-x[0], x[1]))
        inserted = [x[2] for x in pending_market]
        market[:] = (inserted + market)[: max(0, int(max_market_orders))]
    after = json.dumps(action, sort_keys=True, separators=(",", ":"))
    return action, InvariantExecutionStats(activations=activations, action_changes=int(before != after))


def _action_params(action: AbstractAction) -> tuple[int, int, int, int, int]:
    if action.kind in ("ROUTE_OVERRIDE", "COMMITTED_ROUTE"):
        return int(action.route_id), 0, 0, 0, 0
    if action.kind == "UNIT_REWRITE":
        return int(action.match_op), int(action.unit_op), int(action.unit_arg), int(action.quantity), 0
    if action.kind == "MARKET_INTENT":
        return int(action.market_op), int(action.item), int(action.quantity), int(action.authority), int(action.priority)
    return (0, 0, 0, 0, 0)


def pack_candidate(module: InvariantStrategicModule) -> bytes:
    module.validate()
    raw = bytearray(_HEADER.pack(B4IM1_MAGIC, B4IM1_VERSION, len(module.rules), 0, 0))
    for ri in range(MAX_RULES):
        if ri < len(module.rules):
            r = module.rules[ri]
            guards = (*r.guard.atoms, *r.exit_guard.atoms)
            params = _action_params(r.action)
            raw += _RULE.pack(
                _DOMAINS[r.domain], _CONTROLS[r.sketch.control_pattern], _ACTIONS[r.action.kind], len(r.guard.atoms),
                len(r.exit_guard.atoms), _EFFECTS[r.sketch.effect_class], int(r.action.authority) & 0xFF,
                int(r.action.priority) & 0xFF, *params,
            )
            for gi in range(MAX_GUARDS_TOTAL):
                if gi < len(guards):
                    g = guards[gi]
                    if r.domain == "EACH_OWN_UNIT" and g.feature in _TILE_ARG_FEATURES:
                        arg_mode = 1  # bound unit -> current tile -> tile feature
                    elif r.domain == "EACH_OWN_UNIT" and g.feature in _UNIT_ARG_FEATURES:
                        arg_mode = 2  # bound unit directly
                    else:
                        arg_mode = 0
                    raw += _GUARD.pack(int(bkg.FEATURES[g.feature]), _CMP[g.cmp], arg_mode, int(g.arg), int(g.value))
                else:
                    raw += bytes(_GUARD.size)
        else:
            raw += bytes(_RULE.size + MAX_GUARDS_TOTAL * _GUARD.size)
    if len(raw) != B4IM1_CANDIDATE_SIZE:
        raise AssertionError(len(raw))
    return bytes(raw)


def unpack_candidate(raw: bytes) -> InvariantStrategicModule:
    if len(raw) != B4IM1_CANDIDATE_SIZE:
        raise ValueError(f"B4IM1 candidate must be {B4IM1_CANDIDATE_SIZE} bytes")
    magic, version, n_rules, flags, reserved = _HEADER.unpack_from(raw, 0)
    if magic != B4IM1_MAGIC or version != B4IM1_VERSION or flags != 0 or reserved != 0:
        raise ValueError("invalid B4IM1 candidate header")
    if n_rules > MAX_RULES:
        raise ValueError("invalid B4IM1 rule count")
    pos = _HEADER.size
    rules: list[InvariantRule] = []
    feature_names = {int(v): k for k, v in bkg.FEATURES.items()}
    for ri in range(MAX_RULES):
        fields = _RULE.unpack_from(raw, pos); pos += _RULE.size
        domain_id, control_id, action_id, ng, ne, effect_id, authority, priority, *params = fields
        atoms: list[GuardAtom] = []
        for gi in range(MAX_GUARDS_TOTAL):
            fid, cmp_id, arg_mode, arg, value = _GUARD.unpack_from(raw, pos); pos += _GUARD.size
            if ri < n_rules and gi < ng + ne:
                if fid not in feature_names or cmp_id not in _CMP_NAMES:
                    raise ValueError("invalid B4IM1 guard enum")
                # arg_mode is derivable from domain+feature and therefore not part of canonical AST.
                if arg_mode not in (0, 1, 2):
                    raise ValueError("invalid B4IM1 guard arg mode")
                atoms.append(GuardAtom(feature_names[fid], _CMP_NAMES[cmp_id], int(value), int(arg)))
        if ri >= n_rules:
            continue
        if domain_id not in _DOMAIN_NAMES or control_id not in _CONTROL_NAMES or action_id not in _ACTION_NAMES or effect_id not in _EFFECT_NAMES:
            raise ValueError("invalid B4IM1 rule enum")
        kind = _ACTION_NAMES[action_id]
        p0, p1, p2, p3, p4 = [int(x) for x in params]
        if kind in ("ROUTE_OVERRIDE", "COMMITTED_ROUTE"):
            action = AbstractAction(kind, route_id=p0, authority=int(authority), priority=int(priority))
        elif kind == "UNIT_REWRITE":
            action = AbstractAction(kind, match_op=p0, unit_op=p1, unit_arg=p2, quantity=p3, authority=int(authority), priority=int(priority))
        elif kind == "MARKET_INTENT":
            action = AbstractAction(kind, market_op=p0, item=p1, quantity=p2, authority=p3, priority=p4)
        else:
            action = AbstractAction(kind, authority=int(authority), priority=int(priority))
        effect_name = _EFFECT_NAMES[effect_id]
        rules.append(InvariantRule(
            ModuleSketch(_EFFECT_DOMAINS[effect_name], _CONTROL_NAMES[control_id], effect_name),
            _DOMAIN_NAMES[domain_id],
            LiftedGuard(tuple(atoms[:ng])), action, LiftedGuard(tuple(atoms[ng:ng + ne])),
        ))
    out = InvariantStrategicModule(tuple(rules))
    out.validate()
    return out


def _feature_ref(atom: GuardAtom, *, lifted_domain: str, tile_slot: str | None = None, bound_unit: int | None = None) -> dict[str, Any]:
    if lifted_domain == "EACH_OWN_UNIT" and atom.feature in _TILE_ARG_FEATURES:
        if tile_slot is None:
            raise ValueError("lifted tile guard requires tile slot")
        return {"feature": atom.feature, "arg": {"blackboard": tile_slot}}
    if lifted_domain == "EACH_OWN_UNIT" and atom.feature in _UNIT_ARG_FEATURES:
        if bound_unit is None:
            raise ValueError("lifted unit guard requires static unit binding")
        return {"feature": atom.feature, "arg": {"const": int(bound_unit), "type": "UNIT"}}
    if lifted_domain == "EACH_OWN_UNIT" and atom.feature == "OWN_INVENTORY":
        if bound_unit is None:
            raise ValueError("lifted inventory guard requires static unit binding")
        return {"feature": atom.feature, "args": [
            {"const": int(bound_unit), "type": "UNIT"},
            {"const": int(atom.arg), "type": "ITEM"},
        ]}
    if atom.feature in _ITEM_ARG_FEATURES:
        return {"feature": atom.feature, "arg": {"const": int(atom.arg), "type": "ITEM"}}
    if atom.feature in _UNIT_ARG_FEATURES:
        return {"feature": atom.feature, "arg": {"const": int(atom.arg), "type": "UNIT"}}
    if atom.feature in _TILE_ARG_FEATURES:
        return {"feature": atom.feature, "arg": {"const": int(atom.arg), "type": "TILE"}}
    if atom.feature in ("SHOP_AT", "SHOP_SEEN"):
        return {"feature": atom.feature, "arg": int(atom.arg)}
    return {"feature": atom.feature}


def _rhs(atom: GuardAtom) -> Any:
    if atom.feature.endswith("_TILE_ITEM"):
        return _ITEM_NAMES.get(int(atom.value), int(atom.value))
    return int(atom.value)


def _condition(atom: GuardAtom, *, lifted_domain: str, tile_slot: str | None = None, bound_unit: int | None = None) -> dict[str, Any]:
    return {"type": "Condition", "lhs": _feature_ref(atom, lifted_domain=lifted_domain, tile_slot=tile_slot, bound_unit=bound_unit), "op": atom.cmp, "rhs": _rhs(atom)}


def _guard_tree(guard: LiftedGuard, leaf: dict[str, Any], *, lifted_domain: str, tile_slot: str | None = None) -> dict[str, Any]:
    if not guard.atoms:
        return leaf
    return {"type": "Sequence", "children": [
        *[_condition(a, lifted_domain=lifted_domain, tile_slot=tile_slot) for a in guard.atoms], leaf,
    ]}


def _authority_name(value: int) -> str:
    return _AUTH_NAMES.get(int(value), "COMMITTED_STRATEGY")


def _market_leaf(a: AbstractAction) -> dict[str, Any]:
    op = _MARKET_OP_NAMES[int(a.market_op)]
    leaf: dict[str, Any] = {"type": "EmitMarketIntent", "action": op, "authority": _authority_name(a.authority), "priority": int(a.priority)}
    if op not in ("HIRE", "BUY_LAND"):
        leaf["item"] = _ITEM_NAMES[int(a.item)]
        leaf["quantity"] = int(a.quantity)
    return leaf


def distill_invariant_module(
    module: InvariantStrategicModule,
    *,
    max_units: int = 13,
    route_slot: str = "slot_19",
    provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Translate B4IM1 semantics into ordinary schema-4 BKG4 plus bounded action reconciliation metadata.

    `UNIT_REWRITE` is represented exactly like the accepted feed-deferral module: BKG4 owns the lifted
    authorization decision through per-unit BB slots, while a generic reconciler only applies the declared
    match->replacement transformation to the already materialized route action. Market and route actions are
    ordinary BKG4 nodes. No feature-registry-v2 feature is required by this v1 grammar.
    """
    module.validate()
    if not 1 <= int(max_units) <= 40:
        raise ValueError("max_units must be 1..40")
    bb: list[dict[str, Any]] = []
    children: list[dict[str, Any]] = []
    rewrite_meta: list[dict[str, Any]] = []
    needs_tile_probe = any(r.domain == "EACH_OWN_UNIT" and any(a.feature in _TILE_ARG_FEATURES for a in (*r.guard.atoms, *r.exit_guard.atoms)) for r in module.rules)
    tile_slot = "invariant_probe_tile"
    if needs_tile_probe:
        bb.append({"name": tile_slot, "type": "TILE", "initial": -1, "owner": "invariant_module"})

    mode_slots: list[str] = []
    for ri, rule in enumerate(module.rules):
        a = rule.action
        if a.kind == "UNIT_REWRITE":
            if rule.domain != "EACH_OWN_UNIT":
                raise ValueError("UNIT_REWRITE currently requires EACH_OWN_UNIT")
            for unit in range(int(max_units)):
                slot = f"invariant_rewrite_r{ri}_u{unit}"
                bb.append({"name": slot, "type": "BOOL", "initial": 0, "owner": "invariant_module"})
                children.append({"type": "SetBlackboard", "slot": slot, "value": 0})
                branch_children: list[dict[str, Any]] = []
                if needs_tile_probe:
                    branch_children.append({"type": "SetBlackboard", "slot": tile_slot, "value": {"feature": "OWN_WORKER_TILE", "arg": unit}})
                branch_children.extend(_condition(g, lifted_domain=rule.domain, tile_slot=tile_slot, bound_unit=unit) for g in rule.guard.atoms)
                branch_children.append({"type": "SetBlackboard", "slot": slot, "value": 1})
                children.append({
                    "type": "If",
                    "condition": {"type": "Condition", "lhs": {"feature": "OWN_UNITS"}, "op": "GT", "rhs": unit},
                    "then": {"type": "ForceSuccess", "child": {"type": "Sequence", "children": branch_children}},
                    "else": {"type": "Success"},
                })
                rewrite_meta.append({
                    "slot": slot, "unit": unit, "match_op": _UNIT_OP_NAMES[int(a.match_op)],
                    "replacement_op": _UNIT_OP_NAMES[int(a.unit_op)], "arg": int(a.unit_arg), "quantity": int(a.quantity),
                    "rule": ri,
                })
            continue

        if rule.domain != "GLOBAL":
            raise ValueError(f"{a.kind} distillation currently requires GLOBAL domain")
        if a.kind == "MARKET_INTENT":
            children.append({"type": "ForceSuccess", "child": _guard_tree(rule.guard, _market_leaf(a), lifted_domain="GLOBAL")})
        elif a.kind == "ROUTE_OVERRIDE":
            leaf = {"type": "SetBlackboard", "slot": str(route_slot), "value": int(a.route_id)}
            children.append({"type": "ForceSuccess", "child": _guard_tree(rule.guard, leaf, lifted_domain="GLOBAL")})
        elif a.kind == "COMMITTED_ROUTE":
            mode = f"invariant_mode_r{ri}"
            mode_slots.append(mode)
            bb.append({"name": mode, "type": "BOOL", "initial": 0, "owner": "invariant_module"})
            enter = _guard_tree(rule.guard, {"type": "SetBlackboard", "slot": mode, "value": 1}, lifted_domain="GLOBAL")
            exit_ = _guard_tree(rule.exit_guard, {"type": "SetBlackboard", "slot": mode, "value": 0}, lifted_domain="GLOBAL")
            apply_route = {
                "type": "If",
                "condition": {"type": "Condition", "lhs": {"blackboard": mode}, "op": "EQ", "rhs": 1},
                "then": {"type": "SetBlackboard", "slot": str(route_slot), "value": int(a.route_id)},
                "else": {"type": "Success"},
            }
            children.extend([
                {"type": "ForceSuccess", "child": enter},
                {"type": "ForceSuccess", "child": exit_},
                apply_route,
            ])
        elif a.kind == "NOOP":
            continue
        else:
            raise ValueError(f"unsupported distillation action {a.kind}")

    tree: dict[str, Any] = {"type": "Sequence", "children": children + [{"type": "Success"}], "module": "invariant_strategic_module_v1", "category": "strategic_invariant"}
    result: dict[str, Any] = {
        "schema_version": 4,
        "feature_registry_version": 1,
        "name": "bkg4_invariant_strategic_module_v1",
        "blackboard": bb,
        "tree": tree,
        "manifest": {
            "module": "invariant_strategic_module_v1",
            "grammar_id": GRAMMAR_ID,
            "semantic_id": module.semantic_id(),
            "feature_registry": 1,
            "provenance": dict(provenance or {}),
        },
    }
    if rewrite_meta:
        result["runtime_reconciler"] = {
            "kind": "invariant_unit_rewrite_v1",
            "rewrite_slots": [x["slot"] for x in rewrite_meta],
            "rewrites": rewrite_meta,
            "materialization_rule": "apply declared unit rewrite iff BKG4 authorization slot is true and planned op matches",
        }
    return result


def compose_distilled_module(parent: Mapping[str, Any], invariant: Mapping[str, Any]) -> dict[str, Any]:
    """Append a distilled invariant module without replacing parent runtime mechanics."""
    out = copy.deepcopy(dict(parent))
    child = copy.deepcopy(dict(invariant))
    if int(out.get("schema_version", 0)) != 4 or int(child.get("schema_version", 0)) != 4:
        raise ValueError("distilled invariant composition requires schema 4 modules")
    if int(child.get("feature_registry_version", 0)) != 1:
        raise ValueError("distilled invariant composition requires feature registry v1")

    parent_bb = list(out.get("blackboard") or [])
    child_bb = list(child.get("blackboard") or [])
    names = [str(x.get("name") or "") for x in (*parent_bb, *child_bb)]
    if any(not x for x in names) or len(names) != len(set(names)):
        raise ValueError("composed distilled module blackboard names must be nonempty and unique")
    out["blackboard"] = parent_bb + child_bb

    parent_tree = out.get("tree")
    child_tree = child.get("tree")
    if not isinstance(parent_tree, dict) or not isinstance(child_tree, dict):
        raise ValueError("composed distilled modules require trees")
    if parent_tree.get("type") == "Sequence":
        parent_tree.setdefault("children", []).append(child_tree)
    else:
        out["tree"] = {"type": "Sequence", "children": [parent_tree, child_tree]}

    profiles = []
    for profile in (out.get("runtime_reconciler"), child.get("runtime_reconciler")):
        if not profile:
            continue
        if profile.get("kind") == "composite_v1":
            profiles.extend(copy.deepcopy(list(profile.get("reconcilers") or [])))
        else:
            profiles.append(copy.deepcopy(dict(profile)))
    if profiles:
        out["runtime_reconciler"] = {
            "kind": "composite_v1",
            "reconcilers": profiles,
            "materialization_rule": "apply reconciler profiles in declared order after route and exploit materialization",
        }

    parent_manifest = dict(out.get("manifest") or {})
    child_manifest = dict(child.get("manifest") or {})
    composition = list(parent_manifest.get("composition") or [])
    composition.append(str(child_manifest.get("module") or "invariant_strategic_module_v1"))
    parent_manifest.update({
        "module": "invariant_strategic_bkg4_composite_v1",
        "composition": composition,
        "invariant_semantic_id": child_manifest.get("semantic_id"),
        "invariant_grammar_id": child_manifest.get("grammar_id", GRAMMAR_ID),
    })
    out["manifest"] = parent_manifest
    out["name"] = "invariant_strategic_bkg4_composite_v1"
    return out


def pack_file(modules: Sequence[InvariantStrategicModule]) -> bytes:
    header = struct.pack("<8sII", b"B4IM1\0\0\0", B4IM1_VERSION, len(modules))
    return header + b"".join(pack_candidate(m) for m in modules)


def unpack_file(raw: bytes) -> tuple[InvariantStrategicModule, ...]:
    fh = struct.Struct("<8sII")
    if len(raw) < fh.size:
        raise ValueError("B4IM1 file too short")
    magic, version, n = fh.unpack_from(raw, 0)
    if magic != b"B4IM1\0\0\0" or version != B4IM1_VERSION:
        raise ValueError("invalid B4IM1 file header")
    expected = fh.size + int(n) * B4IM1_CANDIDATE_SIZE
    if len(raw) != expected:
        raise ValueError("B4IM1 file size mismatch")
    return tuple(unpack_candidate(raw[fh.size + i * B4IM1_CANDIDATE_SIZE: fh.size + (i + 1) * B4IM1_CANDIDATE_SIZE]) for i in range(int(n)))
