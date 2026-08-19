from __future__ import annotations

import copy
import hashlib
import json
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

# BKG4 deliberately reuses only the frozen BKG3 legal observation implementation and
# Kaggriculture domain constants.  The compiler, ABI, static analysis and VM below are
# independent.  This keeps the privacy boundary single-sourced without mutating BKG3.
import bonsai_kg as _bkg3

SCHEMA_VERSION = 4
ABI_VERSION = 4
MAGIC = b"BKG4"
SEMANTIC_DOMAIN = b"BKG4SEM\0"
BOARD_SIZE = 10
FEATURE_REGISTRY_VERSION = 1

MAX_NODES = 65534
MAX_CHILD_REFS = 65533
MAX_DEPTH = 48
MAX_BB = 128
MAX_STATE = 512
MAX_TABLE = 65534
MAX_PARALLEL_CHILDREN = 16
MAX_UNITS = 40
MAX_MARKET_INTENTS = 16
MAX_LIVE_INTENTS = 128
MAX_INSTRUCTION_BUDGET = 65534
NORMAL_VISIT_WARNING = 4096
NONE_U16 = 0xFFFF
ACTIVE_BIT = 1 << 31
RUNNING_MASK16 = 0xFFFF

STATUS_SUCCESS = 0
STATUS_FAILURE = 1
STATUS_RUNNING = 2
STATUS_SKIPPED = 3
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
    "SwitchEnum": 22,
    "ReactiveIf": 23,
}
NODE_NAMES = {v: k for k, v in NODE_TYPES.items()}
DECORATOR_TYPES = {NODE_TYPES[x] for x in (
    "Invert", "ForceSuccess", "ForceFailure", "TurnTimeout", "Retry", "Repeat",
    "KeepRunningUntilFailure"
)}

CMP = {"EQ": 0, "NE": 1, "LT": 2, "LE": 3, "GT": 4, "GE": 5}
CMP_NAMES = {v: k for k, v in CMP.items()}

BB_TYPES = {"BOOL": 0, "I32": 1, "U16": 2, "ENUM8": 3, "TILE": 4, "UNIT": 5, "ITEM": 6}
BB_NAMES = {v: k for k, v in BB_TYPES.items()}
VALUE_TYPES = {name: BB_TYPES[name] for name in ("I32", "TILE", "UNIT", "ITEM")}
VALUE_KINDS = {"CONST": 0, "BB": 1, "LEGAL_FEATURE": 2}
FEATURE_ARG_KINDS = {"CONST": 0, "BB": 1}
EFFECT_OP = {"set": 0, "clear": 1, "add": 2}

AUTHORITY = {
    "FALLBACK": 0,
    "TACTICAL": 1,
    "DOMAIN": 2,
    "COMMITTED_STRATEGY": 3,
    "RECOVERY": 4,
    "INVARIANT": 5,
}
AUTHORITY_NAMES = {v: k for k, v in AUTHORITY.items()}
MAX_AUTHORITY = max(AUTHORITY.values())

# Registry v1 is executable-identical to BKG3's legal/current observation surface.
FEATURES = dict(_bkg3.FEATURES)
FEATURE_NAMES = dict(_bkg3.FEATURE_NAMES)
FEATURE_TYPES = dict(_bkg3.FEATURE_TYPES)
FEATURE_ARG_TYPES = dict(_bkg3.FEATURE_ARG_TYPES)
ITEMS = tuple(_bkg3.ITEMS)
ITEM_ID = dict(_bkg3.ITEM_ID)
SHOPS = tuple(_bkg3.SHOPS)
SHOP_ID = dict(_bkg3.SHOP_ID)
N_PRODUCTS = _bkg3.N_PRODUCTS
N_CROPS = _bkg3.N_CROPS
N_ANIMALS = _bkg3.N_ANIMALS
UNIT_OPS = dict(_bkg3.UNIT_OPS)
MARKET_OPS = dict(_bkg3.MARKET_OPS)

# Header4 is exactly 64 bytes. Child-ref count is n_nodes-1; root and board size are
# semantic constants.  Header stores actual proof metadata, never hard maxima.
HEADER = struct.Struct("<4sHH32s12H")
NODE_STRUCT = struct.Struct("<BBHHHHHiiiiI")
BB_STRUCT = struct.Struct("<BBHi")
VALUE_STRUCT = struct.Struct("<BBHii")
FEATURE_ARG_STRUCT = struct.Struct("<BBHi")
PRED_STRUCT = struct.Struct("<HHB7s")
LIFECYCLE_STRUCT = struct.Struct("<12H")
EFFECT_STRUCT = struct.Struct("<BBHi")
INTENT_STRUCT = struct.Struct("<HHHHBBBB")
assert HEADER.size == 64
assert NODE_STRUCT.size == 32
assert BB_STRUCT.size == 8
assert VALUE_STRUCT.size == 12
assert FEATURE_ARG_STRUCT.size == 8
assert PRED_STRUCT.size == 12
assert LIFECYCLE_STRUCT.size == 24
assert EFFECT_STRUCT.size == 8
assert INTENT_STRUCT.size == 12


class PolicyError(ValueError):
    pass


@dataclass(frozen=True)
class EnumType:
    tag: int
    name: str
    values: Tuple[str, ...]


@dataclass
class Node:
    type: int
    flags: int = 0
    first_child: int = 0
    child_count: int = 0
    state_slot: int = NONE_U16
    lifecycle_slot: int = NONE_U16
    subtree_end: int = 0
    p0: int = 0
    p1: int = 0
    p2: int = 0
    p3: int = 0
    aux: int = 0


@dataclass
class BbDecl:
    type: int
    enum_size: int = 0
    enum_tag: int = 0
    initial: int = 0
    name: str = ""
    owner: str = ""


@dataclass(frozen=True)
class ValueRef:
    kind: int
    type: int
    id: int = 0
    arg: int = 0
    value: int = 0


@dataclass(frozen=True)
class FeatureArgRef:
    kind: int
    type: int
    id: int = 0
    value: int = 0


@dataclass(frozen=True)
class Predicate:
    lhs: int
    rhs: int
    cmp: int


@dataclass(frozen=True)
class Effect:
    op: int
    slot: int
    value: int


@dataclass(frozen=True)
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
class PortBinding:
    value_type: int
    enum_tag: int
    mode: str
    read: Any = None
    dest: Optional[str] = None


@dataclass
class StaticInfo:
    may_run: bool = False
    needs_state: bool = False
    max_depth: int = 1
    max_visits: int = 1
    max_live_intents: int = 0
    max_emissions: int = 0
    bb_reads: Set[int] = field(default_factory=set)
    bb_writes: Set[int] = field(default_factory=set)
    features: Set[int] = field(default_factory=set)
    action_domains: Set[str] = field(default_factory=set)
    authorities: Set[int] = field(default_factory=set)


@dataclass
class CompiledPolicy:
    semantic_hash: str
    nodes: List[Node]
    child_refs: List[int]
    bb: List[BbDecl]
    values: List[ValueRef]
    feature_args: List[FeatureArgRef]
    predicates: List[Predicate]
    lifecycles: List[Lifecycle]
    effects: List[Effect]
    state_count: int
    instruction_budget: int
    max_live_intents: int
    max_depth: int
    feature_registry_version: int = FEATURE_REGISTRY_VERSION
    source: Optional[Dict[str, Any]] = None
    analysis: Optional[List[StaticInfo]] = None
    manifest: Dict[str, Any] = field(default_factory=dict)
    warnings: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def root(self) -> int:
        return 0

    @property
    def board_size(self) -> int:
        return BOARD_SIZE

    @property
    def compiled_bytes(self) -> int:
        return (HEADER.size + len(self.nodes) * NODE_STRUCT.size + len(self.child_refs) * 2
                + len(self.bb) * BB_STRUCT.size + len(self.values) * VALUE_STRUCT.size
                + len(self.feature_args) * FEATURE_ARG_STRUCT.size
                + len(self.predicates) * PRED_STRUCT.size
                + len(self.lifecycles) * LIFECYCLE_STRUCT.size
                + len(self.effects) * EFFECT_STRUCT.size)

    @property
    def canonical_hash(self) -> str:
        return self.semantic_hash


@dataclass
class NodeState:
    a: int = 0
    b: int = 0


@dataclass
class DeferredIntent:
    priority_key: int
    source: int
    order: int
    quantity: int
    op: int
    arg: int
    subject_kind: int


@dataclass
class UnitClaim:
    priority_key: int
    order: int
    op: int
    arg: int
    quantity: int


@dataclass
class MarketClaim:
    priority_key: int
    order: int
    op: int
    item: int
    quantity: int


@dataclass
class RuntimeState:
    blackboard: List[int]
    node_state: List[NodeState]
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
class TickProfile:
    visits: int = 0
    max_depth: int = 0
    feature_evals: int = 0
    emitted_intents: int = 0
    max_live_intents: int = 0
    node_visits: Dict[int, int] = field(default_factory=dict)
    statuses: Dict[int, List[int]] = field(default_factory=dict)
    halts: Dict[int, int] = field(default_factory=dict)


@dataclass
class TickResult:
    status: int
    action: ActionResult
    fault: int
    blackboard: Tuple[int, ...]
    node_state: Tuple[Tuple[int, int], ...]
    trace: Tuple[Tuple[int, int, str, str], ...]
    profile: Optional[TickProfile] = None


SyntheticObservation = _bkg3.SyntheticObservation
DictObservation = _bkg3.DictObservation


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
    if bb_type == BB_TYPES["ITEM"] and not -1 <= value < len(ITEMS): raise PolicyError(f"{what} ITEM out of range")


def _effective_type(bb: BbDecl) -> int:
    return bb.type if bb.type in (BB_TYPES["TILE"], BB_TYPES["UNIT"], BB_TYPES["ITEM"]) else BB_TYPES["I32"]


def _feature_const_arg_ok(fid: int, position: int, arg: int) -> bool:
    sig = FEATURE_ARG_TYPES.get(fid)
    if sig is None or not 0 <= position < len(sig): return False
    typ = sig[position]
    if typ == BB_TYPES["TILE"]: return -1 <= arg < BOARD_SIZE * BOARD_SIZE
    if typ == BB_TYPES["UNIT"]: return -1 <= arg < MAX_UNITS
    if typ == BB_TYPES["ITEM"]: return -1 <= arg < len(ITEMS)
    if fid in (FEATURES["SHOP_AT"], FEATURES["SHOP_SEEN"]): return 0 <= arg < 8
    if fid in (FEATURES["OWN_QUADRANT_UNLOCKED"], FEATURES["OPP_QUADRANT_UNLOCKED"]): return 0 <= arg < 4
    return True


def priority_key(authority: str | int, local_priority: int) -> int:
    a = AUTHORITY.get(authority, authority) if isinstance(authority, str) else authority
    if not isinstance(a, int) or not 0 <= a <= MAX_AUTHORITY: raise PolicyError("invalid authority")
    if not isinstance(local_priority, int) or not 0 <= local_priority <= 255: raise PolicyError("priority must be 0..255")
    return (a << 8) | local_priority


def _authority_key_from_node(obj: Dict[str, Any]) -> int:
    return priority_key(obj.get("authority", "TACTICAL"), _fit_i32(obj.get("priority", 0), "priority"))


class Compiler:
    def __init__(self, policy: Dict[str, Any]):
        self.source = copy.deepcopy(policy)
        self.nodes: List[Node] = []
        self.child_refs: List[int] = []
        self.bb: List[BbDecl] = []
        self.bb_by_name: Dict[str, int] = {}
        self.enums: Dict[str, EnumType] = {}
        self.enums_by_tag: Dict[int, EnumType] = {}
        self.values: List[ValueRef] = []
        self.value_cache: Dict[ValueRef, int] = {}
        self.feature_args: List[FeatureArgRef] = []
        self.feature_arg_seq_cache: Dict[Tuple[FeatureArgRef, ...], int] = {}
        self.predicates: List[Predicate] = []
        self.predicate_cache: Dict[Predicate, int] = {}
        self.effects: List[Effect] = []
        self.effect_seq_cache: Dict[Tuple[Effect, ...], Tuple[int, int]] = {}
        self.lifecycles: List[Lifecycle] = []
        self.lifecycle_cache: Dict[Lifecycle, int] = {}
        self.subtrees: Dict[str, Dict[str, Any]] = {}
        self.analysis: List[StaticInfo] = []
        self.module_roots: List[Tuple[int, str, str, Dict[str, Any]]] = []
        self.warnings: List[Dict[str, Any]] = []

    def compile(self) -> CompiledPolicy:
        p = self.source
        if not isinstance(p, dict) or p.get("schema_version") != SCHEMA_VERSION:
            raise PolicyError("schema_version must be 4")
        allowed_top = {"schema_version", "name", "semantic_hash", "feature_registry_version", "enums",
                       "blackboard", "subtrees", "tree", "instruction_budget", "provenance"}
        extra = set(p) - allowed_top
        if extra: raise PolicyError(f"unknown policy fields: {sorted(extra)}")
        registry = p.get("feature_registry_version", FEATURE_REGISTRY_VERSION)
        if registry != FEATURE_REGISTRY_VERSION: raise PolicyError(f"unsupported feature_registry_version {registry}")
        self._compile_enums(p.get("enums", []))
        self._compile_blackboard(p.get("blackboard", []))
        self._load_subtrees(p.get("subtrees", []))
        if "tree" not in p: raise PolicyError("missing tree")
        root = self._compile_node(self._expand(p["tree"], {}, ()), 1, "root")
        if root != 0 or not self.nodes: raise PolicyError("compiled root must be node 0")
        if len(self.nodes) > MAX_NODES: raise PolicyError("node limit exceeds 65534")
        if len(self.child_refs) != len(self.nodes) - 1: raise PolicyError("strict tree child-ref invariant violated")
        self.analysis = analyze_policy(self.nodes, self.child_refs, self.values, self.feature_args,
                                       self.predicates, self.lifecycles, self.effects, self.bb)
        state_count = 0
        for i, info in enumerate(self.analysis):
            self.nodes[i].state_slot = state_count if info.needs_state else NONE_U16
            if info.needs_state:
                state_count += 1
        if state_count > MAX_STATE: raise PolicyError(f"state slot limit exceeded: {state_count}>{MAX_STATE}")
        max_depth = self.analysis[0].max_depth
        max_visits = self.analysis[0].max_visits
        max_live = self.analysis[0].max_live_intents
        if max_depth > MAX_DEPTH: raise PolicyError(f"tree depth exceeds {MAX_DEPTH}")
        if max_live > MAX_LIVE_INTENTS: raise PolicyError(f"static live intent bound {max_live} exceeds {MAX_LIVE_INTENTS}")
        if max_visits > MAX_INSTRUCTION_BUDGET: raise PolicyError("static visit bound exceeds uint16 execution capacity")
        requested_budget = p.get("instruction_budget")
        if requested_budget is None:
            budget = max(64, max_visits)
        else:
            budget = _fit_i32(requested_budget, "instruction_budget")
            if budget < max_visits: raise PolicyError(f"instruction_budget {budget} under static visit bound {max_visits}")
            if not 1 <= budget <= MAX_INSTRUCTION_BUDGET: raise PolicyError("instruction_budget out of range")
        if max_visits > NORMAL_VISIT_WARNING:
            self.warnings.append({"code":"HOT_PATH_GT_4096","max_visits":max_visits})
        self._lint_source(p["tree"], "root")
        self._lint_bb_ownership()
        cp = CompiledPolicy("", self.nodes, self.child_refs, self.bb, self.values, self.feature_args,
                            self.predicates, self.lifecycles, self.effects, state_count, budget, max_live,
                            max_depth, registry, copy.deepcopy(p), self.analysis, {}, list(self.warnings))
        cp.manifest = self._build_manifest(cp)
        validate_compiled(cp)
        cp.semantic_hash = hashlib.sha256(semantic_payload_bytes(cp)).hexdigest()
        supplied = p.get("semantic_hash")
        if supplied is not None and supplied != cp.semantic_hash:
            raise PolicyError("semantic_hash does not match normalized BKG4 executable")
        return cp

    def _compile_enums(self, defs: Any) -> None:
        if not isinstance(defs, list): raise PolicyError("enums must be a list")
        for i, d in enumerate(defs, 1):
            if not isinstance(d, dict) or set(d) != {"name", "values"}: raise PolicyError("enum requires name/values")
            name, values = d["name"], d["values"]
            if not isinstance(name, str) or not name or name in self.enums: raise PolicyError("invalid/duplicate enum name")
            if name in BB_TYPES: raise PolicyError("named enum collides with builtin type")
            if not isinstance(values, list) or not 1 <= len(values) <= 255 or any(not isinstance(x,str) or not x for x in values):
                raise PolicyError("enum values must be 1..255 nonempty strings")
            if len(set(values)) != len(values): raise PolicyError(f"duplicate enum value in {name}")
            e = EnumType(i, name, tuple(values)); self.enums[name] = e; self.enums_by_tag[i] = e

    def _compile_blackboard(self, decls: Any) -> None:
        if not isinstance(decls, list) or len(decls) > MAX_BB: raise PolicyError(f"blackboard must be list with <={MAX_BB} slots")
        used_enum_tags: Set[int] = set()
        for i, d in enumerate(decls):
            if not isinstance(d, dict) or set(d) - {"name","type","initial","owner"}: raise PolicyError("invalid blackboard declaration")
            name, type_name = d.get("name"), d.get("type")
            if not isinstance(name,str) or not name or not name[0].isalpha() or not all(c.isalnum() or c=="_" for c in name): raise PolicyError("invalid blackboard slot name")
            if name in self.bb_by_name: raise PolicyError(f"duplicate blackboard slot {name}")
            owner = d.get("owner", "root")
            if not isinstance(owner,str) or not owner: raise PolicyError("blackboard owner must be nonempty string")
            enum_tag=enum_size=0
            if type_name in self.enums:
                e=self.enums[type_name]; typ=BB_TYPES["ENUM8"];enum_tag=e.tag;enum_size=len(e.values);used_enum_tags.add(e.tag)
                initial_raw=d.get("initial",e.values[0])
                if isinstance(initial_raw,str):
                    if initial_raw not in e.values: raise PolicyError(f"unknown {e.name} enum value {initial_raw}")
                    initial=e.values.index(initial_raw)
                else: initial=_fit_i32(initial_raw,f"blackboard {name} initial")
            else:
                if type_name not in BB_TYPES: raise PolicyError(f"unknown blackboard type {type_name}")
                typ=BB_TYPES[type_name]; initial=_fit_i32(d.get("initial",0),f"blackboard {name} initial")
            _validate_bb_value(typ,initial,f"blackboard {name} initial")
            if enum_tag and not 0 <= initial < enum_size: raise PolicyError(f"blackboard {name} enum initial out of range")
            self.bb_by_name[name]=i;self.bb.append(BbDecl(typ,enum_size,enum_tag,initial,name,owner))
        unused=set(self.enums_by_tag)-used_enum_tags
        if unused: raise PolicyError(f"named enums must back at least one blackboard slot: {[self.enums_by_tag[x].name for x in sorted(unused)]}")

    def _load_subtrees(self, defs: Any) -> None:
        if not isinstance(defs,list): raise PolicyError("subtrees must be a list")
        for d in defs:
            if not isinstance(d,dict) or set(d)-{"name","parameters","tree","category","provenance"} or not {"name","parameters","tree"}<=set(d):
                raise PolicyError("subtree definition requires name/parameters/tree")
            name=d["name"];params=d["parameters"]
            if not isinstance(name,str) or not name or name in self.subtrees or not isinstance(params,dict): raise PolicyError("invalid/duplicate subtree")
            normalized={}
            for key,spec in params.items():
                if isinstance(spec,str): type_name,mode=spec,"in"
                elif isinstance(spec,dict) and set(spec)<={"type","mode"} and "type" in spec: type_name,mode=spec["type"],spec.get("mode","in")
                else: raise PolicyError(f"invalid port {key} in subtree {name}")
                if type_name in self.enums: value_type,tag=BB_TYPES["I32"],self.enums[type_name].tag
                elif type_name in VALUE_TYPES: value_type,tag=VALUE_TYPES[type_name],0
                else: raise PolicyError(f"invalid port type {type_name}")
                if mode not in ("in","out","inout"): raise PolicyError("invalid port mode")
                normalized[key]={"type_name":type_name,"value_type":value_type,"enum_tag":tag,"mode":mode}
            stored=copy.deepcopy(d);stored["parameters"]=normalized;self.subtrees[name]=stored
        graph={name:set() for name in self.subtrees}
        def calls(x:Any,out:Set[str])->None:
            if isinstance(x,dict):
                if x.get("type")=="Subtree":
                    target=x.get("subtree")
                    if target not in self.subtrees: raise PolicyError(f"unknown subtree {target}")
                    out.add(target)
                for v in x.values():calls(v,out)
            elif isinstance(x,list):
                for v in x:calls(v,out)
        for name,d in self.subtrees.items():calls(d["tree"],graph[name])
        color={n:0 for n in graph}
        def visit(n:str)->None:
            if color[n]==1:raise PolicyError(f"recursive subtree {n}")
            if color[n]==2:return
            color[n]=1
            for t in graph[n]:visit(t)
            color[n]=2
        for n in graph:visit(n)

    def _expand(self,obj:Any,env:Dict[str,PortBinding],stack:Tuple[str,...],context:str="value")->Any:
        if isinstance(obj,list):return [self._expand(x,env,stack,context) for x in obj]
        if not isinstance(obj,dict):return copy.deepcopy(obj)
        if set(obj)=={"binding"}:
            name=obj["binding"]
            if name not in env:raise PolicyError(f"unknown subtree binding {name}")
            b=env[name]
            if context=="dest":
                if b.mode not in ("out","inout") or b.dest is None:raise PolicyError(f"subtree port {name} is not writable")
                return b.dest
            if b.mode=="out" or b.read is None:raise PolicyError(f"subtree OUT port {name} is not readable")
            return copy.deepcopy(b.read)
        if obj.get("type")=="Subtree":
            if set(obj)-{"type","name","subtree","bindings","module"}:raise PolicyError("unknown Subtree invocation field")
            name=obj.get("subtree")
            if name not in self.subtrees or name in stack:raise PolicyError(f"unknown/recursive subtree {name}")
            definition=self.subtrees[name];params=definition["parameters"];actuals=obj.get("bindings")
            if not isinstance(actuals,dict) or set(actuals)!=set(params):raise PolicyError(f"subtree {name} bindings must exactly match parameters")
            resolved={}
            for key,spec in params.items():
                expected_type=spec["value_type"];expected_tag=spec["enum_tag"];mode=spec["mode"];actual=actuals[key]
                if isinstance(actual,dict) and set(actual)=={"binding"}:
                    outer=actual["binding"]
                    if outer not in env:raise PolicyError(f"unknown forwarded binding {outer}")
                    src=env[outer]
                    if (src.value_type,src.enum_tag)!=(expected_type,expected_tag):raise PolicyError("forwarded port type mismatch")
                    if mode=="in":
                        if src.mode=="out" or src.read is None:raise PolicyError("OUT cannot feed IN")
                        resolved[key]=PortBinding(expected_type,expected_tag,mode,copy.deepcopy(src.read),None)
                    elif mode=="out":
                        if src.mode not in ("out","inout") or src.dest is None:raise PolicyError("forwarded port not writable")
                        resolved[key]=PortBinding(expected_type,expected_tag,mode,None,src.dest)
                    else:
                        if src.mode!="inout" or src.dest is None or src.read is None:raise PolicyError("forwarded port cannot satisfy INOUT")
                        resolved[key]=PortBinding(expected_type,expected_tag,mode,copy.deepcopy(src.read),src.dest)
                    continue
                if mode=="in":
                    read=self._expand(actual,env,stack,"value");self._make_value_ref(read,expected_type,expected_tag,False)
                    resolved[key]=PortBinding(expected_type,expected_tag,mode,read,None)
                else:
                    dest_obj=self._expand(actual,env,stack,"value")
                    if not isinstance(dest_obj,dict) or set(dest_obj)!={"blackboard"}:raise PolicyError(f"{mode} port must bind blackboard")
                    dest=dest_obj["blackboard"]
                    if dest not in self.bb_by_name:raise PolicyError("invalid port destination")
                    bb=self.bb[self.bb_by_name[dest]]
                    if (_effective_type(bb),bb.enum_tag)!=(expected_type,expected_tag):raise PolicyError("port destination type mismatch")
                    resolved[key]=PortBinding(expected_type,expected_tag,mode,{"blackboard":dest} if mode=="inout" else None,dest)
            expanded=self._expand(definition["tree"],resolved,stack+(name,),"value")
            if isinstance(expanded,dict) and "module" not in expanded:
                expanded["module"]=obj.get("module",name)
                if "category" in definition:expanded["category"]=definition["category"]
                if "provenance" in definition:expanded["provenance"]=definition["provenance"]
            return expanded
        out={}
        for k,v in obj.items():out[k]=self._expand(v,env,stack,"dest" if k=="slot" else "value")
        return out

    def _enum_const(self,raw:Any,tag:int,what:str)->int:
        if not tag: return _fit_i32(raw,what)
        e=self.enums_by_tag[tag]
        if isinstance(raw,str):
            if raw not in e.values:raise PolicyError(f"unknown {e.name} value {raw}")
            return e.values.index(raw)
        v=_fit_i32(raw,what)
        if not 0<=v<len(e.values):raise PolicyError(f"{e.name} value out of range")
        return v

    def _make_feature_arg(self,obj:Any,expected_type:int)->FeatureArgRef:
        if expected_type==BB_TYPES["ITEM"] and isinstance(obj,str):obj={"const":obj,"type":"ITEM"}
        if isinstance(obj,(int,bool)):
            value=_fit_i32(obj,"feature arg");_validate_bb_value(expected_type,value,"feature arg");return FeatureArgRef(FEATURE_ARG_KINDS["CONST"],expected_type,0,value)
        if isinstance(obj,str) and expected_type==BB_TYPES["ITEM"]:
            return FeatureArgRef(FEATURE_ARG_KINDS["CONST"],expected_type,0,ITEM_ID[obj])
        if isinstance(obj,dict) and "const" in obj:
            if set(obj)-{"const","type"}:raise PolicyError("invalid feature CONST fields")
            typ=VALUE_TYPES.get(obj.get("type"),expected_type)
            if typ!=expected_type:raise PolicyError("feature arg type mismatch")
            raw=obj["const"];raw=ITEM_ID[raw] if typ==BB_TYPES["ITEM"] and isinstance(raw,str) else raw
            value=_fit_i32(raw,"feature arg");_validate_bb_value(typ,value,"feature arg");return FeatureArgRef(FEATURE_ARG_KINDS["CONST"],typ,0,value)
        if isinstance(obj,dict) and set(obj)=={"blackboard"}:
            name=obj["blackboard"]
            if name not in self.bb_by_name:raise PolicyError(f"unknown blackboard {name}")
            slot=self.bb_by_name[name];bb=self.bb[slot]
            if bb.enum_tag:raise PolicyError("named enum cannot be a feature index")
            typ=_effective_type(bb)
            if typ!=expected_type:raise PolicyError("feature arg BB type mismatch")
            return FeatureArgRef(FEATURE_ARG_KINDS["BB"],typ,slot,0)
        raise PolicyError("dynamic feature arguments are restricted to CONST or blackboard")

    def _feature_args(self,fid:int,obj:Dict[str,Any])->Tuple[int,int]:
        sig=FEATURE_ARG_TYPES[fid]
        if "arg" in obj and "args" in obj:raise PolicyError("feature uses arg or args, not both")
        if "args" in obj:raw=obj["args"]
        elif "arg" in obj:
            if fid==FEATURES["OWN_INVENTORY"] and isinstance(obj["arg"],int):raw=[(obj["arg"]>>8)&0xFF,obj["arg"]&0xFF]
            else:raw=[obj["arg"]]
        else:raw=[0]*len(sig)
        if not isinstance(raw,list) or len(raw)!=len(sig):raise PolicyError(f"feature {FEATURE_NAMES[fid]} arity mismatch")
        seq=[]
        for pos,(actual,typ) in enumerate(zip(raw,sig)):
            ref=self._make_feature_arg(actual,typ)
            if ref.kind==FEATURE_ARG_KINDS["CONST"] and not _feature_const_arg_ok(fid,pos,ref.value):raise PolicyError("feature const arg out of range")
            seq.append(ref)
        key=tuple(seq)
        if not key:return 0,0
        if key in self.feature_arg_seq_cache:return self.feature_arg_seq_cache[key],len(key)
        if len(self.feature_args)+len(key)>MAX_TABLE:raise PolicyError("FeatureArgRef table exceeds uint16 addressability")
        start=len(self.feature_args);self.feature_args.extend(key);self.feature_arg_seq_cache[key]=start;return start,len(key)

    def _make_value_ref(self,obj:Any,expected_type:Optional[int]=None,expected_enum_tag:int=0,allocate:bool=True)->ValueRef:
        if expected_enum_tag:
            if isinstance(obj,str):value=self._enum_const(obj,expected_enum_tag,"enum constant");ref=ValueRef(VALUE_KINDS["CONST"],BB_TYPES["I32"],0,expected_enum_tag,value)
            elif isinstance(obj,dict) and "const" in obj:
                if set(obj)-{"const","type"}:raise PolicyError("invalid enum CONST")
                t=obj.get("type")
                if t is not None and (t not in self.enums or self.enums[t].tag!=expected_enum_tag):raise PolicyError("enum CONST type mismatch")
                value=self._enum_const(obj["const"],expected_enum_tag,"enum constant");ref=ValueRef(VALUE_KINDS["CONST"],BB_TYPES["I32"],0,expected_enum_tag,value)
            elif isinstance(obj,dict) and set(obj)=={"blackboard"}:
                name=obj["blackboard"]
                if name not in self.bb_by_name:raise PolicyError("unknown blackboard")
                slot=self.bb_by_name[name];bb=self.bb[slot]
                if bb.enum_tag!=expected_enum_tag:raise PolicyError("named enum blackboard type mismatch")
                ref=ValueRef(VALUE_KINDS["BB"],BB_TYPES["I32"],slot,expected_enum_tag,0)
            elif isinstance(obj,int):
                value=self._enum_const(obj,expected_enum_tag,"enum constant");ref=ValueRef(VALUE_KINDS["CONST"],BB_TYPES["I32"],0,expected_enum_tag,value)
            else:raise PolicyError("named enum ValueRef must be enum const or matching blackboard")
        elif expected_type==BB_TYPES["ITEM"] and isinstance(obj,str):
            ref=ValueRef(VALUE_KINDS["CONST"],BB_TYPES["ITEM"],0,0,ITEM_ID[obj] if obj in ITEM_ID else -999)
            _validate_bb_value(ref.type,ref.value,"ITEM constant")
        elif isinstance(obj,(int,bool)):
            typ=expected_type if expected_type is not None else BB_TYPES["I32"];value=_fit_i32(obj,"constant ValueRef");_validate_bb_value(typ,value,"constant ValueRef");ref=ValueRef(VALUE_KINDS["CONST"],typ,0,0,value)
        elif isinstance(obj,dict) and "const" in obj:
            if set(obj)-{"const","type"}:raise PolicyError("invalid CONST ValueRef")
            type_name=obj.get("type");typ=VALUE_TYPES.get(type_name) if type_name is not None else expected_type
            if typ not in VALUE_TYPES.values():raise PolicyError("CONST ValueRef requires portable type")
            raw=obj["const"]
            if typ==BB_TYPES["ITEM"] and isinstance(raw,str):
                if raw not in ITEM_ID:raise PolicyError(f"unknown item {raw}")
                raw=ITEM_ID[raw]
            value=_fit_i32(raw,"constant ValueRef");_validate_bb_value(typ,value,"constant ValueRef");ref=ValueRef(VALUE_KINDS["CONST"],typ,0,0,value)
        elif isinstance(obj,dict) and set(obj)=={"blackboard"}:
            name=obj["blackboard"]
            if name not in self.bb_by_name:raise PolicyError(f"unknown blackboard {name}")
            slot=self.bb_by_name[name];bb=self.bb[slot];ref=ValueRef(VALUE_KINDS["BB"],_effective_type(bb),slot,bb.enum_tag,0)
        elif isinstance(obj,dict) and "feature" in obj:
            if set(obj)-{"feature","arg","args"}:raise PolicyError("invalid LEGAL_FEATURE ValueRef")
            name=obj["feature"]
            if name not in FEATURES:raise PolicyError(f"illegal/unknown feature {name}")
            fid=FEATURES[name];start,count=self._feature_args(fid,obj);ref=ValueRef(VALUE_KINDS["LEGAL_FEATURE"],FEATURE_TYPES[fid],fid,start,count)
        else:raise PolicyError("ValueRef must be CONST, BB, or LEGAL_FEATURE")
        if expected_type is not None and ref.type!=expected_type:raise PolicyError(f"ValueRef type mismatch: expected {BB_NAMES[expected_type]}, got {BB_NAMES.get(ref.type,ref.type)}")
        if expected_enum_tag and ref.arg!=expected_enum_tag:raise PolicyError("named enum tag mismatch")
        if expected_type is not None and not expected_enum_tag and ref.kind in (VALUE_KINDS["CONST"],VALUE_KINDS["BB"]) and ref.arg!=0:raise PolicyError("named enum used where ordinary I32 expected")
        if allocate:
            if ref in self.value_cache:return self.values[self.value_cache[ref]]
            if len(self.values)>=MAX_TABLE:raise PolicyError("ValueRef table exceeds uint16 addressability")
            self.value_cache[ref]=len(self.values);self.values.append(ref)
        return ref

    def _compile_value_ref(self,obj:Any,expected_type:Optional[int]=None,expected_enum_tag:int=0)->int:
        ref=self._make_value_ref(obj,expected_type,expected_enum_tag,True);return self.value_cache[ref]

    def _compile_predicate(self,obj:Any)->int:
        if not isinstance(obj,dict) or set(obj)!={"lhs","op","rhs"}:raise PolicyError("predicate requires lhs/op/rhs")
        if obj["op"] not in CMP:raise PolicyError("invalid comparison")
        lhs=self._compile_value_ref(obj["lhs"]);lv=self.values[lhs];tag=lv.arg if lv.kind in (VALUE_KINDS["CONST"],VALUE_KINDS["BB"]) else 0
        rhs=self._compile_value_ref(obj["rhs"],lv.type,tag);rv=self.values[rhs]
        if tag and obj["op"] not in ("EQ","NE"):raise PolicyError("named enums support EQ/NE comparisons only")
        q=Predicate(lhs,rhs,CMP[obj["op"]])
        if q in self.predicate_cache:return self.predicate_cache[q]
        if len(self.predicates)>=MAX_TABLE:raise PolicyError("predicate table exceeds uint16 addressability")
        self.predicate_cache[q]=len(self.predicates);self.predicates.append(q);return len(self.predicates)-1

    def _compile_effects(self,raw:Any,label:str)->Tuple[int,int]:
        if raw is None or raw==[]:return (0,0)
        if not isinstance(raw,list):raise PolicyError(f"{label} must be list")
        seq=[]
        for e in raw:
            if not isinstance(e,dict) or set(e)-{"op","slot","value"}:raise PolicyError("invalid effect")
            op=e.get("op");name=e.get("slot")
            if op not in EFFECT_OP or name not in self.bb_by_name:raise PolicyError("invalid effect op/slot")
            slot=self.bb_by_name[name];bb=self.bb[slot]
            if op=="clear":
                if "value" in e:raise PolicyError("clear effect has no value")
                value=0
            else:
                if "value" not in e:raise PolicyError(f"{op} effect requires value")
                if op=="add" and bb.type!=BB_TYPES["I32"]:raise PolicyError("add effect requires I32")
                value=self._compile_value_ref(e["value"],_effective_type(bb),bb.enum_tag)
                v=self.values[value]
                if op=="set" and v.kind==VALUE_KINDS["CONST"]:_validate_bb_value(bb.type,v.value,"effect SET")
            seq.append(Effect(EFFECT_OP[op],slot,value))
        key=tuple(seq)
        if key in self.effect_seq_cache:return self.effect_seq_cache[key]
        if len(self.effects)+len(key)>MAX_TABLE:raise PolicyError("effect table exceeds uint16 addressability")
        start=len(self.effects);self.effects.extend(key);rng=(start,len(key));self.effect_seq_cache[key]=rng;return rng

    def _compile_lifecycle(self,obj:Any)->int:
        if not isinstance(obj,dict):raise PolicyError("lifecycle must be object")
        allowed={"failure_if","success_if","skip_if","while","on_success","on_failure","on_halted","post"}
        if set(obj)-allowed:raise PolicyError("unknown lifecycle field")
        def pred(name:str)->int:return self._compile_predicate(obj[name]) if name in obj else NONE_U16
        lc=Lifecycle(pred("failure_if"),pred("success_if"),pred("skip_if"),pred("while"),
                     self._compile_effects(obj.get("on_success"),"on_success"),self._compile_effects(obj.get("on_failure"),"on_failure"),
                     self._compile_effects(obj.get("on_halted"),"on_halted"),self._compile_effects(obj.get("post"),"post"))
        if lc in self.lifecycle_cache:return self.lifecycle_cache[lc]
        if len(self.lifecycles)>=MAX_TABLE:raise PolicyError("lifecycle table exceeds uint16 addressability")
        self.lifecycle_cache[lc]=len(self.lifecycles);self.lifecycles.append(lc);return len(self.lifecycles)-1

    def _node_children(self,typ:str,obj:Dict[str,Any])->List[Any]:
        if typ in ("Sequence","Selector","ReactiveSequence","ReactiveSelector","Parallel"):
            children=obj.get("children")
            if not isinstance(children,list) or not children:raise PolicyError(f"{typ} requires nonempty children")
            if typ=="Parallel" and not 2<=len(children)<=MAX_PARALLEL_CHILDREN:raise PolicyError("Parallel requires 2..16 children")
            return children
        if typ in ("If","ReactiveIf"):
            if "condition" not in obj or "then" not in obj:raise PolicyError(f"{typ} requires condition/then")
            return [obj["condition"],obj["then"],obj.get("else",{"type":"Failure"})]
        if typ=="SwitchEnum":
            slot=obj.get("slot")
            if slot not in self.bb_by_name:raise PolicyError("SwitchEnum requires named enum blackboard slot")
            bb=self.bb[self.bb_by_name[slot]]
            if not bb.enum_tag:raise PolicyError("SwitchEnum only accepts named enum slots")
            e=self.enums_by_tag[bb.enum_tag];cases=obj.get("cases")
            if not isinstance(cases,dict) or set(cases)-set(e.values):raise PolicyError("SwitchEnum cases contain invalid enum value")
            default=obj.get("default",{"type":"Failure"})
            return [cases.get(v,default) for v in e.values]+[default]
        if NODE_TYPES[typ] in DECORATOR_TYPES:
            if "child" not in obj:raise PolicyError(f"{typ} requires child")
            return [obj["child"]]
        return []

    def _compile_node(self,obj:Any,depth:int,path:str)->int:
        if depth>MAX_DEPTH:raise PolicyError(f"tree depth exceeds {MAX_DEPTH}")
        if len(self.nodes)>=MAX_NODES:raise PolicyError("node limit exceeds 65534")
        if not isinstance(obj,dict):raise PolicyError("node must be object")
        typ=obj.get("type")
        if typ not in NODE_TYPES:raise PolicyError(f"unknown/nonportable node type {typ}")
        common={"type","name","lifecycle","module","category","provenance"}
        fields={
            "Sequence":{"children"},"Selector":{"children"},"ReactiveSequence":{"children"},"ReactiveSelector":{"children"},
            "Parallel":{"children","success_threshold","failure_threshold"},"If":{"condition","then","else"},"ReactiveIf":{"condition","then","else"},
            "SwitchEnum":{"slot","cases","default"},"Invert":{"child"},"ForceSuccess":{"child"},"ForceFailure":{"child"},"TurnTimeout":{"child","turns"},
            "Retry":{"child","n"},"Repeat":{"child","n"},"KeepRunningUntilFailure":{"child"},"Success":set(),"Failure":set(),
            "Condition":{"lhs","op","rhs"},"SetBlackboard":{"slot","value"},"AddBlackboard":{"slot","value"},
            "EmitUnitIntent":{"unit","action","item","quantity","priority","authority"},"EmitMarketIntent":{"action","item","quantity","priority","authority"},
            "WaitTurns":{"turns"},"MoveUnitTo":{"unit","target","x","y","priority","authority"},
        }
        extra=set(obj)-common-fields[typ]
        if extra:raise PolicyError(f"unknown fields for {typ}: {sorted(extra)}")
        idx=len(self.nodes);node=Node(NODE_TYPES[typ]);self.nodes.append(node)
        module=obj.get("module")
        if module is not None:
            if not isinstance(module,str) or not module:raise PolicyError("module must be nonempty string")
            self.module_roots.append((idx,module,obj.get("category","generic"),copy.deepcopy(obj.get("provenance",{}))))
        children=self._node_children(typ,obj)
        child_ids=[self._compile_node(c,depth+1,f"{path}/{i}") for i,c in enumerate(children)]
        node.first_child=len(self.child_refs) if child_ids else 0;node.child_count=len(child_ids);self.child_refs.extend(child_ids)
        if len(self.child_refs)>MAX_CHILD_REFS:raise PolicyError("child-ref limit exceeded")
        if "lifecycle" in obj:node.lifecycle_slot=self._compile_lifecycle(obj["lifecycle"])
        if typ=="Parallel":
            node.p0=_fit_i32(obj.get("success_threshold",len(children)),"parallel success_threshold");node.p1=_fit_i32(obj.get("failure_threshold",1),"parallel failure_threshold")
            if not 1<=node.p0<=len(children) or not 1<=node.p1<=len(children):raise PolicyError("parallel threshold out of range")
        elif typ=="TurnTimeout":
            node.p0=_fit_i32(obj.get("turns"),"timeout turns")
            if not 1<=node.p0<=720:raise PolicyError("TurnTimeout turns must be 1..720")
        elif typ in ("Retry","Repeat"):
            node.p0=_fit_i32(obj.get("n"),f"{typ} n")
            if not 1<=node.p0<=255:raise PolicyError(f"{typ} n must be 1..255")
        elif typ=="Condition":
            op=obj.get("op")
            if op not in CMP:raise PolicyError("invalid comparison")
            lhs=self._compile_value_ref(obj.get("lhs"));lv=self.values[lhs];tag=lv.arg if lv.kind in (VALUE_KINDS["CONST"],VALUE_KINDS["BB"]) else 0
            rhs=self._compile_value_ref(obj.get("rhs"),lv.type,tag)
            if tag and op not in ("EQ","NE"):raise PolicyError("named enums support EQ/NE comparisons only")
            node.flags,node.p0,node.p1=CMP[op],lhs,rhs
        elif typ in ("SetBlackboard","AddBlackboard"):
            name=obj.get("slot")
            if name not in self.bb_by_name:raise PolicyError(f"unknown blackboard slot {name}")
            slot=self.bb_by_name[name];bb=self.bb[slot]
            if typ=="AddBlackboard" and bb.type!=BB_TYPES["I32"]:raise PolicyError("AddBlackboard requires I32")
            ref=self._compile_value_ref(obj.get("value"),_effective_type(bb),bb.enum_tag);node.p0=slot;node.p1=ref
        elif typ=="EmitUnitIntent":
            act=obj.get("action")
            if act not in UNIT_OPS:raise PolicyError("invalid unit action")
            unit=self._compile_value_ref(obj.get("unit",0),BB_TYPES["UNIT"])
            if act in ("PLANT","PICKUP","PLACE"):
                if "item" not in obj:raise PolicyError(f"{act} requires item")
                item=self._compile_value_ref(obj["item"],BB_TYPES["ITEM"])
            else:
                if "item" in obj:raise PolicyError(f"{act} does not accept item")
                item=self._compile_value_ref({"const":-1,"type":"ITEM"},BB_TYPES["ITEM"])
            if act in ("PICKUP","PLACE"):qty=self._compile_value_ref(obj.get("quantity",1),BB_TYPES["I32"])
            else:
                if "quantity" in obj:
                    q=self._make_value_ref(obj["quantity"],BB_TYPES["I32"],0,False)
                    if q.kind!=VALUE_KINDS["CONST"] or q.value!=1:raise PolicyError(f"{act} quantity must be constant 1")
                qty=self._compile_value_ref(1,BB_TYPES["I32"])
            node.p0,node.p1,node.p2,node.p3=unit,UNIT_OPS[act],item,qty;node.aux=_authority_key_from_node(obj)
        elif typ=="EmitMarketIntent":
            act=obj.get("action")
            if act not in MARKET_OPS:raise PolicyError("invalid market action")
            if act in ("HIRE","BUY_LAND"):
                if "item" in obj:raise PolicyError(f"{act} takes no item")
                item=self._compile_value_ref({"const":-1,"type":"ITEM"},BB_TYPES["ITEM"]);qty=self._compile_value_ref(1,BB_TYPES["I32"])
            else:
                if "item" not in obj:raise PolicyError(f"{act} requires item")
                item=self._compile_value_ref(obj["item"],BB_TYPES["ITEM"]);qty=self._compile_value_ref(obj.get("quantity",1),BB_TYPES["I32"])
            node.p0,node.p1,node.p2=MARKET_OPS[act],item,qty;node.aux=_authority_key_from_node(obj)
        elif typ=="WaitTurns":
            node.p0=_fit_i32(obj.get("turns"),"wait turns")
            if not 1<=node.p0<=720:raise PolicyError("WaitTurns turns must be 1..720")
        elif typ=="MoveUnitTo":
            unit=self._compile_value_ref(obj.get("unit",0),BB_TYPES["UNIT"])
            if "target" in obj:
                if "x" in obj or "y" in obj:raise PolicyError("MoveUnitTo uses target or x/y")
                target=self._compile_value_ref(obj["target"],BB_TYPES["TILE"])
            else:
                if "x" not in obj or "y" not in obj:raise PolicyError("MoveUnitTo requires target or x/y")
                x,y=_fit_i32(obj["x"],"move x"),_fit_i32(obj["y"],"move y")
                if not 0<=x<10 or not 0<=y<10:raise PolicyError("MoveUnitTo coordinates out of range")
                target=self._compile_value_ref({"const":y*10+x,"type":"TILE"},BB_TYPES["TILE"])
            node.p0,node.p1=unit,target;node.aux=_authority_key_from_node(obj)
        elif typ=="SwitchEnum":
            slot=self.bb_by_name[obj["slot"]];bb=self.bb[slot];node.p0=slot;node.p1=bb.enum_tag;node.p2=bb.enum_size
        node.subtree_end=len(self.nodes)
        return idx

    def _lint_source(self,obj:Any,path:str)->None:
        if not isinstance(obj,dict):return
        typ=obj.get("type")
        if typ in ("Selector","ReactiveSelector"):
            ch=obj.get("children",[])
            if isinstance(ch,list) and len(ch)>=24:
                self.warnings.append({"code":"LARGE_LINEAR_DISPATCH","path":path,"children":len(ch),"recommendation":"SwitchEnum"})
        if typ=="Condition" and obj.get("op")=="EQ":
            lhs=obj.get("lhs");rhs=obj.get("rhs");feature=lhs.get("feature") if isinstance(lhs,dict) else None
            if feature in ("STEP","OWN_MONEY","OPP_MONEY","MARKET_INVENTORY","MARKET_PRICE"):
                self.warnings.append({"code":"SPECIFIC_EXACT_EQUALITY","path":path,"feature":feature,"rhs":rhs})
        for key in ("children",):
            for i,c in enumerate(obj.get(key,[]) if isinstance(obj.get(key),list) else []):self._lint_source(c,f"{path}/{i}")
        for key in ("child","condition","then","else","default"):
            if key in obj:self._lint_source(obj[key],f"{path}/{key}")
        if typ=="SwitchEnum":
            for k,c in (obj.get("cases") or {}).items():self._lint_source(c,f"{path}/case:{k}")

    def _lint_bb_ownership(self)->None:
        writers={i:set() for i in range(len(self.bb))}
        module_for_node=["root"]*len(self.nodes)
        for root,name,_,_ in sorted(self.module_roots,key=lambda x:x[0]):
            end=self.nodes[root].subtree_end
            for i in range(root,end):module_for_node[i]=name
        for i,n in enumerate(self.nodes):
            if n.type in (NODE_TYPES["SetBlackboard"],NODE_TYPES["AddBlackboard"]):writers[n.p0].add(module_for_node[i])
            if n.lifecycle_slot!=NONE_U16:
                lc=self.lifecycles[n.lifecycle_slot]
                for start,count in (lc.on_success,lc.on_failure,lc.on_halted,lc.post):
                    for e in self.effects[start:start+count]:writers[e.slot].add(module_for_node[i])
        for slot,mods in writers.items():
            owner=self.bb[slot].owner
            foreign=sorted(m for m in mods if m!=owner)
            if foreign:self.warnings.append({"code":"BB_FOREIGN_WRITER","slot":self.bb[slot].name,"owner":owner,"writers":sorted(mods)})
            if len(mods)>1:self.warnings.append({"code":"BB_MULTI_MODULE_WRITER","slot":self.bb[slot].name,"writers":sorted(mods)})

    def _build_manifest(self,cp:CompiledPolicy)->Dict[str,Any]:
        modules=[]
        for root,name,category,prov in self.module_roots:
            end=cp.nodes[root].subtree_end;infos=cp.analysis[root]
            state_slots=[cp.nodes[i].state_slot for i in range(root,end) if cp.nodes[i].state_slot!=NONE_U16]
            modules.append({"name":name,"category":category,"node_interval":[root,end],"node_count":end-root,
                            "max_depth":infos.max_depth,"max_visits":infos.max_visits,"state_slots":state_slots,
                            "max_live_intents":infos.max_live_intents,"bb_reads":[cp.bb[i].name for i in sorted(infos.bb_reads)],
                            "bb_writes":[cp.bb[i].name for i in sorted(infos.bb_writes)],"features":[FEATURE_NAMES[i] for i in sorted(infos.features)],
                            "action_domains":sorted(infos.action_domains),"authorities":[AUTHORITY_NAMES.get(a,str(a)) for a in sorted(infos.authorities)],
                            "may_run":infos.may_run,"provenance":prov})
        return {"schema_version":4,"nodes":len(cp.nodes),"state_slots":cp.state_count,"bb_slots":len(cp.bb),
                "metadata":{"values":len(cp.values),"feature_args":len(cp.feature_args),"predicates":len(cp.predicates),"lifecycles":len(cp.lifecycles),"effects":len(cp.effects)},
                "max_depth":cp.max_depth,"max_visits":cp.analysis[0].max_visits,"instruction_budget":cp.instruction_budget,
                "max_live_intents":cp.max_live_intents,"modules":modules,"warnings":list(cp.warnings)}


def _value_dependencies(ref:ValueRef,feature_args:Sequence[FeatureArgRef])->Tuple[Set[int],Set[int]]:
    bb=set();features=set()
    if ref.kind==VALUE_KINDS["BB"]:bb.add(ref.id)
    elif ref.kind==VALUE_KINDS["LEGAL_FEATURE"]:
        features.add(ref.id)
        for a in feature_args[ref.arg:ref.arg+ref.value]:
            if a.kind==FEATURE_ARG_KINDS["BB"]:bb.add(a.id)
    return bb,features


def analyze_policy(nodes:Sequence[Node],child_refs:Sequence[int],values:Sequence[ValueRef],feature_args:Sequence[FeatureArgRef],predicates:Sequence[Predicate],lifecycles:Sequence[Lifecycle],effects:Sequence[Effect],bb:Sequence[BbDecl])->List[StaticInfo]:
    info=[StaticInfo() for _ in nodes]
    for i in range(len(nodes)-1,-1,-1):
        n=nodes[i];kids=child_refs[n.first_child:n.first_child+n.child_count];ci=[info[c] for c in kids]
        t=n.type;x=StaticInfo();x.max_depth=1+(max((q.max_depth for q in ci),default=0))
        if t in (NODE_TYPES["WaitTurns"],NODE_TYPES["MoveUnitTo"],NODE_TYPES["KeepRunningUntilFailure"]):x.may_run=True
        elif t in (NODE_TYPES["Retry"],NODE_TYPES["Repeat"]):x.may_run=n.p0>1 or bool(ci and ci[0].may_run)
        elif t in (NODE_TYPES["Sequence"],NODE_TYPES["Selector"],NODE_TYPES["ReactiveSequence"],NODE_TYPES["ReactiveSelector"],NODE_TYPES["If"],NODE_TYPES["Parallel"],NODE_TYPES["TurnTimeout"],NODE_TYPES["Invert"],NODE_TYPES["ForceSuccess"],NODE_TYPES["ForceFailure"]):x.may_run=any(q.may_run for q in ci)
        elif t==NODE_TYPES["ReactiveIf"]:
            if ci and ci[0].may_run:raise PolicyError("ReactiveIf condition must be synchronous")
            x.may_run=any(q.may_run for q in ci[1:])
        elif t==NODE_TYPES["SwitchEnum"]:x.may_run=any(q.may_run for q in ci)
        else:x.may_run=False
        semantic_state=t in (NODE_TYPES["Sequence"],NODE_TYPES["Selector"],NODE_TYPES["ReactiveSequence"],NODE_TYPES["ReactiveSelector"],NODE_TYPES["If"],NODE_TYPES["Parallel"],NODE_TYPES["ReactiveIf"],NODE_TYPES["SwitchEnum"],NODE_TYPES["Retry"],NODE_TYPES["Repeat"],NODE_TYPES["WaitTurns"])
        if t==NODE_TYPES["TurnTimeout"]:semantic_state=bool(ci and ci[0].may_run)
        x.needs_state=semantic_state and x.may_run
        if n.lifecycle_slot!=NONE_U16 and x.may_run:x.needs_state=True
        # Node-visit upper bound. BKG4 has no same-tick loops; exclusive control uses max where applicable.
        if t==NODE_TYPES["SwitchEnum"]:x.max_visits=1+max((q.max_visits for q in ci),default=0)
        elif t in (NODE_TYPES["If"],NODE_TYPES["ReactiveIf"]):x.max_visits=1+(ci[0].max_visits if ci else 0)+max((q.max_visits for q in ci[1:]),default=0)
        elif t in DECORATOR_TYPES:x.max_visits=1+(ci[0].max_visits if ci else 0)
        else:x.max_visits=1+sum(q.max_visits for q in ci)
        # Safe live-journal bound. Selector failures roll back before the next branch.
        if t in (NODE_TYPES["Selector"],NODE_TYPES["ReactiveSelector"],NODE_TYPES["SwitchEnum"]):x.max_live_intents=max((q.max_live_intents for q in ci),default=0)
        elif t in (NODE_TYPES["If"],NODE_TYPES["ReactiveIf"]):x.max_live_intents=(ci[0].max_live_intents if ci else 0)+max((q.max_live_intents for q in ci[1:]),default=0)
        elif t in DECORATOR_TYPES:x.max_live_intents=ci[0].max_live_intents if ci else 0
        else:x.max_live_intents=sum(q.max_live_intents for q in ci)
        if t in (NODE_TYPES["EmitUnitIntent"],NODE_TYPES["EmitMarketIntent"],NODE_TYPES["MoveUnitTo"]):x.max_live_intents=max(x.max_live_intents,1)
        # Cumulative emissions may exceed live journal because rolled-back branches still consume order numbers.
        x.max_emissions=sum(q.max_emissions for q in ci)
        if t in (NODE_TYPES["EmitUnitIntent"],NODE_TYPES["EmitMarketIntent"],NODE_TYPES["MoveUnitTo"]):x.max_emissions+=1
        for q in ci:
            x.bb_reads|=q.bb_reads;x.bb_writes|=q.bb_writes;x.features|=q.features;x.action_domains|=q.action_domains;x.authorities|=q.authorities
        def add_value(ref_idx:int,write:bool=False)->None:
            if not 0<=ref_idx<len(values):return
            bs,fs=_value_dependencies(values[ref_idx],feature_args);x.bb_reads.update(bs);x.features.update(fs)
        if t==NODE_TYPES["Condition"]:add_value(n.p0);add_value(n.p1)
        elif t in (NODE_TYPES["SetBlackboard"],NODE_TYPES["AddBlackboard"]):x.bb_writes.add(n.p0);add_value(n.p1)
        elif t==NODE_TYPES["EmitUnitIntent"]:add_value(n.p0);add_value(n.p2);add_value(n.p3);x.action_domains.add("unit");x.authorities.add((n.aux>>8)&0xFF)
        elif t==NODE_TYPES["EmitMarketIntent"]:add_value(n.p1);add_value(n.p2);x.action_domains.add("market");x.authorities.add((n.aux>>8)&0xFF)
        elif t==NODE_TYPES["MoveUnitTo"]:add_value(n.p0);add_value(n.p1);x.action_domains.add("unit");x.authorities.add((n.aux>>8)&0xFF);x.features.update({FEATURES["OWN_UNITS"],FEATURES["OWN_WORKER_X"],FEATURES["OWN_WORKER_Y"]})
        elif t==NODE_TYPES["SwitchEnum"]:x.bb_reads.add(n.p0)
        if n.lifecycle_slot!=NONE_U16:
            lc=lifecycles[n.lifecycle_slot]
            for pi in (lc.failure_if,lc.success_if,lc.skip_if,lc.while_pred):
                if pi!=NONE_U16:
                    q=predicates[pi];add_value(q.lhs);add_value(q.rhs)
            for start,count in (lc.on_success,lc.on_failure,lc.on_halted,lc.post):
                for e in effects[start:start+count]:
                    x.bb_writes.add(e.slot)
                    if e.op!=EFFECT_OP["clear"]:add_value(e.value)
        if x.max_emissions>65534:raise PolicyError("per-tick emission sequence exceeds uint16 capacity")
        info[i]=x
    return info


def _enum_sizes_from_bb(bb:Sequence[BbDecl])->Dict[int,int]:
    out={}
    for b in bb:
        if b.enum_tag:
            if b.type!=BB_TYPES["ENUM8"] or not 1<=b.enum_size<=255:raise PolicyError("named enum BB encoding invalid")
            if b.enum_tag in out and out[b.enum_tag]!=b.enum_size:raise PolicyError("named enum tag size conflict")
            out[b.enum_tag]=b.enum_size
        elif b.enum_size:raise PolicyError("raw BB has enum_size without enum_tag")
    if out and set(out)!=set(range(1,max(out)+1)):raise PolicyError("named enum tags must be dense")
    return out


def validate_compiled(p:CompiledPolicy)->None:
    if not p.nodes or len(p.nodes)>MAX_NODES:raise PolicyError("compiled node count invalid")
    if len(p.child_refs)!=len(p.nodes)-1 or len(p.child_refs)>MAX_CHILD_REFS:raise PolicyError("compiled strict-tree child count invalid")
    if p.feature_registry_version!=FEATURE_REGISTRY_VERSION:raise PolicyError("unsupported feature registry")
    if not 0<=p.state_count<=MAX_STATE or len(p.bb)>MAX_BB:raise PolicyError("compiled state/BB count exceeds limits")
    for table,name in ((p.values,"values"),(p.feature_args,"feature_args"),(p.predicates,"predicates"),(p.lifecycles,"lifecycles"),(p.effects,"effects")):
        if len(table)>MAX_TABLE:raise PolicyError(f"compiled {name} exceeds uint16 addressability")
    if not 1<=p.instruction_budget<=MAX_INSTRUCTION_BUDGET:raise PolicyError("instruction budget invalid")
    if not 0<=p.max_live_intents<=MAX_LIVE_INTENTS:raise PolicyError("max_live_intents invalid")
    if not 1<=p.max_depth<=MAX_DEPTH:raise PolicyError("max_depth invalid")
    enum_sizes=_enum_sizes_from_bb(p.bb)
    for i,b in enumerate(p.bb):
        if b.type not in BB_NAMES:raise PolicyError(f"BB {i} type invalid")
        _validate_bb_value(b.type,b.initial,f"BB {i} initial")
        if b.enum_tag and not 0<=b.initial<b.enum_size:raise PolicyError("named enum initial invalid")
    for i,a in enumerate(p.feature_args):
        if a.kind not in FEATURE_ARG_KINDS.values() or a.type not in VALUE_TYPES.values():raise PolicyError(f"FeatureArgRef {i} invalid")
        if a.kind==FEATURE_ARG_KINDS["CONST"]:
            if a.id!=0:_raise("FeatureArg CONST id noncanonical")
            _validate_bb_value(a.type,a.value,"FeatureArg CONST")
        else:
            if not 0<=a.id<len(p.bb) or a.value!=0 or p.bb[a.id].enum_tag or _effective_type(p.bb[a.id])!=a.type:raise PolicyError("FeatureArg BB invalid")
    used_feature_args:set[int]=set();used_values:set[int]=set();used_pred:set[int]=set();used_effects:set[int]=set();used_lc:set[int]=set();used_state:set[int]=set()
    for i,v in enumerate(p.values):
        if v.kind not in VALUE_KINDS.values() or v.type not in VALUE_TYPES.values():raise PolicyError(f"ValueRef {i} invalid")
        if v.kind==VALUE_KINDS["CONST"]:
            if v.id!=0:raise PolicyError("CONST ValueRef id noncanonical")
            if v.arg:
                if v.type!=BB_TYPES["I32"] or v.arg not in enum_sizes or not 0<=v.value<enum_sizes[v.arg]:raise PolicyError("named enum CONST invalid")
            else:_validate_bb_value(v.type,v.value,"CONST ValueRef")
        elif v.kind==VALUE_KINDS["BB"]:
            if not 0<=v.id<len(p.bb) or v.value!=0 or v.type!=_effective_type(p.bb[v.id]) or v.arg!=p.bb[v.id].enum_tag:raise PolicyError("BB ValueRef invalid")
        else:
            if v.id not in FEATURE_NAMES or v.type!=FEATURE_TYPES[v.id] or v.value!=len(FEATURE_ARG_TYPES[v.id]):raise PolicyError("LEGAL_FEATURE ValueRef invalid")
            if v.value==0:
                if v.arg!=0:raise PolicyError("zero-arg feature range noncanonical")
            elif v.arg<0 or v.arg+v.value>len(p.feature_args):raise PolicyError("feature arg range invalid")
            for pos,typ in enumerate(FEATURE_ARG_TYPES[v.id]):
                ai=v.arg+pos;used_feature_args.add(ai);a=p.feature_args[ai]
                if a.type!=typ:raise PolicyError("feature arg type mismatch")
                if a.kind==FEATURE_ARG_KINDS["CONST"] and not _feature_const_arg_ok(v.id,pos,a.value):raise PolicyError("feature const arg invalid")
    for i,q in enumerate(p.predicates):
        if q.cmp not in CMP_NAMES or not 0<=q.lhs<len(p.values) or not 0<=q.rhs<len(p.values):raise PolicyError("predicate invalid")
        a,b=p.values[q.lhs],p.values[q.rhs]
        if a.type!=b.type:raise PolicyError("predicate type mismatch")
        at=a.arg if a.kind in (VALUE_KINDS["CONST"],VALUE_KINDS["BB"]) else 0;bt=b.arg if b.kind in (VALUE_KINDS["CONST"],VALUE_KINDS["BB"]) else 0
        if at!=bt:raise PolicyError("predicate enum tag mismatch")
        if at and q.cmp not in (CMP["EQ"],CMP["NE"]):raise PolicyError("named enum predicate only supports EQ/NE")
    for i,e in enumerate(p.effects):
        if e.op not in EFFECT_OP.values() or not 0<=e.slot<len(p.bb):raise PolicyError("effect invalid")
        if e.op==EFFECT_OP["clear"]:
            if e.value!=0:raise PolicyError("CLEAR must encode zero value")
        else:
            if not 0<=e.value<len(p.values):raise PolicyError("effect ValueRef invalid")
            b=p.bb[e.slot];v=p.values[e.value]
            if v.type!=_effective_type(b) or (v.arg if v.kind in (VALUE_KINDS["CONST"],VALUE_KINDS["BB"]) else 0)!=b.enum_tag:raise PolicyError("effect ValueRef type/tag mismatch")
            if e.op==EFFECT_OP["add"] and b.type!=BB_TYPES["I32"]:raise PolicyError("ADD effect requires I32")
    parents=[0]*len(p.nodes);child_used=[0]*len(p.child_refs)
    def require_ref(r:int,typ:int,tag:int=0)->None:
        if not 0<=r<len(p.values):raise PolicyError("node ValueRef out of range")
        v=p.values[r];actual_tag=v.arg if v.kind in (VALUE_KINDS["CONST"],VALUE_KINDS["BB"]) else 0
        if v.type!=typ or actual_tag!=tag:raise PolicyError("node ValueRef type/tag mismatch")
        used_values.add(r)
    for i,n in enumerate(p.nodes):
        if n.type not in NODE_NAMES or not i<n.subtree_end<=len(p.nodes):raise PolicyError(f"node {i} encoding invalid")
        if n.first_child<0 or n.child_count<0 or n.first_child+n.child_count>len(p.child_refs) or (n.child_count==0 and n.first_child!=0):raise PolicyError("child range invalid")
        kids=p.child_refs[n.first_child:n.first_child+n.child_count]
        for pos in range(n.first_child,n.first_child+n.child_count):
            if child_used[pos]:raise PolicyError("child-ref range alias")
            child_used[pos]=1
        for c in kids:
            if not 0<=c<len(p.nodes):raise PolicyError("child id invalid")
            parents[c]+=1
            if parents[c]>1:raise PolicyError("runtime DAG/multiple parent forbidden")
        if n.state_slot!=NONE_U16:
            if not 0<=n.state_slot<p.state_count or n.state_slot in used_state:raise PolicyError("state slot invalid/aliased")
            used_state.add(n.state_slot)
        if n.lifecycle_slot!=NONE_U16:
            if not 0<=n.lifecycle_slot<len(p.lifecycles):raise PolicyError("lifecycle slot invalid")
            used_lc.add(n.lifecycle_slot)
        t=n.type
        if t in (NODE_TYPES["Sequence"],NODE_TYPES["Selector"],NODE_TYPES["ReactiveSequence"],NODE_TYPES["ReactiveSelector"]):
            if n.child_count<1:raise PolicyError("empty composite")
        elif t in (NODE_TYPES["If"],NODE_TYPES["ReactiveIf"]):
            if n.child_count!=3:raise PolicyError("If/ReactiveIf cardinality invalid")
        elif t==NODE_TYPES["SwitchEnum"]:
            if not 0<=n.p0<len(p.bb) or not p.bb[n.p0].enum_tag or n.p1!=p.bb[n.p0].enum_tag or n.p2!=p.bb[n.p0].enum_size or n.child_count!=n.p2+1:raise PolicyError("SwitchEnum encoding invalid")
        elif t==NODE_TYPES["Parallel"]:
            if not 2<=n.child_count<=16 or not 1<=n.p0<=n.child_count or not 1<=n.p1<=n.child_count:raise PolicyError("Parallel invalid")
        elif t in DECORATOR_TYPES:
            if n.child_count!=1:raise PolicyError("decorator cardinality invalid")
        elif n.child_count:raise PolicyError("leaf has children")
        if t==NODE_TYPES["Condition"]:require_ref(n.p0,p.values[n.p1].type if 0<=n.p1<len(p.values) else -1,p.values[n.p1].arg if 0<=n.p1<len(p.values) and p.values[n.p1].kind in (VALUE_KINDS["CONST"],VALUE_KINDS["BB"]) else 0);require_ref(n.p1,p.values[n.p0].type,p.values[n.p0].arg if p.values[n.p0].kind in (VALUE_KINDS["CONST"],VALUE_KINDS["BB"]) else 0)
        elif t in (NODE_TYPES["SetBlackboard"],NODE_TYPES["AddBlackboard"]):
            if not 0<=n.p0<len(p.bb):raise PolicyError("BB leaf slot invalid")
            b=p.bb[n.p0];require_ref(n.p1,_effective_type(b),b.enum_tag)
        elif t==NODE_TYPES["EmitUnitIntent"]:
            require_ref(n.p0,BB_TYPES["UNIT"]);require_ref(n.p2,BB_TYPES["ITEM"]);require_ref(n.p3,BB_TYPES["I32"]);_validate_priority_key(n.aux);_validate_unit_static(p,n)
        elif t==NODE_TYPES["EmitMarketIntent"]:
            require_ref(n.p1,BB_TYPES["ITEM"]);require_ref(n.p2,BB_TYPES["I32"]);_validate_priority_key(n.aux);_validate_market_static(p,n)
        elif t==NODE_TYPES["MoveUnitTo"]:
            require_ref(n.p0,BB_TYPES["UNIT"]);require_ref(n.p1,BB_TYPES["TILE"]);_validate_priority_key(n.aux)
        elif t==NODE_TYPES["WaitTurns"] and not 1<=n.p0<=720:raise PolicyError("WaitTurns bound invalid")
        elif t==NODE_TYPES["TurnTimeout"] and not 1<=n.p0<=720:raise PolicyError("TurnTimeout bound invalid")
        elif t in (NODE_TYPES["Retry"],NODE_TYPES["Repeat"]) and not 1<=n.p0<=255:raise PolicyError("Retry/Repeat bound invalid")
    if parents[0]!=0 or any(x!=1 for x in parents[1:]):raise PolicyError("strict tree parent invariant invalid")
    if child_used!=[1]*len(p.child_refs):raise PolicyError("child-ref table not dense/used")
    preorder=[]
    def walk(i:int,d:int)->None:
        if d>MAX_DEPTH or i!=len(preorder):raise PolicyError("canonical preorder/depth invalid")
        preorder.append(i);n=p.nodes[i]
        for c in p.child_refs[n.first_child:n.first_child+n.child_count]:walk(c,d+1)
        if n.subtree_end!=len(preorder):raise PolicyError("subtree_end noncanonical")
    walk(0,1)
    if len(preorder)!=len(p.nodes):raise PolicyError("unreachable nodes")
    # Recompute all safety proofs independently of compiler-assigned header/state metadata.
    analysis=analyze_policy(p.nodes,p.child_refs,p.values,p.feature_args,p.predicates,p.lifecycles,p.effects,p.bb)
    expected_state=0
    for i,a in enumerate(analysis):
        n=p.nodes[i]
        if a.needs_state:
            if n.state_slot!=expected_state:raise PolicyError("state-elision proof/state assignment mismatch")
            expected_state+=1
        elif n.state_slot!=NONE_U16:raise PolicyError("unnecessary state slot")
    if expected_state!=p.state_count or used_state!=set(range(p.state_count)):raise PolicyError("state count/density mismatch")
    if p.max_depth!=analysis[0].max_depth:raise PolicyError("header max_depth inconsistent with proof")
    if p.max_live_intents!=analysis[0].max_live_intents:raise PolicyError("header max_live_intents inconsistent with proof")
    if p.instruction_budget<analysis[0].max_visits:raise PolicyError("instruction budget under static visit proof")
    # All immutable table entries must be referenced at least once; aliasing is explicitly legal in BKG4.
    for n in p.nodes:
        if n.lifecycle_slot!=NONE_U16:used_lc.add(n.lifecycle_slot)
    for lc in p.lifecycles:
        for q in (lc.failure_if,lc.success_if,lc.skip_if,lc.while_pred):
            if q!=NONE_U16:used_pred.add(q)
        for start,count in (lc.on_success,lc.on_failure,lc.on_halted,lc.post):
            for ei in range(start,start+count):used_effects.add(ei)
    for q in p.predicates:used_values.update((q.lhs,q.rhs))
    for e in p.effects:
        if e.op!=EFFECT_OP["clear"]:used_values.add(e.value)
    if used_feature_args!=set(range(len(p.feature_args))):raise PolicyError("FeatureArg table contains unused records")
    if used_values!=set(range(len(p.values))):raise PolicyError("ValueRef table contains unused records")
    if used_pred!=set(range(len(p.predicates))):raise PolicyError("predicate table contains unused records")
    if used_effects!=set(range(len(p.effects))):raise PolicyError("effect table contains unused records")
    if used_lc!=set(range(len(p.lifecycles))):raise PolicyError("lifecycle table contains unused records")


def _raise(msg:str)->None:raise PolicyError(msg)

def _validate_priority_key(k:int)->None:
    if not 0<=k<=0xFFFF or ((k>>8)&0xFF)>MAX_AUTHORITY:raise PolicyError("invalid authority priority key")

def _const(p:CompiledPolicy,idx:int)->Optional[int]:
    return p.values[idx].value if 0<=idx<len(p.values) and p.values[idx].kind==VALUE_KINDS["CONST"] else None

def _validate_unit_static(p:CompiledPolicy,n:Node)->None:
    if n.p1 not in UNIT_OPS.values():raise PolicyError("unit op invalid")
    item,qty=p.values[n.p2],p.values[n.p3];ic=_const(p,n.p2);qc=_const(p,n.p3)
    if n.p1==UNIT_OPS["PLANT"]:
        if qty.kind!=VALUE_KINDS["CONST"] or qty.value!=1 or (ic is not None and not 0<=ic<N_CROPS):raise PolicyError("PLANT operands noncanonical")
    elif n.p1 in (UNIT_OPS["PICKUP"],UNIT_OPS["PLACE"]):
        if ic is not None and not 0<=ic<len(ITEMS):raise PolicyError("PICKUP/PLACE item invalid")
        if qc is not None and not 1<=qc<=32767:raise PolicyError("PICKUP/PLACE quantity invalid")
    else:
        if item.kind!=VALUE_KINDS["CONST"] or item.value!=-1 or qty.kind!=VALUE_KINDS["CONST"] or qty.value!=1:raise PolicyError("unit hidden operands noncanonical")

def _validate_market_static(p:CompiledPolicy,n:Node)->None:
    if n.p0 not in MARKET_OPS.values():raise PolicyError("market op invalid")
    item,qty=p.values[n.p1],p.values[n.p2];ic=_const(p,n.p1);qc=_const(p,n.p2)
    if n.p0 in (MARKET_OPS["HIRE"],MARKET_OPS["BUY_LAND"]):
        if item.kind!=VALUE_KINDS["CONST"] or item.value!=-1 or qty.kind!=VALUE_KINDS["CONST"] or qty.value!=1:raise PolicyError("HIRE/BUY_LAND operands noncanonical")
    else:
        if ic is not None:
            if n.p0==MARKET_OPS["BUY_SEED"] and not 0<=ic<N_CROPS:raise PolicyError("BUY_SEED item invalid")
            if n.p0==MARKET_OPS["BUY_PRODUCT"] and ic not in (ITEM_ID["WHEAT"],ITEM_ID["FERTILIZER"]):raise PolicyError("BUY_PRODUCT item invalid")
            if n.p0==MARKET_OPS["BUY_ANIMAL"] and not ITEM_ID["GOOSE"]<=ic<=ITEM_ID["SHEEP"]:raise PolicyError("BUY_ANIMAL item invalid")
            if n.p0==MARKET_OPS["SELL"] and not 0<=ic<N_PRODUCTS:raise PolicyError("SELL item invalid")
        if qc is not None and not 1<=qc<=65535:raise PolicyError("market quantity invalid")


def compile_policy(policy:Dict[str,Any])->CompiledPolicy:return Compiler(policy).compile()


def semantic_payload_bytes(p:CompiledPolicy)->bytes:
    data=bytearray(SEMANTIC_DOMAIN)
    def u8(v:int):data.extend(struct.pack("<B",v))
    def u16(v:int):data.extend(struct.pack("<H",v))
    def i32(v:int):data.extend(struct.pack("<i",v))
    for v in (ABI_VERSION,p.feature_registry_version,len(p.nodes),p.state_count,len(p.bb),len(p.values),len(p.feature_args),len(p.predicates),len(p.lifecycles),len(p.effects),p.instruction_budget,p.max_live_intents,p.max_depth,BOARD_SIZE):u16(v)
    for n in p.nodes:
        u8(n.type);u8(n.flags)
        for v in (n.first_child,n.child_count,n.state_slot,n.lifecycle_slot,n.subtree_end):u16(v)
        for v in (n.p0,n.p1,n.p2,n.p3):i32(v)
        data.extend(struct.pack("<I",n.aux))
    for c in p.child_refs:u16(c)
    for b in p.bb:u8(b.type);u8(b.enum_size);u16(b.enum_tag);i32(b.initial)
    for v in p.values:u8(v.kind);u8(v.type);u16(v.id);i32(v.arg);i32(v.value)
    for a in p.feature_args:u8(a.kind);u8(a.type);u16(a.id);i32(a.value)
    for q in p.predicates:u16(q.lhs);u16(q.rhs);u8(q.cmp)
    for lc in p.lifecycles:
        for v in (lc.failure_if,lc.success_if,lc.skip_if,lc.while_pred,lc.on_success[0],lc.on_success[1],lc.on_failure[0],lc.on_failure[1],lc.on_halted[0],lc.on_halted[1],lc.post[0],lc.post[1]):u16(v)
    for e in p.effects:u8(e.op);u8(e.slot);i32(e.value)
    return bytes(data)


def _compiled_image(p:CompiledPolicy,digest:bytes)->bytes:
    data=bytearray(HEADER.pack(MAGIC,ABI_VERSION,HEADER.size,digest,len(p.nodes),p.state_count,len(p.bb),len(p.values),len(p.feature_args),len(p.predicates),len(p.lifecycles),len(p.effects),p.instruction_budget,p.max_live_intents,p.max_depth,p.feature_registry_version))
    for n in p.nodes:data+=NODE_STRUCT.pack(n.type,n.flags,n.first_child,n.child_count,n.state_slot,n.lifecycle_slot,n.subtree_end,n.p0,n.p1,n.p2,n.p3,n.aux)
    for c in p.child_refs:data+=struct.pack("<H",c)
    for b in p.bb:data+=BB_STRUCT.pack(b.type,b.enum_size,b.enum_tag,b.initial)
    for v in p.values:data+=VALUE_STRUCT.pack(v.kind,v.type,v.id,v.arg,v.value)
    for a in p.feature_args:data+=FEATURE_ARG_STRUCT.pack(a.kind,a.type,a.id,a.value)
    for q in p.predicates:data+=PRED_STRUCT.pack(q.lhs,q.rhs,q.cmp,b"\0"*7)
    for lc in p.lifecycles:data+=LIFECYCLE_STRUCT.pack(lc.failure_if,lc.success_if,lc.skip_if,lc.while_pred,lc.on_success[0],lc.on_success[1],lc.on_failure[0],lc.on_failure[1],lc.on_halted[0],lc.on_halted[1],lc.post[0],lc.post[1])
    for e in p.effects:data+=EFFECT_STRUCT.pack(e.op,e.slot,0,e.value)
    return bytes(data)


def save_compiled(p:CompiledPolicy,path:str|Path,manifest_path:str|Path|None=None)->None:
    validate_compiled(p);digest=hashlib.sha256(semantic_payload_bytes(p)).digest();p.semantic_hash=digest.hex();Path(path).write_bytes(_compiled_image(p,digest))
    if manifest_path is not None:Path(manifest_path).write_text(json.dumps(p.manifest,sort_keys=True,indent=2)+"\n",encoding="utf-8")


def load_compiled(path:str|Path)->CompiledPolicy:
    raw=Path(path).read_bytes()
    if len(raw)<HEADER.size:raise PolicyError("compiled file too short")
    f=HEADER.unpack_from(raw,0);magic,version,hsize,digest=f[:4]
    if magic!=MAGIC or version!=ABI_VERSION or hsize!=HEADER.size:raise PolicyError("invalid BKG4 header")
    nn,ns,nb,nv,nfa,np,nl,ne,budget,max_live,max_depth,registry=f[4:]
    if not nn or nn>MAX_NODES or ns>MAX_STATE or nb>MAX_BB or max_live>MAX_LIVE_INTENTS or max_depth>MAX_DEPTH or registry!=FEATURE_REGISTRY_VERSION:raise PolicyError("header counts/proofs invalid")
    for count in (nv,nfa,np,nl,ne):
        if count>MAX_TABLE:raise PolicyError("metadata count exceeds uint16 addressability")
    nc=nn-1;expected=HEADER.size+nn*NODE_STRUCT.size+nc*2+nb*BB_STRUCT.size+nv*VALUE_STRUCT.size+nfa*FEATURE_ARG_STRUCT.size+np*PRED_STRUCT.size+nl*LIFECYCLE_STRUCT.size+ne*EFFECT_STRUCT.size
    if len(raw)!=expected:raise PolicyError("compiled file size/count mismatch or trailing bytes")
    off=HEADER.size;nodes=[]
    for _ in range(nn):nodes.append(Node(*NODE_STRUCT.unpack_from(raw,off)));off+=NODE_STRUCT.size
    child_refs=list(struct.unpack_from(f"<{nc}H",raw,off)) if nc else [];off+=nc*2
    bb=[]
    for _ in range(nb):bb.append(BbDecl(*BB_STRUCT.unpack_from(raw,off)));off+=BB_STRUCT.size
    values=[]
    for _ in range(nv):values.append(ValueRef(*VALUE_STRUCT.unpack_from(raw,off)));off+=VALUE_STRUCT.size
    args=[]
    for _ in range(nfa):args.append(FeatureArgRef(*FEATURE_ARG_STRUCT.unpack_from(raw,off)));off+=FEATURE_ARG_STRUCT.size
    preds=[]
    for _ in range(np):
        a,b,c,pad=PRED_STRUCT.unpack_from(raw,off);off+=PRED_STRUCT.size
        if pad!=b"\0"*7:raise PolicyError("Predicate padding nonzero")
        preds.append(Predicate(a,b,c))
    lcs=[]
    for _ in range(nl):
        x=LIFECYCLE_STRUCT.unpack_from(raw,off);off+=LIFECYCLE_STRUCT.size;lcs.append(Lifecycle(x[0],x[1],x[2],x[3],(x[4],x[5]),(x[6],x[7]),(x[8],x[9]),(x[10],x[11])))
    effects=[]
    for _ in range(ne):
        op,slot,res,val=EFFECT_STRUCT.unpack_from(raw,off);off+=EFFECT_STRUCT.size
        if res:raise PolicyError("Effect reserved field nonzero")
        effects.append(Effect(op,slot,val))
    cp=CompiledPolicy(digest.hex(),nodes,child_refs,bb,values,args,preds,lcs,effects,ns,budget,max_live,max_depth,registry)
    validate_compiled(cp);actual=hashlib.sha256(semantic_payload_bytes(cp)).digest()
    if actual!=digest:raise PolicyError("compiled semantic hash mismatch")
    cp.analysis=analyze_policy(cp.nodes,cp.child_refs,cp.values,cp.feature_args,cp.predicates,cp.lifecycles,cp.effects,cp.bb)
    return cp


class VM:
    def __init__(self,p:CompiledPolicy):
        validate_compiled(p);self.p=p;self.state=RuntimeState([],[]);self.reset();self._obs:Any=None;self._intents:List[DeferredIntent]=[];self._emit_seq=0;self._visits=0;self._trace=None;self._profile=None;self._depth=0
    def reset(self)->RuntimeState:
        self.state=RuntimeState([b.initial for b in self.p.bb],[NodeState() for _ in range(self.p.state_count)],0,0);return self.state
    def tick(self,observation:Any,max_market_orders:int=10,trace:bool=False,profile:bool=False)->TickResult:
        self._obs=observation if hasattr(observation,"feature") else DictObservation(observation);self._intents=[];self._emit_seq=0;self._visits=0;self._trace=[] if trace else None;self._profile=TickProfile() if profile else None;self._depth=0
        own_units=max(0,min(MAX_UNITS,self._feature(FEATURES["OWN_UNITS"])))
        raw=STATUS_FAILURE if self.state.fault else self._eval_node(0)
        if self.state.fault or raw in (STATUS_FAILURE,STATUS_SKIPPED):self._intents=[]
        unit_claims:[Optional[UnitClaim]]=[None]*MAX_UNITS;market_claims:List[MarketClaim]=[]
        if not self.state.fault and raw in (STATUS_SUCCESS,STATUS_RUNNING):
            for x in self._intents:
                if not (x.subject_kind&0x80):
                    u=x.subject_kind
                    if 0<=u<MAX_UNITS:
                        old=unit_claims[u]
                        if old is None or x.priority_key>old.priority_key or (x.priority_key==old.priority_key and x.order<old.order):unit_claims[u]=UnitClaim(x.priority_key,x.order,x.op,x.arg,x.quantity)
                else:market_claims.append(MarketClaim(x.priority_key,x.order,x.op,x.arg,x.quantity))
            market_claims.sort(key=lambda z:(-z.priority_key,z.order));del market_claims[MAX_MARKET_INTENTS:]
        units=[(c.op,c.arg,c.quantity) if c else (UNIT_OPS["PASS"],0,1) for c in unit_claims[:own_units]]
        cap=max(0,min(MAX_MARKET_INTENTS,int(max_market_orders)));market=[(c.op,c.item,c.quantity) for c in market_claims[:cap]]
        status=STATUS_FAILURE if self.state.fault else STATUS_SUCCESS if raw==STATUS_SKIPPED else raw;self.state.ticks+=1
        return TickResult(status,ActionResult(own_units,units,market),self.state.fault,tuple(self.state.blackboard),tuple((s.a&0xFFFFFFFF,s.b&0xFFFFFFFF) for s in self.state.node_state),tuple(self._trace or ()),self._profile)
    def _feature(self,fid:int,a0:int=0,a1:int=0)->int:
        if self._profile:self._profile.feature_evals+=1
        return int(self._obs.feature(fid,a0,a1))
    def _value(self,i:int)->int:
        v=self.p.values[i]
        if v.kind==VALUE_KINDS["CONST"]:return v.value
        if v.kind==VALUE_KINDS["BB"]:return self.state.blackboard[v.id]
        args=[]
        for j in range(v.value):
            a=self.p.feature_args[v.arg+j];args.append(a.value if a.kind==FEATURE_ARG_KINDS["CONST"] else self.state.blackboard[a.id])
        return self._feature(v.id,args[0] if args else 0,args[1] if len(args)>1 else 0)
    def _cmp(self,a:int,b:int,c:int)->bool:return (a==b,a!=b,a<b,a<=b,a>b,a>=b)[c]
    def _pred(self,i:int)->bool:
        q=self.p.predicates[i];return self._cmp(self._value(q.lhs),self._value(q.rhs),q.cmp)
    def _set_fault(self,c:int)->None:
        if not self.state.fault:self.state.fault=c
    def _write(self,slot:int,v:int)->bool:
        try:_validate_bb_value(self.p.bb[slot].type,v,"runtime BB write")
        except PolicyError:self._set_fault(5);return False
        b=self.p.bb[slot]
        if b.enum_tag and not 0<=v<b.enum_size:self._set_fault(5);return False
        self.state.blackboard[slot]=v;return True
    def _effects(self,rng:Tuple[int,int])->None:
        for e in self.p.effects[rng[0]:rng[0]+rng[1]]:
            if self.state.fault:return
            if e.op==EFFECT_OP["set"]:self._write(e.slot,self._value(e.value))
            elif e.op==EFFECT_OP["clear"]:self.state.blackboard[e.slot]=self.p.bb[e.slot].initial
            else:
                v=self.state.blackboard[e.slot]+self._value(e.value)
                if not -(1<<31)<=v<(1<<31):self._set_fault(2);return
                self.state.blackboard[e.slot]=v
    def _children(self,n:Node)->Sequence[int]:return self.p.child_refs[n.first_child:n.first_child+n.child_count]
    def _st(self,n:Node)->Optional[NodeState]:return self.state.node_state[n.state_slot] if n.state_slot!=NONE_U16 else None
    def _active(self,n:Node)->bool:
        st=self._st(n);return bool(st and st.b&ACTIVE_BIT)
    def _set_active(self,n:Node,v:bool)->None:
        st=self._st(n)
        if st:st.b=(st.b|ACTIVE_BIT) if v else (st.b&~ACTIVE_BIT)
    def _trace_event(self,i:int,r:int,event:str="tick")->None:
        if self._trace is not None:self._trace.append((self._feature(FEATURES["STEP"]),i,event,STATUS_NAMES[r]))
        if self._profile:self._profile.statuses.setdefault(i,[0,0,0,0])[r]+=1
    def _cancel(self,root:int)->None:
        hi=self.p.nodes[root].subtree_end;self._intents[:]=[x for x in self._intents if not root<=x.source<hi]
    def _halt_node(self,i:int,cancel:bool=True)->None:
        if cancel:self._cancel(i)
        n=self.p.nodes[i];kids=self._children(n);st=self._st(n);was=self._active(n)
        if n.type in (NODE_TYPES["Sequence"],NODE_TYPES["Selector"]) and st and st.a<len(kids):self._halt_node(kids[st.a],False)
        elif n.type in (NODE_TYPES["ReactiveSequence"],NODE_TYPES["ReactiveSelector"]) and st and st.a and st.a-1<len(kids):self._halt_node(kids[st.a-1],False)
        elif n.type==NODE_TYPES["If"] and st and kids:
            phase=st.a
            if phase==0:self._halt_node(kids[0],False)
            elif phase in (1,2):self._halt_node(kids[phase],False)
        elif n.type==NODE_TYPES["ReactiveIf"] and st and st.a in (1,2):self._halt_node(kids[st.a],False)
        elif n.type==NODE_TYPES["SwitchEnum"] and st and st.a and st.a-1<len(kids):self._halt_node(kids[st.a-1],False)
        elif n.type==NODE_TYPES["Parallel"] and st:
            running=st.b&RUNNING_MASK16
            for k,c in enumerate(kids):
                if running&(1<<k):self._halt_node(c,False)
        elif n.type in DECORATOR_TYPES and kids:self._halt_node(kids[0],False)
        if st:st.a=0;st.b=0
        if was and n.lifecycle_slot!=NONE_U16:self._effects(self.p.lifecycles[n.lifecycle_slot].on_halted)
        if was:self._trace_event(i,STATUS_SKIPPED,"halt")
        if self._profile:self._profile.halts[i]=self._profile.halts.get(i,0)+1
    def _eval_node(self,i:int)->int:
        if self.state.fault:return STATUS_FAILURE
        checkpoint=len(self._intents);self._visits+=1;self._depth+=1
        if self._profile:
            self._profile.visits+=1;self._profile.max_depth=max(self._profile.max_depth,self._depth);self._profile.node_visits[i]=self._profile.node_visits.get(i,0)+1
        if self._visits>self.p.instruction_budget:self._set_fault(1);self._depth-=1;return STATUS_FAILURE
        n=self.p.nodes[i];lc=self.p.lifecycles[n.lifecycle_slot] if n.lifecycle_slot!=NONE_U16 else None;active=self._active(n)
        if lc:
            if not active:
                if lc.failure_if!=NONE_U16 and self._pred(lc.failure_if):self._effects(lc.on_failure);self._effects(lc.post);r=STATUS_FAILURE;self._trace_event(i,r,"precondition");del self._intents[checkpoint:];self._depth-=1;return r
                if lc.success_if!=NONE_U16 and self._pred(lc.success_if):self._effects(lc.on_success);self._effects(lc.post);r=STATUS_SUCCESS;self._trace_event(i,r,"precondition");self._depth-=1;return r
                if lc.skip_if!=NONE_U16 and self._pred(lc.skip_if):r=STATUS_SKIPPED;self._trace_event(i,r,"precondition");del self._intents[checkpoint:];self._depth-=1;return r
                if lc.while_pred!=NONE_U16 and not self._pred(lc.while_pred):r=STATUS_SKIPPED;self._trace_event(i,r,"precondition");del self._intents[checkpoint:];self._depth-=1;return r
            elif lc.while_pred!=NONE_U16 and not self._pred(lc.while_pred):self._halt_node(i);r=STATUS_SKIPPED;self._trace_event(i,r,"while_halt");del self._intents[checkpoint:];self._depth-=1;return r
        r=self._core(i,n)
        if self.state.fault:r=STATUS_FAILURE
        if r==STATUS_RUNNING:self._set_active(n,True)
        else:
            self._set_active(n,False)
            if lc and r==STATUS_SUCCESS:self._effects(lc.on_success);self._effects(lc.post)
            elif lc and r==STATUS_FAILURE:self._effects(lc.on_failure);self._effects(lc.post)
        if r in (STATUS_FAILURE,STATUS_SKIPPED) or self.state.fault:del self._intents[checkpoint:]
        self._trace_event(i,r);self._depth-=1;return STATUS_FAILURE if self.state.fault else r
    def _core(self,i:int,n:Node)->int:
        t=n.type;kids=self._children(n);st=self._st(n)
        if t==NODE_TYPES["Sequence"]:
            k=int(st.a) if st else 0
            while k<len(kids):
                r=self._eval_node(kids[k])
                if r in (STATUS_SUCCESS,STATUS_SKIPPED):k+=1;st and setattr(st,"a",k);continue
                if r==STATUS_RUNNING:
                    if st is None:self._set_fault(3);return STATUS_FAILURE
                    st.a=k;return r
                if st:st.a=0
                return STATUS_FAILURE
            if st:st.a=0
            return STATUS_SUCCESS
        if t==NODE_TYPES["Selector"]:
            k=int(st.a) if st else 0
            while k<len(kids):
                r=self._eval_node(kids[k])
                if r in (STATUS_FAILURE,STATUS_SKIPPED):k+=1;st and setattr(st,"a",k);continue
                if r==STATUS_RUNNING:
                    if st is None:self._set_fault(3);return STATUS_FAILURE
                    st.a=k;return r
                if st:st.a=0
                return STATUS_SUCCESS
            if st:st.a=0
            return STATUS_FAILURE
        if t in (NODE_TYPES["ReactiveSequence"],NODE_TYPES["ReactiveSelector"]):
            prev=int(st.a)-1 if st and st.a else -1
            for k,c in enumerate(kids):
                r=self._eval_node(c)
                if k==prev:prev=-1
                good=(r in (STATUS_SUCCESS,STATUS_SKIPPED)) if t==NODE_TYPES["ReactiveSequence"] else (r in (STATUS_FAILURE,STATUS_SKIPPED))
                if good:continue
                if prev>=0:self._halt_node(kids[prev])
                if r==STATUS_RUNNING:
                    if st is None:self._set_fault(3);return STATUS_FAILURE
                    st.a=k+1
                elif st:st.a=0
                return r
            if prev>=0:self._halt_node(kids[prev])
            if st:st.a=0
            return STATUS_SUCCESS if t==NODE_TYPES["ReactiveSequence"] else STATUS_FAILURE
        if t==NODE_TYPES["If"]:
            phase=int(st.a) if st else 0
            if phase==0:
                r=self._eval_node(kids[0])
                if r==STATUS_RUNNING:
                    if st is None:self._set_fault(3);return STATUS_FAILURE
                    return r
                if r==STATUS_SKIPPED:return r
                phase=1 if r==STATUS_SUCCESS else 2
                if st:st.a=phase
            r=self._eval_node(kids[phase])
            if r==STATUS_RUNNING and st is None:self._set_fault(3);return STATUS_FAILURE
            if r!=STATUS_RUNNING and st:st.a=0
            return r
        if t==NODE_TYPES["ReactiveIf"]:
            cr=self._eval_node(kids[0])
            if cr==STATUS_RUNNING:self._set_fault(3);return STATUS_FAILURE
            if cr==STATUS_SKIPPED:return cr
            sel=1 if cr==STATUS_SUCCESS else 2;old=st.a if st and self._active(n) else 0
            if old in (1,2) and old!=sel:self._halt_node(kids[old])
            r=self._eval_node(kids[sel])
            if r==STATUS_RUNNING:
                if st is None:self._set_fault(3);return STATUS_FAILURE
                st.a=sel
            elif st:st.a=0
            return r
        if t==NODE_TYPES["SwitchEnum"]:
            v=self.state.blackboard[n.p0];sel=v if 0<=v<n.p2 else n.p2;old=(st.a-1) if st and self._active(n) and st.a else -1
            if old>=0 and old!=sel:self._halt_node(kids[old])
            r=self._eval_node(kids[sel])
            if r==STATUS_RUNNING:
                if st is None:self._set_fault(3);return STATUS_FAILURE
                st.a=sel+1
            elif st:st.a=0
            return r
        if t==NODE_TYPES["Parallel"]:
            if st is None:
                success=failure=running=0
            else:success=st.a&0xFFFF;failure=(st.a>>16)&0xFFFF;running=st.b&RUNNING_MASK16
            skipped=0
            for k,c in enumerate(kids):
                bit=1<<k
                if success&bit or failure&bit:continue
                r=self._eval_node(c)
                if r==STATUS_SUCCESS:success|=bit;running&=~bit
                elif r==STATUS_FAILURE:failure|=bit;running&=~bit
                elif r==STATUS_RUNNING:
                    if st is None:self._set_fault(3);return STATUS_FAILURE
                    running|=bit
                else:skipped+=1;running&=~bit
                if st:st.a=success|(failure<<16);st.b=(st.b&ACTIVE_BIT)|running
                sc,fc=success.bit_count(),failure.bit_count();possible=len(kids)-fc-skipped
                if sc>=n.p0 or fc>=n.p1 or possible<n.p0:
                    for j,cc in enumerate(kids):
                        if running&(1<<j):self._halt_node(cc)
                    if st:st.a=st.b=0
                    return STATUS_SUCCESS if sc>=n.p0 else STATUS_FAILURE
            if running:return STATUS_RUNNING
            if st:st.a=st.b=0
            return STATUS_SKIPPED if success==0 and failure==0 and skipped==len(kids) else STATUS_FAILURE
        if t==NODE_TYPES["Invert"]:
            r=self._eval_node(kids[0]);return STATUS_FAILURE if r==STATUS_SUCCESS else STATUS_SUCCESS if r==STATUS_FAILURE else r
        if t in (NODE_TYPES["ForceSuccess"],NODE_TYPES["ForceFailure"]):
            r=self._eval_node(kids[0]);return r if r in (STATUS_RUNNING,STATUS_SKIPPED) else STATUS_SUCCESS if t==NODE_TYPES["ForceSuccess"] else STATUS_FAILURE
        if t==NODE_TYPES["TurnTimeout"]:
            if not self.p.analysis[i].may_run:return self._eval_node(kids[0])
            if st is None:self._set_fault(3);return STATUS_FAILURE
            now=self._feature(FEATURES["STEP"])
            if not self._active(n):st.a=now&0xFFFFFFFF
            elif now-int(st.a)>=n.p0:self._halt_node(kids[0]);st.a=0;return STATUS_FAILURE
            r=self._eval_node(kids[0])
            if r!=STATUS_RUNNING:st.a=0
            return r
        if t==NODE_TYPES["Retry"]:
            r=self._eval_node(kids[0])
            if r==STATUS_SUCCESS:
                if st:st.a=0
                return r
            if r==STATUS_FAILURE:
                if n.p0<=1:return STATUS_FAILURE
                if st is None:self._set_fault(3);return STATUS_FAILURE
                st.a+=1
                if st.a>=n.p0:st.a=0;return STATUS_FAILURE
                return STATUS_RUNNING
            return r
        if t==NODE_TYPES["Repeat"]:
            r=self._eval_node(kids[0])
            if r==STATUS_FAILURE:
                if st:st.a=0
                return r
            if r==STATUS_SUCCESS:
                if n.p0<=1:return STATUS_SUCCESS
                if st is None:self._set_fault(3);return STATUS_FAILURE
                st.a+=1
                if st.a>=n.p0:st.a=0;return STATUS_SUCCESS
                return STATUS_RUNNING
            return r
        if t==NODE_TYPES["KeepRunningUntilFailure"]:
            r=self._eval_node(kids[0]);return STATUS_SUCCESS if r==STATUS_FAILURE else STATUS_RUNNING if r==STATUS_SUCCESS else r
        if t==NODE_TYPES["Success"]:return STATUS_SUCCESS
        if t==NODE_TYPES["Failure"]:return STATUS_FAILURE
        if t==NODE_TYPES["Condition"]:return STATUS_SUCCESS if self._cmp(self._value(n.p0),self._value(n.p1),n.flags) else STATUS_FAILURE
        if t==NODE_TYPES["SetBlackboard"]:return STATUS_SUCCESS if self._write(n.p0,self._value(n.p1)) else STATUS_FAILURE
        if t==NODE_TYPES["AddBlackboard"]:
            v=self.state.blackboard[n.p0]+self._value(n.p1)
            if not -(1<<31)<=v<(1<<31):self._set_fault(2);return STATUS_FAILURE
            self.state.blackboard[n.p0]=v;return STATUS_SUCCESS
        if t==NODE_TYPES["EmitUnitIntent"]:
            unit,item,qty=self._value(n.p0),self._value(n.p2),self._value(n.p3)
            if not 0<=unit<self._feature(FEATURES["OWN_UNITS"]):return STATUS_FAILURE
            if n.p1==UNIT_OPS["PLANT"]:
                if not 0<=item<N_CROPS or qty!=1:return STATUS_FAILURE
            elif n.p1 in (UNIT_OPS["PICKUP"],UNIT_OPS["PLACE"]):
                if not 0<=item<len(ITEMS) or not 1<=qty<=32767:return STATUS_FAILURE
            elif item!=-1 or qty!=1:return STATUS_FAILURE
            self._emit(DeferredIntent(n.aux,i,self._emit_seq,qty,n.p1,max(item,0),unit));return STATUS_SUCCESS
        if t==NODE_TYPES["EmitMarketIntent"]:
            item,qty=self._value(n.p1),self._value(n.p2)
            if not 1<=qty<=65535:return STATUS_FAILURE
            if n.p0 in (MARKET_OPS["HIRE"],MARKET_OPS["BUY_LAND"]):
                if item!=-1 or qty!=1:return STATUS_FAILURE
            elif n.p0==MARKET_OPS["BUY_SEED"] and not 0<=item<N_CROPS:return STATUS_FAILURE
            elif n.p0==MARKET_OPS["BUY_PRODUCT"] and item not in (ITEM_ID["WHEAT"],ITEM_ID["FERTILIZER"]):return STATUS_FAILURE
            elif n.p0==MARKET_OPS["BUY_ANIMAL"] and not ITEM_ID["GOOSE"]<=item<=ITEM_ID["SHEEP"]:return STATUS_FAILURE
            elif n.p0==MARKET_OPS["SELL"] and not 0<=item<N_PRODUCTS:return STATUS_FAILURE
            self._emit(DeferredIntent(n.aux,i,self._emit_seq,qty,n.p0,max(item,0),0x80));return STATUS_SUCCESS
        if t==NODE_TYPES["WaitTurns"]:
            if st is None:self._set_fault(3);return STATUS_FAILURE
            now=self._feature(FEATURES["STEP"])
            if not self._active(n):st.a=now&0xFFFFFFFF;return STATUS_RUNNING
            if now-int(st.a)>=n.p0:st.a=0;return STATUS_SUCCESS
            return STATUS_RUNNING
        if t==NODE_TYPES["MoveUnitTo"]:
            unit,target=self._value(n.p0),self._value(n.p1)
            if unit<0 or target<0 or target>=100 or unit>=self._feature(FEATURES["OWN_UNITS"]):return STATUS_FAILURE
            x,y=self._feature(FEATURES["OWN_WORKER_X"],unit),self._feature(FEATURES["OWN_WORKER_Y"],unit);tx,ty=target%10,target//10
            if x==tx and y==ty:return STATUS_SUCCESS
            op=UNIT_OPS["EAST"] if x<tx else UNIT_OPS["WEST"] if x>tx else UNIT_OPS["SOUTH"] if y<ty else UNIT_OPS["NORTH"]
            self._emit(DeferredIntent(n.aux,i,self._emit_seq,1,op,0,unit));return STATUS_RUNNING
        self._set_fault(3);return STATUS_FAILURE
    def _emit(self,x:DeferredIntent)->None:
        self._emit_seq+=1
        if self._emit_seq>65535 or len(self._intents)>=self.p.max_live_intents:self._set_fault(4);return
        self._intents.append(x)
        if self._profile:
            self._profile.emitted_intents+=1;self._profile.max_live_intents=max(self._profile.max_live_intents,len(self._intents))


def decompile_policy(p:CompiledPolicy)->Dict[str,Any]:
    # BKG4 decompilation preserves executable identity. Named enums are reconstructed
    # with deterministic synthetic names because human enum labels are non-executable.
    enum_sizes=_enum_sizes_from_bb(p.bb);enum_names={tag:f"Enum{tag}" for tag in enum_sizes};enum_values={tag:tuple(f"V{i}" for i in range(size)) for tag,size in enum_sizes.items()}
    slot_names=[b.name or f"slot_{i}" for i,b in enumerate(p.bb)]
    def val(i:int)->Any:
        v=p.values[i]
        if v.kind==VALUE_KINDS["CONST"]:
            if v.arg:return {"const":enum_values[v.arg][v.value],"type":enum_names[v.arg]}
            if v.type==BB_TYPES["I32"]:return v.value
            if v.type==BB_TYPES["ITEM"] and v.value>=0:return {"const":ITEMS[v.value],"type":"ITEM"}
            return {"const":v.value,"type":BB_NAMES[v.type]}
        if v.kind==VALUE_KINDS["BB"]:return {"blackboard":slot_names[v.id]}
        out={"feature":FEATURE_NAMES[v.id]};args=[]
        for j in range(v.value):
            a=p.feature_args[v.arg+j];args.append({"blackboard":slot_names[a.id]} if a.kind==FEATURE_ARG_KINDS["BB"] else ({"const":ITEMS[a.value],"type":"ITEM"} if a.type==BB_TYPES["ITEM"] and a.value>=0 else a.value))
        if len(args)==1:out["arg"]=args[0]
        elif args:out["args"]=args
        return out
    unit_names={v:k for k,v in UNIT_OPS.items()};market_names={v:k for k,v in MARKET_OPS.items()}
    def life(idx:int)->Dict[str,Any]:
        lc=p.lifecycles[idx];out={}
        for name,q in (("failure_if",lc.failure_if),("success_if",lc.success_if),("skip_if",lc.skip_if),("while",lc.while_pred)):
            if q!=NONE_U16:
                z=p.predicates[q];out[name]={"lhs":val(z.lhs),"op":CMP_NAMES[z.cmp],"rhs":val(z.rhs)}
        rev={v:k for k,v in EFFECT_OP.items()}
        for name,rng in (("on_success",lc.on_success),("on_failure",lc.on_failure),("on_halted",lc.on_halted),("post",lc.post)):
            xs=[]
            for e in p.effects[rng[0]:rng[0]+rng[1]]:
                d={"op":rev[e.op],"slot":slot_names[e.slot]}
                if e.op!=EFFECT_OP["clear"]:d["value"]=val(e.value)
                xs.append(d)
            if xs:out[name]=xs
        return out
    def node(i:int)->Dict[str,Any]:
        n=p.nodes[i];typ=NODE_NAMES[n.type];kids=p.child_refs[n.first_child:n.first_child+n.child_count];out={"type":typ}
        if typ in ("Sequence","Selector","ReactiveSequence","ReactiveSelector","Parallel"):out["children"]=[node(c) for c in kids]
        if typ=="Parallel":out["success_threshold"],out["failure_threshold"]=n.p0,n.p1
        elif typ in ("If","ReactiveIf"):out["condition"],out["then"],out["else"]=node(kids[0]),node(kids[1]),node(kids[2])
        elif typ=="SwitchEnum":
            b=p.bb[n.p0];vals=enum_values[b.enum_tag];out["slot"]=slot_names[n.p0];out["cases"]={vals[k]:node(kids[k]) for k in range(len(vals))};out["default"]=node(kids[-1])
        elif n.type in DECORATOR_TYPES:
            out["child"]=node(kids[0])
            if typ=="TurnTimeout":out["turns"]=n.p0
            elif typ in ("Retry","Repeat"):out["n"]=n.p0
        elif typ=="Condition":out.update(lhs=val(n.p0),op=CMP_NAMES[n.flags],rhs=val(n.p1))
        elif typ in ("SetBlackboard","AddBlackboard"):out.update(slot=slot_names[n.p0],value=val(n.p1))
        elif typ=="EmitUnitIntent":
            out.update(unit=val(n.p0),action=unit_names[n.p1],priority=n.aux&0xFF,authority=AUTHORITY_NAMES[(n.aux>>8)&0xFF])
            if n.p1 in (UNIT_OPS["PLANT"],UNIT_OPS["PICKUP"],UNIT_OPS["PLACE"]):out["item"]=val(n.p2)
            if n.p1 in (UNIT_OPS["PICKUP"],UNIT_OPS["PLACE"]):out["quantity"]=val(n.p3)
        elif typ=="EmitMarketIntent":
            out.update(action=market_names[n.p0],priority=n.aux&0xFF,authority=AUTHORITY_NAMES[(n.aux>>8)&0xFF])
            if n.p0 not in (MARKET_OPS["HIRE"],MARKET_OPS["BUY_LAND"]):out.update(item=val(n.p1),quantity=val(n.p2))
        elif typ=="WaitTurns":out["turns"]=n.p0
        elif typ=="MoveUnitTo":out.update(unit=val(n.p0),target=val(n.p1),priority=n.aux&0xFF,authority=AUTHORITY_NAMES[(n.aux>>8)&0xFF])
        if n.lifecycle_slot!=NONE_U16:out["lifecycle"]=life(n.lifecycle_slot)
        return out
    result={"schema_version":4,"feature_registry_version":p.feature_registry_version,"tree":node(0),"instruction_budget":p.instruction_budget}
    if enum_sizes:result["enums"]=[{"name":enum_names[tag],"values":list(enum_values[tag])} for tag in sorted(enum_sizes)]
    if p.bb:
        result["blackboard"]=[]
        for i,b in enumerate(p.bb):
            type_name=enum_names[b.enum_tag] if b.enum_tag else BB_NAMES[b.type];initial=enum_values[b.enum_tag][b.initial] if b.enum_tag else b.initial
            result["blackboard"].append({"name":slot_names[i],"type":type_name,"initial":initial,"owner":"root"})
    return result


def bkg3_to_bkg4_source(source_or_compiled:Any)->Dict[str,Any]:
    if isinstance(source_or_compiled,_bkg3.CompiledPolicy):src=_bkg3.decompile_policy(source_or_compiled)
    else:src=copy.deepcopy(source_or_compiled)
    if src.get("schema_version")!=3:raise PolicyError("converter requires BKG3 source")
    src["schema_version"]=4;src["feature_registry_version"]=FEATURE_REGISTRY_VERSION;src.pop("semantic_hash",None)
    # Raw ENUM8 remains raw. All BKG3 priorities map to one authority class, preserving
    # relative arbitration exactly.
    def walk(x:Any)->None:
        if isinstance(x,dict):
            if x.get("type") in ("EmitUnitIntent","EmitMarketIntent","MoveUnitTo"):x.setdefault("authority","TACTICAL")
            for v in x.values():walk(v)
        elif isinstance(x,list):
            for v in x:walk(v)
    walk(src);return src


def compile_bkg3_to_v4(source_or_compiled:Any)->CompiledPolicy:return compile_policy(bkg3_to_bkg4_source(source_or_compiled))


def to_official_action(action:ActionResult)->Dict[str,Any]:
    unit_name={v:k for k,v in UNIT_OPS.items()};market_name={v:k for k,v in MARKET_OPS.items()}
    def unit(a):
        op,arg,n=a;name=unit_name.get(op,"PASS")
        if name in ("PICKUP","PLACE"):return [name,ITEMS[arg],n]
        if name=="PLANT":return [name,ITEMS[arg]]
        return [name]
    def order(a):
        op,item,n=a;name=market_name.get(op,"")
        return [name] if name in ("HIRE","BUY_LAND") else [name,ITEMS[item],n]
    return {"farmer":unit(action.units[0]) if action.units else ["PASS"],"hands":[unit(x) for x in action.units[1:]],"market":[order(x) for x in action.market]}


class KaggricultureAgent:
    def __init__(self,policy:CompiledPolicy|Dict[str,Any],max_market_orders:int=10):self.compiled=policy if isinstance(policy,CompiledPolicy) else compile_policy(policy);self.vm=VM(self.compiled);self.max_market_orders=max_market_orders;self.last_step=None
    def reset(self):self.vm.reset();self.last_step=None
    def __call__(self,observation:Dict[str,Any])->Dict[str,Any]:
        step=int(observation.get("step",int(observation.get("day",0) or 0)*24+int(observation.get("hour",0) or 0)) or 0)
        if self.last_step is not None and step<=self.last_step:self.vm.reset()
        self.last_step=step;return to_official_action(self.vm.tick(observation,self.max_market_orders).action)


def load_policy_json(path:str|Path)->Dict[str,Any]:return json.loads(Path(path).read_text(encoding="utf-8"))
def compile_file(src:str|Path,dst:str|Path,manifest_path:str|Path|None=None)->CompiledPolicy:
    p=compile_policy(load_policy_json(src));save_compiled(p,dst,manifest_path);return p

def policy_hash(src:Dict[str,Any])->str:return compile_policy(src).semantic_hash


if __name__=="__main__":
    import argparse
    ap=argparse.ArgumentParser(description="Compile/inspect Bonsai-KG BKG4")
    ap.add_argument("policy");ap.add_argument("--out");ap.add_argument("--manifest");ap.add_argument("--decompile",action="store_true")
    args=ap.parse_args();path=Path(args.policy)
    if path.suffix==".bkgc":cp=load_compiled(path)
    else:
        cp=compile_policy(load_policy_json(path))
        if args.out:save_compiled(cp,args.out,args.manifest)
    if args.decompile:print(json.dumps(decompile_policy(cp),sort_keys=True,indent=2))
    print(json.dumps({"hash":cp.semantic_hash,"nodes":len(cp.nodes),"state_slots":cp.state_count,"bb_slots":len(cp.bb),"compiled_bytes":cp.compiled_bytes,"instruction_budget":cp.instruction_budget,"max_live_intents":cp.max_live_intents,"max_depth":cp.max_depth,"max_visits":cp.analysis[0].max_visits if cp.analysis else None,"warnings":cp.warnings},sort_keys=True))
