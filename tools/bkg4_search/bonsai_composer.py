from __future__ import annotations

import argparse
import copy
import dataclasses
import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from tools.bkg4_search.bounded_macro import (
    BoundedRouteMacro,
    ExactRoutePreemption,
    validate_identity_free,
)
from tools.bkg4_search.module_codec import ConditionGene


ROOT = Path(__file__).resolve().parents[2]
COMPOSER_KIND = "bounded_bonsai_composition_v1"
SPECIALIST_KINDS = {"embedded_authority", "bounded_route_macro"}
END_POLICIES = {"RETURN_TO_PARENT"}
RESUME_POLICIES = {"BOTH_WINDOWS_END_TOGETHER"}
GATE_OVERLAP_STATES = {
    "PROVEN_DISJOINT",
    "POSSIBLE_OVERLAP",
    "OBSERVED_OVERLAP",
    "UNKNOWN",
}


def _canonical_hash(prefix: bytes, value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(prefix + raw).hexdigest()


def _condition_from_json(value: Mapping[str, Any]) -> ConditionGene:
    return ConditionGene(
        feature=str(value["feature"]),
        cmp=str(value["cmp"]),
        index=int(value.get("index", 0)),
        value=int(value["value"]),
    )


def bounded_macro_from_json(value: Mapping[str, Any]) -> BoundedRouteMacro:
    if value.get("kind") != "bounded_route_macro_v1" or int(value.get("schema", 0)) != 1:
        raise ValueError("unsupported bounded macro declaration")
    macro = BoundedRouteMacro(
        name=str(value["name"]),
        route_id=int(value["route_id"]),
        base_route_id=int(value["base_route_id"]),
        start_step=int(value["start_step"]),
        duration=int(value["duration"]),
        gate_clauses=tuple(
            tuple(_condition_from_json(condition) for condition in clause)
            for clause in value["gate_clauses"]
        ),
        provenance=dict(value.get("provenance") or {}),
    )
    macro.validate()
    return macro


@dataclass(frozen=True)
class StrategicAuthority:
    name: str
    blackboard_slot: str
    blackboard_type: str = "I32"

    def validate(self) -> None:
        if not self.name or not self.blackboard_slot:
            raise ValueError("strategic authority requires a name and blackboard slot")
        if self.blackboard_type not in {"I32", "U16"}:
            raise ValueError("unsupported strategic authority blackboard type")

    def canonical(self) -> dict[str, Any]:
        self.validate()
        return dataclasses.asdict(self)


@dataclass(frozen=True)
class SpecialistDeclaration:
    name: str
    kind: str
    route_id: int
    authorization_step: int
    active_start: int
    active_end: int
    active_slot: str
    authorities: tuple[str, ...]
    route_source: str
    semantic_id: str
    gate_clauses: tuple[tuple[ConditionGene, ...], ...] | None
    end_policy: str = "RETURN_TO_PARENT"
    macro: BoundedRouteMacro | None = None
    macro_source: str | None = None
    provenance: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_macro(
        cls,
        macro: BoundedRouteMacro,
        *,
        route_source: str,
        authorities: Sequence[str],
        macro_source: str | None = None,
        provenance: Mapping[str, Any] | None = None,
    ) -> "SpecialistDeclaration":
        macro.validate()
        return cls(
            name=macro.name,
            kind="bounded_route_macro",
            route_id=macro.route_id,
            authorization_step=macro.start_step,
            active_start=macro.start_step,
            active_end=macro.end_step,
            active_slot=macro.active_slot,
            authorities=tuple(authorities),
            route_source=route_source,
            semantic_id=macro.semantic_id,
            gate_clauses=macro.gate_clauses,
            end_policy="RETURN_TO_PARENT",
            macro=macro,
            macro_source=macro_source,
            provenance=dict(provenance or {}),
        )

    def validate(self) -> None:
        if not self.name or not self.name.replace("_", "").isalnum():
            raise ValueError("specialist name must be alphanumeric/underscore")
        if self.kind not in SPECIALIST_KINDS:
            raise ValueError(f"unsupported specialist kind {self.kind!r}")
        if not 0 <= int(self.route_id) <= 65535:
            raise ValueError("specialist route id out of range")
        if not 0 <= int(self.authorization_step) < 719:
            raise ValueError("specialist authorization step out of range")
        if not 0 <= int(self.active_start) <= int(self.active_end) < 719:
            raise ValueError("specialist active window out of range")
        if int(self.authorization_step) != int(self.active_start):
            raise ValueError("bounded composer currently requires authorization at active-window start")
        if not self.active_slot or not self.route_source or not self.semantic_id:
            raise ValueError("specialist requires active slot, route source and semantic id")
        if not self.authorities or len(set(self.authorities)) != len(self.authorities):
            raise ValueError("specialist authorities must be non-empty and unique")
        if self.end_policy not in END_POLICIES:
            raise ValueError(f"unsupported end policy {self.end_policy!r}")
        if self.gate_clauses is not None:
            if not self.gate_clauses:
                raise ValueError("declared gate must contain at least one clause")
            for clause in self.gate_clauses:
                if not clause:
                    raise ValueError("declared gate clause cannot be empty")
                for condition in clause:
                    condition.validate()
        if self.kind == "bounded_route_macro":
            if self.macro is None:
                raise ValueError("bounded route specialist requires a loaded macro")
            self.macro.validate()
            expected = (
                self.macro.name,
                self.macro.route_id,
                self.macro.start_step,
                self.macro.end_step,
                self.macro.active_slot,
                self.macro.semantic_id,
            )
            actual = (
                self.name,
                self.route_id,
                self.active_start,
                self.active_end,
                self.active_slot,
                self.semantic_id,
            )
            if actual != expected or self.gate_clauses != self.macro.gate_clauses:
                raise ValueError("specialist declaration does not match bounded macro")
        elif self.macro is not None:
            raise ValueError("embedded authority cannot contain a bounded macro")
        validate_identity_free(self.provenance, f"specialists.{self.name}.provenance")

    def canonical(self) -> dict[str, Any]:
        self.validate()
        out: dict[str, Any] = {
            "name": self.name,
            "kind": self.kind,
            "route_id": int(self.route_id),
            "semantic_id": self.semantic_id,
            "authorization_step": int(self.authorization_step),
            "active_window": [int(self.active_start), int(self.active_end)],
            "active_slot": self.active_slot,
            "authorities": sorted(self.authorities),
            "route_source": self.route_source,
            "end_policy": self.end_policy,
            "gate_clauses": None if self.gate_clauses is None else [
                [dataclasses.asdict(condition) for condition in clause]
                for clause in self.gate_clauses
            ],
            "provenance": dict(self.provenance),
        }
        if self.macro_source is not None:
            out["macro_source"] = self.macro_source
        return out


@dataclass(frozen=True)
class PreemptionDeclaration:
    preemptor: str
    preempted: str
    authority: str
    overlap_start: int
    overlap_end: int
    resume_policy: str
    evidence: Mapping[str, Any]

    def validate(self) -> None:
        if not self.preemptor or not self.preempted or not self.authority:
            raise ValueError("preemption requires source, target and authority")
        if self.preemptor == self.preempted:
            raise ValueError("self-preemption is forbidden")
        if not 0 <= int(self.overlap_start) <= int(self.overlap_end) < 719:
            raise ValueError("preemption overlap window out of range")
        if self.resume_policy not in RESUME_POLICIES:
            raise ValueError(f"unsupported resume policy {self.resume_policy!r}")
        if not isinstance(self.evidence, Mapping) or not self.evidence.get("kind"):
            raise ValueError("preemption requires an evidence kind")
        validate_identity_free(self.evidence, "preemption.evidence")

    def canonical(self) -> dict[str, Any]:
        self.validate()
        return {
            "preemptor": self.preemptor,
            "preempted": self.preempted,
            "authority": self.authority,
            "overlap_window": [int(self.overlap_start), int(self.overlap_end)],
            "resume_policy": self.resume_policy,
            "evidence": dict(self.evidence),
        }


@dataclass(frozen=True)
class GateOverlapObservation:
    left: str
    right: str
    evidence: Mapping[str, Any]

    def validate(self) -> None:
        if not self.left or not self.right or self.left == self.right:
            raise ValueError("observed gate overlap requires two distinct specialists")
        if not self.evidence.get("kind"):
            raise ValueError("observed gate overlap requires evidence")
        validate_identity_free(self.evidence, "gate_overlap.evidence")


@dataclass(frozen=True)
class CompositionResult:
    module: dict[str, Any]
    report: dict[str, Any]
    semantic_id: str
    elapsed_ms: float


def _constraint_key(condition: ConditionGene) -> tuple[str, int]:
    return condition.feature, int(condition.index)


def _conditions_satisfiable(conditions: Iterable[ConditionGene]) -> bool:
    bounds: dict[tuple[str, int], list[Any]] = {}
    for condition in conditions:
        key = _constraint_key(condition)
        low, high, excluded = bounds.setdefault(key, [-(1 << 63), (1 << 63) - 1, set()])
        value = int(condition.value)
        if condition.cmp == "EQ":
            low = max(low, value)
            high = min(high, value)
        elif condition.cmp == "NE":
            excluded.add(value)
        elif condition.cmp == "LE":
            high = min(high, value)
        elif condition.cmp == "GE":
            low = max(low, value)
        bounds[key] = [low, high, excluded]
    for low, high, excluded in bounds.values():
        if low > high:
            return False
        if low == high and low in excluded:
            return False
    return True


def analyze_gate_overlap(left: SpecialistDeclaration, right: SpecialistDeclaration) -> str:
    left.validate()
    right.validate()
    if left.gate_clauses is None or right.gate_clauses is None:
        return "UNKNOWN"
    left_step = ConditionGene("STEP", "EQ", 0, left.authorization_step)
    right_step = ConditionGene("STEP", "EQ", 0, right.authorization_step)
    for left_clause in left.gate_clauses:
        for right_clause in right.gate_clauses:
            if _conditions_satisfiable((*left_clause, left_step, *right_clause, right_step)):
                return "POSSIBLE_OVERLAP"
    return "PROVEN_DISJOINT"


def _windows_overlap(left: SpecialistDeclaration, right: SpecialistDeclaration) -> bool:
    return max(left.active_start, right.active_start) <= min(left.active_end, right.active_end)


def _validate_precedence_graph(names: set[str], rules: Sequence[PreemptionDeclaration]) -> None:
    graph = {name: set() for name in names}
    indegree = {name: 0 for name in names}
    for rule in rules:
        rule.validate()
        if rule.preemptor not in names or rule.preempted not in names:
            raise ValueError("preemption target or source does not exist")
        if rule.preempted not in graph[rule.preemptor]:
            graph[rule.preemptor].add(rule.preempted)
            indegree[rule.preempted] += 1
    ready = sorted(name for name, degree in indegree.items() if degree == 0)
    visited = 0
    while ready:
        name = ready.pop(0)
        visited += 1
        for target in sorted(graph[name]):
            indegree[target] -= 1
            if indegree[target] == 0:
                ready.append(target)
                ready.sort()
    if visited != len(names):
        raise ValueError("precedence cycle detected")


def _ordered_generated(
    specialists: Sequence[SpecialistDeclaration],
    rules: Sequence[PreemptionDeclaration],
) -> list[SpecialistDeclaration]:
    generated = {item.name: item for item in specialists if item.kind == "bounded_route_macro"}
    dependencies = {name: set() for name in generated}
    for rule in rules:
        if rule.preemptor in generated and rule.preempted in generated:
            dependencies[rule.preemptor].add(rule.preempted)
    result: list[SpecialistDeclaration] = []
    pending = dict(dependencies)
    while pending:
        ready = [generated[name] for name, deps in pending.items() if not deps]
        if not ready:
            raise ValueError("precedence cycle detected")
        ready.sort(key=lambda item: (int(item.route_id), item.name))
        for item in ready:
            result.append(item)
            pending.pop(item.name)
            for deps in pending.values():
                deps.discard(item.name)
    return result


def compose_specialists(
    parent: Mapping[str, Any],
    specialists: Sequence[SpecialistDeclaration],
    preemptions: Sequence[PreemptionDeclaration],
    *,
    strategic_authorities: Sequence[StrategicAuthority],
    output_name: str,
    observed_gate_overlaps: Sequence[GateOverlapObservation] = (),
    declaration_provenance: Mapping[str, Any] | None = None,
) -> CompositionResult:
    started = time.perf_counter()
    items = tuple(specialists)
    rules = tuple(preemptions)
    authorities = tuple(strategic_authorities)
    if not output_name or not output_name.replace("_", "").isalnum():
        raise ValueError("composition output name must be alphanumeric/underscore")
    validate_identity_free(declaration_provenance or {}, "composition.provenance")
    if len({item.name for item in items}) != len(items):
        raise ValueError("duplicate specialist name")
    if len({int(item.route_id) for item in items}) != len(items):
        raise ValueError("duplicate specialist route id conflict")
    if len({item.semantic_id for item in items}) != len(items):
        raise ValueError("duplicate specialist semantic id conflict")
    for item in items:
        item.validate()
    if len({authority.name for authority in authorities}) != len(authorities):
        raise ValueError("duplicate strategic authority name")
    if len({authority.blackboard_slot for authority in authorities}) != len(authorities):
        raise ValueError("duplicate strategic authority blackboard slot")
    for authority in authorities:
        authority.validate()
    authority_by_name = {authority.name: authority for authority in authorities}
    for item in items:
        missing = set(item.authorities) - set(authority_by_name)
        if missing:
            raise ValueError(f"specialist {item.name} owns unknown authority {sorted(missing)}")

    out = copy.deepcopy(dict(parent))
    if out.get("tree", {}).get("type") != "Sequence":
        raise ValueError("composer parent module must have a Sequence tree")
    parent_bb = list(out.get("blackboard") or [])
    bb_by_name: dict[str, Mapping[str, Any]] = {}
    for row in parent_bb:
        name = str(row.get("name") or "")
        if not name or name in bb_by_name:
            raise ValueError("duplicate BB owner or name in parent")
        if not row.get("owner") or not row.get("type"):
            raise ValueError("parent BB declaration requires owner and type")
        bb_by_name[name] = row
    for authority in authorities:
        row = bb_by_name.get(authority.blackboard_slot)
        if row is None or str(row.get("type")) != authority.blackboard_type:
            raise ValueError(f"strategic authority slot {authority.blackboard_slot} is missing or has wrong type")

    parent_route_rows = list(out.get("strategic_routes") or [])
    parent_routes = {int(row["route_id"]): row for row in parent_route_rows}
    if len(parent_routes) != len(parent_route_rows):
        raise ValueError("duplicate strategic route id in parent")
    for item in items:
        if item.kind == "embedded_authority":
            route = parent_routes.get(int(item.route_id))
            if route is None or str(route.get("name")) != item.name:
                raise ValueError(f"embedded specialist {item.name} is not present in parent routes")
            slot = bb_by_name.get(item.active_slot)
            if slot is None or str(slot.get("owner")) != item.name or str(slot.get("type")) != "BOOL":
                raise ValueError(f"embedded specialist {item.name} has wrong BB owner")
        else:
            if int(item.route_id) in parent_routes:
                raise ValueError(f"generated specialist route id collides with parent: {item.route_id}")
            if item.active_slot in bb_by_name:
                raise ValueError(f"generated specialist BB already exists: {item.active_slot}")
            bb_by_name[item.active_slot] = item.macro.blackboard_decl()  # type: ignore[union-attr]

    _validate_precedence_graph(set(item.name for item in items), rules)
    by_name = {item.name: item for item in items}
    rule_by_pair: dict[frozenset[str], PreemptionDeclaration] = {}
    if len({rule.preemptor for rule in rules}) != len(rules):
        raise ValueError("multiple suppression targets for one preemptor are unsupported")
    for rule in rules:
        pair = frozenset((rule.preemptor, rule.preempted))
        if pair in rule_by_pair:
            raise ValueError("conflicting precedence declarations for specialist pair")
        source = by_name[rule.preemptor]
        target = by_name[rule.preempted]
        if source.kind != "bounded_route_macro":
            raise ValueError("preemptor must be a generated bounded route macro")
        if rule.authority not in source.authorities or rule.authority not in target.authorities:
            raise ValueError("declared preemption pair does not share the named authority")
        overlap = (max(source.active_start, target.active_start), min(source.active_end, target.active_end))
        if overlap[0] > overlap[1]:
            raise ValueError("declared preemption pair does not have a window conflict")
        if (rule.overlap_start, rule.overlap_end) != overlap:
            raise ValueError("declared preemption window does not match specialist windows")
        if rule.resume_policy == "BOTH_WINDOWS_END_TOGETHER":
            if (source.active_start, source.active_end) != (target.active_start, target.active_end):
                raise ValueError("BOTH_WINDOWS_END_TOGETHER requires identical active windows")
        target_slot = bb_by_name.get(target.active_slot)
        if target_slot is None or str(target_slot.get("owner")) != target.name:
            raise ValueError("suppression target has wrong BB owner")
        rule_by_pair[pair] = rule

    observed_pairs: dict[frozenset[str], GateOverlapObservation] = {}
    for observation in observed_gate_overlaps:
        observation.validate()
        if observation.left not in by_name or observation.right not in by_name:
            raise ValueError("observed gate-overlap specialist does not exist")
        pair = frozenset((observation.left, observation.right))
        if pair in observed_pairs:
            raise ValueError("duplicate observed gate-overlap evidence")
        observed_pairs[pair] = observation

    conflict_rows: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []
    sorted_items = sorted(items, key=lambda item: item.name)
    for index, left in enumerate(sorted_items):
        for right in sorted_items[index + 1:]:
            pair = frozenset((left.name, right.name))
            window_overlap = _windows_overlap(left, right)
            shared = sorted(set(left.authorities) & set(right.authorities))
            gate_overlap = analyze_gate_overlap(left, right)
            if pair in observed_pairs:
                if gate_overlap == "PROVEN_DISJOINT":
                    raise ValueError("observed gate overlap contradicts proven-disjoint gates")
                gate_overlap = "OBSERVED_OVERLAP"
            if gate_overlap not in GATE_OVERLAP_STATES:
                raise AssertionError("invalid gate-overlap state")
            kinds: list[str] = []
            if window_overlap:
                kinds.append("WINDOW_OVERLAP")
            if shared:
                kinds.append("AUTHORITY_CONFLICT")
            if gate_overlap in {"POSSIBLE_OVERLAP", "OBSERVED_OVERLAP"}:
                kinds.append("GATE_OVERLAP")
            resolution = rule_by_pair.get(pair)
            if resolution is not None:
                kinds.append("DECLARED_PREEMPTION")
            row = {
                "specialists": [left.name, right.name],
                "window_overlap": window_overlap,
                "shared_authorities": shared,
                "gate_overlap": gate_overlap,
                "conflict_classes": kinds,
                "resolution": None if resolution is None else resolution.canonical(),
            }
            conflict_rows.append(row)
            if window_overlap and shared and resolution is None:
                unresolved.append(row)
    if unresolved:
        pairs = [row["specialists"] for row in unresolved]
        raise ValueError(f"unresolved strategic authority conflict: {pairs}")

    generated_order = _ordered_generated(items, rules)
    preemption_by_source = {rule.preemptor: rule for rule in rules}
    for item in generated_order:
        macro = item.macro
        if macro is None:
            raise AssertionError("generated specialist lost its macro")
        out.setdefault("blackboard", []).append(macro.blackboard_decl())
        rule = preemption_by_source.get(item.name)
        authority_slot = authority_by_name[item.authorities[0]].blackboard_slot
        if len(item.authorities) != 1:
            raise ValueError("bounded composer currently supports one strategic authority per macro")
        if rule is None:
            tree = macro.specialist_tree(route_override_slot=authority_slot)
        else:
            target = by_name[rule.preempted]
            exact = ExactRoutePreemption(
                preemptor_name=item.name,
                preemptor_route_id=item.route_id,
                preempted_name=target.name,
                preempted_route_id=target.route_id,
                preempted_active_slot=target.active_slot,
                start_step=item.active_start,
                end_step=item.active_end,
                evidence_kind=str(rule.evidence["kind"]),
            )
            tree = exact.specialist_tree(macro, route_override_slot=authority_slot)
        out["tree"]["children"].append(tree)
        out.setdefault("strategic_routes", []).append({
            "route_id": int(item.route_id),
            "name": item.name,
            "source": item.route_source,
        })

    parent_identity = {
        "name": str(parent.get("name") or ""),
        "canonical_sha256": _canonical_hash(b"BKG4-COMPOSER-PARENT-V1\0", parent),
    }
    canonical_input = {
        "schema": 1,
        "kind": COMPOSER_KIND,
        "parent": parent_identity,
        "output_name": output_name,
        "strategic_authorities": [authority.canonical() for authority in sorted(authorities, key=lambda row: row.name)],
        "specialists": [item.canonical() for item in sorted(items, key=lambda row: row.name)],
        "preemptions": [rule.canonical() for rule in sorted(rules, key=lambda row: (row.preemptor, row.preempted))],
        "observed_gate_overlaps": [
            {"specialists": sorted((row.left, row.right)), "evidence": dict(row.evidence)}
            for row in sorted(observed_gate_overlaps, key=lambda row: tuple(sorted((row.left, row.right))))
        ],
        "provenance": dict(declaration_provenance or {}),
    }
    semantic_id = _canonical_hash(b"BKG4-BOUNDED-COMPOSER-V1\0", canonical_input)
    out["name"] = output_name
    out["manifest"] = {
        "module": output_name,
        "parent": parent_identity,
        "category": "automatic_composition/bounded_v1",
        "composer_schema": COMPOSER_KIND,
        "composition_semantic_id": semantic_id,
        "composition": [item.name for item in generated_order],
        "embedded_authorities": [item.name for item in sorted(items, key=lambda row: row.name) if item.kind == "embedded_authority"],
        "strategic_authorities": [authority.canonical() for authority in sorted(authorities, key=lambda row: row.name)],
        "precedence": [rule.canonical() for rule in sorted(rules, key=lambda row: (row.preemptor, row.preempted))],
        "identity_rule": "Composition declarations and runtime gates contain legal causal state only; identity-like provenance is rejected.",
    }
    report = {
        **canonical_input,
        "composition_semantic_id": semantic_id,
        "composition_order": [item.name for item in generated_order],
        "conflicts": conflict_rows,
        "unresolved_conflicts": [],
        "owned_blackboard": {
            item.name: item.active_slot for item in sorted(items, key=lambda row: row.name)
        },
    }
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return CompositionResult(module=out, report=report, semantic_id=semantic_id, elapsed_ms=elapsed_ms)


def _resolve_repo_path(value: str, repo_root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def _load_route_document(path: Path, specialist: SpecialistDeclaration) -> None:
    value = json.loads(path.read_text(encoding="utf-8"))
    actual = (
        str(value.get("name")),
        int(value.get("route_id", -1)),
        tuple(int(x) for x in value.get("active_window", ())),
    )
    expected = (specialist.name, int(specialist.route_id), (specialist.active_start, specialist.active_end))
    if actual != expected:
        raise ValueError(f"route document does not match specialist {specialist.name}")
    if specialist.kind == "bounded_route_macro" and str(value.get("macro_id")) != specialist.semantic_id:
        raise ValueError(f"route document macro id does not match specialist {specialist.name}")


def compose_declaration(value: Mapping[str, Any], *, repo_root: Path = ROOT) -> CompositionResult:
    if int(value.get("schema", 0)) != 1 or value.get("kind") != COMPOSER_KIND:
        raise ValueError("unsupported composition declaration")
    parent_path = _resolve_repo_path(str(value["parent_module"]), repo_root)
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    authorities = tuple(
        StrategicAuthority(
            name=str(row["name"]),
            blackboard_slot=str(row["blackboard_slot"]),
            blackboard_type=str(row.get("blackboard_type", "I32")),
        )
        for row in value.get("strategic_authorities", ())
    )
    specialists: list[SpecialistDeclaration] = []
    for row in value.get("specialists", ()):
        kind = str(row["kind"])
        macro: BoundedRouteMacro | None = None
        macro_source: str | None = None
        if kind == "bounded_route_macro":
            macro_source = str(row["macro_source"])
            macro_value = json.loads(_resolve_repo_path(macro_source, repo_root).read_text(encoding="utf-8"))
            macro = bounded_macro_from_json(macro_value)
            gate_clauses = macro.gate_clauses
        else:
            raw_gate = row.get("gate_clauses")
            gate_clauses = None if raw_gate is None else tuple(
                tuple(_condition_from_json(condition) for condition in clause)
                for clause in raw_gate
            )
        active_start, active_end = (int(x) for x in row["active_window"])
        specialist = SpecialistDeclaration(
            name=str(row["name"]),
            kind=kind,
            route_id=int(row["route_id"]),
            authorization_step=int(row["authorization_step"]),
            active_start=active_start,
            active_end=active_end,
            active_slot=str(row["active_slot"]),
            authorities=tuple(str(x) for x in row["authorities"]),
            route_source=str(row["route_source"]),
            semantic_id=str(row["semantic_id"]),
            gate_clauses=gate_clauses,
            end_policy=str(row.get("end_policy", "RETURN_TO_PARENT")),
            macro=macro,
            macro_source=macro_source,
            provenance=dict(row.get("provenance") or {}),
        )
        specialist.validate()
        _load_route_document(_resolve_repo_path(specialist.route_source, repo_root), specialist)
        specialists.append(specialist)
    preemptions = tuple(
        PreemptionDeclaration(
            preemptor=str(row["preemptor"]),
            preempted=str(row["preempted"]),
            authority=str(row["authority"]),
            overlap_start=int(row["overlap_window"][0]),
            overlap_end=int(row["overlap_window"][1]),
            resume_policy=str(row["resume_policy"]),
            evidence=dict(row.get("evidence") or {}),
        )
        for row in value.get("preemptions", ())
    )
    observations = tuple(
        GateOverlapObservation(
            left=str(row["specialists"][0]),
            right=str(row["specialists"][1]),
            evidence=dict(row.get("evidence") or {}),
        )
        for row in value.get("observed_gate_overlaps", ())
    )
    return compose_specialists(
        parent,
        specialists,
        preemptions,
        strategic_authorities=authorities,
        output_name=str(value["output_name"]),
        observed_gate_overlaps=observations,
        declaration_provenance=dict(value.get("provenance") or {}),
    )


def load_and_compose(path: str | Path, *, repo_root: Path = ROOT) -> CompositionResult:
    declaration = json.loads(Path(path).read_text(encoding="utf-8"))
    return compose_declaration(declaration, repo_root=repo_root)


def main() -> int:
    parser = argparse.ArgumentParser(description="Deterministic fail-closed bounded Bonsai specialist composer")
    parser.add_argument("--declaration", type=Path, required=True)
    parser.add_argument("--out-module", type=Path)
    parser.add_argument("--out-manifest", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    result = load_and_compose(args.declaration.resolve())
    if not args.validate_only:
        if args.out_module is None or args.out_manifest is None:
            parser.error("--out-module and --out-manifest are required unless --validate-only is used")
        args.out_module.parent.mkdir(parents=True, exist_ok=True)
        args.out_manifest.parent.mkdir(parents=True, exist_ok=True)
        args.out_module.write_text(json.dumps(result.module, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        args.out_manifest.write_text(json.dumps(result.report, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "composition_semantic_id": result.semantic_id,
        "specialists": len(result.report["specialists"]),
        "conflicts": len(result.report["conflicts"]),
        "unresolved_conflicts": len(result.report["unresolved_conflicts"]),
        "composition_order": result.report["composition_order"],
        "elapsed_ms": round(result.elapsed_ms, 3),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
