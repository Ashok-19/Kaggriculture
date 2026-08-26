"""Deterministic RLRMB Option-Program materialization for the BKG4 composer.

The generated module uses the existing bounded-route authority envelope and owns
an explicit, collision-checked state layout.  ``MaterializedInterpreter`` is the
auditable action-side executor for that layout; game physics remain in the exact
C++ engine, just as for the Phase-4 reference path.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Mapping

from tools.bkg4_search.bonsai_composer import SpecialistDeclaration
from tools.bkg4_search.bounded_macro import BoundedRouteMacro, validate_identity_free
from tools.bkg4_search.module_codec import ConditionGene

from .canonical import canonical_action
from .evaluator import ReferenceInterpreter, _condition_matches
from .ir import MacroProgram
from .motifs import Motif


MATERIALIZATION_SCHEMA_VERSION = 1


def _hash(document: Mapping[str, Any]) -> str:
    raw = json.dumps(document, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(b"RLRMB-BKG4-MATERIALIZATION-V1\0" + raw).hexdigest()


def _condition_document(value: ConditionGene) -> dict[str, Any]:
    value.validate()
    return dataclasses.asdict(value)


@dataclass(frozen=True)
class MaterializationInput:
    candidate_id: str
    program: MacroProgram
    outer_gate: tuple[tuple[ConditionGene, ...], ...]
    motifs: Mapping[str, Motif]
    route_id: int
    base_route_id: int
    route_source: str
    authority: str
    authority_slot: str
    reserved_blackboard: tuple[str, ...] = ()
    audit_metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = MATERIALIZATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != MATERIALIZATION_SCHEMA_VERSION:
            raise ValueError("unsupported materialization input version")
        if not isinstance(self.program, MacroProgram):
            raise TypeError("program must be a MacroProgram")
        if len(self.candidate_id) != 64 or any(c not in "0123456789abcdef" for c in self.candidate_id):
            raise ValueError("candidate_id must be a lowercase SHA-256 identity")
        if not self.outer_gate or any(not clause for clause in self.outer_gate):
            raise ValueError("outer gate requires non-empty DNF clauses")
        for clause in self.outer_gate:
            for condition in clause:
                condition.validate()
        needed = {
            phase.motif_id for phase in (*self.program.phases, *((self.program.recovery_phase,) if self.program.recovery_phase else ()))
            if phase.motif_id is not None
        }
        if not needed.issubset(self.motifs):
            raise ValueError(f"materialization is missing motifs: {sorted(needed - set(self.motifs))}")
        if not self.route_source or not self.authority or not self.authority_slot:
            raise ValueError("route source and authority fields are required")
        validate_identity_free(self.audit_metadata, "materialization.audit_metadata")
        prefix = f"rlrmb_{self.candidate_id[:10]}_"
        collisions = set(self.reserved_blackboard) & {
            prefix + suffix for suffix in ("active", "phase", "phase_steps", "program_steps", "branch_0", "branch_1", "abort", "recovery")
        }
        if collisions:
            raise ValueError(f"materialization blackboard collision: {sorted(collisions)}")


@dataclass(frozen=True)
class MaterializationArtifact:
    candidate_id: str
    program: MacroProgram
    outer_gate: tuple[tuple[ConditionGene, ...], ...]
    module: Mapping[str, Any]
    specialist: SpecialistDeclaration
    blackboard: tuple[Mapping[str, Any], ...]
    state_mapping: Mapping[str, Any]
    node_count: int
    materialization_id: str
    schema_version: int = MATERIALIZATION_SCHEMA_VERSION

    def canonical_document(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "candidate_id": self.candidate_id,
            "program": self.program.to_document(),
            "outer_gate": [[_condition_document(c) for c in clause] for clause in self.outer_gate],
            "module": self.module,
            "state_mapping": self.state_mapping,
            "node_count": self.node_count,
        }

    def canonical_json(self) -> str:
        return json.dumps(self.canonical_document(), sort_keys=True, separators=(",", ":"))


def _count_nodes(node: Mapping[str, Any]) -> int:
    count = 1
    for key in ("children",):
        count += sum(_count_nodes(child) for child in node.get(key, ()))
    for key in ("child", "then", "else"):
        child = node.get(key)
        if isinstance(child, Mapping):
            count += _count_nodes(child)
    return count


def materialize_program(source: MaterializationInput) -> MaterializationArtifact:
    name = f"rlrmb_{source.candidate_id[:10]}"
    macro = BoundedRouteMacro(
        name=name,
        route_id=source.route_id,
        base_route_id=source.base_route_id,
        start_step=source.program.initiation.step,
        duration=source.program.max_total_steps,
        gate_clauses=source.outer_gate,
        provenance={"program_semantic_id": source.program.semantic_id},
    )
    macro.validate()
    specialist = SpecialistDeclaration.from_macro(
        macro, route_source=source.route_source, authorities=(source.authority,),
        provenance={"candidate_semantic_id": source.candidate_id, "program_semantic_id": source.program.semantic_id},
    )
    prefix = f"{name}_"
    state_rows = [
        {"name": prefix + "phase", "type": "I32", "initial": 0, "owner": name},
        {"name": prefix + "phase_steps", "type": "I32", "initial": 0, "owner": name},
        {"name": prefix + "program_steps", "type": "I32", "initial": 0, "owner": name},
    ]
    for index, _ in enumerate(source.program.decisions):
        state_rows.append({"name": prefix + f"branch_{index}", "type": "I32", "initial": -1, "owner": name})
    if any(c.kind.value == "ABORT_TO_PARENT" for d in source.program.decisions for c in (d.when_true, d.when_false)):
        state_rows.append({"name": prefix + "abort", "type": "BOOL", "initial": 0, "owner": name})
    if source.program.recovery_phase is not None:
        state_rows.append({"name": prefix + "recovery", "type": "BOOL", "initial": 0, "owner": name})
    blackboard = (macro.blackboard_decl(), *state_rows)
    if len({row["name"] for row in blackboard}) != len(blackboard):
        raise ValueError("materialization generated colliding blackboard state")
    tree = macro.specialist_tree(route_override_slot=source.authority_slot)
    module = {
        "schema": 1,
        "name": name,
        "blackboard": list(blackboard),
        "tree": tree,
        "strategic_routes": [{"name": name, "route_id": source.route_id, "source": source.route_source}],
    }
    mapping = {
        "active": macro.active_slot,
        "phase": prefix + "phase",
        "phase_steps": prefix + "phase_steps",
        "program_steps": prefix + "program_steps",
        "phases": [phase.to_document() for phase in source.program.phases],
        "terminations": [phase.termination.to_document() for phase in source.program.phases],
        "decisions": [decision.to_document() for decision in source.program.decisions],
        "recovery": None if source.program.recovery_phase is None else source.program.recovery_phase.to_document(),
        "parent_return": "CLEAR_AND_RETURN_TO_PARENT",
    }
    document = {
        "schema_version": MATERIALIZATION_SCHEMA_VERSION,
        "candidate_id": source.candidate_id,
        "program": source.program.to_document(),
        "outer_gate": [[_condition_document(c) for c in clause] for clause in source.outer_gate],
        "module": module,
        "state_mapping": mapping,
        "node_count": _count_nodes(tree),
    }
    return MaterializationArtifact(
        source.candidate_id, source.program, source.outer_gate, module, specialist,
        tuple(blackboard), mapping, document["node_count"], _hash(document),
    )


def _gate_matches(gate: tuple[tuple[ConditionGene, ...], ...], observation: Mapping[str, Any]) -> bool:
    from .ir import PublicCondition
    return any(all(_condition_matches(PublicCondition(c.feature, c.cmp, c.index, c.value), observation) for c in clause) for clause in gate)


class MaterializedInterpreter:
    """Executable projection of a materialized state map over exact observations."""

    def __init__(self, artifact: MaterializationArtifact, motifs: Mapping[str, Motif]):
        if artifact.schema_version != MATERIALIZATION_SCHEMA_VERSION or artifact.materialization_id != _hash(artifact.canonical_document()):
            raise ValueError("invalid materialization artifact")
        self.artifact = artifact
        self._runner = ReferenceInterpreter(artifact.program, motifs)
        self.gate_latched = False
        self._gate_open = False
        self.branch_latches: dict[int, bool] = {}

    @property
    def active(self) -> bool:
        return self._gate_open and self._runner.active

    @property
    def complete(self) -> bool:
        return (self.gate_latched and not self._gate_open) or self._runner.complete

    def action(self, observation: Mapping[str, Any], parent_action: Mapping[str, Any]) -> dict[str, Any]:
        parent = canonical_action(parent_action)
        if not self.gate_latched and int(observation["step"]) == self.artifact.program.initiation.step:
            self.gate_latched = True
            self._gate_open = _gate_matches(self.artifact.outer_gate, observation)
        if not self._gate_open:
            return parent
        return self._runner.action(observation, parent)

    def observe(self, post_observation: Mapping[str, Any]) -> None:
        if not self._gate_open:
            return
        before = self._runner.phase_index
        self._runner.observe(post_observation)
        decision = next((d for d in self.artifact.program.decisions if d.after_phase == before), None)
        if decision is not None and (self._runner.phase_index != before or self._runner.complete or self._runner._in_recovery):
            self.branch_latches[before] = _condition_matches(decision.predicate, post_observation)
