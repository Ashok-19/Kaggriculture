"""Execution and BKG4 materialization adapters for SpecialistProgram v1."""
from __future__ import annotations

from typing import Any, Mapping

from kaggriculture_bonsai.routing.materialize import (
    MaterializationArtifact,
    MaterializationInput,
    materialize_program,
)
from kaggriculture_bonsai.search.evaluator import _condition_matches
from kaggriculture_bonsai.search.motifs import Motif

from .genome import SpecialistProgram


def initiation_matches(program: SpecialistProgram, observation: Mapping[str, Any]) -> bool:
    """Evaluate the canonical DNF initiation gate on legal current state."""
    if int(observation["step"]) != program.activation_step:
        return False
    return any(
        all(_condition_matches(condition, observation) for condition in clause)
        for clause in program.initiation_clauses
    )


def materialize_specialist(
    program: SpecialistProgram,
    motifs: Mapping[str, Motif],
    *,
    route_id: int,
    base_route_id: int,
    route_source: str,
    authority: str,
    authority_slot: str,
    reserved_blackboard: tuple[str, ...] = (),
) -> MaterializationArtifact:
    """Project one ASPE program through the existing BKG4 materializer."""
    source = MaterializationInput(
        candidate_id=program.semantic_id,
        program=program.to_macro_program(),
        outer_gate=program.outer_gate,
        motifs=motifs,
        route_id=route_id,
        base_route_id=base_route_id,
        route_source=route_source,
        authority=authority,
        authority_slot=authority_slot,
        reserved_blackboard=reserved_blackboard,
        audit_metadata={"representation": "ASPE_SPECIALIST_PROGRAM_V1"},
    )
    return materialize_program(source)
