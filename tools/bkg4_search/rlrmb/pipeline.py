"""Generic Phase-12 candidate certification orchestration.

This module binds exact CUDA effects and exact B4IS legal-state captures to the
already validated routability, CEGIS and materialization contracts.  It contains
no historical route, gate or threshold policy.
"""
from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from tools.bkg4_search.edit_search.goals import CONTRASTIVE_FEATURES
from tools.bkg4_search.information_state_io import load_r0, registry_manifest
from tools.bkg4_search.bonsai_composer import StrategicAuthority, compose_specialists
from tools.bkg4_search.module_codec import ConditionGene

from .beam import CudaBatchEvaluator
from .cegis_adapter import AdapterCase, CegisAdapterInput, CegisAdapterResult, run_hardened_cegis
from .ir import MacroProgram
from .motifs import Motif
from .canonical import canonical_action, load_route_actions
from .materialize import MaterializationArtifact, MaterializationInput, materialize_program
from .routability import (
    BoundaryState, RequiredContinuation, RoutabilityEvidence, RoutabilityResult,
    TrajectoryEvidence, screen_routability,
)
from .schema import DatasetRole, stable_identity

PIPELINE_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class CaseMeta:
    case_id: str
    tape: str
    seat: int
    seed: int
    family: str


@dataclass(frozen=True)
class ExactCaseEvidence:
    meta: CaseMeta
    role: DatasetRole
    baseline_margin: int
    candidate_margin: int
    legal_context: tuple[tuple[str, int, int], ...]
    exact_signature: tuple[int, ...]
    candidate_id: str

    @property
    def delta(self) -> int:
        return self.candidate_margin - self.baseline_margin


def load_case_panel(path: str | Path, label: str) -> tuple[CaseMeta, ...]:
    rows = []
    for index, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines()):
        if not line.strip() or line.startswith("#"):
            continue
        fields = line.split("\t")
        if len(fields) < 3:
            raise ValueError("case panel row requires tape, seat and seed")
        rows.append(CaseMeta(f"{label}:{index}", fields[0], int(fields[1]), int(fields[2]), Path(fields[0]).stem))
    if not rows:
        raise ValueError("case panel is empty")
    return tuple(rows)


def evaluate_candidate_panel(program: MacroProgram, motifs: Mapping[str, Motif], role: DatasetRole,
                             cases_path: str | Path, *, binary: str | Path, asset_dir: str | Path) -> tuple[ExactCaseEvidence, ...]:
    if role is DatasetRole.SEALED:
        raise ValueError("SEALED cannot enter generation-time candidate evaluation")
    meta = load_case_panel(cases_path, role.value.lower())
    batch = CudaBatchEvaluator(binary=binary, asset_dir=asset_dir, cases_path=cases_path)((program,), motifs)
    block = batch.programs[0].effect.blocks[0]
    return tuple(ExactCaseEvidence(row, role, base, base + delta, (), (), program.semantic_id)
                 for row, base, delta in zip(meta, block.baseline_margins, block.margin_deltas))


def capture_candidate_legal_states(program: MacroProgram, rows: Sequence[ExactCaseEvidence], cases_path: str | Path,
                                   out_dir: str | Path, *, capture_binary: str | Path, asset_dir: str | Path,
                                   feed_slot: int = 5, feed_count: int = 13, strategic_route_slot: int = 19) -> tuple[ExactCaseEvidence, ...]:
    if any(row.candidate_id != program.semantic_id for row in rows):
        raise ValueError("legal-state evidence is bound to a different candidate")
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    probes = out_dir / f"{program.semantic_id}.probes"; capture = out_dir / f"{program.semantic_id}.b4is"
    truth = out_dir / f"{program.semantic_id}.truth_unused.b4is"
    probe_steps = (program.initiation.step, *[program.initiation.step + sum(p.termination.max_steps for p in program.phases[:d.after_phase + 1]) for d in program.decisions])
    probes.write_text("\n".join(map(str, probe_steps)) + "\n", encoding="utf-8")
    command = [str(capture_binary), "--controller", str(Path(asset_dir) / "controller.bkgc"), "--routes", str(Path(asset_dir) / "routes.bkv6"),
               "--legacy-routes", str(Path(asset_dir) / "legacy_routes.bkv6"), "--cases", str(cases_path), "--probes", str(probes),
               "--out", str(capture), "--truth-out", str(truth), "--feed-slot", str(feed_slot), "--feed-count", str(feed_count),
               "--strategic-route-slot", str(strategic_route_slot)]
    subprocess.run(command, check=True, capture_output=True, text=True)
    manifest = capture.with_suffix(".manifest.json")
    manifest.write_text(json.dumps({"schema_version": PIPELINE_SCHEMA_VERSION, "candidate_id": program.semantic_id,
                                    "cases_path": str(Path(cases_path)), "probe_steps": probe_steps}, sort_keys=True) + "\n",
                        encoding="utf-8")
    return load_candidate_legal_states(program, rows, cases_path, capture)


def load_candidate_legal_states(program: MacroProgram, rows: Sequence[ExactCaseEvidence], cases_path: str | Path,
                                capture: str | Path) -> tuple[ExactCaseEvidence, ...]:
    """Bind a persisted exact capture only when its candidate/case manifest agrees."""
    capture = Path(capture)
    manifest_path = capture.with_suffix(".manifest.json")
    if not manifest_path.exists():
        raise ValueError("candidate-specific capture manifest is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != PIPELINE_SCHEMA_VERSION or manifest.get("candidate_id") != program.semantic_id:
        raise ValueError("legal-state capture belongs to a different candidate/version")
    if manifest.get("cases_path") != str(Path(cases_path)):
        raise ValueError("legal-state capture belongs to a different case panel")
    data = load_r0(capture); registry = registry_manifest()
    if data.cases != len(rows) or data.probes[0] != program.initiation.step:
        raise ValueError("candidate capture shape/boundary mismatch")
    result = []
    for index, row in enumerate(rows):
        frame = data.frame(index, 0); signature = tuple(int(x) for x in frame[:data.registry_width + data.bb_width + data.compat_width])
        context = tuple((feature, int(args[0]) if args else 0, int(frame[ri])) for ri, (feature, args) in enumerate(registry)
                        if feature in CONTRASTIVE_FEATURES and len(args) <= 1)
        result.append(ExactCaseEvidence(row.meta, row.role, row.baseline_margin, row.candidate_margin, context, signature, row.candidate_id))
    return tuple(result)


def build_routability_evidence(program: MacroProgram, rows: Sequence[ExactCaseEvidence]) -> RoutabilityEvidence:
    if any(row.candidate_id != program.semantic_id for row in rows):
        raise ValueError("evidence is bound to a different candidate")
    if any(row.role is DatasetRole.SEALED for row in rows):
        raise ValueError("SEALED cannot enter routability evidence")
    trajectories = []
    for row in rows:
        if not row.exact_signature or not row.legal_context:
            raise ValueError("routability requires exact captured legal state")
        continuation = program.semantic_id if row.delta > 0 else "PARENT"
        trajectories.append(TrajectoryEvidence(
            row.meta.case_id, row.role,
            (BoundaryState(program.initiation.step, row.exact_signature, row.legal_context, row.legal_context),),
            (RequiredContinuation(program.initiation.step, continuation),), row.delta,
            row.meta.family, row.meta.family, row.meta.seat, required_support=row.role is DatasetRole.TARGET and row.delta > 0,
        ))
    return RoutabilityEvidence(program.semantic_id, program.initiation.step, tuple(trajectories))


def build_cegis_input(program: MacroProgram, rows: Sequence[ExactCaseEvidence], routing: RoutabilityResult) -> CegisAdapterInput:
    cases = []
    target_positive = [row for row in rows if row.role is DatasetRole.TARGET and row.delta > 0]
    for row in rows:
        required = row in target_positive
        l2w = row.baseline_margin < 0 < row.candidate_margin
        cases.append(AdapterCase(row.meta.case_id, row.role, row.baseline_margin, row.delta, row.legal_context,
                                 row.exact_signature, row.meta.family, row.meta.family, row.meta.seat,
                                 required, required and l2w, stable_identity("RLRMB-PIPELINE-EVIDENCE-V1", row.meta.case_id, row.candidate_id)))
    return CegisAdapterInput(program.semantic_id, program.semantic_id, program.schema_version, len(program.decisions), tuple(cases), routing)


def certify_candidate(program: MacroProgram, rows: Sequence[ExactCaseEvidence]) -> tuple[RoutabilityResult, CegisAdapterResult | None]:
    routing = screen_routability(build_routability_evidence(program, rows))
    if routing.classification.value != "ROUTABLE_CANDIDATE":
        return routing, None
    return routing, run_hardened_cegis(build_cegis_input(program, rows, routing))


def _gate_conditions(gate: Any) -> tuple[tuple[ConditionGene, ...], ...]:
    if not gate:
        raise ValueError("accepted CEGIS result has no outer gate")
    return tuple(tuple(ConditionGene(str(c["feature"]), str(c["cmp"]), int(c.get("index", 0)), int(c["value"]))
                       for c in clause) for clause in gate)


def emit_candidate_route(program: MacroProgram, motifs: Mapping[str, Motif], parent_route: str | Path,
                         destination: str | Path) -> Path:
    """Emit the deterministic actuator route for a fixed-boundary v1 program.

    This is a transport artifact derived from semantic motif content, never a
    historical specialist declaration. State/event/reactive programs continue
    to use their explicit state mapping and therefore fail closed here.
    """
    if program.schema_version != 1 or program.decisions or any(p.termination.kind.value != "FIXED_CAP" for p in program.phases):
        raise ValueError("static route emission requires an open-loop fixed-boundary program")
    actions = [canonical_action(a) for a in load_route_actions(parent_route)]
    cursor = program.initiation.step
    for phase in program.phases:
        if phase.action_source.value == "PARENT":
            cursor += phase.termination.max_steps
            continue
        motif = motifs.get(phase.motif_id or "")
        if motif is None or motif.source_length < phase.termination.max_steps:
            raise ValueError("route emission is missing finite motif content")
        source_actions = motif.semantic_document()["action_block"]
        selected = {family.value for family in phase.action_families}
        for offset in range(phase.termination.max_steps):
            index = cursor + offset
            if index >= len(actions):
                raise ValueError("program exceeds parent route")
            parent, source = actions[index], source_actions[offset]
            actions[index] = {
                "farmer": source["farmer"] if "FARMER" in selected else parent["farmer"],
                "hands": source["hands"] if "HANDS" in selected else parent["hands"],
                "market": source["market"] if "MARKET" in selected else parent["market"],
            }
        cursor += phase.termination.max_steps
    destination = Path(destination); destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps({"actions": actions}, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    return destination


def materialize_certified_candidate(program: MacroProgram, motifs: Mapping[str, Motif], result: CegisAdapterResult,
                                    *, parent_route: str | Path, output_dir: str | Path, route_id: int,
                                    base_route_id: int = 64) -> tuple[MaterializationArtifact, Mapping[str, Any]]:
    if result.candidate_id != program.semantic_id or not result.accepted:
        raise ValueError("materialization requires the matching accepted CEGIS result")
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    route_path = emit_candidate_route(program, motifs, parent_route, output_dir / f"{program.semantic_id}.route.json")
    source = MaterializationInput(program.semantic_id, program, _gate_conditions(result.gate), motifs, route_id,
                                  base_route_id, str(route_path), "strategic_route", "slot_1")
    artifact = materialize_program(source)
    parent = {"name": "rlrmb_certification_parent", "blackboard": [{"name": "slot_1", "type": "I32", "initial": 0, "owner": "parent"}],
              "tree": {"type": "Sequence", "children": []}, "strategic_routes": []}
    composed = compose_specialists(parent, (artifact.specialist,), (),
                                   strategic_authorities=(StrategicAuthority("strategic_route", "slot_1"),),
                                   output_name=f"certified_{program.semantic_id[:12]}")
    audit = {"materialization_id": artifact.materialization_id, "node_count": artifact.node_count,
             "state_slots": len(artifact.blackboard), "composition_order": composed.report["composition_order"],
             "unresolved_conflicts": composed.report["unresolved_conflicts"], "semantic_mismatches": 0}
    (output_dir / "materialization.json").write_text(artifact.canonical_json() + "\n", encoding="utf-8")
    (output_dir / "composition.json").write_text(json.dumps(audit, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return artifact, audit
