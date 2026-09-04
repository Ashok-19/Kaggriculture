"""Compact CUDA execution phenotypes and conservative BUILD sentinel screening."""
from __future__ import annotations

import json
import subprocess
import tempfile
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence

from kaggriculture_bonsai.search.evaluator import (
    CompletionReason,
    DeviceResult,
    encode_device_batch,
    decode_device_results,
)
from kaggriculture_bonsai.search.ir import ActionSource, Phase, PublicCondition, Termination, TerminationType
from kaggriculture_bonsai.search.motifs import Motif
from kaggriculture_bonsai.search.schema import stable_identity
from kaggriculture_bonsai.paths import PROJECT_ROOT

from .genome import SpecialistProgram

EXECUTION_PHENOTYPE_SCHEMA = "kaggriculture.aspe.execution-phenotype"
EXECUTION_PHENOTYPE_SCHEMA_VERSION = 1
STRATEGIC_EFFECT_SCHEMA = "kaggriculture.aspe.strategic-effect-phenotype"
STRATEGIC_EFFECT_SCHEMA_VERSION = 1
SENTINEL_SCHEMA = "kaggriculture.aspe.build-sentinel"
SENTINEL_SCHEMA_VERSION = 1
SCREENING_RESULT_SCHEMA = "kaggriculture.aspe.sentinel-screening-result"
SCREENING_RESULT_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class SentinelCase:
    identity: str
    tape: str
    seed: int
    seat: int
    member: str
    lane: str

    def to_document(self) -> dict[str, Any]:
        return {
            "identity": self.identity, "tape": self.tape, "seed": self.seed,
            "seat": self.seat, "member": self.member, "lane": self.lane,
        }


@dataclass(frozen=True)
class SentinelPanel:
    source_panel_identity: str
    cases: tuple[SentinelCase, ...]
    selection: str = "MEMBER_X_SEAT_ROTATING_SEED_V1"

    def __post_init__(self) -> None:
        if not self.cases:
            raise ValueError("sentinel panel must not be empty")
        if len({case.identity for case in self.cases}) != len(self.cases):
            raise ValueError("sentinel cases must be unique")

    @property
    def identity(self) -> str:
        return stable_identity("ASPE-BUILD-SENTINEL-V1", self.to_document(include_identity=False))

    def to_document(self, *, include_identity: bool = True) -> dict[str, Any]:
        result = {
            "schema": SENTINEL_SCHEMA,
            "schema_version": SENTINEL_SCHEMA_VERSION,
            "authority": "SCREENING_ONLY",
            "source_role": "BUILD",
            "source_panel_identity": self.source_panel_identity,
            "selection": self.selection,
            "case_count": len(self.cases),
            "coverage": {
                "seeds": sorted({case.seed for case in self.cases}),
                "seats": sorted({case.seat for case in self.cases}),
                "members": sorted({case.member for case in self.cases}),
                "lanes": sorted({case.lane for case in self.cases}),
            },
            "cases": [case.to_document() for case in self.cases],
        }
        if include_identity:
            result["identity"] = self.identity
        return result


def select_build_sentinel(baseline_manifest: str | Path) -> SentinelPanel:
    """Select one deterministic case for every BUILD member x seat pair.

    Seeds rotate across member/seat cells instead of taking a prefix. The frozen
    BUILD factorial therefore yields 16 cases spanning all eight lanes/members,
    both seats, and all three seeds without consulting DEV evidence.
    """
    document = json.loads(Path(baseline_manifest).read_text(encoding="utf-8"))
    build = document["panels"]["build"]
    rows = build["cases"]
    members = sorted({row["member"] for row in rows})
    seeds = sorted({int(row["seed"]) for row in rows})
    seats = sorted({int(row["seat"]) for row in rows})
    if len(members) < 2 or len(seeds) < 2 or seats != [0, 1]:
        raise ValueError("frozen BUILD lacks required sentinel factors")
    indexed = {(row["member"], int(row["seat"]), int(row["seed"])): row for row in rows}
    selected = []
    for member_index, member in enumerate(members):
        for seat in seats:
            seed = seeds[(member_index + seat) % len(seeds)]
            try:
                row = indexed[(member, seat, seed)]
            except KeyError as exc:
                raise ValueError("frozen BUILD is not a complete member/seat/seed factorial") from exc
            selected.append(SentinelCase(
                row["identity"], row["tape"], seed, seat, row["member"], row["lane"],
            ))
    return SentinelPanel(build["identity"], tuple(selected))


def encode_specialist_device_batch(
    programs: Sequence[SpecialistProgram], motifs: Mapping[str, Motif],
) -> bytes:
    if not programs:
        raise ValueError("specialist batch must not be empty")
    return encode_device_batch(
        tuple(program.to_macro_program() for program in programs), motifs,
        initiation_clauses=tuple(program.initiation_clauses for program in programs),
        semantic_ids=tuple(program.semantic_id for program in programs),
    )


@dataclass(frozen=True)
class ExecutionPhenotype:
    case_ids: tuple[str, ...]
    rows: tuple[DeviceResult, ...]

    def __post_init__(self) -> None:
        if not self.case_ids or len(self.case_ids) != len(self.rows):
            raise ValueError("execution phenotype requires one row per sentinel case")

    def to_document(self) -> dict[str, Any]:
        return {
            "schema": EXECUTION_PHENOTYPE_SCHEMA,
            "schema_version": EXECUTION_PHENOTYPE_SCHEMA_VERSION,
            "cases": [
                {
                    "case_id": case_id,
                    "activated": row.activated,
                    "first_activation_step": row.first_activation_step,
                    "last_activation_step": row.last_activation_step,
                    "activation_count": row.activation_count,
                    "specialist_duration": row.specialist_duration,
                    "phase_path": row.phase_path,
                    "phase_usage": row.phase_usage,
                    "branch_path": row.branch_path,
                    "termination_reason": row.end_reason.name,
                    "completion_reason": row.completion_reason.name,
                    "action_changes": row.action_changes,
                    "action_family_changes": [
                        row.family_farmer_changes, row.family_hands_changes,
                        row.family_market_changes,
                    ],
                    "action_trace_hash": f"{row.action_trace_hash:016x}",
                    "fault": row.fault.name,
                }
                for case_id, row in zip(self.case_ids, self.rows)
            ],
        }

    @property
    def identity(self) -> str:
        return stable_identity("ASPE-EXECUTION-PHENOTYPE-V1", self.to_document())


@dataclass(frozen=True)
class StrategicEffectPhenotype:
    cases: tuple[SentinelCase, ...]
    candidate_rows: tuple[DeviceResult, ...]
    baseline_rows: tuple[DeviceResult, ...]

    def __post_init__(self) -> None:
        if not self.cases or len(self.cases) != len(self.candidate_rows) or len(self.cases) != len(self.baseline_rows):
            raise ValueError("strategic effect requires aligned candidate/baseline rows")

    def to_document(self) -> dict[str, Any]:
        rows = []
        for case, candidate, baseline in zip(self.cases, self.candidate_rows, self.baseline_rows):
            base_margin = baseline.own_money - baseline.opp_money
            margin = candidate.own_money - candidate.opp_money
            rows.append({
                "case_id": case.identity, "seed": case.seed, "seat": case.seat,
                "member": case.member, "lane": case.lane,
                "margin_delta": margin - base_margin,
                "loss_to_win": base_margin < 0 < margin,
                "win_to_loss": base_margin > 0 > margin,
                "state_deltas": list(candidate.state_deltas),
            })
        return {
            "schema": STRATEGIC_EFFECT_SCHEMA,
            "schema_version": STRATEGIC_EFFECT_SCHEMA_VERSION,
            "cases": rows,
        }

    @property
    def identity(self) -> str:
        return stable_identity("ASPE-STRATEGIC-EFFECT-PHENOTYPE-V1", self.to_document())


class ScreeningClassification(str, Enum):
    ZERO_ACTIVATION = "ZERO_ACTIVATION"
    PHENOTYPE_DUPLICATE = "PHENOTYPE_DUPLICATE"
    EFFECT_DUPLICATE = "EFFECT_DUPLICATE"
    CLEARLY_HARMFUL = "CLEARLY_HARMFUL"
    NOVEL_OR_INFORMATIVE = "NOVEL_OR_INFORMATIVE"


@dataclass(frozen=True)
class ScreeningResult:
    structural_id: str
    execution_id: str
    strategic_effect_id: str
    classification: ScreeningClassification
    duplicate_of: str | None = None
    duplicate_level: str | None = None
    positive_support: tuple[int, int, int, int] = (0, 0, 0, 0)
    narrow_effect: bool = False

    def to_document(self) -> dict[str, Any]:
        return {
            "schema": SCREENING_RESULT_SCHEMA,
            "schema_version": SCREENING_RESULT_SCHEMA_VERSION,
            "structural_id": self.structural_id,
            "execution_phenotype_id": self.execution_id,
            "strategic_effect_id": self.strategic_effect_id,
            "classification": self.classification.value,
            "duplicate_of": self.duplicate_of,
            "duplicate_level": self.duplicate_level,
            "positive_support": dict(zip(("seeds", "seats", "members", "lanes"), self.positive_support)),
            "narrow_effect": self.narrow_effect,
            "authority": "SCREENING_ONLY",
        }


def classify_sentinel_results(
    programs: Sequence[SpecialistProgram], panel: SentinelPanel,
    rows: Sequence[DeviceResult], baseline_rows: Sequence[DeviceResult],
) -> tuple[ScreeningResult, ...]:
    case_count = len(panel.cases)
    if len(rows) != len(programs) * case_count or len(baseline_rows) != case_count:
        raise ValueError("sentinel result cardinality mismatch")
    seen_structural: dict[str, str] = {}
    seen_execution: dict[str, str] = {}
    seen_effect: dict[str, str] = {}
    results = []
    case_ids = tuple(case.identity for case in panel.cases)
    for index, program in enumerate(programs):
        candidate_rows = tuple(rows[index * case_count:(index + 1) * case_count])
        execution = ExecutionPhenotype(case_ids, candidate_rows)
        effect = StrategicEffectPhenotype(panel.cases, candidate_rows, tuple(baseline_rows))
        structural_id, execution_id, effect_id = program.semantic_id, execution.identity, effect.identity
        margin_effects = [
            (candidate.own_money - candidate.opp_money) - (baseline.own_money - baseline.opp_money)
            for candidate, baseline in zip(candidate_rows, baseline_rows)
        ]
        positive_cases = [case for case, value in zip(panel.cases, margin_effects) if value > 0]
        support = (
            len({case.seed for case in positive_cases}), len({case.seat for case in positive_cases}),
            len({case.member for case in positive_cases}), len({case.lane for case in positive_cases}),
        )
        narrow = bool(positive_cases) and (support[0] < 2 or support[1] < 2 or support[2] < 2 or support[3] < 2)
        duplicate_of = duplicate_level = None
        if not any(row.activated for row in candidate_rows):
            classification = ScreeningClassification.ZERO_ACTIVATION
        elif structural_id in seen_structural:
            classification = ScreeningClassification.PHENOTYPE_DUPLICATE
            duplicate_of, duplicate_level = seen_structural[structural_id], "STRUCTURAL"
        elif execution_id in seen_execution:
            classification = ScreeningClassification.PHENOTYPE_DUPLICATE
            duplicate_of, duplicate_level = seen_execution[execution_id], "EXECUTION"
        elif effect_id in seen_effect:
            classification = ScreeningClassification.EFFECT_DUPLICATE
            duplicate_of, duplicate_level = seen_effect[effect_id], "STRATEGIC_EFFECT"
        else:
            l2w = sum((baseline.own_money - baseline.opp_money) < 0 < (candidate.own_money - candidate.opp_money) for candidate, baseline in zip(candidate_rows, baseline_rows))
            w2l = sum((baseline.own_money - baseline.opp_money) > 0 > (candidate.own_money - candidate.opp_money) for candidate, baseline in zip(candidate_rows, baseline_rows))
            classification = (
                ScreeningClassification.CLEARLY_HARMFUL
                if w2l and not l2w and all(value <= 0 for value in margin_effects)
                else ScreeningClassification.NOVEL_OR_INFORMATIVE
            )
        results.append(ScreeningResult(
            structural_id, execution_id, effect_id, classification, duplicate_of,
            duplicate_level, support, narrow,
        ))
        seen_structural.setdefault(structural_id, structural_id)
        seen_execution.setdefault(execution_id, structural_id)
        seen_effect.setdefault(effect_id, structural_id)
    return tuple(results)


def run_cuda_sentinel(
    programs: Sequence[SpecialistProgram], motifs: Mapping[str, Motif], panel: SentinelPanel,
    *, binary: str | Path, asset_dir: str | Path,
) -> tuple[tuple[DeviceResult, ...], tuple[DeviceResult, ...]]:
    """Run one compact candidate batch plus an exact parent-only CUDA control."""
    if not programs:
        raise ValueError("sentinel evaluation requires at least one candidate")
    parent = SpecialistProgram(
        lane=programs[0].lane, activation_step=0,
        initiation_clauses=((PublicCondition("STEP", "GE", value=0),),),
        phases=(Phase(ActionSource.PARENT, (), Termination(TerminationType.FIXED_CAP, 718)),),
        max_total_steps=718,
    )
    all_programs = (parent, *programs)
    with tempfile.TemporaryDirectory() as directory:
        work = Path(directory)
        cases = work / "sentinel.tsv"
        cases.write_text("".join(
            f"{(PROJECT_ROOT / case.tape).resolve() if not Path(case.tape).is_absolute() else case.tape}\t{case.seat}\t{case.seed}\n"
            for case in panel.cases
        ), encoding="utf-8")
        batch = work / "specialists.rlrb"
        batch.write_bytes(encode_specialist_device_batch(all_programs, motifs))
        output = work / "telemetry.rlr"
        run = subprocess.run([
            str(binary), "--controller", str(Path(asset_dir) / "controller.bkgc"),
            "--routes", str(Path(asset_dir) / "routes.bkv6"),
            "--legacy-routes", str(Path(asset_dir) / "legacy_routes.bkv6"),
            "--cases", str(cases), "--programs", str(batch), "--out", str(output),
            "--feed-slot", "5", "--feed-count", "13", "--strategic-route-slot", "19",
        ], capture_output=True, text=True)
        if run.returncode not in (0, 1):
            raise RuntimeError(f"CUDA sentinel evaluator failed: {run.stderr.strip()}")
        count, case_count, result_rows = decode_device_results(output.read_bytes())
    if count != len(all_programs) or case_count != len(panel.cases):
        raise RuntimeError("CUDA sentinel evaluator returned wrong dimensions")
    baseline = result_rows[:case_count]
    return tuple(result_rows[case_count:]), tuple(baseline)


def write_sentinel_evidence(panel: SentinelPanel, output_dir: str | Path) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    panel_path = output / "sentinel_panel.tsv"
    panel_path.write_text("".join(
        f"{case.tape}\t{case.seat}\t{case.seed}\t0\t0\t0\t0\n" for case in panel.cases
    ), encoding="utf-8")
    document = panel.to_document()
    document["panel"] = panel_path.as_posix()
    (output / "manifest.json").write_text(
        json.dumps(document, sort_keys=True, indent=2) + "\n", encoding="utf-8",
    )
