"""Persistent, deterministic semantic memory for evaluated ASPE specialists."""
from __future__ import annotations

import json
import os
import copy
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from kaggriculture_bonsai.search.schema import stable_identity

from .genome import SpecialistProgram
from .phenotype import (
    ExecutionPhenotype,
    ScreeningClassification,
    ScreeningResult,
    SentinelPanel,
    StrategicEffectPhenotype,
    classify_sentinel_results,
)

ASPE_ARCHIVE_SCHEMA = "kaggriculture.aspe.semantic-archive"
ASPE_ARCHIVE_SCHEMA_VERSION = 1
ASPE_EVALUATION_SCHEMA = "kaggriculture.aspe.evaluation-record"
ASPE_EVALUATION_SCHEMA_VERSION = 1
DEFAULT_MAX_EVALUATIONS = 100_000


class FailureReason(str, Enum):
    STRUCTURAL_DUPLICATE = "STRUCTURAL_DUPLICATE"
    PHENOTYPE_DUPLICATE = "PHENOTYPE_DUPLICATE"
    EFFECT_DUPLICATE = "EFFECT_DUPLICATE"
    ZERO_ACTIVATION = "ZERO_ACTIVATION"
    CLEARLY_HARMFUL = "CLEARLY_HARMFUL"
    NARROW_SUPPORT = "NARROW_SUPPORT"
    BUILD_REGRESSION = "BUILD_REGRESSION"
    DEV_REGRESSION = "DEV_REGRESSION"
    SEMANTIC_FAULT = "SEMANTIC_FAULT"
    SEMANTIC_MISMATCH = "SEMANTIC_MISMATCH"
    DOWNSTREAM_FAILURE = "DOWNSTREAM_FAILURE"


def _exact_nonnegative(value: Any, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _string(value: Any, name: str, *, empty: bool = False) -> str:
    if type(value) is not str or (not empty and not value):
        raise ValueError(f"{name} must be a {'string' if empty else 'non-empty string'}")
    return value


def _canonical_mapping(value: Mapping[str, Any] | None, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    decoded = json.loads(encoded)
    if not isinstance(decoded, dict):
        raise TypeError(f"{name} must encode a JSON object")
    return decoded


@dataclass(frozen=True)
class ProposalMetadata:
    operator: str = "UNSPECIFIED"
    mutation_site: str = ""
    context: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        _string(self.operator, "operator")
        _string(self.mutation_site, "mutation_site", empty=True)
        object.__setattr__(self, "context", _canonical_mapping(self.context, "proposal context"))

    def to_document(self) -> dict[str, Any]:
        return {
            "operator": self.operator,
            "mutation_site": self.mutation_site,
            "context": dict(self.context or {}),
        }

    @classmethod
    def from_document(cls, value: Mapping[str, Any]) -> "ProposalMetadata":
        if set(value) != {"operator", "mutation_site", "context"}:
            raise ValueError("malformed proposal metadata")
        return cls(value["operator"], value["mutation_site"], value["context"])


@dataclass(frozen=True)
class CaseOutcome:
    case_id: str
    baseline_margin: int
    candidate_margin: int
    seed: int | None = None
    seat: int | None = None
    member: str = ""
    lane: str = ""

    def __post_init__(self) -> None:
        _string(self.case_id, "case_id")
        if type(self.baseline_margin) is not int or type(self.candidate_margin) is not int:
            raise TypeError("case margins must be integers")
        if self.seed is not None and type(self.seed) is not int:
            raise TypeError("case seed must be an integer or None")
        if self.seat is not None and self.seat not in (0, 1):
            raise ValueError("case seat must be 0, 1, or None")
        _string(self.member, "member", empty=True)
        _string(self.lane, "lane", empty=True)

    @property
    def margin_delta(self) -> int:
        return self.candidate_margin - self.baseline_margin

    def to_document(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "baseline_margin": self.baseline_margin,
            "candidate_margin": self.candidate_margin,
            "margin_delta": self.margin_delta,
            "loss_to_win": self.baseline_margin < 0 < self.candidate_margin,
            "win_to_loss": self.baseline_margin > 0 > self.candidate_margin,
            "seed": self.seed,
            "seat": self.seat,
            "member": self.member,
            "lane": self.lane,
        }

    @classmethod
    def from_document(cls, value: Mapping[str, Any]) -> "CaseOutcome":
        required = {"case_id", "baseline_margin", "candidate_margin", "margin_delta", "loss_to_win", "win_to_loss", "seed", "seat", "member", "lane"}
        if set(value) != required:
            raise ValueError("malformed case outcome")
        result = cls(value["case_id"], value["baseline_margin"], value["candidate_margin"], value["seed"], value["seat"], value["member"], value["lane"])
        if result.to_document() != dict(value):
            raise ValueError("case outcome derived fields disagree")
        return result


@dataclass(frozen=True)
class PanelEvidence:
    role: str
    panel_id: str
    cases: tuple[CaseOutcome, ...]
    passed: bool | None = None

    def __post_init__(self) -> None:
        if self.role not in {"BUILD", "INTERNAL_DEV", "TRANSFER", "VALIDATION", "CEGIS"}:
            raise ValueError("archive panel role must be pre-SEALED research evidence")
        _string(self.panel_id, "panel_id")
        if not isinstance(self.cases, tuple) or not self.cases or any(not isinstance(row, CaseOutcome) for row in self.cases):
            raise ValueError("panel evidence requires CaseOutcome values")
        if self.passed is not None and type(self.passed) is not bool:
            raise TypeError("panel passed must be bool or None")

    def to_document(self) -> dict[str, Any]:
        positive = tuple(row for row in self.cases if row.margin_delta > 0)
        return {
            "role": self.role,
            "panel_id": self.panel_id,
            "passed": self.passed,
            "case_count": len(self.cases),
            "loss_to_win": sum(row.baseline_margin < 0 < row.candidate_margin for row in self.cases),
            "win_to_loss": sum(row.baseline_margin > 0 > row.candidate_margin for row in self.cases),
            "support": {
                "seeds": sorted({row.seed for row in positive if row.seed is not None}),
                "seats": sorted({row.seat for row in positive if row.seat is not None}),
                "members": sorted({row.member for row in positive if row.member}),
                "lanes": sorted({row.lane for row in positive if row.lane}),
            },
            "margins": [row.to_document() for row in self.cases],
        }

    @classmethod
    def from_document(cls, value: Mapping[str, Any]) -> "PanelEvidence":
        required = {"role", "panel_id", "passed", "case_count", "loss_to_win", "win_to_loss", "support", "margins"}
        if set(value) != required or not isinstance(value["margins"], list):
            raise ValueError("malformed panel evidence")
        result = cls(value["role"], value["panel_id"], tuple(CaseOutcome.from_document(row) for row in value["margins"]), value["passed"])
        if result.to_document() != dict(value):
            raise ValueError("panel evidence derived fields disagree")
        return result


@dataclass(frozen=True)
class EvaluationRecord:
    program: SpecialistProgram
    context_id: str
    generation: int
    execution_phenotype: Mapping[str, Any]
    strategic_effect: Mapping[str, Any]
    screening: ScreeningResult
    parent_id: str | None = None
    lineage: tuple[str, ...] = ()
    proposal: ProposalMetadata = field(default_factory=ProposalMetadata)
    build_evidence: PanelEvidence | None = None
    downstream_evidence: tuple[PanelEvidence, ...] = ()
    failure_reasons: tuple[FailureReason, ...] = ()
    semantic_fault: str | None = None
    semantic_mismatch: bool = False
    stepping_stone: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.program, SpecialistProgram):
            raise TypeError("program must be a SpecialistProgram")
        _string(self.context_id, "context_id")
        _exact_nonnegative(self.generation, "generation")
        execution = _canonical_mapping(self.execution_phenotype, "execution phenotype")
        effect = _canonical_mapping(self.strategic_effect, "strategic effect")
        if stable_identity("ASPE-EXECUTION-PHENOTYPE-V1", execution) != self.screening.execution_id:
            raise ValueError("execution phenotype identity mismatch")
        if stable_identity("ASPE-STRATEGIC-EFFECT-PHENOTYPE-V1", effect) != self.screening.strategic_effect_id:
            raise ValueError("strategic-effect phenotype identity mismatch")
        object.__setattr__(self, "execution_phenotype", execution)
        object.__setattr__(self, "strategic_effect", effect)
        if self.screening.structural_id != self.program.semantic_id:
            raise ValueError("screening structural identity mismatch")
        if self.parent_id is not None:
            _string(self.parent_id, "parent_id")
        if not isinstance(self.lineage, tuple) or any(type(item) is not str or not item for item in self.lineage):
            raise ValueError("lineage must contain non-empty identities")
        if self.parent_id is not None and (not self.lineage or self.lineage[-1] != self.parent_id):
            raise ValueError("lineage must end at parent_id")
        if not isinstance(self.proposal, ProposalMetadata):
            raise TypeError("proposal must be ProposalMetadata")
        if self.build_evidence is not None and self.build_evidence.role != "BUILD":
            raise ValueError("build_evidence must have BUILD role")
        if not isinstance(self.downstream_evidence, tuple) or any(not isinstance(row, PanelEvidence) for row in self.downstream_evidence):
            raise ValueError("downstream evidence must contain PanelEvidence values")
        if any(row.role == "BUILD" for row in self.downstream_evidence):
            raise ValueError("downstream evidence must exclude BUILD")
        roles = [row.role for row in self.downstream_evidence]
        if len(set(roles)) != len(roles):
            raise ValueError("downstream evidence roles must be unique")
        failures = tuple(sorted({FailureReason(item) for item in self.failure_reasons}, key=lambda item: item.value))
        object.__setattr__(self, "failure_reasons", failures)
        if self.semantic_fault is not None:
            _string(self.semantic_fault, "semantic_fault")
            if FailureReason.SEMANTIC_FAULT not in failures:
                raise ValueError("semantic fault requires SEMANTIC_FAULT failure evidence")
        if self.semantic_mismatch and FailureReason.SEMANTIC_MISMATCH not in failures:
            raise ValueError("semantic mismatch requires SEMANTIC_MISMATCH failure evidence")

    @property
    def evaluation_id(self) -> str:
        return stable_identity("ASPE-EVALUATION-KEY-V1", self.program.semantic_id, self.context_id)

    def to_document(self) -> dict[str, Any]:
        return {
            "schema": ASPE_EVALUATION_SCHEMA,
            "schema_version": ASPE_EVALUATION_SCHEMA_VERSION,
            "evaluation_id": self.evaluation_id,
            "program_id": self.program.semantic_id,
            "structural_id": self.program.semantic_id,
            "context_id": self.context_id,
            "generation": self.generation,
            "parent_id": self.parent_id,
            "lineage": list(self.lineage),
            "proposal": self.proposal.to_document(),
            "execution_phenotype_id": self.screening.execution_id,
            "strategic_effect_id": self.screening.strategic_effect_id,
            "sentinel": self.screening.to_document(),
            "build_evidence": None if self.build_evidence is None else self.build_evidence.to_document(),
            "downstream_evidence": [row.to_document() for row in self.downstream_evidence],
            "failure_reasons": [item.value for item in self.failure_reasons],
            "semantic_fault": self.semantic_fault,
            "semantic_mismatch": self.semantic_mismatch,
            "stepping_stone": self.stepping_stone,
            "authority": "SEARCH_MEMORY_ONLY",
        }

    @classmethod
    def from_document(
        cls,
        value: Mapping[str, Any],
        *,
        program: SpecialistProgram,
        execution_phenotype: Mapping[str, Any],
        strategic_effect: Mapping[str, Any],
    ) -> "EvaluationRecord":
        required = {
            "schema", "schema_version", "evaluation_id", "program_id", "structural_id",
            "context_id", "generation", "parent_id", "lineage", "proposal",
            "execution_phenotype_id", "strategic_effect_id", "sentinel", "build_evidence",
            "downstream_evidence", "failure_reasons", "semantic_fault", "semantic_mismatch",
            "stepping_stone", "authority",
        }
        if set(value) != required or value["schema"] != ASPE_EVALUATION_SCHEMA or value["schema_version"] != ASPE_EVALUATION_SCHEMA_VERSION:
            raise ValueError("malformed ASPE evaluation record")
        if value["program_id"] != program.semantic_id or value["structural_id"] != program.semantic_id:
            raise ValueError("evaluation program identity mismatch")
        sentinel = value["sentinel"]
        screening = ScreeningResult(
            structural_id=sentinel["structural_id"],
            execution_id=sentinel["execution_phenotype_id"],
            strategic_effect_id=sentinel["strategic_effect_id"],
            classification=ScreeningClassification(sentinel["classification"]),
            duplicate_of=sentinel["duplicate_of"],
            duplicate_level=sentinel["duplicate_level"],
            positive_support=tuple(sentinel["positive_support"][key] for key in ("seeds", "seats", "members", "lanes")),
            narrow_effect=sentinel["narrow_effect"],
        )
        result = cls(
            program=program,
            context_id=value["context_id"],
            generation=value["generation"],
            parent_id=value["parent_id"],
            lineage=tuple(value["lineage"]),
            proposal=ProposalMetadata.from_document(value["proposal"]),
            execution_phenotype=execution_phenotype,
            strategic_effect=strategic_effect,
            screening=screening,
            build_evidence=None if value["build_evidence"] is None else PanelEvidence.from_document(value["build_evidence"]),
            downstream_evidence=tuple(PanelEvidence.from_document(row) for row in value["downstream_evidence"]),
            failure_reasons=tuple(FailureReason(item) for item in value["failure_reasons"]),
            semantic_fault=value["semantic_fault"],
            semantic_mismatch=value["semantic_mismatch"],
            stepping_stone=value["stepping_stone"],
        )
        if result.to_document() != dict(value):
            raise ValueError("ASPE evaluation record roundtrip mismatch")
        return result


@dataclass(frozen=True)
class InsertResult:
    inserted: bool
    evaluation_id: str
    structural_duplicate_of: str | None
    execution_duplicate_of: str | None
    effect_duplicate_of: str | None


@dataclass(frozen=True)
class Neighbor:
    phenotype_id: str
    distance: int
    program_ids: tuple[str, ...]


def _execution_vector(document: Mapping[str, Any]) -> tuple[Any, ...]:
    vector: list[Any] = []
    for row in document.get("cases", ()):
        vector.extend((
            row["activated"], row["first_activation_step"], row["last_activation_step"],
            row["activation_count"], row["specialist_duration"], row["phase_path"],
            row["phase_usage"], row["branch_path"], row["termination_reason"],
            row["completion_reason"], row["action_changes"],
            *row["action_family_changes"], row.get("action_trace_hash", "0"), row["fault"],
        ))
    return tuple(vector)


def phenotype_distance(left: Mapping[str, Any], right: Mapping[str, Any]) -> int:
    """Exact component Hamming distance over compact Phase-2 behavior fields."""
    a, b = _execution_vector(left), _execution_vector(right)
    common = sum(x != y for x, y in zip(a, b))
    return common + abs(len(a) - len(b))


class SemanticArchive:
    """One canonical JSON archive with atomic, duplicate-safe replacement."""

    def __init__(self, path: str | Path, *, max_evaluations: int = DEFAULT_MAX_EVALUATIONS) -> None:
        self.path = Path(path)
        if type(max_evaluations) is not int or max_evaluations < 1:
            raise ValueError("max_evaluations must be positive")
        self.max_evaluations = max_evaluations
        self._document: dict[str, Any] | None = None

    def open(self) -> "SemanticArchive":
        if self.path.exists():
            document = json.loads(self.path.read_text(encoding="utf-8"))
            self._validate(document)
            self._document = document
        else:
            self._document = self.empty_document()
            self._persist()
        return self

    def close(self) -> None:
        self._document = None

    def __enter__(self) -> "SemanticArchive":
        return self.open()

    def __exit__(self, *_: Any) -> None:
        self.close()

    @staticmethod
    def empty_document() -> dict[str, Any]:
        return {
            "schema": ASPE_ARCHIVE_SCHEMA,
            "schema_version": ASPE_ARCHIVE_SCHEMA_VERSION,
            "authority": "SEARCH_MEMORY_ONLY",
            "programs": {},
            "execution_phenotypes": {},
            "strategic_effects": {},
            "evaluations": {},
        }

    @property
    def document(self) -> dict[str, Any]:
        if self._document is None:
            raise RuntimeError("SemanticArchive must be opened")
        return self._document

    def _validate(self, document: Mapping[str, Any]) -> None:
        if set(document) != {"schema", "schema_version", "authority", "programs", "execution_phenotypes", "strategic_effects", "evaluations"}:
            raise ValueError("malformed ASPE archive document")
        if document["schema"] != ASPE_ARCHIVE_SCHEMA or document["schema_version"] != ASPE_ARCHIVE_SCHEMA_VERSION:
            raise ValueError("unsupported ASPE archive schema")
        if document["authority"] != "SEARCH_MEMORY_ONLY":
            raise ValueError("ASPE archive cannot grant promotion authority")
        for key in ("programs", "execution_phenotypes", "strategic_effects", "evaluations"):
            if not isinstance(document[key], dict):
                raise ValueError(f"archive {key} must be an object")
        for identity, program in document["programs"].items():
            if SpecialistProgram.from_document(program).semantic_id != identity:
                raise ValueError("archive program identity mismatch")
        for identity, phenotype in document["execution_phenotypes"].items():
            if stable_identity("ASPE-EXECUTION-PHENOTYPE-V1", phenotype) != identity:
                raise ValueError("archive execution phenotype identity mismatch")
        for identity, effect in document["strategic_effects"].items():
            if stable_identity("ASPE-STRATEGIC-EFFECT-PHENOTYPE-V1", effect) != identity:
                raise ValueError("archive strategic effect identity mismatch")
        for identity, evaluation in document["evaluations"].items():
            if evaluation.get("evaluation_id") != identity:
                raise ValueError("archive evaluation identity mismatch")
            if evaluation.get("program_id") not in document["programs"]:
                raise ValueError("archive evaluation references missing program")
            if evaluation.get("execution_phenotype_id") not in document["execution_phenotypes"]:
                raise ValueError("archive evaluation references missing execution phenotype")
            if evaluation.get("strategic_effect_id") not in document["strategic_effects"]:
                raise ValueError("archive evaluation references missing strategic effect")
            EvaluationRecord.from_document(
                evaluation,
                program=SpecialistProgram.from_document(document["programs"][evaluation["program_id"]]),
                execution_phenotype=document["execution_phenotypes"][evaluation["execution_phenotype_id"]],
                strategic_effect=document["strategic_effects"][evaluation["strategic_effect_id"]],
            )

    def _persist(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(self.document, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
        temporary = self.path.with_name(f".{self.path.name}.tmp")
        try:
            with temporary.open("w", encoding="utf-8") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.path)
            directory = os.open(self.path.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        finally:
            if temporary.exists():
                temporary.unlink()

    def has_program(self, program_id: str) -> bool:
        return program_id in self.document["programs"]

    def has_execution_phenotype(self, phenotype_id: str) -> bool:
        return phenotype_id in self.document["execution_phenotypes"]

    def has_strategic_effect(self, effect_id: str) -> bool:
        return effect_id in self.document["strategic_effects"]

    def get_evaluation(self, program_id: str, context_id: str) -> Mapping[str, Any] | None:
        key = stable_identity("ASPE-EVALUATION-KEY-V1", program_id, context_id)
        return self.document["evaluations"].get(key)

    def _record(self, evaluation: Mapping[str, Any]) -> EvaluationRecord:
        return EvaluationRecord.from_document(
            evaluation,
            program=SpecialistProgram.from_document(self.document["programs"][evaluation["program_id"]]),
            execution_phenotype=self.document["execution_phenotypes"][evaluation["execution_phenotype_id"]],
            strategic_effect=self.document["strategic_effects"][evaluation["strategic_effect_id"]],
        )

    def update_evidence(
        self,
        program_id: str,
        context_id: str,
        evidence: PanelEvidence,
        *,
        failure_reasons: Iterable[FailureReason] = (),
        semantic_fault: str | None = None,
        semantic_mismatch: bool | None = None,
    ) -> None:
        """Atomically enrich an existing sentinel evaluation with later evidence."""
        evaluation_id = stable_identity("ASPE-EVALUATION-KEY-V1", program_id, context_id)
        current_document = self.document["evaluations"].get(evaluation_id)
        if current_document is None:
            raise KeyError("cannot enrich an unevaluated program/context")
        current = self._record(current_document)
        failures = set(current.failure_reasons)
        failures.update(FailureReason(item) for item in failure_reasons)
        if evidence.role == "BUILD" and any(row.baseline_margin > 0 > row.candidate_margin for row in evidence.cases):
            failures.add(FailureReason.BUILD_REGRESSION)
        elif evidence.role == "INTERNAL_DEV" and evidence.passed is False:
            failures.add(FailureReason.DEV_REGRESSION)
        elif evidence.role not in {"BUILD", "INTERNAL_DEV"} and evidence.passed is False:
            failures.add(FailureReason.DOWNSTREAM_FAILURE)
        fault = current.semantic_fault if semantic_fault is None else semantic_fault
        mismatch = current.semantic_mismatch if semantic_mismatch is None else semantic_mismatch
        if fault is not None:
            failures.add(FailureReason.SEMANTIC_FAULT)
        if mismatch:
            failures.add(FailureReason.SEMANTIC_MISMATCH)
        if evidence.role == "BUILD":
            build = evidence
            downstream = current.downstream_evidence
        else:
            build = current.build_evidence
            by_role = {row.role: row for row in current.downstream_evidence}
            by_role[evidence.role] = evidence
            downstream = tuple(by_role[key] for key in sorted(by_role))
        updated = EvaluationRecord(
            program=current.program, context_id=current.context_id, generation=current.generation,
            execution_phenotype=current.execution_phenotype, strategic_effect=current.strategic_effect,
            screening=current.screening, parent_id=current.parent_id, lineage=current.lineage,
            proposal=current.proposal, build_evidence=build, downstream_evidence=downstream,
            failure_reasons=tuple(failures), semantic_fault=fault, semantic_mismatch=mismatch,
            stepping_stone=current.stepping_stone,
        )
        before = copy.deepcopy(self.document)
        try:
            self.document["evaluations"][evaluation_id] = updated.to_document()
            self._persist()
        except Exception:
            self._document = before
            raise

    def programs_for_execution(self, phenotype_id: str) -> tuple[str, ...]:
        return tuple(sorted({row["program_id"] for row in self.document["evaluations"].values() if row["execution_phenotype_id"] == phenotype_id}))

    def programs_for_effect(self, effect_id: str) -> tuple[str, ...]:
        return tuple(sorted({row["program_id"] for row in self.document["evaluations"].values() if row["strategic_effect_id"] == effect_id}))

    def insert(self, record: EvaluationRecord) -> InsertResult:
        evaluation_id = record.evaluation_id
        canonical = record.to_document()
        existing = self.document["evaluations"].get(evaluation_id)
        if existing is not None:
            duplicate_failures = {
                FailureReason.STRUCTURAL_DUPLICATE.value,
                FailureReason.PHENOTYPE_DUPLICATE.value,
                FailureReason.EFFECT_DUPLICATE.value,
            }
            left, right = copy.deepcopy(existing), copy.deepcopy(canonical)
            left["failure_reasons"] = [value for value in left["failure_reasons"] if value not in duplicate_failures]
            right["failure_reasons"] = [value for value in right["failure_reasons"] if value not in duplicate_failures]
            if left != right:
                raise ValueError("conflicting repeated evaluation for program/context")
            return InsertResult(False, evaluation_id, record.program.semantic_id, record.screening.execution_id, record.screening.strategic_effect_id)
        if len(self.document["evaluations"]) >= self.max_evaluations:
            raise OverflowError("ASPE archive evaluation bound reached")
        structural = record.program.semantic_id if self.has_program(record.program.semantic_id) else None
        execution = record.screening.execution_id if self.has_execution_phenotype(record.screening.execution_id) else None
        effect = record.screening.strategic_effect_id if self.has_strategic_effect(record.screening.strategic_effect_id) else None
        remembered_failures = set(canonical["failure_reasons"])
        if structural is not None:
            remembered_failures.add(FailureReason.STRUCTURAL_DUPLICATE.value)
        if execution is not None:
            remembered_failures.add(FailureReason.PHENOTYPE_DUPLICATE.value)
        if effect is not None:
            remembered_failures.add(FailureReason.EFFECT_DUPLICATE.value)
        canonical["failure_reasons"] = sorted(remembered_failures)
        before = copy.deepcopy(self.document)
        try:
            self.document["programs"].setdefault(record.program.semantic_id, record.program.to_document())
            self.document["execution_phenotypes"].setdefault(record.screening.execution_id, dict(record.execution_phenotype))
            self.document["strategic_effects"].setdefault(record.screening.strategic_effect_id, dict(record.strategic_effect))
            self.document["evaluations"][evaluation_id] = canonical
            self._persist()
        except Exception:
            self._document = before
            raise
        return InsertResult(True, evaluation_id, structural, execution, effect)

    def nearest_phenotypes(self, phenotype: Mapping[str, Any], *, limit: int = 5) -> tuple[Neighbor, ...]:
        if type(limit) is not int or limit < 1:
            raise ValueError("neighbor limit must be positive")
        rows = [
            Neighbor(identity, phenotype_distance(phenotype, document), self.programs_for_execution(identity))
            for identity, document in self.document["execution_phenotypes"].items()
        ]
        return tuple(sorted(rows, key=lambda row: (row.distance, row.phenotype_id))[:limit])

    def novelty(self, phenotype: Mapping[str, Any]) -> tuple[int, int]:
        """Return exact (nearest distance, descriptor length); no float drift."""
        vector_length = len(_execution_vector(phenotype))
        neighbors = self.nearest_phenotypes(phenotype, limit=1) if self.document["execution_phenotypes"] else ()
        return (vector_length if not neighbors else neighbors[0].distance, vector_length)

    def stepping_stones(self) -> tuple[Mapping[str, Any], ...]:
        return tuple(sorted(
            (row for row in self.document["evaluations"].values() if row["stepping_stone"]),
            key=lambda row: row["evaluation_id"],
        ))

    def lineage(self, program_id: str) -> tuple[str, ...] | None:
        rows = [row for row in self.document["evaluations"].values() if row["program_id"] == program_id]
        if not rows:
            return None
        rows.sort(key=lambda row: (row["generation"], row["evaluation_id"]))
        return tuple(rows[0]["lineage"])

    def failure_statistics(self, *, parent_id: str | None = None, operator: str | None = None, context_id: str | None = None) -> dict[str, int]:
        counts: dict[str, int] = {}
        for row in self.document["evaluations"].values():
            if parent_id is not None and row["parent_id"] != parent_id:
                continue
            if operator is not None and row["proposal"]["operator"] != operator:
                continue
            if context_id is not None and row["context_id"] != context_id:
                continue
            for reason in row["failure_reasons"]:
                counts[reason] = counts.get(reason, 0) + 1
        return dict(sorted(counts.items()))

    def novel_since_generation(self, generation: int) -> tuple[str, ...]:
        _exact_nonnegative(generation, "generation")
        first: dict[str, int] = {}
        for row in self.document["evaluations"].values():
            identity = row["execution_phenotype_id"]
            first[identity] = min(first.get(identity, row["generation"]), row["generation"])
        return tuple(sorted(identity for identity, first_generation in first.items() if first_generation >= generation))


def _screening_failures(screening: ScreeningResult) -> tuple[FailureReason, ...]:
    mapping = {
        ScreeningClassification.ZERO_ACTIVATION: FailureReason.ZERO_ACTIVATION,
        ScreeningClassification.PHENOTYPE_DUPLICATE: FailureReason.PHENOTYPE_DUPLICATE,
        ScreeningClassification.EFFECT_DUPLICATE: FailureReason.EFFECT_DUPLICATE,
        ScreeningClassification.CLEARLY_HARMFUL: FailureReason.CLEARLY_HARMFUL,
    }
    values = []
    if screening.duplicate_level == "STRUCTURAL":
        values.append(FailureReason.STRUCTURAL_DUPLICATE)
    elif screening.classification in mapping:
        values.append(mapping[screening.classification])
    if screening.narrow_effect:
        values.append(FailureReason.NARROW_SUPPORT)
    return tuple(values)


def record_sentinel_batch(
    archive: SemanticArchive,
    programs: Sequence[SpecialistProgram],
    panel: SentinelPanel,
    candidate_rows: Sequence[Any],
    baseline_rows: Sequence[Any],
    *,
    context_id: str,
    generation: int,
    parents: Sequence[str | None] | None = None,
    lineages: Sequence[Sequence[str]] | None = None,
    proposals: Sequence[ProposalMetadata] | None = None,
    stepping_stones: Sequence[bool] | None = None,
    semantic_mismatches: Sequence[bool] | None = None,
) -> tuple[InsertResult, ...]:
    """Persist one already-evaluated Phase-2 CUDA sentinel batch."""
    size = len(programs)
    parents = tuple(parents) if parents is not None else (None,) * size
    lineages = tuple(lineages) if lineages is not None else ((),) * size
    proposals = tuple(proposals) if proposals is not None else (ProposalMetadata(),) * size
    stepping_stones = tuple(stepping_stones) if stepping_stones is not None else (False,) * size
    semantic_mismatches = tuple(semantic_mismatches) if semantic_mismatches is not None else (False,) * size
    if not all(len(values) == size for values in (parents, lineages, proposals, stepping_stones, semantic_mismatches)):
        raise ValueError("sentinel archive metadata cardinality mismatch")
    screenings = classify_sentinel_results(programs, panel, candidate_rows, baseline_rows)
    case_count = len(panel.cases)
    inserted = []
    for index, (program, screening) in enumerate(zip(programs, screenings)):
        existing = archive.get_evaluation(program.semantic_id, context_id)
        if existing is not None:
            inserted.append(InsertResult(
                False, existing["evaluation_id"], program.semantic_id,
                existing["execution_phenotype_id"], existing["strategic_effect_id"],
            ))
            continue
        rows = tuple(candidate_rows[index * case_count:(index + 1) * case_count])
        execution = ExecutionPhenotype(tuple(case.identity for case in panel.cases), rows)
        effect = StrategicEffectPhenotype(panel.cases, rows, tuple(baseline_rows))
        semantic_fault = next((row.fault.name for row in rows if int(row.fault) != 0), None)
        failures = list(_screening_failures(screening))
        if semantic_fault is not None:
            failures.append(FailureReason.SEMANTIC_FAULT)
        if semantic_mismatches[index]:
            failures.append(FailureReason.SEMANTIC_MISMATCH)
        record = EvaluationRecord(
            program=program, context_id=context_id, generation=generation,
            parent_id=parents[index], lineage=tuple(lineages[index]), proposal=proposals[index],
            execution_phenotype=execution.to_document(), strategic_effect=effect.to_document(),
            screening=screening, failure_reasons=tuple(failures),
            semantic_fault=semantic_fault,
            semantic_mismatch=bool(semantic_mismatches[index]),
            stepping_stone=bool(stepping_stones[index]),
        )
        inserted.append(archive.insert(record))
    return tuple(inserted)
