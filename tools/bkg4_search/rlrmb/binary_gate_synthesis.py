"""Phase 1.2 — Binary SPECIALIST-vs-PARENT gate synthesis.

Synthesizes bounded deterministic gates for admitted BINARY_ROUTE_CANDIDATE
specialists.  Each gate routes to the specialist only in TARGET capability
states and defaults to PARENT everywhere else, ensuring all observed
protected BREAK states are excluded.

Reuses the existing ConditionGene/DNF grammar and contrastive synthesis
machinery from macro_initiation_cegis.  Does NOT perform routed CUDA
evaluation (Phase 1.3) or commit binary specialists.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence

from tools.bkg4_search.edit_search.goals import SearchGoal, synthesize_contrastive_conditions
from tools.bkg4_search.macro_initiation_cegis import (
    ConditionGene,
    GateScore,
    MacroEffectCase,
    _contexts as _macro_contexts,
    condition_hits,
    evaluate_gate,
    gate_hits,
    selected_mixed_exact_cells,
)
from tools.bkg4_search.module_codec import COMPARATORS, FEATURES

from .binary_admission import (
    AdmissionResult,
    BinaryAdmissionEvidence,
    CandidateAdmissionClass,
)
from .diagnostics import ImperfectCandidateRecord, PerCaseEffect
from .schema import canonical_json, stable_identity

GATE_SYNTHESIS_SCHEMA_VERSION = 1

DEFAULT_MAX_CLAUSES = 3
DEFAULT_MAX_CONDITIONS_PER_CLAUSE = 4
DEFAULT_MAX_PREDICATES = 128
DEFAULT_MAX_SYNTHESIS_CANDIDATES = 256
DEFAULT_MIN_CAPABILITY_PRESERVED = 1
DEFAULT_THRESHOLD_RADIUS = 2
DEFAULT_FEATURE_BUDGET = 128


class GateSynthesisStatus(str, Enum):
    GATE_SYNTHESIZED = "GATE_SYNTHESIZED"
    TARGET_CAPABILITY_LOST = "TARGET_CAPABILITY_LOST"
    NO_SAFE_SEPARATOR = "NO_SAFE_SEPARATOR"
    GATE_LIMITED = "GATE_LIMITED"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


@dataclass(frozen=True)
class BinaryGateCase:
    """One case for binary gate synthesis with legal context."""
    case_id: str
    role: str  # "TARGET" or "PROTECTED"
    baseline_margin: int
    candidate_margin: int
    delta: int
    seat: int
    family: str
    is_target_positive: bool
    is_protected_break: bool
    legal_context: Mapping[tuple[str, int], int]

    def to_macro_effect_case(self) -> MacroEffectCase:
        return MacroEffectCase(
            case_id=self.case_id,
            baseline_margin=self.baseline_margin,
            macro_margin=self.candidate_margin,
            context=dict(self.legal_context),
            exact_signature=(),
            support_group=self.family,
            dataset_role="target_capability" if self.is_target_positive else "validation",
            required_target_positive=self.is_target_positive,
            required_target_l2w=(
                self.is_target_positive
                and self.baseline_margin < 0 < self.candidate_margin
            ),
        )

    def to_document(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "role": self.role,
            "baseline_margin": self.baseline_margin,
            "candidate_margin": self.candidate_margin,
            "delta": self.delta,
            "seat": self.seat,
            "family": self.family,
            "is_target_positive": self.is_target_positive,
            "is_protected_break": self.is_protected_break,
            "legal_context": {
                f"{f}:{i}": v for (f, i), v in sorted(self.legal_context.items())
            },
        }


def _gate_canonical(
    clauses: Sequence[Sequence[ConditionGene]],
) -> list[list[dict[str, Any]]]:
    result = []
    for clause in clauses:
        conds = sorted(clause, key=lambda c: (c.feature, c.index, c.cmp, c.value))
        result.append([
            {"feature": c.feature, "cmp": c.cmp, "index": c.index, "value": c.value}
            for c in conds
        ])
    result.sort(key=lambda cl: json.dumps(cl, sort_keys=True))
    return result


def _gate_identity(clauses: Sequence[Sequence[ConditionGene]]) -> str:
    canonical = _gate_canonical(clauses)
    return stable_identity(canonical_json(canonical))


@dataclass(frozen=True)
class GateSynthesisResult:
    """Result of binary gate synthesis for one candidate."""
    candidate_semantic_id: str
    parent_generation_context: str
    status: GateSynthesisStatus
    gate: tuple[tuple[ConditionGene, ...], ...] | None
    gate_canonical: list[list[dict[str, Any]]] | None
    gate_identity: str | None
    gate_complexity: int
    target_positives_total: int
    target_positives_matched: int
    target_positives_missed: int
    protected_breaks_total: int
    protected_breaks_matched: int
    protected_breaks_excluded: int
    gate_features_used: tuple[str, ...]
    failure_reason: str | None
    gate_score: GateScore | None
    source_admission_identity: str

    def to_document(self) -> dict[str, Any]:
        return {
            "schema_version": GATE_SYNTHESIS_SCHEMA_VERSION,
            "candidate_semantic_id": self.candidate_semantic_id,
            "parent_generation_context": self.parent_generation_context,
            "status": self.status.value,
            "gate": self.gate_canonical,
            "gate_identity": self.gate_identity,
            "gate_complexity": self.gate_complexity,
            "target_positives_total": self.target_positives_total,
            "target_positives_matched": self.target_positives_matched,
            "target_positives_missed": self.target_positives_missed,
            "protected_breaks_total": self.protected_breaks_total,
            "protected_breaks_matched": self.protected_breaks_matched,
            "protected_breaks_excluded": self.protected_breaks_excluded,
            "gate_features_used": list(self.gate_features_used),
            "failure_reason": self.failure_reason,
            "gate_score": None if self.gate_score is None else {
                "activated": self.gate_score.activated,
                "positive": self.gate_score.positive,
                "harmful": self.gate_score.harmful,
                "win_to_loss": self.gate_score.win_to_loss,
                "loss_to_win": self.gate_score.loss_to_win,
                "target_positive": self.gate_score.target_positive,
                "target_loss_to_win": self.gate_score.target_loss_to_win,
                "complexity": self.gate_score.complexity,
            },
            "source_admission_identity": self.source_admission_identity,
        }

    @classmethod
    def from_document(cls, doc: Mapping[str, Any]) -> "GateSynthesisResult":
        if doc.get("schema_version") != GATE_SYNTHESIS_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported gate synthesis schema: {doc.get('schema_version')!r}"
            )
        gate_clauses = None
        if doc["gate"] is not None:
            gate_clauses = tuple(
                tuple(
                    ConditionGene(
                        feature=c["feature"], cmp=c["cmp"],
                        index=c["index"], value=c["value"],
                    )
                    for c in clause
                )
                for clause in doc["gate"]
            )
        score = None
        if doc.get("gate_score") is not None:
            gs = doc["gate_score"]
            score = GateScore(
                activated=gs["activated"], positive=gs["positive"],
                neutral=0, harmful=gs["harmful"],
                loss_to_win=gs["loss_to_win"], win_to_loss=gs["win_to_loss"],
                worst_delta=0, aggregate_delta=0, support_groups=0,
                complexity=gs["complexity"],
                target_positive=gs["target_positive"],
                target_loss_to_win=gs["target_loss_to_win"],
            )
        return cls(
            candidate_semantic_id=doc["candidate_semantic_id"],
            parent_generation_context=doc["parent_generation_context"],
            status=GateSynthesisStatus(doc["status"]),
            gate=gate_clauses,
            gate_canonical=doc["gate"],
            gate_identity=doc.get("gate_identity"),
            gate_complexity=doc["gate_complexity"],
            target_positives_total=doc["target_positives_total"],
            target_positives_matched=doc["target_positives_matched"],
            target_positives_missed=doc["target_positives_missed"],
            protected_breaks_total=doc["protected_breaks_total"],
            protected_breaks_matched=doc["protected_breaks_matched"],
            protected_breaks_excluded=doc["protected_breaks_excluded"],
            gate_features_used=tuple(doc["gate_features_used"]),
            failure_reason=doc.get("failure_reason"),
            gate_score=score,
            source_admission_identity=doc["source_admission_identity"],
        )


def _enumerate_binary_conjunctions(
    cases: Sequence[MacroEffectCase],
    *,
    max_conditions: int = 4,
    limit: int = 256,
    feature_budget: int = 128,
) -> list[tuple[tuple[ConditionGene, ...], GateScore]]:
    """Enumerate harm-excluding conjunctions allowing temporal features.

    Unlike enumerate_safe_conjunctions, this allows STEP/DAY/HOUR since
    binary gate synthesis at specialist initiation legitimately uses them.
    """
    if not cases:
        return []
    positive = [x.case_id for x in cases if x.delta > 0]
    harmful = [x.case_id for x in cases if x.delta < 0]
    l2w = [x.case_id for x in cases if x.loss_to_win]
    required_target_positive = [
        x.case_id for x in cases if x.required_target_positive and x.delta > 0
    ]
    required_target_l2w = [
        x.case_id for x in cases if x.required_target_l2w and x.loss_to_win
    ]
    if not positive or not harmful:
        return []
    contexts = _macro_contexts(cases)
    target_sets: list[tuple[str, ...]] = []
    for target_ids in (required_target_l2w, required_target_positive):
        target_tuple = tuple(target_ids)
        if target_tuple and target_tuple not in target_sets:
            target_sets.append(target_tuple)
    target_sets.append(tuple(positive))
    if l2w:
        target_sets.append(tuple(l2w))
        target_sets.extend((case_id,) for case_id in l2w)
    candidates: dict[tuple[ConditionGene, ...], GateScore] = {}
    for target_ids in target_sets:
        goal = SearchGoal(tuple(target_ids), tuple(harmful), (), tuple(harmful))
        rows = synthesize_contrastive_conditions(
            goal,
            contexts,
            max_conditions=max_conditions,
            limit=limit,
            feature_budget=feature_budget,
            guard_excluding_only=True,
            allow_temporal=True,
        )
        for conditions in rows:
            score = evaluate_gate((conditions,), cases)
            if score.harmful or score.win_to_loss:
                continue
            if selected_mixed_exact_cells((conditions,), cases):
                continue
            old = candidates.get(conditions)
            if old is None or score.lexicographic_key() < old.lexicographic_key():
                candidates[conditions] = score
    return sorted(candidates.items(), key=lambda row: row[1].lexicographic_key())


def _failure_result(
    candidate_id: str,
    parent_ctx: str,
    status: GateSynthesisStatus,
    reason: str,
    target_total: int,
    breaks_total: int,
    admission_id: str,
) -> GateSynthesisResult:
    return GateSynthesisResult(
        candidate_semantic_id=candidate_id,
        parent_generation_context=parent_ctx,
        status=status,
        gate=None,
        gate_canonical=None,
        gate_identity=None,
        gate_complexity=0,
        target_positives_total=target_total,
        target_positives_matched=0,
        target_positives_missed=target_total,
        protected_breaks_total=breaks_total,
        protected_breaks_matched=0,
        protected_breaks_excluded=breaks_total,
        gate_features_used=(),
        failure_reason=reason,
        gate_score=None,
        source_admission_identity=admission_id,
    )


def _extract_features_used(
    clauses: Sequence[Sequence[ConditionGene]],
) -> tuple[str, ...]:
    features: set[str] = set()
    for clause in clauses:
        for cond in clause:
            features.add(cond.feature)
    return tuple(sorted(features))


def _binary_gate_sort_key(
    score: GateScore,
    gate: tuple[tuple[ConditionGene, ...], ...],
    target_positives_matched: int,
    target_positives_total: int,
) -> tuple[Any, ...]:
    return (
        score.win_to_loss,
        score.harmful,
        -(target_positives_matched),
        -score.target_loss_to_win,
        -score.target_positive,
        -score.support_groups,
        score.complexity,
        -score.aggregate_delta,
        json.dumps(_gate_canonical(gate), sort_keys=True),
    )


def synthesize_binary_gate(
    candidate_id: str,
    cases: Sequence[BinaryGateCase],
    *,
    parent_generation_context: str,
    source_admission_identity: str = "",
    max_clauses: int = DEFAULT_MAX_CLAUSES,
    max_conditions: int = DEFAULT_MAX_CONDITIONS_PER_CLAUSE,
    max_synthesis_candidates: int = DEFAULT_MAX_SYNTHESIS_CANDIDATES,
    min_capability_preserved: int = DEFAULT_MIN_CAPABILITY_PRESERVED,
    threshold_radius: int = DEFAULT_THRESHOLD_RADIUS,
    feature_budget: int = DEFAULT_FEATURE_BUDGET,
) -> GateSynthesisResult:
    """Synthesize a bounded deterministic SPECIALIST-vs-PARENT gate.

    Returns GATE_SYNTHESIZED only when:
    - zero protected BREAK states match the gate
    - at least min_capability_preserved TARGET positive states match
    """
    if max_clauses < 1:
        raise ValueError("max_clauses must be positive")
    if max_conditions < 1:
        raise ValueError("max_conditions must be positive")

    target_positives = [c for c in cases if c.is_target_positive]
    protected_breaks = [c for c in cases if c.is_protected_break]
    admission_id = source_admission_identity or candidate_id

    if not target_positives:
        return _failure_result(
            candidate_id, parent_generation_context,
            GateSynthesisStatus.INSUFFICIENT_EVIDENCE,
            "no_target_positive_cases",
            0, len(protected_breaks), admission_id,
        )

    if not protected_breaks:
        return _failure_result(
            candidate_id, parent_generation_context,
            GateSynthesisStatus.INSUFFICIENT_EVIDENCE,
            "no_protected_break_cases",
            len(target_positives), 0, admission_id,
        )

    macro_cases = [c.to_macro_effect_case() for c in cases]
    all_positives = [c.to_macro_effect_case() for c in target_positives]
    all_breaks = [c.to_macro_effect_case() for c in protected_breaks]

    best_gate: tuple[tuple[ConditionGene, ...], ...] | None = None
    best_score: GateScore | None = None
    best_target_matched = 0

    ranked_conjunctions = _enumerate_binary_conjunctions(
        macro_cases,
        max_conditions=max_conditions,
        limit=max_synthesis_candidates,
        feature_budget=feature_budget,
    )

    for clause, score in ranked_conjunctions:
        gate = (clause,)
        break_matched = sum(
            1 for b in all_breaks if gate_hits(gate, b.context)
        )
        if break_matched > 0:
            continue
        target_matched = sum(
            1 for p in all_positives if gate_hits(gate, p.context)
        )
        if target_matched == 0:
            continue

        if best_gate is None or _binary_gate_sort_key(
            score, gate, target_matched, len(target_positives)
        ) < _binary_gate_sort_key(
            best_score, best_gate, best_target_matched, len(target_positives)
        ):
            best_gate = gate
            best_score = score
            best_target_matched = target_matched

    if best_gate is None and max_clauses > 1:
        best_gate, best_score, best_target_matched = _try_dnf_synthesis(
            all_positives, all_breaks, macro_cases,
            max_clauses=max_clauses,
            max_conditions=max_conditions,
            max_synthesis_candidates=max_synthesis_candidates,
            feature_budget=feature_budget,
        )

    if best_gate is not None and best_target_matched == 0:
        return _failure_result(
            candidate_id, parent_generation_context,
            GateSynthesisStatus.TARGET_CAPABILITY_LOST,
            "gate_excludes_all_target_positives",
            len(target_positives), len(protected_breaks), admission_id,
        )

    if best_gate is None:
        has_legal_evidence = any(c.legal_context for c in cases)
        if has_legal_evidence:
            legal_features = set()
            for c in cases:
                for (f, _), _ in c.legal_context.items():
                    legal_features.add(f)
            gate_features = frozenset(FEATURES)
            non_gate_legal = legal_features - gate_features
            if non_gate_legal:
                return _failure_result(
                    candidate_id, parent_generation_context,
                    GateSynthesisStatus.GATE_LIMITED,
                    "legal_separator_exists_but_gate_grammar_insufficient",
                    len(target_positives), len(protected_breaks), admission_id,
                )
        return _failure_result(
            candidate_id, parent_generation_context,
            GateSynthesisStatus.NO_SAFE_SEPARATOR,
            "no_harm_free_gate_found",
            len(target_positives), len(protected_breaks), admission_id,
        )

    if best_target_matched < min_capability_preserved:
        return _failure_result(
            candidate_id, parent_generation_context,
            GateSynthesisStatus.TARGET_CAPABILITY_LOST,
            f"target_preserved_{best_target_matched}_below_minimum_{min_capability_preserved}",
            len(target_positives), len(protected_breaks), admission_id,
        )

    break_matched = sum(
        1 for b in all_breaks if gate_hits(best_gate, b.context)
    )

    if best_gate is not None and threshold_radius > 0:
        best_gate = _select_robust_threshold(
            best_gate, all_positives, all_breaks, macro_cases,
            radius=threshold_radius,
        )
        best_score = evaluate_gate(best_gate, macro_cases)
        best_target_matched = sum(
            1 for p in all_positives if gate_hits(best_gate, p.context)
        )
        break_matched = sum(
            1 for b in all_breaks if gate_hits(best_gate, b.context)
        )
        if break_matched > 0:
            return _failure_result(
                candidate_id, parent_generation_context,
                GateSynthesisStatus.NO_SAFE_SEPARATOR,
                "threshold_perturbation_exposed_break",
                len(target_positives), len(protected_breaks), admission_id,
            )

    return GateSynthesisResult(
        candidate_semantic_id=candidate_id,
        parent_generation_context=parent_generation_context,
        status=GateSynthesisStatus.GATE_SYNTHESIZED,
        gate=best_gate,
        gate_canonical=_gate_canonical(best_gate),
        gate_identity=_gate_identity(best_gate),
        gate_complexity=sum(len(cl) for cl in best_gate),
        target_positives_total=len(target_positives),
        target_positives_matched=best_target_matched,
        target_positives_missed=len(target_positives) - best_target_matched,
        protected_breaks_total=len(protected_breaks),
        protected_breaks_matched=break_matched,
        protected_breaks_excluded=len(protected_breaks) - break_matched,
        gate_features_used=_extract_features_used(best_gate),
        failure_reason=None,
        gate_score=best_score,
        source_admission_identity=admission_id,
    )


def _try_dnf_synthesis(
    positives: Sequence[MacroEffectCase],
    breaks: Sequence[MacroEffectCase],
    all_cases: Sequence[MacroEffectCase],
    *,
    max_clauses: int,
    max_conditions: int,
    max_synthesis_candidates: int,
    feature_budget: int,
) -> tuple[
    tuple[tuple[ConditionGene, ...], ...] | None,
    GateScore | None,
    int,
]:
    """Attempt bounded DNF synthesis by adding clauses for missed positives."""
    covered_positives: set[str] = set()
    clauses: list[tuple[ConditionGene, ...]] = []

    for _ in range(max_clauses):
        uncovered = [
            p for p in positives if p.case_id not in covered_positives
        ]
        if not uncovered:
            break

        sub_cases = list(uncovered) + list(breaks)
        ranked = _enumerate_binary_conjunctions(
            sub_cases,
            max_conditions=max_conditions,
            limit=max_synthesis_candidates,
            feature_budget=feature_budget,
        )
        found = False
        for clause, _ in ranked:
            break_hit = any(gate_hits((clause,), b.context) for b in breaks)
            if break_hit:
                continue
            target_hit = [p for p in uncovered if gate_hits((clause,), p.context)]
            if not target_hit:
                continue
            clauses.append(clause)
            covered_positives.update(p.case_id for p in target_hit)
            found = True
            break
        if not found:
            break

    if not clauses:
        return None, None, 0

    gate = tuple(clauses)
    break_matched = sum(1 for b in breaks if gate_hits(gate, b.context))
    if break_matched > 0:
        return None, None, 0

    target_matched = sum(1 for p in positives if gate_hits(gate, p.context))
    if target_matched == 0:
        return None, None, 0

    score = evaluate_gate(gate, list(all_cases))
    return gate, score, target_matched


def _select_robust_threshold(
    gate: tuple[tuple[ConditionGene, ...], ...],
    positives: Sequence[MacroEffectCase],
    breaks: Sequence[MacroEffectCase],
    all_cases: Sequence[MacroEffectCase],
    *,
    radius: int,
) -> tuple[tuple[ConditionGene, ...], ...]:
    """Try nearby threshold values for each condition, preferring stable separators."""
    best_gate = gate
    best_score = evaluate_gate(gate, list(all_cases))
    best_target = sum(1 for p in positives if gate_hits(gate, p.context))
    best_key = _binary_gate_sort_key(best_score, gate, best_target, len(positives))

    for ci, clause in enumerate(gate):
        for cj, cond in enumerate(clause):
            if cond.cmp not in ("LE", "GE", "EQ"):
                continue
            for offset in range(-radius, radius + 1):
                if offset == 0:
                    continue
                new_cond = ConditionGene(
                    feature=cond.feature, cmp=cond.cmp,
                    index=cond.index, value=cond.value + offset,
                )
                try:
                    new_cond.validate()
                except ValueError:
                    continue
                new_clause = clause[:cj] + (new_cond,) + clause[cj + 1:]
                new_gate = gate[:ci] + (new_clause,) + gate[ci + 1:]
                break_hit = any(gate_hits(new_gate, b.context) for b in breaks)
                if break_hit:
                    continue
                target_matched = sum(
                    1 for p in positives if gate_hits(new_gate, p.context)
                )
                if target_matched < best_target:
                    continue
                new_score = evaluate_gate(new_gate, list(all_cases))
                new_key = _binary_gate_sort_key(
                    new_score, new_gate, target_matched, len(positives),
                )
                if new_key < best_key:
                    best_gate = new_gate
                    best_score = new_score
                    best_target = target_matched
                    best_key = new_key

    return best_gate


def synthesize_gates_for_shortlist(
    shortlist: Sequence[tuple[ImperfectCandidateRecord, AdmissionResult]],
    case_contexts: Mapping[str, Mapping[str, Mapping[tuple[str, int], int]]],
    *,
    parent_generation_context: str,
    max_clauses: int = DEFAULT_MAX_CLAUSES,
    max_conditions: int = DEFAULT_MAX_CONDITIONS_PER_CLAUSE,
    min_capability_preserved: int = DEFAULT_MIN_CAPABILITY_PRESERVED,
) -> tuple[GateSynthesisResult, ...]:
    """Synthesize gates for all admitted BINARY_ROUTE_CANDIDATEs.

    case_contexts maps candidate_semantic_id → case_id → legal_context.
    Only candidates with BINARY_ROUTE_CANDIDATE admission are processed.
    """
    results: list[GateSynthesisResult] = []
    for record, admission in shortlist:
        if admission.admission_class is not CandidateAdmissionClass.BINARY_ROUTE_CANDIDATE:
            continue
        candidate_contexts = case_contexts.get(
            record.candidate_semantic_id, {}
        )
        cases = build_gate_cases(record, candidate_contexts)
        result = synthesize_binary_gate(
            record.candidate_semantic_id,
            cases,
            parent_generation_context=parent_generation_context,
            source_admission_identity=record.candidate_semantic_id,
            max_clauses=max_clauses,
            max_conditions=max_conditions,
            min_capability_preserved=min_capability_preserved,
        )
        results.append(result)
    return tuple(results)


def build_gate_cases(
    record: ImperfectCandidateRecord,
    case_contexts: Mapping[str, Mapping[tuple[str, int], int]],
) -> tuple[BinaryGateCase, ...]:
    """Build gate synthesis cases from an ImperfectCandidateRecord.

    case_contexts maps case_id → legal_context for this candidate.
    """
    cases: list[BinaryGateCase] = []
    for effect in record.per_case_effects:
        ctx = case_contexts.get(effect.case_id, {})
        if not ctx:
            continue
        is_target_positive = (
            effect.role == "TARGET" and effect.solves
        )
        is_protected_break = (
            effect.role == "PROTECTED" and effect.breaks
        )
        if not is_target_positive and not is_protected_break:
            continue
        cases.append(BinaryGateCase(
            case_id=effect.case_id,
            role=effect.role,
            baseline_margin=effect.baseline_margin,
            candidate_margin=effect.candidate_margin,
            delta=effect.delta,
            seat=effect.seat,
            family=effect.family,
            is_target_positive=is_target_positive,
            is_protected_break=is_protected_break,
            legal_context=ctx,
        ))
    return tuple(cases)


@dataclass
class GateSynthesisMetrics:
    """Phase 1.2 gate synthesis metrics collector."""
    candidates_attempted: int = 0
    gates_synthesized: int = 0
    target_capability_lost: int = 0
    no_safe_separator: int = 0
    gate_limited: int = 0
    insufficient_evidence: int = 0
    total_target_positives_preserved: int = 0
    total_protected_breaks_excluded: int = 0

    def observe(self, result: GateSynthesisResult) -> None:
        self.candidates_attempted += 1
        status = result.status
        if status is GateSynthesisStatus.GATE_SYNTHESIZED:
            self.gates_synthesized += 1
            self.total_target_positives_preserved += result.target_positives_matched
            self.total_protected_breaks_excluded += result.protected_breaks_excluded
        elif status is GateSynthesisStatus.TARGET_CAPABILITY_LOST:
            self.target_capability_lost += 1
        elif status is GateSynthesisStatus.NO_SAFE_SEPARATOR:
            self.no_safe_separator += 1
        elif status is GateSynthesisStatus.GATE_LIMITED:
            self.gate_limited += 1
        elif status is GateSynthesisStatus.INSUFFICIENT_EVIDENCE:
            self.insufficient_evidence += 1

    def to_document(self) -> dict[str, Any]:
        return {
            "candidates_attempted": self.candidates_attempted,
            "gates_synthesized": self.gates_synthesized,
            "target_capability_lost": self.target_capability_lost,
            "no_safe_separator": self.no_safe_separator,
            "gate_limited": self.gate_limited,
            "insufficient_evidence": self.insufficient_evidence,
            "total_target_positives_preserved": self.total_target_positives_preserved,
            "total_protected_breaks_excluded": self.total_protected_breaks_excluded,
        }


def write_gate_synthesis_artifact(
    output_dir: Path,
    results: Sequence[GateSynthesisResult],
    metrics: GateSynthesisMetrics,
    *,
    parent_generation_context: str,
) -> dict[str, str]:
    """Persist Phase-1.2 gate synthesis artifacts and return path manifest."""
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, str] = {}

    synthesized = [r for r in results if r.status is GateSynthesisStatus.GATE_SYNTHESIZED]
    failed = [r for r in results if r.status is not GateSynthesisStatus.GATE_SYNTHESIZED]

    summary = {
        "schema_version": GATE_SYNTHESIS_SCHEMA_VERSION,
        "phase": "1.2",
        "label": "BINARY_GATE_SYNTHESIS",
        "parent_generation_context": parent_generation_context,
        "metrics": metrics.to_document(),
        "synthesized_count": len(synthesized),
        "failed_count": len(failed),
        "synthesized_candidates": [r.candidate_semantic_id for r in synthesized],
        "failed_candidates": [
            {"candidate": r.candidate_semantic_id, "status": r.status.value,
             "reason": r.failure_reason}
            for r in failed
        ],
        "status_distribution": {
            s.value: sum(1 for r in results if r.status is s)
            for s in GateSynthesisStatus
        },
    }
    summary_path = output_dir / "gate_synthesis_summary.json"
    summary_path.write_text(
        json.dumps(summary, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    manifest["gate_synthesis_summary"] = str(summary_path)

    if synthesized:
        candidates_doc = [r.to_document() for r in synthesized]
        candidates_path = output_dir / "binary_gate_candidates.json"
        candidates_path.write_text(
            json.dumps(candidates_doc, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        manifest["binary_gate_candidates"] = str(candidates_path)

    if results:
        all_doc = [r.to_document() for r in results]
        all_path = output_dir / "all_gate_results.json"
        all_path.write_text(
            json.dumps(all_doc, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        manifest["all_gate_results"] = str(all_path)

    return manifest


def record_gate_synthesis_to_memory(
    memory: Any,
    context_id: str,
    generation: int,
    result: GateSynthesisResult,
) -> None:
    """Record gate synthesis result in SearchMemory search_feedback."""
    if memory is None:
        return
    try:
        memory.record_search_feedback(
            result.candidate_semantic_id,
            context_id,
            generation=generation,
            stage="BINARY_GATE_SYNTHESIS",
            failure_class=(
                None if result.status is GateSynthesisStatus.GATE_SYNTHESIZED
                else result.status.value
            ),
        )
    except (ValueError, Exception):
        pass
