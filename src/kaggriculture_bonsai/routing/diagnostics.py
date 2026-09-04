"""Phase-0 imperfect-specialist routing diagnostics.

Observational instrumentation only.  Does not change production acceptance,
promotion, CEGIS, routability, or any safety semantics.  All outputs are
clearly labelled as diagnostic evidence, never as composition-safe or
acceptance-authoritative.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence

from kaggriculture_bonsai.composition.module_codec import FEATURES as GATE_GRAMMAR_FEATURES

from kaggriculture_bonsai.search.effects import EffectBlock, EffectVector
from kaggriculture_bonsai.search.schema import DatasetRole, canonical_json, stable_identity

DIAGNOSTICS_SCHEMA_VERSION = 2

_GATE_FEATURE_SET = frozenset(GATE_GRAMMAR_FEATURES)


# ── routability classification with separated legal/gate contexts ────────────

class DiagnosticRoutabilityClass(str, Enum):
    ROUTABLE = "ROUTABLE"
    GATE_LIMITATION = "GATE_LIMITATION"
    FUTURE_DEPENDENT = "FUTURE_DEPENDENT"
    UNKNOWN = "UNKNOWN"


class BinaryOpportunityClass(str, Enum):
    BINARY_PARENT_PLAUSIBLE = "BINARY_PARENT_PLAUSIBLE"
    GATE_LIMITED = "GATE_LIMITED"
    DEFERRED_DECISION_PLAUSIBLE = "DEFERRED_DECISION_PLAUSIBLE"
    TRUE_FUTURE_DEPENDENT = "TRUE_FUTURE_DEPENDENT"
    UNKNOWN = "UNKNOWN"


def classify_binary_opportunity(routability_doc: Mapping[str, Any]) -> BinaryOpportunityClass:
    """Map exact useful-vs-protected routing evidence to an architecture hint."""
    classification = routability_doc.get("classification", "")
    diagnostic = classify_diagnostic_routability(routability_doc)
    if diagnostic is DiagnosticRoutabilityClass.ROUTABLE:
        return BinaryOpportunityClass.BINARY_PARENT_PLAUSIBLE
    if classification in {
        "REPAIR_LATER_CHECKPOINT", "REPAIR_COMMON_PREFIX",
        "REPAIR_BRANCH_AT_REVEAL", "REPAIR_ABORT_RECOVERY",
    }:
        return BinaryOpportunityClass.DEFERRED_DECISION_PLAUSIBLE
    if diagnostic is DiagnosticRoutabilityClass.GATE_LIMITATION:
        return BinaryOpportunityClass.GATE_LIMITED
    if diagnostic is DiagnosticRoutabilityClass.FUTURE_DEPENDENT:
        return BinaryOpportunityClass.TRUE_FUTURE_DEPENDENT
    return BinaryOpportunityClass.UNKNOWN


def classify_diagnostic_routability(
    routability_doc: Mapping[str, Any],
) -> DiagnosticRoutabilityClass:
    """Re-classify using the separated legal/gate evidence in the routability result.

    Uses the diagnostics already computed by screen_routability with the now-
    separated legal_context (full legal observability) and gate_context
    (deployable ConditionGene grammar).
    """
    classification = routability_doc.get("classification", "")
    diagnostics = routability_doc.get("diagnostics", {})
    legal_pred = diagnostics.get("legal_predicate")
    gate_pred = diagnostics.get("gate_predicate")

    if classification == "ROUTABLE_CANDIDATE":
        return DiagnosticRoutabilityClass.ROUTABLE
    if classification == "FUTURE_DEPENDENT":
        return DiagnosticRoutabilityClass.FUTURE_DEPENDENT
    if classification == "GATE_LIMITATION_SUSPECTED":
        if legal_pred and not gate_pred:
            return DiagnosticRoutabilityClass.GATE_LIMITATION
        return DiagnosticRoutabilityClass.UNKNOWN
    if legal_pred and not gate_pred:
        return DiagnosticRoutabilityClass.GATE_LIMITATION
    if not legal_pred and not gate_pred:
        return DiagnosticRoutabilityClass.FUTURE_DEPENDENT
    return DiagnosticRoutabilityClass.UNKNOWN


# ── per-case effect record for SOLVES/BREAKS reconstruction ──────────────────

@dataclass(frozen=True)
class PerCaseEffect:
    case_id: str
    role: str
    baseline_margin: int
    candidate_margin: int
    delta: int
    seat: int
    family: str

    @property
    def solves(self) -> bool:
        return self.baseline_margin < 0 < self.candidate_margin

    @property
    def breaks(self) -> bool:
        return self.baseline_margin > 0 > self.candidate_margin

    @property
    def positive(self) -> bool:
        return self.delta > 0

    @property
    def harmful(self) -> bool:
        return self.delta < 0

    @property
    def preserves_win(self) -> bool:
        return self.baseline_margin > 0 and self.candidate_margin > 0

    def to_document(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "role": self.role,
            "baseline_margin": self.baseline_margin,
            "candidate_margin": self.candidate_margin,
            "delta": self.delta,
            "seat": self.seat,
            "family": self.family,
        }


def per_case_effects_from_evidence(rows: Sequence[Any]) -> tuple[PerCaseEffect, ...]:
    """Extract compact per-case effects from ExactCaseEvidence sequences."""
    return tuple(
        PerCaseEffect(
            case_id=row.meta.case_id,
            role=row.role.value,
            baseline_margin=row.baseline_margin,
            candidate_margin=row.candidate_margin,
            delta=row.delta,
            seat=row.meta.seat,
            family=row.meta.family,
        )
        for row in rows
    )


def per_case_effects_from_effect_vector(
    effect: EffectVector, case_ids: Sequence[str], families: Sequence[str],
    seats: Sequence[int],
) -> tuple[PerCaseEffect, ...]:
    """Extract per-case effects from an EffectVector with case metadata."""
    results = []
    for block in effect.blocks:
        for i, (baseline, delta) in enumerate(
            zip(block.baseline_margins, block.margin_deltas)
        ):
            if i < len(case_ids):
                results.append(PerCaseEffect(
                    case_id=case_ids[i],
                    role=block.role.value,
                    baseline_margin=baseline,
                    candidate_margin=baseline + delta,
                    delta=delta,
                    seat=seats[i] if i < len(seats) else 0,
                    family=families[i] if i < len(families) else "",
                ))
    return tuple(results)


def protected_effect_metrics(block: EffectBlock) -> dict[str, int]:
    """Summarize exact protected outcomes without conflating harm and W->L."""
    if not isinstance(block, EffectBlock):
        raise TypeError("protected metrics require an EffectBlock")
    return {
        "protected_win_to_loss": sum(block.win_to_loss),
        "protected_loss_to_win": sum(block.loss_to_win),
        "protected_harm_count": sum(delta < 0 for delta in block.margin_deltas),
        "protected_positive_count": sum(delta > 0 for delta in block.margin_deltas),
        "protected_worst_delta": min(block.margin_deltas),
        "protected_aggregate_margin": sum(block.margin_deltas),
    }


def select_binary_routing_rows(rows: Sequence[Any]) -> tuple[Any, ...]:
    """Keep only useful TARGET solves and actual protected/validation breaks."""
    return tuple(row for row in rows if (
        (row.role is DatasetRole.TARGET and
         row.baseline_margin < 0 < row.candidate_margin)
        or
        (row.role is DatasetRole.VALIDATION and
         row.baseline_margin > 0 > row.candidate_margin)
    ))


# ── imperfect candidate record ───────────────────────────────────────────────

@dataclass(frozen=True)
class ImperfectCandidateRecord:
    candidate_semantic_id: str
    structural_id: str
    family: str
    lane: str
    target_loss_to_win: int
    target_harm_count: int
    target_worst_delta: int
    target_aggregate_margin: int
    protected_win_to_loss: int
    protected_loss_to_win: int
    protected_harm_count: int
    protected_positive_count: int
    protected_worst_delta: int
    protected_aggregate_margin: int
    effect_signature: str
    routability_classification: str
    diagnostic_routability: str
    binary_opportunity: str
    per_case_effects: tuple[PerCaseEffect, ...]
    effect_vector_doc: dict[str, Any] | None
    initiation_step: int
    checkpoint_metadata: dict[str, Any]
    gate_evidence: dict[str, Any]
    program_document: dict[str, Any] | None = None

    def solves_set(self, role: str = "TARGET") -> frozenset[str]:
        return frozenset(
            e.case_id for e in self.per_case_effects
            if e.role == role and e.solves
        )

    def breaks_set(self, role: str = "TARGET") -> frozenset[str]:
        return frozenset(
            e.case_id for e in self.per_case_effects
            if e.role == role and e.breaks
        )

    def to_document(self) -> dict[str, Any]:
        return {
            "schema_version": DIAGNOSTICS_SCHEMA_VERSION,
            "candidate_semantic_id": self.candidate_semantic_id,
            "structural_id": self.structural_id,
            "family": self.family,
            "lane": self.lane,
            "target_loss_to_win": self.target_loss_to_win,
            "target_harm_count": self.target_harm_count,
            "target_worst_delta": self.target_worst_delta,
            "target_aggregate_margin": self.target_aggregate_margin,
            "protected_win_to_loss": self.protected_win_to_loss,
            "protected_loss_to_win": self.protected_loss_to_win,
            "protected_harm_count": self.protected_harm_count,
            "protected_positive_count": self.protected_positive_count,
            "protected_worst_delta": self.protected_worst_delta,
            "protected_aggregate_margin": self.protected_aggregate_margin,
            "effect_signature": self.effect_signature,
            "routability_classification": self.routability_classification,
            "diagnostic_routability": self.diagnostic_routability,
            "binary_opportunity": self.binary_opportunity,
            "per_case_effects": [e.to_document() for e in self.per_case_effects],
            "effect_vector": self.effect_vector_doc,
            "initiation_step": self.initiation_step,
            "checkpoint_metadata": self.checkpoint_metadata,
            "gate_evidence": self.gate_evidence,
            "program_document": self.program_document,
        }

    @classmethod
    def from_document(cls, doc: Mapping[str, Any]) -> "ImperfectCandidateRecord":
        if doc.get("schema_version") != DIAGNOSTICS_SCHEMA_VERSION:
            raise ValueError(f"unsupported diagnostics schema: {doc.get('schema_version')!r}")
        return cls(
            candidate_semantic_id=doc["candidate_semantic_id"],
            structural_id=doc["structural_id"],
            family=doc["family"],
            lane=doc["lane"],
            target_loss_to_win=doc["target_loss_to_win"],
            target_harm_count=doc["target_harm_count"],
            target_worst_delta=doc["target_worst_delta"],
            target_aggregate_margin=doc["target_aggregate_margin"],
            protected_win_to_loss=doc["protected_win_to_loss"],
            protected_loss_to_win=doc["protected_loss_to_win"],
            protected_harm_count=doc["protected_harm_count"],
            protected_positive_count=doc["protected_positive_count"],
            protected_worst_delta=doc["protected_worst_delta"],
            protected_aggregate_margin=doc["protected_aggregate_margin"],
            effect_signature=doc["effect_signature"],
            routability_classification=doc["routability_classification"],
            diagnostic_routability=doc["diagnostic_routability"],
            binary_opportunity=doc["binary_opportunity"],
            per_case_effects=tuple(
                PerCaseEffect(**e) for e in doc["per_case_effects"]
            ),
            effect_vector_doc=doc.get("effect_vector"),
            initiation_step=doc["initiation_step"],
            checkpoint_metadata=doc["checkpoint_metadata"],
            gate_evidence=doc["gate_evidence"],
            program_document=doc.get("program_document"),
        )


# ── diagnostic collector ────────────────────────────────────────────────────

class ImperfectSpecialistDiagnostics:
    """Collects imperfect-specialist diagnostic records during a campaign."""

    def __init__(self) -> None:
        self._records: list[ImperfectCandidateRecord] = []
        self._total_candidates_evaluated = 0

    def observe_population(self, count: int) -> None:
        if type(count) is not int or count < 0:
            raise ValueError("evaluated candidate count must be non-negative")
        self._total_candidates_evaluated += count

    @property
    def total_candidates_evaluated(self) -> int:
        return self._total_candidates_evaluated

    def record(self, entry: ImperfectCandidateRecord) -> None:
        if not isinstance(entry, ImperfectCandidateRecord):
            raise TypeError("entry must be ImperfectCandidateRecord")
        self._records.append(entry)

    @property
    def records(self) -> tuple[ImperfectCandidateRecord, ...]:
        return tuple(self._records)

    def serialize(self) -> str:
        """Deterministic JSONL for compact per-record persistence."""
        lines = []
        for record in sorted(self._records, key=lambda r: r.candidate_semantic_id):
            lines.append(canonical_json(record.to_document()))
        return "\n".join(lines) + "\n" if lines else ""

    @classmethod
    def deserialize(cls, payload: str) -> "ImperfectSpecialistDiagnostics":
        diag = cls()
        for line in payload.strip().splitlines():
            if line.strip():
                diag.record(ImperfectCandidateRecord.from_document(json.loads(line)))
        return diag

    def persist(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.serialize(), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "ImperfectSpecialistDiagnostics":
        return cls.deserialize(path.read_text(encoding="utf-8"))


def select_diagnostic_shortlist(
    candidates: Sequence[Any], *, limit: int,
) -> tuple[Any, ...]:
    """Select useful pre-promotion candidates deterministically and within cap.

    Callers provide compact candidate views with the named diagnostic fields.
    Diversity is rewarded after target quality, without weakening the hard cap.
    """
    if type(limit) is not int or limit <= 0:
        raise ValueError("diagnostic shortlist limit must be positive")
    unique: dict[str, Any] = {}
    for item in candidates:
        if item.target_loss_to_win <= 0:
            continue
        current = unique.get(item.candidate_semantic_id)
        breadth = (
            -int(getattr(item, "cross_regime_supported", False)),
            -int(getattr(item, "seed_support", 0)),
            -int(getattr(item, "seat_support", 0)),
            -int(getattr(item, "member_support", 0)),
        )
        quality = (*breadth, -item.target_loss_to_win, item.target_harm_count,
                   -item.target_worst_delta, -item.target_aggregate_margin,
                   item.candidate_semantic_id)
        if current is None:
            unique[item.candidate_semantic_id] = item
        else:
            old = (
                -int(getattr(current, "cross_regime_supported", False)),
                -int(getattr(current, "seed_support", 0)),
                -int(getattr(current, "seat_support", 0)),
                -int(getattr(current, "member_support", 0)),
                -current.target_loss_to_win, current.target_harm_count,
                   -current.target_worst_delta, -current.target_aggregate_margin,
                   current.candidate_semantic_id)
            if quality < old:
                unique[item.candidate_semantic_id] = item
    remaining = list(unique.values())
    selected: list[Any] = []
    seen_effects: set[str] = set()
    seen_structures: set[str] = set()
    seen_epochs: set[tuple[str, int]] = set()
    while remaining and len(selected) < limit:
        def key(item: Any) -> tuple[Any, ...]:
            epoch = (item.lane, item.initiation_step // 24)
            novelty = ((item.effect_signature not in seen_effects) +
                       (item.structural_id not in seen_structures) +
                       (epoch not in seen_epochs))
            return (-int(getattr(item, "cross_regime_supported", False)),
                    -int(getattr(item, "seed_support", 0)),
                    -int(getattr(item, "seat_support", 0)),
                    -int(getattr(item, "member_support", 0)),
                    -item.target_loss_to_win, item.target_harm_count,
                    -item.target_worst_delta, -item.target_aggregate_margin,
                    -novelty, item.lane, item.initiation_step,
                    item.candidate_semantic_id)
        chosen = min(remaining, key=key)
        remaining.remove(chosen)
        selected.append(chosen)
        seen_effects.add(chosen.effect_signature)
        seen_structures.add(chosen.structural_id)
        seen_epochs.add((chosen.lane, chosen.initiation_step // 24))
    return tuple(selected)


# ── aggregate diagnostic summaries ───────────────────────────────────────────

def aggregate_diagnostics(
    records: Sequence[ImperfectCandidateRecord],
    *, total_candidates_evaluated: int | None = None,
) -> dict[str, Any]:
    """Compute distribution and classification summaries."""
    total = len(records)
    l2w_positive = [r for r in records if r.target_loss_to_win > 0]
    l2w_and_w2l = [r for r in l2w_positive if r.protected_win_to_loss > 0]

    # W→L distribution buckets
    w2l_buckets = {"w2l_eq_0": 0, "w2l_eq_1": 0, "w2l_eq_2": 0,
                   "w2l_3_to_5": 0, "w2l_gt_5": 0}
    for r in records:
        w = r.protected_win_to_loss
        if w == 0:
            w2l_buckets["w2l_eq_0"] += 1
        elif w == 1:
            w2l_buckets["w2l_eq_1"] += 1
        elif w == 2:
            w2l_buckets["w2l_eq_2"] += 1
        elif w <= 5:
            w2l_buckets["w2l_3_to_5"] += 1
        else:
            w2l_buckets["w2l_gt_5"] += 1

    # Useful imperfect buckets
    useful = {
        "l2w_ge2_w2l_eq1": sum(
            1 for r in records if r.target_loss_to_win >= 2 and r.protected_win_to_loss == 1
        ),
        "l2w_ge3_w2l_le2": sum(
            1 for r in records if r.target_loss_to_win >= 3 and r.protected_win_to_loss <= 2
        ),
        "l2w_ge5_w2l_le2": sum(
            1 for r in records if r.target_loss_to_win >= 5 and r.protected_win_to_loss <= 2
        ),
        "l2w_gt_w2l": sum(
            1 for r in records if r.target_loss_to_win > r.protected_win_to_loss
        ),
    }

    # Diagnostic routability classes
    routability_classes: dict[str, int] = defaultdict(int)
    for r in records:
        routability_classes[r.diagnostic_routability] += 1
    opportunity_classes: dict[str, int] = defaultdict(int)
    for r in records:
        opportunity_classes[r.binary_opportunity] += 1

    # By family
    by_family: dict[str, int] = defaultdict(int)
    for r in records:
        by_family[r.family] += 1

    # By lane
    by_lane: dict[str, int] = defaultdict(int)
    for r in records:
        by_lane[r.lane] += 1

    return {
        "schema_version": DIAGNOSTICS_SCHEMA_VERSION,
        "protected_panel_role": "VALIDATION_AS_PROTECTED",
        "total_candidates_evaluated": total if total_candidates_evaluated is None else total_candidates_evaluated,
        "diagnostic_shortlist_size": total,
        "candidates_with_l2w_gt_0": len(l2w_positive),
        "candidates_with_l2w_gt_0_and_w2l_gt_0": len(l2w_and_w2l),
        "protected_w2l_distribution": dict(sorted(w2l_buckets.items())),
        "useful_imperfect_buckets": useful,
        "diagnostic_routability_classes": dict(sorted(routability_classes.items())),
        "binary_opportunity_classes": dict(sorted(opportunity_classes.items())),
        "target_l2w_distribution": dict(sorted(Counter(r.target_loss_to_win for r in records).items())),
        "target_harm_distribution": dict(sorted(Counter(r.target_harm_count for r in records).items())),
        "protected_harm_distribution": dict(sorted(Counter(r.protected_harm_count for r in records).items())),
        "protected_worst_delta_distribution": dict(sorted(Counter(r.protected_worst_delta for r in records).items())),
        "by_family": dict(sorted(by_family.items())),
        "by_lane": dict(sorted(by_lane.items())),
    }


def summarize_policy_attribution(
    records: Sequence[ImperfectCandidateRecord],
    *,
    downstream_ready_candidate_ids: frozenset[str] | set[str] = frozenset(),
) -> dict[str, dict[str, int]]:
    """Attribute useful BUILD evidence to fresh policy regimes by case, not identity.

    A candidate can legitimately solve cases from more than one generated policy;
    it is therefore counted once per policy only when it has a TARGET L->W case
    there.  HOLDOUT cases never contribute because only TARGET effects are read.
    """
    rows: dict[str, dict[str, Any]] = {}
    ready = frozenset(downstream_ready_candidate_ids)
    for record in records:
        policies = {
            effect.family
            for effect in record.per_case_effects
            if effect.role == "TARGET" and effect.solves
        }
        for policy in policies:
            row = rows.setdefault(policy, {
                "target_loss_to_win": 0,
                "useful_specialist_ids": set(),
                "routable_ids": set(),
                "future_dependent_ids": set(),
                "downstream_ready_ids": set(),
            })
            row["target_loss_to_win"] += sum(
                effect.solves and effect.role == "TARGET" and effect.family == policy
                for effect in record.per_case_effects
            )
            row["useful_specialist_ids"].add(record.candidate_semantic_id)
            if record.diagnostic_routability == DiagnosticRoutabilityClass.ROUTABLE.value:
                row["routable_ids"].add(record.candidate_semantic_id)
            if record.diagnostic_routability == DiagnosticRoutabilityClass.FUTURE_DEPENDENT.value:
                row["future_dependent_ids"].add(record.candidate_semantic_id)
            if record.candidate_semantic_id in ready:
                row["downstream_ready_ids"].add(record.candidate_semantic_id)
    return {
        policy: {
            "target_loss_to_win": int(row["target_loss_to_win"]),
            "useful_specialists": len(row["useful_specialist_ids"]),
            "routable": len(row["routable_ids"]),
            "future_dependent": len(row["future_dependent_ids"]),
            "downstream_ready": len(row["downstream_ready_ids"]),
        }
        for policy, row in sorted(rows.items())
    }


# ── feature set audit ────────────────────────────────────────────────────────

def audit_feature_sets() -> dict[str, Any]:
    """Exact enumeration of legal, gate-grammar, and routability feature sets."""
    from kaggriculture_bonsai.composition.goals import CONTRASTIVE_FEATURES
    from kaggriculture_bonsai.composition.information_state_io import registry_manifest

    registry = registry_manifest()
    legal_features = sorted({feature for feature, args in registry if len(args) <= 1})
    gate_features = sorted(GATE_GRAMMAR_FEATURES)
    contrastive_features = sorted(CONTRASTIVE_FEATURES)

    legal_set = frozenset(legal_features)
    gate_set = frozenset(gate_features)
    contrastive_set = frozenset(contrastive_features)

    return {
        "schema_version": DIAGNOSTICS_SCHEMA_VERSION,
        "full_legal_observability_features": legal_features,
        "full_legal_observability_count": len(legal_features),
        "deployable_gate_grammar_features": gate_features,
        "deployable_gate_grammar_count": len(gate_features),
        "prior_contrastive_features": contrastive_features,
        "prior_contrastive_count": len(contrastive_features),
        "in_gate_grammar_not_in_prior_contrastive": sorted(gate_set - contrastive_set),
        "in_legal_not_in_gate_grammar": sorted(legal_set - gate_set),
        "in_legal_not_in_prior_contrastive": sorted(legal_set - contrastive_set),
        "in_gate_grammar_not_in_legal": sorted(gate_set - legal_set),
    }


# ── static complementarity diagnostics (offline only) ────────────────────────

@dataclass(frozen=True)
class ComplementarityPair:
    candidate_a: str
    candidate_b: str
    a_protects_b_breaks: frozenset[str]
    b_protects_a_breaks: frozenset[str]
    distinct_target_gains: int
    classification: str

    def to_document(self) -> dict[str, Any]:
        return {
            "candidate_a": self.candidate_a,
            "candidate_b": self.candidate_b,
            "a_protects_b_breaks": sorted(self.a_protects_b_breaks),
            "b_protects_a_breaks": sorted(self.b_protects_a_breaks),
            "a_protects_b_breaks_count": len(self.a_protects_b_breaks),
            "b_protects_a_breaks_count": len(self.b_protects_a_breaks),
            "distinct_target_gains": self.distinct_target_gains,
            "classification": self.classification,
        }


def _compatible_for_comparison(a: ImperfectCandidateRecord, b: ImperfectCandidateRecord) -> bool:
    """Cheap compatibility check using obvious keys."""
    if a.candidate_semantic_id == b.candidate_semantic_id:
        return False
    if a.lane != b.lane:
        return False
    return a.checkpoint_metadata.get("routing_epoch") == b.checkpoint_metadata.get("routing_epoch")


def compute_static_complementarity(
    records: Sequence[ImperfectCandidateRecord],
    *,
    max_pairs: int = 1000,
) -> dict[str, Any]:
    """STATIC COMPLEMENTARITY HEURISTIC — offline diagnostic only.

    Never labelled COMPOSITION_SAFE. Uses actual protected BREAKS and positive
    protected effects, within a deterministic lane/day routing epoch.
    """
    useful = [r for r in records if r.target_loss_to_win > 0]
    pairs: list[ComplementarityPair] = []
    pairs_examined = 0

    for i, a in enumerate(useful):
        if pairs_examined >= max_pairs:
            break
        a_solves = a.solves_set("TARGET")
        a_breaks = a.breaks_set("PROTECTED")
        a_protects = frozenset(e.case_id for e in a.per_case_effects
                               if e.role == "PROTECTED" and e.preserves_win)
        if not a_solves and not a_breaks:
            continue
        for b in useful[i + 1:]:
            if pairs_examined >= max_pairs:
                break
            if not _compatible_for_comparison(a, b):
                continue
            pairs_examined += 1
            b_solves = b.solves_set("TARGET")
            b_breaks = b.breaks_set("PROTECTED")
            b_protects = frozenset(e.case_id for e in b.per_case_effects
                                   if e.role == "PROTECTED" and e.preserves_win)
            if not b_solves and not b_breaks:
                continue
            a_protects_b_breaks = a_protects & b_breaks
            b_protects_a_breaks = b_protects & a_breaks
            if not a_protects_b_breaks and not b_protects_a_breaks:
                continue

            if a_protects_b_breaks and b_protects_a_breaks:
                if a_breaks <= b_protects and b_breaks <= a_protects:
                    classification = "NEAR_CLOSED_PAIR"
                else:
                    classification = "TWO_WAY_COMPLEMENT"
            elif a_protects_b_breaks or b_protects_a_breaks:
                classification = "ONE_WAY_COMPLEMENT"
            else:
                classification = "UNRESOLVED_PAIR"

            pairs.append(ComplementarityPair(
                candidate_a=a.candidate_semantic_id,
                candidate_b=b.candidate_semantic_id,
                a_protects_b_breaks=a_protects_b_breaks,
                b_protects_a_breaks=b_protects_a_breaks,
                distinct_target_gains=len(a_solves ^ b_solves),
                classification=classification,
            ))

    classifications: dict[str, int] = defaultdict(int)
    for p in pairs:
        classifications[p.classification] += 1

    return {
        "schema_version": DIAGNOSTICS_SCHEMA_VERSION,
        "label": "STATIC_COMPLEMENTARITY_HEURISTIC",
        "warning": "NOT_COMPOSITION_SAFE",
        "candidates_analyzed": len(useful),
        "pairs_examined": pairs_examined,
        "complement_pairs": len(pairs),
        "max_pairs_cap": max_pairs,
        "classifications": dict(sorted(classifications.items())),
        "pairs": [p.to_document() for p in sorted(
            pairs, key=lambda p: (p.classification, p.candidate_a, p.candidate_b)
        )],
    }


# ── campaign-level diagnostic artifact writer ────────────────────────────────

def write_diagnostic_artifacts(
    output_dir: Path,
    diagnostics: ImperfectSpecialistDiagnostics,
    *, max_complementarity_pairs: int = 1000,
) -> dict[str, str]:
    """Persist all diagnostic artifacts and return path manifest."""
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, str] = {}

    records = diagnostics.records
    if not records:
        summary = {
            "schema_version": DIAGNOSTICS_SCHEMA_VERSION,
            "total_candidates_evaluated": diagnostics.total_candidates_evaluated,
            "diagnostic_shortlist_size": 0,
            "note": "No imperfect candidates observed in this campaign.",
        }
        path = output_dir / "imperfect_specialist_summary.json"
        path.write_text(
            json.dumps(summary, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )
        manifest["summary"] = path.name
        return manifest

    # Per-candidate JSONL
    records_path = output_dir / "useful_candidates_protected_effects.jsonl"
    diagnostics.persist(records_path)
    manifest["useful_candidates_protected_effects"] = records_path.name

    # Aggregate summary
    agg = aggregate_diagnostics(
        records, total_candidates_evaluated=diagnostics.total_candidates_evaluated
    )
    agg_path = output_dir / "imperfect_specialist_summary.json"
    agg_path.write_text(
        json.dumps(agg, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    manifest["summary"] = agg_path.name

    # This BUILD-only attribution is refreshed after Phase 1 routing adds
    # downstream-ready outcomes.  The initial artifact is still useful when no
    # candidate reaches routing.
    attribution_path = output_dir / "policy_attribution.json"
    attribution_path.write_text(
        json.dumps(
            summarize_policy_attribution(records), sort_keys=True, indent=2
        ) + "\n",
        encoding="utf-8",
    )
    manifest["policy_attribution"] = attribution_path.name

    # Feature set audit
    audit = audit_feature_sets()
    audit_path = output_dir / "feature_set_audit.json"
    audit_path.write_text(
        json.dumps(audit, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    manifest["feature_audit"] = audit_path.name

    # Static complementarity
    comp = compute_static_complementarity(records, max_pairs=max_complementarity_pairs)
    comp_path = output_dir / "static_complementarity.json"
    comp_path.write_text(
        json.dumps(comp, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    manifest["complementarity"] = comp_path.name

    return manifest
