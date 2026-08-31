"""Phase-0 imperfect-specialist routing diagnostics.

Observational instrumentation only.  Does not change production acceptance,
promotion, CEGIS, routability, or any safety semantics.  All outputs are
clearly labelled as diagnostic evidence, never as composition-safe or
acceptance-authoritative.
"""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence

from tools.bkg4_search.module_codec import FEATURES as GATE_GRAMMAR_FEATURES

from .effects import EffectBlock, EffectVector
from .schema import DatasetRole, canonical_json, stable_identity

DIAGNOSTICS_SCHEMA_VERSION = 1

_GATE_FEATURE_SET = frozenset(GATE_GRAMMAR_FEATURES)


# ── routability classification with separated legal/gate contexts ────────────

class DiagnosticRoutabilityClass(str, Enum):
    ROUTABLE = "ROUTABLE"
    GATE_LIMITATION = "GATE_LIMITATION"
    FUTURE_DEPENDENT = "FUTURE_DEPENDENT"
    UNKNOWN = "UNKNOWN"


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


# ── imperfect candidate record ───────────────────────────────────────────────

@dataclass(frozen=True)
class ImperfectCandidateRecord:
    candidate_semantic_id: str
    structural_id: str
    family: str
    lane: str
    target_loss_to_win: int
    target_win_to_loss: int
    target_harm_count: int
    worst_delta: int
    aggregate_margin: int
    effect_signature: str
    routability_classification: str
    diagnostic_routability: str
    per_case_effects: tuple[PerCaseEffect, ...]
    effect_vector_doc: dict[str, Any] | None
    initiation_step: int
    checkpoint_metadata: dict[str, Any]
    gate_evidence: dict[str, Any]

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
            "target_win_to_loss": self.target_win_to_loss,
            "target_harm_count": self.target_harm_count,
            "worst_delta": self.worst_delta,
            "aggregate_margin": self.aggregate_margin,
            "effect_signature": self.effect_signature,
            "routability_classification": self.routability_classification,
            "diagnostic_routability": self.diagnostic_routability,
            "per_case_effects": [e.to_document() for e in self.per_case_effects],
            "effect_vector": self.effect_vector_doc,
            "initiation_step": self.initiation_step,
            "checkpoint_metadata": self.checkpoint_metadata,
            "gate_evidence": self.gate_evidence,
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
            target_win_to_loss=doc["target_win_to_loss"],
            target_harm_count=doc["target_harm_count"],
            worst_delta=doc["worst_delta"],
            aggregate_margin=doc["aggregate_margin"],
            effect_signature=doc["effect_signature"],
            routability_classification=doc["routability_classification"],
            diagnostic_routability=doc["diagnostic_routability"],
            per_case_effects=tuple(
                PerCaseEffect(**e) for e in doc["per_case_effects"]
            ),
            effect_vector_doc=doc.get("effect_vector"),
            initiation_step=doc["initiation_step"],
            checkpoint_metadata=doc["checkpoint_metadata"],
            gate_evidence=doc["gate_evidence"],
        )


# ── diagnostic collector ────────────────────────────────────────────────────

class ImperfectSpecialistDiagnostics:
    """Collects imperfect-specialist diagnostic records during a campaign."""

    def __init__(self) -> None:
        self._records: list[ImperfectCandidateRecord] = []

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


# ── aggregate diagnostic summaries ───────────────────────────────────────────

def aggregate_diagnostics(
    records: Sequence[ImperfectCandidateRecord],
) -> dict[str, Any]:
    """Compute distribution and classification summaries."""
    total = len(records)
    l2w_positive = [r for r in records if r.target_loss_to_win > 0]
    l2w_and_w2l = [r for r in l2w_positive if r.target_win_to_loss > 0]

    # W→L distribution buckets
    w2l_buckets = {"w2l_eq_0": 0, "w2l_eq_1": 0, "w2l_eq_2": 0,
                   "w2l_3_to_5": 0, "w2l_gt_5": 0}
    for r in records:
        w = r.target_win_to_loss
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
            1 for r in records if r.target_loss_to_win >= 2 and r.target_win_to_loss == 1
        ),
        "l2w_ge3_w2l_le2": sum(
            1 for r in records if r.target_loss_to_win >= 3 and r.target_win_to_loss <= 2
        ),
        "l2w_ge5_w2l_le2": sum(
            1 for r in records if r.target_loss_to_win >= 5 and r.target_win_to_loss <= 2
        ),
        "l2w_gt_w2l": sum(
            1 for r in records if r.target_loss_to_win > r.target_win_to_loss
        ),
    }

    # Diagnostic routability classes
    routability_classes: dict[str, int] = defaultdict(int)
    for r in records:
        routability_classes[r.diagnostic_routability] += 1

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
        "total_evaluated_candidates": total,
        "candidates_with_l2w_gt_0": len(l2w_positive),
        "candidates_with_l2w_gt_0_and_w2l_gt_0": len(l2w_and_w2l),
        "w2l_distribution": dict(sorted(w2l_buckets.items())),
        "useful_imperfect_buckets": useful,
        "diagnostic_routability_classes": dict(sorted(routability_classes.items())),
        "by_family": dict(sorted(by_family.items())),
        "by_lane": dict(sorted(by_lane.items())),
    }


# ── feature set audit ────────────────────────────────────────────────────────

def audit_feature_sets() -> dict[str, Any]:
    """Exact enumeration of legal, gate-grammar, and routability feature sets."""
    from tools.bkg4_search.edit_search.goals import CONTRASTIVE_FEATURES
    from tools.bkg4_search.information_state_io import registry_manifest

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
    a_solves_b_breaks: frozenset[str]
    b_solves_a_breaks: frozenset[str]
    classification: str

    def to_document(self) -> dict[str, Any]:
        return {
            "candidate_a": self.candidate_a,
            "candidate_b": self.candidate_b,
            "a_solves_b_breaks": sorted(self.a_solves_b_breaks),
            "b_solves_a_breaks": sorted(self.b_solves_a_breaks),
            "a_solves_b_breaks_count": len(self.a_solves_b_breaks),
            "b_solves_a_breaks_count": len(self.b_solves_a_breaks),
            "classification": self.classification,
        }


def _compatible_for_comparison(a: ImperfectCandidateRecord, b: ImperfectCandidateRecord) -> bool:
    """Cheap compatibility check using obvious keys."""
    if a.candidate_semantic_id == b.candidate_semantic_id:
        return False
    if a.lane != b.lane:
        return False
    return True


def compute_static_complementarity(
    records: Sequence[ImperfectCandidateRecord],
    *,
    max_pairs: int = 1000,
) -> dict[str, Any]:
    """STATIC COMPLEMENTARITY HEURISTIC — offline diagnostic only.

    Never labelled COMPOSITION_SAFE.  Uses per-case TARGET EffectVector
    SOLVES/BREAKS sets for pairwise analysis within compatible candidates.
    """
    useful = [r for r in records if r.target_loss_to_win > 0]
    pairs: list[ComplementarityPair] = []
    pair_count = 0

    for i, a in enumerate(useful):
        if pair_count >= max_pairs:
            break
        a_solves = a.solves_set()
        a_breaks = a.breaks_set()
        if not a_solves and not a_breaks:
            continue
        for b in useful[i + 1:]:
            if pair_count >= max_pairs:
                break
            if not _compatible_for_comparison(a, b):
                continue
            b_solves = b.solves_set()
            b_breaks = b.breaks_set()
            if not b_solves and not b_breaks:
                continue
            a_solves_b_breaks = a_solves & b_breaks
            b_solves_a_breaks = b_solves & a_breaks
            if not a_solves_b_breaks and not b_solves_a_breaks:
                continue

            if a_solves_b_breaks and b_solves_a_breaks:
                if a_breaks <= b_solves and b_breaks <= a_solves:
                    classification = "NEAR_CLOSED_PAIR"
                else:
                    classification = "TWO_WAY_COMPLEMENT"
            elif a_solves_b_breaks or b_solves_a_breaks:
                classification = "ONE_WAY_COMPLEMENT"
            else:
                classification = "UNRESOLVED_PAIR"

            pairs.append(ComplementarityPair(
                candidate_a=a.candidate_semantic_id,
                candidate_b=b.candidate_semantic_id,
                a_solves_b_breaks=a_solves_b_breaks,
                b_solves_a_breaks=b_solves_a_breaks,
                classification=classification,
            ))
            pair_count += 1

    classifications: dict[str, int] = defaultdict(int)
    for p in pairs:
        classifications[p.classification] += 1

    return {
        "schema_version": DIAGNOSTICS_SCHEMA_VERSION,
        "label": "STATIC_COMPLEMENTARITY_HEURISTIC",
        "warning": "NOT_COMPOSITION_SAFE",
        "candidates_analyzed": len(useful),
        "pairs_evaluated": len(pairs),
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
) -> dict[str, str]:
    """Persist all diagnostic artifacts and return path manifest."""
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, str] = {}

    records = diagnostics.records
    if not records:
        summary = {
            "schema_version": DIAGNOSTICS_SCHEMA_VERSION,
            "total_evaluated_candidates": 0,
            "note": "No imperfect candidates observed in this campaign.",
        }
        path = output_dir / "imperfect_specialist_summary.json"
        path.write_text(
            json.dumps(summary, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )
        manifest["summary"] = str(path)
        return manifest

    # Per-candidate JSONL
    records_path = output_dir / "imperfect_candidates.jsonl"
    diagnostics.persist(records_path)
    manifest["imperfect_candidates"] = str(records_path)

    # Aggregate summary
    agg = aggregate_diagnostics(records)
    agg_path = output_dir / "imperfect_specialist_summary.json"
    agg_path.write_text(
        json.dumps(agg, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    manifest["summary"] = str(agg_path)

    # Feature set audit
    audit = audit_feature_sets()
    audit_path = output_dir / "feature_set_audit.json"
    audit_path.write_text(
        json.dumps(audit, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    manifest["feature_audit"] = str(audit_path)

    # Static complementarity
    comp = compute_static_complementarity(records)
    comp_path = output_dir / "static_complementarity.json"
    comp_path.write_text(
        json.dumps(comp, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    manifest["complementarity"] = str(comp_path)

    return manifest
