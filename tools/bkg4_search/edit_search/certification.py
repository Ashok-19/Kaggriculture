from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence

from tools.bkg4_search.bonsai_editformer.actor_grammar import EditAction, node_refs

from .fitness import PairEvaluation
from .safety import SafetyAssessment, SafetySpec, SafetyTier
from .schema import EditProgram


class DiscoveryQuality(str, Enum):
    DISCOVERY_ELITE_FLIP = "discovery_elite_flip"
    DISCOVERY_ELITE_MARGIN = "discovery_elite_margin"
    DISCOVERY_SAFE_USEFUL = "discovery_safe_useful"
    DISCOVERY_SAFE_INCREMENTAL = "discovery_safe_incremental"
    DISCOVERY_BOUNDARY = "discovery_boundary"
    DISCOVERY_HARD_NEGATIVE = "discovery_hard_negative"
    DISCOVERY_CONTROLLED_NOOP = "discovery_controlled_noop"


class CertificationClass(str, Enum):
    UNCERTIFIED = "uncertified"
    CERTIFIED_ELITE_FLIP = "certified_elite_flip"
    CERTIFIED_ELITE_MARGIN = "certified_elite_margin"
    CERTIFIED_SAFE_USEFUL = "certified_safe_useful"
    CERTIFIED_SAFE_INCREMENTAL = "certified_safe_incremental"
    GENERALIZATION_BOUNDARY = "generalization_boundary"
    GENERALIZATION_HARD_NEGATIVE = "generalization_hard_negative"


class TransferFailure(str, Enum):
    NONE = "none"
    NOT_CERTIFIED = "not_certified"
    UNSAFE_TRANSFER = "unsafe_transfer"
    SAFE_BUT_INERT = "safe_but_inert"
    ACTIVE_NOT_USEFUL = "active_not_useful"
    TEMPORAL_OVERFIT = "temporal_overfit"


_DISCOVERY_MAP = {
    SafetyTier.ELITE_FLIP: DiscoveryQuality.DISCOVERY_ELITE_FLIP,
    SafetyTier.ELITE_MARGIN: DiscoveryQuality.DISCOVERY_ELITE_MARGIN,
    SafetyTier.STRICT_SAFE_USEFUL: DiscoveryQuality.DISCOVERY_SAFE_USEFUL,
    SafetyTier.SAFE_INCREMENTAL: DiscoveryQuality.DISCOVERY_SAFE_INCREMENTAL,
    SafetyTier.BOUNDARY: DiscoveryQuality.DISCOVERY_BOUNDARY,
    SafetyTier.HARD_NEGATIVE: DiscoveryQuality.DISCOVERY_HARD_NEGATIVE,
    SafetyTier.CONTROLLED_NOOP: DiscoveryQuality.DISCOVERY_CONTROLLED_NOOP,
}


@dataclass(frozen=True)
class PanelCertification:
    panel_id: str
    safety_tier: SafetyTier
    feasible: bool
    activated: bool
    behavior_changing: bool
    safe_useful: bool
    l2w: bool
    worst_protected_delta: int
    mean_margin_delta: float

    @classmethod
    def from_pair(cls, pair: PairEvaluation, spec: SafetySpec) -> "PanelCertification":
        a = spec.assess(pair)
        return cls(
            panel_id=str(pair.panel_id),
            safety_tier=a.tier,
            feasible=bool(a.feasible),
            activated=int(a.activation_total) > 0,
            behavior_changing=bool(a.behavior_changing),
            safe_useful=bool(a.safe_useful),
            l2w=int(a.l2w) > 0,
            worst_protected_delta=int(a.worst_protected_delta),
            mean_margin_delta=float(a.mean_margin_delta),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "panel_id": self.panel_id,
            "safety_tier": self.safety_tier.value,
            "feasible": self.feasible,
            "activated": self.activated,
            "behavior_changing": self.behavior_changing,
            "safe_useful": self.safe_useful,
            "l2w": self.l2w,
            "worst_protected_delta": self.worst_protected_delta,
            "mean_margin_delta": self.mean_margin_delta,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "PanelCertification":
        return cls(
            panel_id=str(raw["panel_id"]),
            safety_tier=SafetyTier(str(raw["safety_tier"])),
            feasible=bool(raw["feasible"]),
            activated=bool(raw["activated"]),
            behavior_changing=bool(raw["behavior_changing"]),
            safe_useful=bool(raw["safe_useful"]),
            l2w=bool(raw["l2w"]),
            worst_protected_delta=int(raw["worst_protected_delta"]),
            mean_margin_delta=float(raw["mean_margin_delta"]),
        )


@dataclass(frozen=True)
class CertificationVector:
    discovery_quality: DiscoveryQuality
    certification_class: CertificationClass
    transfer_failure: TransferFailure
    panel_count: int
    strict_safe_retention: float
    activation_retention: float
    behavior_retention: float
    safe_useful_retention: float
    elite_retention: float
    l2w_retention: float
    worst_certification_protected_delta: int
    certification_mean_margin_delta: float
    temporal: bool
    panels: tuple[PanelCertification, ...]

    @property
    def certified_positive(self) -> bool:
        return self.certification_class in {
            CertificationClass.CERTIFIED_ELITE_FLIP,
            CertificationClass.CERTIFIED_ELITE_MARGIN,
            CertificationClass.CERTIFIED_SAFE_USEFUL,
        }

    @property
    def certified_elite(self) -> bool:
        return self.certification_class in {
            CertificationClass.CERTIFIED_ELITE_FLIP,
            CertificationClass.CERTIFIED_ELITE_MARGIN,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "discovery_quality": self.discovery_quality.value,
            "certification_class": self.certification_class.value,
            "transfer_failure": self.transfer_failure.value,
            "panel_count": self.panel_count,
            "strict_safe_retention": self.strict_safe_retention,
            "activation_retention": self.activation_retention,
            "behavior_retention": self.behavior_retention,
            "safe_useful_retention": self.safe_useful_retention,
            "elite_retention": self.elite_retention,
            "l2w_retention": self.l2w_retention,
            "worst_certification_protected_delta": self.worst_certification_protected_delta,
            "certification_mean_margin_delta": self.certification_mean_margin_delta,
            "temporal": self.temporal,
            "panels": [x.to_dict() for x in self.panels],
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "CertificationVector":
        return cls(
            discovery_quality=DiscoveryQuality(str(raw["discovery_quality"])),
            certification_class=CertificationClass(str(raw["certification_class"])),
            transfer_failure=TransferFailure(str(raw["transfer_failure"])),
            panel_count=int(raw["panel_count"]),
            strict_safe_retention=float(raw["strict_safe_retention"]),
            activation_retention=float(raw["activation_retention"]),
            behavior_retention=float(raw["behavior_retention"]),
            safe_useful_retention=float(raw["safe_useful_retention"]),
            elite_retention=float(raw["elite_retention"]),
            l2w_retention=float(raw["l2w_retention"]),
            worst_certification_protected_delta=int(raw["worst_certification_protected_delta"]),
            certification_mean_margin_delta=float(raw["certification_mean_margin_delta"]),
            temporal=bool(raw.get("temporal", False)),
            panels=tuple(PanelCertification.from_dict(x) for x in raw.get("panels", ())),
        )

_CERTIFICATION_ORDER = {
    CertificationClass.CERTIFIED_ELITE_FLIP: 0,
    CertificationClass.CERTIFIED_ELITE_MARGIN: 1,
    CertificationClass.CERTIFIED_SAFE_USEFUL: 2,
    CertificationClass.CERTIFIED_SAFE_INCREMENTAL: 3,
    CertificationClass.UNCERTIFIED: 4,
    CertificationClass.GENERALIZATION_BOUNDARY: 5,
    CertificationClass.GENERALIZATION_HARD_NEGATIVE: 6,
}


def certification_rank(value: CertificationClass | str) -> int:
    cls = value if isinstance(value, CertificationClass) else CertificationClass(str(value))
    return _CERTIFICATION_ORDER[cls]


def program_contains_temporal(program: Any) -> bool:
    for atom in getattr(program, "atoms", ()):
        if str(getattr(atom, "operation", "")) == "CHANGE_WINDOW":
            return True
        action = dict(getattr(atom, "action", {}) or {})
        if str(action.get("feature", "")) in {"STEP", "DAY", "HOUR"}:
            return True
        if str(action.get("feature2", "")) in {"STEP", "DAY", "HOUR"}:
            return True
    return False


@dataclass(frozen=True)
class CertificationBatch:
    vectors: Mapping[str, CertificationVector]
    root_pairs: Mapping[str, tuple[PairEvaluation, ...]]
    source_pairs: Mapping[str, tuple[PairEvaluation, ...]]


def certify_programs_exact(
    archive: Any,
    programs: Sequence[Any],
    discovery_pairs: Mapping[str, PairEvaluation],
    *,
    evaluator: Any,
    case_bank: Any,
    generation: int,
    work_dir: str | Path,
    panel_parts: int = 2,
    panel_indices: Sequence[int] | None = None,
    spec: SafetySpec | None = None,
) -> CertificationBatch:
    """Exact-CUDA certification on disjoint evaluation subpanels.

    Search/discovery cases are already bound in CaseRegistry; certification cases are
    registered separately and never become search predicates or normal ranker validation.
    """
    from .context_bank import certification_panels
    programs = tuple(programs)
    if not programs:
        return CertificationBatch({}, {}, {})
    bank_identity = case_bank.identity()
    archive.register_external_case_bank("certification", bank_identity, tuple(case_bank.by_id))
    panels = certification_panels(case_bank, parts=max(1, int(panel_parts)))
    if panel_indices is None:
        selected_indices = tuple(range(len(panels)))
    else:
        selected_indices = tuple(dict.fromkeys(int(x) for x in panel_indices))
        if not selected_indices or any(x < 0 or x >= len(panels) for x in selected_indices):
            raise ValueError("certification panel_indices reference an invalid/empty mini-panel selection")
    root_pairs: dict[str, list[PairEvaluation]] = {p.program_id: [] for p in programs}
    source_pairs: dict[str, list[PairEvaluation]] = {p.program_id: [] for p in programs}
    work = Path(work_dir); work.mkdir(parents=True, exist_ok=True)
    for part_i in selected_indices:
        panel = panels[part_i]
        panel_id = f"{archive.run_id}/g{int(generation)}/cert/p{part_i}"
        sids = {archive.root_semantic_id}
        for program in programs:
            sids.add(program.source_parent_semantic_id); sids.add(program.child_semantic_id)
        exact = evaluator.evaluate(
            [archive.genome(sid) for sid in sorted(sids)], panel,
            work_dir=work / f"p{part_i:02d}", panel_id=panel_id,
        )
        root_ev = exact[archive.root_semantic_id]
        protected = {c.case_id: c.protected for c in panel}
        for program in programs:
            child = exact[program.child_semantic_id]
            root_pairs[program.program_id].append(PairEvaluation.from_states(root_ev, child, protected=protected))
            source_pairs[program.program_id].append(PairEvaluation.from_states(exact[program.source_parent_semantic_id], child, protected=protected))
    vectors: dict[str, CertificationVector] = {}
    for program in programs:
        if program.program_id not in discovery_pairs:
            raise ValueError("certification candidate lacks authoritative discovery pair")
        cpairs = {pair.panel_id: pair for pair in root_pairs[program.program_id]}
        vector = certify_program(
            discovery_pairs[program.program_id], cpairs,
            temporal=program_contains_temporal(program), spec=spec,
        )
        archive.record_certification(program.program_id, vector, certification_bank_identity=bank_identity)
        vectors[program.program_id] = vector
    return CertificationBatch(
        vectors=vectors,
        root_pairs={k: tuple(v) for k, v in root_pairs.items()},
        source_pairs={k: tuple(v) for k, v in source_pairs.items()},
    )



def propose_certified_neighborhood(
    archive: Any,
    source_program: EditProgram,
    *,
    n: int = 64,
    seed: int = 0,
) -> list[EditProgram]:
    """Bounded typed local neighborhood around a certified source state.

    Route alternatives are enumerated first because they can change strategic payoff
    without changing a proven guard. Threshold and window perturbations then explore
    locality/activation breadth. No hidden state or certification labels enter proposal
    construction; exact CUDA decides all quality after generation.
    """
    n = max(0, int(n))
    if n == 0 or source_program.program_id not in archive.programs:
        return []
    parent = archive.genome(source_program.child_semantic_id)
    refs = [ref for ref in node_refs(parent) if ref is not None]
    actions: list[EditAction] = []

    # Keep the certified guard fixed while sweeping strategic route payload first.
    for ref in refs:
        if ref.kind != "rule" or ref.rule_index < 0:
            continue
        current_route = int(parent.rules[ref.rule_index].route)
        for route in archive.route_ids:
            if int(route) == current_route:
                continue
            actions.append(EditAction("CHANGE_ROUTE", int(ref.node_index), route=int(route)))

    # Local threshold breadth around each certified predicate.
    for ref in refs:
        if ref.kind != "condition" or ref.rule_index < 0 or ref.condition_index < 0:
            continue
        old = parent.rules[ref.rule_index].conditions[ref.condition_index]
        step = max(1, abs(int(old.value)) // 50)
        values = (
            int(old.value) - 1, int(old.value) + 1,
            int(old.value) - step, int(old.value) + step,
            int(old.value) - 2 * step, int(old.value) + 2 * step,
        )
        for value in dict.fromkeys(values):
            try:
                action = EditAction(
                    "CHANGE_THRESHOLD", int(ref.node_index),
                    feature=old.feature, cmp=old.cmp, index=old.index, value=int(value),
                    route=int(parent.rules[ref.rule_index].route),
                )
                actions.append(action)
            except ValueError:
                continue

    # Strategic start-window shifts are late because guard/route semantics are stronger.
    window_values = (
        int(parent.prefix_step) - 72, int(parent.prefix_step) - 24,
        int(parent.prefix_step) + 24, int(parent.prefix_step) + 72,
        0, 72, 144, 216, 288, 432, 576,
    )
    for value in dict.fromkeys(max(0, min(719, int(x))) for x in window_values):
        if value != int(parent.prefix_step):
            actions.append(EditAction("CHANGE_WINDOW", 0, prefix_step=value))

    out: list[EditProgram] = []
    seen: set[str] = set()
    for i, action in enumerate(actions):
        try:
            program = EditProgram.build(
                parent, (action,), archive.route_ids,
                emitter="certified_neighborhood",
                seed=int(seed) + i,
                generation_depth=int(source_program.generation_depth) + 1,
                parent_program_id=source_program.program_id,
                run_id=archive.run_id,
            )
        except ValueError:
            continue
        if program.child_semantic_id == parent.semantic_id() or program.child_semantic_id in seen:
            continue
        if program.child_semantic_id in archive.states:
            continue
        seen.add(program.child_semantic_id)
        out.append(program)
        if len(out) >= n:
            break
    return out


@dataclass(frozen=True)
class CertifiedNeighborhoodBatch:
    programs: tuple[EditProgram, ...]
    vectors: Mapping[str, CertificationVector]
    root_pairs: Mapping[str, PairEvaluation]
    source_pairs: Mapping[str, PairEvaluation]


def evaluate_certified_neighborhood_exact(
    archive: Any,
    source_program: EditProgram,
    *,
    evaluator: Any,
    search_bank: Any,
    certification_bank: Any,
    generation: int,
    seed: int,
    work_dir: str | Path,
    budget: int = 64,
    spec: SafetySpec | None = None,
) -> CertifiedNeighborhoodBatch:
    """Generate, exact-evaluate, admit and independently certify a local elite neighborhood.

    Proposal construction uses only the certified child structure. Search-bank exact truth
    controls QD admission; certification cases are evaluated only after a candidate is
    search-feasible and strategically useful. This keeps certification outcome out of the
    proposal path while producing high-information local supervision around robust edits.
    """
    spec = spec or SafetySpec()
    programs = tuple(propose_certified_neighborhood(archive, source_program, n=budget, seed=seed))
    if not programs:
        return CertifiedNeighborhoodBatch((), {}, {}, {})
    parent = archive.genome(source_program.child_semantic_id)
    for program in programs:
        archive.add_program(program, program.reconstruct(parent, archive.route_ids))

    archive.bind_case_bank(search_bank.identity(), tuple(search_bank.by_id))
    panel_id = f"{archive.run_id}/g{int(generation)}/certified-neighborhood/full"
    archive.register_panel(panel_id, tuple(case.case_id for case in search_bank.rows))
    sids = {archive.root_semantic_id, source_program.child_semantic_id}
    sids.update(program.child_semantic_id for program in programs)
    exact = evaluator.evaluate(
        [archive.genome(sid) for sid in sorted(sids)], search_bank.rows,
        work_dir=Path(work_dir) / "search", panel_id=panel_id,
    )
    for evaluation in exact.values():
        archive.record_state_evaluation(evaluation)
    protected = {case.case_id: case.protected for case in search_bank.rows}
    root_ev = exact[archive.root_semantic_id]
    source_ev = exact[source_program.child_semantic_id]
    root_pairs: dict[str, PairEvaluation] = {}
    source_pairs: dict[str, PairEvaluation] = {}
    cert_candidates: list[EditProgram] = []
    for program in programs:
        child_ev = exact[program.child_semantic_id]
        root_pair = PairEvaluation.from_states(root_ev, child_ev, protected=protected)
        source_pair = PairEvaluation.from_states(source_ev, child_ev, protected=protected)
        root_pairs[program.program_id] = root_pair
        source_pairs[program.program_id] = source_pair
        archive.consider(program, root_pair, source_pair=source_pair)
        assessment = spec.assess(root_pair)
        if assessment.feasible and assessment.behavior_changing and assessment.safe_useful:
            cert_candidates.append(program)

    vectors: dict[str, CertificationVector] = {}
    if cert_candidates:
        seat0 = sum(case.seat == 0 for case in certification_bank.rows)
        seat1 = sum(case.seat == 1 for case in certification_bank.rows)
        panel_parts = min(2, max(1, min(seat0, seat1)))
        certified = certify_programs_exact(
            archive, cert_candidates, root_pairs,
            evaluator=evaluator, case_bank=certification_bank,
            generation=generation, work_dir=Path(work_dir) / "certification",
            panel_parts=panel_parts, spec=spec,
        )
        vectors.update(certified.vectors)
        for program in cert_candidates:
            archive.observe_motif(program, root_pairs[program.program_id], certification=vectors.get(program.program_id))

    return CertifiedNeighborhoodBatch(programs, vectors, root_pairs, source_pairs)


def discovery_quality(pair: PairEvaluation, spec: SafetySpec | None = None) -> DiscoveryQuality:
    assessment = (spec or SafetySpec()).assess(pair)
    return _DISCOVERY_MAP[assessment.tier]


def certify_program(
    discovery_pair: PairEvaluation,
    certification_pairs: Mapping[str, PairEvaluation],
    *,
    temporal: bool = False,
    behavior_floor: float = 0.50,
    useful_floor: float = 0.40,
    elite_floor: float = 0.25,
    spec: SafetySpec | None = None,
) -> CertificationVector:
    """Classify discovery quality and independent transfer without collapsing them.

    Individual certified positives require every supplied certification panel to be
    hard-feasible. Cohort-level gates may tolerate lower retention, but a program that
    is known unsafe on any certification panel is never promoted as a certified positive.
    """
    spec = spec or SafetySpec()
    dq = discovery_quality(discovery_pair, spec)
    rows = tuple(
        PanelCertification.from_pair(pair, spec)
        for _, pair in sorted(certification_pairs.items(), key=lambda kv: str(kv[0]))
    )
    n = len(rows)
    if not n:
        return CertificationVector(
            dq, CertificationClass.UNCERTIFIED, TransferFailure.NOT_CERTIFIED,
            0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0.0, bool(temporal), (),
        )

    frac = lambda predicate: sum(bool(predicate(x)) for x in rows) / n
    safe = frac(lambda x: x.feasible)
    active = frac(lambda x: x.activated)
    behavior = frac(lambda x: x.behavior_changing)
    useful = frac(lambda x: x.safe_useful)
    elite = frac(lambda x: x.safety_tier in {SafetyTier.ELITE_FLIP, SafetyTier.ELITE_MARGIN})
    l2w = frac(lambda x: x.l2w)
    worst_protected = min(x.worst_protected_delta for x in rows)
    mean_delta = sum(x.mean_margin_delta for x in rows) / n

    if safe < 1.0:
        cls = CertificationClass.GENERALIZATION_HARD_NEGATIVE
        failure = TransferFailure.UNSAFE_TRANSFER
    elif behavior < float(behavior_floor):
        cls = CertificationClass.GENERALIZATION_BOUNDARY
        failure = TransferFailure.TEMPORAL_OVERFIT if temporal and active < float(behavior_floor) else TransferFailure.SAFE_BUT_INERT
    elif useful < float(useful_floor):
        cls = CertificationClass.GENERALIZATION_BOUNDARY
        failure = TransferFailure.ACTIVE_NOT_USEFUL
    elif elite >= float(elite_floor) and l2w > 0.0 and any(x.safety_tier == SafetyTier.ELITE_FLIP for x in rows):
        cls = CertificationClass.CERTIFIED_ELITE_FLIP
        failure = TransferFailure.NONE
    elif elite >= float(elite_floor):
        cls = CertificationClass.CERTIFIED_ELITE_MARGIN
        failure = TransferFailure.NONE
    else:
        cls = CertificationClass.CERTIFIED_SAFE_USEFUL
        failure = TransferFailure.NONE

    return CertificationVector(
        discovery_quality=dq,
        certification_class=cls,
        transfer_failure=failure,
        panel_count=n,
        strict_safe_retention=safe,
        activation_retention=active,
        behavior_retention=behavior,
        safe_useful_retention=useful,
        elite_retention=elite,
        l2w_retention=l2w,
        worst_certification_protected_delta=int(worst_protected),
        certification_mean_margin_delta=float(mean_delta),
        temporal=bool(temporal),
        panels=rows,
    )
