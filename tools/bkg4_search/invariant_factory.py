from __future__ import annotations

import hashlib
import json
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

from tools.bkg4_search.edit_search.panels import CaseRecord
from tools.bkg4_search.invariant_module import InvariantStrategicModule
from tools.bkg4_search.invariant_search import (
    InvariantSearchBackend,
    local_structural_neighborhood,
    pair_case_rows,
    pair_relative_case_rows,
)
from tools.bkg4_search.invariant_groups import tape_group_id
from tools.bkg4_search.invariant_schema import make_corpus_record
from tools.bkg4_search.invariant_lineage import (
    SPECIALIST_LINEAGE_SCHEMA_VERSION,
    make_parent_evidence,
)
from tools.bkg4_search.invariant_specialists import (
    DiversityReport,
    SpecialistEvidence,
    SpecialistLineage,
    SpecialistTierThresholds,
    analyze_specialist,
    calibrate_specialist_thresholds,
    diversity_report,
    select_corpus_quota,
    select_screen_quota,
)


@dataclass(frozen=True)
class OfflineSpecialistBank:
    cases: tuple[CaseRecord, ...]
    cohort_by_group: Mapping[str, str]
    source_roles: tuple[str, ...]
    cohort_display: Mapping[str, str]
    model_context_json: str = "{}"


def split_specialist_panels(
    cases: Sequence[CaseRecord],
    *,
    screen_cases: int,
    label_cases: int,
    seed: int,
) -> tuple[tuple[CaseRecord, ...], tuple[CaseRecord, ...]]:
    """Choose exact-size tape-disjoint screen and label panels."""
    grouped: dict[str, list[CaseRecord]] = {}
    for case in cases:
        if case.role != "discovery":
            raise ValueError("specialist panels accept discovery-role cases only")
        grouped.setdefault(case.tape_path, []).append(case)
    ordered = sorted(
        grouped,
        key=lambda tape: hashlib.sha256(f"{int(seed)}\0{tape}".encode()).digest(),
    )

    def take(target: int, available: list[str]) -> tuple[list[str], list[str]]:
        target = int(target)
        chosen = []
        remaining = list(available)
        total = 0
        while total < target:
            need = target - total
            index = next((i for i, tape in enumerate(remaining) if len(grouped[tape]) <= need), None)
            if index is None:
                raise ValueError(f"cannot form tape-complete specialist panel of {target} cases")
            tape = remaining.pop(index); chosen.append(tape); total += len(grouped[tape])
        return chosen, remaining

    screen_tapes, remaining = take(screen_cases, ordered)
    label_tapes, _ = take(label_cases, remaining)
    screen = tuple(sorted((case for tape in screen_tapes for case in grouped[tape]), key=lambda row: row.case_id))
    label = tuple(sorted((case for tape in label_tapes for case in grouped[tape]), key=lambda row: row.case_id))
    return screen, label


def load_offline_specialist_bank(
    path: str | Path,
    *,
    cohort_kind: str,
) -> OfflineSpecialistBank:
    """Normalize an offline opponent bank while keeping identity outside executable input."""
    source = Path(path)
    forbidden = " ".join((str(source), str(cohort_kind))).lower()
    if any(term in forbidden for term in ("certification", "holdout", "ranker_validation")):
        raise ValueError("certification/holdout banks cannot enter specialist search")
    cases = []
    cohorts: dict[str, str] = {}
    roles: set[str] = set()
    displays: dict[str, str] = {}
    for line_no, line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        role = str(row.get("role", "offline"))
        if any(term in role.lower() for term in ("certification", "holdout", "ranker_validation")):
            raise ValueError("certification/holdout rows cannot enter specialist search")
        tape_value = row.get("tape_path", row.get("tape"))
        if not tape_value:
            raise ValueError(f"offline specialist row {line_no} has no tape")
        tape = Path(str(tape_value))
        if not tape.is_absolute():
            candidates = ((source.parent / tape).resolve(), (Path.cwd() / tape).resolve())
            tape = next((candidate for candidate in candidates if candidate.exists()), candidates[-1])
        inferred = re.sub(r"^(?:\d+_){0,2}", "", tape.stem)
        inferred = re.sub(r"_s[01]_ep\d+$", "", inferred)
        identity = str(row.get("team", row.get("family", inferred or role)))
        digest = hashlib.sha256(identity.encode()).hexdigest()[:12]
        cohort = f"{cohort_kind}:{digest}"
        displays[cohort] = identity
        seed = int(row["seed"])
        seat = int(row["seat"])
        raw_case_id = str(row.get("case_id") or hashlib.sha256(
            f"{tape}\0{seat}\0{seed}\0{cohort_kind}".encode()
        ).hexdigest())
        case_id = f"{cohort_kind}:{raw_case_id}"
        case = CaseRecord(
            case_id=case_id, tape_path=str(tape), seat=seat, seed=seed, family=cohort,
            protected=bool(row.get("protected", True)), own_history=bool(row.get("own_history", False)),
            role="discovery", baseline_margin=int(row.get("baseline_margin", 0)), model_context_json="{}",
        )
        cases.append(case); cohorts[case.tape_path] = cohort; roles.add(role)
    if not cases:
        raise ValueError("offline specialist bank is empty")
    if len({case.case_id for case in cases}) != len(cases):
        raise ValueError("offline specialist bank has duplicate cases")
    return OfflineSpecialistBank(tuple(cases), cohorts, tuple(sorted(roles)), dict(sorted(displays.items())))


@dataclass(frozen=True)
class ProposalBatch:
    modules: tuple[InvariantStrategicModule, ...]
    lineages: tuple[SpecialistLineage, ...]
    diversity: DiversityReport
    generation_seconds: float
    proposals_per_second: float
    duplicate_rejections: int
    operator_timing_seconds: Mapping[str, float] = field(default_factory=dict)


class SpecialistFactory:
    """Permanent high-recall proposal and exact-screen facade around B4IM search."""

    def __init__(self, backend: InvariantSearchBackend, *, root_controller: str):
        self.backend = backend
        self.root_controller = str(root_controller)
        self._seen: set[str] = set()
        self._lineages: dict[str, SpecialistLineage] = {}

    def module_for_semantic_id(self, semantic_id: str) -> InvariantStrategicModule:
        lineage = self._lineages.get(str(semantic_id))
        if lineage is None:
            raise KeyError(f"unknown specialist lineage module {semantic_id}")
        return lineage.module

    def lineage_for_semantic_id(self, semantic_id: str) -> SpecialistLineage:
        lineage = self._lineages.get(str(semantic_id))
        if lineage is None:
            raise KeyError(f"unknown specialist lineage {semantic_id}")
        return lineage

    def ancestry_for_lineage(self, lineage: SpecialistLineage) -> tuple[SpecialistLineage, ...]:
        """Return the durable root-through-parent chain for one specialist child."""
        if lineage.depth == 0:
            return ()
        chain: list[SpecialistLineage] = []
        parent_sid = lineage.parent_semantic_id
        seen = {lineage.module.semantic_id()}
        while parent_sid is not None:
            if parent_sid in seen:
                raise ValueError("specialist lineage contains an ancestry cycle")
            seen.add(parent_sid)
            parent = self.lineage_for_semantic_id(parent_sid)
            chain.append(parent)
            parent_sid = parent.parent_semantic_id
        chain.reverse()
        if len(chain) != lineage.depth:
            raise ValueError("specialist lineage depth does not match ancestry length")
        if not chain or chain[0].depth != 0 or chain[0].root_semantic_id != lineage.root_semantic_id:
            raise ValueError("specialist lineage has an inconsistent root ancestry")
        for expected_depth, row in enumerate(chain):
            if row.depth != expected_depth or row.root_semantic_id != lineage.root_semantic_id:
                raise ValueError("specialist ancestry depth/root mismatch")
            if expected_depth and row.parent_semantic_id != chain[expected_depth - 1].module.semantic_id():
                raise ValueError("specialist ancestry parent mismatch")
        if chain[-1].module.semantic_id() != lineage.parent_semantic_id:
            raise ValueError("specialist immediate parent is not the ancestry tail")
        return tuple(chain)

    def _legacy_recursive_source_priority(self, lineage: SpecialistLineage) -> int:
        """Preserve pre-lineage-v2 recursive ordering while provenance becomes more exact."""
        current = lineage
        seen: set[str] = set()
        while True:
            sid = current.module.semantic_id()
            if sid in seen:
                return 0
            seen.add(sid)
            # Genuine campaign MCTS roots had no source-goal metadata. Recovering that
            # provenance must not promote them in the recursive parent ordering.
            if current.depth == 0 and current.operator == "mcts":
                return 0
            if current.source_goal_id:
                return 1
            if current.parent_semantic_id is None:
                return 0
            parent = self._lineages.get(current.parent_semantic_id)
            if parent is None:
                return 0
            current = parent

    def propose(
        self,
        *,
        count: int,
        seed: int,
        target_goal_ids: Sequence[str] | None = None,
        recursive: bool = False,
        capability_level: int | None = None,
    ) -> ProposalBatch:
        count = max(0, int(count))
        started = time.perf_counter()
        rows: list[InvariantStrategicModule] = []
        lineages: list[SpecialistLineage] = []
        duplicate_rejections = 0
        operator_timing: Counter[str] = Counter()
        if recursive:
            recursive_started = time.perf_counter()
            research = [
                evidence for evidence in self.backend.specialist_archive.research.values()
                if evidence.action_change_count > 0 and evidence.tier != "INERT"
            ]
            research.sort(key=lambda evidence: (
                -int((self._lineages.get(evidence.module.semantic_id()) or SpecialistLineage.root(
                    evidence.module, parent_controller=self.root_controller,
                )).depth),
                -self._legacy_recursive_source_priority(
                    self._lineages.get(evidence.module.semantic_id()) or SpecialistLineage.root(
                        evidence.module, parent_controller=self.root_controller,
                    )
                ),
                -int(evidence.useful_groups),
                evidence.quality_key(),
            ))
            recursive_limit = count if capability_level is None else max(1, count // 2)
            for parent_evidence in research:
                parent = parent_evidence.module
                parent_lineage = self._lineages.get(parent.semantic_id()) or SpecialistLineage.root(
                    parent, parent_controller=self.root_controller,
                )
                self._lineages.setdefault(parent.semantic_id(), parent_lineage)
                children = local_structural_neighborhood(
                    parent, route_ids=self.backend.route_ids,
                    budget=max(8, recursive_limit // max(1, len(research))),
                    seed=seed + len(rows) * 1009,
                )
                for child in children:
                    sid = child.semantic_id()
                    if sid in self._seen:
                        duplicate_rejections += 1; continue
                    retrospective = self.backend._goal_for_module(child)
                    lineage = parent_lineage.child(
                        child, operator="local_structural_neighborhood",
                        source_margin_delta=max((row.mean_delta for row in parent_evidence.cohorts), default=0.0),
                        retrospective_goal_id=(retrospective.goal_id if retrospective is not None else None),
                    )
                    rows.append(child); lineages.append(lineage); self._seen.add(sid); self._lineages[sid] = lineage
                    if len(rows) >= recursive_limit:
                        break
                if len(rows) >= recursive_limit:
                    break
            operator_timing["local_neighborhood"] += time.perf_counter() - recursive_started
        previous_level = self.backend.ladder.level
        requested_level = 1 if capability_level is None else max(1, int(capability_level))
        self.backend.ladder.level = max(requested_level, previous_level)
        try:
            stalled = 0
            attempt = 0
            while len(rows) < count and stalled < 6:
                need = count - len(rows)
                proposed = self.backend.propose(
                    n=max(need, min(4096, count)), seed=seed + attempt * 104729,
                    target_goal_ids=target_goal_ids,
                )
                for name, seconds in self.backend.last_proposal_profile.items():
                    operator_timing[name] += float(seconds)
                added = 0
                for module in proposed:
                    sid = module.semantic_id()
                    if sid in self._seen:
                        duplicate_rejections += 1; continue
                    origin = self.backend.last_proposal_origins.get(sid)
                    lineage = SpecialistLineage.root(
                        module,
                        parent_controller=self.root_controller,
                        operator=(origin.operator if origin is not None else "root"),
                        source_goal_id=(origin.source_goal_id if origin is not None else None),
                        source_goal_path=(origin.source_goal_path if origin is not None else None),
                        structural_mechanisms=(origin.structural_mechanisms if origin is not None else None),
                        retrospective_goal_id=(origin.retrospective_goal_id if origin is not None else None),
                    )
                    rows.append(module); lineages.append(lineage); self._seen.add(sid); self._lineages[sid] = lineage
                    added += 1
                    if len(rows) >= count:
                        break
                stalled = stalled + 1 if added == 0 else 0
                attempt += 1
        finally:
            self.backend.ladder.level = previous_level
        if len(rows) != count:
            raise RuntimeError(f"specialist proposal underfill: {len(rows)} != {count}")
        elapsed = time.perf_counter() - started
        modules = tuple(rows)
        return ProposalBatch(
            modules, tuple(lineages), diversity_report(modules), elapsed,
            len(modules) / max(elapsed, 1e-9), duplicate_rejections,
            dict(sorted(operator_timing.items())),
        )

    def screen(
        self,
        batch: ProposalBatch,
        panel: Sequence[CaseRecord],
        *,
        work_dir: str,
        panel_id: str,
    ) -> tuple[SpecialistEvidence, ...]:
        self.backend.evaluate_batch(batch.modules, panel, work_dir=work_dir, panel_id=panel_id)
        return tuple(
            self.backend.specialist_archive.research[module.semantic_id()]
            for module in batch.modules
        )

    def cegis_repairs(self, *, limit: int | None = None) -> ProposalBatch:
        started = time.perf_counter()
        rows = self.backend.consume_cegis_repairs()
        if limit is not None:
            rows = rows[:max(0, int(limit))]
        modules = []
        lineages = []
        duplicates = 0
        for parent_sid, module in rows:
            sid = module.semantic_id()
            if sid in self._seen:
                duplicates += 1; continue
            parent_lineage = self._lineages.get(parent_sid)
            if parent_lineage is None:
                lineage = SpecialistLineage.root(module, parent_controller=self.root_controller)
            else:
                lineage = parent_lineage.child(
                    module, operator="cegis_repair", source_margin_delta=0.0,
                )
            modules.append(module); lineages.append(lineage); self._seen.add(sid); self._lineages[sid] = lineage
        elapsed = time.perf_counter() - started
        module_rows = tuple(modules)
        return ProposalBatch(
            module_rows, tuple(lineages), diversity_report(module_rows), elapsed,
            len(module_rows) / max(elapsed, 1e-9), duplicates, {"cegis": elapsed},
        )

    def structural_ablations(self, *, limit: int) -> ProposalBatch:
        started = time.perf_counter(); modules = []; lineages = []; duplicates = 0
        for evidence in self.backend.specialist_archive.specialist_elites():
            parent = evidence.module
            parent_lineage = self._lineages.get(parent.semantic_id()) or SpecialistLineage.root(
                parent, parent_controller=self.root_controller,
            )
            candidates = []
            if len(parent.rules) > 1:
                candidates.extend(InvariantStrategicModule(parent.rules[:index] + parent.rules[index + 1:]) for index in range(len(parent.rules)))
            for index, rule in enumerate(parent.rules):
                if not rule.guard.atoms:
                    continue
                rules = list(parent.rules)
                from dataclasses import replace
                from tools.bkg4_search.invariant_module import LiftedGuard
                rules[index] = replace(rule, guard=LiftedGuard(rule.guard.atoms[:-1]))
                candidates.append(InvariantStrategicModule(tuple(rules)))
            for module in candidates:
                module.validate(); sid = module.semantic_id()
                if sid in self._seen:
                    duplicates += 1; continue
                lineage = parent_lineage.child(module, operator="structural_ablation", source_margin_delta=0.0)
                modules.append(module); lineages.append(lineage); self._seen.add(sid); self._lineages[sid] = lineage
                if len(modules) >= max(0, int(limit)):
                    break
            if len(modules) >= max(0, int(limit)):
                break
        elapsed = time.perf_counter() - started; rows = tuple(modules)
        return ProposalBatch(rows, tuple(lineages), diversity_report(rows), elapsed, len(rows) / max(elapsed, 1e-9), duplicates, {"structural_ablation": elapsed})


def _intervals(module: InvariantStrategicModule) -> dict[tuple[str, int], tuple[int | None, int | None]] | None:
    if len(module.rules) != 1 or module.rules[0].action.kind not in {"ROUTE_OVERRIDE", "COMMITTED_ROUTE"}:
        return None
    atoms = module.rules[0].guard.atoms
    if not atoms:
        return None
    intervals: dict[tuple[str, int], tuple[int | None, int | None]] = {}
    for atom in atoms:
        key = (atom.feature, int(atom.arg)); low, high = intervals.get(key, (None, None))
        value = int(atom.value)
        if atom.cmp == "GE": low = value if low is None else max(low, value)
        elif atom.cmp == "GT": low = value + 1 if low is None else max(low, value + 1)
        elif atom.cmp == "LE": high = value if high is None else min(high, value)
        elif atom.cmp == "LT": high = value - 1 if high is None else min(high, value - 1)
        elif atom.cmp == "EQ":
            low = value if low is None else max(low, value)
            high = value if high is None else min(high, value)
        else:
            return None
        intervals[key] = (low, high)
    return intervals


def compose_gated_specialists(
    first: InvariantStrategicModule,
    second: InvariantStrategicModule,
) -> InvariantStrategicModule:
    """Compose route specialists only when their public scalar guards are provably disjoint."""
    a = _intervals(first); b = _intervals(second)
    if a is None or b is None:
        raise ValueError("route specialist guards have unproven overlap")
    disjoint = False
    for key in set(a) & set(b):
        alo, ahi = a[key]; blo, bhi = b[key]
        if (ahi is not None and blo is not None and ahi < blo) or (
            bhi is not None and alo is not None and bhi < alo
        ):
            disjoint = True
            break
    if not disjoint:
        raise ValueError("route specialist guards overlap")
    module = InvariantStrategicModule((*first.rules, *second.rules))
    module.validate()
    return module


@dataclass(frozen=True)
class SpecialistFunnelResult:
    root_controller: str
    proposal_batch: ProposalBatch
    selected_lineages: Mapping[str, SpecialistLineage]
    ancestry_by_semantic_id: Mapping[str, tuple[SpecialistLineage, ...]]
    parent_evaluations: Mapping[str, StateEvaluation]
    screen_evidence: tuple[SpecialistEvidence, ...]
    labeled_evidence: tuple[SpecialistEvidence, ...]
    case_rows: Mapping[str, tuple[Mapping[str, object], ...]]
    source_case_rows: Mapping[str, tuple[Mapping[str, object], ...]]
    tier_counts: Mapping[str, int]
    timing_seconds: Mapping[str, float]
    exact_games: int
    screen_cases: int
    label_cases: int
    tier_thresholds: SpecialistTierThresholds
    screen_selected_count: int
    exact_labeled_count: int
    final_retained_count: int


def _calibrated_evidence(
    evidence: Sequence[SpecialistEvidence],
    panel: Sequence[CaseRecord],
    *,
    thresholds: SpecialistTierThresholds | None = None,
) -> tuple[tuple[SpecialistEvidence, ...], SpecialistTierThresholds]:
    if thresholds is None:
        positive = [
            row.mean_delta for item in evidence for row in item.cohorts
            if row.useful_groups >= 2 and row.behavior_groups >= 2 and row.mean_delta > 0
        ]
        thresholds = calibrate_specialist_thresholds(positive)
    cohort_by_group = {tape_group_id(case): str(case.family or "unclassified") for case in panel}
    return tuple(analyze_specialist(
        item.module, item.outcomes, cohort_by_group=cohort_by_group,
        activation_count=item.activation_count, action_change_count=item.action_change_count,
        thresholds=thresholds,
    ) for item in evidence), thresholds


def run_specialist_funnel(
    factory: SpecialistFactory,
    *,
    screen_panel: Sequence[CaseRecord],
    label_panel: Sequence[CaseRecord],
    proposal_count: int,
    retain_count: int,
    seed: int,
    work_dir: str,
    label_count: int | None = None,
    enable_cegis: bool = False,
) -> SpecialistFunnelResult:
    """Generate broadly, exact-screen once, quota-select, then exact-label once."""
    total_started = time.perf_counter()
    proposal = factory.propose(count=proposal_count, seed=seed)
    return run_proposal_funnel(
        factory, proposal=proposal, screen_panel=screen_panel, label_panel=label_panel,
        retain_count=retain_count, label_count=label_count, seed=seed, work_dir=work_dir,
        total_started=total_started, enable_cegis=enable_cegis,
    )


def run_proposal_funnel(
    factory: SpecialistFactory,
    *,
    proposal: ProposalBatch,
    screen_panel: Sequence[CaseRecord],
    label_panel: Sequence[CaseRecord],
    retain_count: int,
    seed: int,
    work_dir: str,
    label_count: int | None = None,
    total_started: float | None = None,
    enable_cegis: bool = False,
) -> SpecialistFunnelResult:
    total_started = time.perf_counter() if total_started is None else float(total_started)
    screen_started = time.perf_counter()
    previous_level = factory.backend.ladder.level
    if enable_cegis:
        factory.backend.ladder.level = max(6, previous_level)
    try:
        raw_screen_evidence = factory.screen(
            proposal, screen_panel, work_dir=f"{work_dir}/screen", panel_id="specialist/screen",
        )
    finally:
        factory.backend.ladder.level = previous_level
    screen_seconds = time.perf_counter() - screen_started
    screen_evidence, tier_thresholds = _calibrated_evidence(raw_screen_evidence, screen_panel)
    select_started = time.perf_counter()
    screen_limit = retain_count if label_count is None else max(retain_count, int(label_count))
    selected = select_screen_quota(screen_evidence, limit=screen_limit, seed=seed)
    select_seconds = time.perf_counter() - select_started
    selected_modules = tuple(row.module for row in selected)
    if len(selected_modules) != min(max(0, int(screen_limit)), len(screen_evidence)):
        raise RuntimeError("specialist funnel selection underfilled")
    selected_semantic_ids = {module.semantic_id() for module in selected_modules}
    lineage_by_sid = {
        lineage.module.semantic_id(): lineage for lineage in proposal.lineages
        if lineage.module.semantic_id() in selected_semantic_ids
    }
    ancestry_by_sid = {
        sid: factory.ancestry_for_lineage(lineage)
        for sid, lineage in lineage_by_sid.items()
    }
    parent_modules: dict[str, InvariantStrategicModule] = {}
    for lineage in lineage_by_sid.values():
        if lineage.parent_semantic_id and lineage.parent_semantic_id not in selected_semantic_ids:
            parent_modules.setdefault(
                lineage.parent_semantic_id,
                factory.module_for_semantic_id(lineage.parent_semantic_id),
            )
    evaluation_modules = tuple((*selected_modules, *parent_modules.values()))
    label_backend = InvariantSearchBackend(
        route_ids=factory.backend.route_ids,
        evaluator=factory.backend.evaluator,
        goals=factory.backend.goals,
    )
    label_started = time.perf_counter()
    label_round = label_backend.evaluate_batch(
        evaluation_modules, label_panel, work_dir=f"{work_dir}/label", panel_id="specialist/label",
    )
    label_seconds = time.perf_counter() - label_started
    empty = InvariantStrategicModule()
    baseline = label_round.evaluations[empty.semantic_id()]
    raw_labeled = tuple(label_backend.specialist_archive.research[module.semantic_id()] for module in selected_modules)
    exact_labeled, label_thresholds = _calibrated_evidence(raw_labeled, label_panel)
    labeled = select_corpus_quota(exact_labeled, limit=retain_count, seed=seed, lineages=lineage_by_sid)
    final_semantic_ids = {row.module.semantic_id() for row in labeled}
    case_rows_all = {
        module.semantic_id(): pair_case_rows(baseline, label_round.evaluations[module.semantic_id()])
        for module in selected_modules
    }
    source_case_rows_all = {
        module.semantic_id(): pair_relative_case_rows(
            label_round.evaluations[lineage_by_sid[module.semantic_id()].parent_semantic_id]
            if lineage_by_sid[module.semantic_id()].parent_semantic_id else baseline,
            label_round.evaluations[module.semantic_id()],
        )
        for module in selected_modules
    }
    parent_evaluations_all = {
        sid: label_round.evaluations[lineage.parent_semantic_id]
        for sid, lineage in lineage_by_sid.items()
        if lineage.parent_semantic_id is not None
    }
    exact_games = (len(proposal.modules) + 1) * len(screen_panel) + (len(evaluation_modules) + 1) * len(label_panel)
    return SpecialistFunnelResult(
        root_controller=factory.root_controller,
        proposal_batch=proposal,
        selected_lineages={sid: row for sid, row in lineage_by_sid.items() if sid in final_semantic_ids},
        ancestry_by_semantic_id={sid: row for sid, row in ancestry_by_sid.items() if sid in final_semantic_ids},
        parent_evaluations={sid: row for sid, row in parent_evaluations_all.items() if sid in final_semantic_ids},
        screen_evidence=screen_evidence,
        labeled_evidence=labeled,
        case_rows={sid: row for sid, row in case_rows_all.items() if sid in final_semantic_ids},
        source_case_rows={sid: row for sid, row in source_case_rows_all.items() if sid in final_semantic_ids},
        tier_counts=dict(sorted(Counter(row.tier for row in labeled).items())),
        timing_seconds={
            "proposal": proposal.generation_seconds,
            "screen_exact": screen_seconds,
            "quota_selection": select_seconds,
            "label_exact": label_seconds,
            "total": time.perf_counter() - total_started,
        },
        exact_games=exact_games,
        screen_cases=len(screen_panel),
        label_cases=len(label_panel),
        tier_thresholds=label_thresholds,
        screen_selected_count=len(selected_modules),
        exact_labeled_count=len(exact_labeled),
        final_retained_count=len(labeled),
    )


def combine_proposal_batches(*batches: ProposalBatch) -> ProposalBatch:
    modules = []; lineages = []; seen = set(); duplicates = 0
    for batch in batches:
        for module, lineage in zip(batch.modules, batch.lineages):
            sid = module.semantic_id()
            if sid in seen:
                duplicates += 1; continue
            seen.add(sid); modules.append(module); lineages.append(lineage)
        duplicates += batch.duplicate_rejections
    rows = tuple(modules)
    seconds = sum(batch.generation_seconds for batch in batches)
    return ProposalBatch(
        rows, tuple(lineages), diversity_report(rows), seconds,
        len(rows) / max(seconds, 1e-9), duplicates,
        dict(sorted(sum((Counter(batch.operator_timing_seconds) for batch in batches), Counter()).items())),
    )


def _source_quality(evidence: SpecialistEvidence, lineage: SpecialistLineage) -> str:
    if lineage.operator == "cegis_repair": return "cegis_repair"
    if lineage.operator == "local_structural_neighborhood": return "local_neighbor"
    if lineage.operator == "structural_ablation": return "structural_ablation"
    if evidence.tier == "GLOBAL_SAFE": return "global_safe"
    if evidence.tier == "ELITE_SPECIALIST": return "elite_specialist"
    if evidence.tier == "STRONG_SPECIALIST": return "strong_specialist"
    if evidence.tier == "BOUNDARY_SPECIALIST":
        return "boundary_specialist" if evidence.harmed_groups else "narrow_specialist"
    if evidence.tier == "HARD_NEGATIVE": return "hard_negative"
    return "inert"


def specialist_corpus_records(
    result: SpecialistFunnelResult,
    *,
    evaluator_abi: str,
    feature_registry: int,
) -> tuple[dict[str, object], ...]:
    records = []
    evidence_by_sid = {row.module.semantic_id(): row for row in result.labeled_evidence}
    for evidence in result.labeled_evidence:
        sid = evidence.module.semantic_id()
        lineage = result.selected_lineages[sid]
        composition = evidence.composition
        ancestry = result.ancestry_by_semantic_id[sid]
        parent_evidence = (
            None if lineage.parent_semantic_id is None
            else make_parent_evidence(ancestry, result.parent_evaluations[sid])
        )
        case_outcomes = list(result.case_rows[sid])
        source_outcomes = list(result.source_case_rows[sid])
        group_deltas = [float(row.margin_delta) for row in evidence.outcomes]
        mean_delta = sum(group_deltas) / len(group_deltas)
        worst_delta = min(group_deltas)
        w2l = sum(bool(row.get("w2l", False)) for row in case_outcomes)
        compatible = []
        conflicting = []
        for other_sid, other in evidence_by_sid.items():
            if other_sid == sid:
                continue
            target = compatible if composition.compatible_with(other.composition) else conflicting
            if len(target) < 8:
                target.append(other_sid)
        categories = {evidence.tier.lower()}
        if evidence.risk_profile == "mixed":
            categories.update(("high_risk_specialist", "generalization_boundary"))
        if evidence.target_cohorts:
            categories.add("counter_specialist")
        if evidence.support_breadth == "narrow" and evidence.useful_groups:
            categories.add("narrow_specialist")
        if evidence.l2w_groups:
            categories.add("l2w_specialist")
        elif evidence.useful_groups:
            categories.add("margin_specialist")
        if conflicting:
            categories.add("incompatible_specialist_pair")
        if lineage.operator == "cegis_repair": categories.add("cegis_repair")
        if lineage.operator == "local_structural_neighborhood": categories.add("local_neighbor")
        if lineage.operator == "structural_ablation": categories.add("structural_ablation")
        labels = {
            "exact_data_role": "specialist_discovery",
            "source_quality": _source_quality(evidence, lineage),
            "specialist_categories": sorted(categories),
            "regression": float(len(evidence.harmed_groups)),
            "flip": float(evidence.l2w_groups - w2l),
            "margin_target": mean_delta,
            "worst_target": worst_delta,
            "rank_target": max((row.mean_delta for row in evidence.cohorts), default=mean_delta),
            "safe": float(not evidence.harmed_groups),
            "l2w": float(evidence.l2w_groups),
            "w2l": float(w2l),
            "mean_delta": mean_delta,
            "specialist_tier": evidence.tier,
            "activation_count": evidence.activation_count,
            "action_change_count": evidence.action_change_count,
            "activation_topology": evidence.activation_topology,
            "support_breadth": evidence.support_breadth,
            "risk_profile": evidence.risk_profile,
            "structural_complexity": evidence.structural_complexity,
            "useful_groups": evidence.useful_groups,
            "l2w_groups": evidence.l2w_groups,
            "harmed_groups": list(evidence.harmed_groups),
            "target_cohorts": list(evidence.target_cohorts),
            "group_outcomes": [row.__dict__ for row in evidence.outcomes],
            "cohort_outcomes": [row.__dict__ for row in evidence.cohorts],
            "case_outcomes": case_outcomes,
            "source_relative_case_outcomes": source_outcomes,
            "source_relative_mean_delta": sum(float(row["margin_delta"]) for row in source_outcomes) / len(source_outcomes),
            "source_relative_l2w": sum(
                int(row["baseline_margin"] < 0 <= row["candidate_margin"])
                for row in source_outcomes
            ),
            "source_relative_w2l": sum(bool(row.get("w2l", False)) for row in source_outcomes),
            "composition": composition.__dict__,
            "composition_relationships": {
                "compatible": sorted(compatible), "conflicting": sorted(conflicting),
            },
            "lineage": {
                "lineage_schema_version": SPECIALIST_LINEAGE_SCHEMA_VERSION,
                "root_controller": lineage.root_controller,
                "root_semantic_id": lineage.root_semantic_id,
                "parent_semantic_id": lineage.parent_semantic_id,
                "operator": lineage.operator,
                "depth": lineage.depth,
                "source_margin_delta": lineage.source_margin_delta,
                "lineage_id": lineage.lineage_id,
                "source_goal_id": lineage.source_goal_id,
                "source_goal_path": list(lineage.source_goal_path),
                "structural_mechanisms": list(lineage.structural_mechanisms),
                "retrospective_goal_id": lineage.retrospective_goal_id,
                "parent_evidence": parent_evidence,
            },
        }
        records.append(make_corpus_record(
            evidence.module,
            labels=labels,
            provenance={
                "parent": result.root_controller,
                "controller": result.root_controller,
                "exact_evaluator": True,
                "data_role": "specialist_discovery",
                "candidate_abi": "B4IM1/v1",
                "evaluator_abi": str(evaluator_abi),
                "feature_registry": int(feature_registry),
                "screen_cases": result.screen_cases,
                "label_cases": result.label_cases,
            },
        ))
    return tuple(records)
