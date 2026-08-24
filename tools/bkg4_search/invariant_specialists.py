from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass
from statistics import median
from typing import Any, Mapping, Sequence

from tools.bkg4_search.edit_search.panels import CaseRecord
from tools.bkg4_search.information_state_io import registry_manifest
from tools.bkg4_search.invariant_groups import GroupOutcome, production_qualification
from tools.bkg4_search.invariant_module import GuardAtom, InvariantStrategicModule
from tools.bkg4_search.invariant_search import structural_skeleton


SPECIALIST_TIERS = (
    "GLOBAL_SAFE",
    "ELITE_SPECIALIST",
    "STRONG_SPECIALIST",
    "BOUNDARY_SPECIALIST",
    "HARD_NEGATIVE",
    "INERT",
)

SPECIALIST_QD_DIMENSIONS = (
    "mechanism_effect",
    "lifted_domain",
    "control_pattern",
    "action_family",
    "guard_family",
    "target_cohort",
    "activation_topology",
    "support_breadth",
    "risk_profile",
    "structural_complexity",
)


@dataclass(frozen=True)
class SpecialistTierThresholds:
    strong_mean_delta: float = 250.0
    elite_mean_delta: float = 1000.0


def calibrate_specialist_thresholds(
    repeat_positive_cohort_deltas: Sequence[float],
) -> SpecialistTierThresholds:
    """Fit robust tier boundaries from search-only repeat-support evidence."""
    values = sorted(float(value) for value in repeat_positive_cohort_deltas if float(value) > 0)
    if len(values) < 4:
        return SpecialistTierThresholds()

    def nearest_rank(fraction: float) -> float:
        index = max(0, min(len(values) - 1, int(fraction * len(values) + 0.999999) - 1))
        return values[index]

    strong = nearest_rank(0.25)
    elite = max(strong, nearest_rank(0.75))
    return SpecialistTierThresholds(strong, elite)


@dataclass(frozen=True)
class CohortEvidence:
    cohort: str
    groups: int
    behavior_groups: int
    useful_groups: int
    harmed_groups: int
    l2w_groups: int
    mean_delta: float
    median_delta: float
    total_positive_delta: float


@dataclass(frozen=True)
class CompositionMetadata:
    bb_reads: tuple[str, ...]
    bb_writes: tuple[str, ...]
    action_domains: tuple[str, ...]
    authorities: tuple[int, ...]
    feature_dependencies: tuple[str, ...]
    activation_guards: tuple[tuple[tuple[str, str, int, int], ...], ...]
    exit_guards: tuple[tuple[tuple[str, str, int, int], ...], ...]
    target_mechanisms: tuple[str, ...]
    side_effects: tuple[str, ...]
    action_signatures: tuple[tuple[Any, ...], ...]
    rule_bb_reads: tuple[tuple[str, ...], ...]
    rule_bb_writes: tuple[tuple[str, ...], ...]

    def compatibility_issues(self, other: "CompositionMetadata") -> tuple[str, ...]:
        issues: set[str] = set()
        for left_index, left_action in enumerate(self.action_signatures):
            left_guard = self.activation_guards[left_index]
            for right_index, right_action in enumerate(other.action_signatures):
                right_guard = other.activation_guards[right_index]
                if not _guards_may_overlap(left_guard, right_guard):
                    continue
                left_reads = set(self.rule_bb_reads[left_index])
                left_writes = set(self.rule_bb_writes[left_index])
                right_reads = set(other.rule_bb_reads[right_index])
                right_writes = set(other.rule_bb_writes[right_index])
                if left_writes & (right_reads | right_writes) or right_writes & left_reads:
                    issues.add("blackboard_read_write_hazard")
                left_domain = str(left_action[0]); right_domain = str(right_action[0])
                if left_domain == right_domain == "route":
                    issues.add("route_controller_conflict")
                elif left_domain == right_domain == "unit":
                    _, left_match, left_op, left_arg, left_qty = left_action
                    _, right_match, right_op, right_arg, right_qty = right_action
                    if (
                        (left_match == right_match and (left_op, left_arg, left_qty) != (right_op, right_arg, right_qty))
                        or left_op == right_match or right_op == left_match
                    ):
                        issues.add("unit_rewrite_conflict")
                elif left_domain == right_domain == "market":
                    _, left_op, left_item, left_qty, left_auth, left_priority = left_action
                    _, right_op, right_item, right_qty, right_auth, right_priority = right_action
                    same_target = (left_op, left_item) == (right_op, right_item)
                    same_effect = (left_qty, left_auth, left_priority) == (right_qty, right_auth, right_priority)
                    if same_target and not same_effect:
                        issues.add("market_intent_conflict")
        return tuple(sorted(issues))

    def compatible_with(self, other: "CompositionMetadata") -> bool:
        return not self.compatibility_issues(other)


def _guards_may_overlap(
    first: Sequence[tuple[str, str, int, int]],
    second: Sequence[tuple[str, str, int, int]],
) -> bool:
    """Conservatively prove whether two integer conjunctions can be active together."""
    bounds: dict[tuple[str, int], list[Any]] = {}
    for feature, cmp, arg, value in (*first, *second):
        key = (str(feature), int(arg))
        low, high, excluded = bounds.setdefault(key, [None, None, set()])
        value = int(value)
        if cmp == "EQ":
            low = value if low is None else max(low, value)
            high = value if high is None else min(high, value)
        elif cmp == "NE":
            excluded.add(value)
        elif cmp == "GE":
            low = value if low is None else max(low, value)
        elif cmp == "GT":
            low = value + 1 if low is None else max(low, value + 1)
        elif cmp == "LE":
            high = value if high is None else min(high, value)
        elif cmp == "LT":
            high = value - 1 if high is None else min(high, value - 1)
        bounds[key] = [low, high, excluded]
    for low, high, excluded in bounds.values():
        if low is not None and high is not None and low > high:
            return False
        if low is not None and high is not None and low == high and low in excluded:
            return False
    return True


@dataclass(frozen=True)
class SpecialistEvidence:
    module: InvariantStrategicModule
    tier: str
    outcomes: tuple[GroupOutcome, ...]
    cohorts: tuple[CohortEvidence, ...]
    target_cohorts: tuple[str, ...]
    harmed_groups: tuple[str, ...]
    activation_count: int
    action_change_count: int
    behavior_groups: int
    useful_groups: int
    l2w_groups: int
    production_safe: bool
    activation_topology: str
    support_breadth: str
    risk_profile: str
    structural_complexity: int
    structural_skeleton: tuple[Any, ...]
    composition: CompositionMetadata
    guard_families: tuple[str, ...]

    def niche_key(self) -> tuple[Any, ...]:
        rules = self.module.rules
        return (
            tuple(rule.sketch.effect_class for rule in rules),
            tuple(rule.domain for rule in rules),
            tuple(rule.sketch.control_pattern for rule in rules),
            tuple(rule.action.kind for rule in rules),
            self.guard_families,
            self.target_cohorts,
            self.activation_topology,
            self.support_breadth,
            self.risk_profile,
            self.structural_complexity,
        )

    def quality_key(self) -> tuple[Any, ...]:
        tier_order = {name: i for i, name in enumerate(SPECIALIST_TIERS)}
        best = max((row.mean_delta for row in self.cohorts), default=0.0)
        return (
            tier_order[self.tier],
            -self.l2w_groups,
            -self.useful_groups,
            -best,
            len(self.harmed_groups),
            self.module.semantic_id(),
        )


def _cohort_rows(
    outcomes: Sequence[GroupOutcome],
    cohort_by_group: Mapping[str, str],
) -> tuple[CohortEvidence, ...]:
    buckets: dict[str, list[GroupOutcome]] = {}
    for outcome in outcomes:
        cohort = str(cohort_by_group.get(outcome.group_id, "unclassified"))
        buckets.setdefault(cohort, []).append(outcome)
    rows = []
    for cohort, group_rows in sorted(buckets.items()):
        deltas = [float(row.margin_delta) for row in group_rows]
        rows.append(CohortEvidence(
            cohort=cohort,
            groups=len(group_rows),
            behavior_groups=sum(bool(row.behavior_changed) for row in group_rows),
            useful_groups=sum(bool(row.useful) for row in group_rows),
            harmed_groups=sum(bool(row.new_loss or row.severe_regression) for row in group_rows),
            l2w_groups=sum(int(row.l2w) > 0 for row in group_rows),
            mean_delta=sum(deltas) / len(deltas),
            median_delta=float(median(deltas)),
            total_positive_delta=sum(max(0.0, value) for value in deltas),
        ))
    return tuple(rows)


def analyze_specialist(
    module: InvariantStrategicModule,
    outcomes: Sequence[GroupOutcome],
    *,
    cohort_by_group: Mapping[str, str],
    activation_count: int,
    action_change_count: int,
    force_tier: str | None = None,
    thresholds: SpecialistTierThresholds | None = None,
) -> SpecialistEvidence:
    """Describe targeted value without converting unrelated harm into hard rejection."""
    module.validate()
    rows = tuple(outcomes)
    if not rows:
        raise ValueError("specialist analysis requires grouped outcomes")
    cohorts = _cohort_rows(rows, cohort_by_group)
    useful_groups = sum(bool(row.useful) for row in rows)
    behavior_groups = sum(bool(row.behavior_changed) for row in rows)
    l2w_groups = sum(int(row.l2w) > 0 for row in rows)
    harmed = tuple(sorted(row.group_id for row in rows if row.new_loss or row.severe_regression))
    production_safe = production_qualification(rows).production
    thresholds = thresholds or SpecialistTierThresholds()
    supported = tuple(sorted(
        row.cohort for row in cohorts
        if row.useful_groups >= 2 and row.behavior_groups >= 2
        and row.mean_delta >= thresholds.strong_mean_delta
    ))
    elite = tuple(
        row.cohort for row in cohorts
        if row.useful_groups >= 2
        and row.behavior_groups >= 2
        and (row.mean_delta >= thresholds.elite_mean_delta or row.l2w_groups > 0)
    )
    if force_tier is not None:
        if force_tier not in SPECIALIST_TIERS:
            raise ValueError(f"unknown specialist tier {force_tier}")
        tier = force_tier
    elif int(action_change_count) <= 0 or behavior_groups == 0:
        tier = "INERT"
    elif production_safe and useful_groups >= 2:
        tier = "GLOBAL_SAFE"
    elif elite:
        tier = "ELITE_SPECIALIST"
    elif supported:
        tier = "STRONG_SPECIALIST"
    elif useful_groups > 0:
        tier = "BOUNDARY_SPECIALIST"
    else:
        tier = "HARD_NEGATIVE"
    targets = tuple(sorted(elite or supported or (
        row.cohort for row in cohorts if row.useful_groups > 0
    )))
    changed_fraction = behavior_groups / len(rows)
    activation_topology = "none" if behavior_groups == 0 else "sparse" if changed_fraction < 0.34 else "mixed" if changed_fraction < 0.75 else "broad"
    support_fraction = useful_groups / len(rows)
    support_breadth = "none" if useful_groups == 0 else "narrow" if support_fraction < 0.25 else "regional" if support_fraction < 0.60 else "broad"
    risk_profile = "mixed" if harmed and useful_groups else "safe" if not harmed else "harmful"
    complexity = len(module.rules) + sum(len(rule.guard.atoms) + len(rule.exit_guard.atoms) for rule in module.rules)
    return SpecialistEvidence(
        module=module,
        tier=tier,
        outcomes=rows,
        cohorts=cohorts,
        target_cohorts=targets,
        harmed_groups=harmed,
        activation_count=max(0, int(activation_count)),
        action_change_count=max(0, int(action_change_count)),
        behavior_groups=behavior_groups,
        useful_groups=useful_groups,
        l2w_groups=l2w_groups,
        production_safe=production_safe,
        activation_topology=activation_topology,
        support_breadth=support_breadth,
        risk_profile=risk_profile,
        structural_complexity=complexity,
        structural_skeleton=structural_skeleton(module),
        composition=composition_metadata(module),
        guard_families=tuple(sorted({
            _guard_family(atom.feature)
            for rule in module.rules for atom in (*rule.guard.atoms, *rule.exit_guard.atoms)
        })),
    )


class SpecialistArchive:
    """Specialist QD map alongside, not instead of, deployment-safe qualification."""

    def __init__(self) -> None:
        self.research: dict[str, SpecialistEvidence] = {}
        self.elites: dict[tuple[Any, ...], SpecialistEvidence] = {}

    def consider(self, evidence: SpecialistEvidence) -> bool:
        sid = evidence.module.semantic_id()
        self.research[sid] = evidence
        if evidence.tier in {"INERT", "HARD_NEGATIVE"}:
            return False
        key = evidence.niche_key()
        incumbent = self.elites.get(key)
        if incumbent is None or evidence.quality_key() < incumbent.quality_key():
            self.elites[key] = evidence
            return True
        return False

    def specialist_elites(self) -> tuple[SpecialistEvidence, ...]:
        return tuple(self.elites[key] for key in sorted(self.elites, key=repr))


@dataclass(frozen=True)
class DiversityReport:
    programs: int
    semantic_ids: int
    structural_skeletons: int
    mechanisms: int
    action_families: int
    controls: int
    guard_families: int
    activation_profiles: int
    top1_skeleton_share: float
    top10_skeleton_share: float
    top_action_family_share: float
    top_mechanism_share: float


def _guard_family(feature: str) -> str:
    if feature.startswith("MARKET_"): return "market"
    if feature.startswith("OWN_TILE_"): return "bound_tile"
    if feature.startswith("OPP_TILE_"): return "opponent_tile"
    if feature.startswith("OWN_SHED") or feature.startswith("OWN_INVENTORY") or feature.startswith("OWN_SEED"): return "inventory"
    if feature.startswith("OWN_") or feature.startswith("OPP_"): return "economy"
    if feature in {"STEP", "DAY", "HOUR"}: return "phase"
    if feature.startswith("SHOP_"): return "town"
    return feature.lower()


def diversity_report(
    modules: Sequence[InvariantStrategicModule],
    *,
    activation_profiles: Sequence[str] = (),
) -> DiversityReport:
    rows = tuple(modules)
    if len(activation_profiles) not in {0, len(rows)}:
        raise ValueError("activation profile count must match modules")
    skeleton_counts = Counter(repr(structural_skeleton(module)) for module in rows)
    actions = Counter(rule.action.kind for module in rows for rule in module.rules)
    mechanisms = Counter(rule.sketch.effect_class for module in rows for rule in module.rules)
    controls = {rule.sketch.control_pattern for module in rows for rule in module.rules}
    guards = {_guard_family(atom.feature) for module in rows for rule in module.rules for atom in (*rule.guard.atoms, *rule.exit_guard.atoms)}
    total = max(1, len(rows))
    action_total = max(1, sum(actions.values()))
    mechanism_total = max(1, sum(mechanisms.values()))
    return DiversityReport(
        programs=len(rows),
        semantic_ids=len({module.semantic_id() for module in rows}),
        structural_skeletons=len(skeleton_counts),
        mechanisms=len(mechanisms),
        action_families=len(actions),
        controls=len(controls),
        guard_families=len(guards),
        activation_profiles=len(set(str(value) for value in activation_profiles)),
        top1_skeleton_share=(max(skeleton_counts.values(), default=0) / total),
        top10_skeleton_share=(sum(sorted(skeleton_counts.values(), reverse=True)[:10]) / total),
        top_action_family_share=(max(actions.values(), default=0) / action_total),
        top_mechanism_share=(max(mechanisms.values(), default=0) / mechanism_total),
    )


def composition_metadata(module: InvariantStrategicModule) -> CompositionMetadata:
    module = module.canonicalized()
    reads: set[str] = set()
    writes: set[str] = set()
    domains: set[str] = set()
    authorities: set[int] = set()
    features: set[str] = set()
    guards = []
    exits = []
    effects: set[str] = set()
    side_effects: set[str] = set()
    action_signatures = []
    rule_reads = []
    rule_writes = []
    for rule in module.rules:
        entry = tuple((atom.feature, atom.cmp, int(atom.arg), int(atom.value)) for atom in rule.guard.atoms)
        exit_ = tuple((atom.feature, atom.cmp, int(atom.arg), int(atom.value)) for atom in rule.exit_guard.atoms)
        features.update(atom.feature for atom in (*rule.guard.atoms, *rule.exit_guard.atoms))
        guards.append(entry); exits.append(exit_)
        effects.add(rule.sketch.effect_class)
        kind = rule.action.kind
        local_reads: set[str] = set()
        local_writes: set[str] = set()
        if kind in {"ROUTE_OVERRIDE", "COMMITTED_ROUTE"}:
            domains.add("route")
            local_writes.add("strategic_route")
            side_effects.add("route_selection")
            action_signatures.append(("route", kind, int(rule.action.route_id)))
            if kind == "COMMITTED_ROUTE":
                side_effects.add("persistent_mode")
        elif kind == "UNIT_REWRITE":
            domains.add("unit")
            side_effects.add("unit_action_replacement")
            action_signatures.append((
                "unit", int(rule.action.match_op), int(rule.action.unit_op),
                int(rule.action.unit_arg), int(rule.action.quantity),
            ))
        elif kind == "MARKET_INTENT":
            domains.add("market")
            authorities.add(int(rule.action.authority))
            side_effects.add("market_order_prepend")
            action_signatures.append((
                "market", int(rule.action.market_op), int(rule.action.item), int(rule.action.quantity),
                int(rule.action.authority), int(rule.action.priority),
            ))
        else:
            action_signatures.append(("noop", kind))
        reads.update(local_reads); writes.update(local_writes)
        rule_reads.append(tuple(sorted(local_reads))); rule_writes.append(tuple(sorted(local_writes)))
    return CompositionMetadata(
        tuple(sorted(reads)), tuple(sorted(writes)), tuple(sorted(domains)),
        tuple(sorted(authorities)), tuple(sorted(features)), tuple(guards), tuple(exits),
        tuple(sorted(effects)), tuple(sorted(side_effects)), tuple(action_signatures),
        tuple(rule_reads), tuple(rule_writes),
    )


def select_screen_quota(
    evidences: Sequence[SpecialistEvidence],
    *,
    limit: int,
    seed: int,
) -> tuple[SpecialistEvidence, ...]:
    """Round-robin specialist classes, then fill by new skeleton and targeted quality."""
    limit = max(0, int(limit))
    if limit == 0:
        return ()
    buckets: dict[str, list[SpecialistEvidence]] = {name: [] for name in SPECIALIST_TIERS}
    for evidence in evidences:
        buckets[evidence.tier].append(evidence)
    for rows in buckets.values():
        rows.sort(key=lambda row: row.quality_key())
    selected: list[SpecialistEvidence] = []
    seen: set[str] = set()
    tier_fractions = {
        "GLOBAL_SAFE": 0.10,
        "ELITE_SPECIALIST": 0.25,
        "STRONG_SPECIALIST": 0.10,
        "BOUNDARY_SPECIALIST": 0.20,
        "HARD_NEGATIVE": 0.20,
        "INERT": 0.15,
    }
    for tier in SPECIALIST_TIERS:
        target = max(1, int(round(limit * tier_fractions[tier])))
        rows = buckets[tier]
        chosen = rows[:min(target, len(rows), max(0, limit - len(selected)))]
        selected.extend(chosen)
        seen.update(row.module.semantic_id() for row in chosen)
        del rows[:len(chosen)]
    remaining = [row for rows in buckets.values() for row in rows if row.module.semantic_id() not in seen]
    remaining.sort(key=lambda row: (
        row.quality_key(),
        hashlib.sha256(f"{int(seed)}\0{row.module.semantic_id()}".encode()).hexdigest(),
    ))

    def select_representatives(key_fn) -> None:
        covered = {key_fn(row) for row in selected}
        for row in remaining:
            key = key_fn(row)
            if key is None or key in covered or row.module.semantic_id() in seen:
                continue
            selected.append(row); seen.add(row.module.semantic_id()); covered.add(key)
            if len(selected) >= limit:
                return

    if len(selected) < limit:
        select_representatives(lambda row: tuple(rule.action.kind for rule in row.module.rules))
    if len(selected) < limit:
        select_representatives(lambda row: tuple(rule.sketch.effect_class for rule in row.module.rules))
    if len(selected) < limit:
        select_representatives(lambda row: row.target_cohorts or None)
    if len(selected) < limit:
        select_representatives(lambda row: repr(row.structural_skeleton))
    if len(selected) < limit:
        fill = [row for row in remaining if row.module.semantic_id() not in seen]
        selected.extend(fill[:limit - len(selected)])
    return tuple(selected)


FINAL_CORPUS_TIER_FRACTIONS = {
    "ELITE_SPECIALIST": 0.40,
    "STRONG_SPECIALIST": 0.25,
    "GLOBAL_SAFE": 0.08,
    "BOUNDARY_SPECIALIST": 0.12,
    "HARD_NEGATIVE": 0.12,
    "INERT": 0.03,
}


def _training_quality_key(evidence: SpecialistEvidence) -> tuple[Any, ...]:
    """Rank information value within a tier; lower is better."""
    best = max((row.mean_delta for row in evidence.cohorts), default=0.0)
    effect = max((abs(float(row.margin_delta)) for row in evidence.outcomes), default=0.0)
    failures = sum(bool(row.new_loss or row.severe_regression) for row in evidence.outcomes)
    if evidence.tier == "HARD_NEGATIVE":
        value = (-failures, -evidence.behavior_groups, -effect, -evidence.action_change_count)
    elif evidence.tier == "INERT":
        value = (0, 0, 0, 0)
    else:
        value = (-evidence.l2w_groups, -evidence.useful_groups, -best, len(evidence.harmed_groups))
    return (*value, evidence.module.semantic_id())


def _diverse_tier_order(
    rows: Sequence[SpecialistEvidence],
    lineages: Mapping[str, SpecialistLineage] | None = None,
) -> list[SpecialistEvidence]:
    """Round-robin structural skeletons while preserving quality inside each skeleton."""
    buckets: dict[str, list[SpecialistEvidence]] = {}
    for row in rows:
        buckets.setdefault(repr(row.structural_skeleton), []).append(row)
    for skeleton, bucket in tuple(buckets.items()):
        lineage_buckets: dict[tuple[Any, ...], list[SpecialistEvidence]] = {}
        for row in bucket:
            lineage = (lineages or {}).get(row.module.semantic_id())
            key = (lineage.operator, lineage.source_goal_id, lineage.depth) if lineage else ("unknown", None, 0)
            lineage_buckets.setdefault(key, []).append(row)
        for lineage_rows in lineage_buckets.values():
            lineage_rows.sort(key=_training_quality_key)
        bucket = []
        offset = 0
        while True:
            added = False
            for key in sorted(lineage_buckets, key=repr):
                if offset < len(lineage_buckets[key]):
                    bucket.append(lineage_buckets[key][offset]); added = True
            if not added:
                break
            offset += 1
        buckets[skeleton] = bucket
    keys = sorted(buckets, key=lambda key: (_training_quality_key(buckets[key][0]), key))
    ordered: list[SpecialistEvidence] = []
    offset = 0
    while True:
        added = False
        for key in keys:
            if offset < len(buckets[key]):
                ordered.append(buckets[key][offset])
                added = True
        if not added:
            return ordered
        offset += 1


def select_corpus_quota(
    evidences: Sequence[SpecialistEvidence],
    *,
    limit: int,
    seed: int,
    lineages: Mapping[str, SpecialistLineage] | None = None,
) -> tuple[SpecialistEvidence, ...]:
    """Select exact-labeled training rows by tier, information value, and diversity."""
    del seed  # Selection is deterministic; retained for a stable public API.
    limit = max(0, int(limit))
    if limit == 0:
        return ()
    unique: dict[str, SpecialistEvidence] = {}
    for evidence in evidences:
        sid = evidence.module.semantic_id()
        previous = unique.get(sid)
        if previous is None or _training_quality_key(evidence) < _training_quality_key(previous):
            unique[sid] = evidence
    target = min(limit, len(unique))
    buckets = {
        tier: _diverse_tier_order([row for row in unique.values() if row.tier == tier], lineages)
        for tier in SPECIALIST_TIERS
    }
    selected: list[SpecialistEvidence] = []
    seen: set[str] = set()
    for tier in FINAL_CORPUS_TIER_FRACTIONS:
        quota = min(len(buckets[tier]), int(round(target * FINAL_CORPUS_TIER_FRACTIONS[tier])))
        for row in buckets[tier][:min(quota, target - len(selected))]:
            selected.append(row)
            seen.add(row.module.semantic_id())
    if len(selected) < target:
        remaining = [
            row for tier in FINAL_CORPUS_TIER_FRACTIONS for row in buckets[tier]
            if row.module.semantic_id() not in seen
        ]
        for row in remaining:
            selected.append(row)
            if len(selected) >= target:
                break
    return tuple(selected)


@dataclass(frozen=True)
class SpecialistLineage:
    module: InvariantStrategicModule
    root_controller: str
    root_semantic_id: str
    parent_semantic_id: str | None
    operator: str
    depth: int
    source_margin_delta: float
    lineage_id: str
    source_goal_id: str | None
    source_goal_path: tuple[str, ...]
    structural_mechanisms: tuple[str, ...]
    retrospective_goal_id: str | None

    @staticmethod
    def _mechanisms(module: InvariantStrategicModule) -> tuple[str, ...]:
        return tuple(sorted({
            f"{rule.sketch.effect_class}:{rule.action.kind}" for rule in module.rules
        }))

    @classmethod
    def root(
        cls,
        module: InvariantStrategicModule,
        *,
        parent_controller: str,
        operator: str = "root",
        source_goal_id: str | None = None,
        source_goal_path: Sequence[str] | None = None,
        structural_mechanisms: Sequence[str] | None = None,
        retrospective_goal_id: str | None = None,
    ) -> "SpecialistLineage":
        module.validate()
        semantic_id = module.semantic_id()
        lineage_id = hashlib.sha256(f"{parent_controller}\0{semantic_id}".encode()).hexdigest()
        mechanisms = tuple(structural_mechanisms) if structural_mechanisms is not None else cls._mechanisms(module)
        goal_path = tuple(str(value) for value in source_goal_path) if source_goal_path is not None else (() if source_goal_id is None else (str(source_goal_id),))
        if source_goal_id is not None and (not goal_path or goal_path[-1] != str(source_goal_id)):
            raise ValueError("source goal path must end at source_goal_id")
        direct_goal = goal_path[-1] if goal_path else None
        return cls(
            module, str(parent_controller), semantic_id, None, str(operator), 0, 0.0, lineage_id,
            direct_goal, goal_path, mechanisms, retrospective_goal_id,
        )

    def child(
        self,
        module: InvariantStrategicModule,
        *,
        operator: str,
        source_margin_delta: float,
        source_goal_id: str | None = None,
        source_goal_path: Sequence[str] | None = None,
        structural_mechanisms: Sequence[str] | None = None,
        retrospective_goal_id: str | None = None,
    ) -> "SpecialistLineage":
        module.validate()
        lineage_id = hashlib.sha256(
            f"{self.lineage_id}\0{operator}\0{module.semantic_id()}".encode()
        ).hexdigest()
        mechanisms = tuple(structural_mechanisms) if structural_mechanisms is not None else self._mechanisms(module)
        if source_goal_path is not None:
            goal_path = tuple(str(value) for value in source_goal_path)
        elif source_goal_id is not None:
            goal_path = (str(source_goal_id),)
        else:
            goal_path = ()
        if source_goal_id is not None and (not goal_path or goal_path[-1] != str(source_goal_id)):
            raise ValueError("source goal path must end at source_goal_id")
        direct_goal = goal_path[-1] if goal_path else None
        return SpecialistLineage(
            module, self.root_controller, self.root_semantic_id, self.module.semantic_id(), str(operator),
            self.depth + 1, float(source_margin_delta), lineage_id,
            direct_goal, goal_path, mechanisms,
            self.retrospective_goal_id if retrospective_goal_id is None else retrospective_goal_id,
        )


@dataclass(frozen=True)
class CoverageRow:
    domain: str
    mechanisms: tuple[str, ...]
    primitive: str
    representable: bool
    searchable: bool
    cuda_executable: bool
    distillable: bool
    emitter_exists: bool
    meaningful_observed: bool


@dataclass(frozen=True)
class BoundaryGuardEvidence:
    atom: GuardAtom
    useful_groups: int
    harmed_groups: int
    useful_hits: int
    harmed_hits: int
    score: tuple[int, int, int, int, str]


def specialist_boundary_guards(
    outcomes: Sequence[GroupOutcome],
    panel: Sequence[CaseRecord],
    *,
    limit: int = 8,
) -> tuple[BoundaryGuardEvidence, ...]:
    """Find public-state guards separating repeated help from known downside groups."""
    outcome_by_group = {row.group_id: row for row in outcomes}
    manifest = tuple(registry_manifest())
    allowed_prefixes = ("MARKET_", "OWN_SHED", "OWN_SEED", "SHOP_", "OWN_QUADRANT_", "OPP_QUADRANT_")
    allowed_scalars = {
        "DAY", "HOUR", "OWN_MONEY", "OPP_MONEY", "OWN_UNITS", "OPP_UNITS",
        "OWN_LAND", "OPP_LAND", "OWN_HIRES_TODAY", "OPP_HIRES_TODAY", "SHOP_COUNT",
    }
    feature_positions = [
        (index, feature, args)
        for index, (feature, args) in enumerate(manifest)
        if feature in allowed_scalars or feature.startswith(allowed_prefixes)
    ]
    samples: dict[tuple[str, int], dict[str, list[int]]] = {}
    for case in panel:
        outcome = outcome_by_group.get(str(case.tape_path))
        if outcome is None or not (outcome.useful or outcome.new_loss or outcome.severe_regression or outcome.margin_delta < 0):
            continue
        if case.public_context_registry_width != len(manifest) or case.public_context_width < len(manifest):
            continue
        frames = max(1, len(case.context_probe_steps))
        for position, feature, args in feature_positions:
            arg = int(args[0]) if args else 0
            key = (feature, arg)
            group_values = samples.setdefault(key, {}).setdefault(outcome.group_id, [])
            group_values.extend(
                int(case.public_context_values[frame * case.public_context_width + position])
                for frame in range(frames)
            )
    evidence = []
    for (feature, arg), groups in samples.items():
        values = sorted({value for rows in groups.values() for value in rows})
        if len(values) < 2:
            continue
        thresholds = sorted({values[0], values[-1], *((a + b) // 2 for a, b in zip(values, values[1:]))})
        for comparator in ("GE", "LE"):
            for threshold in thresholds:
                def active(rows: Sequence[int]) -> bool:
                    if comparator == "GE": return any(value >= threshold for value in rows)
                    return any(value <= threshold for value in rows)
                useful = {
                    group for group, rows in groups.items()
                    if outcome_by_group[group].useful and active(rows)
                }
                harmed = {
                    group for group, rows in groups.items()
                    if (outcome_by_group[group].new_loss or outcome_by_group[group].severe_regression or outcome_by_group[group].margin_delta < 0)
                    and active(rows)
                }
                if len(useful) < 2:
                    continue
                useful_hits = sum(active(rows) for group, rows in groups.items() if outcome_by_group[group].useful)
                harmed_hits = sum(active(rows) for group, rows in groups.items() if group in harmed)
                score = (len(harmed), -len(useful), harmed_hits, -useful_hits, f"{feature}:{arg}:{comparator}:{threshold}")
                evidence.append(BoundaryGuardEvidence(
                    GuardAtom(feature, comparator, threshold, arg=arg), len(useful), len(harmed),
                    useful_hits, harmed_hits, score,
                ))
    evidence.sort(key=lambda row: row.score)
    unique: dict[tuple[str, str, int], BoundaryGuardEvidence] = {}
    for row in evidence:
        key = (row.atom.feature, row.atom.cmp, row.atom.arg)
        unique.setdefault(key, row)
        if len(unique) >= max(0, int(limit)):
            break
    return tuple(unique.values())


def strategic_coverage_matrix(
    route_ids: Sequence[int],
    *,
    observed_domains: Sequence[str] = (),
) -> tuple[CoverageRow, ...]:
    """Honest v1 capability map; observation is evidence, not inferred from expressibility."""
    if not tuple(route_ids):
        raise ValueError("coverage matrix requires parent routes")
    specs = (
        ("opening", ("phase routing", "capital cadence"), "ROUTE_OVERRIDE"),
        ("market", ("liquidation", "financing", "shared price"), "MARKET_INTENT"),
        ("workforce", ("hire cadence", "labor recovery"), "MARKET_INTENT+UNIT_REWRITE"),
        ("land", ("unlock timing", "capital authorization"), "MARKET_INTENT"),
        ("crops", ("water rescue", "fertilize priority", "harvest routing"), "UNIT_REWRITE+ROUTE_OVERRIDE"),
        ("animals", ("feed rescue", "care suppression", "fertilizer priority"), "UNIT_REWRITE"),
        ("inventory", ("overflow liquidation", "pickup/drop/place routing"), "MARKET_INTENT+ROUTE_OVERRIDE"),
        ("routing", ("phase mode", "hysteresis", "fallback"), "ROUTE_OVERRIDE+COMMITTED_ROUTE"),
        ("recovery", ("maintenance debt", "resource invariant"), "UNIT_REWRITE+COMMITTED_ROUTE"),
        ("engine_mechanics", ("price-floor sale", "daily refresh", "order cap"), "MARKET_INTENT+ROUTE_OVERRIDE"),
    )
    observed = {str(value) for value in observed_domains}
    return tuple(CoverageRow(
        domain=domain,
        mechanisms=mechanisms,
        primitive=primitive,
        representable=True,
        searchable=True,
        cuda_executable=True,
        distillable=True,
        emitter_exists=True,
        meaningful_observed=domain in observed,
    ) for domain, mechanisms, primitive in specs)
