from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Iterable, Mapping, Sequence

from tools.bkg4_search.edit_search.fitness import StateEvaluation
from tools.bkg4_search.edit_search.panels import CaseRecord
from tools.bkg4_search.information_state_io import registry_manifest
from tools.bkg4_search.invariant_groups import (
    GroupOutcome,
    ProductionQualification,
    RobustGroupFitness,
    group_outcomes_from_case_rows,
    production_qualification,
    solve_cross_tape_threshold,
    tape_group_id,
)
from tools.bkg4_search.invariant_module import (
    MAX_GUARDS_TOTAL,
    MAX_RULES,
    AbstractAction,
    GuardAtom,
    InvariantRule,
    InvariantStrategicModule,
    LiftedGuard,
    ModuleSketch,
)
from tools.bkg4_search.module_codec import ModuleGenome


CEGIS_FAILURES = (
    "guard_too_broad",
    "wrong_action_effect",
    "unstable_constant",
    "resource_invariant",
    "action_conflict",
    "lifecycle_missing",
)

_REGISTRY_MANIFEST = tuple(registry_manifest())
_REGISTRY_POS = {entry: index for index, entry in enumerate(_REGISTRY_MANIFEST)}
_REGISTRY_ARG_FEATURES = {
    "MARKET_PRICE", "MARKET_INVENTORY", "OWN_SHED", "OWN_SEED",
    "SHOP_AT", "SHOP_SEEN", "OWN_QUADRANT_UNLOCKED", "OPP_QUADRANT_UNLOCKED",
    "OWN_WORKER_X", "OWN_WORKER_Y", "OPP_WORKER_X", "OPP_WORKER_Y",
    "OWN_WORKER_TILE", "OPP_WORKER_TILE",
}


@dataclass(frozen=True)
class SearchGoal:
    """Mechanism-level synthesis target. It deliberately contains no case/opponent identity."""

    goal_id: str
    effect_class: str
    domain: str
    action_family: str
    guard_features: tuple[str, ...]
    description: str

    def __post_init__(self) -> None:
        text = " ".join((self.goal_id, self.description, *self.guard_features)).lower()
        for forbidden in ("opponent", "tape_id", "replay_path", "episode_seed", "team_identity"):
            if forbidden in text:
                raise ValueError(f"search goal contains forbidden identity term {forbidden!r}")


@dataclass(frozen=True)
class SemanticDescriptor:
    effects: tuple[str, ...]
    domains: tuple[str, ...]
    controls: tuple[str, ...]
    guard_families: tuple[tuple[str, ...], ...]
    actions: tuple[str, ...]
    activation_topology: str = "unknown"

    def key(self) -> tuple[Any, ...]:
        return (self.effects, self.domains, self.controls, self.guard_families, self.actions, self.activation_topology)


@dataclass(frozen=True)
class SearchElite:
    module: InvariantStrategicModule
    descriptor: SemanticDescriptor
    outcomes: tuple[GroupOutcome, ...]
    fitness: RobustGroupFitness
    qualification: ProductionQualification


@dataclass(frozen=True)
class CapabilityEvent:
    level: int
    escalated: bool
    reason: str


@dataclass(frozen=True)
class SearchRound:
    attempted: int
    unique: int
    duplicate_rate: float
    useful: int
    production_qualified: int
    archive_insertions: int
    unsafe: int
    exact_candidate_cases: int
    capability: CapabilityEvent
    evaluations: Mapping[str, StateEvaluation]


@dataclass(frozen=True)
class SearchRunResult:
    rounds: tuple[SearchRound, ...]
    archive: "InvariantSearchArchive"
    attempted_semantics: int
    exact_candidate_cases: int


@dataclass(frozen=True)
class InternalCrossFitSplit:
    search: tuple[CaseRecord, ...]
    selection: tuple[CaseRecord, ...]
    search_tapes: tuple[str, ...]
    selection_tapes: tuple[str, ...]


@dataclass(frozen=True)
class TransferableElite:
    module: InvariantStrategicModule
    descriptor: SemanticDescriptor
    search_outcomes: tuple[GroupOutcome, ...]
    selection_outcomes: tuple[GroupOutcome, ...]
    combined_fitness: RobustGroupFitness
    combined_qualification: ProductionQualification


@dataclass(frozen=True)
class CrossFitSearchResult:
    split: InternalCrossFitSplit
    search: SearchRunResult
    frozen_semantic_ids: tuple[str, ...]
    selection_evaluations: Mapping[str, StateEvaluation]
    transferable_elites: tuple[TransferableElite, ...]
    exact_candidate_cases: int


def internal_crossfit_split(
    panel: Sequence[CaseRecord],
    *,
    selection_fraction: float = 0.25,
    seed: int,
) -> InternalCrossFitSplit:
    """Make a deterministic tape-disjoint search/one-shot-selection partition."""
    rows = tuple(panel)
    bad_roles = sorted({str(case.role) for case in rows if str(case.role) != "discovery"})
    if bad_roles:
        raise ValueError(f"internal cross-fit accepts discovery cases only, got roles {bad_roles}")
    if not 0.10 <= float(selection_fraction) <= 0.50:
        raise ValueError("selection_fraction must be between 0.10 and 0.50")
    by_tape: dict[str, list[CaseRecord]] = {}
    for case in rows:
        by_tape.setdefault(tape_group_id(case), []).append(case)
    if len(by_tape) < 4:
        raise ValueError("internal cross-fit requires at least four independent tape groups")
    tapes = sorted(by_tape)
    rng = random.Random(int(seed) ^ 0x43524F5353464954)
    rng.shuffle(tapes)
    selection_count = max(1, min(len(tapes) - 2, round(len(tapes) * float(selection_fraction))))
    selection_tapes = tuple(sorted(tapes[:selection_count]))
    search_tapes = tuple(sorted(tapes[selection_count:]))
    selection_set = set(selection_tapes)
    search = tuple(sorted((x for x in rows if tape_group_id(x) not in selection_set), key=lambda x: x.case_id))
    selection = tuple(sorted((x for x in rows if tape_group_id(x) in selection_set), key=lambda x: x.case_id))
    if set(search_tapes) & set(selection_tapes):
        raise AssertionError("internal cross-fit tape overlap")
    return InternalCrossFitSplit(search, selection, search_tapes, selection_tapes)


class CapabilityLadder:
    """Escalate synthesis capability when novelty/objective progress saturates."""

    LEVELS = (
        "typed_random",
        "systematic_structural",
        "boundary_solver",
        "large_neighborhood",
        "compatible_recombination",
        "structural_mcts",
        "grouped_cegis",
    )

    def __init__(self, *, patience: int = 2):
        self.patience = max(1, int(patience))
        self.level = 0
        self._stalled = 0

    def observe(
        self, *, attempted: int, unique: int, useful: int,
        production_qualified: int = 0, archive_insertions: int = 0,
    ) -> CapabilityEvent:
        attempted = max(0, int(attempted)); unique = max(0, int(unique)); useful = max(0, int(useful))
        production_qualified = max(0, int(production_qualified))
        archive_insertions = max(0, int(archive_insertions))
        duplicate_rate = 1.0 - unique / max(1, attempted)
        robust_progress = production_qualified > 0 or archive_insertions > 0
        stalled = not robust_progress or duplicate_rate >= 0.60
        self._stalled = self._stalled + 1 if stalled else 0
        # Severe semantic collapse deserves immediate escalation; otherwise use patience.
        trigger = stalled and (duplicate_rate >= 0.70 or self._stalled >= self.patience)
        if trigger and self.level < len(self.LEVELS) - 1:
            self.level += 1
            self._stalled = 0
            return CapabilityEvent(self.level, True, self.LEVELS[self.level])
        return CapabilityEvent(self.level, False, "healthy" if not stalled else "stagnation_accumulating")


class InvariantSearchArchive:
    """Safety-first semantic-QD archive. Held-out certification is intentionally absent."""

    def __init__(self) -> None:
        self.elites: dict[tuple[Any, ...], SearchElite] = {}
        self.research: dict[str, SearchElite] = {}

    @staticmethod
    def _topology(outcomes: Sequence[GroupOutcome]) -> str:
        if not outcomes:
            return "none"
        active = sum(bool(x.behavior_changed) for x in outcomes)
        frac = active / len(outcomes)
        if active == 0: return "none"
        if frac < 0.34: return "sparse"
        if frac < 0.75: return "mixed"
        return "broad"

    def consider(self, module: InvariantStrategicModule, outcomes: Sequence[GroupOutcome]) -> bool:
        rows = tuple(outcomes)
        fitness = RobustGroupFitness.from_outcomes(rows)
        qual = production_qualification(rows)
        descriptor = semantic_descriptor(module, activation_topology=self._topology(rows))
        elite = SearchElite(module, descriptor, rows, fitness, qual)
        self.research[module.semantic_id()] = elite
        if not qual.production:
            return False
        module.validate(production=True, multi_tape_support=qual.useful_groups)
        key = descriptor.key()
        old = self.elites.get(key)
        if old is None or fitness.key() < old.fitness.key():
            self.elites[key] = elite
            return True
        return False

    def production_modules(self) -> tuple[InvariantStrategicModule, ...]:
        return tuple(self.elites[k].module for k in sorted(self.elites, key=repr))


def _guard_family(feature: str) -> str:
    if feature.startswith("MARKET_"): return "market"
    if feature.startswith("OWN_TILE_"): return "own_tile_relational"
    if feature.startswith("OWN_WORKER_"): return "own_unit_relational"
    if feature.startswith("OWN_SHED") or feature.startswith("OWN_INVENTORY") or feature.startswith("OWN_SEED"): return "own_resource"
    if feature.startswith("OWN_MONEY") or feature.startswith("OWN_UNITS") or feature.startswith("OWN_LAND"): return "own_economy"
    if feature in ("DAY", "HOUR", "STEP"): return "lifecycle_time"
    if feature.startswith("SHOP_"): return "town_regime"
    return feature.lower()


def semantic_descriptor(module: InvariantStrategicModule, *, activation_topology: str = "unknown") -> SemanticDescriptor:
    """QD descriptor uses structural semantics, never scalar thresholds or identities."""
    return SemanticDescriptor(
        effects=tuple(r.sketch.effect_class for r in module.rules),
        domains=tuple(r.domain for r in module.rules),
        controls=tuple(r.sketch.control_pattern for r in module.rules),
        guard_families=tuple(tuple(sorted({_guard_family(a.feature) for a in (*r.guard.atoms, *r.exit_guard.atoms)})) for r in module.rules),
        actions=tuple(r.action.kind for r in module.rules),
        activation_topology=str(activation_topology),
    )


def structural_skeleton(module: InvariantStrategicModule) -> tuple[Any, ...]:
    """Constant-free program shape used to measure real mechanism coverage."""
    def action_shape(action: AbstractAction) -> tuple[Any, ...]:
        if action.kind in {"ROUTE_OVERRIDE", "COMMITTED_ROUTE"}:
            return (action.kind, "parent_route")
        if action.kind == "UNIT_REWRITE":
            return (action.kind, action.match_op, action.unit_op, action.unit_arg)
        if action.kind == "MARKET_INTENT":
            return (action.kind, action.market_op, action.item, action.authority)
        return (action.kind,)

    return tuple((
        rule.sketch.effect_class,
        rule.domain,
        rule.sketch.control_pattern,
        tuple((atom.feature, atom.cmp, atom.arg) for atom in rule.guard.atoms),
        action_shape(rule.action),
        tuple((atom.feature, atom.cmp, atom.arg) for atom in rule.exit_guard.atoms),
    ) for rule in module.rules)


def translate_route_genome(genome: ModuleGenome, *, route_ids: Sequence[int]) -> InvariantStrategicModule:
    """Translate the portable route-selector subset into equivalent B4IM rules.

    Rules are reversed because ordinary ModuleGenome selectors use first-match priority,
    while active B4IM route overrides are applied in rule order and the last write wins.
    Concrete worker/tile patches and exact high-cardinality equality are intentionally not
    admitted into the invariant seed pool.
    """
    genome.validate(route_ids)
    if len(genome.rules) > MAX_RULES:
        raise ValueError(f"portable route genome exceeds B4IM rule limit {MAX_RULES}")
    translated: list[InvariantRule] = []
    for source in reversed(genome.rules):
        atoms: list[GuardAtom] = []
        if int(genome.prefix_step) > 0:
            atoms.append(GuardAtom("STEP", "GE", int(genome.prefix_step)))
        for condition in source.conditions:
            if "WORKER_" in condition.feature or "TILE_" in condition.feature:
                raise ValueError("portable route translation rejects concrete worker/tile identity guards")
            if condition.feature == "STEP" and condition.cmp == "EQ":
                raise ValueError("portable route translation rejects exact step patches")
            if condition.cmp == "EQ" and condition.feature in {
                "OWN_MONEY", "OPP_MONEY", "MARKET_PRICE", "MARKET_INVENTORY",
            }:
                raise ValueError("portable route translation rejects exact high-cardinality guards")
            atom = GuardAtom(condition.feature, condition.cmp, int(condition.value), arg=int(condition.index))
            if atom not in atoms:
                atoms.append(atom)
        if len(atoms) > MAX_GUARDS_TOTAL:
            raise ValueError("portable route genome rule exceeds B4IM guard limit")
        translated.append(InvariantRule(
            ModuleSketch("route", "INSTANT", "ROUTE_MODE"),
            "GLOBAL",
            LiftedGuard(tuple(atoms)),
            AbstractAction("ROUTE_OVERRIDE", route_id=int(source.route)),
        ))
    module = InvariantStrategicModule(tuple(translated))
    module.validate()
    return module


def default_search_goals(route_ids: Sequence[int]) -> tuple[SearchGoal, ...]:
    routes = tuple(int(x) for x in route_ids)
    if not routes:
        raise ValueError("invariant search requires at least one parent route")
    return (
        SearchGoal(
            "price_floor_product_liquidation", "CREATE_FINANCING_RUNWAY", "GLOBAL", "MARKET_INTENT",
            ("MARKET_PRICE", "OWN_SHED"),
            "Sell own non-input product inventory at the market price floor to create financing runway without increasing shared supply.",
        ),
        SearchGoal(
            "price_floor_financing_mechanic", "ENGINE_MECHANIC", "GLOBAL", "MARKET_INTENT",
            ("MARKET_PRICE", "OWN_SHED", "OWN_MONEY"),
            "Use the price-floor sale mechanic only while own cash is below a financing boundary.",
        ),
        SearchGoal(
            "starving_animal_rescue", "RECOVER_RESOURCE_INVARIANT", "EACH_OWN_UNIT", "UNIT_REWRITE",
            ("OWN_TILE_HAS_ANIMAL", "OWN_TILE_CONSECUTIVE_UNFED", "OWN_INVENTORY"),
            "Replace a lower-priority planned unit action with feed when the bound own animal is at a starvation boundary and feed is locally available.",
        ),
        SearchGoal(
            "unwatered_plant_rescue", "RECOVER_RESOURCE_INVARIANT", "EACH_OWN_UNIT", "UNIT_REWRITE",
            ("OWN_TILE_CONSECUTIVE_UNWATERED",),
            "Replace a less urgent planned plant action with water when the bound own crop is at a lifecycle watering boundary.",
        ),
        SearchGoal(
            "avoid_overbuy_inventory", "AVOID_OVERBUY", "GLOBAL", "ROUTE_OVERRIDE",
            ("OWN_SHED_TOTAL", "OWN_MONEY"),
            "Enter a parent route whose materialized market behavior is safer when own inventory is near capacity.",
        ),
        SearchGoal(
            "overflow_product_liquidation", "AVOID_OVERBUY", "GLOBAL", "MARKET_INTENT",
            ("OWN_SHED_TOTAL", "OWN_SHED", "MARKET_PRICE"),
            "Liquidate bounded product inventory before end-of-day shed overflow would discard it.",
        ),
        SearchGoal(
            "accelerate_expansion", "ACCELERATE_EXPANSION", "GLOBAL", "MARKET_INTENT",
            ("OWN_UNITS", "OWN_MONEY", "OWN_LAND"),
            "Add one bounded expansion transaction only when own financing and capacity invariants permit it.",
        ),
        SearchGoal(
            "committed_financing_mode", "ROUTE_MODE", "GLOBAL", "COMMITTED_ROUTE",
            ("OWN_MONEY", "OWN_SHED_TOTAL"),
            "Latch a parent strategic route under a broad financing condition and leave it under an explicit resource exit invariant.",
        ),
        SearchGoal(
            "adaptive_route_regime", "ROUTE_MODE", "GLOBAL", "ROUTE_OVERRIDE",
            ("OWN_MONEY", "OWN_UNITS", "OWN_LAND", "OWN_SHED_TOTAL", "DAY", "HOUR"),
            "Select an existing parent route under a broad economic, capacity, or season-lifecycle regime.",
        ),
        SearchGoal(
            "maintenance_redundancy", "REDUCE_MAINTENANCE_BACKLOG", "EACH_OWN_UNIT", "UNIT_REWRITE",
            ("OWN_TILE_FED", "OWN_TILE_CARED"),
            "Suppress already-satisfied animal maintenance work so the representation can learn safe relational action-rewrite boundaries.",
        ),
    )


def _module(rule: InvariantRule) -> InvariantStrategicModule:
    out = InvariantStrategicModule((rule,)); out.validate(); return out


def _goal_candidates(goal: SearchGoal, route_ids: Sequence[int]) -> list[InvariantStrategicModule]:
    out: list[InvariantStrategicModule] = []
    if goal.goal_id == "price_floor_product_liquidation":
        for item in (5, 6, 7):  # EGG, MILK, WOOL: outputs, not lifecycle inputs.
            for reserve in (1, 2, 4, 8):
                for quantity in (1, 2, 4):
                    if quantity > reserve:
                        continue
                    g = LiftedGuard((GuardAtom("MARKET_PRICE", "LE", 1, arg=item), GuardAtom("OWN_SHED", "GE", reserve, arg=item)))
                    out.append(_module(InvariantRule(ModuleSketch("market", "INSTANT", goal.effect_class), "GLOBAL", g,
                        AbstractAction("MARKET_INTENT", market_op=6, item=item, quantity=quantity, authority=5, priority=220))))
    elif goal.goal_id == "price_floor_financing_mechanic":
        for item in (5, 6, 7):
            for cash in (100, 250, 500, 1000, 2000, 5000):
                for quantity in (1, 2, 4):
                    g = LiftedGuard((
                        GuardAtom("MARKET_PRICE", "LE", 1, arg=item),
                        GuardAtom("OWN_SHED", "GE", quantity, arg=item),
                        GuardAtom("OWN_MONEY", "LE", cash),
                    ))
                    out.append(_module(InvariantRule(
                        ModuleSketch("engine_mechanic", "INSTANT", goal.effect_class), "GLOBAL", g,
                        AbstractAction("MARKET_INTENT", market_op=6, item=item, quantity=quantity, authority=5, priority=225),
                    )))
    elif goal.goal_id == "starving_animal_rescue":
        # FEED consumes the bound worker's inventory. Guard local/global inventory plus relational animal state.
        for match in (0, 17, 16, 11):  # PASS/lower-priority work -> FEED
            for require_unfed_today in (False, True):
                atoms = [
                    GuardAtom("OWN_TILE_HAS_ANIMAL", "EQ", 1),
                    GuardAtom("OWN_TILE_CONSECUTIVE_UNFED", "GE", 1),
                    GuardAtom("OWN_INVENTORY", "GE", 1, arg=0),
                ]
                if require_unfed_today:
                    atoms.append(GuardAtom("OWN_TILE_FED", "EQ", 0))
                out.append(_module(InvariantRule(
                    ModuleSketch("recovery", "INSTANT", goal.effect_class), "EACH_OWN_UNIT", LiftedGuard(tuple(atoms)),
                    AbstractAction("UNIT_REWRITE", match_op=match, unit_op=15, unit_arg=0, quantity=1),
                )))
    elif goal.goal_id == "unwatered_plant_rescue":
        # Preserve profitable HARVEST and intentional DIG decisions. Replacing only
        # FERTILIZE keeps this family a maintenance-priority correction.
        for debt in (1, 2):
            for match in (0, 11):  # PASS or FERTILIZE -> WATER; never sacrifice HARVEST.
                for require_unwatered_today in (False, True):
                    atoms = [GuardAtom("OWN_TILE_CONSECUTIVE_UNWATERED", "GE", debt)]
                    if require_unwatered_today:
                        atoms.append(GuardAtom("OWN_TILE_WATERED", "EQ", 0))
                    out.append(_module(InvariantRule(
                        ModuleSketch("recovery", "INSTANT", goal.effect_class), "EACH_OWN_UNIT", LiftedGuard(tuple(atoms)),
                        AbstractAction("UNIT_REWRITE", match_op=match, unit_op=9),
                    )))
    elif goal.goal_id == "avoid_overbuy_inventory":
        for route in route_ids:
            for cap in (80, 90, 96, 99):
                g = LiftedGuard((GuardAtom("OWN_SHED_TOTAL", "GE", cap),))
                out.append(_module(InvariantRule(ModuleSketch("market", "INSTANT", goal.effect_class), "GLOBAL", g,
                    AbstractAction("ROUTE_OVERRIDE", route_id=int(route)))))
    elif goal.goal_id == "overflow_product_liquidation":
        for item in range(8):  # harvested products only; fertilizer is a strategic input.
            for cap in (90, 96, 99):
                for quantity in (1, 2, 4):
                    g = LiftedGuard((
                        GuardAtom("OWN_SHED_TOTAL", "GE", cap),
                        GuardAtom("OWN_SHED", "GE", quantity, arg=item),
                    ))
                    out.append(_module(InvariantRule(
                        ModuleSketch("market", "INSTANT", goal.effect_class), "GLOBAL", g,
                        AbstractAction("MARKET_INTENT", market_op=6, item=item, quantity=quantity, authority=5, priority=230),
                    )))
    elif goal.goal_id == "accelerate_expansion":
        for units in (2, 3, 4, 5):
            for cash in (200, 500, 1000, 2000):
                g = LiftedGuard((GuardAtom("OWN_UNITS", "LE", units), GuardAtom("OWN_MONEY", "GE", cash)))
                out.append(_module(InvariantRule(ModuleSketch("expansion", "INSTANT", goal.effect_class), "GLOBAL", g,
                    AbstractAction("MARKET_INTENT", market_op=1, quantity=1, authority=3, priority=180))))
        # One successful hire disables the rule for the rest of that day. This turns a
        # potentially exponential order stream into an invariant one-hire daily cadence.
        for units in (2, 3, 4, 5):
            for cash in (200, 500, 1000, 2000):
                for final_day in (7, 14, 21, 27):
                    g = LiftedGuard((
                        GuardAtom("OWN_HIRES_TODAY", "EQ", 0),
                        GuardAtom("OWN_UNITS", "LE", units),
                        GuardAtom("OWN_MONEY", "GE", cash),
                        GuardAtom("DAY", "LE", final_day),
                    ))
                    out.append(_module(InvariantRule(
                        ModuleSketch("expansion", "INSTANT", goal.effect_class), "GLOBAL", g,
                        AbstractAction("MARKET_INTENT", market_op=1, quantity=1, authority=3, priority=190),
                    )))
        # A single designated-day intervention is a season-lifecycle policy, not a tape
        # position: hires_today makes it self-disabling after one successful order.
        for day in (0, 1, 2, 3, 5, 7):
            for units in (2, 3, 4, 5):
                for cash in (200, 500, 1000, 2000):
                    g = LiftedGuard((
                        GuardAtom("DAY", "EQ", day),
                        GuardAtom("OWN_HIRES_TODAY", "EQ", 0),
                        GuardAtom("OWN_UNITS", "LE", units),
                        GuardAtom("OWN_MONEY", "GE", cash),
                    ))
                    out.append(_module(InvariantRule(
                        ModuleSketch("expansion", "INSTANT", goal.effect_class), "GLOBAL", g,
                        AbstractAction("MARKET_INTENT", market_op=1, quantity=1, authority=3, priority=195),
                    )))
        for land, cash in ((1, 1200), (2, 2500), (3, 5000)):
            g = LiftedGuard((GuardAtom("OWN_LAND", "LE", land), GuardAtom("OWN_MONEY", "GE", cash)))
            out.append(_module(InvariantRule(
                ModuleSketch("expansion", "INSTANT", goal.effect_class), "GLOBAL", g,
                AbstractAction("MARKET_INTENT", market_op=2, quantity=1, authority=3, priority=185),
            )))
    elif goal.goal_id == "committed_financing_mode":
        for route in route_ids:
            for enter, exit_ in ((500, 1200), (1000, 2500), (2000, 5000)):
                out.append(_module(InvariantRule(ModuleSketch("route", "COMMITTED", goal.effect_class), "GLOBAL",
                    LiftedGuard((GuardAtom("OWN_MONEY", "LE", enter),)), AbstractAction("COMMITTED_ROUTE", route_id=int(route)),
                    LiftedGuard((GuardAtom("OWN_MONEY", "GE", exit_),)))))
    elif goal.goal_id == "adaptive_route_regime":
        single_boundaries = {
            "OWN_MONEY": (200, 500, 1000, 2000, 5000, 10000),
            "OWN_UNITS": (1, 2, 4, 8, 12),
            "OWN_LAND": (25, 50, 75),
            "OWN_SHED_TOTAL": (25, 50, 75, 90, 96),
            "DAY": (3, 7, 14, 21, 27),
            "HOUR": (4, 8, 12, 16, 20),
        }
        for route in route_ids:
            for day in range(30):
                out.append(_module(InvariantRule(
                    ModuleSketch("route", "INSTANT", goal.effect_class), "GLOBAL",
                    LiftedGuard((GuardAtom("DAY", "EQ", day),)),
                    AbstractAction("ROUTE_OVERRIDE", route_id=int(route)),
                )))
            for day in (1, 2, 3, 5, 7):
                for cash in (10, 20, 50, 100, 250, 500, 1000, 2000):
                    out.append(_module(InvariantRule(
                        ModuleSketch("route", "INSTANT", goal.effect_class), "GLOBAL",
                        LiftedGuard((GuardAtom("DAY", "EQ", day), GuardAtom("OWN_MONEY", "GE", cash))),
                        AbstractAction("ROUTE_OVERRIDE", route_id=int(route)),
                    )))
            for feature, values in single_boundaries.items():
                for comparator in ("LE", "GE"):
                    for value in values:
                        out.append(_module(InvariantRule(
                            ModuleSketch("route", "INSTANT", goal.effect_class), "GLOBAL",
                            LiftedGuard((GuardAtom(feature, comparator, value),)),
                            AbstractAction("ROUTE_OVERRIDE", route_id=int(route)),
                        )))
            for start, end in ((0, 3), (4, 7), (8, 14), (15, 21), (22, 27), (25, 29)):
                out.append(_module(InvariantRule(
                    ModuleSketch("route", "INSTANT", goal.effect_class), "GLOBAL",
                    LiftedGuard((GuardAtom("DAY", "GE", start), GuardAtom("DAY", "LE", end))),
                    AbstractAction("ROUTE_OVERRIDE", route_id=int(route)),
                )))
            for cash in (500, 1000, 2000, 5000):
                for units in (2, 4, 8):
                    out.append(_module(InvariantRule(
                        ModuleSketch("route", "INSTANT", goal.effect_class), "GLOBAL",
                        LiftedGuard((GuardAtom("OWN_MONEY", "GE", cash), GuardAtom("OWN_UNITS", "LE", units))),
                        AbstractAction("ROUTE_OVERRIDE", route_id=int(route)),
                    )))
            for cash in (1000, 2500, 5000):
                for land in (25, 50, 75):
                    out.append(_module(InvariantRule(
                        ModuleSketch("route", "INSTANT", goal.effect_class), "GLOBAL",
                        LiftedGuard((GuardAtom("OWN_MONEY", "GE", cash), GuardAtom("OWN_LAND", "LE", land))),
                        AbstractAction("ROUTE_OVERRIDE", route_id=int(route)),
                    )))
            for cash in (500, 1000, 2000):
                for shed in (75, 90, 96):
                    out.append(_module(InvariantRule(
                        ModuleSketch("route", "INSTANT", goal.effect_class), "GLOBAL",
                        LiftedGuard((GuardAtom("OWN_MONEY", "LE", cash), GuardAtom("OWN_SHED_TOTAL", "GE", shed))),
                        AbstractAction("ROUTE_OVERRIDE", route_id=int(route)),
                    )))
    elif goal.goal_id == "maintenance_redundancy":
        out.extend((
            _module(InvariantRule(ModuleSketch("maintenance", "INSTANT", goal.effect_class), "EACH_OWN_UNIT",
                LiftedGuard((GuardAtom("OWN_TILE_FED", "EQ", 1),)), AbstractAction("UNIT_REWRITE", match_op=15, unit_op=0))),
            _module(InvariantRule(ModuleSketch("maintenance", "INSTANT", goal.effect_class), "EACH_OWN_UNIT",
                LiftedGuard((GuardAtom("OWN_TILE_CARED", "EQ", 1),)), AbstractAction("UNIT_REWRITE", match_op=17, unit_op=0))),
            _module(InvariantRule(ModuleSketch("maintenance", "INSTANT", goal.effect_class), "EACH_OWN_UNIT",
                LiftedGuard((GuardAtom("OWN_TILE_WATERED", "EQ", 1),)), AbstractAction("UNIT_REWRITE", match_op=9, unit_op=0))),
        ))
    else:
        raise ValueError(f"no structural emitter for search goal {goal.goal_id}")
    return out


def emit_skeletons(goal: SearchGoal, *, n: int, seed: int, route_ids: Sequence[int]) -> tuple[InvariantStrategicModule, ...]:
    """Skeleton-first typed emission. Constants come from small semantic basins, not arbitrary int32 mutation."""
    rows = _goal_candidates(goal, route_ids)
    rng = random.Random(int(seed)); rng.shuffle(rows)
    seen: set[str] = set(); out: list[InvariantStrategicModule] = []
    for m in rows:
        sid = m.semantic_id()
        if sid in seen: continue
        seen.add(sid); out.append(m)
        if len(out) >= max(0, int(n)): break
    return tuple(out)


def mechanism_anchors(route_ids: Sequence[int]) -> tuple[InvariantStrategicModule, ...]:
    """Small parent-portable anchors for source-derived public-state regimes."""
    routes = {int(route) for route in route_ids}
    out = []
    # This composition survived every outer discovery fold. The first rule restores the
    # portable wool-price route selector that supplies the repeated L->W behavior; the
    # second rule is the resource-safe day-one specialization and wins on overlap.
    if 70 in routes:
        wool_rule = InvariantRule(
            ModuleSketch("route", "INSTANT", "ROUTE_MODE"), "GLOBAL",
            LiftedGuard((GuardAtom("MARKET_PRICE", "LE", 192, arg=7),)),
            AbstractAction("ROUTE_OVERRIDE", route_id=70),
        )
        for route in (212, 211):
            if route not in routes:
                continue
            day_one_rule = InvariantRule(
                ModuleSketch("route", "INSTANT", "ROUTE_MODE"), "GLOBAL",
                LiftedGuard((GuardAtom("DAY", "EQ", 1), GuardAtom("OWN_MONEY", "GE", 20))),
                AbstractAction("ROUTE_OVERRIDE", route_id=route),
            )
            module = InvariantStrategicModule((wool_rule, day_one_rule))
            module.validate()
            out.append(module)
    for route in route_ids:
        out.append(_module(InvariantRule(
            ModuleSketch("route", "INSTANT", "ROUTE_MODE"), "GLOBAL",
            LiftedGuard((GuardAtom("DAY", "EQ", 1), GuardAtom("OWN_MONEY", "GE", 20))),
            AbstractAction("ROUTE_OVERRIDE", route_id=int(route)),
        )))
    return tuple(out)


def recombine_compatible(a: InvariantStrategicModule, b: InvariantStrategicModule) -> InvariantStrategicModule | None:
    """Combine compatible semantic subtrees; competing route controllers are never stacked blindly."""
    route_kinds = {"ROUTE_OVERRIDE", "COMMITTED_ROUTE"}
    if any(r.action.kind in route_kinds for r in a.rules) and any(r.action.kind in route_kinds for r in b.rules):
        return None
    rules: list[InvariantRule] = []
    seen: set[str] = set()
    for r in (*a.rules, *b.rules):
        ident = repr((r.sketch, r.domain, r.guard, r.action, r.exit_guard))
        if ident in seen: continue
        seen.add(ident); rules.append(r)
    if not rules or len(rules) > MAX_RULES:
        return None
    out = InvariantStrategicModule(tuple(rules))
    try:
        out.validate()
    except ValueError:
        return None
    return out


def compose_structural_modules(
    modules: Sequence[InvariantStrategicModule],
    *,
    budget: int,
    seed: int,
) -> tuple[InvariantStrategicModule, ...]:
    """Compose one global mechanism with one relational unit mechanism.

    This deliberately avoids stacking two market controllers or two rewrite chains. One
    representative per constant-free skeleton is used so threshold variants cannot consume
    the compositional budget.
    """
    representatives: dict[tuple[Any, ...], InvariantStrategicModule] = {}
    for module in modules:
        if len(module.rules) == 1:
            representatives.setdefault(structural_skeleton(module), module)
    global_rows = [m for m in representatives.values() if m.rules[0].domain == "GLOBAL"]
    unit_rows = [m for m in representatives.values() if m.rules[0].domain == "EACH_OWN_UNIT"]
    pairs = [(global_module, unit_module) for global_module in global_rows for unit_module in unit_rows]
    random.Random(int(seed) ^ 0x434F4D504F5345).shuffle(pairs)
    out: dict[str, InvariantStrategicModule] = {}
    for global_module, unit_module in pairs:
        child = recombine_compatible(global_module, unit_module)
        if child is None:
            continue
        out.setdefault(child.semantic_id(), child)
        if len(out) >= max(0, int(budget)):
            break
    return tuple(out.values())


def large_neighborhood(parent: InvariantStrategicModule, goal: SearchGoal, *, seed: int, route_ids: Sequence[int]) -> InvariantStrategicModule:
    """Replace one meaningful rule/sub-program while retaining the rest of the mechanism."""
    donors = emit_skeletons(goal, n=8, seed=seed, route_ids=route_ids)
    if not donors:
        raise ValueError("large-neighborhood search has no structural donor")
    donor_rule = donors[0].rules[0]
    if not parent.rules:
        return InvariantStrategicModule((donor_rule,))
    rng = random.Random(int(seed) ^ 0x4C4E53)
    rules = list(parent.rules)
    idx = rng.randrange(len(rules)); rules[idx] = donor_rule
    out = InvariantStrategicModule(tuple(rules[:MAX_RULES]))
    out.validate()
    if out.semantic_id() == parent.semantic_id() and len(donors) > 1:
        rules[idx] = donors[1].rules[0]; out = InvariantStrategicModule(tuple(rules[:MAX_RULES])); out.validate()
    return out


def structural_mcts(
    roots: Sequence[InvariantStrategicModule],
    goals: Sequence[SearchGoal],
    *,
    budget: int,
    seed: int,
    route_ids: Sequence[int],
    quality: Mapping[str, float] | None = None,
) -> tuple[InvariantStrategicModule, ...]:
    """Bounded grammar-expansion MCTS/UCB. Scalar thresholds are not expanded here."""
    roots = tuple(roots) or (InvariantStrategicModule(),)
    goals = tuple(goals)
    if not goals or budget <= 0:
        return ()
    q = dict(quality or {})
    visits: dict[str, int] = {m.semantic_id(): 1 for m in roots}
    reward: dict[str, float] = {m.semantic_id(): float(q.get(m.semantic_id(), 0.0)) for m in roots}
    frontier = list(roots); emitted: dict[str, InvariantStrategicModule] = {}
    rng = random.Random(int(seed))
    for step in range(int(budget)):
        total = sum(visits.values()) + 1
        def ucb(m: InvariantStrategicModule) -> tuple[float, str]:
            sid = m.semantic_id(); n = max(1, visits.get(sid, 1)); mean = reward.get(sid, 0.0) / n
            return (mean + math.sqrt(2.0 * math.log(total) / n), sid)
        parent = max(frontier, key=ucb)
        goal = goals[(step + rng.randrange(len(goals))) % len(goals)]
        child = large_neighborhood(parent, goal, seed=seed + step * 1009, route_ids=route_ids)
        sid = child.semantic_id(); visits[parent.semantic_id()] = visits.get(parent.semantic_id(), 1) + 1
        if sid not in emitted:
            emitted[sid] = child; frontier.append(child); visits[sid] = 1; reward[sid] = float(q.get(sid, 0.0))
    return tuple(emitted.values())


def classify_counterexample(row: Mapping[str, Any]) -> str:
    """Map exact failure evidence to one closed structural CEGIS class."""
    if bool(row.get("action_conflict", False)):
        return "action_conflict"
    if bool(row.get("lifecycle_missing", False)):
        return "lifecycle_missing"
    if bool(row.get("resource_invariant", False)):
        return "resource_invariant"
    if bool(row.get("unstable_constant", False)):
        return "unstable_constant"
    if bool(row.get("wrong_action_effect", False)):
        return "wrong_action_effect"
    if bool(row.get("new_loss", False)) or bool(row.get("w2l", False)) or int(row.get("activation_count", 0) or 0) > 0:
        return "guard_too_broad"
    if int(row.get("margin_delta", 0) or 0) < 0:
        return "wrong_action_effect"
    return "lifecycle_missing"


def cegis_repair(
    module: InvariantStrategicModule,
    failure_class: str,
    *,
    route_ids: Sequence[int],
    goal: SearchGoal | None = None,
    threshold_samples: Mapping[str, Mapping[str, Sequence[tuple[int, bool, bool]]]] | None = None,
) -> InvariantStrategicModule:
    """Structural CEGIS repair. Counterexamples refine guards/actions; they never add case identity."""
    if failure_class not in CEGIS_FAILURES:
        raise ValueError(f"unknown CEGIS failure class {failure_class}")
    if not module.rules:
        if goal is None: return module
        rows = emit_skeletons(goal, n=1, seed=0, route_ids=route_ids)
        return rows[0] if rows else module
    rules = list(module.rules)
    r = rules[0]
    if failure_class == "guard_too_broad" and len(r.guard.atoms) + len(r.exit_guard.atoms) < MAX_GUARDS_TOTAL:
        existing = set(r.guard.atoms) | set(r.exit_guard.atoms)
        candidates = (
            GuardAtom("OWN_MONEY", "GE", 1),
            GuardAtom("OWN_UNITS", "GE", 1),
            GuardAtom("OWN_LAND", "GE", 1),
            GuardAtom("OWN_SHED_TOTAL", "LE", 99),
        )
        atom = next((x for x in candidates if x not in existing), None)
        if atom is not None:
            rules[0] = replace(r, guard=LiftedGuard((*r.guard.atoms, atom)))
    elif failure_class == "unstable_constant" and threshold_samples:
        atom_index = next((
            i for i, atom in enumerate(r.guard.atoms)
            if atom.cmp in {"GE", "GT", "LE", "LT"} and guard_evidence_key(atom) in threshold_samples
        ), None)
        if atom_index is not None:
            atom = r.guard.atoms[atom_index]
            sol = solve_cross_tape_threshold(threshold_samples[guard_evidence_key(atom)], comparator=atom.cmp)
            atoms = list(r.guard.atoms); atoms[atom_index] = replace(atom, value=sol.threshold)
            rules[0] = replace(r, guard=LiftedGuard(tuple(atoms)))
    elif failure_class in {"wrong_action_effect", "resource_invariant"} and goal is not None:
        donors = emit_skeletons(goal, n=2, seed=17, route_ids=route_ids)
        if donors: rules[0] = donors[0].rules[0]
    elif failure_class == "action_conflict" and len(rules) > 1:
        rules = rules[:-1]
    elif failure_class == "lifecycle_missing" and r.action.kind == "ROUTE_OVERRIDE" and r.guard.atoms:
        exit_atom = GuardAtom(r.guard.atoms[0].feature, "GE" if r.guard.atoms[0].cmp in {"LE", "LT"} else "LE", r.guard.atoms[0].value, r.guard.atoms[0].arg)
        rules[0] = replace(r, sketch=replace(r.sketch, control_pattern="COMMITTED"), action=replace(r.action, kind="COMMITTED_ROUTE"), exit_guard=LiftedGuard((exit_atom,)))
    out = InvariantStrategicModule(tuple(rules)); out.validate(); return out


def pair_case_rows(baseline: StateEvaluation, candidate: StateEvaluation) -> tuple[dict[str, Any], ...]:
    if baseline.panel_id != candidate.panel_id or baseline.case_ids != candidate.case_ids:
        raise ValueError("invariant baseline/candidate exact evaluations are not aligned")
    rows = []
    for i, cid in enumerate(baseline.case_ids):
        b = int(baseline.margins[i]); c = int(candidate.margins[i]); d = c - b
        rows.append({
            "case_id": cid,
            "baseline_margin": b,
            "candidate_margin": c,
            "margin_delta": d,
            "new_loss": b >= 0 and c < 0,
            "w2l": b > 0 and c <= 0,
            "action_changed": int(candidate.action_changes[i]) > 0,
            "activation_count": int(candidate.activations[i]),
            "trace_hash": int(candidate.trace_hashes[i]),
        })
    return tuple(rows)


def guard_evidence_key(atom: GuardAtom) -> str:
    return f"{atom.feature}:{int(atom.arg)}"


def derive_threshold_evidence(
    module: InvariantStrategicModule,
    rows: Sequence[Mapping[str, Any]],
    panel: Sequence[CaseRecord],
) -> dict[str, dict[str, tuple[tuple[int, bool, bool], ...]]]:
    """Build feature-specific threshold evidence from public probes and grouped exact outcomes."""
    module.validate()
    cases = {case.case_id: case for case in panel}
    if set(cases) != {str(row["case_id"]) for row in rows}:
        raise ValueError("threshold evidence rows/panel are not aligned")
    case_to_tape = {case.case_id: tape_group_id(case) for case in panel}
    outcomes = {x.group_id: x for x in group_outcomes_from_case_rows(rows, case_to_tape)}
    atoms = {
        atom for rule in module.rules if rule.domain == "GLOBAL"
        for atom in (*rule.guard.atoms, *rule.exit_guard.atoms)
        if atom.cmp in {"GE", "GT", "LE", "LT"}
    }
    evidence: dict[str, dict[str, list[tuple[int, bool, bool]]]] = {}
    for atom in sorted(atoms, key=lambda x: (x.feature, x.arg, x.cmp, x.value)):
        args = (int(atom.arg),) if atom.feature in _REGISTRY_ARG_FEATURES else ()
        position = _REGISTRY_POS.get((atom.feature, args))
        if position is None:
            continue
        key = guard_evidence_key(atom)
        for case in panel:
            if case.public_context_registry_width != len(_REGISTRY_MANIFEST) or case.public_context_width < len(_REGISTRY_MANIFEST):
                continue
            outcome = outcomes[tape_group_id(case)]
            protected_negative = bool(
                outcome.new_loss or outcome.severe_regression or (case.protected and outcome.margin_delta < 0)
            )
            for frame in range(max(1, len(case.context_probe_steps))):
                value = int(case.public_context_values[frame * case.public_context_width + position])
                evidence.setdefault(key, {}).setdefault(outcome.group_id, []).append(
                    (value, bool(outcome.useful), protected_negative)
                )
    return {
        key: {group: tuple(samples) for group, samples in sorted(groups.items())}
        for key, groups in sorted(evidence.items())
    }


def grouped_counterexample_rows(
    rows: Sequence[Mapping[str, Any]],
    case_to_tape: Mapping[str, str],
) -> tuple[dict[str, Any], ...]:
    """Select one deterministic worst counterexample per independent tape."""
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        case_id = str(row["case_id"])
        if case_id not in case_to_tape:
            raise ValueError(f"counterexample case has no tape group: {case_id}")
        grouped.setdefault(str(case_to_tape[case_id]), []).append(row)
    selected = []
    for tape, tape_rows in sorted(grouped.items()):
        worst = min(tape_rows, key=lambda x: (
            not bool(x.get("new_loss", False) or x.get("w2l", False)),
            int(x.get("margin_delta", 0)),
            str(x["case_id"]),
        ))
        selected.append(dict(worst))
    return tuple(selected)


def classify_grouped_counterexamples(rows: Sequence[Mapping[str, Any]]) -> str:
    if not rows:
        raise ValueError("grouped CEGIS requires counterexample rows")
    classes = [classify_counterexample(row) for row in rows]
    return min(CEGIS_FAILURES, key=lambda name: (-classes.count(name), CEGIS_FAILURES.index(name)))


class InvariantSearchBackend:
    """Permanent grouped B4IM synthesis backend backed exclusively by exact CUDA evaluation."""

    def __init__(
        self,
        *,
        route_ids: Sequence[int],
        evaluator: Any,
        goals: Sequence[SearchGoal] | None = None,
        archive: InvariantSearchArchive | None = None,
        ladder: CapabilityLadder | None = None,
        portable_route_seeds: Sequence[ModuleGenome] = (),
    ):
        self.route_ids = tuple(int(x) for x in route_ids)
        if not self.route_ids:
            raise ValueError("InvariantSearchBackend requires route IDs")
        self.evaluator = evaluator
        self.goals = tuple(goals or default_search_goals(self.route_ids))
        self.archive = archive or InvariantSearchArchive()
        self.ladder = ladder or CapabilityLadder()
        self.portable_route_seeds = tuple(
            translate_route_genome(genome, route_ids=self.route_ids)
            for genome in portable_route_seeds
        )
        self._cegis_queue: dict[str, InvariantStrategicModule] = {}
        self._threshold_evidence: dict[str, dict[str, dict[str, set[tuple[int, bool, bool]]]]] = {}

    def _merge_threshold_evidence(
        self,
        goal_id: str,
        evidence: Mapping[str, Mapping[str, Sequence[tuple[int, bool, bool]]]],
    ) -> None:
        goal_rows = self._threshold_evidence.setdefault(str(goal_id), {})
        for key, groups in evidence.items():
            key_rows = goal_rows.setdefault(str(key), {})
            for group, samples in groups.items():
                key_rows.setdefault(str(group), set()).update(
                    (int(value), bool(beneficial), bool(protected_negative))
                    for value, beneficial, protected_negative in samples
                )

    def threshold_evidence(self) -> dict[str, dict[str, dict[str, tuple[tuple[int, bool, bool], ...]]]]:
        return {
            goal: {
                key: {group: tuple(sorted(samples)) for group, samples in sorted(groups.items())}
                for key, groups in sorted(keys.items())
            }
            for goal, keys in sorted(self._threshold_evidence.items())
        }

    def _goal_for_module(self, module: InvariantStrategicModule) -> SearchGoal | None:
        if not module.rules:
            return None
        rule = module.rules[0]
        features = {x.feature for x in (*rule.guard.atoms, *rule.exit_guard.atoms)}
        compatible = [x for x in self.goals if x.effect_class == rule.sketch.effect_class]
        if not compatible:
            return None
        return max(compatible, key=lambda x: (
            3 * len(features & set(x.guard_features)),
            int(x.domain == rule.domain),
            int(x.action_family == rule.action.kind),
            -self.goals.index(x),
        ))

    def propose(
        self,
        *,
        n: int,
        seed: int,
    ) -> tuple[InvariantStrategicModule, ...]:
        n = max(0, int(n))
        evidence = self.threshold_evidence()
        if n == 0: return ()
        rows: list[InvariantStrategicModule] = []
        advanced: list[InvariantStrategicModule] = list(mechanism_anchors(self.route_ids))
        goal_rows: list[tuple[InvariantStrategicModule, ...]] = []
        for gi, goal in enumerate(self.goals):
            goal_rows.append(emit_skeletons(goal, n=n, seed=seed + gi * 65537, route_ids=self.route_ids))
        for index in range(max((len(x) for x in goal_rows), default=0)):
            for family in goal_rows:
                if index < len(family):
                    rows.append(family[index])
        if self.ladder.level >= 1:
            advanced.extend(self.portable_route_seeds)
            advanced.extend(compose_structural_modules(
                rows, budget=max(1, n // 3), seed=seed,
            ))
        if self.ladder.level >= 2:
            # Boundary solver alters only a scalar on an already chosen structure.
            solved: list[InvariantStrategicModule] = []
            for m in rows:
                g = self.goals[next((i for i, x in enumerate(self.goals) if x.effect_class == m.rules[0].sketch.effect_class), 0)] if m.rules else None
                samples = evidence.get(g.goal_id, {}) if g else {}
                if samples and m.rules and m.rules[0].guard.atoms:
                    r = m.rules[0]; atoms = list(r.guard.atoms)
                    ai = next((
                        i for i, atom in enumerate(atoms)
                        if atom.cmp in {"GE", "GT", "LE", "LT"} and guard_evidence_key(atom) in samples
                    ), None)
                    if ai is not None:
                        sol = solve_cross_tape_threshold(samples[guard_evidence_key(atoms[ai])], comparator=atoms[ai].cmp)
                        atoms[ai] = replace(atoms[ai], value=sol.threshold)
                        rules = list(m.rules); rules[0] = replace(r, guard=LiftedGuard(tuple(atoms)))
                        child = InvariantStrategicModule(tuple(rules)); child.validate()
                        if child.semantic_id() != m.semantic_id():
                            solved.append(child)
            advanced.extend(solved)
        elites = self.archive.production_modules()
        if self.ladder.level >= 3 and elites:
            for i, parent in enumerate(elites[:max(1, n // 8)]):
                advanced.append(large_neighborhood(parent, self.goals[(i + seed) % len(self.goals)], seed=seed + i, route_ids=self.route_ids))
        if self.ladder.level >= 4 and len(elites) >= 2:
            for i in range(min(len(elites) - 1, max(1, n // 12))):
                child = recombine_compatible(elites[i], elites[i + 1])
                if child is not None: advanced.append(child)
        if self.ladder.level >= 5:
            advanced.extend(structural_mcts(elites or (InvariantStrategicModule(),), self.goals, budget=max(1, n // 6), seed=seed, route_ids=self.route_ids))
        if self.ladder.level >= 6 and self._cegis_queue:
            advanced = list(self._cegis_queue.values()) + advanced
            self._cegis_queue.clear()
        seen: set[str] = set(); unique: list[InvariantStrategicModule] = []
        for m in (*advanced, *rows):
            sid = m.semantic_id()
            if sid in seen: continue
            seen.add(sid); unique.append(m)
            if len(unique) >= n: break
        return tuple(unique)

    def evaluate_batch(
        self,
        modules: Sequence[InvariantStrategicModule],
        panel: Sequence[CaseRecord],
        *,
        work_dir: str,
        panel_id: str,
    ) -> SearchRound:
        modules = tuple(modules)
        if not modules: raise ValueError("cannot evaluate an empty invariant proposal batch")
        bad_roles = sorted({str(case.role) for case in panel if str(case.role) != "discovery"})
        if bad_roles:
            raise ValueError(f"invariant search accepts discovery cases only, got roles {bad_roles}")
        # Empty B4IM is exact baseline and is evaluated in the same CUDA invocation.
        baseline_module = InvariantStrategicModule()
        batch = (baseline_module, *modules)
        exact = self.evaluator.evaluate(batch, panel, work_dir=work_dir, panel_id=panel_id)
        baseline = exact[baseline_module.semantic_id()]
        case_to_tape = {c.case_id: tape_group_id(c) for c in panel}
        useful = production = inserted = unsafe = 0
        for m in modules:
            rows = pair_case_rows(baseline, exact[m.semantic_id()])
            outcomes = group_outcomes_from_case_rows(rows, case_to_tape)
            goal = self._goal_for_module(m)
            derived = derive_threshold_evidence(m, rows, panel)
            if goal is not None and derived:
                self._merge_threshold_evidence(goal.goal_id, derived)
            qual = production_qualification(outcomes)
            production += int(qual.production)
            useful += int(any(x.useful for x in outcomes))
            unsafe += int(any(x.new_loss or x.severe_regression for x in outcomes))
            inserted += int(self.archive.consider(m, outcomes))
            if self.ladder.level >= 6 and not qual.production:
                counterexamples = grouped_counterexample_rows(rows, case_to_tape)
                failure_class = classify_grouped_counterexamples(counterexamples)
                if derived and any(x.useful for x in outcomes) and any(x.margin_delta < 0 for x in outcomes):
                    failure_class = "unstable_constant"
                repaired = cegis_repair(
                    m, failure_class, route_ids=self.route_ids, goal=goal,
                    threshold_samples=(self.threshold_evidence().get(goal.goal_id, {}) if goal is not None else {}),
                )
                if repaired.semantic_id() != m.semantic_id():
                    self._cegis_queue[repaired.semantic_id()] = repaired
        capability = self.ladder.observe(
            attempted=len(modules), unique=len({m.semantic_id() for m in modules}), useful=useful,
            production_qualified=production, archive_insertions=inserted,
        )
        return SearchRound(
            attempted=len(modules), unique=len({m.semantic_id() for m in modules}), duplicate_rate=0.0,
            useful=useful, production_qualified=production, archive_insertions=inserted, unsafe=unsafe,
            exact_candidate_cases=len(batch) * len(panel), capability=capability, evaluations=exact,
        )

    def search(
        self,
        panel: Sequence[CaseRecord],
        *,
        rounds: int,
        candidates_per_round: int,
        seed: int,
        work_dir: str,
    ) -> SearchRunResult:
        history: list[SearchRound] = []; attempted = exact_cc = 0
        seen: set[str] = set()
        for ri in range(max(1, int(rounds))):
            proposed = [m for m in self.propose(n=candidates_per_round, seed=seed + ri * 100003) if m.semantic_id() not in seen]
            for m in proposed: seen.add(m.semantic_id())
            if not proposed:
                event = self.ladder.observe(attempted=candidates_per_round, unique=0, useful=0)
                history.append(SearchRound(candidates_per_round, 0, 1.0, 0, 0, 0, 0, 0, event, {}))
                continue
            rr = self.evaluate_batch(proposed, panel, work_dir=work_dir, panel_id=f"invariant-search/r{ri}")
            history.append(rr); attempted += len(proposed); exact_cc += rr.exact_candidate_cases
        return SearchRunResult(tuple(history), self.archive, attempted, exact_cc)

    def search_crossfit(
        self,
        panel: Sequence[CaseRecord],
        *,
        rounds: int,
        candidates_per_round: int,
        seed: int,
        work_dir: str,
        top_k: int = 8,
        selection_fraction: float = 0.25,
    ) -> CrossFitSearchResult:
        """Search one partition, freeze top-K, then label the other partition once.

        Selection outcomes are deliberately computed without `evaluate_batch`: they cannot
        update threshold evidence, CEGIS, the capability ladder, or the search archive.
        """
        if int(top_k) < 1:
            raise ValueError("cross-fit top_k must be positive")
        if (
            self.archive.research or self.archive.elites or self._threshold_evidence
            or self._cegis_queue or self.ladder.level or self.ladder._stalled
        ):
            raise ValueError("cross-fit requires a pristine backend with no prior search evidence")
        split = internal_crossfit_split(panel, selection_fraction=selection_fraction, seed=seed)
        search_result = self.search(
            split.search,
            rounds=rounds,
            candidates_per_round=candidates_per_round,
            seed=seed,
            work_dir=f"{work_dir}/search",
        )
        ranked = sorted(
            self.archive.research.values(),
            key=lambda elite: (
                not elite.qualification.production,
                elite.fitness.key(),
                elite.module.semantic_id(),
            ),
        )
        frozen = tuple(elite.module for elite in ranked[:int(top_k)])
        if not frozen:
            return CrossFitSearchResult(split, search_result, (), {}, (), search_result.exact_candidate_cases)

        search_state = (
            self.ladder.level,
            self.ladder._stalled,
            tuple(sorted(self.archive.research)),
            tuple(sorted(repr(x) for x in self.archive.elites)),
            repr(self.threshold_evidence()),
            tuple(sorted(self._cegis_queue)),
        )
        baseline_module = InvariantStrategicModule()
        selection_evaluations = self.evaluator.evaluate(
            (baseline_module, *frozen),
            split.selection,
            work_dir=f"{work_dir}/selection",
            panel_id="discovery/internal/selection",
        )
        baseline = selection_evaluations[baseline_module.semantic_id()]
        selection_case_to_tape = {x.case_id: tape_group_id(x) for x in split.selection}
        transferable: list[TransferableElite] = []
        for module in frozen:
            search_elite = self.archive.research[module.semantic_id()]
            selection_rows = pair_case_rows(baseline, selection_evaluations[module.semantic_id()])
            selection_outcomes = group_outcomes_from_case_rows(selection_rows, selection_case_to_tape)
            selection_safe_useful = (
                bool(selection_outcomes)
                and not any(x.new_loss or x.severe_regression for x in selection_outcomes)
                and any(x.useful for x in selection_outcomes)
            )
            combined = tuple((*search_elite.outcomes, *selection_outcomes))
            combined_qualification = production_qualification(combined)
            if not (
                search_elite.qualification.production
                and selection_safe_useful
                and combined_qualification.production
            ):
                continue
            module.validate(production=True, multi_tape_support=combined_qualification.useful_groups)
            transferable.append(TransferableElite(
                module=module,
                descriptor=semantic_descriptor(module, activation_topology=self.archive._topology(combined)),
                search_outcomes=search_elite.outcomes,
                selection_outcomes=tuple(selection_outcomes),
                combined_fitness=RobustGroupFitness.from_outcomes(combined),
                combined_qualification=combined_qualification,
            ))
        post_selection_state = (
            self.ladder.level,
            self.ladder._stalled,
            tuple(sorted(self.archive.research)),
            tuple(sorted(repr(x) for x in self.archive.elites)),
            repr(self.threshold_evidence()),
            tuple(sorted(self._cegis_queue)),
        )
        if post_selection_state != search_state:
            raise AssertionError("one-shot selection mutated invariant search state")
        transferable.sort(key=lambda elite: (elite.combined_fitness.key(), elite.module.semantic_id()))
        selection_cases = (len(frozen) + 1) * len(split.selection)
        return CrossFitSearchResult(
            split=split,
            search=search_result,
            frozen_semantic_ids=tuple(x.semantic_id() for x in frozen),
            selection_evaluations=selection_evaluations,
            transferable_elites=tuple(transferable),
            exact_candidate_cases=search_result.exact_candidate_cases + selection_cases,
        )
