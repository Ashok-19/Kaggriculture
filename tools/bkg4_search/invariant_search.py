from __future__ import annotations

import hashlib
import math
import random
import time
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
class ProposalOrigin:
    operator: str
    source_goal_id: str | None
    source_goal_path: tuple[str, ...]
    structural_mechanisms: tuple[str, ...]
    retrospective_goal_id: str | None


@dataclass(frozen=True)
class MctsProposal:
    module: InvariantStrategicModule
    source_goal_path: tuple[str, ...]

    @property
    def source_goal_id(self) -> str:
        if not self.source_goal_path:
            raise ValueError("MCTS proposal has no generating-goal path")
        return self.source_goal_path[-1]


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
    specialist_elites: int = 0
    strong_specialists: int = 0
    boundary_specialists: int = 0


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
        "engine_mechanic_specialist",
        "specialist_composition",
        "new_structural_skeleton",
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
    module = module.canonicalized()

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
        SearchGoal("opening_route_regime", "ROUTE_MODE", "GLOBAL", "ROUTE_OVERRIDE",
            ("DAY", "OWN_MONEY", "OWN_UNITS"), "Select a parent opening lane from public phase, capital, and labor state."),
        SearchGoal("midgame_production_route", "ROUTE_MODE", "GLOBAL", "ROUTE_OVERRIDE",
            ("DAY", "OWN_LAND", "OWN_UNITS", "OWN_SHED_TOTAL"), "Select a production route during a bounded midgame capacity regime."),
        SearchGoal("late_game_route", "ROUTE_MODE", "GLOBAL", "ROUTE_OVERRIDE",
            ("DAY", "OWN_MONEY", "OWN_SHED_TOTAL"), "Select a conversion route when remaining season length makes capital recovery dominant."),
        SearchGoal("endgame_liquidation", "CREATE_FINANCING_RUNWAY", "GLOBAL", "MARKET_INTENT",
            ("DAY", "OWN_SHED", "MARKET_PRICE"), "Liquidate bounded output inventory near season end before held value expires."),
        SearchGoal("market_scarcity_liquidation", "CREATE_FINANCING_RUNWAY", "GLOBAL", "MARKET_INTENT",
            ("MARKET_PRICE", "MARKET_INVENTORY", "OWN_SHED"), "Sell output inventory into a public scarcity-price regime."),
        SearchGoal("fertilizer_replenishment", "RECOVER_RESOURCE_INVARIANT", "GLOBAL", "MARKET_INTENT",
            ("OWN_SHED", "OWN_MONEY", "DAY"), "Buy bounded fertilizer when a productive phase has capital but lacks the input."),
        SearchGoal("crop_input_procurement", "RECOVER_RESOURCE_INVARIANT", "GLOBAL", "MARKET_INTENT",
            ("OWN_SEED", "OWN_MONEY", "DAY"), "Buy bounded crop seed input before a phase-specific production lane."),
        SearchGoal("animal_acquisition", "ACCELERATE_EXPANSION", "GLOBAL", "MARKET_INTENT",
            ("OWN_SHED", "OWN_MONEY", "DAY"), "Acquire one animal under a bounded capital and lifecycle regime."),
        SearchGoal("land_unlock_timing", "ACCELERATE_EXPANSION", "GLOBAL", "MARKET_INTENT",
            ("OWN_LAND", "OWN_MONEY", "DAY"), "Authorize one land unlock only while its production payback remains available."),
        SearchGoal("weed_recovery", "RECOVER_RESOURCE_INVARIANT", "EACH_OWN_UNIT", "UNIT_REWRITE",
            ("OWN_TILE_KIND", "DAY"), "Rewrite low-priority bound-unit work to DIG while standing on a weed."),
        SearchGoal("animal_collection_priority", "PRESERVE_PRODUCTION_SAVE_FEED", "EACH_OWN_UNIT", "UNIT_REWRITE",
            ("OWN_TILE_HAS_ANIMAL", "OWN_TILE_YIELD", "OWN_TILE_FED"), "Prioritize collection of accumulated animal product after maintenance is satisfied."),
        SearchGoal("inventory_evacuation", "RECOVER_RESOURCE_INVARIANT", "EACH_OWN_UNIT", "UNIT_REWRITE",
            ("OWN_INVENTORY", "OWN_WORKER_X", "OWN_WORKER_Y"), "Replace low-value work with DROP when a bound worker carries material at the shed boundary."),
        SearchGoal("market_order_cap_protection", "ENGINE_MECHANIC", "GLOBAL", "ROUTE_OVERRIDE",
            ("OWN_HIRES_TODAY", "OWN_MONEY", "HOUR"), "Route away from redundant market orders when the ten-order processing cap is congested."),
        SearchGoal("production_deadline_recovery", "ENGINE_MECHANIC", "EACH_OWN_UNIT", "UNIT_REWRITE",
            ("STEP", "OWN_TILE_MAX_LIFESPAN_STEP", "OWN_TILE_YIELD"), "Prioritize harvest over lower-value work at a public production deadline."),
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
    elif goal.goal_id in {"opening_route_regime", "midgame_production_route", "late_game_route"}:
        windows = {
            "opening_route_regime": ((0, 0), (0, 1), (1, 3), (0, 5)),
            "midgame_production_route": ((4, 9), (7, 14), (10, 18), (14, 21)),
            "late_game_route": ((19, 24), (22, 27), (25, 29), (27, 29)),
        }[goal.goal_id]
        for route in route_ids:
            for start, end in windows:
                for feature, comparator, value in (
                    ("OWN_MONEY", "GE", 500), ("OWN_MONEY", "LE", 2000),
                    ("OWN_UNITS", "GE", 2), ("OWN_SHED_TOTAL", "GE", 50),
                ):
                    out.append(_module(InvariantRule(
                        ModuleSketch("route", "INSTANT", goal.effect_class), "GLOBAL",
                        LiftedGuard((
                            GuardAtom("DAY", "GE", start), GuardAtom("DAY", "LE", end),
                            GuardAtom(feature, comparator, value),
                        )), AbstractAction("ROUTE_OVERRIDE", route_id=int(route)),
                    )))
    elif goal.goal_id == "endgame_liquidation":
        for day in (24, 26, 27, 28, 29):
            for item in range(8):
                for quantity in (1, 2, 4, 8):
                    out.append(_module(InvariantRule(
                        ModuleSketch("market", "INSTANT", goal.effect_class), "GLOBAL",
                        LiftedGuard((GuardAtom("DAY", "GE", day), GuardAtom("OWN_SHED", "GE", quantity, arg=item))),
                        AbstractAction("MARKET_INTENT", market_op=6, item=item, quantity=quantity, authority=5, priority=245),
                    )))
    elif goal.goal_id == "market_scarcity_liquidation":
        for item in range(8):
            for price in (25, 50, 100, 200, 500):
                for quantity in (1, 2, 4):
                    out.append(_module(InvariantRule(
                        ModuleSketch("market", "INSTANT", goal.effect_class), "GLOBAL",
                        LiftedGuard((GuardAtom("MARKET_PRICE", "GE", price, arg=item), GuardAtom("OWN_SHED", "GE", quantity, arg=item))),
                        AbstractAction("MARKET_INTENT", market_op=6, item=item, quantity=quantity, authority=5, priority=235),
                    )))
    elif goal.goal_id == "fertilizer_replenishment":
        for day_end in (7, 14, 21, 27):
            for cash in (100, 250, 500, 1000, 2000):
                for quantity in (1, 2, 4):
                    out.append(_module(InvariantRule(
                        ModuleSketch("recovery", "INSTANT", goal.effect_class), "GLOBAL",
                        LiftedGuard((GuardAtom("DAY", "LE", day_end), GuardAtom("OWN_SHED", "LE", 1, arg=8), GuardAtom("OWN_MONEY", "GE", cash))),
                        AbstractAction("MARKET_INTENT", market_op=4, item=8, quantity=quantity, authority=4, priority=205),
                    )))
    elif goal.goal_id == "crop_input_procurement":
        for crop in range(5):
            for day_end in (3, 7, 14, 21):
                for cash in (25, 100, 250, 500, 1000):
                    out.append(_module(InvariantRule(
                        ModuleSketch("recovery", "INSTANT", goal.effect_class), "GLOBAL",
                        LiftedGuard((GuardAtom("DAY", "LE", day_end), GuardAtom("OWN_SEED", "LE", 0, arg=crop), GuardAtom("OWN_MONEY", "GE", cash))),
                        AbstractAction("MARKET_INTENT", market_op=3, item=crop, quantity=1, authority=4, priority=200),
                    )))
    elif goal.goal_id == "animal_acquisition":
        for animal in (9, 10, 11):
            for day_end in (7, 14, 21):
                for cash in (500, 1000, 2500, 5000):
                    out.append(_module(InvariantRule(
                        ModuleSketch("expansion", "INSTANT", goal.effect_class), "GLOBAL",
                        LiftedGuard((GuardAtom("DAY", "LE", day_end), GuardAtom("OWN_SHED", "LE", 0, arg=animal), GuardAtom("OWN_MONEY", "GE", cash))),
                        AbstractAction("MARKET_INTENT", market_op=5, item=animal, quantity=1, authority=3, priority=185),
                    )))
    elif goal.goal_id == "land_unlock_timing":
        for land in (25, 50, 75):
            for day_end in (7, 14, 21):
                for cash in (1000, 2000, 4000, 8000):
                    out.append(_module(InvariantRule(
                        ModuleSketch("expansion", "INSTANT", goal.effect_class), "GLOBAL",
                        LiftedGuard((GuardAtom("DAY", "LE", day_end), GuardAtom("OWN_LAND", "LE", land), GuardAtom("OWN_MONEY", "GE", cash))),
                        AbstractAction("MARKET_INTENT", market_op=2, quantity=1, authority=3, priority=190),
                    )))
    elif goal.goal_id == "weed_recovery":
        for match in (0, 9, 11, 16, 17):
            for day_end in (7, 14, 21, 29):
                out.append(_module(InvariantRule(
                    ModuleSketch("recovery", "INSTANT", goal.effect_class), "EACH_OWN_UNIT",
                    LiftedGuard((GuardAtom("OWN_TILE_KIND", "EQ", 2), GuardAtom("DAY", "LE", day_end))),
                    AbstractAction("UNIT_REWRITE", match_op=match, unit_op=12),
                )))
    elif goal.goal_id == "animal_collection_priority":
        for match in (0, 16, 17):
            for yield_units in (1, 2, 4, 8):
                out.append(_module(InvariantRule(
                    ModuleSketch("animal_lifecycle", "INSTANT", goal.effect_class), "EACH_OWN_UNIT",
                    LiftedGuard((GuardAtom("OWN_TILE_HAS_ANIMAL", "EQ", 1), GuardAtom("OWN_TILE_YIELD", "GE", yield_units), GuardAtom("OWN_TILE_FED", "EQ", 1))),
                    AbstractAction("UNIT_REWRITE", match_op=match, unit_op=10),
                )))
    elif goal.goal_id == "inventory_evacuation":
        for item in range(12):
            for x, y in ((4, 4), (5, 4), (4, 5), (5, 5)):
                for match in (0, 1, 2, 3, 4):
                    out.append(_module(InvariantRule(
                        ModuleSketch("recovery", "INSTANT", goal.effect_class), "EACH_OWN_UNIT",
                        LiftedGuard((GuardAtom("OWN_INVENTORY", "GE", 1, arg=item), GuardAtom("OWN_WORKER_X", "EQ", x), GuardAtom("OWN_WORKER_Y", "EQ", y))),
                        AbstractAction("UNIT_REWRITE", match_op=match, unit_op=6),
                    )))
    elif goal.goal_id == "market_order_cap_protection":
        for route in route_ids:
            for hour in (0, 4, 8, 12, 16, 20):
                for hires in (1, 2, 4, 8):
                    out.append(_module(InvariantRule(
                        ModuleSketch("engine_mechanic", "INSTANT", goal.effect_class), "GLOBAL",
                        LiftedGuard((GuardAtom("HOUR", "GE", hour), GuardAtom("OWN_HIRES_TODAY", "GE", hires))),
                        AbstractAction("ROUTE_OVERRIDE", route_id=int(route)),
                    )))
    elif goal.goal_id == "production_deadline_recovery":
        for match in (0, 9, 11, 16, 17):
            for deadline in (480, 576, 624, 672, 696):
                out.append(_module(InvariantRule(
                    ModuleSketch("engine_mechanic", "INSTANT", goal.effect_class), "EACH_OWN_UNIT",
                    LiftedGuard((GuardAtom("STEP", "GE", deadline), GuardAtom("OWN_TILE_KIND", "EQ", 5), GuardAtom("OWN_TILE_YIELD", "GE", 1), GuardAtom("OWN_TILE_MAX_LIFESPAN_STEP", "LE", 720))),
                    AbstractAction("UNIT_REWRITE", match_op=match, unit_op=10),
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
    """Combine only modules whose computed composition metadata proves no direct conflict."""
    from tools.bkg4_search.invariant_specialists import composition_metadata

    a.validate(); b.validate()
    if not composition_metadata(a).compatible_with(composition_metadata(b)):
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


def local_structural_neighborhood(
    parent: InvariantStrategicModule,
    *,
    route_ids: Sequence[int],
    budget: int,
    seed: int,
) -> tuple[InvariantStrategicModule, ...]:
    """Generate bounded structural contrasts around a useful specialist."""
    parent.validate()
    budget = max(0, int(budget))
    if budget == 0 or not parent.rules:
        return ()
    candidates: list[InvariantStrategicModule] = []
    for rule_index, rule in enumerate(parent.rules):
        atoms = list(rule.guard.atoms)
        for atom_index, atom in enumerate(atoms):
            for comparator in ("LE", "GE", "LT", "GT"):
                if comparator == atom.cmp:
                    continue
                changed = list(atoms); changed[atom_index] = replace(atom, cmp=comparator)
                rules = list(parent.rules); rules[rule_index] = replace(rule, guard=LiftedGuard(tuple(changed)))
                candidates.append(InvariantStrategicModule(tuple(rules)))
            for value in sorted({atom.value - max(1, abs(atom.value) // 2), atom.value + max(1, abs(atom.value) // 2)}):
                changed = list(atoms); changed[atom_index] = replace(atom, value=value)
                rules = list(parent.rules); rules[rule_index] = replace(rule, guard=LiftedGuard(tuple(changed)))
                candidates.append(InvariantStrategicModule(tuple(rules)))
        if atoms:
            rules = list(parent.rules); rules[rule_index] = replace(rule, guard=LiftedGuard(tuple(atoms[:-1])))
            candidates.append(InvariantStrategicModule(tuple(rules)))
        if len(atoms) + len(rule.exit_guard.atoms) < MAX_GUARDS_TOTAL:
            alternatives = (
                GuardAtom("DAY", "GE", 1), GuardAtom("OWN_MONEY", "GE", 500),
                GuardAtom("OWN_SHED_TOTAL", "GE", 75), GuardAtom("OWN_UNITS", "GE", 2),
            ) if rule.domain == "GLOBAL" else (
                GuardAtom("OWN_TILE_KIND", "NE", 1), GuardAtom("OWN_TILE_YIELD", "GE", 1),
                GuardAtom("OWN_INVENTORY", "GE", 1, arg=0),
            )
            for extra in alternatives:
                if extra in atoms:
                    continue
                rules = list(parent.rules); rules[rule_index] = replace(rule, guard=LiftedGuard(tuple((*atoms, extra))))
                candidates.append(InvariantStrategicModule(tuple(rules)))
                if atoms:
                    replacement = list(atoms); replacement[0] = extra
                    rules = list(parent.rules); rules[rule_index] = replace(rule, guard=LiftedGuard(tuple(replacement)))
                    candidates.append(InvariantStrategicModule(tuple(rules)))
        if rule.action.kind in {"ROUTE_OVERRIDE", "COMMITTED_ROUTE"}:
            for route in route_ids:
                if int(route) == int(rule.action.route_id):
                    continue
                rules = list(parent.rules); rules[rule_index] = replace(rule, action=replace(rule.action, route_id=int(route)))
                candidates.append(InvariantStrategicModule(tuple(rules)))
            if rule.action.kind == "ROUTE_OVERRIDE" and atoms:
                atom = atoms[0]
                exit_cmp = "GE" if atom.cmp in {"LE", "LT"} else "LE"
                rules = list(parent.rules); rules[rule_index] = replace(
                    rule,
                    sketch=replace(rule.sketch, control_pattern="COMMITTED"),
                    action=replace(rule.action, kind="COMMITTED_ROUTE"),
                    exit_guard=LiftedGuard((replace(atom, cmp=exit_cmp),)),
                )
                candidates.append(InvariantStrategicModule(tuple(rules)))
        elif rule.action.kind == "MARKET_INTENT":
            quantity = int(rule.action.quantity)
            for candidate_quantity in sorted({1, max(1, quantity - 1), quantity + 1, min(32767, quantity * 2)}):
                if candidate_quantity == quantity:
                    continue
                rules = list(parent.rules)
                rules[rule_index] = replace(rule, action=replace(rule.action, quantity=candidate_quantity))
                candidates.append(InvariantStrategicModule(tuple(rules)))
            for priority in sorted({max(0, rule.action.priority - 16), min(255, rule.action.priority + 16)}):
                if priority == rule.action.priority:
                    continue
                rules = list(parent.rules)
                rules[rule_index] = replace(rule, action=replace(rule.action, priority=priority))
                candidates.append(InvariantStrategicModule(tuple(rules)))
        elif rule.action.kind == "UNIT_REWRITE":
            for match_op in (0, 9, 11, 15, 16, 17):
                if match_op == rule.action.match_op:
                    continue
                rules = list(parent.rules)
                rules[rule_index] = replace(rule, action=replace(rule.action, match_op=match_op))
                candidates.append(InvariantStrategicModule(tuple(rules)))
            for unit_op in (0, 6, 9, 10, 12, 15, 17):
                if unit_op == rule.action.unit_op:
                    continue
                rules = list(parent.rules)
                rules[rule_index] = replace(
                    rule, action=replace(rule.action, unit_op=unit_op, unit_arg=0, quantity=1),
                )
                candidates.append(InvariantStrategicModule(tuple(rules)))
    rng = random.Random(int(seed) ^ 0x4E45494748424F52)
    rng.shuffle(candidates)
    unique: dict[str, InvariantStrategicModule] = {}
    for candidate in candidates:
        try:
            candidate.validate()
        except ValueError:
            continue
        if candidate.semantic_id() == parent.semantic_id():
            continue
        unique.setdefault(candidate.semantic_id(), candidate)
        if len(unique) >= budget:
            break
    return tuple(unique.values())


def structural_mcts(
    roots: Sequence[InvariantStrategicModule],
    goals: Sequence[SearchGoal],
    *,
    budget: int,
    seed: int,
    route_ids: Sequence[int],
    quality: Mapping[str, float] | None = None,
) -> tuple[MctsProposal, ...]:
    """Bounded grammar-expansion MCTS/UCB with exact generating-goal provenance."""
    roots = tuple(roots) or (InvariantStrategicModule(),)
    goals = tuple(goals)
    if not goals or budget <= 0:
        return ()
    q = dict(quality or {})
    visits: dict[str, int] = {m.semantic_id(): 1 for m in roots}
    reward: dict[str, float] = {m.semantic_id(): float(q.get(m.semantic_id(), 0.0)) for m in roots}
    goal_paths: dict[str, tuple[str, ...]] = {m.semantic_id(): () for m in roots}
    frontier = list(roots); emitted: dict[str, MctsProposal] = {}
    rng = random.Random(int(seed))
    for step in range(int(budget)):
        total = sum(visits.values()) + 1
        def ucb(m: InvariantStrategicModule) -> tuple[float, str]:
            sid = m.semantic_id(); n = max(1, visits.get(sid, 1)); mean = reward.get(sid, 0.0) / n
            return (mean + math.sqrt(2.0 * math.log(total) / n), sid)
        parent = max(frontier, key=ucb)
        goal = goals[(step + rng.randrange(len(goals))) % len(goals)]
        child = large_neighborhood(parent, goal, seed=seed + step * 1009, route_ids=route_ids)
        parent_sid = parent.semantic_id()
        sid = child.semantic_id(); visits[parent_sid] = visits.get(parent_sid, 1) + 1
        if sid not in emitted:
            path = (*goal_paths.get(parent_sid, ()), goal.goal_id)
            proposal = MctsProposal(child, tuple(path))
            emitted[sid] = proposal
            goal_paths[sid] = proposal.source_goal_path
            frontier.append(child); visits[sid] = 1; reward[sid] = float(q.get(sid, 0.0))
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
    elif (
        failure_class == "lifecycle_missing"
        and r.action.kind == "ROUTE_OVERRIDE"
        and r.guard.atoms
        and len(r.guard.atoms) + len(r.exit_guard.atoms) < MAX_GUARDS_TOTAL
    ):
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


def pair_relative_case_rows(parent: StateEvaluation, candidate: StateEvaluation) -> tuple[dict[str, Any], ...]:
    """Pair exact child outcomes against its immediate parent, including trace-level change."""
    rows = [dict(row) for row in pair_case_rows(parent, candidate)]
    for index, row in enumerate(rows):
        parent_trace = int(parent.trace_hashes[index])
        candidate_trace = int(candidate.trace_hashes[index])
        row.update({
            "parent_trace_hash": parent_trace,
            "candidate_trace_hash": candidate_trace,
            "behavior_changed_vs_parent": parent_trace != candidate_trace,
            "parent_action_changes": int(parent.action_changes[index]),
            "candidate_action_changes": int(candidate.action_changes[index]),
            "parent_activations": int(parent.activations[index]),
            "candidate_activations": int(candidate.activations[index]),
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
        specialist_archive: Any | None = None,
    ):
        self.route_ids = tuple(int(x) for x in route_ids)
        if not self.route_ids:
            raise ValueError("InvariantSearchBackend requires route IDs")
        self.evaluator = evaluator
        self.goals = tuple(goals or default_search_goals(self.route_ids))
        self.archive = archive or InvariantSearchArchive()
        if specialist_archive is None:
            from tools.bkg4_search.invariant_specialists import SpecialistArchive
            specialist_archive = SpecialistArchive()
        self.specialist_archive = specialist_archive
        self.ladder = ladder or CapabilityLadder()
        self.portable_route_seeds = tuple(
            translate_route_genome(genome, route_ids=self.route_ids)
            for genome in portable_route_seeds
        )
        self._cegis_queue: dict[str, InvariantStrategicModule] = {}
        self._cegis_parent: dict[str, str] = {}
        self._threshold_evidence: dict[str, dict[str, dict[str, set[tuple[int, bool, bool]]]]] = {}
        self.last_proposal_profile = {
            name: 0.0 for name in (
                "goal_construction", "composition", "boundary_solver", "lns", "local_neighborhood",
                "recombination", "mcts", "cegis", "canonical_dedupe",
            )
        }
        self.last_proposal_origins: dict[str, ProposalOrigin] = {}

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

    def _queue_cegis_repair(
        self,
        parent: InvariantStrategicModule,
        repaired: InvariantStrategicModule,
    ) -> None:
        sid = repaired.semantic_id()
        if sid == parent.semantic_id():
            return
        self._cegis_queue[sid] = repaired
        self._cegis_parent[sid] = parent.semantic_id()

    def consume_cegis_repairs(self) -> tuple[tuple[str, InvariantStrategicModule], ...]:
        rows = tuple(
            (self._cegis_parent.get(sid, ""), module)
            for sid, module in self._cegis_queue.items()
        )
        self._cegis_queue.clear(); self._cegis_parent.clear()
        return rows

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
        target_goal_ids: Sequence[str] | None = None,
    ) -> tuple[InvariantStrategicModule, ...]:
        n = max(0, int(n))
        profile = {name: 0.0 for name in self.last_proposal_profile}
        evidence = self.threshold_evidence()
        origins: dict[str, ProposalOrigin] = {}

        def note(
            module: InvariantStrategicModule,
            operator: str,
            source_goal_id: str | None = None,
            source_goal_path: Sequence[str] | None = None,
        ) -> None:
            sid = module.semantic_id()
            mechanisms = tuple(sorted({
                f"{rule.sketch.effect_class}:{rule.action.kind}" for rule in module.rules
            }))
            path = tuple(str(value) for value in source_goal_path) if source_goal_path is not None else (() if source_goal_id is None else (str(source_goal_id),))
            if source_goal_id is not None and (not path or path[-1] != str(source_goal_id)):
                raise ValueError("source goal path must end at source_goal_id")
            direct_goal = path[-1] if path else None
            origins.setdefault(sid, ProposalOrigin(str(operator), direct_goal, path, mechanisms, None))

        def source_goal(module: InvariantStrategicModule) -> str | None:
            origin = origins.get(module.semantic_id())
            if origin is not None and origin.source_goal_id:
                return origin.source_goal_id
            goal = self._goal_for_module(module)
            return goal.goal_id if goal is not None else None
        if n == 0:
            self.last_proposal_profile = profile
            self.last_proposal_origins = {}
            return ()
        started = time.perf_counter()
        selected_goals = self.goals
        if target_goal_ids is not None:
            wanted = {str(value) for value in target_goal_ids}
            unknown = sorted(wanted - {goal.goal_id for goal in self.goals})
            if unknown:
                raise ValueError(f"unknown targeted specialist goals: {unknown}")
            selected_goals = tuple(goal for goal in self.goals if goal.goal_id in wanted)
            if not selected_goals:
                raise ValueError("targeted proposal mode requires at least one goal")
        rows: list[InvariantStrategicModule] = []
        advanced: list[InvariantStrategicModule] = []
        if target_goal_ids is None:
            for module in mechanism_anchors(self.route_ids):
                advanced.append(module); note(module, "mechanism_anchor")
        goal_rows: list[tuple[SearchGoal, tuple[InvariantStrategicModule, ...]]] = []
        for gi, goal in enumerate(selected_goals):
            goal_rows.append((goal, emit_skeletons(goal, n=n, seed=seed + gi * 65537, route_ids=self.route_ids)))
        for index in range(max((len(family) for _, family in goal_rows), default=0)):
            for goal, family in goal_rows:
                if index < len(family):
                    module = family[index]
                    rows.append(module); note(module, "goal_emitter", goal.goal_id)
        profile["goal_construction"] += time.perf_counter() - started
        if self.ladder.level >= 1:
            started = time.perf_counter()
            for module in self.portable_route_seeds:
                advanced.append(module); note(module, "portable_seed")
            for module in compose_structural_modules(rows, budget=max(1, n // 3), seed=seed):
                advanced.append(module); note(module, "structural_composition")
            profile["composition"] += time.perf_counter() - started
        if self.ladder.level >= 2:
            started = time.perf_counter()
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
            for module in solved:
                advanced.append(module); note(module, "boundary_solver", source_goal(module))
            profile["boundary_solver"] += time.perf_counter() - started
        specialist_modules = tuple(
            evidence.module for evidence in self.specialist_archive.specialist_elites()
        )
        elites = tuple(dict.fromkeys((*self.archive.production_modules(), *specialist_modules)))
        if self.ladder.level >= 3 and elites:
            started = time.perf_counter()
            lns_parents = sorted(elites, key=lambda module: (-len(module.rules), module.semantic_id()))
            for i, parent in enumerate(lns_parents[:max(1, n // 16)]):
                goal = self._goal_for_module(parent) or selected_goals[i % len(selected_goals)]
                child = large_neighborhood(parent, goal, seed=seed + i * 8191, route_ids=self.route_ids)
                if child.semantic_id() != parent.semantic_id():
                    advanced.append(child); note(child, "lns", source_goal(parent) or goal.goal_id)
            profile["lns"] += time.perf_counter() - started
            started = time.perf_counter()
            for i, parent in enumerate(elites[:max(1, n // 8)]):
                children = local_structural_neighborhood(
                    parent, route_ids=self.route_ids, budget=max(2, n // max(1, len(elites))), seed=seed + i,
                )
                for child in children:
                    advanced.append(child); note(child, "local_neighborhood", source_goal(parent))
            profile["local_neighborhood"] += time.perf_counter() - started
        if self.ladder.level >= 4 and len(elites) >= 2:
            started = time.perf_counter()
            recombined = 0
            recombination_limit = max(1, n // 12)
            pool = elites[:min(len(elites), 32)]
            for i, first in enumerate(pool):
                for second in pool[i + 1:]:
                    child = recombine_compatible(first, second)
                    if child is None:
                        continue
                    advanced.append(child); note(child, "recombination")
                    recombined += 1
                    if recombined >= recombination_limit:
                        break
                if recombined >= recombination_limit:
                    break
            profile["recombination"] += time.perf_counter() - started
        if self.ladder.level >= 5:
            started = time.perf_counter()
            for proposal in structural_mcts(
                elites or (InvariantStrategicModule(),), self.goals,
                budget=max(1, n // 6), seed=seed, route_ids=self.route_ids,
            ):
                module = proposal.module
                advanced.append(module)
                note(
                    module, "mcts", proposal.source_goal_id,
                    source_goal_path=proposal.source_goal_path,
                )
            profile["mcts"] += time.perf_counter() - started
        if self.ladder.level >= 6 and self._cegis_queue:
            started = time.perf_counter()
            queued = list(self._cegis_queue.items())
            queued_modules = []
            for sid, module in queued:
                queued_modules.append(module); note(module, "cegis_repair")
            advanced = queued_modules + advanced
            self._cegis_queue.clear(); self._cegis_parent.clear()
            profile["cegis"] += time.perf_counter() - started
        if self.ladder.level >= 7:
            started = time.perf_counter()
            engine_goals = tuple(goal for goal in selected_goals if goal.effect_class == "ENGINE_MECHANIC")
            for gi, goal in enumerate(engine_goals):
                for module in emit_skeletons(
                    goal, n=max(1, n // max(1, len(engine_goals))),
                    seed=seed + gi * 7919, route_ids=self.route_ids,
                ):
                    advanced.append(module); note(module, "engine_goal_emitter", goal.goal_id)
            profile["goal_construction"] += time.perf_counter() - started
        if self.ladder.level >= 8 and specialist_modules:
            started = time.perf_counter()
            for module in compose_structural_modules(specialist_modules, budget=max(1, n // 2), seed=seed):
                advanced.append(module); note(module, "specialist_composition")
            from tools.bkg4_search.invariant_factory import compose_gated_specialists
            gated = 0
            for index, first in enumerate(specialist_modules):
                for second in specialist_modules[index + 1:]:
                    try:
                        child = compose_gated_specialists(first, second)
                    except ValueError:
                        continue
                    advanced.append(child); note(child, "specialist_gating"); gated += 1
                    if gated >= max(1, n // 4):
                        break
                if gated >= max(1, n // 4):
                    break
            profile["composition"] += time.perf_counter() - started
        if self.ladder.level >= 9 and specialist_modules:
            started = time.perf_counter()
            for i, parent in enumerate(specialist_modules[:max(1, n // 4)]):
                for child in local_structural_neighborhood(
                    parent, route_ids=self.route_ids, budget=8, seed=seed + i * 3571,
                ):
                    advanced.append(child); note(child, "local_neighborhood", source_goal(parent))
            profile["local_neighborhood"] += time.perf_counter() - started
        started = time.perf_counter()
        # Preserve at least one candidate from every advanced operator that produced a
        # canonical program instead of allowing early high-volume operators to starve later ones.
        buckets: dict[str, list[InvariantStrategicModule]] = {}
        for module in advanced:
            operator = origins.get(module.semantic_id(), ProposalOrigin("unknown", None, (), (), None)).operator
            buckets.setdefault(operator, []).append(module)
        ordered_advanced: list[InvariantStrategicModule] = []
        operator_order = tuple(buckets)
        cursor = 0
        while operator_order and any(buckets[name] for name in operator_order):
            name = operator_order[cursor % len(operator_order)]
            if buckets[name]:
                ordered_advanced.append(buckets[name].pop(0))
            cursor += 1
        seen: set[str] = set(); unique: list[InvariantStrategicModule] = []
        for m in (*ordered_advanced, *rows):
            sid = m.semantic_id()
            if sid in seen:
                continue
            seen.add(sid); unique.append(m)
            if len(unique) >= n:
                break
        final_origins: dict[str, ProposalOrigin] = {}
        for module in unique:
            sid = module.semantic_id()
            base = origins.get(sid, ProposalOrigin("unknown", None, (), (), None))
            retrospective = self._goal_for_module(module)
            final_origins[sid] = replace(
                base,
                retrospective_goal_id=(retrospective.goal_id if retrospective is not None else None),
            )
        profile["canonical_dedupe"] += time.perf_counter() - started
        self.last_proposal_profile = profile
        self.last_proposal_origins = final_origins
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
        cohort_by_tape: dict[str, str] = {}
        for case in panel:
            tape = tape_group_id(case)
            cohort = str(case.family or "unclassified")
            previous = cohort_by_tape.setdefault(tape, cohort)
            if previous != cohort:
                raise ValueError(f"tape group {tape} spans multiple specialist cohorts")
        useful = production = inserted = unsafe = 0
        specialist_inserted = elite_specialists = strong_specialists = boundary_specialists = 0
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
            from tools.bkg4_search.invariant_specialists import analyze_specialist
            evidence = analyze_specialist(
                m, outcomes, cohort_by_group=cohort_by_tape,
                activation_count=sum(exact[m.semantic_id()].activations),
                action_change_count=sum(exact[m.semantic_id()].action_changes),
            )
            specialist_inserted += int(self.specialist_archive.consider(evidence))
            elite_specialists += int(evidence.tier in {"GLOBAL_SAFE", "ELITE_SPECIALIST"})
            strong_specialists += int(evidence.tier == "STRONG_SPECIALIST")
            boundary_specialists += int(evidence.tier == "BOUNDARY_SPECIALIST")
            boundary_queued = False
            if self.ladder.level >= 6 and evidence.useful_groups >= 2 and evidence.harmed_groups:
                from tools.bkg4_search.invariant_specialists import specialist_boundary_guards
                boundary_rows = specialist_boundary_guards(outcomes, panel, limit=4)
                for boundary in boundary_rows:
                    repaired_rules = list(m.rules)
                    target_index = next((
                        index for index, rule in enumerate(repaired_rules)
                        if rule.domain == "GLOBAL"
                        and len(rule.guard.atoms) + len(rule.exit_guard.atoms) < MAX_GUARDS_TOTAL
                        and boundary.atom not in rule.guard.atoms
                    ), None)
                    if target_index is None:
                        break
                    rule = repaired_rules[target_index]
                    repaired_rules[target_index] = replace(
                        rule, guard=LiftedGuard((*rule.guard.atoms, boundary.atom)),
                    )
                    repaired = InvariantStrategicModule(tuple(repaired_rules))
                    repaired.validate()
                    self._queue_cegis_repair(m, repaired)
                    boundary_queued = True
                    break
            if self.ladder.level >= 6 and not qual.production:
                if boundary_queued:
                    continue
                counterexamples = grouped_counterexample_rows(rows, case_to_tape)
                failure_class = classify_grouped_counterexamples(counterexamples)
                if derived and any(x.useful for x in outcomes) and any(x.margin_delta < 0 for x in outcomes):
                    failure_class = "unstable_constant"
                if goal is not None and any(x.useful for x in outcomes) and any(x.margin_delta < 0 for x in outcomes):
                    structural = large_neighborhood(
                        m, goal,
                        seed=int(m.semantic_id()[:8], 16) ^ len(counterexamples),
                        route_ids=self.route_ids,
                    )
                    self._queue_cegis_repair(m, structural)
                repaired = cegis_repair(
                    m, failure_class, route_ids=self.route_ids, goal=goal,
                    threshold_samples=(self.threshold_evidence().get(goal.goal_id, {}) if goal is not None else {}),
                )
                self._queue_cegis_repair(m, repaired)
        capability = self.ladder.observe(
            attempted=len(modules), unique=len({m.semantic_id() for m in modules}), useful=useful,
            production_qualified=production, archive_insertions=inserted + specialist_inserted,
        )
        return SearchRound(
            attempted=len(modules), unique=len({m.semantic_id() for m in modules}), duplicate_rate=0.0,
            useful=useful, production_qualified=production, archive_insertions=inserted, unsafe=unsafe,
            exact_candidate_cases=len(batch) * len(panel), capability=capability, evaluations=exact,
            specialist_elites=elite_specialists, strong_specialists=strong_specialists,
            boundary_specialists=boundary_specialists,
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
            or self._cegis_queue or self._cegis_parent or self.ladder.level or self.ladder._stalled
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
            tuple(sorted(self._cegis_parent.items())),
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
            tuple(sorted(self._cegis_parent.items())),
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
