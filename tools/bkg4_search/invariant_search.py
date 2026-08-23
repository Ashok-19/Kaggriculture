from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Iterable, Mapping, Sequence

from tools.bkg4_search.edit_search.fitness import StateEvaluation
from tools.bkg4_search.edit_search.panels import CaseRecord
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


CEGIS_FAILURES = (
    "guard_too_broad",
    "wrong_action_effect",
    "unstable_constant",
    "resource_invariant",
    "action_conflict",
    "lifecycle_missing",
)


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
                    g = LiftedGuard((GuardAtom("MARKET_PRICE", "LE", 1, arg=item), GuardAtom("OWN_SHED", "GE", reserve, arg=item)))
                    out.append(_module(InvariantRule(ModuleSketch("market", "INSTANT", goal.effect_class), "GLOBAL", g,
                        AbstractAction("MARKET_INTENT", market_op=6, item=item, quantity=quantity, authority=5, priority=220))))
    elif goal.goal_id == "starving_animal_rescue":
        # FEED consumes the bound worker's inventory. Guard local/global inventory plus relational animal state.
        for match in (17, 16, 11):  # CARE, COLLECT_FERTILIZER, FERTILIZE -> FEED
            g = LiftedGuard((
                GuardAtom("OWN_TILE_HAS_ANIMAL", "EQ", 1),
                GuardAtom("OWN_TILE_CONSECUTIVE_UNFED", "GE", 1),
                GuardAtom("OWN_INVENTORY", "GE", 1, arg=0),
            ))
            out.append(_module(InvariantRule(ModuleSketch("recovery", "INSTANT", goal.effect_class), "EACH_OWN_UNIT", g,
                AbstractAction("UNIT_REWRITE", match_op=match, unit_op=15, unit_arg=0, quantity=1))))
    elif goal.goal_id == "unwatered_plant_rescue":
        # Preserve profitable HARVEST and intentional DIG decisions. Replacing only
        # FERTILIZE keeps this family a maintenance-priority correction.
        for debt in (1, 2):
            g = LiftedGuard((GuardAtom("OWN_TILE_CONSECUTIVE_UNWATERED", "GE", debt),))
            out.append(_module(InvariantRule(ModuleSketch("recovery", "INSTANT", goal.effect_class), "EACH_OWN_UNIT", g,
                AbstractAction("UNIT_REWRITE", match_op=11, unit_op=9))))
    elif goal.goal_id == "avoid_overbuy_inventory":
        for route in route_ids:
            for cap in (80, 90, 96, 99):
                g = LiftedGuard((GuardAtom("OWN_SHED_TOTAL", "GE", cap),))
                out.append(_module(InvariantRule(ModuleSketch("market", "INSTANT", goal.effect_class), "GLOBAL", g,
                    AbstractAction("ROUTE_OVERRIDE", route_id=int(route)))))
    elif goal.goal_id == "accelerate_expansion":
        for units in (2, 3, 4, 5):
            for cash in (200, 500, 1000, 2000):
                g = LiftedGuard((GuardAtom("OWN_UNITS", "LE", units), GuardAtom("OWN_MONEY", "GE", cash)))
                out.append(_module(InvariantRule(ModuleSketch("expansion", "INSTANT", goal.effect_class), "GLOBAL", g,
                    AbstractAction("MARKET_INTENT", market_op=1, quantity=1, authority=3, priority=180))))
    elif goal.goal_id == "committed_financing_mode":
        for route in route_ids:
            for enter, exit_ in ((500, 1200), (1000, 2500), (2000, 5000)):
                out.append(_module(InvariantRule(ModuleSketch("route", "COMMITTED", goal.effect_class), "GLOBAL",
                    LiftedGuard((GuardAtom("OWN_MONEY", "LE", enter),)), AbstractAction("COMMITTED_ROUTE", route_id=int(route)),
                    LiftedGuard((GuardAtom("OWN_MONEY", "GE", exit_),)))))
    elif goal.goal_id == "maintenance_redundancy":
        out.extend((
            _module(InvariantRule(ModuleSketch("maintenance", "INSTANT", goal.effect_class), "EACH_OWN_UNIT",
                LiftedGuard((GuardAtom("OWN_TILE_FED", "EQ", 1),)), AbstractAction("UNIT_REWRITE", match_op=15, unit_op=0))),
            _module(InvariantRule(ModuleSketch("maintenance", "INSTANT", goal.effect_class), "EACH_OWN_UNIT",
                LiftedGuard((GuardAtom("OWN_TILE_CARED", "EQ", 1),)), AbstractAction("UNIT_REWRITE", match_op=17, unit_op=0))),
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
    threshold_samples: Mapping[str, Sequence[tuple[int, bool, bool]]] | None = None,
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
        feature = "OWN_MONEY" if r.domain == "GLOBAL" else "OWN_UNITS"
        guard = LiftedGuard((*r.guard.atoms, GuardAtom(feature, "GE", 1)))
        rules[0] = replace(r, guard=guard)
    elif failure_class == "unstable_constant" and threshold_samples:
        atom_index = next((i for i, a in enumerate(r.guard.atoms) if a.cmp in {"GE", "GT", "LE", "LT"}), None)
        if atom_index is not None:
            atom = r.guard.atoms[atom_index]
            sol = solve_cross_tape_threshold(threshold_samples, comparator=atom.cmp)
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
    ):
        self.route_ids = tuple(int(x) for x in route_ids)
        if not self.route_ids:
            raise ValueError("InvariantSearchBackend requires route IDs")
        self.evaluator = evaluator
        self.goals = tuple(goals or default_search_goals(self.route_ids))
        self.archive = archive or InvariantSearchArchive()
        self.ladder = ladder or CapabilityLadder()
        self._cegis_queue: dict[str, InvariantStrategicModule] = {}

    def _goal_for_module(self, module: InvariantStrategicModule) -> SearchGoal | None:
        if not module.rules:
            return None
        effect = module.rules[0].sketch.effect_class
        return next((x for x in self.goals if x.effect_class == effect), None)

    def propose(
        self,
        *,
        n: int,
        seed: int,
        threshold_samples: Mapping[str, Mapping[str, Sequence[tuple[int, bool, bool]]]] | None = None,
    ) -> tuple[InvariantStrategicModule, ...]:
        n = max(0, int(n)); threshold_samples = dict(threshold_samples or {})
        if n == 0: return ()
        rows: list[InvariantStrategicModule] = []
        advanced: list[InvariantStrategicModule] = []
        goal_rows: list[tuple[InvariantStrategicModule, ...]] = []
        for gi, goal in enumerate(self.goals):
            goal_rows.append(emit_skeletons(goal, n=n, seed=seed + gi * 65537, route_ids=self.route_ids))
        for index in range(max((len(x) for x in goal_rows), default=0)):
            for family in goal_rows:
                if index < len(family):
                    rows.append(family[index])
        if self.ladder.level >= 2:
            # Boundary solver alters only a scalar on an already chosen structure.
            solved: list[InvariantStrategicModule] = []
            for m in rows:
                g = self.goals[next((i for i, x in enumerate(self.goals) if x.effect_class == m.rules[0].sketch.effect_class), 0)] if m.rules else None
                samples = threshold_samples.get(g.goal_id, {}) if g else {}
                if samples and m.rules and m.rules[0].guard.atoms:
                    r = m.rules[0]; atoms = list(r.guard.atoms)
                    ai = next((i for i, a in enumerate(atoms) if a.cmp in {"GE", "GT", "LE", "LT"}), None)
                    if ai is not None:
                        sol = solve_cross_tape_threshold(samples, comparator=atoms[ai].cmp)
                        atoms[ai] = replace(atoms[ai], value=sol.threshold)
                        solved.append(InvariantStrategicModule((replace(r, guard=LiftedGuard(tuple(atoms))),)))
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
            qual = production_qualification(outcomes)
            production += int(qual.production)
            useful += int(any(x.useful for x in outcomes))
            unsafe += int(any(x.new_loss or x.severe_regression for x in outcomes))
            inserted += int(self.archive.consider(m, outcomes))
            if self.ladder.level >= 6 and not qual.production:
                worst = min(rows, key=lambda x: (int(x["margin_delta"]), str(x["case_id"])))
                repaired = cegis_repair(
                    m, classify_counterexample(worst), route_ids=self.route_ids,
                    goal=self._goal_for_module(m),
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
        threshold_samples: Mapping[str, Mapping[str, Sequence[tuple[int, bool, bool]]]] | None = None,
    ) -> SearchRunResult:
        history: list[SearchRound] = []; attempted = exact_cc = 0
        seen: set[str] = set()
        for ri in range(max(1, int(rounds))):
            proposed = [m for m in self.propose(n=candidates_per_round, seed=seed + ri * 100003, threshold_samples=threshold_samples) if m.semantic_id() not in seen]
            for m in proposed: seen.add(m.semantic_id())
            if not proposed:
                event = self.ladder.observe(attempted=candidates_per_round, unique=0, useful=0)
                history.append(SearchRound(candidates_per_round, 0, 1.0, 0, 0, 0, 0, 0, event, {}))
                continue
            rr = self.evaluate_batch(proposed, panel, work_dir=work_dir, panel_id=f"invariant-search/r{ri}")
            history.append(rr); attempted += len(proposed); exact_cc += rr.exact_candidate_cases
        return SearchRunResult(tuple(history), self.archive, attempted, exact_cc)
