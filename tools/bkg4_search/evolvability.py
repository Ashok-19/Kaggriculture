from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, List, Sequence, Tuple


@dataclass(frozen=True)
class FitnessVector:
    """Direct candidate evidence. Lower fields are safety gates, not averages."""
    catastrophic_regressions: int
    protected_regressions: int
    own_history_regressions: int
    losing_families: int
    losses: int
    ties: int
    wins: int
    worst_family_milli: int
    floor_margin: float
    mean_margin: float
    hot_visits: int = 0
    complexity: int = 0

    @property
    def cases(self) -> int:
        return self.wins + self.ties + self.losses

    def lexicographic_key(self) -> Tuple[float, ...]:
        # Major regression cannot be compensated by aggregate margin.
        return (
            self.catastrophic_regressions,
            self.protected_regressions,
            self.own_history_regressions,
            self.losing_families,
            self.losses,
            -self.worst_family_milli,
            -self.wins,
            self.ties,
            -self.floor_margin,
            -self.mean_margin,
            self.hot_visits,
            self.complexity,
        )

    def no_worse_than(self, other: "FitnessVector") -> bool:
        return self.lexicographic_key() <= other.lexicographic_key()


def direct_scalar(f: FitnessVector, floor: FitnessVector) -> float:
    """Monotone [0,1] search steering score within a direct-quality floor.

    This is *not* the promotion score. Candidates violating the floor get 0.
    Among viable candidates, family robustness dominates aggregate wins/margin.
    """
    if f.catastrophic_regressions > floor.catastrophic_regressions:
        return 0.0
    if f.protected_regressions > floor.protected_regressions:
        return 0.0
    if f.own_history_regressions > floor.own_history_regressions:
        return 0.0
    if f.losing_families > floor.losing_families + 1:
        return 0.0
    n = max(1, f.cases)
    wr = (f.wins + 0.5 * f.ties) / n
    fam = max(0.0, min(1.0, f.worst_family_milli / 1000.0))
    # Small bounded margin term only after robust outcome terms.
    margin = 0.5 + 0.5 * math.tanh(f.mean_margin / 20000.0)
    return max(0.0, min(1.0, 0.50 * fam + 0.45 * wr + 0.05 * margin))


@dataclass
class SearchNode:
    parent: int
    fitness: FitnessVector
    v: float
    visits: int = 1
    value_samples: int = 1
    lineage_sum: float = 0.0
    children: List[int] = field(default_factory=list)
    accepted: bool = True

    def __post_init__(self) -> None:
        if self.lineage_sum == 0.0 and self.value_samples == 1:
            self.lineage_sum = self.v

    @property
    def q(self) -> float:
        return self.lineage_sum / max(1, self.value_samples)

    @property
    def brittleness(self) -> float:
        return self.v - self.q


def ucb_score(node: SearchNode, parent_visits: int, exploration_c: float = 0.4, lambda_evo: float = 1.0) -> float:
    # Paper Eq. (3): v + lambda(Q-v) + standard exploration. v/Q are already [0,1].
    exploit = node.v + lambda_evo * (node.q - node.v)
    explore = exploration_c * math.sqrt(math.log(max(1, parent_visits)) / max(1, node.visits))
    return exploit + explore


class EvolvabilityTree:
    """Paper-faithful lineage-Q bookkeeping with separate visits/value samples."""
    def __init__(self, root_fitness: FitnessVector, direct_floor: FitnessVector):
        self.direct_floor = direct_floor
        v = direct_scalar(root_fitness, direct_floor)
        self.nodes = [SearchNode(parent=-1, fitness=root_fitness, v=v)]

    def add_accepted(self, parent: int, fitness: FitnessVector) -> int:
        v = direct_scalar(fitness, self.direct_floor)
        idx = len(self.nodes)
        self.nodes.append(SearchNode(parent=parent, fitness=fitness, v=v))
        self.nodes[parent].children.append(idx)
        cur = parent
        while cur >= 0:
            n = self.nodes[cur]
            n.visits += 1
            n.value_samples += 1
            n.lineage_sum += v
            cur = n.parent
        return idx

    def add_rejected_visit(self, parent: int) -> None:
        # Paper Eq. (6): rejection is evidence about where not to spend budget,
        # not a scored sample of region quality. N changes; m/W/Q do not.
        cur = parent
        while cur >= 0:
            self.nodes[cur].visits += 1
            cur = self.nodes[cur].parent

    def child_cap(self, node: int, c_w: float = 1.0, alpha: float = 0.5) -> int:
        return max(1, math.ceil(c_w * (self.nodes[node].value_samples ** alpha)))

    def select(self, start: int = 0, exploration_c: float = 0.4, lambda_evo: float = 1.0,
               c_w: float = 1.0, alpha: float = 0.5) -> int:
        cur = start
        while self.nodes[cur].children and len(self.nodes[cur].children) >= self.child_cap(cur, c_w, alpha):
            pv = self.nodes[cur].visits
            cur = max(self.nodes[cur].children, key=lambda i: ucb_score(self.nodes[i], pv, exploration_c, lambda_evo))
        return cur

    def best_direct(self) -> int:
        # Search steering and shipping are deliberately separate, as in the paper.
        viable = [i for i, n in enumerate(self.nodes) if n.v > 0.0]
        if not viable:
            return 0
        return min(viable, key=lambda i: self.nodes[i].fitness.lexicographic_key())


def sanity_fixture() -> dict:
    """Controlled robust/brittle/mediocre/weak landscape used by tests and CLI sanity checks."""
    elite = FitnessVector(0,0,0,0,0,0,20,1000,1000,10000)
    elite_child = FitnessVector(0,0,0,0,0,0,20,1000,900,9000)
    brittle_child = FitnessVector(0,0,0,3,7,0,13,500,-5000,-1500)
    mediocre = FitnessVector(0,0,0,1,2,1,17,750,-1000,1000)
    weak = FitnessVector(0,1,0,5,12,1,7,250,-12000,-6000)
    floor = FitnessVector(0,0,0,1,4,0,16,600,-5000,0)
    return {"elite":elite,"elite_child":elite_child,"brittle_child":brittle_child,"mediocre":mediocre,"weak":weak,"floor":floor}
