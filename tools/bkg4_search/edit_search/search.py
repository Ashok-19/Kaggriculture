from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Mapping, Sequence

from tools.bkg4_search.information_state_io import registry_manifest
from tools.bkg4_search.module_codec import ConditionGene

from . import GRAMMAR_ID
from .archive import SearchArchive
from .emitters import (
    DEFAULT_EMITTERS,
    EmitterContext,
    EvolvabilityMCTSEmitter,
    SearchEmitter,
)
from .fitness import PairEvaluation, StateEvaluation
from .certification import (
    CertificationClass, CertificationVector, certification_rank, certify_programs_exact,
    evaluate_certified_neighborhood_exact,
)
from .repair import run_cegis_repair
from .goals import SearchGoal, build_search_goals, synthesize_contrastive_conditions
from .panels import CaseBank, CasePanelManager
from .safety import SafetySpec
from .schema import EditProgram


def _registry_identity() -> str:
    raw = json.dumps(registry_manifest(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(("BKG4-PUBLIC-REGISTRY-V1\0" + raw).encode()).hexdigest()


def contrastive_cache_key(bank_identity: str, goal_id: str, *, grammar_id: str = GRAMMAR_ID, registry_identity: str | None = None) -> str:
    payload = json.dumps({
        "bank": str(bank_identity), "goal": str(goal_id), "grammar": str(grammar_id),
        "registry": str(registry_identity or _registry_identity()),
    }, sort_keys=True, separators=(",", ":"))
    return "goalcache:" + hashlib.sha256(("BKG4-CONTRASTIVE-CACHE-V1\0" + payload).encode()).hexdigest()


def _serialize_condition_sets(rows: Sequence[Sequence[ConditionGene]]) -> list[list[dict[str, object]]]:
    return [[{"feature": c.feature, "cmp": c.cmp, "index": c.index, "value": c.value} for c in conds] for conds in rows]


def _deserialize_condition_sets(rows: Sequence[Sequence[Mapping[str, object]]]) -> tuple[tuple[ConditionGene, ...], ...]:
    return tuple(tuple(ConditionGene(str(c["feature"]), str(c["cmp"]), int(c.get("index", 0)), int(c["value"])) for c in conds) for conds in rows)


_BASE_WEIGHTS = {
    "contrastive": 0.18,
    "systematic": 0.16,
    "local_refine": 0.13,
    "structural": 0.15,
    "motif": 0.07,
    "recombine": 0.08,
    "lns": 0.07,
    "evolvability_mcts": 0.11,
    "random_typed": 0.05,
}


class FixedEmitterScheduler:
    """Reference scheduler. Escalation changes neighborhoods but never removes exploration."""

    def __init__(self, random_floor: float = 0.05):
        self.random_floor = max(0.01, min(0.25, float(random_floor)))

    def weights(self, escalation_level: int, emitter_stats: Mapping[str, Mapping[str, int]] | None = None) -> dict[str, float]:
        w = dict(_BASE_WEIGHTS); level = max(0, int(escalation_level))
        if level >= 2:
            w["structural"] += 0.08; w["systematic"] -= 0.04; w["local_refine"] -= 0.04
        if level >= 3:
            w["recombine"] += 0.10; w["systematic"] -= 0.05; w["local_refine"] -= 0.05
        if level >= 4:
            w["lns"] += 0.15; w["systematic"] -= 0.05; w["local_refine"] -= 0.05; w["structural"] -= 0.05
        if level >= 5:
            w["evolvability_mcts"] += 0.15; w["local_refine"] -= 0.05; w["systematic"] -= 0.05; w["structural"] -= 0.05
        if level >= 6:
            w["random_typed"] += 0.07; w["recombine"] += 0.03; w["local_refine"] -= 0.05; w["systematic"] -= 0.05
        # Multi-emitter adaptation: exact safe contribution changes future budget,
        # while untried emitters retain their prior and random exploration keeps a floor.
        stats = dict(emitter_stats or {})
        for key in w:
            row = dict(stats.get(key) or {})
            attempts = max(0, int(row.get("attempts", 0)))
            if attempts < 4:
                continue
            useful = max(0, int(row.get("useful_safe", 0)))
            production = max(0, int(row.get("production_insertions", row.get("archive_insertions", 0))))
            repair = max(0, int(row.get("repair_insertions", 0)))
            unsafe = max(0, int(row.get("unsafe", 0)))
            new_loss = max(0, int(row.get("new_loss", 0)))
            productive = min(1.0, (useful + 0.50 * production) / max(1, attempts))
            danger = min(1.0, max(unsafe, new_loss) / max(1, attempts))
            repair_credit = min(0.20, 0.20 * repair / max(1, attempts))
            multiplier = (0.50 + 2.50 * productive) * (1.0 - 0.75 * danger) + repair_credit
            w[key] *= max(0.20, multiplier)
        for key in w: w[key] = max(0.01, w[key])
        w["random_typed"] = max(self.random_floor, w["random_typed"])
        z = sum(w.values())
        return {k: v / z for k, v in w.items()}

    def refill_order(self, emitter_stats: Mapping[str, Mapping[str, int]] | None = None) -> tuple[str, ...]:
        """Safety-first order for reclaiming conditionally unused proposal budget."""
        names = ("local_refine", "random_typed", "evolvability_mcts", "structural", "systematic")
        prior = {name: i for i, name in enumerate(names)}
        stats = dict(emitter_stats or {})
        def key(name: str) -> tuple[float, ...]:
            row = dict(stats.get(name) or {})
            attempts = max(0, int(row.get("attempts", 0)))
            if attempts:
                unsafe = max(int(row.get("unsafe", 0)), int(row.get("new_loss", 0))) / attempts
                productive = (int(row.get("useful_safe", 0)) + 0.5 * int(row.get("production_insertions", 0))) / attempts
            else:
                unsafe = 0.50
                productive = 0.0
            return (float(unsafe), -float(productive), float(prior[name]), name)
        return tuple(sorted(names, key=key))

    def allocations(self, total: int, escalation_level: int, emitter_stats: Mapping[str, Mapping[str, int]] | None = None) -> dict[str, int]:
        total = max(0, int(total)); names = list(_BASE_WEIGHTS)
        if total == 0: return {k: 0 for k in names}
        weights = self.weights(escalation_level, emitter_stats=emitter_stats)
        raw = {k: total * weights[k] for k in names}
        out = {k: int(math.floor(raw[k])) for k in names}
        # Preserve a permanent random floor even at very small budgets.
        random_min = min(total, max(1, int(math.ceil(total * self.random_floor))))
        if out["random_typed"] < random_min:
            need = random_min - out["random_typed"]
            out["random_typed"] += need
            donors = sorted((k for k in names if k != "random_typed"), key=lambda k: (-out[k], k))
            for k in donors:
                take = min(need, max(0, out[k]))
                out[k] -= take; need -= take
                if need == 0: break
        left = total - sum(out.values())
        order = sorted(names, key=lambda k: (-(raw[k] - math.floor(raw[k])), k))
        i = 0
        while left > 0:
            out[order[i % len(order)]] += 1; left -= 1; i += 1
        while left < 0:
            candidates = [k for k in names if k != "random_typed" and out[k] > 0]
            if not candidates: break
            k = max(candidates, key=lambda x: out[x]); out[k] -= 1; left += 1
        return out


@dataclass(frozen=True)
class GenerationSignals:
    proposed: int
    unique: int
    archive_insertions: int
    useful_safe: int
    target_improvements: int
    novelty_mean: float
    new_niches: int
    duplicate_rate: float = 0.0
    production_insertions: int = 0
    repair_insertions: int = 0
    underfill_rate: float = 0.0
    source_parent_count: int = 0
    operation_count: int = 0
    discovery_elite: int = 0
    certified_safe_useful: int = 0
    certified_elite: int = 0
    repair_to_feasible: int = 0
    generalization_improved: int = 0


@dataclass(frozen=True)
class EscalationEvent:
    level: int
    escalated: bool
    reason: str


@dataclass(frozen=True)
class RoundResult:
    generation: int
    screened_program_ids: tuple[str, ...]
    finalist_program_ids: tuple[str, ...]
    screen_source_pairs: Mapping[str, PairEvaluation]
    full_source_pairs: Mapping[str, PairEvaluation]
    full_root_pairs: Mapping[str, PairEvaluation]
    counterexamples: tuple[str, ...]
    stage_recall: Mapping[str, float] | None
    signals: GenerationSignals
    escalation: EscalationEvent
    certification_vectors: Mapping[str, CertificationVector] | None = None
    repair_reports: tuple[Mapping[str, object], ...] = ()


_ESCALATION_REASONS = {
    1: "wider contrastive semantic thresholds",
    2: "alternate failure clusters and broaden QD parents",
    3: "increase non-temporal guard synthesis and structural mutation",
    4: "stronger CEGIS repair and large-neighborhood repair",
    5: "cross-context temporal robustness search",
    6: "recombine certified-safe motifs",
    7: "near-feasible repair search",
    8: "grammar-capacity diagnostic and rare-niche exploration",
}


class StagnationMonitor:
    def __init__(self, *, patience: int = 3, max_level: int = 8):
        self.patience = max(1, int(patience)); self.max_level = max(0, int(max_level))
        self.level = 0; self._stalled = 0

    @staticmethod
    def is_stalled(s: GenerationSignals) -> bool:
        unique_rate = s.unique / max(1, s.proposed)
        # Novel research behavior is useful evidence, but it must not reset a search
        # that is making no progress toward feasibility or strategic repair.
        certified_progress = s.certified_safe_useful > 0 or s.certified_elite > 0 or s.repair_to_feasible > 0 or s.generalization_improved > 0
        objective_plateau = s.useful_safe <= 0 and s.target_improvements <= 0 and s.repair_insertions <= 0 and not certified_progress
        search_collapse = unique_rate < 0.35 or float(s.duplicate_rate) >= 0.70 or float(s.underfill_rate) >= 0.50
        archive_plateau = s.production_insertions <= 0 and s.repair_insertions <= 0 and s.useful_safe <= 0 and not certified_progress
        return search_collapse or objective_plateau or archive_plateau

    def observe(self, signals: GenerationSignals) -> EscalationEvent:
        if self.is_stalled(signals):
            self._stalled += 1
        else:
            self._stalled = 0
        if self._stalled >= self.patience and self.level < self.max_level:
            self.level += 1; self._stalled = 0
            return EscalationEvent(self.level, True, _ESCALATION_REASONS.get(self.level, "expanded search regime"))
        return EscalationEvent(self.level, False, "healthy" if not self.is_stalled(signals) else "stagnation accumulating")

    def to_dict(self) -> dict[str, int]:
        return {"level": int(self.level), "stalled": int(self._stalled), "patience": int(self.patience), "max_level": int(self.max_level)}

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> "StagnationMonitor":
        obj = {str(k): int(v) for k, v in raw.items()}
        if set(obj) != {"level", "stalled", "patience", "max_level"}:
            raise ValueError("invalid persisted stagnation state")
        out = cls(patience=obj["patience"], max_level=obj["max_level"])
        if obj["level"] < 0 or obj["level"] > out.max_level or obj["stalled"] < 0:
            raise ValueError("invalid persisted stagnation values")
        out.level = obj["level"]; out._stalled = obj["stalled"]
        return out


def lexicase_parent(evaluations: Mapping[str, StateEvaluation], *, seed: int, epsilon: int = 0) -> str:
    """Case-wise specialist parent selection; aggregate margin never replaces the case filters."""
    if not evaluations:
        raise ValueError("lexicase selection requires evaluated parents")
    ids = sorted(evaluations)
    first = evaluations[ids[0]]
    if any(x.panel_id != first.panel_id or x.case_ids != first.case_ids for x in evaluations.values()):
        raise ValueError("lexicase parent evaluations must share one aligned panel")
    order = list(range(len(first.case_ids))); random.Random(int(seed)).shuffle(order)
    survivors = ids
    for ci in order:
        best = max(evaluations[sid].margins[ci] for sid in survivors)
        survivors = [sid for sid in survivors if evaluations[sid].margins[ci] >= best - int(epsilon)]
        if len(survivors) <= 1: break
    return random.Random(int(seed) ^ 0x1E71CA5E).choice(sorted(survivors))

def exact_state_quality_map(evaluations: Mapping[str, StateEvaluation]) -> dict[str, float]:
    """Panel-local exact quality for MCTS/UCB; incompatible certifications are never mixed."""
    if not evaluations:
        return {}
    ids = sorted(evaluations); first = evaluations[ids[0]]
    if any(ev.panel_id != first.panel_id or ev.case_ids != first.case_ids for ev in evaluations.values()):
        raise ValueError("exact state quality requires one aligned full-panel certification")
    out: dict[str, float] = {}
    for sid in ids:
        margins = evaluations[sid].margins
        losses = sum(int(x) < 0 for x in margins)
        ties = sum(int(x) == 0 for x in margins)
        worst = min(margins); mean = sum(margins) / len(margins)
        out[sid] = -4.0 * losses - 0.25 * ties + math.tanh(worst / 5000.0) + 0.25 * math.tanh(mean / 5000.0)
    return out

def certification_quality_map(classes: Mapping[str, CertificationClass | str]) -> dict[str, float]:
    """Categorical quality tiers for MCTS; gaps preserve certification ordering."""
    return {str(sid): float(1000 - 100 * certification_rank(cls)) for sid, cls in classes.items()}


def archive_quality_map(
    archive: SearchArchive,
    local_quality: Mapping[str, float],
    *,
    certification_classes: Mapping[str, CertificationClass | str] | None = None,
) -> dict[str, float]:
    """Lexicographic search-state quality with certification dominating local panel score."""
    classes: dict[str, CertificationClass] = {}
    if certification_classes is not None:
        classes = {str(k): (v if isinstance(v, CertificationClass) else CertificationClass(str(v))) for k, v in certification_classes.items()}
    else:
        for pid, raw in archive.certification_evidence.items():
            program = archive.programs.get(str(pid))
            if program is None:
                continue
            vector = archive.certification_vector(str(pid))
            if vector is None:
                continue
            sid = program.child_semantic_id
            prior = classes.get(sid)
            if prior is None or certification_rank(vector.certification_class) < certification_rank(prior):
                classes[sid] = vector.certification_class
    production = set(archive.production_parent_state_ids())
    repair = set(archive.repair_parent_state_ids(limit=max(16, len(archive.states))))
    out: dict[str, float] = {}
    for sid, local in local_quality.items():
        cls = classes.get(str(sid))
        if cls in {CertificationClass.CERTIFIED_ELITE_FLIP, CertificationClass.CERTIFIED_ELITE_MARGIN}:
            band = 1000.0 if cls == CertificationClass.CERTIFIED_ELITE_FLIP else 950.0
        elif cls == CertificationClass.CERTIFIED_SAFE_USEFUL:
            band = 900.0
        elif cls == CertificationClass.CERTIFIED_SAFE_INCREMENTAL:
            band = 800.0
        elif cls == CertificationClass.GENERALIZATION_HARD_NEGATIVE:
            band = 50.0
        elif cls == CertificationClass.GENERALIZATION_BOUNDARY:
            band = 150.0
        elif sid in production:
            band = 600.0
        elif sid in repair:
            band = 250.0
        else:
            band = 100.0
        out[str(sid)] = band + 0.1 * math.tanh(float(local) / 10.0)
    return out


def select_stage2_finalists(
    programs: Sequence[EditProgram],
    root_pairs: Mapping[str, PairEvaluation],
    *,
    budget: int,
    max_noop_fraction: float = 0.10,
) -> list[EditProgram]:
    """Quality-first Stage-2 selection with an explicit exact-screen no-op ceiling."""
    budget = min(max(0, int(budget)), len(programs))
    if budget <= 0:
        return []
    frac = float(max_noop_fraction)
    if not 0.0 <= frac < 1.0:
        raise ValueError("max_noop_fraction must be in [0,1)")
    ordered = SearchEngine._stage2_order(programs, root_pairs)
    informative = [p for p in ordered if not bool(root_pairs[p.program_id].certified_aggregate()["exact_noop"])]
    noops = [p for p in ordered if bool(root_pairs[p.program_id].certified_aggregate()["exact_noop"])]
    selected = informative[:budget]
    remaining = budget - len(selected)
    noop_cap = min(remaining, int(math.floor(budget * frac)))
    if noop_cap > 0:
        selected.extend(noops[:noop_cap])
    return selected



class SearchEngine:
    def __init__(
        self,
        *,
        route_ids: Sequence[int],
        emitters: Sequence[SearchEmitter] = DEFAULT_EMITTERS,
        scheduler: FixedEmitterScheduler | None = None,
        stagnation: StagnationMonitor | None = None,
    ):
        self.route_ids = tuple(int(x) for x in route_ids)
        self.emitters = {x.name: x for x in emitters}
        self.scheduler = scheduler or FixedEmitterScheduler()
        self.stagnation = stagnation or StagnationMonitor()
        self.last_proposal_stats: dict[str, object] = {}
        self._contrastive_condition_cache: dict[str, tuple[tuple[ConditionGene, ...], ...]] = {}
        missing = set(_BASE_WEIGHTS) - set(self.emitters)
        if missing:
            raise ValueError(f"missing required search emitters: {sorted(missing)}")

    @classmethod
    def from_archive(cls, archive: SearchArchive, *, route_ids: Sequence[int] | None = None, emitters: Sequence[SearchEmitter] = DEFAULT_EMITTERS, scheduler: FixedEmitterScheduler | None = None) -> "SearchEngine":
        routes = tuple(int(x) for x in (archive.route_ids if route_ids is None else route_ids))
        if routes != tuple(archive.route_ids):
            raise ValueError("resume route identity differs from archive")
        monitor = StagnationMonitor.from_dict(archive.runtime_state.stagnation)
        out = cls(route_ids=routes, emitters=emitters, scheduler=scheduler, stagnation=monitor)
        out.last_proposal_stats = dict(archive.runtime_state.last_proposal_stats)
        out._contrastive_condition_cache = {
            str(key): _deserialize_condition_sets(value)
            for key, value in archive.runtime_state.contrastive_goal_cache.items()
        }
        return out

    def persist_runtime(self, archive: SearchArchive, *, generation: int | None = None, evaluated: bool = False) -> None:
        archive.runtime_state.stagnation = self.stagnation.to_dict()
        archive.runtime_state.last_proposal_stats = dict(self.last_proposal_stats)
        archive.runtime_state.contrastive_goal_cache = {
            str(key): _serialize_condition_sets(value)
            for key, value in sorted(self._contrastive_condition_cache.items())
        }
        if generation is not None:
            g = int(generation)
            if evaluated:
                archive.runtime_state.last_evaluated_generation = max(archive.runtime_state.last_evaluated_generation, g)
            else:
                archive.runtime_state.last_proposed_generation = max(archive.runtime_state.last_proposed_generation, g)
            snapshot = {"generation": g, **dict(self.last_proposal_stats)}
            hist = archive.runtime_state.proposal_history
            if not hist or hist[-1] != snapshot:
                hist.append(snapshot)
                del hist[:-64]

    @staticmethod
    def _state_depth(archive: SearchArchive, semantic_id: str) -> int:
        depths = [p.generation_depth for p in archive.programs.values() if p.child_semantic_id == semantic_id]
        return max(depths, default=0)

    @staticmethod
    def _lineage_program(archive: SearchArchive, semantic_id: str) -> str | None:
        rows = [p for p in archive.programs.values() if p.child_semantic_id == semantic_id]
        if not rows: return None
        return max(rows, key=lambda p: (p.generation_depth, p.program_id)).program_id

    def _emitter_for_level(self, name: str, level: int) -> SearchEmitter:
        if name == "evolvability_mcts" and level >= 5:
            return EvolvabilityMCTSEmitter(max_depth=min(6, 3 + (level - 4)))
        return self.emitters[name]

    def parent_schedule(self, archive: SearchArchive, *, total: int, seed: int) -> list[tuple[str, str]]:
        """Deterministic constrained parent mixture: feasible first, then repair, bounded research."""
        total = max(0, int(total))
        if total == 0:
            return []
        feasible = archive.production_parent_state_ids() or [archive.root_semantic_id]
        repair = archive.repair_parent_state_ids(limit=max(4, total))
        research = sorted({
            str(row.get("child_semantic_id"))
            for row in archive.exploration_niches.values()
            if str(row.get("child_semantic_id", "")) in archive.states
        } | {
            str(row.get("child_semantic_id"))
            for row in archive.novelty_archive
            if str(row.get("child_semantic_id", "")) in archive.states
        })

        if repair:
            repair_n = max(1, int(round(total * 0.25)))
        else:
            repair_n = 0
        if research:
            research_n = min(max(1, int(round(total * 0.10))), max(0, total - repair_n - 1))
        else:
            research_n = 0
        feasible_n = total - repair_n - research_n
        if feasible_n <= 0:
            feasible_n = 1
            if research_n > 0: research_n -= 1
            elif repair_n > 0: repair_n -= 1

        def lane_rows(name: str, ids: Sequence[str], count: int, salt: int) -> list[tuple[str, str]]:
            if count <= 0 or not ids:
                return []
            ordered = list(sorted(ids))
            rng = random.Random(int(seed) ^ int(salt)); rng.shuffle(ordered)
            return [(name, ordered[i % len(ordered)]) for i in range(count)]

        out = (
            lane_rows("feasible", feasible, feasible_n, 0xF3A51B1E)
            + lane_rows("repair", repair, repair_n, 0xA11CE)
            + lane_rows("exploration", research, research_n, 0xE71A)
        )
        random.Random(int(seed) ^ 0x51CED).shuffle(out)
        return out

    def propose_generation(
        self,
        archive: SearchArchive,
        *,
        observations: Sequence[Mapping[str, object]],
        total: int,
        generation: int,
        seed: int,
        parent_evaluations: Mapping[str, StateEvaluation] | None = None,
        search_goals: Sequence[SearchGoal] = (),
        case_contexts: Mapping[str, Sequence[Mapping[str, object]]] | None = None,
        case_bank: CaseBank | None = None,
    ) -> list[EditProgram]:
        if tuple(archive.route_ids) != self.route_ids:
            raise ValueError("search engine/archive route identity mismatch")
        requested = max(0, int(total))
        if requested == 0:
            self.last_proposal_stats = {
                "requested": 0, "raw_emitted": 0, "unique": 0, "duplicate_rejected": 0,
                "duplicate_rate": 0.0, "underfill_rate": 0.0, "source_parent_count": 0,
                "operation_count": 0, "emitter_raw": {}, "emitter_unique": {},
            }
            return []
        production_parent_ids = archive.production_parent_state_ids() or [archive.root_semantic_id]
        repair_parent_ids = archive.repair_parent_state_ids(limit=max(4, len(production_parent_ids)))
        allocations = self.scheduler.allocations(requested, self.stagnation.level, emitter_stats=archive.emitter_stats)
        proposed: list[EditProgram] = []
        seen_children = set(archive.states)
        seen_pairs: set[tuple[str, str]] = set()
        parent_evaluations = dict(parent_evaluations or {})
        quality_by_state = archive_quality_map(archive, exact_state_quality_map(parent_evaluations))
        quality_panel_id = next(iter(parent_evaluations.values())).panel_id if parent_evaluations else None
        goals = tuple(search_goals or ())
        if not goals and case_bank is not None:
            goals = build_search_goals(case_bank, counterexample_ids=tuple(archive.hard_cases))
        contexts_by_case = {str(k): tuple(dict(r) for r in v) for k, v in dict(case_contexts or {}).items()}
        goal_condition_cache: dict[str, tuple[tuple[object, ...], ...]] = {}
        bank_identity = case_bank.identity() if case_bank is not None else ""
        donors_all = {sid: archive.genome(sid) for sid in production_parent_ids}
        donor_quality = archive_quality_map(archive, {sid: 0.0 for sid in production_parent_ids})
        base_context = EmitterContext.from_observations(
            self.route_ids,
            observations,
            donor_genomes=tuple(donors_all.values()),
            donor_quality_by_state=donor_quality,
            motifs=tuple(archive.motifs.values()),
            generation_depth=1,
            escalation_level=self.stagnation.level,
            parent_program_id=None,
            run_id=archive.run_id,
            quality_by_state=quality_by_state,
            search_goal=None,
            case_contexts=contexts_by_case,
            contrastive_condition_sets=None,
        )
        raw_emitted = 0
        duplicate_rejected = 0
        emitter_raw: dict[str, int] = {}
        emitter_unique: dict[str, int] = {}

        def accept_rows(name: str, parent: ModuleGenome, rows: Sequence[EditProgram]) -> None:
            nonlocal raw_emitted, duplicate_rejected
            emitter_raw[name] = emitter_raw.get(name, 0) + len(rows)
            raw_emitted += len(rows)
            for program in rows:
                if len(proposed) >= requested:
                    break
                pair = (program.source_parent_semantic_id, program.child_semantic_id)
                if program.child_semantic_id in seen_children or pair in seen_pairs:
                    duplicate_rejected += 1
                    continue
                child = program.reconstruct(parent, self.route_ids)
                archive.add_program(program, child)
                seen_children.add(program.child_semantic_id)
                seen_pairs.add(pair)
                proposed.append(program)
                emitter_unique[name] = emitter_unique.get(name, 0) + 1

        repair_capable = {"contrastive", "local_refine", "structural", "lns", "evolvability_mcts"}
        context_cache: dict[tuple[str, str], EmitterContext] = {}

        def parents_for(name: str) -> list[str]:
            if name not in repair_capable or not repair_parent_ids:
                return list(production_parent_ids)
            repair_cap = min(len(repair_parent_ids), max(1, len(production_parent_ids) // 2))
            return list(production_parent_ids) + list(repair_parent_ids[:repair_cap])

        def goal_for(name: str, sid: str, ordinal: int = 0) -> SearchGoal | None:
            if name != "contrastive" or not goals:
                return None
            return goals[(int(seed) + int(sid[:8], 16) + int(ordinal)) % len(goals)]

        def context_for(sid: str, goal: SearchGoal | None = None, *, typed_pool_size: int = 0) -> EmitterContext:
            goal_key = "" if goal is None else goal.goal_id
            cache_key = (sid, goal_key)
            cached = context_cache.get(cache_key)
            requested_pool = min(64, max(48, int(typed_pool_size))) if goal is None else 0
            if cached is not None:
                if goal is not None or len(cached.typed_pool) >= requested_pool:
                    return cached
                parent = archive.genome(sid)
                built = cached.with_typed_pool(
                    parent, n=requested_pool,
                    seed=int(seed) + int(generation) * 65537 + int(sid[:8], 16),
                )
                context_cache[cache_key] = built
                return built
            parent = archive.genome(sid)
            donors = tuple(g for dsid, g in donors_all.items() if dsid != sid)
            depth = self._state_depth(archive, sid) + 1
            precomputed_conditions = None
            if goal is not None:
                persistent_key = contrastive_cache_key(bank_identity, goal_key) if bank_identity else None
                if persistent_key is not None and persistent_key in self._contrastive_condition_cache:
                    goal_condition_cache[goal_key] = self._contrastive_condition_cache[persistent_key]
                if goal_key not in goal_condition_cache:
                    semantic = synthesize_contrastive_conditions(
                        goal, contexts_by_case, max_conditions=2, limit=16, feature_budget=32,
                        guard_excluding_only=True, allow_temporal=False,
                    )
                    # Temporal feature synthesis is only a fallback when semantic public
                    # state cannot separate the failure. The emitter itself can add broad/
                    # narrow/exact temporal localizers around a semantic separator.
                    merged = list(semantic)
                    if not merged:
                        merged = list(synthesize_contrastive_conditions(
                            goal, contexts_by_case, max_conditions=2, limit=16, feature_budget=32,
                            guard_excluding_only=True, allow_temporal=True,
                        ))
                    goal_condition_cache[goal_key] = tuple(tuple(conds) for conds in merged[:16])
                    if persistent_key is not None:
                        self._contrastive_condition_cache[persistent_key] = goal_condition_cache[goal_key]
                        archive.runtime_state.contrastive_goal_cache[persistent_key] = _serialize_condition_sets(goal_condition_cache[goal_key])
                precomputed_conditions = goal_condition_cache[goal_key]
            built = replace(
                base_context,
                donor_genomes=donors,
                generation_depth=depth,
                parent_program_id=self._lineage_program(archive, sid),
                search_goal=goal,
                contrastive_condition_sets=precomputed_conditions,
                typed_pool_parent_semantic_id=None,
                typed_pool=(),
            )
            if goal is None:
                built = built.with_typed_pool(
                    parent, n=requested_pool,
                    seed=int(seed) + int(generation) * 65537 + int(sid[:8], 16),
                )
            context_cache[cache_key] = built
            return built

        for emitter_index, (name, budget) in enumerate(allocations.items()):
            if budget <= 0 or len(proposed) >= requested:
                continue
            emitter = self._emitter_for_level(name, self.stagnation.level)
            parent_ids = parents_for(name)
            assignments: dict[str, int] = {sid: 0 for sid in parent_ids}
            for j in range(budget):
                if parent_evaluations and j % max(2, len(parent_ids)) == 0:
                    eligible = {sid: ev for sid, ev in parent_evaluations.items() if sid in assignments}
                    sid = lexicase_parent(eligible, seed=seed + emitter_index * 100003 + j) if eligible else parent_ids[j % len(parent_ids)]
                else:
                    sid = parent_ids[(j + emitter_index) % len(parent_ids)]
                assignments[sid] += 1
            for pi, sid in enumerate(parent_ids):
                count = assignments[sid]
                if count <= 0 or len(proposed) >= requested:
                    continue
                parent = archive.genome(sid)
                base_seed = int(seed) + int(generation) * 1000003 + emitter_index * 10007 + pi * 101
                if name == "contrastive" and goals and count > 1:
                    # A single feasible parent must still cover several independent
                    # failure-repair goals in one generation. Otherwise one seed picks
                    # one goal and the whole contrastive budget becomes seed-sensitive.
                    span = min(int(count), len(goals))
                    base = int(count) // span; extra = int(count) % span
                    for gi in range(span):
                        share = base + int(gi < extra)
                        if share <= 0 or len(proposed) >= requested:
                            continue
                        rows = emitter.propose(
                            parent, n=share,
                            seed=base_seed + gi * 4099,
                            context=context_for(sid, goal_for(name, sid, pi + gi), typed_pool_size=max(16, share * 8)),
                        )
                        accept_rows(name, parent, rows)
                else:
                    rows = emitter.propose(
                        parent, n=count,
                        seed=base_seed,
                        context=context_for(sid, goal_for(name, sid, pi), typed_pool_size=max(16, count * 8)),
                    )
                    accept_rows(name, parent, rows)

        # Some emitters are conditionally inapplicable (motifs before any motif exists,
        # recombination before multiple parents exist). Reallocate their unused budget
        # rather than silently sacrificing CUDA throughput. The fallback remains typed,
        # deterministic and exploration-preserving.
        # Contrastive specialists have an explicit first-class budget and are intentionally
        # excluded from generic refill. Otherwise unavailable motif/recombination capacity
        # can collapse the generation into one aggressive specialist family.
        fallback = self.scheduler.refill_order(archive.emitter_stats)
        for refill_pass in range(4):
            if len(proposed) >= requested:
                break
            remaining = requested - len(proposed)
            progress_before = len(proposed)
            for fi, name in enumerate(fallback):
                if len(proposed) >= requested:
                    break
                emitter = self._emitter_for_level(name, self.stagnation.level)
                parent_ids = parents_for(name)
                per_parent = max(1, math.ceil(remaining / max(1, len(parent_ids))))
                for pi, sid in enumerate(parent_ids):
                    if len(proposed) >= requested:
                        break
                    parent = archive.genome(sid)
                    count = min(per_parent, requested - len(proposed))
                    rows = emitter.propose(
                        parent, n=max(count, count * 2),
                        seed=int(seed) + int(generation) * 3000017 + refill_pass * 65537 + fi * 4099 + pi * 131,
                        context=context_for(sid, goal_for(name, sid, pi + refill_pass), typed_pool_size=max(16, count * 8)),
                    )
                    accept_rows(name, parent, rows)
            if len(proposed) == progress_before:
                break

        unique = len(proposed)
        duplicate_rate = duplicate_rejected / max(1, raw_emitted)
        self.last_proposal_stats = {
            "requested": requested,
            "raw_emitted": raw_emitted,
            "unique": unique,
            "duplicate_rejected": duplicate_rejected,
            "duplicate_rate": duplicate_rate,
            "underfill_rate": max(0.0, (requested - unique) / requested),
            "source_parent_count": len({p.source_parent_semantic_id for p in proposed}),
            "operation_count": len({a.operation for p in proposed for a in p.atoms}),
            "emitter_raw": dict(sorted(emitter_raw.items())),
            "emitter_unique": dict(sorted(emitter_unique.items())),
            "quality_state_count": len(quality_by_state),
            "quality_panel_id": quality_panel_id,
        }
        self.persist_runtime(archive, generation=generation)
        return proposed


    @staticmethod
    def _stage2_order(programs: Sequence[EditProgram], root_pairs: Mapping[str, PairEvaluation]) -> list[EditProgram]:
        """Balanced exploitation/boundary/negative/diversity order for exact Stage-2 truth."""
        buckets: dict[str, list[EditProgram]] = {"exploitation": [], "boundary": [], "negative": [], "diversity": []}
        for program in programs:
            pair = root_pairs[program.program_id]
            a = pair.certified_aggregate()
            useful = bool(a["strict_safe"]) and bool(a["behavior_changing"]) and (
                int(a["l2w_count"]) > 0 or float(a["mean_margin_delta"]) > 0.0
            )
            dangerous = int(a["w2l_count"]) > 0 or int(a["severe_regression_count"]) > 0
            if useful:
                bucket = "exploitation"
            elif dangerous or float(a["mean_margin_delta"]) < 0.0:
                bucket = "negative"
            elif bool(a["behavior_changing"]):
                bucket = "boundary"
            else:
                bucket = "diversity"
            buckets[bucket].append(program)
        def key(p: EditProgram):
            a = root_pairs[p.program_id].aggregate
            return (
                int(a["w2l_count"]), int(a["severe_regression_count"]), int(a["losses_after"]),
                -int(a["l2w_count"]), int(a["fragile_wins_after"]), -int(a["worst_candidate_margin"]),
                -float(a["mean_margin_delta"]), p.program_id,
            )
        for name in buckets:
            buckets[name].sort(key=key)
        order: list[EditProgram] = []
        pattern = ("exploitation", "exploitation", "boundary", "negative", "diversity")
        pos = {k: 0 for k in buckets}
        while len(order) < len(programs):
            added = False
            for name in pattern:
                i = pos[name]
                if i < len(buckets[name]):
                    order.append(buckets[name][i]); pos[name] += 1; added = True
            if not added:
                break
        return order

    def evaluate_round(
        self,
        archive: SearchArchive,
        programs: Sequence[EditProgram],
        *,
        evaluator,
        panel_manager: CasePanelManager,
        screen_size: int,
        final_size: int,
        stage2_budget: int,
        generation: int,
        seed: int,
        work_dir,
        audit_all: bool = False,
        certification_evaluator=None,
        certification_bank: CaseBank | None = None,
        case_contexts: Mapping[str, Sequence[Mapping[str, object]]] | None = None,
        cegis_max_programs: int = 4,
        cegis_rounds: int = 3,
        cegis_candidates_per_round: int = 16,
        max_noop_fraction: float = 0.10,
        certified_neighborhood_budget: int = 48,
    ) -> RoundResult:
        """Run one exact multi-fidelity round and feed broader failures back into search state."""
        programs = list(programs)
        if not programs:
            raise ValueError("evaluate_round requires proposed EditPrograms")
        for program in programs:
            if program.program_id not in archive.programs:
                raise ValueError("evaluate_round program is not registered in archive")
        stage1_program_ids = {program.program_id for program in programs}
        archive.bind_case_bank(panel_manager.bank.identity(), tuple(panel_manager.bank.by_id))
        screen_panel = panel_manager.select_screen(screen_size, seed=seed)
        if not screen_panel:
            raise ValueError("screening panel is empty")
        if final_size >= len(panel_manager.bank):
            full_panel = list(panel_manager.bank.rows)
        else:
            full_panel = panel_manager.select_screen(final_size, seed=seed ^ 0xF17A1)
        if len(full_panel) <= len(screen_panel) and set(c.case_id for c in full_panel) == set(c.case_id for c in screen_panel):
            raise ValueError("final panel must be broader than the screening panel")
        screen_panel_id = f"{archive.run_id}/g{int(generation)}/screen"
        full_panel_id = f"{archive.run_id}/g{int(generation)}/full"
        archive.register_panel(screen_panel_id, tuple(c.case_id for c in screen_panel))
        archive.register_panel(full_panel_id, tuple(c.case_id for c in full_panel))

        screen_sids = {archive.root_semantic_id}
        for program in programs:
            screen_sids.add(program.source_parent_semantic_id); screen_sids.add(program.child_semantic_id)
        screen_eval = evaluator.evaluate(
            [archive.genome(sid) for sid in sorted(screen_sids)], screen_panel,
            work_dir=work_dir, panel_id=screen_panel_id,
        )
        root_screen = screen_eval[archive.root_semantic_id]
        protected_screen = {c.case_id: c.protected for c in screen_panel}
        screen_source_pairs: dict[str, PairEvaluation] = {}
        screen_root_pairs: dict[str, PairEvaluation] = {}
        for program in programs:
            child = screen_eval[program.child_semantic_id]
            screen_source_pairs[program.program_id] = PairEvaluation.from_states(
                screen_eval[program.source_parent_semantic_id], child, protected=protected_screen,
            )
            screen_root_pairs[program.program_id] = PairEvaluation.from_states(root_screen, child, protected=protected_screen)
        finalists = select_stage2_finalists(
            programs, screen_root_pairs,
            budget=min(max(1, int(stage2_budget)), len(programs)),
            max_noop_fraction=max_noop_fraction,
        )
        if not finalists:
            raise RuntimeError("Stage-2 selection found no informative candidate within no-op cap")
        full_programs = programs if audit_all else finalists

        full_sids = {archive.root_semantic_id}
        for program in full_programs:
            full_sids.add(program.source_parent_semantic_id); full_sids.add(program.child_semantic_id)
        full_eval = evaluator.evaluate(
            [archive.genome(sid) for sid in sorted(full_sids)], full_panel,
            work_dir=work_dir, panel_id=full_panel_id,
        )
        root_full = full_eval[archive.root_semantic_id]
        protected_full = {c.case_id: c.protected for c in full_panel}
        for evaluation in full_eval.values():
            archive.record_state_evaluation(evaluation)
        screen_ids = {c.case_id for c in screen_panel}
        full_source_pairs: dict[str, PairEvaluation] = {}
        full_root_pairs: dict[str, PairEvaluation] = {}
        counterexamples: set[str] = set()
        production_insertions = repair_insertions = useful_safe = target_improvements = 0
        novelty_values: list[float] = []
        assessments: dict[str, object] = {}
        safety_spec = SafetySpec(severe_delta=panel_manager.severe_delta, fragile_margin=panel_manager.fragile_margin)
        finalist_ids = {p.program_id for p in finalists}
        considered_ids = {p.program_id for p in full_programs} if audit_all else finalist_ids
        for program in full_programs:
            child = full_eval[program.child_semantic_id]
            source_pair = PairEvaluation.from_states(full_eval[program.source_parent_semantic_id], child, protected=protected_full)
            root_pair = PairEvaluation.from_states(root_full, child, protected=protected_full)
            full_source_pairs[program.program_id] = source_pair
            full_root_pairs[program.program_id] = root_pair
            if program.program_id in considered_ids:
                assessment = safety_spec.assess_program(source_pair=source_pair, root_pair=root_pair)
                assessments[program.program_id] = assessment
                considered = archive.consider(program, root_pair, source_pair=source_pair)
                production_insertions += int(considered.production_inserted)
                repair_insertions += int(considered.repair_inserted)
                is_useful = bool(assessment.root.safe_useful)
                useful_safe += int(is_useful)
                target_improvements += int(assessment.root.l2w > 0)
                novelty_values.append(len(root_pair.changed_case_ids) / max(1, len(root_pair.case_rows)))
                archive.note_emitter(
                    program.emitter, attempts=1, unique=1,
                    archive_insertions=int(considered.production_inserted or considered.repair_inserted),
                    production_insertions=int(considered.production_inserted),
                    repair_insertions=int(considered.repair_inserted),
                    useful_safe=int(is_useful),
                    unsafe=int(not assessment.root.feasible),
                    new_loss=int(assessment.root.violation.new_losses > 0),
                )
                # Production runs defer motif reuse until independent certification. Diagnostic/
                # legacy rounds without a certification bank may still retain discovery-only motifs
                # as exploratory evidence; certified motif counters dominate ranking when available.
                if assessment.high_quality_supervision and (certification_evaluator is None or certification_bank is None):
                    archive.observe_motif(program, root_pair, certification=None)
            if bool(screen_root_pairs[program.program_id].certified_aggregate()["strict_safe"]):
                for row in root_pair.case_rows:
                    cid = str(row["case_id"])
                    new_loss = bool(row.get("new_loss", int(row["baseline_margin"]) >= 0 and int(row["candidate_margin"]) < 0))
                    dangerous = new_loss or bool(row["w2l"]) or (int(row["baseline_margin"]) > 0 and int(row["margin_delta"]) < -3000)
                    if dangerous and cid not in screen_ids:
                        severity = max(1, -int(row["candidate_margin"]) if new_loss or bool(row["w2l"]) else -int(row["margin_delta"]))
                        reason = "new_loss" if new_loss and not bool(row["w2l"]) else ("w2l" if bool(row["w2l"]) else "severe_regression")
                        panel_manager.add_hard_case(cid, severity=severity, reason=reason)
                        archive.record_counterexample(cid, reason=reason, severity=severity, panel_id=root_pair.panel_id)
                        counterexamples.add(cid)

        certification_vectors: dict[str, CertificationVector] = {}
        if certification_evaluator is not None and certification_bank is not None:
            candidates_for_cert = []
            for program in full_programs:
                assessment = assessments.get(program.program_id)
                if assessment is None:
                    continue
                root_assessment = assessment.root
                if root_assessment.feasible and root_assessment.behavior_changing and (
                    root_assessment.safe_useful or root_assessment.elite
                ):
                    candidates_for_cert.append(program)
            if candidates_for_cert:
                batch = certify_programs_exact(
                    archive, candidates_for_cert, full_root_pairs,
                    evaluator=certification_evaluator, case_bank=certification_bank,
                    generation=generation, work_dir=Path(work_dir) / "certification", spec=safety_spec,
                )
                certification_vectors.update(batch.vectors)
            for program in candidates_for_cert:
                archive.observe_motif(
                    program, full_root_pairs[program.program_id],
                    certification=certification_vectors.get(program.program_id),
                )

        repair_reports: list[dict[str, object]] = []
        if case_contexts and int(cegis_max_programs) > 0:
            repairable: list[tuple[tuple[object, ...], EditProgram]] = []
            for program in full_programs:
                assessment = assessments.get(program.program_id)
                if assessment is None or assessment.root.feasible or not assessment.root.behavior_changing:
                    continue
                if int(assessment.root.l2w) <= 0 and float(assessment.root.mean_margin_delta) <= 0.0:
                    continue
                repairable.append((
                    (assessment.root.violation.key(), -int(assessment.root.l2w), -float(assessment.root.mean_margin_delta), program.program_id),
                    program,
                ))
            repairable.sort(key=lambda row: row[0])
            for ri, (_, program) in enumerate(repairable[:max(0, int(cegis_max_programs))]):
                report = run_cegis_repair(
                    archive, program, evaluator=evaluator, panel_manager=panel_manager,
                    case_contexts=case_contexts, generation=generation,
                    seed=int(seed) + 900001 + ri * 65537,
                    work_dir=Path(work_dir) / f"cegis_{ri:02d}",
                    max_rounds=cegis_rounds, candidates_per_round=cegis_candidates_per_round,
                    # The searchable CaseBank may already contain the rotating
                    # certification-repair mini-panel. Never expose the held-out
                    # certification bank to CEGIS; certify the repaired terminal only
                    # after it is feasible/useful on searchable truth.
                    certification_evaluator=None,
                    certification_bank=None,
                )
                repair_reports.append(dict(report.__dict__))
                if report.terminal_program_id and report.became_feasible > 0:
                    terminal_program = archive.programs[report.terminal_program_id]
                    repair_panel_id = f"{archive.run_id}/g{int(generation)}/repair-final/{ri}"
                    archive.register_panel(repair_panel_id, tuple(c.case_id for c in full_panel))
                    repair_exact = evaluator.evaluate(
                        [archive.genome(archive.root_semantic_id), archive.genome(terminal_program.source_parent_semantic_id), archive.genome(terminal_program.child_semantic_id)],
                        full_panel, work_dir=Path(work_dir) / f"cegis_{ri:02d}" / "export_pair", panel_id=repair_panel_id,
                    )
                    child_ev = repair_exact[terminal_program.child_semantic_id]
                    root_pair = PairEvaluation.from_states(repair_exact[archive.root_semantic_id], child_ev, protected=protected_full)
                    source_pair = PairEvaluation.from_states(repair_exact[terminal_program.source_parent_semantic_id], child_ev, protected=protected_full)
                    full_root_pairs[terminal_program.program_id] = root_pair
                    full_source_pairs[terminal_program.program_id] = source_pair
                    terminal_vector = None
                    terminal_assessment = safety_spec.assess(root_pair)
                    if (
                        certification_evaluator is not None and certification_bank is not None
                        and terminal_assessment.feasible and terminal_assessment.safe_useful
                    ):
                        terminal_batch = certify_programs_exact(
                            archive, (terminal_program,), {terminal_program.program_id: root_pair},
                            evaluator=certification_evaluator, case_bank=certification_bank,
                            generation=generation,
                            work_dir=Path(work_dir) / f"cegis_{ri:02d}" / "heldout_certification",
                            panel_parts=1, spec=safety_spec,
                        )
                        terminal_vector = terminal_batch.vectors[terminal_program.program_id]
                        certification_vectors[terminal_program.program_id] = terminal_vector
                    archive.observe_motif(terminal_program, root_pair, certification=terminal_vector)

        # High-information neighborhoods around independently certified positives.
        # Certification quality chooses source states, but candidate construction sees
        # only their public tree/edit structure; every neighbor must re-earn search
        # feasibility and held-out certification through exact evaluation.
        if certification_evaluator is not None and certification_bank is not None and int(certified_neighborhood_budget) > 0:
            source_rows: list[tuple[tuple[object, ...], EditProgram]] = []
            for pid, raw in archive.certification_evidence.items():
                program = archive.programs.get(str(pid))
                if program is None:
                    continue
                vector = archive.certification_vector(str(pid))
                if vector is None or not vector.certified_positive:
                    continue
                source_rows.append(((
                    certification_rank(vector.certification_class),
                    int(bool(vector.temporal)),
                    -float(vector.certification_mean_margin_delta),
                    str(pid),
                ), program))
            source_rows.sort(key=lambda row: row[0])
            chosen_sources = [program for _, program in source_rows[:2]]
            if chosen_sources:
                per_source = max(1, int(math.ceil(int(certified_neighborhood_budget) / len(chosen_sources))))
                for ni, source_program in enumerate(chosen_sources):
                    batch = evaluate_certified_neighborhood_exact(
                        archive, source_program, evaluator=evaluator,
                        search_bank=panel_manager.bank, certification_bank=certification_bank,
                        generation=generation, seed=int(seed) + 1700003 + ni * 65537,
                        work_dir=Path(work_dir) / f"certified_neighborhood_{ni:02d}",
                        budget=per_source, spec=safety_spec,
                    )
                    full_root_pairs.update(batch.root_pairs)
                    full_source_pairs.update(batch.source_pairs)
                    certification_vectors.update(batch.vectors)



        stage_recall = None
        if audit_all:
            useful_truth = {
                pid for pid, pair in full_root_pairs.items()
                if pid in stage1_program_ids
                and bool(pair.certified_aggregate()["strict_safe"]) and bool(pair.certified_aggregate()["behavior_changing"])
                and (int(pair.certified_aggregate()["l2w_count"]) > 0 or float(pair.certified_aggregate()["mean_margin_delta"]) > 0)
            }
            l2w_truth = {
                pid for pid, pair in full_root_pairs.items()
                if pid in stage1_program_ids and int(pair.certified_aggregate()["l2w_count"]) > 0
            }
            danger_truth = {
                pid for pid, pair in full_root_pairs.items()
                if pid in stage1_program_ids
                and (int(pair.certified_aggregate()["new_loss_count"]) > 0 or int(pair.certified_aggregate()["w2l_count"]) > 0 or int(pair.certified_aggregate()["severe_regression_count"]) > 0)
            }
            screen_useful = {
                pid for pid, pair in screen_root_pairs.items()
                if bool(pair.certified_aggregate()["strict_safe"]) and bool(pair.certified_aggregate()["behavior_changing"])
                and (int(pair.certified_aggregate()["l2w_count"]) > 0 or float(pair.certified_aggregate()["mean_margin_delta"]) > 0)
            }
            screen_l2w = {pid for pid, pair in screen_root_pairs.items() if int(pair.certified_aggregate()["l2w_count"]) > 0}
            screen_danger = {
                pid for pid, pair in screen_root_pairs.items()
                if int(pair.certified_aggregate()["new_loss_count"]) > 0 or int(pair.certified_aggregate()["w2l_count"]) > 0
                or int(pair.certified_aggregate()["severe_regression_count"]) > 0
            }
            stage_recall = panel_manager.stage_detection_recall(
                screen_useful_ids=screen_useful,
                screen_l2w_ids=screen_l2w,
                screen_dangerous_ids=screen_danger,
                full_useful_ids=useful_truth,
                full_l2w_ids=l2w_truth,
                full_dangerous_ids=danger_truth,
                finalist_ids=finalist_ids,
            )
        proposal_stats = self.last_proposal_stats if int(self.last_proposal_stats.get("unique", -1)) == len(programs) else {}
        strategic_insertions = production_insertions + repair_insertions
        discovery_elite = sum(1 for a in assessments.values() if bool(a.root.elite))
        certified_safe_useful = sum(1 for v in certification_vectors.values() if bool(v.certified_positive))
        certified_elite = sum(1 for v in certification_vectors.values() if bool(v.certified_elite))
        repair_to_feasible = sum(int(r.get("became_feasible", 0)) > 0 for r in repair_reports)
        generalization_improved = certified_safe_useful
        signals = GenerationSignals(
            proposed=int(proposal_stats.get("requested", len(programs))),
            unique=len({p.child_semantic_id for p in programs}),
            archive_insertions=strategic_insertions, useful_safe=useful_safe, target_improvements=target_improvements,
            novelty_mean=sum(novelty_values) / max(1, len(novelty_values)), new_niches=strategic_insertions,
            production_insertions=production_insertions, repair_insertions=repair_insertions,
            duplicate_rate=float(proposal_stats.get("duplicate_rate", 0.0)),
            underfill_rate=float(proposal_stats.get("underfill_rate", 0.0)),
            source_parent_count=int(proposal_stats.get("source_parent_count", len({p.source_parent_semantic_id for p in programs}))),
            operation_count=int(proposal_stats.get("operation_count", len({a.operation for p in programs for a in p.atoms}))),
            discovery_elite=discovery_elite, certified_safe_useful=certified_safe_useful,
            certified_elite=certified_elite, repair_to_feasible=repair_to_feasible,
            generalization_improved=generalization_improved,
        )
        event = self.stagnation.observe(signals)
        if event.escalated:
            archive.escalation_history.append({
                "generation": int(generation), "level": event.level, "reason": event.reason,
                "signals": signals.__dict__,
            })
        self.persist_runtime(archive, generation=generation, evaluated=True)
        return RoundResult(
            generation=int(generation), screened_program_ids=tuple(p.program_id for p in programs),
            finalist_program_ids=tuple(p.program_id for p in finalists), screen_source_pairs=screen_source_pairs,
            full_source_pairs=full_source_pairs, full_root_pairs=full_root_pairs,
            counterexamples=tuple(sorted(counterexamples)), stage_recall=stage_recall,
            signals=signals, escalation=event,
            certification_vectors=certification_vectors, repair_reports=tuple(repair_reports),
        )
