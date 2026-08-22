from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from tools.bkg4_search.bonsai_editformer.actor_grammar import EditAction
from tools.bkg4_search.module_codec import ModuleGenome, genome_from_json
from tools.edit_corpus.pipeline import build_edit_plan

from .archive import SearchArchive, structural_descriptor
from .emitters import EmitterContext, RandomTypedEmitter, SystematicEmitter
from .fitness import LexicographicFitness, PairEvaluation, StateEvaluation
from .goals import build_search_goals, decode_case_contexts
from .panels import CaseBank, CasePanelManager, CaseRecord
from .schema import EditProgram
from .search import SearchEngine, StagnationMonitor


def load_evaluator_cases_tsv(path: str | Path) -> CaseBank:
    """Load the existing seven-field B4EC1 case format for benchmark-only use."""
    rows: list[CaseRecord] = []
    for i, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines()):
        if not line.strip() or line.startswith("#"):
            continue
        fields = line.split("\t")
        if len(fields) != 7:
            raise ValueError("benchmark B4EC1 case line must have seven tab-separated fields")
        tape, seat, seed, family, protected, own_history, role = fields
        rows.append(CaseRecord(
            case_id=f"bench-case-{i:06d}", tape_path=tape, seat=int(seat), seed=int(seed), family=str(family),
            protected=bool(int(protected)), own_history=bool(int(own_history)), role=str(role), baseline_margin=0,
        ))
    return CaseBank(rows)


def _old_shallow_programs(
    root_genome: ModuleGenome, observations: Sequence[Mapping[str, Any]], route_ids: Sequence[int], *, budget: int, seed: int,
) -> list[tuple[EditProgram, ModuleGenome]]:
    # This intentionally invokes the historical hard-coded-prefix, seed-parent + one-layer plan.
    plan = build_edit_plan(
        "benchmark-old-shallow", observations, seed=int(seed), seed_parent_count=4,
        candidates_per_parent=max(32, int(budget)), route_ids=route_ids, root_overlay=root_genome,
    )
    parents = {
        str(row["overlay_semantic_id"]): genome_from_json(json.loads(str(row["overlay_genome_json"])))
        for row in plan["parents"]
    }
    root_like = {sid for sid, genome in parents.items() if not genome.rules}
    if root_genome.semantic_id() not in parents:
        raise RuntimeError("old shallow benchmark arm did not preserve the injected common root")
    root_like = {root_genome.semantic_id()}
    out: list[tuple[EditProgram, ModuleGenome]] = []
    seen: set[str] = set()
    for i, row in enumerate(plan["edits"]):
        parent_sid = str(row["parent_overlay_semantic_id"])
        parent = parents[parent_sid]
        payload = json.loads(str(row["action_json"]))
        for key in ("target_kind", "target_rule_index", "target_condition_index"):
            payload.pop(key, None)
        action = EditAction(**payload)
        depth = 1 if parent_sid in root_like else 2
        try:
            program = EditProgram.build(
                parent, (action,), route_ids, emitter="old_shallow", seed=int(seed) + i,
                generation_depth=depth, run_id="benchmark-old-shallow",
            )
            child = program.reconstruct(parent, route_ids)
        except (TypeError, ValueError):
            continue
        if child.semantic_id() in seen:
            continue
        seen.add(child.semantic_id()); out.append((program, child))
        if len(out) >= int(budget):
            break
    if len(out) < int(budget):
        raise RuntimeError(f"old shallow generator underfilled equal budget: {len(out)} < {budget}")
    return out


def _single_parent_programs(
    root: ModuleGenome,
    observations: Sequence[Mapping[str, Any]],
    route_ids: Sequence[int],
    *,
    budget: int,
    seed: int,
    random_typed: bool,
) -> list[tuple[EditProgram, ModuleGenome]]:
    context = EmitterContext.from_observations(route_ids, observations, generation_depth=1, run_id="benchmark")
    emitter = RandomTypedEmitter() if random_typed else SystematicEmitter()
    rows = emitter.propose(root, n=max(int(budget), int(budget) * 2), seed=int(seed), context=context)
    unique: list[tuple[EditProgram, ModuleGenome]] = []
    seen: set[str] = set()
    for program in rows:
        child = program.reconstruct(root, route_ids)
        if child.semantic_id() in seen:
            continue
        seen.add(child.semantic_id()); unique.append((program, child))
        if len(unique) >= int(budget):
            break
    if len(unique) < int(budget):
        raise RuntimeError(f"single-parent baseline underfilled equal budget: {len(unique)} < {budget}")
    return unique


def _evaluate_children(
    entries: Sequence[tuple[EditProgram, ModuleGenome]],
    *,
    evaluator,
    panel: Sequence[CaseRecord],
    work_dir: Path,
    panel_id: str,
) -> dict[str, StateEvaluation]:
    genomes = [child for _, child in entries]
    return evaluator.evaluate(genomes, panel, work_dir=work_dir, panel_id=panel_id)


def _pairs_against_root(
    entries: Sequence[tuple[EditProgram, ModuleGenome]],
    evaluations: Mapping[str, StateEvaluation],
    root_eval: StateEvaluation,
    panel: Sequence[CaseRecord],
) -> dict[str, PairEvaluation]:
    protected = {c.case_id: c.protected for c in panel}
    return {
        program.program_id: PairEvaluation.from_states(root_eval, evaluations[child.semantic_id()], protected=protected)
        for program, child in entries
    }


def _metric_row(
    entries: Sequence[tuple[EditProgram, ModuleGenome]],
    pairs: Mapping[str, PairEvaluation],
    root_eval: StateEvaluation,
    *,
    candidate_budget: int,
    case_count: int,
    proposal_stats: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    programs = [p for p, _ in entries]
    if len(programs) != int(candidate_budget):
        raise ValueError("benchmark arm did not consume the exact common candidate budget")
    aggregates = [pairs[p.program_id].aggregate for p in programs]
    useful = [
        p for p in programs
        if bool(pairs[p.program_id].aggregate["strict_safe"])
        and bool(pairs[p.program_id].aggregate["behavior_changing"])
        and (
            int(pairs[p.program_id].aggregate["l2w_count"]) > 0
            or float(pairs[p.program_id].aggregate["mean_margin_delta"]) > 0.0
        )
    ]
    strict_l2w = [
        p for p in programs
        if bool(pairs[p.program_id].aggregate["strict_safe"])
        and int(pairs[p.program_id].aggregate["l2w_count"]) > 0
    ]
    target_improvers = []
    target_best = 0
    for p in programs:
        pair = pairs[p.program_id]
        improved = False
        for row in pair.case_rows:
            if int(row["baseline_margin"]) < 0 and int(row["margin_delta"]) > 0:
                improved = True
                target_best = max(target_best, int(row["margin_delta"]))
        if improved:
            target_improvers.append(p)
    top_n = max(1, len(programs) // 4)
    top = sorted(
        programs,
        key=lambda p: LexicographicFitness.from_pair(
            pairs[p.program_id], complexity=p.program_length
        ).key(),
    )[:top_n]
    root_worst = min(root_eval.margins)
    qd_keys = set()
    target_kinds = set()
    parent_rule_counts = set()
    operations = set()
    emitters = set()
    useful_shapes = set()
    for p, child in entries:
        emitters.add(p.emitter)
        parent_rules = int(p.atoms[0].target_rule_index >= 0)
        # Parent rule count can be reconstructed from target metadata only imperfectly;
        # use the source atom ledger: max addressed rule + one, or zero for root-only edits.
        addressed = [a.target_rule_index for a in p.atoms if a.target_rule_index >= 0]
        source_rule_count_lb = max(addressed) + 1 if addressed else 0
        parent_rule_counts.add(source_rule_count_lb)
        desc = structural_descriptor(p, child)
        qd_keys.add(json.dumps(desc, sort_keys=True, separators=(",", ":")))
        for atom in p.atoms:
            operations.add(atom.operation); target_kinds.add(atom.target_kind)
        if p in useful:
            useful_shapes.add((tuple(a.operation for a in p.atoms), tuple(a.target_kind for a in p.atoms), source_rule_count_lb))
    stats = dict(proposal_stats or {})
    attempted = max(len(programs), int(stats.get("raw_emitted", len(programs))))
    duplicates = max(0, int(stats.get("duplicate_rejected", attempted - len(programs))))
    scale = 1000.0 / max(1, len(programs))
    best_worst = max(int(a["worst_candidate_margin"]) - root_worst for a in aggregates)
    return {
        "requested_candidates": int(candidate_budget),
        "evaluated_candidates": len(programs),
        "exact_candidate_case_evaluations": len(programs) * int(case_count),
        "unique_valid_proposals": len({p.child_semantic_id for p in programs}),
        "duplicate_rate": duplicates / max(1, attempted),
        "behavior_changing_rate": sum(bool(a["behavior_changing"]) for a in aggregates) / len(programs),
        "operation_coverage": len(operations),
        "operation_families": sorted(operations),
        "emitter_families": sorted(emitters),
        "target_depth_coverage": len(target_kinds),
        "target_kinds": sorted(target_kinds),
        "parent_rule_count_coverage": len(parent_rule_counts),
        "parent_rule_count_lower_bounds": sorted(parent_rule_counts),
        "source_parent_count": len({p.source_parent_semantic_id for p in programs}),
        "new_loss_rate": sum(int(a.get("new_loss_count", 0)) > 0 for a in aggregates) / len(programs),
        "new_loss_candidates": sum(int(a.get("new_loss_count", 0)) > 0 for a in aggregates),
        "qd_niches": len(qd_keys),
        "safe_useful_per_1000": len(useful) * scale,
        "target_loss_improving_per_1000": len(target_improvers) * scale,
        "l2w_per_1000": sum(int(a["l2w_count"]) > 0 for a in aggregates) * scale,
        "strict_safe_l2w_per_1000": len(strict_l2w) * scale,
        "w2l_rate": sum(int(pairs[p.program_id].aggregate["w2l_count"]) > 0 for p in top) / len(top),
        "severe_regression_rate": sum(int(a["severe_regression_count"]) > 0 for a in aggregates) / len(programs),
        "best_target_improvement": int(target_best),
        "best_worst_margin_improvement": int(best_worst),
        "useful_diversity": len(useful_shapes),
        "max_generation_depth": max(p.generation_depth for p in programs),
    }


def _recursive_entries(
    root_genome: ModuleGenome,
    route_ids: Sequence[int],
    observations: Sequence[Mapping[str, Any]],
    *,
    panel: Sequence[CaseRecord],
    evaluator,
    root_eval: StateEvaluation,
    budget: int,
    seed: int,
    work_dir: Path,
    panel_id: str,
) -> tuple[list[tuple[EditProgram, ModuleGenome]], dict[str, PairEvaluation], dict[str, Any]]:
    archive = SearchArchive.new(
        root_genome=root_genome, base_identity="benchmark-parent", route_ids=route_ids,
        run_id="benchmark-permanent", master_seed=int(seed), source_identity="benchmark",
    )
    engine = SearchEngine(route_ids=route_ids)
    protected = {c.case_id: c.protected for c in panel}
    benchmark_bank = CaseBank(panel)
    search_goals = build_search_goals(benchmark_bank)
    case_contexts = decode_case_contexts(benchmark_bank) if search_goals else {}
    archive.bind_case_bank(benchmark_bank.identity(), tuple(benchmark_bank.by_id))
    archive.register_panel(panel_id, tuple(c.case_id for c in panel))
    first_budget = max(1, int(budget) // 2)
    second_budget = int(budget) - first_budget
    all_entries: list[tuple[EditProgram, ModuleGenome]] = []
    all_pairs: dict[str, PairEvaluation] = {}
    combined_stats = {"raw_emitted": 0, "duplicate_rejected": 0}
    parent_evals: dict[str, StateEvaluation] = {root_genome.semantic_id(): root_eval}
    for generation, count in ((1, first_budget), (2, second_budget)):
        if count <= 0:
            continue
        programs = engine.propose_generation(
            archive, observations=observations, total=count, generation=generation,
            seed=int(seed) + generation * 1009, parent_evaluations=parent_evals,
            search_goals=search_goals, case_contexts=case_contexts, case_bank=benchmark_bank,
        )
        if len(programs) != count:
            raise RuntimeError(f"permanent recursive proposer underfilled equal budget at generation {generation}: {len(programs)} < {count}")
        entries = [(p, archive.genome(p.child_semantic_id)) for p in programs]
        evaluations = _evaluate_children(
            entries, evaluator=evaluator, panel=panel, work_dir=work_dir / f"g{generation}", panel_id=panel_id,
        )
        for p, child in entries:
            pair = PairEvaluation.from_states(root_eval, evaluations[child.semantic_id()], protected=protected)
            all_pairs[p.program_id] = pair
            archive.record_state_evaluation(evaluations[child.semantic_id()])
            archive.consider(p, pair)
        parent_evals = {root_genome.semantic_id(): root_eval, **evaluations}
        all_entries.extend(entries)
        stats = dict(getattr(engine, "last_proposal_stats", {}) or {})
        combined_stats["raw_emitted"] += int(stats.get("raw_emitted", len(programs)))
        combined_stats["duplicate_rejected"] += int(stats.get("duplicate_rejected", 0))
    if len(all_entries) != int(budget):
        raise RuntimeError("permanent recursive benchmark did not consume exact common budget")
    return all_entries, all_pairs, combined_stats


def benchmark_proposers(
    *,
    root_genome: ModuleGenome,
    route_ids: Sequence[int],
    observations: Sequence[Mapping[str, Any]],
    panel: Sequence[CaseRecord],
    evaluator,
    candidate_budget: int,
    seed: int,
    work_dir: str | Path,
) -> dict[str, Any]:
    """Compare four proposal systems with exactly the same child×case truth budget."""
    candidate_budget = int(candidate_budget)
    if candidate_budget < 4:
        raise ValueError("benchmark candidate budget must be at least four")
    panel = tuple(panel)
    if not panel:
        raise ValueError("benchmark panel is empty")
    root = Path(work_dir); root.mkdir(parents=True, exist_ok=True)
    panel_id = f"benchmark/equal-budget/{int(seed)}"
    root_eval = evaluator.evaluate([root_genome], panel, work_dir=root / "shared_root", panel_id=panel_id)[root_genome.semantic_id()]

    arms: dict[str, tuple[list[tuple[EditProgram, ModuleGenome]], dict[str, PairEvaluation], Mapping[str, Any]]] = {}
    arm_roots = {
        "old_shallow": root_genome.semantic_id(),
        "random_typed": root_genome.semantic_id(),
        "fixed_mutation": root_genome.semantic_id(),
        "permanent_recursive": root_genome.semantic_id(),
    }
    if len(set(arm_roots.values())) != 1:
        raise ValueError("equal-budget benchmark arms do not share one exact root semantic identity")
    old = _old_shallow_programs(root_genome, observations, route_ids, budget=candidate_budget, seed=seed + 11)
    old_eval = _evaluate_children(old, evaluator=evaluator, panel=panel, work_dir=root / "old_shallow", panel_id=panel_id)
    arms["old_shallow"] = (old, _pairs_against_root(old, old_eval, root_eval, panel), {})

    random_rows = _single_parent_programs(root_genome, observations, route_ids, budget=candidate_budget, seed=seed + 23, random_typed=True)
    random_eval = _evaluate_children(random_rows, evaluator=evaluator, panel=panel, work_dir=root / "random_typed", panel_id=panel_id)
    arms["random_typed"] = (random_rows, _pairs_against_root(random_rows, random_eval, root_eval, panel), {})

    fixed = _single_parent_programs(root_genome, observations, route_ids, budget=candidate_budget, seed=seed + 37, random_typed=False)
    fixed_eval = _evaluate_children(fixed, evaluator=evaluator, panel=panel, work_dir=root / "fixed_mutation", panel_id=panel_id)
    arms["fixed_mutation"] = (fixed, _pairs_against_root(fixed, fixed_eval, root_eval, panel), {})

    recursive_entries, recursive_pairs, recursive_stats = _recursive_entries(
        root_genome, route_ids, observations, panel=panel, evaluator=evaluator, root_eval=root_eval,
        budget=candidate_budget, seed=seed + 53, work_dir=root / "permanent_recursive", panel_id=panel_id,
    )
    arms["permanent_recursive"] = (recursive_entries, recursive_pairs, recursive_stats)

    metrics = {
        name: _metric_row(entries, pairs, root_eval, candidate_budget=candidate_budget, case_count=len(panel), proposal_stats=stats)
        for name, (entries, pairs, stats) in arms.items()
    }
    old_row = metrics["old_shallow"]; permanent = metrics["permanent_recursive"]
    structural_wins = sum([
        permanent["parent_rule_count_coverage"] > old_row["parent_rule_count_coverage"],
        permanent["max_generation_depth"] > old_row["max_generation_depth"],
        permanent["useful_diversity"] > old_row["useful_diversity"],
        permanent["source_parent_count"] > old_row["source_parent_count"],
    ])
    safety_ok = (
        permanent["new_loss_rate"] <= old_row["new_loss_rate"]
        and permanent["w2l_rate"] <= old_row["w2l_rate"]
        and permanent["severe_regression_rate"] <= old_row["severe_regression_rate"]
    )
    useful_floor = 0.8 * old_row["safe_useful_per_1000"]
    repair_floor = 0.8 * old_row["strict_safe_l2w_per_1000"]
    yield_ok = permanent["safe_useful_per_1000"] >= useful_floor and permanent["strict_safe_l2w_per_1000"] >= repair_floor
    recursive_ok = permanent["max_generation_depth"] > old_row["max_generation_depth"] and permanent["source_parent_count"] > 1
    diversity_ok = permanent["useful_diversity"] > old_row["useful_diversity"] or permanent["parent_rule_count_coverage"] > old_row["parent_rule_count_coverage"]
    material = bool(safety_ok and yield_ok and recursive_ok and diversity_ok)
    return {
        "schema": 1,
        "candidate_budget_per_proposer": candidate_budget,
        "cases": len(panel),
        "exact_candidate_case_budget_per_proposer": candidate_budget * len(panel),
        "root_truth_shared": True,
        "same_root_parent": True,
        "arm_root_semantic_ids": arm_roots,
        "material_gate": {"safety_ok": safety_ok, "yield_ok": yield_ok, "recursive_ok": recursive_ok, "diversity_ok": diversity_ok},
        "proposers": metrics,
        "permanent_vs_old_structural_wins": structural_wins,
        "material_improvement": material,
    }


def run_anti_saturation(
    *,
    root_genome: ModuleGenome,
    route_ids: Sequence[int],
    observations: Sequence[Mapping[str, Any]],
    case_bank: CaseBank,
    evaluator,
    generations: int,
    candidates_per_generation: int,
    screen_size: int,
    final_size: int,
    seed: int,
    work_dir: str | Path,
    stagnation_patience: int = 2,
) -> dict[str, Any]:
    """Bounded persistent-search probe that records collapse signals and escalation responses."""
    if int(generations) <= 0 or int(candidates_per_generation) <= 0:
        raise ValueError("anti-saturation generations/candidates must be positive")
    if int(final_size) <= int(screen_size):
        raise ValueError("anti-saturation full panel must be broader than screen panel")
    archive = SearchArchive.new(
        root_genome=root_genome, base_identity="anti-saturation-parent", route_ids=route_ids,
        run_id="anti-saturation", master_seed=int(seed), source_identity="anti-saturation",
    )
    engine = SearchEngine(
        route_ids=route_ids,
        stagnation=StagnationMonitor(patience=int(stagnation_patience), max_level=6),
    )
    manager = CasePanelManager(case_bank, random_anchor_fraction=0.10)
    root = Path(work_dir); root.mkdir(parents=True, exist_ok=True)
    trajectory: list[dict[str, Any]] = []
    escalation_events: list[dict[str, Any]] = []
    parent_evals: dict[str, StateEvaluation] = {}
    for generation in range(1, int(generations) + 1):
        programs = engine.propose_generation(
            archive, observations=observations, total=int(candidates_per_generation), generation=generation,
            seed=int(seed) + generation * 7919, parent_evaluations=parent_evals,
        )
        if not programs:
            # Explicitly feed total proposal exhaustion to the same production monitor.
            event = engine.stagnation.observe(
                __import__("tools.bkg4_search.edit_search.search", fromlist=["GenerationSignals"]).GenerationSignals(
                    proposed=int(candidates_per_generation), unique=0, archive_insertions=0, useful_safe=0,
                    target_improvements=0, novelty_mean=0.0, new_niches=0, duplicate_rate=1.0, underfill_rate=1.0,
                )
            )
            if event.escalated:
                escalation_events.append({"generation": generation, **event.__dict__})
            trajectory.append({
                "generation": generation, "requested": int(candidates_per_generation), "unique": 0,
                "duplicate_rate": 1.0, "archive_size": len(archive.states), "production_niches": len(archive.production_niches),
                "exploration_niches": len(archive.exploration_niches), "novelty_entries": len(archive.novelty_archive),
                "novelty_mean": 0.0, "emitter_yield": {}, "escalation_level": engine.stagnation.level,
                "source_parent_count": 0, "operation_count": 0,
            })
            continue
        result = engine.evaluate_round(
            archive, programs, evaluator=evaluator, panel_manager=manager,
            screen_size=min(int(screen_size), len(case_bank)), final_size=min(int(final_size), len(case_bank)),
            stage2_budget=len(programs), generation=generation, seed=int(seed) + generation * 104729,
            work_dir=root / f"g{generation:03d}", audit_all=True,
        )
        stats = dict(getattr(engine, "last_proposal_stats", {}) or {})
        parent_evals = {}
        if result.full_root_pairs:
            panel_id = next(iter(result.full_root_pairs.values())).panel_id
            for sid in archive.parent_state_ids():
                try:
                    parent_evals[sid] = archive.state_evaluation(sid, panel_id)
                except KeyError:
                    pass
        if result.escalation.escalated:
            escalation_events.append({"generation": generation, **result.escalation.__dict__})
        trajectory.append({
            "generation": generation,
            "requested": int(stats.get("requested", candidates_per_generation)),
            "unique": len(programs),
            "duplicate_rate": float(stats.get("duplicate_rate", result.signals.duplicate_rate)),
            "archive_size": len(archive.states),
            "production_niches": len(archive.production_niches),
            "exploration_niches": len(archive.exploration_niches),
            "novelty_entries": len(archive.novelty_archive),
            "novelty_mean": float(result.signals.novelty_mean),
            "emitter_yield": dict(stats.get("emitter_unique", {})),
            "escalation_level": int(engine.stagnation.level),
            "source_parent_count": int(stats.get("source_parent_count", len({p.source_parent_semantic_id for p in programs}))),
            "operation_count": int(stats.get("operation_count", len({a.operation for p in programs for a in p.atoms}))),
            "underfill_rate": float(stats.get("underfill_rate", 0.0)),
            "useful_safe": int(result.signals.useful_safe),
            "target_improvements": int(result.signals.target_improvements),
        })
    return {
        "schema": 1,
        "generations": int(generations),
        "candidates_per_generation": int(candidates_per_generation),
        "cases": len(case_bank),
        "trajectory": trajectory,
        "escalation_events": escalation_events,
        "final_escalation_level": int(engine.stagnation.level),
        "final_archive_states": len(archive.states),
        "final_production_niches": len(archive.production_niches),
        "final_exploration_niches": len(archive.exploration_niches),
        "final_novelty_entries": len(archive.novelty_archive),
    }
