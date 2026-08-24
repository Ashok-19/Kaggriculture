from __future__ import annotations

import argparse
import dataclasses
import json
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

from tools.bkg4_search.bonsai_editformer.invariant_data import invariant_corpus_arrays
from tools.bkg4_search.edit_search.schema import ParentProfile
from tools.bkg4_search.invariant_evaluator import InvariantExactCudaEvaluator
from tools.bkg4_search.invariant_factory import (
    OfflineSpecialistBank,
    ProposalBatch,
    SpecialistFactory,
    combine_proposal_batches,
    load_offline_specialist_bank,
    run_proposal_funnel,
    run_specialist_funnel,
    specialist_corpus_records,
    split_specialist_panels,
)
from tools.bkg4_search.invariant_schema import validate_specialist_corpus
from tools.bkg4_search.invariant_search import InvariantSearchBackend, default_search_goals
from tools.bkg4_search.invariant_specialists import (
    SPECIALIST_QD_DIMENSIONS,
    diversity_report,
    strategic_coverage_matrix,
)


def _bank_specs(values: Sequence[str]) -> tuple[tuple[str, Path], ...]:
    rows = []
    for value in values:
        if "=" not in value:
            raise ValueError("specialist bank must use KIND=PATH")
        kind, path = value.split("=", 1)
        if not re.fullmatch(r"[a-z][a-z0-9_]*", kind):
            raise ValueError(f"invalid cohort kind {kind!r}")
        rows.append((kind, Path(path)))
    if not rows:
        raise ValueError("at least one offline specialist bank is required")
    return tuple(rows)


def combine_offline_banks(banks: Sequence[OfflineSpecialistBank]) -> OfflineSpecialistBank:
    cases = tuple(case for bank in banks for case in bank.cases)
    if len({case.case_id for case in cases}) != len(cases):
        raise ValueError("combined specialist bank has duplicate case IDs")
    tape_cohorts: dict[str, str] = {}
    displays: dict[str, str] = {}
    roles = set()
    for bank in banks:
        for tape, cohort in bank.cohort_by_group.items():
            previous = tape_cohorts.setdefault(tape, cohort)
            if previous != cohort:
                raise ValueError(f"tape {tape} belongs to multiple offline cohorts")
        displays.update(bank.cohort_display)
        roles.update(bank.source_roles)
    return OfflineSpecialistBank(cases, tape_cohorts, tuple(sorted(roles)), dict(sorted(displays.items())))


def _write_jsonl(path: Path, rows: Sequence[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _tier_yield(records: Sequence[dict[str, object]]) -> dict[str, int]:
    return dict(sorted(Counter(str(row["target_labels"]["source_quality"]) for row in records).items()))


def _cohort_headroom(records: Sequence[dict[str, object]]) -> dict[str, dict[str, Any]]:
    best: dict[str, dict[str, Any]] = {}
    for record in records:
        labels = record["target_labels"]
        for cohort in labels["cohort_outcomes"]:
            if int(cohort["useful_groups"]) <= 0:
                continue
            row = {
                "semantic_id": record["semantic_id"],
                "source_quality": labels["source_quality"],
                "useful_groups": int(cohort["useful_groups"]),
                "l2w_groups": int(cohort["l2w_groups"]),
                "mean_delta": float(cohort["mean_delta"]),
                "harmed_groups": int(cohort["harmed_groups"]),
            }
            incumbent = best.get(str(cohort["cohort"]))
            key = (row["l2w_groups"], row["useful_groups"], row["mean_delta"])
            if incumbent is None or key > (
                incumbent["l2w_groups"], incumbent["useful_groups"], incumbent["mean_delta"]
            ):
                best[str(cohort["cohort"])] = row
    return dict(sorted(best.items()))


def _yield_metrics(evidence) -> dict[str, Any]:
    rows = tuple(evidence)
    total = max(1, len(rows))
    count = lambda predicate: sum(bool(predicate(row)) for row in rows)
    values = {
        "candidates": len(rows),
        "valid": len(rows),
        "behavior_changing": count(lambda row: row.action_change_count > 0),
        "meaningful_specialist": count(lambda row: row.useful_groups > 0),
        "strong_specialist": count(lambda row: row.tier in {"GLOBAL_SAFE", "ELITE_SPECIALIST", "STRONG_SPECIALIST"}),
        "elite_specialist": count(lambda row: row.tier in {"GLOBAL_SAFE", "ELITE_SPECIALIST"}),
        "global_safe": count(lambda row: row.production_safe and row.useful_groups >= 2),
        "l2w_specialist": count(lambda row: row.l2w_groups > 0),
        "hard_negative": count(lambda row: row.tier == "HARD_NEGATIVE"),
        "inert": count(lambda row: row.tier == "INERT"),
    }
    return {
        **values,
        **{f"{name}_rate": value / total for name, value in values.items() if name not in {"candidates", "valid"}},
        "valid_rate": values["valid"] / total,
    }


def _observed_strategic_domains(evidence) -> list[str]:
    observed = set()
    for item in evidence:
        if item.useful_groups <= 0:
            continue
        for rule in item.module.rules:
            action = rule.action
            features = {atom.feature for atom in (*rule.guard.atoms, *rule.exit_guard.atoms)}
            if action.kind in {"ROUTE_OVERRIDE", "COMMITTED_ROUTE"}:
                observed.add("routing")
                if "DAY" in features: observed.add("opening")
            if action.kind == "MARKET_INTENT":
                observed.add("market")
                if action.market_op == 1: observed.add("workforce")
                if action.market_op == 2: observed.add("land")
                if action.market_op in {3, 4, 5}: observed.add("recovery")
                if action.market_op == 6 and "OWN_SHED_TOTAL" in features: observed.add("inventory")
            if action.kind == "UNIT_REWRITE":
                if action.unit_op in {9, 10, 12}: observed.add("crops")
                if action.unit_op in {15, 16, 17} or "OWN_TILE_HAS_ANIMAL" in features: observed.add("animals")
                if action.unit_op in {5, 6, 7}: observed.add("inventory")
                observed.add("workforce")
            if rule.sketch.effect_class in {"RECOVER_RESOURCE_INVARIANT", "REDUCE_MAINTENANCE_BACKLOG"}:
                observed.add("recovery")
            if rule.sketch.effect_class == "ENGINE_MECHANIC":
                observed.add("engine_mechanics")
    return sorted(observed)


def _operator_evidence(results) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    skeletons: dict[str, set[str]] = {}
    semantics: dict[str, set[str]] = {}
    for result in results:
        evidence_by_sid = {item.module.semantic_id(): item for item in result.screen_evidence}
        for lineage in result.proposal_batch.lineages:
            operator = str(lineage.operator)
            item = evidence_by_sid.get(lineage.module.semantic_id())
            row = rows.setdefault(operator, {
                "proposals": 0, "canonical_programs": 0, "behavior_changing": 0,
                "useful": 0, "strong_or_elite": 0,
            })
            row["proposals"] += 1
            semantics.setdefault(operator, set()).add(lineage.module.semantic_id())
            if item is not None:
                row["behavior_changing"] += int(item.action_change_count > 0)
                row["useful"] += int(item.useful_groups > 0)
                row["strong_or_elite"] += int(item.tier in {"GLOBAL_SAFE", "ELITE_SPECIALIST", "STRONG_SPECIALIST"})
                skeletons.setdefault(operator, set()).add(repr(item.structural_skeleton))
    for operator, row in rows.items():
        row["canonical_programs"] = len(semantics.get(operator, ()))
        row["unique_skeletons"] = len(skeletons.get(operator, ()))
    return dict(sorted(rows.items()))


def _goal_evidence(results, route_ids: Sequence[int]) -> dict[str, dict[str, int]]:
    rows = {
        goal.goal_id: {"generated": 0, "behavior_changing": 0, "useful": 0, "strong": 0, "elite": 0}
        for goal in default_search_goals(route_ids)
    }
    for result in results:
        evidence_by_sid = {item.module.semantic_id(): item for item in result.screen_evidence}
        for lineage in result.proposal_batch.lineages:
            goal_id = lineage.source_goal_id
            if goal_id not in rows:
                continue
            item = evidence_by_sid.get(lineage.module.semantic_id())
            row = rows[goal_id]
            row["generated"] += 1
            if item is None:
                continue
            row["behavior_changing"] += int(item.action_change_count > 0)
            row["useful"] += int(item.useful_groups > 0)
            row["strong"] += int(item.tier in {"GLOBAL_SAFE", "ELITE_SPECIALIST", "STRONG_SPECIALIST"})
            row["elite"] += int(item.tier in {"GLOBAL_SAFE", "ELITE_SPECIALIST"})
    return rows


def _depth_evidence(results) -> dict[str, dict[str, int]]:
    rows: dict[str, dict[str, int]] = {}
    for result in results:
        evidence_by_sid = {item.module.semantic_id(): item for item in result.screen_evidence}
        for lineage in result.proposal_batch.lineages:
            key = str(int(lineage.depth))
            row = rows.setdefault(key, {"generated": 0, "behavior_changing": 0, "useful": 0, "strong_or_elite": 0})
            row["generated"] += 1
            item = evidence_by_sid.get(lineage.module.semantic_id())
            if item is not None:
                row["behavior_changing"] += int(item.action_change_count > 0)
                row["useful"] += int(item.useful_groups > 0)
                row["strong_or_elite"] += int(item.tier in {"GLOBAL_SAFE", "ELITE_SPECIALIST", "STRONG_SPECIALIST"})
    return dict(sorted(rows.items(), key=lambda item: int(item[0])))


def _cegis_effectiveness(records: Sequence[dict[str, object]]) -> dict[str, Any]:
    repairs = [
        row for row in records
        if str(row["target_labels"]["lineage"].get("operator")) == "cegis_repair"
    ]
    behavior_changed = outcome_changed = improved = worsened = inert = 0
    for record in repairs:
        source_rows = record["target_labels"].get("source_relative_case_outcomes", [])
        changed = any(bool(row.get("behavior_changed_vs_parent", False)) for row in source_rows)
        outcome = any(int(row.get("margin_delta", 0)) != 0 for row in source_rows)
        total_delta = sum(int(row.get("margin_delta", 0)) for row in source_rows)
        behavior_changed += int(changed)
        outcome_changed += int(outcome)
        improved += int(total_delta > 0)
        worsened += int(total_delta < 0)
        inert += int(not changed and not outcome)
    return {
        "repairs_labeled": len(repairs),
        "behavior_changing_vs_parent": behavior_changed,
        "outcome_changing_vs_parent": outcome_changed,
        "outcome_change_rate": outcome_changed / max(1, len(repairs)),
        "improved_parent_boundary": improved,
        "worsened_parent_boundary": worsened,
        "inert": inert,
        "previous_baseline": {"outcome_changing": 9, "repairs": 134, "rate": 9 / 134},
    }


def _followup_batch(factory: SpecialistFactory, *, recursive: int, repairs: int, ablations: int, seed: int) -> ProposalBatch:
    batches = []
    if repairs:
        batches.append(factory.cegis_repairs(limit=repairs))
    if recursive:
        batches.append(factory.propose(
            count=recursive, seed=seed, recursive=True,
            capability_level=len(factory.backend.ladder.LEVELS) - 1,
        ))
    if ablations:
        batches.append(factory.structural_ablations(limit=ablations))
    batches = [batch for batch in batches if batch.modules]
    if not batches:
        return ProposalBatch((), (), diversity_report(()), 0.0, 0.0, 0)
    return combine_proposal_batches(*batches)


def _saturation_curve(profile: ParentProfile, evaluator: InvariantExactCudaEvaluator, *, seed: int) -> list[dict[str, Any]]:
    rows = []
    for budget in (256, 1024, 4096, 10000):
        backend = InvariantSearchBackend(route_ids=profile.route_ids, evaluator=evaluator)
        batch = SpecialistFactory(backend, root_controller=profile.combined_agent_identity).propose(
            count=budget, seed=seed + budget,
        )
        rows.append({
            "requested": budget, "generated": len(batch.modules),
            "seconds": batch.generation_seconds,
            "proposals_per_second": batch.proposals_per_second,
            **dataclasses.asdict(batch.diversity),
        })
    return rows


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    profile = ParentProfile.from_directory(args.parent)
    loaded = [load_offline_specialist_bank(path, cohort_kind=kind) for kind, path in _bank_specs(args.bank)]
    bank = combine_offline_banks(loaded)
    for case in bank.cases:
        if not Path(case.tape_path).is_file():
            raise FileNotFoundError(f"specialist tape does not exist: {case.tape_path}")
    screen, label = split_specialist_panels(
        bank.cases, screen_cases=args.screen_cases, label_cases=args.label_cases, seed=args.seed,
    )
    evaluator = InvariantExactCudaEvaluator(
        profile, binary=args.evaluator, device=args.device, tpb=args.tpb,
        memory_fraction=args.memory_fraction,
    )
    backend = InvariantSearchBackend(route_ids=profile.route_ids, evaluator=evaluator)
    factory = SpecialistFactory(backend, root_controller=profile.combined_agent_identity)
    work = Path(args.work_dir)
    initial = run_specialist_funnel(
        factory, screen_panel=screen, label_panel=label,
        proposal_count=args.proposals, retain_count=args.retain, seed=args.seed,
        work_dir=str(work / "initial"), enable_cegis=True,
    )
    results = [initial]
    for followup_index in range(max(0, int(args.recursive_rounds))):
        followup_seed = args.seed + followup_index + 1
        followup = _followup_batch(
            factory, recursive=args.recursive, repairs=args.repairs,
            ablations=(args.ablations if followup_index == 0 else 0), seed=followup_seed,
        )
        if not followup.modules:
            break
        results.append(run_proposal_funnel(
            factory, proposal=followup, screen_panel=screen, label_panel=label,
            retain_count=min(args.followup_retain, len(followup.modules)), seed=followup_seed,
            work_dir=str(work / f"followup-{followup_index + 1}"), enable_cegis=True,
        ))
    records = tuple(
        record for result in results
        for record in specialist_corpus_records(
            result, evaluator_abi=evaluator.RESULT_ABI, feature_registry=profile.feature_registry,
        )
    )
    corpus_started = time.perf_counter()
    corpus_validation = validate_specialist_corpus(records)
    validation_seconds = time.perf_counter() - corpus_started
    arrays_started = time.perf_counter()
    arrays = invariant_corpus_arrays(records)
    array_seconds = time.perf_counter() - arrays_started
    corpus_path = work / "corpus" / "invariant_specialists_exact_cuda.jsonl"
    write_started = time.perf_counter()
    _write_jsonl(corpus_path, records)
    write_seconds = time.perf_counter() - write_started
    all_modules = tuple(result.proposal_batch.modules for result in results)
    flattened_modules = tuple(module for batch in all_modules for module in batch)
    all_labeled = tuple(item for result in results for item in result.labeled_evidence)
    activation_profiles = tuple(item.activation_topology for item in all_labeled)
    observed_domains = _observed_strategic_domains(all_labeled)
    labeled_diversity = diversity_report(tuple(item.module for item in all_labeled), activation_profiles=activation_profiles)
    saturation = [] if bool(getattr(args, "skip_saturation", False)) else _saturation_curve(
        profile, evaluator, seed=args.seed + 900000,
    )
    elapsed = time.perf_counter() - started
    exact_games = sum(result.exact_games for result in results)
    screen_seconds = sum(float(result.timing_seconds["screen_exact"]) for result in results)
    label_seconds = sum(float(result.timing_seconds["label_exact"]) for result in results)
    screened = sum(len(result.proposal_batch.modules) for result in results)
    labeled = sum(len(result.labeled_evidence) for result in results)
    report = {
        "schema": "bkg4_elite_specialist_factory_report_v1",
        "parent": profile.canonical_identity(),
        "candidate_abi": evaluator.CANDIDATE_ABI,
        "evaluator_abi": evaluator.RESULT_ABI,
        "exact_cuda": True,
        "ranker_trained": False,
        "bank": {
            "cases": len(bank.cases), "tapes": len({case.tape_path for case in bank.cases}),
            "screen_cases": len(screen), "screen_tapes": len({case.tape_path for case in screen}),
            "label_cases": len(label), "label_tapes": len({case.tape_path for case in label}),
            "tape_disjoint": not bool({case.tape_path for case in screen} & {case.tape_path for case in label}),
            "source_roles": list(bank.source_roles), "cohort_display": bank.cohort_display,
        },
        "proposal_diversity": dataclasses.asdict(diversity_report(flattened_modules)),
        "labeled_diversity": dataclasses.asdict(labeled_diversity),
        "saturation_curve": saturation,
        "tier_yield": _tier_yield(records),
        "screen_yield": _yield_metrics(tuple(item for result in results for item in result.screen_evidence)),
        "label_yield": _yield_metrics(all_labeled),
        "depth_evidence": _depth_evidence(results),
        "operator_evidence": _operator_evidence(results),
        "mechanism_goal_evidence": _goal_evidence(results, profile.route_ids),
        "cegis_effectiveness": _cegis_effectiveness(records),
        "qd_descriptor_contract": list(SPECIALIST_QD_DIMENSIONS),
        "campaign_capability": {
            "initial_min_level": 1,
            "followup_level": len(backend.ladder.LEVELS) - 1,
            "followup_level_name": backend.ladder.LEVELS[-1],
            "recursive_rounds_requested": int(args.recursive_rounds),
        },
        "cohort_headroom": _cohort_headroom(records),
        "coverage": [dataclasses.asdict(row) for row in strategic_coverage_matrix(
            profile.route_ids, observed_domains=observed_domains,
        )],
        "corpus": {**corpus_validation, "path": str(corpus_path), "array_shapes": {key: list(value.shape) for key, value in arrays.items()}},
        "rounds": [{
            "proposals": len(result.proposal_batch.modules),
            "retained": len(result.labeled_evidence),
            "tier_counts": result.tier_counts,
            "tier_thresholds": dataclasses.asdict(result.tier_thresholds),
            "timing_seconds": result.timing_seconds,
            "operator_timing_seconds": result.proposal_batch.operator_timing_seconds,
            "exact_games": result.exact_games,
        } for result in results],
        "throughput": {
            "elapsed_seconds": elapsed,
            "exact_games": exact_games,
            "games_per_second": exact_games / max(elapsed, 1e-9),
            "proposals_per_second": len(flattened_modules) / max(sum(result.proposal_batch.generation_seconds for result in results), 1e-9),
            "unique_candidates_per_second": len({module.semantic_id() for module in flattened_modules}) / max(elapsed, 1e-9),
            "screened_candidates_per_hour": len(flattened_modules) / max(elapsed, 1e-9) * 3600.0,
            "screened_candidates_per_day": len(flattened_modules) / max(elapsed, 1e-9) * 86400.0,
            "screen_stage_candidates_per_hour": screened / max(screen_seconds, 1e-9) * 3600.0,
            "screen_stage_candidates_per_day": screened / max(screen_seconds, 1e-9) * 86400.0,
            "label_stage_candidates_per_hour": labeled / max(label_seconds, 1e-9) * 3600.0,
            "label_stage_candidates_per_day": labeled / max(label_seconds, 1e-9) * 86400.0,
            "corpus_validation_seconds": validation_seconds,
            "ranker_array_seconds": array_seconds,
            "corpus_write_seconds": write_seconds,
            "cuda_profiles": evaluator.profile_history,
        },
    }
    report_path = work / "specialist_factory_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def parser() -> argparse.ArgumentParser:
    out = argparse.ArgumentParser(description="Generate a diverse exact-CUDA B4IM specialist corpus")
    out.add_argument("--parent", required=True)
    out.add_argument("--evaluator", required=True)
    out.add_argument("--bank", action="append", required=True, help="offline cohort as KIND=PATH")
    out.add_argument("--work-dir", required=True)
    out.add_argument("--proposals", type=int, default=4096)
    out.add_argument("--retain", type=int, default=512)
    out.add_argument("--screen-cases", type=int, default=64)
    out.add_argument("--label-cases", type=int, default=512)
    out.add_argument("--recursive", type=int, default=1024)
    out.add_argument("--recursive-rounds", type=int, default=2)
    out.add_argument("--repairs", type=int, default=512)
    out.add_argument("--ablations", type=int, default=512)
    out.add_argument("--followup-retain", type=int, default=256)
    out.add_argument("--seed", type=int, default=20260823)
    out.add_argument("--device", type=int, default=0)
    out.add_argument("--tpb", type=int, default=64)
    out.add_argument("--memory-fraction", type=float, default=0.70)
    out.add_argument(
        "--skip-saturation", action="store_true",
        help="skip the 256/1024/4096/10000 reporting-only proposal sweep (production shards)",
    )
    return out


def main(argv: Sequence[str] | None = None) -> int:
    report = run(parser().parse_args(argv))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
