from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from tools.bkg4_search.edit_search.benchmark import (
    _evaluate_children,
    _metric_row,
    _pairs_against_root,
    _recursive_entries,
    _single_parent_programs,
)
from tools.bkg4_search.edit_search.evaluator import ExactCudaEvaluator
from tools.bkg4_search.edit_search.fitness import LexicographicFitness, PairEvaluation, StateEvaluation
from tools.bkg4_search.edit_search.panels import CaseBank, CaseRecord
from tools.bkg4_search.edit_search.schema import ParentProfile
from tools.bkg4_search.invariant_evaluator import InvariantExactCudaEvaluator
from tools.bkg4_search.invariant_groups import (
    RobustGroupFitness,
    group_outcomes_from_case_rows,
    production_qualification,
    rotating_tape_folds,
    tape_group_id,
)
from tools.bkg4_search.invariant_module import InvariantStrategicModule
from tools.bkg4_search.invariant_schema import make_corpus_record, validate_quality_gated_corpus
from tools.bkg4_search.invariant_search import (
    InvariantSearchBackend,
    InvariantSearchArchive,
    pair_case_rows,
    semantic_descriptor,
)
from tools.bkg4_search.module_codec import ModuleGenome


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def _panel(rows: Sequence[CaseRecord], ids: Sequence[str]) -> tuple[CaseRecord, ...]:
    wanted = set(ids)
    return tuple(x for x in rows if x.case_id in wanted)


def _invariant_truth(
    module: InvariantStrategicModule,
    baseline: StateEvaluation,
    candidate: StateEvaluation,
    panel: Sequence[CaseRecord],
) -> dict[str, Any]:
    rows = pair_case_rows(baseline, candidate)
    outcomes = group_outcomes_from_case_rows(rows, {x.case_id: tape_group_id(x) for x in panel})
    fitness = RobustGroupFitness.from_outcomes(outcomes)
    qualification = production_qualification(outcomes)
    return {
        "semantic_id": module.semantic_id(),
        "descriptor": semantic_descriptor(module, activation_topology=InvariantSearchArchive._topology(outcomes)).__dict__,
        "fitness": fitness.__dict__,
        "qualification": qualification.__dict__,
        "cases": len(panel),
        "groups": len(outcomes),
        "activation_count": sum(candidate.activations),
        "action_change_count": sum(candidate.action_changes),
        "case_rows": list(rows),
        "group_outcomes": [x.__dict__ for x in outcomes],
    }


def _state_truth(
    baseline: StateEvaluation,
    candidate: StateEvaluation,
    panel: Sequence[CaseRecord],
) -> dict[str, Any]:
    rows = pair_case_rows(baseline, candidate)
    outcomes = group_outcomes_from_case_rows(rows, {x.case_id: tape_group_id(x) for x in panel})
    return {
        "semantic_id": candidate.semantic_id,
        "fitness": RobustGroupFitness.from_outcomes(outcomes).__dict__,
        "qualification": production_qualification(outcomes).__dict__,
        "activation_count": sum(candidate.activations),
        "action_change_count": sum(candidate.action_changes),
        "group_outcomes": [x.__dict__ for x in outcomes],
    }


def _best_research(archive: InvariantSearchArchive):
    if not archive.research:
        raise RuntimeError("invariant search archive is empty")
    return min(
        archive.research.values(),
        key=lambda x: (x.fitness.key(), x.module.semantic_id()),
    )


def _legacy_best(entries, pairs):
    program, child = min(
        entries,
        key=lambda x: (
            LexicographicFitness.from_pair(pairs[x[0].program_id], complexity=x[0].program_length).key(),
            x[0].program_id,
        ),
    )
    return program, child


def _legacy_ranked(entries, pairs, *, limit: int = 8):
    return tuple(sorted(
        entries,
        key=lambda x: (
            LexicographicFitness.from_pair(
                pairs[x[0].program_id], complexity=x[0].program_length,
            ).key(),
            x[0].program_id,
        ),
    )[:max(1, int(limit))])


def _safe_useful(truth: Mapping[str, Any]) -> bool:
    return bool(
        truth["fitness"]["unsafe_groups"] == 0
        and truth["fitness"]["useful_groups"] > 0
    )


def _throughput(root: Path) -> dict[str, Any]:
    pattern = re.compile(r"games=(\d+).*?gpu_ms=([0-9.]+).*?games_s=([0-9.]+)")
    games = 0
    gpu_ms = 0.0
    rates = []
    logs = 0
    for path in root.rglob("*.cuda.log"):
        match = pattern.search(path.read_text(encoding="utf-8", errors="replace"))
        if not match:
            continue
        logs += 1
        games += int(match.group(1))
        gpu_ms += float(match.group(2))
        rates.append(float(match.group(3)))
    return {
        "cuda_invocations": logs,
        "games": games,
        "gpu_ms": gpu_ms,
        "effective_games_s": (1000.0 * games / gpu_ms if gpu_ms else 0.0),
        "min_invocation_games_s": min(rates) if rates else 0.0,
        "max_invocation_games_s": max(rates) if rates else 0.0,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    work = args.work_dir.resolve()
    work.mkdir(parents=True, exist_ok=True)
    profile = ParentProfile.from_directory(args.parent_dir)
    discovery = CaseBank.load_jsonl(args.discovery).rows
    certification = CaseBank.load_jsonl(args.certification).rows
    ranker_validation = CaseBank.load_jsonl(args.ranker_validation).rows
    if any("cert" in x.role.lower() or "holdout" in x.role.lower() for x in discovery):
        raise ValueError("discovery bank contains certification roles")
    discovery_ids = {x.case_id for x in discovery}
    if discovery_ids & {x.case_id for x in certification} or discovery_ids & {x.case_id for x in ranker_validation}:
        raise ValueError("search banks overlap by case ID")

    invariant_eval = InvariantExactCudaEvaluator(profile, binary=args.invariant_evaluator)
    legacy_eval = ExactCudaEvaluator(profile, binary=args.legacy_evaluator)

    spike_backend = InvariantSearchBackend(route_ids=profile.route_ids, evaluator=invariant_eval)
    spike_modules = spike_backend.propose(n=args.spike_budget, seed=args.seed)
    if len(spike_modules) != args.spike_budget:
        raise RuntimeError(f"invariant spike underfilled: {len(spike_modules)} != {args.spike_budget}")
    spike_round = spike_backend.evaluate_batch(
        spike_modules, discovery, work_dir=str(work / "spike"), panel_id="discovery/spike",
    )
    empty = InvariantStrategicModule()
    spike_baseline = spike_round.evaluations[empty.semantic_id()]
    spike_truth = [
        _invariant_truth(m, spike_baseline, spike_round.evaluations[m.semantic_id()], discovery)
        for m in spike_modules
    ]
    best_spike = _best_research(spike_backend.archive)

    folds = rotating_tape_folds(discovery, n_folds=5, seed=args.seed, discovery_only=True)
    fold_rows = []
    fold_modules: dict[str, InvariantStrategicModule] = {}
    fold_support: dict[str, int] = {}
    outer_support: dict[str, int] = {}
    for fold in folds:
        train = _panel(discovery, fold.train_case_ids)
        validation = _panel(discovery, fold.validation_case_ids)
        backend = InvariantSearchBackend(route_ids=profile.route_ids, evaluator=invariant_eval)
        crossfit = backend.search_crossfit(
            train,
            rounds=args.fold_rounds,
            candidates_per_round=args.fold_budget,
            seed=args.seed + 1009 * (fold.fold + 1),
            work_dir=str(work / "folds" / f"f{fold.fold}"),
            top_k=args.fold_top_k,
            selection_fraction=args.selection_fraction,
        )
        transferable = tuple(x.module for x in crossfit.transferable_elites)
        for module in transferable:
            fold_modules.setdefault(module.semantic_id(), module)
            fold_support[module.semantic_id()] = fold_support.get(module.semantic_id(), 0) + 1
        val_exact = invariant_eval.evaluate(
            (empty, *transferable), validation,
            work_dir=work / "folds" / f"f{fold.fold}" / "outer_validation",
            panel_id=f"discovery/fold{fold.fold}/outer-validation",
        ) if transferable else {}
        heldout_candidates = [
            _invariant_truth(
                module, val_exact[empty.semantic_id()], val_exact[module.semantic_id()], validation,
            )
            for module in transferable
        ]
        for module, truth in zip(transferable, heldout_candidates):
            if _safe_useful(truth):
                outer_support[module.semantic_id()] = outer_support.get(module.semantic_id(), 0) + 1
        best_heldout = heldout_candidates[0] if heldout_candidates else None
        fold_rows.append({
            "fold": fold.fold,
            "candidate_budget_per_round": args.fold_budget,
            "rounds": args.fold_rounds,
            "train_cases": len(train),
            "validation_cases": len(validation),
            "train_tapes": len(fold.train_tapes),
            "validation_tapes": len(fold.validation_tapes),
            "tape_disjoint": not bool(set(fold.train_tapes) & set(fold.validation_tapes)),
            "internal_search_tapes": len(crossfit.split.search_tapes),
            "internal_selection_tapes": len(crossfit.split.selection_tapes),
            "internal_tape_disjoint": not bool(
                set(crossfit.split.search_tapes) & set(crossfit.split.selection_tapes)
            ),
            "frozen_top_k": len(crossfit.frozen_semantic_ids),
            "transferable_candidates": len(transferable),
            "exact_candidate_case_evaluations": (
                crossfit.exact_candidate_cases + (len(transferable) + 1) * len(validation)
            ),
            "search": {
                "attempted_semantics": crossfit.search.attempted_semantics,
                "production_qualified": sum(
                    round_.production_qualified for round_ in crossfit.search.rounds
                ),
                "unsafe": sum(round_.unsafe for round_ in crossfit.search.rounds),
            },
            "heldout_candidates": heldout_candidates,
            "heldout": best_heldout,
            "heldout_safe_useful": sum(_safe_useful(x) for x in heldout_candidates),
            "heldout_production": sum(x["qualification"]["production"] for x in heldout_candidates),
            "heldout_l2w": sum(x["fitness"]["l2w_groups"] for x in heldout_candidates),
        })

    transfer_folds = sum(row["heldout_safe_useful"] > 0 for row in fold_rows)
    repeated_elite_folds = max(outer_support.values(), default=0)
    l2w_folds = sum(row["heldout_l2w"] > 0 for row in fold_rows)
    if transfer_folds < 4 or max(repeated_elite_folds, l2w_folds) < 2:
        raise RuntimeError(
            "invariant outer-fold gate failed: "
            f"safe_useful={transfer_folds}/5 repeated_elite={repeated_elite_folds}/5 "
            f"l2w={l2w_folds}/5"
        )

    benchmark_fold = folds[0]
    bench_train = _panel(discovery, benchmark_fold.train_case_ids)
    bench_holdout = _panel(discovery, benchmark_fold.validation_case_ids)
    observations = _jsonl(args.observations)
    root_genome = ModuleGenome(0, ())
    root_genome.validate(profile.route_ids)
    legacy_train_root = legacy_eval.evaluate(
        (root_genome,), bench_train, work_dir=work / "benchmark" / "root_train", panel_id="benchmark/train",
    )[root_genome.semantic_id()]
    legacy_holdout_root = legacy_eval.evaluate(
        (root_genome,), bench_holdout, work_dir=work / "benchmark" / "root_holdout", panel_id="benchmark/holdout",
    )[root_genome.semantic_id()]

    legacy_arms = {}
    for name, random_typed, offset in (("random_typed_legacy", True, 23), ("fixed_systematic_legacy", False, 37)):
        entries = _single_parent_programs(
            root_genome, observations, profile.route_ids,
            budget=args.benchmark_budget, seed=args.seed + offset, random_typed=random_typed,
        )
        evaluations = _evaluate_children(
            entries, evaluator=legacy_eval, panel=bench_train,
            work_dir=work / "benchmark" / name / "train", panel_id="benchmark/train",
        )
        pairs = _pairs_against_root(entries, evaluations, legacy_train_root, bench_train)
        ranked = _legacy_ranked(entries, pairs, limit=args.benchmark_top_k)
        held_children = tuple(child for _, child in ranked)
        hold_evals = legacy_eval.evaluate(
            held_children, bench_holdout,
            work_dir=work / "benchmark" / name / "holdout", panel_id="benchmark/holdout",
        )
        held_truth = [
            _state_truth(legacy_holdout_root, hold_evals[child.semantic_id()], bench_holdout)
            for child in held_children
        ]
        legacy_arms[name] = {
            **_metric_row(entries, pairs, legacy_train_root, candidate_budget=args.benchmark_budget, case_count=len(bench_train)),
            "frozen_top_k": len(held_children),
            "best_train_semantic_id": held_children[0].semantic_id(),
            "heldout_candidates": held_truth,
            "heldout_safe_useful": sum(_safe_useful(x) for x in held_truth),
            "heldout_production": sum(x["qualification"]["production"] for x in held_truth),
            "heldout_l2w": sum(x["fitness"]["l2w_groups"] for x in held_truth),
        }

    permanent_entries, permanent_pairs, permanent_stats = _recursive_entries(
        root_genome, profile.route_ids, observations, panel=bench_train, evaluator=legacy_eval,
        root_eval=legacy_train_root, budget=args.benchmark_budget, seed=args.seed + 53,
        work_dir=work / "benchmark" / "permanent_module_genome" / "train", panel_id="benchmark/train",
    )
    permanent_ranked = _legacy_ranked(permanent_entries, permanent_pairs, limit=args.benchmark_top_k)
    permanent_children = tuple(child for _, child in permanent_ranked)
    permanent_hold = legacy_eval.evaluate(
        permanent_children, bench_holdout,
        work_dir=work / "benchmark" / "permanent_module_genome" / "holdout", panel_id="benchmark/holdout",
    )
    permanent_truth = [
        _state_truth(legacy_holdout_root, permanent_hold[child.semantic_id()], bench_holdout)
        for child in permanent_children
    ]
    legacy_arms["permanent_module_genome"] = {
        **_metric_row(
            permanent_entries, permanent_pairs, legacy_train_root,
            candidate_budget=args.benchmark_budget, case_count=len(bench_train), proposal_stats=permanent_stats,
        ),
        "frozen_top_k": len(permanent_children),
        "best_train_semantic_id": permanent_children[0].semantic_id(),
        "heldout_candidates": permanent_truth,
        "heldout_safe_useful": sum(_safe_useful(x) for x in permanent_truth),
        "heldout_production": sum(x["qualification"]["production"] for x in permanent_truth),
        "heldout_l2w": sum(x["fitness"]["l2w_groups"] for x in permanent_truth),
    }

    invariant_bench = InvariantSearchBackend(route_ids=profile.route_ids, evaluator=invariant_eval)
    invariant_modules = invariant_bench.propose(n=args.benchmark_budget, seed=args.seed + 71)
    invariant_round = invariant_bench.evaluate_batch(
        invariant_modules, bench_train,
        work_dir=str(work / "benchmark" / "invariant_strategic" / "train"), panel_id="benchmark/train",
    )
    invariant_ranked = sorted(
        invariant_bench.archive.research.values(),
        key=lambda elite: (
            not elite.qualification.production,
            elite.fitness.key(),
            elite.module.semantic_id(),
        ),
    )[:args.benchmark_top_k]
    invariant_children = tuple(x.module for x in invariant_ranked)
    invariant_hold = invariant_eval.evaluate(
        (empty, *invariant_children), bench_holdout,
        work_dir=work / "benchmark" / "invariant_strategic" / "holdout", panel_id="benchmark/holdout",
    )
    invariant_hold_truth = [
        _invariant_truth(
            module, invariant_hold[empty.semantic_id()], invariant_hold[module.semantic_id()], bench_holdout,
        )
        for module in invariant_children
    ]
    invariant_train_truth = [
        _invariant_truth(
            m, invariant_round.evaluations[empty.semantic_id()],
            invariant_round.evaluations[m.semantic_id()], bench_train,
        ) for m in invariant_modules
    ]
    legacy_arms["invariant_strategic_b4im"] = {
        "requested_candidates": args.benchmark_budget,
        "evaluated_candidates": len(invariant_modules),
        "exact_candidate_case_evaluations": args.benchmark_budget * len(bench_train),
        "unique_valid_proposals": len({m.semantic_id() for m in invariant_modules}),
        "behavior_changing_rate": sum(x["action_change_count"] > 0 for x in invariant_train_truth) / len(invariant_train_truth),
        "new_loss_candidates": sum(x["fitness"]["groups_with_new_loss"] > 0 for x in invariant_train_truth),
        "production_qualified_elites": sum(x["qualification"]["production"] for x in invariant_train_truth),
        "semantic_niches": len({json.dumps(x["descriptor"], sort_keys=True) for x in invariant_train_truth}),
        "activation_count": sum(x["activation_count"] for x in invariant_train_truth),
        "frozen_top_k": len(invariant_children),
        "best_train_semantic_id": invariant_children[0].semantic_id(),
        "heldout_candidates": invariant_hold_truth,
        "heldout_safe_useful": sum(_safe_useful(x) for x in invariant_hold_truth),
        "heldout_production": sum(x["qualification"]["production"] for x in invariant_hold_truth),
        "heldout_l2w": sum(x["fitness"]["l2w_groups"] for x in invariant_hold_truth),
    }

    corpus_modules_by_id = {module.semantic_id(): module for module in spike_modules}
    corpus_modules_by_id.update(fold_modules)
    corpus_modules = tuple(corpus_modules_by_id.values())
    corpus_exact = invariant_eval.evaluate(
        (empty, *corpus_modules), discovery,
        work_dir=work / "corpus" / "exact_labels", panel_id="discovery/corpus-labels",
    )
    corpus_baseline = corpus_exact[empty.semantic_id()]
    corpus_truth_by_id = {
        module.semantic_id(): _invariant_truth(
            module, corpus_baseline, corpus_exact[module.semantic_id()], discovery,
        )
        for module in corpus_modules
    }
    frozen_pool = [
        module for module in fold_modules.values()
        if corpus_truth_by_id[module.semantic_id()]["qualification"]["production"]
    ]
    if not frozen_pool:
        raise RuntimeError("outer-fold validation produced no full-discovery production candidate")
    frozen = min(
        frozen_pool,
        key=lambda module: (
            -outer_support.get(module.semantic_id(), 0),
            -fold_support.get(module.semantic_id(), 0),
            RobustGroupFitness(**corpus_truth_by_id[module.semantic_id()]["fitness"]).key(),
            module.semantic_id(),
        ),
    )
    certification_exact = invariant_eval.evaluate(
        (empty, frozen), certification,
        work_dir=work / "certification", panel_id="certification/frozen-label-only",
    )
    certification_truth = _invariant_truth(
        frozen, certification_exact[empty.semantic_id()], certification_exact[frozen.semantic_id()], certification,
    )

    corpus_records = []
    for module in corpus_modules:
        truth = corpus_truth_by_id[module.semantic_id()]
        support = int(outer_support.get(module.semantic_id(), 0))
        internal_support = int(fold_support.get(module.semantic_id(), 0))
        if support >= 2 and truth["qualification"]["production"]:
            source_quality = "crossfit_positive"
        elif truth["qualification"]["production"] or _safe_useful(truth):
            source_quality = "discovery_boundary"
        elif truth["fitness"]["unsafe_groups"]:
            source_quality = "unsafe_negative"
        elif truth["action_change_count"] == 0:
            source_quality = "inert_negative"
        else:
            source_quality = "safe_negative"
        labels = {
            "fitness": truth["fitness"],
            "qualification": truth["qualification"],
            "activation_count": truth["activation_count"],
            "action_change_count": truth["action_change_count"],
            "group_outcomes": truth["group_outcomes"],
            "exact_data_role": "discovery",
            "source_quality": source_quality,
            "crossfit_fold_support": internal_support,
            "outer_fold_support": support,
        }
        corpus_records.append(make_corpus_record(
            module, labels=labels,
            provenance={
                "parent": profile.combined_agent_identity,
                "controller": profile.controller_semantic_hash,
                "evaluator_abi": invariant_eval.RESULT_ABI,
                "candidate_abi": invariant_eval.CANDIDATE_ABI,
                "feature_registry": profile.feature_registry,
                "exact_evaluator": True,
                "data_role": "discovery",
            },
        ))
    corpus_report = validate_quality_gated_corpus(corpus_records)
    corpus_report["quality_counts"] = {
        name: corpus_report[name] for name in (
            "crossfit_positive", "discovery_boundary", "unsafe_negative",
            "inert_negative", "safe_negative",
        )
    }
    corpus_path = work / "corpus" / "invariant_exact_cuda_corpus.jsonl"
    corpus_path.parent.mkdir(parents=True, exist_ok=True)
    corpus_path.write_text("".join(json.dumps(x, sort_keys=True) + "\n" for x in corpus_records), encoding="utf-8")

    report = {
        "schema": 1,
        "parent": profile.canonical_identity(),
        "isolation": {
            "discovery_cases": len(discovery),
            "certification_cases": len(certification),
            "ranker_validation_cases": len(ranker_validation),
            "search_inputs": [str(args.discovery)],
            "certification_used_after_freeze": True,
            "ranker_validation_used_in_search": False,
        },
        "spike": {
            "candidate_budget": args.spike_budget,
            "exact_candidate_case_evaluations": spike_round.exact_candidate_cases,
            "behavior_changing_candidates": sum(x["action_change_count"] > 0 for x in spike_truth),
            "active_candidates": sum(x["activation_count"] > 0 for x in spike_truth),
            "unsafe_candidates": sum(x["fitness"]["unsafe_groups"] > 0 for x in spike_truth),
            "production_qualified": sum(x["qualification"]["production"] for x in spike_truth),
            "semantic_niches": len({json.dumps(x["descriptor"], sort_keys=True) for x in spike_truth}),
            "best": corpus_truth_by_id[frozen.semantic_id()],
        },
        "generalization_gate": {
            "required_safe_useful_folds": 4,
            "safe_useful_folds": transfer_folds,
            "repeated_elite_folds": repeated_elite_folds,
            "l2w_folds": l2w_folds,
            "passed": True,
        },
        "folds": fold_rows,
        "benchmark": {
            "candidate_budget_per_arm": args.benchmark_budget,
            "frozen_top_k_per_arm": args.benchmark_top_k,
            "train_cases": len(bench_train),
            "heldout_cases": len(bench_holdout),
            "tape_disjoint": not bool(set(benchmark_fold.train_tapes) & set(benchmark_fold.validation_tapes)),
            "arms": legacy_arms,
        },
        "certification": {
            "frozen_before_evaluation": True,
            "semantic_id": frozen.semantic_id(),
            "discovery_crossfit_fold_support": fold_support.get(frozen.semantic_id(), 0),
            "discovery_outer_fold_support": outer_support.get(frozen.semantic_id(), 0),
            "truth": certification_truth,
            "candidate_promoted": False,
        },
        "corpus": {**corpus_report, "path": str(corpus_path), "ranker_trained": False},
    }
    report["throughput"] = _throughput(work)
    return report


def _markdown(report: Mapping[str, Any]) -> str:
    spike = report["spike"]
    lines = [
        "# Invariant Strategic Generator Validation",
        "",
        "All game labels in this report were produced by the exact resident CUDA evaluators.",
        "Certification was evaluated only after the discovery candidate was frozen; ranker validation was not read by search.",
        "",
        "## Representation Spike",
        "",
        f"- Candidates: {spike['candidate_budget']}",
        f"- Candidate-case evaluations (including shared baseline): {spike['exact_candidate_case_evaluations']}",
        f"- Active / behavior-changing: {spike['active_candidates']} / {spike['behavior_changing_candidates']}",
        f"- Production-qualified: {spike['production_qualified']}",
        f"- Unsafe: {spike['unsafe_candidates']}",
        f"- Structural semantic niches: {spike['semantic_niches']}",
        "",
        "## Five Tape-Disjoint Splits",
        "",
        "| Fold | Budget | Train cases | Held-out cases | Held-out useful groups | Held-out regressions | Production |",
        "|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in report["folds"]:
        held = row["heldout"]
        lines.append(
            f"| {row['fold']} | {row['candidate_budget_per_round']}x{row['rounds']} | {row['train_cases']} | {row['validation_cases']} | "
            f"{held['fitness']['useful_groups']} | {held['fitness']['unsafe_groups']} | "
            f"{held['qualification']['production']} |"
        )
    lines.extend([
        "",
        "## Equal-Budget Benchmark",
        "",
        f"Each arm evaluated {report['benchmark']['candidate_budget_per_arm']} children on "
        f"{report['benchmark']['train_cases']} discovery cases, then evaluated its frozen best child on "
        f"{report['benchmark']['heldout_cases']} tape-disjoint discovery cases.",
        "",
        "```json",
        json.dumps(report["benchmark"]["arms"], indent=2, sort_keys=True),
        "```",
        "",
        "## Certification And Corpus",
        "",
        f"- Frozen certification semantic ID: `{report['certification']['semantic_id']}`",
        f"- Certification production qualification: {report['certification']['truth']['qualification']['production']}",
        f"- Candidate promoted: {report['certification']['candidate_promoted']}",
        f"- Fresh corpus rows / unique semantics: {report['corpus']['rows']} / {report['corpus']['unique_semantics']}",
        f"- Ranker trained: {report['corpus']['ranker_trained']}",
        "",
        "## Throughput",
        "",
        "```json",
        json.dumps(report["throughput"], indent=2, sort_keys=True),
        "```",
        "",
    ])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-dir", type=Path, required=True)
    parser.add_argument("--discovery", type=Path, required=True)
    parser.add_argument("--certification", type=Path, required=True)
    parser.add_argument("--ranker-validation", type=Path, required=True)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--invariant-evaluator", type=Path, required=True)
    parser.add_argument("--legacy-evaluator", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    parser.add_argument("--spike-budget", type=int, default=128)
    parser.add_argument("--fold-budget", type=int, default=64)
    parser.add_argument("--fold-rounds", type=int, default=1)
    parser.add_argument("--fold-top-k", type=int, default=8)
    parser.add_argument("--selection-fraction", type=float, default=0.25)
    parser.add_argument("--benchmark-budget", type=int, default=64)
    parser.add_argument("--benchmark-top-k", type=int, default=8)
    parser.add_argument("--seed", type=int, default=842301)
    args = parser.parse_args()
    report = run(args)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text(_markdown(report), encoding="utf-8")
    print(json.dumps({
        "report": str(args.report), "spike": report["spike"],
        "folds": len(report["folds"]), "benchmark_arms": list(report["benchmark"]["arms"]),
        "certification": report["certification"]["truth"]["qualification"],
        "corpus": report["corpus"], "throughput": report["throughput"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
