from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from tools.bkg4_search.module_codec import ModuleGenome

from .archive import SearchArchive
from .benchmark import benchmark_proposers, load_evaluator_cases_tsv, run_anti_saturation
from .corpus import SearchCorpusStore, validate_export
from .evaluator import ExactCudaEvaluator
from .panels import CaseBank, CasePanelManager
from .schema import ParentProfile
from .search import SearchEngine


def _git_head() -> str:
    try:
        p = subprocess.run(["git", "rev-parse", "HEAD"], text=True, capture_output=True, check=False)
        value = p.stdout.strip()
        return value if p.returncode == 0 and len(value) == 40 else "unknown"
    except OSError:
        return "unknown"


def _load_observations(path: str | Path) -> list[dict[str, Any]]:
    rows = [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]
    if not rows:
        raise ValueError("search observations file is empty")
    required = {"feature", "value"}
    if any(not isinstance(x, dict) or not required.issubset(x) for x in rows):
        raise ValueError("each observation row needs at least feature and value")
    return rows


def _write_programs(path: str | Path, programs) -> None:
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("".join(json.dumps(x.canonical(include_provenance=True), sort_keys=True) + "\n" for x in programs), encoding="utf-8")


def assert_archive_parent(archive: SearchArchive, profile: ParentProfile) -> None:
    if archive.base_identity != profile.combined_agent_identity:
        raise ValueError(
            "concrete search archive belongs to parent "
            f"{archive.base_identity}, not {profile.combined_agent_identity}; create a fresh archive and import/reinstantiate motifs"
        )
    if tuple(archive.route_ids) != tuple(profile.route_ids):
        raise ValueError("concrete search archive route bank differs from supplied parent profile")


def _archive_summary(archive: SearchArchive) -> dict[str, Any]:
    return {
        "schema_version": archive.schema_version,
        "grammar_id": archive.grammar_id,
        "emitter_registry_version": archive.emitter_registry_version,
        "qd_descriptor_schema_version": archive.qd_descriptor_schema_version,
        "base_identity": archive.base_identity,
        "root_semantic_id": archive.root_semantic_id,
        "run_id": archive.run_id,
        "master_seed": archive.master_seed,
        "source_identity": archive.source_identity,
        "states": len(archive.states),
        "programs": len(archive.programs),
        "production_niches": len(archive.production_niches),
        "exploration_niches": len(archive.exploration_niches),
        "novelty_entries": len(archive.novelty_archive),
        "hard_cases": len(archive.hard_cases),
        "motifs": len(archive.motifs),
        "exact_state_certifications": sum(len(x) for x in archive.state_evaluations.values()),
        "escalation_events": len(archive.escalation_history),
        "run_config": archive.run_config,
    }


def cmd_init(args) -> int:
    profile = ParentProfile.from_directory(args.parent_dir)
    root = ModuleGenome(prefix_step=int(args.root_prefix), rules=())
    root.validate(profile.route_ids)
    archive = SearchArchive.new(
        root_genome=root,
        base_identity=profile.combined_agent_identity,
        route_ids=profile.route_ids,
        run_id=args.run_id,
        master_seed=args.seed,
        source_identity=_git_head(),
        run_config={"root_prefix": int(args.root_prefix), "parent_dir": str(Path(args.parent_dir).resolve())},
    )
    if args.motifs_from:
        previous = SearchArchive.load(args.motifs_from)
        archive.motifs = json.loads(json.dumps(previous.motifs))
    archive.save(args.archive)
    print(json.dumps(_archive_summary(archive), sort_keys=True))
    return 0


def cmd_validate(args) -> int:
    archive = SearchArchive.load(args.archive)
    print(json.dumps({"valid": True, **_archive_summary(archive)}, sort_keys=True))
    return 0


def cmd_resume(args) -> int:
    archive = SearchArchive.load(args.archive)
    print(json.dumps({"resumable": True, **_archive_summary(archive)}, sort_keys=True))
    return 0


def cmd_propose(args) -> int:
    archive = SearchArchive.load(args.archive)
    observations = _load_observations(args.observations)
    engine = SearchEngine(route_ids=archive.route_ids)
    programs = engine.propose_generation(
        archive, observations=observations, total=args.count, generation=args.generation, seed=args.seed,
    )
    if not programs:
        raise RuntimeError("search produced no unique legal proposals")
    _write_programs(args.out, programs)
    archive.run_config.update({
        "last_proposal_generation": int(args.generation),
        "last_proposal_count": len(programs),
        "last_observations": str(Path(args.observations).resolve()),
    })
    archive.save(args.archive)
    print(json.dumps({
        "generation": int(args.generation), "requested": int(args.count), "unique_programs": len(programs),
        "source_parents": len({p.source_parent_semantic_id for p in programs}),
        "operations": sorted({a.operation for p in programs for a in p.atoms}),
        "out": str(Path(args.out).resolve()),
    }, sort_keys=True))
    return 0


def _seed_panel_manager_from_archive(manager: CasePanelManager, archive: SearchArchive) -> None:
    for cid, meta in archive.hard_cases.items():
        if cid not in manager.bank.by_id:
            continue
        manager.hard_cases[cid] = json.loads(json.dumps(meta))


def _latest_parent_evaluations(archive: SearchArchive) -> dict[str, Any]:
    parent_ids = set(archive.parent_state_ids())
    by_panel: dict[str, dict[str, Any]] = {}
    for sid, panels in archive.state_evaluations.items():
        if sid not in parent_ids:
            continue
        for panel_id, payload in panels.items():
            if not str(panel_id).endswith("/full"):
                continue
            from .fitness import StateEvaluation
            by_panel.setdefault(str(panel_id), {})[sid] = StateEvaluation.from_dict(payload)
    if not by_panel:
        return {}
    def generation(panel_id: str) -> int:
        for part in panel_id.split("/"):
            if part.startswith("g") and part[1:].isdigit():
                return int(part[1:])
        return -1
    best_panel = max(by_panel, key=lambda p: (len(by_panel[p]), generation(p), p))
    return by_panel[best_panel]


def cmd_search(args) -> int:
    archive = SearchArchive.load(args.archive)
    profile = ParentProfile.from_directory(args.parent_dir)
    assert_archive_parent(archive, profile)
    observations = _load_observations(args.observations)
    bank = CaseBank.load_jsonl(args.case_bank)
    manager = CasePanelManager(
        bank,
        fragile_margin=args.fragile_margin,
        random_anchor_fraction=args.random_anchor_fraction,
        useful_recall_floor=args.useful_recall_floor,
        danger_recall_floor=args.danger_recall_floor,
    )
    _seed_panel_manager_from_archive(manager, archive)
    evaluator = ExactCudaEvaluator(
        profile,
        binary=args.evaluator,
        device=args.device,
        tpb=args.tpb,
        memory_fraction=args.memory_fraction,
        cuda_visible_devices=args.cuda_visible_devices,
    )
    corpus_root = Path(args.corpus)
    if (corpus_root / "corpus_manifest.json").is_file():
        corpus = SearchCorpusStore(corpus_root)
        manifest = json.loads((corpus_root / "corpus_manifest.json").read_text())
        if str(manifest.get("base_identity")) != archive.base_identity:
            raise ValueError("existing exact corpus belongs to a different BKG4 parent")
        exported_generations = {int(x["generation"]) for x in manifest.get("generations", [])}
    else:
        corpus = SearchCorpusStore.create(corpus_root, profile=profile, case_bank=bank, archive=archive)
        exported_generations = set()
    engine = SearchEngine(route_ids=archive.route_ids)
    summaries = []
    for offset in range(int(args.rounds)):
        generation = int(args.start_generation) + offset
        if generation in exported_generations:
            raise ValueError(f"generation {generation} already exists in exact corpus")
        parent_evals = _latest_parent_evaluations(archive)
        programs = engine.propose_generation(
            archive,
            observations=observations,
            total=args.candidates,
            generation=generation,
            seed=int(args.seed) + generation * 100003,
            parent_evaluations=parent_evals,
        )
        if not programs:
            raise RuntimeError(f"generation {generation} produced no unique legal candidates")
        result = engine.evaluate_round(
            archive,
            programs,
            evaluator=evaluator,
            panel_manager=manager,
            screen_size=args.screen_size,
            final_size=args.final_size,
            stage2_budget=args.stage2_budget,
            generation=generation,
            seed=int(args.seed) + generation * 104729,
            work_dir=Path(args.work_dir) / f"g{generation:06d}",
            audit_all=bool(args.audit_all),
        )
        corpus.append_exact(generation=generation, archive=archive, program_pairs=result.full_source_pairs)
        archive.run_config.update({
            "last_search_generation": generation,
            "candidates_per_round": int(args.candidates),
            "screen_size": int(args.screen_size),
            "final_size": int(args.final_size),
            "stage2_budget": int(args.stage2_budget),
            "case_bank": str(Path(args.case_bank).resolve()),
            "case_bank_identity": bank.identity(),
            "observations": str(Path(args.observations).resolve()),
            "evaluator": evaluator.config_record(),
        })
        archive.save(args.archive)
        summary = {
            "generation": generation,
            "proposed": len(programs),
            "finalists": len(result.finalist_program_ids),
            "full_exact_programs": len(result.full_source_pairs),
            "counterexamples": list(result.counterexamples),
            "stage_recall": result.stage_recall,
            "signals": result.signals.__dict__,
            "escalation": result.escalation.__dict__,
            "production_niches": len(archive.production_niches),
            "exploration_niches": len(archive.exploration_niches),
            "motifs": len(archive.motifs),
        }
        summaries.append(summary)
        print("ROUND", json.dumps(summary, sort_keys=True), flush=True)
    report = {
        "archive": str(Path(args.archive).resolve()),
        "corpus": str(corpus_root.resolve()),
        "rounds": summaries,
        "archive_summary": _archive_summary(archive),
    }
    if args.report:
        p = Path(args.report); p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print("RESULT", json.dumps(report, sort_keys=True))
    return 0


def _cuda_evaluator_from_args(profile: ParentProfile, args) -> ExactCudaEvaluator:
    return ExactCudaEvaluator(
        profile,
        binary=args.evaluator,
        device=args.device,
        tpb=args.tpb,
        memory_fraction=args.memory_fraction,
        cuda_visible_devices=args.cuda_visible_devices,
    )


def _root_scored_bank(
    bank: CaseBank,
    *,
    root: ModuleGenome,
    evaluator: ExactCudaEvaluator,
    work_dir: str | Path,
    panel_id: str,
) -> CaseBank:
    evaluation = evaluator.evaluate([root], bank.rows, work_dir=work_dir, panel_id=panel_id)[root.semantic_id()]
    return CaseBank(
        replace(case, baseline_margin=int(margin))
        for case, margin in zip(bank.rows, evaluation.margins)
    )


def _write_json_report(path: str | Path | None, payload: Mapping[str, Any]) -> None:
    if not path:
        return
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def cmd_benchmark(args) -> int:
    profile = ParentProfile.from_directory(args.parent_dir)
    observations = _load_observations(args.observations)
    bank = load_evaluator_cases_tsv(args.cases_tsv)
    root = ModuleGenome(prefix_step=int(args.root_prefix), rules=())
    root.validate(profile.route_ids)
    evaluator = _cuda_evaluator_from_args(profile, args)
    report = benchmark_proposers(
        root_genome=root,
        route_ids=profile.route_ids,
        observations=observations,
        panel=bank.rows,
        evaluator=evaluator,
        candidate_budget=args.candidate_budget,
        seed=args.seed,
        work_dir=args.work_dir,
    )
    payload = {
        **report,
        "parent_identity": profile.combined_agent_identity,
        "root_semantic_id": root.semantic_id(),
        "evaluator": evaluator.config_record(),
    }
    _write_json_report(args.report, payload)
    print(json.dumps(payload, sort_keys=True))
    return 2 if bool(args.require_material_improvement) and not bool(report["material_improvement"]) else 0


def cmd_anti_saturation(args) -> int:
    profile = ParentProfile.from_directory(args.parent_dir)
    observations = _load_observations(args.observations)
    raw_bank = load_evaluator_cases_tsv(args.cases_tsv)
    root = ModuleGenome(prefix_step=int(args.root_prefix), rules=())
    root.validate(profile.route_ids)
    evaluator = _cuda_evaluator_from_args(profile, args)
    scored_bank = _root_scored_bank(
        raw_bank,
        root=root,
        evaluator=evaluator,
        work_dir=Path(args.work_dir) / "root_score",
        panel_id=f"anti-saturation/root/{int(args.seed)}",
    )
    report = run_anti_saturation(
        root_genome=root,
        route_ids=profile.route_ids,
        observations=observations,
        case_bank=scored_bank,
        evaluator=evaluator,
        generations=args.generations,
        candidates_per_generation=args.candidates_per_generation,
        screen_size=args.screen_size,
        final_size=args.final_size,
        seed=args.seed,
        work_dir=args.work_dir,
        stagnation_patience=args.stagnation_patience,
    )
    payload = {
        **report,
        "parent_identity": profile.combined_agent_identity,
        "root_semantic_id": root.semantic_id(),
        "root_losses": sum(int(x.baseline_margin) < 0 for x in scored_bank.rows),
        "root_fragile_wins": sum(0 < int(x.baseline_margin) <= int(args.fragile_margin) for x in scored_bank.rows),
        "evaluator": evaluator.config_record(),
    }
    _write_json_report(args.report, payload)
    print(json.dumps(payload, sort_keys=True))
    return 2 if bool(args.require_escalation) and not report["escalation_events"] else 0


def cmd_export_corpus(args) -> int:
    archive = SearchArchive.load(args.archive) if args.archive else None
    report = validate_export(args.corpus)
    if archive is not None:
        manifest = json.loads((Path(args.corpus) / "corpus_manifest.json").read_text())
        if str(manifest.get("base_identity")) != archive.base_identity:
            raise ValueError("corpus/archive BKG4 parent identity mismatch")
    payload = {"export_valid": True, **report}
    if args.report:
        p = Path(args.report); p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, sort_keys=True))
    return 0


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Permanent recursive typed BKG4 edit search")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("init")
    p.add_argument("--parent-dir", required=True)
    p.add_argument("--archive", required=True)
    p.add_argument("--run-id", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--root-prefix", type=int, default=0)
    p.add_argument("--motifs-from")
    p.set_defaults(fn=cmd_init)

    for name, fn in (("validate", cmd_validate), ("resume", cmd_resume)):
        p = sub.add_parser(name); p.add_argument("--archive", required=True); p.set_defaults(fn=fn)

    p = sub.add_parser("propose")
    p.add_argument("--archive", required=True); p.add_argument("--observations", required=True)
    p.add_argument("--count", type=int, default=1024); p.add_argument("--generation", type=int, required=True)
    p.add_argument("--seed", type=int, required=True); p.add_argument("--out", required=True); p.set_defaults(fn=cmd_propose)

    p = sub.add_parser("search")
    p.add_argument("--archive", required=True); p.add_argument("--parent-dir", required=True)
    p.add_argument("--case-bank", required=True); p.add_argument("--observations", required=True)
    p.add_argument("--evaluator", required=True); p.add_argument("--corpus", required=True); p.add_argument("--work-dir", required=True)
    p.add_argument("--rounds", type=int, default=1); p.add_argument("--start-generation", type=int, required=True)
    p.add_argument("--candidates", type=int, default=4096); p.add_argument("--screen-size", type=int, default=128)
    p.add_argument("--final-size", type=int, default=512); p.add_argument("--stage2-budget", type=int, default=512)
    p.add_argument("--seed", type=int, required=True); p.add_argument("--device", type=int, default=0); p.add_argument("--tpb", type=int, default=64)
    p.add_argument("--memory-fraction", type=float, default=0.70); p.add_argument("--cuda-visible-devices")
    p.add_argument("--fragile-margin", type=int, default=5000); p.add_argument("--random-anchor-fraction", type=float, default=0.10)
    p.add_argument("--useful-recall-floor", type=float, default=0.90); p.add_argument("--danger-recall-floor", type=float, default=0.98)
    p.add_argument("--audit-all", action="store_true"); p.add_argument("--report"); p.set_defaults(fn=cmd_search)

    p = sub.add_parser("benchmark")
    p.add_argument("--parent-dir", required=True); p.add_argument("--cases-tsv", required=True)
    p.add_argument("--observations", required=True); p.add_argument("--evaluator", required=True); p.add_argument("--work-dir", required=True)
    p.add_argument("--candidate-budget", type=int, default=64); p.add_argument("--seed", type=int, required=True)
    p.add_argument("--root-prefix", type=int, default=0); p.add_argument("--device", type=int, default=0); p.add_argument("--tpb", type=int, default=64)
    p.add_argument("--memory-fraction", type=float, default=0.70); p.add_argument("--cuda-visible-devices"); p.add_argument("--report")
    p.add_argument("--require-material-improvement", action="store_true"); p.set_defaults(fn=cmd_benchmark)

    p = sub.add_parser("anti-saturation")
    p.add_argument("--parent-dir", required=True); p.add_argument("--cases-tsv", required=True)
    p.add_argument("--observations", required=True); p.add_argument("--evaluator", required=True); p.add_argument("--work-dir", required=True)
    p.add_argument("--generations", type=int, default=8); p.add_argument("--candidates-per-generation", type=int, default=24)
    p.add_argument("--screen-size", type=int, default=8); p.add_argument("--final-size", type=int, default=16)
    p.add_argument("--stagnation-patience", type=int, default=2); p.add_argument("--fragile-margin", type=int, default=5000)
    p.add_argument("--seed", type=int, required=True); p.add_argument("--root-prefix", type=int, default=0)
    p.add_argument("--device", type=int, default=0); p.add_argument("--tpb", type=int, default=64); p.add_argument("--memory-fraction", type=float, default=0.70)
    p.add_argument("--cuda-visible-devices"); p.add_argument("--report"); p.add_argument("--require-escalation", action="store_true")
    p.set_defaults(fn=cmd_anti_saturation)

    p = sub.add_parser("export-corpus")
    p.add_argument("--corpus", required=True); p.add_argument("--archive"); p.add_argument("--report"); p.set_defaults(fn=cmd_export_corpus)
    p = sub.add_parser("validate-corpus")
    p.add_argument("--corpus", required=True); p.add_argument("--archive"); p.add_argument("--report"); p.set_defaults(fn=cmd_export_corpus)
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    return int(args.fn(args))


if __name__ == "__main__":
    raise SystemExit(main())
