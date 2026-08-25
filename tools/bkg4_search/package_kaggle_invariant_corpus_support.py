#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tarfile
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from tools.bkg4_search.generate_invariant_specialists import combine_offline_banks
from tools.bkg4_search.invariant_factory import load_offline_specialist_bank, split_specialist_panels
from tools.bkg4_search.invariant_module import (
    AbstractAction,
    GuardAtom,
    InvariantRule,
    InvariantStrategicModule,
    LiftedGuard,
    ModuleSketch,
    pack_file,
)
from tools.bkg4_search.kaggle_invariant_corpus import sha256_file
from tools.bkg4_search.invariant_specialists import FINAL_CORPUS_TIER_FRACTIONS

ROOT = Path(__file__).resolve().parents[2]
FROZEN_SOURCE_COMMIT = "c012c2c6c1de18fc250cf9e5425c3aea1f1d273c"
SUPPORT_SCHEMA_VERSION = 1
SUPPORT_PACKAGE_VERSION = "2026-08-25-v4"
SUPPORT_DATASET = "ashok205/kaggriculture-b4im-elite-corpus-support"
SUPPORT_TITLE = "Kaggriculture B4IM Elite Specialist Corpus Support"
CURRENT_PARENT = "bonsai-kg/integrations/bkg4_champion/champions/w33_s172_expansion_v1"
VALIDATED_CORPUS = ROOT / "scratch/b4im_elite_factory/final_hardening_20260824/rebuild_lineage_v2_preserved_final/corpus/invariant_specialists_exact_cuda.jsonl"
FRESH_ARCHIVE_DATASET = "ashok205/kaggriculture-top10-replay-archive"
FRESH_ARCHIVE_DATE = "2026-08-23"
FRESH_ARCHIVE_ROOT = ROOT / "scratch/top10_archive_20260824"
FRESH_CURRENT_BANK = ROOT / "scratch/bkg4_search/current_archive_aug23/cases.jsonl"
SMOKE_BANK_SOURCE = ROOT / "scratch/bkg4_search/current_shifted_aug20/current140_cases.jsonl"

BANK_SOURCES = {
    "current_top": FRESH_CURRENT_BANK,
    "historical_top": ROOT / "scratch/bkg4_search/current_shifted_aug20/historical336_cases.jsonl",
    "hard_current": ROOT / "scratch/bkg4_search/hard_capability_aug20/current_ryo_bank/cases.jsonl",
}


def _materialize_fresh_current_bank() -> None:
    manifest = json.loads((FRESH_ARCHIVE_ROOT / "archive_manifest.json").read_text(encoding="utf-8"))
    shard = FRESH_ARCHIVE_ROOT / f"replays_{FRESH_ARCHIVE_DATE}.parquet"
    provenance = FRESH_CURRENT_BANK.with_suffix(".manifest.json")
    expected = {
        "archive_dataset": FRESH_ARCHIVE_DATASET,
        "archive_manifest_sha256": manifest["manifest_sha256"],
        "archive_date": FRESH_ARCHIVE_DATE,
        "source_shard_sha256": manifest["files"][shard.name]["sha256"],
        "max_replays": 192,
    }
    if FRESH_CURRENT_BANK.is_file() and provenance.is_file():
        if json.loads(provenance.read_text(encoding="utf-8")) == expected:
            return
    from tools.edit_corpus.pipeline import stream_replay_shard_to_cases

    tape_dir = FRESH_CURRENT_BANK.parent / "tapes"
    if FRESH_CURRENT_BANK.parent.exists():
        shutil.rmtree(FRESH_CURRENT_BANK.parent)
    tapes, cases, _ = stream_replay_shard_to_cases(
        shard, date=FRESH_ARCHIVE_DATE, out_dir=tape_dir, max_replays=192,
    )
    if len(cases) < 640 or len(tapes) < 320:
        raise RuntimeError(f"fresh archive did not yield panel capacity: tapes={len(tapes)} cases={len(cases)}")
    rows = []
    for case in cases:
        rows.append({
            "case_id": case["case_id"], "tape": case["runtime_tape_path"],
            "seat": int(case["seat"]), "seed": int(case["prov_seed"]),
            "team": str(case["prov_team_name"]), "role": f"current_archive_{FRESH_ARCHIVE_DATE}",
            "protected": 1, "own_history": 0, "archive_date": FRESH_ARCHIVE_DATE,
        })
    FRESH_CURRENT_BANK.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    provenance.write_text(json.dumps(expected, indent=2, sort_keys=True) + "\n", encoding="utf-8")

REQUIRED_PATHS = tuple(sorted((
    "bonsai-kg/cuda/kag_adapter.cuh",
    "bonsai-kg/cuda/kag_adapter_v4.cuh",
    "bonsai-kg/include/bkg4_invariant_module.hpp",
    "bonsai-kg/include/bonsai_kg.hpp",
    "bonsai-kg/include/bonsai_kg_abi.hpp",
    "bonsai-kg/include/bonsai_kg_io.hpp",
    "bonsai-kg/include/bonsai_kg_v4.hpp",
    "bonsai-kg/include/bonsai_kg_v4_abi.hpp",
    "bonsai-kg/include/bonsai_kg_v4_io.hpp",
    "bonsai-kg/integrations/bkg4_champion/assets/identity.hpp",
    "bonsai-kg/integrations/bkg4_champion/cuda_compat.cuh",
    "bonsai-kg/integrations/bkg4_champion/extended_route_pack.hpp",
    f"{CURRENT_PARENT}/controller.bkgc",
    f"{CURRENT_PARENT}/controller.json",
    f"{CURRENT_PARENT}/controller.manifest.json",
    f"{CURRENT_PARENT}/identity.hpp",
    f"{CURRENT_PARENT}/identity.json",
    f"{CURRENT_PARENT}/legacy_routes.bkv6",
    f"{CURRENT_PARENT}/routes.bkv6",
    "bonsai-kg/integrations/stateful_v6/assets/identity.hpp",
    "bonsai-kg/integrations/stateful_v6/cuda_runtime.cuh",
    "bonsai-kg/integrations/stateful_v6/engine_exploits.cuh",
    "bonsai-kg/integrations/stateful_v6/route_pack.hpp",
    "bonsai-kg/python/bonsai_kg.py",
    "bonsai-kg/python/bonsai_kg_v4.py",
    "cuda-engine/Makefile",
    "cuda-engine/include/observation.cuh",
    "cuda-engine/include/pyrandom.cuh",
    "cuda-engine/include/runtime.cuh",
    "cuda-engine/include/sim.cuh",
    "cuda-engine/include/tape_io.hpp",
    "cuda-engine/src/bkg4_invariant_edit_eval.cu",
    "tools/bkg4_search/__init__.py",
    "tools/bkg4_search/bonsai_editformer/__init__.py",
    "tools/bkg4_search/bonsai_editformer/actor_grammar.py",
    "tools/bkg4_search/bonsai_editformer/invariant_data.py",
    "tools/bkg4_search/edit_search/__init__.py",
    "tools/bkg4_search/edit_search/fitness.py",
    "tools/bkg4_search/edit_search/panels.py",
    "tools/bkg4_search/edit_search/schema.py",
    "tools/bkg4_search/generate_invariant_specialists.py",
    "tools/bkg4_search/information_state_io.py",
    "tools/bkg4_search/invariant_evaluator.py",
    "tools/bkg4_search/invariant_factory.py",
    "tools/bkg4_search/invariant_groups.py",
    "tools/bkg4_search/invariant_lineage.py",
    "tools/bkg4_search/invariant_module.py",
    "tools/bkg4_search/invariant_schema.py",
    "tools/bkg4_search/invariant_search.py",
    "tools/bkg4_search/invariant_specialists.py",
    "tools/bkg4_search/kaggle_invariant_corpus.py",
    "tools/bkg4_search/kaggle_invariant_corpus_runner.py",
    "tools/bkg4_search/merge_invariant_corpus_shards.py",
    "tools/bkg4_search/module_codec.py",
    "tools/edit_corpus/__init__.py",
    "tools/edit_corpus/corpus.py",
)))


def dataset_metadata() -> dict[str, Any]:
    return {
        "title": SUPPORT_TITLE,
        "id": SUPPORT_DATASET,
        "licenses": [{"name": "CC0-1.0"}],
        "subtitle": "Portable exact-CUDA B4IM1/B4IC1 dual-T4 corpus runtime",
        "description": "Private support bundle for the frozen Kaggriculture B4IM elite-specialist factory. Includes the minimal runtime source, validated BKG4 parent, portable discovery banks and tapes, and exact dual-T4 smoke gate.",
    }


def _git_head() -> str:
    proc = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True, check=True)
    return proc.stdout.strip()


def _copy(src: Path, dst: Path) -> None:
    if not src.is_file():
        raise FileNotFoundError(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _portable_banks(stage: Path) -> tuple[list[str], dict[str, Any], int]:
    tape_dest = stage / "tapes"
    tape_dest.mkdir(parents=True, exist_ok=True)
    bank_paths: list[str] = []
    summaries: dict[str, Any] = {}
    copied_tapes: dict[str, str] = {}
    for kind, source in BANK_SOURCES.items():
        if not source.is_file():
            raise FileNotFoundError(source)
        target = stage / "banks" / f"{kind}.jsonl"
        target.parent.mkdir(parents=True, exist_ok=True)
        output = []
        roles: Counter[str] = Counter()
        for line in source.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            tape_value = row.get("tape_path", row.get("tape"))
            tape = Path(str(tape_value))
            if not tape.is_absolute():
                candidates = ((source.parent / tape).resolve(), (ROOT / tape).resolve())
                tape = next((candidate for candidate in candidates if candidate.exists()), candidates[-1])
            if not tape.is_file():
                raise FileNotFoundError(f"bank tape is missing: {tape}")
            row.setdefault("case_id", hashlib.sha256(
                f"{tape}\0{int(row['seat'])}\0{int(row['seed'])}\0{kind}".encode()
            ).hexdigest())
            digest = sha256_file(tape)
            name = f"{digest}.json"
            previous = copied_tapes.setdefault(digest, name)
            if previous != name:
                raise ValueError("tape digest naming conflict")
            destination = tape_dest / name
            if not destination.exists():
                _copy(tape, destination)
            row.pop("tape", None)
            row["tape_path"] = f"../tapes/{name}"
            roles[str(row.get("role", "offline"))] += 1
            output.append(row)
        target.write_text("".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in output), encoding="utf-8")
        bank_paths.append(target.relative_to(stage).as_posix())
        summaries[kind] = {"cases": len(output), "source_roles": dict(sorted(roles.items()))}
    return bank_paths, summaries, len(copied_tapes)


def smoke_modules() -> tuple[InvariantStrategicModule, ...]:
    return (
        InvariantStrategicModule(),
        InvariantStrategicModule((InvariantRule(ModuleSketch("route", "INSTANT", "ROUTE_MODE"), "GLOBAL", LiftedGuard(), AbstractAction("ROUTE_OVERRIDE", route_id=70)),)),
        InvariantStrategicModule((InvariantRule(ModuleSketch("route", "COMMITTED", "ROUTE_MODE"), "GLOBAL", LiftedGuard((GuardAtom("OWN_MONEY", "GE", 0),)), AbstractAction("COMMITTED_ROUTE", route_id=70), LiftedGuard((GuardAtom("OWN_MONEY", "LT", 0),))),)),
        InvariantStrategicModule((InvariantRule(ModuleSketch("maintenance", "INSTANT", "REDUCE_MAINTENANCE_BACKLOG"), "EACH_OWN_UNIT", LiftedGuard(), AbstractAction("UNIT_REWRITE", match_op=17, unit_op=0)),)),
        InvariantStrategicModule((InvariantRule(ModuleSketch("market", "INSTANT", "CREATE_FINANCING_RUNWAY"), "GLOBAL", LiftedGuard(), AbstractAction("MARKET_INTENT", market_op=1, item=0, quantity=1, authority=4, priority=250)),)),
    )


def _write_smoke_fixture(stage: Path, bank_paths: Sequence[str]) -> dict[str, Any]:
    fixture = stage / "smoke"
    fixture.mkdir(parents=True, exist_ok=True)
    modules = smoke_modules()
    (fixture / "candidates.b4im").write_bytes(pack_file(modules))
    del bank_paths
    expected_source = ROOT / "tools/bkg4_search/fixtures/b4im_t4_smoke_expected.json"
    if not expected_source.is_file():
        raise FileNotFoundError(f"committed exact smoke expectations are missing: {expected_source}")
    expected = json.loads(expected_source.read_text(encoding="utf-8"))
    smoke_bank = load_offline_specialist_bank(SMOKE_BANK_SOURCE, cohort_kind="current_top")
    by_id = {case.case_id: case for case in smoke_bank.cases}
    cases = tuple(by_id[case_id] for case_id in expected["case_ids"])
    case_lines = []
    for case in cases:
        fields = case.evaluator_tsv().split("\t")
        tape = Path(case.tape_path)
        name = f"{sha256_file(tape)}.json"
        _copy(tape, stage / "tapes" / name)
        fields[0] = f"../tapes/{name}"
        case_lines.append("\t".join(fields))
    (fixture / "cases.tsv").write_text("\n".join(case_lines) + "\n", encoding="utf-8")
    template = {
        "schema_version": 1,
        "candidate_abi": "B4IM1/v1",
        "evaluator_abi": "B4IC1/v1",
        "candidate_kinds": ["EMPTY", "ROUTE_OVERRIDE", "COMMITTED_ROUTE", "UNIT_REWRITE_EACH_OWN_UNIT", "MARKET_INTENT"],
        "semantic_ids": [module.semantic_id() for module in modules],
        "case_ids": [case.case_id for case in cases],
        "expected_records": None,
    }
    for key in ("candidate_kinds", "semantic_ids"):
        if expected.get(key) != template[key]:
            raise ValueError(f"exact smoke fixture {key} does not match packaged candidates")
    if len(expected.get("expected_records", [])) != len(modules):
        raise ValueError("exact smoke fixture candidate dimensions are invalid")
    (fixture / "expected.json").write_text(json.dumps(expected, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"candidates": len(modules), "cases": len(cases), "kinds": template["candidate_kinds"], "exact_fields": ["own_money", "opp_money", "action_trace_hash", "activations", "action_changes", "flags"]}


def _validate_portable_banks(stage: Path, bank_paths: Sequence[str]) -> dict[str, Any]:
    loaded = [load_offline_specialist_bank(stage / path, cohort_kind=Path(path).stem) for path in bank_paths]
    bank = combine_offline_banks(loaded)
    for case in bank.cases:
        if not Path(case.tape_path).is_file():
            raise FileNotFoundError(case.tape_path)
    screen, label = split_specialist_panels(bank.cases, screen_cases=128, label_cases=512, seed=20260824)
    return {
        "cases": len(bank.cases),
        "tape_groups": len({case.tape_path for case in bank.cases}),
        "cohorts": len(bank.cohort_display),
        "source_roles": list(bank.source_roles),
        "forbidden_roles": [],
        "screen_cases": len(screen),
        "label_cases": len(label),
        "tape_disjoint": not bool({case.tape_path for case in screen} & {case.tape_path for case in label}),
    }


def _inventory(stage: Path) -> dict[str, Any]:
    files = {}
    for path in sorted(p for p in stage.rglob("*") if p.is_file()):
        rel = path.relative_to(stage).as_posix()
        files[rel] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    return files


def build_support_dataset(output: str | Path, *, force: bool = False) -> dict[str, Any]:
    _materialize_fresh_current_bank()
    output = Path(output)
    if not output.is_absolute():
        output = ROOT / output
    if output.exists():
        if not force:
            raise FileExistsError(output)
        shutil.rmtree(output)
    output.mkdir(parents=True)
    with tempfile.TemporaryDirectory(dir=output.parent, prefix="b4im-support-") as temp:
        stage = Path(temp) / "support_bundle"
        for rel in REQUIRED_PATHS:
            _copy(ROOT / rel, stage / rel)
        bank_paths, bank_sources, tape_count = _portable_banks(stage)
        bank_validation = _validate_portable_banks(stage, bank_paths)
        smoke = _write_smoke_fixture(stage, bank_paths)
        files = _inventory(stage)
        bundle = output / "support_bundle.tar"
        with tarfile.open(bundle, "w", format=tarfile.PAX_FORMAT) as archive:
            for rel in sorted(files):
                src = stage / rel
                info = archive.gettarinfo(str(src), arcname=rel)
                info.uid = info.gid = 0
                info.uname = info.gname = ""
                info.mtime = 0
                with src.open("rb") as handle:
                    archive.addfile(info, handle)
    identity = json.loads((ROOT / CURRENT_PARENT / "identity.json").read_text(encoding="utf-8"))
    manifest = {
        "schema_version": SUPPORT_SCHEMA_VERSION,
        "support_package_version": SUPPORT_PACKAGE_VERSION,
        "source_commit": FROZEN_SOURCE_COMMIT,
        "implementation_commit": _git_head(),
        "generator": {"entrypoint": "python3 -m tools.bkg4_search.generate_invariant_specialists", "identity": "bkg4_elite_specialist_factory_report_v1"},
        "candidate_abi": "B4IM1/v1",
        "result_abi": "B4IC1/v1",
        "bkg4_abi": 4,
        "feature_registry": 1,
        "fast_math": False,
        "selection": {
            "screen_cases": 128, "label_cases": 512,
            "policy": "specialist_quality_diversity_v1",
            "tier_fractions": FINAL_CORPUS_TIER_FRACTIONS,
            "semantic_dedup": True, "skeleton_round_robin": True,
        },
        "parent": {"path": CURRENT_PARENT, "canonical_identity": identity["combined_agent_identity"], "controller_semantic_hash": identity["controller_semantic_hash"], "route_ids": identity["v6_route_pack"]["route_ids"], "route_count": identity["v6_route_pack"]["route_count"]},
        "banks": bank_paths,
        "fresh_archive": {
            "dataset": FRESH_ARCHIVE_DATASET,
            "date": FRESH_ARCHIVE_DATE,
            "manifest_sha256": json.loads((FRESH_ARCHIVE_ROOT / "archive_manifest.json").read_text(encoding="utf-8"))["manifest_sha256"],
        },
        "bank_sources": bank_sources,
        "bank_validation": bank_validation,
        "packaged_tape_count": tape_count,
        "cuda": {"source": "cuda-engine/src/bkg4_invariant_edit_eval.cu", "build_target": "bkg4-invariant-edit-eval", "required_architecture": "sm_75", "fast_math": False},
        "smoke_fixture": smoke,
        "files": files,
        "paths": sorted(files),
        "bundle_file": bundle.name,
        "bundle_bytes": bundle.stat().st_size,
        "bundle_sha256": sha256_file(bundle),
    }
    (output / "support_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output / "dataset-metadata.json").write_text(json.dumps(dataset_metadata(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the private portable B4IM Kaggle support dataset")
    parser.add_argument("--output", type=Path, default=Path("scratch/kaggriculture_b4im_elite_corpus_support"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    print(json.dumps(build_support_dataset(args.output, force=args.force), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
