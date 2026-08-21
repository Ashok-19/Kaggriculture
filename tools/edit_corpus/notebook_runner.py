from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from tools.bkg4_search.cuda_io import pack_genomes
from tools.bkg4_search.module_codec import ModuleGenome, genome_from_json
from tools.edit_corpus.corpus import canonical_json, pending_replay_shards, select_representative_cases, select_screening_cases, stable_identity
from tools.edit_corpus.cuda_results import load_edit_results
from tools.edit_corpus.kaggle_runtime import (
    consolidate_corpus,
    copy_prior_corpus,
    initialize_empty_corpus,
    source_rows_from_archive,
    upsert_structures,
    validate_corpus,
    write_corpus_manifest,
    commit_day,
)
from tools.edit_corpus.pipeline import (
    build_edit_plan,
    build_label_rows,
    build_screening_rows,
    load_public_context,
    select_screened_candidates,
    stream_replay_shard_to_cases,
    write_cases_tsv,
)

DEFAULT_PROBES = (72, 144, 216, 288, 360, 432, 504, 576, 648)
CORPUS_DATASET = "ashok205/kaggriculture-bonsai-edit-corpus"
CORPUS_TITLE = "Kaggriculture BonsaiEditFormer CUDA Edit Corpus"


def run(cmd: Sequence[str | os.PathLike[str]], *, cwd: str | Path | None = None, timeout: int = 7200) -> subprocess.CompletedProcess[str]:
    argv = [str(x) for x in cmd]
    print("$", " ".join(argv), flush=True)
    p = subprocess.run(argv, cwd=None if cwd is None else str(cwd), text=True, capture_output=True, timeout=timeout)
    if p.returncode != 0:
        raise RuntimeError(f"command failed ({p.returncode}): {' '.join(argv)}\n{p.stdout}\n{p.stderr}")
    if p.stdout.strip():
        print(p.stdout.strip(), flush=True)
    if p.stderr.strip():
        print(p.stderr.strip(), flush=True)
    return p


def discover_cuda_devices(*, max_devices: int = 2) -> tuple[int, ...]:
    """Discover logical CUDA devices without importing a heavyweight ML framework."""
    limit = max(1, int(max_devices))
    try:
        p = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            text=True,
            capture_output=True,
            timeout=10,
        )
        if p.returncode == 0:
            count = sum(1 for line in p.stdout.splitlines() if line.strip())
            if count > 0:
                return tuple(range(min(limit, count)))
    except (OSError, subprocess.SubprocessError):
        pass
    return (0,)


def adaptive_candidates_per_parent(
    *,
    selected_cases: int,
    seed_parent_count: int,
    cap: int,
    target_games: int,
) -> int:
    cases = max(0, int(selected_cases))
    if cases == 0:
        return 0
    generation_parents = max(1, 1 + int(seed_parent_count))
    denom = cases * generation_parents
    wanted = (max(1, int(target_games)) + denom - 1) // denom
    return min(max(1, int(cap)), max(16, int(wanted)))


def locate_dataset_root(preferred: str | Path, marker: str, *, search_root: str | Path = "/kaggle/input") -> Path | None:
    preferred = Path(preferred)
    if (preferred / marker).is_file():
        return preferred
    root = Path(search_root)
    hits = sorted({p.parent for p in root.rglob(marker)}) if root.exists() else []
    if not hits:
        return None
    if len(hits) > 1:
        raise RuntimeError(f"ambiguous Kaggle inputs for {marker}: {hits}")
    return hits[0]


def compile_cuda_tools(support_root: str | Path, bin_dir: str | Path) -> dict[str, Path]:
    support = Path(support_root)
    out = Path(bin_dir)
    out.mkdir(parents=True, exist_ok=True)
    sources = {
        "edit_eval": support / "cuda-engine/src/bkg4_edit_corpus_eval.cu",
        "capture": support / "cuda-engine/src/bkg4_information_state_capture.cu",
    }
    binaries: dict[str, Path] = {}
    for name, source in sources.items():
        if not source.is_file():
            raise FileNotFoundError(source)
        binary = out / name
        run(["nvcc", "-O3", "-std=c++17", "-lineinfo", "-o", binary, source], timeout=1800)
        binaries[name] = binary
    converter_source = support / "cuda-engine/src/replay_to_native_tapes.cpp"
    if not converter_source.is_file():
        raise FileNotFoundError(converter_source)
    converter = out / "replay_convert"
    run([
        "g++", "-O3", "-std=c++17",
        "-I", support / "bonsai-kg/include",
        "-o", converter, converter_source,
    ], timeout=300)
    binaries["replay_convert"] = converter
    return binaries


def _daily_episode_metadata(archive_root: Path, date: str) -> dict[int, dict[int, dict[str, Any]]]:
    path = archive_root / "episodes.parquet"
    if not path.is_file():
        return {}
    cols = ["episode_id", "team_name", "daily_rank", "opponent", "participants_json"]
    try:
        df = pd.read_parquet(path, filters=[("date", "=", str(date))], columns=cols)
    except Exception:
        df = pd.read_parquet(path, columns=["date", *cols])
        df = df[df["date"].astype(str) == str(date)]
    grouped: dict[int, dict[str, Any]] = {}
    for r in df.itertuples(index=False):
        eid = int(r.episode_id)
        g = grouped.setdefault(eid, {"participants": None, "ranks": {}})
        g["ranks"][str(r.team_name)] = int(r.daily_rank)
        if g["participants"] is None:
            try:
                participants = json.loads(str(r.participants_json))
            except Exception:
                participants = None
            if isinstance(participants, list) and len(participants) == 2:
                g["participants"] = [str(participants[0]), str(participants[1])]
    out: dict[int, dict[int, dict[str, Any]]] = {}
    for eid, g in grouped.items():
        participants = g["participants"]
        if not participants:
            continue
        ranks = g["ranks"]
        out[eid] = {
            seat: {
                "team_name": participants[seat],
                "opponent_name": participants[1 - seat],
                "daily_rank": int(ranks.get(participants[seat], 999)),
            }
            for seat in (0, 1)
        }
    return out


def _parent_assets(support_root: Path, support_manifest: Mapping[str, Any]) -> dict[str, Path | int | str]:
    parent = support_root / str(support_manifest["base_parent"])
    identity = json.loads((parent / "identity.json").read_text(encoding="utf-8"))
    route_ids = tuple(int(x) for x in identity["v6_route_pack"]["route_ids"])
    return {
        "root": parent,
        "controller": parent / "controller.bkgc",
        "controller_json": parent / "controller.json",
        "routes": parent / "routes.bkv6",
        "legacy_routes": parent / "legacy_routes.bkv6",
        "base_id": str(identity["combined_agent_identity"]),
        "controller_semantic_hash": str(identity["controller_semantic_hash"]),
        "controller_nodes": int(identity["controller_nodes"]),
        "controller_bb_slots": int(identity["controller_bb_slots"]),
        "strategic_route_slot": int(identity["runtime_reconciler"]["route_override_slot"]),
        "route_ids": route_ids,
    }


def _daily_rank_map(archive_root: Path, date: str) -> dict[str, int]:
    path = archive_root / "top10_history.parquet"
    if not path.is_file():
        return {}
    try:
        df = pd.read_parquet(path, filters=[("date", "=", str(date))], columns=["team_name", "rank"])
    except Exception:
        df = pd.read_parquet(path, columns=["date", "team_name", "rank"])
        df = df[df["date"].astype(str) == str(date)]
    return {str(r.team_name): int(r.rank) for r in df.itertuples(index=False)}


def _evaluate_genomes(
    binary: Path,
    assets: Mapping[str, Any],
    cases_tsv: Path,
    genomes: Sequence[ModuleGenome],
    tmp_dir: Path,
    *,
    chunk_size: int,
    cuda_devices: Sequence[int] | None = None,
    memory_fraction: float = 0.70,
) -> dict[str, list[Any]]:
    """Evaluate one genome set with one long-lived case load per GPU lane.

    Python-side chunking is intentionally retired: the CUDA executable plans and
    reuses its own memory-sized job buffers while keeping cases/tapes resident.
    ``chunk_size`` remains accepted so old generated notebooks resume cleanly.
    """
    if not genomes:
        return {}
    _ = chunk_size
    devices = tuple(int(x) for x in (cuda_devices or discover_cuda_devices(max_devices=2)))
    if not devices:
        devices = (0,)
    lanes = min(len(devices), len(genomes))
    lane_specs: list[tuple[int, int, list[ModuleGenome], Path, Path]] = []
    for lane in range(lanes):
        start = len(genomes) * lane // lanes
        end = len(genomes) * (lane + 1) // lanes
        chunk = list(genomes[start:end])
        gp = tmp_dir / f"genomes_lane{lane}.b4gm"
        rp = tmp_dir / f"results_lane{lane}.b4ec"
        pack_genomes(chunk, gp)
        lane_specs.append((lane, devices[lane], chunk, gp, rp))

    def launch(spec: tuple[int, int, list[ModuleGenome], Path, Path]) -> tuple[list[ModuleGenome], list[Any]]:
        _lane, device, chunk, gp, rp = spec
        run([
            binary,
            "--controller", assets["controller"],
            "--routes", assets["routes"],
            "--legacy-routes", assets["legacy_routes"],
            "--cases", cases_tsv,
            "--genomes", gp,
            "--out", rp,
            "--strategic-route-slot", str(assets["strategic_route_slot"]),
            "--tpb", "64",
            "--device", str(device),
            "--memory-fraction", f"{float(memory_fraction):.6f}",
        ])
        rows = load_edit_results(rp)
        if len(rows) != len(chunk):
            raise RuntimeError("CUDA result genome count mismatch")
        return chunk, rows

    out: dict[str, list[Any]] = {}
    try:
        if lanes == 1:
            completed = [launch(lane_specs[0])]
        else:
            with ThreadPoolExecutor(max_workers=lanes, thread_name_prefix="edit-corpus-gpu") as pool:
                futures = [pool.submit(launch, spec) for spec in lane_specs]
                completed = [future.result() for future in futures]
        for chunk, rows in completed:
            for genome, case_results in zip(chunk, rows):
                out[genome.semantic_id()] = case_results
    finally:
        for _lane, _device, _chunk, gp, rp in lane_specs:
            gp.unlink(missing_ok=True)
            rp.unlink(missing_ok=True)
    return out


def _capture_context(
    binary: Path,
    assets: Mapping[str, Any],
    cases_tsv: Path,
    case_ids: Sequence[str],
    tmp_dir: Path,
    probes: Sequence[int],
    *,
    cuda_device: int = 0,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    probe_path = tmp_dir / "probes.txt"
    r0_path = tmp_dir / "public_r0.bin"
    truth_path = tmp_dir / "hidden_truth_discard.bin"
    probe_path.write_text("\n".join(str(int(x)) for x in probes) + "\n", encoding="utf-8")
    run([
        binary,
        "--controller", assets["controller"],
        "--routes", assets["routes"],
        "--legacy-routes", assets["legacy_routes"],
        "--cases", cases_tsv,
        "--probes", probe_path,
        "--out", r0_path,
        "--truth-out", truth_path,
        "--strategic-route-slot", str(assets["strategic_route_slot"]),
        "--tpb", "64",
        "--device", str(int(cuda_device)),
    ])
    # The capture binary emits hidden opponent holdings only for offline diagnostics.
    # Corpus construction never reads them; delete immediately before parsing R0.
    truth_path.unlink(missing_ok=True)
    contexts, observed = load_public_context(r0_path, case_ids)
    r0_path.unlink(missing_ok=True)
    probe_path.unlink(missing_ok=True)
    return contexts, observed


def _base_structure_row(assets: Mapping[str, Any], support_manifest: Mapping[str, Any]) -> dict[str, Any]:
    structure = json.loads(Path(assets["controller_json"]).read_text(encoding="utf-8"))
    return {
        "base_id": str(assets["base_id"]),
        "combined_agent_identity": str(assets["base_id"]),
        "controller_semantic_hash": str(assets["controller_semantic_hash"]),
        "controller_nodes": int(assets["controller_nodes"]),
        "controller_bb_slots": int(assets["controller_bb_slots"]),
        "controller_structure_json": canonical_json(structure),
        "support_source_commit": str(support_manifest.get("source_commit", "")),
        "bkg4_abi": int(support_manifest["bkg4_abi"]),
        "feature_registry": int(support_manifest["feature_registry"]),
    }


def _safe_case_rows(selected: Sequence[Mapping[str, Any]], contexts: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    ctx_by_id = {str(x["case_id"]): dict(x) for x in contexts}
    rows: list[dict[str, Any]] = []
    for raw in selected:
        row = dict(raw)
        ctx = ctx_by_id[str(row["case_id"])]
        model = json.loads(str(ctx["model_context_json"]))
        model["seat"] = int(row["seat"])
        # Baseline margin remains an exact supervision column on the case/label rows;
        # it is deliberately not injected into the ranker's model-context payload.
        row["model_context_json"] = canonical_json(model)
        row["context_probe_steps"] = ctx["context_probe_steps"]
        row["public_context_width"] = int(ctx["public_context_width"])
        row["public_context_registry_width"] = int(ctx["public_context_registry_width"])
        row["public_context_values"] = ctx["public_context_values"]
        row["prov_tape_semantic_id"] = str(row.pop("tape_semantic_id"))
        row.pop("runtime_tape_path", None)
        rows.append(row)
    return rows


def process_one_day(
    *,
    date_row: Mapping[str, Any],
    archive_root: Path,
    stage_root: Path,
    support_root: Path,
    support_manifest: Mapping[str, Any],
    binaries: Mapping[str, Path],
    tmp_root: Path,
    max_replays: int | None,
    max_cases: int,
    screen_cases: int,
    seed_parent_count: int,
    candidates_per_parent: int,
    eval_chunk_genomes: int,
    replay_converter_workers: int,
    cuda_devices: Sequence[int],
    screen_target_cuda_games: int,
    final_retained_edits: int,
    eval_gpu_memory_fraction: float,
    probes: Sequence[int],
) -> dict[str, Any]:
    date = str(date_row["date"])
    day_tmp = tmp_root / f"day_{date}"
    if day_tmp.exists():
        shutil.rmtree(day_tmp)
    day_tmp.mkdir(parents=True)
    assets = _parent_assets(support_root, support_manifest)
    shard = archive_root / str(date_row["replay_shard"])
    print(f"\n=== EDIT CORPUS {date} {shard.name} ===", flush=True)
    try:
        ranks = _daily_rank_map(archive_root, date)
        episode_meta = _daily_episode_metadata(archive_root, date)
        _, all_cases, _ = stream_replay_shard_to_cases(
            shard,
            date=date,
            out_dir=day_tmp / "tapes",
            daily_rank_by_team=ranks,
            max_replays=max_replays,
            converter_binary=binaries.get("replay_convert"),
            episode_metadata=episode_meta,
            converter_workers=int(replay_converter_workers),
        )
        if not all_cases:
            raise RuntimeError(f"no valid replay-derived cases for {date}")
        all_tsv = day_tmp / "all_cases.tsv"
        write_cases_tsv(all_tsv, all_cases)
        empty = ModuleGenome(prefix_step=216, rules=())
        baseline_results = _evaluate_genomes(
            binaries["edit_eval"], assets, all_tsv, [empty], day_tmp, chunk_size=1,
            cuda_devices=cuda_devices, memory_fraction=float(eval_gpu_memory_fraction),
        )[empty.semantic_id()]
        if len(baseline_results) != len(all_cases):
            raise RuntimeError("baseline result/case mismatch")
        for case, result in zip(all_cases, baseline_results):
            case["baseline_margin"] = int(result.margin)
        selected = select_representative_cases(all_cases, max_cases=min(int(max_cases), len(all_cases)))
        if not selected:
            raise RuntimeError(f"representative case selection empty for {date}")
        selected_tsv = day_tmp / "selected_cases.tsv"
        write_cases_tsv(selected_tsv, selected)
        screening_cases = select_screening_cases(
            selected,
            max_cases=min(int(screen_cases), len(selected)),
        )
        if not screening_cases:
            raise RuntimeError(f"screening case selection empty for {date}")
        screening_tsv = day_tmp / "screening_cases.tsv"
        write_cases_tsv(screening_tsv, screening_cases)
        contexts, observed = _capture_context(
            binaries["capture"], assets, selected_tsv, [str(x["case_id"]) for x in selected], day_tmp, probes,
            cuda_device=int(cuda_devices[0]),
        )
        final_cases = _safe_case_rows(selected, contexts)

        seed = int(hashlib.sha256((date + str(assets["base_id"])).encode()).hexdigest()[:16], 16)
        effective_candidates_per_parent = adaptive_candidates_per_parent(
            selected_cases=len(screening_cases),
            seed_parent_count=int(seed_parent_count),
            cap=int(candidates_per_parent),
            target_games=int(screen_target_cuda_games),
        )
        plan = build_edit_plan(
            str(assets["base_id"]),
            observed,
            seed=seed,
            seed_parent_count=int(seed_parent_count),
            candidates_per_parent=int(effective_candidates_per_parent),
            route_ids=assets["route_ids"],
        )
        genome_by_overlay = {
            str(row["overlay_semantic_id"]): genome_from_json(json.loads(str(row["overlay_genome_json"])))
            for row in plan["parents"]
        }
        screening_genomes = [genome_by_overlay[key] for key in sorted(genome_by_overlay)]
        screening_gpu_games = len(screening_genomes) * len(screening_cases)
        print({
            "cuda_devices": list(cuda_devices),
            "replay_converter_workers": int(replay_converter_workers),
            "candidate_cap": int(candidates_per_parent),
            "effective_candidates_per_parent": int(effective_candidates_per_parent),
            "screening_cases": len(screening_cases),
            "screening_candidates": len(plan["edits"]),
            "screening_gpu_games": int(screening_gpu_games),
        }, flush=True)
        screening_results = _evaluate_genomes(
            binaries["edit_eval"], assets, screening_tsv, screening_genomes, day_tmp, chunk_size=int(eval_chunk_genomes),
            cuda_devices=cuda_devices, memory_fraction=float(eval_gpu_memory_fraction),
        )
        screening_rows = build_screening_rows(
            plan["edits"], screening_cases, screening_results, date=date,
        )
        retained_edits, screening_rows = select_screened_candidates(
            plan["edits"], screening_rows, target_count=int(final_retained_edits),
        )
        if not retained_edits:
            raise RuntimeError(f"screening funnel retained no edits for {date}")
        final_overlay_ids = {
            str(edit[key])
            for edit in retained_edits
            for key in ("parent_overlay_semantic_id", "child_overlay_semantic_id")
        }
        final_genomes = [genome_by_overlay[key] for key in sorted(final_overlay_ids)]
        final_gpu_games = len(final_genomes) * len(selected)
        final_results = _evaluate_genomes(
            binaries["edit_eval"], assets, selected_tsv, final_genomes, day_tmp, chunk_size=int(eval_chunk_genomes),
            cuda_devices=cuda_devices, memory_fraction=float(eval_gpu_memory_fraction),
        )
        labels, _day_targets = build_label_rows(retained_edits, selected, final_results)
        # Semantic edit definitions are date-independent. Global utility targets are
        # recomputed from authoritative full-panel labels during consolidation.
        edit_rows = [dict(x) for x in retained_edits]
        retained_parent_rows = [
            dict(row) for row in plan["parents"]
            if str(row["overlay_semantic_id"]) in final_overlay_ids
        ]

        upsert_structures(stage_root, [_base_structure_row(assets, support_manifest)], retained_parent_rows)
        committed = commit_day(
            stage_root,
            date=date,
            source_sha=str(date_row["replay_shard_sha256"]),
            source_shard=str(date_row["replay_shard"]),
            cases=final_cases,
            edits=edit_rows,
            labels=labels,
            screening=screening_rows,
        )
        if not committed:
            raise RuntimeError(f"unexpected already-committed day inside pending loop: {date}")
        print({
            "date": date,
            "archive_cases": len(all_cases),
            "selected_cases": len(final_cases),
            "screening_cases": len(screening_cases),
            "screened_edits": len(plan["edits"]),
            "retained_edits": len(edit_rows),
            "exact_cuda_labels": len(labels),
            "screening_gpu_games": int(screening_gpu_games),
            "final_gpu_games": int(final_gpu_games),
        }, flush=True)
        return {
            "date": date,
            "cases": len(final_cases),
            "screening_cases": len(screening_cases),
            "screened_edits": len(plan["edits"]),
            "retained_edits": len(edit_rows),
            "edits": len(edit_rows),
            "labels": len(labels),
            "screening_gpu_games": int(screening_gpu_games),
            "final_gpu_games": int(final_gpu_games),
            "gpu_games": int(screening_gpu_games + final_gpu_games),
        }
    finally:
        shutil.rmtree(day_tmp, ignore_errors=True)


def run_builder(
    *,
    archive_root: str | Path,
    support_root: str | Path,
    prior_corpus_root: str | Path | None,
    stage_root: str | Path,
    tmp_root: str | Path,
    max_dates: int | None = None,
    max_replays_per_day: int | None = None,
    max_cases_per_day: int = 512,
    screen_cases_per_day: int = 128,
    seed_parent_count: int = 8,
    candidates_per_parent: int = 4096,
    eval_chunk_genomes: int = 512,
    replay_converter_workers: int = 2,
    max_cuda_devices: int = 2,
    screen_target_cuda_games: int = 4_000_000,
    final_retained_edits_per_day: int = 4_096,
    eval_gpu_memory_fraction: float = 0.70,
    probes: Sequence[int] = DEFAULT_PROBES,
) -> dict[str, Any]:
    archive = Path(archive_root)
    support = Path(support_root)
    stage = Path(stage_root)
    tmp = Path(tmp_root)
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    support_manifest = json.loads((support / "support_manifest.json").read_text(encoding="utf-8"))
    archive_manifest = json.loads((archive / "archive_manifest.json").read_text(encoding="utf-8"))
    if int(max_cases_per_day) <= 0:
        raise ValueError("max_cases_per_day must be positive")
    if int(screen_cases_per_day) <= 0 or int(screen_cases_per_day) > int(max_cases_per_day):
        raise ValueError("screen_cases_per_day must be positive and no larger than max_cases_per_day")
    if int(final_retained_edits_per_day) <= 0:
        raise ValueError("final_retained_edits_per_day must be positive")

    if prior_corpus_root is not None and (Path(prior_corpus_root) / "corpus_manifest.json").is_file():
        prior_manifest = copy_prior_corpus(prior_corpus_root, stage)
        print("validated prior corpus", prior_manifest["manifest_sha256"], flush=True)
    else:
        if stage.exists():
            shutil.rmtree(stage)
        initialize_empty_corpus(stage)
        print("first corpus build: no prior corpus manifest", flush=True)

    source_rows = source_rows_from_archive(archive)
    processed_path = stage / "processed_shards.json"
    processed = json.loads(processed_path.read_text(encoding="utf-8")) if processed_path.exists() else {}
    pending = pending_replay_shards(source_rows, processed)
    if max_dates is not None:
        pending = pending[:max(0, int(max_dates))]
    print("pending replay dates", [x["date"] for x in pending], flush=True)
    binaries = compile_cuda_tools(support, tmp / "bin") if pending else {}
    cuda_devices = discover_cuda_devices(max_devices=max_cuda_devices) if pending else (0,)
    if pending:
        print("CUDA lanes", list(cuda_devices), "replay converter workers", int(replay_converter_workers), flush=True)

    processed_this_run = []
    for row in pending:
        result = process_one_day(
            date_row=row,
            archive_root=archive,
            stage_root=stage,
            support_root=support,
            support_manifest=support_manifest,
            binaries=binaries,
            tmp_root=tmp,
            max_replays=max_replays_per_day,
            max_cases=max_cases_per_day,
            screen_cases=screen_cases_per_day,
            seed_parent_count=seed_parent_count,
            candidates_per_parent=candidates_per_parent,
            eval_chunk_genomes=eval_chunk_genomes,
            replay_converter_workers=replay_converter_workers,
            cuda_devices=cuda_devices,
            screen_target_cuda_games=screen_target_cuda_games,
            final_retained_edits=final_retained_edits_per_day,
            eval_gpu_memory_fraction=eval_gpu_memory_fraction,
            probes=probes,
        )
        processed_this_run.append(result)

    counts = consolidate_corpus(stage)
    dataset_metadata = {
        "title": CORPUS_TITLE,
        "id": CORPUS_DATASET,
        "licenses": [{"name": "CC0-1.0"}],
        "subtitle": "Exact CUDA supervision for generalized BonsaiEditFormer ranking",
        "description": "Incremental replay-derived parent/edit/legal-context corpus. Outcome labels come only from the exact Kaggriculture CUDA engine; team/opponent/replay/seed identifiers are provenance only.",
    }
    (stage / "dataset-metadata.json").write_text(json.dumps(dataset_metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    engine_identity = stable_identity(
        "BKG4-EDIT-CORPUS-CUDA-ENGINE-V1",
        support_manifest["files"]["cuda-engine/src/bkg4_edit_corpus_eval.cu"]["sha256"],
        support_manifest["base_parent_combined_identity"],
        {"fast_math": False, "result_abi": support_manifest["cuda_result_abi"]},
    )
    manifest = write_corpus_manifest(
        stage,
        replay_archive_version=str(archive_manifest.get("manifest_sha256", "unknown")),
        engine_identity=engine_identity,
        generator_version=str(support_manifest["candidate_generator"]),
        edit_grammar_version=str(support_manifest["edit_grammar"]),
    )
    validate_corpus(stage)
    shutil.rmtree(tmp, ignore_errors=True)
    return {
        "processed_this_run": processed_this_run,
        "counts": counts,
        "manifest": manifest,
        "stage_root": str(stage),
    }
