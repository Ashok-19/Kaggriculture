from __future__ import annotations

import csv
import json
import struct
import subprocess
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from tools.bkg4_search.information_state_io import load_r0, registry_manifest
from tools.bkg4_search.module_codec import FEATURES, ModuleGenome, genome_from_json
from tools.edit_corpus.candidates import DEFAULT_ROUTE_IDS, GeneratedEdit, generate_typed_candidates
from tools.edit_corpus.corpus import (
    assert_model_payload_safe,
    build_case_candidates,
    canonical_json,
    derive_case_label,
    derive_edit_labels,
    replay_to_native_tapes,
    stable_identity,
)
from tools.edit_corpus.cuda_results import EditCudaResult


def _parent_id(base_id: str, genome: ModuleGenome) -> str:
    return stable_identity("BKG4-EDIT-CORPUS-PARENT-V1", str(base_id), genome.semantic_id())


def _parent_row(base_id: str, genome: ModuleGenome) -> dict[str, Any]:
    return {
        "parent_id": _parent_id(base_id, genome),
        "base_id": str(base_id),
        "overlay_semantic_id": genome.semantic_id(),
        "overlay_genome_json": canonical_json(genome.canonical()),
        "overlay_prefix_step": int(genome.prefix_step),
        "overlay_rule_count": int(len(genome.rules)),
    }


def _edit_row(base_id: str, generated: GeneratedEdit) -> dict[str, Any]:
    parent = genome_from_json({
        "prefix_step": generated.action_payload.get("parent_prefix_step", 216),
        "rules": generated.action_payload.get("parent_rules", []),
    }) if "parent_rules" in generated.action_payload else None
    payload = dict(generated.action_payload)
    row = {
        "edit_id": str(generated.edit_id),
        "base_id": str(base_id),
        "parent_overlay_semantic_id": str(generated.parent_semantic_id),
        "child_overlay_semantic_id": str(generated.child_semantic_id),
        "operation": str(generated.action.operation),
        "target_node": int(generated.action.target_node),
        "target_kind": str(payload.get("target_kind", "unknown")),
        "target_rule_index": int(payload.get("target_rule_index", -1)),
        "target_condition_index": int(payload.get("target_condition_index", -1)),
        "action_json": canonical_json(payload),
        "child_genome_json": canonical_json(generated.child_genome),
    }
    if parent is not None:
        row["parent_id"] = _parent_id(base_id, parent)
    return row


def _write_replay_stream(shard: Path, sink: Any, max_replays: int | None) -> int:
    """Stream raw replay JSON strings without decoding them into Python objects."""
    import pyarrow.parquet as pq

    count = 0
    sink.write(b"B4RSP01\0")
    pf = pq.ParquetFile(shard)
    for batch in pf.iter_batches(batch_size=32, columns=["episode_id", "replay_json"]):
        ids = batch.column(0)
        payloads = batch.column(1)
        for i in range(len(batch)):
            if max_replays is not None and count >= int(max_replays):
                return count
            episode_id = int(ids[i].as_py())
            replay_json = payloads[i].as_py()
            if not isinstance(replay_json, str):
                continue
            raw = replay_json.encode("utf-8")
            sink.write(struct.pack("<qQ", episode_id, len(raw)))
            sink.write(raw)
            count += 1
    return count


def _write_replay_stream_multi(shard: Path, sinks: Sequence[Any], max_replays: int | None) -> int:
    """Round-robin replay rows across compiled converters without Python JSON decoding."""
    import pyarrow.parquet as pq

    if not sinks:
        raise ValueError("at least one replay converter sink is required")
    for sink in sinks:
        sink.write(b"B4RSP01\0")
    count = 0
    pf = pq.ParquetFile(shard)
    for batch in pf.iter_batches(batch_size=64, columns=["episode_id", "replay_json"], use_threads=True):
        ids = batch.column(0)
        payloads = batch.column(1)
        for i in range(len(batch)):
            if max_replays is not None and count >= int(max_replays):
                return count
            episode_id = int(ids[i].as_py())
            replay_json = payloads[i].as_py()
            if not isinstance(replay_json, str):
                continue
            raw = replay_json.encode("utf-8")
            sink = sinks[count % len(sinks)]
            sink.write(struct.pack("<qQ", episode_id, len(raw)))
            sink.write(raw)
            count += 1
    return count


def _write_replay_spool(shard: Path, spool: Path, max_replays: int | None) -> int:
    """Compatibility helper used only by tests/tools that explicitly need a spool file."""
    with spool.open("wb") as f:
        return _write_replay_stream(shard, f, max_replays)


def _stream_replay_shard_compiled(
    shard: Path,
    *,
    date: str,
    out: Path,
    converter_binary: str | Path,
    episode_metadata: Mapping[int, Mapping[int, Mapping[str, Any]]] | None,
    daily_rank_by_team: Mapping[str, int],
    max_replays: int | None,
    converter_workers: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[tuple[str, int], str]]:
    import pyarrow.parquet as pq

    out.mkdir(parents=True, exist_ok=True)
    row_count = int(pq.ParquetFile(shard).metadata.num_rows)
    replay_limit = row_count if max_replays is None else min(row_count, max(0, int(max_replays)))
    if replay_limit <= 0:
        return [], [], {}
    worker_count = min(max(1, int(converter_workers)), replay_limit)
    procs: list[subprocess.Popen[bytes]] = []
    logs: list[Any] = []
    metadata_paths: list[Path] = []
    commands: list[list[str]] = []
    try:
        for worker in range(worker_count):
            worker_out = out if worker_count == 1 else out / f"worker_{worker}"
            worker_out.mkdir(parents=True, exist_ok=True)
            metadata_path = out.parent / f"converted_tapes_{worker}.tsv"
            log_path = out.parent / f"replay_convert_{worker}.log"
            cmd = [str(converter_binary), "--stdin", "--out-dir", str(worker_out), "--metadata", str(metadata_path)]
            log = log_path.open("w+b")
            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=log, stderr=subprocess.STDOUT)
            if proc.stdin is None:
                raise RuntimeError("compiled replay converter stdin pipe unavailable")
            procs.append(proc)
            logs.append(log)
            metadata_paths.append(metadata_path)
            commands.append(cmd)
        n = _write_replay_stream_multi(shard, [p.stdin for p in procs if p.stdin is not None], max_replays)
        for proc in procs:
            if proc.stdin is not None:
                proc.stdin.close()
                proc.stdin = None
        outputs: list[str] = []
        for worker, (proc, log, cmd) in enumerate(zip(procs, logs, commands)):
            proc.wait(timeout=3600)
            log.flush()
            log.seek(0)
            output = log.read().decode("utf-8", errors="replace")
            outputs.append(output)
            if proc.returncode != 0:
                raise RuntimeError(f"compiled replay converter worker {worker} failed ({proc.returncode}): {' '.join(cmd)}\n{output}")
        for output in outputs:
            if output.strip():
                print(output.strip(), flush=True)
    except Exception:
        for proc in procs:
            if proc.poll() is None:
                proc.kill()
            try:
                proc.wait(timeout=5)
            except Exception:
                pass
        raise
    finally:
        for log in logs:
            try:
                log.close()
            except Exception:
                pass
        for worker in range(len(logs)):
            (out.parent / f"replay_convert_{worker}.log").unlink(missing_ok=True)

    if n <= 0:
        for metadata_path in metadata_paths:
            metadata_path.unlink(missing_ok=True)
        return [], [], {}

    ranks = {str(k): int(v) for k, v in daily_rank_by_team.items()}
    prov = episode_metadata or {}
    tapes: list[dict[str, Any]] = []
    tape_paths: dict[tuple[str, int], str] = {}
    seen_instances: set[tuple[str, int]] = set()
    for metadata_path in metadata_paths:
        with metadata_path.open("r", encoding="utf-8", newline="") as f:
            for raw in csv.DictReader(f, delimiter="\t"):
                episode_id = int(raw["episode_id"])
                seed = int(raw["seed"])
                seat = int(raw["source_seat"])
                tape_id = str(raw["tape_semantic_id"])
                key = (tape_id, seed)
                if key in seen_instances:
                    continue
                seen_instances.add(key)
                seat_meta = dict(prov.get(episode_id, {}).get(seat, {}))
                team = str(seat_meta.get("team_name", f"seat{seat}"))
                opponent = str(seat_meta.get("opponent_name", f"seat{1-seat}"))
                rank = int(seat_meta.get("daily_rank", ranks.get(team, 999)))
                tape = {
                    "source_seat": seat,
                    "tape_semantic_id": tape_id,
                    "prov_episode_id": episode_id,
                    "prov_team_name": team,
                    "prov_opponent_name": opponent,
                    "prov_seed": seed,
                    "prov_date": str(date),
                    "prov_daily_rank": rank,
                    "prov_reward": None,
                    "prov_opponent_reward": None,
                }
                path = str(raw["tape_path"])
                if not Path(path).is_file():
                    raise RuntimeError(f"compiled replay converter did not create tape: {path}")
                tape_paths[key] = path
                tapes.append(tape)
        metadata_path.unlink(missing_ok=True)

    cases = build_case_candidates(tapes)
    for case in cases:
        key = (str(case["tape_semantic_id"]), int(case["prov_seed"]))
        if key not in tape_paths:
            raise RuntimeError(f"missing compiled tape path for case {case['case_id']}")
        case["runtime_tape_path"] = tape_paths[key]
    return tapes, cases, tape_paths


def stream_replay_shard_to_cases(
    shard_path: str | Path,
    *,
    date: str,
    out_dir: str | Path,
    daily_rank_by_team: Mapping[str, int] | None = None,
    max_replays: int | None = None,
    converter_binary: str | Path | None = None,
    episode_metadata: Mapping[int, Mapping[int, Mapping[str, Any]]] | None = None,
    converter_workers: int = 1,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[tuple[str, int], str]]:
    """Stream one replay Parquet shard and materialize only temporary native tapes.

    The archive shard remains compressed on disk. Each replay row is decoded once and
    immediately converted into the native tape representation consumed by the CUDA
    engines. The returned case rows still contain provenance, but model_context_json is
    identity-free and audited by corpus.assert_model_payload_safe.
    """
    import pyarrow.parquet as pq

    shard = Path(shard_path)
    out = Path(out_dir)
    if converter_binary is not None:
        return _stream_replay_shard_compiled(
            shard,
            date=str(date),
            out=out,
            converter_binary=converter_binary,
            episode_metadata=episode_metadata,
            daily_rank_by_team=daily_rank_by_team or {},
            max_replays=max_replays,
            converter_workers=converter_workers,
        )
    out.mkdir(parents=True, exist_ok=True)
    ranks = {str(k): int(v) for k, v in (daily_rank_by_team or {}).items()}
    tapes: list[dict[str, Any]] = []
    tape_paths: dict[tuple[str, int], str] = {}
    seen_instances: set[tuple[str, int]] = set()
    seen_replays = 0
    pf = pq.ParquetFile(shard)
    for batch in pf.iter_batches(batch_size=8, columns=["episode_id", "replay_json"]):
        table = batch.to_pydict()
        for episode_id, replay_json in zip(table["episode_id"], table["replay_json"]):
            if max_replays is not None and seen_replays >= int(max_replays):
                break
            seen_replays += 1
            rows = replay_to_native_tapes(
                str(replay_json),
                {"date": str(date), "episode_id": int(episode_id), "daily_rank_by_team": ranks},
            )
            for tape in rows:
                key = (str(tape["tape_semantic_id"]), int(tape["prov_seed"]))
                if key in seen_instances:
                    continue
                seen_instances.add(key)
                instance_id = stable_identity("KAGGRICULTURE-TAPE-INSTANCE-V1", key[0], key[1])
                path = out / f"{instance_id}.json"
                # The CUDA engine needs only seat/seed from temporary tape metadata.
                # Identity/date/team provenance remains in the compact corpus tables and
                # is intentionally excluded here to keep the tape parser surface minimal.
                payload = {
                    "source": {
                        "seat": int(tape["source_seat"]),
                        "seed": int(tape["prov_seed"]),
                    },
                    "actions": tape["actions"],
                }
                path.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
                tape_paths[key] = str(path)
                tapes.append(dict(tape))
        if max_replays is not None and seen_replays >= int(max_replays):
            break

    cases = build_case_candidates(tapes)
    for case in cases:
        key = (str(case["tape_semantic_id"]), int(case["prov_seed"]))
        if key not in tape_paths:
            raise RuntimeError(f"missing temporary tape path for case {case['case_id']}")
        case["runtime_tape_path"] = tape_paths[key]
    return tapes, cases, tape_paths


def write_cases_tsv(path: str | Path, cases: Sequence[Mapping[str, Any]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for row in cases:
        lines.append("\t".join(map(str, (
            row["runtime_tape_path"], int(row["seat"]), int(row["prov_seed"]),
            0, 1, 0, 0,
        ))))
    p.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def load_public_context(
    r0_path: str | Path,
    case_ids: Sequence[str],
) -> tuple[list[dict[str, Any]], list[dict[str, int | str]]]:
    """Load only the legal/public R0 capture; hidden-truth files are never accepted."""
    cap = load_r0(r0_path)
    if cap.cases != len(case_ids):
        raise ValueError(f"R0 case count mismatch: {cap.cases} != {len(case_ids)}")
    registry = registry_manifest()
    if cap.registry_width != len(registry):
        raise ValueError(f"R0 registry width mismatch: {cap.registry_width} != {len(registry)}")
    contexts: list[dict[str, Any]] = []
    observed: set[tuple[str, int, int]] = set()
    feature_set = set(FEATURES)
    for ci, case_id in enumerate(case_ids):
        values: list[int] = []
        for pi, _step in enumerate(cap.probes):
            frame = cap.frame(ci, pi)
            values.extend(int(x) for x in frame)
            for ri, (name, args) in enumerate(registry):
                if name not in feature_set or len(args) > 1:
                    continue
                idx = int(args[0]) if args else 0
                observed.add((str(name), idx, int(frame[ri])))
        model_context = {
            "context_schema": "bkg4_r0_public_v1",
            "probe_steps": [int(x) for x in cap.probes],
            "frame_width": int(cap.width),
            "registry_width": int(cap.registry_width),
        }
        assert_model_payload_safe(model_context)
        contexts.append({
            "case_id": str(case_id),
            "model_context_json": canonical_json(model_context),
            "context_probe_steps": [int(x) for x in cap.probes],
            "public_context_width": int(cap.width),
            "public_context_registry_width": int(cap.registry_width),
            "public_context_values": values,
        })
    observed_rows = [
        {"feature": feature, "index": index, "value": value}
        for feature, index, value in sorted(observed)
    ]
    return contexts, observed_rows


def build_edit_plan(
    base_id: str,
    context_rows: Iterable[Mapping[str, Any]],
    *,
    seed: int,
    seed_parent_count: int = 4,
    candidates_per_parent: int = 64,
    route_ids: Sequence[int] = DEFAULT_ROUTE_IDS,
) -> dict[str, list[dict[str, Any]]]:
    context = sorted(
        (dict(x) for x in context_rows),
        key=lambda x: (str(x.get("feature", "")), int(x.get("index", 0)), int(x.get("value", 0))),
    )
    base_overlay = ModuleGenome(prefix_step=216, rules=())
    seed_candidates = generate_typed_candidates(
        base_overlay, context, n=max(16, int(seed_parent_count) * 8), seed=int(seed), route_ids=route_ids
    )
    seed_genomes: list[ModuleGenome] = []
    seed_relations: list[GeneratedEdit] = []
    for generated in seed_candidates:
        child = genome_from_json(generated.child_genome)
        if not child.rules:
            continue
        seed_genomes.append(child)
        seed_relations.append(generated)
        if len(seed_genomes) >= int(seed_parent_count):
            break
    if seed_parent_count > 0 and not seed_genomes:
        raise RuntimeError("could not create any non-empty legal seed overlay parent")

    generation_parents = [base_overlay, *seed_genomes]
    generated_all: list[GeneratedEdit] = list(seed_relations)
    for i, parent in enumerate(generation_parents):
        generated_all.extend(generate_typed_candidates(
            parent,
            context,
            n=int(candidates_per_parent),
            seed=int(seed) + 1009 * (i + 1),
            route_ids=route_ids,
        ))

    genomes: dict[str, ModuleGenome] = {base_overlay.semantic_id(): base_overlay}
    for parent in generation_parents:
        genomes[parent.semantic_id()] = parent
    for generated in generated_all:
        genomes[generated.child_semantic_id] = genome_from_json(generated.child_genome)

    parents = [_parent_row(base_id, genomes[k]) for k in sorted(genomes)]
    parent_id_by_overlay = {row["overlay_semantic_id"]: row["parent_id"] for row in parents}
    edits: dict[str, dict[str, Any]] = {}
    for generated in generated_all:
        row = _edit_row(base_id, generated)
        row["parent_id"] = parent_id_by_overlay[generated.parent_semantic_id]
        row["child_parent_id"] = parent_id_by_overlay[generated.child_semantic_id]
        edits.setdefault(row["edit_id"], row)
    return {
        "parents": sorted(parents, key=lambda x: str(x["parent_id"])),
        "edits": sorted(edits.values(), key=lambda x: str(x["edit_id"])),
    }


def build_label_rows(
    edits: Sequence[Mapping[str, Any]],
    cases: Sequence[Mapping[str, Any]],
    results_by_overlay: Mapping[str, Sequence[EditCudaResult]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    labels: list[dict[str, Any]] = []
    targets: list[dict[str, Any]] = []
    for edit in sorted((dict(x) for x in edits), key=lambda x: str(x["edit_id"])):
        parent_results = results_by_overlay[str(edit["parent_overlay_semantic_id"])]
        child_results = results_by_overlay[str(edit["child_overlay_semantic_id"])]
        if len(parent_results) != len(cases) or len(child_results) != len(cases):
            raise ValueError(f"CUDA result/case count mismatch for edit {edit['edit_id']}")
        one: list[dict[str, Any]] = []
        for case, parent, child in zip(cases, parent_results, child_results):
            row = derive_case_label(
                parent.margin,
                child.margin,
                child.action_changes,
                child.activations,
                protected=True,
                parent_action_trace_hash=parent.action_trace_hash,
                candidate_action_trace_hash=child.action_trace_hash,
            )
            row.update({
                "edit_id": str(edit["edit_id"]),
                "case_id": str(case["case_id"]),
                "date": str(case.get("prov_date", "")),
                "parent_action_changes_from_base": int(parent.action_changes),
                "candidate_action_changes_from_base": int(child.action_changes),
                "parent_activations": int(parent.activations),
                "candidate_activations": int(child.activations),
            })
            one.append(row)
            labels.append(row)
        agg = derive_edit_labels(one)
        targets.append({"edit_id": str(edit["edit_id"]), **agg})
    labels.sort(key=lambda x: (str(x["edit_id"]), str(x["case_id"])))
    targets.sort(key=lambda x: str(x["edit_id"]))
    return labels, targets



def build_screening_rows(
    edits: Sequence[Mapping[str, Any]],
    cases: Sequence[Mapping[str, Any]],
    results_by_overlay: Mapping[str, Sequence[EditCudaResult]],
    *,
    date: str = "",
) -> list[dict[str, Any]]:
    """Aggregate exact subpanel outcomes without materializing screening labels."""
    rows: list[dict[str, Any]] = []
    for edit in sorted((dict(x) for x in edits), key=lambda x: str(x["edit_id"])):
        parent_results = results_by_overlay[str(edit["parent_overlay_semantic_id"])]
        child_results = results_by_overlay[str(edit["child_overlay_semantic_id"])]
        if len(parent_results) != len(cases) or len(child_results) != len(cases):
            raise ValueError(f"screening CUDA result/case count mismatch for edit {edit['edit_id']}")
        case_rows: list[dict[str, Any]] = []
        min_delta: int | None = None
        for parent, child in zip(parent_results, child_results):
            case_row = derive_case_label(
                parent.margin,
                child.margin,
                child.action_changes,
                child.activations,
                protected=True,
                parent_action_trace_hash=parent.action_trace_hash,
                candidate_action_trace_hash=child.action_trace_hash,
            )
            delta = int(case_row["margin_delta"])
            min_delta = delta if min_delta is None else min(min_delta, delta)
            case_rows.append(case_row)
        agg = derive_edit_labels(case_rows)
        rows.append({
            "edit_id": str(edit["edit_id"]),
            "date": str(date),
            "screening_case_count": int(len(cases)),
            "screening_losses_before": int(agg["losses_before"]),
            "screening_losses_after": int(agg["losses_after"]),
            "screening_l2w": int(agg["l2w_count"]),
            "screening_w2l": int(agg["w2l_count"]),
            "screening_severe_regressions": int(agg["severe_regression_count"]),
            "screening_fragile_wins_after": int(agg["fragile_wins_after"]),
            "screening_mean_delta": float(agg["mean_margin_delta"]),
            "screening_min_delta": int(min_delta or 0),
            "screening_worst_candidate_margin": int(agg["worst_candidate_margin"]),
            "screening_worst_protected_win_delta": int(agg["worst_protected_win_delta"]),
            "screening_strict_safe": bool(agg["strict_safe"]),
            "screening_exact_noop": bool(agg["exact_noop"]),
            "screening_behavior_changing": bool(agg["behavior_changing"]),
            "screening_action_changes": int(agg["action_change_total"]),
            "screening_activations": int(agg["activation_total"]),
        })
    return rows


def _screen_exploitation_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        int(row["screening_losses_after"]),
        int(row["screening_w2l"]),
        int(row["screening_severe_regressions"]),
        int(row["screening_fragile_wins_after"]),
        -int(row["screening_worst_candidate_margin"]),
        -float(row["screening_mean_delta"]),
        1 if bool(row["screening_exact_noop"]) else 0,
        str(row["edit_id"]),
    )


def _screen_round_robin(
    ranked: Sequence[dict[str, Any]],
    count: int,
    *,
    selected: set[str],
    group_fields: Sequence[str],
) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in ranked:
        edit_id = str(row["edit_id"])
        if edit_id in selected:
            continue
        key = tuple(row[field] for field in group_fields)
        groups.setdefault(key, []).append(row)
    keys = list(groups)
    out: list[dict[str, Any]] = []
    depth = 0
    while len(out) < max(0, int(count)):
        added = False
        for key in keys:
            items = groups[key]
            if depth >= len(items):
                continue
            row = items[depth]
            edit_id = str(row["edit_id"])
            if edit_id in selected:
                continue
            selected.add(edit_id)
            out.append(row)
            added = True
            if len(out) >= int(count):
                break
        if not added:
            break
        depth += 1
    return out


def select_screened_candidates(
    edits: Sequence[Mapping[str, Any]],
    screening_rows: Sequence[Mapping[str, Any]],
    *,
    target_count: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Deterministically retain useful positives, boundary cases, negatives, and diversity."""
    edit_by_id = {str(x["edit_id"]): dict(x) for x in edits}
    if len(edit_by_id) != len(edits):
        raise ValueError("duplicate edit_id in screening candidates")
    screen_by_id = {str(x["edit_id"]): dict(x) for x in screening_rows}
    if set(screen_by_id) != set(edit_by_id):
        missing = sorted(set(edit_by_id) - set(screen_by_id))
        extra = sorted(set(screen_by_id) - set(edit_by_id))
        raise ValueError(f"screening/edit identity mismatch missing={missing[:3]} extra={extra[:3]}")
    target = min(max(0, int(target_count)), len(edit_by_id))
    annotated: list[dict[str, Any]] = []
    for edit_id in sorted(edit_by_id):
        edit = edit_by_id[edit_id]
        row = dict(screen_by_id[edit_id])
        row["screening_parent_overlay_semantic_id"] = str(edit["parent_overlay_semantic_id"])
        row["screening_operation"] = str(edit.get("operation", "unknown"))
        row["screening_target_kind"] = str(edit.get("target_kind", "unknown"))
        row["screening_target_node"] = int(edit.get("target_node", -1))
        row["screening_retained"] = False
        row["screening_bucket"] = "rejected"
        annotated.append(row)

    ranked = sorted(annotated, key=_screen_exploitation_key)
    for rank, row in enumerate(ranked, 1):
        row["screening_rank"] = int(rank)

    selected: set[str] = set()
    selected_rows: list[dict[str, Any]] = []
    exploitation_n = (target * 2 + 4) // 5
    boundary_n = target // 5
    negative_n = target // 5
    diversity_n = target - exploitation_n - boundary_n - negative_n
    balance_fields = ("screening_parent_overlay_semantic_id", "screening_operation")

    exploitation = [
        row for row in ranked
        if bool(row["screening_strict_safe"])
        and bool(row["screening_behavior_changing"])
        and not bool(row["screening_exact_noop"])
    ]
    for row in _screen_round_robin(exploitation, exploitation_n, selected=selected, group_fields=balance_fields):
        row["screening_bucket"] = "exploitation"
        selected_rows.append(row)

    boundary = sorted(
        (row for row in annotated if str(row["edit_id"]) not in selected and bool(row["screening_behavior_changing"])),
        key=lambda row: (
            abs(float(row["screening_mean_delta"])),
            int(row["screening_w2l"]),
            int(row["screening_severe_regressions"]),
            _screen_exploitation_key(row),
        ),
    )
    for row in _screen_round_robin(boundary, boundary_n, selected=selected, group_fields=balance_fields):
        row["screening_bucket"] = "boundary"
        selected_rows.append(row)

    negative = [
        row for row in annotated
        if str(row["edit_id"]) not in selected
        and (
            int(row["screening_w2l"]) > 0
            or int(row["screening_severe_regressions"]) > 0
            or int(row["screening_losses_after"]) > int(row["screening_losses_before"])
            or float(row["screening_mean_delta"]) < 0.0
            or bool(row["screening_exact_noop"])
        )
    ]
    negative.sort(key=lambda row: (
        0 if (int(row["screening_w2l"]) > 0 or int(row["screening_severe_regressions"]) > 0) else 1,
        abs(float(row["screening_mean_delta"])),
        _screen_exploitation_key(row),
    ))
    for row in _screen_round_robin(
        negative,
        negative_n,
        selected=selected,
        group_fields=("screening_parent_overlay_semantic_id", "screening_operation", "screening_exact_noop"),
    ):
        row["screening_bucket"] = "negative"
        selected_rows.append(row)

    diverse = sorted(
        (row for row in annotated if str(row["edit_id"]) not in selected),
        key=lambda row: (
            str(row["screening_parent_overlay_semantic_id"]),
            str(row["screening_operation"]),
            str(row["screening_target_kind"]),
            int(row["screening_target_node"]),
            _screen_exploitation_key(row),
        ),
    )
    for row in _screen_round_robin(
        diverse,
        diversity_n,
        selected=selected,
        group_fields=(
            "screening_parent_overlay_semantic_id",
            "screening_operation",
            "screening_target_kind",
            "screening_target_node",
        ),
    ):
        row["screening_bucket"] = "diversity"
        selected_rows.append(row)

    if len(selected) < target:
        remaining = sorted(
            (row for row in ranked if str(row["edit_id"]) not in selected),
            key=_screen_exploitation_key,
        )
        for row in _screen_round_robin(remaining, target - len(selected), selected=selected, group_fields=balance_fields):
            row["screening_bucket"] = "diversity"
            selected_rows.append(row)

    for row in selected_rows:
        row["screening_retained"] = True
    kept = [edit_by_id[edit_id] for edit_id in sorted(selected)]
    return kept, sorted(annotated, key=lambda row: str(row["edit_id"]))
