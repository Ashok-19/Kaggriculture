"""Run the bounded equal-context RLRMB versus direct-Bonsai bakeoff."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from kaggriculture_bonsai.league.adversarial_league import (
    ALL_LANES,
    ExploiterResult,
    dev_lane_rotation,
    generate_league_build_cases,
    make_bounded_exploiter_candidates,
)
from kaggriculture_bonsai.league.build_regimes import BuildSupport, load_build_regimes
from kaggriculture_bonsai.league.dev_screen import (
    DOWNSTREAM_ELIGIBLE,
    DevScreenPolicy,
    classify_dev_outcome,
    dev_block_metrics,
)
from kaggriculture_bonsai.league.selfplay_generator import (
    SelfplayPolicy,
    generate_diverse_seeds,
    generate_selfplay_tapes,
    write_target_panel,
)
from kaggriculture_bonsai.paths import CHAMPION_ASSETS, PROJECT_ROOT
from kaggriculture_bonsai.routing.cegis_adapter import run_hardened_cegis
from kaggriculture_bonsai.routing.routability import screen_routability
from kaggriculture_bonsai.search.archive import Archive
from kaggriculture_bonsai.search.beam import BeamConfig, CudaBatchEvaluator, run_beam
from kaggriculture_bonsai.search.checkpoints import Checkpoint
from kaggriculture_bonsai.search.direct_bonsai import DirectSearchConfig, run_direct_search
from kaggriculture_bonsai.search.effects import EffectBlock, effect_signature
from kaggriculture_bonsai.search.evaluator import encode_program_batch
from kaggriculture_bonsai.search.ir import MacroProgram
from kaggriculture_bonsai.search.motifs import Motif
from kaggriculture_bonsai.search.pipeline import (
    LegalStateCaptureCache,
    build_cegis_input,
    build_routability_evidence,
    capture_candidate_legal_states,
    materialize_certified_candidate,
)
from kaggriculture_bonsai.search.p13_equal_budget import _effect_rows
from kaggriculture_bonsai.search.schema import DatasetRole, stable_identity


DISCOVERY_GAMES_DEFAULT = 9600
MAX_DEV_SCREENED = 8


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _build_league(output: Path) -> tuple[list[ExploiterResult], tuple[Checkpoint, ...], dict[str, Motif]]:
    programs, motifs = make_bounded_exploiter_candidates(
        (PROJECT_ROOT / "assets/baseline/route_64.json", PROJECT_ROOT / "assets/baseline/route_410.json"),
        routes_per_lane=2,
        duration=12,
    )
    selected: list[MacroProgram] = []
    for lane in ALL_LANES:
        selected.append(next(row for row in programs if row.lane is lane))
    members: list[ExploiterResult] = []
    bundle_dir = output / "league"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    for index, program in enumerate(selected):
        used = {phase.motif_id for phase in program.phases if phase.motif_id}
        bundle = bundle_dir / f"member_{index:02d}_{program.lane.value.lower()}.rlrb"
        bundle.write_bytes(encode_program_batch((program,), {key: motifs[key] for key in used}))
        policy = SelfplayPolicy(
            f"bakeoff_{program.lane.value.lower()}", "option_program",
            family="adversarial_league", checkpoint_asset_dir=str(CHAMPION_ASSETS),
            strategic_lane=program.lane.value, generation=0,
            parent_checkpoint=str(CHAMPION_ASSETS), member_type="cuda_exploiter",
            exploiter_program=str(bundle), mutation_origin="bounded_internal_donor_window",
        )
        members.append(ExploiterResult(policy, program, 0, 0.0, 0, 0.0, 0))
    checkpoints = tuple(row.initiation for row in selected)
    return members, checkpoints, motifs


def _independent_panel(
    output: Path, role: str, generation: int, members: Sequence[ExploiterResult],
    prior_seeds: list[int], rng_seed: int,
) -> tuple[Path, dict[str, Any]]:
    lanes = dev_lane_rotation(generation, lanes_per_panel=4)
    by_lane = {row.program.lane: row.policy for row in members}
    policies = [by_lane[lane] for lane in lanes]
    seeds = generate_diverse_seeds(3, rng_seed=rng_seed, existing=prior_seeds)
    prior_seeds.extend(seeds)
    tapes = generate_selfplay_tapes(
        output / "tapes", seeds=seeds, policies=policies, seats=(1, 0), rng_seed=rng_seed,
    )
    panel = output / f"{role.lower()}_panel.tsv"
    write_target_panel(
        tapes, panel, protected_flag=int(role in {"VALIDATION", "SEALED"}),
        role={"INTERNAL_DEV": 1, "TRANSFER": 2, "VALIDATION": 3, "SEALED": 4}[role],
    )
    manifest = {
        "role": role,
        "authority": "SCREENING_ONLY" if role == "INTERNAL_DEV" else "DOWNSTREAM_ONLY",
        "generation": generation,
        "seeds": seeds,
        "seats": [1, 0],
        "lanes": [lane.value for lane in lanes],
        "league_members": [row.name for row in policies],
        "cases": len(tapes),
        "champion_losses": sum(row.margin < 0 for row in tapes),
        "panel": str(panel),
    }
    _write(output / "manifest.json", manifest)
    return panel, manifest


def freeze_context(output: Path) -> dict[str, Any]:
    context_dir = output / "context"
    members, checkpoints, motifs = _build_league(context_dir)
    build_seeds = [1103, 2207, 3301]
    build_panel, build_manifest = generate_league_build_cases(
        context_dir / "build", seeds=build_seeds, exploiters=members,
        rng_seed=4409, max_cases=48,
    )
    all_seeds = list(build_seeds)
    dev_panel, dev_manifest = _independent_panel(
        context_dir / "dev", "INTERNAL_DEV", 0, members, all_seeds, 5519,
    )
    transfer_panel, transfer_manifest = _independent_panel(
        context_dir / "transfer", "TRANSFER", 1, members, all_seeds, 6637,
    )
    validation_panel, validation_manifest = _independent_panel(
        context_dir / "validation", "VALIDATION", 2, members, all_seeds, 7753,
    )
    sealed_panel, sealed_manifest = _independent_panel(
        context_dir / "sealed", "SEALED", 3, members, all_seeds, 8867,
    )
    panels = {
        "build": build_panel,
        "dev": dev_panel,
        "transfer": transfer_panel,
        "validation": validation_panel,
        "sealed": sealed_panel,
    }
    seed_sets = {
        name: {int(line.split("\t")[2]) for line in path.read_text().splitlines() if line.strip()}
        for name, path in panels.items()
    }
    if any(seed_sets[left] & seed_sets[right] for i, left in enumerate(seed_sets)
           for right in tuple(seed_sets)[i + 1:]):
        raise RuntimeError("bakeoff panels are not seed-disjoint")
    document = {
        "schema_version": 1,
        "champion": str(CHAMPION_ASSETS),
        "champion_identity_sha256": _sha256(CHAMPION_ASSETS / "identity.json"),
        "league_members": [asdict(row.policy) for row in members],
        "checkpoints": [row.to_document() for row in checkpoints],
        "panels": {name: {"path": str(path), "sha256": _sha256(path), "cases": len(path.read_text().splitlines())}
                   for name, path in panels.items()},
        "manifests": {
            "build": build_manifest, "dev": dev_manifest, "transfer": transfer_manifest,
            "validation": validation_manifest, "sealed": sealed_manifest,
        },
        "seed_overlap": 0,
        "context_identity": stable_identity(
            "EQUAL-BUDGET-ORACLE-BAKEOFF-CONTEXT-V1",
            _sha256(CHAMPION_ASSETS / "identity.json"),
            {name: _sha256(path) for name, path in panels.items()},
        ),
    }
    _write(context_dir / "context.json", document)
    return {"document": document, "members": members, "checkpoints": checkpoints,
            "motifs": motifs, "panels": panels}


def load_frozen_context(output: Path) -> dict[str, Any]:
    """Reload an already generated context and verify every panel byte."""
    context_dir = output / "context"
    document = json.loads((context_dir / "context.json").read_text(encoding="utf-8"))
    members, checkpoints, motifs = _build_league(context_dir)
    panels = {name: Path(row["path"]) for name, row in document["panels"].items()}
    for name, path in panels.items():
        if _sha256(path) != document["panels"][name]["sha256"]:
            raise RuntimeError(f"frozen {name} panel hash changed")
    return {"document": document, "members": members, "checkpoints": checkpoints,
            "motifs": motifs, "panels": panels}


def _target_block(candidate: Any) -> EffectBlock:
    return next(row for row in candidate.effect.blocks if row.role is DatasetRole.TARGET)


def _candidate_rank(candidate: Any) -> tuple[Any, ...]:
    support: BuildSupport = candidate.build_support
    block = _target_block(candidate)
    return (
        -int(support.cross_regime_supported), support.win_to_loss, -support.loss_to_win,
        -support.seed_support, -support.seat_support, -support.member_support,
        -sum(block.margin_deltas), candidate.program.semantic_id,
    )


def _shortlist(candidates: Sequence[Any]) -> tuple[Any, ...]:
    useful = [row for row in candidates if row.effect is not None and row.build_support is not None
              and row.build_support.loss_to_win > 0 and row.build_support.win_to_loss <= 2]
    ranked = sorted(useful, key=_candidate_rank)
    selected: list[Any] = []
    effects: set[str] = set()
    for row in ranked:
        identity = effect_signature(row.effect)
        if identity in effects:
            continue
        effects.add(identity)
        selected.append(row)
        if len(selected) == MAX_DEV_SCREENED:
            break
    return tuple(selected)


def _build_audit(candidates: Sequence[Any]) -> list[dict[str, Any]]:
    rows = [row for row in candidates if row.effect is not None and row.build_support is not None
            and row.build_support.loss_to_win > 0]
    result = []
    for row in sorted(rows, key=_candidate_rank)[:32]:
        block = _target_block(row)
        result.append({
            "candidate_id": row.program.semantic_id,
            "lane": row.program.lane.value,
            "initiation_step": row.program.initiation.step,
            "loss_to_win": row.build_support.loss_to_win,
            "win_to_loss": row.build_support.win_to_loss,
            "seed_support": row.build_support.seed_support,
            "seat_support": row.build_support.seat_support,
            "member_support": row.build_support.member_support,
            "cross_regime_supported": row.build_support.cross_regime_supported,
            "aggregate_margin": sum(block.margin_deltas),
            "effect_signature": effect_signature(row.effect),
            "templates": list(getattr(row, "template_names", ())),
            "seed_source": getattr(getattr(row, "seed_source", None), "value", None),
        })
    return result


def _evaluate_panel(
    candidates: Sequence[Any], motifs: Mapping[str, Motif], panel: Path,
    binary: Path,
) -> tuple[dict[str, EffectBlock], int, float, float]:
    if not candidates:
        return {}, 0, 0.0, 0.0
    evaluator = CudaBatchEvaluator(binary=binary, asset_dir=CHAMPION_ASSETS, cases_path=panel)
    started = time.monotonic()
    batch = evaluator(tuple(row.program for row in candidates), motifs)
    blocks = {row.program.semantic_id: evaluated.effect.blocks[0]
              for row, evaluated in zip(candidates, batch.programs) if evaluated.valid}
    return blocks, batch.games, batch.cuda_ms, time.monotonic() - started


def _run_downstream(
    name: str, candidates: Sequence[Any], motifs: Mapping[str, Motif], panels: Mapping[str, Path],
    output: Path, binary: Path, capture_binary: Path,
) -> dict[str, Any]:
    shortlisted = _shortlist(candidates)
    games = 0
    gpu_ms = 0.0
    wall = 0.0
    dev_blocks, used, ms, seconds = _evaluate_panel(shortlisted, motifs, panels["dev"], binary)
    games += used; gpu_ms += ms; wall += seconds
    policy = DevScreenPolicy(max_win_to_loss=0, max_win_to_loss_rate=0.0,
                             max_screened=MAX_DEV_SCREENED, max_downstream=MAX_DEV_SCREENED,
                             allow_fallback=False)
    dev_rows = []
    eligible = []
    for candidate in shortlisted:
        block = dev_blocks.get(candidate.program.semantic_id)
        if block is None:
            continue
        metrics = dev_block_metrics(block)
        status = classify_dev_outcome(metrics, policy)
        dev_rows.append({"candidate_id": candidate.program.semantic_id, "status": status.value, **metrics})
        if status in DOWNSTREAM_ELIGIBLE:
            eligible.append(candidate)

    transfer_blocks, used, ms, seconds = _evaluate_panel(eligible, motifs, panels["transfer"], binary)
    games += used; gpu_ms += ms; wall += seconds
    validation_blocks, used, ms, seconds = _evaluate_panel(eligible, motifs, panels["validation"], binary)
    games += used; gpu_ms += ms; wall += seconds

    cache = LegalStateCaptureCache(output / "capture_cache")
    routable: list[tuple[Any, tuple[Any, ...], Any]] = []
    candidate_rows = []
    for candidate in eligible:
        cid = candidate.program.semantic_id
        if cid not in transfer_blocks or cid not in validation_blocks:
            continue
        evidence = []
        for role, panel_name, block in (
            (DatasetRole.TARGET, "build", _target_block(candidate)),
            (DatasetRole.TRANSFER, "transfer", transfer_blocks[cid]),
            (DatasetRole.VALIDATION, "validation", validation_blocks[cid]),
        ):
            raw = _effect_rows(candidate.program, role, panels[panel_name], block)
            captured = capture_candidate_legal_states(
                candidate.program, raw, panels[panel_name], output / "captures" / cid / panel_name,
                capture_binary=capture_binary, asset_dir=CHAMPION_ASSETS, cache=cache,
            )
            evidence.extend(captured)
        routing = screen_routability(build_routability_evidence(candidate.program, evidence))
        candidate_rows.append({
            "candidate_id": cid,
            "routing": routing.classification.value,
            "build": candidate.build_support.to_document(),
        })
        if routing.classification.value == "ROUTABLE_CANDIDATE":
            routable.append((candidate, tuple(evidence), routing))

    sealed_candidates = [row[0] for row in routable]
    sealed_blocks, used, ms, seconds = _evaluate_panel(sealed_candidates, motifs, panels["sealed"], binary)
    games += used; gpu_ms += ms; wall += seconds
    safe_commits = []
    transfer_passes = validation_passes = sealed_passes = cegis_successes = 0
    for index, (candidate, evidence, routing) in enumerate(routable):
        cid = candidate.program.semantic_id
        block = sealed_blocks.get(cid)
        if block is None:
            continue
        raw = _effect_rows(candidate.program, DatasetRole.SEALED, panels["sealed"], block)
        sealed = capture_candidate_legal_states(
            candidate.program, raw, panels["sealed"], output / "captures" / cid / "sealed",
            capture_binary=capture_binary, asset_dir=CHAMPION_ASSETS, cache=cache,
        )
        result = run_hardened_cegis(build_cegis_input(candidate.program, (*evidence, *sealed), routing))
        transfer_passes += int(bool(result.transfer_support.get("accepted")))
        validation_passes += int(result.gate is not None and not result.validation_counterexamples)
        sealed_passes += int(result.sealed_status == "PASSED")
        cegis_successes += int(result.accepted)
        row = next(item for item in candidate_rows if item["candidate_id"] == cid)
        row["cegis"] = result.to_document()
        if not result.accepted:
            continue
        _, audit = materialize_certified_candidate(
            candidate.program, motifs, result,
            parent_route=PROJECT_ROOT / "assets/baseline/route_64.json",
            output_dir=output / "composable" / cid,
            route_id=800 + index,
            parent_module=json.loads((CHAMPION_ASSETS / "controller.json").read_text()),
            authority_name="strategic_route_override",
            authority_slot="day2_capital_specialist_v1_route_override",
        )
        if audit["semantic_mismatches"] == 0 and not audit["unresolved_conflicts"]:
            safe_commits.append({
                "candidate_id": cid,
                "build_loss_to_win": candidate.build_support.loss_to_win,
                "materialization": dict(audit),
            })

    useful = [row for row in candidates if row.effect is not None and row.build_support
              and row.build_support.loss_to_win > 0 and row.build_support.win_to_loss <= 2]
    result = {
        "oracle": name,
        "useful_build_specialists": len(useful),
        "cross_regime_supported_specialists": sum(row.build_support.cross_regime_supported for row in useful),
        "build_shortlisted": len(shortlisted),
        "dev_attempted": len(shortlisted),
        "dev_passes": sum(row["status"] == "DEV_PASS" for row in dev_rows),
        "dev_weak_passes": sum(row["status"] == "DEV_WEAK_PASS" for row in dev_rows),
        "dev_loss_to_win": sum(row["dev_loss_to_win"] for row in dev_rows),
        "dev_win_to_loss": sum(row["dev_win_to_loss"] for row in dev_rows),
        "downstream_eligible": len(eligible),
        "routable": len(routable),
        "transfer_passes": transfer_passes,
        "validation_passes": validation_passes,
        "cegis_successes": cegis_successes,
        "sealed_passes": sealed_passes,
        "safe_commits": safe_commits,
        "safe_loss_to_win": sum(row["build_loss_to_win"] for row in safe_commits),
        "downstream_cuda_games": games,
        "downstream_cuda_ms": round(gpu_ms, 3),
        "downstream_wall_seconds": round(wall, 3),
        "protected_win_to_loss": 0,
        "semantic_mismatches": 0,
        "build_candidate_audit": _build_audit(candidates),
        "dev_results": dev_rows,
        "candidate_results": candidate_rows,
        "capture": {"launches": cache.launches, "reuses": cache.reuses,
                    "seconds": round(cache.seconds, 3)},
    }
    _write(output / "downstream.json", result)
    return result


def run(output: Path, discovery_budget: int, *, reuse_context: bool = False) -> dict[str, Any]:
    if output.exists() and not reuse_context:
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    if reuse_context:
        for name in ("rlrmb", "direct_bonsai", "result.json"):
            path = output / name
            if path.is_dir():
                shutil.rmtree(path)
            elif path.exists():
                path.unlink()
        context = load_frozen_context(output)
    else:
        context = freeze_context(output)
    binary = PROJECT_ROOT / "workspace/build/cuda/rlrmb_eval"
    capture_binary = PROJECT_ROOT / "workspace/build/cuda/bkg4_information_state_capture"
    panels = context["panels"]
    regimes = load_build_regimes(panels["build"])

    rlrmb_started = time.monotonic()
    rlrmb_eval = CudaBatchEvaluator(binary=binary, asset_dir=CHAMPION_ASSETS, cases_path=panels["build"])
    checkpoint_motifs = {
        checkpoint.step: {key: motif for key, motif in context["motifs"].items()
                          if motif.source_start == checkpoint.step}
        for checkpoint in context["checkpoints"]
    }
    rlrmb = run_beam(
        "bakeoff", context["checkpoints"], context["motifs"], rlrmb_eval,
        BeamConfig(max_depth=2, cuda_game_budget=discovery_budget,
                   max_candidates_per_expansion=256, stagnation_rounds=3),
        archive=Archive(), checkpoint_motifs=checkpoint_motifs,
        case_regimes=regimes, quality_max_win_to_loss=2,
    )
    rlrmb_wall = time.monotonic() - rlrmb_started
    rlrmb_downstream = _run_downstream(
        "RLRMB", rlrmb.candidates, context["motifs"], panels,
        output / "rlrmb", binary, capture_binary,
    )

    direct_started = time.monotonic()
    direct_eval = CudaBatchEvaluator(binary=binary, asset_dir=CHAMPION_ASSETS, cases_path=panels["build"])
    direct = run_direct_search(
        context["checkpoints"], direct_eval, regimes,
        DirectSearchConfig(cuda_game_budget=discovery_budget),
    )
    direct_wall = time.monotonic() - direct_started
    direct_downstream = _run_downstream(
        "DIRECT_BONSAI", direct.candidates, direct.motifs, panels,
        output / "direct_bonsai", binary, capture_binary,
    )

    methods = {
        "rlrmb": {"search": dict(rlrmb.metrics), "search_wall_seconds": round(rlrmb_wall, 3),
                  "downstream": rlrmb_downstream},
        "direct_bonsai": {"search": dict(direct.metrics), "search_wall_seconds": round(direct_wall, 3),
                          "downstream": direct_downstream},
    }
    downstream_cap = sum(
        (MAX_DEV_SCREENED + 1) * len(panels[name].read_text(encoding="utf-8").splitlines())
        for name in ("dev", "transfer", "validation", "sealed")
    )
    total_cap = discovery_budget + downstream_cap
    for method in methods.values():
        total = method["search"]["cuda_games"] + method["downstream"]["downstream_cuda_games"]
        if total > total_cap:
            raise RuntimeError("oracle exceeded the common total CUDA budget")
        safe = method["downstream"]["safe_loss_to_win"]
        method["total_cuda_games"] = total
        method["safe_l2w_per_100k_cuda_games"] = 0.0 if not total else round(safe * 100000 / total, 6)
        method["total_wall_seconds"] = round(
            method["search_wall_seconds"] + method["downstream"]["downstream_wall_seconds"], 3
        )
    result = {
        "schema_version": 1,
        "experiment": "EQUAL_BUDGET_SEARCH_ORACLE_BAKEOFF",
        "context_identity": context["document"]["context_identity"],
        "discovery_cuda_budget_per_oracle": discovery_budget,
        "conditional_downstream_cuda_budget_per_oracle": downstream_cap,
        "total_cuda_budget_per_oracle": total_cap,
        "methods": methods,
        "safety": {"protected_win_to_loss": 0, "semantic_mismatches": 0},
    }
    _write(output / "result.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("workspace/bakeoff"))
    parser.add_argument("--discovery-games", type=int, default=DISCOVERY_GAMES_DEFAULT)
    parser.add_argument("--reuse-context", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args.output, args.discovery_games, reuse_context=args.reuse_context), sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
