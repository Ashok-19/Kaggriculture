"""Freeze and validate the deterministic ASPE Phase-0 experiment context."""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from itertools import combinations
from pathlib import Path
from typing import Any, Mapping, Sequence

from kaggriculture_bonsai.league.adversarial_league import (
    ALL_LANES,
    ExploiterResult,
    dev_lane_rotation,
    generate_league_build_cases,
    make_bounded_exploiter_candidates,
)
from kaggriculture_bonsai.league.selfplay_generator import (
    SELFPLAY_SCHEMA_VERSION,
    SelfplayPolicy,
    generate_diverse_seeds,
    generate_selfplay_tapes,
    write_target_panel,
)
from kaggriculture_bonsai.paths import CHAMPION_ASSETS, PROJECT_ROOT
from kaggriculture_bonsai.search.evaluator import encode_program_batch
from kaggriculture_bonsai.search.ir import MacroProgram
from kaggriculture_bonsai.search.motifs import Motif
from kaggriculture_bonsai.search.schema import stable_identity

ASPE_BASELINE_SCHEMA = "kaggriculture.aspe.baseline"
ASPE_BASELINE_SCHEMA_VERSION = 1
ASPE_BASELINE_GENERATOR = "ASPE_PHASE_0_EQUAL_CONTEXT_V1"

BASELINE_CONFIG: dict[str, Any] = {
    "league": {
        "donor_routes": ["assets/baseline/route_64.json", "assets/baseline/route_410.json"],
        "routes_per_lane": 2,
        "selected_per_lane": 1,
        "duration": 12,
        "generation": 0,
        "mutation_origin": "bounded_internal_donor_window",
    },
    "panels": {
        "build": {
            "directory": "build_panel",
            "role": "BUILD",
            "seeds": [1103, 2207, 3301],
            "seats": [0, 1],
            "rng_seed": 4409,
            "max_cases": 48,
        },
        "internal_dev": {
            "directory": "internal_dev",
            "role": "INTERNAL_DEV",
            "generation": 0,
            "seed_count": 3,
            "seats": [1, 0],
            "lanes_per_panel": 4,
            "rng_seed": 5519,
        },
        "transfer": {
            "directory": "transfer",
            "role": "TRANSFER",
            "generation": 1,
            "seed_count": 3,
            "seats": [1, 0],
            "lanes_per_panel": 4,
            "rng_seed": 6637,
        },
        "validation": {
            "directory": "validation",
            "role": "VALIDATION",
            "generation": 2,
            "seed_count": 3,
            "seats": [1, 0],
            "lanes_per_panel": 4,
            "rng_seed": 7753,
        },
        "sealed": {
            "directory": "sealed",
            "role": "SEALED",
            "generation": 3,
            "seed_count": 3,
            "seats": [1, 0],
            "lanes_per_panel": 4,
            "rng_seed": 8867,
        },
    },
}


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(f"ASPE baseline evidence must be under the project root: {path}") from exc


def _source_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, check=True,
        capture_output=True, text=True,
    ).stdout.strip()


def _normalize_panel_paths(panel: Path) -> None:
    """Keep panel references repository-relative and portable."""
    lines = []
    for line in panel.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        fields = line.split("\t")
        fields[0] = _relative(Path(fields[0]))
        lines.append("\t".join(fields))
    panel.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _prune_unreferenced_tapes(panel: Path) -> None:
    """Discard companion opponent tapes that are not evaluation cases."""
    referenced = {
        (PROJECT_ROOT / line.split("\t", 1)[0]).resolve()
        for line in panel.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    for tape in (panel.parent / "tapes").glob("*.json"):
        if tape.resolve() not in referenced:
            tape.unlink()


def planned_panel_contract() -> dict[str, dict[str, Any]]:
    """Resolve the fixed seed/member rotation without running self-play."""
    result: dict[str, dict[str, Any]] = {}
    used = list(BASELINE_CONFIG["panels"]["build"]["seeds"])
    result["build"] = {
        **BASELINE_CONFIG["panels"]["build"],
        "lanes": [lane.value for lane in ALL_LANES],
        "members": [f"bakeoff_{lane.value.lower()}" for lane in ALL_LANES],
    }
    for name in ("internal_dev", "transfer", "validation", "sealed"):
        config = BASELINE_CONFIG["panels"][name]
        lanes = dev_lane_rotation(config["generation"], lanes_per_panel=config["lanes_per_panel"])
        seeds = generate_diverse_seeds(
            config["seed_count"], rng_seed=config["rng_seed"], existing=used,
        )
        used.extend(seeds)
        result[name] = {
            **config,
            "seeds": seeds,
            "lanes": [lane.value for lane in lanes],
            "members": [f"bakeoff_{lane.value.lower()}" for lane in lanes],
        }
    return result


def _build_league(context_dir: Path) -> tuple[list[ExploiterResult], tuple[Any, ...], dict[str, Motif]]:
    league_config = BASELINE_CONFIG["league"]
    donors = tuple(PROJECT_ROOT / path for path in league_config["donor_routes"])
    programs, motifs = make_bounded_exploiter_candidates(
        donors,
        routes_per_lane=league_config["routes_per_lane"],
        duration=league_config["duration"],
    )
    selected: list[MacroProgram] = [next(row for row in programs if row.lane is lane) for lane in ALL_LANES]
    members: list[ExploiterResult] = []
    league_dir = context_dir / "league"
    league_dir.mkdir(parents=True, exist_ok=True)
    for index, program in enumerate(selected):
        used = {phase.motif_id for phase in program.phases if phase.motif_id}
        bundle = league_dir / f"member_{index:02d}_{program.lane.value.lower()}.rlrb"
        bundle.write_bytes(encode_program_batch((program,), {key: motifs[key] for key in used}))
        policy = SelfplayPolicy(
            f"bakeoff_{program.lane.value.lower()}",
            "option_program",
            family="adversarial_league",
            checkpoint_asset_dir=str(CHAMPION_ASSETS),
            strategic_lane=program.lane.value,
            generation=league_config["generation"],
            parent_checkpoint=str(CHAMPION_ASSETS),
            member_type="cuda_exploiter",
            exploiter_program=_relative(bundle),
            mutation_origin=league_config["mutation_origin"],
        )
        members.append(ExploiterResult(policy, program, 0, 0.0, 0, 0.0, 0))
    return members, tuple(row.initiation for row in selected), motifs


def _generate_independent_panel(
    context_dir: Path,
    name: str,
    members: Sequence[ExploiterResult],
    prior_seeds: list[int],
) -> tuple[Path, dict[str, Any]]:
    config = BASELINE_CONFIG["panels"][name]
    output = context_dir / config["directory"]
    lanes = dev_lane_rotation(config["generation"], lanes_per_panel=config["lanes_per_panel"])
    by_lane = {row.program.lane: row.policy for row in members}
    policies = [by_lane[lane] for lane in lanes]
    seeds = generate_diverse_seeds(
        config["seed_count"], rng_seed=config["rng_seed"], existing=prior_seeds,
    )
    prior_seeds.extend(seeds)
    tapes = generate_selfplay_tapes(
        output / "tapes", seeds=seeds, policies=policies,
        seats=config["seats"], rng_seed=config["rng_seed"],
    )
    panel = output / f"{name}_panel.tsv"
    role_id = {"internal_dev": 1, "transfer": 2, "validation": 3, "sealed": 4}[name]
    write_target_panel(
        tapes, panel, protected_flag=int(name in {"validation", "sealed"}), role=role_id,
    )
    _normalize_panel_paths(panel)
    _prune_unreferenced_tapes(panel)
    summary = {
        "role": config["role"],
        "authority": "SCREENING_ONLY" if name == "internal_dev" else "DOWNSTREAM_ONLY",
        "generation": config["generation"],
        "seeds": seeds,
        "seats": config["seats"],
        "lanes": [lane.value for lane in lanes],
        "league_members": [row.name for row in policies],
        "cases": len(tapes),
        "champion_losses": sum(row.margin < 0 for row in tapes),
        "panel": _relative(panel),
    }
    _write_json(output / "manifest.json", summary)
    return panel, summary


def _panel_cases(panel: Path) -> list[dict[str, Any]]:
    cases = []
    for line in panel.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        tape_value, seat, seed, family, protected, own_history, role = line.split("\t")
        tape = PROJECT_ROOT / tape_value
        source = json.loads(tape.read_text(encoding="utf-8"))["source"]
        cases.append({
            "identity": stable_identity(
                "ASPE-PANEL-CASE-V1", int(seed), int(seat),
                source["opponent_policy"], source["strategic_lane"], _sha256(tape),
            ),
            "tape": tape_value,
            "tape_sha256": _sha256(tape),
            "seed": int(seed),
            "seat": int(seat),
            "member": source["opponent_policy"],
            "lane": source["strategic_lane"],
            "family_id": int(family),
            "protected": bool(int(protected)),
            "own_history": bool(int(own_history)),
            "role_id": int(role),
        })
    return cases


def _coverage(cases: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    def counts(key: str) -> dict[str, int]:
        values = sorted({row[key] for row in cases}, key=str)
        return {str(value): sum(row[key] == value for row in cases) for value in values}

    return {
        "seeds": counts("seed"),
        "seats": counts("seat"),
        "members": counts("member"),
        "lanes": counts("lane"),
        "regime_count": len({(row["seed"], row["seat"], row["member"], row["lane"]) for row in cases}),
    }


def _champion_document() -> dict[str, Any]:
    identity_path = CHAMPION_ASSETS / "identity.json"
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    assets = {
        path.name: _sha256(path)
        for path in sorted(CHAMPION_ASSETS.iterdir())
        if path.is_file()
    }
    return {
        "path": _relative(CHAMPION_ASSETS),
        "combined_agent_identity": identity["combined_agent_identity"],
        "controller_semantic_hash": identity["controller_semantic_hash"],
        "identity_json_sha256": assets["identity.json"],
        "assets": assets,
    }


def build_manifest(output: Path, *, source_head: str) -> dict[str, Any]:
    context_dir = output / "context"
    panel_paths = {
        "build": context_dir / "build_panel" / "build_panel.tsv",
        "internal_dev": context_dir / "internal_dev" / "internal_dev_panel.tsv",
        "transfer": context_dir / "transfer" / "transfer_panel.tsv",
        "validation": context_dir / "validation" / "validation_panel.tsv",
        "sealed": context_dir / "sealed" / "sealed_panel.tsv",
    }
    panels: dict[str, Any] = {}
    for name, panel in panel_paths.items():
        cases = _panel_cases(panel)
        panel_hash = _sha256(panel)
        panels[name] = {
            "role": BASELINE_CONFIG["panels"][name]["role"],
            "identity": stable_identity("ASPE-PANEL-V1", name, panel_hash, cases),
            "path": _relative(panel),
            "sha256": panel_hash,
            "case_count": len(cases),
            "seeds": sorted({row["seed"] for row in cases}),
            "seats": sorted({row["seat"] for row in cases}),
            "members": sorted({row["member"] for row in cases}),
            "lanes": sorted({row["lane"] for row in cases}),
            "coverage": _coverage(cases),
            "cases": cases,
        }

    league = []
    for bundle in sorted((context_dir / "league").glob("*.rlrb")):
        index = int(bundle.name.split("_", 2)[1])
        lane = ALL_LANES[index].value
        name = f"bakeoff_{lane.lower()}"
        bundle_hash = _sha256(bundle)
        league.append({
            "identity": stable_identity("ASPE-LEAGUE-MEMBER-V1", name, lane, bundle_hash),
            "name": name,
            "lane": lane,
            "program": _relative(bundle),
            "program_sha256": bundle_hash,
            "generation": BASELINE_CONFIG["league"]["generation"],
            "member_type": "cuda_exploiter",
            "mutation_origin": BASELINE_CONFIG["league"]["mutation_origin"],
        })

    overlaps = {}
    for left, right in combinations(panels, 2):
        left_seeds = set(panels[left]["seeds"])
        right_seeds = set(panels[right]["seeds"])
        left_cases = {row["identity"] for row in panels[left]["cases"]}
        right_cases = {row["identity"] for row in panels[right]["cases"]}
        overlaps[f"{left}__{right}"] = {
            "seed_count": len(left_seeds & right_seeds),
            "seeds": sorted(left_seeds & right_seeds),
            "case_identity_count": len(left_cases & right_cases),
        }

    config_identity = hashlib.sha256(_canonical_bytes(BASELINE_CONFIG)).hexdigest()
    generator_path = Path(__file__)
    champion = _champion_document()
    context_identity = stable_identity(
        "ASPE-BASELINE-CONTEXT-V1",
        champion["combined_agent_identity"],
        [row["identity"] for row in league],
        {name: row["identity"] for name, row in panels.items()},
        config_identity,
    )
    return {
        "schema": ASPE_BASELINE_SCHEMA,
        "schema_version": ASPE_BASELINE_SCHEMA_VERSION,
        "context_identity": context_identity,
        "source_head": source_head,
        "champion": champion,
        "league": {"size": len(league), "members": league},
        "panels": panels,
        "generation": {
            "generator": ASPE_BASELINE_GENERATOR,
            "implementation": _relative(generator_path),
            "implementation_sha256": _sha256(generator_path),
            "config_identity": config_identity,
            "config": BASELINE_CONFIG,
            "selfplay_schema_version": SELFPLAY_SCHEMA_VERSION,
            "kaggle_environments_version": "1.32.7",
        },
        "cross_panel_overlap": {
            "all_seed_disjoint": all(row["seed_count"] == 0 for row in overlaps.values()),
            "all_case_identity_disjoint": all(row["case_identity_count"] == 0 for row in overlaps.values()),
            "pairs": overlaps,
        },
    }


def baseline_results(source: Path) -> dict[str, Any]:
    document = json.loads(source.read_text(encoding="utf-8"))

    def method(name: str) -> dict[str, Any]:
        row = document["methods"][name]
        search = row["search"]
        downstream = row["downstream"]
        return {
            "candidate_count": search["cuda_candidates_evaluated"],
            "structurally_unique_programs": search.get("unique_structural_ids"),
            "unique_effect_phenotypes": search["unique_effect_signatures"],
            "distinct_loss_to_win_patterns": search.get("distinct_loss_to_win_patterns"),
            "useful_build_specialists": downstream["useful_build_specialists"],
            "cross_regime_build_specialists": downstream["cross_regime_supported_specialists"],
            "dev_attempted": downstream["dev_attempted"],
            "dev_passes": downstream["dev_passes"],
            "dev_weak_passes": downstream["dev_weak_passes"],
            "dev_loss_to_win": downstream["dev_loss_to_win"],
            "dev_win_to_loss": downstream["dev_win_to_loss"],
            "downstream_eligible": downstream["downstream_eligible"],
            "transfer_passes": downstream["transfer_passes"],
            "validation_passes": downstream["validation_passes"],
            "cegis_successes": downstream["cegis_successes"],
            "sealed_passes": downstream["sealed_passes"],
            "safe_commits": len(downstream["safe_commits"]),
            "discovery_cuda_games": search["cuda_games"],
            "downstream_cuda_games": downstream["downstream_cuda_games"],
            "semantic_mismatches": downstream["semantic_mismatches"],
            "protected_win_to_loss": downstream["protected_win_to_loss"],
        }

    return {
        "schema": "kaggriculture.aspe.search-baselines",
        "schema_version": 1,
        "source": _relative(source),
        "source_sha256": _sha256(source),
        "source_context_identity": document["context_identity"],
        "discovery_cuda_budget_per_oracle": document["discovery_cuda_budget_per_oracle"],
        "conditional_downstream_cuda_budget_per_oracle": document["conditional_downstream_cuda_budget_per_oracle"],
        "rlrmb": method("rlrmb"),
        "direct_bonsai": {
            **method("direct_bonsai"),
            "strategic_effect_phenotypes": 1,
            "strategic_effect_note": "All 13 useful BUILD candidates collapsed to one step-96 phenotype.",
        },
        "safety": document["safety"],
    }


def validate_manifest(manifest: Mapping[str, Any], *, verify_files: bool = True) -> None:
    if manifest.get("schema") != ASPE_BASELINE_SCHEMA:
        raise ValueError("unexpected ASPE baseline manifest schema")
    if manifest.get("schema_version") != ASPE_BASELINE_SCHEMA_VERSION:
        raise ValueError("unexpected ASPE baseline manifest schema version")
    if manifest["league"]["size"] != len(ALL_LANES):
        raise ValueError("ASPE baseline must contain one member per strategic lane")
    if not manifest["cross_panel_overlap"]["all_seed_disjoint"]:
        raise ValueError("ASPE baseline panel seeds overlap")
    if not manifest["cross_panel_overlap"]["all_case_identity_disjoint"]:
        raise ValueError("ASPE baseline panel cases overlap")
    if manifest["panels"]["build"]["case_count"] != 48:
        raise ValueError("ASPE BUILD must contain 48 regimes")
    if set(manifest["panels"]["build"]["seats"]) != {0, 1}:
        raise ValueError("ASPE BUILD must cover both seats")
    if len(manifest["panels"]["build"]["members"]) != len(ALL_LANES):
        raise ValueError("ASPE BUILD must cover every league member")
    if len(manifest["panels"]["build"]["lanes"]) != len(ALL_LANES):
        raise ValueError("ASPE BUILD must cover every strategic lane")
    for name in ("internal_dev", "transfer", "validation", "sealed"):
        if manifest["panels"][name]["case_count"] != 24:
            raise ValueError(f"ASPE {name} must contain 24 cases")
    if verify_files:
        for panel in manifest["panels"].values():
            path = PROJECT_ROOT / panel["path"]
            if _sha256(path) != panel["sha256"]:
                raise ValueError(f"panel hash mismatch: {panel['path']}")
            for case in panel["cases"]:
                if _sha256(PROJECT_ROOT / case["tape"]) != case["tape_sha256"]:
                    raise ValueError(f"tape hash mismatch: {case['tape']}")
        for member in manifest["league"]["members"]:
            if _sha256(PROJECT_ROOT / member["program"]) != member["program_sha256"]:
                raise ValueError(f"league member hash mismatch: {member['program']}")
        for name, digest in manifest["champion"]["assets"].items():
            if _sha256(CHAMPION_ASSETS / name) != digest:
                raise ValueError(f"champion asset hash mismatch: {name}")


def freeze_aspe_baseline(
    output: Path,
    *,
    source_head: str | None = None,
    replace: bool = False,
) -> dict[str, Any]:
    output = PROJECT_ROOT / output if not output.is_absolute() else output
    _relative(output)
    if output.exists():
        if not replace:
            raise FileExistsError(f"baseline already exists: {output}")
        shutil.rmtree(output)
    context_dir = output / "context"
    members, _, _ = _build_league(context_dir)
    build_config = BASELINE_CONFIG["panels"]["build"]
    build_panel, build_summary = generate_league_build_cases(
        context_dir / build_config["directory"],
        seeds=build_config["seeds"],
        exploiters=members,
        rng_seed=build_config["rng_seed"],
        max_cases=build_config["max_cases"],
    )
    _normalize_panel_paths(build_panel)
    _prune_unreferenced_tapes(build_panel)
    build_summary["panel"] = _relative(build_panel)
    _write_json(context_dir / build_config["directory"] / "build_manifest.json", build_summary)

    used_seeds = list(build_config["seeds"])
    for name in ("internal_dev", "transfer", "validation", "sealed"):
        _generate_independent_panel(context_dir, name, members, used_seeds)

    manifest = build_manifest(output, source_head=source_head or _source_head())
    validate_manifest(manifest)
    _write_json(output / "manifest.json", manifest)
    _write_json(output / "baseline_results.json", baseline_results(PROJECT_ROOT / "workspace/oracle_bakeoff/result.json"))
    return manifest
