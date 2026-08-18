#!/usr/bin/env python3
"""Build the development-only Kaggriculture dual-T4 meta-search input bundle."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from collections import defaultdict
from pathlib import Path

MOUNT_NAME = "kaggriculture-meta-search"
EXPECTED_DEV = 168
EXPECTED_VALIDATION = 84
EXPECTED_BLIND = 50
EXPECTED_FAMILY_HOLDOUT = 28
V6_ROUTE_IDS = (22, 23, 29, 32, 34, 46, 63, 64, 70, 77, 79)
SEARCH_PRIORS = (
    ("broad_all81.bin", "scratch/bonsai_next/broad_all81_seed1273000_n256.txt.bin"),
    ("efe4_tree.bin", "scratch/bonsai_next/efe4_tree_win_end2end_seed1265000_n256.txt.bin"),
    ("prefix_route64.bin", "scratch/bonsai_next/prefix_route64_to_continuation.bin"),
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_lines(path: Path) -> list[str]:
    return [x.strip() for x in path.read_text(encoding="utf-8").splitlines() if x.strip() and not x.lstrip().startswith("#")]


def copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def normalize_tape_json(src: Path, dst: Path) -> None:
    """Write tape JSON as raw UTF-8 and canonicalize Kawashigi metadata."""
    def normalize(value):
        if isinstance(value, str):
            return value.replace("カワシギ", "Kawashigi")
        if isinstance(value, list):
            return [normalize(x) for x in value]
        if isinstance(value, dict):
            return {k: normalize(v) for k, v in value.items()}
        return value

    obj = normalize(json.loads(src.read_text(encoding="utf-8")))
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")


def copy_tree_files(src: Path, dst: Path, patterns: tuple[str, ...]) -> None:
    for pattern in patterns:
        for p in sorted(src.glob(pattern)):
            if p.is_file():
                copy_file(p, dst / p.relative_to(src))


def load_forbidden_hashes(root: Path, validation_rows: list[dict[str, str]]) -> tuple[set[str], dict[str, int]]:
    forbidden: set[str] = set()
    counts = {"validation": len(validation_rows), "blind": 0, "family_holdout": 0}
    for row in validation_rows:
        p = root / row["native_tape_path"]
        if p.exists():
            forbidden.add(sha256(p))
    for key, rel, expected in (
        ("blind", "scratch/bonsai_kg_v6_reinforcement/population/blind_holdout_routes.txt", EXPECTED_BLIND),
        ("family_holdout", "scratch/bonsai_kg_v6_reinforcement/population/family_holdout_routes.txt", EXPECTED_FAMILY_HOLDOUT),
    ):
        lines = read_lines(root / rel)
        counts[key] = len(lines)
        if len(lines) != expected:
            raise RuntimeError(f"{key} count changed: expected {expected}, got {len(lines)}")
        for line in lines:
            p = root / line
            if p.exists():
                forbidden.add(sha256(p))
    return forbidden, counts


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, default=Path("scratch/kaggriculture_meta_search_bundle"))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[2]
    out = args.output if args.output.is_absolute() else root / args.output
    if out.exists():
        if not args.force:
            raise RuntimeError(f"output exists: {out}; pass --force to replace it")
        shutil.rmtree(out)
    out.mkdir(parents=True)

    bank = root / "scratch/bonsai_kg_v6_reinforcement/multireplay_bank"
    manifest_path = bank / "priority_multireplay_manifest.tsv"
    with manifest_path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    dev = [dict(r, team=("Kawashigi" if r["team"] == "カワシギ" else r["team"])) for r in rows if r["bank_role"] == "dev"]
    validation = [r for r in rows if r["bank_role"] == "validation"]
    if len(dev) != EXPECTED_DEV or len(validation) != EXPECTED_VALIDATION:
        raise RuntimeError(f"bank split changed: dev={len(dev)} validation={len(validation)}")
    families = sorted({r["team"] for r in dev})
    if len(families) != 21:
        raise RuntimeError(f"expected 21 development families, got {len(families)}")

    forbidden_hashes, holdout_counts = load_forbidden_hashes(root, validation)

    # Development tapes only. Never copy the source bank directory wholesale.
    bundle_rows: list[dict[str, str]] = []
    for i, row in enumerate(dev):
        src = root / row["native_tape_path"]
        name = f"{i:04d}_{src.name}"
        rel = Path("populations/dev_tapes") / name
        if sha256(src) in forbidden_hashes:
            raise RuntimeError(f"development tape collides with frozen validation/holdout: {src}")
        normalize_tape_json(src, out / rel)
        nr = dict(row)
        nr["bundle_path"] = rel.as_posix()
        bundle_rows.append(nr)

    manifest_fields = list(bundle_rows[0])
    with (out / "populations/dev_manifest.tsv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=manifest_fields, delimiter="\t")
        w.writeheader(); w.writerows(bundle_rows)

    # Baseline evidence selects hard representatives without exposing validation.
    baseline_src = bank / "baseline_dev_cases.tsv"
    with baseline_src.open(encoding="utf-8", newline="") as f:
        baseline = list(csv.DictReader(f, delimiter="\t"))
    by_original = {r["native_tape_path"]: r for r in bundle_rows}
    rewritten_baseline = []
    path_worst: dict[str, float] = defaultdict(lambda: float("inf"))
    for row in baseline:
        source_rel = row["path"]
        if source_rel not in by_original:
            raise RuntimeError(f"baseline references non-development tape: {source_rel}")
        br = dict(row)
        br["path"] = by_original[source_rel]["bundle_path"]
        rewritten_baseline.append(br)
        path_worst[br["path"]] = min(path_worst[br["path"]], float(br["margin"]))
    with (out / "populations/baseline_dev_cases.tsv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rewritten_baseline[0]), delimiter="\t")
        w.writeheader(); w.writerows(rewritten_baseline)

    per_family: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in bundle_rows:
        per_family[row["team"]].append(row)
    for fam in per_family:
        per_family[fam].sort(key=lambda r: (path_worst[r["bundle_path"]], int(r["rank"]), r["action_sha256"]))

    # Progressive development-only funnel. Ultra-core includes every baseline-losing family first,
    # then adds the most fragile perfect families as regression guards. Both seats are evaluated by CUDA.
    family_losses = defaultdict(int)
    family_floor = defaultdict(lambda: float("inf"))
    for row in rewritten_baseline:
        fam = by_original[dev[int(row["route_index"])]["native_tape_path"]]["team"]
        family_losses[fam] += int(row["class"] == "L")
        family_floor[fam] = min(family_floor[fam], float(row["margin"]))
    hard_families = sorted((f for f in families if family_losses[f] > 0), key=lambda f: (-family_losses[f], family_floor[f], f))
    guard_families = sorted((f for f in families if family_losses[f] == 0), key=lambda f: (family_floor[f], f))

    # Keep the ultra-core adversarial rather than purely loss-driven. The current rank-1
    # Kawashigi family and Ueddy are strategically important live-meta families even when
    # the frozen baseline happens to beat their sampled tapes. Reserve explicit Stage-1
    # slots for them, then fill the remaining budget with baseline hard/fragile families.
    mandatory_stage1_elites = [f for f in ("Kawashigi", "Ueddy") if f in families]
    stage1_families = []
    for fam in hard_families + mandatory_stage1_elites + guard_families:
        if fam not in stage1_families:
            stage1_families.append(fam)
        if len(stage1_families) == 12:
            break
    stage1 = [per_family[f][0] for f in stage1_families]
    stage2 = [per_family[f][0] for f in families]
    stage3 = [r for f in families for r in per_family[f][:2]]
    for name, selected in (("stage1_core12.txt", stage1), ("stage2_core21.txt", stage2), ("stage3_core42.txt", stage3), ("dev168.txt", bundle_rows)):
        (out / "populations" / name).write_text("".join(Path(r["bundle_path"]).relative_to("populations").as_posix() + "\n" for r in selected), encoding="utf-8")

    # Candidate actuator bank: authoritative v6 routes only. Opponent tapes stay separate.
    actuator_lines = []
    for rid in V6_ROUTE_IDS:
        src = root / f"scratch/bonsai_cpu/stateful_core_v6/route{rid}.json"
        rel = Path("actuators/v6") / src.name
        normalize_tape_json(src, out / rel)
        actuator_lines.append(rel.relative_to("actuators").as_posix())
    (out / "actuators/v6_routes.txt").write_text("".join(x + "\n" for x in actuator_lines), encoding="utf-8")

    # Runtime source: exact CUDA engine search subset and the BKG3-v6 CUDA smoke runner.
    copy_tree_files(root / "cuda-engine/include", out / "cuda-engine/include", ("*.cuh", "*.hpp"))
    for rel in ("cuda-engine/src/bonsai_search.cu", "cuda-engine/src/bonsai_memetic_search.cu"):
        copy_file(root / rel, out / rel)

    official_engine = root / "kaggle-environments-master/kaggle_environments/envs/kaggriculture/kaggriculture.py"
    if not official_engine.exists():
        raise RuntimeError(f"official Kaggriculture engine source missing: {official_engine}")
    copy_file(official_engine, out / "official_engine/kaggriculture.py")
    official_engine_sha256 = sha256(official_engine)

    prior_names = []
    for name, rel in SEARCH_PRIORS:
        src = root / rel
        if not src.exists() or src.stat().st_size != 4096:
            raise RuntimeError(f"invalid 4096-byte GPProgram search prior: {src}")
        copy_file(src, out / "priors" / name)
        prior_names.append(name)
    copy_tree_files(root / "bonsai-kg/include", out / "bonsai-kg/include", ("*.hpp",))
    copy_tree_files(root / "bonsai-kg/cuda", out / "bonsai-kg/cuda", ("*.cuh",))
    for rel in (
        "bonsai-kg/integrations/stateful_v6/cuda_runtime.cuh",
        "bonsai-kg/integrations/stateful_v6/route_pack.hpp",
        "bonsai-kg/integrations/stateful_v6/assets/controller.bkgc",
        "bonsai-kg/integrations/stateful_v6/assets/routes.bkv6",
        "bonsai-kg/integrations/stateful_v6/assets/identity.hpp",
        "bonsai-kg/tests/stateful_v6_game_cuda.cu",
        "tools/meta_search/island_campaign.py",
        "tools/meta_search/autotune.py",
        "tools/meta_search/merge_elites.py",
        "tools/meta_search/v6_smoke_reference.json",
        "tools/meta_search/default_config.json",
    ):
        copy_file(root / rel, out / rel)

    # Leakage audit by content hash, not just filenames.
    copied_json = list(out.rglob("*.json"))
    collisions = [(p.relative_to(out).as_posix(), sha256(p)) for p in copied_json if sha256(p) in forbidden_hashes]
    if collisions:
        raise RuntimeError(f"validation/holdout content leaked into bundle: {collisions[:5]}")

    manifest = {
        "mount_name": MOUNT_NAME,
        "dev_tapes": len(dev),
        "families": len(families),
        "stage1_routes": len(stage1),
        "stage2_routes": len(stage2),
        "stage3_routes": len(stage3),
        "stage1_families": stage1_families,
        "stage1_mandatory_elites": mandatory_stage1_elites,
        "validation_excluded": len(validation),
        "blind_excluded": holdout_counts["blind"],
        "family_holdout_excluded": holdout_counts["family_holdout"],
        "candidate_actuator_routes": list(V6_ROUTE_IDS),
        "search_priors": prior_names,
        "resident_search_genome_bytes": 256,
        "engine_exploitation_search": True,
        "engine_sub_islands": {"exploit": 12, "explore": 20, "total_default": 32},
        "official_engine_path": "official_engine/kaggriculture.py",
        "official_engine_sha256": official_engine_sha256,
        "pinned_semantics": "kaggle-environments==1.32.7",
        "fast_math": False,
    }
    (out / "bundle_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (out / "README.md").write_text(
        "# Kaggriculture dual-T4 meta-search input\n\n"
        "Development-only runtime bundle. Mount this Kaggle Dataset as `kaggriculture-meta-search`. "
        "It contains 168 development tapes and explicitly excludes all frozen/blind/family holdouts.\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(out), **manifest}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
