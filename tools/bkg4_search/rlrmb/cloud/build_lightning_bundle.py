"""Build a minimal, dependency-closed Lightning AI RLRMB bundle."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tarfile
from pathlib import Path
from typing import Iterable, Sequence

from ..cloud_runner import load_cloud_config
from ..schema import stable_identity

SCHEMA_VERSION = 1
ROOT = Path(__file__).resolve().parents[4]


def _panel_inputs(path: Path) -> tuple[Path, ...]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.startswith("#"):
            rows.append(ROOT / line.split("\t", 1)[0])
    return tuple(rows)


def required_paths(config_path: Path) -> tuple[Path, ...]:
    config = load_cloud_config(config_path)
    source_document = json.loads(config_path.read_text(encoding="utf-8"))
    base_config = ROOT / source_document["base_config"] if "base_config" in source_document else config_path
    direct = [config_path, base_config,
              ROOT / "tools", ROOT / "cuda-engine", ROOT / "cpp-engine", ROOT / "bonsai-kg", ROOT / "agent_system",
              ROOT / config["asset_dir"], ROOT / config["donor_manifest"], ROOT / config["parent_route"],
              ROOT / config["transfer_panel"], ROOT / config["validation_panel"], ROOT / config["sealed_panel"]]
    direct.extend(ROOT / row["target_panel"] for row in config["families"])
    direct.extend(ROOT / row["checkpoint_evidence"] for row in config["families"])
    referenced = []
    for path in direct:
        if path.is_file() and path.suffix == ".tsv":
            referenced.extend(_panel_inputs(path))
        if path.is_file() and path.name == "donors64.txt":
            referenced.extend(ROOT / line.strip() for line in path.read_text().splitlines() if line.strip())
    unique = {path.resolve() for path in (*direct, *referenced)}
    missing = [str(path) for path in unique if not path.exists()]
    if missing:
        raise ValueError(f"bundle inputs are missing: {missing[:8]}")
    return tuple(sorted(unique))


def _copy_path(source: Path, destination_root: Path) -> None:
    relative = source.relative_to(ROOT)
    destination = destination_root / relative
    if source.is_dir():
        if relative.parts[0] in {"tools", "cuda-engine", "cpp-engine", "bonsai-kg", "agent_system"}:
            tracked = subprocess.run(("git", "ls-files", "--", str(relative)), cwd=ROOT, check=True,
                                     capture_output=True, text=True).stdout.splitlines()
            for item in tracked:
                tracked_source = ROOT / item
                if tracked_source.is_file():
                    tracked_destination = destination_root / item
                    tracked_destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(tracked_source, tracked_destination)
        else:
            shutil.copytree(source, destination, dirs_exist_ok=True,
                            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache", "build"))
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def _files(root: Path) -> Iterable[Path]:
    return sorted(path for path in root.rglob("*") if path.is_file())


def build_bundle(config_path: Path, output_dir: Path, archive: Path) -> dict:
    if output_dir.exists():
        raise ValueError("bundle output directory already exists")
    output_dir.mkdir(parents=True)
    for path in required_paths(config_path.resolve()):
        _copy_path(path, output_dir)
    head = subprocess.run(("git", "rev-parse", "HEAD"), cwd=ROOT, check=True,
                          capture_output=True, text=True).stdout.strip()
    config = load_cloud_config(config_path.resolve())
    included = [str(path.relative_to(output_dir)) for path in _files(output_dir)]
    included.append("phase16_bundle_manifest.json")
    included.sort()
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "source_head": head,
        "production_config_identity": stable_identity("RLRMB-P16-CONFIG-V1", config),
        "parent_identity": hashlib.sha256((output_dir / config["asset_dir"] / "controller.bkgc").read_bytes()).hexdigest(),
        "included_files": included,
        "input_panels": [config["transfer_panel"], config["validation_panel"], config["sealed_panel"],
                         *[row["target_panel"] for row in config["families"]]],
        "cuda_requirements": {"nvcc": "CUDA 12.x", "architecture": "auto-detected by lightning_bootstrap.sh"},
        "expected_hardware": config["cloud"]["expected_hardware"],
        "dependency_identities": {"python": "stdlib-only orchestration", "cuda_engine": "source-bundled",
                                  "bonsai_kg": "source-bundled"},
        "cloud_command": "./tools/bkg4_search/rlrmb/cloud/lightning_run.sh pilot OUTPUT [runtime options...]",
        "expected_results": ["generation_state.json", "cloud_result.json", "telemetry.jsonl",
                             "generations/", "completed_attempts/"],
    }
    manifest_path = output_dir / "phase16_bundle_manifest.json"
    size = -1
    for _ in range(4):
        manifest["generated_bundle_bytes"] = max(0, size)
        manifest_path.write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        with tarfile.open(archive, "w:gz", compresslevel=6) as tar:
            tar.add(output_dir, arcname="kaggriculture_phase16")
        current = archive.stat().st_size
        if current == size:
            break
        size = current
    manifest["generated_bundle_bytes"] = archive.stat().st_size
    manifest_path.write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    sidecar = archive.with_name(archive.name + ".manifest.json")
    sidecar_manifest = dict(manifest)
    sidecar_manifest["generated_bundle_bytes"] = archive.stat().st_size
    sidecar_manifest["archive_sha256"] = hashlib.sha256(archive.read_bytes()).hexdigest()
    sidecar.write_text(json.dumps(sidecar_manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    args = parser.parse_args(argv)
    manifest = build_bundle(args.config, args.output_dir, args.archive)
    print(json.dumps({"files": len(manifest["included_files"]),
                      "bundle_bytes": manifest["generated_bundle_bytes"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
