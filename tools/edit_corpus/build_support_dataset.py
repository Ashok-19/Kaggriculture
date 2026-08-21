#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SUPPORT_SCHEMA_VERSION = 1
SUPPORT_DATASET = "ashok205/kaggriculture-edit-corpus-support"
SUPPORT_TITLE = "Kaggriculture Bonsai Edit Corpus CUDA Support"
CURRENT_PARENT = "bonsai-kg/integrations/bkg4_champion/champions/w33_s172_expansion_v1"

REQUIRED_PATHS = tuple(sorted((
    "bonsai-kg/cuda/kag_adapter.cuh",
    "bonsai-kg/cuda/kag_adapter_v4.cuh",
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
    f"{CURRENT_PARENT}/identity.hpp",
    f"{CURRENT_PARENT}/identity.json",
    f"{CURRENT_PARENT}/legacy_routes.bkv6",
    f"{CURRENT_PARENT}/routes.bkv6",
    "bonsai-kg/integrations/stateful_v6/assets/identity.hpp",
    "bonsai-kg/integrations/stateful_v6/cuda_runtime.cuh",
    "bonsai-kg/integrations/stateful_v6/engine_exploits.cuh",
    "bonsai-kg/integrations/stateful_v6/route_pack.hpp",
    "bonsai-kg/python/bonsai_kg_v4.py",
    "cuda-engine/include/observation.cuh",
    "cuda-engine/include/pyrandom.cuh",
    "cuda-engine/include/runtime.cuh",
    "cuda-engine/include/sim.cuh",
    "cuda-engine/include/tape_io.hpp",
    "cuda-engine/src/bkg4_edit_corpus_eval.cu",
    "cuda-engine/src/bkg4_information_state_capture.cu",
    "tools/bkg4_search/__init__.py",
    "tools/bkg4_search/bonsai_editformer/__init__.py",
    "tools/bkg4_search/bonsai_editformer/actor_grammar.py",
    "tools/bkg4_search/campaign.py",
    "tools/bkg4_search/cuda_io.py",
    "tools/bkg4_search/evolvability.py",
    "tools/bkg4_search/information_state_io.py",
    "tools/bkg4_search/make_genomes.py",
    "tools/bkg4_search/module_codec.py",
    "tools/edit_corpus/__init__.py",
    "tools/edit_corpus/candidates.py",
    "tools/edit_corpus/corpus.py",
    "tools/edit_corpus/cuda_results.py",
    "tools/edit_corpus/kaggle_runtime.py",
    "tools/edit_corpus/notebook_runner.py",
    "tools/edit_corpus/pipeline.py",
)))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def dataset_metadata() -> dict[str, Any]:
    return {
        "title": SUPPORT_TITLE,
        "id": SUPPORT_DATASET,
        "licenses": [{"name": "CC0-1.0"}],
        "subtitle": "BKG4 CUDA support for replay-to-EditFormer corpus building",
        "description": "Private versioned source/assets bundle used by the Kaggriculture GPU edit-corpus builder notebook. Contains only the current BKG4 parent, exact CUDA evaluators, typed edit grammar/codecs, and corpus runtime dependencies.",
    }


def _git_head() -> str:
    p = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True, check=True)
    return p.stdout.strip()


def build_bundle(output: str | Path, *, force: bool = False) -> dict[str, Any]:
    out = Path(output)
    if not out.is_absolute():
        out = ROOT / out
    if out.exists():
        if not force:
            raise RuntimeError(f"output already exists: {out}")
        shutil.rmtree(out)
    out.mkdir(parents=True)

    files: dict[str, Any] = {}
    total_bytes = 0
    for rel in REQUIRED_PATHS:
        src = ROOT / rel
        if not src.is_file():
            raise FileNotFoundError(f"required support file missing: {rel}")
        dst = out / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        size = int(dst.stat().st_size)
        total_bytes += size
        files[rel] = {"bytes": size, "sha256": sha256_file(dst)}

    parent_identity = json.loads((ROOT / CURRENT_PARENT / "identity.json").read_text(encoding="utf-8"))
    manifest = {
        "schema_version": SUPPORT_SCHEMA_VERSION,
        "source_commit": _git_head(),
        "paths": list(REQUIRED_PATHS),
        "files": files,
        "total_bytes": total_bytes,
        "bkg4_abi": int(parent_identity["bkg4_abi"]),
        "feature_registry": int(parent_identity["feature_registry"]),
        "base_parent": CURRENT_PARENT,
        "base_parent_combined_identity": str(parent_identity["combined_agent_identity"]),
        "base_parent_controller_semantic_hash": str(parent_identity["controller_semantic_hash"]),
        "base_parent_nodes": int(parent_identity["controller_nodes"]),
        "base_parent_bb_slots": int(parent_identity["controller_bb_slots"]),
        "strategic_route_slot": int(parent_identity["runtime_reconciler"]["route_override_slot"]),
        "edit_grammar": "bonsai_edit_actor_grammar_v1",
        "candidate_generator": "edit_corpus_typed_generator_v1",
        "cuda_result_abi": "B4EC1/v1",
        "fast_math": False,
    }
    (out / "support_manifest.json").write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    (out / "dataset-metadata.json").write_text(json.dumps(dataset_metadata(), sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, default=Path("scratch/kaggriculture_edit_corpus_support"))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    manifest = build_bundle(args.output, force=args.force)
    print(json.dumps(manifest, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
