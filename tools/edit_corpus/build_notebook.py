#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "notebooks/kaggriculture_edit_corpus_builder"
OUT_NOTEBOOK = OUT_DIR / "kaggriculture_edit_corpus_builder.ipynb"
OUT_META = OUT_DIR / "kernel-metadata.json"


def _src(text: str) -> list[str]:
    return [line + "\n" for line in text.strip("\n").splitlines()]


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": _src(text)}


def code(text: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": _src(text)}


def render_notebook() -> str:
    cells = [
        md(r'''
# Kaggriculture BonsaiEditFormer CUDA Edit Corpus Builder

This GPU notebook has one job: incrementally convert the persistent top-10 replay archive into compact structured supervision for a future generalized BonsaiEditFormer ranker.

Pipeline:

`top-10 replay archive -> replay-derived cases -> legal typed BKG4 edits -> exact CUDA outcomes -> persistent edit corpus`

It **does not train a ranker**, modify/promote the champion, or submit to the competition. Team/opponent names, replay IDs and seeds are stored only as provenance; model context contains only legal/public state plus BKG4 structure/edit semantics and exact CUDA supervision.
'''),
        code(r'''
from pathlib import Path
import hashlib, json, shutil, sys, tarfile

ARCHIVE_DATASET = "ashok205/kaggriculture-top10-replay-archive"
SUPPORT_DATASET = "ashok205/kaggriculture-edit-corpus-support"
CORPUS_DATASET = "ashok205/kaggriculture-bonsai-edit-corpus"

PUBLISH = True
SMOKE_ONLY = False
MAX_DATES = 1 if SMOKE_ONLY else None
MAX_REPLAYS_PER_DAY = 4 if SMOKE_ONLY else None
MAX_CASES_PER_DAY = 8 if SMOKE_ONLY else 512
SEED_PARENT_COUNT = 2 if SMOKE_ONLY else 8
CANDIDATES_PER_PARENT = 32 if SMOKE_ONLY else 768
REPLAY_CONVERTER_WORKERS = 1 if SMOKE_ONLY else 2
MAX_CUDA_DEVICES = 1 if SMOKE_ONLY else 2
TARGET_CUDA_GAMES = 2_048 if SMOKE_ONLY else 1_500_000
EVAL_GPU_MEMORY_FRACTION = 0.50 if SMOKE_ONLY else 0.70

STAGE = Path("/kaggle/working/kaggriculture_bonsai_edit_corpus")
TMP = Path("/kaggle/working/kaggriculture_bonsai_edit_corpus_tmp")


def locate_input(slug, marker):
    candidates = [
        Path("/kaggle/input/datasets") / slug,
        Path("/kaggle/input") / slug,
    ]
    for p in candidates:
        if (p / marker).is_file():
            return p
    hits = sorted({p.parent for p in Path("/kaggle/input").rglob(marker)})
    if len(hits) != 1:
        raise RuntimeError(f"cannot uniquely locate {slug} via {marker}: {hits}")
    return hits[0]

SUPPORT_INPUT = locate_input("ashok205/kaggriculture-edit-corpus-support", "support_manifest.json")
ARCHIVE = locate_input("ashok205/kaggriculture-top10-replay-archive", "archive_manifest.json")

support_manifest = json.loads((SUPPORT_INPUT / "support_manifest.json").read_text())
SUPPORT = Path("/kaggle/working/kaggriculture_edit_corpus_support_runtime")
if SUPPORT.exists():
    shutil.rmtree(SUPPORT)

# Kaggle expands uploaded archives into a directory named after the archive.
expanded = SUPPORT_INPUT / "support_bundle"
bundle = SUPPORT_INPUT / str(support_manifest.get("bundle_file", "support_bundle.tar"))
if expanded.is_dir():
    shutil.copytree(expanded, SUPPORT)
elif bundle.is_file():
    if bundle.stat().st_size != int(support_manifest["bundle_bytes"]):
        raise RuntimeError("support bundle size mismatch")
    h = hashlib.sha256()
    with bundle.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    if h.hexdigest() != str(support_manifest["bundle_sha256"]):
        raise RuntimeError("support bundle SHA256 mismatch")
    SUPPORT.mkdir(parents=True)
    root_resolved = SUPPORT.resolve()
    with tarfile.open(bundle, "r") as tf:
        for member in tf.getmembers():
            target = (SUPPORT / member.name).resolve()
            if root_resolved != target and root_resolved not in target.parents:
                raise RuntimeError(f"unsafe support bundle member: {member.name}")
            if not member.isfile():
                raise RuntimeError(f"unexpected non-file support bundle member: {member.name}")
        tf.extractall(SUPPORT)
else:
    raise RuntimeError("support dataset has neither support_bundle/ nor support_bundle.tar")

# Verify every runtime file against the support manifest after materialization.
for rel, meta in support_manifest["files"].items():
    fp = SUPPORT / rel
    if not fp.is_file():
        raise RuntimeError(f"support runtime missing {rel}")
    if fp.stat().st_size != int(meta["bytes"]):
        raise RuntimeError(f"support runtime size mismatch: {rel}")
    h = hashlib.sha256()
    with fp.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    if h.hexdigest() != str(meta["sha256"]):
        raise RuntimeError(f"support runtime SHA256 mismatch: {rel}")
shutil.copy2(SUPPORT_INPUT / "support_manifest.json", SUPPORT / "support_manifest.json")

# The seed corpus dataset may not contain corpus_manifest.json on the first build.
_prior_candidates = [
    Path("/kaggle/input/datasets/ashok205/kaggriculture-bonsai-edit-corpus"),
    Path("/kaggle/input/ashok205/kaggriculture-bonsai-edit-corpus"),
]
PRIOR_CORPUS = next((p for p in _prior_candidates if p.exists()), None)
if PRIOR_CORPUS is None:
    _hits = [p.parent for p in Path("/kaggle/input").rglob("dataset-metadata.json") if "bonsai-edit-corpus" in str(p).lower()]
    PRIOR_CORPUS = _hits[0] if len(_hits) == 1 else None

sys.path.insert(0, str(SUPPORT))
from tools.edit_corpus.notebook_runner import run_builder, run as run_cmd
from tools.edit_corpus.kaggle_runtime import validate_corpus

print({
    "archive": str(ARCHIVE),
    "support": str(SUPPORT),
    "prior_corpus": None if PRIOR_CORPUS is None else str(PRIOR_CORPUS),
    "smoke_only": SMOKE_ONLY,
    "publish": PUBLISH,
})
'''),
        code(r'''
# Fail fast on support/archive provenance before compiling CUDA.
support_manifest = json.loads((SUPPORT / "support_manifest.json").read_text())
archive_manifest = json.loads((ARCHIVE / "archive_manifest.json").read_text())
assert int(support_manifest["bkg4_abi"]) == 4
assert support_manifest["fast_math"] is False
assert support_manifest["cuda_result_abi"] == "B4EC1/v1"
assert archive_manifest.get("processed_dates"), "replay archive has no processed dates"
print({
    "support_source_commit": support_manifest["source_commit"],
    "base_parent": support_manifest["base_parent_combined_identity"],
    "archive_dates": len(archive_manifest["processed_dates"]),
    "archive_manifest": archive_manifest["manifest_sha256"],
})
'''),
        code(r'''
result = run_builder(
    archive_root=ARCHIVE,
    support_root=SUPPORT,
    prior_corpus_root=PRIOR_CORPUS,
    stage_root=STAGE,
    tmp_root=TMP,
    max_dates=MAX_DATES,
    max_replays_per_day=MAX_REPLAYS_PER_DAY,
    max_cases_per_day=MAX_CASES_PER_DAY,
    seed_parent_count=SEED_PARENT_COUNT,
    candidates_per_parent=CANDIDATES_PER_PARENT,
    replay_converter_workers=REPLAY_CONVERTER_WORKERS,
    max_cuda_devices=MAX_CUDA_DEVICES,
    target_cuda_games=TARGET_CUDA_GAMES,
    eval_gpu_memory_fraction=EVAL_GPU_MEMORY_FRACTION,
)
processed_this_run = result["processed_this_run"]
manifest = result["manifest"]
print(json.dumps({
    "processed_this_run": processed_this_run,
    "counts": result["counts"],
    "manifest_sha256": manifest["manifest_sha256"],
}, indent=2))
'''),
        code(r'''
# Publication is the final transaction boundary. The corpus is revalidated immediately
# before creating a new version of the same persistent dataset.
validate_corpus(STAGE)
if PUBLISH and processed_this_run:
    last_date = max(x["date"] for x in processed_this_run)
    msg = f"CUDA EditFormer corpus through {last_date}; +{len(processed_this_run)} replay day(s)"
    run_cmd(["kaggle", "datasets", "version", "-p", STAGE, "-m", msg, "-q", "-t", "-r", "skip"])
    print("PUBLISHED", CORPUS_DATASET, msg)
elif not processed_this_run:
    print("UP TO DATE: no corpus dataset version created")
else:
    print("PUBLISH=False: validated corpus left in", STAGE)
'''),
        code(r'''
final_manifest = validate_corpus(STAGE)
print(json.dumps({
    "corpus_dataset": CORPUS_DATASET,
    "processed_dates_total": len(final_manifest["processed_dates"]),
    "new_dates": [x["date"] for x in processed_this_run],
    "counts": final_manifest["counts"],
    "cuda_engine_identity": final_manifest["cuda_engine_identity"],
    "candidate_generator_version": final_manifest["candidate_generator_version"],
    "edit_grammar_version": final_manifest["edit_grammar_version"],
    "manifest_sha256": final_manifest["manifest_sha256"],
}, indent=2))
'''),
    ]
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    return json.dumps(nb, ensure_ascii=False, sort_keys=True, indent=1) + "\n"


def render_kernel_metadata() -> str:
    meta = {
        "id": "ashok205/kaggriculture-bonsai-edit-corpus-builder",
        "title": "Kaggriculture Bonsai Edit Corpus Builder",
        "code_file": "kaggriculture_edit_corpus_builder.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": True,
        "dataset_sources": [
            "ashok205/kaggriculture-top10-replay-archive",
            "ashok205/kaggriculture-edit-corpus-support",
            "ashok205/kaggriculture-bonsai-edit-corpus",
        ],
        "kernel_sources": [],
        "competition_sources": [],
        "model_sources": [],
    }
    return json.dumps(meta, ensure_ascii=False, sort_keys=True, indent=2) + "\n"


def validate_notebook(text: str) -> None:
    nb = json.loads(text)
    if nb.get("nbformat") != 4:
        raise RuntimeError("unexpected notebook format")
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"<cell-{i}>", "exec")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    notebook = render_notebook()
    metadata = render_kernel_metadata()
    validate_notebook(notebook)
    if args.check:
        if not OUT_NOTEBOOK.exists() or OUT_NOTEBOOK.read_text(encoding="utf-8") != notebook:
            raise SystemExit(f"generated notebook is stale: {OUT_NOTEBOOK}")
        if not OUT_META.exists() or OUT_META.read_text(encoding="utf-8") != metadata:
            raise SystemExit(f"generated kernel metadata is stale: {OUT_META}")
        print("notebook generation check passed")
        return 0
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_NOTEBOOK.write_text(notebook, encoding="utf-8")
    OUT_META.write_text(metadata, encoding="utf-8")
    print(OUT_NOTEBOOK)
    print(OUT_META)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
