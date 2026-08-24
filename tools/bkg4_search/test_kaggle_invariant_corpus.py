from __future__ import annotations

import gzip
import json
import subprocess
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path

import numpy as np


class KaggleInvariantCorpusTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            from tools.bkg4_search import kaggle_invariant_corpus as corpus
            from tools.bkg4_search import package_kaggle_invariant_corpus_support as package
        except ImportError as exc:
            raise AssertionError(f"Kaggle B4IM corpus deployment is not implemented: {exc}") from exc
        cls.corpus = corpus
        cls.package = package

    def test_support_inventory_has_exact_runtime_contract(self):
        required = set(self.package.REQUIRED_PATHS)
        self.assertIn("cuda-engine/src/bkg4_invariant_edit_eval.cu", required)
        self.assertIn("bonsai-kg/include/bkg4_invariant_module.hpp", required)
        self.assertIn("tools/bkg4_search/generate_invariant_specialists.py", required)
        self.assertIn("tools/bkg4_search/kaggle_invariant_corpus_runner.py", required)
        self.assertIn("tools/bkg4_search/merge_invariant_corpus_shards.py", required)
        self.assertIn("tools/bkg4_search/bonsai_editformer/invariant_data.py", required)
        self.assertEqual(list(self.package.REQUIRED_PATHS), sorted(required))
        self.assertFalse(any(path.startswith("scratch/") for path in required))

    def test_safe_extract_rejects_traversal_and_links(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            archive = root / "bad.tar"
            payload = root / "payload"
            payload.write_text("x", encoding="utf-8")
            with tarfile.open(archive, "w") as tf:
                tf.add(payload, arcname="../escape")
            with self.assertRaisesRegex(ValueError, "unsafe"):
                self.corpus.safe_extract_tar(archive, root / "out")

    def test_streaming_shard_roundtrip_and_arrays(self):
        source = self.package.VALIDATED_CORPUS
        self.assertTrue(source.is_file())
        rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines()[:8]]
        with tempfile.TemporaryDirectory() as td:
            shard = Path(td) / "shard-0000"
            manifest = self.corpus.write_validated_shard(
                shard, rows, island=0, shard_id=0, seed=17,
                screen_panel={"id": "screen", "cases": 64},
                label_panel={"id": "label", "cases": 400},
                runtime={"candidate_abi": "B4IM1/v1", "evaluator_abi": "B4IC1/v1"},
            )
            self.assertEqual(manifest["counts"]["rows"], 8)
            self.assertTrue((shard / "metadata.jsonl.gz").is_file())
            self.assertTrue((shard / "training_arrays.npz").is_file())
            with np.load(shard / "training_arrays.npz") as arrays:
                self.assertEqual(arrays["semantic_id"].shape, (8,))
                self.assertEqual(arrays["b4im_cat"].shape[0], 8)
            with gzip.open(shard / "index.jsonl.gz", "rt", encoding="utf-8") as f:
                index = [json.loads(line) for line in f]
            self.assertEqual(len(index), 8)
            self.assertEqual(len({row["semantic_id"] for row in index}), 8)

    def test_scalable_split_matches_reference_on_validated_384(self):
        source = self.package.VALIDATED_CORPUS
        rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(rows), 384)
        from tools.bkg4_search.bonsai_editformer.invariant_data import invariant_grouped_split

        reference = invariant_grouped_split(rows)
        compact = self.corpus.scalable_grouped_split(self.corpus.index_rows(rows))
        self.assertEqual(tuple(compact.train_indices), reference.train_indices)
        self.assertEqual(tuple(compact.validation_indices), reference.validation_indices)
        self.assertFalse(set(compact.train_skeletons) & set(compact.validation_skeletons))

    def test_global_merge_deduplicates_and_preserves_secondary_panels(self):
        source = self.package.VALIDATED_CORPUS
        row = json.loads(source.read_text(encoding="utf-8").splitlines()[0])
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for island, panel in ((0, "panel-a"), (1, "panel-b")):
                self.corpus.write_validated_shard(
                    root / f"island{island}" / "shard-0000", [row], island=island,
                    shard_id=0, seed=100 + island,
                    screen_panel={"id": "screen", "cases": 64},
                    label_panel={"id": panel, "cases": 400},
                    runtime={"candidate_abi": "B4IM1/v1", "evaluator_abi": "B4IC1/v1"},
                )
            report = self.corpus.merge_shards(root)
            self.assertEqual(report["deduplicated_rows"], 1)
            self.assertEqual(report["duplicates"], 1)
            with gzip.open(root / "global_index.jsonl.gz", "rt", encoding="utf-8") as f:
                merged = json.loads(next(f))
            self.assertEqual(len(merged["secondary_evidence"]), 1)

    def test_notebook_is_generated_and_code_cells_compile(self):
        from tools.bkg4_search import build_kaggle_invariant_corpus_notebook as builder

        notebook = json.loads(builder.render_notebook())
        code = "\n".join("".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code")
        for marker in (
            "TOTAL_WALLTIME_SECONDS", "FINAL_RESERVE_SECONDS", "sm_75",
            "CUDA_VISIBLE_DEVICES", "Select Kaggle T4 x2 accelerator and restart",
            "PUBLISH_CORPUS", "support_manifest.json", "run-smoke", "calibrate",
            "worker", "merge", "nvidia-smi",
        ):
            self.assertIn(marker, code)
        builder.validate_notebook(json.dumps(notebook))

    def test_packaged_runtime_imports_without_repository(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "dataset"
            manifest = self.package.build_support_dataset(out)
            runtime = Path(td) / "runtime"
            self.corpus.safe_extract_tar(out / manifest["bundle_file"], runtime)
            smoke_tapes = [line.split("\t", 1)[0] for line in (runtime / "smoke/cases.tsv").read_text().splitlines()]
            self.assertTrue(smoke_tapes)
            self.assertFalse(any(Path(path).is_absolute() for path in smoke_tapes))
            cmd = [
                sys.executable, "-c",
                "import sys; sys.path.insert(0, sys.argv[1]); "
                "from tools.bkg4_search.kaggle_invariant_corpus_runner import main; "
                "from tools.bkg4_search.invariant_factory import load_offline_specialist_bank; "
                "import bonsai_kg_v4; print('B4IM_SUPPORT_IMPORT_OK')",
                str(runtime),
            ]
            proc = subprocess.run(cmd, cwd=td, text=True, capture_output=True)
            self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
            self.assertIn("B4IM_SUPPORT_IMPORT_OK", proc.stdout)


if __name__ == "__main__":
    unittest.main()
