from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path


class SupportDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            from tools.edit_corpus import build_support_dataset as b
        except ImportError as exc:
            raise AssertionError(f"support bundle packager is not implemented yet: {exc}") from exc
        cls.b = b

    def test_required_inventory_is_explicit_and_contains_runtime_assets(self):
        paths = list(self.b.REQUIRED_PATHS)
        self.assertEqual(paths, sorted(paths))
        required = {
            "cuda-engine/src/bkg4_edit_corpus_eval.cu",
            "cuda-engine/src/bkg4_information_state_capture.cu",
            "cuda-engine/include/runtime.cuh",
            "cuda-engine/include/tape_io.hpp",
            "bonsai-kg/include/bonsai_kg_v4.hpp",
            "bonsai-kg/cuda/kag_adapter_v4.cuh",
            "bonsai-kg/integrations/bkg4_champion/cuda_compat.cuh",
            "bonsai-kg/integrations/bkg4_champion/champions/w33_s172_expansion_v1/controller.bkgc",
            "bonsai-kg/integrations/bkg4_champion/champions/w33_s172_expansion_v1/controller.json",
            "bonsai-kg/integrations/bkg4_champion/champions/w33_s172_expansion_v1/routes.bkv6",
            "bonsai-kg/integrations/bkg4_champion/champions/w33_s172_expansion_v1/legacy_routes.bkv6",
            "tools/bkg4_search/bonsai_editformer/actor_grammar.py",
            "tools/bkg4_search/module_codec.py",
            "tools/edit_corpus/corpus.py",
            "tools/edit_corpus/candidates.py",
            "tools/edit_corpus/cuda_results.py",
            "tools/edit_corpus/kaggle_runtime.py",
            "tools/edit_corpus/pipeline.py",
            "tools/edit_corpus/notebook_runner.py",
        }
        self.assertTrue(required.issubset(set(paths)))
        self.assertFalse(any(p.startswith("scratch/") for p in paths))
        self.assertFalse(any(p.startswith("notebooks/") for p in paths))

    def test_build_bundle_copies_only_allowlist_and_writes_manifest(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "bundle"
            manifest = self.b.build_bundle(out, force=False)
            self.assertEqual(manifest["schema_version"], self.b.SUPPORT_SCHEMA_VERSION)
            self.assertEqual(manifest["paths"], list(self.b.REQUIRED_PATHS))
            self.assertEqual(sorted(manifest["files"]), list(self.b.REQUIRED_PATHS))
            for rel in self.b.REQUIRED_PATHS:
                self.assertTrue((out / rel).is_file(), rel)
            copied = sorted(
                p.relative_to(out).as_posix()
                for p in out.rglob("*")
                if p.is_file() and p.name not in {"support_manifest.json", "dataset-metadata.json"}
            )
            self.assertEqual(copied, list(self.b.REQUIRED_PATHS))
            self.assertEqual(json.loads((out / "dataset-metadata.json").read_text())["id"], self.b.SUPPORT_DATASET)
            persisted = json.loads((out / "support_manifest.json").read_text())
            self.assertEqual(persisted, manifest)
            self.assertGreater(manifest["total_bytes"], 0)
            self.assertEqual(manifest["bkg4_abi"], 4)
            self.assertEqual(manifest["edit_grammar"], "bonsai_edit_actor_grammar_v1")


if __name__ == "__main__":
    unittest.main()
