from __future__ import annotations

import json
import unittest


class EditCorpusNotebookBuilderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            from tools.edit_corpus.build_notebook import render_kernel_metadata, render_notebook
        except ImportError as exc:
            raise AssertionError(f"notebook builder is not implemented yet: {exc}") from exc
        cls.render_notebook = staticmethod(render_notebook)
        cls.render_kernel_metadata = staticmethod(render_kernel_metadata)

    def test_notebook_is_deterministic_and_all_code_cells_compile(self):
        a = self.render_notebook()
        b = self.render_notebook()
        self.assertEqual(a, b)
        nb = json.loads(a)
        self.assertEqual(nb["nbformat"], 4)
        self.assertGreaterEqual(len(nb["cells"]), 5)
        for i, cell in enumerate(nb["cells"]):
            if cell["cell_type"] == "code":
                compile("".join(cell["source"]), f"<cell-{i}>", "exec")

    def test_kernel_metadata_uses_gpu_and_exact_three_dataset_inputs(self):
        meta = json.loads(self.render_kernel_metadata())
        self.assertTrue(meta["enable_gpu"])
        self.assertEqual(meta["title"], "Kaggriculture Bonsai Edit Corpus Builder")
        self.assertTrue(meta["id"].endswith("/kaggriculture-bonsai-edit-corpus-builder"))
        self.assertTrue(meta["enable_internet"])
        self.assertTrue(meta["is_private"])
        self.assertEqual(meta["competition_sources"], [])
        self.assertEqual(set(meta["dataset_sources"]), {
            "ashok205/kaggriculture-top10-replay-archive",
            "ashok205/kaggriculture-edit-corpus-support",
            "ashok205/kaggriculture-bonsai-edit-corpus",
        })

    def test_notebook_mounts_support_and_runs_incremental_gpu_builder(self):
        nb = json.loads(self.render_notebook())
        text = "\n".join("".join(cell["source"]) for cell in nb["cells"] if cell["cell_type"] == "code")
        for needle in (
            "support_manifest.json",
            "archive_manifest.json",
            "corpus_manifest.json",
            "run_builder(",
            "max_cases_per_day",
            "screen_cases_per_day",
            "candidates_per_parent",
            "replay_converter_workers",
            "max_cuda_devices",
            "screen_target_cuda_games",
            "final_retained_edits_per_day",
            "eval_gpu_memory_fraction",
            "processed_this_run",
            "support_bundle.tar",
            'SUPPORT_INPUT / "support_bundle"',
            "shutil.copytree",
            "kaggriculture_edit_corpus_support_runtime",
        ):
            self.assertIn(needle, text)
        self.assertNotIn("kaggle_environments", text)
        self.assertNotIn("20M", text)
        self.assertNotIn("train_state", text)

    def test_notebook_publishes_only_after_validated_new_dates(self):
        nb = json.loads(self.render_notebook())
        text = "\n".join("".join(cell["source"]) for cell in nb["cells"] if cell["cell_type"] == "code")
        self.assertIn("validate_corpus(STAGE)", text)
        self.assertIn('"datasets", "version"', text)
        self.assertIn("if PUBLISH and processed_this_run", text)
        self.assertIn("UP TO DATE", text)

    def test_notebook_has_small_smoke_switch_without_changing_default_full_incremental_run(self):
        text = self.render_notebook()
        self.assertIn("SMOKE_ONLY = False", text)
        self.assertIn("MAX_DATES = 1 if SMOKE_ONLY else None", text)
        self.assertIn("MAX_REPLAYS_PER_DAY = 4 if SMOKE_ONLY else None", text)
        self.assertIn("FINAL_CASES_PER_DAY = 8 if SMOKE_ONLY else 512", text)
        self.assertIn("SCREEN_CASES_PER_DAY = 4 if SMOKE_ONLY else 128", text)
        self.assertIn("SEED_PARENT_COUNT = 2 if SMOKE_ONLY else 8", text)
        self.assertIn("CANDIDATES_PER_PARENT = 32 if SMOKE_ONLY else 4_096", text)
        self.assertIn("REPLAY_CONVERTER_WORKERS = 1 if SMOKE_ONLY else 2", text)
        self.assertIn("MAX_CUDA_DEVICES = 1 if SMOKE_ONLY else 2", text)
        self.assertIn("SCREEN_TARGET_CUDA_GAMES = 512 if SMOKE_ONLY else 4_000_000", text)
        self.assertIn("FINAL_RETAINED_EDITS_PER_DAY = 8 if SMOKE_ONLY else 4_096", text)
        self.assertNotIn("EVAL_CHUNK_GENOMES =", text)


if __name__ == "__main__":
    unittest.main()
