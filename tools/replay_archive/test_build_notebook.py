import json
import unittest

from tools.replay_archive.build_notebook import render_kernel_metadata, render_notebook


class NotebookBuilderTests(unittest.TestCase):
    def test_notebook_render_is_deterministic_and_compilable(self):
        a = render_notebook()
        b = render_notebook()
        self.assertEqual(a, b)
        nb = json.loads(a)
        self.assertEqual(nb["nbformat"], 4)
        self.assertGreaterEqual(len(nb["cells"]), 6)
        for cell in nb["cells"]:
            if cell["cell_type"] == "code":
                compile("".join(cell["source"]), "<notebook-cell>", "exec")

    def test_kernel_metadata_is_private_cpu_and_has_required_sources(self):
        meta = json.loads(render_kernel_metadata())
        self.assertFalse(meta["enable_gpu"])
        self.assertTrue(meta["enable_internet"])
        self.assertTrue(meta["is_private"])
        self.assertIn("kaggle/kaggriculture-episodes-index", meta["dataset_sources"])
        self.assertIn("ashok205/kaggriculture-top10-replay-archive", meta["dataset_sources"])
        self.assertEqual(meta["competition_sources"], ["kaggriculture"])

    def test_index_dataset_is_required_as_attached_input_without_download_fallback(self):
        text = render_notebook()
        self.assertIn("/kaggle/input/datasets/kaggle/kaggriculture-episodes-index/manifest.csv", text)
        self.assertNotIn("/kaggle/input/kaggriculture-episodes-index/manifest.csv", text)
        self.assertIn("episodes-index is not attached", text)
        self.assertNotIn('\"datasets\", \"download\", INDEX_DATASET', text)


if __name__ == "__main__":
    unittest.main()
