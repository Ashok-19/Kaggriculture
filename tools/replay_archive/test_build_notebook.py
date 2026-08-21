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

    def test_kernel_metadata_is_private_cpu_and_minimal(self):
        meta = json.loads(render_kernel_metadata())
        self.assertFalse(meta["enable_gpu"])
        self.assertTrue(meta["enable_internet"])
        self.assertTrue(meta["is_private"])
        self.assertIn("kaggle/kaggriculture-episodes-index", meta["dataset_sources"])
        self.assertIn("ashok205/kaggriculture-top10-replay-archive", meta["dataset_sources"])
        self.assertNotIn("kaggle/meta-kaggle", meta["dataset_sources"])
        self.assertEqual(meta["competition_sources"], ["kaggriculture"])

    def test_index_dataset_uses_exact_attached_mount(self):
        text = render_notebook()
        self.assertIn("/kaggle/input/datasets/kaggle/kaggriculture-episodes-index/manifest.csv", text)
        self.assertNotIn("/kaggle/input/kaggriculture-episodes-index/manifest.csv", text)
        self.assertIn("episodes-index is not attached", text)

    def test_broken_metadata_dependencies_are_removed(self):
        text = render_notebook()
        self.assertNotIn("meta-kaggle", text.lower())
        self.assertNotIn("EpisodeAgents.csv", text)
        self.assertNotIn("EpisodeService", text)
        self.assertNotIn("kaggle_environments", text)
        self.assertNotIn("list_episodes(", text)

    def test_notebook_uses_daily_zip_header_scan_and_deletes_temp_day(self):
        nb = json.loads(render_notebook())
        text = "\n".join("".join(cell["source"]) for cell in nb["cells"] if cell["cell_type"] == "code")
        self.assertIn("zipfile.ZipFile", text)
        self.assertIn("HEADER_SCAN_BYTES", text)
        self.assertIn("extract_team_names_from_prefix", text)
        self.assertIn("write_replay_shard_from_zip", text)
        self.assertIn("shutil.rmtree(day_tmp", text)
        self.assertIn('["kaggle", "datasets", "download", f"kaggle/{slug}"', text)
        self.assertNotIn('"--unzip"', text)

    def test_notebook_skips_unavailable_replays_and_records_counts(self):
        nb = json.loads(render_notebook())
        text = "\n".join("".join(cell["source"]) for cell in nb["cells"] if cell["cell_type"] == "code")
        self.assertIn("scan_team_names_from_zip", text)
        self.assertIn("read_replay_bytes_from_zip", text)
        self.assertIn("header_skips", text)
        self.assertIn("body_skips", text)
        self.assertIn("archived_ids", text)
        self.assertIn("source_unavailable_episode_count", text)
        self.assertNotIn("daily ZIP missing replay JSON for episode", text)
        self.assertNotIn("selected replay {episode_id} missing from ZIP", text)


if __name__ == "__main__":
    unittest.main()
