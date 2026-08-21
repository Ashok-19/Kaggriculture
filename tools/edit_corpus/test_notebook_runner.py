from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock


class _Genome:
    def __init__(self, name: str):
        self.name = name

    def semantic_id(self) -> str:
        return self.name


class NotebookRunnerPerformanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from tools.edit_corpus import notebook_runner
        cls.r = notebook_runner

    def test_discover_cuda_devices_caps_at_two_logical_gpus(self):
        probe = mock.Mock(returncode=0, stdout="0\n1\n2\n", stderr="")
        with mock.patch("tools.edit_corpus.notebook_runner.subprocess.run", return_value=probe):
            self.assertEqual(self.r.discover_cuda_devices(max_devices=2), (0, 1))

    def test_evaluate_genomes_uses_one_process_per_gpu_lane_not_one_per_chunk(self):
        genomes = [_Genome(f"g{i}") for i in range(6)]
        packed_counts: dict[str, int] = {}
        launches: list[tuple[str, str]] = []

        def fake_pack(chunk, path):
            packed_counts[str(path)] = len(chunk)
            Path(path).write_bytes(b"fake")

        def fake_run(cmd, *, cwd=None, timeout=7200):
            argv = [str(x) for x in cmd]
            launches.append((argv[argv.index("--device") + 1], argv[argv.index("--genomes") + 1]))
            Path(argv[argv.index("--out") + 1]).write_bytes(b"fake")
            return mock.Mock(returncode=0, stdout="", stderr="")

        def fake_results(path):
            genome_path = str(path).replace("results_", "genomes_").replace(".b4ec", ".b4gm")
            return [[f"result-{i}"] for i in range(packed_counts[genome_path])]

        assets = {
            "controller": Path("controller"),
            "routes": Path("routes"),
            "legacy_routes": Path("legacy"),
            "strategic_route_slot": 19,
        }
        with tempfile.TemporaryDirectory() as td, \
             mock.patch.object(self.r, "pack_genomes", side_effect=fake_pack), \
             mock.patch.object(self.r, "run", side_effect=fake_run), \
             mock.patch.object(self.r, "load_edit_results", side_effect=fake_results):
            out = self.r._evaluate_genomes(
                Path("edit_eval"), assets, Path("cases.tsv"), genomes, Path(td),
                chunk_size=1,
                cuda_devices=(0, 1),
            )

        self.assertEqual(len(out), 6)
        self.assertEqual(len(launches), 2)
        self.assertEqual({x[0] for x in launches}, {"0", "1"})
        self.assertEqual(sorted(packed_counts.values()), [3, 3])

    def test_candidate_count_is_derived_from_case_count_and_target_games(self):
        self.assertEqual(
            self.r.adaptive_candidates_per_parent(
                selected_cases=512,
                seed_parent_count=8,
                cap=768,
                target_games=1_500_000,
            ),
            326,
        )
        self.assertEqual(
            self.r.adaptive_candidates_per_parent(
                selected_cases=256,
                seed_parent_count=8,
                cap=768,
                target_games=1_500_000,
            ),
            652,
        )
        self.assertEqual(
            self.r.adaptive_candidates_per_parent(
                selected_cases=1024,
                seed_parent_count=8,
                cap=768,
                target_games=1_500_000,
            ),
            163,
        )

    def test_cuda_source_has_memory_derived_in_process_chunking(self):
        source = Path("cuda-engine/src/bkg4_edit_corpus_eval.cu").read_text(encoding="utf-8")
        self.assertIn("cudaMemGetInfo", source)
        self.assertIn("memory_fraction", source)
        self.assertIn("chunk_genomes", source)
        self.assertIn("chunks=", source)


if __name__ == "__main__":
    unittest.main()
