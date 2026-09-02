from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tools.bkg4_search.rlrmb.selfplay_generator import (
    DEFAULT_POLICIES,
    SelfplayPolicy,
    generate_diverse_seeds,
    generate_selfplay_family,
    generate_selfplay_tapes,
    write_target_panel,
)


class SelfplayGeneratorTests(unittest.TestCase):
    def test_generate_diverse_seeds_unique(self):
        seeds = generate_diverse_seeds(100, rng_seed=42)
        self.assertEqual(len(seeds), 100)
        self.assertEqual(len(set(seeds)), 100)

    def test_generate_diverse_seeds_avoids_existing(self):
        existing = [1, 2, 3]
        seeds = generate_diverse_seeds(5, rng_seed=42, existing=existing)
        self.assertEqual(len(seeds), 5)
        self.assertFalse(set(seeds) & set(existing))

    def test_generate_diverse_seeds_deterministic(self):
        a = generate_diverse_seeds(20, rng_seed=7)
        b = generate_diverse_seeds(20, rng_seed=7)
        self.assertEqual(a, b)

    def test_generate_selfplay_tapes_produces_valid_tapes(self):
        with tempfile.TemporaryDirectory() as d:
            output = Path(d)
            tapes = generate_selfplay_tapes(
                output,
                seeds=[42],
                policies=[SelfplayPolicy("mirror", "champion")],
                seats=[0],
                rng_seed=0,
            )
            self.assertEqual(len(tapes), 1)
            tape = tapes[0]
            self.assertTrue(tape.path.exists())
            doc = json.loads(tape.path.read_text())
            self.assertIn("source", doc)
            self.assertIn("actions", doc)
            self.assertEqual(doc["source"]["kind"], "selfplay")
            self.assertEqual(doc["source"]["seed"], 42)
            self.assertEqual(doc["source"]["seat"], 0)
            self.assertIsInstance(doc["actions"], list)
            self.assertGreater(len(doc["actions"]), 100)

    def test_write_target_panel_format(self):
        with tempfile.TemporaryDirectory() as d:
            output = Path(d)
            tapes = generate_selfplay_tapes(
                output,
                seeds=[42],
                policies=[SelfplayPolicy("mirror", "champion")],
                seats=[0],
                rng_seed=0,
            )
            panel = output / "panel.tsv"
            count = write_target_panel(tapes, panel)
            self.assertEqual(count, 1)
            lines = panel.read_text().strip().split("\n")
            self.assertEqual(len(lines), 1)
            fields = lines[0].split("\t")
            self.assertEqual(len(fields), 7)
            self.assertTrue(Path(fields[0]).exists())
            self.assertEqual(int(fields[1]), 0)
            self.assertEqual(int(fields[2]), 42)
            self.assertEqual(int(fields[3]), 0)  # family
            self.assertEqual(int(fields[4]), 0)  # protected_flag
            self.assertEqual(int(fields[5]), 0)  # own_history
            self.assertEqual(int(fields[6]), 0)  # role

    def test_generate_selfplay_family_end_to_end(self):
        with tempfile.TemporaryDirectory() as d:
            output = Path(d) / "family"
            panel_path, manifest = generate_selfplay_family(
                output,
                num_seeds=2,
                policies=[SelfplayPolicy("mirror", "champion")],
                seats=[0],
                rng_seed=0,
            )
            self.assertTrue(panel_path.exists())
            self.assertEqual(manifest["source"], "selfplay")
            self.assertEqual(manifest["num_seeds"], 2)
            self.assertEqual(manifest["num_tapes"], 2)
            lines = panel_path.read_text().strip().split("\n")
            self.assertEqual(len(lines), 2)
            for line in lines:
                tape_path = line.split("\t")[0]
                self.assertTrue(Path(tape_path).exists())

    def test_policy_diversity_creates_different_tapes(self):
        with tempfile.TemporaryDirectory() as d:
            output = Path(d)
            tapes = generate_selfplay_tapes(
                output,
                seeds=[42],
                policies=[
                    SelfplayPolicy("mirror", "champion"),
                    SelfplayPolicy("noisy", "noisy", noise_rate=0.3),
                ],
                seats=[0],
                rng_seed=0,
            )
            self.assertEqual(len(tapes), 2)
            self.assertNotEqual(tapes[0].opponent_policy, tapes[1].opponent_policy)

    def test_selfplay_deterministic(self):
        with tempfile.TemporaryDirectory() as d:
            output1 = Path(d) / "run1"
            output2 = Path(d) / "run2"
            policies = [SelfplayPolicy("mirror", "champion")]
            t1 = generate_selfplay_tapes(output1, seeds=[42], policies=policies, seats=[0], rng_seed=0)
            t2 = generate_selfplay_tapes(output2, seeds=[42], policies=policies, seats=[0], rng_seed=0)
            self.assertEqual(t1[0].margin, t2[0].margin)
            self.assertEqual(t1[0].reward, t2[0].reward)


if __name__ == "__main__":
    unittest.main()
