from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tools.bkg4_search.rlrmb.selfplay_bank import SelfplayBank, BankEntry
from tools.bkg4_search.rlrmb.selfplay_generator import SelfplayPolicy


class SelfplayBankTests(unittest.TestCase):
    def test_empty_bank(self):
        with tempfile.TemporaryDirectory() as d:
            bank = SelfplayBank(Path(d) / "bank")
            self.assertEqual(bank.size, 0)
            self.assertEqual(bank.stats(), {"size": 0})

    def test_generate_populates_bank(self):
        with tempfile.TemporaryDirectory() as d:
            bank = SelfplayBank(Path(d) / "bank")
            result = bank.generate(
                num_seeds=2,
                policies=[SelfplayPolicy("mirror", "champion")],
                seats=[0],
                workers=1,
            )
            self.assertEqual(result["new_tapes"], 2)
            self.assertEqual(bank.size, 2)
            self.assertTrue(bank.index_path.exists())

    def test_generate_avoids_duplicates(self):
        with tempfile.TemporaryDirectory() as d:
            bank = SelfplayBank(Path(d) / "bank")
            bank.generate(
                num_seeds=2,
                policies=[SelfplayPolicy("mirror", "champion")],
                seats=[0], workers=1,
            )
            result = bank.generate(
                num_seeds=2,
                policies=[SelfplayPolicy("mirror", "champion")],
                seats=[0], workers=1, rng_seed=99,
            )
            self.assertEqual(result["new_tapes"], 2)
            self.assertEqual(bank.size, 4)

    def test_persistence(self):
        with tempfile.TemporaryDirectory() as d:
            bank_dir = Path(d) / "bank"
            bank = SelfplayBank(bank_dir)
            bank.generate(
                num_seeds=1,
                policies=[SelfplayPolicy("mirror", "champion")],
                seats=[0], workers=1,
            )
            bank2 = SelfplayBank(bank_dir)
            self.assertEqual(bank2.size, 1)

    def test_assemble_panel(self):
        with tempfile.TemporaryDirectory() as d:
            bank = SelfplayBank(Path(d) / "bank")
            bank.generate(
                num_seeds=2,
                policies=[SelfplayPolicy("mirror", "champion")],
                seats=[0], workers=1,
            )
            panel = Path(d) / "panel.tsv"
            count = bank.assemble_panel(panel)
            self.assertEqual(count, 2)
            lines = panel.read_text().strip().split("\n")
            self.assertEqual(len(lines), 2)
            for line in lines:
                fields = line.split("\t")
                self.assertEqual(len(fields), 7)

    def test_stats(self):
        with tempfile.TemporaryDirectory() as d:
            bank = SelfplayBank(Path(d) / "bank")
            bank.generate(
                num_seeds=2,
                policies=[SelfplayPolicy("mirror", "champion")],
                seats=[0], workers=1,
            )
            s = bank.stats()
            self.assertEqual(s["size"], 2)
            self.assertEqual(s["unique_seeds"], 2)
            self.assertIn("mirror", s["policies"])

    def test_refresh_adds_new_seeds(self):
        with tempfile.TemporaryDirectory() as d:
            bank = SelfplayBank(Path(d) / "bank")
            bank.generate(
                num_seeds=2,
                policies=[SelfplayPolicy("mirror", "champion")],
                seats=[0], workers=1,
            )
            result = bank.refresh(
                num_new_seeds=2,
                policies=[SelfplayPolicy("mirror", "champion")],
                seats=[0], workers=1, generation=1,
            )
            self.assertEqual(result["new_tapes"], 2)
            self.assertEqual(bank.size, 4)

    def test_legacy_entries_gain_regime_provenance_on_load(self):
        with tempfile.TemporaryDirectory() as d:
            bank_dir = Path(d) / "bank"
            bank_dir.mkdir()
            (bank_dir / "index.json").write_text(json.dumps({"entries": [{
                "tape_path": "old.json", "seat": 0, "seed": 1,
                "policy_name": "bkg4_candidate", "opponent_policy": "bkg4_noisy_010",
                "margin": 1.0, "reward": 2.0, "opponent_reward": 1.0,
                "generation": 0, "bkg4_version": "current",
            }]}))
            bank = SelfplayBank(bank_dir)
            self.assertEqual(bank.entries[0].policy_family, "bkg4_mutation")
            self.assertEqual(bank.entries[0].regime_metadata["legacy_schema"], 1)

    def test_assemble_panel_rejects_missing_tape_payload(self):
        with tempfile.TemporaryDirectory() as d:
            bank = SelfplayBank(Path(d) / "bank")
            bank._entries = [BankEntry(
                tape_path=str(Path(d) / "missing.json"), seat=0, seed=1,
                policy_name="bkg4_candidate", opponent_policy="mirror",
                margin=0.0, reward=0.0, opponent_reward=0.0,
                generation=0, bkg4_version="current",
            )]
            with self.assertRaisesRegex(ValueError, "missing tape payload"):
                bank.assemble_panel(Path(d) / "panel.tsv")


if __name__ == "__main__":
    unittest.main()
