from __future__ import annotations

import json
import random
import tempfile
import unittest
from pathlib import Path

from tools.bkg4_search.rlrmb.selfplay_generator import (
    DEFAULT_POLICIES,
    SelfplayPolicy,
    apply_selfplay_policy,
    generate_diverse_seeds,
    generate_selfplay_family,
    generate_selfplay_tapes,
    write_target_panel,
)


def _observation(*, step: int = 0, tile=None, seeds=None, shed=None, money: int = 5000):
    return {
        "player": 0,
        "step": step,
        "farms": [{
            "money": money,
            "farmer": [0, 0],
            "tiles": [[tile]],
            "unlocked_quadrants": ["NW"],
        }],
        "private": {"seeds": seeds or {}, "shed": shed or {}},
    }


class SelfplayGeneratorTests(unittest.TestCase):
    def test_generate_diverse_seeds_unique(self):
        seeds = generate_diverse_seeds(100, rng_seed=42)
        self.assertEqual(len(seeds), 100)
        self.assertEqual(len(set(seeds)), 100)

    def test_generate_diverse_seeds_avoids_existing(self):
        seeds = generate_diverse_seeds(5, rng_seed=42, existing=[1, 2, 3])
        self.assertFalse(set(seeds) & {1, 2, 3})

    def test_generate_diverse_seeds_deterministic(self):
        self.assertEqual(
            generate_diverse_seeds(20, rng_seed=7),
            generate_diverse_seeds(20, rng_seed=7),
        )

    def test_generate_selfplay_tapes_produces_valid_tapes(self):
        with tempfile.TemporaryDirectory() as d:
            tapes = generate_selfplay_tapes(
                Path(d), seeds=[42], policies=[SelfplayPolicy("mirror", "champion")],
                seats=[0], rng_seed=0,
            )
            doc = json.loads(tapes[0].path.read_text())
            self.assertEqual(doc["source"]["kind"], "selfplay")
            self.assertEqual(doc["source"]["schema_version"], 2)
            self.assertGreater(len(doc["actions"]), 100)

    def test_write_target_panel_format(self):
        with tempfile.TemporaryDirectory() as d:
            output = Path(d)
            tapes = generate_selfplay_tapes(
                output, seeds=[42], policies=[SelfplayPolicy("mirror", "champion")],
                seats=[0], rng_seed=0,
            )
            panel = output / "panel.tsv"
            self.assertEqual(write_target_panel(tapes, panel), 1)
            self.assertEqual(len(panel.read_text().strip().split("\t")), 7)

    def test_generate_selfplay_family_end_to_end(self):
        with tempfile.TemporaryDirectory() as d:
            panel, manifest = generate_selfplay_family(
                Path(d) / "family", num_seeds=2,
                policies=[SelfplayPolicy("mirror", "champion")], seats=[0], rng_seed=0,
            )
            self.assertTrue(panel.exists())
            self.assertEqual(manifest["num_tapes"], 2)

    def test_policy_diversity_creates_different_tapes(self):
        with tempfile.TemporaryDirectory() as d:
            tapes = generate_selfplay_tapes(
                Path(d), seeds=[42],
                policies=[SelfplayPolicy("mirror", "champion"), SelfplayPolicy("noisy", "noisy", noise_rate=0.3)],
                seats=[0], rng_seed=0,
            )
            self.assertNotEqual(tapes[0].opponent_policy, tapes[1].opponent_policy)

    def test_selfplay_deterministic(self):
        with tempfile.TemporaryDirectory() as d:
            policy = [SelfplayPolicy("mirror", "champion")]
            first = generate_selfplay_tapes(Path(d) / "a", seeds=[42], policies=policy, seats=[0], rng_seed=0)
            second = generate_selfplay_tapes(Path(d) / "b", seeds=[42], policies=policy, seats=[0], rng_seed=0)
            self.assertEqual(first[0].margin, second[0].margin)

    def test_default_mix_has_distinct_strategic_families(self):
        families = {policy.family for policy in DEFAULT_POLICIES}
        self.assertTrue({
            "baseline", "market_liquidity", "capital_timing", "land_investment",
            "farming", "animals", "inventory", "bkg4_mutation",
        }.issubset(families))

    def test_liquidator_converts_shed_to_market_sales(self):
        action = apply_selfplay_policy(
            SelfplayPolicy("liquidator", "liquidator", passive_until=300),
            _observation(step=400, shed={"WHEAT": 3, "MILK": 2}),
            {"farmer": ["PASS"], "hands": [], "market": [["BUY_SEED", "WHEAT", 1]]},
            random.Random(7),
        )
        self.assertEqual(action["market"], [["SELL", "MILK", 2], ["SELL", "WHEAT", 3]])

    def test_farming_variant_plants_its_available_crop(self):
        action = apply_selfplay_policy(
            SelfplayPolicy("farm_tomato", "farm_heavy", crop="TOMATO"),
            _observation(tile=None, seeds={"TOMATO": 2}),
            {"farmer": ["PASS"], "hands": [], "market": []},
            random.Random(7),
        )
        self.assertEqual(action["farmer"], ["PLANT", "TOMATO"])

    def test_early_expansion_prioritizes_unlocked_land(self):
        action = apply_selfplay_policy(
            SelfplayPolicy("early_land", "early_expansion", land_target=4, passive_until=288),
            _observation(step=100),
            {"farmer": ["PASS"], "hands": [], "market": [["SELL", "WHEAT", 1]]},
            random.Random(7),
        )
        self.assertEqual(action["market"][0], ["BUY_LAND"])


if __name__ == "__main__":
    unittest.main()
