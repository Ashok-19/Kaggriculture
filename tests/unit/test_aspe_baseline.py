from __future__ import annotations

import json
import unittest
from pathlib import Path

from kaggriculture_bonsai.league.aspe_baseline import (
    ASPE_BASELINE_SCHEMA,
    ASPE_BASELINE_SCHEMA_VERSION,
    baseline_results,
    build_manifest,
    planned_panel_contract,
    validate_manifest,
)


class AspeBaselineTests(unittest.TestCase):
    def test_panel_plan_is_deterministic_and_seed_disjoint(self):
        first = planned_panel_contract()
        self.assertEqual(first, planned_panel_contract())
        seed_sets = {name: set(row["seeds"]) for name, row in first.items()}
        for left_index, left in enumerate(seed_sets):
            for right in tuple(seed_sets)[left_index + 1:]:
                self.assertFalse(seed_sets[left] & seed_sets[right])
        self.assertEqual(first["build"]["seats"], [0, 1])
        self.assertEqual(len(first["build"]["members"]), 8)
        self.assertEqual(len(first["build"]["lanes"]), 8)

    def test_preserved_search_baselines_match_tracked_result(self):
        result = baseline_results(Path("workspace/oracle_bakeoff/result.json"))
        self.assertEqual(result["rlrmb"]["candidate_count"], 197)
        self.assertEqual(result["rlrmb"]["unique_effect_phenotypes"], 86)
        self.assertEqual(result["rlrmb"]["cross_regime_build_specialists"], 0)
        self.assertEqual(result["direct_bonsai"]["candidate_count"], 198)
        self.assertEqual(result["direct_bonsai"]["useful_build_specialists"], 13)
        self.assertEqual(result["direct_bonsai"]["dev_attempted"], 1)
        self.assertEqual(result["direct_bonsai"]["dev_passes"], 0)

    def test_tracked_manifest_validates_and_regenerates(self):
        output = Path("workspace/aspe/baseline")
        manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["schema"], ASPE_BASELINE_SCHEMA)
        self.assertEqual(manifest["schema_version"], ASPE_BASELINE_SCHEMA_VERSION)
        validate_manifest(manifest)
        self.assertEqual(
            build_manifest(output, source_head=manifest["source_head"]), manifest,
        )


if __name__ == "__main__":
    unittest.main()
