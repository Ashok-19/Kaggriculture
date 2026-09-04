from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from kaggriculture_bonsai.search.effects import (
    EffectBlock,
    EffectVector,
    effect_signature,
    effect_vectors_from_stage_a_run,
)
from kaggriculture_bonsai.search.schema import DatasetRole


def block(
    role: DatasetRole = DatasetRole.TARGET,
    baseline: tuple[int, ...] = (-10, 20),
    deltas: tuple[int, ...] = (20, -30),
    changes: tuple[int, ...] | None = (2, 3),
    panel_id: str = "ordered-panel-v1",
) -> EffectBlock:
    return EffectBlock(role, panel_id, baseline, deltas, changes)


class EffectVectorTests(unittest.TestCase):
    def test_exact_vector_equality_has_same_signature(self) -> None:
        self.assertEqual(effect_signature(EffectVector((block(),))), effect_signature(EffectVector((block(),))))

    def test_one_per_case_difference_changes_signature(self) -> None:
        first = EffectVector((block(deltas=(100, -100)),))
        second = EffectVector((block(deltas=(100, -99)),))
        self.assertNotEqual(effect_signature(first), effect_signature(second))

    def test_same_aggregate_different_vector_remains_distinct(self) -> None:
        first = EffectVector((block(baseline=(0, 0), deltas=(100, -100)),))
        second = EffectVector((block(baseline=(0, 0), deltas=(0, 0)),))
        self.assertNotEqual(effect_signature(first), effect_signature(second))

    def test_loss_to_win_pattern_is_semantic(self) -> None:
        flips = EffectVector((block(baseline=(-10, 10), deltas=(20, 0)),))
        no_flips = EffectVector((block(baseline=(10, 10), deltas=(20, 0)),))
        self.assertNotEqual(flips.blocks[0].loss_to_win, no_flips.blocks[0].loss_to_win)
        self.assertNotEqual(effect_signature(flips), effect_signature(no_flips))

    def test_win_to_loss_pattern_is_semantic(self) -> None:
        failure = EffectVector((block(baseline=(10, 10), deltas=(-20, 0)),))
        safe = EffectVector((block(baseline=(-10, 10), deltas=(-20, 0)),))
        self.assertNotEqual(failure.blocks[0].win_to_loss, safe.blocks[0].win_to_loss)
        self.assertNotEqual(effect_signature(failure), effect_signature(safe))

    def test_dataset_role_is_semantic(self) -> None:
        target = EffectVector((block(DatasetRole.TARGET),))
        transfer = EffectVector((block(DatasetRole.TRANSFER),))
        self.assertNotEqual(effect_signature(target), effect_signature(transfer))

    def test_panel_identity_and_order_are_semantic(self) -> None:
        first = EffectVector((block(panel_id="panel-a"),))
        other_panel = EffectVector((block(panel_id="panel-b"),))
        reordered = EffectVector((block(baseline=(20, -10), deltas=(-30, 20), changes=(3, 2)),))
        self.assertNotEqual(effect_signature(first), effect_signature(other_panel))
        self.assertNotEqual(effect_signature(first), effect_signature(reordered))

    def test_provenance_does_not_enter_effect_signature(self) -> None:
        evidence = {
            "baseline_margins": [-10, 20],
            "deltas": [20, -30],
            "action_changes": [2, 3],
        }
        attempts = []
        for candidate_id, donor_path in ((1, "/a/route_1.json"), (88, "/b/route_99.json")):
            candidate = {**evidence, "candidate_id": candidate_id, "donor_path": donor_path}
            attempts.append({"candidate": candidate, "transfer": candidate})
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "run.json"
            path.write_text(json.dumps({"candidate_attempts": attempts}), encoding="utf-8")
            vectors = effect_vectors_from_stage_a_run(path, include_transfer=False)
        self.assertEqual(effect_signature(vectors[0]), effect_signature(vectors[1]))

    def test_canonical_form_is_deterministic_and_role_ordered(self) -> None:
        target = block(DatasetRole.TARGET, panel_id="target-panel")
        transfer = block(DatasetRole.TRANSFER, panel_id="transfer-panel")
        first = EffectVector((transfer, target))
        second = EffectVector((target, transfer))
        self.assertEqual(first.to_document(), second.to_document())
        self.assertEqual(effect_signature(first), effect_signature(second))

    def test_invalid_lengths_fail_closed(self) -> None:
        with self.assertRaises(ValueError):
            block(baseline=(-10,), deltas=(20, 30))
        with self.assertRaises(ValueError):
            block(changes=(1,))

    def test_unknown_schema_fails_closed(self) -> None:
        with self.assertRaises(ValueError):
            EffectVector((block(),), schema_version=2)

    def test_action_changes_are_semantic_when_available(self) -> None:
        first = EffectVector((block(changes=(2, 3)),))
        second = EffectVector((block(changes=(2, 4)),))
        self.assertNotEqual(effect_signature(first), effect_signature(second))


if __name__ == "__main__":
    unittest.main()
