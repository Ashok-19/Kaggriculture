"""Phase-5 effect-vector quality-diversity archive contracts."""

from __future__ import annotations

import unittest

from tools.bkg4_search.rlrmb.archive import (
    ARCHIVE_SCHEMA_VERSION,
    Archive,
    ArchiveCellKey,
    ArchiveEntry,
    InsertResult,
    TemporalPhase,
    effect_cluster,
    initiation_cluster,
    make_cell_key,
    temporal_phase_for_step,
)
from tools.bkg4_search.rlrmb.checkpoints import Checkpoint, EventType, Lane
from tools.bkg4_search.rlrmb.effects import EffectBlock, EffectVector, effect_signature
from tools.bkg4_search.rlrmb.schema import DatasetRole


def vector(
    deltas: tuple[int, ...],
    *,
    baseline: tuple[int, ...] | None = None,
    transfer: tuple[int, ...] | None = None,
) -> EffectVector:
    baseline = baseline or tuple(-100 for _ in deltas)
    blocks = [EffectBlock(DatasetRole.TARGET, "target-panel", baseline, deltas)]
    if transfer is not None:
        blocks.append(
            EffectBlock(
                DatasetRole.TRANSFER,
                "transfer-panel",
                tuple(-100 for _ in transfer),
                transfer,
            )
        )
    return EffectVector(tuple(blocks))


def checkpoint(lane: Lane = Lane.GENERAL, step: int = 120) -> Checkpoint:
    return Checkpoint(step, lane, (EventType.REGRET_PEAK,), severity=10, regret=20)


def entry(
    name: str,
    effect: EffectVector,
    *,
    lane: Lane = Lane.GENERAL,
    valid: bool = True,
    wtl: int = 0,
    harm: int = 0,
    ltw: int = 0,
    required: int = 0,
    independent: int = 0,
    both_seat: int = 0,
    worst: int = 0,
    routability: bool | None = None,
    complexity: int = 2,
    aggregate: int = 0,
    provenance: tuple[str, ...] = (),
    cross_regime: bool = False,
    seed_support: int = 0,
    seat_support: int = 0,
    member_support: int = 0,
) -> ArchiveEntry:
    cell = make_cell_key(checkpoint(lane), effect)
    return ArchiveEntry(
        candidate_id=f"candidate-{name}",
        program_id=f"program-{name}",
        effect_signature=effect_signature(effect),
        cell_key=cell,
        semantic_valid=valid,
        target_win_to_loss=wtl,
        target_harm_count=harm,
        target_loss_to_win=ltw,
        required_target_positive_support=required,
        independent_positive_support=independent,
        both_seat_support=both_seat,
        worst_delta=worst,
        routability=routability,
        program_simplicity=complexity,
        aggregate_margin=aggregate,
        loss_to_win_pattern=effect.blocks[0].loss_to_win,
        provenance=provenance,
        cross_regime_supported=cross_regime,
        seed_support=seed_support,
        seat_support=seat_support,
        member_support=member_support,
    )


class ArchiveTests(unittest.TestCase):
    def test_same_cell_superior_replaces_and_inferior_rejects(self) -> None:
        effect = vector((10, 10))
        archive = Archive()
        weak = entry("weak", effect, worst=-10)
        strong = entry("strong", effect, worst=5)
        self.assertEqual(archive.insert(weak), InsertResult.INSERTED_NEW_CELL)
        self.assertEqual(archive.insert(strong), InsertResult.REPLACED_ELITE)
        self.assertEqual(archive.insert(weak), InsertResult.REJECTED_DUPLICATE)
        self.assertEqual(archive.entries(), (strong,))
        other_effect = vector((20, 20))  # same coarse positive-response niche
        inferior = entry("inferior", other_effect, worst=-100)
        self.assertEqual(archive.insert(inferior), InsertResult.REJECTED_INFERIOR)

    def test_safety_and_support_outrank_aggregate_margin(self) -> None:
        effect = vector((20, 20))
        safe = entry("safe", effect, aggregate=10)
        harmful = entry("harm", effect, wtl=1, aggregate=1_000_000)
        archive = Archive((harmful, safe))
        self.assertEqual(archive.entries(), (safe,))

        priorities = (
            entry("ltw", effect, ltw=2, aggregate=-10),
            entry("required", effect, required=2, aggregate=-10),
            entry("independent", effect, independent=2, aggregate=-10),
            entry("both", effect, both_seat=2, aggregate=-10),
        )
        for preferred in priorities:
            baseline = entry("baseline-" + preferred.candidate_id, effect, aggregate=1000)
            field = preferred.candidate_id.removeprefix("candidate-")
            if field != "ltw":
                baseline = ArchiveEntry.from_document({
                    **preferred.to_document(),
                    "candidate_id": "candidate-baseline",
                    "program_id": "program-baseline",
                    field.replace("independent", "independent_positive_support")
                         .replace("required", "required_target_positive_support")
                         .replace("both", "both_seat_support"): 0,
                    "aggregate_margin": 1000,
                })
            else:
                baseline = entry("baseline", effect, aggregate=1000)
            self.assertEqual(Archive((baseline, preferred)).entries()[0], preferred)

    def test_cross_regime_breadth_outranks_single_regime_flip_count(self) -> None:
        effect = vector((20, 20))
        narrow = entry("narrow", effect, ltw=20, aggregate=1_000_000)
        broad = entry(
            "broad", effect, ltw=2, wtl=1, cross_regime=True,
            seed_support=2, seat_support=2, member_support=2,
        )
        self.assertEqual(Archive((narrow, broad)).entries()[0], broad)

    def test_worst_simplicity_then_aggregate_are_late_tiebreakers(self) -> None:
        effect = vector((10,))
        worst = entry("worst", effect, worst=5, complexity=9, aggregate=-100)
        simple = entry("simple", effect, worst=5, complexity=1, aggregate=-100)
        rich = entry("rich", effect, worst=5, complexity=1, aggregate=20)
        self.assertEqual(Archive((entry("base", effect, aggregate=999), worst)).entries()[0], worst)
        self.assertEqual(Archive((worst, simple)).entries()[0], simple)
        self.assertEqual(Archive((simple, rich)).entries()[0], rich)

    def test_distinct_effect_shapes_and_lanes_survive(self) -> None:
        opposite = vector((100, -100), baseline=(-50, 50))
        neutral = vector((0, 0), baseline=(-50, 50))
        flips = vector((100, -100), baseline=(-50, 50))
        positives = vector((10, 10), baseline=(-50, 50))
        self.assertNotEqual(effect_cluster(opposite), effect_cluster(neutral))
        self.assertNotEqual(effect_cluster(flips), effect_cluster(positives))
        archive = Archive((entry("a", opposite), entry("b", neutral)))
        self.assertEqual(len(archive.entries()), 2)
        same = vector((10, 10))
        lanes = Archive((entry("market", same, lane=Lane.MARKET_LIQUIDITY), entry("land", same, lane=Lane.LAND_INVESTMENT)))
        self.assertEqual(len(lanes.entries()), 2)

    def test_provenance_and_exact_semantic_duplicates_collapse(self) -> None:
        effect = vector((10,))
        first = entry("same", effect, provenance=("route-a.json", "attempt-1"))
        second = ArchiveEntry.from_document({
            **first.to_document(),
            "provenance": ["route-b.json", "attempt-99"],
        })
        archive = Archive()
        archive.insert(first)
        self.assertEqual(archive.insert(second), InsertResult.REJECTED_DUPLICATE)
        self.assertEqual(len(archive.entries()), 1)
        conflict = ArchiveEntry.from_document({
            **first.to_document(),
            "cell_key": make_cell_key(checkpoint(Lane.LAND_INVESTMENT), effect).to_document(),
        })
        with self.assertRaises(ValueError):
            archive.insert(conflict)

    def test_insertion_order_and_metrics_are_deterministic(self) -> None:
        effect = vector((10,))
        values = (entry("c", effect, worst=0), entry("a", effect, worst=3), entry("b", effect, worst=2))
        forward = Archive(values)
        reverse = Archive(reversed(values))
        self.assertEqual(forward.entries(), reverse.entries())
        self.assertEqual(forward.metrics(), reverse.metrics())
        self.assertEqual(forward.metrics()["novelty_rate"], 1 / 3)
        self.assertEqual(forward.serialize(), reverse.serialize())

    def test_serialization_round_trip_and_unknown_version(self) -> None:
        archive = Archive((entry("a", vector((20, -5))), entry("b", vector((0, 0)))))
        restored = Archive.deserialize(archive.serialize())
        self.assertEqual(restored.entries(), archive.entries())
        self.assertEqual(restored.metrics(), archive.metrics())
        bad = archive.serialize().replace(
            f'"schema_version":{ARCHIVE_SCHEMA_VERSION}', '"schema_version":999', 1
        )
        with self.assertRaises(ValueError):
            Archive.deserialize(bad)

    def test_invalid_and_unknown_references_fail_closed(self) -> None:
        effect = vector((10,))
        invalid = entry("invalid", effect, valid=False)
        archive = Archive()
        self.assertEqual(archive.insert(invalid), InsertResult.REJECTED_INVALID)
        with self.assertRaises(ValueError):
            ArchiveEntry.from_document({**entry("a", effect).to_document(), "effect_signature": ""})
        with self.assertRaises(ValueError):
            ArchiveEntry.from_document({**entry("a", effect).to_document(), "program_id": ""})

    def test_routability_unknown_is_neutral_and_available_score_is_used(self) -> None:
        effect = vector((10,))
        unknown = entry("unknown", effect, routability=None, aggregate=100)
        routed = entry("routed", effect, routability=True, aggregate=-100)
        self.assertEqual(Archive((unknown, routed)).entries(), (routed,))

    def test_cell_key_is_versioned_and_temporal_boundaries_are_coarse(self) -> None:
        self.assertEqual(temporal_phase_for_step(0), TemporalPhase.EARLY)
        self.assertEqual(temporal_phase_for_step(168), TemporalPhase.MID)
        self.assertEqual(temporal_phase_for_step(360), TemporalPhase.LATE)
        self.assertEqual(temporal_phase_for_step(576), TemporalPhase.ENDGAME)
        key = make_cell_key(checkpoint(), vector((1,)))
        self.assertEqual(ArchiveCellKey.from_document(key.to_document()), key)
        with self.assertRaises(ValueError):
            ArchiveCellKey.from_document({**key.to_document(), "schema_version": 999})
        economic = {"own_money": 999, "unlocked_quadrants": 1, "worker_count": 1, "market_regime": "SCARCE"}
        self.assertEqual(initiation_cluster(checkpoint(), economic), initiation_cluster(checkpoint(), dict(reversed(tuple(economic.items())))))

    def test_sealed_effects_cannot_define_archive_clusters(self) -> None:
        sealed = EffectVector((EffectBlock(DatasetRole.SEALED, "sealed", (-10,), (20,)),))
        with self.assertRaises(ValueError):
            effect_cluster(sealed)

    def test_insert_updates_only_affected_cell_without_full_archive_rebuild(self) -> None:
        from unittest.mock import patch
        effect = vector((10,))
        archive = Archive()
        values = (
            entry("weak", effect, worst=-10),
            entry("strong", effect, worst=5),
            entry("other", vector((0, 0)), worst=0),
        )
        with patch.object(archive, "_rebuild", wraps=archive._rebuild) as rebuild:
            results = tuple(archive.insert(value) for value in values)
        self.assertEqual(results, (
            InsertResult.INSERTED_NEW_CELL,
            InsertResult.REPLACED_ELITE,
            InsertResult.INSERTED_NEW_CELL,
        ))
        self.assertEqual(rebuild.call_count, 0)
        self.assertEqual(set(archive.entries()), {values[1], values[2]})


if __name__ == "__main__":
    unittest.main()
