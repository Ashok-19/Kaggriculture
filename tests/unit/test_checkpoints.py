from __future__ import annotations

import unittest

from kaggriculture_bonsai.search.checkpoints import (
    Checkpoint,
    EventEvidence,
    EventType,
    Lane,
    extract_trace_events,
    select_checkpoints,
)


class CheckpointSelectorTests(unittest.TestCase):
    def test_determinism_and_stable_serialization(self) -> None:
        evidence = [
            EventEvidence(120, (EventType.DAY_BOUNDARY, EventType.HIRE), severity=20, regret=30),
            EventEvidence(156, (EventType.FIRST_EFFECT_DIVERGENCE,), severity=40, regret=50),
        ]
        first = select_checkpoints(evidence)
        second = select_checkpoints(reversed(evidence))
        self.assertEqual(first, second)
        self.assertEqual([item.semantic_id for item in first], [item.semantic_id for item in second])
        self.assertEqual(first[0].to_document(), first[0].to_document())

    def test_event_extraction_from_known_trace_fields_and_actions(self) -> None:
        records = [
            {"step": 24, "shop_revealed": True, "policy_diverged": True},
            {
                "step": 25,
                "action": {
                    "farmer": ["BUILD_PASTURE"],
                    "market": [["BUY_LAND"], ["HIRE"], ["BUY_ANIMAL", "SHEEP", 1]],
                },
                "market_shift": True,
                "production_event": True,
                "route_changed": True,
            },
            {"step": 26, "regret_peak": True, "first_effect_divergence": True},
        ]
        events = extract_trace_events(records)
        reasons = {reason for event in events for reason in event.reasons}
        self.assertTrue(
            {
                EventType.DAY_BOUNDARY,
                EventType.SHOP_REVEAL,
                EventType.POLICY_DIVERGENCE,
                EventType.BUY_LAND,
                EventType.HIRE,
                EventType.ANIMAL_OR_STRUCTURE_TRANSITION,
                EventType.MARKET_PRICE_OR_INVENTORY_SHIFT,
                EventType.PRODUCTION_OR_PAYOFF_EVENT,
                EventType.ROUTE_OR_PHASE_CHANGE,
                EventType.REGRET_PEAK,
                EventType.FIRST_EFFECT_DIVERGENCE,
            }.issubset(reasons)
        )

    def test_boundary_offsets_remain_in_playable_range(self) -> None:
        evidence = [
            EventEvidence(0, (EventType.SHOP_REVEAL,)),
            EventEvidence(718, (EventType.DAY_BOUNDARY,)),
        ]
        checkpoints = select_checkpoints(evidence, max_step=718)
        self.assertTrue(checkpoints)
        self.assertTrue(all(0 <= item.step <= 718 for item in checkpoints))

    def test_same_state_and_step_merge_reasons(self) -> None:
        evidence = [
            EventEvidence(120, (EventType.HIRE,), state_signature="state-a", recurrence_support=2),
            EventEvidence(120, (EventType.BUY_LAND,), state_signature="state-a", recurrence_support=3),
        ]
        exact = [item for item in select_checkpoints(evidence) if item.step == 120]
        self.assertEqual(len(exact), 1)
        self.assertEqual(exact[0].reasons, (EventType.BUY_LAND, EventType.HIRE))
        self.assertEqual(exact[0].recurrence_support, 3)

    def test_recurrent_both_seat_event_outranks_one_off(self) -> None:
        recurrent = EventEvidence(
            120,
            (EventType.HIRE,),
            severity=50,
            regret=50,
            recurrence_support=4,
            both_seat_support=2,
        )
        one_off = EventEvidence(240, (EventType.HIRE,), severity=50, regret=50, recurrence_support=1)
        self.assertEqual(select_checkpoints([one_off, recurrent])[0].step, 120)

    def test_severity_breaks_otherwise_equal_rank(self) -> None:
        weak = EventEvidence(120, (EventType.REGRET_PEAK,), severity=10, regret=20)
        severe = EventEvidence(240, (EventType.REGRET_PEAK,), severity=30, regret=20)
        self.assertEqual(select_checkpoints([weak, severe])[0].step, 240)

    def test_causal_event_outranks_fallback(self) -> None:
        causal = EventEvidence(120, (EventType.REGRET_PEAK,), severity=1)
        ranked = select_checkpoints([causal], always_include_fallback=True)
        self.assertFalse(ranked[0].fallback)
        self.assertTrue(any(item.fallback for item in ranked))

    def test_fallback_at_same_step_does_not_contaminate_causal_identity(self) -> None:
        causal = EventEvidence(144, (EventType.SHOP_REVEAL,), severity=1)
        checkpoint = next(item for item in select_checkpoints([causal], always_include_fallback=True) if item.step == 144)
        self.assertFalse(checkpoint.fallback)
        self.assertNotIn(EventType.FALLBACK_GRID, checkpoint.reasons)

    def test_lane_assignment(self) -> None:
        cases = {
            EventType.MARKET_PRICE_OR_INVENTORY_SHIFT: Lane.MARKET_LIQUIDITY,
            EventType.HIRE: Lane.WORKER_PRODUCTION,
            EventType.BUY_LAND: Lane.LAND_INVESTMENT,
            EventType.ANIMAL_OR_STRUCTURE_TRANSITION: Lane.ANIMAL_TRANSITION,
        }
        for reason, lane in cases.items():
            with self.subTest(reason=reason):
                checkpoint = select_checkpoints([EventEvidence(100, (reason,))])[0]
                self.assertEqual(checkpoint.lane, lane)
        crop = select_checkpoints(
            [EventEvidence(100, (EventType.PRODUCTION_OR_PAYOFF_EVENT,), lane_hint=Lane.CROP_TRANSITION)]
        )[0]
        self.assertEqual(crop.lane, Lane.CROP_TRANSITION)

    def test_adjacent_event_lane_does_not_override_exact_causal_lane(self) -> None:
        exact = EventEvidence(120, (EventType.REGRET_PEAK,), lane_hint=Lane.WORKER_PRODUCTION, regret=50)
        adjacent = EventEvidence(121, (EventType.MARKET_PRICE_OR_INVENTORY_SHIFT,))
        at_120 = next(item for item in select_checkpoints([adjacent, exact]) if item.step == 120)
        self.assertEqual(at_120.lane, Lane.WORKER_PRODUCTION)

    def test_provenance_identity_does_not_change_semantics_or_rank(self) -> None:
        left = EventEvidence(
            120,
            (EventType.HIRE,),
            family_ref="family-a",
            provenance=("opponent=alpha", "replay=r1", "episode=e1", "filename=a.json"),
        )
        right = EventEvidence(
            120,
            (EventType.HIRE,),
            family_ref="family-b",
            provenance=("opponent=beta", "replay=r2", "episode=e2", "filename=b.json"),
        )
        left_checkpoint = select_checkpoints([left])[0]
        right_checkpoint = select_checkpoints([right])[0]
        self.assertEqual(left_checkpoint.semantic_id, right_checkpoint.semantic_id)
        self.assertEqual(left_checkpoint.rank_key(), right_checkpoint.rank_key())

    def test_sparse_fallback_is_bounded_and_deterministic(self) -> None:
        first = select_checkpoints([])
        second = select_checkpoints([])
        self.assertEqual(first, second)
        self.assertEqual([item.step for item in first], [0, 72, 144, 216, 288, 432, 576])
        self.assertTrue(all(item.fallback for item in first))

    def test_historical_regions_reachable_from_compact_event_fixtures(self) -> None:
        fixtures = {
            "route435-like": (
                [{"step": 155, "action": {"market": [["BUY_PRODUCT", "WHEAT", 2]]}, "first_effect_divergence": True}],
                156,
            ),
            "route410-like": (
                [{"step": 120, "action": {"market": [["SELL", "WHEAT", 8], ["HIRE"]]}}],
                120,
            ),
            "route437-like": (
                [{"step": 264, "action": {"market": [["BUY_ANIMAL", "SHEEP", 2], ["BUY_LAND"], ["HIRE"]]}}],
                264,
            ),
        }
        for name, (records, useful_step) in fixtures.items():
            with self.subTest(name=name):
                proposals = select_checkpoints(extract_trace_events(records))
                self.assertLessEqual(min(abs(item.step - useful_step) for item in proposals), 1)
                self.assertFalse(min(proposals, key=lambda item: abs(item.step - useful_step)).fallback)

    def test_checkpoint_schema_version_fails_closed(self) -> None:
        with self.assertRaises(ValueError):
            Checkpoint(120, Lane.GENERAL, (EventType.REGRET_PEAK,), schema_version=2)


if __name__ == "__main__":
    unittest.main()
