from __future__ import annotations

import unittest

from tools.bkg4_search.rlrmb.routability import (
    ROUTABILITY_SCHEMA_VERSION,
    BoundaryState,
    RequiredContinuation,
    RoutabilityClassification,
    RoutabilityEvidence,
    RoutabilityResult,
    TrajectoryEvidence,
    screen_routability,
)
from tools.bkg4_search.rlrmb.schema import DatasetRole


def state(step: int, signature: int, *, legal=(), gate=None) -> BoundaryState:
    legal_context = tuple(legal)
    return BoundaryState(
        step,
        (signature,),
        legal_context=legal_context,
        gate_context=legal_context if gate is None else tuple(gate),
    )


def trajectory(
    case_id: str,
    states,
    requirements,
    *,
    delta: int,
    role: DatasetRole = DatasetRole.TARGET,
    group: str = "family-a",
    seat: int = 0,
    drift: bool = False,
    required: bool | None = None,
    provenance=(),
) -> TrajectoryEvidence:
    return TrajectoryEvidence(
        case_id,
        role,
        tuple(states),
        tuple(RequiredContinuation(step, value) for step, value in requirements),
        effect_delta=delta,
        support_group=group,
        family=group,
        seat=seat,
        candidate_induced_drift=drift,
        required_support=delta > 0 if required is None else required,
        provenance=tuple(provenance),
    )


def evidence(*rows, checkpoints=(), reveals=(), candidate_id="a" * 64) -> RoutabilityEvidence:
    return RoutabilityEvidence(candidate_id, 0, tuple(rows), tuple(checkpoints), tuple(reveals))


class RoutabilityTests(unittest.TestCase):
    def test_exact_routable_case_only_recommends_cegis(self):
        rows = (
            trajectory("p", (state(0, 1, legal=(("OWN_MONEY", 0, 10),)),), ((0, "CANDIDATE"),), delta=5),
            trajectory("h", (state(0, 2, legal=(("OWN_MONEY", 0, 1),)),), ((0, "PARENT"),), delta=-2),
        )
        result = screen_routability(evidence(*rows))
        self.assertEqual(result.classification, RoutabilityClassification.ROUTABLE_CANDIDATE)
        self.assertEqual(result.recommended_action, "ENTER_HARDENED_CEGIS")
        self.assertNotIn("ACCEPT", result.to_document()["recommended_action"])

    def test_nonrequired_mixed_positive_may_be_left_for_cegis(self):
        rows = (
            trajectory("safe", (state(0, 2, legal=(("OWN_MONEY", 0, 10),)),), ((0, "CANDIDATE"),), delta=5),
            trajectory("mixed-positive", (state(0, 1, legal=(("OWN_MONEY", 0, 1),)),), ((0, "CANDIDATE"),), delta=1, required=False),
            trajectory("mixed-harm", (state(0, 1, legal=(("OWN_MONEY", 0, 1),)),), ((0, "PARENT"),), delta=-2),
        )
        result = screen_routability(evidence(*rows))
        self.assertEqual(result.classification, RoutabilityClassification.ROUTABLE_CANDIDATE)
        self.assertEqual(result.diagnostics["ignored_nonrequired_conflicts"], 1)

    def test_later_checkpoint_common_prefix_branch_and_abort_classification(self):
        base = (
            trajectory("p", (state(0, 1), state(4, 10), state(8, 20)), ((10, "A"),), delta=5),
            trajectory("h", (state(0, 1), state(4, 11), state(8, 21)), ((10, "B"),), delta=-2),
        )
        self.assertEqual(screen_routability(evidence(*base, checkpoints=(4,))).classification, RoutabilityClassification.REPAIR_LATER_CHECKPOINT)
        self.assertEqual(screen_routability(evidence(*base)).classification, RoutabilityClassification.REPAIR_COMMON_PREFIX)
        self.assertEqual(screen_routability(evidence(*base, reveals=(4,))).classification, RoutabilityClassification.REPAIR_BRANCH_AT_REVEAL)
        drift = tuple(
            TrajectoryEvidence(**{**row.__dict__, "candidate_induced_drift": True}) for row in base
        )
        self.assertEqual(screen_routability(evidence(*drift)).classification, RoutabilityClassification.REPAIR_ABORT_RECOVERY)

    def test_future_dependent_exact_collision_and_late_divergence(self):
        rows = (
            trajectory("positive", (state(0, 1), state(5, 2)), ((3, "A"),), delta=5),
            trajectory("harm", (state(0, 1), state(5, 3)), ((3, "B"),), delta=-2),
        )
        result = screen_routability(evidence(*rows))
        self.assertEqual(result.classification, RoutabilityClassification.FUTURE_DEPENDENT)
        self.assertEqual(result.first_required_action_divergence, 3)
        self.assertEqual(result.first_legal_divergence, 5)
        self.assertEqual(result.conflicting_groups, (("harm", "positive"),))

    def test_gate_limitation_requires_full_legal_but_not_gate_separation(self):
        rows = (
            trajectory("p", (state(0, 1, legal=(("LEGAL_X", 0, 1),), gate=()),), ((0, "A"),), delta=5),
            trajectory("h", (state(0, 2, legal=(("LEGAL_X", 0, 0),), gate=()),), ((0, "B"),), delta=-2),
        )
        result = screen_routability(evidence(*rows))
        self.assertEqual(result.classification, RoutabilityClassification.GATE_LIMITATION_SUSPECTED)
        self.assertEqual(result.recommended_action, "REVIEW_GATE_VOCABULARY")

    def test_identity_is_provenance_only_and_runtime_injection_rejects(self):
        left = trajectory("p", (state(0, 1),), ((0, "A"),), delta=5, provenance=("replay_id=r1", "opponent_name=x"))
        right = trajectory("h", (state(0, 2),), ((0, "B"),), delta=-1, provenance=("route filename=/a",))
        first = screen_routability(evidence(left, right))
        changed = screen_routability(evidence(
            TrajectoryEvidence(**{**left.__dict__, "provenance": ("team=y", "episode_id=e2")}),
            TrajectoryEvidence(**{**right.__dict__, "provenance": ("submission_id=s",)}),
        ))
        self.assertEqual((first.classification, first.score), (changed.classification, changed.score))
        for feature in ("opponent_name", "replay_id", "episode_id", "tape_id", "route_filename", "source_file", "submission_id", "team_name"):
            with self.subTest(feature=feature), self.assertRaises(ValueError):
                state(0, 1, legal=((feature, 0, 1),))

    def test_sealed_and_certification_roles_fail_closed(self):
        for role in (DatasetRole.SEALED, DatasetRole.CERTIFICATION):
            with self.subTest(role=role), self.assertRaises(ValueError):
                evidence(trajectory("x", (state(0, 1),), ((0, "A"),), delta=1, role=role))

    def test_group_and_seat_diagnostic_is_deterministic(self):
        rows = (
            trajectory("a0", (state(0, 1, legal=(("OWN_MONEY", 0, 10),)),), ((0, "A"),), delta=2, group="a", seat=0),
            trajectory("a1", (state(0, 2, legal=(("OWN_MONEY", 0, 10),)),), ((0, "A"),), delta=2, group="a", seat=1),
            trajectory("b0", (state(0, 3, legal=(("OWN_MONEY", 0, 1),)),), ((0, "B"),), delta=-2, group="b", seat=0),
        )
        first = screen_routability(evidence(*rows))
        second = screen_routability(evidence(*reversed(rows)))
        self.assertEqual(first.to_document(), second.to_document())
        self.assertEqual(first.diagnostics["both_seat_support_groups"], 1)
        self.assertEqual(first.diagnostics["legal_predicate"]["train_support"], 2)
        self.assertEqual(first.diagnostics["legal_predicate"]["held_out_support"], 0)

    def test_family0_regression_is_future_dependent(self):
        rows = (
            trajectory("transfer-positive", tuple(state(step, 7) for step in range(24, 49)), ((26, "CARE"),), delta=99, required=True),
            trajectory("transfer-harm", tuple(state(step, 7) for step in range(24, 49)), ((26, "WEST"),), delta=-2965),
        )
        result = screen_routability(RoutabilityEvidence("f" * 64, 24, rows))
        self.assertEqual(result.classification, RoutabilityClassification.FUTURE_DEPENDENT)
        self.assertEqual(result.first_required_action_divergence, 26)
        self.assertIsNone(result.first_legal_divergence)

    def test_candidate50_generic_mixed_state_remains_non_promotable(self):
        rows = (
            trajectory("target-support", (state(24, 50),), ((24, "CANDIDATE"),), delta=104),
            trajectory("validation-harm", (state(24, 50),), ((24, "PARENT"),), delta=-1, role=DatasetRole.VALIDATION),
        )
        result = screen_routability(RoutabilityEvidence("5" * 64, 24, rows))
        self.assertEqual(result.classification, RoutabilityClassification.FUTURE_DEPENDENT)
        self.assertNotEqual(result.recommended_action, "ENTER_HARDENED_CEGIS")

    def test_result_serialization_score_and_version_fail_closed(self):
        rows = (trajectory("p", (state(0, 1),), ((0, "A"),), delta=1),)
        result = screen_routability(evidence(*rows))
        self.assertEqual(RoutabilityResult.from_document(result.to_document()), result)
        self.assertEqual(result.score, screen_routability(evidence(*rows)).score)
        document = result.to_document(); document["schema_version"] = ROUTABILITY_SCHEMA_VERSION + 1
        with self.assertRaises(ValueError):
            RoutabilityResult.from_document(document)


if __name__ == "__main__":
    unittest.main()
