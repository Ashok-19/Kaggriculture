from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from actor_data import (
    GoalClass,
    assert_no_relabel_leakage,
    classify_goal_outcome,
    Transition,
    goal_example_weight,
    goal_supervision_arrays,
    operation_support_mask,
    relabel_failure_goals,
    stratified_validation_indices,
)
from actor_grammar import EDIT_OPERATIONS, OP_INDEX
from actor_model import BonsaiEditActorCritic, actor_param_mask
from model import EditFormerConfig
from actor_cli import cuda_goal_metrics, directional_loss, resume_training_span
from actor_generate import GeneratedCandidate, directional_case_score, directional_shortlist


class GoalRelabelTests(unittest.TestCase):
    def test_goal_class_is_lexicographic_and_safety_first(self):
        self.assertEqual(
            classify_goal_outcome(parent_margin=-100, child_margin=50, w2l=0, reg3=0),
            GoalClass.SAFE_FLIP,
        )
        self.assertEqual(
            classify_goal_outcome(parent_margin=-1000, child_margin=-100, w2l=0, reg3=0),
            GoalClass.SAFE_IMPROVE,
        )
        self.assertEqual(
            classify_goal_outcome(parent_margin=-1000, child_margin=-100, w2l=1, reg3=0),
            GoalClass.UNSAFE_IMPROVE,
        )
        self.assertEqual(
            classify_goal_outcome(parent_margin=-1000, child_margin=-1000, w2l=0, reg3=0),
            GoalClass.NO_EFFECT,
        )
        self.assertEqual(
            classify_goal_outcome(parent_margin=-1000, child_margin=-1400, w2l=0, reg3=0),
            GoalClass.TARGET_HARM,
        )
        self.assertEqual(
            classify_goal_outcome(parent_margin=-1000, child_margin=-1400, w2l=0, reg3=1),
            GoalClass.SEVERE_REGRESSION,
        )

    def test_failure_goal_relabeling_uses_exact_signed_case_outcomes(self):
        parent = np.asarray([-1000, -200, 5000, 8000], np.int32)
        child = np.asarray([100, -500, 5500, 3000], np.int32)
        examples, summary = relabel_failure_goals(parent, child)
        self.assertEqual([x["goal_case_id"] for x in examples], [0, 1])
        self.assertEqual(examples[0]["goal_class"], GoalClass.UNSAFE_IMPROVE)
        self.assertEqual(examples[0]["target_delta_raw"], 1100)
        self.assertEqual(examples[1]["goal_class"], GoalClass.SEVERE_REGRESSION)
        self.assertEqual(examples[1]["target_delta_raw"], -300)
        self.assertEqual(summary["w2l"], 0)
        self.assertEqual(summary["reg3"], 1)

    def test_goal_weight_applies_safety_class_before_magnitude(self):
        safe = goal_example_weight(GoalClass.SAFE_FLIP, target_delta_raw=6000, beta=2.0, max_positive=12.0, max_negative=8.0)
        unsafe = goal_example_weight(GoalClass.UNSAFE_IMPROVE, target_delta_raw=60000, beta=2.0, max_positive=12.0, max_negative=8.0)
        noop = goal_example_weight(GoalClass.NO_EFFECT, target_delta_raw=0, beta=2.0, max_positive=12.0, max_negative=8.0)
        self.assertTrue(safe["positive"])
        self.assertFalse(unsafe["positive"])
        self.assertFalse(noop["positive"])
        self.assertGreater(safe["weight"], noop["weight"])
        self.assertLessEqual(safe["weight"], 12.0)
        self.assertLessEqual(unsafe["weight"], 8.0)

    def test_goal_supervision_arrays_are_compact_and_pair_relative(self):
        from actor_grammar import EditAction
        rows = [
            SimpleNamespace(
                baseline=np.asarray([-1000, -200, 5000], np.int32),
                margins=np.asarray([-1000, -200, 5000], np.int32),
            ),
            SimpleNamespace(
                baseline=np.asarray([-1000, -200, 5000], np.int32),
                margins=np.asarray([200, -500, 5200], np.int32),
            ),
        ]
        transitions = [Transition(0, 1, EditAction("CHANGE_ROUTE", 0, route=22), 1.0, True, 1.0)]
        arrays, support = goal_supervision_arrays(transitions, rows, case_cap=4)
        self.assertEqual(arrays["pair_case_mask"].shape, (1, 4))
        np.testing.assert_array_equal(arrays["pair_case_delta_raw"][0, :3], [1200, -300, 200])
        self.assertEqual(arrays["transition_idx"].tolist(), [0, 0])
        self.assertEqual(arrays["goal_case_id"].tolist(), [0, 1])
        self.assertEqual(arrays["goal_class"].tolist(), [int(GoalClass.SAFE_FLIP), int(GoalClass.TARGET_HARM)])
        self.assertEqual(support["CHANGE_ROUTE"]["safe_positive_examples"], 1)
        self.assertEqual(support["CHANGE_ROUTE"]["harmful_examples"], 1)
        self.assertNotIn("actor_target_mask", arrays)

    def test_operation_support_mask_uses_safe_positive_examples(self):
        support = {name: {"safe_positive_examples": 100} for name in EDIT_OPERATIONS}
        support["INSERT_RECOVERY"] = {"safe_positive_examples": 0}
        support["REPLACE_MACRO"] = {"safe_positive_examples": 0}
        support["SPLIT_LANE"] = {"safe_positive_examples": 20}
        support["CHANGE_PRIORITY"] = {"safe_positive_examples": 43}
        mask = operation_support_mask(support, min_safe_positive=32)
        self.assertFalse(mask[OP_INDEX["INSERT_RECOVERY"]])
        self.assertFalse(mask[OP_INDEX["REPLACE_MACRO"]])
        self.assertFalse(mask[OP_INDEX["SPLIT_LANE"]])
        self.assertTrue(mask[OP_INDEX["CHANGE_PRIORITY"]])
        self.assertTrue(mask[OP_INDEX["ADD_CONDITION"]])

    def test_relabels_of_same_transition_cannot_cross_split(self):
        transition_idx = np.asarray([0, 0, 1, 1, 2, 2], np.int32)
        train_idx = np.asarray([0, 1, 4, 5], np.int32)
        val_idx = np.asarray([2, 3], np.int32)
        assert_no_relabel_leakage(transition_idx, train_idx, val_idx)
        with self.assertRaisesRegex(ValueError, "relabel leakage"):
            assert_no_relabel_leakage(
                transition_idx,
                np.asarray([0, 2, 4], np.int32),
                np.asarray([1, 3, 5], np.int32),
            )

    def test_balanced_validation_contains_required_negative_classes_and_ops(self):
        goal_class = np.asarray([
            GoalClass.SAFE_FLIP,
            GoalClass.SAFE_IMPROVE,
            GoalClass.TARGET_HARM,
            GoalClass.NO_EFFECT,
            GoalClass.SEVERE_REGRESSION,
            GoalClass.SAFE_IMPROVE,
            GoalClass.TARGET_HARM,
            GoalClass.NO_EFFECT,
            GoalClass.SEVERE_REGRESSION,
            GoalClass.SAFE_FLIP,
        ], np.int8)
        actor_op = np.asarray([
            OP_INDEX["ADD_CONDITION"], OP_INDEX["CHANGE_ROUTE"],
            OP_INDEX["CHANGE_THRESHOLD"], OP_INDEX["ADD_CONDITION"],
            OP_INDEX["REPLACE_SUBTREE"], OP_INDEX["CHANGE_ROUTE"],
            OP_INDEX["ADD_SPECIALIST"], OP_INDEX["CHANGE_WINDOW"],
            OP_INDEX["CHANGE_THRESHOLD"], OP_INDEX["ADD_SPECIALIST"],
        ], np.int32)
        ids, counts = stratified_validation_indices(
            np.arange(len(goal_class), dtype=np.int32), goal_class, actor_op, max_samples=10,
        )
        self.assertEqual(len(ids), 10)
        self.assertGreater(counts["target_positive"], 0)
        self.assertGreater(counts["target_harm"], 0)
        self.assertGreater(counts["no_effect"], 0)
        self.assertGreater(counts["severe_regression"], 0)
        self.assertGreaterEqual(counts["operation_families"], 4)

    def test_balanced_validation_fails_when_required_negative_class_missing(self):
        goal_class = np.asarray([GoalClass.SAFE_FLIP, GoalClass.SAFE_IMPROVE], np.int8)
        actor_op = np.asarray([OP_INDEX["ADD_CONDITION"], OP_INDEX["CHANGE_ROUTE"]], np.int32)
        with self.assertRaisesRegex(ValueError, "required validation class"):
            stratified_validation_indices(
                np.asarray([0, 1], np.int32), goal_class, actor_op, max_samples=2,
            )




class ResumePlanTests(unittest.TestCase):
    def test_resume_training_span_finishes_exact_total_epoch_budget(self):
        self.assertEqual(resume_training_span(0, 10), (0, 10))
        self.assertEqual(resume_training_span(2, 10), (2, 8))
        self.assertEqual(resume_training_span(6, 10), (6, 4))
        self.assertEqual(resume_training_span(10, 10), (10, 0))
        with self.assertRaises(ValueError):
            resume_training_span(11, 10)


class CudaGoalBookkeepingTests(unittest.TestCase):
    def test_candidate_goal_maps_to_exact_cuda_case(self):
        baseline=np.asarray([100,-50,-500,200],np.int32)
        margins=np.asarray([100,25,-600,200],np.int32)
        c=GeneratedCandidate("actor","x","p",{"operation":"CHANGE_ROUTE"},{},goal_case_ids=(1,),generated_goal_case_ids=(1,2),goal_mode="LOSS_REPAIR")
        m=cuda_goal_metrics(c,baseline,margins)
        self.assertEqual(m["goal_case_ids"],[1])
        self.assertEqual(m["generated_goal_case_ids"],[1,2])
        self.assertTrue(m["target_improved"])
        self.assertFalse(m["target_worsened"])
        self.assertEqual(m["target_best_delta"],75)
        self.assertEqual(m["target_flip_count"],1)


class DirectionalShortlistTests(unittest.TestCase):
    def test_directional_case_score_uses_pair_relative_head_not_frozen_case_delta(self):
        score = directional_case_score(
            parent_margins=np.asarray([-1000, 5000], np.int32),
            case_ids=np.asarray([0, 1], np.int32),
            case_mask=np.asarray([True, True]),
            goal_mask=np.asarray([True, False]),
            direction_delta=np.asarray([-0.2, 0.1], np.float32),
            improve_prob=np.asarray([0.2, 0.8], np.float32),
            flip_prob=np.asarray([0.1, 0.0], np.float32),
            regress_prob=np.asarray([0.0, 0.3], np.float32),
        )
        self.assertAlmostEqual(score["goal_delta_norm"], -0.2, places=5)
        self.assertLess(score["goal_delta_margin"], 0.0)
        self.assertEqual(score["predicted_target_flips"], 0)
        self.assertEqual(score["predicted_win_crossings"], 0)

    def test_positive_target_direction_beats_large_destructive_absolute_effect(self):
        good = GeneratedCandidate(
            "actor", "good", "parent", {"operation": "CHANGE_THRESHOLD"}, {},
            critic_regression_prob=0.05, critic_effect_mean_abs=0.1,
            critic_goal_delta_norm=0.25, critic_goal_improve_prob=0.9,
            critic_goal_flip_prob=0.7, critic_protected_regress_prob_max=0.05,
        )
        bad = GeneratedCandidate(
            "actor", "bad", "parent", {"operation": "ADD_SPECIALIST"}, {},
            critic_regression_prob=0.01, critic_effect_mean_abs=0.95,
            critic_goal_delta_norm=-0.7, critic_goal_improve_prob=0.1,
            critic_goal_flip_prob=0.0, critic_protected_regress_prob_max=0.01,
        )
        selected, diag = directional_shortlist([bad, good], 1)
        self.assertEqual(selected[0].semantic_id, "good")
        self.assertEqual(diag["positive_direction_available"], 1)
        self.assertEqual(diag["selected_positive_direction"], 1)


class DirectionalLossTests(unittest.TestCase):
    def test_directional_loss_prefers_correct_pair_relative_predictions(self):
        import jax.numpy as jnp
        batch = {
            "direction_mask": jnp.asarray([[True, True, True]]),
            "direction_delta_norm": jnp.asarray([[0.5, -0.4, 0.2]], jnp.float32),
            "direction_improve": jnp.asarray([[True, False, True]]),
            "direction_flip": jnp.asarray([[True, False, False]]),
            "direction_regress": jnp.asarray([[False, True, False]]),
            "weight": jnp.asarray([1.0], jnp.float32),
        }
        good = {
            "actor_direction_delta": jnp.asarray([[0.5, -0.4, 0.2]], jnp.float32),
            "actor_direction_improve_logit": jnp.asarray([[5.0, -5.0, 5.0]], jnp.float32),
            "actor_direction_flip_logit": jnp.asarray([[5.0, -5.0, -5.0]], jnp.float32),
            "actor_direction_regress_logit": jnp.asarray([[-5.0, 5.0, -5.0]], jnp.float32),
        }
        bad = {
            "actor_direction_delta": -good["actor_direction_delta"],
            "actor_direction_improve_logit": -good["actor_direction_improve_logit"],
            "actor_direction_flip_logit": -good["actor_direction_flip_logit"],
            "actor_direction_regress_logit": -good["actor_direction_regress_logit"],
        }
        good_loss, parts = directional_loss(good, batch)
        bad_loss, _ = directional_loss(bad, batch)
        self.assertLess(float(good_loss), float(bad_loss))
        self.assertEqual(set(parts), {"delta", "improve", "flip", "regress"})


class GoalConditionedModelTests(unittest.TestCase):
    @staticmethod
    def _cfg():
        return EditFormerConfig(
            d_model=48, heads=6, ffn_dim=96, tree_blocks=1, case_blocks=1, fusion_blocks=1,
            tree_cap=8, edit_cap=4, case_cap=6, dropout=0.0, attention_impl="xla", compute_dtype="float32",
        )

    @staticmethod
    def _batch(goal_case: int):
        cfg = GoalConditionedModelTests._cfg()
        tree_cat = np.zeros((1, cfg.tree_cap, 6), np.int32)
        tree_num = np.zeros((1, cfg.tree_cap, 10), np.float32)
        tree_mask = np.zeros((1, cfg.tree_cap), bool); tree_mask[:, :5] = True
        tree_parent = np.full((1, cfg.tree_cap), -1, np.int32)
        tree_parent[0, 1:5] = [0, 1, 0, 3]
        tree_depth = np.zeros((1, cfg.tree_cap), np.int32); tree_depth[0, 1:5] = [1, 2, 1, 2]
        tree_rule = np.zeros((1, cfg.tree_cap), np.int32); tree_rule[0, 3:5] = 1
        edit_cat = np.zeros((1, cfg.edit_cap, 5), np.int32)
        edit_num = np.zeros((1, cfg.edit_cap, 14), np.float32)
        edit_mask = np.zeros((1, cfg.edit_cap), bool); edit_mask[:, :2] = True
        case_features = np.zeros((1, cfg.case_cap, 22), np.float32)
        case_features[0, 0, [0, 2, 4, 11, 12, 19, 21]] = [-0.7, 1.0, 1.0, 0.1, 0.8, 0.2, 0.37]
        case_features[0, 1, [0, 2, 4, 11, 12, 19, 21]] = [-0.2, 1.0, 1.0, 0.9, 0.2, 0.9, 0.40]
        case_mask = np.zeros((1, cfg.case_cap), bool); case_mask[:, :2] = True
        goal_mask = np.zeros((1, cfg.case_cap), bool); goal_mask[0, goal_case] = True
        return {
            "tree_cat": tree_cat, "tree_num": tree_num, "tree_mask": tree_mask,
            "tree_parent": tree_parent, "tree_depth": tree_depth, "tree_rule": tree_rule,
            "edit_cat": edit_cat, "edit_num": edit_num, "edit_mask": edit_mask,
            "case_features": case_features, "case_mask": case_mask,
            "case_delta": np.zeros((1, cfg.case_cap), np.float32),
            "heuristic": np.zeros((1,), np.float32), "labels": np.zeros((1, 9), np.float32),
            "group": np.zeros((1,), np.int32),
            "goal_mask": goal_mask,
            "goal_spec": np.asarray([[1.0, 0.0, 1.0, 0.0]], np.float32),
            "actor_target_mask": np.ones((1, len(EDIT_OPERATIONS), cfg.tree_cap), bool),
            "actor_op": np.asarray([OP_INDEX["CHANGE_ROUTE"]], np.int32),
            "actor_target": np.asarray([3], np.int32),
        }

    def test_same_parent_different_goal_changes_all_actor_stages_but_not_critic(self):
        import jax
        import jax.numpy as jnp
        cfg = self._cfg(); model = BonsaiEditActorCritic(cfg)
        b0 = {k: jnp.asarray(v) for k, v in self._batch(0).items()}
        b1 = {k: jnp.asarray(v) for k, v in self._batch(1).items()}
        params = model.init({"params": jax.random.PRNGKey(7)}, b0, train=False)["params"]
        o0 = model.apply({"params": params}, b0, train=False)
        o1 = model.apply({"params": params}, b1, train=False)
        for key in ("regression_logit", "flip_logit", "margin", "worst_win", "rank_residual", "case_delta"):
            np.testing.assert_array_equal(np.asarray(o0[key]), np.asarray(o1[key]))
        self.assertFalse(np.allclose(np.asarray(o0["actor_op_logits"]), np.asarray(o1["actor_op_logits"])))
        self.assertFalse(np.allclose(np.asarray(o0["actor_target_logits_all"]), np.asarray(o1["actor_target_logits_all"])))
        self.assertFalse(np.allclose(np.asarray(o0["actor_feature_logits"]), np.asarray(o1["actor_feature_logits"])))

    def test_directional_case_heads_exist_and_actor_mask_keeps_critic_frozen(self):
        import jax
        import jax.numpy as jnp
        from flax import traverse_util
        from flax.core import unfreeze
        cfg = self._cfg(); model = BonsaiEditActorCritic(cfg)
        batch = {k: jnp.asarray(v) for k, v in self._batch(0).items()}
        params = model.init({"params": jax.random.PRNGKey(11)}, batch, train=False)["params"]
        out = model.apply({"params": params}, batch, train=False)
        for key in ("actor_direction_delta", "actor_direction_improve_logit", "actor_direction_flip_logit", "actor_direction_regress_logit"):
            self.assertEqual(tuple(out[key].shape), (1, cfg.case_cap))
        mask = actor_param_mask(params)
        flat = traverse_util.flatten_dict(unfreeze(mask), sep="/")
        self.assertTrue(any(bool(v) for v in flat.values()))
        for name, enabled in flat.items():
            if bool(enabled):
                self.assertTrue(any(part.startswith("actor_") for part in name.split("/")), name)

if __name__ == "__main__":
    unittest.main()
