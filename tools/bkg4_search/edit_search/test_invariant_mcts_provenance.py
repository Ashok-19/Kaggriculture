from __future__ import annotations

import unittest

from tools.bkg4_search.invariant_search import InvariantSearchBackend


class InvariantMctsProvenanceTests(unittest.TestCase):
    def test_mcts_preserves_direct_generating_goal_and_full_goal_path_through_dedup(self):
        backend = InvariantSearchBackend(route_ids=(29, 64, 70, 211, 212), evaluator=object())
        backend.ladder.level = backend.ladder.LEVELS.index("grouped_cegis")
        modules = backend.propose(n=192, seed=20260824)
        origins = [
            backend.last_proposal_origins[module.semantic_id()]
            for module in modules
            if backend.last_proposal_origins[module.semantic_id()].operator == "mcts"
        ]
        self.assertTrue(origins)
        self.assertTrue(all(origin.source_goal_id for origin in origins))
        self.assertTrue(all(origin.source_goal_path for origin in origins))
        self.assertTrue(all(origin.source_goal_path[-1] == origin.source_goal_id for origin in origins))

    def test_mcts_source_goal_remains_distinct_from_retrospective_classification(self):
        from tools.bkg4_search.invariant_module import InvariantStrategicModule
        from tools.bkg4_search.invariant_search import default_search_goals, emit_skeletons, structural_mcts

        routes = (29, 64, 70, 211, 212)
        goals = default_search_goals(routes)
        route_goal = next(goal for goal in goals if goal.goal_id == "opening_route_regime")
        source_goal = next(goal for goal in goals if goal.goal_id == "endgame_liquidation")
        route = emit_skeletons(route_goal, n=1, seed=1, route_ids=routes)[0]
        market = emit_skeletons(source_goal, n=1, seed=2, route_ids=routes)[0]
        root = InvariantStrategicModule((route.rules[0], market.rules[0]))
        root.validate()

        proposal = structural_mcts(
            (root,), (source_goal,), budget=1, seed=1, route_ids=routes,
        )[0]
        backend = InvariantSearchBackend(route_ids=routes, evaluator=object())
        retrospective = backend._goal_for_module(proposal.module)
        self.assertIsNotNone(retrospective)
        self.assertEqual(proposal.source_goal_id, "endgame_liquidation")
        self.assertNotEqual(proposal.source_goal_id, retrospective.goal_id)

    def test_mcts_provenance_is_metadata_only_for_recursive_parent_order(self):
        from tools.bkg4_search.invariant_factory import SpecialistFactory
        from tools.bkg4_search.invariant_groups import GroupOutcome
        from tools.bkg4_search.invariant_search import default_search_goals, emit_skeletons
        from tools.bkg4_search.invariant_specialists import SpecialistLineage, analyze_specialist

        routes = (29, 64, 70, 211, 212)
        goals = default_search_goals(routes)
        non_mcts_goal = next(goal for goal in goals if goal.goal_id == "opening_route_regime")
        mcts_goal = next(goal for goal in goals if goal.goal_id == "endgame_liquidation")
        non_mcts = emit_skeletons(non_mcts_goal, n=1, seed=7, route_ids=routes)[0]
        mcts = emit_skeletons(mcts_goal, n=1, seed=11, route_ids=routes)[0]

        def evidence(module, useful_groups: int):
            rows = tuple(
                GroupOutcome(f"g{i}", False, False, True, i < useful_groups, 0, 1000 if i < useful_groups else 0)
                for i in range(3)
            )
            return analyze_specialist(
                module, rows,
                cohort_by_group={f"g{i}": "cohort" for i in range(3)},
                activation_count=3, action_change_count=3,
            )

        def recursive_semantics(mcts_source_goal_id):
            backend = InvariantSearchBackend(route_ids=routes, evaluator=object())
            factory = SpecialistFactory(backend, root_controller="parent-controller")
            for item in (evidence(mcts, 3), evidence(non_mcts, 2)):
                backend.specialist_archive.consider(item)
            factory._lineages[non_mcts.semantic_id()] = SpecialistLineage.root(
                non_mcts, parent_controller="parent-controller", operator="goal_emitter",
                source_goal_id=non_mcts_goal.goal_id,
            )
            factory._lineages[mcts.semantic_id()] = SpecialistLineage.root(
                mcts, parent_controller="parent-controller", operator="mcts",
                source_goal_id=mcts_source_goal_id,
            )
            batch = factory.propose(
                count=2, seed=20260826, recursive=True, capability_level=1,
            )
            return tuple(module.semantic_id() for module in batch.modules)

        self.assertEqual(
            recursive_semantics(None),
            recursive_semantics(mcts_goal.goal_id),
        )

    def test_direct_goal_lineage_does_not_change_legacy_recursive_priority(self):
        from tools.bkg4_search.invariant_factory import SpecialistFactory
        from tools.bkg4_search.invariant_groups import GroupOutcome
        from tools.bkg4_search.invariant_search import default_search_goals, emit_skeletons, local_structural_neighborhood
        from tools.bkg4_search.invariant_specialists import SpecialistLineage, analyze_specialist

        routes = (29, 64, 70, 211, 212)
        goals = default_search_goals(routes)
        first_goal = next(goal for goal in goals if goal.goal_id == "opening_route_regime")
        second_goal = next(goal for goal in goals if goal.goal_id == "endgame_liquidation")
        first_root = emit_skeletons(first_goal, n=1, seed=17, route_ids=routes)[0]
        second_root = emit_skeletons(second_goal, n=1, seed=19, route_ids=routes)[0]
        first_child = local_structural_neighborhood(first_root, route_ids=routes, budget=1, seed=23)[0]
        second_child = local_structural_neighborhood(second_root, route_ids=routes, budget=1, seed=29)[0]

        def evidence(module, useful_groups: int):
            rows = tuple(
                GroupOutcome(f"g{i}", False, False, True, i < useful_groups, 0, 1000 if i < useful_groups else 0)
                for i in range(3)
            )
            return analyze_specialist(
                module, rows,
                cohort_by_group={f"g{i}": "cohort" for i in range(3)},
                activation_count=3, action_change_count=3,
            )

        def recursive_semantics(inherit_first_goal: bool):
            backend = InvariantSearchBackend(route_ids=routes, evaluator=object())
            factory = SpecialistFactory(backend, root_controller="parent-controller")
            for item in (evidence(first_child, 3), evidence(second_child, 2)):
                backend.specialist_archive.consider(item)
            first_lineage = SpecialistLineage.root(
                first_root, parent_controller="parent-controller", operator="goal_emitter",
                source_goal_id=first_goal.goal_id,
            ).child(
                first_child, operator="local_structural_neighborhood", source_margin_delta=1.0,
                source_goal_id=(first_goal.goal_id if inherit_first_goal else None),
            )
            second_lineage = SpecialistLineage.root(
                second_root, parent_controller="parent-controller", operator="goal_emitter",
                source_goal_id=second_goal.goal_id,
            ).child(
                second_child, operator="local_structural_neighborhood", source_margin_delta=1.0,
                source_goal_id=second_goal.goal_id,
            )
            for lineage in (first_lineage, second_lineage):
                factory._lineages[lineage.root_semantic_id] = (
                    SpecialistLineage.root(first_root, parent_controller="parent-controller", operator="goal_emitter", source_goal_id=first_goal.goal_id)
                    if lineage.root_semantic_id == first_root.semantic_id()
                    else SpecialistLineage.root(second_root, parent_controller="parent-controller", operator="goal_emitter", source_goal_id=second_goal.goal_id)
                )
                factory._lineages[lineage.module.semantic_id()] = lineage
            batch = factory.propose(
                count=2, seed=20260827, recursive=True, capability_level=1,
            )
            return tuple(module.semantic_id() for module in batch.modules)

        self.assertEqual(
            recursive_semantics(True),
            recursive_semantics(False),
        )


if __name__ == "__main__":
    unittest.main()
