from __future__ import annotations

import unittest
from unittest import mock

from tools.bkg4_search.module_codec import ConditionGene, ModuleGenome, RuleGene

ROUTES = (22, 23, 29, 32, 34, 46, 63, 64, 70, 77, 79, 211, 212, 213, 214)


def parent0(prefix=37):
    return ModuleGenome(prefix, ())


def parent2(prefix=144):
    return ModuleGenome(prefix, (
        RuleGene((ConditionGene("SHOP_AT", "EQ", 0, 4), ConditionGene("OWN_MONEY", "LE", 0, 3000)), 213),
        RuleGene((ConditionGene("MARKET_PRICE", "GE", 3, 120),), 214),
    ))


def donor3():
    return ModuleGenome(144, (
        RuleGene((ConditionGene("DAY", "GE", 0, 4),), 22),
        RuleGene((ConditionGene("OPP_MONEY", "GE", 0, 5000),), 70),
        RuleGene((ConditionGene("MARKET_INVENTORY", "LE", 3, 20),), 77),
    ))


class EditSearchEmitterTests(unittest.TestCase):
    def context(self, donors=()):
        from tools.bkg4_search.edit_search.emitters import EmitterContext
        rows = [
            {"feature": "OWN_MONEY", "index": 0, "value": x} for x in (500, 1000, 2500, 5000, 9000)
        ] + [
            {"feature": "MARKET_PRICE", "index": 3, "value": x} for x in (30, 81, 120, 172, 240)
        ] + [{"feature": "OPP_UNITS", "index": 0, "value": x} for x in (2, 4, 8, 12)]
        return EmitterContext.from_observations(ROUTES, rows, donor_genomes=donors, generation_depth=4, escalation_level=0)

    def test_systematic_searches_what_where_how_and_two_rule_priority(self):
        from tools.bkg4_search.edit_search.emitters import SystematicEmitter
        p = parent2()
        xs = SystematicEmitter().propose(p, n=256, seed=11, context=self.context())
        ops = {x.atoms[0].operation for x in xs}
        self.assertIn("CHANGE_PRIORITY", ops)
        self.assertIn("CHANGE_THRESHOLD", ops)
        self.assertIn("CHANGE_ROUTE", ops)
        self.assertIn("CHANGE_WINDOW", ops)
        self.assertIn("ADD_CONDITION", ops)
        self.assertGreaterEqual(len({(x.atoms[0].operation, x.atoms[0].target_node) for x in xs}), 8)
        self.assertTrue(any(x.reconstruct(p, ROUTES).prefix_step != p.prefix_step for x in xs))
        self.assertTrue(any(dict(x.atoms[0].action).get("value") in {500, 1000, 2500, 5000, 9000} for x in xs))

    def test_zero_rule_parent_is_searchable_and_not_forced_to_prefix_216(self):
        from tools.bkg4_search.edit_search.emitters import SystematicEmitter, RandomTypedEmitter
        p = parent0(37)
        xs = SystematicEmitter().propose(p, n=64, seed=2, context=self.context())
        ys = RandomTypedEmitter().propose(p, n=64, seed=3, context=self.context())
        self.assertTrue(xs and ys)
        for program in xs + ys:
            program.reconstruct(p, ROUTES)
        self.assertTrue(any(x.child_semantic_id != p.semantic_id() for x in xs))
        self.assertTrue(any(x.reconstruct(p, ROUTES).prefix_step != 216 for x in xs))

    def test_local_and_structural_emitters_use_evolved_parent(self):
        from tools.bkg4_search.edit_search.emitters import LocalRefinementEmitter, StructuralEmitter
        p = donor3()
        for emitter in (LocalRefinementEmitter(), StructuralEmitter()):
            xs = emitter.propose(p, n=96, seed=7, context=self.context())
            self.assertTrue(xs, emitter.name)
            self.assertTrue(all(x.source_parent_semantic_id == p.semantic_id() for x in xs))
            self.assertTrue(all(x.reconstruct(p, ROUTES).semantic_id() == x.child_semantic_id for x in xs))
        self.assertTrue(any(len(x.reconstruct(p, ROUTES).rules) >= 3 for x in StructuralEmitter().propose(p, n=96, seed=8, context=self.context())))

    def test_one_slot_recursive_budgets_rotate_across_legal_operation_target_families(self):
        from tools.bkg4_search.edit_search.emitters import LocalRefinementEmitter, StructuralEmitter
        p = parent2()
        ctx = self.context()
        local = [
            x for seed in range(100, 116)
            for x in LocalRefinementEmitter().propose(p, n=1, seed=seed, context=ctx)
        ]
        structural = [
            x for seed in range(100, 116)
            for x in StructuralEmitter().propose(p, n=1, seed=seed, context=ctx)
        ]
        local_ops = {x.atoms[0].operation for x in local}
        structural_ops = {x.atoms[0].operation for x in structural}
        self.assertIn("CHANGE_THRESHOLD", local_ops)
        self.assertTrue(any(x.atoms[0].target_kind == "condition" for x in local))
        self.assertGreaterEqual(len(local_ops), 4)
        self.assertGreaterEqual(len(structural_ops), 4)

    def test_semantic_recombination_and_lns_make_reconstructable_multiedit_jumps(self):
        from tools.bkg4_search.edit_search.emitters import RecombinationEmitter, LargeNeighborhoodEmitter
        p = parent2()
        ctx = self.context(donors=(donor3(),))
        recombined = RecombinationEmitter().propose(p, n=32, seed=13, context=ctx)
        lns = LargeNeighborhoodEmitter().propose(p, n=32, seed=14, context=ctx)
        self.assertTrue(recombined)
        self.assertTrue(lns)
        self.assertTrue(any(x.program_length >= 2 for x in recombined + lns))
        for program in recombined + lns:
            self.assertEqual(program.reconstruct(p, ROUTES).semantic_id(), program.child_semantic_id)

    def test_mcts_emitter_explores_bounded_multistep_derivations(self):
        from tools.bkg4_search.edit_search.emitters import EvolvabilityMCTSEmitter
        p = parent2()
        xs = EvolvabilityMCTSEmitter(max_depth=3).propose(p, n=24, seed=31, context=self.context())
        self.assertTrue(xs)
        self.assertTrue(any(x.program_length >= 2 for x in xs))
        self.assertLessEqual(max(x.program_length for x in xs), 3)
        self.assertGreater(len({x.child_semantic_id for x in xs}), 12)

    def test_mcts_tiny_budget_does_not_pay_fixed_64_expansion_floor(self):
        from tools.bkg4_search.edit_search import emitters as emitter_module
        from tools.bkg4_search.edit_search.emitters import EvolvabilityMCTSEmitter
        p = parent2()
        real = emitter_module.generate_typed_candidates
        with mock.patch.object(emitter_module, "generate_typed_candidates", wraps=real) as generate:
            xs = EvolvabilityMCTSEmitter(max_depth=3).propose(p, n=1, seed=37, context=self.context())
        self.assertTrue(xs)
        self.assertLessEqual(generate.call_count, 8)
        self.assertLessEqual(max(x.program_length for x in xs), 3)

    def test_mcts_candidate_pool_is_cached_per_semantic_state(self):
        from tools.bkg4_search.edit_search import emitters as emitter_module
        from tools.bkg4_search.edit_search.emitters import _mcts_candidate_pool
        p = parent2(); ctx = self.context(); cache = {}; row_cache = {}
        real = emitter_module.generate_typed_candidates
        with mock.patch.object(emitter_module, "generate_typed_candidates", wraps=real) as generate:
            first = _mcts_candidate_pool(p, ctx, seed=41, cache=cache, row_cache=row_cache, pool_size=16)
            second = _mcts_candidate_pool(p, ctx, seed=999, cache=cache, row_cache=row_cache, pool_size=16)
        self.assertTrue(first)
        self.assertEqual(first, second)
        self.assertEqual(generate.call_count, 1)
        self.assertEqual(len(row_cache), 1)

    def test_mcts_exact_quality_changes_branch_preference_and_missing_quality_explores(self):
        from tools.bkg4_search.edit_search.emitters import EmitterContext, EvolvabilityMCTSEmitter
        from tools.bkg4_search.edit_search.schema import EditProgram
        from tools.edit_corpus.candidates import generate_typed_candidates
        p = parent2(); rows = self.context().context_rows; seed = 31
        base = EmitterContext.from_observations(ROUTES, rows, generation_depth=4)
        first = generate_typed_candidates(p, base.rows_for_parent(p), n=12, seed=seed + 1009, route_ids=ROUTES)[0]
        known = EditProgram.build(p, (first.action,), ROUTES, emitter="probe", seed=1).reconstruct(p, ROUTES)
        emitter = EvolvabilityMCTSEmitter(max_depth=3)
        none = emitter.propose(p, n=12, seed=seed, context=base)
        high = emitter.propose(p, n=12, seed=seed, context=EmitterContext.from_observations(ROUTES, rows, generation_depth=4, quality_by_state={known.semantic_id(): 20.0}))
        low = emitter.propose(p, n=12, seed=seed, context=EmitterContext.from_observations(ROUTES, rows, generation_depth=4, quality_by_state={known.semantic_id(): -20.0}))
        self.assertTrue(none and high and low)
        self.assertEqual(none[0].child_semantic_id, known.semantic_id())
        self.assertEqual(high[0].child_semantic_id, known.semantic_id())
        self.assertEqual(low[0].child_semantic_id, known.semantic_id())
        self.assertNotEqual([x.child_semantic_id for x in high], [x.child_semantic_id for x in low])
        self.assertNotEqual([x.program_length for x in high], [x.program_length for x in low])
        self.assertGreater(len({x.child_semantic_id for x in none}), 8)

    def test_motif_emitter_reinstantiates_template_against_compatible_future_parent(self):
        from tools.bkg4_search.edit_search.emitters import EmitterContext, MotifEmitter
        p = parent2(prefix=311)
        motif = {
            "motif_id": "motif:test",
            "safe_useful_hits": 5,
            "l2w_total": 3,
            "template": {
                "program_length": 1,
                "atoms": [{
                    "operation": "ADD_CONDITION",
                    "target_kind": "rule",
                    "target_rule_index_hint": 0,
                    "target_condition_index_hint": -1,
                    "action": {
                        "operation": "ADD_CONDITION", "feature": "OPP_MONEY", "cmp": "GE", "index": 0,
                        "value": 4000, "route": 0, "feature2": "DAY", "cmp2": "GE", "index2": 0,
                        "value2": 0, "condition_count": 1, "priority": 0, "prefix_step": 216,
                    },
                }],
            },
        }
        ctx = EmitterContext.from_observations(ROUTES, self.context().context_rows, motifs=(motif,), generation_depth=9)
        xs = MotifEmitter().propose(p, n=8, seed=44, context=ctx)
        self.assertTrue(xs)
        self.assertTrue(all(x.emitter == "motif" for x in xs))
        self.assertGreaterEqual(len({x.atoms[0].target_node for x in xs}), 2)
        for program in xs:
            self.assertEqual(program.reconstruct(p, ROUTES).semantic_id(), program.child_semantic_id)
            self.assertEqual(program.source_parent_semantic_id, p.semantic_id())


    def test_contrastive_specialist_emitter_uses_guard_excluding_predicates(self):
        import tools.bkg4_search.edit_search.emitters as emitters
        from tools.bkg4_search.edit_search.goals import SearchGoal, conjunction_stats
        self.assertTrue(hasattr(emitters,"ContrastiveSpecialistEmitter"))
        if not hasattr(emitters,"ContrastiveSpecialistEmitter"): return
        p=ModuleGenome(0,())
        contexts={
            "loss":({"feature":"OWN_MONEY","index":0,"value":100},{"feature":"DAY","index":0,"value":2}),
            "guard":({"feature":"OWN_MONEY","index":0,"value":2000},{"feature":"DAY","index":0,"value":2}),
        }
        goal=SearchGoal(("loss",),("guard",),(),())
        ctx=emitters.EmitterContext.from_observations(ROUTES,self.context().context_rows,generation_depth=1,search_goal=goal,case_contexts=contexts)
        rows=emitters.ContrastiveSpecialistEmitter().propose(p,n=8,seed=41,context=ctx)
        self.assertTrue(rows)
        for program in rows:
            child=program.reconstruct(p,ROUTES)
            self.assertTrue(child.rules)
            conds=child.rules[0].conditions
            diag=conjunction_stats(conds,goal,contexts)
            self.assertGreater(diag["target_hits"],0)
            self.assertEqual(diag["guard_hits"],0)

    def test_precomputed_parent_typed_pool_is_reusable_across_one_step_emitters(self):
        from tools.bkg4_search.edit_search.emitters import LocalRefinementEmitter, RandomTypedEmitter, StructuralEmitter, SystematicEmitter
        p = parent2()
        ctx = self.context().with_typed_pool(p, n=192, seed=12345)
        self.assertEqual(ctx.typed_pool_parent_semantic_id, p.semantic_id())
        self.assertGreaterEqual(len(ctx.typed_pool), 96)
        for emitter in (SystematicEmitter(), LocalRefinementEmitter(), StructuralEmitter(), RandomTypedEmitter()):
            rows = emitter.propose(p, n=8, seed=77, context=ctx)
            self.assertTrue(rows, emitter.name)
            for program in rows:
                self.assertEqual(program.reconstruct(p, ROUTES).semantic_id(), program.child_semantic_id)

    def test_contrastive_specialist_prefers_semantic_before_temporal_localizers(self):
        from tools.bkg4_search.edit_search.emitters import ContrastiveSpecialistEmitter, EmitterContext
        from tools.bkg4_search.edit_search.goals import SearchGoal, temporal_specificity
        p=ModuleGenome(0,())
        contexts={
            "loss":(
                {"probe_index":0,"probe_step":264,"feature":"OWN_MONEY","index":0,"value":100},
                {"probe_index":0,"probe_step":264,"feature":"DAY","index":0,"value":2},
                {"probe_index":1,"probe_step":288,"feature":"OWN_MONEY","index":0,"value":120},
                {"probe_index":1,"probe_step":288,"feature":"DAY","index":0,"value":12},
            ),
            "guard0":(
                {"probe_index":0,"probe_step":264,"feature":"OWN_MONEY","index":0,"value":2000},
                {"probe_index":0,"probe_step":264,"feature":"DAY","index":0,"value":2},
            ),
            "guard1":(
                {"probe_index":0,"probe_step":264,"feature":"OWN_MONEY","index":0,"value":2200},
                {"probe_index":0,"probe_step":264,"feature":"DAY","index":0,"value":9},
            ),
        }
        goal=SearchGoal(("loss",),("guard0","guard1"),(),())
        ctx=EmitterContext.from_observations(
            ROUTES,[{"feature":"OWN_MONEY","index":0,"value":100}],
            generation_depth=1,search_goal=goal,case_contexts=contexts,
        )
        rows=ContrastiveSpecialistEmitter().propose(p,n=40,seed=19,context=ctx)
        self.assertTrue(rows)
        children=[x.reconstruct(p,ROUTES) for x in rows]
        specs=[temporal_specificity(g.rules[-1].conditions) for g in children]
        self.assertEqual(specs[0],min(specs))
        self.assertEqual(specs,sorted(specs))
        self.assertEqual(specs[0][0],0)
        self.assertTrue(any(s[0]>0 for s in specs),"temporal localizers must remain available after semantic candidates")
        first_temporal=next(i for i,s in enumerate(specs) if s[0]>0)
        self.assertTrue(all(s[0]==0 for s in specs[:first_temporal]))
        self.assertTrue(any(x.program_length>=2 and len(g.rules[-1].conditions)>=3 for x,g in zip(rows,children) if temporal_specificity(g.rules[-1].conditions)[0]>0))
    def test_contrastive_emitter_uses_precomputed_guarded_condition_sets(self):
        from tools.bkg4_search.edit_search.emitters import ContrastiveSpecialistEmitter,EmitterContext
        from tools.bkg4_search.edit_search.goals import SearchGoal
        p=parent0(0); goal=SearchGoal(("loss",),("guard",),(),())
        contexts={
            "loss":({"feature":"OWN_MONEY","index":0,"value":100},),
            "guard":({"feature":"OWN_MONEY","index":0,"value":2000},),
        }
        pre=((ConditionGene("OWN_MONEY","LE",0,100),),)
        ctx=EmitterContext.from_observations(
            ROUTES,(),generation_depth=1,search_goal=goal,case_contexts=contexts,
            contrastive_condition_sets=pre,
        )
        with mock.patch(
            "tools.bkg4_search.edit_search.emitters.synthesize_contrastive_conditions",
            side_effect=AssertionError("precomputed condition sets should bypass synthesis"),
        ):
            rows=ContrastiveSpecialistEmitter().propose(p,n=4,seed=7,context=ctx)
        self.assertTrue(rows)
        self.assertTrue(all(x.emitter=="contrastive" for x in rows))


    def test_recombination_prefers_certified_donor_quality(self):
        from tools.bkg4_search.edit_search.emitters import EmitterContext, RecombinationEmitter
        parent=ModuleGenome(0,())
        weak=ModuleGenome(0,(RuleGene((ConditionGene("OWN_MONEY","GE",0,1000),),22),))
        strong=ModuleGenome(0,(RuleGene((ConditionGene("OPP_MONEY","GE",0,1000),),77),))
        ctx=EmitterContext.from_observations(
            ROUTES,[{"feature":"OWN_MONEY","index":0,"value":1000}],
            donor_genomes=(weak,strong), donor_quality_by_state={weak.semantic_id():100.0,strong.semantic_id():900.0},
        )
        out=RecombinationEmitter().propose(parent,n=1,seed=5,context=ctx)
        self.assertTrue(out)
        child=out[0].reconstruct(parent,ROUTES)
        self.assertEqual(child.rules[0].route,77)


if __name__ == "__main__":
    unittest.main(verbosity=2)
