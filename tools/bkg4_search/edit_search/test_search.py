from __future__ import annotations

import unittest

from tools.bkg4_search.bonsai_editformer.actor_grammar import EditAction
from tools.bkg4_search.module_codec import ConditionGene, ModuleGenome, RuleGene
from tools.bkg4_search.edit_search.archive import SearchArchive
from tools.bkg4_search.edit_search.fitness import PairEvaluation, StateEvaluation
from tools.bkg4_search.edit_search.schema import EditProgram

ROUTES = (22, 23, 29, 32, 34, 46, 63, 64, 70, 77, 79, 211, 212, 213, 214)


def root():
    return ModuleGenome(144, (RuleGene((ConditionGene("OWN_MONEY", "GE", 0, 1000),), 22),))


def ev(sid, margins):
    return StateEvaluation(sid, "p", tuple(f"c{i}" for i in range(len(margins))), tuple(margins), tuple([1]*len(margins)), tuple([1]*len(margins)), tuple(range(100,100+len(margins))))


class EditSearchLoopTests(unittest.TestCase):
    def test_fixed_scheduler_keeps_random_floor_and_escalates_search_regimes(self):
        from tools.bkg4_search.edit_search.search import FixedEmitterScheduler
        s = FixedEmitterScheduler()
        a0 = s.allocations(140, escalation_level=0)
        a4 = s.allocations(140, escalation_level=4)
        a5 = s.allocations(140, escalation_level=5)
        self.assertGreater(a0["random_typed"], 0)
        self.assertGreater(a4["random_typed"], 0)
        self.assertGreater(a5["random_typed"], 0)
        self.assertGreater(a4["lns"], a0["lns"])
        self.assertGreater(a5["evolvability_mcts"], a0["evolvability_mcts"])
        self.assertEqual(sum(a0.values()), 140)
        self.assertEqual(sum(a4.values()), 140)

    def test_stagnation_monitor_advances_and_records_real_escalation(self):
        from tools.bkg4_search.edit_search.search import GenerationSignals, StagnationMonitor
        m = StagnationMonitor(patience=2, max_level=6)
        healthy = GenerationSignals(100, 90, 12, 4, 3, 0.2, 5)
        stalled = GenerationSignals(100, 20, 0, 0, 0, 0.01, 0)
        self.assertEqual(m.observe(healthy).level, 0)
        self.assertEqual(m.observe(stalled).level, 0)
        e = m.observe(stalled)
        self.assertEqual(e.level, 1)
        self.assertTrue(e.escalated)
        self.assertIn("wider", e.reason)
        m.observe(stalled)
        e2 = m.observe(stalled)
        self.assertEqual(e2.level, 2)
        self.assertNotEqual(e.reason, e2.reason)

    def test_recursive_engine_selects_qd_descendant_as_future_parent(self):
        from tools.bkg4_search.edit_search.search import SearchEngine
        p = root()
        archive = SearchArchive.new(root_genome=p, base_identity="base", route_ids=ROUTES, run_id="run", master_seed=3)
        first = EditProgram.build(p, (EditAction("ADD_SPECIALIST", 0, feature="DAY", cmp="GE", value=4, route=70),), ROUTES, emitter="seed", seed=1)
        child = first.reconstruct(p, ROUTES)
        archive.add_program(first, child)
        base = ev(p.semantic_id(), [100, -10, 200])
        ce = ev(child.semantic_id(), [110, 5, 220])
        archive.consider(first, PairEvaluation.from_states(base, ce))
        engine = SearchEngine(route_ids=ROUTES)
        observations = [{"feature":"OWN_MONEY","index":0,"value":x} for x in (500,1000,2500,5000)]
        proposed = engine.propose_generation(archive, observations=observations, total=80, generation=2, seed=91)
        sources = {x.source_parent_semantic_id for x in proposed}
        self.assertIn(child.semantic_id(), sources)
        self.assertTrue(any(archive.genome(x.source_parent_semantic_id).rules for x in proposed))
        self.assertTrue(all(x.child_semantic_id != x.source_parent_semantic_id for x in proposed))

    def test_lexicase_parent_can_preserve_unique_case_specialist(self):
        from tools.bkg4_search.edit_search.search import lexicase_parent
        evals = {
            "general": ev("general", [10, 10, 10]),
            "specialist": ev("specialist", [1000, -100, -100]),
            "other": ev("other", [9, 20, 20]),
        }
        wins = {lexicase_parent(evals, seed=s) for s in range(32)}
        self.assertIn("specialist", wins)
        self.assertGreaterEqual(len(wins), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
