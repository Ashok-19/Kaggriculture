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


    def test_checkpoint_resume_preserves_stagnation_regime_and_next_proposals(self):
        import tempfile
        from pathlib import Path
        from tools.bkg4_search.edit_search.archive import SearchArchive
        from tools.bkg4_search.edit_search.search import SearchEngine
        p = root(); obs = [{"feature":"OWN_MONEY","index":0,"value":x} for x in (500,1000,2500,5000)]
        a = SearchArchive.new(root_genome=p, base_identity="b", route_ids=ROUTES, run_id="r", master_seed=7)
        direct = SearchEngine(route_ids=ROUTES); direct.stagnation.level = 4; direct.stagnation._stalled = 2
        direct.persist_runtime(a, generation=5)
        with tempfile.TemporaryDirectory() as td:
            path = Path(td)/"a.json"; a.save(path); loaded = SearchArchive.load(path)
            resumed = SearchEngine.from_archive(loaded, route_ids=ROUTES)
            self.assertEqual(resumed.stagnation.level, 4); self.assertEqual(resumed.stagnation._stalled, 2)
            d = direct.propose_generation(a, observations=obs, total=20, generation=6, seed=91)
            r = resumed.propose_generation(loaded, observations=obs, total=20, generation=6, seed=91)
            self.assertEqual([x.program_id for x in d], [x.program_id for x in r])

    def test_persisted_exact_certification_rebuilds_identical_mcts_quality(self):
        import tempfile
        from pathlib import Path
        from tools.bkg4_search.edit_search.archive import SearchArchive
        from tools.bkg4_search.edit_search.search import exact_state_quality_map
        p = root(); archive = SearchArchive.new(root_genome=p, base_identity="b", route_ids=ROUTES, run_id="r", master_seed=1)
        cert = StateEvaluation(p.semantic_id(), "r/g4/full", ("c0","c1"), (-3,20), (0,0), (0,0), (1,2))
        archive.record_state_evaluation(cert)
        before = exact_state_quality_map({p.semantic_id(): cert})
        with tempfile.TemporaryDirectory() as td:
            path = Path(td)/"a.json"; archive.save(path); loaded = SearchArchive.load(path)
            restored = loaded.state_evaluation(p.semantic_id(), "r/g4/full")
            after = exact_state_quality_map({p.semantic_id(): restored})
        self.assertEqual(before, after)

    def test_level_one_widens_real_proposal_neighborhood(self):
        from tools.bkg4_search.edit_search.archive import SearchArchive
        from tools.bkg4_search.edit_search.search import SearchEngine
        p = root(); obs = [{"feature":"OWN_MONEY","index":0,"value":x} for x in (500,1000,2500,5000,9000)]
        a0 = SearchArchive.new(root_genome=p, base_identity="b", route_ids=ROUTES, run_id="r", master_seed=1)
        a1 = SearchArchive.new(root_genome=p, base_identity="b", route_ids=ROUTES, run_id="r", master_seed=1)
        e0 = SearchEngine(route_ids=ROUTES); e1 = SearchEngine(route_ids=ROUTES); e1.stagnation.level = 1
        x0 = e0.propose_generation(a0, observations=obs, total=48, generation=3, seed=123)
        x1 = e1.propose_generation(a1, observations=obs, total=48, generation=3, seed=123)
        self.assertNotEqual({x.child_semantic_id for x in x0}, {x.child_semantic_id for x in x1})
        def window_dist(programs):
            vals=[]
            for pr in programs:
                for atom in pr.atoms:
                    if atom.operation == "CHANGE_WINDOW": vals.append(abs(int(atom.action["prefix_step"]) - p.prefix_step))
            return max(vals, default=0)
        self.assertGreater(window_dist(x1), window_dist(x0))

    def test_production_mcts_receives_aligned_exact_state_quality_and_rejects_incompatible_panels(self):
        from tools.bkg4_search.edit_search.archive import SearchArchive
        from tools.bkg4_search.edit_search.fitness import StateEvaluation
        from tools.bkg4_search.edit_search.search import SearchEngine, exact_state_quality_map
        p = root(); archive = SearchArchive.new(root_genome=p, base_identity="b", route_ids=ROUTES, run_id="r", master_seed=1)
        ev = StateEvaluation(p.semantic_id(), "r/g2/full", ("c0","c1"), (-5,10), (0,0), (0,0), (1,2))
        quality = exact_state_quality_map({p.semantic_id(): ev})
        self.assertEqual(set(quality), {p.semantic_id()})
        engine = SearchEngine(route_ids=ROUTES)
        engine.propose_generation(archive, observations=[{"feature":"OWN_MONEY","index":0,"value":1000}], total=24, generation=3, seed=4, parent_evaluations={p.semantic_id():ev})
        self.assertEqual(engine.last_proposal_stats["quality_state_count"], 1)
        self.assertEqual(engine.last_proposal_stats["quality_panel_id"], "r/g2/full")
        other = StateEvaluation("other", "other/full", ("c0","c1"), (1,2), (0,0), (0,0), (3,4))
        with self.assertRaises(ValueError):
            exact_state_quality_map({p.semantic_id():ev, "other":other})

if __name__ == "__main__":
    unittest.main(verbosity=2)
