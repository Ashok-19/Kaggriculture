from __future__ import annotations

import unittest
from unittest import mock

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
        stats={
            "systematic":{"attempts":20,"production_insertions":1,"repair_insertions":15,"useful_safe":0,"unsafe":18,"new_loss":17},
            "local_refine":{"attempts":20,"production_insertions":10,"repair_insertions":0,"useful_safe":8,"unsafe":1,"new_loss":1},
        }
        adaptive=s.allocations(140,escalation_level=0,emitter_stats=stats)
        self.assertGreater(adaptive["local_refine"],adaptive["systematic"])
        refill=s.refill_order(stats)
        self.assertEqual(refill[0],"local_refine")
        self.assertEqual(refill[-1],"systematic")
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

    def test_shared_typed_pool_scales_to_emitter_budget_instead_of_fixed_192_floor(self):
        from tools.bkg4_search.edit_search import emitters as emitter_module
        from tools.bkg4_search.edit_search.search import SearchEngine
        p = root(); archive = SearchArchive.new(root_genome=p, base_identity="b", route_ids=ROUTES, run_id="r", master_seed=1)
        observations=[{"feature":"OWN_MONEY","index":0,"value":x} for x in (100,500,1000,2500,5000,9000)]
        requested=[]; real=emitter_module.generate_typed_candidates
        def wrapped(*args,**kwargs):
            requested.append(int(kwargs.get("n",0)))
            return real(*args,**kwargs)
        with mock.patch.object(emitter_module,"generate_typed_candidates",side_effect=wrapped):
            rows=SearchEngine(route_ids=ROUTES).propose_generation(archive,observations=observations,total=32,generation=1,seed=13)
        self.assertEqual(len(rows),32)
        self.assertTrue(requested)
        self.assertLessEqual(max(requested),64)

    def test_generation_builds_expanded_base_context_once_across_multiple_parents(self):
        from tools.bkg4_search.edit_search.archive import SearchArchive
        from tools.bkg4_search.edit_search.emitters import EmitterContext
        from tools.bkg4_search.edit_search.search import SearchEngine
        p=root(); archive=SearchArchive.new(root_genome=p,base_identity="b",route_ids=ROUTES,run_id="r",master_seed=1)
        first=EditProgram.build(p,(EditAction("CHANGE_ROUTE",3,route=70),),ROUTES,emitter="seed",seed=1)
        child=first.reconstruct(p,ROUTES); archive.add_program(first,child)
        archive.consider(first,PairEvaluation.from_states(ev(p.semantic_id(),[100,-10,200]),ev(child.semantic_id(),[110,5,220])))
        observations=[{"feature":"OWN_MONEY","index":0,"value":x} for x in (100,500,1000,2500,5000,9000)]
        real=EmitterContext.from_observations
        with mock.patch.object(EmitterContext,"from_observations",wraps=real) as build:
            rows=SearchEngine(route_ids=ROUTES).propose_generation(archive,observations=observations,total=32,generation=2,seed=17)
        self.assertEqual(len(rows),32)
        self.assertEqual(build.call_count,1)
        self.assertGreater(len({x.source_parent_semantic_id for x in rows}),1)

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
        archive.bind_case_bank("bank", ("c0", "c1"))
        archive.register_panel("r/g4/full", ("c0", "c1"))
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

    def test_production_parent_schedule_prefers_feasible_and_near_feasible_over_broad_novelty(self):
        from tools.bkg4_search.edit_search.archive import SearchArchive
        from tools.bkg4_search.edit_search.search import SearchEngine
        p=root(); archive=SearchArchive.new(root_genome=p,base_identity="b",route_ids=ROUTES,run_id="r",master_seed=1)
        archive.production_niches={
            "p0":{"child_semantic_id":p.semantic_id(),"program_id":"root"},
        }
        # parent_pools is intentionally tested as a pure policy surface; records point at registered states.
        archive.repair_niches={"r0":{"child_semantic_id":p.semantic_id(),"program_id":"root","repair_key":[1,1,0,0,1,1.0]}}
        archive.exploration_niches={"e0":{"child_semantic_id":p.semantic_id(),"program_id":"root"}}
        engine=SearchEngine(route_ids=ROUTES)
        schedule=engine.parent_schedule(archive,total=20,seed=77)
        self.assertEqual(len(schedule),20)
        self.assertGreaterEqual(sum(x[0]=="feasible" for x in schedule),10)
        self.assertGreaterEqual(sum(x[0]=="repair" for x in schedule),3)
        self.assertLessEqual(sum(x[0]=="exploration" for x in schedule),4)
        self.assertEqual(schedule,engine.parent_schedule(archive,total=20,seed=77))

    def test_goal_aware_generation_allocates_contrastive_proposals_deterministically(self):
        from tools.bkg4_search.edit_search.archive import SearchArchive
        from tools.bkg4_search.edit_search.panels import CaseBank,CaseRecord
        from tools.bkg4_search.edit_search.search import SearchEngine
        p=ModuleGenome(0,()); archive=SearchArchive.new(root_genome=p,base_identity="b",route_ids=ROUTES,run_id="r",master_seed=1)
        bank=CaseBank((
            CaseRecord("loss","/tmp/l",0,1,"f0",protected=True,baseline_margin=-100),
            CaseRecord("guard","/tmp/g",1,2,"f1",protected=True,baseline_margin=1000),
            CaseRecord("strong","/tmp/s",0,3,"f2",protected=True,baseline_margin=10000),
        ))
        contexts={
            "loss":({"feature":"OWN_MONEY","index":0,"value":100},),
            "guard":({"feature":"OWN_MONEY","index":0,"value":2000},),
            "strong":({"feature":"OWN_MONEY","index":0,"value":5000},),
        }
        engine=SearchEngine(route_ids=ROUTES)
        a=engine.propose_generation(archive,observations=[{"feature":"OWN_MONEY","index":0,"value":100}],total=24,generation=1,seed=99,case_bank=bank,case_contexts=contexts)
        emitters=[x.emitter for x in a]
        self.assertIn("contrastive",emitters)
        archive2=SearchArchive.new(root_genome=p,base_identity="b",route_ids=ROUTES,run_id="r",master_seed=1)
        engine2=SearchEngine(route_ids=ROUTES)
        b=engine2.propose_generation(archive2,observations=[{"feature":"OWN_MONEY","index":0,"value":100}],total=24,generation=1,seed=99,case_bank=bank,case_contexts=contexts)
        self.assertEqual([x.program_id for x in a],[x.program_id for x in b])

    def test_contrastive_budget_spans_multiple_failure_goals_for_one_parent(self):
        from tools.bkg4_search.edit_search.archive import SearchArchive
        from tools.bkg4_search.edit_search.goals import SearchGoal
        from tools.bkg4_search.edit_search.search import SearchEngine
        p=ModuleGenome(0,()); archive=SearchArchive.new(root_genome=p,base_identity="b",route_ids=ROUTES,run_id="r",master_seed=1)
        goals=(
            SearchGoal(("loss-a",),("guard",),(),()),
            SearchGoal(("loss-b",),("guard",),(),()),
        )
        contexts={
            "loss-a":({"feature":"OWN_MONEY","index":0,"value":100},),
            "loss-b":({"feature":"OPP_MONEY","index":0,"value":100},),
            "guard":(
                {"feature":"OWN_MONEY","index":0,"value":2000},
                {"feature":"OPP_MONEY","index":0,"value":2000},
            ),
        }
        engine=SearchEngine(route_ids=ROUTES)
        programs=engine.propose_generation(
            archive,observations=[{"feature":"OWN_MONEY","index":0,"value":100}],
            total=64,generation=1,seed=31337,search_goals=goals,case_contexts=contexts,
        )
        features=set()
        for program in programs:
            if program.emitter!="contrastive": continue
            child=program.reconstruct(p,ROUTES)
            features.update(c.feature for c in child.rules[-1].conditions if c.feature!="STEP")
        self.assertIn("OWN_MONEY",features)
        self.assertIn("OPP_MONEY",features)

    def test_goal_aware_refill_does_not_let_contrastive_monopolize_unused_budget(self):
        from tools.bkg4_search.edit_search.archive import SearchArchive
        from tools.bkg4_search.edit_search.goals import SearchGoal
        from tools.bkg4_search.edit_search.search import SearchEngine
        p=ModuleGenome(0,()); archive=SearchArchive.new(root_genome=p,base_identity="b",route_ids=ROUTES,run_id="r",master_seed=1)
        goal=SearchGoal(("loss",),("guard",),(),())
        contexts={
            "loss":({"feature":"OWN_MONEY","index":0,"value":100},),
            "guard":({"feature":"OWN_MONEY","index":0,"value":2000},),
        }
        engine=SearchEngine(route_ids=ROUTES)
        total=32
        engine.propose_generation(
            archive,observations=[{"feature":"OWN_MONEY","index":0,"value":100}],
            total=total,generation=1,seed=101,search_goals=(goal,),case_contexts=contexts,
        )
        scheduled=engine.scheduler.allocations(total,0)["contrastive"]
        self.assertLessEqual(int(engine.last_proposal_stats["emitter_raw"].get("contrastive",0)),scheduled)

    def test_search_caches_contrastive_synthesis_once_per_goal_across_parents(self):
        from tools.bkg4_search.edit_search.archive import SearchArchive
        from tools.bkg4_search.edit_search.goals import SearchGoal
        from tools.bkg4_search.edit_search.panels import CaseBank,CaseRecord
        from tools.bkg4_search.edit_search.search import SearchEngine
        p=ModuleGenome(0,()); archive=SearchArchive.new(root_genome=p,base_identity="b",route_ids=ROUTES,run_id="r",master_seed=1)
        seed_program=EditProgram.build(
            p,(EditAction("ADD_SPECIALIST",0,feature="OWN_MONEY",cmp="GE",value=5000,route=22),),
            ROUTES,emitter="systematic",seed=1,
        )
        child=seed_program.reconstruct(p,ROUTES); archive.add_program(seed_program,child)
        archive.production_niches={"seed":{"child_semantic_id":child.semantic_id(),"program_id":seed_program.program_id}}
        goal=SearchGoal(("loss",),("guard",),(),())
        bank=CaseBank((
            CaseRecord("loss","/tmp/l",0,1,"f0",baseline_margin=-1),
            CaseRecord("guard","/tmp/g",1,2,"f1",baseline_margin=1),
        ))
        contexts={
            "loss":({"feature":"OWN_MONEY","index":0,"value":100},),
            "guard":({"feature":"OWN_MONEY","index":0,"value":2000},),
        }
        pre=[(ConditionGene("OWN_MONEY","LE",0,100),)]
        engine=SearchEngine(route_ids=ROUTES)
        with mock.patch(
            "tools.bkg4_search.edit_search.search.synthesize_contrastive_conditions",
            return_value=pre,create=True,
        ) as synth:
            first=engine.propose_generation(
                archive,observations=[{"feature":"OWN_MONEY","index":0,"value":100}],
                total=64,generation=1,seed=77,search_goals=(goal,),case_contexts=contexts,case_bank=bank,
            )
            second=engine.propose_generation(
                archive,observations=[{"feature":"OWN_MONEY","index":0,"value":100}],
                total=64,generation=2,seed=78,search_goals=(goal,),case_contexts=contexts,case_bank=bank,
            )
        self.assertEqual(synth.call_count,1)
        self.assertIn("contrastive",[x.emitter for x in first+second])

    def test_research_novelty_does_not_mask_strategic_stagnation(self):
        from tools.bkg4_search.edit_search.search import GenerationSignals, StagnationMonitor
        monitor = StagnationMonitor(patience=1)
        signals = GenerationSignals(
            proposed=64, unique=64, archive_insertions=0, useful_safe=0,
            target_improvements=0, novelty_mean=0.95, new_niches=20,
            duplicate_rate=0.0, production_insertions=0, repair_insertions=0,
        )
        event = monitor.observe(signals)
        self.assertTrue(event.escalated)
        self.assertEqual(event.level, 1)

    def test_adaptive_emitter_allocation_rewards_safe_contribution_without_losing_random_floor(self):
        from tools.bkg4_search.edit_search.search import FixedEmitterScheduler
        scheduler = FixedEmitterScheduler(random_floor=0.05)
        stats = {
            "contrastive": {"attempts": 20, "unique": 20, "archive_insertions": 12, "useful_safe": 10},
            "systematic": {"attempts": 20, "unique": 20, "archive_insertions": 1, "useful_safe": 0},
        }
        allocation = scheduler.allocations(100, escalation_level=0, emitter_stats=stats)
        self.assertGreater(allocation["contrastive"], allocation["systematic"])
        self.assertGreaterEqual(allocation["random_typed"], 5)
        self.assertEqual(sum(allocation.values()), 100)

    def test_parent_lanes_keep_research_only_states_out_of_normal_production_parent_pool(self):
        from tools.bkg4_search.edit_search.archive import SearchArchive
        p = root(); archive = SearchArchive.new(root_genome=p, base_identity="b", route_ids=ROUTES, run_id="r", master_seed=1)
        unsafe_program = EditProgram.build(p, (EditAction("CHANGE_WINDOW",0,prefix_step=200),), ROUTES, emitter="systematic", seed=1)
        unsafe_child = unsafe_program.reconstruct(p, ROUTES); archive.add_program(unsafe_program, unsafe_child)
        base = ev(p.semantic_id(), [100, 100]); unsafe = ev(unsafe_child.semantic_id(), [-1, 500])
        archive.consider(unsafe_program, PairEvaluation.from_states(base, unsafe))
        self.assertNotIn(unsafe_child.semantic_id(), archive.production_parent_state_ids())
        self.assertIn(unsafe_child.semantic_id(), archive.repair_parent_state_ids())
        self.assertNotIn(unsafe_child.semantic_id(), archive.parent_state_ids())

    def test_contrastive_emitter_receives_first_class_scheduler_budget(self):
        from tools.bkg4_search.edit_search.search import FixedEmitterScheduler
        allocation = FixedEmitterScheduler().allocations(64, escalation_level=0)
        self.assertIn("contrastive", allocation)
        self.assertGreater(allocation["contrastive"], 0)

    def test_search_engine_routes_contrastive_goal_context_into_real_proposals(self):
        from tools.bkg4_search.edit_search.goals import SearchGoal
        from tools.bkg4_search.edit_search.search import SearchEngine
        p = root()
        archive = SearchArchive.new(root_genome=p, base_identity="b", route_ids=ROUTES, run_id="r", master_seed=1)
        goal = SearchGoal(("loss",), ("guard",), (), ())
        contexts = {
            "loss": ({"feature":"OWN_MONEY","index":0,"value":100}, {"feature":"DAY","index":0,"value":2}),
            "guard": ({"feature":"OWN_MONEY","index":0,"value":2000}, {"feature":"DAY","index":0,"value":2}),
        }
        obs = [{"feature":"OWN_MONEY","index":0,"value":x} for x in (100,500,1000,2000,5000)]
        engine = SearchEngine(route_ids=ROUTES)
        programs = engine.propose_generation(
            archive, observations=obs, total=32, generation=1, seed=77,
            search_goals=(goal,), case_contexts=contexts,
        )
        contrastive = [x for x in programs if x.emitter == "contrastive"]
        self.assertTrue(contrastive)
        self.assertGreater(engine.last_proposal_stats["emitter_raw"].get("contrastive",0),0)
    def test_certification_labels_never_reset_search_stagnation(self):
        from tools.bkg4_search.edit_search.search import GenerationSignals, StagnationMonitor
        base=dict(proposed=32,unique=32,archive_insertions=0,useful_safe=0,target_improvements=0,novelty_mean=1.0,new_niches=0)
        stalled=GenerationSignals(**base)
        certified=GenerationSignals(**base,certified_safe_useful=9,certified_elite=9,generalization_improved=9)
        self.assertTrue(StagnationMonitor.is_stalled(stalled))
        self.assertTrue(StagnationMonitor.is_stalled(certified),"held-out certification labels must never alter future search escalation")

    def test_certification_quality_map_prefers_certified_state(self):
        from tools.bkg4_search.edit_search.certification import CertificationClass
        from tools.bkg4_search.edit_search.search import certification_quality_map
        rows={"a":CertificationClass.UNCERTIFIED,"b":CertificationClass.CERTIFIED_SAFE_USEFUL,"c":CertificationClass.CERTIFIED_ELITE_FLIP,"d":CertificationClass.GENERALIZATION_HARD_NEGATIVE}
        q=certification_quality_map(rows)
        self.assertGreater(q["c"],q["b"])
        self.assertGreater(q["b"],q["a"])
        self.assertGreater(q["a"],q["d"])

    def test_archive_quality_map_ignores_certification_labels_for_search(self):
        from tools.bkg4_search.edit_search.certification import CertificationClass
        from tools.bkg4_search.edit_search.search import archive_quality_map
        p=root(); archive=SearchArchive.new(root_genome=p,base_identity="b",route_ids=ROUTES,run_id="r",master_seed=1)
        d=EditProgram.build(p,(EditAction("CHANGE_WINDOW",0,prefix_step=145),),ROUTES,emitter="x",seed=1); dc=d.reconstruct(p,ROUTES); archive.add_program(d,dc)
        r=EditProgram.build(p,(EditAction("CHANGE_WINDOW",0,prefix_step=146),),ROUTES,emitter="x",seed=2); rc=r.reconstruct(p,ROUTES); archive.add_program(r,rc)
        archive.production_niches={"p":{"child_semantic_id":dc.semantic_id(),"program_id":d.program_id}}
        archive.repair_niches={"r":{"child_semantic_id":rc.semantic_id(),"program_id":r.program_id,"repair_key":[1]}}
        local={p.semantic_id():0.0,dc.semantic_id():0.0,rc.semantic_id():0.0}
        q0=archive_quality_map(archive,local)
        q1=archive_quality_map(archive,local,certification_classes={dc.semantic_id():CertificationClass.CERTIFIED_ELITE_FLIP,rc.semantic_id():CertificationClass.GENERALIZATION_HARD_NEGATIVE})
        self.assertEqual(q0,q1,"held-out certification evidence must be a label sink, never a search-quality input")
        self.assertGreater(q0[dc.semantic_id()],q0[rc.semantic_id()])
        self.assertGreaterEqual(q0[p.semantic_id()],q0[rc.semantic_id()])

    def test_stage2_selection_caps_screen_noops_instead_of_padding(self):
        from tools.bkg4_search.edit_search.search import select_stage2_finalists
        p=root()
        programs=[]; pairs={}
        parent=ev(p.semantic_id(),[-10,100])
        for i in range(20):
            pr=EditProgram.build(p,(EditAction("CHANGE_WINDOW",0,prefix_step=200+i),),ROUTES,emitter="x",seed=i)
            child=pr.reconstruct(p,ROUTES); programs.append(pr)
            if i<10:
                ce=StateEvaluation(child.semantic_id(),"p",("c0","c1"),(-10,100),(0,0),(0,0),(100,101))
            else:
                ce=StateEvaluation(child.semantic_id(),"p",("c0","c1"),(-9,101),(1,1),(1,1),(200+i,300+i))
            pairs[pr.program_id]=PairEvaluation.from_states(parent,ce)
        selected=select_stage2_finalists(programs,pairs,budget=10,max_noop_fraction=0.10)
        noops=sum(bool(pairs[x.program_id].certified_aggregate()["exact_noop"]) for x in selected)
        self.assertLessEqual(noops,1)
        self.assertEqual(len(selected),10)


if __name__ == "__main__":
    unittest.main(verbosity=2)
