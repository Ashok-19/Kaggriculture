from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tools.bkg4_search.bonsai_editformer.actor_grammar import EditAction
from tools.bkg4_search.module_codec import ModuleGenome
from tools.bkg4_search.edit_search.archive import SearchArchive
from tools.bkg4_search.edit_search.fitness import StateEvaluation
from tools.bkg4_search.edit_search.panels import CaseBank, CasePanelManager, CaseRecord
from tools.bkg4_search.edit_search.schema import EditProgram

ROUTES = (22, 70)


class ToyExactEvaluator:
    """Unit-test oracle for orchestration only; production has no non-CUDA fallback."""
    def __init__(self, root_sid: str): self.root_sid = root_sid; self.calls = []
    def evaluate(self, genomes, panel, *, work_dir, panel_id):
        self.calls.append((tuple(g.semantic_id() for g in genomes), tuple(c.case_id for c in panel), panel_id))
        out = {}
        for g in genomes:
            margins=[]
            for c in panel:
                base = {"c0":100, "c1":-10, "c2":100, "c3":200}[c.case_id]
                if g.semantic_id() == self.root_sid: m = base
                elif g.prefix_step == 145: m = {"c0":110,"c1":5,"c2":-1,"c3":210}[c.case_id]
                else: m = {"c0":101,"c1":1,"c2":101,"c3":201}[c.case_id]
                margins.append(m)
            out[g.semantic_id()] = StateEvaluation(
                g.semantic_id(), panel_id, tuple(c.case_id for c in panel), tuple(margins),
                tuple([1]*len(panel)), tuple([2]*len(panel)), tuple(1000+i+(g.prefix_step*10) for i in range(len(panel))),
            )
        return out


class EditSearchMultiFidelityTests(unittest.TestCase):
    def test_broader_failure_becomes_counterexample_and_unsafe_state_stays_out_of_production(self):
        from tools.bkg4_search.edit_search.search import SearchEngine
        root = ModuleGenome(144, ())
        archive = SearchArchive.new(root_genome=root, base_identity="base", route_ids=ROUTES, run_id="r", master_seed=1)
        programs = [
            EditProgram.build(root, (EditAction("CHANGE_WINDOW",0,prefix_step=145),), ROUTES, emitter="systematic", seed=1),
            EditProgram.build(root, (EditAction("CHANGE_WINDOW",0,prefix_step=146),), ROUTES, emitter="random_typed", seed=2),
        ]
        for p in programs: archive.add_program(p, p.reconstruct(root, ROUTES))
        bank = CaseBank([
            CaseRecord("c0","/tmp/t0",0,1,"f0",baseline_margin=100),
            CaseRecord("c1","/tmp/t1",1,2,"f1",baseline_margin=-10),
            CaseRecord("c2","/tmp/t2",0,3,"f2",baseline_margin=100),
            CaseRecord("c3","/tmp/t3",1,4,"f3",baseline_margin=200),
        ])
        manager = CasePanelManager(bank, random_anchor_fraction=0.0)
        evaluator = ToyExactEvaluator(root.semantic_id())
        engine = SearchEngine(route_ids=ROUTES)
        with tempfile.TemporaryDirectory() as td:
            result = engine.evaluate_round(
                archive, programs, evaluator=evaluator, panel_manager=manager,
                screen_size=2, final_size=4, stage2_budget=2, generation=1, seed=7,
                work_dir=Path(td), audit_all=True,
            )
        self.assertEqual(len(evaluator.calls), 2)
        self.assertIn("c2", result.counterexamples)
        self.assertIn("c2", manager.hard_cases)
        self.assertIn("c2", archive.hard_cases)
        unsafe = programs[0].child_semantic_id
        safe = programs[1].child_semantic_id
        self.assertFalse(any(x["child_semantic_id"] == unsafe for x in archive.production_niches.values()))
        self.assertTrue(any(x["child_semantic_id"] == safe for x in archive.production_niches.values()))
        self.assertIsNotNone(result.stage_recall)
        self.assertEqual(result.stage_recall["danger_recall"], 0.0)
        self.assertEqual(result.stage_recall["selection_danger_recall"], 1.0)
        self.assertEqual(set(result.full_source_pairs), {p.program_id for p in programs})
        self.assertEqual(archive.state_evaluation(safe, result.full_root_pairs[programs[1].program_id].panel_id).margins, (101,1,101,201))
        self.assertTrue(archive.repair_niches)
        self.assertEqual(result.signals.production_insertions, 1)
        self.assertEqual(result.signals.repair_insertions, 1)
        self.assertEqual(result.signals.archive_insertions, 2)
        self.assertTrue(archive.motifs)


    def test_periodic_full_audit_recovers_missed_safe_candidate_into_production_archive(self):
        from tools.bkg4_search.edit_search.search import SearchEngine
        root = ModuleGenome(144, ())
        archive = SearchArchive.new(root_genome=root, base_identity="base", route_ids=ROUTES, run_id="r", master_seed=1)
        programs = [
            EditProgram.build(root, (EditAction("CHANGE_WINDOW",0,prefix_step=145),), ROUTES, emitter="systematic", seed=1),
            EditProgram.build(root, (EditAction("CHANGE_WINDOW",0,prefix_step=146),), ROUTES, emitter="contrastive", seed=2),
        ]
        for p in programs: archive.add_program(p, p.reconstruct(root, ROUTES))
        bank = CaseBank([
            CaseRecord("c0","/tmp/t0",0,1,"f0",baseline_margin=100),
            CaseRecord("c1","/tmp/t1",1,2,"f1",baseline_margin=-10),
            CaseRecord("c2","/tmp/t2",0,3,"f2",baseline_margin=100),
            CaseRecord("c3","/tmp/t3",1,4,"f3",baseline_margin=200),
        ])
        manager = CasePanelManager(bank, random_anchor_fraction=0.0)
        evaluator = ToyExactEvaluator(root.semantic_id())
        engine = SearchEngine(route_ids=ROUTES)
        with tempfile.TemporaryDirectory() as td:
            result = engine.evaluate_round(
                archive, programs, evaluator=evaluator, panel_manager=manager,
                screen_size=2, final_size=4, stage2_budget=1, generation=1, seed=7,
                work_dir=Path(td), audit_all=True,
            )
        self.assertEqual(len(result.finalist_program_ids),1)
        self.assertIn(programs[1].program_id, result.full_root_pairs)
        self.assertTrue(any(x["child_semantic_id"] == programs[1].child_semantic_id for x in archive.production_niches.values()))


    def test_round_expands_new_certified_positive_into_exact_certified_neighborhood(self):
        from tools.bkg4_search.edit_search.search import SearchEngine
        root = ModuleGenome(0, ())
        archive = SearchArchive.new(root_genome=root, base_identity="base", route_ids=(64,70,22), run_id="r", master_seed=1)
        source = EditProgram.build(root, (EditAction("ADD_SPECIALIST",0,feature="OWN_MONEY",cmp="GE",value=1,route=70),), (64,70,22), emitter="contrastive", seed=1)
        archive.add_program(source, source.reconstruct(root, (64,70,22)))
        search = CaseBank([
            CaseRecord("s0","/tmp/s0",0,1,"loss",protected=False,baseline_margin=-100),
            CaseRecord("s1","/tmp/s1",1,2,"fragile",protected=True,baseline_margin=1000),
            CaseRecord("s2","/tmp/s2",0,3,"strong",protected=True,baseline_margin=9000),
            CaseRecord("s3","/tmp/s3",1,4,"strong",protected=True,baseline_margin=10000),
        ])
        cert = CaseBank([
            CaseRecord("c0","/tmp/c0",0,11,"loss",protected=False,baseline_margin=-100),
            CaseRecord("c1","/tmp/c1",1,12,"fragile",protected=True,baseline_margin=1000),
            CaseRecord("c2","/tmp/c2",0,13,"strong",protected=True,baseline_margin=9000),
            CaseRecord("c3","/tmp/c3",1,14,"strong",protected=True,baseline_margin=10000),
        ])
        class NeighborhoodOracle:
            def evaluate(self, genomes, panel, *, work_dir, panel_id):
                ids=tuple(x.case_id for x in panel); out={}
                baseline={x.case_id:x.baseline_margin for x in panel}
                for genome in genomes:
                    route=genome.rules[0].route if genome.rules else 0
                    margins=[]; actions=[]; activations=[]; traces=[]
                    for i,cid in enumerate(ids):
                        b=baseline[cid]
                        if not genome.rules: m=b
                        elif route==64: m=(600 if b < 0 else b)
                        else: m=b+10
                        margins.append(m); actions.append(0 if not genome.rules else 1); activations.append(0 if not genome.rules else 2); traces.append((100 if not genome.rules else 1000)+i+route)
                    out[genome.semantic_id()]=StateEvaluation(genome.semantic_id(),panel_id,ids,tuple(margins),tuple(actions),tuple(activations),tuple(traces))
                return out
        manager=CasePanelManager(search,random_anchor_fraction=0.0)
        engine=SearchEngine(route_ids=(64,70,22))
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            result=engine.evaluate_round(
                archive,[source],evaluator=NeighborhoodOracle(),panel_manager=manager,
                screen_size=2,final_size=4,stage2_budget=1,generation=1,seed=7,
                work_dir=Path(td),audit_all=True,certification_evaluator=NeighborhoodOracle(),
                certification_bank=cert,certified_neighborhood_budget=24,
            )
        elite=[pid for pid,v in result.certification_vectors.items() if v.certified_elite]
        self.assertTrue(elite)
        self.assertTrue(any(archive.programs[pid].emitter=="certified_neighborhood" for pid in elite))
        self.assertTrue(all(pid in result.full_root_pairs and pid in result.full_source_pairs for pid in elite))
        self.assertEqual(result.stage_recall["useful_recall"],1.0)
        self.assertEqual(result.stage_recall["danger_recall"],1.0)

if __name__ == "__main__": unittest.main(verbosity=2)
