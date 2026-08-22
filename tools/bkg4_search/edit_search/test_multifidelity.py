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
        self.assertEqual(set(result.full_source_pairs), {p.program_id for p in programs})
        self.assertEqual(archive.state_evaluation(safe, result.full_root_pairs[programs[1].program_id].panel_id).margins, (101,1,101,201))
        self.assertTrue(archive.motifs)


if __name__ == "__main__": unittest.main(verbosity=2)
