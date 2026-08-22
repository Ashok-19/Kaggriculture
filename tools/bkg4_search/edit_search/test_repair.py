from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tools.bkg4_search.bonsai_editformer.actor_grammar import EditAction
from tools.bkg4_search.edit_search.archive import SearchArchive
from tools.bkg4_search.edit_search.fitness import StateEvaluation
from tools.bkg4_search.edit_search.panels import CaseBank, CasePanelManager, CaseRecord
from tools.bkg4_search.edit_search.schema import EditProgram
from tools.bkg4_search.module_codec import ModuleGenome

ROUTES=(22,70,29)


class RepairOracle:
    def evaluate(self, genomes, cases, *, work_dir, panel_id):
        out={}
        ids=tuple(x.case_id for x in cases)
        for genome in genomes:
            guarded=any(c.feature=="OWN_MONEY" and c.cmp=="LE" and c.value<=500 for r in genome.rules for c in r.conditions)
            if not genome.rules:
                margins=(-100,1000,1200,9000)
            elif guarded:
                margins=(100,1000,1200,9000)
            else:
                margins=(100,-100,1200,9000)
            changed=bool(genome.rules)
            out[genome.semantic_id()]=StateEvaluation(
                genome.semantic_id(),panel_id,ids,margins,
                tuple(1 if changed else 0 for _ in ids),tuple(2 if changed else 0 for _ in ids),
                tuple((1000 if changed else 100)+i for i in range(len(ids))),
            )
        return out


def contexts():
    return {
        "target":({"probe_index":0,"probe_step":72,"feature":"OWN_MONEY","index":0,"value":100},{"probe_index":0,"probe_step":72,"feature":"OPP_MONEY","index":0,"value":100}),
        "ce0":({"probe_index":0,"probe_step":72,"feature":"OWN_MONEY","index":0,"value":2000},{"probe_index":0,"probe_step":72,"feature":"OPP_MONEY","index":0,"value":100}),
        "ce1":({"probe_index":0,"probe_step":72,"feature":"OWN_MONEY","index":0,"value":100},{"probe_index":0,"probe_step":72,"feature":"OPP_MONEY","index":0,"value":2000}),
        "strong":({"probe_index":0,"probe_step":72,"feature":"OWN_MONEY","index":0,"value":3000},{"probe_index":0,"probe_step":72,"feature":"OPP_MONEY","index":0,"value":3000}),
    }

class CrossContextRepairOracle:
    def evaluate(self, genomes, cases, *, work_dir, panel_id):
        ids=tuple(x.case_id for x in cases); out={}; is_cert=bool(ids) and all(x.startswith("cert_") for x in ids)
        for genome in genomes:
            has_own=any(c.feature=="OWN_MONEY" and c.cmp=="LE" and c.value<=500 for r in genome.rules for c in r.conditions)
            has_opp=any(c.feature=="OPP_MONEY" and c.cmp=="LE" and c.value<=500 for r in genome.rules for c in r.conditions)
            if is_cert:
                if not genome.rules:
                    margins=tuple(-100 if "target" in cid else 1000 for cid in ids)
                elif has_own and has_opp:
                    margins=tuple(100 if "target" in cid else 1000 for cid in ids)
                else:
                    margins=tuple(100 if "target" in cid else -100 for cid in ids)
            elif not genome.rules:
                margins=(-100,1000,9000)
            else:
                margins=(100,1000,9000) if has_own else (100,-100,9000)
            changed=bool(genome.rules)
            out[genome.semantic_id()]=StateEvaluation(
                genome.semantic_id(),panel_id,ids,margins,
                tuple(1 if changed else 0 for _ in ids),tuple(2 if changed else 0 for _ in ids),
                tuple((4000 if changed else 400)+i for i in range(len(ids))),
            )
        return out


def certification_contexts():
    return {
        "cert_target0":({"probe_index":0,"probe_step":72,"feature":"OWN_MONEY","index":0,"value":100},{"probe_index":0,"probe_step":72,"feature":"OPP_MONEY","index":0,"value":100}),
        "cert_ce0":({"probe_index":0,"probe_step":72,"feature":"OWN_MONEY","index":0,"value":100},{"probe_index":0,"probe_step":72,"feature":"OPP_MONEY","index":0,"value":2000}),
        "cert_target1":({"probe_index":0,"probe_step":72,"feature":"OWN_MONEY","index":0,"value":100},{"probe_index":0,"probe_step":72,"feature":"OPP_MONEY","index":0,"value":100}),
        "cert_ce1":({"probe_index":0,"probe_step":72,"feature":"OWN_MONEY","index":0,"value":100},{"probe_index":0,"probe_step":72,"feature":"OPP_MONEY","index":0,"value":2000}),
    }



class CEGISRepairTests(unittest.TestCase):
    def test_multi_counterexample_guard_synthesis_finds_two_condition_separator(self):
        from tools.bkg4_search.edit_search.repair import synthesize_counterexample_guards
        rows=synthesize_counterexample_guards(("target",),("ce0","ce1"),contexts(),max_conditions=2,limit=32)
        self.assertTrue(rows)
        self.assertTrue(any(len(x)==2 for x in rows))
        best=rows[0]
        self.assertEqual(len(best),2)
        self.assertEqual({c.feature for c in best},{"OWN_MONEY","OPP_MONEY"})

    def test_propose_guard_repairs_builds_descendant_without_exceeding_grammar(self):
        from tools.bkg4_search.edit_search.repair import propose_guard_repairs
        root=ModuleGenome(0,())
        first=EditProgram.build(root,(EditAction("ADD_SPECIALIST",0,feature="STEP",cmp="GE",value=0,route=22),),ROUTES,emitter="structural",seed=1)
        child=first.reconstruct(root,ROUTES)
        archive=SearchArchive.new(root_genome=root,base_identity="b",route_ids=ROUTES,run_id="r",master_seed=1);archive.add_program(first,child)
        rows=propose_guard_repairs(archive,first,target_case_ids=("target",),counterexample_ids=("ce0","ce1"),case_contexts=contexts(),n=8,seed=3)
        self.assertTrue(rows)
        self.assertTrue(all(x.parent_program_id==first.program_id for x in rows))
        for program in rows:
            repaired=program.reconstruct(child,ROUTES)
            self.assertLessEqual(max(len(r.conditions) for r in repaired.rules),4)
            self.assertGreater(len(repaired.rules[0].conditions),1)

    def test_cegis_chain_converts_unsafe_candidate_to_root_feasible(self):
        from tools.bkg4_search.edit_search.repair import run_cegis_repair
        root=ModuleGenome(0,())
        first=EditProgram.build(root,(EditAction("ADD_SPECIALIST",0,feature="STEP",cmp="GE",value=0,route=22),),ROUTES,emitter="structural",seed=1)
        child=first.reconstruct(root,ROUTES)
        archive=SearchArchive.new(root_genome=root,base_identity="b",route_ids=ROUTES,run_id="r",master_seed=1);archive.add_program(first,child)
        bank=CaseBank([
            CaseRecord("target","/tmp/t",0,1,"loss",protected=False,baseline_margin=-100),
            CaseRecord("ce0","/tmp/c0",1,2,"fragile",protected=True,baseline_margin=1000),
            CaseRecord("ce1","/tmp/c1",0,3,"fragile",protected=True,baseline_margin=1200),
            CaseRecord("strong","/tmp/s",1,4,"strong",protected=True,baseline_margin=9000),
        ])
        manager=CasePanelManager(bank)
        with tempfile.TemporaryDirectory() as td:
            report=run_cegis_repair(
                archive,first,evaluator=RepairOracle(),panel_manager=manager,case_contexts=contexts(),
                generation=1,seed=19,work_dir=Path(td),max_rounds=2,candidates_per_round=8,
            )
        self.assertGreater(report.attempts,0)
        self.assertGreater(report.less_infeasible,0)
        self.assertGreater(report.became_feasible,0)
        self.assertGreater(report.became_safe_useful,0)
        self.assertTrue(report.chain_program_ids)

    def test_cegis_repairs_rotating_certification_counterexample_after_discovery_feasibility(self):
        from tools.bkg4_search.edit_search.repair import run_cegis_repair
        root=ModuleGenome(0,())
        first=EditProgram.build(root,(EditAction("ADD_SPECIALIST",0,feature="STEP",cmp="GE",value=0,route=22),),ROUTES,emitter="contrastive",seed=1)
        child=first.reconstruct(root,ROUTES)
        archive=SearchArchive.new(root_genome=root,base_identity="b",route_ids=ROUTES,run_id="r",master_seed=1);archive.add_program(first,child)
        discovery=CaseBank([
            CaseRecord("target","/tmp/t",0,1,"loss",protected=False,baseline_margin=-100),
            CaseRecord("ce0","/tmp/c",1,2,"guard",protected=True,baseline_margin=1000),
            CaseRecord("strong","/tmp/s",0,3,"strong",protected=True,baseline_margin=9000),
        ])
        certification=CaseBank([
            CaseRecord("cert_target0","/tmp/ct0",0,11,"loss-a",protected=False,baseline_margin=-100),
            CaseRecord("cert_ce0","/tmp/cc0",1,12,"guard-a",protected=True,baseline_margin=1000),
            CaseRecord("cert_target1","/tmp/ct1",0,13,"loss-b",protected=False,baseline_margin=-100),
            CaseRecord("cert_ce1","/tmp/cc1",1,14,"guard-b",protected=True,baseline_margin=1000),
        ])
        manager=CasePanelManager(discovery)
        ctx={k:v for k,v in contexts().items() if k in {"target","ce0","strong"}}
        with tempfile.TemporaryDirectory() as td:
            report=run_cegis_repair(
                archive,first,evaluator=CrossContextRepairOracle(),panel_manager=manager,case_contexts=ctx,
                generation=1,seed=41,work_dir=Path(td),max_rounds=2,candidates_per_round=8,
                certification_evaluator=CrossContextRepairOracle(),certification_bank=certification,
                certification_case_contexts=certification_contexts(),
            )
        self.assertGreater(report.became_feasible,0)
        self.assertGreater(report.became_safe_useful,0)
        self.assertGreater(report.became_certified_useful,0)
        vector=archive.certification_vector(report.terminal_program_id)
        self.assertIsNotNone(vector)
        self.assertTrue(vector.certified_positive)
        self.assertEqual(vector.panel_count,1)
        self.assertTrue(all('/cert/p1' in p.panel_id for p in vector.panels))
        terminal=archive.genome(archive.programs[report.terminal_program_id].child_semantic_id)
        features={c.feature for r in terminal.rules for c in r.conditions}
        self.assertIn("OWN_MONEY",features)
        self.assertIn("OPP_MONEY",features)

    def test_cross_context_key_prefers_certification_useful_after_hard_feasibility(self):
        from tools.bkg4_search.edit_search.repair import _cross_context_repair_key
        from tools.bkg4_search.edit_search.safety import SafetySpec
        from tools.bkg4_search.edit_search.fitness import PairEvaluation, StateEvaluation
        def pe(parent, child):
            ids=("a","b")
            p=StateEvaluation("p","x",ids,tuple(parent),(0,0),(0,0),(1,2))
            c=StateEvaluation("c","x",ids,tuple(child),(1,1),(1,1),(3,4))
            return SafetySpec().assess(PairEvaluation.from_states(p,c))
        discovery=pe((-100,1000),(100,1000))
        cert_useful=pe((-100,1000),(100,1100))
        cert_inert=pe((-100,1000),(-100,1000))
        self.assertTrue(discovery.feasible)
        self.assertTrue(cert_useful.feasible); self.assertTrue(cert_useful.safe_useful)
        self.assertTrue(cert_inert.feasible); self.assertFalse(cert_inert.safe_useful)
        self.assertLess(
            _cross_context_repair_key(discovery,cert_useful,"useful"),
            _cross_context_repair_key(discovery,cert_inert,"inert"),
        )

    def test_cross_context_cegis_keeps_discovery_feasible_terminal_when_no_safe_cross_repair_exists(self):
        from tools.bkg4_search.edit_search.repair import _should_advance_cross_context
        from tools.bkg4_search.edit_search.safety import SafetySpec
        from tools.bkg4_search.edit_search.fitness import PairEvaluation, StateEvaluation
        def assess(parent, child, sid):
            ids=("a","b")
            p=StateEvaluation("p","x",ids,tuple(parent),(0,0),(0,0),(1,2))
            c=StateEvaluation(sid,"x",ids,tuple(child),(1,1),(1,1),(3,4))
            return SafetySpec().assess(PairEvaluation.from_states(p,c))
        current_cert=assess((-100,1000),(100,-100),"current")
        regressed_discovery=assess((-100,1000),(100,-100),"bad")
        improved_cert=assess((-100,1000),(100,1000),"good")
        self.assertFalse(regressed_discovery.feasible)
        self.assertTrue(improved_cert.feasible)
        self.assertFalse(_should_advance_cross_context(current_cert, []))
        self.assertFalse(_should_advance_cross_context(current_cert, [(regressed_discovery, improved_cert)]))


if __name__=="__main__": unittest.main(verbosity=2)
