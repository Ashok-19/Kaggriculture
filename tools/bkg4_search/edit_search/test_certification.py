from __future__ import annotations

import unittest

from tools.bkg4_search.edit_search.fitness import PairEvaluation, StateEvaluation


def ev(sid: str, panel: str, case_ids: tuple[str, ...], margins: tuple[int, ...], *, traces: tuple[int, ...], activations: tuple[int, ...]) -> StateEvaluation:
    return StateEvaluation(sid, panel, case_ids, margins, tuple(1 for _ in case_ids), activations, traces)


def pair(panel: str, parent_margins: tuple[int, ...], child_margins: tuple[int, ...], *, child_trace_offset: int = 100, child_activation: int = 1) -> PairEvaluation:
    ids = tuple(f"{panel}-c{i}" for i in range(len(parent_margins)))
    parent = ev("root", panel, ids, parent_margins, traces=tuple(10 + i for i in range(len(ids))), activations=tuple(0 for _ in ids))
    child = ev("child", panel, ids, child_margins, traces=tuple(child_trace_offset + i for i in range(len(ids))), activations=tuple(child_activation for _ in ids))
    protected = {cid: i > 0 for i, cid in enumerate(ids)}
    return PairEvaluation.from_states(parent, child, protected=protected)


class CertificationSemanticsTests(unittest.TestCase):
    def test_discovery_elite_is_not_certified_without_independent_evidence(self):
        from tools.bkg4_search.edit_search.certification import CertificationClass, DiscoveryQuality, certify_program
        discovery = pair("discovery", (-100, 1000), (10, 1200))
        vector = certify_program(discovery, {})
        self.assertEqual(vector.discovery_quality, DiscoveryQuality.DISCOVERY_ELITE_FLIP)
        self.assertEqual(vector.certification_class, CertificationClass.UNCERTIFIED)
        self.assertFalse(vector.certified_positive)

    def test_discovery_elite_that_becomes_unsafe_is_generalization_hard_negative(self):
        from tools.bkg4_search.edit_search.certification import CertificationClass, TransferFailure, certify_program
        discovery = pair("discovery", (-100, 1000), (10, 1200))
        unsafe = pair("cert-a", (-50, 1000), (-20, -1))
        vector = certify_program(discovery, {"cert-a": unsafe})
        self.assertEqual(vector.certification_class, CertificationClass.GENERALIZATION_HARD_NEGATIVE)
        self.assertEqual(vector.transfer_failure, TransferFailure.UNSAFE_TRANSFER)
        self.assertLess(vector.strict_safe_retention, 1.0)

    def test_discovery_elite_that_is_safe_but_inert_is_generalization_boundary(self):
        from tools.bkg4_search.edit_search.certification import CertificationClass, TransferFailure, certify_program
        discovery = pair("discovery", (-100, 1000), (10, 1200))
        ids = ("cert-c0", "cert-c1")
        root = ev("root", "cert", ids, (-50, 1000), traces=(1, 2), activations=(0, 0))
        inert = ev("child", "cert", ids, (-50, 1000), traces=(1, 2), activations=(0, 0))
        cert = PairEvaluation.from_states(root, inert, protected={"cert-c0": False, "cert-c1": True})
        vector = certify_program(discovery, {"cert": cert}, temporal=True)
        self.assertEqual(vector.certification_class, CertificationClass.GENERALIZATION_BOUNDARY)
        self.assertEqual(vector.transfer_failure, TransferFailure.TEMPORAL_OVERFIT)
        self.assertEqual(vector.strict_safe_retention, 1.0)
        self.assertEqual(vector.behavior_retention, 0.0)
        self.assertEqual(vector.activation_retention, 0.0)

    def test_certified_elite_requires_safe_active_useful_transfer(self):
        from tools.bkg4_search.edit_search.certification import CertificationClass, TransferFailure, certify_program
        discovery = pair("discovery", (-100, 1000), (10, 1200))
        c1 = pair("cert-a", (-100, 1000), (20, 1100), child_trace_offset=200, child_activation=3)
        c2 = pair("cert-b", (-50, 800), (5, 900), child_trace_offset=300, child_activation=2)
        vector = certify_program(discovery, {"cert-a": c1, "cert-b": c2})
        self.assertEqual(vector.certification_class, CertificationClass.CERTIFIED_ELITE_FLIP)
        self.assertEqual(vector.transfer_failure, TransferFailure.NONE)
        self.assertTrue(vector.certified_positive)
        self.assertEqual(vector.strict_safe_retention, 1.0)
        self.assertEqual(vector.activation_retention, 1.0)
        self.assertEqual(vector.behavior_retention, 1.0)
        self.assertEqual(vector.safe_useful_retention, 1.0)
        self.assertEqual(vector.l2w_retention, 1.0)
        self.assertGreaterEqual(vector.elite_retention, 0.25)


    def test_archive_persists_certification_evidence_and_role_banks_are_disjoint(self):
        import tempfile
        from pathlib import Path
        from tools.bkg4_search.bonsai_editformer.actor_grammar import EditAction
        from tools.bkg4_search.edit_search.archive import SearchArchive
        from tools.bkg4_search.edit_search.certification import CertificationVector
        from tools.bkg4_search.edit_search.certification import certify_program
        from tools.bkg4_search.edit_search.schema import EditProgram
        from tools.bkg4_search.module_codec import ModuleGenome
        routes=(22,70)
        root=ModuleGenome(0,())
        program=EditProgram.build(root,(EditAction("ADD_SPECIALIST",0,feature="OWN_MONEY",cmp="LE",value=500,route=22),),routes,emitter="contrastive",seed=1)
        child=program.reconstruct(root,routes)
        archive=SearchArchive.new(root_genome=root,base_identity="b",route_ids=routes,run_id="r",master_seed=1)
        archive.add_program(program,child)
        archive.bind_case_bank("discovery-bank",("d0","d1"))
        archive.register_external_case_bank("certification","cert-bank",("c0","c1"))
        archive.register_external_case_bank("ranker_validation","val-bank",("v0","v1"))
        with self.assertRaises(ValueError):
            archive.register_external_case_bank("certification","bad-overlap",("d0","x"))
        discovery=pair("discovery",(-100,1000),(10,1200))
        cert=pair("cert-a",(-100,1000),(20,1100),child_trace_offset=200,child_activation=2)
        vector=certify_program(discovery,{"cert-a":cert})
        archive.record_certification(program.program_id,vector,certification_bank_identity="cert-bank")
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/"search.json";archive.save(path);loaded=SearchArchive.load(path)
        got=CertificationVector.from_dict(loaded.certification_evidence[program.program_id]["vector"])
        self.assertEqual(got,vector)
        self.assertEqual(loaded.external_case_banks["certification"]["identity"],"cert-bank")

    def test_certification_panels_are_disjoint_and_cover_bank(self):
        from tools.bkg4_search.edit_search.context_bank import certification_panels
        from tools.bkg4_search.edit_search.panels import CaseBank, CaseRecord
        rows=[CaseRecord(f"z{i}",f"/tmp/z{i}",i%2,100+i,f"fam-{i%4}",baseline_margin=(-10 if i<4 else 1000 if i<8 else 9000)) for i in range(12)]
        panels=certification_panels(CaseBank(rows),parts=2)
        self.assertEqual(len(panels),2)
        a={x.case_id for x in panels[0]}; b={x.case_id for x in panels[1]}
        self.assertFalse(a & b)
        self.assertEqual(a|b,{x.case_id for x in rows})
        self.assertEqual({x.seat for x in panels[0]},{0,1})
        self.assertEqual({x.seat for x in panels[1]},{0,1})

    def test_certification_quality_orders_certified_above_discovery_only(self):
        from tools.bkg4_search.edit_search.certification import CertificationClass, certification_rank
        self.assertLess(certification_rank(CertificationClass.CERTIFIED_ELITE_FLIP),certification_rank(CertificationClass.CERTIFIED_SAFE_USEFUL))
        self.assertLess(certification_rank(CertificationClass.CERTIFIED_SAFE_USEFUL),certification_rank(CertificationClass.UNCERTIFIED))
        self.assertLess(certification_rank(CertificationClass.UNCERTIFIED),certification_rank(CertificationClass.GENERALIZATION_HARD_NEGATIVE))


    def test_certified_neighborhood_includes_deterministic_route_threshold_and_window_alternatives(self):
        from tools.bkg4_search.bonsai_editformer.actor_grammar import EditAction
        from tools.bkg4_search.edit_search.archive import SearchArchive
        from tools.bkg4_search.edit_search.certification import propose_certified_neighborhood
        from tools.bkg4_search.edit_search.schema import EditProgram
        from tools.bkg4_search.module_codec import ModuleGenome
        routes=(64,70,22)
        root=ModuleGenome(0,())
        source=EditProgram.build(root,(EditAction("ADD_SPECIALIST",0,feature="OWN_MONEY",cmp="GE",value=1000,route=70),),routes,emitter="seed",seed=1,generation_depth=1,run_id="r")
        child=source.reconstruct(root,routes)
        archive=SearchArchive.new(root_genome=root,base_identity="b",route_ids=routes,run_id="r",master_seed=1); archive.add_program(source,child)
        rows=propose_certified_neighborhood(archive,source,n=32,seed=9)
        self.assertTrue(rows)
        self.assertEqual([x.program_id for x in rows], [x.program_id for x in propose_certified_neighborhood(archive,source,n=32,seed=9)])
        ops={a.operation for p in rows for a in p.atoms}
        self.assertIn("CHANGE_ROUTE",ops); self.assertIn("CHANGE_THRESHOLD",ops); self.assertIn("CHANGE_WINDOW",ops)
        route64=[p for p in rows if any(a.operation=="CHANGE_ROUTE" and int(a.action.get("route",-1))==64 for a in p.atoms)]
        self.assertTrue(route64)
        self.assertTrue(all(p.source_parent_semantic_id==child.semantic_id() for p in rows))


    def test_exact_certified_neighborhood_promotes_route_variant_to_certified_elite(self):
        import tempfile
        from pathlib import Path
        from tools.bkg4_search.bonsai_editformer.actor_grammar import EditAction
        from tools.bkg4_search.edit_search.archive import SearchArchive
        from tools.bkg4_search.edit_search.certification import evaluate_certified_neighborhood_exact
        from tools.bkg4_search.edit_search.fitness import StateEvaluation
        from tools.bkg4_search.edit_search.panels import CaseBank, CaseRecord
        from tools.bkg4_search.edit_search.schema import EditProgram
        from tools.bkg4_search.module_codec import ModuleGenome
        routes=(64,70,22)
        root=ModuleGenome(0,())
        source=EditProgram.build(root,(EditAction("ADD_SPECIALIST",0,feature="OWN_MONEY",cmp="GE",value=1000,route=70),),routes,emitter="seed",seed=1,generation_depth=1,run_id="r")
        child=source.reconstruct(root,routes)
        archive=SearchArchive.new(root_genome=root,base_identity="b",route_ids=routes,run_id="r",master_seed=1); archive.add_program(source,child)
        search=CaseBank([
            CaseRecord("s0","/tmp/s0",0,1,"loss",protected=False,baseline_margin=-10),
            CaseRecord("s1","/tmp/s1",1,2,"guard",protected=True,baseline_margin=1000),
        ])
        cert=CaseBank([
            CaseRecord("c0","/tmp/c0",0,3,"loss",protected=False,baseline_margin=-10),
            CaseRecord("c1","/tmp/c1",1,4,"guard",protected=True,baseline_margin=1000),
        ])
        archive.bind_case_bank(search.identity(),tuple(search.by_id)); archive.register_external_case_bank("certification",cert.identity(),tuple(cert.by_id))
        class Oracle:
            def evaluate(self,genomes,panel,*,work_dir,panel_id):
                ids=tuple(x.case_id for x in panel); out={}
                for g in genomes:
                    route=(g.rules[0].route if g.rules else 0)
                    if not g.rules: margins=(-10,1000)
                    elif route==64: margins=(590,1000)
                    else: margins=(10,1000)
                    changed=bool(g.rules)
                    out[g.semantic_id()]=StateEvaluation(g.semantic_id(),panel_id,ids,margins,tuple(1 if changed else 0 for _ in ids),tuple(2 if changed else 0 for _ in ids),tuple((1000 if changed else 100)+i for i in range(len(ids))))
                return out
        with tempfile.TemporaryDirectory() as td:
            batch=evaluate_certified_neighborhood_exact(archive,source,evaluator=Oracle(),search_bank=search,certification_bank=cert,generation=2,seed=9,work_dir=Path(td),budget=32)
        elite=[pid for pid,v in batch.vectors.items() if v.certified_elite]
        self.assertTrue(elite)
        self.assertTrue(any(archive.genome(archive.programs[pid].child_semantic_id).rules[0].route==64 for pid in elite))
        self.assertTrue(all(pid in batch.root_pairs and pid in batch.source_pairs for pid in elite))
        self.assertTrue(any(x.get("child_semantic_id")==archive.programs[elite[0]].child_semantic_id for x in archive.production_niches.values()))


if __name__ == "__main__":
    unittest.main(verbosity=2)
