from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from tools.bkg4_search.bonsai_editformer.actor_grammar import EditAction
from tools.bkg4_search.module_codec import ModuleGenome
from tools.bkg4_search.edit_search.archive import SearchArchive
from tools.bkg4_search.edit_search.fitness import PairEvaluation, StateEvaluation
from tools.bkg4_search.edit_search.panels import CaseBank, CaseRecord
from tools.bkg4_search.edit_search.schema import EditProgram, ParentProfile

ROUTES=(22,70)


def ev(sid, margins, traces):
    return StateEvaluation(sid,"full",("c0","c1"),tuple(margins),(0,3),(0,4),tuple(traces))


class EditSearchCorpusTests(unittest.TestCase):
    def fixture(self, root: Path):
        for name in ("controller.bkgc","routes.bkv6","legacy_routes.bkv6"):(root/name).write_bytes(b"asset")
        (root/"controller.json").write_text(json.dumps({"schema_version":4,"tree":{"type":"Success"}}))
        profile=ParentProfile(root,"base-id","ctrl",root/"controller.bkgc",root/"routes.bkv6",root/"legacy_routes.bkv6",ROUTES,4,1,5,13,19)
        bank=CaseBank([
            CaseRecord("c0","/tmp/t0",0,1,"f0",baseline_margin=-10,model_context_json='{"context_schema":"x","seat":0}',context_probe_steps=(72,),public_context_width=1,public_context_registry_width=1,public_context_values=(11,)),
            CaseRecord("c1","/tmp/t1",1,2,"f1",baseline_margin=100,model_context_json='{"context_schema":"x","seat":1}',context_probe_steps=(72,),public_context_width=1,public_context_registry_width=1,public_context_values=(22,)),
        ])
        base=ModuleGenome(144,())
        archive=SearchArchive.new(root_genome=base,base_identity="base-id",route_ids=ROUTES,run_id="run-1",master_seed=9)
        program=EditProgram.build(base,(EditAction("ADD_SPECIALIST",0,feature="OWN_MONEY",cmp="GE",value=1000,route=70),),ROUTES,emitter="systematic",seed=3,generation_depth=1,run_id="run-1")
        child=program.reconstruct(base,ROUTES);archive.add_program(program,child)
        pair=PairEvaluation.from_states(ev(base.semantic_id(),[-10,100],[1,2]),ev(child.semantic_id(),[5,120],[3,4]),protected={"c0":True,"c1":True})
        return profile,bank,archive,program,pair

    def test_export_is_reconstructable_leakage_safe_and_contains_split_groups(self):
        from tools.bkg4_search.edit_search.corpus import SearchCorpusStore, validate_export
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);profile,bank,archive,program,pair=self.fixture(root)
            out=root/"corpus";store=SearchCorpusStore.create(out,profile=profile,case_bank=bank,archive=archive)
            store.append_exact(generation=1,archive=archive,program_pairs={program.program_id:pair})
            report=validate_export(out)
            self.assertEqual(report["programs"],1);self.assertEqual(report["labels"],2);self.assertEqual(report["reconstruction_failures"],0)
            prows=pq.read_table(out/"shards/programs_g000001.parquet").to_pylist()
            self.assertEqual(prows[0]["sample_category"],"discovery_only_positive")
            self.assertEqual(prows[0]["certification_class"],"uncertified")
            self.assertTrue(prows[0]["discovery_quality"].startswith("discovery_"))
            for key in ("exact_program_group","parent_family_group","child_semantic_group","action_skeleton_group","operation_target_group","lineage_group","run_temporal_group","target_holdout_group","temporal_regime_group","search_generation_group","certification_panel_group"):
                self.assertTrue(prows[0][key])
            labels=pq.read_table(out/"shards/labels_g000001.parquet").column_names
            for key in ("parent_action_changes_from_base","candidate_action_changes_from_base","parent_activations","candidate_activations"):
                self.assertIn(key,labels)
            cases=pq.read_table(out/"cases.parquet").column_names
            self.assertIn("prov_seed",cases);self.assertNotIn("seed",cases)
            manifest=json.loads((out/"corpus_manifest.json").read_text())
            self.assertNotIn("prov_seed",manifest["model_feature_columns"])
            self.assertFalse(any(x.startswith("search_") for x in manifest["model_feature_columns"]))

    def test_v4_export_preserves_source_root_and_certification_semantics(self):
        from tools.bkg4_search.edit_search.certification import certify_program
        from tools.bkg4_search.edit_search.corpus import SearchCorpusStore, validate_export
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);profile,bank,archive,program,source_pair=self.fixture(root);out=root/"corpus"
            root_pair=source_pair
            cert_bank=CaseBank([
                CaseRecord("k0","/tmp/k0",0,11,"cf0",baseline_margin=-10,model_context_json='{"context_schema":"x","seat":0}',context_probe_steps=(72,),public_context_width=1,public_context_registry_width=1,public_context_values=(31,)),
                CaseRecord("k1","/tmp/k1",1,12,"cf1",baseline_margin=100,model_context_json='{"context_schema":"x","seat":1}',context_probe_steps=(72,),public_context_width=1,public_context_registry_width=1,public_context_values=(32,)),
            ])
            cert_pair=PairEvaluation.from_states(
                StateEvaluation(archive.root_semantic_id,"cert",("k0","k1"),(-10,100),(0,0),(0,0),(1,2)),
                StateEvaluation(program.child_semantic_id,"cert",("k0","k1"),(5,120),(2,2),(2,2),(3,4)),
                protected={"k0":True,"k1":True},
            )
            archive.bind_case_bank(bank.identity(),tuple(bank.by_id)); archive.register_external_case_bank("certification",cert_bank.identity(),tuple(cert_bank.by_id))
            vector=certify_program(root_pair,{"cert":cert_pair}); archive.record_certification(program.program_id,vector,certification_bank_identity=cert_bank.identity())
            store=SearchCorpusStore.create(out,profile=profile,case_bank=bank,archive=archive)
            store.append_exact(generation=1,archive=archive,program_pairs={program.program_id:source_pair},root_pairs={program.program_id:root_pair})
            row=pq.read_table(out/"shards/programs_g000001.parquet").to_pylist()[0]
            self.assertEqual(row["certification_class"],"certified_elite_flip")
            self.assertEqual(row["sample_category"],"certified_elite")
            roles=set(pq.read_table(out/"shards/labels_g000001.parquet").column("label_role").to_pylist())
            self.assertEqual(roles,{"source_discovery","root_discovery"})
            report=validate_export(out)
            self.assertEqual(report["certified_elite"],1)

    def test_high_information_sampler_caps_controlled_noops(self):
        from tools.bkg4_search.edit_search.corpus import select_high_information_program_ids
        rows={f"p{i}":{"sample_category":"controlled_noop" if i<20 else "hard_negative"} for i in range(40)}
        kept=select_high_information_program_ids(rows,max_noop_fraction=0.10)
        noop=sum(rows[x]["sample_category"]=="controlled_noop" for x in kept)
        self.assertLessEqual(noop/max(1,len(kept)),0.10)
        self.assertEqual(sum(rows[x]["sample_category"]=="hard_negative" for x in kept),20)

    def test_high_information_sampler_caps_easy_hard_negatives_when_premium_evidence_exists(self):
        from tools.bkg4_search.edit_search.corpus import select_high_information_program_ids
        rows={}
        for i in range(10): rows[f"elite{i}"]={"sample_category":"certified_elite"}
        for i in range(5): rows[f"boundary{i}"]={"sample_category":"generalization_boundary"}
        for i in range(5): rows[f"repair{i}"]={"sample_category":"near_feasible_repair"}
        for i in range(100): rows[f"hard{i:03d}"]={"sample_category":"hard_negative","new_loss_count":1+i,"w2l_count":0,"severe_regression_count":0,"mean_margin_delta":-i}
        for i in range(20): rows[f"struct{i}"]={"sample_category":"structural_diversity"}
        for i in range(20): rows[f"noop{i}"]={"sample_category":"controlled_noop"}
        kept=select_high_information_program_ids(rows,max_noop_fraction=0.10)
        categories=[rows[x]["sample_category"] for x in kept]
        self.assertTrue(all(f"elite{i}" in kept for i in range(10)))
        self.assertTrue(all(f"boundary{i}" in kept for i in range(5)))
        self.assertTrue(all(f"repair{i}" in kept for i in range(5)))
        self.assertLessEqual(categories.count("hard_negative")/len(kept),0.30)
        self.assertLessEqual(categories.count("structural_diversity")/len(kept),0.15)
        self.assertLessEqual(categories.count("controlled_noop")/len(kept),0.10)
        self.assertIn("hard000",kept)
        self.assertNotIn("hard099",kept)

    def test_quality_category_promotes_boundaries_and_certified_neighbors_to_informative_classes(self):
        from tools.bkg4_search.edit_search.certification import CertificationClass,DiscoveryQuality
        from tools.bkg4_search.edit_search.corpus import _quality_category
        aggregate={"exact_noop":False,"new_loss_count":1,"w2l_count":0,"severe_regression_count":0,"mean_margin_delta":-10.0,"behavior_changing":True}
        self.assertEqual(_quality_category(discovery=DiscoveryQuality.DISCOVERY_BOUNDARY,certification=CertificationClass.UNCERTIFIED,root_aggregate=aggregate,emitter="systematic"),"near_feasible_repair")
        self.assertEqual(_quality_category(discovery=DiscoveryQuality.DISCOVERY_HARD_NEGATIVE,certification=CertificationClass.UNCERTIFIED,root_aggregate=aggregate,emitter="certified_neighborhood"),"local_contrastive_neighbor")

    def test_create_rejects_archive_from_different_parent_identity(self):
        from tools.bkg4_search.edit_search.corpus import SearchCorpusStore
        from tools.bkg4_search.edit_search.archive import SearchArchive
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);profile,bank,_,_,_=self.fixture(root)
            wrong=SearchArchive.new(root_genome=ModuleGenome(144,()),base_identity="different-base",route_ids=ROUTES,run_id="wrong",master_seed=1)
            with self.assertRaisesRegex(ValueError,"archive/base parent identity mismatch"):
                SearchCorpusStore.create(root/"wrong-corpus",profile=profile,case_bank=bank,archive=wrong)

    def test_validator_fails_closed_on_label_tamper(self):
        from tools.bkg4_search.edit_search.corpus import SearchCorpusStore, validate_export
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);profile,bank,archive,program,pair=self.fixture(root);out=root/"corpus"
            store=SearchCorpusStore.create(out,profile=profile,case_bank=bank,archive=archive);store.append_exact(generation=1,archive=archive,program_pairs={program.program_id:pair})
            p=out/"shards/labels_g000001.parquet";table=pq.read_table(p);rows=table.to_pylist();rows[0]["margin_delta"]+=1
            pq.write_table(pa.Table.from_pylist(rows,schema=table.schema),p)
            with self.assertRaisesRegex(ValueError,"label recomputation"):
                validate_export(out)


    def test_uint64_trace_hashes_and_seed_round_trip_bit_exactly(self):
        from tools.bkg4_search.edit_search.corpus import SearchCorpusStore, validate_export
        from tools.bkg4_search.edit_search.fitness import PairEvaluation
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);profile,bank,archive,program,_=self.fixture(root);out=root/"corpus"
            source=archive.genome(program.source_parent_semantic_id);child=archive.genome(program.child_semantic_id)
            max_u64=(1<<64)-1; hi=(1<<63)+12345
            pair=PairEvaluation.from_states(ev(source.semantic_id(),[-10,100],[max_u64,hi]),ev(child.semantic_id(),[5,120],[hi,max_u64]),protected={"c0":True,"c1":True})
            store=SearchCorpusStore.create(out,profile=profile,case_bank=bank,archive=archive);store.append_exact(generation=1,archive=archive,program_pairs={program.program_id:pair})
            table=pq.read_table(out/"shards/labels_g000001.parquet")
            self.assertEqual(table.schema.field("parent_action_trace_hash").type,pa.uint64())
            self.assertEqual(table.schema.field("candidate_action_trace_hash").type,pa.uint64())
            self.assertEqual(table.column("parent_action_trace_hash").to_pylist(),[max_u64,hi])
            self.assertEqual(table.column("candidate_action_trace_hash").to_pylist(),[hi,max_u64])
            self.assertEqual(pq.read_table(out/"cases.parquet").schema.field("prov_seed").type,pa.uint64())
            self.assertEqual(validate_export(out)["labels"],2)

    def test_empty_and_orphan_corpus_layouts_fail_closed(self):
        from tools.bkg4_search.edit_search.corpus import SearchCorpusStore, validate_export
        for orphan_name in (None,"programs_g000001.parquet","labels_g000001.parquet"):
            with self.subTest(orphan=orphan_name), tempfile.TemporaryDirectory() as td:
                root=Path(td);profile,bank,archive,_,_=self.fixture(root);out=root/"corpus"
                SearchCorpusStore.create(out,profile=profile,case_bank=bank,archive=archive)
                if orphan_name:
                    pq.write_table(pa.table({"x":[1]}),out/"shards"/orphan_name)
                with self.assertRaises(ValueError):
                    validate_export(out)

    def test_manifest_referencing_missing_shard_fails_closed(self):
        from tools.bkg4_search.edit_search.corpus import SearchCorpusStore, validate_export
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);profile,bank,archive,program,pair=self.fixture(root);out=root/"corpus"
            store=SearchCorpusStore.create(out,profile=profile,case_bank=bank,archive=archive);store.append_exact(generation=1,archive=archive,program_pairs={program.program_id:pair})
            (out/"shards/labels_g000001.parquet").unlink()
            with self.assertRaises(ValueError): validate_export(out)

    def test_failed_generation_transaction_rolls_back_and_retry_is_exactly_once(self):
        from unittest.mock import patch
        import tools.bkg4_search.edit_search.corpus as corpus_mod
        from tools.bkg4_search.bonsai_editformer.actor_grammar import EditAction
        from tools.bkg4_search.edit_search.corpus import SearchCorpusStore, validate_export
        from tools.bkg4_search.edit_search.fitness import PairEvaluation
        from tools.bkg4_search.edit_search.schema import EditProgram
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);profile,bank,archive,p1,pair1=self.fixture(root);out=root/"corpus"
            store=SearchCorpusStore.create(out,profile=profile,case_bank=bank,archive=archive);store.append_exact(generation=1,archive=archive,program_pairs={p1.program_id:pair1})
            base=archive.genome(archive.root_semantic_id);p2=EditProgram.build(base,(EditAction("CHANGE_WINDOW",0,prefix_step=333),),ROUTES,emitter="systematic",seed=4,generation_depth=2,run_id="run-1")
            child=p2.reconstruct(base,ROUTES);archive.add_program(p2,child)
            pair2=PairEvaluation.from_states(ev(base.semantic_id(),[-10,100],[5,6]),ev(child.semantic_id(),[-8,110],[7,8]),protected={"c0":True,"c1":True})
            original=corpus_mod._write_rows
            def fail_labels(path,rows,*args,**kwargs):
                if "g000002" in str(path) and "label" in Path(path).name:
                    raise RuntimeError("injected label write failure")
                return original(path,rows,*args,**kwargs)
            with patch.object(corpus_mod,"_write_rows",side_effect=fail_labels):
                with self.assertRaisesRegex(RuntimeError,"injected"):
                    store.append_exact(generation=2,archive=archive,program_pairs={p2.program_id:pair2})
            self.assertFalse((out/"shards/programs_g000002.parquet").exists())
            self.assertFalse((out/"shards/labels_g000002.parquet").exists())
            self.assertEqual(validate_export(out)["programs"],1)
            store.append_exact(generation=2,archive=archive,program_pairs={p2.program_id:pair2})
            self.assertEqual(validate_export(out)["programs"],2)
            with self.assertRaises(ValueError):
                store.append_exact(generation=2,archive=archive,program_pairs={p2.program_id:pair2})

    def test_validator_recomputes_all_persisted_aggregates_and_category(self):
        from tools.bkg4_search.edit_search.corpus import SearchCorpusStore, validate_export
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);profile,bank,archive,program,pair=self.fixture(root);out=root/"corpus"
            store=SearchCorpusStore.create(out,profile=profile,case_bank=bank,archive=archive);store.append_exact(generation=1,archive=archive,program_pairs={program.program_id:pair})
            p=out/"shards/programs_g000001.parquet";rows=pq.read_table(p).to_pylist()
            for field,bad in (("l2w_count",999),("strict_safe",False),("sample_category","hard_negative"),("mean_margin_delta",123456.0)):
                with self.subTest(field=field):
                    tampered=[dict(x) for x in rows];tampered[0][field]=bad;pq.write_table(pa.Table.from_pylist(tampered,schema=pq.read_table(p).schema),p)
                    with self.assertRaises(ValueError): validate_export(out)
                    pq.write_table(pa.Table.from_pylist(rows,schema=pq.read_table(p).schema),p)

    def test_model_feature_contract_is_allowlisted_and_nested_future_truth_is_rejected(self):
        from tools.bkg4_search.edit_search.corpus import SearchCorpusStore, validate_export
        bad_features=("candidate_margin","strict_safe","screening_rank","screening_mean_delta","w2l_count","l2w_count","search_emitter","prov_seed","prov_team_name")
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);profile,bank,archive,program,pair=self.fixture(root);out=root/"corpus"
            store=SearchCorpusStore.create(out,profile=profile,case_bank=bank,archive=archive);store.append_exact(generation=1,archive=archive,program_pairs={program.program_id:pair})
            manifest_path=out/"corpus_manifest.json";original=json.loads(manifest_path.read_text())
            for field in bad_features:
                with self.subTest(declared=field):
                    obj=json.loads(json.dumps(original));obj["model_feature_columns"].append(field);manifest_path.write_text(json.dumps(obj))
                    with self.assertRaises(ValueError): validate_export(out)
            manifest_path.write_text(json.dumps(original))
            cases_path=out/"cases.parquet";case_table=pq.read_table(cases_path);case_rows=case_table.to_pylist()
            for field in bad_features:
                with self.subTest(nested=field):
                    rows=[dict(x) for x in case_rows];ctx=json.loads(rows[0]["model_context_json"]);ctx["nested"]={field:1};rows[0]["model_context_json"]=json.dumps(ctx)
                    pq.write_table(pa.Table.from_pylist(rows,schema=case_table.schema),cases_path)
                    with self.assertRaises(ValueError): validate_export(out)
            pq.write_table(pa.Table.from_pylist(case_rows,schema=case_table.schema),cases_path)
            self.assertEqual(validate_export(out)["programs"],1)

    def test_committed_generation_carries_immutable_archive_transaction_identity(self):
        """A corpus generation must be cryptographically tied to its matching post-evaluation archive state."""
        from tools.bkg4_search.edit_search.corpus import SearchCorpusStore
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);profile,bank,archive,program,pair=self.fixture(root);out=root/"corpus"
            store=SearchCorpusStore.create(out,profile=profile,case_bank=bank,archive=archive)
            archive.bind_case_bank(bank.identity(), ("c0", "c1"))
            archive.register_panel("full", ("c0", "c1"))
            archive.record_state_evaluation(ev(archive.root_semantic_id,[-10,100],[1,2]))
            archive.record_state_evaluation(ev(program.child_semantic_id,[5,120],[3,4]))
            store.append_exact(generation=1,archive=archive,program_pairs={program.program_id:pair})
            manifest=json.loads((out/"corpus_manifest.json").read_text())
            row=manifest["generations"][0]
            self.assertIn("transaction_id", row)
            self.assertIn("archive_digest", row)
            self.assertTrue(str(row["transaction_id"]))
            self.assertEqual(len(str(row["archive_digest"])),64)

    def test_committed_corpus_recovers_stale_archive_from_matching_generation_snapshot(self):
        from tools.bkg4_search.edit_search.corpus import SearchCorpusStore
        from tools.bkg4_search.edit_search.archive import SearchArchive
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);profile,bank,archive,program,pair=self.fixture(root);out=root/"corpus";archive_path=root/"search.json"
            archive.bind_case_bank(bank.identity(), ("c0", "c1")); archive.register_panel("full", ("c0", "c1"))
            archive.record_state_evaluation(ev(archive.root_semantic_id,[-10,100],[1,2]))
            archive.record_state_evaluation(ev(program.child_semantic_id,[5,120],[3,4]))
            stale=SearchArchive.new(root_genome=archive.genome(archive.root_semantic_id),base_identity=archive.base_identity,route_ids=ROUTES,run_id=archive.run_id,master_seed=archive.master_seed)
            stale.bind_case_bank(bank.identity(), ("c0", "c1")); stale.save(archive_path)
            store=SearchCorpusStore.create(out,profile=profile,case_bank=bank,archive=archive)
            store.append_exact(generation=1,archive=archive,program_pairs={program.program_id:pair})
            recover=getattr(store,"recover_archive",None)
            self.assertIsNotNone(recover,"SearchCorpusStore must provide crash recovery for corpus/archive pairs")
            result=recover(archive_path)
            got=SearchArchive.load(archive_path); manifest=json.loads((out/"corpus_manifest.json").read_text()); tx=manifest["generations"][0]["transaction_id"]
            self.assertEqual(got.runtime_state.exported_generations,[1])
            self.assertEqual(got.runtime_state.generation_transactions,{1:tx})
            self.assertIn(program.child_semantic_id,got.states)
            self.assertEqual(got.state_evaluation(program.child_semantic_id,"full").margins,(5,120))
            self.assertEqual(result["recovered_generations"],[1])

    def test_prepared_uncommitted_transaction_is_rolled_back_on_recovery(self):
        from unittest.mock import patch
        import tools.bkg4_search.edit_search.corpus as corpus_mod
        from tools.bkg4_search.edit_search.corpus import SearchCorpusStore
        from tools.bkg4_search.edit_search.archive import SearchArchive
        class SimulatedCrash(BaseException): pass
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);profile,bank,archive,program,pair=self.fixture(root);out=root/"corpus";archive_path=root/"search.json"
            archive.bind_case_bank(bank.identity(), ("c0", "c1")); archive.register_panel("full", ("c0", "c1")); archive.save(archive_path)
            store=SearchCorpusStore.create(out,profile=profile,case_bank=bank,archive=archive)
            original=store._write_manifest
            def crash_before_manifest(manifest):
                if manifest.get("generations"):
                    raise SimulatedCrash()
                return original(manifest)
            with patch.object(store,"_write_manifest",side_effect=crash_before_manifest):
                with self.assertRaises(SimulatedCrash):
                    store.append_exact(generation=1,archive=archive,program_pairs={program.program_id:pair})
            recover=getattr(store,"recover_archive",None)
            self.assertIsNotNone(recover,"SearchCorpusStore must recover prepared crash artifacts")
            result=recover(archive_path)
            manifest=json.loads((out/"corpus_manifest.json").read_text())
            self.assertEqual(manifest["generations"],[])
            self.assertFalse((out/"shards/programs_g000001.parquet").exists())
            self.assertFalse((out/"shards/labels_g000001.parquet").exists())
            self.assertEqual(SearchArchive.load(archive_path).runtime_state.exported_generations,[])
            self.assertIn(1,result["rolled_back_generations"])

if __name__=="__main__":unittest.main(verbosity=2)
