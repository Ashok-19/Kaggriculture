from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

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
            self.assertEqual(prows[0]["sample_category"],"strict_safe_useful")
            for key in ("exact_program_group","parent_family_group","child_semantic_group","action_skeleton_group","operation_target_group","lineage_group","run_temporal_group"):
                self.assertTrue(prows[0][key])
            labels=pq.read_table(out/"shards/labels_g000001.parquet").column_names
            for key in ("parent_action_changes_from_base","candidate_action_changes_from_base","parent_activations","candidate_activations"):
                self.assertIn(key,labels)
            cases=pq.read_table(out/"cases.parquet").column_names
            self.assertIn("prov_seed",cases);self.assertNotIn("seed",cases)
            manifest=json.loads((out/"corpus_manifest.json").read_text())
            self.assertNotIn("prov_seed",manifest["model_feature_columns"])
            self.assertFalse(any(x.startswith("search_") for x in manifest["model_feature_columns"]))

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
            p=out/"shards/labels_g000001.parquet";rows=pq.read_table(p).to_pylist();rows[0]["margin_delta"]+=1
            import pyarrow as pa
            pq.write_table(pa.Table.from_pylist(rows),p)
            with self.assertRaisesRegex(ValueError,"label recomputation"):
                validate_export(out)


if __name__=="__main__":unittest.main(verbosity=2)
