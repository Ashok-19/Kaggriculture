from __future__ import annotations

import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path


class EditSearchCliTests(unittest.TestCase):
    def parent_dir(self, root: Path, identity: str = "parent-a") -> Path:
        p = root / identity; p.mkdir()
        for name in ("controller.bkgc", "routes.bkv6", "legacy_routes.bkv6"):
            (p / name).write_bytes(b"asset")
        (p / "controller.json").write_text(json.dumps({"schema_version":4,"tree":{"type":"Success"}}))
        (p / "identity.json").write_text(json.dumps({
            "combined_agent_identity": identity,
            "controller_semantic_hash": identity + "-ctrl",
            "bkg4_abi": 4,
            "feature_registry": 1,
            "v6_route_pack": {"route_ids": [22,70]},
            "runtime_reconciler": {"defer_slots": list(range(5,18)), "route_override_slot": 19},
        }))
        return p

    def test_init_resume_validate_and_propose_lifecycle(self):
        from tools.bkg4_search.edit_search.cli import main
        from tools.bkg4_search.edit_search.archive import SearchArchive
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);parent=self.parent_dir(root);archive=root/"search.json";obs=root/"obs.jsonl";out=root/"proposals.jsonl"
            obs.write_text("\n".join(json.dumps({"feature":"OWN_MONEY","index":0,"value":x}) for x in (500,1000,2500,5000))+"\n")
            self.assertEqual(main(["init","--parent-dir",str(parent),"--archive",str(archive),"--run-id","r1","--seed","41","--root-prefix","37"]),0)
            a=SearchArchive.load(archive);self.assertEqual(a.genome(a.root_semantic_id).prefix_step,37);self.assertEqual(a.base_identity,"parent-a")
            self.assertEqual(main(["validate","--archive",str(archive)]),0)
            self.assertEqual(main(["resume","--archive",str(archive)]),0)
            self.assertEqual(main(["propose","--archive",str(archive),"--observations",str(obs),"--count","24","--generation","1","--seed","9","--out",str(out)]),0)
            rows=[json.loads(x) for x in out.read_text().splitlines() if x.strip()]
            self.assertGreater(len(rows),8)
            self.assertTrue(all(x["source_parent_semantic_id"]==a.root_semantic_id for x in rows))
            SearchArchive.load(archive)

    def test_benchmark_and_anti_saturation_commands_are_first_class_cli_surfaces(self):
        from tools.bkg4_search.edit_search.cli import cmd_anti_saturation, cmd_benchmark, parser
        common = ["--parent-dir","p","--cases-tsv","c.tsv","--observations","o.jsonl","--evaluator","eval","--work-dir","w","--seed","7"]
        b = parser().parse_args(["benchmark", *common, "--candidate-budget", "32", "--require-material-improvement"])
        self.assertIs(b.fn, cmd_benchmark); self.assertEqual(b.candidate_budget, 32); self.assertTrue(b.require_material_improvement)
        a = parser().parse_args(["anti-saturation", *common, "--generations", "6", "--require-escalation"])
        self.assertIs(a.fn, cmd_anti_saturation); self.assertEqual(a.generations, 6); self.assertTrue(a.require_escalation)

    def test_parent_change_is_rejected_for_concrete_archive_but_motifs_remain_portable_data(self):
        from tools.bkg4_search.edit_search.cli import assert_archive_parent
        from tools.bkg4_search.edit_search.archive import SearchArchive
        from tools.bkg4_search.edit_search.schema import ParentProfile
        from tools.bkg4_search.module_codec import ModuleGenome
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);a_dir=self.parent_dir(root,"a");b_dir=self.parent_dir(root,"b")
            archive=SearchArchive.new(root_genome=ModuleGenome(0,()),base_identity="a",route_ids=(22,70),run_id="r",master_seed=1)
            archive.motifs["motif:x"]={"template":{"atoms":[]},"safe_useful_hits":1}
            assert_archive_parent(archive,ParentProfile.from_directory(a_dir))
            with self.assertRaisesRegex(ValueError,"concrete search archive belongs to parent"):
                assert_archive_parent(archive,ParentProfile.from_directory(b_dir))
            self.assertIn("motif:x",archive.motifs)


    def test_search_context_enforces_mixed_production_bank_and_builds_failure_goals(self):
        import tools.bkg4_search.edit_search.cli as cli
        from tools.bkg4_search.edit_search.panels import CaseBank, CaseRecord
        rows = [
            CaseRecord("loss","/tmp/l",0,1,"fam-a",baseline_margin=-100),
            CaseRecord("fragile","/tmp/f",1,2,"fam-a",baseline_margin=1000),
            CaseRecord("strong0","/tmp/s0",0,3,"fam-b",baseline_margin=9000),
            CaseRecord("strong1","/tmp/s1",1,4,"fam-b",baseline_margin=12000),
        ]
        bank = CaseBank(rows)
        self.assertTrue(hasattr(cli,"_build_production_search_context"))
        if not hasattr(cli,"_build_production_search_context"): return
        manager, contexts, goals = cli._build_production_search_context(
            bank, fragile_margin=5000, random_anchor_fraction=0.1,
            useful_recall_floor=0.9, danger_recall_floor=0.98, counterexample_ids=(),
        )
        self.assertTrue(manager.production_mode)
        self.assertTrue(goals)
        with self.assertRaises(ValueError):
            cli._build_production_search_context(
                CaseBank([CaseRecord("only-loss","/tmp/l",0,1,"fam",baseline_margin=-10)]),
                fragile_margin=5000, random_anchor_fraction=0.1,
                useful_recall_floor=0.9, danger_recall_floor=0.98, counterexample_ids=(),
            )

    def test_search_parser_has_periodic_recall_audit_governance(self):
        from tools.bkg4_search.edit_search.cli import parser
        args = parser().parse_args([
            "search","--archive","a","--parent-dir","p","--case-bank","c","--observations","o",
            "--evaluator","e","--corpus","x","--work-dir","w","--start-generation","1","--seed","7",
        ])
        self.assertGreaterEqual(args.recall_audit_every, 1)

    def test_search_resume_accepts_valid_initialized_empty_corpus_without_weakening_export_validation(self):
        import tools.bkg4_search.edit_search.cli as cli
        from tools.bkg4_search.edit_search.archive import SearchArchive
        from tools.bkg4_search.edit_search.corpus import SearchCorpusStore, validate_export
        from tools.bkg4_search.edit_search.panels import CaseBank, CaseRecord
        from tools.bkg4_search.edit_search.schema import ParentProfile
        from tools.bkg4_search.module_codec import ModuleGenome
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); parent=self.parent_dir(root,"empty-parent")
            profile=ParentProfile.from_directory(parent)
            bank=CaseBank([CaseRecord(
                "c0","/tmp/t",0,1,"f",baseline_margin=-1,
                model_context_json='{"context_schema":"test","seat":0}',
                context_probe_steps=(72,),public_context_width=1,
                public_context_registry_width=1,public_context_values=(11,),
            )])
            archive=SearchArchive.new(root_genome=ModuleGenome(0,()),base_identity=profile.combined_agent_identity,route_ids=profile.route_ids,run_id="r",master_seed=1)
            corpus=root/"corpus"; SearchCorpusStore.create(corpus,profile=profile,case_bank=bank,archive=archive)
            self.assertEqual(cli._validate_search_corpus_for_resume(corpus)["generations"],0)
            with self.assertRaises(ValueError): validate_export(corpus)

    def test_generation_commit_recovery_preserves_search_engine_runtime_cache(self):
        import tools.bkg4_search.edit_search.cli as cli
        from tools.bkg4_search.edit_search.archive import SearchArchive
        from tools.bkg4_search.edit_search.search import SearchEngine
        from tools.bkg4_search.module_codec import ModuleGenome
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/"search.json"
            archive=SearchArchive.new(root_genome=ModuleGenome(0,()),base_identity="b",route_ids=(22,70),run_id="r",master_seed=1)
            archive.save(path)
            engine=SearchEngine(route_ids=(22,70))
            engine._contrastive_condition_cache[("bank","goal")]=(("sentinel",),)
            class Corpus:
                def recover_archive(self, archive_path):
                    self.path=Path(archive_path)
                    return {"committed_generations":[]}
            corpus=Corpus()
            got_archive, got_engine = cli._recover_search_after_generation(corpus, path, engine)
            self.assertIs(got_engine, engine)
            self.assertEqual(got_engine._contrastive_condition_cache[("bank","goal")],(("sentinel",),))
            self.assertEqual(got_archive.root_semantic_id, archive.root_semantic_id)

if __name__=="__main__":unittest.main(verbosity=2)
