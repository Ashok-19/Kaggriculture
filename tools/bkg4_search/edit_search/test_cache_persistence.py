from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tools.bkg4_search.edit_search.archive import SearchArchive
from tools.bkg4_search.edit_search.goals import SearchGoal
from tools.bkg4_search.edit_search.panels import CaseBank, CaseRecord
from tools.bkg4_search.edit_search.search import SearchEngine
from tools.bkg4_search.module_codec import ConditionGene, ModuleGenome

ROUTES=(22,70,29)


class SearchCachePersistenceTests(unittest.TestCase):
    def _bank(self, suffix: str = "") -> CaseBank:
        return CaseBank((
            CaseRecord("loss"+suffix,"/tmp/l",0,1,"f0",baseline_margin=-1),
            CaseRecord("guard"+suffix,"/tmp/g",1,2,"f1",baseline_margin=1000),
        ))

    def _contexts(self, suffix: str = ""):
        return {
            "loss"+suffix:({"feature":"OWN_MONEY","index":0,"value":100},),
            "guard"+suffix:({"feature":"OWN_MONEY","index":0,"value":2000},),
        }

    def test_searchgoal_cache_restores_after_archive_restart(self):
        root=ModuleGenome(0,())
        archive=SearchArchive.new(root_genome=root,base_identity="b",route_ids=ROUTES,run_id="r",master_seed=1)
        bank=self._bank();goal=SearchGoal(("loss",),("guard",),(),())
        pre=[(ConditionGene("OWN_MONEY","LE",0,100),)]
        engine=SearchEngine(route_ids=ROUTES)
        with mock.patch("tools.bkg4_search.edit_search.search.synthesize_contrastive_conditions",return_value=pre) as synth:
            engine.propose_generation(archive,observations=[{"feature":"OWN_MONEY","index":0,"value":100}],total=24,generation=1,seed=7,search_goals=(goal,),case_contexts=self._contexts(),case_bank=bank)
        self.assertEqual(synth.call_count,1,"semantic separators should avoid a redundant temporal synthesis pass")
        self.assertTrue(archive.runtime_state.contrastive_goal_cache)
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/"search.json";archive.save(path);loaded=SearchArchive.load(path)
        engine2=SearchEngine.from_archive(loaded,route_ids=ROUTES)
        with mock.patch("tools.bkg4_search.edit_search.search.synthesize_contrastive_conditions",side_effect=AssertionError("warm cache should survive restart")):
            engine2.propose_generation(loaded,observations=[{"feature":"OWN_MONEY","index":0,"value":100}],total=24,generation=2,seed=8,search_goals=(goal,),case_contexts=self._contexts(),case_bank=bank)

    def test_cache_key_invalidates_on_bank_goal_grammar_or_registry_change(self):
        from tools.bkg4_search.edit_search.search import contrastive_cache_key
        goal=SearchGoal(("loss",),("guard",),(),())
        base=contrastive_cache_key("bank-a",goal.goal_id,grammar_id="g1",registry_identity="r1")
        self.assertNotEqual(base,contrastive_cache_key("bank-b",goal.goal_id,grammar_id="g1",registry_identity="r1"))
        self.assertNotEqual(base,contrastive_cache_key("bank-a",SearchGoal(("loss2",),("guard",),(),()).goal_id,grammar_id="g1",registry_identity="r1"))
        self.assertNotEqual(base,contrastive_cache_key("bank-a",goal.goal_id,grammar_id="g2",registry_identity="r1"))
        self.assertNotEqual(base,contrastive_cache_key("bank-a",goal.goal_id,grammar_id="g1",registry_identity="r2"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
