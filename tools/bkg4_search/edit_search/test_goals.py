from __future__ import annotations

import importlib.util
import unittest


class ContrastiveGoalTests(unittest.TestCase):
    def _api(self):
        spec=importlib.util.find_spec("tools.bkg4_search.edit_search.goals")
        self.assertIsNotNone(spec,"contrastive SearchGoal subsystem must exist")
        if spec is None: return None
        from tools.bkg4_search.edit_search.goals import SearchGoal, synthesize_contrastive_conditions, conjunction_stats
        return SearchGoal,synthesize_contrastive_conditions,conjunction_stats

    def test_contrastive_threshold_covers_failure_and_excludes_protected_wins(self):
        api=self._api()
        if api is None: return
        SearchGoal,synth,stats=api
        contexts={
            "loss":({"feature":"OWN_MONEY","index":0,"value":100},{"feature":"DAY","index":0,"value":2}),
            "fragile":({"feature":"OWN_MONEY","index":0,"value":1000},{"feature":"DAY","index":0,"value":2}),
            "strong":({"feature":"OWN_MONEY","index":0,"value":2000},{"feature":"DAY","index":0,"value":2}),
        }
        goal=SearchGoal(target_case_ids=("loss",),protected_guard_ids=("fragile","strong"),fragile_win_ids=("fragile",),counterexample_ids=())
        rows=synth(goal,contexts,max_conditions=2,limit=16)
        self.assertTrue(rows)
        best=rows[0]
        diag=stats(best,goal,contexts)
        self.assertEqual(diag["target_hits"],1)
        self.assertEqual(diag["guard_hits"],0)
        self.assertTrue(any(c.feature=="OWN_MONEY" and c.cmp=="LE" and 100<=c.value<1000 for c in best))

    def test_conjunction_can_exclude_counterexample_when_one_predicate_cannot(self):
        api=self._api()
        if api is None: return
        SearchGoal,synth,stats=api
        contexts={
            "t0":({"feature":"DAY","index":0,"value":2},{"feature":"OWN_MONEY","index":0,"value":100}),
            "t1":({"feature":"DAY","index":0,"value":2},{"feature":"OWN_MONEY","index":0,"value":150}),
            "guard0":({"feature":"DAY","index":0,"value":2},{"feature":"OWN_MONEY","index":0,"value":2000}),
            "guard1":({"feature":"DAY","index":0,"value":9},{"feature":"OWN_MONEY","index":0,"value":120}),
        }
        goal=SearchGoal(target_case_ids=("t0","t1"),protected_guard_ids=("guard0","guard1"),fragile_win_ids=("guard0",),counterexample_ids=("guard1",))
        rows=synth(goal,contexts,max_conditions=2,limit=32)
        perfect=[r for r in rows if stats(r,goal,contexts)["target_hits"]==2 and stats(r,goal,contexts)["guard_hits"]==0]
        self.assertTrue(perfect)
        self.assertTrue(any(len(x)==2 for x in perfect))


    def test_wide_context_contrastive_synthesis_has_explicit_feature_budget(self):
        from tools.bkg4_search.edit_search.goals import SearchGoal, synthesize_contrastive_conditions
        from tools.bkg4_search.information_state_io import registry_manifest
        from tools.bkg4_search.module_codec import FEATURES
        keys=[]
        for name,args in registry_manifest():
            if name in FEATURES and len(args)<=1:
                keys.append((name,int(args[0]) if args else 0))
            if len(keys)>=256: break
        contexts={}
        for cid,offset in (("loss",0),("guard0",100),("guard1",200)):
            contexts[cid]=tuple({"feature":name,"index":idx,"value":i+offset} for i,(name,idx) in enumerate(keys))
        goal=SearchGoal(("loss",),("guard0","guard1"),(),())
        rows=synthesize_contrastive_conditions(goal,contexts,max_conditions=2,limit=24,feature_budget=32)
        self.assertLessEqual(len(rows),24)
        self.assertTrue(rows)
        used={(c.feature,c.index) for conds in rows for c in conds}
        self.assertLessEqual(len(used),32)
    def test_conjunction_requires_predicates_to_cooccur_in_one_probe_frame(self):
        from tools.bkg4_search.edit_search.goals import SearchGoal, conjunction_stats
        from tools.bkg4_search.module_codec import ConditionGene
        contexts={
            "loss":(
                {"probe_index":0,"feature":"OWN_MONEY","index":0,"value":100},
                {"probe_index":0,"feature":"DAY","index":0,"value":0},
                {"probe_index":1,"feature":"OWN_MONEY","index":0,"value":0},
                {"probe_index":1,"feature":"DAY","index":0,"value":2},
            ),
            "guard":(
                {"probe_index":0,"feature":"OWN_MONEY","index":0,"value":100},
                {"probe_index":0,"feature":"DAY","index":0,"value":2},
            ),
        }
        goal=SearchGoal(("loss",),("guard",),(),())
        conds=(ConditionGene("OWN_MONEY","GE",0,100),ConditionGene("DAY","GE",0,2))
        diag=conjunction_stats(conds,goal,contexts)
        self.assertEqual(diag["target_hits"],0)
        self.assertEqual(diag["guard_hits"],1)

    def test_matching_probe_steps_preserves_temporal_localization(self):
        from tools.bkg4_search.edit_search.goals import matching_probe_steps
        from tools.bkg4_search.module_codec import ConditionGene
        contexts={"c":(
            {"probe_index":0,"probe_step":264,"feature":"OWN_MONEY","index":0,"value":100},
            {"probe_index":1,"probe_step":266,"feature":"OWN_MONEY","index":0,"value":500},
        )}
        cond=(ConditionGene("OWN_MONEY","LE",0,200),)
        self.assertEqual(matching_probe_steps(cond,"c",contexts),(264,))

    def test_contrastive_feature_surface_prefers_generalizable_public_regimes(self):
        from tools.bkg4_search.edit_search.goals import CONTRASTIVE_FEATURES
        self.assertIn("OWN_MONEY", CONTRASTIVE_FEATURES)
        self.assertIn("MARKET_PRICE", CONTRASTIVE_FEATURES)
        self.assertIn("SHOP_SEEN", CONTRASTIVE_FEATURES)
        self.assertNotIn("OWN_WORKER_X", CONTRASTIVE_FEATURES)
        self.assertNotIn("OWN_TILE_KIND", CONTRASTIVE_FEATURES)
    def test_guard_excluding_synthesis_returns_only_protected_safe_predicates(self):
        SearchGoal,synthesize_contrastive_conditions,conjunction_stats=self._api()
        contexts={
            "loss":({"probe_index":0,"feature":"OWN_MONEY","index":0,"value":100},),
            "guard":({"probe_index":0,"feature":"OWN_MONEY","index":0,"value":2000},),
        }
        goal=SearchGoal(("loss",),("guard",),(),())
        rows=synthesize_contrastive_conditions(
            goal,contexts,max_conditions=2,limit=16,guard_excluding_only=True,
        )
        self.assertTrue(rows)
        for conds in rows:
            diag=conjunction_stats(conds,goal,contexts)
            self.assertGreater(diag["target_hits"],0)
            self.assertEqual(diag["guard_hits"],0)



    def test_non_temporal_synthesis_can_exclude_temporal_features(self):
        from tools.bkg4_search.edit_search.goals import SearchGoal, synthesize_contrastive_conditions
        contexts={
            "loss":(
                {"probe_index":0,"probe_step":264,"feature":"STEP","index":0,"value":264},
                {"probe_index":0,"probe_step":264,"feature":"OWN_MONEY","index":0,"value":100},
            ),
            "guard":(
                {"probe_index":0,"probe_step":264,"feature":"STEP","index":0,"value":264},
                {"probe_index":0,"probe_step":264,"feature":"OWN_MONEY","index":0,"value":2000},
            ),
        }
        rows=synthesize_contrastive_conditions(
            SearchGoal(("loss",),("guard",),(),()),contexts,limit=16,
            guard_excluding_only=True,allow_temporal=False,
        )
        self.assertTrue(rows)
        self.assertTrue(all(c.feature not in {"STEP","DAY","HOUR"} for conds in rows for c in conds))

    def test_temporal_specificity_prefers_semantic_then_broad_then_exact(self):
        from tools.bkg4_search.edit_search.goals import temporal_specificity
        from tools.bkg4_search.module_codec import ConditionGene
        semantic=(ConditionGene("OWN_MONEY","LE",0,1000),)
        broad=semantic+(ConditionGene("STEP","GE",0,144),ConditionGene("STEP","LE",0,287))
        narrow=semantic+(ConditionGene("STEP","GE",0,260),ConditionGene("STEP","LE",0,280))
        exact=semantic+(ConditionGene("STEP","EQ",0,274),)
        self.assertLess(temporal_specificity(semantic),temporal_specificity(broad))
        self.assertLess(temporal_specificity(broad),temporal_specificity(narrow))
        self.assertLess(temporal_specificity(narrow),temporal_specificity(exact))

    def test_failure_clustering_does_not_split_only_on_day_or_hour(self):
        from tools.bkg4_search.edit_search.goals import _cluster_signature
        a=(
            {"feature":"DAY","index":0,"value":2},
            {"feature":"HOUR","index":0,"value":4},
            {"feature":"OWN_UNITS","index":0,"value":3},
            {"feature":"OWN_LAND","index":0,"value":2},
        )
        b=(
            {"feature":"DAY","index":0,"value":20},
            {"feature":"HOUR","index":0,"value":22},
            {"feature":"OWN_UNITS","index":0,"value":3},
            {"feature":"OWN_LAND","index":0,"value":2},
        )
        self.assertEqual(_cluster_signature(a),_cluster_signature(b))


if __name__=="__main__": unittest.main(verbosity=2)
