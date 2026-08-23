from __future__ import annotations

import unittest


class EditSearchPanelTests(unittest.TestCase):
    def cases(self):
        from tools.bkg4_search.edit_search.panels import CaseRecord, CaseBank
        rows = []
        for i in range(24):
            margin = -100 - i if i < 4 else (100 + i if i < 10 else 10000 + i)
            rows.append(CaseRecord(
                case_id=f"c{i:02d}", tape_path=f"/tmp/t{i}.b4tp", seat=i % 2, seed=100 + i,
                family=f"fam{i % 4}", protected=True, own_history=i % 3 == 0, role="league",
                baseline_margin=margin, model_context_json='{"seat":0}',
            ))
        return CaseBank(rows)

    def test_screen_panel_mixes_hard_fragile_protected_both_seats_and_families(self):
        from tools.bkg4_search.edit_search.panels import CasePanelManager
        bank = self.cases()
        manager = CasePanelManager(bank, fragile_margin=5000, random_anchor_fraction=0.15)
        manager.add_hard_case("c20", severity=20, reason="w2l")
        panel = manager.select_screen(12, seed=17)
        ids = {c.case_id for c in panel}
        self.assertIn("c20", ids)
        self.assertEqual({c.seat for c in panel}, {0, 1})
        self.assertGreaterEqual(len({c.family for c in panel}), 4)
        self.assertTrue(any(c.baseline_margin < 0 for c in panel))
        self.assertTrue(any(0 < c.baseline_margin <= 5000 for c in panel))
        self.assertTrue(any(c.baseline_margin > 5000 for c in panel))
        self.assertEqual([c.case_id for c in panel], [c.case_id for c in manager.select_screen(12, seed=17)])

    def test_broader_failures_become_future_counterexamples(self):
        from tools.bkg4_search.edit_search.panels import CasePanelManager
        bank = self.cases()
        manager = CasePanelManager(bank)
        screen = {"c00": {"parent": -10, "child": 100}, "c04": {"parent": 104, "child": 110}}
        full = dict(screen)
        full["c18"] = {"parent": 10018, "child": -1}
        diag = manager.compare_screen_to_full(screen, full)
        self.assertEqual(diag["new_counterexamples"], ["c18"])
        self.assertIn("c18", manager.hard_cases)
        panel = manager.select_screen(8, seed=5)
        self.assertIn("c18", {c.case_id for c in panel})

    def test_recall_diagnostics_trigger_refresh_when_useful_recall_drops(self):
        from tools.bkg4_search.edit_search.panels import CasePanelManager
        manager = CasePanelManager(self.cases(), useful_recall_floor=0.8, danger_recall_floor=0.95)
        diag = manager.stage_recall(
            screened_program_ids={"a", "b"},
            full_useful_ids={"a", "c"},
            full_l2w_ids={"a", "c"},
            full_dangerous_ids={"b", "d"},
        )
        self.assertAlmostEqual(diag["useful_recall"], 0.5)
        self.assertAlmostEqual(diag["danger_recall"], 0.5)
        self.assertTrue(manager.refresh_required(diag))


    def test_production_mode_rejects_all_loss_bank_and_reserves_protected_wins(self):
        from tools.bkg4_search.edit_search.panels import CaseBank, CasePanelManager, CaseRecord
        losses=[CaseRecord(f"l{i}",f"/tmp/l{i}",i%2,100+i,f"f{i%3}",protected=True,baseline_margin=-100-i) for i in range(8)]
        with self.assertRaises(ValueError):
            CasePanelManager(CaseBank(losses),production_mode=True)
        wins=[
            CaseRecord("fw0","/tmp/fw0",0,501,"g0",protected=True,baseline_margin=500),
            CaseRecord("fw1","/tmp/fw1",1,502,"g1",protected=True,baseline_margin=2500),
            CaseRecord("sw0","/tmp/sw0",0,503,"g2",protected=True,baseline_margin=15000),
            CaseRecord("sw1","/tmp/sw1",1,504,"g3",protected=True,baseline_margin=30000),
        ]
        manager=CasePanelManager(CaseBank(losses+wins),production_mode=True,min_protected_fraction=0.25)
        panel=manager.select_screen(8,seed=19)
        protected=[x for x in panel if x.protected and x.baseline_margin>0]
        self.assertGreaterEqual(len(protected),2)
        self.assertTrue(any(x.baseline_margin<0 for x in panel))
        self.assertEqual({x.seat for x in protected},{0,1})
    def test_detection_recall_is_separate_from_stage2_selection_recall(self):
        from tools.bkg4_search.edit_search.panels import CasePanelManager
        manager = CasePanelManager(self.cases(), useful_recall_floor=0.8, danger_recall_floor=0.9)
        diag = manager.stage_detection_recall(
            screen_useful_ids={"u0"}, screen_l2w_ids={"u0"}, screen_dangerous_ids={"d0"},
            full_useful_ids={"u0", "u1"}, full_l2w_ids={"u0", "u1"}, full_dangerous_ids={"d0", "d1"},
            finalist_ids={"u0", "d0", "d1"},
        )
        self.assertAlmostEqual(diag["useful_recall"], 0.5)
        self.assertAlmostEqual(diag["danger_recall"], 0.5)
        self.assertAlmostEqual(diag["selection_useful_recall"], 0.5)
        self.assertAlmostEqual(diag["selection_danger_recall"], 1.0)
        self.assertTrue(manager.refresh_required(diag))


    def test_counterexamples_cannot_saturate_production_screen(self):
        from tools.bkg4_search.edit_search.panels import CaseBank, CasePanelManager, CaseRecord
        rows=[]
        for i in range(120):
            if i<40: margin=-100-i
            elif i<80: margin=500+(i%20)*100
            else: margin=10000+i
            rows.append(CaseRecord(f"c{i}",f"/tmp/t{i}",i%2,1000+i,f"family-{i%8}",protected=i>=40,baseline_margin=margin))
        manager=CasePanelManager(CaseBank(rows),production_mode=True,random_anchor_fraction=0.1)
        for row in rows:
            manager.add_hard_case(row.case_id,severity=1000+(row.case_id.__hash__()&255),reason="counterexample")
        panel=manager.select_screen(20,seed=33)
        ids={x.case_id for x in panel}
        self.assertEqual(len(panel),20)
        self.assertTrue(any(x.baseline_margin<0 for x in panel))
        self.assertTrue(any(0<x.baseline_margin<=5000 for x in panel))
        self.assertTrue(any(x.baseline_margin>5000 for x in panel))
        self.assertEqual({x.seat for x in panel},{0,1})
        active=set(manager.active_counterexamples(20))
        active_in_panel=sum(x.case_id in active for x in panel)
        self.assertLessEqual(active_in_panel,6,"20-case screen with 30% CE quota must never contain more than six active counterexamples")
        composition=manager.screen_composition(panel)
        self.assertLessEqual(composition["counterexamples"],6)
        self.assertGreater(composition["target_losses"],0)
        self.assertGreater(composition["fragile_wins"],0)
        self.assertGreater(composition["strong_wins"],0)
        self.assertGreater(composition["counterexamples"],0)

    def test_counterexample_memory_compresses_redundant_public_failure_signatures(self):
        from tools.bkg4_search.edit_search.panels import CaseBank, CasePanelManager, CaseRecord
        rows=[]
        for i in range(40):
            payload='{"context_schema":"x"}'
            rows.append(CaseRecord(f"c{i}",f"/tmp/t{i}",i%2,10+i,"same-family",protected=True,baseline_margin=1000,model_context_json=payload,context_probe_steps=(72,),public_context_width=1,public_context_registry_width=1,public_context_values=(100,)))
        manager=CasePanelManager(CaseBank(rows),counterexample_memory_limit=8)
        for i in range(40): manager.add_hard_case(f"c{i}",severity=i+1,reason="w2l")
        compressed=manager.compressed_counterexamples()
        self.assertLessEqual(len(compressed),8)
        self.assertIn("c39",compressed)

    def test_detection_recall_includes_target_repair_recall(self):
        from tools.bkg4_search.edit_search.panels import CasePanelManager
        manager=CasePanelManager(self.cases(),useful_recall_floor=0.8,danger_recall_floor=0.9)
        diag=manager.stage_detection_recall(
            screen_useful_ids={"u"},screen_l2w_ids={"l"},screen_dangerous_ids={"d"},
            full_useful_ids={"u"},full_l2w_ids={"l"},full_dangerous_ids={"d"},finalist_ids={"u","l","d"},
            screen_target_repair_ids={"t0"},full_target_repair_ids={"t0","t1"},
        )
        self.assertEqual(diag["target_repair_recall"],0.5)
        self.assertTrue(manager.refresh_required(diag))


if __name__ == "__main__":
    unittest.main(verbosity=2)
