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


if __name__ == "__main__":
    unittest.main(verbosity=2)
