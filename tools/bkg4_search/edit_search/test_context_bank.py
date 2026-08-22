from __future__ import annotations

import json
import struct
import tempfile
import unittest
from pathlib import Path

from tools.bkg4_search.edit_search.panels import CaseBank, CaseRecord
from tools.bkg4_search.information_state_io import HEADER, registry_manifest


class MultiTemporalContextBankTests(unittest.TestCase):
    def _capture(self, root: Path, cases: int = 4) -> Path:
        from tools.bkg4_search.edit_search.context_bank import strategic_probe_schedule
        probes = strategic_probe_schedule().probe_steps
        registry = len(registry_manifest())
        width = registry
        path = root / "r0.b4is"
        values = []
        for ci in range(cases):
            for step in probes:
                frame = [0] * width
                frame[1] = step
                frame[4] = 1000 + ci * 100 + step
                values.extend(frame)
        raw = bytearray(HEADER.pack(b"B4ISR01\0", 1, cases, len(probes), width, registry, 0, 0, 0))
        raw.extend(struct.pack(f"<{len(probes)}i", *probes))
        raw.extend(struct.pack(f"<{len(values)}i", *values))
        path.write_bytes(raw)
        return path

    def _bank(self) -> CaseBank:
        return CaseBank([
            CaseRecord(f"c{i}", f"/tmp/t{i}", i % 2, 100 + i, f"family-{i % 2}", protected=True,
                       baseline_margin=(-100 if i == 0 else 1000 if i == 1 else 10000 + i))
            for i in range(4)
        ])

    def test_strategic_schedule_spans_five_temporally_separated_windows(self):
        from tools.bkg4_search.edit_search.context_bank import strategic_probe_schedule
        schedule = strategic_probe_schedule()
        self.assertEqual(tuple(x.name for x in schedule.windows), ("early", "early_mid", "mid", "late_mid", "late"))
        self.assertEqual(len(schedule.windows), 5)
        for window in schedule.windows:
            self.assertGreaterEqual(len(window.probe_steps), 2)
        self.assertLess(max(schedule.windows[0].probe_steps), min(schedule.windows[-1].probe_steps))
        self.assertIn(72, schedule.probe_steps)
        self.assertIn(288, schedule.probe_steps)
        self.assertIn(648, schedule.probe_steps)

    def test_build_context_bank_uses_only_public_capture_and_explicit_role(self):
        from tools.bkg4_search.edit_search.context_bank import CaseBankRole, build_context_bank_from_r0
        with tempfile.TemporaryDirectory() as td:
            path = self._capture(Path(td))
            bank = build_context_bank_from_r0(self._bank(), path, role=CaseBankRole.CERTIFICATION)
        self.assertEqual(len(bank), 4)
        self.assertEqual({x.seat for x in bank}, {0, 1})
        self.assertEqual({x.role for x in bank}, {"certification"})
        self.assertGreater(len(bank.rows[0].context_probe_steps), 10)
        payload = json.loads(bank.rows[0].model_context_json)
        self.assertEqual(payload["context_schema"], "bkg4_r0_public_multitemporal_v1")
        forbidden = {"team", "opponent", "submission", "episode", "seed", "replay_id"}
        self.assertFalse(forbidden & set(payload))

    def test_role_partition_is_disjoint_and_preserves_outcome_and_seat_coverage(self):
        from tools.bkg4_search.edit_search.context_bank import partition_role_banks
        rows = []
        for i in range(30):
            if i < 10: margin = -100 - i
            elif i < 20: margin = 1000 + i
            else: margin = 10000 + i
            rows.append(CaseRecord(f"c{i}", f"/tmp/t{i}", i % 2, 100 + i, f"family-{i % 5}", baseline_margin=margin))
        split = partition_role_banks(CaseBank(rows), discovery=12, certification=10, ranker_validation=8, seed=77)
        ids = {k: {x.case_id for x in v} for k, v in split.items()}
        self.assertFalse(ids["discovery"] & ids["certification"])
        self.assertFalse(ids["discovery"] & ids["ranker_validation"])
        self.assertFalse(ids["certification"] & ids["ranker_validation"])
        for bank in split.values():
            self.assertEqual({x.seat for x in bank}, {0, 1})
            self.assertTrue(any(x.baseline_margin < 0 for x in bank))
            self.assertTrue(any(0 < x.baseline_margin <= 5000 for x in bank))
            self.assertTrue(any(x.baseline_margin > 5000 for x in bank))


if __name__ == "__main__":
    unittest.main(verbosity=2)
