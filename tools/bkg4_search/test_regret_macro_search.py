from __future__ import annotations

import json
import struct
import tempfile
import unittest
from pathlib import Path

from tools.bkg4_search.cuda_io import FRAGMENT_RESULT_HEAD, RESULT, load_fragment_results
from tools.bkg4_search.regret_macro_search import sanitize_native_tape, summarize_fragment_run


class RegretMacroSearchTests(unittest.TestCase):
    def write_result(self, path: Path) -> None:
        rows = [
            [(-5, 0, 0, 0), (10, 0, 0, 0), (-2, 0, 0, 0)],
            [(3, 0, 7, 0), (8, 0, 4, 0), (-1, 0, 2, 0)],
        ]
        raw = bytearray(FRAGMENT_RESULT_HEAD.pack(b"B4FR1\0\0\0", 1, 2, 3, 0, 2))
        for candidate in rows:
            for row in candidate:
                raw += RESULT.pack(*row)
        path.write_bytes(raw)

    def test_fragment_result_parser_and_regret_summary(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            result = root / "result.b4fr"
            self.write_result(result)
            meta, rows = load_fragment_results(result)
            self.assertEqual(meta, {"candidates": 2, "cases": 3, "start_index": 0, "total_candidates": 2})
            self.assertEqual(rows[1][0], (3, 0, 7, 0))
            (root / "grid.tsv").write_text(
                "id\tdonor\tstart\tduration\tpath\n"
                "0\t-1\t0\t0\tBASE\n"
                "1\t0\t48\t24\tdonor.json\n",
                encoding="utf-8",
            )
            (root / "cases.json").write_text(json.dumps([{"id": "a"}, {"id": "b"}, {"id": "c"}]), encoding="utf-8")
            summary = summarize_fragment_run(result, root / "grid.tsv", root / "cases.json")
            stat = summary["candidate_stats"][0]
            self.assertEqual(stat["positive"], 2)
            self.assertEqual(stat["harmful"], 1)
            self.assertEqual(stat["loss_to_win"], 1)
            self.assertEqual(stat["win_to_loss"], 0)
            self.assertEqual(stat["sum_delta"], 7)
            self.assertEqual(stat["min_delta"], -2)
            self.assertEqual(stat["max_delta"], 8)
            self.assertTrue(summary["best_per_case"][0]["loss_to_win"])
            self.assertEqual(summary["best_per_case"][1]["best_candidate_id"], 0)
            self.assertEqual(summary["best_per_case"][1]["best_margin"], 10)
            self.assertEqual(summary["best_per_case"][1]["best_delta"], 0)

    def test_sanitize_native_tape_removes_metadata_without_changing_actions(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            actions = [{"farmer": ["PASS"], "hands": [], "market": []}]
            src = root / "source.json"
            dst = root / "sanitized.json"
            src.write_text(json.dumps({"team": "カワシギ", "episode": 7, "actions": actions}), encoding="utf-8")
            sanitize_native_tape(src, dst)
            self.assertEqual(json.loads(dst.read_text(encoding="utf-8")), {"actions": actions})

    def test_fragment_result_parser_rejects_truncated_payload(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "bad.b4fr"
            path.write_bytes(FRAGMENT_RESULT_HEAD.pack(b"B4FR1\0\0\0", 1, 1, 1, 0, 1))
            with self.assertRaisesRegex(ValueError, "size mismatch"):
                load_fragment_results(path)


if __name__ == "__main__":
    unittest.main(verbosity=2)
