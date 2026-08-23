from __future__ import annotations

import struct
import tempfile
import unittest
from pathlib import Path

from tools.bkg4_search.invariant_evaluator import load_invariant_results


class InvariantCudaIoTests(unittest.TestCase):
    def test_b4ic1_result_parser(self):
        hdr = struct.pack("<8sIII", b"B4IC1\0\0\0", 1, 2, 1)
        row = struct.Struct("<iiHHHHQ")
        raw = hdr + row.pack(100, 80, 3, 4, 0, 0, 123) + row.pack(10, 20, 5, 6, 0, 0, 456)
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "x.b4ic"; p.write_bytes(raw)
            got = load_invariant_results(p)
        self.assertEqual(len(got), 2)
        self.assertEqual(got[0][0].margin, 20)
        self.assertEqual(got[1][0].action_trace_hash, 456)

    def test_parser_fails_closed_on_flags(self):
        hdr = struct.pack("<8sIII", b"B4IC1\0\0\0", 1, 1, 1)
        row = struct.Struct("<iiHHHHQ")
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "x.b4ic"; p.write_bytes(hdr + row.pack(1, 2, 0, 0, 1, 0, 0))
            with self.assertRaisesRegex(ValueError, "flags"):
                load_invariant_results(p)


if __name__ == "__main__": unittest.main()
