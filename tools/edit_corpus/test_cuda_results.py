from __future__ import annotations

import struct
import tempfile
import unittest
from pathlib import Path


class EditCudaResultTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            from tools.edit_corpus import cuda_results
        except ImportError as exc:
            raise AssertionError(f"CUDA result parser is not implemented yet: {exc}") from exc
        cls.r = cuda_results

    def write(self, raw: bytes) -> Path:
        td = tempfile.TemporaryDirectory()
        self.addCleanup(td.cleanup)
        p = Path(td.name) / "x.b4ec"
        p.write_bytes(raw)
        return p

    def test_load_edit_results_validates_dimensions_and_fields(self):
        head = struct.pack("<8sIII", b"B4EC1\0\0\0", 1, 2, 2)
        rec = struct.Struct("<iiHHHHQ")
        raw = head + b"".join([
            rec.pack(100, 50, 0, 0, 0, 0, 11),
            rec.pack(80, 90, 3, 4, 0, 0, 12),
            rec.pack(120, 30, 7, 8, 0, 0, 21),
            rec.pack(40, 60, 2, 2, 0, 0, 22),
        ])
        rows = self.r.load_edit_results(self.write(raw))
        self.assertEqual(len(rows), 2)
        self.assertEqual(len(rows[0]), 2)
        self.assertEqual(rows[0][0].margin, 50)
        self.assertEqual(rows[0][1].margin, -10)
        self.assertEqual(rows[1][0].action_changes, 7)
        self.assertEqual(rows[1][0].activations, 8)
        self.assertEqual(rows[1][0].action_trace_hash, 21)

    def test_load_edit_results_rejects_bad_magic_trailing_bytes_and_flags(self):
        rec = struct.Struct("<iiHHHHQ")
        good_head = struct.pack("<8sIII", b"B4EC1\0\0\0", 1, 1, 1)
        with self.subTest("magic"):
            p = self.write(struct.pack("<8sIII", b"WRONG\0\0\0", 1, 1, 1) + rec.pack(1, 2, 0, 0, 0, 0, 1))
            with self.assertRaisesRegex(ValueError, "header"):
                self.r.load_edit_results(p)
        with self.subTest("trailing"):
            p = self.write(good_head + rec.pack(1, 2, 0, 0, 0, 0, 1) + b"x")
            with self.assertRaisesRegex(ValueError, "size"):
                self.r.load_edit_results(p)
        with self.subTest("flags"):
            p = self.write(good_head + rec.pack(1, 2, 0, 0, 4, 0, 1))
            with self.assertRaisesRegex(ValueError, "flags=4"):
                self.r.load_edit_results(p)


if __name__ == "__main__":
    unittest.main()
