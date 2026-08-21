from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "cuda-engine/src/bkg4_edit_corpus_eval.cu"


class EditCorpusCudaSourceTests(unittest.TestCase):
    def test_source_is_current_bkg4_overlay_evaluator_with_expected_abi(self):
        self.assertTrue(SOURCE.exists(), f"missing {SOURCE}")
        text = SOURCE.read_text(encoding="utf-8")
        self.assertIn("B4EC1", text)
        self.assertIn("ModuleGenomeBin", text)
        self.assertIn("module_route", text)
        self.assertIn("load_compiled4", text)
        self.assertIn("load_extended_route_pack", text)
        self.assertIn("strategic_route_slot", text)
        self.assertIn("apply_cuda_engine_exploits", text)
        self.assertIn("reconcile_feed", text)
        self.assertIn("action_changes", text)
        self.assertIn("activations", text)
        self.assertIn("action_trace_hash", text)
        self.assertNotIn("--use_fast_math", text)
        self.assertNotIn("__FAST_MATH__", text)


if __name__ == "__main__":
    unittest.main()
