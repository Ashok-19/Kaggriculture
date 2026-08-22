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

    def test_inactive_overlay_tick_uses_single_final_action_materialization(self):
        text = SOURCE.read_text(encoding="utf-8")
        self.assertIn("if (override_route == 0xffffu)", text)
        self.assertIn("candidate = baseline", text)
        self.assertIn("legacy_cmp", text)
        self.assertIn("exploit_cmp", text)

    def test_cuda_corpus_tools_cache_duplicate_native_tapes(self):
        edit = (ROOT / "cuda-engine/src/bkg4_edit_corpus_eval.cu").read_text(encoding="utf-8")
        capture = (ROOT / "cuda-engine/src/bkg4_information_state_capture.cu").read_text(encoding="utf-8")
        self.assertIn("tape_cache", edit)
        self.assertIn("tape_cache", capture)


    def test_cuda_makefile_exposes_reproducible_b4ec1_build_target(self):
        makefile = (ROOT / "cuda-engine/Makefile").read_text(encoding="utf-8")
        self.assertIn("$(BUILD_DIR)/bkg4_edit_corpus_eval:", makefile)
        self.assertIn("bkg4-edit-corpus-eval:", makefile)
        self.assertIn("src/bkg4_edit_corpus_eval.cu", makefile)


if __name__ == "__main__":
    unittest.main()
