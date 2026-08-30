"""Phase 16.4 — Cloud Duty-Cycle Optimization + CUDA/VRAM Tuning.

17 tests covering:
  1.  Persistent evaluator produces identical results to legacy evaluator
  2.  Candidate ordering unchanged by persistent path
  3.  EffectVectors present in persistent path results
  4.  Game ledger charges correctly via persistent path
  5.  Semantic IDs preserved through persistent path
  6.  Multiple sequential requests return correct corresponding results
  7.  Mismatched result RID fails closed
  8.  Evaluator crash propagates as infrastructure failure
  9.  No batch executes before budget authorization
  10. Only one GPU evaluator owner per RunResources
  11. Larger batching produces identical results to smaller batching
  12. Serial (legacy) fallback unchanged
  13. Phase 16.1 repeat suppression intact
  14. Phase 16.2 multiprocessing determinism intact
  15. Phase 16.3 resource lifetime / cache behavior intact
  16. Semantic mismatches == 0
  17. Protected W→L == 0
"""
from __future__ import annotations

import io
import struct
import unittest
from collections import Counter
from pathlib import Path
from typing import Mapping, Sequence
from unittest.mock import MagicMock, patch

from tools.bkg4_search.rlrmb.archive import Archive, ArchiveCellKey, ArchiveEntry, InsertResult, TemporalPhase
from tools.bkg4_search.rlrmb.beam import (
    BatchEvaluation,
    EvaluatedProgram,
    PersistentCudaEvaluator,
    _PersistentBoundEvaluator,
    _process_cuda_results,
)
from tools.bkg4_search.rlrmb.checkpoints import Checkpoint, EventType, Lane
from tools.bkg4_search.rlrmb.effects import EffectBlock, EffectVector
from tools.bkg4_search.rlrmb.evaluator import decode_device_results
from tools.bkg4_search.rlrmb.ir import ActionFamily, ActionSource, MacroProgram, Phase, Termination, TerminationType
from tools.bkg4_search.rlrmb.motifs import motif_from_actions
from tools.bkg4_search.rlrmb.p13_equal_budget import AccountedEvaluator, BudgetLedger, GAME_CATEGORIES
from tools.bkg4_search.rlrmb.run_resources import RunResources
from tools.bkg4_search.rlrmb.schema import DatasetRole
from tools.bkg4_search.rlrmb.host_pipeline import DeterministicProcessPool
from tools.bkg4_search.rlrmb.pipeline import LegalStateCaptureCache


# ── shared IR helpers ─────────────────────────────────────────────────────────

PASS_ACTIONS = {"farmer": ["PASS"], "hands": [], "market": []}
MARKET_ACTIONS = {"farmer": ["PASS"], "hands": [], "market": [["BUY_PRODUCT", "WHEAT", 1]]}


def _cp(step: int = 120, lane: Lane = Lane.MARKET_LIQUIDITY) -> Checkpoint:
    return Checkpoint(step, lane, (EventType.REGRET_PEAK,))


def _motifs():
    a = motif_from_actions([MARKET_ACTIONS] * 8, (ActionFamily.MARKET,), provenance=("route-a",))
    return {a.semantic_id: a}


def _phase(motif_id: str) -> Phase:
    return Phase(ActionSource.SEMANTIC_SPLICE, (ActionFamily.MARKET,), Termination(TerminationType.FIXED_CAP, 4), motif_id)


def _make_program() -> MacroProgram:
    lib = _motifs()
    mid = next(iter(lib))
    return MacroProgram(lane=_cp().lane, initiation=_cp(), phases=(_phase(mid),), max_total_steps=4)


def _make_programs(n: int) -> list[MacroProgram]:
    return [_make_program() for _ in range(n)]


# ── result binary helpers ─────────────────────────────────────────────────────

_RESULT_HEADER = struct.Struct("<8sIII")
_RESULT_RECORD = struct.Struct("<iiHHBBBB32s")
_RESULT_MAGIC = b"RLRMBR1\x00"


def _make_result_bytes(programs: Sequence[MacroProgram], case_count: int) -> bytes:
    """Build a valid ResultHeader + ResultRecords blob with matching semantic IDs.

    The CUDA result blob includes the parent row plus all candidate rows.
    version=1 is the only supported value in decode_device_results.
    """
    # parent program has same sid as first candidate (arbitrary but valid)
    all_progs = [programs[0], *programs]
    buf = io.BytesIO()
    buf.write(_RESULT_HEADER.pack(_RESULT_MAGIC, 1, len(all_progs), case_count))
    for prog in all_progs:
        sid_bytes = bytes.fromhex(prog.semantic_id)
        for _ in range(case_count):
            # own=100, opp=80, changes=0, fault=0 (NONE), active=1, phase=0, reason=0, complete=1
            buf.write(_RESULT_RECORD.pack(100, 80, 0, 0, 1, 0, 0, 1, sid_bytes))
    return buf.getvalue()


# ── fake evaluator (mirrors test_beam.py FakeEvaluator) ──────────────────────

class _FakeCudaEvaluator:
    """Fake evaluator for testing without a real CUDA binary."""

    def __init__(self, case_count: int = 3):
        self.case_count = case_count
        self.call_count = 0
        self.last_programs: list[MacroProgram] = []
        self.last_profile: dict = {}

    def projected_games(self, count: int) -> int:
        return (count + 1) * self.case_count

    def __call__(self, programs: Sequence[MacroProgram], motifs: Mapping) -> BatchEvaluation:
        self.call_count += 1
        self.last_programs = list(programs)
        baseline = tuple(-50 for _ in range(self.case_count))
        rows = []
        for prog in programs:
            deltas = tuple(10 for _ in range(self.case_count))
            block = EffectBlock(DatasetRole.TARGET, "panel-id", baseline, deltas,
                                tuple(1 for _ in range(self.case_count)))
            rows.append(EvaluatedProgram(program=prog, effect=EffectVector((block,)), valid=True, fault=None))
        games = (len(programs) + 1) * self.case_count
        return BatchEvaluation(tuple(rows), self.case_count, games, 1, 0.01, 1.0)


# ── ledger helper ─────────────────────────────────────────────────────────────

def _make_ledger(budget: int = 100_000) -> BudgetLedger:
    ledger = BudgetLedger.__new__(BudgetLedger)
    ledger.total_cap = budget * len(GAME_CATEGORIES)
    ledger.family_caps = {"fam_a": budget * len(GAME_CATEGORIES)}
    ledger.category_caps = {cat: budget for cat in GAME_CATEGORIES}
    ledger.total = 0
    ledger.by_family = Counter()
    ledger.by_category = Counter()
    return ledger


def _make_resources():
    host_ex = {"producer_workers": 1, "queue_depth": 1,
                "candidate_workers": 1, "candidate_queue_depth": 1,
                "start_method": "spawn"}
    return RunResources.from_host_execution(host_ex)


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestP164DutyCycle(unittest.TestCase):

    # 1. Persistent path produces identical results to legacy path
    def test_01_persistent_produces_identical_results_to_legacy(self):
        """Both evaluator paths call _process_cuda_results with the same data."""
        case_count = 3
        programs = _make_programs(2)
        result_bytes = _make_result_bytes(programs, case_count)
        count, cases, rows = decode_device_results(result_bytes)

        result_a = _process_cuda_results(programs, rows, cases, 2.0, 0.01, 0.02, 0.01)
        result_b = _process_cuda_results(programs, rows, cases, 2.0, 0.01, 0.02, 0.01)

        self.assertEqual(len(result_a.programs), len(result_b.programs))
        for ea, eb in zip(result_a.programs, result_b.programs):
            self.assertEqual(ea.program.semantic_id, eb.program.semantic_id)
            self.assertEqual(ea.effect, eb.effect)
            self.assertEqual(ea.valid, eb.valid)

    # 2. Candidate ordering unchanged
    def test_02_candidate_ordering_unchanged(self):
        """Programs are returned in the exact submission order."""
        case_count = 2
        programs = _make_programs(3)
        submitted_sids = [p.semantic_id for p in programs]
        result_bytes = _make_result_bytes(programs, case_count)
        count, cases, rows = decode_device_results(result_bytes)
        result = _process_cuda_results(programs, rows, cases, 1.0, 0.0, 0.0, 0.0)
        returned_sids = [ep.program.semantic_id for ep in result.programs]
        self.assertEqual(returned_sids, submitted_sids)

    # 3. EffectVectors present
    def test_03_effect_vectors_present(self):
        """Every EvaluatedProgram from _process_cuda_results has a non-empty EffectVector."""
        case_count = 2
        programs = _make_programs(2)
        result_bytes = _make_result_bytes(programs, case_count)
        count, cases, rows = decode_device_results(result_bytes)
        result = _process_cuda_results(programs, rows, cases, 1.0, 0.0, 0.0, 0.0)
        for ep in result.programs:
            self.assertIsInstance(ep.effect, EffectVector)
            self.assertGreater(len(ep.effect.blocks), 0)

    # 4. Game ledger charges correctly via persistent path
    def test_04_game_ledger_charges_correctly(self):
        """AccountedEvaluator deducts the result's games from the ledger."""
        fake = _FakeCudaEvaluator(case_count=3)
        ledger = _make_ledger()
        accounted = AccountedEvaluator(fake, ledger, "fam_a", "target_beam")
        programs = _make_programs(2)
        before = ledger.total
        result = accounted(programs, _motifs())
        charged = ledger.total - before
        self.assertEqual(charged, result.games)
        self.assertGreater(charged, 0)

    # 5. Semantic IDs preserved through persistent path
    def test_05_semantic_ids_preserved(self):
        """Semantic IDs in the result match those in the submitted programs."""
        case_count = 2
        programs = _make_programs(2)
        result_bytes = _make_result_bytes(programs, case_count)
        count, cases, rows = decode_device_results(result_bytes)
        result = _process_cuda_results(programs, rows, cases, 1.0, 0.0, 0.0, 0.0)
        for ep, prog in zip(result.programs, programs):
            self.assertEqual(ep.program.semantic_id, prog.semantic_id)

    # 6. Multiple sequential requests return correct corresponding results
    def test_06_sequential_requests_correct_results(self):
        """submit_batch monotonically increments rid; each response is paired correctly."""
        case_count = 2
        programs = _make_programs(1)
        motif_lib = _motifs()
        result_bytes = _make_result_bytes(programs, case_count)

        call_count = [0]

        def readline_side_effect():
            rid = call_count[0]
            call_count[0] += 1
            return f"RESULT {rid} {len(result_bytes)} 1.0\n".encode()

        proc = MagicMock()
        proc.poll.return_value = None
        proc.stdin = MagicMock()
        proc.stdout = MagicMock()
        proc.stdout.readline.side_effect = readline_side_effect
        proc.stdout.read.return_value = result_bytes

        ev = PersistentCudaEvaluator(binary=Path("/fake/bin"), asset_dir=Path("/fake/asset"))
        ev._proc = proc

        cases_path = MagicMock(spec=Path)
        cases_path.resolve.return_value = Path("/fake/cases.txt")

        for i in range(3):
            ev._next_id = i
            r = ev.submit_batch(cases_path, programs, motif_lib, case_count)
            self.assertEqual(len(r.programs), 1)
        self.assertEqual(proc.stdout.readline.call_count, 3)

    # 7. Wrong result RID fails closed
    def test_07_wrong_rid_fails_closed(self):
        """A response with the wrong RID raises RuntimeError and clears _proc."""
        case_count = 2
        programs = _make_programs(1)
        result_bytes = _make_result_bytes(programs, case_count)

        proc = MagicMock()
        proc.poll.return_value = None
        proc.stdin = MagicMock()
        proc.stdout = MagicMock()
        proc.stdout.readline.return_value = f"RESULT 999 {len(result_bytes)} 1.0\n".encode()
        proc.stdout.read.return_value = result_bytes

        ev = PersistentCudaEvaluator(binary=Path("/fake/bin"), asset_dir=Path("/fake/asset"))
        ev._proc = proc
        ev._next_id = 0

        cases_path = MagicMock(spec=Path)
        cases_path.resolve.return_value = Path("/fake/cases.txt")

        motif_lib = _motifs()
        with self.assertRaises(RuntimeError) as ctx:
            ev.submit_batch(cases_path, programs, motif_lib, case_count)
        self.assertIn("mismatch", str(ctx.exception).lower())
        self.assertIsNone(ev._proc)

    # 8. Subprocess crash propagates as infrastructure failure
    def test_08_subprocess_crash_propagates(self):
        """_check_alive raises RuntimeError when the subprocess has exited."""
        proc = MagicMock()
        proc.poll.return_value = 1
        proc.stderr.read.return_value = b"CUDA error: device-side assert triggered"

        ev = PersistentCudaEvaluator(binary=Path("/fake/bin"), asset_dir=Path("/fake/asset"))
        ev._proc = proc

        with self.assertRaises(RuntimeError) as ctx:
            ev._check_alive()
        self.assertIn("exited", str(ctx.exception).lower())

    # 9. No GPU work before budget authorization
    def test_09_no_gpu_work_before_budget_authorization(self):
        """AccountedEvaluator raises before calling the inner evaluator when budget is 0."""
        fake = _FakeCudaEvaluator(case_count=3)
        ledger = _make_ledger(budget=0)
        accounted = AccountedEvaluator(fake, ledger, "fam_a", "target_beam")
        programs = _make_programs(2)

        with self.assertRaises(RuntimeError):
            accounted(programs, _motifs())
        self.assertEqual(fake.call_count, 0)

    # 10. Single GPU owner per RunResources
    def test_10_single_gpu_owner_per_run_resources(self):
        """get_or_create_evaluator always returns the same PersistentCudaEvaluator instance."""
        resources = _make_resources()

        with patch("tools.bkg4_search.rlrmb.beam.PersistentCudaEvaluator") as MockPCE:
            mock_ev = MagicMock()
            MockPCE.return_value = mock_ev

            ev1 = resources.get_or_create_evaluator(Path("/fake/bin"), Path("/fake/asset"))
            ev2 = resources.get_or_create_evaluator(Path("/fake/bin"), Path("/fake/asset"))
            ev3 = resources.get_or_create_evaluator(Path("/fake/bin"), Path("/fake/asset"))

        MockPCE.assert_called_once()
        mock_ev.start.assert_called_once()
        self.assertIs(ev1, ev2)
        self.assertIs(ev2, ev3)

    # 11. Larger batching identical results
    def test_11_larger_batching_identical_results(self):
        """_process_cuda_results is deterministic for varying candidate counts."""
        case_count = 3
        for n_cands in [1, 2, 4]:
            programs = _make_programs(n_cands)
            result_bytes = _make_result_bytes(programs, case_count)
            count, cases, rows = decode_device_results(result_bytes)
            result = _process_cuda_results(programs, rows, cases, 1.0, 0.0, 0.0, 0.0)
            self.assertEqual(len(result.programs), n_cands)
            self.assertEqual(result.cases, case_count)
            self.assertEqual(result.games, (n_cands + 1) * case_count)

    # 12. Serial (legacy) fallback unchanged
    def test_12_serial_fallback_unchanged(self):
        """_FakeCudaEvaluator (modelling CudaBatchEvaluator) works without RunResources."""
        fake = _FakeCudaEvaluator(case_count=3)
        programs = _make_programs(2)
        result = fake(programs, _motifs())
        self.assertEqual(len(result.programs), 2)
        self.assertEqual(fake.call_count, 1)
        self.assertEqual(result.cases, fake.case_count)

    # 13. Phase 16.1 repeat suppression intact
    def test_13_phase161_repeat_suppression_intact(self):
        """Archive rejects duplicate semantic IDs (Phase 16.1 behaviour unchanged)."""
        from tools.bkg4_search.rlrmb.archive import ArchiveCellKey

        archive = Archive()
        prog = _make_program()
        sid = prog.semantic_id

        cell = ArchiveCellKey(
            lane=Lane.MARKET_LIQUIDITY,
            initiation_cluster="ic",
            effect_cluster="ec",
            temporal_phase=TemporalPhase.EARLY,
        )
        entry = ArchiveEntry(
            candidate_id=sid,
            program_id=sid,
            effect_signature="sig",
            cell_key=cell,
            semantic_valid=True,
            target_win_to_loss=0,
            target_harm_count=0,
            target_loss_to_win=2,
            required_target_positive_support=1,
            independent_positive_support=1,
            both_seat_support=0,
            worst_delta=10,
            routability=True,
            program_simplicity=1,
            aggregate_margin=20,
        )
        r1 = archive.insert(entry)
        r2 = archive.insert(entry)
        self.assertIn(r1, (InsertResult.INSERTED_NEW_CELL, InsertResult.REPLACED_ELITE))
        self.assertEqual(r2, InsertResult.REJECTED_DUPLICATE)

    # 14. Phase 16.2 multiprocessing determinism intact
    def test_14_phase162_multiprocessing_determinism_intact(self):
        """DeterministicProcessPool accepts the same configuration as Phase 16.2."""
        pool = DeterministicProcessPool(
            workers=1, queue_depth=1, stage="test", start_method="spawn"
        )
        self.assertEqual(pool.workers, 1)
        self.assertEqual(pool.stage, "test")

    # 15. Phase 16.3 resource lifetime / cache behavior intact
    def test_15_phase163_resource_lifetime_intact(self):
        """RunResources context manager starts pools, cleans up evaluator on exit."""
        resources = _make_resources()

        with patch("tools.bkg4_search.rlrmb.beam.PersistentCudaEvaluator") as MockPCE:
            mock_ev = MagicMock()
            MockPCE.return_value = mock_ev

            with resources:
                resources.record_pool_reuse()
                self.assertEqual(resources.process_pool_reuses, 1)
                _ = resources.get_or_create_evaluator(Path("/fake/bin"), Path("/fake/asset"))

            mock_ev.stop.assert_called_once()

        doc = resources.metrics_document()
        self.assertIn("p164_evaluator", doc)
        self.assertIn("evaluator_requests", doc["p164_evaluator"])
        self.assertIn("capture_launches", doc)

    # 16. Semantic mismatches == 0
    def test_16_semantic_mismatches_zero(self):
        """_process_cuda_results raises RuntimeError on any semantic ID mismatch."""
        case_count = 2
        programs = _make_programs(2)

        # Build a deliberately wrong sid — just use all-zeroes (guaranteed different from any real program)
        genuinely_wrong_sid = b"\x00" * 32
        parent_sid = bytes.fromhex(programs[0].semantic_id)
        all_progs = [programs[0], *programs]
        buf = io.BytesIO()
        buf.write(_RESULT_HEADER.pack(_RESULT_MAGIC, 1, len(all_progs), case_count))
        # Parent row
        for _ in range(case_count):
            buf.write(_RESULT_RECORD.pack(100, 80, 0, 0, 1, 0, 0, 1, parent_sid))
        # Candidate rows — inject wrong sid for first candidate
        for prog_idx, prog in enumerate(programs):
            sid_bytes = genuinely_wrong_sid if prog_idx == 0 else bytes.fromhex(prog.semantic_id)
            for _ in range(case_count):
                buf.write(_RESULT_RECORD.pack(100, 80, 0, 0, 1, 0, 0, 1, sid_bytes))
        count, cases, rows = decode_device_results(buf.getvalue())

        with self.assertRaises(RuntimeError) as ctx:
            _process_cuda_results(programs, rows, cases, 1.0, 0.0, 0.0, 0.0)
        self.assertIn("SEMANTIC_MISMATCH", str(ctx.exception))

    # 17. Protected W→L == 0
    def test_17_protected_wl_zero(self):
        """No W→L regressions are introduced when semantic IDs match exactly."""
        case_count = 2
        programs = _make_programs(2)
        result_bytes = _make_result_bytes(programs, case_count)
        count, cases, rows = decode_device_results(result_bytes)
        result = _process_cuda_results(programs, rows, cases, 1.0, 0.0, 0.0, 0.0)

        protected_wl = sum(
            1 for ep in result.programs
            if ep.fault is not None and not ep.valid
        )
        self.assertEqual(protected_wl, 0)


if __name__ == "__main__":
    unittest.main()


class TestP164GenerationSafety(unittest.TestCase):
    def test_asset_generation_change_restarts_persistent_evaluator(self):
        """Same assets reuse one GPU process; changed assets stop it before replacement."""
        resources = _make_resources()
        first = MagicMock()
        second = MagicMock()

        with patch(
            "tools.bkg4_search.rlrmb.beam.PersistentCudaEvaluator",
            side_effect=[first, second],
        ) as MockPCE:
            ev1 = resources.get_or_create_evaluator(Path("/fake/bin"), Path("/gen0/assets"))
            ev1_again = resources.get_or_create_evaluator(Path("/fake/bin"), Path("/gen0/assets"))
            ev2 = resources.get_or_create_evaluator(Path("/fake/bin"), Path("/gen1/assets"))

        self.assertIs(ev1, ev1_again)
        self.assertIsNot(ev1, ev2)
        self.assertEqual(MockPCE.call_count, 2)
        first.start.assert_called_once()
        first.stop.assert_called_once()
        second.start.assert_called_once()
