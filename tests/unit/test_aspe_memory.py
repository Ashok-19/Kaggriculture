from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

from kaggriculture_bonsai.search.aspe import (
    ASPE_ARCHIVE_SCHEMA_VERSION,
    CaseOutcome,
    FailureReason,
    PanelEvidence,
    ProposalMetadata,
    SemanticArchive,
    SpecialistProgram,
    record_sentinel_batch,
    select_build_sentinel,
)
from kaggriculture_bonsai.search.aspe.phenotype import SentinelPanel
from kaggriculture_bonsai.search.checkpoints import Lane
from kaggriculture_bonsai.search.evaluator import CompletionReason, DeviceResult, FaultCode, TerminationReason
from kaggriculture_bonsai.search.ir import ActionSource, Phase, PublicCondition, Termination, TerminationType


def program(feature: str, value: int = 0) -> SpecialistProgram:
    return SpecialistProgram(
        Lane.GENERAL, 24, ((PublicCondition(feature, "GE", value=value),),),
        (Phase(ActionSource.PARENT, (), Termination(TerminationType.FIXED_CAP, 2)),), 2,
    )


def result(*, trace: int = 1, own: int = 100, opp: int = 80, active: bool = True) -> DeviceResult:
    return DeviceResult(
        own_money=own, opp_money=opp, action_changes=int(active), fault=FaultCode.NONE,
        activated=active, final_phase=0,
        end_reason=TerminationReason.FIXED_CAP if active else TerminationReason.NONE,
        complete=active, semantic_id="11" * 32,
        first_activation_step=24 if active else -1, last_activation_step=25 if active else -1,
        activation_count=int(active), specialist_duration=2 if active else 0,
        phase_path=int(active), phase_usage=int(active), branch_path=0,
        completion_reason=CompletionReason.FINISHED if active else CompletionReason.NONE,
        family_farmer_changes=int(active), action_trace_hash=trace if active else 0,
        state_deltas=(own - 90, opp - 80, 0, 0, 0, 0, 0, 0, 0, 0) if active else (0,) * 10,
    )


class AspeMemoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        full = select_build_sentinel("workspace/aspe/baseline/manifest.json")
        cls.panel = SentinelPanel(full.source_panel_identity, full.cases[:2])
        cls.baseline = (result(trace=0, own=90, opp=80),) * 2

    def populate(self, path: Path) -> tuple[SemanticArchive, tuple[SpecialistProgram, ...]]:
        archive = SemanticArchive(path).open()
        programs = (
            program("OWN_MONEY"),
            program("OPP_MONEY"),
            program("OWN_LAND"),
            program("OWN_UNITS"),
        )
        rows = (
            result(trace=1), result(trace=1),
            result(trace=1), result(trace=1),
            result(trace=2), result(trace=2),
            result(trace=3, own=110), result(trace=3, own=110),
        )
        parent = "champion-parent"
        inserted = record_sentinel_batch(
            archive, programs, self.panel, rows, self.baseline,
            context_id="sentinel-context", generation=3,
            parents=(parent,) * 4,
            lineages=(("root", parent),) * 4,
            proposals=(
                ProposalMetadata("SEED", "", {"lane_hint": "GENERAL"}),
                ProposalMetadata("CHANGE_GATE", "initiation[0]", {"lane_hint": "GENERAL"}),
                ProposalMetadata("CHANGE_BODY", "phase[0]", {"lane_hint": "GENERAL"}),
                ProposalMetadata("CHANGE_EFFECT", "phase[0]", {"lane_hint": "GENERAL"}),
            ),
            stepping_stones=(False, True, True, True),
        )
        self.assertTrue(all(item.inserted for item in inserted))
        return archive, programs

    def test_phase2_flow_persists_distinct_dedup_hierarchy(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "archive.json"
            archive, programs = self.populate(path)
            self.assertEqual(len(archive.document["programs"]), 4)
            self.assertEqual(len(archive.document["execution_phenotypes"]), 3)
            self.assertEqual(len(archive.document["strategic_effects"]), 2)
            first = archive.get_evaluation(programs[0].semantic_id, "sentinel-context")
            second = archive.get_evaluation(programs[1].semantic_id, "sentinel-context")
            third = archive.get_evaluation(programs[2].semantic_id, "sentinel-context")
            self.assertNotEqual(first["program_id"], second["program_id"])
            self.assertEqual(first["execution_phenotype_id"], second["execution_phenotype_id"])
            self.assertEqual(first["strategic_effect_id"], third["strategic_effect_id"])
            self.assertIn(FailureReason.PHENOTYPE_DUPLICATE.value, second["failure_reasons"])
            self.assertIn(FailureReason.EFFECT_DUPLICATE.value, third["failure_reasons"])
            self.assertEqual(archive.lineage(programs[1].semantic_id), ("root", "champion-parent"))
            self.assertEqual(len(archive.stepping_stones()), 3)
            self.assertTrue(all(row["authority"] == "SEARCH_MEMORY_ONLY" for row in archive.stepping_stones()))
            archive.close()

    def test_insert_reload_is_byte_deterministic_and_duplicate_bounded(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "archive.json"
            archive = SemanticArchive(path).open()
            p = program("OWN_MONEY")
            rows = (result(), result())
            first = record_sentinel_batch(
                archive, (p,), self.panel, rows, self.baseline,
                context_id="ctx", generation=0,
            )[0]
            original = path.read_bytes()
            for _ in range(20):
                repeated = record_sentinel_batch(
                    archive, (p,), self.panel, rows, self.baseline,
                    context_id="ctx", generation=0,
                )[0]
                self.assertFalse(repeated.inserted)
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(len(archive.document["evaluations"]), 1)
            archive.close()
            with SemanticArchive(path) as restarted:
                self.assertTrue(restarted.has_program(p.semantic_id))
                self.assertEqual(restarted.get_evaluation(p.semantic_id, "ctx")["evaluation_id"], first.evaluation_id)
                self.assertEqual(path.read_bytes(), original)

    def test_exact_duplicates_inside_one_phase2_batch_are_safe(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "archive.json"
            archive = SemanticArchive(path).open()
            p = program("OWN_MONEY")
            rows = (result(), result())
            inserted = record_sentinel_batch(
                archive, (p, p), self.panel, (*rows, *rows), self.baseline,
                context_id="ctx", generation=0,
            )
            self.assertEqual(tuple(item.inserted for item in inserted), (True, False))
            self.assertEqual(len(archive.document["programs"]), 1)
            self.assertEqual(len(archive.document["evaluations"]), 1)
            second_context = record_sentinel_batch(
                archive, (p,), self.panel, rows, self.baseline,
                context_id="ctx-2", generation=1,
                parents=("parent-2",), proposals=(ProposalMetadata("RETRY", "gate"),),
                lineages=(("parent-2",),),
            )[0]
            self.assertTrue(second_context.inserted)
            stored = archive.get_evaluation(p.semantic_id, "ctx-2")
            self.assertIn(FailureReason.STRUCTURAL_DUPLICATE.value, stored["failure_reasons"])
            self.assertEqual(
                archive.failure_statistics(parent_id="parent-2", operator="RETRY", context_id="ctx-2"),
                {
                    FailureReason.EFFECT_DUPLICATE.value: 1,
                    FailureReason.NARROW_SUPPORT.value: 1,
                    FailureReason.PHENOTYPE_DUPLICATE.value: 1,
                    FailureReason.STRUCTURAL_DUPLICATE.value: 1,
                },
            )

    def test_schema_roundtrip_and_failure_queries(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "archive.json"
            archive, programs = self.populate(path)
            document = json.loads(path.read_text())
            self.assertEqual(document["schema_version"], ASPE_ARCHIVE_SCHEMA_VERSION)
            stats = archive.failure_statistics(parent_id="champion-parent", operator="CHANGE_GATE", context_id="sentinel-context")
            self.assertEqual(stats, {
                FailureReason.EFFECT_DUPLICATE.value: 1,
                FailureReason.NARROW_SUPPORT.value: 1,
                FailureReason.PHENOTYPE_DUPLICATE.value: 1,
            })
            self.assertEqual(archive.novel_since_generation(3), tuple(sorted(archive.document["execution_phenotypes"])))
            archive.close()
            with SemanticArchive(path) as restarted:
                self.assertEqual(restarted.document, document)

    def test_novelty_and_local_neighbors_are_deterministic(self):
        with tempfile.TemporaryDirectory() as directory:
            archive, programs = self.populate(Path(directory) / "archive.json")
            known_id = archive.get_evaluation(programs[0].semantic_id, "sentinel-context")["execution_phenotype_id"]
            known = archive.document["execution_phenotypes"][known_id]
            self.assertEqual(archive.novelty(known)[0], 0)
            changed = json.loads(json.dumps(known))
            changed["cases"][0]["specialist_duration"] += 1
            neighbors = archive.nearest_phenotypes(changed, limit=2)
            self.assertEqual((neighbors[0].phenotype_id, neighbors[0].distance), (known_id, 1))
            self.assertEqual(neighbors, archive.nearest_phenotypes(changed, limit=2))
            self.assertIn(programs[0].semantic_id, neighbors[0].program_ids)

    def test_panel_evidence_preserves_build_and_downstream_summaries(self):
        build = PanelEvidence("BUILD", "build-panel", (
            CaseOutcome("a", -5, 3, 1, 0, "member-a", "GENERAL"),
            CaseOutcome("b", 5, -2, 2, 1, "member-b", "LAND_INVESTMENT"),
        ))
        dev = PanelEvidence("INTERNAL_DEV", "dev-panel", (CaseOutcome("d", 1, -1),), False)
        self.assertEqual((build.to_document()["loss_to_win"], build.to_document()["win_to_loss"]), (1, 1))
        self.assertEqual(PanelEvidence.from_document(build.to_document()), build)
        self.assertFalse(dev.to_document()["passed"])
        with self.assertRaises(ValueError):
            PanelEvidence("SEALED", "sealed", (CaseOutcome("x", 0, 0),))

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "archive.json"
            archive = SemanticArchive(path).open()
            p = program("OWN_MONEY")
            record_sentinel_batch(
                archive, (p,), self.panel, (result(), result()), self.baseline,
                context_id="ctx", generation=0,
            )
            archive.update_evidence(p.semantic_id, "ctx", build)
            archive.update_evidence(p.semantic_id, "ctx", dev, semantic_mismatch=True)
            stored = archive.get_evaluation(p.semantic_id, "ctx")
            self.assertEqual(stored["build_evidence"], build.to_document())
            self.assertEqual(stored["downstream_evidence"], [dev.to_document()])
            self.assertIn(FailureReason.BUILD_REGRESSION.value, stored["failure_reasons"])
            self.assertIn(FailureReason.DEV_REGRESSION.value, stored["failure_reasons"])
            self.assertIn(FailureReason.SEMANTIC_MISMATCH.value, stored["failure_reasons"])
            archive.close()
            with SemanticArchive(path) as restarted:
                self.assertEqual(restarted.get_evaluation(p.semantic_id, "ctx"), stored)

    def test_failure_evidence_and_atomic_write_failure_are_restart_safe(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "archive.json"
            archive = SemanticArchive(path).open()
            before = path.read_bytes()
            p = program("OWN_MONEY")
            with mock.patch("kaggriculture_bonsai.search.aspe.memory.os.replace", side_effect=OSError("power loss")):
                with self.assertRaises(OSError):
                    record_sentinel_batch(
                        archive, (p,), self.panel, (result(), result()), self.baseline,
                        context_id="ctx", generation=0,
                    )
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(archive.document["evaluations"], {})
            self.assertFalse(path.with_name(".archive.json.tmp").exists())
            archive.close()
            with SemanticArchive(path) as restarted:
                self.assertEqual(restarted.document["evaluations"], {})
                faulted = replace(result(), fault=FaultCode.INVALID_FEATURE)
                record_sentinel_batch(
                    restarted, (p,), self.panel, (faulted, result()), self.baseline,
                    context_id="fault-context", generation=0,
                )
                stored = restarted.get_evaluation(p.semantic_id, "fault-context")
                self.assertEqual(stored["semantic_fault"], FaultCode.INVALID_FEATURE.name)
                self.assertIn(FailureReason.SEMANTIC_FAULT.value, stored["failure_reasons"])
            with SemanticArchive(path) as restarted_again:
                self.assertEqual(
                    restarted_again.get_evaluation(p.semantic_id, "fault-context")["semantic_fault"],
                    FaultCode.INVALID_FEATURE.name,
                )

    def test_archive_bound_and_corrupt_restart_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "archive.json"
            archive = SemanticArchive(path, max_evaluations=1).open()
            record_sentinel_batch(
                archive, (program("OWN_MONEY"),), self.panel, (result(), result()), self.baseline,
                context_id="a", generation=0,
            )
            with self.assertRaises(OverflowError):
                record_sentinel_batch(
                    archive, (program("OWN_LAND"),), self.panel, (result(trace=2), result(trace=2)), self.baseline,
                    context_id="b", generation=0,
                )
            archive.close()
            document = json.loads(path.read_text())
            document["schema_version"] = 999
            path.write_text(json.dumps(document))
            with self.assertRaises(ValueError):
                SemanticArchive(path).open()


if __name__ == "__main__":
    unittest.main()
