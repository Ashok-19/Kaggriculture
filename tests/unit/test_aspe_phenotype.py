from __future__ import annotations

import struct
import json
import unittest
from dataclasses import replace
from pathlib import Path

from kaggriculture_bonsai.search.aspe import (
    ExecutionPhenotype,
    ScreeningClassification,
    SpecialistProgram,
    StrategicEffectPhenotype,
    classify_sentinel_results,
    select_build_sentinel,
)
from kaggriculture_bonsai.search.aspe.phenotype import SentinelPanel
from kaggriculture_bonsai.search.checkpoints import Lane
from kaggriculture_bonsai.search.evaluator import (
    CompletionReason,
    DeviceResult,
    FaultCode,
    TerminationReason,
    decode_device_results,
)
from kaggriculture_bonsai.search.ir import ActionSource, Phase, PublicCondition, Termination, TerminationType


def specialist(feature: str = "OWN_MONEY", value: int = 0) -> SpecialistProgram:
    return SpecialistProgram(
        lane=Lane.GENERAL, activation_step=24,
        initiation_clauses=((PublicCondition(feature, "GE", value=value),),),
        phases=(Phase(ActionSource.PARENT, (), Termination(TerminationType.FIXED_CAP, 2)),),
        max_total_steps=2,
    )


def row(*, activated: bool = True, own: int = 100, opp: int = 80, duration: int = 2) -> DeviceResult:
    return DeviceResult(
        own, opp, 1 if activated else 0, FaultCode.NONE, activated, 0,
        TerminationReason.FIXED_CAP if activated else TerminationReason.NONE,
        activated, "11" * 32, 24 if activated else -1, 25 if activated else -1,
        1 if activated else 0, duration if activated else 0, 1 if activated else 0,
        1 if activated else 0, 0, CompletionReason.FINISHED if activated else CompletionReason.NONE,
        1 if activated else 0, 0, 0, 0,
        (5, 0, 0, 0, 0, 0, 0, 0, 0, 0) if activated else (0,) * 10,
    )


class AspePhenotypeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.panel = select_build_sentinel(Path("workspace/aspe/baseline/manifest.json"))

    def test_sentinel_is_deterministic_and_factor_balanced(self):
        again = select_build_sentinel(Path("workspace/aspe/baseline/manifest.json"))
        self.assertEqual(self.panel, again)
        self.assertEqual(self.panel.identity, "aab5fac507011fb2929e4b1b3837a58a8e386e936a553257739bf16f86c2d4bb")
        self.assertEqual(len(self.panel.cases), 16)
        self.assertEqual({case.seed for case in self.panel.cases}, {1103, 2207, 3301})
        self.assertEqual({case.seat for case in self.panel.cases}, {0, 1})
        self.assertEqual(len({case.member for case in self.panel.cases}), 8)
        self.assertEqual(len({case.lane for case in self.panel.cases}), 8)
        self.assertEqual({(case.member, case.seat) for case in self.panel.cases}, {
            (member, seat) for member in {case.member for case in self.panel.cases} for seat in (0, 1)
        })
        tracked = json.loads(Path("workspace/aspe/sentinel/manifest.json").read_text())
        self.assertEqual(tracked["identity"], self.panel.identity)
        self.assertEqual(tracked["cases"], [case.to_document() for case in self.panel.cases])

    def test_telemetry_v2_decoding(self):
        record = struct.Struct("<iiHHBBBBHHHHHHHHQBBBBhhhhhhhhhh32s")
        payload = record.pack(
            120, 90, 3, 0, 1, 2, 5, 1, 24, 30, 2, 7, 0o123, 1, 2, 3,
            0x123456789ABCDEF0, 0b111, 0b0110, 2, 0, *range(10), bytes.fromhex("ab" * 32),
        )
        data = struct.pack("<8sIII", b"RLRMBR1\0", 2, 1, 1) + payload
        candidates, cases, rows = decode_device_results(data)
        self.assertEqual((candidates, cases), (1, 1))
        decoded = rows[0]
        self.assertEqual((decoded.first_activation_step, decoded.last_activation_step), (24, 30))
        self.assertEqual((decoded.activation_count, decoded.specialist_duration), (2, 7))
        self.assertEqual((decoded.phase_path, decoded.phase_usage, decoded.branch_path), (0o123, 0b111, 0b0110))
        self.assertEqual(decoded.completion_reason, CompletionReason.ABORT)
        self.assertEqual(decoded.action_trace_hash, 0x123456789ABCDEF0)
        self.assertEqual(decoded.state_deltas, tuple(range(10)))

    def test_versioned_phenotype_identities_are_deterministic(self):
        cases = self.panel.cases[:2]
        rows = (row(), row(duration=3))
        execution = ExecutionPhenotype(tuple(case.identity for case in cases), rows)
        effect = StrategicEffectPhenotype(cases, rows, (row(own=90, opp=100),) * 2)
        self.assertEqual(execution.identity, ExecutionPhenotype(execution.case_ids, rows).identity)
        self.assertEqual(effect.identity, StrategicEffectPhenotype(cases, rows, (row(own=90, opp=100),) * 2).identity)
        self.assertNotEqual(execution.identity, effect.identity)
        self.assertEqual(execution.to_document()["schema_version"], 1)
        self.assertEqual(effect.to_document()["schema_version"], 1)
        different_body = (replace(rows[0], action_trace_hash=7), rows[1])
        self.assertNotEqual(execution.identity, ExecutionPhenotype(execution.case_ids, different_body).identity)

    def test_structural_execution_and_effect_duplicates_are_distinct(self):
        panel = SentinelPanel(self.panel.source_panel_identity, self.panel.cases[:2])
        a = specialist("OWN_MONEY", 0)
        structural_copy = specialist("OWN_MONEY", 0)
        same_execution = specialist("OPP_MONEY", 0)
        same_effect = specialist("OWN_LAND", 0)
        programs = (a, structural_copy, same_execution, same_effect)
        a_rows = (row(), row())
        different_execution = (row(duration=3), row(duration=3))
        rows = (*a_rows, *a_rows, *a_rows, *different_execution)
        baseline = (row(own=90, opp=100), row(own=90, opp=100))
        classified = classify_sentinel_results(programs, panel, rows, baseline)
        self.assertEqual(classified[0].classification, ScreeningClassification.NOVEL_OR_INFORMATIVE)
        self.assertEqual((classified[1].classification, classified[1].duplicate_level), (ScreeningClassification.PHENOTYPE_DUPLICATE, "STRUCTURAL"))
        self.assertEqual((classified[2].classification, classified[2].duplicate_level), (ScreeningClassification.PHENOTYPE_DUPLICATE, "EXECUTION"))
        self.assertEqual((classified[3].classification, classified[3].duplicate_level), (ScreeningClassification.EFFECT_DUPLICATE, "STRATEGIC_EFFECT"))

    def test_zero_activation_and_only_obvious_harm_are_screened(self):
        panel = SentinelPanel(self.panel.source_panel_identity, self.panel.cases[:2])
        baseline = (row(own=100, opp=80), row(own=100, opp=80))
        zero = (row(activated=False), row(activated=False))
        harmful = (row(own=70, opp=90), row(own=75, opp=90))
        useful_or_uncertain = (row(own=110, opp=80, duration=3), row(own=90, opp=80, duration=3))
        result = classify_sentinel_results(
            (specialist("OWN_MONEY"), specialist("OWN_LAND"), specialist("OWN_UNITS")),
            panel, (*zero, *harmful, *useful_or_uncertain), baseline,
        )
        self.assertEqual(result[0].classification, ScreeningClassification.ZERO_ACTIVATION)
        self.assertEqual(result[1].classification, ScreeningClassification.CLEARLY_HARMFUL)
        self.assertEqual(result[2].classification, ScreeningClassification.NOVEL_OR_INFORMATIVE)


if __name__ == "__main__":
    unittest.main()
