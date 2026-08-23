from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tools.bkg4_search.edit_search.fitness import StateEvaluation
from tools.bkg4_search.edit_search.panels import CaseRecord
from tools.bkg4_search.invariant_factory import (
    SpecialistFactory,
    compose_gated_specialists,
    load_offline_specialist_bank,
    run_specialist_funnel,
    run_proposal_funnel,
    specialist_corpus_records,
    split_specialist_panels,
)
from tools.bkg4_search.generate_invariant_specialists import combine_offline_banks
from tools.bkg4_search.generate_invariant_specialists import _yield_metrics
from tools.bkg4_search.invariant_module import AbstractAction, GuardAtom, InvariantRule, InvariantStrategicModule, LiftedGuard, ModuleSketch
from tools.bkg4_search.invariant_search import InvariantSearchBackend
from tools.bkg4_search.invariant_schema import validate_specialist_corpus


class HashEvaluator:
    def evaluate(self, modules, panel, *, work_dir, panel_id):
        out = {}
        for mi, module in enumerate(modules):
            base = 1000
            values = []
            mode = int(module.semantic_id()[:8], 16) % 5
            for case in panel:
                if mi == 0:
                    values.append(base)
                else:
                    cash = case.family == "cash"
                    delta = {
                        0: 0,
                        1: 3000 if cash else -5000,
                        2: 2000,
                        3: -3000,
                        4: 500 if cash else -5000,
                    }[mode]
                    values.append(base + delta)
            out[module.semantic_id()] = StateEvaluation(
                module.semantic_id(), panel_id, tuple(case.case_id for case in panel), tuple(values),
                tuple(0 if mi == 0 or mode == 0 else 1 for _ in panel),
                tuple(0 if mi == 0 or mode == 0 else 2 for _ in panel),
                tuple(mi + 1 for _ in panel),
            )
        return out


def _route(feature: str, cmp: str, value: int, route: int) -> InvariantStrategicModule:
    return InvariantStrategicModule((InvariantRule(
        ModuleSketch("route", "INSTANT", "ROUTE_MODE"), "GLOBAL",
        LiftedGuard((GuardAtom(feature, cmp, value),)),
        AbstractAction("ROUTE_OVERRIDE", route_id=route),
    ),))


class InvariantFactoryTests(unittest.TestCase):
    def test_yield_metrics_returns_structured_numbers_for_empty_and_nonempty_input(self):
        from types import SimpleNamespace

        empty = _yield_metrics(())
        self.assertIsInstance(empty, dict)
        self.assertEqual(empty["candidates"], 0)
        self.assertEqual(empty["valid_rate"], 0.0)
        rows = (
            SimpleNamespace(action_change_count=1, useful_groups=2, tier="ELITE_SPECIALIST", production_safe=False, l2w_groups=1),
            SimpleNamespace(action_change_count=0, useful_groups=0, tier="INERT", production_safe=False, l2w_groups=0),
        )
        metrics = _yield_metrics(rows)
        self.assertEqual(metrics["candidates"], 2)
        self.assertEqual(metrics["behavior_changing"], 1)
        self.assertEqual(metrics["elite_specialist"], 1)
        self.assertEqual(metrics["inert"], 1)
        self.assertIsInstance(metrics["behavior_changing_rate"], float)

    def test_factory_fills_large_unique_batches_with_real_skeleton_diversity(self):
        backend = InvariantSearchBackend(route_ids=(29, 64, 70, 211, 212), evaluator=HashEvaluator())
        factory = SpecialistFactory(backend, root_controller="parent")
        batch = factory.propose(count=2000, seed=101)
        self.assertEqual(len(batch.modules), 2000)
        self.assertEqual(len({module.semantic_id() for module in batch.modules}), 2000)
        self.assertGreaterEqual(batch.diversity.structural_skeletons, 300)
        self.assertLess(batch.diversity.top1_skeleton_share, 0.10)
        self.assertGreater(batch.proposals_per_second, 100)

    def test_recursive_factory_uses_specialist_parent_lineage(self):
        backend = InvariantSearchBackend(route_ids=(70, 212), evaluator=HashEvaluator())
        factory = SpecialistFactory(backend, root_controller="parent")
        panel = tuple(
            CaseRecord(f"c{i}", f"t{i}", 0, i + 1, "cash" if i < 2 else "late", role="discovery")
            for i in range(4)
        )
        first = factory.propose(count=64, seed=3)
        factory.screen(first, panel, work_dir="unused", panel_id="screen")
        recursive = factory.propose(count=32, seed=5, recursive=True)
        self.assertTrue(any(lineage.depth > 0 for lineage in recursive.lineages))
        self.assertTrue(all(lineage.root_controller == "parent" for lineage in recursive.lineages))

    def test_recursive_factory_can_refine_a_screened_child_to_real_depth_two(self):
        backend = InvariantSearchBackend(route_ids=(29, 64, 70, 211, 212), evaluator=HashEvaluator())
        factory = SpecialistFactory(backend, root_controller="parent")
        panel = tuple(
            CaseRecord(f"c{i}", f"t{i}", 0, i + 1, "cash" if i < 4 else "late", role="discovery")
            for i in range(8)
        )
        roots = factory.propose(count=128, seed=301)
        factory.screen(roots, panel, work_dir="unused", panel_id="roots")
        depth_one = factory.propose(count=32, seed=303, recursive=True)
        factory.screen(depth_one, panel, work_dir="unused", panel_id="depth-one")
        depth_two = factory.propose(count=32, seed=305, recursive=True)
        descendants = [lineage for lineage in depth_two.lineages if lineage.depth >= 2]
        self.assertTrue(descendants)
        for lineage in descendants:
            self.assertIsNotNone(lineage.parent_semantic_id)
            self.assertNotEqual(lineage.module.semantic_id(), lineage.parent_semantic_id)


    def test_offline_cohort_loader_normalizes_identity_to_label_only_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            tape = Path(tmp) / "tape.json"
            tape.write_text("{}", encoding="utf-8")
            bank = Path(tmp) / "cases.jsonl"
            bank.write_text(json.dumps({
                "tape": str(tape), "seat": 0, "seed": 7, "team": "hard-agent",
                "role": "current_top", "protected": 1,
            }) + "\n", encoding="utf-8")
            loaded = load_offline_specialist_bank(bank, cohort_kind="current_top")
        self.assertEqual(len(loaded.cases), 1)
        self.assertEqual(loaded.cases[0].role, "discovery")
        self.assertEqual(loaded.cases[0].family, loaded.cohort_by_group[loaded.cases[0].tape_path])
        self.assertNotIn("hard-agent", loaded.model_context_json)

    def test_combined_offline_banks_namespace_case_ids_and_keep_cohorts_separate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            banks = []
            for kind in ("current_top", "self_history"):
                tape = root / f"{kind}.json"
                tape.write_text("{}", encoding="utf-8")
                path = root / f"{kind}.jsonl"
                path.write_text(json.dumps({
                    "case_id": "same", "tape": str(tape), "seat": 0, "seed": 1,
                    "team": kind, "role": kind,
                }) + "\n", encoding="utf-8")
                banks.append(load_offline_specialist_bank(path, cohort_kind=kind))
            combined = combine_offline_banks(banks)
        self.assertEqual(len(combined.cases), 2)
        self.assertEqual(len({case.case_id for case in combined.cases}), 2)
        self.assertEqual(len(set(combined.cohort_by_group.values())), 2)

    def test_offline_loader_rejects_certification_or_ranker_holdout(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "certification.jsonl"
            path.write_text(json.dumps({"tape": "x", "seat": 0, "seed": 1, "role": "certification"}) + "\n")
            with self.assertRaisesRegex(ValueError, "certification|holdout"):
                load_offline_specialist_bank(path, cohort_kind="certification")

    def test_disjoint_public_guards_allow_legal_route_specialist_composition(self):
        opening = _route("DAY", "LE", 7, 70)
        endgame = _route("DAY", "GE", 24, 212)
        composed = compose_gated_specialists(opening, endgame)
        self.assertEqual(len(composed.rules), 2)
        composed.validate(production=True, multi_tape_support=2)
        overlapping = _route("DAY", "GE", 5, 211)
        with self.assertRaisesRegex(ValueError, "overlap"):
            compose_gated_specialists(opening, overlapping)

    def test_multi_guard_phase_specialists_can_be_proven_disjoint(self):
        def phase(low, high, route):
            return InvariantStrategicModule((InvariantRule(
                ModuleSketch("route", "INSTANT", "ROUTE_MODE"), "GLOBAL",
                LiftedGuard((GuardAtom("DAY", "GE", low), GuardAtom("DAY", "LE", high), GuardAtom("OWN_MONEY", "GE", 500))),
                AbstractAction("ROUTE_OVERRIDE", route_id=route),
            ),))
        composed = compose_gated_specialists(phase(0, 5, 70), phase(24, 29, 212))
        self.assertEqual(len(composed.rules), 2)
        with self.assertRaisesRegex(ValueError, "overlap"):
            compose_gated_specialists(phase(0, 5, 70), phase(5, 10, 212))

    def test_cuda_funnel_preserves_specialist_classes_and_exact_case_labels(self):
        panel = tuple(
            CaseRecord(f"c{i}", f"t{i}", 0, i + 1, "cash" if i < 4 else "late", role="discovery")
            for i in range(8)
        )
        backend = InvariantSearchBackend(route_ids=(29, 64, 70, 211, 212), evaluator=HashEvaluator())
        factory = SpecialistFactory(backend, root_controller="parent")
        result = run_specialist_funnel(
            factory, screen_panel=panel, label_panel=panel,
            proposal_count=128, retain_count=24, seed=77, work_dir="unused",
        )
        self.assertEqual(len(result.screen_evidence), 128)
        self.assertEqual(len(result.labeled_evidence), 24)
        self.assertEqual(result.exact_games, 129 * 8 + 25 * 8)
        self.assertGreaterEqual(len(result.tier_counts), 3)
        self.assertTrue(all(len(rows) == 8 for rows in result.case_rows.values()))
        records = specialist_corpus_records(result, evaluator_abi="B4IC1/v1", feature_registry=1)
        self.assertIn("mean_delta", records[0]["target_labels"])
        self.assertIn("margin_target", records[0]["target_labels"])
        self.assertIn("w2l", records[0]["target_labels"])
        report = validate_specialist_corpus(records)
        self.assertEqual(report["rows"], 24)
        self.assertEqual(report["unique_semantics"], 24)

    def test_factory_exports_real_cegis_repairs_and_structural_ablations(self):
        panel = tuple(
            CaseRecord(f"c{i}", f"t{i}", 0, i + 1, "cash" if i < 4 else "late", role="discovery")
            for i in range(8)
        )
        backend = InvariantSearchBackend(route_ids=(29, 64, 70, 211, 212), evaluator=HashEvaluator())
        backend.ladder.level = 6
        factory = SpecialistFactory(backend, root_controller="parent")
        batch = factory.propose(count=128, seed=81)
        factory.screen(batch, panel, work_dir="unused", panel_id="screen")
        repairs = factory.cegis_repairs(limit=16)
        ablations = factory.structural_ablations(limit=16)
        self.assertTrue(repairs.modules)
        self.assertTrue(all(lineage.operator == "cegis_repair" for lineage in repairs.lineages))
        self.assertTrue(ablations.modules)
        self.assertTrue(all(lineage.operator == "structural_ablation" for lineage in ablations.lineages))

    def test_recursive_funnel_exact_labels_child_against_immediate_parent(self):
        panel = tuple(
            CaseRecord(f"c{i}", f"t{i}", 0, i + 1, "cash" if i < 4 else "late", role="discovery")
            for i in range(8)
        )
        backend = InvariantSearchBackend(route_ids=(29, 64, 70, 211, 212), evaluator=HashEvaluator())
        factory = SpecialistFactory(backend, root_controller="parent")
        roots = factory.propose(count=128, seed=91)
        factory.screen(roots, panel, work_dir="unused", panel_id="roots")
        recursive = factory.propose(count=24, seed=93, recursive=True)
        result = run_proposal_funnel(
            factory, proposal=recursive, screen_panel=panel, label_panel=panel,
            retain_count=12, seed=95, work_dir="unused",
        )
        records = specialist_corpus_records(result, evaluator_abi="B4IC1/v1", feature_registry=1)
        self.assertTrue(any(record["target_labels"]["lineage"]["depth"] > 0 for record in records))
        self.assertTrue(all(record["target_labels"]["source_relative_case_outcomes"] for record in records))
        self.assertGreater(result.exact_games, (24 + 1) * 8 + (12 + 1) * 8)
        self.assertTrue(all(
            "behavior_changed_vs_parent" in row
            for record in records for row in record["target_labels"]["source_relative_case_outcomes"]
        ))
        self.assertTrue(any(
            record["target_labels"]["lineage"].get("source_goal_id")
            for record in records
        ))

    def test_specialist_screen_and_label_panels_are_tape_disjoint(self):
        rows = tuple(
            CaseRecord(f"c{i}s{seat}", f"t{i}", seat, i + 1, f"cohort-{i % 3}", role="discovery")
            for i in range(12) for seat in (0, 1)
        )
        screen, label = split_specialist_panels(rows, screen_cases=8, label_cases=12, seed=9)
        self.assertFalse({row.tape_path for row in screen} & {row.tape_path for row in label})
        self.assertEqual(len(screen), 8)
        self.assertEqual(len(label), 12)
        for tape in {row.tape_path for row in (*screen, *label)}:
            self.assertEqual(sum(row.tape_path == tape for row in (*screen, *label)), 2)


if __name__ == "__main__":
    unittest.main()
