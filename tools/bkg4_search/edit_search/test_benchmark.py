from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tools.bkg4_search.edit_search.fitness import StateEvaluation
from tools.bkg4_search.edit_search.panels import CaseBank, CaseRecord
from tools.bkg4_search.module_codec import ModuleGenome

ROUTES = (22, 23, 29, 32, 34, 46, 63, 64, 70, 77, 79, 211, 212, 213, 214)
OBS = [
    {"feature": "OWN_MONEY", "index": 0, "value": x}
    for x in (500, 1000, 2500, 5000, 7500)
] + [
    {"feature": "MARKET_PRICE", "index": 3, "value": x}
    for x in (80, 120, 160, 200)
]


class DeterministicExactOracle:
    """Orchestration-only exact-vector oracle; production benchmark uses B4EC1."""

    def __init__(self):
        self.evaluated_children = 0

    def evaluate(self, genomes, panel, *, work_dir, panel_id):
        out = {}
        for genome in genomes:
            sid = genome.semantic_id()
            margins = []
            changes = []
            activations = []
            traces = []
            for i, case in enumerate(panel):
                # Deterministic landscape with specialists, unsafe regressions and plateaus.
                base = (-30 if i == 0 else 120 + i * 10)
                rule_bonus = 38 * len(genome.rules)
                route_bonus = sum((r.route % 11) - 5 for r in genome.rules)
                prefix_penalty = abs(genome.prefix_step - 180) // 12
                specialist = 65 if genome.rules and i == (genome.rules[0].route % len(panel)) else 0
                margin = base + rule_bonus + route_bonus + specialist - prefix_penalty
                margins.append(int(margin))
                changes.append(0 if not genome.rules else 2 + len(genome.rules))
                activations.append(0 if not genome.rules else 5 + len(genome.rules))
                traces.append(100000 + i if not genome.rules else int(sid[:12], 16) + i)
            out[sid] = StateEvaluation(
                sid, panel_id, tuple(c.case_id for c in panel), tuple(margins),
                tuple(changes), tuple(activations), tuple(traces),
            )
        self.evaluated_children += len(genomes)
        return out


class PlateauExactOracle:
    """No behavioral/objective progress so anti-stagnation must eventually escalate."""

    def evaluate(self, genomes, panel, *, work_dir, panel_id):
        return {
            g.semantic_id(): StateEvaluation(
                g.semantic_id(), panel_id, tuple(c.case_id for c in panel),
                tuple(100 for _ in panel), tuple(0 for _ in panel), tuple(0 for _ in panel),
                tuple(777 + i for i in range(len(panel))),
            )
            for g in genomes
        }


def bank():
    return CaseBank([
        CaseRecord("c0", "/tmp/t0", 0, 1, "f0", baseline_margin=-30),
        CaseRecord("c1", "/tmp/t1", 1, 2, "f1", baseline_margin=130),
        CaseRecord("c2", "/tmp/t2", 0, 3, "f2", baseline_margin=140),
        CaseRecord("c3", "/tmp/t3", 1, 4, "f3", baseline_margin=150),
    ])


class EditSearchBenchmarkTests(unittest.TestCase):
    def test_equal_budget_benchmark_reports_required_search_and_safety_metrics(self):
        from tools.bkg4_search.edit_search.benchmark import benchmark_proposers

        evaluator = DeterministicExactOracle()
        with tempfile.TemporaryDirectory() as td:
            report = benchmark_proposers(
                root_genome=ModuleGenome(0, ()), route_ids=ROUTES, observations=OBS,
                panel=bank().rows, evaluator=evaluator, candidate_budget=24, seed=91,
                work_dir=Path(td),
            )
        self.assertEqual(set(report["proposers"]), {"old_shallow", "random_typed", "fixed_mutation", "permanent_recursive"})
        for row in report["proposers"].values():
            self.assertEqual(row["evaluated_candidates"], 24)
            self.assertEqual(row["exact_candidate_case_evaluations"], 96)
            for key in (
                "unique_valid_proposals", "duplicate_rate", "behavior_changing_rate", "operation_coverage",
                "target_depth_coverage", "parent_rule_count_coverage", "qd_niches", "safe_useful_per_1000",
                "target_loss_improving_per_1000", "l2w_per_1000", "strict_safe_l2w_per_1000",
                "w2l_rate", "severe_regression_rate", "best_target_improvement", "best_worst_margin_improvement",
                "useful_diversity", "max_generation_depth",
            ):
                self.assertIn(key, row)
        self.assertGreater(report["proposers"]["permanent_recursive"]["max_generation_depth"], 1)
        self.assertGreater(
            report["proposers"]["permanent_recursive"]["parent_rule_count_coverage"],
            report["proposers"]["fixed_mutation"]["parent_rule_count_coverage"],
        )

    def test_equal_budget_benchmark_enforces_same_root_parent_for_every_arm(self):
        from tools.bkg4_search.edit_search.benchmark import benchmark_proposers
        root = ModuleGenome(37, ())
        with tempfile.TemporaryDirectory() as td:
            report = benchmark_proposers(root_genome=root, route_ids=ROUTES, observations=OBS, panel=bank().rows,
                evaluator=DeterministicExactOracle(), candidate_budget=16, seed=101, work_dir=Path(td))
        self.assertTrue(report["same_root_parent"])
        self.assertEqual(set(report["arm_root_semantic_ids"].values()), {root.semantic_id()})
        self.assertTrue(report["root_truth_shared"])
        for row in report["proposers"].values():
            self.assertIn("source_parent_count", row)
            self.assertIn("new_loss_rate", row)

    def test_mixed_benchmark_exercises_goal_aware_contrastive_search(self):
        from tools.bkg4_search.edit_search.benchmark import benchmark_proposers
        from tools.bkg4_search.information_state_io import registry_manifest

        registry = registry_manifest(); width = len(registry)
        own_money = next(i for i, (name, args) in enumerate(registry) if name == "OWN_MONEY" and not args)
        rows = []
        for i, margin in enumerate((-30, 130, 140, 150)):
            values = [0] * width; values[own_money] = 100 if i == 0 else 1000
            rows.append(CaseRecord(
                f"c{i}", f"/tmp/t{i}", i % 2, i + 1, f"f{i}", protected=i > 0,
                baseline_margin=margin, context_probe_steps=(264,), public_context_width=width,
                public_context_registry_width=width, public_context_values=tuple(values),
            ))
        with tempfile.TemporaryDirectory() as td:
            report = benchmark_proposers(
                root_genome=ModuleGenome(0, ()), route_ids=ROUTES, observations=OBS,
                panel=rows, evaluator=DeterministicExactOracle(), candidate_budget=24,
                seed=123, work_dir=Path(td),
            )
        permanent = report["proposers"]["permanent_recursive"]
        self.assertIn("emitter_families", permanent)
        self.assertIn("contrastive", permanent["emitter_families"])

    def test_anti_saturation_records_trajectories_and_real_escalation(self):
        from tools.bkg4_search.edit_search.benchmark import run_anti_saturation

        with tempfile.TemporaryDirectory() as td:
            report = run_anti_saturation(
                root_genome=ModuleGenome(0, ()), route_ids=ROUTES, observations=OBS,
                case_bank=bank(), evaluator=PlateauExactOracle(), generations=5,
                candidates_per_generation=20, screen_size=2, final_size=4,
                seed=17, work_dir=Path(td), stagnation_patience=1,
            )
        self.assertEqual(len(report["trajectory"]), 5)
        self.assertTrue(report["escalation_events"])
        levels = [x["level"] for x in report["escalation_events"]]
        self.assertEqual(levels, sorted(levels))
        self.assertGreater(max(levels), 0)
        for row in report["trajectory"]:
            for key in (
                "generation", "requested", "unique", "duplicate_rate", "archive_size", "production_niches",
                "exploration_niches", "novelty_entries", "novelty_mean", "emitter_yield", "escalation_level",
                "source_parent_count", "operation_count",
            ):
                self.assertIn(key, row)
        self.assertTrue(any(row["escalation_level"] > 0 for row in report["trajectory"]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
