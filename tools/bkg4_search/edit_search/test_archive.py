from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tools.bkg4_search.bonsai_editformer.actor_grammar import EditAction
from tools.bkg4_search.module_codec import ConditionGene, ModuleGenome, RuleGene
from tools.bkg4_search.edit_search.schema import EditProgram

ROUTES = (22, 23, 29, 32, 34, 46, 63, 64, 70, 77, 79, 211, 212, 213, 214)


def root() -> ModuleGenome:
    return ModuleGenome(144, (RuleGene((ConditionGene("OWN_MONEY", "GE", 0, 1000),), 22),))


def state_eval(sid: str, margins: list[int], traces: list[int] | None = None):
    from tools.bkg4_search.edit_search.fitness import StateEvaluation
    n = len(margins)
    return StateEvaluation(
        semantic_id=sid,
        panel_id="p",
        case_ids=tuple(f"c{i}" for i in range(n)),
        margins=tuple(margins),
        action_changes=tuple(1 for _ in range(n)),
        activations=tuple(1 for _ in range(n)),
        trace_hashes=tuple(traces or [10 + i for i in range(n)]),
    )


class EditSearchArchiveTests(unittest.TestCase):
    def test_lexicographic_fitness_never_trades_new_loss_for_margin(self):
        from tools.bkg4_search.edit_search.fitness import PairEvaluation, LexicographicFitness
        base = state_eval("base", [100, -10], [1, 2])
        huge_but_unsafe = state_eval("unsafe", [-1, 100000], [3, 4])
        safe_small = state_eval("safe", [101, 1], [5, 6])
        a = LexicographicFitness.from_pair(PairEvaluation.from_states(base, huge_but_unsafe))
        b = LexicographicFitness.from_pair(PairEvaluation.from_states(base, safe_small))
        self.assertLess(b.key(), a.key())
        self.assertEqual(a.new_losses, 1)
        self.assertEqual(b.new_losses, 0)

    def test_production_and_exploration_niches_are_separate_and_deterministic(self):
        from tools.bkg4_search.edit_search.archive import SearchArchive
        from tools.bkg4_search.edit_search.fitness import PairEvaluation
        p = root()
        prog1 = EditProgram.build(p, (EditAction("CHANGE_WINDOW", 0, prefix_step=145),), ROUTES, emitter="systematic", seed=1)
        prog2 = EditProgram.build(p, (EditAction("CHANGE_WINDOW", 0, prefix_step=146),), ROUTES, emitter="systematic", seed=2)
        base = state_eval(p.semantic_id(), [100, -10], [1, 2])
        e1 = state_eval(prog1.child_semantic_id, [110, 5], [3, 4])
        e2 = state_eval(prog2.child_semantic_id, [-1, 500], [5, 6])
        a = SearchArchive.new(root_genome=p, base_identity="base", route_ids=ROUTES, run_id="run", master_seed=7)
        a.add_program(prog1, prog1.reconstruct(p, ROUTES))
        a.add_program(prog2, prog2.reconstruct(p, ROUTES))
        r1 = a.consider(prog1, PairEvaluation.from_states(base, e1))
        r2 = a.consider(prog2, PairEvaluation.from_states(base, e2))
        self.assertTrue(r1.production_inserted)
        self.assertFalse(r2.production_inserted)
        self.assertTrue(r2.exploration_inserted)
        self.assertNotEqual(set(a.production_niches), set(a.exploration_niches))

    def test_checkpoint_resume_preserves_counterexamples_stats_and_canonical_state(self):
        from tools.bkg4_search.edit_search.archive import SearchArchive
        p = root()
        program = EditProgram.build(p, (EditAction("CHANGE_WINDOW", 0, prefix_step=200),), ROUTES, emitter="random_typed", seed=9)
        archive = SearchArchive.new(root_genome=p, base_identity="base", route_ids=ROUTES, run_id="r1", master_seed=99)
        archive.add_program(program, program.reconstruct(p, ROUTES))
        archive.record_counterexample("c9", reason="w2l", severity=10, panel_id="full")
        archive.record_counterexample("c9", reason="w2l", severity=10, panel_id="full")
        archive.note_emitter("random_typed", attempts=5, unique=4, archive_insertions=1, useful_safe=0)
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "archive.json"
            archive.save(path)
            got = SearchArchive.load(path)
        self.assertEqual(got.root_semantic_id, archive.root_semantic_id)
        self.assertEqual(got.states[program.child_semantic_id], archive.states[program.child_semantic_id])
        self.assertEqual(got.hard_cases["c9"]["count"], 2)
        self.assertEqual(got.emitter_stats["random_typed"]["attempts"], 5)
        self.assertEqual(got.programs[program.program_id].program_id, program.program_id)

    def test_exact_elite_evaluations_and_successful_motifs_survive_resume(self):
        from tools.bkg4_search.edit_search.archive import SearchArchive
        from tools.bkg4_search.edit_search.fitness import PairEvaluation
        p = root()
        program = EditProgram.build(p, (EditAction("ADD_CONDITION", 3, feature="OPP_MONEY", cmp="GE", value=4000),), ROUTES, emitter="systematic", seed=12)
        child = program.reconstruct(p, ROUTES)
        archive = SearchArchive.new(root_genome=p, base_identity="base", route_ids=ROUTES, run_id="r2", master_seed=5)
        archive.add_program(program, child)
        base = state_eval(p.semantic_id(), [100, -10], [1, 2])
        exact = state_eval(child.semantic_id(), [120, 5], [3, 4])
        pair = PairEvaluation.from_states(base, exact)
        archive.consider(program, pair)
        archive.record_state_evaluation(base)
        archive.record_state_evaluation(exact)
        motif_id = archive.observe_motif(program, pair)
        self.assertIsNotNone(motif_id)
        self.assertEqual(archive.state_evaluation(child.semantic_id(), "p").margins, (120, 5))
        self.assertGreaterEqual(archive.motifs[motif_id]["safe_useful_hits"], 1)
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "archive.json"
            archive.save(path)
            got = SearchArchive.load(path)
        self.assertEqual(got.state_evaluation(child.semantic_id(), "p").margins, (120, 5))
        self.assertEqual(got.motifs[motif_id]["safe_useful_hits"], archive.motifs[motif_id]["safe_useful_hits"])


    def test_production_safety_covers_all_outcome_transitions_and_tie_to_loss(self):
        from tools.bkg4_search.edit_search.fitness import LexicographicFitness, PairEvaluation, StateEvaluation
        transitions = [
            (10, -1, False, "win-loss"),
            (0, -1, False, "tie-loss"),
            (-10, -20, True, "loss-worse-loss"),
            (-10, 0, True, "loss-tie"),
            (-10, 10, True, "loss-win"),
            (10, 0, False, "win-tie"),
            (0, 10, True, "tie-win"),
            (10, 5, True, "win-smaller-win"),
        ]
        for parent_margin, child_margin, expected_safe, label in transitions:
            with self.subTest(label=label):
                p = StateEvaluation("p", "full", ("c",), (parent_margin,), (0,), (0,), (1,))
                c = StateEvaluation("c", "full", ("c",), (child_margin,), (1,), (1,), (2,))
                pair = PairEvaluation.from_states(p, c, protected={"c": True})
                fit = LexicographicFitness.from_pair(pair)
                self.assertEqual(fit.strict_safe, expected_safe)
                self.assertEqual(bool(pair.aggregate["strict_safe"]), expected_safe)
        p = StateEvaluation("p", "full", ("bad", "gain"), (0, 1), (0, 0), (0, 0), (1, 1))
        c = StateEvaluation("c", "full", ("bad", "gain"), (-1, 1000000), (1, 1), (1, 1), (2, 2))
        bad = PairEvaluation.from_states(p, c, protected={"bad": True, "gain": True})
        self.assertEqual(bad.aggregate["new_loss_count"], 1)
        self.assertEqual(LexicographicFitness.from_pair(bad).new_losses, 1)
        safe_c = StateEvaluation("s", "full", ("bad", "gain"), (0, 2), (0, 1), (0, 1), (1, 3))
        safe = PairEvaluation.from_states(p, safe_c, protected={"bad": True, "gain": True})
        self.assertLess(LexicographicFitness.from_pair(safe).key(), LexicographicFitness.from_pair(bad).key())

    def test_archive_defense_in_depth_rejects_tie_to_loss_from_production(self):
        from tools.bkg4_search.bonsai_editformer.actor_grammar import EditAction
        from tools.bkg4_search.edit_search.archive import SearchArchive
        from tools.bkg4_search.edit_search.fitness import PairEvaluation, StateEvaluation
        from tools.bkg4_search.edit_search.schema import EditProgram
        p = root(); archive = SearchArchive.new(root_genome=p, base_identity="b", route_ids=ROUTES, run_id="r", master_seed=1)
        program = EditProgram.build(p, (EditAction("CHANGE_WINDOW", 0, prefix_step=200),), ROUTES, emitter="systematic", seed=1)
        child = program.reconstruct(p, ROUTES); archive.add_program(program, child)
        pe = StateEvaluation(p.semantic_id(), "full", ("c",), (0,), (0,), (0,), (10,))
        ce = StateEvaluation(child.semantic_id(), "full", ("c",), (-1,), (1,), (1,), (11,))
        pair = PairEvaluation.from_states(pe, ce, protected={"c": True})
        result = archive.consider(program, pair)
        self.assertFalse(result.production_inserted)
        self.assertEqual(len(archive.production_niches), 0)

    def test_convergent_programs_do_not_inflate_policy_state_qd_or_novelty(self):
        from tools.bkg4_search.bonsai_editformer.actor_grammar import EditAction
        from tools.bkg4_search.edit_search.archive import SearchArchive
        from tools.bkg4_search.edit_search.fitness import PairEvaluation, StateEvaluation
        from tools.bkg4_search.edit_search.schema import EditProgram
        p = root(); archive = SearchArchive.new(root_genome=p, base_identity="b", route_ids=ROUTES, run_id="r", master_seed=1)
        direct = EditProgram.build(p, (EditAction("CHANGE_WINDOW", 0, prefix_step=216),), ROUTES, emitter="systematic", seed=1)
        via = EditProgram.build(p, (EditAction("CHANGE_WINDOW", 0, prefix_step=200), EditAction("CHANGE_WINDOW", 0, prefix_step=216)), ROUTES, emitter="lns", seed=2)
        self.assertNotEqual(direct.program_id, via.program_id); self.assertEqual(direct.child_semantic_id, via.child_semantic_id)
        child = direct.reconstruct(p, ROUTES); archive.add_program(direct, child); archive.add_program(via, child)
        pe = StateEvaluation(p.semantic_id(), "full", ("c",), (1,), (0,), (0,), (10,))
        ce = StateEvaluation(child.semantic_id(), "full", ("c",), (2,), (1,), (1,), (11,))
        pair = PairEvaluation.from_states(pe, ce, protected={"c": True})
        archive.consider(direct, pair); archive.consider(via, pair)
        self.assertEqual(len(archive.production_niches), 1)
        self.assertEqual(len(archive.exploration_niches), 1)
        self.assertEqual(len([x for x in archive.novelty_archive if x["child_semantic_id"] == child.semantic_id()]), 1)

    def test_archive_load_fails_closed_on_dangling_qd_reference(self):
        from tools.bkg4_search.bonsai_editformer.actor_grammar import EditAction
        from tools.bkg4_search.edit_search.archive import SearchArchive
        from tools.bkg4_search.edit_search.fitness import PairEvaluation, StateEvaluation
        from tools.bkg4_search.edit_search.schema import EditProgram
        import json, tempfile
        from pathlib import Path
        p = root(); archive = SearchArchive.new(root_genome=p, base_identity="b", route_ids=ROUTES, run_id="r", master_seed=1)
        program = EditProgram.build(p, (EditAction("CHANGE_WINDOW", 0, prefix_step=200),), ROUTES, emitter="systematic", seed=1)
        child = program.reconstruct(p, ROUTES); archive.add_program(program, child)
        pe = StateEvaluation(p.semantic_id(), "full", ("c",), (1,), (0,), (0,), (10,)); ce = StateEvaluation(child.semantic_id(), "full", ("c",), (2,), (1,), (1,), (11,))
        archive.consider(program, PairEvaluation.from_states(pe, ce, protected={"c": True}))
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "a.json"; archive.save(path); raw = json.loads(path.read_text())
            key = next(iter(raw["production_niches"])); raw["production_niches"][key]["child_semantic_id"] = "deadbeef"
            path.write_text(json.dumps(raw))
            with self.assertRaises(ValueError):
                SearchArchive.load(path)

if __name__ == "__main__":
    unittest.main(verbosity=2)
