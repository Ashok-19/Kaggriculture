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


if __name__ == "__main__":
    unittest.main(verbosity=2)
