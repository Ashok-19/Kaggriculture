"""Phase-6 hierarchical diverse beam contracts."""

from __future__ import annotations

import unittest

from tools.bkg4_search.rlrmb.archive import Archive, InsertResult
from tools.bkg4_search.rlrmb.beam import (
    ALL_MASKS,
    BeamConfig,
    BeamCandidate,
    BatchEvaluation,
    EvaluatedProgram,
    ExpansionOperator,
    SeedSource,
    deduplicate_candidates,
    expand_program,
    generate_seeds,
    run_beam,
)
from tools.bkg4_search.rlrmb.checkpoints import Checkpoint, EventType, Lane
from tools.bkg4_search.rlrmb.effects import EffectBlock, EffectVector
from tools.bkg4_search.rlrmb.ir import ActionFamily, ActionSource, MacroProgram, Phase, Termination, TerminationType
from tools.bkg4_search.rlrmb.motifs import motif_from_actions
from tools.bkg4_search.rlrmb.schema import DatasetRole


PASS = {"farmer": ["PASS"], "hands": [], "market": []}
MARKET = {"farmer": ["PASS"], "hands": [], "market": [["BUY_PRODUCT", "WHEAT", 1]]}
HANDS = {"farmer": ["PASS"], "hands": [["NORTH"]], "market": []}


def cp(step: int = 120, lane: Lane = Lane.MARKET_LIQUIDITY) -> Checkpoint:
    return Checkpoint(step, lane, (EventType.REGRET_PEAK,))


def motifs():
    a = motif_from_actions([MARKET] * 8, (ActionFamily.MARKET,), provenance=("route-a",))
    b = motif_from_actions([HANDS] * 8, (ActionFamily.HANDS,), provenance=("route-b",))
    c = motif_from_actions([{**MARKET, "farmer": ["NORTH"], "hands": [["SOUTH"]]}] * 8,
                           tuple(ActionFamily), provenance=("route-c",))
    return {item.semantic_id: item for item in (a, b, c)}


def phase(motif_id: str, family: ActionFamily = ActionFamily.MARKET, cap: int = 4) -> Phase:
    return Phase(ActionSource.SEMANTIC_SPLICE, (family,), Termination(TerminationType.FIXED_CAP, cap), motif_id)


def program() -> MacroProgram:
    lib = motifs()
    market = next(item for item in lib.values() if item.supported_families == (ActionFamily.MARKET,))
    return MacroProgram(cp().lane, cp(), (phase(market.semantic_id),), 4)


def effect(deltas: tuple[int, ...], baseline: tuple[int, ...] | None = None) -> EffectVector:
    baseline = baseline or tuple(-100 for _ in deltas)
    return EffectVector((EffectBlock(DatasetRole.TARGET, "panel", baseline, deltas, tuple(1 for _ in deltas)),))


class FakeEvaluator:
    def __init__(self, same: bool = False):
        self.case_count = 2
        self.calls = 0
        self.seen: list[tuple[str, ...]] = []
        self.same = same

    def projected_games(self, count):
        return count * self.case_count

    def __call__(self, programs, motif_library):
        self.calls += 1
        self.seen.append(tuple(item.semantic_id for item in programs))
        rows = []
        for index, item in enumerate(programs):
            deltas = (10, 10) if self.same else ((100, -100) if len(item.phases) == 1 else (0, 20))
            rows.append(EvaluatedProgram(item, effect(deltas), True, None))
        return BatchEvaluation(tuple(rows), cases=2, games=2 * len(programs), launches=1, host_seconds=0.01, cuda_ms=1.0)


class BeamTests(unittest.TestCase):
    def test_config_rejects_invalid_limits_and_operators(self):
        for kwargs in ({"max_depth": 0}, {"max_phases": 5}, {"cuda_game_budget": 0},
                       {"max_candidates_per_expansion": 0}, {"stagnation_rounds": 0}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                BeamConfig(**kwargs)
        with self.assertRaises(ValueError):
            BeamConfig(allowed_operators=("UNKNOWN",))

    def test_deterministic_seed_order_ids_and_provenance_independence(self):
        first = generate_seeds((cp(),), motifs(), BeamConfig())
        changed = {
            key: motif_from_actions([MARKET] * 8, (ActionFamily.MARKET,), provenance=("other-file",))
            if value.supported_families == (ActionFamily.MARKET,) else value
            for key, value in motifs().items()
        }
        second = generate_seeds((cp(),), changed, BeamConfig())
        self.assertEqual([item.program.semantic_id for item in first], [item.program.semantic_id for item in second])
        self.assertEqual([item.seed_source.value for item in first], sorted(item.seed_source.value for item in first))
        self.assertEqual({item.seed_source for item in first}, set(SeedSource))

    def test_insert_phase_and_four_phase_cap(self):
        base = program()
        generated = expand_program(base, ExpansionOperator.INSERT_PHASE, motifs(), (cp(124),))
        self.assertTrue(any(len(item.phases) == 2 and item.phases[0] == base.phases[0] for item in generated))
        four = MacroProgram(base.lane, base.initiation, base.phases * 4, 16)
        self.assertEqual(expand_program(four, ExpansionOperator.INSERT_PHASE, motifs(), (cp(124),)), ())

    def test_replace_source_changes_only_requested_phase(self):
        base = program()
        generated = expand_program(base, ExpansionOperator.REPLACE_PHASE_SOURCE, motifs(), (cp(124),))
        self.assertTrue(generated)
        self.assertEqual(generated[0].phases[0].termination, base.phases[0].termination)
        self.assertNotEqual(generated[0].phases[0].motif_id, base.phases[0].motif_id)

    def test_shift_truncate_and_extend_use_adjacent_events(self):
        base = program()
        events = (cp(122), cp(126))
        shifted = expand_program(base, ExpansionOperator.SHIFT_PHASE_BOUNDARY_TO_ADJACENT_EVENT, motifs(), events)
        self.assertEqual({item.phases[0].termination.max_steps for item in shifted}, {2, 6})
        truncated = expand_program(base, ExpansionOperator.TRUNCATE_AT_EVENT, motifs(), events)
        extended = expand_program(base, ExpansionOperator.EXTEND_TO_EVENT, motifs(), events)
        self.assertEqual([item.phases[0].termination.max_steps for item in truncated], [2])
        self.assertEqual([item.phases[0].termination.max_steps for item in extended], [6])

    def test_change_mask_only_changes_mask(self):
        lib = motifs()
        full = next(item for item in lib.values() if len(item.supported_families) == 3)
        base = MacroProgram(cp().lane, cp(), (phase(full.semantic_id, ActionFamily.MARKET),), 4)
        generated = expand_program(base, ExpansionOperator.CHANGE_ACTION_FAMILY_MASK, lib, ())
        self.assertTrue(generated)
        for item in generated:
            self.assertEqual(item.phases[0].motif_id, base.phases[0].motif_id)
            self.assertEqual(item.phases[0].termination, base.phases[0].termination)

    def test_parameter_mutation_is_bounded_and_deterministic(self):
        primitive = Phase(ActionSource.PARAMETERIZED_PRIMITIVE, (ActionFamily.MARKET,),
                          Termination(TerminationType.FIXED_CAP, 2), primitive_name="LIQUIDITY_REBALANCE",
                          parameters={"enabled": True, "mode": "WHEAT", "quantity": 2})
        base = MacroProgram(cp().lane, cp(), (primitive,), 2)
        generated = expand_program(base, ExpansionOperator.MUTATE_BOUNDED_NUMERIC_PARAMETER, motifs(), ())
        self.assertEqual([dict(item.phases[0].parameters)["quantity"] for item in generated], [1, 3])

    def test_motif_swap_changes_one_phase_only(self):
        lib = motifs(); values = tuple(sorted(lib.values(), key=lambda item: item.semantic_id))
        base = MacroProgram(cp().lane, cp(), (phase(values[0].semantic_id, values[0].supported_families[0]),
                                             phase(values[0].semantic_id, values[0].supported_families[0])), 8)
        generated = expand_program(base, ExpansionOperator.SWAP_SEMANTIC_DONOR_OR_MOTIF_PHASE, lib, ())
        self.assertTrue(any(item.phases[1] == base.phases[1] and item.phases[0] != base.phases[0] for item in generated))

    def test_structural_dedup_and_expansion_order(self):
        base = program()
        candidates = (BeamCandidate.seed(base, SeedSource.HISTORICAL_MOTIF),
                      BeamCandidate.seed(MacroProgram(base.lane, base.initiation, base.phases, base.max_total_steps,
                                                      provenance=("different",)), SeedSource.HISTORICAL_MOTIF))
        unique, removed = deduplicate_candidates(candidates)
        self.assertEqual((len(unique), removed), (1, 1))
        first = expand_program(base, ExpansionOperator.INSERT_PHASE, motifs(), (cp(124),))
        second = expand_program(base, ExpansionOperator.INSERT_PHASE, dict(reversed(tuple(motifs().items()))), (cp(124),))
        self.assertEqual([item.semantic_id for item in first], [item.semantic_id for item in second])

    def test_budget_depth_lineage_and_no_overshoot(self):
        evaluator = FakeEvaluator()
        config = BeamConfig(max_depth=1, cuda_game_budget=100, max_candidates_per_expansion=20)
        result = run_beam("family", (cp(),), motifs(), evaluator, config)
        self.assertLessEqual(result.metrics["cuda_games"], 100)
        self.assertEqual(result.metrics["depth_reached"], 1)
        self.assertTrue(all(item.lineage.result_id == item.program.semantic_id for item in result.candidates))
        self.assertTrue(any(item.depth == 1 and item.lineage.parent_id for item in result.candidates))
        self.assertTrue(all(item.lineage.parameters for item in result.candidates if item.depth == 1))

    def test_checkpoint_motif_batches_are_deterministic_and_complete(self):
        checkpoints=(cp(),cp(124)); library=motifs()
        mapping={point.step:library for point in checkpoints}
        config=BeamConfig(max_depth=1,cuda_game_budget=196,max_candidates_per_expansion=48)
        first=run_beam("family",checkpoints,library,FakeEvaluator(),config,checkpoint_motifs=mapping)
        second=run_beam("family",tuple(reversed(checkpoints)),dict(reversed(tuple(library.items()))),FakeEvaluator(),config,
                        checkpoint_motifs={key:dict(reversed(tuple(value.items()))) for key,value in mapping.items()})
        self.assertEqual([row.program.semantic_id for row in first.candidates],
                         [row.program.semantic_id for row in second.candidates])
        self.assertEqual(first.metrics["cuda_games"],second.metrics["cuda_games"])
        self.assertEqual(len({row.program.semantic_id for row in first.candidates}),len(first.candidates))

    def test_stagnation_stops_exactly(self):
        evaluator = FakeEvaluator(same=True)
        result = run_beam("family", (cp(),), motifs(), evaluator,
                          BeamConfig(max_depth=4, cuda_game_budget=200, stagnation_rounds=1))
        self.assertEqual(result.terminal_reason, "STAGNATION")
        self.assertLess(result.metrics["depth_reached"], 4)

    def test_archive_backed_diversity_and_safety(self):
        evaluator = FakeEvaluator()
        result = run_beam("family", (cp(),), motifs(), evaluator,
                          BeamConfig(max_depth=1, cuda_game_budget=100))
        aggregates = {entry.aggregate_margin for entry in result.archive.entries()}
        self.assertGreater(len(aggregates), 1)
        self.assertGreater(len(result.archive.entries()), 1)

        # Direct safety falsifier: the archive must reject a huge-margin W->L rival.
        safe_effect = effect((10, 10), baseline=(100, 100))
        unsafe_effect = effect((-200, 1_000_000), baseline=(100, 100))
        safe = result.entry_for_effect(program(), safe_effect, "safe")
        unsafe = result.entry_for_effect(program(), unsafe_effect, "unsafe", candidate_id="unsafe")
        unsafe = unsafe.__class__.from_document({**unsafe.to_document(), "cell_key": safe.cell_key.to_document()})
        archive = Archive((unsafe, safe))
        self.assertEqual(archive.entries()[0].candidate_id, safe.candidate_id)

    def test_all_masks_and_operators_are_exactly_planned_sets(self):
        self.assertEqual(len(ALL_MASKS), 7)
        self.assertEqual({item.value for item in ExpansionOperator}, {
            "INSERT_PHASE", "REPLACE_PHASE_SOURCE", "SHIFT_PHASE_BOUNDARY_TO_ADJACENT_EVENT",
            "CHANGE_ACTION_FAMILY_MASK", "TRUNCATE_AT_EVENT", "EXTEND_TO_EVENT",
            "MUTATE_BOUNDED_NUMERIC_PARAMETER", "SWAP_SEMANTIC_DONOR_OR_MOTIF_PHASE",
        })


if __name__ == "__main__":
    unittest.main()
