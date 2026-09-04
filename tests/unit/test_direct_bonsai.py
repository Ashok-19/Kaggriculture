from __future__ import annotations

from dataclasses import dataclass

from kaggriculture_bonsai.league.build_regimes import BuildRegime
from kaggriculture_bonsai.search.beam import BatchEvaluation, EvaluatedProgram
from kaggriculture_bonsai.search.checkpoints import Checkpoint, EventType, Lane
from kaggriculture_bonsai.search.direct_bonsai import (
    DirectSearchConfig,
    direct_rule_templates,
    generate_direct_seeds,
    run_direct_search,
)
from kaggriculture_bonsai.search.effects import EffectBlock, EffectVector
from kaggriculture_bonsai.search.evaluator import ReferenceInterpreter
from kaggriculture_bonsai.search.schema import DatasetRole


def checkpoints() -> tuple[Checkpoint, ...]:
    return (
        Checkpoint(24, Lane.LAND_INVESTMENT, (EventType.BUY_LAND,)),
        Checkpoint(72, Lane.WORKER_PRODUCTION, (EventType.HIRE,)),
    )


@dataclass
class FakeEvaluator:
    case_count: int = 4

    def __call__(self, programs, motifs):
        base = (-10, -10, 10, 10)
        rows = []
        for program in programs:
            delta = (20, 20, 0, 0)
            block = EffectBlock(DatasetRole.TARGET, "panel", base, delta, (1, 1, 0, 0))
            rows.append(EvaluatedProgram(program, EffectVector((block,)), True, None))
        return BatchEvaluation(tuple(rows), 4, (len(programs) + 1) * 4, 1, 0.01, 0.1)


def regimes() -> tuple[BuildRegime, ...]:
    return (
        BuildRegime(1, 0, "a", "LAND", "a0"),
        BuildRegime(2, 1, "b", "WORKER", "b1"),
        BuildRegime(1, 0, "a", "LAND", "a2"),
        BuildRegime(2, 1, "b", "WORKER", "b3"),
    )


def test_templates_are_action_grammar_not_donor_windows() -> None:
    templates = direct_rule_templates()
    assert len(templates) >= 32
    assert len({row.name for row in templates}) == len(templates)
    assert all(set(row.action) == {"farmer", "hands", "market"} for row in templates)


def test_seed_generation_is_deterministic_and_has_no_donor_provenance() -> None:
    first, motifs = generate_direct_seeds(checkpoints())
    second, _ = generate_direct_seeds(tuple(reversed(checkpoints())))
    assert [row.program.semantic_id for row in first] == [row.program.semantic_id for row in second]
    assert all("source_path=" not in " ".join(motif.provenance) for motif in motifs.values())
    assert all("direct_bonsai_action_grammar" in motif.provenance for motif in motifs.values())


def test_direct_rule_executes_through_existing_reference_interpreter() -> None:
    candidates, motifs = generate_direct_seeds((checkpoints()[0],))
    candidate = candidates[0]
    runner = ReferenceInterpreter(candidate.program, motifs)
    observation = {"step": 24, "day": 1, "hour": 0, "farms": [{"money": 3000}, {"money": 3000}],
                   "player": 0, "market": {"prices": {}, "inventory": {}}, "town": {"unlocked_shops": []}}
    parent = {"farmer": ["PASS"], "hands": [], "market": []}
    expected = motifs[candidate.program.phases[0].motif_id].semantic_document()["action_block"][0]
    action = runner.action(observation, parent)
    assert action["farmer"] == expected["farmer"]
    assert action["hands"] == [] and action["market"] == []


def test_direct_search_obeys_exact_cuda_cap_and_tracks_cross_regime_support() -> None:
    result = run_direct_search(
        checkpoints(), FakeEvaluator(), regimes(),
        DirectSearchConfig(cuda_game_budget=40, initial_candidates=5, elite_count=2),
    )
    assert result.metrics["cuda_games"] == 40
    assert result.metrics["cuda_games"] <= result.metrics["cuda_budget"]
    assert result.metrics["semantic_mismatches"] == 0
    assert result.metrics["cross_regime_supported_specialists"] > 0
