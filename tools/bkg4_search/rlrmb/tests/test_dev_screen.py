"""Tests for the INTERNAL_DEV cross-regime generalization screen."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.bkg4_search.rlrmb.dev_screen import (
    DevScreenMetrics, DevScreenPolicy, DevScreenResult, DevScreenStatus,
    apply_dev_screen, assert_dev_independence, classify_dev_outcome,
    dev_block_metrics, rank_dev_results, select_dev_panel,
    write_dev_screen_artifact,
)
from tools.bkg4_search.rlrmb.effects import EffectBlock
from tools.bkg4_search.rlrmb.schema import DatasetRole
from tools.bkg4_search.rlrmb.adversarial_league import dev_lane_rotation, dev_seat_rotation


def _block(baselines, deltas):
    return EffectBlock(DatasetRole.TARGET, "dev-panel", tuple(baselines), tuple(deltas))


def _result(name, status, **kwargs):
    stats = {"dev_cases": 4, "dev_activated": 4, "dev_loss_to_win": 1, "dev_win_to_loss": 0,
             "dev_positive": 2, "dev_harmful": 1, "dev_aggregate_margin": 10,
             "dev_worst_delta": -2}
    stats.update(kwargs)
    return DevScreenResult(
        candidate_semantic_id=name, family="f", lane="GENERAL", status=status,
        dev_panel="dev.tsv", generation=0, rotation_index=0,
        build_loss_to_win=3, build_protected_win_to_loss=0, **stats,
    )


def test_dev_block_metrics_separate_harm_from_win_to_loss():
    metrics = dev_block_metrics(_block((-5, 4, 3, 1), (10, -1, -6, 0)))
    assert metrics["dev_loss_to_win"] == 1
    assert metrics["dev_win_to_loss"] == 1      # 3 -> -3
    assert metrics["dev_harmful"] == 2          # harm is not the same as W->L
    assert metrics["dev_activated"] == 3
    assert metrics["dev_aggregate_margin"] == 3


def test_inert_candidate_is_classified_as_build_only_overfit():
    policy = DevScreenPolicy()
    metrics = dev_block_metrics(_block((-5, 4), (0, 0)))
    status = classify_dev_outcome(metrics, policy)
    assert status is DevScreenStatus.DEV_NO_CROSS_REGIME_EFFECT
    assert not _result("a", status).downstream_eligible
    assert _result("a", status).rejection_reason == "BUILD_ONLY_OVERFIT"


def test_broad_dev_regression_is_rejected_by_count_and_by_rate():
    strict = DevScreenPolicy(max_win_to_loss=2, max_win_to_loss_rate=0.25)
    by_count = dev_block_metrics(_block((5, 6, 7, 8), (-10, -12, -14, -16)))
    assert classify_dev_outcome(by_count, strict) is DevScreenStatus.DEV_REGRESSION
    by_rate = dev_block_metrics(_block((5, 6, -1), (-10, 1, 1)))
    assert by_rate["dev_win_to_loss"] == 1
    assert classify_dev_outcome(by_rate, strict) is DevScreenStatus.DEV_REGRESSION


def test_cross_regime_gain_passes_and_neutral_gain_weak_passes():
    policy = DevScreenPolicy()
    gain = dev_block_metrics(_block((-5, 4), (10, 1)))
    assert classify_dev_outcome(gain, policy) is DevScreenStatus.DEV_PASS
    neutral = dev_block_metrics(_block((4, 5), (1, -1)))
    assert classify_dev_outcome(neutral, policy) is DevScreenStatus.DEV_WEAK_PASS


def test_screen_withholds_regressions_from_downstream_budget():
    results = [
        _result("regress", DevScreenStatus.DEV_REGRESSION, dev_win_to_loss=9),
        _result("overfit", DevScreenStatus.DEV_NO_CROSS_REGIME_EFFECT, dev_activated=0),
        _result("good", DevScreenStatus.DEV_PASS),
    ]
    admitted, withheld = apply_dev_screen(results, DevScreenPolicy(max_downstream=3))
    assert [row.candidate_semantic_id for row in admitted] == ["good"]
    assert {row.candidate_semantic_id for row in withheld} == {"regress", "overfit"}


def test_fallback_keeps_discovery_alive_but_never_admits_a_regression():
    only_regressions = [_result("r", DevScreenStatus.DEV_REGRESSION, dev_win_to_loss=9)]
    admitted, withheld = apply_dev_screen(only_regressions, DevScreenPolicy())
    assert admitted == ()
    assert len(withheld) == 1

    overfits = [_result("o", DevScreenStatus.DEV_NO_CROSS_REGIME_EFFECT, dev_activated=0)]
    admitted, _ = apply_dev_screen(overfits, DevScreenPolicy())
    assert [row.status for row in admitted] == [DevScreenStatus.DEV_FALLBACK_ADMITTED]

    admitted, withheld = apply_dev_screen(overfits, DevScreenPolicy(allow_fallback=False))
    assert admitted == () and len(withheld) == 1


def test_ranking_prefers_cross_regime_evidence_over_build_gain():
    ranked = rank_dev_results([
        _result("build_heavy", DevScreenStatus.DEV_WEAK_PASS, dev_loss_to_win=0),
        _result("cross_regime", DevScreenStatus.DEV_PASS),
    ])
    assert [row.candidate_semantic_id for row in ranked] == ["cross_regime", "build_heavy"]


def test_panel_rotation_is_deterministic_and_cycles():
    panels = ("a.tsv", "b.tsv", "c.tsv")
    assert [select_dev_panel(panels, g).panel for g in range(4)] == ["a.tsv", "b.tsv", "c.tsv", "a.tsv"]
    assert select_dev_panel(panels, 5) == select_dev_panel(panels, 5)
    with pytest.raises(ValueError):
        select_dev_panel((), 0)


def test_dev_lane_and_seat_rotation_change_across_generations():
    assert dev_lane_rotation(0) != dev_lane_rotation(1)
    assert dev_seat_rotation(0) == (1, 0) and dev_seat_rotation(1) == (0, 1)
    assert dev_lane_rotation(3) == dev_lane_rotation(3)


def test_independence_check_fails_closed_on_build_reuse(tmp_path: Path):
    build = tmp_path / "build.tsv"
    build.write_text("tape_a.json\t0\t1\t0\t0\t0\t0\n", encoding="utf-8")
    clean = tmp_path / "dev.tsv"
    clean.write_text("tape_b.json\t1\t2\t0\t0\t0\t0\n", encoding="utf-8")
    report = assert_dev_independence([build], clean)
    assert report["independent"] and report["overlap"] == 0

    dirty = tmp_path / "dirty.tsv"
    dirty.write_text("tape_a.json\t1\t2\t0\t0\t0\t0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="reuses BUILD cases"):
        assert_dev_independence([build], dirty)


def test_artifact_records_provenance_and_declares_no_authority(tmp_path: Path):
    selection = select_dev_panel(("dev_a.tsv", "dev_b.tsv"), 1)
    results = [_result("good", DevScreenStatus.DEV_PASS)]
    results[0].provenance.update({"dev_panel": "dev_b.tsv", "generation": 1})
    metrics = DevScreenMetrics()
    metrics.observe(results[0])
    admitted, withheld = apply_dev_screen(results, DevScreenPolicy())
    metrics.observe_admission(admitted, withheld)
    document = write_dev_screen_artifact(
        tmp_path / "dev", selection, DevScreenPolicy(), results, admitted, metrics,
        {"independent": True, "overlap": 0},
    )
    assert document["authority"] == "SCREENING_ONLY"
    assert document["holdout_authority_unchanged"] is True
    assert document["panel_selection"]["panel"] == "dev_b.tsv"
    assert document["results"][0]["provenance"]["generation"] == 1
    assert document["metrics"]["dev_passes"] == 1
    on_disk = json.loads((tmp_path / "dev" / "dev_screen.json").read_text(encoding="utf-8"))
    assert on_disk == document


def _production_config():
    return json.loads(Path(
        "tools/bkg4_search/rlrmb/configs/p0_7_equal_budget_v1.json"
    ).read_text(encoding="utf-8"))


def test_internal_dev_config_must_be_bounded_funded_and_non_holdout(tmp_path: Path):
    from tools.bkg4_search.rlrmb.pipeline import validate_production_config

    def _check(dev, categories_extra=None, expect=None):
        config = _production_config()
        config["internal_dev"] = dev
        categories = config["budget"]["per_family_categories"]
        for key, value in (categories_extra or {}).items():
            categories[key] = value
            categories["target_beam"] -= value
        path = tmp_path / "config.json"
        path.write_text(json.dumps(config), encoding="utf-8")
        with pytest.raises(ValueError, match=expect):
            validate_production_config(path, tmp_path / "out")

    base = _production_config()
    _check({"panels": []}, expect="non-empty list")
    _check({"panels": [base["transfer_panel"]]},
           {"internal_dev": 10}, expect="must not reuse TRANSFER")
    _check({"panels": [base["families"][0]["target_panel"]]},
           {"internal_dev": 10}, expect="independent of BUILD")
    _check({"panels": ["scratch/does_not_exist_dev.tsv"]},
           {"internal_dev": 10}, expect="missing internal_dev panels")
    _check({"panels": [base["sealed_panel"]]}, {"internal_dev": 10},
           expect="must not reuse TRANSFER")
    _check({"panels": [base["validation_panel"]], "bogus": 1},
           {"internal_dev": 10}, expect="unsupported internal_dev fields")
