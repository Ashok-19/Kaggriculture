"""Fresh panel separation and genuinely label-independent control proposals."""
import pytest

from kaggriculture_bonsai.search import phase2_ccpr as c
from kaggriculture_bonsai.search import phase2d_generalization as d
from kaggriculture_bonsai.search.phase2_ccpr_corrected import STEPS, control_plan, groups, stop_reason


def test_corrected_whole_seed_panels():
    games = groups()
    assert len({g["seed"] for g in games}) == len(games) == 48
    assert sum(len(g["checkpoints"]) for g in games) == 72
    fit = {g["seed"] for g in games if g["split"] == "discovery"}
    guard = {g["seed"] for g in games if g["split"] == "guard"}
    assert len(fit) == len(guard) == 24 and not fit & guard
    assert not (fit | guard) & set(range(49092001, 49092033))
    for split in ("discovery", "guard"):
        for step in STEPS:
            rr = [g for g in games if g["split"] == split and step in g["checkpoints"]]
            assert len(rr) == 6
            assert {r["seat"] for r in rr} == {r["member"] for r in rr} == {0, 1}


def test_control_is_independent_bounded_and_reproducible():
    roots = [{"seed": i, "seat": i % 2, "features": [10 * i, i % 3, 2, 100 + i, 50 - i, i % 4, 8, 0, 0, i, i % 3, i % 2],
              "delta": -i, "transition": "T->L", "duration": 48} for i in range(6)]
    initial = c.program(359, 1)
    plan = control_plan(initial, roots)
    changed_labels = [{**r, "delta": 999999, "transition": "L->W"} for r in reversed(roots)]
    assert plan == control_plan(initial, changed_labels)
    assert plan[0] == initial and len(plan) == 7
    assert len({c.identity(p) for p in plan}) == len(plan)
    assert {len(p["predicates"]) for p in plan} == {0, 1, 2}
    assert all(p["mode"] == 1 and p["start"] == 359 and p["duration"] <= 48 for p in plan)
    assert stop_reason([{**r, "delta": 0} for r in roots]) == "ZERO_VALUE"
    assert stop_reason([{**r, "delta": 1, "transition": "T->W"} for r in roots]) == "SAFE_USEFUL"


def test_shared_budget_counts_reference_work_before_launch(monkeypatch):
    config = {"budget_fge": 1}
    monkeypatch.setattr(c, "load", lambda name: config if name == "config.json" else [{"turns": 350}])
    monkeypatch.setattr(d, "regression_turns", lambda: 11)
    launches = []
    monkeypatch.setattr(c, "evaluate", lambda *args, **kwargs: launches.append(args) or [{}])
    with pytest.raises(RuntimeError, match="budget"):
        d.evaluate(c.program(359, 1), [(1, 0, 0, 359)], "blocked")
    assert not launches
    config["budget_fge"] = 2
    assert d.evaluate(c.program(359, 1), [(1, 0, 0, 359)], "allowed") == [{}]
    assert len(launches) == 1
