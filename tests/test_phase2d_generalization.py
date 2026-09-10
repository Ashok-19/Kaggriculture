"""Whole-seed holdout and safety-first ranking contracts."""
from kaggriculture_bonsai.search.phase2d_generalization import group_metrics, rank, trajectories
from kaggriculture_bonsai.search.phase2_ccpr import program


def test_whole_trajectory_holdout():
    games = trajectories()
    fit = {g["seed"] for g in games if g["split"] == "discovery"}
    guard = {g["seed"] for g in games if g["split"] == "guard"}
    assert not fit & guard
    assert len(fit) == len(guard) == 16
    assert len({g["seed"] for g in games}) == 32
    assert sum(len(g["checkpoints"]) for g in games) == 48
    assert all(1 <= len(g["checkpoints"]) <= 2 for g in games)
    for split in ("discovery", "guard"):
        for start in (335, 359, 575, 647):
            panel = [g for g in games if g["split"] == split and start in g["checkpoints"]]
            assert {g["seat"] for g in panel} == {0, 1}
            assert {g["member"] for g in panel} == {0, 1}


def test_negative_group_cannot_hide_behind_aggregate_gain():
    def row(seed, seat, delta, transition="W->W"):
        return {"seed": seed, "seat": seat, "delta": delta, "duration": 24, "transition": transition}
    brittle = [row(1, 0, 2000), row(2, 1, 1800), row(3, 0, 1400), row(4, 1, -1000)]
    safe = [row(1, 0, 10), row(2, 1, 10), row(3, 0, 0), row(4, 1, 0)]
    p = program(359, 1, 24)
    assert group_metrics(brittle)["net"] > group_metrics(safe)["net"]
    assert not group_metrics(brittle)["eligible"]
    assert group_metrics(safe)["eligible"]
    assert rank(safe, p) < rank(brittle, p)
    assert rank(list(reversed(safe)), p) == rank(safe, p)
    duplicated_seed = [row(1, 0, 10), row(1, 1, 10)]
    assert not group_metrics(duplicated_seed)["eligible"]
    regressed = [row(1, 0, 10000), row(2, 1, -1, "T->L")]
    assert rank(safe, p) < rank(regressed, p)
