from __future__ import annotations

import json
from pathlib import Path

from tools.bkg4_search.rlrmb.build_regimes import build_support, load_build_regimes
from tools.bkg4_search.rlrmb.effects import EffectBlock
from tools.bkg4_search.rlrmb.schema import DatasetRole


def test_support_requires_multiple_seeds_seats_and_members(tmp_path: Path):
    panel = tmp_path / "build.tsv"
    lines = []
    for index, (seed, seat, member, lane) in enumerate((
        (11, 0, "member_a", "CROP_TRANSITION"),
        (11, 1, "member_a", "CROP_TRANSITION"),
        (23, 0, "member_b", "MARKET_LIQUIDITY"),
        (23, 1, "member_b", "MARKET_LIQUIDITY"),
    )):
        tape = tmp_path / f"tape_{index}.json"
        tape.write_text(json.dumps({"source": {
            "kind": "selfplay", "opponent_policy": member,
            "strategic_lane": lane, "exploiter_program": f"{member}.rlrb",
        }}), encoding="utf-8")
        lines.append(f"{tape}\t{seat}\t{seed}\t0\t0\t0\t0")
    panel.write_text("\n".join(lines) + "\n", encoding="utf-8")
    regimes = load_build_regimes(panel)
    block = EffectBlock(DatasetRole.TARGET, "build", (-1, -1, -1, -1), (2, 2, 2, 2))

    support = build_support(block, regimes)

    assert support.cross_regime_supported
    assert support.seed_support == support.seat_support == support.member_support == 2
    assert support.lane_support == support.trajectory_support == 2
    assert support.both_seat_member_support == 2


def test_duplicate_regime_coordinates_fail_closed(tmp_path: Path):
    tape = tmp_path / "tape.json"
    tape.write_text(json.dumps({"source": {"opponent_policy": "member"}}), encoding="utf-8")
    panel = tmp_path / "build.tsv"
    panel.write_text(f"{tape}\t0\t11\n{tape}\t0\t11\n", encoding="utf-8")
    try:
        load_build_regimes(panel)
    except ValueError as exc:
        assert "duplicate regime" in str(exc)
    else:
        raise AssertionError("duplicate BUILD regimes must fail closed")
