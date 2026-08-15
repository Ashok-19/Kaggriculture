"""Export exact-action ground truth from kaggle-environments 1.32.7.

Run this through the pinned uv environment, for example:
  uv run --with kaggle-environments==1.32.7 python cpp-engine/tools/export_trace.py --agents starter,random 11 22 33
"""
from __future__ import annotations

import contextlib
import io
import re
import sys
from importlib.metadata import version
from pathlib import Path

EXPECTED_VERSION = "1.32.7"

OPS = ["PASS", "NORTH", "SOUTH", "EAST", "WEST", "PICKUP", "DROP", "PLACE",
       "PLANT", "WATER", "HARVEST", "FERTILIZE", "DIG",
       "BUILD_COOP", "BUILD_PASTURE", "FEED", "COLLECT_FERTILIZER", "CARE"]
ITEMS = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL",
         "FERTILIZER", "GOOSE", "COW", "SHEEP"]
MOPS = ["NONE", "HIRE", "BUY_LAND", "BUY_SEED", "BUY_PRODUCT", "BUY_ANIMAL", "SELL"]
OPI = {o: i for i, o in enumerate(OPS)}
ITI = {o: i for i, o in enumerate(ITEMS)}
MOPI = {o: i for i, o in enumerate(MOPS)}


def _require_version() -> None:
    actual = version("kaggle-environments")
    if actual != EXPECTED_VERSION:
        raise RuntimeError(f"expected kaggle-environments {EXPECTED_VERSION}, got {actual}")


def enc_unit(a):
    if not isinstance(a, list) or not a:
        return (0, 0, 1)
    op = OPI.get(a[0], len(OPS))
    arg = ITI.get(a[1], 255) if len(a) >= 2 and isinstance(a[1], str) else 0
    try:
        n = int(a[2]) if len(a) >= 3 else 1
    except (TypeError, ValueError):
        n = 1
    return (op, arg, n)


def enc_order(o):
    if not isinstance(o, list) or not o:
        return (0, 0, 0)
    op = MOPI.get(o[0], 0)
    if op in (1, 2):
        return (op, 0, 1)
    if len(o) < 3:
        return (0, 0, 0)
    return (op, ITI.get(o[1], 255), int(o[2]))


def _slug(agent: str) -> str:
    value = Path(agent).stem if ("/" in agent or agent.endswith(".py")) else agent
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def export(seed: int, agents=("starter", "random"), out_dir=".") -> Path:
    _require_version()
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        from kaggle_environments import make
        env = make("kaggriculture", configuration={"episodeSteps": 720, "seed": seed}, debug=False)
        env.run(list(agents))

    n = len(env.steps) - 1
    cfg = env.configuration
    conf = [cfg["episodeSteps"], cfg["boardSize"], cfg["startingMoney"],
            cfg["maxMarketOrdersPerTurn"], cfg["turnsPerDay"], cfg["shedCapacity"],
            cfg["weedSpawnChance"], cfg["townShopUnlockInterval"],
            cfg["townShopSellInterval"], cfg["townCenterSellInterval"],
            cfg["farmHandCostMult"]]
    lines = [f"{seed} {n}", "CONFIG " + " ".join(str(v) for v in conf)]
    for t in range(n):
        for seat in (0, 1):
            act = env.steps[t + 1][seat].action or {}
            units = [act.get("farmer") or ["PASS"]] + list(act.get("hands") or [])
            orders = list(act.get("market") or [])
            parts = [str(len(units)), str(len(orders))]
            for u in units:
                parts += [str(v) for v in enc_unit(u)]
            for o in orders:
                parts += [str(v) for v in enc_order(o)]
            lines.append(" ".join(parts))

    money = [[float(s[0].observation.farms[0]["money"]),
              float(s[0].observation.farms[1]["money"])] for s in env.steps]
    inv = [[int(v) for v in s[0].observation.market["inventory"].values()] for s in env.steps]
    lines.append("TRUTH")
    for i in range(len(money)):
        lines.append(" ".join([f"{money[i][0]:.0f}", f"{money[i][1]:.0f}"] + [str(v) for v in inv[i]]))

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    p = out / f"replay_{_slug(agents[0])}_vs_{_slug(agents[1])}_{seed}.txt"
    p.write_text("\n".join(lines) + "\n")
    print(f"seed {seed}: final money {money[-1]} -> {p}")
    return p


def main() -> None:
    args = sys.argv[1:]
    agents = ("starter", "random")
    out_dir = "."
    while args and args[0].startswith("--"):
        flag = args.pop(0)
        if flag == "--agents":
            agents = tuple(args.pop(0).split(","))
        elif flag == "--out-dir":
            out_dir = args.pop(0)
        else:
            raise SystemExit(f"unknown option {flag}")
    seeds = [int(x) for x in (args or [70117])]
    for seed in seeds:
        export(seed, agents, out_dir)


if __name__ == "__main__":
    main()
