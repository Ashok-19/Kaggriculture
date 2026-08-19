from __future__ import annotations

import argparse
import copy
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
KAG_ENV = ROOT / "kaggle-environments-master"
CHAMPION = ROOT / "bonsai-kg/integrations/bkg4_champion"
for p in (KAG_ENV, CHAMPION):
    sys.path.insert(0, str(p))

from kaggle_environments import make
from runtime import BKG4ChampionAgent

PRODUCTS = ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL", "FERTILIZER")
CROPS = PRODUCTS[:5]
ANIMALS = ("GOOSE", "COW", "SHEEP")
MARKET_OPS = ("HIRE", "BUY_LAND", "BUY_SEED", "BUY_PRODUCT", "BUY_ANIMAL", "SELL")


def tape_actions(path: Path) -> list[dict]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    actions = obj.get("actions") if isinstance(obj, dict) else None
    if not isinstance(actions, list) or not actions:
        raise ValueError(path)
    return actions


def tile_counts(farm: dict) -> dict:
    crop = Counter(); animal = Counter(); kinds = Counter()
    for row in farm.get("tiles") or []:
        for tile in row:
            if tile == "LOCKED":
                kinds["LOCKED"] += 1
            elif tile is None:
                kinds["EMPTY"] += 1
            elif isinstance(tile, dict):
                kinds[str(tile.get("kind") or "?")] += 1
                if tile.get("kind") == "PLANT":
                    crop[str(tile.get("crop"))] += 1
                if "animal" in tile:
                    animal[str(tile.get("animal"))] += 1
    out = {f"crop_{k}": int(crop[k]) for k in CROPS}
    out.update({f"animal_{k}": int(animal[k]) for k in ANIMALS})
    out.update({f"tiles_{k.lower()}": int(v) for k, v in kinds.items()})
    return out


def market_summary(action: dict) -> dict:
    orders = list((action or {}).get("market") or [])
    by_op = Counter(); by_item = Counter()
    for order in orders:
        if not isinstance(order, list) or not order:
            continue
        op = str(order[0]); by_op[op] += 1
        if op in ("BUY_SEED", "BUY_PRODUCT", "BUY_ANIMAL", "SELL") and len(order) >= 3:
            try: n = int(order[2])
            except Exception: n = 0
            item = str(order[1]) if len(order) >= 2 else "?"
            by_item[f"{op}_{item}"] += max(0, n)
    out = {f"market_orders_{op.lower()}": int(by_op[op]) for op in MARKET_OPS}
    for op in ("BUY_SEED", "BUY_PRODUCT", "BUY_ANIMAL", "SELL"):
        for item in (*PRODUCTS, *ANIMALS):
            q = by_item.get(f"{op}_{item}", 0)
            if q:
                out[f"market_qty_{op.lower()}_{item.lower()}"] = int(q)
    out["market_order_count"] = len(orders)
    return out


def frame(case_index: int, case: dict, obs: dict, action: dict, champion: BKG4ChampionAgent) -> dict:
    p = int(obs.get("player", 0) or 0)
    farms = list(obs.get("farms") or [])
    own = farms[p]; opp = farms[1 - p]
    private = obs.get("private") or {}
    town = obs.get("town") or {}
    market = obs.get("market") or {}
    shops = list(town.get("unlocked_shops") or [])
    shed = private.get("shed") or {}; seeds = private.get("seeds") or {}
    invs = list(private.get("inventories") or [])
    carried = Counter()
    for inv in invs:
        if isinstance(inv, dict):
            for k, v in inv.items():
                carried[str(k)] += int(v or 0)
    out = {
        "case": case_index,
        "step": int(obs.get("step", 0) or 0),
        "seat": int(case["seat"]),
        "role": case["role"],
        "own_money": int(round(float(own.get("money", 0) or 0))),
        "opp_money": int(round(float(opp.get("money", 0) or 0))),
        "own_units": 1 + len(own.get("hands") or []),
        "opp_units": 1 + len(opp.get("hands") or []),
        "own_land": len(own.get("unlocked_quadrants") or []),
        "opp_land": len(opp.get("unlocked_quadrants") or []),
        "own_hires_today": int(own.get("hires_today", 0) or 0),
        "opp_hires_today": int(opp.get("hires_today", 0) or 0),
        "shop_count": len(shops),
        "first_shop": shops[0] if shops else None,
        "second_shop": shops[1] if len(shops) > 1 else None,
        "route": champion.state().get("route"),
        "feed_defer_count": len(champion.state().get("feed_defer_units") or []),
        "carried_total": sum(carried.values()),
        "shed_total": sum(int(v or 0) for v in shed.values()),
    }
    for shop in sorted(set(shops)):
        out[f"shop_seen_{shop.lower()}"] = shops.count(shop)
    for item in PRODUCTS:
        out[f"shed_{item.lower()}"] = int(shed.get(item, 0) or 0)
        out[f"carried_{item.lower()}"] = int(carried.get(item, 0))
        out[f"market_price_{item.lower()}"] = int((market.get("prices") or {}).get(item, 0) or 0)
        out[f"market_inv_{item.lower()}"] = int((market.get("inventory") or {}).get(item, 0) or 0)
    for crop in CROPS:
        out[f"seed_{crop.lower()}"] = int(seeds.get(crop, 0) or 0)
    for prefix, farm in (("own", own), ("opp", opp)):
        for k, v in tile_counts(farm).items():
            out[f"{prefix}_{k}"] = v
    out.update(market_summary(action))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=Path, default=Path("scratch/bkg4_search/optimization_oracle/screen_cases.jsonl"))
    ap.add_argument("--assets", type=Path, default=Path("bonsai-kg/integrations/bkg4_champion/champions/feed_deferral_v1"))
    ap.add_argument("--probe", type=int, action="append", default=[])
    ap.add_argument("--out", type=Path, default=Path("scratch/bkg4_search/optimization_oracle/public_features_96_120_144.jsonl"))
    args = ap.parse_args()
    cases_path = (ROOT / args.cases).resolve() if not args.cases.is_absolute() else args.cases
    assets = (ROOT / args.assets).resolve() if not args.assets.is_absolute() else args.assets
    out = (ROOT / args.out).resolve() if not args.out.is_absolute() else args.out
    probes = sorted(set(args.probe or [96, 120, 144]))
    rows = [json.loads(x) for x in cases_path.read_text(encoding="utf-8").splitlines() if x.strip()]
    records = []
    for ci, case in enumerate(rows):
        actions = tape_actions(Path(case["tape"]))
        champion = BKG4ChampionAgent(assets)
        env = make("kaggriculture", configuration={"seed": int(case["seed"]), "episodeSteps": 720}, debug=True)
        env.reset(2)
        seat = int(case["seat"])
        for step in range(max(probes) + 1):
            obs = copy.deepcopy(dict(env.state[seat].observation))
            # Manual env.step does not advance the secondary player's framework
            # observation.step field the same way env.run agent dispatch does.
            # Inject the authoritative loop clock so capture exactly matches the
            # observation clock seen by a normal agent call.
            obs["step"] = step
            obs["day"] = step // 24
            obs["hour"] = step % 24
            ca = champion(obs, env.configuration)
            if step in probes:
                records.append(frame(ci, case, obs, ca, champion))
            oa = copy.deepcopy(actions[min(step, len(actions) - 1)])
            env.step([ca, oa] if seat == 0 else [oa, ca])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in records), encoding="utf-8")
    print(json.dumps({"cases": len(rows), "probes": probes, "records": len(records), "out": str(out)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
