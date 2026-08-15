"""Exhaustively compare integer market prices with kaggle-environments 1.32.7."""
from __future__ import annotations

import subprocess
import sys
from importlib.metadata import version

EXPECTED_VERSION = "1.32.7"
ITEMS = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL", "FERTILIZER"]


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: check_price_parity.py PRICE_PROBE")
    actual = version("kaggle-environments")
    if actual != EXPECTED_VERSION:
        raise SystemExit(f"expected kaggle-environments {EXPECTED_VERSION}, got {actual}")
    from kaggle_environments.envs.kaggriculture.kaggriculture import market_price

    inventories = range(-10000, 20001)
    queries = [(item_i, inv) for item_i in range(len(ITEMS)) for inv in inventories]
    payload = "".join(f"{item_i} {inv}\n" for item_i, inv in queries)
    proc = subprocess.run([sys.argv[1]], input=payload, text=True, capture_output=True, check=True)
    got = [int(x) for x in proc.stdout.splitlines()]
    if len(got) != len(queries):
        raise SystemExit(f"price probe returned {len(got)} rows, expected {len(queries)}")

    mismatches = []
    for (item_i, inv), cpp in zip(queries, got):
        py = market_price(ITEMS[item_i], inv)
        if cpp != py:
            mismatches.append((ITEMS[item_i], inv, py, cpp))
            if len(mismatches) >= 20:
                break
    if mismatches:
        for row in mismatches:
            print("MISMATCH", *row)
        raise SystemExit(1)
    print(f"PASS market-price parity: {len(queries)} integer states exact ({EXPECTED_VERSION})")


if __name__ == "__main__":
    main()
