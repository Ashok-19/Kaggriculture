from __future__ import annotations

import csv
import json
import shutil
import struct
import subprocess
import tempfile
import unittest
from pathlib import Path

from tools.edit_corpus.corpus import stable_identity

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "cuda-engine/src/replay_to_native_tapes.cpp"

TAPE_HEADER = struct.Struct("<8sIiqI")
ACTION_HEADER = struct.Struct("<BBH")
UNIT = struct.Struct("<BBh")
ORDER = struct.Struct("<BBhi")

UNIT_NAMES = {
    "PASS": 0, "NORTH": 1, "SOUTH": 2, "EAST": 3, "WEST": 4,
    "PICKUP": 5, "DROP": 6, "PLACE": 7, "PLANT": 8, "WATER": 9,
    "HARVEST": 10, "FERTILIZE": 11, "DIG": 12, "BUILD_COOP": 13,
    "BUILD_PASTURE": 14, "FEED": 15, "COLLECT_FERTILIZER": 16, "CARE": 17,
}
ITEM_NAMES = {
    "WHEAT": 0, "CARROT": 1, "TOMATO": 2, "STRAWBERRY": 3, "MELON": 4,
    "EGG": 5, "MILK": 6, "WOOL": 7, "FERTILIZER": 8, "GOOSE": 9,
    "COW": 10, "SHEEP": 11,
}
MARKET_NAMES = {"HIRE": 1, "BUY_LAND": 2, "BUY_SEED": 3, "BUY_PRODUCT": 4, "BUY_ANIMAL": 5, "SELL": 6}


def read_binary_tape(path: Path) -> dict:
    raw = path.read_bytes()
    magic, version, seat, seed, count = TAPE_HEADER.unpack_from(raw, 0)
    if magic != b"B4TP1\0\0\0" or version != 1:
        raise AssertionError("bad test tape header")
    off = TAPE_HEADER.size
    actions = []
    for _ in range(count):
        n_units, n_orders, reserved = ACTION_HEADER.unpack_from(raw, off)
        off += ACTION_HEADER.size
        if reserved:
            raise AssertionError("nonzero action reserved field")
        units = []
        for _ in range(n_units):
            units.append(UNIT.unpack_from(raw, off))
            off += UNIT.size
        orders = []
        for _ in range(n_orders):
            orders.append(ORDER.unpack_from(raw, off))
            off += ORDER.size
        actions.append((units, orders))
    if off != len(raw):
        raise AssertionError("trailing binary tape bytes")
    return {"seat": seat, "seed": seed, "actions": actions}


class ReplayConverterTests(unittest.TestCase):
    def test_compiled_converter_normalizes_edge_actions_and_preserves_v1_identity(self):
        gpp = shutil.which("g++")
        if not gpp:
            self.skipTest("g++ unavailable")
        replay = {
            "info": {"EpisodeId": 1, "TeamNames": ["A", "B"], "seed": 77},
            "steps": [
                [{"action": None}, {"action": None}],
                [
                    {"action": {
                        "farmer": "PLANT MELON",
                        "hands": [["PICKUP", "WHEAT", 2.9]],
                        "market": [["BUY_SEED", "TOMATO", 20.0], ["SELL"]],
                    }},
                    {"action": {
                        "farmer": ["NORTH"],
                        "hands": ["BAD"],
                        "market": [["HIRE"], ["SELL", "MILK", "2"]],
                    }},
                ],
            ],
        }
        raw = json.dumps(replay, separators=(",", ":")).encode()
        expected_actions = {
            0: [{"farmer": ["PASS"], "hands": [["PICKUP", "WHEAT", 2]], "market": [["BUY_SEED", "TOMATO", 20], []]}],
            1: [{"farmer": ["NORTH"], "hands": [["PASS"]], "market": [["HIRE"], ["SELL", "MILK", 2]]}],
        }
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            binary = td / "convert"
            cp = subprocess.run(
                [gpp, "-O1", "-std=c++17", "-I", str(ROOT / "bonsai-kg/include"), "-o", str(binary), str(SOURCE)],
                text=True,
                capture_output=True,
            )
            self.assertEqual(cp.returncode, 0, cp.stdout + "\n" + cp.stderr)
            stream = b"B4RSP01\0" + struct.pack("<qQ", 1, len(raw)) + raw
            meta = td / "meta.tsv"
            out = td / "tapes"
            rp = subprocess.run(
                [str(binary), "--stdin", "--out-dir", str(out), "--metadata", str(meta)],
                input=stream,
                capture_output=True,
            )
            stdout = rp.stdout.decode("utf-8", errors="replace")
            stderr = rp.stderr.decode("utf-8", errors="replace")
            self.assertEqual(rp.returncode, 0, stdout + "\n" + stderr)
            self.assertIn("tapes=2", stdout)
            with meta.open(encoding="utf-8", newline="") as f:
                rows = list(csv.DictReader(f, delimiter="\t"))
            self.assertEqual(len(rows), 2)
            self.assertTrue(all(Path(r["tape_path"]).suffix == ".b4tp" for r in rows))

            tapes = {int(r["source_seat"]): read_binary_tape(Path(r["tape_path"])) for r in rows}
            self.assertEqual(tapes[0]["seed"], 77)
            self.assertEqual(tapes[0]["actions"][0][0], [
                (UNIT_NAMES["PASS"], 0, 1),
                (UNIT_NAMES["PICKUP"], ITEM_NAMES["WHEAT"], 2),
            ])
            self.assertEqual(tapes[0]["actions"][0][1], [
                (MARKET_NAMES["BUY_SEED"], ITEM_NAMES["TOMATO"], 0, 20),
                (0, 0, 0, 0),
            ])
            self.assertEqual(tapes[1]["actions"][0][0], [
                (UNIT_NAMES["NORTH"], 0, 1),
                (UNIT_NAMES["PASS"], 0, 1),
            ])
            self.assertEqual(tapes[1]["actions"][0][1], [
                (MARKET_NAMES["HIRE"], 0, 0, 1),
                (MARKET_NAMES["SELL"], ITEM_NAMES["MILK"], 0, 2),
            ])
            for row in rows:
                seat = int(row["source_seat"])
                self.assertEqual(
                    row["tape_semantic_id"],
                    stable_identity("KAGGRICULTURE-NATIVE-TAPE-V1", expected_actions[seat]),
                )

    def test_compiled_converter_tolerates_worker_with_only_skipped_replays(self):
        gpp = shutil.which("g++")
        if not gpp:
            self.skipTest("g++ unavailable")
        replay = {"info": {"seed": 77}, "steps": []}
        raw = json.dumps(replay, separators=(",", ":")).encode()
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            binary = td / "convert"
            cp = subprocess.run(
                [gpp, "-O1", "-std=c++17", "-I", str(ROOT / "bonsai-kg/include"), "-o", str(binary), str(SOURCE)],
                text=True,
                capture_output=True,
            )
            self.assertEqual(cp.returncode, 0, cp.stdout + "\n" + cp.stderr)
            stream = b"B4RSP01\0" + struct.pack("<qQ", 1, len(raw)) + raw
            meta = td / "meta.tsv"
            out = td / "tapes"
            rp = subprocess.run(
                [str(binary), "--stdin", "--out-dir", str(out), "--metadata", str(meta)],
                input=stream,
                capture_output=True,
            )
            self.assertEqual(rp.returncode, 0, rp.stdout.decode(errors="replace") + rp.stderr.decode(errors="replace"))
            self.assertIn("replays=1 tapes=0 skipped=1", rp.stdout.decode(errors="replace"))
            with meta.open(encoding="utf-8", newline="") as f:
                self.assertEqual(list(csv.DictReader(f, delimiter="\t")), [])

    def test_cuda_tape_loader_round_trips_binary_converter_output(self):
        gpp = shutil.which("g++")
        nvcc = shutil.which("nvcc")
        if not gpp or not nvcc:
            self.skipTest("g++/nvcc unavailable")
        replay = {
            "info": {"seed": 91},
            "steps": [
                [{"action": None}, {"action": None}],
                [{"action": {"farmer": ["EAST"], "market": [["BUY_LAND"]]}}, {"action": {"farmer": ["WEST"]}}],
            ],
        }
        raw = json.dumps(replay, separators=(",", ":")).encode()
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            binary = td / "convert"
            cp = subprocess.run([gpp, "-O1", "-std=c++17", "-I", str(ROOT / "bonsai-kg/include"), "-o", str(binary), str(SOURCE)], capture_output=True, text=True)
            self.assertEqual(cp.returncode, 0, cp.stdout + cp.stderr)
            meta = td / "meta.tsv"
            out = td / "tapes"
            stream = b"B4RSP01\0" + struct.pack("<qQ", 9, len(raw)) + raw
            rp = subprocess.run([str(binary), "--stdin", "--out-dir", str(out), "--metadata", str(meta)], input=stream, capture_output=True)
            self.assertEqual(rp.returncode, 0, rp.stdout.decode(errors="replace") + rp.stderr.decode(errors="replace"))
            with meta.open(encoding="utf-8", newline="") as f:
                row = next(csv.DictReader(f, delimiter="\t"))

            loader_src = td / "load.cu"
            loader_src.write_text(
                '#include "cuda-engine/include/tape_io.hpp"\n'
                '#include <cstdio>\n'
                'int main(int argc,char**argv){auto t=kagcuda::load_native_tape(argv[1]);'
                'std::printf("%llu %d %zu %u %u %d\\n",(unsigned long long)t.source_seed,t.source_seat,t.actions.size(),'
                't.actions[0].units[0].op,t.actions[0].orders[0].op,t.actions[0].orders[0].n);}\n'
            )
            loader = td / "load"
            cc = subprocess.run([nvcc, "-std=c++17", "-I", str(ROOT), str(loader_src), "-o", str(loader)], capture_output=True, text=True)
            self.assertEqual(cc.returncode, 0, cc.stdout + cc.stderr)
            got = subprocess.run([str(loader), row["tape_path"]], capture_output=True, text=True)
            self.assertEqual(got.returncode, 0, got.stdout + got.stderr)
            self.assertEqual(got.stdout.strip(), "91 0 1 3 2 1")


if __name__ == "__main__":
    unittest.main()
