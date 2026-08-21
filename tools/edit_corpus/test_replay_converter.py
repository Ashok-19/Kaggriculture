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
            tapes = {int(r["source_seat"]): json.loads(Path(r["tape_path"]).read_text()) for r in rows}
            self.assertEqual(tapes[0]["actions"][0]["farmer"], ["PASS"])
            self.assertEqual(tapes[0]["actions"][0]["hands"], [["PICKUP", "WHEAT", 2]])
            self.assertEqual(tapes[0]["actions"][0]["market"], [["BUY_SEED", "TOMATO", 20], []])
            self.assertEqual(tapes[1]["actions"][0]["hands"], [["PASS"]])
            self.assertEqual(tapes[1]["actions"][0]["market"], [["HIRE"], ["SELL", "MILK", 2]])
            for row in rows:
                actions = json.loads(Path(row["tape_path"]).read_text())["actions"]
                self.assertEqual(
                    row["tape_semantic_id"],
                    stable_identity("KAGGRICULTURE-NATIVE-TAPE-V1", actions),
                )


if __name__ == "__main__":
    unittest.main()
