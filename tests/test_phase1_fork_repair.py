"""Two focused Phase-1 contracts, not a second simulator or search suite."""
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

from kaggriculture_bonsai.runtime import bonsai_kg_v4 as bkg
from kaggriculture_bonsai.search import phase1_fork_repair as experiment


class Phase1Contracts(unittest.TestCase):
    def test_legalized_gate_bounds_and_identity(self):
        frozen = json.loads((experiment.OUT / "legal_shortlist.json").read_text())
        for candidate in frozen["candidates"]:
            path = experiment.OUT / (candidate["name"] + ".bkgc")
            self.assertEqual(experiment.sha(path), candidate["gate_sha256"])
            _, compiled = experiment.legal_gate(candidate["mode"], candidate["start"], candidate["duration"])
            self.assertEqual(compiled.semantic_hash, bkg.load_compiled(path).semantic_hash)
            self.assertEqual(compiled.state_count, 0)
            for step in range(719):
                observation = bkg.SyntheticObservation({(bkg.FEATURES["STEP"], 0): step})
                result = bkg.VM(compiled).tick(observation, max_market_orders=0)
                expected = candidate["mode"] if candidate["start"] <= step < candidate["start"] + candidate["duration"] else 0
                self.assertEqual(result.blackboard, (expected,))
                self.assertEqual(result.fault, 0)

    def test_native_snapshot_controller_and_cuda_parity(self):
        backends = [experiment.BUILD / name for name in ("cpu_gate", "cuda_gate")]
        if not all(path.exists() for path in backends):
            self.skipTest("compile the isolated Phase-1 CPU/CUDA harness first")
        outputs = []
        with tempfile.TemporaryDirectory(prefix="phase1_fidelity_") as directory:
            directory = Path(directory)
            cases, candidates = directory / "cases", directory / "candidates"
            # A fresh BUILD route-12 regime omitted from the initial route-0 fixtures.
            cases.write_text("19091103 0 0 359\n19091103 1 1 647\n")
            candidates.write_text("0 0 0\n1 1 24\n3 3 8\n")
            for binary in backends:
                command = [str(binary), str(experiment.PARENT), str(experiment.PORT / "routes.bkv6"),
                           str(experiment.PORT / "arity.bin"), str(cases), str(candidates), "1", "1"]
                result = subprocess.run(command, check=True, capture_output=True, text=True)
                rows = [dict(zip(experiment.FIELDS, map(int, line.split()))) for line in result.stdout.splitlines()]
                self.assertEqual(len(rows), 14)
                self.assertTrue(all(r["fault"] == 0 for r in rows))
                for offset in (2, 6, 10):
                    for fork, full in zip(rows[offset:offset + 2], rows[offset + 2:offset + 4]):
                        self.assertEqual(experiment.comparable(fork), experiment.comparable(full))
                outputs.append(rows)
            self.assertEqual(outputs[0], outputs[1])
            self.assertTrue(cases.with_suffix(".cpu.snapshots").exists())
            self.assertTrue(cases.with_suffix(".cuda.snapshots").exists())
        report = os.environ.get("PHASE1_TEST_EVIDENCE")
        if report:
            Path(report).write_text(json.dumps({"backend": "focused_test_cpu_cuda", "mismatches": 0,
                "simulated_turns": sum(r["turns"] for rows in outputs for r in rows),
                "rows": outputs[0], "checkpoints": [359, 647], "route": 12}, indent=2) + "\n")


if __name__ == "__main__":
    unittest.main()
