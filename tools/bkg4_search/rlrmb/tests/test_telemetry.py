"""Lightweight Lightning machine telemetry contracts."""
from __future__ import annotations

import importlib
import json
from pathlib import Path
import tempfile
import time
import unittest


class TelemetryTests(unittest.TestCase):
    def module(self):
        return importlib.import_module("tools.bkg4_search.rlrmb.cloud.telemetry")

    def test_cpu_percentages_use_machine_tick_deltas(self):
        module = self.module()
        previous = ((100, 100), (50, 50), (50, 50))
        current = ((200, 150), (100, 75), (100, 75))
        aggregate, cores = module.cpu_percentages(previous, current)
        self.assertAlmostEqual(aggregate, 50.0)
        self.assertEqual(tuple(round(value, 6) for value in cores), (50.0, 50.0))

    def test_sampler_writes_compact_jsonl_and_stops_cleanly(self):
        module = self.module()
        counter = {"value": 0}
        def sample():
            counter["value"] += 1
            return {"sample": counter["value"]}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "telemetry.jsonl"
            with module.TelemetrySampler(path, interval_seconds=0.02, sample_provider=sample):
                time.sleep(0.075)
            rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        self.assertGreaterEqual(len(rows), 2)
        self.assertEqual([row["sample"] for row in rows], list(range(1, len(rows) + 1)))


if __name__ == "__main__":
    unittest.main()
