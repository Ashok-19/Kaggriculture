"""Focused integrity, state-boundary and active-overlay parity tests."""
import copy
import json
import tempfile
import unittest
from pathlib import Path

from .extract import ASSETS, generate, overlay_source, source, HERE, routes
from .validate import Audit


def observation(step=0, seat=0, yarn=False, egg=10000):
    farm = {"money": 1000, "farmer": [4, 4], "hands": [[4, 4] for _ in range(12)],
            "tiles": [[None for _ in range(10)] for _ in range(10)], "unlocked_quadrants": [0], "hires_today": 0}
    return {"step": step, "player": seat, "farms": [copy.deepcopy(farm), copy.deepcopy(farm)],
            "town": {"unlocked_shops": ["YARN_STORE"] if yarn else []},
            "market": {"inventory": {"EGG": egg}, "prices": {k: 3 for k in ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL", "FERTILIZER")}},
            "private": {"shed": {}, "seeds": {k: 100 for k in ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON")}, "inventories": [{} for _ in range(13)]}}


class GoldV25Tests(unittest.TestCase):
    def audit(self):
        audit = Audit(Path("/tmp/gold_v25_native/gold_cpu.so"), Path("/tmp/gold_v25_native/gold_cuda.so"))
        self.addCleanup(audit.close)
        return audit

    def test_route_integrity(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)
            a = generate(target)
            first = {p.name: p.read_bytes() for p in target.iterdir()}
            self.assertEqual(a, generate(target))
            self.assertEqual(first, {p.name: p.read_bytes() for p in target.iterdir()})
            for name in ("routes.bkv6", "arity.bin", "controller.bkgc", "controller.json"):
                self.assertEqual(first[name], (ASSETS / name).read_bytes())
            self.assertEqual([r["actions"] for r in a["routes"]], [719] * 4)
            self.assertEqual(overlay_source(source()), (HERE / "overlay.py").read_text())

    def test_routing_and_reset_boundaries(self):
        audit = self.audit()
        for yarn in (False, True):
            for egg in (9889, 9888):
                for seat in (0, 1):
                    for step in (0, 143, 144, 145, 647, 648, 649):
                        obs = observation(step, seat, yarn if step <= 144 else not yarn, egg if step <= 648 else 10000)
                        audit(obs)
                        st = audit.ref["_IMPL"].chassis.players[seat]
                        expected = 0 if step < 144 else int(yarn) if step < 648 else 2 if egg <= 9888 else 3
                        self.assertEqual(st["route"], expected)
                    # Repeat and backward steps each reset both routing latches.
                    audit(observation(649, seat, not yarn, 9889))
                    self.assertEqual(audit.cpu.state["route"], 3)
                    audit(observation(145, seat, yarn))
                    self.assertFalse(audit.cpu.state["day27"])
                    audit(observation(0, seat))
                    self.assertFalse(audit.cpu.state["day6"])
        # Missing EGG differs from BKG's generic zero default; YARN alias is not accepted.
        obs = observation(648)
        obs["market"]["inventory"] = {}
        obs["town"]["unlocked_shops"] = ["YARN"]
        audit(obs)
        self.assertEqual(audit.cpu.state["route"], 3)

    def test_active_reactive_layers(self):
        tapes = routes(source())
        units = lambda a: [a["farmer"]] + a["hands"]
        audit = self.audit()
        for op in ("PLANT", "BUILD_COOP", "BUILD_PASTURE"):
            audit(observation(0))
            witness = next((step, i) for step, a in enumerate(tapes[0][:646]) for i, u in enumerate(units(a))
                           if u[0] == op and i < len(units(tapes[0][step + 1])) and units(tapes[0][step + 1])[i][0] in ("WATER", "PLACE", "PASS", "CARE", "HARVEST")
                           and (op != "PLANT" or (i < len(units(tapes[0][step + 2])) and units(tapes[0][step + 2])[i][0] not in ("NORTH", "SOUTH", "EAST", "WEST"))))
            step, index = witness
            obs = observation(step)
            obs["farms"][0]["tiles"][4][4] = {"kind": "WEED"}
            result = audit(obs)
            self.assertEqual(units(result)[index], ["DIG"])
            self.assertTrue(audit.ref["_IMPL"].chassis.players[0]["pending"])
            # Empty target and the donor's next action exercise queued replay.
            obs = observation(step + 1)
            replay = audit(obs)
            self.assertEqual(units(replay)[index], units(tapes[0][step])[index])
        witness = next((step, o[1]) for step in range(1, 647) if step % 4 and (step + 1) % 72
                       for o in tapes[0][step + 1]["market"] if o and o[0] == "SELL" and o[1] not in ("WHEAT", "FERTILIZER")
                       and not any(x and x[:2] == ["SELL", o[1]] for x in tapes[0][step]["market"]))
        step, item = witness
        audit(observation(0))
        obs = observation(step)
        obs["private"]["shed"] = {item: 20}
        audit(obs)
        self.assertEqual(audit.cpu.state["due"], step + 1)
        led = audit.cpu.state["suppress"][item]
        result = audit(observation(step + 1))
        self.assertTrue(any(o and o[:2] == ["SELL", item] for o in result["market"]))
        before = sum(o[2] for o in tapes[0][step + 1]["market"] if o and o[:2] == ["SELL", item])
        after = sum(o[2] for o in result["market"] if o and o[:2] == ["SELL", item])
        self.assertEqual(after, before - led)
        for boundary in (4, 71, 143, 647, 718):
            obs = observation(boundary)
            obs["private"]["shed"] = {item: 20}
            audit(obs)
            self.assertEqual(audit.cpu.state["due"], -1)
        obs = observation(718)
        obs["private"]["shed"] = {"WHEAT": 90}
        obs["private"]["inventories"][0] = {"WOOL": 8, "MILK": 8}
        obs["private"]["inventories"][1] = {"EGG": 0}
        result = audit(obs)
        self.assertEqual(result["farmer"], ["DROP"])
        self.assertEqual(result["hands"][0], ["DROP"])
        self.assertEqual(result["market"], [["SELL", "WHEAT", 90], ["SELL", "WOOL", 8], ["SELL", "MILK", 2]])
        obs["step"] = 719
        self.assertEqual(audit(obs), result)


if __name__ == "__main__":
    unittest.main()
