"""Exact-port integrity: identity, route dispatch, donor parity, CPU parity."""
import contextlib
import copy
import io
import itertools
import json
import unittest
from pathlib import Path

from kaggriculture_bonsai.runtime import bonsai_kg_v4 as bkg

from . import extract
from .agent import ShopRouter0909Agent
from .native import BUILD, NativeAgent, state_summary
from .pack import ROUTES, STEPS

IDENTITY = json.loads((extract.ASSETS / "identity.json").read_text())


def _engine():
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        from kaggle_environments import make
    return make


class TestIdentity(unittest.TestCase):
    def test_pinned_donor_hashes(self):
        self.assertEqual(extract.sha(extract.NOTEBOOK.read_bytes()), extract.NOTEBOOK_SHA256)
        self.assertEqual(extract.sha(extract.source().encode()), extract.SOURCE_SHA256)

    def test_assets_match_identity(self):
        for name, key in (("controller.bkgc", "controller_sha256"), ("arity.bin", "arity_sha256")):
            self.assertEqual(extract.sha((extract.ASSETS / name).read_bytes()), IDENTITY[key])
        self.assertEqual(extract.sha((extract.ASSETS / "routes.bkv6").read_bytes()),
                         IDENTITY["route_pack"]["pack_sha256"])

    def test_controller_footprint(self):
        self.assertEqual(IDENTITY["controller_state_slots"], 0)
        self.assertEqual(IDENTITY["controller_blackboard_slots"], 1)
        self.assertEqual(IDENTITY["controller_nodes"], 61)

    def test_route_pack_is_lossless(self):
        agent = ShopRouter0909Agent()
        donor = extract.tapes()
        self.assertEqual(len(donor), ROUTES)
        for route in range(ROUTES):
            self.assertEqual([agent.tapes[route][step] for step in range(STEPS)], donor[route])


class TestControllerDispatch(unittest.TestCase):
    """Every ordered shop pair, plus the short and unknown-shop fall-throughs."""

    def setUp(self):
        self.compiled = bkg.load_compiled(extract.ASSETS / "controller.bkgc")

    def _route(self, step, first, second):
        vm = bkg.VM(self.compiled)
        obs = bkg.SyntheticObservation({(bkg.FEATURES["STEP"], 0): step,
                                        (bkg.FEATURES["SHOP_AT"], 0): first,
                                        (bkg.FEATURES["SHOP_AT"], 1): second})
        result = vm.tick(obs, max_market_orders=0)
        self.assertFalse(result.fault)
        return int(vm.state.blackboard[0])

    def test_every_ordered_pair_matches_the_donor_table(self):
        donor = {}
        donor.update({(name, "YARN_STORE"): plan for name, plan in extract.YARN_SECOND.items()})
        donor.update({("YARN_STORE", name): plan for name, plan in extract.YARN_FIRST.items()})
        for first, second in itertools.product(bkg.SHOPS, bkg.SHOPS):
            with self.subTest(pair=(first, second)):
                self.assertEqual(self._route(144, bkg.SHOP_ID[first], bkg.SHOP_ID[second]),
                                 donor.get((first, second), 0))

    def test_missing_and_unknown_shops_keep_plan_zero(self):
        for first, second in ((-1, -1), (-1, 7), (7, -1), (0, -1)):
            self.assertEqual(self._route(144, first, second), 0)

    def test_latch_steps_are_exact(self):
        self.assertEqual(self._route(143, 0, 7), 0)
        self.assertEqual(self._route(145, 0, 7), 0)
        self.assertEqual(self._route(648, 7, 4), 2)


class TestDonorParity(unittest.TestCase):
    """One full official game per seat: donor, Python port and CPU port agree."""

    def test_action_and_state_equality(self):
        make = _engine()
        library = BUILD / "shop_cpu.so"
        if not Path(library).exists():
            self.skipTest("native CPU library not built")
        for seat in (0, 1):
            reference = extract.reference()
            port, native = ShopRouter0909Agent(), NativeAgent(library)
            steps = 0
            try:
                def paired(obs, cfg=None):
                    nonlocal steps
                    expected = reference["agent"](copy.deepcopy(obs), cfg)
                    player = int(obs["player"])
                    summary = state_summary(reference["_POLICY"].players[player])
                    self.assertEqual(port(copy.deepcopy(obs), cfg), expected)
                    self.assertEqual(state_summary(port.players[player]), summary)
                    self.assertEqual(native(copy.deepcopy(obs), cfg), expected)
                    self.assertEqual(native.state, summary)
                    steps += 1
                    return expected
                agents = ["random", "random"]
                agents[seat] = paired
                env = make("kaggriculture", configuration={"episodeSteps": 720, "seed": 4242}, debug=True)
                env.run(agents)
            finally:
                native.close()
            self.assertEqual(steps, STEPS)


if __name__ == "__main__":
    unittest.main()
