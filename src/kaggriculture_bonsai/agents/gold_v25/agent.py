"""BKG4 controller, native route pack and the frozen donor overlays."""
import json

from kaggriculture_bonsai.runtime import bonsai_kg_v4 as bkg
from . import overlay as donor
from .extract import ASSETS, sha
from .pack import GoldRoutePack, Tape

SETTINGS = {"hand_align": True, "weed_repair": True, "sell_lead": True,
            "front_run": False, "budget_guard": False, "room_guard": False,
            "clamp_sells": False, "dead_stock": False, "terminal_liquidation": False}


class GoldV25Agent:
    def __init__(self, assets=ASSETS):
        identity = json.loads((assets / "identity.json").read_text())
        arity = (assets / "arity.bin").read_bytes()
        if sha(arity) != identity["arity_sha256"] or sha((assets / "controller.bkgc").read_bytes()) != identity["controller_sha256"]:
            raise ValueError("gold-v25 candidate hash mismatch")
        self.pack = GoldRoutePack(assets / "routes.bkv6", identity, arity)
        self.controller = bkg.load_compiled(assets / "controller.bkgc")
        self.vms = {}
        self.impl = donor.make_agent({}, router=self.router, **SETTINGS)
        self.chassis = self.impl.chassis
        self.chassis.routes = {rid: Tape(self.pack, rid) for rid in range(4)}
        self.chassis.diagnostics["terminal_rescue_errors"] = 0

    def router(self, observation, step, state):
        player = donor._int(donor._get(observation, "player", 0))
        prior = self.vms.get(player)
        if prior is None or prior[0] is not state:
            prior = (state, bkg.VM(self.controller))
            self.vms[player] = prior
        shops = donor._get(donor._get(observation, "town", {}), "unlocked_shops", []) or []
        inventory = donor._get(donor._get(observation, "market", {}), "inventory", {}) or {}
        # The donor defaults missing EGG to 10000 and accepts only literal YARN_STORE.
        # Supply its exact current-observation interpretation, not BKG's alias mapping.
        obs = bkg.SyntheticObservation({
            (bkg.FEATURES["STEP"], 0): step,
            (bkg.FEATURES["SHOP_SEEN"], 7): int("YARN_STORE" in shops),
            (bkg.FEATURES["MARKET_INVENTORY"], bkg.ITEM_ID["EGG"]): donor._int(donor._get(inventory, "EGG", 10000)),
        })
        result = prior[1].tick(obs, max_market_orders=0)
        if result.fault or result.status != bkg.STATUS_SUCCESS:
            raise RuntimeError("gold-v25 controller fault")
        route, day6, day27 = result.blackboard
        if day6:
            state["route"], state["day6"] = route, True
        if day27:
            state["day27"] = True
        return route

    def __call__(self, observation, configuration=None):
        try:
            action = self.impl(observation, configuration)
        except Exception:
            action = {"farmer": ["PASS"], "hands": [], "market": []}
        try:
            if donor._step_of(observation) >= 718:
                view = donor._View(observation, donor._int(donor._get(observation, "player", 0)), self.chassis.cfg)
                units = [["DROP"] if donor._shed_adjacent(pos, view.board) and view.inv(i) else ["PASS"] for i, pos in enumerate(view.positions)]
                action = {"farmer": units[0], "hands": units[1:], "market": []}
                projected = self.chassis._projected_shed(action, view)
                action["market"] = [["SELL", item, projected.get(item, 0)] for item in donor.PRODUCTS if projected.get(item, 0) > 0]
                action["market"].sort(key=lambda o: -view.prices.get(o[1], 0) * o[2])
        except Exception:
            self.chassis.diagnostics["terminal_rescue_errors"] += 1
        return action


_DEFAULT = None


def agent(observation, configuration=None):
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = GoldV25Agent()
    return _DEFAULT(observation, configuration)
