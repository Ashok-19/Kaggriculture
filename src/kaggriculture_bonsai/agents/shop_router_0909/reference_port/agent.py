"""Exact BKG4 port of the Shop Router 0909 notebook callable.

The donor's 13 complete 719-turn tapes live in the existing BKV6 route pack; the
donor's two-shop plan choice and its final continuation are the BKG4 controller.
Every remaining rule is the donor's own observation-based repair, transcribed
without simplification: same order, same fallbacks, same state lifetimes.
"""
import copy
import functools
import json
from collections import deque

from kaggriculture_bonsai.runtime import bonsai_kg_v4 as bkg

from .extract import ASSETS, sha
from .pack import ShopRoutePack, Tape

TURNS_PER_DAY = 24
LAST_STEP = 718
SHED_CAPACITY = 100
MAX_ORDERS = 10
PRODUCTS = (
    "WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON",
    "EGG", "MILK", "WOOL", "FERTILIZER",
)
WEED_BLOCKED_WORK = {"PLANT", "BUILD_COOP", "BUILD_PASTURE"}
ANIMALS = {"GOOSE", "COW", "SHEEP"}


class FarmView:
    """Only the current own farm, private inventory, and public prices."""

    def __init__(self, observation):
        farm = observation["farms"][observation["player"]]
        private = observation["private"]
        self.tiles = farm["tiles"]
        self.positions = [farm["farmer"], *farm["hands"]]
        self.inventories = private["inventories"]
        self.shed = {item: max(0, int(qty)) for item, qty in private["shed"].items()}
        self.prices = observation["market"]["prices"]

    def inventory(self, worker):
        return self.inventories[worker] if worker < len(self.inventories) else {}

    def beside_shed(self, position):
        center = len(self.tiles) // 2
        return position[0] in (center - 1, center) and position[1] in (center - 1, center)


class DayState:
    """Per-player memory; queues expire at dawn and sales expire next turn."""

    def __init__(self, controller):
        self.vm = bkg.VM(controller)
        self.last_step = -1
        self.day = -1
        self.queues = {}
        self.sale_due_step = -1
        self.advanced_sales = {}

    @property
    def plan(self):
        return int(self.vm.state.blackboard[0])


def repair_weeds(action, view, state, step):
    """Insert DIG without consuming the blocked action; shift only this worker."""
    day = step // TURNS_PER_DAY
    if day != state.day:
        state.day = day
        state.queues.clear()  # Unfinished work never spills into tomorrow.

    workers = [action.get("farmer") or ["PASS"], *(action.get("hands") or [])]
    for worker in range(min(len(workers), len(view.positions))):
        queue = state.queues.setdefault(worker, deque())
        queue.append(list(workers[worker]))
        x, y = view.positions[worker]
        tile = view.tiles[y][x]
        blocked = (queue[0][0] in WEED_BLOCKED_WORK
                   and isinstance(tile, dict) and tile.get("kind") == "WEED")
        workers[worker] = ["DIG"] if blocked else queue.popleft()
    action["farmer"], action["hands"] = workers[0], workers[1:]


def projected_shed(action, view):
    """Estimate stock after this turn's nearby PICKUP, DROP and PLACE actions."""
    stock = {item: view.shed.get(item, 0) for item in PRODUCTS}
    stock.update(view.shed)
    total = sum(stock.values())
    workers = [action.get("farmer") or ["PASS"], *(action.get("hands") or [])]
    for worker in range(min(len(workers), len(view.positions))):
        if not view.beside_shed(view.positions[worker]):
            continue
        work = workers[worker]
        operation = work[0] if work else "PASS"
        inventory = view.inventory(worker)
        if operation == "PICKUP" and len(work) >= 2 and work[1] in stock:
            quantity = max(0, int(work[2]) if len(work) >= 3 else 1)
            taken = min(stock[work[1]], quantity)
            stock[work[1]] -= taken
            total -= taken
        elif operation == "DROP":
            for item, held in inventory.items():
                added = min(max(0, int(held)), max(0, SHED_CAPACITY - total))
                if added > 0:
                    stock[item] = stock.get(item, 0) + added
                    total += added
        elif operation == "PLACE" and len(work) >= 2 and work[1] not in ANIMALS:
            item = work[1]
            quantity = max(0, int(work[2]) if len(work) >= 3 else 1)
            added = min(quantity, max(0, int(inventory.get(item, 0))),
                        max(0, SHED_CAPACITY - total))
            if added > 0:
                stock[item] = stock.get(item, 0) + added
                total += added
    return stock


def subtract_advanced_sales(action, state, step):
    """Remove quantities already requested one turn early, retaining order slots."""
    if state.sale_due_step == step:
        remaining = dict(state.advanced_sales)
        for order in action["market"]:
            if order and order[0] == "SELL" and len(order) >= 3:
                item = order[1]
                removed = min(max(0, int(order[2])), remaining.get(item, 0))
                if removed > 0:
                    order[2] = int(order[2]) - removed
                    remaining[item] -= removed
    state.advanced_sales = {}
    state.sale_due_step = -1


def advance_sales(action, view, state, tape, step):
    """Bring eligible sales from our next planned action forward by one turn."""
    next_step = step + 1
    if next_step > LAST_STEP or next_step % 72 == 0 or step % 4 == 0:
        return
    planned = {}
    for order in tape[next_step].get("market") or []:
        if order and order[0] == "SELL" and len(order) >= 3 and order[1] in PRODUCTS:
            item = order[1]
            planned[item] = planned.get(item, 0) + max(0, int(order[2]))
    already_selling = {order[1] for order in action["market"]
                       if order and order[0] == "SELL" and len(order) > 1}
    stock = projected_shed(action, view)
    for item in PRODUCTS:
        if item in ("WHEAT", "FERTILIZER") or item in already_selling:
            continue
        quantity = min(stock.get(item, 0), planned.get(item, 0))
        if quantity <= 0 or int(view.prices.get(item, 0)) < 2:
            continue
        if len(action["market"]) >= MAX_ORDERS:
            break
        action["market"].append(["SELL", item, quantity])
        state.advanced_sales[item] = quantity
    if state.advanced_sales:
        state.sale_due_step = next_step


def liquidate(view):
    """On the last turn, drop reachable inventory and sell the projected shed."""
    workers = [["DROP"] if view.beside_shed(pos) and view.inventory(worker) else ["PASS"]
               for worker, pos in enumerate(view.positions)]
    action = {"farmer": workers[0], "hands": workers[1:], "market": []}
    stock = projected_shed(action, view)
    action["market"] = [["SELL", item, stock[item]] for item in PRODUCTS if stock[item] > 0]
    action["market"].sort(key=lambda order: -int(view.prices.get(order[1], 0)) * order[2])
    return action


class ShopRouter0909Agent:
    """Frozen exact parent. Route from the BKG4 controller, tapes from the pack."""

    def __init__(self, assets=ASSETS):
        self.identity, self.pack, self.controller, self.tapes = _frozen_assets(assets)
        self.players = {}

    def route(self, observation, step, state):
        """Donor plan choice, evaluated by the controller on the donor's features."""
        shops = observation["town"]["unlocked_shops"]
        obs = bkg.SyntheticObservation({
            (bkg.FEATURES["STEP"], 0): step,
            (bkg.FEATURES["SHOP_AT"], 0): bkg.SHOP_ID.get(str(shops[0]), -1) if len(shops) > 0 else -1,
            (bkg.FEATURES["SHOP_AT"], 1): bkg.SHOP_ID.get(str(shops[1]), -1) if len(shops) > 1 else -1,
        })
        result = state.vm.tick(obs, max_market_orders=0)
        if result.fault or result.status != bkg.STATUS_SUCCESS:
            raise RuntimeError("shop-router-0909 controller fault")
        return state.plan

    def act(self, observation):
        step, player = int(observation["step"]), int(observation["player"])
        state = self.players.get(player)
        if state is None or step <= state.last_step:
            state = self.players[player] = DayState(self.controller)
        state.last_step = step

        plan = self.route(observation, step, state)
        view = FarmView(observation)
        tape = self.tapes[plan]
        action = copy.deepcopy(tape[step])
        repair_weeds(action, view, state, step)
        subtract_advanced_sales(action, state, step)
        advance_sales(action, view, state, tape, step)
        action["market"] = action["market"][:MAX_ORDERS]
        return liquidate(view) if step == LAST_STEP else action

    __call__ = lambda self, observation, configuration=None: self.act(observation)


@functools.lru_cache(maxsize=2)
def _frozen_assets(assets):
    """Hash-checked immutable assets. Read-only, so instances may share them."""
    identity = json.loads((assets / "identity.json").read_text())
    arity = (assets / "arity.bin").read_bytes()
    if sha(arity) != identity["arity_sha256"]:
        raise ValueError("shop-router-0909 candidate hash mismatch")
    if sha((assets / "controller.bkgc").read_bytes()) != identity["controller_sha256"]:
        raise ValueError("shop-router-0909 candidate hash mismatch")
    pack = ShopRoutePack(assets / "routes.bkv6", identity, arity)
    controller = bkg.load_compiled(assets / "controller.bkgc")
    if controller.semantic_hash != identity["controller_semantic_hash"]:
        raise ValueError("shop-router-0909 controller semantic hash mismatch")
    return identity, pack, controller, [Tape(pack, rid) for rid in range(13)]


_DEFAULT = None


def agent(observation, configuration=None):
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = ShopRouter0909Agent()
    return _DEFAULT.act(observation)
