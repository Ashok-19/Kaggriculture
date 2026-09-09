"""Reinforced candidate: the exact parent plus bounded Bonsai specialists.

The candidate keeps the parent's whole pipeline. Two things may differ, and both
are decided by one compact BKG4 controller reading only legal observable state:

* `routes` - the plan chosen from the ordered first two unlocked shops at step
  144. A different donor tape is a genuinely distinct strategic schedule, so it
  stays in the existing route pack rather than in controller nodes.
* `endgame` - a step-windowed gate enabling bounded terminal repairs on top of
  the parent's own action.

With the donor route table and no gate the candidate is the parent, action for
action; `plus.test_plus` pins that fallback.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

from kaggriculture_bonsai.runtime import bonsai_kg_v4 as bkg

from ..reference_port import agent as parent
from ..reference_port.extract import ROUTE_STEP, FINAL_PLAN_STEP, YARN_FIRST, YARN_SECOND

LEAD_TWO, FORCE_DROP, FULL_SELL, VALUE_SORT, APPROACH = 1, 2, 4, 8, 16
DONOR_ROUTES = {(name, "YARN_STORE"): plan for name, plan in YARN_SECOND.items()}
DONOR_ROUTES.update({("YARN_STORE", name): plan for name, plan in YARN_FIRST.items()})


def route_table(overrides=None):
    """Donor shop-pair mapping with searched overrides applied on top."""
    table = dict(DONOR_ROUTES)
    for key, plan in (overrides or {}).items():
        pair = tuple(key.split("|")) if isinstance(key, str) else tuple(key)
        table[pair] = int(plan)
    return {pair: plan for pair, plan in table.items() if plan}


def endgame_rules(endgame):
    """One rule, a list of rules, or nothing. The first matching rule wins."""
    if not endgame:
        return []
    return [endgame] if isinstance(endgame, dict) else list(endgame)


def controller(table, endgame=None):
    """Grouped two-shop dispatch, plus an optional step-windowed gate."""
    def cond(lhs, op, rhs):
        return {"type": "Condition", "lhs": lhs, "op": op, "rhs": rhs}
    def shop(index):
        return {"feature": "SHOP_AT", "arg": index}
    grouped = {}
    for (first, second), plan in sorted(table.items()):
        grouped.setdefault(first, []).append((second, plan))
    children = []
    for first, entries in grouped.items():
        inner = {"type": "Selector", "children": [
            {"type": "Sequence", "children": [cond(shop(1), "EQ", bkg.SHOP_ID[second]),
                                              {"type": "SetBlackboard", "slot": "route", "value": plan}]}
            for second, plan in entries]}
        children.append({"type": "Sequence", "children": [cond(shop(0), "EQ", bkg.SHOP_ID[first]), inner]})
    children.append({"type": "SetBlackboard", "slot": "route", "value": 0})
    tree = [
        {"type": "If", "condition": cond({"feature": "STEP"}, "EQ", ROUTE_STEP),
         "then": {"type": "Selector", "children": children}, "else": {"type": "Success"}},
        {"type": "If", "condition": cond({"feature": "STEP"}, "EQ", FINAL_PLAN_STEP),
         "then": {"type": "SetBlackboard", "slot": "route", "value": 2}, "else": {"type": "Success"}},
    ]
    blackboard = [{"name": "route", "type": "I32", "initial": 0}]
    rules = endgame_rules(endgame)
    if rules:
        blackboard.append({"name": "endgame", "type": "I32", "initial": 0})
        arms = []
        for rule in rules:
            guard = [cond({"feature": "STEP"}, "GE", int(rule["start"])),
                     cond({"feature": "STEP"}, "LE", int(rule["end"]))]
            allowed = [int(r) for r in rule.get("routes") or ()]
            if allowed:
                # One repair per strategic regime; the route is already latched.
                guard.append({"type": "Selector", "children": [
                    cond({"blackboard": "route"}, "EQ", route) for route in allowed]})
            blocked = [int(r) for r in rule.get("exclude_routes") or ()]
            if blocked:
                guard.append({"type": "Invert", "child": {"type": "Selector", "children": [
                    cond({"blackboard": "route"}, "EQ", route) for route in blocked]}})
            guard.append({"type": "SetBlackboard", "slot": "endgame", "value": int(rule["mode"])})
            arms.append({"type": "Sequence", "children": guard})
        arms.append({"type": "SetBlackboard", "slot": "endgame", "value": 0})
        tree.append({"type": "Selector", "children": arms})
    return {"schema_version": 4, "name": "BKG4_SHOP_ROUTER_0909_PLUS",
            "blackboard": blackboard, "tree": {"type": "Sequence", "children": tree}}


class PlusState(parent.DayState):
    """Parent state; `endgame` is read back from the same controller tick."""

    def __init__(self, compiled):
        super().__init__(compiled)
        self.endgame = 0


class ShopRouter0909Plus:
    """Parent pipeline under a searched route table and a bounded terminal gate."""

    def __init__(self, spec):
        self.spec = spec if isinstance(spec, dict) else json.loads(Path(spec).read_text())
        self.parent = parent.ShopRouter0909Agent()
        self.table = route_table(self.spec.get("routes"))
        self.endgame = self.spec.get("endgame")
        self.compiled = bkg.compile_policy(controller(self.table, self.endgame))
        self.tapes = self.parent.tapes
        self.players = {}

    @property
    def identity(self):
        return {"nodes": len(self.compiled.nodes), "blackboard": len(self.compiled.bb),
                "state_slots": self.compiled.state_count,
                "semantic_hash": self.compiled.semantic_hash,
                "routes": {"|".join(k): v for k, v in sorted(self.table.items())},
                "endgame": self.endgame}

    def _tick(self, observation, step, state):
        shops = observation["town"]["unlocked_shops"]
        obs = bkg.SyntheticObservation({
            (bkg.FEATURES["STEP"], 0): step,
            (bkg.FEATURES["SHOP_AT"], 0): bkg.SHOP_ID.get(str(shops[0]), -1) if len(shops) > 0 else -1,
            (bkg.FEATURES["SHOP_AT"], 1): bkg.SHOP_ID.get(str(shops[1]), -1) if len(shops) > 1 else -1,
        })
        result = state.vm.tick(obs, max_market_orders=0)
        if result.fault or result.status != bkg.STATUS_SUCCESS:
            raise RuntimeError("shop-router-0909 plus controller fault")
        board = result.blackboard
        state.endgame = int(board[1]) if len(board) > 1 else 0
        return int(board[0])

    def _repairs(self, action, view, state, tape, step):
        """Bounded terminal repairs, applied only inside the gate."""
        mode = state.endgame
        if mode & APPROACH:
            # Carried stock only becomes money once it reaches the shed, so a
            # holder that is still off the shed squares steps toward them.
            units = [action["farmer"], *action["hands"]]
            centre = len(view.tiles) // 2
            for worker in range(min(len(units), len(view.positions))):
                position = view.positions[worker]
                if view.beside_shed(position) or not view.inventory(worker):
                    continue
                x, y = int(position[0]), int(position[1])
                if x < centre - 1:
                    units[worker] = ["EAST"]
                elif x > centre:
                    units[worker] = ["WEST"]
                elif y < centre - 1:
                    units[worker] = ["SOUTH"]
                elif y > centre:
                    units[worker] = ["NORTH"]
            action["farmer"], action["hands"] = units[0], units[1:]
        if mode & FORCE_DROP:
            units = [action["farmer"], *action["hands"]]
            for worker in range(min(len(units), len(view.positions))):
                if view.beside_shed(view.positions[worker]) and view.inventory(worker):
                    units[worker] = ["DROP"]
            action["farmer"], action["hands"] = units[0], units[1:]
        stock = parent.projected_shed(action, view)
        if mode & LEAD_TWO and step + 2 <= parent.LAST_STEP and (step + 2) % 72 and step % 4:
            planned, already = {}, {order[1] for order in action["market"]
                                   if order and order[0] == "SELL" and len(order) > 1}
            for order in tape[step + 2].get("market") or []:
                if order and order[0] == "SELL" and len(order) >= 3 and order[1] in parent.PRODUCTS:
                    planned[order[1]] = planned.get(order[1], 0) + max(0, int(order[2]))
            for item in parent.PRODUCTS:
                if item in ("WHEAT", "FERTILIZER") or item in already:
                    continue
                quantity = min(stock.get(item, 0), planned.get(item, 0))
                if quantity <= 0 or int(view.prices.get(item, 0)) < 2 or len(action["market"]) >= parent.MAX_ORDERS:
                    continue
                action["market"].append(["SELL", item, quantity])
                state.advanced_sales[item] = state.advanced_sales.get(item, 0) + quantity
                state.sale_due_step = step + 1
        if mode & FULL_SELL and all(not order or order[0] == "SELL" for order in action["market"]):
            for item in parent.PRODUCTS:
                if item in ("WHEAT", "FERTILIZER") or stock.get(item, 0) <= 0 or int(view.prices.get(item, 0)) < 2:
                    continue
                existing = [o for o in action["market"] if o and o[0] == "SELL" and o[1] == item]
                if existing:
                    for order in existing:
                        order[2] = stock[item]
                elif len(action["market"]) < parent.MAX_ORDERS:
                    action["market"].append(["SELL", item, stock[item]])
        if mode & VALUE_SORT and all(not order or order[0] == "SELL" for order in action["market"]):
            action["market"].sort(key=lambda o: -int(view.prices.get(o[1], 0)) * int(o[2]) if o else 0)
        return action

    def act(self, observation):
        step, player = int(observation["step"]), int(observation["player"])
        state = self.players.get(player)
        if state is None or step <= state.last_step:
            state = self.players[player] = PlusState(self.compiled)
        state.last_step = step

        plan = self._tick(observation, step, state)
        view = parent.FarmView(observation)
        tape = self.tapes[plan]
        action = copy.deepcopy(tape[step])
        parent.repair_weeds(action, view, state, step)
        parent.subtract_advanced_sales(action, state, step)
        parent.advance_sales(action, view, state, tape, step)
        if state.endgame:
            action = self._repairs(action, view, state, tape, step)
        action["market"] = action["market"][:parent.MAX_ORDERS]
        return parent.liquidate(view) if step == parent.LAST_STEP else action

    __call__ = lambda self, observation, configuration=None: self.act(observation)


ACTIVE = Path(__file__).resolve().parents[4].parent / "workspace/shop_router_0909/active_candidate.json"
_DEFAULT = None


def agent(observation, configuration=None):
    """Entry for the explicitly selected isolated candidate, never the champion."""
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = ShopRouter0909Plus(json.loads(ACTIVE.read_text())["candidate"])
    return _DEFAULT.act(observation)
