"""Legal Python integration of the frozen Phase-1 BKG4 repair shortlist."""
from __future__ import annotations

import copy
import json

from kaggriculture_bonsai.agents.shop_router_0909.plus.agent import ACTIVE, ShopRouter0909Plus
from kaggriculture_bonsai.agents.shop_router_0909.reference_port.agent import FarmView
from kaggriculture_bonsai.runtime import bonsai_kg_v4 as bkg


class Phase1LegalAgent:
    """A pure legal-state BKG4 gate over an unchanged parent instance.

    The gate has no persistent state. PARENT is always ticked before the action
    override, including while the repair is active, so fallback preserves queues
    and advance-sale suppression exactly.
    """

    def __init__(self, gate):
        self.parent = ShopRouter0909Plus(json.loads(ACTIVE.read_text())["candidate"])
        self.compiled = bkg.load_compiled(gate)
        if len(self.compiled.bb) != 1 or self.compiled.state_count:
            raise ValueError("Phase-1 gate must have one output and no persistent node state")
        self.activation_turns = 0

    def __call__(self, observation, configuration=None):
        action = self.parent(observation, configuration)
        vm = bkg.VM(self.compiled)
        result = vm.tick(bkg.SyntheticObservation({(bkg.FEATURES["STEP"], 0): int(observation["step"])}),
                         max_market_orders=0)
        if result.fault:
            raise RuntimeError("Phase-1 gate fault")
        mode = int(result.blackboard[0])
        if not mode:
            return action
        if mode not in (1, 6):
            raise ValueError("mode was not legalized in Phase 1")
        self.activation_turns += 1
        state = copy.copy(self.parent.players[int(observation["player"])])
        state.endgame = 4 if mode == 1 else 8
        return self.parent._repairs(action, FarmView(observation), state,
                                    self.parent.tapes[state.plan], int(observation["step"]))
