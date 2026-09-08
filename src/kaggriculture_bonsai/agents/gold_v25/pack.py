"""Existing BKV6 records plus donor-only output spelling metadata."""
import struct

from kaggriculture_bonsai.runtime.route_pack import RoutePack


class GoldRoutePack(RoutePack):
    def __init__(self, path, identity, arity):
        super().__init__(path, identity)
        self.arity = arity
        if len(arity) != 4 * 719 * 2:
            raise ValueError("invalid gold-v25 arity table")

    @staticmethod
    def _order_obj(op, item, n):
        return [] if op == 0 else RoutePack._order_obj(op, item, n)

    def raw_action(self, route, step):
        if not 0 <= step < 719:
            return {"farmer": ["PASS"], "hands": [], "market": []}
        saved = self._record(route, step)[0]
        action = super().action(route, step, saved)
        short = struct.unpack_from("<H", self.arity, 2 * (route * 719 + step))[0]
        for i, unit in enumerate([action["farmer"]] + action["hands"]):
            if short & (1 << i):
                del unit[2:]
        return action


class Tape:
    def __init__(self, pack, route):
        self.pack, self.route = pack, route

    def __len__(self):
        return 719

    def __getitem__(self, step):
        if not 0 <= step < 719:
            raise IndexError(step)
        return self.pack.raw_action(self.route, step)
