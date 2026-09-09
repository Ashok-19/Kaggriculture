"""Existing BKV6 records plus donor-only output spelling metadata.

The donor tapes carry their own hand count per step and keep no-op market slots
as `[]`. `raw_action` therefore returns the tape entry verbatim rather than the
chassis-resized view produced by `RoutePack.action`.
"""
import struct

from kaggriculture_bonsai.runtime.route_pack import RoutePack

ROUTES = 13
STEPS = 719


class ShopRoutePack(RoutePack):
    def __init__(self, path, identity, arity):
        super().__init__(path, identity)
        self.arity = arity
        if len(arity) != 2 * ROUTES * STEPS:
            raise ValueError("invalid shop-router arity table")

    @staticmethod
    def _order_obj(op, item, n):
        return [] if op == 0 else RoutePack._order_obj(op, item, n)

    def raw_action(self, route, step):
        if not 0 <= step < STEPS:
            raise IndexError(step)
        saved = self._record(route, step)[0]
        action = super().action(route, step, saved)
        short = struct.unpack_from("<H", self.arity, 2 * (route * STEPS + step))[0]
        for i, unit in enumerate([action["farmer"]] + action["hands"]):
            if short & (1 << i):
                del unit[2:]
        return action


class Tape:
    """Read-only sequence view of one packed route."""

    def __init__(self, pack, route):
        self.pack, self.route = pack, route

    def __len__(self):
        return STEPS

    def __getitem__(self, step):
        if not 0 <= step < STEPS:
            raise IndexError(step)
        return self.pack.raw_action(self.route, step)
