from __future__ import annotations

import struct

LEAVES = ("A1", "A2", "B", "C1", "C2", "D1", "D2", "E", "F", "G", "H", "I", "J1", "J2", "K")
LEAF_ID = {name: index for index, name in enumerate(LEAVES)}
LEAF_NONE = 255
MAX_SAVED_HANDS = 12
MAX_MARKET = 10
PACK_MAGIC = b"BKV6RP1\0"
PACK_VERSION = 1
HEADER = struct.Struct("<8sHHI32s")
DESC = struct.Struct("<HHI")
UNIT = struct.Struct("<BBh")
ORDER = struct.Struct("<BBH")
ACTION_SIZE = 2 + UNIT.size * (1 + MAX_SAVED_HANDS) + ORDER.size * MAX_MARKET
