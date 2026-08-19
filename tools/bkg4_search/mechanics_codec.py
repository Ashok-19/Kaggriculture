from __future__ import annotations

import hashlib
import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

MAGIC = b"B4MX1\0\0\0"
VERSION = 1
HEADER = struct.Struct("<8sII")
GENOME = struct.Struct("<BBBBHH")

MODE_BASE = 0
MODE_FEED_REWRITE = 1

REPL_PASS = 0
REPL_HARVEST = 1
REPL_COLLECT = 2
REPL_BEST = 3

FLAG_PROTECT_PRODUCTION = 1 << 0
FLAG_REQUIRE_NOT_CARED = 1 << 1
FLAG_REQUIRE_PENDING_ZERO = 1 << 2


@dataclass(frozen=True)
class MechanicGenome:
    mode: int = MODE_BASE
    animal_mask: int = 0
    replacement: int = REPL_PASS
    flags: int = 0
    wheat_cap: int = 0

    def pack(self) -> bytes:
        return GENOME.pack(self.mode, self.animal_mask, self.replacement, self.flags, self.wheat_cap, 0)

    def canonical(self) -> dict:
        return {
            "mode": self.mode,
            "animal_mask": self.animal_mask,
            "replacement": self.replacement,
            "flags": self.flags,
            "wheat_cap": self.wheat_cap,
        }

    def semantic_id(self) -> str:
        raw = json.dumps(self.canonical(), sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(raw).hexdigest()


def save_genomes(genomes: Iterable[MechanicGenome], path: Path) -> None:
    gs = list(genomes)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(HEADER.pack(MAGIC, VERSION, len(gs)) + b"".join(g.pack() for g in gs))


def grid() -> List[MechanicGenome]:
    out = [MechanicGenome()]
    # Each mutation is semantically local: alter an already-planned FEED on the
    # same animal tile only when the animal was fed on the previous day
    # (consecutive_unfed == 0), so one skipped day cannot itself cause escape.
    for animal_mask in range(1, 8):
        for replacement in (REPL_PASS, REPL_HARVEST, REPL_COLLECT, REPL_BEST):
            for flags in range(8):
                for wheat_cap in (0, 4, 8, 16, 32, 64):
                    out.append(MechanicGenome(MODE_FEED_REWRITE, animal_mask, replacement, flags, wheat_cap))
    return out


def neighborhood(g: MechanicGenome) -> List[MechanicGenome]:
    if g.mode == MODE_BASE:
        return grid()[1:65]
    cand = {g}
    for bit in (1, 2, 4):
        cand.add(MechanicGenome(g.mode, g.animal_mask ^ bit or g.animal_mask, g.replacement, g.flags, g.wheat_cap))
    for r in (REPL_PASS, REPL_HARVEST, REPL_COLLECT, REPL_BEST):
        cand.add(MechanicGenome(g.mode, g.animal_mask, r, g.flags, g.wheat_cap))
    for bit in (FLAG_PROTECT_PRODUCTION, FLAG_REQUIRE_NOT_CARED, FLAG_REQUIRE_PENDING_ZERO):
        cand.add(MechanicGenome(g.mode, g.animal_mask, g.replacement, g.flags ^ bit, g.wheat_cap))
    caps = (0, 4, 8, 16, 32, 64)
    if g.wheat_cap in caps:
        i = caps.index(g.wheat_cap)
        if i: cand.add(MechanicGenome(g.mode, g.animal_mask, g.replacement, g.flags, caps[i - 1]))
        if i + 1 < len(caps): cand.add(MechanicGenome(g.mode, g.animal_mask, g.replacement, g.flags, caps[i + 1]))
    return sorted(cand, key=lambda x: x.semantic_id())
