from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

HEADER = struct.Struct("<8sIIIIII")
RESULT = struct.Struct("<iiHH")
MAGIC = b"B4OR1\0\0\0"
VERSION = 1
BRANCH_U = 1
BRANCH_W = 2
BRANCH_M = 3


@dataclass(frozen=True)
class OracleFile:
    branch: int
    candidates: int
    cases: int
    probes: int
    case_offset: int
    probe_offset: int
    rows: list[list[list[tuple[int, int, int, int]]]]


def load(path: str | Path) -> OracleFile:
    raw = Path(path).read_bytes()
    if len(raw) < HEADER.size:
        raise ValueError(f"short oracle result: {path}")
    magic, version, branch, ncand, ncases, nprobes, packed_offsets = HEADER.unpack_from(raw, 0)
    if magic != MAGIC or version != VERSION or branch not in (BRANCH_U, BRANCH_W, BRANCH_M):
        raise ValueError(f"bad oracle header: {path}")
    case_offset = packed_offsets >> 16
    probe_offset = packed_offsets & 0xFFFF
    expected = HEADER.size + ncand * ncases * nprobes * RESULT.size
    if len(raw) != expected:
        raise ValueError(f"oracle size mismatch {len(raw)} != {expected}: {path}")
    off = HEADER.size
    rows = []
    for _case in range(ncases):
        prows = []
        for _probe in range(nprobes):
            crows = []
            for _cand in range(ncand):
                crows.append(RESULT.unpack_from(raw, off))
                off += RESULT.size
            prows.append(crows)
        rows.append(prows)
    return OracleFile(branch, ncand, ncases, nprobes, case_offset, probe_offset, rows)


def u_candidate_name(i: int) -> str:
    local = ("HARVEST", "WATER", "FEED", "CARE", "COLLECT_FERT", "FERTILIZE")
    market = ("HOLD_SELL", "HOLD_HIRE", "HOLD_LAND", "HOLD_BUYS", "HOLD_ALL_MARKET", "SELLS_FIRST", "BUYS_FIRST", "REVERSE_MARKET")
    if i == 0:
        return "BASELINE"
    j = i - 1
    if j < 13 * len(local):
        return f"UNIT{j // len(local)}_{local[j % len(local)]}"
    return market[j - 13 * len(local)]


def w_candidate_name(i: int) -> str:
    if i == 0:
        return "BASELINE"
    k = i - 1
    pairs = []
    for a in range(13):
        for b in range(a + 1, 13):
            pairs.append((a, b))
    if k < len(pairs):
        a, b = pairs[k]
        return f"SWAP_{a}_{b}"
    if k == len(pairs):
        return "ROTATE_LEFT"
    if k == len(pairs) + 1:
        return "ROTATE_RIGHT"
    return f"W_UNKNOWN_{i}"


def m_candidate_name(i: int, routes: tuple[int, ...], horizons: tuple[int, ...]) -> str:
    if i == 0:
        return "BASELINE"
    k = i - 1
    r = routes[k // len(horizons)]
    h = horizons[k % len(horizons)]
    return f"ROUTE_{r}_H{h}"
