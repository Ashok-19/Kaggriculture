from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

HEADER = struct.Struct("<8sIII")
RECORD = struct.Struct("<iiHHHHQ")
MAGIC = b"B4EC1\0\0\0"
VERSION = 1


@dataclass(frozen=True)
class EditCudaResult:
    own_money: int
    opp_money: int
    action_changes: int
    activations: int
    flags: int
    action_trace_hash: int

    @property
    def margin(self) -> int:
        return int(self.own_money) - int(self.opp_money)


def load_edit_results(path: str | Path) -> list[list[EditCudaResult]]:
    raw = Path(path).read_bytes()
    if len(raw) < HEADER.size:
        raise ValueError("edit CUDA result header is truncated")
    magic, version, genomes, cases = HEADER.unpack_from(raw, 0)
    if magic != MAGIC or version != VERSION or genomes < 1 or cases < 1:
        raise ValueError(f"invalid edit CUDA result header: magic={magic!r} version={version} genomes={genomes} cases={cases}")
    expected = HEADER.size + int(genomes) * int(cases) * RECORD.size
    if len(raw) != expected:
        raise ValueError(f"edit CUDA result size mismatch: {len(raw)} != {expected}")
    out: list[list[EditCudaResult]] = []
    off = HEADER.size
    for gi in range(int(genomes)):
        row: list[EditCudaResult] = []
        for ci in range(int(cases)):
            own, opp, changes, activations, flags, reserved, trace_hash = RECORD.unpack_from(raw, off)
            off += RECORD.size
            if reserved:
                raise ValueError(f"reserved edit CUDA result bits are nonzero at genome={gi} case={ci}: {reserved}")
            if flags:
                raise ValueError(f"edit CUDA result flags={flags} at genome={gi} case={ci}")
            row.append(EditCudaResult(
                int(own), int(opp), int(changes), int(activations), int(flags), int(trace_hash)
            ))
        out.append(row)
    return out
