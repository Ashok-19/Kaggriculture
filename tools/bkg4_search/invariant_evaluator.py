from __future__ import annotations

import os
import struct
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from tools.bkg4_search.edit_search.fitness import StateEvaluation
from tools.bkg4_search.edit_search.panels import CaseRecord
from tools.bkg4_search.edit_search.schema import ParentProfile
from tools.bkg4_search.invariant_module import InvariantStrategicModule, pack_file

_HEADER = struct.Struct("<8sIII")
_RECORD = struct.Struct("<iiHHHHQ")
_MAGIC = b"B4IC1\0\0\0"
_VERSION = 1


@dataclass(frozen=True)
class InvariantCudaResult:
    own_money: int
    opp_money: int
    action_changes: int
    activations: int
    flags: int
    action_trace_hash: int

    @property
    def margin(self) -> int:
        return int(self.own_money) - int(self.opp_money)


def load_invariant_results(path: str | Path) -> list[list[InvariantCudaResult]]:
    raw = Path(path).read_bytes()
    if len(raw) < _HEADER.size:
        raise ValueError("invariant CUDA result header is truncated")
    magic, version, candidates, cases = _HEADER.unpack_from(raw, 0)
    if magic != _MAGIC or version != _VERSION or candidates < 1 or cases < 1:
        raise ValueError(
            f"invalid invariant CUDA result header: magic={magic!r} version={version} candidates={candidates} cases={cases}"
        )
    expected = _HEADER.size + int(candidates) * int(cases) * _RECORD.size
    if len(raw) != expected:
        raise ValueError(f"invariant CUDA result size mismatch: {len(raw)} != {expected}")
    out: list[list[InvariantCudaResult]] = []
    off = _HEADER.size
    for gi in range(int(candidates)):
        row: list[InvariantCudaResult] = []
        for ci in range(int(cases)):
            own, opp, changes, activations, flags, reserved, trace_hash = _RECORD.unpack_from(raw, off)
            off += _RECORD.size
            if reserved:
                raise ValueError(f"reserved invariant CUDA result bits are nonzero at candidate={gi} case={ci}: {reserved}")
            if flags:
                raise ValueError(f"invariant CUDA result flags={flags} at candidate={gi} case={ci}")
            row.append(InvariantCudaResult(
                int(own), int(opp), int(changes), int(activations), int(flags), int(trace_hash)
            ))
        out.append(row)
    return out


class InvariantExactCudaEvaluator:
    """Batched exact evaluator for compact `B4IM1` strategic modules.

    There is intentionally no CPU game-evaluation fallback. CPU C++ exists only for
    differential parity on bounded panels; production truth comes from the resident CUDA
    candidate×case evaluator.
    """

    RESULT_ABI = "B4IC1/v1"
    CANDIDATE_ABI = "B4IM1/v1"

    def __init__(
        self,
        profile: ParentProfile,
        *,
        binary: str | Path,
        device: int = 0,
        tpb: int = 64,
        memory_fraction: float = 0.70,
        cuda_visible_devices: str | None = None,
    ):
        self.profile = profile
        self.binary = Path(binary).resolve()
        self.device = int(device)
        self.tpb = int(tpb)
        self.memory_fraction = float(memory_fraction)
        self.cuda_visible_devices = cuda_visible_devices
        if self.device < 0 or self.tpb <= 0 or not (0.10 <= self.memory_fraction <= 0.90):
            raise ValueError("invalid invariant CUDA evaluator configuration")
        if int(profile.bkg4_abi) != 4 or int(profile.feature_registry) != 1:
            raise ValueError("B4IM1/v1 requires BKG4 ABI 4 and feature registry v1")

    def evaluate(
        self,
        modules: Sequence[InvariantStrategicModule],
        panel: Sequence[CaseRecord],
        *,
        work_dir: str | Path,
        panel_id: str,
    ) -> dict[str, StateEvaluation]:
        if not self.binary.is_file():
            raise FileNotFoundError(f"exact B4IC1 CUDA evaluator binary not found: {self.binary}")
        if not modules:
            return {}
        if not panel:
            raise ValueError("exact invariant CUDA evaluation requires a non-empty case panel")
        semantic_ids = [m.semantic_id() for m in modules]
        if len(set(semantic_ids)) != len(semantic_ids):
            raise ValueError("exact invariant CUDA batch contains duplicate semantic states")
        for m in modules:
            m.validate()
        case_ids = [c.case_id for c in panel]
        if len(set(case_ids)) != len(case_ids):
            raise ValueError("exact invariant CUDA panel contains duplicate case IDs")

        root = Path(work_dir)
        root.mkdir(parents=True, exist_ok=True)
        stem = f"{str(panel_id).replace('/', '_')}_{len(modules)}x{len(panel)}"
        candidates_path = root / f"{stem}.b4im"
        cases_path = root / f"{stem}.cases.tsv"
        result_path = root / f"{stem}.b4ic"
        log_path = root / f"{stem}.cuda.log"
        candidates_path.write_bytes(pack_file(modules))
        cases_path.write_text("\n".join(c.evaluator_tsv() for c in panel) + "\n", encoding="utf-8")
        cmd = [
            str(self.binary),
            "--controller", str(self.profile.controller_path),
            "--routes", str(self.profile.routes_path),
            "--legacy-routes", str(self.profile.legacy_routes_path),
            "--cases", str(cases_path),
            "--candidates", str(candidates_path),
            "--out", str(result_path),
            "--feed-slot", str(self.profile.feed_slot),
            "--feed-count", str(self.profile.feed_count),
            "--strategic-route-slot", str(self.profile.strategic_route_slot),
            "--tpb", str(self.tpb),
            "--memory-fraction", str(self.memory_fraction),
            "--device", str(self.device),
        ]
        env = os.environ.copy()
        if self.cuda_visible_devices is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(self.cuda_visible_devices)
        proc = subprocess.run(cmd, text=True, capture_output=True, env=env, check=False)
        log_path.write_text(proc.stdout + proc.stderr, encoding="utf-8")
        if proc.returncode:
            raise RuntimeError(f"exact B4IC1 CUDA evaluator failed rc={proc.returncode}: {(proc.stderr or proc.stdout)[-4000:]}")
        rows = load_invariant_results(result_path)
        if len(rows) != len(modules) or any(len(row) != len(panel) for row in rows):
            raise ValueError("exact B4IC1 result dimensions do not match requested batch")
        out: dict[str, StateEvaluation] = {}
        for sid, row in zip(semantic_ids, rows):
            out[sid] = StateEvaluation(
                semantic_id=sid,
                panel_id=str(panel_id),
                case_ids=tuple(case_ids),
                margins=tuple(x.margin for x in row),
                action_changes=tuple(int(x.action_changes) for x in row),
                activations=tuple(int(x.activations) for x in row),
                trace_hashes=tuple(int(x.action_trace_hash) for x in row),
            )
        return out

    def config_record(self) -> dict[str, object]:
        return {
            "result_abi": self.RESULT_ABI,
            "candidate_abi": self.CANDIDATE_ABI,
            "binary": str(self.binary),
            "parent": self.profile.canonical_identity(),
            "device": self.device,
            "tpb": self.tpb,
            "memory_fraction": self.memory_fraction,
            "cuda_visible_devices": self.cuda_visible_devices,
        }
