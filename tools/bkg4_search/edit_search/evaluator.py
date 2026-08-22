from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Sequence

from tools.bkg4_search.cuda_io import pack_genomes
from tools.bkg4_search.module_codec import ModuleGenome
from tools.edit_corpus.cuda_results import load_edit_results

from .fitness import StateEvaluation
from .panels import CaseRecord
from .schema import ParentProfile


class ExactCudaEvaluator:
    """Thin batched wrapper around B4EC1. There is intentionally no CPU evaluator fallback."""

    RESULT_ABI = "B4EC1/v1"

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
            raise ValueError("invalid exact CUDA evaluator configuration")

    def evaluate(
        self,
        genomes: Sequence[ModuleGenome],
        panel: Sequence[CaseRecord],
        *,
        work_dir: str | Path,
        panel_id: str,
    ) -> dict[str, StateEvaluation]:
        if not self.binary.is_file():
            raise FileNotFoundError(f"exact B4EC1 CUDA evaluator binary not found: {self.binary}")
        if not genomes:
            return {}
        if not panel:
            raise ValueError("exact CUDA evaluation requires a non-empty case panel")
        semantic_ids = [g.semantic_id() for g in genomes]
        if len(set(semantic_ids)) != len(semantic_ids):
            raise ValueError("exact CUDA batch contains duplicate semantic states")
        for g in genomes:
            g.validate(self.profile.route_ids)
        case_ids = [c.case_id for c in panel]
        if len(set(case_ids)) != len(case_ids):
            raise ValueError("exact CUDA panel contains duplicate case IDs")

        root = Path(work_dir)
        root.mkdir(parents=True, exist_ok=True)
        stem = f"{str(panel_id).replace('/', '_')}_{len(genomes)}x{len(panel)}"
        genomes_path = root / f"{stem}.b4gm"
        cases_path = root / f"{stem}.cases.tsv"
        result_path = root / f"{stem}.b4ec"
        log_path = root / f"{stem}.cuda.log"
        pack_genomes(genomes, genomes_path)
        cases_path.write_text("\n".join(c.evaluator_tsv() for c in panel) + "\n", encoding="utf-8")
        cmd = [
            str(self.binary),
            "--controller", str(self.profile.controller_path),
            "--routes", str(self.profile.routes_path),
            "--legacy-routes", str(self.profile.legacy_routes_path),
            "--cases", str(cases_path),
            "--genomes", str(genomes_path),
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
            raise RuntimeError(f"exact B4EC1 CUDA evaluator failed rc={proc.returncode}: {(proc.stderr or proc.stdout)[-4000:]}")
        rows = load_edit_results(result_path)
        if len(rows) != len(genomes) or any(len(row) != len(panel) for row in rows):
            raise ValueError("exact B4EC1 result dimensions do not match requested batch")
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
            "binary": str(self.binary),
            "parent": self.profile.canonical_identity(),
            "device": self.device,
            "tpb": self.tpb,
            "memory_fraction": self.memory_fraction,
            "cuda_visible_devices": self.cuda_visible_devices,
        }
