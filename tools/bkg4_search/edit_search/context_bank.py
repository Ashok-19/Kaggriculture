from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Sequence

from tools.bkg4_search.information_state_io import load_r0, registry_manifest
from tools.edit_corpus.corpus import canonical_json

from .panels import CaseBank, CaseRecord


class CaseBankRole(str, Enum):
    DISCOVERY = "discovery"
    CERTIFICATION_REPAIR = "certification_repair"
    CERTIFICATION = "certification"
    RANKER_VALIDATION = "ranker_validation"


@dataclass(frozen=True)
class StrategicWindow:
    name: str
    start_step: int
    end_step: int
    probe_steps: tuple[int, ...]


@dataclass(frozen=True)
class StrategicProbeSchedule:
    schema: str
    windows: tuple[StrategicWindow, ...]

    @property
    def probe_steps(self) -> tuple[int, ...]:
        return tuple(step for window in self.windows for step in window.probe_steps)


def strategic_probe_schedule() -> StrategicProbeSchedule:
    """Public probes spanning five strategic phases of the 720-step season.

    The schedule is aligned to the engine's 24-step day and 72-step shop cadence.
    It samples multiple frames per phase rather than learning one narrow step-270 region.
    """
    windows = (
        StrategicWindow("early", 0, 143, (24, 48, 72, 96, 120)),
        StrategicWindow("early_mid", 144, 287, (144, 168, 192, 216, 240, 264)),
        StrategicWindow("mid", 288, 431, (288, 336, 360, 384)),
        StrategicWindow("late_mid", 432, 575, (432, 480, 504, 528)),
        StrategicWindow("late", 576, 719, (576, 624, 648, 672)),
    )
    return StrategicProbeSchedule("bkg4_strategic_multitemporal_v1", windows)


def build_context_bank_from_r0(
    base_bank: CaseBank,
    r0_path: str | Path,
    *,
    role: CaseBankRole,
    probe_steps: Sequence[int] | None = None,
) -> CaseBank:
    cap = load_r0(r0_path)
    if cap.cases != len(base_bank):
        raise ValueError(f"R0/base CaseBank case count mismatch: {cap.cases} != {len(base_bank)}")
    registry_width = len(registry_manifest())
    if cap.registry_width != registry_width:
        raise ValueError(f"R0 registry mismatch: {cap.registry_width} != {registry_width}")
    requested = tuple(int(x) for x in (probe_steps or strategic_probe_schedule().probe_steps))
    if len(set(requested)) != len(requested) or any(x not in cap.probes for x in requested):
        raise ValueError("requested strategic probes are missing/duplicated in R0 capture")
    indices = tuple(cap.probes.index(x) for x in requested)
    rows: list[CaseRecord] = []
    for ci, base in enumerate(base_bank.rows):
        values: list[int] = []
        for pi in indices:
            values.extend(int(x) for x in cap.frame(ci, pi))
        model_context = canonical_json({
            "context_schema": "bkg4_r0_public_multitemporal_v1",
            "seat": int(base.seat),
            "probe_steps": list(requested),
            "frame_width": int(cap.width),
            "registry_width": int(cap.registry_width),
        })
        rows.append(replace(
            base,
            role=role.value,
            model_context_json=model_context,
            context_probe_steps=requested,
            public_context_width=int(cap.width),
            public_context_registry_width=int(cap.registry_width),
            public_context_values=tuple(values),
        ))
    return CaseBank(rows)


def _bucket(row: CaseRecord, fragile_margin: int) -> str:
    if row.baseline_margin < 0:
        return "loss"
    if row.baseline_margin <= int(fragile_margin):
        return "fragile"
    return "strong"


def _stable_key(seed: int, row: CaseRecord) -> tuple[str, str]:
    h = hashlib.sha256(f"BKG4-CONTEXT-SPLIT-V1\0{int(seed)}\0{row.case_id}".encode()).hexdigest()
    return h, row.case_id


def certification_panels(bank: CaseBank, *, parts: int = 2, fragile_margin: int = 5000) -> tuple[tuple[CaseRecord, ...], ...]:
    """Deterministically partition certification cases into disjoint balanced subpanels.

    Family/outcome/seat are evaluation grouping only and never enter model input or predicates.
    """
    parts = int(parts)
    if parts < 1 or parts > len(bank):
        raise ValueError("invalid certification panel count")
    strata: dict[tuple[str, int, str], list[CaseRecord]] = {}
    for row in bank.rows:
        strata.setdefault((_bucket(row, fragile_margin), int(row.seat), str(row.family)), []).append(row)
    for rows in strata.values():
        rows.sort(key=lambda r: _stable_key(0xCE471F, r))
    out: list[list[CaseRecord]] = [[] for _ in range(parts)]
    cursor = 0
    order = {"loss": 0, "fragile": 1, "strong": 2}
    for key in sorted(strata, key=lambda k: (order[k[0]], k[1], k[2])):
        for row in strata[key]:
            out[cursor % parts].append(row)
            cursor += 1
    for seat in (0, 1):
        for dst in [i for i, rows in enumerate(out) if not any(r.seat == seat for r in rows)]:
            donor = next((i for i, rows in enumerate(out) if sum(r.seat == seat for r in rows) > 1), None)
            if donor is None:
                raise ValueError("certification cases cannot provide both-seat panel coverage")
            candidate = next(r for r in reversed(out[donor]) if r.seat == seat)
            swap = next((r for r in out[dst] if r.seat != seat), None)
            if swap is None:
                raise ValueError("certification panel seat rebalance failed")
            out[donor].remove(candidate); out[dst].remove(swap)
            out[donor].append(swap); out[dst].append(candidate)
    return tuple(tuple(sorted(rows, key=lambda r: r.case_id)) for rows in out)


def partition_role_banks(
    bank: CaseBank,
    *,
    discovery: int,
    certification: int,
    ranker_validation: int,
    seed: int,
    fragile_margin: int = 5000,
) -> dict[str, CaseBank]:
    """Make source-disjoint role banks while preserving outcome/seat coverage.

    Family is evaluation grouping only. It is never copied into model_context_json.
    """
    requests = {
        CaseBankRole.DISCOVERY.value: int(discovery),
        CaseBankRole.CERTIFICATION.value: int(certification),
        CaseBankRole.RANKER_VALIDATION.value: int(ranker_validation),
    }
    if any(v <= 0 for v in requests.values()) or sum(requests.values()) > len(bank):
        raise ValueError("invalid role-bank sizes")
    if any(v < 6 for v in requests.values()):
        raise ValueError("each role bank needs at least six cases for outcome/seat coverage")
    unused = {row.case_id: row for row in bank.rows}
    selected: dict[str, list[CaseRecord]] = {k: [] for k in requests}

    # Seed every role with loss/fragile/strong for both seats.
    for role_i, role in enumerate(requests):
        for bucket in ("loss", "fragile", "strong"):
            for seat in (0, 1):
                candidates = [
                    row for row in unused.values()
                    if _bucket(row, fragile_margin) == bucket and row.seat == seat
                ]
                candidates.sort(key=lambda row: _stable_key(int(seed) + role_i * 1009, row))
                if not candidates:
                    raise ValueError(f"cannot preserve {bucket}/seat{seat} coverage for {role}")
                row = candidates[0]
                selected[role].append(row)
                unused.pop(row.case_id)

    # Fill each role with deterministic bucket×seat×family round-robin pressure.
    for role_i, (role, target) in enumerate(requests.items()):
        while len(selected[role]) < target:
            candidates = list(unused.values())
            if not candidates:
                raise ValueError("role partition underfilled")
            counts: dict[tuple[str, int, str], int] = {}
            for row in selected[role]:
                key = (_bucket(row, fragile_margin), row.seat, row.family)
                counts[key] = counts.get(key, 0) + 1
            candidates.sort(key=lambda row: (
                counts.get((_bucket(row, fragile_margin), row.seat, row.family), 0),
                _stable_key(int(seed) + role_i * 65537 + len(selected[role]), row),
            ))
            row = candidates[0]
            selected[role].append(row)
            unused.pop(row.case_id)

    out: dict[str, CaseBank] = {}
    for role, rows in selected.items():
        out[role] = CaseBank(replace(row, role=role) for row in rows)
    return out
