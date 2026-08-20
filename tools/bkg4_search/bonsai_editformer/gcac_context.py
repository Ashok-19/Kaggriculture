from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from tools.bkg4_search.information_state_io import load_r0
from tools.bkg4_search.cuda_io import load_results


def sha256_file(path: str | Path) -> str:
    p = Path(path)
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def case_signatures(path: str | Path) -> list[tuple[str, int, int]]:
    """Return basename/seat/seed bookkeeping signatures from 6- or 7-field TSV."""
    out: list[tuple[str, int, int]] = []
    for line in Path(path).read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        z = line.split("\t")
        if len(z) not in (6, 7):
            raise ValueError(f"goal-context case row must have 6 or 7 fields, got {len(z)}")
        out.append((Path(z[0]).name, int(z[1]), int(z[2])))
    if not out:
        raise ValueError("empty goal-context case file")
    return out


def assert_case_alignment(reference: str | Path, candidate: str | Path) -> None:
    a, b = case_signatures(reference), case_signatures(candidate)
    if a != b:
        first = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
        av = a[first] if first < len(a) else None
        bv = b[first] if first < len(b) else None
        raise ValueError(f"goal-context case alignment mismatch at row {first}: {av} != {bv}; sizes={len(a)}/{len(b)}")


def case_meta_from_tsv(path: str | Path) -> list[dict[str, int]]:
    """Return only legal/non-identity metadata used by the case feature builder."""
    out = []
    for line in Path(path).read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        z = line.split("\t")
        if len(z) not in (6, 7):
            raise ValueError("invalid goal-context case row")
        out.append({"seat": int(z[1]), "protected": int(z[4]), "own_history": int(z[5]), "family": 0})
    return out


def baseline_margins(result_path: str | Path) -> np.ndarray:
    rows = load_results(result_path)
    if not rows:
        raise ValueError("empty CUDA result")
    if any(int(flags) != 0 for _, _, _, flags in rows[0]):
        raise ValueError("baseline CUDA result contains evaluator flags")
    return np.asarray([int(own) - int(opp) for own, opp, _, _ in rows[0]], np.int32)


def create_goal_context_manifest(
    *,
    out: str | Path,
    capture: str | Path,
    cases: str | Path,
    reference_cases: str | Path,
    baseline_result: str | Path,
    parent_semantic: str,
    controller_identity: str,
    expected_probes: tuple[int, ...] = tuple(range(264, 288)),
) -> dict[str, Any]:
    capture = Path(capture); cases = Path(cases); reference_cases = Path(reference_cases); baseline_result = Path(baseline_result)
    assert_case_alignment(reference_cases, cases)
    r0 = load_r0(capture)
    signatures = case_signatures(cases)
    if r0.cases != len(signatures):
        raise ValueError(f"capture case count mismatch {r0.cases} != {len(signatures)}")
    if tuple(r0.probes) != tuple(expected_probes):
        raise ValueError(f"capture probe mismatch: {r0.probes} != {expected_probes}")
    margins = baseline_margins(baseline_result)
    if len(margins) != len(signatures):
        raise ValueError(f"baseline result case count mismatch {len(margins)} != {len(signatures)}")
    loss_ids = [int(i) for i in np.flatnonzero(margins < 0)]
    win_ids = [int(i) for i in np.flatnonzero(margins > 0)]
    unique_seeds = sorted({seed for _, _, seed in signatures})
    seats_by_seed = {s: sorted({seat for _, seat, seed in signatures if seed == s}) for s in unique_seeds}
    manifest = {
        "schema": 1,
        "architecture": "BonsaiEditFormer-GCAC-v2",
        "parent_semantic": str(parent_semantic),
        "context_semantics": "frozen-345 legal public-state trajectory; ModuleGenome candidate activation evaluated offline over captured step264..287 states",
        "identity_features_embedded": False,
        "capture": str(capture),
        "capture_sha256": sha256_file(capture),
        "capture_cases": int(r0.cases),
        "capture_probes": [int(x) for x in r0.probes],
        "capture_registry_width": int(r0.registry_width),
        "capture_bb_width": int(r0.bb_width),
        "controller_identity": str(controller_identity),
        "cases": str(cases),
        "cases_sha256": sha256_file(cases),
        "reference_cases": str(reference_cases),
        "reference_cases_sha256": sha256_file(reference_cases),
        "baseline_result": str(baseline_result),
        "baseline_result_sha256": sha256_file(baseline_result),
        "baseline_margins": [int(x) for x in margins],
        "loss_case_ids": loss_ids,
        "win_case_ids": win_ids,
        "cases_count": len(signatures),
        "unique_seeds": len(unique_seeds),
        "both_seats_per_seed": all(v == [0, 1] for v in seats_by_seed.values()),
        "seeds": unique_seeds,
    }
    p = Path(out); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def load_goal_context_manifest(path: str | Path, *, expected_parent: str | None = None) -> dict[str, Any]:
    p = Path(path)
    m = json.loads(p.read_text())
    if int(m.get("schema", 0)) != 1 or m.get("architecture") != "BonsaiEditFormer-GCAC-v2":
        raise ValueError("invalid GCAC-v2 goal-context manifest")
    if expected_parent is not None and str(m.get("parent_semantic")) != str(expected_parent):
        raise ValueError(f"goal-context parent mismatch {m.get('parent_semantic')} != {expected_parent}")
    for key, sha_key in (("capture", "capture_sha256"), ("cases", "cases_sha256"), ("reference_cases", "reference_cases_sha256"), ("baseline_result", "baseline_result_sha256")):
        q = Path(m[key])
        if not q.exists():
            raise FileNotFoundError(q)
        got = sha256_file(q)
        if got != m[sha_key]:
            raise ValueError(f"goal-context hash mismatch for {key}: {got} != {m[sha_key]}")
    assert_case_alignment(m["reference_cases"], m["cases"])
    r0 = load_r0(m["capture"])
    if int(r0.cases) != int(m["cases_count"]):
        raise ValueError("goal-context capture case count changed")
    if [int(x) for x in r0.probes] != [int(x) for x in m["capture_probes"]]:
        raise ValueError("goal-context probe schedule changed")
    margins = baseline_margins(m["baseline_result"])
    if [int(x) for x in margins] != [int(x) for x in m["baseline_margins"]]:
        raise ValueError("goal-context baseline margins changed")
    if [int(i) for i in np.flatnonzero(margins < 0)] != [int(x) for x in m["loss_case_ids"]]:
        raise ValueError("goal-context loss ids changed")
    if not bool(m.get("both_seats_per_seed")) or int(m.get("unique_seeds", 0)) < 6:
        raise ValueError("goal-context no longer satisfies six-seed/both-seat gate")
    return m
