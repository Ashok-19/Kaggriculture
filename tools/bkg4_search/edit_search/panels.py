from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from tools.edit_corpus.corpus import assert_model_payload_safe


@dataclass(frozen=True)
class CaseRecord:
    case_id: str
    tape_path: str
    seat: int
    seed: int
    family: str
    protected: bool = True
    own_history: bool = False
    role: str = "league"
    baseline_margin: int = 0
    model_context_json: str = "{}"
    context_probe_steps: tuple[int, ...] = ()
    public_context_width: int = 0
    public_context_registry_width: int = 0
    public_context_values: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.seat not in (0, 1):
            raise ValueError("case seat must be 0 or 1")
        payload = json.loads(self.model_context_json or "{}")
        if not isinstance(payload, dict):
            raise ValueError("model_context_json must contain an object")
        assert_model_payload_safe(payload)
        if self.public_context_width and len(self.public_context_values) != self.public_context_width * max(1, len(self.context_probe_steps)):
            raise ValueError("public context shape mismatch")

    def evaluator_tsv(self) -> str:
        role_code = 0
        return "\t".join([
            self.tape_path, str(self.seat), str(self.seed), "0", "1" if self.protected else "0",
            "1" if self.own_history else "0", str(role_code),
        ])

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id, "tape_path": self.tape_path, "seat": self.seat, "seed": self.seed,
            "family": self.family, "protected": self.protected, "own_history": self.own_history, "role": self.role,
            "baseline_margin": self.baseline_margin, "model_context_json": self.model_context_json,
            "context_probe_steps": list(self.context_probe_steps), "public_context_width": self.public_context_width,
            "public_context_registry_width": self.public_context_registry_width,
            "public_context_values": list(self.public_context_values),
        }

    @classmethod
    def from_dict(cls, obj: Mapping[str, Any]) -> "CaseRecord":
        return cls(
            case_id=str(obj["case_id"]), tape_path=str(obj["tape_path"]), seat=int(obj["seat"]), seed=int(obj["seed"]),
            family=str(obj.get("family", "unknown")), protected=bool(obj.get("protected", True)),
            own_history=bool(obj.get("own_history", False)), role=str(obj.get("role", "league")),
            baseline_margin=int(obj.get("baseline_margin", 0)), model_context_json=str(obj.get("model_context_json", "{}")),
            context_probe_steps=tuple(int(x) for x in obj.get("context_probe_steps", ())),
            public_context_width=int(obj.get("public_context_width", 0)),
            public_context_registry_width=int(obj.get("public_context_registry_width", 0)),
            public_context_values=tuple(int(x) for x in obj.get("public_context_values", ())),
        )


class CaseBank:
    def __init__(self, rows: Iterable[CaseRecord]):
        self.rows = tuple(rows)
        if not self.rows:
            raise ValueError("case bank must not be empty")
        self.by_id = {x.case_id: x for x in self.rows}
        if len(self.by_id) != len(self.rows):
            raise ValueError("duplicate case ID in case bank")

    def __len__(self) -> int:
        return len(self.rows)

    def __iter__(self):
        return iter(self.rows)

    @classmethod
    def load_jsonl(cls, path: str | Path) -> "CaseBank":
        return cls(CaseRecord.from_dict(json.loads(line)) for line in Path(path).read_text().splitlines() if line.strip())

    def save_jsonl(self, path: str | Path) -> None:
        p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("".join(json.dumps(x.to_dict(), sort_keys=True) + "\n" for x in self.rows))

    def write_evaluator_tsv(self, path: str | Path, case_ids: Sequence[str] | None = None) -> None:
        selected = self.rows if case_ids is None else tuple(self.by_id[str(x)] for x in case_ids)
        p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("\n".join(x.evaluator_tsv() for x in selected) + "\n")

    def identity(self) -> str:
        h = hashlib.sha256(b"BKG4-CASE-BANK-V1\0")
        for row in sorted(self.rows, key=lambda x: x.case_id):
            h.update(json.dumps(row.to_dict(), sort_keys=True, separators=(",", ":")).encode())
        return h.hexdigest()


class CasePanelManager:
    def __init__(
        self,
        bank: CaseBank,
        *,
        fragile_margin: int = 5000,
        severe_delta: int = -3000,
        random_anchor_fraction: float = 0.10,
        useful_recall_floor: float = 0.90,
        danger_recall_floor: float = 0.98,
    ):
        self.bank = bank
        self.fragile_margin = int(fragile_margin)
        self.severe_delta = int(severe_delta)
        self.random_anchor_fraction = max(0.0, min(0.5, float(random_anchor_fraction)))
        self.useful_recall_floor = float(useful_recall_floor)
        self.danger_recall_floor = float(danger_recall_floor)
        self.hard_cases: dict[str, dict[str, Any]] = {}

    def add_hard_case(self, case_id: str, *, severity: int, reason: str) -> None:
        cid = str(case_id)
        if cid not in self.bank.by_id:
            raise KeyError(f"hard case not in case bank: {cid}")
        row = self.hard_cases.setdefault(cid, {"count": 0, "severity": 0, "reasons": {}})
        row["count"] += 1
        row["severity"] = max(int(row["severity"]), int(severity))
        row["reasons"][str(reason)] = int(row["reasons"].get(str(reason), 0)) + 1

    def _bucket(self, row: CaseRecord) -> int:
        if row.baseline_margin < 0: return 0
        if row.baseline_margin <= self.fragile_margin: return 1
        return 2

    @staticmethod
    def _stable_random_key(seed: int, case_id: str) -> str:
        return hashlib.sha256(f"{int(seed)}:{case_id}".encode()).hexdigest()

    def select_screen(self, n: int, *, seed: int) -> list[CaseRecord]:
        n = min(max(0, int(n)), len(self.bank))
        if n == 0: return []
        selected: list[CaseRecord] = []
        used: set[str] = set()
        # Counterexamples are first-class, severity then repeated evidence.
        for cid, meta in sorted(self.hard_cases.items(), key=lambda kv: (-int(kv[1]["severity"]), -int(kv[1]["count"]), kv[0])):
            if len(selected) >= n: break
            selected.append(self.bank.by_id[cid]); used.add(cid)
        # Round-robin nonredundant strata: outcome bucket x seat x family.
        groups: dict[tuple[int, int, str], list[CaseRecord]] = {}
        for row in self.bank:
            if row.case_id in used: continue
            groups.setdefault((self._bucket(row), row.seat, row.family), []).append(row)
        for xs in groups.values():
            xs.sort(key=lambda r: (r.baseline_margin, self._stable_random_key(seed, r.case_id), r.case_id))
        keys = sorted(groups)
        depth = 0
        anchor_n = min(n - len(selected), max(1, int(round(n * self.random_anchor_fraction)))) if self.random_anchor_fraction else 0
        structured_target = max(len(selected), n - anchor_n)
        while len(selected) < structured_target:
            added = False
            for key in keys:
                xs = groups[key]
                if depth < len(xs):
                    row = xs[depth]
                    if row.case_id not in used:
                        selected.append(row); used.add(row.case_id); added = True
                        if len(selected) >= structured_target: break
            if not added: break
            depth += 1
        # Stable pseudo-random anchors protect against panel over-specialization.
        rest = [r for r in self.bank if r.case_id not in used]
        rest.sort(key=lambda r: self._stable_random_key(seed ^ 0xA17C, r.case_id))
        for row in rest:
            if len(selected) >= n: break
            selected.append(row); used.add(row.case_id)
        return selected

    def compare_screen_to_full(self, screen: Mapping[str, Mapping[str, int]], full: Mapping[str, Mapping[str, int]]) -> dict[str, Any]:
        new: list[str] = []
        for cid, row in full.items():
            parent = int(row["parent"]); child = int(row["child"]); delta = child - parent
            dangerous = (parent > 0 and child <= 0) or (parent > 0 and delta < self.severe_delta)
            if dangerous and cid not in screen:
                severity = max(1, -child if child <= 0 else -delta)
                self.add_hard_case(cid, severity=severity, reason="w2l" if child <= 0 else "severe_regression")
                new.append(str(cid))
        return {"screen_cases": len(screen), "full_cases": len(full), "new_counterexamples": sorted(new)}

    @staticmethod
    def _recall(selected: set[str], truth: set[str]) -> float:
        return 1.0 if not truth else len(selected & truth) / len(truth)

    def stage_recall(
        self,
        *,
        screened_program_ids: set[str],
        full_useful_ids: set[str],
        full_l2w_ids: set[str],
        full_dangerous_ids: set[str],
    ) -> dict[str, float]:
        selected = set(screened_program_ids)
        return {
            "useful_recall": self._recall(selected, set(full_useful_ids)),
            "l2w_recall": self._recall(selected, set(full_l2w_ids)),
            "danger_recall": self._recall(selected, set(full_dangerous_ids)),
        }

    def refresh_required(self, diag: Mapping[str, float]) -> bool:
        return (
            float(diag.get("useful_recall", 1.0)) < self.useful_recall_floor
            or float(diag.get("l2w_recall", 1.0)) < self.useful_recall_floor
            or float(diag.get("danger_recall", 1.0)) < self.danger_recall_floor
        )
