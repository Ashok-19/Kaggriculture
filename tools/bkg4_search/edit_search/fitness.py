from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from tools.edit_corpus.corpus import derive_case_label, derive_edit_labels


@dataclass(frozen=True)
class StateEvaluation:
    """Exact CUDA truth for one semantic search state on one named panel."""

    semantic_id: str
    panel_id: str
    case_ids: tuple[str, ...]
    margins: tuple[int, ...]
    action_changes: tuple[int, ...]
    activations: tuple[int, ...]
    trace_hashes: tuple[int, ...]

    def __post_init__(self) -> None:
        n = len(self.case_ids)
        if not n or any(len(x) != n for x in (self.margins, self.action_changes, self.activations, self.trace_hashes)):
            raise ValueError("StateEvaluation vectors must be non-empty and aligned")
        if len(set(self.case_ids)) != n:
            raise ValueError("StateEvaluation case IDs must be unique")
        if any(not -(1 << 63) <= int(x) < (1 << 63) for x in self.margins):
            raise ValueError("StateEvaluation margin must fit signed int64")
        if any(not 0 <= int(x) <= 0xFFFF for x in self.action_changes):
            raise ValueError("StateEvaluation action_changes must fit B4EC1 uint16")
        if any(not 0 <= int(x) <= 0xFFFF for x in self.activations):
            raise ValueError("StateEvaluation activations must fit B4EC1 uint16")
        if any(not 0 <= int(x) < (1 << 64) for x in self.trace_hashes):
            raise ValueError("StateEvaluation trace hash must fit B4EC1 uint64")

    def to_dict(self) -> dict[str, Any]:
        return {
            "semantic_id": self.semantic_id,
            "panel_id": self.panel_id,
            "case_ids": list(self.case_ids),
            "margins": list(self.margins),
            "action_changes": list(self.action_changes),
            "activations": list(self.activations),
            "trace_hashes": list(self.trace_hashes),
        }

    @classmethod
    def from_dict(cls, obj: Mapping[str, Any]) -> "StateEvaluation":
        return cls(
            semantic_id=str(obj["semantic_id"]), panel_id=str(obj["panel_id"]),
            case_ids=tuple(str(x) for x in obj["case_ids"]),
            margins=tuple(int(x) for x in obj["margins"]),
            action_changes=tuple(int(x) for x in obj["action_changes"]),
            activations=tuple(int(x) for x in obj["activations"]),
            trace_hashes=tuple(int(x) for x in obj["trace_hashes"]),
        )


@dataclass(frozen=True)
class PairEvaluation:
    """Exact parent→child labels computed from two aligned CUDA state vectors."""

    parent_semantic_id: str
    child_semantic_id: str
    panel_id: str
    case_rows: tuple[dict[str, Any], ...]
    aggregate: Mapping[str, Any]

    @classmethod
    def from_states(
        cls,
        parent: StateEvaluation,
        child: StateEvaluation,
        *,
        protected: Mapping[str, bool] | None = None,
    ) -> "PairEvaluation":
        if parent.panel_id != child.panel_id or parent.case_ids != child.case_ids:
            raise ValueError("parent/child StateEvaluation panel/case alignment mismatch")
        rows: list[dict[str, Any]] = []
        protected = dict(protected or {})
        for i, cid in enumerate(parent.case_ids):
            row = derive_case_label(
                parent.margins[i], child.margins[i], child.action_changes[i], child.activations[i],
                protected=bool(protected.get(cid, True)),
                parent_action_trace_hash=parent.trace_hashes[i],
                candidate_action_trace_hash=child.trace_hashes[i],
            )
            row["case_id"] = cid
            row["parent_action_changes_from_base"] = int(parent.action_changes[i])
            row["candidate_action_changes_from_base"] = int(child.action_changes[i])
            row["parent_activations"] = int(parent.activations[i])
            row["candidate_activations"] = int(child.activations[i])
            rows.append(row)
        agg = derive_edit_labels(rows)
        return cls(
            parent_semantic_id=parent.semantic_id,
            child_semantic_id=child.semantic_id,
            panel_id=parent.panel_id,
            case_rows=tuple(rows),
            aggregate=agg,
        )

    def certified_aggregate(self) -> dict[str, Any]:
        """Recompute aggregate truth from exact per-case rows; persisted/cached aggregates are never authoritative."""
        return derive_edit_labels(self.case_rows)

    @property
    def changed_case_ids(self) -> tuple[str, ...]:
        return tuple(str(r["case_id"]) for r in self.case_rows if bool(r["behavior_changing"]))

    @property
    def dangerous_case_ids(self) -> tuple[str, ...]:
        return tuple(
            str(r["case_id"]) for r in self.case_rows
            if bool(r["w2l"]) or (int(r["baseline_margin"]) > 0 and int(r["margin_delta"]) < -3000)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "parent_semantic_id": self.parent_semantic_id,
            "child_semantic_id": self.child_semantic_id,
            "panel_id": self.panel_id,
            "case_rows": [dict(x) for x in self.case_rows],
            "aggregate": dict(self.aggregate),
        }


@dataclass(frozen=True)
class LexicographicFitness:
    """Production ordering. Aggregate gains can never compensate for safety failures."""

    new_losses: int
    w2l: int
    severe_regressions: int
    losses_after: int
    l2w: int
    fragile_wins_after: int
    worst_candidate_margin: int
    worst_protected_win_delta: int
    mean_margin_delta: float
    complexity: int = 0

    @classmethod
    def from_pair(cls, pair: PairEvaluation, *, complexity: int = 0) -> "LexicographicFitness":
        a = pair.certified_aggregate()
        return cls(
            new_losses=int(a["new_loss_count"]),
            w2l=int(a["w2l_count"]),
            severe_regressions=int(a["severe_regression_count"]),
            losses_after=int(a["losses_after"]),
            l2w=int(a["l2w_count"]),
            fragile_wins_after=int(a["fragile_wins_after"]),
            worst_candidate_margin=int(a["worst_candidate_margin"]),
            worst_protected_win_delta=int(a["worst_protected_win_delta"]),
            mean_margin_delta=float(a["mean_margin_delta"]),
            complexity=int(complexity),
        )

    @property
    def strict_safe(self) -> bool:
        return self.new_losses == 0 and self.w2l == 0 and self.severe_regressions == 0

    def key(self) -> tuple[float, ...]:
        return (
            self.new_losses,
            self.w2l,
            self.severe_regressions,
            self.losses_after,
            -self.l2w,
            self.fragile_wins_after,
            -self.worst_candidate_margin,
            -self.worst_protected_win_delta,
            -self.mean_margin_delta,
            self.complexity,
        )
