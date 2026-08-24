from __future__ import annotations

import hashlib
from typing import Any, Mapping, Sequence

from tools.bkg4_search.edit_search.fitness import StateEvaluation
from tools.bkg4_search.invariant_module import InvariantStrategicModule


SPECIALIST_LINEAGE_SCHEMA_VERSION = 2
SPECIALIST_PARENT_EVIDENCE_VERSION = 1

LINEAGE_FIELDS = frozenset({
    "lineage_schema_version",
    "root_controller",
    "root_semantic_id",
    "parent_semantic_id",
    "operator",
    "depth",
    "source_margin_delta",
    "lineage_id",
    "source_goal_id",
    "source_goal_path",
    "structural_mechanisms",
    "retrospective_goal_id",
    "parent_evidence",
})
PARENT_EVIDENCE_FIELDS = frozenset({
    "evidence_version",
    "ancestry",
    "exact_evaluation",
})
ANCESTOR_FIELDS = frozenset({
    "semantic_id",
    "canonical_ast",
    "depth",
    "parent_semantic_id",
    "root_semantic_id",
    "lineage_id",
    "operator",
})


def ancestor_node_from_record(record: Mapping[str, Any]) -> dict[str, Any]:
    labels = record.get("target_labels")
    lineage = labels.get("lineage") if isinstance(labels, Mapping) else None
    canonical = record.get("canonical_ast")
    if not isinstance(lineage, Mapping) or not isinstance(canonical, Mapping):
        raise ValueError("specialist record cannot be converted to an ancestry node")
    return {
        "semantic_id": str(record["semantic_id"]),
        "canonical_ast": dict(canonical),
        "depth": int(lineage["depth"]),
        "parent_semantic_id": lineage.get("parent_semantic_id"),
        "root_semantic_id": str(lineage["root_semantic_id"]),
        "lineage_id": str(lineage["lineage_id"]),
        "operator": str(lineage["operator"]),
    }


def ancestor_node_from_lineage(lineage: Any) -> dict[str, Any]:
    return {
        "semantic_id": lineage.module.semantic_id(),
        "canonical_ast": lineage.module.canonical(),
        "depth": int(lineage.depth),
        "parent_semantic_id": lineage.parent_semantic_id,
        "root_semantic_id": str(lineage.root_semantic_id),
        "lineage_id": str(lineage.lineage_id),
        "operator": str(lineage.operator),
    }


def make_parent_evidence(
    ancestry: Sequence[Any],
    exact_parent: StateEvaluation,
) -> dict[str, Any]:
    chain = tuple(ancestry)
    if not chain:
        raise ValueError("parent evidence requires a non-empty ancestry chain")
    if chain[-1].module.semantic_id() != exact_parent.semantic_id:
        raise ValueError("exact parent evaluation does not match ancestry tail")
    return {
        "evidence_version": SPECIALIST_PARENT_EVIDENCE_VERSION,
        "ancestry": [ancestor_node_from_lineage(row) for row in chain],
        "exact_evaluation": exact_parent.to_dict(),
    }


def _canonical_ancestor(node: Mapping[str, Any], *, row_index: int) -> dict[str, Any]:
    fields = frozenset(str(key) for key in node)
    if fields != ANCESTOR_FIELDS:
        raise ValueError(f"specialist corpus row {row_index} has invalid ancestor fields")
    canonical = node.get("canonical_ast")
    if not isinstance(canonical, Mapping):
        raise ValueError(f"specialist corpus row {row_index} ancestor lacks canonical AST")
    module = InvariantStrategicModule.from_dict(canonical)
    module.validate()
    canonical_module = module.canonical()
    if canonical_module != dict(canonical):
        raise ValueError(f"specialist corpus row {row_index} ancestor AST is noncanonical")
    semantic_id = module.semantic_id()
    if str(node.get("semantic_id")) != semantic_id:
        raise ValueError(f"specialist corpus row {row_index} ancestor semantic ID mismatch")
    depth = int(node.get("depth", -1))
    if depth < 0:
        raise ValueError(f"specialist corpus row {row_index} ancestor depth is invalid")
    parent = node.get("parent_semantic_id")
    if parent is not None and (not isinstance(parent, str) or not parent):
        raise ValueError(f"specialist corpus row {row_index} ancestor parent ID is invalid")
    root = str(node.get("root_semantic_id", ""))
    lineage_id = str(node.get("lineage_id", ""))
    operator = str(node.get("operator", ""))
    if not root or not lineage_id or not operator:
        raise ValueError(f"specialist corpus row {row_index} ancestor metadata is incomplete")
    return {
        "semantic_id": semantic_id,
        "canonical_ast": canonical_module,
        "depth": depth,
        "parent_semantic_id": parent,
        "root_semantic_id": root,
        "lineage_id": lineage_id,
        "operator": operator,
    }


def _validate_parent_exact_evidence(
    *,
    row_index: int,
    parent_semantic_id: str,
    evidence: Mapping[str, Any],
    source_rows: Any,
) -> StateEvaluation:
    fields = frozenset(str(key) for key in evidence)
    if fields != PARENT_EVIDENCE_FIELDS:
        raise ValueError(f"specialist corpus row {row_index} has invalid parent-evidence fields")
    if int(evidence.get("evidence_version", -1)) != SPECIALIST_PARENT_EVIDENCE_VERSION:
        raise ValueError(f"specialist corpus row {row_index} has incompatible parent-evidence version")
    exact_raw = evidence.get("exact_evaluation")
    if not isinstance(exact_raw, Mapping):
        raise ValueError(f"specialist corpus row {row_index} lacks exact parent evaluation")
    exact = StateEvaluation.from_dict(exact_raw)
    if exact.semantic_id != parent_semantic_id:
        raise ValueError(f"specialist corpus row {row_index} exact parent semantic ID mismatch")
    if not isinstance(source_rows, list) or len(source_rows) != len(exact.case_ids):
        raise ValueError(f"specialist corpus row {row_index} parent-relative exact rows are unaligned")
    for case_index, (case_id, source_row) in enumerate(zip(exact.case_ids, source_rows)):
        if not isinstance(source_row, Mapping):
            raise ValueError(f"specialist corpus row {row_index} has invalid source-relative case row")
        expected = {
            "case_id": str(case_id),
            "baseline_margin": int(exact.margins[case_index]),
            "parent_trace_hash": int(exact.trace_hashes[case_index]),
            "parent_action_changes": int(exact.action_changes[case_index]),
            "parent_activations": int(exact.activations[case_index]),
        }
        for key, value in expected.items():
            if source_row.get(key) != value:
                raise ValueError(
                    f"specialist corpus row {row_index} parent-relative evidence disagrees with exact parent {key}"
                )
    return exact


def validate_specialist_lineage_graph(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Validate every retained or external exact-labeled specialist ancestry edge."""
    if not records:
        raise ValueError("specialist lineage graph is empty")
    by_sid = {str(record["semantic_id"]): record for record in records}
    if len(by_sid) != len(records):
        raise ValueError("specialist lineage graph contains duplicate semantic IDs")

    lineage_by_sid: dict[str, Mapping[str, Any]] = {}
    ancestor_registry: dict[str, dict[str, Any]] = {}
    exact_parent_registry: dict[str, StateEvaluation] = {}
    roots = 0
    external_parent_rows = 0

    for index, record in enumerate(records):
        sid = str(record["semantic_id"])
        labels = record.get("target_labels")
        lineage = labels.get("lineage") if isinstance(labels, Mapping) else None
        if not isinstance(lineage, Mapping):
            raise ValueError(f"specialist corpus row {index} lacks lineage metadata")
        fields = frozenset(str(key) for key in lineage)
        if fields != LINEAGE_FIELDS:
            raise ValueError(
                f"specialist corpus row {index} lineage fields mismatch: {sorted(fields ^ LINEAGE_FIELDS)}"
            )
        if int(lineage.get("lineage_schema_version", -1)) != SPECIALIST_LINEAGE_SCHEMA_VERSION:
            raise ValueError(f"specialist corpus row {index} has incompatible lineage schema")

        root_controller = str(lineage.get("root_controller", ""))
        root_sid = str(lineage.get("root_semantic_id", ""))
        operator = str(lineage.get("operator", ""))
        lineage_id = str(lineage.get("lineage_id", ""))
        if not root_controller or not root_sid or not operator or not lineage_id:
            raise ValueError(f"specialist corpus row {index} has incomplete lineage identity")
        depth = int(lineage.get("depth", -1))
        if depth < 0:
            raise ValueError(f"specialist corpus row {index} has invalid lineage depth")
        parent = lineage.get("parent_semantic_id")
        if parent is not None and (not isinstance(parent, str) or not parent):
            raise ValueError(f"specialist corpus row {index} has invalid parent semantic ID")
        if parent == sid:
            raise ValueError(f"specialist corpus row {index} cannot parent itself")

        source_goal = lineage.get("source_goal_id")
        goal_path = lineage.get("source_goal_path")
        if source_goal is not None and (not isinstance(source_goal, str) or not source_goal):
            raise ValueError(f"specialist corpus row {index} has invalid source goal ID")
        if not isinstance(goal_path, list) or any(not isinstance(value, str) or not value for value in goal_path):
            raise ValueError(f"specialist corpus row {index} has invalid source goal path")
        if (source_goal is None) != (len(goal_path) == 0):
            raise ValueError(f"specialist corpus row {index} source goal/path disagree")
        if source_goal is not None and goal_path[-1] != source_goal:
            raise ValueError(f"specialist corpus row {index} source goal path does not end at source goal")
        if operator == "mcts" and source_goal is None:
            raise ValueError(f"specialist corpus row {index} MCTS lineage lacks generating goal")

        mechanisms = lineage.get("structural_mechanisms")
        if (
            not isinstance(mechanisms, list)
            or any(not isinstance(value, str) or not value for value in mechanisms)
            or len(set(mechanisms)) != len(mechanisms)
        ):
            raise ValueError(f"specialist corpus row {index} has invalid structural mechanism provenance")
        lineage_by_sid[sid] = lineage

        if depth == 0:
            roots += 1
            if parent is not None:
                raise ValueError(f"specialist corpus row {index} depth-0 lineage has a parent")
            if root_sid != sid:
                raise ValueError(f"specialist corpus row {index} depth-0 root semantic ID is not self")
            if lineage.get("parent_evidence") is not None:
                raise ValueError(f"specialist corpus row {index} depth-0 lineage has parent evidence")
            expected_lineage_id = hashlib.sha256(f"{root_controller}\0{sid}".encode()).hexdigest()
            if lineage_id != expected_lineage_id:
                raise ValueError(f"specialist corpus row {index} depth-0 lineage ID mismatch")
            continue

        if parent is None:
            raise ValueError(f"specialist corpus row {index} non-root lineage has no parent")
        evidence = lineage.get("parent_evidence")
        if not isinstance(evidence, Mapping):
            raise ValueError(f"specialist corpus row {index} non-root lineage lacks parent evidence")
        ancestry_raw = evidence.get("ancestry")
        if not isinstance(ancestry_raw, list) or len(ancestry_raw) != depth:
            raise ValueError(f"specialist corpus row {index} ancestry length does not match depth")
        ancestry = []
        for raw_node in ancestry_raw:
            if not isinstance(raw_node, Mapping):
                raise ValueError(f"specialist corpus row {index} has invalid ancestry node")
            ancestry.append(_canonical_ancestor(raw_node, row_index=index))
        if sid in {node["semantic_id"] for node in ancestry}:
            raise ValueError(f"specialist corpus row {index} ancestry contains the child semantic ID")
        if len({node["semantic_id"] for node in ancestry}) != len(ancestry):
            raise ValueError(f"specialist corpus row {index} ancestry repeats a semantic ID")

        for expected_depth, node in enumerate(ancestry):
            if node["depth"] != expected_depth:
                raise ValueError(f"specialist corpus row {index} ancestry depth sequence is invalid")
            if node["root_semantic_id"] != root_sid:
                raise ValueError(f"specialist corpus row {index} ancestry root semantic ID changes")
            if expected_depth == 0:
                if node["parent_semantic_id"] is not None or node["semantic_id"] != root_sid:
                    raise ValueError(f"specialist corpus row {index} ancestry does not terminate at its root")
                expected_id = hashlib.sha256(f"{root_controller}\0{node['semantic_id']}".encode()).hexdigest()
            else:
                previous = ancestry[expected_depth - 1]
                if node["parent_semantic_id"] != previous["semantic_id"]:
                    raise ValueError(f"specialist corpus row {index} ancestry parent chain is broken")
                expected_id = hashlib.sha256(
                    f"{previous['lineage_id']}\0{node['operator']}\0{node['semantic_id']}".encode()
                ).hexdigest()
            if node["lineage_id"] != expected_id:
                raise ValueError(f"specialist corpus row {index} ancestor lineage ID mismatch")
            previous_node = ancestor_registry.setdefault(node["semantic_id"], node)
            if previous_node != node:
                raise ValueError(
                    f"specialist corpus row {index} reuses parent semantic ID with contradictory metadata"
                )

        immediate_parent = ancestry[-1]
        if immediate_parent["semantic_id"] != parent or depth != immediate_parent["depth"] + 1:
            raise ValueError(f"specialist corpus row {index} immediate parent/depth mismatch")
        expected_child_id = hashlib.sha256(
            f"{immediate_parent['lineage_id']}\0{operator}\0{sid}".encode()
        ).hexdigest()
        if lineage_id != expected_child_id:
            raise ValueError(f"specialist corpus row {index} child lineage ID mismatch")

        exact_parent = _validate_parent_exact_evidence(
            row_index=index,
            parent_semantic_id=parent,
            evidence=evidence,
            source_rows=labels.get("source_relative_case_outcomes"),
        )
        previous_exact = exact_parent_registry.setdefault(parent, exact_parent)
        if previous_exact != exact_parent:
            raise ValueError(
                f"specialist corpus row {index} reuses parent semantic ID with contradictory exact evidence"
            )

        internal_parent = by_sid.get(parent)
        if internal_parent is None:
            external_parent_rows += 1
        else:
            expected_parent_node = ancestor_node_from_record(internal_parent)
            if immediate_parent != expected_parent_node:
                raise ValueError(
                    f"specialist corpus row {index} parent evidence contradicts internal parent row"
                )

    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(sid: str) -> None:
        if sid in visited:
            return
        if sid in visiting:
            raise ValueError("specialist corpus lineage graph contains a cycle")
        visiting.add(sid)
        parent = lineage_by_sid[sid].get("parent_semantic_id")
        if parent in lineage_by_sid:
            visit(str(parent))
        visiting.remove(sid)
        visited.add(sid)

    for sid in lineage_by_sid:
        visit(sid)

    complete_depth2_chains = 0
    for sid, lineage in lineage_by_sid.items():
        if int(lineage["depth"]) != 2:
            continue
        parent = lineage.get("parent_semantic_id")
        if parent not in lineage_by_sid:
            continue
        grandparent = lineage_by_sid[str(parent)].get("parent_semantic_id")
        if grandparent in lineage_by_sid and int(lineage_by_sid[str(grandparent)]["depth"]) == 0:
            complete_depth2_chains += 1

    return {
        "lineage_schema_version": SPECIALIST_LINEAGE_SCHEMA_VERSION,
        "parent_evidence_version": SPECIALIST_PARENT_EVIDENCE_VERSION,
        "lineage_roots": roots,
        "non_root_rows": len(records) - roots,
        "external_parent_rows": external_parent_rows,
        "complete_depth2_chains": complete_depth2_chains,
    }
