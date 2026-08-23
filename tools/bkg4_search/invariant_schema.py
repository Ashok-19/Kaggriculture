from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

from tools.bkg4_search.invariant_module import GRAMMAR_ID, InvariantStrategicModule


SCHEMA_ID = "bkg4_invariant_model_ast_v1"
CORPUS_RECORD_VERSION = 1
FORBIDDEN_MODEL_TERMS = (
    "tape_path", "tape_id", "replay_id", "episode_id", "seed", "opponent",
    "recursion", "team_identity", "certification", "holdout", "heldout", "validation_membership",
    "outcome", "margin_delta", "candidate_margin", "qualification_status", "search_rank",
    "search_score", "useful_group", "group_support", "evaluator_margin",
)
MODEL_INPUT_FIELDS = frozenset({"schema", "grammar", "tokens"})
CORPUS_RECORD_FIELDS = frozenset({
    "record_version", "semantic_id", "canonical_ast", "model_input", "target_labels", "provenance",
})
CORPUS_SOURCE_QUALITIES = frozenset({
    "crossfit_positive", "discovery_boundary", "unsafe_negative",
    "inert_negative", "safe_negative",
})
SPECIALIST_SOURCE_QUALITIES = frozenset({
    "global_safe", "elite_specialist", "strong_specialist", "boundary_specialist",
    "high_risk_specialist", "hard_negative", "inert", "cegis_repair",
    "local_neighbor", "structural_ablation", "narrow_specialist", "l2w_specialist",
    "margin_specialist", "counter_specialist", "generalization_boundary",
    "incompatible_specialist_pair",
})


def _encode(value: Any, out: list[str]) -> None:
    if isinstance(value, dict):
        out.append("{")
        for key in sorted(value):
            out.append("K:" + json.dumps(str(key), separators=(",", ":")))
            _encode(value[key], out)
        out.append("}")
    elif isinstance(value, list):
        out.append("[")
        for item in value:
            _encode(item, out)
        out.append("]")
    elif isinstance(value, bool):
        out.append("B:1" if value else "B:0")
    elif value is None:
        out.append("N:")
    elif isinstance(value, int):
        out.append(f"I:{value}")
    elif isinstance(value, str):
        out.append("S:" + json.dumps(value, separators=(",", ":")))
    else:
        raise ValueError(f"unsupported invariant AST scalar {type(value).__name__}")


def _decode(tokens: Sequence[str], pos: int = 0) -> tuple[Any, int]:
    if pos >= len(tokens):
        raise ValueError("truncated invariant AST token stream")
    token = str(tokens[pos])
    if token == "{":
        obj: dict[str, Any] = {}
        pos += 1
        while pos < len(tokens) and tokens[pos] != "}":
            key_token = str(tokens[pos])
            if not key_token.startswith("K:"):
                raise ValueError("invariant AST object is missing a key token")
            key = json.loads(key_token[2:])
            if key in obj:
                raise ValueError("duplicate invariant AST object key")
            value, pos = _decode(tokens, pos + 1)
            obj[str(key)] = value
        if pos >= len(tokens):
            raise ValueError("unterminated invariant AST object")
        return obj, pos + 1
    if token == "[":
        rows = []
        pos += 1
        while pos < len(tokens) and tokens[pos] != "]":
            value, pos = _decode(tokens, pos)
            rows.append(value)
        if pos >= len(tokens):
            raise ValueError("unterminated invariant AST array")
        return rows, pos + 1
    if token.startswith("S:"):
        return str(json.loads(token[2:])), pos + 1
    if token.startswith("I:"):
        return int(token[2:]), pos + 1
    if token == "B:1":
        return True, pos + 1
    if token == "B:0":
        return False, pos + 1
    if token == "N:":
        return None, pos + 1
    raise ValueError(f"invalid invariant AST token {token!r}")


def invariant_model_input(module: InvariantStrategicModule) -> dict[str, Any]:
    module.validate()
    tokens: list[str] = []
    _encode(module.canonical(), tokens)
    result = {"schema": SCHEMA_ID, "grammar": GRAMMAR_ID, "tokens": tokens}
    _assert_model_input_safe(result)
    return result


def module_from_model_input(model_input: Mapping[str, Any]) -> InvariantStrategicModule:
    _assert_model_input_safe(model_input)
    if model_input.get("schema") != SCHEMA_ID or model_input.get("grammar") != GRAMMAR_ID:
        raise ValueError("incompatible invariant model schema")
    tokens = model_input.get("tokens")
    if not isinstance(tokens, list) or not tokens:
        raise ValueError("invariant model input requires AST tokens")
    obj, pos = _decode(tuple(str(x) for x in tokens))
    if pos != len(tokens) or not isinstance(obj, dict):
        raise ValueError("invalid trailing invariant AST tokens")
    _scan_forbidden(obj)
    module = InvariantStrategicModule.from_dict(obj)
    if dict(model_input) != invariant_model_input(module):
        raise ValueError("noncanonical invariant model input")
    return module


def _scan_forbidden(value: Any) -> None:
    def walk(value: Any) -> None:
        if isinstance(value, Mapping):
            for key, item in value.items():
                lowered = str(key).lower()
                if any(term in lowered for term in FORBIDDEN_MODEL_TERMS):
                    raise ValueError(f"forbidden model input field {key!r}")
                walk(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                walk(item)
    walk(value)


def _assert_model_input_safe(model_input: Mapping[str, Any]) -> None:
    _scan_forbidden(model_input)
    fields = frozenset(str(x) for x in model_input)
    if fields != MODEL_INPUT_FIELDS:
        raise ValueError(f"unexpected model input fields: {sorted(fields - MODEL_INPUT_FIELDS)}")


def make_corpus_record(
    module: InvariantStrategicModule,
    *,
    labels: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    if not provenance.get("parent") or not provenance.get("evaluator_abi"):
        raise ValueError("invariant corpus provenance requires parent and evaluator_abi")
    return {
        "record_version": CORPUS_RECORD_VERSION,
        "semantic_id": module.semantic_id(),
        "canonical_ast": module.canonical(),
        "model_input": invariant_model_input(module),
        "target_labels": dict(labels),
        "provenance": dict(provenance),
    }


def validate_corpus_records(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not records:
        raise ValueError("invariant corpus is empty")
    semantics: set[str] = set()
    duplicates = 0
    for index, record in enumerate(records):
        fields = frozenset(str(x) for x in record)
        if fields != CORPUS_RECORD_FIELDS:
            raise ValueError(f"invariant corpus row fields mismatch at {index}: {sorted(fields ^ CORPUS_RECORD_FIELDS)}")
        if int(record.get("record_version", 0)) != CORPUS_RECORD_VERSION:
            raise ValueError(f"invariant corpus row {index} has an incompatible version")
        model_input = record.get("model_input")
        if not isinstance(model_input, Mapping):
            raise ValueError(f"invariant corpus row {index} has no model input")
        module = module_from_model_input(model_input)
        canonical = record.get("canonical_ast")
        if canonical != module.canonical():
            raise ValueError(f"invariant corpus row {index} canonical AST mismatch")
        sid = module.semantic_id()
        if str(record.get("semantic_id")) != sid:
            raise ValueError(f"invariant corpus row {index} semantic ID mismatch")
        provenance = record.get("provenance")
        if not isinstance(provenance, Mapping) or not provenance.get("parent") or not provenance.get("evaluator_abi"):
            raise ValueError(f"invariant corpus row {index} lacks evaluator provenance")
        duplicates += int(sid in semantics)
        semantics.add(sid)
    return {"rows": len(records), "unique_semantics": len(semantics), "duplicate_semantics": duplicates}


def validate_quality_gated_corpus(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Validate exact provenance and keep robust positives distinct from contrast rows."""
    from tools.bkg4_search.invariant_search import structural_skeleton

    base = validate_corpus_records(records)
    if base["duplicate_semantics"]:
        raise ValueError("quality-gated invariant corpus contains duplicate semantics")
    qualities = {name: 0 for name in sorted(CORPUS_SOURCE_QUALITIES)}
    positive_skeletons = set()
    all_skeletons = set()
    for index, record in enumerate(records):
        provenance = record["provenance"]
        required_provenance = (
            provenance.get("exact_evaluator") is True
            and provenance.get("data_role") == "discovery"
            and provenance.get("controller")
            and provenance.get("candidate_abi") == "B4IM1/v1"
            and provenance.get("evaluator_abi") == "B4IC1/v1"
            and int(provenance.get("feature_registry", 0)) == 1
        )
        if not required_provenance:
            raise ValueError(f"invariant corpus row {index} lacks exact CUDA provenance")
        labels = record.get("target_labels")
        if not isinstance(labels, Mapping) or labels.get("exact_data_role") != "discovery":
            raise ValueError(f"invariant corpus row {index} has invalid exact discovery labels")
        quality = str(labels.get("source_quality", ""))
        if quality not in CORPUS_SOURCE_QUALITIES:
            raise ValueError(f"invariant corpus row {index} has invalid source quality")
        qualities[quality] += 1
        module = module_from_model_input(record["model_input"])
        skeleton = structural_skeleton(module)
        all_skeletons.add(skeleton)
        if quality != "crossfit_positive":
            continue
        fitness = labels.get("fitness")
        qualification = labels.get("qualification")
        robust = (
            isinstance(fitness, Mapping)
            and isinstance(qualification, Mapping)
            and int(fitness.get("unsafe_groups", -1)) == 0
            and int(fitness.get("useful_groups", 0)) >= 2
            and bool(qualification.get("production"))
            and int(labels.get("crossfit_fold_support", 0)) >= 2
            and int(labels.get("outer_fold_support", 0)) >= 2
            and int(labels.get("action_change_count", 0)) > 0
        )
        if not robust:
            raise ValueError(
                f"crossfit positive row {index} is not safe and production-qualified"
            )
        positive_skeletons.add(skeleton)
    if not qualities["crossfit_positive"] or not positive_skeletons:
        raise ValueError("quality-gated invariant corpus has no robust positive source")
    return {
        **base,
        **qualities,
        "structural_skeletons": len(all_skeletons),
        "positive_structural_skeletons": len(positive_skeletons),
        "exact_cuda_labels": True,
    }


def validate_specialist_corpus(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Validate exact specialist labels without imposing global deployment safety."""
    from tools.bkg4_search.invariant_search import structural_skeleton

    base = validate_corpus_records(records)
    if base["duplicate_semantics"]:
        raise ValueError("specialist corpus contains duplicate semantics")
    qualities = {name: 0 for name in sorted(SPECIALIST_SOURCE_QUALITIES)}
    categories = {name: 0 for name in sorted(SPECIALIST_SOURCE_QUALITIES)}
    skeletons = set()
    positive_skeletons = set()
    mixed_risk = 0
    for index, record in enumerate(records):
        provenance = record["provenance"]
        exact = (
            provenance.get("exact_evaluator") is True
            and provenance.get("data_role") == "specialist_discovery"
            and provenance.get("controller")
            and provenance.get("candidate_abi") == "B4IM1/v1"
            and provenance.get("evaluator_abi") == "B4IC1/v1"
            and int(provenance.get("feature_registry", 0)) == 1
        )
        if not exact:
            raise ValueError(f"specialist corpus row {index} lacks exact CUDA provenance")
        labels = record.get("target_labels")
        if not isinstance(labels, Mapping) or labels.get("exact_data_role") != "specialist_discovery":
            raise ValueError(f"specialist corpus row {index} has invalid exact data role")
        quality = str(labels.get("source_quality", ""))
        if quality not in SPECIALIST_SOURCE_QUALITIES:
            raise ValueError(f"specialist corpus row {index} has invalid source quality")
        qualities[quality] += 1
        required = ("specialist_tier", "specialist_categories", "activation_count", "action_change_count", "useful_groups",
                    "harmed_groups", "target_cohorts", "group_outcomes", "cohort_outcomes",
                    "source_relative_case_outcomes", "source_relative_mean_delta",
                    "source_relative_l2w", "source_relative_w2l",
                    "composition", "composition_relationships", "lineage")
        missing = [field for field in required if field not in labels]
        if missing:
            raise ValueError(f"specialist corpus row {index} lacks applicability labels: {missing}")
        if not isinstance(labels["group_outcomes"], list) or not labels["group_outcomes"]:
            raise ValueError(f"specialist corpus row {index} has no per-group truth")
        row_categories = labels["specialist_categories"]
        if not isinstance(row_categories, list) or not row_categories:
            raise ValueError(f"specialist corpus row {index} has invalid specialist categories")
        if len(set(row_categories)) != len(row_categories) or any(
            str(category) not in SPECIALIST_SOURCE_QUALITIES for category in row_categories
        ):
            raise ValueError(f"specialist corpus row {index} has unknown/duplicate specialist category")
        for category in row_categories:
            categories[str(category)] += 1
        relationships = labels["composition_relationships"]
        if not isinstance(relationships, Mapping) or set(relationships) != {"compatible", "conflicting"}:
            raise ValueError(f"specialist corpus row {index} has invalid composition relationships")
        useful = int(labels["useful_groups"])
        changes = int(labels["action_change_count"])
        harmed = tuple(str(value) for value in labels["harmed_groups"])
        if quality in {"elite_specialist", "strong_specialist", "global_safe"}:
            if useful < 2 or changes <= 0:
                raise ValueError(f"specialist corpus row {index} positive lacks repeat support")
            positive_skeletons.add(structural_skeleton(module_from_model_input(record["model_input"])))
        if quality == "boundary_specialist" and (useful < 1 or not harmed):
            raise ValueError(f"specialist corpus row {index} boundary lacks help/harm contrast")
        if quality == "inert" and changes != 0:
            raise ValueError(f"specialist corpus row {index} inert row changes behavior")
        mixed_risk += int(useful > 0 and bool(harmed))
        skeletons.add(structural_skeleton(module_from_model_input(record["model_input"])))
    return {
        **base,
        **qualities,
        **{f"category_{name}": count for name, count in categories.items()},
        "structural_skeletons": len(skeletons),
        "positive_structural_skeletons": len(positive_skeletons),
        "mixed_risk_rows": mixed_risk,
        "exact_cuda_labels": True,
    }
