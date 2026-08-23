from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

from tools.bkg4_search.invariant_module import GRAMMAR_ID, InvariantStrategicModule


SCHEMA_ID = "bkg4_invariant_model_ast_v1"
CORPUS_RECORD_VERSION = 1
FORBIDDEN_MODEL_TERMS = (
    "tape_path", "tape_id", "replay_id", "episode_id", "seed", "opponent",
    "recursion", "team_identity", "certification", "holdout", "validation_membership",
    "outcome", "margin_delta", "candidate_margin", "qualification_status",
)


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
    return InvariantStrategicModule.from_dict(obj)


def _assert_model_input_safe(model_input: Mapping[str, Any]) -> None:
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
    walk(model_input)


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

