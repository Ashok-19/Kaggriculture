"""Foundational, dependency-free schema and semantic identity contracts."""

from __future__ import annotations

import hashlib
import json
from enum import Enum
from typing import Any, Mapping

RLRMB_SCHEMA_VERSION = 1


class DatasetRole(str, Enum):
    """Leakage-sensitive role of an RLRMB evidence dataset."""

    TARGET = "TARGET"
    TRANSFER = "TRANSFER"
    VALIDATION = "VALIDATION"
    SEALED = "SEALED"
    CERTIFICATION = "CERTIFICATION"

    @classmethod
    def parse(cls, value: str) -> "DatasetRole":
        """Parse an exact serialized role, rejecting coercion and unknown values."""
        if type(value) is not str:
            raise TypeError("dataset role must be an exact string")
        try:
            return cls(value)
        except ValueError as exc:
            raise ValueError(f"unknown dataset role: {value!r}") from exc


def validate_schema_version(version: object) -> int:
    """Return the supported version or fail closed for every other value."""
    if type(version) is not int:
        raise TypeError("schema version must be an integer")
    if version != RLRMB_SCHEMA_VERSION:
        raise ValueError(f"unsupported RLRMB schema version: {version}")
    return version


def canonical_json(value: Any) -> str:
    """Serialize a JSON-compatible value in the canonical RLRMB form."""
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def stable_identity(namespace: str, *payloads: Any) -> str:
    """Hash canonical semantic payloads with explicit namespace separation."""
    if type(namespace) is not str or not namespace:
        raise ValueError("identity namespace must be a non-empty string")
    digest = hashlib.sha256()
    namespace_bytes = namespace.encode("utf-8")
    digest.update(len(namespace_bytes).to_bytes(8, "big"))
    digest.update(namespace_bytes)
    for payload in payloads:
        encoded = canonical_json(payload).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


# Exact normalized keys only: semantic names such as OWN_SEED are intentionally
# legal, while offline seed provenance has its own explicit forbidden keys.
FORBIDDEN_RUNTIME_IDENTITY_FIELDS = frozenset(
    {
        "opponent",
        "opponent_name",
        "team",
        "team_name",
        "submission",
        "submission_id",
        "replay",
        "replay_id",
        "episode",
        "episode_id",
        "tape",
        "tape_id",
        "filename",
        "rank",
        "seed",
        "offline_seed",
        "replay_seed",
        "episode_seed",
        "tape_seed",
    }
)


def validate_runtime_identity_safe(payload: Any, path: str = "payload") -> None:
    """Reject identity/provenance keys recursively from runtime semantic data."""
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            if type(key) is not str:
                raise TypeError(f"runtime semantic key at {path} must be a string")
            normalized_key = key.casefold().replace("-", "_")
            if normalized_key in FORBIDDEN_RUNTIME_IDENTITY_FIELDS:
                raise ValueError(f"identity field forbidden at {path}: {key}")
            validate_runtime_identity_safe(value, f"{path}.{key}")
    elif isinstance(payload, (list, tuple)):
        for index, value in enumerate(payload):
            validate_runtime_identity_safe(value, f"{path}[{index}]")
