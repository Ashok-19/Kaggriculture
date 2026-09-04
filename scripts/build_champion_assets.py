"""Compile a composed BKG4 module while retaining the canonical route packs."""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

from kaggriculture_bonsai.paths import CHAMPION_ASSETS
from kaggriculture_bonsai.runtime import bonsai_kg_v4


def build(output_dir: Path, module_path: Path) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    source = json.loads(Path(module_path).read_text(encoding="utf-8"))
    executable_fields = {
        "schema_version", "name", "feature_registry_version", "enums", "blackboard",
        "subtrees", "tree", "provenance",
    }
    executable = {key: value for key, value in source.items() if key in executable_fields}
    compiled = bonsai_kg_v4.compile_policy(executable)
    bonsai_kg_v4.save_compiled(compiled, output_dir / "controller.bkgc")
    (output_dir / "controller.json").write_text(
        json.dumps(source, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    for name in ("routes.bkv6", "legacy_routes.bkv6"):
        shutil.copy2(CHAMPION_ASSETS / name, output_dir / name)
    identity = json.loads((CHAMPION_ASSETS / "identity.json").read_text(encoding="utf-8"))
    identity["controller_semantic_hash"] = compiled.semantic_hash
    identity["controller_nodes"] = len(compiled.nodes)
    identity["controller_state_slots"] = compiled.state_count
    identity["controller_bb_slots"] = len(compiled.bb)
    identity["controller_sha256"] = hashlib.sha256(
        (output_dir / "controller.bkgc").read_bytes()
    ).hexdigest()
    (output_dir / "identity.json").write_text(
        json.dumps(identity, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    return identity
