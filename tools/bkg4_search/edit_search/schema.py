from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from tools.bkg4_search.bonsai_editformer.actor_grammar import EditAction, apply_edit, node_refs
from tools.bkg4_search.module_codec import ModuleGenome
from tools.edit_corpus.corpus import canonical_json, stable_identity

from . import EDIT_PROGRAM_SCHEMA_VERSION, GRAMMAR_ID


@dataclass(frozen=True)
class ParentProfile:
    """Data-driven evaluator/search profile for one concrete BKG4 parent lineage."""

    root: Path
    combined_agent_identity: str
    controller_semantic_hash: str
    controller_path: Path
    routes_path: Path
    legacy_routes_path: Path
    route_ids: tuple[int, ...]
    bkg4_abi: int
    feature_registry: int
    feed_slot: int
    feed_count: int
    strategic_route_slot: int

    @classmethod
    def from_directory(cls, path: str | Path) -> "ParentProfile":
        root = Path(path).resolve()
        identity_path = root / "identity.json"
        if not identity_path.is_file():
            raise FileNotFoundError(f"missing BKG4 parent identity: {identity_path}")
        obj = json.loads(identity_path.read_text(encoding="utf-8"))
        routes = tuple(int(x) for x in obj.get("v6_route_pack", {}).get("route_ids", ()))
        if not routes:
            raise ValueError("BKG4 parent identity has no v6 route IDs")
        controller = root / "controller.bkgc"
        route_pack = root / "routes.bkv6"
        legacy = root / "legacy_routes.bkv6"
        for required in (controller, route_pack, legacy):
            if not required.is_file():
                raise FileNotFoundError(f"missing BKG4 parent asset: {required}")
        reconciler = dict(obj.get("runtime_reconciler") or {})
        defer_slots = [int(x) for x in reconciler.get("defer_slots", ())]
        if defer_slots:
            if defer_slots != list(range(defer_slots[0], defer_slots[0] + len(defer_slots))):
                raise ValueError("runtime reconciler defer slots must be contiguous")
            feed_slot, feed_count = defer_slots[0], len(defer_slots)
        else:
            feed_slot, feed_count = 5, 13
        return cls(
            root=root,
            combined_agent_identity=str(obj["combined_agent_identity"]),
            controller_semantic_hash=str(obj["controller_semantic_hash"]),
            controller_path=controller,
            routes_path=route_pack,
            legacy_routes_path=legacy,
            route_ids=routes,
            bkg4_abi=int(obj.get("bkg4_abi", 4)),
            feature_registry=int(obj.get("feature_registry", 1)),
            feed_slot=int(feed_slot),
            feed_count=int(feed_count),
            strategic_route_slot=int(reconciler.get("route_override_slot", 19)),
        )

    def canonical_identity(self) -> dict[str, Any]:
        return {
            "combined_agent_identity": self.combined_agent_identity,
            "controller_semantic_hash": self.controller_semantic_hash,
            "route_ids": list(self.route_ids),
            "bkg4_abi": self.bkg4_abi,
            "feature_registry": self.feature_registry,
            "feed_slot": self.feed_slot,
            "feed_count": self.feed_count,
            "strategic_route_slot": self.strategic_route_slot,
        }


@dataclass(frozen=True)
class EditAtom:
    """One complete WHAT+WHERE+HOW edit relative to the immediately preceding state."""

    parent_semantic_id: str
    child_semantic_id: str
    operation: str
    target_node: int
    target_kind: str
    target_rule_index: int
    target_condition_index: int
    action: Mapping[str, Any]

    @classmethod
    def build(cls, parent: ModuleGenome, action: EditAction, route_ids: Sequence[int]) -> tuple["EditAtom", ModuleGenome]:
        refs = node_refs(parent)
        if not 0 <= int(action.target_node) < len(refs) or refs[int(action.target_node)] is None:
            raise ValueError("invalid editor target node")
        ref = refs[int(action.target_node)]
        assert ref is not None
        child = apply_edit(parent, action, route_ids)
        payload = json.loads(canonical_json(dataclasses.asdict(action)))
        return cls(
            parent_semantic_id=parent.semantic_id(),
            child_semantic_id=child.semantic_id(),
            operation=str(action.operation),
            target_node=int(action.target_node),
            target_kind=str(ref.kind),
            target_rule_index=int(ref.rule_index),
            target_condition_index=int(ref.condition_index),
            action=payload,
        ), child

    def to_action(self) -> EditAction:
        return EditAction(**dict(self.action))

    def canonical(self) -> dict[str, Any]:
        return {
            "parent_semantic_id": self.parent_semantic_id,
            "child_semantic_id": self.child_semantic_id,
            "operation": self.operation,
            "target_node": self.target_node,
            "target_kind": self.target_kind,
            "target_rule_index": self.target_rule_index,
            "target_condition_index": self.target_condition_index,
            "action": json.loads(canonical_json(dict(self.action))),
        }


@dataclass(frozen=True)
class EditProgram:
    """Canonical ordered typed edits from one search parent to one exact child state."""

    schema_version: int
    grammar_id: str
    source_parent_semantic_id: str
    child_semantic_id: str
    atoms: tuple[EditAtom, ...]
    intermediate_semantic_ids: tuple[str, ...]
    emitter: str
    seed: int
    generation_depth: int
    parent_program_id: str | None
    run_id: str | None
    program_id: str

    @property
    def program_length(self) -> int:
        return len(self.atoms)

    @classmethod
    def build(
        cls,
        parent: ModuleGenome,
        actions: Sequence[EditAction],
        route_ids: Sequence[int],
        *,
        emitter: str,
        seed: int,
        generation_depth: int = 1,
        parent_program_id: str | None = None,
        run_id: str | None = None,
    ) -> "EditProgram":
        if not actions:
            raise ValueError("EditProgram requires at least one edit atom")
        state = parent
        atoms: list[EditAtom] = []
        intermediate: list[str] = []
        for action in actions:
            atom, state = EditAtom.build(state, action, route_ids)
            atoms.append(atom)
            intermediate.append(state.semantic_id())
        canonical_body = {
            "schema_version": EDIT_PROGRAM_SCHEMA_VERSION,
            "grammar_id": GRAMMAR_ID,
            "source_parent_semantic_id": parent.semantic_id(),
            "child_semantic_id": state.semantic_id(),
            "atoms": [x.canonical() for x in atoms],
            "intermediate_semantic_ids": intermediate,
        }
        program_id = stable_identity("BKG4-EDIT-PROGRAM-V2", canonical_body)
        return cls(
            schema_version=EDIT_PROGRAM_SCHEMA_VERSION,
            grammar_id=GRAMMAR_ID,
            source_parent_semantic_id=parent.semantic_id(),
            child_semantic_id=state.semantic_id(),
            atoms=tuple(atoms),
            intermediate_semantic_ids=tuple(intermediate),
            emitter=str(emitter),
            seed=int(seed),
            generation_depth=max(1, int(generation_depth)),
            parent_program_id=None if parent_program_id is None else str(parent_program_id),
            run_id=None if run_id is None else str(run_id),
            program_id=program_id,
        )

    def reconstruct(self, parent: ModuleGenome, route_ids: Sequence[int]) -> ModuleGenome:
        if parent.semantic_id() != self.source_parent_semantic_id:
            raise ValueError("EditProgram source parent semantic identity mismatch")
        if self.schema_version != EDIT_PROGRAM_SCHEMA_VERSION or self.grammar_id != GRAMMAR_ID:
            raise ValueError("EditProgram schema/grammar is incompatible with this search backend")
        state = parent
        seen_ids: list[str] = []
        for atom in self.atoms:
            if state.semantic_id() != atom.parent_semantic_id:
                raise ValueError("EditProgram intermediate parent semantic identity mismatch")
            refs = node_refs(state)
            ref = refs[atom.target_node] if 0 <= atom.target_node < len(refs) else None
            if ref is None:
                raise ValueError("EditProgram target no longer resolves")
            got = (str(ref.kind), int(ref.rule_index), int(ref.condition_index))
            expected = (atom.target_kind, atom.target_rule_index, atom.target_condition_index)
            if got != expected:
                raise ValueError(f"EditProgram target metadata mismatch: {got} != {expected}")
            state = apply_edit(state, atom.to_action(), route_ids)
            if state.semantic_id() != atom.child_semantic_id:
                raise ValueError("EditProgram atom child semantic identity mismatch")
            seen_ids.append(state.semantic_id())
        if tuple(seen_ids) != self.intermediate_semantic_ids:
            raise ValueError("EditProgram intermediate semantic ledger mismatch")
        if state.semantic_id() != self.child_semantic_id:
            raise ValueError("EditProgram final child semantic identity mismatch")
        return state

    def canonical(self, *, include_provenance: bool = True) -> dict[str, Any]:
        out: dict[str, Any] = {
            "schema_version": self.schema_version,
            "grammar_id": self.grammar_id,
            "program_id": self.program_id,
            "source_parent_semantic_id": self.source_parent_semantic_id,
            "child_semantic_id": self.child_semantic_id,
            "atoms": [x.canonical() for x in self.atoms],
            "intermediate_semantic_ids": list(self.intermediate_semantic_ids),
            "program_length": self.program_length,
        }
        if include_provenance:
            out.update(
                emitter=self.emitter,
                seed=self.seed,
                generation_depth=self.generation_depth,
                parent_program_id=self.parent_program_id,
                run_id=self.run_id,
            )
        return out

    @classmethod
    def from_dict(cls, obj: Mapping[str, Any]) -> "EditProgram":
        atoms = tuple(EditAtom(**dict(x)) for x in obj["atoms"])
        body = {
            "schema_version": int(obj["schema_version"]),
            "grammar_id": str(obj["grammar_id"]),
            "source_parent_semantic_id": str(obj["source_parent_semantic_id"]),
            "child_semantic_id": str(obj["child_semantic_id"]),
            "atoms": [a.canonical() for a in atoms],
            "intermediate_semantic_ids": [str(x) for x in obj["intermediate_semantic_ids"]],
        }
        expected_id = stable_identity("BKG4-EDIT-PROGRAM-V2", body)
        supplied = str(obj.get("program_id") or expected_id)
        if supplied != expected_id:
            raise ValueError("EditProgram canonical identity mismatch")
        return cls(
            schema_version=body["schema_version"],
            grammar_id=body["grammar_id"],
            source_parent_semantic_id=body["source_parent_semantic_id"],
            child_semantic_id=body["child_semantic_id"],
            atoms=atoms,
            intermediate_semantic_ids=tuple(body["intermediate_semantic_ids"]),
            emitter=str(obj.get("emitter", "unknown")),
            seed=int(obj.get("seed", 0)),
            generation_depth=max(1, int(obj.get("generation_depth", len(atoms)))),
            parent_program_id=obj.get("parent_program_id"),
            run_id=obj.get("run_id"),
            program_id=supplied,
        )


def genome_canonical_json(genome: ModuleGenome) -> str:
    return canonical_json(genome.canonical())
