from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from tools.bkg4_search.module_codec import ModuleGenome, genome_from_json
from tools.edit_corpus.corpus import canonical_json

from . import EMITTER_REGISTRY_VERSION, GRAMMAR_ID, QD_DESCRIPTOR_SCHEMA_VERSION, SEARCH_STATE_SCHEMA_VERSION
from .fitness import LexicographicFitness, PairEvaluation, StateEvaluation
from .schema import EditProgram


def _feature_family(feature: str) -> str:
    f = str(feature)
    if f.startswith("MARKET_"): return "market"
    if f.startswith("SHOP_"): return "shop"
    if "MONEY" in f: return "money"
    if "UNIT" in f or "WORKER" in f or "HIRES" in f: return "units"
    if "LAND" in f or "TILE" in f: return "land_tile"
    if f in {"STEP", "DAY", "HOUR"}: return "time"
    return "other"


def structural_descriptor(program: EditProgram, child: ModuleGenome) -> dict[str, Any]:
    atom = program.atoms[-1]
    action = dict(atom.action)
    return {
        "schema": QD_DESCRIPTOR_SCHEMA_VERSION,
        "operation": atom.operation,
        "target_kind": atom.target_kind,
        "target_depth": 0 if atom.target_kind == "root" else (1 if atom.target_kind == "rule" else 2),
        "feature_family": _feature_family(str(action.get("feature", ""))),
        "rule_count": len(child.rules),
        "rule_count_bucket": min(8, len(child.rules)),
        "prefix_bucket": min(11, int(child.prefix_step) // 60),
        "program_length": program.program_length,
    }

def policy_state_descriptor(child: ModuleGenome) -> dict[str, Any]:
    """Path-independent QD descriptor for the resulting semantic policy state."""
    return {
        "schema": QD_DESCRIPTOR_SCHEMA_VERSION,
        "prefix_bucket": min(11, int(child.prefix_step) // 60),
        "rule_count": len(child.rules),
        "condition_counts": [len(r.conditions) for r in child.rules],
        "route_signature": [int(r.route) for r in child.rules],
        "feature_families": [[_feature_family(c.feature) for c in r.conditions] for r in child.rules],
    }


def behavior_descriptor(pair: PairEvaluation) -> dict[str, Any]:
    changed = sorted(pair.changed_case_ids)
    digest = hashlib.sha256(("\n".join(changed)).encode()).hexdigest()[:12]
    a = pair.certified_aggregate()
    return {
        "changed_bucket": min(15, len(changed) // max(1, len(pair.case_rows) // 16 or 1)),
        "changed_signature": digest,
        "l2w_bucket": min(8, int(a["l2w_count"])),
        "w2l_bucket": min(8, int(a["w2l_count"])),
    }


def _niche(prefix: str, *parts: Mapping[str, Any]) -> str:
    raw = canonical_json({"prefix": prefix, "parts": [dict(p) for p in parts]})
    return f"{prefix}:{hashlib.sha256(raw.encode()).hexdigest()[:20]}"


def stable_motif_identity(template: Mapping[str, Any]) -> str:
    raw = canonical_json(dict(template))
    return "motif:" + hashlib.sha256(("BKG4-SEARCH-MOTIF-V1\0" + raw).encode()).hexdigest()


@dataclass(frozen=True)
class ConsiderResult:
    production_inserted: bool
    exploration_inserted: bool
    production_niche: str | None
    exploration_niche: str | None

@dataclass
class SearchRuntimeState:
    schema_version: int = 1
    last_proposed_generation: int = -1
    last_evaluated_generation: int = -1
    stagnation: dict[str, int] = field(default_factory=lambda: {"level": 0, "stalled": 0, "patience": 3, "max_level": 6})
    last_proposal_stats: dict[str, Any] = field(default_factory=dict)
    proposal_history: list[dict[str, Any]] = field(default_factory=list)
    exported_generations: list[int] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "last_proposed_generation": self.last_proposed_generation,
            "last_evaluated_generation": self.last_evaluated_generation,
            "stagnation": dict(self.stagnation),
            "last_proposal_stats": json.loads(json.dumps(self.last_proposal_stats)),
            "proposal_history": json.loads(json.dumps(self.proposal_history[-64:])),
            "exported_generations": sorted(set(int(x) for x in self.exported_generations)),
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any] | None) -> "SearchRuntimeState":
        obj = dict(raw or {})
        if int(obj.get("schema_version", 1)) != 1:
            raise ValueError("incompatible search runtime-state schema")
        stagnation = dict(obj.get("stagnation") or {"level":0,"stalled":0,"patience":3,"max_level":6})
        if set(stagnation) != {"level", "stalled", "patience", "max_level"}:
            raise ValueError("invalid persisted stagnation state")
        values = {k:int(v) for k,v in stagnation.items()}
        if values["level"] < 0 or values["stalled"] < 0 or values["patience"] < 1 or values["max_level"] < values["level"]:
            raise ValueError("invalid persisted stagnation values")
        exported = [int(x) for x in obj.get("exported_generations", ())]
        if any(x < 0 for x in exported) or len(set(exported)) != len(exported):
            raise ValueError("invalid exported-generation ledger")
        return cls(
            schema_version=1,
            last_proposed_generation=int(obj.get("last_proposed_generation", -1)),
            last_evaluated_generation=int(obj.get("last_evaluated_generation", -1)),
            stagnation=values,
            last_proposal_stats=dict(obj.get("last_proposal_stats") or {}),
            proposal_history=[dict(x) for x in obj.get("proposal_history", ())],
            exported_generations=sorted(exported),
        )


class SearchArchive:
    """Compact persistent search/QD state. Exact large corpus rows live in exported shards."""

    def __init__(self) -> None:
        self.schema_version = SEARCH_STATE_SCHEMA_VERSION
        self.grammar_id = GRAMMAR_ID
        self.emitter_registry_version = EMITTER_REGISTRY_VERSION
        self.qd_descriptor_schema_version = QD_DESCRIPTOR_SCHEMA_VERSION
        self.base_identity = ""
        self.root_semantic_id = ""
        self.route_ids: tuple[int, ...] = ()
        self.run_id = ""
        self.master_seed = 0
        self.source_identity = "unknown"
        self.run_config: dict[str, Any] = {}
        self.states: dict[str, dict[str, Any]] = {}
        self.programs: dict[str, EditProgram] = {}
        self.production_niches: dict[str, dict[str, Any]] = {}
        self.exploration_niches: dict[str, dict[str, Any]] = {}
        self.novelty_archive: list[dict[str, Any]] = []
        self.hard_cases: dict[str, dict[str, Any]] = {}
        self.motifs: dict[str, dict[str, Any]] = {}
        self.emitter_stats: dict[str, dict[str, int]] = {}
        self.evaluated_semantic_ids: set[str] = set()
        self.escalation_history: list[dict[str, Any]] = []
        self.state_evaluations: dict[str, dict[str, dict[str, Any]]] = {}
        self.runtime_state = SearchRuntimeState()

    @classmethod
    def new(
        cls, *, root_genome: ModuleGenome, base_identity: str, route_ids: Sequence[int], run_id: str, master_seed: int,
        source_identity: str = "unknown", run_config: Mapping[str, Any] | None = None
    ) -> "SearchArchive":
        obj = cls()
        root_genome.validate(route_ids)
        obj.base_identity = str(base_identity)
        obj.root_semantic_id = root_genome.semantic_id()
        obj.route_ids = tuple(int(x) for x in route_ids)
        obj.run_id = str(run_id)
        obj.master_seed = int(master_seed)
        obj.source_identity = str(source_identity)
        obj.run_config = dict(run_config or {})
        obj.states[obj.root_semantic_id] = root_genome.canonical()
        return obj

    def genome(self, semantic_id: str) -> ModuleGenome:
        try:
            return genome_from_json(self.states[str(semantic_id)])
        except KeyError as exc:
            raise KeyError(f"unknown search state {semantic_id}") from exc

    def add_program(self, program: EditProgram, child: ModuleGenome) -> bool:
        if program.child_semantic_id != child.semantic_id():
            raise ValueError("program/child semantic identity mismatch")
        if program.source_parent_semantic_id not in self.states:
            raise ValueError("program source parent is not in search archive")
        source = self.genome(program.source_parent_semantic_id)
        rebuilt = program.reconstruct(source, self.route_ids)
        if rebuilt.semantic_id() != child.semantic_id():
            raise ValueError("program reconstruction mismatch")
        fresh = program.program_id not in self.programs
        previous = self.programs.get(program.program_id)
        if previous is not None and previous.canonical(include_provenance=False) != program.canonical(include_provenance=False):
            raise ValueError("program ID collision")
        self.programs.setdefault(program.program_id, program)
        old_state = self.states.get(child.semantic_id())
        if old_state is not None and canonical_json(old_state) != canonical_json(child.canonical()):
            raise ValueError("semantic state ID collision")
        self.states.setdefault(child.semantic_id(), child.canonical())
        return fresh

    def _elite_record(self, program: EditProgram, pair: PairEvaluation, child: ModuleGenome) -> dict[str, Any]:
        fit = LexicographicFitness.from_pair(pair, complexity=len(child.rules) + program.program_length)
        return {
            "program_id": program.program_id,
            "alternate_program_ids": [],
            "child_semantic_id": child.semantic_id(),
            "fitness_key": list(fit.key()),
            "strict_safe": fit.strict_safe,
            "aggregate": pair.certified_aggregate(),
            "structural": structural_descriptor(program, child),
            "policy_state": policy_state_descriptor(child),
            "behavior": behavior_descriptor(pair),
            "generation_depth": program.generation_depth,
        }

    @staticmethod
    def _child_niche(table: Mapping[str, Mapping[str, Any]], child_semantic_id: str) -> str | None:
        for niche, row in table.items():
            if str(row.get("child_semantic_id")) == str(child_semantic_id):
                return str(niche)
        return None

    @staticmethod
    def _preserve_alternate_program(record: dict[str, Any], program_id: str) -> None:
        if str(record.get("program_id")) == str(program_id):
            return
        alt = set(str(x) for x in record.get("alternate_program_ids", ()))
        alt.add(str(program_id)); record["alternate_program_ids"] = sorted(alt)

    def consider(self, program: EditProgram, pair: PairEvaluation) -> ConsiderResult:
        if pair.child_semantic_id != program.child_semantic_id:
            raise ValueError("pair/program child mismatch")
        child = self.genome(program.child_semantic_id)
        rec = self._elite_record(program, pair, child)
        fit = LexicographicFitness.from_pair(pair, complexity=len(child.rules) + program.program_length)
        agg = pair.certified_aggregate()
        prod_niche = _niche("production", rec["policy_state"])
        exp_niche = _niche("exploration", rec["policy_state"], rec["behavior"])
        production_inserted = False
        exploration_inserted = False

        existing_prod = self._child_niche(self.production_niches, child.semantic_id())
        if fit.strict_safe and bool(agg["behavior_changing"]):
            target_niche = existing_prod or prod_niche
            old = self.production_niches.get(target_niche)
            if old is None or tuple(rec["fitness_key"]) < tuple(old["fitness_key"]):
                if old is not None and str(old.get("child_semantic_id")) == child.semantic_id():
                    self._preserve_alternate_program(rec, str(old.get("program_id")))
                    for alt in old.get("alternate_program_ids", ()): self._preserve_alternate_program(rec, str(alt))
                self.production_niches[target_niche] = rec
                production_inserted = old is None or str(old.get("child_semantic_id")) != child.semantic_id()
            elif old is not None and str(old.get("child_semantic_id")) == child.semantic_id():
                self._preserve_alternate_program(old, program.program_id)

        exploration_key = (
            int(agg["new_loss_count"]), int(agg["w2l_count"]), -int(agg["l2w_count"]),
            -len(pair.changed_case_ids), -float(agg["mean_margin_delta"]),
        )
        existing_exp = self._child_niche(self.exploration_niches, child.semantic_id())
        target_exp = existing_exp or exp_niche
        old = self.exploration_niches.get(target_exp)
        if old is None or exploration_key < tuple(old.get("exploration_key", (10**9, 10**9, 0, 0, 0))):
            exp_rec = dict(rec); exp_rec["exploration_key"] = list(exploration_key)
            if old is not None and str(old.get("child_semantic_id")) == child.semantic_id():
                self._preserve_alternate_program(exp_rec, str(old.get("program_id")))
                for alt in old.get("alternate_program_ids", ()): self._preserve_alternate_program(exp_rec, str(alt))
            self.exploration_niches[target_exp] = exp_rec
            exploration_inserted = old is None or str(old.get("child_semantic_id")) != child.semantic_id()
        elif old is not None and str(old.get("child_semantic_id")) == child.semantic_id():
            self._preserve_alternate_program(old, program.program_id)

        self.evaluated_semantic_ids.add(program.child_semantic_id)
        if pair.changed_case_ids:
            novelty = {
                "program_id": program.program_id,
                "child_semantic_id": program.child_semantic_id,
                "changed_signature": rec["behavior"]["changed_signature"],
                "changed_cases": len(pair.changed_case_ids),
            }
            if not any(str(x.get("child_semantic_id")) == child.semantic_id() or x.get("changed_signature") == novelty["changed_signature"] for x in self.novelty_archive):
                self.novelty_archive.append(novelty)
        return ConsiderResult(production_inserted, exploration_inserted, prod_niche if production_inserted else None, exp_niche if exploration_inserted else None)

    def record_counterexample(self, case_id: str, *, reason: str, severity: int, panel_id: str) -> None:
        cid = str(case_id)
        row = self.hard_cases.setdefault(cid, {"count": 0, "severity": 0, "reasons": {}, "panels": {}})
        row["count"] = int(row["count"]) + 1
        row["severity"] = max(int(row["severity"]), int(severity))
        row["reasons"][str(reason)] = int(row["reasons"].get(str(reason), 0)) + 1
        row["panels"][str(panel_id)] = int(row["panels"].get(str(panel_id), 0)) + 1

    def record_state_evaluation(self, evaluation: StateEvaluation) -> None:
        sid = str(evaluation.semantic_id)
        if sid not in self.states:
            raise ValueError("cannot certify unknown search state")
        by_panel = self.state_evaluations.setdefault(sid, {})
        existing = by_panel.get(evaluation.panel_id)
        payload = evaluation.to_dict()
        if existing is not None and existing != payload:
            raise ValueError("conflicting exact state evaluation for the same panel")
        by_panel[evaluation.panel_id] = payload

    def state_evaluation(self, semantic_id: str, panel_id: str) -> StateEvaluation:
        try:
            return StateEvaluation.from_dict(self.state_evaluations[str(semantic_id)][str(panel_id)])
        except KeyError as exc:
            raise KeyError(f"missing exact state evaluation {semantic_id} on panel {panel_id}") from exc

    def observe_motif(self, program: EditProgram, pair: PairEvaluation) -> str | None:
        a = pair.certified_aggregate()
        useful = bool(a["strict_safe"]) and bool(a["behavior_changing"]) and (
            int(a["l2w_count"]) > 0 or float(a["mean_margin_delta"]) > 0.0
        )
        if not useful:
            return None
        template_atoms = []
        for atom in program.atoms:
            action = dict(atom.action)
            action.pop("target_node", None)
            template_atoms.append({
                "operation": atom.operation,
                "target_kind": atom.target_kind,
                "target_rule_index_hint": atom.target_rule_index,
                "target_condition_index_hint": atom.target_condition_index,
                "action": action,
            })
        template = {"atoms": template_atoms, "program_length": program.program_length}
        motif_id = stable_motif_identity(template)
        row = self.motifs.setdefault(motif_id, {
            "motif_id": motif_id,
            "template": template,
            "observations": 0,
            "safe_useful_hits": 0,
            "l2w_total": 0,
            "mean_delta_sum": 0.0,
            "source_programs": [],
        })
        row["observations"] = int(row["observations"]) + 1
        row["safe_useful_hits"] = int(row["safe_useful_hits"]) + 1
        row["l2w_total"] = int(row["l2w_total"]) + int(a["l2w_count"])
        row["mean_delta_sum"] = float(row["mean_delta_sum"]) + float(a["mean_margin_delta"])
        if program.program_id not in row["source_programs"]:
            row["source_programs"].append(program.program_id)
            row["source_programs"].sort()
        return motif_id

    def note_emitter(self, name: str, *, attempts: int = 0, unique: int = 0, archive_insertions: int = 0, useful_safe: int = 0) -> None:
        row = self.emitter_stats.setdefault(str(name), {"attempts": 0, "unique": 0, "archive_insertions": 0, "useful_safe": 0})
        for key, value in (("attempts", attempts), ("unique", unique), ("archive_insertions", archive_insertions), ("useful_safe", useful_safe)):
            row[key] = int(row[key]) + int(value)

    def parent_state_ids(self) -> list[str]:
        ids = {self.root_semantic_id}
        for table in (self.production_niches, self.exploration_niches):
            ids.update(str(x["child_semantic_id"]) for x in table.values())
        ids.update(str(x["child_semantic_id"]) for x in self.novelty_archive)
        return sorted(ids)

    def mark_exported_generation(self, generation: int) -> None:
        g = int(generation)
        if g < 0:
            raise ValueError("generation must be non-negative")
        if g not in self.runtime_state.exported_generations:
            self.runtime_state.exported_generations.append(g)
            self.runtime_state.exported_generations.sort()

    def validate_integrity(self) -> None:
        if not self.base_identity or not self.run_id or not self.route_ids or len(set(self.route_ids)) != len(self.route_ids):
            raise ValueError("invalid archive identity/runtime metadata")
        if self.root_semantic_id not in self.states:
            raise ValueError("search archive root state missing")
        for sid, raw in self.states.items():
            genome = genome_from_json(raw); genome.validate(self.route_ids)
            if genome.semantic_id() != sid:
                raise ValueError(f"archive state semantic identity mismatch: {sid}")
        for pid, program in self.programs.items():
            if pid != program.program_id or program.source_parent_semantic_id not in self.states or program.child_semantic_id not in self.states:
                raise ValueError("search archive program references invalid state")
            program.reconstruct(self.genome(program.source_parent_semantic_id), self.route_ids)
            if program.parent_program_id is not None:
                parent_program = self.programs.get(str(program.parent_program_id))
                if parent_program is None or parent_program.child_semantic_id != program.source_parent_semantic_id:
                    raise ValueError("search archive lineage parent is dangling/incompatible")
        for kind, table in (("production", self.production_niches), ("exploration", self.exploration_niches)):
            seen_children: set[str] = set()
            for niche, row in table.items():
                sid = str(row.get("child_semantic_id", "")); pid = str(row.get("program_id", ""))
                if sid not in self.states or pid not in self.programs or self.programs[pid].child_semantic_id != sid:
                    raise ValueError(f"{kind} QD niche has dangling state/program reference")
                if sid in seen_children:
                    raise ValueError(f"{kind} QD contains duplicate semantic child")
                seen_children.add(sid)
                expected_state = policy_state_descriptor(self.genome(sid))
                if dict(row.get("policy_state") or {}) != expected_state:
                    raise ValueError(f"{kind} QD policy-state descriptor mismatch")
                expected_niche = _niche(kind, expected_state) if kind == "production" else _niche(kind, expected_state, dict(row.get("behavior") or {}))
                if niche != expected_niche:
                    raise ValueError(f"{kind} QD niche identity mismatch")
                if kind == "production" and not bool(row.get("strict_safe")):
                    raise ValueError("production QD contains non-safe elite")
                for alt in row.get("alternate_program_ids", ()):
                    if str(alt) not in self.programs or self.programs[str(alt)].child_semantic_id != sid:
                        raise ValueError(f"{kind} QD alternate program is dangling")
        novelty_children: set[str] = set()
        for row in self.novelty_archive:
            sid = str(row.get("child_semantic_id", "")); pid = str(row.get("program_id", ""))
            if sid not in self.states or pid not in self.programs or self.programs[pid].child_semantic_id != sid or sid in novelty_children:
                raise ValueError("novelty archive has dangling/duplicate semantic state")
            novelty_children.add(sid)
        if not self.evaluated_semantic_ids.issubset(self.states):
            raise ValueError("evaluated-state ledger references unknown state")
        for sid, panels in self.state_evaluations.items():
            if sid not in self.states:
                raise ValueError("exact certification references unknown state")
            for panel_id, payload in panels.items():
                ev = StateEvaluation.from_dict(payload)
                if ev.semantic_id != sid or ev.panel_id != panel_id:
                    raise ValueError("exact certification identity mismatch")
        for motif_id, row in self.motifs.items():
            template = dict(row.get("template") or {})
            if motif_id != stable_motif_identity(template) or str(row.get("motif_id")) != motif_id:
                raise ValueError("motif identity mismatch")
            if any(str(pid) not in self.programs for pid in row.get("source_programs", ())):
                raise ValueError("motif source program is dangling")
        self.runtime_state = SearchRuntimeState.from_dict(self.runtime_state.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "grammar_id": self.grammar_id,
            "emitter_registry_version": self.emitter_registry_version,
            "qd_descriptor_schema_version": self.qd_descriptor_schema_version,
            "base_identity": self.base_identity,
            "root_semantic_id": self.root_semantic_id,
            "route_ids": list(self.route_ids),
            "run_id": self.run_id,
            "master_seed": self.master_seed,
            "source_identity": self.source_identity,
            "run_config": self.run_config,
            "states": self.states,
            "programs": {k: v.canonical(include_provenance=True) for k, v in self.programs.items()},
            "production_niches": self.production_niches,
            "exploration_niches": self.exploration_niches,
            "novelty_archive": self.novelty_archive,
            "hard_cases": self.hard_cases,
            "motifs": self.motifs,
            "emitter_stats": self.emitter_stats,
            "evaluated_semantic_ids": sorted(self.evaluated_semantic_ids),
            "escalation_history": self.escalation_history,
            "state_evaluations": self.state_evaluations,
            "runtime_state": self.runtime_state.to_dict(),
        }

    def save(self, path: str | Path) -> None:
        self.validate_integrity()
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(target.suffix + ".tmp")
        tmp.write_text(json.dumps(self.to_dict(), sort_keys=True, indent=2) + "\n", encoding="utf-8")
        tmp.replace(target)

    @classmethod
    def load(cls, path: str | Path) -> "SearchArchive":
        obj = json.loads(Path(path).read_text(encoding="utf-8"))
        if int(obj.get("schema_version", -1)) != SEARCH_STATE_SCHEMA_VERSION:
            raise ValueError("incompatible search-state schema")
        if str(obj.get("grammar_id")) != GRAMMAR_ID:
            raise ValueError("incompatible search grammar")
        if int(obj.get("emitter_registry_version", -1)) != EMITTER_REGISTRY_VERSION:
            raise ValueError("incompatible emitter registry")
        if int(obj.get("qd_descriptor_schema_version", -1)) != QD_DESCRIPTOR_SCHEMA_VERSION:
            raise ValueError("incompatible QD descriptor schema")
        out = cls()
        out.base_identity = str(obj["base_identity"])
        out.root_semantic_id = str(obj["root_semantic_id"])
        out.route_ids = tuple(int(x) for x in obj["route_ids"])
        out.run_id = str(obj["run_id"])
        out.master_seed = int(obj["master_seed"])
        out.source_identity = str(obj.get("source_identity", "unknown"))
        out.run_config = dict(obj.get("run_config") or {})
        out.states = {str(k): dict(v) for k, v in obj["states"].items()}
        out.programs = {str(k): EditProgram.from_dict(v) for k, v in obj["programs"].items()}
        out.production_niches = {str(k): dict(v) for k, v in obj.get("production_niches", {}).items()}
        out.exploration_niches = {str(k): dict(v) for k, v in obj.get("exploration_niches", {}).items()}
        out.novelty_archive = [dict(x) for x in obj.get("novelty_archive", [])]
        out.hard_cases = {str(k): dict(v) for k, v in obj.get("hard_cases", {}).items()}
        out.motifs = {str(k): dict(v) for k, v in obj.get("motifs", {}).items()}
        out.emitter_stats = {str(k): {str(a): int(b) for a, b in v.items()} for k, v in obj.get("emitter_stats", {}).items()}
        out.evaluated_semantic_ids = set(str(x) for x in obj.get("evaluated_semantic_ids", ()))
        out.escalation_history = [dict(x) for x in obj.get("escalation_history", [])]
        out.state_evaluations = {
            str(sid): {str(panel): dict(payload) for panel, payload in panels.items()}
            for sid, panels in obj.get("state_evaluations", {}).items()
        }
        out.runtime_state = SearchRuntimeState.from_dict(obj.get("runtime_state"))
        out.validate_integrity()
        return out
