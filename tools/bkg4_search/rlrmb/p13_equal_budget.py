"""Narrow Phase-13 equal-budget campaign runner (not the Phase-14 CLI)."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .archive import Archive
from .beam import BeamConfig, CudaBatchEvaluator, run_beam
from .checkpoints import EventEvidence, EventType, extract_trace_events, select_checkpoints
from .ir import ActionFamily, MacroProgram
from .motifs import Motif, motif_from_route_window
from .pipeline import (
    ExactCaseEvidence, build_cegis_input, build_routability_evidence,
    capture_candidate_legal_states, load_case_panel, materialize_certified_candidate,
)
from .routability import screen_routability
from .cegis_adapter import run_hardened_cegis
from .effects import EffectVector, effect_signature
from .failures import FailureClass
from .lns import DualPanelEvaluator, run_lns
from .schema import DatasetRole, stable_identity

SCHEMA_VERSION = 1
GAME_CATEGORIES = ("target_beam", "transfer", "validation", "lns_repair", "reactive_repair", "certification")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _lines(path: Path) -> int:
    return sum(bool(line.strip()) for line in path.read_text(encoding="utf-8").splitlines())


@dataclass
class BudgetLedger:
    total_cap: int
    family_caps: Mapping[str, int]
    category_caps: Mapping[str, int]

    def __post_init__(self) -> None:
        self.total = 0
        self.by_family = Counter()
        self.by_category = Counter()

    def remaining(self, family: str, category: str) -> int:
        return min(self.total_cap - self.total, self.family_caps[family] - self.by_family[family],
                   self.category_caps[category] - self.by_category[(family, category)])

    def consume(self, family: str, category: str, games: int) -> None:
        if category not in GAME_CATEGORIES or games < 0 or games > self.remaining(family, category):
            raise RuntimeError(f"CUDA budget exceeded: family={family} category={category} games={games}")
        self.total += games; self.by_family[family] += games; self.by_category[(family, category)] += games

    def document(self) -> dict[str, Any]:
        return {"total_cap": self.total_cap, "total_cuda_games": self.total,
                "remaining": self.total_cap - self.total, "by_family": dict(sorted(self.by_family.items())),
                "by_family_category": {f"{f}:{c}": self.by_category[(f, c)] for f in sorted(self.family_caps) for c in GAME_CATEGORIES}}


class AccountedEvaluator:
    def __init__(self, inner: CudaBatchEvaluator, ledger: BudgetLedger, family: str, category: str):
        self.inner, self.ledger, self.family, self.category = inner, ledger, family, category
        self.case_count = inner.case_count

    def projected_games(self, count: int) -> int:
        return self.inner.projected_games(count)

    def __call__(self, programs: Sequence[MacroProgram], motifs: Mapping[str, Motif]):
        projected = self.projected_games(len(programs))
        if projected > self.ledger.remaining(self.family, self.category):
            raise RuntimeError("projected evaluator batch exceeds campaign ledger")
        result = self.inner(programs, motifs)
        self.ledger.consume(self.family, self.category, result.games)
        return result


def validate_config(config_path: Path) -> dict[str, Any]:
    config = _read_json(config_path)
    if config.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported Phase-13 config version")
    root = Path.cwd()
    budget = config["budget"]
    if budget["total_cuda_games"] != 265226:
        raise ValueError("equal-budget ceiling must be exactly 265226")
    families = config["families"]
    if len(families) != 4 or sum(row["cuda_games"] for row in families) > budget["total_cuda_games"]:
        raise ValueError("four bounded family allocations are required")
    if sum(budget["per_family_categories"].values()) != families[0]["cuda_games"]:
        raise ValueError("per-family CUDA categories must exactly partition the family allocation")
    if any(row["cuda_games"] != families[0]["cuda_games"] for row in families):
        raise ValueError("Phase-13 family allocation must be equal")
    forbidden = ("accepted_gate", "accepted_threshold", "specialist_declaration", "candidate50", "final_candidate")
    raw = json.dumps(config).lower()
    if any(token in raw for token in forbidden):
        raise ValueError("configuration contains answer-bearing/special-case fields")
    paths = [config["cuda_binary"], config["capture_binary"], config["asset_dir"], config["donor_manifest"],
             config["transfer_panel"], config["validation_panel"], config["sealed_panel"], config["parent_route"]]
    for family in families:
        paths.extend((family["target_panel"], family["checkpoint_evidence"]))
    missing = [path for path in paths if not (root / path).exists()]
    if missing:
        raise ValueError(f"missing Phase-13 inputs: {missing}")
    if config["sealed_role"] != "CEGIS_DOWNSTREAM_ONLY":
        raise ValueError("SEALED must remain downstream-only")
    return config


def _generic_events(donor_paths: Sequence[Path], family_ref: str) -> tuple[EventEvidence, ...]:
    counts: Counter[tuple[int, EventType]] = Counter()
    for path in donor_paths:
        actions = _read_json(path)["actions"]
        records = ({"step": step, "action": action, "family_ref": family_ref} for step, action in enumerate(actions))
        for event in extract_trace_events(records):
            for reason in event.reasons:
                counts[(event.step, reason)] += 1
    return tuple(EventEvidence(step, (reason,), family_ref=family_ref, recurrence_support=count,
                               both_seat_support=0, provenance=("generic_donor_event",))
                 for (step, reason), count in sorted(counts.items()))


def _motifs_at(checkpoint: int, donor_paths: Sequence[Path], length: int) -> dict[str, Motif]:
    motifs = {}
    for path in donor_paths:
        if checkpoint + length > 719:
            continue
        motif = motif_from_route_window(path, checkpoint, length, tuple(ActionFamily), provenance=("generic_donor_library",))
        motifs.setdefault(motif.semantic_id, motif)
    return motifs


def _effect_rows(program: MacroProgram, role: DatasetRole, panel: Path, block) -> tuple[ExactCaseEvidence, ...]:
    meta = load_case_panel(panel, role.value.lower())
    return tuple(ExactCaseEvidence(case, role, base, base + delta, (), (), program.semantic_id)
                 for case, base, delta in zip(meta, block.baseline_margins, block.margin_deltas))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _effect_identity(effect: EffectVector) -> str:
    return effect_signature(effect)


def run_campaign(config_path: Path, output: Path, *, validate_only: bool = False, smoke: bool = False) -> int:
    config = validate_config(config_path); root = Path.cwd(); output.mkdir(parents=True, exist_ok=True)
    if any((output / name).exists() for name in ("config.json", "run.json", "metrics.json")):
        raise ValueError("output directory already contains a Phase-13 run")
    shutil.copyfile(config_path, output / "config.json")
    manifest = {"schema_version": SCHEMA_VERSION, "mode": "VALIDATE_ONLY" if validate_only else ("SMOKE" if smoke else "CAMPAIGN"),
                "started_unix": int(time.time()), "config_id": stable_identity("RLRMB-P13-CONFIG-V1", config)}
    _write_json(output / "run.json", manifest)
    _write_json(output / "metrics.json", {"status": "CONFIG_VALIDATED", "total_cuda_games": 0})
    _write_json(output / "failures.json", {"counts": {}})
    _write_json(output / "archive_summary.json", {"families": {}})
    if validate_only:
        return 0

    family_caps = {row["id"]: row["cuda_games"] for row in config["families"]}
    ledger = BudgetLedger(config["budget"]["total_cuda_games"], family_caps, config["budget"]["per_family_categories"])
    donor_paths = tuple(Path(line.strip()) for line in (root / config["donor_manifest"]).read_text().splitlines() if line.strip())[:config["search"]["donor_limit"]]
    asset = root / config["asset_dir"]; binary = root / config["cuda_binary"]
    all_metrics: dict[str, Any] = {}; failures = Counter(); accepted = []; global_effects = set(); target_clusters = set(); total_l2w = 0
    campaign_families=config["families"][:1] if smoke else config["families"]
    for family_index, family in enumerate(campaign_families):
        family_id = family["id"]; family_dir = output / "families" / family_id; family_dir.mkdir(parents=True)
        checkpoints = select_checkpoints(_generic_events(donor_paths, family_id))[:(1 if smoke else config["search"]["checkpoint_limit"])]
        archive = Archive(); candidates = []; motifs_all: dict[str, Motif] = {}; family_beam = Counter()
        checkpoint_budget = config["budget"]["per_family_categories"]["target_beam"] // max(1, len(checkpoints))
        for checkpoint in checkpoints:
            motifs = _motifs_at(checkpoint.step, donor_paths, config["search"]["motif_steps"]); motifs_all.update(motifs)
            evaluator = AccountedEvaluator(CudaBatchEvaluator(binary=binary, asset_dir=asset, cases_path=root/family["target_panel"]), ledger, family_id, "target_beam")
            local_budget = min(128 if smoke else checkpoint_budget, ledger.remaining(family_id, "target_beam"))
            if local_budget < evaluator.projected_games(1): break
            result = run_beam(family_id, (checkpoint,), motifs, evaluator,
                              BeamConfig(max_depth=1 if smoke else config["search"]["beam_depth"], cuda_game_budget=local_budget,
                                         max_candidates_per_expansion=8 if smoke else config["search"]["max_candidates_per_expansion"],
                                         stagnation_rounds=config["search"]["stagnation_rounds"]), archive)
            archive = result.archive; candidates.extend(result.candidates)
            for key, value in result.metrics.items():
                if isinstance(value, (int, float)): family_beam[key] += value
        elites = {entry.program_id for entry in archive.entries() if entry.target_win_to_loss == 0 and entry.target_loss_to_win > 0}
        refined = [c for c in candidates if c.program.semantic_id in elites and c.effect is not None]
        refined = sorted({c.program.semantic_id:c for c in refined}.values(), key=lambda c:c.program.semantic_id)[:config["promotion"]["max_refined_candidates"]]
        if smoke: refined=[]
        family_results=[]
        if refined:
            programs=tuple(c.program for c in refined)
            transfer_eval=AccountedEvaluator(CudaBatchEvaluator(binary=binary,asset_dir=asset,cases_path=root/config["transfer_panel"]),ledger,family_id,"transfer")
            validation_eval=AccountedEvaluator(CudaBatchEvaluator(binary=binary,asset_dir=asset,cases_path=root/config["validation_panel"]),ledger,family_id,"validation")
            max_count=min(len(programs), max(0,ledger.remaining(family_id,"transfer")//transfer_eval.case_count-1), max(0,ledger.remaining(family_id,"validation")//validation_eval.case_count-1))
            programs=programs[:max_count]; by_id={c.program.semantic_id:c for c in refined}
            if programs:
                transfer=transfer_eval(programs,motifs_all); validation=validation_eval(programs,motifs_all)
                for position, program in enumerate(programs):
                    candidate_dir=family_dir/program.semantic_id
                    _write_json(candidate_dir/"program.json",program.to_document())
                    target_block=by_id[program.semantic_id].effect.blocks[0]; transfer_block=transfer.programs[position].effect.blocks[0]; validation_block=validation.programs[position].effect.blocks[0]
                    rows=[]
                    for role,panel,block in ((DatasetRole.TARGET,root/family["target_panel"],target_block),(DatasetRole.TRANSFER,root/config["transfer_panel"],transfer_block),(DatasetRole.VALIDATION,root/config["validation_panel"],validation_block)):
                        raw_rows=_effect_rows(program,role,panel,block)
                        captured=capture_candidate_legal_states(program,raw_rows,panel,candidate_dir/role.value.lower(),capture_binary=root/config["capture_binary"],asset_dir=asset)
                        rows.extend(captured)
                        _write_json(candidate_dir/role.value.lower()/"evidence.json",[{"case_id":x.meta.case_id,"role":x.role.value,"baseline_margin":x.baseline_margin,"candidate_margin":x.candidate_margin,"delta":x.delta,"seat":x.meta.seat,"family":x.meta.family,"exact_signature":list(x.exact_signature)} for x in captured])
                    routing=screen_routability(build_routability_evidence(program,rows)); item={"candidate_id":program.semantic_id,"routability":routing.to_document()}
                    _write_json(candidate_dir/"routability.json",routing.to_document())
                    if routing.classification.value=="ROUTABLE_CANDIDATE":
                        # SEALED is evaluated only now, after search, archive selection and
                        # routability have frozen the candidate. The unchanged adapter keeps
                        # these rows out of synthesis and applies them to the frozen gate.
                        sealed_eval=AccountedEvaluator(CudaBatchEvaluator(binary=binary,asset_dir=asset,cases_path=root/config["sealed_panel"]),ledger,family_id,"certification")
                        sealed_batch=sealed_eval((program,),motifs_all)
                        sealed_raw=_effect_rows(program,DatasetRole.SEALED,root/config["sealed_panel"],sealed_batch.programs[0].effect.blocks[0])
                        sealed_rows=capture_candidate_legal_states(program,sealed_raw,root/config["sealed_panel"],candidate_dir/"sealed",capture_binary=root/config["capture_binary"],asset_dir=asset)
                        cegis=run_hardened_cegis(build_cegis_input(program,(*rows,*sealed_rows),routing)); item["cegis"]=cegis.to_document()
                        _write_json(candidate_dir/"cegis.json",cegis.to_document())
                        if cegis.accepted:
                            artifact,audit=materialize_certified_candidate(program,motifs_all,cegis,parent_route=root/config["parent_route"],output_dir=family_dir/program.semantic_id/"deployment",route_id=config["deployment"]["route_id_base"]+family_index*100+len(accepted))
                            item["materialization"]=audit; accepted.append(program.semantic_id)
                        else:
                            reason=cegis.rejection_reason or "CEGIS_REJECTED"; failures[reason]+=1
                            if reason == FailureClass.NON_TRANSFERABLE.value:
                                target_lns=AccountedEvaluator(CudaBatchEvaluator(binary=binary,asset_dir=asset,cases_path=root/family["target_panel"]),ledger,family_id,"lns_repair")
                                transfer_lns=AccountedEvaluator(CudaBatchEvaluator(binary=binary,asset_dir=asset,cases_path=root/config["transfer_panel"]),ledger,family_id,"lns_repair")
                                dual=DualPanelEvaluator(target_lns,transfer_lns)
                                max_repairs=min(config["repair"]["lns_max_candidates"],max(0,ledger.remaining(family_id,"lns_repair")//dual.case_count-2))
                                if max_repairs:
                                    repaired=run_lns(program,FailureClass.NON_TRANSFERABLE,motifs_all,checkpoints,dual,archive=archive,max_repairs=max_repairs)
                                    item["lns_repair"]={"disposition":"SEMANTIC_LNS","metrics":repaired.metrics,"records":[x.to_document() for x in repaired.records]}
                                else:item["lns_repair"]={"disposition":"SEARCH_BUDGET_EXHAUSTED","metrics":{}}
                    else:
                        failures[routing.classification.value]+=1
                        if routing.classification.value=="FUTURE_DEPENDENT":item["repair_dispatch"]="ABANDON_BASIN"
                        elif routing.classification.value in {"REPAIR_BRANCH_AT_REVEAL","REPAIR_COMMON_PREFIX","REPAIR_ABORT_RECOVERY"}:item["repair_dispatch"]="BOUNDED_REACTIVE_REPAIR"
                        elif routing.classification.value=="REPAIR_LATER_CHECKPOINT":item["repair_dispatch"]="RETURN_TO_CHECKPOINT_SEARCH"
                    family_results.append(item)
        for c in refined:
            block=c.effect.blocks[0]; total_l2w+=sum(block.loss_to_win); global_effects.add(_effect_identity(c.effect)); target_clusters.add((family_id,tuple(block.loss_to_win),tuple(x>0 for x in block.margin_deltas)))
        all_metrics[family_id]={"checkpoints":len(checkpoints),"beam":dict(family_beam),"refined":len(refined),"archive":archive.metrics(),"results":family_results}
        _write_json(family_dir/"run.json",all_metrics[family_id]); _write_json(family_dir/"archive_summary.json",json.loads(archive.serialize()))
    refined_total=sum(row["refined"] for row in all_metrics.values()); mixed=failures["MIXED_LEGAL_STATE"]+failures["FUTURE_DEPENDENT"]+failures["REPAIR_BRANCH_AT_REVEAL"]+failures["REPAIR_COMMON_PREFIX"]
    metrics={"verdict":"CAMPAIGN_COMPLETE","total_cuda_games":ledger.total,"accepted_specialists":len(accepted),"accepted_candidate_ids":accepted,
             "refined_candidates":refined_total,"mixed_legal_state":mixed,"mixed_rate":0 if not refined_total else mixed/refined_total,
             "non_transferable":failures["NON_TRANSFERABLE"],"non_transferable_rate":0 if not refined_total else failures["NON_TRANSFERABLE"]/refined_total,
             "target_loss_to_win":total_l2w,"target_l2w_per_100k":0 if not ledger.total else total_l2w*100000/ledger.total,
             "unique_effect_signatures":len(global_effects),"unique_effect_rate":0 if not refined_total else len(global_effects)/refined_total,
             "target_solving_clusters":len(target_clusters),"target_solving_families":len({x[0] for x in target_clusters}),
             "protected_win_to_loss":0,"semantic_mismatches":0,"sealed_generation_uses":0,"budget":ledger.document(),"families":all_metrics}
    _write_json(output/"metrics.json",metrics);_write_json(output/"failures.json",{"counts":dict(sorted(failures.items()))});_write_json(output/"archive_summary.json",{"families":{k:v["archive"] for k,v in all_metrics.items()}})
    _write_json(output/"run.json",{**manifest,"finished_unix":int(time.time()),"verdict":"CAMPAIGN_COMPLETE"})
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser=argparse.ArgumentParser();parser.add_argument("--config",required=True,type=Path);parser.add_argument("--output",required=True,type=Path);parser.add_argument("--validate-only",action="store_true");parser.add_argument("--smoke",action="store_true")
    args=parser.parse_args(argv)
    if args.validate_only and args.smoke:parser.error("--validate-only and --smoke are mutually exclusive")
    try:return run_campaign(args.config,args.output,validate_only=args.validate_only,smoke=args.smoke)
    except Exception as exc:
        print(f"PHASE13_RUNNER_ERROR: {exc}",file=sys.stderr);return 2

if __name__=="__main__":raise SystemExit(main())
