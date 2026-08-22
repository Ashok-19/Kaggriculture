from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from tools.bkg4_search.bonsai_editformer.actor_grammar import EditAction, node_refs
from tools.bkg4_search.module_codec import ConditionGene

from .archive import SearchArchive
from .fitness import PairEvaluation
from .goals import SearchGoal, synthesize_contrastive_conditions, temporal_specificity
from .panels import CaseBank, CasePanelManager
from .safety import SafetyAssessment, SafetySpec
from .schema import EditProgram


@dataclass(frozen=True)
class Repairability:
    counterexamples: int
    coherent_separator_count: int
    remaining_condition_capacity: int
    temporal_specificity_rank: int

    def key(self) -> tuple[int, int, int, int]:
        return (
            int(self.counterexamples),
            -int(self.coherent_separator_count),
            -int(self.remaining_condition_capacity),
            int(self.temporal_specificity_rank),
        )


@dataclass(frozen=True)
class RepairReport:
    attempts: int
    less_infeasible: int
    became_feasible: int
    became_safe_useful: int
    became_elite: int
    became_certified_useful: int
    rounds: int
    chain_program_ids: tuple[str, ...]
    terminal_program_id: str | None


def synthesize_counterexample_guards(
    target_case_ids: Sequence[str],
    counterexample_ids: Sequence[str],
    case_contexts: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    fragile_case_ids: Sequence[str] = (),
    max_conditions: int = 2,
    limit: int = 32,
) -> list[tuple[ConditionGene, ...]]:
    targets = tuple(dict.fromkeys(str(x) for x in target_case_ids))
    counters = tuple(dict.fromkeys(str(x) for x in counterexample_ids))
    if not targets or not counters:
        return []
    goal = SearchGoal(
        target_case_ids=targets,
        protected_guard_ids=counters,
        fragile_win_ids=tuple(str(x) for x in fragile_case_ids if str(x) in counters),
        counterexample_ids=counters,
        desired_repair_type="counterexample_guard",
    )
    semantic = synthesize_contrastive_conditions(
        goal, case_contexts, max_conditions=max_conditions, limit=limit,
        guard_excluding_only=True, allow_temporal=False,
    )
    fallback = synthesize_contrastive_conditions(
        goal, case_contexts, max_conditions=max_conditions, limit=limit,
        guard_excluding_only=True, allow_temporal=True,
    )
    out: list[tuple[ConditionGene, ...]] = []
    seen: set[tuple[ConditionGene, ...]] = set()
    for row in (*semantic, *fallback):
        key = tuple(row)
        if key not in seen:
            seen.add(key); out.append(key)
    out.sort(key=lambda row: (len(row), temporal_specificity(row), tuple((c.feature, c.index, c.cmp, c.value) for c in row)))
    # A multi-counterexample repair should prefer the smallest conjunction that
    # excludes every supplied counterexample. synthesize_contrastive_conditions
    # already enforces zero guard hits; shorter rows therefore dominate.
    return out[: max(1, int(limit))]


def _rule_nodes(genome) -> dict[int, int]:
    out: dict[int, int] = {}
    for ref in node_refs(genome):
        if ref is not None and ref.kind == "rule" and ref.rule_index >= 0:
            out.setdefault(int(ref.rule_index), int(ref.node_index))
    return out


def propose_guard_repairs(
    archive: SearchArchive,
    source_program: EditProgram,
    *,
    target_case_ids: Sequence[str],
    counterexample_ids: Sequence[str],
    case_contexts: Mapping[str, Sequence[Mapping[str, Any]]],
    n: int,
    seed: int,
) -> list[EditProgram]:
    if source_program.program_id not in archive.programs or n <= 0:
        return []
    parent = archive.genome(source_program.child_semantic_id)
    nodes = _rule_nodes(parent)
    if not nodes:
        return []
    guards = synthesize_counterexample_guards(
        target_case_ids, counterexample_ids, case_contexts,
        max_conditions=2, limit=max(16, int(n) * 4),
    )
    if not guards:
        return []
    candidates: list[tuple[tuple[Any, ...], EditProgram]] = []
    for rule_index, rule in enumerate(parent.rules):
        room = 4 - len(rule.conditions)
        if room <= 0 or rule_index not in nodes:
            continue
        existing = set(rule.conditions)
        for gi, guard_set in enumerate(guards):
            additions = tuple(c for c in guard_set if c not in existing)
            if not additions or len(additions) > min(2, room):
                continue
            c1 = additions[0]; c2 = additions[1] if len(additions) > 1 else c1
            action = EditAction(
                "ADD_CONDITION", nodes[rule_index],
                feature=c1.feature, cmp=c1.cmp, index=c1.index, value=c1.value,
                feature2=c2.feature, cmp2=c2.cmp, index2=c2.index, value2=c2.value,
                condition_count=len(additions),
            )
            try:
                program = EditProgram.build(
                    parent, (action,), archive.route_ids, emitter="cegis_repair",
                    seed=int(seed) + rule_index * 1009 + gi * 17,
                    generation_depth=source_program.generation_depth + 1,
                    parent_program_id=source_program.program_id,
                    run_id=archive.run_id,
                )
            except ValueError:
                continue
            key = (len(additions), temporal_specificity(additions), rule_index, gi, program.program_id)
            candidates.append((key, program))
    candidates.sort(key=lambda row: row[0])
    seen: set[str] = set(); out: list[EditProgram] = []
    for _, program in candidates:
        if program.child_semantic_id in seen or program.child_semantic_id in archive.states:
            continue
        seen.add(program.child_semantic_id); out.append(program)
        if len(out) >= int(n): break
    return out


def repairability(
    archive: SearchArchive,
    program: EditProgram,
    *,
    target_case_ids: Sequence[str],
    counterexample_ids: Sequence[str],
    case_contexts: Mapping[str, Sequence[Mapping[str, Any]]],
) -> Repairability:
    child = archive.genome(program.child_semantic_id)
    guards = synthesize_counterexample_guards(target_case_ids, counterexample_ids, case_contexts, max_conditions=2, limit=8)
    capacity = sum(max(0, 4 - len(rule.conditions)) for rule in child.rules)
    specificity = min((temporal_specificity(x)[0] for x in guards), default=9)
    return Repairability(len(set(counterexample_ids)), len(guards), capacity, specificity)


def _counterexamples(pair: PairEvaluation, severe_delta: int) -> tuple[str, ...]:
    out=[]
    for row in pair.case_rows:
        dangerous = bool(row["new_loss"]) or bool(row["w2l"]) or (
            bool(row.get("protected", True)) and int(row["baseline_margin"]) > 0 and int(row["margin_delta"]) < int(severe_delta)
        )
        if dangerous: out.append(str(row["case_id"]))
    return tuple(out)


def _targets(pair: PairEvaluation) -> tuple[str, ...]:
    improved = [
        str(row["case_id"]) for row in pair.case_rows
        if int(row["baseline_margin"]) < 0 and int(row["margin_delta"]) > 0 and bool(row["behavior_changing"])
    ]
    if improved:
        return tuple(improved)
    return tuple(str(row["case_id"]) for row in pair.case_rows if int(row["baseline_margin"]) < 0)


def _cross_context_repair_key(
    discovery: SafetyAssessment,
    certification: SafetyAssessment,
    program_id: str,
) -> tuple[Any, ...]:
    """Hard feasibility first, then transfer usefulness, then locality/quality.

    Discovery regressions are categorically worse than any discovery-feasible child.
    While certification is unsafe we optimize distance to certification feasibility.
    Once both contexts are feasible, certification usefulness/elite behavior dominates
    ordinary production tie-breakers so CEGIS does not settle on safe-but-inert guards.
    """
    pid = str(program_id)
    if not discovery.feasible:
        return (2, discovery.repair_key(), certification.repair_key(), pid)
    if not certification.feasible:
        return (1, certification.repair_key(), discovery.production_key(), pid)
    return (
        0,
        0 if certification.safe_useful else 1,
        0 if certification.elite else 1,
        certification.production_key(),
        discovery.production_key(),
        pid,
    )


def _should_advance_cross_context(
    current_certification: SafetyAssessment,
    candidates: Sequence[tuple[SafetyAssessment, SafetyAssessment]],
) -> bool:
    """Advance only through discovery-feasible children that improve certification."""
    eligible = [(d, c) for d, c in candidates if d.feasible]
    if not eligible:
        return False
    if not current_certification.feasible:
        return any(c.feasible or c.repair_key() < current_certification.repair_key() for _, c in eligible)
    # Cross-context repair is not used to mutate an already certification-feasible
    # terminal. Utility improvement at that point belongs to ordinary search.
    return False


def run_cegis_repair(
    archive: SearchArchive,
    source_program: EditProgram,
    *,
    evaluator,
    panel_manager: CasePanelManager,
    case_contexts: Mapping[str, Sequence[Mapping[str, Any]]],
    generation: int,
    seed: int,
    work_dir: str | Path,
    max_rounds: int = 3,
    candidates_per_round: int = 16,
    certification_evaluator=None,
    certification_bank=None,
    certification_case_contexts: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
) -> RepairReport:
    """Bounded exact-evaluation repair cascade for one promising unsafe program."""
    bank = panel_manager.bank
    archive.bind_case_bank(bank.identity(), tuple(bank.by_id))
    spec = SafetySpec(severe_delta=panel_manager.severe_delta, fragile_margin=panel_manager.fragile_margin)
    current = source_program
    attempts = less = feasible = useful = elite = 0
    chain: list[str] = []
    terminal: str | None = None
    rounds_done = 0
    work = Path(work_dir); work.mkdir(parents=True, exist_ok=True)

    for round_i in range(max(0, int(max_rounds))):
        rounds_done += 1
        current_state = archive.genome(current.child_semantic_id)
        panel_id = f"{archive.run_id}/g{int(generation)}/cegis{round_i}"
        ids = tuple(x.case_id for x in bank.rows)
        archive.register_panel(panel_id, ids)
        base_eval = evaluator.evaluate(
            [archive.genome(archive.root_semantic_id), current_state], bank.rows,
            work_dir=work / f"r{round_i:02d}_base", panel_id=panel_id,
        )
        root_ev = base_eval[archive.root_semantic_id]
        current_ev = base_eval[current.child_semantic_id]
        current_pair = PairEvaluation.from_states(root_ev, current_ev, protected={x.case_id:x.protected for x in bank})
        current_assessment = spec.assess(current_pair)
        ce_ids = _counterexamples(current_pair, panel_manager.severe_delta)
        target_ids = _targets(current_pair)
        if current_assessment.feasible or not ce_ids or not target_ids:
            terminal = current.program_id
            break
        repairs = propose_guard_repairs(
            archive, current, target_case_ids=target_ids, counterexample_ids=ce_ids,
            case_contexts=case_contexts, n=candidates_per_round,
            seed=int(seed) + round_i * 100003,
        )
        if not repairs:
            terminal = current.program_id
            break
        for program in repairs:
            archive.add_program(program, program.reconstruct(current_state, archive.route_ids))
        evals = evaluator.evaluate(
            [archive.genome(p.child_semantic_id) for p in repairs], bank.rows,
            work_dir=work / f"r{round_i:02d}_repairs", panel_id=panel_id,
        )
        attempts += len(repairs)
        ranked=[]
        for program in repairs:
            child_ev = evals[program.child_semantic_id]
            root_pair = PairEvaluation.from_states(root_ev, child_ev, protected={x.case_id:x.protected for x in bank})
            source_pair = PairEvaluation.from_states(current_ev, child_ev, protected={x.case_id:x.protected for x in bank})
            assessment = spec.assess(root_pair)
            archive.consider(program, root_pair, source_pair=source_pair)
            if assessment.repair_key() < current_assessment.repair_key(): less += 1
            if assessment.feasible:
                feasible += 1
                useful += int(assessment.safe_useful)
                elite += int(assessment.elite)
            ranked.append((assessment.repair_key(), program, assessment))
        ranked.sort(key=lambda row: (row[0], row[1].program_id))
        best_key, best_program, best_assessment = ranked[0]
        chain.append(best_program.program_id)
        current = best_program
        terminal = best_program.program_id
        if best_assessment.feasible:
            break

    # A discovery-feasible repair can still expose unseen protected failures on the
    # rotating certification bank. Feed those exact counterexamples back into CEGIS,
    # while preserving the original discovery target behavior. The frozen ranker
    # validation bank is never accepted by this API.
    if terminal is not None and certification_evaluator is not None and certification_bank is not None:
        cert_contexts = dict(certification_case_contexts or {})
        if not cert_contexts:
            from .goals import decode_case_contexts
            cert_contexts = decode_case_contexts(certification_bank)
        combined_contexts = {**dict(case_contexts), **cert_contexts}
        if set(case_contexts) & set(cert_contexts):
            raise ValueError("discovery and certification case identities must be disjoint for CEGIS")
        # Rotating certification: one deterministic mini-panel may produce repair
        # counterexamples, while a disjoint mini-panel remains unseen until the final
        # certificate. Small synthetic banks that cannot be split safely fall back to
        # one panel, preserving unit/smoke behavior without pretending independence.
        from .context_bank import certification_panels
        certification_panel_parts = 1
        final_cert_panel_indices: tuple[int, ...] | None = None
        repair_cert_bank = certification_bank
        try:
            rotated = certification_panels(certification_bank, parts=2)
        except ValueError:
            rotated = ()
        if len(rotated) == 2:
            certification_panel_parts = 2
            repair_cert_bank = CaseBank(rotated[0])
            final_cert_panel_indices = (1,)
        for cert_round in range(max(0, int(max_rounds))):
            terminal_program = archive.programs.get(terminal)
            if terminal_program is None:
                break
            current_state = archive.genome(terminal_program.child_semantic_id)
            discovery_panel_id = f"{archive.run_id}/g{int(generation)}/cegis-cert{cert_round}/discovery"
            discovery_ids = tuple(x.case_id for x in bank.rows)
            archive.register_panel(discovery_panel_id, discovery_ids)
            discovery_exact = evaluator.evaluate(
                [archive.genome(archive.root_semantic_id), current_state], bank.rows,
                work_dir=work / f"cert_r{cert_round:02d}_discovery_base", panel_id=discovery_panel_id,
            )
            discovery_root = discovery_exact[archive.root_semantic_id]
            current_discovery = discovery_exact[current_state.semantic_id()]
            discovery_pair = PairEvaluation.from_states(
                discovery_root, current_discovery, protected={x.case_id:x.protected for x in bank},
            )
            discovery_assessment = spec.assess(discovery_pair)
            if not discovery_assessment.feasible:
                break
            discovery_target_ids = _targets(discovery_pair)
            cert_panel_id = f"{archive.run_id}/g{int(generation)}/cegis-cert{cert_round}/certification-repair"
            cert_exact = certification_evaluator.evaluate(
                [archive.genome(archive.root_semantic_id), current_state], repair_cert_bank.rows,
                work_dir=work / f"cert_r{cert_round:02d}_cert_base", panel_id=cert_panel_id,
            )
            cert_root = cert_exact[archive.root_semantic_id]
            current_cert = cert_exact[current_state.semantic_id()]
            cert_pair = PairEvaluation.from_states(
                cert_root, current_cert, protected={x.case_id:x.protected for x in repair_cert_bank},
            )
            cert_assessment = spec.assess(cert_pair)
            if cert_assessment.feasible:
                break
            # Preserve useful behavior on certification-side root losses as well as
            # discovery targets. Otherwise a guard can satisfy safety by simply turning
            # the behavior off on the entire certification regime.
            cert_target_ids = _targets(cert_pair)
            target_ids = tuple(dict.fromkeys((*discovery_target_ids, *cert_target_ids)))
            cert_counterexamples = _counterexamples(cert_pair, panel_manager.severe_delta)
            if not target_ids or not cert_counterexamples or not combined_contexts:
                break
            repairs = propose_guard_repairs(
                archive, terminal_program, target_case_ids=target_ids,
                counterexample_ids=cert_counterexamples, case_contexts=combined_contexts,
                n=candidates_per_round, seed=int(seed) + 700001 + cert_round * 100003,
            )
            if not repairs:
                break
            for program in repairs:
                archive.add_program(program, program.reconstruct(current_state, archive.route_ids))
            repair_states = [archive.genome(p.child_semantic_id) for p in repairs]
            discovery_repairs = evaluator.evaluate(
                repair_states, bank.rows,
                work_dir=work / f"cert_r{cert_round:02d}_discovery_repairs", panel_id=discovery_panel_id,
            )
            cert_repairs = certification_evaluator.evaluate(
                repair_states, repair_cert_bank.rows,
                work_dir=work / f"cert_r{cert_round:02d}_cert_repairs", panel_id=cert_panel_id,
            )
            attempts += len(repairs)
            ranked_cross=[]
            for program in repairs:
                dchild = discovery_repairs[program.child_semantic_id]
                cchild = cert_repairs[program.child_semantic_id]
                dpair = PairEvaluation.from_states(discovery_root, dchild, protected={x.case_id:x.protected for x in bank})
                spair = PairEvaluation.from_states(current_discovery, dchild, protected={x.case_id:x.protected for x in bank})
                cpair = PairEvaluation.from_states(cert_root, cchild, protected={x.case_id:x.protected for x in repair_cert_bank})
                da = spec.assess(dpair); ca = spec.assess(cpair)
                archive.consider(program, dpair, source_pair=spair)
                if da.feasible and ca.repair_key() < cert_assessment.repair_key():
                    less += 1
                if da.feasible and ca.feasible:
                    feasible += 1
                    useful += int(da.safe_useful)
                    elite += int(da.elite)
                ranked_cross.append((_cross_context_repair_key(da, ca, program.program_id), program, da, ca))
            eligible_cross = [row for row in ranked_cross if row[2].feasible]
            if not eligible_cross or not _should_advance_cross_context(
                cert_assessment, [(row[2], row[3]) for row in eligible_cross]
            ):
                # Never replace a discovery-feasible terminal with a child that loses
                # discovery safety or fails to improve the certification violation.
                break
            eligible_cross.sort(key=lambda row: row[0])
            _, best_program, best_discovery, best_cert = eligible_cross[0]
            chain.append(best_program.program_id)
            terminal = best_program.program_id
            rounds_done += 1
            if best_cert.feasible:
                break

    certified_useful = 0
    if terminal is not None and certification_evaluator is not None and certification_bank is not None:
        terminal_program = archive.programs.get(terminal)
        if terminal_program is not None:
            panel_id = f"{archive.run_id}/g{int(generation)}/cegis-final"
            ids = tuple(x.case_id for x in bank.rows)
            archive.register_panel(panel_id, ids)
            exact = evaluator.evaluate(
                [archive.genome(archive.root_semantic_id), archive.genome(terminal_program.child_semantic_id)], bank.rows,
                work_dir=work / "terminal_discovery", panel_id=panel_id,
            )
            discovery_pair = PairEvaluation.from_states(
                exact[archive.root_semantic_id], exact[terminal_program.child_semantic_id],
                protected={x.case_id:x.protected for x in bank},
            )
            if spec.assess(discovery_pair).feasible:
                from .certification import certify_programs_exact
                batch = certify_programs_exact(
                    archive, (terminal_program,), {terminal_program.program_id: discovery_pair},
                    evaluator=certification_evaluator, case_bank=certification_bank,
                    generation=generation, work_dir=work / "terminal_cert", spec=spec,
                    panel_parts=certification_panel_parts if 'certification_panel_parts' in locals() else 1,
                    panel_indices=final_cert_panel_indices if 'final_cert_panel_indices' in locals() else None,
                )
                vector = batch.vectors[terminal_program.program_id]
                certified_useful = int(vector.certified_positive)
    return RepairReport(
        attempts=attempts, less_infeasible=less, became_feasible=feasible,
        became_safe_useful=useful, became_elite=elite, became_certified_useful=certified_useful,
        rounds=rounds_done, chain_program_ids=tuple(chain), terminal_program_id=terminal,
    )


__all__ = [
    "Repairability", "RepairReport", "synthesize_counterexample_guards",
    "propose_guard_repairs", "repairability", "run_cegis_repair",
]
