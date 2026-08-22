from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from tools.bkg4_search.bonsai_editformer.actor_grammar import EditAction, OP_INDEX, node_refs, target_masks
from tools.bkg4_search.module_codec import ConditionGene, ModuleGenome
from tools.edit_corpus.candidates import generate_typed_candidates

from .schema import EditProgram


STRUCTURAL_OPS = frozenset({
    "ADD_SPECIALIST", "INSERT_RECOVERY", "SPLIT_LANE", "COMPOSE_PREDICATES",
    "REPLACE_SUBTREE", "REMOVE_REDUNDANT_BRANCH", "REPLACE_MACRO",
})
LOCAL_OPS = frozenset({
    "CHANGE_THRESHOLD", "CHANGE_ROUTE", "CHANGE_PRIORITY", "CHANGE_WINDOW",
    "ADD_CONDITION", "COMPOSE_PREDICATES",
})


def _quantile(values: Sequence[int], q: float) -> int:
    xs = sorted(int(x) for x in values)
    if not xs:
        return 0
    pos = q * (len(xs) - 1)
    lo = int(pos); hi = min(len(xs) - 1, lo + 1); w = pos - lo
    return int(round(xs[lo] * (1.0 - w) + xs[hi] * w))


def _expanded_context_rows(rows: Iterable[Mapping[str, Any]]) -> tuple[dict[str, Any], ...]:
    """Convert legal observed-state samples into deterministic threshold neighborhoods."""
    grouped: dict[tuple[str, int], list[int]] = {}
    for raw in rows:
        try:
            c = ConditionGene(str(raw["feature"]), "GE", int(raw.get("index", 0)), int(round(float(raw["value"]))))
            c.validate()
        except (KeyError, TypeError, ValueError, OverflowError):
            continue
        grouped.setdefault((c.feature, c.index), []).append(c.value)
    out: dict[tuple[str, int, int], dict[str, Any]] = {}
    meaningful = {
        "STEP": (0, 24, 72, 144, 192, 216, 288, 360, 432, 576, 719),
        "DAY": tuple(range(0, 11)),
        "HOUR": tuple(range(0, 24)),
        "OWN_MONEY": (0, 250, 500, 750, 1000, 1500, 2000, 3000, 5000, 7500, 10000, 15000),
        "OPP_MONEY": (0, 250, 500, 750, 1000, 1500, 2000, 3000, 5000, 7500, 10000, 15000),
        "OWN_UNITS": (1, 2, 3, 4, 5, 6, 8, 10, 12),
        "OPP_UNITS": (1, 2, 3, 4, 5, 6, 8, 10, 12),
        "MARKET_PRICE": (30, 50, 70, 81, 100, 120, 140, 160, 192),
        "MARKET_INVENTORY": (0, 5, 10, 20, 40, 80),
    }
    for (feature, index), vals in sorted(grouped.items()):
        candidates = set(vals)
        candidates.update(_quantile(vals, q) for q in (0.0, 0.10, 0.25, 0.5, 0.75, 0.90, 1.0))
        for v in list(candidates):
            step = max(1, abs(int(v)) // 20)
            candidates.update((int(v) - 1, int(v) + 1, int(v) - step, int(v) + step))
        candidates.update(meaningful.get(feature, ()))
        for value in sorted(candidates):
            try:
                ConditionGene(feature, "GE", index, int(value)).validate()
            except ValueError:
                continue
            out[(feature, index, int(value))] = {"feature": feature, "index": index, "value": int(value)}
    return tuple(out[k] for k in sorted(out))


@dataclass(frozen=True)
class EmitterContext:
    route_ids: tuple[int, ...]
    context_rows: tuple[dict[str, Any], ...]
    donor_genomes: tuple[ModuleGenome, ...] = ()
    motifs: tuple[Mapping[str, Any], ...] = ()
    generation_depth: int = 1
    escalation_level: int = 0
    parent_program_id: str | None = None
    run_id: str | None = None
    quality_by_state: Mapping[str, float] = field(default_factory=dict)

    @classmethod
    def from_observations(
        cls,
        route_ids: Sequence[int],
        rows: Iterable[Mapping[str, Any]],
        *,
        donor_genomes: Sequence[ModuleGenome] = (),
        motifs: Sequence[Mapping[str, Any]] = (),
        generation_depth: int = 1,
        escalation_level: int = 0,
        parent_program_id: str | None = None,
        run_id: str | None = None,
        quality_by_state: Mapping[str, float] | None = None,
    ) -> "EmitterContext":
        routes = tuple(int(x) for x in route_ids)
        if not routes:
            raise ValueError("emitter context requires legal routes")
        for donor in donor_genomes:
            donor.validate(routes)
        return cls(
            route_ids=routes,
            context_rows=_expanded_context_rows(rows),
            donor_genomes=tuple(donor_genomes),
            motifs=tuple(dict(x) for x in motifs),
            generation_depth=max(1, int(generation_depth)),
            escalation_level=max(0, int(escalation_level)),
            parent_program_id=parent_program_id,
            run_id=run_id,
            quality_by_state=dict(quality_by_state or {}),
        )

    def rows_for_parent(self, parent: ModuleGenome) -> tuple[dict[str, Any], ...]:
        rows = list(self.context_rows)
        for rule in parent.rules:
            for c in rule.conditions:
                rows.append({"feature": c.feature, "index": c.index, "value": c.value})
        return _expanded_context_rows(rows)


class SearchEmitter:
    name = "base"

    def propose(self, parent: ModuleGenome, *, n: int, seed: int, context: EmitterContext) -> list[EditProgram]:
        raise NotImplementedError

    def _one_step(self, parent: ModuleGenome, *, n: int, seed: int, context: EmitterContext, allowed: set[str] | frozenset[str] | None = None) -> list[EditProgram]:
        if n <= 0:
            return []
        raw = generate_typed_candidates(
            parent,
            context.rows_for_parent(parent),
            n=max(n, min(4096, n * 3)),
            seed=int(seed),
            route_ids=context.route_ids,
        )
        if allowed is not None:
            raw = [x for x in raw if x.action.operation in allowed]
        raw.sort(key=lambda x: (
            x.action.operation, int(x.action.target_node), str(x.action.feature), str(x.action.cmp),
            int(x.action.index), int(x.action.value), int(x.action.route), int(x.action.priority), int(x.action.prefix_step), x.edit_id,
        ))
        out: list[EditProgram] = []
        seen: set[str] = set()
        for i, row in enumerate(raw):
            try:
                program = EditProgram.build(
                    parent, (row.action,), context.route_ids, emitter=self.name, seed=int(seed) + i,
                    generation_depth=context.generation_depth, parent_program_id=context.parent_program_id, run_id=context.run_id,
                )
            except ValueError:
                continue
            if program.child_semantic_id in seen:
                continue
            seen.add(program.child_semantic_id); out.append(program)
            if len(out) >= n:
                break
        return out


class SystematicEmitter(SearchEmitter):
    name = "systematic"

    def propose(self, parent: ModuleGenome, *, n: int, seed: int, context: EmitterContext) -> list[EditProgram]:
        return self._one_step(parent, n=n, seed=seed, context=context)


class LocalRefinementEmitter(SearchEmitter):
    name = "local_refine"

    def propose(self, parent: ModuleGenome, *, n: int, seed: int, context: EmitterContext) -> list[EditProgram]:
        return self._one_step(parent, n=n, seed=seed, context=context, allowed=LOCAL_OPS)


class StructuralEmitter(SearchEmitter):
    name = "structural"

    def propose(self, parent: ModuleGenome, *, n: int, seed: int, context: EmitterContext) -> list[EditProgram]:
        return self._one_step(parent, n=n, seed=seed, context=context, allowed=STRUCTURAL_OPS)


def _rule_action(operation: str, target: int, rule, *, priority: int = 0) -> EditAction:
    c1 = rule.conditions[0]
    c2 = rule.conditions[1] if len(rule.conditions) > 1 else c1
    return EditAction(
        operation=operation, target_node=int(target), feature=c1.feature, cmp=c1.cmp, index=c1.index, value=c1.value,
        route=rule.route, feature2=c2.feature, cmp2=c2.cmp, index2=c2.index, value2=c2.value,
        condition_count=min(2, len(rule.conditions)), priority=int(priority),
    )


class RecombinationEmitter(SearchEmitter):
    name = "recombine"

    def propose(self, parent: ModuleGenome, *, n: int, seed: int, context: EmitterContext) -> list[EditProgram]:
        if n <= 0 or not context.donor_genomes or len(parent.rules) >= 8:
            return []
        rng = random.Random(int(seed))
        donors = list(context.donor_genomes); rng.shuffle(donors)
        out: list[EditProgram] = []; seen: set[str] = set()
        for donor in donors:
            donor_rules = list(donor.rules); rng.shuffle(donor_rules)
            actions: list[EditAction] = []
            current_rules = set(parent.rules)
            room = 8 - len(parent.rules)
            for rule in donor_rules:
                if rule in current_rules or len(rule.conditions) > 2 or room <= 0:
                    continue
                actions.append(_rule_action("ADD_SPECIALIST", 0, rule, priority=min(len(parent.rules) + len(actions), 7)))
                current_rules.add(rule); room -= 1
                if len(actions) >= 2:
                    break
            if not actions:
                continue
            # Pair donor transfer with a local window/route adjustment when only one rule was transferable.
            if len(actions) == 1 and parent.prefix_step != donor.prefix_step:
                actions.append(EditAction("CHANGE_WINDOW", 0, prefix_step=donor.prefix_step))
            try:
                program = EditProgram.build(
                    parent, actions, context.route_ids, emitter=self.name, seed=seed + len(out),
                    generation_depth=context.generation_depth + len(actions) - 1,
                    parent_program_id=context.parent_program_id, run_id=context.run_id,
                )
            except ValueError:
                continue
            if program.child_semantic_id not in seen:
                seen.add(program.child_semantic_id); out.append(program)
            if len(out) >= n:
                break
        return out


class LargeNeighborhoodEmitter(SearchEmitter):
    name = "lns"

    def propose(self, parent: ModuleGenome, *, n: int, seed: int, context: EmitterContext) -> list[EditProgram]:
        if n <= 0:
            return []
        rng = random.Random(int(seed))
        rows = context.rows_for_parent(parent)
        repair_candidates = generate_typed_candidates(parent, rows, n=max(32, n * 4), seed=seed + 701, route_ids=context.route_ids)
        conditions = []
        for r in rows:
            try:
                c = ConditionGene(str(r["feature"]), "GE" if str(r["feature"]) not in {"SHOP_AT", "SHOP_SEEN"} else "EQ", int(r.get("index", 0)), int(r["value"]))
                c.validate(); conditions.append(c)
            except (KeyError, ValueError):
                pass
        out: list[EditProgram] = []; seen: set[str] = set()
        attempts = max(n * 12, 32)
        for i in range(attempts):
            actions: list[EditAction] = []
            if parent.rules:
                ri = i % len(parent.rules)
                target = 3 + sum(1 + len(r.conditions) for r in parent.rules[:ri])
                actions.append(EditAction("REMOVE_REDUNDANT_BRANCH", target))
                # Repair from a donor or observed public-state predicate with a different route.
                donor_rule = None
                if context.donor_genomes:
                    donor = context.donor_genomes[i % len(context.donor_genomes)]
                    if donor.rules:
                        donor_rule = donor.rules[i % len(donor.rules)]
                if donor_rule is not None and len(donor_rule.conditions) <= 2:
                    actions.append(_rule_action("ADD_SPECIALIST", 0, donor_rule, priority=min(ri, max(0, len(parent.rules) - 1))))
                elif conditions:
                    c = conditions[i % len(conditions)]
                    route_choices = [r for r in context.route_ids if r != parent.rules[ri].route]
                    route = route_choices[i % len(route_choices)] if route_choices else parent.rules[ri].route
                    actions.append(EditAction("INSERT_RECOVERY", 0, feature=c.feature, cmp=c.cmp, index=c.index, value=c.value, route=route, priority=min(ri, 7)))
            else:
                # A two-coordinate jump from an empty module: create a rule then alter the global window.
                if not repair_candidates:
                    break
                actions.append(repair_candidates[i % len(repair_candidates)].action)
                actions.append(EditAction("CHANGE_WINDOW", 0, prefix_step=(parent.prefix_step + 37 + i * 19) % 720))
            if context.escalation_level >= 4 and len(actions) < 3 and parent.prefix_step != 0:
                actions.append(EditAction("CHANGE_WINDOW", 0, prefix_step=(parent.prefix_step + 73 + i * 17) % 720))
            try:
                program = EditProgram.build(
                    parent, actions, context.route_ids, emitter=self.name, seed=seed + i,
                    generation_depth=context.generation_depth + len(actions) - 1,
                    parent_program_id=context.parent_program_id, run_id=context.run_id,
                )
            except ValueError:
                continue
            if program.child_semantic_id in seen or program.child_semantic_id == parent.semantic_id():
                continue
            seen.add(program.child_semantic_id); out.append(program)
            if len(out) >= n:
                break
        return out


class MotifEmitter(SearchEmitter):
    """Re-instantiate exact-CUDA-proven symbolic templates against compatible current targets."""

    name = "motif"

    def propose(self, parent: ModuleGenome, *, n: int, seed: int, context: EmitterContext) -> list[EditProgram]:
        if n <= 0 or not context.motifs:
            return []
        ranked = sorted(
            context.motifs,
            key=lambda x: (-int(x.get("safe_useful_hits", 0)), -int(x.get("l2w_total", 0)), str(x.get("motif_id", ""))),
        )
        out: list[EditProgram] = []
        seen: set[str] = set()
        for mi, motif in enumerate(ranked):
            atoms = list(dict(motif.get("template") or {}).get("atoms") or [])
            if not atoms:
                continue
            variants = max(1, n - len(out))
            for variant in range(variants * 3):
                state = parent
                actions: list[EditAction] = []
                valid = True
                for ai, raw_atom in enumerate(atoms):
                    atom = dict(raw_atom)
                    op = str(atom.get("operation", ""))
                    if op not in OP_INDEX:
                        valid = False; break
                    masks = target_masks(state)
                    refs = node_refs(state)
                    target_kind = str(atom.get("target_kind", "unknown"))
                    targets = [
                        ti for ti, ok in enumerate(masks[OP_INDEX[op]])
                        if ok and ti < len(refs) and refs[ti] is not None and str(refs[ti].kind) == target_kind
                    ]
                    if not targets:
                        valid = False; break
                    hint_rule = int(atom.get("target_rule_index_hint", -1))
                    hint_cond = int(atom.get("target_condition_index_hint", -1))
                    targets.sort(key=lambda ti: (
                        int(refs[ti].rule_index != hint_rule), int(refs[ti].condition_index != hint_cond), ti
                    ))
                    target = targets[(variant + ai + mi) % len(targets)]
                    payload = dict(atom.get("action") or {})
                    payload["operation"] = op
                    payload["target_node"] = int(target)
                    # Threshold motifs keep the learned scalar but must bind to the actual target predicate semantics.
                    if op == "CHANGE_THRESHOLD":
                        ref = refs[target]
                        old = state.rules[ref.rule_index].conditions[ref.condition_index]
                        payload.update(feature=old.feature, cmp=old.cmp, index=old.index)
                    try:
                        action = EditAction(**payload)
                        one = EditProgram.build(state, (action,), context.route_ids, emitter=self.name, seed=seed + variant + ai)
                        state = one.reconstruct(state, context.route_ids)
                    except (TypeError, ValueError):
                        valid = False; break
                    actions.append(action)
                if not valid or not actions:
                    continue
                try:
                    program = EditProgram.build(
                        parent, actions, context.route_ids, emitter=self.name, seed=seed + mi * 1009 + variant,
                        generation_depth=context.generation_depth + len(actions) - 1,
                        parent_program_id=context.parent_program_id, run_id=context.run_id,
                    )
                except ValueError:
                    continue
                if program.child_semantic_id in seen or program.child_semantic_id == parent.semantic_id():
                    continue
                seen.add(program.child_semantic_id); out.append(program)
                if len(out) >= n:
                    return out
        return out


class RandomTypedEmitter(SearchEmitter):
    name = "random_typed"

    def propose(self, parent: ModuleGenome, *, n: int, seed: int, context: EmitterContext) -> list[EditProgram]:
        pool = self._one_step(parent, n=max(n, n * 3), seed=seed ^ 0x5EED, context=context)
        rng = random.Random(int(seed)); rng.shuffle(pool)
        return pool[:max(0, int(n))]


@dataclass
class _MCTSNode:
    state: ModuleGenome
    actions: tuple[EditAction, ...]
    visits: int = 1
    reward_sum: float = 0.0
    children: int = 0

    def score(self, total_visits: int, quality: float, exploration_c: float) -> float:
        exploit = (self.reward_sum / max(1, self.visits)) + float(quality)
        novelty = 1.0 / (1.0 + self.children)
        explore = exploration_c * math.sqrt(math.log(max(2, total_visits)) / max(1, self.visits))
        return exploit + 0.15 * novelty + explore


class EvolvabilityMCTSEmitter(SearchEmitter):
    """Bounded grammar MCTS; exact evaluated state quality can be supplied by the archive context."""

    name = "evolvability_mcts"

    def __init__(self, *, max_depth: int = 3, exploration_c: float = 0.45):
        self.max_depth = max(1, int(max_depth)); self.exploration_c = float(exploration_c)

    def propose(self, parent: ModuleGenome, *, n: int, seed: int, context: EmitterContext) -> list[EditProgram]:
        if n <= 0:
            return []
        nodes = [_MCTSNode(parent, ())]
        out: list[EditProgram] = []; seen = {parent.semantic_id()}; total_visits = 1
        for step in range(max(n * 20, 64)):
            expandable = [x for x in nodes if len(x.actions) < self.max_depth]
            if not expandable: break
            # UCB balances known exact state quality (when supplied), lineage reward, visits and local branching novelty.
            node = max(
                expandable,
                key=lambda x: x.score(total_visits, float(context.quality_by_state.get(x.state.semantic_id(), 0.0)), self.exploration_c),
            )
            raw = generate_typed_candidates(
                node.state, context.rows_for_parent(node.state), n=12,
                seed=int(seed) + 1009 * (step + 1) + len(node.actions) * 7919, route_ids=context.route_ids,
            )
            if not raw:
                node.visits += 1; total_visits += 1; continue
            choice = raw[(step + node.children) % len(raw)]
            try:
                next_state = choice.action and EditProgram.build(
                    node.state, (choice.action,), context.route_ids, emitter=self.name, seed=seed + step
                ).reconstruct(node.state, context.route_ids)
            except ValueError:
                node.visits += 1; total_visits += 1; continue
            node.children += 1; node.visits += 1; total_visits += 1
            sid = next_state.semantic_id()
            if sid in seen:
                continue
            seen.add(sid)
            actions = node.actions + (choice.action,)
            quality = float(context.quality_by_state.get(sid, 0.0))
            child_node = _MCTSNode(next_state, actions, reward_sum=quality)
            nodes.append(child_node)
            # Backpropagate the known/direct quality when available; otherwise novel legal expansion gets a tiny positive sample.
            reward = quality if sid in context.quality_by_state else 0.05
            node.reward_sum += reward
            try:
                program = EditProgram.build(
                    parent, actions, context.route_ids, emitter=self.name, seed=seed + step,
                    generation_depth=context.generation_depth + len(actions) - 1,
                    parent_program_id=context.parent_program_id, run_id=context.run_id,
                )
            except ValueError:
                continue
            out.append(program)
            if len(out) >= n:
                # Require actual derivation depth when possible instead of returning only the first frontier.
                if any(x.program_length >= min(2, self.max_depth) for x in out):
                    break
        return out[:n]


DEFAULT_EMITTERS: tuple[SearchEmitter, ...] = (
    SystematicEmitter(), LocalRefinementEmitter(), StructuralEmitter(), MotifEmitter(), RecombinationEmitter(),
    LargeNeighborhoodEmitter(), EvolvabilityMCTSEmitter(), RandomTypedEmitter(),
)
