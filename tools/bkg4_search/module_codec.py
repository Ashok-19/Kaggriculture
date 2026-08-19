from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

MAX_RULES = 8
MAX_CONDS = 4
ROUTE_SLOT = "slot_1"  # frozen stateful-v6 controller route BB after decompilation.

# Deliberately restricted to feature shapes executable in BKG4 feature-registry v1.
# Aggregate crop/animal/weeds GP features are excluded from blind search. A future
# module may spell such a causal aggregate out as ordinary BKG4 per-tile conditions.
FEATURES = (
    "STEP", "DAY", "HOUR",
    "SHOP_AT", "SHOP_SEEN",
    "OWN_MONEY", "OPP_MONEY", "OWN_UNITS", "OPP_UNITS", "OWN_LAND", "OPP_LAND",
    "OWN_HIRES_TODAY", "OPP_HIRES_TODAY",
    "MARKET_PRICE", "MARKET_INVENTORY", "OWN_SHED", "OWN_SEED",
    "OWN_WORKER_X", "OWN_WORKER_Y", "OPP_WORKER_X", "OPP_WORKER_Y",
    "OWN_TILE_KIND", "OPP_TILE_KIND", "OWN_TILE_ITEM", "OPP_TILE_ITEM",
    "OWN_TILE_YIELD", "OPP_TILE_YIELD",
)
COMPARATORS = ("EQ", "NE", "LE", "GE")

ITEM_INDEX_FEATURES = {"MARKET_PRICE", "MARKET_INVENTORY", "OWN_SHED", "OWN_SEED"}
SHOP_INDEX_FEATURES = {"SHOP_AT", "SHOP_SEEN"}
UNIT_INDEX_FEATURES = {"OWN_WORKER_X", "OWN_WORKER_Y", "OPP_WORKER_X", "OPP_WORKER_Y"}
TILE_INDEX_FEATURES = {
    "OWN_TILE_KIND", "OPP_TILE_KIND", "OWN_TILE_ITEM", "OPP_TILE_ITEM",
    "OWN_TILE_YIELD", "OPP_TILE_YIELD",
}


@dataclass(frozen=True)
class ConditionGene:
    feature: str
    cmp: str
    index: int = 0
    value: int = 0

    def validate(self) -> None:
        if self.feature not in FEATURES:
            raise ValueError(f"nonportable module feature {self.feature}")
        if self.cmp not in COMPARATORS:
            raise ValueError(f"invalid comparator {self.cmp}")
        if not -(1 << 31) <= int(self.value) < (1 << 31):
            raise ValueError("condition value must fit int32")
        idx = int(self.index)
        if self.feature in ITEM_INDEX_FEATURES and not 0 <= idx < (5 if self.feature == "OWN_SEED" else 12):
            raise ValueError(f"item/crop index out of range for {self.feature}")
        if self.feature == "SHOP_AT" and not 0 <= idx < 8:
            raise ValueError("SHOP_AT index out of range")
        if self.feature == "SHOP_SEEN" and not 0 <= idx < 8:
            raise ValueError("SHOP_SEEN shop id out of range")
        if self.feature in UNIT_INDEX_FEATURES and not 0 <= idx < 40:
            raise ValueError("worker index out of range")
        if self.feature in TILE_INDEX_FEATURES and not 0 <= idx < 100:
            raise ValueError("tile index out of range")
        if self.feature not in ITEM_INDEX_FEATURES | SHOP_INDEX_FEATURES | UNIT_INDEX_FEATURES | TILE_INDEX_FEATURES and idx != 0:
            raise ValueError(f"zero index required for {self.feature}")

    def feature_ref(self) -> Dict[str, Any]:
        self.validate()
        if self.feature in ITEM_INDEX_FEATURES:
            arg_type = "ITEM"
            return {"feature": self.feature, "arg": {"const": self.index, "type": arg_type}}
        if self.feature == "SHOP_AT":
            return {"feature": self.feature, "arg": self.index}
        if self.feature == "SHOP_SEEN":
            return {"feature": self.feature, "arg": self.index}
        if self.feature in UNIT_INDEX_FEATURES:
            return {"feature": self.feature, "arg": {"const": self.index, "type": "UNIT"}}
        if self.feature in TILE_INDEX_FEATURES:
            return {"feature": self.feature, "arg": {"const": self.index, "type": "TILE"}}
        return {"feature": self.feature}

    def node(self) -> Dict[str, Any]:
        return {"type": "Condition", "lhs": self.feature_ref(), "op": self.cmp, "rhs": int(self.value)}


@dataclass(frozen=True)
class RuleGene:
    conditions: Tuple[ConditionGene, ...]
    route: int

    def validate(self, route_ids: Sequence[int]) -> None:
        if not 1 <= len(self.conditions) <= MAX_CONDS:
            raise ValueError(f"rule requires 1..{MAX_CONDS} conditions")
        if len(set(self.conditions)) != len(self.conditions):
            raise ValueError("duplicate condition in rule")
        for c in self.conditions:
            c.validate()
        if self.route not in route_ids:
            raise ValueError(f"route {self.route} not in frozen actuator bank")


@dataclass(frozen=True)
class ModuleGenome:
    prefix_step: int = 216
    rules: Tuple[RuleGene, ...] = field(default_factory=tuple)

    def validate(self, route_ids: Sequence[int]) -> None:
        if not 0 <= self.prefix_step < 720:
            raise ValueError("prefix_step out of range")
        if len(self.rules) > MAX_RULES:
            raise ValueError(f"module exceeds {MAX_RULES} rules")
        for rule in self.rules:
            rule.validate(route_ids)

    def canonical(self) -> Dict[str, Any]:
        return {
            "prefix_step": self.prefix_step,
            "rules": [
                {
                    "conditions": [dataclasses.asdict(c) for c in r.conditions],
                    "route": r.route,
                }
                for r in self.rules
            ],
        }

    def semantic_id(self) -> str:
        raw = json.dumps(self.canonical(), sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(b"BKG4-MODULE-GENOME\0" + raw).hexdigest()


def distill_module(genome: ModuleGenome, route_ids: Sequence[int], provenance: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Distill one compact genome into ordinary BKG4 schema-4 nodes.

    The module can only change the frozen controller route BB. Failure of every
    searched rule is a Success leaf, so exact incumbent behavior is preserved.
    """
    genome.validate(route_ids)
    rules: List[Dict[str, Any]] = []
    for i, rule in enumerate(genome.rules):
        children = [c.node() for c in rule.conditions]
        children.append({"type": "SetBlackboard", "slot": ROUTE_SLOT, "value": int(rule.route)})
        rules.append({
            "type": "Sequence",
            "name": f"route_rule_{i}",
            "children": children,
        })
    body: Dict[str, Any] = {"type": "Selector", "children": rules + [{"type": "Success"}]} if rules else {"type": "Success"}
    if genome.prefix_step > 0:
        body = {
            "type": "If",
            "condition": {"type": "Condition", "lhs": {"feature": "STEP"}, "op": "GE", "rhs": genome.prefix_step},
            "then": body,
            "else": {"type": "Success"},
        }
    body["module"] = "searched_route_module"
    body["category"] = "strategic_lane"
    body["provenance"] = {
        "genome_id": genome.semantic_id(),
        "grammar": "portable_route_selector_v1",
        **(provenance or {}),
    }
    return body


def distill_commitment_package(genome: ModuleGenome, route_ids: Sequence[int], provenance: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Distill a one-time public-state lane decision with persistent route commitment.

    Rules are evaluated only at `prefix_step`. If one matches, two module-owned BB slots
    latch the chosen route. On all subsequent ticks the committed route overwrites the
    frozen controller's transient route choice. If no rule matches, BASE_0 remains exact.
    """
    genome.validate(route_ids)
    search_active="bkg4_search_active";search_route="bkg4_search_route"
    choices=[]
    for i,rule in enumerate(genome.rules):
        choices.append({"type":"Sequence","name":f"commit_rule_{i}","children":[
            *[c.node() for c in rule.conditions],
            {"type":"SetBlackboard","slot":search_route,"value":int(rule.route)},
            {"type":"SetBlackboard","slot":search_active,"value":1},
        ]})
    decide={"type":"Selector","children":choices+[{"type":"Success"}]} if choices else {"type":"Success"}
    root={"type":"Sequence","module":"searched_route_commitment","category":"strategic_lane","provenance":{"genome_id":genome.semantic_id(),"grammar":"portable_route_commitment_v1",**(provenance or {})},"children":[
        {"type":"If","condition":{"type":"Condition","lhs":{"feature":"STEP"},"op":"EQ","rhs":genome.prefix_step},"then":decide,"else":{"type":"Success"}},
        {"type":"If","condition":{"type":"Condition","lhs":{"blackboard":search_active},"op":"EQ","rhs":1},"then":{"type":"SetBlackboard","slot":ROUTE_SLOT,"value":{"blackboard":search_route}},"else":{"type":"Success"}},
    ]}
    return {"blackboard":[{"name":search_active,"type":"BOOL","initial":0,"owner":"searched_route_commitment"},{"name":search_route,"type":"U16","initial":0,"owner":"searched_route_commitment"}],"tree":root}


def genome_from_json(obj: Dict[str, Any]) -> ModuleGenome:
    return ModuleGenome(
        prefix_step=int(obj.get("prefix_step", 216)),
        rules=tuple(
            RuleGene(
                tuple(ConditionGene(str(c["feature"]), str(c["cmp"]), int(c.get("index", 0)), int(c["value"])) for c in r["conditions"]),
                int(r["route"]),
            )
            for r in obj.get("rules", [])
        ),
    )


def load_genome(path: str | Path) -> ModuleGenome:
    return genome_from_json(json.loads(Path(path).read_text(encoding="utf-8")))


def save_genome(genome: ModuleGenome, path: str | Path) -> None:
    Path(path).write_text(json.dumps(genome.canonical(), sort_keys=True, indent=2) + "\n", encoding="utf-8")
