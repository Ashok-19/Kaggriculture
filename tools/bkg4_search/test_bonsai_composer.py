from __future__ import annotations

import copy
import importlib.util
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from tools.bkg4_search.bonsai_composer import (
    GateOverlapObservation,
    PreemptionDeclaration,
    SpecialistDeclaration,
    StrategicAuthority,
    analyze_gate_overlap,
    compose_declaration,
    compose_specialists,
)
from tools.bkg4_search.bounded_macro import BoundedRouteMacro
from tools.bkg4_search.module_codec import ConditionGene


ROOT = Path(__file__).resolve().parents[2]
DECLARATION = ROOT / "bonsai-kg/integrations/bkg4_champion/compositions/p0_5_current_specialists_v1.json"
MANUAL = ROOT / "bonsai-kg/integrations/bkg4_champion/modules/animal_feed_deferral_plus_day2_plus_hire4_plus_fert_wool_plus_w33_s172_plus_route435_plus_route410_plus_route437_wheat_scarcity_sheep_shift_v1.json"
AUTO_CONTROLLER = ROOT / "bonsai-kg/integrations/bkg4_champion/champions/bounded_automatic_composer_v1/controller.bkgc"
MANUAL_CONTROLLER = ROOT / "bonsai-kg/integrations/bkg4_champion/champions/route437_wheat_scarcity_sheep_shift_v1/controller.bkgc"
AUTHORITY = StrategicAuthority("strategic_route_override", "route_override", "I32")


def fixture_parent() -> dict:
    return {
        "schema_version": 4,
        "name": "fixture_parent",
        "blackboard": [
            {"name": "route_override", "type": "I32", "initial": 0, "owner": "parent"},
            {"name": "embedded_active", "type": "BOOL", "initial": 0, "owner": "embedded"},
        ],
        "tree": {"type": "Sequence", "children": [{"type": "Success"}]},
        "strategic_routes": [
            {"route_id": 300, "name": "embedded", "source": "fixture_route.json"},
        ],
    }


def fixture_macro(name: str, route_id: int, start: int, duration: int) -> BoundedRouteMacro:
    return BoundedRouteMacro(
        name=name,
        route_id=route_id,
        base_route_id=22,
        start_step=start,
        duration=duration,
        gate_clauses=((ConditionGene("OWN_MONEY", "GE", 0, route_id),),),
        provenance={"source_kind": "fixture"},
    )


def declared_macro(name: str, route_id: int, start: int, duration: int) -> SpecialistDeclaration:
    macro = fixture_macro(name, route_id, start, duration)
    return SpecialistDeclaration.from_macro(
        macro,
        route_source=f"{name}.json",
        authorities=(AUTHORITY.name,),
        macro_source=f"{name}.macro.json",
        provenance={"acceptance": "fixture"},
    )


def embedded(start: int = 100, end: int = 123) -> SpecialistDeclaration:
    return SpecialistDeclaration(
        name="embedded",
        kind="embedded_authority",
        route_id=300,
        authorization_step=start,
        active_start=start,
        active_end=end,
        active_slot="embedded_active",
        authorities=(AUTHORITY.name,),
        route_source="fixture_route.json",
        semantic_id="e" * 64,
        gate_clauses=((ConditionGene("OWN_MONEY", "LE", 0, 500),),),
        provenance={"acceptance": "fixture"},
    )


def preemption(source: str, target: str, start: int, end: int, *, resume: str = "BOTH_WINDOWS_END_TOGETHER") -> PreemptionDeclaration:
    return PreemptionDeclaration(
        preemptor=source,
        preempted=target,
        authority=AUTHORITY.name,
        overlap_start=start,
        overlap_end=end,
        resume_policy=resume,
        evidence={"kind": "exact_fixture_conflict"},
    )


class BoundedBonsaiComposerTests(unittest.TestCase):
    @classmethod
    def current_result(cls):
        return compose_declaration(json.loads(DECLARATION.read_text(encoding="utf-8")), repo_root=ROOT)

    def compose(self, specialists=(), rules=(), **kwargs):
        return compose_specialists(
            fixture_parent(), specialists, rules,
            strategic_authorities=(AUTHORITY,),
            output_name="fixture_output",
            **kwargs,
        )

    def test_empty_specialist_set_returns_parent_equivalent_structure(self):
        parent = fixture_parent()
        result = self.compose()
        for key in ("blackboard", "tree", "strategic_routes"):
            self.assertEqual(result.module[key], parent[key])
        self.assertEqual(result.report["composition_order"], [])

    def test_one_disjoint_specialist_composes_successfully(self):
        result = self.compose((declared_macro("single", 301, 20, 10),))
        self.assertEqual(result.report["composition_order"], ["single"])
        self.assertEqual(result.module["strategic_routes"][-1]["route_id"], 301)

    def test_route217_and_route216_compose_successfully(self):
        route217 = declared_macro("route217", 217, 120, 18)
        route216 = declared_macro("route216", 216, 156, 48)
        result = self.compose((route217, route216))
        self.assertEqual(result.report["composition_order"], ["route216", "route217"])
        pair = next(row for row in result.report["conflicts"] if set(row["specialists"]) == {"route216", "route217"})
        self.assertFalse(pair["window_overlap"])
        self.assertEqual(pair["gate_overlap"], "PROVEN_DISJOINT")

    def test_current_four_authorities_compose_with_explicit_preemption(self):
        result = self.current_result()
        self.assertEqual(result.report["composition_order"], [
            "route435_pizza_glut_rebalance_v1",
            "route410_wheat_liquidity_expansion_v1",
            "route437_wheat_scarcity_sheep_shift_v1",
        ])
        conflict = next(row for row in result.report["conflicts"] if set(row["specialists"]) == {
            "fert_wool_expansion_v1", "route437_wheat_scarcity_sheep_shift_v1"
        })
        self.assertEqual(conflict["gate_overlap"], "POSSIBLE_OVERLAP")
        self.assertIn("DECLARED_PREEMPTION", conflict["conflict_classes"])

    def test_undeclared_authority_overlap_is_rejected(self):
        incumbent = embedded(100, 123)
        challenger = declared_macro("challenger", 301, 100, 24)
        with self.assertRaisesRegex(ValueError, "unresolved strategic authority conflict"):
            self.compose((incumbent, challenger))

    def test_declared_generated_over_embedded_preemption_succeeds(self):
        incumbent = embedded(100, 123)
        challenger = declared_macro("challenger", 301, 100, 24)
        result = self.compose((incumbent, challenger), (preemption("challenger", "embedded", 100, 123),))
        authorization = result.module["tree"]["children"][-1]["children"][0]["then"]["child"]["children"]
        self.assertEqual(authorization[-2], {"type": "SetBlackboard", "slot": "embedded_active", "value": 0})

    def test_reversed_embedded_precedence_is_unsupported(self):
        incumbent = embedded(100, 123)
        challenger = declared_macro("challenger", 301, 100, 24)
        with self.assertRaisesRegex(ValueError, "preemptor must be a generated"):
            self.compose((incumbent, challenger), (preemption("embedded", "challenger", 100, 123),))

    def test_self_preemption_is_rejected(self):
        challenger = declared_macro("challenger", 301, 100, 24)
        with self.assertRaisesRegex(ValueError, "self-preemption"):
            self.compose((challenger,), (preemption("challenger", "challenger", 100, 123),))

    def test_precedence_cycle_is_rejected(self):
        a = declared_macro("a", 301, 100, 24)
        b = declared_macro("b", 302, 100, 24)
        c = declared_macro("c", 303, 100, 24)
        rules = (
            preemption("a", "b", 100, 123),
            preemption("b", "c", 100, 123),
            preemption("c", "a", 100, 123),
        )
        with self.assertRaisesRegex(ValueError, "precedence cycle"):
            self.compose((a, b, c), rules)

    def test_missing_preemption_target_is_rejected(self):
        challenger = declared_macro("challenger", 301, 100, 24)
        with self.assertRaisesRegex(ValueError, "does not exist"):
            self.compose((challenger,), (preemption("challenger", "missing", 100, 123),))

    def test_wrong_embedded_bb_owner_is_rejected(self):
        parent = fixture_parent()
        parent["blackboard"][1]["owner"] = "wrong"
        with self.assertRaisesRegex(ValueError, "wrong BB owner"):
            compose_specialists(parent, (embedded(),), (), strategic_authorities=(AUTHORITY,), output_name="bad")

    def test_duplicate_bb_name_is_rejected(self):
        parent = fixture_parent()
        parent["blackboard"].append(copy.deepcopy(parent["blackboard"][1]))
        with self.assertRaisesRegex(ValueError, "duplicate BB owner or name"):
            compose_specialists(parent, (), (), strategic_authorities=(AUTHORITY,), output_name="bad")

    def test_duplicate_route_id_and_semantic_id_are_rejected(self):
        first = declared_macro("first", 301, 20, 10)
        duplicate_route = declared_macro("second", 301, 40, 10)
        with self.assertRaisesRegex(ValueError, "duplicate specialist route id"):
            self.compose((first, duplicate_route))
        second = dataclass_replace(declared_macro("second", 302, 40, 10), semantic_id=first.semantic_id)
        with self.assertRaisesRegex(ValueError, "duplicate specialist semantic id"):
            self.compose((first, second))

    def test_declared_window_mismatch_is_rejected(self):
        incumbent = embedded(100, 123)
        challenger = declared_macro("challenger", 301, 100, 24)
        with self.assertRaisesRegex(ValueError, "window does not match"):
            self.compose((incumbent, challenger), (preemption("challenger", "embedded", 100, 122),))

    def test_specialist_window_must_match_referenced_macro(self):
        challenger = declared_macro("challenger", 301, 100, 24)
        challenger = dataclass_replace(challenger, active_end=122)
        with self.assertRaisesRegex(ValueError, "does not match bounded macro"):
            self.compose((challenger,))

    def test_unsupported_resume_policy_is_rejected(self):
        incumbent = embedded(100, 123)
        challenger = declared_macro("challenger", 301, 100, 24)
        with self.assertRaisesRegex(ValueError, "unsupported resume policy"):
            self.compose((incumbent, challenger), (
                preemption("challenger", "embedded", 100, 123, resume="RESUME_STACK"),
            ))

    def test_input_order_is_deterministic(self):
        current = json.loads(DECLARATION.read_text(encoding="utf-8"))
        reversed_input = copy.deepcopy(current)
        reversed_input["specialists"].reverse()
        first = compose_declaration(current, repo_root=ROOT)
        second = compose_declaration(reversed_input, repo_root=ROOT)
        self.assertEqual(first.semantic_id, second.semantic_id)
        self.assertEqual(first.module, second.module)
        self.assertEqual(first.report, second.report)

    def test_identity_like_provenance_is_rejected(self):
        specialist = dataclass_replace(
            declared_macro("challenger", 301, 20, 10),
            provenance={"episode": "forbidden"},
        )
        with self.assertRaisesRegex(ValueError, "identity-like"):
            self.compose((specialist,))

    def test_observed_gate_overlap_is_reported_and_contradictions_fail(self):
        incumbent = embedded(100, 123)
        challenger = declared_macro("challenger", 301, 100, 24)
        observation = GateOverlapObservation("embedded", "challenger", {"kind": "bounded_fixture_capture"})
        result = self.compose(
            (incumbent, challenger),
            (preemption("challenger", "embedded", 100, 123),),
            observed_gate_overlaps=(observation,),
        )
        self.assertEqual(result.report["conflicts"][0]["gate_overlap"], "OBSERVED_OVERLAP")

    def test_existing_three_bounded_macro_windows_are_exact(self):
        result = self.current_result()
        windows = {
            row["name"]: row["active_window"]
            for row in result.report["specialists"]
            if row["kind"] == "bounded_route_macro"
        }
        self.assertEqual(windows, {
            "route410_wheat_liquidity_expansion_v1": [120, 137],
            "route435_pizza_glut_rebalance_v1": [156, 203],
            "route437_wheat_scarcity_sheep_shift_v1": [264, 287],
        })

    def test_cli_validates_and_emits_deterministic_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            module_path = directory / "module.json"
            manifest_path = directory / "manifest.json"
            command = [
                "python3", "-m", "tools.bkg4_search.bonsai_composer",
                "--declaration", str(DECLARATION),
                "--out-module", str(module_path),
                "--out-manifest", str(manifest_path),
            ]
            first = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
            self.assertEqual(first.returncode, 0, first.stderr)
            expected = self.current_result()
            self.assertEqual(json.loads(module_path.read_text()), expected.module)
            self.assertEqual(json.loads(manifest_path.read_text()), expected.report)
            before = (module_path.read_bytes(), manifest_path.read_bytes())
            second = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
            self.assertEqual(second.returncode, 0, second.stderr)
            self.assertEqual(before, (module_path.read_bytes(), manifest_path.read_bytes()))

    def test_automatic_module_is_structurally_and_semantically_manual_p04(self):
        automatic = self.current_result().module
        manual = json.loads(MANUAL.read_text(encoding="utf-8"))
        for key in ("blackboard", "tree", "strategic_routes", "runtime_reconciler"):
            self.assertEqual(automatic[key], manual[key])
        build_assets = load_build_assets()
        auto_cp = build_assets.build_controller(automatic)
        manual_cp = build_assets.build_controller(manual)
        self.assertEqual(auto_cp.semantic_hash, manual_cp.semantic_hash)
        self.assertEqual(auto_cp.semantic_hash, "4e611d80f0bbb38bd696aa3dbb8a5e9e903be556e4ceafa233b5cb5685597be6")
        self.assertEqual((len(auto_cp.nodes), len(auto_cp.bb), auto_cp.state_count), (398, 26, 0))

    def test_cpu_cuda_collision_and_manual_auto_decisions_match(self):
        cpp = ROOT / "scratch/bonsai_kg/v4_cpp_differential_runner"
        cuda = ROOT / "scratch/bonsai_kg/v4_cuda_differential_runner"
        if not all(path.exists() for path in (cpp, cuda, AUTO_CONTROLLER, MANUAL_CONTROLLER)):
            self.skipTest("local differential runners or composed controller unavailable")
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            sequence = directory / "composition_boundaries.bks"
            frames = []
            for step in range(119, 290):
                frames.append((
                    (1, 0, 0, step),
                    (12, 0, 0, 9934),
                    (12, 6, 0, 9995),
                    (5, 0, 0, 115 if step == 120 else 15750 if step == 264 else 5000),
                    (18, 1, 0, 5),
                    (7, 0, 0, 8),
                    (9, 0, 0, 2),
                    (12, 3, 0, 9993),
                    (11, 0, 0, 38),
                    (18, 2, 0, 0),
                    (12, 8, 0, 10105 if step == 264 else 10000),
                    (12, 7, 0, 9930 if step == 264 else 10000),
                ))
            with sequence.open("w", encoding="utf-8") as handle:
                handle.write(f"BKS2 2 {len(frames)}\n")
                for frame in frames:
                    handle.write(f"{len(frame)}\n")
                    for fid, arg0, arg1, value in frame:
                        handle.write(f"{fid} {arg0} {arg1} {value}\n")
            manifest = directory / "manifest.txt"
            manifest.write_text(
                f"{MANUAL_CONTROLLER} {sequence}\n{AUTO_CONTROLLER} {sequence}\n",
                encoding="utf-8",
            )
            cpu = subprocess.run([str(cpp), str(manifest)], text=True, capture_output=True)
            gpu = subprocess.run([str(cuda), str(manifest)], text=True, capture_output=True)
            if gpu.returncode != 0:
                self.skipTest(f"CUDA differential runner unavailable: {gpu.stderr.strip()}")
            self.assertEqual(cpu.returncode, 0, cpu.stderr)
            self.assertEqual(cpu.stdout, gpu.stdout)
            rows = cpu.stdout.splitlines()
            self.assertEqual(len(rows), 342)
            normalized = [row.split(" ", 1)[1] for row in rows]
            self.assertEqual(normalized[:171], normalized[171:])
            self.assertIn(",217,", rows[1])
            self.assertNotIn(",217,", rows[19])
            self.assertIn(",216,", rows[37])
            self.assertNotIn(",216,", rows[85])
            self.assertIn(",218,", rows[145])
            self.assertNotIn(",213,", rows[145])
            self.assertIn(",218,", rows[168])
            self.assertNotIn(",218,", rows[169])


def dataclass_replace(value, **changes):
    import dataclasses
    return dataclasses.replace(value, **changes)


def load_build_assets():
    path = ROOT / "bonsai-kg/integrations/bkg4_champion/build_assets.py"
    spec = importlib.util.spec_from_file_location("bkg4_build_assets_p05_test", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load BKG4 build assets")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


if __name__ == "__main__":
    unittest.main(verbosity=2)
