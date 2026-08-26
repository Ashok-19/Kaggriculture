import json
from pathlib import Path
import tempfile
import unittest
from tools.bkg4_search.rlrmb.pipeline import CaseMeta, ExactCaseEvidence, build_routability_evidence, build_cegis_input, emit_candidate_route
from tools.bkg4_search.rlrmb.routability import RoutabilityClassification, RoutabilityResult
from tools.bkg4_search.rlrmb.schema import DatasetRole
from tools.bkg4_search.rlrmb.ir import ActionSource, Phase
from tools.bkg4_search.rlrmb.tests.test_ir import fixed, program
from tools.bkg4_search.rlrmb.motifs import motif_from_actions
from tools.bkg4_search.rlrmb.ir import ActionFamily


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.program = program(Phase(ActionSource.PARENT, (), fixed(1)), cap=1)
        self.context = (("OWN_MONEY", 0, 3000),)
    def row(self, name, role, delta, signature=(1,), candidate=None):
        return ExactCaseEvidence(CaseMeta(name, "tape.json", 0, 1, name), role, -1, -1 + delta,
                                 self.context, signature, candidate or self.program.semantic_id)
    def test_candidate_specific_capture_binding_and_role_conversion(self):
        rows=(self.row("t",DatasetRole.TARGET,2),self.row("x",DatasetRole.TRANSFER,2,(2,)),self.row("v",DatasetRole.VALIDATION,0,(3,)))
        evidence=build_routability_evidence(self.program,rows)
        self.assertEqual({x.role for x in evidence.trajectories},{DatasetRole.TARGET,DatasetRole.TRANSFER,DatasetRole.VALIDATION})
        bad=self.row("bad",DatasetRole.TARGET,2,candidate="other")
        with self.assertRaises(ValueError): build_routability_evidence(self.program,(bad,))
    def test_adapter_has_no_historical_gate_and_sealed_fails_closed(self):
        rows=(self.row("t",DatasetRole.TARGET,2),self.row("x",DatasetRole.TRANSFER,2,(2,)),self.row("v",DatasetRole.VALIDATION,0,(3,)))
        routing=RoutabilityResult(self.program.semantic_id,RoutabilityClassification.ROUTABLE_CANDIDATE,"ok",(),(),None,None,"ENTER_HARDENED_CEGIS",{},("TARGET","TRANSFER","VALIDATION"),100)
        adapted=build_cegis_input(self.program,rows,routing)
        self.assertFalse(hasattr(adapted,"gate"))
        with self.assertRaises(ValueError): build_routability_evidence(self.program,(self.row("s",DatasetRole.SEALED,1),))

    def test_arbitrary_candidate_route_is_derived_from_semantic_content(self):
        motif = motif_from_actions(({"farmer": ["SOUTH"], "hands": [], "market": []},), tuple(ActionFamily))
        candidate = program(Phase(ActionSource.DONOR_BLOCK, tuple(ActionFamily), fixed(1), motif_id=motif.semantic_id), step=1, cap=1)
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory) / "parent.json"
            parent.write_text(json.dumps({"actions": [{"farmer": ["PASS"], "hands": [], "market": []}] * 3}))
            output = emit_candidate_route(candidate, {motif.semantic_id: motif}, parent, Path(directory) / "candidate.json")
            actions = json.loads(output.read_text())["actions"]
            self.assertEqual(actions[0]["farmer"], ["PASS"])
            self.assertEqual(actions[1]["farmer"], ["SOUTH"])


if __name__ == "__main__": unittest.main()
