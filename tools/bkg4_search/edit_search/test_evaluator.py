from __future__ import annotations

import json
import os
import stat
import tempfile
import unittest
from pathlib import Path

from tools.bkg4_search.module_codec import ConditionGene, ModuleGenome, RuleGene
from tools.bkg4_search.edit_search.panels import CaseRecord
from tools.bkg4_search.edit_search.schema import ParentProfile


class EditSearchExactEvaluatorTests(unittest.TestCase):
    def profile(self, root: Path) -> ParentProfile:
        for name in ("controller.bkgc", "routes.bkv6", "legacy_routes.bkv6"):
            (root / name).write_bytes(b"asset")
        return ParentProfile(
            root=root, combined_agent_identity="base", controller_semantic_hash="ctrl",
            controller_path=root/"controller.bkgc", routes_path=root/"routes.bkv6", legacy_routes_path=root/"legacy_routes.bkv6",
            route_ids=(22,70), bkg4_abi=4, feature_registry=1, feed_slot=5, feed_count=13, strategic_route_slot=19,
        )

    def fake_binary(self, root: Path) -> Path:
        p = root / "fake_eval.py"
        p.write_text(
            "#!/usr/bin/env python3\n"
            "import argparse, pathlib, struct\n"
            "a=argparse.ArgumentParser();\n"
            "a.add_argument('--controller');a.add_argument('--routes');a.add_argument('--legacy-routes');a.add_argument('--cases');a.add_argument('--genomes');a.add_argument('--out');a.add_argument('--feed-slot');a.add_argument('--feed-count');a.add_argument('--strategic-route-slot');a.add_argument('--tpb');a.add_argument('--memory-fraction');a.add_argument('--device');x=a.parse_args()\n"
            "raw=pathlib.Path(x.genomes).read_bytes(); magic,ver,ng=struct.unpack_from('<8sII',raw,0); assert magic==b'B4GM1\\x00\\x00\\x00'\n"
            "nc=sum(1 for z in pathlib.Path(x.cases).read_text().splitlines() if z.strip())\n"
            "out=bytearray(struct.pack('<8sIII',b'B4EC1\\x00\\x00\\x00',1,ng,nc))\n"
            "for gi in range(ng):\n"
            "  for ci in range(nc):\n"
            "    own=1000+100*gi+ci; opp=500; out+=struct.pack('<iiHHHHQ',own,opp,gi+ci,gi+1,0,0,100000+gi*100+ci)\n"
            "pathlib.Path(x.out).write_bytes(out)\n"
            "pathlib.Path(x.out+'.command.json').write_text(__import__('json').dumps(vars(x),sort_keys=True))\n"
        )
        p.chmod(p.stat().st_mode | stat.S_IXUSR)
        return p

    def test_evaluate_batches_all_states_into_one_exact_b4ec1_invocation(self):
        from tools.bkg4_search.edit_search.evaluator import ExactCudaEvaluator
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); binary = self.fake_binary(root); profile = self.profile(root)
            genomes = [ModuleGenome(0,()), ModuleGenome(37,(RuleGene((ConditionGene("OWN_MONEY","GE",0,1000),),22),))]
            panel = [
                CaseRecord("c0", "/tmp/t0.b4tp", 0, 7, "f0"),
                CaseRecord("c1", "/tmp/t1.b4tp", 1, 8, "f1"),
            ]
            evaluator = ExactCudaEvaluator(profile, binary=binary, device=0)
            out = evaluator.evaluate(genomes, panel, work_dir=root/"work", panel_id="screen")
            self.assertEqual(set(out), {g.semantic_id() for g in genomes})
            self.assertEqual(out[genomes[0].semantic_id()].margins, (500,501))
            self.assertEqual(out[genomes[1].semantic_id()].margins, (600,601))
            commands = list((root/"work").glob("*.command.json"))
            self.assertEqual(len(commands), 1)
            cmd = json.loads(commands[0].read_text())
            self.assertEqual(cmd["device"], "0")
            self.assertEqual(cmd["strategic_route_slot"], "19")

    def test_missing_exact_cuda_binary_fails_closed_without_cpu_fallback(self):
        from tools.bkg4_search.edit_search.evaluator import ExactCudaEvaluator
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); profile = self.profile(root)
            evaluator = ExactCudaEvaluator(profile, binary=root/"missing")
            with self.assertRaises(FileNotFoundError):
                evaluator.evaluate([ModuleGenome(0,())], [CaseRecord("c","/tmp/t",0,1,"f")], work_dir=root/"w", panel_id="p")


if __name__ == "__main__":
    unittest.main(verbosity=2)
