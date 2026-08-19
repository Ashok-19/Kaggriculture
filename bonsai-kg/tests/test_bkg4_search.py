from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'bonsai-kg/python'))

import bonsai_kg_v4 as bkg4
from tools.bkg4_search.evolvability import EvolvabilityTree, sanity_fixture
from tools.bkg4_search.module_codec import ConditionGene, ModuleGenome, RuleGene, distill_module


class BKG4SearchTests(unittest.TestCase):
    def test_evolvability_separates_brittle_peak(self):
        f=sanity_fixture();t=EvolvabilityTree(f['elite'],f['floor'])
        robust=t.add_accepted(0,f['elite']);t.add_accepted(robust,f['elite_child'])
        brittle=t.add_accepted(0,f['elite']);t.add_accepted(brittle,f['brittle_child'])
        self.assertGreater(t.nodes[robust].q,t.nodes[brittle].q)
        self.assertLessEqual(t.nodes[robust].brittleness,t.nodes[brittle].brittleness)

    def test_rejection_changes_N_not_Q(self):
        f=sanity_fixture();t=EvolvabilityTree(f['elite'],f['floor']);q=t.nodes[0].q;m=t.nodes[0].value_samples;n=t.nodes[0].visits
        t.add_rejected_visit(0)
        self.assertEqual((t.nodes[0].q,t.nodes[0].value_samples),(q,m));self.assertEqual(t.nodes[0].visits,n+1)

    def test_mediocre_flat_region_cannot_win_quality_floor(self):
        f=sanity_fixture();t=EvolvabilityTree(f['elite'],f['floor']);i=t.add_accepted(0,f['weak'])
        self.assertEqual(t.nodes[i].v,0.0)

    def test_portable_module_compiles_as_bkg4(self):
        g=ModuleGenome(216,(RuleGene((ConditionGene('OPP_MONEY','GE',0,7500),ConditionGene('HOUR','LE',0,4)),46),))
        module=distill_module(g,[22,23,29,32,34,46,63,64,70,77,79])
        # Use the frozen controller as authoring source and append the module.
        import bonsai_kg as b3
        p3=b3.load_compiled(ROOT/'bonsai-kg/integrations/stateful_v6/assets/controller.bkgc')
        src=bkg4.bkg3_to_bkg4_source(p3);src['tree']['children'].append(module)
        cp=bkg4.compile_policy(src)
        self.assertGreater(len(cp.nodes),99);self.assertEqual(cp.state_count,0)
        self.assertEqual(cp.analysis[0].max_live_intents,0)

if __name__=='__main__':unittest.main(verbosity=2)
